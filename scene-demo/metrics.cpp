#include <algorithm>
#include <array>
#include <charconv>
#include <cerrno>
#include <csignal>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

// Never emit diagnostics when stderr may refer to an input or image output.
bool diagnostics_safe = true;

struct Image {
    std::size_t width;
    std::size_t height;
    std::vector<double> rgb; // Encoded PPM sample values, normalised to [0, 1].
};

struct Region {
    std::size_t x;
    std::size_t y;
    std::size_t width;
    std::size_t height;
};

struct Options {
    std::array<std::string, 3> inputs;
    bool has_region = false;
    Region region{};
    std::string cheap_diff;
    std::string neural_diff;
    std::string sheet;
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

bool whitespace(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

std::size_t number(std::string_view text, const std::string& what) {
    std::size_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        fail("invalid " + what + ": " + std::string(text));
    }
    return value;
}

std::size_t multiply(std::size_t a, std::size_t b) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        fail("image dimensions overflow");
    }
    return a * b;
}

void skip_comment(std::istream& in) {
    char c = 0;
    while (in.get(c)) {
        if (c == '\r' || c == '\n') {
            return;
        }
    }
    fail("unterminated PPM comment");
}

std::string token(std::istream& in) {
    char c = 0;
    while (in.get(c)) {
        if (whitespace(c)) {
            continue;
        }
        if (c == '#') {
            skip_comment(in);
            continue;
        }
        std::string result(1, c);
        while (in.peek() != std::char_traits<char>::eof()) {
            const auto next = static_cast<char>(in.peek());
            if (whitespace(next) || next == '#') {
                break;
            }
            in.get(c);
            result.push_back(c);
            if (result.size() > 64) {
                fail("PPM header token too long");
            }
        }
        return result;
    }
    fail("truncated PPM header");
}

Image load(const std::string& path, int fd) {
    // Reopen the pinned descriptor, never the caller-supplied path.
    std::ifstream in("/proc/self/fd/" + std::to_string(fd), std::ios::binary);
    if (!in) {
        fail("cannot open input: " + path);
    }
    if (token(in) != "P6") {
        fail("not a P6 PPM: " + path);
    }
    const auto width = number(token(in), "width");
    const auto height = number(token(in), "height");
    const auto maxval = number(token(in), "maxval");
    if (width == 0 || height == 0 || maxval == 0 || maxval > 65535) {
        fail("invalid PPM dimensions or maxval: " + path);
    }
    const auto samples = multiply(multiply(width, height), 3);
    const auto bytes = multiply(samples, maxval < 256 ? 1 : 2);
    // A comment immediately following maxval precedes the mandatory separator.
    // Once that separator is consumed, every subsequent byte is raster data,
    // except that CRLF may be a single separator if the file length requires it.
    char separator = 0;
    if (!in.get(separator)) {
        fail("missing PPM raster separator: " + path);
    }
    if (separator == '#') {
        do {
            if (!in.get(separator)) {
                fail("unterminated PPM comment: " + path);
            }
        } while (separator != '\r' && separator != '\n');
    } else if (!whitespace(separator)) {
        fail("missing PPM raster separator: " + path);
    }
    const auto after_separator = in.tellg();
    in.seekg(0, std::ios::end);
    const auto end = in.tellg();
    if (after_separator < 0 || end < after_separator ||
        bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()) ||
        static_cast<std::uintmax_t>(bytes) > static_cast<std::uintmax_t>(std::numeric_limits<std::streamoff>::max())) {
        fail("PPM raster length mismatch: " + path);
    }
    const auto remaining = static_cast<std::uintmax_t>(end - after_separator);
    in.seekg(after_separator);
    if (separator == '\r' && remaining > 0 && remaining - 1 == bytes && in.peek() == '\n') {
        in.get();
    } else if (remaining != bytes) {
        fail("PPM raster length mismatch: " + path);
    }
    std::vector<unsigned char> raw(bytes);
    if (!in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(bytes))) {
        fail("cannot read PPM raster: " + path);
    }
    Image image{width, height, std::vector<double>(samples)};
    for (std::size_t i = 0; i < samples; ++i) {
        const auto sample = maxval < 256 ? static_cast<std::size_t>(raw[i]) :
            (static_cast<std::size_t>(raw[2 * i]) * 256 + raw[2 * i + 1]);
        if (sample > maxval) {
            fail("PPM sample exceeds maxval: " + path);
        }
        image.rgb[i] = static_cast<double>(sample) / static_cast<double>(maxval);
    }
    return image;
}

double luma(const Image& image, std::size_t x, std::size_t y) {
    const auto i = (y * image.width + x) * 3;
    return 0.2126 * image.rgb[i] + 0.7152 * image.rgb[i + 1] + 0.0722 * image.rgb[i + 2];
}

struct Metrics {
    double psnr;
    double ssim;
    double luma_mae;
    double detail_gradient_mae;
    double lighting_block_rmse;
    std::size_t window;
};

Metrics compare(const Image& reference, const Image& candidate, const Region& r) {
    double squared = 0;
    double lum_error = 0;
    double gradient_error = 0;
    std::size_t gradients = 0;
    for (std::size_t y = r.y; y < r.y + r.height; ++y) {
        for (std::size_t x = r.x; x < r.x + r.width; ++x) {
            const auto i = (y * reference.width + x) * 3;
            for (std::size_t c = 0; c < 3; ++c) {
                const auto delta = reference.rgb[i + c] - candidate.rgb[i + c];
                squared += delta * delta;
            }
            const auto a = luma(reference, x, y);
            const auto b = luma(candidate, x, y);
            lum_error += std::abs(a - b);
            if (x + 1 < r.x + r.width) {
                gradient_error += std::abs((luma(reference, x + 1, y) - a) - (luma(candidate, x + 1, y) - b));
                ++gradients;
            }
            if (y + 1 < r.y + r.height) {
                gradient_error += std::abs((luma(reference, x, y + 1) - a) - (luma(candidate, x, y + 1) - b));
                ++gradients;
            }
        }
    }
    const auto pixels = static_cast<double>(multiply(r.width, r.height));
    const auto mse = squared / (3.0 * pixels);
    const auto psnr = mse == 0 ? std::numeric_limits<double>::infinity() : -10.0 * std::log10(mse);

    // Gaussian SSIM, population moments, valid windows entirely inside the region.
    auto window = std::min({std::size_t{11}, r.width, r.height});
    if (window % 2 == 0) {
        --window;
    }
    const auto radius = window / 2;
    std::vector<double> weights(window * window);
    double weight_sum = 0;
    for (std::size_t y = 0; y < window; ++y) {
        for (std::size_t x = 0; x < window; ++x) {
            const auto dx = static_cast<double>(x) - static_cast<double>(radius);
            const auto dy = static_cast<double>(y) - static_cast<double>(radius);
            const auto weight = std::exp(-(dx * dx + dy * dy) / (2.0 * 1.5 * 1.5));
            weights[y * window + x] = weight;
            weight_sum += weight;
        }
    }
    for (auto& weight : weights) {
        weight /= weight_sum;
    }
    double ssim_sum = 0;
    std::size_t windows = 0;
    constexpr double c1 = 0.01 * 0.01;
    constexpr double c2 = 0.03 * 0.03;
    for (std::size_t y = r.y; y <= r.y + r.height - window; ++y) {
        for (std::size_t x = r.x; x <= r.x + r.width - window; ++x) {
            double ma = 0, mb = 0, aa = 0, bb = 0, ab = 0;
            for (std::size_t wy = 0; wy < window; ++wy) {
                for (std::size_t wx = 0; wx < window; ++wx) {
                    const auto a = luma(reference, x + wx, y + wy);
                    const auto b = luma(candidate, x + wx, y + wy);
                    const auto weight = weights[wy * window + wx];
                    ma += weight * a;
                    mb += weight * b;
                    aa += weight * a * a;
                    bb += weight * b * b;
                    ab += weight * a * b;
                }
            }
            const auto va = std::max(0.0, aa - ma * ma);
            const auto vb = std::max(0.0, bb - mb * mb);
            const auto covariance = ab - ma * mb;
            ssim_sum += ((2 * ma * mb + c1) * (2 * covariance + c2)) /
                        ((ma * ma + mb * mb + c1) * (va + vb + c2));
            ++windows;
        }
    }

    // Area-weighted RMSE of signed mean luma differences in non-overlapping 8x8 tiles.
    double tile_squared = 0;
    for (std::size_t by = 0; by < r.height; by += std::min(std::size_t{8}, r.height - by)) {
        const auto h = std::min(std::size_t{8}, r.height - by);
        for (std::size_t bx = 0; bx < r.width; bx += std::min(std::size_t{8}, r.width - bx)) {
            const auto w = std::min(std::size_t{8}, r.width - bx);
            double delta_sum = 0;
            for (std::size_t y = 0; y < h; ++y) {
                for (std::size_t x = 0; x < w; ++x) {
                    delta_sum += luma(reference, r.x + bx + x, r.y + by + y) -
                                 luma(candidate, r.x + bx + x, r.y + by + y);
                }
            }
            const auto count = static_cast<double>(w * h);
            const auto mean = delta_sum / count;
            tile_squared += count * mean * mean;
        }
    }
    return {psnr, ssim_sum / static_cast<double>(windows), lum_error / pixels,
            gradients == 0 ? 0.0 : gradient_error / static_cast<double>(gradients),
            std::sqrt(tile_squared / pixels), window};
}

unsigned char byte(double value) {
    return static_cast<unsigned char>(std::lround(std::clamp(value, 0.0, 1.0) * 255.0));
}

bool same_file(const struct stat& a, const struct stat& b) {
    return a.st_dev == b.st_dev && a.st_ino == b.st_ino;
}

void protect_diagnostics(const struct stat& identity) {
    struct stat diagnostic{};
    if (fstat(STDERR_FILENO, &diagnostic) != 0 || same_file(identity, diagnostic)) {
        diagnostics_safe = false;
    }
}

// Run before parsing or emitting usage: even malformed invocations can name inputs.
void inspect_diagnostic_aliases(int argc, char** argv) {
    struct stat diagnostic{};
    if (fstat(STDERR_FILENO, &diagnostic) != 0) {
        diagnostics_safe = false;
        return;
    }
    for (int i = 1; i < argc; ++i) {
        struct stat candidate{};
        if (stat(argv[i], &candidate) == 0) {
            if (same_file(candidate, diagnostic)) { diagnostics_safe = false; }
        } else if (errno != ENOENT && errno != ENOTDIR) {
            // If a possible path cannot be inspected, fail closed for diagnostics.
            diagnostics_safe = false;
        }
    }
}

struct Input {
    int fd = -1;
    struct stat identity{};
    Input(const std::string& path, const struct stat& report) {
        fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            fail("cannot open input: " + path);
        }
        if (fstat(fd, &identity) != 0) {
            close(fd);
            fd = -1;
            fail("cannot inspect input: " + path);
        }
        protect_diagnostics(identity);
        if (same_file(identity, report)) {
            close(fd);
            fd = -1;
            fail("input aliases report output: " + path);
        }
    }
    Input(const Input&) = delete;
    Input& operator=(const Input&) = delete;
    ~Input() { if (fd >= 0) { close(fd); } }
};

struct Output {
    std::string path;
    std::string name;
    int directory = -1;
    int fd = -1;
    int pin = -1;
    struct stat identity{};
    bool identified = false;
    bool committed = false;

    explicit Output(const std::string& p) : path(p) {}
    Output(const Output&) = delete;
    Output& operator=(const Output&) = delete;
    ~Output() {
        if (!committed && directory >= 0 && identified) {
            struct stat current{};
            if (fstatat(directory, name.c_str(), &current, AT_SYMLINK_NOFOLLOW) == 0 &&
                same_file(identity, current)) {
                unlinkat(directory, name.c_str(), 0);
            }
        }
        if (fd >= 0) { close(fd); }
        if (pin >= 0) { close(pin); }
        if (directory >= 0) { close(directory); }
    }
    void create() {
        const std::filesystem::path full(path);
        name = full.filename().string();
        const auto parent = full.parent_path().empty() ? std::filesystem::path(".") : full.parent_path();
        if (name.empty() || name == "." || name == "..") {
            fail("invalid output path: " + path);
        }
        directory = open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (directory < 0) {
            fail("cannot open output directory: " + path);
        }
        fd = openat(directory, name.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0666);
        if (fd < 0) {
            fail("cannot exclusively create output: " + path);
        }
        if (fstat(fd, &identity) != 0) {
            // Without an identity, stderr cannot safely be ruled out as an alias.
            diagnostics_safe = false;
            // Keep the descriptor open so the destructor can retry identification.
            struct stat retry{};
            if (fstat(fd, &retry) == 0) { identity = retry; identified = true; }
            fail("cannot inspect output: " + path);
        }
        identified = true;
        protect_diagnostics(identity);
    }
    void write_all(const void* data, std::size_t size) const {
        auto bytes = static_cast<const unsigned char*>(data);
        while (size > 0) {
            const auto chunk = std::min(size, static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
            const auto n = write(fd, bytes, chunk);
            if (n < 0 && errno == EINTR) { continue; }
            if (n <= 0) { fail("cannot write output: " + path); }
            bytes += static_cast<std::size_t>(n);
            size -= static_cast<std::size_t>(n);
        }
    }
    void finish() {
        // A duplicated writable descriptor would postpone the final writable
        // close (and its possible error) until after the report. O_PATH pins the
        // inode for identity-safe rollback without retaining a writable handle.
        const auto descriptor_path = "/proc/self/fd/" + std::to_string(fd);
        pin = open(descriptor_path.c_str(), O_PATH | O_CLOEXEC);
        if (pin < 0) { fail("cannot retain output identity: " + path); }
        struct stat pinned{};
        if (fstat(pin, &pinned) != 0 || !same_file(identity, pinned)) {
            fail("cannot verify output identity: " + path);
        }
        const auto result = close(fd);
        fd = -1;
        if (result != 0) { fail("cannot close output: " + path); }
    }
};

void save(const Output& out, std::size_t width, std::size_t height,
          const std::vector<unsigned char>& pixels) {
    const auto header = "P6\n" + std::to_string(width) + " " + std::to_string(height) + "\n255\n";
    out.write_all(header.data(), header.size());
    out.write_all(pixels.data(), pixels.size());
}

void diff(const Output& out, const Image& a, const Image& b) {
    std::vector<unsigned char> pixels(a.rgb.size());
    for (std::size_t i = 0; i < pixels.size(); ++i) {
        pixels[i] = byte(std::abs(a.rgb[i] - b.rgb[i]));
    }
    save(out, a.width, a.height, pixels);
}

void contact_sheet(const Output& out, const std::array<Image, 3>& images) {
    const auto width = multiply(images[0].width, 3);
    std::vector<unsigned char> pixels(multiply(multiply(width, images[0].height), 3));
    for (std::size_t y = 0; y < images[0].height; ++y) {
        for (std::size_t panel = 0; panel < 3; ++panel) {
            for (std::size_t x = 0; x < images[0].width; ++x) {
                for (std::size_t c = 0; c < 3; ++c) {
                    pixels[(y * width + panel * images[0].width + x) * 3 + c] =
                        byte(images[panel].rgb[(y * images[0].width + x) * 3 + c]);
                }
            }
        }
    }
    save(out, width, images[0].height, pixels);
}

void usage(std::ostream& out) {
    out << "Usage: metrics REFERENCE.ppm CHEAP.ppm NEURAL.ppm [options]\n"
           "  --region X Y WIDTH HEIGHT    zero-based rectangle, entirely within each image\n"
           "  --diff-cheap PATH           save full-image absolute RGB difference to reference\n"
           "  --diff-neural PATH          save full-image absolute RGB difference to reference\n"
           "  --sheet PATH                save full-image contact sheet (reference, cheap, neural)\n"
           "Output images must be new paths; existing files and symlinks are never overwritten.\n"
           "Use trusted output directories: cleanup is not atomic against concurrent path changes.\n"
           "Failures clean up owned outputs where possible; fatal signals can prevent cleanup.\n"
           "Input and rollback identity descriptors use /proc/self/fd.\n"
           "Reports (including help) are provisional until a successful exit status.\n"
           "Close errors on stdout are checked, but external holders of its file\n"
           "description can postpone final-close errors beyond this process.\n"
           "Do not redirect stdout or stderr to an input: shell > truncates before metrics can check it.\n"
           "Diagnostics are suppressed if stderr may alias an input or output image.\n"
           "P6 maxval 1..65535; exact raster length required; comments and CRLF supported.\n"
           "Metrics use encoded normalised RGB, not linear-light or perceptual colour.\n"
           "SSIM: luma, Gaussian 11x11 sigma 1.5, valid windows; smaller regions use the largest\n"
           "odd square fitting the region (same sigma); population moments, K1=.01 K2=.03.\n"
           "Detail: MAE of signed horizontal/vertical luma gradients; 0 if no edges exist.\n"
           "Lighting: area-weighted RMSE of mean luma difference in region-aligned 8x8 tiles.\n";
}

// When stderr shares the report destination, close our stderr handle before
// writing the report. Otherwise it can hide a last-close error on stdout. This
// deliberately sacrifices diagnostics in that case; external duplicates remain
// outside our control. Comparing identities is conservative for separate opens.
void prepare_report() {
    struct stat report_identity{};
    struct stat diagnostic_identity{};
    if (fstat(STDOUT_FILENO, &report_identity) != 0) {
        fail("cannot inspect report output");
    }
    if (fstat(STDERR_FILENO, &diagnostic_identity) == 0 &&
        same_file(report_identity, diagnostic_identity)) {
        std::cerr.flush();
        const auto flushed = static_cast<bool>(std::cerr);
        diagnostics_safe = false;
        const auto closed = close(STDERR_FILENO) == 0;
        if (!flushed || !closed) { fail("cannot close report diagnostics"); }
    }
}

void finish_report() {
    std::cout.flush();
    const auto flushed = static_cast<bool>(std::cout);
    // On Linux a failed close still consumes the descriptor; never retry it.
    const auto closed = close(STDOUT_FILENO) == 0;
    if (!flushed) { fail("cannot write report"); }
    if (!closed) { fail("cannot close report"); }
}

Options parse(int argc, char** argv) {
    if (argc < 4) {
        if (diagnostics_safe) { usage(std::cerr); }
        fail("expected three input paths");
    }
    Options opts{{argv[1], argv[2], argv[3]}, false, {}, {}, {}, {}};
    for (int i = 4; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--region") {
            if (opts.has_region || argc - i <= 4) {
                fail("--region requires four values and may occur once");
            }
            opts.region = {number(argv[i + 1], "region x"), number(argv[i + 2], "region y"),
                           number(argv[i + 3], "region width"), number(argv[i + 4], "region height")};
            opts.has_region = true;
            i += 4;
        } else if (arg == "--diff-cheap" || arg == "--diff-neural" || arg == "--sheet") {
            if (++i >= argc) {
                fail(arg + " requires a path");
            }
            auto& target = arg == "--diff-cheap" ? opts.cheap_diff :
                           arg == "--diff-neural" ? opts.neural_diff : opts.sheet;
            if (!target.empty() || std::string_view(argv[i]).empty()) {
                fail(arg + " requires one non-empty path and may occur once");
            }
            target = argv[i];
        } else {
            fail("unknown option: " + arg);
        }
    }
    return opts;
}


void report(const char* name, const std::string& path, const Metrics& m) {
    std::cout << name << " path=" << std::quoted(path) << '\n'
              << "  PSNR_RGB_dB=" << m.psnr << '\n'
              << "  SSIM_luma=" << m.ssim << " window=" << m.window << 'x' << m.window << '\n'
              << "  luma_MAE=" << m.luma_mae << '\n'
              << "  detail_gradient_MAE=" << m.detail_gradient_mae << '\n'
              << "  lighting_tile_mean_RMSE=" << m.lighting_block_rmse << '\n';
}

} // namespace

int main(int argc, char** argv) {
    inspect_diagnostic_aliases(argc, argv);
    try {
        // Convert broken report pipes and file-size limits into ordinary I/O failures,
        // so owned outputs can be cleaned up before exiting.
        if (std::signal(SIGPIPE, SIG_IGN) == SIG_ERR || std::signal(SIGXFSZ, SIG_IGN) == SIG_ERR) {
            fail("cannot configure I/O failure handling");
        }
        if (argc == 2 && std::string_view(argv[1]) == "--help") {
            prepare_report();
            usage(std::cout);
            finish_report();
            return 0;
        }
        const auto opts = parse(argc, argv);
        struct stat report_identity{};
        if (fstat(STDOUT_FILENO, &report_identity) != 0) {
            fail("cannot inspect report output");
        }
        const std::array<Input, 3> inputs{{{opts.inputs[0], report_identity},
                                          {opts.inputs[1], report_identity},
                                          {opts.inputs[2], report_identity}}};
        const std::array<Image, 3> images{load(opts.inputs[0], inputs[0].fd),
                                           load(opts.inputs[1], inputs[1].fd),
                                           load(opts.inputs[2], inputs[2].fd)};
        for (const auto& image : images) {
            if (image.width != images[0].width || image.height != images[0].height) {
                fail("input dimensions differ");
            }
        }
        const auto r = opts.has_region ? opts.region : Region{0, 0, images[0].width, images[0].height};
        if (r.width == 0 || r.height == 0 || r.x >= images[0].width || r.y >= images[0].height ||
            r.width > images[0].width - r.x || r.height > images[0].height - r.y) {
            fail("region is empty or outside the image");
        }
        std::array<std::unique_ptr<Output>, 3> outputs;
        const std::array<const std::string*, 3> paths{&opts.cheap_diff, &opts.neural_diff, &opts.sheet};
        for (std::size_t i = 0; i < paths.size(); ++i) {
            if (paths[i]->empty()) { continue; }
            outputs[i] = std::make_unique<Output>(*paths[i]);
            outputs[i]->create();
            if (same_file(outputs[i]->identity, report_identity)) {
                fail("output aliases report output: " + *paths[i]);
            }
            for (const auto& input : inputs) {
                if (same_file(outputs[i]->identity, input.identity)) {
                    fail("output aliases input: " + *paths[i]);
                }
            }
            for (std::size_t j = 0; j < i; ++j) {
                if (outputs[j] && same_file(outputs[i]->identity, outputs[j]->identity)) {
                    fail("output aliases another output: " + *paths[i]);
                }
            }
        }
        const auto cheap = compare(images[0], images[1], r);
        const auto neural = compare(images[0], images[2], r);
        if (!opts.cheap_diff.empty()) {
            diff(*outputs[0], images[0], images[1]);
        }
        if (!opts.neural_diff.empty()) {
            diff(*outputs[1], images[0], images[2]);
        }
        if (!opts.sheet.empty()) {
            contact_sheet(*outputs[2], images);
        }
        // Close and check every image before reporting success; keep each pin
        // and the rollback guard alive until the report is finalised.
        for (const auto& output : outputs) {
            if (output) { output->finish(); }
        }

        prepare_report();
        std::cout << "Report is provisional until successful process exit.\n"
                  << std::setprecision(10) << "Reference path=" << std::quoted(opts.inputs[0]) << '\n'
                  << "Image=" << images[0].width << 'x' << images[0].height
                  << " region=" << r.x << ',' << r.y << ',' << r.width << ',' << r.height << '\n'
                  << "Encoded RGB/luma metrics; reference compared with each candidate.\n";
        report("Cheap", opts.inputs[1], cheap);
        report("Neural", opts.inputs[2], neural);
        if (!opts.cheap_diff.empty()) {
            std::cout << "Cheap diff (full image, absolute RGB, unscaled): " << std::quoted(opts.cheap_diff) << '\n';
        }
        if (!opts.neural_diff.empty()) {
            std::cout << "Neural diff (full image, absolute RGB, unscaled): " << std::quoted(opts.neural_diff) << '\n';
        }
        if (!opts.sheet.empty()) {
            std::cout << "Contact sheet (full image, left to right: reference, cheap, neural; no embedded labels): "
                      << std::quoted(opts.sheet) << '\n';
        }
        finish_report();
        for (const auto& output : outputs) {
            if (output) { output->committed = true; }
        }
        return 0;
    } catch (const std::exception& error) {
        if (diagnostics_safe) { std::cerr << "metrics: " << error.what() << '\n'; }
        return 1;
    }
}
