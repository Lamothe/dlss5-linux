#include "model.h"
#include <torch/torch.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
namespace {
constexpr int width = 4, height = 4;
void require(bool ok, const std::string& message) { if (!ok) { throw std::runtime_error(message); } }
struct Temp {
    fs::path path;
    Temp() {
        auto s = (fs::temp_directory_path() / "scene-integration-XXXXXX").string();
        std::vector<char> name(s.begin(), s.end()); name.push_back('\0');
        auto* p = ::mkdtemp(name.data()); require(p != nullptr, "mkdtemp failed"); path = p;
    }
    ~Temp() { std::error_code ec; fs::remove_all(path, ec); }
};
void save(const fs::path& path, const std::string& bytes) {
    std::ofstream out(path, std::ios::binary); out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    out.close(); require(static_cast<bool>(out), "write failed: " + path.string());
}
std::string read(const fs::path& path) {
    std::ifstream in(path, std::ios::binary); require(static_cast<bool>(in), "read failed: " + path.string());
    std::string bytes{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    require(!in.bad(), "read failed: " + path.string()); return bytes;
}
void run(const std::string& exe, const std::vector<std::string>& args, const fs::path& output) {
    auto pid = ::fork(); require(pid >= 0, "fork failed");
    if (pid == 0) {
        auto fd = ::open(output.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
        if (fd < 0 || ::dup2(fd, STDOUT_FILENO) < 0) { _exit(126); }
        ::close(fd);
        std::vector<char*> argv{const_cast<char*>(exe.c_str())};
        for (const auto& arg : args) { argv.push_back(const_cast<char*>(arg.c_str())); }
        argv.push_back(nullptr); ::execv(exe.c_str(), argv.data()); _exit(127);
    }
    int status = 0;
    while (::waitpid(pid, &status, 0) < 0) { if (errno != EINTR) { throw std::runtime_error("waitpid failed"); } }
    require(WIFEXITED(status) && WEXITSTATUS(status) == 0, "command failed: " + exe);
}
std::string ppm(const std::vector<uint8_t>& pixels) {
    return "P6\n4 4\n255\n" + std::string(reinterpret_cast<const char*>(pixels.data()), pixels.size());
}
std::vector<uint8_t> fixture(int offset, bool target) {
    std::vector<uint8_t> pixels;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < 3; ++c) {
                auto value = (offset + x * 37 + y * 51 + c * 73) % 256;
                pixels.push_back(static_cast<uint8_t>(target ? (value * 3 / 4 + 20) : value));
            }
        }
    }
    return pixels;
}
double psnr(const std::vector<uint8_t>& reference, const std::vector<uint8_t>& candidate) {
    double sum = 0;
    for (size_t i = 0; i < reference.size(); ++i) {
        auto d = (static_cast<double>(reference[i]) - candidate[i]) / 255.0; sum += d * d;
    }
    return sum == 0 ? std::numeric_limits<double>::infinity() : -10 * std::log10(sum / reference.size());
}
double value(const std::string& line, const std::string& key, const std::string& suffix = "") {
    auto prefix = "  " + key + "=";
    require(line.starts_with(prefix) && line.ends_with(suffix) && line.size() > prefix.size() + suffix.size(), "invalid report field: " + line);
    auto text = line.substr(prefix.size(), line.size() - prefix.size() - suffix.size());
    size_t used = 0; auto result = std::stod(text, &used);
    require(used == text.size() && !std::isnan(result), "invalid report number: " + line); return result;
}
std::string quoted(const fs::path& path) { std::ostringstream out; out << std::quoted(path.string()); return out.str(); }
void report(const std::string& text, const fs::path& reference, const fs::path& cheap, const fs::path& neural,
            const std::vector<uint8_t>& target_pixels, const std::vector<uint8_t>& cheap_pixels,
            const std::vector<uint8_t>& neural_pixels) {
    std::istringstream in(text); std::string line;
    auto expect = [&](const std::string& expected) { require(static_cast<bool>(std::getline(in, line)) && line == expected, "unexpected report line: " + line); };
    auto number = [&](const std::string& key, const std::string& suffix = "") {
        require(static_cast<bool>(std::getline(in, line)), "missing report field: " + key); return value(line, key, suffix);
    };
    expect("Report is provisional until successful process exit.");
    expect("Reference path=" + quoted(reference));
    expect("Image=4x4 region=0,0,4,4");
    expect("Encoded RGB/luma metrics; reference compared with each candidate.");
    for (const auto& entry : std::array<std::pair<std::string, const std::vector<uint8_t>*>, 2>{{
             {"Cheap path=" + quoted(cheap), &cheap_pixels},
             {"Neural path=" + quoted(neural), &neural_pixels}}}) {
        expect(entry.first);
        auto measured = number("PSNR_RGB_dB"); auto expected = psnr(target_pixels, *entry.second);
        require((std::isinf(expected) && measured == expected) || (std::isfinite(measured) && measured >= 0 && std::abs(measured - expected) < 1e-7), "PSNR mismatch");
        auto ssim = number("SSIM_luma", " window=3x3");
        auto luma = number("luma_MAE"); auto detail = number("detail_gradient_MAE"); auto lighting = number("lighting_tile_mean_RMSE");
        require(std::isfinite(ssim) && ssim >= -1.000001 && ssim <= 1.000001 &&
                std::isfinite(luma) && luma >= 0 && luma <= 1 &&
                std::isfinite(detail) && detail >= 0 && detail <= 2 &&
                std::isfinite(lighting) && lighting >= 0 && lighting <= 1, "metric outside bounds");
    }
    require(!std::getline(in, line) && in.eof(), "unexpected extra report output");
}
}
int main(int argc, char** argv) {
    try {
        require(argc == 4, "expected train, infer and metrics executables");
        Temp temp; torch::set_num_threads(1);
        auto train_cheap = fixture(9, false), train_full = fixture(9, true);
        auto val_cheap = fixture(27, false), val_full = fixture(27, true);
        auto path = [&](const char* name) { return temp.path / name; };
        save(path("train-cheap.ppm"), ppm(train_cheap)); save(path("train-full.ppm"), ppm(train_full));
        save(path("val-cheap.ppm"), ppm(val_cheap)); save(path("val-full.ppm"), ppm(val_full));
        save(path("train.tsv"), "train-cheap.ppm\ttrain-full.ppm\n");
        save(path("val.tsv"), "val-cheap.ppm\tval-full.ppm\n");
        run(argv[1], {"--train", path("train.tsv").string(), "--val", path("val.tsv").string(),
             "--checkpoint", path("model.pt").string(), "--device", "cpu", "--epochs", "3", "--patch", "4",
             "--patches-per-image", "1", "--lr", "0.001", "--seed", "42"}, path("train.log"));
        auto log = read(path("train.log"));
        require(log.find("epoch=3 train_mse=") != std::string::npos && log.find("best_epoch=") != std::string::npos, "incomplete training log");
        run(argv[2], {"--input", path("val-cheap.ppm").string(), "--output", path("inferred.ppm").string(),
             "--checkpoint", path("model.pt").string()}, path("infer.log"));
        auto actual = read(path("inferred.ppm"));
        constexpr auto header = "P6\n4 4\n255\n";
        require(actual.starts_with(header) && actual.size() == std::string(header).size() + val_cheap.size(), "output dimensions or P6 raster length incorrect");
        std::vector<uint8_t> inferred(actual.begin() + static_cast<std::ptrdiff_t>(std::string(header).size()), actual.end());
        // Load the actual CLI checkpoint and independently apply the host's documented conversion.
        scene_train::ResidualNet model; torch::load(model, path("model.pt").string()); model->eval(); torch::NoGradGuard guard;
        auto input = torch::from_blob(val_cheap.data(), {height, width, 3}, torch::kUInt8).permute({2, 0, 1}).unsqueeze(0).to(torch::kFloat32).div(255.0);
        auto raw = model->forward(input); require(torch::isfinite(raw).all().item<bool>(), "non-finite model output");
        auto direct = raw.clamp(0, 1).mul(255).round().to(torch::kUInt8).squeeze(0).permute({1, 2, 0}).contiguous();
        auto* bytes = direct.data_ptr<uint8_t>();
        require(std::equal(inferred.begin(), inferred.end(), bytes), "trained checkpoint direct output differs from host inference");
        run(argv[3], {path("val-full.ppm").string(), path("val-cheap.ppm").string(), path("inferred.ppm").string()}, path("metrics.txt"));
        report(read(path("metrics.txt")), path("val-full.ppm"), path("val-cheap.ppm"), path("inferred.ppm"), val_full, val_cheap, inferred);
        std::cout << "Synthetic CPU train/infer/metrics integration passed (not scene-quality evidence).\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "integration: " << e.what() << '\n'; return 1; }
}
