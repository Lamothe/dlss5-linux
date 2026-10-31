#include "host.h"

#include <array>
#include <charconv>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {
constexpr size_t max_header = 65536;
constexpr size_t max_pixels = 67108864;
constexpr size_t max_file = max_header + max_pixels * 3;

struct Fd {
    int value = -1;
    ~Fd() { if (value >= 0) { ::close(value); } }
};
struct Image { uint32_t width, height; std::vector<uint8_t> pixels; };

std::vector<uint8_t> read_file(const std::string& path) {
    Fd fd{::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK)};
    if (fd.value < 0) { throw std::runtime_error("cannot open input: " + std::string(std::strerror(errno))); }
    struct stat st{};
    if (::fstat(fd.value, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0 ||
        static_cast<uint64_t>(st.st_size) > max_file) {
        throw std::runtime_error("input must be a bounded regular file");
    }
    std::vector<uint8_t> data(static_cast<size_t>(st.st_size));
    size_t offset = 0;
    while (offset < data.size()) {
        auto n = ::read(fd.value, data.data() + offset, data.size() - offset);
        if (n < 0 && errno == EINTR) { continue; }
        if (n <= 0) { throw std::runtime_error("input read failed or size changed"); }
        offset += static_cast<size_t>(n);
    }
    uint8_t extra;
    ssize_t n;
    do { n = ::read(fd.value, &extra, 1); } while (n < 0 && errno == EINTR);
    if (n != 0) { throw std::runtime_error("input read failed or size changed"); }
    return data;
}

Image read_ppm(const std::string& path) {
    auto data = read_file(path);
    size_t pos = 0;
    auto check = [&] { if (pos >= max_header) { throw std::runtime_error("PPM header exceeds limit"); } };
    auto token = [&]() {
        while (pos < data.size()) {
            check();
            if (std::isspace(data[pos])) { ++pos; continue; }
            if (data[pos] == '#') {
                while (pos < data.size() && data[pos] != '\r' && data[pos] != '\n') { check(); ++pos; }
                continue;
            }
            break;
        }
        std::string s;
        while (pos < data.size() && !std::isspace(data[pos]) && data[pos] != '#') {
            check();
            if (s.size() >= 32) { throw std::runtime_error("PPM header token too long"); }
            s.push_back(static_cast<char>(data[pos++]));
        }
        if (s.empty()) { throw std::runtime_error("invalid or truncated PPM header"); }
        return s;
    };
    auto number = [&]() {
        auto s = token(); uint32_t value = 0;
        auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), value);
        if (ec != std::errc{} || end != s.data() + s.size() || value == 0 || value > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("invalid PPM number");
        }
        return value;
    };
    if (token() != "P6") { throw std::runtime_error("only binary P6 PPM is supported"); }
    auto width = number(); auto height = number();
    if (number() != 255) { throw std::runtime_error("PPM maxval must be 255"); }
    if (pos < data.size() && data[pos] == '#') {
        while (pos < data.size() && data[pos] != '\r' && data[pos] != '\n') { check(); ++pos; }
    }
    check();
    if (pos == data.size() || !std::isspace(data[pos])) { throw std::runtime_error("missing PPM raster separator"); }
    auto separator = data[pos++];
    if (width > 16384 || height > 16384 || static_cast<uint64_t>(width) * height > max_pixels) {
        throw std::runtime_error("PPM dimensions exceed limit");
    }
    auto bytes = static_cast<size_t>(width) * height * 3;
    if (separator == '\r' && pos < data.size() && data[pos] == '\n' && data.size() - pos == bytes + 1) { check(); ++pos; }
    if (data.size() - pos != bytes) { throw std::runtime_error("PPM raster length does not match dimensions"); }
    return {width, height, std::vector<uint8_t>(data.begin() + static_cast<std::ptrdiff_t>(pos), data.end())};
}

void write_output(const std::string& path, const Image& image, const std::vector<uint8_t>& pixels) {
    Fd fd{::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0666)};
    if (fd.value < 0) { throw std::runtime_error("cannot create output exclusively: " + std::string(std::strerror(errno))); }
    // Only this successful exclusive creation owns cleanup. The destination
    // directory and its path must be exclusively controlled, including against
    // changes by trusted concurrent processes.
    try {
        auto write_all = [&](const void* buffer, size_t length) {
            auto* data = static_cast<const uint8_t*>(buffer);
            while (length) {
                auto n = ::write(fd.value, data, length);
                if (n < 0 && errno == EINTR) { continue; }
                if (n <= 0) { throw std::runtime_error("output write failed"); }
                data += n; length -= static_cast<size_t>(n);
            }
        };
        auto header = "P6\n" + std::to_string(image.width) + " " + std::to_string(image.height) + "\n255\n";
        write_all(header.data(), header.size());
        write_all(pixels.data(), pixels.size());
        auto closing = fd.value; fd.value = -1;
        if (::close(closing) != 0) { throw std::runtime_error("output close failed"); }
    } catch (...) {
        ::unlink(path.c_str());
        throw;
    }
}
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--help") {
            std::cout << "Usage: scene-infer --input INPUT.ppm --output NEW.ppm --checkpoint TRUSTED.pt\n";
            return 0;
        }
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc || !args.emplace(argv[i], argv[i + 1]).second) { throw std::runtime_error("missing value or duplicate option"); }
        }
        for (const auto& [key, value] : args) {
            if (key != "--input" && key != "--output" && key != "--checkpoint") { throw std::runtime_error("unknown option: " + key); }
            if (value.empty()) { throw std::runtime_error("empty path"); }
        }
        if (args.size() != 3) { throw std::runtime_error("input, output and trusted checkpoint are required"); }
        auto image = read_ppm(args.at("--input"));
        std::array<char, 1024> error{};
        scene_host* raw = nullptr;
        auto loaded = scene_host_load(args.at("--checkpoint").c_str(), &raw, error.data(), error.size());
        std::unique_ptr<scene_host, decltype(&scene_host_destroy)> host(raw, scene_host_destroy);
        if (loaded != 0) { throw std::runtime_error(std::string("host load: ") + error.data()); }
        std::vector<uint8_t> output(image.pixels.size());
        auto stride = static_cast<size_t>(image.width) * 3;
        if (scene_host_infer(host.get(), image.pixels.data(), image.pixels.size(), stride,
                             output.data(), output.size(), stride, image.width, image.height,
                             error.data(), error.size()) != 0) {
            throw std::runtime_error(std::string("host inference: ") + error.data());
        }
        write_output(args.at("--output"), image, output);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
