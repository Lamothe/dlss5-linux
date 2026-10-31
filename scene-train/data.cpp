#include "data.h"
#include <charconv>
#include <cctype>
#include <cerrno>
#include <fcntl.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
namespace fs = std::filesystem;
namespace scene_train {
namespace {
// One open descriptor supplies identity (fstat) and content (read): a path
// replaced after open cannot defeat the alias checks or swap the bytes read.
struct FileBytes { std::string path; std::string bytes; dev_t dev; ino_t ino; };
FileBytes read_file(const fs::path& path) {
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) { throw std::runtime_error("cannot open PPM: " + path.string()); }
    struct stat st{};
    if (fstat(fd, &st) != 0) { auto e = errno; ::close(fd); errno = e; throw std::runtime_error("cannot inspect input: " + path.string()); }
    if (!S_ISREG(st.st_mode)) { ::close(fd); throw std::runtime_error("not a regular file: " + path.string()); }
    std::string bytes;
    char buffer[65536];
    while (true) {
        auto n = ::read(fd, buffer, sizeof buffer);
        if (n < 0) {
            if (errno == EINTR) { continue; }
            auto e = errno; ::close(fd); errno = e; throw std::runtime_error("cannot read input: " + path.string());
        }
        if (n == 0) { break; }
        bytes.append(buffer, static_cast<size_t>(n));
    }
    ::close(fd);
    return {path.string(), std::move(bytes), st.st_dev, st.st_ino};
}
void record_unique(const FileBytes& file, std::vector<InputFile>& seen) {
    for (const auto& old : seen) {
        if (old.dev == file.dev && old.ino == file.ino) {
            throw std::runtime_error("input file reused (including aliases/hard links): " + file.path);
        }
    }
    seen.push_back({file.path, file.dev, file.ino});
}
std::string token(std::istream& in) {
    while (true) {
        int c = in.peek();
        if (c == EOF) { throw std::runtime_error("truncated PPM header"); }
        if (std::isspace(static_cast<unsigned char>(c))) { in.get(); continue; }
        if (c == '#') {
            while (in.peek() != EOF && in.peek() != '\r' && in.peek() != '\n') { in.get(); }
            continue;
        }
        break;
    }
    std::string s;
    while (true) {
        int c = in.peek();
        if (c == EOF || std::isspace(static_cast<unsigned char>(c)) || c == '#') { break; }
        if (s.size() >= 32) { throw std::runtime_error("PPM header token too long"); }
        s.push_back(static_cast<char>(in.get()));
    }
    if (s.empty()) { throw std::runtime_error("invalid PPM header"); }
    return s;
}
int number(const std::string& s) {
    int n = 0; auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), n);
    if (ec != std::errc{} || p != s.data() + s.size() || n <= 0) { throw std::runtime_error("invalid PPM number"); }
    return n;
}
Image parse_ppm(const FileBytes& file) {
    std::istringstream in(file.bytes);
    if (token(in) != "P6") { throw std::runtime_error("only binary P6 PPM is supported: " + file.path); }
    int w = number(token(in)), h = number(token(in));
    if (number(token(in)) != 255) { throw std::runtime_error("PPM maxval must be 255"); }
    // At this boundary only a comment directly after maxval is header.
    // Its CR or LF terminator is the raster separator, not part of the comment.
    if (in.peek() == '#') {
        while (in.peek() != EOF && in.peek() != '\r' && in.peek() != '\n') { in.get(); }
    }
    int separator = in.get();
    if (separator == EOF || !std::isspace(static_cast<unsigned char>(separator))) { throw std::runtime_error("missing PPM raster separator"); }
    if (w > 16384 || h > 16384 || static_cast<int64_t>(w) * h > 67108864) { throw std::runtime_error("PPM dimensions exceed limit"); }
    size_t bytes = static_cast<size_t>(w) * static_cast<size_t>(h) * 3;
    // CR can be the sole separator when the first pixel byte is LF. Exact
    // remaining length distinguishes that case from a CRLF separator.
    if (separator == '\r' && in.peek() == '\n') {
        auto raster_start = in.tellg();
        in.seekg(0, std::ios::end);
        auto end = in.tellg();
        if (raster_start < 0 || end < raster_start) { throw std::runtime_error("cannot determine PPM raster length"); }
        in.seekg(raster_start);
        if (end - raster_start == static_cast<std::streamoff>(bytes + 1)) { in.get(); }
    }
    std::vector<unsigned char> buffer(bytes);
    if (!in.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(bytes)) || in.peek() != EOF) {
        throw std::runtime_error("PPM raster length does not match dimensions: " + file.path);
    }
    auto tensor = torch::from_blob(buffer.data(), {h, w, 3}, torch::kUInt8).permute({2, 0, 1}).to(torch::kFloat32).div_(255.0).contiguous();
    return {w, h, tensor};
}
std::vector<Pair> manifest(const FileBytes& file, const fs::path& parent,
                           std::vector<InputFile>& seen, int& width, int& height, int patch) {
    std::istringstream in(file.bytes);
    std::vector<Pair> pairs;
    std::string line; size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (!line.empty() && line.back() == '\r') { line.pop_back(); }
        if (line.empty() || line[0] == '#') { continue; }
        auto tab = line.find('\t');
        if (tab == std::string::npos || tab == 0 || tab + 1 == line.size() || line.find('\t', tab + 1) != std::string::npos) {
            throw std::runtime_error("manifest must contain exactly two tab-separated paths at line " + std::to_string(line_no));
        }
        auto cheap = read_file(parent / line.substr(0, tab));
        record_unique(cheap, seen);
        auto full = read_file(parent / line.substr(tab + 1));
        record_unique(full, seen);
        auto a = parse_ppm(cheap); auto b = parse_ppm(full);
        if (a.width != b.width || a.height != b.height) { throw std::runtime_error("pair dimensions differ: " + cheap.path); }
        if (width == 0) { width = a.width; height = a.height; }
        if (a.width != width || a.height != height) { throw std::runtime_error("all images must have identical dimensions"); }
        if (patch > width || patch > height) { throw std::runtime_error("patch exceeds image dimensions"); }
        pairs.push_back({cheap.path, full.path, std::move(a), std::move(b)});
    }
    if (in.bad()) { throw std::runtime_error("error reading manifest"); }
    if (pairs.empty()) { throw std::runtime_error("empty manifest: " + file.path); }
    return pairs;
}
}
Image read_ppm(const fs::path& path) {
    return parse_ppm(read_file(path));
}
Dataset load_dataset(const fs::path& train, const fs::path& val, int patch) {
    if (patch <= 0) { throw std::runtime_error("patch must be positive"); }
    auto train_file = read_file(train);
    auto val_file = read_file(val);
    if (train_file.dev == val_file.dev && train_file.ino == val_file.ino) {
        throw std::runtime_error("train and validation manifests must differ");
    }
    std::vector<InputFile> seen;
    seen.push_back({train_file.path, train_file.dev, train_file.ino});
    seen.push_back({val_file.path, val_file.dev, val_file.ino});
    int width = 0, height = 0;
    auto a = manifest(train_file, fs::canonical(train).parent_path(), seen, width, height, patch);
    auto b = manifest(val_file, fs::canonical(val).parent_path(), seen, width, height, patch);
    return {std::move(a), std::move(b), std::move(seen)};
}
}
