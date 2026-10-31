#include "trainer.h"
#include <charconv>
#include <cerrno>
#include <cmath>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
namespace fs = std::filesystem;
namespace {
template<class T> T integer(const std::string& s) {
    T v{}; auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (ec != std::errc{} || end != s.data() + s.size()) { throw std::runtime_error("invalid integer: " + s); }
    return v;
}
double real(const std::string& s) {
    double v{}; auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (ec != std::errc{} || end != s.data() + s.size() || !std::isfinite(v)) { throw std::runtime_error("invalid number: " + s); }
    return v;
}
bool same_file(const struct stat& a, const struct stat& b) {
    return a.st_dev == b.st_dev && a.st_ino == b.st_ino;
}
struct LogSafety {
    // A stream is ok to write only when it cannot name (alias) an input file.
    bool stdout_ok = false;
    bool stderr_ok = false;
    bool stdout_alias = false;
    bool stderr_alias = false;
};
void record_identity(const std::string& path, std::vector<struct stat>& candidates, bool& complete) {
    struct stat st{};
    if (stat(path.c_str(), &st) == 0) { candidates.push_back(st); }
    else if (errno != ENOENT && errno != ENOTDIR) { complete = false; }
}
// Collect every path this invocation can name as an input: argv values and
// every image listed by the --train/--val manifests (mirroring data.cpp's
// manifest-relative resolution). A regular log stream is only safe when the
// candidate set is complete and the stream matches none of them; anything
// else is fail-closed, in preference to corrupting an input with a log write.
// Run before parsing or emitting usage: even a malformed invocation can name
// inputs. The trainer repeats this against the open input descriptors after
// loading; this scan closes the error paths before loading.
LogSafety verify_log_safety(int argc, char** argv) {
    std::vector<struct stat> candidates;
    bool complete = true;
    for (int i = 1; i < argc; ++i) { record_identity(argv[i], candidates, complete); }
    std::string manifests[2];
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string_view(argv[i]) == "--train") { manifests[0] = argv[i + 1]; }
        if (std::string_view(argv[i]) == "--val") { manifests[1] = argv[i + 1]; }
    }
    if (manifests[0].empty() || manifests[1].empty()) { complete = false; }
    for (const auto& m : manifests) {
        if (m.empty()) { continue; }
        std::ifstream in(m);
        if (!in) { complete = false; continue; }
        fs::path parent;
        try { parent = fs::canonical(m).parent_path(); }
        catch (const fs::filesystem_error&) { complete = false; continue; }
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') { line.pop_back(); }
            if (line.empty() || line[0] == '#') { continue; }
            auto tab = line.find('\t');
            if (tab == std::string::npos || tab == 0 || tab + 1 == line.size() || line.find('\t', tab + 1) != std::string::npos) {
                complete = false;
                continue;
            }
            record_identity((parent / line.substr(0, tab)).string(), candidates, complete);
            record_identity((parent / line.substr(tab + 1)).string(), candidates, complete);
        }
        if (in.bad()) { complete = false; }
    }
    LogSafety safety;
    for (int fd : {STDOUT_FILENO, STDERR_FILENO}) {
        struct stat st{};
        if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
            (fd == STDOUT_FILENO ? safety.stdout_ok : safety.stderr_ok) = true;
            continue;
        }
        bool alias = false;
        for (const auto& c : candidates) { if (same_file(st, c)) { alias = true; break; } }
        (fd == STDOUT_FILENO ? safety.stdout_alias : safety.stderr_alias) = alias;
        (fd == STDOUT_FILENO ? safety.stdout_ok : safety.stderr_ok) = !alias && complete;
    }
    return safety;
}
void write_all(int fd, const std::string& text) {
    const char* p = text.data();
    size_t left = text.size();
    while (left > 0) {
        auto n = ::write(fd, p, left);
        if (n < 0) { if (errno == EINTR) { continue; } break; }
        p += n; left -= static_cast<size_t>(n);
    }
}
void emit(const std::string& text, bool safe, std::ostream& stream) {
    if (safe) { stream << text; return; }
    // Never fall through to a stream that may name an input.
    int fd = ::open("/dev/tty", O_WRONLY | O_CLOEXEC);
    if (fd < 0) { return; }
    write_all(fd, text);
    ::close(fd);
}
void report(const std::string& message, bool safe_stderr) {
    emit("error: " + message + '\n', safe_stderr, std::cerr);
}
void usage(bool safe_stdout) {
    emit("Usage: scene-train --train MANIFEST --val MANIFEST --checkpoint NEW_FILE [--device cpu|cuda] [--epochs N] [--patch N] [--patches-per-image N] [--lr RATE] [--seed N]\n", safe_stdout, std::cout);
}
}
int main(int argc, char** argv) {
    auto safety = verify_log_safety(argc, argv);
    try {
        if (safety.stdout_alias || safety.stderr_alias) {
            throw scene_train::LogAliasError(safety.stdout_alias, safety.stderr_alias);
        }
        if (argc == 2 && std::string_view(argv[1]) == "--help") { usage(safety.stdout_ok); return 0; }
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc || !args.emplace(argv[i], argv[i + 1]).second) { throw std::runtime_error("missing value or duplicate option"); }
        }
        scene_train::Options o;
        for (const auto& [k, v] : args) {
            if (k == "--train") { o.train_manifest = v; }
            else if (k == "--val") { o.val_manifest = v; }
            else if (k == "--checkpoint") { o.checkpoint = v; }
            else if (k == "--device") { o.device = v; }
            else if (k == "--epochs") { o.epochs = integer<int>(v); }
            else if (k == "--patch") { o.patch = integer<int>(v); }
            else if (k == "--patches-per-image") { o.patches_per_image = integer<int>(v); }
            else if (k == "--seed") { o.seed = integer<uint64_t>(v); }
            else if (k == "--lr") { o.learning_rate = real(v); }
            else { throw std::runtime_error("unknown option: " + k); }
        }
        if (o.train_manifest.empty() || o.val_manifest.empty() || o.checkpoint.empty()) { throw std::runtime_error("train, val and checkpoint are required"); }
        scene_train::train(o); return 0;
    } catch (const scene_train::LogAliasError& e) {
        report(e.what(), safety.stderr_ok && !e.stderr_aliased);
        return 2;
    } catch (const std::exception& e) {
        report(e.what(), safety.stderr_ok);
        return 1;
    }
}
