#include "trainer.h"
#include <charconv>
#include <cerrno>
#include <cmath>
#include <fcntl.h>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
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
void usage() {
    std::cout << "Usage: scene-train --train MANIFEST --val MANIFEST --checkpoint NEW_FILE [--device cpu|cuda] [--epochs N] [--patch N] [--patches-per-image N] [--lr RATE] [--seed N]\n";
}
bool same_file(const struct stat& a, const struct stat& b) {
    return a.st_dev == b.st_dev && a.st_ino == b.st_ino;
}
// Path-level alias scan of the invocation against the log streams, mirroring
// scene-demo/metrics.cpp: run before parsing or emitting usage, because even a
// malformed invocation can name inputs. The trainer repeats this against the
// opened input descriptors after loading; this scan closes the earlier window.
void scan_log_aliases(int argc, char** argv, bool& stdout_aliased, bool& stderr_aliased) {
    struct stat out_st{}, err_st{};
    bool out_regular = fstat(STDOUT_FILENO, &out_st) == 0 && S_ISREG(out_st.st_mode);
    bool err_regular = fstat(STDERR_FILENO, &err_st) == 0 && S_ISREG(err_st.st_mode);
    for (int i = 1; i < argc; ++i) {
        struct stat candidate{};
        if (stat(argv[i], &candidate) != 0) { continue; }
        if (out_regular && same_file(out_st, candidate)) { stdout_aliased = true; }
        if (err_regular && same_file(err_st, candidate)) { stderr_aliased = true; }
    }
}
// Report through stderr only when it cannot corrupt an input; otherwise try
// the controlling terminal, and stay silent rather than corrupt a file.
void report(const std::string& message, bool stderr_aliased) {
    auto text = "error: " + message + '\n';
    if (!stderr_aliased) { std::cerr << text; return; }
    int fd = ::open("/dev/tty", O_WRONLY | O_CLOEXEC);
    if (fd < 0) { return; }
    const char* p = text.data();
    size_t left = text.size();
    while (left > 0) {
        auto n = ::write(fd, p, left);
        if (n < 0) { if (errno == EINTR) { continue; } break; }
        p += n; left -= static_cast<size_t>(n);
    }
    ::close(fd);
}
}
int main(int argc, char** argv) {
    bool stdout_aliased = false, stderr_aliased = false;
    try {
        scan_log_aliases(argc, argv, stdout_aliased, stderr_aliased);
        if (stdout_aliased || stderr_aliased) { throw scene_train::LogAliasError(stdout_aliased, stderr_aliased); }
        if (argc == 2 && std::string_view(argv[1]) == "--help") { usage(); return 0; }
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
        report(e.what(), e.stderr_aliased);
        return 2;
    } catch (const std::exception& e) {
        report(e.what(), stderr_aliased);
        return 1;
    }
}
