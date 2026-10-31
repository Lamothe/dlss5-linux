#include "checkpoint.h"
#include <array>
#include <cerrno>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <system_error>
#include <sys/random.h>
#include <sys/stat.h>
#include <unistd.h>

namespace scene_train {
namespace {
[[noreturn]] void fail(const char* message) { throw std::system_error(errno, std::generic_category(), message); }
void remove_if_same(int dir, const std::string& name, int fd) noexcept {
    struct stat a{}, b{};
    if (fd >= 0 && fstat(fd, &a) == 0 && fstatat(dir, name.c_str(), &b, AT_SYMLINK_NOFOLLOW) == 0 &&
        a.st_dev == b.st_dev && a.st_ino == b.st_ino) { unlinkat(dir, name.c_str(), 0); }
}
std::string temporary_name() {
    std::array<unsigned char, 16> bytes{};
    size_t done = 0;
    while (done < bytes.size()) {
        auto n = getrandom(bytes.data() + done, bytes.size() - done, 0);
        if (n < 0 && errno == EINTR) { continue; }
        if (n <= 0) { if (n == 0) { errno = EIO; } fail("random temporary name"); }
        done += static_cast<size_t>(n);
    }
    constexpr char hex[] = "0123456789abcdef";
    std::string result = ".scene-train-";
    for (auto b : bytes) { result += hex[b >> 4]; result += hex[b & 15]; }
    return result;
}
}
CheckpointFile::CheckpointFile(const std::filesystem::path& path) {
    name_ = path.filename().string();
    if (name_.empty() || name_ == "." || name_ == "..") { throw std::runtime_error("checkpoint must be a new file"); }
    auto parent = path.parent_path();
    if (parent.empty()) { parent = "."; }
    dir_ = open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dir_ < 0) { fail("open checkpoint directory"); }
    reserved_ = openat(dir_, name_.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (reserved_ < 0) { auto error = errno; close(dir_); dir_ = -1; errno = error; fail("reserve new checkpoint"); }
}
CheckpointFile::~CheckpointFile() {
    if (!committed_) { remove_if_same(dir_, name_, reserved_); }
    if (reserved_ >= 0) { close(reserved_); }
    if (dir_ >= 0) { close(dir_); }
}
void CheckpointFile::replace(const std::function<void(int)>& writer) {
    std::string temp;
    int fd = -1;
    for (int attempt = 0; attempt < 100; ++attempt) {
        temp = temporary_name();
        fd = openat(dir_, temp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
        if (fd >= 0) { break; }
        if (errno != EEXIST) { fail("create checkpoint temporary file"); }
    }
    if (fd < 0) { throw std::runtime_error("cannot create unique checkpoint temporary file"); }
    try {
        writer(fd);
        if (fsync(fd) != 0) { fail("sync checkpoint temporary file"); }
        if (renameat(dir_, temp.c_str(), dir_, name_.c_str()) != 0) { fail("replace checkpoint"); }
        committed_ = true;
    } catch (...) {
        remove_if_same(dir_, temp, fd);
        close(fd);
        throw;
    }
    close(fd);
}
}
