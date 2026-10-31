#pragma once
#include <filesystem>
#include <functional>
#include <string>

namespace scene_train {
// The caller must maintain exclusive control of the containing directory:
// it must not be writable by untrusted parties; no other process, including
// trusted concurrent processes, may modify entries or change its path.
class CheckpointFile {
public:
    explicit CheckpointFile(const std::filesystem::path& path);
    ~CheckpointFile();
    CheckpointFile(const CheckpointFile&) = delete;
    CheckpointFile& operator=(const CheckpointFile&) = delete;
    void replace(const std::function<void(int)>& writer);
private:
    int dir_ = -1, reserved_ = -1;
    std::string name_;
    bool committed_ = false;
};
}
