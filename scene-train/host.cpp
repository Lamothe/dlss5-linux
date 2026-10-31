#include "host.h"
#include "model.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <limits>
#include <fcntl.h>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

struct scene_host {
    scene_train::ResidualNet model;
};

namespace {
struct Fd {
    int value = -1;
    ~Fd() { if (value >= 0) { ::close(value); } }
};
void message(char* error, size_t capacity, const char* text) noexcept {
    if (error && capacity) { std::snprintf(error, capacity, "%s", text); }
}
size_t required_size(size_t stride, size_t row, uint32_t height) {
    if (stride < row || (height > 1 && stride > (std::numeric_limits<size_t>::max() - row) / (height - 1))) {
        throw std::runtime_error("invalid or overflowing stride");
    }
    return stride * (height - 1) + row;
}
}

extern "C" const char* scene_host_label(void) {
    return "neural rendering demonstration - texture & lighting update (not DLSS5)";
}

extern "C" int scene_host_load(const char* checkpoint, scene_host** out, char* error, size_t error_capacity) {
    if (out) { *out = nullptr; }
    try {
        if (!out || !checkpoint || !*checkpoint) { throw std::runtime_error("checkpoint and output handle are required"); }
        // Read once so validation and load always see the same checkpoint bytes.
        // The fixed model has only 1,027 float parameters. This cap leaves
        // ample room for archive metadata; it does not bound decoded storage.
        constexpr size_t max_checkpoint_bytes = 1024 * 1024;
        Fd file{::open(checkpoint, O_RDONLY | O_CLOEXEC | O_NONBLOCK)};
        if (file.value < 0) { throw std::runtime_error("cannot open checkpoint"); }
        struct stat st{};
        if (::fstat(file.value, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0) {
            throw std::runtime_error("checkpoint must be a regular file");
        }
        if (static_cast<uint64_t>(st.st_size) > max_checkpoint_bytes) {
            throw std::runtime_error("checkpoint exceeds size limit");
        }
        std::string saved(static_cast<size_t>(st.st_size), '\0');
        size_t offset = 0;
        while (offset < saved.size()) {
            auto n = ::read(file.value, saved.data() + offset, saved.size() - offset);
            if (n < 0 && errno == EINTR) { continue; }
            if (n <= 0) { throw std::runtime_error("cannot read checkpoint or checkpoint size changed"); }
            offset += static_cast<size_t>(n);
        }
        char extra;
        ssize_t n;
        do { n = ::read(file.value, &extra, 1); } while (n < 0 && errno == EINTR);
        if (n != 0) { throw std::runtime_error("cannot read checkpoint or checkpoint size changed"); }
        auto host = std::make_unique<scene_host>();
        // Validate archive tensors before Module::load can silently copy/cast them.
        torch::serialize::InputArchive archive;
        archive.load_from(saved.data(), saved.size(), torch::Device(torch::kCPU));
        for (const auto& child : host->model->named_children()) {
            torch::serialize::InputArchive nested;
            archive.read(child.key(), nested);
            for (const auto& parameter : child.value()->named_parameters(false)) {
                torch::Tensor tensor;
                nested.read(parameter.key(), tensor);
                if (!tensor.defined() || tensor.device().type() != torch::kCPU ||
                    tensor.scalar_type() != torch::kFloat32 || tensor.layout() != torch::kStrided ||
                    tensor.sizes() != parameter.value().sizes() || !torch::isfinite(tensor).all().item<bool>()) {
                    throw std::runtime_error("invalid or nonfinite checkpoint parameters");
                }
            }
        }
        std::vector<std::vector<int64_t>> shapes;
        for (const auto& p : host->model->parameters()) { shapes.push_back(p.sizes().vec()); }
        std::istringstream stream(saved, std::ios::in | std::ios::binary);
        torch::load(host->model, stream, torch::Device(torch::kCPU));
        auto params = host->model->parameters();
        if (params.size() != shapes.size()) { throw std::runtime_error("invalid checkpoint parameters"); }
        for (size_t i = 0; i < params.size(); ++i) {
            const auto& p = params[i];
            if (!p.defined() || p.device().type() != torch::kCPU || p.scalar_type() != torch::kFloat32 ||
                p.sizes().vec() != shapes[i] || !torch::isfinite(p).all().item<bool>()) {
                throw std::runtime_error("invalid or nonfinite checkpoint parameters");
            }
        }
        host->model->eval();
        *out = host.release();
        message(error, error_capacity, "");
        return 0;
    } catch (const std::exception& e) {
        message(error, error_capacity, e.what());
    } catch (...) {
        message(error, error_capacity, "unknown error loading checkpoint");
    }
    return 1;
}

extern "C" void scene_host_destroy(scene_host* host) {
    try { delete host; } catch (...) { /* No exceptions cross the ABI. */ }
}

extern "C" int scene_host_infer(scene_host* host, const uint8_t* input, size_t input_size,
                                 size_t input_stride, uint8_t* output, size_t output_size,
                                 size_t output_stride, uint32_t width, uint32_t height,
                                 char* error, size_t error_capacity) {
    try {
        if (!host || !input || !output) { throw std::runtime_error("handle and buffers are required"); }
        if (!width || !height || width > 16384 || height > 16384 ||
            static_cast<uint64_t>(width) * height > 67108864) {
            throw std::runtime_error("invalid image dimensions");
        }
        auto row = static_cast<size_t>(width) * 3;
        if (input_size < required_size(input_stride, row, height) ||
            output_size < required_size(output_stride, row, height)) {
            throw std::runtime_error("buffer too small");
        }
        torch::NoGradGuard guard;
        auto rgb = torch::empty({static_cast<int64_t>(height), static_cast<int64_t>(width), 3}, torch::kUInt8);
        auto* pixels = rgb.data_ptr<uint8_t>();
        for (size_t y = 0; y < height; ++y) { std::memcpy(pixels + y * row, input + y * input_stride, row); }
        auto nchw = rgb.permute({2, 0, 1}).unsqueeze(0).to(torch::kFloat32).div(255.0);
        auto result = host->model->forward(nchw);
        if (!torch::isfinite(result).all().item<bool>()) { throw std::runtime_error("model produced nonfinite output"); }
        auto bytes = result.clamp(0, 1).mul(255).round().to(torch::kUInt8).squeeze(0).permute({1, 2, 0}).contiguous();
        const auto* data = bytes.data_ptr<uint8_t>();
        for (size_t y = 0; y < height; ++y) { std::memcpy(output + y * output_stride, data + y * row, row); }
        message(error, error_capacity, "");
        return 0;
    } catch (const std::exception& e) {
        message(error, error_capacity, e.what());
    } catch (...) {
        message(error, error_capacity, "unknown inference error");
    }
    return 1;
}
