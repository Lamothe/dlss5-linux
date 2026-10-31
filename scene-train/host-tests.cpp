#ifdef HOST_C_CONSUMER
#include "host.h"
#include <string.h>
/* This translation unit is compiled as C and linked against the shared ABI. */
int main(void) {
    char error[64];
    scene_host *handle = 0;
    if (strcmp(scene_host_label(), "neural rendering demonstration - texture & lighting update (not DLSS5)") != 0 ||
        scene_host_load(0, &handle, error, sizeof error) == 0 || handle != 0 || error[0] == 0) { return 1; }
    scene_host_destroy(handle);
    return 0;
}
#else
#include "host.h"
#include "model.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/wait.h>
#include <cerrno>
#include <vector>
#include <unistd.h>

namespace fs = std::filesystem;
namespace {
void check(bool condition, const char* text) { if (!condition) { throw std::runtime_error(text); } }
struct Owner { scene_host* value = nullptr; ~Owner() { scene_host_destroy(value); } };
}
int main() {
    auto path = (fs::current_path() / "scene-host-tests-XXXXXX").string();
    if (!mkdtemp(path.data())) { return 1; }
    fs::path dir(path);
    try {
        // A FIFO without a writer must be rejected promptly rather than block.
        auto fifo = dir / "checkpoint.fifo";
        check(::mkfifo(fifo.c_str(), 0600) == 0, "create checkpoint FIFO");
        auto pid = ::fork();
        check(pid >= 0, "fork FIFO test");
        if (pid == 0) {
            ::alarm(3);
            scene_host* handle = nullptr;
            char fifo_error[128]{};
            auto result = scene_host_load(fifo.c_str(), &handle, fifo_error, sizeof fifo_error);
            _exit(result != 0 && !handle && std::strstr(fifo_error, "regular file") ? 0 : 1);
        }
        int status;
        pid_t waited;
        do { waited = ::waitpid(pid, &status, 0); } while (waited < 0 && errno == EINTR);
        check(waited == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0,
              "FIFO checkpoint blocked or was not rejected as non-regular");
        torch::NoGradGuard guard;
        scene_train::ResidualNet model;
        // Deterministic asymmetric weights expose channel order and clamping.
        for (auto& p : model->parameters()) { p.fill_(0.01f); }
        model->conv3->bias.copy_(torch::tensor({-0.2f, 0.1f, 0.5f}));
        model->eval();
        auto checkpoint = (dir / "model.pt").string();
        torch::save(model, checkpoint);
        Owner first, second;
        char error[512]{};
        check(scene_host_load(checkpoint.c_str(), &first.value, error, sizeof error) == 0, "load model");
        check(scene_host_load(checkpoint.c_str(), &second.value, error, sizeof error) == 0, "load second model");
        check(first.value != second.value, "independent handles");
        constexpr size_t w = 5, h = 3, in_stride = 19, out_stride = 20, row = w * 3;
        std::vector<uint8_t> input(in_stride * h, 99), output(out_stride * h, 0xa5);
        for (size_t y = 0; y < h; ++y) {
            for (size_t x = 0; x < row; ++x) { input[y * in_stride + x] = static_cast<uint8_t>((y * 47 + x * 19) % 256); }
        }
        auto expected_input = torch::empty({1, 3, h, w}, torch::kFloat32);
        auto acc = expected_input.accessor<float, 4>();
        for (size_t y = 0; y < h; ++y) {
            for (size_t x = 0; x < w; ++x) {
                for (size_t c = 0; c < 3; ++c) { acc[0][c][y][x] = input[y * in_stride + x * 3 + c] / 255.0f; }
            }
        }
        auto result = model->forward(expected_input);
        check(torch::isfinite(result).all().item<bool>(), "direct output finite");
        auto expected = result.clamp(0, 1).mul(255).round().to(torch::kUInt8).contiguous();
        auto values = expected.accessor<uint8_t, 4>();
        check(scene_host_infer(first.value, input.data(), input.size(), in_stride, output.data(), output.size(), out_stride, w, h, error, sizeof error) == 0, "inference");
        for (size_t y = 0; y < h; ++y) {
            for (size_t x = 0; x < w; ++x) {
                for (size_t c = 0; c < 3; ++c) { check(output[y * out_stride + x * 3 + c] == values[0][c][y][x], "forward/layout mismatch"); }
            }
            for (size_t x = row; x < out_stride; ++x) { check(output[y * out_stride + x] == 0xa5, "padding modified"); }
        }
        auto reference = output;
        std::fill(output.begin(), output.end(), 0xa5);
        check(scene_host_infer(second.value, input.data(), input.size(), in_stride, output.data(), output.size(), out_stride, w, h, error, sizeof error) == 0 && output == reference, "independent output");
        // In-place overlapping buffers are staged before writing.
        auto overlap = input;
        check(scene_host_infer(second.value, overlap.data(), overlap.size(), in_stride, overlap.data(), overlap.size(), in_stride, w, h, error, sizeof error) == 0, "in-place inference");
        for (size_t y = 0; y < h; ++y) { check(std::equal(overlap.begin() + y * in_stride, overlap.begin() + y * in_stride + row, reference.begin() + y * out_stride), "in-place output"); }
        auto rejects = [&](scene_host* host, const uint8_t* in, size_t is, size_t istride, uint8_t* out, size_t os, size_t ostride, uint32_t width, uint32_t height) {
            std::fill(output.begin(), output.end(), 0xa5);
            check(scene_host_infer(host, in, is, istride, out, os, ostride, width, height, error, sizeof error) != 0, "invalid inference accepted");
            check(error[0] && std::all_of(output.begin(), output.end(), [](uint8_t b) { return b == 0xa5; }), "failure must preserve output and report error");
        };
        rejects(nullptr, input.data(), input.size(), in_stride, output.data(), output.size(), out_stride, w, h);
        rejects(first.value, nullptr, input.size(), in_stride, output.data(), output.size(), out_stride, w, h);
        rejects(first.value, input.data(), input.size(), row - 1, output.data(), output.size(), out_stride, w, h);
        rejects(first.value, input.data(), input.size(), in_stride, output.data(), out_stride * (h - 1) + row - 1, out_stride, w, h);
        rejects(first.value, input.data(), input.size(), std::numeric_limits<size_t>::max(), output.data(), output.size(), out_stride, w, h);
        rejects(first.value, input.data(), input.size(), in_stride, output.data(), output.size(), out_stride, 0, h);
        rejects(first.value, input.data(), input.size(), in_stride, output.data(), output.size(), out_stride, 16385, 1);
        rejects(first.value, input.data(), input.size(), in_stride, output.data(), output.size(), out_stride, 16384, 16384);
        { std::ofstream bad(dir / "bad.pt"); bad << "not a checkpoint"; }
        scene_host* failed = first.value;
        check(scene_host_load((dir / "bad.pt").c_str(), &failed, error, sizeof error) != 0 && !failed && error[0], "malformed load");
        { std::ofstream large(dir / "large.pt", std::ios::binary); large << "x"; }
        fs::resize_file(dir / "large.pt", 1024 * 1024 + 1);
        check(scene_host_load((dir / "large.pt").c_str(), &failed, error, sizeof error) != 0 &&
              !failed && std::strstr(error, "size limit"), "oversized checkpoint rejected before parsing");
        model->conv1->weight[0][0][0][0] = std::numeric_limits<float>::quiet_NaN();
        torch::save(model, (dir / "nan.pt").string());
        check(scene_host_load((dir / "nan.pt").c_str(), &failed, error, sizeof error) != 0 && !failed, "nonfinite load");
        model->conv1->weight.fill_(std::numeric_limits<float>::infinity());
        torch::save(model, (dir / "inf.pt").string());
        check(scene_host_load((dir / "inf.pt").c_str(), &failed, error, sizeof error) != 0 && !failed, "infinite load");
        // Construct an archive with a wrong-shaped parameter through a normal model.
        scene_train::ResidualNet malformed;
        malformed->conv1->weight.set_data(torch::zeros({7, 3, 3, 3}));
        torch::save(malformed, (dir / "shape.pt").string());
        check(scene_host_load((dir / "shape.pt").c_str(), &failed, error, sizeof error) != 0 && !failed, "wrong shape load");
        malformed->conv1->weight.set_data(torch::zeros({8, 3, 3, 3}, torch::kFloat64));
        torch::save(malformed, (dir / "dtype.pt").string());
        check(scene_host_load((dir / "dtype.pt").c_str(), &failed, error, sizeof error) != 0 && !failed, "wrong dtype load");
        model->conv1->weight.fill_(1e38f);
        torch::save(model, (dir / "overflow.pt").string());
        Owner overflow;
        check(scene_host_load((dir / "overflow.pt").c_str(), &overflow.value, error, sizeof error) == 0, "finite large load");
        rejects(overflow.value, input.data(), input.size(), in_stride, output.data(), output.size(), out_stride, w, h);
        // Failed loads cannot change existing handles.
        std::fill(output.begin(), output.end(), 0xa5);
        check(scene_host_infer(first.value, input.data(), input.size(), in_stride, output.data(), output.size(), out_stride, w, h, error, sizeof error) == 0 && output == reference, "failed load affected existing handle");
        fs::remove_all(dir);
        std::cout << "host tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        fs::remove_all(dir);
        return 1;
    }
}
#endif
