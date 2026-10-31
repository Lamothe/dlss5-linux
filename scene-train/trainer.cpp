#include "trainer.h"
#include "checkpoint.h"
#include <algorithm>
#include <cerrno>
#include <system_error>
#include <sys/stat.h>
#include <unistd.h>
#include <ATen/Context.h>
#include <cmath>
#include <iostream>
#include <mutex>
#include <random>
#include <stdexcept>
namespace scene_train {
namespace {
Validation validate(ResidualNet& model, const std::vector<Pair>& data, const torch::Device& device) {
    model->eval(); torch::NoGradGuard no_grad;
    double float_sum = 0, deployed_sum = 0;
    for (const auto& pair : data) {
        auto target = pair.full.pixels.unsqueeze(0).to(device);
        auto prediction = model->forward(pair.cheap.pixels.unsqueeze(0).to(device));
        auto float_loss = torch::mse_loss(prediction, target).item<double>();
        // Deployed criterion: exactly the host inference conversion (host.cpp),
        // so checkpoint selection sees what inference will actually produce.
        auto quantised = prediction.clamp(0, 1).mul(255).round().to(torch::kUInt8).to(torch::kFloat32).div_(255.0);
        auto deployed_loss = torch::mse_loss(quantised, target).item<double>();
        if (!std::isfinite(float_loss) || !std::isfinite(deployed_loss)) { throw std::runtime_error("non-finite validation loss"); }
        float_sum += float_loss;
        deployed_sum += deployed_loss;
    }
    return {float_sum / static_cast<double>(data.size()), deployed_sum / static_cast<double>(data.size())};
}
// Refuse to run when a log stream names an input file: the trainer's own log
// writes would corrupt that input. Identity comes from fstat against the open
// input descriptors recorded at load time.
void check_log_targets(const Dataset& dataset) {
    bool out_aliased = false, err_aliased = false;
    for (int fd : {STDOUT_FILENO, STDERR_FILENO}) {
        struct stat st{};
        if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) { continue; }
        for (const auto& input : dataset.inputs) {
            if (st.st_dev == input.dev && st.st_ino == input.ino) {
                (fd == STDOUT_FILENO ? out_aliased : err_aliased) = true;
            }
        }
    }
    if (out_aliased || err_aliased) { throw LogAliasError(out_aliased, err_aliased); }
}
}
void check_parameters(ResidualNet& model) {
    for (const auto& p : model->parameters()) {
        if (!torch::isfinite(p).all().item<bool>()) { throw std::runtime_error("non-finite model parameter"); }
    }
}
torch::Tensor patch_forward(ResidualNet& model, const torch::Tensor& input, int x, int y, int patch, const torch::Device& device) {
    constexpr int halo = 3; // Three 3x3 convolutions, each with radius one.
    auto left = std::max(0, x - halo), top = std::max(0, y - halo);
    auto right = std::min<int64_t>(input.size(2), static_cast<int64_t>(x) + patch + halo);
    auto bottom = std::min<int64_t>(input.size(1), static_cast<int64_t>(y) + patch + halo);
    auto context = input.slice(1, top, bottom).slice(2, left, right).unsqueeze(0).to(device);
    return model->forward(context).slice(2, y - top, y - top + patch).slice(3, x - left, x - left + patch);
}
Result train(const Options& o) {
    if (o.epochs <= 0 || o.patch <= 0 || o.patches_per_image <= 0 || !std::isfinite(o.learning_rate) || o.learning_rate <= 0 || o.seed > static_cast<uint64_t>(INT64_MAX)) {
        throw std::runtime_error("invalid training options");
    }
    if (o.device != "cpu" && o.device != "cuda") { throw std::runtime_error("device must be cpu or cuda"); }
    // Validate every image before creating any output or initialising a device.
    auto dataset = load_dataset(o.train_manifest, o.val_manifest, o.patch);
    // Refuse before any log write if a log stream aliases an input file.
    check_log_targets(dataset);
    CheckpointFile checkpoint(o.checkpoint);
    torch::set_num_threads(1);
    at::globalContext().setDeterministicAlgorithms(true, false);
    at::globalContext().setBenchmarkCuDNN(false);
    { auto gen = at::globalContext().defaultGenerator(c10::DeviceType::CPU);
      std::lock_guard<std::mutex> lock(gen.mutex()); gen.set_current_seed(o.seed); }
    if (o.device == "cuda" && !torch::cuda::is_available()) { throw std::runtime_error("CUDA unavailable"); }
    torch::Device device(o.device == "cuda" ? torch::kCUDA : torch::kCPU);
    ResidualNet model;
    model->to(device);
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(o.learning_rate));
    std::mt19937_64 rng(o.seed);
    auto initial = validate(model, dataset.validation, device);
    Result result{initial, initial, 0};
    // Copy rather than moving the optimiser's live parameters between devices.
    auto save = [&] {
        check_parameters(model);
        checkpoint.replace([&](int fd) {
            auto write = [fd](const void* data, size_t size) -> size_t {
                auto bytes = static_cast<const char*>(data);
                size_t done = 0;
                while (done < size) {
                    auto n = ::write(fd, bytes + done, size - done);
                    if (n < 0 && errno == EINTR) { continue; }
                    if (n <= 0) { if (n == 0) { errno = EIO; } throw std::system_error(errno, std::generic_category(), "write checkpoint"); }
                    done += static_cast<size_t>(n);
                }
                return done;
            };
            if (device.is_cpu()) { torch::save(model, std::function<size_t(const void*, size_t)>(write)); return; }
            torch::NoGradGuard guard;
            ResidualNet cpu_model;
            auto source = model->parameters(); auto target = cpu_model->parameters();
            for (size_t i = 0; i < source.size(); ++i) { target[i].copy_(source[i].to(torch::kCPU)); }
            torch::save(cpu_model, std::function<size_t(const void*, size_t)>(write));
        });
    };
    save();
    std::cout << "scene-train: neural rendering demonstration, not DLSS5\n";
    std::cout << "epoch=0 validation_mse=" << result.initial_validation.float_mse
              << " deployed_mse=" << result.initial_validation.deployed_mse << " (initial)\n";
    for (int64_t epoch = 1; epoch <= o.epochs; ++epoch) {
        model->train(); double total = 0; uint64_t steps = 0;
        for (const auto& pair : dataset.train) {
            std::uniform_int_distribution<int> x_dist(0, pair.cheap.width - o.patch), y_dist(0, pair.cheap.height - o.patch);
            for (int i = 0; i < o.patches_per_image; ++i) {
                auto x = x_dist(rng), y = y_dist(rng);
                auto crop = [&](const torch::Tensor& t) { return t.slice(1, y, y + o.patch).slice(2, x, x + o.patch).unsqueeze(0).to(device); };
                optimizer.zero_grad();
                auto loss = torch::mse_loss(patch_forward(model, pair.cheap.pixels, x, y, o.patch, device), crop(pair.full.pixels));
                double value = loss.item<double>();
                if (!std::isfinite(value)) { throw std::runtime_error("non-finite training loss"); }
                loss.backward();
                for (const auto& p : model->parameters()) {
                    if (!p.grad().defined() || !torch::isfinite(p.grad()).all().item<bool>()) { throw std::runtime_error("missing or non-finite gradient"); }
                }
                optimizer.step();
                check_parameters(model);
                total += value; ++steps;
            }
        }
        auto val = validate(model, dataset.validation, device);
        std::cout << "epoch=" << epoch << " train_mse=" << total / static_cast<double>(steps)
                  << " validation_mse=" << val.float_mse << " deployed_mse=" << val.deployed_mse << '\n';
        // Checkpoint selection uses the deployed criterion; ties keep the earlier checkpoint.
        if (val.deployed_mse < result.best_validation.deployed_mse) { result.best_validation = val; result.best_epoch = static_cast<int>(epoch); save(); }
    }
    std::cout << "best_epoch=" << result.best_epoch << " best_validation_mse=" << result.best_validation.float_mse
              << " best_deployed_mse=" << result.best_validation.deployed_mse << '\n';
    return result;
}
}
