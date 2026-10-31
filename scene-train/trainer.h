#pragma once
#include "data.h"
#include "model.h"
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
namespace scene_train {
struct Options {
    std::filesystem::path train_manifest, val_manifest, checkpoint;
    std::string device = "cpu";
    int epochs = 10, patch = 32, patches_per_image = 4;
    double learning_rate = 0.001;
    uint64_t seed = 1;
};
// Both validation criteria are reported; checkpoint selection uses deployed_mse.
struct Validation {
    // Unclamped float RGB MSE, matching training's objective.
    double float_mse;
    // RGB MSE after the host inference conversion (clamp [0,1], x255, round to
    // nearest ties-to-even, quantise to 8-bit), matching deployed output.
    double deployed_mse;
};
struct Result { Validation initial_validation; Validation best_validation; int best_epoch; };
// Thrown when stdout or stderr names (aliases) an input file; running would
// corrupt that input. Carries which streams are aliased so the caller can
// report through a safe sink only.
struct LogAliasError : std::runtime_error {
    bool stdout_aliased;
    bool stderr_aliased;
    LogAliasError(bool out, bool err)
        : std::runtime_error("stdout/stderr name an input file; refusing to run and corrupt it"), stdout_aliased(out), stderr_aliased(err) {}
};
// Input is CHW; returns the supervised NCHW patch with full-image context.
torch::Tensor patch_forward(ResidualNet& model, const torch::Tensor& input, int x, int y, int patch, const torch::Device& device);
void check_parameters(ResidualNet& model);
Result train(const Options& options);
}
