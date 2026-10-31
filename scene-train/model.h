#pragma once
#include <torch/torch.h>

namespace scene_train {
// This fixed definition is also the native C++ checkpoint inference interface.
struct ResidualNetImpl : torch::nn::Module {
    ResidualNetImpl();
    torch::Tensor forward(const torch::Tensor& rgb);
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
};
TORCH_MODULE(ResidualNet);
}
