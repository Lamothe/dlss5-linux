#include "model.h"
namespace scene_train {
ResidualNetImpl::ResidualNetImpl() {
    conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 8, 3).padding(1)));
    conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(8, 8, 3).padding(1)));
    conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(8, 3, 3).padding(1)));
    torch::NoGradGuard guard;
    conv3->weight.zero_();
    conv3->bias.zero_();
}
torch::Tensor ResidualNetImpl::forward(const torch::Tensor& rgb) {
    return rgb + conv3->forward(torch::relu(conv2->forward(torch::relu(conv1->forward(rgb)))));
}
}
