#pragma once
#include <torch/torch.h>
#include <filesystem>
#include <sys/types.h>
#include <vector>
namespace scene_train {
struct Image { int width; int height; torch::Tensor pixels; };
struct Pair { std::filesystem::path cheap_path, full_path; Image cheap, full; };
// Identity of an opened input file, recorded from the open descriptor's fstat.
struct InputFile { std::string path; dev_t dev; ino_t ino; };
struct Dataset { std::vector<Pair> train, validation; std::vector<InputFile> inputs; };
Image read_ppm(const std::filesystem::path& path);
Dataset load_dataset(const std::filesystem::path& train_manifest, const std::filesystem::path& val_manifest, int patch);
}
