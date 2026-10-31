#include "model.h"
#include <torch/torch.h>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>
namespace fs = std::filesystem;
namespace {
void require(bool ok, const std::string& message) { if (!ok) { throw std::runtime_error(message); } }
struct Temp {
    fs::path path;
    Temp() { auto s = (fs::temp_directory_path() / "scene-infer-test-XXXXXX").string(); std::vector<char> name(s.begin(), s.end()); name.push_back('\0'); auto* p = ::mkdtemp(name.data()); if (!p) { throw std::runtime_error("mkdtemp failed"); } path = p; }
    ~Temp() { std::error_code ec; fs::remove_all(path, ec); }
};
void save(const fs::path& path, const std::string& bytes) { std::ofstream out(path, std::ios::binary); out.write(bytes.data(), static_cast<std::streamsize>(bytes.size())); require(static_cast<bool>(out), "fixture write failed"); }
std::string read(const fs::path& path) { std::ifstream in(path, std::ios::binary); require(static_cast<bool>(in), "cannot read output"); return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()}; }
int run(const std::string& exe, const std::vector<std::string>& args) {
    auto pid = ::fork(); if (pid < 0) { throw std::runtime_error("fork failed"); }
    if (pid == 0) { std::vector<char*> argv{const_cast<char*>(exe.c_str())}; for (const auto& arg : args) { argv.push_back(const_cast<char*>(arg.c_str())); } argv.push_back(nullptr); ::execv(exe.c_str(), argv.data()); _exit(127); }
    int status; while (::waitpid(pid, &status, 0) < 0) { if (errno != EINTR) { throw std::runtime_error("waitpid failed"); } }
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128;
}
}
int main(int argc, char** argv) {
    try {
        require(argc == 2, "expected infer executable"); torch::set_num_threads(1); torch::manual_seed(42); Temp temp;
        auto input = temp.path / "input.ppm", output = temp.path / "output.ppm", model_path = temp.path / "model.pt";
        scene_train::ResidualNet model; model->eval(); torch::save(model, model_path.string());
        std::vector<uint8_t> pixels{10, 35, 0, 255, 128, 1, 45, 200, 100, 5, 31, 90};
        auto raster = std::string(reinterpret_cast<const char*>(pixels.data()), pixels.size());
        save(input, "P6\n# synthetic\n2 2\n255#comment\r\n" + raster);
        auto cli = [&](const fs::path& in, const fs::path& out, const fs::path& ckpt) { return run(argv[1], {"--input", in.string(), "--output", out.string(), "--checkpoint", ckpt.string()}); };
        require(cli(input, output, model_path) == 0, "normal inference failed");
        torch::NoGradGuard guard;
        auto rgb = torch::from_blob(pixels.data(), {2, 2, 3}, torch::kUInt8).permute({2, 0, 1}).unsqueeze(0).to(torch::kFloat32).div(255.0);
        auto direct = model->forward(rgb).clamp(0, 1).mul(255).round().to(torch::kUInt8).squeeze(0).permute({1, 2, 0}).contiguous();
        auto expected = std::string("P6\n2 2\n255\n") + std::string(reinterpret_cast<const char*>(direct.data_ptr<uint8_t>()), pixels.size());
        require(read(output) == expected, "CLI output differs from direct model");
        // Comments ending at CR must not swallow the next token or raster.
        for (const auto& ending : {std::string("\r"), std::string("\n"), std::string("\r\n")}) {
            for (const auto& header : {std::string("#lead") + ending + "P6\n2 2\n255\n",
                                       std::string("P6#magic") + ending + "2 2\n255\n",
                                       std::string("P6\n2#width") + ending + "2\n255\n",
                                       std::string("P6\n2 2#height") + ending + "255\n"}) {
                auto variant = temp.path / "variant.ppm", result_path = temp.path / "variant-output.ppm";
                save(variant, header + raster);
                require(cli(variant, result_path, model_path) == 0 && read(result_path) == expected,
                        "header comment changed inference");
                fs::remove(result_path);
            }
        }
        // The separator counts towards the 65,536-byte header limit. An LF
        // following CR is raster data when the remaining length is exact.
        auto boundary_input = temp.path / "boundary.ppm", boundary_output = temp.path / "boundary-output.ppm";
        auto boundary_header = [](size_t length, const std::string& separator) {
            auto prefix = std::string("P6\n"), suffix = std::string("2 2\n255") + separator;
            return prefix + std::string(length - prefix.size() - suffix.size(), ' ') + suffix;
        };
        for (const auto& separator : {std::string("\n"), std::string("\r"), std::string("\r\n")}) {
            save(boundary_input, boundary_header(65536, separator) + raster);
            require(cli(boundary_input, boundary_output, model_path) == 0 && read(boundary_output) == expected,
                    "maximum-length header rejected or raster changed");
            fs::remove(boundary_output);
            save(boundary_input, boundary_header(65537, separator) + raster);
            require(cli(boundary_input, boundary_output, model_path) != 0 && !fs::exists(boundary_output),
                    "overlong header accepted");
        }
        auto original = read(input), checkpoint = read(model_path), existing = read(output);
        require(cli(input, output, model_path) != 0 && read(output) == existing, "existing output overwritten");
        require(cli(input, input, model_path) != 0 && read(input) == original, "input overwritten");
        require(cli(input, model_path, model_path) != 0 && read(model_path) == checkpoint, "checkpoint overwritten");
        auto hard = temp.path / "hard.ppm", link = temp.path / "link.ppm", dangling = temp.path / "dangling.ppm";
        fs::create_hard_link(input, hard); fs::create_symlink(input, link); fs::create_symlink(temp.path / "missing", dangling);
        for (const auto& path : {hard, link, dangling}) { require(cli(input, path, model_path) != 0, "alias output accepted"); }
        require(read(input) == original && fs::is_symlink(dangling) && !fs::exists(temp.path / "missing"), "alias changed");
        require(cli(input, temp.path, model_path) != 0 && fs::is_directory(temp.path), "directory output accepted");
        require(cli(input, temp.path / "absent" / "output.ppm", model_path) != 0 && !fs::exists(temp.path / "absent"), "missing output parent accepted");
        auto bad = temp.path / "bad.ppm", fresh = temp.path / "fresh.ppm";
        for (const auto& contents : {std::string("P3\n2 2\n255\n") + raster, std::string("P6\n2 2\n254\n") + raster, std::string("P6\n2 2\n255\n") + raster.substr(1), std::string("P6\n2 2\n255\n") + raster + "x", std::string("P6\n16385 1\n255\n"), std::string("P6\n") + std::string(65536, ' ') + "1 1\n255\nabc"}) { save(bad, contents); require(cli(bad, fresh, model_path) != 0 && !fs::exists(fresh), "malformed PPM accepted"); }
        save(bad, "P6\n#unterminated"); require(cli(bad, fresh, model_path) != 0 && !fs::exists(fresh), "unterminated header comment accepted");
        save(bad, "P6\n1 1\n255\r" + std::string("\n#x", 3)); require(cli(bad, fresh, model_path) == 0, "CR separator plus LF raster rejected"); fs::remove(fresh);
        save(bad, "not a checkpoint"); require(cli(input, fresh, bad) != 0 && !fs::exists(fresh), "bad checkpoint accepted");
        auto overflow = temp.path / "overflow.pt"; for (auto& p : model->parameters()) { p.fill_(std::numeric_limits<float>::max()); } torch::save(model, overflow.string());
        require(cli(input, fresh, overflow) != 0 && !fs::exists(fresh), "nonfinite output accepted");
        require(run(argv[1], {"--help"}) == 0 && run(argv[1], {"--help", "extra"}) != 0, "help validation");
        require(run(argv[1], {"--input", input.string(), "--input", input.string()}) != 0, "duplicate option accepted");
        std::cout << "scene-infer tests passed\n"; return 0;
    } catch (const std::exception& e) { std::cerr << "test failure: " << e.what() << '\n'; return 1; }
}
