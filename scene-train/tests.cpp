#include "trainer.h"
#include "checkpoint.h"
#include <algorithm>
#include <fcntl.h>
#include <ATen/Context.h>
#include <mutex>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <unistd.h>
namespace fs = std::filesystem;
using namespace scene_train;
namespace {
void check(bool ok, const std::string& what) { if (!ok) { throw std::runtime_error(what); } }
template<class F> void rejects(F f, const std::string& what) {
    try { f(); } catch (const std::exception&) { return; }
    throw std::runtime_error("did not reject " + what);
}
void ppm(const fs::path& p, int w, int h, unsigned char value) {
    std::ofstream f(p, std::ios::binary); f << "P6\n" << w << ' ' << h << "\n255\n";
    std::string pixels(static_cast<size_t>(w * h * 3), 0);
    for (int y = 0; y < h; ++y) { for (int x = 0; x < w; ++x) { for (int c = 0; c < 3; ++c) { pixels[static_cast<size_t>((y * w + x) * 3 + c)] = static_cast<char>(value + x + 2 * y + c); } } }
    f.write(pixels.data(), pixels.size());
}
void text(const fs::path& p, const std::string& s) { std::ofstream f(p); f << s; }
}
int main() {
    auto base = fs::current_path() / "scene-train-tests-XXXXXX";
    std::string name = base.string();
    if (!mkdtemp(name.data())) { std::cerr << "cannot create test directory\n"; return 1; }
    fs::path dir(name);
    try {
        ppm(dir / "tc.ppm", 8, 8, 50); ppm(dir / "tf.ppm", 8, 8, 100);
        ppm(dir / "vc.ppm", 8, 8, 50); ppm(dir / "vf.ppm", 8, 8, 100);
        text(dir / "train.tsv", "tc.ppm\ttf.ppm\n"); text(dir / "val.tsv", "vc.ppm\tvf.ppm\n");
        // Exercise atomic reservation, symlink collision and interrupted updates.
        text(dir / "collision.pt", "keep");
        rejects([&] { CheckpointFile c(dir / "collision.pt"); }, "existing checkpoint reservation");
        { std::ifstream f(dir / "collision.pt"); std::string value; f >> value; check(value == "keep", "collision preserves existing file"); }
        fs::create_symlink("collision.pt", dir / "existing-link.pt");
        rejects([&] { CheckpointFile c(dir / "existing-link.pt"); }, "existing checkpoint symlink");
        { std::ifstream f(dir / "collision.pt"); std::string value; f >> value; check(value == "keep", "symlink target untouched"); }
        {
            CheckpointFile first(dir / "reserved.pt");
            rejects([&] { CheckpointFile second(dir / "reserved.pt"); }, "competing checkpoint reservation");
        }
        check(!fs::exists(dir / "reserved.pt"), "unused reservation cleaned");
        fs::create_symlink("missing-target", dir / "dangling.pt");
        rejects([&] { CheckpointFile c(dir / "dangling.pt"); }, "dangling checkpoint symlink");
        check(fs::is_symlink(dir / "dangling.pt") && !fs::exists(dir / "missing-target"), "dangling symlink untouched");
        {
            CheckpointFile c(dir / "atomic.pt");
            auto put = [](int fd, const char* value) {
                std::string str(value);
                if (write(fd, str.data(), str.size()) != static_cast<ssize_t>(str.size())) { throw std::runtime_error("test write"); }
            };
            c.replace([&](int fd) { put(fd, "good"); });
            rejects([&] { c.replace([&](int fd) { put(fd, "bad"); throw std::runtime_error("injected failure"); }); }, "failed checkpoint write");
            { std::ifstream f(dir / "atomic.pt"); std::string value; f >> value; check(value == "good", "failed update preserves good checkpoint"); }
            fs::rename(dir / "atomic.pt", dir / "saved.pt");
            fs::create_directory(dir / "atomic.pt");
            rejects([&] { c.replace([&](int fd) { put(fd, "bad"); }); }, "failed checkpoint rename");
            { std::ifstream f(dir / "saved.pt"); std::string value; f >> value; check(value == "good", "failed rename preserves previous checkpoint"); }
        }
        for (const auto& entry : fs::directory_iterator(dir)) {
            check(entry.path().filename().string().find(".scene-train-") != 0, "temporary checkpoint cleaned");
        }
        {
            CheckpointFile c(dir / "unused.pt");
            rejects([&] { c.replace([](int) { throw std::runtime_error("injected first failure"); }); }, "first save failure");
        }
        check(!fs::exists(dir / "unused.pt"), "failed initial reservation cleaned");
        // Comments before each header token may end at CR, LF or CRLF.
        for (const auto& ending : {std::string("\r"), std::string("\n"), std::string("\r\n")}) {
            for (const auto& header : {std::string("#lead") + ending + "P6\n1 1\n255\n",
                                       std::string("P6#magic") + ending + "1 1\n255\n",
                                       std::string("P6\n1#width") + ending + "1\n255\n",
                                       std::string("P6\n1 1#height") + ending + "255\n"}) {
                text(dir / "header-comment.ppm", header + std::string("# \n", 3));
                auto pixels = read_ppm(dir / "header-comment.ppm").pixels;
                check(std::abs(pixels[0][0][0].item<float>() - 35.0f / 255.0f) < 1e-7f &&
                      std::abs(pixels[1][0][0].item<float>() - 32.0f / 255.0f) < 1e-7f &&
                      std::abs(pixels[2][0][0].item<float>() - 10.0f / 255.0f) < 1e-7f,
                      "header comment preserves tokens and raster");
            }
        }
        text(dir / "header-comment.ppm", "P6\n#unterminated");
        rejects([&] { read_ppm(dir / "header-comment.ppm"); }, "unterminated header comment");
        // CR-only separator with LF as raster byte and genuine CRLF headers.
        text(dir / "cr.ppm", std::string("P6\n1 1\n255\r") + std::string("\nAB", 3));
        text(dir / "crlf.ppm", std::string("P6\n1 1\n255\r\n") + std::string("\nAB", 3));
        auto cr = read_ppm(dir / "cr.ppm").pixels;
        check(torch::equal(cr, read_ppm(dir / "crlf.ppm").pixels), "CR and CRLF preserve LF first raster byte");
        check(std::abs(cr[0][0][0].item<float>() - 10.0f / 255.0f) < 1e-7f, "first raster byte decoded");
        text(dir / "crbad.ppm", std::string("P6\n1 1\n255\r\n") + std::string("ABCD", 4));
        rejects([&] { read_ppm(dir / "crbad.ppm"); }, "bad CR raster length");
        // A comment adjacent to maxval ends before CR or LF; once the
        // separator is consumed, even '#' and whitespace are raster bytes.
        for (const auto& header : {std::string("P6\n1 1\n255#comment\n"),
                                   std::string("P6\n1 1\n255#comment\r"),
                                   std::string("P6\n1 1\n255#comment\r\n")}) {
            text(dir / "comment.ppm", header + std::string("# \n", 3));
            auto pixels = read_ppm(dir / "comment.ppm").pixels;
            check(std::abs(pixels[0][0][0].item<float>() - 35.0f / 255.0f) < 1e-7f &&
                  std::abs(pixels[1][0][0].item<float>() - 32.0f / 255.0f) < 1e-7f &&
                  std::abs(pixels[2][0][0].item<float>() - 10.0f / 255.0f) < 1e-7f,
                  "comment separator preserves raster bytes");
        }
        text(dir / "comment-cr.ppm", std::string("P6\n1 1\n255#comment\r") + std::string("\nAB", 3));
        text(dir / "comment-crlf.ppm", std::string("P6\n1 1\n255#comment\r\n") + std::string("\nAB", 3));
        check(torch::equal(read_ppm(dir / "comment-cr.ppm").pixels, read_ppm(dir / "comment-crlf.ppm").pixels) &&
              torch::equal(cr, read_ppm(dir / "comment-cr.ppm").pixels), "comment CR/CRLF preserve LF raster byte");
        text(dir / "comment.ppm", std::string("P6\n1 1\n255#comment\n") + "AB");
        rejects([&] { read_ppm(dir / "comment.ppm"); }, "comment followed by truncated raster");
        text(dir / "comment.ppm", std::string("P6\n1 1\n255#comment\n") + "ABCD");
        rejects([&] { read_ppm(dir / "comment.ppm"); }, "comment followed by extra raster byte");
        text(dir / "comment.ppm", "P6\n1 1\n255#unterminated");
        rejects([&] { read_ppm(dir / "comment.ppm"); }, "unterminated maxval comment");
        text(dir / "comment.ppm", std::string("P6\n1 1\n255 ") + "#AB");
        check(std::abs(read_ppm(dir / "comment.ppm").pixels[0][0][0].item<float>() - 35.0f / 255.0f) < 1e-7f,
              "hash after separator is raster");
        auto data = load_dataset(dir / "train.tsv", dir / "val.tsv", 8);
        check(data.train.size() == 1 && data.validation.size() == 1, "dataset sizes");
        { auto gen = at::globalContext().defaultGenerator(c10::DeviceType::CPU); std::lock_guard<std::mutex> lock(gen.mutex()); gen.set_current_seed(5); }
        // Compare patch outputs and parameter gradients against full-frame
        // supervision for corners, edges and an interior crop.
        ResidualNet context_model;
        // Exercise all three convolutions even though new models start as identity.
        { torch::NoGradGuard guard; context_model->conv3->weight.fill_(0.03f); context_model->conv3->bias.fill_(0.01f); }
        auto image = torch::rand({3, 17, 19});
        auto target = torch::rand({1, 3, 4, 4});
        for (const auto& xy : {std::pair{0, 0}, std::pair{15, 0}, std::pair{0, 13}, std::pair{15, 13}, std::pair{0, 6}, std::pair{8, 0}, std::pair{7, 6}, std::pair{1, 2}, std::pair{14, 12}}) {
            auto x = xy.first, y = xy.second;
            context_model->zero_grad();
            auto full = context_model->forward(image.unsqueeze(0)).slice(2, y, y + 4).slice(3, x, x + 4);
            torch::mse_loss(full, target).backward();
            std::vector<torch::Tensor> grads;
            for (const auto& p : context_model->parameters()) { grads.push_back(p.grad().clone()); }
            context_model->zero_grad();
            auto patch = patch_forward(context_model, image, x, y, 4, torch::kCPU);
            check(torch::allclose(full, patch, 1e-5, 1e-6), "halo patch output matches full image");
            torch::mse_loss(patch, target).backward();
            auto params = context_model->parameters();
            for (size_t i = 0; i < params.size(); ++i) {
                check(torch::allclose(grads[i], params[i].grad(), 1e-5, 1e-6), "halo patch gradient matches full image");
            }
        }
        ResidualNet initial;
        check(initial->conv3->weight.count_nonzero().item<int64_t>() == 0 &&
              initial->conv3->bias.count_nonzero().item<int64_t>() == 0, "zero final convolution");
        // All pixels, including corners and borders, are compared before clamping.
        for (const auto& shape : {std::pair{1, 1}, std::pair{1, 9}, std::pair{7, 1}, std::pair{8, 11}}) {
            for (const auto& image : {torch::zeros({2, 3, shape.first, shape.second}),
                                      torch::ones({2, 3, shape.first, shape.second}),
                                      torch::rand({2, 3, shape.first, shape.second}) * 4 - 2}) {
                check(torch::equal(initial->forward(image), image), "initial identity including borders");
            }
        }
        // Archives with nonzero final weights must overwrite constructor defaults.
        {
            ResidualNet old_model;
            { torch::NoGradGuard guard; old_model->conv3->weight.fill_(0.125f); old_model->conv3->bias.fill_(-0.25f); }
            torch::save(old_model, (dir / "old-format.pt").string());
            ResidualNet loaded; torch::load(loaded, (dir / "old-format.pt").string());
            auto old_params = old_model->named_parameters(), loaded_params = loaded->named_parameters();
            check(old_params.size() == loaded_params.size(), "checkpoint parameter count");
            for (const auto& entry : old_params) {
                check(loaded_params.contains(entry.key()) && torch::equal(entry.value(), loaded_params[entry.key()]),
                      "checkpoint restores named parameter values");
            }
            auto sample = torch::rand({1, 3, 5, 6});
            check(torch::equal(old_model->forward(sample), loaded->forward(sample)), "nonzero checkpoint output preserved");
        }
        // With a zero output layer, earlier layers initially have zero gradients;
        // once it updates, gradients and updates must reach them too.
        {
            torch::NoGradGuard guard;
            initial->conv1->weight.fill_(0.05f); initial->conv1->bias.fill_(0.05f);
            initial->conv2->weight.fill_(0.05f); initial->conv2->bias.fill_(0.05f);
        }
        auto learning_input = torch::ones({1, 3, 4, 5});
        auto learning_target = learning_input + 0.25f;
        torch::optim::Adam learning_optimizer(initial->parameters(), torch::optim::AdamOptions(0.01));
        auto before = initial->conv3->weight.clone(), before_bias = initial->conv3->bias.clone();
        torch::mse_loss(initial->forward(learning_input), learning_target).backward();
        check(initial->conv3->weight.grad().abs().sum().item<double>() > 0 &&
              initial->conv3->bias.grad().abs().sum().item<double>() > 0, "output layer gradients");
        check(initial->conv1->weight.grad().count_nonzero().item<int64_t>() == 0 &&
              initial->conv2->weight.grad().count_nonzero().item<int64_t>() == 0, "initial hidden gradients are zero");
        learning_optimizer.step();
        check(!torch::equal(before, initial->conv3->weight) && !torch::equal(before_bias, initial->conv3->bias), "output layer updates");
        learning_optimizer.zero_grad();
        auto before1 = initial->conv1->weight.clone(), before2 = initial->conv2->weight.clone();
        torch::mse_loss(initial->forward(learning_input), learning_target).backward();
        for (const auto& p : initial->parameters()) {
            check(p.grad().defined() && torch::isfinite(p.grad()).all().item<bool>() &&
                  p.grad().abs().sum().item<double>() > 0, "finite nonzero gradients after first update");
        }
        learning_optimizer.step();
        check(!torch::equal(before1, initial->conv1->weight) && !torch::equal(before2, initial->conv2->weight),
              "hidden layers update after output layer");
        auto input = data.train[0].cheap.pixels.unsqueeze(0);
        Options o; o.train_manifest = dir / "train.tsv"; o.val_manifest = dir / "val.tsv";
        o.checkpoint = dir / "one.pt"; o.epochs = 20; o.patch = 6; o.patches_per_image = 1; o.learning_rate = 0.01; o.seed = 5;
        auto r = train(o);
        auto baseline = torch::mse_loss(data.validation[0].cheap.pixels, data.validation[0].full.pixels).item<double>();
        check(r.initial_validation.float_mse == baseline, "epoch zero validation equals float identity baseline");
        check(r.initial_validation.deployed_mse == baseline, "epoch zero deployed validation equals identity baseline");
        check(r.best_epoch > 0 && r.best_validation.deployed_mse < r.initial_validation.deployed_mse * 0.5,
              "bounded learning must improve deployed validation");
        ResidualNet m1, m2; torch::load(m1, o.checkpoint.string()); torch::load(m2, o.checkpoint.string());
        m1->eval(); m2->eval();
        auto out1 = m1->forward(input), out2 = m2->forward(input);
        check(torch::equal(out1, out2), "checkpoint reload equality");
        check(m1->conv3->weight.count_nonzero().item<int64_t>() > 0 &&
              torch::equal(m1->conv3->weight, m2->conv3->weight), "load retains nonzero checkpoint output weights");
        {
            // Selection is the deployed criterion: the quantised checkpoint output
            // must reproduce the selected deployed validation loss exactly.
            auto deployed_out = out1.clamp(0, 1).mul(255).round().to(torch::kUInt8).to(torch::kFloat32).div_(255.0);
            auto deployed = torch::mse_loss(deployed_out, data.validation[0].full.pixels.unsqueeze(0)).item<double>();
            check(std::abs(deployed - r.best_validation.deployed_mse) < 1e-8, "checkpoint corresponds to selected validation epoch (deployed criterion)");
        }
        o.checkpoint = dir / "two.pt"; auto r2 = train(o);
        ResidualNet repeat; torch::load(repeat, o.checkpoint.string()); repeat->eval();
        check(r.best_epoch == r2.best_epoch && r.best_validation.float_mse == r2.best_validation.float_mse &&
              r.best_validation.deployed_mse == r2.best_validation.deployed_mse &&
              torch::equal(out1, repeat->forward(input)), "seeded training repeatability");
        // An identity validation target keeps epoch zero on ties or regression.
        ppm(dir / "same-val.ppm", 8, 8, 50);
        text(dir / "identity-val.tsv", "vc.ppm\tsame-val.ppm\n");
        o.val_manifest = dir / "identity-val.tsv"; o.checkpoint = dir / "identity.pt"; o.epochs = 1;
        auto identity_result = train(o);
        check(identity_result.initial_validation.float_mse == 0 && identity_result.initial_validation.deployed_mse == 0 &&
              identity_result.best_validation.float_mse == 0 && identity_result.best_validation.deployed_mse == 0 &&
              identity_result.best_epoch == 0, "identity baseline retained as epoch zero checkpoint");
        ResidualNet identity_checkpoint; torch::load(identity_checkpoint, o.checkpoint.string());
        check(torch::equal(identity_checkpoint->forward(input), input), "epoch zero checkpoint restores identity");
        o.val_manifest = dir / "val.tsv"; o.checkpoint = dir / "two.pt"; o.epochs = 20;
        // Finite parameters and gradients can still overflow an optimiser step.
        ResidualNet explosive;
        {
            torch::NoGradGuard guard;
            for (auto& p : explosive->parameters()) {
                p.fill_(3.4e38f);
                p.mutable_grad() = torch::full_like(p, -1.0f);
            }
        }
        check_parameters(explosive);
        torch::optim::Adam explosive_optimizer(explosive->parameters(), torch::optim::AdamOptions(1e37));
        explosive_optimizer.step();
        rejects([&] { check_parameters(explosive); }, "non-finite parameters after optimizer step");
        rejects([&] { train(o); }, "existing checkpoint");
        o.checkpoint = dir / "dangling.pt";
        rejects([&] { train(o); }, "dangling checkpoint through trainer");
        check(fs::is_symlink(o.checkpoint) && !fs::exists(dir / "missing-target"), "trainer leaves dangling symlink unchanged");
        ppm(dir / "vf2.ppm", 8, 8, 110); text(dir / "val2.tsv", "vc.ppm\tvf2.ppm\n");
        o.epochs = 1; o.val_manifest = dir / "val.tsv"; o.checkpoint = dir / "iso1.pt";
        auto a = train(o);
        o.val_manifest = dir / "val2.tsv"; o.checkpoint = dir / "iso2.pt"; auto b = train(o);
        check(a.best_epoch == 1 && b.best_epoch == 1, "isolation fixtures must select epoch 1");
        ResidualNet iso1, iso2; torch::load(iso1, (dir / "iso1.pt").string()); torch::load(iso2, (dir / "iso2.pt").string());
        check(torch::equal(iso1->forward(input), iso2->forward(input)), "validation labels must not affect training gradients");
        text(dir / "overlap.tsv", "tc.ppm\tvf.ppm\n");
        rejects([&] { load_dataset(dir / "train.tsv", dir / "overlap.tsv", 8); }, "train/validation overlap");
        fs::create_hard_link(dir / "tc.ppm", dir / "alias.ppm"); text(dir / "overlap.tsv", "alias.ppm\tvf.ppm\n");
        rejects([&] { load_dataset(dir / "train.tsv", dir / "overlap.tsv", 8); }, "hard link overlap");
        rejects([&] { load_dataset(dir / "train.tsv", dir / "train.tsv", 8); }, "same manifest");
        ppm(dir / "small.ppm", 7, 8, 100); text(dir / "bad.tsv", "vc.ppm\tsmall.ppm\n");
        rejects([&] { load_dataset(dir / "train.tsv", dir / "bad.tsv", 8); }, "mismatched dimensions");
        text(dir / "bad.tsv", "vc.ppm vf.ppm\n"); rejects([&] { load_dataset(dir / "train.tsv", dir / "bad.tsv", 8); }, "malformed manifest");
        text(dir / "bad.ppm", "P6\n8 8\n255\nshort"); rejects([&] { read_ppm(dir / "bad.ppm"); }, "truncated PPM");
        text(dir / "bad.ppm", "P3\n1 1\n255\n0 0 0\n"); rejects([&] { read_ppm(dir / "bad.ppm"); }, "unsupported PPM");
        ppm(dir / "bad.ppm", 1, 1, 0); { std::ofstream f(dir / "bad.ppm", std::ios::app | std::ios::binary); f << 'x'; }
        rejects([&] { read_ppm(dir / "bad.ppm"); }, "trailing raster data");
        rejects([&] { load_dataset(dir / "train.tsv", dir / "val.tsv", 9); }, "oversized patch");
        // A symlink alias is the same file identity and must be rejected.
        fs::create_symlink("tc.ppm", dir / "symalias.ppm");
        text(dir / "sym-overlap.tsv", "symalias.ppm\tvf.ppm\n");
        rejects([&] { load_dataset(dir / "train.tsv", dir / "sym-overlap.tsv", 8); }, "symlink alias overlap");
        // An image that aliases a manifest is also input-file reuse.
        text(dir / "man-overlap.tsv", "train.tsv\tvf.ppm\n");
        rejects([&] { load_dataset(dir / "train.tsv", dir / "man-overlap.tsv", 8); }, "manifest alias as image");
        // A log stream naming an input must refuse before any write corrupts it.
        {
            std::ifstream before(dir / "tc.ppm", std::ios::binary);
            std::string before_bytes((std::istreambuf_iterator<char>(before)), std::istreambuf_iterator<char>());
            Options alias; alias.train_manifest = dir / "train.tsv"; alias.val_manifest = dir / "val.tsv";
            alias.checkpoint = dir / "alias-run.pt"; alias.epochs = 1; alias.patch = 6; alias.patches_per_image = 1;
            int saved = dup(STDOUT_FILENO);
            int target = open((dir / "tc.ppm").c_str(), O_WRONLY | O_CLOEXEC);
            check(saved >= 0 && target >= 0 && dup2(target, STDOUT_FILENO) >= 0, "alias test setup");
            close(target);
            bool refused = false;
            try { train(alias); }
            catch (const LogAliasError& e) { refused = e.stdout_aliased && !e.stderr_aliased; }
            dup2(saved, STDOUT_FILENO); close(saved);
            check(refused, "trainer refuses stdout aliased to an input");
            check(!fs::exists(dir / "alias-run.pt"), "alias refusal creates no checkpoint");
            std::ifstream after(dir / "tc.ppm", std::ios::binary);
            std::string after_bytes((std::istreambuf_iterator<char>(after)), std::istreambuf_iterator<char>());
            check(before_bytes == after_bytes, "aliased input file untouched");
        }
        fs::remove_all(dir); std::cout << "all tests passed\n"; return 0;
    } catch (const std::exception& e) { fs::remove_all(dir); std::cerr << "test failure: " << e.what() << '\n'; return 1; }
}
