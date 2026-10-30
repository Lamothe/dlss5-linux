// The EDSR-base x2 runtime smoke (real open weights, torch-save
// container): loader verification, determinism, finiteness, and the
// analytic smooth-gradient implementation check. NOT DLSS5 - a
// demonstration transform with real weights.
//
// Usage: native-demo-edsr-smoke <weights.pt> [w] [h] [threads]

#include <chrono>
#include <cstdio>
#include <fstream>
#include <cstdlib>

#include "native-demo-upscale.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: %s <weights.pt> [w] [h] [threads]\n", argv[0]);
        return 2;
    }
    const int w = argc > 2 ? std::atoi(argv[2]) : 64;
    const int h = argc > 3 ? std::atoi(argv[3]) : 48;
    const int threads = argc > 4 ? std::atoi(argv[4]) : 8;
    using namespace dlss5_nr::demoup;

    if (!demo_model_load(argv[1])) {
        std::printf("FAIL load: %s\n", demo_model_error().c_str());
        return 1;
    }
    std::printf("loaded: kind=%d label=%s\n", demo_model_kind(),
                demo_model_label());
    if (demo_model_kind() != 2) {
        std::printf("FAIL: not the EDSR container (kind=%d)\n",
                    demo_model_kind());
        return 1;
    }

    // Determinism + finiteness through the public transform.
    // PLANAR input (the documented contract: 3 planes of w*h).
    std::vector<float> in(3 * w * h), o1(3 * 4 * w * h), o2(3 * 4 * w * h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t i = static_cast<size_t>(y) * w + x;
            in[0 * w * h + i] = x / static_cast<float>(w);
            in[1 * w * h + i] = y / static_cast<float>(h);
            in[2 * w * h + i] = 0.5f;
        }
    }
    DemoUpscaleParams p;
    p.scale = 2;
    p.kernel = DemoKernel::Neural;
    p.threads = threads;
    auto t0 = std::chrono::steady_clock::now();
    if (!demo_upscale(in.data(), w, h, o1.data(), 2 * w, 2 * h, p)) {
        std::printf("FAIL upscale\n");
        return 1;
    }
    auto t1 = std::chrono::steady_clock::now();
    if (!demo_upscale(in.data(), w, h, o2.data(), 2 * w, 2 * h, p)) {
        std::printf("FAIL upscale (second)\n");
        return 1;
    }
    for (size_t i = 0; i < o1.size(); ++i) {
        if (o1[i] != o2[i]) {
            std::printf("FAIL determinism at %zu\n", i);
            return 1;
        }
        if (!std::isfinite(o1[i])) {
            std::printf("FAIL non-finite at %zu\n", i);
            return 1;
        }
    }
    std::printf("forward %dx%d: %.1f ms (%d threads), deterministic\n", w, h,
                std::chrono::duration<double, std::milli>(t1 - t0).count(),
                threads);

    // The analytic implementation check: a smooth gradient whose x2
    // upscale is computable. A correct implementation tracks it closely
    // (the measured ~30 dB is a model-perturbation level, not a quality
    // claim); a broken sub-pixel shuffle would score under 20 dB.
    auto fx = [](float u) { return 0.15f + 0.7f * u; };
    auto fy = [](float v) { return 0.25f + 0.5f * v * v; };
    std::vector<float> r(w * h), g(w * h), b(w * h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            r[y * w + x] = fx((x + 0.5f) / w);
            g[y * w + x] = fy((y + 0.5f) / h);
            b[y * w + x] = 0.4f;
        }
    }
    auto& m = demo_model_edsr();
    std::vector<float> orr(4 * w * h), og(4 * w * h), ob(4 * w * h);
    m.forward_x2(r.data(), g.data(), b.data(), w, h, orr.data(), og.data(),
                 ob.data(), threads);
    double mse = 0;
    for (int y = 0; y < 2 * h; ++y) {
        for (int x = 0; x < 2 * w; ++x) {
            const double dr = orr[y * 2 * w + x] - fx((x + 0.5f) / (2 * w));
            const double dg = og[y * 2 * w + x] - fy((y + 0.5f) / (2 * h));
            const double db = ob[y * 2 * w + x] - 0.4;
            mse += dr * dr + dg * dg + db * db;
        }
    }
    mse /= 3.0 * 4 * w * h;
    const double psnr = 10.0 * std::log10(1.0 / mse);
    std::printf("analytic smooth-gradient check: PSNR %.2f dB\n", psnr);
    if (psnr < 25.0) {
        std::printf("FAIL: the implementation is broken (a correct model "
                    "tracks the analytic target well above 25 dB)\n");
        return 1;
    }

    // The wrapper LAYOUT contract: the transform's PLANAR in/out must be
    // equivalent to a direct planar forward (an interleaved/planar mix-up
    // scrambles and fails this bit-exactly).
    {
        std::vector<float> pin(3 * w * h), direct(3 * 4 * w * h);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const size_t i = static_cast<size_t>(y) * w + x;
                pin[0 * w * h + i] = r[i];
                pin[1 * w * h + i] = g[i];
                pin[2 * w * h + i] = b[i];
            }
        }
        std::vector<float> wrap(3 * 4 * w * h);
        DemoUpscaleParams pw;
        pw.scale = 2;
        pw.kernel = DemoKernel::Neural;
        pw.threads = threads;
        if (!demo_upscale(pin.data(), w, h, wrap.data(), 2 * w, 2 * h, pw)) {
            std::printf("FAIL wrapper upscale\n");
            return 1;
        }
        m.forward_x2(r.data(), g.data(), b.data(), w, h,
                     direct.data() + 0 * 4 * w * h,
                     direct.data() + 1 * 4 * w * h,
                     direct.data() + 2 * 4 * w * h, threads);
        for (size_t i = 0; i < wrap.size(); ++i) {
            if (wrap[i] != direct[i]) {
                std::printf("FAIL wrapper layout at %zu (the planar "
                            "contract is broken)\n", i);
                return 1;
            }
        }
        std::printf("wrapper planar layout: bit-identical to direct "
                    "forward\n");
    }

    // Loader negatives: a truncated file and a wrong-magic file must
    // reject with NAMED errors (never a crash, never a partial load).
    {
        const std::string trunc = std::string(argv[1]) + ".truncated-test";
        {
            std::ifstream in(argv[1], std::ios::binary);
            std::vector<char> head(64);
            in.read(head.data(), 64);
            std::ofstream out(trunc, std::ios::binary);
            out.write(head.data(), 64);
        }
        if (demo_model_load(trunc)) {
            std::printf("FAIL: the truncated file was accepted\n");
            std::remove(trunc.c_str());
            return 1;
        }
        std::printf("truncated file rejected: %s\n",
                    demo_model_error().c_str());
        if (demo_model_error().empty()) {
            std::printf("FAIL: no named error for the truncated file\n");
            return 1;
        }
        std::remove(trunc.c_str());
        const std::string wrong = std::string(argv[1]) + ".wrongmagic-test";
        {
            std::ofstream out(wrong, std::ios::binary);
            out << "not a weights file";
        }
        if (demo_model_load(wrong)) {
            std::printf("FAIL: the wrong-magic file was accepted\n");
            std::remove(wrong.c_str());
            return 1;
        }
        std::printf("wrong-magic file rejected: %s\n",
                    demo_model_error().c_str());
        if (demo_model_error().empty()) {
            std::printf("FAIL: no named error for the wrong-magic file\n");
            return 1;
        }
        std::remove(wrong.c_str());
        // A failed reload must NOT destroy the loaded model.
        if (!demo_model_loaded() || demo_model_kind() != 2) {
            std::printf("FAIL: the loaded model did not survive a failed "
                        "reload (kind=%d loaded=%d)\n",
                        demo_model_kind(), (int)demo_model_loaded());
            return 1;
        }
        std::printf("loaded model survives failed reloads\n");
    }

    // The measured parameter count pin (the loader verifies the full
    // topology; this pin fails if a tensor is silently dropped).
    {
        const int params = m.param_count();
        std::printf("params: %d\n", params);
        if (params != 1369883) {
            std::printf("FAIL: parameter count %d != 1369883\n", params);
            return 1;
        }
    }
    std::printf("smoke ok\n");
    return 0;
}
