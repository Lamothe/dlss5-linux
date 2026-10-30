// Smoke test for the Real-ESRGAN x2 demonstration runtime.
#include "../native-demo-rrdbnet.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>

int main(int argc, char** argv)
{
    const char* path = argc > 1
        ? argv[1]
        : ".tools/models/RealESRGAN_x2plus.safetensors";
    demo_rrdb::RRDBNet net;
    if (!net.load(path))
    {
        std::printf("FAIL load: %s\n", net.error().c_str());
        return 1;
    }
    std::printf("loaded: %d params\n", net.param_count());
    if (net.param_count() != 16703171)
    {
        std::printf("FAIL param count\n");
        return 1;
    }
    // Determinism + extent + finiteness on a synthetic gradient.
    const int w = (argc > 2) ? atoi(argv[2]) : 64;
    const int h = (argc > 3) ? atoi(argv[3]) : 48;
    const int th = (argc > 4) ? atoi(argv[4]) : 4;
    std::vector<float> r(w * h), g(w * h), b(w * h);
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            r[y * w + x] = x / (float)w;
            g[y * w + x] = y / (float)h;
            b[y * w + x] = 0.5f;
        }
    }
    std::vector<float> o1r(4 * w * h), o1g(4 * w * h), o1b(4 * w * h);
    std::vector<float> o2r(4 * w * h), o2g(4 * w * h), o2b(4 * w * h);
    auto t0 = std::chrono::steady_clock::now();
    net.forward_x2(r.data(), g.data(), b.data(), w, h, o1r.data(), o1g.data(),
                   o1b.data(), th);
    auto t1 = std::chrono::steady_clock::now();
    net.forward_x2(r.data(), g.data(), b.data(), w, h, o2r.data(), o2g.data(),
                   o2b.data(), th);
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("forward %dx%d: %.1f ms (%d threads)\n", w, h, ms, th);
    int nans = 0;
    for (int i = 0; i < 4 * w * h; ++i)
    {
        if (!std::isfinite(o1r[i]) || !std::isfinite(o1g[i]) ||
            !std::isfinite(o1b[i]))
        {
            ++nans;
        }
    }
    std::printf("non-finite outputs: %d\n", nans);
    for (int i = 0; i < 4 * w * h; ++i)
    {
        if (o1r[i] != o2r[i] || o1g[i] != o2g[i] || o1b[i] != o2b[i])
        {
            std::printf("FAIL determinism at %d (%f vs %f)\n", i, o1r[i],
                        o2r[i]);
            return 1;
        }
        if (!std::isfinite(o1r[i]) || !std::isfinite(o1g[i]) ||
            !std::isfinite(o1b[i]))
        {
            std::printf("FAIL non-finite output\n");
            return 1;
        }
    }
    // The output should not be constant (the network reacts to input).
    float mn = 1e9f, mx = -1e9f;
    for (int i = 0; i < 4 * w * h; ++i)
    {
        mn = std::min(mn, o1r[i]);
        mx = std::max(mx, o1r[i]);
    }
    std::printf("output range r: %.3f..%.3f\n", mn, mx);
    if (mx - mn < 0.01f)
    {
        std::printf("FAIL output looks constant\n");
        return 1;
    }
    std::printf("smoke ok\n");
    return 0;
}
