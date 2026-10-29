// native-demo-upscale.hpp — the demonstration transform (NOT DLSS5).
//
// A real, openly documented upscaling transform with real parameters:
// Lanczos3 resampling (the analytic windowed-sinc kernel, a = 3) followed
// by an unsharp enhancement pass (a small Gaussian blur and a documented
// gain). The "weights" are the published kernel and enhancement
// coefficients recorded in `DemoUpscaleParams` - openly licensed by
// construction (the mathematical definitions), obtained openly, and
// entirely independent of any vendor.
//
// THIS IS NOT DLSS5. It is a demonstration transform with real weights
// whose only claim is the measured one (vr/tests/native-demo-upscale-test.cpp:
// PSNR/SSIM against the source and against a bilinear baseline). Every
// artefact, UI label and report must carry:
//   "not DLSS5 - demonstration transform with real weights"
//
// Determinism: f32 arithmetic in a fixed order; no threading inside the
// transform; identical inputs give identical outputs bit for bit within a
// build (the tests pin it). The kernel weights are built with libm
// sin/exp, so cross-toolchain bit-identity is not claimed (the documented
// PSNR/SSIM numbers are build-specific measurements).

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace dlss5_nr {
namespace demoup {

// The parameter set (the "weights"): documented, real, openly published.
//   scale        integer upscale factor (2..4 in the harness)
//   sharpen      unsharp gain applied after resampling (0 = off)
//   sigma        the unsharp blur's Gaussian sigma in OUTPUT pixels
//                (the pass runs on the upscaled buffer)
//   lanczos_a   the Lanczos window (3.0 = the classic Lanczos3)
// The demonstration kernels: Lanczos3 (the transform) and the bilinear
// baseline (the honest A/B comparison floor - also a real, openly
// documented weight set). The kernel is a parameter of the SAME
// resampling path, so the A/B comparison is apples to apples.
enum class DemoKernel : int {
    Lanczos3 = 0,
    Bilinear = 1,
};

struct DemoUpscaleParams {
    int scale = 4;
    // The measured default (the harness's sweep): sharpen = 0 keeps the
    // pure Lanczos3 result, which beats the bilinear baseline on BOTH
    // PSNR and SSIM. The harness's printed sweep shows the sharpen gain
    // trading PSNR for SSIM at the measured points.
    float sharpen = 0.0f;
    float sigma = 1.0f;
    float lanczos_a = 3.0f;
    // The resampling kernel: Lanczos3 (default) or the bilinear baseline.
    DemoKernel kernel = DemoKernel::Lanczos3;
};

// The Lanczos kernel: sinc(x) * sinc(x / a) for |x| < a, else 0.
inline float lanczos(float x, float a) {
    if (x == 0.0f) {
        return 1.0f;
    }
    if (x <= -a || x >= a) {
        return 0.0f;
    }
    const float px = 3.14159265358979323846f * x;
    const float s = std::sin(px) / px * std::sin(px / a) / (px / a);
    return s;
}

// The bilinear (tent) kernel: max(0, 1 - |x|).
inline float tent(float x) {
    const float a = x < 0.0f ? -x : x;
    return a >= 1.0f ? 0.0f : 1.0f - a;
}

// One-dimensional resampling weights for an output line of `out_count`
// samples mapping back onto `in_count` inputs at the given scale (the
// standard (x_out + 0.5) / scale - 0.5 mapping). `taps` = the kernel
// support in input samples (ceil(a) per side + 1). The caller allocates
// `out_count * taps` entries; `index[o * taps + k]` is the input sample
// (clamped to range) and `weight[o * taps + k]` its normalised weight.
inline void resample_weights(int in_count, int out_count, float scale,
                             float a, int taps, std::vector<int>& index,
                             std::vector<float>& weight,
                             DemoKernel kernel = DemoKernel::Lanczos3) {
    index.assign(static_cast<std::size_t>(out_count) * taps, 0);
    weight.assign(static_cast<std::size_t>(out_count) * taps, 0.0f);
    for (int o = 0; o < out_count; ++o) {
        const float center =
            (static_cast<float>(o) + 0.5f) / scale - 0.5f;
        // floor(center) - ceil(a) + 1 .. floor(center) + ceil(a) covers
        // the kernel support (|x| < a) with no zero-valued tap.
        int base = static_cast<int>(std::floor(center))
            - static_cast<int>(std::ceil(a)) + 1;
        float total = 0.0f;
        for (int k = 0; k < taps; ++k) {
            int src = base + k;
            const int clamped = src < 0 ? 0 : (src >= in_count ? in_count - 1
                                                               : src);
            const float w =
                kernel == DemoKernel::Bilinear
                    ? tent(center - static_cast<float>(src))
                    : lanczos(center - static_cast<float>(src), a);
            index[static_cast<std::size_t>(o) * taps + k] = clamped;
            weight[static_cast<std::size_t>(o) * taps + k] = w;
            total += w;
        }
        if (total != 0.0f) {
            for (int k = 0; k < taps; ++k) {
                weight[static_cast<std::size_t>(o) * taps + k] /= total;
            }
        }
    }
}

// The Gaussian blur used by the unsharp pass (separable, sampled at
// +/- 2 sigma, normalised).
inline void gaussian_weights(float sigma, std::vector<float>& w) {
    const int radius = sigma > 0.0f ? static_cast<int>(std::ceil(2.0f * sigma))
                                    : 0;
    w.assign(static_cast<std::size_t>(2 * radius + 1), 0.0f);
    float total = 0.0f;
    for (int i = -radius; i <= radius; ++i) {
        const float v = std::exp(-0.5f * static_cast<float>(i * i)
                                 / (sigma * sigma));
        w[static_cast<std::size_t>(i + radius)] = v;
        total += v;
    }
    for (auto& v : w) {
        v /= total;
    }
}

// The transform: planar RGB f32 in [0, 1] (3 planes of lw*lh) in, planar
// RGB f32 out (3 planes of ow*oh). ow == lw * scale and oh == lh * scale.
// Returns false on a bad shape or null pointer (the output is untouched).
inline bool demo_upscale(const float* in_rgb, int lw, int lh, float* out_rgb,
                         int ow, int oh,
                         const DemoUpscaleParams& p = DemoUpscaleParams{}) {
    if (in_rgb == nullptr || out_rgb == nullptr || lw < 1 || lh < 1
        || ow != lw * p.scale || oh != lh * p.scale || p.scale < 1
        || !std::isfinite(p.lanczos_a) || !std::isfinite(p.sharpen)
        || !std::isfinite(p.sigma) || p.lanczos_a < 1.0f
        || lw > 1 << 14 || lh > 1 << 14 || p.scale > 16) {
        return false;
    }
    const std::size_t in_plane = static_cast<std::size_t>(lw) * lh;
    const std::size_t out_plane = static_cast<std::size_t>(ow) * oh;
    // The bilinear kernel has unit support; Lanczos3 honours lanczos_a.
    const float a = p.kernel == DemoKernel::Bilinear ? 1.0f : p.lanczos_a;
    const int taps = 2 * static_cast<int>(std::ceil(a));

    std::vector<int> ix, iy;
    std::vector<float> wx, wy;
    resample_weights(lw, ow, static_cast<float>(p.scale), a, taps,
                     ix, wx, p.kernel);
    resample_weights(lh, oh, static_cast<float>(p.scale), a, taps,
                     iy, wy, p.kernel);

    // Per channel: the horizontal pass into a channel-sized intermediate,
    // then the vertical pass into the output plane.
    std::vector<float> up(3 * out_plane, 0.0f);
    for (int c = 0; c < 3; ++c) {
        std::vector<float> midc(static_cast<std::size_t>(ow) * lh, 0.0f);
        for (int y = 0; y < lh; ++y) {
            for (int x = 0; x < ow; ++x) {
                float acc = 0.0f;
                for (int k = 0; k < taps; ++k) {
                    const int sx = ix[static_cast<std::size_t>(x) * taps + k];
                    acc += wx[static_cast<std::size_t>(x) * taps + k]
                        * in_rgb[c * in_plane
                                 + static_cast<std::size_t>(y) * lw + sx];
                }
                midc[static_cast<std::size_t>(y) * ow + x] = acc;
            }
        }
        for (int y = 0; y < oh; ++y) {
            for (int x = 0; x < ow; ++x) {
                float acc = 0.0f;
                for (int k = 0; k < taps; ++k) {
                    const int sy = iy[static_cast<std::size_t>(y) * taps + k];
                    acc += wy[static_cast<std::size_t>(y) * taps + k]
                        * midc[static_cast<std::size_t>(sy) * ow + x];
                }
                up[c * out_plane + static_cast<std::size_t>(y) * ow + x] =
                    acc;
            }
        }
    }

    // The unsharp pass (skipped when the gain is zero).
    if (p.sharpen != 0.0f && p.sigma > 0.0f) {
        std::vector<float> blur;
        gaussian_weights(p.sigma, blur);
        const int radius = static_cast<int>(blur.size() - 1) / 2;
        std::vector<float> tmp(3 * out_plane, 0.0f);
        // Horizontal blur.
        for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < oh; ++y) {
                for (int x = 0; x < ow; ++x) {
                    float acc = 0.0f;
                    for (int k = -radius; k <= radius; ++k) {
                        const int sx = x + k < 0
                            ? 0
                            : (x + k >= ow ? ow - 1 : x + k);
                        acc += blur[static_cast<std::size_t>(k + radius)]
                            * up[c * out_plane
                                 + static_cast<std::size_t>(y) * ow + sx];
                    }
                    tmp[c * out_plane + static_cast<std::size_t>(y) * ow + x]
                        = acc;
                }
            }
        }
        // Vertical blur + the unsharp combine.
        for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < oh; ++y) {
                for (int x = 0; x < ow; ++x) {
                    float acc = 0.0f;
                    for (int k = -radius; k <= radius; ++k) {
                        const int sy = y + k < 0
                            ? 0
                            : (y + k >= oh ? oh - 1 : y + k);
                        acc += blur[static_cast<std::size_t>(k + radius)]
                            * tmp[c * out_plane
                                  + static_cast<std::size_t>(sy) * ow + x];
                    }
                    const float v =
                        up[c * out_plane + static_cast<std::size_t>(y) * ow
                           + x];
                    out_rgb[c * out_plane
                            + static_cast<std::size_t>(y) * ow + x] =
                        v + p.sharpen * (v - acc);
                }
            }
        }
    } else {
        std::memcpy(out_rgb, up.data(), up.size() * sizeof(float));
    }
    return true;
}

// The mandatory label for anything this transform produces.
inline const char* demo_upscale_label() {
    return "not DLSS5 - demonstration transform with real weights";
}

}  // namespace demoup
}  // namespace dlss5_nr
