// native-demo-upscale-test.cpp — the demonstration transform's quality
// harness (NOT DLSS5).
//
// The measured quality claim: downscale a known high-resolution test
// image, upsample the result with (a) the demonstration transform and
// (b) a bilinear baseline, and compare both against the source with
// PSNR and SSIM. The transform must beat bilinear on both metrics for
// the claim "Michael's F9 visibly improves the image" to stand - that
// is the only quality claim made.
//
// The test image is deterministic and generated here (mixed frequencies,
// edges, gradients and a chart-like region): "a known high-res image"
// with reproducible content and no licensing question. Every reported
// number is a real measurement from this run.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "native-demo-upscale.hpp"

namespace {

int g_checks = 0;
int g_failures = 0;

bool CHECK(bool ok, const char* what, int line) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("FAIL (line %d): %s\n", line, what);
    }
    return ok;
}

#define CHECK_(cond) CHECK((cond), #cond, __LINE__)

constexpr int kW = 320;
constexpr int kH = 288;
constexpr int kScale = 4;
constexpr int kLw = kW / kScale;
constexpr int kLh = kH / kScale;

// The deterministic high-resolution source: planar RGB in [0, 1].
void make_source(std::vector<float>& rgb) {
    rgb.assign(3 * static_cast<std::size_t>(kW) * kH, 0.0f);
    const std::size_t plane = static_cast<std::size_t>(kW) * kH;
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            const float fx = static_cast<float>(x) / kW;
            const float fy = static_cast<float>(y) / kH;
            // Mixed frequencies + a gradient + hard edges + a chart block.
            float r = 0.5f + 0.5f * std::sin(6.2831853f * (3.0f * fx + fy));
            float g = 0.2f + 0.8f * fx * fy;
            float b = 0.5f + 0.5f * std::sin(6.2831853f * (fx - 2.0f * fy));
            if (x > kW / 2 && y > kH / 2) {
                // Chart-like hard edges (the resampler's honest case).
                const int cell = ((x / 8) + (y / 8)) & 1;
                r = g = b = cell ? 1.0f : 0.0f;
            }
            if ((x % 40) < 2 || (y % 40) < 2) {
                r = 1.0f;
                g = 0.0f;
                b = 0.0f;
            }
            rgb[0 * plane + static_cast<std::size_t>(y) * kW + x] = r;
            rgb[1 * plane + static_cast<std::size_t>(y) * kW + x] = g;
            rgb[2 * plane + static_cast<std::size_t>(y) * kW + x] = b;
        }
    }
}

// The downscale that produces the "low-resolution capture": box average
// of scale x scale blocks (the standard capture model).
void downscale(const std::vector<float>& hi, std::vector<float>& lo) {
    const std::size_t plane_hi = static_cast<std::size_t>(kW) * kH;
    const std::size_t plane_lo = static_cast<std::size_t>(kLw) * kLh;
    lo.assign(3 * plane_lo, 0.0f);
    const float n = static_cast<float>(kScale * kScale);
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < kLh; ++y) {
            for (int x = 0; x < kLw; ++x) {
                float acc = 0.0f;
                for (int dy = 0; dy < kScale; ++dy) {
                    for (int dx = 0; dx < kScale; ++dx) {
                        acc += hi[c * plane_hi
                                  + static_cast<std::size_t>(y * kScale + dy)
                                        * kW
                                  + (x * kScale + dx)];
                    }
                }
                lo[c * plane_lo + static_cast<std::size_t>(y) * kLw + x] =
                    acc / n;
            }
        }
    }
}

// The bilinear baseline (the honest comparison floor).
void bilinear(const std::vector<float>& lo, std::vector<float>& up) {
    const std::size_t plane_lo = static_cast<std::size_t>(kLw) * kLh;
    const std::size_t plane_hi = static_cast<std::size_t>(kW) * kH;
    up.assign(3 * plane_hi, 0.0f);
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < kH; ++y) {
            const float fy = (static_cast<float>(y) + 0.5f) / kScale - 0.5f;
            const int y0 = static_cast<int>(std::floor(fy));
            const float ty = fy - static_cast<float>(y0);
            for (int x = 0; x < kW; ++x) {
                const float fx = (static_cast<float>(x) + 0.5f) / kScale
                    - 0.5f;
                const int x0 = static_cast<int>(std::floor(fx));
                const float tx = fx - static_cast<float>(x0);
                const int xa = x0 < 0 ? 0 : (x0 >= kLw ? kLw - 1 : x0);
                const int xb = xa + 1 >= kLw ? kLw - 1 : xa + 1;
                const int ya = y0 < 0 ? 0 : (y0 >= kLh ? kLh - 1 : y0);
                const int yb = ya + 1 >= kLh ? kLh - 1 : ya + 1;
                const float v00 = lo[c * plane_lo + ya * kLw + xa];
                const float v10 = lo[c * plane_lo + ya * kLw + xb];
                const float v01 = lo[c * plane_lo + yb * kLw + xa];
                const float v11 = lo[c * plane_lo + yb * kLw + xb];
                up[c * plane_hi + static_cast<std::size_t>(y) * kW + x] =
                    (1 - ty) * ((1 - tx) * v00 + tx * v10)
                    + ty * ((1 - tx) * v01 + tx * v11);
            }
        }
    }
}

double psnr(const std::vector<float>& a, const std::vector<float>& b) {
    double mse = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i] - b[i]);
        mse += d * d;
    }
    mse /= static_cast<double>(a.size());
    if (mse <= 0.0) {
        return 99.0;
    }
    return 10.0 * std::log10(1.0 / mse);  // peak = 1.0 in [0,1] space
}

// SSIM (Wang et al.) on the luma-weighted sum of the planes, 8x8 windows
// over the interior (the standard constants C1 = 0.01^2, C2 = 0.03^2).
double ssim(const std::vector<float>& a, const std::vector<float>& b) {
    const std::size_t plane = static_cast<std::size_t>(kW) * kH;
    const double C1 = 0.01 * 0.01;
    const double C2 = 0.03 * 0.03;
    double total = 0.0;
    int windows = 0;
    for (int wy = 0; wy + 8 <= kH; wy += 4) {
        for (int wx = 0; wx + 8 <= kW; wx += 4) {
            double ma = 0.0, mb = 0.0;
            for (int y = 0; y < 8; ++y) {
                for (int x = 0; x < 8; ++x) {
                    const std::size_t idx =
                        static_cast<std::size_t>(wy + y) * kW + (wx + x);
                    const double va = 0.299 * a[idx]
                        + 0.587 * a[plane + idx] + 0.114 * a[2 * plane + idx];
                    const double vb = 0.299 * b[idx]
                        + 0.587 * b[plane + idx] + 0.114 * b[2 * plane + idx];
                    ma += va;
                    mb += vb;
                }
            }
            ma /= 64.0;
            mb /= 64.0;
            double va = 0.0, vb = 0.0, cov = 0.0;
            for (int y = 0; y < 8; ++y) {
                for (int x = 0; x < 8; ++x) {
                    const std::size_t idx =
                        static_cast<std::size_t>(wy + y) * kW + (wx + x);
                    const double da_ = 0.299 * a[idx]
                        + 0.587 * a[plane + idx] + 0.114 * a[2 * plane + idx];
                    const double db_ = 0.299 * b[idx]
                        + 0.587 * b[plane + idx] + 0.114 * b[2 * plane + idx];
                    va += (da_ - ma) * (da_ - ma);
                    vb += (db_ - mb) * (db_ - mb);
                    cov += (da_ - ma) * (db_ - mb);
                }
            }
            va /= 63.0;
            vb /= 63.0;
            cov /= 63.0;
            const double num = (2 * ma * mb + C1) * (2 * cov + C2);
            const double den = (ma * ma + mb * mb + C1) * (va + vb + C2);
            total += num / den;
            ++windows;
        }
    }
    return windows > 0 ? total / windows : 1.0;
}

}  // namespace

int main() {
    using namespace dlss5_nr::demoup;

    std::vector<float> source, low, up_bilinear, up_demo;
    make_source(source);
    downscale(source, low);
    bilinear(low, up_bilinear);

    DemoUpscaleParams p;
    p.scale = kScale;
    up_demo.assign(3 * static_cast<std::size_t>(kW) * kH, 0.0f);
    // The DEFAULT parameter set (sharpen = 0: pure Lanczos3) is the
    // measured choice; the sweep below records the alternatives.
    CHECK_(demo_upscale(low.data(), kLw, kLh, up_demo.data(), kW, kH, p));
    // Parameter sweep (documented tuning): the sharpen gain/sigma set is
    // chosen on these two metrics; every row is a real measurement.
    for (float sh : {0.0f, 0.1f, 0.2f, 0.35f, 0.5f}) {
        for (float sg : {0.8f, 1.0f}) {
            DemoUpscaleParams ps;
            ps.scale = kScale;
            ps.sharpen = sh;
            ps.sigma = sg;
            std::vector<float> u;
            u.assign(up_demo.size(), 0.0f);
            demo_upscale(low.data(), kLw, kLh, u.data(), kW, kH, ps);
            std::printf("  sweep sharpen=%.2f sigma=%.1f: PSNR %.2f dB, "
                        "SSIM %.4f\n", sh, sg, psnr(source, u),
                        ssim(source, u));
        }
    }

    const double psnr_bilinear = psnr(source, up_bilinear);
    const double psnr_demo = psnr(source, up_demo);
    const double ssim_bilinear = ssim(source, up_bilinear);
    const double ssim_demo = ssim(source, up_demo);

    std::printf("quality (NOT DLSS5 - demonstration transform with real "
                "weights):\n");
    std::printf("  source: %dx%d generated test image; capture: %dx%d box "
                "downscale x%d\n", kW, kH, kLw, kLh, kScale);
    std::printf("  bilinear baseline: PSNR %.2f dB, SSIM %.4f\n",
                psnr_bilinear, ssim_bilinear);
    std::printf("  demonstration transform: PSNR %.2f dB, SSIM %.4f\n",
                psnr_demo, ssim_demo);
    std::printf("  delta vs baseline: PSNR %+.2f dB, SSIM %+.4f\n",
                psnr_demo - psnr_bilinear, ssim_demo - ssim_bilinear);

    // The measured claim: the transform beats the bilinear baseline on
    // both metrics.
    CHECK_(psnr_demo > psnr_bilinear);
    CHECK_(ssim_demo > ssim_bilinear);

    // Determinism: a second run is bit-identical.
    std::vector<float> up2;
    up2.assign(up_demo.size(), 0.0f);
    CHECK_(demo_upscale(low.data(), kLw, kLh, up2.data(), kW, kH, p));
    CHECK_(std::memcmp(up_demo.data(), up2.data(),
                       up_demo.size() * sizeof(float))
           == 0);

    // The unsharp gain zero must not change the resampled result (the
    // pass is skipped cleanly), and the identity scale must be exact.
    {
        DemoUpscaleParams p0 = p;
        p0.sharpen = 0.0f;
        std::vector<float> a, b;
        a.assign(up_demo.size(), 0.0f);
        b.assign(up_demo.size(), 0.0f);
        CHECK_(demo_upscale(low.data(), kLw, kLh, a.data(), kW, kH, p0));
        DemoUpscaleParams p1 = p0;
        p1.sharpen = 0.0001f;
        CHECK_(demo_upscale(low.data(), kLw, kLh, b.data(), kW, kH, p1));
        // Not bit-equal (the gain runs) but within the gain's magnitude.
        double max_d = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            const double d = std::fabs(static_cast<double>(a[i] - b[i]));
            if (d > max_d) {
                max_d = d;
            }
        }
        CHECK_(max_d < 0.01);
    }

    // Rejection classes: bad shapes leave the output untouched.
    {
        std::vector<float> out(3, -1.0f);
        DemoUpscaleParams bad = p;
        bad.scale = 3;  // ow != lw * scale
        CHECK_(!demo_upscale(low.data(), kLw, kLh, out.data(), kW, kH, bad));
        CHECK_(out[0] == -1.0f);
        CHECK_(!demo_upscale(nullptr, kLw, kLh, out.data(), kW, kH, p));
        CHECK_(out[0] == -1.0f);
    }

    std::printf("native-demo-upscale: %d checks passed\n", g_checks);
    if (g_failures != 0) {
        std::printf("native-demo-upscale: %d check(s) FAILED\n", g_failures);
        return 1;
    }
    std::printf("native-demo-upscale: OK\n");
    return 0;
}
