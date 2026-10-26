// Independent tests for the native output-conversion stage (vr/native-compose.hpp):
// the 32 -> 3 residual projection and the output compose (residual add, clamp,
// intensity blend, intensity > 1 luminance-ratio amplification). CPU-only. Synthetic
// frames, residuals and weights are original; passing establishes synthetic arithmetic
// correctness of this implementation against independently derived controls only —
// never vendor correspondence.
//
// Independence: the compose oracles are scalar re-derivations of the documented formula
// (sharing no code with the production routines; same process and libm, so bit-exact
// agreement is expected and checked exactly). The independent pins are the
// hand-derived literals — including the amplification clamp band, whose outcome is
// libm-independent by construction. The projection's ascending accumulation order is
// pinned by an order-discriminating hand literal. Cross-platform bit-exactness of the
// pow is not claimed anywhere.

#include "../native-compose.hpp"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>
#include <new>

// ---- allocation accounting ----------------------------------------------------------------------
static std::atomic<unsigned long long> g_allocations{0};
void* operator new(size_t n) {
    g_allocations.fetch_add(1);
    void* p = malloc(n ? n : 1);
    if (!p) {
        abort();
    }
    return p;
}
void* operator new[](size_t n) { return ::operator new(n); }
void* operator new(size_t n, std::align_val_t) {
    g_allocations.fetch_add(1);
    void* p = malloc(n ? n : 1);
    if (!p) {
        abort();
    }
    return p;
}
void* operator new[](size_t n, std::align_val_t a) { return ::operator new(n, a); }
void operator delete(void* p) noexcept { free(p); }
void operator delete[](void* p) noexcept { free(p); }
void operator delete(void* p, size_t) noexcept { free(p); }
void operator delete[](void* p, size_t) noexcept { free(p); }
void operator delete(void* p, std::align_val_t) noexcept { free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { free(p); }
void operator delete(void* p, size_t, std::align_val_t) noexcept { free(p); }
void operator delete[](void* p, size_t, std::align_val_t) noexcept { free(p); }

// ---- test harness -------------------------------------------------------------------------------
static unsigned long long g_checks = 0;
#define CHECK(condition) do { assert(condition); ++g_checks; } while (0)

static const float kSentinel = -987.25f;
static void fill_sentinel(float* values, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        values[i] = kSentinel;
    }
}

static bool same_value(float a, float b) {
    return (a == b) || (isnan(a) && isnan(b));
}

static uint32_t g_rng = 0x12345678u;
static uint32_t rng(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}

static float random_f16(void) {
    return dlss5_nr::f16_from_bits((uint16_t)rng());
}

static float random_finite_f16(void) {
    float value = random_f16();
    while (!std::isfinite(value)) {
        value = random_f16();
    }
    return value;
}

static float random_unit(void) {
    return (float)(rng() >> 8) * (1.0f / 16777216.0f);
}

// ---- independent oracles -------------------------------------------------------------------------
// Scalar re-derivations of the documented formulae, written from the specification
// without calling the production routines. Same process and libm, so bit-exact
// agreement is expected and checked exactly; the independent pins are the hand
// literals (the amplification clamp band is libm-independent by construction).
static void oracle_compose_pixel(const float frame[3], const float residual[3],
                                 double intensity, float out[3]) {
    float network[3];
    for (int c = 0; c < 3; ++c) {
        float value = frame[c] + residual[c] * 0.25f;
        if (value < 0.0f) {
            value = 0.0f;
        }
        if (value > 1.0f) {
            value = 1.0f;
        }
        network[c] = value;
    }
    double capped = intensity < 1.0 ? intensity : 1.0;
    capped = capped < -(double)FLT_MAX ? -(double)FLT_MAX
        : (capped > (double)FLT_MAX ? (double)FLT_MAX : capped);
    float t = (float)capped;
    float blended[3];
    for (int c = 0; c < 3; ++c) {
        float value = frame[c] + t * (network[c] - frame[c]);
        if (value < 0.0f) {
            value = 0.0f;
        }
        if (value > 1.0f) {
            value = 1.0f;
        }
        blended[c] = value;
    }
    if (intensity <= 1.0) {
        out[0] = blended[0];
        out[1] = blended[1];
        out[2] = blended[2];
        return;
    }
    float luma_out = (blended[0] * 0.2126f + blended[1] * 0.7152f)
        + blended[2] * 0.0722f;
    float luma_frame = (frame[0] * 0.2126f + frame[1] * 0.7152f)
        + frame[2] * 0.0722f;
    float ratio = (luma_out + 0x1p-9f) / (luma_frame + 0x1p-9f);
    double amp = std::pow((double)ratio, intensity);
    amp = amp < 0.5 ? 0.5 : (amp > 2.0 ? 2.0 : amp);
    float amp_f = (float)amp;
    for (int c = 0; c < 3; ++c) {
        float value = (blended[c] * amp_f) / ratio;
        if (value < 0.0f) {
            value = 0.0f;
        }
        if (value > 1.0f) {
            value = 1.0f;
        }
        out[c] = value;
    }
}

static float oracle_project(const float* weights, const float* wide32) {
    float sum = 0.0f;
    for (int c = 0; c < 32; ++c) {
        sum += weights[c] * wide32[c];
    }
    return sum;
}

// ---- layout helpers ------------------------------------------------------------------------------
struct Batch {
    float* data;
    int channels;
    int rows;
    int cols;
    int plane_stride;
    int row_stride;
};

static dlss5_nr::FeatureBatchView view(const Batch& b) {
    dlss5_nr::FeatureBatchView v = {};
    v.data = b.data;
    v.channels = b.channels;
    v.rows = b.rows;
    v.cols = b.cols;
    v.plane_stride = b.plane_stride;
    v.row_stride = b.row_stride;
    return v;
}

// Interleaved RGB frame helper.
struct Frame {
    float* rgb;
    int rows;
    int cols;
    int row_stride;
};

static dlss5_nr::FrameView frame_view(const Frame& f) {
    dlss5_nr::FrameView v = {};
    v.rgb = f.rgb;
    v.rows = f.rows;
    v.cols = f.cols;
    v.row_stride = f.row_stride;
    return v;
}

static void fill_weights_identity(float* weights) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 32; ++c) {
            weights[r * 32 + c] = (c == r) ? 1.0f : 0.0f;
        }
    }
}

// ---- compose_pixel literals -----------------------------------------------------------------------
static void check_compose_literals(void) {
    using dlss5_nr::compose_pixel;
    // Identity: zero residual keeps the frame for every intensity.
    {
        const float frame[3] = { 0.25f, 0.5f, 0.75f };
        const float residual[3] = { 0.0f, 0.0f, 0.0f };
        for (int i = 0; i < 4; ++i) {
            const double intensity[4] = { 0.0, 0.5, 1.0, 100.0 };
            float out[3];
            compose_pixel(frame, residual, intensity[i], out);
            CHECK(out[0] == 0.25f && out[1] == 0.5f && out[2] == 0.75f);
        }
    }
    // Intensity 0 returns the frame unchanged even for a large residual.
    {
        const float frame[3] = { 0.2f, 0.4f, 0.6f };
        const float residual[3] = { 4.0f, -4.0f, 8.0f };
        float out[3];
        compose_pixel(frame, residual, 0.0, out);
        CHECK(out[0] == 0.2f && out[1] == 0.4f && out[2] == 0.6f);
    }
    // Intensity 1 takes the clamped network: frame 0.5 + residual 2 * 0.25 = 1.0.
    {
        const float frame[3] = { 0.5f, 0.5f, 0.5f };
        const float residual[3] = { 2.0f, 2.0f, 2.0f };
        float out[3];
        compose_pixel(frame, residual, 1.0, out);
        CHECK(out[0] == 1.0f && out[1] == 1.0f && out[2] == 1.0f);
    }
    // Network clamp, both ends, at intensity 1.
    {
        const float hi[3] = { 0.9f, 0.9f, 0.9f };
        const float lo[3] = { 0.1f, 0.1f, 0.1f };
        const float big[3] = { 8.0f, 8.0f, 8.0f };
        const float big_neg[3] = { -8.0f, -8.0f, -8.0f };
        float out[3];
        compose_pixel(hi, big, 1.0, out);
        CHECK(out[0] == 1.0f && out[1] == 1.0f && out[2] == 1.0f);
        compose_pixel(lo, big_neg, 1.0, out);
        CHECK(out[0] == 0.0f && out[1] == 0.0f && out[2] == 0.0f);
    }
    // Half-intensity blend, exact f32: frame 0.25 + 0.5 * (0.5 - 0.25) = 0.375.
    {
        const float frame[3] = { 0.25f, 0.25f, 0.25f };
        const float residual[3] = { 1.0f, 1.0f, 1.0f };
        float out[3];
        compose_pixel(frame, residual, 0.5, out);
        CHECK(out[0] == 0.375f && out[1] == 0.375f && out[2] == 0.375f);
    }
    // The blend evaluates frame + t * (network - frame) exactly as written; at
    // intensity 1 with frame 0.25 and network 0.75 the result is 0.75.
    {
        const float frame[3] = { 0.25f, 0.25f, 0.25f };
        const float residual[3] = { 2.0f, 2.0f, 2.0f };
        float out[3];
        compose_pixel(frame, residual, 1.0, out);
        CHECK(out[0] == 0.75f && out[1] == 0.75f && out[2] == 0.75f);
    }
    // Negative intensity extrapolates away from the network (defined semantics):
    // frame 0.5, residual 2 -> network 1.0; intensity -1 -> 0.5 - 0.5 = 0.0.
    {
        const float frame[3] = { 0.5f, 0.5f, 0.5f };
        const float residual[3] = { 2.0f, 2.0f, 2.0f };
        float out[3];
        compose_pixel(frame, residual, -1.0, out);
        CHECK(out[0] == 0.0f && out[1] == 0.0f && out[2] == 0.0f);
    }
    // Intensity just above 1 still uses the amplification branch, which at ratio 1
    // (zero residual) leaves the colour exactly in place for any intensity.
    {
        const float frame[3] = { 0.3f, 0.6f, 0.9f };
        const float residual[3] = { 0.0f, 0.0f, 0.0f };
        float out[3];
        compose_pixel(frame, residual, 1.0001, out);
        CHECK(out[0] == 0.3f && out[1] == 0.6f && out[2] == 0.9f);
        compose_pixel(frame, residual, 37.5, out);
        CHECK(out[0] == 0.3f && out[1] == 0.6f && out[2] == 0.9f);
    }
    // Amplification clamp band, pinned exactly (the clamp makes the outcome
    // libm-independent): a strong brightening at intensity 100 gives amplification
    // 2.0, and a strong darkening gives 0.5. The final colours equal the documented
    // formula with those exact amplifications.
    {
        const float bright_frame[3] = { 0.2f, 0.2f, 0.2f };
        const float bright_residual[3] = { 4.0f, 4.0f, 4.0f };
        float out[3];
        compose_pixel(bright_frame, bright_residual, 100.0, out);
        // network = 1.0; ratio = (luma(1) + 2^-9) / (luma(0.2) + 2^-9) > 2, so the
        // amplification clamps at 2 and out = clamp(2 / ratio).
        const float luma_one = (1.0f * 0.2126f + 1.0f * 0.7152f) + 1.0f * 0.0722f;
        const float luma_fifth = (0.2f * 0.2126f + 0.2f * 0.7152f) + 0.2f * 0.0722f;
        const float ratio = (luma_one + 0x1p-9f) / (luma_fifth + 0x1p-9f);
        const float expected = 2.0f / ratio;
        CHECK(ratio > 2.0f);
        CHECK(out[0] == expected && out[1] == expected && out[2] == expected);
        const float dark_frame[3] = { 1.0f, 1.0f, 1.0f };
        const float dark_residual[3] = { -2.0f, -2.0f, -2.0f };
        compose_pixel(dark_frame, dark_residual, 100.0, out);
        // network = 0.5; ratio < 1, amplification clamps at 0.5.
        const float luma_half = (0.5f * 0.2126f + 0.5f * 0.7152f) + 0.5f * 0.0722f;
        const float ratio_dark = (luma_half + 0x1p-9f) / (luma_one + 0x1p-9f);
        const float expected_dark = (0.5f * 0.5f) / ratio_dark;
        CHECK(ratio_dark < 1.0f);
        CHECK(out[0] == expected_dark && out[1] == expected_dark
            && out[2] == expected_dark);
    }
    // Out-of-float32-range finite intensities are range-guarded, not undefined: the
    // intensity -1e300 narrows t to -FLT_MAX, whose blend clamps to 0; the intensity
    // 1e300 pow overflows to +inf in double and clamps to 2.0 before narrowing,
    // giving exactly the intensity-100 brightening result.
    {
        const float frame[3] = { 0.5f, 0.5f, 0.5f };
        const float residual[3] = { 2.0f, 2.0f, 2.0f };
        float out[3];
        compose_pixel(frame, residual, -1e300, out);
        CHECK(out[0] == 0.0f && out[1] == 0.0f && out[2] == 0.0f);
        const float bright_frame[3] = { 0.2f, 0.2f, 0.2f };
        const float bright_residual[3] = { 4.0f, 4.0f, 4.0f };
        compose_pixel(bright_frame, bright_residual, 1e300, out);
        const float luma_one = (1.0f * 0.2126f + 1.0f * 0.7152f) + 1.0f * 0.0722f;
        const float luma_fifth = (0.2f * 0.2126f + 0.2f * 0.7152f) + 0.2f * 0.0722f;
        const float ratio = (luma_one + 0x1p-9f) / (luma_fifth + 0x1p-9f);
        const float expected = 2.0f / ratio;
        CHECK(out[0] == expected && out[1] == expected && out[2] == expected);
    }
}

// ---- compose oracle + sensitivity ----------------------------------------------------------------
static void check_compose_oracle(void) {
    using dlss5_nr::compose_pixel;
    // Exact-oracle fuzz on every path (pure f32 and amplification): the oracle
    // mirrors the documented double-then-clamp-then-narrow pow, so bit-exact
    // agreement is expected and checked exactly; the independent pins are the hand
    // literals (the amplification clamp band is libm-independent by construction).
    for (int i = 0; i < 20000; ++i) {
        float frame[3];
        float residual[3];
        for (int c = 0; c < 3; ++c) {
            frame[c] = random_unit();
            residual[c] = random_finite_f16() * 0.25f;
        }
        double intensity;
        switch (i % 4) {
            case 0: intensity = 0.0; break;
            case 1: intensity = (double)random_unit(); break;
            case 2: intensity = 1.0; break;
            default: intensity = 1.0 + (double)random_unit() * 3.0; break;
        }
        float out[3];
        float expected[3];
        compose_pixel(frame, residual, intensity, out);
        oracle_compose_pixel(frame, residual, intensity, expected);
        for (int c = 0; c < 3; ++c) {
            CHECK(same_value(out[c], expected[c]));
        }
    }
    // Changed-input sensitivity: one channel's residual moves only that channel.
    {
        const float frame[3] = { 0.5f, 0.5f, 0.5f };
        const float residual_a[3] = { 1.0f, 1.0f, 1.0f };
        const float residual_b[3] = { 1.0f, 1.5f, 1.0f };
        float out_a[3];
        float out_b[3];
        compose_pixel(frame, residual_a, 1.0, out_a);
        compose_pixel(frame, residual_b, 1.0, out_b);
        CHECK(out_a[0] == out_b[0]);
        CHECK(out_a[1] != out_b[1]);
        CHECK(out_a[2] == out_b[2]);
    }
    // Determinism: identical inputs, bit-identical outputs.
    {
        const float frame[3] = { 0.1f, 0.7f, 0.3f };
        const float residual[3] = { -1.25f, 2.5f, 0.75f };
        float out_a[3];
        float out_b[3];
        compose_pixel(frame, residual, 2.25, out_a);
        compose_pixel(frame, residual, 2.25, out_b);
        CHECK(memcmp(out_a, out_b, sizeof out_a) == 0);
    }
}

// ---- project_residual literals --------------------------------------------------------------------
static void check_project_literals(void) {
    using dlss5_nr::project_residual;
    using dlss5_nr::Status;
    // Identity rows: residual channel r reads wide channel r.
    {
        float weights[96];
        fill_weights_identity(weights);
        float wide_data[32];
        for (int c = 0; c < 32; ++c) {
            wide_data[c] = (float)(c + 1) * 0.25f;
        }
        float out[3];
        Batch wide = { wide_data, 32, 1, 1, 1, 1 };
        CHECK(project_residual(view(wide), weights, 96, out, 3) == Status::kOk);
        CHECK(out[0] == 0.25f && out[1] == 0.5f && out[2] == 0.75f);
    }
    // Ascending accumulation order discriminator: terms 1.0, 2^-24, 2^-24 with unit
    // weights. Accumulating c = 0 first, each 2^-24 lands on a tie at the f32 grid
    // point 1.0 and ties to even, so the pinned result is 1.0; summing the two small
    // terms first would give 1 + 2^-23 = 1.0000001192092896.
    {
        float weights[96] = {};
        float wide_data[32] = {};
        for (int c = 0; c < 32; ++c) {
            weights[c] = 1.0f;
        }
        wide_data[0] = 1.0f;
        wide_data[1] = 0x1p-24f;
        wide_data[2] = 0x1p-24f;
        float out[3];
        Batch wide = { wide_data, 32, 1, 1, 1, 1 };
        CHECK(project_residual(view(wide), weights, 96, out, 3) == Status::kOk);
        CHECK(out[0] == 1.0f);
        CHECK(1.0f + 0x1p-23f != 1.0f);  // the discarded alternative is distinct
    }
    // Mixed gains through the projection, hand-computed: two channels with weights
    // 0.5 and -2 for row 0.
    {
        float weights[96] = {};
        weights[0] = 0.5f;
        weights[1] = -2.0f;
        float wide_data[32] = {};
        wide_data[0] = 4.0f;
        wide_data[1] = 0.25f;
        float out[3];
        Batch wide = { wide_data, 32, 1, 1, 1, 1 };
        CHECK(project_residual(view(wide), weights, 96, out, 3) == Status::kOk);
        CHECK(out[0] == 1.5f);       // 4 * 0.5 + 0.25 * -2
        CHECK(out[1] == 0.0f && out[2] == 0.0f);
    }
}

// ---- project_residual oracle + permutation ---------------------------------------------------------
static void check_project_oracle(void) {
    using dlss5_nr::project_residual;
    using dlss5_nr::Status;
    float weights[96];
    float wide_data[32 * 2 * 2];
    float out[3 * 2 * 2];
    for (int i = 0; i < 96; ++i) {
        weights[i] = random_finite_f16();
    }
    for (int i = 0; i < 32 * 4; ++i) {
        wide_data[i] = random_finite_f16();
    }
    Batch wide = { wide_data, 32, 2, 2, 4, 2 };
    CHECK(project_residual(view(wide), weights, 96, out, 12) == Status::kOk);
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            for (int r = 0; r < 3; ++r) {
                float pixel[32];
                for (int c = 0; c < 32; ++c) {
                    pixel[c] = wide_data[c * 4 + y * 2 + x];
                }
                CHECK(same_value(out[r * 4 + y * 2 + x],
                    oracle_project(weights + r * 32, pixel)));
            }
        }
    }
    // Determinism: rerun is bit-identical.
    float out_again[3 * 2 * 2];
    CHECK(project_residual(view(wide), weights, 96, out_again, 12) == Status::kOk);
    CHECK(memcmp(out, out_again, sizeof out) == 0);
    // Channel-permutation control: rows are independent and ordered; a permuted
    // weight matrix must permute the residuals correspondingly.
    {
        float permuted[96] = {};
        for (int c = 0; c < 32; ++c) {
            permuted[0 * 32 + c] = weights[2 * 32 + c];
            permuted[1 * 32 + c] = weights[0 * 32 + c];
            permuted[2 * 32 + c] = weights[1 * 32 + c];
        }
        float swapped[12];
        CHECK(project_residual(view(wide), permuted, 96, swapped, 12) == Status::kOk);
        for (int i = 0; i < 4; ++i) {
            CHECK(swapped[0 * 4 + i] == out[2 * 4 + i]);
            CHECK(swapped[1 * 4 + i] == out[0 * 4 + i]);
            CHECK(swapped[2 * 4 + i] == out[1 * 4 + i]);
        }
    }
}

// ---- compose batch ---------------------------------------------------------------------------------
static void check_compose_batch(void) {
    using dlss5_nr::compose;
    using dlss5_nr::Status;
    // Known answer through the full batch path, 2x3 frame with padded residual
    // planes; intensity 0.5 with hand-computed outputs.
    {
        float frame_data[18];
        float residual_data[3 * 2 * 4];
        float out[18];
        for (int i = 0; i < 18; ++i) {
            frame_data[i] = (float)(i % 5) * 0.2f;
        }
        for (int i = 0; i < 24; ++i) {
            residual_data[i] = 0.0f;
        }
        // Residual 2.0 on the red channel of pixel (0, 0) only.
        residual_data[0] = 2.0f;
        Frame frame = { frame_data, 2, 3, 9 };
        Batch residual = { residual_data, 3, 2, 3, 8, 4 };
        CHECK(compose(frame_view(frame), view(residual), 0.5, out, 18) == Status::kOk);
        // Pixel (0, 0): frame = (0, 0.2, 0.4); red network = clamp(0 + 0.5) = 0.5,
        // out red = 0 + 0.5 * 0.5 = 0.25; green/blue unchanged.
        CHECK(out[0] == 0.25f);
        CHECK(out[1] == 0.2f);
        CHECK(out[2] == 0.4f);
        // The remaining pixels keep their frame colours (zero residual, intensity
        // 0.5), independently of the padded stride.
        for (int p = 1; p < 6; ++p) {
            for (int c = 0; c < 3; ++c) {
                CHECK(out[p * 3 + c] == frame_data[p * 3 + c]);
            }
        }
    }
    // Intensity > 1 through the batch path equals the pixel path.
    {
        float frame_data[6] = { 0.1f, 0.2f, 0.3f, 0.9f, 0.5f, 0.25f };
        float residual_data[6] = { 1.0f, -0.5f, 0.25f, -2.0f, 0.75f, 1.5f };
        float out[6];
        float expected[6];
        Frame frame = { frame_data, 2, 1, 3 };
        Batch residual = { residual_data, 3, 2, 1, 2, 1 };
        CHECK(compose(frame_view(frame), view(residual), 2.5, out, 6) == Status::kOk);
        for (int y = 0; y < 2; ++y) {
            float pixel_frame[3];
            float pixel_residual[3];
            float pixel_out[3];
            for (int c = 0; c < 3; ++c) {
                pixel_frame[c] = frame_data[y * 3 + c];
                pixel_residual[c] = residual_data[c * 2 + y];
            }
            dlss5_nr::compose_pixel(pixel_frame, pixel_residual, 2.5, pixel_out);
            oracle_compose_pixel(pixel_frame, pixel_residual, 2.5, expected);
            for (int c = 0; c < 3; ++c) {
                CHECK(out[y * 3 + c] == pixel_out[c]);
                CHECK(same_value(out[y * 3 + c], expected[c]));
            }
        }
    }
}

// ---- negative controls ------------------------------------------------------------------------------
static void check_negative(void) {
    using dlss5_nr::project_residual;
    using dlss5_nr::compose;
    using dlss5_nr::Status;
    float weights[96];
    fill_weights_identity(weights);
    float wide_data[32 * 4] = {};
    float project_out[12];
    Batch wide = { wide_data, 32, 2, 2, 4, 2 };
    {
        Batch t = wide; t.data = nullptr; wide = t;
        CHECK(project_residual(view(wide), weights, 96, project_out, 12)
            == Status::kNullPointer);
        wide = { wide_data, 32, 2, 2, 4, 2 };
    }
    CHECK(project_residual(view(wide), nullptr, 96, project_out, 12)
        == Status::kNullPointer);
    CHECK(project_residual(view(wide), weights, 96, nullptr, 12)
        == Status::kNullPointer);
    {
        Batch t = wide; t.channels = 31; wide = t;
        CHECK(project_residual(view(wide), weights, 96, project_out, 12)
            == Status::kInvalidDimensions);
        wide = { wide_data, 32, 2, 2, 4, 2 };
        t = wide; t.rows = 0; wide = t;
        CHECK(project_residual(view(wide), weights, 96, project_out, 12)
            == Status::kInvalidDimensions);
        wide = { wide_data, 32, 2, 2, 4, 2 };
        t = wide; t.cols = dlss5_nr::kMaxCols + 1; wide = t;
        CHECK(project_residual(view(wide), weights, 96, project_out, 12)
            == Status::kInvalidDimensions);
        wide = { wide_data, 32, 2, 2, 4, 2 };
    }
    {
        // rows/cols within bounds but their product above kMaxPixels.
        Batch t = wide; t.rows = dlss5_nr::kMaxRows; t.cols = dlss5_nr::kMaxCols; wide = t;
        CHECK(project_residual(view(wide), weights, 96, project_out, 12)
            == Status::kInvalidDimensions);
        wide = { wide_data, 32, 2, 2, 4, 2 };
    }
    {
        Batch t = wide; t.row_stride = 1; wide = t;
        CHECK(project_residual(view(wide), weights, 96, project_out, 12)
            == Status::kInvalidStride);
        wide = { wide_data, 32, 2, 2, 4, 2 };
        t = wide; t.row_stride = dlss5_nr::kMaxCols + 1; wide = t;
        CHECK(project_residual(view(wide), weights, 96, project_out, 12)
            == Status::kInvalidStride);
        wide = { wide_data, 32, 2, 2, 4, 2 };
        t = wide; t.plane_stride = 3; wide = t;
        CHECK(project_residual(view(wide), weights, 96, project_out, 12)
            == Status::kInvalidStride);
        wide = { wide_data, 32, 2, 2, 4, 2 };
        t = wide;
        t.plane_stride = (ptrdiff_t)dlss5_nr::kMaxRows * dlss5_nr::kMaxCols + 1;
        wide = t;
        CHECK(project_residual(view(wide), weights, 96, project_out, 12)
            == Status::kInvalidStride);
        wide = { wide_data, 32, 2, 2, 4, 2 };
    }
    CHECK(project_residual(view(wide), weights, 95, project_out, 12)
        == Status::kInvalidBuffer);
    {
        float bad[96];
        fill_weights_identity(bad);
        bad[5] = 0.1f;  // not f16-representable
        CHECK(project_residual(view(wide), bad, 96, project_out, 12)
            == Status::kInvalidValues);
        fill_weights_identity(bad);
        bad[7] = INFINITY;
        CHECK(project_residual(view(wide), bad, 96, project_out, 12)
            == Status::kInvalidValues);
        fill_weights_identity(bad);
        bad[9] = NAN;
        CHECK(project_residual(view(wide), bad, 96, project_out, 12)
            == Status::kInvalidValues);
    }
    {
        fill_sentinel(project_out, 12);
        CHECK(project_residual(view(wide), weights, 96, project_out, 11)
            == Status::kInvalidBuffer);
        CHECK(project_out[0] == kSentinel && project_out[11] == kSentinel);
    }
    {
        wide_data[13] = NAN;
        fill_sentinel(project_out, 12);
        CHECK(project_residual(view(wide), weights, 96, project_out, 12)
            == Status::kInvalidValues);
        CHECK(project_out[0] == kSentinel && project_out[11] == kSentinel);
        wide_data[13] = 0.0f;
    }

    // compose negatives.
    float frame_data[2 * 2 * 3];
    float residual_data[3 * 2 * 2];
    float out[12];
    for (int i = 0; i < 12; ++i) {
        frame_data[i] = 0.5f;
        residual_data[i] = 0.0f;
    }
    Frame frame = { frame_data, 2, 2, 6 };
    Batch residual = { residual_data, 3, 2, 2, 4, 2 };
    {
        Frame t = frame; t.rgb = nullptr; frame = t;
        CHECK(compose(frame_view(frame), view(residual), 1.0, out, 12)
            == Status::kNullPointer);
        frame = { frame_data, 2, 2, 6 };
    }
    {
        Batch t = residual; t.data = nullptr; residual = t;
        CHECK(compose(frame_view(frame), view(residual), 1.0, out, 12)
            == Status::kNullPointer);
        residual = { residual_data, 3, 2, 2, 4, 2 };
    }
    CHECK(compose(frame_view(frame), view(residual), NAN, out, 12)
        == Status::kInvalidControls);
    CHECK(compose(frame_view(frame), view(residual), INFINITY, out, 12)
        == Status::kInvalidControls);
    CHECK(compose(frame_view(frame), view(residual), 1.0, nullptr, 12)
        == Status::kNullPointer);
    {
        Batch t = residual; t.channels = 2; residual = t;
        CHECK(compose(frame_view(frame), view(residual), 1.0, out, 12)
            == Status::kInvalidDimensions);
        residual = { residual_data, 3, 2, 2, 4, 2 };
        t = residual; t.rows = 3; residual = t;
        CHECK(compose(frame_view(frame), view(residual), 1.0, out, 12)
            == Status::kInvalidDimensions);
        residual = { residual_data, 3, 2, 2, 4, 2 };
    }
    {
        Frame t = frame; t.rows = 0; frame = t;
        CHECK(compose(frame_view(frame), view(residual), 1.0, out, 12)
            == Status::kInvalidDimensions);
        frame = { frame_data, 2, 2, 6 };
        t = frame; t.cols = dlss5_nr::kMaxCols + 1; frame = t;
        CHECK(compose(frame_view(frame), view(residual), 1.0, out, 12)
            == Status::kInvalidDimensions);
        frame = { frame_data, 2, 2, 6 };
    }
    {
        // rows/cols within bounds but their product above kMaxPixels (frame and
        // residual agree, so the pixel-count check is what rejects).
        Frame t = frame; t.rows = dlss5_nr::kMaxRows; t.cols = dlss5_nr::kMaxCols;
        frame = t;
        Batch r = residual;
        r.rows = dlss5_nr::kMaxRows; r.cols = dlss5_nr::kMaxCols;
        residual = r;
        CHECK(compose(frame_view(frame), view(residual), 1.0, out, 12)
            == Status::kInvalidDimensions);
        frame = { frame_data, 2, 2, 6 };
        residual = { residual_data, 3, 2, 2, 4, 2 };
    }
    {
        Frame t = frame; t.row_stride = 5; frame = t;
        CHECK(compose(frame_view(frame), view(residual), 1.0, out, 12)
            == Status::kInvalidStride);
        frame = { frame_data, 2, 2, 6 };
        t = frame; t.row_stride = 3 * dlss5_nr::kMaxCols + 1; frame = t;
        CHECK(compose(frame_view(frame), view(residual), 1.0, out, 12)
            == Status::kInvalidStride);
        frame = { frame_data, 2, 2, 6 };
    }
    {
        Batch t = residual; t.row_stride = 1; residual = t;
        CHECK(compose(frame_view(frame), view(residual), 1.0, out, 12)
            == Status::kInvalidStride);
        residual = { residual_data, 3, 2, 2, 4, 2 };
        t = residual; t.row_stride = dlss5_nr::kMaxCols + 1; residual = t;
        CHECK(compose(frame_view(frame), view(residual), 1.0, out, 12)
            == Status::kInvalidStride);
        residual = { residual_data, 3, 2, 2, 4, 2 };
        t = residual; t.plane_stride = 3; residual = t;
        CHECK(compose(frame_view(frame), view(residual), 1.0, out, 12)
            == Status::kInvalidStride);
        residual = { residual_data, 3, 2, 2, 4, 2 };
        t = residual;
        t.plane_stride = (ptrdiff_t)dlss5_nr::kMaxRows * dlss5_nr::kMaxCols + 1;
        residual = t;
        CHECK(compose(frame_view(frame), view(residual), 1.0, out, 12)
            == Status::kInvalidStride);
        residual = { residual_data, 3, 2, 2, 4, 2 };
    }
    {
        fill_sentinel(out, 12);
        CHECK(compose(frame_view(frame), view(residual), 1.0, out, 11)
            == Status::kInvalidBuffer);
        CHECK(out[0] == kSentinel && out[11] == kSentinel);
    }
    {
        frame_data[4] = 1.5f;  // out of [0, 1]
        fill_sentinel(out, 12);
        CHECK(compose(frame_view(frame), view(residual), 1.0, out, 12)
            == Status::kInvalidValues);
        CHECK(out[0] == kSentinel && out[11] == kSentinel);
        frame_data[4] = -0.25f;
        CHECK(compose(frame_view(frame), view(residual), 1.0, out, 12)
            == Status::kInvalidValues);
        frame_data[4] = NAN;
        CHECK(compose(frame_view(frame), view(residual), 1.0, out, 12)
            == Status::kInvalidValues);
        frame_data[4] = 0.5f;
    }
    {
        residual_data[9] = INFINITY;
        fill_sentinel(out, 12);
        CHECK(compose(frame_view(frame), view(residual), 1.0, out, 12)
            == Status::kInvalidValues);
        CHECK(out[0] == kSentinel && out[11] == kSentinel);
        residual_data[9] = 0.0f;
    }
}

// ---- allocation discipline --------------------------------------------------------------------------
static void check_allocation_discipline(void) {
    using dlss5_nr::compose;
    using dlss5_nr::project_residual;
    using dlss5_nr::Status;
    float weights[96];
    fill_weights_identity(weights);
    float wide_data[32] = {};
    float project_out[3];
    float frame_data[3] = { 0.5f, 0.5f, 0.5f };
    float residual_data[3] = { 0.0f, 0.0f, 0.0f };
    float out[3];
    Batch wide = { wide_data, 32, 1, 1, 1, 1 };
    Frame frame = { frame_data, 1, 1, 3 };
    Batch residual = { residual_data, 3, 1, 1, 1, 1 };
    const unsigned long long before = g_allocations.load();
    CHECK(project_residual(view(wide), weights, 96, project_out, 3) == Status::kOk);
    CHECK(compose(frame_view(frame), view(residual), 1.5, out, 3) == Status::kOk);
    CHECK(g_allocations.load() == before);
}

int main(void) {
    check_compose_literals();
    check_compose_oracle();
    check_project_literals();
    check_project_oracle();
    check_compose_batch();
    check_negative();
    check_allocation_discipline();
    printf("native-compose-test: %llu checks passed\n", g_checks);
    return 0;
}
