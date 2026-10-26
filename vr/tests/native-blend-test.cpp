// Independent tests for the native post-block blend stage (vr/native-blend.hpp): the
// double -> f16 rounding boundary, the elementwise f16 mix and the 2x replication with
// replicate padding. CPU-only. Synthetic data and weights are original; passing
// establishes synthetic arithmetic correctness of this implementation against
// independently derived controls only — never vendor correspondence.
//
// Independence: the mix/rounding oracles use the compiler's _Float16 conversion path and
// double arithmetic (sharing no code with the production bit-twiddling); the blend
// oracle independently re-derives the same closed-form spatial reduction as production
// (it does not materialise the reference's expansion/pad/crop intermediates, so that
// reduction is instead pinned by the hand-computed crop, replicate-pad and dual-clamp
// known-answer cases), and the known-answer
// literals are hand-derived from the IEEE 754 binary16 definition.

#include "../native-blend.hpp"

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

// ---- independent oracles -------------------------------------------------------------------------
// Compiler _Float16 conversion (verified IEEE RNE, overflow to infinity) and double
// arithmetic: no shared code with the production software conversions.
static float oracle_f16d(double value) {
    return (float)(_Float16)value;
}

static float oracle_mix(float up, float skip, float main_gain, float skip_gain) {
    float main = (float)(_Float16)(up * main_gain);
    double fused = (double)skip * (double)skip_gain + (double)main;
    return (float)(_Float16)fused;
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

// ---- f16_from_double ------------------------------------------------------------------------------
static void check_f16d_literals(void) {
    using dlss5_nr::f16_from_double;
    CHECK(f16_from_double(0.0) == 0.0f);
    CHECK(!signbit(f16_from_double(0.0)));
    CHECK(f16_from_double(-0.0) == 0.0f);
    CHECK(signbit(f16_from_double(-0.0)));
    CHECK(f16_from_double(1.0) == 1.0f);
    CHECK(f16_from_double(-2.0) == -2.0f);
    CHECK(f16_from_double(65504.0) == 65504.0f);
    CHECK(f16_from_double(65512.0) == 65504.0f);          // (65504, 65520) -> 65504
    CHECK(f16_from_double(65519.999999999) == 65504.0f);  // guard: float32 would round up
    CHECK(isinf(f16_from_double(65520.0)));               // the tie overflows to infinity
    CHECK(f16_from_double(65520.0) > 0.0f);
    CHECK(isinf(f16_from_double(-65520.0)));
    CHECK(f16_from_double(-65520.0) < 0.0f);
    CHECK(f16_from_double(32759.0) == 32752.0f);          // below the 32760 midpoint
    CHECK(f16_from_double(32760.0) == 32768.0f);          // binade-top tie -> even carry
    CHECK(f16_from_double(32761.0) == 32768.0f);          // rounding carry to next binade
    CHECK(isinf(f16_from_double(1e300)));
    CHECK(isinf(f16_from_double(-1e300)));
    CHECK(f16_from_double(0x1.8p-25) == 0x1p-24f);        // 1.5 * 2^-25: above midpoint
    CHECK(f16_from_double(0x1p-26) == 0.0f);              // below the midpoint tie
    CHECK(f16_from_double(0x1p-25) == 0.0f);              // 2^-25: midpoint tie -> even zero
    CHECK(!signbit(f16_from_double(0x1p-25)));
    CHECK(f16_from_double(-0x1p-25) == 0.0f);
    CHECK(signbit(f16_from_double(-0x1p-25)));
    CHECK(f16_from_double(0x1p-24) == 0x1p-24f);          // exact subnormal step
    CHECK(isnan(f16_from_double((double)NAN)));
}

static void check_f16d_oracle(void) {
    using dlss5_nr::f16_from_double;
    // Targeted neighbourhoods around the guard boundaries, the binade-top carry and
    // the subnormal tie.
    for (int k = -4; k <= 4; ++k) {
        double d = 65504.0 + (double)k * 0.001;
        CHECK(same_value(f16_from_double(d), oracle_f16d(d)));
        d = 65520.0 + (double)k * 0.001;
        CHECK(same_value(f16_from_double(d), oracle_f16d(d)));
        d = 32760.0 + (double)k * 0.001;
        CHECK(same_value(f16_from_double(d), oracle_f16d(d)));
        d = 0x1p-25 * (1.0 + (double)k * 0x0.000000000001p0);
        CHECK(same_value(f16_from_double(d), oracle_f16d(d)));
    }
    // Wide fuzz over random double bit patterns (NaN/inf included; compared as values).
    for (int i = 0; i < 200000; ++i) {
        uint64_t bits = ((uint64_t)rng() << 32) | rng();
        double d = 0.0;
        memcpy(&d, &bits, sizeof d);
        if (!same_value(f16_from_double(d), oracle_f16d(d))) {
            CHECK(false);
        }
    }
    ++g_checks;
}

// ---- mix_element ------------------------------------------------------------------------------------
static void check_mix_literals(void) {
    using dlss5_nr::mix_element;
    // Basic exact path.
    CHECK(mix_element(1.5f, 0.5f, 2.0f, 4.0f) == 5.0f);
    CHECK(mix_element(0.0f, 0.0f, 1.0f, 1.0f) == 0.0f);
    CHECK(mix_element(-1.5f, -2.0f, 2.0f, 0.5f) == -4.0f);
    // Product pre-rounding discriminator: up * main_gain = 4105 * 2^-12 sits a quarter
    // ulp above the binary16 grid point 1 + 2^-9 and rounds down to it; adding the skip
    // term 2 * 2^-12 then lands exactly on the midpoint 1 + 2.5 * 2^-10, which ties to
    // even down to 1 + 2^-9 = 1.001953125. Without the product's f16 pre-rounding the
    // sum would be 4107 * 2^-12, above that midpoint, rounding up to 1.0029296875.
    CHECK(mix_element(821.0f / 512.0f, 2.0f * 0x1p-12f, 0.625f, 1.0f) == 1.001953125f);
    // Sum-rounding literal (the pre-rounding is not observable in the result):
    // (1 + 2^-10)^2 = 1 + 2^-9 + 2^-20 pre-rounds to 1 + 2^-9; fusing the exact skip
    // product gives 2 + 2^-8 + 2^-20, which rounds to 2 + 2^-8 = 2.00390625 either way.
    CHECK(mix_element(1.0009765625f, 1.0009765625f, 1.0009765625f, 1.0009765625f)
        == 2.00390625f);
    // Subnormal tie: skip * 0.5 = 3 * 2^-25, the midpoint on the 2^-24 grid, ties to
    // the even neighbour 2 * 2^-24 = 2^-23.
    CHECK(mix_element(0.0f, 3.0f * 0x1p-24f, 1.0f, 0.5f) == 0x1p-23f);
    // Overflow boundary of the fused sum (exact products).
    CHECK(mix_element(0.0f, 32752.0f, 1.0f, 2.0f) == 65504.0f);   // exactly 65504
    CHECK(mix_element(0.0f, 32756.0f, 1.0f, 2.0f) == 65504.0f);   // 65512 -> 65504
    CHECK(isinf(mix_element(0.0f, 32760.0f, 1.0f, 2.0f)));        // 65520 tie -> infinity
    CHECK(mix_element(0.0f, 32760.0f, 1.0f, 2.0f) > 0.0f);
    CHECK(isinf(mix_element(0.0f, -32760.0f, 1.0f, 2.0f)));
    CHECK(mix_element(0.0f, -32760.0f, 1.0f, 2.0f) < 0.0f);
    // Main-product overflow: f16(65504 * 2) = inf poisons the fused sum.
    CHECK(isinf(mix_element(65504.0f, 0.0f, 2.0f, 1.0f)));
    // NaN policy: canonical quiet NaN, sign dropped.
    CHECK(isnan(mix_element((float)NAN, 1.0f, 1.0f, 1.0f)));
    CHECK(isnan(mix_element(1.0f, (float)NAN, 1.0f, 1.0f)));
}

static void check_mix_oracle(void) {
    using dlss5_nr::mix_element;
    for (int i = 0; i < 200000; ++i) {
        float up = random_f16();
        float skip = random_f16();
        float mg = random_f16();
        float sg = random_f16();
        if (!same_value(mix_element(up, skip, mg, sg), oracle_mix(up, skip, mg, sg))) {
            CHECK(false);
        }
    }
    ++g_checks;
}

// ---- blend ------------------------------------------------------------------------------------------
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

// Read a batch's logical element regardless of padding.
static float at(const Batch& b, int c, int y, int x) {
    return b.data[(size_t)c * (size_t)b.plane_stride + (size_t)y * (size_t)b.row_stride + x];
}

// The reference's three explicit steps (2x2 expansion, replicate padding to the skip
// extent, crop) reduce to the same closed form production uses; this oracle re-derives
// that reduction rather than sharing production code, and the reduction itself is
// pinned by the hand-computed known answers in check_blend_known_answers.
static void oracle_blend(const Batch& up, const Batch& skip,
                         const float* main_gain, const float* skip_gain, float* out) {
    for (int c = 0; c < up.channels; ++c) {
        for (int y = 0; y < skip.rows; ++y) {
            for (int x = 0; x < skip.cols; ++x) {
                int sy = y / 2;
                int sx = x / 2;
                if (sy >= up.rows) {
                    sy = up.rows - 1;  // replicate-pad repeats the last row
                }
                if (sx >= up.cols) {
                    sx = up.cols - 1;  // replicate-pad repeats the last column
                }
                out[(size_t)c * (size_t)skip.rows * skip.cols + (size_t)y * skip.cols + x]
                    = oracle_mix(at(up, c, sy, sx), at(skip, c, y, x), main_gain[c], skip_gain[c]);
            }
        }
    }
}

static void check_blend_known_answers(void) {
    using dlss5_nr::blend;
    using dlss5_nr::Status;
    const float one = 1.0f;
    // 1x1 up replicated over a 2x2 skip.
    {
        float up_data[1] = { 2.0f };
        float skip_data[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        float out[4];
        Batch up = { up_data, 1, 1, 1, 1, 1 };
        Batch skip = { skip_data, 1, 2, 2, 4, 2 };
        CHECK(blend(view(up), view(skip), &one, &one, 1, out, 4) == Status::kOk);
        CHECK(out[0] == 2.0f && out[1] == 2.0f && out[2] == 2.0f && out[3] == 2.0f);
    }
    // Exact 2x: 2x2 up over a 4x4 skip, 2x2 block replication.
    {
        float up_data[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
        float skip_data[16] = {};
        float out[16];
        Batch up = { up_data, 1, 2, 2, 4, 2 };
        Batch skip = { skip_data, 1, 4, 4, 16, 4 };
        CHECK(blend(view(up), view(skip), &one, &one, 1, out, 16) == Status::kOk);
        const float expected[16] = {
            1.0f, 1.0f, 2.0f, 2.0f,
            1.0f, 1.0f, 2.0f, 2.0f,
            3.0f, 3.0f, 4.0f, 4.0f,
            3.0f, 3.0f, 4.0f, 4.0f,
        };
        for (int i = 0; i < 16; ++i) {
            CHECK(out[i] == expected[i]);
        }
    }
    // Crop: 3x3 up over a 4x4 skip (the third source row/column is cropped away).
    {
        float up_data[9] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };
        float skip_data[16] = {};
        float out[16];
        Batch up = { up_data, 1, 3, 3, 9, 3 };
        Batch skip = { skip_data, 1, 4, 4, 16, 4 };
        CHECK(blend(view(up), view(skip), &one, &one, 1, out, 16) == Status::kOk);
        const float expected[16] = {
            1.0f, 1.0f, 2.0f, 2.0f,
            1.0f, 1.0f, 2.0f, 2.0f,
            4.0f, 4.0f, 5.0f, 5.0f,
            4.0f, 4.0f, 5.0f, 5.0f,
        };
        for (int i = 0; i < 16; ++i) {
            CHECK(out[i] == expected[i]);
        }
    }
    // Replicate pad + crop: 1x2 up over a 5x3 skip (rows repeated, right column cropped).
    {
        float up_data[2] = { 5.0f, 7.0f };
        float skip_data[15] = {};
        float out[15];
        Batch up = { up_data, 1, 1, 2, 2, 2 };
        Batch skip = { skip_data, 1, 5, 3, 15, 3 };
        CHECK(blend(view(up), view(skip), &one, &one, 1, out, 15) == Status::kOk);
        for (int y = 0; y < 5; ++y) {
            CHECK(out[y * 3] == 5.0f);
            CHECK(out[y * 3 + 1] == 5.0f);
            CHECK(out[y * 3 + 2] == 7.0f);
        }
    }
    // Dual clamp: 2x2 up over a 6x5 skip. Both clamps fire non-degenerately — output
    // rows y >= 4 must repeat the LAST up row (3.5/4.5 half-values, not the first row's
    // 1.5/2.5) and column x = 4 must repeat the LAST up column (2.5/4.5, not 1.5/3.5).
    // mix = f16(skip 0.5 + f16(up * 1)) = up + 0.5, exact on the f16 grid.
    {
        float up_data[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
        float skip_data[30];
        float out[30];
        for (int i = 0; i < 30; ++i) {
            skip_data[i] = 0.5f;
        }
        Batch up = { up_data, 1, 2, 2, 4, 2 };
        Batch skip = { skip_data, 1, 6, 5, 30, 5 };
        CHECK(blend(view(up), view(skip), &one, &one, 1, out, 30) == Status::kOk);
        const float expected[30] = {
            1.5f, 1.5f, 2.5f, 2.5f, 2.5f,
            1.5f, 1.5f, 2.5f, 2.5f, 2.5f,
            3.5f, 3.5f, 4.5f, 4.5f, 4.5f,
            3.5f, 3.5f, 4.5f, 4.5f, 4.5f,
            3.5f, 3.5f, 4.5f, 4.5f, 4.5f,
            3.5f, 3.5f, 4.5f, 4.5f, 4.5f,
        };
        for (int i = 0; i < 30; ++i) {
            CHECK(out[i] == expected[i]);
        }
    }
    // Gains through the spatial path, two channels with distinct gains (4x4 skip so
    // the 2x2 up is fully consumed).
    {
        float up_data[8] = {
            1.0f, 2.0f, 3.0f, 4.0f,      // channel 0
            1.0f, 1.0f, 1.0f, 1.0f,      // channel 1
        };
        float skip_data[32];
        float mg[2] = { 2.0f, 0.0f };
        float sg[2] = { 1.0f, 3.0f };
        float out[32];
        for (int i = 0; i < 16; ++i) {
            skip_data[i] = 0.5f;
            skip_data[16 + i] = 2.0f;
        }
        Batch up = { up_data, 2, 2, 2, 4, 2 };
        Batch skip = { skip_data, 2, 4, 4, 16, 4 };
        CHECK(blend(view(up), view(skip), mg, sg, 2, out, 32) == Status::kOk);
        const float expected0[16] = {
            2.5f, 2.5f, 4.5f, 4.5f,
            2.5f, 2.5f, 4.5f, 4.5f,
            6.5f, 6.5f, 8.5f, 8.5f,
            6.5f, 6.5f, 8.5f, 8.5f,
        };
        for (int i = 0; i < 16; ++i) {
            CHECK(out[i] == expected0[i]);
            CHECK(out[16 + i] == 6.0f);   // channel 1: main 0 + skip 2 * gain 3
        }
    }
    // The product pre-rounding discriminator through the full stage.
    {
        float up_data[1] = { 821.0f / 512.0f };
        float skip_data[1] = { 2.0f * 0x1p-12f };
        float mg[1] = { 0.625f };
        float sg[1] = { 1.0f };
        float out[1];
        Batch up = { up_data, 1, 1, 1, 1, 1 };
        Batch skip = { skip_data, 1, 1, 1, 1, 1 };
        CHECK(blend(view(up), view(skip), mg, sg, 1, out, 1) == Status::kOk);
        CHECK(out[0] == 1.001953125f);
    }
}

static void check_blend_oracle_and_layout(void) {
    using dlss5_nr::blend;
    using dlss5_nr::Status;
    float up_data[4 * 8 * 8];
    float skip_data[4 * 8 * 8];
    float out[4 * 8 * 8];
    float out2[4 * 8 * 8];
    for (int iter = 0; iter < 2000; ++iter) {
        const int channels = 1 + (int)(rng() % 4);
        const int up_rows = 1 + (int)(rng() % 5);
        const int up_cols = 1 + (int)(rng() % 5);
        const int skip_rows = 1 + (int)(rng() % 6);
        const int skip_cols = 1 + (int)(rng() % 6);
        const int up_row_stride = up_cols + (int)(rng() % 3);
        const int up_plane_stride = up_rows * up_row_stride + (int)(rng() % 4);
        const int skip_row_stride = skip_cols + (int)(rng() % 3);
        const int skip_plane_stride = skip_rows * skip_row_stride + (int)(rng() % 4);
        Batch up = { up_data, channels, up_rows, up_cols, up_plane_stride, up_row_stride };
        Batch skip = { skip_data, channels, skip_rows, skip_cols, skip_plane_stride,
            skip_row_stride };
        float mg[4];
        float sg[4];
        for (int c = 0; c < channels; ++c) {
            mg[c] = random_f16();
            sg[c] = random_f16();
            if (!std::isfinite(mg[c]) || !dlss5_nr::is_f16(mg[c])) {
                mg[c] = 1.0f;
            }
            if (!std::isfinite(sg[c]) || !dlss5_nr::is_f16(sg[c])) {
                sg[c] = 1.0f;
            }
        }
        for (int c = 0; c < channels; ++c) {
            for (int y = 0; y < up_rows; ++y) {
                for (int x = 0; x < up_cols; ++x) {
                    up_data[(size_t)c * up_plane_stride + (size_t)y * up_row_stride + x]
                        = random_finite_f16();
                }
            }
            for (int y = 0; y < skip_rows; ++y) {
                for (int x = 0; x < skip_cols; ++x) {
                    skip_data[(size_t)c * skip_plane_stride + (size_t)y * skip_row_stride + x]
                        = random_finite_f16();
                }
            }
        }
        const size_t out_count = (size_t)channels * skip_rows * skip_cols;
        // Determinism: two runs must agree bit for bit.
        if (blend(view(up), view(skip), mg, sg, (size_t)channels, out, out_count) != Status::kOk
            || blend(view(up), view(skip), mg, sg, (size_t)channels, out2, out_count) != Status::kOk
            || memcmp(out, out2, out_count * sizeof(float)) != 0) {
            CHECK(false);
            continue;
        }
        float expected[4 * 8 * 8];
        oracle_blend(up, skip, mg, sg, expected);
        if (memcmp(out, expected, out_count * sizeof(float)) != 0) {
            CHECK(false);
        }
    }
    ++g_checks;
}

static void check_blend_permutation(void) {
    using dlss5_nr::blend;
    using dlss5_nr::Status;
    // 32 channels, channel-coded inputs and distinct per-channel gains: no channel may
    // observe another channel's inputs or gains.
    float up_data[32 * 2 * 2];
    float skip_data[32 * 2 * 2];
    float out[32 * 2 * 2];
    float mg[32];
    float sg[32];
    for (int c = 0; c < 32; ++c) {
        mg[c] = (float)(c + 1);
        sg[c] = (float)(2 * (c + 1));
        for (int y = 0; y < 2; ++y) {
            for (int x = 0; x < 2; ++x) {
                up_data[c * 4 + y * 2 + x] = (float)(c + 1) + 0.25f * (float)(y * 2 + x);
                skip_data[c * 4 + y * 2 + x] = 0.5f * (float)(c + 1);
            }
        }
    }
    Batch up = { up_data, 32, 2, 2, 4, 2 };
    Batch skip = { skip_data, 32, 2, 2, 4, 2 };
    CHECK(blend(view(up), view(skip), mg, sg, 32, out, 32 * 4) == Status::kOk);
    for (int c = 0; c < 32; ++c) {
        for (int i = 0; i < 4; ++i) {
            // A 2x2 skip consumes only up[0][0] of the half-resolution input.
            CHECK(out[c * 4 + i]
                == oracle_mix(up_data[c * 4], skip_data[c * 4 + i], mg[c], sg[c]));
        }
    }
}

static void check_blend_sensitivity_and_stride(void) {
    using dlss5_nr::blend;
    using dlss5_nr::Status;
    const float one = 1.0f;
    float up_data[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float skip_data[16] = {};
    float base[16];
    float changed[16];
    Batch up = { up_data, 1, 2, 2, 4, 2 };
    Batch skip = { skip_data, 1, 4, 4, 16, 4 };
    CHECK(blend(view(up), view(skip), &one, &one, 1, base, 16) == Status::kOk);
    // One up element feeds exactly its 2x2 replicated block.
    up_data[0] = 9.0f;
    CHECK(blend(view(up), view(skip), &one, &one, 1, changed, 16) == Status::kOk);
    int diffs = 0;
    for (int i = 0; i < 16; ++i) {
        if (changed[i] != base[i]) {
            ++diffs;
        }
    }
    CHECK(diffs == 4);
    CHECK(changed[0] == 9.0f && changed[1] == 9.0f && changed[4] == 9.0f
        && changed[5] == 9.0f);
    up_data[0] = 1.0f;
    // One skip element feeds exactly its own output.
    skip_data[11] = 2.5f;
    CHECK(blend(view(up), view(skip), &one, &one, 1, changed, 16) == Status::kOk);
    diffs = 0;
    for (int i = 0; i < 16; ++i) {
        if (changed[i] != base[i]) {
            ++diffs;
        }
    }
    CHECK(diffs == 1);
    CHECK(changed[11] != base[11]);
    skip_data[11] = 0.0f;
    // Stride independence: padded layouts must equal the tight layout.
    {
        float up_padded[2 * 3] = { 1.0f, 2.0f, -1.0f, 3.0f, 4.0f, -1.0f };
        float skip_padded[2 * 3] = { 0.5f, 0.25f, -1.0f, 1.0f, 2.0f, -1.0f };
        float out_tight[4];
        float out_padded[4];
        Batch up_t = { up_data, 1, 2, 2, 4, 2 };
        Batch up_p = { up_padded, 1, 2, 2, 6, 3 };
        Batch skip_t = { skip_data, 1, 2, 2, 4, 2 };
        float skip_tight[4] = { 0.5f, 0.25f, 1.0f, 2.0f };
        skip_t.data = skip_tight;
        Batch skip_p = { skip_padded, 1, 2, 2, 6, 3 };
        CHECK(blend(view(up_t), view(skip_t), &one, &one, 1, out_tight, 4) == Status::kOk);
        CHECK(blend(view(up_p), view(skip_p), &one, &one, 1, out_padded, 4) == Status::kOk);
        for (int i = 0; i < 4; ++i) {
            CHECK(out_tight[i] == out_padded[i]);
        }
    }
}

static void check_blend_negative(void) {
    using dlss5_nr::blend;
    using dlss5_nr::FeatureBatchView;
    using dlss5_nr::Status;
    float up_data[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float skip_data[4] = { 0.5f, 0.5f, 0.5f, 0.5f };
    float mg[1] = { 2.0f };
    float sg[1] = { 1.0f };
    float out[4];
    Batch up = { up_data, 1, 2, 2, 4, 2 };
    Batch skip = { skip_data, 1, 2, 2, 4, 2 };
    // Every rejected call must leave the output untouched.
    #define EXPECT_REJECT(expected_status) do { \
        fill_sentinel(out, 4); \
        CHECK(blend(view(up), view(skip), mg, sg, 1, out, 4) == (expected_status)); \
        CHECK(out[0] == kSentinel && out[1] == kSentinel \
            && out[2] == kSentinel && out[3] == kSentinel); \
    } while (0)
    {
        Batch t = up; t.data = nullptr; up = t;
        EXPECT_REJECT(Status::kNullPointer);
        up = { up_data, 1, 2, 2, 4, 2 };
    }
    {
        Batch t = skip; t.data = nullptr; skip = t;
        EXPECT_REJECT(Status::kNullPointer);
        skip = { skip_data, 1, 2, 2, 4, 2 };
    }
    {
        fill_sentinel(out, 4);
        CHECK(blend(view(up), view(skip), nullptr, sg, 1, out, 4) == Status::kNullPointer);
        CHECK(blend(view(up), view(skip), mg, nullptr, 1, out, 4) == Status::kNullPointer);
        CHECK(out[0] == kSentinel && out[3] == kSentinel);
    }
    {
        Batch u = up; u.channels = 0; Batch s = skip; s.channels = 0;
        fill_sentinel(out, 4);
        CHECK(blend(view(u), view(s), mg, sg, 0, out, 4) == Status::kInvalidDimensions);
        CHECK(out[0] == kSentinel);
        u.channels = 33; s.channels = 33;
        CHECK(blend(view(u), view(s), mg, sg, 33, out, 4) == Status::kInvalidDimensions);
        CHECK(out[0] == kSentinel);
        u.channels = 2; s.channels = 1;  // mismatched channel counts
        CHECK(blend(view(u), view(s), mg, sg, 1, out, 4) == Status::kInvalidDimensions);
        CHECK(out[0] == kSentinel);
    }
    {
        Batch t = up; t.rows = 0; up = t;
        EXPECT_REJECT(Status::kInvalidDimensions);
        up = { up_data, 1, 2, 2, 4, 2 };
        t = up; t.cols = -1; up = t;
        EXPECT_REJECT(Status::kInvalidDimensions);
        up = { up_data, 1, 2, 2, 4, 2 };
        t = up; t.rows = dlss5_nr::kMaxRows + 1; up = t;
        EXPECT_REJECT(Status::kInvalidDimensions);
        up = { up_data, 1, 2, 2, 4, 2 };
        t = up; t.cols = dlss5_nr::kMaxCols + 1; up = t;
        EXPECT_REJECT(Status::kInvalidDimensions);
        up = { up_data, 1, 2, 2, 4, 2 };
        t = up; t.rows = dlss5_nr::kMaxRows; t.cols = dlss5_nr::kMaxCols; up = t;
        EXPECT_REJECT(Status::kInvalidDimensions);  // rows*cols beyond kMaxPixels
        up = { up_data, 1, 2, 2, 4, 2 };
    }
    {
        Batch t = skip; t.rows = 0; skip = t;
        EXPECT_REJECT(Status::kInvalidDimensions);
        skip = { skip_data, 1, 2, 2, 4, 2 };
        t = skip; t.rows = dlss5_nr::kMaxRows; t.cols = dlss5_nr::kMaxCols; skip = t;
        EXPECT_REJECT(Status::kInvalidDimensions);
        skip = { skip_data, 1, 2, 2, 4, 2 };
    }
    {
        Batch t = up; t.row_stride = 1; up = t;
        EXPECT_REJECT(Status::kInvalidStride);
        up = { up_data, 1, 2, 2, 4, 2 };
        t = up; t.row_stride = dlss5_nr::kMaxCols + 1; up = t;
        EXPECT_REJECT(Status::kInvalidStride);
        up = { up_data, 1, 2, 2, 4, 2 };
        t = up; t.plane_stride = 3; up = t;  // below rows * row_stride
        EXPECT_REJECT(Status::kInvalidStride);
        up = { up_data, 1, 2, 2, 4, 2 };
        t = up; t.plane_stride = (ptrdiff_t)dlss5_nr::kMaxRows * dlss5_nr::kMaxCols + 1; up = t;
        EXPECT_REJECT(Status::kInvalidStride);
        up = { up_data, 1, 2, 2, 4, 2 };
    }
    {
        Batch t = skip; t.row_stride = 1; skip = t;
        EXPECT_REJECT(Status::kInvalidStride);
        skip = { skip_data, 1, 2, 2, 4, 2 };
        t = skip; t.row_stride = dlss5_nr::kMaxCols + 1; skip = t;
        EXPECT_REJECT(Status::kInvalidStride);
        skip = { skip_data, 1, 2, 2, 4, 2 };
        t = skip; t.plane_stride = 3; skip = t;
        EXPECT_REJECT(Status::kInvalidStride);
        skip = { skip_data, 1, 2, 2, 4, 2 };
        t = skip; t.plane_stride = (ptrdiff_t)dlss5_nr::kMaxRows * dlss5_nr::kMaxCols + 1;
        skip = t;
        EXPECT_REJECT(Status::kInvalidStride);
        skip = { skip_data, 1, 2, 2, 4, 2 };
    }
    {
        fill_sentinel(out, 4);
        CHECK(blend(view(up), view(skip), mg, sg, 0, out, 4) == Status::kInvalidBuffer);
        CHECK(blend(view(up), view(skip), mg, sg, 2, out, 4) == Status::kInvalidBuffer);
        CHECK(out[0] == kSentinel);
    }
    {
        float bad[1];
        bad[0] = 0.1f;  // not f16-representable
        fill_sentinel(out, 4);
        CHECK(blend(view(up), view(skip), bad, sg, 1, out, 4) == Status::kInvalidValues);
        bad[0] = INFINITY;
        CHECK(blend(view(up), view(skip), bad, sg, 1, out, 4) == Status::kInvalidValues);
        CHECK(blend(view(up), view(skip), mg, bad, 1, out, 4) == Status::kInvalidValues);
        bad[0] = NAN;
        CHECK(blend(view(up), view(skip), mg, bad, 1, out, 4) == Status::kInvalidValues);
        CHECK(out[0] == kSentinel);
    }
    {
        fill_sentinel(out, 4);
        CHECK(blend(view(up), view(skip), mg, sg, 1, nullptr, 4) == Status::kNullPointer);
        CHECK(blend(view(up), view(skip), mg, sg, 1, out, 3) == Status::kInvalidBuffer);
        CHECK(out[0] == kSentinel);
    }
    {
        float dirty[4];
        memcpy(dirty, up_data, sizeof dirty);
        dirty[2] = NAN;
        Batch u = { dirty, 1, 2, 2, 4, 2 };
        fill_sentinel(out, 4);
        CHECK(blend(view(u), view(skip), mg, sg, 1, out, 4) == Status::kInvalidValues);
        dirty[2] = -INFINITY;
        u.data = dirty;
        CHECK(blend(view(u), view(skip), mg, sg, 1, out, 4) == Status::kInvalidValues);
        memcpy(dirty, skip_data, sizeof dirty);
        dirty[0] = NAN;
        Batch s = { dirty, 1, 2, 2, 4, 2 };
        CHECK(blend(view(up), view(s), mg, sg, 1, out, 4) == Status::kInvalidValues);
        dirty[0] = INFINITY;
        s.data = dirty;
        CHECK(blend(view(up), view(s), mg, sg, 1, out, 4) == Status::kInvalidValues);
        CHECK(out[0] == kSentinel && out[3] == kSentinel);
    }
    #undef EXPECT_REJECT
}

static void check_allocation_discipline(void) {
    using dlss5_nr::blend;
    using dlss5_nr::mix_element;
    using dlss5_nr::Status;
    float up_data[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float skip_data[4] = { 0.5f, 0.5f, 0.5f, 0.5f };
    float mg[1] = { 1.0f };
    float sg[1] = { 1.0f };
    float out[4];
    Batch up = { up_data, 1, 2, 2, 4, 2 };
    Batch skip = { skip_data, 1, 2, 2, 4, 2 };
    unsigned long long before = g_allocations.load();
    CHECK(blend(view(up), view(skip), mg, sg, 1, out, 4) == Status::kOk);
    CHECK(mix_element(1.5f, 0.5f, 2.0f, 4.0f) == 5.0f);
    CHECK(dlss5_nr::f16_from_double(65512.0) == 65504.0f);
    unsigned long long after = g_allocations.load();
    CHECK(after == before);  // the whole stage is caller-allocated
}

int main(void) {
    check_f16d_literals();
    check_f16d_oracle();
    check_mix_literals();
    check_mix_oracle();
    check_blend_known_answers();
    check_blend_oracle_and_layout();
    check_blend_permutation();
    check_blend_sensitivity_and_stride();
    check_blend_negative();
    check_allocation_discipline();
    printf("native-blend-test: %llu checks passed\n", g_checks);
    return 0;
}
