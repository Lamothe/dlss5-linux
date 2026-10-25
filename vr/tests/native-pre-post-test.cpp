// Independent tests for the native pre/post stages (vr/native-pre-post.hpp): the e4m3
// rounding boundary and the 2x2 f16 pooling. CPU-only. Synthetic data and weights are
// original; passing establishes synthetic arithmetic correctness of this implementation
// against independently derived controls only — never vendor correspondence.
//
// Independence: the e4m3 rounding oracle is a brute-force nearest-code search over the
// grid decoded straight from the format definition with ldexpf (sharing no code with
// production), the f16 narrowing in the pooling oracle uses the compiler's _Float16
// conversion path, and the known-answer literals are hand-derived from the IEEE 754
// binary16 and e4m3 definitions.

#include "../native-pre-post.hpp"

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

// ---- independent oracles -------------------------------------------------------------------------
// Independent e4m3 decode from the format definition (e4m3fn: 1-4-3, bias 7, subnormal
// step 2^-9 below 2^-6, codes 0x7f/0xff are NaN), sharing no code with production.
static float oracle_e4m3_decode(uint8_t raw) {
    unsigned e = (raw >> 3) & 0xfu;
    unsigned m = raw & 7u;
    float sign = (raw & 0x80u) ? -1.0f : 1.0f;
    if (e == 0xfu && m == 7u) {
        return (float)NAN;
    }
    if (e == 0u) {
        return sign * ldexpf((float)m, -9);
    }
    return sign * ldexpf((float)(8u + m), (int)e - 10);
}

// Brute-force nearest-code e4m3 rounding: scan the whole finite grid, ties to even
// code parity (the mantissa parity), saturating at +-448; NaN maps to code 0x7f.
static uint8_t oracle_e4m3_bits(float value) {
    if (isnan(value)) {
        return 0x7fu;
    }
    if (value >= 448.0f) {
        return 0x7eu;
    }
    if (value <= -448.0f) {
        return 0xfeu;
    }
    bool negative = signbit(value) != 0;
    float magnitude = negative ? -value : value;
    uint8_t best = 0;
    float best_distance = INFINITY;
    for (uint32_t c = 0; c <= 0x7eu; ++c) {
        float candidate = oracle_e4m3_decode((uint8_t)c);
        float distance = fabsf(candidate - magnitude);
        if (distance < best_distance
            || (distance == best_distance && (c & 1u) < (best & 1u))) {
            best = (uint8_t)c;
            best_distance = distance;
        }
    }
    return (uint8_t)(negative ? (0x80u | best) : best);
}

// Independent f16 narrowing: the compiler's _Float16 conversion path.
static float oracle_narrow(float value) {
    _Float16 half = (_Float16)value;
    return (float)half;
}

// Independent pool oracle from the documented semantics: pixel pairs across the columns
// added in float32 and narrowed to f16, the two row sums added in f16, the quarter
// exact, rounded to the e4m3 grid.
static float oracle_pool_block(float a, float b, float c, float d) {
    float s0 = oracle_narrow(a + b);
    float s1 = oracle_narrow(c + d);
    float total = oracle_narrow(s0 + s1);
    return oracle_e4m3_decode(oracle_e4m3_bits(total * 0.25f));
}

// ---- synthetic data (original) ---------------------------------------------------------------------
static uint32_t g_random = 0x9e3779b9u;
static uint32_t next_random(void) {
    g_random ^= g_random << 13;
    g_random ^= g_random >> 17;
    g_random ^= g_random << 5;
    return g_random;
}
static float random_float(void) {
    // Finite float32 across a wide exponent range, occasionally exactly on the e4m3
    // or f16 grid; never NaN or infinity.
    uint32_t bits = next_random() & 0x7f7fffffu;
    if (bits >= 0x7f000000u) {
        bits -= 0x00800000u;
    }
    float value = 0.0f;
    memcpy(&value, &bits, sizeof value);
    return (next_random() & 1u) ? value : -value;
}

// ---- check sections ---------------------------------------------------------------------------------
static void check_e4m3_literals(void) {
    // Hand-derived from the e4m3fn definition (1-4-3, bias 7, 2^-9 subnormal step,
    // ties to even, saturating at +-448, 0x7f/0xff NaN).
    static const struct { float value; uint8_t bits; } encode_cases[] = {
        { 0.0f, 0x00 }, { -0.0f, 0x80 },
        { 0x1p-9f, 0x01 }, { -0x1p-9f, 0x81 },          // min subnormal
        { 0x1p-10f, 0x00 }, { -0x1p-10f, 0x80 },        // tie -> even (zero)
        { 3.0f * 0x1p-10f, 0x02 },                      // 1.5 * 2^-9 tie -> 2 * 2^-9
        { 0x1p-6f, 0x08 },                              // min normal
        { 0x1p-6f - 0x1p-10f, 0x08 },                   // 7.5 * 2^-9 tie -> 2^-6
        { 0.5f, 0x30 }, { 1.0f, 0x38 }, { -1.0f, 0xb8 }, { 2.0f, 0x40 }, { 4.0f, 0x48 },
        { 1.0625f, 0x38 },                              // tie -> even (1.0)
        { 1.1875f, 0x3a },                              // tie -> even (1.25)
        { 1.3f, 0x3a }, { 0.1f, 0x1d }, { -0.1f, 0x9d },
        { 2.5f, 0x42 }, { 3.5f, 0x46 }, { 5.5f, 0x4b }, { -2.5f, 0xc2 },
        { 256.0f, 0x78 }, { 288.0f, 0x79 }, { 416.0f, 0x7d },
        { 432.0f, 0x7e },                               // tie -> even (448)
        { 447.0f, 0x7e }, { 448.0f, 0x7e }, { -448.0f, 0xfe },
        { 449.0f, 0x7e }, { 1e9f, 0x7e },               // saturating clamp
        { INFINITY, 0x7e }, { -INFINITY, 0xfe },        // infinity saturates
    };
    for (size_t i = 0; i < sizeof encode_cases / sizeof encode_cases[0]; ++i) {
        CHECK(dlss5_nr::e4m3_bits(encode_cases[i].value) == encode_cases[i].bits);
        CHECK(oracle_e4m3_bits(encode_cases[i].value) == encode_cases[i].bits);
    }
    CHECK(dlss5_nr::e4m3_bits((float)NAN) == 0x7f);   // canonical NaN, sign dropped
    CHECK(dlss5_nr::e4m3_bits(-(float)NAN) == 0x7f);
    CHECK(dlss5_nr::is_e4m3(1.0f));
    CHECK(dlss5_nr::is_e4m3(-0.0f));
    CHECK(dlss5_nr::is_e4m3(2.5f));
    CHECK(dlss5_nr::is_e4m3(0.1015625f));
    CHECK(!dlss5_nr::is_e4m3(0.1f));
    CHECK(!dlss5_nr::is_e4m3((float)NAN));
    CHECK(!dlss5_nr::is_e4m3(450.0f));

    // Hand-derived decodes (0x7f/0xff decode to NaN).
    static const struct { uint8_t bits; float value; } decode_cases[] = {
        { 0x00, 0.0f }, { 0x01, 0x1p-9f }, { 0x07, 7.0f * 0x1p-9f },
        { 0x08, 0x1p-6f }, { 0x30, 0.5f }, { 0x38, 1.0f }, { 0x42, 2.5f },
        { 0x78, 256.0f }, { 0x7e, 448.0f }, { 0xfe, -448.0f }, { 0xc2, -2.5f },
    };
    for (size_t i = 0; i < sizeof decode_cases / sizeof decode_cases[0]; ++i) {
        CHECK(dlss5_nr::e4m3_from_bits(decode_cases[i].bits) == decode_cases[i].value);
        CHECK(oracle_e4m3_decode(decode_cases[i].bits) == decode_cases[i].value);
    }
    CHECK(dlss5_nr::e4m3_from_bits(0x80) == 0.0f);
    CHECK(signbit(dlss5_nr::e4m3_from_bits(0x80)));
    CHECK(isnan(dlss5_nr::e4m3_from_bits(0x7f)));
    CHECK(isnan(dlss5_nr::e4m3_from_bits(0xff)));
}

static void check_e4m3_sweep(void) {
    // Every code decodes identically in production and oracle, and every non-NaN code
    // round-trips through production encode.
    for (uint32_t h = 0; h < 256u; ++h) {
        uint8_t raw = (uint8_t)h;
        float production = dlss5_nr::e4m3_from_bits(raw);
        float oracle = oracle_e4m3_decode(raw);
        uint32_t production_bits = 0;
        uint32_t oracle_bits = 0;
        memcpy(&production_bits, &production, sizeof production_bits);
        memcpy(&oracle_bits, &oracle, sizeof oracle_bits);
        if (production_bits != oracle_bits) {
            CHECK(false);
        }
        bool nan_code = ((h >> 3) & 0xfu) == 0xfu && (h & 7u) == 7u;
        if (nan_code) {
            continue;
        }
        if (dlss5_nr::e4m3_bits(production) != raw) {
            CHECK(false);
        }
    }
    ++g_checks;
    // Structured probes near every rounding boundary and pseudorandom floats against
    // the brute-force oracle.
    static const float probes[] = {
        0.0f, -0.0f, 0x1p-12f, 0x1p-10f, 0x1p-9f, 0x1p-8f, 0x1p-7f, 0x1p-6f,
        0x1p-5f, 0x1p-4f, 0x1p-3f, 0x1p-2f, 0x1p-1f, 1.0f, 2.0f, 100.0f,
        255.0f, 256.0f, 300.0f, 400.0f, 432.0f, 447.9f, 448.0f, 500.0f, 65504.0f,
        1e30f, -432.0f, -1.0f,
    };
    for (size_t i = 0; i < sizeof probes / sizeof probes[0]; ++i) {
        CHECK(dlss5_nr::e4m3_bits(probes[i]) == oracle_e4m3_bits(probes[i]));
    }
    for (int i = 0; i < 200000; ++i) {
        float value = random_float();
        if (dlss5_nr::e4m3_bits(value) != oracle_e4m3_bits(value)) {
            CHECK(false);
        }
    }
    ++g_checks;
}

static void check_pool_known_answers(void) {
    using dlss5_nr::pool2x2;
    using dlss5_nr::PlaneView;
    float out[4];
    PlaneView src = {};

    // Case A: exact arithmetic block -> 2.5.
    float a[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    src.data = a;
    src.rows = 2;
    src.cols = 2;
    src.row_stride = 2;
    CHECK(pool2x2(src, out, 1) == dlss5_nr::Status::kOk);
    CHECK(out[0] == 2.5f);

    // Case B: grouping discriminator. Row-first (the documented order) gives
    // f16(4.25 + 2^-9) = 4.25 (tie to even), then e4m3(1.0625) = 1.0 (tie to even).
    // A column-first grouping would give 1.125 instead.
    float b[4] = { 4.25f, 0x1p-9f, 0.0f, 0x1p-9f };
    src.data = b;
    CHECK(pool2x2(src, out, 1) == dlss5_nr::Status::kOk);
    CHECK(out[0] == 1.0f);
    CHECK(out[0] != 1.125f);

    // Case C: f16 tie in the column-pair sum (2 + 2^-10 -> 2.0, ties to even).
    float c[4] = { 2.0f, 0x1p-10f, 0.0f, 0.0f };
    src.data = c;
    CHECK(pool2x2(src, out, 1) == dlss5_nr::Status::kOk);
    CHECK(out[0] == 0.5f);

    // Case D: negatives.
    float d[4] = { -1.0f, -2.0f, -3.0f, -4.0f };
    src.data = d;
    CHECK(pool2x2(src, out, 1) == dlss5_nr::Status::kOk);
    CHECK(out[0] == -2.5f);

    // Case E: the f16 sums overflow to infinity; the quarter of infinity saturates.
    float e[4] = { 1e6f, 1e6f, 1e6f, 1e6f };
    src.data = e;
    CHECK(pool2x2(src, out, 1) == dlss5_nr::Status::kOk);
    CHECK(out[0] == 448.0f);
    float e_neg[4] = { -1e6f, -1e6f, -1e6f, -1e6f };
    src.data = e_neg;
    CHECK(pool2x2(src, out, 1) == dlss5_nr::Status::kOk);
    CHECK(out[0] == -448.0f);

    // Case F: NaN propagates to the canonical NaN.
    float f[4] = { (float)NAN, 0.0f, 0.0f, 0.0f };
    src.data = f;
    CHECK(pool2x2(src, out, 1) == dlss5_nr::Status::kOk);
    CHECK(isnan(out[0]));

    // Case G: 2x4 plane -> two outputs, hand-computed (3.5 and 5.5).
    float g[8] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
    src.data = g;
    src.cols = 4;
    src.row_stride = 4;
    float out2[2];
    CHECK(pool2x2(src, out2, 2) == dlss5_nr::Status::kOk);
    CHECK(out2[0] == 3.5f);
    CHECK(out2[1] == 5.5f);
}

static void check_pool_oracle_and_layout(void) {
    using dlss5_nr::pool2x2;
    using dlss5_nr::PlaneView;
    // 6x6 plane with a padded 8-float stride; full-buffer oracle comparison,
    // determinism and stride independence.
    const int rows = 6;
    const int cols = 6;
    const int stride = 8;
    float padded[6 * 8];
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            padded[y * stride + x] = random_float() * 0.001f;
        }
        for (int x = cols; x < stride; ++x) {
            padded[y * stride + x] = kSentinel;
        }
    }
    PlaneView src = {};
    src.data = padded;
    src.rows = rows;
    src.cols = cols;
    src.row_stride = stride;
    float out[3 * 3];
    float reference[3 * 3];
    CHECK(pool2x2(src, out, 9) == dlss5_nr::Status::kOk);
    for (int oy = 0; oy < rows / 2; ++oy) {
        for (int ox = 0; ox < cols / 2; ++ox) {
            const float* top = padded + (2 * oy) * stride + 2 * ox;
            const float* bottom = padded + (2 * oy + 1) * stride + 2 * ox;
            reference[oy * (cols / 2) + ox]
                = oracle_pool_block(top[0], top[1], bottom[0], bottom[1]);
        }
    }
    CHECK(memcmp(out, reference, sizeof out) == 0);

    // Determinism: a second run is bit-identical.
    float again[3 * 3];
    CHECK(pool2x2(src, again, 9) == dlss5_nr::Status::kOk);
    CHECK(memcmp(out, again, sizeof out) == 0);

    // Row-stride independence: the same content in a tight 6-float stride.
    float tight[6 * 6];
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            tight[y * cols + x] = padded[y * stride + x];
        }
    }
    PlaneView tight_src = src;
    tight_src.data = tight;
    tight_src.row_stride = cols;
    float tight_out[3 * 3];
    CHECK(pool2x2(tight_src, tight_out, 9) == dlss5_nr::Status::kOk);
    CHECK(memcmp(out, tight_out, sizeof out) == 0);
}

static void check_pool_negative(void) {
    using dlss5_nr::pool2x2;
    using dlss5_nr::PlaneView;
    float pixels[2 * 4] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
    float out[2];
    PlaneView src = {};
    src.data = pixels;
    src.rows = 2;
    src.cols = 4;
    src.row_stride = 4;

    auto rejected = [&](const PlaneView& s, float* o, size_t oc, dlss5_nr::Status expected) {
        if (o != nullptr) {
            fill_sentinel(o, 2);
        }
        CHECK(pool2x2(s, o, oc) == expected);
        if (o != nullptr) {
            CHECK(o[0] == kSentinel);  // validation precedes any output write
            CHECK(o[1] == kSentinel);
        }
    };

    PlaneView bad = src;
    bad.data = nullptr;
    rejected(bad, out, 2, dlss5_nr::Status::kNullPointer);
    bad = src;
    bad.rows = 0;
    rejected(bad, out, 2, dlss5_nr::Status::kInvalidDimensions);
    bad = src;
    bad.cols = -1;
    rejected(bad, out, 2, dlss5_nr::Status::kInvalidDimensions);
    bad = src;
    bad.rows = 3;  // odd extent
    rejected(bad, out, 2, dlss5_nr::Status::kInvalidDimensions);
    bad = src;
    bad.cols = 3;  // odd extent
    rejected(bad, out, 2, dlss5_nr::Status::kInvalidDimensions);
    bad = src;
    bad.rows = dlss5_nr::kMaxRows + 2;
    bad.cols = 2;
    bad.row_stride = 2;
    rejected(bad, out, 1, dlss5_nr::Status::kInvalidDimensions);
    bad = src;
    bad.cols = dlss5_nr::kMaxCols + 2;
    bad.row_stride = dlss5_nr::kMaxCols + 2;
    rejected(bad, out, 1, dlss5_nr::Status::kInvalidDimensions);
    bad = src;
    bad.rows = 8192;
    bad.cols = 8192;
    bad.row_stride = 8192;
    rejected(bad, out, 1, dlss5_nr::Status::kInvalidDimensions);  // > kMaxPixels
    bad = src;
    bad.row_stride = 3;  // below cols
    rejected(bad, out, 2, dlss5_nr::Status::kInvalidStride);
    bad = src;
    bad.row_stride = dlss5_nr::kMaxCols + 2;
    rejected(bad, out, 2, dlss5_nr::Status::kInvalidStride);
    rejected(src, nullptr, 2, dlss5_nr::Status::kNullPointer);
    rejected(src, out, 1, dlss5_nr::Status::kInvalidBuffer);
    // A valid call still succeeds after the rejections.
    fill_sentinel(out, 2);
    CHECK(pool2x2(src, out, 2) == dlss5_nr::Status::kOk);
    CHECK(out[0] == 3.5f);
    CHECK(out[1] == 5.5f);
}

static void check_allocation_discipline(void) {
    using dlss5_nr::pool2x2;
    using dlss5_nr::PlaneView;
    float pixels[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float out[1];
    PlaneView src = {};
    src.data = pixels;
    src.rows = 2;
    src.cols = 2;
    src.row_stride = 2;
    unsigned long long before = g_allocations.load();
    CHECK(pool2x2(src, out, 1) == dlss5_nr::Status::kOk);
    CHECK(dlss5_nr::e4m3(1.3f) == 1.25f);
    CHECK(dlss5_nr::is_e4m3(2.5f));
    unsigned long long after = g_allocations.load();
    CHECK(after == before);  // the whole stage is caller-allocated
}

int main(void) {
    check_e4m3_literals();
    check_e4m3_sweep();
    check_pool_known_answers();
    check_pool_oracle_and_layout();
    check_pool_negative();
    check_allocation_discipline();
    printf("native-pre-post-test: %llu checks passed\n", g_checks);
    return 0;
}
