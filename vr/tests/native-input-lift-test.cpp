// Independent regression for the typed CPU input-lift slice (vr/native-input-lift.hpp).
//
// Checks are anchored on values derived OUTSIDE the production code:
//  - binary16 rounding literals derived from the IEEE 754 definition (Fraction-based
//    Python) and cross-checked with numpy float16 and the compiler's _Float16,
//  - noise hash/uniform literals derived with Python integers from the specification,
//  - gaussian noise-feature literals derived with a separate scalar C program (glibc
//    libm; compared with a small tolerance because transcendentals are libm-dependent),
//  - hand-computed end-to-end cases whose values are exact in binary16,
//  - an in-file scalar oracle written from the specification (uint64 multiply variant).
//
// Synthetic frames and weights are original. Passing establishes synthetic arithmetic
// correctness of this slice only; it says nothing about vendor model correspondence.
#include "../native-input-lift.hpp"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>
#include <new>

// ---- allocation accounting: the lift path must not allocate ---------------------------------
static std::atomic<unsigned long long> g_allocations{0};

static void* counted_alloc(size_t size) {
    g_allocations.fetch_add(1);
    void* p = malloc(size);
    if (p == NULL) {
        throw std::bad_alloc();
    }
    return p;
}
static void* counted_aligned_alloc(size_t size, size_t alignment) {
    g_allocations.fetch_add(1);
    size_t rounded = ((size + alignment - 1) / alignment) * alignment;
    void* p = aligned_alloc(alignment, rounded);
    if (p == NULL) {
        throw std::bad_alloc();
    }
    return p;
}

void* operator new(size_t size) { return counted_alloc(size); }
void* operator new[](size_t size) { return counted_alloc(size); }
void* operator new(size_t size, std::align_val_t alignment) {
    return counted_aligned_alloc(size, static_cast<size_t>(alignment));
}
void* operator new[](size_t size, std::align_val_t alignment) {
    return counted_aligned_alloc(size, static_cast<size_t>(alignment));
}
void operator delete(void* p) noexcept { free(p); }
void operator delete[](void* p) noexcept { free(p); }
void operator delete(void* p, size_t) noexcept { free(p); }
void operator delete[](void* p, size_t) noexcept { free(p); }
void operator delete(void* p, std::align_val_t) noexcept { free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { free(p); }
void operator delete(void* p, size_t, std::align_val_t) noexcept { free(p); }
void operator delete[](void* p, size_t, std::align_val_t) noexcept { free(p); }

// ---- test harness ---------------------------------------------------------------------------
static unsigned long long g_checks = 0;
#define CHECK(condition) do { assert(condition); ++g_checks; } while (0)

// Independent rounding oracle: the compiler's _Float16 conversion path.
static uint16_t oracle_f16_bits(float value) {
    _Float16 half = (_Float16)value;
    uint16_t bits = 0;
    memcpy(&bits, &half, sizeof bits);
    return bits;
}

static float half_value(uint16_t bits) { return dlss5_nr::f16_from_bits(bits); }

// Independent statement of the stored lift permutation. This restates the documented
// formula verbatim; genuine independence rests on the hand-derived anchor table in
// check_lift_permutation and on the pinned upstream source, not on this function.
static size_t oracle_offset(int channel, int k) {
    return (size_t)((channel / 16) * 256 + 32 * (channel % 8) + 8 * ((k % 8) / 2)
        + 4 * ((channel / 8) % 2) + (k % 2) + 2 * (k / 8));
}

// Independent binary16 decode built straight from the IEEE 754 definition with ldexpf,
// sharing no code with the production decoder. NaN patterns map to a canonical NaN.
static float oracle_half(uint16_t bits) {
    uint16_t e = (uint16_t)((bits >> 10) & 0x1fu);
    uint16_t m = (uint16_t)(bits & 0x3ffu);
    float sign = (bits & 0x8000u) ? -1.0f : 1.0f;
    if (e == 0x1fu) {
        return m == 0 ? sign * (float)INFINITY : (float)NAN;
    }
    if (e == 0) {
        return sign * ldexpf((float)m, -24);                   // subnormal: m * 2^-24
    }
    return sign * ldexpf((float)(1024u + m), (int)e - 25);     // (1024 + m) * 2^(e-25)
}

// Independent scalar noise oracle: 64-bit multiply variant of the uint32 mixer.
static void oracle_uniforms(uint32_t x, uint32_t y, uint32_t seed, float out[4]) {
    auto pcg = [](uint32_t v) -> uint32_t {
        uint32_t shifted = v >> (((v >> 28) + 4u) & 31u);
        uint32_t folded = shifted ^ v;
        return (uint32_t)((uint64_t)folded * 0x108ef2d9ull);
    };
    auto unit = [&pcg](uint32_t stream) -> float {
        uint32_t t = pcg(stream);
        uint32_t v = ((t >> 30) ^ (t >> 8)) + 1u;
        return (float)v * (1.0f / 16777216.0f);
    };
    uint32_t base = x * 0x8da6b343u;
    base ^= seed * 0x9e3779b9u;
    base ^= y * 0xd8163841u;
    base ^= 0x243f6a88u;
    uint32_t t = pcg(base);
    uint32_t h = (t >> 22) ^ t;
    out[0] = unit(h * 0x2c9277b5u + 0xac564b05u);
    out[1] = unit(h * 0xfa6dc5f9u + 0x4712a88eu);
    out[2] = unit(h * 0xcaa5b80du + 0x21dd796bu);
    out[3] = unit(h * 0x83232c31u + 0x3463e0acu);
}

static void oracle_noise(uint32_t x, uint32_t y, uint32_t seed, float out[3]) {
    float u[4] = {};
    oracle_uniforms(x, y, seed, u);
    const float two_pi = 6.28318530718f;
    float radius_a = sqrtf(-2.0f * logf(u[0]));
    float radius_c = sqrtf(-2.0f * logf(u[2]));
    float angle_a = u[1] * two_pi;
    float angle_c = u[3] * two_pi;
    out[0] = oracle_half(oracle_f16_bits(1.0f * (radius_a * cosf(angle_a))));
    out[1] = oracle_half(oracle_f16_bits(1.0f * (radius_a * sinf(angle_a))));
    out[2] = oracle_half(oracle_f16_bits(1.0f * (radius_c * cosf(angle_c))));
}

static void oracle_centred(const float rgb[3], float out[3]) {
    for (int c = 0; c < 3; ++c) {
        float a = oracle_half(oracle_f16_bits(rgb[c]));
        float b = oracle_half(oracle_f16_bits(a - 0.5f));
        out[c] = oracle_half(oracle_f16_bits(b * 0.125f));
    }
}

// Independent end-to-end oracle for the full slice on a validated frame.
static void oracle_input_lift(const dlss5_nr::FrameView& frame, const dlss5_nr::Controls& controls,
                              const float* lift, float* lifted) {
    // Conditioning written from the specification: the no-mask path emits
    // (style, tone, structure, -1, -1); the auto-mask path emits
    // (style, tone, 1, skin-with-fallback, other = local_structure).
    float condition[5];
    if (!controls.auto_mask) {
        condition[0] = oracle_half(oracle_f16_bits((float)controls.style * (1.0f / 128.0f)));
        condition[1] = oracle_half(oracle_f16_bits(controls.local_tone));
        condition[2] = oracle_half(oracle_f16_bits(controls.local_structure));
        condition[3] = -1.0f;
        condition[4] = -1.0f;
    } else {
        condition[0] = oracle_half(oracle_f16_bits((float)controls.style * (1.0f / 128.0f)));
        condition[1] = oracle_half(oracle_f16_bits(controls.local_tone));
        condition[2] = 1.0f;
        condition[3] = oracle_half(oracle_f16_bits(controls.skin_structure < 0.0f
                                                       ? controls.local_structure
                                                       : controls.skin_structure));
        condition[4] = oracle_half(oracle_f16_bits(controls.local_structure));
    }
    size_t pixels = (size_t)frame.rows * (size_t)frame.cols;
    for (int y = 0; y < frame.rows; ++y) {
        const float* row = frame.rgb + (ptrdiff_t)y * frame.row_stride;
        for (int x = 0; x < frame.cols; ++x) {
            float features[16] = {};
            oracle_noise((uint32_t)x, (uint32_t)y, controls.seed, &features[0]);
            features[3] = 1.0f;
            const float* rgb = row + 3 * (ptrdiff_t)x;
            float colour[3] = {};
            oracle_centred(rgb, colour);
            for (int c = 0; c < 3; ++c) {
                features[4 + c] = colour[c];
                features[7 + c] = colour[c];
            }
            for (int i = 0; i < 5; ++i) {
                features[10 + i] = condition[i];
            }
            features[15] = 0.0f;
            size_t pixel = (size_t)y * (size_t)frame.cols + (size_t)x;
            for (int c = 0; c < 32; ++c) {
                float sum = 0.0f;
                for (int k = 0; k < 16; ++k) {
                    sum += lift[c * 16 + k] * features[k];
                }
                lifted[c * pixels + pixel] = oracle_half(oracle_f16_bits(sum));
            }
        }
    }
}

// ---- synthetic data (original) -----------------------------------------------------------------
// Deterministic synthetic frame values on the 8-bit grid: ((x*7 + y*13 + channel*29) % 256)/255.
static void synthetic_frame(float* rgb, int rows, int cols) {
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            for (int c = 0; c < 3; ++c) {
                int code = (x * 7 + y * 13 + c * 29) % 256;
                rgb[((size_t)y * (size_t)cols + (size_t)x) * 3 + (size_t)c] = (float)code / 255.0f;
            }
        }
    }
}

// Synthetic stored lift record: every entry f16-exact by construction (integer / 128).
static void synthetic_stored(float* stored) {
    for (int o = 0; o < 512; ++o) {
        stored[o] = (float)((o * 37) % 251 - 125) / 128.0f;
    }
}

// Lift with channel c reading feature c for c < 16 (identity select), zero rows above.
static void select_lift(float* lift) {
    for (int c = 0; c < 32; ++c) {
        for (int k = 0; k < 16; ++k) {
            lift[c * 16 + k] = (c == k) ? 1.0f : 0.0f;
        }
    }
}

static const float kSentinel = -1234.5f;
static void fill_sentinel(float* values, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        values[i] = kSentinel;
    }
}

// ---- check sections -----------------------------------------------------------------------------
static void check_f16_literals(void) {
    static const struct { float value; uint16_t bits; } cases[] = {
        { 0.0f, 0x0000 }, { -0.0f, 0x8000 }, { 1.0f, 0x3c00 }, { 0.5f, 0x3800 },
        { -0.5f, 0xb800 }, { 0.6f, 0x38cd }, { 0.625f, 0x3900 }, { 0.25f, 0x3400 },
        { 0.75f, 0x3a00 }, { 0.125f, 0x3000 }, { 0.015625f, 0x2400 }, { -0.03125f, 0xa800 },
        { 0.03125f, 0x2800 }, { 1.00048828125f, 0x3c00 }, { 1.00146484375f, 0x3c02 },
        { 1.0009765625f, 0x3c01 }, { 65504.0f, 0x7bff }, { 65519.0f, 0x7bff },
        { 65520.0f, 0x7c00 }, { 65536.0f, 0x7c00 }, { 6.103515625e-05f, 0x0400 },
        { 3.0517578125e-05f, 0x0200 }, { 5.960464477539063e-08f, 0x0001 },
        { 2.9802322387695312e-08f, 0x0000 }, { 4.470348358154297e-08f, 0x0001 },
        { 1.4901161193847656e-08f, 0x0000 }, { 0.1f, 0x2e66 }, { 0.3333333333333333f, 0x3555 },
        { -1.00048828125f, 0xbc00 }, { 1.9990234375f, 0x3fff },
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; ++i) {
        CHECK(dlss5_nr::f16_bits(cases[i].value) == cases[i].bits);
    }
    // Canonical NaN handling and f16-representability validation.
    float nan_value = (float)NAN;
    CHECK(dlss5_nr::f16_bits(nan_value) == 0x7e00);
    CHECK(dlss5_nr::f16_bits(-nan_value) == 0xfe00);
    CHECK(dlss5_nr::is_f16(0.625f));
    CHECK(dlss5_nr::is_f16(-0.0f));
    CHECK(dlss5_nr::is_f16(half_value(0x0001)));   // 2^-24 subnormal
    CHECK(!dlss5_nr::is_f16(0.1f));                // not on the f16 grid
    CHECK(!dlss5_nr::is_f16(nan_value));
}

static void check_f16_sweep(void) {
    // Every binary16 value round-trips (NaN payloads are canonicalised by design), and
    // the production decode is bit-identical to the independent ldexpf decode oracle.
    for (uint32_t h = 0; h < 0x10000u; ++h) {
        bool nan_bits = ((h >> 10) & 0x1f) == 0x1f && (h & 0x3ff) != 0;
        float value = half_value((uint16_t)h);
        if (nan_bits) {
            continue;
        }
        if (dlss5_nr::f16_bits(value) != (uint16_t)h) {
            CHECK(false);
        }
        uint32_t production_bits = 0;
        uint32_t oracle_bits = 0;
        float oracle_value = oracle_half((uint16_t)h);
        memcpy(&production_bits, &value, sizeof production_bits);
        memcpy(&oracle_bits, &oracle_value, sizeof oracle_bits);
        if (production_bits != oracle_bits) {
            CHECK(false);
        }
    }
    ++g_checks;
    // Structured and pseudorandom float32 values against the _Float16 oracle.
    uint32_t state = 0x9e3779b9u;
    for (int i = 0; i < 200000; ++i) {
        state = state * 1664525u + 1013904223u;
        uint32_t bits = state ^ (uint32_t)(i << 12);
        float value = 0.0f;
        memcpy(&value, &bits, sizeof value);
        if ((bits & 0x7fffffffu) > 0x7f800000u) {
            continue;  // NaN: canonicalisation policy, not IEEE payload preservation
        }
        if (dlss5_nr::f16_bits(value) != oracle_f16_bits(value)) {
            CHECK(false);
        }
    }
    for (int exponent = -24; exponent <= 16; ++exponent) {
        float base = ldexpf(1.0f, exponent);
        const float probes[6] = {
            base, 1.5f * base, base * (1.0f + 0x1p-11f), base * (1.0f + 0x3p-11f),
            base * (1.0f - 0x1p-11f), -1.5f * base,
        };
        for (int i = 0; i < 6; ++i) {
            CHECK(dlss5_nr::f16_bits(probes[i]) == oracle_f16_bits(probes[i]));
        }
    }
}

static void check_lift_permutation(void) {
    // Hand-checked spot values of the stored permutation, derived by hand from the
    // documented formula (not via oracle_offset): (c,k) -> (c/16)*256 + 32*(c%8)
    // + 8*((k%8)/2) + 4*((c/8)%2) + (k%2) + 2*(k/8).
    static const struct { int channel; int k; size_t offset; } spots[] = {
        { 0, 0, 0 }, { 1, 0, 32 }, { 0, 1, 1 }, { 0, 2, 8 }, { 0, 8, 2 },
        { 15, 15, 255 }, { 16, 0, 256 }, { 31, 15, 511 }, { 8, 4, 20 }, { 2, 7, 89 },
        { 3, 5, 113 }, { 20, 10, 394 }, { 9, 3, 45 }, { 31, 0, 484 },
        { 16, 15, 283 }, { 7, 9, 227 },
    };
    for (size_t i = 0; i < sizeof spots / sizeof spots[0]; ++i) {
        CHECK(dlss5_nr::lift_offset(spots[i].channel, spots[i].k) == spots[i].offset);
        CHECK(oracle_offset(spots[i].channel, spots[i].k) == spots[i].offset);
    }
    // Complete permutation coverage: every stored slot is claimed exactly once.
    int visits[512] = {0};
    for (int c = 0; c < 32; ++c) {
        for (int k = 0; k < 16; ++k) {
            ++visits[dlss5_nr::lift_offset(c, k)];
        }
    }
    for (int o = 0; o < 512; ++o) {
        CHECK(visits[o] == 1);
    }
    // Decode maps stored[offset(c,k)] to lift[c][k]; stored[o] = o/8 - 4 (f16-exact).
    float stored[512] = {};
    for (int o = 0; o < 512; ++o) {
        stored[o] = (float)o / 8.0f - 4.0f;
    }
    float lift[512] = {};
    CHECK(dlss5_nr::decode_lift(stored, 512, lift) == dlss5_nr::Status::kOk);
    for (int c = 0; c < 32; ++c) {
        for (int k = 0; k < 16; ++k) {
            CHECK(lift[c * 16 + k] == (float)oracle_offset(c, k) / 8.0f - 4.0f);
        }
    }
}

static void check_noise_literals(void) {
    // Hash/uniform integers derived independently with Python; u = v * 2^-24 is exact.
    static const struct { uint32_t x, y, seed, units[4]; } cases[] = {
        { 0, 0, 0, { 9653346, 4756067, 11369803, 12969588 } },
        { 1, 0, 0, { 2530865, 3960466, 14726869, 14427493 } },
        { 0, 1, 0, { 1254652, 4635569, 10440990, 12905293 } },
        { 5, 3, 7, { 14512208, 13074347, 15988184, 6175718 } },
        { 3839, 2159, 0xffffffffu, { 16177276, 13525629, 12746019, 14875355 } },
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; ++i) {
        float uniforms[4] = {};
        CHECK(dlss5_nr::noise_uniforms(cases[i].x, cases[i].y, cases[i].seed, uniforms)
              == dlss5_nr::Status::kOk);
        for (int j = 0; j < 4; ++j) {
            CHECK(uniforms[j] == (float)cases[i].units[j] * 0x1p-24f);
        }
    }
    // Gaussian feature literals (f16 values derived from the scalar C derivation). The
    // tolerance covers libm last-bit variation; the exact cross-check is the oracle below.
    static const struct { uint32_t x, y, seed; float expected[3]; } noise_cases[] = {
        { 0, 0, 0, { -0.2196044921875f, 1.0283203125f, 0.1273193359375f } },
        { 1, 0, 0, { 0.170166015625f, 1.9375f, 0.325439453125f } },
        { 0, 1, 0, { -0.374755859375f, 2.24609375f, 0.1173095703125f } },
        { 5, 3, 7, { 0.09857177734375f, -0.529296875f, -0.209716796875f } },
        { 3839, 2159, 0xffffffffu, { 0.09332275390625f, -0.253173828125f, 0.56103515625f } },
    };
    const float tolerance = 0x1p-9f;
    for (size_t i = 0; i < sizeof noise_cases / sizeof noise_cases[0]; ++i) {
        float features[3] = {};
        CHECK(dlss5_nr::noise_features(noise_cases[i].x, noise_cases[i].y, noise_cases[i].seed,
                                       features) == dlss5_nr::Status::kOk);
        for (int j = 0; j < 3; ++j) {
            CHECK(fabsf(features[j] - noise_cases[i].expected[j]) <= tolerance);
        }
    }
    // Exact oracle comparison at the same points.
    for (size_t i = 0; i < sizeof noise_cases / sizeof noise_cases[0]; ++i) {
        float expected[3] = {};
        oracle_noise(noise_cases[i].x, noise_cases[i].y, noise_cases[i].seed, expected);
        float features[3] = {};
        CHECK(dlss5_nr::noise_features(noise_cases[i].x, noise_cases[i].y, noise_cases[i].seed,
                                       features) == dlss5_nr::Status::kOk);
        for (int j = 0; j < 3; ++j) {
            CHECK(features[j] == expected[j]);
        }
    }
    // Seed and position sensitivity of the deterministic generator.
    float a[3] = {}, b[3] = {};
    CHECK(dlss5_nr::noise_features(4, 2, 0, a) == dlss5_nr::Status::kOk);
    CHECK(dlss5_nr::noise_features(4, 2, 1, b) == dlss5_nr::Status::kOk);
    CHECK(memcmp(a, b, sizeof a) != 0);
    CHECK(dlss5_nr::noise_features(5, 2, 0, b) == dlss5_nr::Status::kOk);
    CHECK(memcmp(a, b, sizeof a) != 0);
    CHECK(dlss5_nr::noise_features(4, 3, 0, b) == dlss5_nr::Status::kOk);
    CHECK(memcmp(a, b, sizeof a) != 0);
}

static void check_centred_colour(void) {
    // Chain literals (a = f16(rgb); b = f16(a - 0.5); c = f16(b * 0.125)) derived with
    // numpy float16 from the definition; exact f16 values.
    static const struct { float rgb[3]; float expected[3]; } cases[] = {
        { { 0.5f, 0.5f, 0.5f }, { 0.0f, 0.0f, 0.0f } },
        { { 0.625f, 0.25f, 0.75f }, { 0.015625f, -0.03125f, 0.03125f } },
        { { 0.6f, 1.0f / 3.0f, 0.1f },
          { 0.01251220703125f, -0.020843505859375f, -0.04998779296875f } },
        { { 1.0f, 0.0f, 254.0f / 255.0f },
          { 0.0625f, -0.0625f, 0.06201171875f } },
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; ++i) {
        float out[3] = { kSentinel, kSentinel, kSentinel };
        CHECK(dlss5_nr::centred_colour(cases[i].rgb, out) == dlss5_nr::Status::kOk);
        for (int c = 0; c < 3; ++c) {
            CHECK(out[c] == cases[i].expected[c]);
        }
        float reference[3] = {};
        oracle_centred(cases[i].rgb, reference);
        for (int c = 0; c < 3; ++c) {
            CHECK(out[c] == reference[c]);
        }
    }
}

static void check_end_to_end_known_answers(void) {
    // Case A: 1x1 grey frame, default controls, identity-select lift. The constant and
    // conditioning channels are hand-computed exact values; the noise channels anchor to
    // the (0, 0, 0) literals above through the select rows.
    float stored[512] = {};
    float lift[512] = {};
    select_lift(lift);
    float out[32] = {};
    dlss5_nr::FrameView frame = {};
    static const float grey_pixel[3] = { 0.5f, 0.5f, 0.5f };
    frame.rgb = grey_pixel;
    frame.rows = 1;
    frame.cols = 1;
    frame.row_stride = 3;
    dlss5_nr::Controls controls = {};
    CHECK(dlss5_nr::input_lift(frame, controls, lift, 512, out, 32) == dlss5_nr::Status::kOk);
    float noise_expected[3] = {};
    oracle_noise(0, 0, 0, noise_expected);
    for (int c = 0; c < 3; ++c) {
        CHECK(out[c] == noise_expected[c]);
    }
    CHECK(out[3] == 1.0f);
    for (int c = 4; c <= 10; ++c) {
        CHECK(out[c] == 0.0f);
    }
    for (int c = 11; c <= 14; ++c) {
        CHECK(out[c] == 1.0f);
    }
    CHECK(out[15] == 0.0f);
    for (int c = 16; c < 32; ++c) {
        CHECK(out[c] == 0.0f);
    }

    // Case B: exact arithmetic lift on a 2x2 frame of (0.625, 0.25, 0.75). Centred colour
    // is (0.015625, -0.03125, 0.03125); channel 0 = 0.5*const + 2*R - 4*G + 0.5*B
    // = 0.5 + 0.03125 + 0.125 + 0.015625 = 0.671875 at every pixel.
    for (int i = 0; i < 512; ++i) {
        lift[i] = 0.0f;
    }
    lift[0 * 16 + 3] = 0.5f;
    lift[0 * 16 + 4] = 2.0f;
    lift[0 * 16 + 5] = -4.0f;
    lift[0 * 16 + 6] = 0.5f;
    // Channel 1 cancels the duplicated current-colour copy against the first copy.
    lift[1 * 16 + 7] = 1.0f;
    lift[1 * 16 + 4] = -1.0f;
    // Channel 2: 1.0*const + 2^-11*tone = 1 + 2^-11, a tie that rounds to 1.0.
    lift[2 * 16 + 3] = 1.0f;
    lift[2 * 16 + 11] = 0x1p-11f;
    // Channel 3: 1.0*const + 3*2^-11*tone = 1 + 3*2^-11, a tie that rounds to 1.001953125.
    lift[3 * 16 + 3] = 1.0f;
    lift[3 * 16 + 11] = 0x3p-11f;
    float pixels[2 * 2 * 3];
    for (int i = 0; i < 4; ++i) {
        pixels[i * 3 + 0] = 0.625f;
        pixels[i * 3 + 1] = 0.25f;
        pixels[i * 3 + 2] = 0.75f;
    }
    frame.rgb = pixels;
    frame.rows = 2;
    frame.cols = 2;
    frame.row_stride = 6;
    float out2[32 * 4] = {};
    CHECK(dlss5_nr::input_lift(frame, controls, lift, 512, out2, 32 * 4) == dlss5_nr::Status::kOk);
    for (int pixel = 0; pixel < 4; ++pixel) {
        CHECK(out2[pixel] == 0.671875f);
        CHECK(out2[4 + pixel] == 0.0f);
        CHECK(out2[8 + pixel] == 1.0f);
        CHECK(out2[12 + pixel] == 1.001953125f);
    }

    // Case C: conditioning features with the 1x1 grey frame and select lift.
    select_lift(lift);
    frame.rgb = grey_pixel;
    frame.rows = 1;
    frame.cols = 1;
    frame.row_stride = 3;
    controls.style = 2;                    // feature 10 = 2/128 = 0.015625
    controls.local_tone = 1.5f;            // feature 11 = 1.5
    controls.skin_structure = 0.5f;        // feature 13 = 0.5
    float out3[32] = {};
    CHECK(dlss5_nr::input_lift(frame, controls, lift, 512, out3, 32) == dlss5_nr::Status::kOk);
    CHECK(out3[10] == 0.015625f);
    CHECK(out3[11] == 1.5f);
    CHECK(out3[12] == 1.0f);
    CHECK(out3[13] == 0.5f);
    CHECK(out3[14] == 1.0f);   // "other" carries local_structure (default 1.0)
    controls.auto_mask = false;            // features 13/14 = -1, feature 12 = local_structure
    controls.local_structure = 0.25f;
    CHECK(dlss5_nr::input_lift(frame, controls, lift, 512, out3, 32) == dlss5_nr::Status::kOk);
    CHECK(out3[12] == 0.25f);
    CHECK(out3[13] == -1.0f);
    CHECK(out3[14] == -1.0f);

    // Stored-weights path: decode the synthetic stored record and use it directly.
    synthetic_stored(stored);
    float decoded[512] = {};
    CHECK(dlss5_nr::decode_lift(stored, 512, decoded) == dlss5_nr::Status::kOk);
    CHECK(dlss5_nr::validate_weights(stored, 512) == dlss5_nr::Status::kOk);
    controls = dlss5_nr::Controls{};
    float out4[32] = {};
    CHECK(dlss5_nr::input_lift(frame, controls, decoded, 512, out4, 32) == dlss5_nr::Status::kOk);
    float reference[32] = {};
    oracle_input_lift(frame, controls, decoded, reference);
    for (int c = 0; c < 32; ++c) {
        CHECK(out4[c] == reference[c]);
    }
}

static void check_sensitivity_and_locality(void) {
    // 3x4 synthetic frame with the generic synthetic stored lift; full-buffer oracle match.
    float pixels[3 * 4 * 3];
    synthetic_frame(pixels, 3, 4);
    float stored[512] = {};
    synthetic_stored(stored);
    float lift[512] = {};
    CHECK(dlss5_nr::decode_lift(stored, 512, lift) == dlss5_nr::Status::kOk);
    dlss5_nr::FrameView frame = {};
    frame.rgb = pixels;
    frame.rows = 3;
    frame.cols = 4;
    frame.row_stride = 12;
    dlss5_nr::Controls controls = {};
    controls.style = 1;
    controls.local_tone = 1.25f;
    controls.local_structure = 0.75f;
    controls.skin_structure = 0.5f;
    float out[32 * 12] = {};
    CHECK(dlss5_nr::input_lift(frame, controls, lift, 512, out, 32 * 12) == dlss5_nr::Status::kOk);
    float reference[32 * 12] = {};
    oracle_input_lift(frame, controls, lift, reference);
    CHECK(memcmp(out, reference, sizeof out) == 0);

    // No-mask conditioning branch: end-to-end oracle match with (style, tone, structure,
    // -1, -1) features and the skin/other overrides.
    controls.auto_mask = false;
    float out_nomask[32 * 12] = {};
    CHECK(dlss5_nr::input_lift(frame, controls, lift, 512, out_nomask, 32 * 12)
          == dlss5_nr::Status::kOk);
    float reference_nomask[32 * 12] = {};
    oracle_input_lift(frame, controls, lift, reference_nomask);
    CHECK(memcmp(out_nomask, reference_nomask, sizeof out_nomask) == 0);
    controls.auto_mask = true;

    // Determinism: a second run is bit-identical.
    float out_again[32 * 12] = {};
    CHECK(dlss5_nr::input_lift(frame, controls, lift, 512, out_again, 32 * 12)
          == dlss5_nr::Status::kOk);
    CHECK(memcmp(out, out_again, sizeof out) == 0);

    // Row-stride independence: padded rows give bit-identical results.
    float padded[3 * 6 * 3];
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 4; ++x) {
            for (int c = 0; c < 3; ++c) {
                padded[((size_t)y * 6 + (size_t)x) * 3 + (size_t)c]
                    = pixels[((size_t)y * 4 + (size_t)x) * 3 + (size_t)c];
            }
            for (int c = 0; c < 6; ++c) {  // padding pixels x = 4, 5 beyond cols*3
                padded[((size_t)y * 6 + 4) * 3 + (size_t)c] = kSentinel;
            }
        }
    }
    dlss5_nr::FrameView padded_frame = frame;
    padded_frame.rgb = padded;
    padded_frame.row_stride = 18;
    float out_padded[32 * 12] = {};
    CHECK(dlss5_nr::input_lift(padded_frame, controls, lift, 512, out_padded, 32 * 12)
          == dlss5_nr::Status::kOk);
    CHECK(memcmp(out, out_padded, sizeof out) == 0);

    // Changed-input sensitivity and spatial locality: one changed colour sample moves only
    // that pixel's lifted output; every other pixel stays bit-identical.
    float changed[3 * 4 * 3];
    memcpy(changed, pixels, sizeof pixels);
    changed[(2 * 4 + 1) * 3 + 1] = changed[(2 * 4 + 1) * 3 + 1] + 1.0f / 255.0f;  // (x=1, y=2)
    dlss5_nr::FrameView changed_frame = frame;
    changed_frame.rgb = changed;
    float out_changed[32 * 12] = {};
    CHECK(dlss5_nr::input_lift(changed_frame, controls, lift, 512, out_changed, 32 * 12)
          == dlss5_nr::Status::kOk);
    bool pixel_moved = false;
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 4; ++x) {
            for (int c = 0; c < 32; ++c) {
                size_t index = (size_t)c * 12 + (size_t)y * 4 + (size_t)x;
                if (x == 1 && y == 2) {
                    pixel_moved = pixel_moved || out_changed[index] != out[index];
                } else {
                    CHECK(out_changed[index] == out[index]);
                }
            }
        }
    }
    CHECK(pixel_moved);

    // Seed sensitivity with the select lift: the three noise channels move at every pixel.
    float select[512] = {};
    select_lift(select);
    float out_seed0[32 * 12] = {};
    float out_seed1[32 * 12] = {};
    dlss5_nr::Controls controls1 = controls;
    controls1.seed = 1;
    CHECK(dlss5_nr::input_lift(frame, controls, select, 512, out_seed0, 32 * 12)
          == dlss5_nr::Status::kOk);
    CHECK(dlss5_nr::input_lift(frame, controls1, select, 512, out_seed1, 32 * 12)
          == dlss5_nr::Status::kOk);
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 4; ++x) {
            size_t pixel = (size_t)y * 4 + (size_t)x;
            CHECK(out_seed0[pixel] != out_seed1[pixel]
                  || out_seed0[12 + pixel] != out_seed1[12 + pixel]
                  || out_seed0[24 + pixel] != out_seed1[24 + pixel]);
        }
    }
    // Conditioning sensitivity: the tone feature moves at every pixel.
    dlss5_nr::Controls controls2 = controls;
    controls2.local_tone = 1.75f;
    float out_tone[32 * 12] = {};
    CHECK(dlss5_nr::input_lift(frame, controls2, select, 512, out_tone, 32 * 12)
          == dlss5_nr::Status::kOk);
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 4; ++x) {
            size_t pixel = (size_t)y * 4 + (size_t)x;
            CHECK(out_tone[11 * 12 + pixel] == 1.75f);
            CHECK(out_tone[11 * 12 + pixel] != out_seed0[11 * 12 + pixel]);
        }
    }
}

static void check_negative_controls(void) {
    float pixels[2 * 2 * 3];
    synthetic_frame(pixels, 2, 2);
    float stored[512] = {};
    synthetic_stored(stored);
    float lift[512] = {};
    CHECK(dlss5_nr::decode_lift(stored, 512, lift) == dlss5_nr::Status::kOk);
    dlss5_nr::FrameView frame = {};
    frame.rgb = pixels;
    frame.rows = 2;
    frame.cols = 2;
    frame.row_stride = 6;
    dlss5_nr::Controls controls = {};
    float out[32 * 4];
    float baseline[32 * 4];
    CHECK(dlss5_nr::input_lift(frame, controls, lift, 512, out, 32 * 4) == dlss5_nr::Status::kOk);
    memcpy(baseline, out, sizeof out);

    auto rejected = [&](const dlss5_nr::FrameView& f, const dlss5_nr::Controls& c,
                        const float* l, size_t lc, float* o, size_t oc,
                        dlss5_nr::Status expected) {
        if (o != nullptr) {
            fill_sentinel(o, 32 * 4);
        }
        CHECK(dlss5_nr::input_lift(f, c, l, lc, o, oc) == expected);
        if (o != nullptr) {
            for (size_t i = 0; i < 32 * 4; ++i) {
                CHECK(o[i] == kSentinel);  // validation precedes any output write
            }
        }
    };

    // Invalid frames.
    dlss5_nr::FrameView bad = frame;
    bad.rgb = nullptr;
    rejected(bad, controls, lift, 512, out, 32 * 4, dlss5_nr::Status::kNullPointer);
    bad = frame;
    bad.rows = 0;
    rejected(bad, controls, lift, 512, out, 32 * 4, dlss5_nr::Status::kInvalidDimensions);
    bad = frame;
    bad.cols = -1;
    rejected(bad, controls, lift, 512, out, 32 * 4, dlss5_nr::Status::kInvalidDimensions);
    bad = frame;
    bad.rows = dlss5_nr::kMaxRows + 1;
    rejected(bad, controls, lift, 512, out, 32 * 4, dlss5_nr::Status::kInvalidDimensions);
    bad = frame;
    bad.cols = dlss5_nr::kMaxCols + 1;
    rejected(bad, controls, lift, 512, out, 32 * 4, dlss5_nr::Status::kInvalidDimensions);
    bad = frame;
    bad.rows = 16384;
    bad.cols = 16384;
    bad.row_stride = 3 * 16384;
    rejected(bad, controls, lift, 512, out, 32 * 4, dlss5_nr::Status::kInvalidDimensions);
    bad = frame;
    bad.row_stride = 5;
    rejected(bad, controls, lift, 512, out, 32 * 4, dlss5_nr::Status::kInvalidStride);
    bad = frame;
    bad.row_stride = -6;
    rejected(bad, controls, lift, 512, out, 32 * 4, dlss5_nr::Status::kInvalidStride);
    // Invalid pixel values.
    float tainted[2 * 2 * 3];
    for (int variant = 0; variant < 3; ++variant) {
        memcpy(tainted, pixels, sizeof pixels);
        tainted[11] = variant == 0 ? (float)NAN : variant == 1 ? 1.5f : -0.25f;
        bad = frame;
        bad.rgb = tainted;
        rejected(bad, controls, lift, 512, out, 32 * 4, dlss5_nr::Status::kInvalidValues);
    }
    // Invalid controls.
    dlss5_nr::Controls bad_controls = controls;
    bad_controls.style = 3;
    rejected(frame, bad_controls, lift, 512, out, 32 * 4, dlss5_nr::Status::kInvalidControls);
    bad_controls = controls;
    bad_controls.style = -1;
    rejected(frame, bad_controls, lift, 512, out, 32 * 4, dlss5_nr::Status::kInvalidControls);
    bad_controls = controls;
    bad_controls.local_tone = 2.5f;
    rejected(frame, bad_controls, lift, 512, out, 32 * 4, dlss5_nr::Status::kInvalidControls);
    bad_controls = controls;
    bad_controls.local_tone = (float)NAN;
    rejected(frame, bad_controls, lift, 512, out, 32 * 4, dlss5_nr::Status::kInvalidControls);
    bad_controls = controls;
    bad_controls.local_structure = -0.5f;
    rejected(frame, bad_controls, lift, 512, out, 32 * 4, dlss5_nr::Status::kInvalidControls);
    bad_controls = controls;
    bad_controls.skin_structure = -1.5f;
    rejected(frame, bad_controls, lift, 512, out, 32 * 4, dlss5_nr::Status::kInvalidControls);
    bad_controls = controls;
    bad_controls.skin_structure = 2.5f;
    rejected(frame, bad_controls, lift, 512, out, 32 * 4, dlss5_nr::Status::kInvalidControls);
    // Invalid weights/buffers.
    rejected(frame, controls, nullptr, 512, out, 32 * 4, dlss5_nr::Status::kNullPointer);
    rejected(frame, controls, lift, 511, out, 32 * 4, dlss5_nr::Status::kInvalidBuffer);
    rejected(frame, controls, lift, 513, out, 32 * 4, dlss5_nr::Status::kInvalidBuffer);
    rejected(frame, controls, lift, 0, out, 32 * 4, dlss5_nr::Status::kInvalidBuffer);
    float bad_lift[512];
    memcpy(bad_lift, lift, sizeof lift);
    bad_lift[100] = (float)NAN;
    rejected(frame, controls, bad_lift, 512, out, 32 * 4, dlss5_nr::Status::kInvalidValues);
    memcpy(bad_lift, lift, sizeof lift);
    bad_lift[100] = 0.1f;  // finite but off the f16 grid
    rejected(frame, controls, bad_lift, 512, out, 32 * 4, dlss5_nr::Status::kInvalidValues);
    memcpy(bad_lift, lift, sizeof lift);
    bad_lift[100] = (float)INFINITY;  // f16-representable but not a finite weight
    rejected(frame, controls, bad_lift, 512, out, 32 * 4, dlss5_nr::Status::kInvalidValues);
    rejected(frame, controls, lift, 512, nullptr, 32 * 4, dlss5_nr::Status::kNullPointer);
    rejected(frame, controls, lift, 512, out, 32 * 4 - 1, dlss5_nr::Status::kInvalidBuffer);
    // Check precedence: dimensions are rejected before the null output pointer.
    bad = frame;
    bad.rows = -2;
    fill_sentinel(out, 32 * 4);
    CHECK(dlss5_nr::input_lift(bad, controls, lift, 512, nullptr, 0)
          == dlss5_nr::Status::kInvalidDimensions);
    // Valid call still succeeds and reproduces the baseline exactly.
    fill_sentinel(out, 32 * 4);
    CHECK(dlss5_nr::input_lift(frame, controls, lift, 512, out, 32 * 4) == dlss5_nr::Status::kOk);
    CHECK(memcmp(out, baseline, sizeof out) == 0);

    // decode_lift / validate_weights / stage API negative controls.
    float decoded[512];
    CHECK(dlss5_nr::decode_lift(nullptr, 512, decoded) == dlss5_nr::Status::kNullPointer);
    CHECK(dlss5_nr::decode_lift(stored, 512, nullptr) == dlss5_nr::Status::kNullPointer);
    fill_sentinel(decoded, 512);
    CHECK(dlss5_nr::decode_lift(stored, 510, decoded) == dlss5_nr::Status::kInvalidBuffer);
    bool decode_untouched = true;
    for (size_t i = 0; i < 512; ++i) {
        decode_untouched = decode_untouched && (decoded[i] == kSentinel);
    }
    CHECK(decode_untouched);  // a rejected decode leaves the output buffer untouched
    CHECK(dlss5_nr::validate_weights(nullptr, 512) == dlss5_nr::Status::kNullPointer);
    CHECK(dlss5_nr::validate_weights(stored, 1) == dlss5_nr::Status::kInvalidBuffer);
    float uniforms[4];
    CHECK(dlss5_nr::noise_uniforms(0, 0, 0, nullptr) == dlss5_nr::Status::kNullPointer);
    CHECK(dlss5_nr::noise_features(0, 0, 0, nullptr) == dlss5_nr::Status::kNullPointer);
    float rgb[3] = { 0.5f, 0.5f, 0.5f };
    float colour[3];
    CHECK(dlss5_nr::centred_colour(nullptr, colour) == dlss5_nr::Status::kNullPointer);
    CHECK(dlss5_nr::centred_colour(rgb, nullptr) == dlss5_nr::Status::kNullPointer);
    float condition[5];
    CHECK(dlss5_nr::conditioning(controls, nullptr) == dlss5_nr::Status::kNullPointer);
    bad_controls = controls;
    bad_controls.style = 9;
    CHECK(dlss5_nr::conditioning(bad_controls, condition) == dlss5_nr::Status::kInvalidControls);
    CHECK(dlss5_nr::noise_uniforms(0, 0, 0, uniforms) == dlss5_nr::Status::kOk);
    (void)uniforms;
}

static void check_allocation_discipline(void) {
    float pixels[2 * 2 * 3];
    synthetic_frame(pixels, 2, 2);
    float stored[512] = {};
    synthetic_stored(stored);
    float lift[512] = {};
    float out[32 * 4] = {};
    dlss5_nr::FrameView frame = {};
    frame.rgb = pixels;
    frame.rows = 2;
    frame.cols = 2;
    frame.row_stride = 6;
    dlss5_nr::Controls controls = {};
    unsigned long long before = g_allocations.load();
    CHECK(dlss5_nr::decode_lift(stored, 512, lift) == dlss5_nr::Status::kOk);
    CHECK(dlss5_nr::input_lift(frame, controls, lift, 512, out, 32 * 4) == dlss5_nr::Status::kOk);
    float features[3];
    CHECK(dlss5_nr::noise_features(1, 1, 2, features) == dlss5_nr::Status::kOk);
    float colour[3];
    CHECK(dlss5_nr::centred_colour(pixels, colour) == dlss5_nr::Status::kOk);
    unsigned long long after = g_allocations.load();
    CHECK(after == before);  // the whole slice is caller-allocated
}

int main(void) {
    check_f16_literals();
    check_f16_sweep();
    check_lift_permutation();
    check_noise_literals();
    check_centred_colour();
    check_end_to_end_known_answers();
    check_sensitivity_and_locality();
    check_negative_controls();
    check_allocation_discipline();
    printf("native-input-lift-test: %llu checks passed\n", g_checks);
    return 0;
}
