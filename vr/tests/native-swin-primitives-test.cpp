// Independent tests for the native Swin body scalar primitives (vr/native-swin-primitives.hpp).
// CPU-only, no GPU, headset, proprietary payload, Python or PyTorch runtime required.
//
// Independence: the norm oracle is a scalar re-derivation of the documented formula.
// It shares no arithmetic code with the production routines, but it does reuse the
// shared f16 rounding helper (dlss5_nr::f16_from_double) and the shared constants,
// so its binary16 steps are not independent of production; those steps are pinned by
// the hand-derived literals. Same process and libm otherwise, so bit-exact agreement
// is expected and checked exactly. The genuinely independent pins are the
// hand-derived literals: the activation's dyadic rationals, the fma double-rounding
// discriminator, the exact-root reciprocal square roots, and the norm's accumulation
// discriminator (stepwise float32 rounding with the binary16 midpoint tie, versus a
// double-accumulation alternative proven distinct). Random corpora do NOT separate
// the accumulation semantics at the binary16 output — the hand-derived discriminator
// is the pin, as documented in vr/NATIVE-SWIN-PRIMITIVES.md.

#include "native-swin-primitives.hpp"

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

static bool same_value(float a, float b) {
    return (a == b) || (isnan(a) && isnan(b));
}

static uint32_t g_rng = 0x2468ace0u;
static uint32_t rng(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}
static float random_unit(void) {
    return (float)(rng() & 0xffffffu) / (float)0x1000000u;
}

// ---- independent oracles -------------------------------------------------------------------------
// Scalar re-derivations of the documented formulae, written from the specification.
// They call no production routine except the shared f16 rounding helper
// (dlss5_nr::f16_from_double) and the shared constants, whose f16 steps are pinned
// independently by the hand literals. Same process and libm otherwise, so bit-exact
// agreement is expected and checked exactly.
static float oracle_activation(float x) {
    float t = x < -4.0f ? -4.0f : (x > 4.0f ? 4.0f : x);
    float mag = t < 0.0f ? -t : t;
    float inner = (float)((double)mag * (double)dlss5_nr::kActCubic
        + (double)dlss5_nr::kActQuadratic);
    float ramp = (float)((double)t * (double)inner + (double)dlss5_nr::kActLinear);
    return x * ramp;
}

static float oracle_rsqrt(float x) {
    float root = (float)sqrt((double)x);
    return 1.0f / root;
}

static float oracle_norm_scale(const float* values) {
    double squares[2][2][4][2];
    for (int i = 0; i < 32; ++i) {
        double v = (double)values[i];
        squares[(i >> 4) & 1][(i >> 3) & 1][(i >> 1) & 3][i & 1] = v * v;
    }
    double totals[4][2] = {};
    for (int g = 0; g < 2; ++g) {
        for (int f = 0; f < 2; ++f) {
            for (int c = 0; c < 4; ++c) {
                for (int h = 0; h < 2; ++h) {
                    totals[c][h] = (double)(float)(squares[f][g][c][h] + totals[c][h]);
                }
            }
        }
    }
    float halves[2];
    for (int h = 0; h < 2; ++h) {
        float a = (float)totals[0][h] + (float)totals[1][h];
        float b = (float)totals[2][h] + (float)totals[3][h];
        halves[h] = dlss5_nr::f16_from_double((double)(a + b));
    }
    float total = dlss5_nr::f16_from_double((double)halves[0] + (double)halves[1]);
    if (total < dlss5_nr::kNormFloor) {
        total = dlss5_nr::kNormFloor;
    }
    return oracle_rsqrt(total);
}

// The double-accumulation alternative: each chain's four squares accumulate in
// double (removing only the per-step float32 chain rounding; the pair sums and the
// half narrowing round as in the production path). Used to prove the production
// stepwise rounding observable where the hand literal says it is.
static float variant_norm_double(const float* values) {
    double chains[4][2] = {};
    for (int i = 0; i < 32; ++i) {
        double v = (double)values[i];
        chains[(i >> 1) & 3][i & 1] += v * v;
    }
    float halves[2];
    for (int h = 0; h < 2; ++h) {
        float a = (float)chains[0][h] + (float)chains[1][h];
        float b = (float)chains[2][h] + (float)chains[3][h];
        halves[h] = dlss5_nr::f16_from_double((double)(a + b));
    }
    float total = dlss5_nr::f16_from_double((double)halves[0] + (double)halves[1]);
    if (total < dlss5_nr::kNormFloor) {
        total = dlss5_nr::kNormFloor;
    }
    return oracle_rsqrt(total);
}

// ---- activation literals ---------------------------------------------------------------------------
// Every literal is a hand-derived dyadic rational of the constants -229/4096,
// 229/512 and 229/256 (see vr/NATIVE-SWIN-PRIMITIVES.md).
static void check_activation_literals(void) {
    using dlss5_nr::swin_activation;
    CHECK(swin_activation(0.0f) == 0.0f);
    CHECK(!signbit(swin_activation(0.0f)));
    CHECK(swin_activation(-0.0f) == 0.0f);
    CHECK(signbit(swin_activation(-0.0f)));
    // t = 1: inner = 1603/4096; ramp = 5267/4096.
    CHECK(swin_activation(1.0f) == 1.285888671875f);
    // t = -1: ramp = 2061/4096, sign from x.
    CHECK(swin_activation(-1.0f) == -0.503173828125f);
    // t = 1/2: inner = 1717.5/4096; ramp = 1.10418701171875.
    CHECK(swin_activation(0.5f) == 0.552093505859375f);
    // t = 2: inner = 1374/4096; ramp = 1.5654296875.
    CHECK(swin_activation(2.0f) == 3.130859375f);
    CHECK(swin_activation(-2.0f) == -0.447265625f);
    // t = 4 (the clamp): inner = 916/4096; ramp = 229/128.
    CHECK(swin_activation(4.0f) == 7.15625f);
    // At x = -4 the ramp fma (the second of the two) cancels to +0.0 and the outer
    // multiply carries x's sign.
    CHECK(swin_activation(-4.0f) == 0.0f);
    CHECK(signbit(swin_activation(-4.0f)));
    // Beyond the clamp the ramp saturates but the outer factor does not.
    CHECK(swin_activation(4.5f) == 8.05078125f);
    CHECK(swin_activation(-4.5f) == 0.0f);
    CHECK(signbit(swin_activation(-4.5f)));
    CHECK(swin_activation(10.0f) == 17.890625f);
    CHECK(swin_activation(-10.0f) == 0.0f);
    CHECK(signbit(swin_activation(-10.0f)));
    // Interior non-power-of-two: t = 15/4 gives inner = 3893/16384 and
    // ramp = 1.7855682373046875 (exact dyadics throughout).
    CHECK(swin_activation(3.75f) == 6.695880889892578f);
    // t = 1/8: inner = 14427/32768.
    CHECK(swin_activation(0.125f) == 0.11869573593139648f);
    // Saturation at large finite x keeps the unclamped outer multiply.
    CHECK(same_value(swin_activation(1e38f), 1.789062470530969e+38f));
}

// ---- fma semantics ---------------------------------------------------------------------------------
static void check_fma_semantics(void) {
    using dlss5_nr::swin_fma;
    // Double-rounding discriminator: (1 + 2^-23)(1 - 2^-23) = 1 - 2^-46 exactly in
    // double; plus (-1) gives -2^-46, representable in float32. A float32
    // multiply-add rounds the product to 1.0 first and gives 0.
    const float near_up = 1.0f + 0x1p-23f;
    const float near_down = 1.0f - 0x1p-23f;
    const float product = near_up * near_down;
    const float naive = product + -1.0f;
    CHECK(naive == 0.0f);
    CHECK(swin_fma(near_up, near_down, -1.0f) == -0x1p-46f);
    CHECK(swin_fma(near_up, near_down, -1.0f) != naive);
    // Exact cases.
    CHECK(swin_fma(2.0f, 3.0f, 4.0f) == 10.0f);
    CHECK(swin_fma(0.5f, dlss5_nr::kActCubic, dlss5_nr::kActQuadratic)
        == 0.4193115234375f);
    CHECK(swin_fma(-4.0f, 0.2236328125f, dlss5_nr::kActLinear) == 0.0f);
    CHECK(!signbit(swin_fma(-4.0f, 0.2236328125f, dlss5_nr::kActLinear)));
    CHECK(swin_fma(0.0f, 3.0f, -0.5f) == -0.5f);
}

// ---- reciprocal square root literals ----------------------------------------------------------------
static void check_rsqrt_literals(void) {
    using dlss5_nr::swin_rsqrt;
    CHECK(swin_rsqrt(4.0f) == 0.5f);
    CHECK(swin_rsqrt(1.0f) == 1.0f);
    CHECK(swin_rsqrt(0.25f) == 2.0f);
    // sqrt(6.25) = 2.5 exactly; 1/2.5 rounds to the f32 nearest of 0.4.
    CHECK(swin_rsqrt(6.25f) == 0.4000000059604645f);
    // Definition-derived CPU bits (sqrt in double, round to f32, exact division).
    CHECK(swin_rsqrt(0.5f) == 1.4142135381698608f);
    CHECK(swin_rsqrt(2.0f) == 0.7071067690849304f);
    CHECK(swin_rsqrt(8.0f) == 0.3535533845424652f);
    CHECK(swin_rsqrt(2.5f) == 0.6324555277824402f);
    CHECK(swin_rsqrt(dlss5_nr::kNormFloor) == 127.01156616210938f);
    // An overflowed sum maps to a zero scale through this point.
    CHECK(swin_rsqrt(INFINITY) == 0.0f);
}

// ---- norm literals ---------------------------------------------------------------------------------
static void check_norm_literals(void) {
    using dlss5_nr::kNormFloor;
    using dlss5_nr::swin_norm_scale;
    float values[32];
    // All-zero vector: the sum clamps at kNormFloor.
    for (int i = 0; i < 32; ++i) {
        values[i] = 0.0f;
    }
    CHECK(swin_norm_scale(values) == 127.01156616210938f);
    // A single unit value: chain 1.0 -> halves [1, 0] -> total 1 -> scale 1.
    values[0] = 1.0f;
    CHECK(swin_norm_scale(values) == 1.0f);
    // Sensitivity along that slot: v = 1/2 -> 2.0, v = 2 -> 0.5.
    values[0] = 0.5f;
    CHECK(swin_norm_scale(values) == 2.0f);
    values[0] = 2.0f;
    CHECK(swin_norm_scale(values) == 0.5f);
    values[0] = 1.0f;
    // Uniform vector: every chain sums to 1, halves [4, 4], total 8.
    for (int i = 0; i < 32; ++i) {
        values[i] = 0.5f;
    }
    CHECK(swin_norm_scale(values) == 0.3535533845424652f);
    // Below-midpoint rounding in the halves add: v[0] = 1 (half 0) and v[3] = 2^-6
    // (half 1) give halves [1, 2^-12]. Their binary16 sum 1 + 2^-12 lies strictly
    // below the f16 midpoint 1 + 2^-11 and rounds DOWN to 1.0; the scale is 1.0,
    // while rounding up would give rsqrt(1 + 2^-10). (The genuine f16 midpoint tie
    // is pinned by the accumulation discriminator below.)
    for (int i = 0; i < 32; ++i) {
        values[i] = 0.0f;
    }
    values[0] = 1.0f;
    values[3] = 0x1p-6f;
    CHECK(swin_norm_scale(values) == 1.0f);
    CHECK(dlss5_nr::swin_rsqrt(1.0f + 0x1p-10f) == 0.9995120763778687f);
    CHECK(swin_norm_scale(values) != dlss5_nr::swin_rsqrt(1.0f + 0x1p-10f));
    // Non-tie contrast: v[3] = 2^-5 gives halves [1, 2^-10] and total 1 + 2^-10.
    values[3] = 0x1p-5f;
    CHECK(swin_norm_scale(values) == 0.9995120763778687f);
    // Accumulation discriminator: v[0] = 1 in chain (c0, h0) with two 2^-24
    // squares (v[8], v[16]) landing on 1.0 one at a time — each is an exact
    // tie at the float32 midpoint and ties to even DOWN, so the documented
    // stepwise chain stays 1.0. A double accumulation keeps 1 + 2^-23. The
    // second chain of that same first half, (c1, h0), carries 2^-11
    // (v[2] = v[10] = 2^-6, both h = 0), putting the half-0 sum at
    // the binary16 midpoint 1 + 2^-11: the documented order ties DOWN to 1.0
    // (scale 1.0) while the double accumulation lands above the midpoint and
    // rounds UP to 1 + 2^-10 (scale 0.9995120763778687) — the two semantics
    // are observably distinct through the whole stage.
    for (int i = 0; i < 32; ++i) {
        values[i] = 0.0f;
    }
    values[0] = 1.0f;
    values[8] = 0x1p-12f;
    values[16] = 0x1p-12f;
    values[2] = 0x1p-6f;
    values[10] = 0x1p-6f;
    CHECK(swin_norm_scale(values) == 1.0f);
    CHECK(variant_norm_double(values) == 0.9995120763778687f);
    CHECK(swin_norm_scale(values) != variant_norm_double(values));
    // Overflow: every chain overflows binary16 and the scale is zero.
    for (int i = 0; i < 32; ++i) {
        values[i] = 256.0f;
    }
    CHECK(swin_norm_scale(values) == 0.0f);
    // Sign-insensitivity: the squares wash out the signs.
    for (int i = 0; i < 32; ++i) {
        values[i] = (i % 2 == 0) ? 1.0f : -2.0f;
    }
    float unsigned_copy[32];
    for (int i = 0; i < 32; ++i) {
        unsigned_copy[i] = (i % 2 == 0) ? 1.0f : 2.0f;
    }
    CHECK(swin_norm_scale(values) == swin_norm_scale(unsigned_copy));
    CHECK(swin_norm_scale(values) != 0.0f);
    // Determinism: identical inputs, bit-identical outputs.
    float out_a = swin_norm_scale(values);
    float out_b = swin_norm_scale(values);
    CHECK(memcmp(&out_a, &out_b, sizeof out_a) == 0);
}

// ---- norm oracle + sensitivity ---------------------------------------------------------------------
static void check_norm_oracle(void) {
    using dlss5_nr::swin_norm_scale;
    for (int i = 0; i < 100000; ++i) {
        float values[32];
        for (int c = 0; c < 32; ++c) {
            const int exponent = (int)(rng() % 18u) - 12;
            const float sign = (rng() & 1u) ? 1.0f : -1.0f;
            values[c] = sign * random_unit() * ldexpf(1.0f, exponent);
        }
        CHECK(same_value(swin_norm_scale(values), oracle_norm_scale(values)));
        if (i % 100 == 0) {
            float rerun = swin_norm_scale(values);
            float first = swin_norm_scale(values);
            CHECK(memcmp(&first, &rerun, sizeof first) == 0);
        }
    }
    // Activation oracle over randomised finite inputs across the clamp band.
    for (int i = 0; i < 20000; ++i) {
        const int exponent = (int)(rng() % 16u) - 8;
        const float sign = (rng() & 1u) ? 1.0f : -1.0f;
        const float x = sign * (0.25f + random_unit() * 7.75f) * ldexpf(1.0f, exponent);
        CHECK(same_value(dlss5_nr::swin_activation(x), oracle_activation(x)));
    }
    // Reciprocal square root oracle over positive finite inputs.
    for (int i = 0; i < 20000; ++i) {
        const int exponent = (int)(rng() % 40u) - 20;
        const float x = (0.25f + random_unit() * 0.75f) * ldexpf(1.0f, exponent);
        CHECK(same_value(dlss5_nr::swin_rsqrt(x), oracle_rsqrt(x)));
    }
}

// ---- allocation discipline --------------------------------------------------------------------------
static void check_allocation_discipline(void) {
    const unsigned long long before = g_allocations.load();
    float values[32];
    for (int i = 0; i < 32; ++i) {
        values[i] = 0.25f * (float)(i + 1);
    }
    for (int i = 0; i < 64; ++i) {
        (void)dlss5_nr::swin_activation(0.125f * (float)i - 2.0f);
        (void)dlss5_nr::swin_fma(0.5f, 0.25f, -0.125f);
        (void)dlss5_nr::swin_rsqrt(0.5f + (float)i);
        (void)dlss5_nr::swin_norm_scale(values);
    }
    CHECK(g_allocations.load() == before);
}

int main(void) {
    check_activation_literals();
    check_fma_semantics();
    check_rsqrt_literals();
    check_norm_literals();
    check_norm_oracle();
    check_allocation_discipline();
    printf("native-swin-primitives-test: %llu checks passed\n", g_checks);
    return 0;
}
