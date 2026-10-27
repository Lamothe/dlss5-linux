// SPDX-License-Identifier: MIT
// Independent known-answer tests for vr/native-vit.hpp — the native CPU
// vision-transformer bottleneck family. Hand-derived answers only (every
// float16/e4m3 step derived on paper from the pinned reference semantics); no
// Python, no torch, no proprietary weights.

#include "../native-vit.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace dlss5_nr;

static long long g_checks = 0;

static void check(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        std::fprintf(stderr, "FAILED (%d): %s\n", line, expr);
        std::exit(1);
    }
}

static void check_close(float got, float want, float tol, const char* expr,
                        int line) {
    ++g_checks;
    if (!(std::fabs(got - want) <= tol)) {
        std::fprintf(stderr, "FAILED (%d): %s got %.9g want %.9g\n", line, expr,
                     static_cast<double>(got), static_cast<double>(want));
        std::exit(1);
    }
}

static void fill_bytes(void* dst, std::size_t n, unsigned char value) {
    std::memset(dst, static_cast<int>(value), n);
}

#define CHECK(expr) check((expr), #expr, __LINE__)
#define CHECK_CLOSE(got, want, tol) \
    check_close((got), (want), (tol), #got " ~= " #want, __LINE__)

int main(void) {
    // ---- approximate_exp anchors (y = round((x*slope+offset)*1024), clamp,
    // mantissa -> ldexp). Every anchor derived by hand:
    //   x = 0:   y = 1750, mantissa 726, exponent -4, fraction 22/64
    //            -> ldexp(1.34375, -4) = 0.083984375
    //   x = 1:   y = round(1841.6875) = 1842, mantissa 818, exponent -3,
    //            fraction 50/64 -> ldexp(1.78125, -3) = 0.22265625
    //   x = 100: clamped to 2025, mantissa 1001, exponent 0, fraction 41/64
    //            -> 1.640625
    //   x = -100: clamped to 1474, mantissa 450, exponent -8, fraction 2/64
    //            -> ldexp(1.03125, -8) = 0.0040283203125
    CHECK_CLOSE(approximate_exp(0.0f), 0.083984375f, 0.0f);
    CHECK_CLOSE(approximate_exp(1.0f), 0.22265625f, 0.0f);
    CHECK_CLOSE(approximate_exp(100.0f), 1.640625f, 0.0f);
    CHECK_CLOSE(approximate_exp(-100.0f), 0.0040283203125f, 0.0f);
    // The logit is rounded to float16 first: extreme magnitudes stay finite.
    CHECK(std::isfinite(approximate_exp(1e30f)));
    CHECK_CLOSE(approximate_exp(1e30f), 1.640625f, 0.0f);

    // ---- ffn_activation anchors (clipped cubic SiLU):
    //   act(0)  = 0
    //   act(2)  = 2 * (-0.2236328125 + 0.89453125 + 0.89453125) = 3.130859375
    //   act(-2) = -2 * (0.2236328125 - 0.89453125 + 0.89453125) = -0.447265625
    //   act(5)  = 5 * (-0.89453125 + 1.7890625 + 0.89453125) = 8.9453125
    //   act(-5) = -5 * (0.89453125 - 1.7890625 + 0.89453125) = 0
    CHECK_CLOSE(ffn_activation(0.0f), 0.0f, 0.0f);
    CHECK_CLOSE(ffn_activation(2.0f), 3.130859375f, 0.0f);
    CHECK_CLOSE(ffn_activation(-2.0f), -0.447265625f, 0.0f);
    CHECK_CLOSE(ffn_activation(5.0f), 8.9453125f, 0.0f);
    CHECK_CLOSE(ffn_activation(-5.0f), 0.0f, 0.0f);

    // ---- inverse_norm_head: one head, |x| = 3 on channel 0:
    // squares 9 exactly; the butterfly keeps 9; 1/sqrt(9) in f32 is
    // 0.33333334f; f16 rounds it to 1365 * 2^-12 = 0.333251953125.
    {
        float head[kVitHeadDim] = {0.0f};
        head[0] = 3.0f;
        CHECK_CLOSE(inverse_norm_head(head), 0.333251953125f, 0.0f);
        // Two channels of 2: squares 4 + 4; the pair d with d+16 keeps 8;
        // 1/sqrt(8) in f32 lands at 0.35355339f, whose f16 rounding is kept
        // inside a narrow bracket by this range check.
        float head2[kVitHeadDim] = {0.0f};
        head2[0] = 2.0f;
        head2[16] = 2.0f;
        const float inv2 = inverse_norm_head(head2);
        CHECK(inv2 > 0.3535f && inv2 < 0.3536f);
    }

    // ---- matmul_e4m3: hand-computed dot products on the e4m3 grid.
    {
        // tokens=2, c_in=3, rows=2. Weights (e4m3):
        //   n0 = [1, 2, 0.5]; n1 = [0, 4, -1]
        const float in[6] = {1.0f, 2.0f, 4.0f,   // t0
                             -2.0f, 0.5f, 1.0f}; // t1
        std::uint8_t w[6] = {0};
        w[0] = e4m3_bits(1.0f);
        w[1] = e4m3_bits(2.0f);
        w[2] = e4m3_bits(0.5f);
        w[3] = e4m3_bits(0.0f);
        w[4] = e4m3_bits(4.0f);
        w[5] = e4m3_bits(-1.0f);
        float out[4] = {0};
        E4m3Matrix m{w, 2, 3};
        CHECK(matmul_e4m3(in, 2, 3, m, out) == Status::kOk);
        CHECK_CLOSE(out[0], 1.0f * 1.0f + 2.0f * 2.0f + 4.0f * 0.5f, 0.0f);
        CHECK_CLOSE(out[1], 0.0f + 2.0f * 4.0f + 4.0f * -1.0f, 0.0f);
        CHECK_CLOSE(out[2], -2.0f * 1.0f + 0.5f * 2.0f + 1.0f * 0.5f, 0.0f);
        CHECK_CLOSE(out[3], 0.0f + 0.5f * 4.0f + 1.0f * -1.0f, 0.0f);
    }

    // ---- vit_qkv end-to-end anchor: one token, one head, identity q/k/v
    // weights, x[0] = 8, temperature 1.
    //   projected = 8; inv = 1/sqrt(64) = 0.125 exactly;
    //   q = f16(f16(8*5.65625)*0.125) = 5.65625 -> e4m3 -> 5.5
    //      (the [4, 8) e4m3 step is 0.5; 5.65625 is nearest to 5.5)
    //   k = f16(8*0.125) = 1.0 -> e4m3 -> 1.0
    //   v = 8.0 -> e4m3 -> 8.0
    {
        const int channels = kVitHeadDim;
        float in[kVitHeadDim] = {0.0f};
        in[0] = 8.0f;
        std::uint8_t identity[kVitHeadDim * kVitHeadDim] = {0};
        for (int i = 0; i < channels; ++i) {
            identity[i * channels + i] = e4m3_bits(1.0f);
        }
        E4m3Matrix m{identity, channels, channels};
        const float temperature = 1.0f;
        float out[3 * kVitHeadDim] = {0};
        CHECK(vit_qkv(in, 1, channels, m, m, m, &temperature, out) == Status::kOk);
        CHECK_CLOSE(out[0], 5.5f, 0.0f);
        CHECK_CLOSE(out[channels], 1.0f, 0.0f);
        CHECK_CLOSE(out[2 * channels], 8.0f, 0.0f);
        // The remaining head channels stay zero.
        CHECK_CLOSE(out[1], 0.0f, 0.0f);
        CHECK_CLOSE(out[channels + 1], 0.0f, 0.0f);
    }

    // ---- group_sum64: all ones sum to 64 exactly (every f16 add exact).
    {
        float p[kVitKeyGroup];
        for (int i = 0; i < kVitKeyGroup; ++i) {
            p[i] = 1.0f;
        }
        CHECK_CLOSE(group_sum64(p), 64.0f, 0.0f);
        // A sparse asymmetric case: only p[0] = 3 and p[8] = 5 pair up.
        for (int i = 0; i < kVitKeyGroup; ++i) {
            p[i] = 0.0f;
        }
        p[0] = 3.0f;
        p[8] = 5.0f;
        p[20] = 1.5f;  // block 1, pair c = 4
        CHECK_CLOSE(group_sum64(p), 9.5f, 0.0f);
    }

    // ---- vit_attention anchor: one token, one head, q = k = 0, v = (1, 2).
    // Keys pad to 64; every key has logit 0 so p = 0.083984375:
    //   group sum = 5.375 (exact f16), padding = f16(63*0.083984375) = 5.2890625
    //   (tie on the f16 grid, even), den = f16(5.375 - 5.2890625) = 0.0859375,
    //   inverse = f16(1/0.0859375) = 11.6328125,
    //   numerator[0] = e4m3(p)*1 = 0.0859375 -> f16 * inverse = 16379/16384
    //   -> f16 1.0 -> e4m3 1.0.
    //   numerator[1] = 0.0859375*2 = 0.171875 -> *inverse = 16379/8192
    //   = 1.9993896484375 -> f16 1.9990234375 -> e4m3 2.0.
    {
        const int channels = kVitHeadDim;
        float qkv[3 * kVitHeadDim] = {0.0f};
        qkv[2 * channels + 0] = 1.0f;
        qkv[2 * channels + 1] = 2.0f;
        float out[kVitHeadDim] = {0};
        CHECK(vit_attention(qkv, 1, channels, out) == Status::kOk);
        CHECK_CLOSE(out[0], 1.0f, 0.0f);
        CHECK_CLOSE(out[1], 2.0f, 0.0f);
        CHECK_CLOSE(out[2], 0.0f, 0.0f);
    }

    // ---- vit_attention with a full 64-key group (no padding): uniform
    // logits make every probability exp(0); the output is the e4m3 average of
    // the values through the denominator path (derived: 1.0).
    {
        const int channels = kVitHeadDim;
        const int tokens = kVitKeyGroup;
        std::size_t n = static_cast<std::size_t>(tokens) * 3 * channels;
        std::size_t bytes = n * sizeof(float);
        float* qkv = static_cast<float*>(std::malloc(bytes));
        float* out = static_cast<float*>(std::malloc(
            static_cast<std::size_t>(tokens) * channels * sizeof(float)));
        CHECK(qkv != nullptr && out != nullptr);
        std::memset(qkv, 0, bytes);
        for (int t = 0; t < tokens; ++t) {
            qkv[static_cast<std::size_t>(t) * 3 * channels + 2 * channels] = 1.0f;
        }
        CHECK(vit_attention(qkv, tokens, channels, out) == Status::kOk);
        for (int t = 0; t < tokens; ++t) {
            CHECK_CLOSE(out[t * channels], 1.0f, 0.0f);
        }
        std::free(qkv);
        std::free(out);
    }

    // ---- vit_ffn_expand: one token, c_in = 2, c_out = 2.
    //   n0 = [1, 1]: acc 3 -> act(3) = 3 * (-0.055908203125*9 + 1.341796875
    //   + 0.89453125) = 3 * 1.733154296875 = 5.199462890625 -> f16 -> e4m3.
    //   n1 = [-1, 1]: acc 1 -> act(1) = 1 * (-0.055908203125 + 0.447265625
    //   + 0.89453125) = 1.285888671875.
    {
        const float in[2] = {1.0f, 2.0f};
        std::uint8_t w[4] = {e4m3_bits(1.0f), e4m3_bits(1.0f),
                             e4m3_bits(-1.0f), e4m3_bits(1.0f)};
        E4m3Matrix m{w, 2, 2};
        float out[2] = {0};
        CHECK(vit_ffn_expand(in, 1, 2, 2, m, out) == Status::kOk);
        // act(3) = 5.199462890625 exactly (all dyadic); f16 rounds it to
        // 1331 * 2^-8 = 5.19921875 (5.199462890625 * 256 = 1331.0625).
        // e4m3 [4, 8) step 0.5: nearest to 5.0.
        CHECK_CLOSE(out[0], 5.0f, 0.0f);
        // act(1) = 1.285888671875; f16 rounds to 1317 * 2^-10 = 1.2861328125
        // (1.285888671875 * 1024 = 1316.75). e4m3 [1, 2) step 0.125:
        // 10.289 -> 10 -> 1.25.
        CHECK_CLOSE(out[1], 1.25f, 0.0f);
    }

    // ---- vit_residual_linear: product + gain * skip through f16 and e4m3.
    //   acc = 1 * 1.0 = 1.0; gain 2.0 * skip 3.0 = 6.0; 7.0 -> e4m3 7.0.
    {
        const float in[1] = {1.0f};
        const float skip[1] = {3.0f};
        std::uint8_t w[1] = {e4m3_bits(1.0f)};
        E4m3Matrix m{w, 1, 1};
        const float gain[1] = {2.0f};
        float out[1] = {0};
        CHECK(vit_residual_linear(in, skip, 1, 1, 1, m, gain, out) == Status::kOk);
        CHECK_CLOSE(out[0], 7.0f, 0.0f);
        // Rounding path: f16(0.1) = 0.0999755859375; acc 0.5 + that * 1.0 =
        // 0.5999755859375, whose f16 rounding is 0.60009765625 -> e4m3
        // [0.5, 1) step 1/16: 9.6016 -> 10 -> 0.625.
        const float in2[1] = {0.5f};
        const float skip2[1] = {1.0f};
        const float gain2[1] = {0.1f};
        out[0] = 0.0f;
        CHECK(vit_residual_linear(in2, skip2, 1, 1, 1, m, gain2, out) == Status::kOk);
        CHECK_CLOSE(out[0], 0.625f, 0.0f);
    }

    // ---- vit_attention multi-head + multi-group: 2 heads, 128 tokens (two
    // 64-key groups, no padding), q = k = 0, head 0 values 1, head 1 values 2.
    // Per head: p = exp(0) for all keys, two group sums of 5.375 add to
    // 10.75 (f16, exact), inverse = f16(1/10.75) = 1524 * 2^-14 =
    // 0.093017578125. Head 0: numerator = 128 * 0.0859375 = 11.0 exactly;
    // 11.0 * inverse = 1.023193359375 -> f16 1048 * 2^-10 = 1.0234375 ->
    // e4m3 1.0. Head 1 doubles: 22.0 * inverse = 2.04638671875 -> f16
    // 1048 * 2^-9 = 2.046875 -> e4m3 2.0.
    {
        const int channels = 2 * kVitHeadDim;
        const int tokens = 2 * kVitKeyGroup;
        std::size_t n = static_cast<std::size_t>(tokens) * 3 * channels;
        float* qkv = static_cast<float*>(std::malloc(n * sizeof(float)));
        float* out = static_cast<float*>(std::malloc(
            static_cast<std::size_t>(tokens) * channels * sizeof(float)));
        CHECK(qkv != nullptr && out != nullptr);
        std::memset(qkv, 0, n * sizeof(float));
        for (int t = 0; t < tokens; ++t) {
            float* row = qkv + static_cast<std::size_t>(t) * 3 * channels;
            for (int d = 0; d < kVitHeadDim; ++d) {
                row[2 * channels + d] = 1.0f;                        // head 0 v
                row[2 * channels + kVitHeadDim + d] = 2.0f;          // head 1 v
            }
        }
        CHECK(vit_attention(qkv, tokens, channels, out) == Status::kOk);
        for (int t = 0; t < tokens; ++t) {
            CHECK_CLOSE(out[t * channels], 1.0f, 0.0f);
            CHECK_CLOSE(out[t * channels + kVitHeadDim], 2.0f, 0.0f);
        }
        std::free(qkv);
        std::free(out);
    }

    // ---- rejection classes.
    {
        E4m3Matrix empty{nullptr, 1, 1};
        float x = 1.0f;
        float y = 0.0f;
        CHECK(matmul_e4m3(&x, 1, 1, empty, &y) == Status::kNullPointer);
        std::uint8_t bits[1] = {e4m3_bits(1.0f)};
        E4m3Matrix m{bits, 1, 1};
        CHECK(matmul_e4m3(&x, 0, 1, m, &y) == Status::kInvalidDimensions);
        CHECK(matmul_e4m3(&x, 1, 2, m, &y) == Status::kInvalidDimensions);
        CHECK(vit_attention(&x, 1, 3, &y) == Status::kInvalidDimensions);
        CHECK(vit_attention(&x, 0, kVitHeadDim, &y) == Status::kInvalidDimensions);
        CHECK(vit_attention(nullptr, 1, kVitHeadDim, &y) == Status::kNullPointer);
    }

    // ---- sentinel guards: kernel writes stay inside the output buffer.
    {
        const int channels = kVitHeadDim;
        float guarded_in[2 * 3 * kVitHeadDim + 2];
        float guarded_out[kVitHeadDim + 2];
        fill_bytes(guarded_in, sizeof(guarded_in), 0xAA);
        fill_bytes(guarded_out, sizeof(guarded_out), 0x55);
        std::memset(guarded_in, 0, sizeof(float) * 3 * kVitHeadDim);
        CHECK(vit_attention(guarded_in, 1, channels, guarded_out + 1) == Status::kOk);
        for (std::size_t i = 0; i < 3 * kVitHeadDim; ++i) {
            std::uint32_t pattern;
            std::memcpy(&pattern, &guarded_in[i], sizeof(pattern));
            (void)pattern;
        }
        // Leading and trailing guards of the output are untouched.
        std::uint32_t lead;
        std::uint32_t tail;
        std::memcpy(&lead, &guarded_out[0], sizeof(lead));
        std::memcpy(&tail, &guarded_out[kVitHeadDim + 1], sizeof(tail));
        CHECK(lead == 0x55555555u);
        CHECK(tail == 0x55555555u);
    }

    // ---- determinism: identical inputs give identical outputs.
    {
        const int channels = kVitHeadDim;
        float qkv[3 * kVitHeadDim];
        for (int i = 0; i < 3 * channels; ++i) {
            qkv[i] = 0.01f * static_cast<float>((i * 7) % 13) - 0.06f;
        }
        float a[kVitHeadDim] = {0};
        float b[kVitHeadDim] = {0};
        CHECK(vit_attention(qkv, 1, channels, a) == Status::kOk);
        CHECK(vit_attention(qkv, 1, channels, b) == Status::kOk);
        for (int i = 0; i < channels; ++i) {
            CHECK(a[i] == b[i]);
        }
    }

    std::printf("native-vit-test: %lld checks passed\n", g_checks);
    return 0;
}
