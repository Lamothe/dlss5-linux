// SPDX-License-Identifier: MIT
// Independent known-answer tests for vr/native-split-swin.hpp — the native CPU
// "split Swin" layer family (grouped feed-forward, gained 1x1 projections,
// pooled projection, final head, windowed shifted multi-head attention).
// Hand-derived answers only (every float16/e4m3 step derived on paper from the
// pinned reference semantics); no Python, no torch, no proprietary weights.

#include "../native-split-swin.hpp"

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

// Weight records for the fixed-shape layers (BSS zero-filled; the tests set
// only the few entries they need). Synthetic values; no learned weights.
static std::uint8_t g_expand_in[kSplitChannels * kSplitChannels];
static std::uint8_t g_expand_hidden[kSplitGroups * kSplitGroupHidden * kSplitGroupIn];
static std::uint8_t g_contract[kSplitGroups * kSplitGroupIn * kSplitGroupHidden];
static std::uint8_t g_qkv[kSplitQkvOut * kSplitChannels];
static float g_bias[kSplitHeads * kSplitTokensPerWindow * kSplitTokensPerWindow];
static float g_gain[kSplitHeads];
static float g_map[kSplitTokensPerWindow * kSplitChannels];
static float g_out[kSplitTokensPerWindow * kSplitChannels];

// The grouped feed-forward's identity weights: row n of `expand_in` reads map
// channel n; row g * 256 + j of `expand_hidden` reads group input j; row
// g * 64 + j of `contract` reads hidden j.
static void set_ffwd_identity() {
    std::memset(g_expand_in, 0, sizeof g_expand_in);
    std::memset(g_expand_hidden, 0, sizeof g_expand_hidden);
    std::memset(g_contract, 0, sizeof g_contract);
    const std::uint8_t one = e4m3_bits(1.0f);
    for (int n = 0; n < kSplitChannels; ++n) {
        g_expand_in[static_cast<std::size_t>(n) * kSplitChannels + n] = one;
    }
    for (int g = 0; g < kSplitGroups; ++g) {
        for (int j = 0; j < kSplitGroupIn; ++j) {
            g_expand_hidden[static_cast<std::size_t>(g * kSplitGroupHidden + j)
                            * kSplitGroupIn + j] = one;
        }
        for (int j = 0; j < kSplitGroupIn; ++j) {
            g_contract[static_cast<std::size_t>(g * kSplitGroupIn + j)
                       * kSplitGroupHidden + j] = one;
        }
    }
}

// The attention's sparse synthetic weights: only head 0's row 0 (query
// channel 0), row 32 (key channel 0) and row 64 (value channel 0) read map
// channel 0; every other row is zero. All gains are 1, all biases zero.
static void set_attn_sparse() {
    std::memset(g_qkv, 0, sizeof g_qkv);
    std::memset(g_bias, 0, sizeof g_bias);
    const std::uint8_t one = e4m3_bits(1.0f);
    g_qkv[0 * kSplitChannels + 0] = one;
    g_qkv[32 * kSplitChannels + 0] = one;
    g_qkv[64 * kSplitChannels + 0] = one;
    for (int h = 0; h < kSplitHeads; ++h) {
        g_gain[h] = 1.0f;
    }
}

static SplitFfwdWeight ffwd_weight() {
    return SplitFfwdWeight{
        E4m3Matrix{g_expand_in, kSplitChannels, kSplitChannels},
        E4m3Matrix{g_expand_hidden, kSplitGroups * kSplitGroupHidden, kSplitGroupIn},
        E4m3Matrix{g_contract, kSplitGroups * kSplitGroupIn, kSplitGroupHidden},
    };
}

static SplitQkvWeight attn_weight() {
    return SplitQkvWeight{
        E4m3Matrix{g_qkv, kSplitQkvOut, kSplitChannels},
        g_bias,
        g_gain,
    };
}

int main(void) {
    // ---- split_fma_half anchors (exact product and sum in binary64, one
    // rounding to float16):
    //   2 * 3 + 1 = 7.
    //   tie: (1 + 2^-10)^2 - (2^-11 + 2^-20) = 1 + 3 * 2^-11, the float16
    //   midpoint between 1.0009765625 (odd) and 1.001953125 (even) -> the
    //   ties-to-even result is 1.001953125.
    CHECK_CLOSE(split_fma_half(2.0f, 3.0f, 1.0f), 7.0f, 0.0f);
    CHECK_CLOSE(split_fma_half(1.0009765625f, 1.0009765625f,
                               -0.00048923492431640625f),
                1.001953125f, 0.0f);
    CHECK_CLOSE(split_fma_half(0.0f, -1.0f, 0.0f), 0.0f, 0.0f);

    // ---- split_store (float32 -> float16 -> e4m3 double rounding):
    //   f16(68.0001) = 68 and e4m3(68) is the midpoint 64/72 -> ties to even
    //   -> 64, while a direct e4m3 rounding of 68.0001 gives 72.
    CHECK_CLOSE(split_store(68.0001f), 64.0f, 0.0f);
    CHECK_CLOSE(e4m3(68.0001f), 72.0f, 0.0f);
    CHECK_CLOSE(split_store(1.0f), 1.0f, 0.0f);
    CHECK_CLOSE(split_store(2.5f), 2.5f, 0.0f);

    // ---- split_ffn_activation anchors (the float16 cubic SiLU):
    //   act(0)  = 0
    //   act(1)  = f16(1 * v), v = f16(0.391357421875 + 0.89453125)
    //           = f16(5267 * 2^-12) = 1317 * 2^-10 = 1.2861328125
    //   act(-1) = f16(-1 * v), v = f16(0.89453125 - 0.391357421875)
    //           = f16(2061 * 2^-12) = 1030 * 2^-11 = 0.5029296875
    //           (2061 * 2^-12 = 1030.5 * 2^-11 is the midpoint; the even
    //           neighbour 1030 wins) -> -0.5029296875
    //   act(-2) = f16(-2 * 0.2236328125) = -0.447265625
    //   act(-4) = f16(-4 * 0) = -0.0 (the ramp fma cancels at t = -4)
    //   act(10) = f16(10 * 1.7890625) = 17.890625 (clamped ramp, outer x)
    CHECK_CLOSE(split_ffn_activation(0.0f), 0.0f, 0.0f);
    CHECK_CLOSE(split_ffn_activation(1.0f), 1.2861328125f, 0.0f);
    CHECK_CLOSE(split_ffn_activation(-1.0f), -0.5029296875f, 0.0f);
    CHECK_CLOSE(split_ffn_activation(-2.0f), -0.447265625f, 0.0f);
    CHECK_CLOSE(split_ffn_activation(-4.0f), 0.0f, 0.0f);
    CHECK(std::signbit(split_ffn_activation(-4.0f)));
    CHECK_CLOSE(split_ffn_activation(10.0f), 17.890625f, 0.0f);

    // ---- split_exp anchors (y = fma(f16(x), slope, offset) clamped to
    // [1.03125, 1.5693359375]; the result is the float16 whose bits are the
    // ten mantissa bits of y shifted left by five):
    //   x = 0:   y = 1332 * 2^-10, mantissa 308 -> bits 308 << 5 = 0x2680
    //            -> 2^-6 * 1.625 = 0.025390625
    //   x = 1:   y = 1378 * 2^-10, mantissa 354 -> 0x2C40
    //            -> 2^-4 * 1.0625 = 0.06640625
    //   x = -1:  y = 1286 * 2^-10, mantissa 262 -> 0x20C0
    //            -> 2^-7 * 1.1875 = 0.00927734375
    //   x = 2:   y = 1424 * 2^-10, mantissa 400 -> 0x3200
    //            -> 2^-3 * 1.5 = 0.1875
    //   x = 100: clamped to 1607 * 2^-10, mantissa 583 -> 0x48E0
    //            -> 2^3 * 1.21875 = 9.75
    //   x = -100: clamped to 1056 * 2^-10, mantissa 32 -> 0x400 = 2^-14
    CHECK_CLOSE(split_exp(0.0f), 0.025390625f, 0.0f);
    CHECK_CLOSE(split_exp(1.0f), 0.06640625f, 0.0f);
    CHECK_CLOSE(split_exp(-1.0f), 0.00927734375f, 0.0f);
    CHECK_CLOSE(split_exp(2.0f), 0.1875f, 0.0f);
    CHECK_CLOSE(split_exp(100.0f), 9.75f, 0.0f);
    CHECK_CLOSE(split_exp(-100.0f), 6.103515625e-05f, 0.0f);
    // The logit rounds to float16 first: an infinite logit clamps high.
    CHECK_CLOSE(split_exp(1e30f), 9.75f, 0.0f);

    // ---- split_normalise anchors:
    //   [32, 0*7, 2^-6, 0*23]: squares 1024 and 2^-12; the first butterfly
    //   pair keeps 1024 + 0 and 2^-12 + 0; the next sum rounds 1024 + 2^-12
    //   to 1024 (float16 step 1 at that binade). norm = f16(1/sqrt(1024)) =
    //   0.03125 -> out[0] = 1, out[8] = f16(2^-6 * 2^-5) = 2^-11.
    {
        float head[kSplitHeadDim] = {0.0f};
        head[0] = 32.0f;
        head[8] = 0.015625f;
        float out[kSplitHeadDim] = {0.0f};
        split_normalise(head, out);
        CHECK_CLOSE(out[0], 1.0f, 0.0f);
        CHECK_CLOSE(out[8], 0.00048828125f, 0.0f);
        for (int d = 1; d < kSplitHeadDim; ++d) {
            if (d != 8) {
                CHECK_CLOSE(out[d], 0.0f, 0.0f);
            }
        }
    }
    //   [2^-12, 0*31]: the square underflows to the 2^-24 subnormal, below
    //   the floor f16(0.000062) = 1040 * 2^-24 = 6.198883056640625e-05.
    //   norm = f16(2^12 / sqrt(1040)) = f16(127.0115) = 127 ->
    //   out[0] = f16(2^-12 * 127) = 127 * 2^-12 = 0.031005859375.
    {
        float head[kSplitHeadDim] = {0.0f};
        head[0] = 0.000244140625f;
        float out[kSplitHeadDim] = {0.0f};
        split_normalise(head, out);
        CHECK_CLOSE(out[0], 0.031005859375f, 0.0f);
        for (int d = 1; d < kSplitHeadDim; ++d) {
            CHECK_CLOSE(out[d], 0.0f, 0.0f);
        }
    }
    //   32 ones: the butterfly total is 32; norm = f16(1/sqrt(32)) =
    //   f16(0.17677670) = 1448 * 2^-13 = 0.1767578125 -> every output that.
    {
        float head[kSplitHeadDim];
        for (int d = 0; d < kSplitHeadDim; ++d) {
            head[d] = 1.0f;
        }
        float out[kSplitHeadDim] = {0.0f};
        split_normalise(head, out);
        for (int d = 0; d < kSplitHeadDim; ++d) {
            CHECK_CLOSE(out[d], 0.1767578125f, 0.0f);
        }
    }

    // ---- split_softmax_denominator anchors:
    //   64 ones: every tile sum is 4, the pair sums 8, the halves 32 -> 64.
    {
        float leaves[kSplitTokensPerWindow];
        for (int j = 0; j < kSplitTokensPerWindow; ++j) {
            leaves[j] = 1.0f;
        }
        CHECK_CLOSE(split_softmax_denominator(leaves), 64.0f, 0.0f);
    }
    //   The pinned reduction order: tiles are summed ((0+1)+2)+3 per
    //   (half, c). Leaves 2^-25 in tiles 0-2 and 1 in tile 3 of (half 0,
    //   c 0) give 3 * 2^-25 + 1, which rounds up to 1 + 2^-23 in float32;
    //   any grouping that adds 1 earlier keeps 1.0 exactly.
    {
        float leaves[kSplitTokensPerWindow] = {0.0f};
        leaves[0] = 0x1p-25f;
        leaves[16] = 0x1p-25f;
        leaves[32] = 0x1p-25f;
        leaves[48] = 1.0f;
        CHECK_CLOSE(split_softmax_denominator(leaves), 1.0000001192092896f, 0.0f);
    }

    // ---- split_matmul known answers (key-ascending float32 products):
    //   W = [[1, 2, 3], [0.5, -1, 0.25]]; in = [[1, 1, 1], [2, 0, 1]].
    {
        const std::uint8_t bits[2 * 3] = {
            e4m3_bits(1.0f), e4m3_bits(2.0f), e4m3_bits(3.0f),
            e4m3_bits(0.5f), e4m3_bits(-1.0f), e4m3_bits(0.25f),
        };
        const E4m3Matrix w{bits, 2, 3};
        const float in[2 * 3] = {1.0f, 1.0f, 1.0f, 2.0f, 0.0f, 1.0f};
        float out[2 * 2] = {0.0f, 0.0f, 0.0f, 0.0f};
        CHECK(split_matmul(in, 2, 3, w, out) == Status::kOk);
        CHECK_CLOSE(out[0], 6.0f, 0.0f);
        CHECK_CLOSE(out[1], -0.25f, 0.0f);
        CHECK_CLOSE(out[2], 5.0f, 0.0f);
        CHECK_CLOSE(out[3], 1.25f, 0.0f);
        float unused = 0.0f;
        CHECK(split_matmul(nullptr, 2, 3, w, out) == Status::kNullPointer);
        CHECK(split_matmul(in, 2, 3, w, nullptr) == Status::kNullPointer);
        CHECK(split_matmul(in, 0, 3, w, out) == Status::kInvalidDimensions);
        CHECK(split_matmul(in, 2, 2, w, out) == Status::kInvalidDimensions);
        const E4m3Matrix empty{nullptr, 1, 1};
        CHECK(split_matmul(in, 1, 3, empty, &unused) == Status::kNullPointer);
    }

    // ---- split_projection_value anchors (value = f16(W x + gain * skip)):
    //   W = [[1, 2], [0.5, -1]], gain [1, 0.5], x [1, 1], skip [3, 4]:
    //   out = f16(3 + 3) = 6, f16(-0.5 + 2) = 1.5.
    {
        const std::uint8_t bits[2 * 2] = {
            e4m3_bits(1.0f), e4m3_bits(2.0f),
            e4m3_bits(0.5f), e4m3_bits(-1.0f),
        };
        const float gain[2] = {1.0f, 0.5f};
        SplitProjectionWeight w{E4m3Matrix{bits, 2, 2}, gain};
        const float x[2] = {1.0f, 1.0f};
        const float skip[2] = {3.0f, 4.0f};
        float value[2] = {0.0f, 0.0f};
        CHECK(split_projection_value(x, skip, 1, 2, 2, w, value) == Status::kOk);
        CHECK_CLOSE(value[0], 6.0f, 0.0f);
        CHECK_CLOSE(value[1], 1.5f, 0.0f);
    }
    //   The fused sum rounds once to float16: 0 + 1 * 1.00048828125 is the
    //   midpoint between 1.0 and 1.0009765625 -> ties to even -> 1.0, while
    //   1.0009765625 itself is exact.
    {
        const std::uint8_t bits[1] = {e4m3_bits(0.0f)};
        const float gain[1] = {1.0f};
        SplitProjectionWeight w{E4m3Matrix{bits, 1, 1}, gain};
        const float x[1] = {5.0f};
        float value[1] = {-1.0f};
        const float tie_skip[1] = {1.00048828125f};
        CHECK(split_projection_value(x, tie_skip, 1, 1, 1, w, value) == Status::kOk);
        CHECK_CLOSE(value[0], 1.0f, 0.0f);
        const float exact_skip[1] = {1.0009765625f};
        CHECK(split_projection_value(x, exact_skip, 1, 1, 1, w, value) == Status::kOk);
        CHECK_CLOSE(value[0], 1.0009765625f, 0.0f);
        // Residual and gain must appear together.
        CHECK(split_projection_value(x, tie_skip, 1, 1, 1,
                                     SplitProjectionWeight{E4m3Matrix{bits, 1, 1}, nullptr},
                                     value) == Status::kNullPointer);
        SplitProjectionWeight no_gain{E4m3Matrix{bits, 1, 1}, nullptr};
        CHECK(split_projection_value(x, nullptr, 1, 1, 1, no_gain, value) == Status::kOk);
        CHECK(split_projection_value(x, nullptr, 0, 1, 1, no_gain, value)
              == Status::kInvalidDimensions);
        CHECK(split_projection_value(nullptr, nullptr, 1, 1, 1, no_gain, value)
              == Status::kNullPointer);
    }

    // ---- split_ffwd_proj / split_proj anchors (stored projection):
    //   W = [[2]], gain [1]; tokens x [3, 0.5], skip [4, 0] -> 10, 1.
    {
        const std::uint8_t bits[1] = {e4m3_bits(2.0f)};
        const float gain[1] = {1.0f};
        SplitProjectionWeight w{E4m3Matrix{bits, 1, 1}, gain};
        const float x[2] = {3.0f, 0.5f};
        const float skip[2] = {4.0f, 0.0f};
        float out[2] = {0.0f, 0.0f};
        CHECK(split_ffwd_proj(x, skip, 1, 2, w, out) == Status::kOk);
        CHECK_CLOSE(out[0], 10.0f, 0.0f);
        CHECK_CLOSE(out[1], 1.0f, 0.0f);
        float out2[2] = {0.0f, 0.0f};
        CHECK(split_proj(x, skip, 1, 2, w, out2) == Status::kOk);
        CHECK_CLOSE(out2[0], 10.0f, 0.0f);
        CHECK_CLOSE(out2[1], 1.0f, 0.0f);
        CHECK(split_ffwd_proj(nullptr, skip, 1, 2, w, out) == Status::kNullPointer);
        CHECK(split_ffwd_proj(x, nullptr, 1, 2, w, out) == Status::kNullPointer);
        CHECK(split_ffwd_proj(x, skip, 0, 2, w, out) == Status::kInvalidDimensions);
    }

    // ---- split_proj_pool anchors (2x2 mean of the float16 values, pairwise
    // column sums then row sums, * 0.25, e4m3; extent ceil(extent/2) padded
    // to a multiple of 4 with zeros):
    //   2x2 map [4, 8; 12, 16]: top = 12, bottom = 28, sum = 40 ->
    //   pooled[0] = e4m3(10) = 10 in a 4x4 zero-padded map.
    {
        const std::uint8_t bits[1] = {e4m3_bits(1.0f)};
        const float gain[1] = {0.0f};
        SplitProjectionWeight w{E4m3Matrix{bits, 1, 1}, gain};
        const float x[4] = {4.0f, 8.0f, 12.0f, 16.0f};
        const float skip[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float proj[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float pooled[16];
        fill_bytes(pooled, sizeof pooled, 0xAA);
        int pooled_rows = 0;
        int pooled_cols = 0;
        CHECK(split_proj_pool(x, skip, 2, 2, w, proj, pooled,
                              &pooled_rows, &pooled_cols) == Status::kOk);
        CHECK(pooled_rows == 4);
        CHECK(pooled_cols == 4);
        CHECK_CLOSE(proj[0], 4.0f, 0.0f);
        CHECK_CLOSE(proj[3], 16.0f, 0.0f);
        CHECK_CLOSE(pooled[0], 10.0f, 0.0f);
        for (int i = 1; i < 16; ++i) {
            CHECK_CLOSE(pooled[i], 0.0f, 0.0f);
        }
    }
    //   3x3 map [1..9] (row-major): the ceil pairing leaves the trailing row
    //   and column paired with an implicit zero:
    //     block (0,0): (1+2) + (4+5) = 12 -> 3
    //     block (0,1): (3+0) + (6+0) = 9 -> 2.25
    //     block (1,0): (7+8) + (0+0) = 15 -> 3.75
    //     block (1,1): (9+0) + (0+0) = 9 -> 2.25
    //   in a 4x4 zero-padded map: pooled rows have stride pad_cols = 4, so
    //   the 2x2 pooled block sits at flattened indices 0, 1, 4, 5.
    {
        const std::uint8_t bits[1] = {e4m3_bits(1.0f)};
        const float gain[1] = {0.0f};
        SplitProjectionWeight w{E4m3Matrix{bits, 1, 1}, gain};
        float x[9];
        float skip[9];
        for (int i = 0; i < 9; ++i) {
            x[i] = static_cast<float>(i + 1);
            skip[i] = 0.0f;
        }
        float proj[9] = {0};
        float pooled[16];
        fill_bytes(pooled, sizeof pooled, 0xAA);
        int pooled_rows = 0;
        int pooled_cols = 0;
        CHECK(split_proj_pool(x, skip, 3, 3, w, proj, pooled,
                              &pooled_rows, &pooled_cols) == Status::kOk);
        CHECK(pooled_rows == 4);
        CHECK(pooled_cols == 4);
        CHECK_CLOSE(pooled[0], 3.0f, 0.0f);
        CHECK_CLOSE(pooled[1], 2.25f, 0.0f);
        CHECK_CLOSE(pooled[4], 3.75f, 0.0f);
        CHECK_CLOSE(pooled[5], 2.25f, 0.0f);
        for (int i = 2; i < 16; ++i) {
            if (i != 4 && i != 5) {
                CHECK_CLOSE(pooled[i], 0.0f, 0.0f);
            }
        }
        CHECK_CLOSE(proj[8], 9.0f, 0.0f);
    }
    //   The column pair sum rounds once to float16: f16(1 + 2^-11) is the
    //   midpoint -> 1.0, then 0.25 -> pooled 0.25.
    {
        const std::uint8_t bits[1] = {e4m3_bits(1.0f)};
        const float gain[1] = {0.0f};
        SplitProjectionWeight w{E4m3Matrix{bits, 1, 1}, gain};
        const float x[4] = {1.0f, 0.00048828125f, 0.0f, 0.0f};
        const float skip[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float proj[4] = {0};
        float pooled[16] = {0};
        int pooled_rows = 0;
        int pooled_cols = 0;
        CHECK(split_proj_pool(x, skip, 2, 2, w, proj, pooled,
                              &pooled_rows, &pooled_cols) == Status::kOk);
        CHECK_CLOSE(pooled[0], 0.25f, 0.0f);
    }
    //   Multi-channel pooled copy (c_out = 2) with sentinel guards: both
    //   channels of every pooled token are written and the zero padding
    //   covers the whole c_out-wide map (a miscounted fill leaves stale
    //   padding for c_out > 1). W rows read x * 1 and x * 2; tokens
    //   [4, 8, 12, 16] give channel pairs (4, 8), (8, 16), (12, 24),
    //   (16, 32) -> pooled token (10, 20), the padded map 4x4.
    {
        const std::uint8_t bits[2] = {e4m3_bits(1.0f), e4m3_bits(2.0f)};
        const float gain[2] = {0.0f, 0.0f};
        SplitProjectionWeight w{E4m3Matrix{bits, 2, 1}, gain};
        const float x[4] = {4.0f, 8.0f, 12.0f, 16.0f};
        const float skip[8] = {0.0f};
        static float guarded_proj[4 * 2 + 2];
        static float guarded_pooled[4 * 4 * 2 + 2];
        fill_bytes(guarded_proj, sizeof guarded_proj, 0x55);
        fill_bytes(guarded_pooled, sizeof guarded_pooled, 0x55);
        int pooled_rows = 0;
        int pooled_cols = 0;
        CHECK(split_proj_pool(x, skip, 2, 2, w, guarded_proj + 1,
                              guarded_pooled + 1, &pooled_rows,
                              &pooled_cols) == Status::kOk);
        CHECK(pooled_rows == 4);
        CHECK(pooled_cols == 4);
        CHECK_CLOSE(guarded_proj[1], 4.0f, 0.0f);
        CHECK_CLOSE(guarded_proj[2], 8.0f, 0.0f);
        CHECK_CLOSE(guarded_proj[7], 16.0f, 0.0f);
        CHECK_CLOSE(guarded_proj[8], 32.0f, 0.0f);
        CHECK_CLOSE(guarded_pooled[1], 10.0f, 0.0f);
        CHECK_CLOSE(guarded_pooled[2], 20.0f, 0.0f);
        for (int i = 3; i < 4 * 4 * 2 + 1; ++i) {
            CHECK_CLOSE(guarded_pooled[i], 0.0f, 0.0f);
        }
        std::uint32_t lead = 0;
        std::uint32_t tail = 0;
        std::memcpy(&lead, &guarded_proj[0], sizeof lead);
        std::memcpy(&tail, &guarded_proj[4 * 2 + 1], sizeof tail);
        CHECK(lead == 0x55555555u);
        CHECK(tail == 0x55555555u);
        std::memcpy(&lead, &guarded_pooled[0], sizeof lead);
        std::memcpy(&tail, &guarded_pooled[4 * 4 * 2 + 1], sizeof tail);
        CHECK(lead == 0x55555555u);
        CHECK(tail == 0x55555555u);
    }

    // ---- split_final_head anchors (512 -> 1024 in the network; the generic
    // form here is 1 -> 2): W = [[2], [0.5]], pooled [2] -> [4, 1].
    {
        const std::uint8_t bits[2] = {e4m3_bits(2.0f), e4m3_bits(0.5f)};
        SplitProjectionWeight w{E4m3Matrix{bits, 2, 1}, nullptr};
        const float pooled[1] = {2.0f};
        float out[2] = {0.0f, 0.0f};
        CHECK(split_final_head(pooled, 1, 1, w, out) == Status::kOk);
        CHECK_CLOSE(out[0], 4.0f, 0.0f);
        CHECK_CLOSE(out[1], 1.0f, 0.0f);
        const float gain[1] = {1.0f};
        SplitProjectionWeight gained{E4m3Matrix{bits, 2, 1}, gain};
        CHECK(split_final_head(pooled, 1, 1, gained, out) == Status::kNullPointer);
        CHECK(split_final_head(nullptr, 1, 1, w, out) == Status::kNullPointer);
        CHECK(split_final_head(pooled, 0, 1, w, out) == Status::kInvalidDimensions);
    }

    // ---- split_ffwd anchors (identity matrices):
    //   token 0 with map channel 0 = 1: reduced 1 -> act(1) = 1.2861328125 ->
    //   e4m3 1.25 -> out channel 0 = 1.25.
    //   token 1 with map channel 511 = -2: reduced -2 -> act(-2) =
    //   -0.447265625 -> e4m3 -0.4375 -> out channel 511 = -0.4375.
    set_ffwd_identity();
    {
        SplitFfwdWeight w = ffwd_weight();
        std::memset(g_map, 0, sizeof g_map);
        g_map[0] = 1.0f;
        g_map[kSplitChannels + kSplitChannels - 1] = -2.0f;
        std::memset(g_out, 0x33, sizeof g_out);
        CHECK(split_ffwd(g_map, 1, 2, w, g_out) == Status::kOk);
        CHECK_CLOSE(g_out[0], 1.25f, 0.0f);
        CHECK_CLOSE(g_out[kSplitChannels + kSplitChannels - 1], -0.4375f, 0.0f);
        for (int c = 1; c < kSplitChannels; ++c) {
            CHECK_CLOSE(g_out[c], 0.0f, 0.0f);
        }
        for (int c = 0; c < kSplitChannels - 1; ++c) {
            CHECK_CLOSE(g_out[kSplitChannels + c], 0.0f, 0.0f);
        }
        CHECK(split_ffwd(nullptr, 1, 2, w, g_out) == Status::kNullPointer);
        CHECK(split_ffwd(g_map, 1, 2, w, nullptr) == Status::kNullPointer);
        CHECK(split_ffwd(g_map, 0, 2, w, g_out) == Status::kInvalidDimensions);
        SplitFfwdWeight broken = w;
        broken.expand_in = E4m3Matrix{g_expand_in, 1, 1};
        CHECK(split_ffwd(g_map, 1, 2, broken, g_out) == Status::kInvalidDimensions);
    }

    // ---- split_qkv_attn anchors. The sparse weights give head 0 the query,
    // key and value vectors (m, 0..) for a token whose map channel 0 is m;
    // every other head is all-zero. With m = 1 the normalised query and key
    // are (1, 0..), the value e4m3(1) = 1.
    //   8x8 map, shift (0,0): one window; only token (0,0) has m = 1.
    //   Query (0,0): logits 1 at key 0 and 0 elsewhere, so the leaves are
    //   split_exp(1) = 0.06640625 and 63 * split_exp(0) = 0.025390625.
    //   The pinned denominator is 0.06640625 + 63 * 0.025390625 =
    //   853 * 2^-9 = 1.666015625 (exact at every step), the probability
    //   p0 = e4m3(f16(34/853)) = e4m3(1306 * 2^-15) = 0.0390625 and the
    //   context p0 * 1 = 0.0390625.
    //   Every other query has a zero query vector: all 64 leaves are
    //   split_exp(0), the denominator is 64 * 13 * 2^-9 = 832 * 2^-9 =
    //   1.625, the probability is e4m3(f16(1/64)) = 0.015625 and the
    //   context is 0.015625 * value(0,0) = 0.015625.
    //   Heads 1..15 are all-zero -> their channels stay zero.
    set_attn_sparse();
    {
        SplitQkvWeight w = attn_weight();
        std::memset(g_map, 0, sizeof g_map);
        g_map[0] = 1.0f;
        std::memset(g_out, 0x44, sizeof g_out);
        CHECK(split_qkv_attn(g_map, 8, 8, 0, 0, w, g_out) == Status::kOk);
        CHECK_CLOSE(g_out[0], 0.0390625f, 0.0f);
        for (int t = 1; t < 64; ++t) {
            CHECK_CLOSE(g_out[t * kSplitChannels], 0.015625f, 0.0f);
        }
        for (int t = 0; t < 64; ++t) {
            for (int c = 1; c < kSplitChannels; ++c) {
                CHECK_CLOSE(g_out[t * kSplitChannels + c], 0.0f, 0.0f);
            }
        }
    }
    //   Position-bias orientation: bias[head 0][query 2][key 0] = 2. Query 2
    //   (a zero query) then has leaves split_exp(2) = 0.1875 at key 0 and
    //   0.025390625 elsewhere: denominator 915 * 2^-9 = 1.787109375,
    //   p0 = e4m3(f16(32/305)) = e4m3(1719 * 2^-14) = 0.1015625, context
    //   0.1015625. Query 0's row of the bias is untouched (0.0390625 as
    //   above) and the remaining queries read 0.015625. A transposed
    //   reading would boost query 0 instead (0.03515625).
    {
        SplitQkvWeight w = attn_weight();
        g_bias[(0 * kSplitTokensPerWindow + 2) * kSplitTokensPerWindow + 0] = 2.0f;
        std::memset(g_map, 0, sizeof g_map);
        g_map[0] = 1.0f;
        CHECK(split_qkv_attn(g_map, 8, 8, 0, 0, w, g_out) == Status::kOk);
        CHECK_CLOSE(g_out[2 * kSplitChannels], 0.1015625f, 0.0f);
        CHECK_CLOSE(g_out[0], 0.0390625f, 0.0f);
        for (int t = 1; t < 64; ++t) {
            if (t != 2) {
                CHECK_CLOSE(g_out[t * kSplitChannels], 0.015625f, 0.0f);
            }
        }
        g_bias[(0 * kSplitTokensPerWindow + 2) * kSplitTokensPerWindow + 0] = 0.0f;
    }
    //   Window-token order: query 16 is map token (0, 4) in the documented
    //   tile-major order (top-left tile 0..15, top-right 16..31) but map
    //   token (2, 0) under a naive row-major order. bias[query 16][key 0] =
    //   2 must therefore boost map token (0, 4) to 0.1015625 and leave map
    //   token (2, 0) at 0.015625.
    {
        SplitQkvWeight w = attn_weight();
        g_bias[(0 * kSplitTokensPerWindow + 16) * kSplitTokensPerWindow + 0] = 2.0f;
        std::memset(g_map, 0, sizeof g_map);
        g_map[0] = 1.0f;
        CHECK(split_qkv_attn(g_map, 8, 8, 0, 0, w, g_out) == Status::kOk);
        CHECK_CLOSE(g_out[4 * kSplitChannels], 0.1015625f, 0.0f);
        CHECK_CLOSE(g_out[(2 * 8) * kSplitChannels], 0.015625f, 0.0f);
        g_bias[(0 * kSplitTokensPerWindow + 16) * kSplitTokensPerWindow + 0] = 0.0f;
    }
    //   Fully shifted grid (both shifts -1): 8x8 map -> 2x2 windows; the
    //   padded top-left window covers map rows 0..3 and columns 0..3 plus
    //   zero padding, so the live token (0, 0) sits at window token 48
    //   (bottom-right tile) and keeps 0.0390625; its window's other live
    //   queries read 0.015625; the three windows that never see the live
    //   value stay zero.
    {
        SplitQkvWeight w = attn_weight();
        std::memset(g_map, 0, sizeof g_map);
        g_map[0] = 1.0f;
        CHECK(split_qkv_attn(g_map, 8, 8, -1, -1, w, g_out) == Status::kOk);
        CHECK_CLOSE(g_out[0], 0.0390625f, 0.0f);
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                const int t = r * 8 + c;
                if (t != 0) {
                    CHECK_CLOSE(g_out[t * kSplitChannels], 0.015625f, 0.0f);
                }
            }
        }
        for (int r = 0; r < 8; ++r) {
            for (int c = 0; c < 8; ++c) {
                if (r >= 4 || c >= 4) {
                    CHECK_CLOSE(g_out[(r * 8 + c) * kSplitChannels], 0.0f, 0.0f);
                }
            }
        }
    }
    //   Head gain: a 1x1 map (one live token, 63 padded zero tokens) with
    //   gain 2 gives the query (2, 0..): logits 2 at the live key, the
    //   denominator 915 * 2^-9 and the context 0.1015625; with gain 1 the
    //   answer is the 8x8 query-0 value 0.0390625.
    {
        SplitQkvWeight w = attn_weight();
        std::memset(g_map, 0, sizeof g_map);
        g_map[0] = 1.0f;
        g_gain[0] = 2.0f;
        CHECK(split_qkv_attn(g_map, 1, 1, 0, 0, w, g_out) == Status::kOk);
        CHECK_CLOSE(g_out[0], 0.1015625f, 0.0f);
        g_gain[0] = 1.0f;
        CHECK(split_qkv_attn(g_map, 1, 1, 0, 0, w, g_out) == Status::kOk);
        CHECK_CLOSE(g_out[0], 0.0390625f, 0.0f);
    }
    //   Shifted grid (shift_x = -1 moves the grid 4 pixels along the
    //   columns): 8x8 map -> two windows covering map columns 0..3 (plus
    //   padded zero tokens) and 4..7 (plus padding). Token (0,0) keeps its
    //   0.0390625; the live window's other queries read 0.015625 * value
    //   (0,0) = 0.015625; the window over columns 4..7 never sees the live
    //   value and stays zero.
    {
        SplitQkvWeight w = attn_weight();
        std::memset(g_map, 0, sizeof g_map);
        g_map[0] = 1.0f;
        CHECK(split_qkv_attn(g_map, 8, 8, -1, 0, w, g_out) == Status::kOk);
        CHECK_CLOSE(g_out[0], 0.0390625f, 0.0f);
        for (int r = 0; r < 8; ++r) {
            for (int c = 0; c < 4; ++c) {
                const int t = r * 8 + c;
                if (t != 0) {
                    CHECK_CLOSE(g_out[t * kSplitChannels], 0.015625f, 0.0f);
                }
            }
            for (int c = 4; c < 8; ++c) {
                CHECK_CLOSE(g_out[(r * 8 + c) * kSplitChannels], 0.0f, 0.0f);
            }
        }
    }

    // ---- rejection classes for the attention and the feed-forward.
    {
        SplitQkvWeight w = attn_weight();
        CHECK(split_qkv_attn(nullptr, 8, 8, 0, 0, w, g_out) == Status::kNullPointer);
        CHECK(split_qkv_attn(g_map, 8, 8, 0, 0, w, nullptr) == Status::kNullPointer);
        CHECK(split_qkv_attn(g_map, 8, 8, -2, 0, w, g_out) == Status::kInvalidControls);
        CHECK(split_qkv_attn(g_map, 8, 8, 0, 1, w, g_out) == Status::kInvalidControls);
        CHECK(split_qkv_attn(g_map, 0, 8, 0, 0, w, g_out) == Status::kInvalidDimensions);
        SplitQkvWeight bad = w;
        bad.qkv = E4m3Matrix{g_qkv, 1, 1};
        CHECK(split_qkv_attn(g_map, 8, 8, 0, 0, bad, g_out) == Status::kInvalidDimensions);
        SplitQkvWeight no_bias = w;
        no_bias.bias = nullptr;
        CHECK(split_qkv_attn(g_map, 8, 8, 0, 0, no_bias, g_out) == Status::kNullPointer);
        SplitQkvWeight no_gain = w;
        no_gain.gain = nullptr;
        CHECK(split_qkv_attn(g_map, 8, 8, 0, 0, no_gain, g_out) == Status::kNullPointer);
    }

    // ---- sentinel guards: the attention writes only its output map.
    {
        SplitQkvWeight w = attn_weight();
        std::memset(g_map, 0, sizeof g_map);
        g_map[0] = 1.0f;
        static float guarded_out[64 * kSplitChannels + 2];
        fill_bytes(guarded_out, sizeof guarded_out, 0x55);
        CHECK(split_qkv_attn(g_map, 8, 8, 0, 0, w, guarded_out + 1) == Status::kOk);
        std::uint32_t lead = 0;
        std::uint32_t tail = 0;
        std::memcpy(&lead, &guarded_out[0], sizeof lead);
        std::memcpy(&tail, &guarded_out[64 * kSplitChannels + 1], sizeof tail);
        CHECK(lead == 0x55555555u);
        CHECK(tail == 0x55555555u);
    }
    // ---- sentinel guards: the feed-forward, the stored projection and the
    // final head write only their output maps.
    {
        set_ffwd_identity();
        SplitFfwdWeight fw = ffwd_weight();
        std::memset(g_map, 0, sizeof g_map);
        g_map[0] = 1.0f;
        static float guarded_out[2 * kSplitChannels + 2];
        fill_bytes(guarded_out, sizeof guarded_out, 0x55);
        CHECK(split_ffwd(g_map, 1, 2, fw, guarded_out + 1) == Status::kOk);
        std::uint32_t lead = 0;
        std::uint32_t tail = 0;
        std::memcpy(&lead, &guarded_out[0], sizeof lead);
        std::memcpy(&tail, &guarded_out[2 * kSplitChannels + 1], sizeof tail);
        CHECK(lead == 0x55555555u);
        CHECK(tail == 0x55555555u);
    }
    {
        const std::uint8_t bits[1] = {e4m3_bits(1.0f)};
        const float gain[1] = {1.0f};
        SplitProjectionWeight w{E4m3Matrix{bits, 1, 1}, gain};
        const float x[2] = {1.0f, -1.0f};
        const float skip[2] = {1.0f, 1.0f};
        static float guarded_out[2 + 2];
        fill_bytes(guarded_out, sizeof guarded_out, 0x55);
        CHECK(split_ffwd_proj(x, skip, 1, 2, w, guarded_out + 1) == Status::kOk);
        std::uint32_t lead = 0;
        std::uint32_t tail = 0;
        std::memcpy(&lead, &guarded_out[0], sizeof lead);
        std::memcpy(&tail, &guarded_out[2 + 1], sizeof tail);
        CHECK(lead == 0x55555555u);
        CHECK(tail == 0x55555555u);
    }
    {
        const std::uint8_t bits[2] = {e4m3_bits(1.0f), e4m3_bits(1.0f)};
        SplitProjectionWeight w{E4m3Matrix{bits, 2, 1}, nullptr};
        const float pooled[1] = {1.0f};
        static float guarded_out[2 + 2];
        fill_bytes(guarded_out, sizeof guarded_out, 0x55);
        CHECK(split_final_head(pooled, 1, 1, w, guarded_out + 1) == Status::kOk);
        std::uint32_t lead = 0;
        std::uint32_t tail = 0;
        std::memcpy(&lead, &guarded_out[0], sizeof lead);
        std::memcpy(&tail, &guarded_out[2 + 1], sizeof tail);
        CHECK(lead == 0x55555555u);
        CHECK(tail == 0x55555555u);
    }

    // ---- determinism: identical inputs give bitwise identical outputs.
    {
        SplitQkvWeight w = attn_weight();
        std::memset(g_map, 0, sizeof g_map);
        g_map[0] = 1.0f;
        g_map[kSplitChannels + 3] = -0.25f;
        static float again[64 * kSplitChannels];
        CHECK(split_qkv_attn(g_map, 8, 8, -1, -1, w, g_out) == Status::kOk);
        CHECK(split_qkv_attn(g_map, 8, 8, -1, -1, w, again) == Status::kOk);
        CHECK(std::memcmp(g_out, again, sizeof g_out) == 0);
        set_ffwd_identity();
        SplitFfwdWeight fw = ffwd_weight();
        static float ffwd_again[64 * kSplitChannels];
        CHECK(split_ffwd(g_map, 8, 8, fw, g_out) == Status::kOk);
        CHECK(split_ffwd(g_map, 8, 8, fw, ffwd_again) == Status::kOk);
        CHECK(std::memcmp(g_out, ffwd_again, sizeof g_out) == 0);
    }

    std::printf("native-split-swin-test: %lld checks passed\n", g_checks);
    return 0;
}
