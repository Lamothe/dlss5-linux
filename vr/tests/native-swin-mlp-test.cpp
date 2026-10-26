// Independent tests for the native Swin MLP assembly (vr/native-swin-mlp.hpp):
// the grouped middle projection with its store narrowing, the contract projection
// and the f16 residual scale, and the composed swin_mlp. CPU-only, no GPU, headset,
// proprietary payload, Python or PyTorch runtime required.
//
// Independence: every known answer is a hand-derived dyadic rational. The e4m3
// operand products (hidden/mid, in/contract and x/expand) are exact, and the
// expected sums are exact or hand-rounded through the documented store/tail
// roundings; the residual-scale product is hand-rounded through the pinned tail
// roundings — deliberately inexact in the two-rounding discriminator below. Controls
// cover: the store double-rounding discriminator (367.875 stores as 384 where a
// direct e4m3 rounding gives 352), the residual-tail two-rounding discriminator
// (the eager mul-then-add gives 12582914 where a fused multiply-add would give
// 12582913), head-grouping, permutation and changed-input sensitivity controls on an
// exact corpus with guarded distinct operands, a bit-exact equivalence control of
// the composed assembly against the staged chain over a pseudorandom corpus,
// sentinel-checked untouched outputs on every validation rejection class (and
// untouched scratch in the assembly), determinism and zero-allocation accounting.
// See vr/NATIVE-SWIN-MLP.md.

#include "native-swin-mlp.hpp"

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

static constexpr float kSentinel = 12345.0f;

static void fill_sentinel(float* buffer, std::size_t count) {
    for (std::size_t p = 0; p < count; ++p) {
        buffer[p] = kSentinel;
    }
}
static bool all_sentinel(const float* buffer, std::size_t count) {
    for (std::size_t p = 0; p < count; ++p) {
        if (memcmp(&buffer[p], &kSentinel, sizeof(float)) != 0) {
            return false;
        }
    }
    return true;
}

using dlss5_nr::MatrixView;
using dlss5_nr::MutableMatrixView;
using dlss5_nr::Status;

static MatrixView make_view(const float* data, int rows, int cols,
                            std::ptrdiff_t row_stride) {
    MatrixView m;
    m.data = data;
    m.rows = rows;
    m.cols = cols;
    m.row_stride = row_stride;
    return m;
}

static MutableMatrixView make_out(float* data, int rows, int cols,
                                  std::ptrdiff_t row_stride) {
    MutableMatrixView m;
    m.data = data;
    m.rows = rows;
    m.cols = cols;
    m.row_stride = row_stride;
    return m;
}

// ---- the store double rounding ---------------------------------------------------------------------
// 367.875: the f16 tie lands on 368 (even significand) and the e4m3 tie on the even
// code 384 -> 384. A direct e4m3 rounding gives 352. The saturation chain:
// 65519 -> f16 65504 -> 448; 65530 -> f16 inf -> 448.
static void check_store_pins(void) {
    using dlss5_nr::e4m3;
    using dlss5_nr::swin_store;
    CHECK(swin_store(1.0f) == 1.0f);
    CHECK(swin_store(367.875f) == 384.0f);
    CHECK(e4m3(367.875f) == 352.0f);
    CHECK(384.0f != 352.0f);
    CHECK(swin_store(-367.875f) == -384.0f);
    CHECK(swin_store(65519.0f) == 448.0f);
    CHECK(swin_store(65530.0f) == 448.0f);
    CHECK(swin_store(-65530.0f) == -448.0f);
    CHECK(swin_store(0.0f) == 0.0f);
    CHECK(std::signbit(swin_store(-0.0f)));
    float canonical = 0.0f;
    const std::uint32_t bits = 0x7fc00000u;
    memcpy(&canonical, &bits, sizeof canonical);
    float stored_nan = swin_store((float)NAN);
    CHECK(memcmp(&stored_nan, &canonical, sizeof stored_nan) == 0);
}

// ---- the grouped middle projection ------------------------------------------------------------------
static void check_mid_known_answers(void) {
    using dlss5_nr::swin_mid;
    using dlss5_nr::kHeadDim;
    // heads = 1, group width 2: hand sums 2, 0, 0.75 and the zero rows.
    {
        const float hidden[2] = {1.0f, 0.5f};
        float mid[32 * 2];
        fill_sentinel(mid, 32 * 2);
        for (int d = 0; d < 32; ++d) {
            mid[d * 2] = 0.0f;
            mid[d * 2 + 1] = 0.0f;
        }
        mid[0] = 1.0f;
        mid[1] = 2.0f;
        mid[2] = 2.0f;
        mid[3] = -4.0f;
        mid[4] = 0.5f;
        mid[5] = 0.5f;
        float out[32];
        fill_sentinel(out, 32);
        MutableMatrixView o = make_out(out, 1, 32, 32);
        CHECK(swin_mid(make_view(hidden, 1, 2, 2), make_view(mid, 32, 2, 2), 1, &o,
                32) == Status::kOk);
        CHECK(out[0] == 2.0f);
        CHECK(out[1] == 0.0f);
        CHECK(out[2] == 0.75f);
        CHECK(out[3] == 0.0f);
        CHECK(out[31] == 0.0f);
    }
    // heads = 2, group width 2: the grouping discriminator. head 0 reads hidden
    // (1, 2), head 1 reads hidden (4, 8); a mis-grouped implementation that fed
    // both heads all four terms would give 15 at out[0] instead of 3.
    {
        const float hidden[4] = {1.0f, 2.0f, 4.0f, 8.0f};
        float mid[64 * 2];
        for (int p = 0; p < 64 * 2; ++p) {
            mid[p] = 0.0f;
        }
        mid[0] = 1.0f;
        mid[1] = 1.0f;
        mid[32 * 2] = 1.0f;
        mid[32 * 2 + 1] = 1.0f;
        mid[33 * 2] = 0.5f;
        mid[33 * 2 + 1] = -1.0f;
        float out[64];
        MutableMatrixView o = make_out(out, 1, 64, 64);
        CHECK(swin_mid(make_view(hidden, 1, 4, 4), make_view(mid, 64, 2, 2), 2, &o,
                64) == Status::kOk);
        CHECK(out[0] == 3.0f);
        CHECK(out[32] == 12.0f);
        CHECK(out[33] == -6.0f);
        CHECK(out[1] == 0.0f);
        CHECK(out[31] == 0.0f);
        CHECK(out[63] == 0.0f);
    }
    // The store double rounding through the projection: 1.875 * 192 + 1.125 * 7 =
    // 360 + 7.875 = 367.875 stores as 384 (a direct e4m3 rounding would give 352).
    {
        const float hidden[2] = {1.875f, 1.125f};
        float mid[32 * 2];
        for (int p = 0; p < 32 * 2; ++p) {
            mid[p] = 0.0f;
        }
        mid[0] = 192.0f;
        mid[1] = 7.0f;
        float out[32];
        MutableMatrixView o = make_out(out, 1, 32, 32);
        CHECK(swin_mid(make_view(hidden, 1, 2, 2), make_view(mid, 32, 2, 2), 1, &o,
                32) == Status::kOk);
        CHECK(out[0] == 384.0f);
    }
    // Padded strides: input stride padding is never read, output stride padding is
    // never written. Row 1's hidden (2, 1) gives store(2*1 + 1*2) = 4 at out[33].
    {
        const float hidden[2 * 3] = {1.0f, 0.5f, kSentinel, 2.0f, 1.0f, kSentinel};
        float mid[32 * 3];
        fill_sentinel(mid, 32 * 3);
        for (int d = 0; d < 32; ++d) {
            mid[d * 3] = (d == 0) ? 1.0f : 0.0f;
            mid[d * 3 + 1] = (d == 0) ? 2.0f : 0.0f;
        }
        float out[2 * 33];
        fill_sentinel(out, 2 * 33);
        MutableMatrixView o = make_out(out, 2, 32, 33);
        CHECK(swin_mid(make_view(hidden, 2, 2, 3), make_view(mid, 32, 2, 3), 1, &o,
                2 * 33) == Status::kOk);
        CHECK(out[0] == 2.0f);            // row 0: 1*1 + 0.5*2
        CHECK(out[1] == 0.0f && out[31] == 0.0f);
        CHECK(out[32] == kSentinel);      // row 0 stride padding
        CHECK(out[33] == 4.0f);           // row 1: 2*1 + 1*2
        CHECK(out[34] == 0.0f && out[64] == 0.0f);
        CHECK(out[65] == kSentinel);      // row 1 stride padding
        CHECK(memcmp(&hidden[2], &kSentinel, sizeof(float)) == 0);
        CHECK(memcmp(&hidden[5], &kSentinel, sizeof(float)) == 0);
    }
    (void)kHeadDim;
}

// ---- the contract projection and residual scale -------------------------------------------------------
static void check_tail_known_answers(void) {
    using dlss5_nr::swin_mlp_tail;
    // K = 1: 0.5 + 2 * 3 = 6.5.
    {
        const float in[1] = {2.0f};
        const float contract[1] = {3.0f};
        const float residual[1] = {1.0f};
        const float scale[1] = {0.5f};
        float out[1];
        MutableMatrixView o = make_out(out, 1, 1, 1);
        CHECK(swin_mlp_tail(make_view(in, 1, 1, 1), make_view(contract, 1, 1, 1),
                make_view(residual, 1, 1, 1), make_view(scale, 1, 1, 1), &o, 1)
            == Status::kOk);
        CHECK(out[0] == 6.5f);
    }
    // A zero product still carries the residual scale: 3 * 2 + (1*2 + 0.5*(-4)) = 6.
    {
        const float in[2] = {1.0f, 0.5f};
        const float contract[2] = {2.0f, -4.0f};
        const float residual[1] = {3.0f};
        const float scale[1] = {2.0f};
        float out[1];
        MutableMatrixView o = make_out(out, 1, 1, 1);
        CHECK(swin_mlp_tail(make_view(in, 1, 2, 2), make_view(contract, 1, 2, 2),
                make_view(residual, 1, 1, 1), make_view(scale, 1, 1, 1), &o, 1)
            == Status::kOk);
        CHECK(out[0] == 6.0f);
    }
    // rows = 2, k = 2, n = 2 with padded strides:
    //   products (1, 10) / (0.5, -10); residual (1, -2) / (0.5, 4); scale (2, 0.5)
    //   -> (1*2 + 1, -2*0.5 + 10, 0.5*2 + 0.5, 4*0.5 - 10) = (3, 9, 1.5, -8).
    {
        const float in[2 * 3] = {1.0f, 2.0f, kSentinel, 3.0f, -4.0f, kSentinel};
        const float contract[2 * 3] = {0.5f, 0.25f, kSentinel, 2.0f, 4.0f, kSentinel};
        const float residual[2 * 3] = {1.0f, -2.0f, kSentinel, 0.5f, 4.0f, kSentinel};
        const float scale[2] = {2.0f, 0.5f};
        float out[2 * 3];
        fill_sentinel(out, 2 * 3);
        MutableMatrixView o = make_out(out, 2, 2, 3);
        CHECK(swin_mlp_tail(make_view(in, 2, 2, 3), make_view(contract, 2, 2, 3),
                make_view(residual, 2, 2, 3), make_view(scale, 1, 2, 2), &o, 2 * 3)
            == Status::kOk);
        CHECK(out[0] == 3.0f && out[1] == 9.0f);
        CHECK(out[3] == 1.5f && out[4] == -8.0f);
        CHECK(out[2] == kSentinel && out[5] == kSentinel);
    }
    // The two-rounding discriminator: 8388609 * 1.5 = 12582913.5 ties to even to
    // 12582914 (first rounding), and 12582914 - 0.5 = 12582913.5 ties to even to
    // 12582914 again (second rounding). A fused multiply-add would round the exact
    // 12582913.0 once, giving 12582913, so the eager two-rounding value is pinned
    // and the fused alternative is asserted distinct.
    {
        const float in[1] = {-0.5f};
        const float contract[1] = {1.0f};
        const float residual[1] = {8388609.0f};
        const float scale[1] = {1.5f};
        float out[1];
        MutableMatrixView o = make_out(out, 1, 1, 1);
        CHECK(swin_mlp_tail(make_view(in, 1, 1, 1), make_view(contract, 1, 1, 1),
                make_view(residual, 1, 1, 1), make_view(scale, 1, 1, 1), &o, 1)
            == Status::kOk);
        CHECK(out[0] == 12582914.0f);
        CHECK(12582913.0f != 12582914.0f);
    }
}

// ---- the composed assembly --------------------------------------------------------------------------
// Full-chain hand answers through swin_hidden's pinned activations (product 1 ->
// 1.25, product -1 -> -0.5, product 4 -> 7.0, product 0 -> 0).
static void check_mlp_known_answers(void) {
    using dlss5_nr::swin_mlp;
    const float x[2] = {1.0f, 1.0f};
    // expand rows give products 1, -1, 4, 0 -> hidden (1.25, -0.5, 7, 0).
    const float expand[4 * 2] = {1.0f, 0.0f, 0.5f, -1.5f, 2.0f, 2.0f, 0.0f, 0.0f};
    const float residual[2] = {0.5f, 2.0f};
    const float scale[2] = {2.0f, 0.25f};
    float scratch_buf[4];
    MutableMatrixView scratch = make_out(scratch_buf, 1, 4, 4);
    // Single-head (mid null): contract rows (1, 0, 0, 0) and (0, 2, 0, 0) give
    // products 1.25 and -1 -> out = (0.5*2 + 1.25, 2*0.25 - 1) = (2.25, -0.5).
    {
        const float contract[2 * 4] = {1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 2.0f, 0.0f, 0.0f};
        float out[2];
        MutableMatrixView o = make_out(out, 1, 2, 2);
        CHECK(swin_mlp(make_view(x, 1, 2, 2), make_view(expand, 4, 2, 2), nullptr, 1,
                make_view(contract, 2, 4, 4), make_view(residual, 1, 2, 2),
                make_view(scale, 1, 2, 2), &o, 2, &scratch, 4) == Status::kOk);
        CHECK(out[0] == 2.25f);
        CHECK(out[1] == -0.5f);
    }
    // Mid path (heads = 2, group width 2): line = (0.25, 0, ..., 0, 7 at index 33)
    // from mid rows (1, 2) and (1, 0); contract rows pick line[0] and 2*line[33]
    // -> products 0.25 and 14 -> out = (0.25, 14).
    {
        float mid[64 * 2];
        for (int p = 0; p < 64 * 2; ++p) {
            mid[p] = 0.0f;
        }
        mid[0] = 1.0f;
        mid[1] = 2.0f;
        mid[33 * 2] = 1.0f;
        MatrixView mid_view = make_view(mid, 64, 2, 2);
        float contract[2 * 64];
        for (int p = 0; p < 2 * 64; ++p) {
            contract[p] = 0.0f;
        }
        contract[0] = 1.0f;
        contract[64 + 33] = 2.0f;
        const float zero_residual[2] = {0.0f, 0.0f};
        const float one_scale[2] = {1.0f, 1.0f};
        float out[2];
        MutableMatrixView o = make_out(out, 1, 2, 2);
        CHECK(swin_mlp(make_view(x, 1, 2, 2), make_view(expand, 4, 2, 2), &mid_view, 2,
                make_view(contract, 2, 64, 64),
                make_view(zero_residual, 1, 2, 2), make_view(one_scale, 1, 2, 2),
                &o, 2, &scratch, 4) == Status::kOk);
        CHECK(out[0] == 0.25f);
        CHECK(out[1] == 14.0f);
    }
}

// ---- permutation and sensitivity controls -----------------------------------------------------------
// Exact dyadic corpus: every product and every partial sum is exactly representable
// in float32 (multiples of 0.25 within +-2 for the mid path, multiples of 0.125
// within +-10 for the tail), so the documented k-order accumulation is
// order-independent and each permutation control is bit-exact. The store is the
// identity on the mid sums (all on the f16 and e4m3 grids), so the sensitivity
// deltas are exact too. Guards below make every control non-vacuous.
static void check_mid_permutation_and_sensitivity(void) {
    using dlss5_nr::swin_mid;
    constexpr int kRows = 3;
    constexpr int kHeads = 2;
    constexpr int kGroup = 8;
    constexpr int kWidth = kHeads * kGroup;             // 16
    constexpr int kOut = kHeads * dlss5_nr::kHeadDim;   // 64
    float hidden[kRows * kWidth];
    float mid[kOut * kGroup];
    float base[kRows * kOut];
    float other[kRows * kOut];
    // hidden rows are cyclic shifts (pairwise distinct below), values {0, +-0.25};
    // every mid row binary-encodes its row index in the sign pattern (all 64 rows
    // pairwise distinct below), values +-1.
    for (int r = 0; r < kRows; ++r) {
        for (int c = 0; c < kWidth; ++c) {
            hidden[r * kWidth + c] = (float)(((r + c) % 3) - 1) * 0.25f;
        }
    }
    for (int r = 0; r < kOut; ++r) {
        for (int c = 0; c < kGroup; ++c) {
            mid[r * kGroup + c] = ((r >> c) & 1) ? 1.0f : -1.0f;
        }
    }
    for (int a = 0; a < kRows; ++a) {
        for (int b = a + 1; b < kRows; ++b) {
            CHECK(memcmp(&hidden[a * kWidth], &hidden[b * kWidth],
                kWidth * sizeof(float)) != 0);
        }
    }
    for (int a = 0; a < kOut; ++a) {
        for (int b = a + 1; b < kOut; ++b) {
            CHECK(memcmp(&mid[a * kGroup], &mid[b * kGroup],
                kGroup * sizeof(float)) != 0);
        }
    }
    MutableMatrixView o_base = make_out(base, kRows, kOut, kOut);
    CHECK(swin_mid(make_view(hidden, kRows, kWidth, kWidth),
            make_view(mid, kOut, kGroup, kGroup), kHeads, &o_base,
            kRows * kOut) == Status::kOk);
    // The output rows differ (the row-permutation control cannot be vacuous).
    for (int a = 0; a < kRows; ++a) {
        for (int b = a + 1; b < kRows; ++b) {
            CHECK(memcmp(&base[a * kOut], &base[b * kOut],
                kOut * sizeof(float)) != 0);
        }
    }
    // Row permutation of hidden permutes output rows identically (bit-exact).
    float hidden_perm[kRows * kWidth];
    for (int r = 0; r < kRows; ++r) {
        const int src = (r + 1) % kRows;
        memcpy(&hidden_perm[r * kWidth], &hidden[src * kWidth],
            kWidth * sizeof(float));
    }
    MutableMatrixView o_other = make_out(other, kRows, kOut, kOut);
    CHECK(swin_mid(make_view(hidden_perm, kRows, kWidth, kWidth),
            make_view(mid, kOut, kGroup, kGroup), kHeads, &o_other,
            kRows * kOut) == Status::kOk);
    for (int r = 0; r < kRows; ++r) {
        const int src = (r + 1) % kRows;
        CHECK(memcmp(&other[r * kOut], &base[src * kOut],
            kOut * sizeof(float)) == 0);
    }
    // Head permutation (hidden groups and mid row blocks swapped together) swaps
    // the output head blocks identically. The two base head blocks differ (the
    // binary-encoded mid blocks are distinct), so this cannot be vacuous.
    CHECK(memcmp(base, &base[dlss5_nr::kHeadDim],
        dlss5_nr::kHeadDim * sizeof(float)) != 0);
    float hidden_head[kRows * kWidth];
    float mid_head[kOut * kGroup];
    for (int r = 0; r < kRows; ++r) {
        for (int g = 0; g < kHeads; ++g) {
            memcpy(&hidden_head[r * kWidth + g * kGroup],
                &hidden[r * kWidth + (kHeads - 1 - g) * kGroup],
                kGroup * sizeof(float));
        }
    }
    for (int g = 0; g < kHeads; ++g) {
        for (int d = 0; d < dlss5_nr::kHeadDim; ++d) {
            memcpy(&mid_head[(g * dlss5_nr::kHeadDim + d) * kGroup],
                &mid[((kHeads - 1 - g) * dlss5_nr::kHeadDim + d) * kGroup],
                kGroup * sizeof(float));
        }
    }
    CHECK(swin_mid(make_view(hidden_head, kRows, kWidth, kWidth),
            make_view(mid_head, kOut, kGroup, kGroup), kHeads, &o_other,
            kRows * kOut) == Status::kOk);
    for (int r = 0; r < kRows; ++r) {
        for (int g = 0; g < kHeads; ++g) {
            for (int d = 0; d < dlss5_nr::kHeadDim; ++d) {
                CHECK(memcmp(&other[r * kOut + g * dlss5_nr::kHeadDim + d],
                    &base[r * kOut
                        + (kHeads - 1 - g) * dlss5_nr::kHeadDim + d],
                    sizeof(float)) == 0);
            }
        }
    }
    // Group-internal K-permutation (the group columns reversed in both operands)
    // leaves the exact sums bit-identical.
    float hidden_k[kRows * kWidth];
    float mid_k[kOut * kGroup];
    for (int r = 0; r < kRows; ++r) {
        for (int g = 0; g < kHeads; ++g) {
            for (int c = 0; c < kGroup; ++c) {
                hidden_k[r * kWidth + g * kGroup + c]
                    = hidden[r * kWidth + g * kGroup + (kGroup - 1 - c)];
            }
        }
    }
    for (int r = 0; r < kOut; ++r) {
        for (int c = 0; c < kGroup; ++c) {
            mid_k[r * kGroup + c] = mid[r * kGroup + (kGroup - 1 - c)];
        }
    }
    CHECK(swin_mid(make_view(hidden_k, kRows, kWidth, kWidth),
            make_view(mid_k, kOut, kGroup, kGroup), kHeads, &o_other,
            kRows * kOut) == Status::kOk);
    CHECK(memcmp(other, base, sizeof base) == 0);
    // Changed-input sensitivity with exact deltas: +1.0 at hidden row 1 column 2
    // (head 0's group) moves out row 1 cell (0, d) by exactly mid_k[(0, d)][2],
    // and +1.0 at column 13 (head 1's group) moves cell (1, d) by exactly
    // mid_k[(1, d)][5] — note the K-reversed mid columns. Each head's block moves
    // only under its own group's perturbation (the other block is untouched), so
    // the head grouping and the per-column pairing are both pinned.
    hidden_k[1 * kWidth + 2] += 1.0f;
    hidden_k[1 * kWidth + kGroup + 5] += 1.0f;
    CHECK(swin_mid(make_view(hidden_k, kRows, kWidth, kWidth),
            make_view(mid_k, kOut, kGroup, kGroup), kHeads, &o_other,
            kRows * kOut) == Status::kOk);
    for (int d = 0; d < dlss5_nr::kHeadDim; ++d) {
        CHECK(other[kOut + d] == base[kOut + d] + mid_k[d * kGroup + 2]);
        CHECK(other[kOut + dlss5_nr::kHeadDim + d]
            == base[kOut + dlss5_nr::kHeadDim + d]
                + mid_k[(dlss5_nr::kHeadDim + d) * kGroup + 5]);
    }
    CHECK(memcmp(other, base, kOut * sizeof(float)) == 0);
    CHECK(memcmp(&other[2 * kOut], &base[2 * kOut], kOut * sizeof(float)) == 0);
    CHECK(memcmp(&other[kOut], &base[kOut], kOut * sizeof(float)) != 0);
    // Determinism: identical inputs, bit-identical output.
    float again[kRows * kOut];
    MutableMatrixView o_again = make_out(again, kRows, kOut, kOut);
    CHECK(swin_mid(make_view(hidden_k, kRows, kWidth, kWidth),
            make_view(mid_k, kOut, kGroup, kGroup), kHeads, &o_again,
            kRows * kOut) == Status::kOk);
    CHECK(memcmp(again, other, sizeof again) == 0);
}

static void check_tail_permutation_and_sensitivity(void) {
    using dlss5_nr::swin_mlp_tail;
    constexpr int kRows = 3;
    constexpr int kK = 5;
    constexpr int kN = 4;
    float in[kRows * kK];
    float contract[kN * kK];
    float residual[kRows * kN];
    float scale[kN];
    float base[kRows * kN];
    float other[kRows * kN];
    for (int r = 0; r < kRows; ++r) {
        for (int c = 0; c < kK; ++c) {
            in[r * kK + c] = (float)((r * kK + c) % 7 - 3) * 0.25f;
        }
    }
    // Contract rows pairwise distinct (the modulus 7 is not a multiple of kK, so
    // the wrap does not repeat rows), and the sensitivity column kK - 2 is
    // pairwise distinct and non-zero.
    for (int n = 0; n < kN; ++n) {
        for (int c = 0; c < kK; ++c) {
            contract[n * kK + c] = (float)((n * kK + c) % 7 - 2) * 0.5f;
        }
    }
    for (int a = 0; a < kN; ++a) {
        for (int b = a + 1; b < kN; ++b) {
            CHECK(memcmp(&contract[a * kK], &contract[b * kK],
                kK * sizeof(float)) != 0);
        }
    }
    for (int n = 0; n < kN; ++n) {
        CHECK(contract[n * kK + kK - 2] != 0.0f);
    }
    for (int a = 0; a < kN; ++a) {
        for (int b = a + 1; b < kN; ++b) {
            CHECK(contract[a * kK + kK - 2] != contract[b * kK + kK - 2]);
        }
    }
    for (int r = 0; r < kRows; ++r) {
        for (int c = 0; c < kN; ++c) {
            residual[r * kN + c] = (float)((r * kN + c) % 5 - 2) * 0.5f;
        }
    }
    // Scale values are f16-exact, pairwise distinct and non-zero (the residual
    // sensitivity delta pins each channel).
    for (int c = 0; c < kN; ++c) {
        scale[c] = (float)(c + 1) * 0.25f;
    }
    for (int a = 0; a < kN; ++a) {
        for (int b = a + 1; b < kN; ++b) {
            CHECK(scale[a] != scale[b]);
        }
    }
    // In rows pairwise distinct (the row-permutation control cannot be vacuous).
    for (int a = 0; a < kRows; ++a) {
        for (int b = a + 1; b < kRows; ++b) {
            CHECK(memcmp(&in[a * kK], &in[b * kK], kK * sizeof(float)) != 0);
        }
    }
    MutableMatrixView o_base = make_out(base, kRows, kN, kN);
    CHECK(swin_mlp_tail(make_view(in, kRows, kK, kK),
            make_view(contract, kN, kK, kK),
            make_view(residual, kRows, kN, kN), make_view(scale, 1, kN, kN),
            &o_base, kRows * kN) == Status::kOk);
    // Hand-derived anchor row 0 (the same operand formulas as the linears slice
    // anchor, whose k-order sums are (1.25, -1.875, -1.5, 0.625)): residual row 0
    // is (-1, -0.5, 0, 0.5) and scale (0.25, 0.5, 0.75, 1), so the scaled residual
    // is (-0.25, -0.25, 0, 0.5) and
    //   out = (1.25 - 0.25, -1.875 - 0.25, -1.5 + 0, 0.625 + 0.5)
    //       = (1, -2.125, -1.5, 1.125).
    CHECK(base[0] == 1.0f && base[1] == -2.125f && base[2] == -1.5f
        && base[3] == 1.125f);
    // The output rows differ (the row-permutation control cannot be vacuous).
    for (int a = 0; a < kRows; ++a) {
        for (int b = a + 1; b < kRows; ++b) {
            CHECK(memcmp(&base[a * kN], &base[b * kN], kN * sizeof(float)) != 0);
        }
    }
    // Row permutation of `in` and `residual` together permutes output rows
    // identically (bit-exact).
    float in_perm[kRows * kK];
    float residual_perm[kRows * kN];
    for (int r = 0; r < kRows; ++r) {
        const int src = (r + 1) % kRows;
        memcpy(&in_perm[r * kK], &in[src * kK], kK * sizeof(float));
        memcpy(&residual_perm[r * kN], &residual[src * kN], kN * sizeof(float));
    }
    MutableMatrixView o_other = make_out(other, kRows, kN, kN);
    CHECK(swin_mlp_tail(make_view(in_perm, kRows, kK, kK),
            make_view(contract, kN, kK, kK),
            make_view(residual_perm, kRows, kN, kN),
            make_view(scale, 1, kN, kN), &o_other, kRows * kN) == Status::kOk);
    for (int r = 0; r < kRows; ++r) {
        const int src = (r + 1) % kRows;
        CHECK(memcmp(&other[r * kN], &base[src * kN], kN * sizeof(float)) == 0);
    }
    // Output-channel permutation (contract rows, residual columns and scale
    // entries permuted together) permutes output columns identically.
    float contract_perm[kN * kK];
    float residual_ch[kRows * kN];
    float scale_perm[kN];
    for (int n = 0; n < kN; ++n) {
        const int src = (n + 1) % kN;
        memcpy(&contract_perm[n * kK], &contract[src * kK], kK * sizeof(float));
        scale_perm[n] = scale[src];
    }
    for (int r = 0; r < kRows; ++r) {
        for (int n = 0; n < kN; ++n) {
            residual_ch[r * kN + n] = residual[r * kN + (n + 1) % kN];
        }
    }
    CHECK(swin_mlp_tail(make_view(in, kRows, kK, kK),
            make_view(contract_perm, kN, kK, kK),
            make_view(residual_ch, kRows, kN, kN),
            make_view(scale_perm, 1, kN, kN), &o_other, kRows * kN)
        == Status::kOk);
    for (int r = 0; r < kRows; ++r) {
        for (int n = 0; n < kN; ++n) {
            const int src = (n + 1) % kN;
            CHECK(memcmp(&other[r * kN + n], &base[r * kN + src],
                sizeof(float)) == 0);
        }
    }
    // K-permutation applied to both `in` and `contract` leaves the exact sums
    // bit-identical.
    float in_k[kRows * kK];
    float contract_k[kN * kK];
    for (int r = 0; r < kRows; ++r) {
        for (int c = 0; c < kK; ++c) {
            in_k[r * kK + c] = in[r * kK + (kK - 1 - c)];
        }
    }
    for (int n = 0; n < kN; ++n) {
        for (int c = 0; c < kK; ++c) {
            contract_k[n * kK + c] = contract[n * kK + (kK - 1 - c)];
        }
    }
    CHECK(swin_mlp_tail(make_view(in_k, kRows, kK, kK),
            make_view(contract_k, kN, kK, kK),
            make_view(residual, kRows, kN, kN), make_view(scale, 1, kN, kN),
            &o_other, kRows * kN) == Status::kOk);
    CHECK(memcmp(other, base, sizeof base) == 0);
    // Changed-input sensitivity with exact deltas: +1.0 at in_k row 1 column 1
    // moves out row 1 column n by exactly contract_k[n][1] == contract[n][kK - 2]
    // (the pairwise distinct, non-zero column guarded above, so the channel
    // pairing is pinned), and +1.0 at residual row 1 column 2 moves out[1][2] by
    // exactly scale[2]. Other rows/columns are untouched.
    in_k[1 * kK + 1] += 1.0f;
    CHECK(swin_mlp_tail(make_view(in_k, kRows, kK, kK),
            make_view(contract_k, kN, kK, kK),
            make_view(residual, kRows, kN, kN), make_view(scale, 1, kN, kN),
            &o_other, kRows * kN) == Status::kOk);
    for (int n = 0; n < kN; ++n) {
        CHECK(other[kN + n] == base[kN + n] + contract_k[n * kK + 1]);
    }
    CHECK(memcmp(other, base, kN * sizeof(float)) == 0);
    CHECK(memcmp(&other[2 * kN], &base[2 * kN], kN * sizeof(float)) == 0);
    CHECK(memcmp(&other[kN], &base[kN], kN * sizeof(float)) != 0);
    residual[1 * kN + 2] += 1.0f;
    CHECK(swin_mlp_tail(make_view(in_k, kRows, kK, kK),
            make_view(contract_k, kN, kK, kK),
            make_view(residual, kRows, kN, kN), make_view(scale, 1, kN, kN),
            &o_other, kRows * kN) == Status::kOk);
    CHECK(other[kN + 2] == base[kN + 2] + scale[2] + contract_k[2 * kK + 1]);
    CHECK(other[kN + 0] == base[kN + 0] + contract_k[0 * kK + 1]);
    CHECK(other[kN + 1] == base[kN + 1] + contract_k[1 * kK + 1]);
    CHECK(other[kN + 3] == base[kN + 3] + contract_k[3 * kK + 1]);
    CHECK(memcmp(other, base, kN * sizeof(float)) == 0);
    CHECK(memcmp(&other[2 * kN], &base[2 * kN], kN * sizeof(float)) == 0);
    // Determinism: identical inputs, bit-identical output.
    float again[kRows * kN];
    MutableMatrixView o_again = make_out(again, kRows, kN, kN);
    CHECK(swin_mlp_tail(make_view(in_k, kRows, kK, kK),
            make_view(contract_k, kN, kK, kK),
            make_view(residual, kRows, kN, kN), make_view(scale, 1, kN, kN),
            &o_again, kRows * kN) == Status::kOk);
    CHECK(memcmp(again, other, sizeof again) == 0);
}

// ---- the composed assembly against the staged chain -------------------------------------------------
// Pseudorandom e4m3-grid corpus (deterministic LCG, so the test is reproducible):
// the composed swin_mlp must be bit-identical to swin_hidden -> swin_mid ->
// swin_mlp_tail over the same buffers, both with and without the mid projection,
// and with padded strides whose padding is never written.
static void check_assembly_equivalence(void) {
    using dlss5_nr::swin_hidden;
    using dlss5_nr::swin_mid;
    using dlss5_nr::swin_mlp;
    using dlss5_nr::swin_mlp_tail;
    constexpr int kRows = 3;
    constexpr int kXCols = 4;
    constexpr int kExpandRows = 8;   // heads 2 * group width 4 on the mid path
    constexpr int kHeads = 2;
    constexpr int kMidRows = kHeads * dlss5_nr::kHeadDim;  // 64
    constexpr int kN = 5;
    std::uint32_t state = 0x2f6e2b1u;
    const auto next = [&state]() {
        state = state * 1664525u + 1013904223u;
        return state;
    };
    float x[kRows * kXCols];
    float expand[kExpandRows * kXCols];
    float mid[kMidRows * 4];
    float contract_mid[kN * kMidRows];
    float contract_plain[kN * kExpandRows];
    float residual[kRows * kN];
    float scale[kN];
    for (float& v : x) {
        v = (float)(next() % 16u) * 0.25f - 2.0f;
    }
    for (float& v : expand) {
        v = (float)(next() % 16u) * 0.25f - 2.0f;
    }
    for (float& v : mid) {
        v = (float)(next() % 16u) * 0.25f - 2.0f;
    }
    for (float& v : contract_mid) {
        v = (float)(next() % 16u) * 0.25f - 2.0f;
    }
    for (float& v : contract_plain) {
        v = (float)(next() % 16u) * 0.25f - 2.0f;
    }
    for (float& v : residual) {
        v = (float)(next() % 256u) * 0.125f - 16.0f;
    }
    for (int c = 0; c < kN; ++c) {
        scale[c] = (float)(next() % 16u) * 0.125f;
    }
    // Padded strides: scratch rows carry two sentinel floats of padding, out rows
    // one.
    float scratch_buf[kRows * (kExpandRows + 2)];
    float out_buf[kRows * (kN + 1)];
    float line_buf[kRows * (kMidRows + 2)];
    float staged_buf[kRows * (kN + 1)];
    fill_sentinel(scratch_buf, kRows * (kExpandRows + 2));
    fill_sentinel(out_buf, kRows * (kN + 1));
    fill_sentinel(line_buf, kRows * (kMidRows + 2));
    fill_sentinel(staged_buf, kRows * (kN + 1));
    MutableMatrixView scratch =
        make_out(scratch_buf, kRows, kExpandRows, kExpandRows + 2);
    MutableMatrixView out = make_out(out_buf, kRows, kN, kN + 1);
    MutableMatrixView line = make_out(line_buf, kRows, kMidRows, kMidRows + 2);
    MutableMatrixView staged = make_out(staged_buf, kRows, kN, kN + 1);
    const MatrixView xv = make_view(x, kRows, kXCols, kXCols);
    const MatrixView expand_v = make_view(expand, kExpandRows, kXCols, kXCols);
    const MatrixView mid_v = make_view(mid, kMidRows, 4, 4);
    const MatrixView contract_mid_v =
        make_view(contract_mid, kN, kMidRows, kMidRows);
    const MatrixView contract_plain_v =
        make_view(contract_plain, kN, kExpandRows, kExpandRows);
    const MatrixView residual_v = make_view(residual, kRows, kN, kN);
    const MatrixView scale_v = make_view(scale, 1, kN, kN);
    // Single-head (mid null) path.
    CHECK(swin_mlp(xv, expand_v, nullptr, 1, contract_plain_v, residual_v, scale_v,
            &out, kRows * (kN + 1), &scratch,
            kRows * (kExpandRows + 2)) == Status::kOk);
    {
        MutableMatrixView hidden = make_out(scratch_buf, kRows, kExpandRows,
            kExpandRows + 2);
        CHECK(swin_hidden(xv, expand_v, &hidden,
                kRows * (kExpandRows + 2)) == Status::kOk);
        CHECK(swin_mlp_tail(make_view(scratch_buf, kRows, kExpandRows,
                kExpandRows + 2), contract_plain_v, residual_v, scale_v, &staged,
                kRows * (kN + 1)) == Status::kOk);
    }
    for (int r = 0; r < kRows; ++r) {
        CHECK(memcmp(&out_buf[r * (kN + 1)], &staged_buf[r * (kN + 1)],
            kN * sizeof(float)) == 0);
    }
    CHECK(out_buf[kN] == kSentinel && out_buf[2 * (kN + 1) - 1] == kSentinel
        && staged_buf[kN] == kSentinel);
    // Mid path: hidden -> grouped mid projection -> tail, composed vs staged.
    CHECK(swin_mlp(xv, expand_v, &mid_v, kHeads, contract_mid_v, residual_v,
            scale_v, &out, kRows * (kN + 1), &scratch,
            kRows * (kExpandRows + 2)) == Status::kOk);
    {
        MutableMatrixView hidden = make_out(scratch_buf, kRows, kExpandRows,
            kExpandRows + 2);
        CHECK(swin_hidden(xv, expand_v, &hidden,
                kRows * (kExpandRows + 2)) == Status::kOk);
        CHECK(swin_mid(make_view(scratch_buf, kRows, kExpandRows, kExpandRows + 2),
                mid_v, kHeads, &line, kRows * (kMidRows + 2)) == Status::kOk);
        CHECK(swin_mlp_tail(make_view(line_buf, kRows, kMidRows, kMidRows + 2),
                contract_mid_v, residual_v, scale_v, &staged,
                kRows * (kN + 1)) == Status::kOk);
    }
    for (int r = 0; r < kRows; ++r) {
        CHECK(memcmp(&out_buf[r * (kN + 1)], &staged_buf[r * (kN + 1)],
            kN * sizeof(float)) == 0);
    }
    // The stride padding of every buffer is untouched by both runs.
    for (int r = 0; r < kRows; ++r) {
        CHECK(out_buf[r * (kN + 1) + kN] == kSentinel);
        CHECK(staged_buf[r * (kN + 1) + kN] == kSentinel);
        CHECK(scratch_buf[r * (kExpandRows + 2) + kExpandRows] == kSentinel);
        CHECK(scratch_buf[r * (kExpandRows + 2) + kExpandRows + 1] == kSentinel);
        CHECK(line_buf[r * (kMidRows + 2) + kMidRows] == kSentinel);
        CHECK(line_buf[r * (kMidRows + 2) + kMidRows + 1] == kSentinel);
    }
}

// ---- negative controls --------------------------------------------------------------------------------
static void check_negative_mid(void) {
    using dlss5_nr::swin_mid;
    constexpr int kRows = 2;
    constexpr int kGroup = 2;
    constexpr int kHeads = 1;
    constexpr int kOut = kHeads * dlss5_nr::kHeadDim;  // 32
    // The buffers carry one extra row/column of padding so the operand grid scans
    // of the over-sized pairing views below stay in bounds (the pairing check runs
    // after the scans, so those views must be scanned safely to reach it).
    float hidden[kRows * (kGroup + 1)];
    float mid[(kOut + 1) * kGroup];
    float out[kRows * kOut];
    for (int p = 0; p < kRows * (kGroup + 1); ++p) {
        hidden[p] = 0.5f;
    }
    for (int p = 0; p < (kOut + 1) * kGroup; ++p) {
        mid[p] = 0.5f;
    }
    fill_sentinel(out, kRows * kOut);
    const std::size_t capacity = kRows * kOut;
    const MatrixView hv = make_view(hidden, kRows, kGroup, kGroup);
    const MatrixView mv = make_view(mid, kOut, kGroup, kGroup);
    MutableMatrixView o = make_out(out, kRows, kOut, kOut);
    // Null pointers (the null-output rejection cannot write by construction).
    CHECK(swin_mid(make_view(nullptr, kRows, kGroup, kGroup), mv, kHeads, &o,
            capacity) == Status::kNullPointer);
    CHECK(swin_mid(hv, make_view(nullptr, kOut, kGroup, kGroup), kHeads, &o,
            capacity) == Status::kNullPointer);
    CHECK(swin_mid(hv, mv, kHeads, nullptr, capacity) == Status::kNullPointer);
    MutableMatrixView no_out = o;
    no_out.data = nullptr;
    CHECK(swin_mid(hv, mv, kHeads, &no_out, capacity) == Status::kNullPointer);
    CHECK(all_sentinel(out, capacity));
    // Dimension classes: the heads bound, the operand bounds and the pairings.
    CHECK(swin_mid(hv, mv, 0, &o, capacity) == Status::kInvalidDimensions);
    CHECK(swin_mid(hv, mv, dlss5_nr::kMaxLinearDim / dlss5_nr::kHeadDim + 1, &o,
            capacity) == Status::kInvalidDimensions);
    CHECK(swin_mid(make_view(hidden, 0, kGroup, kGroup), mv, kHeads, &o, capacity)
        == Status::kInvalidDimensions);
    CHECK(swin_mid(make_view(hidden, kRows, 0, kGroup), mv, kHeads, &o, capacity)
        == Status::kInvalidDimensions);
    CHECK(swin_mid(make_view(hidden, dlss5_nr::kMaxLinearRows + 1, kGroup, kGroup),
            mv, kHeads, &o, capacity) == Status::kInvalidDimensions);
    CHECK(swin_mid(make_view(hidden, kRows, dlss5_nr::kMaxLinearDim + 1,
            kGroup + 1), mv, kHeads, &o, capacity) == Status::kInvalidDimensions);
    CHECK(swin_mid(hv, make_view(mid, kOut + 1, kGroup, kGroup), kHeads, &o,
            capacity) == Status::kInvalidDimensions);
    CHECK(swin_mid(make_view(hidden, kRows, kGroup + 1, kGroup + 1), mv, kHeads,
            &o, capacity) == Status::kInvalidDimensions);
    MutableMatrixView wrong_rows = o;
    wrong_rows.rows = kRows + 1;
    CHECK(swin_mid(hv, mv, kHeads, &wrong_rows, capacity)
        == Status::kInvalidDimensions);
    CHECK(all_sentinel(out, capacity));
    // Stride classes (operands and output).
    CHECK(swin_mid(make_view(hidden, kRows, kGroup, kGroup - 1), mv, kHeads, &o,
            capacity) == Status::kInvalidStride);
    CHECK(swin_mid(make_view(hidden, kRows, kGroup,
            dlss5_nr::kMaxLinearDim + 1), mv, kHeads, &o, capacity)
        == Status::kInvalidStride);
    CHECK(swin_mid(hv, make_view(mid, kOut, kGroup, kGroup - 1), kHeads, &o,
            capacity) == Status::kInvalidStride);
    CHECK(swin_mid(hv, make_view(mid, kOut, kGroup, dlss5_nr::kMaxLinearDim + 1),
            kHeads, &o, capacity) == Status::kInvalidStride);
    MutableMatrixView bad_out = o;
    bad_out.row_stride = kOut - 1;
    CHECK(swin_mid(hv, mv, kHeads, &bad_out, capacity) == Status::kInvalidStride);
    bad_out = o;
    bad_out.row_stride = dlss5_nr::kMaxLinearDim + 1;
    CHECK(swin_mid(hv, mv, kHeads, &bad_out, capacity) == Status::kInvalidStride);
    CHECK(all_sentinel(out, capacity));
    // Short capacity (one float short of rows * heads * kHeadDim).
    CHECK(swin_mid(hv, mv, kHeads, &o, capacity - 1) == Status::kInvalidBuffer);
    CHECK(all_sentinel(out, capacity));
    // e4m3 operand-grid class: off-grid, out-of-range and non-finite operands in
    // both the hidden and the mid side.
    const float bad_values[] = {0.3f, 1.0625f, 0x1p-10f, 500.0f, -500.0f,
        (float)NAN, (float)INFINITY, (float)-INFINITY};
    for (std::size_t i = 0; i < sizeof bad_values / sizeof bad_values[0]; ++i) {
        hidden[0] = bad_values[i];
        CHECK(swin_mid(hv, mv, kHeads, &o, capacity) == Status::kInvalidValues);
        hidden[0] = 0.5f;
        mid[0] = bad_values[i];
        CHECK(swin_mid(hv, mv, kHeads, &o, capacity) == Status::kInvalidValues);
        mid[0] = 0.5f;
    }
    CHECK(all_sentinel(out, capacity));
    // A valid boundary operand class still succeeds: +-448 and the subnormal grid.
    hidden[0] = 448.0f;
    hidden[1] = -448.0f;
    hidden[2] = 0x1p-9f;
    mid[0] = -0.0f;
    CHECK(swin_mid(hv, mv, kHeads, &o, capacity) == Status::kOk);
}

static void check_negative_tail(void) {
    using dlss5_nr::swin_mlp_tail;
    constexpr int kRows = 2;
    constexpr int kK = 3;
    constexpr int kN = 2;
    float in[kRows * kK];
    // One extra column of padding so the over-sized pairing view's grid scan stays
    // in bounds (the pairing check runs after the scan).
    float contract[kN * (kK + 1)];
    float residual[kRows * kN];
    float scale[kN];
    float out[kRows * kN];
    for (int p = 0; p < kRows * kK; ++p) {
        in[p] = 1.0f;
    }
    for (int p = 0; p < kN * (kK + 1); ++p) {
        contract[p] = 0.5f;
    }
    for (int p = 0; p < kRows * kN; ++p) {
        residual[p] = 1.0f;
    }
    for (int p = 0; p < kN; ++p) {
        scale[p] = 1.0f;
    }
    fill_sentinel(out, kRows * kN);
    const std::size_t capacity = kRows * kN;
    const MatrixView iv = make_view(in, kRows, kK, kK);
    const MatrixView cv = make_view(contract, kN, kK, kK);
    const MatrixView rv = make_view(residual, kRows, kN, kN);
    const MatrixView sv = make_view(scale, 1, kN, kN);
    MutableMatrixView o = make_out(out, kRows, kN, kN);
    // Null pointers (the null-output rejection cannot write by construction).
    CHECK(swin_mlp_tail(make_view(nullptr, kRows, kK, kK), cv, rv, sv, &o,
            capacity) == Status::kNullPointer);
    CHECK(swin_mlp_tail(iv, make_view(nullptr, kN, kK, kK), rv, sv, &o, capacity)
        == Status::kNullPointer);
    CHECK(swin_mlp_tail(iv, cv, make_view(nullptr, kRows, kN, kN), sv, &o,
            capacity) == Status::kNullPointer);
    CHECK(swin_mlp_tail(iv, cv, rv, make_view(nullptr, 1, kN, kN), &o, capacity)
        == Status::kNullPointer);
    CHECK(swin_mlp_tail(iv, cv, rv, sv, nullptr, capacity)
        == Status::kNullPointer);
    MutableMatrixView no_out = o;
    no_out.data = nullptr;
    CHECK(swin_mlp_tail(iv, cv, rv, sv, &no_out, capacity)
        == Status::kNullPointer);
    CHECK(all_sentinel(out, capacity));
    // Operand dimension classes for `in` and `contract`.
    CHECK(swin_mlp_tail(make_view(in, 0, kK, kK), cv, rv, sv, &o, capacity)
        == Status::kInvalidDimensions);
    CHECK(swin_mlp_tail(make_view(in, kRows, 0, kK), cv, rv, sv, &o, capacity)
        == Status::kInvalidDimensions);
    CHECK(swin_mlp_tail(make_view(in, dlss5_nr::kMaxLinearRows + 1, kK, kK), cv,
            rv, sv, &o, capacity) == Status::kInvalidDimensions);
    CHECK(swin_mlp_tail(make_view(in, kRows, dlss5_nr::kMaxLinearDim + 1,
            kK + 1), cv, rv, sv, &o, capacity) == Status::kInvalidDimensions);
    CHECK(all_sentinel(out, capacity));
    // Pairing and output dimension classes.
    CHECK(swin_mlp_tail(iv, make_view(contract, kN, kK + 1, kK + 1), rv, sv, &o,
            capacity) == Status::kInvalidDimensions);
    CHECK(swin_mlp_tail(iv, cv, make_view(residual, kRows + 1, kN, kN), sv, &o,
            capacity) == Status::kInvalidDimensions);
    CHECK(swin_mlp_tail(iv, cv, make_view(residual, kRows, kN + 1, kN + 1), sv,
            &o, capacity) == Status::kInvalidDimensions);
    CHECK(swin_mlp_tail(iv, cv, rv, make_view(scale, 2, kN, kN), &o, capacity)
        == Status::kInvalidDimensions);
    CHECK(swin_mlp_tail(iv, cv, rv, make_view(scale, 1, kN + 1, kN + 1), &o,
            capacity) == Status::kInvalidDimensions);
    MutableMatrixView wrong_rows = o;
    wrong_rows.rows = kRows + 1;
    CHECK(swin_mlp_tail(iv, cv, rv, sv, &wrong_rows, capacity)
        == Status::kInvalidDimensions);
    CHECK(all_sentinel(out, capacity));
    // Residual/scale stride classes (the residual scale validator's own bounds).
    CHECK(swin_mlp_tail(iv, cv, make_view(residual, kRows, kN, kN - 1), sv, &o,
            capacity) == Status::kInvalidStride);
    CHECK(swin_mlp_tail(iv, cv, make_view(residual, kRows, kN,
            dlss5_nr::kMaxLinearDim + 1), sv, &o, capacity)
        == Status::kInvalidStride);
    CHECK(swin_mlp_tail(iv, cv, rv, make_view(scale, 1, kN, kN - 1), &o, capacity)
        == Status::kInvalidStride);
    CHECK(swin_mlp_tail(iv, cv, rv, make_view(scale, 1, kN,
            dlss5_nr::kMaxLinearDim + 1), &o, capacity) == Status::kInvalidStride);
    CHECK(all_sentinel(out, capacity));
    // Operand stride classes and the output stride classes.
    CHECK(swin_mlp_tail(make_view(in, kRows, kK, kK - 1), cv, rv, sv, &o,
            capacity) == Status::kInvalidStride);
    CHECK(swin_mlp_tail(iv, make_view(contract, kN, kK, kK - 1), rv, sv, &o,
            capacity) == Status::kInvalidStride);
    MutableMatrixView bad_out = o;
    bad_out.row_stride = kN - 1;
    CHECK(swin_mlp_tail(iv, cv, rv, sv, &bad_out, capacity)
        == Status::kInvalidStride);
    bad_out = o;
    bad_out.row_stride = dlss5_nr::kMaxLinearDim + 1;
    CHECK(swin_mlp_tail(iv, cv, rv, sv, &bad_out, capacity)
        == Status::kInvalidStride);
    CHECK(all_sentinel(out, capacity));
    // Short capacity (one float short of rows * n).
    CHECK(swin_mlp_tail(iv, cv, rv, sv, &o, capacity - 1)
        == Status::kInvalidBuffer);
    CHECK(all_sentinel(out, capacity));
    // Value classes: the e4m3 grids of `in`/`contract`, the finiteness of the
    // residual, and the finite f16-exact scale grid.
    const float bad_grid[] = {0.3f, 1.0625f, 0x1p-10f, 500.0f, -500.0f,
        (float)NAN, (float)INFINITY, (float)-INFINITY};
    for (std::size_t i = 0; i < sizeof bad_grid / sizeof bad_grid[0]; ++i) {
        in[0] = bad_grid[i];
        CHECK(swin_mlp_tail(iv, cv, rv, sv, &o, capacity)
            == Status::kInvalidValues);
        in[0] = 1.0f;
        contract[0] = bad_grid[i];
        CHECK(swin_mlp_tail(iv, cv, rv, sv, &o, capacity)
            == Status::kInvalidValues);
        contract[0] = 0.5f;
    }
    const float bad_residual[] = {(float)NAN, (float)INFINITY,
        (float)-INFINITY};
    for (std::size_t i = 0; i < sizeof bad_residual / sizeof bad_residual[0];
            ++i) {
        residual[0] = bad_residual[i];
        CHECK(swin_mlp_tail(iv, cv, rv, sv, &o, capacity)
            == Status::kInvalidValues);
        residual[0] = 1.0f;
    }
    const float bad_scale[] = {(float)NAN, (float)INFINITY, (float)-INFINITY,
        0.3f, 0x1p-25f};
    for (std::size_t i = 0; i < sizeof bad_scale / sizeof bad_scale[0]; ++i) {
        scale[0] = bad_scale[i];
        CHECK(swin_mlp_tail(iv, cv, rv, sv, &o, capacity)
            == Status::kInvalidValues);
        scale[0] = 1.0f;
    }
    CHECK(all_sentinel(out, capacity));
    // A valid boundary operand class still succeeds: +-448 and the subnormal grid
    // on the e4m3 sides, an f16-exact scale and finite residuals.
    in[0] = 448.0f;
    in[1] = -448.0f;
    in[2] = 0x1p-9f;
    contract[0] = -0.0f;
    residual[0] = -3.0f;
    scale[0] = 0x1p-14f;
    CHECK(swin_mlp_tail(iv, cv, rv, sv, &o, capacity) == Status::kOk);
    // The documented overflow passthrough: a finite residual scaled past the
    // float32 range gives the IEEE infinity, not a rejection or a NaN.
    residual[0] = 3.4028234663852886e38f;
    scale[0] = 2.0f;
    CHECK(swin_mlp_tail(iv, cv, rv, sv, &o, capacity) == Status::kOk);
    CHECK(out[0] == (float)INFINITY);
}

static void check_negative_mlp(void) {
    using dlss5_nr::swin_mlp;
    constexpr int kRows = 2;
    constexpr int kXCols = 2;
    constexpr int kExpandRows = 4;
    constexpr int kN = 3;
    // One spare column/row in the pairing-sensitive buffers so the over-sized
    // pairing views' grid scans stay in bounds (those pairings are rejected only
    // after the scans).
    float x[kRows * (kXCols + 1)];
    float expand[kExpandRows * kXCols];
    float contract[kN * (kExpandRows + 1)];
    float residual[kRows * kN];
    float scale[kN];
    float out[kRows * kN];
    float scratch_buf[kRows * kExpandRows];
    for (int p = 0; p < kRows * (kXCols + 1); ++p) {
        x[p] = 1.0f;
    }
    for (int p = 0; p < kExpandRows * kXCols; ++p) {
        expand[p] = 0.5f;
    }
    for (int p = 0; p < kN * (kExpandRows + 1); ++p) {
        contract[p] = 0.5f;
    }
    for (int p = 0; p < kRows * kN; ++p) {
        residual[p] = 1.0f;
    }
    for (int p = 0; p < kN; ++p) {
        scale[p] = 1.0f;
    }
    fill_sentinel(out, kRows * kN);
    fill_sentinel(scratch_buf, kRows * kExpandRows);
    const std::size_t capacity = kRows * kN;
    const std::size_t scratch_capacity = kRows * kExpandRows;
    const MatrixView xv = make_view(x, kRows, kXCols, kXCols);
    const MatrixView expand_v = make_view(expand, kExpandRows, kXCols, kXCols);
    const MatrixView cv = make_view(contract, kN, kExpandRows, kExpandRows);
    const MatrixView rv = make_view(residual, kRows, kN, kN);
    const MatrixView sv = make_view(scale, 1, kN, kN);
    MutableMatrixView o = make_out(out, kRows, kN, kN);
    MutableMatrixView sc =
        make_out(scratch_buf, kRows, kExpandRows, kExpandRows);
    // A local sentinel check so every rejection group asserts both buffers
    // untouched (out and scratch).
    const auto untouched = [&out, &scratch_buf, capacity, scratch_capacity]() {
        CHECK(all_sentinel(out, capacity));
        CHECK(all_sentinel(scratch_buf, scratch_capacity));
    };
    // Null pointers (the null-output rejection cannot write by construction).
    CHECK(swin_mlp(make_view(nullptr, kRows, kXCols, kXCols), expand_v, nullptr, 1,
            cv, rv, sv, &o, capacity, &sc, scratch_capacity)
        == Status::kNullPointer);
    CHECK(swin_mlp(xv, make_view(nullptr, kExpandRows, kXCols, kXCols), nullptr, 1,
            cv, rv, sv, &o, capacity, &sc, scratch_capacity)
        == Status::kNullPointer);
    CHECK(swin_mlp(xv, expand_v, nullptr, 1,
            make_view(nullptr, kN, kExpandRows, kExpandRows), rv, sv, &o,
            capacity, &sc, scratch_capacity) == Status::kNullPointer);
    CHECK(swin_mlp(xv, expand_v, nullptr, 1, cv,
            make_view(nullptr, kRows, kN, kN), sv, &o, capacity, &sc,
            scratch_capacity) == Status::kNullPointer);
    CHECK(swin_mlp(xv, expand_v, nullptr, 1, cv, rv,
            make_view(nullptr, 1, kN, kN), &o, capacity, &sc, scratch_capacity)
        == Status::kNullPointer);
    {
        MatrixView null_mid = make_view(nullptr, dlss5_nr::kHeadDim, 1, 1);
        CHECK(swin_mlp(xv, expand_v, &null_mid, 1, cv, rv, sv, &o, capacity, &sc,
                scratch_capacity) == Status::kNullPointer);
    }
    CHECK(swin_mlp(xv, expand_v, nullptr, 1, cv, rv, sv, nullptr, capacity, &sc,
            scratch_capacity) == Status::kNullPointer);
    {
        MutableMatrixView no_out = o;
        no_out.data = nullptr;
        CHECK(swin_mlp(xv, expand_v, nullptr, 1, cv, rv, sv, &no_out, capacity,
                &sc, scratch_capacity) == Status::kNullPointer);
    }
    CHECK(swin_mlp(xv, expand_v, nullptr, 1, cv, rv, sv, &o, capacity, nullptr,
            scratch_capacity) == Status::kNullPointer);
    {
        MutableMatrixView no_scratch = sc;
        no_scratch.data = nullptr;
        CHECK(swin_mlp(xv, expand_v, nullptr, 1, cv, rv, sv, &o, capacity,
                &no_scratch, scratch_capacity) == Status::kNullPointer);
    }
    untouched();
    // Operand dimension classes (x, expand and contract in turn).
    CHECK(swin_mlp(make_view(x, 0, kXCols, kXCols), expand_v, nullptr, 1, cv, rv,
            sv, &o, capacity, &sc, scratch_capacity)
        == Status::kInvalidDimensions);
    CHECK(swin_mlp(xv, make_view(expand, kExpandRows, 0, kXCols), nullptr, 1, cv,
            rv, sv, &o, capacity, &sc, scratch_capacity)
        == Status::kInvalidDimensions);
    CHECK(swin_mlp(xv, make_view(expand, dlss5_nr::kMaxLinearRows + 1, kXCols,
            kXCols), nullptr, 1, cv, rv, sv, &o, capacity, &sc, scratch_capacity)
        == Status::kInvalidDimensions);
    CHECK(swin_mlp(xv, make_view(expand, kExpandRows,
            dlss5_nr::kMaxLinearDim + 1, kXCols + 1), nullptr, 1, cv, rv, sv, &o,
            capacity, &sc, scratch_capacity) == Status::kInvalidDimensions);
    CHECK(swin_mlp(xv, expand_v, nullptr, 1,
            make_view(contract, kN, 0, kExpandRows), rv, sv, &o, capacity, &sc,
            scratch_capacity) == Status::kInvalidDimensions);
    untouched();
    // Pairing and output dimension classes (mid null and mid paths).
    CHECK(swin_mlp(make_view(x, kRows, kXCols + 1, kXCols + 1), expand_v, nullptr,
            1, cv, rv, sv, &o, capacity, &sc, scratch_capacity)
        == Status::kInvalidDimensions);
    {
        MutableMatrixView wrong_rows = o;
        wrong_rows.rows = kRows + 1;
        CHECK(swin_mlp(xv, expand_v, nullptr, 1, cv, rv, sv, &wrong_rows,
                capacity, &sc, scratch_capacity) == Status::kInvalidDimensions);
    }
    CHECK(swin_mlp(xv, expand_v, nullptr, 1,
            make_view(contract, kN, kExpandRows + 1, kExpandRows + 1), rv, sv,
            &o, capacity, &sc, scratch_capacity) == Status::kInvalidDimensions);
    // Residual/scale dimension classes.
    CHECK(swin_mlp(xv, expand_v, nullptr, 1, cv,
            make_view(residual, kRows + 1, kN, kN), sv, &o, capacity, &sc,
            scratch_capacity) == Status::kInvalidDimensions);
    CHECK(swin_mlp(xv, expand_v, nullptr, 1, cv,
            make_view(residual, kRows, kN + 1, kN + 1), sv, &o, capacity, &sc,
            scratch_capacity) == Status::kInvalidDimensions);
    CHECK(swin_mlp(xv, expand_v, nullptr, 1, cv, rv,
            make_view(scale, 2, kN, kN), &o, capacity, &sc, scratch_capacity)
        == Status::kInvalidDimensions);
    CHECK(swin_mlp(xv, expand_v, nullptr, 1, cv, rv,
            make_view(scale, 1, kN + 1, kN + 1), &o, capacity, &sc,
            scratch_capacity) == Status::kInvalidDimensions);
    untouched();
    // Mid-path dimension classes: the heads bound and the three pairings.
    float mid_buf[(dlss5_nr::kHeadDim + 1) * (kExpandRows + 1)];
    for (int p = 0; p < (dlss5_nr::kHeadDim + 1) * (kExpandRows + 1); ++p) {
        mid_buf[p] = 0.5f;
    }
    // heads = 1, group width kExpandRows, so contract must be kN x kHeadDim.
    float contract_mid[kN * (dlss5_nr::kHeadDim + 1)];
    for (int p = 0; p < kN * (dlss5_nr::kHeadDim + 1); ++p) {
        contract_mid[p] = 0.5f;
    }
    const MatrixView mid_v = make_view(mid_buf, dlss5_nr::kHeadDim, kExpandRows,
        kExpandRows);
    const MatrixView cmv = make_view(contract_mid, kN, dlss5_nr::kHeadDim,
        dlss5_nr::kHeadDim);
    CHECK(swin_mlp(xv, expand_v, &mid_v, 0, cmv, rv, sv, &o, capacity, &sc,
            scratch_capacity) == Status::kInvalidDimensions);
    CHECK(swin_mlp(xv, expand_v, &mid_v,
            dlss5_nr::kMaxLinearDim / dlss5_nr::kHeadDim + 1, cmv, rv, sv, &o,
            capacity, &sc, scratch_capacity) == Status::kInvalidDimensions);
    {
        // The over-sized mid/contract pair: mid.rows 33 != heads * kHeadDim (and
        // contract.cols 33 != mid->rows for the oversized contract), rejected at
        // the mid branch after both grid scans.
        const MatrixView wide_mid = make_view(mid_buf, dlss5_nr::kHeadDim + 1,
            kExpandRows, kExpandRows + 1);
        const MatrixView wide_contract = make_view(contract_mid, kN,
            dlss5_nr::kHeadDim + 1, dlss5_nr::kHeadDim + 1);
        CHECK(swin_mlp(xv, expand_v, &wide_mid, 1, wide_contract, rv, sv, &o,
                capacity, &sc, scratch_capacity) == Status::kInvalidDimensions);
    }
    // expand.rows != heads * mid->cols: a mid with kExpandRows - 1 columns.
    {
        const MatrixView narrow_mid = make_view(mid_buf, dlss5_nr::kHeadDim,
            kExpandRows - 1, kExpandRows - 1);
        CHECK(swin_mlp(xv, expand_v, &narrow_mid, 1, cmv, rv, sv, &o, capacity,
                &sc, scratch_capacity) == Status::kInvalidDimensions);
    }
    // contract.cols != mid->rows on the mid path (a plain-width contract).
    CHECK(swin_mlp(xv, expand_v, &mid_v, 1, cv, rv, sv, &o, capacity, &sc,
            scratch_capacity) == Status::kInvalidDimensions);
    untouched();
    // Scratch dimension classes.
    {
        MutableMatrixView wrong_scratch = sc;
        wrong_scratch.rows = kRows + 1;
        CHECK(swin_mlp(xv, expand_v, nullptr, 1, cv, rv, sv, &o, capacity,
                &wrong_scratch, scratch_capacity) == Status::kInvalidDimensions);
        wrong_scratch = sc;
        wrong_scratch.cols = kExpandRows + 1;
        CHECK(swin_mlp(xv, expand_v, nullptr, 1, cv, rv, sv, &o, capacity,
                &wrong_scratch, scratch_capacity) == Status::kInvalidDimensions);
    }
    // The over-wide scratch branch of the output validator: on the mid path
    // expand.rows = heads * mid.cols can exceed kMaxLinearDim while every operand
    // stays valid (mid.cols 513, heads 2 -> expand.rows 1026), which is rejected
    // by validate_linear_output's cols bound at the scratch check — the only
    // reachable entry to that bound through these three APIs (swin_mid's output
    // width is heads * kHeadDim <= kMaxLinearDim and the tail's contract.rows is
    // confined by the residual/scale stride bound).
    {
        constexpr int kWideHeads = 2;
        constexpr int kWideGroup = 513;
        constexpr int kWideExpand = kWideHeads * kWideGroup;  // 1026 > kMaxLinearDim
        static float wide_expand[kWideExpand * kXCols];
        static float wide_mid[kWideHeads * dlss5_nr::kHeadDim * kWideGroup];
        static float wide_contract[kN * kWideHeads * dlss5_nr::kHeadDim];
        static float wide_scratch[kRows * kWideExpand];
        for (int p = 0; p < kWideExpand * kXCols; ++p) {
            wide_expand[p] = 0.5f;
        }
        for (int p = 0; p < kWideHeads * dlss5_nr::kHeadDim * kWideGroup; ++p) {
            wide_mid[p] = 0.5f;
        }
        for (int p = 0; p < kN * kWideHeads * dlss5_nr::kHeadDim; ++p) {
            wide_contract[p] = 0.5f;
        }
        fill_sentinel(wide_scratch, kRows * kWideExpand);
        const MatrixView wide_mid_v = make_view(wide_mid,
            kWideHeads * dlss5_nr::kHeadDim, kWideGroup, kWideGroup);
        const MatrixView wide_contract_v = make_view(wide_contract, kN,
            kWideHeads * dlss5_nr::kHeadDim, kWideHeads * dlss5_nr::kHeadDim);
        MutableMatrixView wide_scratch_v = make_out(wide_scratch, kRows,
            kWideExpand, kWideExpand);
        CHECK(swin_mlp(xv, make_view(wide_expand, kWideExpand, kXCols, kXCols),
                &wide_mid_v, kWideHeads, wide_contract_v, rv, sv, &o, capacity,
                &wide_scratch_v, kRows * kWideExpand) == Status::kInvalidDimensions);
        CHECK(all_sentinel(wide_scratch, kRows * kWideExpand));
    }
    untouched();
    // Residual/scale stride and value classes.
    CHECK(swin_mlp(xv, expand_v, nullptr, 1, cv,
            make_view(residual, kRows, kN, kN - 1), sv, &o, capacity, &sc,
            scratch_capacity) == Status::kInvalidStride);
    CHECK(swin_mlp(xv, expand_v, nullptr, 1, cv,
            make_view(residual, kRows, kN, dlss5_nr::kMaxLinearDim + 1), sv, &o,
            capacity, &sc, scratch_capacity) == Status::kInvalidStride);
    CHECK(swin_mlp(xv, expand_v, nullptr, 1, cv, rv,
            make_view(scale, 1, kN, kN - 1), &o, capacity, &sc, scratch_capacity)
        == Status::kInvalidStride);
    CHECK(swin_mlp(xv, expand_v, nullptr, 1, cv, rv,
            make_view(scale, 1, kN, dlss5_nr::kMaxLinearDim + 1), &o, capacity,
            &sc, scratch_capacity) == Status::kInvalidStride);
    residual[0] = (float)NAN;
    CHECK(swin_mlp(xv, expand_v, nullptr, 1, cv, rv, sv, &o, capacity, &sc,
            scratch_capacity) == Status::kInvalidValues);
    residual[0] = 1.0f;
    scale[0] = 0.3f;
    CHECK(swin_mlp(xv, expand_v, nullptr, 1, cv, rv, sv, &o, capacity, &sc,
            scratch_capacity) == Status::kInvalidValues);
    scale[0] = 1.0f;
    untouched();
    // Output and scratch stride classes and capacities.
    {
        MutableMatrixView bad_out = o;
        bad_out.row_stride = kN - 1;
        CHECK(swin_mlp(xv, expand_v, nullptr, 1, cv, rv, sv, &bad_out, capacity,
                &sc, scratch_capacity) == Status::kInvalidStride);
        bad_out = o;
        bad_out.row_stride = dlss5_nr::kMaxLinearDim + 1;
        CHECK(swin_mlp(xv, expand_v, nullptr, 1, cv, rv, sv, &bad_out, capacity,
                &sc, scratch_capacity) == Status::kInvalidStride);
        MutableMatrixView bad_scratch = sc;
        bad_scratch.row_stride = kExpandRows - 1;
        CHECK(swin_mlp(xv, expand_v, nullptr, 1, cv, rv, sv, &o, capacity,
                &bad_scratch, scratch_capacity) == Status::kInvalidStride);
        bad_scratch = sc;
        bad_scratch.row_stride = dlss5_nr::kMaxLinearDim + 1;
        CHECK(swin_mlp(xv, expand_v, nullptr, 1, cv, rv, sv, &o, capacity,
                &bad_scratch, scratch_capacity) == Status::kInvalidStride);
    }
    CHECK(swin_mlp(xv, expand_v, nullptr, 1, cv, rv, sv, &o, capacity - 1, &sc,
            scratch_capacity) == Status::kInvalidBuffer);
    CHECK(swin_mlp(xv, expand_v, nullptr, 1, cv, rv, sv, &o, capacity, &sc,
            scratch_capacity - 1) == Status::kInvalidBuffer);
    untouched();
    // e4m3 operand-grid classes on every grid side.
    const float bad_grid[] = {0.3f, 1.0625f, 0x1p-10f, 500.0f, -500.0f,
        (float)NAN, (float)INFINITY, (float)-INFINITY};
    for (std::size_t i = 0; i < sizeof bad_grid / sizeof bad_grid[0]; ++i) {
        x[0] = bad_grid[i];
        CHECK(swin_mlp(xv, expand_v, nullptr, 1, cv, rv, sv, &o, capacity, &sc,
                scratch_capacity) == Status::kInvalidValues);
        x[0] = 1.0f;
        expand[0] = bad_grid[i];
        CHECK(swin_mlp(xv, expand_v, nullptr, 1, cv, rv, sv, &o, capacity, &sc,
                scratch_capacity) == Status::kInvalidValues);
        expand[0] = 0.5f;
        contract[0] = bad_grid[i];
        CHECK(swin_mlp(xv, expand_v, nullptr, 1, cv, rv, sv, &o, capacity, &sc,
                scratch_capacity) == Status::kInvalidValues);
        contract[0] = 0.5f;
        mid_buf[0] = bad_grid[i];
        CHECK(swin_mlp(xv, expand_v, &mid_v, 1, cmv, rv, sv, &o, capacity, &sc,
                scratch_capacity) == Status::kInvalidValues);
        mid_buf[0] = 0.5f;
        contract_mid[0] = bad_grid[i];
        CHECK(swin_mlp(xv, expand_v, &mid_v, 1, cmv, rv, sv, &o, capacity, &sc,
                scratch_capacity) == Status::kInvalidValues);
        contract_mid[0] = 0.5f;
    }
    untouched();
    // A valid boundary operand class still succeeds on both paths.
    x[0] = 448.0f;
    x[1] = -0.0f;
    x[2] = 0x1p-9f;
    CHECK(swin_mlp(xv, expand_v, nullptr, 1, cv, rv, sv, &o, capacity, &sc,
            scratch_capacity) == Status::kOk);
    CHECK(swin_mlp(xv, expand_v, &mid_v, 1, cmv, rv, sv, &o, capacity, &sc,
            scratch_capacity) == Status::kOk);
}

// ---- allocation discipline --------------------------------------------------------------------------
static void check_allocation_discipline(void) {
    using dlss5_nr::swin_mid;
    using dlss5_nr::swin_mlp;
    using dlss5_nr::swin_mlp_tail;
    constexpr int kRows = 2;
    constexpr int kXCols = 2;
    constexpr int kExpandRows = 4;
    constexpr int kN = 3;
    float x[kRows * kXCols];
    float expand[kExpandRows * kXCols];
    float contract[kN * kExpandRows];
    float residual[kRows * kN];
    float scale[kN];
    float out[kRows * kN];
    float out2[kRows * kN];
    float scratch_buf[kRows * kExpandRows];
    for (int p = 0; p < kRows * kXCols; ++p) {
        x[p] = (float)(p % 5) * 0.5f - 1.0f;
    }
    for (int p = 0; p < kExpandRows * kXCols; ++p) {
        expand[p] = (float)(p % 7) * 0.25f - 0.75f;
    }
    for (int p = 0; p < kN * kExpandRows; ++p) {
        contract[p] = (float)(p % 3) * 0.5f - 0.5f;
    }
    for (int p = 0; p < kRows * kN; ++p) {
        residual[p] = (float)(p % 4) * -0.5f + 1.0f;
    }
    for (int p = 0; p < kN; ++p) {
        scale[p] = (float)(p + 1) * 0.25f;
    }
    float mid_buf[dlss5_nr::kHeadDim * kExpandRows];
    for (int p = 0; p < dlss5_nr::kHeadDim * kExpandRows; ++p) {
        mid_buf[p] = (float)(p % 5) * 0.25f - 0.5f;
    }
    float contract_mid[kN * dlss5_nr::kHeadDim];
    for (int p = 0; p < kN * dlss5_nr::kHeadDim; ++p) {
        contract_mid[p] = (float)(p % 3) * -0.25f + 0.25f;
    }
    const MatrixView xv = make_view(x, kRows, kXCols, kXCols);
    const MatrixView expand_v = make_view(expand, kExpandRows, kXCols, kXCols);
    const MatrixView cv = make_view(contract, kN, kExpandRows, kExpandRows);
    const MatrixView mid_v = make_view(mid_buf, dlss5_nr::kHeadDim, kExpandRows,
        kExpandRows);
    const MatrixView cmv = make_view(contract_mid, kN, dlss5_nr::kHeadDim,
        dlss5_nr::kHeadDim);
    const MatrixView rv = make_view(residual, kRows, kN, kN);
    const MatrixView sv = make_view(scale, 1, kN, kN);
    MutableMatrixView o = make_out(out, kRows, kN, kN);
    MutableMatrixView o2 = make_out(out2, kRows, kN, kN);
    MutableMatrixView sc =
        make_out(scratch_buf, kRows, kExpandRows, kExpandRows);
    const unsigned long long before = g_allocations.load();
    for (int i = 0; i < 64; ++i) {
        CHECK(swin_mlp(xv, expand_v, nullptr, 1, cv, rv, sv, &o, kRows * kN, &sc,
                kRows * kExpandRows) == Status::kOk);
        CHECK(swin_mlp(xv, expand_v, &mid_v, 1, cmv, rv, sv, &o, kRows * kN, &sc,
                kRows * kExpandRows) == Status::kOk);
    }
    CHECK(g_allocations.load() == before);
    // Determinism of the composed assembly: repeated runs of the same path are
    // bit-identical (each path compared against its own repeated run).
    CHECK(swin_mlp(xv, expand_v, nullptr, 1, cv, rv, sv, &o, kRows * kN, &sc,
            kRows * kExpandRows) == Status::kOk);
    CHECK(swin_mlp(xv, expand_v, nullptr, 1, cv, rv, sv, &o2, kRows * kN, &sc,
            kRows * kExpandRows) == Status::kOk);
    CHECK(memcmp(out2, out, sizeof out) == 0);
    CHECK(swin_mlp(xv, expand_v, &mid_v, 1, cmv, rv, sv, &o, kRows * kN, &sc,
            kRows * kExpandRows) == Status::kOk);
    CHECK(swin_mlp(xv, expand_v, &mid_v, 1, cmv, rv, sv, &o2, kRows * kN, &sc,
            kRows * kExpandRows) == Status::kOk);
    CHECK(memcmp(out2, out, sizeof out) == 0);
}

int main(void) {
    check_store_pins();
    check_mid_known_answers();
    check_tail_known_answers();
    check_mlp_known_answers();
    check_mid_permutation_and_sensitivity();
    check_tail_permutation_and_sensitivity();
    check_assembly_equivalence();
    check_negative_mid();
    check_negative_tail();
    check_negative_mlp();
    check_allocation_discipline();
    printf("native-swin-mlp-test: %llu checks passed\n", g_checks);
    return 0;
}
