// Independent tests for the native Swin e4m3 linears (vr/native-swin-linear.hpp).
// CPU-only, no GPU, headset, proprietary payload, Python or PyTorch runtime required.
//
// Independence: every known answer is a hand-derived dyadic rational (the corpus is
// drawn from e4m3-grid values, so all products are exact and the expected sums are
// exact or hand-rounded). The accumulation-order discriminator pins this slice's
// explicit sequential k-order float32 accumulator with a tie-to-even midpoint case
// where summing the two tail products first, reverse k-order or exact (float64)
// summation each give a provably different value; not every reordering is
// discriminated (moving a single trailing product to the front or transposing the
// two trailing products still hits the recurring tie-to-even midpoint), and the
// test pins those boundaries explicitly. Controls cover row/channel/K permutations
// (the corpus has pairwise-distinct weight rows and sensitivity deltas, guarded so
// the controls cannot be vacuous), changed-input sensitivity with exact deltas, the
// e4m3 operand-grid validation class, sentinel-checked untouched outputs on every
// validation rejection class (the null-output rejection cannot write by
// construction), determinism and zero-allocation accounting.
// See vr/NATIVE-SWIN-LINEAR.md.

#include "native-swin-linear.hpp"

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

// ---- hand-derived known answers -------------------------------------------------------------------
// Corpus values are e4m3-grid dyadic rationals, so every product and the expected
// sums below are exact.
static void check_known_answers(void) {
    using dlss5_nr::swin_linear;
    // K = 1: 2 * 3 = 6.
    {
        const float x[1] = {2.0f};
        const float w[1] = {3.0f};
        float out[1];
        MutableMatrixView o = make_out(out, 1, 1, 1);
        CHECK(swin_linear(make_view(x, 1, 1, 1), make_view(w, 1, 1, 1), &o, 1)
            == Status::kOk);
        CHECK(out[0] == 6.0f);
    }
    // K = 2: 1*0.5 + 2*0.25 = 1.
    {
        const float x[2] = {1.0f, 2.0f};
        const float w[2] = {0.5f, 0.25f};
        float out[1];
        MutableMatrixView o = make_out(out, 1, 1, 1);
        CHECK(swin_linear(make_view(x, 1, 2, 2), make_view(w, 1, 2, 2), &o, 1)
            == Status::kOk);
        CHECK(out[0] == 1.0f);
    }
    // Signs: (-1)*2 + 0.5*(-4) = -4.
    {
        const float x[2] = {-1.0f, 0.5f};
        const float w[2] = {2.0f, -4.0f};
        float out[1];
        MutableMatrixView o = make_out(out, 1, 1, 1);
        CHECK(swin_linear(make_view(x, 1, 2, 2), make_view(w, 1, 2, 2), &o, 1)
            == Status::kOk);
        CHECK(out[0] == -4.0f);
    }
    // rows = 2, k = 2, n = 2 with padded strides:
    //   out[0] = (1*0.5 + 2*0.25, 1*2 + 2*4) = (1, 10)
    //   out[1] = (3*0.5 + (-4)*0.25, 3*2 + (-4)*4) = (0.5, -10)
    {
        float x[2 * 3];
        float w[2 * 3];
        float out[2 * 4];
        const float x_values[4] = {1.0f, 2.0f, 3.0f, -4.0f};
        const float w_values[4] = {0.5f, 0.25f, 2.0f, 4.0f};
        for (int r = 0; r < 2; ++r) {
            for (int c = 0; c < 2; ++c) {
                x[r * 3 + c] = x_values[r * 2 + c];
                w[r * 3 + c] = w_values[r * 2 + c];
            }
            x[r * 3 + 2] = kSentinel;
            w[r * 3 + 2] = kSentinel;
        }
        fill_sentinel(out, 8);
        MutableMatrixView o = make_out(out, 2, 2, 4);
        CHECK(swin_linear(make_view(x, 2, 2, 3), make_view(w, 2, 2, 3), &o, 8)
            == Status::kOk);
        CHECK(out[0] == 1.0f && out[1] == 10.0f);
        CHECK(out[4] == 0.5f && out[5] == -10.0f);
        // Stride padding is never written (inputs never written at all).
        CHECK(out[2] == kSentinel && out[3] == kSentinel);
        CHECK(out[6] == kSentinel && out[7] == kSentinel);
        CHECK(x[2] == kSentinel && w[2] == kSentinel && x[5] == kSentinel
            && w[5] == kSentinel);
    }
}

// ---- accumulation-order discriminator ---------------------------------------------------------------
// Products [2^16 x 128, 0.5, 0.5]: the sequential k-order float32 accumulator reaches
// exactly 2^23 after the 128 leading products, then each 0.5 hits the tie-to-even
// midpoint (ulp 1 at 2^23) and is dropped -> 2^23 = 8388608. Summing the two tail
// products first (or reverse k-order, or exact/float64 summation) gives 8388609.
// Both asserted distinct. Not every reordering is discriminated: moving a single
// trailing 0.5 to the front or transposing the two trailing products hits the same
// recurring tie-to-even midpoint and still gives 8388608; the boundary pins below
// assert exactly that, so the discriminator's scope is itself pinned.
static void check_accumulation_order(void) {
    using dlss5_nr::swin_linear;
    constexpr int kK = 130;
    float x[kK];
    float w[kK];
    for (int c = 0; c < 128; ++c) {
        x[c] = 256.0f;
        w[c] = 256.0f;
    }
    x[128] = 1.0f;
    x[129] = 1.0f;
    w[128] = 0.5f;
    w[129] = 0.5f;
    float out[1];
    MutableMatrixView o = make_out(out, 1, 1, 1);
    CHECK(swin_linear(make_view(x, 1, kK, kK), make_view(w, 1, kK, kK), &o, 1)
        == Status::kOk);
    CHECK(out[0] == 8388608.0f);
    CHECK(8388609.0f != 8388608.0f);
    // Reverse k-order float32 summation of the same products: the tail sums to 1
    // before the 2^23 block, and every intermediate is an exact integer below 2^24
    // -> 8388609.
    {
        float sum = 0.0f;
        for (int c = kK - 1; c >= 0; --c) {
            sum += x[c] * w[c];
        }
        CHECK(sum == 8388609.0f);
    }
    // One tail product first, the block, then the other tail product: 0.5 + 2^23
    // rounds to 2^23 at the recurring tie-to-even midpoint (all earlier partial sums
    // are exact multiples of 0.5 below 2^23), and the final 0.5 is dropped again
    // -> 8388608. This is the pinned counterexample to "any other grouping differs".
    {
        float sum = x[128] * w[128];
        for (int c = 0; c < 128; ++c) {
            sum += x[c] * w[c];
        }
        sum += x[129] * w[129];
        CHECK(sum == 8388608.0f);
    }
}

// ---- the e4m3 hidden ---------------------------------------------------------------------------------
// Hand-derived activations as exact rationals of the cubic constants
// (-229/4096, 229/512, 229/256):
//   product 1:  act = 229/256 + 1603/4096 = 5267/4096 = 1.285888671875 -> e4m3 1.25
//   product -1: act = -(229/256 - 1603/4096) = -2061/4096 = -0.503173828125
//               -> e4m3 -0.5
//   product 4:  act = 4 * 229/128 = 229/32 = 7.15625 -> e4m3 7.0
//   product 200704 (448*448): act = 200704 * 229/128 = 359072 -> e4m3 saturates 448
//   product 0:  act = 0 -> e4m3 0
static void check_hidden_known_answers(void) {
    using dlss5_nr::swin_hidden;
    {
        const float x[1] = {1.0f};
        const float w[1] = {1.0f};
        float out[1];
        MutableMatrixView o = make_out(out, 1, 1, 1);
        CHECK(swin_hidden(make_view(x, 1, 1, 1), make_view(w, 1, 1, 1), &o, 1)
            == Status::kOk);
        CHECK(out[0] == 1.25f);
    }
    {
        const float x[1] = {-1.0f};
        const float w[1] = {1.0f};
        float out[1];
        MutableMatrixView o = make_out(out, 1, 1, 1);
        CHECK(swin_hidden(make_view(x, 1, 1, 1), make_view(w, 1, 1, 1), &o, 1)
            == Status::kOk);
        CHECK(out[0] == -0.5f);
    }
    {
        const float x[1] = {4.0f};
        const float w[1] = {1.0f};
        float out[1];
        MutableMatrixView o = make_out(out, 1, 1, 1);
        CHECK(swin_hidden(make_view(x, 1, 1, 1), make_view(w, 1, 1, 1), &o, 1)
            == Status::kOk);
        CHECK(out[0] == 7.0f);
    }
    {
        const float x[1] = {448.0f};
        const float w[1] = {448.0f};
        float out[1];
        MutableMatrixView o = make_out(out, 1, 1, 1);
        CHECK(swin_hidden(make_view(x, 1, 1, 1), make_view(w, 1, 1, 1), &o, 1)
            == Status::kOk);
        CHECK(out[0] == 448.0f);
    }
    {
        const float x[1] = {0.0f};
        const float w[1] = {5.0f};
        float out[1];
        MutableMatrixView o = make_out(out, 1, 1, 1);
        CHECK(swin_hidden(make_view(x, 1, 1, 1), make_view(w, 1, 1, 1), &o, 1)
            == Status::kOk);
        CHECK(out[0] == 0.0f);
    }
}

// ---- permutation and sensitivity controls -----------------------------------------------------------
// Small dyadic corpus with pairwise-distinct weight rows and sensitivity deltas
// (guarded below so these controls cannot become vacuous): all sums are exact, so
// K-permutation is bit-exact too.
static void check_permutation_and_sensitivity(void) {
    using dlss5_nr::swin_linear;
    constexpr int kRows = 3;
    constexpr int kK = 5;
    constexpr int kN = 4;
    float x[kRows * kK];
    float w[kN * kK];
    float base[kRows * kN];
    float other[kRows * kN];
    for (int r = 0; r < kRows; ++r) {
        for (int c = 0; c < kK; ++c) {
            x[r * kK + c] = (float)((r * kK + c) % 7 - 3) * 0.25f;
        }
    }
    // Weight rows are pairwise distinct (the modulus 7 is not a multiple of kK, so
    // the wrap does not repeat rows), and the sensitivity deltas w[n][kK - 2] are
    // pairwise distinct and non-zero.
    for (int n = 0; n < kN; ++n) {
        for (int c = 0; c < kK; ++c) {
            w[n * kK + c] = (float)((n * kK + c) % 7 - 2) * 0.5f;
        }
    }
    for (int a = 0; a < kN; ++a) {
        for (int b = a + 1; b < kN; ++b) {
            CHECK(memcmp(&w[a * kK], &w[b * kK], kK * sizeof(float)) != 0);
        }
    }
    for (int n = 0; n < kN; ++n) {
        CHECK(w[n * kK + kK - 2] != 0.0f);
    }
    for (int a = 0; a < kN; ++a) {
        for (int b = a + 1; b < kN; ++b) {
            CHECK(w[a * kK + kK - 2] != w[b * kK + kK - 2]);
        }
    }
    // x rows are pairwise distinct (the row-permutation control cannot be vacuous).
    for (int a = 0; a < kRows; ++a) {
        for (int b = a + 1; b < kRows; ++b) {
            CHECK(memcmp(&x[a * kK], &x[b * kK], kK * sizeof(float)) != 0);
        }
    }
    MutableMatrixView o_base = make_out(base, kRows, kN, kN);
    CHECK(swin_linear(make_view(x, kRows, kK, kK), make_view(w, kN, kK, kK), &o_base,
            kRows * kN) == Status::kOk);
    // Hand-derived anchor row 0 of base: x[0] = (-3..1)/4, w[n] as computed above:
    //   out[0] = 0.75 + 0.25 + 0 + 0 + 0.25 = 1.25
    //   out[1] = -1.125 - 1 + 0.25 = -1.875
    //   out[2] = -0.375 - 0.5 - 0.375 - 0.25 = -1.5
    //   out[3] = 0.375 - 0.125 + 0.375 = 0.625
    CHECK(base[0] == 1.25f && base[1] == -1.875f && base[2] == -1.5f
        && base[3] == 0.625f);
    // Row permutation of x permutes output rows identically (bit-exact).
    float x_perm[kRows * kK];
    for (int r = 0; r < kRows; ++r) {
        const int src = (r + 1) % kRows;
        memcpy(&x_perm[r * kK], &x[src * kK], kK * sizeof(float));
    }
    MutableMatrixView o_other = make_out(other, kRows, kN, kN);
    CHECK(swin_linear(make_view(x_perm, kRows, kK, kK), make_view(w, kN, kK, kK),
            &o_other, kRows * kN) == Status::kOk);
    for (int r = 0; r < kRows; ++r) {
        const int src = (r + 1) % kRows;
        CHECK(memcmp(&other[r * kN], &base[src * kN], kN * sizeof(float)) == 0);
    }
    // Weight-row (= output channel) permutation permutes output columns identically.
    float w_perm[kN * kK];
    for (int n = 0; n < kN; ++n) {
        const int src = (n + 1) % kN;
        memcpy(&w_perm[n * kK], &w[src * kK], kK * sizeof(float));
    }
    CHECK(swin_linear(make_view(x, kRows, kK, kK), make_view(w_perm, kN, kK, kK),
            &o_other, kRows * kN) == Status::kOk);
    for (int r = 0; r < kRows; ++r) {
        for (int n = 0; n < kN; ++n) {
            const int src = (n + 1) % kN;
            CHECK(memcmp(&other[r * kN + n], &base[r * kN + src], sizeof(float)) == 0);
        }
    }
    // K-permutation applied to both operands leaves the exact sums bit-identical.
    float x_k[kRows * kK];
    float w_k[kN * kK];
    for (int r = 0; r < kRows; ++r) {
        for (int c = 0; c < kK; ++c) {
            x_k[r * kK + c] = x[r * kK + (kK - 1 - c)];
        }
    }
    for (int n = 0; n < kN; ++n) {
        for (int c = 0; c < kK; ++c) {
            w_k[n * kK + c] = w[n * kK + (kK - 1 - c)];
        }
    }
    CHECK(swin_linear(make_view(x_k, kRows, kK, kK), make_view(w_k, kN, kK, kK),
            &o_other, kRows * kN) == Status::kOk);
    CHECK(memcmp(other, base, sizeof base) == 0);
    // Changed-input sensitivity with an exact delta: +1.0 at x_k[0][1] moves
    // out[0][n] by the paired weight w_k[n][1] (pairwise-distinct and non-zero by
    // the corpus guards above, so the channel pairing is pinned) and leaves every
    // other row untouched.
    x_k[1] += 1.0f;
    CHECK(swin_linear(make_view(x_k, kRows, kK, kK), make_view(w_k, kN, kK, kK),
            &o_other, kRows * kN) == Status::kOk);
    for (int n = 0; n < kN; ++n) {
        CHECK(other[n] == base[n] + w_k[n * kK + 1]);
    }
    CHECK(memcmp(&other[kN], &base[kN], (kRows - 1) * kN * sizeof(float)) == 0);
    // Determinism: identical inputs, bit-identical output.
    float again[kRows * kN];
    MutableMatrixView o_again = make_out(again, kRows, kN, kN);
    CHECK(swin_linear(make_view(x_k, kRows, kK, kK), make_view(w_k, kN, kK, kK),
            &o_again, kRows * kN) == Status::kOk);
    CHECK(memcmp(again, other, sizeof again) == 0);
}

// ---- negative controls --------------------------------------------------------------------------------
static void check_negative_linear(void) {
    using dlss5_nr::swin_linear;
    constexpr int kRows = 2;
    constexpr int kK = 3;
    constexpr int kN = 2;
    float x[kRows * kK];
    float w[kN * kK];
    float out[kRows * kN];
    for (int p = 0; p < kRows * kK; ++p) {
        x[p] = 1.0f;
    }
    for (int p = 0; p < kN * kK; ++p) {
        w[p] = 0.5f;
    }
    fill_sentinel(out, kRows * kN);
    const std::size_t capacity = kRows * kN;
    const MatrixView xv = make_view(x, kRows, kK, kK);
    const MatrixView wv = make_view(w, kN, kK, kK);
    MutableMatrixView o = make_out(out, kRows, kN, kN);
    // Null pointers.
    CHECK(swin_linear(make_view(nullptr, kRows, kK, kK), wv, &o, capacity)
        == Status::kNullPointer);
    CHECK(swin_linear(xv, make_view(nullptr, kN, kK, kK), &o, capacity)
        == Status::kNullPointer);
    CHECK(swin_linear(xv, wv, nullptr, capacity) == Status::kNullPointer);
    MutableMatrixView no_out = o;
    no_out.data = nullptr;
    CHECK(swin_linear(xv, wv, &no_out, capacity) == Status::kNullPointer);
    CHECK(all_sentinel(out, capacity));
    // Dimension classes.
    CHECK(swin_linear(make_view(x, 0, kK, kK), wv, &o, capacity)
        == Status::kInvalidDimensions);
    CHECK(swin_linear(make_view(x, kRows, 0, kK), wv, &o, capacity)
        == Status::kInvalidDimensions);
    CHECK(swin_linear(make_view(x, kRows, dlss5_nr::kMaxLinearDim + 1, kK),
            make_view(w, kN, dlss5_nr::kMaxLinearDim + 1, kK + 1), &o, capacity)
        == Status::kInvalidDimensions);
    CHECK(swin_linear(make_view(x, kRows, kK, kK), make_view(w, kN, kK + 1, kK + 1),
            &o, capacity) == Status::kInvalidDimensions);
    CHECK(swin_linear(make_view(x, dlss5_nr::kMaxLinearRows + 1, kK, kK), wv, &o,
            capacity) == Status::kInvalidDimensions);
    // K mismatch and output row mismatch.
    CHECK(swin_linear(xv, make_view(w, kN, kK + 1, kK + 1), &o, capacity)
        == Status::kInvalidDimensions);
    // The remaining reachable rejection branches of the operand/output validation:
    // a zero-K pair (both operands with cols 0 passes the K-match pre-check and is
    // rejected by the operand cols<1 bound) and a weight with kMaxLinearDim + 1
    // rows (rejected by the output cols>bound; the operand buffer is genuinely
    // sized so its grid scan stays in bounds).
    CHECK(swin_linear(make_view(x, kRows, 0, kK), make_view(w, kN, 0, kK + 1), &o,
            capacity) == Status::kInvalidDimensions);
    {
        float big_w[dlss5_nr::kMaxLinearDim + 1];
        for (int p = 0; p < dlss5_nr::kMaxLinearDim + 1; ++p) {
            big_w[p] = 1.0f;
        }
        CHECK(swin_linear(make_view(x, kRows, 1, 1),
                make_view(big_w, dlss5_nr::kMaxLinearDim + 1, 1, 1), &o, capacity)
            == Status::kInvalidDimensions);
    }
    // Every rejection above (rows 0, cols 0, both over-dimension bounds, the weight
    // cols mismatch, the K mismatch, the zero-K pair and the over-wide weight) must
    // leave the output untouched; checked here before the valid write below
    // overwrites it.
    CHECK(all_sentinel(out, capacity));
    CHECK(swin_linear(xv, wv, &o, capacity) == Status::kOk);
    fill_sentinel(out, capacity);
    MutableMatrixView wrong_rows = o;
    wrong_rows.rows = kRows + 1;
    CHECK(swin_linear(xv, wv, &wrong_rows, capacity) == Status::kInvalidDimensions);
    CHECK(all_sentinel(out, capacity));
    // Stride classes.
    CHECK(swin_linear(make_view(x, kRows, kK, kK - 1), wv, &o, capacity)
        == Status::kInvalidStride);
    CHECK(swin_linear(xv, make_view(w, kN, kK, kK - 1), &o, capacity)
        == Status::kInvalidStride);
    CHECK(swin_linear(make_view(x, kRows, kK, dlss5_nr::kMaxLinearDim + 1), wv, &o,
            capacity) == Status::kInvalidStride);
    MutableMatrixView bad_out = o;
    bad_out.row_stride = kN - 1;
    CHECK(swin_linear(xv, wv, &bad_out, capacity) == Status::kInvalidStride);
    bad_out = o;
    bad_out.row_stride = dlss5_nr::kMaxLinearDim + 1;
    CHECK(swin_linear(xv, wv, &bad_out, capacity) == Status::kInvalidStride);
    CHECK(all_sentinel(out, capacity));
    // Short capacity (one float short of rows * n).
    CHECK(swin_linear(xv, wv, &o, capacity - 1) == Status::kInvalidBuffer);
    CHECK(all_sentinel(out, capacity));
    // e4m3 operand-grid class: off-grid, out-of-range and non-finite operands in
    // both the x and the weight side.
    const float bad_values[] = {0.3f, 1.0625f, 0x1p-10f, 500.0f, -500.0f,
        (float)NAN, (float)INFINITY, (float)-INFINITY};
    for (std::size_t i = 0; i < sizeof bad_values / sizeof bad_values[0]; ++i) {
        x[0] = bad_values[i];
        CHECK(swin_linear(xv, wv, &o, capacity) == Status::kInvalidValues);
        x[0] = 1.0f;
        w[0] = bad_values[i];
        CHECK(swin_linear(xv, wv, &o, capacity) == Status::kInvalidValues);
        w[0] = 0.5f;
    }
    CHECK(all_sentinel(out, capacity));
    // A valid boundary operand class still succeeds: +-448 and the subnormal grid.
    x[0] = 448.0f;
    x[1] = -448.0f;
    x[2] = 0x1p-9f;
    w[0] = -0.0f;
    CHECK(swin_linear(xv, wv, &o, capacity) == Status::kOk);
}

static void check_negative_hidden(void) {
    using dlss5_nr::swin_hidden;
    constexpr int kRows = 2;
    constexpr int kK = 3;
    constexpr int kN = 4;
    float x[kRows * kK];
    float w[kN * kK];
    float out[kRows * kN];
    for (int p = 0; p < kRows * kK; ++p) {
        x[p] = 1.0f;
    }
    for (int p = 0; p < kN * kK; ++p) {
        w[p] = 0.5f;
    }
    fill_sentinel(out, kRows * kN);
    const std::size_t capacity = kRows * kN;
    const MatrixView xv = make_view(x, kRows, kK, kK);
    const MatrixView wv = make_view(w, kN, kK, kK);
    MutableMatrixView o = make_out(out, kRows, kN, kN);
    // swin_hidden delegates all validation to swin_linear; every rejection class of
    // the contract is exercised through this entry point too, each group followed by
    // a sentinel-checked untouched-output assertion.
    // Null pointers.
    CHECK(swin_hidden(make_view(nullptr, kRows, kK, kK), wv, &o, capacity)
        == Status::kNullPointer);
    CHECK(swin_hidden(xv, make_view(nullptr, kN, kK, kK), &o, capacity)
        == Status::kNullPointer);
    CHECK(swin_hidden(xv, wv, nullptr, capacity) == Status::kNullPointer);
    MutableMatrixView no_out = o;
    no_out.data = nullptr;
    CHECK(swin_hidden(xv, wv, &no_out, capacity) == Status::kNullPointer);
    CHECK(all_sentinel(out, capacity));
    // Dimension classes.
    CHECK(swin_hidden(make_view(x, 0, kK, kK), wv, &o, capacity)
        == Status::kInvalidDimensions);
    CHECK(swin_hidden(make_view(x, kRows, 0, kK), wv, &o, capacity)
        == Status::kInvalidDimensions);
    CHECK(swin_hidden(make_view(x, kRows, dlss5_nr::kMaxLinearDim + 1, kK),
            make_view(w, kN, dlss5_nr::kMaxLinearDim + 1, kK + 1), &o, capacity)
        == Status::kInvalidDimensions);
    CHECK(swin_hidden(make_view(x, kRows, kK, kK), make_view(w, kN, kK + 1, kK + 1),
            &o, capacity) == Status::kInvalidDimensions);
    CHECK(swin_hidden(make_view(x, dlss5_nr::kMaxLinearRows + 1, kK, kK), wv, &o,
            capacity) == Status::kInvalidDimensions);
    CHECK(swin_hidden(xv, make_view(w, kN, kK + 1, kK + 1), &o, capacity)
        == Status::kInvalidDimensions);
    // The remaining reachable rejection branches: a zero-K pair and a weight with
    // kMaxLinearDim + 1 rows (genuinely sized operand buffer, output cols>bound).
    CHECK(swin_hidden(make_view(x, kRows, 0, kK), make_view(w, kN, 0, kK + 1), &o,
            capacity) == Status::kInvalidDimensions);
    {
        float big_w[dlss5_nr::kMaxLinearDim + 1];
        for (int p = 0; p < dlss5_nr::kMaxLinearDim + 1; ++p) {
            big_w[p] = 1.0f;
        }
        CHECK(swin_hidden(make_view(x, kRows, 1, 1),
                make_view(big_w, dlss5_nr::kMaxLinearDim + 1, 1, 1), &o, capacity)
            == Status::kInvalidDimensions);
    }
    CHECK(all_sentinel(out, capacity));
    // The fixture is valid; the output row-match rejection is checked around a
    // real write so a write-before-return in it would be detected.
    CHECK(swin_hidden(xv, wv, &o, capacity) == Status::kOk);
    fill_sentinel(out, capacity);
    MutableMatrixView wrong_rows = o;
    wrong_rows.rows = kRows + 1;
    CHECK(swin_hidden(xv, wv, &wrong_rows, capacity) == Status::kInvalidDimensions);
    CHECK(all_sentinel(out, capacity));
    // Stride classes.
    CHECK(swin_hidden(make_view(x, kRows, kK, kK - 1), wv, &o, capacity)
        == Status::kInvalidStride);
    CHECK(swin_hidden(xv, make_view(w, kN, kK, kK - 1), &o, capacity)
        == Status::kInvalidStride);
    CHECK(swin_hidden(make_view(x, kRows, kK, dlss5_nr::kMaxLinearDim + 1), wv, &o,
            capacity) == Status::kInvalidStride);
    MutableMatrixView bad_out = o;
    bad_out.row_stride = kN - 1;
    CHECK(swin_hidden(xv, wv, &bad_out, capacity) == Status::kInvalidStride);
    bad_out = o;
    bad_out.row_stride = dlss5_nr::kMaxLinearDim + 1;
    CHECK(swin_hidden(xv, wv, &bad_out, capacity) == Status::kInvalidStride);
    CHECK(all_sentinel(out, capacity));
    // Short capacity (one float short of rows * n).
    CHECK(swin_hidden(xv, wv, &o, capacity - 1) == Status::kInvalidBuffer);
    CHECK(all_sentinel(out, capacity));
    // e4m3 operand-grid class: off-grid, out-of-range and non-finite operands in
    // both the x and the weight side.
    const float bad_values[] = {0.3f, 1.0625f, 0x1p-10f, 500.0f, -500.0f,
        (float)NAN, (float)INFINITY, (float)-INFINITY};
    for (std::size_t i = 0; i < sizeof bad_values / sizeof bad_values[0]; ++i) {
        x[0] = bad_values[i];
        CHECK(swin_hidden(xv, wv, &o, capacity) == Status::kInvalidValues);
        x[0] = 1.0f;
        w[0] = bad_values[i];
        CHECK(swin_hidden(xv, wv, &o, capacity) == Status::kInvalidValues);
        w[0] = 0.5f;
    }
    CHECK(all_sentinel(out, capacity));
    // A valid boundary operand class still succeeds through this entry point:
    // +-448 and the subnormal grid.
    x[0] = 448.0f;
    x[1] = -448.0f;
    x[2] = 0x1p-9f;
    w[0] = -0.0f;
    CHECK(swin_hidden(xv, wv, &o, capacity) == Status::kOk);
}

// ---- allocation discipline --------------------------------------------------------------------------
static void check_allocation_discipline(void) {
    using dlss5_nr::swin_hidden;
    using dlss5_nr::swin_linear;
    const unsigned long long before = g_allocations.load();
    constexpr int kRows = 4;
    constexpr int kK = 6;
    constexpr int kN = 5;
    float x[kRows * kK];
    float w[kN * kK];
    float out[kRows * kN];
    for (int p = 0; p < kRows * kK; ++p) {
        x[p] = (float)(p % 9) * 0.5f - 2.0f;
    }
    for (int p = 0; p < kN * kK; ++p) {
        w[p] = (float)(p % 7) * 0.25f - 1.0f;
    }
    const MatrixView xv = make_view(x, kRows, kK, kK);
    const MatrixView wv = make_view(w, kN, kK, kK);
    MutableMatrixView o = make_out(out, kRows, kN, kN);
    for (int i = 0; i < 64; ++i) {
        CHECK(swin_linear(xv, wv, &o, kRows * kN) == Status::kOk);
        CHECK(swin_hidden(xv, wv, &o, kRows * kN) == Status::kOk);
    }
    CHECK(g_allocations.load() == before);
}

int main(void) {
    check_known_answers();
    check_accumulation_order();
    check_hidden_known_answers();
    check_permutation_and_sensitivity();
    check_negative_linear();
    check_negative_hidden();
    check_allocation_discipline();
    printf("native-swin-linear-test: %llu checks passed\n", g_checks);
    return 0;
}
