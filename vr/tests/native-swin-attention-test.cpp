// Independent tests for the native Swin windowed attention
// (vr/native-swin-attention.hpp): the position-bias deswizzle index and batch
// copy, the exponent-ramp bias, the exponent-ramp softmax numerators, the f16
// denominator tree, the softmax rows, the QKV normalisation with the head scale,
// the attention logits and context einsums and the assembled swin_attention.
// CPU-only, no GPU, headset, proprietary payload, Python or PyTorch runtime
// required.
//
// Independence: every known answer is a hand-derived dyadic rational (the exact
// decompositions are written next to each case). Controls cover: deswizzle
// anchors plus an exhaustive 4096-slot bijection and full decode mapping; the
// exp-bias ramp anchors plus an exhaustive sweep of all 63 488 finite f16 raw
// patterns pinning the single-rounding fma against a two-rounding re-derivation
// (they agree everywhere over f16 raw inputs and c = EXP_OFFSET); the numerator
// mantissa-to-exponent bit transplant (interior pins, both clamp boundaries,
// two ramp ties-to-even) and a fma-vs-two-roundings discriminator that survives
// the f16 ramp rounding (0.00262451171875 where the two-rounding path gives
// 0.002685546875); the denominator sum-order discriminators (block, pair,
// member/half nesting orders, 2050 vs 2048 and 2048 vs 2050), f16 overflow
// passthrough and the SUM_FLOOR clamp bound (the clamp provably never binds on a
// 64-key row: the smallest denominator is 64 * 2^-14 = 2^-8); the
// head_scale/norm_scale grouping discriminator (0.025390625 where per-dimension
// regrouping gives 0.0234375) and the q-vs-v rounding asymmetry (367.875 scales
// to 352 on the q path and stores to 384 on the v path); the logit accumulation
// order discriminator (200704 where a smalls-first order gives 200704.125); the
// store double rounding through the context (367.875 -> 384); permutation and
// changed-input sensitivity controls; bit-exact equivalence of every batch entry
// against its row/staged chain and of swin_attention against the full staged
// chain on a deterministic pseudorandom corpus with padded strides;
// sentinel-checked untouched outputs (and untouched scratch in the assembly) on
// every validation rejection class; determinism and zero-allocation accounting.
// See vr/NATIVE-SWIN-ATTENTION.md.

#include "native-swin-attention.hpp"

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

using dlss5_nr::AttentionTensorView;
using dlss5_nr::MutableAttentionTensorView;

static AttentionTensorView make_tensor(const float* data, int windows, int heads,
                                       std::ptrdiff_t window_stride,
                                       std::ptrdiff_t head_stride) {
    AttentionTensorView t;
    t.data = data;
    t.windows = windows;
    t.heads = heads;
    t.window_stride = window_stride;
    t.head_stride = head_stride;
    return t;
}

static MutableAttentionTensorView make_mutable_tensor(float* data, int windows,
                                                      int heads,
                                                      std::ptrdiff_t window_stride,
                                                      std::ptrdiff_t head_stride) {
    MutableAttentionTensorView t;
    t.data = data;
    t.windows = windows;
    t.heads = heads;
    t.window_stride = window_stride;
    t.head_stride = head_stride;
    return t;
}

// ---- the deswizzle index ---------------------------------------------------------------------------
// Anchors from the pinned definition
//   lane  = 4 * ((i % 16) % 8) + ((j % 16) % 8) / 2
//   slot  = (j % 2) | (((i % 16) / 8) << 1) | (((j % 16) / 8) << 2)
//   index = (4 * (i / 16) + (j / 16)) * 256 + lane * 8 + slot
// hand-evaluated: (0,0)->0; (0,1)->1; (1,0)->32; (8,0)->2; (0,8)->4;
// (4,2)->136; (15,15)->255; (0,16)->256; (16,0)->1024; (63,63)->4095.
static void check_deswizzle_index_pins(void) {
    using dlss5_nr::swin_deswizzle_index;
    CHECK(swin_deswizzle_index(-1, 0) == -1);
    CHECK(swin_deswizzle_index(0, -1) == -1);
    CHECK(swin_deswizzle_index(64, 0) == -1);
    CHECK(swin_deswizzle_index(0, 64) == -1);
    CHECK(swin_deswizzle_index(63, 64) == -1);
    CHECK(swin_deswizzle_index(64, 64) == -1);
    CHECK(swin_deswizzle_index(0, 0) == 0);
    CHECK(swin_deswizzle_index(0, 1) == 1);
    CHECK(swin_deswizzle_index(1, 0) == 32);
    CHECK(swin_deswizzle_index(8, 0) == 2);
    CHECK(swin_deswizzle_index(0, 8) == 4);
    CHECK(swin_deswizzle_index(4, 2) == 136);
    CHECK(swin_deswizzle_index(15, 15) == 255);
    CHECK(swin_deswizzle_index(0, 16) == 256);
    CHECK(swin_deswizzle_index(16, 0) == 1024);
    CHECK(swin_deswizzle_index(63, 63) == 4095);
    // Exhaustive: the 4096 (i, j) slots map onto [0, 4096) exactly once, and the
    // decode formula above (re-derived here from the pinned definition) agrees.
    bool seen[dlss5_nr::kBiasTable];
    memset(seen, 0, sizeof seen);
    for (int i = 0; i < dlss5_nr::kWindowTokens; ++i) {
        for (int j = 0; j < dlss5_nr::kWindowTokens; ++j) {
            const int index = swin_deswizzle_index(i, j);
            CHECK(index >= 0 && index < dlss5_nr::kBiasTable);
            CHECK(!seen[index]);
            seen[index] = true;
            const int lane = 4 * ((i % 16) % 8) + ((j % 16) % 8) / 2;
            const int slot = (j % 2) | (((i % 16) / 8) << 1) | (((j % 16) / 8) << 2);
            const int decoded = (4 * (i / 16) + (j / 16)) * 256 + lane * 8 + slot;
            CHECK(index == decoded);
        }
    }
    for (int p = 0; p < dlss5_nr::kBiasTable; ++p) {
        CHECK(seen[p]);
    }
}

// ---- the batch deswizzle ----------------------------------------------------------------------------
// raw[p] = ((p * 37) % 256) * 0.125 (every value exactly f16-representable),
// out[i * 64 + j] = raw[deswizzle(i, j)] for both heads; padded strides whose
// padding is never written; a one-entry raw change moves exactly one output.
static void check_deswizzle_bias_pins(void) {
    using dlss5_nr::is_f16;
    using dlss5_nr::swin_deswizzle_bias;
    using dlss5_nr::swin_deswizzle_index;
    constexpr int kHeads = 2;
    constexpr std::ptrdiff_t kRowStride = dlss5_nr::kBiasTable + 2;
    static float raw[kHeads * kRowStride];
    static float out[kHeads * kRowStride];
    for (int h = 0; h < kHeads; ++h) {
        for (int p = 0; p < dlss5_nr::kBiasTable; ++p) {
            raw[h * kRowStride + p] =
                (float)(((p * 37 + h * 11) % 256)) * 0.125f;
            CHECK(is_f16(raw[h * kRowStride + p]));
        }
        raw[h * kRowStride + dlss5_nr::kBiasTable] = kSentinel;
        raw[h * kRowStride + dlss5_nr::kBiasTable + 1] = kSentinel;
    }
    fill_sentinel(out, kHeads * kRowStride);
    const MatrixView raw_v = make_view(raw, kHeads, dlss5_nr::kBiasTable, kRowStride);
    MutableMatrixView out_v = make_out(out, kHeads, dlss5_nr::kBiasTable, kRowStride);
    CHECK(swin_deswizzle_bias(raw_v, &out_v, kHeads * kRowStride) == Status::kOk);
    for (int h = 0; h < kHeads; ++h) {
        for (int i = 0; i < dlss5_nr::kWindowTokens; ++i) {
            for (int j = 0; j < dlss5_nr::kWindowTokens; ++j) {
                const int index = swin_deswizzle_index(i, j);
                CHECK(out[h * kRowStride + i * dlss5_nr::kWindowTokens + j]
                    == raw[h * kRowStride + index]);
            }
        }
        CHECK(out[h * kRowStride + dlss5_nr::kBiasTable] == kSentinel);
        CHECK(out[h * kRowStride + dlss5_nr::kBiasTable + 1] == kSentinel);
    }
    // Changed-input sensitivity: one raw entry changed moves exactly one output
    // cell, at the unique (i, j) whose deswizzle index reads that entry; every
    // other cell (including the other head and the padding) is bit-identical.
    static float snapshot[kHeads * kRowStride];
    memcpy(snapshot, out, sizeof snapshot);
    const int src = 1234;
    const float before = raw[src];
    raw[src] = before + 0.125f;  // still an f16 grid value
    CHECK(swin_deswizzle_bias(raw_v, &out_v, kHeads * kRowStride) == Status::kOk);
    int moved = 0;
    int moved_to = -1;
    for (std::size_t p = 0; p < kHeads * kRowStride; ++p) {
        if (memcmp(&out[p], &snapshot[p], sizeof(float)) != 0) {
            ++moved;
            moved_to = (int)p;
        }
    }
    CHECK(moved == 1);
    CHECK(moved_to >= 0 && moved_to < dlss5_nr::kBiasTable
        && swin_deswizzle_index(moved_to / dlss5_nr::kWindowTokens,
                                moved_to % dlss5_nr::kWindowTokens) == src);
    raw[src] = before;
}

// ---- the exponent-ramp bias ---------------------------------------------------------------------------
// Anchors (all exact dyadic): raw 0 -> 1.30078125; raw 1 -> 1.345703125;
// raw -29 -> -0.001953125; raw 0.5 -> 1.3232421875. Then the exhaustive f16
// sweep: for every finite f16 raw pattern the single-rounding fma
// (double product + double add, rounded once) and the two-rounding path agree,
// and swin_exp_bias reproduces the value at the deswizzled position.
static void check_exp_bias_pins(void) {
    using dlss5_nr::swin_deswizzle_index;
    using dlss5_nr::swin_exp_bias;
    constexpr std::ptrdiff_t kRowStride = dlss5_nr::kBiasTable + 2;
    static float raw[kRowStride];
    static float out[kRowStride];
    memset(raw, 0, sizeof raw);
    raw[0] = 1.0f;
    raw[1] = -29.0f;
    raw[2] = 0.5f;
    fill_sentinel(out, kRowStride);
    const MatrixView raw_v = make_view(raw, 1, dlss5_nr::kBiasTable, kRowStride);
    MutableMatrixView out_v = make_out(out, 1, dlss5_nr::kBiasTable, kRowStride);
    CHECK(swin_exp_bias(raw_v, &out_v, kRowStride) == Status::kOk);
    // (i, j) = (0, 0) reads raw[0] = 1 -> 23/512 + 1 + 77/256 = 1.345703125.
    CHECK(out[0] == 1.345703125f);
    // (i, j) = (0, 1) reads raw[1] = -29 -> 1.30078125 - 667/512 = -0.001953125.
    CHECK(out[1] == -0.001953125f);
    // (i, j) = (1, 0) reads raw[32] = 0 -> EXP_OFFSET.
    CHECK(out[dlss5_nr::kWindowTokens] == 1.30078125f);
    // (i, j) = (8, 0) reads raw[2] = 0.5 -> 1.30078125 + 23/1024 = 1.3232421875.
    CHECK(out[8 * dlss5_nr::kWindowTokens] == 1.3232421875f);
    for (int p = 0; p < dlss5_nr::kBiasTable; ++p) {
        const int index = swin_deswizzle_index(p / dlss5_nr::kWindowTokens,
            p % dlss5_nr::kWindowTokens);
        const float expected =
            (float)((double)raw[index] * (double)dlss5_nr::kExpSlope
                + (double)dlss5_nr::kExpOffset);
        CHECK(out[p] == expected);
    }
    CHECK(out[kRowStride - 2] == kSentinel);
    CHECK(out[kRowStride - 1] == kSentinel);
    // Exhaustive sweep over the 16-bit f16 pattern space in 4096-pattern chunks:
    // every finite pattern lands in the table (non-finite patterns are filled as
    // 0 and skipped), the fma and two-rounding paths agree, and the entry is
    // reproduced at the deswizzled position.
    for (int chunk = 0; chunk < 16; ++chunk) {
        for (int p = 0; p < dlss5_nr::kBiasTable; ++p) {
            const int pattern = chunk * dlss5_nr::kBiasTable + p;
            if ((pattern & 0x7C00) == 0x7C00) {
                raw[p] = 0.0f;
            } else {
                raw[p] = dlss5_nr::f16_from_bits((std::uint16_t)pattern);
            }
        }
        CHECK(swin_exp_bias(raw_v, &out_v, kRowStride) == Status::kOk);
        for (int i = 0; i < dlss5_nr::kWindowTokens; ++i) {
            for (int j = 0; j < dlss5_nr::kWindowTokens; ++j) {
                const int index = swin_deswizzle_index(i, j);
                const float a = raw[index];
                const float fma_side =
                    (float)((double)a * (double)dlss5_nr::kExpSlope
                        + (double)dlss5_nr::kExpOffset);
                const float product =
                    (float)((double)a * (double)dlss5_nr::kExpSlope);
                const float two_side =
                    (float)((double)product + (double)dlss5_nr::kExpOffset);
                CHECK(out[i * dlss5_nr::kWindowTokens + j] == fma_side);
                CHECK(fma_side == two_side);
            }
        }
    }
}

// ---- the exponent-ramp softmax numerator ------------------------------------------------------------
// The ramp is swin_fma(logit, EXP_SLOPE, bias) narrowed to f16 (ties to even)
// and clamped to [EXP_LOW, EXP_HIGH]; the clamp forces [1, 2), and the mantissa
// bits are transplanted into the exponent field: (m >> 5) - 15 with fraction
// (m & 31) / 32. Every expected value below is hand-derived from that rule.
static void check_numerator_pins(void) {
    using dlss5_nr::swin_softmax_numerator;
    // logit 0, bias 1.30078125: ramp mantissa 308 -> 2^-6 * 1.625.
    CHECK(swin_softmax_numerator(0.0f, 1.30078125f) == 0.025390625f);
    // logit 0, bias 1.25: ramp mantissa 256 -> 2^(8-15) = 2^-7.
    CHECK(swin_softmax_numerator(0.0f, 1.25f) == 0.0078125f);
    // logit 0, bias 1.5: ramp mantissa 512 -> 2^(16-15) = 2.
    CHECK(swin_softmax_numerator(0.0f, 1.5f) == 2.0f);
    // logit 0, bias 1.03125: ramp mantissa 32 (the low clamp boundary kept
    // exact) -> 2^(1-15) = 2^-14.
    CHECK(swin_softmax_numerator(0.0f, 1.03125f) == 6.103515625e-05f);
    // logit 0, bias 1.5693359375: ramp mantissa 583 (the high clamp boundary)
    // -> 2^(18-15) * (1 + 224/1024) = 9.75.
    CHECK(swin_softmax_numerator(0.0f, 1.5693359375f) == 9.75f);
    // Ramp ties-to-even below and above: logit 0.25, bias 1.25 gives the exact
    // ramp 1 + 535/2048 -> one binary16 tie (535/2048 = 267.5/1024) -> mantissa
    // 268 (even) -> 2^(8-15) * 1.375 = 0.0107421875; logit 1, bias 1.30078125
    // gives 1 + 177/512 = 1 + 354/1024 exactly (no tie) -> mantissa 354
    // -> 2^(11-15) * (1 + 64/1024) = 0.06640625.
    CHECK(swin_softmax_numerator(0.25f, 1.25f) == 0.0107421875f);
    CHECK(swin_softmax_numerator(1.0f, 1.30078125f) == 0.06640625f);
    // The clamp is on the ramp: logit 0.25, bias 1.0 gives the exact ramp
    // 1 + 23/2048 (mantissa 12 after the tie) which is BELOW EXP_LOW -> 2^-14.
    CHECK(swin_softmax_numerator(0.25f, 1.0f) == 6.103515625e-05f);
    CHECK(swin_softmax_numerator(-1000.0f, 1.30078125f) == 6.103515625e-05f);
    CHECK(swin_softmax_numerator(1000.0f, 1.30078125f) == 9.75f);
    // A ramp whose binary16 narrowing overflows gives +-inf, which saturates at
    // the clamp exactly as torch.clamp on inf (the float32 fma result itself
    // stays finite here; only the narrowing overflows).
    CHECK(swin_softmax_numerator(1e30f, 0.0f) == 9.75f);
    CHECK(swin_softmax_numerator(-1e30f, 0.0f) == 6.103515625e-05f);
    CHECK(swin_softmax_numerator(0.0f, 0.0f) == 6.103515625e-05f);
    // The fma-vs-two-roundings discriminator that survives the f16 ramp
    // rounding. logit 0x4168611e (14.523710250854492), bias 0x3f0bda31
    // (0.546298086643218994140625): the exact double product is
    // 0.6524322964251041412353516 (23/512 * logit); the float32 product rounds
    // up to 0.65243232250213623046875 (0x3f2705ce), so the two-rounding sum
    // 1.19873046875 (0x3f997000) lands one f16 step above the single-rounding
    // fma sum 1.19873034954071044921875 (0x3f996fff): f16 0x3ccb (mantissa 203)
    // vs 0x3ccc (mantissa 204), transplanting to 2^-9 * 1.34375 =
    // 0.00262451171875 vs 2^-9 * 1.375 = 0.002685546875.
    const float logit = 14.523710250854492f;
    const float bias = 0.546298086643218994140625f;
    CHECK(swin_softmax_numerator(logit, bias) == 0.00262451171875f);
    const float product = (float)((double)logit * (double)dlss5_nr::kExpSlope);
    const float two_sum = (float)((double)product + (double)bias);
    CHECK(two_sum == 1.19873046875f);
    const float two_ramp = dlss5_nr::f16(two_sum);
    CHECK(dlss5_nr::f16_bits(two_ramp) == 0x3cccu);
    CHECK(dlss5_nr::f16_from_bits(
        (std::uint16_t)((dlss5_nr::f16_bits(two_ramp) & 0x3FFu) << 5))
        == 0.002685546875f);
    // A signed-zero logit leaves the bias ramp untouched: -0.0 * slope is -0.0
    // and -0.0 + bias rounds to bias exactly.
    CHECK(swin_softmax_numerator(-0.0f, 1.30078125f) == 0.025390625f);
}

// ---- the softmax denominator --------------------------------------------------------------------------
// The keys of a query are summed as block/pair/member/half
// (key = 16 block + 4 pair + 2 member + half): blocks ((0+1)+2)+3, then pairs
// ((0+1)+2)+3, then the two halves of a member, then the two members. Each add
// is a binary16 add. The cases below separate that exact nesting from the
// alternative orders named in the comments.
static void check_denominator_pins(void) {
    using dlss5_nr::swin_softmax_denominator_row;
    float keys[dlss5_nr::kWindowTokens];
    // Case A: k[0] = 2048, k[1] = k[2] = k[3] = 1. The members/halves stage
    // meets the pair stage at 2048 + 2 = 2050; a sequential left-to-right sum
    // loses the ones at the 2048 tie and gives 2048.
    memset(keys, 0, sizeof keys);
    keys[0] = 2048.0f;
    keys[1] = 1.0f;
    keys[2] = 1.0f;
    keys[3] = 1.0f;
    CHECK(swin_softmax_denominator_row(keys) == 2050.0f);
    // Case B: k[0] = 2048, k[2] = k[3] = 1 (k[2] and k[3] are the two halves
    // of member 1). Summing the halves of each member before the members gives
    // 2048 + 2 = 2050; a half-major nesting that adds the same half of both
    // members first ((k[0] + k[2]) + (k[1] + k[3])) gives (2048 + 1) + (0 + 1)
    // = 2048, every node rounding to binary16 where 2048 + 1 ties to 2048.
    memset(keys, 0, sizeof keys);
    keys[0] = 2048.0f;
    keys[2] = 1.0f;
    keys[3] = 1.0f;
    CHECK(swin_softmax_denominator_row(keys) == 2050.0f);
    // Case C: k[0] = 2048, k[8] = k[12] = 1 (pairs 0, 2, 3 of the same
    // member/half). The pinned pair order ((0+1)+2)+3 gives 2048; a
    // smalls-first pair order gives 2050.
    memset(keys, 0, sizeof keys);
    keys[0] = 2048.0f;
    keys[8] = 1.0f;
    keys[12] = 1.0f;
    CHECK(swin_softmax_denominator_row(keys) == 2048.0f);
    // Case D: k[32] = 2048 (block 2), k[0] = k[16] = 1 (blocks 0, 1) of one
    // (pair, member, half). The pinned block order ((0+1)+2)+3 gives
    // (1+1)+2048 = 2050; a descending block order gives 2048.
    memset(keys, 0, sizeof keys);
    keys[32] = 2048.0f;
    keys[0] = 1.0f;
    keys[16] = 1.0f;
    CHECK(swin_softmax_denominator_row(keys) == 2050.0f);
    // Exact uniform rows: 64 * 2^-14 = 2^-8 and 64 * 2^-6 = 1 (every partial
    // sum f16-exact).
    for (int j = 0; j < dlss5_nr::kWindowTokens; ++j) {
        keys[j] = 6.103515625e-05f;  // 2^-14
    }
    CHECK(swin_softmax_denominator_row(keys) == 0.00390625f);  // 2^-8
    for (int j = 0; j < dlss5_nr::kWindowTokens; ++j) {
        keys[j] = 0.015625f;  // 2^-6
    }
    CHECK(swin_softmax_denominator_row(keys) == 1.0f);
    // f16 overflow passes through as +-inf, exactly as the f16 add chain.
    for (int j = 0; j < dlss5_nr::kWindowTokens; ++j) {
        keys[j] = 65504.0f;
    }
    CHECK(swin_softmax_denominator_row(keys) == INFINITY);
    // SUM_FLOOR can never bind on a 64-key row: every numerator is at least the
    // smallest transplant 2^-14, so the denominator is at least 64 * 2^-14 =
    // 2^-8 > SUM_FLOOR. The clamp is kept for exactness with the reference.
    CHECK(dlss5_nr::kSumFloor == 0x1p-14f);
    CHECK(64.0f * 6.103515625e-05f == 0.00390625f);
    CHECK(0.00390625f > dlss5_nr::kSumFloor);
}

// ---- the softmax rows ---------------------------------------------------------------------------------
// Uniform rows are exact end to end: numerators 2^-6 (bias 1.28125), 2 (bias
// 1.5) and 2^-7 (bias 1.25) each give a power-of-two denominator (1, 128,
// 0.5), an exact reciprocal and a stored probability of 2^-6 per key.
static void check_softmax_row_pins(void) {
    using dlss5_nr::swin_softmax_row;
    float logits[dlss5_nr::kWindowTokens];
    float bias[dlss5_nr::kWindowTokens];
    float out[dlss5_nr::kWindowTokens];
    memset(logits, 0, sizeof logits);
    const float biases[] = {1.28125f, 1.5f, 1.25f};
    for (std::size_t b = 0; b < sizeof biases / sizeof biases[0]; ++b) {
        for (int j = 0; j < dlss5_nr::kWindowTokens; ++j) {
            bias[j] = biases[b];
        }
        fill_sentinel(out, dlss5_nr::kWindowTokens);
        swin_softmax_row(logits, bias, out);
        for (int j = 0; j < dlss5_nr::kWindowTokens; ++j) {
            CHECK(out[j] == 0.015625f);
        }
    }
}

// ---- the QKV normalisation head row -------------------------------------------------------------------
// q/k are scaled straight from the float32 accumulator to e4m3 in one
// rounding; v takes the stored double rounding. The cases pin the q/v
// asymmetry (367.875 -> 352 vs 384), the exact reciprocal-square-root scale
// (q = (3, 0...): norm = fl(1/3), the grouping discriminator) and the key path
// (k = (128, 0...): norm = 2^-7 exactly).
static void check_normalise_head_row_pins(void) {
    using dlss5_nr::swin_normalise_head_row;
    float q[dlss5_nr::kHeadDim];
    float k[dlss5_nr::kHeadDim];
    float v[dlss5_nr::kHeadDim];
    float out_q[dlss5_nr::kHeadDim];
    float out_k[dlss5_nr::kHeadDim];
    float out_v[dlss5_nr::kHeadDim];
    memset(q, 0, sizeof q);
    memset(k, 0, sizeof k);
    memset(v, 0, sizeof v);
    // q = (1, 0...): the squared norm is exactly 1, so the scale is exactly 1
    // and head_scale applies exactly: e4m3(367.875) = 352 on the q path, while
    // value 367.875 stores as 384 on the v path (the f16 tie lands on 368 and
    // the e4m3 tie on the even code 384).
    q[0] = 1.0f;
    v[0] = 367.875f;
    swin_normalise_head_row(q, k, v, 367.875f, out_q, out_k, out_v);
    CHECK(out_q[0] == 352.0f);
    CHECK(out_v[0] == 384.0f);
    CHECK(out_q[1] == 0.0f);
    CHECK(out_v[1] == 0.0f);
    // k = (128, 0...): the squared norm is 16384 (f16-exact), the reciprocal
    // root is exactly 2^-7 and e4m3(128 * 2^-7) = 1.
    memset(q, 0, sizeof q);
    memset(v, 0, sizeof v);
    k[0] = 128.0f;
    swin_normalise_head_row(q, k, v, 1.0f, out_q, out_k, out_v);
    CHECK(out_k[0] == 1.0f);
    CHECK(out_k[1] == 0.0f);
    // The head_scale/norm_scale grouping discriminator: q = (3, 0...) gives
    // norm = fl(1/3) (0x3eaaaaab); the pinned grouping computes the scale
    // product once (fl(fl(1/3) * 0.0244140625) = 0x3c055556) and then
    // e4m3(3 * scale) = 0.025390625, where a per-dimension regrouping
    // e4m3(fl(fl(3 * fl(1/3)) * 0.0244140625)) = e4m3(0.0244140625) hits the
    // f16/e4m3 midpoint tie and gives 0.0234375.
    memset(k, 0, sizeof k);
    memset(v, 0, sizeof v);
    q[0] = 3.0f;
    swin_normalise_head_row(q, k, v, 0.0244140625f, out_q, out_k, out_v);
    CHECK(out_q[0] == 0.025390625f);
}

// ---- the attention logit dot ---------------------------------------------------------------------------
// Exact dyadic answers and the ascending-d accumulation-order discriminator:
// q = (448, 2^-9 * 31), k = (448, 2 * 31): the d = 0 product 200704 absorbs
// every later 2^-8 product (each is below half an ulp of 200704) giving
// 200704; a smalls-first order would give 200704.125.
static void check_logit_pins(void) {
    using dlss5_nr::swin_logit_dot;
    float q[dlss5_nr::kHeadDim];
    float k[dlss5_nr::kHeadDim];
    for (int d = 0; d < dlss5_nr::kHeadDim; ++d) {
        q[d] = 0.0f;
        k[d] = 0.0f;
    }
    q[0] = 448.0f;
    k[0] = 448.0f;
    for (int d = 1; d < dlss5_nr::kHeadDim; ++d) {
        q[d] = 0.001953125f;  // 2^-9
        k[d] = 2.0f;
    }
    CHECK(swin_logit_dot(q, k) == 200704.0f);
    // Small exact answer: one shared dimension.
    for (int d = 0; d < dlss5_nr::kHeadDim; ++d) {
        q[d] = 0.0f;
        k[d] = 0.0f;
    }
    q[3] = 1.5f;
    k[3] = 2.0f;
    q[7] = -0.25f;
    k[7] = 8.0f;
    CHECK(swin_logit_dot(q, k) == 1.0f);  // 3 + (-2)
    CHECK(swin_logit_dot(k, q) == 1.0f);  // symmetric
}

// ---- the context einsum ---------------------------------------------------------------------------------
// out[t][d] = swin_store(sum_j prob[t][j] * v[j][d]): a four-token exact sum
// 256 + 104 + 7 + 0.875 = 367.875 stores as 384 (the double-rounding
// discriminator), and a uniform-probability mean stays exact.
static void check_context_pins(void) {
    using dlss5_nr::swin_context;
    constexpr std::ptrdiff_t kProbHeadStride = dlss5_nr::kBiasTable + 2;
    static float prob[2 * kProbHeadStride];
    static float value[dlss5_nr::kWindowTokens * dlss5_nr::kHeadDim];
    static float out[dlss5_nr::kWindowTokens * dlss5_nr::kHeadDim];
    memset(prob, 0, sizeof prob);
    memset(value, 0, sizeof value);
    // prob[0][0][j] = 1 for j = 1..4, v[1][0] = 256, v[2][0] = 104,
    // v[3][0] = 7, v[4][0] = 0.875 (all e4m3 grid values).
    for (int j = 1; j <= 4; ++j) {
        prob[j] = 1.0f;
    }
    value[1 * dlss5_nr::kHeadDim + 0] = 256.0f;
    value[2 * dlss5_nr::kHeadDim + 0] = 104.0f;
    value[3 * dlss5_nr::kHeadDim + 0] = 7.0f;
    value[4 * dlss5_nr::kHeadDim + 0] = 0.875f;
    fill_sentinel(out, dlss5_nr::kWindowTokens * dlss5_nr::kHeadDim);
    const AttentionTensorView prob_v = make_tensor(prob, 1, 1,
        kProbHeadStride, kProbHeadStride);
    const MatrixView value_v = make_view(value, dlss5_nr::kWindowTokens,
        dlss5_nr::kHeadDim, dlss5_nr::kHeadDim);
    MutableMatrixView out_v = make_out(out, dlss5_nr::kWindowTokens,
        dlss5_nr::kHeadDim, dlss5_nr::kHeadDim);
    CHECK(swin_context(prob_v, value_v, &out_v,
        dlss5_nr::kWindowTokens * dlss5_nr::kHeadDim) == Status::kOk);
    CHECK(out[0] == 384.0f);  // swin_store(367.875)
    CHECK(out[1] == 0.0f);
    // A uniform probability row averages the value rows exactly: prob = 2^-6
    // for all 64 keys, v[j][1] = 2^-9 -> sum = 64 * 2^-15 = 2^-9.
    for (int j = 0; j < dlss5_nr::kWindowTokens; ++j) {
        prob[j] = 0.015625f;  // 2^-6
        value[j * dlss5_nr::kHeadDim + 1] = 0.001953125f;  // 2^-9
    }
    CHECK(swin_context(prob_v, value_v, &out_v,
        dlss5_nr::kWindowTokens * dlss5_nr::kHeadDim) == Status::kOk);
    CHECK(out[1] == 0.001953125f);
    CHECK(out[2] == 0.0f);
}

// ---- batch entries against their row chains ------------------------------------------------------------
// Deterministic LCG corpora with padded strides whose padding is never read or
// written: swin_softmax_numerators, swin_softmax_denominator, swin_softmax,
// swin_normalise_qkv, swin_logits and swin_context reproduce the per-row
// arithmetic at every (window, head, token) position.
static void check_batch_equivalence(void) {
    using dlss5_nr::swin_context;
    using dlss5_nr::swin_logit_dot;
    using dlss5_nr::swin_logits;
    using dlss5_nr::swin_normalise_head_row;
    using dlss5_nr::swin_normalise_qkv;
    using dlss5_nr::swin_softmax;
    using dlss5_nr::swin_softmax_denominator;
    using dlss5_nr::swin_softmax_denominator_row;
    using dlss5_nr::swin_softmax_numerators;
    using dlss5_nr::swin_softmax_numerators_row;
    using dlss5_nr::swin_softmax_row;
    using dlss5_nr::swin_store;
    constexpr int kWindows = 2;
    constexpr int kHeads = 2;
    constexpr int kRows = kWindows * dlss5_nr::kWindowTokens;
    constexpr int kChannels = kHeads * dlss5_nr::kHeadDim;
    constexpr std::ptrdiff_t kHeadStride = dlss5_nr::kBiasTable + 3;
    constexpr std::ptrdiff_t kWindowStride = kHeads * kHeadStride + 2;
    std::uint32_t state = 0x51e3c2d9u;
    const auto next = [&state]() {
        state = state * 1664525u + 1013904223u;
        return state;
    };
    static float logits[kWindows * kWindowStride + 2];
    static float exp_bias[kHeads * (dlss5_nr::kBiasTable + 1)];
    static float numerators[kWindows * kWindowStride + 2];
    static float probs[kWindows * kWindowStride + 2];
    static float dens[kRows * kHeads * dlss5_nr::kWindowTokens + 2];
    static float query[kRows * (kChannels + 2)];
    static float key[kRows * (kChannels + 2)];
    static float value[kRows * (kChannels + 2)];
    static float head_scale[kHeads + 1];
    static float out_q[kRows * (kChannels + 2)];
    static float out_k[kRows * (kChannels + 2)];
    static float out_v[kRows * (kChannels + 2)];
    static float row_q[kChannels];
    static float row_k[kChannels];
    static float row_v[kChannels];
    static float row_oq[kChannels];
    static float row_ok[kChannels];
    static float row_ov[kChannels];
    fill_sentinel(logits, kWindows * kWindowStride + 2);
    fill_sentinel(numerators, kWindows * kWindowStride + 2);
    fill_sentinel(probs, kWindows * kWindowStride + 2);
    fill_sentinel(out_q, kRows * (kChannels + 2));
    fill_sentinel(out_k, kRows * (kChannels + 2));
    fill_sentinel(out_v, kRows * (kChannels + 2));
    for (int w = 0; w < kWindows; ++w) {
        for (int h = 0; h < kHeads; ++h) {
            float* block = logits + w * kWindowStride + h * kHeadStride;
            for (int p = 0; p < dlss5_nr::kBiasTable; ++p) {
                block[p] = (float)(next() % 2048u) * 0.03125f - 32.0f;
            }
        }
    }
    for (int h = 0; h < kHeads; ++h) {
        for (int p = 0; p < dlss5_nr::kBiasTable; ++p) {
            const float raw = (float)(next() % 256u) * 0.0625f - 4.0f;
            exp_bias[h * (dlss5_nr::kBiasTable + 1) + p] =
                (float)((double)raw * (double)dlss5_nr::kExpSlope
                    + (double)dlss5_nr::kExpOffset);
        }
        exp_bias[h * (dlss5_nr::kBiasTable + 1) + dlss5_nr::kBiasTable] = kSentinel;
    }
    const AttentionTensorView logits_v = make_tensor(logits, kWindows, kHeads,
        kWindowStride, kHeadStride);
    const MatrixView exp_bias_v = make_view(exp_bias, kHeads,
        dlss5_nr::kBiasTable, dlss5_nr::kBiasTable + 1);
    MutableAttentionTensorView numerators_v = make_mutable_tensor(numerators,
        kWindows, kHeads, kWindowStride, kHeadStride);
    CHECK(swin_softmax_numerators(logits_v, exp_bias_v, &numerators_v,
        kWindows * kWindowStride + 2) == Status::kOk);
    for (int w = 0; w < kWindows; ++w) {
        for (int h = 0; h < kHeads; ++h) {
            const float* lblock = logits + w * kWindowStride + h * kHeadStride;
            float* nblock = numerators + w * kWindowStride + h * kHeadStride;
            const float* brow = exp_bias + h * (dlss5_nr::kBiasTable + 1);
            float expected[dlss5_nr::kWindowTokens];
            for (int i = 0; i < dlss5_nr::kWindowTokens; ++i) {
                swin_softmax_numerators_row(lblock + i * dlss5_nr::kWindowTokens,
                    brow + i * dlss5_nr::kWindowTokens, expected);
                for (int j = 0; j < dlss5_nr::kWindowTokens; ++j) {
                    CHECK(nblock[i * dlss5_nr::kWindowTokens + j] == expected[j]);
                }
            }
            CHECK(nblock[dlss5_nr::kBiasTable] == kSentinel);
            CHECK(nblock[dlss5_nr::kBiasTable + 1] == kSentinel);
            CHECK(nblock[dlss5_nr::kBiasTable + 2] == kSentinel);
        }
    }
    // Denominator batch over the f16-grid numerator tensor.
    static float numerators_f16[kWindows * kWindowStride + 2];
    for (std::size_t p = 0; p < kWindows * kWindowStride + 2; ++p) {
        numerators_f16[p] = kSentinel;
    }
    for (int w = 0; w < kWindows; ++w) {
        for (int h = 0; h < kHeads; ++h) {
            float* block = numerators_f16 + w * kWindowStride + h * kHeadStride;
            for (int p = 0; p < dlss5_nr::kBiasTable; ++p) {
                block[p] = dlss5_nr::f16(
                    (float)(next() % 1024u) * 0.0078125f - 2.0f);
            }
        }
    }
    const AttentionTensorView numerators_f16_v = make_tensor(numerators_f16,
        kWindows, kHeads, kWindowStride, kHeadStride);
    fill_sentinel(dens, kRows * kHeads * dlss5_nr::kWindowTokens + 2);
    CHECK(swin_softmax_denominator(numerators_f16_v, dens,
        kRows * kHeads * dlss5_nr::kWindowTokens + 2) == Status::kOk);
    for (int w = 0; w < kWindows; ++w) {
        for (int h = 0; h < kHeads; ++h) {
            const float* block =
                numerators_f16 + w * kWindowStride + h * kHeadStride;
            float* drow = dens + (w * kHeads + h) * dlss5_nr::kWindowTokens;
            for (int i = 0; i < dlss5_nr::kWindowTokens; ++i) {
                CHECK(drow[i]
                    == swin_softmax_denominator_row(block
                        + i * dlss5_nr::kWindowTokens));
            }
        }
    }
    CHECK(dens[kRows * kHeads * dlss5_nr::kWindowTokens] == kSentinel);
    CHECK(dens[kRows * kHeads * dlss5_nr::kWindowTokens + 1] == kSentinel);
    // Softmax batch against the row chain.
    MutableAttentionTensorView probs_v = make_mutable_tensor(probs, kWindows,
        kHeads, kWindowStride, kHeadStride);
    CHECK(swin_softmax(logits_v, exp_bias_v, &probs_v,
        kWindows * kWindowStride + 2) == Status::kOk);
    for (int w = 0; w < kWindows; ++w) {
        for (int h = 0; h < kHeads; ++h) {
            const float* lblock = logits + w * kWindowStride + h * kHeadStride;
            const float* brow = exp_bias + h * (dlss5_nr::kBiasTable + 1);
            float* pblock = probs + w * kWindowStride + h * kHeadStride;
            float expected[dlss5_nr::kWindowTokens];
            for (int i = 0; i < dlss5_nr::kWindowTokens; ++i) {
                swin_softmax_row(lblock + i * dlss5_nr::kWindowTokens,
                    brow + i * dlss5_nr::kWindowTokens, expected);
                for (int j = 0; j < dlss5_nr::kWindowTokens; ++j) {
                    CHECK(pblock[i * dlss5_nr::kWindowTokens + j] == expected[j]);
                }
            }
            CHECK(pblock[dlss5_nr::kBiasTable] == kSentinel);
            CHECK(pblock[dlss5_nr::kBiasTable + 1] == kSentinel);
            CHECK(pblock[dlss5_nr::kBiasTable + 2] == kSentinel);
        }
    }
    // QKV normalisation batch against the head rows.
    for (int r = 0; r < kRows; ++r) {
        for (int c = 0; c < kChannels; ++c) {
            query[r * (kChannels + 2) + c] = (float)(next() % 512u) * 0.0625f - 8.0f;
            key[r * (kChannels + 2) + c] = (float)(next() % 512u) * 0.0625f - 8.0f;
            value[r * (kChannels + 2) + c] = (float)(next() % 512u) * 0.0625f - 8.0f;
        }
        query[r * (kChannels + 2) + kChannels] = kSentinel;
        query[r * (kChannels + 2) + kChannels + 1] = kSentinel;
        key[r * (kChannels + 2) + kChannels] = kSentinel;
        key[r * (kChannels + 2) + kChannels + 1] = kSentinel;
        value[r * (kChannels + 2) + kChannels] = kSentinel;
        value[r * (kChannels + 2) + kChannels + 1] = kSentinel;
    }
    for (int h = 0; h < kHeads; ++h) {
        head_scale[h] = (float)(next() % 256u) * 0.015625f - 1.0f;
    }
    head_scale[kHeads] = kSentinel;
    const MatrixView query_v = make_view(query, kRows, kChannels, kChannels + 2);
    const MatrixView key_v = make_view(key, kRows, kChannels, kChannels + 2);
    const MatrixView value_v = make_view(value, kRows, kChannels, kChannels + 2);
    const MatrixView head_scale_v = make_view(head_scale, 1, kHeads, kHeads + 1);
    MutableMatrixView out_q_v = make_out(out_q, kRows, kChannels, kChannels + 2);
    MutableMatrixView out_k_v = make_out(out_k, kRows, kChannels, kChannels + 2);
    MutableMatrixView out_v_v = make_out(out_v, kRows, kChannels, kChannels + 2);
    CHECK(swin_normalise_qkv(query_v, key_v, value_v, head_scale_v, kHeads,
        &out_q_v, kRows * (kChannels + 2), &out_k_v, kRows * (kChannels + 2),
        &out_v_v, kRows * (kChannels + 2)) == Status::kOk);
    for (int r = 0; r < kRows; ++r) {
        for (int g = 0; g < kHeads; ++g) {
            memcpy(row_q, query + r * (kChannels + 2) + g * dlss5_nr::kHeadDim,
                sizeof row_q);
            memcpy(row_k, key + r * (kChannels + 2) + g * dlss5_nr::kHeadDim,
                sizeof row_k);
            memcpy(row_v, value + r * (kChannels + 2) + g * dlss5_nr::kHeadDim,
                sizeof row_v);
            swin_normalise_head_row(row_q, row_k, row_v, head_scale[g],
                row_oq, row_ok, row_ov);
            for (int d = 0; d < dlss5_nr::kHeadDim; ++d) {
                CHECK(out_q[r * (kChannels + 2) + g * dlss5_nr::kHeadDim + d]
                    == row_oq[d]);
                CHECK(out_k[r * (kChannels + 2) + g * dlss5_nr::kHeadDim + d]
                    == row_ok[d]);
                CHECK(out_v[r * (kChannels + 2) + g * dlss5_nr::kHeadDim + d]
                    == row_ov[d]);
            }
        }
        CHECK(out_q[r * (kChannels + 2) + kChannels] == kSentinel);
        CHECK(out_q[r * (kChannels + 2) + kChannels + 1] == kSentinel);
        CHECK(out_k[r * (kChannels + 2) + kChannels] == kSentinel);
        CHECK(out_k[r * (kChannels + 2) + kChannels + 1] == kSentinel);
        CHECK(out_v[r * (kChannels + 2) + kChannels] == kSentinel);
        CHECK(out_v[r * (kChannels + 2) + kChannels + 1] == kSentinel);
    }
    // Logits batch against the dot rows. The einsum contract reads the
    // normalised e4m3 q/k (the reference einsums the `_normalise_qkv` outputs),
    // so both the batch entry and the row chain use out_q/out_k.
    const MatrixView out_q_read = make_view(out_q, kRows, kChannels,
        kChannels + 2);
    const MatrixView out_k_read = make_view(out_k, kRows, kChannels,
        kChannels + 2);
    static float logits2[kWindows * kWindowStride + 2];
    fill_sentinel(logits2, kWindows * kWindowStride + 2);
    MutableAttentionTensorView logits2_v = make_mutable_tensor(logits2,
        kWindows, kHeads, kWindowStride, kHeadStride);
    CHECK(swin_logits(out_q_read, out_k_read, kHeads, &logits2_v,
        kWindows * kWindowStride + 2) == Status::kOk);
    for (int w = 0; w < kWindows; ++w) {
        for (int h = 0; h < kHeads; ++h) {
            float* block = logits2 + w * kWindowStride + h * kHeadStride;
            for (int i = 0; i < dlss5_nr::kWindowTokens; ++i) {
                const float* qrow = out_q + (w * dlss5_nr::kWindowTokens + i)
                    * (kChannels + 2) + h * dlss5_nr::kHeadDim;
                for (int j = 0; j < dlss5_nr::kWindowTokens; ++j) {
                    const float* krow = out_k + (w * dlss5_nr::kWindowTokens + j)
                        * (kChannels + 2) + h * dlss5_nr::kHeadDim;
                    CHECK(block[i * dlss5_nr::kWindowTokens + j]
                        == swin_logit_dot(qrow, krow));
                }
            }
            CHECK(block[dlss5_nr::kBiasTable] == kSentinel);
            CHECK(block[dlss5_nr::kBiasTable + 1] == kSentinel);
            CHECK(block[dlss5_nr::kBiasTable + 2] == kSentinel);
        }
    }
    // Context batch against the staged einsum through store. The value operand
    // is the normalised e4m3 v (the reference's `q8(_f16(value))`), so both the
    // batch entry and the row chain use out_v.
    const MatrixView out_v_read = make_view(out_v, kRows, kChannels,
        kChannels + 2);
    const AttentionTensorView probs_read = make_tensor(probs, kWindows, kHeads,
        kWindowStride, kHeadStride);
    static float context[kRows * (kChannels + 2)];
    fill_sentinel(context, kRows * (kChannels + 2));
    MutableMatrixView context_v = make_out(context, kRows, kChannels,
        kChannels + 2);
    CHECK(swin_context(probs_read, out_v_read, &context_v,
        kRows * (kChannels + 2)) == Status::kOk);
    for (int w = 0; w < kWindows; ++w) {
        for (int i = 0; i < dlss5_nr::kWindowTokens; ++i) {
            const int r = w * dlss5_nr::kWindowTokens + i;
            for (int g = 0; g < kHeads; ++g) {
                const float* pblock = probs + w * kWindowStride + g * kHeadStride;
                for (int d = 0; d < dlss5_nr::kHeadDim; ++d) {
                    float sum = 0.0f;
                    for (int j = 0; j < dlss5_nr::kWindowTokens; ++j) {
                        const float* vrow = out_v
                            + (w * dlss5_nr::kWindowTokens + j) * (kChannels + 2)
                            + g * dlss5_nr::kHeadDim;
                        sum += pblock[i * dlss5_nr::kWindowTokens + j] * vrow[d];
                    }
                    CHECK(context[r * (kChannels + 2) + g * dlss5_nr::kHeadDim + d]
                        == swin_store(sum));
                }
            }
            CHECK(context[r * (kChannels + 2) + kChannels] == kSentinel);
            CHECK(context[r * (kChannels + 2) + kChannels + 1] == kSentinel);
        }
    }
}

// ---- the assembled swin_attention ------------------------------------------------------------------------
// Hand pins through the full chain (all-zero projections with an exact
// residual product; a value passthrough with uniform 2^-6 probabilities) and a
// bit-exact equivalence control against the staged chain over a deterministic
// e4m3 corpus with padded strides and untouched padding.
static void check_attention_assembly(void) {
    using dlss5_nr::swin_attention;
    using dlss5_nr::swin_context;
    using dlss5_nr::swin_linear;
    using dlss5_nr::swin_logits;
    using dlss5_nr::swin_mlp_tail;
    using dlss5_nr::swin_normalise_qkv;
    using dlss5_nr::swin_softmax;
    constexpr int kHeads = 2;
    constexpr int kChannels = kHeads * dlss5_nr::kHeadDim;
    constexpr int kWindows = 2;
    constexpr int kRows = kWindows * dlss5_nr::kWindowTokens;
    constexpr int kQkvRows = 3 * kChannels;
    std::uint32_t state = 0x2468ace0u;
    const auto next = [&state]() {
        state = state * 1664525u + 1013904223u;
        return state;
    };
    static float quantised[kRows * (kChannels + 1)];
    static float qkv[kQkvRows * kChannels];
    static float head_scale[kHeads];
    static float exp_bias[kHeads * dlss5_nr::kBiasTable];
    static float skip[kRows * (kChannels + 1)];
    static float attn_scale[kChannels];
    static float out_proj[kChannels * kChannels];
    static float out[kRows * (kChannels + 1)];
    static float out_staged[kRows * (kChannels + 1)];
    static float scratch[kRows * kQkvRows + 4 * kRows * kChannels
        + 2 * kHeads * dlss5_nr::kBiasTable + 4];
    // Hand pin 1: zero projections give zero qkv/context; the output is the
    // exact residual product skip * attn_scale.
    memset(quantised, 0, sizeof quantised);
    memset(qkv, 0, sizeof qkv);
    memset(out_proj, 0, sizeof out_proj);
    for (int h = 0; h < kHeads; ++h) {
        head_scale[h] = 1.0f;
        for (int p = 0; p < dlss5_nr::kBiasTable; ++p) {
            exp_bias[h * dlss5_nr::kBiasTable + p] = 1.28125f;
        }
    }
    for (int c = 0; c < kChannels; ++c) {
        attn_scale[c] = (float)((c % 4) + 1) * 0.25f;  // f16-exact
    }
    for (int r = 0; r < kRows; ++r) {
        for (int c = 0; c < kChannels; ++c) {
            skip[r * (kChannels + 1) + c] = (float)((r % 3) - 1) * 2.0f;
            out[r * (kChannels + 1) + c] = kSentinel;
        }
        skip[r * (kChannels + 1) + kChannels] = kSentinel;
        out[r * (kChannels + 1) + kChannels] = kSentinel;
    }
    const MatrixView quantised_v = make_view(quantised, kRows, kChannels,
        kChannels + 1);
    const MatrixView qkv_v = make_view(qkv, kQkvRows, kChannels, kChannels);
    const MatrixView head_scale_v = make_view(head_scale, 1, kHeads, kHeads);
    const MatrixView exp_bias_v = make_view(exp_bias, kHeads,
        dlss5_nr::kBiasTable, dlss5_nr::kBiasTable);
    const MatrixView skip_v = make_view(skip, kRows, kChannels, kChannels + 1);
    const MatrixView attn_scale_v = make_view(attn_scale, 1, kChannels, kChannels);
    const MatrixView out_proj_v = make_view(out_proj, kChannels, kChannels,
        kChannels);
    MutableMatrixView out_v = make_out(out, kRows, kChannels, kChannels + 1);
    fill_sentinel(scratch, kRows * kQkvRows + 4 * kRows * kChannels
        + 2 * kHeads * dlss5_nr::kBiasTable + 4);
    CHECK(swin_attention(quantised_v, qkv_v, head_scale_v, exp_bias_v, skip_v,
        attn_scale_v, out_proj_v, &out_v, kRows * (kChannels + 1), scratch,
        kRows * kQkvRows + 4 * kRows * kChannels
            + 2 * kHeads * dlss5_nr::kBiasTable + 4) == Status::kOk);
    for (int r = 0; r < kRows; ++r) {
        for (int c = 0; c < kChannels; ++c) {
            const float expected = skip[r * (kChannels + 1) + c] * attn_scale[c];
            CHECK(out[r * (kChannels + 1) + c] == expected);
        }
        CHECK(out[r * (kChannels + 1) + kChannels] == kSentinel);
    }
    // Hand pin 2: quantised[all][0] = 1 with qkv weight row 64 (head 0, v dim 0)
    // carrying 2^-9 gives value 2^-9 on every token; zero q and k give uniform
    // 2^-6 probabilities (bias 1.28125 -> numerator 2^-6 -> denominator 1), so
    // context[t][0] = 64 * 2^-15 = 2^-9; the identity out_proj passes it
    // through and the output is 2^-9 with a zero skip.
    memset(quantised, 0, sizeof quantised);
    memset(qkv, 0, sizeof qkv);
    memset(skip, 0, sizeof skip);
    memset(out_proj, 0, sizeof out_proj);
    for (int c = 0; c < kChannels; ++c) {
        attn_scale[c] = 1.0f;
        out_proj[c * kChannels + c] = 1.0f;
    }
    for (int r = 0; r < kRows; ++r) {
        quantised[r * (kChannels + 1)] = 1.0f;
    }
    qkv[64 * kChannels + 0] = 0.001953125f;  // 2^-9
    CHECK(swin_attention(quantised_v, qkv_v, head_scale_v, exp_bias_v, skip_v,
        attn_scale_v, out_proj_v, &out_v, kRows * (kChannels + 1), scratch,
        kRows * kQkvRows + 4 * kRows * kChannels
            + 2 * kHeads * dlss5_nr::kBiasTable + 4) == Status::kOk);
    for (int r = 0; r < kRows; ++r) {
        CHECK(out[r * (kChannels + 1) + 0] == 0.001953125f);
        for (int c = 1; c < kChannels; ++c) {
            CHECK(out[r * (kChannels + 1) + c] == 0.0f);
        }
    }
    // Equivalence: the assembly is bit-identical to the staged chain
    // (swin_linear -> split -> swin_normalise_qkv -> swin_logits ->
    // swin_softmax -> swin_context -> swin_mlp_tail) over an e4m3 corpus with
    // padded strides and untouched padding.
    for (int r = 0; r < kRows; ++r) {
        for (int c = 0; c < kChannels; ++c) {
            quantised[r * (kChannels + 1) + c] =
                dlss5_nr::e4m3((float)(next() % 256u) * 0.0625f - 4.0f);
            skip[r * (kChannels + 1) + c] =
                (float)(next() % 128u) * 0.03125f - 2.0f;
        }
    }
    for (int p = 0; p < kQkvRows * kChannels; ++p) {
        qkv[p] = dlss5_nr::e4m3((float)(next() % 256u) * 0.0625f - 4.0f);
    }
    for (int p = 0; p < kChannels * kChannels; ++p) {
        out_proj[p] = dlss5_nr::e4m3((float)(next() % 256u) * 0.0625f - 4.0f);
    }
    for (int c = 0; c < kChannels; ++c) {
        attn_scale[c] = dlss5_nr::f16((float)(next() % 256u) * 0.015625f - 1.0f);
    }
    for (int h = 0; h < kHeads; ++h) {
        head_scale[h] = (float)(next() % 256u) * 0.015625f - 1.0f;
        for (int p = 0; p < dlss5_nr::kBiasTable; ++p) {
            const float raw = dlss5_nr::f16((float)(next() % 256u) * 0.0625f - 4.0f);
            exp_bias[h * dlss5_nr::kBiasTable + p] =
                (float)((double)raw * (double)dlss5_nr::kExpSlope
                    + (double)dlss5_nr::kExpOffset);
        }
    }
    fill_sentinel(out, kRows * (kChannels + 1));
    fill_sentinel(scratch, kRows * kQkvRows + 4 * kRows * kChannels
        + 2 * kHeads * dlss5_nr::kBiasTable + 4);
    CHECK(swin_attention(quantised_v, qkv_v, head_scale_v, exp_bias_v, skip_v,
        attn_scale_v, out_proj_v, &out_v, kRows * (kChannels + 1), scratch,
        kRows * kQkvRows + 4 * kRows * kChannels
            + 2 * kHeads * dlss5_nr::kBiasTable + 4) == Status::kOk);
    // The staged chain on the same buffers.
    static float qkv_staged[kRows * (kQkvRows + 1)];
    static float q_sep[kRows * (kChannels + 1)];
    static float k_sep[kRows * (kChannels + 1)];
    static float v_sep[kRows * (kChannels + 1)];
    static float logits_stage[kWindows * kHeads * dlss5_nr::kBiasTable];
    static float probs_stage[kWindows * kHeads * dlss5_nr::kBiasTable];
    static float context_stage[kRows * (kChannels + 1)];
    fill_sentinel(qkv_staged, kRows * (kQkvRows + 1));
    fill_sentinel(q_sep, kRows * (kChannels + 1));
    fill_sentinel(k_sep, kRows * (kChannels + 1));
    fill_sentinel(v_sep, kRows * (kChannels + 1));
    fill_sentinel(context_stage, kRows * (kChannels + 1));
    fill_sentinel(out_staged, kRows * (kChannels + 1));
    MutableMatrixView qkv_staged_v = make_out(qkv_staged, kRows, kQkvRows,
        kQkvRows + 1);
    CHECK(swin_linear(quantised_v, qkv_v, &qkv_staged_v, kRows * (kQkvRows + 1))
        == Status::kOk);
    for (int r = 0; r < kRows; ++r) {
        for (int g = 0; g < kHeads; ++g) {
            for (int d = 0; d < dlss5_nr::kHeadDim; ++d) {
                q_sep[r * (kChannels + 1) + g * dlss5_nr::kHeadDim + d] =
                    qkv_staged[r * (kQkvRows + 1) + g * 3 * dlss5_nr::kHeadDim + d];
                k_sep[r * (kChannels + 1) + g * dlss5_nr::kHeadDim + d] =
                    qkv_staged[r * (kQkvRows + 1) + g * 3 * dlss5_nr::kHeadDim
                        + dlss5_nr::kHeadDim + d];
                v_sep[r * (kChannels + 1) + g * dlss5_nr::kHeadDim + d] =
                    qkv_staged[r * (kQkvRows + 1) + g * 3 * dlss5_nr::kHeadDim
                        + 2 * dlss5_nr::kHeadDim + d];
            }
        }
    }
    const MatrixView q_sep_v = make_view(q_sep, kRows, kChannels, kChannels + 1);
    const MatrixView k_sep_v = make_view(k_sep, kRows, kChannels, kChannels + 1);
    const MatrixView v_sep_v = make_view(v_sep, kRows, kChannels, kChannels + 1);
    MutableMatrixView q_out_v = make_out(q_sep, kRows, kChannels, kChannels + 1);
    MutableMatrixView k_out_v = make_out(k_sep, kRows, kChannels, kChannels + 1);
    MutableMatrixView v_out_v = make_out(v_sep, kRows, kChannels, kChannels + 1);
    CHECK(swin_normalise_qkv(q_sep_v, k_sep_v, v_sep_v, head_scale_v, kHeads,
        &q_out_v, kRows * (kChannels + 1), &k_out_v, kRows * (kChannels + 1),
        &v_out_v, kRows * (kChannels + 1)) == Status::kOk);
    for (int w = 0; w < kWindows; ++w) {
        MatrixView q_window = make_view(q_sep + w * dlss5_nr::kWindowTokens
            * (kChannels + 1), dlss5_nr::kWindowTokens, kChannels, kChannels + 1);
        MatrixView k_window = make_view(k_sep + w * dlss5_nr::kWindowTokens
            * (kChannels + 1), dlss5_nr::kWindowTokens, kChannels, kChannels + 1);
        MatrixView v_window = make_view(v_sep + w * dlss5_nr::kWindowTokens
            * (kChannels + 1), dlss5_nr::kWindowTokens, kChannels, kChannels + 1);
        MutableAttentionTensorView logits_window = make_mutable_tensor(
            logits_stage + w * kHeads * dlss5_nr::kBiasTable, 1, kHeads,
            kHeads * dlss5_nr::kBiasTable, dlss5_nr::kBiasTable);
        CHECK(swin_logits(q_window, k_window, kHeads, &logits_window,
            kHeads * dlss5_nr::kBiasTable) == Status::kOk);
        AttentionTensorView logits_read = make_tensor(
            logits_stage + w * kHeads * dlss5_nr::kBiasTable, 1, kHeads,
            kHeads * dlss5_nr::kBiasTable, dlss5_nr::kBiasTable);
        MutableAttentionTensorView probs_window = make_mutable_tensor(
            probs_stage + w * kHeads * dlss5_nr::kBiasTable, 1, kHeads,
            kHeads * dlss5_nr::kBiasTable, dlss5_nr::kBiasTable);
        CHECK(swin_softmax(logits_read, exp_bias_v, &probs_window,
            kHeads * dlss5_nr::kBiasTable) == Status::kOk);
        AttentionTensorView probs_read = make_tensor(
            probs_stage + w * kHeads * dlss5_nr::kBiasTable, 1, kHeads,
            kHeads * dlss5_nr::kBiasTable, dlss5_nr::kBiasTable);
        MutableMatrixView context_window = make_out(
            context_stage + w * dlss5_nr::kWindowTokens * (kChannels + 1),
            dlss5_nr::kWindowTokens, kChannels, kChannels + 1);
        CHECK(swin_context(probs_read, v_window, &context_window,
            dlss5_nr::kWindowTokens * (kChannels + 1)) == Status::kOk);
    }
    const MatrixView context_v = make_view(context_stage, kRows, kChannels,
        kChannels + 1);
    MutableMatrixView out_staged_v = make_out(out_staged, kRows, kChannels,
        kChannels + 1);
    CHECK(swin_mlp_tail(context_v, out_proj_v, skip_v, attn_scale_v,
        &out_staged_v, kRows * (kChannels + 1)) == Status::kOk);
    for (int r = 0; r < kRows; ++r) {
        for (int c = 0; c < kChannels + 1; ++c) {
            CHECK(out[r * (kChannels + 1) + c]
                == out_staged[r * (kChannels + 1) + c]);
        }
    }
    // Determinism: a repeated run is bit-identical.
    static float out2[kRows * (kChannels + 1)];
    fill_sentinel(out2, kRows * (kChannels + 1));
    MutableMatrixView out2_v = make_out(out2, kRows, kChannels, kChannels + 1);
    CHECK(swin_attention(quantised_v, qkv_v, head_scale_v, exp_bias_v, skip_v,
        attn_scale_v, out_proj_v, &out2_v, kRows * (kChannels + 1), scratch,
        kRows * kQkvRows + 4 * kRows * kChannels
            + 2 * kHeads * dlss5_nr::kBiasTable + 4) == Status::kOk);
    for (int r = 0; r < kRows; ++r) {
        for (int c = 0; c < kChannels + 1; ++c) {
            CHECK(out2[r * (kChannels + 1) + c] == out[r * (kChannels + 1) + c]);
        }
    }
}

// ---- negative controls ------------------------------------------------------------------------------
// Every entry rejects its whole validation surface before the first write:
// null pointers, dimension bounds and pairing mismatches, stride bounds, wrong
// value classes (non-f16 bias tables, non-finite matrices, non-e4m3 operands)
// and short capacities. Every rejection leaves the outputs (and the assembly
// scratch) sentinel-untouched.

static void check_negative_bias_tables(void) {
    using dlss5_nr::swin_deswizzle_bias;
    using dlss5_nr::swin_exp_bias;
    using BiasEntry = Status (*)(const MatrixView&, MutableMatrixView*,
        std::size_t);
    const BiasEntry entries[] = {swin_deswizzle_bias, swin_exp_bias};
    constexpr int kHeads = 2;
    constexpr std::ptrdiff_t kRowStride = dlss5_nr::kBiasTable + 2;
    static float raw[kHeads * kRowStride];
    static float out[kHeads * kRowStride];
    for (std::size_t p = 0; p < kHeads * kRowStride; ++p) {
        raw[p] = 0.5f;  // f16 grid
    }
    fill_sentinel(out, kHeads * kRowStride);
    const std::size_t capacity = kHeads * kRowStride;
    const std::size_t needed =
        kRowStride + (std::size_t)dlss5_nr::kBiasTable;
    const MatrixView raw_v = make_view(raw, kHeads, dlss5_nr::kBiasTable,
        kRowStride);
    MutableMatrixView out_v = make_out(out, kHeads, dlss5_nr::kBiasTable,
        kRowStride);
    for (std::size_t e = 0; e < sizeof entries / sizeof entries[0]; ++e) {
        const BiasEntry entry = entries[e];
        // Null pointers (the null-output rejection cannot write by construction).
        CHECK(entry(make_view(nullptr, kHeads, dlss5_nr::kBiasTable, kRowStride),
            &out_v, capacity) == Status::kNullPointer);
        CHECK(entry(raw_v, nullptr, capacity) == Status::kNullPointer);
        MutableMatrixView no_out = make_out(nullptr, kHeads,
            dlss5_nr::kBiasTable, kRowStride);
        CHECK(entry(raw_v, &no_out, capacity) == Status::kNullPointer);
        // Dimension classes: the row bounds and the fixed 4096-column table.
        CHECK(entry(make_view(raw, 0, dlss5_nr::kBiasTable, kRowStride),
            &out_v, capacity) == Status::kInvalidDimensions);
        CHECK(entry(make_view(raw, dlss5_nr::kMaxHeads + 1,
            dlss5_nr::kBiasTable, kRowStride),
            &out_v, capacity) == Status::kInvalidDimensions);
        CHECK(entry(make_view(raw, kHeads, dlss5_nr::kBiasTable - 1,
            kRowStride), &out_v, capacity) == Status::kInvalidDimensions);
        // Stride classes on both sides (scans never run after these).
        CHECK(entry(make_view(raw, kHeads, dlss5_nr::kBiasTable,
            dlss5_nr::kBiasTable - 1), &out_v, capacity)
            == Status::kInvalidStride);
        CHECK(entry(make_view(raw, kHeads, dlss5_nr::kBiasTable,
            dlss5_nr::kMaxAttentionSpan + 1), &out_v, capacity)
            == Status::kInvalidStride);
        MutableMatrixView bad_out = make_out(out, kHeads, dlss5_nr::kBiasTable,
            dlss5_nr::kBiasTable - 1);
        CHECK(entry(raw_v, &bad_out, capacity) == Status::kInvalidStride);
        bad_out = make_out(out, kHeads, dlss5_nr::kBiasTable,
            dlss5_nr::kMaxAttentionSpan + 1);
        CHECK(entry(raw_v, &bad_out, capacity) == Status::kInvalidStride);
        // Output-side pairing: the output rows must match the raw rows. (The
        // output validator's own rows bounds are unreachable through these
        // entries: the pairing check runs first and confines out->rows to the
        // raw table's 1..kMaxHeads.)
        MutableMatrixView bad_rows = make_out(out, kHeads + 1,
            dlss5_nr::kBiasTable, kRowStride);
        CHECK(entry(raw_v, &bad_rows, capacity) == Status::kInvalidDimensions);
        bad_rows = make_out(out, kHeads - 1, dlss5_nr::kBiasTable, kRowStride);
        CHECK(entry(raw_v, &bad_rows, capacity) == Status::kInvalidDimensions);
        // Capacity class on the output.
        bad_out = make_out(out, kHeads, dlss5_nr::kBiasTable, kRowStride);
        CHECK(entry(raw_v, &bad_out, needed - 1) == Status::kInvalidBuffer);
        CHECK(all_sentinel(out, capacity));
        // Binary16 value class: off-grid, subnormal gap, out-of-range, non-finite.
        const float bad_values[] = {0.1f, 3.14159f, 0x1p-25f, 1e30f, -1e30f,
            (float)NAN, (float)INFINITY, (float)-INFINITY};
        for (std::size_t i = 0; i < sizeof bad_values / sizeof bad_values[0];
                ++i) {
            raw[0] = bad_values[i];
            CHECK(entry(raw_v, &out_v, capacity) == Status::kInvalidValues);
            raw[0] = 0.5f;
        }
        CHECK(all_sentinel(out, capacity));
        // A valid boundary operand class still succeeds: the smallest f16
        // subnormal, the f16-exact +-448 and -0.0.
        raw[0] = 0x1p-24f;
        raw[1] = 448.0f;
        raw[2] = -448.0f;
        raw[3] = -0.0f;
        CHECK(entry(raw_v, &out_v, capacity) == Status::kOk);
        raw[0] = 0.5f;
        raw[1] = 0.5f;
        raw[2] = 0.5f;
        raw[3] = 0.5f;
        fill_sentinel(out, kHeads * kRowStride);
    }
}

static void check_negative_softmax_family(void) {
    using dlss5_nr::swin_softmax;
    using dlss5_nr::swin_softmax_denominator;
    using dlss5_nr::swin_softmax_numerators;
    constexpr int kWindows = 1;
    constexpr int kHeads = 2;
    constexpr std::ptrdiff_t kHeadStride = dlss5_nr::kBiasTable + 2;
    constexpr std::ptrdiff_t kWindowStride = kHeads * kHeadStride + 2;
    static float logits[kWindows * kWindowStride + 2];
    static float exp_bias[kHeads * (dlss5_nr::kBiasTable + 2)];
    static float out[kWindows * kWindowStride + 2];
    static float dens[kWindows * kHeads * dlss5_nr::kWindowTokens + 2];
    for (std::size_t p = 0; p < kWindows * kWindowStride + 2; ++p) {
        logits[p] = 0.25f;
    }
    for (std::size_t p = 0; p < kHeads * (dlss5_nr::kBiasTable + 2); ++p) {
        exp_bias[p] = 1.28125f;
    }
    const std::size_t out_cap = kWindows * kWindowStride + 2;
    const std::size_t tensor_needed = (std::size_t)(kHeads - 1)
            * (std::size_t)kHeadStride
        + (std::size_t)dlss5_nr::kBiasTable;
    const std::size_t dens_needed =
        (std::size_t)kWindows * kHeads * dlss5_nr::kWindowTokens;
    const std::size_t dens_cap =
        kWindows * kHeads * dlss5_nr::kWindowTokens + 2;
    fill_sentinel(out, out_cap);
    fill_sentinel(dens, dens_cap);
    const AttentionTensorView logits_v = make_tensor(logits, kWindows, kHeads,
        kWindowStride, kHeadStride);
    const MatrixView exp_bias_v = make_view(exp_bias, kHeads,
        dlss5_nr::kBiasTable, dlss5_nr::kBiasTable + 2);
    MutableAttentionTensorView out_v = make_mutable_tensor(out, kWindows,
        kHeads, kWindowStride, kHeadStride);
    // Null pointers.
    CHECK(swin_softmax_numerators(make_tensor(nullptr, kWindows, kHeads,
        kWindowStride, kHeadStride), exp_bias_v, &out_v, out_cap)
        == Status::kNullPointer);
    CHECK(swin_softmax_numerators(logits_v,
        make_view(nullptr, kHeads, dlss5_nr::kBiasTable, dlss5_nr::kBiasTable + 2),
        &out_v, out_cap) == Status::kNullPointer);
    CHECK(swin_softmax_numerators(logits_v, exp_bias_v, nullptr, out_cap)
        == Status::kNullPointer);
    MutableAttentionTensorView no_out = make_mutable_tensor(nullptr, kWindows,
        kHeads, kWindowStride, kHeadStride);
    CHECK(swin_softmax_numerators(logits_v, exp_bias_v, &no_out, out_cap)
        == Status::kNullPointer);
    CHECK(swin_softmax_denominator(make_tensor(nullptr, kWindows, kHeads,
        kWindowStride, kHeadStride), dens, dens_cap) == Status::kNullPointer);
    CHECK(swin_softmax_denominator(logits_v, nullptr, dens_cap)
        == Status::kNullPointer);
    CHECK(swin_softmax(make_tensor(nullptr, kWindows, kHeads, kWindowStride,
        kHeadStride), exp_bias_v, &out_v, out_cap) == Status::kNullPointer);
    CHECK(swin_softmax(logits_v,
        make_view(nullptr, kHeads, dlss5_nr::kBiasTable, dlss5_nr::kBiasTable + 2),
        &out_v, out_cap) == Status::kNullPointer);
    CHECK(swin_softmax(logits_v, exp_bias_v, nullptr, out_cap)
        == Status::kNullPointer);
    CHECK(swin_softmax(logits_v, exp_bias_v, &no_out, out_cap)
        == Status::kNullPointer);
    // Tensor dimension and stride classes (structure fails before any scan).
    CHECK(swin_softmax_numerators(make_tensor(logits, 0, kHeads, kWindowStride,
        kHeadStride), exp_bias_v, &out_v, out_cap) == Status::kInvalidDimensions);
    CHECK(swin_softmax_numerators(make_tensor(logits, kWindows, 0, kWindowStride,
        kHeadStride), exp_bias_v, &out_v, out_cap) == Status::kInvalidDimensions);
    CHECK(swin_softmax_numerators(make_tensor(logits, kWindows,
        dlss5_nr::kMaxHeads + 1, kWindowStride, kHeadStride), exp_bias_v,
        &out_v, out_cap) == Status::kInvalidDimensions);
    CHECK(swin_softmax_numerators(make_tensor(logits, kWindows, kHeads,
        kWindowStride, dlss5_nr::kBiasTable - 1), exp_bias_v, &out_v, out_cap)
        == Status::kInvalidStride);
    CHECK(swin_softmax_numerators(make_tensor(logits, kWindows, kHeads,
        kWindowStride, dlss5_nr::kMaxAttentionSpan + 1), exp_bias_v, &out_v,
        out_cap) == Status::kInvalidStride);
    CHECK(swin_softmax_numerators(make_tensor(logits, kWindows, kHeads,
        kHeads * kHeadStride - 1, kHeadStride), exp_bias_v, &out_v, out_cap)
        == Status::kInvalidStride);
    CHECK(swin_softmax_numerators(make_tensor(logits, kWindows, kHeads,
        dlss5_nr::kMaxAttentionSpan + 1, kHeadStride), exp_bias_v, &out_v,
        out_cap) == Status::kInvalidStride);
    CHECK(swin_softmax_numerators(make_tensor(logits, 2, kHeads,
        dlss5_nr::kMaxAttentionSpan, dlss5_nr::kBiasTable), exp_bias_v, &out_v,
        out_cap) == Status::kInvalidStride);
    CHECK(swin_softmax_denominator(make_tensor(logits, kWindows, kHeads,
        kWindowStride, dlss5_nr::kBiasTable - 1), dens, dens_cap)
        == Status::kInvalidStride);
    // Bias-table dimension and stride classes.
    CHECK(swin_softmax_numerators(logits_v, make_view(exp_bias, 0,
        dlss5_nr::kBiasTable, dlss5_nr::kBiasTable + 2), &out_v, out_cap)
        == Status::kInvalidDimensions);
    CHECK(swin_softmax_numerators(logits_v, make_view(exp_bias, kHeads,
        dlss5_nr::kBiasTable - 1, dlss5_nr::kBiasTable + 2), &out_v, out_cap)
        == Status::kInvalidDimensions);
    CHECK(swin_softmax_numerators(logits_v, make_view(exp_bias, kHeads,
        dlss5_nr::kBiasTable, dlss5_nr::kBiasTable - 1), &out_v, out_cap)
        == Status::kInvalidStride);
    // Pairing class: one bias row per head.
    CHECK(swin_softmax_numerators(logits_v, make_view(exp_bias, 1,
        dlss5_nr::kBiasTable, dlss5_nr::kBiasTable + 2), &out_v, out_cap)
        == Status::kInvalidDimensions);
    // Value classes: non-finite logits and bias entries.
    logits[0] = (float)NAN;
    CHECK(swin_softmax_numerators(logits_v, exp_bias_v, &out_v, out_cap)
        == Status::kInvalidValues);
    CHECK(swin_softmax(logits_v, exp_bias_v, &out_v, out_cap)
        == Status::kInvalidValues);
    logits[0] = 0.25f;
    exp_bias[0] = (float)INFINITY;
    CHECK(swin_softmax_numerators(logits_v, exp_bias_v, &out_v, out_cap)
        == Status::kInvalidValues);
    CHECK(swin_softmax(logits_v, exp_bias_v, &out_v, out_cap)
        == Status::kInvalidValues);
    exp_bias[0] = 1.28125f;
    // f16-grid class on the denominator operand.
    static float numerators[kWindows * kWindowStride + 2];
    for (std::size_t p = 0; p < kWindows * kWindowStride + 2; ++p) {
        numerators[p] = 1.0f;
    }
    const AttentionTensorView numerators_v = make_tensor(numerators, kWindows,
        kHeads, kWindowStride, kHeadStride);
    numerators[0] = 0.1f;
    CHECK(swin_softmax_denominator(numerators_v, dens, dens_cap)
        == Status::kInvalidValues);
    numerators[0] = (float)NAN;
    CHECK(swin_softmax_denominator(numerators_v, dens, dens_cap)
        == Status::kInvalidValues);
    numerators[0] = 1.0f;
    // Output dimension, stride and capacity classes.
    MutableAttentionTensorView bad_out = make_mutable_tensor(out, kWindows + 1,
        kHeads, kWindowStride, kHeadStride);
    CHECK(swin_softmax_numerators(logits_v, exp_bias_v, &bad_out, out_cap)
        == Status::kInvalidDimensions);
    CHECK(swin_softmax(logits_v, exp_bias_v, &bad_out, out_cap)
        == Status::kInvalidDimensions);
    bad_out = make_mutable_tensor(out, kWindows, kHeads, kWindowStride,
        dlss5_nr::kBiasTable - 1);
    CHECK(swin_softmax_numerators(logits_v, exp_bias_v, &bad_out, out_cap)
        == Status::kInvalidStride);
    bad_out = make_mutable_tensor(out, kWindows, kHeads, kWindowStride,
        kHeadStride);
    CHECK(swin_softmax_numerators(logits_v, exp_bias_v, &bad_out,
        tensor_needed - 1) == Status::kInvalidBuffer);
    CHECK(swin_softmax(logits_v, exp_bias_v, &bad_out, tensor_needed - 1)
        == Status::kInvalidBuffer);
    CHECK(swin_softmax_denominator(numerators_v, dens, dens_needed - 1)
        == Status::kInvalidBuffer);
    // The rejected runs wrote nothing; a valid call then succeeds.
    CHECK(all_sentinel(out, out_cap));
    CHECK(all_sentinel(dens, dens_cap));
    CHECK(swin_softmax_numerators(logits_v, exp_bias_v, &out_v, out_cap)
        == Status::kOk);
    CHECK(!all_sentinel(out, out_cap));
}

static void check_negative_normalise(void) {
    using dlss5_nr::swin_normalise_qkv;
    constexpr int kRows = 2;
    constexpr int kHeads = 2;
    constexpr int kCols = kHeads * dlss5_nr::kHeadDim;
    static float query[kRows * (kCols + 2)];
    static float key[kRows * (kCols + 2)];
    static float value[kRows * (kCols + 2)];
    static float head_scale[dlss5_nr::kMaxHeads + 2];
    static float out_q[kRows * (kCols + 2)];
    static float out_k[kRows * (kCols + 2)];
    static float out_v[kRows * (kCols + 2)];
    for (int r = 0; r < kRows; ++r) {
        for (int c = 0; c < kCols + 2; ++c) {
            query[r * (kCols + 2) + c] = 1.5f;
            key[r * (kCols + 2) + c] = 2.0f;
            value[r * (kCols + 2) + c] = 4.0f;
        }
    }
    for (int h = 0; h < dlss5_nr::kMaxHeads + 2; ++h) {
        head_scale[h] = 1.0f;
    }
    fill_sentinel(out_q, kRows * (kCols + 2));
    fill_sentinel(out_k, kRows * (kCols + 2));
    fill_sentinel(out_v, kRows * (kCols + 2));
    const std::size_t out_cap = kRows * (kCols + 2);
    const std::size_t out_needed = static_cast<std::size_t>(kRows - 1)
            * static_cast<std::size_t>(kCols + 2) + kCols;
    const MatrixView query_v = make_view(query, kRows, kCols, kCols + 2);
    const MatrixView key_v = make_view(key, kRows, kCols, kCols + 2);
    const MatrixView value_v = make_view(value, kRows, kCols, kCols + 2);
    const MatrixView head_scale_v = make_view(head_scale, 1, kHeads, kHeads);
    MutableMatrixView oq = make_out(out_q, kRows, kCols, kCols + 2);
    MutableMatrixView ok = make_out(out_k, kRows, kCols, kCols + 2);
    MutableMatrixView ov = make_out(out_v, kRows, kCols, kCols + 2);
    // Null pointers on every operand and output.
    CHECK(swin_normalise_qkv(make_view(nullptr, kRows, kCols, kCols + 2),
        key_v, value_v, head_scale_v, kHeads, &oq, out_cap, &ok, out_cap, &ov,
        out_cap) == Status::kNullPointer);
    CHECK(swin_normalise_qkv(query_v, make_view(nullptr, kRows, kCols,
        kCols + 2), value_v, head_scale_v, kHeads, &oq, out_cap, &ok, out_cap,
        &ov, out_cap) == Status::kNullPointer);
    CHECK(swin_normalise_qkv(query_v, key_v, make_view(nullptr, kRows, kCols,
        kCols + 2), head_scale_v, kHeads, &oq, out_cap, &ok, out_cap, &ov,
        out_cap) == Status::kNullPointer);
    CHECK(swin_normalise_qkv(query_v, key_v, value_v, make_view(nullptr, 1,
        kHeads, kHeads), kHeads, &oq, out_cap, &ok, out_cap, &ov, out_cap)
        == Status::kNullPointer);
    CHECK(swin_normalise_qkv(query_v, key_v, value_v, head_scale_v, kHeads,
        nullptr, out_cap, &ok, out_cap, &ov, out_cap) == Status::kNullPointer);
    MutableMatrixView no_out = make_out(nullptr, kRows, kCols, kCols + 2);
    CHECK(swin_normalise_qkv(query_v, key_v, value_v, head_scale_v, kHeads,
        &no_out, out_cap, &ok, out_cap, &ov, out_cap) == Status::kNullPointer);
    CHECK(swin_normalise_qkv(query_v, key_v, value_v, head_scale_v, kHeads,
        &oq, out_cap, &no_out, out_cap, &ov, out_cap) == Status::kNullPointer);
    CHECK(swin_normalise_qkv(query_v, key_v, value_v, head_scale_v, kHeads,
        &oq, out_cap, &ok, out_cap, &no_out, out_cap) == Status::kNullPointer);
    // Finite-matrix dimension and stride classes on one operand.
    CHECK(swin_normalise_qkv(make_view(query, 0, kCols, kCols + 2), key_v,
        value_v, head_scale_v, kHeads, &oq, out_cap, &ok, out_cap, &ov, out_cap)
        == Status::kInvalidDimensions);
    CHECK(swin_normalise_qkv(make_view(query, kRows, 0, kCols + 2), key_v,
        value_v, head_scale_v, kHeads, &oq, out_cap, &ok, out_cap, &ov, out_cap)
        == Status::kInvalidDimensions);
    CHECK(swin_normalise_qkv(make_view(query, kRows,
        dlss5_nr::kMaxLinearDim + 1, kCols + 2), key_v, value_v, head_scale_v,
        kHeads, &oq, out_cap, &ok, out_cap, &ov, out_cap)
        == Status::kInvalidDimensions);
    CHECK(swin_normalise_qkv(make_view(query, kRows, kCols, kCols - 1), key_v,
        value_v, head_scale_v, kHeads, &oq, out_cap, &ok, out_cap, &ov, out_cap)
        == Status::kInvalidStride);
    CHECK(swin_normalise_qkv(make_view(query, kRows, kCols,
        dlss5_nr::kMaxLinearDim + 1), key_v, value_v, head_scale_v, kHeads,
        &oq, out_cap, &ok, out_cap, &ov, out_cap) == Status::kInvalidStride);
    // Pairing classes (each scanned operand stays inside its buffer).
    CHECK(swin_normalise_qkv(query_v, key_v, value_v, head_scale_v, 0, &oq,
        out_cap, &ok, out_cap, &ov, out_cap) == Status::kInvalidDimensions);
    CHECK(swin_normalise_qkv(query_v, key_v, value_v, head_scale_v,
        dlss5_nr::kMaxHeads + 1, &oq, out_cap, &ok, out_cap, &ov, out_cap)
        == Status::kInvalidDimensions);
    CHECK(swin_normalise_qkv(query_v, key_v, value_v,
        make_view(head_scale, 1, 1, 1), kHeads, &oq, out_cap, &ok, out_cap, &ov,
        out_cap) == Status::kInvalidDimensions);
    CHECK(swin_normalise_qkv(query_v, make_view(key, 1, kCols, kCols + 2),
        value_v, head_scale_v, kHeads, &oq, out_cap, &ok, out_cap, &ov, out_cap)
        == Status::kInvalidDimensions);
    CHECK(swin_normalise_qkv(query_v, key_v,
        make_view(value, 1, kCols, kCols + 2), head_scale_v, kHeads, &oq,
        out_cap, &ok, out_cap, &ov, out_cap) == Status::kInvalidDimensions);
    // Non-finite value classes on every scanned operand (query, key, value and
    // head_scale).
    const float bad_values[] = {(float)NAN, (float)INFINITY,
        (float)-INFINITY};
    for (std::size_t i = 0; i < sizeof bad_values / sizeof bad_values[0];
            ++i) {
        query[0] = bad_values[i];
        CHECK(swin_normalise_qkv(query_v, key_v, value_v, head_scale_v, kHeads,
            &oq, out_cap, &ok, out_cap, &ov, out_cap) == Status::kInvalidValues);
        query[0] = 1.5f;
        key[0] = bad_values[i];
        CHECK(swin_normalise_qkv(query_v, key_v, value_v, head_scale_v, kHeads,
            &oq, out_cap, &ok, out_cap, &ov, out_cap) == Status::kInvalidValues);
        key[0] = 2.0f;
        value[0] = bad_values[i];
        CHECK(swin_normalise_qkv(query_v, key_v, value_v, head_scale_v, kHeads,
            &oq, out_cap, &ok, out_cap, &ov, out_cap) == Status::kInvalidValues);
        value[0] = 4.0f;
        head_scale[0] = bad_values[i];
        CHECK(swin_normalise_qkv(query_v, key_v, value_v, head_scale_v, kHeads,
            &oq, out_cap, &ok, out_cap, &ov, out_cap) == Status::kInvalidValues);
        head_scale[0] = 1.0f;
    }
    // The head-scale magnitude bound: finite values beyond kMaxHeadScale are
    // rejected (they would overflow the norm*head scale product to infinity).
    head_scale[0] = 1e38f;
    CHECK(swin_normalise_qkv(query_v, key_v, value_v, head_scale_v, kHeads,
        &oq, out_cap, &ok, out_cap, &ov, out_cap) == Status::kInvalidValues);
    head_scale[0] = -70000.0f;
    CHECK(swin_normalise_qkv(query_v, key_v, value_v, head_scale_v, kHeads,
        &oq, out_cap, &ok, out_cap, &ov, out_cap) == Status::kInvalidValues);
    head_scale[0] = -dlss5_nr::kMaxHeadScale;  // the boundary stays accepted
    CHECK(swin_normalise_qkv(query_v, key_v, value_v, head_scale_v, kHeads,
        &oq, out_cap, &ok, out_cap, &ov, out_cap) == Status::kOk);
    fill_sentinel(out_q, out_cap);
    fill_sentinel(out_k, out_cap);
    fill_sentinel(out_v, out_cap);
    head_scale[0] = 1.0f;
    // Pairing class: exactly one head_scale row.
    CHECK(swin_normalise_qkv(query_v, key_v, value_v,
        make_view(head_scale, 2, kHeads, kHeads), kHeads, &oq, out_cap, &ok,
        out_cap, &ov, out_cap) == Status::kInvalidDimensions);
    // Output classes: the late out_v rejection must leave all three untouched.
    MutableMatrixView bad_rows = make_out(out_q, kRows - 1, kCols, kCols + 2);
    CHECK(swin_normalise_qkv(query_v, key_v, value_v, head_scale_v, kHeads,
        &bad_rows, out_cap, &ok, out_cap, &ov, out_cap)
        == Status::kInvalidDimensions);
    MutableMatrixView bad_stride = make_out(out_q, kRows, kCols, kCols - 1);
    CHECK(swin_normalise_qkv(query_v, key_v, value_v, head_scale_v, kHeads,
        &bad_stride, out_cap, &ok, out_cap, &ov, out_cap)
        == Status::kInvalidStride);
    // Short capacity on each output (the late out_v rejection must leave all
    // three untouched).
    CHECK(swin_normalise_qkv(query_v, key_v, value_v, head_scale_v, kHeads,
        &oq, out_needed - 1, &ok, out_cap, &ov, out_cap)
        == Status::kInvalidBuffer);
    CHECK(swin_normalise_qkv(query_v, key_v, value_v, head_scale_v, kHeads,
        &oq, out_cap, &ok, out_needed - 1, &ov, out_cap)
        == Status::kInvalidBuffer);
    CHECK(swin_normalise_qkv(query_v, key_v, value_v, head_scale_v, kHeads,
        &oq, out_cap, &ok, out_cap, &ov, out_needed - 1)
        == Status::kInvalidBuffer);
    CHECK(all_sentinel(out_q, out_cap));
    CHECK(all_sentinel(out_k, out_cap));
    CHECK(all_sentinel(out_v, out_cap));
}

static void check_negative_logits_context(void) {
    using dlss5_nr::swin_context;
    using dlss5_nr::swin_logits;
    constexpr int kWindows = 1;
    constexpr int kHeads = 2;
    constexpr int kRows = kWindows * dlss5_nr::kWindowTokens;
    constexpr int kCols = kHeads * dlss5_nr::kHeadDim;
    constexpr std::ptrdiff_t kHeadStride = dlss5_nr::kBiasTable + 2;
    constexpr std::ptrdiff_t kWindowStride = kHeads * kHeadStride + 2;
    static float query[kRows * (kCols + 2)];
    static float key[kRows * (kCols + 2)];
    static float logits[kWindows * kWindowStride + 2];
    static float probs[kWindows * kWindowStride + 2];
    static float out[kRows * (kCols + 2)];
    for (int p = 0; p < kRows * (kCols + 2); ++p) {
        query[p] = 0.5f;  // e4m3 grid
        key[p] = 1.0f;
    }
    // logits is the swin_logits output (sentinel-checked); the swin_context
    // probability operand is a separate e4m3-grid tensor so every rejection
    // below is rejected for its own reason and not a probability-grid scan.
    fill_sentinel(logits, kWindows * kWindowStride + 2);
    for (std::size_t p = 0; p < kWindows * kWindowStride + 2; ++p) {
        probs[p] = 1.0f;
    }
    fill_sentinel(out, kRows * (kCols + 2));
    const std::size_t tensor_cap = kWindows * kWindowStride + 2;
    const std::size_t tensor_needed = static_cast<std::size_t>(kWindows - 1)
            * static_cast<std::size_t>(kWindowStride)
        + static_cast<std::size_t>(kHeads - 1) * static_cast<std::size_t>(kHeadStride)
        + static_cast<std::size_t>(dlss5_nr::kBiasTable);
    const std::size_t out_cap = kRows * (kCols + 2);
    const std::size_t out_needed = static_cast<std::size_t>(kRows - 1)
            * static_cast<std::size_t>(kCols + 2) + kCols;
    const MatrixView query_v = make_view(query, kRows, kCols, kCols + 2);
    const MatrixView key_v = make_view(key, kRows, kCols, kCols + 2);
    const AttentionTensorView probs_v = make_tensor(probs, kWindows, kHeads,
        kWindowStride, kHeadStride);
    MutableAttentionTensorView logits_v = make_mutable_tensor(logits, kWindows,
        kHeads, kWindowStride, kHeadStride);
    MutableMatrixView out_v = make_out(out, kRows, kCols, kCols + 2);
    // Null pointers.
    CHECK(swin_logits(make_view(nullptr, kRows, kCols, kCols + 2), key_v,
        kHeads, &logits_v, tensor_cap) == Status::kNullPointer);
    CHECK(swin_logits(query_v, make_view(nullptr, kRows, kCols, kCols + 2),
        kHeads, &logits_v, tensor_cap) == Status::kNullPointer);
    CHECK(swin_logits(query_v, key_v, kHeads, nullptr, tensor_cap)
        == Status::kNullPointer);
    CHECK(swin_context(make_tensor(nullptr, kWindows, kHeads, kWindowStride,
        kHeadStride), key_v, &out_v, out_cap) == Status::kNullPointer);
    CHECK(swin_context(probs_v, make_view(nullptr, kRows, kCols, kCols + 2),
        &out_v, out_cap) == Status::kNullPointer);
    CHECK(swin_context(probs_v, key_v, nullptr, out_cap)
        == Status::kNullPointer);
    // e4m3 operand-grid class on both sides of each einsum entry.
    const float bad_values[] = {0.1f, 1.0625f, 0x1p-10f, 500.0f, -500.0f,
        (float)NAN, (float)INFINITY, (float)-INFINITY};
    for (std::size_t i = 0; i < sizeof bad_values / sizeof bad_values[0];
            ++i) {
        query[0] = bad_values[i];
        CHECK(swin_logits(query_v, key_v, kHeads, &logits_v, tensor_cap)
            == Status::kInvalidValues);
        query[0] = 0.5f;
        key[0] = bad_values[i];
        CHECK(swin_logits(query_v, key_v, kHeads, &logits_v, tensor_cap)
            == Status::kInvalidValues);
        CHECK(swin_context(probs_v, key_v, &out_v, out_cap)
            == Status::kInvalidValues);
        key[0] = 1.0f;
    }
    // Pairing classes (rows must be whole windows; columns must be heads * 32).
    CHECK(swin_logits(query_v, key_v, 0, &logits_v, tensor_cap)
        == Status::kInvalidDimensions);
    CHECK(swin_logits(query_v, key_v, 3, &logits_v, tensor_cap)
        == Status::kInvalidDimensions);
    CHECK(swin_logits(make_view(query, dlss5_nr::kWindowTokens - 1, kCols,
        kCols + 2), make_view(key, dlss5_nr::kWindowTokens - 1, kCols,
        kCols + 2), kHeads, &logits_v, tensor_cap)
        == Status::kInvalidDimensions);
    CHECK(swin_logits(query_v, make_view(key, kRows - 1, kCols, kCols + 2),
        kHeads, &logits_v, tensor_cap) == Status::kInvalidDimensions);
    CHECK(swin_context(probs_v, make_view(key, kRows - 1, kCols, kCols + 2),
        &out_v, out_cap) == Status::kInvalidDimensions);
    CHECK(swin_context(probs_v, make_view(key, kRows, kCols - 1, kCols + 2),
        &out_v, out_cap) == Status::kInvalidDimensions);
    // e4m3 class on the probability tensor.
    probs[0] = 0.1f;
    CHECK(swin_context(probs_v, key_v, &out_v, out_cap)
        == Status::kInvalidValues);
    probs[0] = 1.0f;
    // Output classes on both entries.
    MutableAttentionTensorView bad_tensor = make_mutable_tensor(logits,
        kWindows + 1, kHeads, kWindowStride, kHeadStride);
    CHECK(swin_logits(query_v, key_v, kHeads, &bad_tensor, tensor_cap)
        == Status::kInvalidDimensions);
    bad_tensor = make_mutable_tensor(logits, kWindows, kHeads + 1,
        kWindowStride, kHeadStride);
    CHECK(swin_logits(query_v, key_v, kHeads, &bad_tensor, tensor_cap)
        == Status::kInvalidDimensions);
    MutableAttentionTensorView bad_stride = make_mutable_tensor(logits,
        kWindows, kHeads, kWindowStride, dlss5_nr::kBiasTable - 1);
    CHECK(swin_logits(query_v, key_v, kHeads, &bad_stride, tensor_cap)
        == Status::kInvalidStride);
    MutableAttentionTensorView ok_tensor = make_mutable_tensor(logits,
        kWindows, kHeads, kWindowStride, kHeadStride);
    CHECK(swin_logits(query_v, key_v, kHeads, &ok_tensor, tensor_needed - 1)
        == Status::kInvalidBuffer);
    MutableMatrixView bad_out = make_out(out, kRows - 1, kCols, kCols + 2);
    CHECK(swin_context(probs_v, key_v, &bad_out, out_cap)
        == Status::kInvalidDimensions);
    MutableMatrixView bad_out_stride = make_out(out, kRows, kCols, kCols - 1);
    CHECK(swin_context(probs_v, key_v, &bad_out_stride, out_cap)
        == Status::kInvalidStride);
    CHECK(swin_context(probs_v, key_v, &out_v, out_needed - 1)
        == Status::kInvalidBuffer);
    // The rejected runs wrote nothing.
    CHECK(all_sentinel(out, out_cap));
    CHECK(all_sentinel(logits, tensor_cap));
}

static void check_negative_attention(void) {
    using dlss5_nr::swin_attention;
    constexpr int kHeads = 2;
    constexpr int kChannels = kHeads * dlss5_nr::kHeadDim;
    constexpr int kRows = dlss5_nr::kWindowTokens;
    constexpr int kQkvRows = 3 * kChannels;
    static float quantised[kRows * (kChannels + 1)];
    static float qkv[(kQkvRows + 8) * kChannels];
    static float head_scale[kHeads + 2];
    static float exp_bias[kHeads * (dlss5_nr::kBiasTable + 1)];
    static float skip[kRows * (kChannels + 1)];
    static float attn_scale[kChannels + 2];
    static float out_proj[kChannels * kChannels];
    static float out[kRows * (kChannels + 1)];
    static float scratch[kRows * kQkvRows + 4 * kRows * kChannels
        + 2 * kHeads * dlss5_nr::kBiasTable + 4];
    for (int p = 0; p < kRows * (kChannels + 1); ++p) {
        quantised[p] = 0.5f;
        skip[p] = 1.5f;
    }
    for (int p = 0; p < (kQkvRows + 8) * kChannels; ++p) {
        qkv[p] = 1.0f;
    }
    for (int p = 0; p < kHeads + 2; ++p) {
        head_scale[p] = 1.0f;
    }
    for (int p = 0; p < kHeads * (dlss5_nr::kBiasTable + 1); ++p) {
        exp_bias[p] = 1.28125f;
    }
    for (int p = 0; p < kChannels + 2; ++p) {
        attn_scale[p] = 1.0f;
    }
    for (int p = 0; p < kChannels * kChannels; ++p) {
        out_proj[p] = 0.5f;
    }
    const std::size_t out_cap = kRows * (kChannels + 1);
    const std::size_t out_needed = static_cast<std::size_t>(kRows - 1)
            * static_cast<std::size_t>(kChannels + 1) + kChannels;
    const std::size_t scratch_cap = kRows * kQkvRows
        + 4 * kRows * kChannels + 2 * kHeads * dlss5_nr::kBiasTable + 4;
    const std::size_t needed = kRows * kQkvRows
        + 4 * kRows * kChannels + 2 * kHeads * dlss5_nr::kBiasTable;
    const MatrixView quantised_v = make_view(quantised, kRows, kChannels,
        kChannels + 1);
    const MatrixView qkv_v = make_view(qkv, kQkvRows, kChannels, kChannels);
    const MatrixView head_scale_v = make_view(head_scale, 1, kHeads, kHeads);
    const MatrixView exp_bias_v = make_view(exp_bias, kHeads,
        dlss5_nr::kBiasTable, dlss5_nr::kBiasTable + 1);
    const MatrixView skip_v = make_view(skip, kRows, kChannels, kChannels + 1);
    const MatrixView attn_scale_v = make_view(attn_scale, 1, kChannels,
        kChannels);
    const MatrixView out_proj_v = make_view(out_proj, kChannels, kChannels,
        kChannels);
    MutableMatrixView out_v = make_out(out, kRows, kChannels, kChannels + 1);
    const auto untouched = [&]() {
        CHECK(all_sentinel(out, out_cap));
        CHECK(all_sentinel(scratch, scratch_cap));
    };
    fill_sentinel(out, out_cap);
    fill_sentinel(scratch, scratch_cap);
    // Null pointers on every operand, the scratch and the output.
    CHECK(swin_attention(make_view(nullptr, kRows, kChannels, kChannels + 1),
        qkv_v, head_scale_v, exp_bias_v, skip_v, attn_scale_v, out_proj_v,
        &out_v, out_cap, scratch, scratch_cap) == Status::kNullPointer);
    CHECK(swin_attention(quantised_v, make_view(nullptr, kQkvRows, kChannels,
        kChannels), head_scale_v, exp_bias_v, skip_v, attn_scale_v, out_proj_v,
        &out_v, out_cap, scratch, scratch_cap) == Status::kNullPointer);
    CHECK(swin_attention(quantised_v, qkv_v, make_view(nullptr, 1, kHeads,
        kHeads), exp_bias_v, skip_v, attn_scale_v, out_proj_v, &out_v, out_cap,
        scratch, scratch_cap) == Status::kNullPointer);
    CHECK(swin_attention(quantised_v, qkv_v, head_scale_v,
        make_view(nullptr, kHeads, dlss5_nr::kBiasTable,
            dlss5_nr::kBiasTable + 1),
        skip_v, attn_scale_v, out_proj_v, &out_v, out_cap, scratch, scratch_cap)
        == Status::kNullPointer);
    CHECK(swin_attention(quantised_v, qkv_v, head_scale_v, exp_bias_v,
        make_view(nullptr, kRows, kChannels, kChannels + 1), attn_scale_v,
        out_proj_v, &out_v, out_cap, scratch, scratch_cap)
        == Status::kNullPointer);
    CHECK(swin_attention(quantised_v, qkv_v, head_scale_v, exp_bias_v, skip_v,
        make_view(nullptr, 1, kChannels, kChannels), out_proj_v, &out_v,
        out_cap, scratch, scratch_cap) == Status::kNullPointer);
    CHECK(swin_attention(quantised_v, qkv_v, head_scale_v, exp_bias_v, skip_v,
        attn_scale_v, make_view(nullptr, kChannels, kChannels, kChannels),
        &out_v, out_cap, scratch, scratch_cap) == Status::kNullPointer);
    CHECK(swin_attention(quantised_v, qkv_v, head_scale_v, exp_bias_v, skip_v,
        attn_scale_v, out_proj_v, &out_v, out_cap, nullptr, scratch_cap)
        == Status::kNullPointer);
    CHECK(swin_attention(quantised_v, qkv_v, head_scale_v, exp_bias_v, skip_v,
        attn_scale_v, out_proj_v, nullptr, out_cap, scratch, scratch_cap)
        == Status::kNullPointer);
    MutableMatrixView no_out = make_out(nullptr, kRows, kChannels,
        kChannels + 1);
    CHECK(swin_attention(quantised_v, qkv_v, head_scale_v, exp_bias_v, skip_v,
        attn_scale_v, out_proj_v, &no_out, out_cap, scratch, scratch_cap)
        == Status::kNullPointer);
    untouched();
    // e4m3 operand-grid class on quantised, qkv and out_proj.
    quantised[0] = 0.1f;
    CHECK(swin_attention(quantised_v, qkv_v, head_scale_v, exp_bias_v, skip_v,
        attn_scale_v, out_proj_v, &out_v, out_cap, scratch, scratch_cap)
        == Status::kInvalidValues);
    quantised[0] = 0.5f;
    qkv[0] = 1.0625f;
    CHECK(swin_attention(quantised_v, qkv_v, head_scale_v, exp_bias_v, skip_v,
        attn_scale_v, out_proj_v, &out_v, out_cap, scratch, scratch_cap)
        == Status::kInvalidValues);
    qkv[0] = 1.0f;
    out_proj[0] = (float)NAN;
    CHECK(swin_attention(quantised_v, qkv_v, head_scale_v, exp_bias_v, skip_v,
        attn_scale_v, out_proj_v, &out_v, out_cap, scratch, scratch_cap)
        == Status::kInvalidValues);
    out_proj[0] = 0.5f;
    // Non-finite and non-f16 value classes on the auxiliary operands.
    head_scale[0] = (float)INFINITY;
    CHECK(swin_attention(quantised_v, qkv_v, head_scale_v, exp_bias_v, skip_v,
        attn_scale_v, out_proj_v, &out_v, out_cap, scratch, scratch_cap)
        == Status::kInvalidValues);
    head_scale[0] = 1e38f;  // finite but beyond the kMaxHeadScale bound
    CHECK(swin_attention(quantised_v, qkv_v, head_scale_v, exp_bias_v, skip_v,
        attn_scale_v, out_proj_v, &out_v, out_cap, scratch, scratch_cap)
        == Status::kInvalidValues);
    head_scale[0] = -70000.0f;
    CHECK(swin_attention(quantised_v, qkv_v, head_scale_v, exp_bias_v, skip_v,
        attn_scale_v, out_proj_v, &out_v, out_cap, scratch, scratch_cap)
        == Status::kInvalidValues);
    head_scale[0] = 1.0f;
    exp_bias[0] = (float)NAN;
    CHECK(swin_attention(quantised_v, qkv_v, head_scale_v, exp_bias_v, skip_v,
        attn_scale_v, out_proj_v, &out_v, out_cap, scratch, scratch_cap)
        == Status::kInvalidValues);
    exp_bias[0] = 1.28125f;
    skip[0] = (float)-INFINITY;
    CHECK(swin_attention(quantised_v, qkv_v, head_scale_v, exp_bias_v, skip_v,
        attn_scale_v, out_proj_v, &out_v, out_cap, scratch, scratch_cap)
        == Status::kInvalidValues);
    skip[0] = 1.5f;
    attn_scale[0] = 0.1f;  // not an f16 value
    CHECK(swin_attention(quantised_v, qkv_v, head_scale_v, exp_bias_v, skip_v,
        attn_scale_v, out_proj_v, &out_v, out_cap, scratch, scratch_cap)
        == Status::kInvalidValues);
    attn_scale[0] = 1.0f;
    untouched();
    // Dimension and pairing classes.
    CHECK(swin_attention(make_view(quantised, kRows, 0, kChannels + 1), qkv_v,
        head_scale_v, exp_bias_v, skip_v, attn_scale_v, out_proj_v, &out_v,
        out_cap, scratch, scratch_cap) == Status::kInvalidDimensions);
    CHECK(swin_attention(make_view(quantised, kRows, kChannels + 1,
        kChannels + 1), qkv_v, head_scale_v, exp_bias_v, skip_v, attn_scale_v,
        out_proj_v, &out_v, out_cap, scratch, scratch_cap)
        == Status::kInvalidDimensions);
    CHECK(swin_attention(make_view(quantised, kRows - 1, kChannels,
        kChannels + 1), qkv_v, head_scale_v, exp_bias_v, skip_v, attn_scale_v,
        out_proj_v, &out_v, out_cap, scratch, scratch_cap)
        == Status::kInvalidDimensions);
    CHECK(swin_attention(quantised_v, make_view(qkv, kQkvRows - 1, kChannels,
        kChannels), head_scale_v, exp_bias_v, skip_v, attn_scale_v, out_proj_v,
        &out_v, out_cap, scratch, scratch_cap) == Status::kInvalidDimensions);
    CHECK(swin_attention(quantised_v, make_view(qkv, kQkvRows, kChannels - 1,
        kChannels), head_scale_v, exp_bias_v, skip_v, attn_scale_v, out_proj_v,
        &out_v, out_cap, scratch, scratch_cap) == Status::kInvalidDimensions);
    CHECK(swin_attention(quantised_v, qkv_v,
        make_view(head_scale, 1, 1, 1), exp_bias_v, skip_v, attn_scale_v,
        out_proj_v, &out_v, out_cap, scratch, scratch_cap)
        == Status::kInvalidDimensions);
    CHECK(swin_attention(quantised_v, qkv_v, head_scale_v,
        make_view(exp_bias, kHeads - 1, dlss5_nr::kBiasTable,
            dlss5_nr::kBiasTable + 1),
        skip_v, attn_scale_v, out_proj_v, &out_v, out_cap, scratch, scratch_cap)
        == Status::kInvalidDimensions);
    CHECK(swin_attention(quantised_v, qkv_v, head_scale_v, exp_bias_v,
        make_view(skip, kRows - 1, kChannels, kChannels + 1), attn_scale_v,
        out_proj_v, &out_v, out_cap, scratch, scratch_cap)
        == Status::kInvalidDimensions);
    CHECK(swin_attention(quantised_v, qkv_v, head_scale_v, exp_bias_v, skip_v,
        make_view(attn_scale, 1, 1, 1), out_proj_v, &out_v, out_cap, scratch,
        scratch_cap) == Status::kInvalidDimensions);
    CHECK(swin_attention(quantised_v, qkv_v, head_scale_v, exp_bias_v, skip_v,
        attn_scale_v, make_view(out_proj, kChannels - 1, kChannels, kChannels),
        &out_v, out_cap, scratch, scratch_cap) == Status::kInvalidDimensions);
    MutableMatrixView bad_rows = make_out(out, kRows - 1, kChannels,
        kChannels + 1);
    CHECK(swin_attention(quantised_v, qkv_v, head_scale_v, exp_bias_v, skip_v,
        attn_scale_v, out_proj_v, &bad_rows, out_cap, scratch, scratch_cap)
        == Status::kInvalidDimensions);
    CHECK(swin_attention(make_view(quantised, kRows / 2, kChannels,
        kChannels + 1), qkv_v, head_scale_v, exp_bias_v, skip_v, attn_scale_v,
        out_proj_v, &out_v, out_cap, scratch, scratch_cap)
        == Status::kInvalidDimensions);
    untouched();
    // The qkv width bound: the projection writes 3 * C outputs per token, so
    // the linear slice's width bound (3 * C <= kMaxLinearDim 1024) caps the
    // assembly at heads <= 10. A full heads = 10 run (zero projections: the
    // output is the exact residual product skip * attn_scale) and a heads = 11
    // rejection pin the observable accept/reject boundary only: no status can
    // attribute the rejection to a specific check (the entry's heads bound, the
    // qkv pairing against 3 * channels and the qkv linear width cap all enforce
    // the same limit), so no single rejecting check is claimed.
    constexpr int kWideHeads = 10;
    constexpr int kWideChannels = kWideHeads * dlss5_nr::kHeadDim;
    constexpr int kWideQkvRows = 3 * kWideChannels;
    static float wide_quantised[kRows * (kWideChannels + 1)];
    static float wide_qkv[kWideQkvRows * kWideChannels];
    static float wide_head_scale[kWideHeads];
    static float wide_exp_bias[kWideHeads * dlss5_nr::kBiasTable];
    static float wide_skip[kRows * (kWideChannels + 1)];
    static float wide_attn_scale[kWideChannels];
    static float wide_out_proj[kWideChannels * kWideChannels];
    static float wide_out[kRows * (kWideChannels + 1)];
    static float wide_scratch[kRows * kWideQkvRows + 4 * kRows * kWideChannels
        + 2 * kWideHeads * dlss5_nr::kBiasTable + 2];
    memset(wide_quantised, 0, sizeof wide_quantised);
    memset(wide_qkv, 0, sizeof wide_qkv);
    memset(wide_out_proj, 0, sizeof wide_out_proj);
    for (int h = 0; h < kWideHeads; ++h) {
        wide_head_scale[h] = 1.0f;
        for (int p = 0; p < dlss5_nr::kBiasTable; ++p) {
            wide_exp_bias[h * dlss5_nr::kBiasTable + p] = 1.28125f;
        }
    }
    for (int p = 0; p < kRows * (kWideChannels + 1); ++p) {
        wide_skip[p] = 1.5f;
    }
    for (int c = 0; c < kWideChannels; ++c) {
        wide_attn_scale[c] = 1.0f;
    }
    fill_sentinel(wide_out, kRows * (kWideChannels + 1));
    const std::size_t wide_scratch_cap = kRows * kWideQkvRows
        + 4 * kRows * kWideChannels + 2 * kWideHeads * dlss5_nr::kBiasTable + 2;
    const MatrixView wide_quantised_v = make_view(wide_quantised, kRows,
        kWideChannels, kWideChannels + 1);
    const MatrixView wide_qkv_v = make_view(wide_qkv, kWideQkvRows,
        kWideChannels, kWideChannels);
    const MatrixView wide_head_scale_v = make_view(wide_head_scale, 1,
        kWideHeads, kWideHeads);
    const MatrixView wide_exp_bias_v = make_view(wide_exp_bias, kWideHeads,
        dlss5_nr::kBiasTable, dlss5_nr::kBiasTable);
    const MatrixView wide_skip_v = make_view(wide_skip, kRows, kWideChannels,
        kWideChannels + 1);
    const MatrixView wide_attn_scale_v = make_view(wide_attn_scale, 1,
        kWideChannels, kWideChannels);
    const MatrixView wide_out_proj_v = make_view(wide_out_proj, kWideChannels,
        kWideChannels, kWideChannels);
    MutableMatrixView wide_out_v = make_out(wide_out, kRows, kWideChannels,
        kWideChannels + 1);
    CHECK(swin_attention(wide_quantised_v, wide_qkv_v, wide_head_scale_v,
        wide_exp_bias_v, wide_skip_v, wide_attn_scale_v, wide_out_proj_v,
        &wide_out_v, kRows * (kWideChannels + 1), wide_scratch,
        wide_scratch_cap) == Status::kOk);
    for (int r = 0; r < kRows; ++r) {
        for (int c = 0; c < kWideChannels; ++c) {
            CHECK(wide_out[r * (kWideChannels + 1) + c] == 1.5f);
        }
        CHECK(wide_out[r * (kWideChannels + 1) + kWideChannels] == kSentinel);
    }
    // heads = 11 (3 * 352 = 1056 > kMaxLinearDim) is rejected with
    // kInvalidDimensions, pinning the observable boundary only. The call reuses
    // the heads = 10 wide buffers (960-row qkv, wide skip/projection widths),
    // and several checks would reject this class with the same status (the
    // entry's heads bound, the qkv pairing against 3 * channels and the qkv
    // linear width cap), so no specific rejecting check or check order is
    // claimed.
    static float too_wide[kRows * (11 * dlss5_nr::kHeadDim + 1)];
    for (std::size_t p = 0; p < kRows * (11 * dlss5_nr::kHeadDim + 1); ++p) {
        too_wide[p] = 0.5f;
    }
    CHECK(swin_attention(make_view(too_wide, kRows, 11 * dlss5_nr::kHeadDim,
        11 * dlss5_nr::kHeadDim + 1), wide_qkv_v, wide_head_scale_v,
        wide_exp_bias_v, wide_skip_v, wide_attn_scale_v, wide_out_proj_v,
        &wide_out_v, kRows * (kWideChannels + 1), wide_scratch,
        wide_scratch_cap) == Status::kInvalidDimensions);
    untouched();
    // Output stride bounds and both capacity classes.
    MutableMatrixView bad_stride = make_out(out, kRows, kChannels, kChannels - 1);
    CHECK(swin_attention(quantised_v, qkv_v, head_scale_v, exp_bias_v, skip_v,
        attn_scale_v, out_proj_v, &bad_stride, out_cap, scratch, scratch_cap)
        == Status::kInvalidStride);
    bad_stride = make_out(out, kRows, kChannels, dlss5_nr::kMaxLinearDim + 1);
    CHECK(swin_attention(quantised_v, qkv_v, head_scale_v, exp_bias_v, skip_v,
        attn_scale_v, out_proj_v, &bad_stride, out_cap, scratch, scratch_cap)
        == Status::kInvalidStride);
    MutableMatrixView ok_out = make_out(out, kRows, kChannels, kChannels + 1);
    CHECK(swin_attention(quantised_v, qkv_v, head_scale_v, exp_bias_v, skip_v,
        attn_scale_v, out_proj_v, &ok_out, out_needed - 1, scratch, scratch_cap)
        == Status::kInvalidBuffer);
    CHECK(swin_attention(quantised_v, qkv_v, head_scale_v, exp_bias_v, skip_v,
        attn_scale_v, out_proj_v, &ok_out, out_cap, scratch, needed - 1)
        == Status::kInvalidBuffer);
    untouched();
    // A full valid call still succeeds after the barrage, at the exact minimum
    // spans: the out span (rows - 1) * stride + cols and the scratch span.
    CHECK(swin_attention(quantised_v, qkv_v, head_scale_v, exp_bias_v, skip_v,
        attn_scale_v, out_proj_v, &ok_out, out_needed, scratch, needed)
        == Status::kOk);
}

static void check_zero_allocation(void) {
    using dlss5_nr::swin_attention;
    using dlss5_nr::swin_context;
    using dlss5_nr::swin_deswizzle_bias;
    using dlss5_nr::swin_exp_bias;
    using dlss5_nr::swin_logits;
    using dlss5_nr::swin_normalise_qkv;
    using dlss5_nr::swin_softmax;
    using dlss5_nr::swin_softmax_denominator;
    using dlss5_nr::swin_softmax_numerators;
    constexpr int kHeads = 2;
    constexpr int kChannels = kHeads * dlss5_nr::kHeadDim;
    constexpr int kRows = dlss5_nr::kWindowTokens;
    constexpr int kQkvRows = 3 * kChannels;
    constexpr std::ptrdiff_t kRowStride = dlss5_nr::kBiasTable + 2;
    static float raw[kHeads * kRowStride];
    static float bias_out[kHeads * kRowStride];
    static float tensor[kHeads * kRowStride];
    static float exp_bias[kHeads * kRowStride];
    static float dens[kRows * kHeads];
    static float query[kRows * (kChannels + 2)];
    static float key[kRows * (kChannels + 2)];
    static float value[kRows * (kChannels + 2)];
    static float head_scale[kHeads];
    static float out_q[kRows * (kChannels + 2)];
    static float out_k[kRows * (kChannels + 2)];
    static float out_v[kRows * (kChannels + 2)];
    static float quantised[kRows * (kChannels + 1)];
    static float qkv[kQkvRows * kChannels];
    static float skip[kRows * (kChannels + 1)];
    static float attn_scale[kChannels];
    static float out_proj[kChannels * kChannels];
    static float out[kRows * (kChannels + 1)];
    static float scratch[kRows * kQkvRows + 4 * kRows * kChannels
        + 2 * kHeads * dlss5_nr::kBiasTable + 4];
    for (std::size_t p = 0; p < kHeads * kRowStride; ++p) {
        raw[p] = 0.5f;
        bias_out[p] = 0.0f;
        tensor[p] = 0.25f;
        exp_bias[p] = 1.28125f;
    }
    for (int p = 0; p < kRows * kHeads; ++p) {
        dens[p] = 0.0f;
    }
    for (int p = 0; p < kRows * (kChannels + 2); ++p) {
        query[p] = 1.5f;
        key[p] = 2.0f;
        value[p] = 4.0f;
        out_q[p] = 0.0f;
        out_k[p] = 0.0f;
        out_v[p] = 0.0f;
    }
    for (int p = 0; p < kRows * (kChannels + 1); ++p) {
        quantised[p] = 0.5f;
        skip[p] = 1.5f;
        out[p] = 0.0f;
    }
    for (int p = 0; p < kQkvRows * kChannels; ++p) {
        qkv[p] = 1.0f;
    }
    for (int p = 0; p < kChannels; ++p) {
        head_scale[p / dlss5_nr::kHeadDim] = 1.0f;
        attn_scale[p] = 1.0f;
    }
    for (int p = 0; p < kChannels * kChannels; ++p) {
        out_proj[p] = 0.5f;
    }
    const std::size_t row_cap = kRows * (kChannels + 2);
    const MatrixView raw_v = make_view(raw, kHeads, dlss5_nr::kBiasTable,
        kRowStride);
    MutableMatrixView bias_out_v = make_out(bias_out, kHeads,
        dlss5_nr::kBiasTable, kRowStride);
    const std::ptrdiff_t kHeadStride = kRowStride;
    const AttentionTensorView tensor_v = make_tensor(tensor, 1, kHeads,
        kHeads * kHeadStride, kHeadStride);
    const MatrixView exp_bias_v = make_view(exp_bias, kHeads,
        dlss5_nr::kBiasTable, kRowStride);
    MutableAttentionTensorView tensor_out_v = make_mutable_tensor(tensor, 1,
        kHeads, kHeads * kHeadStride, kHeadStride);
    const MatrixView query_v = make_view(query, kRows, kChannels,
        kChannels + 2);
    const MatrixView key_v = make_view(key, kRows, kChannels, kChannels + 2);
    const MatrixView value_v = make_view(value, kRows, kChannels,
        kChannels + 2);
    const MatrixView head_scale_v = make_view(head_scale, 1, kHeads, kHeads);
    MutableMatrixView oq = make_out(out_q, kRows, kChannels, kChannels + 2);
    MutableMatrixView ok = make_out(out_k, kRows, kChannels, kChannels + 2);
    MutableMatrixView ov = make_out(out_v, kRows, kChannels, kChannels + 2);
    const MatrixView quantised_v = make_view(quantised, kRows, kChannels,
        kChannels + 1);
    const MatrixView qkv_v = make_view(qkv, kQkvRows, kChannels, kChannels);
    const MatrixView skip_v = make_view(skip, kRows, kChannels, kChannels + 1);
    const MatrixView attn_scale_v = make_view(attn_scale, 1, kChannels,
        kChannels);
    const MatrixView out_proj_v = make_view(out_proj, kChannels, kChannels,
        kChannels);
    MutableMatrixView attn_out_v = make_out(out, kRows, kChannels,
        kChannels + 1);
    const unsigned long long before = g_allocations.load();
    for (int i = 0; i < 64; ++i) {
        CHECK(swin_deswizzle_bias(raw_v, &bias_out_v,
            kHeads * kRowStride) == Status::kOk);
        CHECK(swin_exp_bias(raw_v, &bias_out_v,
            kHeads * kRowStride) == Status::kOk);
        CHECK(swin_softmax_numerators(tensor_v, exp_bias_v, &tensor_out_v,
            kHeads * kRowStride) == Status::kOk);
        CHECK(swin_softmax_denominator(tensor_v, dens,
            kRows * kHeads) == Status::kOk);
        CHECK(swin_softmax(tensor_v, exp_bias_v, &tensor_out_v,
            kHeads * kRowStride) == Status::kOk);
        CHECK(swin_normalise_qkv(query_v, key_v, value_v, head_scale_v, kHeads,
            &oq, row_cap, &ok, row_cap, &ov, row_cap) == Status::kOk);
        CHECK(swin_logits(query_v, key_v, kHeads, &tensor_out_v,
            kHeads * kRowStride) == Status::kOk);
        CHECK(swin_context(tensor_v, value_v, &ov, row_cap) == Status::kOk);
        CHECK(swin_attention(quantised_v, qkv_v, head_scale_v, exp_bias_v,
            skip_v, attn_scale_v, out_proj_v, &attn_out_v,
            kRows * (kChannels + 1), scratch, kRows * kQkvRows
                + 4 * kRows * kChannels + 2 * kHeads * dlss5_nr::kBiasTable + 4)
            == Status::kOk);
    }
    CHECK(g_allocations.load() == before);
}

int main(void) {
    check_deswizzle_index_pins();
    check_deswizzle_bias_pins();
    check_exp_bias_pins();
    check_numerator_pins();
    check_denominator_pins();
    check_softmax_row_pins();
    check_normalise_head_row_pins();
    check_logit_pins();
    check_context_pins();
    check_batch_equivalence();
    check_attention_assembly();
    check_negative_bias_tables();
    check_negative_softmax_family();
    check_negative_normalise();
    check_negative_logits_context();
    check_negative_attention();
    check_zero_allocation();
    printf("native-swin-attention-test: %llu checks passed\n", g_checks);
    return 0;
}
