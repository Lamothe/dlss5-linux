// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Michael Lamothe
//
// Known-answer tests for the int4 mixed-network operand path
// (native-int4.hpp), pinned to dlssnr/int4.py at the campaign's MIT commit.
// Hand-derived answers: the unnormalised Hadamard butterflies (fwht([1,2,3,4])
// = [10,-2,-4,0]), the 1.5 * 2^23 integer-rounding trick's ties-to-even
// (2.5 -> 2, 3.5 -> 4, -1.5 -> -2, -0.5 -> 0), the clamp(rint, 0, 15) - 8
// code form, the exact integer dot with the float64 affine, the binary16
// hop of the half form, the permuted mixed int4 + e4m3-tail contraction, and
// the Swin site's hard-swish and fold-constant chain.
//
// Built by vr/test-native-int4.sh at -O2 and -O0 with -ffp-contract=off.
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>

#include "native-int4.hpp"

using namespace dlss5_nr;

namespace {

long long g_checks = 0;

void check(bool ok, const char* expression, int line) {
    ++g_checks;
    if (!ok) {
        std::fprintf(stderr, "FAILED (%d): %s\n", line, expression);
        std::exit(1);
    }
}

#define CHECK(expression) check((expression), #expression, __LINE__)

void check_status(Status got, Status want, const char* expression, int line) {
    ++g_checks;
    if (got != want) {
        std::fprintf(stderr, "FAILED (%d): %s got %d want %d\n", line,
                     expression, static_cast<int>(got), static_cast<int>(want));
        std::exit(1);
    }
}

#define CHECK_STATUS(expression, want) \
    check_status((expression), (want), #expression " == " #want, __LINE__)

std::atomic<unsigned long long> g_allocations{0};

}  // namespace

void* operator new(std::size_t n) {
    g_allocations.fetch_add(1);
    void* p = std::malloc(n);
    if (p == nullptr) {
        std::abort();
    }
    return p;
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void* operator new(std::size_t n, std::align_val_t a) {
    g_allocations.fetch_add(1);
    const std::size_t alignment = static_cast<std::size_t>(a);
    const std::size_t rounded = (n + alignment - 1) / alignment * alignment;
    void* p = std::aligned_alloc(alignment, rounded == 0 ? alignment : rounded);
    if (p == nullptr) {
        std::abort();
    }
    return p;
}
void* operator new[](std::size_t n, std::align_val_t a) { return ::operator new(n, a); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }

int main() {
    // ---- the unnormalised Walsh-Hadamard butterflies -------------------------
    {
        float one[1] = {7.5f};
        CHECK_STATUS(int4_fwht(one, one, 1, 1), Status::kOk);
        CHECK(one[0] == 7.5f);  // block 1 is the identity

        float two[2] = {1.0f, 3.0f};
        CHECK_STATUS(int4_fwht(two, two, 2, 2), Status::kOk);
        CHECK(two[0] == 4.0f && two[1] == -2.0f);

        // h=1 -> [3,-1,7,-1]; h=2 pairs (0,2),(1,3) -> [10,-2,-4,0].
        float four[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        CHECK_STATUS(int4_fwht(four, four, 4, 4), Status::kOk);
        CHECK(four[0] == 10.0f && four[1] == -2.0f);
        CHECK(four[2] == -4.0f && four[3] == 0.0f);

        // Two independent groups of 4.
        float eight[8] = {1, 2, 3, 4, -1, 1, 0.5f, 0.25f};
        float out[8];
        CHECK_STATUS(int4_fwht(eight, out, 8, 4), Status::kOk);
        CHECK(out[0] == 10.0f && out[1] == -2.0f);
        CHECK(out[2] == -4.0f && out[3] == 0.0f);
        // group 2 [-1,1,0.5,0.25]: h=1 -> [0,-2,0.75,0.25]; h=2 pairs
        // (0,0.75)->(0.75,-0.75) and (-2,0.25)->(-1.75,-2.25).
        CHECK(out[4] == 0.75f && out[5] == -1.75f);
        CHECK(out[6] == -0.75f && out[7] == -2.25f);
    }

    // ---- the 1.5 * 2^23 integer rounding (ties to even) ---------------------
    CHECK(int4_detail::round_to_integer(2.5f) == 2.0f);
    CHECK(int4_detail::round_to_integer(3.5f) == 4.0f);
    CHECK(int4_detail::round_to_integer(-1.5f) == -2.0f);
    CHECK(int4_detail::round_to_integer(-0.5f) == 0.0f);
    CHECK(int4_detail::round_to_integer(7.0f) == 7.0f);

    // ---- quantiser parameters: gain/zero derivation -------------------------
    {
        std::int8_t q[4] = {1, -1, 0, 0};
        float scale[2] = {1.0f, 1.0f};
        float constant[2] = {0.0f, 0.0f};
        // block 1, non-half: gain = inverse; zero = magic-rounded.
        float aq[4] = {2.0f, 2.5f, 0.1f, -1.5f};
        Int4MatrixView matrix{q, scale, constant, aq, 2, 2, 1, false};
        float gain[2] = {};
        float zero[2] = {};
        CHECK_STATUS(int4_quantiser_params(matrix, gain, zero), Status::kOk);
        CHECK(gain[0] == 2.0f && zero[0] == 2.0f);
        CHECK(gain[1] == 0.1f && zero[1] == -2.0f);

        // block 4: gain = inverse * (1/sqrt(4)) = inverse * 0.5 exactly.
        float aq4[8] = {2.0f, 1.5f, 4.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
        Int4MatrixView matrix4{q, scale, constant, aq4, 2, 4, 4, false};
        float gain4[4] = {};
        float zero4[4] = {};
        CHECK_STATUS(int4_quantiser_params(matrix4, gain4, zero4), Status::kOk);
        CHECK(gain4[0] == 1.0f && zero4[0] == 2.0f);
        CHECK(gain4[1] == 2.0f && zero4[1] == 0.0f);

        // The half form: binary16 hops on both values, no magic round.
        float aqh[4] = {0.1f, 0.049f, 1.0f, 2.5f};
        Int4MatrixView matrixh{q, scale, constant, aqh, 2, 2, 1, true};
        float gainh[2] = {};
        float zeroh[2] = {};
        CHECK_STATUS(int4_quantiser_params(matrixh, gainh, zeroh), Status::kOk);
        CHECK(gainh[0] == f16(0.1f));
        CHECK(zeroh[0] == f16(0.049f));
        CHECK(zeroh[1] == 2.5f);  // kept, not integer-rounded
    }

    // ---- quantise: clamp(rint, 0, 15) - 8 ----------------------------------
    {
        std::int8_t q[4] = {1, -1, 0, 0};
        float scale[1] = {1.0f};
        float constant[1] = {0.0f};
        float aq[8] = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
        Int4MatrixView matrix{q, scale, constant, aq, 1, 4, 1, false};
        const float value[4] = {1.5f, 2.5f, -10.0f, 100.0f};
        float codes[4] = {};
        CHECK_STATUS(int4_quantise(matrix, value, 1, codes), Status::kOk);
        // rint half-even: 1.5 -> 2, 2.5 -> 2; clamped: 0 and 15.
        CHECK(codes[0] == -6.0f && codes[1] == -6.0f);
        CHECK(codes[2] == -8.0f && codes[3] == 7.0f);

        // NaN maps to code 0 (-8); +inf saturates to 15 (7).
        std::int8_t qsp[2] = {0, 0};
        float aqsp[4] = {1.0f, 0.0f, 1.0f, 0.0f};
        Int4MatrixView sp{qsp, scale, constant, aqsp, 1, 2, 1, false};
        const float special[2] = {std::nanf(""), std::numeric_limits<float>::infinity()};
        float special_codes[2] = {};
        CHECK_STATUS(int4_quantise(sp, special, 1, special_codes), Status::kOk);
        CHECK(special_codes[0] == -8.0f);
        CHECK(special_codes[1] == 7.0f);

        // The Hadamard runs first: [1,3] -> [4,-2] with gain 1/sqrt(2).
        std::int8_t q2[4] = {1, 1, 1, 1};
        float aq2[4] = {1.0f, 0.0f, 1.0f, 0.0f};
        Int4MatrixView matrix2{q2, scale, constant, aq2, 1, 2, 2, false};
        const float value2[2] = {1.0f, 3.0f};
        float codes2[2];
        CHECK_STATUS(int4_quantise(matrix2, value2, 1, codes2), Status::kOk);
        // 4 / sqrt(2) = 2.828... -> rint 3 -> code -5.
        CHECK(codes2[0] == -5.0f);
        // rint(-2/sqrt2) = -1 clamps to 0 before the -8.
        CHECK(codes2[1] == -8.0f);
    }

    // ---- accumulate: exact int dot, float64 affine, half hop ---------------
    {
        std::int8_t q[3] = {1, -8, 7};
        float scale[1] = {0.5f};
        float constant[1] = {0.25f};
        float aq[6] = {1, 0, 1, 0, 1, 0};
        Int4MatrixView matrix{q, scale, constant, aq, 1, 3, 1, false};
        const float codes[3] = {2.0f, 3.0f, -4.0f};
        float out[1] = {};
        CHECK_STATUS(int4_accumulate(matrix, codes, 1, out), Status::kOk);
        // 1*2 + (-8)*3 + 7*(-4) = -50; -50*0.5 + 0.25 = -24.75.
        CHECK(out[0] == -24.75f);

        // The half form rounds s and c through binary16 first: s = 0.1f ->
        // f16 0.0999755859375, c = 0.049f -> f16 0.04901123046875, so
        // -50 * s_h + c_h = -4.94976806640625 exactly. The non-half form
        // would give about -4.951: the hop is observable.
        float scale_h[1] = {0.1f};
        float constant_h[1] = {0.049f};
        Int4MatrixView half{q, scale_h, constant_h, aq, 1, 3, 1, true};
        float outh[1] = {};
        CHECK_STATUS(int4_accumulate(half, codes, 1, outh), Status::kOk);
        CHECK(outh[0] == -4.94976806640625f);
    }

    // ---- the mixed int4 + e4m3-tail contraction -----------------------------
    {
        // perm [1, 0, 3, 2]: int4 channels are hidden 1, 0; tail channels
        // hidden 3, 2. q row [2, -1], codes [3, 5] -> int4 sum 1; s 1, c 0.5
        // -> 1.5; tail [2, 4] over e4m3 [0.25, 0.5, 0.75, 1.0]:
        // 1.0*2 + 0.75*4 = 5. Total 1.5 + 2 + 3 = 6.5 (left to right).
        std::int8_t q[2] = {2, -1};
        float scale[1] = {1.0f};
        float constant[1] = {0.5f};
        float aq[4] = {1, 0, 1, 0};
        Int4MatrixView matrix{q, scale, constant, aq, 1, 2, 1, false};
        const float codes[2] = {3.0f, 5.0f};
        const float tail[2] = {2.0f, 4.0f};
        const float e4m3[4] = {0.25f, 0.5f, 0.75f, 1.0f};
        const int perm[4] = {1, 0, 3, 2};
        float out[1] = {};
        CHECK_STATUS(int4_mixed_contraction(matrix, codes, tail, 2, e4m3, perm,
                                            1, out), Status::kOk);
        CHECK(out[0] == 6.5f);
    }

    // ---- hard-swish: the gate clamp and the fma form ------------------------
    {
        const float in[4] = {0.0f, 1.0f, 100.0f, -100.0f};
        float out[4];
        CHECK_STATUS(hard_swish(in, 4, 32, out), Status::kOk);
        CHECK(out[0] == 0.0f);
        const float gate1 = static_cast<float>(
            static_cast<double>(0.187367f) + static_cast<double>(0.500139f));
        CHECK(out[1] == gate1);
        CHECK(out[2] == 100.0f);   // gate clamps to 1
        CHECK(out[3] == 0.0f);     // gate clamps to 0
        float outw[4];
        CHECK_STATUS(hard_swish(in, 4, 64, outw), Status::kOk);
        CHECK(outw[1] != out[1]);  // the wide (A, B) differ
        CHECK(outw[2] == 100.0f && outw[3] == 0.0f);
    }

    // ---- the Swin fold-constant chain --------------------------------------
    {
        const int listed[1] = {3};
        SwinFoldSettings none;
        CHECK(swin_fold_constant(none, 5, 64) == 0.0f);

        SwinFoldSettings wide;
        wide.wide = 0.5f;
        CHECK(swin_fold_constant(wide, 5, 64) == 0.5f);
        CHECK(swin_fold_constant(wide, 5, 32) == 0.5f);  // narrow defaults to wide
        CHECK(swin_fold_constant(wide, 0, 64) == 0.5f);  // first/last to narrow
        CHECK(swin_fold_constant(wide, 70, 32) == 0.5f);

        SwinFoldSettings narrow;
        narrow.wide = 0.5f;
        narrow.narrow = 0.25f;
        narrow.has_narrow = true;
        CHECK(swin_fold_constant(narrow, 5, 32) == 0.25f);
        CHECK(swin_fold_constant(narrow, 5, 64) == 0.5f);
        CHECK(swin_fold_constant(narrow, 0, 64) == 0.25f);  // first/last -> narrow

        SwinFoldSettings all;
        all.wide = 0.5f;
        all.narrow = 0.25f;
        all.has_narrow = true;
        all.first_last = 0.125f;
        all.has_first_last = true;
        CHECK(swin_fold_constant(all, 0, 64) == 0.125f);
        CHECK(swin_fold_constant(all, 70, 64) == 0.125f);

        SwinFoldSettings gated;
        gated.wide = 0.5f;
        gated.first_last = 0.125f;
        gated.has_first_last = true;
        gated.fold_blocks = listed;
        gated.fold_block_count = 1;
        CHECK(swin_fold_constant(gated, 3, 64) == 0.5f);
        CHECK(swin_fold_constant(gated, 5, 64) == 0.0f);
        // Gating beats the first/last selection: unlisted blocks 0/70 drop.
        CHECK(swin_fold_constant(gated, 0, 64) == 0.0f);
        CHECK(swin_fold_constant(gated, 70, 64) == 0.0f);
        // A present-but-empty listed set folds nowhere; a null list with a
        // positive count is treated as absent (fold everywhere).
        SwinFoldSettings empty;
        empty.wide = 0.5f;
        empty.fold_blocks = listed;
        empty.fold_block_count = 0;
        CHECK(swin_fold_constant(empty, 5, 64) == 0.0f);
        SwinFoldSettings null_list;
        null_list.wide = 0.5f;
        null_list.fold_block_count = 1;
        CHECK(swin_fold_constant(null_list, 5, 64) == 0.5f);
    }

    // ---- rejections leave the outputs untouched -----------------------------
    {
        // q is accepted as int8 (the pinned's escape corrections wrap on the
        // astype); the bounds are what reject.
        std::int8_t q[2] = {9, -9};
        float scale[1] = {1.0f};
        float constant[1] = {0.0f};
        float aq[4] = {1, 0, 1, 0};
        Int4MatrixView bad{q, scale, constant, aq, 1, 2, 1, false};
        const float codes[2] = {0.0f, 0.0f};
        float out[1] = {-777.0f};
        CHECK_STATUS(int4_accumulate(bad, codes, 1, out), Status::kOk);
        out[0] = -777.0f;  // re-arm the sentinel for the rejections
        Int4MatrixView wide{q, scale, constant, aq, 1, 20000, 1, false};
        CHECK_STATUS(int4_accumulate(wide, codes, 1, out),
                     Status::kInvalidDimensions);
        std::int8_t good[2] = {1, -1};
        Int4MatrixView matrix{good, scale, constant, aq, 1, 2, 1, false};
        CHECK_STATUS(int4_accumulate(matrix, codes, 0, out),
                     Status::kInvalidDimensions);
        CHECK_STATUS(int4_accumulate(matrix, nullptr, 1, out),
                     Status::kNullPointer);
        Int4MatrixView odd{good, scale, constant, aq, 1, 2, 3, false};
        CHECK_STATUS(int4_quantise(odd, codes, 1, out), Status::kInvalidValues);
        Int4MatrixView nulls{nullptr, scale, constant, aq, 1, 2, 1, false};
        CHECK_STATUS(int4_quantise(nulls, codes, 1, out), Status::kNullPointer);
        CHECK_STATUS(int4_fwht(nullptr, out, 2, 2), Status::kNullPointer);
        CHECK_STATUS(int4_fwht(codes, out, 3, 2), Status::kInvalidDimensions);
        float big[2] = {1.0f, 2.0f};
        CHECK_STATUS(hard_swish(big, 2, 0, out), Status::kInvalidDimensions);
        for (int i = 0; i < 1; ++i) {
            CHECK(out[i] == -777.0f);
        }
    }

    // ---- determinism and zero-allocation accounting -------------------------
    {
        std::int8_t q[4] = {1, -2, 3, -4};
        float scale[2] = {0.5f, -0.25f};
        float constant[2] = {0.125f, 0.375f};
        float aq[4] = {1.5f, 0.25f, 0.75f, -0.5f};
        Int4MatrixView matrix{q, scale, constant, aq, 2, 2, 1, false};
        const float value[4] = {0.5f, -1.25f, 2.0f, 3.5f};
        float codes[4] = {};
        float out[2] = {};
        const unsigned long long before = g_allocations.load();
        CHECK_STATUS(int4_quantise(matrix, value, 2, codes), Status::kOk);
        CHECK_STATUS(int4_accumulate(matrix, codes, 2, out), Status::kOk);
        float codes2[4];
        float out2[2] = {};
        CHECK_STATUS(int4_quantise(matrix, value, 2, codes2), Status::kOk);
        CHECK_STATUS(int4_accumulate(matrix, codes2, 2, out2), Status::kOk);
        CHECK(std::memcmp(codes, codes2, sizeof(codes)) == 0);
        CHECK(std::memcmp(out, out2, sizeof(out)) == 0);
        CHECK(g_allocations.load() == before);
    }

    std::printf("native-int4: %lld checks passed\n", g_checks);
    return 0;
}
