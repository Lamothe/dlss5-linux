// Independent known-answer tests for vr/native-decoder.hpp — the native CPU
// decoder input stage (`DecInputUpsample`). Hand-derived answers only (every
// binary16/e4m3 step derived on paper from the pinned reference semantics); no
// Python, no torch, no proprietary weights.
//
// Controls: the gated element (including a pre-rounding discriminator and a
// product tie), the q8 store (including the binary16-hop double-rounding
// discriminator, saturation and NaN canonicalisation), end-to-end kernel
// answers with e4m3 ties in the store, the crop-to-skip-extent semantics (a
// crop/replicate-extent discriminator: rejects, not clamps), padded-stride
// reads with poisoned padding (catching kernel over-reads and validation
// over-scans), the overflow/NaN edges, sentinel-checked untouched out/scratch
// on every validation rejection class, determinism and zero-allocation
// accounting. See vr/NATIVE-DECODER-INPUT.md.

#include "../native-decoder.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <atomic>
#include <new>

using namespace dlss5_nr;

static std::atomic<unsigned long long> g_allocations{0};
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

static void check_status(Status got, Status want, const char* expr, int line) {
    ++g_checks;
    if (got != want) {
        std::fprintf(stderr, "FAILED (%d): %s got %d want %d\n", line, expr,
                     static_cast<int>(got), static_cast<int>(want));
        std::exit(1);
    }
}

static std::uint32_t float_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof bits);
    return bits;
}

static void check_bits(float got, std::uint32_t want, const char* expr,
                       int line) {
    ++g_checks;
    if (float_bits(got) != want) {
        std::fprintf(stderr, "FAILED (%d): %s got bits 0x%08x want 0x%08x\n",
                     line, expr, float_bits(got), want);
        std::exit(1);
    }
}

#define CHECK(expr) check((expr), #expr, __LINE__)
#define CHECK_CLOSE(got, want, tol) \
    check_close((got), (want), (tol), #got " ~= " #want, __LINE__)
#define CHECK_STATUS(got, want) \
    check_status((got), (want), #got " == " #want, __LINE__)
#define CHECK_BITS(got, want) \
    check_bits((got), (want), #got " bits == " #want, __LINE__)

static constexpr float kSentinel = -12345.5f;
static constexpr std::size_t kStorage = 64;

static void fill_sentinel(float* p) {
    for (std::size_t i = 0; i < kStorage; ++i) {
        p[i] = kSentinel;
    }
}

static void expect_sentinel(const float* p, std::size_t n, int line) {
    for (std::size_t i = 0; i < n; ++i) {
        ++g_checks;
        if (p[i] != kSentinel) {
            std::fprintf(stderr, "FAILED (%d): sentinel cell %zu modified: %.9g\n",
                         line, i, static_cast<double>(p[i]));
            std::exit(1);
        }
    }
}

static void expect_values(const float* p, const float* want, std::size_t n,
                          const char* what, int line) {
    for (std::size_t i = 0; i < n; ++i) {
        ++g_checks;
        if (p[i] != want[i]) {
            std::fprintf(stderr, "FAILED (%d): %s cell %zu got %.9g want %.9g\n",
                         line, what, i, static_cast<double>(p[i]),
                         static_cast<double>(want[i]));
            std::exit(1);
        }
    }
}

int main(void) {
    // ---- dec_gated_element anchors (f16(f16(skip) * gain)) -----------------------------------------
    // Basic exact products (both factors binary16, the product exact in
    // float32, one rounding that changes nothing):
    CHECK_CLOSE(dec_gated_element(2.0f, 0.5f), 1.0f, 0.0f);
    CHECK_CLOSE(dec_gated_element(-3.0f, 2.0f), -6.0f, 0.0f);
    CHECK_CLOSE(dec_gated_element(1.5f, 3.0f), 4.5f, 0.0f);
    CHECK_CLOSE(dec_gated_element(0.0f, 5.0f), 0.0f, 0.0f);
    // Pre-rounding discriminator: skip = 1 + 511 * 2^-20
    // (1.00048732757568359375, exactly 1049087 * 2^-20, 21 significant bits).
    // f16(skip): the grid of [1, 2) steps 2^-10 with midpoint 1 + 2^-11 =
    // 1 + 512 * 2^-20; the value sits 2^-20 below the midpoint and rounds
    // DOWN to 1.0. gain = 1.5 (binary16).
    //   correct:   f16(1.0 * 1.5)            = 1.5
    //   no f16(skip): f16(1.5 * skip) = f16(3147261 * 2^-21) = 1.5 + 2^-10
    //              (the product is exact; it sits 1533 * 2^-21 above 1.5,
    //               past the midpoint 1024 * 2^-21, so it rounds UP)
    CHECK_CLOSE(dec_gated_element(1.00048732757568359375f, 1.5f), 1.5f, 0.0f);
    // Product tie: 1.5 * (683 * 2^-10) = 2049 * 2^-11 = 1 + 2^-11 exactly,
    // the midpoint of 1.0 and 1 + 2^-10; ties to even selects the even
    // significand 1.0 (the upper candidate's significand is odd).
    CHECK_CLOSE(dec_gated_element(1.5f, 0.6669921875f), 1.0f, 0.0f);
    // Overflow edges: f16 pre-rounding overflows to infinity (>= 65520) and
    // the product of the two largest finite binary16 values (65504^2 =
    // 4.29e9, exact in float32) rounds to infinity.
    CHECK(std::isinf(dec_gated_element(70000.0f, 1.0f)));
    CHECK(dec_gated_element(70000.0f, 1.0f) > 0.0f);
    CHECK(std::isinf(dec_gated_element(65504.0f, -65504.0f)));
    CHECK(dec_gated_element(65504.0f, -65504.0f) < 0.0f);
    // infinity * zero is NaN, canonicalised by the binary16 rounding to a
    // quiet NaN with zero payload. f16_bits preserves the raw product's sign
    // (the default NaN sign is platform-defined: negative on x86), so the
    // canonical magnitude is asserted and the sign is left to the platform,
    // matching the blend slice's "canonical quiet NaN" policy.
    CHECK(std::isnan(dec_gated_element(70000.0f, 0.0f)));
    CHECK((float_bits(dec_gated_element(70000.0f, 0.0f)) & 0x7fffffffu)
          == 0x7fc00000u);
    // The in-kernel overflow rounding: the exact product 65504 * 65504
    // saturates the binary16 rounding to +infinity.
    CHECK(std::isinf(dec_gated_element(65504.0f, 65504.0f)));

    // ---- dec_store anchors (q8(f16(value))) ---------------------------------------------------------
    CHECK_CLOSE(dec_store(0.0f), 0.0f, 0.0f);
    CHECK_CLOSE(dec_store(2.5f), 2.5f, 0.0f);
    CHECK_CLOSE(dec_store(-2.5f), -2.5f, 0.0f);
    // Binary16-hop double-rounding discriminator: 68.00390625 = 68 + 2^-8
    // rounds to binary16 68.0 (the [64, 128) grid steps 2^-4; the midpoint
    // is 68.03125) and 68.0 is the e4m3 midpoint of 64 and 72, a tie the
    // even significand 64 wins. Direct e4m3 rounding of the same value skips
    // the tie and lands on 72; the store must land on 64.
    CHECK_CLOSE(dec_store(68.00390625f), 64.0f, 0.0f);
    CHECK_CLOSE(e4m3(68.00390625f), 72.0f, 0.0f);
    // e4m3 ties to even through the store: 9.5 -> 10 (even), 10.5 -> 10
    // (even), 11.5 -> 12 (even); the [8, 16) grid steps 1.
    CHECK_CLOSE(dec_store(9.5f), 10.0f, 0.0f);
    CHECK_CLOSE(dec_store(10.5f), 10.0f, 0.0f);
    CHECK_CLOSE(dec_store(11.5f), 12.0f, 0.0f);
    // Saturation: the binary16 hop overflows to infinity and the store
    // clamps exactly as q8 (+-448).
    CHECK_CLOSE(dec_store(1e30f), 448.0f, 0.0f);
    CHECK_CLOSE(dec_store(-1e30f), -448.0f, 0.0f);
    CHECK_CLOSE(dec_store(65504.0f), 448.0f, 0.0f);
    // Subnormal store: f16(0.001f) = 1049 * 2^-20 = 0.00100040435791015625
    // (the [2^-10, 2^-9) grid steps 2^-20; 0.001f * 2^20 = 1048.576... ->
    // 1049), then the e4m3 subnormal grid m * 2^-9: scaled = 0.5122... ->
    // m = 1 -> 2^-9 = 0.001953125.
    CHECK_CLOSE(dec_store(0.001f), 0.001953125f, 0.0f);
    // NaN is canonicalised to the e4m3 NaN code 0x7f exactly as q8 ("NaN is
    // not preserved"): the result is the canonical quiet NaN 0x7fc00000.
    {
        const float nan_value = std::numeric_limits<float>::quiet_NaN();
        CHECK_BITS(dec_store(nan_value), 0x7fc00000u);
    }

    // ---- K1: end-to-end known answer (exact arithmetic, e4m3 store ties) ---------------------------
    // c_in = 2, c_out = 2; x [2, 2, 2] tight; skip [2, 3, 3] tight (3 <= 2*2,
    // so the 4x4 replicated map is cropped to 3x3).
    //   W rows: {1, 2} (codes 0x38, 0x40), {0.5, -1} (codes 0x30, 0xB8)
    //   x plane 0 = {1, 2, 3, 4}; plane 1 = {4, 0, -2, 1}
    //   proj[c][r][col] = x0 * W[c][0] + x1 * W[c][1] (exact):
    //     proj0 = {9, 2, -1, 6}; proj1 = {-3.5, 1, 3.5, 1}
    //   gain = {0.5, 2.0}; skip plane 0 = {1..9}, plane 1 = 0.
    //   out (y, col) = e4m3(f16(proj[y>>1][col>>1] + f16(skip * gain))):
    //     ch0 sums: 9.5, 10, 3.5, 11, 11.5, 5, 2.5, 3, 10.5
    //       -> e4m3: 10 (tie, even), 10, 3.5, 11, 12 (tie, even), 5,
    //                2.5, 3, 10 (tie, even)
    //     ch1 sums = proj1 replicated (gated = 0):
    //       -> {-3.5, -3.5, 1, -3.5, -3.5, 1, 3.5, 3.5, 1}
    {
        const float x_data[8] = {1.0f, 2.0f, 3.0f, 4.0f,
                                 4.0f, 0.0f, -2.0f, 1.0f};
        const std::uint8_t w_codes[4] = {0x38, 0x40, 0x30, 0xB8};
        const float gain[2] = {0.5f, 2.0f};
        const float skip_data[18] = {1.0f, 2.0f, 3.0f,
                                     4.0f, 5.0f, 6.0f,
                                     7.0f, 8.0f, 9.0f,
                                     0.0f, 0.0f, 0.0f,
                                     0.0f, 0.0f, 0.0f,
                                     0.0f, 0.0f, 0.0f};
        FeatureBatchView x;
        x.data = x_data;
        x.channels = 2;
        x.rows = 2;
        x.cols = 2;
        x.plane_stride = 4;
        x.row_stride = 2;
        FeatureBatchView skip;
        skip.data = skip_data;
        skip.channels = 2;
        skip.rows = 3;
        skip.cols = 3;
        skip.plane_stride = 9;
        skip.row_stride = 3;
        DecInputWeight w;
        w.weight.bits = w_codes;
        w.weight.rows = 2;
        w.weight.cols = 2;
        w.gain = gain;
        w.gain_count = 2;
        float out[kStorage];
        float scratch[kStorage];
        fill_sentinel(out);
        fill_sentinel(scratch);
        CHECK_STATUS(dec_input_upsample(x, skip, w, out, 18, scratch, 8),
                     Status::kOk);
        const float want_scratch[8] = {9.0f, 2.0f, -1.0f, 6.0f,
                                       -3.5f, 1.0f, 3.5f, 1.0f};
        expect_values(scratch, want_scratch, 8, "K1 scratch", __LINE__);
        const float want_out[18] = {10.0f, 10.0f, 3.5f,
                                    11.0f, 12.0f, 5.0f,
                                    2.5f, 3.0f, 10.0f,
                                    -3.5f, -3.5f, 1.0f,
                                    -3.5f, -3.5f, 1.0f,
                                    3.5f, 3.5f, 1.0f};
        expect_values(out, want_out, 18, "K1 out", __LINE__);
        expect_sentinel(out + 18, kStorage - 18, __LINE__);
        expect_sentinel(scratch + 8, kStorage - 8, __LINE__);
    }

    // ---- K2: the double-rounding discriminator through the kernel ---------------------------------
    // up = 64 (projection exact), gated = f16(4.00390625 * 1) = 4.00390625
    // (a binary16 value: the [4, 8) grid steps 2^-8), sum = 68.00390625 ->
    // f16 68.0 -> e4m3 tie -> 64. A direct e4m3 store would give 72.
    {
        const float x_data[1] = {64.0f};
        const std::uint8_t w_codes[1] = {0x38};
        const float gain[1] = {1.0f};
        const float skip_data[1] = {4.00390625f};
        FeatureBatchView x;
        x.data = x_data;
        x.channels = 1;
        x.rows = 1;
        x.cols = 1;
        x.plane_stride = 1;
        x.row_stride = 1;
        FeatureBatchView skip = x;
        skip.data = skip_data;
        DecInputWeight w;
        w.weight.bits = w_codes;
        w.weight.rows = 1;
        w.weight.cols = 1;
        w.gain = gain;
        w.gain_count = 1;
        float out[kStorage];
        float scratch[kStorage];
        fill_sentinel(out);
        fill_sentinel(scratch);
        CHECK_STATUS(dec_input_upsample(x, skip, w, out, 1, scratch, 1),
                     Status::kOk);
        const float want_out[1] = {64.0f};
        expect_values(out, want_out, 1, "K2 out", __LINE__);
        expect_sentinel(out + 1, kStorage - 1, __LINE__);
    }

    // ---- K3: zero gain drops the skip -------------------------------------------------------------
    // up = f16(2 * 4) = 8; gated = f16(f16(-7) * 0) = -0.0; sum = 8.
    {
        const float x_data[1] = {2.0f};
        const std::uint8_t w_codes[1] = {0x48};
        const float gain[1] = {0.0f};
        const float skip_data[1] = {-7.0f};
        FeatureBatchView x;
        x.data = x_data;
        x.channels = 1;
        x.rows = 1;
        x.cols = 1;
        x.plane_stride = 1;
        x.row_stride = 1;
        FeatureBatchView skip = x;
        skip.data = skip_data;
        DecInputWeight w;
        w.weight.bits = w_codes;
        w.weight.rows = 1;
        w.weight.cols = 1;
        w.gain = gain;
        w.gain_count = 1;
        float out[kStorage];
        float scratch[kStorage];
        fill_sentinel(out);
        fill_sentinel(scratch);
        CHECK_STATUS(dec_input_upsample(x, skip, w, out, 1, scratch, 1),
                     Status::kOk);
        const float want_out[1] = {8.0f};
        expect_values(out, want_out, 1, "K3 out", __LINE__);
    }

    // ---- K4: padded strides with poisoned padding -------------------------------------------------
    // The K1 inputs re-laid out with row/plane padding: x padding holds
    // 0.1 (off the e4m3 grid: a kernel over-read changes the sums and a
    // validation over-scan rejects spuriously), skip padding holds NaN (a
    // finite-scan over-reach rejects, an over-read yields NaN outputs). The
    // result must equal K1's.
    {
        float x_storage[30];
        float skip_storage[44];
        for (std::size_t i = 0; i < 30; ++i) {
            x_storage[i] = 0.1f;
        }
        for (std::size_t i = 0; i < 44; ++i) {
            skip_storage[i] = std::numeric_limits<float>::quiet_NaN();
        }
        // x: row_stride 5, plane_stride 15; logical cells at c*15 + r*5 + col.
        x_storage[0] = 1.0f;   x_storage[1] = 2.0f;
        x_storage[5] = 3.0f;   x_storage[6] = 4.0f;
        x_storage[15] = 4.0f;  x_storage[16] = 0.0f;
        x_storage[20] = -2.0f; x_storage[21] = 1.0f;
        // skip: row_stride 5, plane_stride 22; logical cells at c*22 + r*5 + col.
        skip_storage[0] = 1.0f;  skip_storage[1] = 2.0f;  skip_storage[2] = 3.0f;
        skip_storage[5] = 4.0f;  skip_storage[6] = 5.0f;  skip_storage[7] = 6.0f;
        skip_storage[10] = 7.0f; skip_storage[11] = 8.0f; skip_storage[12] = 9.0f;
        // Only the nine logical plane-1 cells are zero (the K1 values); the
        // rest of plane 1 (its row padding and the tail) stays NaN like the
        // rest of the padding, so a channel-1 over-read poisons the outputs
        // exactly as a channel-0 over-read does.
        skip_storage[22] = 0.0f; skip_storage[23] = 0.0f; skip_storage[24] = 0.0f;
        skip_storage[27] = 0.0f; skip_storage[28] = 0.0f; skip_storage[29] = 0.0f;
        skip_storage[32] = 0.0f; skip_storage[33] = 0.0f; skip_storage[34] = 0.0f;
        const std::uint8_t w_codes[4] = {0x38, 0x40, 0x30, 0xB8};
        const float gain[2] = {0.5f, 2.0f};
        FeatureBatchView x;
        x.data = x_storage;
        x.channels = 2;
        x.rows = 2;
        x.cols = 2;
        x.plane_stride = 15;
        x.row_stride = 5;
        FeatureBatchView skip;
        skip.data = skip_storage;
        skip.channels = 2;
        skip.rows = 3;
        skip.cols = 3;
        skip.plane_stride = 22;
        skip.row_stride = 5;
        DecInputWeight w;
        w.weight.bits = w_codes;
        w.weight.rows = 2;
        w.weight.cols = 2;
        w.gain = gain;
        w.gain_count = 2;
        float out[kStorage];
        float scratch[kStorage];
        fill_sentinel(out);
        fill_sentinel(scratch);
        CHECK_STATUS(dec_input_upsample(x, skip, w, out, 18, scratch, 8),
                     Status::kOk);
        const float want_out[18] = {10.0f, 10.0f, 3.5f,
                                    11.0f, 12.0f, 5.0f,
                                    2.5f, 3.0f, 10.0f,
                                    -3.5f, -3.5f, 1.0f,
                                    -3.5f, -3.5f, 1.0f,
                                    3.5f, 3.5f, 1.0f};
        expect_values(out, want_out, 18, "K4 out", __LINE__);
    }

    // ---- K5: crop mapping at odd extents and a crop below the map ---------------------------------
    // x [1, 1, 2] = {3, 5}, W = {1}; skip [1, 2, 3]; gain {0.5}.
    // proj = {3, 5}; out(y, col) = e4m3(proj[col>>1] + f16(skip*0.5)):
    //   y=0: 3.5, 4, 6.5   y=1: 5, 5.5, 8
    // (pins the y >> 1 / col >> 1 mapping at odd extents: the 2x4 replicated
    //  map is cropped to 2x3.)
    {
        const float x_data[2] = {3.0f, 5.0f};
        const std::uint8_t w_codes[1] = {0x38};
        const float gain[1] = {0.5f};
        const float skip_data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        FeatureBatchView x;
        x.data = x_data;
        x.channels = 1;
        x.rows = 1;
        x.cols = 2;
        x.plane_stride = 2;
        x.row_stride = 2;
        FeatureBatchView skip;
        skip.data = skip_data;
        skip.channels = 1;
        skip.rows = 2;
        skip.cols = 3;
        skip.plane_stride = 6;
        skip.row_stride = 3;
        DecInputWeight w;
        w.weight.bits = w_codes;
        w.weight.rows = 1;
        w.weight.cols = 1;
        w.gain = gain;
        w.gain_count = 1;
        float out[kStorage];
        float scratch[kStorage];
        fill_sentinel(out);
        fill_sentinel(scratch);
        CHECK_STATUS(dec_input_upsample(x, skip, w, out, 6, scratch, 2),
                     Status::kOk);
        const float want_out[6] = {3.5f, 4.0f, 6.5f, 5.0f, 5.5f, 8.0f};
        expect_values(out, want_out, 6, "K5 out", __LINE__);
        expect_sentinel(out + 6, kStorage - 6, __LINE__);
        // The scratch mirror holds the full projected map in row-major order
        // ({3, 5}); the 2x3 crop reads both cells (col 2 -> projected col 1),
        // and nothing beyond the two is written.
        const float want_scratch[2] = {3.0f, 5.0f};
        expect_values(scratch, want_scratch, 2, "K5 scratch", __LINE__);
        expect_sentinel(scratch + 2, kStorage - 2, __LINE__);
    }
    // Exactly-2x extents (no crop): x [1, 1, 1] = {7}, skip [1, 2, 2], gain
    // {0.5}. out = e4m3(7 + f16(skip * 0.5)) = 7.5, 8, 8 (8.5 is an e4m3 tie
    // to the even 8), 9.
    {
        const float x_data[1] = {7.0f};
        const std::uint8_t w_codes[1] = {0x38};
        const float gain[1] = {0.5f};
        const float skip_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        FeatureBatchView x;
        x.data = x_data;
        x.channels = 1;
        x.rows = 1;
        x.cols = 1;
        x.plane_stride = 1;
        x.row_stride = 1;
        FeatureBatchView skip;
        skip.data = skip_data;
        skip.channels = 1;
        skip.rows = 2;
        skip.cols = 2;
        skip.plane_stride = 4;
        skip.row_stride = 2;
        DecInputWeight w;
        w.weight.bits = w_codes;
        w.weight.rows = 1;
        w.weight.cols = 1;
        w.gain = gain;
        w.gain_count = 1;
        float out[kStorage];
        float scratch[kStorage];
        fill_sentinel(out);
        fill_sentinel(scratch);
        CHECK_STATUS(dec_input_upsample(x, skip, w, out, 4, scratch, 1),
                     Status::kOk);
        const float want_out[4] = {7.5f, 8.0f, 8.0f, 9.0f};
        expect_values(out, want_out, 4, "K5b out", __LINE__);
    }

    // ---- K6: overflow/NaN edges through the kernel ------------------------------------------------
    // (a) projection overflow: 448 * 448 = 200704 sums to f16 infinity; the
    //     store saturates exactly as q8's clamp -> 448.
    {
        const float x_data[1] = {448.0f};
        const std::uint8_t w_codes[1] = {0x7e};
        const float gain[1] = {1.0f};
        const float skip_data[1] = {1.0f};
        FeatureBatchView x;
        x.data = x_data;
        x.channels = 1;
        x.rows = 1;
        x.cols = 1;
        x.plane_stride = 1;
        x.row_stride = 1;
        FeatureBatchView skip = x;
        skip.data = skip_data;
        DecInputWeight w;
        w.weight.bits = w_codes;
        w.weight.rows = 1;
        w.weight.cols = 1;
        w.gain = gain;
        w.gain_count = 1;
        float out[kStorage];
        float scratch[kStorage];
        fill_sentinel(out);
        fill_sentinel(scratch);
        CHECK_STATUS(dec_input_upsample(x, skip, w, out, 1, scratch, 1),
                     Status::kOk);
        CHECK(std::isinf(scratch[0]) && scratch[0] > 0.0f);
        const float want_out[1] = {448.0f};
        expect_values(out, want_out, 1, "K6a out", __LINE__);
    }
    // (b) +inf projection with a -inf gate: inf + (-inf) is NaN, stored as
    //     the canonical quiet NaN (the reference's q8 NaN code 0x7f).
    {
        const float x_data[1] = {448.0f};
        const std::uint8_t w_codes[1] = {0x7e};
        const float gain[1] = {-65504.0f};
        const float skip_data[1] = {65504.0f};
        FeatureBatchView x;
        x.data = x_data;
        x.channels = 1;
        x.rows = 1;
        x.cols = 1;
        x.plane_stride = 1;
        x.row_stride = 1;
        FeatureBatchView skip = x;
        skip.data = skip_data;
        DecInputWeight w;
        w.weight.bits = w_codes;
        w.weight.rows = 1;
        w.weight.cols = 1;
        w.gain = gain;
        w.gain_count = 1;
        float out[kStorage];
        float scratch[kStorage];
        fill_sentinel(out);
        fill_sentinel(scratch);
        CHECK_STATUS(dec_input_upsample(x, skip, w, out, 1, scratch, 1),
                     Status::kOk);
        CHECK_BITS(out[0], 0x7fc00000u);
    }
    // (c) gate overflow to -inf added to a finite up: the store saturates
    //     to -448.
    {
        const float x_data[1] = {-1.0f};
        const std::uint8_t w_codes[1] = {0x38};
        const float gain[1] = {-65504.0f};
        const float skip_data[1] = {65504.0f};
        FeatureBatchView x;
        x.data = x_data;
        x.channels = 1;
        x.rows = 1;
        x.cols = 1;
        x.plane_stride = 1;
        x.row_stride = 1;
        FeatureBatchView skip = x;
        skip.data = skip_data;
        DecInputWeight w;
        w.weight.bits = w_codes;
        w.weight.rows = 1;
        w.weight.cols = 1;
        w.gain = gain;
        w.gain_count = 1;
        float out[kStorage];
        float scratch[kStorage];
        fill_sentinel(out);
        fill_sentinel(scratch);
        CHECK_STATUS(dec_input_upsample(x, skip, w, out, 1, scratch, 1),
                     Status::kOk);
        const float want_out[1] = {-448.0f};
        expect_values(out, want_out, 1, "K6c out", __LINE__);
    }
    // (d) gate NaN path (f16(skip) overflows to +inf, gain zero):
    //     inf * 0 = NaN -> stored canonical.
    {
        const float x_data[1] = {1.0f};
        const std::uint8_t w_codes[1] = {0x38};
        const float gain[1] = {0.0f};
        const float skip_data[1] = {70000.0f};
        FeatureBatchView x;
        x.data = x_data;
        x.channels = 1;
        x.rows = 1;
        x.cols = 1;
        x.plane_stride = 1;
        x.row_stride = 1;
        FeatureBatchView skip = x;
        skip.data = skip_data;
        DecInputWeight w;
        w.weight.bits = w_codes;
        w.weight.rows = 1;
        w.weight.cols = 1;
        w.gain = gain;
        w.gain_count = 1;
        float out[kStorage];
        float scratch[kStorage];
        fill_sentinel(out);
        fill_sentinel(scratch);
        CHECK_STATUS(dec_input_upsample(x, skip, w, out, 1, scratch, 1),
                     Status::kOk);
        CHECK_BITS(out[0], 0x7fc00000u);
    }

    // ---- K7: the projection's binary16 rounding (finite-range discriminator) ----------------------
    // Every other kernel answer's projection sum is exactly binary16, so the
    // rounding itself is unpinned there; these two sums need it.
    // (a) rounds DOWN: x = {9, 9}, W = {7.5, 15*2^-8} (codes 0x4f, 0x17):
    //     acc = 67.5 + 0.52734375 = 68.02734375, below the midpoint 68.03125
    //     of the [64, 128) binary16 grid (steps 2^-4) -> projected 68.0.
    //     With skip 2^-6, gain 1: out = e4m3(f16(68.0 + 2^-6)) = e4m3(68.0)
    //     = 64 (the e4m3 tie of 68.0 goes to the even 64). An unrounded
    //     projection gives f16(68.04296875) = 68.0625 -> out 72.
    {
        const float x_data[2] = {9.0f, 9.0f};
        const std::uint8_t w_codes[2] = {0x4f, 0x17};
        const float gain[1] = {1.0f};
        const float skip_data[1] = {0.015625f};
        FeatureBatchView x;
        x.data = x_data;
        x.channels = 2;
        x.rows = 1;
        x.cols = 1;
        x.plane_stride = 1;
        x.row_stride = 1;
        FeatureBatchView skip;
        skip.data = skip_data;
        skip.channels = 1;
        skip.rows = 1;
        skip.cols = 1;
        skip.plane_stride = 1;
        skip.row_stride = 1;
        DecInputWeight w;
        w.weight.bits = w_codes;
        w.weight.rows = 1;
        w.weight.cols = 2;
        w.gain = gain;
        w.gain_count = 1;
        float out[kStorage];
        float scratch[kStorage];
        fill_sentinel(out);
        fill_sentinel(scratch);
        CHECK_STATUS(dec_input_upsample(x, skip, w, out, 1, scratch, 1),
                     Status::kOk);
        const float want_scratch[1] = {68.0f};
        expect_values(scratch, want_scratch, 1, "K7a scratch", __LINE__);
        const float want_out[1] = {64.0f};
        expect_values(out, want_out, 1, "K7a out", __LINE__);
    }
    // (b) rounds UP (catches a truncating projection): x = {9, 11},
    //     W = {7.5, 13*2^-8} (codes 0x4f, 0x15): acc = 67.5 + 0.55859375 =
    //     68.05859375, above the 68.03125 midpoint -> projected 68.0625.
    //     out = e4m3(f16(68.0625 + 2^-6)) = e4m3(68.0625) = 72. Truncation
    //     gives projected 68.0 and out 64; the unrounded sum fails the
    //     scratch check (68.05859375).
    {
        const float x_data[2] = {9.0f, 11.0f};
        const std::uint8_t w_codes[2] = {0x4f, 0x15};
        const float gain[1] = {1.0f};
        const float skip_data[1] = {0.015625f};
        FeatureBatchView x;
        x.data = x_data;
        x.channels = 2;
        x.rows = 1;
        x.cols = 1;
        x.plane_stride = 1;
        x.row_stride = 1;
        FeatureBatchView skip;
        skip.data = skip_data;
        skip.channels = 1;
        skip.rows = 1;
        skip.cols = 1;
        skip.plane_stride = 1;
        skip.row_stride = 1;
        DecInputWeight w;
        w.weight.bits = w_codes;
        w.weight.rows = 1;
        w.weight.cols = 2;
        w.gain = gain;
        w.gain_count = 1;
        float out[kStorage];
        float scratch[kStorage];
        fill_sentinel(out);
        fill_sentinel(scratch);
        CHECK_STATUS(dec_input_upsample(x, skip, w, out, 1, scratch, 1),
                     Status::kOk);
        const float want_scratch[1] = {68.0625f};
        expect_values(scratch, want_scratch, 1, "K7b scratch", __LINE__);
        const float want_out[1] = {72.0f};
        expect_values(out, want_out, 1, "K7b out", __LINE__);
    }

    // ---- K8: three-term accumulation in key order -------------------------------------------------
    // (a) term completeness: x = {1, 1, 1}, W = {2, 3, 5} (codes 0x40, 0x44,
    //     0x4a): acc = 10 exactly; a dropped, duplicated or reordered k-loop
    //     term changes the answer. skip 0 -> out 10.
    {
        const float x_data[3] = {1.0f, 1.0f, 1.0f};
        const std::uint8_t w_codes[3] = {0x40, 0x44, 0x4a};
        const float gain[1] = {1.0f};
        const float skip_data[1] = {0.0f};
        FeatureBatchView x;
        x.data = x_data;
        x.channels = 3;
        x.rows = 1;
        x.cols = 1;
        x.plane_stride = 1;
        x.row_stride = 1;
        FeatureBatchView skip;
        skip.data = skip_data;
        skip.channels = 1;
        skip.rows = 1;
        skip.cols = 1;
        skip.plane_stride = 1;
        skip.row_stride = 1;
        DecInputWeight w;
        w.weight.bits = w_codes;
        w.weight.rows = 1;
        w.weight.cols = 3;
        w.gain = gain;
        w.gain_count = 1;
        float out[kStorage];
        float scratch[kStorage];
        fill_sentinel(out);
        fill_sentinel(scratch);
        CHECK_STATUS(dec_input_upsample(x, skip, w, out, 1, scratch, 1),
                     Status::kOk);
        const float want_scratch[1] = {10.0f};
        expect_values(scratch, want_scratch, 1, "K8a scratch", __LINE__);
        const float want_out[1] = {10.0f};
        expect_values(out, want_out, 1, "K8a out", __LINE__);
    }
    // (b) key-order accumulation (all three products are exact; only the
    //     summation rounds): x = {448, 2^-9, 448}, W = {448, 2^-9, -448}
    //     (codes 0x7e, 0x01, 0xfe) give the exact products 200704, 2^-18 and
    //     -200704. Key order absorbs the tiny middle term ((200704 + 2^-18)
    //     + -200704 = 0; 2^-18 is far below half an ulp of 200704), so the
    //     projected map is 0; a pairwise or reassociated sum ((200704 +
    //     -200704) + 2^-18) keeps 2^-18. The store cannot see the difference
    //     (the e4m3 subnormal grid rounds 2^-18 to 0), so the scratch pins it.
    {
        const float x_data[3] = {448.0f, 0.001953125f, 448.0f};
        const std::uint8_t w_codes[3] = {0x7e, 0x01, 0xfe};
        const float gain[1] = {1.0f};
        const float skip_data[1] = {0.0f};
        FeatureBatchView x;
        x.data = x_data;
        x.channels = 3;
        x.rows = 1;
        x.cols = 1;
        x.plane_stride = 1;
        x.row_stride = 1;
        FeatureBatchView skip;
        skip.data = skip_data;
        skip.channels = 1;
        skip.rows = 1;
        skip.cols = 1;
        skip.plane_stride = 1;
        skip.row_stride = 1;
        DecInputWeight w;
        w.weight.bits = w_codes;
        w.weight.rows = 1;
        w.weight.cols = 3;
        w.gain = gain;
        w.gain_count = 1;
        float out[kStorage];
        float scratch[kStorage];
        fill_sentinel(out);
        fill_sentinel(scratch);
        CHECK_STATUS(dec_input_upsample(x, skip, w, out, 1, scratch, 1),
                     Status::kOk);
        const float want_scratch[1] = {0.0f};
        expect_values(scratch, want_scratch, 1, "K8b scratch", __LINE__);
        const float want_out[1] = {0.0f};
        expect_values(out, want_out, 1, "K8b out", __LINE__);
    }
    // (c) the same three products with the tiny term FIRST (x = {2^-9, 448,
    //     448}, W = {2^-9, 448, -448}, codes 0x01, 0x7e, 0xfe): key order
    //     ((2^-18 + 200704) + -200704) still absorbs the tiny term -> 0, but
    //     a reversed k-loop ((-200704 + 200704) + 2^-18) keeps it -> 2^-18.
    //     (b) and (c) together pin the documented key order against both
    //     reassociation and reversal.
    {
        const float x_data[3] = {0.001953125f, 448.0f, 448.0f};
        const std::uint8_t w_codes[3] = {0x01, 0x7e, 0xfe};
        const float gain[1] = {1.0f};
        const float skip_data[1] = {0.0f};
        FeatureBatchView x;
        x.data = x_data;
        x.channels = 3;
        x.rows = 1;
        x.cols = 1;
        x.plane_stride = 1;
        x.row_stride = 1;
        FeatureBatchView skip;
        skip.data = skip_data;
        skip.channels = 1;
        skip.rows = 1;
        skip.cols = 1;
        skip.plane_stride = 1;
        skip.row_stride = 1;
        DecInputWeight w;
        w.weight.bits = w_codes;
        w.weight.rows = 1;
        w.weight.cols = 3;
        w.gain = gain;
        w.gain_count = 1;
        float out[kStorage];
        float scratch[kStorage];
        fill_sentinel(out);
        fill_sentinel(scratch);
        CHECK_STATUS(dec_input_upsample(x, skip, w, out, 1, scratch, 1),
                     Status::kOk);
        const float want_scratch[1] = {0.0f};
        expect_values(scratch, want_scratch, 1, "K8c scratch", __LINE__);
        const float want_out[1] = {0.0f};
        expect_values(out, want_out, 1, "K8c out", __LINE__);
    }

    // ---- rejection matrix: every class leaves out and scratch untouched ---------------------------
    // Baseline: c_in = 2, c_out = 2, x [2, 2, 2] tight, skip [2, 2, 2] with
    // row_stride 8 / plane_stride 128 (headroom so extent mutations keep the
    // strides legal and reach the crop check), K1 weights and gains.
    {
        float x_storage[8];
        float skip_storage[256];
        x_storage[0] = 1.0f;  x_storage[1] = 2.0f;
        x_storage[2] = 3.0f;  x_storage[3] = 4.0f;
        x_storage[4] = 4.0f;  x_storage[5] = 0.0f;
        x_storage[6] = -2.0f; x_storage[7] = 1.0f;
        for (std::size_t i = 0; i < 256; ++i) {
            skip_storage[i] = 0.0f;
        }
        skip_storage[0] = 1.0f; skip_storage[1] = 2.0f;
        skip_storage[8] = 3.0f; skip_storage[9] = 4.0f;
        skip_storage[128] = 5.0f; skip_storage[129] = 6.0f;
        skip_storage[136] = 7.0f; skip_storage[137] = 8.0f;
        std::uint8_t w_codes[4] = {0x38, 0x40, 0x30, 0xB8};
        const float gain[2] = {0.5f, 2.0f};
        FeatureBatchView x;
        x.data = x_storage;
        x.channels = 2;
        x.rows = 2;
        x.cols = 2;
        x.plane_stride = 4;
        x.row_stride = 2;
        FeatureBatchView skip;
        skip.data = skip_storage;
        skip.channels = 2;
        skip.rows = 2;
        skip.cols = 2;
        skip.plane_stride = 128;
        skip.row_stride = 8;
        DecInputWeight w;
        w.weight.bits = w_codes;
        w.weight.rows = 2;
        w.weight.cols = 2;
        w.gain = gain;
        w.gain_count = 2;
        float out[kStorage];
        float scratch[kStorage];

        // The baseline itself is valid (the mutations below start from it).
        fill_sentinel(out);
        fill_sentinel(scratch);
        CHECK_STATUS(dec_input_upsample(x, skip, w, out, 8, scratch, 8),
                     Status::kOk);

        // One helper: run the mutated call and require both buffers intact.
        const auto reject = [&](const FeatureBatchView& rx,
                                const FeatureBatchView& rskip,
                                const DecInputWeight& rw, float* rout,
                                std::size_t rout_count, float* rscratch,
                                std::size_t rscratch_capacity, Status want,
                                const char* what, int line) {
            ++g_checks;
            if (rout == nullptr || rscratch == nullptr) {
                // Null-buffer cases: sentinel-check whichever buffer survives.
            }
            if (rout != nullptr) {
                fill_sentinel(rout);
            }
            if (rscratch != nullptr) {
                fill_sentinel(rscratch);
            }
            const Status got = dec_input_upsample(rx, rskip, rw, rout, rout_count,
                                                  rscratch, rscratch_capacity);
            check_status(got, want, what, line);
            if (rout != nullptr) {
                expect_sentinel(rout, kStorage, line);
            }
            if (rscratch != nullptr) {
                expect_sentinel(rscratch, kStorage, line);
            }
        };

        // Null pointers.
        {
            FeatureBatchView rx = x;
            rx.data = nullptr;
            reject(rx, skip, w, out, 8, scratch, 8, Status::kNullPointer,
                   "null x", __LINE__);
        }
        {
            FeatureBatchView rskip = skip;
            rskip.data = nullptr;
            reject(x, rskip, w, out, 8, scratch, 8, Status::kNullPointer,
                   "null skip", __LINE__);
        }
        {
            DecInputWeight rw = w;
            rw.weight.bits = nullptr;
            reject(x, skip, rw, out, 8, scratch, 8, Status::kNullPointer,
                   "null weight", __LINE__);
        }
        {
            DecInputWeight rw = w;
            rw.gain = nullptr;
            reject(x, skip, rw, out, 8, scratch, 8, Status::kNullPointer,
                   "null gain", __LINE__);
        }
        reject(x, skip, w, nullptr, 8, scratch, 8, Status::kNullPointer,
               "null out", __LINE__);
        reject(x, skip, w, out, 8, nullptr, 8, Status::kNullPointer,
               "null scratch", __LINE__);

        // Map dimensions.
        {
            FeatureBatchView rx = x;
            rx.rows = 0;
            reject(rx, skip, w, out, 8, scratch, 8, Status::kInvalidDimensions,
                   "x rows 0", __LINE__);
        }
        {
            FeatureBatchView rx = x;
            rx.cols = 0;
            reject(rx, skip, w, out, 8, scratch, 8, Status::kInvalidDimensions,
                   "x cols 0", __LINE__);
        }
        {
            FeatureBatchView rx = x;
            rx.channels = 0;
            reject(rx, skip, w, out, 8, scratch, 8, Status::kInvalidDimensions,
                   "x channels 0", __LINE__);
        }
        {
            FeatureBatchView rx = x;
            rx.rows = kDecMaxExtent + 1;
            reject(rx, skip, w, out, 8, scratch, 8, Status::kInvalidDimensions,
                   "x rows over extent", __LINE__);
        }
        {
            FeatureBatchView rx = x;
            rx.cols = kDecMaxExtent + 1;
            reject(rx, skip, w, out, 8, scratch, 8, Status::kInvalidDimensions,
                   "x cols over extent", __LINE__);
        }
        {
            FeatureBatchView rskip = skip;
            rskip.rows = 0;
            reject(x, rskip, w, out, 8, scratch, 8, Status::kInvalidDimensions,
                   "skip rows 0", __LINE__);
        }
        {
            FeatureBatchView rskip = skip;
            rskip.channels = 0;
            reject(x, rskip, w, out, 8, scratch, 8, Status::kInvalidDimensions,
                   "skip channels 0", __LINE__);
        }
        {
            FeatureBatchView rskip = skip;
            rskip.rows = kDecMaxExtent + 1;
            reject(x, rskip, w, out, 8, scratch, 8, Status::kInvalidDimensions,
                   "skip rows over extent", __LINE__);
        }

        // Strides.
        {
            FeatureBatchView rx = x;
            rx.row_stride = 1;
            reject(rx, skip, w, out, 8, scratch, 8, Status::kInvalidStride,
                   "x row_stride < cols", __LINE__);
        }
        {
            FeatureBatchView rx = x;
            rx.row_stride = kDecMaxExtent + 1;
            reject(rx, skip, w, out, 8, scratch, 8, Status::kInvalidStride,
                   "x row_stride over bound", __LINE__);
        }
        {
            FeatureBatchView rx = x;
            rx.plane_stride = 3;
            reject(rx, skip, w, out, 8, scratch, 8, Status::kInvalidStride,
                   "x plane_stride < rows*row_stride", __LINE__);
        }
        {
            FeatureBatchView rx = x;
            rx.plane_stride =
                static_cast<std::ptrdiff_t>(kDecMaxExtent) * kDecMaxExtent + 1;
            reject(rx, skip, w, out, 8, scratch, 8, Status::kInvalidStride,
                   "x plane_stride over bound", __LINE__);
        }
        {
            FeatureBatchView rskip = skip;
            rskip.row_stride = 1;
            reject(x, rskip, w, out, 8, scratch, 8, Status::kInvalidStride,
                   "skip row_stride < cols", __LINE__);
        }
        {
            FeatureBatchView rskip = skip;
            rskip.row_stride = kDecMaxExtent + 1;
            reject(x, rskip, w, out, 8, scratch, 8, Status::kInvalidStride,
                   "skip row_stride over bound", __LINE__);
        }
        {
            FeatureBatchView rskip = skip;
            rskip.plane_stride = 15;
            reject(x, rskip, w, out, 8, scratch, 8, Status::kInvalidStride,
                   "skip plane_stride < rows*row_stride", __LINE__);
        }
        {
            FeatureBatchView rskip = skip;
            rskip.plane_stride =
                static_cast<std::ptrdiff_t>(kDecMaxExtent) * kDecMaxExtent + 1;
            reject(x, rskip, w, out, 8, scratch, 8, Status::kInvalidStride,
                   "skip plane_stride over bound", __LINE__);
        }

        // Crop/replicate-extent discriminator: a skip beyond the 2x
        // replicated map is the reference's shape error (the resample
        // family's clamp-replicate would accept it), rejected untouched.
        {
            FeatureBatchView rskip = skip;
            rskip.rows = 5;
            reject(x, rskip, w, out, 8, scratch, 8, Status::kInvalidDimensions,
                   "skip rows beyond 2x", __LINE__);
        }
        {
            FeatureBatchView rskip = skip;
            rskip.cols = 5;
            reject(x, rskip, w, out, 8, scratch, 8, Status::kInvalidDimensions,
                   "skip cols beyond 2x", __LINE__);
        }

        // Weight and gain shapes.
        {
            DecInputWeight rw = w;
            rw.weight.rows = 1;
            reject(x, skip, rw, out, 8, scratch, 8, Status::kInvalidDimensions,
                   "weight rows mismatch", __LINE__);
        }
        {
            DecInputWeight rw = w;
            rw.weight.cols = 1;
            reject(x, skip, rw, out, 8, scratch, 8, Status::kInvalidDimensions,
                   "weight cols mismatch", __LINE__);
        }
        {
            DecInputWeight rw = w;
            rw.gain_count = 1;
            reject(x, skip, rw, out, 8, scratch, 8, Status::kInvalidBuffer,
                   "gain_count too small", __LINE__);
        }
        {
            DecInputWeight rw = w;
            rw.gain_count = 3;
            reject(x, skip, rw, out, 8, scratch, 8, Status::kInvalidBuffer,
                   "gain_count too large", __LINE__);
        }

        // Value classes.
        {
            x_storage[3] = 0.1f;
            reject(x, skip, w, out, 8, scratch, 8, Status::kInvalidValues,
                   "x off e4m3 grid", __LINE__);
            x_storage[3] = 4.0f;
        }
        {
            x_storage[3] = std::numeric_limits<float>::quiet_NaN();
            reject(x, skip, w, out, 8, scratch, 8, Status::kInvalidValues,
                   "x NaN", __LINE__);
            x_storage[3] = 4.0f;
        }
        {
            x_storage[3] = std::numeric_limits<float>::infinity();
            reject(x, skip, w, out, 8, scratch, 8, Status::kInvalidValues,
                   "x inf", __LINE__);
            x_storage[3] = 4.0f;
        }
        {
            skip_storage[0] = std::numeric_limits<float>::quiet_NaN();
            reject(x, skip, w, out, 8, scratch, 8, Status::kInvalidValues,
                   "skip NaN", __LINE__);
            skip_storage[0] = 1.0f;
        }
        {
            skip_storage[0] = std::numeric_limits<float>::infinity();
            reject(x, skip, w, out, 8, scratch, 8, Status::kInvalidValues,
                   "skip inf", __LINE__);
            skip_storage[0] = 1.0f;
        }
        {
            float bad_gain[2] = {0.5f, 0.1f};
            DecInputWeight rw = w;
            rw.gain = bad_gain;
            reject(x, skip, rw, out, 8, scratch, 8, Status::kInvalidValues,
                   "gain off f16 grid", __LINE__);
        }
        {
            float bad_gain[2] = {0.5f, 2.0f};
            bad_gain[0] = std::numeric_limits<float>::quiet_NaN();
            DecInputWeight rw = w;
            rw.gain = bad_gain;
            reject(x, skip, rw, out, 8, scratch, 8, Status::kInvalidValues,
                   "gain NaN", __LINE__);
        }
        {
            float bad_gain[2] = {0.5f, 2.0f};
            bad_gain[1] = std::numeric_limits<float>::infinity();
            DecInputWeight rw = w;
            rw.gain = bad_gain;
            reject(x, skip, rw, out, 8, scratch, 8, Status::kInvalidValues,
                   "gain inf", __LINE__);
        }
        {
            w_codes[0] = 0x7f;
            reject(x, skip, w, out, 8, scratch, 8, Status::kInvalidValues,
                   "weight NaN code 0x7f", __LINE__);
            w_codes[0] = 0x38;
        }
        {
            w_codes[3] = 0xff;
            reject(x, skip, w, out, 8, scratch, 8, Status::kInvalidValues,
                   "weight NaN code 0xff", __LINE__);
            w_codes[3] = 0xB8;
        }

        // Output and scratch capacities.
        reject(x, skip, w, out, 7, scratch, 8, Status::kInvalidBuffer,
               "out_count too small", __LINE__);
        reject(x, skip, w, out, 8, scratch, 7, Status::kInvalidBuffer,
               "scratch_capacity too small", __LINE__);
    }

    // ---- determinism and zero-allocation accounting ----------------------------------------------
    {
        const float x_data[8] = {1.0f, 2.0f, 3.0f, 4.0f,
                                 4.0f, 0.0f, -2.0f, 1.0f};
        const std::uint8_t w_codes[4] = {0x38, 0x40, 0x30, 0xB8};
        const float gain[2] = {0.5f, 2.0f};
        const float skip_data[18] = {1.0f, 2.0f, 3.0f,
                                     4.0f, 5.0f, 6.0f,
                                     7.0f, 8.0f, 9.0f,
                                     -1.0f, -2.0f, -3.0f,
                                     -4.0f, -5.0f, -6.0f,
                                     -7.0f, -8.0f, -9.0f};
        FeatureBatchView x;
        x.data = x_data;
        x.channels = 2;
        x.rows = 2;
        x.cols = 2;
        x.plane_stride = 4;
        x.row_stride = 2;
        FeatureBatchView skip;
        skip.data = skip_data;
        skip.channels = 2;
        skip.rows = 3;
        skip.cols = 3;
        skip.plane_stride = 9;
        skip.row_stride = 3;
        DecInputWeight w;
        w.weight.bits = w_codes;
        w.weight.rows = 2;
        w.weight.cols = 2;
        w.gain = gain;
        w.gain_count = 2;
        float out_a[kStorage];
        float out_b[kStorage];
        float scratch_a[kStorage];
        float scratch_b[kStorage];
        fill_sentinel(out_a);
        fill_sentinel(out_b);
        fill_sentinel(scratch_a);
        fill_sentinel(scratch_b);
        const unsigned long long before = g_allocations.load();
        CHECK_STATUS(dec_input_upsample(x, skip, w, out_a, 18, scratch_a, 8),
                     Status::kOk);
        CHECK_STATUS(dec_input_upsample(x, skip, w, out_b, 18, scratch_b, 8),
                     Status::kOk);
        // The element forms are pure as well.
        volatile float sink = dec_gated_element(1.00048732757568359375f, 1.5f);
        sink = sink + dec_store(68.00390625f);
        (void)sink;
        CHECK(g_allocations.load() == before);
        CHECK(std::memcmp(out_a, out_b, 18 * sizeof(float)) == 0);
        CHECK(std::memcmp(scratch_a, scratch_b, 8 * sizeof(float)) == 0);
    }

    std::printf("native-decoder: %lld checks passed\n", g_checks);
    return 0;
}
