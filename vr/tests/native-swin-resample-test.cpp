// Independent tests for the native Swin resampling slice
// (vr/native-swin-resample.hpp): the pooled learned C -> 2C projection
// (`downsample`), the learned 2C -> C projected nearest-2x gated blend
// (`upsample_blend`) and the thin SwinLayer compositions (`swin_layer_plain`,
// `swin_layer_upsample`). CPU-only, no GPU, headset, proprietary payload,
// Python or PyTorch runtime required.
//
// Independence: every known answer is a hand-derived dyadic rational (the exact
// decompositions are written next to each case). Controls cover: the gated
// sum's single float64-rounding discriminator (1037, not 1038), its overflow
// band (65504/65520) and subnormal ties-to-even; the projection's k-order
// accumulator discriminator (200704, not 200704.015625) and the stored-result
// double rounding (367.875 -> 384 where a direct e4m3 rounding gives 352);
// the replication index arithmetic with its clamp edge; the resize semantics
// of the downsample output (top-left kept, zero padded, cropped); a bit-exact
// equivalence control of every composed entry against the staged chain over
// deterministic corpora (zero-weight and non-trivial weight blocks, all shift
// pairs, pad and crop output extents, clamped replication extents, and
// poisoned padded input strides on the entries' own stride reads — the
// block's x strides are pinned in the block suite, not re-shared here); the
// `C == 32` f16-residual upsample path discriminator
// (384 versus 352); sentinel-checked untouched outputs and scratch on every
// validation rejection class (with the documented blend-overflow scratch
// exception); determinism and zero-allocation accounting.
// See vr/NATIVE-SWIN-RESAMPLE.md.

#include "native-swin-resample.hpp"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>
#include <new>

static std::atomic<unsigned long long> g_allocations{0};
void* operator new(size_t n) {
    g_allocations.fetch_add(1);
    void* p = malloc(n);
    if (p == nullptr) {
        abort();
    }
    return p;
}
void* operator new[](size_t n) { return ::operator new(n); }
void* operator new(size_t n, std::align_val_t) {
    g_allocations.fetch_add(1);
    void* p = malloc(n);
    if (p == nullptr) {
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

static long long g_checks = 0;
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

using dlss5_nr::FeatureBatchView;
using dlss5_nr::MatrixView;
using dlss5_nr::MutableMatrixView;
using dlss5_nr::SwinBlockParams;
using dlss5_nr::Status;

static MatrixView make_view(const float* data, int rows, int cols,
                            std::ptrdiff_t row_stride) {
    MatrixView v;
    v.data = data;
    v.rows = rows;
    v.cols = cols;
    v.row_stride = row_stride;
    return v;
}

static FeatureBatchView make_batch(const float* data, int channels, int rows,
                                   int cols, std::ptrdiff_t plane_stride,
                                   std::ptrdiff_t row_stride) {
    FeatureBatchView v;
    v.data = data;
    v.channels = channels;
    v.rows = rows;
    v.cols = cols;
    v.plane_stride = plane_stride;
    v.row_stride = row_stride;
    return v;
}

// ---- deterministic corpus helpers ----------------------------------------------------------------
// xorshift32; the f16 corpus draws arbitrary sign/exponent/mantissa patterns
// (the full 16-bit pattern, sign bit included) with the exponent field in
// 0..30 (finite values only, NaN/Inf excluded).
static uint32_t rng_next(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static float rnd_f16(uint32_t* state) {
    for (;;) {
        const uint16_t bits = static_cast<uint16_t>(rng_next(state));
        if ((bits & 0x7C00u) != 0x7C00u) {
            return dlss5_nr::f16_from_bits(bits);
        }
    }
}

static float rnd_e4m3(uint32_t* state) {
    // On the e4m3 grid: e4m3 of a modest finite draw (the grid projection is
    // the tested e4m3 primitive; the corpus needs grid membership, not
    // uniformity).
    const int32_t raw = static_cast<int32_t>(rng_next(state) % 2001u) - 1000;
    return dlss5_nr::e4m3(static_cast<float>(raw) * 0.03125f);
}

// Modest exactly-binary16 draws for the blend corpora (the layer entries
// reject blends rounded outside the finite binary16 range, so the corpora
// stay clear of that disclosed class): multiples of 2^-5 in [-4, 4] for
// gains and of 2^-1 in [-500, 500] for skips.
static float rnd_gain(uint32_t* state) {
    const int32_t raw = static_cast<int32_t>(rng_next(state) % 257u) - 128;
    return static_cast<float>(raw) * 0.03125f;
}

static float rnd_skip(uint32_t* state) {
    const int32_t raw = static_cast<int32_t>(rng_next(state) % 2001u) - 1000;
    return static_cast<float>(raw) * 0.5f;
}

// Gap fill for padded input layouts: exactly binary16 and on the e4m3 grid,
// so it satisfies both input-map classes if a wrong tight-pack read ever
// touches it, and distinct enough from the corpora to move any such result.
static constexpr float kPoison = -224.0f;

// Scatters a tight [channels, rows, cols] corpus into a padded plane/row
// layout, leaving every gap at kPoison so a tight-pack read changes results.
static void scatter_padded(float* out, const float* tight, int channels,
                           int rows, int cols, std::ptrdiff_t plane_stride,
                           std::ptrdiff_t row_stride) {
    const std::size_t total = static_cast<std::size_t>(channels)
        * static_cast<std::size_t>(plane_stride);
    for (std::size_t p = 0; p < total; ++p) {
        out[p] = kPoison;
    }
    for (int c = 0; c < channels; ++c) {
        for (int r = 0; r < rows; ++r) {
            for (int j = 0; j < cols; ++j) {
                out[static_cast<std::size_t>(c)
                        * static_cast<std::size_t>(plane_stride)
                    + static_cast<std::size_t>(r)
                        * static_cast<std::size_t>(row_stride)
                    + static_cast<std::size_t>(j)] =
                    tight[(static_cast<std::size_t>(c) * rows + r) * cols + j];
            }
        }
    }
}

// ---- fixture --------------------------------------------------------------------------------------
// One zero-weight block: with every weight zero and unit scales the block's
// output is its residual (the block slice's pinned closed form).

namespace fixture {

constexpr int kChannels = 32;   // heads = 1
constexpr int kWideChannels = 64;  // heads = 2
constexpr int kHidden = 32;

struct BlockFixture {
    float expand[kHidden * kChannels];
    float contract[kChannels * kHidden];
    float mlp_scale[kChannels];
    float qkv[3 * kChannels * kChannels];
    float head_scale[1];
    float exp_bias[dlss5_nr::kBiasTable];
    float attn_scale[kChannels];
    float out_proj[kChannels * kChannels];
    SwinBlockParams params;

    void zero_init() {
        memset(expand, 0, sizeof expand);
        memset(contract, 0, sizeof contract);
        for (int c = 0; c < kChannels; ++c) {
            mlp_scale[c] = 1.0f;
            attn_scale[c] = 1.0f;
        }
        memset(qkv, 0, sizeof qkv);
        head_scale[0] = 1.0f;
        memset(exp_bias, 0, sizeof exp_bias);
        memset(out_proj, 0, sizeof out_proj);
        params = SwinBlockParams{};
        params.expand = make_view(expand, kHidden, kChannels, kChannels);
        params.mid = nullptr;
        params.heads = 1;
        params.contract = make_view(contract, kChannels, kHidden, kHidden);
        params.mlp_scale = make_view(mlp_scale, 1, kChannels, kChannels);
        params.qkv = make_view(qkv, 3 * kChannels, kChannels, kChannels);
        params.head_scale = make_view(head_scale, 1, 1, 1);
        params.exp_bias = make_view(exp_bias, 1, dlss5_nr::kBiasTable,
            dlss5_nr::kBiasTable);
        params.attn_scale = make_view(attn_scale, 1, kChannels, kChannels);
        params.out_proj = make_view(out_proj, kChannels, kChannels, kChannels);
    }

    // Non-trivial weights on the documented grids (deterministic).
    void randomise(uint32_t* state) {
        for (std::size_t p = 0; p < sizeof expand / sizeof expand[0]; ++p) {
            expand[p] = rnd_e4m3(state);
        }
        for (std::size_t p = 0; p < sizeof contract / sizeof contract[0]; ++p) {
            contract[p] = rnd_e4m3(state);
        }
        for (int c = 0; c < kChannels; ++c) {
            mlp_scale[c] = dlss5_nr::f16_from_bits(
                static_cast<uint16_t>(rng_next(state) & 0x7BFFu));
            attn_scale[c] = dlss5_nr::f16_from_bits(
                static_cast<uint16_t>(rng_next(state) & 0x7BFFu));
            if (mlp_scale[c] == 0.0f) {
                mlp_scale[c] = 1.0f;
            }
            if (attn_scale[c] == 0.0f) {
                attn_scale[c] = 1.0f;
            }
        }
        for (std::size_t p = 0; p < sizeof qkv / sizeof qkv[0]; ++p) {
            qkv[p] = rnd_e4m3(state);
        }
        head_scale[0] = 8.0f;
        for (std::size_t p = 0; p < sizeof exp_bias / sizeof exp_bias[0]; ++p) {
            exp_bias[p] = static_cast<float>(
                static_cast<int32_t>(rng_next(state) % 65u) - 32) * 0.125f;
        }
        for (std::size_t p = 0; p < sizeof out_proj / sizeof out_proj[0]; ++p) {
            out_proj[p] = rnd_e4m3(state);
        }
    }
};

// The heads = 2 fixture on C = 64 (the upsample C != 32 path).
struct WideFixture {
    float expand[kHidden * kWideChannels];
    float contract[kWideChannels * kHidden];
    float mlp_scale[kWideChannels];
    float qkv[3 * kWideChannels * kWideChannels];
    float head_scale[2];
    float exp_bias[2 * dlss5_nr::kBiasTable];
    float attn_scale[kWideChannels];
    float out_proj[kWideChannels * kWideChannels];
    SwinBlockParams params;

    void zero_init() {
        memset(expand, 0, sizeof expand);
        memset(contract, 0, sizeof contract);
        for (int c = 0; c < kWideChannels; ++c) {
            mlp_scale[c] = 1.0f;
            attn_scale[c] = 1.0f;
        }
        memset(qkv, 0, sizeof qkv);
        head_scale[0] = 1.0f;
        head_scale[1] = 1.0f;
        memset(exp_bias, 0, sizeof exp_bias);
        memset(out_proj, 0, sizeof out_proj);
        params = SwinBlockParams{};
        params.expand = make_view(expand, kHidden, kWideChannels, kWideChannels);
        params.mid = nullptr;
        params.heads = 2;
        params.contract = make_view(contract, kWideChannels, kHidden, kHidden);
        params.mlp_scale = make_view(mlp_scale, 1, kWideChannels, kWideChannels);
        params.qkv = make_view(qkv, 3 * kWideChannels, kWideChannels,
            kWideChannels);
        params.head_scale = make_view(head_scale, 1, 2, 2);
        params.exp_bias = make_view(exp_bias, 2, dlss5_nr::kBiasTable,
            dlss5_nr::kBiasTable);
        params.attn_scale = make_view(attn_scale, 1, kWideChannels,
            kWideChannels);
        params.out_proj = make_view(out_proj, kWideChannels, kWideChannels,
            kWideChannels);
    }

    // Non-trivial weights on the documented grids (deterministic).
    void randomise(uint32_t* state) {
        for (std::size_t p = 0; p < sizeof expand / sizeof expand[0]; ++p) {
            expand[p] = rnd_e4m3(state);
        }
        for (std::size_t p = 0; p < sizeof contract / sizeof contract[0]; ++p) {
            contract[p] = rnd_e4m3(state);
        }
        for (int c = 0; c < kWideChannels; ++c) {
            mlp_scale[c] = dlss5_nr::f16_from_bits(
                static_cast<uint16_t>(rng_next(state) & 0x7BFFu));
            attn_scale[c] = dlss5_nr::f16_from_bits(
                static_cast<uint16_t>(rng_next(state) & 0x7BFFu));
            if (mlp_scale[c] == 0.0f) {
                mlp_scale[c] = 1.0f;
            }
            if (attn_scale[c] == 0.0f) {
                attn_scale[c] = 1.0f;
            }
        }
        for (std::size_t p = 0; p < sizeof qkv / sizeof qkv[0]; ++p) {
            qkv[p] = rnd_e4m3(state);
        }
        head_scale[0] = 8.0f;
        head_scale[1] = 8.0f;
        for (std::size_t p = 0; p < sizeof exp_bias / sizeof exp_bias[0]; ++p) {
            exp_bias[p] = static_cast<float>(
                static_cast<int32_t>(rng_next(state) % 65u) - 32) * 0.125f;
        }
        for (std::size_t p = 0; p < sizeof out_proj / sizeof out_proj[0]; ++p) {
            out_proj[p] = rnd_e4m3(state);
        }
    }
};

}  // namespace fixture

// ---- gated sum known answers ---------------------------------------------------------------------

static void check_gated_sum(void) {
    using dlss5_nr::gated_sum_element;
    // The single-rounding discriminator: 25 * 41.5 = 1037.5 exactly (25 =
    // 1.5625 * 2^4, 41.5 = 1.296875 * 2^5, both binary16-exact) and
    // other = -3 * 2^-24 is a binary16 subnormal (3 units of 2^-24), so the
    // exact sum 1037.5 - 1.5 * 2^-23 is closer to 1037 than 1038 and rounds
    // to 1037. A float32 form of the sum rounds onto the 1037.5 midpoint
    // first and then breaks that tie to even, giving 1038.
    CHECK(gated_sum_element(25.0f, 41.5f, -3.0f * 0x1p-24f) == 1037.0f);
    CHECK(gated_sum_element(25.0f, 41.5f, -3.0f * 0x1p-24f) != 1038.0f);
    CHECK(dlss5_nr::f16(25.0f * 41.5f + -3.0f * 0x1p-24f) == 1038.0f);
    // The overflow band: 65504 + 15.5 = 65519.5 lands in (65504, 65520) and
    // rounds to 65504; 65504 + 16 = 65520 rounds to infinity (the reference's
    // float16 conversion), with the mirrored negative cases.
    CHECK(gated_sum_element(65504.0f, 1.0f, 15.5f) == 65504.0f);
    CHECK(gated_sum_element(65504.0f, 1.0f, 16.0f) == (float)INFINITY);
    CHECK(gated_sum_element(-65504.0f, 1.0f, -15.5f) == -65504.0f);
    CHECK(gated_sum_element(-65504.0f, 1.0f, -16.0f) == -(float)INFINITY);
    // Subnormal ties to even: 1.5 units of 2^-24 tie between 1 and 2 units and
    // round to 2; 2.5 units tie between 2 and 3 and round to 2; 3.5 -> 4.
    CHECK(gated_sum_element(0x1p-24f, 1.5f, 0.0f) == 0x1p-23f);
    CHECK(gated_sum_element(0x1p-24f, 2.5f, 0.0f) == 0x1p-23f);
    CHECK(gated_sum_element(0x1p-24f, 3.5f, 0.0f) == 0x1p-22f);
    // Plain exact cases and the sign of zero.
    CHECK(gated_sum_element(1.0f, 2.0f, 0.5f) == 2.5f);
    CHECK(gated_sum_element(0.0f, 5.0f, 7.0f) == 7.0f);
    CHECK(gated_sum_element(-2.0f, 3.0f, 1.0f) == -5.0f);
    CHECK(gated_sum_element(-0.0f, 5.0f, 0.0f) == 0.0f);
}

// ---- projection known answers --------------------------------------------------------------------

static void check_projections(void) {
    // The stored-result double rounding: products 1.875 * 192 = 360 and
    // 1.125 * 7 = 7.875 (every factor binary16-exact and on the e4m3 grid)
    // sum to 367.875 exactly in float32. 367.875 is the binary16 midpoint
    // between 367.75 (mantissa 447, odd) and 368 (mantissa 448, even), so
    // store's first rounding breaks that tie to even and gives 368; 368 is
    // then the 352/384 e4m3 midpoint (352's mantissa 3 is odd, 384's
    // mantissa 4 is even), giving 384, where a direct e4m3 rounding of
    // 367.875 (closer to 352 than to 384) gives 352.
    float weight[2] = {1.875f, 1.125f};
    float planes[2] = {192.0f, 7.0f};
    CHECK(dlss5_nr::swin_project_dot(weight, 2, planes, 1, 0) == 367.875f);
    CHECK(dlss5_nr::swin_project_store(weight, 2, planes, 1, 0) == 384.0f);
    CHECK(dlss5_nr::swin_project_narrow(weight, 2, planes, 1, 0) == 368.0f);
    CHECK(dlss5_nr::e4m3(367.875f) == 352.0f);
    // The k-order float32 accumulator discriminator: products 448 * 448 =
    // 200704 and two of 2^-7 (2^-7 * 1) tie against the 2^-6 grid step of the
    // [2^17, 2^18) binade; sequential k-order accumulation rounds both ties
    // to even and stays at 200704, where summing the two tail products first
    // (or exact summation) gives 200704.015625.
    float w3[3] = {448.0f, 0x1p-7f, 0x1p-7f};
    float p3[3] = {448.0f, 1.0f, 1.0f};
    CHECK(dlss5_nr::swin_project_dot(w3, 3, p3, 1, 0) == 200704.0f);
    CHECK(dlss5_nr::swin_project_dot(w3, 3, p3, 1, 0) != 200704.015625f);
    // Both alternatives pinned in float32 itself: sequential left-to-right
    // addition (the accumulator's order) repeats the same tie-to-even steps
    // and stays at 200704, while summing the two tail products first gives
    // the exact sum 200704.015625 (the discriminated alternative).
    CHECK(200704.0f + 0x1p-7f + 0x1p-7f == 200704.0f);
    CHECK(200704.0f + (0x1p-7f + 0x1p-7f) == 200704.015625f);
    CHECK(0x1p-7f + 0x1p-7f == 0x1p-6f);
    // The k-ascending plane walk: plane stride and offset are honoured.
    // Same planes as above (192 and 7) held at k * 2 + 1, with a
    // discriminator (-1) in every other slot; offset 1 selects them.
    float strided[4] = {-1.0f, 192.0f, -1.0f, 7.0f};
    CHECK(dlss5_nr::swin_project_dot(weight, 2, strided, 2, 1) == 367.875f);
    CHECK(dlss5_nr::swin_project_dot(weight, 2, strided, 2, 0) != 367.875f);
}

// ---- downsample known answers --------------------------------------------------------------------

static void check_downsample_closed_forms(void) {
    using dlss5_nr::swin_downsample;
    fixture::BlockFixture f;
    f.zero_init();
    static float x[fixture::kChannels * 4 * 4];
    static float next_dst[2 * fixture::kChannels * 4 * 4];
    static float body_dst[fixture::kChannels * 4 * 4];
    static float scratch[1 << 20];
    static float resample[2 * fixture::kChannels * fixture::kChannels];
    memset(resample, 0, sizeof resample);
    // resample[n, n % 32] = 1: each output plane copies pooled plane n % 32.
    for (int n = 0; n < 2 * fixture::kChannels; ++n) {
        resample[n * fixture::kChannels + (n % fixture::kChannels)] = 1.0f;
    }
    MatrixView resample_v = make_view(resample, 2 * fixture::kChannels,
        fixture::kChannels, fixture::kChannels);
    // The map: every 2x2 pool group holds [[1, 2], [3, 4]] on the e4m3 grid.
    // pool2x2: (1 + 2) narrows to f16 3, (3 + 4) to f16 7, 3 + 7 = 10 in f16,
    // the quarter is exact, e4m3(2.5) = 2.5. The block with zero weights
    // returns its residual round8(x) = x (every value already e4m3), so the
    // pooled map is 2.5 at every position.
    for (int c = 0; c < fixture::kChannels; ++c) {
        for (int r = 0; r < 4; ++r) {
            for (int col = 0; col < 4; ++col) {
                x[(c * 4 + r) * 4 + col] =
                    (r % 2 == 0)
                        ? (col % 2 == 0 ? 1.0f : 2.0f)
                        : (col % 2 == 0 ? 3.0f : 4.0f);
            }
        }
    }
    const FeatureBatchView x_v = make_batch(x, fixture::kChannels, 4, 4,
        16, 4);
    // Output extent 4x4: the pooled 2x2 map is zero padded bottom/right.
    CHECK(swin_downsample(x_v, 0, 0, f.params, resample_v, 4, 4,
        next_dst, 2 * fixture::kChannels * 16,
        body_dst, fixture::kChannels * 16,
        scratch, sizeof scratch / sizeof scratch[0]) == Status::kOk);
    for (int n = 0; n < 2 * fixture::kChannels; ++n) {
        for (int r = 0; r < 4; ++r) {
            for (int col = 0; col < 4; ++col) {
                const float expected = (r < 2 && col < 2) ? 2.5f : 0.0f;
                CHECK(next_dst[(n * 4 + r) * 4 + col] == expected);
            }
        }
    }
    // body = round8(wide) = e4m3(x) = x here.
    for (std::size_t p = 0; p < fixture::kChannels * 16u; ++p) {
        CHECK(body_dst[p] == x[p]);
    }
    // The projection sum: resample row n with two units sums two pooled
    // planes (2.5 + 2.5 = 5, store keeps 5 on the e4m3 grid).
    for (int n = 0; n < 2 * fixture::kChannels; ++n) {
        resample[n * fixture::kChannels + (n % fixture::kChannels)] = 0.0f;
        resample[n * fixture::kChannels] = 1.0f;
        resample[n * fixture::kChannels + 1] = 1.0f;
    }
    CHECK(swin_downsample(x_v, 0, 0, f.params, resample_v, 4, 4,
        next_dst, 2 * fixture::kChannels * 16,
        body_dst, fixture::kChannels * 16,
        scratch, sizeof scratch / sizeof scratch[0]) == Status::kOk);
    for (int n = 0; n < 2 * fixture::kChannels; ++n) {
        CHECK(next_dst[n * 16] == 5.0f);
    }
    // The crop path: a 1x1 output keeps only the top-left pooled value.
    CHECK(swin_downsample(x_v, 0, 0, f.params, resample_v, 1, 1,
        next_dst, 2 * fixture::kChannels,
        body_dst, fixture::kChannels * 16,
        scratch, sizeof scratch / sizeof scratch[0]) == Status::kOk);
    for (int n = 0; n < 2 * fixture::kChannels; ++n) {
        CHECK(next_dst[n] == 5.0f);
    }
}

// ---- upsample_blend known answers ----------------------------------------------------------------

static void check_upsample_blend_closed_forms(void) {
    using dlss5_nr::swin_upsample_blend;
    static float project[fixture::kChannels * 2 * fixture::kChannels];
    static float gain[fixture::kChannels];
    static float x[2 * fixture::kChannels * 2 * 2];
    static float skip[fixture::kChannels * 3 * 5];
    static float blend_dst[fixture::kChannels * 3 * 5];
    static float scratch[1 << 16];
    memset(project, 0, sizeof project);
    // project[c, c] = 1: the projected map is x plane c.
    for (int c = 0; c < fixture::kChannels; ++c) {
        project[c * 2 * fixture::kChannels + c] = 1.0f;
        gain[c] = 1.0f;
    }
    MatrixView project_v = make_view(project, fixture::kChannels,
        2 * fixture::kChannels, 2 * fixture::kChannels);
    MatrixView gain_v = make_view(gain, 1, fixture::kChannels,
        fixture::kChannels);
    // Every x plane holds the 2x2 pattern [[1, 2], [3, 4]] (e4m3).
    for (int c = 0; c < 2 * fixture::kChannels; ++c) {
        x[(c * 2 + 0) * 2 + 0] = 1.0f;
        x[(c * 2 + 0) * 2 + 1] = 2.0f;
        x[(c * 2 + 1) * 2 + 0] = 3.0f;
        x[(c * 2 + 1) * 2 + 1] = 4.0f;
    }
    const FeatureBatchView x_v = make_batch(x, 2 * fixture::kChannels, 2, 2,
        4, 2);
    memset(skip, 0, sizeof skip);
    const FeatureBatchView skip_v = make_batch(skip, fixture::kChannels, 3, 5,
        15, 5);
    // The replication index arithmetic on the skip's 3x5 grid: row i reads
    // projected row min(i / 2, 1) and column j reads projected column
    // min(j / 2, 1) — the last projected row/column repeats at the clamp.
    CHECK(swin_upsample_blend(project_v, gain_v, x_v, skip_v,
        blend_dst, fixture::kChannels * 15,
        scratch, sizeof scratch / sizeof scratch[0]) == Status::kOk);
    const float expected[3][5] = {
        {1.0f, 1.0f, 2.0f, 2.0f, 2.0f},
        {1.0f, 1.0f, 2.0f, 2.0f, 2.0f},
        {3.0f, 3.0f, 4.0f, 4.0f, 4.0f},
    };
    for (int c = 0; c < fixture::kChannels; ++c) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 5; ++j) {
                CHECK(blend_dst[(c * 3 + i) * 5 + j] == expected[i][j]);
            }
        }
    }
    // The gated sum through the entry: gain 41.5 with skip 25 gives the
    // exact product 1037.5, and the projected value -2^-15 (weight 2^-9
    // against x -2^-6, both on the e4m3 grid, narrowed to binary16 exactly)
    // sits a quarter of a float32 ulp below the 1037.5 midpoint (2^-15
    // against the [1024, 2048) binade's 2^-13 ulp): the fused sum rounds
    // below the midpoint to 1037, where a float32-sum alternative rounds
    // back onto the midpoint and breaks the f16 tie to even (1038).
    memset(project, 0, sizeof project);
    project[0] = 0x1p-9f;
    for (int c = 0; c < fixture::kChannels; ++c) {
        gain[c] = 0.0f;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 5; ++j) {
                skip[(c * 3 + i) * 5 + j] = 0.0f;
            }
        }
    }
    gain[0] = 41.5f;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 5; ++j) {
            skip[i * 5 + j] = 25.0f;
        }
    }
    for (int c = 0; c < 2 * fixture::kChannels; ++c) {
        for (int r = 0; r < 2; ++r) {
            for (int col = 0; col < 2; ++col) {
                x[(c * 2 + r) * 2 + col] = c == 0 ? -0x1p-6f : 0.0f;
            }
        }
    }
    CHECK(swin_upsample_blend(project_v, gain_v, x_v, skip_v,
        blend_dst, fixture::kChannels * 15,
        scratch, sizeof scratch / sizeof scratch[0]) == Status::kOk);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 5; ++j) {
            CHECK(blend_dst[i * 5 + j] == 1037.0f);
        }
    }
    // The documented overflow band through the entry, with every operand
    // valid (skip finite binary16, gain binary16, x on the e4m3 grid): the
    // fused sum decides the threshold. 65504 + 15.5 = 65519.5 is below the
    // 65520 midpoint and stays on max-finite 65504; 65504 + 16 = 65520 is
    // exactly the ties-to-even midpoint (65504's mantissa is odd) and rounds
    // to infinity (arithmetic, not an error). Note 65520 is not itself a
    // binary16 value and cannot appear as an operand.
    memset(project, 0, sizeof project);
    project[0] = 0.875f;
    project[1] = 0.5f;
    gain[0] = 1.0f;
    for (int c = 0; c < 2 * fixture::kChannels; ++c) {
        for (int r = 0; r < 2; ++r) {
            for (int col = 0; col < 2; ++col) {
                x[(c * 2 + r) * 2 + col] = 0.0f;
            }
        }
    }
    // Projected (0, 0) = 0.875 * 16 + 0.5 * 3 = 15.5 (exactly binary16 after
    // the narrow); projected (0, 1) = 0.875 * 16 + 0.5 * 4 = 16. Rows/other
    // columns of x stay zero, so every other projected cell is 0.
    x[(0 * 2 + 0) * 2 + 0] = 16.0f;
    x[(0 * 2 + 0) * 2 + 1] = 16.0f;
    x[(1 * 2 + 0) * 2 + 0] = 3.0f;
    x[(1 * 2 + 0) * 2 + 1] = 4.0f;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 5; ++j) {
            skip[i * 5 + j] = 65504.0f;
        }
    }
    CHECK(swin_upsample_blend(project_v, gain_v, x_v, skip_v,
        blend_dst, fixture::kChannels * 15,
        scratch, sizeof scratch / sizeof scratch[0]) == Status::kOk);
    {
        // Replication: (i, j) reads projected (min(i / 2, 1), min(j / 2, 1)).
        const float expected_overflow[3][5] = {
            {65504.0f, 65504.0f, (float)INFINITY, (float)INFINITY,
                (float)INFINITY},
            {65504.0f, 65504.0f, (float)INFINITY, (float)INFINITY,
                (float)INFINITY},
            {65504.0f, 65504.0f, 65504.0f, 65504.0f, 65504.0f},
        };
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 5; ++j) {
                CHECK(blend_dst[i * 5 + j] == expected_overflow[i][j]);
            }
        }
    }
}

// ---- equivalence against the staged chain ---------------------------------------------------------

static void check_equivalence_downsample(void) {
    using dlss5_nr::pool2x2;
    using dlss5_nr::swin_block;
    using dlss5_nr::swin_downsample;
    fixture::BlockFixture f;
    f.zero_init();
    static float x[fixture::kChannels * 24 * 24];
    static float resample[2 * fixture::kChannels * fixture::kChannels];
    static float next_dst[2 * fixture::kChannels * 24 * 24];
    static float body_dst[fixture::kChannels * 24 * 24];
    static float expect_next[2 * fixture::kChannels * 24 * 24];
    static float expect_body[fixture::kChannels * 24 * 24];
    static float wide[fixture::kChannels * 24 * 24];
    static float acc[fixture::kChannels * 24 * 24];
    static float pooled[fixture::kChannels * 12 * 12];
    static float scratch[1 << 20];
    static float scratch2[1 << 20];
    const int rows = 24;
    const int cols = 24;
    const int out_rows = 6;
    const int out_cols = 20;
    uint32_t state = 0x2468ACE1u;
    for (int config = 0; config < 4; ++config) {
        // config 0/2 tight, config 1/3 padded with kPoison gap columns (the
        // entry's own row-stride read of the projection weights); the
        // zero-weight corpus on 0/1, randomised weights and corpus on 2/3.
        const bool padded = (config % 2) == 1;
        if (config < 2) {
            f.zero_init();
            memset(resample, 0, sizeof resample);
            for (int n = 0; n < 2 * fixture::kChannels; ++n) {
                resample[n * fixture::kChannels + (n % fixture::kChannels)]
                    = 1.0f;
            }
        } else {
            f.randomise(&state);
            for (std::size_t p = 0;
                 p < sizeof resample / sizeof resample[0]; ++p) {
                resample[p] = rnd_e4m3(&state);
            }
        }
        for (std::size_t p = 0;
             p < fixture::kChannels * (std::size_t)rows * cols; ++p) {
            x[p] = rnd_f16(&state);
        }
        static float resample_padded[2 * fixture::kChannels
            * (fixture::kChannels + 4)];
        scatter_padded(resample_padded, resample, 1, 2 * fixture::kChannels,
            fixture::kChannels,
            2 * fixture::kChannels * (fixture::kChannels + 4),
            fixture::kChannels + 4);
        const MatrixView resample_v = padded
            ? make_view(resample_padded, 2 * fixture::kChannels,
                fixture::kChannels, fixture::kChannels + 4)
            : make_view(resample, 2 * fixture::kChannels,
                fixture::kChannels, fixture::kChannels);
        const FeatureBatchView x_v = make_batch(x, fixture::kChannels,
            rows, cols, rows * cols, cols);
        for (int shift_rows = 0; shift_rows >= -1; --shift_rows) {
            for (int shift_cols = 0; shift_cols >= -1; --shift_cols) {
                CHECK(swin_downsample(x_v, shift_rows, shift_cols, f.params,
                    resample_v, out_rows, out_cols,
                    next_dst, 2 * fixture::kChannels * out_rows * out_cols,
                    body_dst, fixture::kChannels * rows * cols,
                    scratch, sizeof scratch / sizeof scratch[0])
                    == Status::kOk);
                // The staged chain, written out directly: the block in
                // accumulator mode, pool2x2 per plane, the k-order projection
                // with the stored-result rounding, the top-left resize and
                // round8(wide).
                CHECK(swin_block(x_v, nullptr, shift_rows, shift_cols,
                    f.params, wide, fixture::kChannels * rows * cols,
                    acc, fixture::kChannels * rows * cols,
                    scratch2, sizeof scratch2 / sizeof scratch2[0])
                    == Status::kOk);
                for (int c = 0; c < fixture::kChannels; ++c) {
                    dlss5_nr::PlaneView plane;
                    plane.data = acc + (std::size_t)c * rows * cols;
                    plane.rows = rows;
                    plane.cols = cols;
                    plane.row_stride = cols;
                    CHECK(pool2x2(plane, pooled + (std::size_t)c * (rows / 2)
                        * (cols / 2), (rows / 2) * (cols / 2))
                        == Status::kOk);
                }
                for (int n = 0; n < 2 * fixture::kChannels; ++n) {
                    float* out_plane = expect_next
                        + (std::size_t)n * out_rows * out_cols;
                    for (int oy = 0; oy < out_rows; ++oy) {
                        for (int ox = 0; ox < out_cols; ++ox) {
                            float value = 0.0f;
                            if (oy < rows / 2 && ox < cols / 2) {
                                const float* row = resample_v.data
                                    + (std::size_t)n
                                        * (std::size_t)resample_v.row_stride;
                                value = 0.0f;
                                for (int k = 0; k < fixture::kChannels; ++k) {
                                    value += row[k]
                                        * pooled[(std::size_t)k * (rows / 2)
                                            * (cols / 2)
                                            + (std::size_t)oy * (cols / 2)
                                            + ox];
                                }
                                value = dlss5_nr::swin_store(value);
                            }
                            out_plane[(std::size_t)oy * out_cols + ox] = value;
                        }
                    }
                }
                for (std::size_t p = 0;
                     p < fixture::kChannels * (std::size_t)rows * cols; ++p) {
                    expect_body[p] = dlss5_nr::e4m3(wide[p]);
                }
                CHECK(memcmp(next_dst, expect_next,
                    sizeof(float) * 2 * fixture::kChannels * out_rows
                        * out_cols) == 0);
                CHECK(memcmp(body_dst, expect_body,
                    sizeof(float) * fixture::kChannels * rows * cols) == 0);
                // Determinism: a repeat run is bit-identical.
                CHECK(swin_downsample(x_v, shift_rows, shift_cols, f.params,
                    resample_v, out_rows, out_cols,
                    expect_next, 2 * fixture::kChannels * out_rows * out_cols,
                    expect_body, fixture::kChannels * rows * cols,
                    scratch2, sizeof scratch2 / sizeof scratch2[0])
                    == Status::kOk);
                CHECK(memcmp(next_dst, expect_next,
                    sizeof(float) * 2 * fixture::kChannels * out_rows
                        * out_cols) == 0);
                CHECK(memcmp(body_dst, expect_body,
                    sizeof(float) * fixture::kChannels * rows * cols) == 0);
            }
        }
    }
}

static void check_equivalence_upsample_blend(void) {
    using dlss5_nr::gated_sum_element;
    using dlss5_nr::swin_upsample_blend;
    static float project[fixture::kChannels * 2 * fixture::kChannels];
    static float gain[fixture::kChannels];
    static float x[2 * fixture::kChannels * 9 * 7];
    static float skip[fixture::kChannels * 21 * 19];
    static float x_padded[2 * fixture::kChannels * 111];
    static float skip_padded[fixture::kChannels * 507];
    static float blend_dst[fixture::kChannels * 21 * 19];
    static float expect[fixture::kChannels * 21 * 19];
    static float projected[fixture::kChannels * 9 * 7];
    static float scratch[1 << 18];
    const int hx = 9;
    const int wx = 7;
    const int hs = 21;   // larger than 2 * hx - 1: the clamp repeats edges
    const int ws = 19;
    uint32_t state = 0x13579BDFu;
    for (int config = 0; config < 3; ++config) {
        for (std::size_t p = 0;
             p < sizeof project / sizeof project[0]; ++p) {
            project[p] = rnd_e4m3(&state);
        }
        for (int c = 0; c < fixture::kChannels; ++c) {
            gain[c] = dlss5_nr::f16_from_bits(
                static_cast<uint16_t>(rng_next(&state) & 0x7BFFu));
        }
        for (std::size_t p = 0;
             p < 2 * fixture::kChannels * (std::size_t)hx * wx; ++p) {
            x[p] = rnd_e4m3(&state);
        }
        for (std::size_t p = 0;
             p < fixture::kChannels * (std::size_t)hs * ws; ++p) {
            skip[p] = rnd_f16(&state);
        }
        scatter_padded(x_padded, x, 2 * fixture::kChannels, hx, wx, 111, 12);
        scatter_padded(skip_padded, skip, fixture::kChannels, hs, ws, 507, 24);
        MatrixView project_v = make_view(project, fixture::kChannels,
            2 * fixture::kChannels, 2 * fixture::kChannels);
        MatrixView gain_v = make_view(gain, 1, fixture::kChannels,
            fixture::kChannels);
        // Tight and padded input strides (gaps at kPoison), both bit-exact
        // against the same staged chain.
        for (int pad = 0; pad < 2; ++pad) {
            const FeatureBatchView x_v = pad
                ? make_batch(x_padded, 2 * fixture::kChannels,
                    hx, wx, 111, 12)
                : make_batch(x, 2 * fixture::kChannels, hx, wx, hx * wx, wx);
            const FeatureBatchView skip_v = pad
                ? make_batch(skip_padded, fixture::kChannels, hs, ws, 507, 24)
                : make_batch(skip, fixture::kChannels, hs, ws, hs * ws, ws);
            CHECK(swin_upsample_blend(project_v, gain_v, x_v, skip_v,
                blend_dst, fixture::kChannels * hs * ws,
                scratch, sizeof scratch / sizeof scratch[0]) == Status::kOk);
            // The staged chain written out directly: the k-order projection
            // with one binary16 rounding, the clamped replication indices and
            // the gated sum primitive, every input read through its strides.
            for (int c = 0; c < fixture::kChannels; ++c) {
                const float* row = project
                    + (std::size_t)c * (std::size_t)project_v.row_stride;
                for (int r = 0; r < hx; ++r) {
                    for (int col = 0; col < wx; ++col) {
                        float sum = 0.0f;
                        for (int k = 0; k < 2 * fixture::kChannels; ++k) {
                            sum += row[k] * x_v.data[
                                (std::size_t)k
                                    * (std::size_t)x_v.plane_stride
                                + (std::size_t)r
                                    * (std::size_t)x_v.row_stride
                                + (std::size_t)col];
                        }
                        projected[(std::size_t)c * hx * wx
                            + (std::size_t)r * wx + col] = dlss5_nr::f16(sum);
                    }
                }
            }
            for (int c = 0; c < fixture::kChannels; ++c) {
                for (int i = 0; i < hs; ++i) {
                    const int src_r = i / 2 < hx ? i / 2 : hx - 1;
                    for (int j = 0; j < ws; ++j) {
                        const int src_c = j / 2 < wx ? j / 2 : wx - 1;
                        expect[(std::size_t)c * hs * ws
                            + (std::size_t)i * ws + j] = gated_sum_element(
                                skip_v.data[(std::size_t)c
                                        * (std::size_t)skip_v.plane_stride
                                    + (std::size_t)i
                                        * (std::size_t)skip_v.row_stride
                                    + (std::size_t)j],
                                gain[c],
                                projected[(std::size_t)c * hx * wx
                                    + (std::size_t)src_r * wx + src_c]);
                    }
                }
            }
            CHECK(memcmp(blend_dst, expect,
                sizeof(float) * fixture::kChannels * hs * ws) == 0);
            // Determinism.
            CHECK(swin_upsample_blend(project_v, gain_v, x_v, skip_v,
                expect, fixture::kChannels * hs * ws,
                scratch, sizeof scratch / sizeof scratch[0]) == Status::kOk);
            CHECK(memcmp(blend_dst, expect,
                sizeof(float) * fixture::kChannels * hs * ws) == 0);
        }
    }
    // A crop configuration (skip smaller than the replicated map).
    const int hs2 = 4;
    const int ws2 = 5;
    static float skip2[fixture::kChannels * 4 * 5];
    static float blend2[fixture::kChannels * 4 * 5];
    static float expect2[fixture::kChannels * 4 * 5];
    for (std::size_t p = 0; p < fixture::kChannels * 20u; ++p) {
        skip2[p] = rnd_f16(&state);
    }
    const FeatureBatchView x_v = make_batch(x, 2 * fixture::kChannels,
        hx, wx, hx * wx, wx);
    const FeatureBatchView skip2_v = make_batch(skip2, fixture::kChannels,
        hs2, ws2, hs2 * ws2, ws2);
    MatrixView project_v = make_view(project, fixture::kChannels,
        2 * fixture::kChannels, 2 * fixture::kChannels);
    MatrixView gain_v = make_view(gain, 1, fixture::kChannels,
        fixture::kChannels);
    CHECK(swin_upsample_blend(project_v, gain_v, x_v, skip2_v,
        blend2, fixture::kChannels * 20,
        scratch, sizeof scratch / sizeof scratch[0]) == Status::kOk);
    for (int c = 0; c < fixture::kChannels; ++c) {
        for (int i = 0; i < hs2; ++i) {
            for (int j = 0; j < ws2; ++j) {
                expect2[(std::size_t)c * hs2 * ws2
                    + (std::size_t)i * ws2 + j] = gated_sum_element(
                        skip2[(std::size_t)c * hs2 * ws2
                            + (std::size_t)i * ws2 + j],
                        gain[c],
                        projected[(std::size_t)c * hx * wx
                            + (std::size_t)(i / 2) * wx + j / 2]);
            }
        }
    }
    CHECK(memcmp(blend2, expect2,
        sizeof(float) * fixture::kChannels * hs2 * ws2) == 0);
}

// ---- the layer compositions ----------------------------------------------------------------------

static void check_layer_plain(void) {
    using dlss5_nr::swin_layer_plain;
    fixture::BlockFixture f;
    f.zero_init();
    static float x[fixture::kChannels * 8 * 8];
    static float out_dst[fixture::kChannels * 8 * 8];
    static float expect[fixture::kChannels * 8 * 8];
    static float wide[fixture::kChannels * 8 * 8];
    static float scratch[1 << 20];
    static float scratch2[1 << 20];
    uint32_t state = 0x55AA33CCu;
    for (std::size_t p = 0; p < fixture::kChannels * 64u; ++p) {
        x[p] = rnd_f16(&state);
    }
    const FeatureBatchView x_v = make_batch(x, fixture::kChannels, 8, 8,
        64, 8);
    CHECK(swin_layer_plain(x_v, 0, -1, f.params,
        out_dst, fixture::kChannels * 64,
        scratch, sizeof scratch / sizeof scratch[0]) == Status::kOk);
    // The staged chain: the block's wide output rounded to the e4m3 grid.
    CHECK(dlss5_nr::swin_block(x_v, nullptr, 0, -1, f.params,
        wide, fixture::kChannels * 64, nullptr, 0,
        scratch2, sizeof scratch2 / sizeof scratch2[0]) == Status::kOk);
    for (std::size_t p = 0; p < fixture::kChannels * 64u; ++p) {
        expect[p] = dlss5_nr::e4m3(wide[p]);
    }
    CHECK(memcmp(out_dst, expect,
        sizeof(float) * fixture::kChannels * 64) == 0);
    // Determinism.
    CHECK(swin_layer_plain(x_v, 0, -1, f.params,
        expect, fixture::kChannels * 64,
        scratch2, sizeof scratch2 / sizeof scratch2[0]) == Status::kOk);
    CHECK(memcmp(out_dst, expect,
        sizeof(float) * fixture::kChannels * 64) == 0);
}

static void check_layer_upsample(void) {
    using dlss5_nr::swin_layer_upsample;
    // The C == 32 f16-residual path discriminator. The blend is the gated
    // sum narrowed to binary16, so the pinned value must be binary16-exact:
    // 367.75 (skip 367.75, gain 1, projected zero) with the f16 scale
    // 1 + 2^-10.
    //   C == 32: the blend doubles as the MLP residual unrounded by e4m3,
    //   y = 367.75 * (1 + 2^-10) = 368.109130859375, wide = f16(y) = 368.0,
    //   out = e4m3(368.0) = 384 (the 352/384 midpoint ties to even).
    //   C == 64: round8(blend) = e4m3(367.75) = 352 is the residual,
    //   y = 352 * (1 + 2^-10) = 352.34375, the multi-head skip keeps
    //   store(y) = 352, out = e4m3 = 352.
    fixture::BlockFixture f32;
    f32.zero_init();
    fixture::WideFixture f64;
    f64.zero_init();
    for (int c = 0; c < fixture::kChannels; ++c) {
        f32.mlp_scale[c] = 1.0f + 0x1p-10f;
    }
    for (int c = 0; c < fixture::kWideChannels; ++c) {
        f64.mlp_scale[c] = 1.0f + 0x1p-10f;
    }
    static float project32[fixture::kChannels * 2 * fixture::kChannels];
    static float project64[fixture::kWideChannels * 2 * fixture::kWideChannels];
    static float gain32[fixture::kChannels];
    static float gain64[fixture::kWideChannels];
    static float x32[2 * fixture::kChannels * 4 * 4];
    static float x64[2 * fixture::kWideChannels * 4 * 4];
    static float skip32[fixture::kChannels * 4 * 4];
    static float skip64[fixture::kWideChannels * 4 * 4];
    static float out32[fixture::kChannels * 16];
    static float out64[fixture::kWideChannels * 16];
    static float scratch[1 << 20];
    memset(project32, 0, sizeof project32);
    memset(project64, 0, sizeof project64);
    memset(x32, 0, sizeof x32);
    memset(x64, 0, sizeof x64);
    for (int c = 0; c < fixture::kChannels; ++c) {
        gain32[c] = 1.0f;
    }
    for (int c = 0; c < fixture::kWideChannels; ++c) {
        gain64[c] = 1.0f;
    }
    for (int c = 0; c < fixture::kChannels; ++c) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                skip32[(c * 4 + i) * 4 + j] = 367.75f;
            }
        }
    }
    for (int c = 0; c < fixture::kWideChannels; ++c) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                skip64[(c * 4 + i) * 4 + j] = 367.75f;
            }
        }
    }
    MatrixView project32_v = make_view(project32, fixture::kChannels,
        2 * fixture::kChannels, 2 * fixture::kChannels);
    MatrixView gain32_v = make_view(gain32, 1, fixture::kChannels,
        fixture::kChannels);
    const FeatureBatchView x32_v = make_batch(x32, 2 * fixture::kChannels,
        4, 4, 16, 4);
    const FeatureBatchView skip32_v = make_batch(skip32, fixture::kChannels,
        4, 4, 16, 4);
    CHECK(swin_layer_upsample(x32_v, skip32_v, 0, 0, f32.params,
        project32_v, gain32_v, out32, fixture::kChannels * 16,
        scratch, sizeof scratch / sizeof scratch[0]) == Status::kOk);
    for (std::size_t p = 0; p < fixture::kChannels * 16u; ++p) {
        CHECK(out32[p] == 384.0f);
    }
    MatrixView project64_v = make_view(project64, fixture::kWideChannels,
        2 * fixture::kWideChannels, 2 * fixture::kWideChannels);
    MatrixView gain64_v = make_view(gain64, 1, fixture::kWideChannels,
        fixture::kWideChannels);
    const FeatureBatchView x64_v = make_batch(x64, 2 * fixture::kWideChannels,
        4, 4, 16, 4);
    const FeatureBatchView skip64_v = make_batch(skip64, fixture::kWideChannels,
        4, 4, 16, 4);
    CHECK(swin_layer_upsample(x64_v, skip64_v, 0, 0, f64.params,
        project64_v, gain64_v, out64, fixture::kWideChannels * 16,
        scratch, sizeof scratch / sizeof scratch[0]) == Status::kOk);
    for (std::size_t p = 0; p < fixture::kWideChannels * 16u; ++p) {
        CHECK(out64[p] == 352.0f);
    }
    // The disclosed extreme-overflow class: valid operands whose fused sum
    // lands on the 65520 threshold round the blend element to infinity, and
    // the layer rejects it before the block call; the outputs stay untouched.
    // skip 65504 (max finite binary16), gain 1, projected 16 (project weight
    // 1 on the x plane value 16) give exactly 65504 + 16 = 65520. (A skip of
    // 65520 itself is not a binary16 value and is rejected earlier as an
    // invalid operand.)
    fill_sentinel(out32, fixture::kChannels * 16);
    project32[0] = 1.0f;
    x32[0] = 16.0f;
    skip32[0] = 65504.0f;
    CHECK(swin_layer_upsample(x32_v, skip32_v, 0, 0, f32.params,
        project32_v, gain32_v, out32, fixture::kChannels * 16,
        scratch, sizeof scratch / sizeof scratch[0])
        == Status::kInvalidValues);
    CHECK(all_sentinel(out32, fixture::kChannels * 16));
    project32[0] = 0.0f;
    x32[0] = 0.0f;
    skip32[0] = 367.75f;
}

// ---- the layer compositions, staged-chain equivalence ----------------------------------------------

static void check_equivalence_layer_plain(void) {
    using dlss5_nr::swin_block;
    using dlss5_nr::swin_layer_plain;
    fixture::BlockFixture f;
    static float x[fixture::kChannels * 8 * 8];
    static float x_padded[fixture::kChannels * 107];
    static float out_dst[fixture::kChannels * 64];
    static float expect[fixture::kChannels * 64];
    static float wide[fixture::kChannels * 64];
    static float scratch[1 << 20];
    static float scratch2[1 << 20];
    uint32_t state = 0x0F1E2D3Cu;
    for (int config = 0; config < 2; ++config) {
        if (config == 0) {
            f.zero_init();
        } else {
            f.randomise(&state);
        }
        for (std::size_t p = 0; p < fixture::kChannels * 64u; ++p) {
            x[p] = rnd_f16(&state);
        }
        scatter_padded(x_padded, x, fixture::kChannels, 8, 8, 107, 13);
        for (int pad = 0; pad < 2; ++pad) {
            const FeatureBatchView x_v = pad
                ? make_batch(x_padded, fixture::kChannels, 8, 8, 107, 13)
                : make_batch(x, fixture::kChannels, 8, 8, 64, 8);
            for (int shift_rows = 0; shift_rows >= -1; --shift_rows) {
                for (int shift_cols = 0; shift_cols >= -1; --shift_cols) {
                    CHECK(swin_layer_plain(x_v, shift_rows, shift_cols, f.params,
                        out_dst, fixture::kChannels * 64,
                        scratch, sizeof scratch / sizeof scratch[0])
                        == Status::kOk);
                    // The staged chain: the block's wide output rounded to
                    // the e4m3 grid, from an independent block call.
                    CHECK(swin_block(x_v, nullptr, shift_rows, shift_cols,
                        f.params, wide, fixture::kChannels * 64, nullptr, 0,
                        scratch2, sizeof scratch2 / sizeof scratch2[0])
                        == Status::kOk);
                    for (std::size_t p = 0; p < fixture::kChannels * 64u; ++p) {
                        expect[p] = dlss5_nr::e4m3(wide[p]);
                    }
                    CHECK(memcmp(out_dst, expect,
                        sizeof(float) * fixture::kChannels * 64) == 0);
                    // Determinism: a repeat run is bit-identical.
                    CHECK(swin_layer_plain(x_v, shift_rows, shift_cols, f.params,
                        expect, fixture::kChannels * 64,
                        scratch2, sizeof scratch2 / sizeof scratch2[0])
                        == Status::kOk);
                    CHECK(memcmp(out_dst, expect,
                        sizeof(float) * fixture::kChannels * 64) == 0);
                }
            }
        }
    }
}

// The staged-chain equivalence for `swin_layer_upsample` at one fixture
// width: C == 32 (the f16-residual path) or C == 64 (the quantised-residual
// path). Runs both extents (exact 2x replication and the clamp edge) and both
// stride layouts over every shift pair. The blend stage's replication and
// stride reads are written out directly here so a stride defect in the
// entry's blend could not hide behind the same code on both sides; the
// elementwise gated-sum primitive (`gated_sum_element`) IS shared with the
// entry — its arithmetic is pinned by the hand-derived literals in
// `check_gated_sum`, not by this equivalence.
static void upsample_layer_equivalence(const SwinBlockParams& params,
                                       int channels, uint32_t* state) {
    using dlss5_nr::gated_sum_element;
    using dlss5_nr::swin_block;
    using dlss5_nr::swin_layer_upsample;
    static float x_tight[2 * 64 * 4 * 4];
    static float x_padded[2 * 64 * 39];
    static float skip_tight[64 * 8 * 8];
    static float skip_padded[64 * 107];
    static float project[64 * 128];
    static float project_padded[64 * 132];
    static float gain[64];
    static float gain_padded[68];
    static float projected[64 * 4 * 4];
    static float blend[64 * 8 * 8];
    static float wide[64 * 8 * 8];
    static float expect[64 * 8 * 8];
    static float out_dst[64 * 8 * 8];
    static float scratch[1 << 20];
    static float scratch2[1 << 20];
    for (int clamp_extents = 0; clamp_extents < 2; ++clamp_extents) {
        // Exact replication (4 -> 8) and the clamp edge (3 -> 6 into 8: the
        // last projected row/column repeats).
        const int hx = clamp_extents ? 3 : 4;
        const int wx = clamp_extents ? 3 : 4;
        for (int p = 0; p < 2 * channels * hx * wx; ++p) {
            x_tight[p] = rnd_e4m3(state);
        }
        for (int p = 0; p < channels * 64; ++p) {
            skip_tight[p] = rnd_skip(state);
        }
        for (int p = 0; p < channels * 2 * channels; ++p) {
            project[p] = rnd_e4m3(state);
        }
        for (int c = 0; c < channels; ++c) {
            gain[c] = rnd_gain(state);
        }
        const std::ptrdiff_t x_pad_rs = wx + 5;
        const std::ptrdiff_t x_pad_ps = hx * x_pad_rs + 3;
        scatter_padded(x_padded, x_tight, 2 * channels, hx, wx,
                       x_pad_ps, x_pad_rs);
        scatter_padded(skip_padded, skip_tight, channels, 8, 8, 107, 13);
        // Padded matrix row strides (kPoison in the trailing columns).
        for (int r = 0; r < channels; ++r) {
            for (int j = 0; j < 2 * channels + 4; ++j) {
                project_padded[r * (2 * channels + 4) + j] = kPoison;
            }
            for (int j = 0; j < 2 * channels; ++j) {
                project_padded[r * (2 * channels + 4) + j] =
                    project[r * 2 * channels + j];
            }
        }
        for (int j = 0; j < channels + 4; ++j) {
            gain_padded[j] = kPoison;
        }
        for (int j = 0; j < channels; ++j) {
            gain_padded[j] = gain[j];
        }
        for (int pad = 0; pad < 2; ++pad) {
            const FeatureBatchView x_v = pad
                ? make_batch(x_padded, 2 * channels, hx, wx, x_pad_ps, x_pad_rs)
                : make_batch(x_tight, 2 * channels, hx, wx, hx * wx, wx);
            const FeatureBatchView skip_v = pad
                ? make_batch(skip_padded, channels, 8, 8, 107, 13)
                : make_batch(skip_tight, channels, 8, 8, 64, 8);
            const MatrixView project_v = pad
                ? make_view(project_padded, channels, 2 * channels,
                    2 * channels + 4)
                : make_view(project, channels, 2 * channels, 2 * channels);
            const MatrixView gain_v = pad
                ? make_view(gain_padded, 1, channels, channels + 4)
                : make_view(gain, 1, channels, channels);
            for (int shift_rows = 0; shift_rows >= -1; --shift_rows) {
                for (int shift_cols = 0; shift_cols >= -1; --shift_cols) {
                    CHECK(swin_layer_upsample(x_v, skip_v, shift_rows,
                        shift_cols, params, project_v, gain_v,
                        out_dst, (std::size_t)channels * 64,
                        scratch, sizeof scratch / sizeof scratch[0])
                        == Status::kOk);
                    // The staged chain written out directly: the learned
                    // projection (k-order float32 sum, one binary16
                    // rounding), the clamped nearest-2x replication, the
                    // gated sum, then the block on the blend (the f16
                    // residual when C == 32, the e4m3-rounded blend
                    // otherwise) and the final e4m3 rounding — every input
                    // read through its strides.
                    for (int c = 0; c < channels; ++c) {
                        const float* row = project_v.data
                            + (std::size_t)c
                                * (std::size_t)project_v.row_stride;
                        for (int r = 0; r < hx; ++r) {
                            for (int col = 0; col < wx; ++col) {
                                float sum = 0.0f;
                                for (int k = 0; k < 2 * channels; ++k) {
                                    sum += row[k] * x_v.data[
                                        (std::size_t)k
                                            * (std::size_t)x_v.plane_stride
                                        + (std::size_t)r
                                            * (std::size_t)x_v.row_stride
                                        + (std::size_t)col];
                                }
                                projected[(std::size_t)c * hx * wx
                                    + (std::size_t)r * wx + col] =
                                    dlss5_nr::f16(sum);
                            }
                        }
                    }
                    for (int c = 0; c < channels; ++c) {
                        for (int i = 0; i < 8; ++i) {
                            const int src_r = i / 2 < hx ? i / 2 : hx - 1;
                            for (int j = 0; j < 8; ++j) {
                                const int src_c = j / 2 < wx ? j / 2 : wx - 1;
                                blend[(std::size_t)c * 64
                                    + (std::size_t)i * 8 + j] =
                                    gated_sum_element(
                                        skip_v.data[(std::size_t)c
                                                * (std::size_t)skip_v.plane_stride
                                            + (std::size_t)i
                                                * (std::size_t)skip_v.row_stride
                                            + (std::size_t)j],
                                        gain_v.data[(std::size_t)c],
                                        projected[(std::size_t)c * hx * wx
                                            + (std::size_t)src_r * wx + src_c]);
                            }
                        }
                    }
                    const FeatureBatchView blend_v = make_batch(blend,
                        channels, 8, 8, 64, 8);
                    if (channels == 32) {
                        CHECK(swin_block(blend_v, &blend_v, shift_rows,
                            shift_cols, params, wide,
                            (std::size_t)channels * 64, nullptr, 0, scratch2,
                            sizeof scratch2 / sizeof scratch2[0])
                            == Status::kOk);
                    } else {
                        for (int p = 0; p < channels * 64; ++p) {
                            blend[p] = dlss5_nr::e4m3(blend[p]);
                        }
                        CHECK(swin_block(blend_v, nullptr, shift_rows,
                            shift_cols, params, wide,
                            (std::size_t)channels * 64, nullptr, 0, scratch2,
                            sizeof scratch2 / sizeof scratch2[0])
                            == Status::kOk);
                    }
                    for (int p = 0; p < channels * 64; ++p) {
                        expect[p] = dlss5_nr::e4m3(wide[p]);
                    }
                    CHECK(memcmp(out_dst, expect,
                        sizeof(float) * (std::size_t)channels * 64) == 0);
                    // Determinism: a repeat run is bit-identical.
                    CHECK(swin_layer_upsample(x_v, skip_v, shift_rows,
                        shift_cols, params, project_v, gain_v,
                        expect, (std::size_t)channels * 64,
                        scratch2, sizeof scratch2 / sizeof scratch2[0])
                        == Status::kOk);
                    CHECK(memcmp(out_dst, expect,
                        sizeof(float) * (std::size_t)channels * 64) == 0);
                }
            }
        }
    }
}

static void check_equivalence_layer_upsample(void) {
    fixture::BlockFixture f32;
    fixture::WideFixture f64;
    uint32_t state = 0x778899AAu;
    for (int config = 0; config < 2; ++config) {
        if (config == 0) {
            f32.zero_init();
            f64.zero_init();
        } else {
            f32.randomise(&state);
            f64.randomise(&state);
        }
        upsample_layer_equivalence(f32.params, fixture::kChannels, &state);
        upsample_layer_equivalence(f64.params, fixture::kWideChannels, &state);
    }
}

// ---- negative controls ---------------------------------------------------------------------------

static void check_negative_downsample(void) {
    using dlss5_nr::swin_downsample;
    fixture::BlockFixture f;
    f.zero_init();
    static float x[fixture::kChannels * 8 * 8];
    static float resample[2 * fixture::kChannels * fixture::kChannels];
    static float next_dst[2 * fixture::kChannels * 4 * 4];
    static float body_dst[fixture::kChannels * 8 * 8];
    static float scratch[1 << 20];
    memset(resample, 0, sizeof resample);
    for (int n = 0; n < 2 * fixture::kChannels; ++n) {
        resample[n * fixture::kChannels + (n % fixture::kChannels)] = 1.0f;
    }
    memset(x, 0, sizeof x);
    MatrixView resample_v = make_view(resample, 2 * fixture::kChannels,
        fixture::kChannels, fixture::kChannels);
    const FeatureBatchView x_v = make_batch(x, fixture::kChannels, 8, 8,
        64, 8);
    const std::size_t next_cap = 2 * fixture::kChannels * 16u;
    const std::size_t body_cap = fixture::kChannels * 64u;
    const std::size_t scratch_cap = sizeof scratch / sizeof scratch[0];
    auto untouched = [&]() {
        CHECK(all_sentinel(next_dst, next_cap));
        CHECK(all_sentinel(body_dst, body_cap));
        CHECK(all_sentinel(scratch, scratch_cap));
    };
    fill_sentinel(next_dst, next_cap);
    fill_sentinel(body_dst, body_cap);
    fill_sentinel(scratch, scratch_cap);
    // Null classes.
    CHECK(swin_downsample(make_batch(nullptr, fixture::kChannels, 8, 8, 64, 8),
        0, 0, f.params, resample_v, 4, 4, next_dst, next_cap,
        body_dst, body_cap, scratch, scratch_cap) == Status::kNullPointer);
    CHECK(swin_downsample(x_v, 0, 0, f.params, resample_v, 4, 4,
        nullptr, next_cap, body_dst, body_cap, scratch, scratch_cap)
        == Status::kNullPointer);
    CHECK(swin_downsample(x_v, 0, 0, f.params, resample_v, 4, 4,
        next_dst, next_cap, nullptr, body_cap, scratch, scratch_cap)
        == Status::kNullPointer);
    CHECK(swin_downsample(x_v, 0, 0, f.params, resample_v, 4, 4,
        next_dst, next_cap, body_dst, body_cap, nullptr, scratch_cap)
        == Status::kNullPointer);
    CHECK(swin_downsample(x_v, 0, 0, f.params,
        make_view(nullptr, 2 * fixture::kChannels, fixture::kChannels,
            fixture::kChannels),
        4, 4, next_dst, next_cap, body_dst, body_cap, scratch, scratch_cap)
        == Status::kNullPointer);
    SwinBlockParams p = f.params;
    p.expand.data = nullptr;
    CHECK(swin_downsample(x_v, 0, 0, p, resample_v, 4, 4,
        next_dst, next_cap, body_dst, body_cap, scratch, scratch_cap)
        == Status::kNullPointer);
    SwinBlockParams null_mid = f.params;
    static MatrixView null_mid_view = make_view(nullptr, fixture::kChannels,
        fixture::kHidden, fixture::kHidden);
    null_mid.mid = &null_mid_view;
    CHECK(swin_downsample(x_v, 0, 0, null_mid, resample_v, 4, 4,
        next_dst, next_cap, body_dst, body_cap, scratch, scratch_cap)
        == Status::kNullPointer);
    untouched();
    // Dimension classes: odd extents (pool2x2's even-extent contract).
    CHECK(swin_downsample(make_batch(x, fixture::kChannels, 7, 8, 64, 8),
        0, 0, f.params, resample_v, 4, 4, next_dst, next_cap,
        body_dst, body_cap, scratch, scratch_cap)
        == Status::kInvalidDimensions);
    CHECK(swin_downsample(make_batch(x, fixture::kChannels, 8, 7, 64, 8),
        0, 0, f.params, resample_v, 4, 4, next_dst, next_cap,
        body_dst, body_cap, scratch, scratch_cap)
        == Status::kInvalidDimensions);
    CHECK(swin_downsample(make_batch(x, 0, 8, 8, 64, 8),
        0, 0, f.params, resample_v, 4, 4, next_dst, next_cap,
        body_dst, body_cap, scratch, scratch_cap)
        == Status::kInvalidDimensions);
    CHECK(swin_downsample(make_batch(x, fixture::kChannels, 0, 8, 64, 8),
        0, 0, f.params, resample_v, 4, 4, next_dst, next_cap,
        body_dst, body_cap, scratch, scratch_cap)
        == Status::kInvalidDimensions);
    untouched();
    // Stride classes.
    CHECK(swin_downsample(make_batch(x, fixture::kChannels, 8, 8, 64, 7),
        0, 0, f.params, resample_v, 4, 4, next_dst, next_cap,
        body_dst, body_cap, scratch, scratch_cap)
        == Status::kInvalidStride);
    CHECK(swin_downsample(make_batch(x, fixture::kChannels, 8, 8, 63, 8),
        0, 0, f.params, resample_v, 4, 4, next_dst, next_cap,
        body_dst, body_cap, scratch, scratch_cap)
        == Status::kInvalidStride);
    untouched();
    // Pairing classes: channels vs heads, resample shape and the mid
    // pairing (mid [32, 32] is the single-head accept edge, exercised
    // below; a mid row mismatch is the pairing rejection here).
    p = f.params;
    p.heads = 2;
    CHECK(swin_downsample(x_v, 0, 0, p, resample_v, 4, 4,
        next_dst, next_cap, body_dst, body_cap, scratch, scratch_cap)
        == Status::kInvalidDimensions);
    CHECK(swin_downsample(x_v, 0, 0, f.params,
        make_view(resample, 2 * fixture::kChannels - 1, fixture::kChannels,
            fixture::kChannels),
        4, 4, next_dst, next_cap, body_dst, body_cap, scratch, scratch_cap)
        == Status::kInvalidDimensions);
    CHECK(swin_downsample(x_v, 0, 0, f.params,
        make_view(resample, 2 * fixture::kChannels, fixture::kChannels - 1,
            fixture::kChannels),
        4, 4, next_dst, next_cap, body_dst, body_cap, scratch, scratch_cap)
        == Status::kInvalidDimensions);
    static float mid_table[fixture::kChannels * fixture::kChannels];
    memset(mid_table, 0, sizeof mid_table);
    static MatrixView mid_bad_view = make_view(mid_table,
        fixture::kChannels - 1, fixture::kChannels, fixture::kChannels);
    SwinBlockParams mid_p = f.params;
    mid_p.mid = &mid_bad_view;
    CHECK(swin_downsample(x_v, 0, 0, mid_p, resample_v, 4, 4,
        next_dst, next_cap, body_dst, body_cap, scratch, scratch_cap)
        == Status::kInvalidDimensions);
    untouched();
    // Value classes: an off-grid resample operand, and the forwarded
    // input-map value scan (the block call is the first write and is atomic).
    resample[0] = 0.1f;
    CHECK(swin_downsample(x_v, 0, 0, f.params, resample_v, 4, 4,
        next_dst, next_cap, body_dst, body_cap, scratch, scratch_cap)
        == Status::kInvalidValues);
    resample[0] = 1.0f;
    x[0] = (float)NAN;
    CHECK(swin_downsample(x_v, 0, 0, f.params, resample_v, 4, 4,
        next_dst, next_cap, body_dst, body_cap, scratch, scratch_cap)
        == Status::kInvalidValues);
    x[0] = 0.0f;
    untouched();
    // Shift classes.
    CHECK(swin_downsample(x_v, 1, 0, f.params, resample_v, 4, 4,
        next_dst, next_cap, body_dst, body_cap, scratch, scratch_cap)
        == Status::kInvalidControls);
    CHECK(swin_downsample(x_v, 0, -2, f.params, resample_v, 4, 4,
        next_dst, next_cap, body_dst, body_cap, scratch, scratch_cap)
        == Status::kInvalidControls);
    untouched();
    // Output extent classes.
    CHECK(swin_downsample(x_v, 0, 0, f.params, resample_v, 0, 4,
        next_dst, next_cap, body_dst, body_cap, scratch, scratch_cap)
        == Status::kInvalidDimensions);
    CHECK(swin_downsample(x_v, 0, 0, f.params, resample_v, 4,
        dlss5_nr::kMaxCols + 1, next_dst, next_cap,
        body_dst, body_cap, scratch, scratch_cap)
        == Status::kInvalidDimensions);
    untouched();
    // Capacity classes: exact boundaries.
    CHECK(swin_downsample(x_v, 0, 0, f.params, resample_v, 4, 4,
        next_dst, next_cap - 1, body_dst, body_cap, scratch, scratch_cap)
        == Status::kInvalidBuffer);
    CHECK(swin_downsample(x_v, 0, 0, f.params, resample_v, 4, 4,
        next_dst, next_cap, body_dst, body_cap - 1, scratch, scratch_cap)
        == Status::kInvalidBuffer);
    const std::size_t needed = fixture::kChannels * 64u
        + fixture::kChannels * 16u
        + dlss5_nr::swin_block_scratch_needed(x_v, false, false, 0, 0,
                                              f.params);
    CHECK(swin_downsample(x_v, 0, 0, f.params, resample_v, 4, 4,
        next_dst, next_cap, body_dst, body_cap, scratch, needed - 1)
        == Status::kInvalidBuffer);
    untouched();
    // The scratch capacity accepting edge (writes; sentinel state ends here).
    CHECK(swin_downsample(x_v, 0, 0, f.params, resample_v, 4, 4,
        next_dst, next_cap, body_dst, body_cap, scratch, needed)
        == Status::kOk);
    // The mid pairing's accept edge: mid [32, 32] with the fixture's
    // expand/contract satisfies the single-head grouped-mid pairings and
    // runs through this entry (the mid path arithmetic is pinned in the
    // block suite).
    static MatrixView mid_view = make_view(mid_table, fixture::kChannels,
        fixture::kChannels, fixture::kChannels);
    mid_p.mid = &mid_view;
    CHECK(swin_downsample(x_v, 0, 0, mid_p, resample_v, 4, 4,
        next_dst, next_cap, body_dst, body_cap, scratch, scratch_cap)
        == Status::kOk);
}

static void check_negative_upsample_blend(void) {
    using dlss5_nr::swin_upsample_blend;
    static float project[fixture::kChannels * 2 * fixture::kChannels];
    static float gain[fixture::kChannels];
    static float x[2 * fixture::kChannels * 4 * 4];
    static float skip[fixture::kChannels * 8 * 8];
    static float blend_dst[fixture::kChannels * 64];
    static float scratch[1 << 16];
    memset(project, 0, sizeof project);
    memset(x, 0, sizeof x);
    memset(skip, 0, sizeof skip);
    for (int c = 0; c < fixture::kChannels; ++c) {
        project[c * 2 * fixture::kChannels + c] = 1.0f;
        gain[c] = 1.0f;
    }
    MatrixView project_v = make_view(project, fixture::kChannels,
        2 * fixture::kChannels, 2 * fixture::kChannels);
    MatrixView gain_v = make_view(gain, 1, fixture::kChannels,
        fixture::kChannels);
    const FeatureBatchView x_v = make_batch(x, 2 * fixture::kChannels,
        4, 4, 16, 4);
    const FeatureBatchView skip_v = make_batch(skip, fixture::kChannels,
        8, 8, 64, 8);
    const std::size_t blend_cap = fixture::kChannels * 64u;
    const std::size_t scratch_cap = sizeof scratch / sizeof scratch[0];
    auto untouched = [&]() {
        CHECK(all_sentinel(blend_dst, blend_cap));
        CHECK(all_sentinel(scratch, scratch_cap));
    };
    fill_sentinel(blend_dst, blend_cap);
    fill_sentinel(scratch, scratch_cap);
    // Null classes.
    CHECK(swin_upsample_blend(make_view(nullptr, fixture::kChannels,
        2 * fixture::kChannels, 2 * fixture::kChannels), gain_v, x_v, skip_v,
        blend_dst, blend_cap, scratch, scratch_cap) == Status::kNullPointer);
    CHECK(swin_upsample_blend(project_v,
        make_view(nullptr, 1, fixture::kChannels, fixture::kChannels),
        x_v, skip_v, blend_dst, blend_cap, scratch, scratch_cap)
        == Status::kNullPointer);
    CHECK(swin_upsample_blend(project_v, gain_v,
        make_batch(nullptr, 2 * fixture::kChannels, 4, 4, 16, 4), skip_v,
        blend_dst, blend_cap, scratch, scratch_cap) == Status::kNullPointer);
    CHECK(swin_upsample_blend(project_v, gain_v, x_v,
        make_batch(nullptr, fixture::kChannels, 8, 8, 64, 8),
        blend_dst, blend_cap, scratch, scratch_cap) == Status::kNullPointer);
    CHECK(swin_upsample_blend(project_v, gain_v, x_v, skip_v,
        nullptr, blend_cap, scratch, scratch_cap) == Status::kNullPointer);
    CHECK(swin_upsample_blend(project_v, gain_v, x_v, skip_v,
        blend_dst, blend_cap, nullptr, scratch_cap) == Status::kNullPointer);
    untouched();
    // Dimension and pairing classes.
    CHECK(swin_upsample_blend(project_v, gain_v, x_v,
        make_batch(skip, fixture::kChannels, 0, 8, 64, 8),
        blend_dst, blend_cap, scratch, scratch_cap)
        == Status::kInvalidDimensions);
    CHECK(swin_upsample_blend(project_v, gain_v, x_v,
        make_batch(skip, fixture::kChannels + 1, 8, 8, 64, 8),
        blend_dst, blend_cap, scratch, scratch_cap)
        == Status::kInvalidDimensions);
    CHECK(swin_upsample_blend(project_v, gain_v,
        make_batch(x, 2 * fixture::kChannels - 1, 4, 4, 16, 4), skip_v,
        blend_dst, blend_cap, scratch, scratch_cap)
        == Status::kInvalidDimensions);
    CHECK(swin_upsample_blend(make_view(project, fixture::kChannels,
        2 * fixture::kChannels - 1, 2 * fixture::kChannels), gain_v, x_v,
        skip_v, blend_dst, blend_cap, scratch, scratch_cap)
        == Status::kInvalidDimensions);
    CHECK(swin_upsample_blend(project_v,
        make_view(gain, 1, fixture::kChannels - 1, fixture::kChannels),
        x_v, skip_v, blend_dst, blend_cap, scratch, scratch_cap)
        == Status::kInvalidDimensions);
    // Stride classes.
    CHECK(swin_upsample_blend(project_v, gain_v,
        make_batch(x, 2 * fixture::kChannels, 4, 4, 16, 3), skip_v,
        blend_dst, blend_cap, scratch, scratch_cap) == Status::kInvalidStride);
    CHECK(swin_upsample_blend(project_v, gain_v, x_v,
        make_batch(skip, fixture::kChannels, 8, 8, 64, 7),
        blend_dst, blend_cap, scratch, scratch_cap) == Status::kInvalidStride);
    untouched();
    // Value classes: off-grid project, off-grid x, off-grid/non-finite skip,
    // non-f16 gain.
    project[0] = 0.1f;
    CHECK(swin_upsample_blend(project_v, gain_v, x_v, skip_v,
        blend_dst, blend_cap, scratch, scratch_cap) == Status::kInvalidValues);
    project[0] = 1.0f;
    x[0] = 0.1f;
    CHECK(swin_upsample_blend(project_v, gain_v, x_v, skip_v,
        blend_dst, blend_cap, scratch, scratch_cap) == Status::kInvalidValues);
    x[0] = (float)NAN;
    CHECK(swin_upsample_blend(project_v, gain_v, x_v, skip_v,
        blend_dst, blend_cap, scratch, scratch_cap) == Status::kInvalidValues);
    x[0] = 0.0f;
    skip[0] = 0.1f;
    CHECK(swin_upsample_blend(project_v, gain_v, x_v, skip_v,
        blend_dst, blend_cap, scratch, scratch_cap) == Status::kInvalidValues);
    skip[0] = (float)INFINITY;
    CHECK(swin_upsample_blend(project_v, gain_v, x_v, skip_v,
        blend_dst, blend_cap, scratch, scratch_cap) == Status::kInvalidValues);
    skip[0] = 0.0f;
    gain[0] = 0.1f;
    CHECK(swin_upsample_blend(project_v, gain_v, x_v, skip_v,
        blend_dst, blend_cap, scratch, scratch_cap) == Status::kInvalidValues);
    gain[0] = (float)NAN;
    CHECK(swin_upsample_blend(project_v, gain_v, x_v, skip_v,
        blend_dst, blend_cap, scratch, scratch_cap) == Status::kInvalidValues);
    gain[0] = 1.0f;
    untouched();
    // Capacity classes: exact boundaries. The scratch holds the projected
    // map, C planes of the x extent (C * 4 * 4), not the 2C-plane x map.
    CHECK(swin_upsample_blend(project_v, gain_v, x_v, skip_v,
        blend_dst, blend_cap - 1, scratch, scratch_cap)
        == Status::kInvalidBuffer);
    CHECK(swin_upsample_blend(project_v, gain_v, x_v, skip_v,
        blend_dst, blend_cap, scratch, fixture::kChannels * 16u - 1)
        == Status::kInvalidBuffer);
    untouched();
    // The scratch capacity accepting edge (writes; sentinel state ends here).
    CHECK(swin_upsample_blend(project_v, gain_v, x_v, skip_v,
        blend_dst, blend_cap, scratch, fixture::kChannels * 16u)
        == Status::kOk);
}

static void check_negative_layers(void) {
    using dlss5_nr::swin_block_scratch_needed;
    using dlss5_nr::swin_layer_plain;
    using dlss5_nr::swin_layer_upsample;
    fixture::BlockFixture f;
    f.zero_init();
    static float x[fixture::kChannels * 8 * 8];
    static float x2[2 * fixture::kChannels * 4 * 4];
    static float skip[fixture::kChannels * 8 * 8];
    static float project[fixture::kChannels * 2 * fixture::kChannels];
    static float gain[fixture::kChannels];
    static float mid_table[fixture::kChannels * fixture::kChannels];
    static float out_dst[fixture::kChannels * 64];
    static float scratch[1 << 20];
    memset(x, 0, sizeof x);
    memset(x2, 0, sizeof x2);
    memset(skip, 0, sizeof skip);
    memset(project, 0, sizeof project);
    memset(mid_table, 0, sizeof mid_table);
    for (int c = 0; c < fixture::kChannels; ++c) {
        project[c * 2 * fixture::kChannels + c] = 1.0f;
        gain[c] = 1.0f;
    }
    const FeatureBatchView x_v = make_batch(x, fixture::kChannels, 8, 8,
        64, 8);
    const FeatureBatchView x2_v = make_batch(x2, 2 * fixture::kChannels,
        4, 4, 16, 4);
    const FeatureBatchView skip_v = make_batch(skip, fixture::kChannels,
        8, 8, 64, 8);
    MatrixView project_v = make_view(project, fixture::kChannels,
        2 * fixture::kChannels, 2 * fixture::kChannels);
    MatrixView gain_v = make_view(gain, 1, fixture::kChannels,
        fixture::kChannels);
    MatrixView mid_ok_view = make_view(mid_table, fixture::kChannels,
        fixture::kChannels, fixture::kChannels);
    MatrixView mid_bad_view = make_view(mid_table, fixture::kChannels - 1,
        fixture::kChannels, fixture::kChannels);
    MatrixView mid_null_view = make_view(nullptr, fixture::kChannels,
        fixture::kChannels, fixture::kChannels);
    const std::size_t out_cap = fixture::kChannels * 64u;
    const std::size_t scratch_cap = sizeof scratch / sizeof scratch[0];
    // The exact scratch boundaries the entries validate: swin_layer_plain
    // takes swin_block's mirrored `needed`; swin_layer_upsample takes
    // blend + projected + the block's mirrored `needed` on the blend map.
    const std::size_t plain_needed = swin_block_scratch_needed(x_v, false,
        true, 0, 0, f.params);
    const std::size_t blend_count = fixture::kChannels * 64u;
    const std::size_t projected_count = fixture::kChannels * 16u;
    const std::size_t up_needed = blend_count + projected_count
        + swin_block_scratch_needed(
            make_batch(skip, fixture::kChannels, 8, 8, 64, 8),
            true, true, 0, 0, f.params);
    auto untouched = [&]() {
        CHECK(all_sentinel(out_dst, out_cap));
        CHECK(all_sentinel(scratch, scratch_cap));
    };
    fill_sentinel(out_dst, out_cap);
    fill_sentinel(scratch, scratch_cap);
    // swin_layer_plain: null classes (entry operands and every weight).
    CHECK(swin_layer_plain(make_batch(nullptr, fixture::kChannels, 8, 8, 64, 8),
        0, 0, f.params, out_dst, out_cap, scratch, scratch_cap)
        == Status::kNullPointer);
    CHECK(swin_layer_plain(x_v, 0, 0, f.params, nullptr, out_cap, scratch,
        scratch_cap) == Status::kNullPointer);
    CHECK(swin_layer_plain(x_v, 0, 0, f.params, out_dst, out_cap, nullptr,
        scratch_cap) == Status::kNullPointer);
    SwinBlockParams p = f.params;
    p.expand.data = nullptr;
    CHECK(swin_layer_plain(x_v, 0, 0, p, out_dst, out_cap, scratch,
        scratch_cap) == Status::kNullPointer);
    p = f.params;
    p.contract.data = nullptr;
    CHECK(swin_layer_plain(x_v, 0, 0, p, out_dst, out_cap, scratch,
        scratch_cap) == Status::kNullPointer);
    p = f.params;
    p.mlp_scale.data = nullptr;
    CHECK(swin_layer_plain(x_v, 0, 0, p, out_dst, out_cap, scratch,
        scratch_cap) == Status::kNullPointer);
    p = f.params;
    p.qkv.data = nullptr;
    CHECK(swin_layer_plain(x_v, 0, 0, p, out_dst, out_cap, scratch,
        scratch_cap) == Status::kNullPointer);
    p = f.params;
    p.head_scale.data = nullptr;
    CHECK(swin_layer_plain(x_v, 0, 0, p, out_dst, out_cap, scratch,
        scratch_cap) == Status::kNullPointer);
    p = f.params;
    p.exp_bias.data = nullptr;
    CHECK(swin_layer_plain(x_v, 0, 0, p, out_dst, out_cap, scratch,
        scratch_cap) == Status::kNullPointer);
    p = f.params;
    p.attn_scale.data = nullptr;
    CHECK(swin_layer_plain(x_v, 0, 0, p, out_dst, out_cap, scratch,
        scratch_cap) == Status::kNullPointer);
    p = f.params;
    p.out_proj.data = nullptr;
    CHECK(swin_layer_plain(x_v, 0, 0, p, out_dst, out_cap, scratch,
        scratch_cap) == Status::kNullPointer);
    p = f.params;
    p.mid = &mid_null_view;
    CHECK(swin_layer_plain(x_v, 0, 0, p, out_dst, out_cap, scratch,
        scratch_cap) == Status::kNullPointer);
    untouched();
    // Dimension and pairing classes.
    CHECK(swin_layer_plain(make_batch(x, fixture::kChannels, 0, 8, 64, 8),
        0, 0, f.params, out_dst, out_cap, scratch, scratch_cap)
        == Status::kInvalidDimensions);
    CHECK(swin_layer_plain(make_batch(x, fixture::kChannels, 8, 0, 64, 8),
        0, 0, f.params, out_dst, out_cap, scratch, scratch_cap)
        == Status::kInvalidDimensions);
    CHECK(swin_layer_plain(make_batch(x, 0, 8, 8, 64, 8),
        0, 0, f.params, out_dst, out_cap, scratch, scratch_cap)
        == Status::kInvalidDimensions);
    p = f.params;
    p.heads = 2;
    CHECK(swin_layer_plain(x_v, 0, 0, p, out_dst, out_cap, scratch,
        scratch_cap) == Status::kInvalidDimensions);
    p = f.params;
    p.mid = &mid_bad_view;
    CHECK(swin_layer_plain(x_v, 0, 0, p, out_dst, out_cap, scratch,
        scratch_cap) == Status::kInvalidDimensions);
    untouched();
    // Stride classes (the block's own x stride bounds).
    CHECK(swin_layer_plain(make_batch(x, fixture::kChannels, 8, 8, 64, 7),
        0, 0, f.params, out_dst, out_cap, scratch, scratch_cap)
        == Status::kInvalidStride);
    CHECK(swin_layer_plain(make_batch(x, fixture::kChannels, 8, 8, 63, 8),
        0, 0, f.params, out_dst, out_cap, scratch, scratch_cap)
        == Status::kInvalidStride);
    untouched();
    // Shift classes.
    CHECK(swin_layer_plain(x_v, 1, 0, f.params, out_dst, out_cap, scratch,
        scratch_cap) == Status::kInvalidControls);
    CHECK(swin_layer_plain(x_v, 0, -2, f.params, out_dst, out_cap, scratch,
        scratch_cap) == Status::kInvalidControls);
    untouched();
    // swin_layer_plain value classes: the block's weight grids, scale rows
    // and bias table, plus the forwarded input-map finite scan. The block
    // validates every class before its first write, so each rejection leaves
    // the outputs and the full scratch untouched.
    f.expand[0] = 0.1f;
    CHECK(swin_layer_plain(x_v, 0, 0, f.params, out_dst, out_cap, scratch,
        scratch_cap) == Status::kInvalidValues);
    f.expand[0] = 0.0f;
    f.contract[0] = 0.1f;
    CHECK(swin_layer_plain(x_v, 0, 0, f.params, out_dst, out_cap, scratch,
        scratch_cap) == Status::kInvalidValues);
    f.contract[0] = 0.0f;
    f.qkv[0] = 0.1f;
    CHECK(swin_layer_plain(x_v, 0, 0, f.params, out_dst, out_cap, scratch,
        scratch_cap) == Status::kInvalidValues);
    f.qkv[0] = 0.0f;
    f.out_proj[0] = 0.1f;
    CHECK(swin_layer_plain(x_v, 0, 0, f.params, out_dst, out_cap, scratch,
        scratch_cap) == Status::kInvalidValues);
    f.out_proj[0] = 0.0f;
    f.mlp_scale[0] = 0.1f;
    CHECK(swin_layer_plain(x_v, 0, 0, f.params, out_dst, out_cap, scratch,
        scratch_cap) == Status::kInvalidValues);
    f.mlp_scale[0] = 0.0f;
    f.attn_scale[0] = 0.1f;
    CHECK(swin_layer_plain(x_v, 0, 0, f.params, out_dst, out_cap, scratch,
        scratch_cap) == Status::kInvalidValues);
    f.attn_scale[0] = 0.0f;
    f.head_scale[0] = 65520.0f;
    CHECK(swin_layer_plain(x_v, 0, 0, f.params, out_dst, out_cap, scratch,
        scratch_cap) == Status::kInvalidValues);
    f.head_scale[0] = 0.0f;
    f.exp_bias[0] = (float)NAN;
    CHECK(swin_layer_plain(x_v, 0, 0, f.params, out_dst, out_cap, scratch,
        scratch_cap) == Status::kInvalidValues);
    f.exp_bias[0] = 0.0f;
    p = f.params;
    p.mid = &mid_ok_view;
    mid_table[0] = 0.1f;
    CHECK(swin_layer_plain(x_v, 0, 0, p, out_dst, out_cap, scratch,
        scratch_cap) == Status::kInvalidValues);
    mid_table[0] = 0.0f;
    x[0] = (float)NAN;
    CHECK(swin_layer_plain(x_v, 0, 0, f.params, out_dst, out_cap, scratch,
        scratch_cap) == Status::kInvalidValues);
    x[0] = (float)INFINITY;
    CHECK(swin_layer_plain(x_v, 0, 0, f.params, out_dst, out_cap, scratch,
        scratch_cap) == Status::kInvalidValues);
    x[0] = 0.0f;
    untouched();
    // Capacity classes: the exact output and scratch boundaries.
    CHECK(swin_layer_plain(x_v, 0, 0, f.params, out_dst, out_cap - 1, scratch,
        scratch_cap) == Status::kInvalidBuffer);
    CHECK(swin_layer_plain(x_v, 0, 0, f.params, out_dst, out_cap, scratch,
        plain_needed - 1) == Status::kInvalidBuffer);
    untouched();
    // The accepting edge (writes; sentinel state ends here).
    CHECK(swin_layer_plain(x_v, 0, 0, f.params, out_dst, out_cap, scratch,
        plain_needed) == Status::kOk);
    fill_sentinel(out_dst, out_cap);
    fill_sentinel(scratch, scratch_cap);
    // swin_layer_upsample: null classes (entry operands and every weight).
    CHECK(swin_layer_upsample(make_batch(nullptr, 2 * fixture::kChannels,
        4, 4, 16, 4), skip_v, 0, 0, f.params, project_v, gain_v,
        out_dst, out_cap, scratch, scratch_cap) == Status::kNullPointer);
    CHECK(swin_layer_upsample(x2_v,
        make_batch(nullptr, fixture::kChannels, 8, 8, 64, 8),
        0, 0, f.params, project_v, gain_v, out_dst, out_cap, scratch,
        scratch_cap) == Status::kNullPointer);
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, f.params, project_v, gain_v,
        nullptr, out_cap, scratch, scratch_cap) == Status::kNullPointer);
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, f.params, project_v, gain_v,
        out_dst, out_cap, nullptr, scratch_cap) == Status::kNullPointer);
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, f.params,
        make_view(nullptr, fixture::kChannels, 2 * fixture::kChannels,
            2 * fixture::kChannels),
        gain_v, out_dst, out_cap, scratch, scratch_cap)
        == Status::kNullPointer);
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, f.params, project_v,
        make_view(nullptr, 1, fixture::kChannels, fixture::kChannels),
        out_dst, out_cap, scratch, scratch_cap) == Status::kNullPointer);
    p = f.params;
    p.expand.data = nullptr;
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, p, project_v, gain_v,
        out_dst, out_cap, scratch, scratch_cap) == Status::kNullPointer);
    p = f.params;
    p.contract.data = nullptr;
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, p, project_v, gain_v,
        out_dst, out_cap, scratch, scratch_cap) == Status::kNullPointer);
    p = f.params;
    p.mlp_scale.data = nullptr;
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, p, project_v, gain_v,
        out_dst, out_cap, scratch, scratch_cap) == Status::kNullPointer);
    p = f.params;
    p.qkv.data = nullptr;
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, p, project_v, gain_v,
        out_dst, out_cap, scratch, scratch_cap) == Status::kNullPointer);
    p = f.params;
    p.head_scale.data = nullptr;
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, p, project_v, gain_v,
        out_dst, out_cap, scratch, scratch_cap) == Status::kNullPointer);
    p = f.params;
    p.exp_bias.data = nullptr;
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, p, project_v, gain_v,
        out_dst, out_cap, scratch, scratch_cap) == Status::kNullPointer);
    p = f.params;
    p.attn_scale.data = nullptr;
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, p, project_v, gain_v,
        out_dst, out_cap, scratch, scratch_cap) == Status::kNullPointer);
    p = f.params;
    p.out_proj.data = nullptr;
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, p, project_v, gain_v,
        out_dst, out_cap, scratch, scratch_cap) == Status::kNullPointer);
    p = f.params;
    p.mid = &mid_null_view;
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, p, project_v, gain_v,
        out_dst, out_cap, scratch, scratch_cap) == Status::kNullPointer);
    untouched();
    // Dimension and pairing classes.
    CHECK(swin_layer_upsample(make_batch(x2, 2 * fixture::kChannels, 0, 4,
        16, 4), skip_v, 0, 0, f.params, project_v, gain_v,
        out_dst, out_cap, scratch, scratch_cap) == Status::kInvalidDimensions);
    CHECK(swin_layer_upsample(x2_v,
        make_batch(skip, fixture::kChannels, 0, 8, 64, 8),
        0, 0, f.params, project_v, gain_v, out_dst, out_cap, scratch,
        scratch_cap) == Status::kInvalidDimensions);
    CHECK(swin_layer_upsample(make_batch(x2, 2 * fixture::kChannels - 1, 4, 4,
        16, 4), skip_v, 0, 0, f.params, project_v, gain_v,
        out_dst, out_cap, scratch, scratch_cap) == Status::kInvalidDimensions);
    CHECK(swin_layer_upsample(x2_v,
        make_batch(skip, fixture::kChannels + 1, 8, 8, 64, 8),
        0, 0, f.params, project_v, gain_v, out_dst, out_cap, scratch,
        scratch_cap) == Status::kInvalidDimensions);
    p = f.params;
    p.heads = 2;
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, p, project_v, gain_v,
        out_dst, out_cap, scratch, scratch_cap) == Status::kInvalidDimensions);
    p = f.params;
    p.mid = &mid_bad_view;
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, p, project_v, gain_v,
        out_dst, out_cap, scratch, scratch_cap) == Status::kInvalidDimensions);
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, f.params,
        make_view(project, fixture::kChannels - 1, 2 * fixture::kChannels,
            2 * fixture::kChannels),
        gain_v, out_dst, out_cap, scratch, scratch_cap)
        == Status::kInvalidDimensions);
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, f.params,
        make_view(project, fixture::kChannels, 2 * fixture::kChannels - 1,
            2 * fixture::kChannels),
        gain_v, out_dst, out_cap, scratch, scratch_cap)
        == Status::kInvalidDimensions);
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, f.params, project_v,
        make_view(gain, 2, fixture::kChannels, fixture::kChannels),
        out_dst, out_cap, scratch, scratch_cap) == Status::kInvalidDimensions);
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, f.params, project_v,
        make_view(gain, 1, fixture::kChannels - 1, fixture::kChannels),
        out_dst, out_cap, scratch, scratch_cap) == Status::kInvalidDimensions);
    untouched();
    // Stride classes (the entry's own x and skip map stride bounds).
    CHECK(swin_layer_upsample(make_batch(x2, 2 * fixture::kChannels, 4, 4,
        16, 3), skip_v, 0, 0, f.params, project_v, gain_v,
        out_dst, out_cap, scratch, scratch_cap) == Status::kInvalidStride);
    CHECK(swin_layer_upsample(x2_v,
        make_batch(skip, fixture::kChannels, 8, 8, 64, 7),
        0, 0, f.params, project_v, gain_v, out_dst, out_cap, scratch,
        scratch_cap) == Status::kInvalidStride);
    untouched();
    // Shift classes.
    CHECK(swin_layer_upsample(x2_v, skip_v, 1, 0, f.params, project_v, gain_v,
        out_dst, out_cap, scratch, scratch_cap) == Status::kInvalidControls);
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, -2, f.params, project_v, gain_v,
        out_dst, out_cap, scratch, scratch_cap) == Status::kInvalidControls);
    untouched();
    // Layer weight-rejection sentinels: the entry pre-validates the block's
    // weight classes in the block's own order, so every rejection must leave
    // the outputs and the full scratch untouched (no blend-stage write first).
    f.expand[0] = 0.1f;
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, f.params, project_v, gain_v,
        out_dst, out_cap, scratch, scratch_cap) == Status::kInvalidValues);
    untouched();
    f.expand[0] = 0.0f;
    f.contract[0] = 0.1f;
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, f.params, project_v, gain_v,
        out_dst, out_cap, scratch, scratch_cap) == Status::kInvalidValues);
    untouched();
    f.contract[0] = 0.0f;
    f.qkv[0] = 0.1f;
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, f.params, project_v, gain_v,
        out_dst, out_cap, scratch, scratch_cap) == Status::kInvalidValues);
    untouched();
    f.qkv[0] = 0.0f;
    f.out_proj[0] = 0.1f;
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, f.params, project_v, gain_v,
        out_dst, out_cap, scratch, scratch_cap) == Status::kInvalidValues);
    untouched();
    f.out_proj[0] = 0.0f;
    f.mlp_scale[0] = 0.1f;
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, f.params, project_v, gain_v,
        out_dst, out_cap, scratch, scratch_cap) == Status::kInvalidValues);
    untouched();
    f.mlp_scale[0] = 0.0f;
    f.attn_scale[0] = 0.1f;
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, f.params, project_v, gain_v,
        out_dst, out_cap, scratch, scratch_cap) == Status::kInvalidValues);
    untouched();
    f.attn_scale[0] = 0.0f;
    f.head_scale[0] = 65520.0f;
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, f.params, project_v, gain_v,
        out_dst, out_cap, scratch, scratch_cap) == Status::kInvalidValues);
    untouched();
    f.head_scale[0] = 0.0f;
    f.exp_bias[0] = (float)NAN;
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, f.params, project_v, gain_v,
        out_dst, out_cap, scratch, scratch_cap) == Status::kInvalidValues);
    untouched();
    f.exp_bias[0] = 0.0f;
    p = f.params;
    p.mid = &mid_ok_view;
    mid_table[0] = 0.1f;
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, p, project_v, gain_v,
        out_dst, out_cap, scratch, scratch_cap) == Status::kInvalidValues);
    untouched();
    mid_table[0] = 0.0f;
    // Projected/gain classes (validated before the blend call).
    project[0] = 0.1f;
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, f.params, project_v, gain_v,
        out_dst, out_cap, scratch, scratch_cap) == Status::kInvalidValues);
    untouched();
    project[0] = 1.0f;
    gain[0] = 0.1f;
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, f.params, project_v, gain_v,
        out_dst, out_cap, scratch, scratch_cap) == Status::kInvalidValues);
    untouched();
    gain[0] = 1.0f;
    // Input-map classes (the blend's own pre-write scans: x on the e4m3
    // grid, skip finite and exactly binary16).
    x2[0] = 0.1f;
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, f.params, project_v, gain_v,
        out_dst, out_cap, scratch, scratch_cap) == Status::kInvalidValues);
    untouched();
    x2[0] = (float)NAN;
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, f.params, project_v, gain_v,
        out_dst, out_cap, scratch, scratch_cap) == Status::kInvalidValues);
    untouched();
    x2[0] = 0.0f;
    skip[0] = 0.1f;
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, f.params, project_v, gain_v,
        out_dst, out_cap, scratch, scratch_cap) == Status::kInvalidValues);
    untouched();
    skip[0] = (float)INFINITY;
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, f.params, project_v, gain_v,
        out_dst, out_cap, scratch, scratch_cap) == Status::kInvalidValues);
    untouched();
    skip[0] = 0.0f;
    // Capacity classes: the exact output and blend + projected + block
    // scratch boundaries.
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, f.params, project_v, gain_v,
        out_dst, out_cap - 1, scratch, scratch_cap) == Status::kInvalidBuffer);
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, f.params, project_v, gain_v,
        out_dst, out_cap, scratch, up_needed - 1) == Status::kInvalidBuffer);
    untouched();
    // The documented blend-overflow exception: an extreme operand set rounds
    // a blend element outside the finite binary16 range (the reference's
    // float16 conversion returns infinity there). The entry rejects with the
    // outputs untouched but after the blend stage has written the blend and
    // projected scratch regions - the one disclosed scratch exception.
    skip[0] = 65504.0f;
    gain[0] = 2.0f;
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, f.params, project_v, gain_v,
        out_dst, out_cap, scratch, scratch_cap) == Status::kInvalidValues);
    CHECK(all_sentinel(out_dst, out_cap));
    CHECK(!all_sentinel(scratch, blend_count));
    CHECK(!all_sentinel(scratch + blend_count, projected_count));
    skip[0] = 0.0f;
    gain[0] = 1.0f;
    fill_sentinel(out_dst, out_cap);
    fill_sentinel(scratch, scratch_cap);
    // The accepting edge (writes; sentinel state ends here): the exact
    // blend + projected + block scratch boundary.
    CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, f.params, project_v, gain_v,
        out_dst, out_cap, scratch, up_needed) == Status::kOk);
}

// ---- determinism and allocation accounting --------------------------------------------------------

static void check_allocations(void) {
    using dlss5_nr::swin_downsample;
    using dlss5_nr::swin_layer_plain;
    using dlss5_nr::swin_layer_upsample;
    using dlss5_nr::swin_upsample_blend;
    fixture::BlockFixture f;
    f.zero_init();
    static float x[fixture::kChannels * 8 * 8];
    static float resample[2 * fixture::kChannels * fixture::kChannels];
    static float project[fixture::kChannels * 2 * fixture::kChannels];
    static float gain[fixture::kChannels];
    static float skip[fixture::kChannels * 8 * 8];
    static float next_dst[2 * fixture::kChannels * 16];
    static float body_dst[fixture::kChannels * 64];
    static float blend_dst[fixture::kChannels * 64];
    static float out_dst[fixture::kChannels * 64];
    static float scratch[1 << 20];
    memset(x, 0, sizeof x);
    memset(skip, 0, sizeof skip);
    memset(resample, 0, sizeof resample);
    memset(project, 0, sizeof project);
    for (int c = 0; c < fixture::kChannels; ++c) {
        resample[c * fixture::kChannels + c] = 1.0f;
        project[c * 2 * fixture::kChannels + c] = 1.0f;
        gain[c] = 1.0f;
    }
    MatrixView resample_v = make_view(resample, 2 * fixture::kChannels,
        fixture::kChannels, fixture::kChannels);
    MatrixView project_v = make_view(project, fixture::kChannels,
        2 * fixture::kChannels, 2 * fixture::kChannels);
    MatrixView gain_v = make_view(gain, 1, fixture::kChannels,
        fixture::kChannels);
    const FeatureBatchView x_v = make_batch(x, fixture::kChannels, 8, 8,
        64, 8);
    const FeatureBatchView x2_v = make_batch(x, 2 * fixture::kChannels,
        4, 4, 16, 4);
    const FeatureBatchView skip_v = make_batch(skip, fixture::kChannels,
        8, 8, 64, 8);
    const unsigned long long before = g_allocations.load();
    for (int i = 0; i < 64; ++i) {
        CHECK(swin_downsample(x_v, 0, 0, f.params, resample_v, 4, 4,
            next_dst, 2 * fixture::kChannels * 16,
            body_dst, fixture::kChannels * 64,
            scratch, sizeof scratch / sizeof scratch[0]) == Status::kOk);
        CHECK(swin_upsample_blend(project_v, gain_v, x2_v, skip_v,
            blend_dst, fixture::kChannels * 64,
            scratch, sizeof scratch / sizeof scratch[0]) == Status::kOk);
        CHECK(swin_layer_plain(x_v, 0, 0, f.params,
            out_dst, fixture::kChannels * 64,
            scratch, sizeof scratch / sizeof scratch[0]) == Status::kOk);
        CHECK(swin_layer_upsample(x2_v, skip_v, 0, 0, f.params,
            project_v, gain_v, out_dst, fixture::kChannels * 64,
            scratch, sizeof scratch / sizeof scratch[0]) == Status::kOk);
    }
    CHECK(g_allocations.load() == before);
}

int main(void) {
    check_gated_sum();
    check_projections();
    check_downsample_closed_forms();
    check_upsample_blend_closed_forms();
    check_equivalence_downsample();
    check_equivalence_upsample_blend();
    check_layer_plain();
    check_layer_upsample();
    check_equivalence_layer_plain();
    check_equivalence_layer_upsample();
    check_negative_downsample();
    check_negative_upsample_blend();
    check_negative_layers();
    check_allocations();
    printf("native-swin-resample-test: %lld checks passed\n", g_checks);
    return 0;
}
