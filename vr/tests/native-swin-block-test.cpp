// Independent tests for the native Swin block composition
// (vr/native-swin-block.hpp): the e4m3 input rounding, the shifted window
// passes, the per-window MLP -> store -> attention chain with the one-head
// unrounded-skip rule, the map reassembly and the f16 wide output.
// CPU-only, no GPU, headset, proprietary payload, Python or PyTorch runtime
// required.
//
// Independence: every known answer is a hand-derived dyadic rational (the exact
// decompositions are written next to each case). Controls cover: the residual
// source rule (x is rounded to e4m3 before the products; x_wide passes through
// as f16 data), the one-head unrounded-skip rule (heads == 1 keeps the float32
// MLP output as the attention skip; several heads use its stored e4m3 form) and
// wide = f16(acc) tie-to-even narrowing; a bit-exact equivalence control of
// the composed entry against the staged chain over deterministic pseudorandom
// corpora (both residual paths, the single-head null-mid and multi-head
// grouped-mid paths, all four shift pairs, zero-weight and non-trivial weight
// corpora, multi-window grids with the front zero padding and padded input
// strides), including the non-accumulator mode (acc_dst null keeps the acc map
// in scratch and ignores acc_count); the head-scale magnitude bound and the
// mid-path width bound with their accepting and rejecting edges;
// sentinel-checked untouched outputs and scratch on every validation
// rejection class; determinism and zero-allocation accounting.
// See vr/NATIVE-SWIN-BLOCK.md.

#include "native-swin-block.hpp"

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

static MutableMatrixView make_out(float* data, int rows, int cols,
                                  std::ptrdiff_t row_stride) {
    MutableMatrixView v;
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

// ---- fixture ---------------------------------------------------------------------------------------
// One zero-weight block: with every weight zero and unit scales the composition
// collapses to documented closed forms:
//   hidden = 0, y = residual * mlp_scale (one float32 product),
//   q = k = v = 0 -> logits 0 -> every numerator 2^-14 -> probability 2^-6,
//   context = 0, out_proj 0 -> out = skip * attn_scale = skip.
// So the whole block observes the residual source, the skip rule and the f16
// wide narrowing directly.

namespace fixture {

constexpr int kChannels = 32;   // heads = 1
constexpr int kWideChannels = 64;  // heads = 2
constexpr int kHidden = 32;
constexpr int kMapRows = 8;
constexpr int kMapCols = 8;

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
};

}  // namespace fixture

// ---- known answers: residual source, skip rule and the wide narrowing --------------------------------

static void check_closed_forms(void) {
    using dlss5_nr::swin_block;
    using dlss5_nr::swin_core;
    fixture::BlockFixture f;
    f.zero_init();
    static float x[fixture::kChannels * fixture::kMapRows * fixture::kMapCols];
    static float x_wide[fixture::kChannels * fixture::kMapRows * fixture::kMapCols];
    static float wide[fixture::kChannels * fixture::kMapRows * fixture::kMapCols];
    static float acc[fixture::kChannels * fixture::kMapRows * fixture::kMapCols];
    static float scratch[1 << 20];
    const std::size_t map_count = fixture::kChannels * fixture::kMapRows
        * fixture::kMapCols;
    const FeatureBatchView x_v = make_batch(x, fixture::kChannels,
        fixture::kMapRows, fixture::kMapCols,
        fixture::kMapRows * fixture::kMapCols, fixture::kMapCols);
    const FeatureBatchView xw_v = make_batch(x_wide, fixture::kChannels,
        fixture::kMapRows, fixture::kMapCols,
        fixture::kMapRows * fixture::kMapCols, fixture::kMapCols);

    // Residual source rule: x = 1 + 2^-11 rounds to e4m3 1.0 before the
    // products (the quantised residual), while x_wide = 1 + 2^-10 passes
    // through as f16 data. With unit scales y equals the residual exactly.
    for (std::size_t p = 0; p < map_count; ++p) {
        x[p] = 1.0f + 0x1p-11f;
        x_wide[p] = 1.0f + 0x1p-10f;
    }
    CHECK(swin_block(x_v, nullptr, 0, 0, f.params, wide, map_count,
        acc, map_count, scratch, sizeof scratch / sizeof scratch[0])
        == Status::kOk);
    for (std::size_t p = 0; p < map_count; ++p) {
        CHECK(acc[p] == 1.0f);              // e4m3(1 + 2^-11) = 1.0
        CHECK(wide[p] == 1.0f);
    }
    CHECK(swin_block(x_v, &xw_v, 0, 0, f.params, wide, map_count,
        acc, map_count, scratch, sizeof scratch / sizeof scratch[0])
        == Status::kOk);
    for (std::size_t p = 0; p < map_count; ++p) {
        CHECK(acc[p] == 1.0f + 0x1p-10f);   // the f16 x_wide, unrounded
        CHECK(wide[p] == 1.0f + 0x1p-10f);
    }
    // swin_core: x doubles as the MLP residual (f16-exact x required).
    for (std::size_t p = 0; p < map_count; ++p) {
        x[p] = 1.0f + 0x1p-10f;
    }
    CHECK(swin_core(x_v, 0, 0, f.params, wide, map_count, acc, map_count,
        scratch, sizeof scratch / sizeof scratch[0]) == Status::kOk);
    for (std::size_t p = 0; p < map_count; ++p) {
        CHECK(acc[p] == 1.0f + 0x1p-10f);
        CHECK(wide[p] == 1.0f + 0x1p-10f);
    }

    // Skip rule + wide narrowing discriminator: residual 1.5 with the f16 scale
    // 1 + 2^-10 gives the exact float32 product
    //   1.5 * (1 + 2^-10) = 1.5 + 2^-10 + 2^-11 = 1.50146484375,
    // which ties on the binary16 grid between mantissas 513 and 514 and rounds
    // to even -> 1.501953125; store() takes it on to the e4m3 grid 1.5.
    //   heads == 1: skip = y -> acc = 1.50146484375, wide = 1.501953125
    //   heads == 2: skip = store(y) -> acc = 1.5, wide = 1.5
    for (std::size_t p = 0; p < map_count; ++p) {
        x[p] = 1.5f;
        x_wide[p] = 1.5f;
    }
    for (int c = 0; c < fixture::kChannels; ++c) {
        f.mlp_scale[c] = 1.0f + 0x1p-10f;
    }
    f.params.mlp_scale = make_view(f.mlp_scale, 1, fixture::kChannels,
        fixture::kChannels);
    CHECK(swin_block(x_v, &xw_v, 0, 0, f.params, wide, map_count,
        acc, map_count, scratch, sizeof scratch / sizeof scratch[0])
        == Status::kOk);
    for (std::size_t p = 0; p < map_count; ++p) {
        CHECK(acc[p] == 1.50146484375f);
        CHECK(wide[p] == 1.501953125f);
    }
    // The multi-head rule (a dedicated heads = 2 fixture on C = 64).
    static float wide_x[fixture::kWideChannels * fixture::kMapRows
        * fixture::kMapCols];
    static float wide_xw[fixture::kWideChannels * fixture::kMapRows
        * fixture::kMapCols];
    static float wide_out[fixture::kWideChannels * fixture::kMapRows
        * fixture::kMapCols];
    static float wide_acc[fixture::kWideChannels * fixture::kMapRows
        * fixture::kMapCols];
    static float wide_expand[fixture::kHidden * fixture::kWideChannels];
    static float wide_contract[fixture::kWideChannels * fixture::kHidden];
    static float wide_mlp_scale[fixture::kWideChannels];
    static float wide_qkv[3 * fixture::kWideChannels * fixture::kWideChannels];
    static float wide_head_scale[2];
    static float wide_exp_bias[2 * dlss5_nr::kBiasTable];
    static float wide_attn_scale[fixture::kWideChannels];
    static float wide_out_proj[fixture::kWideChannels * fixture::kWideChannels];
    const std::size_t wide_map = fixture::kWideChannels * fixture::kMapRows
        * fixture::kMapCols;
    memset(wide_expand, 0, sizeof wide_expand);
    memset(wide_contract, 0, sizeof wide_contract);
    memset(wide_qkv, 0, sizeof wide_qkv);
    memset(wide_exp_bias, 0, sizeof wide_exp_bias);
    memset(wide_out_proj, 0, sizeof wide_out_proj);
    for (int c = 0; c < fixture::kWideChannels; ++c) {
        wide_mlp_scale[c] = c == 0 ? 1.0f + 0x1p-10f : 1.0f;
        wide_attn_scale[c] = 1.0f;
    }
    wide_head_scale[0] = 1.0f;
    wide_head_scale[1] = 1.0f;
    SwinBlockParams wp;
    wp.expand = make_view(wide_expand, fixture::kHidden, fixture::kWideChannels,
        fixture::kWideChannels);
    wp.mid = nullptr;
    wp.heads = 2;
    wp.contract = make_view(wide_contract, fixture::kWideChannels,
        fixture::kHidden, fixture::kHidden);
    wp.mlp_scale = make_view(wide_mlp_scale, 1, fixture::kWideChannels,
        fixture::kWideChannels);
    wp.qkv = make_view(wide_qkv, 3 * fixture::kWideChannels,
        fixture::kWideChannels, fixture::kWideChannels);
    wp.head_scale = make_view(wide_head_scale, 1, 2, 2);
    wp.exp_bias = make_view(wide_exp_bias, 2, dlss5_nr::kBiasTable,
        dlss5_nr::kBiasTable);
    wp.attn_scale = make_view(wide_attn_scale, 1, fixture::kWideChannels,
        fixture::kWideChannels);
    wp.out_proj = make_view(wide_out_proj, fixture::kWideChannels,
        fixture::kWideChannels, fixture::kWideChannels);
    for (std::size_t p = 0; p < wide_map; ++p) {
        wide_x[p] = 1.5f;
        wide_xw[p] = 1.5f;
    }
    const FeatureBatchView wx_v = make_batch(wide_x, fixture::kWideChannels,
        fixture::kMapRows, fixture::kMapCols,
        fixture::kMapRows * fixture::kMapCols, fixture::kMapCols);
    const FeatureBatchView wxw_v = make_batch(wide_xw, fixture::kWideChannels,
        fixture::kMapRows, fixture::kMapCols,
        fixture::kMapRows * fixture::kMapCols, fixture::kMapCols);
    CHECK(swin_block(wx_v, &wxw_v, 0, 0, wp, wide_out, wide_map,
        wide_acc, wide_map, scratch, sizeof scratch / sizeof scratch[0])
        == Status::kOk);
    for (int r = 0; r < fixture::kMapRows; ++r) {
        for (int c = 0; c < fixture::kMapCols; ++c) {
            // Channel 0 carried the 1 + 2^-10 scale: y = 1.50146484375,
            // stored to 1.5 (the multi-head skip), so both maps hold 1.5.
            const std::size_t p0 = 0 * fixture::kMapRows * fixture::kMapCols
                + static_cast<std::size_t>(r) * fixture::kMapCols + c;
            CHECK(wide_acc[p0] == 1.5f);
            CHECK(wide_out[p0] == 1.5f);
            // Channel 1 kept the unit scale: y = 1.5 everywhere.
            const std::size_t p1 = 1 * fixture::kMapRows * fixture::kMapCols
                + static_cast<std::size_t>(r) * fixture::kMapCols + c;
            CHECK(wide_acc[p1] == 1.5f);
            CHECK(wide_out[p1] == 1.5f);
        }
    }
    // The same heads == 2 closed form through the grouped mid path (a non-null
    // mid [64, 16] with expand.rows = heads * mid.cols = 32, contract [64, 64]):
    // every weight is zero, so the grouped projection collapses exactly as the
    // single-stage path and both maps hold 1.5.
    static float wide_mid[fixture::kWideChannels * 16];
    static float wide_expand_mid[16 * 2 * fixture::kWideChannels];
    static float wide_contract_mid[fixture::kWideChannels * fixture::kWideChannels];
    static float wide_mid_out[fixture::kWideChannels * fixture::kMapRows
        * fixture::kMapCols];
    static float wide_mid_acc[fixture::kWideChannels * fixture::kMapRows
        * fixture::kMapCols];
    memset(wide_mid, 0, sizeof wide_mid);
    memset(wide_expand_mid, 0, sizeof wide_expand_mid);
    memset(wide_contract_mid, 0, sizeof wide_contract_mid);
    SwinBlockParams mp = wp;
    const MatrixView wide_mid_v = make_view(wide_mid, fixture::kWideChannels,
        16, 16);
    mp.mid = &wide_mid_v;
    mp.expand = make_view(wide_expand_mid, 32, fixture::kWideChannels,
        fixture::kWideChannels);
    mp.contract = make_view(wide_contract_mid, fixture::kWideChannels,
        fixture::kWideChannels, fixture::kWideChannels);
    CHECK(swin_block(wx_v, &wxw_v, 0, 0, mp, wide_mid_out, wide_map,
        wide_mid_acc, wide_map, scratch, sizeof scratch / sizeof scratch[0])
        == Status::kOk);
    for (std::size_t p = 0; p < wide_map; ++p) {
        CHECK(wide_mid_acc[p] == 1.5f);
        CHECK(wide_mid_out[p] == 1.5f);
    }
}

// ---- the staged chain re-derivation ------------------------------------------------------------------
// The reference composition written out directly against the independently
// tested stage entries. The equivalence control pins the block's plumbing
// (buffer management, the residual source and the skip rule wiring) against it.

static void staged_block(const FeatureBatchView& x, const FeatureBatchView* x_wide,
                         int shift_rows, int shift_cols,
                         const SwinBlockParams& p,
                         float* wide_dst, float* acc_dst, float* scratch) {
    using dlss5_nr::e4m3;
    using dlss5_nr::f16;
    using dlss5_nr::kBiasTable;
    using dlss5_nr::kWindowTokens;
    using dlss5_nr::swin_attention;
    using dlss5_nr::swin_from_windows;
    using dlss5_nr::swin_mlp;
    using dlss5_nr::swin_store;
    using dlss5_nr::swin_to_windows;
    using dlss5_nr::WindowGrid;
    using dlss5_nr::WindowTokenView;
    const int channels = x.channels;
    const std::size_t map_count = static_cast<std::size_t>(channels)
        * x.rows * x.cols;
    float* map_q = scratch;
    float* tokens_q = map_q + map_count;
    int grid_rows = 0;
    int front_h = 0;
    int grid_cols = 0;
    int front_w = 0;
    CHECK(dlss5_nr::swin_window_grid(x.rows, shift_rows, &grid_rows, &front_h)
        == Status::kOk);
    CHECK(dlss5_nr::swin_window_grid(x.cols, shift_cols, &grid_cols, &front_w)
        == Status::kOk);
    const std::size_t token_count = static_cast<std::size_t>(grid_rows)
        * grid_cols * kWindowTokens * channels;
    // Round x to e4m3, then window it.
    for (int c = 0; c < channels; ++c) {
        for (int r = 0; r < x.rows; ++r) {
            for (int col = 0; col < x.cols; ++col) {
                map_q[(static_cast<std::size_t>(c) * x.rows + r) * x.cols + col]
                    = e4m3(x.data[static_cast<std::size_t>(c) * x.plane_stride
                        + static_cast<std::size_t>(r) * x.row_stride + col]);
            }
        }
    }
    FeatureBatchView q_map = make_batch(map_q, channels, x.rows, x.cols,
        static_cast<std::ptrdiff_t>(x.rows) * x.cols, x.cols);
    WindowTokenView q_tokens;
    q_tokens.data = tokens_q;
    q_tokens.channels = channels;
    q_tokens.window_stride = static_cast<std::ptrdiff_t>(kWindowTokens) * channels;
    q_tokens.token_stride = channels;
    WindowGrid grid;
    CHECK(swin_to_windows(q_map, shift_rows, shift_cols, &grid, &q_tokens,
        token_count) == Status::kOk);
    float* tokens_r = tokens_q + token_count;
    float* tokens_out = tokens_r + token_count;
    WindowTokenView r_tokens = q_tokens;
    if (x_wide != nullptr) {
        r_tokens.data = tokens_r;
        WindowGrid grid2;
        CHECK(swin_to_windows(*x_wide, shift_rows, shift_cols, &grid2,
            &r_tokens, token_count) == Status::kOk);
    }
    float* y_buffer = tokens_out + token_count;
    float* y_q_buffer = y_buffer + kWindowTokens * channels;
    float* mlp_scratch = y_q_buffer + kWindowTokens * channels;
    float* attn_scratch = mlp_scratch + kWindowTokens * p.expand.rows;
    const std::size_t attn_needed =
        static_cast<std::size_t>(kWindowTokens) * 3 * channels
        + 4 * static_cast<std::size_t>(kWindowTokens) * channels
        + 2 * static_cast<std::size_t>(p.exp_bias.rows) * kBiasTable;
    const std::size_t windows = static_cast<std::size_t>(grid_rows) * grid_cols;
    for (std::size_t w = 0; w < windows; ++w) {
        MatrixView x_window = make_view(tokens_q + w * kWindowTokens * channels,
            kWindowTokens, channels, channels);
        MatrixView residual_window = make_view(
            r_tokens.data + w * r_tokens.window_stride,
            kWindowTokens, channels, r_tokens.token_stride);
        MutableMatrixView y_view = make_out(y_buffer, kWindowTokens, channels,
            channels);
        MutableMatrixView mlp_scratch_view = make_out(mlp_scratch,
            kWindowTokens, p.expand.rows, p.expand.rows);
        CHECK(swin_mlp(x_window, p.expand, p.mid, p.heads, p.contract,
            residual_window, p.mlp_scale, &y_view,
            static_cast<std::size_t>(kWindowTokens) * channels,
            &mlp_scratch_view,
            static_cast<std::size_t>(kWindowTokens) * p.expand.rows)
            == Status::kOk);
        for (int i = 0; i < kWindowTokens * channels; ++i) {
            y_q_buffer[i] = swin_store(y_buffer[i]);
        }
        MatrixView y_q_view = make_view(y_q_buffer, kWindowTokens, channels,
            channels);
        MatrixView skip_view = make_view(p.heads == 1 ? y_buffer : y_q_buffer,
            kWindowTokens, channels, channels);
        MutableMatrixView out_window = make_out(
            tokens_out + w * kWindowTokens * channels, kWindowTokens, channels,
            channels);
        CHECK(swin_attention(y_q_view, p.qkv, p.head_scale, p.exp_bias,
            skip_view, p.attn_scale, p.out_proj, &out_window,
            static_cast<std::size_t>(kWindowTokens) * channels, attn_scratch,
            attn_needed) == Status::kOk);
    }
    WindowTokenView out_tokens;
    out_tokens.data = tokens_out;
    out_tokens.windows = static_cast<int>(windows);
    out_tokens.channels = channels;
    out_tokens.window_stride = static_cast<std::ptrdiff_t>(kWindowTokens) * channels;
    out_tokens.token_stride = channels;
    CHECK(swin_from_windows(out_tokens, grid, x.rows, x.cols, acc_dst,
        map_count) == Status::kOk);
    for (std::size_t i = 0; i < map_count; ++i) {
        wide_dst[i] = f16(acc_dst[i]);
    }
}

// Deterministic pseudorandom f16 corpus (xorshift32; every value is exactly
// binary16-representable and finite: any sign and exponent field 0..30).
static std::uint32_t g_rng = 0x243F6A88u;
static float rnd_f16(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    std::uint16_t bits = static_cast<std::uint16_t>(g_rng & 0xFFFFu);
    if ((bits & 0x7C00u) == 0x7C00u) {
        bits = static_cast<std::uint16_t>(bits & 0x83FFu);  // exponent field 0
    }
    return dlss5_nr::f16_from_bits(bits);
}

// Non-trivial weight corpus for the arithmetic-rich equivalence configurations:
// e4m3-grid matrices, f16 scale rows, a bounded head-scale row and a finite
// exp-bias table (the two zero-weight fixtures stay as closed-form anchors).
static void fill_random_weights(fixture::BlockFixture* f) {
    for (std::size_t p = 0; p < sizeof f->expand / sizeof f->expand[0]; ++p) {
        f->expand[p] = dlss5_nr::e4m3(rnd_f16() - 1.5f);
    }
    for (std::size_t p = 0; p < sizeof f->contract / sizeof f->contract[0];
            ++p) {
        f->contract[p] = dlss5_nr::e4m3(rnd_f16() - 1.5f);
    }
    for (std::size_t p = 0; p < sizeof f->qkv / sizeof f->qkv[0]; ++p) {
        f->qkv[p] = dlss5_nr::e4m3(rnd_f16() - 1.5f);
    }
    for (std::size_t p = 0; p < sizeof f->out_proj / sizeof f->out_proj[0];
            ++p) {
        f->out_proj[p] = dlss5_nr::e4m3(rnd_f16() - 1.5f);
    }
    for (int c = 0; c < fixture::kChannels; ++c) {
        f->mlp_scale[c] = rnd_f16();
        f->attn_scale[c] = rnd_f16();
    }
    f->head_scale[0] = 1.25f;
    for (std::size_t p = 0; p < sizeof f->exp_bias / sizeof f->exp_bias[0];
            ++p) {
        f->exp_bias[p] = rnd_f16();
    }
}

static void check_equivalence(void) {
    using dlss5_nr::swin_block;
    static float scratch_impl[1 << 21];
    static float scratch_stage[1 << 21];
    static float wide_impl[1 << 16];
    static float wide_stage[1 << 16];
    static float wide_noacc[1 << 16];
    static float acc_impl[1 << 16];
    static float acc_stage[1 << 16];
    static float x_buf[1 << 16];
    static float xw_buf[1 << 16];
    // Shape/shift/path matrix: both residual paths, all four shift pairs,
    // multi-window grids with the front zero padding and poisoned padded input
    // strides. Configurations 2-4 carry non-trivial random weights (0-1 stay
    // zero-weight); the single-head null-mid path is the loop's, the multi-head
    // grouped mid path is the dedicated configuration matrix below.
    const int rows[] = {8, 8, 12, 16, 20};
    const int cols[] = {8, 16, 12, 16, 12};
    const int shift_r[] = {0, -1, 0, -1, -1};
    const int shift_c[] = {0, 0, -1, -1, -1};
    for (int cfg = 0; cfg < 5; ++cfg) {
        for (int path = 0; path < 2; ++path) {
            fixture::BlockFixture f;
            f.zero_init();
            if (cfg >= 2) {
                fill_random_weights(&f);
            }
            const int channels = fixture::kChannels;
            const std::size_t map_count = static_cast<std::size_t>(channels)
                * rows[cfg] * cols[cfg];
            for (std::size_t p = 0; p < map_count; ++p) {
                xw_buf[p] = rnd_f16();
            }
            // Padded input strides on x (never read past cols): poison the
            // padded region, then fill every map cell through the strides.
            const std::ptrdiff_t row_stride = cols[cfg] + 3;
            const std::ptrdiff_t plane_stride =
                static_cast<std::ptrdiff_t>(rows[cfg]) * row_stride;
            for (std::ptrdiff_t p = 0; p < plane_stride * channels; ++p) {
                // Padding cells are poisoned but stay finite.
                x_buf[p] = 0.25f;
            }
            for (int c = 0; c < channels; ++c) {
                for (int r = 0; r < rows[cfg]; ++r) {
                    for (int col = 0; col < cols[cfg]; ++col) {
                        x_buf[c * plane_stride + r * row_stride + col]
                            = rnd_f16();
                    }
                }
            }
            const FeatureBatchView x_v = make_batch(x_buf, channels,
                rows[cfg], cols[cfg], plane_stride, row_stride);
            const FeatureBatchView xw_v = make_batch(xw_buf, channels,
                rows[cfg], cols[cfg],
                static_cast<std::ptrdiff_t>(rows[cfg]) * cols[cfg], cols[cfg]);
            const FeatureBatchView* wide_arg = path == 0 ? nullptr : &xw_v;
            fill_sentinel(wide_impl, 1 << 16);
            fill_sentinel(wide_stage, 1 << 16);
            fill_sentinel(acc_impl, 1 << 16);
            fill_sentinel(acc_stage, 1 << 16);
            CHECK(swin_block(x_v, wide_arg, shift_r[cfg], shift_c[cfg],
                f.params, wide_impl, map_count, acc_impl, map_count,
                scratch_impl, sizeof scratch_impl / sizeof scratch_impl[0])
                == Status::kOk);
            staged_block(x_v, wide_arg, shift_r[cfg], shift_c[cfg], f.params,
                wide_stage, acc_stage, scratch_stage);
            for (std::size_t p = 0; p < map_count; ++p) {
                CHECK(wide_impl[p] == wide_stage[p]);
                CHECK(acc_impl[p] == acc_stage[p]);
            }
            CHECK(all_sentinel(wide_impl + map_count, (1 << 16) - map_count));
            CHECK(all_sentinel(acc_impl + map_count, (1 << 16) - map_count));
            // The non-accumulator mode: acc_dst null keeps the acc map in
            // scratch (acc_count is ignored entirely) and must produce a
            // bit-identical wide map with the same write bounds.
            fill_sentinel(wide_noacc, 1 << 16);
            CHECK(swin_block(x_v, wide_arg, shift_r[cfg], shift_c[cfg],
                f.params, wide_noacc, map_count, nullptr, 0,
                scratch_impl, sizeof scratch_impl / sizeof scratch_impl[0])
                == Status::kOk);
            for (std::size_t p = 0; p < map_count; ++p) {
                CHECK(wide_noacc[p] == wide_impl[p]);
            }
            CHECK(all_sentinel(wide_noacc + map_count, (1 << 16) - map_count));
        }
    }
    // Multi-head grouped-mid configuration matrix (heads = 2, non-null mid
    // [64, 8]): all four shift pairs x both residual paths, deterministic
    // non-trivial weights, poisoned padded input strides on x, output
    // sentinels, and the in-scratch acc mode compared bit-exactly.
    static float mid_buf[fixture::kWideChannels * 8];
    static float expand_buf[16 * fixture::kWideChannels];
    static float contract_buf[fixture::kWideChannels * fixture::kWideChannels];
    static float mlp_scale_buf[fixture::kWideChannels];
    static float qkv_buf[3 * fixture::kWideChannels * fixture::kWideChannels];
    static float head_scale_buf[2];
    static float exp_bias_buf[2 * dlss5_nr::kBiasTable];
    static float attn_scale_buf[fixture::kWideChannels];
    static float out_proj_buf[fixture::kWideChannels * fixture::kWideChannels];
    static float x2_buf[1 << 16];
    static float x2w_buf[1 << 16];
    const int channels = fixture::kWideChannels;
    const int rows2 = 12;
    const int cols2 = 20;
    const std::size_t map_count = static_cast<std::size_t>(channels) * rows2 * cols2;
    for (std::size_t p = 0; p < sizeof mid_buf / sizeof mid_buf[0]; ++p) {
        mid_buf[p] = dlss5_nr::e4m3(rnd_f16() - 1.5f);
    }
    for (std::size_t p = 0; p < sizeof expand_buf / sizeof expand_buf[0]; ++p) {
        expand_buf[p] = dlss5_nr::e4m3(rnd_f16() - 1.5f);
    }
    for (std::size_t p = 0; p < sizeof contract_buf / sizeof contract_buf[0];
            ++p) {
        contract_buf[p] = dlss5_nr::e4m3(rnd_f16() - 1.5f);
    }
    for (int c = 0; c < channels; ++c) {
        mlp_scale_buf[c] = rnd_f16();
        attn_scale_buf[c] = rnd_f16();
    }
    for (std::size_t p = 0; p < sizeof qkv_buf / sizeof qkv_buf[0]; ++p) {
        qkv_buf[p] = dlss5_nr::e4m3(rnd_f16() - 1.5f);
    }
    for (std::size_t p = 0; p < sizeof out_proj_buf / sizeof out_proj_buf[0];
            ++p) {
        out_proj_buf[p] = dlss5_nr::e4m3(rnd_f16() - 1.5f);
    }
    head_scale_buf[0] = 1.25f;
    head_scale_buf[1] = 0.75f;
    for (std::size_t p = 0; p < sizeof exp_bias_buf / sizeof exp_bias_buf[0];
            ++p) {
        exp_bias_buf[p] = rnd_f16();
    }
    MatrixView mid_v = make_view(mid_buf, channels, 8, 8);
    SwinBlockParams p2;
    p2.expand = make_view(expand_buf, 16, channels, channels);
    p2.mid = &mid_v;
    p2.heads = 2;
    p2.contract = make_view(contract_buf, channels, channels, channels);
    p2.mlp_scale = make_view(mlp_scale_buf, 1, channels, channels);
    p2.qkv = make_view(qkv_buf, 3 * channels, channels, channels);
    p2.head_scale = make_view(head_scale_buf, 1, 2, 2);
    p2.exp_bias = make_view(exp_bias_buf, 2, dlss5_nr::kBiasTable,
        dlss5_nr::kBiasTable);
    p2.attn_scale = make_view(attn_scale_buf, 1, channels, channels);
    p2.out_proj = make_view(out_proj_buf, channels, channels, channels);
    const int shift_pairs[4][2] = {{0, 0}, {-1, 0}, {0, -1}, {-1, -1}};
    for (int sp = 0; sp < 4; ++sp) {
        for (int path = 0; path < 2; ++path) {
            // Poisoned padded strides on x (the padding cells stay finite);
            // x_wide is tight.
            const std::ptrdiff_t row_stride2 = cols2 + 3;
            const std::ptrdiff_t plane_stride2 =
                static_cast<std::ptrdiff_t>(rows2) * row_stride2;
            for (std::ptrdiff_t p = 0; p < plane_stride2 * channels; ++p) {
                x2_buf[p] = 0.25f;
            }
            for (int c = 0; c < channels; ++c) {
                for (int r = 0; r < rows2; ++r) {
                    for (int col = 0; col < cols2; ++col) {
                        x2_buf[c * plane_stride2 + r * row_stride2 + col]
                            = rnd_f16();
                    }
                }
            }
            for (std::size_t p = 0; p < map_count; ++p) {
                x2w_buf[p] = rnd_f16();
            }
            const FeatureBatchView x_v2 = make_batch(x2_buf, channels, rows2,
                cols2, plane_stride2, row_stride2);
            const FeatureBatchView xw_v2 = make_batch(x2w_buf, channels, rows2,
                cols2, static_cast<std::ptrdiff_t>(rows2) * cols2, cols2);
            const FeatureBatchView* wide_arg2 = path == 0 ? nullptr : &xw_v2;
            fill_sentinel(wide_impl, 1 << 16);
            fill_sentinel(wide_stage, 1 << 16);
            fill_sentinel(acc_impl, 1 << 16);
            fill_sentinel(acc_stage, 1 << 16);
            CHECK(swin_block(x_v2, wide_arg2, shift_pairs[sp][0],
                shift_pairs[sp][1], p2, wide_impl, map_count, acc_impl,
                map_count, scratch_impl,
                sizeof scratch_impl / sizeof scratch_impl[0]) == Status::kOk);
            staged_block(x_v2, wide_arg2, shift_pairs[sp][0],
                shift_pairs[sp][1], p2, wide_stage, acc_stage, scratch_stage);
            for (std::size_t p = 0; p < map_count; ++p) {
                CHECK(wide_impl[p] == wide_stage[p]);
                CHECK(acc_impl[p] == acc_stage[p]);
            }
            CHECK(all_sentinel(wide_impl + map_count, (1 << 16) - map_count));
            CHECK(all_sentinel(acc_impl + map_count, (1 << 16) - map_count));
            fill_sentinel(wide_noacc, 1 << 16);
            CHECK(swin_block(x_v2, wide_arg2, shift_pairs[sp][0],
                shift_pairs[sp][1], p2, wide_noacc, map_count, nullptr, 0,
                scratch_impl,
                sizeof scratch_impl / sizeof scratch_impl[0]) == Status::kOk);
            for (std::size_t p = 0; p < map_count; ++p) {
                CHECK(wide_noacc[p] == wide_impl[p]);
            }
            CHECK(all_sentinel(wide_noacc + map_count, (1 << 16) - map_count));
        }
    }
    // Boundary head-scale run: the kMaxHeadScale edge values flow through the
    // normalisation products (the scale product stays finite) and match the
    // staged chain bit-exactly.
    head_scale_buf[0] = dlss5_nr::kMaxHeadScale;
    head_scale_buf[1] = -dlss5_nr::kMaxHeadScale;
    {
        const std::ptrdiff_t row_stride2 = cols2 + 3;
        const std::ptrdiff_t plane_stride2 =
            static_cast<std::ptrdiff_t>(rows2) * row_stride2;
        for (std::ptrdiff_t p = 0; p < plane_stride2 * channels; ++p) {
            x2_buf[p] = 0.25f;
        }
        for (int c = 0; c < channels; ++c) {
            for (int r = 0; r < rows2; ++r) {
                for (int col = 0; col < cols2; ++col) {
                    x2_buf[c * plane_stride2 + r * row_stride2 + col]
                        = rnd_f16();
                }
            }
        }
        for (std::size_t p = 0; p < map_count; ++p) {
            x2w_buf[p] = rnd_f16();
        }
        const FeatureBatchView x_v2 = make_batch(x2_buf, channels, rows2,
            cols2, plane_stride2, row_stride2);
        const FeatureBatchView xw_v2 = make_batch(x2w_buf, channels, rows2,
            cols2, static_cast<std::ptrdiff_t>(rows2) * cols2, cols2);
        fill_sentinel(wide_impl, 1 << 16);
        fill_sentinel(wide_stage, 1 << 16);
        fill_sentinel(acc_impl, 1 << 16);
        fill_sentinel(acc_stage, 1 << 16);
        CHECK(swin_block(x_v2, &xw_v2, -1, -1, p2, wide_impl, map_count,
            acc_impl, map_count, scratch_impl,
            sizeof scratch_impl / sizeof scratch_impl[0]) == Status::kOk);
        staged_block(x_v2, &xw_v2, -1, -1, p2, wide_stage, acc_stage,
            scratch_stage);
        for (std::size_t p = 0; p < map_count; ++p) {
            CHECK(wide_impl[p] == wide_stage[p]);
            CHECK(acc_impl[p] == acc_stage[p]);
        }
        CHECK(all_sentinel(wide_impl + map_count, (1 << 16) - map_count));
        CHECK(all_sentinel(acc_impl + map_count, (1 << 16) - map_count));
    }
    head_scale_buf[0] = 1.25f;
    head_scale_buf[1] = 0.75f;
    // The mid-path width bound's accepting edge: heads = 2 with mid.cols = 512
    // gives expand.rows == kMaxLinearDim and runs end to end against the staged
    // chain (the same shape with mid.cols = 513 is rejected in the negatives).
    static float edge_mid[fixture::kWideChannels * 512];
    static float edge_expand[1024 * fixture::kWideChannels];
    static float edge_contract[fixture::kWideChannels * fixture::kWideChannels];
    for (std::size_t p = 0; p < sizeof edge_mid / sizeof edge_mid[0]; ++p) {
        edge_mid[p] = dlss5_nr::e4m3(rnd_f16() - 1.5f);
    }
    for (std::size_t p = 0; p < sizeof edge_expand / sizeof edge_expand[0];
            ++p) {
        edge_expand[p] = dlss5_nr::e4m3(rnd_f16() - 1.5f);
    }
    for (std::size_t p = 0; p < sizeof edge_contract / sizeof edge_contract[0];
            ++p) {
        edge_contract[p] = dlss5_nr::e4m3(rnd_f16() - 1.5f);
    }
    {
        const std::ptrdiff_t row_stride2 = cols2 + 3;
        const std::ptrdiff_t plane_stride2 =
            static_cast<std::ptrdiff_t>(rows2) * row_stride2;
        for (std::ptrdiff_t p = 0; p < plane_stride2 * channels; ++p) {
            x2_buf[p] = 0.25f;
        }
        for (int c = 0; c < channels; ++c) {
            for (int r = 0; r < rows2; ++r) {
                for (int col = 0; col < cols2; ++col) {
                    x2_buf[c * plane_stride2 + r * row_stride2 + col]
                        = rnd_f16();
                }
            }
        }
        for (std::size_t p = 0; p < map_count; ++p) {
            x2w_buf[p] = rnd_f16();
        }
        const FeatureBatchView x_v2 = make_batch(x2_buf, channels, rows2,
            cols2, plane_stride2, row_stride2);
        const FeatureBatchView xw_v2 = make_batch(x2w_buf, channels, rows2,
            cols2, static_cast<std::ptrdiff_t>(rows2) * cols2, cols2);
        MatrixView edge_mid_v = make_view(edge_mid, channels, 512, 512);
        SwinBlockParams p3 = p2;
        p3.mid = &edge_mid_v;
        p3.expand = make_view(edge_expand, 1024, channels, channels);
        p3.contract = make_view(edge_contract, channels, channels, channels);
        fill_sentinel(wide_impl, 1 << 16);
        fill_sentinel(wide_stage, 1 << 16);
        fill_sentinel(acc_impl, 1 << 16);
        fill_sentinel(acc_stage, 1 << 16);
        CHECK(swin_block(x_v2, &xw_v2, -1, -1, p3, wide_impl, map_count,
            acc_impl, map_count, scratch_impl,
            sizeof scratch_impl / sizeof scratch_impl[0]) == Status::kOk);
        staged_block(x_v2, &xw_v2, -1, -1, p3, wide_stage, acc_stage,
            scratch_stage);
        for (std::size_t p = 0; p < map_count; ++p) {
            CHECK(wide_impl[p] == wide_stage[p]);
            CHECK(acc_impl[p] == acc_stage[p]);
        }
        CHECK(all_sentinel(wide_impl + map_count, (1 << 16) - map_count));
        CHECK(all_sentinel(acc_impl + map_count, (1 << 16) - map_count));
    }
}

// ---- negatives --------------------------------------------------------------------------------------

static void check_negatives(void) {
    using dlss5_nr::swin_block;
    using dlss5_nr::swin_core;
    fixture::BlockFixture f;
    f.zero_init();
    static float x[fixture::kChannels * fixture::kMapRows * fixture::kMapCols];
    static float x_wide[fixture::kChannels * fixture::kMapRows * fixture::kMapCols];
    static float wide[fixture::kChannels * fixture::kMapRows * fixture::kMapCols];
    static float acc[fixture::kChannels * fixture::kMapRows * fixture::kMapCols];
    static float scratch[1 << 18];
    const std::size_t map_count = fixture::kChannels * fixture::kMapRows
        * fixture::kMapCols;
    for (std::size_t p = 0; p < map_count; ++p) {
        x[p] = 1.0f;
        x_wide[p] = 1.0f;
    }
    const FeatureBatchView x_v = make_batch(x, fixture::kChannels,
        fixture::kMapRows, fixture::kMapCols,
        fixture::kMapRows * fixture::kMapCols, fixture::kMapCols);
    const FeatureBatchView xw_v = make_batch(x_wide, fixture::kChannels,
        fixture::kMapRows, fixture::kMapCols,
        fixture::kMapRows * fixture::kMapCols, fixture::kMapCols);
    const std::size_t scratch_cap = sizeof scratch / sizeof scratch[0];
    const auto untouched = [&](void) {
        CHECK(all_sentinel(wide, map_count));
        CHECK(all_sentinel(acc, map_count));
        CHECK(all_sentinel(scratch, scratch_cap));
    };
    fill_sentinel(wide, map_count);
    fill_sentinel(acc, map_count);
    fill_sentinel(scratch, scratch_cap);

    // Null pointers.
    CHECK(swin_block(make_batch(nullptr, fixture::kChannels, fixture::kMapRows,
        fixture::kMapCols, 0, 0), nullptr, 0, 0, f.params, wide, map_count,
        acc, map_count, scratch, scratch_cap) == Status::kNullPointer);
    CHECK(swin_block(x_v, nullptr, 0, 0, f.params, nullptr, map_count,
        acc, map_count, scratch, scratch_cap) == Status::kNullPointer);
    CHECK(swin_block(x_v, nullptr, 0, 0, f.params, wide, map_count,
        acc, map_count, nullptr, scratch_cap) == Status::kNullPointer);
    {
        SwinBlockParams p = f.params;
        p.expand.data = nullptr;
        CHECK(swin_block(x_v, nullptr, 0, 0, p, wide, map_count, acc, map_count,
            scratch, scratch_cap) == Status::kNullPointer);
        p = f.params;
        p.qkv.data = nullptr;
        CHECK(swin_block(x_v, nullptr, 0, 0, p, wide, map_count, acc, map_count,
            scratch, scratch_cap) == Status::kNullPointer);
        p = f.params;
        p.exp_bias.data = nullptr;
        CHECK(swin_block(x_v, nullptr, 0, 0, p, wide, map_count, acc, map_count,
            scratch, scratch_cap) == Status::kNullPointer);
        MatrixView mid_null = make_view(nullptr, 32, 2, 2);
        p = f.params;
        p.mid = &mid_null;
        CHECK(swin_block(x_v, nullptr, 0, 0, p, wide, map_count, acc, map_count,
            scratch, scratch_cap) == Status::kNullPointer);
    }
    {
        const FeatureBatchView xw_null = make_batch(nullptr, fixture::kChannels,
            fixture::kMapRows, fixture::kMapCols, 0, 0);
        CHECK(swin_block(x_v, &xw_null, 0, 0, f.params, wide, map_count,
            acc, map_count, scratch, scratch_cap) == Status::kNullPointer);
    }
    untouched();

    // Dimension classes of x (row/col/plane bounds and the channels bound).
    CHECK(swin_block(make_batch(x, 0, fixture::kMapRows, fixture::kMapCols, 0,
        fixture::kMapCols), nullptr, 0, 0, f.params, wide, map_count, acc,
        map_count, scratch, scratch_cap) == Status::kInvalidDimensions);
    CHECK(swin_block(make_batch(x, dlss5_nr::kMaxTokenChannels + 1,
        fixture::kMapRows, fixture::kMapCols, 0, fixture::kMapCols),
        nullptr, 0, 0, f.params, wide, map_count, acc, map_count,
        scratch, scratch_cap) == Status::kInvalidDimensions);
    CHECK(swin_block(make_batch(x, fixture::kChannels, 0, fixture::kMapCols, 0,
        fixture::kMapCols), nullptr, 0, 0, f.params, wide, map_count, acc,
        map_count, scratch, scratch_cap) == Status::kInvalidDimensions);
    CHECK(swin_block(make_batch(x, fixture::kChannels, dlss5_nr::kMaxRows + 1,
        fixture::kMapCols, 0, fixture::kMapCols), nullptr, 0, 0, f.params,
        wide, map_count, acc, map_count, scratch, scratch_cap)
        == Status::kInvalidDimensions);
    CHECK(swin_block(make_batch(x, fixture::kChannels, fixture::kMapRows,
        dlss5_nr::kMaxCols + 1, 0, fixture::kMapCols), nullptr, 0, 0,
        f.params, wide, map_count, acc, map_count, scratch, scratch_cap)
        == Status::kInvalidDimensions);
    // Stride classes of x.
    CHECK(swin_block(make_batch(x, fixture::kChannels, fixture::kMapRows,
        fixture::kMapCols, fixture::kMapRows * fixture::kMapCols,
        fixture::kMapCols - 1), nullptr, 0, 0, f.params, wide, map_count, acc,
        map_count, scratch, scratch_cap) == Status::kInvalidStride);
    CHECK(swin_block(make_batch(x, fixture::kChannels, fixture::kMapRows,
        fixture::kMapCols, fixture::kMapRows * fixture::kMapCols,
        dlss5_nr::kMaxCols + 1), nullptr, 0, 0, f.params, wide, map_count, acc,
        map_count, scratch, scratch_cap) == Status::kInvalidStride);
    CHECK(swin_block(make_batch(x, fixture::kChannels, fixture::kMapRows,
        fixture::kMapCols, fixture::kMapRows * fixture::kMapCols - 1,
        fixture::kMapCols), nullptr, 0, 0, f.params, wide, map_count, acc,
        map_count, scratch, scratch_cap) == Status::kInvalidStride);
    CHECK(swin_block(make_batch(x, fixture::kChannels, fixture::kMapRows,
        fixture::kMapCols, dlss5_nr::kMaxRows * dlss5_nr::kMaxCols + 1,
        fixture::kMapCols), nullptr, 0, 0, f.params, wide, map_count, acc,
        map_count, scratch, scratch_cap) == Status::kInvalidStride);
    // Shift classes (only 0 and -1 exist; swin_window_grid rejects others
    // with kInvalidControls, which the block forwards).
    CHECK(swin_block(x_v, nullptr, 1, 0, f.params, wide, map_count, acc,
        map_count, scratch, scratch_cap) == Status::kInvalidControls);
    CHECK(swin_block(x_v, nullptr, 0, -2, f.params, wide, map_count, acc,
        map_count, scratch, scratch_cap) == Status::kInvalidControls);
    untouched();

    // x_wide pairing/stride/value classes.
    {
        const FeatureBatchView xw_bad_dims = make_batch(x_wide,
            fixture::kChannels - 1, fixture::kMapRows, fixture::kMapCols, 0,
            fixture::kMapCols);
        CHECK(swin_block(x_v, &xw_bad_dims, 0, 0, f.params, wide, map_count,
            acc, map_count, scratch, scratch_cap) == Status::kInvalidDimensions);
    }
    {
        const FeatureBatchView xw_bad_stride = make_batch(x_wide,
            fixture::kChannels, fixture::kMapRows, fixture::kMapCols, 0,
            fixture::kMapCols - 1);
        CHECK(swin_block(x_v, &xw_bad_stride, 0, 0, f.params, wide, map_count,
            acc, map_count, scratch, scratch_cap) == Status::kInvalidStride);
    }
    x_wide[0] = 0.1f;  // off the binary16 grid
    CHECK(swin_block(x_v, &xw_v, 0, 0, f.params, wide, map_count, acc,
        map_count, scratch, scratch_cap) == Status::kInvalidValues);
    x_wide[0] = (float)NAN;
    CHECK(swin_block(x_v, &xw_v, 0, 0, f.params, wide, map_count, acc,
        map_count, scratch, scratch_cap) == Status::kInvalidValues);
    x_wide[0] = 1.0f;
    x[0] = (float)INFINITY;
    CHECK(swin_block(x_v, nullptr, 0, 0, f.params, wide, map_count, acc,
        map_count, scratch, scratch_cap) == Status::kInvalidValues);
    x[0] = 1.0f;
    untouched();

    // Pairing classes (the staged entries' pairings, mirrored).
    {
        SwinBlockParams p = f.params;
        p.heads = 2;  // channels != heads * kHeadDim
        CHECK(swin_block(x_v, nullptr, 0, 0, p, wide, map_count, acc, map_count,
            scratch, scratch_cap) == Status::kInvalidDimensions);
        p = f.params;
        p.heads = 0;
        CHECK(swin_block(x_v, nullptr, 0, 0, p, wide, map_count, acc, map_count,
            scratch, scratch_cap) == Status::kInvalidDimensions);
        p = f.params;
        p.heads = 11;
        // The 352-channel map exceeds kMaxTokenChannels, so this call observes
        // the token-channel bound in swin_block, not the pairings' heads bound
        // (3 * 352 > kMaxLinearDim is only the underlying reason that bound
        // exists).
        CHECK(swin_block(make_batch(x, 11 * dlss5_nr::kHeadDim, 1, 1, 1, 1),
            nullptr, 0, 0, p, wide, map_count, acc, map_count, scratch,
            scratch_cap) == Status::kInvalidDimensions);
        // The pairings' heads bound itself (heads = 11 >
        // kMaxLinearDim / (3 * kHeadDim) = 10) pinned both through the entry
        // and through the helper: with an in-bound 32-channel map it is the
        // first check to fire (the channels pairing below it would also
        // reject this shape, with the same status).
        CHECK(swin_block(x_v, nullptr, 0, 0, p, wide, map_count, acc,
            map_count, scratch, scratch_cap) == Status::kInvalidDimensions);
        CHECK(swin_block_pairings(x_v, p) == Status::kInvalidDimensions);
        p = f.params;
        p.expand = make_view(f.expand, fixture::kHidden,
            fixture::kChannels - 1, fixture::kChannels - 1);
        CHECK(swin_block(x_v, nullptr, 0, 0, p, wide, map_count, acc, map_count,
            scratch, scratch_cap) == Status::kInvalidDimensions);
        p = f.params;
        p.qkv = make_view(f.qkv, 3 * fixture::kChannels - 1, fixture::kChannels,
            fixture::kChannels);
        CHECK(swin_block(x_v, nullptr, 0, 0, p, wide, map_count, acc, map_count,
            scratch, scratch_cap) == Status::kInvalidDimensions);
        p = f.params;
        p.out_proj = make_view(f.out_proj, fixture::kChannels - 1,
            fixture::kChannels, fixture::kChannels);
        CHECK(swin_block(x_v, nullptr, 0, 0, p, wide, map_count, acc, map_count,
            scratch, scratch_cap) == Status::kInvalidDimensions);
        p = f.params;
        p.contract = make_view(f.contract, fixture::kChannels,
            fixture::kHidden - 1, fixture::kHidden - 1);
        CHECK(swin_block(x_v, nullptr, 0, 0, p, wide, map_count, acc, map_count,
            scratch, scratch_cap) == Status::kInvalidDimensions);
        p = f.params;
        p.mlp_scale = make_view(f.mlp_scale, 1, fixture::kChannels - 1,
            fixture::kChannels - 1);
        CHECK(swin_block(x_v, nullptr, 0, 0, p, wide, map_count, acc, map_count,
            scratch, scratch_cap) == Status::kInvalidDimensions);
        p = f.params;
        p.head_scale = make_view(f.head_scale, 2, 1, 1);
        CHECK(swin_block(x_v, nullptr, 0, 0, p, wide, map_count, acc, map_count,
            scratch, scratch_cap) == Status::kInvalidDimensions);
        p = f.params;
        p.exp_bias = make_view(f.exp_bias, 2, dlss5_nr::kBiasTable,
            dlss5_nr::kBiasTable);
        CHECK(swin_block(x_v, nullptr, 0, 0, p, wide, map_count, acc, map_count,
            scratch, scratch_cap) == Status::kInvalidDimensions);
        p = f.params;
        p.attn_scale = make_view(f.attn_scale, 1, fixture::kChannels - 1,
            fixture::kChannels - 1);
        CHECK(swin_block(x_v, nullptr, 0, 0, p, wide, map_count, acc, map_count,
            scratch, scratch_cap) == Status::kInvalidDimensions);
        // The mid-path width bound: expand.rows == heads * mid.cols must stay
        // within kMaxLinearDim (the MLP scratch width bound the staged call
        // validates), so heads = 2 with mid.cols = 513 is rejected here rather
        // than failing after the first write. Full-size operand buffers are
        // passed so the case is robust to any future validation reordering.
        static float x64[64 * fixture::kMapRows * fixture::kMapCols];
        static float neg_expand[1026 * 64];
        static float neg_mid[64 * 513];
        static float neg_contract[64 * 64];
        static float neg_qkv[3 * 64 * 64];
        static float neg_out_proj[64 * 64];
        static float neg_mlp_scale[64];
        static float neg_attn_scale[64];
        static float neg_head_scale[2];
        static float neg_exp_bias[2 * dlss5_nr::kBiasTable];
        memset(x64, 0, sizeof x64);
        memset(neg_expand, 0, sizeof neg_expand);
        memset(neg_mid, 0, sizeof neg_mid);
        memset(neg_contract, 0, sizeof neg_contract);
        memset(neg_qkv, 0, sizeof neg_qkv);
        memset(neg_out_proj, 0, sizeof neg_out_proj);
        memset(neg_exp_bias, 0, sizeof neg_exp_bias);
        for (int c = 0; c < 64; ++c) {
            neg_mlp_scale[c] = 1.0f;
            neg_attn_scale[c] = 1.0f;
        }
        neg_head_scale[0] = 1.0f;
        neg_head_scale[1] = 1.0f;
        const FeatureBatchView x64_v = make_batch(x64, 64, fixture::kMapRows,
            fixture::kMapCols, fixture::kMapRows * fixture::kMapCols,
            fixture::kMapCols);
        p = f.params;
        p.heads = 2;
        const MatrixView neg_mid_v = make_view(neg_mid, 64, 513, 513);
        p.mid = &neg_mid_v;
        p.expand = make_view(neg_expand, 1026, 64, 64);
        p.contract = make_view(neg_contract, 64, 64, 64);
        p.qkv = make_view(neg_qkv, 3 * 64, 64, 64);
        p.out_proj = make_view(neg_out_proj, 64, 64, 64);
        p.mlp_scale = make_view(neg_mlp_scale, 1, 64, 64);
        p.attn_scale = make_view(neg_attn_scale, 1, 64, 64);
        p.head_scale = make_view(neg_head_scale, 1, 2, 2);
        p.exp_bias = make_view(neg_exp_bias, 2, dlss5_nr::kBiasTable,
            dlss5_nr::kBiasTable);
        CHECK(swin_block(x64_v, nullptr, 0, 0, p, wide, map_count, acc,
            map_count, scratch, scratch_cap) == Status::kInvalidDimensions);
    }
    untouched();

    // Weight value classes (one per operand class).
    {
        static float bad_expand[32 * fixture::kChannels];
        memcpy(bad_expand, f.expand, sizeof bad_expand);
        bad_expand[0] = 0.1f;  // off the e4m3 grid
        SwinBlockParams p = f.params;
        p.expand = make_view(bad_expand, 32, fixture::kChannels,
            fixture::kChannels);
        CHECK(swin_block(x_v, nullptr, 0, 0, p, wide, map_count, acc, map_count,
            scratch, scratch_cap) == Status::kInvalidValues);
        static float bad_scale[fixture::kChannels];
        memcpy(bad_scale, f.mlp_scale, sizeof bad_scale);
        bad_scale[0] = 0.1f;  // off the binary16 grid
        p = f.params;
        p.mlp_scale = make_view(bad_scale, 1, fixture::kChannels,
            fixture::kChannels);
        CHECK(swin_block(x_v, nullptr, 0, 0, p, wide, map_count, acc, map_count,
            scratch, scratch_cap) == Status::kInvalidValues);
        static float bad_head[1];
        bad_head[0] = (float)INFINITY;
        p = f.params;
        p.head_scale = make_view(bad_head, 1, 1, 1);
        CHECK(swin_block(x_v, nullptr, 0, 0, p, wide, map_count, acc, map_count,
            scratch, scratch_cap) == Status::kInvalidValues);
        // The head-scale magnitude bound: finite values large enough to overflow
        // the `_norm_scale * head_scale` product to infinity are rejected at
        // validation (an infinity there times a zero q dimension would be NaN
        // and would otherwise fail the e4m3 operand checks mid-write).
        static float huge_head[1];
        huge_head[0] = 1e38f;
        p = f.params;
        p.head_scale = make_view(huge_head, 1, 1, 1);
        CHECK(swin_block(x_v, nullptr, 0, 0, p, wide, map_count, acc, map_count,
            scratch, scratch_cap) == Status::kInvalidValues);
        static float over_head[1];
        over_head[0] = -70000.0f;  // finite, just beyond kMaxHeadScale
        p = f.params;
        p.head_scale = make_view(over_head, 1, 1, 1);
        CHECK(swin_block(x_v, nullptr, 0, 0, p, wide, map_count, acc, map_count,
            scratch, scratch_cap) == Status::kInvalidValues);
        static float bad_bias[dlss5_nr::kBiasTable];
        memcpy(bad_bias, f.exp_bias, sizeof bad_bias);
        bad_bias[0] = (float)NAN;
        p = f.params;
        p.exp_bias = make_view(bad_bias, 1, dlss5_nr::kBiasTable,
            dlss5_nr::kBiasTable);
        CHECK(swin_block(x_v, nullptr, 0, 0, p, wide, map_count, acc, map_count,
            scratch, scratch_cap) == Status::kInvalidValues);
    }
    untouched();

    // Boundary head-scale acceptance: the kMaxHeadScale edge is inside the
    // accepted domain (the scale product stays finite, so the zero-weight
    // closed form survives it unchanged).
    {
        static float edge_head[1];
        edge_head[0] = -dlss5_nr::kMaxHeadScale;
        SwinBlockParams p = f.params;
        p.head_scale = make_view(edge_head, 1, 1, 1);
        CHECK(swin_block(x_v, nullptr, 0, 0, p, wide, map_count, acc, map_count,
            scratch, scratch_cap) == Status::kOk);
        for (std::size_t i = 0; i < map_count; ++i) {
            CHECK(wide[i] == 1.0f);
            CHECK(acc[i] == 1.0f);
        }
    }
    fill_sentinel(wide, map_count);
    fill_sentinel(acc, map_count);
    fill_sentinel(scratch, scratch_cap);

    // Capacity classes: the scratch sizing includes the in-scratch acc map
    // exactly when acc_dst is null (acc_count is ignored there).
    CHECK(swin_block(x_v, nullptr, 0, 0, f.params, wide, map_count - 1,
        acc, map_count, scratch, scratch_cap) == Status::kInvalidBuffer);
    CHECK(swin_block(x_v, nullptr, 0, 0, f.params, wide, map_count,
        acc, map_count - 1, scratch, scratch_cap) == Status::kInvalidBuffer);
    CHECK(swin_block(x_v, nullptr, 0, 0, f.params, wide, map_count,
        acc, map_count, scratch, 0) == Status::kInvalidBuffer);
    untouched();
    // The exact scratch boundary of this fixture (C = 32, 8x8, one window,
    // expand.rows = 32: 2048 map + 2048 tokens + 2048 out tokens + 4096 y/y_q
    // + 2048 MLP scratch + 22528 attention scratch = 34816 with acc_dst given,
    // plus one 2048-float acc map when acc_dst is null = 36864). One float
    // less rejects; exactly there accepts, and the acc-null call takes its acc
    // map from scratch and ignores acc_count entirely.
    const std::size_t needed_acc = 34816;
    const std::size_t needed_no_acc = 36864;
    CHECK(swin_block(x_v, nullptr, 0, 0, f.params, wide, map_count,
        acc, map_count, scratch, needed_acc - 1) == Status::kInvalidBuffer);
    untouched();
    CHECK(swin_block(x_v, nullptr, 0, 0, f.params, wide, map_count,
        nullptr, 0, scratch, needed_no_acc - 1) == Status::kInvalidBuffer);
    untouched();
    CHECK(swin_block(x_v, nullptr, 0, 0, f.params, wide, map_count,
        acc, map_count, scratch, needed_acc) == Status::kOk);
    for (std::size_t i = 0; i < map_count; ++i) {
        CHECK(wide[i] == 1.0f);  // the zero-weight closed form
        CHECK(acc[i] == 1.0f);
    }
    fill_sentinel(wide, map_count);
    fill_sentinel(acc, map_count);
    fill_sentinel(scratch, scratch_cap);
    CHECK(swin_block(x_v, nullptr, 0, 0, f.params, wide, map_count,
        nullptr, 0, scratch, needed_no_acc) == Status::kOk);
    for (std::size_t i = 0; i < map_count; ++i) {
        CHECK(wide[i] == 1.0f);  // identical wide through the in-scratch acc map
    }
    fill_sentinel(wide, map_count);
    fill_sentinel(acc, map_count);
    fill_sentinel(scratch, scratch_cap);
    // swin_core requires f16-exact x for its residual role.
    x[0] = 1.0f + 0x1p-11f;
    CHECK(swin_core(x_v, 0, 0, f.params, wide, map_count, acc, map_count,
        scratch, scratch_cap) == Status::kInvalidValues);
    x[0] = 1.0f;
    untouched();
}

// ---- determinism and zero allocation -----------------------------------------------------------------

static void check_determinism_and_allocations(void) {
    using dlss5_nr::swin_block;
    fixture::BlockFixture f;
    f.zero_init();
    static float x[fixture::kChannels * fixture::kMapRows * fixture::kMapCols];
    static float x_wide[fixture::kChannels * fixture::kMapRows * fixture::kMapCols];
    static float wide_a[fixture::kChannels * fixture::kMapRows * fixture::kMapCols];
    static float wide_b[fixture::kChannels * fixture::kMapRows * fixture::kMapCols];
    static float acc_a[fixture::kChannels * fixture::kMapRows * fixture::kMapCols];
    static float acc_b[fixture::kChannels * fixture::kMapRows * fixture::kMapCols];
    static float scratch[1 << 18];
    const std::size_t map_count = fixture::kChannels * fixture::kMapRows
        * fixture::kMapCols;
    for (std::size_t p = 0; p < map_count; ++p) {
        x[p] = 1.0f + 0x1p-10f;
        x_wide[p] = 1.5f;
    }
    const FeatureBatchView x_v = make_batch(x, fixture::kChannels,
        fixture::kMapRows, fixture::kMapCols,
        fixture::kMapRows * fixture::kMapCols, fixture::kMapCols);
    const FeatureBatchView xw_v = make_batch(x_wide, fixture::kChannels,
        fixture::kMapRows, fixture::kMapCols,
        fixture::kMapRows * fixture::kMapCols, fixture::kMapCols);
    const unsigned long long before = g_allocations.load();
    for (int run = 0; run < 64; ++run) {
        CHECK(swin_block(x_v, (run % 2) == 0 ? &xw_v : nullptr, -1, -1,
            f.params, (run % 2) == 0 ? wide_a : wide_b, map_count,
            (run % 2) == 0 ? acc_a : acc_b, map_count, scratch,
            sizeof scratch / sizeof scratch[0]) == Status::kOk);
    }
    CHECK(g_allocations.load() == before);
    // Determinism within each residual path.
    CHECK(swin_block(x_v, &xw_v, -1, -1, f.params, wide_a, map_count, acc_a,
        map_count, scratch, sizeof scratch / sizeof scratch[0]) == Status::kOk);
    CHECK(swin_block(x_v, &xw_v, -1, -1, f.params, wide_b, map_count, acc_b,
        map_count, scratch, sizeof scratch / sizeof scratch[0]) == Status::kOk);
    for (std::size_t p = 0; p < map_count; ++p) {
        CHECK(memcmp(&wide_a[p], &wide_b[p], sizeof(float)) == 0);
        CHECK(memcmp(&acc_a[p], &acc_b[p], sizeof(float)) == 0);
    }
    CHECK(swin_block(x_v, nullptr, -1, -1, f.params, wide_a, map_count, acc_a,
        map_count, scratch, sizeof scratch / sizeof scratch[0]) == Status::kOk);
    CHECK(swin_block(x_v, nullptr, -1, -1, f.params, wide_b, map_count, acc_b,
        map_count, scratch, sizeof scratch / sizeof scratch[0]) == Status::kOk);
    for (std::size_t p = 0; p < map_count; ++p) {
        CHECK(memcmp(&wide_a[p], &wide_b[p], sizeof(float)) == 0);
        CHECK(memcmp(&acc_a[p], &acc_b[p], sizeof(float)) == 0);
    }
}

int main(void) {
    check_closed_forms();
    check_equivalence();
    check_negatives();
    check_determinism_and_allocations();
    printf("native-swin-block-test: %lld checks passed\n", g_checks);
    return 0;
}
