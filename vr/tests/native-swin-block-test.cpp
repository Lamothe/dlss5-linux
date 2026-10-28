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
// The G8 batched chunk seam: chunked vs per-window vs no-site bit-identity
// across the window-count matrix {1, 2, 16, 32, 33, 92} (both head families,
// both residual paths, zero and non-trivial corpora, the acc modes), the
// executor call-shape contract (one expand + one QKV call per
// kSwinBatchWindows-window chunk, chunk-sized, versus one per window), every
// capacity/shape fallback predicate keeping the per-window path, the
// null-executor rejection class (the G4 site ordering) and the
// swin_layer_plain site threading — all through a counting CPU executor that
// honours the ProjectionExecutor product contract.
// See vr/NATIVE-SWIN-BLOCK.md.

#include "native-swin-resample.hpp"  // swin_block_scratch_needed + the layer entries

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>
#include <limits>
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

// ---- the G8 batched chunk seam -------------------------------------------------------------------
// The chunked path (one swin_mlp/swin_attention call per kSwinBatchWindows
// windows when the site carries chunk-sized batch work) must be bit-identical
// to the per-window path stage for stage, and the executor must see
// CHUNK-SIZED calls — the whole point of the seam: no per-window transfer
// thrash. A counting CPU executor stands in for the GPU backend: it honours
// the documented ProjectionExecutor contract (row-wise k-ordered float32
// accumulation from +0.0, bit-identical to the CPU kernel) and records every
// call's shape.

struct CountingExecutor {
    static constexpr int kMaxCalls = 8192;
    long long calls = 0;
    int tokens[kMaxCalls];
    int c_in[kMaxCalls];
    int rows[kMaxCalls];
    bool overflow = false;
};

static Status counting_matmul_f32(void* engine, const float* in, int tokens,
                                  int c_in, const float* weights, int rows,
                                  float* out) {
    auto* ctx = static_cast<CountingExecutor*>(engine);
    if (ctx->calls < CountingExecutor::kMaxCalls) {
        ctx->tokens[ctx->calls] = tokens;
        ctx->c_in[ctx->calls] = c_in;
        ctx->rows[ctx->calls] = rows;
    } else {
        ctx->overflow = true;
    }
    ++ctx->calls;
    for (int t = 0; t < tokens; ++t) {
        for (int n = 0; n < rows; ++n) {
            float sum = 0.0f;
            for (int c = 0; c < c_in; ++c) {
                sum += in[static_cast<std::size_t>(t) * c_in + c]
                    * weights[static_cast<std::size_t>(n) * c_in + c];
            }
            out[static_cast<std::size_t>(t) * rows + n] = sum;
        }
    }
    return Status::kOk;
}

// The three site shapes for one config: exact chunk-sized input/products and
// the swin_batch_work_needed chunk region (the swin_site_scratch_needed
// order). `chunk_rows` is the largest chunk the map can produce. The executor
// record lives in the handle so each site keeps its own engine; the
// gather/scatter/batch buffers below are shared statics — build and use one
// site at a time.
struct BatchSite {
    static constexpr std::size_t kMaxChunkRows =
        static_cast<std::size_t>(dlss5_nr::kSwinBatchWindows)
        * dlss5_nr::kWindowTokens;
    static float in_dense[kMaxChunkRows * 64];
    static float products[kMaxChunkRows * 192];
    static float work[1 << 21];
    dlss5_nr::ProjectionExecutor ex;

    void build(dlss5_nr::SwinProjectionSite& site, CountingExecutor& engine,
               int chunk_rows, int channels, int expand_rows, int heads) {
        site = dlss5_nr::SwinProjectionSite{};
        ex = dlss5_nr::ProjectionExecutor{};
        ex.ctx = &engine;
        ex.matmul_f32 = &counting_matmul_f32;
        site.ex = &ex;
        site.input_dense = in_dense;
        site.input_capacity = static_cast<std::size_t>(chunk_rows) * channels;
        site.products = products;
        const std::size_t expand = static_cast<std::size_t>(expand_rows);
        const std::size_t qkv = static_cast<std::size_t>(3) * channels;
        site.products_capacity = static_cast<std::size_t>(chunk_rows)
            * (expand > qkv ? expand : qkv);
        site.batch_work = work;
        site.batch_capacity = dlss5_nr::swin_batch_work_needed(
            chunk_rows, channels, expand_rows, heads);
    }
};

float BatchSite::in_dense[BatchSite::kMaxChunkRows * 64];
float BatchSite::products[BatchSite::kMaxChunkRows * 192];
float BatchSite::work[1 << 21];

static void check_batched_seam(void) {
    using dlss5_nr::swin_block;
    using dlss5_nr::swin_core;
    using dlss5_nr::swin_layer_plain;
    using dlss5_nr::SwinProjectionSite;
    // Window-count matrix around the chunk boundary (windows = 1, 2, 16 with
    // the shifted 4x4 grid, 32 = one exact chunk, 33 = chunk + tail, and
    // 92 = three chunks + tail at shift (-1, -1)), both residual paths, zero
    // and non-trivial weights, both head families, and the acc-destination
    // modes.
    const int rows[] = {8, 8, 24, 8, 8, 8};
    const int cols[] = {8, 16, 24, 256, 264, 360};
    const int shift_r[] = {0, 0, -1, 0, 0, -1};
    const int shift_c[] = {0, 0, -1, 0, 0, -1};
    const int heads_cfg[] = {1, 1, 2, 1, 1, 1};
    const long long windows_cfg[] = {1, 2, 16, 32, 33, 92};
    static float x_buf[1 << 18];
    static float xw_buf[1 << 18];
    static float wide_cpu[1 << 18];
    static float wide_gpu[1 << 18];
    static float wide_pw[1 << 18];
    static float acc_cpu[1 << 18];
    static float acc_gpu[1 << 18];
    static float acc_pw[1 << 18];
    static float scratch[1 << 21];
    for (int cfg = 0; cfg < 6; ++cfg) {
        for (int path = 0; path < 2; ++path) {
            for (int corpus = 0; corpus < 2; ++corpus) {
                const int channels = heads_cfg[cfg] == 1 ? fixture::kChannels
                                                        : fixture::kWideChannels;
                const int heads = heads_cfg[cfg];
                // The single-head fixture keeps expand.rows = kHidden; the
                // multi-head grouped-mid fixture is built inline below.
                fixture::BlockFixture f;
                f.zero_init();
                static MatrixView mid_view;
                static float mid_buf[fixture::kWideChannels * 8];
                static float expand_buf[16 * fixture::kWideChannels];
                static float contract_buf[fixture::kWideChannels
                    * fixture::kWideChannels];
                static float mlp_scale_buf[fixture::kWideChannels];
                static float qkv_buf[3 * fixture::kWideChannels
                    * fixture::kWideChannels];
                static float head_scale_buf[2];
                static float exp_bias_buf[2 * dlss5_nr::kBiasTable];
                static float attn_scale_buf[fixture::kWideChannels];
                static float out_proj_buf[fixture::kWideChannels
                    * fixture::kWideChannels];
                SwinBlockParams params = f.params;
                int expand_rows = fixture::kHidden;
                if (heads == 2) {
                    for (std::size_t p = 0; p < sizeof mid_buf
                            / sizeof mid_buf[0]; ++p) {
                        mid_buf[p] = 0.0f;
                    }
                    memset(expand_buf, 0, sizeof expand_buf);
                    memset(contract_buf, 0, sizeof contract_buf);
                    memset(qkv_buf, 0, sizeof qkv_buf);
                    memset(out_proj_buf, 0, sizeof out_proj_buf);
                    memset(exp_bias_buf, 0, sizeof exp_bias_buf);
                    for (int c = 0; c < fixture::kWideChannels; ++c) {
                        mlp_scale_buf[c] = 1.0f;
                        attn_scale_buf[c] = 1.0f;
                    }
                    head_scale_buf[0] = 1.0f;
                    head_scale_buf[1] = 1.0f;
                    mid_view = make_view(mid_buf, fixture::kWideChannels, 8, 8);
                    params.mid = &mid_view;
                    params.heads = 2;
                    params.expand = make_view(expand_buf, 16,
                        fixture::kWideChannels, fixture::kWideChannels);
                    params.contract = make_view(contract_buf,
                        fixture::kWideChannels, fixture::kWideChannels,
                        fixture::kWideChannels);
                    params.mlp_scale = make_view(mlp_scale_buf, 1,
                        fixture::kWideChannels, fixture::kWideChannels);
                    params.qkv = make_view(qkv_buf, 3 * fixture::kWideChannels,
                        fixture::kWideChannels, fixture::kWideChannels);
                    params.head_scale = make_view(head_scale_buf, 1, 2, 2);
                    params.exp_bias = make_view(exp_bias_buf, 2,
                        dlss5_nr::kBiasTable, dlss5_nr::kBiasTable);
                    params.attn_scale = make_view(attn_scale_buf, 1,
                        fixture::kWideChannels, fixture::kWideChannels);
                    params.out_proj = make_view(out_proj_buf,
                        fixture::kWideChannels, fixture::kWideChannels,
                        fixture::kWideChannels);
                    expand_rows = 16;
                    if (corpus == 1) {
                        for (std::size_t p = 0; p < sizeof mid_buf
                                / sizeof mid_buf[0]; ++p) {
                            mid_buf[p] = dlss5_nr::e4m3(rnd_f16() - 1.5f);
                        }
                        for (std::size_t p = 0; p < sizeof expand_buf
                                / sizeof expand_buf[0]; ++p) {
                            expand_buf[p] = dlss5_nr::e4m3(rnd_f16() - 1.5f);
                        }
                        for (std::size_t p = 0; p < sizeof contract_buf
                                / sizeof contract_buf[0]; ++p) {
                            contract_buf[p] = dlss5_nr::e4m3(rnd_f16() - 1.5f);
                        }
                        for (std::size_t p = 0; p < sizeof qkv_buf
                                / sizeof qkv_buf[0]; ++p) {
                            qkv_buf[p] = dlss5_nr::e4m3(rnd_f16() - 1.5f);
                        }
                        for (std::size_t p = 0; p < sizeof out_proj_buf
                                / sizeof out_proj_buf[0]; ++p) {
                            out_proj_buf[p] = dlss5_nr::e4m3(rnd_f16() - 1.5f);
                        }
                        for (int c = 0; c < fixture::kWideChannels; ++c) {
                            mlp_scale_buf[c] = rnd_f16();
                            attn_scale_buf[c] = rnd_f16();
                        }
                        for (std::size_t p = 0; p < sizeof exp_bias_buf
                                / sizeof exp_bias_buf[0]; ++p) {
                            exp_bias_buf[p] = rnd_f16();
                        }
                    }
                } else if (corpus == 1) {
                    fill_random_weights(&f);
                }
                const std::size_t map_count =
                    static_cast<std::size_t>(channels) * rows[cfg] * cols[cfg];
                for (std::size_t p = 0; p < map_count; ++p) {
                    x_buf[p] = rnd_f16();
                    xw_buf[p] = rnd_f16();
                }
                const FeatureBatchView x_v = make_batch(x_buf, channels,
                    rows[cfg], cols[cfg],
                    static_cast<std::ptrdiff_t>(rows[cfg]) * cols[cfg],
                    cols[cfg]);
                const FeatureBatchView xw_v = make_batch(xw_buf, channels,
                    rows[cfg], cols[cfg],
                    static_cast<std::ptrdiff_t>(rows[cfg]) * cols[cfg],
                    cols[cfg]);
                const FeatureBatchView* wide_arg = path == 0 ? nullptr : &xw_v;
                const std::size_t scratch_need = swin_block_scratch_needed(
                    x_v, path != 0, true, shift_r[cfg], shift_c[cfg], params);
                CHECK(scratch_need > 0);
                int wr = 0, fr = 0, wc = 0, fc = 0;
                CHECK(dlss5_nr::swin_window_grid(rows[cfg], shift_r[cfg], &wr,
                                                 &fr) == Status::kOk);
                CHECK(dlss5_nr::swin_window_grid(cols[cfg], shift_c[cfg], &wc,
                                                 &fc) == Status::kOk);
                const long long windows =
                    static_cast<long long>(wr) * wc;
                CHECK(windows == windows_cfg[cfg]);
                const long long chunk_windows =
                    windows < dlss5_nr::kSwinBatchWindows
                        ? windows : dlss5_nr::kSwinBatchWindows;
                const int chunk_rows = static_cast<int>(
                    chunk_windows * dlss5_nr::kWindowTokens);
                // (1) the CPU reference (no site: the per-window kernels).
                CHECK(swin_block(x_v, wide_arg, shift_r[cfg], shift_c[cfg],
                    params, wide_cpu, map_count, acc_cpu, map_count, scratch,
                    scratch_need) == Status::kOk);
                // (2) the chunked seam: batch work present and exact.
                CountingExecutor engine;
                SwinProjectionSite site;
                BatchSite batch;
                batch.build(site, engine, chunk_rows, channels,
                            expand_rows, heads);
                CHECK(swin_block(x_v, wide_arg, shift_r[cfg], shift_c[cfg],
                    params, wide_gpu, map_count, acc_gpu, map_count, scratch,
                    scratch_need, &site) == Status::kOk);
                for (std::size_t p = 0; p < map_count; ++p) {
                    CHECK(memcmp(&wide_cpu[p], &wide_gpu[p], sizeof(float))
                          == 0);
                    CHECK(memcmp(&acc_cpu[p], &acc_gpu[p], sizeof(float)) == 0);
                }
                // The executor saw one expand + one QKV call per chunk, each
                // chunk-sized — no per-window calls.
                long long chunks = (windows + dlss5_nr::kSwinBatchWindows - 1)
                    / dlss5_nr::kSwinBatchWindows;
                long long expect_calls = 2 * chunks;
                CHECK(!engine.overflow);
                CHECK(engine.calls == expect_calls);
                long long w0 = 0;
                for (long long k = 0; k < expect_calls
                        && k < CountingExecutor::kMaxCalls; k += 2) {
                    const long long left = windows - w0;
                    const long long wc_batch =
                        left < dlss5_nr::kSwinBatchWindows
                            ? left : dlss5_nr::kSwinBatchWindows;
                    const int t = static_cast<int>(
                        wc_batch * dlss5_nr::kWindowTokens);
                    CHECK(engine.tokens[k] == t);
                    CHECK(engine.c_in[k] == channels);
                    CHECK(engine.rows[k] == expand_rows);
                    CHECK(engine.tokens[k + 1] == t);
                    CHECK(engine.c_in[k + 1] == channels);
                    CHECK(engine.rows[k + 1] == 3 * channels);
                    w0 += wc_batch;
                }
                // (3) the per-window seam (site without batch work): the
                // pre-G8 wiring, one expand + one QKV call PER WINDOW, and
                // the same bit-exact results.
                CountingExecutor engine_pw;
                SwinProjectionSite site_pw;
                BatchSite batch_pw;
                batch_pw.build(site_pw, engine_pw, chunk_rows, channels,
                               expand_rows, heads);
                site_pw.batch_work = nullptr;
                site_pw.batch_capacity = 0;
                CHECK(swin_block(x_v, wide_arg, shift_r[cfg], shift_c[cfg],
                    params, wide_pw, map_count, acc_pw, map_count, scratch,
                    scratch_need, &site_pw) == Status::kOk);
                for (std::size_t p = 0; p < map_count; ++p) {
                    CHECK(memcmp(&wide_cpu[p], &wide_pw[p], sizeof(float)) == 0);
                    CHECK(memcmp(&acc_cpu[p], &acc_pw[p], sizeof(float)) == 0);
                }
                CHECK(!engine_pw.overflow);
                CHECK(engine_pw.calls == 2 * windows);
                for (long long k = 0; k < 2 * windows
                        && k < CountingExecutor::kMaxCalls; ++k) {
                    CHECK(engine_pw.tokens[k]
                          == dlss5_nr::kWindowTokens);
                    CHECK(engine_pw.c_in[k] == channels);
                    CHECK(engine_pw.rows[k]
                          == ((k % 2) == 0 ? expand_rows : 3 * channels));
                }
            }
        }
    }
    // The swin_core form (x doubles as the f16 residual) is the same body:
    // one small chunked-vs-CPU equivalence pin.
    {
        fixture::BlockFixture f;
        f.zero_init();
        fill_random_weights(&f);
        const int channels = fixture::kChannels;
        static float x[fixture::kChannels * 8 * 360];
        static float wide_a[fixture::kChannels * 8 * 360];
        static float wide_b[fixture::kChannels * 8 * 360];
        static float acc_a[fixture::kChannels * 8 * 360];
        static float acc_b[fixture::kChannels * 8 * 360];
        const std::size_t map_count = std::size_t{channels} * 8 * 360;
        for (std::size_t p = 0; p < map_count; ++p) {
            x[p] = rnd_f16();
        }
        const FeatureBatchView x_v = make_batch(x, channels, 8, 360,
            static_cast<std::ptrdiff_t>(8) * 360, 360);
        const std::size_t need = swin_block_scratch_needed(x_v, true, true, -1,
            -1, f.params);
        CHECK(swin_core(x_v, -1, -1, f.params, wide_a, map_count, acc_a,
            map_count, scratch, need) == Status::kOk);
        CountingExecutor engine;
        SwinProjectionSite site;
        BatchSite batch;
        batch.build(site, engine, static_cast<int>(
            BatchSite::kMaxChunkRows), channels, fixture::kHidden, 1);
        CHECK(swin_core(x_v, -1, -1, f.params, wide_b, map_count, acc_b,
            map_count, scratch, need, &site) == Status::kOk);
        for (std::size_t p = 0; p < map_count; ++p) {
            CHECK(memcmp(&wide_a[p], &wide_b[p], sizeof(float)) == 0);
            CHECK(memcmp(&acc_a[p], &acc_b[p], sizeof(float)) == 0);
        }
        CHECK(engine.calls == 2 * 3);  // 92 windows at shift (-1, -1): 3 chunks
    }
    // The fallback predicates: every capacity/shape shortfall keeps the
    // per-window path (bit-identical results, per-window call shapes), and a
    // site without an executor keeps the CPU kernels (no executor calls).
    {
        fixture::BlockFixture f;
        f.zero_init();
        fill_random_weights(&f);
        const int channels = fixture::kChannels;
        const int rows = 8;
        const int cols = 264;
        const std::size_t map_count = std::size_t{channels} * rows * cols;
        static float x[1 << 18];
        static float wide_ref[1 << 18];
        static float wide_try[1 << 18];
        for (std::size_t p = 0; p < map_count; ++p) {
            x[p] = rnd_f16();
        }
        const FeatureBatchView x_v = make_batch(x, channels, rows, cols,
            static_cast<std::ptrdiff_t>(rows) * cols, cols);
        const std::size_t need = swin_block_scratch_needed(x_v, false, true, 0,
            0, f.params);
        CHECK(swin_block(x_v, nullptr, 0, 0, f.params, wide_ref, map_count,
            nullptr, 0, scratch, need) == Status::kOk);
        const int chunk_rows = static_cast<int>(BatchSite::kMaxChunkRows);
        const std::size_t batch_need = dlss5_nr::swin_batch_work_needed(
            chunk_rows, channels, fixture::kHidden, 1);
        const std::size_t input_need =
            static_cast<std::size_t>(chunk_rows) * channels;
        const std::size_t expand = static_cast<std::size_t>(fixture::kHidden);
        const std::size_t qkv = static_cast<std::size_t>(3) * channels;
        const std::size_t product_need = static_cast<std::size_t>(chunk_rows)
            * (expand > qkv ? expand : qkv);
        for (int pred = 0; pred < 8; ++pred) {
            CountingExecutor engine;
            SwinProjectionSite site;
            BatchSite batch;
            batch.build(site, engine, chunk_rows, channels,
                        fixture::kHidden, 1);
            // Pred 7 swaps in padded-stride expand AND qkv matrices (the G4
            // padded-weight CPU fallback: swin_linear runs its CPU kernel for
            // those projections).
            static float expand_pad[fixture::kHidden * (fixture::kChannels + 2)];
            for (int r = 0; r < fixture::kHidden; ++r) {
                for (int c = 0; c < fixture::kChannels; ++c) {
                    expand_pad[r * (fixture::kChannels + 2) + c]
                        = f.expand[r * fixture::kChannels + c];
                }
                expand_pad[r * (fixture::kChannels + 2) + fixture::kChannels]
                    = 2.0f;
                expand_pad[r * (fixture::kChannels + 2) + fixture::kChannels
                    + 1] = -2.0f;
            }
            static float qkv_pad[3 * fixture::kChannels
                * (fixture::kChannels + 2)];
            for (int r = 0; r < 3 * fixture::kChannels; ++r) {
                for (int c = 0; c < fixture::kChannels; ++c) {
                    qkv_pad[r * (fixture::kChannels + 2) + c]
                        = f.qkv[r * fixture::kChannels + c];
                }
                qkv_pad[r * (fixture::kChannels + 2) + fixture::kChannels]
                    = 2.0f;
                qkv_pad[r * (fixture::kChannels + 2) + fixture::kChannels
                    + 1] = -2.0f;
            }
            SwinBlockParams padded = f.params;
            padded.expand = make_view(expand_pad, fixture::kHidden,
                fixture::kChannels, fixture::kChannels + 2);
            padded.qkv = make_view(qkv_pad, 3 * fixture::kChannels,
                fixture::kChannels, fixture::kChannels + 2);
            const SwinBlockParams& run_params =
                pred == 7 ? padded : f.params;
            switch (pred) {
                case 0: site.batch_work = nullptr; break;
                case 1: site.batch_capacity = batch_need - 1; break;
                case 2: site.input_capacity = input_need - 1; break;
                case 3: site.products_capacity = product_need - 1; break;
                case 4: site.ex = nullptr; break;
                case 5: {
                    static dlss5_nr::ProjectionExecutor no_f32;
                    no_f32 = dlss5_nr::ProjectionExecutor{};
                    site.ex = &no_f32;
                    break;
                }
                case 6: site.input_dense = nullptr; break;
                default: break;
            }
            fill_sentinel(wide_try, 1 << 18);
            const Status status = swin_block(x_v, nullptr, 0, 0, run_params,
                wide_try, map_count, nullptr, 0, scratch, need, &site);
            if (pred == 4 || pred == 6) {
                // A set site with a null executor or a null gather buffer is
                // INVALID (the G4 site ordering, identical in both paths),
                // not a fallback: no output writes at all.
                CHECK(status == Status::kInvalidBuffer);
                CHECK(all_sentinel(wide_try, 1 << 18));
                CHECK(engine.calls == 0);
                continue;
            }
            CHECK(status == Status::kOk);
            for (std::size_t p = 0; p < map_count; ++p) {
                CHECK(memcmp(&wide_ref[p], &wide_try[p], sizeof(float)) == 0);
            }
            if (pred < 4) {
                // The per-window seam: 33 windows * two 64-token calls.
                CHECK(!engine.overflow);
                CHECK(engine.calls == 2 * 33);
                CHECK(engine.tokens[0] == dlss5_nr::kWindowTokens);
            } else {
                // matmul_f32 == null, or a padded weight: the CPU kernels,
                // no executor calls (the G4 fallbacks).
                CHECK(engine.calls == 0);
            }
        }
    }
    // Sentinel-checked rejection with a full batched site: the block's
    // validation contract is unchanged by the seam (no write at all).
    {
        fixture::BlockFixture f;
        f.zero_init();
        const int channels = fixture::kChannels;
        static float x[1 << 18];
        static float wide_try[1 << 18];
        const std::size_t map_count = std::size_t{channels} * 8 * 16;
        for (std::size_t p = 0; p < map_count; ++p) {
            x[p] = 1.0f;
        }
        x[map_count / 2] = std::numeric_limits<float>::infinity();
        const FeatureBatchView x_v = make_batch(x, channels, 8, 16,
            static_cast<std::ptrdiff_t>(8) * 16, 16);
        const std::size_t need = swin_block_scratch_needed(x_v, false, true, 0,
            0, f.params);
        CountingExecutor engine;
        SwinProjectionSite site;
        BatchSite batch;
        batch.build(site, engine, static_cast<int>(
            BatchSite::kMaxChunkRows), channels, fixture::kHidden, 1);
        fill_sentinel(wide_try, 1 << 18);
        fill_sentinel(scratch, 1 << 21);
        CHECK(swin_block(x_v, nullptr, 0, 0, f.params, wide_try, map_count,
            nullptr, 0, scratch, need, &site) == Status::kInvalidValues);
        CHECK(all_sentinel(wide_try, 1 << 18));
        CHECK(all_sentinel(scratch, 1 << 21));
        CHECK(engine.calls == 0);
    }
    // swin_layer_plain threads the site like the other layer entries: the
    // chunked run equals the CPU run bit for bit (and the default keeps the
    // CPU kernels).
    {
        fixture::BlockFixture f;
        f.zero_init();
        fill_random_weights(&f);
        const int channels = fixture::kChannels;
        static float x[fixture::kChannels * 8 * 264];
        static float out_a[fixture::kChannels * 8 * 264];
        static float out_b[fixture::kChannels * 8 * 264];
        const std::size_t map_count = std::size_t{channels} * 8 * 264;
        for (std::size_t p = 0; p < map_count; ++p) {
            x[p] = rnd_f16();
        }
        const FeatureBatchView x_v = make_batch(x, channels, 8, 264,
            static_cast<std::ptrdiff_t>(8) * 264, 264);
        const std::size_t need = swin_block_scratch_needed(x_v, false, true, 0,
            0, f.params);
        CHECK(swin_layer_plain(x_v, 0, 0, f.params, out_a, map_count, scratch,
            need) == Status::kOk);
        CountingExecutor engine;
        SwinProjectionSite site;
        BatchSite batch;
        batch.build(site, engine, static_cast<int>(
            BatchSite::kMaxChunkRows), channels, fixture::kHidden, 1);
        CHECK(swin_layer_plain(x_v, 0, 0, f.params, out_b, map_count, scratch,
            need, &site) == Status::kOk);
        for (std::size_t p = 0; p < map_count; ++p) {
            CHECK(memcmp(&out_a[p], &out_b[p], sizeof(float)) == 0);
        }
        CHECK(engine.calls == 2 * 2);  // 33 windows: 32 + 1 chunks
    }
    // Determinism and zero-allocation accounting across the chunked path.
    {
        fixture::BlockFixture f;
        f.zero_init();
        fill_random_weights(&f);
        const int channels = fixture::kChannels;
        static float x[fixture::kChannels * 8 * 360];
        static float xw[fixture::kChannels * 8 * 360];
        static float wide_a[fixture::kChannels * 8 * 360];
        static float wide_b[fixture::kChannels * 8 * 360];
        static float acc_a[fixture::kChannels * 8 * 360];
        static float acc_b[fixture::kChannels * 8 * 360];
        const std::size_t map_count = std::size_t{channels} * 8 * 360;
        for (std::size_t p = 0; p < map_count; ++p) {
            x[p] = rnd_f16();
            xw[p] = rnd_f16();
        }
        const FeatureBatchView x_v = make_batch(x, channels, 8, 360,
            static_cast<std::ptrdiff_t>(8) * 360, 360);
        const FeatureBatchView xw_v = make_batch(xw, channels, 8, 360,
            static_cast<std::ptrdiff_t>(8) * 360, 360);
        const std::size_t need = swin_block_scratch_needed(x_v, true, true, 0,
            0, f.params);
        CountingExecutor engine;
        SwinProjectionSite site;
        BatchSite batch;
        batch.build(site, engine, static_cast<int>(
            BatchSite::kMaxChunkRows), channels, fixture::kHidden, 1);
        const unsigned long long before = g_allocations.load();
        CHECK(swin_block(x_v, &xw_v, 0, 0, f.params, wide_a, map_count, acc_a,
            map_count, scratch, need, &site) == Status::kOk);
        CHECK(swin_block(x_v, &xw_v, 0, 0, f.params, wide_b, map_count, acc_b,
            map_count, scratch, need, &site) == Status::kOk);
        CHECK(g_allocations.load() == before);
        for (std::size_t p = 0; p < map_count; ++p) {
            CHECK(memcmp(&wide_a[p], &wide_b[p], sizeof(float)) == 0);
            CHECK(memcmp(&acc_a[p], &acc_b[p], sizeof(float)) == 0);
        }
    }
}

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
    check_batched_seam();
    check_determinism_and_allocations();
    printf("native-swin-block-test: %lld checks passed\n", g_checks);
    return 0;
}
