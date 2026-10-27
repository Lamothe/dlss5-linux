// Independent tests for the native network graph plan slice
// (vr/native-graph-plan.hpp): the typed layer table (152 layers / 71 blocks for
// "ht", 76 layers / 31 blocks for "small"), the extent walk, frame padding and
// skip wiring. CPU-only, no GPU, headset, proprietary payload, Python or
// PyTorch runtime required.
//
// Independence: every known answer is hand-derived from the pinned
// construction semantics (the derivations are written next to each case), and
// the whole-plan sweeps re-derive each row's contract from the documented
// rules rather than from the header's tables. Controls cover: the structural
// counts and per-family tables; the padding table ((1,1), (33,33), (321,33),
// (321,241), (1920,1080), (1921,1081), (16384,16384) for both networks,
// including the "small" wide-alignment discriminator 1080 -> 1152); the
// halving counters ((1,1) -> 0,0; (320,240) -> 6,6; (1920,1080) -> 6,6;
// (1921,1081) -> 6,6; (321,33) -> 6,4); the ht and small row anchors (kinds,
// variants, extents, tokens, channels, heads, shifts, grid, input/skip wiring
// and names); the
// skip-source rules for every row of both networks; the grid/shift, token and
// input-block formulas; the extent-walk coherence of every row against its
// predecessor and its skip source; sentinel-checked untouched outputs on
// every validation rejection class; determinism and zero-allocation
// accounting. See vr/NATIVE-GRAPH-PLAN.md.

#include "native-graph-plan.hpp"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>
#include <cstdlib>
#include <initializer_list>
#include <new>

// Deliberate byte-level fill for the sentinel untouched-output checks; routed
// through a void* helper so the byte intent is explicit at every call site.
static void fill_bytes(void* p, int value, size_t bytes) {
    unsigned char* out = static_cast<unsigned char*>(p);
    for (size_t i = 0; i < bytes; ++i) {
        out[i] = static_cast<unsigned char>(value);
    }
}

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
void* operator new(size_t n, std::align_val_t a) {
    g_allocations.fetch_add(1);
    const size_t alignment = static_cast<size_t>(a);
    const size_t rounded = (n + alignment - 1) / alignment * alignment;
    void* p = std::aligned_alloc(alignment, rounded == 0 ? alignment : rounded);
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

// Self-contained check: the assertions carry side effects (the calls under
// test), so they must never compile away.
static long long g_checks = 0;
#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        abort(); \
    } \
    ++g_checks; \
} while (0)

using dlss5_nr::LayerKind;
using dlss5_nr::LayerSpec;
using dlss5_nr::LayerVariant;
using dlss5_nr::NetworkKind;
using dlss5_nr::PlanPadding;
using dlss5_nr::Status;

static constexpr NetworkKind kHt = NetworkKind::kHt;
static constexpr NetworkKind kSmall = NetworkKind::kSmall;

// ---- independent reference arithmetic -------------------------------------------------------------
// Round n (>= 0) up to a multiple of k; the halving rule of the extent walk.
static int ref_up(int n, int k) {
    return (n + k - 1) / k * k;
}
static int ref_halve(int n) {
    return ref_up((n + 1) / 2, 4);
}

// The row-role mapping a consumer derives from (kind, variant): the pre block,
// the encoder downsample (a "ds" Swin), the final head, the decoder entry, a
// Swin upsample (an "upsample" Swin), the post block, and plain rows.
enum RoleTag {
    kRolePlain = 0,
    kRolePre,
    kRolePost,
    kRoleDown,
    kRoleSwinUp,
    kRoleDecoderUp,
    kRoleFinalHead,
};

static bool is_swin_kind(LayerKind kind) {
    return kind == LayerKind::kSwin1H || kind == LayerKind::kSwin2H
        || kind == LayerKind::kSwin4H || kind == LayerKind::kSwin8H;
}

static bool is_vit_kind(LayerKind kind) {
    return kind == LayerKind::kVitFfnExpand || kind == LayerKind::kVitFfnContract
        || kind == LayerKind::kVitQKV || kind == LayerKind::kVitAttention
        || kind == LayerKind::kVitProjection;
}

static RoleTag role_of(const LayerSpec& s) {
    if (s.kind == LayerKind::kPreBlock) {
        return kRolePre;
    }
    if (s.kind == LayerKind::kPostBlock) {
        return kRolePost;
    }
    if (s.kind == LayerKind::kSplitFinalHead) {
        return kRoleFinalHead;
    }
    if (s.kind == LayerKind::kDecInputUpsample) {
        return kRoleDecoderUp;
    }
    if (is_swin_kind(s.kind) && s.variant == LayerVariant::kDs) {
        return kRoleDown;
    }
    if (is_swin_kind(s.kind) && s.variant == LayerVariant::kUpsample) {
        return kRoleSwinUp;
    }
    return kRolePlain;
}

// Find the row at (block, layer) with an independent scan.
static int find_row(const LayerSpec* specs, int count, int block, int layer) {
    for (int i = 0; i < count; ++i) {
        if (specs[i].block == block && specs[i].layer == layer) {
            return i;
        }
    }
    return -1;
}

// ---- structural counts ------------------------------------------------------------------------------

static void check_structural_counts() {
    using dlss5_nr::plan_block_count;
    using dlss5_nr::plan_detail::build_rows;
    using dlss5_nr::plan_detail::PlanRow;
    using dlss5_nr::plan_layer_count;
    using dlss5_nr::kPlanMaxLayers;
    using dlss5_nr::kPlanMaxBlocks;

    CHECK(plan_layer_count(kHt) == 152);
    CHECK(plan_block_count(kHt) == 71);
    CHECK(plan_layer_count(kSmall) == 76);
    CHECK(plan_block_count(kSmall) == 31);
    CHECK(kPlanMaxLayers == 152);
    CHECK(kPlanMaxBlocks == 71);

    // The pinned small docstring's "77" counts weight-file records, not
    // layers: the plan is 76 layers in 31 blocks.
    CHECK(plan_layer_count(kSmall) != 77);

    for (NetworkKind network : {kHt, kSmall}) {
        PlanRow rows[kPlanMaxLayers];
        fill_bytes(rows, 0, sizeof rows);
        const int count = build_rows(network, rows);
        CHECK(count == plan_layer_count(network));
        // Blocks and layers are dense: block b holds layers 0..k-1 in order and
        // every (block, layer) position exists exactly once.
        int block_max = -1;
        for (int i = 0; i < count; ++i) {
            CHECK(rows[i].block >= 0);
            CHECK(rows[i].layer >= 0);
            CHECK(rows[i].name != nullptr);
            if (rows[i].block > block_max) {
                block_max = rows[i].block;
            }
            if (i > 0) {
                CHECK(rows[i].block >= rows[i - 1].block);
                if (rows[i].block == rows[i - 1].block) {
                    CHECK(rows[i].layer == rows[i - 1].layer + 1);
                } else {
                    CHECK(rows[i].layer == 0);
                }
            } else {
                CHECK(rows[i].layer == 0);
            }
        }
        CHECK(block_max == plan_block_count(network) - 1);
        for (int b = 0; b <= block_max; ++b) {
            int layers = 0;
            for (int i = 0; i < count; ++i) {
                layers += rows[i].block == b ? 1 : 0;
            }
            CHECK(layers == 4 || layers == 5 || layers == 1);
        }
    }

    // Per-family layer counts, hand-counted from the construction rules:
    // ht: 1 pre + (4+4) Swin1H + (4+4) Swin2H + (6+6) Swin4H + (8+8) Swin8H
    //     + (8+8) split ffwd/ffwd-proj/qkv + (7+8) split proj + pool + final
    //     head + 8 x 5 vit + decoder entry + post = 152. small replaces each
    //     per-level count by 2 and each split section by 2 blocks = 76.
    static const int kExpectedKinds[2][18] = {
        // ht
        {1, 8, 8, 12, 16, 16, 16, 16, 15, 1, 1, 8, 8, 8, 8, 8, 1, 1},
        // small
        {1, 4, 4, 4, 4, 4, 4, 4, 3, 1, 1, 8, 8, 8, 8, 8, 1, 1},
    };
    for (int n = 0; n < 2; ++n) {
        const NetworkKind network = n == 0 ? kHt : kSmall;
        LayerSpec specs[dlss5_nr::kPlanMaxLayers];
        int count = 0;
        CHECK(dlss5_nr::build_plan(320, 240, network, specs,
                                   dlss5_nr::kPlanMaxLayers, count) == Status::kOk);
        CHECK(count == plan_layer_count(network));
        for (int k = 0; k < 18; ++k) {
            int seen = 0;
            for (int i = 0; i < count; ++i) {
                seen += specs[i].kind == static_cast<LayerKind>(k) ? 1 : 0;
            }
            CHECK(seen == kExpectedKinds[n][k]);
        }
    }

    // Name tables are total and distinct per value.
    for (int k = 0; k < 18; ++k) {
        const char* name = dlss5_nr::layer_kind_name(static_cast<LayerKind>(k));
        CHECK(name != nullptr);
        CHECK(name[0] != '\0');
        for (int other = 0; other < k; ++other) {
            CHECK(strcmp(name, dlss5_nr::layer_kind_name(
                           static_cast<LayerKind>(other))) != 0);
        }
    }
    for (int v = 0; v < 7; ++v) {
        const char* name = dlss5_nr::layer_variant_name(static_cast<LayerVariant>(v));
        CHECK(name != nullptr);
    }
    CHECK(strcmp(dlss5_nr::layer_variant_name(LayerVariant::kNone), "") == 0);
    CHECK(strcmp(dlss5_nr::layer_variant_name(LayerVariant::kInpview), "inpview") == 0);
    CHECK(strcmp(dlss5_nr::layer_variant_name(LayerVariant::kDs), "ds") == 0);
    CHECK(strcmp(dlss5_nr::layer_variant_name(LayerVariant::kUpsample), "upsample") == 0);
    CHECK(strcmp(dlss5_nr::layer_variant_name(LayerVariant::kOutview), "outview") == 0);
    CHECK(strcmp(dlss5_nr::layer_variant_name(LayerVariant::kProjPool), "proj_pool") == 0);
    CHECK(strcmp(dlss5_nr::layer_variant_name(LayerVariant::kFinalHead), "final_head") == 0);
}

// ---- padding and halvings --------------------------------------------------------------------------
// Known answers, hand-derived from the walk (each halving stage needs its axis
// divisible by two, floor 320, one extra alignment unit when both axes already
// fill full 4-cell rows; "small" doubles the alignment where the pre block's
// output along an axis is 4 mod 8):
//   (1,1): halvings 0,0 -> align 1; floor 320x320; both axes fill -> +1 on w
//          = 321x320 (ht); small doubles both aligns -> 322x320.
//   (33,33): halvings 4,4 -> align 16 -> 336x320 after the +16 (ht);
//          small doubles to 32 -> no bump = 320x320.
//   (321,33): halvings 6,4 -> align 64/16 -> 384x320, no bump (384 % 256).
//   (321,241): halvings 6,6 -> align 64 -> 384x320 (241 -> 320 floor).
//   (1920,1080): halvings 6,6 -> 1920x1088 (1080 -> 1088), no bump; small's
//          rows axis is 4 mod 8 at the pre output -> align 128 -> 1152
//          (the pinned docstring's "1080 pads to 1152, not 1088").
//   (1921,1081): halvings 6,6 -> ht 1984x1088; small doubles the columns
//          axis (pre output 964 = 4 mod 8) -> 2048x1088.
//   (16384,16384): halvings 6,6 -> 16448x16384 (the +64 only).

static void check_padding_table() {
    using dlss5_nr::plan_padding;
    using dlss5_nr::plan_working_extent;

    struct Case {
        int frame_w, frame_h;
        NetworkKind network;
        int width, height, pad_x, pad_y;
    };
    static const Case kCases[] = {
        {1, 1, kHt, 321, 320, 320, 319},
        {1, 1, kSmall, 322, 320, 321, 319},
        {33, 33, kHt, 336, 320, 303, 287},
        {33, 33, kSmall, 320, 320, 287, 287},
        {321, 33, kHt, 384, 320, 63, 287},
        {321, 33, kSmall, 384, 320, 63, 287},
        {321, 241, kHt, 384, 320, 63, 79},
        {321, 241, kSmall, 384, 320, 63, 79},
        {1920, 1080, kHt, 1920, 1088, 0, 8},
        {1920, 1080, kSmall, 1920, 1152, 0, 72},
        {1921, 1081, kHt, 1984, 1088, 63, 7},
        {1921, 1081, kSmall, 2048, 1088, 127, 7},
        {16384, 16384, kHt, 16448, 16384, 64, 0},
        {16384, 16384, kSmall, 16448, 16384, 64, 0},
    };
    for (const Case& c : kCases) {
        PlanPadding out;
        fill_bytes(&out, 0xAB, sizeof out);
        CHECK(plan_padding(c.frame_w, c.frame_h, c.network, out) == Status::kOk);
        CHECK(out.width == c.width);
        CHECK(out.height == c.height);
        CHECK(out.pad_x == c.pad_x);
        CHECK(out.pad_y == c.pad_y);
        CHECK(out.pad_x == out.width - c.frame_w);
        CHECK(out.pad_y == out.height - c.frame_h);
        CHECK(out.width >= dlss5_nr::kPlanFloorExtent);
        CHECK(out.height >= dlss5_nr::kPlanFloorExtent);
        int w = -1;
        int h = -1;
        CHECK(plan_working_extent(c.frame_w, c.frame_h, c.network, w, h) == Status::kOk);
        CHECK(w == c.width);
        CHECK(h == c.height);
    }
}

static void check_halvings() {
    using dlss5_nr::plan_halvings;
    struct Case {
        int w, h;
        int halvings_x, halvings_y;
    };
    static const Case kCases[] = {
        {1, 1, 0, 0},
        {320, 240, 6, 6},
        {1920, 1080, 6, 6},
        {1921, 1081, 6, 6},
        {321, 33, 6, 4},
    };
    for (NetworkKind network : {kHt, kSmall}) {
        for (const Case& c : kCases) {
            int x = -1;
            int y = -1;
            CHECK(plan_halvings(c.w, c.h, network, x, y) == Status::kOk);
            CHECK(x == c.halvings_x);
            CHECK(y == c.halvings_y);
        }
    }
}

// ---- row anchors -----------------------------------------------------------------------------------

struct Anchor {
    int index;
    int block, layer;
    LayerKind kind;
    LayerVariant variant;
    int w, h, tokens;
    int channels;
    int out_w, out_h;
    int c_in, c_out, heads;
    int shift_x, shift_y;
    int shifted;
    int input_block, skip_block;
    int grid_x, grid_y;
    int skip_src;
    const char* name;
};

static void check_anchor(const LayerSpec* specs, int count, const Anchor& a) {
    CHECK(a.index >= 0 && a.index < count);
    const LayerSpec& s = specs[a.index];
    CHECK(s.index == a.index);
    CHECK(s.block == a.block);
    CHECK(s.layer == a.layer);
    CHECK(s.kind == a.kind);
    CHECK(s.variant == a.variant);
    CHECK(s.w == a.w);
    CHECK(s.h == a.h);
    CHECK(s.tokens == a.tokens);
    CHECK(s.channels == a.channels);
    CHECK(s.out_w == a.out_w);
    CHECK(s.out_h == a.out_h);
    CHECK(s.c_in == a.c_in);
    CHECK(s.c_out == a.c_out);
    CHECK(s.heads == a.heads);
    CHECK(s.shift_x == a.shift_x);
    CHECK(s.shift_y == a.shift_y);
    CHECK((s.shifted ? 1 : 0) == a.shifted);
    CHECK(s.input_block == a.input_block);
    CHECK(s.skip_block == a.skip_block);
    CHECK(s.grid_x == a.grid_x);
    CHECK(s.grid_y == a.grid_y);
    CHECK(dlss5_nr::plan_skip_source(specs, count, a.index) == a.skip_src);
    CHECK(s.name != nullptr && strcmp(s.name, a.name) == 0);
}

// ht at frame 320x240 -> padded 320x320. The walk on (320,320) halves
// 320 -> 160 (pre) -> 80 -> 40 -> 20 -> 12 (encoder downsamples) -> 8 (final
// head); the decoder entry takes the pool block's projection extent (12), the
// Swin upsamples take their skip's pre-downsample extent (20/40/80/160, the
// last one cell-aligned to 160), and the post block returns to 320x320.
// Shifts come from the cycle ((0,0), (-1,-1), (-1,0), (0,-1)) at each block's
// position in its section; grids from (grid - 4*shift + 7) / 8 over the plan
// extent (out extent for upsample/post rows).
static void check_rows_ht() {
    static const Anchor kAnchors[] = {
        {0, 0, 0, LayerKind::kPreBlock, LayerVariant::kDs, 320, 320, 102400,
         3, 160, 160, 3, 32, 0, 0, 0, 0, -1, -1, 40, 40, -1,
         "cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8"},
        {1, 1, 0, LayerKind::kSwin1H, LayerVariant::kInpview, 160, 160, 25600,
         32, 160, 160, 32, 32, 1, 0, 0, 0, 0, -1, 20, 20, -1,
         "cc_tinlayout_fused_swin_1h_32_1_inpview_fp8"},
        {4, 4, 0, LayerKind::kSwin1H, LayerVariant::kDs, 160, 160, 25600,
         32, 80, 80, 32, 64, 1, 0, -1, 1, 3, -1, 20, 21, -1,
         "cc_tinlayout_fused_swin_1h_32_1_ds_fp8"},
        {22, 22, 0, LayerKind::kSwin8H, LayerVariant::kDs, 20, 20, 576,
         256, 12, 12, 256, 512, 8, 0, -1, 1, 21, -1, 3, 3, -1,
         "cc_tinlayout_fused_swin_8h_256_8_ds_fp8"},
        {23, 23, 0, LayerKind::kSplitFfwd, LayerVariant::kInpview, 12, 12, 256,
         512, 12, 12, 512, 512, 4, 0, 0, 0, 22, -1, 2, 2, -1,
         "cc_split_swin_16h_ffwd_inpview_512_fp8"},
        {26, 23, 3, LayerKind::kSplitProj, LayerVariant::kNone, 12, 12, 256,
         512, 12, 12, 512, 512, 8, 0, 0, 0, 23, -1, 2, 2, 24,
         "cc_split_swin_16h_proj_512_fp8"},
        {51, 30, 0, LayerKind::kSplitFfwd, LayerVariant::kNone, 12, 12, 256,
         512, 12, 12, 512, 512, 4, 0, 0, 0, 29, -1, 2, 2, -1,
         "cc_split_swin_16h_ffwd_512_fp8"},
        {54, 30, 3, LayerKind::kSplitProjPool, LayerVariant::kProjPool,
         12, 12, 256, 512, 12, 12, 512, 512, 4, 0, 0, 0, 30, -1, 2, 2, 52,
         "cc_split_swin_16h_proj_pool_512_fp8"},
        {55, 30, 4, LayerKind::kSplitFinalHead, LayerVariant::kFinalHead,
         8, 8, 64, 512, 8, 8, 512, 1024, 8, 0, 0, 0, 30, -1, 1, 1, 54,
         "cc_split_swin_16h_final_head_512_fp8"},
        {56, 31, 0, LayerKind::kVitFfnExpand, LayerVariant::kNone,
         8, 8, 64, 1024, 8, 8, 1024, 4096, 4, 0, 0, 0, 30, -1, 1, 1, -1,
         "cc_vit_1d_ffn_expand_fp8"},
        {57, 31, 1, LayerKind::kVitFfnContract, LayerVariant::kNone,
         8, 8, 64, 1024, 8, 8, 4096, 1024, 4, 0, 0, 0, 31, -1, 1, 1, 55,
         "cc_vit_1d_ffn_contract_fp8"},
        {58, 31, 2, LayerKind::kVitQKV, LayerVariant::kNone,
         8, 8, 64, 1024, 8, 8, 1024, 1024, 4, 0, 0, 0, 31, -1, 1, 1, -1,
         "cc_vit_1d_qkv_fp8"},
        {59, 31, 3, LayerKind::kVitAttention, LayerVariant::kNone,
         8, 8, 64, 1024, 8, 8, 1024, 1024, 4, 0, 0, 0, 31, -1, 1, 1, -1,
         "cc_vit_1d_attention_fp8"},
        {60, 31, 4, LayerKind::kVitProjection, LayerVariant::kNone,
         8, 8, 64, 1024, 8, 8, 1024, 1024, 4, 0, 0, 0, 31, -1, 1, 1, 57,
         "cc_vit_1d_projection_fp8"},
        {96, 39, 0, LayerKind::kDecInputUpsample, LayerVariant::kUpsample,
         12, 12, 256, 512, 12, 12, 1024, 512, 1, 0, 0, 0, 38, 30, 2, 2, 54,
         "cc_dec_input_upsample_1024_512_fp8"},
        {128, 47, 3, LayerKind::kSplitProj, LayerVariant::kOutview,
         12, 12, 256, 512, 12, 12, 512, 512, 4, 0, 0, 0, 47, -1, 2, 2, 126,
         "cc_split_swin_16h_proj_512_outview_fp8"},
        {129, 48, 0, LayerKind::kSwin8H, LayerVariant::kUpsample,
         12, 12, 256, 256, 20, 20, 512, 256, 8, 0, 0, 0, 47, 22, 3, 3, 22,
         "cc_tinlayout_fused_swin_8h_256_8_upsample_fp8"},
        {137, 56, 0, LayerKind::kSwin4H, LayerVariant::kUpsample,
         20, 20, 576, 128, 40, 40, 256, 128, 4, -1, 0, 1, 55, 14, 6, 5, 14,
         "cc_tinlayout_fused_swin_4h_128_4_upsample_fp8"},
        {143, 62, 0, LayerKind::kSwin2H, LayerVariant::kUpsample,
         40, 40, 1600, 64, 80, 80, 128, 64, 2, 0, 0, 0, 61, 8, 10, 10, 8,
         "cc_tinlayout_fused_swin_2h_64_2_upsample_fp8"},
        {147, 66, 0, LayerKind::kSwin1H, LayerVariant::kUpsample,
         80, 80, 6400, 32, 160, 160, 64, 32, 1, 0, 0, 0, 65, 4, 20, 20, 4,
         "cc_tinlayout_fused_swin_1h_32_1_upsample_fp8"},
        {151, 70, 0, LayerKind::kPostBlock, LayerVariant::kNone,
         160, 160, 25600, 3, 320, 320, 32, 3, 0, -1, -1, 0, 69, 0, 41, 41, 0,
         "cc_tinlayout_fused_post_block_swin_1h_32_fp8"},
    };
    LayerSpec specs[dlss5_nr::kPlanMaxLayers];
    int count = 0;
    CHECK(dlss5_nr::build_plan(320, 240, kHt, specs,
                               dlss5_nr::kPlanMaxLayers, count) == Status::kOk);
    CHECK(count == 152);
    for (const Anchor& a : kAnchors) {
        check_anchor(specs, count, a);
    }
}

// small at frame 320x240 -> padded 320x320: the same walk with one block per
// encoder/decoder level and two split blocks per section; the decoder levels
// start the shift cycle at position 2, so their upsamples shift (-1,0).
static void check_rows_small() {
    static const Anchor kAnchors[] = {
        {0, 0, 0, LayerKind::kPreBlock, LayerVariant::kDs, 320, 320, 102400,
         3, 160, 160, 3, 32, 0, 0, 0, 0, -1, -1, 40, 40, -1,
         "cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8"},
        {2, 2, 0, LayerKind::kSwin1H, LayerVariant::kDs, 160, 160, 25600,
         32, 80, 80, 32, 64, 1, -1, -1, 1, 1, -1, 21, 21, -1,
         "cc_tinlayout_fused_swin_1h_32_1_ds_fp8"},
        {8, 8, 0, LayerKind::kSwin8H, LayerVariant::kDs, 20, 20, 576,
         256, 12, 12, 256, 512, 8, -1, -1, 1, 7, -1, 3, 3, -1,
         "cc_tinlayout_fused_swin_8h_256_8_ds_fp8"},
        {13, 10, 0, LayerKind::kSplitFfwd, LayerVariant::kNone, 12, 12, 256,
         512, 12, 12, 512, 512, 4, 0, 0, 0, 9, -1, 2, 2, -1,
         "cc_split_swin_16h_ffwd_512_fp8"},
        {15, 10, 2, LayerKind::kSplitQKVAttn, LayerVariant::kNone, 12, 12, 256,
         512, 12, 12, 512, 512, 4, -1, -1, 1, 10, -1, 2, 2, -1,
         "cc_split_swin_16h_qkv_512_fp8"},
        {16, 10, 3, LayerKind::kSplitProjPool, LayerVariant::kProjPool,
         12, 12, 256, 512, 12, 12, 512, 512, 4, 0, 0, 0, 10, -1, 2, 2, 14,
         "cc_split_swin_16h_proj_pool_512_fp8"},
        {17, 10, 4, LayerKind::kSplitFinalHead, LayerVariant::kFinalHead,
         8, 8, 64, 512, 8, 8, 512, 1024, 8, 0, 0, 0, 10, -1, 1, 1, 16,
         "cc_split_swin_16h_final_head_512_fp8"},
        {18, 11, 0, LayerKind::kVitFfnExpand, LayerVariant::kNone,
         8, 8, 64, 1024, 8, 8, 1024, 4096, 4, 0, 0, 0, 10, -1, 1, 1, -1,
         "cc_vit_1d_ffn_expand_fp8"},
        {19, 11, 1, LayerKind::kVitFfnContract, LayerVariant::kNone,
         8, 8, 64, 1024, 8, 8, 4096, 1024, 4, 0, 0, 0, 11, -1, 1, 1, 17,
         "cc_vit_1d_ffn_contract_fp8"},
        {22, 11, 4, LayerKind::kVitProjection, LayerVariant::kNone,
         8, 8, 64, 1024, 8, 8, 1024, 1024, 4, 0, 0, 0, 11, -1, 1, 1, 19,
         "cc_vit_1d_projection_fp8"},
        {58, 19, 0, LayerKind::kDecInputUpsample, LayerVariant::kUpsample,
         12, 12, 256, 512, 12, 12, 1024, 512, 1, 0, 0, 0, 18, 10, 2, 2, 16,
         "cc_dec_input_upsample_1024_512_fp8"},
        {67, 22, 0, LayerKind::kSwin8H, LayerVariant::kUpsample,
         12, 12, 256, 256, 20, 20, 512, 256, 8, -1, 0, 1, 21, 8, 3, 3, 8,
         "cc_tinlayout_fused_swin_8h_256_8_upsample_fp8"},
        {69, 24, 0, LayerKind::kSwin4H, LayerVariant::kUpsample,
         20, 20, 576, 128, 40, 40, 256, 128, 4, -1, 0, 1, 23, 6, 6, 5, 6,
         "cc_tinlayout_fused_swin_4h_128_4_upsample_fp8"},
        {71, 26, 0, LayerKind::kSwin2H, LayerVariant::kUpsample,
         40, 40, 1600, 64, 80, 80, 128, 64, 2, -1, 0, 1, 25, 4, 11, 10, 4,
         "cc_tinlayout_fused_swin_2h_64_2_upsample_fp8"},
        {73, 28, 0, LayerKind::kSwin1H, LayerVariant::kUpsample,
         80, 80, 6400, 32, 160, 160, 64, 32, 1, -1, 0, 1, 27, 2, 21, 20, 2,
         "cc_tinlayout_fused_swin_1h_32_1_upsample_fp8"},
        {75, 30, 0, LayerKind::kPostBlock, LayerVariant::kNone,
         160, 160, 25600, 3, 320, 320, 32, 3, 0, -1, -1, 0, 29, 0, 41, 41, 0,
         "cc_tinlayout_fused_post_block_swin_1h_32_fp8"},
    };
    LayerSpec specs[dlss5_nr::kPlanMaxLayers];
    int count = 0;
    CHECK(dlss5_nr::build_plan(320, 240, kSmall, specs,
                               dlss5_nr::kPlanMaxLayers, count) == Status::kOk);
    CHECK(count == 76);
    for (const Anchor& a : kAnchors) {
        check_anchor(specs, count, a);
    }
}

// ---- skip wiring -----------------------------------------------------------------------------------
// The skip-source rules, applied to every row of both networks: a split
// projection reads its block's feed-forward projection (the final head the
// pooled copy), a vit contract reads the previous block's projection and a vit
// projection its own contract, the decoder entry the pool block's projection,
// a Swin upsample its skip block's downsample, the post block the pre block,
// and nothing else has a skip input.
static void check_skip_wiring() {
    for (NetworkKind network : {kHt, kSmall}) {
        LayerSpec specs[dlss5_nr::kPlanMaxLayers];
        int count = 0;
        CHECK(dlss5_nr::build_plan(320, 240, network, specs,
                                   dlss5_nr::kPlanMaxLayers, count) == Status::kOk);
        for (int i = 0; i < count; ++i) {
            const LayerSpec& s = specs[i];
            int want_block = -2;
            int want_layer = -2;
            switch (s.kind) {
                case LayerKind::kSplitProj:
                case LayerKind::kSplitProjPool:
                    want_block = s.block;
                    want_layer = 1;
                    break;
                case LayerKind::kSplitFinalHead:
                    want_block = s.block;
                    want_layer = 3;
                    break;
                case LayerKind::kVitFfnContract:
                    want_block = s.block - 1;
                    want_layer = 4;
                    break;
                case LayerKind::kVitProjection:
                    want_block = s.block;
                    want_layer = 1;
                    break;
                case LayerKind::kDecInputUpsample:
                    want_block = s.skip_block;
                    want_layer = 3;
                    break;
                case LayerKind::kPostBlock:
                    want_block = 0;
                    want_layer = 0;
                    break;
                default:
                    if (is_swin_kind(s.kind) && s.variant == LayerVariant::kUpsample) {
                        want_block = s.skip_block;
                        want_layer = 0;
                    }
                    break;
            }
            const int src = dlss5_nr::plan_skip_source(specs, count, i);
            if (want_block == -2) {
                CHECK(src == -1);
            } else {
                CHECK(src == find_row(specs, count, want_block, want_layer));
                CHECK(src >= 0);
            }
        }
    }
    // Bounds rejections.
    LayerSpec specs[dlss5_nr::kPlanMaxLayers];
    int count = 0;
    CHECK(dlss5_nr::build_plan(320, 240, kHt, specs,
                               dlss5_nr::kPlanMaxLayers, count) == Status::kOk);
    CHECK(dlss5_nr::plan_skip_source(nullptr, count, 0) == -1);
    CHECK(dlss5_nr::plan_skip_source(specs, count, -1) == -1);
    CHECK(dlss5_nr::plan_skip_source(specs, count, count) == -1);
}

// ---- grid, tokens, input block ----------------------------------------------------------------------
// Per-row formulas over every row of both networks, re-derived here:
// tokens are the vit token count (w*h) or the cell-aligned grid area
// (up(w,8) * up(h,8)); the grid comes from the plan extent, from the output
// extent for upsample and post rows; shifted is the shift pair outside the
// post block; input_block is the previous block for a block's first layer.
static void check_grid_tokens_input_block() {
    for (NetworkKind network : {kHt, kSmall}) {
        LayerSpec specs[dlss5_nr::kPlanMaxLayers];
        int count = 0;
        CHECK(dlss5_nr::build_plan(320, 240, network, specs,
                                   dlss5_nr::kPlanMaxLayers, count) == Status::kOk);
        for (int i = 0; i < count; ++i) {
            const LayerSpec& s = specs[i];
            const RoleTag role = role_of(s);
            // Tokens.
            const int expected_tokens = is_vit_kind(s.kind)
                ? s.w * s.h
                : ref_up(s.w, 8) * ref_up(s.h, 8);
            CHECK(s.tokens == expected_tokens);
            // Grid (plan extent, output extent for upsample/post rows).
            const int grid_w = (role == kRolePost || role == kRoleSwinUp)
                ? s.out_w : s.w;
            const int grid_h = (role == kRolePost || role == kRoleSwinUp)
                ? s.out_h : s.h;
            CHECK(s.grid_x == (grid_h - 4 * s.shift_x + 7) / 8);
            CHECK(s.grid_y == (grid_w - 4 * s.shift_y + 7) / 8);
            // Channels: the body width per family (the split family's 512,
            // the vit family's 1024, the level channels for Swin rows - the
            // upsample row's c_out, every other Swin row's c_in - and 3 for
            // the pre/post blocks, 512 for the decoder entry).
            int expected_channels = 0;
            if (s.kind == LayerKind::kPreBlock || s.kind == LayerKind::kPostBlock) {
                expected_channels = 3;
            } else if (s.kind == LayerKind::kDecInputUpsample) {
                expected_channels = 512;
            } else if (is_swin_kind(s.kind)) {
                expected_channels = s.variant == LayerVariant::kUpsample ? s.c_out : s.c_in;
            } else if (is_vit_kind(s.kind)) {
                expected_channels = 1024;
            } else {
                expected_channels = 512;  // the split family
            }
            CHECK(s.channels == expected_channels);
            // Shifts and input wiring.
            CHECK((s.shifted ? 1 : 0)
                  == ((s.shift_x != 0 || s.shift_y != 0)
                      && s.kind != LayerKind::kPostBlock ? 1 : 0));
            CHECK(s.input_block == (s.layer == 0 ? s.block - 1 : s.block));
        }
    }
    // The shift-margin discriminator: a (-1) shift widens the window grid by
    // one step ((160 + 4 + 7) / 8 = 21, not 20), and the post row's (-1,-1)
    // gives 41 over the 320-wide output extent.
    LayerSpec specs[dlss5_nr::kPlanMaxLayers];
    int count = 0;
    CHECK(dlss5_nr::build_plan(320, 240, kHt, specs,
                               dlss5_nr::kPlanMaxLayers, count) == Status::kOk);
    CHECK(specs[4].grid_y == 21);
    CHECK(specs[151].grid_x == 41);
    CHECK(specs[151].grid_y == 41);
    CHECK(specs[55].tokens == 64);      // 8 x 8 final-head grid
    CHECK(specs[56].tokens == 64);      // vit rows take w * h
    CHECK(specs[96].tokens == 256);     // 12 x 12 cell-aligned
    CHECK(specs[0].tokens == 102400);   // 320 x 320 cell-aligned

    // The 16384 bound runs the full plan over the padded 16448x16384 extent;
    // the largest token product stays int-safe (16384 * 16448 = 269484032).
    for (NetworkKind network : {kHt, kSmall}) {
        LayerSpec big[dlss5_nr::kPlanMaxLayers];
        int big_count = 0;
        CHECK(dlss5_nr::build_plan(16384, 16384, network, big,
                                   dlss5_nr::kPlanMaxLayers, big_count) == Status::kOk);
        CHECK(big_count == dlss5_nr::plan_layer_count(network));
        CHECK(big[0].w == 16384 && big[0].h == 16448);
        CHECK(big[0].tokens == 269484032);
        CHECK(big[big_count - 1].out_w == 16384);
        CHECK(big[big_count - 1].out_h == 16448);
    }
}

// ---- extent-walk coherence --------------------------------------------------------------------------
// Every row's contract against its predecessor and its skip source, for both
// networks: the plain chain (this row's grid is the previous row's output),
// the halving rule of pre/down/final-head rows, the skip extents of the
// upsample rows (the encoder skip before its downsample, cell-aligned where
// the output has 32 channels; the pool block's output for the decoder entry),
// and the post row's return to the padded frame.
static void check_extent_coherence() {
    for (NetworkKind network : {kHt, kSmall}) {
        LayerSpec specs[dlss5_nr::kPlanMaxLayers];
        int count = 0;
        CHECK(dlss5_nr::build_plan(320, 240, network, specs,
                                   dlss5_nr::kPlanMaxLayers, count) == Status::kOk);
        PlanPadding padded;
        CHECK(dlss5_nr::plan_padding(320, 240, network, padded) == Status::kOk);
        for (int i = 0; i < count; ++i) {
            const LayerSpec& s = specs[i];
            const RoleTag role = role_of(s);
            if (i > 0 && role != kRoleFinalHead && role != kRoleDecoderUp
                && role != kRolePost) {
                CHECK(s.w == specs[i - 1].out_w);
                CHECK(s.h == specs[i - 1].out_h);
            }
            switch (role) {
                case kRolePre:
                case kRoleDown:
                    CHECK(s.out_w == ref_halve(s.w));
                    CHECK(s.out_h == ref_halve(s.h));
                    break;
                case kRoleFinalHead:
                    // The final head's plan field is its output extent; the
                    // halved input is the previous row's output.
                    CHECK(i > 0);
                    CHECK(s.out_w == ref_halve(specs[i - 1].out_w));
                    CHECK(s.out_h == ref_halve(specs[i - 1].out_h));
                    CHECK(s.w == s.out_w);
                    CHECK(s.h == s.out_h);
                    break;
                case kRoleSwinUp: {
                    const int src = dlss5_nr::plan_skip_source(specs, count, i);
                    CHECK(src >= 0);
                    if (s.c_out == 32) {
                        CHECK(s.out_w == ref_up(specs[src].w, 8));
                        CHECK(s.out_h == ref_up(specs[src].h, 8));
                    } else {
                        CHECK(s.out_w == specs[src].w);
                        CHECK(s.out_h == specs[src].h);
                    }
                    break;
                }
                case kRoleDecoderUp: {
                    const int src = dlss5_nr::plan_skip_source(specs, count, i);
                    CHECK(src >= 0);
                    CHECK(s.out_w == specs[src].out_w);
                    CHECK(s.out_h == specs[src].out_h);
                    CHECK(s.w == ref_up(s.out_w, 4));
                    CHECK(s.h == ref_up(s.out_h, 4));
                    break;
                }
                case kRolePost:
                    CHECK(s.out_w == padded.height);
                    CHECK(s.out_h == padded.width);
                    CHECK(s.w == padded.height / 2);
                    CHECK(s.h == padded.width / 2);
                    break;
                case kRolePlain:
                    CHECK(s.out_w == s.w);
                    CHECK(s.out_h == s.h);
                    break;
            }
        }
        // The walk closes: the last row's output is the padded frame.
        CHECK(specs[count - 1].out_w == padded.height);
        CHECK(specs[count - 1].out_h == padded.width);
    }
}

// ---- validation rejections ---------------------------------------------------------------------------
// Every rejection class leaves its outputs untouched (sentinel-checked).
static void check_rejections() {
    using dlss5_nr::plan_halvings;
    using dlss5_nr::plan_padding;
    using dlss5_nr::plan_working_extent;
    using dlss5_nr::kPlanMaxLayers;

    LayerSpec specs[kPlanMaxLayers];
    LayerSpec pristine[kPlanMaxLayers];
    fill_bytes(specs, 0xAB, sizeof specs);
    fill_bytes(pristine, 0xAB, sizeof pristine);
    int count = 7;
    // Null output pointer.
    CHECK(dlss5_nr::build_plan(320, 240, kHt, nullptr,
                               kPlanMaxLayers, count) == Status::kNullPointer);
    CHECK(count == 7);
    // Frame dimensions: 0, negative and beyond 16384 on either axis.
    static const int kBad[] = {0, -1, -320, 16385, 1 << 20};
    for (int bad : kBad) {
        CHECK(dlss5_nr::build_plan(bad, 240, kHt, specs,
                                   kPlanMaxLayers, count) == Status::kInvalidDimensions);
        CHECK(dlss5_nr::build_plan(320, bad, kHt, specs,
                                   kPlanMaxLayers, count) == Status::kInvalidDimensions);
        CHECK(dlss5_nr::build_plan(bad, 240, kSmall, specs,
                                   kPlanMaxLayers, count) == Status::kInvalidDimensions);
        CHECK(dlss5_nr::build_plan(320, bad, kSmall, specs,
                                   kPlanMaxLayers, count) == Status::kInvalidDimensions);
        CHECK(memcmp(specs, pristine, sizeof specs) == 0);
        CHECK(count == 7);
    }
    // Capacity: one short of the plan is rejected untouched; the exact
    // capacity is the accepting edge.
    CHECK(dlss5_nr::build_plan(320, 240, kHt, specs,
                               151, count) == Status::kInvalidBuffer);
    CHECK(dlss5_nr::build_plan(320, 240, kSmall, specs,
                               75, count) == Status::kInvalidBuffer);
    CHECK(memcmp(specs, pristine, sizeof specs) == 0);
    CHECK(count == 7);
    CHECK(dlss5_nr::build_plan(320, 240, kHt, specs,
                               152, count) == Status::kOk);
    CHECK(count == 152);
    fill_bytes(specs, 0xAB, sizeof specs);
    CHECK(dlss5_nr::build_plan(320, 240, kSmall, specs,
                               76, count) == Status::kOk);
    CHECK(count == 76);
    // Padding / working extent / halvings reject the same dimension classes,
    // outputs untouched.
    PlanPadding out;
    for (int bad : kBad) {
        fill_bytes(&out, 0xAB, sizeof out);
        CHECK(plan_padding(bad, 240, kHt, out) == Status::kInvalidDimensions);
        CHECK(plan_padding(320, bad, kHt, out) == Status::kInvalidDimensions);
        CHECK(plan_padding(bad, bad, kSmall, out) == Status::kInvalidDimensions);
        int sentinel = 0x5A5A5A5A;
        int w = sentinel;
        int h = sentinel;
        CHECK(plan_working_extent(bad, 240, kHt, w, h) == Status::kInvalidDimensions);
        CHECK(plan_working_extent(320, bad, kSmall, w, h) == Status::kInvalidDimensions);
        CHECK(w == sentinel && h == sentinel);
        int x = sentinel;
        int y = sentinel;
        CHECK(plan_halvings(bad, 240, kHt, x, y) == Status::kInvalidDimensions);
        CHECK(plan_halvings(320, bad, kSmall, x, y) == Status::kInvalidDimensions);
        CHECK(x == sentinel && y == sentinel);
        PlanPadding probe;
        fill_bytes(&probe, 0xAB, sizeof probe);
        CHECK(memcmp(&out, &probe, sizeof out) == 0);
    }
    // The 16384 boundary is accepted on both networks.
    PlanPadding edge;
    CHECK(plan_padding(16384, 16384, kHt, edge) == Status::kOk);
    CHECK(plan_padding(16384, 16384, kSmall, edge) == Status::kOk);
}

// ---- internal vocabulary controls ---------------------------------------------------------------------

static void check_internal_vocabulary() {
    using dlss5_nr::plan_detail::row_index;
    using dlss5_nr::plan_detail::shift_cycle;
    using dlss5_nr::plan_detail::up;
    using dlss5_nr::plan_detail::variant_from_name;

    // up(n, k) on the documented domain n >= 0.
    CHECK(up(0, 4) == 0);
    CHECK(up(1, 4) == 4);
    CHECK(up(4, 4) == 4);
    CHECK(up(5, 4) == 8);
    CHECK(up(8, 8) == 8);
    CHECK(up(9, 8) == 16);
    CHECK(up(1, 1) == 1);
    CHECK(up(321, 64) == 384);
    CHECK(up(33, 16) == 48);

    // The shift cycle repeats every four positions.
    static const int kCycle[4][2] = {{0, 0}, {-1, -1}, {-1, 0}, {0, -1}};
    for (int position = 0; position < 8; ++position) {
        int sx = 99;
        int sy = 99;
        shift_cycle(position, sx, sy);
        CHECK(sx == kCycle[position & 3][0]);
        CHECK(sy == kCycle[position & 3][1]);
    }

    // The variant scan takes the first tag in the pinned order.
    CHECK(variant_from_name("plain_name_fp8") == LayerVariant::kNone);
    CHECK(variant_from_name("x_inpview_y") == LayerVariant::kInpview);
    CHECK(variant_from_name("x_ds_y") == LayerVariant::kDs);
    CHECK(variant_from_name("x_upsample_y") == LayerVariant::kUpsample);
    CHECK(variant_from_name("x_outview_y") == LayerVariant::kOutview);
    CHECK(variant_from_name("x_proj_pool_y") == LayerVariant::kProjPool);
    CHECK(variant_from_name("x_final_head_y") == LayerVariant::kFinalHead);
    CHECK(variant_from_name("x_inpview_ds_y") == LayerVariant::kInpview);
    CHECK(variant_from_name("x_ds_upsample_y") == LayerVariant::kDs);

    // row_index finds (block, layer) positions in a built row table and
    // rejects missing ones.
    using dlss5_nr::plan_detail::PlanRow;
    PlanRow rows[dlss5_nr::kPlanMaxLayers];
    const int count = dlss5_nr::plan_detail::build_rows(kHt, rows);
    CHECK(count == 152);
    CHECK(row_index(rows, count, 0, 0) == 0);
    CHECK(row_index(rows, count, 30, 4) == 55);
    CHECK(row_index(rows, count, 70, 0) == 151);
    CHECK(row_index(rows, count, 30, 9) == -1);
    CHECK(row_index(rows, count, 99, 0) == -1);
}

// ---- determinism and allocation accounting --------------------------------------------------------

static void check_determinism_and_allocations() {
    using dlss5_nr::plan_halvings;
    using dlss5_nr::plan_padding;
    using dlss5_nr::plan_working_extent;
    using dlss5_nr::kPlanMaxLayers;

    // Determinism: the same call twice produces byte-identical metadata.
    LayerSpec a[kPlanMaxLayers];
    LayerSpec b[kPlanMaxLayers];
    for (NetworkKind network : {kHt, kSmall}) {
        fill_bytes(a, 0, sizeof a);
        fill_bytes(b, 0, sizeof b);
        int ca = 0;
        int cb = 0;
        CHECK(dlss5_nr::build_plan(321, 241, network, a, kPlanMaxLayers, ca)
              == Status::kOk);
        CHECK(dlss5_nr::build_plan(321, 241, network, b, kPlanMaxLayers, cb)
              == Status::kOk);
        CHECK(ca == cb);
        CHECK(memcmp(a, b, sizeof a) == 0);
    }

    // Zero-allocation accounting over the whole public surface.
    const unsigned long long before = g_allocations.load();
    LayerSpec specs[kPlanMaxLayers];
    for (int i = 0; i < 64; ++i) {
        for (NetworkKind network : {kHt, kSmall}) {
            int count = 0;
            CHECK(dlss5_nr::build_plan(1920, 1080, network, specs,
                                       kPlanMaxLayers, count) == Status::kOk);
            PlanPadding padded;
            CHECK(plan_padding(1920, 1080, network, padded) == Status::kOk);
            int w = 0;
            int h = 0;
            CHECK(plan_working_extent(1920, 1080, network, w, h) == Status::kOk);
            int x = 0;
            int y = 0;
            CHECK(plan_halvings(1920, 1080, network, x, y) == Status::kOk);
            for (int index = 0; index < count; ++index) {
                CHECK(dlss5_nr::plan_skip_source(specs, count, index) >= -1);
            }
        }
    }
    CHECK(g_allocations.load() == before);
}

int main(void) {
    check_structural_counts();
    check_padding_table();
    check_halvings();
    check_rows_ht();
    check_rows_small();
    check_skip_wiring();
    check_grid_tokens_input_block();
    check_extent_coherence();
    check_rejections();
    check_internal_vocabulary();
    check_determinism_and_allocations();
    printf("native-graph-plan-test: %lld checks passed\n", g_checks);
    return 0;
}
