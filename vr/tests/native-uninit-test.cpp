// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Michael Lamothe
//
// Scratch-content independence (the MALLOC_PERTURB_ regression): the native
// evaluator's outputs must not depend on the content of memory that the
// kernels are supposed to overwrite before reading. Three levels pin it:
//
//   1. The swin scratch partitions and the per-row decode staging
//      (the recorded uninitialised-read candidates, NATIVE-LIVE.md): the
//      decoded-weight staging buffer and swin_block/swin_mlp/swin_attention
//      scratch are poisoned with patterns including NaN (0xFF), denormal
//      (0x01) and pseudorandom bytes before every call; the outputs must be
//      bit-identical (memcmp) to the zeroed-scratch baseline. Any
//      read-before-write that reaches an output or a value class breaks the
//      comparison.
//   2. The full-network arena (the caller's scratch partition per row): the
//      arena is poisoned before a frame; the temporal consumer's output
//      frames must be bit-identical to the unpainted baseline sequence,
//      painted frame 1 (fresh garbage) and frame 2 (run-1 leftovers) both
//      compared.
//   3. Cross-process malloc-fill sweep (the literal MALLOC_PERTURB_
//      contract, driven by vr/test-native-uninit.sh): the printed frame
//      hashes must be identical across perturb values.
//
// This pins CONTENT INDEPENDENCE of every observable output. A hypothetical
// uninitialised read whose garbage affects only timing (dead computation)
// is outside what an output regression can observe; NATIVE-LIVE.md records
// the timing evidence separately. CPU-only; no GPU, headset, proprietary
// payload, Python or PyTorch runtime required. Synthetic correctness only:
// no vendor correspondence.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "native-demo-support.hpp"
#include "native-io.hpp"
#include "native-swin-resample.hpp"

using namespace dlss5_nr;

namespace {

MatrixView make_view(const float* data, int rows, int cols, int stride) {
    MatrixView v;
    v.data = data;
    v.rows = rows;
    v.cols = cols;
    v.row_stride = stride;
    return v;
}

MutableMatrixView make_out(float* data, int rows, int cols, int stride) {
    MutableMatrixView v;
    v.data = data;
    v.rows = rows;
    v.cols = cols;
    v.row_stride = stride;
    return v;
}

long long g_checks = 0;

void check(int ok, const char* expression, int line) {
    ++g_checks;
    if (!ok) {
        std::fprintf(stderr, "FAILED (%d): %s\n", line, expression);
        std::exit(1);
    }
}

#define CHECK(expression) check((expression) != 0, #expression, __LINE__)

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

// The poison patterns: zeros (baseline), denormal-byte, mid-byte, the
// glibc MALLOC_PERTURB_ 165 fill, NaN and a seeded pseudorandom byte.
const unsigned char kPatterns[] = {0x00, 0x01, 0x5A, 0xA5, 0xFF};
constexpr int kPatternCount = 5;

void paint(void* buffer, std::size_t bytes, unsigned char pattern) {
    std::memset(buffer, static_cast<int>(pattern), bytes);
}

void paint_random(void* buffer, std::size_t bytes, std::uint32_t seed) {
    auto* p = static_cast<unsigned char*>(buffer);
    std::uint32_t s = seed;
    for (std::size_t i = 0; i < bytes; ++i) {
        s = s * 1664525u + 1013904223u;
        p[i] = static_cast<unsigned char>(s >> 24);
    }
}

// The poison schedule: kPatternCount painted configurations (the kPatterns
// bytes including the zeroed control) plus the seeded pseudorandom paint.
// Against the zeroed-scratch baseline that is five poisons: denormal 0x01,
// 0x5A (the glibc MALLOC_PERTURB_ fill byte), 0xA5, NaN 0xFF and random.
void paint_pattern(void* buffer, std::size_t bytes, int pattern,
                   std::uint32_t seed) {
    if (pattern < kPatternCount) {
        paint(buffer, bytes, kPatterns[pattern]);
    } else {
        paint_random(buffer, bytes, seed);
    }
}

unsigned long long fnv1a(const void* data, std::size_t bytes) {
    const auto* p = static_cast<const unsigned char*>(data);
    unsigned long long h = 1469598103934665603ull;
    for (std::size_t i = 0; i < bytes; ++i) {
        h = (h ^ p[i]) * 1099511628211ull;
    }
    return h;
}

constexpr int kFrameW = 64;
constexpr int kFrameH = 72;
constexpr int kPaddedDim = 320;
// The network section's sequence depth and pattern set (the runner bounds
// the sweep invocations and the slow -O0 level through these).
int g_net_frames = 2;
bool g_patterns_reduced = false;

void fill_frame(std::uint8_t* host, int frame) {
    for (int y = 0; y < kFrameH; ++y) {
        for (int x = 0; x < kFrameW; ++x) {
            for (int c = 0; c < 3; ++c) {
                host[3 * (y * kFrameW + x) + c] = static_cast<std::uint8_t>(
                    (x * 7 + y * 13 + c * 29 + frame * 5) & 0xFF);
            }
        }
    }
}

// ---- section 1: the swin scratch partitions and decode staging ----------

// An f16-exact input map (the swin_core operand class).
void fill_f16_map(float* map, std::size_t count, std::uint32_t seed) {
    std::uint32_t s = seed;
    for (std::size_t i = 0; i < count; ++i) {
        s = s * 1664525u + 1013904223u;
        const float v = static_cast<float>(static_cast<int>(s >> 24) - 128)
            * 0.25f;
        map[i] = f16(v);
    }
}

// One (block, layer, channels, heads) swin row's weights from the seeded
// store, decoded through the per-row staging seam.
struct SwinRow {
    eval_detail::SwinFetch fetch;
    int channels = 0;
    int heads = 0;
};

void run_swin_kernel_section(const EvalWeights& weights, int block, int layer,
                             int channels, int heads) {
    const std::size_t c = static_cast<std::size_t>(channels);
    SwinRow row;
    row.channels = channels;
    row.heads = heads;
    CHECK_STATUS(eval_detail::swin_fetch(&weights, block, layer, channels, heads,
                                         row.fetch),
                 Status::kOk);
    const std::size_t decoded_floats =
        eval_detail::swin_decoded_floats(channels, heads, row.fetch);
    std::vector<float> decoded_a(decoded_floats + 1, 0.0f);
    std::vector<float> decoded_b(decoded_floats + 1, 0.0f);
    // The decoded staging: poison the buffer every time; the decoded
    // matrices and derived exp_bias table must be identical to the
    // zero-filled baseline (and the guard word beyond the region untouched).
    MatrixView mid_a;
    MatrixView mid_b;
    SwinBlockParams params_a;
    SwinBlockParams params_b;
    decoded_a[decoded_floats] = 777.0f;
    decoded_b[decoded_floats] = 777.0f;
    CHECK_STATUS(eval_detail::swin_decode(row.fetch, channels, heads,
                                          decoded_a.data(), decoded_floats,
                                          mid_a, params_a),
                 Status::kOk);
    CHECK(decoded_a[decoded_floats] == 777.0f);
    for (int pattern = 0; pattern <= kPatternCount; ++pattern) {
        paint_pattern(decoded_b.data(), decoded_floats * sizeof(float), pattern,
                      1234u);
        decoded_b[decoded_floats] = 777.0f;
        CHECK_STATUS(eval_detail::swin_decode(row.fetch, channels, heads,
                                              decoded_b.data(), decoded_floats,
                                              mid_b, params_b),
                     Status::kOk);
        CHECK(std::memcmp(decoded_a.data(), decoded_b.data(),
                          decoded_floats * sizeof(float)) == 0);
        CHECK(decoded_b[decoded_floats] == 777.0f);
    }

    // swin_core over the poisoned block scratch: every pattern's wide and
    // acc outputs must equal the zeroed-scratch baseline bit for bit.
    int rows = 16;
    int cols = 16;
    const std::size_t map_count = c * static_cast<std::size_t>(rows) * cols;
    std::vector<float> x(map_count);
    fill_f16_map(x.data(), map_count, 4242u + static_cast<std::uint32_t>(block));
    FeatureBatchView x_view;
    x_view.data = x.data();
    x_view.channels = channels;
    x_view.rows = rows;
    x_view.cols = cols;
    x_view.plane_stride = static_cast<std::ptrdiff_t>(rows) * cols;
    x_view.row_stride = static_cast<std::ptrdiff_t>(cols);
    const std::size_t block_needed = swin_block_scratch_needed(
        x_view, true, false, 0, 0, params_a);
    CHECK(block_needed > 0);
    std::vector<float> scratch_a(block_needed + 1, 0.0f);
    std::vector<float> scratch_b(block_needed + 1, 0.0f);
    std::vector<float> wide_a(map_count + 1, 0.0f);
    std::vector<float> wide_b(map_count + 1, 0.0f);
    std::vector<float> acc_a(map_count + 1, 0.0f);
    std::vector<float> acc_b(map_count + 1, 0.0f);
    wide_a[map_count] = 777.0f;
    acc_a[map_count] = 777.0f;
    CHECK_STATUS(swin_core(x_view, 0, 0, params_a, wide_a.data(), map_count,
                           acc_a.data(), map_count, scratch_a.data(),
                           block_needed),
                 Status::kOk);
    CHECK(wide_a[map_count] == 777.0f);
    CHECK(acc_a[map_count] == 777.0f);
    for (int pattern = 0; pattern <= kPatternCount; ++pattern) {
        paint_pattern(scratch_b.data(), block_needed * sizeof(float), pattern,
                      99u);
        wide_b[map_count] = 777.0f;
        acc_b[map_count] = 777.0f;
        CHECK_STATUS(swin_core(x_view, 0, 0, params_b, wide_b.data(), map_count,
                               acc_b.data(), map_count, scratch_b.data(),
                               block_needed),
                     Status::kOk);
        CHECK(std::memcmp(wide_a.data(), wide_b.data(),
                          map_count * sizeof(float)) == 0);
        CHECK(std::memcmp(acc_a.data(), acc_b.data(),
                          map_count * sizeof(float)) == 0);
        CHECK(wide_b[map_count] == 777.0f);
        CHECK(acc_b[map_count] == 777.0f);
    }

    // The direct swin_mlp + swin_attention seam with poisoned scratch (the
    // recorded stall's stack samples landed in these kernels). The token
    // batch is e4m3-grid (the linear operand class), like the block's
    // quantised windows.
    const std::size_t token_rows = 64;
    std::vector<float> tokens(token_rows * c);
    {
        std::uint32_t s = 777u;
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            s = s * 1664525u + 1013904223u;
            tokens[i] = e4m3(static_cast<float>(
                                 static_cast<int>(s >> 24) - 128) * 0.25f);
        }
    }
    MatrixView x_batch = make_view(tokens.data(), static_cast<int>(token_rows),
                                   channels, channels);
    const std::size_t hidden_count = token_rows
        * static_cast<std::size_t>(params_a.expand.rows);
    std::vector<float> hidden(hidden_count + 1, 0.0f);
    std::vector<float> y_base(token_rows * c + 1, 0.0f);
    std::vector<float> y_test(token_rows * c + 1, 0.0f);
    std::vector<float> hidden_poison(hidden_count + 1, 0.0f);
    y_base[token_rows * c] = 777.0f;
    hidden[hidden_count] = 777.0f;
    MutableMatrixView y_out = make_out(y_base.data(),
                                       static_cast<int>(token_rows), channels,
                                       channels);
    MutableMatrixView hidden_out = make_out(hidden.data(),
                                            static_cast<int>(token_rows),
                                            params_a.expand.rows,
                                            params_a.expand.rows);
    CHECK_STATUS(swin_mlp(x_batch, params_a.expand, params_a.mid, params_a.heads,
                          params_a.contract, x_batch, params_a.mlp_scale, &y_out,
                          token_rows * c, &hidden_out, hidden_count),
                 Status::kOk);
    CHECK(y_base[token_rows * c] == 777.0f);
    CHECK(hidden[hidden_count] == 777.0f);
    for (int pattern = 0; pattern <= kPatternCount; ++pattern) {
        paint_pattern(hidden_poison.data(),
                      hidden_poison.size() * sizeof(float), pattern, 5u);
        MutableMatrixView y_try = make_out(y_test.data(),
                                           static_cast<int>(token_rows),
                                           channels, channels);
        MutableMatrixView hidden_try = make_out(hidden_poison.data(),
                                                static_cast<int>(token_rows),
                                                params_a.expand.rows,
                                                params_a.expand.rows);
        y_test[token_rows * c] = 777.0f;
        hidden_poison[hidden_count] = 777.0f;
        CHECK_STATUS(swin_mlp(x_batch, params_a.expand, params_a.mid,
                              params_a.heads, params_a.contract, x_batch,
                              params_a.mlp_scale, &y_try, token_rows * c,
                              &hidden_try, hidden_count),
                     Status::kOk);
        CHECK(std::memcmp(y_base.data(), y_test.data(),
                          token_rows * c * sizeof(float)) == 0);
        CHECK(y_test[token_rows * c] == 777.0f);
        CHECK(hidden_poison[hidden_count] == 777.0f);
    }

    const std::size_t attn_needed =
        token_rows * (3 * c) + 4 * token_rows * c
        + 2 * static_cast<std::size_t>(params_a.exp_bias.rows) * kBiasTable;
    std::vector<float> attn_scratch(attn_needed, 0.0f);
    std::vector<float> attn_poison(attn_needed, 0.0f);
    std::vector<float> out_base(token_rows * c + 1, 0.0f);
    std::vector<float> out_test(token_rows * c + 1, 0.0f);
    out_base[token_rows * c] = 777.0f;
    MutableMatrixView attn_out = make_out(out_base.data(),
                                          static_cast<int>(token_rows), channels,
                                          channels);
    CHECK_STATUS(swin_attention(x_batch, params_a.qkv, params_a.head_scale,
                                params_a.exp_bias, x_batch, params_a.attn_scale,
                                params_a.out_proj, &attn_out, token_rows * c,
                                attn_scratch.data(), attn_needed),
                 Status::kOk);
    CHECK(out_base[token_rows * c] == 777.0f);
    for (int pattern = 0; pattern <= kPatternCount; ++pattern) {
        paint_pattern(attn_poison.data(), attn_needed * sizeof(float), pattern,
                      31u);
        MutableMatrixView attn_try = make_out(out_test.data(),
                                              static_cast<int>(token_rows),
                                              channels, channels);
        out_test[token_rows * c] = 777.0f;
        CHECK_STATUS(swin_attention(x_batch, params_a.qkv, params_a.head_scale,
                                    params_a.exp_bias, x_batch,
                                    params_a.attn_scale, params_a.out_proj,
                                    &attn_try, token_rows * c,
                                    attn_poison.data(), attn_needed),
                     Status::kOk);
        CHECK(std::memcmp(out_base.data(), out_test.data(),
                          token_rows * c * sizeof(float)) == 0);
        CHECK(out_test[token_rows * c] == 777.0f);
    }

    // Negative control: an undersized block scratch rejects and leaves the
    // outputs untouched (sentinel-checked), poisoned or not.
    if (block_needed > 1) {
        paint(wide_b.data(), (map_count + 1) * sizeof(float), 0xA5);
        paint(acc_b.data(), (map_count + 1) * sizeof(float), 0xA5);
        paint(scratch_b.data(), block_needed * sizeof(float), 0xFF);
        CHECK_STATUS(swin_core(x_view, 0, 0, params_b, wide_b.data(), map_count,
                               acc_b.data(), map_count, scratch_b.data(),
                               block_needed - 1),
                     Status::kInvalidBuffer);
        for (std::size_t i = 0; i < map_count + 1; ++i) {
            std::uint32_t bits = 0;
            std::memcpy(&bits, &wide_b[i], sizeof(bits));
            CHECK(bits == 0xA5A5A5A5u);
            std::memcpy(&bits, &acc_b[i], sizeof(bits));
            CHECK(bits == 0xA5A5A5A5u);
        }
    }
}

// ---- section 2: the full-network arena sweep ----------------------------

constexpr int kNetFrames = 2;

void run_network_section(const LayerSpec* specs, int count, EvalWeights& weights,
                         std::size_t image_floats, int work_rows, int work_cols) {
    const int net_frames = g_net_frames;
    CHECK(net_frames >= 1 && net_frames <= kNetFrames);
    const std::size_t arena_floats = 1u << 27;
    static float* arena =
        static_cast<float*>(std::malloc(arena_floats * sizeof(float)));
    static float* image_storage =
        static_cast<float*>(std::malloc(image_floats * sizeof(float)));
    static float* store_scratch =
        static_cast<float*>(std::malloc(image_floats * sizeof(float)));
    static float* carrier_map =
        static_cast<float*>(std::malloc(image_floats * sizeof(float)));
    static LayerValue outputs[kPlanMaxLayers];
    static int skip_scratch[kPlanMaxLayers];
    static int last_scratch[kPlanMaxLayers];
    static std::uint8_t host[3 * kFrameW * kFrameH];
    CHECK(arena != nullptr && image_storage != nullptr
          && store_scratch != nullptr && carrier_map != nullptr);
    if (arena == nullptr || image_storage == nullptr || store_scratch == nullptr
        || carrier_map == nullptr) {
        return;
    }
    EvalContext context;
    context.weights = &weights;
    context.scratch = arena;
    context.scratch_capacity = arena_floats;
    NetworkRun run;
    run.outputs = outputs;
    run.skip_sources = skip_scratch;
    run.last_uses = last_scratch;
    run.layer = eval_layer;
    run.release = eval_release;
    std::printf("native-uninit: network sweep (arena poison), %d frame(s) per "
                "pattern\n", net_frames);

    // Patterns: no paint (baseline), the five bytes and the seeded random
    // paint. Each sequence paints only the FIRST frame, so the second frame
    // exercises the leftover-content case. The reduced set (the runner's
    // sweep and -O0 bound) keeps the baseline, NaN and random paints.
    std::uint8_t baseline[kNetFrames][3 * kFrameW * kFrameH];
    for (int pattern = -1; pattern <= kPatternCount; ++pattern) {
        // -1 is the unpainted baseline; the reduced set keeps the baseline,
        // the NaN paint (4) and the random paint (kPatternCount).
        const bool keep = pattern == -1 || pattern == 4
            || pattern == kPatternCount;
        if (g_patterns_reduced && !keep) {
            continue;
        }
        TemporalCarrier carrier{};
        CHECK_STATUS(temporal_init(carrier, carrier_map, image_floats, work_rows,
                                   work_cols),
                     Status::kOk);
        std::uint8_t out[kNetFrames][3 * kFrameW * kFrameH] = {};
        for (int frame = 0; frame < net_frames; ++frame) {
            fill_frame(host, frame);
            // The paint goes BEFORE eval_reset (which writes the arena state
            // header at the front of the buffer).
            if (frame == 0 && pattern >= 0) {
                if (pattern == kPatternCount) {
                    paint_random(arena, arena_floats * sizeof(float), 20261029u);
                } else {
                    paint(arena, arena_floats * sizeof(float),
                          kPatterns[pattern]);
                }
            }
            CHECK_STATUS(eval_reset(context), Status::kOk);
            CHECK_STATUS(temporal_image_evaluation(
                             host, 3 * kFrameW, kFrameW, kFrameH, out[frame],
                             3 * kFrameW, image_storage, image_floats, specs,
                             count, run, context, carrier, frame == 0,
                             store_scratch, image_floats),
                         Status::kOk);
            if (pattern < 0) {
                std::memcpy(baseline[frame], out[frame], sizeof(out[frame]));
            } else {
                CHECK(std::memcmp(baseline[frame], out[frame],
                                  sizeof(out[frame])) == 0);
            }
        }
        const char* name = pattern == -1 ? "baseline"
                          : pattern == kPatternCount ? "random"
                          : "byte";
        std::printf("native-uninit: pattern %d (%s) frames identical "
                    "(hashes", pattern, name);
        for (int frame = 0; frame < net_frames; ++frame) {
            std::printf(" %016llx", fnv1a(out[frame], sizeof(out[frame])));
        }
        std::printf(")\n");
    }
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    // The thread count (the live suite's convention: the measured 8-thread
    // configuration; 1 is the serial reference).
    int threads = 8;
    if (const char* env = std::getenv("DLSS5_UNINIT_THREADS")) {
        const int parsed = std::atoi(env);
        if (parsed >= 1 && parsed <= 64) {
            threads = parsed;
        }
    }
    set_parallel_threads(threads);
    // The network-section bounds (the runner's sweep and -O0 level).
    if (const char* env = std::getenv("DLSS5_UNINIT_NET_FRAMES")) {
        const int parsed = std::atoi(env);
        if (parsed >= 1 && parsed <= kNetFrames) {
            g_net_frames = parsed;
        }
    }
    if (const char* env = std::getenv("DLSS5_UNINIT_PATTERNS")) {
        g_patterns_reduced = std::strcmp(env, "reduced") == 0;
    }
    // ---- the seeded store and plan ---------------------------------------
    static LayerSpec specs[kPlanMaxLayers];
    int count = 0;
    CHECK_STATUS(build_plan(kFrameW, kFrameH, NetworkKind::kSmall, specs,
                            kPlanMaxLayers, count),
                 Status::kOk);
    CHECK(count > 70);
    PlanPadding padding{};
    CHECK_STATUS(plan_padding(kFrameW, kFrameH, NetworkKind::kSmall, padding),
                 Status::kOk);
    CHECK(padding.width == kPaddedDim && padding.height == kPaddedDim);
    static dlss5_demo::DemoStore store;
    store.preset = dlss5_demo::DemoStore::Preset::kSeeded;
    store.seed = 20261028u;
    static EvalWeights weights;
    dlss5_demo::demo_bind_weights(weights, store);

    // ---- section 1: one row per head family -----------------------------
    std::printf("native-uninit: swin scratch + decode staging sweep\n");
    int seen[4] = {0, 0, 0, 0};
    for (int i = 0; i < count; ++i) {
        const LayerSpec& spec = specs[i];
        const bool swin = spec.kind == LayerKind::kSwin1H
            || spec.kind == LayerKind::kSwin2H
            || spec.kind == LayerKind::kSwin4H
            || spec.kind == LayerKind::kSwin8H;
        if (!swin) {
            continue;
        }
        const int heads = spec.heads;
        const int family = heads == 1 ? 0 : heads == 2 ? 1 : heads == 4 ? 2 : 3;
        if (seen[family] != 0) {
            continue;
        }
        seen[family] = 1;
        run_swin_kernel_section(weights, spec.block, spec.layer, spec.channels,
                                heads);
    }
    CHECK(seen[0] != 0 && seen[1] != 0 && seen[2] != 0 && seen[3] != 0);

    // ---- section 2: the network arena ------------------------------------
    const std::size_t image_floats = 3u
        * static_cast<std::size_t>(padding.width)
        * static_cast<std::size_t>(padding.height);
    run_network_section(specs, count, weights, image_floats, padding.height,
                        padding.width);

    std::printf("native-uninit: synthetic weights - content independence "
                "only; no image-quality or vendor-parity claim\n");
    std::printf("native-uninit: %lld checks passed\n", g_checks);
    return 0;
}
