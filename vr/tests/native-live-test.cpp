// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Michael Lamothe
//
// The sustained live-input harness: multi-frame feeding of the full native
// evaluator (decode -> pipeline -> output) through the temporal history
// consumer, at the demo's geometry (64x72 frame, the small plan's 320x320
// working floor). The checks are the sustained-run contracts:
//   * every frame of the sequence succeeds through temporal_image_evaluation
//     (a mid-stream reset included);
//   * determinism: replaying the whole sequence on a fresh carrier
//     reproduces every output frame bit-exactly;
//   * input responsiveness: a different input stream produces different
//     output frames (the evaluator responds to the live input, not just the
//     temporal state);
//   * no resource leaks: the operator-new accounting balances across every
//     frame (the arena covers the run), the arena high-water is constant
//     across frames (the per-frame rewind contract), and the process RSS is
//     bounded after warm-up over the whole session;
//   * the frames-since-reset counter follows the sequence, including the
//     reset's zero.
// Throughput is MEASURED and printed (mean ms/frame, frames/s and the
// decode/pipeline/output split derived from the boundary entries measured
// directly) — reported, never asserted: the numbers carry the synthetic
// weights' honest label (the transformation is arbitrary; the speed is real).
//
// Built by vr/test-native-live.sh at -O2 and -O0 with -ffp-contract=off.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cstring>
#include <new>

#include "native-demo-support.hpp"
#include "native-io.hpp"

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
// The live-allocation balance (new minus delete): the parallel layer's
// documented spawn-and-join design allocates transient thread state per
// parallel_for call, so the LEAK contract is balance (everything freed by
// the frame's end), not zero allocations. The churn itself is measured and
// reported.
std::atomic<long long> g_live{0};

constexpr int kFrameW = 64;
constexpr int kFrameH = 72;
constexpr int kMaxSeqFrames = 8;
// The sequence depth (frames per sequence). The full soak runs at -O2; the
// -O0 level (the recorded codegen class, reinstated) and any sanitised build
// run the same assertion classes at reduced depth (DLSS5_LIVE_DEPTH).
// Cross-optimisation bit-identity is pinned by the eval/io suites' known
// answers, not re-measured here.
int g_depth = 3;

// The live frame source: frame N's bytes are a deterministic function of
// (salt, N, x, y), a drifting ramp on the 8-bit grid. A different salt is a
// different input stream.
void fill_frame(std::uint8_t* host, int frame, int salt) {
    for (int y = 0; y < kFrameH; ++y) {
        for (int x = 0; x < kFrameW; ++x) {
            for (int c = 0; c < 3; ++c) {
                host[3 * (y * kFrameW + x) + c] = static_cast<std::uint8_t>(
                    (x * 7 + y * 13 + c * 29 + frame * 5 + salt * 101) & 0xFF);
            }
        }
    }
}

// The process RSS in KiB from /proc/self/statm (Linux; page 1 = resident).
// Reported and bounds-checked after warm-up.
long long read_rss_kib() {
    std::FILE* f = std::fopen("/proc/self/statm", "r");
    if (f == nullptr) {
        return -1;
    }
    unsigned long long total = 0;
    unsigned long long resident = 0;
    const int got = std::fscanf(f, "%llu %llu", &total, &resident);
    std::fclose(f);
    // statm reports pages; honour the host page size (16K/64K-page kernels
    // would otherwise under-report and weaken the growth bound).
    const long page = sysconf(_SC_PAGESIZE);
    return got == 2
        ? static_cast<long long>(resident) * (page > 0 ? page / 1024 : 4) : -1;
}

struct SequenceResult {
    double total_ms = 0.0;
    double decode_ms = 0.0;
    double output_ms = 0.0;
    unsigned long long alloc_churn = 0;
    std::uint32_t counters[kMaxSeqFrames] = {};
    std::uint8_t frames[kMaxSeqFrames][3 * kFrameW * kFrameH] = {};
    long long rss_warm_kib = -1;
    long long rss_end_kib = -1;
};

// One sequence of live frames through the temporal consumer, with
// `reset_at` (-1 = never) and the input stream `salt`. Per frame: the
// allocation accounting must balance, the arena high-water must equal the
// first frame's (the rewind contract) and the counter must follow the
// reset schedule. The boundary stages are measured directly around the step
// (the pipeline portion is derived by subtraction in the report).
void run_sequence(int reset_at, int salt, SequenceResult& out,
                  const LayerSpec* specs, int count, EvalContext& context,
                  TemporalCarrier& carrier, float* image_storage,
                  std::size_t image_floats, float* store_scratch,
                  std::size_t store_floats, int work_rows, int work_cols,
                  NetworkRun& run) {
    static std::uint8_t host[3 * kFrameW * kFrameH];
    static float image_probe[3 * 320 * 320];
    static std::uint8_t out_probe[3 * kFrameW * kFrameH];
    std::size_t arena_first = 0;
    for (int frame = 0; frame < g_depth; ++frame) {
        const bool reset_now = reset_at >= 0 && frame == reset_at;
        out.counters[frame] = reset_now ? 0u : temporal_counter(carrier);
        fill_frame(host, frame, salt);
        CHECK_STATUS(eval_reset(context), Status::kOk);
        const unsigned long long before = g_allocations.load();
        const long long live_before = g_live.load();
        const auto t0 = std::chrono::steady_clock::now();
        CHECK_STATUS(temporal_image_evaluation(
            host, 3 * kFrameW, kFrameW, kFrameH, out.frames[frame],
            3 * kFrameW, image_storage, image_floats, specs, count, run,
            context, carrier, reset_now, store_scratch, store_floats),
            Status::kOk);
        const auto t1 = std::chrono::steady_clock::now();
        // The reset's zero and the per-run increment are asserted directly on
        // the carrier's POST-call state (not only on the next frame's read):
        // after a reset the counter must be 1 (zeroed, then advanced by the
        // run); otherwise exactly one more than the recorded entry. Without
        // the reset honoured at this frame the value would be reset_at+1.
        CHECK(temporal_counter(carrier)
              == (reset_now ? 1u : out.counters[frame] + 1u));
        // The leak contract is BALANCE: the frame's transient allocations
        // (the parallel layer's documented spawn-and-join thread state) are
        // all freed by its end. The churn count is measured and reported.
        CHECK(g_live.load() == live_before);
        out.alloc_churn += g_allocations.load() - before;
        // The arena rewinds per frame; the high-water is constant for a
        // constant plan geometry.
        const std::size_t used = eval_arena_used(context);
        if (frame == 0) {
            arena_first = used;
        }
        CHECK(used == arena_first);
        out.total_ms += std::chrono::duration<double, std::milli>(t1 - t0)
            .count();
        const auto d0 = std::chrono::steady_clock::now();
        CHECK_STATUS(frame_from_u8(host, 3 * kFrameW, kFrameW, kFrameH,
                                   image_probe, image_floats, work_rows,
                                   work_cols),
                     Status::kOk);
        const auto d1 = std::chrono::steady_clock::now();
        out.decode_ms += std::chrono::duration<double, std::milli>(d1 - d0)
            .count();
        const auto e0 = std::chrono::steady_clock::now();
        CHECK_STATUS(frame_to_u8(image_probe, work_rows, work_cols, out_probe,
                                 3 * kFrameW, kFrameW, kFrameH), Status::kOk);
        const auto e1 = std::chrono::steady_clock::now();
        out.output_ms += std::chrono::duration<double, std::milli>(e1 - e0)
            .count();
        if (frame == 0) {
            out.rss_warm_kib = read_rss_kib();
        }
        out.rss_end_kib = read_rss_kib();
    }
}

}  // namespace

void* operator new(std::size_t n) {
    g_allocations.fetch_add(1);
    g_live.fetch_add(1);
    void* p = std::malloc(n);
    if (p == nullptr) {
        std::abort();
    }
    return p;
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void* operator new(std::size_t n, std::align_val_t a) {
    g_allocations.fetch_add(1);
    g_live.fetch_add(1);
    const std::size_t alignment = static_cast<std::size_t>(a);
    const std::size_t rounded = (n + alignment - 1) / alignment * alignment;
    void* p = std::aligned_alloc(alignment, rounded == 0 ? alignment : rounded);
    if (p == nullptr) {
        std::abort();
    }
    return p;
}
void* operator new[](std::size_t n, std::align_val_t a) { return ::operator new(n, a); }
void operator delete(void* p) noexcept { if (p != nullptr) { g_live.fetch_sub(1); } std::free(p); }
void operator delete[](void* p) noexcept { if (p != nullptr) { g_live.fetch_sub(1); } std::free(p); }
void operator delete(void* p, std::size_t) noexcept { if (p != nullptr) { g_live.fetch_sub(1); } std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { if (p != nullptr) { g_live.fetch_sub(1); } std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { if (p != nullptr) { g_live.fetch_sub(1); } std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { if (p != nullptr) { g_live.fetch_sub(1); } std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { if (p != nullptr) { g_live.fetch_sub(1); } std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { if (p != nullptr) { g_live.fetch_sub(1); } std::free(p); }

int main() {
    // The sequence depth (the runner bounds the slow optimisation levels).
    if (const char* env = std::getenv("DLSS5_LIVE_DEPTH")) {
        const int depth = std::atoi(env);
        if (depth >= 1 && depth <= kMaxSeqFrames) {
            g_depth = depth;
        }
    }
    // ---- the plan, the records, the arena --------------------------------
    static LayerSpec specs[kPlanMaxLayers];
    static LayerValue outputs[kPlanMaxLayers];
    static int skip_scratch[kPlanMaxLayers];
    static int last_scratch[kPlanMaxLayers];
    int count = 0;
    CHECK_STATUS(build_plan(kFrameW, kFrameH, NetworkKind::kSmall, specs,
                            kPlanMaxLayers, count), Status::kOk);
    CHECK(count > 70);
    PlanPadding padding{};
    CHECK_STATUS(plan_padding(kFrameW, kFrameH, NetworkKind::kSmall, padding),
                 Status::kOk);
    // The harness' fixed 3*320*320 probe maps assume the pinned working
    // floor; fail loudly if the padding rule ever yields a larger extent
    // (a changed geometry must resize the probes, not silently overflow).
    CHECK(padding.width == 320 && padding.height == 320);
    static dlss5_demo::DemoStore store;
    store.preset = dlss5_demo::DemoStore::Preset::kSeeded;
    store.seed = 20261028u;
    static EvalWeights weights;
    dlss5_demo::demo_bind_weights(weights, store);

    const std::size_t image_floats = 3u
        * static_cast<std::size_t>(padding.width)
        * static_cast<std::size_t>(padding.height);
    float* image_storage = static_cast<float*>(
        std::malloc(image_floats * sizeof(float)));
    float* store_scratch = static_cast<float*>(
        std::malloc(image_floats * sizeof(float)));
    float* carrier_map = static_cast<float*>(
        std::malloc(image_floats * sizeof(float)));
    float* carrier_map_replay = static_cast<float*>(
        std::malloc(image_floats * sizeof(float)));
    float* carrier_map_b = static_cast<float*>(
        std::malloc(image_floats * sizeof(float)));
    const std::size_t arena_floats = 1u << 27;
    float* arena = static_cast<float*>(
        std::malloc(arena_floats * sizeof(float)));
    if (image_storage == nullptr || store_scratch == nullptr
        || carrier_map == nullptr || carrier_map_replay == nullptr
        || carrier_map_b == nullptr || arena == nullptr) {
        std::fprintf(stderr, "harness allocation failed\n");
        return 1;
    }

    static EvalContext context;
    context.weights = &weights;
    context.scratch = arena;
    context.scratch_capacity = arena_floats;
    CHECK_STATUS(eval_reset(context), Status::kOk);
    // The thread count. The runner pins 8 at every level (the measured
    // configuration); the serial reference is covered by the eval/io suites.
    int threads = 8;
    if (const char* env = std::getenv("DLSS5_LIVE_THREADS")) {
        const int parsed = std::atoi(env);
        if (parsed >= 1 && parsed <= 64) {
            threads = parsed;
        }
    }
    dlss5_nr::set_parallel_threads(threads);

    NetworkRun run;
    run.outputs = outputs;
    run.skip_sources = skip_scratch;
    run.last_uses = last_scratch;
    run.layer = dlss5_nr::eval_layer;
    run.release = dlss5_nr::eval_release;

    TemporalCarrier carrier{};
    TemporalCarrier carrier_replay{};
    TemporalCarrier carrier_b{};
    CHECK_STATUS(temporal_init(carrier, carrier_map, image_floats,
                              padding.height, padding.width), Status::kOk);
    CHECK_STATUS(temporal_init(carrier_replay, carrier_map_replay, image_floats,
                              padding.height, padding.width), Status::kOk);
    CHECK_STATUS(temporal_init(carrier_b, carrier_map_b, image_floats,
                              padding.height, padding.width), Status::kOk);

    // ---- sequence A: sustain + state (reset before frame depth/2) ---------
    const int reset_at = g_depth > 1 ? g_depth / 2 : 0;
    SequenceResult a;
    run_sequence(reset_at, 0, a, specs, count, context, carrier, image_storage,
                 image_floats, store_scratch, image_floats, padding.height,
                 padding.width, run);
    // The counter follows the reset schedule (the contract's input counter:
    // frames since the last reset, zero at the reset frame).
    for (int frame = 0; frame < g_depth; ++frame) {
        const std::uint32_t want = frame < reset_at
            ? static_cast<std::uint32_t>(frame)
            : static_cast<std::uint32_t>(frame - reset_at);
        CHECK(a.counters[frame] == want);
    }

    // ---- sequence A': the replay must be bit-identical --------------------
    SequenceResult replay;
    run_sequence(reset_at, 0, replay, specs, count, context, carrier_replay,
                 image_storage, image_floats, store_scratch, image_floats,
                 padding.height, padding.width, run);
    for (int frame = 0; frame < g_depth; ++frame) {
        CHECK(replay.counters[frame] == a.counters[frame]);
        CHECK(std::memcmp(a.frames[frame], replay.frames[frame],
                          sizeof(a.frames[frame])) == 0);
    }

    // ---- sequence B: a different input stream must differ -----------------
    SequenceResult b;
    run_sequence(-1, 1, b, specs, count, context, carrier_b, image_storage,
                 image_floats, store_scratch, image_floats, padding.height,
                 padding.width, run);
    for (int frame = 0; frame < g_depth; ++frame) {
        CHECK(b.counters[frame] == static_cast<std::uint32_t>(frame));
        CHECK(std::memcmp(a.frames[frame], b.frames[frame],
                          sizeof(a.frames[frame])) != 0);
    }

    // ---- the resource assertions ------------------------------------------
    // Bounded RSS growth after warm-up across both sequences (the arena and
    // maps are allocated up front; per-frame growth would be a leak).
    if (a.rss_warm_kib >= 0 && b.rss_end_kib >= 0) {
        const long long growth = b.rss_end_kib - a.rss_warm_kib;
        std::printf("native-live: RSS warm %lld KiB -> end %lld KiB "
                    "(growth %lld KiB)\n", a.rss_warm_kib, b.rss_end_kib,
                    growth);
        CHECK(growth < 4096);
    } else {
        std::printf("native-live: RSS probe unavailable\n");
    }

    // ---- the measured throughput (reported, never asserted) ---------------
    const double frames_run = 3.0 * g_depth;
    const double total = a.total_ms + replay.total_ms + b.total_ms;
    const double decode = a.decode_ms + replay.decode_ms + b.decode_ms;
    const double output = a.output_ms + replay.output_ms + b.output_ms;
    const double mean_ms = total / frames_run;
    std::printf("native-live: %g frames at %dx%d (working %dx%d), %d thread(s) "
                "(depth %d per sequence)\n",
                frames_run, kFrameW, kFrameH, padding.width, padding.height,
                dlss5_nr::parallel_threads(), g_depth);
    std::printf("native-live: mean %.2f ms/frame (%.3f frames/s)\n", mean_ms,
                1000.0 / mean_ms);
    std::printf("native-live: decode %.3f ms, pipeline %.3f ms, output %.3f ms "
                "(means; pipeline derived by subtraction)\n",
                decode / frames_run, (total - decode - output) / frames_run,
                output / frames_run);
    std::printf("native-live: allocation churn %llu per frame (balanced; the "
                "parallel layer's spawn-and-join thread state), arena "
                "constant, RSS bounded\n",
                (a.alloc_churn + replay.alloc_churn + b.alloc_churn)
                    / static_cast<unsigned long long>(frames_run));
    std::printf("native-live: synthetic weights (seeded preset) - the "
                "transformation is arbitrary; the speed is real\n");

    std::free(image_storage);
    std::free(store_scratch);
    std::free(carrier_map);
    std::free(carrier_map_replay);
    std::free(carrier_map_b);
    std::free(arena);
    std::printf("native-live: %lld checks passed\n", g_checks);
    return 0;
}
