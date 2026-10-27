// Independent known-answer tests for vr/native-network.hpp — the plan-row
// dispatch (the pinned model.py NRNet.run / _skip_sources / _last_uses
// semantics). Every wiring expectation is re-derived here from the pinned
// rules written out independently of the header: which tensor each layer
// reads (main previous / skip main / skip aside / the raw pair / the padded
// frame), which layers produce (main, aside), and the last-use release
// schedule, over the full "ht" (152 rows) and "small" (76 rows) plans and
// hand-built mini plans for the edge contracts. No Python, no torch, no
// proprietary weights.

#include "native-network.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static void check_status(Status got, Status want, const char* expr, int line) {
    ++g_checks;
    if (got != want) {
        std::fprintf(stderr, "FAILED (%d): %s got %d want %d\n", line, expr,
                     static_cast<int>(got), static_cast<int>(want));
        std::exit(1);
    }
}

#define CHECK(ok) check((ok), #ok, __LINE__)
#define CHECK_STATUS(got, want) \
    check_status((got), (want), #got " == " #want, __LINE__)

// ---- the independent contract rules (from the pinned forwards) -----------------------------------

// The pinned tuple producers: `PreBlock.forward -> (pooled, skip)`, the `ds`
// Swin layers `-> (next, body)`, `SplitFfwd -> (out, x)`,
// `SplitProjPool -> (projection, pooled)`. Everything else returns one tensor.
static bool stub_pair(LayerKind kind, LayerVariant variant) {
    if (kind == LayerKind::kPreBlock
        || kind == LayerKind::kSplitFfwd
        || kind == LayerKind::kSplitProjPool) {
        return true;
    }
    return kind >= LayerKind::kSwin1H && kind <= LayerKind::kSwin8H
        && variant == LayerVariant::kDs;
}

// The pinned readers of the padded frame: `PreBlock.forward(image)` and
// `PostBlock.forward(x, skip, frame)`.
static bool stub_reads_image(LayerKind kind) {
    return kind == LayerKind::kPreBlock || kind == LayerKind::kPostBlock;
}

// The pinned `_skip_sources`, re-derived here from the rules in model.py:
// the split projections read their block's feed-forward projection (layer 1),
// the final head the pooled copy (layer 3), a vit contract the previous
// block's projection (layer 4), a vit projection its own contract (layer 1),
// and otherwise the named skip block's layer 0 (the decoder entry: layer 3).
static int expected_skip(const LayerSpec* specs, int count, int index) {
    int want_block = 0;
    int want_layer = 0;
    const LayerKind kind = specs[index].kind;
    if (kind == LayerKind::kSplitProj || kind == LayerKind::kSplitProjPool) {
        want_block = specs[index].block;
        want_layer = 1;
    } else if (kind == LayerKind::kSplitFinalHead) {
        want_block = specs[index].block;
        want_layer = 3;
    } else if (kind == LayerKind::kVitFfnContract) {
        want_block = specs[index].block - 1;
        want_layer = 4;
    } else if (kind == LayerKind::kVitProjection) {
        want_block = specs[index].block;
        want_layer = 1;
    } else {
        if (specs[index].skip_block < 0) {
            return -1;
        }
        want_block = specs[index].skip_block;
        want_layer = kind == LayerKind::kDecInputUpsample ? 3 : 0;
    }
    for (int i = 0; i < count; ++i) {
        if (specs[i].block == want_block && specs[i].layer == want_layer) {
            return i;
        }
    }
    return -1;
}

// The pinned `_last_uses`, re-derived here: every output is read by its
// sequential reader and by every layer naming it as a skip source.
static void expected_last_uses(const int* skip, int count, int* out) {
    for (int i = 0; i < count; ++i) {
        out[i] = i;
    }
    for (int i = 0; i < count; ++i) {
        if (skip[i] >= 0 && out[skip[i]] < i) {
            out[skip[i]] = i;
        }
    }
    for (int i = 1; i < count; ++i) {
        if (out[i - 1] < i) {
            out[i - 1] = i;
        }
    }
}

// The release schedule, re-derived from the pinned lifetimes. First the
// pinned del-loop: entry j is deleted at the first step i != j with
// last[j] <= i (ascending over the alive dict). Then the tensor lifetimes:
// a `SplitFfwd` row keeps its input's tensor as its second element (the
// pinned `(out, x)` tuple), so the input row's buffer is released only when
// both entries holding it have been deleted (a never-deleted entry keeps its
// buffer for the whole run).
static void expected_releases(const LayerSpec* specs, const int* last, int count,
                              int* out, int& out_count) {
    int deleted[kPlanMaxLayers];
    for (int j = 0; j < count; ++j) {
        deleted[j] = -1;
    }
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < count; ++j) {
            if (deleted[j] < 0 && j != i && last[j] <= i) {
                deleted[j] = i;
            }
        }
    }
    int release[kPlanMaxLayers];
    for (int j = 0; j < count; ++j) {
        release[j] = deleted[j];
    }
    for (int i = 0; i < count; ++i) {
        if (specs[i].kind == LayerKind::kSplitFfwd && i > 0) {
            if (deleted[i] < 0 || release[i - 1] < deleted[i]) {
                release[i - 1] = deleted[i];
            }
        }
    }
    out_count = 0;
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < count; ++j) {
            if (release[j] == i) {
                out[out_count++] = j;
            }
        }
    }
}

// ---- the recording stub ---------------------------------------------------------------------------

static const float* token_main(int index) {
    return reinterpret_cast<const float*>(static_cast<std::uintptr_t>(0x1000 + index));
}
static const float* token_aside(int index) {
    return reinterpret_cast<const float*>(static_cast<std::uintptr_t>(0x8000 + index));
}
static const float* token_input(void) {
    return reinterpret_cast<const float*>(static_cast<std::uintptr_t>(0x0BAD));
}

struct StubContext {
    const LayerValue* outputs = nullptr;
    // recorded routing per row
    const FeatureBatchView* got_x[kPlanMaxLayers] = {};
    const FeatureBatchView* got_aux[kPlanMaxLayers] = {};
    const FeatureBatchView* got_image[kPlanMaxLayers] = {};
    int calls = 0;
    // recorded releases in order
    int releases[kPlanMaxLayers * 2] = {};
    int release_count = 0;
    // buffer lifetimes: a released entry's main buffer is poisoned, and any
    // later call touching it is flagged (the SplitFfwd aside aliases its
    // input's buffer)
    bool freed[kPlanMaxLayers] = {};
    bool use_after_free = false;
    int aside_owner[kPlanMaxLayers] = {};
    // fault injection
    int fail_at = -1;     // this row's layer returns kInvalidValues
    int violate_at = -1;  // this row's layer violates its output contract
    int violate_mode = 0; // 0: drop/poison the pair flag, 1: wrong main extent
};

static Status stub_layer(void* context, const LayerSpec& spec,
                         const LayerCall& call, LayerValue& out) {
    StubContext* ctx = static_cast<StubContext*>(context);
    const int i = spec.index;
    ctx->got_x[i] = call.x;
    ctx->got_aux[i] = call.aux;
    ctx->got_image[i] = call.image;
    ++ctx->calls;
    auto buffer_of = [ctx](const FeatureBatchView* view) {
        if (view == nullptr) {
            return -1;
        }
        for (int j = 0; j < kPlanMaxLayers; ++j) {
            if (view == &ctx->outputs[j].main) {
                return j;
            }
            if (view == &ctx->outputs[j].aside) {
                return ctx->aside_owner[j];
            }
        }
        return -1;
    };
    const int x_buffer = buffer_of(call.x);
    const int aux_buffer = buffer_of(call.aux);
    if ((x_buffer >= 0 && ctx->freed[x_buffer])
        || (aux_buffer >= 0 && ctx->freed[aux_buffer])) {
        ctx->use_after_free = true;
    }
    if (i == ctx->fail_at) {
        return Status::kInvalidValues;
    }
    out.main.data = token_main(i);
    out.main.channels = 1;
    out.main.rows = spec.out_w;
    out.main.cols = spec.out_h;
    out.main.row_stride = spec.out_h;
    out.main.plane_stride = spec.out_w * spec.out_h;
    ctx->aside_owner[i] = i;
    if (stub_pair(spec.kind, spec.variant)) {
        out.has_aside = true;
        if (spec.kind == LayerKind::kSplitFfwd) {
            // `SplitFfwd` returns its unchanged input as the second element.
            out.aside = call.x != nullptr ? *call.x : LayerValue().main;
            if (out.aside.data == nullptr) {
                out.aside.data = token_aside(i);
            }
            // The second element aliases the input's buffer.
            ctx->aside_owner[i] = x_buffer >= 0 ? x_buffer : i;
        } else {
            out.aside.data = token_aside(i);
            out.aside.channels = 1;
            out.aside.rows = spec.out_w;
            out.aside.cols = spec.out_h;
            out.aside.row_stride = spec.out_h;
            out.aside.plane_stride = spec.out_w * spec.out_h;
        }
    }
    if (i == ctx->violate_at) {
        if (ctx->violate_mode == 0) {
            out.has_aside = !out.has_aside;
            if (!out.has_aside) {
                out.aside = LayerValue().main;
            }
        } else if (ctx->violate_mode == 1) {
            out.main.rows = spec.out_w + 1;
        } else {
            // The pair flag set but the second tensor missing.
            out.has_aside = true;
            out.aside = LayerValue().main;
        }
    }
    return Status::kOk;
}

static void stub_release(void* context, int index) {
    StubContext* ctx = static_cast<StubContext*>(context);
    if (ctx->release_count < kPlanMaxLayers * 2) {
        ctx->releases[ctx->release_count++] = index;
    }
    if (index >= 0 && index < kPlanMaxLayers) {
        ctx->freed[index] = true;
    }
}

// ---- helpers --------------------------------------------------------------------------------------

static FeatureBatchView make_view(float* storage, int rows, int cols) {
    FeatureBatchView view;
    view.data = storage;
    view.channels = 1;
    view.rows = rows;
    view.cols = cols;
    view.row_stride = cols;
    view.plane_stride = rows * cols;
    return view;
}

static void fill_sentinel_value(LayerValue& value) {
    value.main.data = token_input();
    value.main.channels = -1;
    value.main.rows = -1;
    value.main.cols = -1;
    value.aside = value.main;
    value.has_aside = true;
}

static bool same_sentinel_value(const LayerValue& value) {
    return value.main.data == token_input() && value.main.channels == -1
        && value.main.rows == -1 && value.main.cols == -1
        && value.has_aside && value.aside.data == token_input();
}

// The independent walk over one plan: run the dispatch with the recording
// stub and check every row's routing, pair form and release against the
// expectations re-derived from the pinned rules.
static void walk_plan(NetworkKind network, int frame_w, int frame_h) {
    static LayerSpec specs[kPlanMaxLayers];
    static LayerValue outputs[kPlanMaxLayers];
    static int skip_scratch[kPlanMaxLayers];
    static int last_scratch[kPlanMaxLayers];
    int count = 0;
    CHECK_STATUS(build_plan(frame_w, frame_h, network, specs, kPlanMaxLayers, count),
                 Status::kOk);
    CHECK(count == (network == NetworkKind::kHt ? 152 : 76));
    float image_storage[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    FeatureBatchView image = make_view(image_storage, 2, 2);
    StubContext ctx;
    ctx.outputs = outputs;
    NetworkRun run;
    run.image = &image;
    run.outputs = outputs;
    run.skip_sources = skip_scratch;
    run.last_uses = last_scratch;
    run.layer = stub_layer;
    run.release = stub_release;
    run.context = &ctx;
    CHECK_STATUS(network_run(specs, count, run), Status::kOk);
    CHECK(ctx.calls == count);

    // The skip sources, the last uses and the release schedule, re-derived.
    int want_skip[kPlanMaxLayers];
    int want_last[kPlanMaxLayers];
    int want_releases[kPlanMaxLayers * 2];
    int want_release_count = 0;
    for (int i = 0; i < count; ++i) {
        want_skip[i] = expected_skip(specs, count, i);
        CHECK(skip_scratch[i] == want_skip[i]);
    }
    expected_last_uses(want_skip, count, want_last);
    for (int i = 0; i < count; ++i) {
        CHECK(last_scratch[i] == want_last[i]);
    }
    expected_releases(specs, want_last, count, want_releases, want_release_count);
    CHECK(ctx.release_count == want_release_count);
    for (int i = 0; i < want_release_count; ++i) {
        CHECK(ctx.releases[i] == want_releases[i]);
    }
    // No call may touch a released buffer (the SplitFfwd aside aliases its
    // input's buffer, whose release must wait for both holders).
    CHECK(!ctx.use_after_free);

    // Every row's routing, re-derived from the pinned forward contracts.
    for (int i = 0; i < count; ++i) {
        const LayerSpec& spec = specs[i];
        const bool pair = stub_pair(spec.kind, spec.variant);
        CHECK(outputs[i].has_aside == pair);
        CHECK(outputs[i].main.data == token_main(i));
        if (spec.kind == LayerKind::kPreBlock) {
            CHECK(ctx.got_x[i] == nullptr);
            CHECK(ctx.got_aux[i] == nullptr);
        } else {
            CHECK(ctx.got_x[i] == &outputs[i - 1].main);
        }
        if (stub_reads_image(spec.kind)) {
            CHECK(ctx.got_image[i] == &image);
        } else {
            CHECK(ctx.got_image[i] == nullptr);
        }
        const int source = want_skip[i];
        const FeatureBatchView* want_aux = nullptr;
        switch (spec.kind) {
            case LayerKind::kPostBlock:
                CHECK(source >= 0);
                want_aux = &outputs[source].aside;
                break;
            case LayerKind::kSplitFfwdProj:
                // the raw previous pair: `x, skip = x`
                want_aux = outputs[i - 1].has_aside ? &outputs[i - 1].aside : nullptr;
                break;
            case LayerKind::kSplitProj:
            case LayerKind::kSplitProjPool:
            case LayerKind::kVitFfnContract:
            case LayerKind::kVitProjection:
            case LayerKind::kDecInputUpsample:
                CHECK(source >= 0);
                want_aux = &outputs[source].main;
                break;
            case LayerKind::kSplitFinalHead:
                want_aux = (source >= 0 && outputs[source].has_aside)
                    ? &outputs[source].aside : &outputs[i - 1].main;
                break;
            case LayerKind::kSwin1H:
            case LayerKind::kSwin2H:
            case LayerKind::kSwin4H:
            case LayerKind::kSwin8H:
                if (spec.variant == LayerVariant::kUpsample) {
                    CHECK(source >= 0);
                    want_aux = outputs[source].has_aside
                        ? &outputs[source].aside : &outputs[source].main;
                }
                break;
            default:
                break;
        }
        CHECK(ctx.got_aux[i] == want_aux);
        if (spec.kind == LayerKind::kSplitFfwd && i > 0) {
            // the second element is the unchanged block input
            CHECK(outputs[i].aside.data == outputs[i - 1].main.data);
        }
    }
}

int main(void) {
    // ---- the pair rule for every (kind, variant) combination --------------------------------------
    for (int k = 0; k <= static_cast<int>(LayerKind::kPostBlock); ++k) {
        for (int v = 0; v <= static_cast<int>(LayerVariant::kFinalHead); ++v) {
            const LayerKind kind = static_cast<LayerKind>(k);
            const LayerVariant variant = static_cast<LayerVariant>(v);
            CHECK(layer_produces_pair(kind, variant) == stub_pair(kind, variant));
            CHECK(layer_reads_image(kind) == stub_reads_image(kind));
        }
    }

    // ---- the full plans ---------------------------------------------------------------------------
    walk_plan(NetworkKind::kHt, 1280, 720);
    walk_plan(NetworkKind::kSmall, 1280, 720);
    walk_plan(NetworkKind::kHt, 640, 360);

    // ---- mini plans: the edge contracts -----------------------------------------------------------
    float image_storage[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    FeatureBatchView image = make_view(image_storage, 2, 2);

    // (a) a single pre block: pair output, image read, nothing released
    //     (its output leaves through the caller).
    {
        LayerSpec specs[1];
        specs[0] = LayerSpec();
        specs[0].index = 0;
        specs[0].kind = LayerKind::kPreBlock;
        specs[0].skip_block = -1;
        specs[0].out_w = 1;
        specs[0].out_h = 1;
        LayerValue outputs[1];
        int skip_scratch[1];
        int last_scratch[1];
        StubContext ctx;
        ctx.outputs = outputs;
        NetworkRun run;
        run.image = &image;
        run.outputs = outputs;
        run.skip_sources = skip_scratch;
        run.last_uses = last_scratch;
        run.layer = stub_layer;
        run.release = stub_release;
        run.context = &ctx;
        CHECK_STATUS(network_run(specs, 1, run), Status::kOk);
        CHECK(ctx.got_x[0] == nullptr);
        CHECK(ctx.got_aux[0] == nullptr);
        CHECK(ctx.got_image[0] == &image);
        CHECK(outputs[0].has_aside);
        CHECK(last_scratch[0] == 0);
        CHECK(ctx.release_count == 0);
    }

    // (b) a two-row chain: the first row's output is released after the
    //     second runs; the second (final) row's output is never released.
    {
        LayerSpec specs[2];
        specs[0] = LayerSpec();
        specs[0].index = 0;
        specs[0].kind = LayerKind::kPreBlock;
        specs[0].skip_block = -1;
        specs[0].out_w = 1;
        specs[0].out_h = 1;
        specs[1] = LayerSpec();
        specs[1].index = 1;
        specs[1].block = 1;
        specs[1].layer = 0;
        specs[1].kind = LayerKind::kSwin1H;
        specs[1].skip_block = -1;
        specs[1].out_w = 1;
        specs[1].out_h = 1;
        LayerValue outputs[2];
        int skip_scratch[2];
        int last_scratch[2];
        StubContext ctx;
        ctx.outputs = outputs;
        NetworkRun run;
        run.image = &image;
        run.outputs = outputs;
        run.skip_sources = skip_scratch;
        run.last_uses = last_scratch;
        run.layer = stub_layer;
        run.release = stub_release;
        run.context = &ctx;
        CHECK_STATUS(network_run(specs, 2, run), Status::kOk);
        CHECK(ctx.got_x[1] == &outputs[0].main);
        CHECK(ctx.got_aux[1] == nullptr);
        CHECK(last_scratch[0] == 1);
        CHECK(last_scratch[1] == 1);
        CHECK(ctx.release_count == 1);
        CHECK(ctx.releases[0] == 0);
    }

    // (c) the skip span and the two element readers: row 4 (SplitProjPool)
    //     is the skip source of rows 5 (SplitFinalHead, reads the aside) and
    //     6 (DecInputUpsample, reads the main), so row 4 survives both and is
    //     released with row 5 at the final step. Rows 2-4 keep a real split
    //     block layout so the projections' (block, 1) skip resolves.
    {
        LayerSpec specs[7];
        specs[0] = LayerSpec();
        specs[0].index = 0;
        specs[0].kind = LayerKind::kPreBlock;
        specs[0].skip_block = -1;
        specs[0].out_w = 1;
        specs[0].out_h = 1;
        specs[1] = LayerSpec();
        specs[1].index = 1;
        specs[1].block = 2;
        specs[1].layer = 0;
        specs[1].kind = LayerKind::kSwin2H;
        specs[1].skip_block = -1;
        specs[1].out_w = 1;
        specs[1].out_h = 1;
        specs[2] = LayerSpec();
        specs[2].index = 2;
        specs[2].block = 3;
        specs[2].layer = 1;
        specs[2].kind = LayerKind::kSplitFfwdProj;
        specs[2].skip_block = -1;
        specs[2].out_w = 1;
        specs[2].out_h = 1;
        specs[3] = LayerSpec();
        specs[3].index = 3;
        specs[3].block = 3;
        specs[3].layer = 2;
        specs[3].kind = LayerKind::kSplitQKVAttn;
        specs[3].skip_block = -1;
        specs[3].out_w = 1;
        specs[3].out_h = 1;
        specs[4] = LayerSpec();
        specs[4].index = 4;
        specs[4].block = 3;
        specs[4].layer = 3;
        specs[4].kind = LayerKind::kSplitProjPool;
        specs[4].skip_block = -1;
        specs[4].out_w = 1;
        specs[4].out_h = 1;
        specs[5] = LayerSpec();
        specs[5].index = 5;
        specs[5].block = 3;
        specs[5].layer = 4;
        specs[5].kind = LayerKind::kSplitFinalHead;
        specs[5].skip_block = -1;
        specs[5].out_w = 1;
        specs[5].out_h = 1;
        specs[6] = LayerSpec();
        specs[6].index = 6;
        specs[6].block = 9;
        specs[6].layer = 0;
        specs[6].kind = LayerKind::kDecInputUpsample;
        specs[6].skip_block = 3;
        specs[6].out_w = 1;
        specs[6].out_h = 1;
        LayerValue outputs[7];
        int skip_scratch[7];
        int last_scratch[7];
        StubContext ctx;
        ctx.outputs = outputs;
        NetworkRun run;
        run.image = &image;
        run.outputs = outputs;
        run.skip_sources = skip_scratch;
        run.last_uses = last_scratch;
        run.layer = stub_layer;
        run.release = stub_release;
        run.context = &ctx;
        CHECK_STATUS(network_run(specs, 7, run), Status::kOk);
        CHECK(!ctx.use_after_free);
        CHECK(skip_scratch[4] == 2);  // the pool reads its block's layer 1
        CHECK(skip_scratch[5] == 4);  // the final head reads (3, 3) = row 4
        CHECK(skip_scratch[6] == 4);  // the decoder entry reads (3, 3) = row 4
        CHECK(ctx.got_aux[4] == &outputs[2].main);
        CHECK(ctx.got_aux[5] == &outputs[4].aside);
        CHECK(ctx.got_aux[6] == &outputs[4].main);
        CHECK(last_scratch[2] == 4);  // read by row 4 as its skip
        CHECK(last_scratch[4] == 6);  // read by rows 5 and 6
        CHECK(last_scratch[5] == 6);
        CHECK(ctx.release_count == 6);
        CHECK(ctx.releases[0] == 0);
        CHECK(ctx.releases[1] == 1);
        CHECK(ctx.releases[2] == 2);  // released after row 4 (its skip reader)
        CHECK(ctx.releases[3] == 3);
        CHECK(ctx.releases[4] == 4);  // ascending at the final step
        CHECK(ctx.releases[5] == 5);
    }

    // (d) the upsample element fallbacks: with a pair skip source it reads
    //     the aside (`skip[1]`), with a single skip source the whole skip.
    {
        LayerSpec specs[3];
        specs[0] = LayerSpec();
        specs[0].index = 0;
        specs[0].kind = LayerKind::kPreBlock;
        specs[0].skip_block = -1;
        specs[0].out_w = 1;
        specs[0].out_h = 1;
        specs[1] = LayerSpec();
        specs[1].index = 1;
        specs[1].block = 4;
        specs[1].layer = 0;
        specs[1].kind = LayerKind::kSwin4H;
        specs[1].variant = LayerVariant::kDs;
        specs[1].skip_block = -1;
        specs[1].out_w = 1;
        specs[1].out_h = 1;
        specs[2] = LayerSpec();
        specs[2].index = 2;
        specs[2].block = 7;
        specs[2].layer = 0;
        specs[2].kind = LayerKind::kSwin4H;
        specs[2].variant = LayerVariant::kUpsample;
        specs[2].skip_block = 4;
        specs[2].out_w = 1;
        specs[2].out_h = 1;
        LayerValue outputs[3];
        int skip_scratch[3];
        int last_scratch[3];
        StubContext ctx;
        ctx.outputs = outputs;
        NetworkRun run;
        run.image = &image;
        run.outputs = outputs;
        run.skip_sources = skip_scratch;
        run.last_uses = last_scratch;
        run.layer = stub_layer;
        run.release = stub_release;
        run.context = &ctx;
        CHECK_STATUS(network_run(specs, 3, run), Status::kOk);
        CHECK(!ctx.use_after_free);
        CHECK(skip_scratch[2] == 1);
        CHECK(ctx.got_aux[2] == &outputs[1].aside);
        specs[1].variant = LayerVariant::kNone;  // now a single skip source
        StubContext ctx2;
        ctx2.outputs = outputs;
        NetworkRun run2 = run;
        run2.context = &ctx2;
        CHECK_STATUS(network_run(specs, 3, run2), Status::kOk);
        CHECK(ctx2.got_aux[2] == &outputs[1].main);
    }

    // (e) SplitFfwdProj reads the raw pair of the previous row (SplitFfwd's
    //     `(out, x)`); with a single previous there is no second input (the
    //     pinned keeps `skip` None).
    {
        LayerSpec specs[4];
        specs[0] = LayerSpec();
        specs[0].index = 0;
        specs[0].kind = LayerKind::kPreBlock;
        specs[0].skip_block = -1;
        specs[0].out_w = 1;
        specs[0].out_h = 1;
        specs[1] = LayerSpec();
        specs[1].index = 1;
        specs[1].block = 1;
        specs[1].layer = 0;
        specs[1].kind = LayerKind::kSwin1H;
        specs[1].skip_block = -1;
        specs[1].out_w = 1;
        specs[1].out_h = 1;
        specs[2] = LayerSpec();
        specs[2].index = 2;
        specs[2].block = 23;
        specs[2].layer = 0;
        specs[2].kind = LayerKind::kSplitFfwd;
        specs[2].skip_block = -1;
        specs[2].out_w = 1;
        specs[2].out_h = 1;
        specs[3] = LayerSpec();
        specs[3].index = 3;
        specs[3].block = 23;
        specs[3].layer = 1;
        specs[3].kind = LayerKind::kSplitFfwdProj;
        specs[3].skip_block = -1;
        specs[3].out_w = 1;
        specs[3].out_h = 1;
        LayerValue outputs[4];
        int skip_scratch[4];
        int last_scratch[4];
        StubContext ctx;
        ctx.outputs = outputs;
        NetworkRun run;
        run.image = &image;
        run.outputs = outputs;
        run.skip_sources = skip_scratch;
        run.last_uses = last_scratch;
        run.layer = stub_layer;
        run.release = stub_release;
        run.context = &ctx;
        CHECK_STATUS(network_run(specs, 4, run), Status::kOk);
        CHECK(ctx.got_aux[3] == &outputs[2].aside);
        // SplitFfwd's second element is its unchanged input (the alias).
        CHECK(outputs[2].aside.data == outputs[1].main.data);
        CHECK(!ctx.use_after_free);
        // Row 1's buffer is aliased by row 2's second element, so its release
        // waits for row 2's entry (both fire at the final step).
        CHECK(ctx.release_count == 3);
        CHECK(ctx.releases[0] == 0);
        CHECK(ctx.releases[1] == 1);
        CHECK(ctx.releases[2] == 2);
        // A single previous row: no second input.
        specs[2].kind = LayerKind::kSwin1H;
        StubContext ctx2;
        ctx2.outputs = outputs;
        NetworkRun run2 = run;
        run2.context = &ctx2;
        CHECK_STATUS(network_run(specs, 4, run2), Status::kOk);
        CHECK(ctx2.got_aux[3] == nullptr);
    }

    // (f) the final-head fallback: a single skip source sends its own main
    //     (`pooled = skip[1] if isinstance(skip, tuple) else x`).
    {
        LayerSpec specs[4];
        specs[0] = LayerSpec();
        specs[0].index = 0;
        specs[0].kind = LayerKind::kPreBlock;
        specs[0].skip_block = -1;
        specs[0].out_w = 1;
        specs[0].out_h = 1;
        specs[1] = LayerSpec();
        specs[1].index = 1;
        specs[1].block = 3;
        specs[1].layer = 3;
        specs[1].kind = LayerKind::kSwin1H;
        specs[1].skip_block = -1;
        specs[1].out_w = 1;
        specs[1].out_h = 1;
        specs[2] = LayerSpec();
        specs[2].index = 2;
        specs[2].block = 5;
        specs[2].layer = 0;
        specs[2].kind = LayerKind::kSwin1H;
        specs[2].skip_block = -1;
        specs[2].out_w = 1;
        specs[2].out_h = 1;
        specs[3] = LayerSpec();
        specs[3].index = 3;
        specs[3].block = 3;
        specs[3].layer = 4;
        specs[3].kind = LayerKind::kSplitFinalHead;
        specs[3].skip_block = -1;
        specs[3].out_w = 1;
        specs[3].out_h = 1;
        LayerValue outputs[4];
        int skip_scratch[4];
        int last_scratch[4];
        StubContext ctx;
        ctx.outputs = outputs;
        NetworkRun run;
        run.image = &image;
        run.outputs = outputs;
        run.skip_sources = skip_scratch;
        run.last_uses = last_scratch;
        run.layer = stub_layer;
        run.release = stub_release;
        run.context = &ctx;
        CHECK_STATUS(network_run(specs, 4, run), Status::kOk);
        CHECK(skip_scratch[3] == 1);  // (3, 3) = row 1
        // The fallback is x (the previous row's main), not the skip's.
        CHECK(ctx.got_aux[3] == &outputs[2].main);
    }

    // (g) capture mode: a null release callback keeps every output.
    {
        LayerSpec specs[2];
        specs[0] = LayerSpec();
        specs[0].index = 0;
        specs[0].kind = LayerKind::kPreBlock;
        specs[0].skip_block = -1;
        specs[0].out_w = 1;
        specs[0].out_h = 1;
        specs[1] = LayerSpec();
        specs[1].index = 1;
        specs[1].block = 1;
        specs[1].layer = 0;
        specs[1].kind = LayerKind::kSwin1H;
        specs[1].skip_block = -1;
        specs[1].out_w = 1;
        specs[1].out_h = 1;
        LayerValue outputs[2];
        int skip_scratch[2];
        int last_scratch[2];
        StubContext ctx;
        ctx.outputs = outputs;
        NetworkRun run;
        run.image = &image;
        run.outputs = outputs;
        run.skip_sources = skip_scratch;
        run.last_uses = last_scratch;
        run.layer = stub_layer;
        run.release = nullptr;
        run.context = &ctx;
        CHECK_STATUS(network_run(specs, 2, run), Status::kOk);
        CHECK(ctx.release_count == 0);
    }

    // ---- rejection matrix: upfront classes leave outputs and scratch untouched ---------------------
    LayerSpec specs[3];
    for (int i = 0; i < 3; ++i) {
        specs[i] = LayerSpec();
        specs[i].index = i;
        specs[i].block = i;
        specs[i].layer = 0;
        specs[i].kind = LayerKind::kSwin1H;
        specs[i].skip_block = -1;
        specs[i].out_w = 1;
        specs[i].out_h = 1;
    }
    specs[0].kind = LayerKind::kPreBlock;
    LayerValue outputs[3];
    int skip_scratch[3];
    int last_scratch[3];
    for (int i = 0; i < 3; ++i) {
        fill_sentinel_value(outputs[i]);
        skip_scratch[i] = -777;
        last_scratch[i] = -777;
    }
    StubContext ctx;
    ctx.outputs = outputs;
    NetworkRun run;
    run.image = &image;
    run.outputs = outputs;
    run.skip_sources = skip_scratch;
    run.last_uses = last_scratch;
    run.layer = stub_layer;
    run.release = stub_release;
    run.context = &ctx;

    CHECK_STATUS(network_run(nullptr, 3, run), Status::kNullPointer);
    {
        NetworkRun bad = run;
        bad.image = nullptr;
        CHECK_STATUS(network_run(specs, 3, bad), Status::kNullPointer);
        bad = run;
        bad.outputs = nullptr;
        CHECK_STATUS(network_run(specs, 3, bad), Status::kNullPointer);
        bad = run;
        bad.skip_sources = nullptr;
        CHECK_STATUS(network_run(specs, 3, bad), Status::kNullPointer);
        bad = run;
        bad.last_uses = nullptr;
        CHECK_STATUS(network_run(specs, 3, bad), Status::kNullPointer);
        bad = run;
        bad.layer = nullptr;
        CHECK_STATUS(network_run(specs, 3, bad), Status::kNullPointer);
    }
    CHECK_STATUS(network_run(specs, 0, run), Status::kInvalidDimensions);
    CHECK_STATUS(network_run(specs, kPlanMaxLayers + 1, run),
                 Status::kInvalidDimensions);
    {
        FeatureBatchView empty = image;
        empty.data = nullptr;
        NetworkRun bad = run;
        bad.image = &empty;
        CHECK_STATUS(network_run(specs, 3, bad), Status::kInvalidValues);
        empty = image;
        empty.rows = 0;
        bad.image = &empty;
        CHECK_STATUS(network_run(specs, 3, bad), Status::kInvalidValues);
        empty = image;
        empty.cols = 0;
        bad.image = &empty;
        CHECK_STATUS(network_run(specs, 3, bad), Status::kInvalidValues);
        empty = image;
        empty.channels = 0;
        bad.image = &empty;
        CHECK_STATUS(network_run(specs, 3, bad), Status::kInvalidValues);
    }
    {
        LayerSpec bad_specs[3];
        for (int i = 0; i < 3; ++i) {
            bad_specs[i] = specs[i];
        }
        bad_specs[1].index = 2;  // not in plan order
        CHECK_STATUS(network_run(bad_specs, 3, run), Status::kInvalidValues);
        for (int i = 0; i < 3; ++i) {
            bad_specs[i] = specs[i];
        }
        bad_specs[1].kind = static_cast<LayerKind>(19);
        CHECK_STATUS(network_run(bad_specs, 3, run), Status::kInvalidValues);
        for (int i = 0; i < 3; ++i) {
            bad_specs[i] = specs[i];
        }
        bad_specs[1].kind = static_cast<LayerKind>(-1);
        CHECK_STATUS(network_run(bad_specs, 3, run), Status::kInvalidValues);
        for (int i = 0; i < 3; ++i) {
            bad_specs[i] = specs[i];
        }
        bad_specs[1].variant = static_cast<LayerVariant>(8);
        CHECK_STATUS(network_run(bad_specs, 3, run), Status::kInvalidValues);
        for (int i = 0; i < 3; ++i) {
            bad_specs[i] = specs[i];
        }
        bad_specs[1].variant = static_cast<LayerVariant>(-1);
        CHECK_STATUS(network_run(bad_specs, 3, run), Status::kInvalidValues);
        for (int i = 0; i < 3; ++i) {
            bad_specs[i] = specs[i];
        }
        bad_specs[2].out_w = 0;
        CHECK_STATUS(network_run(bad_specs, 3, run), Status::kInvalidDimensions);
    }
    for (int i = 0; i < 3; ++i) {
        CHECK(same_sentinel_value(outputs[i]));
        CHECK(skip_scratch[i] == -777);
        CHECK(last_scratch[i] == -777);
    }
    CHECK(ctx.calls == 0);
    CHECK(ctx.release_count == 0);

    // ---- mid-run rejections: fail fast, outputs written so far kept -------------------------------
    // (a) the layer's own error stops the run.
    {
        for (int i = 0; i < 3; ++i) {
            fill_sentinel_value(outputs[i]);
        }
        StubContext ctx2;
        ctx2.outputs = outputs;
        ctx2.fail_at = 1;
        NetworkRun run2 = run;
        run2.context = &ctx2;
        CHECK_STATUS(network_run(specs, 3, run2), Status::kInvalidValues);
        CHECK(outputs[0].main.data == token_main(0));
        // The failing row's entry is the empty value its layer was handed;
        // rows not reached stay sentinel-untouched.
        CHECK(outputs[1].main.data == nullptr);
        CHECK(!outputs[1].has_aside);
        CHECK(same_sentinel_value(outputs[2]));
    }
    // (b) a pair contract violation (the ds row drops its aside) is caught
    //     before the value is routed on.
    {
        LayerSpec pair_specs[2];
        pair_specs[0] = specs[0];
        pair_specs[1] = specs[1];
        pair_specs[1].variant = LayerVariant::kDs;
        pair_specs[1].kind = LayerKind::kSwin1H;
        pair_specs[1].block = 1;
        for (int i = 0; i < 3; ++i) {
            fill_sentinel_value(outputs[i]);
        }
        StubContext ctx2;
        ctx2.outputs = outputs;
        ctx2.violate_at = 1;
        NetworkRun run2 = run;
        run2.context = &ctx2;
        CHECK_STATUS(network_run(pair_specs, 2, run2), Status::kInvalidValues);
        CHECK(outputs[0].main.data == token_main(0));
    }
    // (c) a wrong main extent is caught the same way.
    {
        LayerSpec pair_specs[2];
        pair_specs[0] = specs[0];
        pair_specs[1] = specs[1];
        for (int i = 0; i < 3; ++i) {
            fill_sentinel_value(outputs[i]);
        }
        StubContext ctx2;
        ctx2.outputs = outputs;
        ctx2.violate_at = 1;
        ctx2.violate_mode = 1;
        NetworkRun run2 = run;
        run2.context = &ctx2;
        CHECK_STATUS(network_run(pair_specs, 2, run2), Status::kInvalidValues);
    }
    // (d) a post block whose plan supplies no pair skip source is rejected
    //     before its layer runs.
    {
        LayerSpec post_specs[2];
        post_specs[0] = specs[0];
        post_specs[1] = specs[1];
        post_specs[1].kind = LayerKind::kPostBlock;
        post_specs[1].block = 70;
        post_specs[1].layer = 0;
        post_specs[1].skip_block = -1;
        for (int i = 0; i < 3; ++i) {
            fill_sentinel_value(outputs[i]);
        }
        StubContext ctx2;
        ctx2.outputs = outputs;
        NetworkRun run2 = run;
        run2.context = &ctx2;
        CHECK_STATUS(network_run(post_specs, 2, run2), Status::kInvalidValues);
        CHECK(ctx2.calls == 1);  // only the pre block ran
    }
    // (e) an upsample with no skip source is rejected before its layer runs.
    {
        LayerSpec up_specs[2];
        up_specs[0] = specs[0];
        up_specs[1] = specs[1];
        up_specs[1].kind = LayerKind::kSwin1H;
        up_specs[1].variant = LayerVariant::kUpsample;
        up_specs[1].skip_block = -1;
        for (int i = 0; i < 3; ++i) {
            fill_sentinel_value(outputs[i]);
        }
        StubContext ctx2;
        ctx2.outputs = outputs;
        NetworkRun run2 = run;
        run2.context = &ctx2;
        CHECK_STATUS(network_run(up_specs, 2, run2), Status::kInvalidValues);
        CHECK(ctx2.calls == 1);
    }
    // (f) a SplitFfwdProj as the first row has no previous pair.
    {
        LayerSpec proj_specs[1];
        proj_specs[0] = specs[1];
        proj_specs[0].index = 0;
        proj_specs[0].kind = LayerKind::kSplitFfwdProj;
        proj_specs[0].skip_block = -1;
        for (int i = 0; i < 3; ++i) {
            fill_sentinel_value(outputs[i]);
        }
        StubContext ctx2;
        ctx2.outputs = outputs;
        NetworkRun run2 = run;
        run2.context = &ctx2;
        CHECK_STATUS(network_run(proj_specs, 1, run2), Status::kInvalidValues);
        CHECK(ctx2.calls == 0);
    }
    // (g) a PostBlock as the first row has no previous at all.
    {
        LayerSpec post_specs[1];
        post_specs[0] = specs[1];
        post_specs[0].index = 0;
        post_specs[0].kind = LayerKind::kPostBlock;
        post_specs[0].skip_block = -1;
        for (int i = 0; i < 3; ++i) {
            fill_sentinel_value(outputs[i]);
        }
        StubContext ctx2;
        ctx2.outputs = outputs;
        NetworkRun run2 = run;
        run2.context = &ctx2;
        CHECK_STATUS(network_run(post_specs, 1, run2), Status::kInvalidValues);
        CHECK(ctx2.calls == 0);
    }
    // (h) the skip-main group with no skip source is rejected before its
    //     layer runs (SplitProj needs its block's layer 1).
    {
        LayerSpec main_specs[2];
        main_specs[0] = specs[0];
        main_specs[1] = specs[1];
        main_specs[1].kind = LayerKind::kSplitProj;
        main_specs[1].block = 30;
        main_specs[1].layer = 3;
        main_specs[1].skip_block = -1;
        for (int i = 0; i < 3; ++i) {
            fill_sentinel_value(outputs[i]);
        }
        StubContext ctx2;
        ctx2.outputs = outputs;
        NetworkRun run2 = run;
        run2.context = &ctx2;
        CHECK_STATUS(network_run(main_specs, 2, run2), Status::kInvalidValues);
        CHECK(ctx2.calls == 1);
    }
    // (i) a pair flag with the second tensor missing is caught.
    {
        LayerSpec pair_specs[2];
        pair_specs[0] = specs[0];
        pair_specs[1] = specs[1];
        pair_specs[1].kind = LayerKind::kSwin1H;
        pair_specs[1].variant = LayerVariant::kDs;
        for (int i = 0; i < 3; ++i) {
            fill_sentinel_value(outputs[i]);
        }
        StubContext ctx2;
        ctx2.outputs = outputs;
        ctx2.violate_at = 1;
        ctx2.violate_mode = 2;
        NetworkRun run2 = run;
        run2.context = &ctx2;
        CHECK_STATUS(network_run(pair_specs, 2, run2), Status::kInvalidValues);
    }
    // (j) a skip source that is the row itself or a later row is rejected
    //     (the pinned would read a not-yet-produced tensor).
    {
        LayerSpec self_specs[2];
        self_specs[0] = specs[0];
        self_specs[1] = specs[1];
        self_specs[1].kind = LayerKind::kSwin1H;
        self_specs[1].variant = LayerVariant::kUpsample;
        self_specs[1].block = 8;
        self_specs[1].layer = 0;
        self_specs[1].skip_block = 8;  // (8, 0) resolves to the row itself
        for (int i = 0; i < 3; ++i) {
            fill_sentinel_value(outputs[i]);
        }
        StubContext ctx2;
        ctx2.outputs = outputs;
        NetworkRun run2 = run;
        run2.context = &ctx2;
        CHECK_STATUS(network_run(self_specs, 2, run2), Status::kInvalidValues);
        CHECK(ctx2.calls == 0);  // rejected before the first layer call
        CHECK(same_sentinel_value(outputs[0]));
        CHECK(same_sentinel_value(outputs[1]));
    }
    {
        // Row 1's skip (9, 0) resolves to row 2: a source that has not run.
        LayerSpec future_specs[3];
        future_specs[0] = specs[0];
        future_specs[1] = specs[1];
        future_specs[1].kind = LayerKind::kSwin1H;
        future_specs[1].variant = LayerVariant::kUpsample;
        future_specs[1].block = 8;
        future_specs[1].layer = 0;
        future_specs[1].skip_block = 9;
        future_specs[2] = specs[2];
        future_specs[2].block = 9;
        future_specs[2].layer = 0;
        future_specs[2].kind = LayerKind::kSwin1H;
        for (int i = 0; i < 3; ++i) {
            fill_sentinel_value(outputs[i]);
        }
        StubContext ctx2;
        ctx2.outputs = outputs;
        NetworkRun run2 = run;
        run2.context = &ctx2;
        CHECK_STATUS(network_run(future_specs, 3, run2), Status::kInvalidValues);
        CHECK(ctx2.calls == 0);  // rejected before the first layer call
    }

    // ---- determinism and zero-allocation accounting -----------------------------------------------
    {
        static LayerSpec plan_specs[kPlanMaxLayers];
        static LayerValue run_a[kPlanMaxLayers];
        static LayerValue run_b[kPlanMaxLayers];
        static int skip_a[kPlanMaxLayers];
        static int last_a[kPlanMaxLayers];
        static int skip_b[kPlanMaxLayers];
        static int last_b[kPlanMaxLayers];
        int count = 0;
        CHECK_STATUS(build_plan(640, 360, NetworkKind::kHt, plan_specs,
                                kPlanMaxLayers, count), Status::kOk);
        StubContext ctx_a;
        ctx_a.outputs = run_a;
        StubContext ctx_b;
        ctx_b.outputs = run_b;
        NetworkRun run_a_desc;
        run_a_desc.image = &image;
        run_a_desc.outputs = run_a;
        run_a_desc.skip_sources = skip_a;
        run_a_desc.last_uses = last_a;
        run_a_desc.layer = stub_layer;
        run_a_desc.release = stub_release;
        run_a_desc.context = &ctx_a;
        NetworkRun run_b_desc = run_a_desc;
        run_b_desc.outputs = run_b;
        run_b_desc.skip_sources = skip_b;
        run_b_desc.last_uses = last_b;
        run_b_desc.context = &ctx_b;
        const unsigned long long before = g_allocations.load();
        CHECK_STATUS(network_run(plan_specs, count, run_a_desc), Status::kOk);
        CHECK_STATUS(network_run(plan_specs, count, run_b_desc), Status::kOk);
        CHECK(g_allocations.load() == before);
        CHECK(std::memcmp(skip_a, skip_b, sizeof(skip_a)) == 0);
        CHECK(std::memcmp(last_a, last_b, sizeof(last_a)) == 0);
        CHECK(ctx_a.release_count == ctx_b.release_count);
        CHECK(std::memcmp(ctx_a.releases, ctx_b.releases, sizeof(ctx_a.releases)) == 0);
        CHECK(ctx_a.calls == ctx_b.calls);
    }

    std::printf("native-network: %lld checks passed\n", g_checks);
    return 0;
}
