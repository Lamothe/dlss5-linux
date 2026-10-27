// Independent known-answer tests for vr/native-eval.hpp — the layer binding
// and synthetic evaluator (eval_layer over the pinned model.py layer
// contracts). Hand-derived answers for the frame/planar/token transposes, the
// record derivations and the linear binding kinds; record-consumption sets
// for the heavy kinds (their arithmetic is pinned by their own family
// suites); an end-to-end zero-signal invariant through network_run over a
// real plan; rejection classes with sentinel-untouched outputs; determinism
// and zero-allocation accounting. No Python, no torch, no proprietary
// weights.

#include "native-eval.hpp"

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

// ---- the synthetic record store -------------------------------------------------------------------
// Records are keyed (block, layer, name) exactly like the pinned `Weights`.
// The callbacks use exact-count/shape semantics (a request is answered only
// when it matches the stored shape); count -1 is the wildcard form and
// answers whenever the record exists. Requests are logged so the tests can
// pin which records each kind consumes.

struct Record {
    int block = 0;
    int layer = 0;
    const char* name = nullptr;
    int rows = 0;   // e4m3/ffwd
    int cols = 0;
    int count = 0;  // f16/f32
    int which = -1;
    const std::uint8_t* codes = nullptr;
    const float* values = nullptr;
};

struct Request {
    int block = 0;
    int layer = 0;
    char name[24] = {};
    int rows = 0;
    int cols = 0;
    int count = 0;
    int which = -2;
};

struct Store {
    const Record* records = nullptr;
    int record_count = 0;
    Request log[256] = {};
    int requests = 0;
    bool serve_any = false;  // generative zero records for the end-to-end run
    // Generous zero storage: the largest requests are the split qkv
    // ([1536, 512] codes) and the split bias table ([16, 4096] floats).
    std::uint8_t zero_codes[1 << 22] = {};  // the vit ffn weight is [4096, 1024]
    float zero_values[1 << 17] = {};
};

static void log_request(Store* s, int block, int layer, const char* name,
                        int rows, int cols, int count, int which) {
    // One entry per (name, which): the binding's wildcard length probe makes
    // thousands of exact-count requests for a single record.
    for (int i = 0; i < s->requests; ++i) {
        if (std::strcmp(s->log[i].name, name) == 0
            && s->log[i].which == which) {
            return;
        }
    }
    if (s->requests < 256) {
        Request& r = s->log[s->requests++];
        r.block = block;
        r.layer = layer;
        std::snprintf(r.name, sizeof r.name, "%s", name);
        r.rows = rows;
        r.cols = cols;
        r.count = count;
        r.which = which;
    }
}

static const Record* find_record(Store* s, int block, int layer,
                                 const char* name, int which) {
    for (int i = 0; i < s->record_count; ++i) {
        const Record& r = s->records[i];
        if (r.block == block && r.layer == layer
            && std::strcmp(r.name, name) == 0 && r.which == which) {
            return &r;
        }
    }
    return nullptr;
}

static const std::uint8_t* store_e4m3(void* context, int block, int layer,
                                      const char* name, int rows, int cols) {
    Store* s = static_cast<Store*>(context);
    log_request(s, block, layer, name, rows, cols, rows * cols, -2);
    const Record* r = find_record(s, block, layer, name, -2);
    if (r != nullptr) {
        return r->rows == rows && r->cols == cols ? r->codes : nullptr;
    }
    return s->serve_any ? s->zero_codes : nullptr;
}

// The wildcard-record names (the pinned constructors slice these records),
// with the fixed synthetic lengths the generative store serves them at.
static bool wildcard_name(const char* name, int& length) {
    if (std::strcmp(name, "upsample_gain") == 0) {
        length = 16;  // shorter than any wide row: exercises the tail fallback
        return true;
    }
    if (std::strcmp(name, "residual_scale") == 0
        || std::strcmp(name, "attn_residual_scale") == 0
        || std::strcmp(name, "scalars_b") == 0) {
        length = 4096;
        return true;
    }
    return false;
}

static const float* store_f16(void* context, int block, int layer,
                              const char* name, int count) {
    Store* s = static_cast<Store*>(context);
    log_request(s, block, layer, name, 0, 0, count, -3);
    const Record* r = find_record(s, block, layer, name, -3);
    if (r != nullptr) {
        return count == -1 || count == r->count ? r->values : nullptr;
    }
    if (s->serve_any) {
        int length = 0;
        if (wildcard_name(name, length)) {
            return count == -1 || count == length ? s->zero_values : nullptr;
        }
        return s->zero_values;
    }
    return nullptr;
}

static const float* store_f32(void* context, int block, int layer,
                              const char* name, int count) {
    Store* s = static_cast<Store*>(context);
    log_request(s, block, layer, name, 0, 0, count, -4);
    const Record* r = find_record(s, block, layer, name, -4);
    if (r != nullptr) {
        return count == -1 || count == r->count ? r->values : nullptr;
    }
    if (s->serve_any) {
        int length = 0;
        if (wildcard_name(name, length)) {
            return count == -1 || count == length ? s->zero_values : nullptr;
        }
        return s->zero_values;
    }
    return nullptr;
}

static const std::uint8_t* store_ffwd(void* context, int block, int layer,
                                      int which, int rows, int cols) {
    Store* s = static_cast<Store*>(context);
    log_request(s, block, layer, "ffwd", rows, cols, rows * cols, which);
    if (s->serve_any) {
        return s->zero_codes;
    }
    const Record* r = find_record(s, block, layer, "ffwd", which);
    if (r == nullptr || r->rows != rows || r->cols != cols) {
        return nullptr;
    }
    return r->codes;
}

static void bind_store(EvalWeights& w, Store& s) {
    w.context = &s;
    w.e4m3 = store_e4m3;
    w.f16 = store_f16;
    w.f32 = store_f32;
    w.ffwd = store_ffwd;
}

static bool saw_request(Store& s, const char* name, int rows, int cols,
                        int which) {
    for (int i = 0; i < s.requests; ++i) {
        if (std::strcmp(s.log[i].name, name) == 0 && s.log[i].rows == rows
            && s.log[i].cols == cols && s.log[i].which == which) {
            return true;
        }
    }
    return false;
}

// ---- helpers --------------------------------------------------------------------------------------

static constexpr float kSentinel = -23456.5f;

static FeatureBatchView make_view(float* storage, int channels, int rows,
                                  int cols) {
    FeatureBatchView view;
    view.data = storage;
    view.channels = channels;
    view.rows = rows;
    view.cols = cols;
    view.row_stride = cols;
    view.plane_stride = static_cast<std::ptrdiff_t>(rows) * cols;
    return view;
}

static LayerSpec row(int index, int block, int layer, LayerKind kind,
                     int c_in, int c_out, int out_w, int out_h,
                     int channels, int heads) {
    LayerSpec spec;
    spec.index = index;
    spec.block = block;
    spec.layer = layer;
    spec.kind = kind;
    spec.c_in = c_in;
    spec.c_out = c_out;
    spec.out_w = out_w;
    spec.out_h = out_h;
    spec.channels = channels;
    spec.heads = heads;
    spec.skip_block = -1;
    return spec;
}

int main(void) {
    // ---- frame/planar/token transposes (hand-derived) ---------------------------------------------
    {
        // Planar [2, 2, 3] -> tokens [6][2]: token = r * cols + col.
        const float planar[12] = {0.0f, 1.0f, 2.0f,   3.0f, 4.0f, 5.0f,
                                  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f};
        FeatureBatchView src = make_view(const_cast<float*>(planar), 2, 2, 3);
        float tokens[12];
        CHECK_STATUS(eval_planar_to_tokens(src, tokens, 12), Status::kOk);
        const float want_tokens[12] = {0.0f, 10.0f, 1.0f, 11.0f, 2.0f, 12.0f,
                                       3.0f, 13.0f, 4.0f, 14.0f, 5.0f, 15.0f};
        for (int i = 0; i < 12; ++i) {
            CHECK(tokens[i] == want_tokens[i]);
        }
        float back[12];
        CHECK_STATUS(eval_tokens_to_planar(tokens, 12, 2, 2, 3, back, 12),
                     Status::kOk);
        for (int i = 0; i < 12; ++i) {
            CHECK(back[i] == planar[i]);
        }
        CHECK_STATUS(eval_planar_to_tokens(src, tokens, 11),
                     Status::kInvalidBuffer);
    }
    {
        // Stride-aware read: the same logical map with padded strides
        // (plane 1's rows start at 10 and 14; 8-9, 13, 17-19 are padding).
        const float padded[20] = {0.0f, 1.0f, 2.0f, -1.0f,
                                  3.0f, 4.0f, 5.0f, -1.0f,
                                  -1.0f, -1.0f,
                                  10.0f, 11.0f, 12.0f, -1.0f,
                                  13.0f, 14.0f, 15.0f, -1.0f,
                                  -1.0f, -1.0f};
        FeatureBatchView src;
        src.data = padded;
        src.channels = 2;
        src.rows = 2;
        src.cols = 3;
        src.row_stride = 4;
        src.plane_stride = 10;
        float tokens[12];
        CHECK_STATUS(eval_planar_to_tokens(src, tokens, 12), Status::kOk);
        const float want_tokens[12] = {0.0f, 10.0f, 1.0f, 11.0f, 2.0f, 12.0f,
                                       3.0f, 13.0f, 4.0f, 14.0f, 5.0f, 15.0f};
        for (int i = 0; i < 12; ++i) {
            CHECK(tokens[i] == want_tokens[i]);
        }
    }
    {
        // The frame boundary: planar [3, 2, 2] <-> interleaved 2x2.
        const float planar[12] = {1.0f, 2.0f, 3.0f, 4.0f,
                                  11.0f, 12.0f, 13.0f, 14.0f,
                                  21.0f, 22.0f, 23.0f, 24.0f};
        FeatureBatchView src = make_view(const_cast<float*>(planar), 3, 2, 2);
        float frame[12];
        CHECK_STATUS(eval_detail::frame_from_planar(src, frame, 12), Status::kOk);
        const float want_frame[12] = {1.0f, 11.0f, 21.0f, 2.0f, 12.0f, 22.0f,
                                      3.0f, 13.0f, 23.0f, 4.0f, 14.0f, 24.0f};
        for (int i = 0; i < 12; ++i) {
            CHECK(frame[i] == want_frame[i]);
        }
        float back[12];
        CHECK_STATUS(eval_detail::frame_to_planar(frame, 12, 2, 2, back, 12), Status::kOk);
        for (int i = 0; i < 12; ++i) {
            CHECK(back[i] == planar[i]);
        }
    }

    // ---- the record derivations (hand-derived) ----------------------------------------------------
    {
        // `_residual_table` content rule: a zero front eight shifts the
        // slice to [8, 8 + C); any non-zero front entry keeps [0, C).
        const float padded[12] = {0, 0, 0, 0, 0, 0, 0, 0, 5, 6, 7, 8};
        const float* out = nullptr;
        CHECK_STATUS(eval_detail::mlp_scale_slice(padded, 12, 4, out),
                     Status::kOk);
        CHECK(out == padded + 8 && out[0] == 5.0f);
        const float unpadded[4] = {1, 2, 3, 4};
        CHECK_STATUS(eval_detail::mlp_scale_slice(unpadded, 4, 4, out),
                     Status::kOk);
        CHECK(out == unpadded && out[0] == 1.0f);
        CHECK_STATUS(eval_detail::mlp_scale_slice(padded, 12, 5, out),
                     Status::kInvalidBuffer);
    }
    {
        // The wide upsample gain fallback: the last (C - m) entries of the
        // residual table prepended to the record's m entries.
        const float gain[2] = {7, 8};
        const float table[6] = {1, 2, 3, 4, 5, 6};
        float work[4];
        const float* out = nullptr;
        CHECK_STATUS(eval_detail::upsample_gain_row(gain, 2, table, 6, 4, work,
                                                    out), Status::kOk);
        CHECK(out == work);
        CHECK(work[0] == 5.0f && work[1] == 6.0f
              && work[2] == 7.0f && work[3] == 8.0f);
        CHECK_STATUS(eval_detail::upsample_gain_row(table, 6, table, 6, 4,
                                                    work, out), Status::kOk);
        CHECK(out == table);
    }

    // ---- DecInputUpsample: the decoder suite's K1 through eval_layer -------------------------------
    {
        static float scratch[1 << 20];
        static std::uint8_t w_codes[4] = {0x38, 0x40, 0x30, 0xB8};
        static float gain[2] = {0.5f, 2.0f};
        const Record records[2] = {
            {5, 0, "weight", 2, 2, 0, -2, w_codes, nullptr},
            {5, 0, "skip_weight", 0, 0, 2, -3, nullptr, gain},
        };
        Store store;
        store.records = records;
        store.record_count = 2;
        EvalWeights weights;
        bind_store(weights, store);
        EvalContext ctx;
        ctx.weights = &weights;
        ctx.scratch = scratch;
        ctx.scratch_capacity = sizeof(scratch) / sizeof(scratch[0]);
        CHECK_STATUS(eval_reset(ctx), Status::kOk);
        const float x_data[8] = {1, 2, 3, 4, 4, 0, -2, 1};
        const float skip_data[18] = {1, 2, 3, 4, 5, 6, 7, 8, 9,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0};
        FeatureBatchView x = make_view(const_cast<float*>(x_data), 2, 2, 2);
        FeatureBatchView skip = make_view(const_cast<float*>(skip_data), 2, 3, 3);
        LayerSpec spec = row(0, 5, 0, LayerKind::kDecInputUpsample, 2, 2, 3, 3,
                             2, 0);
        LayerCall call;
        call.x = &x;
        call.aux = &skip;
        LayerValue out;
        out.main.data = reinterpret_cast<const float*>(
            static_cast<std::uintptr_t>(0x5eed));
        CHECK_STATUS(eval_layer(&ctx, spec, call, out), Status::kOk);
        CHECK(!out.has_aside);
        CHECK(out.main.rows == 3 && out.main.cols == 3);
        const float want[18] = {10, 10, 3.5f, 11, 12, 5, 2.5f, 3, 10,
                                -3.5f, -3.5f, 1, -3.5f, -3.5f, 1,
                                3.5f, 3.5f, 1};
        for (int i = 0; i < 18; ++i) {
            CHECK(out.main.data[i] == want[i]);
        }
    }

    // ---- a projection binding with a permuted identity (mapping-swap control) ----------------------
    // VitFfnContract out[t][j] = q8(f16(sum_k x[t][k] * W[j][k] + g[j] *
    // skip[t][j])) over 32 channels; W permutes the inputs so a swapped
    // operand map changes the answer.
    {
        static float scratch[1 << 20];
        static std::uint8_t w_codes[32 * 32] = {};
        static float gain[32];
        // W[j][k] = 1 exactly when k == p(j), p(j) = (j * 5 + 1) & 31.
        for (int j = 0; j < 32; ++j) {
            for (int k = 0; k < 32; ++k) {
                w_codes[j * 32 + k] = k == ((j * 5 + 1) & 31) ? 0x38u : 0u;
            }
        }
        for (int j = 0; j < 32; ++j) {
            gain[j] = 0.5f;
        }
        const Record records[2] = {
            {7, 1, "weight", 32, 32, 0, -2, w_codes, nullptr},
            {7, 1, "skip_weight", 0, 0, 32, -3, nullptr, gain},
        };
        Store store;
        store.records = records;
        store.record_count = 2;
        EvalWeights weights;
        bind_store(weights, store);
        EvalContext ctx;
        ctx.weights = &weights;
        ctx.scratch = scratch;
        ctx.scratch_capacity = sizeof(scratch) / sizeof(scratch[0]);
        CHECK_STATUS(eval_reset(ctx), Status::kOk);
        static float x_data[32 * 2];
        static float skip_data[32 * 2];
        // Planar [32, 2, 1]: cell (channel c, token t) lives at c * 2 + t.
        for (int t = 0; t < 2; ++t) {
            for (int c = 0; c < 32; ++c) {
                x_data[c * 2 + t] = static_cast<float>(t * 32 + c) * 0.25f;
                skip_data[c * 2 + t] = static_cast<float>(c) * 0.5f;
            }
        }
        // The binding reads maps; lay the two tokens out as a 2x1 planar map.
        FeatureBatchView x = make_view(x_data, 32, 2, 1);
        FeatureBatchView skip = make_view(skip_data, 32, 2, 1);
        LayerSpec spec = row(0, 7, 1, LayerKind::kVitFfnContract, 32, 32, 2, 1,
                             32, 0);
        LayerCall call;
        call.x = &x;
        call.aux = &skip;
        LayerValue out;
        CHECK_STATUS(eval_layer(&ctx, spec, call, out), Status::kOk);
        // Hand derivation (exact arithmetic): token t, channel j reads
        // x[pi(j)] = (t * 32 + pi(j)) * 0.25 plus gain 0.5 on skip j * 0.5 ->
        // v = (t * 32 + pi(j)) * 0.25 + 0.25 * j. The shared, exhaustively
        // pinned e4m3/f16 rounding (native-pre-post) then selects the store
        // grid point — several fixture values fall between grid points
        // (8.25 -> 8, 9.75 -> 10, 11.25 -> 11), which also pins the store.
        for (int t = 0; t < 2; ++t) {
            for (int j = 0; j < 32; ++j) {
                const int pi = (j * 5 + 1) & 31;
                const float v = static_cast<float>(t * 32 + pi) * 0.25f
                    + 0.25f * static_cast<float>(j);
                CHECK(out.main.data[j * 2 + t] == e4m3(f16(v)));
            }
        }
    }

    // ---- record-consumption sets for the heavy kinds -----------------------------------------------
    // The arithmetic of these kinds is pinned by their family suites; here
    // the binding must consume exactly the pinned constructors' records.
    static float scratch[1 << 22];
    static Store big_store;
    big_store.serve_any = true;
    EvalWeights big_weights;
    bind_store(big_weights, big_store);
    EvalContext big_ctx;
    big_ctx.weights = &big_weights;
    big_ctx.scratch = scratch;
    big_ctx.scratch_capacity = sizeof(scratch) / sizeof(scratch[0]);

    static float image_storage[3 * 4 * 4];
    FeatureBatchView image = make_view(image_storage, 3, 4, 4);
    static float map_storage[2 * 32 * 4 * 4];
    FeatureBatchView map32 = make_view(map_storage, 32, 4, 4);

    {
        // A plain Swin body row consumes the nine body records.
        CHECK_STATUS(eval_reset(big_ctx), Status::kOk);
        big_store.requests = 0;
        LayerSpec spec = row(0, 3, 2, LayerKind::kSwin2H, 32, 32, 4, 4, 32, 1);
        LayerCall call;
        call.x = &map32;
        LayerValue out;
        CHECK_STATUS(eval_layer(&big_ctx, spec, call, out), Status::kOk);
        CHECK(saw_request(big_store, "mlp_expand", 128, 32, -2));
        CHECK(saw_request(big_store, "mlp_contract", 32, 128, -2));
        CHECK(saw_request(big_store, "qkv", 96, 32, -2));
        CHECK(saw_request(big_store, "attn_out_proj", 32, 32, -2));
        CHECK(saw_request(big_store, "residual_scale", 0, 0, -3));
        CHECK(saw_request(big_store, "attn_residual_scale", 0, 0, -3));
        CHECK(saw_request(big_store, "attn_pos_bias", 0, 0, -3));
        CHECK(saw_request(big_store, "scalars_b", 0, 0, -4));
    }
    {
        // A ds row adds "resample" [2C, C] and produces the pair.
        CHECK_STATUS(eval_reset(big_ctx), Status::kOk);
        big_store.requests = 0;
        LayerSpec spec = row(0, 3, 2, LayerKind::kSwin2H, 32, 64, 2, 2, 32, 1);
        spec.variant = LayerVariant::kDs;
        LayerCall call;
        call.x = &map32;
        LayerValue out;
        CHECK_STATUS(eval_layer(&big_ctx, spec, call, out), Status::kOk);
        CHECK(out.has_aside);
        CHECK(saw_request(big_store, "resample", 64, 32, -2));
    }
    {
        // A wide upsample row adds "resample" [C, 2C] and "upsample_gain";
        // its input map is twice the body width.
        CHECK_STATUS(eval_reset(big_ctx), Status::kOk);
        big_store.requests = 0;
        LayerSpec spec = row(0, 3, 2, LayerKind::kSwin2H, 64, 32, 4, 4, 32, 1);
        spec.variant = LayerVariant::kUpsample;
        static float map64_storage[2 * 64 * 2 * 2];
        FeatureBatchView map64 = make_view(map64_storage, 64, 2, 2);
        static float skip_storage[2 * 32 * 4 * 4];
        FeatureBatchView skip = make_view(skip_storage, 32, 4, 4);
        LayerCall call;
        call.x = &map64;
        call.aux = &skip;
        LayerValue out;
        CHECK_STATUS(eval_layer(&big_ctx, spec, call, out), Status::kOk);
        CHECK(saw_request(big_store, "resample", 32, 64, -2));
        CHECK(saw_request(big_store, "upsample_gain", 0, 0, -3));
    }
    {
        // The pre block: the lift record plus the body's records.
        CHECK_STATUS(eval_reset(big_ctx), Status::kOk);
        big_store.requests = 0;
        LayerSpec spec = row(0, 0, 0, LayerKind::kPreBlock, 3, 32, 2, 2, 32, 1);
        LayerCall call;
        call.image = &image;
        LayerValue out;
        CHECK_STATUS(eval_layer(&big_ctx, spec, call, out), Status::kOk);
        CHECK(out.has_aside);
        CHECK(saw_request(big_store, "input_lift", 0, 0, -3));
        CHECK(saw_request(big_store, "mlp_expand", 128, 32, -2));
    }
    {
        // The pre block's non-null history (the disclosed gap): the pinned's
        // single-frame contract copies the CURRENT colour into the second
        // colour's slots (7..9) — a history equal to the frame must be
        // indistinguishable from no history at all, and a distinct history
        // must change the output (the lift's slots 7..9 feed the body). The
        // body weights must be non-degenerate for the signal to propagate,
        // so this case runs its own pattern-filled store (e4m3 code 20 = 0.046875 — code 56 saturates the
        // body's activations to the e4m3 ceiling and hides the difference).
        static Store pattern_store;
        pattern_store.serve_any = true;
        for (std::size_t i = 0; i < sizeof(pattern_store.zero_codes); ++i) {
            pattern_store.zero_codes[i] = 20;   // e4m3 0.046875: no saturation
        }
        for (std::size_t i = 0;
             i < sizeof(pattern_store.zero_values)
                 / sizeof(pattern_store.zero_values[0]); ++i) {
            pattern_store.zero_values[i] = 1.0f;
        }
        static EvalContext pattern_ctx;
        static EvalWeights pattern_weights;
        bind_store(pattern_weights, pattern_store);
        pattern_ctx.weights = &pattern_weights;
        static float pattern_scratch[1 << 22];
        pattern_ctx.scratch = pattern_scratch;
        pattern_ctx.scratch_capacity =
            sizeof(pattern_scratch) / sizeof(pattern_scratch[0]);
        CHECK_STATUS(eval_reset(pattern_ctx), Status::kOk);
        LayerSpec spec = row(0, 0, 0, LayerKind::kPreBlock, 3, 32, 2, 2, 32, 1);
        LayerCall call;
        call.image = &image;
        LayerValue out;
        CHECK_STATUS(eval_layer(&pattern_ctx, spec, call, out), Status::kOk);
        static float no_history[32 * 2 * 2];
        std::memcpy(no_history, out.main.data, sizeof(no_history));
        static float same_rgb[48];
        static float distinct_rgb[48];
        // The eval's interleaved frame mirrors call.image (3 x 4 x 4).
        for (int c = 0; c < 3; ++c) {
            for (int p = 0; p < 16; ++p) {
                same_rgb[p * 3 + c] = image.data[c * 16 + p];
                distinct_rgb[p * 3 + c] = image.data[c * 16 + p] + 0.25f;
            }
        }
        static FrameView same{same_rgb, 4, 4, 12};
        pattern_ctx.history = &same;
        CHECK_STATUS(eval_reset(pattern_ctx), Status::kOk);
        CHECK_STATUS(eval_layer(&pattern_ctx, spec, call, out), Status::kOk);
        CHECK(std::memcmp(no_history, out.main.data, sizeof(no_history)) == 0);
        static FrameView distinct{distinct_rgb, 4, 4, 12};
        pattern_ctx.history = &distinct;
        CHECK_STATUS(eval_reset(pattern_ctx), Status::kOk);
        CHECK_STATUS(eval_layer(&pattern_ctx, spec, call, out), Status::kOk);
        std::printf("probe nh %g %g | wh %g %g\n", no_history[0], no_history[1], out.main.data[0], out.main.data[1]);
        CHECK(std::memcmp(no_history, out.main.data, sizeof(no_history)) != 0);
        // Determinism with the history bound.
        static float with_history[32 * 2 * 2];
        std::memcpy(with_history, out.main.data, sizeof(with_history));
        CHECK_STATUS(eval_reset(pattern_ctx), Status::kOk);
        CHECK_STATUS(eval_layer(&pattern_ctx, spec, call, out), Status::kOk);
        CHECK(std::memcmp(with_history, out.main.data,
                          sizeof(with_history)) == 0);
        pattern_ctx.history = nullptr;
    }
    {
        // SplitFfwd: the decoded feed-forward matrices from the boundary.
        CHECK_STATUS(eval_reset(big_ctx), Status::kOk);
        big_store.requests = 0;
        LayerSpec spec = row(0, 23, 0, LayerKind::kSplitFfwd, 512, 512, 1, 1,
                             512, 0);
        static float map512[2 * 512];
        FeatureBatchView map = make_view(map512, 512, 1, 1);
        LayerCall call;
        call.x = &map;
        LayerValue out;
        CHECK_STATUS(eval_layer(&big_ctx, spec, call, out), Status::kOk);
        CHECK(out.has_aside);
        CHECK(saw_request(big_store, "ffwd", 512, 512, 0));
        CHECK(saw_request(big_store, "ffwd", 2048, 64, 1));
        CHECK(saw_request(big_store, "ffwd", 512, 256, 2));
    }
    {
        // SplitQKVAttn: the packed qkv, the unpacked bias and the head gains.
        CHECK_STATUS(eval_reset(big_ctx), Status::kOk);
        big_store.requests = 0;
        LayerSpec spec = row(0, 23, 2, LayerKind::kSplitQKVAttn, 512, 512, 1, 1,
                             512, 0);
        static float map512[2 * 512];
        FeatureBatchView map = make_view(map512, 512, 1, 1);
        LayerCall call;
        call.x = &map;
        LayerValue out;
        CHECK_STATUS(eval_layer(&big_ctx, spec, call, out), Status::kOk);
        CHECK(saw_request(big_store, "qkv", 1536, 512, -2));
        CHECK(saw_request(big_store, "attn_pos_bias", 0, 0, -3));
        CHECK(saw_request(big_store, "tail", 0, 0, -4));
    }
    {
        // The vit QKV/attention pair: q/k/v matrices, the header temperature.
        CHECK_STATUS(eval_reset(big_ctx), Status::kOk);
        big_store.requests = 0;
        LayerSpec spec = row(0, 31, 2, LayerKind::kVitQKV, 32, 96, 1, 1, 32, 1);
        static float map32b[2 * 32];
        FeatureBatchView map = make_view(map32b, 32, 1, 1);
        LayerCall call;
        call.x = &map;
        LayerValue out;
        CHECK_STATUS(eval_layer(&big_ctx, spec, call, out), Status::kOk);
        CHECK(saw_request(big_store, "q", 32, 32, -2));
        CHECK(saw_request(big_store, "k", 32, 32, -2));
        CHECK(saw_request(big_store, "v", 32, 32, -2));
        CHECK(saw_request(big_store, "header", 0, 0, -4));
    }
    {
        // The post block: gains, the output projection and the body's.
        CHECK_STATUS(eval_reset(big_ctx), Status::kOk);
        big_store.requests = 0;
        LayerSpec spec = row(0, 70, 0, LayerKind::kPostBlock, 32, 3, 4, 4, 32, 1);
        static float skip_storage[2 * 32 * 4 * 4];
        FeatureBatchView skip = make_view(skip_storage, 32, 4, 4);
        LayerCall call;
        call.x = &map32;
        call.aux = &skip;
        call.image = &image;
        LayerValue out;
        CHECK_STATUS(eval_layer(&big_ctx, spec, call, out), Status::kOk);
        CHECK(!out.has_aside);
        CHECK(out.main.channels == 3);
        CHECK(saw_request(big_store, "main_gain", 0, 0, -3));
        CHECK(saw_request(big_store, "skip_gain", 0, 0, -3));
        CHECK(saw_request(big_store, "out_project", 0, 0, -3));
    }

    // ---- the pre-block lift decode (non-symmetric record) ------------------------------------------
    // A non-symmetric stored record plus a body whose attention carries only
    // its value rows (uniform bias softmax) makes every output depend on the
    // decoded lift; the oracle reproduces the pinned decomposition with
    // direct kernel calls (decode_lift, input_lift, swin_core), so a missing
    // or wrong decode in the binding changes the answer.
    {
        static float scratch[1 << 22];
        static std::uint8_t qkv_codes[96 * 32];
        static std::uint8_t out_proj_codes[32 * 32];
        static float stored[512];
        static float residual_scale[40];   // zero front 8 (padded form) + ones
        static float attn_scale[32];
        static float scalars_b[4];
        static float bias_table[4096];
        static float lift_expected[512];
        static float lifted_expected[32 * 4 * 4];
        static float frame_data[3 * 4 * 4];
        for (int i = 0; i < 512; ++i) {
            stored[i] = static_cast<float>((i * 7) % 23 - 11) * 0.25f;
        }
        for (int i = 0; i < 40; ++i) {
            residual_scale[i] = i < 8 ? 0.0f : 1.0f;
        }
        for (int i = 0; i < 32; ++i) {
            attn_scale[i] = 1.0f;
        }
        for (int i = 0; i < 4; ++i) {
            scalars_b[i] = 1.0f;
        }
        for (int i = 0; i < 4096; ++i) {
            bias_table[i] = 0.0f;
        }
        // q and k rows stay zero; the v rows form a non-symmetric pattern.
        for (int i = 0; i < 96 * 32; ++i) {
            qkv_codes[i] = 0u;
        }
        for (int r = 64; r < 96; ++r) {
            for (int c = 0; c < 32; ++c) {
                qkv_codes[r * 32 + c] =
                    static_cast<std::uint8_t>((r * 5 + c * 3) % 32 + 1);
            }
        }
        for (int r = 0; r < 32; ++r) {
            for (int c = 0; c < 32; ++c) {
                out_proj_codes[r * 32 + c] = r == c ? 0x38u : 0u;
            }
        }
        for (int i = 0; i < 3 * 16; ++i) {
            frame_data[i] = static_cast<float>((i * 5) % 17) * 0.0625f;
        }
        const Record records[7] = {
            {0, 0, "input_lift", 0, 0, 512, -3, nullptr, stored},
            {0, 0, "residual_scale", 0, 0, 40, -3, nullptr, residual_scale},
            {0, 0, "attn_residual_scale", 0, 0, 32, -3, nullptr, attn_scale},
            {0, 0, "scalars_b", 0, 0, 4, -4, nullptr, scalars_b},
            {0, 0, "attn_pos_bias", 0, 0, 4096, -3, nullptr, bias_table},
            {0, 0, "qkv", 96, 32, 0, -2, qkv_codes, nullptr},
            {0, 0, "attn_out_proj", 32, 32, 0, -2, out_proj_codes, nullptr},
        };
        Store store;
        store.records = records;
        store.record_count = 7;
        store.serve_any = true;  // zero matrices wherever else they exist
        EvalWeights weights;
        bind_store(weights, store);
        EvalContext ctx;
        ctx.weights = &weights;
        ctx.scratch = scratch;
        ctx.scratch_capacity = sizeof(scratch) / sizeof(scratch[0]);
        CHECK_STATUS(eval_reset(ctx), Status::kOk);
        FeatureBatchView image = make_view(frame_data, 3, 4, 4);
        LayerSpec spec = row(0, 0, 0, LayerKind::kPreBlock, 3, 32, 2, 2, 32, 1);
        LayerCall call;
        call.image = &image;
        LayerValue out;
        CHECK_STATUS(eval_layer(&ctx, spec, call, out), Status::kOk);
        // The oracle: the pinned decomposition with direct kernel calls
        // (including the planar -> interleaved frame copy the lift needs).
        static float frame_interleaved[48];
        CHECK_STATUS(eval_detail::frame_from_planar(image, frame_interleaved, 48),
                     Status::kOk);
        CHECK_STATUS(decode_lift(stored, 512, lift_expected), Status::kOk);
        CHECK_STATUS(input_lift(FrameView{frame_interleaved, 4, 4, 12},
                                ctx.controls, lift_expected, 512,
                                lifted_expected, 32 * 16), Status::kOk);
        eval_detail::SwinFetch fetch;
        CHECK_STATUS(eval_detail::swin_fetch(&weights, 0, 0, 32, 1, fetch),
                     Status::kOk);
        static float work[1 << 20];
        MatrixView mid_storage;
        SwinBlockParams params;
        CHECK_STATUS(eval_detail::swin_decode(fetch, 32, 1, work,
                                              (1 << 20), mid_storage, params),
                     Status::kOk);
        FeatureBatchView lifted_map = make_view(lifted_expected, 32, 4, 4);
        static float wide_direct[32 * 4 * 4];
        static float acc_direct[32 * 4 * 4];
        static float block_scratch[1 << 20];
        CHECK_STATUS(swin_core(lifted_map, 0, 0, params, wide_direct,
                               32 * 16, acc_direct, 32 * 16, block_scratch,
                               1 << 20), Status::kOk);
        CHECK(out.has_aside);
        CHECK(out.aside.rows == 4 && out.aside.cols == 4);
        for (int i = 0; i < 32 * 16; ++i) {
            CHECK(out.aside.data[i] == e4m3(wide_direct[i]));
        }
    }

    // ---- the vit QKV operand routing (distinct q/k/v records) --------------------------------------
    {
        static float scratch[1 << 22];
        static std::uint8_t q_codes[32 * 32];
        static std::uint8_t k_codes[32 * 32];
        static std::uint8_t v_codes[32 * 32];
        static float temperature[1];
        static float x_data[32 * 2];
        static float want_tokens[3 * 32 * 2];
        static float want_planar[3 * 32 * 2];
        // Distinct non-symmetric patterns over safe e4m3 codes (0x01..0x3e;
        // the NaN codes 0x7f/0xff would poison the k normalisation).
        for (int i = 0; i < 32 * 32; ++i) {
            q_codes[i] = static_cast<std::uint8_t>((i * 3) % 62 + 1);
            k_codes[i] = static_cast<std::uint8_t>((i * 5) % 62 + 1);
            v_codes[i] = static_cast<std::uint8_t>((i * 7) % 62 + 1);
        }
        temperature[0] = 2.0f;
        for (int i = 0; i < 32 * 2; ++i) {
            x_data[i] = static_cast<float>((i * 3) % 21 - 10) * 0.25f;
        }
        const Record records[4] = {
            {31, 2, "q", 32, 32, 0, -2, q_codes, nullptr},
            {31, 2, "k", 32, 32, 0, -2, k_codes, nullptr},
            {31, 2, "v", 32, 32, 0, -2, v_codes, nullptr},
            {31, 2, "header", 0, 0, 1, -4, nullptr, temperature},
        };
        Store store;
        store.records = records;
        store.record_count = 4;
        EvalWeights weights;
        bind_store(weights, store);
        EvalContext ctx;
        ctx.weights = &weights;
        ctx.scratch = scratch;
        ctx.scratch_capacity = sizeof(scratch) / sizeof(scratch[0]);
        CHECK_STATUS(eval_reset(ctx), Status::kOk);
        FeatureBatchView x = make_view(x_data, 32, 2, 1);
        LayerSpec spec = row(0, 31, 2, LayerKind::kVitQKV, 32, 96, 2, 1, 32, 1);
        LayerCall call;
        call.x = &x;
        LayerValue out;
        CHECK_STATUS(eval_layer(&ctx, spec, call, out), Status::kOk);
        // The oracle: the kernel called directly with the named operands in
        // pinned order (a q/k/v routing swap changes the answer).
        float in_tokens[32 * 2];
        CHECK_STATUS(eval_planar_to_tokens(x, in_tokens, 64), Status::kOk);
        CHECK_STATUS(vit_qkv(in_tokens, 2, 32, E4m3Matrix{q_codes, 32, 32},
                             E4m3Matrix{k_codes, 32, 32},
                             E4m3Matrix{v_codes, 32, 32}, temperature,
                             want_tokens), Status::kOk);
        CHECK_STATUS(eval_tokens_to_planar(want_tokens, 192, 96, 2, 1,
                                           want_planar, 192), Status::kOk);
        for (int i = 0; i < 192; ++i) {
            CHECK(out.main.data[i] == want_planar[i]);
        }
    }

    // ---- the shift-order mapping (the pin's shift_x = columns) ------------------------------------
    {
        LayerSpec s = row(0, 3, 2, LayerKind::kSwin2H, 32, 32, 4, 4, 32, 1);
        s.shift_x = 3;
        s.shift_y = 5;
        int shift_rows = 0;
        int shift_cols = 0;
        eval_detail::swin_shifts(s, shift_rows, shift_cols);
        CHECK(shift_rows == 5);
        CHECK(shift_cols == 3);
    }

    // ---- the wildcard record bound and release idempotence ---------------------------------------
    {
        static float long_record[5000];
        const Record records[1] = {
            {9, 9, "long", 0, 0, 5000, -3, nullptr, long_record},
        };
        Store store;
        store.records = records;
        store.record_count = 1;
        EvalWeights weights;
        bind_store(weights, store);
        const float* out = nullptr;
        int count = 0;
        // Longer than the probe bound: the wildcard form answers but no exact
        // count can be found.
        CHECK_STATUS(eval_detail::fetch_f16_any(&weights, 9, 9, "long", out,
                                                count), Status::kInvalidBuffer);
        // A release is idempotent.
        static float scratch[1 << 14];  // above the state header (kEvalStateWords)
        EvalContext ctx;
        ctx.weights = &weights;
        ctx.scratch = scratch;
        ctx.scratch_capacity = sizeof(scratch) / sizeof(scratch[0]);
        CHECK_STATUS(eval_reset(ctx), Status::kOk);
        eval_release(&ctx, 3);
        eval_release(&ctx, 3);
        CHECK(eval_arena_used(ctx) > 0);
    }

    // ---- the end-to-end zero-signal invariant ------------------------------------------------------
    {
        static LayerSpec specs[kPlanMaxLayers];
        static LayerValue outputs[kPlanMaxLayers];
        static int skip_scratch[kPlanMaxLayers];
        static int last_scratch[kPlanMaxLayers];
        static float out_scratch[1 << 26];  // 256 MiB arena (BSS)
        int count = 0;
        CHECK_STATUS(build_plan(64, 72, NetworkKind::kSmall, specs,
                                kPlanMaxLayers, count), Status::kOk);
        CHECK(count == 76);
        // The run consumes the padded frame (the plan floor: 320x320 here),
        // not the raw 64x72 frame; the host pads before and crops after.
        PlanPadding padding;
        CHECK_STATUS(plan_padding(64, 72, NetworkKind::kSmall, padding),
                     Status::kOk);
        CHECK(padding.width == 320 && padding.height == 320);
        static float frame_storage[3 * 320 * 320];
        for (std::size_t i = 0; i < sizeof(frame_storage) / sizeof(float); ++i) {
            frame_storage[i] = 0.0f;
        }
        FeatureBatchView image64 = make_view(frame_storage, 3, padding.width,
                                             padding.height);
        Store store;
        store.serve_any = true;
        EvalWeights weights;
        bind_store(weights, store);
        EvalContext ctx;
        ctx.weights = &weights;
        ctx.scratch = out_scratch;
        ctx.scratch_capacity =
            sizeof(out_scratch) / sizeof(out_scratch[0]);
        CHECK_STATUS(eval_reset(ctx), Status::kOk);
        NetworkRun run;
        run.image = &image64;
        run.outputs = outputs;
        run.skip_sources = skip_scratch;
        run.last_uses = last_scratch;
        run.layer = eval_layer;
        run.release = eval_release;
        run.context = &ctx;
        const unsigned long long before = g_allocations.load();
        CHECK_STATUS(network_run(specs, count, run), Status::kOk);
        CHECK(g_allocations.load() == before);
#if defined(__OPTIMIZE__)
        static float first_frame[3 * 320 * 320];
        // The determinism rerun (bit-for-bit over a reset arena) is exercised
        // in the optimised build; the -O0 build runs the identical path once
        // (the level-independent arithmetic is pinned here and by the family
        // suites).
        for (std::size_t i = 0; i < 3u * 320u * 320u; ++i) {
            first_frame[i] = outputs[count - 1].main.data[i];
        }
        CHECK_STATUS(eval_reset(ctx), Status::kOk);
        CHECK_STATUS(network_run(specs, count, run), Status::kOk);
        CHECK(g_allocations.load() == before);
        for (std::size_t i = 0; i < 3u * 320u * 320u; ++i) {
            CHECK(outputs[count - 1].main.data[i] == first_frame[i]);
        }
#endif
        // The zero network is the identity on the frame: the black input
        // comes back bit-exact from compose with a zero residual.
        const LayerValue& result = outputs[count - 1];
        CHECK(result.main.channels == 3);
        CHECK(result.main.rows == padding.width
              && result.main.cols == padding.height);
        // The published view is the tight planar map, read through its own
        // strides (a swapped stride pair would leave the block).
        CHECK(result.main.plane_stride
              == static_cast<std::ptrdiff_t>(padding.width) * padding.height);
        CHECK(result.main.row_stride == static_cast<std::ptrdiff_t>(padding.height));
        for (int c = 0; c < 3; ++c) {
            const float* plane = result.main.data
                + static_cast<std::size_t>(c)
                    * static_cast<std::size_t>(result.main.plane_stride);
            for (int r = 0; r < padding.width; ++r) {
                const float* row_data = plane
                    + static_cast<std::size_t>(r)
                        * static_cast<std::size_t>(result.main.row_stride);
                for (int col = 0; col < padding.height; ++col) {
                    CHECK(row_data[col] == 0.0f);
                }
            }
        }
    }

    // ---- rejection classes -------------------------------------------------------------------------
    {
        static float scratch_small[4];
        Store store;
        store.records = nullptr;
        store.record_count = 0;
        EvalWeights weights;
        bind_store(weights, store);
        EvalContext ctx;
        ctx.weights = &weights;
        ctx.scratch = scratch_small;
        ctx.scratch_capacity = 4;
        CHECK_STATUS(eval_reset(ctx), Status::kInvalidBuffer);
    }
    {
        static float scratch[1 << 16];
        Store store;
        EvalWeights weights;
        bind_store(weights, store);
        EvalContext ctx;
        ctx.weights = &weights;
        ctx.scratch = scratch;
        ctx.scratch_capacity = sizeof(scratch) / sizeof(scratch[0]);
        CHECK_STATUS(eval_reset(ctx), Status::kOk);
        const float x_data[4] = {1, 2, 3, 4};
        const float skip_data[4] = {1, 2, 3, 4};
        FeatureBatchView x = make_view(const_cast<float*>(x_data), 1, 2, 2);
        FeatureBatchView skip = make_view(const_cast<float*>(skip_data), 1, 2, 2);
        LayerSpec spec = row(0, 5, 0, LayerKind::kDecInputUpsample, 1, 1, 2, 2,
                             1, 0);
        LayerCall call;
        call.x = &x;
        call.aux = &skip;
        LayerValue out;
        out.main.data = reinterpret_cast<const float*>(
            static_cast<std::uintptr_t>(0x5eed));
        out.main.channels = -9;
        // The weight record is missing: rejected, `out` untouched.
        CHECK_STATUS(eval_layer(&ctx, spec, call, out), Status::kNullPointer);
        CHECK(out.main.data == reinterpret_cast<const float*>(
                   static_cast<std::uintptr_t>(0x5eed)));
        CHECK(out.main.channels == -9);
    }

    std::printf("native-eval: %lld checks passed\n", g_checks);
    return 0;
}
