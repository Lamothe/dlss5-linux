// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Michael Lamothe
//
// Known-answer tests for the int4 network miring (native-eval.hpp's mixed
// ViT feed-forward path over native-int4.hpp's Int4Site record). Hand-derived
// answers pin the mixed contraction: the quantiser codes (including a
// ties-to-even at 4.5), the int4 dot with its float64 affine, the e4m3 tail
// product over the permuted channels, the float32 sum and the residual store
// — plus the record-presence gating (null site = the e4m3 path), the site
// validation rejections and zero-allocation accounting.
//
// Built by vr/test-native-eval-int4.sh at -O2 and -O0 with -ffp-contract=off.
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

static const Int4Site* g_int4_site = nullptr;
static const char* g_int4_name = nullptr;
static int g_int4_block = -1;
static int g_int4_layer = -1;

static const Int4Site* bind_int4(void* context, int block, int layer,
                                 const char* name) {
    (void)context;
    if (g_int4_site == nullptr || name == nullptr || g_int4_name == nullptr) {
        return nullptr;
    }
    if (block != g_int4_block || layer != g_int4_layer
        || std::strcmp(name, g_int4_name) != 0) {
        return nullptr;
    }
    return g_int4_site;
}

static void bind_store(EvalWeights& w, Store& s) {
    w.context = &s;
    w.e4m3 = store_e4m3;
    w.f16 = store_f16;
    w.f32 = store_f32;
    w.ffwd = store_ffwd;
}

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
    // ---- the mixed ViT feed-forward: hand-derived two-token case -----------
    // Site: q [[2,-1],[1,3]], scale [0.5, 2], constant [1, 0], aq (1, 4) per
    // column (gain 1, magic zero 4), block 1; tail [[1, 2], [0.5, -1]] as
    // e4m3 codes; perm [1, 0, 3, 2] (int4 channels hidden 1, 0; tail hidden
    // 3, 2); skip_weight [1, 0.5]; skip zero.
    //
    // token 0 hidden [1.5, -2.25, 0.5, 4]: gathered [-2.25, 1.5] -> codes
    //   clamp(rint(v + 4)) - 8 = [2-8, 6-8] = [-6, -2]; dot [-10, -12];
    //   affine [-4, -24]; tail over [4, 0.5] -> [5, 1.5]; product [1, -22.5];
    //   residual (skip 0) -> [e4m3(1), e4m3(-22.5)] = [1, -22].
    // token 1 hidden [0.25, 0.5, 1, 2]: gathered [0.5, 0.25] -> codes
    //   [rint(4.5)=4 (ties even)-8, rint(4.25)=4-8] = [-4, -4]; dot [-4, -16];
    //   affine [-1, -32]; tail over [2, 1] -> [4, 0]; product [3, -32];
    //   residual -> [3, -32].
    {
        static float scratch[1 << 20];
        static std::int8_t q[4] = {2, -1, 1, 3};
        static float scale[2] = {0.5f, 2.0f};
        static float constant[2] = {1.0f, 0.0f};
        static float aq[4] = {1.0f, 4.0f, 1.0f, 4.0f};
        static std::uint8_t tail_record[4];
        tail_record[0] = e4m3_bits(1.0f);
        tail_record[1] = e4m3_bits(2.0f);
        tail_record[2] = e4m3_bits(0.5f);
        tail_record[3] = e4m3_bits(-1.0f);
        static float gain[2] = {1.0f, 0.5f};
        static int perm[4] = {1, 0, 3, 2};
        static Int4Site site;
        site.matrix = Int4MatrixView{q, scale, constant, aq, 2, 2, 1, false};
        site.perm = perm;
        site.perm_count = 4;

        const Record records[2] = {
            {7, 1, "tail", 2, 2, 0, -2, tail_record, nullptr},
            {7, 1, "skip_weight", 0, 0, 2, -3, nullptr, gain},
        };
        static Store store;
        store.records = records;
        store.record_count = 2;
        EvalWeights weights;
        bind_store(weights, store);
        weights.int4 = bind_int4;
        g_int4_site = &site;
        g_int4_name = "vit_ffn";
        g_int4_block = 7;
        g_int4_layer = 1;

        EvalContext ctx;
        ctx.weights = &weights;
        ctx.scratch = scratch;
        ctx.scratch_capacity = sizeof(scratch) / sizeof(scratch[0]);
        CHECK_STATUS(eval_reset(ctx), Status::kOk);
        static float x_data[4 * 2];
        static float skip_data[2 * 2] = {};
        x_data[0 * 2 + 0] = 1.5f;
        x_data[1 * 2 + 0] = -2.25f;
        x_data[2 * 2 + 0] = 0.5f;
        x_data[3 * 2 + 0] = 4.0f;
        x_data[0 * 2 + 1] = 0.25f;
        x_data[1 * 2 + 1] = 0.5f;
        x_data[2 * 2 + 1] = 1.0f;
        x_data[3 * 2 + 1] = 2.0f;
        FeatureBatchView x = make_view(x_data, 4, 2, 1);
        FeatureBatchView skip = make_view(skip_data, 2, 2, 1);
        LayerSpec spec = row(0, 7, 1, LayerKind::kVitFfnContract, 4, 2, 2, 1,
                             4, 0);
        LayerCall call;
        call.x = &x;
        call.aux = &skip;
        LayerValue out;
        const unsigned long long alloc_before = g_allocations.load();
        CHECK_STATUS(eval_layer(&ctx, spec, call, out), Status::kOk);
        CHECK(g_allocations.load() == alloc_before);
        CHECK(out.main.data[0 * 2 + 0] == 1.0f);
        CHECK(out.main.data[1 * 2 + 0] == -22.0f);
        CHECK(out.main.data[0 * 2 + 1] == 3.0f);
        CHECK(out.main.data[1 * 2 + 1] == -32.0f);

        // Determinism: the same run twice is bit-identical.
        LayerValue out2;
        CHECK_STATUS(eval_layer(&ctx, spec, call, out2), Status::kOk);
        CHECK(std::memcmp(out.main.data, out2.main.data,
                          4 * sizeof(float)) == 0);

        // The residual's gain hop with a non-zero skip.
        skip_data[0 * 2 + 0] = 2.0f;
        skip_data[1 * 2 + 0] = 4.0f;
        skip_data[0 * 2 + 1] = 2.0f;
        skip_data[1 * 2 + 1] = 4.0f;
        LayerValue skip_out;
        CHECK_STATUS(eval_layer(&ctx, spec, call, skip_out), Status::kOk);
        CHECK(skip_out.main.data[0 * 2 + 0] == 3.0f);    // e4m3(f16(1 + 2))
        CHECK(skip_out.main.data[1 * 2 + 0] == -20.0f);  // e4m3(f16(-22.5 + 2))
        CHECK(skip_out.main.data[0 * 2 + 1] == 5.0f);    // e4m3(f16(3 + 2))
        CHECK(skip_out.main.data[1 * 2 + 1] == -30.0f);  // e4m3(f16(-32 + 2))
        for (int i = 0; i < 4; ++i) {
            skip_data[i] = 0.0f;
        }

        // The tail's NaN-code decode (the solver table reads 0x7f as 0):
        // replacing tail[0][1] = 2 with 0x7f changes token 0's row-0 tail
        // product from 4*1 + 0.5*2 = 5 to 4*1 + 0.5*0 = 4 -> product 0.
        tail_record[1] = 0x7fu;
        LayerValue nan_out;
        CHECK_STATUS(eval_layer(&ctx, spec, call, nan_out), Status::kOk);
        CHECK(nan_out.main.data[0 * 2 + 0] == 0.0f);  // e4m3(f16(-4 + 4))
        tail_record[1] = e4m3_bits(2.0f);

        // The gating: a null site runs the e4m3 path over "weight" instead.
        static std::uint8_t w_codes[2 * 4] = {};
        w_codes[0 * 4 + 0] = e4m3_bits(1.0f);
        w_codes[1 * 4 + 3] = e4m3_bits(1.0f);
        const Record e4m3_records[3] = {
            {7, 1, "weight", 2, 4, 0, -2, w_codes, nullptr},
            {7, 1, "tail", 2, 2, 0, -2, tail_record, nullptr},
            {7, 1, "skip_weight", 0, 0, 2, -3, nullptr, gain},
        };
        static Store e4m3_store;
        e4m3_store.records = e4m3_records;
        e4m3_store.record_count = 3;
        EvalWeights e4m3_weights;
        bind_store(e4m3_weights, e4m3_store);
        e4m3_weights.int4 = bind_int4;
        g_int4_site = nullptr;
        EvalContext e4m3_ctx;
        e4m3_ctx.weights = &e4m3_weights;
        e4m3_ctx.scratch = scratch;
        e4m3_ctx.scratch_capacity = sizeof(scratch) / sizeof(scratch[0]);
        CHECK_STATUS(eval_reset(e4m3_ctx), Status::kOk);
        LayerValue e4m3_out;
        CHECK_STATUS(eval_layer(&e4m3_ctx, spec, call, e4m3_out), Status::kOk);
        // e4m3 path: acc = x[0] / x[3] rows; residual (skip 0) stores e4m3.
        CHECK(e4m3_out.main.data[0 * 2 + 0] == 1.5f);
        CHECK(e4m3_out.main.data[1 * 2 + 0] == 4.0f);
        CHECK(e4m3_out.main.data[0 * 2 + 1] == 0.25f);
        CHECK(e4m3_out.main.data[1 * 2 + 1] == 2.0f);
    }

    // ---- site validation rejections -----------------------------------------
    {
        static float scratch[1 << 20];
        static std::int8_t q[4] = {2, -1, 1, 3};
        static float scale[2] = {0.5f, 2.0f};
        static float constant[2] = {1.0f, 0.0f};
        static float aq[4] = {1.0f, 4.0f, 1.0f, 4.0f};
        static std::uint8_t tail_record[4] = {};
        static float gain[2] = {1.0f, 0.5f};
        static int good_perm[4] = {1, 0, 3, 2};
        static int bad_perm[4] = {1, 0, 3, 3};   // duplicate
        static int range_perm[4] = {1, 0, 3, 9}; // out of range
        static Int4Site site;
        site.matrix = Int4MatrixView{q, scale, constant, aq, 2, 2, 1, false};
        site.perm = bad_perm;
        site.perm_count = 4;
        const Record records[2] = {
            {7, 1, "skip_weight", 0, 0, 2, -3, nullptr, gain},
            {7, 1, "tail", 2, 2, 0, -2, tail_record, nullptr},
        };
        static Store store;
        store.records = records;
        store.record_count = 2;
        EvalWeights weights;
        bind_store(weights, store);
        weights.int4 = bind_int4;
        g_int4_site = &site;
        g_int4_name = "vit_ffn";
        g_int4_block = 7;
        g_int4_layer = 1;
        EvalContext ctx;
        ctx.weights = &weights;
        ctx.scratch = scratch;
        ctx.scratch_capacity = sizeof(scratch) / sizeof(scratch[0]);
        CHECK_STATUS(eval_reset(ctx), Status::kOk);
        static float x_data[4 * 2] = {};
        static float skip_data[2 * 2] = {};
        FeatureBatchView x = make_view(x_data, 4, 2, 1);
        FeatureBatchView skip = make_view(skip_data, 2, 2, 1);
        LayerSpec spec = row(0, 7, 1, LayerKind::kVitFfnContract, 4, 2, 2, 1,
                             4, 0);
        LayerCall call;
        call.x = &x;
        call.aux = &skip;
        LayerValue out;
        static float out_storage[2 * 2];
        out.main = make_view(out_storage, 2, 2, 1);
        for (int i = 0; i < 4; ++i) {
            out_storage[i] = -777.0f;
        }
        CHECK_STATUS(eval_layer(&ctx, spec, call, out),
                     Status::kInvalidValues);   // duplicate perm
        site.perm = range_perm;
        CHECK_STATUS(eval_layer(&ctx, spec, call, out),
                     Status::kInvalidValues);   // out of range
        site.perm_count = 3;
        CHECK_STATUS(eval_layer(&ctx, spec, call, out),
                     Status::kInvalidValues);   // perm length
        site.perm_count = 4;
        site.matrix.rows = 3;
        CHECK_STATUS(eval_layer(&ctx, spec, call, out),
                     Status::kInvalidValues);   // rows != c_out
        site.matrix.rows = 2;
        site.perm = good_perm;
        // A missing tail record rejects before any output write (the second
        // record is the tail; dropping it exercises the tail fetch).
        store.record_count = 1;
        CHECK_STATUS(eval_layer(&ctx, spec, call, out), Status::kNullPointer);
        for (int i = 0; i < 4; ++i) {
            CHECK(out_storage[i] == -777.0f);
        }
    }

    // ---- kVitProjection never takes the mixed path ---------------------------
    // The same site is offered to a projection row: it must run the e4m3
    // path over "weight" (hand-derived: acc = x[0] / x[3] rows).
    {
        static float scratch[1 << 20];
        static std::int8_t q[4] = {2, -1, 1, 3};
        static float scale[2] = {0.5f, 2.0f};
        static float constant[2] = {1.0f, 0.0f};
        static float aq[4] = {1.0f, 4.0f, 1.0f, 4.0f};
        static std::uint8_t tail_record[4] = {};
        static std::uint8_t w_codes[2 * 4] = {};
        w_codes[0 * 4 + 0] = e4m3_bits(1.0f);
        w_codes[1 * 4 + 3] = e4m3_bits(1.0f);
        static float gain[2] = {1.0f, 0.5f};
        static int perm[4] = {1, 0, 3, 2};
        static Int4Site site;
        site.matrix = Int4MatrixView{q, scale, constant, aq, 2, 2, 1, false};
        site.perm = perm;
        site.perm_count = 4;
        const Record records[3] = {
            {7, 1, "weight", 2, 4, 0, -2, w_codes, nullptr},
            {7, 1, "tail", 2, 2, 0, -2, tail_record, nullptr},
            {7, 1, "skip_weight", 0, 0, 2, -3, nullptr, gain},
        };
        static Store store;
        store.records = records;
        store.record_count = 3;
        EvalWeights weights;
        bind_store(weights, store);
        weights.int4 = bind_int4;
        g_int4_site = &site;
        g_int4_name = "vit_ffn";
        g_int4_block = 7;
        g_int4_layer = 1;
        EvalContext ctx;
        ctx.weights = &weights;
        ctx.scratch = scratch;
        ctx.scratch_capacity = sizeof(scratch) / sizeof(scratch[0]);
        CHECK_STATUS(eval_reset(ctx), Status::kOk);
        static float x_data[4 * 2];
        x_data[0 * 2 + 0] = 1.5f;
        x_data[3 * 2 + 0] = 4.0f;
        x_data[0 * 2 + 1] = 0.25f;
        x_data[3 * 2 + 1] = 2.0f;
        static float skip_data[2 * 2] = {};
        FeatureBatchView x = make_view(x_data, 4, 2, 1);
        FeatureBatchView skip = make_view(skip_data, 2, 2, 1);
        LayerSpec spec = row(0, 7, 1, LayerKind::kVitProjection, 4, 2, 2, 1,
                             4, 0);
        LayerCall call;
        call.x = &x;
        call.aux = &skip;
        LayerValue out;
        CHECK_STATUS(eval_layer(&ctx, spec, call, out), Status::kOk);
        // The e4m3 path ran (the mixed path would need no "weight" and would
        // give [1, -22, 3, -32] instead).
        CHECK(out.main.data[0 * 2 + 0] == 1.5f);
        CHECK(out.main.data[1 * 2 + 0] == 4.0f);
    }

    // ---- the mixed ViT QKV: hand-derived two-head case ----------------------
    // channels = 64 (two heads of 32), block 16, aq (gain 1 -> 1/4 after the
    // block factor, zero 8); q [192, 64] has q[0][0] = 1 and q[64][0] = 1 (the
    // k block's first row), everything else 0; scale 1, constant 0; header
    // (temperature) [1, 1]. Input [4, 0 * 63] per token.
    //   block 0 = [4, 0 * 15] -> fwht all 4 -> codes 1 (rint(4 * 0.25 + 8)
    //   = 9 - 8); the zero blocks give codes 0. Products: rows 0 and 64 =
    //   1, the rest 0. Post chain head 0: pq = [1, 0 * 31], pk = [1, 0 * 31],
    //   pv all 0; inv = 1, temp = 1, sqrt_head = f16(sqrt(32)) = 5.5 ->
    //   q[0] = e4m3(5.5) = 5.5, k[0] = e4m3(1) = 1, v 0. Head 1's products
    //   are all zero -> all outputs 0.
    {
        static float scratch[1 << 20];
        static std::int8_t q[192 * 64] = {};
        q[0 * 64 + 0] = 1;
        q[64 * 64 + 0] = 1;
        static float scale[192];
        static float constant[192] = {};
        static float aq[64 * 2];
        for (int k = 0; k < 64; ++k) {
            scale[k] = 1.0f;
            scale[191 - k] = 1.0f;
            aq[2 * k] = 1.0f;
            aq[2 * k + 1] = 8.0f;
        }
        for (int k = 0; k < 192; ++k) {
            scale[k] = 1.0f;
        }
        static int perm[4] = {1, 0, 3, 2};
        static Int4Site site;
        site.matrix = Int4MatrixView{q, scale, constant, aq, 192, 64, 16,
                                     false};
        site.perm = nullptr;
        site.perm_count = 0;
        static float header[2] = {1.0f, 1.0f};
        const Record records[1] = {
            {7, 1, "header", 0, 0, 2, -4, nullptr, header},
        };
        static Store store;
        store.records = records;
        store.record_count = 1;
        EvalWeights weights;
        bind_store(weights, store);
        weights.int4 = bind_int4;
        g_int4_site = &site;
        g_int4_name = "vit_qkv";
        g_int4_block = 7;
        g_int4_layer = 1;
        EvalContext ctx;
        ctx.weights = &weights;
        ctx.scratch = scratch;
        ctx.scratch_capacity = sizeof(scratch) / sizeof(scratch[0]);
        CHECK_STATUS(eval_reset(ctx), Status::kOk);
        static float x_data[64 * 2] = {};
        x_data[0] = 4.0f;
        x_data[1] = 4.0f;
        FeatureBatchView x = make_view(x_data, 64, 2, 1);
        LayerSpec spec = row(0, 7, 1, LayerKind::kVitQKV, 64, 192, 2, 1,
                             64, 1);
        LayerCall call;
        call.x = &x;
        LayerValue out;
        const unsigned long long alloc_before = g_allocations.load();
        CHECK_STATUS(eval_layer(&ctx, spec, call, out), Status::kOk);
        CHECK(g_allocations.load() == alloc_before);
        // Row 0 (q) = 5.5, row 64 (k) = 1, everything else 0, per token.
        for (int t = 0; t < 2; ++t) {
            for (int r = 0; r < 192; ++r) {
                const float want = r == 0 ? 5.5f : (r == 64 ? 1.0f : 0.0f);
                CHECK(out.main.data[r * 2 + t] == want);
            }
        }
        // The record-consumption pin: the run succeeded although the store
        // holds no "q"/"k"/"v" records at all — the mixed path never fetched
        // them (the e4m3 path would have rejected).
        LayerValue out2;
        CHECK_STATUS(eval_layer(&ctx, spec, call, out2), Status::kOk);
        CHECK(std::memcmp(out.main.data, out2.main.data,
                          2 * 192 * sizeof(float)) == 0);

        // Rejections: a wrong-shaped site and an FFN-style site (with a
        // permutation) offered to the QKV name.
        site.matrix.rows = 100;
        CHECK_STATUS(eval_layer(&ctx, spec, call, out2),
                     Status::kInvalidValues);
        site.matrix.rows = 192;
        site.perm = perm;
        site.perm_count = 4;
        CHECK_STATUS(eval_layer(&ctx, spec, call, out2),
                     Status::kInvalidValues);
        site.perm = nullptr;
        site.perm_count = 0;
        // A half-form record (the Swin sites' shape) rejects here too.
        site.matrix.half = true;
        CHECK_STATUS(eval_layer(&ctx, spec, call, out2),
                     Status::kInvalidValues);
        site.matrix.half = false;
        // A null site falls through to the e4m3 path (which rejects here for
        // the absent "q" record — the eval suite covers the populated path).
        g_int4_site = nullptr;
        CHECK_STATUS(eval_layer(&ctx, spec, call, out2), Status::kNullPointer);
    }

    std::printf("native-eval-int4: %lld checks passed\n", g_checks);
    return 0;
}
