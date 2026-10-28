// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Michael Lamothe
//
// Plan-level integration tests for the int4 mixed-network path: a full
// synthetic small network where the supported rows run the int4 sites
// ("vit_ffn" contraction + expansion, "vit_qkv", "attn_qkv", the Swin
// overlay) while every other row runs e4m3 — the pinned's table-presence
// gating end to end. Known-answer values are the row suites' domain
// (native-int4-test.cpp, native-eval-int4-test.cpp); this suite pins what
// only the whole plan can:
//   * the record-consumption policy over every row (a mixed row never
//     fetches its e4m3 twin record; a plain row always does) — the run's
//     routing proof;
//   * the binary16 side channel end to end: the ("stream", block+1) chain's
//     gate is the expansion's int4 path itself (it requires the producer's
//     stash), so a mixed expansion row's missing "weight" fetch is the
//     routing discriminator — and the negative control (has_ex off) must
//     fetch "weight" again;
//   * the mixed rows' shared records ("tail" / "header" / "skip_weight")
//     still arrive;
//   * determinism (bit-identical replays) and balanced allocation
//     accounting of the mixed runs;
//   * one network-level rejection (a wrong-shaped site fails at its row:
//     earlier rows keep outputs, later rows stay untouched);
//   * the temporal consumer over a mixed network (reset, counter, replay).
// The synthetic site matrices are arbitrary deterministic records: the
// suite claims routing and determinism, never synthetic network values.
//
// Built by vr/test-native-eval-int4-network.sh at -O2 and -O0 with
// -ffp-contract=off.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "native-demo-support.hpp"
#include "native-eval.hpp"
#include "native-history.hpp"
#include "native-io.hpp"
#include "native-network.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        std::fprintf(stderr, "FAILED (%d): %s\n", line, expr);
        ++g_failures;
    }
}
void check_status(dlss5_nr::Status got, dlss5_nr::Status want,
                  const char* expr, int line) {
    ++g_checks;
    if (got != want) {
        std::fprintf(stderr, "FAILED (%d): %s (got %d, want %d)\n", line,
                     expr, static_cast<int>(got), static_cast<int>(want));
        ++g_failures;
    }
}
#define CHECK(expr) check((expr), #expr, __LINE__)
#define CHECK_STATUS(got, want) \
    check_status((got), (want), #got, __LINE__)

// ---- allocation accounting (the operator-new block, family pattern) ----
std::atomic<unsigned long long> g_allocations{0};
std::atomic<long long> g_live{0};

// ---- the fetch log: every record request the run makes ----------------
constexpr int kLogCap = 1 << 20;
enum class FetchKind : int { kE4m3 = 0, kF16 = 1, kF32 = 2, kFfwd = 3 };
struct FetchEntry {
    FetchKind kind;
    int block;
    int layer;
    char name[24];
};
FetchEntry g_log[kLogCap];
int g_log_count = 0;
bool g_log_overflow = false;

void log_fetch(FetchKind kind, int block, int layer, const char* name) {
    if (g_log_count >= kLogCap) {
        g_log_overflow = true;
        return;
    }
    FetchEntry& e = g_log[g_log_count++];
    e.kind = kind;
    e.block = block;
    e.layer = layer;
    std::snprintf(e.name, sizeof(e.name), "%s", name != nullptr ? name : "");
}

bool log_has(int block, int layer, const char* name) {
    for (int i = 0; i < g_log_count; ++i) {
        const FetchEntry& e = g_log[i];
        if (e.block == block && e.layer == layer
            && std::strcmp(e.name, name) == 0) {
            return true;
        }
    }
    return false;
}

// ---- the synthetic record store ----------------------------------------
// The DemoStore serves every e4m3/f16/f32 record generically; the wrappers
// delegate to it and record the request. The int4 sites come from the
// factory below (the policy selects which rows are mixed).
dlss5_demo::DemoStore g_store;
dlss5_nr::EvalWeights g_base;

struct Policy {
    bool mixed_vit = false;
    bool mixed_split = false;
    bool mixed_swin = false;
    bool has_ex = false;
    bool alternate = false;      // sites on alternating vit blocks only
    int break_contract = -1;     // a vit block whose contract site is wrong
};
Policy g_policy;
const dlss5_nr::LayerSpec* g_specs = nullptr;
int g_spec_count = 0;
int g_first_vit_block = -1;

const dlss5_nr::LayerSpec* find_row(int block, int layer) {
    for (int i = 0; i < g_spec_count; ++i) {
        if (g_specs[i].block == block && g_specs[i].layer == layer) {
            return &g_specs[i];
        }
    }
    return nullptr;
}

int row_index(const dlss5_nr::LayerSpec* specs, int count, int block,
              int layer) {
    for (int i = 0; i < count; ++i) {
        if (specs[i].block == block && specs[i].layer == layer) {
            return i;
        }
    }
    return -1;
}

bool policy_allows(int block) {
    return !g_policy.alternate
        || (block - g_first_vit_block) % 2 == 0;
}

// ---- the synthetic int4 matrices ---------------------------------------
// A tiny bump registry keyed (rows, cols). q carries a deterministic
// non-trivial int8 pattern; the affines are the identity form (scale 1,
// constant 0, aq (1, 0)). Arbitrary records: no value claim.
struct MatEntry {
    int rows = 0;
    int cols = 0;
    std::int8_t* q = nullptr;
    float* scale = nullptr;
    float* constant = nullptr;
    float* aq = nullptr;
};
MatEntry g_mats[24];
int g_mat_count = 0;

const MatEntry& get_mat(int rows, int cols) {
    for (int i = 0; i < g_mat_count; ++i) {
        if (g_mats[i].rows == rows && g_mats[i].cols == cols) {
            return g_mats[i];
        }
    }
    if (rows < 1 || cols < 1 || g_mat_count >= 24) {
        std::fprintf(stderr, "matrix registry exhausted (%d x %d)\n", rows,
                     cols);
        std::abort();
    }
    MatEntry& m = g_mats[g_mat_count++];
    m.rows = rows;
    m.cols = cols;
    const std::size_t cells = static_cast<std::size_t>(rows)
        * static_cast<std::size_t>(cols);
    m.q = static_cast<std::int8_t*>(std::malloc(cells));
    m.scale = static_cast<float*>(
        std::malloc(static_cast<std::size_t>(rows) * sizeof(float)));
    m.constant = static_cast<float*>(
        std::malloc(static_cast<std::size_t>(rows) * sizeof(float)));
    m.aq = static_cast<float*>(
        std::malloc(static_cast<std::size_t>(cols) * 2 * sizeof(float)));
    if (m.q == nullptr || m.scale == nullptr || m.constant == nullptr
        || m.aq == nullptr) {
        std::fprintf(stderr, "matrix allocation failed\n");
        std::abort();
    }
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            m.q[static_cast<std::size_t>(r) * cols + c] =
                static_cast<std::int8_t>((r * 7 + c * 3 + 5) % 15 - 7);
        }
        m.scale[r] = 1.0f;
        m.constant[r] = 0.0f;
    }
    for (int c = 0; c < cols; ++c) {
        m.aq[2 * c] = 1.0f;
        m.aq[2 * c + 1] = 0.0f;
    }
    return m;
}

// The identity permutation (shared by the FFN sites; perm_count is the
// per-site column count). Bounded: sites beyond the table are refused and
// the row rejects visibly.
int g_perm[8192];

// ---- the site factory ---------------------------------------------------
struct SiteEntry {
    int block = -1;
    int layer = -1;
    const char* name = nullptr;
    dlss5_nr::Int4Site site;
};
SiteEntry g_sites[80];
int g_site_count = 0;

const dlss5_nr::Int4Site* serve_int4(void* /*context*/, int block, int layer,
                                     const char* name) {
    if (name == nullptr) {
        return nullptr;
    }
    const dlss5_nr::LayerSpec* row = find_row(block, layer);
    if (row == nullptr) {
        return nullptr;
    }
    const bool exp = row->kind == dlss5_nr::LayerKind::kVitFfnExpand
        && std::strcmp(name, "vit_ffn") == 0;
    const bool con = row->kind == dlss5_nr::LayerKind::kVitFfnContract
        && std::strcmp(name, "vit_ffn") == 0;
    const bool qkv = row->kind == dlss5_nr::LayerKind::kVitQKV
        && std::strcmp(name, "vit_qkv") == 0;
    const bool split = row->kind == dlss5_nr::LayerKind::kSplitQKVAttn
        && std::strcmp(name, "attn_qkv") == 0;
    if (!exp && !con && !qkv && !split) {
        return nullptr;
    }
    if (exp || con || qkv) {
        if (!g_policy.mixed_vit || !policy_allows(block)) {
            return nullptr;
        }
    } else if (!g_policy.mixed_split) {
        return nullptr;
    }
    for (int i = 0; i < g_site_count; ++i) {
        if (g_sites[i].block == block && g_sites[i].layer == layer
            && g_sites[i].name == name) {
            return &g_sites[i].site;
        }
    }
    if (g_site_count >= 80) {
        std::fprintf(stderr, "site registry exhausted\n");
        std::abort();
    }
    SiteEntry& e = g_sites[g_site_count++];
    e.block = block;
    e.layer = layer;
    e.name = name;
    e.site = dlss5_nr::Int4Site{};
    if (exp) {
        // The expansion site: ex [c_out, c_in] block 16 non-half, perm over
        // c_out; has_ex drives the ("stream", block) side channel.
        const int rows = row->c_out;
        const int cols = row->c_in;
        if (rows > 8192) {
            return nullptr;
        }
        const MatEntry& m = get_mat(rows, cols);
        e.site.ex = dlss5_nr::Int4MatrixView{m.q, m.scale, m.constant, m.aq,
                                             rows, cols, 16, false};
        for (int i = 0; i < rows; ++i) {
            g_perm[i] = i;
        }
        e.site.perm = g_perm;
        e.site.perm_count = rows;
        e.site.has_ex = g_policy.has_ex;
    } else if (con) {
        // The contraction site: [c_out, k4] with tail_cols = c_in - k4 >= 1,
        // perm over c_in; one broken site per the rejection config.
        const int rows = g_policy.break_contract == block ? row->c_out + 1
                                                          : row->c_out;
        const int k4 = row->c_in / 2;
        const int block_size = k4 % 16 == 0 ? 16 : 1;
        if (row->c_in > 8192) {
            return nullptr;
        }
        const MatEntry& m = get_mat(rows, k4);
        e.site.matrix = dlss5_nr::Int4MatrixView{m.q, m.scale, m.constant,
                                                 m.aq, rows, k4, block_size,
                                                 false};
        for (int i = 0; i < row->c_in; ++i) {
            g_perm[i] = i;
        }
        e.site.perm = g_perm;
        e.site.perm_count = row->c_in;
    } else if (qkv) {
        // The pinned vit_matrix: [3C, C] block 16 non-half, no permutation.
        const int rows = 3 * row->channels;
        const int cols = row->channels;
        const MatEntry& m = get_mat(rows, cols);
        e.site.matrix = dlss5_nr::Int4MatrixView{m.q, m.scale, m.constant,
                                                 m.aq, rows, cols, 16, false};
    } else {
        // The pinned attention_matrix: [1536, 512] block 16 non-half, no
        // permutation.
        const MatEntry& m = get_mat(dlss5_nr::kSplitQkvOut,
                                    dlss5_nr::kSplitChannels);
        e.site.matrix = dlss5_nr::Int4MatrixView{
            m.q, m.scale, m.constant, m.aq, dlss5_nr::kSplitQkvOut,
            dlss5_nr::kSplitChannels, 16, false};
    }
    return &e.site;
}

// ---- the Swin overlay factory ------------------------------------------
struct SwinEntry {
    int block = -1;
    int layer = -1;
    dlss5_nr::SwinSiteRecord record;
};
SwinEntry g_swin_sites[32];
int g_swin_count = 0;

const dlss5_nr::SwinSiteRecord* serve_swin(void* /*context*/, int block,
                                           int layer) {
    // The DemoStore binds no site records (null = the e4m3 dataflow).
    const dlss5_nr::SwinSiteRecord* base = g_base.swin != nullptr
        ? g_base.swin(g_base.context, block, layer)
        : nullptr;
    if (!g_policy.mixed_swin) {
        return base;
    }
    const dlss5_nr::LayerSpec* row = find_row(block, layer);
    if (row == nullptr) {
        return base;
    }
    for (int i = 0; i < g_swin_count; ++i) {
        if (g_swin_sites[i].block == block && g_swin_sites[i].layer == layer) {
            return &g_swin_sites[i].record;
        }
    }
    if (g_swin_count >= 32) {
        std::fprintf(stderr, "swin site registry exhausted\n");
        std::abort();
    }
    // The synthetic record: fold non-zero (the hard-swish swap; the pinned's
    // fold chain is the host's boundary) and both matrices for every row —
    // build_swin_overlay gates them to heads > 1 exactly as the pinned.
    const int c = row->channels;
    SwinEntry& e = g_swin_sites[g_swin_count++];
    e.block = block;
    e.layer = layer;
    e.record.fold = 1.0f;
    const MatEntry& ex = get_mat(4 * c, c);
    const MatEntry& qkv = get_mat(3 * c, c);
    e.record.ex = dlss5_nr::Int4MatrixView{ex.q, ex.scale, ex.constant, ex.aq,
                                           4 * c, c, 1, true};
    e.record.qkv = dlss5_nr::Int4MatrixView{qkv.q, qkv.scale, qkv.constant,
                                            qkv.aq, 3 * c, c, 1, true};
    e.record.has_ex = true;
    e.record.has_qkv = true;
    return &e.record;
}

dlss5_nr::StashBuffers g_stash;

// ---- the logging wrappers ----------------------------------------------
const std::uint8_t* wrap_e4m3(void* /*context*/, int block, int layer,
                              const char* name, int rows, int cols) {
    log_fetch(FetchKind::kE4m3, block, layer, name);
    return g_base.e4m3(g_base.context, block, layer, name, rows, cols);
}
const float* wrap_f16(void* /*context*/, int block, int layer,
                      const char* name, int count) {
    log_fetch(FetchKind::kF16, block, layer, name);
    return g_base.f16(g_base.context, block, layer, name, count);
}
const float* wrap_f32(void* /*context*/, int block, int layer,
                      const char* name, int count) {
    log_fetch(FetchKind::kF32, block, layer, name);
    return g_base.f32(g_base.context, block, layer, name, count);
}
const std::uint8_t* wrap_ffwd(void* /*context*/, int block, int layer,
                              int which, int rows, int cols) {
    log_fetch(FetchKind::kFfwd, block, layer, "ffwd");
    return g_base.ffwd(g_base.context, block, layer, which, rows, cols);
}

void bind_logged(dlss5_nr::EvalWeights& w) {
    w = g_base;
    w.e4m3 = wrap_e4m3;
    w.f16 = wrap_f16;
    w.f32 = wrap_f32;
    w.ffwd = wrap_ffwd;
    w.int4 = serve_int4;
    w.swin = serve_swin;
}

// ---- the output snapshot/hash helpers ----------------------------------
std::uint64_t hash_view(const dlss5_nr::FeatureBatchView& v) {
    std::uint64_t h = 1469598103934665603ull;
    for (int c = 0; c < v.channels; ++c) {
        for (int r = 0; r < v.rows; ++r) {
            for (int n = 0; n < v.cols; ++n) {
                const float value =
                    v.data[c * v.plane_stride + r * v.row_stride + n];
                std::uint32_t bits = 0;
                std::memcpy(&bits, &value, sizeof(bits));
                h ^= bits;
                h *= 1099511628211ull;
            }
        }
    }
    return h;
}

void copy_view(const dlss5_nr::FeatureBatchView& v, float* dst) {
    std::size_t k = 0;
    for (int c = 0; c < v.channels; ++c) {
        for (int r = 0; r < v.rows; ++r) {
            for (int n = 0; n < v.cols; ++n) {
                dst[k++] =
                    v.data[c * v.plane_stride + r * v.row_stride + n];
            }
        }
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
void* operator new[](std::size_t n, std::align_val_t a) {
    return ::operator new(n, a);
}
void operator delete(void* p) noexcept {
    if (p != nullptr) {
        g_live.fetch_sub(1);
    }
    std::free(p);
}
void operator delete[](void* p) noexcept { ::operator delete(p); }
void operator delete(void* p, std::size_t) noexcept { ::operator delete(p); }
void operator delete[](void* p, std::size_t) noexcept {
    ::operator delete(p);
}
void operator delete(void* p, std::align_val_t) noexcept {
    ::operator delete(p);
}
void operator delete[](void* p, std::align_val_t) noexcept {
    ::operator delete(p);
}
void operator delete(void* p, std::size_t, std::align_val_t) noexcept {
    ::operator delete(p);
}
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept {
    ::operator delete(p);
}

namespace {

int g_frame_w = 64;
int g_frame_h = 72;

struct RunOut {
    std::uint64_t hash = 0;
    dlss5_nr::Status status = dlss5_nr::Status::kOk;
};

std::uint64_t g_row_hashes[dlss5_nr::kPlanMaxLayers];

dlss5_nr::Status traced_layer(void* context, const dlss5_nr::LayerSpec& spec,
                              const dlss5_nr::LayerCall& call,
                              dlss5_nr::LayerValue& out) {
    const dlss5_nr::Status st =
        dlss5_nr::eval_layer(context, spec, call, out);
    if (st == dlss5_nr::Status::kOk) {
        g_row_hashes[spec.index] = out.main.data != nullptr
            ? hash_view(out.main)
            : 0ull;
    }
    return st;
}

// One network run over the plan with `w`'s routing, snapshotting the final
// map (stride-aware) and pinning the allocation balance and log capacity.
RunOut run_once(dlss5_nr::EvalWeights& w,
                const dlss5_nr::FeatureBatchView& image,
                const dlss5_nr::LayerSpec* specs, int count,
                dlss5_nr::LayerValue* outputs, int* skip_scratch,
                int* last_scratch, float* arena, std::size_t arena_floats,
                float* snapshot, dlss5_nr::Status expected) {
    RunOut out;
    static dlss5_nr::EvalContext context;
    context.weights = &w;
    context.scratch = arena;
    context.scratch_capacity = arena_floats;
    CHECK_STATUS(dlss5_nr::eval_reset(context), dlss5_nr::Status::kOk);
    dlss5_nr::NetworkRun run{};
    run.outputs = outputs;
    run.skip_sources = skip_scratch;
    run.last_uses = last_scratch;
    // Every row's output is hashed as it is produced (the released arena
    // blocks make post-run row reads stale): the row-level discriminators
    // need live values.
    run.layer = traced_layer;
    run.release = dlss5_nr::eval_release;
    // The by-value copy is ours to wire (image_map_evaluation's contract):
    // the rows read their records and side channel through this context.
    run.context = &context;
    run.image = &image;
    // The binary16 side channel's caller-owned buffers (one per key).
    run.stash = g_stash;
    for (int i = 0; i < count; ++i) {
        outputs[i] = dlss5_nr::LayerValue();
    }
    g_log_count = 0;
    g_log_overflow = false;
    const long long live_before = g_live.load();
    out.status = dlss5_nr::network_run(specs, count, run);
    CHECK_STATUS(out.status, expected);
    // Balanced accounting: the parallel layer's spawn-and-join thread state
    // is churn (measured, not asserted to zero); balance is the contract.
    CHECK(g_live.load() == live_before);
    CHECK(!g_log_overflow);
    if (out.status != expected || expected != dlss5_nr::Status::kOk) {
        return out;
    }
    out.hash = hash_view(outputs[count - 1].main);
    if (snapshot != nullptr) {
        copy_view(outputs[count - 1].main, snapshot);
    }
    if (std::getenv("DLSS5_I4_ROWS") != nullptr) {
        for (int i = 0; i < count; ++i) {
            std::fprintf(stderr, "rowhash %2d %016llx\n", i,
                         static_cast<unsigned long long>(g_row_hashes[i]));
        }
    }
    return out;}

// The per-row record-consumption policy. `mixed` rows must never fetch
// their e4m3 twin; plain rows must. The expansion row's missing "weight"
// fetch IS the ("stream", block+1) side-channel routing proof (the int4
// expansion path is gated on the producer's stamp); the FIRST vit block is
// the pinned's documented fallback (no producer writes its key) and must
// fetch "weight" in every config.
void pin_policy(const dlss5_nr::LayerSpec* specs, int count, bool mixed,
                bool alternate, bool has_ex, bool mixed_split,
                int first_vit_block) {
    for (int i = 0; i < count; ++i) {
        const dlss5_nr::LayerSpec& row = specs[i];
        const bool allows = !alternate
            || (row.block - first_vit_block) % 2 == 0;
        const bool on = mixed && allows;
        switch (row.kind) {
            case dlss5_nr::LayerKind::kVitFfnExpand: {
                // The int4 path needs the site's has_ex AND the
                // ("stream", block) key (the stamp's block match).
                const bool int4_path = on && has_ex
                    && row.block != first_vit_block;
                CHECK(log_has(row.block, row.layer, "weight") != int4_path);
                break;
            }
            case dlss5_nr::LayerKind::kVitFfnContract: {
                CHECK(log_has(row.block, row.layer, "weight") != on);
                CHECK(log_has(row.block, row.layer, "tail") == on);
                CHECK(log_has(row.block, row.layer, "skip_weight"));
                break;
            }
            case dlss5_nr::LayerKind::kVitQKV: {
                CHECK(log_has(row.block, row.layer, "q") != on);
                CHECK(log_has(row.block, row.layer, "k") != on);
                CHECK(log_has(row.block, row.layer, "v") != on);
                break;
            }
            case dlss5_nr::LayerKind::kSplitQKVAttn: {
                CHECK(log_has(row.block, row.layer, "qkv") != mixed_split);
                CHECK(log_has(row.block, row.layer, "tail"));
                CHECK(log_has(row.block, row.layer, "attn_pos_bias"));
                break;
            }
            case dlss5_nr::LayerKind::kVitProjection: {
                // Never mixed: the projection matmul record is always
                // fetched.
                CHECK(log_has(row.block, row.layer, "weight"));
                break;
            }
            default:
                break;
        }
    }
}

}  // namespace

int main(void) {
    using namespace dlss5_nr;

    // The full small plan over the demo's geometry (the working floor is
    // 320x320 at any frame size; 64x72 is the proven extent walk): the
    // row/block structure and every site shape are geometry-independent
    // (the row suites pin the values).
    static LayerSpec specs[kPlanMaxLayers];
    static LayerValue outputs[kPlanMaxLayers];
    static int skip_scratch[kPlanMaxLayers];
    static int last_scratch[kPlanMaxLayers];
    int count = 0;
    CHECK_STATUS(build_plan(g_frame_w, g_frame_h, NetworkKind::kSmall, specs,
                            kPlanMaxLayers, count),
                 Status::kOk);
    CHECK(count > 70);
    PlanPadding padding{};
    CHECK_STATUS(plan_padding(g_frame_w, g_frame_h, NetworkKind::kSmall,
                              padding),
                 Status::kOk);
    CHECK(padding.width <= 320 && padding.height <= 320);
    CHECK(padding.width >= g_frame_w && padding.height >= g_frame_h);
    for (int i = 0; i < count; ++i) {
        if (specs[i].kind == LayerKind::kVitFfnExpand
            && g_first_vit_block < 0) {
            g_first_vit_block = specs[i].block;
        }
    }
    CHECK(g_first_vit_block >= 0);
    g_specs = specs;
    g_spec_count = count;

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
    const std::size_t arena_floats = 1u << 27;
    float* arena = static_cast<float*>(
        std::malloc(arena_floats * sizeof(float)));
    const std::size_t out_floats = 3u
        * static_cast<std::size_t>(padding.height)
        * static_cast<std::size_t>(padding.width);
    static float snapshot_a[3 * 320 * 320];
    static float snapshot_b[3 * 320 * 320];
    if (image_storage == nullptr || store_scratch == nullptr
        || carrier_map == nullptr || carrier_map_replay == nullptr
        || arena == nullptr) {
        std::fprintf(stderr, "harness allocation failed\n");
        return 1;
    }
    CHECK(out_floats <= 3u * 320u * 320u);
    // The binary16 side channel's caller-owned buffers: one per key, sized
    // for the largest [rows * cols, channels] map (the StashBuffers
    // contract). Left UNINITIALISED on purpose: the stamp gate must keep a
    // never-written buffer from being read (the pinned's absent-dict-key
    // fallback).
    {
        std::size_t stash_floats = 1;
        for (int i = 0; i < count; ++i) {
            const std::size_t cells = static_cast<std::size_t>(specs[i].out_w)
                * static_cast<std::size_t>(specs[i].out_h);
            const int wide = specs[i].c_in > specs[i].c_out ? specs[i].c_in
                                                           : specs[i].c_out;
            if (cells * static_cast<std::size_t>(wide) > stash_floats) {
                stash_floats = cells * static_cast<std::size_t>(wide);
            }
        }
        float* hidden = static_cast<float*>(
            std::malloc(stash_floats * sizeof(float)));
        float* c = static_cast<float*>(
            std::malloc(stash_floats * sizeof(float)));
        float* stream = static_cast<float*>(
            std::malloc(stash_floats * sizeof(float)));
        float* attn = static_cast<float*>(
            std::malloc(stash_floats * sizeof(float)));
        if (hidden == nullptr || c == nullptr || stream == nullptr
            || attn == nullptr) {
            std::fprintf(stderr, "stash allocation failed\n");
            return 1;
        }
        g_stash = dlss5_nr::StashBuffers{hidden, c, stream, attn};
    }
    // A deterministic planar input frame on the binary16 grid in [0, 1).
    for (std::size_t i = 0; i < image_floats; ++i) {
        image_storage[i] = static_cast<float>((i * 37u + 11u) % 64u) / 64.0f;
    }
    const FeatureBatchView image{image_storage, 3, padding.height,
                                 padding.width,
                                 static_cast<std::ptrdiff_t>(padding.height)
                                     * padding.width,
                                 padding.width};

    // The DemoStore bindings (every e4m3/f16/f32 record generatively) and
    // the 8-thread deterministic parallel layer.
    g_store.preset = dlss5_demo::DemoStore::Preset::kSeeded;
    g_store.seed = 20261028u;
    dlss5_demo::demo_bind_weights(g_base, g_store);
    dlss5_nr::set_parallel_threads(8);

    dlss5_nr::EvalWeights weights;
    RunOut out;
    std::uint64_t hash_a = 0;
    std::uint64_t hash_mixed = 0;

    // ---- A: the pure e4m3 network (no sites) ----------------------------
    // The baseline routing: every twin record present, determinism, and the
    // reference hash (the mixed configs must differ from it).
    bind_logged(weights);
    weights.int4 = nullptr;
    weights.swin = g_base.swin;
    out = run_once(weights, image, specs, count, outputs, skip_scratch,
                   last_scratch, arena, arena_floats, snapshot_a,
                   Status::kOk);
    pin_policy(specs, count, false, false, false, false,
                   g_first_vit_block);
    hash_a = out.hash;
    out = run_once(weights, image, specs, count, outputs, skip_scratch,
                   last_scratch, arena, arena_floats, snapshot_b,
                   Status::kOk);
    CHECK(out.hash == hash_a);
    CHECK(std::memcmp(snapshot_a, snapshot_b,
                      out_floats * sizeof(float)) == 0);
    if (std::getenv("DLSS5_I4_LOG") != nullptr) {
        for (int i = 0; i < count; ++i) {
            std::fprintf(stderr, "row %2d b%2d l%d kind=%s\n", i,
                         specs[i].block, specs[i].layer,
                         layer_kind_name(specs[i].kind));
        }
        for (int i = 0; i < g_log_count; ++i) {
            std::fprintf(stderr, "fetch b%2d l%d %s\n", g_log[i].block,
                         g_log[i].layer, g_log[i].name);
        }
    }

    // ---- B: the fully mixed network (vit + split + swin, has_ex) --------
    // Every supported row takes its int4 site; the record policy pins the
    // routing — including the expansion rows' missing "weight" fetch, which
    // only happens when the ("stream", block+1) chain delivered the
    // producer's stash end to end.
    g_policy = Policy{};
    g_policy.mixed_vit = true;
    g_policy.mixed_split = true;
    g_policy.mixed_swin = true;
    g_policy.has_ex = true;
    g_site_count = 0;
    g_swin_count = 0;
    bind_logged(weights);
    out = run_once(weights, image, specs, count, outputs, skip_scratch,
                   last_scratch, arena, arena_floats, snapshot_a,
                   Status::kOk);
    pin_policy(specs, count, true, false, true, true,
                   g_first_vit_block);
    hash_mixed = out.hash;
    if (std::getenv("DLSS5_I4_LOG") != nullptr) {
        std::fprintf(stderr, "hash_a=%016llx hash_mixed=%016llx\n",
                     static_cast<unsigned long long>(hash_a),
                     static_cast<unsigned long long>(hash_mixed));
    }
    CHECK(hash_mixed != hash_a);   // the sites change the computation
    out = run_once(weights, image, specs, count, outputs, skip_scratch,
                   last_scratch, arena, arena_floats, snapshot_b,
                   Status::kOk);
    CHECK(out.hash == hash_mixed);
    CHECK(std::memcmp(snapshot_a, snapshot_b,
                      out_floats * sizeof(float)) == 0);
    // B's per-row hashes are the reference for the row-level
    // discriminators below (the final map cannot see vit-row-only changes:
    // the synthetic weights collapse the bottleneck's late rows to zero,
    // so the routing shows in the ROW values, not the output frame).
    static std::uint64_t b_rows[dlss5_nr::kPlanMaxLayers];
    std::memcpy(b_rows, g_row_hashes, sizeof(b_rows));
    const int row_exp11 = row_index(specs, count, g_first_vit_block, 0);
    const int row_con11 = row_index(specs, count, g_first_vit_block, 1);
    const int row_exp12 = row_index(specs, count, g_first_vit_block + 1, 0);
    const int row_con12 = row_index(specs, count, g_first_vit_block + 1, 1);
    CHECK(row_exp11 >= 0 && row_con11 == row_exp11 + 1
          && row_exp12 == row_con11 + 4 && row_con12 == row_exp12 + 1);

    // ---- C: the alternating mixed network -------------------------------
    // Sites on alternating vit blocks only: the unserved blocks must fetch
    // their twins (the presence gate spans the plan), and the served blocks'
    // expansions must still run int4 — their ("stream", block+1) stash is
    // written by the UNSERVED blocks' projections, so this pins cross-mode
    // side-channel routing.
    g_policy = Policy{};
    g_policy.mixed_vit = true;
    g_policy.mixed_split = true;
    g_policy.mixed_swin = true;
    g_policy.has_ex = true;
    g_policy.alternate = true;
    g_site_count = 0;
    g_swin_count = 0;
    bind_logged(weights);
    out = run_once(weights, image, specs, count, outputs, skip_scratch,
                   last_scratch, arena, arena_floats, nullptr, Status::kOk);
    pin_policy(specs, count, true, true, true, true,
                   g_first_vit_block);
    // The cross-mode side-channel routing is pinned by the record policy
    // (the served blocks' expansions run int4 with a stash produced by the
    // UNSERVED blocks' e4m3 projections); the row values: the shared first
    // block matches B exactly, the first unserved block diverges.
    CHECK(g_row_hashes[row_exp11] == b_rows[row_exp11]);
    CHECK(g_row_hashes[row_con11] == b_rows[row_con11]);
    CHECK(g_row_hashes[row_exp12] != b_rows[row_exp12]);
    CHECK(g_row_hashes[row_con12] != b_rows[row_con12]);
    CHECK(out.hash != hash_a);

    // ---- D: the negative control (has_ex off) ---------------------------
    // The sites stay but the expansion tables are absent: every expansion
    // must fall back to the e4m3 "weight" fetch — proving the B/C pins
    // discriminate and the side channel's gate is the table's has_ex.
    g_policy = Policy{};
    g_policy.mixed_vit = true;
    g_policy.mixed_split = true;
    g_policy.mixed_swin = true;
    g_policy.has_ex = false;
    g_site_count = 0;
    g_swin_count = 0;
    bind_logged(weights);
    out = run_once(weights, image, specs, count, outputs, skip_scratch,
                   last_scratch, arena, arena_floats, nullptr, Status::kOk);
    pin_policy(specs, count, true, false, false, true,
                   g_first_vit_block);
    // The has_ex gate shows at the same rows: the shared fallback matches
    // B, the producer-backed expansion diverges (e4m3 here).
    CHECK(g_row_hashes[row_exp11] == b_rows[row_exp11]);
    CHECK(g_row_hashes[row_con11] == b_rows[row_con11]);
    CHECK(g_row_hashes[row_exp12] != b_rows[row_exp12]);
    CHECK(out.hash != hash_a);

    // ---- E: the network-level rejection ---------------------------------
    // One wrong-shaped contract site: the run must fail at ITS row with
    // kInvalidValues; earlier rows keep their outputs, the failing row and
    // everything after stay untouched (the empty values they were handed).
    {
        int fail_row = -1;
        for (int i = 0; i < count; ++i) {
            if (specs[i].kind == LayerKind::kVitFfnContract
                && specs[i].block == g_first_vit_block + 1) {
                fail_row = i;
            }
        }
        CHECK(fail_row >= 0);
        g_policy = Policy{};
        g_policy.mixed_vit = true;
        g_policy.has_ex = true;
        g_policy.break_contract = g_first_vit_block + 1;
        g_site_count = 0;
        g_swin_count = 0;
        bind_logged(weights);
        out = run_once(weights, image, specs, count, outputs, skip_scratch,
                       last_scratch, arena, arena_floats, nullptr,
                       Status::kInvalidValues);
        if (fail_row >= 0) {
            CHECK(fail_row > 0);
            CHECK(outputs[fail_row - 1].main.data != nullptr);
            CHECK(outputs[fail_row].main.data == nullptr);
            if (fail_row + 1 < count) {
                CHECK(outputs[fail_row + 1].main.data == nullptr);
            }
        }
    }

    // ---- F: the temporal consumer over the mixed network ----------------
    // Phase 1's frame step (reset -> feed -> run -> store -> commit) over
    // Phase 3's routing: the counter schedule, and a bit-identical replay
    // on a fresh carrier.
    {
        g_policy = Policy{};
        g_policy.mixed_vit = true;
        g_policy.mixed_split = true;
        g_policy.mixed_swin = true;
        g_policy.has_ex = true;
        g_site_count = 0;
        g_swin_count = 0;
        bind_logged(weights);
        static std::uint8_t host[3 * 64 * 72];
        static std::uint8_t out_host[3 * 64 * 72];
        static std::uint8_t out_replay[3 * 64 * 72];
        static std::uint8_t snap[4][3 * 64 * 72];
        TemporalCarrier carrier{};
        TemporalCarrier carrier_replay{};
        CHECK_STATUS(temporal_init(carrier, carrier_map, image_floats,
                                   padding.height, padding.width),
                     Status::kOk);
        CHECK_STATUS(temporal_init(carrier_replay, carrier_map_replay,
                                   image_floats, padding.height,
                                   padding.width),
                     Status::kOk);
        for (int frame = 0; frame < 2; ++frame) {
            for (std::size_t i = 0; i < sizeof(host); ++i) {
                host[i] = static_cast<std::uint8_t>((i * 13u + 7u
                                                     + 31u * frame) % 256u);
            }
            static EvalContext context;
            context.weights = &weights;
            context.scratch = arena;
            context.scratch_capacity = arena_floats;
            CHECK_STATUS(eval_reset(context), Status::kOk);
            NetworkRun run{};
            run.outputs = outputs;
            run.skip_sources = skip_scratch;
            run.last_uses = last_scratch;
            run.layer = dlss5_nr::eval_layer;
            run.release = dlss5_nr::eval_release;
            run.stash = g_stash;
            CHECK_STATUS(temporal_image_evaluation(
                             host, 3 * g_frame_w, g_frame_w, g_frame_h,
                             out_host, 3 * g_frame_w, image_storage,
                             image_floats, specs, count, run, context,
                             carrier, frame == 0, store_scratch, image_floats),
                         Status::kOk);
            // The counter: zero at the reset frame, then advancing.
            CHECK(temporal_counter(carrier)
                  == static_cast<std::uint32_t>(frame + 1));
            std::memcpy(snap[frame], out_host, sizeof(out_host));
        }
        // The replay on the fresh carrier is bit-identical.
        for (int frame = 0; frame < 2; ++frame) {
            for (std::size_t i = 0; i < sizeof(host); ++i) {
                host[i] = static_cast<std::uint8_t>((i * 13u + 7u
                                                     + 31u * frame) % 256u);
            }
            static EvalContext context;
            context.weights = &weights;
            context.scratch = arena;
            context.scratch_capacity = arena_floats;
            CHECK_STATUS(eval_reset(context), Status::kOk);
            NetworkRun run{};
            run.outputs = outputs;
            run.skip_sources = skip_scratch;
            run.last_uses = last_scratch;
            run.layer = dlss5_nr::eval_layer;
            run.release = dlss5_nr::eval_release;
            run.stash = g_stash;
            CHECK_STATUS(temporal_image_evaluation(
                             host, 3 * g_frame_w, g_frame_w, g_frame_h,
                             out_replay, 3 * g_frame_w, image_storage,
                             image_floats, specs, count, run, context,
                             carrier_replay, frame == 0, store_scratch,
                             image_floats),
                         Status::kOk);
            std::memcpy(snap[2 + frame], out_replay, sizeof(out_replay));
        }
        CHECK(temporal_counter(carrier_replay) == 2u);
        // The replay is bit-identical frame for frame (both frames; the
        // reset frame's host read-back is a divergence class of its own and
        // is compared directly, not only through the committed carrier).
        for (int frame = 0; frame < 2; ++frame) {
            CHECK(std::memcmp(snap[frame], snap[2 + frame],
                              sizeof(out_host)) == 0);
        }
    }

    if (g_failures == 0) {
        std::printf("native-eval-int4-network: %d checks passed\n", g_checks);
        return 0;
    }
    std::fprintf(stderr, "native-eval-int4-network: %d check(s) FAILED\n",
                 g_failures);
    return 1;
}
