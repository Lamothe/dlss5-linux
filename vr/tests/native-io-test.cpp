// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Michael Lamothe
//
// Known-answer tests for the host-image boundary (native-io.hpp) and the
// temporal history feed into the input lift (native-input-lift.hpp's history
// overload). The boundary follows the pinned reference's dlssnr/io.py at the
// campaign's MIT pin: the levels table (k/255), the `_mirror` edge padding
// (reflect an out-of-range index once about the last pixel, then clamp), the
// crop and the 8-bit round-half-up. The exact rounding table is derived in
// integer arithmetic ((m*255 + 256) / 512 over the dyadic grid m/512, whose
// float products are exact), so the expected values never share the code's
// float expression.
//
// The temporal history consumer (temporal_image_evaluation) is pinned
// against the capture contract (native-history.hpp): the run sees the
// carrier's ZERO map at start-up and after a reset (never the lift's
// single-frame stub — a probe layer records the feed), the FP16
// output-texture store commits bit-exactly (frame_to_history: the planar ->
// interleaved repack, the binary16 round and the value classes, including a
// padded-stride poison control), the frames-since-reset counter follows the
// capture's [0,1,2,3,0,1,2,3] sequence across a mid-stream reset, the reset
// ignores staged history content (injectreset), and every rejection class
// leaves out_host, the store and the carrier untouched (the documented
// reset-first failure form), with determinism and zero-allocation
// accounting.
//
// Built by vr/test-native-io.sh at -O2 and -O0 with -ffp-contract=off.
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

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

// A stub LayerFn for the image_evaluation seam: each row writes
// m(r, col, c)/32 with m = (r*7 + col*3 + c*5) % 32 over the spec's extent.
float stub_rows[2][3 * 4 * 4];
float stub_aside[3 * 4 * 4];

Status stub_layer(void* context, const LayerSpec& spec, const LayerCall& call,
                  LayerValue& out) {
    (void)context;
    (void)call;
    float* storage = stub_rows[spec.index];
    const std::size_t plane = static_cast<std::size_t>(spec.out_w) * spec.out_h;
    for (int c = 0; c < 3; ++c) {
        for (int r = 0; r < spec.out_w; ++r) {
            for (int col = 0; col < spec.out_h; ++col) {
                const int m = (r * 7 + col * 3 + c * 5) % 32;
                storage[c * plane + static_cast<std::size_t>(r) * spec.out_h
                    + col] = static_cast<float>(m) / 32.0f;
            }
        }
    }
    out.main = FeatureBatchView{storage, 3, spec.out_w, spec.out_h,
                                static_cast<std::ptrdiff_t>(plane),
                                static_cast<std::ptrdiff_t>(spec.out_h)};
    if (spec.index == 0) {
        // The pre row also sets the aside the post row's skip read needs.
        for (std::size_t i = 0; i < 3u * 4u * 4u; ++i) {
            stub_aside[i] = storage[i % plane];
        }
        out.aside = FeatureBatchView{stub_aside, 3, spec.out_w, spec.out_h,
                                     static_cast<std::ptrdiff_t>(plane),
                                     static_cast<std::ptrdiff_t>(spec.out_h)};
        out.has_aside = true;
    } else {
        out.has_aside = false;
    }
    return Status::kOk;
}

// The history-feed probe: records what the run receives at
// EvalContext::history (the consumer must wire the carrier's view — the zero
// map at start-up — never the single-frame stub, whose case is history ==
// null) and delegates to stub_layer for the deterministic output map.
bool g_seen_null = true;
int g_seen_rows = 0;
int g_seen_cols = 0;
float g_seen_copy[3 * 4 * 4];

Status probe_layer(void* context, const LayerSpec& spec,
                   const LayerCall& call, LayerValue& out) {
    const EvalContext* ctx = static_cast<const EvalContext*>(context);
    g_seen_null = ctx->history == nullptr;
    if (ctx->history != nullptr) {
        g_seen_rows = ctx->history->rows;
        g_seen_cols = ctx->history->cols;
        for (int r = 0; r < ctx->history->rows && r < 4; ++r) {
            for (int c = 0; c < 3 * ctx->history->cols && c < 12; ++c) {
                g_seen_copy[r * 12 + c] = ctx->history->rgb[
                    r * ctx->history->row_stride + c];
            }
        }
    }
    return stub_layer(context, spec, call, out);
}

// A run whose output cannot be stored (beyond the max finite binary16
// magnitude): the store's rejection class.
Status big_value_layer(void* context, const LayerSpec& spec,
                       const LayerCall& call, LayerValue& out) {
    const Status status = stub_layer(context, spec, call, out);
    if (status != Status::kOk) {
        return status;
    }
    const std::size_t plane = static_cast<std::size_t>(spec.out_w) * spec.out_h;
    for (std::size_t i = 0; i < 3 * plane; ++i) {
        stub_rows[spec.index][i] = 70000.0f;
    }
    return Status::kOk;
}

// The frame-tagged probe: the stub's map plus g_tag/4 (on the binary16
// grid), so consecutive steps commit DIFFERENT maps — the history-feed pin
// then discriminates history_N = output_{N-1} from output_{N-2} or a
// stale/mixed feed. g_tag = 0 reproduces the plain stub map.
float g_tag = 0.0f;

Status tagged_layer(void* context, const LayerSpec& spec,
                    const LayerCall& call, LayerValue& out) {
    const Status status = probe_layer(context, spec, call, out);
    if (status != Status::kOk || spec.kind != LayerKind::kPostBlock) {
        return status;
    }
    const std::size_t plane = static_cast<std::size_t>(spec.out_w) * spec.out_h;
    for (std::size_t i = 0; i < 3 * plane; ++i) {
        stub_rows[spec.index][i] += g_tag / 4.0f;
    }
    return Status::kOk;
}

// A run whose result map is not the planar-layout convention (padded
// strides): the result-geometry pin's rejection class.
Status padded_result_layer(void* context, const LayerSpec& spec,
                           const LayerCall& call, LayerValue& out) {
    const Status status = stub_layer(context, spec, call, out);
    if (status != Status::kOk) {
        return status;
    }
    if (spec.kind == LayerKind::kPostBlock) {
        out.main.plane_stride += 1;
    }
    return Status::kOk;
}

}  // namespace

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

int main() {
    // ---- end-to-end round trip on a full synthetic frame ---------------------
    // frame_from_u8 -> frame_to_u8: every input byte must survive exactly
    // (the levels round trip is exact on its own grid; the mirror padding
    // is not read back).
    {
        constexpr int kRows = 32;
        constexpr int kCols = 24;
        constexpr int kPixels = kRows * kCols;
        static std::uint8_t src[kPixels * 3];
        static std::uint8_t dst[kPixels * 3];
        for (int i = 0; i < kPixels * 3; ++i) {
            src[i] = static_cast<std::uint8_t>((i * 73 + 19) & 0xFF);
        }
        static float storage[3 * kRows * kCols];
        const Status st = dlss5_nr::frame_from_u8(
            src, 3 * kCols, kCols, kRows, storage,
            sizeof(storage) / sizeof(storage[0]), kRows, kCols);
        CHECK(st == Status::kOk);
        const Status st2 = dlss5_nr::frame_to_u8(storage, kRows, kCols, dst,
                                                 3 * kCols, kCols, kRows);
        CHECK(st2 == Status::kOk);
        CHECK(std::memcmp(src, dst, sizeof(src)) == 0);
    }

    // ---- the pinned `_mirror` index map (hand-derived tables) ----------------
    // extent 3: [0,1,2] in range; 3 -> 2*3-3-2 = 1; 4 -> 0; 5 -> -2 clamps to 0.
    {
        const int want[6] = {0, 1, 2, 1, 0, 0};
        for (int i = 0; i < 6; ++i) {
            CHECK(io_detail::mirror_index(i, 3) == want[i]);
        }
    }
    // extent 2: [0,1, 0, -1 -> 0] (the degenerate small-frame mirror).
    {
        const int want[4] = {0, 1, 0, 0};
        for (int i = 0; i < 4; ++i) {
            CHECK(io_detail::mirror_index(i, 2) == want[i]);
        }
    }
    // extent 4: [0,1,2,3, 2, 1, 0, -2 -> 0].
    {
        const int want[8] = {0, 1, 2, 3, 2, 1, 0, 0};
        for (int i = 0; i < 8; ++i) {
            CHECK(io_detail::mirror_index(i, 4) == want[i]);
        }
    }

    // ---- frame_from_u8: the levels table and the mirror padding --------------
    {
        // Host 2x3 (width 2, height 3) padded to the 4x5 working extent in
        // the chain convention (rows = height axis 4, cols = width axis 5).
        // Row map (height, extent 3): [0,1,2,1]; column map (width, extent 2):
        // [0,1,0,0,0].
        static std::uint8_t host[3][2][3];
        for (int y = 0; y < 3; ++y) {
            for (int x = 0; x < 2; ++x) {
                for (int c = 0; c < 3; ++c) {
                    host[y][x][c] = static_cast<std::uint8_t>((y * 2 + x) * 3 + c);
                }
            }
        }
        static float out[3 * 4 * 5];
        for (std::size_t i = 0; i < 3u * 4u * 5u; ++i) {
            out[i] = -777.0f;
        }
        CHECK_STATUS(frame_from_u8(&host[0][0][0], 6, 2, 3, out, 60, 4, 5),
                     Status::kOk);
        const int row_map[4] = {0, 1, 2, 1};
        const int col_map[5] = {0, 1, 0, 0, 0};
        for (int c = 0; c < 3; ++c) {
            for (int r = 0; r < 4; ++r) {
                for (int col = 0; col < 5; ++col) {
                    const float want_level =
                        static_cast<float>(static_cast<int>(
                            host[row_map[r]][col_map[col]][c])) / 255.0f;
                    CHECK(out[c * 20 + r * 5 + col] == want_level);
                }
            }
        }
        // The level mapping itself: the endpoints are exact.
        CHECK(out[0] == 0.0f);                        // host[0][0][0] = code 0
        CHECK(out[2 * 20 + 1 * 5 + 1] == 11.0f / 255.0f);  // host[2][1][2] = 11
    }

    // ---- frame_to_u8: the exact rounding table over the dyadic grid ---------
    // clamped = m/512 is exact in float and scaled = clamped*255 = m*255/512
    // is exact; round-half-up is then the integer division (m*255 + 256)/512.
    // The tie at m = 256 (clamped 0.5, scaled 127.5) pins the half-up rule.
    {
        static float in[3 * 1 * 513];
        for (int c = 0; c < 3; ++c) {
            for (int m = 0; m <= 512; ++m) {
                in[c * 513 + m] = static_cast<float>(m) / 512.0f;
            }
        }
        static std::uint8_t host[513][3];
        std::memset(&host[0][0], 7, sizeof(host));
        CHECK_STATUS(frame_to_u8(in, 1, 513, &host[0][0], 1539, 513, 1),
                     Status::kOk);
        for (int m = 0; m <= 512; ++m) {
            const int want = (m * 255 + 256) / 512;
            CHECK(static_cast<int>(host[m][0]) == want);
            CHECK(static_cast<int>(host[m][1]) == want);
            CHECK(static_cast<int>(host[m][2]) == want);
        }
        // The divergent midpoint: v = fl(0.5/255) = 8421504 * 2^-32 gives
        // scaled = 0.5 - 2^-25 exactly; the pinned frac rule rounds to 0
        // (floor(scaled + 0.5f) would give 1).
        {
            static float mid[3];
            for (int c = 0; c < 3; ++c) {
                mid[c] = 8421504.0f * 0x1p-32f;
            }
            static std::uint8_t mid_out[9];
            CHECK_STATUS(frame_to_u8(mid, 1, 3, mid_out, 9, 3, 1), Status::kOk);
            for (int c = 0; c < 3; ++c) {
                CHECK(mid_out[c] == 0u);
            }
        }
        // Clamping and the NaN default: three pixels (x = 0, 1, 2) whose
        // three planes carry -0.5, 1.5 and NaN respectively.
        static float edge[9];
        for (int c = 0; c < 3; ++c) {
            edge[c * 3 + 0] = -0.5f;
            edge[c * 3 + 1] = 1.5f;
            edge[c * 3 + 2] = std::nanf("");
        }
        static std::uint8_t edge_out[9];
        CHECK_STATUS(frame_to_u8(edge, 1, 3, edge_out, 9, 3, 1), Status::kOk);
        for (int c = 0; c < 3; ++c) {
            CHECK(edge_out[3 * 0 + c] == 0u);
            CHECK(edge_out[3 * 1 + c] == 255u);
            CHECK(edge_out[3 * 2 + c] == 0u);
        }
    }

    // ---- the 8-bit round-trip ----------------------------------------------
    // k/255 then floor(255x + 0.5) returns k for every code: the double
    // rounding error is bounded by 255 * 2^-23 (~3e-5), far below the 0.5 tie.
    {
        static float in[3 * 256];
        for (int c = 0; c < 3; ++c) {
            for (int k = 0; k < 256; ++k) {
                in[c * 256 + k] = static_cast<float>(k) / 255.0f;
            }
        }
        static std::uint8_t host[256][3];
        CHECK_STATUS(frame_to_u8(in, 1, 256, &host[0][0], 768, 256, 1),
                     Status::kOk);
        for (int k = 0; k < 256; ++k) {
            CHECK(static_cast<int>(host[k][0]) == k);
        }
    }

    // ---- boundary rejections leave the outputs untouched --------------------
    {
        static std::uint8_t host[3 * 4][4 * 3];
        static float out[3 * 4 * 5];
        for (std::size_t i = 0; i < 3u * 4u * 5u; ++i) {
            out[i] = -777.0f;
        }
        CHECK_STATUS(frame_from_u8(nullptr, 12, 4, 3, out, 60, 4, 5),
                     Status::kNullPointer);
        CHECK_STATUS(frame_from_u8(&host[0][0], 11, 4, 3, out, 60, 4, 5),
                     Status::kInvalidStride);
        CHECK_STATUS(frame_from_u8(&host[0][0], 12, 4, 3, nullptr, 60, 4, 5),
                     Status::kNullPointer);
        CHECK_STATUS(frame_from_u8(&host[0][0], 12, 4, 3, out, 59, 4, 5),
                     Status::kInvalidBuffer);
        CHECK_STATUS(frame_from_u8(&host[0][0], 12, 4, 3, out, 60, 2, 5),
                     Status::kInvalidDimensions);  // work narrower than the frame
        CHECK_STATUS(frame_from_u8(&host[0][0], 12, 0, 3, out, 60, 4, 5),
                     Status::kInvalidDimensions);
        for (std::size_t i = 0; i < 3u * 4u * 5u; ++i) {
            CHECK(out[i] == -777.0f);
        }
        static std::uint8_t sentinel[4 * 3 * 3];
        std::memset(sentinel, 7u, sizeof(sentinel));
        static float in[3 * 4 * 5] = {};
        CHECK_STATUS(frame_to_u8(nullptr, 4, 5, sentinel, 12, 4, 3),
                     Status::kNullPointer);
        CHECK_STATUS(frame_to_u8(in, 4, 5, sentinel, 11, 4, 3),
                     Status::kInvalidStride);
        CHECK_STATUS(frame_to_u8(in, 4, 5, sentinel, 18, 6, 3),
                     Status::kInvalidDimensions);
        for (std::size_t i = 0; i < sizeof(sentinel); ++i) {
            CHECK(sentinel[i] == 7u);
        }
    }

    // ---- the lift history feed (slots 7..9) ---------------------------------
    // A lift matrix whose only weight is row 0, feature 7 (the history red)
    // makes lifted[0] the centred history red; row 1, feature 4 (the current
    // red) makes lifted[1] the centred current red. The single-frame stub
    // (history == frame) puts the same value in both.
    {
        static float lift[32 * 16] = {};
        lift[0 * 16 + 7] = 1.0f;
        lift[1 * 16 + 4] = 1.0f;
        static float frame_data[3 * 2 * 2];
        static float history_data[3 * 2 * 2];
        for (int i = 0; i < 12; ++i) {
            frame_data[i] = static_cast<float>((i * 7) % 32) / 32.0f;
            history_data[i] = static_cast<float>((i * 11) % 32) / 32.0f;
        }
        static float lifted_stub[32 * 4];
        static float lifted_temporal[32 * 4];
        CHECK_STATUS(input_lift(FrameView{frame_data, 2, 2, 6}, Controls(),
                                lift, 512, lifted_stub, 128), Status::kOk);
        CHECK_STATUS(input_lift(FrameView{frame_data, 2, 2, 6},
                                FrameView{history_data, 2, 2, 6}, Controls(),
                                lift, 512, lifted_temporal, 128), Status::kOk);
        for (int y = 0; y < 2; ++y) {
            for (int x = 0; x < 2; ++x) {
                const int pixel = y * 2 + x;
                float current_centred[3] = {};
                float history_centred[3] = {};
                CHECK_STATUS(centred_colour(&frame_data[3 * pixel],
                                            current_centred), Status::kOk);
                CHECK_STATUS(centred_colour(&history_data[3 * pixel],
                                            history_centred), Status::kOk);
                const float current_red = current_centred[0];
                const float history_red = history_centred[0];
                // The stub: both rows see the current colour.
                CHECK(lifted_stub[0 * 4 + pixel] == f16(current_red));
                CHECK(lifted_stub[1 * 4 + pixel] == f16(current_red));
                // Temporal: row 0 follows the history, row 1 the current frame.
                CHECK(lifted_temporal[0 * 4 + pixel] == f16(history_red));
                CHECK(lifted_temporal[1 * 4 + pixel] == f16(current_red));
            }
        }
        // The extent must match the frame's (a properly sized mismatched
        // map: the extent check is what must fire).
        static float history3[3 * 3 * 2];
        for (std::size_t i = 0; i < 18u; ++i) {
            history3[i] = 0.25f;
        }
        CHECK_STATUS(input_lift(FrameView{frame_data, 2, 2, 6},
                                FrameView{history3, 3, 2, 6}, Controls(),
                                lift, 512, lifted_temporal, 128),
                     Status::kInvalidDimensions);
    }

    // ---- the image-level runner's validation seams --------------------------
    {
        static std::uint8_t host[3 * 2 * 2];
        static std::uint8_t out_host[3 * 2 * 2];
        static float image_storage[3 * 4 * 4];
        (void)host;
        LayerSpec bad = LayerSpec{};
        bad.kind = LayerKind::kSwin2H;
        bad.c_in = 3;
        bad.out_w = 2;
        bad.out_h = 2;
        bad.skip_block = -1;
        EvalContext ctx;
        LayerValue outputs[1];
        NetworkRun run;
        run.outputs = outputs;
        run.layer = stub_layer;
        // The run wiring is validated first.
        NetworkRun empty;
        CHECK_STATUS(image_evaluation(&host[0], 6, 2, 2, out_host, 6,
                                      image_storage, 48, &bad, 1, empty, ctx,
                                      nullptr), Status::kNullPointer);
        // Then the row kinds: not a pre block first.
        CHECK_STATUS(image_evaluation(&host[0], 6, 2, 2, out_host, 6,
                                      image_storage, 48, &bad, 1, run, ctx,
                                      nullptr), Status::kInvalidValues);
        // A pre-only plan leaves no post block to read back.
        LayerSpec pre = LayerSpec{};
        pre.kind = LayerKind::kPreBlock;
        pre.c_in = 3;
        pre.out_w = 2;
        pre.out_h = 2;
        pre.skip_block = -1;
        CHECK_STATUS(image_evaluation(&host[0], 6, 2, 2, out_host, 6,
                                      image_storage, 48, &pre, 1, run, ctx,
                                      nullptr), Status::kInvalidValues);
    }

    // ---- the image_evaluation seam through a stub layer ---------------------
    // The stub writes m(r, col, c)/32 over the plan's final extent
    // (m = (r*7 + col*3 + c*5) % 32); the expected host code is the integer
    // rounding (m*255 + 16)/32 of the cropped top-left, pinning the work
    // extent derivation, the crop, the orientation and the rounding together.
    {
        static std::uint8_t host[3 * 2 * 2];
        static std::uint8_t out_host[3 * 2 * 2];
        static float image_storage[3 * 4 * 4];
        for (std::size_t i = 0; i < 3u * 2u * 2u; ++i) {
            host[i] = static_cast<std::uint8_t>(i * 9u);
        }
        LayerSpec specs[2] = {};
        specs[0].index = 0;
        specs[0].kind = LayerKind::kPreBlock;
        specs[0].c_in = 3;
        specs[0].c_out = 3;
        specs[0].out_w = 2;
        specs[0].out_h = 2;
        specs[0].skip_block = -1;
        specs[1].index = 1;
        specs[1].kind = LayerKind::kPostBlock;
        specs[1].c_in = 3;
        specs[1].c_out = 3;
        specs[1].out_w = 4;
        specs[1].out_h = 4;
        specs[1].skip_block = 0;
        LayerValue outputs[2];
        int skip_sources[2] = {-1, 0};
        int last_uses[2] = {1, 1};
        NetworkRun run;
        run.outputs = outputs;
        run.skip_sources = skip_sources;
        run.last_uses = last_uses;
        run.layer = stub_layer;
        EvalContext ctx;
        CHECK_STATUS(image_evaluation(&host[0], 6, 2, 2, out_host, 6,
                                      image_storage, 48, specs, 2, run, ctx,
                                      nullptr), Status::kOk);
        for (int y = 0; y < 2; ++y) {
            for (int x = 0; x < 2; ++x) {
                for (int c = 0; c < 3; ++c) {
                    const int m = (y * 7 + x * 3 + c * 5) % 32;
                    const int want = (m * 255 + 16) / 32;
                    CHECK(static_cast<int>(out_host[3 * (y * 2 + x) + c])
                          == want);
                }
            }
        }
    }

    // ---- the temporal history consumer (temporal_image_evaluation) --------
    // The frame step of the capture contract (native-history.hpp):
    //   reset -> behave as if history = 0, counter = 0 (the strongest form:
    //   the carrier bytes are zeroed and the staged content is ignored);
    //   the history feed is ALWAYS the carrier's view — the zero map at
    //   start-up and after a reset — so the single-frame stub (the lift's
    //   second colour copy) never serves a temporal frame;
    //   the FP16 output-texture store commits bit-exactly and the
    //   frames-since-reset counter advances.
    // The stub layer's output map is m(r, col, c)/32 planar, so the stored
    // history at pixel (r, col), channel c, is m/32 exactly (m/32 is on the
    // binary16 grid: 5 fraction bits).
    {
        static std::uint8_t host[3 * 2 * 2];
        static std::uint8_t out_host[3 * 2 * 2];
        static float image_storage[3 * 4 * 4];
        static float carrier_map[3 * 4 * 4];
        static float store[3 * 4 * 4];
        for (std::size_t i = 0; i < 3u * 2u * 2u; ++i) {
            host[i] = static_cast<std::uint8_t>(i * 9u);
        }
        LayerSpec specs[2] = {};
        specs[0].index = 0;
        specs[0].kind = LayerKind::kPreBlock;
        specs[0].c_in = 3;
        specs[0].c_out = 3;
        specs[0].out_w = 2;
        specs[0].out_h = 2;
        specs[0].skip_block = -1;
        specs[1].index = 1;
        specs[1].kind = LayerKind::kPostBlock;
        specs[1].c_in = 3;
        specs[1].c_out = 3;
        specs[1].out_w = 4;
        specs[1].out_h = 4;
        specs[1].skip_block = 0;
        LayerValue outputs[2];
        int skip_sources[2] = {-1, 0};
        int last_uses[2] = {1, 1};
        NetworkRun run;
        run.outputs = outputs;
        run.skip_sources = skip_sources;
        run.last_uses = last_uses;
        run.layer = probe_layer;
        EvalContext ctx;
        TemporalCarrier temporal{};
        CHECK_STATUS(temporal_init(temporal, carrier_map,
                                  sizeof(carrier_map) / sizeof(float), 4, 4),
                     Status::kOk);
        run.layer = tagged_layer;
        // The hand-built interleaved maps for the tagged commits: the stub's
        // m(r,col,c)/32 plus g_tag/4 = (m + 8*tag)/32, all binary16-exact.
        static float want_tag0[3 * 4 * 4];
        static float want_tag2[3 * 4 * 4];
        static float want_tag3[3 * 4 * 4];
        for (int r = 0; r < 4; ++r) {
            for (int col = 0; col < 4; ++col) {
                for (int c = 0; c < 3; ++c) {
                    const int m = (r * 7 + col * 3 + c * 5) % 32;
                    const std::size_t p =
                        static_cast<std::size_t>(r) * 4 + col;
                    want_tag0[3 * p + c] = static_cast<float>(m) / 32.0f;
                    want_tag2[3 * p + c] =
                        static_cast<float>(m + 16) / 32.0f;
                    want_tag3[3 * p + c] =
                        static_cast<float>(m + 24) / 32.0f;
                }
            }
        }
        // Frame 0 (start-up, tag 0): the run sees the carrier's ZERO map at
        // the history feed — never the stub's null (second colour copy) case.
        g_tag = 0.0f;
        CHECK_STATUS(temporal_image_evaluation(&host[0], 6, 2, 2, out_host, 6,
                                               image_storage, 48, specs, 2,
                                               run, ctx, temporal, false,
                                               store, 48), Status::kOk);
        CHECK(!g_seen_null);
        CHECK(g_seen_rows == 4 && g_seen_cols == 4);
        for (std::size_t i = 0; i < 3u * 4u * 4u; ++i) {
            CHECK(g_seen_copy[i] == 0.0f);
        }
        CHECK(temporal_counter(temporal) == 1u);
        CHECK(temporal_has_history(temporal));
        // The commit is the FP16 output-texture store of the run's map,
        // bit-exactly (memcmp keeps the sign of zero and any rounding hop
        // visible): pixel p = r*4 + col, channel c: m(r,col,c)/32.
        CHECK(std::memcmp(store, want_tag0, sizeof(store)) == 0);
        CHECK(std::memcmp(temporal_history_view(temporal).rgb, want_tag0,
                          sizeof(store)) == 0);
        // Frame 1 (tag 2): the feed is frame 0's COMMITTED map (tag 0),
        // not frame 1's own output (tag 2) — history_N = output_{N-1}.
        g_tag = 2.0f;
        CHECK_STATUS(temporal_image_evaluation(&host[0], 6, 2, 2, out_host, 6,
                                               image_storage, 48, specs, 2,
                                               run, ctx, temporal, false,
                                               store, 48), Status::kOk);
        CHECK(!g_seen_null);
        CHECK(std::memcmp(g_seen_copy, want_tag0, sizeof(store)) == 0);
        CHECK(std::memcmp(g_seen_copy, want_tag2, sizeof(store)) != 0);
        // Frame 2 (tag 3): the feed is frame 1's committed map (tag 2) —
        // discriminating output_{N-1} from output_{N-2} (tag 0) and from
        // frame 2's own output (tag 3).
        g_tag = 3.0f;
        CHECK_STATUS(temporal_image_evaluation(&host[0], 6, 2, 2, out_host, 6,
                                               image_storage, 48, specs, 2,
                                               run, ctx, temporal, false,
                                               store, 48), Status::kOk);
        CHECK(!g_seen_null);
        CHECK(std::memcmp(g_seen_copy, want_tag2, sizeof(store)) == 0);
        CHECK(std::memcmp(g_seen_copy, want_tag0, sizeof(store)) != 0);
        CHECK(std::memcmp(g_seen_copy, want_tag3, sizeof(store)) != 0);
        CHECK(std::memcmp(temporal_history_view(temporal).rgb, want_tag3,
                          sizeof(store)) == 0);
        g_tag = 0.0f;
        // The frames-since-reset counter through the consumer: the capture's
        // [0, 1, 2, 3, 0, 1, 2, 3] sequence across a mid-stream reset at
        // frame 4 (each entry is the counter the frame EVALUATES with —
        // read before the step, zero at the reset frame by the reset
        // contract; the reset step's post-counter of 1 proves the zero).
        static float counter_map[3 * 4 * 4];
        TemporalCarrier counter_carrier{};
        CHECK_STATUS(temporal_init(counter_carrier, counter_map,
                                  sizeof(counter_map) / sizeof(float), 4, 4),
                     Status::kOk);
        std::uint32_t seen_counters[8] = {};
        for (int frame = 0; frame < 8; ++frame) {
            const bool reset_now = frame == 4;
            seen_counters[frame] = reset_now ? 0u
                : temporal_counter(counter_carrier);
            CHECK_STATUS(temporal_image_evaluation(&host[0], 6, 2, 2, out_host,
                                                   6, image_storage, 48, specs,
                                                   2, run, ctx, counter_carrier,
                                                   reset_now, store, 48),
                         Status::kOk);
            if (reset_now) {
                CHECK(temporal_counter(counter_carrier) == 1u);
            }
        }
        const std::uint32_t want_counters[8] = {0, 1, 2, 3, 0, 1, 2, 3};
        for (int frame = 0; frame < 8; ++frame) {
            CHECK(seen_counters[frame] == want_counters[frame]);
        }
        // The reset frame ignores staged history content (the capture's
        // injectreset): stage known non-zero content, reset, and the feed
        // must be the zero map again.
        for (std::size_t i = 0; i < 3u * 4u * 4u; ++i) {
            carrier_map[i] = 0.75f;
        }
        CHECK_STATUS(temporal_image_evaluation(&host[0], 6, 2, 2, out_host, 6,
                                               image_storage, 48, specs, 2,
                                               run, ctx, temporal, true,
                                               store, 48), Status::kOk);
        CHECK(!g_seen_null);
        for (std::size_t i = 0; i < 3u * 4u * 4u; ++i) {
            CHECK(g_seen_copy[i] == 0.0f);
        }
        CHECK(temporal_counter(temporal) == 1u);
    }

    // ---- frame_to_history: the store's interleaving and rounding ----------
    // The planar [c*rows*cols + r*cols + col] map becomes the interleaved
    // pixel frame [3*(r*cols + col) + c]; 1/3 rounds to the binary16 grid
    // (the significand keeps 11 bits: 1/3 = 1.0101010101|0101..._2 x 2^-2
    // rounds down to 1365/4096), m/32 is exact, and the store's value
    // classes reject before any write.
    {
        static float map[3 * 2 * 2];
        static float store[12];
        map[0] = 1.0f / 3.0f;                       // (c=0, r=0, col=0)
        map[1] = 1.0f;                              // (c=0, r=0, col=1)
        map[2] = 21.0f / 32.0f;                     // (c=0, r=1, col=0)
        map[3] = -0.0f;                             // (c=0, r=1, col=1)
        map[4] = 0.0f;                              // (c=1, r=0, col=0)
        map[5] = 65504.0f;                          // (c=1, r=0, col=1)
        map[6] = 2.0f;
        map[7] = 3.0f;
        map[8] = 4.0f;
        map[9] = 5.0f;
        map[10] = 6.0f;
        map[11] = 7.0f;
        const FeatureBatchView view{map, 3, 2, 2, 4, 2};
        for (std::size_t i = 0; i < 12; ++i) {
            store[i] = -777.0f;
        }
        CHECK_STATUS(frame_to_history(view, store, 12), Status::kOk);
        // pixel (0, 0): c = 0, 1, 2 = map[0], map[4], map[8].
        CHECK(store[0] == 1365.0f / 4096.0f);
        CHECK(store[1] == 0.0f);
        CHECK(store[2] == 4.0f);
        // pixel (0, 1): map[1], map[5], map[9].
        CHECK(store[3] == 1.0f);
        CHECK(store[4] == 65504.0f);
        CHECK(store[5] == 5.0f);
        // pixel (1, 0): map[2], map[6], map[10].
        CHECK(store[6] == 21.0f / 32.0f);
        CHECK(store[7] == 2.0f);
        CHECK(store[8] == 6.0f);
        // pixel (1, 1): map[3], map[7], map[11] — the sign of zero survives
        // the identity round (memcmp, not ==).
        CHECK(store[9] == 0.0f);
        CHECK(std::memcmp(&store[9], &map[3], sizeof(float)) == 0);
        CHECK(store[10] == 3.0f);
        CHECK(store[11] == 7.0f);
        // The value classes reject before any write.
        static float bad[12];
        for (std::size_t i = 0; i < 12; ++i) {
            bad[i] = -777.0f;
        }
        map[6] = 70000.0f;
        CHECK_STATUS(frame_to_history(view, bad, 12), Status::kInvalidValues);
        map[6] = 2.0f;
        map[7] = std::nanf("");
        CHECK_STATUS(frame_to_history(view, bad, 12), Status::kInvalidValues);
        map[7] = 3.0f;
        for (std::size_t i = 0; i < 12; ++i) {
            CHECK(bad[i] == -777.0f);
        }
        CHECK_STATUS(frame_to_history(view, nullptr, 12), Status::kNullPointer);
        CHECK_STATUS(frame_to_history(view, bad, 11), Status::kInvalidBuffer);
        const FeatureBatchView no_data{nullptr, 3, 2, 2, 4, 2};
        CHECK_STATUS(frame_to_history(no_data, bad, 12), Status::kInvalidValues);
        // A padded-stride map: the store reads through plane/row strides and
        // never the padding (poisoned with values that would change the
        // result if an r*cols-style bug read them).
        static float padded[3 * 8];
        for (std::size_t i = 0; i < 24; ++i) {
            padded[i] = -777.0f;
        }
        for (int r = 0; r < 2; ++r) {
            for (int col = 0; col < 2; ++col) {
                for (int c = 0; c < 3; ++c) {
                    const std::size_t src =
                        static_cast<std::size_t>(c) * 4
                        + static_cast<std::size_t>(r) * 2 + col;
                    padded[static_cast<std::size_t>(c) * 8
                        + static_cast<std::size_t>(r) * 3 + col] = map[src];
                }
            }
        }
        const FeatureBatchView pad_view{padded, 3, 2, 2, 8, 3};
        CHECK_STATUS(frame_to_history(pad_view, bad, 12), Status::kOk);
        CHECK(std::memcmp(bad, store, 12 * sizeof(float)) == 0);
        // The view's stride/extent classes are validated before the scan
        // (a negative stride would otherwise wrap to a huge read).
        const FeatureBatchView neg_row{map, 3, 2, 2, 4, -1};
        CHECK_STATUS(frame_to_history(neg_row, bad, 12), Status::kInvalidStride);
        const FeatureBatchView short_plane{map, 3, 2, 2, 3, 2};
        CHECK_STATUS(frame_to_history(short_plane, bad, 12),
                     Status::kInvalidStride);
        const FeatureBatchView wide_rows{map, 3, kMaxRows + 1, 1, 4, 1};
        CHECK_STATUS(frame_to_history(wide_rows, bad, 12),
                     Status::kInvalidValues);
    }

    // ---- the consumer's rejection classes leave the outputs untouched -----
    {
        static std::uint8_t host[3 * 2 * 2];
        static std::uint8_t out_host[3 * 2 * 2];
        static float image_storage[3 * 4 * 4];
        static float carrier_map[3 * 4 * 4];
        static float store[3 * 4 * 4];
        for (std::size_t i = 0; i < 3u * 2u * 2u; ++i) {
            host[i] = static_cast<std::uint8_t>(i * 9u);
        }
        LayerSpec specs[2] = {};
        specs[0].index = 0;
        specs[0].kind = LayerKind::kPreBlock;
        specs[0].c_in = 3;
        specs[0].c_out = 3;
        specs[0].out_w = 2;
        specs[0].out_h = 2;
        specs[0].skip_block = -1;
        specs[1].index = 1;
        specs[1].kind = LayerKind::kPostBlock;
        specs[1].c_in = 3;
        specs[1].c_out = 3;
        specs[1].out_w = 4;
        specs[1].out_h = 4;
        specs[1].skip_block = 0;
        LayerValue outputs[2];
        int skip_sources[2] = {-1, 0};
        int last_uses[2] = {1, 1};
        NetworkRun run;
        run.outputs = outputs;
        run.skip_sources = skip_sources;
        run.last_uses = last_uses;
        run.layer = probe_layer;
        EvalContext ctx;
        TemporalCarrier temporal{};
        CHECK_STATUS(temporal_init(temporal, carrier_map,
                                  sizeof(carrier_map) / sizeof(float), 4, 4),
                     Status::kOk);
        // First commit a frame so the carrier holds state to protect.
        CHECK_STATUS(temporal_image_evaluation(&host[0], 6, 2, 2, out_host, 6,
                                               image_storage, 48, specs, 2,
                                               run, ctx, temporal, false,
                                               store, 48), Status::kOk);
        static float carrier_copy[3 * 4 * 4];
        static float store_copy[3 * 4 * 4];
        std::memcpy(carrier_copy, carrier_map, sizeof(carrier_copy));
        std::memcpy(store_copy, store, sizeof(store_copy));
        // An unstoreable output (beyond the max finite binary16): the store
        // rejects and out_host, store and the carrier are untouched.
        std::memset(out_host, 7u, sizeof(out_host));
        run.layer = big_value_layer;
        CHECK_STATUS(temporal_image_evaluation(&host[0], 6, 2, 2, out_host, 6,
                                               image_storage, 48, specs, 2,
                                               run, ctx, temporal, false,
                                               store, 48), Status::kInvalidValues);
        CHECK(temporal_counter(temporal) == 1u);
        CHECK(std::memcmp(carrier_copy, carrier_map, sizeof(carrier_copy)) == 0);
        CHECK(std::memcmp(store_copy, store, sizeof(store_copy)) == 0);
        for (std::size_t i = 0; i < 3u * 2u * 2u; ++i) {
            CHECK(out_host[i] == 7u);
        }
        // The store and host-out geometry classes.
        run.layer = probe_layer;
        CHECK_STATUS(temporal_image_evaluation(&host[0], 6, 2, 2, out_host, 6,
                                               image_storage, 48, specs, 2,
                                               run, ctx, temporal, false,
                                               nullptr, 48), Status::kNullPointer);
        CHECK_STATUS(temporal_image_evaluation(&host[0], 6, 2, 2, out_host, 6,
                                               image_storage, 48, specs, 2,
                                               run, ctx, temporal, false,
                                               store, 47), Status::kInvalidBuffer);
        CHECK_STATUS(temporal_image_evaluation(&host[0], 6, 2, 2, out_host, 5,
                                               image_storage, 48, specs, 2,
                                               run, ctx, temporal, false,
                                               store, 48), Status::kInvalidStride);
        CHECK(temporal_counter(temporal) == 1u);
        CHECK(std::memcmp(carrier_copy, carrier_map, sizeof(carrier_copy)) == 0);
        CHECK(std::memcmp(store_copy, store, sizeof(store_copy)) == 0);
        for (std::size_t i = 0; i < 3u * 2u * 2u; ++i) {
            CHECK(out_host[i] == 7u);
        }
        // The documented reset failure form: the requested reset applies
        // first, so a failed step leaves the carrier as the reset left it.
        run.layer = big_value_layer;
        CHECK_STATUS(temporal_image_evaluation(&host[0], 6, 2, 2, out_host, 6,
                                               image_storage, 48, specs, 2,
                                               run, ctx, temporal, true,
                                               store, 48), Status::kInvalidValues);
        CHECK(temporal_counter(temporal) == 0u);
        for (std::size_t i = 0; i < 3u * 4u * 4u; ++i) {
            CHECK(carrier_map[i] == 0.0f);
        }
        // The counter-exhaustion guard is hoisted to the step's entry: a
        // carrier at UINT32_MAX cannot commit, so the step writes nothing
        // (out_host, store and the carrier all stay as they are). The reset
        // cures it (Reset -> counter = 0).
        run.layer = probe_layer;
        temporal.counter = 0xFFFFFFFFu;
        std::memset(out_host, 7u, sizeof(out_host));
        static float store_before[3 * 4 * 4];
        CHECK_STATUS(temporal_image_evaluation(&host[0], 6, 2, 2, out_host, 6,
                                               image_storage, 48, specs, 2,
                                               run, ctx, temporal, false,
                                               store, 48), Status::kInvalidValues);
        CHECK(temporal_counter(temporal) == 0xFFFFFFFFu);
        for (std::size_t i = 0; i < 3u * 2u * 2u; ++i) {
            CHECK(out_host[i] == 7u);
        }
        CHECK_STATUS(temporal_image_evaluation(&host[0], 6, 2, 2, out_host, 6,
                                               image_storage, 48, specs, 2,
                                               run, ctx, temporal, true,
                                               store, 48), Status::kOk);
        CHECK(temporal_counter(temporal) == 1u);
        // The result-map layout pin: a result map outside the planar-layout
        // convention is rejected (kInvalidBuffer) rather than misread. The
        // store baseline is the reset-cure step's commit.
        std::memcpy(store_before, store, sizeof(store_before));
        run.layer = padded_result_layer;
        std::memset(out_host, 7u, sizeof(out_host));
        CHECK_STATUS(temporal_image_evaluation(&host[0], 6, 2, 2, out_host, 6,
                                               image_storage, 48, specs, 2,
                                               run, ctx, temporal, false,
                                               store, 48), Status::kInvalidBuffer);
        CHECK(std::memcmp(store_before, store, sizeof(store_before)) == 0);
        for (std::size_t i = 0; i < 3u * 2u * 2u; ++i) {
            CHECK(out_host[i] == 7u);
        }
        // A carrier whose map never initialised is rejected at the seam.
        run.layer = probe_layer;
        TemporalCarrier broken{nullptr, 0, 4, 4, 0, false};
        CHECK_STATUS(temporal_image_evaluation(&host[0], 6, 2, 2, out_host, 6,
                                               image_storage, 48, specs, 2,
                                               run, ctx, broken, false,
                                               store, 48),
                     Status::kNullPointer);
    }

    // ---- the single-frame stub stays the no-carrier path ------------------
    // image_evaluation with no carrier is the reference's no-history path
    // (the lift's second colour copy): the run sees history == null. The
    // temporal consumer never takes this path.
    {
        static std::uint8_t host[3 * 2 * 2];
        static std::uint8_t out_host[3 * 2 * 2];
        static float image_storage[3 * 4 * 4];
        LayerSpec specs[2] = {};
        specs[0].index = 0;
        specs[0].kind = LayerKind::kPreBlock;
        specs[0].c_in = 3;
        specs[0].c_out = 3;
        specs[0].out_w = 2;
        specs[0].out_h = 2;
        specs[0].skip_block = -1;
        specs[1].index = 1;
        specs[1].kind = LayerKind::kPostBlock;
        specs[1].c_in = 3;
        specs[1].c_out = 3;
        specs[1].out_w = 4;
        specs[1].out_h = 4;
        specs[1].skip_block = 0;
        LayerValue outputs[2];
        int skip_sources[2] = {-1, 0};
        int last_uses[2] = {1, 1};
        NetworkRun run;
        run.outputs = outputs;
        run.skip_sources = skip_sources;
        run.last_uses = last_uses;
        run.layer = probe_layer;
        EvalContext ctx;
        CHECK_STATUS(image_evaluation(&host[0], 6, 2, 2, out_host, 6,
                                      image_storage, 48, specs, 2, run, ctx,
                                      nullptr), Status::kOk);
        CHECK(g_seen_null);
    }

    // ---- the consumer: determinism and zero-allocation accounting ---------
    {
        static std::uint8_t host[3 * 2 * 2];
        static std::uint8_t out_a[3 * 2 * 2];
        static std::uint8_t out_b[3 * 2 * 2];
        static float image_storage[3 * 4 * 4];
        static float map_a[3 * 4 * 4];
        static float map_b[3 * 4 * 4];
        static float store[3 * 4 * 4];
        LayerSpec specs[2] = {};
        specs[0].index = 0;
        specs[0].kind = LayerKind::kPreBlock;
        specs[0].c_in = 3;
        specs[0].c_out = 3;
        specs[0].out_w = 2;
        specs[0].out_h = 2;
        specs[0].skip_block = -1;
        specs[1].index = 1;
        specs[1].kind = LayerKind::kPostBlock;
        specs[1].c_in = 3;
        specs[1].c_out = 3;
        specs[1].out_w = 4;
        specs[1].out_h = 4;
        specs[1].skip_block = 0;
        LayerValue outputs[2];
        int skip_sources[2] = {-1, 0};
        int last_uses[2] = {1, 1};
        NetworkRun run;
        run.outputs = outputs;
        run.skip_sources = skip_sources;
        run.last_uses = last_uses;
        run.layer = probe_layer;
        EvalContext ctx;
        TemporalCarrier temp_a{};
        TemporalCarrier temp_b{};
        CHECK_STATUS(temporal_init(temp_a, map_a, 48, 4, 4), Status::kOk);
        CHECK_STATUS(temporal_init(temp_b, map_b, 48, 4, 4), Status::kOk);
        const unsigned long long before = g_allocations.load();
        for (int frame = 0; frame < 4; ++frame) {
            CHECK_STATUS(temporal_image_evaluation(&host[0], 6, 2, 2, out_a, 6,
                                                   image_storage, 48, specs, 2,
                                                   run, ctx, temp_a,
                                                   frame == 2, store, 48),
                         Status::kOk);
        }
        for (int frame = 0; frame < 4; ++frame) {
            CHECK_STATUS(temporal_image_evaluation(&host[0], 6, 2, 2, out_b, 6,
                                                   image_storage, 48, specs, 2,
                                                   run, ctx, temp_b,
                                                   frame == 2, store, 48),
                         Status::kOk);
        }
        CHECK(g_allocations.load() == before);
        CHECK(std::memcmp(out_a, out_b, sizeof(out_a)) == 0);
        CHECK(std::memcmp(map_a, map_b, sizeof(map_a)) == 0);
    }

    // ---- determinism and zero-allocation accounting -------------------------
    {
        static std::uint8_t host[3 * 3 * 3];
        for (std::size_t i = 0; i < 3u * 3u * 3u; ++i) {
            host[i] = static_cast<std::uint8_t>(i * 5u);
        }
        static float out[3 * 4 * 5];
        static std::uint8_t roundtrip[3 * 3][3 * 3];
        const unsigned long long before = g_allocations.load();
        CHECK_STATUS(frame_from_u8(&host[0], 9, 3, 3, out, 60, 4, 5),
                     Status::kOk);
        static float first[60];
        std::memcpy(first, out, sizeof(first));
        CHECK_STATUS(frame_from_u8(&host[0], 9, 3, 3, out, 60, 4, 5),
                     Status::kOk);
        CHECK(std::memcmp(first, out, sizeof(first)) == 0);
        CHECK_STATUS(frame_to_u8(out, 4, 5, &roundtrip[0][0], 9, 3, 3),
                     Status::kOk);
        CHECK(g_allocations.load() == before);
    }

    std::printf("native-io: %lld checks passed\n", g_checks);
    return 0;
}
