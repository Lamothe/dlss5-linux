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
        static float storage[3 * kRows * kCols + 16 * 3];
        static dlss5_nr::HostFrame frame;
        const Status st = dlss5_nr::frame_from_u8(
            src, 3, kRows, kCols, storage,
            sizeof(storage) / sizeof(storage[0]), frame);
        CHECK(st == Status::kOk);
        const Status st2 = dlss5_nr::frame_to_u8(frame, dst);
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
