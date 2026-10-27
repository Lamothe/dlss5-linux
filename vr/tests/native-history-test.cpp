// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Michael Lamothe
//
// Known-answer tests for the temporal history carrier (native-history.hpp).
// The contract is the tested-regime capture evidence of vr/TEMPORAL-HISTORY.md
// (gate stage 13): history_N = output_{N-1} bit-exactly, counter_N = frames
// since the last reset, and a reset behaves as if history = 0 and counter = 0
// (ignoring any staged history content). The pinned counter sequence
// [0,1,2,3,0,1,2,3] across a mid-stream reset at frame 4 is a direct known
// answer from that evidence. The input lift's single-frame stub is
// deliberately out of scope here.
//
// Built by vr/test-native-history.sh at -O2 and -O0 with -ffp-contract=off.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <atomic>
#include <new>

#include "native-history.hpp"

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

// Deterministic f16-grid colour pattern (multiples of 1/8: exact in
// binary16), varying with the pixel so layout errors move the values.
float pattern(int y, int x, int c, int variant) {
    return static_cast<float>(((y * 5 + x * 3 + c * 7 + variant * 11) % 8)) / 8.0f;
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
    // ---- init: the reset value is all zeros, counter 0, no history ------------
    {
        static float storage[3 * 4 * 6];
        for (std::size_t i = 0; i < 3u * 4u * 6u; ++i) {
            storage[i] = 12345.0f;
        }
        TemporalCarrier state;
        CHECK_STATUS(temporal_init(state, storage, 3u * 4u * 6u, 4, 6),
                     Status::kOk);
        CHECK(temporal_counter(state) == 0u);
        CHECK(!temporal_has_history(state));
        for (std::size_t i = 0; i < 3u * 4u * 6u; ++i) {
            CHECK(storage[i] == 0.0f);
        }
        const FrameView history = temporal_history_view(state);
        CHECK(history.rgb == storage);
        CHECK(history.rows == 4 && history.cols == 6);
        CHECK(history.row_stride == 18);
    }

    // ---- init rejections: no state write on failure ---------------------------
    {
        static float storage[3];
        for (int i = 0; i < 3; ++i) {
            storage[i] = 0.625f;
        }
        TemporalCarrier state;
        CHECK_STATUS(temporal_init(state, nullptr, 3, 1, 1), Status::kNullPointer);
        CHECK(state.map == nullptr);
        CHECK(state.map_capacity == 0);
        CHECK_STATUS(temporal_init(state, storage, 3, 0, 1),
                     Status::kInvalidDimensions);
        CHECK(state.map == nullptr && state.map_capacity == 0);
        CHECK_STATUS(temporal_init(state, storage, 3, 1, -1),
                     Status::kInvalidDimensions);
        CHECK(state.map == nullptr && state.map_capacity == 0);
        CHECK_STATUS(temporal_init(state, storage, 2, 1, 1), Status::kInvalidBuffer);
        CHECK(state.map == nullptr && state.map_capacity == 0);
        CHECK_STATUS(temporal_init(state, storage, 3, 16385, 1),
                     Status::kInvalidDimensions);
        CHECK(state.map == nullptr && state.map_capacity == 0);
        // The failure form touches the map fields only; the caller's buffer
        // is byte-untouched.
        CHECK(storage[0] == 0.625f && storage[1] == 0.625f && storage[2] == 0.625f);
    }

    // ---- a failed re-init leaves the counter and history flag untouched ------
    {
        static float storage[3 * 2 * 2];
        static float frame_storage[3 * 2 * 2];
        for (std::size_t i = 0; i < 3u * 2u * 2u; ++i) {
            frame_storage[i] = 0.25f;
        }
        TemporalCarrier state;
        CHECK_STATUS(temporal_init(state, storage, 3u * 2u * 2u, 2, 2),
                     Status::kOk);
        CHECK_STATUS(temporal_commit(state, FrameView{frame_storage, 2, 2, 6}),
                     Status::kOk);
        CHECK_STATUS(temporal_init(state, nullptr, 3, 1, 1), Status::kNullPointer);
        CHECK(state.counter == 1u);
        CHECK(state.has_history);
    }

    // ---- the pinned counter sequence across a mid-stream reset ---------------
    // TEMPORAL-HISTORY.md's capture-reset trace: the counter is
    // [0,1,2,3,0,1,2,3] over eight frames with a reset at frame 4.
    {
        static float storage[3 * 2 * 2];
        static float frame_storage[3 * 2 * 2];
        TemporalCarrier state;
        CHECK_STATUS(temporal_init(state, storage, 3u * 2u * 2u, 2, 2),
                     Status::kOk);
        std::uint32_t sequence[8] = {};
        for (int frame = 0; frame < 8; ++frame) {
            if (frame == 4) {
                CHECK_STATUS(temporal_reset(state), Status::kOk);
            }
            sequence[frame] = temporal_counter(state);
            for (std::size_t i = 0; i < 3u * 2u * 2u; ++i) {
                frame_storage[i] = static_cast<float>((frame * 3 + i) % 8) / 8.0f;
            }
            CHECK_STATUS(temporal_commit(
                             state, FrameView{frame_storage, 2, 2, 6}),
                         Status::kOk);
        }
        const std::uint32_t want[8] = {0u, 1u, 2u, 3u, 0u, 1u, 2u, 3u};
        for (int i = 0; i < 8; ++i) {
            CHECK(sequence[i] == want[i]);
        }
    }

    // ---- commit is bit-exact; history_N = output_{N-1} -----------------------
    {
        static float storage[3 * 3 * 5];
        static float frame_a[3 * 3 * 5];
        static float frame_b[3 * 3 * 5];
        for (int y = 0; y < 3; ++y) {
            for (int x = 0; x < 5; ++x) {
                for (int c = 0; c < 3; ++c) {
                    frame_a[3 * (y * 5 + x) + c] = pattern(y, x, c, 1);
                    frame_b[3 * (y * 5 + x) + c] = pattern(y, x, c, 2);
                }
            }
        }
        TemporalCarrier state;
        CHECK_STATUS(temporal_init(state, storage, 3u * 3u * 5u, 3, 5),
                     Status::kOk);
        CHECK_STATUS(temporal_commit(state, FrameView{frame_a, 3, 5, 15}),
                     Status::kOk);
        CHECK(temporal_has_history(state));
        CHECK(temporal_counter(state) == 1u);
        {
            // The history view of frame N+1 is output_N, read through the
            // view's own stride (non-square map).
            const FrameView history = temporal_history_view(state);
            CHECK(history.row_stride == 15);
            for (int y = 0; y < 3; ++y) {
                for (int x = 0; x < 5; ++x) {
                    for (int c = 0; c < 3; ++c) {
                        const float got = history.rgb[static_cast<std::size_t>(y)
                            * static_cast<std::size_t>(history.row_stride)
                            + 3 * x + c];
                        CHECK(got == pattern(y, x, c, 1));
                    }
                }
            }
        }
        CHECK_STATUS(temporal_commit(state, FrameView{frame_b, 3, 5, 15}),
                     Status::kOk);
        CHECK(temporal_counter(state) == 2u);
        for (std::size_t i = 0; i < 3u * 3u * 5u; ++i) {
            CHECK(storage[i] == frame_b[i]);
        }
    }

    // ---- bit-exactness at the grid edges (memcmp, sign of zero included) -----
    // 0.0 / -0.0 / 2^-24 / 65504 are all binary16-representable; a rounding
    // hop or a sign-of-zero flip in the commit path would move the bytes.
    {
        static float storage[3 * 1 * 4];
        static float edges[12] = {0.0f, -0.0f, 0x1p-24f, 65504.0f,
                                  -0.0f, 0.0f, 65504.0f, 0x1p-24f,
                                  0x1p-24f, 65504.0f, -0.0f, 0.0f};
        TemporalCarrier state;
        CHECK_STATUS(temporal_init(state, storage, 3u * 1u * 4u, 1, 4),
                     Status::kOk);
        CHECK_STATUS(temporal_commit(state, FrameView{edges, 1, 4, 12}),
                     Status::kOk);
        CHECK(std::memcmp(storage, edges, sizeof(edges)) == 0);
    }

    // ---- strided sources: padding is never read ------------------------------
    // The padding rows carry a NaN poison and a wild constant; a kernel that
    // indexed the source as r * cols would bring both into the carrier.
    {
        static float storage[3 * 2 * 3];
        static float padded[2 * 12];
        for (int y = 0; y < 2; ++y) {
            for (int x = 0; x < 3; ++x) {
                for (int c = 0; c < 3; ++c) {
                    padded[12 * y + 3 * x + c] = pattern(y, x, c, 3);
                }
            }
            for (int i = 9; i < 12; ++i) {
                padded[12 * y + i] = i == 9 ? std::nanf("") : 12345.0f;
            }
        }
        TemporalCarrier state;
        CHECK_STATUS(temporal_init(state, storage, 3u * 2u * 3u, 2, 3),
                     Status::kOk);
        CHECK_STATUS(temporal_commit(state, FrameView{padded, 2, 3, 12}),
                     Status::kOk);
        for (int y = 0; y < 2; ++y) {
            for (int x = 0; x < 3; ++x) {
                for (int c = 0; c < 3; ++c) {
                    CHECK(storage[3 * (y * 3 + x) + c] == pattern(y, x, c, 3));
                }
            }
        }
    }

    // ---- reset ignores the staged history (the injectreset control) ----------
    // Garbage committed before the reset has no effect: afterwards the history
    // view is the all-zero map, the flag drops and the counter restarts.
    {
        static float storage[3 * 2 * 2];
        static float garbage[3 * 2 * 2];
        for (std::size_t i = 0; i < 3u * 2u * 2u; ++i) {
            garbage[i] = 0.875f;
        }
        TemporalCarrier state;
        CHECK_STATUS(temporal_init(state, storage, 3u * 2u * 2u, 2, 2),
                     Status::kOk);
        CHECK_STATUS(temporal_commit(state, FrameView{garbage, 2, 2, 6}),
                     Status::kOk);
        CHECK(temporal_counter(state) == 1u);
        CHECK_STATUS(temporal_reset(state), Status::kOk);
        CHECK(temporal_counter(state) == 0u);
        CHECK(!temporal_has_history(state));
        for (std::size_t i = 0; i < 3u * 2u * 2u; ++i) {
            CHECK(storage[i] == 0.0f);
        }
        CHECK_STATUS(temporal_reset(state), Status::kOk);  // idempotent
        CHECK(temporal_counter(state) == 0u);
        CHECK(!temporal_has_history(state));
    }

    // ---- commit rejections leave the carrier untouched -----------------------
    {
        static float storage[3 * 2 * 2];
        static float frame_storage[3 * 2 * 2];
        for (std::size_t i = 0; i < 3u * 2u * 2u; ++i) {
            frame_storage[i] = 0.25f;
        }
        TemporalCarrier state;
        CHECK_STATUS(temporal_init(state, storage, 3u * 2u * 2u, 2, 2),
                     Status::kOk);
        CHECK_STATUS(temporal_commit(state, FrameView{frame_storage, 2, 2, 6}),
                     Status::kOk);
        const std::uint32_t counter_before = temporal_counter(state);
        static float sentinel[3 * 2 * 2];
        std::memcpy(sentinel, storage, sizeof(storage));

        CHECK_STATUS(temporal_commit(state, FrameView{nullptr, 2, 2, 6}),
                     Status::kNullPointer);
        CHECK_STATUS(temporal_commit(state, FrameView{frame_storage, 3, 2, 6}),
                     Status::kInvalidDimensions);
        CHECK_STATUS(temporal_commit(state, FrameView{frame_storage, 2, 3, 6}),
                     Status::kInvalidDimensions);
        CHECK_STATUS(temporal_commit(state, FrameView{frame_storage, 2, 2, 5}),
                     Status::kInvalidStride);
        frame_storage[0] = std::nanf("");
        CHECK_STATUS(temporal_commit(state, FrameView{frame_storage, 2, 2, 6}),
                     Status::kInvalidValues);
        frame_storage[0] = std::numeric_limits<float>::infinity();
        CHECK_STATUS(temporal_commit(state, FrameView{frame_storage, 2, 2, 6}),
                     Status::kInvalidValues);
        frame_storage[0] = 0.1f;  // not on the binary16 grid
        CHECK_STATUS(temporal_commit(state, FrameView{frame_storage, 2, 2, 6}),
                     Status::kInvalidValues);

        CHECK(temporal_counter(state) == counter_before);
        CHECK(temporal_has_history(state));
        CHECK(std::memcmp(sentinel, storage, sizeof(storage)) == 0);
    }

    // ---- reset and commit rejections: capacity and counter exhaustion --------
    {
        static float storage[3 * 2 * 2];
        static float frame_storage[3 * 2 * 2];
        for (std::size_t i = 0; i < 3u * 2u * 2u; ++i) {
            frame_storage[i] = 0.375f;
        }
        // reset after a failed init rejects and touches nothing.
        TemporalCarrier bad;
        CHECK_STATUS(temporal_init(bad, nullptr, 0, 1, 1), Status::kNullPointer);
        CHECK_STATUS(temporal_reset(bad), Status::kNullPointer);

        TemporalCarrier state;
        CHECK_STATUS(temporal_init(state, storage, 3u * 2u * 2u, 2, 2),
                     Status::kOk);
        CHECK_STATUS(temporal_commit(state, FrameView{frame_storage, 2, 2, 6}),
                     Status::kOk);
        static float sentinel[3 * 2 * 2];
        std::memcpy(sentinel, storage, sizeof(storage));
        // a capacity that no longer covers the map rejects.
        state.map_capacity = 3u * 2u * 2u - 1u;
        CHECK_STATUS(temporal_commit(state, FrameView{frame_storage, 2, 2, 6}),
                     Status::kInvalidBuffer);
        CHECK_STATUS(temporal_reset(state), Status::kInvalidBuffer);
        state.map_capacity = 3u * 2u * 2u;
        // counter exhaustion refuses rather than wrapping into the reset value.
        state.counter = 0xFFFFFFFFu;
        CHECK_STATUS(temporal_commit(state, FrameView{frame_storage, 2, 2, 6}),
                     Status::kInvalidValues);
        CHECK(state.counter == 0xFFFFFFFFu);
        CHECK(std::memcmp(sentinel, storage, sizeof(storage)) == 0);
    }

    // ---- determinism and zero-allocation accounting --------------------------
    {
        static float storage[3 * 3 * 4];
        static float frame_storage[3 * 3 * 4];
        for (int y = 0; y < 3; ++y) {
            for (int x = 0; x < 4; ++x) {
                for (int c = 0; c < 3; ++c) {
                    frame_storage[3 * (y * 4 + x) + c] = pattern(y, x, c, 4);
                }
            }
        }
        TemporalCarrier state;
        const unsigned long long before = g_allocations;
        CHECK_STATUS(temporal_init(state, storage, 3u * 3u * 4u, 3, 4),
                     Status::kOk);
        CHECK_STATUS(temporal_commit(state, FrameView{frame_storage, 3, 4, 12}),
                     Status::kOk);
        static float first[3 * 3 * 4];
        std::memcpy(first, storage, sizeof(storage));
        CHECK_STATUS(temporal_reset(state), Status::kOk);
        CHECK_STATUS(temporal_commit(state, FrameView{frame_storage, 3, 4, 12}),
                     Status::kOk);
        CHECK(std::memcmp(first, storage, sizeof(storage)) == 0);
        CHECK(g_allocations.load() == before);
    }

    std::printf("native-history: %lld checks passed\n", g_checks);
    return 0;
}
