// Independent tests for the native Swin window partition (vr/native-swin-windows.hpp).
// CPU-only, no GPU, headset, proprietary payload, Python or PyTorch runtime required.
//
// Independence: these are pure layout transforms (no arithmetic), so the controls are
// direct: hand-derived grid literals, token-index pins (hand literals in the shifted
// fixtures — the unshifted 64-token fixture is a re-derivation of the documented
// formula, not an independent oracle), the zero-pad/crop rules for both shifts, a
// bit-exact round-trip oracle over randomised shapes/strides (including NaN payloads
// and infinities), locality and channel-permutation controls, sentinel-checked
// untouched outputs on every validation rejection class of all three routines (the
// null-output rejection cannot write by construction; null-input rejections with live
// outputs are checked untouched too), and stride-padding invariants. The token index
// formula
//   t = (py / 4) * 32 + (px / 4) * 16 + (py % 4) * 4 + (px % 4)
// is pinned by the known answers; see vr/NATIVE-SWIN-WINDOWS.md.

#include "native-swin-windows.hpp"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>
#include <new>

// ---- allocation accounting ----------------------------------------------------------------------
static std::atomic<unsigned long long> g_allocations{0};
void* operator new(size_t n) {
    g_allocations.fetch_add(1);
    void* p = malloc(n ? n : 1);
    if (!p) {
        abort();
    }
    return p;
}
void* operator new[](size_t n) { return ::operator new(n); }
void* operator new(size_t n, std::align_val_t) {
    g_allocations.fetch_add(1);
    void* p = malloc(n ? n : 1);
    if (!p) {
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

// ---- test harness -------------------------------------------------------------------------------
static unsigned long long g_checks = 0;
#define CHECK(condition) do { assert(condition); ++g_checks; } while (0)

static uint32_t g_rng = 0x13579bdfu;
static uint32_t rng(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}

static constexpr float kSentinel = 12345.0f;
static constexpr int kGridSentinel = 12345;

// The pad constant is pinned as +0.0 (bit 0x00000000): a plain == 0.0f comparison
// would also accept -0.0.
static bool is_positive_zero(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof bits);
    return bits == 0u;
}

static float special_value(uint32_t draw) {
    // Bit patterns: ordinary values, signed zeros, infinities, several NaN payloads.
    switch (draw & 15u) {
        case 0: {
            const uint32_t bits = 0x7fc00000u | ((draw >> 8) & 0x7ffffu);
            float value;
            memcpy(&value, &bits, sizeof value);
            return value;
        }
        case 1: {
            const uint32_t bits = 0xffc00042u;
            float value;
            memcpy(&value, &bits, sizeof value);
            return value;
        }
        case 2: return (draw & 0x80000000u) ? -0.0f : 0.0f;
        case 3: return (draw & 0x80000000u) ? -INFINITY : INFINITY;
        default: {
            const int32_t signed_value = (int32_t)((draw >> 4) % 2001u) - 1000;
            return (float)signed_value * 0.015625f;
        }
    }
}

// ---- grid literals --------------------------------------------------------------------------------
static void check_grid_literals(void) {
    using dlss5_nr::swin_window_grid;
    int windows = -1;
    int front = -1;
    CHECK(swin_window_grid(8, 0, &windows, &front) == dlss5_nr::Status::kOk);
    CHECK(windows == 1 && front == 0);
    CHECK(swin_window_grid(9, 0, &windows, &front) == dlss5_nr::Status::kOk);
    CHECK(windows == 2 && front == 0);
    CHECK(swin_window_grid(1, 0, &windows, &front) == dlss5_nr::Status::kOk);
    CHECK(windows == 1 && front == 0);
    CHECK(swin_window_grid(12, 0, &windows, &front) == dlss5_nr::Status::kOk);
    CHECK(windows == 2 && front == 0);
    CHECK(swin_window_grid(4, -1, &windows, &front) == dlss5_nr::Status::kOk);
    CHECK(windows == 1 && front == 4);
    CHECK(swin_window_grid(5, -1, &windows, &front) == dlss5_nr::Status::kOk);
    CHECK(windows == 2 && front == 4);
    CHECK(swin_window_grid(12, -1, &windows, &front) == dlss5_nr::Status::kOk);
    CHECK(windows == 2 && front == 4);
    CHECK(swin_window_grid(13, -1, &windows, &front) == dlss5_nr::Status::kOk);
    CHECK(windows == 3 && front == 4);
    CHECK(swin_window_grid(16, -1, &windows, &front) == dlss5_nr::Status::kOk);
    CHECK(windows == 3 && front == 4);
    // The maximum extent one axis can carry.
    CHECK(swin_window_grid(dlss5_nr::kMaxRows, -1, &windows, &front)
        == dlss5_nr::Status::kOk);
    CHECK(windows == 2049 && front == 4);
    // The documented batch bound: the largest grid is that axis bound squared.
    CHECK((long long)windows * windows == (long long)dlss5_nr::kMaxWindows);
    // Invalid shifts (the reference raises for anything but 0 and -1). Rejections
    // leave both outputs untouched (marked with sentinel values here).
    windows = kGridSentinel;
    front = kGridSentinel;
    CHECK(swin_window_grid(8, 1, &windows, &front) == dlss5_nr::Status::kInvalidControls);
    CHECK(swin_window_grid(8, -2, &windows, &front) == dlss5_nr::Status::kInvalidControls);
    CHECK(swin_window_grid(8, 2, &windows, &front) == dlss5_nr::Status::kInvalidControls);
    CHECK(swin_window_grid(8, -3, &windows, &front) == dlss5_nr::Status::kInvalidControls);
    CHECK(windows == kGridSentinel && front == kGridSentinel);
    // Invalid extents.
    CHECK(swin_window_grid(0, 0, &windows, &front)
        == dlss5_nr::Status::kInvalidDimensions);
    CHECK(swin_window_grid(-1, 0, &windows, &front)
        == dlss5_nr::Status::kInvalidDimensions);
    CHECK(swin_window_grid(dlss5_nr::kMaxRows + 1, 0, &windows, &front)
        == dlss5_nr::Status::kInvalidDimensions);
    CHECK(windows == kGridSentinel && front == kGridSentinel);
    // Null outputs.
    CHECK(swin_window_grid(8, 0, nullptr, &front) == dlss5_nr::Status::kNullPointer);
    CHECK(swin_window_grid(8, 0, &windows, nullptr) == dlss5_nr::Status::kNullPointer);
    CHECK(windows == kGridSentinel && front == kGridSentinel);
}

// ---- known answers ---------------------------------------------------------------------------------
// x(c, py, px) = 100c + 10py + px over a 2 x 5 x 3 batch. The unshifted fixture
// below re-derives the expected tokens from the documented index formula (a mirrored
// loop, not an independent oracle); the genuinely hand-derived pins are the shifted
// fixtures that follow.
static float expected_pixel(int c, int py, int px) {
    return (float)(100 * c + 10 * py + px);
}

static void check_known_answers_unshifted(void) {
    using dlss5_nr::FeatureBatchView;
    using dlss5_nr::Status;
    using dlss5_nr::swin_to_windows;
    constexpr int kChannels = 2;
    constexpr int kHeight = 5;
    constexpr int kWidth = 3;
    float source[kChannels * kHeight * kWidth];
    for (int c = 0; c < kChannels; ++c) {
        for (int py = 0; py < kHeight; ++py) {
            for (int px = 0; px < kWidth; ++px) {
                source[(c * kHeight + py) * kWidth + px] = expected_pixel(c, py, px);
            }
        }
    }
    FeatureBatchView x;
    x.data = source;
    x.channels = kChannels;
    x.rows = kHeight;
    x.cols = kWidth;
    x.plane_stride = kHeight * kWidth;
    x.row_stride = kWidth;
    // Padded token strides: token_stride = channels + 3, window_stride = 64 * 5 + 2.
    float tokens[1 * (64 * 5 + 2)];
    for (size_t i = 0; i < sizeof tokens / sizeof tokens[0]; ++i) {
        tokens[i] = kSentinel;
    }
    dlss5_nr::WindowGrid grid;
    dlss5_nr::WindowTokenView view;
    view.data = tokens;
    view.windows = -1;
    view.channels = kChannels;
    view.window_stride = 64 * 5 + 2;
    view.token_stride = 5;
    CHECK(swin_to_windows(x, 0, 0, &grid, &view, 64 * 5 + 2) == Status::kOk);
    CHECK(grid.window_rows == 1 && grid.window_cols == 1);
    CHECK(grid.front_h == 0 && grid.front_w == 0);
    CHECK(view.windows == 1);
    for (int t = 0; t < 64; ++t) {
        const int qy = (t / 32) & 1;
        const int qx = (t / 16) & 1;
        const int y = (t / 4) % 4;
        const int px = t % 4;
        const int padded_y = qy * 4 + y;
        const int padded_x = qx * 4 + px;
        for (int c = 0; c < kChannels; ++c) {
            if (padded_y < kHeight && padded_x < kWidth) {
                CHECK(tokens[t * 5 + c] == expected_pixel(c, padded_y, padded_x));
            } else {
                // The pad is pinned as +0.0, not merely == 0.
                CHECK(is_positive_zero(tokens[t * 5 + c]));
            }
        }
        // Token stride padding is never written.
        CHECK(tokens[t * 5 + 2] == kSentinel);
        CHECK(tokens[t * 5 + 3] == kSentinel);
        CHECK(tokens[t * 5 + 4] == kSentinel);
    }
    // Window stride padding is never written (no extra windows here, so the tail only).
    CHECK(tokens[64 * 5 + 0] == kSentinel);
    CHECK(tokens[64 * 5 + 1] == kSentinel);
}

static void check_known_answers_shifted(void) {
    using dlss5_nr::Status;
    using dlss5_nr::swin_to_windows;
    constexpr int kChannels = 2;
    constexpr int kHeight = 5;
    constexpr int kWidth = 3;
    float source[kChannels * kHeight * kWidth];
    for (int c = 0; c < kChannels; ++c) {
        for (int py = 0; py < kHeight; ++py) {
            for (int px = 0; px < kWidth; ++px) {
                source[(c * kHeight + py) * kWidth + px] = expected_pixel(c, py, px);
            }
        }
    }
    dlss5_nr::FeatureBatchView x;
    x.data = source;
    x.channels = kChannels;
    x.rows = kHeight;
    x.cols = kWidth;
    x.plane_stride = kHeight * kWidth;
    x.row_stride = kWidth;
    // Grid: 2 window rows x 1 window column, 4 zero pixels in front of each axis.
    float tokens[2 * 64 * kChannels];
    for (size_t i = 0; i < sizeof tokens / sizeof tokens[0]; ++i) {
        tokens[i] = kSentinel;
    }
    dlss5_nr::WindowGrid grid;
    dlss5_nr::WindowTokenView view;
    view.data = tokens;
    view.windows = -1;
    view.channels = kChannels;
    view.window_stride = 64 * kChannels;
    view.token_stride = kChannels;
    CHECK(swin_to_windows(x, -1, -1, &grid, &view, 2 * 64 * kChannels) == Status::kOk);
    CHECK(grid.window_rows == 2 && grid.window_cols == 1);
    CHECK(grid.front_h == 4 && grid.front_w == 4);
    CHECK(view.windows == 2);
    // Hand-derived pins (padded coordinate = source coordinate + 4):
    // window 0, t = 0 -> padded (0, 0) -> source (-4, -4): pad.
    CHECK(is_positive_zero(tokens[0]) && is_positive_zero(tokens[1]));
    // window 0, t = 48 (qy=1, qx=1, y=0, x=0) -> padded (4, 4) -> source (0, 0).
    CHECK(tokens[48 * kChannels + 0] == expected_pixel(0, 0, 0));
    CHECK(tokens[48 * kChannels + 1] == expected_pixel(1, 0, 0));
    // window 0, t = 49 (qy=1, qx=1, y=0, x=1) -> padded (4, 5) -> source (0, 1)
    // (pins the x=1 decode cell).
    CHECK(tokens[49 * kChannels + 0] == expected_pixel(0, 0, 1));
    CHECK(tokens[49 * kChannels + 1] == expected_pixel(1, 0, 1));
    // window 0, t = 52 (qy=1, qx=1, y=1, x=0) -> padded (5, 4) -> source (1, 0)
    // (pins the y=1 decode cell).
    CHECK(tokens[52 * kChannels + 0] == expected_pixel(0, 1, 0));
    // window 0, t = 32 (qy=1, qx=0) -> padded (4, 0) -> source (0, -4): pad.
    CHECK(is_positive_zero(tokens[32 * kChannels + 0]));
    // window 0, t = 35 (qy=1, qx=0, y=0, x=3) -> padded (4, 3) -> source (0, -1): pad.
    CHECK(is_positive_zero(tokens[35 * kChannels + 0]));
    // window 0, t = 63 (qy=1, qx=1, y=3, x=3) -> padded (7, 7) -> source (3, 3): pad
    // (the source is only 5 x 3).
    CHECK(is_positive_zero(tokens[63 * kChannels + 0]));
    // window 0, t = 51 (qy=1, qx=1, y=0, x=3) -> padded (4, 7) -> source (0, 3): pad.
    CHECK(is_positive_zero(tokens[51 * kChannels + 0]));
    // window 0, t = 50 (qy=1, qx=1, y=0, x=2) -> padded (4, 6) -> source (0, 2).
    CHECK(tokens[50 * kChannels + 0] == expected_pixel(0, 0, 2));
    // window 1 (start 64 tokens in), t = 16 (qy=0, qx=1, y=0, x=0) -> padded (8, 4)
    // -> source (4, 0).
    CHECK(tokens[(64 + 16) * kChannels + 0] == expected_pixel(0, 4, 0));
    CHECK(tokens[(64 + 16) * kChannels + 1] == expected_pixel(1, 4, 0));
    // window 1, t = 19 (qy=0, qx=1, y=0, x=3) -> padded (8, 7) -> source (4, 3): pad.
    CHECK(is_positive_zero(tokens[(64 + 19) * kChannels + 0]));
    // window 1, t = 18 (qy=0, qx=1, y=0, x=2) -> padded (8, 6) -> source (4, 2).
    CHECK(tokens[(64 + 18) * kChannels + 0] == expected_pixel(0, 4, 2));
    // window 1, t = 24 (qy=0, qx=1, y=2, x=0) -> padded (10, 4) -> source (6, 0): pad.
    CHECK(is_positive_zero(tokens[(64 + 24) * kChannels + 0]));
}

// ---- window-batch order and shift-axis pins ---------------------------------------------------------
// A 2x2-window grid (9 x 7 batch, both shifts) pins the window-batch index order
// w = wy * window_cols + wx absolutely: windows (0, 1) and (1, 0) hold different
// hand-derived values, so a transposed (wx * window_rows + wy) order would fail. The
// asymmetric-shift fixture (5 x 3, shift_rows = -1, shift_cols = 0) pins the shift
// axis assignment: swapped axes would move the front zeros to the columns.
static void check_known_answers_window_order(void) {
    using dlss5_nr::FeatureBatchView;
    using dlss5_nr::Status;
    using dlss5_nr::swin_to_windows;
    constexpr int kChannels = 2;
    constexpr int kHeight = 9;
    constexpr int kWidth = 7;
    float source[kChannels * kHeight * kWidth];
    for (int c = 0; c < kChannels; ++c) {
        for (int py = 0; py < kHeight; ++py) {
            for (int px = 0; px < kWidth; ++px) {
                source[(c * kHeight + py) * kWidth + px] = expected_pixel(c, py, px);
            }
        }
    }
    FeatureBatchView x;
    x.data = source;
    x.channels = kChannels;
    x.rows = kHeight;
    x.cols = kWidth;
    x.plane_stride = kHeight * kWidth;
    x.row_stride = kWidth;
    float tokens[4 * 64 * kChannels];
    for (size_t i = 0; i < sizeof tokens / sizeof tokens[0]; ++i) {
        tokens[i] = kSentinel;
    }
    dlss5_nr::WindowGrid grid;
    dlss5_nr::WindowTokenView view;
    view.data = tokens;
    view.windows = -1;
    view.channels = kChannels;
    view.window_stride = 64 * kChannels;
    view.token_stride = kChannels;
    CHECK(swin_to_windows(x, -1, -1, &grid, &view, 4 * 64 * kChannels) == Status::kOk);
    CHECK(grid.window_rows == 2 && grid.window_cols == 2);
    CHECK(grid.front_h == 4 && grid.front_w == 4);
    CHECK(view.windows == 4);
    // window (0, 1) = batch index 1: t = 32 -> padded (4, 8) -> source (0, 4).
    CHECK(tokens[(1 * 64 + 32) * kChannels + 0] == expected_pixel(0, 0, 4));
    CHECK(tokens[(1 * 64 + 32) * kChannels + 1] == expected_pixel(1, 0, 4));
    // window (1, 0) = batch index 2: t = 16 -> padded (8, 4) -> source (4, 0).
    CHECK(tokens[(2 * 64 + 16) * kChannels + 0] == expected_pixel(0, 4, 0));
    CHECK(tokens[(2 * 64 + 16) * kChannels + 1] == expected_pixel(1, 4, 0));
    // window (1, 0), t = 17 (x = 1) -> padded (8, 5) -> source (4, 1).
    CHECK(tokens[(2 * 64 + 17) * kChannels + 0] == expected_pixel(0, 4, 1));
    // window (1, 0), t = 21 (y = 1, x = 1) -> padded (9, 5) -> source (5, 1).
    CHECK(tokens[(2 * 64 + 21) * kChannels + 0] == expected_pixel(0, 5, 1));
    // window (1, 0), t = 0 -> padded (8, 0) -> source (4, -4): pad.
    CHECK(is_positive_zero(tokens[(2 * 64 + 0) * kChannels + 0]));
    // window (1, 1) = batch index 3: t = 0 -> padded (8, 8) -> source (4, 4).
    CHECK(tokens[(3 * 64 + 0) * kChannels + 0] == expected_pixel(0, 4, 4));
    CHECK(tokens[(3 * 64 + 0) * kChannels + 1] == expected_pixel(1, 4, 4));
    // window (1, 1), t = 63 -> padded (15, 15) -> source (11, 11): pad.
    CHECK(is_positive_zero(tokens[(3 * 64 + 63) * kChannels + 0]));
    // window (0, 0) = batch index 0: t = 63 -> padded (7, 7) -> source (3, 3).
    CHECK(tokens[(0 * 64 + 63) * kChannels + 0] == expected_pixel(0, 3, 3));
}

static void check_known_answers_shift_axes(void) {
    using dlss5_nr::FeatureBatchView;
    using dlss5_nr::Status;
    using dlss5_nr::swin_to_windows;
    constexpr int kChannels = 2;
    constexpr int kHeight = 5;
    constexpr int kWidth = 3;
    float source[kChannels * kHeight * kWidth];
    for (int c = 0; c < kChannels; ++c) {
        for (int py = 0; py < kHeight; ++py) {
            for (int px = 0; px < kWidth; ++px) {
                source[(c * kHeight + py) * kWidth + px] = expected_pixel(c, py, px);
            }
        }
    }
    FeatureBatchView x;
    x.data = source;
    x.channels = kChannels;
    x.rows = kHeight;
    x.cols = kWidth;
    x.plane_stride = kHeight * kWidth;
    x.row_stride = kWidth;
    float tokens[2 * 64 * kChannels];
    for (size_t i = 0; i < sizeof tokens / sizeof tokens[0]; ++i) {
        tokens[i] = kSentinel;
    }
    dlss5_nr::WindowGrid grid;
    dlss5_nr::WindowTokenView view;
    view.data = tokens;
    view.windows = -1;
    view.channels = kChannels;
    view.window_stride = 64 * kChannels;
    view.token_stride = kChannels;
    CHECK(swin_to_windows(x, -1, 0, &grid, &view, 2 * 64 * kChannels) == Status::kOk);
    // The front zeros belong to the rows only (shift_rows = -1, shift_cols = 0).
    CHECK(grid.window_rows == 2 && grid.window_cols == 1);
    CHECK(grid.front_h == 4 && grid.front_w == 0);
    CHECK(view.windows == 2);
    // window 0, t = 0 -> padded (0, 0) -> source (-4, 0): row pad.
    CHECK(is_positive_zero(tokens[0 * kChannels + 0]));
    // window 0, t = 32 (qy = 1, qx = 0) -> padded (4, 0) -> source (0, 0): data on
    // the rows' front edge (swapped axes would make this a pad).
    CHECK(tokens[32 * kChannels + 0] == expected_pixel(0, 0, 0));
    CHECK(tokens[32 * kChannels + 1] == expected_pixel(1, 0, 0));
    // window 0, t = 35 -> padded (4, 3) -> source (0, 3): column pad (width = 3).
    CHECK(is_positive_zero(tokens[35 * kChannels + 0]));
    // window 1, t = 0 -> padded (8, 0) -> source (4, 0).
    CHECK(tokens[(64 + 0) * kChannels + 0] == expected_pixel(0, 4, 0));
    CHECK(tokens[(64 + 0) * kChannels + 1] == expected_pixel(1, 4, 0));
}

// ---- round-trip oracle -----------------------------------------------------------------------------
static void check_round_trip(void) {
    using dlss5_nr::FeatureBatchView;
    using dlss5_nr::Status;
    using dlss5_nr::swin_from_windows;
    using dlss5_nr::swin_to_windows;
    for (int i = 0; i < 500; ++i) {
        const int channels = (int)(rng() % 8u) + 1;
        const int height = (int)(rng() % 20u) + 1;
        const int width = (int)(rng() % 20u) + 1;
        const int shift_rows = (rng() & 1u) ? -1 : 0;
        const int shift_cols = (rng() & 1u) ? -1 : 0;
        const std::ptrdiff_t row_stride = width + (std::ptrdiff_t)(rng() % 3u);
        const std::ptrdiff_t plane_stride
            = (std::ptrdiff_t)height * row_stride + (std::ptrdiff_t)(rng() % 3u);
        const std::ptrdiff_t token_stride = channels + (std::ptrdiff_t)(rng() % 3u);
        // Window count: upper bound 2049 x 2049 is far beyond these shapes.
        const int max_window_rows = (height + 4 + 7) / 8;
        const int max_window_cols = (width + 4 + 7) / 8;
        const long long windows_ll = (long long)max_window_rows * max_window_cols;
        const std::ptrdiff_t window_stride
            = 64 * token_stride + (std::ptrdiff_t)(rng() % 5u);
        const std::size_t token_capacity
            = (std::size_t)(windows_ll - 1) * (std::size_t)window_stride
                + (std::size_t)63 * (std::size_t)token_stride
                + (std::size_t)channels;
        float* truth = (float*)malloc((size_t)channels * height * width * sizeof(float));
        float* source = (float*)malloc((size_t)channels
            * (size_t)plane_stride * sizeof(float));
        float* tokens = (float*)malloc(token_capacity * sizeof(float));
        float* out = (float*)malloc((size_t)channels * height * width * sizeof(float));
        assert(truth && source && tokens && out);
        for (int c = 0; c < channels; ++c) {
            for (int y = 0; y < height; ++y) {
                for (int xcol = 0; xcol < width; ++xcol) {
                    const float value = special_value(rng());
                    truth[((size_t)c * height + y) * width + xcol] = value;
                    source[(size_t)c * (size_t)plane_stride
                        + (size_t)y * (size_t)row_stride + xcol] = value;
                }
                for (int xcol = width; xcol < row_stride; ++xcol) {
                    source[(size_t)c * (size_t)plane_stride
                        + (size_t)y * (size_t)row_stride + xcol] = kSentinel;
                }
            }
            for (std::ptrdiff_t p = (std::ptrdiff_t)height * row_stride;
                    p < plane_stride; ++p) {
                source[(size_t)c * (size_t)plane_stride + p] = kSentinel;
            }
        }
        for (std::size_t p = 0; p < token_capacity; ++p) {
            tokens[p] = kSentinel;
        }
        FeatureBatchView x;
        x.data = source;
        x.channels = channels;
        x.rows = height;
        x.cols = width;
        x.plane_stride = plane_stride;
        x.row_stride = row_stride;
        dlss5_nr::WindowGrid grid;
        dlss5_nr::WindowTokenView view;
        view.data = tokens;
        view.windows = -1;
        view.channels = channels;
        view.window_stride = window_stride;
        view.token_stride = token_stride;
        CHECK(swin_to_windows(x, shift_rows, shift_cols, &grid, &view, token_capacity)
            == Status::kOk);
        CHECK(view.windows == grid.window_rows * grid.window_cols);
        CHECK(swin_from_windows(view, grid, height, width, out,
                (std::size_t)channels * height * width) == Status::kOk);
        // Bit-exact round trip (NaN payloads included).
        CHECK(memcmp(out, truth, (size_t)channels * height * width * sizeof(float)) == 0);
        // Determinism: identical inputs, bit-identical tokens.
        float* tokens_again = (float*)malloc(token_capacity * sizeof(float));
        assert(tokens_again);
        memcpy(tokens_again, tokens, token_capacity * sizeof(float));
        dlss5_nr::WindowTokenView view_again = view;
        view_again.data = tokens_again;
        view_again.windows = -1;
        CHECK(swin_to_windows(x, shift_rows, shift_cols, &grid, &view_again,
                token_capacity) == Status::kOk);
        CHECK(memcmp(tokens_again, tokens, token_capacity * sizeof(float)) == 0);
        free(tokens_again);
        free(truth);
        free(source);
        free(tokens);
        free(out);
    }
}

// ---- locality and permutation controls ---------------------------------------------------------------
static void check_locality_and_permutation(void) {
    using dlss5_nr::FeatureBatchView;
    using dlss5_nr::Status;
    using dlss5_nr::swin_to_windows;
    constexpr int kChannels = 3;
    constexpr int kHeight = 9;
    constexpr int kWidth = 7;
    float source[kChannels * kHeight * kWidth];
    float modified[kChannels * kHeight * kWidth];
    for (size_t p = 0; p < sizeof source / sizeof source[0]; ++p) {
        source[p] = (float)(int)(p % 97u) * 0.5f - 8.0f;
        modified[p] = source[p];
    }
    modified[(2 * kHeight + 4) * kWidth + 3] += 1.0f;  // one channel, one pixel
    float tokens_a[2 * 2 * 64 * kChannels];
    float tokens_b[2 * 2 * 64 * kChannels];
    FeatureBatchView x;
    x.channels = kChannels;
    x.rows = kHeight;
    x.cols = kWidth;
    x.plane_stride = kHeight * kWidth;
    x.row_stride = kWidth;
    dlss5_nr::WindowGrid grid;
    dlss5_nr::WindowTokenView a;
    a.windows = -1;
    a.channels = kChannels;
    a.window_stride = 64 * kChannels;
    a.token_stride = kChannels;
    dlss5_nr::WindowTokenView b = a;
    x.data = source;
    a.data = tokens_a;
    CHECK(swin_to_windows(x, -1, -1, &grid, &a, 2 * 2 * 64 * kChannels) == Status::kOk);
    x.data = modified;
    b.data = tokens_b;
    CHECK(swin_to_windows(x, -1, -1, &grid, &b, 2 * 2 * 64 * kChannels) == Status::kOk);
    // Exactly one token element differs: the modified pixel's channel.
    int differences = 0;
    for (int w = 0; w < a.windows; ++w) {
        for (int t = 0; t < 64; ++t) {
            for (int c = 0; c < kChannels; ++c) {
                const std::size_t p = (std::size_t)w * 64 * kChannels
                    + (std::size_t)t * kChannels + c;
                if (memcmp(&tokens_a[p], &tokens_b[p], sizeof(float)) != 0) {
                    ++differences;
                    CHECK(c == 2);
                }
            }
        }
    }
    CHECK(differences == 1);
    // Channel permutation: swapping two channels of the batch swaps exactly those
    // channel slots in every token.
    for (size_t p = 0; p < sizeof source / sizeof source[0]; ++p) {
        modified[p] = source[p];
    }
    for (int py = 0; py < kHeight; ++py) {
        for (int px = 0; px < kWidth; ++px) {
            const float first = source[(0 * kHeight + py) * kWidth + px];
            modified[(0 * kHeight + py) * kWidth + px]
                = source[(2 * kHeight + py) * kWidth + px];
            modified[(2 * kHeight + py) * kWidth + px] = first;
        }
    }
    x.data = modified;
    b.data = tokens_b;
    CHECK(swin_to_windows(x, -1, -1, &grid, &b, 2 * 2 * 64 * kChannels) == Status::kOk);
    for (int w = 0; w < a.windows; ++w) {
        for (int t = 0; t < 64; ++t) {
            const std::size_t base
                = ((std::size_t)w * 64 + t) * kChannels;
            CHECK(memcmp(&tokens_a[base + 0], &tokens_b[base + 2], sizeof(float)) == 0);
            CHECK(memcmp(&tokens_a[base + 1], &tokens_b[base + 1], sizeof(float)) == 0);
            CHECK(memcmp(&tokens_a[base + 2], &tokens_b[base + 0], sizeof(float)) == 0);
        }
    }
}

// ---- negative controls -------------------------------------------------------------------------------
static void fill_sentinel(float* buffer, std::size_t count) {
    for (std::size_t p = 0; p < count; ++p) {
        buffer[p] = kSentinel;
    }
}
static bool all_sentinel(const float* buffer, std::size_t count) {
    for (std::size_t p = 0; p < count; ++p) {
        if (memcmp(&buffer[p], &kSentinel, sizeof(float)) != 0) {
            return false;
        }
    }
    return true;
}

// A rejected swin_to_windows leaves both outputs untouched: the token buffer keeps
// its sentinel bytes, the grid keeps its sentinel fields and the window count stays
// at its sentinel value.
static bool to_windows_untouched(const float* tokens, std::size_t count,
        const dlss5_nr::WindowGrid& grid, const dlss5_nr::WindowTokenView& view) {
    return all_sentinel(tokens, count)
        && grid.window_rows == kGridSentinel && grid.window_cols == kGridSentinel
        && grid.front_h == kGridSentinel && grid.front_w == kGridSentinel
        && view.windows == -1;
}

static void check_negative_to_windows(void) {
    using dlss5_nr::FeatureBatchView;
    using dlss5_nr::Status;
    using dlss5_nr::WindowGrid;
    using dlss5_nr::WindowTokenView;
    using dlss5_nr::swin_to_windows;
    constexpr int kChannels = 2;
    constexpr int kHeight = 5;
    constexpr int kWidth = 3;
    float source[kChannels * kHeight * kWidth];
    fill_sentinel(source, kChannels * kHeight * kWidth);
    FeatureBatchView x;
    x.data = source;
    x.channels = kChannels;
    x.rows = kHeight;
    x.cols = kWidth;
    x.plane_stride = kHeight * kWidth;
    x.row_stride = kWidth;
    float tokens[64 * 5];
    fill_sentinel(tokens, 64 * 5);
    WindowGrid grid;
    grid.window_rows = kGridSentinel;
    grid.window_cols = kGridSentinel;
    grid.front_h = kGridSentinel;
    grid.front_w = kGridSentinel;
    WindowTokenView view;
    view.data = tokens;
    view.windows = -1;
    view.channels = kChannels;
    view.window_stride = 64 * 5;
    view.token_stride = 5;
    // Null pointers.
    FeatureBatchView no_data = x;
    no_data.data = nullptr;
    CHECK(swin_to_windows(no_data, 0, 0, &grid, &view, 64 * 5)
        == Status::kNullPointer);
    CHECK(swin_to_windows(x, 0, 0, nullptr, &view, 64 * 5) == Status::kNullPointer);
    CHECK(swin_to_windows(x, 0, 0, &grid, nullptr, 64 * 5) == Status::kNullPointer);
    WindowTokenView no_view = view;
    no_view.data = nullptr;
    CHECK(swin_to_windows(x, 0, 0, &grid, &no_view, 64 * 5) == Status::kNullPointer);
    CHECK(to_windows_untouched(tokens, 64 * 5, grid, view));
    // Invalid dimensions.
    FeatureBatchView bad = x;
    bad.channels = 0;
    CHECK(swin_to_windows(bad, 0, 0, &grid, &view, 64 * 5)
        == Status::kInvalidDimensions);
    bad = x;
    bad.channels = dlss5_nr::kMaxTokenChannels + 1;
    CHECK(swin_to_windows(bad, 0, 0, &grid, &view, 64 * 5)
        == Status::kInvalidDimensions);
    bad = x;
    bad.rows = 0;
    CHECK(swin_to_windows(bad, 0, 0, &grid, &view, 64 * 5)
        == Status::kInvalidDimensions);
    bad = x;
    bad.cols = dlss5_nr::kMaxCols + 1;
    CHECK(swin_to_windows(bad, 0, 0, &grid, &view, 64 * 5)
        == Status::kInvalidDimensions);
    CHECK(to_windows_untouched(tokens, 64 * 5, grid, view));
    // Invalid strides.
    bad = x;
    bad.row_stride = kWidth - 1;
    CHECK(swin_to_windows(bad, 0, 0, &grid, &view, 64 * 5)
        == Status::kInvalidStride);
    bad = x;
    bad.row_stride = dlss5_nr::kMaxCols + 1;
    CHECK(swin_to_windows(bad, 0, 0, &grid, &view, 64 * 5)
        == Status::kInvalidStride);
    bad = x;
    bad.plane_stride = kHeight * kWidth - 1;
    CHECK(swin_to_windows(bad, 0, 0, &grid, &view, 64 * 5)
        == Status::kInvalidStride);
    bad = x;
    bad.plane_stride = (std::ptrdiff_t)dlss5_nr::kMaxRows * dlss5_nr::kMaxCols + 1;
    CHECK(swin_to_windows(bad, 0, 0, &grid, &view, 64 * 5)
        == Status::kInvalidStride);
    CHECK(to_windows_untouched(tokens, 64 * 5, grid, view));
    // Invalid controls.
    CHECK(swin_to_windows(x, 1, 0, &grid, &view, 64 * 5)
        == Status::kInvalidControls);
    CHECK(swin_to_windows(x, 0, -2, &grid, &view, 64 * 5)
        == Status::kInvalidControls);
    CHECK(to_windows_untouched(tokens, 64 * 5, grid, view));
    // Token view mismatch and strides.
    WindowTokenView bad_view = view;
    bad_view.channels = kChannels + 1;
    CHECK(swin_to_windows(x, 0, 0, &grid, &bad_view, 64 * 5)
        == Status::kInvalidDimensions);
    bad_view = view;
    bad_view.token_stride = kChannels - 1;
    CHECK(swin_to_windows(x, 0, 0, &grid, &bad_view, 64 * 5)
        == Status::kInvalidStride);
    bad_view = view;
    bad_view.token_stride = dlss5_nr::kMaxTokenChannels + 1;
    CHECK(swin_to_windows(x, 0, 0, &grid, &bad_view, 64 * 5)
        == Status::kInvalidStride);
    bad_view = view;
    bad_view.window_stride = 64 * 5 - 1;
    CHECK(swin_to_windows(x, 0, 0, &grid, &bad_view, 64 * 5)
        == Status::kInvalidStride);
    bad_view = view;
    bad_view.window_stride = 64 * dlss5_nr::kMaxTokenChannels + 1;
    CHECK(swin_to_windows(x, 0, 0, &grid, &bad_view, 64 * 5)
        == Status::kInvalidStride);
    CHECK(to_windows_untouched(tokens, 64 * 5, grid, view));
    // Short capacity: one float short of the exact requirement
    // ((windows - 1) * window_stride + 63 * token_stride + channels = 317).
    fill_sentinel(tokens, 64 * 5);
    CHECK(swin_to_windows(x, 0, 0, &grid, &view, 64 * 5 - 4)
        == Status::kInvalidBuffer);
    CHECK(to_windows_untouched(tokens, 64 * 5, grid, view));
    // Exactly the requirement is accepted (the boundary is pinned from both sides).
    fill_sentinel(tokens, 64 * 5);
    CHECK(swin_to_windows(x, 0, 0, &grid, &view, 64 * 5 - 3) == Status::kOk);
    CHECK(view.windows == 1);
    // A late rejection of the controls class again leaves both outputs untouched.
    fill_sentinel(tokens, 64 * 5);
    grid.window_rows = kGridSentinel;
    grid.window_cols = kGridSentinel;
    grid.front_h = kGridSentinel;
    grid.front_w = kGridSentinel;
    view.windows = -1;
    CHECK(swin_to_windows(x, 3, 0, &grid, &view, 64 * 5)
        == Status::kInvalidControls);
    CHECK(to_windows_untouched(tokens, 64 * 5, grid, view));
    // The success path with this buffer writes only in-bounds elements.
    CHECK(swin_to_windows(x, 0, 0, &grid, &view, 64 * 5) == Status::kOk);
    CHECK(view.windows == 1);
}

static void check_negative_from_windows(void) {
    using dlss5_nr::Status;
    using dlss5_nr::WindowGrid;
    using dlss5_nr::WindowTokenView;
    using dlss5_nr::swin_from_windows;
    constexpr int kChannels = 2;
    float tokens[2 * 64 * kChannels];
    fill_sentinel(tokens, 2 * 64 * kChannels);
    WindowTokenView view;
    view.data = tokens;
    view.windows = 2;
    view.channels = kChannels;
    view.window_stride = 64 * kChannels;
    view.token_stride = kChannels;
    WindowGrid grid;
    grid.window_rows = 2;
    grid.window_cols = 1;
    grid.front_h = 4;
    grid.front_w = 4;
    float out[2 * 5 * 3];
    const std::size_t out_count = 2 * 5 * 3;
    fill_sentinel(out, out_count);
    // Null pointers.
    WindowTokenView no_view = view;
    no_view.data = nullptr;
    CHECK(swin_from_windows(no_view, grid, 5, 3, out, out_count)
        == Status::kNullPointer);
    CHECK(swin_from_windows(view, grid, 5, 3, nullptr, out_count)
        == Status::kNullPointer);
    CHECK(all_sentinel(out, out_count));
    // Invalid output dimensions.
    CHECK(swin_from_windows(view, grid, 0, 3, out, out_count)
        == Status::kInvalidDimensions);
    CHECK(swin_from_windows(view, grid, 5, dlss5_nr::kMaxCols + 1, out, out_count)
        == Status::kInvalidDimensions);
    CHECK(all_sentinel(out, out_count));
    // Invalid grids.
    WindowGrid bad_grid = grid;
    bad_grid.window_rows = 0;
    CHECK(swin_from_windows(view, bad_grid, 5, 3, out, out_count)
        == Status::kInvalidDimensions);
    bad_grid = grid;
    bad_grid.window_cols = dlss5_nr::kMaxCols / dlss5_nr::kWindow + 2;
    CHECK(swin_from_windows(view, bad_grid, 5, 3, out, out_count)
        == Status::kInvalidDimensions);
    bad_grid = grid;
    bad_grid.front_h = 2;
    CHECK(swin_from_windows(view, bad_grid, 5, 3, out, out_count)
        == Status::kInvalidDimensions);
    bad_grid = grid;
    bad_grid.front_w = -4;
    CHECK(swin_from_windows(view, bad_grid, 5, 3, out, out_count)
        == Status::kInvalidDimensions);
    // The crop must lie inside the padded plane.
    bad_grid = grid;
    bad_grid.front_h = 4;
    CHECK(swin_from_windows(view, bad_grid, 13, 3, out, out_count)
        == Status::kInvalidDimensions);
    bad_grid = grid;
    bad_grid.front_w = 4;
    CHECK(swin_from_windows(view, bad_grid, 5, 13, out, out_count)
        == Status::kInvalidDimensions);
    CHECK(all_sentinel(out, out_count));
    // Window count mismatch.
    WindowTokenView bad_view = view;
    bad_view.windows = 1;
    CHECK(swin_from_windows(bad_view, grid, 5, 3, out, out_count)
        == Status::kInvalidDimensions);
    bad_view = view;
    bad_view.channels = dlss5_nr::kMaxTokenChannels + 1;
    CHECK(swin_from_windows(bad_view, grid, 5, 3, out, out_count)
        == Status::kInvalidDimensions);
    bad_view = view;
    bad_view.token_stride = kChannels - 1;
    CHECK(swin_from_windows(bad_view, grid, 5, 3, out, out_count)
        == Status::kInvalidStride);
    bad_view = view;
    bad_view.window_stride = 64 * dlss5_nr::kMaxTokenChannels + 1;
    CHECK(swin_from_windows(bad_view, grid, 5, 3, out, out_count)
        == Status::kInvalidStride);
    CHECK(all_sentinel(out, out_count));
    // Short output buffer (one float short of channels * height * width).
    fill_sentinel(out, out_count);
    CHECK(swin_from_windows(view, grid, 5, 3, out, out_count - 1)
        == Status::kInvalidBuffer);
    CHECK(all_sentinel(out, out_count));
    // A late rejection of the dimensions class again leaves the output untouched.
    bad_grid = grid;
    bad_grid.front_h = 2;
    fill_sentinel(out, out_count);
    CHECK(swin_from_windows(view, bad_grid, 5, 3, out, out_count)
        == Status::kInvalidDimensions);
    CHECK(all_sentinel(out, out_count));
}

// ---- allocation discipline --------------------------------------------------------------------------
static void check_allocation_discipline(void) {
    using dlss5_nr::FeatureBatchView;
    using dlss5_nr::Status;
    using dlss5_nr::swin_from_windows;
    using dlss5_nr::swin_to_windows;
    const unsigned long long before = g_allocations.load();
    constexpr int kChannels = 2;
    constexpr int kHeight = 6;
    constexpr int kWidth = 6;
    float source[kChannels * kHeight * kWidth];
    float tokens[4 * 64 * kChannels];
    float out[kChannels * kHeight * kWidth];
    for (size_t p = 0; p < sizeof source / sizeof source[0]; ++p) {
        source[p] = (float)(p % 13u);
    }
    FeatureBatchView x;
    x.data = source;
    x.channels = kChannels;
    x.rows = kHeight;
    x.cols = kWidth;
    x.plane_stride = kHeight * kWidth;
    x.row_stride = kWidth;
    dlss5_nr::WindowGrid grid;
    dlss5_nr::WindowTokenView view;
    view.data = tokens;
    view.windows = -1;
    view.channels = kChannels;
    view.window_stride = 64 * kChannels;
    view.token_stride = kChannels;
    for (int i = 0; i < 64; ++i) {
        CHECK(swin_to_windows(x, -(i % 2), -((i / 2) % 2), &grid, &view,
                4 * 64 * kChannels) == Status::kOk);
        CHECK(swin_from_windows(view, grid, kHeight, kWidth, out,
                kChannels * kHeight * kWidth) == Status::kOk);
    }
    CHECK(g_allocations.load() == before);
}

int main(void) {
    check_grid_literals();
    check_known_answers_unshifted();
    check_known_answers_shifted();
    check_known_answers_window_order();
    check_known_answers_shift_axes();
    check_round_trip();
    check_locality_and_permutation();
    check_negative_to_windows();
    check_negative_from_windows();
    check_allocation_discipline();
    printf("native-swin-windows-test: %llu checks passed\n", g_checks);
    return 0;
}
