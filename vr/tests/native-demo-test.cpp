// SPDX-License-Identifier: MIT
// Copyright (c) 2026 the dlss5-linux contributors
//
// Known-answer tests for the demo support: the deterministic LCG, the
// synthetic image generators, the binary PPM (P6) codec, the contact-sheet
// layout, the synthetic weight presets and the FP16 output-texture store
// helper. Hand-derived literals only; determinism and sentinel checks;
// zero-allocation accounting on the record interface. No GPU, headset,
// proprietary payload or Python required.
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

#include "native-demo-support.hpp"

namespace {

long long g_checks = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        ++g_checks;                                                         \
        if (!(cond)) {                                                      \
            std::printf("FAILED (%d): %s\n", __LINE__, #cond);              \
            return 1;                                                       \
        }                                                                   \
    } while (0)

#define CHECK_STATUS(expr, want)                                            \
    do {                                                                    \
        ++g_checks;                                                         \
        const dlss5_nr::Status got_ = (expr);                               \
        if (got_ != (want)) {                                               \
            std::printf("FAILED (%d): %s == %s got %d want %d\n", __LINE__, \
                        #expr, #want, static_cast<int>(got_),               \
                        static_cast<int>(want));                            \
            return 1;                                                       \
        }                                                                   \
    } while (0)

std::atomic<unsigned long long> g_allocations{0};

}  // namespace

void* operator new(std::size_t n) {
    g_allocations.fetch_add(1);
    void* p = std::malloc(n ? n : 1);
    if (p == nullptr) {
        throw std::bad_alloc();
    }
    return p;
}

void operator delete(void* p) noexcept {
    std::free(p);
}

void operator delete(void* p, std::size_t) noexcept {
    std::free(p);
}

int main() {
    using namespace dlss5_demo;

    // ---- the LCG (Numerical Recipes: x' = 1664525 * x + 1013904223) -------
    {
        std::uint32_t s = 0;
        CHECK(lcg_next(s) == 1013904223u);  // 1664525 * 0 + 1013904223
        CHECK(s == 1013904223u);
        s = 1;
        CHECK(lcg_next(s) == 1015568748u);  // 1664525 * 1 + 1013904223
        // Second step: 1664525 * 1015568748 + 1013904223 mod 2^32
        //   1664525 * 1015568748 = 1690439570264700
        //   + 1013904223        = 1690440584168923
        //   mod 2^32: 393586 * 4294967296 + 1586005467
        CHECK(lcg_next(s) == 1586005467u);
        // Determinism.
        std::uint32_t a = 20261028u;
        std::uint32_t b = 20261028u;
        for (int i = 0; i < 64; ++i) {
            CHECK(lcg_next(a) == lcg_next(b));
        }
    }

    // ---- the FNV-1a key hash ---------------------------------------------
    {
        // The hash is a deterministic key mixer, not a security primitive:
        // the pinned contracts are the empty-string basis and the
        // distinct-name property.
        CHECK(fnv1a("a") != fnv1a("b"));
        CHECK(fnv1a("qkv") != fnv1a("tail"));
        CHECK(fnv1a("") == 2166136261u);  // the empty string is the offset basis
    }

    // ---- the checker generator -------------------------------------------
    {
        std::uint8_t rgb[3 * 16 * 16];
        const std::uint8_t a[3] = {1, 2, 3};
        const std::uint8_t b[3] = {4, 5, 6};
        std::memset(rgb, 0xEE, sizeof(rgb));
        fill_checker(rgb, 16, 16, 8, a, b);
        // Cell parity: (0..7, 0..7) and (8..15, 8..15) are `a`;
        // (8..15, 0..7) and (0..7, 8..15) are `b`.
        CHECK(rgb[0] == 1 && rgb[1] == 2 && rgb[2] == 3);
        CHECK(rgb[3 * 8] == 4 && rgb[3 * 8 + 1] == 5 && rgb[3 * 8 + 2] == 6);
        CHECK(rgb[3 * 16 * 8] == 4);  // (0, 8)
        CHECK(rgb[3 * (16 * 8 + 8)] == 1);  // (8, 8)
        // Exactly 16*16 pixels written (no overflow in the sentinel sense:
        // the buffer is exactly the image size).
    }

    // ---- the gradient generator ------------------------------------------
    {
        std::uint8_t rgb[3 * 5 * 3];
        fill_gradient(rgb, 5, 3);
        CHECK(rgb[0] == 0 && rgb[1] == 0 && rgb[2] == 128);      // (0, 0)
        CHECK(rgb[3 * 4] == 255 && rgb[3 * 4 + 1] == 0);         // (4, 0)
        CHECK(rgb[3 * (2 * 5)] == 0 && rgb[3 * (2 * 5) + 1] == 255);  // (0, 2)
        CHECK(rgb[3 * (2 * 5 + 4)] == 255 && rgb[3 * (2 * 5 + 4) + 1] == 255);
    }

    // ---- the PPM codec ----------------------------------------------------
    {
        const std::uint8_t pixels[3 * 2 * 2] = {10, 20, 30, 40, 50, 60,
                                                70, 80, 90, 100, 110, 120};
        std::uint8_t buf[64];
        std::size_t needed = 0;
        CHECK_STATUS(encode_ppm_p6(buf, sizeof(buf), pixels, 2, 2, needed),
                     dlss5_nr::Status::kOk);
        // "P6\n2 2\n255\n" is 11 bytes; payload 12; total 23.
        CHECK(needed == 23u);
        CHECK(std::memcmp(buf, "P6\n2 2\n255\n", 11) == 0);
        CHECK(std::memcmp(buf + 11, pixels, 12) == 0);
        // A short buffer reports the needed count and writes nothing.
        std::uint8_t small[8];
        std::memset(small, 0xCC, sizeof(small));
        CHECK_STATUS(encode_ppm_p6(small, sizeof(small), pixels, 2, 2, needed),
                     dlss5_nr::Status::kInvalidBuffer);
        CHECK(needed == 23u);
        CHECK(small[0] == 0xCC);
        // Round-trip through the decoder.
        std::uint8_t out[3 * 2 * 2];
        int w = 0;
        int h = 0;
        CHECK_STATUS(decode_ppm_p6(buf, needed, out, sizeof(out), w, h),
                     dlss5_nr::Status::kOk);
        CHECK(w == 2 && h == 2);
        CHECK(std::memcmp(out, pixels, 12) == 0);
        // A truncated payload is rejected without writing.
        std::memset(out, 0xCC, sizeof(out));
        CHECK_STATUS(decode_ppm_p6(buf, needed - 1, out, sizeof(out), w, h),
                     dlss5_nr::Status::kInvalidBuffer);
        CHECK(out[0] == 0xCC);
        // Wrong magic and a wrong max value are rejected.
        std::uint8_t bad[32];
        std::memcpy(bad, "P5\n2 2\n255\n", 11);
        CHECK_STATUS(decode_ppm_p6(bad, 11, out, sizeof(out), w, h),
                     dlss5_nr::Status::kInvalidValues);
        std::memcpy(bad, "P6\n2 2\n254\n", 11);
        std::memcpy(bad + 11, pixels, 12);
        CHECK_STATUS(decode_ppm_p6(bad, 23, out, sizeof(out), w, h),
                     dlss5_nr::Status::kInvalidValues);
    }

    // ---- the contact sheet ------------------------------------------------
    {
        const std::uint8_t before[3 * 2 * 1] = {1, 1, 1, 2, 2, 2};
        const std::uint8_t after[3 * 2 * 1] = {3, 3, 3, 4, 4, 4};
        std::uint8_t sheet[3 * 4 * 1];
        contact_sheet(sheet, before, after, 2, 1);
        CHECK(sheet[0] == 1 && sheet[3] == 2);  // left half: before
        CHECK(sheet[6] == 3 && sheet[9] == 4);  // right half: after
    }

    // ---- the synthetic weight presets ------------------------------------
    {
        static DemoStore zero;
        zero.preset = DemoStore::Preset::kZero;
        const std::uint8_t* z =
            demo_e4m3(&zero, 3, 4, "weight", 4, 4);
        CHECK(z != nullptr);
        for (int i = 0; i < 16; ++i) {
            CHECK(z[i] == 0);
        }
        const float* zv = demo_f16(&zero, 3, 4, "gain", 8);
        CHECK(zv != nullptr);
        for (int i = 0; i < 8; ++i) {
            CHECK(zv[i] == 0.0f);
        }

        static DemoStore a;
        a.preset = DemoStore::Preset::kSeeded;
        a.seed = 7u;
        static DemoStore b;
        b.preset = DemoStore::Preset::kSeeded;
        b.seed = 7u;
        const std::uint8_t* ra = demo_e4m3(&a, 1, 2, "weight", 8, 8);
        const std::uint8_t* rb = demo_e4m3(&b, 1, 2, "weight", 8, 64);
        CHECK(ra != nullptr && rb != nullptr);
        // Determinism: the same key fills the same bytes (the differing
        // second shape only changes how many bytes are filled).
        for (int i = 0; i < 64; ++i) {
            CHECK(ra[i] == rb[i]);
        }
        // The seeded code range (the saturation contract): 12..28.
        for (int i = 0; i < 64; ++i) {
            CHECK(ra[i] >= 12 && ra[i] <= 28);
        }
        // Different keys differ (some byte). The store keeps ONE buffer per
        // kind, so the first record is copied out before the second fetch
        // (a side-by-side pointer compare would alias the same memory).
        std::uint8_t first[64];
        std::memcpy(first, ra, 64);
        bool differs = false;
        const std::uint8_t* other =
            demo_e4m3(&a, 1, 3, "weight", 8, 8);
        for (int i = 0; i < 64; ++i) {
            differs = differs || first[i] != other[i];
        }
        CHECK(differs);
        // The value records stay on the 1/1024 grid in [-1, 1).
        const float* rv = demo_f16(&a, 2, 3, "gain", 32);
        CHECK(rv != nullptr);
        for (int i = 0; i < 32; ++i) {
            CHECK(rv[i] >= -1.0f && rv[i] < 1.0f);
            CHECK(rv[i] * 1024.0f == static_cast<float>(
                        static_cast<int>(rv[i] * 1024.0f)));
        }
        // The wildcard contracts: upsample_gain at its fixed length only.
        CHECK(demo_f16(&a, 0, 0, "upsample_gain", -1) != nullptr);
        CHECK(demo_f16(&a, 0, 0, "upsample_gain", 16) != nullptr);
        CHECK(demo_f16(&a, 0, 0, "upsample_gain", 15) == nullptr);
        // Unknown names at count -1 are not wildcard-served.
        CHECK(demo_f16(&a, 0, 0, "no_such_record", -1) == nullptr);
        // ffwd records are keyed by (block, layer, which).
        const std::uint8_t* f1 = demo_ffwd(&a, 2, 3, 5, 4, 4);
        CHECK(f1 != nullptr);
        std::uint8_t fcopy[16];
        std::memcpy(fcopy, f1, 16);
        const std::uint8_t* f2 = demo_ffwd(&a, 2, 3, 5, 4, 4);
        CHECK(f2 != nullptr);
        for (int i = 0; i < 16; ++i) {
            CHECK(fcopy[i] == f2[i]);  // refill determinism through a copy
        }
        std::uint8_t ffirst[16];
        std::memcpy(ffirst, f1, 16);
        bool fdiffers = false;
        const std::uint8_t* f3 = demo_ffwd(&a, 2, 3, 6, 4, 4);
        for (int i = 0; i < 16; ++i) {
            fdiffers = fdiffers || ffirst[i] != f3[i];
        }
        CHECK(fdiffers);

        // Zero-allocation accounting across the record interface.
        const unsigned long long before_alloc = g_allocations.load();
        for (int i = 0; i < 32; ++i) {
            (void)demo_e4m3(&a, i, i, "weight", 8, 8);
            (void)demo_f16(&a, i, i, "gain", 16);
            (void)demo_f32(&a, i, i, "bias", 16);
            (void)demo_ffwd(&a, i, i, 0, 8, 8);
        }
        CHECK(g_allocations.load() == before_alloc);
    }

    // ---- the FP16 output-texture store -----------------------------------
    {
        float values[6] = {0.25f, 1.0f / 255.0f, 0.0f, 1.0f, -0.5f, 0.5f};
        CHECK_STATUS(round_frame_to_f16(values, 6), dlss5_nr::Status::kOk);
        CHECK(values[0] == 0.25f);   // exact: identity
        CHECK(values[2] == 0.0f);
        CHECK(values[3] == 1.0f);
        CHECK(values[4] == -0.5f);
        CHECK(values[5] == 0.5f);
        // 1/255 = 0.0039215686...: the binary16 interval [2^-8, 2^-7) has
        // step 2^-18; 0.0039215686... = 2^-8 + 4.0157 * 2^-18 -> 4 steps:
        // 0.0039215087890625.
        CHECK(values[1] == 0.0039215087890625f);
        // Non-finite and overflow values reject the whole frame before any
        // write: the good value before the bad one must be untouched (and
        // 1/255 WOULD change under rounding, so the check is discriminating).
        float bad[2] = {1.0f / 255.0f, 70000.0f};
        CHECK_STATUS(round_frame_to_f16(bad, 2),
                     dlss5_nr::Status::kInvalidValues);
        CHECK(bad[0] == 1.0f / 255.0f);
        float nan_v[1] = {__builtin_nanf("")};
        CHECK_STATUS(round_frame_to_f16(nan_v, 1),
                     dlss5_nr::Status::kInvalidValues);
        CHECK_STATUS(round_frame_to_f16(nullptr, 1),
                     dlss5_nr::Status::kNullPointer);
    }

    std::printf("native-demo: %lld checks passed\n", g_checks);
    return 0;
}
