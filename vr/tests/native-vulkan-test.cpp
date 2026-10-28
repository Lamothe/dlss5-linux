// SPDX-License-Identifier: MIT
// Copyright (c) 2026 the dlss5-linux contributors
//
// CPU-vs-GPU parity and timing tests for the Vulkan projection matmul (G1):
// the GPU's out[t][n] = sum_k in[t][k] * w[n][k] must be BIT-IDENTICAL to
// the CPU reference matmul_e4m3 (same element order, `precise` accumulation,
// no contraction), including the denormal-intermediate path. Plus rejection
// classes with sentinel outputs, GPU determinism, and per-kernel timing
// against the CPU reference (the speed numbers are real; with synthetic
// weights the OUTPUT is arbitrary — no image-quality claim).
//
// Usage: native-vulkan-test <path-to-matmul_e4m3.spv>. Clean skip when no
// Vulkan device is present (a recorded environment gap, not a failure).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#include "native-parallel.hpp"
#include "native-vulkan.hpp"
#include "native-vit.hpp"

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

}  // namespace

int main(int argc, char** argv) {
    using namespace dlss5_nr;

    VulkanDevice device;
    bool opened = false;
    CHECK(vulkan_open(device, opened) == Status::kOk);
    if (!opened) {
        std::printf("native-vulkan: SKIP (no compute-capable Vulkan device; "
                    "recorded environment gap)\n");
        return 0;
    }
    if (argc < 2) {
        std::printf("usage: native-vulkan-test <matmul_e4m3.spv>\n");
        vulkan_close(device);
        return 2;
    }
    std::printf("native-vulkan: device %s (timestamps: %s)\n", device.name,
                device.has_timestamps ? "yes" : "no");

    // ---- load the runner-compiled SPIR-V --------------------------------
    std::vector<std::uint32_t> spirv;
    {
        std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
        CHECK(file.good());
        const std::streamsize bytes = file.tellg();
        CHECK(bytes > 0 && bytes % 4 == 0);
        file.seekg(0);
        spirv.resize(static_cast<std::size_t>(bytes) / 4);
        file.read(reinterpret_cast<char*>(spirv.data()), bytes);
        CHECK(file.good());
    }

    VulkanMatmul matmul;
    CHECK(vulkan_matmul_create(device, matmul, spirv.data(), spirv.size(),
                               4096, 1024, 4096)
          == Status::kOk);

    // ---- the hand-known case --------------------------------------------
    {
        // Codes row 0 = [8, 20, 28] -> [0.015625, 0.046875, 0.09375]; row 1
        // = the reverse. Token 0 = [1, 2, 4]: 0.015625 + 0.09375 + 0.375 =
        // 0.484375; the reverse row -> 0.25. Token 1 = [4, 2, 1]: mirrored.
        static const std::uint8_t codes[2 * 3] = {8, 20, 28, 28, 20, 8};
        float w[2 * 3];
        for (int i = 0; i < 6; ++i) {
            w[i] = e4m3_from_bits(codes[i]);
        }
        const float in[2 * 3] = {1.0f, 2.0f, 4.0f, 4.0f, 2.0f, 1.0f};
        static E4m3Matrix m;
        m.bits = codes;
        m.rows = 2;
        m.cols = 3;
        float cpu[2 * 2];
        CHECK(matmul_e4m3(in, 2, 3, m, cpu) == Status::kOk);
        CHECK(cpu[0] == 0.484375f && cpu[1] == 0.25f);
        CHECK(cpu[2] == 0.25f && cpu[3] == 0.484375f);
        float gpu[2 * 2] = {__builtin_nanf(""), __builtin_nanf(""),
                            __builtin_nanf(""), __builtin_nanf("")};
        VulkanMatmulTiming timing;
        CHECK(vulkan_matmul_run(device, matmul, in, 2, 3, w, 2, gpu, timing)
              == Status::kOk);
        CHECK(std::memcmp(cpu, gpu, sizeof(cpu)) == 0);
    }

    // ---- bit-parity on larger deterministic cases ------------------------
    {
        static std::uint8_t codes[37 * 53];
        float w[37 * 53];
        for (int i = 0; i < 37 * 53; ++i) {
            codes[i] = static_cast<std::uint8_t>(12 + (i * 7) % 17);
            w[i] = e4m3_from_bits(codes[i]);
        }
        static E4m3Matrix m;
        m.bits = codes;
        m.rows = 37;
        m.cols = 53;
        for (int tokens : {1, 2, 255, 256, 257, 300}) {
            static float in[300 * 53];
            static float cpu[300 * 37];
            static float gpu[300 * 37];
            for (int i = 0; i < tokens * 53; ++i) {
                in[i] = static_cast<float>((i * 13) % 31) / 8.0f - 2.0f;
            }
            CHECK(matmul_e4m3(in, tokens, 53, m, cpu) == Status::kOk);
            for (int i = 0; i < tokens * 37; ++i) {
                gpu[i] = __builtin_nanf("");
            }
            VulkanMatmulTiming timing;
            CHECK(vulkan_matmul_run(device, matmul, in, tokens, 53, w, 37, gpu,
                                    timing)
                  == Status::kOk);
            // Bit-identical at the padded dispatch boundaries too
            // (255/256/257 exercise the local_size padding).
            CHECK(std::memcmp(cpu, gpu,
                              sizeof(float) * static_cast<std::size_t>(tokens)
                                  * 37)
                  == 0);
        }
    }

    // ---- the documented denormal tolerance -------------------------------
    {
        // THE family tolerance, stated exactly and scoped to FINITE
        // operands (NaN payloads are not bit-stable across devices):
        //
        //  - CONTRACTUAL: when every intermediate product and partial sum
        //    stays within the fp32 normal range, the GPU result is
        //    bit-identical to the CPU reference (matmul_e4m3's key-ordered
        //    accumulation with `precise` on the GPU, -ffp-contract=off on
        //    the CPU).
        //  - MEASURED (driver-dependent, NOT contractual): when any
        //    intermediate falls below the normal range, the RTX 4070 Ti
        //    SUPER driver flushes each denormal intermediate to zero, so
        //    the GPU result can diverge from the CPU reference even when
        //    the CPU's FINAL value is normal. Vulkan leaves denormal
        //    handling unspecified unless DenormPreserve is requested;
        //    a driver update may change this. Both divergence shapes are
        //    pinned below so the behaviour is visible, not assumed.
        static const std::uint8_t codes[2 * 2] = {1, 1, 1, 1};
        float w[2 * 2];
        for (int i = 0; i < 4; ++i) {
            w[i] = e4m3_from_bits(codes[i]);
        }
        static E4m3Matrix m;
        m.bits = codes;
        m.rows = 2;
        m.cols = 2;
        const float in[2 * 2] = {0x1p-120f, 0x1p-120f, 0x1p-120f, 0x1p-120f};
        float cpu[2 * 2];
        CHECK(matmul_e4m3(in, 2, 2, m, cpu) == Status::kOk);
        CHECK(cpu[0] > 0.0f && cpu[0] < 0x1p-126f);  // a denormal sum
        float gpu[2 * 2] = {__builtin_nanf(""), __builtin_nanf(""),
                            __builtin_nanf(""), __builtin_nanf("")};
        VulkanMatmulTiming timing;
        CHECK(vulkan_matmul_run(device, matmul, in, 2, 2, w, 2, gpu, timing)
              == Status::kOk);
        // Shape 1: a denormal final result. Measured driver behaviour:
        // the GPU yields the sign-carrying flush of the CPU subnormal.
        for (int i = 0; i < 4; ++i) {
            float flushed = cpu[i];
            if (std::fpclassify(cpu[i]) == FP_SUBNORMAL) {
                flushed = std::copysign(0.0f, cpu[i]);
            }
            CHECK(std::memcmp(&gpu[i], &flushed, sizeof(float)) == 0);
        }
        // Shape 2: denormal intermediates summing to a NORMAL final. The
        // CPU sums 8 products of 2^-129 to 2^-126 (normal); the GPU's
        // flushed intermediates sum to +0. This is the documented
        // divergence class: the flush acts on INTERMEDIATES, so no
        // function of the CPU's final value predicts the GPU result.
        {
            const int k8 = 8;
            static std::uint8_t codes8w[8 * 8];
            float w8[8 * 8];
            for (int i = 0; i < 64; ++i) {
                codes8w[i] = 1;
                w8[i] = e4m3_from_bits(1);
            }
            static E4m3Matrix m8;
            m8.bits = codes8w;
            m8.rows = 8;
            m8.cols = 8;
            float in8[8 * 8];
            for (int i = 0; i < 64; ++i) {
                in8[i] = 0x1p-120f;
            }
            float cpu8[8 * 8];
            CHECK(matmul_e4m3(in8, 1, 8, m8, cpu8) == Status::kOk);
            CHECK(cpu8[0] == 0x1p-126f);  // 8 x 2^-129, normal
            float gpu8[8 * 8];
            for (int i = 0; i < k8 * k8; ++i) {
                gpu8[i] = __builtin_nanf("");
            }
            VulkanMatmulTiming timing8;
            CHECK(vulkan_matmul_run(device, matmul, in8, 1, 8, w8, 8, gpu8,
                                    timing8)
                  == Status::kOk);
            CHECK(gpu8[0] == 0.0f);  // flushed intermediates -> +0
        }
        // A normal-range control with the same weights stays bit-identical
        // (the CONTRACTUAL half of the tolerance).
        const float in2[2 * 2] = {0x1p-60f, 0x1p-60f, 0x1p-60f, 0x1p-60f};
        float cpu2[2 * 2];
        CHECK(matmul_e4m3(in2, 2, 2, m, cpu2) == Status::kOk);
        float gpu2[2 * 2] = {__builtin_nanf(""), __builtin_nanf(""),
                             __builtin_nanf(""), __builtin_nanf("")};
        CHECK(vulkan_matmul_run(device, matmul, in2, 2, 2, w, 2, gpu2, timing)
              == Status::kOk);
        CHECK(std::memcmp(cpu2, gpu2, sizeof(cpu2)) == 0);
    }

    // ---- GPU determinism -------------------------------------------------
    {
        static std::uint8_t codes[37 * 53];
        float w[37 * 53];
        for (int i = 0; i < 37 * 53; ++i) {
            codes[i] = static_cast<std::uint8_t>(12 + (i * 7) % 17);
            w[i] = e4m3_from_bits(codes[i]);
        }
        static float in[41 * 53];
        for (int i = 0; i < 41 * 53; ++i) {
            in[i] = static_cast<float>((i * 13) % 31) / 8.0f - 2.0f;
        }
        static float a[41 * 37];
        static float b[41 * 37];
        VulkanMatmulTiming timing;
        CHECK(vulkan_matmul_run(device, matmul, in, 41, 53, w, 37, a, timing)
              == Status::kOk);
        CHECK(vulkan_matmul_run(device, matmul, in, 41, 53, w, 37, b, timing)
              == Status::kOk);
        CHECK(std::memcmp(a, b, sizeof(a)) == 0);
    }

    // ---- rejection classes (sentinel-untouched) --------------------------
    {
        static float in[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        static float w[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        static float out[4] = {-777.0f, -777.0f, -777.0f, -777.0f};
        VulkanMatmulTiming timing;
        CHECK(vulkan_matmul_run(device, matmul, nullptr, 1, 2, w, 2, out,
                                timing)
              == Status::kNullPointer);
        CHECK(vulkan_matmul_run(device, matmul, in, 1, 2, w, 2, nullptr,
                                timing)
              == Status::kNullPointer);
        CHECK(vulkan_matmul_run(device, matmul, in, 0, 2, w, 2, out, timing)
              == Status::kInvalidDimensions);
        CHECK(vulkan_matmul_run(device, matmul, in, 1, 0, w, 2, out, timing)
              == Status::kInvalidDimensions);
        CHECK(vulkan_matmul_run(device, matmul, in, 1, 2, w, 0, out, timing)
              == Status::kInvalidDimensions);
        // Over-capacity shapes reject without touching the output.
        CHECK(vulkan_matmul_run(device, matmul, in, 4097, 2, w, 2, out, timing)
              == Status::kInvalidDimensions);
        for (int i = 0; i < 4; ++i) {
            CHECK(out[i] == -777.0f);
        }
    }

    // ---- per-kernel timing: the CPU reference vs the GPU -----------------
    {
        // A representative projection shape (the vit rows' class):
        // tokens x c_in x rows = 100 x 1024 x 4096.
        const int tokens = 100;
        const int c_in = 1024;
        const int rows = 4096;
        static std::uint8_t codes[4096 * 1024];
        static float w[4096 * 1024];
        static float in[100 * 1024];
        static float cpu_out[100 * 4096];
        static float gpu_out[100 * 4096];
        for (int i = 0; i < rows * c_in; ++i) {
            codes[i] = static_cast<std::uint8_t>(8 + (i * 7) % 25);
            w[i] = e4m3_from_bits(codes[i]);
        }
        static E4m3Matrix m;
        m.bits = codes;
        m.rows = rows;
        m.cols = c_in;
        for (int i = 0; i < tokens * c_in; ++i) {
            in[i] = static_cast<float>((i * 13) % 251) / 64.0f - 2.0f;
        }
        set_parallel_threads(1);
        const double cpu_ms = [&]() {
            const auto t0 = std::chrono::steady_clock::now();
            (void)matmul_e4m3(in, tokens, c_in, m, cpu_out);
            const auto t1 = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::milli>(t1 - t0).count();
        }();
        set_parallel_threads(0);
        const double cpu_mt_ms = [&]() {
            const auto t0 = std::chrono::steady_clock::now();
            (void)matmul_e4m3(in, tokens, c_in, m, cpu_out);
            const auto t1 = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::milli>(t1 - t0).count();
        }();
        set_parallel_threads(1);
        VulkanMatmulTiming timing;
        CHECK(vulkan_matmul_run(device, matmul, in, tokens, c_in, w, rows,
                                gpu_out, timing)
              == Status::kOk);
        // The full-size run must also be bit-identical.
        CHECK(std::memcmp(cpu_out, gpu_out, sizeof(cpu_out)) == 0);
        std::printf(
            "timing (%d x %d x %d): cpu-1t %.2f ms, cpu-mt %.2f ms, "
            "gpu kernel %.3f ms (upload %.3f, download %.3f) "
            "= %.1fx vs cpu-1t\n",
            tokens, c_in, rows, cpu_ms, cpu_mt_ms, timing.kernel_ms,
            timing.upload_ms, timing.download_ms,
            timing.kernel_ms > 0.0 ? cpu_ms / timing.kernel_ms : 0.0);
    }

    vulkan_matmul_destroy(device, matmul);
    vulkan_close(device);
    std::printf("native-vulkan: %lld checks passed\n", g_checks);
    return 0;
}
