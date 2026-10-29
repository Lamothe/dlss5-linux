// SPDX-License-Identifier: MIT
// Copyright (c) 2026 the dlss5-linux contributors
//
// GPU-path hardening / sustained-operation tests (stage 4 of the
// performance brief). Every result is checked bit-identical to an
// independently accumulated scalar reference (the same key order as the
// shader's `precise` accumulation; normal-range inputs, the documented
// GPU tolerance envelope). What this covers:
//   1. repeated device/matmul init-...-teardown cycles (no degradation,
//      balanced CPU allocations);
//   2. a long mixed-shape resident compute run (determinism, no drift);
//   3. the weight cache's content-hash mismatch path (same pointer, new
//      content -> re-upload -> correct) and capacity behaviour (> kCapacity
//      distinct keys: the ensure path rejects, existing entries stay
//      correct — no corruption);
//   4. failure paths: invalid shapes and null operands reject with Status
//      without corrupting later runs.
//
// What this does NOT cover (documented honestly in NATIVE-VULKAN-STRESS.md):
// device-loss recovery. Forcing VK_ERROR_DEVICE_LOST needs driver fault
// injection we deliberately do not do; the code's VkResult paths return
// Status (never crash, never write outputs on failure) and the stale-set
// rewrite is guarded by the written_in/written_out handle check (the
// generation counter is inert in the current create path — see
// NATIVE-VULKAN-STRESS.md), but actual device-loss behaviour is untested.
//
// Usage: native-vulkan-stress-test <path-to-matmul_e4m3.spv>. Clean skip
// when no Vulkan device is present (a recorded environment gap).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <new>
#include <vector>

#include "native-parallel.hpp"
#include "native-vulkan.hpp"

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

// Allocation accounting (the house pattern): CPU-side allocations must
// balance across init/teardown cycles.
std::size_t g_live_allocations = 0;
dlss5_nr::VulkanMatmulTiming g_timing{};

// The independent reference: out[t][n] = sum_k in[t][k] * w[n][k], key
// order ascending, one sequential float accumulator (the shader's order).
void ref_matmul(const float* in, int tokens, int c_in, const float* w,
                int rows, float* out) {
    for (int t = 0; t < tokens; ++t) {
        for (int n = 0; n < rows; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < c_in; ++k) {
                acc += in[static_cast<std::size_t>(t) * c_in + k]
                    * w[static_cast<std::size_t>(n) * c_in + k];
            }
            out[static_cast<std::size_t>(t) * rows + n] = acc;
        }
    }
}

}  // namespace

// Observed with this override set in this TU: -Wmismatched-new-delete
// fires inside the delete definitions (GCC 16, -O2) even though every
// operator new here is paired with an operator delete — the pairing is
// verified by g_live_allocations balancing in the cycle test. The
// suppression covers exactly the definitions below; if it fires elsewhere
// the build fails loudly, which is the desired behaviour.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
void* operator new(std::size_t n) {
    void* p = std::malloc(n == 0 ? 1 : n);
    if (p == nullptr) {
        throw std::bad_alloc();
    }
    ++g_live_allocations;
    return p;
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void* operator new(std::size_t n, std::align_val_t a) {
    const std::size_t al = static_cast<std::size_t>(a);
    const std::size_t rounded =
        ((n + al - 1) / al) * al == 0 ? al : ((n + al - 1) / al) * al;
    void* p = std::aligned_alloc(al, rounded);
    if (p == nullptr) {
        throw std::bad_alloc();
    }
    ++g_live_allocations;
    return p;
}
void* operator new[](std::size_t n, std::align_val_t a) {
    return ::operator new(n, a);
}
void operator delete(void* p) noexcept {
    if (p != nullptr) {
        --g_live_allocations;
    }
    std::free(p);
}
void operator delete[](void* p) noexcept { ::operator delete(p); }
void operator delete(void* p, std::size_t) noexcept { ::operator delete(p); }
void operator delete[](void* p, std::size_t) noexcept { ::operator delete(p); }
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
#pragma GCC diagnostic pop

int main(int argc, char** argv) {
    using namespace dlss5_nr;

    VulkanDevice device;
    bool opened = false;
    CHECK(vulkan_open(device, opened) == Status::kOk);
    if (!opened) {
        std::printf("native-vulkan-stress: SKIP (no compute-capable Vulkan "
                    "device; recorded environment gap)\n");
        return 0;
    }
    if (argc < 2) {
        std::printf("usage: native-vulkan-stress-test <matmul_e4m3.spv>\n");
        vulkan_close(device);
        return 2;
    }
    std::vector<std::uint32_t> spv;
    {
        std::ifstream f(argv[1], std::ios::binary);
        std::uint32_t word = 0;
        while (f.read(reinterpret_cast<char*>(&word), sizeof word)) {
            spv.push_back(word);
        }
    }
    CHECK(!spv.empty());
    std::printf("native-vulkan-stress: device %s\n", device.name);

    // Small fixed workload shared by the cycle and determinism checks.
    constexpr int kTokens = 33;
    constexpr int kIn = 257;
    constexpr int kRows = 41;
    static float weights[kRows * kIn];
    static float input[kTokens * kIn];
    static float gpu_out[kTokens * kRows];
    static float cpu_out[kTokens * kRows];
    for (int i = 0; i < kRows * kIn; ++i) {
        weights[i] = static_cast<float>((i * 37) % 256) / 128.0f - 1.0f;
    }
    for (int i = 0; i < kTokens * kIn; ++i) {
        input[i] = static_cast<float>((i * 19) % 256) / 128.0f - 1.0f;
    }
    ref_matmul(input, kTokens, kIn, weights, kRows, cpu_out);

    // ---- 1. init/teardown cycles ------------------------------------------
    // 50 full create -> run -> destroy cycles: every result bit-identical,
    // the CPU allocation count back to its baseline after each cycle.
    {
        const std::size_t baseline = g_live_allocations;
        for (int cycle = 0; cycle < 50; ++cycle) {
            VulkanMatmul m;
            CHECK(vulkan_matmul_create(device, m, spv.data(), spv.size(),
                                 4096, 1024, 4096)
                  == Status::kOk);
            VulkanWeightCache cache;
            CHECK(cache.init(device) == Status::kOk);
            VulkanWeightCache::Entry* entry = nullptr;
            CHECK(vulkan_weight_cache_ensure(
                      cache, device, m, weights, weights, kRows, kIn,
                      vulkan_weight_hash(weights, sizeof weights), &entry)
                  == Status::kOk);
            VulkanMatmulTiming timing{};
            CHECK(vulkan_matmul_run_resident(device, m, input, kTokens, kIn,
                                             *entry, kRows, gpu_out, timing)
                  == Status::kOk);
            CHECK(std::memcmp(gpu_out, cpu_out, sizeof gpu_out) == 0);
            cache.destroy(device);
            vulkan_matmul_destroy(device, m);
            CHECK(g_live_allocations == baseline);
        }
        std::printf("native-vulkan-stress: 50 init/teardown cycles OK\n");
    }

    // ---- 2. long mixed-shape resident run ---------------------------------
    // 200 resident runs (50 x 4 shapes) with one cache and one matmul:
    // every result bit-identical to the reference.
    {
        struct Shape {
            int tokens;
            int in;
            int rows;
        };
        const Shape shapes[] = {
            {17, 256, 8}, {64, 512, 16}, {9, 768, 384}, {128, 1024, 24}};
        VulkanMatmul m;
        CHECK(vulkan_matmul_create(device, m, spv.data(), spv.size(),
                                 4096, 1024, 4096)
              == Status::kOk);
        VulkanWeightCache cache;
        CHECK(cache.init(device) == Status::kOk);
        for (int rep = 0; rep < 50; ++rep) {
            for (const Shape& s : shapes) {
                std::vector<float> w(static_cast<std::size_t>(s.rows) * s.in);
                std::vector<float> in(static_cast<std::size_t>(s.tokens) * s.in);
                std::vector<float> got(static_cast<std::size_t>(s.tokens)
                                           * s.rows,
                                       __builtin_nanf(""));
                std::vector<float> want(got.size());
                for (std::size_t i = 0; i < w.size(); ++i) {
                    w[i] = static_cast<float>(
                               static_cast<int>((i * 29 + rep) % 256u))
                            / 128.0f
                        - 1.0f;
                }
                for (std::size_t i = 0; i < in.size(); ++i) {
                    in[i] = static_cast<float>(
                                static_cast<int>((i * 13 + rep) % 256u))
                            / 128.0f
                        - 1.0f;
                }
                ref_matmul(in.data(), s.tokens, s.in, w.data(), s.rows,
                           want.data());
                VulkanWeightCache::Entry* entry = nullptr;
                CHECK(vulkan_weight_cache_ensure(
                          cache, device, m, w.data(), w.data(), s.rows, s.in,
                          vulkan_weight_hash(w.data(),
                                             w.size() * sizeof(float)),
                          &entry)
                      == Status::kOk);
                CHECK(vulkan_matmul_run_resident(device, m, in.data(),
                                                 s.tokens, s.in, *entry,
                                                 s.rows, got.data(),
                                                 g_timing)
                      == Status::kOk);
                CHECK(std::memcmp(got.data(), want.data(),
                                  got.size() * sizeof(float))
                      == 0);
            }
        }
        cache.destroy(device);
        vulkan_matmul_destroy(device, m);
        std::printf("native-vulkan-stress: 200 resident runs OK\n");
    }

    // ---- 3. cache content-hash mismatch and capacity behaviour -------------
    {
        VulkanMatmul m;
        CHECK(vulkan_matmul_create(device, m, spv.data(), spv.size(),
                                 4096, 1024, 4096)
              == Status::kOk);
        VulkanWeightCache cache;
        CHECK(cache.init(device) == Status::kOk);
        static float mutable_w[8 * 256];
        static float mod_in[4 * 256];
        static float mod_got[4 * 8];
        static float mod_want[4 * 8];
        for (int i = 0; i < 4 * 256; ++i) {
            mod_in[i] = static_cast<float>((i * 7) % 256) / 128.0f - 1.0f;
        }
        // The same pointer with CHANGED content must re-upload (the hash)
        // and stay correct.
        for (int round = 0; round < 5; ++round) {
            for (int i = 0; i < 8 * 256; ++i) {
                mutable_w[i] = static_cast<float>(
                                   static_cast<int>((i * 11 + round * 37)
                                                    % 256u))
                    / 128.0f
                    - 1.0f;
            }
            ref_matmul(mod_in, 4, 256, mutable_w, 8, mod_want);
            VulkanWeightCache::Entry* entry = nullptr;
            CHECK(vulkan_weight_cache_ensure(
                      cache, device, m, mutable_w, mutable_w, 8, 256,
                      vulkan_weight_hash(mutable_w, sizeof mutable_w), &entry)
                  == Status::kOk);
            CHECK(vulkan_matmul_run_resident(device, m, mod_in, 4, 256,
                                             *entry, 8, mod_got,
                                             g_timing)
                  == Status::kOk);
            CHECK(std::memcmp(mod_got, mod_want, sizeof mod_got) == 0);
        }
        // Capacity behaviour: kCapacity+8 distinct keys — the ensure path
        // must reject past the limit (no silent wrap) and the entries
        // already cached must stay usable and correct.
        static float pool[VulkanWeightCache::kCapacity + 8][8 * 256];
        bool rejected = false;
        for (int k = 0; k < VulkanWeightCache::kCapacity + 8; ++k) {
            for (int i = 0; i < 8 * 256; ++i) {
                pool[k][i] = static_cast<float>(
                                static_cast<int>((i + k) % 256u))
                    / 128.0f
                    - 1.0f;
            }
            VulkanWeightCache::Entry* entry = nullptr;
            const Status st = vulkan_weight_cache_ensure(
                cache, device, m, pool[k], pool[k], 8, 256,
                vulkan_weight_hash(pool[k], sizeof pool[k]), &entry);
            if (st != Status::kOk) {
                // Past capacity: rejection is the documented behaviour.
                rejected = true;
                CHECK(cache.count <= VulkanWeightCache::kCapacity);
                break;
            }
            ref_matmul(mod_in, 4, 256, pool[k], 8, mod_want);
            CHECK(vulkan_matmul_run_resident(device, m, mod_in, 4, 256,
                                             *entry, 8, mod_got,
                                             g_timing)
                  == Status::kOk);
            CHECK(std::memcmp(mod_got, mod_want, sizeof mod_got) == 0);
        }
        CHECK(rejected);
        // A still-cached entry remains correct after the exhaustion attempts.
        ref_matmul(mod_in, 4, 256, mutable_w, 8, mod_want);
        VulkanWeightCache::Entry* entry = nullptr;
        CHECK(vulkan_weight_cache_ensure(
                  cache, device, m, mutable_w, mutable_w, 8, 256,
                  vulkan_weight_hash(mutable_w, sizeof mutable_w), &entry)
              == Status::kOk);
        CHECK(vulkan_matmul_run_resident(device, m, mod_in, 4, 256, *entry, 8,
                                         mod_got, g_timing)
              == Status::kOk);
        CHECK(std::memcmp(mod_got, mod_want, sizeof mod_got) == 0);
        cache.destroy(device);
        vulkan_matmul_destroy(device, m);
        std::printf("native-vulkan-stress: cache hash-mismatch + capacity "
                    "OK\n");
    }

    // ---- 4. failure paths do not corrupt later runs ------------------------
    {
        VulkanMatmul m;
        CHECK(vulkan_matmul_create(device, m, spv.data(), spv.size(),
                                 4096, 1024, 4096)
              == Status::kOk);
        VulkanWeightCache cache;
        CHECK(cache.init(device) == Status::kOk);
        VulkanWeightCache::Entry* entry = nullptr;
        CHECK(vulkan_weight_cache_ensure(
                  cache, device, m, weights, weights, kRows, kIn,
                  vulkan_weight_hash(weights, sizeof weights), &entry)
              == Status::kOk);
        static float out[4 * 41];
        // Rejections first (null in/out, zero tokens, entry-shape mismatch,
        // oversized tokens/rows): each must reject AND leave `out` untouched
        // (the house sentinel contract).
        static float sentinel[4 * 41];
        for (std::size_t i = 0; i < 4 * 41; ++i) {
            sentinel[i] = __builtin_nanf("");
        }
        std::memcpy(out, sentinel, sizeof out);
        CHECK(vulkan_matmul_run_resident(device, m, nullptr, 4, kIn, *entry,
                                         kRows, out, g_timing)
              != Status::kOk);
        CHECK(std::memcmp(out, sentinel, sizeof out) == 0);
        std::memcpy(out, sentinel, sizeof out);
        CHECK(vulkan_matmul_run_resident(device, m, input, 0, kIn, *entry,
                                         kRows, out, g_timing)
              != Status::kOk);
        CHECK(std::memcmp(out, sentinel, sizeof out) == 0);
        std::memcpy(out, sentinel, sizeof out);
        CHECK(vulkan_matmul_run_resident(device, m, input, 4, kIn, *entry,
                                         kRows, nullptr,
                                         g_timing)
              != Status::kOk);
        CHECK(std::memcmp(out, sentinel, sizeof out) == 0);
        std::memcpy(out, sentinel, sizeof out);
        CHECK(vulkan_matmul_run_resident(device, m, input, 4, kIn, *entry,
                                         kRows + 1, out,
                                         g_timing)
              != Status::kOk);
        CHECK(std::memcmp(out, sentinel, sizeof out) == 0);
        std::memcpy(out, sentinel, sizeof out);
        CHECK(vulkan_matmul_run_resident(device, m, input, 4097, kIn, *entry,
                                         kRows, out, g_timing)
              != Status::kOk);
        CHECK(std::memcmp(out, sentinel, sizeof out) == 0);
        std::memcpy(out, sentinel, sizeof out);
        CHECK(vulkan_matmul_run_resident(device, m, input, 4, kIn, *entry,
                                         4097, out, g_timing)
              != Status::kOk);
        CHECK(std::memcmp(out, sentinel, sizeof out) == 0);
        // Then a good run must still be correct.
        static float good_in[4 * kIn];
        static float good_want[4 * 41];
        for (int i = 0; i < 4 * kIn; ++i) {
            good_in[i] = static_cast<float>((i * 5) % 256) / 128.0f - 1.0f;
        }
        ref_matmul(good_in, 4, kIn, weights, kRows, good_want);
        CHECK(vulkan_matmul_run_resident(device, m, good_in, 4, kIn, *entry,
                                         kRows, out,
                                         g_timing)
              == Status::kOk);
        CHECK(std::memcmp(out, good_want, sizeof out) == 0);
        cache.destroy(device);
        vulkan_matmul_destroy(device, m);
        std::printf("native-vulkan-stress: failure paths OK\n");
    }

    // ---- 4b. the content-hash contract (findings 12/13) ------------------
    // The hash must be total (no NaN self-mismatches), distinct for
    // different content INCLUDING sign-bit changes, and distinct when the
    // same pointer carries different (rows, cols) extents (the equal-
    // pointer twin case: same address, different extent -> different hash
    // -> a hit cannot bind the wrong extent).
    {
        float plus_zero = 0.0f;
        float minus_zero = -0.0f;
        CHECK(std::memcmp(&plus_zero, &minus_zero, sizeof plus_zero) != 0);
        CHECK(vulkan_weight_hash(&plus_zero, sizeof plus_zero)
              != vulkan_weight_hash(&minus_zero, sizeof minus_zero));
        const float nan_v = __builtin_nanf("");
        float nan_copy = nan_v;
        CHECK(vulkan_weight_hash(&nan_v, sizeof nan_v)
              == vulkan_weight_hash(&nan_copy, sizeof nan_copy));
        static float nine[9];
        for (int i = 0; i < 9; ++i) {
            nine[i] = 0.25f;
        }
        // Same base pointer, two extents: 8 floats vs 9 floats.
        CHECK(vulkan_weight_hash(nine, 8 * sizeof(float))
              != vulkan_weight_hash(nine, 9 * sizeof(float)));
    }

    // ---- 5. forced failures + the live generation counter -----------------
    // The fault seam injects failures WITHOUT touching the driver: each
    // armed failure must surface as Status with the output sentinel
    // untouched, and the next unarmed run must be correct (the path is
    // usable after a failure). The generation counter must advance across
    // a grow-only rebuild (destroy + create), which is what makes the
    // cache's stale-set check meaningful.
    {
        VulkanMatmul m;
        CHECK(vulkan_matmul_create(device, m, spv.data(), spv.size(), 4096,
                                   1024, 4096)
              == Status::kOk);
        CHECK(m.generation == 1);
        VulkanWeightCache cache;
        CHECK(cache.init(device) == Status::kOk);
        VulkanWeightCache::Entry* entry = nullptr;
        CHECK(vulkan_weight_cache_ensure(
                  cache, device, m, weights, weights, kRows, kIn,
                  vulkan_weight_hash(weights, sizeof weights), &entry)
              == Status::kOk);
        CHECK(entry->written_gen == m.generation);

        static float out[33 * 41];
        static float sentinel[33 * 41];
        for (std::size_t i = 0; i < 33 * 41; ++i) {
            sentinel[i] = __builtin_nanf("");
        }
        // Forced submit failure on the next submit: run rejects, output
        // untouched, and the following unarmed run is correct.
        std::memcpy(out, sentinel, sizeof out);
        g_vulkan_faults.fail_submit_countdown = 0;
        CHECK(vulkan_matmul_run_resident(device, m, input, kTokens, kIn,
                                         *entry, kRows, out, g_timing)
              != Status::kOk);
        CHECK(std::memcmp(out, sentinel, sizeof out) == 0);
        CHECK(g_vulkan_faults.fail_submit_countdown == -1);
        CHECK(vulkan_matmul_run_resident(device, m, input, kTokens, kIn,
                                         *entry, kRows, out, g_timing)
              == Status::kOk);
        CHECK(std::memcmp(out, cpu_out, sizeof out) == 0);

        // Forced map failure (the first map hook, the input phase): the resident run
        // rejects before writing `out`; the path stays usable.
        std::memcpy(out, sentinel, sizeof out);
        g_vulkan_faults.fail_map_countdown = 0;
        CHECK(vulkan_matmul_run_resident(device, m, input, kTokens, kIn,
                                         *entry, kRows, out, g_timing)
              != Status::kOk);
        CHECK(std::memcmp(out, sentinel, sizeof out) == 0);
        CHECK(vulkan_matmul_run_resident(device, m, input, kTokens, kIn,
                                         *entry, kRows, out, g_timing)
              == Status::kOk);
        CHECK(std::memcmp(out, cpu_out, sizeof out) == 0);

        // The countdown semantics, checked deterministically at the seam
        // (one run issues one submit — upload/dispatch/download — so
        // run-level "Nth submit" expectations would be fragile; the
        // run-level checks above use countdown = 0, the run's first
        // submit).
        {
            int countdown = 1;
            CHECK(!vulkan_fault_consume(countdown));
            CHECK(countdown == 0);
            CHECK(vulkan_fault_consume(countdown));
            CHECK(countdown == -1);
            CHECK(!vulkan_fault_consume(countdown));
            int never = -1;
            CHECK(!vulkan_fault_consume(never));
            CHECK(never == -1);
        }

        // Grow-only rebuild: the generation must advance (the cache's
        // stale-set check compares it), and the entry's descriptor must be
        // rewritten against the new matmul state (written_gen matches).
        vulkan_matmul_destroy(device, m);
        CHECK(vulkan_matmul_create(device, m, spv.data(), spv.size(), 8192,
                                   1024, 4096)
              == Status::kOk);
        CHECK(m.generation == 2);
        // The entry's descriptor was written against the OLD buffers: a
        // cache hit must detect the stale generation and rewrite the
        // descriptor against the new matmul state (the stale-set guard).
        CHECK(vulkan_weight_cache_ensure(
                  cache, device, m, weights, weights, kRows, kIn,
                  vulkan_weight_hash(weights, sizeof weights), &entry)
              == Status::kOk);
        CHECK(entry->written_gen == m.generation);
        CHECK(entry->written_in == m.in_dev.buffer);
        CHECK(vulkan_matmul_run_resident(device, m, input, kTokens, kIn,
                                         *entry, kRows, out, g_timing)
              == Status::kOk);
        CHECK(std::memcmp(out, cpu_out, sizeof out) == 0);
        cache.destroy(device);
        vulkan_matmul_destroy(device, m);
        std::printf("native-vulkan-stress: forced failures + generation "
                    "OK\n");
    }

    vulkan_close(device);
    std::printf("native-vulkan-stress: %lld checks passed\n", g_checks);
    return 0;
}
