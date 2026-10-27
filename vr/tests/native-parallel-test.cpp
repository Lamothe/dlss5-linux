// SPDX-License-Identifier: MIT
// Copyright (c) 2026 the dlss5-linux contributors
//
// Known-answer and parity tests for the deterministic parallel helpers and
// the threaded projection kernels: parallel_for coverage/chunking/order,
// matmul_e4m3 and swin_linear bit-parity across thread counts (the CPU path
// is the reference: every element keeps its key-ordered accumulation), and
// zero-allocation accounting on the serial reference path. No GPU, headset,
// proprietary payload or Python required.
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

#include "native-parallel.hpp"
#include "native-swin-linear.hpp"
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
    using namespace dlss5_nr;

    // ---- parallel_for semantics ------------------------------------------
    {
        // Coverage: every index runs exactly once at every thread count.
        for (int threads : {1, 2, 3, 4, 8, 16}) {
            set_parallel_threads(threads);
            static int hits[100];
            for (int i = 0; i < 100; ++i) {
                hits[i] = 0;
            }
            parallel_for(0, 100, [&](int i) { hits[i] += 1; });
            for (int i = 0; i < 100; ++i) {
                CHECK(hits[i] == 1);
            }
            // Empty and single-element ranges are well-defined.
            static int touched;
            touched = 0;
            parallel_for(5, 5, [&](int) { touched += 1; });
            CHECK(touched == 0);
            parallel_for(7, 8, [&](int i) { touched += (i == 7 ? 1 : 0); });
            CHECK(touched == 1);
            // Non-zero begin offsets cover [begin, end).
            for (int i = 0; i < 100; ++i) {
                hits[i] = 0;
            }
            parallel_for(25, 75, [&](int i) { hits[i] += 1; });
            for (int i = 0; i < 100; ++i) {
                CHECK(hits[i] == (i >= 25 && i < 75 ? 1 : 0));
            }
        }
        // More threads than elements: every element still runs once.
        set_parallel_threads(16);
        static int hits[3];
        for (int i = 0; i < 3; ++i) {
            hits[i] = 0;
        }
        parallel_for(0, 3, [&](int i) { hits[i] += 1; });
        CHECK(hits[0] == 1 && hits[1] == 1 && hits[2] == 1);
    }

    // ---- matmul_e4m3 parity across thread counts -------------------------
    {
        // A hand-known case first: tokens=2, c_in=3, rows=2.
        // Codes (the tree's e4m3 bias-7 decode): row 0 = [8, 20, 28] ->
        // [0.015625, 0.046875, 0.09375]; row 1 = [28, 20, 8] -> the reverse.
        // Token 0 = [1, 2, 4]: out[0][0] = 1*0.015625 + 2*0.046875 +
        // 4*0.09375 = 0.015625 + 0.09375 + 0.375 = 0.484375;
        // out[0][1] = 1*0.09375 + 2*0.046875 + 4*0.015625 = 0.25.
        // Token 1 = [4, 2, 1]: out[1][0] = 0.25; out[1][1] = 0.484375.
        static std::uint8_t codes[2 * 3] = {8, 20, 28, 28, 20, 8};
        static E4m3Matrix w;
        w.bits = codes;
        w.rows = 2;
        w.cols = 3;
        static float in[2 * 3] = {1.0f, 2.0f, 4.0f, 4.0f, 2.0f, 1.0f};
        static float out[2 * 2];
        CHECK(matmul_e4m3(in, 2, 3, w, out) == Status::kOk);
        CHECK(out[0] == 0.484375f);
        CHECK(out[1] == 0.25f);
        CHECK(out[2] == 0.25f);
        CHECK(out[3] == 0.484375f);

        // A larger deterministic case must be bit-identical at 1/2/3/4/8/16
        // threads: the per-element accumulation order is thread-invariant.
        static std::uint8_t big_codes[37 * 53];
        for (int i = 0; i < 37 * 53; ++i) {
            big_codes[i] = static_cast<std::uint8_t>(12 + (i * 7) % 17);
        }
        static E4m3Matrix big_w;
        big_w.bits = big_codes;
        big_w.rows = 37;
        big_w.cols = 53;
        static float big_in[41 * 53];
        // Multiples of 0.125 in [-2, 1.75]: exactly on the e4m3 grid (its
        // 3-bit mantissa steps are 0.125 in [1,2), 0.0625 in [0.5,1), ...).
        for (int i = 0; i < 41 * 53; ++i) {
            big_in[i] = static_cast<float>((i * 13) % 31) / 8.0f - 2.0f;
        }
        static float ref[41 * 37];
        set_parallel_threads(1);
        CHECK(matmul_e4m3(big_in, 41, 53, big_w, ref) == Status::kOk);
        for (int threads : {2, 3, 4, 8, 16}) {
            set_parallel_threads(threads);
            static float other[41 * 37];
            // Poison (NaN) so a partially-written output cannot alias a
            // legitimate zero and pass the memcmp silently.
            for (int i = 0; i < 41 * 37; ++i) {
                other[i] = __builtin_nanf("");
            }
            CHECK(matmul_e4m3(big_in, 41, 53, big_w, other) == Status::kOk);
            CHECK(std::memcmp(ref, other, sizeof(ref)) == 0);
        }
    }

    // ---- swin_linear parity across thread counts -------------------------
    {
        // A hand-known case: x [2, 3], weight [2, 3].
        // x row 0 = [1, 2, 4]; weight row 0 = [0.0078125, 0.046875,
        // 0.09375] -> 1*0.0078125 + 2*0.046875 + 4*0.09375 = 0.4765625;
        // weight row 1 = the reverse -> 0.21875.
        static float x_data[2 * 3] = {1.0f, 2.0f, 4.0f, 4.0f, 2.0f, 1.0f};
        static float w_data[2 * 3] = {0.0078125f, 0.046875f, 0.09375f,
                                      0.09375f, 0.046875f, 0.0078125f};
        static float out_data[2 * 2];
        const MatrixView x{x_data, 2, 3, 3};
        const MatrixView w{w_data, 2, 3, 3};
        MutableMatrixView out{out_data, 2, 2, 2};
        CHECK(swin_linear(x, w, &out, 4) == Status::kOk);
        CHECK(out_data[0] == 0.4765625f);
        CHECK(out_data[1] == 0.21875f);
        CHECK(out_data[2] == 0.21875f);
        CHECK(out_data[3] == 0.4765625f);

        // Bit-parity at 1/2/3/4/8/16 threads on a larger deterministic case.
        static float big_x[29 * 43];
        static float big_w2[17 * 43];
        // e4m3-grid-exact operand values (validate_linear_operand's
        // contract): multiples of 0.125 within [-2, 2).
        for (int i = 0; i < 29 * 43; ++i) {
            big_x[i] = static_cast<float>((i * 11) % 31) / 8.0f - 2.0f;
        }
        for (int i = 0; i < 17 * 43; ++i) {
            big_w2[i] = static_cast<float>((i * 17) % 23) / 8.0f - 1.0f;
        }
        const MatrixView bx{big_x, 29, 43, 43};
        const MatrixView bw{big_w2, 17, 43, 43};
        static float ref2[29 * 17];
        static float other2[29 * 17];
        MutableMatrixView ref_out{ref2, 29, 17, 17};
        set_parallel_threads(1);
        CHECK(swin_linear(bx, bw, &ref_out, 29 * 17) == Status::kOk);
        for (int threads : {2, 3, 4, 8, 16}) {
            set_parallel_threads(threads);
            for (int i = 0; i < 29 * 17; ++i) {
                other2[i] = __builtin_nanf("");
            }
            MutableMatrixView other_out{other2, 29, 17, 17};
            CHECK(swin_linear(bx, bw, &other_out, 29 * 17) == Status::kOk);
            CHECK(std::memcmp(ref2, other2, sizeof(ref2)) == 0);
        }
    }

    // ---- parity for the remaining parallel sites -------------------------
    {
        // vit_ffn_expand and vit_residual_linear (token loops), vit_qkv (the
        // raw projection loop) and split_ffwd (the long long token loop):
        // bit-identical across thread counts. Operands stay on the e4m3 grid
        // where the kernels require it.
        static E4m3Matrix qkv_w[3];
        static std::uint8_t qkv_codes[3][32 * 32];
        for (int m = 0; m < 3; ++m) {
            for (int i = 0; i < 32 * 32; ++i) {
                qkv_codes[m][i] =
                    static_cast<std::uint8_t>(12 + ((i + m * 5) * 7) % 17);
            }
            qkv_w[m].bits = qkv_codes[m];
            qkv_w[m].rows = 32;
            qkv_w[m].cols = 32;
        }
        static float in[13 * 32];
        static float skip[13 * 32];
        static float gain[32];
        static float temp[1];
        for (int i = 0; i < 13 * 32; ++i) {
            in[i] = static_cast<float>((i * 7) % 31) / 8.0f - 2.0f;
            skip[i] = static_cast<float>((i * 3) % 15) / 8.0f - 1.0f;
        }
        for (int i = 0; i < 32; ++i) {
            gain[i] = static_cast<float>((i % 15)) / 8.0f - 1.0f;
        }
        temp[0] = 0.25f;
        static float ref_ffn[13 * 32];
        static float ref_res[13 * 32];
        static float ref_qkv[13 * 3 * 32];
        set_parallel_threads(1);
        CHECK(vit_ffn_expand(in, 13, 32, 32, qkv_w[0], ref_ffn) == Status::kOk);
        CHECK(vit_residual_linear(in, skip, 13, 32, 32, qkv_w[1], gain,
                                  ref_res) == Status::kOk);
        CHECK(vit_qkv(in, 13, 32, qkv_w[0], qkv_w[1], qkv_w[2], temp,
                      ref_qkv) == Status::kOk);
        // split_ffwd needs its full SplitFfwdWeight record set; its long
        // long token path is covered directly through parallel_for below.
        static float other_ffn[13 * 32];
        static float other_res[13 * 32];
        static float other_qkv[13 * 3 * 32];
        for (int threads : {2, 3, 4, 8, 16}) {
            set_parallel_threads(threads);
            for (int i = 0; i < 13 * 32; ++i) {
                other_ffn[i] = __builtin_nanf("");
                other_res[i] = __builtin_nanf("");
            }
            for (int i = 0; i < 13 * 3 * 32; ++i) {
                other_qkv[i] = __builtin_nanf("");
            }
            CHECK(vit_ffn_expand(in, 13, 32, 32, qkv_w[0], other_ffn)
                  == Status::kOk);
            CHECK(vit_residual_linear(in, skip, 13, 32, 32, qkv_w[1], gain,
                                      other_res) == Status::kOk);
            CHECK(vit_qkv(in, 13, 32, qkv_w[0], qkv_w[1], qkv_w[2], temp,
                          other_qkv) == Status::kOk);
            CHECK(std::memcmp(ref_ffn, other_ffn, sizeof(ref_ffn)) == 0);
            CHECK(std::memcmp(ref_res, other_res, sizeof(ref_res)) == 0);
            CHECK(std::memcmp(ref_qkv, other_qkv, sizeof(ref_qkv)) == 0);
        }

        // The long long index path (split_ffwd's shape): coverage and
        // offsets at several thread counts.
        for (int threads : {1, 2, 4, 16}) {
            set_parallel_threads(threads);
            static int ll_hits[300];
            for (int i = 0; i < 300; ++i) {
                ll_hits[i] = 0;
            }
            parallel_for(0LL, 300LL, [&](long long i) {
                ll_hits[i] += 1;
            });
            for (int i = 0; i < 300; ++i) {
                CHECK(ll_hits[i] == 1);
            }
        }
    }

    // ---- the serial reference path allocates nothing ---------------------
    {
        set_parallel_threads(1);
        static std::uint8_t codes[2 * 3] = {8, 20, 28, 28, 20, 8};
        static E4m3Matrix w;
        w.bits = codes;
        w.rows = 2;
        w.cols = 3;
        static float in[2 * 3] = {1.0f, 2.0f, 4.0f, 4.0f, 2.0f, 1.0f};
        static float out[2 * 2];
        const unsigned long long before_alloc = g_allocations.load();
        for (int i = 0; i < 64; ++i) {
            (void)matmul_e4m3(in, 2, 3, w, out);
        }
        CHECK(g_allocations.load() == before_alloc);
    }

    set_parallel_threads(1);
    std::printf("native-parallel: %lld checks passed\n", g_checks);
    return 0;
}
