// native-weights-source-test.cpp — the source-tensor-dictionary -> NWF1
// converter tests.
//
// Hand-derived known answers for the two packing transforms (the split
// feed-forward grouped scatter and the attn_pos_bias fragment order), full
// permutation coverage of the scatter (every one of the 524,288 codes
// claimed exactly once — not spot checks), the fragment unpack pinned in
// BOTH consumers (the swin exponent-ramp form and the split family's
// no-ramp form), the whole-network round-trip (synthetic dictionary ->
// converter -> NWF1 -> evaluator, artefact byte-identical to the direct
// synthetic path), mutation-verified discriminators (a seeded wrong
// permutation, a wrong fragment order and a dropped group must each fail
// exactly at its designed check), and the named negative matrix (wrong
// packed shapes, truncated fragments, duplicate identities, count/shape
// mismatch, non-finite values, unknown names).
//
// The oracle restatements are transcriptions of the pinned MIT reference
// (mauri870/dlssnr-pytorch commit 3beef7935a8ef976b47ab78b906e75600c52d2de,
// dlssnr/layers/split_swin.py) written in direct arithmetic; the hand
// literals are the independent pins. Synthetic fixtures only: no weights,
// no proprietary payloads, no vendor correspondence is claimed.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "../native-demo-support.hpp"
#include "../native-graph-plan.hpp"
#include "../native-weights-source.hpp"

namespace {

int g_checks = 0;
int g_failures = 0;

bool CHECK(bool ok, const char* what, int line) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("FAIL (line %d): %s\n", line, what);
    }
    return ok;
}

#define CHECK_(cond) CHECK((cond), #cond, __LINE__)

bool contains(const dlss5_nr::weightsfile::WeightsFileError& e,
              const char* needle) {
    return e.str().find(needle) != std::string::npos;
}

using dlss5_nr::FeatureBatchView;
using dlss5_nr::LayerSpec;
using dlss5_nr::LayerValue;
using dlss5_nr::NetworkRun;
using dlss5_nr::Status;
using dlss5_nr::f16_bits;
using dlss5_nr::f16_from_bits;
namespace weightsfile = dlss5_nr::weightsfile;
namespace weightssource = dlss5_nr::weightssource;

// ---- independent oracle restatements --------------------------------------
// Direct arithmetic transcriptions of the pinned reference's expressions
// (`_slice_sources` / `_grouped_ffwd_matrices` / `unpack_bias`), written
// without the implementation's bit() helper. The hand literals below are
// the independent pins; these give the full-space comparisons.

// The pack map inverted position-wise from the reference's forward
// definitions: given a generic [512, 512] element (row, col), the byte
// offset it occupies in the packed region. row = row_group*16 + half*8 +
// lane/4 and col = col_group*32 + 8*(byte/2) + 2*(lane%4) + byte%2 invert
// uniquely, so slice = row_group + 32*col_group and offset = slice*512 +
// lane*16 + half*8 + byte.
int oracle_pack_offset(int row, int col) {
    const int row_group = row / 16;
    const int half = (row % 16) / 8;
    const int r4 = (row % 16) % 8;
    const int col_group = col / 32;
    const int c = col % 32;
    const int byte_low = c % 2;      // byte % 2
    const int m = c / 2;             // 4*(byte/2) + (lane%4)
    const int byte_high = m / 4;     // byte / 2
    const int lane_low = m % 4;      // lane % 4
    const int lane = 4 * r4 + lane_low;
    const int byte = 2 * byte_high + byte_low;
    return (row_group + 32 * col_group) * 512 + lane * 16 + half * 8 + byte;
}

int oracle_expand_in_offset(int g, int j, int k) {
    return ((k >> 0) & 1) * 1 + ((k >> 1) & 1) * 16 + ((k >> 2) & 1) * 32
        + ((k >> 3) & 1) * 2 + ((k >> 4) & 1) * 4 + ((k >> 5) & 1) * 16384
        + ((k >> 6) & 1) * 32768 + ((k >> 7) & 1) * 65536
        + ((k >> 8) & 1) * 131072
        + ((j >> 0) & 1) * 8 + ((j >> 1) & 1) * 64 + ((j >> 2) & 1) * 128
        + ((j >> 3) & 1) * 256 + ((j >> 4) & 1) * 512 + ((j >> 5) & 1) * 1024
        + ((g >> 0) & 1) * 2048 + ((g >> 1) & 1) * 4096 + ((g >> 2) & 1) * 8192;
}

int oracle_expand_hidden_offset(int g, int j, int r) {
    return 262144 + g * 16384
        + ((r >> 0) & 1) * 2 + ((r >> 1) & 1) * 1 + ((r >> 2) & 1) * 16
        + ((r >> 3) & 1) * 32 + ((r >> 4) & 1) * 4 + ((r >> 5) & 1) * 8192
        + ((j >> 0) & 1) * 64 + ((j >> 1) & 1) * 8 + ((j >> 2) & 1) * 512
        + ((j >> 3) & 1) * 128 + ((j >> 4) & 1) * 256 + ((j >> 5) & 1) * 1024
        + ((j >> 6) & 1) * 2048 + ((j >> 7) & 1) * 4096;
}

int oracle_contract_offset(int g, int n, int h) {
    return 393216 + g * 16384
        + ((h >> 0) & 1) * 1 + ((h >> 1) & 1) * 2 + ((h >> 2) & 1) * 4
        + ((h >> 3) & 1) * 16 + ((h >> 4) & 1) * 32 + ((h >> 5) & 1) * 2048
        + ((h >> 6) & 1) * 4096 + ((h >> 7) & 1) * 8192
        + ((n >> 0) & 1) * 64 + ((n >> 1) & 1) * 128 + ((n >> 2) & 1) * 256
        + ((n >> 3) & 1) * 8 + ((n >> 4) & 1) * 512 + ((n >> 5) & 1) * 1024;
}

int oracle_fragment_index(int i, int j) {
    const int lane = 4 * ((i % 16) % 8) + ((j % 16) % 8) / 2;
    const int slot = (j % 2) + (((i % 16) / 8) * 2) + (((j % 16) / 8) * 4);
    return (4 * (i / 16) + (j / 16)) * 256 + lane * 8 + slot;
}

// ---- hand-derived anchors -------------------------------------------------
// Every literal below was derived by hand from the reference's definitions
// (bit tracing recorded in NATIVE-WEIGHTS-SOURCE.md).

struct Anchor3 {
    int a, b, c, want;
};

// Pack anchors: generic [row, col] -> packed byte offset.
const Anchor3 kPackAnchors[] = {
    {0, 0, 0, 0}, {0, 1, 0, 1}, {0, 8, 0, 2}, {0, 2, 0, 16},
    {8, 0, 0, 8}, {16, 0, 0, 512}, {0, 32, 0, 16384}, {1, 3, 0, 81},
};

// expand_in (g, j, k) -> record offset.
const Anchor3 kExpandInAnchors[] = {
    {0, 0, 0, 0}, {0, 0, 1, 1}, {0, 0, 2, 16}, {0, 0, 4, 32},
    {0, 0, 8, 2}, {0, 1, 0, 8}, {1, 0, 0, 2048}, {0, 0, 16, 4},
    {0, 0, 32, 16384}, {0, 32, 0, 1024}, {4, 0, 0, 8192},
};

// expand_hidden (g, j, r) -> record offset.
const Anchor3 kExpandHiddenAnchors[] = {
    {0, 0, 0, 262144}, {0, 0, 1, 262146}, {0, 0, 2, 262145},
    {0, 0, 4, 262160}, {0, 1, 0, 262208}, {1, 0, 0, 278528},
};

// contract (g, n, h) -> record offset.
const Anchor3 kContractAnchors[] = {
    {0, 0, 0, 393216}, {0, 0, 1, 393217}, {0, 0, 2, 393218},
    {0, 0, 4, 393220}, {0, 0, 8, 393232}, {0, 1, 0, 393280},
    {1, 0, 0, 409600},
};

// Fragment anchors: (query i, key j) -> accumulator-fragment index.
const Anchor3 kFragmentAnchors[] = {
    {0, 0, 0, 0}, {0, 1, 0, 1}, {0, 2, 0, 8}, {1, 0, 0, 32},
    {2, 3, 0, 73}, {5, 7, 0, 185}, {8, 0, 0, 2}, {0, 8, 0, 4},
    {15, 15, 0, 255}, {16, 0, 0, 1024}, {32, 32, 0, 2560},
};

// ---- the check battery (parameterised so the mutants run through it) -------

constexpr int kUnclaimed = -1;

using OffsetFn = int (*)(int, int, int);
using FragmentFn = int (*)(int, int);

struct ScatterMap {
    OffsetFn expand_in;
    OffsetFn expand_hidden;
    OffsetFn contract;
};

const ScatterMap kOracleMap = {oracle_expand_in_offset,
                               oracle_expand_hidden_offset,
                               oracle_contract_offset};

// C1: the hand anchors hold.
bool scatter_anchors(const ScatterMap& m) {
    bool ok = true;
    for (const Anchor3& a : kExpandInAnchors) {
        ok = ok && m.expand_in(a.a, a.b, a.c) == a.want;
    }
    for (const Anchor3& a : kExpandHiddenAnchors) {
        ok = ok && m.expand_hidden(a.a, a.b, a.c) == a.want;
    }
    for (const Anchor3& a : kContractAnchors) {
        ok = ok && m.contract(a.a, a.b, a.c) == a.want;
    }
    return ok;
}

// C2: every CLAIMED entry agrees with the oracle (unclaimed entries are the
// coverage check's class, skipped here).
bool scatter_fidelity(const ScatterMap& m) {
    for (int g = 0; g < 8; ++g) {
        for (int j = 0; j < 64; ++j) {
            for (int k = 0; k < 512; ++k) {
                const int got = m.expand_in(g, j, k);
                if (got != kUnclaimed
                    && got != oracle_expand_in_offset(g, j, k)) {
                    return false;
                }
            }
        }
        for (int j = 0; j < 256; ++j) {
            for (int r = 0; r < 64; ++r) {
                const int got = m.expand_hidden(g, j, r);
                if (got != kUnclaimed
                    && got != oracle_expand_hidden_offset(g, j, r)) {
                    return false;
                }
            }
        }
        for (int n = 0; n < 64; ++n) {
            for (int h = 0; h < 256; ++h) {
                const int got = m.contract(g, n, h);
                if (got != kUnclaimed
                    && got != oracle_contract_offset(g, n, h)) {
                    return false;
                }
            }
        }
    }
    return true;
}

// C3: the claimed entries partition [0, 524288) exactly — the full
// permutation coverage (every packed byte consumed exactly once).
bool scatter_coverage(const ScatterMap& m) {
    std::vector<int> seen(2 * 512 * 512, 0);
    auto claim = [&](int off) {
        if (off == kUnclaimed) {
            return;
        }
        if (off < 0 || off >= 2 * 512 * 512) {
            seen[0] = 99;  // out of range: fail below
            return;
        }
        ++seen[static_cast<std::size_t>(off)];
    };
    for (int g = 0; g < 8; ++g) {
        for (int j = 0; j < 64; ++j) {
            for (int k = 0; k < 512; ++k) {
                claim(m.expand_in(g, j, k));
            }
        }
        for (int j = 0; j < 256; ++j) {
            for (int r = 0; r < 64; ++r) {
                claim(m.expand_hidden(g, j, r));
            }
        }
        for (int n = 0; n < 64; ++n) {
            for (int h = 0; h < 256; ++h) {
                claim(m.contract(g, n, h));
            }
        }
    }
    for (int v : seen) {
        if (v != 1) {
            return false;
        }
    }
    return true;
}

// The fragment battery.
bool fragment_anchors(FragmentFn f) {
    bool ok = true;
    for (const Anchor3& a : kFragmentAnchors) {
        ok = ok && f(a.a, a.b) == a.want;
    }
    return ok;
}

bool fragment_fidelity(FragmentFn f) {
    for (int i = 0; i < 64; ++i) {
        for (int j = 0; j < 64; ++j) {
            const int got = f(i, j);
            if (got != kUnclaimed && got != oracle_fragment_index(i, j)) {
                return false;
            }
        }
    }
    return true;
}

bool fragment_coverage(FragmentFn f) {
    std::vector<int> seen(4096, 0);
    for (int i = 0; i < 64; ++i) {
        for (int j = 0; j < 64; ++j) {
            const int idx = f(i, j);
            if (idx < 0 || idx >= 4096) {
                return false;
            }
            ++seen[static_cast<std::size_t>(idx)];
        }
    }
    for (int v : seen) {
        if (v != 1) {
            return false;
        }
    }
    return true;
}

// ---- the scatter known-answer + coverage tests ----------------------------

void check_scatter_maps() {
    // The implementation's maps: the generic pack (`slice_sources`) and the
    // three grouped offsets, pinned against the hand anchors.
    std::vector<int> pack;
    dlss5_nr::weightssource::scatter_detail::slice_sources(pack);
    CHECK_(pack.size() == 262144u);
    for (const Anchor3& a : kPackAnchors) {
        CHECK_(pack[static_cast<std::size_t>(a.a) * 512 + a.b] == a.want);
    }
    // Full-space pack fidelity against the independent restatement and the
    // bijection onto [0, 262144).
    std::vector<int> pack_seen(262144, 0);
    bool pack_fidelity = true;
    for (int row = 0; row < 512; ++row) {
        for (int col = 0; col < 512; ++col) {
            const int got = pack[static_cast<std::size_t>(row) * 512 + col];
            if (got != oracle_pack_offset(row, col)) {
                pack_fidelity = false;
            }
            if (got >= 0 && got < 262144) {
                ++pack_seen[static_cast<std::size_t>(got)];
            }
        }
    }
    CHECK_(pack_fidelity);
    bool pack_bijection = true;
    for (int v : pack_seen) {
        if (v != 1) {
            pack_bijection = false;
        }
    }
    CHECK_(pack_bijection);

    const ScatterMap impl = {dlss5_nr::weightssource::ffwd_expand_in_offset,
                             dlss5_nr::weightssource::ffwd_expand_hidden_offset,
                             dlss5_nr::weightssource::ffwd_contract_offset};
    CHECK_(scatter_anchors(impl));
    CHECK_(scatter_fidelity(impl));
    CHECK_(scatter_coverage(impl));
    // The oracle itself satisfies the battery (the battery is not vacuous).
    CHECK_(scatter_anchors(kOracleMap));
    CHECK_(scatter_fidelity(kOracleMap));
    CHECK_(scatter_coverage(kOracleMap));
}

// Deterministic byte patterns for the value-level scatter (distinct per
// half; the map-level checks above are the exhaustive pin — bytes carry
// only 256 values, so value checks alone have pigeonhole blind spots).
std::uint8_t pattern_a(int i, int j) {
    return static_cast<std::uint8_t>((i * 131 + j * 17 + (i >> 3) * 7) & 0xFF);
}

std::uint8_t pattern_b(int i, int j) {
    return static_cast<std::uint8_t>((i * 29 + j * 211 + (j >> 2) * 13 + 91)
                                     & 0xFF);
}

void check_scatter_values() {
    // Hand-derived end-to-end value anchors (pack -> record -> unpack, all
    // traced by hand): with record[pack(i,j)] = ffwd_a[i*512+j] and
    // record[262144 + pack(i,j)] = ffwd_b[i*512+j]:
    //   expand_in[0][0]   = record[0]      = ffwd_a[0*512+0]  = ffwd_a[0]
    //   expand_in[0][1]   = record[1]      = ffwd_a[0*512+1]  = ffwd_a[1]
    //   expand_in[0][64]  = record[32768]  = ffwd_a[0*512+64] = ffwd_a[64]
    //   expand_in[1][0]   = record[8]      = ffwd_a[8*512+0]  = ffwd_a[4096]
    //   expand_hidden[0][0] = record[262144]  = ffwd_b[0]
    //   expand_hidden[0][1] = record[262146]  = ffwd_b[8]
    //   contract[0][1]    = record[393217] = ffwd_b[0*512+257] = ffwd_b[257]
    std::vector<std::uint8_t> a(262144), b(262144);
    for (int i = 0; i < 512; ++i) {
        for (int j = 0; j < 512; ++j) {
            a[static_cast<std::size_t>(i) * 512 + j] = pattern_a(i, j);
            b[static_cast<std::size_t>(i) * 512 + j] = pattern_b(i, j);
        }
    }
    std::vector<std::uint8_t> expand_in(512 * 512), expand_hidden(2048 * 64),
        contract(512 * 256);
    dlss5_nr::weightssource::scatter_ffwd(a.data(), b.data(), expand_in.data(),
                                          expand_hidden.data(),
                                          contract.data());
    CHECK_(expand_in[0] == a[0]);
    CHECK_(expand_in[1] == a[1]);
    CHECK_(expand_in[64] == a[64]);
    CHECK_(expand_in[512] == a[4096]);
    CHECK_(expand_hidden[0] == b[0]);
    CHECK_(expand_hidden[1] == b[8]);
    CHECK_(contract[1] == b[257]);

    // Full-space value comparison against the oracle maps: every one of the
    // 524,288 outputs checked (not spot checks). The expected record bytes
    // come from the ORACLE pack map inverted position-wise.
    std::vector<int> inv(262144, -1);
    for (int row = 0; row < 512; ++row) {
        for (int col = 0; col < 512; ++col) {
            inv[static_cast<std::size_t>(oracle_pack_offset(row, col))] =
                row * 512 + col;
        }
    }
    std::vector<std::uint8_t> record(2 * 512 * 512);
    for (int p = 0; p < 262144; ++p) {
        const int rc = inv[static_cast<std::size_t>(p)];
        record[static_cast<std::size_t>(p)] =
            a[static_cast<std::size_t>(rc)];
        record[262144 + static_cast<std::size_t>(p)] =
            b[static_cast<std::size_t>(rc)];
    }
    std::size_t mismatches = 0;
    for (int g = 0; g < 8; ++g) {
        for (int j = 0; j < 64; ++j) {
            for (int k = 0; k < 512; ++k) {
                const int off = oracle_expand_in_offset(g, j, k);
                if (expand_in[static_cast<std::size_t>(g * 64 + j) * 512 + k]
                    != record[static_cast<std::size_t>(off)]) {
                    ++mismatches;
                }
            }
        }
        for (int j = 0; j < 256; ++j) {
            for (int r = 0; r < 64; ++r) {
                const int off = oracle_expand_hidden_offset(g, j, r);
                if (expand_hidden[static_cast<std::size_t>(g * 256 + j) * 64 + r]
                    != record[static_cast<std::size_t>(off)]) {
                    ++mismatches;
                }
            }
        }
        for (int n = 0; n < 64; ++n) {
            for (int h = 0; h < 256; ++h) {
                const int off = oracle_contract_offset(g, n, h);
                if (contract[static_cast<std::size_t>(g * 64 + n) * 256 + h]
                    != record[static_cast<std::size_t>(off)]) {
                    ++mismatches;
                }
            }
        }
    }
    CHECK_(mismatches == 0);
}

// ---- the fragment unpack: map + both consumers ----------------------------

void check_fragment_map() {
    // The converter's exported fragment-order map: hand anchors, full-space
    // fidelity against the restatement, and the [0, 4096) bijection.
    for (const Anchor3& a : kFragmentAnchors) {
        CHECK_(dlss5_nr::weightssource::attn_pos_bias_fragment_index(a.a, a.b)
               == a.want);
    }
    bool fidelity = true;
    for (int i = 0; i < 64; ++i) {
        for (int j = 0; j < 64; ++j) {
            if (dlss5_nr::weightssource::attn_pos_bias_fragment_index(i, j)
                != oracle_fragment_index(i, j)) {
                fidelity = false;
            }
        }
    }
    CHECK_(fidelity);
    CHECK_(fragment_coverage(
        dlss5_nr::weightssource::attn_pos_bias_fragment_index));
    // The converter's map is the evaluator's downstream map exactly
    // (swin_deswizzle_index): one fragment order, both consumers.
    bool same_map = true;
    for (int i = 0; i < 64; ++i) {
        for (int j = 0; j < 64; ++j) {
            if (dlss5_nr::weightssource::attn_pos_bias_fragment_index(i, j)
                != dlss5_nr::swin_deswizzle_index(i, j)) {
                same_map = false;
            }
        }
    }
    CHECK_(same_map);
}

// The converter's faithful transfer through BOTH consumer forms. The raw
// fragment-order table (one head, 4096 binary16 entries) is served by the
// source provider and consumed by the evaluator's no-ramp deswizzle (the
// split family) and exponent-ramp form (the swin rows).
void check_fragment_forms() {
    using dlss5_nr::MatrixView;
    using dlss5_nr::MutableMatrixView;
    // raw[idx] = (idx % 5) * 0.25 - 0.5: five binary16-exact values.
    std::vector<std::uint8_t> frag(4096 * 2);
    for (int idx = 0; idx < 4096; ++idx) {
        const float v = static_cast<float>(idx % 5) * 0.25f - 0.5f;
        const std::uint16_t bits = f16_bits(v);
        std::memcpy(frag.data() + 2 * idx, &bits, 2);
    }
    weightssource::SourceStore store;
    store.set(3, 7, "attn_pos_bias", frag);
    const float* raw = weightssource::source_f16(&store, 3, 7,
                                                 "attn_pos_bias", 4096);
    CHECK_(raw != nullptr);
    CHECK_(store.error.ok());
    if (raw == nullptr) {
        return;
    }
    // The transfer is faithful: element-wise f16 decode, no reordering.
    bool faithful = true;
    for (int idx = 0; idx < 4096; ++idx) {
        const float want = static_cast<float>(idx % 5) * 0.25f - 0.5f;
        if (raw[idx] != want) {
            faithful = false;
        }
    }
    CHECK_(faithful);

    const MatrixView raw_view{raw, 1, 4096, 4096};
    std::vector<float> out(4096, -99.0f);
    MutableMatrixView out_view{out.data(), 1, 4096, 4096};

    // Split family's no-ramp form: out[i*64+j] = raw[fragment(i, j)].
    CHECK_(dlss5_nr::swin_deswizzle_bias(raw_view, &out_view, out.size())
           == Status::kOk);
    // Hand-derived values: out[i*64+j] = raw[fragment(i, j)] =
    // (fragment % 5) * 0.25 - 0.5.
    CHECK_(out[2] == 0.25f);     // (0, 2) -> frag 8  -> 3*0.25-0.5
    CHECK_(out[64] == 0.0f);     // (1, 0) -> frag 32 -> 2*0.25-0.5
    CHECK_(out[975] == -0.5f);   // (15, 15) -> frag 255 -> 0*0.25-0.5
    CHECK_(out[1024] == 0.5f);   // (16, 0) -> frag 1024 -> 4*0.25-0.5
    CHECK_(out[131] == 0.25f);   // (2, 3) -> frag 73 -> 3*0.25-0.5

    // Swin ramp form: out = swin_fma(raw[fragment], 23/512, 1332/1024),
    // one float32 rounding from the double intermediate; every case below
    // is dyadic and exact (hand-derived).
    std::fill(out.begin(), out.end(), -99.0f);
    CHECK_(dlss5_nr::swin_exp_bias(raw_view, &out_view, out.size())
           == Status::kOk);
    CHECK_(out[2] == 1.31201171875f);     // raw 0.25
    CHECK_(out[64] == 1.30078125f);       // raw 0.0
    CHECK_(out[975] == 1.2783203125f);    // raw -0.5
    CHECK_(out[1024] == 1.3232421875f);   // raw 0.5
    CHECK_(out[131] == 1.31201171875f);   // raw 0.25
}

// ---- mutation-verified discriminators -------------------------------------
// Three seeded defects, each run through the SAME battery the implementation
// passes. Each must fail EXACTLY its designed check and no other:
//   M1 (seeded wrong permutation) -> scatter fidelity only;
//   M2 (wrong fragment order)     -> fragment fidelity only;
//   M3 (dropped group)            -> scatter coverage only.

// M1: two expand_in entries transposed (a wrong permutation that stays a
// bijection; both positions are off the hand anchors).
int m1_expand_in(int g, int j, int k) {
    if (g == 2 && j == 9 && k == 333) {
        return oracle_expand_in_offset(5, 17, 101);
    }
    if (g == 5 && j == 17 && k == 101) {
        return oracle_expand_in_offset(2, 9, 333);
    }
    return oracle_expand_in_offset(g, j, k);
}

// M2: the (2, 3) and (3, 2) fragment tiles read in each other's order (a
// wrong fragment order that stays a bijection; no anchor sits in either
// tile).
int m2_fragment(int i, int j) {
    const int ti = i / 16;
    const int tj = j / 16;
    if (ti == 2 && tj == 3) {
        return oracle_fragment_index(i + 16, j - 16);
    }
    if (ti == 3 && tj == 2) {
        return oracle_fragment_index(i - 16, j + 16);
    }
    return oracle_fragment_index(i, j);
}

// M3: group 3 of expand_in never claims its offsets (a dropped group: the
// surviving entries are faithful, the map has holes).
int m3_expand_in(int g, int j, int k) {
    if (g == 3) {
        return kUnclaimed;
    }
    return oracle_expand_in_offset(g, j, k);
}

void check_mutations() {
    // M1: designed check = scatter fidelity (exactly that one fails).
    const ScatterMap m1 = {m1_expand_in, oracle_expand_hidden_offset,
                           oracle_contract_offset};
    CHECK_(scatter_anchors(m1));       // not the anchors' class
    CHECK_(!scatter_fidelity(m1));     // DESIGNED: caught here
    CHECK_(scatter_coverage(m1));      // a wrong permutation stays bijective

    // M2: designed check = fragment fidelity (exactly that one fails).
    CHECK_(fragment_anchors(m2_fragment));    // not the anchors' class
    CHECK_(!fragment_fidelity(m2_fragment));  // DESIGNED: caught here
    CHECK_(fragment_coverage(m2_fragment));   // still a permutation

    // M3: designed check = scatter coverage (exactly that one fails).
    const ScatterMap m3 = {m3_expand_in, oracle_expand_hidden_offset,
                           oracle_contract_offset};
    CHECK_(scatter_anchors(m3));       // anchors sit outside group 3
    CHECK_(scatter_fidelity(m3));      // claimed entries are faithful
    CHECK_(!scatter_coverage(m3));     // DESIGNED: caught here
}

// ---- synthetic fixture helpers --------------------------------------------

// Finite [-1, 1) grid value, exact in binary16 (DemoStore-style). The LCG
// runs incrementally across a fill (element i carries the state after i+1
// steps — identical values to a per-element re-derivation, O(n) not O(n^2)).
float grid_next(std::uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return static_cast<float>(static_cast<int>(s % 2048u) - 1024) / 1024.0f;
}

// e4m3/int4 code content: codes 12..28 (off the e4m3 ceiling, far from the
// NaN codes).
void fill_codes(std::vector<std::uint8_t>& b, std::size_t n,
                std::uint32_t seed) {
    b.resize(n);
    std::uint32_t s = seed * 2654435761u + 1u;
    for (std::size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        b[i] = static_cast<std::uint8_t>(12 + ((s >> 24) % 17u));
    }
}

void fill_source(std::vector<std::uint8_t>& b, weightsfile::Kind kind,
                 std::size_t elems, std::uint32_t seed) {
    std::uint32_t s = seed * 2654435761u + 1u;
    if (kind == weightsfile::Kind::kF16) {
        b.resize(2 * elems);
        for (std::size_t i = 0; i < elems; ++i) {
            const std::uint16_t h = f16_bits(grid_next(s));
            std::memcpy(b.data() + 2 * i, &h, 2);
        }
    } else if (kind == weightsfile::Kind::kF32) {
        b.resize(4 * elems);
        for (std::size_t i = 0; i < elems; ++i) {
            const float v = grid_next(s);
            std::memcpy(b.data() + 4 * i, &v, 4);
        }
    } else {
        fill_codes(b, elems, seed);
    }
}

std::string key_of(int block, int layer, const std::string& name) {
    return "block" + std::to_string(block) + ".layer"
        + std::to_string(layer) + "." + name;
}

FeatureBatchView make_view(float* storage, int ch, int rows, int cols) {
    FeatureBatchView v;
    v.data = storage;
    v.channels = ch;
    v.rows = rows;
    v.cols = cols;
    v.plane_stride = static_cast<std::ptrdiff_t>(rows) * cols;
    v.row_stride = static_cast<std::ptrdiff_t>(cols);
    return v;
}

// The whole-network harness state (the small plan at 64x72, the arena and
// the padded image) shared by the round-trip and the convert_source-driven
// negatives.
struct Harness {
    LayerSpec specs[dlss5_nr::kPlanMaxLayers];
    int count = 0;
    dlss5_nr::PlanPadding pad;
    float* arena = nullptr;
    std::size_t arena_floats = std::size_t{1} << 26;
    std::vector<float> image;
    FeatureBatchView view{};
    weightsfile::Manifest manifest;        // generated from the plan/recorder
    weightssource::SourceStore store;      // the synthetic source dictionary

    bool discover() {
        dlss5_nr::build_plan(64, 72, dlss5_nr::NetworkKind::kSmall, specs,
                             dlss5_nr::kPlanMaxLayers, count);
        dlss5_nr::plan_padding(64, 72, dlss5_nr::NetworkKind::kSmall, pad);
        arena = static_cast<float*>(
            std::malloc(arena_floats * sizeof(float)));
        image.assign(static_cast<std::size_t>(3) * pad.width * pad.height,
                     0.0f);
        for (std::size_t i = 0; i < image.size(); ++i) {
            image[i] = static_cast<float>((i * 2654435761u) % 1024u)
                / 1024.0f;  // [0, 1)
        }
        view = make_view(image.data(), 3, pad.width, pad.height);

        // The identity set is GENERATED from the plan by running the
        // recorder over a synthetic provider (DemoStore): never a
        // hand-copied table.
        static dlss5_demo::DemoStore demo;
        demo.preset = dlss5_demo::DemoStore::Preset::kSeeded;
        dlss5_nr::EvalWeights demo_inner;
        dlss5_demo::demo_bind_weights(demo_inner, demo);
        weightsfile::Recorder rec;
        rec.length_for = [](const char* name) -> int {
            int l = 0;
            return dlss5_demo::demo_detail::wildcard_length(name, l) ? l : -1;
        };
        dlss5_nr::EvalWeights demo_served;
        weightsfile::recorder_bind(demo_served, rec, demo_inner);
        static LayerValue outputs[dlss5_nr::kPlanMaxLayers];
        static int skip_s[dlss5_nr::kPlanMaxLayers];
        static int last_s[dlss5_nr::kPlanMaxLayers];
        dlss5_nr::EvalContext ctx;
        ctx.weights = &demo_served;
        ctx.scratch = arena;
        ctx.scratch_capacity = arena_floats;
        if (dlss5_nr::eval_reset(ctx) != Status::kOk) {
            return false;
        }
        NetworkRun run;
        run.image = &view;
        run.outputs = outputs;
        run.skip_sources = skip_s;
        run.last_uses = last_s;
        run.layer = dlss5_nr::eval_layer;
        run.release = dlss5_nr::eval_release;
        run.context = &ctx;
        if (dlss5_nr::network_run(specs, count, run) != Status::kOk) {
            return false;
        }
        if (!rec.err.ok()) {
            return false;
        }
        manifest = rec.manifest;

        // Build the synthetic source dictionary from the generated
        // manifest (ffwd records come from their packed source halves).
        std::set<std::string> have;
        for (const auto& m : manifest) {
            if (m.kind == weightsfile::Kind::kInt4) {
                for (const char* nm : {"ffwd_a", "ffwd_b"}) {
                    const std::string k = key_of(m.block, m.layer, nm);
                    if (have.count(k) != 0) {
                        continue;
                    }
                    have.insert(k);
                    std::vector<std::uint8_t> bytes;
                    fill_codes(bytes, 262144,
                               static_cast<std::uint32_t>(m.block * 7
                                                          + m.layer));
                    store.set(m.block, m.layer, nm, std::move(bytes));
                }
                continue;
            }
            const std::string k = key_of(m.block, m.layer, m.name);
            if (have.count(k) != 0) {
                continue;
            }
            have.insert(k);
            std::size_t elems = m.count;
            if (m.kind == weightsfile::Kind::kE4m3) {
                elems = static_cast<std::size_t>(m.rows) * m.cols;
            }
            std::vector<std::uint8_t> bytes;
            fill_source(bytes, m.kind, elems,
                        static_cast<std::uint32_t>(m.block * 131
                                                   + m.layer * 17
                                                   + static_cast<int>(m.kind)));
            store.set(m.block, m.layer, m.name, std::move(bytes));
        }
        return true;
    }

    // One direct eval run over the bound provider; the final map copied out.
    bool run(dlss5_nr::EvalWeights& weights, std::vector<float>& artefact) {
        static LayerValue outputs[dlss5_nr::kPlanMaxLayers];
        static int skip_s[dlss5_nr::kPlanMaxLayers];
        static int last_s[dlss5_nr::kPlanMaxLayers];
        dlss5_nr::EvalContext ctx;
        ctx.weights = &weights;
        ctx.scratch = arena;
        ctx.scratch_capacity = arena_floats;
        if (dlss5_nr::eval_reset(ctx) != Status::kOk) {
            return false;
        }
        NetworkRun run;
        run.image = &view;
        run.outputs = outputs;
        run.skip_sources = skip_s;
        run.last_uses = last_s;
        run.layer = dlss5_nr::eval_layer;
        run.release = dlss5_nr::eval_release;
        run.context = &ctx;
        if (dlss5_nr::network_run(specs, count, run) != Status::kOk) {
            return false;
        }
        const FeatureBatchView& result = outputs[count - 1].main;
        artefact.assign(result.data,
                        result.data + image.size());
        return true;
    }
};

// ---- the whole-network round-trip -----------------------------------------

void check_round_trip(Harness& h) {
    CHECK_(h.count > 0);
    CHECK_(h.manifest.size() > 0);
    // The generated identity set is unambiguous (this is the evidence the
    // probe recorded as "kind collisions: 0").
    weightsfile::WeightsFileError id_err;
    CHECK_(weightssource::validate_record_identities(h.manifest, id_err));
    CHECK_(id_err.ok());
    // Every record class the plan requests is present in the generated set.
    std::set<std::uint32_t> kinds;
    for (const auto& m : h.manifest) {
        kinds.insert(static_cast<std::uint32_t>(m.kind));
    }
    CHECK_(kinds.count(static_cast<std::uint32_t>(weightsfile::Kind::kE4m3))
           != 0);
    CHECK_(kinds.count(static_cast<std::uint32_t>(weightsfile::Kind::kF16))
           != 0);
    CHECK_(kinds.count(static_cast<std::uint32_t>(weightsfile::Kind::kF32))
           != 0);
    CHECK_(kinds.count(static_cast<std::uint32_t>(weightsfile::Kind::kInt4))
           != 0);

    // Path A: the direct synthetic path (the source provider bound to the
    // evaluator).
    dlss5_nr::EvalWeights srcw;
    weightssource::source_bind(srcw, h.store);
    std::vector<float> artefact_a;
    CHECK_(h.run(srcw, artefact_a));
    CHECK_(h.store.error.ok());
    CHECK_(artefact_a.size() == h.image.size());

    // The conversion: synthetic dictionary -> NWF1 (twice: determinism).
    std::vector<std::uint8_t> nwf;
    weightsfile::WeightsFileError conv_err;
    CHECK_(weightssource::convert_source(h.store, h.specs, h.count, h.view,
                                         h.arena, h.arena_floats, nwf,
                                         conv_err));
    CHECK_(conv_err.ok());
    CHECK_(nwf.size() > 0);
    std::vector<std::uint8_t> nwf2;
    weightsfile::WeightsFileError conv_err2;
    CHECK_(weightssource::convert_source(h.store, h.specs, h.count, h.view,
                                         h.arena, h.arena_floats, nwf2,
                                         conv_err2));
    CHECK_(conv_err2.ok());
    CHECK_(nwf2 == nwf);

    // The container's record set is exactly the generated plan identity set.
    weightsfile::WeightsFile wf;
    weightsfile::WeightsFileError read_err;
    CHECK_(weightsfile::read_container(nwf.data(), nwf.size(), wf, read_err));
    CHECK_(read_err.ok());
    CHECK_(wf.records.size() == h.manifest.size());
    std::size_t matched = 0;
    for (const auto& want : h.manifest) {
        if (wf.find(want.kind, want.block, want.layer, want.name, want.which)
            != nullptr) {
            ++matched;
        }
    }
    CHECK_(matched == h.manifest.size());

    // Path B: converter -> NWF1 -> file provider -> evaluator.
    weightsfile::FileStore fs;
    fs.file = wf;
    dlss5_nr::EvalWeights fbw;
    weightsfile::file_bind(fbw, fs);
    std::vector<float> artefact_b;
    CHECK_(h.run(fbw, artefact_b));
    CHECK_(fs.error.ok());
    CHECK_(artefact_b.size() == artefact_a.size());
    // BYTE-IDENTICAL artefacts (cmp-identical to the direct path): object
    // representations equal, so even a sign-of-zero divergence fails.
    CHECK_(std::memcmp(artefact_b.data(), artefact_a.data(),
                       artefact_a.size() * sizeof(float)) == 0);
}

// ---- the named negative matrix --------------------------------------------

// Direct drives of the fetch boundary (the converter's named rejection
// surface) with crafted dictionaries.
void check_negatives_fetch() {
    // Wrong packed shapes: each half named with its size.
    {
        weightssource::SourceStore s;
        std::vector<std::uint8_t> bytes;
        fill_codes(bytes, 262143, 1);
        s.set(0, 1, "ffwd_a", bytes);
        fill_codes(bytes, 262144, 2);
        s.set(0, 1, "ffwd_b", bytes);
        CHECK_(weightssource::source_ffwd(&s, 0, 1, 0, 512, 512) == nullptr);
        CHECK_(contains(s.error, "ffwd_a"));
        CHECK_(contains(s.error, "512*512"));
    }
    {
        weightssource::SourceStore s;
        std::vector<std::uint8_t> bytes;
        fill_codes(bytes, 262144, 1);
        s.set(0, 1, "ffwd_a", bytes);
        fill_codes(bytes, 262145, 2);
        s.set(0, 1, "ffwd_b", bytes);
        CHECK_(weightssource::source_ffwd(&s, 0, 1, 2, 512, 256) == nullptr);
        CHECK_(contains(s.error, "ffwd_b"));
        CHECK_(contains(s.error, "512*512"));
    }

    // Truncated fragments: the attn_pos_bias table one entry short names
    // the tensor and both counts.
    {
        weightssource::SourceStore s;
        std::vector<std::uint8_t> frag(2 * (16 * 4096 - 1), 0);
        s.set(1, 2, "attn_pos_bias", std::move(frag));
        CHECK_(weightssource::source_f16(&s, 1, 2, "attn_pos_bias", 16 * 4096)
               == nullptr);
        CHECK_(contains(s.error, "attn_pos_bias"));
        CHECK_(contains(s.error, "65535"));
        CHECK_(contains(s.error, "65536"));
    }

    // Count/shape mismatch: e4m3 exact shapes, f16/f32 exact counts, odd
    // binary16 bytes, a non-float32 byte count.
    {
        weightssource::SourceStore s;
        std::vector<std::uint8_t> bytes;
        fill_codes(bytes, 31, 3);
        s.set(2, 0, "qkv", bytes);
        CHECK_(weightssource::source_e4m3(&s, 2, 0, "qkv", 8, 4) == nullptr);
        CHECK_(contains(s.error, "qkv"));
        CHECK_(contains(s.error, "31"));
        CHECK_(contains(s.error, "32"));
    }
    {
        weightssource::SourceStore s;
        std::vector<std::uint8_t> bytes;
        fill_codes(bytes, 33, 3);
        s.set(2, 0, "qkv", bytes);
        CHECK_(weightssource::source_e4m3(&s, 2, 0, "qkv", 8, 4) == nullptr);
        CHECK_(contains(s.error, "33"));
        CHECK_(contains(s.error, "32"));
    }
    {
        weightssource::SourceStore s;
        std::vector<std::uint8_t> bytes(2 * 200, 0);
        s.set(4, 5, "some_table", bytes);
        CHECK_(weightssource::source_f16(&s, 4, 5, "some_table", 100)
               == nullptr);
        CHECK_(contains(s.error, "some_table"));
        CHECK_(contains(s.error, "200"));
        CHECK_(contains(s.error, "100"));
    }
    {
        weightssource::SourceStore s;
        std::vector<std::uint8_t> bytes(4 * 7, 0);
        s.set(4, 5, "tail", bytes);
        CHECK_(weightssource::source_f32(&s, 4, 5, "tail", 8) == nullptr);
        CHECK_(contains(s.error, "tail"));
        CHECK_(contains(s.error, "7"));
        CHECK_(contains(s.error, "8"));
    }
    {
        weightssource::SourceStore s;
        std::vector<std::uint8_t> bytes(5, 0);
        s.set(4, 5, "odd_half", bytes);
        CHECK_(weightssource::source_f16(&s, 4, 5, "odd_half", -1) == nullptr);
        CHECK_(contains(s.error, "odd_half"));
        CHECK_(contains(s.error, "odd binary16 byte count"));
    }
    {
        weightssource::SourceStore s;
        std::vector<std::uint8_t> bytes(6, 0);
        s.set(4, 5, "bad_f32", bytes);
        CHECK_(weightssource::source_f32(&s, 4, 5, "bad_f32", -1) == nullptr);
        CHECK_(contains(s.error, "bad_f32"));
        CHECK_(contains(s.error, "non-float32 byte count"));
    }

    // Non-finite values: binary16 NaN/infinity patterns, float32 NaN and
    // infinity, and the e4m3 NaN codes — each named with its index.
    {
        weightssource::SourceStore s;
        std::vector<std::uint8_t> bytes(2 * 4, 0);
        const std::uint16_t nan_bits = 0x7e00;
        std::memcpy(bytes.data() + 2 * 3, &nan_bits, 2);
        s.set(1, 1, "scale", bytes);
        CHECK_(weightssource::source_f16(&s, 1, 1, "scale", 4) == nullptr);
        CHECK_(contains(s.error, "scale"));
        CHECK_(contains(s.error, "non-finite binary16 value at index 3"));
    }
    {
        weightssource::SourceStore s;
        std::vector<std::uint8_t> bytes(2 * 4, 0);
        const std::uint16_t inf_bits = 0x7c00;
        std::memcpy(bytes.data(), &inf_bits, 2);
        s.set(1, 1, "scale", bytes);
        CHECK_(weightssource::source_f16(&s, 1, 1, "scale", 4) == nullptr);
        CHECK_(contains(s.error, "non-finite binary16 value at index 0"));
    }
    {
        weightssource::SourceStore s;
        std::vector<std::uint8_t> bytes(4 * 4, 0);
        const float nan = std::nanf("");
        std::memcpy(bytes.data() + 4 * 2, &nan, 4);
        s.set(1, 1, "tail", bytes);
        CHECK_(weightssource::source_f32(&s, 1, 1, "tail", 4) == nullptr);
        CHECK_(contains(s.error, "tail"));
        CHECK_(contains(s.error, "non-finite float32 value at index 2"));
    }
    {
        weightssource::SourceStore s;
        std::vector<std::uint8_t> bytes(4 * 4, 0);
        const float inf = std::numeric_limits<float>::infinity();
        std::memcpy(bytes.data() + 4, &inf, 4);
        s.set(1, 1, "tail", bytes);
        CHECK_(weightssource::source_f32(&s, 1, 1, "tail", 4) == nullptr);
        CHECK_(contains(s.error, "non-finite float32 value at index 1"));
    }
    {
        weightssource::SourceStore s;
        std::vector<std::uint8_t> bytes;
        fill_codes(bytes, 32, 5);
        bytes[5] = 0x7f;
        s.set(2, 0, "qkv", bytes);
        CHECK_(weightssource::source_e4m3(&s, 2, 0, "qkv", 8, 4) == nullptr);
        CHECK_(contains(s.error, "qkv"));
        CHECK_(contains(s.error, "non-finite e4m3 code at index 5"));
    }
    {
        weightssource::SourceStore s;
        std::vector<std::uint8_t> a(262144), b(262144);
        fill_codes(a, 262144, 6);
        fill_codes(b, 262144, 7);
        b[9] = 0xff;
        s.set(3, 3, "ffwd_a", std::move(a));
        s.set(3, 3, "ffwd_b", std::move(b));
        CHECK_(weightssource::source_ffwd(&s, 3, 3, 1, 2048, 64) == nullptr);
        CHECK_(contains(s.error, "ffwd_b"));
        CHECK_(contains(s.error, "non-finite e4m3 code at index 9"));
    }

    // Unknown names: a fetch for a key the dictionary lacks is named (the
    // fetch-side class), and the dictionary-side class is exercised through
    // convert_source below.
    {
        weightssource::SourceStore s;
        CHECK_(weightssource::source_e4m3(&s, 0, 9, "nope", 4, 4) == nullptr);
        CHECK_(contains(s.error, "missing source tensor 'block0.layer9.nope'"));
    }

    // The wildcard probe-length convention: a short table serves zero-padded
    // to the plan table length, the exact length answers, the next length
    // refuses silently (length discovery), and an over-long table names its
    // excess.
    {
        weightssource::SourceStore s;
        std::vector<std::uint8_t> table(2 * 5, 0);
        for (int i = 0; i < 5; ++i) {
            const std::uint16_t bits = f16_bits(static_cast<float>(i));
            std::memcpy(table.data() + 2 * i, &bits, 2);
        }
        s.set(-1, -1, "residual_scale", std::move(table));
        const float* got =
            weightssource::source_f16(&s, -1, -1, "residual_scale", -1);
        CHECK_(got != nullptr);
        if (got != nullptr) {
            bool padded = true;
            for (int i = 0; i < 5; ++i) {
                padded = padded && got[i] == static_cast<float>(i);
            }
            for (int i = 5; i < 4096; ++i) {
                padded = padded && got[i] == 0.0f;
            }
            CHECK_(padded);
        }
        CHECK_(weightssource::source_f16(&s, -1, -1, "residual_scale", 4096)
               != nullptr);
        CHECK_(weightssource::source_f16(&s, -1, -1, "residual_scale", 4097)
               == nullptr);
        CHECK_(s.error.ok());  // silent: normal length discovery
    }
    {
        weightssource::SourceStore s;
        std::vector<std::uint8_t> table(2 * 5000, 0);
        s.set(-1, -1, "residual_scale", std::move(table));
        CHECK_(weightssource::source_f16(&s, -1, -1, "residual_scale", -1)
               == nullptr);
        CHECK_(contains(s.error, "residual_scale"));
        CHECK_(contains(s.error, "exceeds the plan table length"));
    }
}

// The converter-level named rejections (duplicate identities, unknown names)
// and one end-to-end shape mismatch through convert_source.
void check_negatives_convert(Harness& h) {
    // Duplicate identity at the dictionary level: one key, two payloads.
    {
        weightssource::SourceStore s;
        std::vector<std::uint8_t> bytes(32, 1);
        s.set(0, 1, "qkv", bytes);
        s.set(0, 1, "qkv", bytes);
        CHECK_(contains(s.error, "duplicate source tensor 'block0.layer1.qkv'"));
        std::vector<std::uint8_t> nwf;
        weightsfile::WeightsFileError err;
        CHECK_(!weightssource::convert_source(s, nullptr, 0, FeatureBatchView{},
                                              nullptr, 0, nwf, err));
        CHECK_(contains(err, "block0.layer1.qkv"));
    }
    // Duplicate identity at the record level: the serialiser refuses a
    // capture set holding one identity twice, naming the record.
    {
        weightsfile::Recorder rec;
        weightsfile::Recorder::Rec r;
        std::memset(r.meta.name, 0, sizeof(r.meta.name));
        std::snprintf(r.meta.name, sizeof(r.meta.name), "%s", "qkv");
        r.meta.kind = weightsfile::Kind::kE4m3;
        r.meta.block = 2;
        r.meta.layer = 0;
        r.meta.which = -1;
        r.meta.rows = 8;
        r.meta.cols = 4;
        r.meta.count = 32;
        r.bytes = 32;
        r.data.assign(32, 1);
        rec.recs.push_back(r);
        r.data.assign(32, 2);
        rec.recs.push_back(r);
        std::vector<std::uint8_t> nwf;
        weightsfile::WeightsFileError err;
        CHECK_(!weightssource::recorder_to_nwf(rec, nwf, err));
        CHECK_(contains(err, "duplicate record identity"));
        CHECK_(contains(err, "qkv"));
    }
    // The container reader's named duplicate rejection (the downstream
    // backstop on the converter's output path).
    {
        std::vector<weightsfile::TensorRecord> records(2);
        for (auto& r : records) {
            std::memset(r.name, 0, sizeof(r.name));
            std::snprintf(r.name, sizeof(r.name), "%s", "qkv");
            r.kind = weightsfile::Kind::kE4m3;
            r.block = 2;
            r.layer = 0;
            r.which = -1;
            r.rows = 8;
            r.cols = 4;
            r.count = 32;
            r.offset = 0;
            r.length = 32;
        }
        std::vector<std::uint8_t> payload(32, 1);
        std::vector<std::uint8_t> nwf;
        weightsfile::WeightsFileError err;
        CHECK_(weightsfile::write_container(records, payload, nwf, err));
        weightsfile::WeightsFile wf;
        weightsfile::WeightsFileError read_err;
        CHECK_(!weightsfile::read_container(nwf.data(), nwf.size(), wf,
                                            read_err));
        CHECK_(contains(read_err, "file: duplicate record"));
        CHECK_(contains(read_err, "qkv"));
    }
    // The identity validator: exact duplicates and one key under two
    // element encodings both name the tensor.
    {
        weightsfile::Manifest m(2);
        for (auto& r : m) {
            std::memset(r.name, 0, sizeof(r.name));
            std::snprintf(r.name, sizeof(r.name), "%s", "qkv");
            r.kind = weightsfile::Kind::kE4m3;
            r.block = 2;
            r.layer = 0;
            r.which = -1;
        }
        weightsfile::WeightsFileError err;
        CHECK_(!weightssource::validate_record_identities(m, err));
        CHECK_(contains(err, "duplicate record identity"));
        CHECK_(contains(err, "qkv"));
        m[1].kind = weightsfile::Kind::kF16;
        weightsfile::WeightsFileError err2;
        CHECK_(!weightssource::validate_record_identities(m, err2));
        CHECK_(contains(err2, "block2.layer0.qkv"));
        CHECK_(contains(err2, "two element encodings"));
    }

    // Unknown name through the full conversion: one stray dictionary entry
    // the generated plan identity set never requests.
    {
        weightssource::SourceStore s = h.store;
        std::vector<std::uint8_t> bytes(16, 1);
        s.set(h.specs[0].block, h.specs[0].layer, "totally_bogus",
              std::move(bytes));
        std::vector<std::uint8_t> nwf;
        weightsfile::WeightsFileError err;
        CHECK_(!weightssource::convert_source(s, h.specs, h.count, h.view,
                                              h.arena, h.arena_floats, nwf,
                                              err));
        CHECK_(contains(err, "unknown source tensor"));
        CHECK_(contains(err, "totally_bogus"));
        CHECK_(contains(err, "never requests"));
    }

    // Count/shape mismatch end to end: one short e4m3 tensor rejects the
    // conversion naming the tensor and both counts.
    {
        weightssource::SourceStore s = h.store;
        // Shorten one record tensor (the first e4m3 manifest entry).
        for (const auto& m : h.manifest) {
            if (m.kind != weightsfile::Kind::kE4m3) {
                continue;
            }
            std::vector<std::uint8_t> bytes;
            fill_codes(bytes, static_cast<std::size_t>(m.rows) * m.cols - 1,
                       11);
            // Replace (not re-`set`) the payload: `set`'s duplicate guard
            // must not mask the count-mismatch path under test.
            s.tensors.erase(key_of(m.block, m.layer, m.name));
            s.set(m.block, m.layer, m.name, std::move(bytes));
            std::vector<std::uint8_t> nwf;
            weightsfile::WeightsFileError err;
            CHECK_(!weightssource::convert_source(s, h.specs, h.count, h.view,
                                                  h.arena, h.arena_floats,
                                                  nwf, err));
            CHECK_(contains(err, m.name));
            CHECK_(contains(err, "codes, the plan requests"));
            break;
        }
    }
}

}  // namespace

int main() {
    std::printf("native-weights-source: plan + scatter + fragment\n");
    check_scatter_maps();
    check_scatter_values();
    check_fragment_map();
    check_fragment_forms();
    check_mutations();

    std::printf("native-weights-source: round-trip + negatives\n");
    static Harness h;
    if (!CHECK(h.discover(), "discovery run builds the identity set",
               __LINE__)) {
        std::printf("native-weights-source: %d check(s) FAILED\n", g_failures);
        return 1;
    }
    check_round_trip(h);
    check_negatives_fetch();
    check_negatives_convert(h);

    std::printf("native-weights-source: %d checks passed\n", g_checks);
    if (g_failures != 0) {
        std::printf("native-weights-source: %d check(s) FAILED\n", g_failures);
        return 1;
    }
    std::printf("native-weights-source: OK\n");
    return 0;
}
