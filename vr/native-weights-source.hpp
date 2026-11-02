// SPDX-License-Identifier: MIT
// native-weights-source.hpp — the source-tensor-dictionary -> NWF1 converter.
//
// The L2 weight-ingestion boundary: take an in-memory dictionary
// name -> tensor supplied by any legitimate decoder and produce an NWF1
// container (vr/native-weights-file.hpp) the native evaluator consumes. This
// is the declared boundary that turns a source-format weight dictionary into
// kernel-ready tensor records. DATA-FREE by construction: it ships no weight
// content, touches no NVIDIA assets, performs no runtime or capture probes,
// and includes no proprietary payload. It accepts a dictionary from a
// legitimate decoder; we neither ship nor extract weights. Establishing this
// boundary proves READINESS only — never vendor parity, never DLSS5 quality.
//
// Source dictionary semantics (the pinned `Weights` keying, mauri870/
// dlssnr-pytorch commit 3beef7935a8ef976b47ab78b906e75600c52d2de,
// dlssnr/weights.py): every tensor is keyed `block{B}.layer{L}.{name}` and
// holds the exact bytes a typed accessor reads, so the element encoding is
// fixed per accessor:
//   - the `e4m3` accessor: raw u8 e4m3 codes (1 byte / element);
//   - the `f16` accessor: raw binary16 halves (2 bytes / element), decoded
//     to binary16-grid float32 for the evaluator (the NWF1 kind-2 record);
//   - the `f32` accessor: raw float32 (4 bytes / element);
//   - the split feed-forward: the two packed `ffwd_a`/`ffwd_b` u8 tensors
//     (each 512*512 codes) SCATTERED into the evaluator's three grouped
//     matrices (the ffwd callback, which 0/1/2).
//
// The converter NEVER hand-copies an identity table. The record set and its
// exact identities (name, kind, block, layer, which, rows, cols, count) are
// GENERATED from the plan by running the recorder (native-weights-file.hpp)
// over the network with the source provider bound underneath: the plan
// requests what it needs, the recorder captures each request at its plan
// identity, and the container records fall out exactly. The dictionary must
// match that generated set EXACTLY, every rejection naming the tensor: a
// requested tensor the dictionary lacks, or serves at the wrong shape or
// count; a dictionary entry the plan never requests (unknown name); one key
// supplied twice (duplicate identity); a key requested under two element
// encodings; and non-finite content (a binary16/float32 non-finite value, or
// an e4m3 NaN code 0x7f/0xff — the finite-code discipline is a disclosed
// strictness, see vr/NATIVE-WEIGHTS-SOURCE.md).
//
// The two packing transforms re-derived from the pinned reference
// (dlssnr/layers/split_swin.py, layers/swin.py, layers/vit.py):
//   1. Split feed-forward grouped scatter. The source's `ffwd_a`/`ffwd_b`
//     are the pinned record's two packed 512x512 halves; undoing the byte
//     permutation and re-grouping recovers, for each of the eight groups,
//     A_g [64, 512] -> expand_in, Q0_g [256, 64] -> expand_hidden and
//     Q2_g [64, 256] -> contract, in the evaluator's natural [out, in]
//     orientation (native-split-swin.hpp's SplitFfwdWeight). The 524,288
//     output codes are a permutation of the 524,288 input codes.
//   2. attn_pos_bias fragment order. The source's `attn_pos_bias` is the
//     [heads*4096] accumulator-fragment-order binary16 table exactly as the
//     evaluator fetches it; the deswizzle to the [64, 64] [query, key] bias
//     is the EVALUATOR's downstream step (swin_deswizzle_bias), the swin
//     rows carrying it through the exponent ramp and the split rows using it
//     raw. The converter transfers the fragment-order table faithfully (a
//     binary16 -> binary16-grid float32 decode); it must not reorder it. The
//     fragment-order index map is exported so the tests pin both the swin
//     ramp form and the split no-ramp form against hand-derived answers and
//     catch a wrong fragment order.
//
// Copyright (c) 2026 Mauri de Souza Meneguzzo (the unpacking semantics,
// adapted from the MIT reference; see vr/NATIVE-WEIGHTS-SOURCE.md for the
// full MIT notice and attribution).

#ifndef DLSS5_NR_NATIVE_WEIGHTS_SOURCE_HPP
#define DLSS5_NR_NATIVE_WEIGHTS_SOURCE_HPP

#include "native-weights-file.hpp"  // TensorRecord, Kind, Recorder, write_container, EvalWeights

#include <cstdint>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace dlss5_nr {
namespace weightssource {

using weightsfile::Kind;
using weightsfile::TensorRecord;
using weightsfile::WeightsFileError;

// ---- the split feed-forward grouped scatter ---------------------------------

namespace scatter_detail {

// `_slice_sources(512, 512)` (the pinned split_swin.py): the byte offset
// inside a packed 512x512 matrix region of every element of the generic
// unpacking. A packed region is a sequence of 512-byte slices, 16 output rows
// by 32 inputs, ordered input-group-major; inside a slice a lane holds eight
// bytes for each of two row halves.
inline void slice_sources(std::vector<int>& sources) {
    constexpr int rows = 512, cols = 512;
    constexpr int groups = rows / 16;  // 32
    sources.assign(static_cast<std::size_t>(rows) * cols, -1);
    for (int slice_index = 0; slice_index < groups * (cols / 32);
         ++slice_index) {
        const int row_group = slice_index % groups;
        const int col_group = slice_index / groups;
        for (int half = 0; half < 2; ++half) {
            for (int lane = 0; lane < 32; ++lane) {
                const int row = row_group * 16 + half * 8 + lane / 4;
                for (int byte = 0; byte < 8; ++byte) {
                    const int offset =
                        slice_index * 512 + lane * 16 + half * 8 + byte;
                    const int k_in_slice =
                        8 * (byte / 2) + 2 * (lane % 4) + byte % 2;
                    const int col = col_group * 32 + k_in_slice;
                    sources[static_cast<std::size_t>(row) * cols + col] =
                        offset;
                }
            }
        }
    }
}

inline int bit(int value, int index, int to) {
    return ((value >> index) & 1) << to;
}

}  // namespace scatter_detail

// The three grouped matrices' byte offsets inside the combined packed record
// (the pinned `_grouped_ffwd_matrices`). Row-major [out, in] per matrix:
//   expand_in     [512, 512]  row g*64 + j   = A_g  row j (512 keys)
//   expand_hidden [2048, 64]  row g*256 + j  = Q0_g row j
//   contract      [512, 256]  row g*64 + j   = Q2_g row j
// Each of the 524,288 output codes is a distinct byte of the packed record
// (ffwd_a ++ ffwd_b), so the scatter is a permutation of the input codes.
// The offset functions are exported so the tests pin the exact map (hand
// anchors, full-space fidelity against an independent restatement, and the
// full permutation coverage) and mutation-verify their discriminators.
inline int ffwd_expand_in_offset(int g, int j, int k) {
    using scatter_detail::bit;
    return bit(k, 0, 0) | bit(k, 1, 4) | bit(k, 2, 5)
        | bit(k, 3, 1) | bit(k, 4, 2) | bit(k, 5, 14)
        | bit(k, 6, 15) | bit(k, 7, 16) | bit(k, 8, 17)
        | bit(j, 0, 3) | bit(j, 1, 6) | bit(j, 2, 7) | bit(j, 3, 8)
        | bit(j, 4, 9) | bit(j, 5, 10) | bit(g, 0, 11)
        | bit(g, 1, 12) | bit(g, 2, 13);
}

inline int ffwd_expand_hidden_offset(int g, int j, int r) {
    using scatter_detail::bit;
    return 262144 + g * 16384
        + (bit(r, 0, 1) | bit(r, 1, 0) | bit(r, 2, 4) | bit(r, 3, 5)
           | bit(r, 4, 2) | bit(r, 5, 13) | bit(j, 0, 6)
           | bit(j, 1, 3) | bit(j, 2, 9) | bit(j, 3, 7)
           | bit(j, 4, 8) | bit(j, 5, 10) | bit(j, 6, 11)
           | bit(j, 7, 12));
}

inline int ffwd_contract_offset(int g, int n, int h) {
    using scatter_detail::bit;
    return 393216 + g * 16384
        + (bit(h, 0, 0) | bit(h, 1, 1) | bit(h, 2, 2) | bit(h, 3, 4)
           | bit(h, 4, 5) | bit(h, 5, 11) | bit(h, 6, 12)
           | bit(h, 7, 13) | bit(n, 0, 6) | bit(n, 1, 7)
           | bit(n, 2, 8) | bit(n, 3, 3) | bit(n, 4, 9)
           | bit(n, 5, 10));
}

inline void scatter_ffwd(const std::uint8_t* ffwd_a, const std::uint8_t* ffwd_b,
                         std::uint8_t* expand_in, std::uint8_t* expand_hidden,
                         std::uint8_t* contract) {
    std::vector<int> sources;
    scatter_detail::slice_sources(sources);
    // The combined packed record: ffwd_a scattered through the generic
    // packing into the first half, ffwd_b into the second.
    std::vector<std::uint8_t> record(2 * 512 * 512);
    for (int i = 0; i < 512; ++i) {
        for (int j = 0; j < 512; ++j) {
            const int p = sources[static_cast<std::size_t>(i) * 512 + j];
            record[static_cast<std::size_t>(p)] =
                ffwd_a[static_cast<std::size_t>(i) * 512 + j];
            record[static_cast<std::size_t>(262144 + p)] =
                ffwd_b[static_cast<std::size_t>(i) * 512 + j];
        }
    }
    // expand_in [512, 512] = record[ffwd_expand_in_offset(g, j, k)].
    for (int g = 0; g < 8; ++g) {
        for (int j = 0; j < 64; ++j) {
            for (int k = 0; k < 512; ++k) {
                expand_in[static_cast<std::size_t>(g * 64 + j) * 512 + k] =
                    record[static_cast<std::size_t>(
                        ffwd_expand_in_offset(g, j, k))];
            }
        }
    }
    // expand_hidden [2048, 64] = record[ffwd_expand_hidden_offset(g, j, r)].
    for (int g = 0; g < 8; ++g) {
        for (int j = 0; j < 256; ++j) {
            for (int r = 0; r < 64; ++r) {
                expand_hidden[static_cast<std::size_t>(g * 256 + j) * 64 + r] =
                    record[static_cast<std::size_t>(
                        ffwd_expand_hidden_offset(g, j, r))];
            }
        }
    }
    // contract [512, 256] = record[ffwd_contract_offset(g, n, h)].
    for (int g = 0; g < 8; ++g) {
        for (int n = 0; n < 64; ++n) {
            for (int h = 0; h < 256; ++h) {
                contract[static_cast<std::size_t>(g * 64 + n) * 256 + h] =
                    record[static_cast<std::size_t>(
                        ffwd_contract_offset(g, n, h))];
            }
        }
    }
}

// ---- attn_pos_bias fragment order -------------------------------------------

// The accumulator-fragment -> [query i, key j] deswizzle index (the pinned
// split_swin.py `unpack_bias` / swin.py `_deswizzle_bias`; the evaluator's
// swin_deswizzle_index). Exported so tests pin the fragment order for both
// the swin ramp form (swin_exp_bias) and the split no-ramp form
// (swin_deswizzle_bias) and catch a wrong fragment order. The converter
// transfers the fragment-order table without reordering; this is the map the
// evaluator applies downstream.
inline int attn_pos_bias_fragment_index(int i, int j) {
    const int lane = 4 * ((i % 16) % 8) + ((j % 16) % 8) / 2;
    const int slot = (j % 2) | (((i % 16) / 8) << 1) | (((j % 16) / 8) << 2);
    return (4 * (i / 16) + (j / 16)) * 256 + lane * 8 + slot;
}

// ---- the source dictionary -------------------------------------------------

// One source tensor: the raw bytes a pinned typed accessor reads for
// `block{B}.layer{L}.{name}`. The element encoding is fixed per accessor (see
// the header comment); the converter checks the byte length against the
// plan's request and names any mismatch.
struct SourceTensor {
    std::vector<std::uint8_t> bytes;
};

// The in-memory dictionary name -> tensor. `add`/`set` key by the pinned
// `block{B}.layer{L}.{name}`; the fetch callbacks look the same key up and
// decode per the accessor that asks. Decoded/scattered results are cached so
// the pointers handed to the evaluator (and the recorder) stay valid across
// fetches. Two named rejection disciplines run at this level: `set` refuses a
// duplicate key (the dictionary supplies one identity twice), and a key the
// plan never requests is an unknown name (the post-run key check in
// `convert_source`) — both fail the conversion naming the tensor.
struct SourceStore {
    std::map<std::string, SourceTensor> tensors;
    // The keys the plan's fetches actually touched (marked on every
    // successful lookup, including the length-discovery probes and both
    // halves of an ffwd fetch).
    std::set<std::string> consumed;
    // Named rejection surface: a request the dictionary cannot serve (missing
    // key, or a byte length that does not match the plan's shape/count).
    WeightsFileError error;

    // F16 decode cache (binary16 halves -> binary16-grid float32) and the
    // split feed-forward scatter cache, keyed by the request identity so each
    // fetched record owns stable storage.
    std::map<std::string, std::vector<float>> f16_cache;
    std::map<std::string, std::vector<float>> f32_cache;
    std::map<std::string, std::vector<std::uint8_t>> codes_cache;

    static std::string key(int block, int layer, const char* name) {
        return "block" + std::to_string(block) + ".layer"
            + std::to_string(layer) + "." + name;
    }

    void set(int block, int layer, const char* name,
             std::vector<std::uint8_t> bytes) {
        const std::string k = key(block, layer, name);
        if (tensors.find(k) != tensors.end()) {
            // Duplicate identity: one key, two payloads. Fail closed and
            // keep the first; `convert_source` refuses the conversion.
            error.add("duplicate source tensor '" + k + "'");
            return;
        }
        tensors[k].bytes = std::move(bytes);
        // A replacement (via the public map) must never be masked by a
        // stale decode/scatter cache: drop any cached renderings of this
        // key so the next fetch decodes the current payload.
        f16_cache.erase(k);
        f32_cache.erase(k);
        codes_cache.erase(k);
    }

    void mark(int block, int layer, const char* name) {
        consumed.insert(key(block, layer, name));
    }

    const SourceTensor* find(int block, int layer, const char* name) const {
        auto it = tensors.find(key(block, layer, name));
        return it == tensors.end() ? nullptr : &it->second;
    }
};

namespace source_detail {

inline void reject(SourceStore& s, const std::string& line) {
    s.error.add(line);
}

// Finite-code discipline (disclosed strictness, vr/NATIVE-WEIGHTS-SOURCE.md):
// e4m3fn's two NaN codes (0x7f/0xff) decode to NaN — a non-finite weight.
// The code space is otherwise dense and every other byte is a valid code.
inline bool e4m3_nan_code(std::uint8_t code) {
    return code == 0x7f || code == 0xff;
}

// Decode `count` binary16 halves to binary16-grid float32 (the pinned
// `raw.view(torch.float16).to(torch.float32)`).
inline void decode_f16(const std::uint8_t* raw, std::size_t count,
                       std::vector<float>& out) {
    out.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        std::uint16_t bits = 0;
        std::memcpy(&bits, raw + 2 * i, 2);
        out[i] = f16_from_bits(bits);
    }
}

// The wildcard record names and their fixed plan table lengths (the pinned
// constructors slice these records; the recorder's `length_for` hook is
// name-only, so the converter pads each to its plan length and serves it
// there). This is a probe-length convention, not the identity set: WHICH
// records exist is still generated from the plan by the recorder.
inline bool wildcard_length(const char* name, int& length) {
    if (std::strcmp(name, "upsample_gain") == 0) {
        length = 16;
        return true;
    }
    if (std::strcmp(name, "residual_scale") == 0
        || std::strcmp(name, "attn_residual_scale") == 0
        || std::strcmp(name, "scalars_b") == 0) {
        length = 4096;
        return true;
    }
    return false;
}

}  // namespace source_detail

// ---- the source provider (the EvalWeights fetch contract) -------------------

inline const std::uint8_t* source_e4m3(void* context, int block, int layer,
                                       const char* name, int rows, int cols) {
    SourceStore* s = static_cast<SourceStore*>(context);
    if (s == nullptr || rows < 0 || cols < 0) {
        return nullptr;
    }
    const SourceTensor* t = s->find(block, layer, name);
    if (t == nullptr) {
        // Identity miss: named (a real incompatibility), not a length probe.
        source_detail::reject(*s, "missing source tensor '"
                              + SourceStore::key(block, layer, name) + "'");
        return nullptr;
    }
    s->mark(block, layer, name);
    const std::size_t want = static_cast<std::size_t>(rows) * cols;
    if (t->bytes.size() != want) {
        // e4m3 fetches are exact (reshape(*shape); no length discovery), so a
        // code-count mismatch is a real incompatibility: name it.
        source_detail::reject(*s, "source tensor '"
                              + SourceStore::key(block, layer, name) + "' holds "
                              + std::to_string(t->bytes.size())
                              + " codes, the plan requests "
                              + std::to_string(want));
        return nullptr;
    }
    for (std::size_t i = 0; i < want; ++i) {
        if (source_detail::e4m3_nan_code(t->bytes[i])) {
            source_detail::reject(*s, "source tensor '"
                                  + SourceStore::key(block, layer, name)
                                  + "' has a non-finite e4m3 code at index "
                                  + std::to_string(i));
            return nullptr;
        }
    }
    return t->bytes.data();
}

inline const float* source_f16(void* context, int block, int layer,
                               const char* name, int count) {
    SourceStore* s = static_cast<SourceStore*>(context);
    if (s == nullptr || count < -1) {
        return nullptr;
    }
    const SourceTensor* t = s->find(block, layer, name);
    if (t == nullptr) {
        source_detail::reject(*s, "missing source tensor '"
                              + SourceStore::key(block, layer, name) + "'");
        return nullptr;
    }
    s->mark(block, layer, name);
    if ((t->bytes.size() % 2) != 0) {
        source_detail::reject(*s, "source tensor '"
                              + SourceStore::key(block, layer, name)
                              + "' has an odd binary16 byte count");
        return nullptr;
    }
    const std::size_t natural = t->bytes.size() / 2;
    int target = static_cast<int>(natural);
    const bool wildcard =
        source_detail::wildcard_length(name, target);
    if (wildcard && natural > static_cast<std::size_t>(target)) {
        source_detail::reject(*s, "source tensor '"
                              + SourceStore::key(block, layer, name)
                              + "' exceeds the plan table length");
        return nullptr;
    }
    if (count != -1 && count != target) {
        if (!wildcard) {
            // Non-wildcard f16 fetches are exact (no length discovery), so a
            // count mismatch is a real incompatibility: a truncated fragment
            // table or a count mismatch. Name it.
            source_detail::reject(*s, "source tensor '"
                                  + SourceStore::key(block, layer, name)
                                  + "' holds " + std::to_string(natural)
                                  + " binary16 entries, the plan requests "
                                  + std::to_string(count));
        }
        return nullptr;  // wildcard: the eval's length discovery
    }
    const std::string id = SourceStore::key(block, layer, name);
    auto it = s->f16_cache.find(id);
    if (it == s->f16_cache.end()) {
        std::vector<float> decoded;
        source_detail::decode_f16(t->bytes.data(), natural, decoded);
        for (std::size_t i = 0; i < decoded.size(); ++i) {
            const float v = decoded[i];
            if (!(v == v) || v > 65504.0f || v < -65504.0f) {
                source_detail::reject(*s, "source tensor '"
                                      + SourceStore::key(block, layer, name)
                                      + "' has a non-finite binary16 value at index "
                                      + std::to_string(i));
                return nullptr;
            }
        }
        if (wildcard) {
            decoded.resize(static_cast<std::size_t>(target), 0.0f);
        }
        it = s->f16_cache.emplace(id, std::move(decoded)).first;
    }
    return it->second.data();
}

inline const float* source_f32(void* context, int block, int layer,
                               const char* name, int count) {
    SourceStore* s = static_cast<SourceStore*>(context);
    if (s == nullptr || count < -1) {
        return nullptr;
    }
    const SourceTensor* t = s->find(block, layer, name);
    if (t == nullptr) {
        source_detail::reject(*s, "missing source tensor '"
                              + SourceStore::key(block, layer, name) + "'");
        return nullptr;
    }
    s->mark(block, layer, name);
    if ((t->bytes.size() % 4) != 0) {
        source_detail::reject(*s, "source tensor '"
                              + SourceStore::key(block, layer, name)
                              + "' has a non-float32 byte count");
        return nullptr;
    }
    const std::size_t natural = t->bytes.size() / 4;
    int target = static_cast<int>(natural);
    const bool wildcard =
        source_detail::wildcard_length(name, target);
    if (wildcard && natural > static_cast<std::size_t>(target)) {
        source_detail::reject(*s, "source tensor '"
                              + SourceStore::key(block, layer, name)
                              + "' exceeds the plan table length");
        return nullptr;
    }
    if (count != -1 && count != target) {
        if (!wildcard) {
            // Non-wildcard f32 fetches are exact: a count mismatch is real.
            source_detail::reject(*s, "source tensor '"
                                  + SourceStore::key(block, layer, name)
                                  + "' holds " + std::to_string(natural)
                                  + " float32 entries, the plan requests "
                                  + std::to_string(count));
        }
        return nullptr;  // wildcard: the eval's length discovery
    }
    // Non-finite rejection (both forms): the raw float32 content must be
    // finite wherever the plan reads it.
    {
        const std::size_t check = wildcard
            ? static_cast<std::size_t>(target) : natural;
        const float* vals = reinterpret_cast<const float*>(t->bytes.data());
        for (std::size_t i = 0; i < check && i < natural; ++i) {
            const float v = vals[i];
            if (!(v == v) || v > 3.402823466e+38f || v < -3.402823466e+38f) {
                source_detail::reject(*s, "source tensor '"
                                      + SourceStore::key(block, layer, name)
                                      + "' has a non-finite float32 value at index "
                                      + std::to_string(i));
                return nullptr;
            }
        }
    }
    if (!wildcard) {
        return reinterpret_cast<const float*>(t->bytes.data());
    }
    const std::string id = SourceStore::key(block, layer, name);
    auto it = s->f32_cache.find(id);
    if (it == s->f32_cache.end()) {
        std::vector<float> decoded(natural);
        std::memcpy(decoded.data(), t->bytes.data(), natural * sizeof(float));
        decoded.resize(static_cast<std::size_t>(target), 0.0f);
        it = s->f32_cache.emplace(id, std::move(decoded)).first;
    }
    return it->second.data();
}

inline const std::uint8_t* source_ffwd(void* context, int block, int layer,
                                       int which, int rows, int cols) {
    SourceStore* s = static_cast<SourceStore*>(context);
    if (s == nullptr || which < 0 || which > 2 || rows < 0 || cols < 0) {
        return nullptr;
    }
    const SourceTensor* a = s->find(block, layer, "ffwd_a");
    const SourceTensor* b = s->find(block, layer, "ffwd_b");
    if (a == nullptr || b == nullptr) {
        source_detail::reject(*s, "missing source tensor '"
                              + SourceStore::key(block, layer,
                                                 a == nullptr ? "ffwd_a"
                                                              : "ffwd_b")
                              + "'");
        return nullptr;
    }
    s->mark(block, layer, "ffwd_a");
    s->mark(block, layer, "ffwd_b");
    // Wrong packed shapes reject naming the offending half: each is exactly
    // one 512x512 packed region (the pinned `.reshape(512, 512)`).
    const SourceTensor* halves[2] = {a, b};
    for (const SourceTensor* half : halves) {
        const char* half_name = half == a ? "ffwd_a" : "ffwd_b";
        if (half->bytes.size() != 262144) {
            source_detail::reject(*s, "source tensor '"
                                  + SourceStore::key(block, layer, half_name)
                                  + "' holds "
                                  + std::to_string(half->bytes.size())
                                  + " packed codes, the plan requires 512*512");
            return nullptr;
        }
        for (std::size_t i = 0; i < half->bytes.size(); ++i) {
            if (source_detail::e4m3_nan_code(half->bytes[i])) {
                source_detail::reject(*s, "source tensor '"
                                      + SourceStore::key(block, layer, half_name)
                                      + "' has a non-finite e4m3 code at index "
                                      + std::to_string(i));
                return nullptr;
            }
        }
    }
    const std::string id = SourceStore::key(block, layer, "ffwd");
    auto it = s->codes_cache.find(id);
    if (it == s->codes_cache.end()) {
        std::vector<std::uint8_t> expand_in(512 * 512);
        std::vector<std::uint8_t> expand_hidden(2048 * 64);
        std::vector<std::uint8_t> contract(512 * 256);
        scatter_ffwd(a->bytes.data(), b->bytes.data(), expand_in.data(),
                     expand_hidden.data(), contract.data());
        std::vector<std::uint8_t> packed;
        packed.insert(packed.end(), expand_in.begin(), expand_in.end());
        packed.insert(packed.end(), expand_hidden.begin(), expand_hidden.end());
        packed.insert(packed.end(), contract.begin(), contract.end());
        it = s->codes_cache.emplace(id, std::move(packed)).first;
    }
    const std::uint8_t* base = it->second.data();
    if (which == 0) {
        return rows == 512 && cols == 512 ? base : nullptr;
    }
    if (which == 1) {
        return rows == 2048 && cols == 64 ? base + 512 * 512 : nullptr;
    }
    return rows == 512 && cols == 256 ? base + 512 * 512 + 2048 * 64 : nullptr;
}

inline void source_bind(EvalWeights& weights, SourceStore& store) {
    weights.context = &store;
    weights.e4m3 = source_e4m3;
    weights.f16 = source_f16;
    weights.f32 = source_f32;
    weights.ffwd = source_ffwd;
    weights.swin = nullptr;
    weights.int4 = nullptr;
}

// ---- source dictionary -> NWF1 ----------------------------------------------

// Serialise a recorder's captures to NWF1 container bytes (the same record
// rewrite `save_file` performs, kept in memory). Fail-closed identity
// discipline: two captures of one identity (name, kind, block, layer, which)
// reject by name rather than serialise a container the reader would refuse.
inline bool recorder_to_nwf(const weightsfile::Recorder& rec,
                            std::vector<std::uint8_t>& out,
                            WeightsFileError& err) {
    std::vector<TensorRecord> records;
    std::vector<std::uint8_t> payload;
    for (std::size_t i = 0; i < rec.recs.size(); ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            if (weightsfile::same_identity(rec.recs[j].meta, rec.recs[i].meta)) {
                char desc[128];
                weightsfile::describe(rec.recs[i].meta, desc, sizeof(desc));
                err.add(std::string("converter: duplicate record identity ")
                        + desc);
                return false;
            }
        }
    }
    for (const auto& r : rec.recs) {
        TensorRecord m = r.meta;
        m.offset = payload.size();
        m.length = r.bytes;
        records.push_back(m);
        payload.insert(payload.end(), r.data.begin(), r.data.end());
        while ((payload.size() % 8) != 0) {
            payload.push_back(0);
        }
    }
    return weightsfile::write_container(records, payload, out, err);
}

// The recorder's wildcard length-probe hook (name-only): the plan table
// length for a wildcard record, -1 when the name is not wildcard.
inline int source_length_for(const char* name) {
    int length = 0;
    return source_detail::wildcard_length(name, length) ? length : -1;
}

// The generated identity set must be unambiguous for a NAME-KEYED source
// dictionary. Two records may share `block{B}.layer{L}.{name}` only within
// one element encoding (the ffwd family's which 0/1/2 records are the pinned
// case: one packed ffwd_a/ffwd_b pair feeds all three). An exact identity
// (name, kind, block, layer, which) appearing twice, or one key requested
// with two element encodings, cannot be served by one dictionary entry:
// both reject by name. Returns true when the set is unambiguous.
inline bool validate_record_identities(const weightsfile::Manifest& manifest,
                                       WeightsFileError& err) {
    bool ok = true;
    for (std::size_t i = 0; i < manifest.size(); ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            const TensorRecord& a = manifest[j];
            const TensorRecord& b = manifest[i];
            if (weightsfile::same_identity(a, b)) {
                char desc[128];
                weightsfile::describe(b, desc, sizeof(desc));
                err.add(std::string("converter: duplicate record identity ")
                        + desc);
                ok = false;
                continue;
            }
            if (a.block == b.block && a.layer == b.layer
                && std::strncmp(a.name, b.name, sizeof(a.name)) == 0
                && a.kind != b.kind) {
                err.add(std::string("converter: source key '")
                        + SourceStore::key(a.block, a.layer, a.name)
                        + "' is requested with two element encodings ('"
                        + weightsfile::kind_name(a.kind) + "' and '"
                        + weightsfile::kind_name(b.kind) + "')");
                ok = false;
            }
        }
    }
    return ok;
}

// The converter entry point: source dictionary -> NWF1 bytes. The record set
// and its exact identities are GENERATED from the plan by running the network
// with the recorder over the source provider (never a hand-copied table): the
// plan requests what it needs at its plan identity, the recorder captures it,
// and the container records fall out exactly. A source tensor the plan
// requests but the dictionary cannot serve (missing, or at the wrong shape)
// is named in `err` and rejects the conversion; so does a dictionary entry
// the plan never requests (an unknown name) or one supplied twice. `scratch`
// is the evaluator arena (kEvalStateWords + the feature-map arena; see
// native-eval.hpp).
inline bool convert_source(SourceStore& store, const LayerSpec* specs,
                           int count, const FeatureBatchView& image,
                           float* scratch, std::size_t scratch_floats,
                           std::vector<std::uint8_t>& nwf_out,
                           WeightsFileError& err) {
    if (!store.error.ok()) {
        // The dictionary itself failed its construction discipline (e.g. a
        // duplicate `set`): surface the named reason before any run.
        for (const auto& p : store.error.problems) {
            err.add(p);
        }
        return false;
    }
    EvalWeights inner;
    source_bind(inner, store);
    weightsfile::Recorder rec;
    rec.length_for = source_length_for;
    EvalWeights served;
    weightsfile::recorder_bind(served, rec, inner);
    EvalContext ctx;
    ctx.weights = &served;
    ctx.scratch = scratch;
    ctx.scratch_capacity = scratch_floats;
    if (eval_reset(ctx) != Status::kOk) {
        err.add("converter: scratch arena too small");
        return false;
    }
    std::vector<LayerValue> outputs(static_cast<std::size_t>(count));
    std::vector<int> skip_scratch(static_cast<std::size_t>(count));
    std::vector<int> last_scratch(static_cast<std::size_t>(count));
    NetworkRun run;
    run.image = &image;
    run.outputs = outputs.data();
    run.skip_sources = skip_scratch.data();
    run.last_uses = last_scratch.data();
    run.layer = eval_layer;
    run.release = eval_release;
    run.context = &ctx;
    const Status st = network_run(specs, count, run);
    if (st != Status::kOk) {
        err.add("converter: network run failed (status "
                + std::to_string(static_cast<int>(st)) + ")");
        // Surface the source provider's named reason (a missing/short
        // tensor) when there is one.
        for (const auto& p : store.error.problems) {
            err.add(p);
        }
        return false;
    }
    // Unknown names: a dictionary entry the plan never requested. The
    // consumed set is the exact fetch footprint (both ffwd halves included),
    // so any leftover key is a name outside the generated identity set.
    bool unknown = false;
    for (const auto& kv : store.tensors) {
        if (store.consumed.find(kv.first) == store.consumed.end()) {
            err.add("converter: unknown source tensor '" + kv.first
                    + "' (the plan never requests it)");
            unknown = true;
        }
    }
    if (unknown) {
        return false;
    }
    if (!validate_record_identities(rec.manifest, err)) {
        return false;
    }
    if (!rec.err.ok()) {
        err = rec.err;
        return false;
    }
    return recorder_to_nwf(rec, nwf_out, err);
}

}  // namespace weightssource
}  // namespace dlss5_nr

#endif  // DLSS5_NR_NATIVE_WEIGHTS_SOURCE_HPP
