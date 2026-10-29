// native-weights-file.hpp — the data-free weights interface.
//
// A weight-file loader + compatibility-verification harness for the
// evaluator's EvalWeights tensor layer. It is deliberately DATA-FREE:
// it defines and verifies a container format, ships no weight content,
// touches no NVIDIA assets, and performs no runtime probing. A file is
// Compatibility is established BEFORE evaluation by the verification
// harness (weights-verify / validate-weights.sh): every mismatch class
// is named there. The loader itself enforces structure and content, the
// fetch callbacks enforce shapes silently (the eval's length-discovery
// probes demand it), and identity misses are named at run time.
//
// Format ("NWF1", little-endian throughout):
//
//   header (40 bytes):
//     magic[8]       "NWF1\x0d\x0a\x1a\x0a"
//     version   u32  1
//     flags     u32  bit 0: payload hash present (always set by this writer)
//     count     u32  tensor record count
//     reserved  u32  0
//     payload_bytes u64
//     payload_hash  u64  FNV-1a 64 over (records blob || payload)
//
//   then `count` records (96 bytes each):
//     name[48]  NUL-padded ASCII tensor name (the EvalWeights name; the
//               ffwd matrices use the fixed name "ffwd")
//     kind   u32   1 = e4m3 codes (u8 elements), 2 = binary16-grid values
//                  held as float32 (4-byte elements; the EvalWeights f16
//                  callback's contract), 3 = free float32 (4-byte
//                  elements), 4 = int4 packed codes (u8 elements)
//     block  i32   the EvalWeights block index (or -1)
//     layer  i32   the EvalWeights layer index (or -1)
//     which  i32   the ffwd `which` selector (or -1)
//     rows   u32   matrix rows (0 for scalar tables)
//     cols   u32   matrix cols (0 for scalar tables)
//     count  u32   element count (rows*cols for matrices, the table
//                  length for scalars)
//     offset u64   payload byte offset (8-byte aligned)
//     length u64   payload byte length
//     pad    u32   0 (record alignment to 96 bytes)
//
//   then the payload blob.
//
// Format-agnostic plumbing: the compatibility verifier consumes TENSOR
// RECORDS (metadata + content), not file bytes, so a legitimately
// obtained weight file in any source format reaches the same checks by
// implementing one decoder that emits records (see
// NATIVE-WEIGHTS-FILE.md, "accepting a source-format file"). The
// reference-validation resume path is the single command
// `vr/validate-weights.sh <file>`; it regenerates the request manifest
// from the plan itself (no static table to drift), verifies the file
// against it, and runs the CPU-vs-GPU byte-identity reference with the
// file provider.
//
// Scope: the loader serves the EvalWeights MATRICES AND SCALARS (e4m3,
// binary16, float32, int4). The structured site records (swin/int4
// sites) are absent in a file, which is the evaluator's documented
// "run the e4m3 dataflow" case, not an error.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "native-eval.hpp"  // EvalWeights, Status, f16

namespace dlss5_nr {
namespace weightsfile {

// ---- error surface ---------------------------------------------------------

// The compatibility/content verifiers accumulate every problem they find
// (not just the first), so a rejected file reports its whole mismatch
// set at once. The structural loader rejects at the first problem (the
// container cannot be trusted past it).
struct WeightsFileError {
    std::vector<std::string> problems;

    bool ok() const { return problems.empty(); }

    void add(const std::string& line) { problems.push_back(line); }

    std::string str() const {
        std::string out;
        for (const auto& p : problems) {
            if (!out.empty()) {
                out += "\n";
            }
            out += p;
        }
        return out;
    }
};

// ---- tensor records --------------------------------------------------------

enum class Kind : std::uint32_t {
    kE4m3 = 1,
    kF16 = 2,
    kF32 = 3,
    kInt4 = 4,
};

inline const char* kind_name(Kind k) {
    switch (k) {
        case Kind::kE4m3: return "e4m3";
        case Kind::kF16: return "binary16";
        case Kind::kF32: return "float32";
        case Kind::kInt4: return "int4";
    }
    return "unknown";
}

struct TensorRecord {
    char name[48] = {0};
    Kind kind = Kind::kE4m3;
    std::int32_t block = -1;
    std::int32_t layer = -1;
    std::int32_t which = -1;
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    std::uint32_t count = 0;
    std::uint64_t offset = 0;
    std::uint64_t length = 0;
};

inline bool same_identity(const TensorRecord& a, const TensorRecord& b) {
    return std::strncmp(a.name, b.name, sizeof(a.name)) == 0
        && a.kind == b.kind && a.block == b.block && a.layer == b.layer
        && a.which == b.which;
}

inline bool same_shape(const TensorRecord& a, const TensorRecord& b) {
    return a.rows == b.rows && a.cols == b.cols && a.count == b.count;
}

inline void describe(const TensorRecord& r, char* buf, std::size_t n) {
    std::snprintf(buf, n,
                  "'%s' (%s, block %d layer %d which %d) %ux%u, %u elements",
                  r.name, kind_name(r.kind), static_cast<int>(r.block),
                  static_cast<int>(r.layer), static_cast<int>(r.which),
                  static_cast<unsigned>(r.rows),
                  static_cast<unsigned>(r.cols),
                  static_cast<unsigned>(r.count));
}

// ---- the container ---------------------------------------------------------

inline std::uint64_t fnv1a(const std::uint8_t* data, std::size_t bytes,
                           std::uint64_t h = 0xcbf29ce484222325ull) {
    for (std::size_t i = 0; i < bytes; ++i) {
        h ^= static_cast<std::uint64_t>(data[i]);
        h *= 0x100000001b3ull;
    }
    return h;
}

constexpr std::uint8_t kMagic[8] = {'N', 'W', 'F', '1', 0x0d, 0x0a, 0x1a, 0x0a};
constexpr std::uint32_t kVersion = 1;
constexpr std::size_t kHeaderBytes = 40;
constexpr std::size_t kRecordBytes = 96;

struct WeightsFile {
    std::vector<TensorRecord> records;
    std::vector<std::uint8_t> payload;

    const TensorRecord* find(Kind kind, std::int32_t block, std::int32_t layer,
                             const char* name, std::int32_t which) const {
        for (const auto& r : records) {
            if (r.kind == kind && r.block == block && r.layer == layer
                && r.which == which
                && std::strncmp(r.name, name, sizeof(r.name)) == 0) {
                return &r;
            }
        }
        return nullptr;
    }
};

// Serialise (records + payload) to the container bytes.
inline bool write_container(const std::vector<TensorRecord>& records,
                            const std::vector<std::uint8_t>& payload,
                            std::vector<std::uint8_t>& out,
                            WeightsFileError& err) {
    out.clear();
    out.resize(kHeaderBytes + records.size() * kRecordBytes + payload.size());
    std::uint8_t* p = out.data();
    std::memcpy(p, kMagic, 8);
    p += 8;
    const std::uint32_t version = kVersion;
    const std::uint32_t flags = 1;  // payload hash present
    const std::uint32_t count = static_cast<std::uint32_t>(records.size());
    const std::uint32_t reserved = 0;
    const std::uint64_t payload_bytes = payload.size();
    std::memcpy(p, &version, 4); p += 4;
    std::memcpy(p, &flags, 4); p += 4;
    std::memcpy(p, &count, 4); p += 4;
    std::memcpy(p, &reserved, 4); p += 4;
    std::memcpy(p, &payload_bytes, 8); p += 8;
    std::uint8_t* hash_slot = p;
    std::memset(hash_slot, 0, 8);
    p += 8;
    for (const auto& r : records) {
        std::uint8_t* rp = p;
        std::memset(rp, 0, kRecordBytes);
        std::memcpy(rp, r.name, sizeof(r.name));
        rp += sizeof(r.name);
        const std::uint32_t kind = static_cast<std::uint32_t>(r.kind);
        std::memcpy(rp, &kind, 4); rp += 4;
        std::memcpy(rp, &r.block, 4); rp += 4;
        std::memcpy(rp, &r.layer, 4); rp += 4;
        std::memcpy(rp, &r.which, 4); rp += 4;
        std::memcpy(rp, &r.rows, 4); rp += 4;
        std::memcpy(rp, &r.cols, 4); rp += 4;
        std::memcpy(rp, &r.count, 4); rp += 4;
        std::memcpy(rp, &r.offset, 8); rp += 8;
        std::memcpy(rp, &r.length, 8); rp += 8;
        p += kRecordBytes;
    }
    if (!payload.empty()) {
        std::memcpy(p, payload.data(), payload.size());
    }
    // Hash from `version` onward (the header fields that partition the
    // body plus the body), with the hash slot zeroed - the reader
    // recomputes over exactly this region.
    const std::uint64_t hash =
        fnv1a(out.data() + 8, out.size() - 8);
    std::memcpy(hash_slot, &hash, 8);
    (void)err;
    return true;
}

// Parse + structural validation (bounds, alignment, hash, duplicates).
// Content validation is the verifier's job.
inline bool read_container(const std::uint8_t* data, std::size_t bytes,
                           WeightsFile& file, WeightsFileError& err) {
    file.records.clear();
    file.payload.clear();
    if (data == nullptr || bytes < kHeaderBytes) {
        err.add("file: too small for the NWF1 header");
        return false;
    }
    if (std::memcmp(data, kMagic, 8) != 0) {
        err.add("file: bad magic (not an NWF1 container)");
        return false;
    }
    const std::uint8_t* p = data + 8;
    std::uint32_t version = 0, flags = 0, count = 0, reserved = 0;
    std::uint64_t payload_bytes = 0, payload_hash = 0;
    std::memcpy(&version, p, 4); p += 4;
    std::memcpy(&flags, p, 4); p += 4;
    std::memcpy(&count, p, 4); p += 4;
    std::memcpy(&reserved, p, 4); p += 4;
    std::memcpy(&payload_bytes, p, 8); p += 8;
    std::memcpy(&payload_hash, p, 8); p += 8;
    if (version != kVersion) {
        char line[96];
        std::snprintf(line, sizeof(line),
                      "file: unsupported version %u (expected %u)",
                      static_cast<unsigned>(version),
                      static_cast<unsigned>(kVersion));
        err.add(line);
        return false;
    }
    (void)reserved;
    if ((flags & 1u) == 0) {
        err.add("file: payload hash not present (bit 0 must be set)");
        return false;
    }
    if ((flags & ~1u) != 0) {
        char line[96];
        std::snprintf(line, sizeof(line),
                      "file: unknown flags 0x%08x",
                      static_cast<unsigned>(flags));
        err.add(line);
        return false;
    }
    const std::uint64_t body_bytes = bytes - kHeaderBytes;
    const std::uint64_t expected_body =
        static_cast<std::uint64_t>(count) * kRecordBytes + payload_bytes;
    if (expected_body != body_bytes) {
        char line[128];
        std::snprintf(line, sizeof(line),
                      "file: size mismatch (header declares %llu body bytes, "
                      "file has %llu)",
                      static_cast<unsigned long long>(expected_body),
                      static_cast<unsigned long long>(body_bytes));
        err.add(line);
        return false;
    }
    {
        // The hash covers the header fields that PARTITION the body as
        // well as the body itself (the count/payload_bytes compensating-
        // edit hole), computed with the hash slot zeroed.
        std::vector<std::uint8_t> region(data + 8, data + bytes);
        // The hash slot is at file offset 32..40 = region[24..32).
        std::fill(region.begin() + 24, region.begin() + 32,
                  std::uint8_t{0});
        const std::uint64_t hash = fnv1a(region.data(), region.size());
        if (hash != payload_hash) {
            err.add("file: payload hash mismatch (corrupt or edited content)");
            return false;
        }
    }
    file.records.resize(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        TensorRecord& r = file.records[i];
        const std::uint8_t* rp = data + kHeaderBytes
            + static_cast<std::size_t>(i) * kRecordBytes;
        std::memcpy(r.name, rp, sizeof(r.name));
        r.name[sizeof(r.name) - 1] = '\0';
        std::uint32_t kind = 0;
        std::memcpy(&kind, rp + 48, 4);
        r.kind = static_cast<Kind>(kind);
        std::memcpy(&r.block, rp + 52, 4);
        std::memcpy(&r.layer, rp + 56, 4);
        std::memcpy(&r.which, rp + 60, 4);
        std::memcpy(&r.rows, rp + 64, 4);
        std::memcpy(&r.cols, rp + 68, 4);
        std::memcpy(&r.count, rp + 72, 4);
        std::memcpy(&r.offset, rp + 76, 8);
        std::memcpy(&r.length, rp + 84, 8);
        char line[160];
        if (kind < static_cast<std::uint32_t>(Kind::kE4m3)
            || kind > static_cast<std::uint32_t>(Kind::kInt4)) {
            std::snprintf(line, sizeof(line),
                          "file: record %u has unknown kind %u",
                          static_cast<unsigned>(i),
                          static_cast<unsigned>(kind));
            err.add(line);
            return false;
        }
        if (r.length != 0 && (r.offset % 8) != 0) {
            std::snprintf(line, sizeof(line),
                          "file: record %u payload offset %llu is not "
                          "8-byte aligned",
                          static_cast<unsigned>(i),
                          static_cast<unsigned long long>(r.offset));
            err.add(line);
            return false;
        }
        if (r.offset > payload_bytes || r.length > payload_bytes - r.offset) {
            std::snprintf(line, sizeof(line),
                          "file: record %u payload range [%llu, %llu) is "
                          "outside the %llu-byte payload",
                          static_cast<unsigned>(i),
                          static_cast<unsigned long long>(r.offset),
                          static_cast<unsigned long long>(r.offset + r.length),
                          static_cast<unsigned long long>(payload_bytes));
            err.add(line);
            return false;
        }
        const std::uint64_t elements = static_cast<std::uint64_t>(r.rows)
            * static_cast<std::uint64_t>(r.cols);
        const std::uint64_t want = r.kind == Kind::kF16
                || r.kind == Kind::kF32
            ? elements * 4ull
            : elements;
        const std::uint64_t want_table =
            r.kind == Kind::kF16 || r.kind == Kind::kF32
            ? static_cast<std::uint64_t>(r.count) * 4ull
            : static_cast<std::uint64_t>(r.count);
        const std::uint64_t want_bytes = elements != 0 ? want : want_table;
        if ((r.rows != 0 || r.cols != 0) && r.count != elements) {
            std::snprintf(line, sizeof(line),
                          "file: record %u count %u does not match its %llu "
                          "matrix elements",
                          static_cast<unsigned>(i),
                          static_cast<unsigned>(r.count),
                          static_cast<unsigned long long>(elements));
            err.add(line);
            return false;
        }
        if (r.length != want_bytes) {
            std::snprintf(line, sizeof(line),
                          "file: record %u length %llu does not match its "
                          "shape (%llu bytes expected)",
                          static_cast<unsigned>(i),
                          static_cast<unsigned long long>(r.length),
                          static_cast<unsigned long long>(want_bytes));
            err.add(line);
            return false;
        }
        for (std::uint32_t j = 0; j < i; ++j) {
            if (same_identity(file.records[j], r)) {
                std::snprintf(line, sizeof(line),
                              "file: duplicate record %u ('%s')",
                              static_cast<unsigned>(i), r.name);
                err.add(line);
                return false;
            }
        }
    }
    file.payload.assign(data + kHeaderBytes + count * kRecordBytes,
                        data + bytes);
    return true;
}

// ---- content verification (format-agnostic) --------------------------------

// Rejects non-finite content and binary16 NaN/infinity bit patterns; the
// e4m3 and int4 code spaces are dense (every byte is a valid code).
inline void verify_content(const WeightsFile& file, WeightsFileError& err) {
    for (const auto& r : file.records) {
        const std::uint8_t* base = file.payload.data() + r.offset;
        char line[320];
        char desc[128];
        describe(r, desc, sizeof(desc));
        if (r.kind == Kind::kF16) {
            // float32 elements that must lie on the binary16 grid (the f16
            // callback's contract): finite, within the finite binary16
            // range, and representable (f16(v) == v).
            const std::size_t n = static_cast<std::size_t>(r.length) / 4;
            for (std::size_t i = 0; i < n; ++i) {
                float v = 0.0f;
                std::memcpy(&v, base + i * 4, 4);
                if (!(v == v) || v > 65504.0f || v < -65504.0f
                    || f16(v) != v) {
                    std::snprintf(line, sizeof(line),
                                  "content: %s has a value off the binary16 "
                                  "grid at index %u (%g)",
                                  desc, static_cast<unsigned>(i),
                                  static_cast<double>(v));
                    err.add(line);
                    break;
                }
            }
        } else if (r.kind == Kind::kF32) {
            const std::size_t n = static_cast<std::size_t>(r.length) / 4;
            for (std::size_t i = 0; i < n; ++i) {
                float v = 0.0f;
                std::memcpy(&v, base + i * 4, 4);
                if (!(v == v) || v > 3.402823466e+38f || v < -3.402823466e+38f) {
                    std::snprintf(line, sizeof(line),
                                  "content: %s has a non-finite float32 at "
                                  "index %u", desc,
                                  static_cast<unsigned>(i));
                    err.add(line);
                    break;
                }
            }
        }
    }
}

// ---- compatibility verification --------------------------------------------

// The request manifest: the exact tensors the evaluator fetches for a run
// (identity + shape). Produced by the recorder during a reference run —
// the plan's request set is deterministic, so the manifest is regenerated
// from the plan itself rather than maintained as a static table.
using Manifest = std::vector<TensorRecord>;

// Verifies `file` against `manifest`: every expected tensor must be
// present with the exact identity and shape; every file record must be
// expected. Content is verified too. Returns true on a full match.
inline bool verify_compatibility(const WeightsFile& file,
                                 const Manifest& manifest,
                                 WeightsFileError& err) {
    for (const auto& want : manifest) {
        const TensorRecord* got =
            file.find(want.kind, want.block, want.layer, want.name, want.which);
        char desc[128];
        describe(want, desc, sizeof(desc));
        if (got == nullptr) {
            char line[176];
            std::snprintf(line, sizeof(line), "missing: %s", desc);
            err.add(line);
            continue;
        }
        if (!same_shape(*got, want)) {
            char line[320];
            char have[128];
            describe(*got, have, sizeof(have));
            std::snprintf(line, sizeof(line), "shape: expected %s, file has %s",
                          desc, have);
            err.add(line);
        }
    }
    for (const auto& r : file.records) {
        bool expected = false;
        for (const auto& want : manifest) {
            if (same_identity(r, want)) {
                expected = true;
                break;
            }
        }
        if (!expected) {
            char line[176];
            char desc[128];
            describe(r, desc, sizeof(desc));
            std::snprintf(line, sizeof(line),
                          "unexpected: %s (not requested by the plan)", desc);
            err.add(line);
        }
    }
    verify_content(file, err);
    return err.ok();
}

// ---- the recording provider ------------------------------------------------

// Wraps an inner EvalWeights and captures every tensor request + the bytes
// returned, producing the manifest and the payload for the container
// writer. Repeat fetches of one identity must return identical bytes (the
// DemoStore contract); a divergence is recorded as an error instead of
// silently dumped.
struct Recorder {
    const EvalWeights* inner = nullptr;
    // The EvalWeights length-probe convention (count == -1): the caller
    // reads the response at the PLAN's known table length (see
    // native-eval.hpp's fetch_f16/f32 and native-demo-support.hpp's
    // wildcard_length). The recorder must therefore know the plan's table
    // lengths to capture probes correctly; `length_for` returns the length
    // for a name (or <= 0 when unknown, in which case the probe passes
    // through to the inner uncaptured and the recorder records an error -
    // the dump refuses to write rather than record a wrong-length record).
    int (*length_for)(const char* name) = nullptr;
    Manifest manifest;
    std::vector<std::uint8_t> payload;
    WeightsFileError err;

    struct Rec {
        TensorRecord meta;
        std::size_t payload_offset = 0;
        std::size_t bytes = 0;
        // Stable storage: pointers handed to the evaluator must survive
        // later captures, so each record owns its bytes (the heap block a
        // vector owns does not move when the Rec vector reallocates).
        std::vector<std::uint8_t> data;
    };
    std::vector<Rec> recs;

    const std::uint8_t* capture(Kind kind, std::int32_t block,
                                std::int32_t layer, const char* name,
                                std::int32_t which, std::uint32_t rows,
                                std::uint32_t cols, std::uint32_t count,
                                const std::uint8_t* src, std::size_t bytes) {
        for (std::size_t i = 0; i < recs.size(); ++i) {
            Rec& r = recs[i];
            if (r.meta.kind == kind && r.meta.block == block
                && r.meta.layer == layer && r.meta.which == which
                && std::strncmp(r.meta.name, name, sizeof(r.meta.name)) == 0) {
                // Extension contract (native-demo-support.hpp): a later
                // request may be LONGER than the first (the plan probes at
                // a short length and fetches at the full one); the first
                // bytes must agree. A SHORTER request reads the same
                // prefix. Only a prefix divergence is an error.
                const std::size_t overlap =
                    bytes < r.bytes ? bytes : r.bytes;
                if (std::memcmp(r.data.data(), src, overlap) != 0) {
                    err.add(std::string("recorder: repeat fetch of '") + name
                            + "' returned different bytes");
                }
                if (bytes > r.bytes) {
                    r.data.assign(src, src + bytes);
                    r.bytes = bytes;
                    r.meta.count = count;
                    if (count != 0 && (rows != 0 || cols != 0)) {
                        r.meta.rows = rows;
                        r.meta.cols = cols;
                    }
                    for (auto& m : manifest) {
                        if (same_identity(m, r.meta)) {
                            m = r.meta;
                            break;
                        }
                    }
                }
                return r.data.data();
            }
        }
        Rec rec;
        std::memset(rec.meta.name, 0, sizeof(rec.meta.name));
        std::snprintf(rec.meta.name, sizeof(rec.meta.name), "%s", name);
        rec.meta.kind = kind;
        rec.meta.block = block;
        rec.meta.layer = layer;
        rec.meta.which = which;
        rec.meta.rows = rows;
        rec.meta.cols = cols;
        rec.meta.count = count;
        rec.payload_offset = payload.size();
        rec.bytes = bytes;
        rec.data.assign(src, src + bytes);
        payload.insert(payload.end(), src, src + bytes);
        // Keep the payload 8-byte aligned for the container contract.
        while ((payload.size() % 8) != 0) {
            payload.push_back(0);
        }
        recs.push_back(std::move(rec));
        manifest.push_back(recs.back().meta);
        return recs.back().data.data();
    }
};

inline const std::uint8_t* recorder_e4m3(void* context, int block, int layer,
                                         const char* name, int rows, int cols) {
    Recorder* r = static_cast<Recorder*>(context);
    if (r == nullptr || r->inner == nullptr || r->inner->e4m3 == nullptr
        || rows < 0 || cols < 0) {
        return nullptr;
    }
    const std::uint8_t* got =
        r->inner->e4m3(r->inner->context, block, layer, name, rows, cols);
    if (got == nullptr) {
        return nullptr;
    }
    return r->capture(Kind::kE4m3, block, layer, name, -1,
                      static_cast<std::uint32_t>(rows),
                      static_cast<std::uint32_t>(cols),
                      static_cast<std::uint32_t>(rows) * cols, got,
                      static_cast<std::size_t>(rows) * cols);
}

inline const float* recorder_f16(void* context, int block, int layer,
                                 const char* name, int count) {
    Recorder* r = static_cast<Recorder*>(context);
    if (r == nullptr || r->inner == nullptr || r->inner->f16 == nullptr
        || count < -1) {
        return nullptr;
    }
    const float* got =
        r->inner->f16(r->inner->context, block, layer, name, count);
    if (got == nullptr) {
        return nullptr;
    }
    // count == -1 is the length probe; the inner filled its contract
    // length, which the capture records as the table count.
    int length = count;
    if (count == -1) {
        length = r->length_for != nullptr ? r->length_for(name) : -1;
        if (length <= 0) {
            r->err.add(std::string("recorder: probe of unknown length: '")
                       + name + "'");
            return got;
        }
        // The hook must agree with the inner's true record length: the
        // inner must answer the exact count and refuse the next one
        // (the record_length contract), else capturing at the hook's
        // length would read past the inner's record.
        if (r->inner->f16(r->inner->context, block, layer, name, length)
                == nullptr
            || r->inner->f16(r->inner->context, block, layer, name,
                             length + 1)
                != nullptr) {
            r->err.add(std::string("recorder: length_for disagrees with the "
                                   "inner record length for '")
                       + name + "'");
            return got;
        }
    }
    const std::uint8_t* bytes = r->capture(
        Kind::kF16, block, layer, name, -1, 0, 0,
        static_cast<std::uint32_t>(length),
        reinterpret_cast<const std::uint8_t*>(got),
        static_cast<std::size_t>(length) * sizeof(float));
    return reinterpret_cast<const float*>(bytes);
}

inline const float* recorder_f32(void* context, int block, int layer,
                                 const char* name, int count) {
    Recorder* r = static_cast<Recorder*>(context);
    if (r == nullptr || r->inner == nullptr || r->inner->f32 == nullptr
        || count < -1) {
        return nullptr;
    }
    const float* got =
        r->inner->f32(r->inner->context, block, layer, name, count);
    if (got == nullptr) {
        return nullptr;
    }
    int length = count;
    if (count == -1) {
        length = r->length_for != nullptr ? r->length_for(name) : -1;
        if (length <= 0) {
            r->err.add(std::string("recorder: probe of unknown length: '")
                       + name + "'");
            return got;
        }
        if (r->inner->f32(r->inner->context, block, layer, name, length)
                == nullptr
            || r->inner->f32(r->inner->context, block, layer, name,
                             length + 1)
                != nullptr) {
            r->err.add(std::string("recorder: length_for disagrees with the "
                                   "inner record length for '")
                       + name + "'");
            return got;
        }
    }
    const std::uint8_t* bytes = r->capture(
        Kind::kF32, block, layer, name, -1, 0, 0,
        static_cast<std::uint32_t>(length),
        reinterpret_cast<const std::uint8_t*>(got),
        static_cast<std::size_t>(length) * sizeof(float));
    return reinterpret_cast<const float*>(bytes);
}

inline const std::uint8_t* recorder_ffwd(void* context, int block, int layer,
                                         int which, int rows, int cols) {
    Recorder* r = static_cast<Recorder*>(context);
    if (r == nullptr || r->inner == nullptr || r->inner->ffwd == nullptr
        || rows < 0 || cols < 0) {
        return nullptr;
    }
    const std::uint8_t* got =
        r->inner->ffwd(r->inner->context, block, layer, which, rows, cols);
    if (got == nullptr) {
        return nullptr;
    }
    return r->capture(Kind::kInt4, block, layer, "ffwd", which,
                      static_cast<std::uint32_t>(rows),
                      static_cast<std::uint32_t>(cols),
                      static_cast<std::uint32_t>(rows) * cols, got,
                      static_cast<std::size_t>(rows) * cols);
}

// The recorder does not serve the structured sites (swin/int4); the
// evaluator runs its documented e4m3 dataflow there.
inline void recorder_bind(EvalWeights& weights, Recorder& rec,
                          const EvalWeights& inner) {
    rec.inner = &inner;
    weights.context = &rec;
    weights.e4m3 = recorder_e4m3;
    weights.f16 = recorder_f16;
    weights.f32 = recorder_f32;
    weights.ffwd = recorder_ffwd;
    weights.swin = nullptr;
    weights.int4 = nullptr;
}

// ---- the file-backed provider ----------------------------------------------

struct FileStore {
    WeightsFile file;
    // Set by any callback whose request the file cannot serve (identity or
    // shape mismatch); the evaluator's null handling then surfaces its
    // usual error, and `error` explains why.
    WeightsFileError error;
};

// The provider mirrors the EvalWeights fetch contract EXACTLY: exact-count
// fetches are served only at the record's true length (the eval discovers
// lengths with record_length_f16/f32's count probing), and count == -1 is
// the wildcard response. A mismatch is rejected with the request named.
inline const std::uint8_t* file_e4m3(void* context, int block, int layer,
                                     const char* name, int rows, int cols) {
    FileStore* s = static_cast<FileStore*>(context);
    if (s == nullptr || rows < 0 || cols < 0) {
        return nullptr;
    }
    const TensorRecord* r = s->file.find(Kind::kE4m3, block, layer, name, -1);
    if (r == nullptr) {
        s->error.add(std::string("request: e4m3 '") + name + "' not in file");
        return nullptr;
    }
    if (static_cast<std::uint32_t>(rows) * cols != r->count) {
        // A count mismatch is the eval's normal length discovery
        // (record_length_* probes exact counts) or a request the
        // verification harness already named; it is not a runtime error.
        return nullptr;
    }
    return s->file.payload.data() + r->offset;
}

inline const float* file_scalar(void* context, int block, int layer,
                                const char* name, int count, Kind kind) {
    FileStore* s = static_cast<FileStore*>(context);
    if (s == nullptr || count < -1) {
        return nullptr;
    }
    const TensorRecord* r = s->file.find(kind, block, layer, name, -1);
    if (r == nullptr) {
        s->error.add(std::string("request: ") + kind_name(kind) + " '" + name
                     + "' not in file");
        return nullptr;
    }
    if (count >= 0 && static_cast<std::uint32_t>(count) != r->count) {
        // Length discovery, not an error (see file_e4m3's note).
        return nullptr;
    }
    return reinterpret_cast<const float*>(s->file.payload.data() + r->offset);
}

inline const float* file_f16(void* context, int block, int layer,
                             const char* name, int count) {
    return file_scalar(context, block, layer, name, count, Kind::kF16);
}

inline const float* file_f32(void* context, int block, int layer,
                             const char* name, int count) {
    return file_scalar(context, block, layer, name, count, Kind::kF32);
}

inline const std::uint8_t* file_ffwd(void* context, int block, int layer,
                                     int which, int rows, int cols) {
    FileStore* s = static_cast<FileStore*>(context);
    if (s == nullptr || rows < 0 || cols < 0) {
        return nullptr;
    }
    const TensorRecord* r = s->file.find(Kind::kInt4, block, layer, "ffwd",
                                         which);
    if (r == nullptr) {
        char line[160];
        std::snprintf(line, sizeof(line),
                      "request: int4 'ffwd' which %d not in file", which);
        s->error.add(line);
        return nullptr;
    }
    if (static_cast<std::uint32_t>(rows) * cols != r->count) {
        // Length discovery, not an error (see file_e4m3's note).
        return nullptr;
    }
    return s->file.payload.data() + r->offset;
}

inline void file_bind(EvalWeights& weights, FileStore& store) {
    weights.context = &store;
    weights.e4m3 = file_e4m3;
    weights.f16 = file_f16;
    weights.f32 = file_f32;
    weights.ffwd = file_ffwd;
    weights.swin = nullptr;
    weights.int4 = nullptr;
}

// ---- file I/O helpers ------------------------------------------------------

inline bool load_file(const char* path, WeightsFile& file,
                      WeightsFileError& err) {
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) {
        err.add(std::string("file: cannot open '") + path + "'");
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (size <= 0) {
        std::fclose(f);
        err.add(std::string("file: '") + path + "' is empty");
        return false;
    }
    std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
    const std::size_t got = std::fread(data.data(), 1, data.size(), f);
    std::fclose(f);
    if (got != data.size()) {
        err.add(std::string("file: short read of '") + path + "'");
        return false;
    }
    return read_container(data.data(), data.size(), file, err);
}

inline bool save_file(const char* path, const Recorder& rec,
                      WeightsFileError& err) {
    std::vector<std::uint8_t> bytes;
    // The recorder's payload is padded to 8 bytes per record; the record
    // offsets/lengths must be rewritten into the container metadata.
    std::vector<TensorRecord> records;
    std::vector<std::uint8_t> payload;
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
    if (!write_container(records, payload, bytes, err)) {
        return false;
    }
    std::FILE* f = std::fopen(path, "wb");
    if (f == nullptr) {
        err.add(std::string("file: cannot create '") + path + "'");
        return false;
    }
    const std::size_t wrote = std::fwrite(bytes.data(), 1, bytes.size(), f);
    std::fclose(f);
    if (wrote != bytes.size()) {
        err.add(std::string("file: short write of '") + path + "'");
        return false;
    }
    return true;
}

}  // namespace weightsfile
}  // namespace dlss5_nr
