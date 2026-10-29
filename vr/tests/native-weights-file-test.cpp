// native-weights-file-test.cpp — the data-free weights interface tests.
//
// Container round-trip, the structural and content rejection matrix with
// named errors, the compatibility verifier's mismatch classes, and the
// recorder <-> file-provider FETCH fidelity (the request contract the
// evaluator exercises). No network run here: the full-network
// byte-identity round-trip runs through the demo in
// vr/test-native-weights-file.sh.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "native-demo-support.hpp"
#include "native-weights-file.hpp"

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

// A deterministic fixture: two e4m3 matrices, one f16 scalar table, one
// f32 table and one int4 matrix, all content derived from the index.
void fixture(std::vector<dlss5_nr::weightsfile::TensorRecord>& records,
             std::vector<std::uint8_t>& payload) {
    using namespace dlss5_nr::weightsfile;
    records.clear();
    payload.clear();
    auto add = [&](const char* name, Kind kind, int block, int layer,
                   int which, std::uint32_t rows, std::uint32_t cols,
                   std::uint32_t count, std::size_t bytes) {
        TensorRecord r;
        std::memset(r.name, 0, sizeof(r.name));
        std::snprintf(r.name, sizeof(r.name), "%s", name);
        r.kind = kind;
        r.block = block;
        r.layer = layer;
        r.which = which;
        r.rows = rows;
        r.cols = cols;
        r.count = count;
        r.offset = payload.size();
        r.length = bytes;
        records.push_back(r);
        for (std::size_t i = 0; i < bytes; ++i) {
            payload.push_back(static_cast<std::uint8_t>((i * 7 + 13) & 0xFF));
        }
        while ((payload.size() % 8) != 0) {
            payload.push_back(0);
        }
    };
    add("w_qkv", Kind::kE4m3, 2, 0, -1, 8, 4, 32, 32);
    add("upsample_gain", Kind::kF16, -1, -1, -1, 0, 0, 16, 64);
    add("table_f32", Kind::kF32, 3, 1, -1, 0, 0, 8, 32);
    add("ffwd", Kind::kInt4, 1, 0, 2, 4, 4, 16, 16);
}

// Test helpers: load an NWF1 file into records + payload, and write raw
// bytes (the shape-edit fixture rebuilds a hash-valid container).
bool load_file_into_records(const char* path,
                            std::vector<dlss5_nr::weightsfile::TensorRecord>& recs,
                            std::vector<std::uint8_t>& payload,
                            dlss5_nr::weightsfile::WeightsFileError& err) {
    using namespace dlss5_nr::weightsfile;
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) {
        err.add("helper: cannot open");
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
    const std::size_t got = std::fread(data.data(), 1, data.size(), f);
    std::fclose(f);
    if (got != data.size()) {
        err.add("helper: short read");
        return false;
    }
    WeightsFile file;
    if (!read_container(data.data(), data.size(), file, err)) {
        return false;
    }
    recs = file.records;
    payload = file.payload;
    return true;
}

bool save_bytes(const char* path, const std::vector<std::uint8_t>& bytes,
                dlss5_nr::weightsfile::WeightsFileError& err) {
    std::FILE* f = std::fopen(path, "wb");
    if (f == nullptr) {
        err.add("helper: cannot create");
        return false;
    }
    const std::size_t wrote = std::fwrite(bytes.data(), 1, bytes.size(), f);
    std::fclose(f);
    if (wrote != bytes.size()) {
        err.add("helper: short write");
        return false;
    }
    return true;
}

// f16-grid fixture content: rewrite the f16 table with binary16-exact
// floats so the content check passes.
void make_f16_grid(std::vector<std::uint8_t>& payload,
                   const std::vector<dlss5_nr::weightsfile::TensorRecord>& recs,
                   bool off_grid) {
    using namespace dlss5_nr::weightsfile;
    for (const auto& r : recs) {
        if (r.kind != Kind::kF16) {
            continue;
        }
        for (std::uint32_t i = 0; i < r.count; ++i) {
            float v = dlss5_nr::f16(0.25f * static_cast<float>(i % 7) + 0.5f);
            if (off_grid && i == 3) {
                v = 0.123456789f;  // not on the binary16 grid
            }
            std::memcpy(payload.data() + r.offset + i * 4, &v, 4);
        }
    }
}

}  // namespace

int main() {
    using namespace dlss5_nr;
    using namespace dlss5_nr::weightsfile;

    // ---- container round-trip ------------------------------------------
    std::vector<TensorRecord> recs;
    std::vector<std::uint8_t> payload;
    fixture(recs, payload);
    make_f16_grid(payload, recs, false);
    std::vector<std::uint8_t> bytes;
    WeightsFileError err;
    CHECK_(write_container(recs, payload, bytes, err));
    WeightsFile file;
    WeightsFileError rerr;
    CHECK_(read_container(bytes.data(), bytes.size(), file, rerr));
    CHECK_(rerr.ok());
    CHECK_(file.records.size() == recs.size());
    CHECK_(file.payload.size() == payload.size());
    {
        bool same = true;
        for (std::size_t i = 0; i < recs.size(); ++i) {
            same = same && same_identity(file.records[i], recs[i])
                && same_shape(file.records[i], recs[i]);
        }
        CHECK_(same);
        CHECK_(std::memcmp(file.payload.data(), payload.data(),
                           payload.size())
               == 0);
    }

    // ---- structural rejection matrix ------------------------------------
    {
        // Bad magic.
        std::vector<std::uint8_t> bad = bytes;
        bad[0] ^= 0xFF;
        WeightsFile f2;
        WeightsFileError e2;
        CHECK_(!read_container(bad.data(), bad.size(), f2, e2));
        CHECK_(contains(e2, "bad magic"));
    }
    {
        // Unsupported version.
        std::vector<std::uint8_t> bad = bytes;
        bad[8] = 9;
        WeightsFile f2;
        WeightsFileError e2;
        CHECK_(!read_container(bad.data(), bad.size(), f2, e2));
        CHECK_(contains(e2, "unsupported version"));
    }
    {
        // Truncated file (size mismatch).
        WeightsFile f2;
        WeightsFileError e2;
        CHECK_(!read_container(bytes.data(), bytes.size() - 3, f2, e2));
        CHECK_(contains(e2, "size mismatch"));
    }
    {
        // Payload hash mismatch (corrupt content).
        std::vector<std::uint8_t> bad = bytes;
        bad[bad.size() - 1] ^= 0x01;
        WeightsFile f2;
        WeightsFileError e2;
        CHECK_(!read_container(bad.data(), bad.size(), f2, e2));
        CHECK_(contains(e2, "hash mismatch"));
    }
    {
        // Out-of-bounds payload range.
        std::vector<std::uint8_t> bad = bytes;
        TensorRecord rebuilt;
        std::vector<TensorRecord> recs2 = recs;
        recs2[0].length = 1ull << 40;
        WeightsFileError werr;
        std::vector<std::uint8_t> bad_bytes;
        write_container(recs2, payload, bad_bytes, werr);
        WeightsFile f2;
        WeightsFileError e2;
        CHECK_(!read_container(bad_bytes.data(), bad_bytes.size(), f2, e2));
        CHECK_(contains(e2, "outside the"));
    }
    {
        // Duplicate record.
        std::vector<TensorRecord> recs2 = recs;
        recs2.push_back(recs[0]);  // same identity AND payload range
        WeightsFileError werr;
        std::vector<std::uint8_t> bad_bytes;
        write_container(recs2, payload, bad_bytes, werr);
        WeightsFile f2;
        WeightsFileError e2;
        CHECK_(!read_container(bad_bytes.data(), bad_bytes.size(), f2, e2));
        CHECK_(contains(e2, "duplicate"));
    }
    {
        // Length/shape mismatch.
        std::vector<TensorRecord> recs2 = recs;
        recs2[0].count += 1;
        WeightsFileError werr;
        std::vector<std::uint8_t> bad_bytes;
        write_container(recs2, payload, bad_bytes, werr);
        WeightsFile f2;
        WeightsFileError e2;
        CHECK_(!read_container(bad_bytes.data(), bad_bytes.size(), f2, e2));
        // The matrix count-consistency class ("does not match its ...").
        CHECK_(contains(e2, "does not match its"));
    }

    // ---- the remaining structural classes (hash-valid fixtures) --------
    {
        // Unaligned payload offset.
        std::vector<TensorRecord> recs2 = recs;
        recs2[0].offset += 4;
        std::vector<std::uint8_t> b2;
        WeightsFileError werr;
        write_container(recs2, payload, b2, werr);
        WeightsFile f2;
        WeightsFileError e2;
        CHECK_(!read_container(b2.data(), b2.size(), f2, e2));
        CHECK_(contains(e2, "not\n8-byte aligned")
               || contains(e2, "8-byte aligned"));
    }
    {
        // Unknown tensor kind.
        std::vector<TensorRecord> recs2 = recs;
        recs2[0].kind = static_cast<Kind>(77);
        std::vector<std::uint8_t> b2;
        WeightsFileError werr;
        write_container(recs2, payload, b2, werr);
        WeightsFile f2;
        WeightsFileError e2;
        CHECK_(!read_container(b2.data(), b2.size(), f2, e2));
        CHECK_(contains(e2, "unknown kind"));
    }
    {
        // A cleared hash flag cannot switch verification off.
        std::vector<std::uint8_t> bad = bytes;
        bad[12] = 0;  // flags = 0 (bit 0 cleared)
        WeightsFile f2;
        WeightsFileError e2;
        CHECK_(!read_container(bad.data(), bad.size(), f2, e2));
        CHECK_(contains(e2, "hash not present"));
    }
    {
        // Too small for the header.
        WeightsFile f2;
        WeightsFileError e2;
        CHECK_(!read_container(bytes.data(), 8, f2, e2));
        CHECK_(contains(e2, "too small"));
    }

    // ---- content verification ------------------------------------------
    {
        std::vector<TensorRecord> recs2 = recs;
        std::vector<std::uint8_t> pay2 = payload;
        make_f16_grid(pay2, recs2, true);  // one off-grid value
        std::vector<std::uint8_t> b2;
        WeightsFileError werr;
        write_container(recs2, pay2, b2, werr);
        WeightsFile f2;
        WeightsFileError e2;
        CHECK_(read_container(b2.data(), b2.size(), f2, e2));
        WeightsFileError c2;
        verify_content(f2, c2);
        CHECK_(!c2.ok());
        CHECK_(contains(c2, "binary16 grid"));
    }
    {
        std::vector<TensorRecord> recs2 = recs;
        std::vector<std::uint8_t> pay2 = payload;
        make_f16_grid(pay2, recs2, false);
        for (const auto& r : recs2) {
            if (r.kind == Kind::kF32) {
                const float nan_v = std::nanf("");
                std::memcpy(pay2.data() + r.offset + 4, &nan_v, 4);
            }
        }
        std::vector<std::uint8_t> b2;
        WeightsFileError werr;
        write_container(recs2, pay2, b2, werr);
        WeightsFile f2;
        WeightsFileError e2;
        CHECK_(read_container(b2.data(), b2.size(), f2, e2));
        WeightsFileError c2;
        verify_content(f2, c2);
        CHECK_(!c2.ok());
        CHECK_(contains(c2, "non-finite float32"));
    }

    // ---- compatibility verification ------------------------------------
    {
        Manifest manifest = recs;  // the "plan requests exactly the fixture"
        WeightsFileError v2;
        CHECK_(verify_compatibility(file, manifest, v2));
        CHECK_(v2.ok());
    }
    {
        Manifest manifest = recs;
        manifest.push_back(recs[0]);
        std::snprintf(manifest.back().name,
                      sizeof(manifest.back().name), "not_in_file");
        WeightsFileError v2;
        CHECK_(!verify_compatibility(file, manifest, v2));
        CHECK_(contains(v2, "missing: 'not_in_file'"));
    }
    {
        Manifest manifest = recs;
        manifest.pop_back();  // the file has an unexpected record now
        WeightsFileError v2;
        CHECK_(!verify_compatibility(file, manifest, v2));
        CHECK_(contains(v2, "unexpected:"));
    }
    {
        Manifest manifest = recs;
        manifest[0].rows += 1;
        manifest[0].count += 1;
        WeightsFileError v2;
        CHECK_(!verify_compatibility(file, manifest, v2));
        CHECK_(contains(v2, "shape:"));
    }

    // ---- recorder <-> file-provider fetch fidelity ----------------------
    {
        dlss5_demo::DemoStore store;
        EvalWeights inner;
        dlss5_demo::demo_bind_weights(inner, store);
        Recorder rec;
        EvalWeights recorded;
        recorder_bind(recorded, rec, inner);

        // Exercise every kind through the recording layer.
        const std::uint8_t* c1 = recorded.e4m3(recorded.context, 2, 0,
                                               "w_qkv", 8, 4);
        const float* t16 = recorded.f16(recorded.context, -1, -1,
                                        "upsample_gain", 16);
        const float* t32 = recorded.f32(recorded.context, 3, 1, "table_f32",
                                        8);
        const std::uint8_t* c4 = recorded.ffwd(recorded.context, 1, 0, 2, 4,
                                               4);
        CHECK_(c1 != nullptr && t16 != nullptr && t32 != nullptr
               && c4 != nullptr);
        CHECK_(rec.err.ok());
        CHECK_(rec.manifest.size() == 4);

        // Save, load, rebind: the file provider must serve IDENTICAL bytes.
        const char* path = "/tmp/native-weights-file-test.nwf";
        WeightsFileError serr;
        CHECK_(save_file(path, rec, serr));
        WeightsFile loaded;
        WeightsFileError lerr;
        CHECK_(load_file(path, loaded, lerr));
        CHECK_(lerr.ok());
        Manifest manifest = rec.manifest;
        WeightsFileError verr;
        CHECK_(verify_compatibility(loaded, manifest, verr));

        FileStore fs;
        fs.file = loaded;
        EvalWeights served;
        file_bind(served, fs);
        const std::uint8_t* c1b =
            served.e4m3(served.context, 2, 0, "w_qkv", 8, 4);
        const float* t16b =
            served.f16(served.context, -1, -1, "upsample_gain", 16);
        const float* t32b = served.f32(served.context, 3, 1, "table_f32", 8);
        const std::uint8_t* c4b = served.ffwd(served.context, 1, 0, 2, 4, 4);
        CHECK_(c1b != nullptr && t16b != nullptr && t32b != nullptr
               && c4b != nullptr);
        CHECK_(fs.error.ok());
        CHECK_(std::memcmp(c1, c1b, 32) == 0);
        CHECK_(std::memcmp(t16, t16b, 16 * sizeof(float)) == 0);
        CHECK_(std::memcmp(t32, t32b, 8 * sizeof(float)) == 0);
        CHECK_(std::memcmp(c4, c4b, 16) == 0);

        // Shape rejections are silent (length discovery); identity
        // misses are named (a real incompatibility).
        CHECK_(served.e4m3(served.context, 2, 0, "w_qkv", 9, 4) == nullptr);
        CHECK_(fs.error.ok());
        CHECK_(served.e4m3(served.context, 2, 0, "no_such", 8, 4) == nullptr);
        CHECK_(contains(fs.error, "not in file"));
        CHECK_(served.ffwd(served.context, 1, 0, 3, 4, 4) == nullptr);
        CHECK_(contains(fs.error, "which 3 not in file"));
        CHECK_(served.f32(served.context, 3, 1, "table_f32", 9) == nullptr);
        // Only the two identity misses were recorded (the shape
        // rejections above were silent).
        CHECK_(fs.error.problems.size() == 2);

        // Repeat-fetch fidelity inside the recorder (the determinism
        // contract): same identity again -> identical bytes, no error.
        const std::uint8_t* c1c = recorded.e4m3(recorded.context, 2, 0,
                                                "w_qkv", 8, 4);
        CHECK_(c1c == c1);
        CHECK_(rec.err.ok());

        std::remove(path);
    }

    // ---- the extension contract (probe then full fetch) -----------------
    {
        dlss5_demo::DemoStore store;
        EvalWeights inner;
        dlss5_demo::demo_bind_weights(inner, store);
        Recorder rec;
        rec.length_for = [](const char* name) -> int {
            int length = 0;
            return dlss5_demo::demo_detail::wildcard_length(name, length) ? length : -1;
        };
        EvalWeights recorded;
        recorder_bind(recorded, rec, inner);
        // Probe short, then fetch long at the same identity: the record
        // must grow and keep the first bytes (the DemoStore contract).
        const float* probe =
            recorded.f16(recorded.context, -1, -1, "upsample_gain", -1);
        CHECK_(probe != nullptr);
        CHECK_(rec.manifest.size() == 1);  // captured at the plan length
        const float* full =
            recorded.f16(recorded.context, -1, -1, "upsample_gain", 16);
        CHECK_(full == probe);
        CHECK_(rec.err.ok());
        CHECK_(rec.manifest.size() == 1);
        CHECK_(rec.manifest[0].count == 16);

        // The file provider mirrors the fetch contract EXACTLY: the
        // wildcard (count == -1) and the true length are served; any other
        // exact count is rejected (the eval probes counts to discover
        // lengths - record_length_f16 in native-eval.hpp).
        const char* path = "/tmp/native-weights-file-ext.nwf";
        WeightsFileError serr;
        CHECK_(save_file(path, rec, serr));
        WeightsFile loaded;
        WeightsFileError lerr;
        CHECK_(load_file(path, loaded, lerr));
        FileStore fs;
        fs.file = loaded;
        EvalWeights served;
        file_bind(served, fs);
        CHECK_(served.f16(served.context, -1, -1, "upsample_gain", -1)
               != nullptr);
        CHECK_(served.f16(served.context, -1, -1, "upsample_gain", 16)
               != nullptr);
        CHECK_(fs.error.ok());
        // Count rejections are silent (length discovery); the
        // verification harness names shape mismatches.
        CHECK_(served.f16(served.context, -1, -1, "upsample_gain", 8)
               == nullptr);
        CHECK_(served.f16(served.context, -1, -1, "upsample_gain", 32)
               == nullptr);
        CHECK_(fs.error.ok());
        std::remove(path);
    }

    // ---- a genuinely shape-edited candidate, named by the harness ------
    {
        dlss5_demo::DemoStore store;
        EvalWeights inner;
        dlss5_demo::demo_bind_weights(inner, store);
        Recorder rec;
        EvalWeights recorded;
        recorder_bind(recorded, rec, inner);
        CHECK_(recorded.e4m3(recorded.context, 2, 0, "w_qkv", 8, 4)
               != nullptr);
        const char* good = "/tmp/native-weights-file-shape-good.nwf";
        const char* bad = "/tmp/native-weights-file-shape-bad.nwf";
        WeightsFileError serr;
        CHECK_(save_file(good, rec, serr));
        // A hash-valid container with one tensor's shape edited.
        std::vector<TensorRecord> recs2;
        std::vector<std::uint8_t> pay2;
        const bool loaded_ok =
            load_file_into_records(good, recs2, pay2, serr);
        CHECK_(loaded_ok);
        if (loaded_ok && !recs2.empty()) {
            // One more row (4 elements): structurally valid, but the
            // manifest's shape differs - the harness must name it. The
            // payload grows to match.
            recs2[0].rows += 1;
            recs2[0].count += 4;
            recs2[0].length += 4;
            for (int i = 0; i < 4; ++i) {
                pay2.push_back(0x5A);
            }
        }
        std::vector<std::uint8_t> b2;
        CHECK_(write_container(recs2, pay2, b2, serr));
        CHECK_(save_bytes(bad, b2, serr));
        WeightsFile cand;
        WeightsFileError cerr;
        CHECK_(load_file(bad, cand, cerr));
        Manifest manifest = rec.manifest;
        WeightsFileError verr;
        CHECK_(!verify_compatibility(cand, manifest, verr));
        CHECK_(contains(verr, "w_qkv"));
        std::remove(good);
        std::remove(bad);
    }

    // ---- the extension path (a growing request at one identity) --------
    {
        // A custom inner that serves a record at 8 and then 16 elements
        // with the first bytes preserved (the DemoStore extension
        // contract). The recorder must grow the record and keep the
        // prefix; the file must then serve the full length exactly.
        struct Growing {
            float data[16];
        };
        static Growing g;
        for (int i = 0; i < 16; ++i) {
            g.data[i] = dlss5_nr::f16(0.5f + 0.25f * static_cast<float>(i));
        }
        auto grow_f16 = [](void* context, int, int, const char* name,
                           int count) -> const float* {
            Growing* gg = static_cast<Growing*>(context);
            if (std::strcmp(name, "grow") != 0) {
                return nullptr;
            }
            if (count == -1) {
                return gg->data;
            }
            return (count == 8 || count == 16) ? gg->data : nullptr;
        };
        EvalWeights inner;
        inner.context = &g;
        inner.f16 = grow_f16;
        Recorder rec;
        rec.length_for = [](const char*) -> int { return -1; };
        EvalWeights recorded;
        recorder_bind(recorded, rec, inner);
        // The probe passes through (unknown length) with an error; the
        // exact fetches capture at 8 and extend to 16.
        CHECK_(recorded.f16(recorded.context, -1, -1, "grow", -1) != nullptr);
        CHECK_(recorded.f16(recorded.context, -1, -1, "grow", 8) != nullptr);
        const float* full =
            recorded.f16(recorded.context, -1, -1, "grow", 16);
        CHECK_(full != nullptr);
        CHECK_(rec.manifest.size() == 1);
        CHECK_(rec.manifest[0].count == 16);
        bool prefix_ok = true;
        for (int i = 0; i < 8; ++i) {
            prefix_ok = prefix_ok && (full[i] == g.data[i]);
        }
        CHECK_(prefix_ok);
        // The probe error is present (unknown length), the extension is
        // silent.
        CHECK_(contains(rec.err, "unknown length"));

        const char* path = "/tmp/native-weights-file-grow.nwf";
        WeightsFileError serr;
        CHECK_(save_file(path, rec, serr));
        WeightsFile loaded;
        WeightsFileError lerr;
        CHECK_(load_file(path, loaded, lerr));
        CHECK_(loaded.records.size() == 1);
        CHECK_(loaded.records[0].count == 16);
        FileStore fs;
        fs.file = loaded;
        EvalWeights served;
        file_bind(served, fs);
        CHECK_(served.f16(served.context, -1, -1, "grow", 16) != nullptr);
        CHECK_(served.f16(served.context, -1, -1, "grow", 8) == nullptr);
        CHECK_(fs.error.ok());
        std::remove(path);
    }

    std::printf("native-weights-file: %d checks passed\n", g_checks);
    if (g_failures != 0) {
        std::printf("native-weights-file: %d check(s) FAILED\n", g_failures);
        return 1;
    }
    std::printf("native-weights-file: OK\n");
    return 0;
}
