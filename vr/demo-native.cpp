// SPDX-License-Identifier: MIT
// Copyright (c) 2026 the dlss5-linux contributors
//
// Native Linux pipeline demo: host frame in -> image_evaluation (the complete
// typed network) -> host frame out, with deterministic synthetic weight
// records, a short temporal sequence with a mid-stream reset, and CPU wall
// timings + arena high-water reporting (the CPU baseline for the later GPU
// stage). Zero external assets: without --input a built-in checkerboard or
// gradient is generated. See README-demo.md for the exact commands.
//
// Honest scope: synthetic weights make the visual transformation arbitrary.
// This proves the complete pipeline runs on Linux — NOT DLSS quality and NOT
// vendor correspondence. No proprietary weights, captures, payloads or
// binaries are used or produced; no runtime/capture probes are performed.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

#include "native-demo-support.hpp"
#include "native-eval.hpp"
#include "native-graph-plan.hpp"
#include "native-history.hpp"
#include "native-io.hpp"

namespace {

using dlss5_nr::Status;

constexpr int kFrameW = 64;
constexpr int kFrameH = 72;
constexpr int kPaddedDim = 320;  // the small plan's padded extent at 64x72

struct Options {
    const char* input = nullptr;
    const char* out_dir = "demo-output";
    const char* preset = "seeded";
    const char* synthetic = "checker";
    unsigned seed = 20261028u;
    int frames = 4;
    int reset_at = 2;  // -1 = never; otherwise reset BEFORE this frame index
    bool profile = false;
};

void usage() {
    std::printf(
        "usage: demo-native [options]\n"
        "  --input PATH       host PPM (P6) image, %dx%d RGB (default: built-in)\n"
        "  --synthetic NAME   built-in image: checker | gradient\n"
        "  --out DIR          output directory (default: demo-output)\n"
        "  --preset NAME      weight preset: seeded | zero\n"
        "  --seed N           preset seed (default: 20261028)\n"
        "  --frames N         frames to evaluate (default: 4)\n"
        "  --reset-at K       temporal reset before frame K (default: 2; -1 = never)\n"
        "  --profile          print the per-kind wall-time breakdown of the run\n",
        kFrameW, kFrameH);
}

bool parse_int(const char* text, int& out) {
    char* end = nullptr;
    const long v = std::strtol(text, &end, 10);
    if (end == text || *end != '\0') {
        return false;
    }
    out = static_cast<int>(v);
    return true;
}

bool parse_uint(const char* text, unsigned& out) {
    char* end = nullptr;
    const unsigned long v = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0') {
        return false;
    }
    out = static_cast<unsigned>(v);
    return true;
}

bool write_file(const char* path, const std::uint8_t* data, std::size_t n) {
    std::FILE* f = std::fopen(path, "wb");
    if (f == nullptr) {
        return false;
    }
    const std::size_t written = std::fwrite(data, 1, n, f);
    const int closed = std::fclose(f);
    return written == n && closed == 0;
}

std::uint8_t* read_file(const char* path, std::size_t& length) {
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) {
        return nullptr;
    }
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (size <= 0 || size > (1 << 24)) {
        std::fclose(f);
        return nullptr;
    }
    std::uint8_t* data = static_cast<std::uint8_t*>(
        std::malloc(static_cast<std::size_t>(size)));
    if (data == nullptr) {
        std::fclose(f);
        return nullptr;
    }
    const std::size_t read =
        std::fread(data, 1, static_cast<std::size_t>(size), f);
    std::fclose(f);
    if (read != static_cast<std::size_t>(size)) {
        std::free(data);
        return nullptr;
    }
    length = read;
    return data;
}

// ---- the profiling seam ------------------------------------------------
// Wraps dlss5_nr::eval_layer at the NetworkRun dispatch boundary and
// accumulates per-kind wall time and row counts. Diagnostic only: the
// wrapper forwards byte-exactly and adds no arithmetic.
constexpr int kKindCount = 18;

const char* kind_name(int kind) {
    static const char* names[kKindCount] = {
        "PreBlock",     "Swin1H",       "Swin2H",       "Swin4H",
        "Swin8H",       "SplitFfwd",    "SplitFfwdProj", "SplitQKVAttn",
        "SplitProj",    "SplitProjPool", "SplitFinalHead", "VitFfnExpand",
        "VitFfnContract", "VitQKV",     "VitAttention",  "VitProjection",
        "DecInputUpsample", "PostBlock"};
    return kind >= 0 && kind < kKindCount ? names[kind] : "?";
}

struct ProfileStats {
    double ms[kKindCount] = {};
    long long rows[kKindCount] = {};
};

ProfileStats* g_profile = nullptr;

dlss5_nr::Status profile_layer(void* context, const dlss5_nr::LayerSpec& spec,
                               const dlss5_nr::LayerCall& call,
                               dlss5_nr::LayerValue& out) {
    const auto t0 = std::chrono::steady_clock::now();
    const dlss5_nr::Status st =
        dlss5_nr::eval_layer(context, spec, call, out);
    const auto t1 = std::chrono::steady_clock::now();
    const int kind = static_cast<int>(spec.kind);
    if (g_profile != nullptr && kind >= 0 && kind < kKindCount) {
        g_profile->ms[kind] +=
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        g_profile->rows[kind] += 1;
    }
    return st;
}

double frame_mean(const std::uint8_t* rgb, int width, int height) {
    const std::size_t n = 3u * static_cast<std::size_t>(width)
        * static_cast<std::size_t>(height);
    unsigned long long sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += rgb[i];
    }
    return static_cast<double>(sum) / static_cast<double>(n);
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        auto need = [&i, argc, argv](void) -> const char* {
            return i + 1 < argc ? argv[++i] : nullptr;
        };
        if (std::strcmp(a, "--input") == 0) {
            opt.input = need();
            if (opt.input == nullptr) {
                usage();
                return 2;
            }
        } else if (std::strcmp(a, "--out") == 0) {
            opt.out_dir = need();
            if (opt.out_dir == nullptr) {
                usage();
                return 2;
            }
        } else if (std::strcmp(a, "--preset") == 0) {
            opt.preset = need();
            if (opt.preset == nullptr) {
                usage();
                return 2;
            }
        } else if (std::strcmp(a, "--synthetic") == 0) {
            opt.synthetic = need();
            if (opt.synthetic == nullptr) {
                usage();
                return 2;
            }
        } else if (std::strcmp(a, "--seed") == 0) {
            const char* v = need();
            if (v == nullptr || !parse_uint(v, opt.seed)) {
                usage();
                return 2;
            }
        } else if (std::strcmp(a, "--frames") == 0) {
            const char* v = need();
            if (v == nullptr || !parse_int(v, opt.frames)) {
                usage();
                return 2;
            }
        } else if (std::strcmp(a, "--reset-at") == 0) {
            const char* v = need();
            if (v == nullptr || !parse_int(v, opt.reset_at)) {
                usage();
                return 2;
            }
        } else if (std::strcmp(a, "--profile") == 0) {
            opt.profile = true;
        } else if (std::strcmp(a, "--help") == 0) {
            usage();
            return 0;
        } else {
            usage();
            return 2;
        }
    }
    if (opt.input == nullptr && opt.synthetic == nullptr) {
        std::fprintf(stderr, "no --input and no --synthetic\n");
        return 2;
    }
    if (opt.frames < 1 || opt.frames > 256) {
        std::fprintf(stderr, "--frames must be 1..256\n");
        return 2;
    }

    // ---- the input frame (built-in or PPM) ------------------------------
    static std::uint8_t before[3 * kFrameW * kFrameH];
    static std::uint8_t after[3 * kFrameW * kFrameH];
    if (opt.input != nullptr) {
        std::size_t length = 0;
        std::uint8_t* file = read_file(opt.input, length);
        if (file == nullptr) {
            std::fprintf(stderr, "cannot read %s\n", opt.input);
            return 1;
        }
        int w = 0;
        int h = 0;
        const Status st = dlss5_demo::decode_ppm_p6(
            file, length, before, sizeof(before), w, h);
        std::free(file);
        if (st != Status::kOk || w != kFrameW || h != kFrameH) {
            std::fprintf(stderr, "%s: need a P6 PPM of exactly %dx%d\n",
                         opt.input, kFrameW, kFrameH);
            return 1;
        }
    } else if (std::strcmp(opt.synthetic, "checker") == 0) {
        const std::uint8_t a[3] = {224, 96, 32};
        const std::uint8_t b[3] = {32, 96, 224};
        dlss5_demo::fill_checker(before, kFrameW, kFrameH, 8, a, b);
    } else if (std::strcmp(opt.synthetic, "gradient") == 0) {
        dlss5_demo::fill_gradient(before, kFrameW, kFrameH);
    } else {
        std::fprintf(stderr, "unknown --synthetic %s\n", opt.synthetic);
        return 2;
    }

    // ---- the plan and the evaluator state --------------------------------
    static dlss5_nr::LayerSpec specs[dlss5_nr::kPlanMaxLayers];
    static dlss5_nr::LayerValue outputs[dlss5_nr::kPlanMaxLayers];
    static int skip_scratch[dlss5_nr::kPlanMaxLayers];
    static int last_scratch[dlss5_nr::kPlanMaxLayers];
    int count = 0;
    Status status = dlss5_nr::build_plan(
        kFrameW, kFrameH, dlss5_nr::NetworkKind::kSmall, specs,
        dlss5_nr::kPlanMaxLayers, count);
    if (status != Status::kOk) {
        std::fprintf(stderr, "build_plan failed: %d\n",
                     static_cast<int>(status));
        return 1;
    }
    dlss5_nr::PlanPadding padding;
    status = dlss5_nr::plan_padding(
        kFrameW, kFrameH, dlss5_nr::NetworkKind::kSmall, padding);
    if (status != Status::kOk || padding.width != kPaddedDim
        || padding.height != kPaddedDim) {
        std::fprintf(stderr, "plan_padding failed: %d (%dx%d)\n",
                     static_cast<int>(status), padding.width, padding.height);
        return 1;
    }

    static dlss5_demo::DemoStore store;
    if (std::strcmp(opt.preset, "zero") == 0) {
        store.preset = dlss5_demo::DemoStore::Preset::kZero;
    } else if (std::strcmp(opt.preset, "seeded") == 0) {
        store.preset = dlss5_demo::DemoStore::Preset::kSeeded;
    } else {
        std::fprintf(stderr, "unknown --preset %s\n", opt.preset);
        return 2;
    }
    store.seed = opt.seed;
    static dlss5_nr::EvalWeights weights;
    dlss5_demo::demo_bind_weights(weights, store);

    // The padded planar image map the runner pads the host into.
    const std::size_t image_floats = 3u
        * static_cast<std::size_t>(padding.width)
        * static_cast<std::size_t>(padding.height);
    float* image_storage = static_cast<float*>(
        std::malloc(image_floats * sizeof(float)));
    if (image_storage == nullptr) {
        std::fprintf(stderr, "image allocation failed\n");
        return 1;
    }

    // The evaluator arena (the tool owns the allocation).
    const std::size_t arena_floats = 1u << 26;  // 256 MiB, as the suites use
    float* arena =
        static_cast<float*>(std::malloc(arena_floats * sizeof(float)));
    if (arena == nullptr) {
        std::fprintf(stderr, "arena allocation failed\n");
        std::free(image_storage);
        return 1;
    }
    static dlss5_nr::EvalContext context;
    context.weights = &weights;
    context.scratch = arena;
    context.scratch_capacity = arena_floats;
    status = dlss5_nr::eval_reset(context);
    if (status != Status::kOk) {
        std::fprintf(stderr, "eval_reset failed: %d\n",
                     static_cast<int>(status));
        std::free(arena);
        std::free(image_storage);
        return 1;
    }

    // The temporal carrier over the plan's padded extent.
    const std::size_t temporal_floats =
        3u * static_cast<std::size_t>(padding.width)
        * static_cast<std::size_t>(padding.height);
    float* temporal_map = static_cast<float*>(
        std::malloc(temporal_floats * sizeof(float)));
    if (temporal_map == nullptr) {
        std::fprintf(stderr, "temporal allocation failed\n");
        std::free(arena);
        std::free(image_storage);
        return 1;
    }
    dlss5_nr::TemporalCarrier temporal{};
    // rows = the height axis (the carrier's orientation contract).
    status = dlss5_nr::temporal_init(temporal, temporal_map, temporal_floats,
                                     padding.height, padding.width);
    if (status != Status::kOk) {
        std::fprintf(stderr, "temporal_init failed: %d\n",
                     static_cast<int>(status));
        std::free(temporal_map);
        std::free(arena);
        std::free(image_storage);
        return 1;
    }

    std::printf(
        "dlss5 native demo: %dx%d frame, plan %d rows (small), "
        "padded extent %dx%d\n",
        kFrameW, kFrameH, count, padding.width, padding.height);
    std::printf(
        "preset: %s (seed %u) - synthetic weights: the visual transformation "
        "is arbitrary; this proves the pipeline runs, not DLSS quality\n",
        opt.preset, opt.seed);

    // ---- the frame sequence ---------------------------------------------
    dlss5_nr::NetworkRun run;
    run.outputs = outputs;
    run.skip_sources = skip_scratch;
    run.last_uses = last_scratch;
    static ProfileStats profile;
    g_profile = opt.profile ? &profile : nullptr;
    run.layer = opt.profile ? profile_layer : dlss5_nr::eval_layer;
    run.release = dlss5_nr::eval_release;

    double total_ms = 0.0;
    std::size_t high_water = 0;
    // The float-level temporal delta: max |out_N - out_{N-1}| over the
    // padded map at the FP16 store precision (before the 8-bit read-back,
    // which can hide a history influence smaller than 1/255).
    static float prev_padded[3 * kPaddedDim * kPaddedDim];
    bool have_prev = false;
    for (int frame = 0; frame < opt.frames; ++frame) {
        if (opt.reset_at >= 0 && frame == opt.reset_at) {
            status = dlss5_nr::temporal_reset(temporal);
            if (status != Status::kOk) {
                std::fprintf(stderr, "temporal_reset failed\n");
                std::free(temporal_map);
                std::free(arena);
                std::free(image_storage);
                return 1;
            }
            std::printf("frame %d: RESET (history cleared, counter -> 0)\n",
                        frame);
        }
        const bool has_history = dlss5_nr::temporal_has_history(temporal);
        const std::uint32_t counter = dlss5_nr::temporal_counter(temporal);
        // The arena rewinds per frame (the suites' per-run contract); the
        // high-water mark survives the rewind.
        status = dlss5_nr::eval_reset(context);
        if (status != Status::kOk) {
            std::fprintf(stderr, "frame %d: eval_reset failed: %d\n", frame,
                         static_cast<int>(status));
            std::free(temporal_map);
            std::free(arena);
            std::free(image_storage);
            return 1;
        }
        const auto t0 = std::chrono::steady_clock::now();
        status = dlss5_nr::image_evaluation(
            before, 3 * kFrameW, kFrameW, kFrameH, after, 3 * kFrameW,
            image_storage, image_floats, specs, count, run, context,
            has_history ? &temporal : nullptr);
        const auto t1 = std::chrono::steady_clock::now();
        if (status != Status::kOk) {
            std::fprintf(stderr, "frame %d: image_evaluation failed: %d\n",
                         frame, static_cast<int>(status));
            std::free(temporal_map);
            std::free(arena);
            std::free(image_storage);
            return 1;
        }
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        const std::size_t used = dlss5_nr::eval_arena_used(context);
        if (used > high_water) {
            high_water = used;
        }

        // The carrier maps the plan's padded extent: the commit source is
        // the run's padded planar [3, rows, cols] output converted to
        // interleaved RGB (pixel p, channel c: rgb[3p + c] = map[c*plane + p]).
        static float padded_rgb[3 * kPaddedDim * kPaddedDim];
        const dlss5_nr::FeatureBatchView& result_map =
            run.outputs[count - 1].main;
        const std::size_t plane = static_cast<std::size_t>(padding.width)
            * static_cast<std::size_t>(padding.height);
        for (std::size_t p = 0; p < plane; ++p) {
            for (int c = 0; c < 3; ++c) {
                padded_rgb[3 * p + c] =
                    result_map.data[static_cast<std::size_t>(c) * plane + p];
            }
        }
        // The FP16 output-texture store of the hardware route (the carrier
        // holds FP16 texture content bit-exactly).
        status = dlss5_demo::round_frame_to_f16(padded_rgb, 3 * plane);
        if (status != dlss5_nr::Status::kOk) {
            std::fprintf(stderr, "frame %d: f16 store failed: %d\n", frame,
                         static_cast<int>(status));
            std::free(temporal_map);
            std::free(arena);
            std::free(image_storage);
            return 1;
        }
        status = dlss5_nr::temporal_commit(
            temporal,
            dlss5_nr::FrameView{padded_rgb, padding.height, padding.width,
                                static_cast<std::ptrdiff_t>(3)
                                    * padding.width});
        if (status != Status::kOk) {
            std::fprintf(stderr, "frame %d: temporal_commit failed: %d\n",
                         frame, static_cast<int>(status));
            std::free(temporal_map);
            std::free(arena);
            std::free(image_storage);
            return 1;
        }
        double delta = -1.0;
        double delta_crop = -1.0;
        if (have_prev) {
            delta = 0.0;
            delta_crop = 0.0;
            for (int y = 0; y < padding.height; ++y) {
                for (int x = 0; x < padding.width; ++x) {
                    for (int c = 0; c < 3; ++c) {
                        const std::size_t i =
                            3u * (static_cast<std::size_t>(y) * padding.width
                                  + static_cast<std::size_t>(x))
                            + static_cast<std::size_t>(c);
                        const double d = static_cast<double>(padded_rgb[i])
                            - static_cast<double>(prev_padded[i]);
                        const double ad = d < 0.0 ? -d : d;
                        if (ad > delta) {
                            delta = ad;
                        }
                        // The read-back crop is the top-left width x height
                        // window (frame_to_u8).
                        if (y < kFrameH && x < kFrameW && ad > delta_crop) {
                            delta_crop = ad;
                        }
                    }
                }
            }
        }
        std::memcpy(prev_padded, padded_rgb, 3 * plane * sizeof(float));
        have_prev = true;
        std::printf(
            "frame %d: frames-since-reset=%u history=%s  %8.2f ms  "
            "mean=%6.2f  arena=%zu floats  float-delta=%s\n",
            frame, counter, has_history ? "yes" : "no ", ms,
            frame_mean(after, kFrameW, kFrameH), used,
            delta < 0.0 ? "n/a" : "");
        if (delta >= 0.0) {
            std::printf("           max |out_%d - out_%d| = %.6g float over "
                        "the padded map; %.6g over the visible crop "
                        "(pre-u8)\n",
                        frame, frame - 1, delta, delta_crop);
        }
    }
    std::printf(
        "total: %.2f ms for %d frames (mean %.2f ms/frame); "
        "arena high-water %zu floats (%.1f MiB)\n",
        total_ms, opt.frames, total_ms / opt.frames, high_water,
        static_cast<double>(high_water) * 4.0 / (1024.0 * 1024.0));

    // ---- the artefacts ---------------------------------------------------
    const std::string out_dir(opt.out_dir);
    std::error_code dir_error;
    std::filesystem::create_directories(out_dir, dir_error);
    if (dir_error) {
        std::fprintf(stderr, "cannot create %s\n", out_dir.c_str());
        std::free(temporal_map);
        std::free(arena);
        std::free(image_storage);
        return 1;
    }
    const std::string before_path = out_dir + "/before.ppm";
    const std::string after_path = out_dir + "/after.ppm";
    const std::string contact_path = out_dir + "/contact.ppm";
    static std::uint8_t ppm_buf[3 * (2 * kFrameW) * kFrameH + 64];
    std::size_t needed = 0;
    bool ok =
        dlss5_demo::encode_ppm_p6(ppm_buf, sizeof(ppm_buf), before, kFrameW,
                                  kFrameH, needed)
            == Status::kOk
        && write_file(before_path.c_str(), ppm_buf, needed);
    ok = ok
        && dlss5_demo::encode_ppm_p6(ppm_buf, sizeof(ppm_buf), after, kFrameW,
                                     kFrameH, needed)
               == Status::kOk
        && write_file(after_path.c_str(), ppm_buf, needed);
    static std::uint8_t contact[3 * (2 * kFrameW) * kFrameH];
    dlss5_demo::contact_sheet(contact, before, after, kFrameW, kFrameH);
    ok = ok
        && dlss5_demo::encode_ppm_p6(ppm_buf, sizeof(ppm_buf), contact,
                                     2 * kFrameW, kFrameH, needed)
               == Status::kOk
        && write_file(contact_path.c_str(), ppm_buf, needed);
    if (!ok) {
        std::fprintf(stderr, "artefact write failed\n");
        std::free(temporal_map);
        std::free(arena);
        std::free(image_storage);
        return 1;
    }
    if (opt.profile) {
        double total = 0.0;
        for (int k = 0; k < kKindCount; ++k) {
            total += profile.ms[k];
        }
        std::printf("per-kind breakdown (all frames, %s):\n",
                    opt.profile ? "timed at the dispatch seam" : "");
        for (int k = 0; k < kKindCount; ++k) {
            if (profile.rows[k] > 0) {
                std::printf("  %-17s %4lld rows  %9.2f ms  %6.2f%%\n",
                            kind_name(k), profile.rows[k], profile.ms[k],
                            total > 0.0 ? 100.0 * profile.ms[k] / total
                                        : 0.0);
            }
        }
        std::printf("  %-17s %4d rows  %9.2f ms  %6.2f%%\n", "TOTAL",
                    count * opt.frames, total, 100.0);
    }
    std::printf("artefacts:\n  %s\n  %s\n  %s\n", before_path.c_str(),
                after_path.c_str(), contact_path.c_str());
    std::free(temporal_map);
    std::free(arena);
    std::free(image_storage);
    return 0;
}
