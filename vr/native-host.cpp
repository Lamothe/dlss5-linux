// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Michael Lamothe
//
// The stable host boundary's implementation (vr/native-host.h): one
// synthetic-weight pipeline per handle over the temporal frame step
// (temporal_image_evaluation — reset, history feed, run, store, read-back,
// commit, counter). The weights are the deterministic synthetic DemoStore
// records: the transformation is arbitrary and the label says so.
#include <vector>

#include "native-host.h"
#include "native-demo-upscale.hpp"

#include <cstdlib>
#include <new>

#include "native-demo-support.hpp"
#include "native-eval.hpp"
#include "native-history.hpp"
#include "native-io.hpp"
#include "native-network.hpp"

// The opaque handle (the C header's forward declaration).
// A file-scope alias for the demonstration transform namespace.
namespace demoup = dlss5_nr::demoup;

struct dlss5_nr_handle {
    // The demonstration-transform mode (0 = off, else the upscale factor).
    int demo_scale = 0;
    int demo_kernel = DLSS5_NR_DEMO_LANCZOS3;  // dlss5_nr_demo_kernel
    bool has_plan = false;  // the synthetic path's plan (absent > 320x320)
    int width = 0;
    int height = 0;
    dlss5_nr::LayerSpec specs[dlss5_nr::kPlanMaxLayers];
    dlss5_nr::LayerValue outputs[dlss5_nr::kPlanMaxLayers];
    int skip_scratch[dlss5_nr::kPlanMaxLayers];
    int last_scratch[dlss5_nr::kPlanMaxLayers];
    int row_count = 0;
    dlss5_demo::DemoStore store;
    dlss5_nr::EvalWeights weights;
    dlss5_nr::EvalContext context;
    dlss5_nr::NetworkRun run;
    dlss5_nr::TemporalCarrier carrier;
    float* arena = nullptr;
    float* image_storage = nullptr;
    float* store_scratch = nullptr;
    float* carrier_map = nullptr;
    std::size_t image_floats = 0;
    std::size_t arena_floats = 0;
    std::size_t arena_high_water = 0;
};

namespace {

constexpr char kLabel[] = "synthetic weights \xe2\x80\x94 arbitrary transform";
// The synthetic network's floor: the small plan's padded extent at 64x72
// (the boundary's fixed arena covers it; the SYNTHETIC path refuses
// larger frames). The demonstration-transform mode needs no plan, so the
// create accepts frames up to kMaxDemoSide (half-window captures) and
// only builds the plan when the padded extent fits the network.
constexpr int kMaxPaddedSide = 320;
constexpr int kMaxDemoSide = 2048;

dlss5_nr_status to_abi(dlss5_nr::Status status) {
    return static_cast<dlss5_nr_status>(static_cast<int>(status));
}

}  // namespace

extern "C" {

const char* dlss5_nr_label(void) { return kLabel; }

const char* dlss5_nr_status_name(dlss5_nr_status status) {
    switch (status) {
        case DLSS5_NR_OK: return "ok";
        case DLSS5_NR_NULL_POINTER: return "null pointer";
        case DLSS5_NR_INVALID_DIMENSIONS: return "invalid dimensions";
        case DLSS5_NR_INVALID_STRIDE: return "invalid stride";
        case DLSS5_NR_INVALID_VALUES: return "invalid values";
        case DLSS5_NR_INVALID_CONTROLS: return "invalid controls";
        case DLSS5_NR_INVALID_BUFFER: return "invalid buffer";
    }
    return "unknown";
}

dlss5_nr_status dlss5_nr_create(int width, int height, int threads,
                                dlss5_nr_handle** out_handle) {
    using namespace dlss5_nr;
    if (out_handle == nullptr) {
        return DLSS5_NR_NULL_POINTER;
    }
    *out_handle = nullptr;
    if (width < 1 || height < 1 || width > kMaxDemoSide
        || height > kMaxDemoSide) {
        return DLSS5_NR_INVALID_DIMENSIONS;
    }
    if (threads < 0 || threads > 64) {
        return DLSS5_NR_INVALID_VALUES;
    }
    PlanPadding padding{};
    const bool fits_network =
        plan_padding(width, height, NetworkKind::kSmall, padding) == Status::kOk
        && padding.width <= kMaxPaddedSide && padding.height <= kMaxPaddedSide;
    if (!fits_network) {
        // Demo-only handle: no synthetic plan (the mode's evaluate refuses
        // it); the demonstration transform works at any accepted extent.
        padding.width = width;
        padding.height = height;
    }
    void* raw = std::calloc(1, sizeof(dlss5_nr_handle));
    if (raw == nullptr) {
        return DLSS5_NR_INVALID_BUFFER;
    }
    dlss5_nr_handle* handle = new (raw) dlss5_nr_handle();
    handle->width = width;
    handle->height = height;
    if (fits_network) {
        if (build_plan(width, height, NetworkKind::kSmall, handle->specs,
                       kPlanMaxLayers, handle->row_count)
            != Status::kOk) {
            handle->~dlss5_nr_handle();
            std::free(raw);
            return DLSS5_NR_INVALID_VALUES;
        }
        handle->has_plan = true;
    }
    handle->image_floats = 3u
        * static_cast<std::size_t>(padding.width)
        * static_cast<std::size_t>(padding.height);
    handle->arena_floats = fits_network ? std::size_t{1} << 27 : 0;
    handle->arena = fits_network
        ? static_cast<float*>(std::malloc(handle->arena_floats * sizeof(float)))
        : nullptr;
    handle->image_storage = static_cast<float*>(
        std::malloc(handle->image_floats * sizeof(float)));
    handle->store_scratch = static_cast<float*>(
        std::malloc(handle->image_floats * sizeof(float)));
    handle->carrier_map = static_cast<float*>(
        std::malloc(handle->image_floats * sizeof(float)));
    if (handle->image_storage == nullptr || handle->store_scratch == nullptr
        || handle->carrier_map == nullptr
        || (fits_network && handle->arena == nullptr)) {
        dlss5_nr_destroy(handle);
        return DLSS5_NR_INVALID_BUFFER;
    }
    handle->store.preset = dlss5_demo::DemoStore::Preset::kSeeded;
    handle->store.seed = 20261028u;
    dlss5_demo::demo_bind_weights(handle->weights, handle->store);
    handle->context.weights = &handle->weights;
    handle->context.scratch = handle->arena;
    handle->context.scratch_capacity = handle->arena_floats;
    if (fits_network) {
        if (eval_reset(handle->context) != Status::kOk
            || temporal_init(handle->carrier, handle->carrier_map,
                             handle->image_floats, padding.height, padding.width)
                != Status::kOk) {
            dlss5_nr_destroy(handle);
            return DLSS5_NR_INVALID_BUFFER;
        }
    }
    handle->run.outputs = handle->outputs;
    handle->run.skip_sources = handle->skip_scratch;
    handle->run.last_uses = handle->last_scratch;
    handle->run.layer = eval_layer;
    handle->run.release = eval_release;
    set_parallel_threads(threads == 0 ? 8 : threads);
    *out_handle = handle;
    return DLSS5_NR_OK;
}

dlss5_nr_status dlss5_nr_set_demo_kernel(dlss5_nr_handle* handle,
                                         dlss5_nr_demo_kernel kernel) {
    if (handle == nullptr) {
        return DLSS5_NR_NULL_POINTER;
    }
    if (kernel != DLSS5_NR_DEMO_LANCZOS3 &&
        kernel != DLSS5_NR_DEMO_BILINEAR &&
        kernel != DLSS5_NR_DEMO_NEURAL) {
        return DLSS5_NR_INVALID_VALUES;
    }
    if (kernel == DLSS5_NR_DEMO_NEURAL && !demoup::demo_model_loaded()) {
        // The neural demonstration needs its real weights loaded first.
        return DLSS5_NR_INVALID_VALUES;
    }
    handle->demo_kernel = static_cast<int>(kernel);
    return DLSS5_NR_OK;
}

dlss5_nr_status dlss5_nr_demo_model_load(const char* path) {
    if (path == nullptr) {
        return DLSS5_NR_NULL_POINTER;
    }
    if (!demoup::demo_model_load(path)) {
        fprintf(stderr, "error: demonstration model: %s\n",
                demoup::demo_model_error().c_str());
        return DLSS5_NR_INVALID_VALUES;
    }
    return DLSS5_NR_OK;
}

int dlss5_nr_demo_model_loaded(void) {
    return demoup::demo_model_loaded() ? 1 : 0;
}

const char* dlss5_nr_demo_model_label(void) {
    return demoup::demo_model_label();
}

dlss5_nr_status dlss5_nr_set_demo_transform(dlss5_nr_handle* handle,
                                            int scale) {
    if (handle == nullptr) {
        return DLSS5_NR_NULL_POINTER;
    }
    if (scale != 0 && (scale < 2 || scale > 4)) {
        return DLSS5_NR_INVALID_DIMENSIONS;
    }
    handle->demo_scale = scale;
    return DLSS5_NR_OK;
}

const char* dlss5_nr_label_for(const dlss5_nr_handle* handle) {
    if (handle != nullptr && handle->demo_scale != 0) {
        const auto k = handle->demo_kernel == DLSS5_NR_DEMO_NEURAL
                           ? demoup::DemoKernel::Neural
                           : (handle->demo_kernel == DLSS5_NR_DEMO_BILINEAR
                                  ? demoup::DemoKernel::Bilinear
                                  : demoup::DemoKernel::Lanczos3);
        return demoup::demo_upscale_label_for(k);
    }
    return dlss5_nr_label();
}

dlss5_nr_status dlss5_nr_evaluate(dlss5_nr_handle* handle,
                                  const unsigned char* rgb, int row_stride,
                                  unsigned char* out_rgb, int out_stride,
                                  int reset) {
    using namespace dlss5_nr;
    if (handle == nullptr || rgb == nullptr || out_rgb == nullptr) {
        return DLSS5_NR_NULL_POINTER;
    }
    if (handle->demo_scale != 0) {
        // The demonstration transform (NOT DLSS5): the low-resolution
        // capture in, scale*width x scale*height out. Every value runs
        // through the documented resampler (vr/native-demo-upscale.hpp);
        // the result must carry dlss5_nr_label_for(handle).
        const int scale = handle->demo_scale;
        const int ow = handle->width * scale;
        const int oh = handle->height * scale;
        if (row_stride < 3 * handle->width
            || out_stride < 3 * ow) {
            return DLSS5_NR_INVALID_STRIDE;
        }
        const std::size_t in_plane = static_cast<std::size_t>(handle->width)
            * handle->height;
        const std::size_t out_plane = static_cast<std::size_t>(ow) * oh;
        std::vector<float> in(3 * in_plane), out(3 * out_plane);
        for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < handle->height; ++y) {
                for (int x = 0; x < handle->width; ++x) {
                    in[c * in_plane
                       + static_cast<std::size_t>(y) * handle->width + x] =
                        static_cast<float>(
                            rgb[static_cast<std::size_t>(y) * row_stride
                                + 3 * x + c])
                        / 255.0f;
                }
            }
        }
        demoup::DemoUpscaleParams p;
        p.kernel = handle->demo_kernel == DLSS5_NR_DEMO_NEURAL
                       ? demoup::DemoKernel::Neural
                       : (handle->demo_kernel == DLSS5_NR_DEMO_BILINEAR
                              ? demoup::DemoKernel::Bilinear
                              : demoup::DemoKernel::Lanczos3);
        p.scale = scale;
        // The neural path uses the demo params' thread default (8); the
        // handle's evaluator parallelism belongs to the synthetic path.
        p.threads = 8;
        if (!demoup::demo_upscale(in.data(), handle->width, handle->height,
                                  out.data(), ow, oh, p)) {
            return DLSS5_NR_INVALID_VALUES;
        }
        for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < oh; ++y) {
                for (int x = 0; x < ow; ++x) {
                    float v = out[c * out_plane
                                  + static_cast<std::size_t>(y) * ow + x];
                    v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
                    out_rgb[static_cast<std::size_t>(y) * out_stride
                            + 3 * x + c] = static_cast<unsigned char>(
                        v * 255.0f + 0.5f);
                }
            }
        }
        return DLSS5_NR_OK;
    }
    const int minimum = 3 * handle->width;
    if (row_stride < minimum || out_stride < minimum) {
        return DLSS5_NR_INVALID_STRIDE;
    }
    if (!handle->has_plan) {
        // A demo-only handle (extent beyond the network's 320x320 floor)
        // has no synthetic plan: the synthetic path refuses it cleanly.
        return DLSS5_NR_INVALID_DIMENSIONS;
    }
    const Status status = temporal_image_evaluation(
        rgb, row_stride, handle->width, handle->height, out_rgb, out_stride,
        handle->image_storage, handle->image_floats, handle->specs,
        handle->row_count, handle->run, handle->context, handle->carrier,
        reset != 0, handle->store_scratch, handle->image_floats);
    if (status == Status::kOk) {
        const std::size_t used = eval_arena_used(handle->context);
        if (used > handle->arena_high_water) {
            handle->arena_high_water = used;
        }
    }
    return to_abi(status);
}

unsigned int dlss5_nr_counter(const dlss5_nr_handle* handle) {
    return handle == nullptr ? 0u : temporal_counter(handle->carrier);
}

dlss5_nr_status dlss5_nr_arena_high_water(const dlss5_nr_handle* handle,
                                          size_t* out_floats) {
    if (handle == nullptr || out_floats == nullptr) {
        return DLSS5_NR_NULL_POINTER;
    }
    *out_floats = handle->arena_high_water;
    return DLSS5_NR_OK;
}

void dlss5_nr_destroy(dlss5_nr_handle* handle) {
    if (handle == nullptr) {
        return;
    }
    std::free(handle->arena);
    std::free(handle->image_storage);
    std::free(handle->store_scratch);
    std::free(handle->carrier_map);
    handle->~dlss5_nr_handle();
    std::free(handle);
}

}  // extern "C"
