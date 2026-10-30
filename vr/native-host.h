// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Michael Lamothe
//
// The stable host boundary for desktop renderers (the Centre Stage
// experiment's seam): a C ABI over the native evaluator. The host renders
// its scene at a LOW resolution, hands the RGB bytes to the boundary and
// presents the upscaled result. Synthetic weights: the transformation is
// ARBITRARY — `dlss5_nr_label()` returns the string a host must record
// alongside any presented output ("synthetic weights — arbitrary
// transform"). No vendor code, no proprietary payloads, no runtime or
// capture probes; CPU-only (the evaluator's kernels are the same ones the
// demo and the GPU seam use).
#ifndef DLSS5_NR_NATIVE_HOST_H
#define DLSS5_NR_NATIVE_HOST_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// The Status enum's exact values (native-pre-post.hpp): the ABI is stable
// and must not track internal renumbering silently.
typedef enum dlss5_nr_status {
    DLSS5_NR_OK = 0,
    DLSS5_NR_NULL_POINTER = 1,
    DLSS5_NR_INVALID_DIMENSIONS = 2,
    DLSS5_NR_INVALID_STRIDE = 3,
    DLSS5_NR_INVALID_VALUES = 4,
    DLSS5_NR_INVALID_CONTROLS = 5,
    DLSS5_NR_INVALID_BUFFER = 6
} dlss5_nr_status;

// The demonstration kernel: the classical Lanczos3 scaler, the bilinear
// baseline (the A/B comparison floor, at the SAME capture geometry), or
// the neural demonstration transform.
typedef enum dlss5_nr_demo_kernel {
    DLSS5_NR_DEMO_LANCZOS3 = 1,
    DLSS5_NR_DEMO_BILINEAR = 2,
    /* The neural demonstration transform (real open weights: EDSR-base
       x2, Apache-2.0, or Real-ESRGAN x2, BSD-3): requires a model loaded
       through dlss5_nr_demo_model_load first. NOT DLSS5 - a
       demonstration. */
    DLSS5_NR_DEMO_NEURAL = 3
} dlss5_nr_demo_kernel;

typedef struct dlss5_nr_handle dlss5_nr_handle;

// The label a host must record alongside any presented output (UTF-8).
// On-screen display is the host's UI choice; the label must not be lost
// from the host's records (reports, logs).
const char* dlss5_nr_label(void);
// The status code's name (a stable string for logs/overlays).
const char* dlss5_nr_status_name(dlss5_nr_status status);

// Create a pipeline for width x height RGB frames (8-bit, 3 bytes per
// pixel). `threads` selects the evaluator's parallelism (1..64; 0 = 8).
// The internal working extent comes from the plan's padding; frames whose
// padded extent exceeds the network's floor (320x320) serve the
// demonstration transform only (accepted up to 2048x2048): the synthetic
// evaluate refuses them with DLSS5_NR_INVALID_DIMENSIONS (the documented
// bound of this boundary's fixed arena). On success `*out_handle` owns the pipeline; on
// any failure it is null and nothing was allocated.
dlss5_nr_status dlss5_nr_create(int width, int height, int threads,
                                dlss5_nr_handle** out_handle);

// One frame: `rgb` (row_stride bytes per row, at least 3*width) in,
// `out_rgb` (out_stride bytes per row, at least 3*width) out at the same
// extent (the host presents the upscaled result itself). A non-zero
// `reset` runs DLSSNR.Reset first (history = 0, counter = 0). The output
// buffer is untouched unless the call returns DLSS5_NR_OK.
dlss5_nr_status dlss5_nr_evaluate(dlss5_nr_handle* handle,
                                  const unsigned char* rgb, int row_stride,
                                  unsigned char* out_rgb, int out_stride,
                                  int reset);

// The demonstration-transform mode (NOT DLSS5): a real, openly
// documented upscaling transform (Lanczos3 + documented parameters -
// vr/native-demo-upscale.hpp) runs instead of the synthetic network, so a
// host can present a VISIBLE demonstration result through this same ABI.
// `scale` is the integer upscale factor (2..4); 0 restores the synthetic
// pipeline. In this mode `dlss5_nr_evaluate` writes scale*width x
// scale*height output (out_stride must cover 3*scale*width) and the input
// is the low-resolution capture. The label `dlss5_nr_label_for(handle)`
// returns must be recorded alongside the presented output (on-screen
// display is the host's UI choice); the demonstration mode's labels are
// per path:
//   neural:     "not DLSS5 - demonstration transform with real weights
//                (<model>, <license>)"  (the phrase "real weights" is
//                reserved for the neural path)
//   classical:  "not DLSS5 - classical demonstration scaler (Lanczos3)"
//               "not DLSS5 - classical demonstration scaler (bilinear baseline)"
// The demonstration kernels (A/B): the neural model (when real weights are
// loaded) or the classical scalers at the SAME capture geometry. Stored
// even while the demo mode is off and applied when the mode is next
// enabled.
dlss5_nr_status dlss5_nr_set_demo_kernel(dlss5_nr_handle* handle,
                                         dlss5_nr_demo_kernel kernel);

/* Load the demonstration model (an openly licensed weights file -
   safetensors (Real-ESRGAN x2 / RRDBNet, BSD-3) or the torch-save
   EDSR-base x2 checkpoint (Apache-2.0); the container is detected),
   Real-ESRGAN x2 / RRDBNet). Process-global; loaded once. Returns
   DLSS5_NR_OK or DLSS5_NR_INVALID_VALUES (the error text is named in
   the log). DLSS5_NR_DEMO_NEURAL requires this to have succeeded. */
dlss5_nr_status dlss5_nr_demo_model_load(const char* path);
int dlss5_nr_demo_model_loaded(void);

/* The loaded demonstration model's honest label ("not DLSS5 - ..."), or
   the no-model placeholder. Always in-band-safe text. */
const char* dlss5_nr_demo_model_label(void);

dlss5_nr_status dlss5_nr_set_demo_transform(dlss5_nr_handle* handle,
                                            int scale);
// The label for THIS handle's current mode (the synthetic pipeline's
// label, or the demonstration transform's label - mandatory to record).
const char* dlss5_nr_label_for(const dlss5_nr_handle* handle);

// The frames-since-reset counter (the capture's schedule). The
// demonstration-transform mode is stateless: the counter stays 0 and the
// `reset` argument has no effect while the mode is on (documented).
unsigned int dlss5_nr_counter(const dlss5_nr_handle* handle);
// The evaluator arena's high-water mark in floats (the host's memory
// accounting; constant across frames).
dlss5_nr_status dlss5_nr_arena_high_water(const dlss5_nr_handle* handle,
                                          size_t* out_floats);

void dlss5_nr_destroy(dlss5_nr_handle* handle);

#ifdef __cplusplus
}
#endif

#endif  // DLSS5_NR_NATIVE_HOST_H
