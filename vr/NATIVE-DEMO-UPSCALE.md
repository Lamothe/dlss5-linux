# NATIVE-DEMO-UPSCALE — the demonstration transform (NOT DLSS5)

Status: complete (30/10/2026, scope-pivot phase, item 2).
Tests: `vr/test-native-demo-upscale.sh` (the quality family), the mock
host family (`vr/test-native-host.sh`, 75 checks incl. the mode).

## What this is — and what it is NOT

A real, openly documented upscaling transform with real weights —
Lanczos3 resampling (the analytic windowed-sinc kernel, a = 3) plus an
optional documented unsharp pass — run behind the SAME C ABI / capture /
present seam that the synthetic network uses, so a host's F9 produces a
**visibly improved** presented frame end to end. The parameters
("weights") are the published mathematical definitions
(`vr/native-demo-upscale.hpp`'s `DemoUpscaleParams`); openly licensed by
construction and independent of any vendor.

**This is NOT DLSS5 and makes no DLSS5 claim.** Every artefact, UI
label and report carries:

    not DLSS5 - demonstration transform with real weights

It proves the PLATFORM (loader → evaluator seam → C ABI → capture →
present) end to end with a real transform. It must never be conflated
with DLSS5 quality or vendor parity.

## The measured quality claim (honest, reproducible)

`vr/tests/native-demo-upscale-test.cpp` generates a deterministic
320x288 source (mixed frequencies, gradients, hard edges, a chart
region), box-downscales to a "capture", and compares the transform
against the source AND a bilinear baseline (PSNR and the standard Wang
SSIM). The DEMO DEFAULT is the x2 ratio (a half-window capture - the
geometry that actually looks sharp); the x4 numbers are retained as the
historical record. NOTE (Michael's F9 verdict, 30/10/2026): at the x4
64x72 capture geometry the margin is real but IMPERCEPTIBLE - every
upsampler pixelates at that size. The visible case is the x2 A/B:

| method (x2 ratio)   | PSNR    | SSIM   |
|---------------------|---------|--------|
| bilinear baseline   | 19.62 dB| 0.9067 |
| demo transform      | 20.49 dB| 0.9138 |
| delta               | +0.87 dB| +0.0071|

| method (x4, historical) | PSNR    | SSIM   |
|-------------------------|---------|--------|
| bilinear baseline       | 14.74 dB| 0.7755 |
| demo transform          | 14.79 dB| 0.7930 |
| delta                   | +0.05 dB| +0.0175|

The default parameter set (sharpen = 0: pure Lanczos3) beats the
baseline on both metrics at both ratios. The SSIM margin at x2 is small
(+0.0071) - stated plainly; the demo's case is the visible A/B toggle
at the same capture geometry, not the number. The A/B kernel
(`DemoKernel::Bilinear`, exposed through the host ABI's
`dlss5_nr_set_demo_kernel`) is the SAME resampling path with the
bilinear tent: the honest comparison floor at the same capture
geometry, matching the harness's independent bilinear reference
bit-exactly in the interior (the two edge conventions - clamped sample
positions vs re-spaced taps - differ only in the outermost border (the pin excludes 2 pixels per side)). The
harness prints a parameter sweep as the tuning record (the sweep shows
the unsharp gain trading PSNR for SSIM at the measured points).

## Geometry (documented)

The capture geometry is the seam's knob; the x2 upscale ratio is the
transform's documented property. The demo default is a **half-window
capture upscaled x2** (e.g. 640x360 -> 1280x720) - the geometry that
actually looks sharp (Michael's F9 verdict: at the 64x72 x4 geometry
the improvement is imperceptible). 64x72 x4 remains available as the
"extreme" mode (env-flagged in the seam) as the recorded stress case.
The C ABI's demo mode (`dlss5_nr_set_demo_transform(handle, scale)`,
scale 2..4) writes scale*width x scale*height output; the synthetic
pipeline's same-extent contract is unchanged when the mode is off (and
keeps its own 64x72 capture - the network's 320x320 floor). The
transform costs milliseconds per frame at this geometry — the seam
stays frame-by-frame (F9-triggered) by design, and the latency figures
in the evidence report are real measurements.

## The interface

- `vr/native-demo-upscale.hpp`: `demo_upscale()` (planar f32 RGB in/out)
  and `demo_upscale_label()` (the mandatory label).
- `vr/native-host.h`: `dlss5_nr_set_demo_transform(handle, scale)`
  (0 = off), `dlss5_nr_set_demo_kernel(handle, kernel)` (the A/B:
  Lanczos3 or the bilinear baseline at the same capture geometry;
  stored while off and applied when enabled) and
  `dlss5_nr_label_for(handle)` (the label for the handle's current
  mode).
- Centre Stage: `Dlss5Pipeline.SetDemoTransform` / `SetDemoKernel` /
  `LabelFor`; the seam enables the demo transform at the x2 half-window
  capture by default with the A/B toggle (F10) at the same capture;
  the presented frame carries the label in-band.

## Honest limits

- The quality claim is exactly the tables above: measured on one
  deterministic generated source (x2 and x4 ratios) against one
  baseline. It is not a photographic or perceptual study.
- The transform is classical resampling (documented weights), not a
  learned upscaler. A genuinely learned open-weights model (e.g. an
  ncnn-distributed Real-ESRGAN variant) would slot behind the same ABI
  as a future slice; nothing here prevents or claims that.
- The demo mode is STATELESS: the ABI's frames-since-reset counter stays
  0 and the `reset` argument has no effect while the mode is on
  (documented in NATIVE-HOST.md).
- The harness scores the unclamped f32 result; the presented path clamps
  to [0,1] before the u8 conversion. Since the source is in [0,1],
  clamping can only reduce error, so the measured numbers are a
  conservative lower bound on the presented quality.
- The kernel weights are computed with libm sin/exp: identical within a
  build (pinned by tests), but the PSNR/SSIM numbers are build-specific
  measurements rather than cross-toolchain constants.
- The demo mode's evaluate allocates working buffers per call (simple
  and correct; the transform's milliseconds hide it at this geometry).
