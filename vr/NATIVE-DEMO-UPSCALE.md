# NATIVE-DEMO-UPSCALE — the demonstration transform (NOT DLSS5)

Status: complete (30/10/2026, scope-pivot phase, item 2).
Tests: `vr/test-native-demo-upscale.sh` (the quality family), the mock
host family (`vr/test-native-host.sh`, 60 checks incl. the mode).

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
region), box-downscales x4 to an 80x72 "capture", and compares the
transform against the source AND a bilinear baseline (PSNR and the
standard Wang SSIM):

| method              | PSNR    | SSIM   |
|---------------------|---------|--------|
| bilinear baseline   | 14.74 dB| 0.7755 |
| demo transform (x4) | 14.79 dB| 0.7930 |
| delta               | +0.05 dB| +0.0175|

The default parameter set (sharpen = 0: pure Lanczos3) beats the
baseline on BOTH metrics — that is the measured basis for "Michael's F9
visibly improves the image". The harness prints a parameter sweep as the
tuning record (the sweep shows the unsharp gain trading PSNR for SSIM at the measured points).

## Geometry (documented)

The Centre Stage seam captures 64x72 at the native boundary's demo
geometry and presents the x4 demonstration result: **64x72 capture ->
256x288 presented frame** (the window scales the presented frame to the
viewport). The C ABI's demo mode (`dlss5_nr_set_demo_transform(handle,
scale)`, scale 2..4) writes scale*width x scale*height output; the
synthetic pipeline's same-extent contract is unchanged when the mode is
off. The transform costs milliseconds per frame at this geometry — the
seam stays frame-by-frame (F9-triggered) by design, and the latency
figures in the evidence report are real measurements.

## The interface

- `vr/native-demo-upscale.hpp`: `demo_upscale()` (planar f32 RGB in/out)
  and `demo_upscale_label()` (the mandatory label).
- `vr/native-host.h`: `dlss5_nr_set_demo_transform(handle, scale)`
  (0 = off) and `dlss5_nr_label_for(handle)` (the label for the
  handle's current mode).
- Centre Stage: `Dlss5Pipeline.SetDemoTransform` / `LabelFor`, the seam
  enables x4 by default; the presented frame carries the label in-band.

## Honest limits

- The quality claim is exactly the table above: measured on one
  deterministic generated source at x4 against one baseline. It is not a
  photographic or perceptual study.
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
