# The demonstration transform: visible quality through the identical seam

**Status: complete.** This is the scope-pivot phase's item 2: a REAL learned
upscaler with REAL openly licensed weights behind the identical C ABI /
capture / present path used by the synthetic pipeline. It proves the
platform end to end with a visible result. **It is NOT DLSS5 and must never
be conflated with one.** Every artefact, UI label and report carries
`not DLSS5` in-band.

## The models (real weights, openly licensed, openly obtained)

Both are fetched by `vr/fetch-demo-model.sh` (pinned checksums; the repo
carries no model binaries). No NVIDIA assets, no proprietary payloads,
captures or binaries, no runtime probes.

| Model | Class | License | Source | Parameters | Timings at 640x400 (CPU) |
|---|---|---|---|---|---|
| **EDSR-base x2** (default) | classical super-resolution (trained on clean degradation, DIV2K, L1) | Apache-2.0 (Lim et al., CVPRW 2017) | `huggingface.co/eugenesiow/edsr-base` (`pytorch_model_2x.pt`) | 1,369,883 (16 ResBlocks x 64 feats, measured and pinned by the smoke) | **10-11 s/frame** (8 threads; measured 10.2-11.2 across runs) |
| Real-ESRGAN x2 | blind restoration (GAN-trained) | BSD-3-Clause (Wang et al., ICCVW 2021) | `huggingface.co/mlx-community/Real-ESRGAN-x2plus` (`model.safetensors`) | 16,703,171 (RRDBNet, F16) | **35 s/frame** (8 threads; measured 35.2 s) |

CPU inference is accepted for frame-by-frame F9; the timings above are
measured worker wall clock around the native call (they vary with load;
the evidence report carries the exact run) and the app never claims
real-time (the processing state is visible while the frame is in
flight).

## Loaders (format-agnostic plumbing)

`vr/native-demo-rrdbnet.hpp` reads **safetensors** (the Real-ESRGAN
weights); `vr/native-demo-edsr.hpp` reads the **torch-save** container (a
zip of a pickle stream plus raw f32 storages; the supported pickle subset
is the state-dict form only - anything else is rejected with a named
error). The container is detected from the file magic; every expected
tensor name and shape is verified and every mismatch is rejected by name
(the weights-interface contract in `NATIVE-WEIGHTS-FILE.md` applies to the
NWF1 path; these demo loaders verify the same way: shapes and names in,
named rejections out, no numeric probing).

## The honesty labelling (fixed in this phase)

- The neural views carry `not DLSS5 - demonstration transform with real
  weights (EDSR-base x2, Apache-2.0)` (or the Real-ESRGAN variant) - the
  phrase "real weights" is reserved for the neural path.
- The classical views carry `not DLSS5 - classical demonstration scaler
  (Lanczos3)` / `(bilinear baseline)`. **Lanczos3 is a fixed-kernel
  scaler; it has no weights and is no longer described as having any.**
- The label is in-band on every presented frame (`dlss5_nr_label_for`) and
  in every report line.

## Measured quality - and the honest finding

Capture geometry (the seam's knob): **640x400 capture, x2, 1280x800
presented** (one uniform scale). The quality family
(`vr/test-native-demo-upscale.sh`) measures on generated chart content;
the evidence run measures on the app's real capture against the ground
truth (the same view rendered at the presented size).

**On generated chart content** (the adversarial synthetic chart:
checkerboards, sine waves, hard edges - hostile to learned models):

| View | PSNR | SSIM |
|---|---|---|
| bilinear | 19.97 dB | 0.9068 |
| Lanczos3 | 20.45 dB | 0.9075 |
| EDSR (neural) | 16.09 dB | 0.7794 |
| Real-ESRGAN (neural) | 11.81 dB | 0.6795 |

Both neural models lose decisively on this content (it is made of
patterns learned models hallucinate over). The chart is a torture test,
not this demo's case; the real capture below is.

**On the real capture** (the evidence run, `report.txt` per view; full
frame PSNR/SSIM, then the pre-declared centre crop (central 50%), then
the detail energy = mean squared Laplacian of the luminance x1000 in the
centre crop; ground truth = 198,570):

| View | Full frame | Centre crop | Detail energy | Evaluate |
|---|---|---|---|---|
| **EDSR (neural)** | PSNR 36.98 dB, SSIM 0.9812 | PSNR 31.92 dB, SSIM 0.9539 | **186,684** | 11.2 s |
| Real-ESRGAN (neural) | PSNR 37.68 dB, SSIM 0.9877 | PSNR 33.78 dB, SSIM 0.9638 | 70,393 | 35.2 s |
| bilinear | PSNR 40.41 dB, SSIM 0.9920 | PSNR 35.10 dB, SSIM 0.9731 | 17,195 | instant |
| Lanczos3 | PSNR 40.18 dB, SSIM 0.9923 | PSNR 34.98 dB, SSIM 0.9748 | 38,100 | instant |

**The honest two-metric story** (both rows are real measurements; the
demo's claim is the detail row, and the fidelity row is reported against
it, not hidden):

1. **Detail recovery - the demo's claim - is supported.** The ground
   truth carries fine structure (198,570); the classical scalers smooth
   81-91% of it away (17,195 / 38,100 - the softness that reads as
   "pixelated"). The neural views reconstruct it: EDSR recovers 94%
   (186,684 - the visibly textured choice) and Real-ESRGAN 35% (70,393 -
   a gentler result). This is what the F10 A/B shows: the neural views
   are textured where the classical views are soft.
2. **Pixel fidelity still favours the classical scalers** (PSNR/SSIM
   2.7-3.4 dB / ~0.01-0.02 higher than the best neural row).
   Reconstruction detail close in magnitude is not pixel-exact; a
   pristine-render ground truth rewards smoothing. The gap to the best
   neural row is 0.004-0.011 SSIM and 2.7-3.4 dB. The two neural models
   also trade against each other (Real-ESRGAN: higher fidelity, less
   detail; EDSR: more detail, lower fidelity). Both metrics are stated;
   neither is oversold.

Earlier in this phase a layout defect (the neural branch read the planar
buffer as interleaved) silently degraded every neural measurement; the
independent review caught it, the layout is now pinned bit-exactly
against a direct forward in the smoke, and the numbers above are from the
corrected path. The earlier scrambled-path rows (EDSR 32.61, Real-ESRGAN
34.05) are superseded by the corrected per-model evidence runs kept in
`.evidence/dlss-demo-edsr/` and `.evidence/dlss-demo-realesrgan/`; on the
chart-content family - the chart remains aliasing-dominated (every
row lands at 5-8 dB) and is a trend only: on it EDSR scores 5.56 and
Real-ESRGAN 7.58 vs bilinear 7.29 - the chart content decides nothing.

## Verified implementation

- The EDSR forward (head -> 16 ResBlocks -> body conv -> global residual
  -> conv+PixelShuffle -> conv) matches the upstream super-image
  implementation; the MeanShift buffers exist in the checkpoint and the
  upstream forward does not call them (verified against the source); the
  output is mean-preserving (measured deltas < 0.015).
- Deterministic (bit-identical repeated forwards at 1/2/8 threads).
- Analytic smooth-gradient target: PSNR 30.6 dB (implementation check
  with the upstream zero-padding semantics, not a quality claim; a broken
  sub-pixel shuffle scores under 20 dB).
- Loader rejection coverage: the host test covers the missing-file and
  NULL-pointer paths (`vr/test-native-host.sh`); the EDSR smoke covers
  truncated and wrong-magic files with named errors and asserts the
  loaded model SURVIVES a failed reload. A deliberate
  topology-mismatched file (correct container, wrong tensors) is
  rejected by name in code but has no dedicated test.

## Evidence (unattended runs)

`CentreStage-dlss5/.evidence/dlss-demo-edsr/`: `live.png`, `processing.png`
(latency visible in-band), `processed.png` (neural), `baseline.png`
(bilinear), `lanczos3.png`, and `report.txt` (per-view PSNR/SSIM full
frame + centre crop, detail energy, timings, labels). The prior
Real-ESRGAN run is kept at `.evidence/dlss-demo-neural/` for comparison.
