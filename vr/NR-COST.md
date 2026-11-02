# NR-COST — what one DLSSNR evaluation costs, and whether that fits a VR frame

Status: measurement, 02/11/2026. One GPU, one model build, synthetic inputs, no image
quality claim. This document exists to answer a single question before any VR port is
attempted: **can a stereo frame afford two DLSSNR evaluations?**

## Method

`vr/nr-bench.c` (Windows PE, run under Proton through `vr/run-nr-bench.sh`, same
containment boundary and prefix as the desktop demo) creates an NGX feature per
instance, uploads synthetic colour/depth/motion inputs once, then times a serialised
loop: record one Evaluate, execute, signal a fence, wait for it. Frames 0–1 are
discarded so pipeline creation is not in the sample; the reported figure is the mean of
the remaining frames.

Resolution is a runtime parameter and the ladder is one Proton launch per binary run,
because the loop creates and destroys features per rung. `--instances 2` creates two
independent features (the two-eye case), each with its own history state.

**The metric is the fence-timed serialised frame, not a GPU timestamp query.** D3D12
timestamp queries are not usable through this stack: `BeginQuery` is not re-stamped per
frame — it returns a near-constant sentinel (byte-identical across frames apart from
128 ticks) while `EndQuery` advances with absolute elapsed time, so the pair does not
bound one frame's work. The raw stamps are printed under `--verbose` and reproduce the
defect. Serialised frame time is also the figure that matters for a consumer: a VR
compositor cannot be handed the frame until the evaluation has completed.

## The ladder

RTX 4070 Ti SUPER, model `sha256 4b8d19bc…`, 38 measured frames per rung, synthetic
stationary gradients.

| per-eye resolution | megapixels | one instance (ms) | two instances (ms) |
|---|---|---|---|
| 720 × 720 | 0.52 | 3.00 | 5.91 |
| 1024 × 1024 | 1.05 | 3.65 | 7.19 |
| 1440 × 1440 | 2.07 | 5.63 | 11.31 |
| 1832 × 1832 | 3.36 | 8.91 | 17.84 |
| 2064 × 2064 | 4.26 | 10.91 | 21.87 |

Two instances cost very nearly twice one instance at every rung, which is the evidence
that the loop is measuring the evaluations rather than submission overhead.

A straight line through the one-instance column is **1.9 ms fixed + 2.11 ms per
megapixel**.

**Content check.** The same run with non-zero motion vectors (`--motion 6.0,1.5`)
measured 5.887 ms at 720² and 21.707 ms at 2064² against 5.913 / 21.867 for the static
inputs — within 0.7%, so the model is not skipping work on stationary input.

## What this means for VR

A Quest 3-class eye buffer is 2064 × 2208 = 4.56 MP, so the fit gives about
**11.5 ms per eye, 23 ms for both** on this GPU.

| target | frame budget | NR alone, both eyes | verdict |
|---|---|---|---|
| 90 Hz | 11.1 ms | 23 ms | 2.1× over — the whole budget, twice |
| 72 Hz | 13.9 ms | 23 ms | 1.7× over |
| 90 Hz at 1440²/eye | 11.1 ms | 11.3 ms | the entire budget, no room to render |
| 90 Hz at 1024²/eye | 11.1 ms | 7.2 ms | ~3.9 ms left to render the scene |
| 72 Hz at 720²/eye | 13.9 ms | 5.9 ms | ~8 ms left to render the scene |

So: **at native headset resolution, on this GPU, DLSSNR cannot hold a VR frame rate.**
It is not marginal — it is a factor of two. It becomes affordable only at roughly
1024² per eye or below, which is low enough that the thing being improved is a small
image. A faster GPU scales the per-megapixel term but not the 1.9 ms fixed cost.

This is the number that decides the VR question, and it is not the answer we wanted.

## What this does establish, and what it does not

Established:

- The model accepts **application-owned textures**, not a swapchain. Nothing in the
  path requires DXGI presentation.
- It accepts **arbitrary sizes** across a 0.5–4.3 MP range: feature creation and every
  evaluation returned success at all five rungs, with no size constraint hit.
- **Two independent feature instances** coexist and both evaluate successfully for
  38 consecutive frames, each with its own history — so a per-eye instance pair is
  possible.
- Cost is content-independent to within 0.7%.

Not established:

- Anything about image quality at these sizes, or vendor output parity.
- Any OpenXR/WiVRn integration: no compositor hand-off, no pose-to-frame latency, no
  late-latching, no foveation.
- Behaviour when the two instances must complete inside one frame's command stream
  alongside scene rendering; the bench measures NR in isolation.
- Anything on a GPU other than the one listed.