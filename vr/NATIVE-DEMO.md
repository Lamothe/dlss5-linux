# NATIVE-DEMO — runnable native Linux pipeline demo

Status: demo slice (test/tool/docs). The typed CPU stages themselves are the
`NATIVE-*.md` slices; this file documents the runnable demo over them and
carries the required attribution for the small amount of reference-derived
material in `native-demo-support.hpp`.

## Scope

`vr/demo-native.cpp` runs the complete typed network on the CPU:

1. Read a host image (binary PPM, P6, exactly 64x72 RGB) or generate a
   built-in checkerboard/gradient (zero assets out of the box).
2. `build_plan(64, 72, kSmall)` — the full 76-row plan over the plan's padded
   320x320 extent — and `image_evaluation` per frame with deterministic
   synthetic `EvalWeights` records.
3. Write `before.ppm`, `after.ppm` and a side-by-side `contact.ppm`.
4. Run a short frame sequence through the temporal carrier with a mid-stream
   reset, printing the frames-since-reset counter and history state.
5. Print per-frame and total CPU wall time and the arena high-water mark
   (the CPU baseline for the later GPU stage).

## What this demo proves — and what it does not

The weight records are synthetic and deterministic (pure functions of
`(preset, seed, block, layer, name)`). The visible transformation is
arbitrary: this proves the complete pipeline runs on Linux, **not** DLSS
quality and **not** vendor correspondence. No proprietary weights, captures,
payloads or binaries are used or produced; no runtime/capture probes are
performed; no vendor parity is claimed or implied.

## Components (`vr/native-demo-support.hpp`)

* `lcg_next` — the Numerical Recipes 32-bit LCG (`x' = 1664525x + 1013904223
  mod 2^32`), hand-derived known answers in the tests.
* `fnv1a` — the FNV-1a name hash used to key records (a deterministic mixer,
  not a security primitive).
* `fill_checker` / `fill_gradient` — the built-in synthetic images.
* `encode_ppm_p6` / `decode_ppm_p6` — a strict binary PPM codec (the exact
  header bytes are pinned in the tests).
* `contact_sheet` — the before|after side-by-side layout.
* `DemoStore` + `demo_bind_weights` — the synthetic weight presets over the
  `EvalWeights` record interface. Each distinct `(block, layer, name, kind)`
  record gets its own storage slot because the consumers hold several record
  pointers live within one row (swin_decode keeps five code records plus the
  scalar tables at once); slot contents are keyed, so a longer later request
  extends a slot without changing the bytes already handed out. Presets:
  `kZero` (all records zero) and `kSeeded` (small e4m3 codes 12..28 — the
  tree's bias-7 decode gives 0.0234375..0.09375 — and 1/1024-grid values in
  [-1, 1)), chosen so multi-layer activations stay off the e4m3 ceiling.
* `round_frame_to_f16` — the FP16 output-texture store model (below).

## The temporal commit and the FP16 store

The network's final map holds compose values on the 8-bit grid (1/255 steps,
not binary16-representable), while `TemporalCarrier::temporal_commit`
validates its source as finite **and** binary16-representable: the capture
contract is that the carrier stores FP16 texture content bit-exactly
(`history_N = output_{N-1}`). The hardware route writes its output into an
FP16 texture, so the demo rounds each value through the tree's verified
binary16 round (`f16`) before committing — modelling that store. For values
already binary16-exact the round is the identity. Validation is a full first
pass; a non-finite value or one beyond the max finite binary16 magnitude
(65504) rejects the whole frame before any write.

The temporal influence of a history frame lands partly outside the top-left
64x72 crop that `frame_to_u8` reads back, so the demo prints `max |out_N -
out_{N-1}|` at the FP16 store precision over both the visible crop and the
full padded map. The frames-since-reset counter/reset behaviour is the
capture-evidence state machine regardless.

## Tests and the runner

`vr/tests/native-demo-test.cpp` — hand-derived known answers: the LCG
constants (1013904223, 1015568748, 1586005467), the PPM byte layout
("P6\n2 2\n255\n" = 11 bytes + 12 payload = 23), the checker parity and
gradient endpoints, the contact-sheet placement, the preset determinism and
ranges, the wildcard record lengths (matching the eval suite's
`wildcard_name`), and the FP16 store rounding (`f16(1/255) =
0.0039215087890625`, the 70000 and NaN rejections). 351 checks green at -O2
and -O0 on the development host (the suite prints its own count).

`vr/test-native-demo.sh` — the runner: the tests at -O2/-O0, the sanitiser
build with the tree's standard missing-runtime gap handling (this toolchain
lacks `/usr/lib64/libubsan.so.1.0.0`, so the sanitised runs skip), and a
bounded one-frame demo smoke checking the artefacts' exact sizes
(13837 = 13 + 3*64*72 and 27662 = 14 + 3*128*72 bytes).

See `README-demo.md` for the exact commands, expected console output and how
to open the images.

## Non-vendor statement

Synthetic correctness is **not** vendor correspondence. No synthetic test in
this repository demonstrates equivalence with any NVIDIA product, model or
image quality, and no software licence grants rights to NVIDIA model data.

## Provenance and licence

The record-name/shape request patterns in `native-demo-support.hpp` follow
the pinned MIT-licensed reference `mauri870/dlssnr-pytorch`
@ `3beef7935a8ef976b47ab78b906e75600c52d2de` (its `dlssnr/layers/*` record
fetch patterns), used as a specification reference only. No reference code,
weights, captures or payloads are copied into this tree beyond small
adapted logic, which retains this notice. The reference is MIT licensed:

```
MIT License

Copyright (c) 2026 Mauri de Souza Meneguzzo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
