# Native host-image boundary and image evaluation

`vr/native-io.hpp` binds the pipeline to host images: `frame_from_u8` (8-bit
RGB rows to the planar float colour map with the pinned mirrored edge
padding), `frame_to_u8` (the post block's planar output cropped and rounded
to 8 bits) and `image_evaluation` (host frame in, enhanced host frame out,
through the typed run with the temporal history feed). CPU-only,
allocation-free; no GPU, headset, proprietary payload, Python or PyTorch
runtime required. No vendor code and no vendor correspondence.

## Contents

* `frame_from_u8` — uint8 `[height][width][3]` rows of `host_row_stride`
  bytes to the planar map `out[c * work_rows * work_cols + r * work_cols +
  col]`, values `k/255` (the pinned `levels` table), padded to
  `work_rows x work_cols` by the pinned `_mirror` (reflect an out-of-range
  index once about the last pixel, `2*extent - index - 2`, then clamp).
* `frame_to_u8` — the planar map cropped to the frame extent, clamped to
  [0,1], rounded exactly as the pinned `tensor_to_frame`
  (`whole = floor(scaled); code = whole + (scaled - whole >= 0.5)`). NaN
  inputs map to code 0 (an explicit policy; the pinned cast leaves NaN's code
  implementation-defined). The rule is implemented literally: `floor(scaled +
  0.5f)` diverges at `scaled = 0.5 - 2^-25` (reachable at `fl(0.5/255)`,
  where the pin gives 0 and the naive form gives 1); the suite pins that
  edge.
* `image_evaluation` — the host-frame entry over `network_run`: the plan's
  padded extent is derived from the pre row (`2 * out_w` rows, `2 * out_h`
  columns; `out_w` is the row count per native-graph-plan.hpp/pinned
  plan.py), the run is wired to the caller's `EvalContext` (`run.context` is
  set on the by-value copy), the temporal carrier's history colour is fed to
  the input lift's slots 7..9 (null keeps the single-frame stub; the carrier
  must match the padded extent in the chain's orientation and hold [0,1]
  colour), and the final row must be the post block whose result map is the
  full padded extent (both preconditions are validated before the run).
* The temporal history feed: `native-input-lift.hpp` gains the primary
  `input_lift(frame, history, ...)`; the old entry delegates with
  `history == frame` — behaviour-preserving (the reference's no-history
  stand-in), verified by the unchanged 5,316-check suite.

## Reference and provenance

Semantics re-derived from the pinned MIT reference `mauri870/dlssnr-pytorch`
commit `3beef7935a8ef976b47ab78b906e75600c52d2de` (`dlssnr/io.py`
`frame_to_tensor`/`tensor_to_frame`/`_mirror`/`Controls`) plus this
repository's temporal contract (`NATIVE-HISTORY.md`). No network weights,
captures, raw payloads or binaries are used or reproduced. The host stays the
caller's boundary: the weight-file load (including the split feed-forward's
`ffwd` boundary decode) is not performed here.

## MIT licence (reference-derived sections)

Copyright (c) 2026 Mauri de Souza Meneguzzo

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Known-answer test

`vr/tests/native-io-test.cpp` (2,045 checks at -O2 and again at -O0 via
`vr/test-native-io.sh`) pins the boundary with hand-derived answers: the
`_mirror` tables for extents 2/3/4; a non-square host (2x3 to a 4x5 extent)
whose map is checked against hand-computed mirror maps in the chain's
orientation; the exact rounding table over the dyadic grid `m/512` in integer
arithmetic (`(m*255 + 256)/512`), including the 127.5 tie and the divergent
midpoint `fl(0.5/255)` -> 0; the 256-code round-trip (error bound
`255 * 2^-23`); clamp and NaN edges; the boundary rejection matrix with
sentinel-untouched outputs; the lift history feed (a one-weight lift matrix
making the history and current colour separately visible, plus the extent
mismatch); the `image_evaluation` seam through a stub layer pinning the work
extent derivation, the crop, the orientation and the rounding together (the
expected codes come from integer arithmetic, not the code's float
expression); the runner's precondition rejections; determinism and
zero-allocation accounting with the repo's full allocation-override block.
Independent read-only review cross-checked the boundary against the pinned
`io.py` and the chain's axis convention and raised eight findings, all fixed
and reverified: a fatal transposed-axis defect (the boundary had treated
`out_w` as the width axis; the plan's `out_w` is the tensor row count =
height — `image_evaluation` rejected every landscape frame before the fix),
 the rounding divergence at `fl(0.5/255)`, a missing result-extent check on
truncated plans, an unchecked `run.context` seam, the carrier/lift [0,1] and
extent contract (now documented and pre-checked; the lift's extent check now
precedes its value scan), overstated validation-before-write (now true: the
run preconditions are checked before the image map is written), the missing
`kMaxPixels` bound, and the undocumented NaN policy. Disclosed gap: the
non-null `EvalContext.history` path through `eval_pre_block` is exercised at
the lift level (the suite above) but not through `eval_pre_block` itself; the
eval suite covers the null (stub) path. The sanitised build is skipped on
this toolchain (missing `/usr/lib64/libubsan.so.1.0.0`), a recorded
environment gap.

## Non-vendor statement

This is not vendor code and not a reimplementation of any vendor's protected
pipeline. It follows the MIT reference's I/O contracts and this repository's
evidence only and never corresponds to NVIDIA's implementation. No
proprietary weights, captures, binaries or proprietary runtime are used or
redistributed; the tree performs no runtime or capture probes. Synthetic
known answers establish synthetic correctness only — never vendor parity.
