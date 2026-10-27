# Native layer binding and synthetic evaluator

`vr/native-eval.hpp` binds every plan-row layer kind of `native-graph-plan.hpp`
to the family kernels and runs a synthetic frame through the whole network:
`eval_layer` is a `LayerFn` for `native-network.hpp`'s `network_run`, over
caller-supplied weight records, a feature-map arena that honours the
dispatch's last-use releases, and the planar/token/interleaved-frame
transposes the kernel families need. CPU-only, allocation-free; no GPU,
headset, proprietary payload, Python or PyTorch runtime required. CPU-only
clean C++ re-derivation of published open-source semantics — **not vendor
code**.

## Contents

* `EvalWeights` — the weight-record interface, keyed exactly like the pinned
  `Weights` (`block{B}.layer{L}.{name}`): `e4m3` (row-major raw codes),
  `f16`/`f32` (float32 values on their grids), and `ffwd` — the **declared
  weight-file boundary**, the split feed-forward's packed `ffwd_a`/`ffwd_b`
  grouped scatter delivered in kernel-ready form
  (`expand_in [512,512]`, `expand_hidden [2048,64]`, `contract [512,256]`).
  Records use exact-count/shape semantics; `count == -1` is the wildcard
  form and the binding probes bounded exact counts
  (`kEvalRecordEntriesMax`) to learn a wildcard record's length, so records
  longer than the probe bound are unresolvable and rejected.
* `EvalContext` — the records, the caller-provided scratch (feature-map arena
  plus every working buffer), the lift `Controls` and the output intensity.
  `eval_reset` initialises the arena; `eval_release` is the `ReleaseFn` for
  `network_run` (recycles a released row's blocks; idempotent);
  `eval_arena_used` reports the high-water mark including the state header.
* The record derivations (`mlp_scale_slice`, `upsample_gain_row`,
  `swin_fetch`/`swin_decode`): the pinned `_residual_table` content rule (a
  zero front eight shifts the slice to `[8, 8 + C)`), the `scalars_b`/
  `attn_residual_scale` prefix slices, and the `attn_pos_bias` deswizzle
  (`swin_deswizzle_index`) plus exponent ramp (`swin_exp_bias`) into the
  `exp_bias` table — the split family's bias uses the same fragment-order
  deswizzle **without** the ramp (its kernel takes the raw unpacked bias).
* The transposes: `eval_planar_to_tokens`/`eval_tokens_to_planar` (the
  split/vit kernels are token-major `[rows*cols][C]`) and
  `frame_from_planar`/`frame_to_planar` (the lift and compose read
  interleaved `FrameView` rows; the run's image and the post block's output
  are planar `[3, rows, cols]` maps).
* The per-kind bindings (`eval_layer`): the pinned constructor records and
  derivations for all 18 kinds — the Swin body (including the pre/post
  bodies, fixed C=32/heads=1), the `ds`/`upsample` variants (`resample`
  `[2C,C]`/`[C,2C]`, the wide-gain fallback merging the `residual_scale`
  tail), the split family (`SplitFfwd` via the boundary matrices; the
  projections `weight`+`skip_weight`; `SplitQKVAttn`'s `qkv`/bias/`tail`),
  the vit family (`q`/`k`/`v`, `header` temperature, `skip_weight` gains),
  `DecInputUpsample` (`weight`+`skip_weight`), the pre block (the stored
  `input_lift` 512-record **decoded** through `decode_lift`) and the post
  block (`main_gain`/`skip_gain`, `out_project`'s first three rows, blend →
  body → residual projection → compose). The pinned `(shift_x = columns,
  shift_y = rows)` window shifts are swapped for the native swin entries
  (`swin_shifts`); `split_qkv_attn` keeps the pin order. The pinned ablation
  hook and the int4 experiment hooks are not modelled.

## Reference and provenance

Semantics re-derived from the pinned MIT reference `mauri870/dlssnr-pytorch`
commit `3beef7935a8ef976b47ab78b906e75600c52d2de` (`dlssnr/model.py` and
`dlssnr/layers/{swin,split_swin,vit,decoder,pre_post}.py`, `dlssnr/quant.py`
at the same pin). No network weights, captures, raw payloads or binaries are
used or reproduced.

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

`vr/tests/native-eval-test.cpp` (615,354 checks at -O2 — including a
bit-for-bit determinism rerun of the whole-network run — and 308,151 at -O0
via `vr/test-native-eval.sh`) validates the binding with hand-derived answers
and independent oracles: the frame/planar/token transposes (hand values,
non-square, padded strides); the record derivations (`_residual_table`
content rule, the wide-gain tail fallback); `DecInputUpsample` reusing the
decoder suite's K1 known answer through `eval_layer`; a permuted-identity
`VitFfnContract` known answer (an operand-swap control — several fixture
values fall between the e4m3 grid points, which also pins the store); a
pre-block case with a non-symmetric stored `input_lift` record whose oracle
reproduces the pinned decomposition with direct kernel calls (`decode_lift`,
`input_lift`, `swin_core`) — the control for the record decode; a `VitQKV`
case with distinct q/k/v records checked against a direct `vit_qkv` call —
the operand-routing control; the shift-order mapping; record-consumption sets
for every heavy kind (the exact pinned constructor records); the
whole-network zero-signal end-to-end over a real `build_plan` (all-zero
records + black padded frame → black output frame read through the view's
own strides, the published-stride control); determinism and
zero-allocation accounting; the wildcard record bound; release idempotence;
and a rejection matrix (missing records, sentinel-untouched outputs,
`eval_reset` buffer bounds). An independent read-only review cross-checked
every kind's operand mapping against the pinned constructors (found the Swin
body derivations, split/vit/decoder mappings, transposes and the arena
correct) and raised five findings, all fixed and reverified: the pre-block
record was not decoded through `decode_lift` (the pre-block case above is its
control), the post-block output view's `plane_stride`/`row_stride` were
swapped (the end-to-end reads the frame through its strides now), the same
swap on a scratch-sizing view, the vit-attention output channel count, and
status-code nits in the derivations. The review also identified the earlier
missing controls (lift decode, strides, QKV routing, shift order, arena,
wildcard bound) which are now in the suite. The determinism rerun is
exercised in the optimised build (the -O0 build runs the identical path
once). The sanitised build is skipped on this toolchain (missing
`/usr/lib64/libubsan.so.1.0.0`), a recorded environment gap.

## Non-vendor statement

This is not vendor code and not a reimplementation of any vendor's protected
network. It follows the MIT reference's layer contracts only and never
corresponds to NVIDIA's implementation. No proprietary weights, captures,
binaries or proprietary runtime are used or redistributed; the tree performs
no runtime or capture probes. Synthetic known answers establish synthetic
correctness only — never vendor parity.
