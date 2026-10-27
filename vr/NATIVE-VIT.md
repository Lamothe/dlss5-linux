# Native CPU vision-transformer family (bottleneck blocks 31-38)

`vr/native-vit.hpp` implements the Vit layer kinds of the network plan — FFN
expand/contract, QKV projection with per-head normalisation, global attention
and the residual projection — as typed, deterministic CPU kernels. The
windowed-transformer families live in the `native-swin-*` slices; the boundary
families in `native-pre-post`/`native-input-lift`/`native-blend`/
`native-compose`. CPU-only clean C++ re-derivation of published open-source
semantics — **not vendor code**.

## Contents

* `matmul_e4m3` — `out[t][n] = Σ_k in[t][k] · W[n][k]` with e4m3 weights held
  exactly in float32, float32 accumulation in key order (the reference's
  `_Matmul` semantics).
* `ffn_activation` — the clipped cubic SiLU of the reference
  (`x · (−0.055908203125·|clip|·clip + 0.447265625·clip + 0.89453125)`,
  `clip = clamp(x, −4, 4)`).
* `approximate_exp` — the network's piecewise-linear exponential: a float16
  fused multiply-add `y = round((x·slope + offset)·1024)` clamped to
  `[1474, 2025]`, whose mantissa bits select `ldexp(1 + frac, exp)`. Rounding
  is explicit ties-to-even (no ambient FP-mode dependence); a NaN logit maps
  deterministically to the clamp floor.
* `inverse_norm_head` — per-head `1/max(sqrt(Σx²), floor)` with the network's
  float16 summation order (channel `d` with `d + 16`, then a butterfly over
  distances 8, 4, 2, 1) and the f16 floor `0.000062`.
* `vit_qkv` — q/k/v projections (e4m3), per-head k/|k| and
  q·√32/|q|·temperature in float16, output `[q | k | v]` rounded
  float32 → float16 → e4m3.
* `group_sum64` — one 64-key group's probability sum in the network's order
  (key `c` added to `c + 8` per 16-key block; four-block accumulation; even
  and odd partials `((p0+p2)+p4)+p6` / `((p1+p3)+p5)+p7`; all adds float16).
* `vit_attention` — global multi-head attention: keys pad to a multiple of 64
  with zero keys (logit 0, counted as `exp(0)` and subtracted at the end);
  probabilities via `approximate_exp`; the denominator is the float16 grouped
  sum; the numerator multiplies e4m3-rounded probabilities with e4m3 values in
  float32; the output is `e4m3(f16(f16(num) · f16(1/max(den, floor))))`.
* `vit_ffn_expand` / `vit_residual_linear` — the VitFfnExpand,
  VitFfnContract and VitProjection layer semantics (activation or
  `f16(product + gain·skip)`, then the float16 → e4m3 hop).

The reference's experimental int4 quantisation hooks are out of scope; this
slice executes the e4m3/float16 dataflow.

## Reference and provenance

Semantics re-derived from the pinned MIT reference `mauri870/dlssnr-pytorch`
commit `3beef7935a8ef976b47ab78b906e75600c52d2de` (`dlssnr/layers/vit.py`).
No network weights, captures, raw payloads or binaries are used or reproduced.

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

`vr/tests/native-vit-test.cpp` (404 checks at -O2 and -O0 via
`vr/test-native-vit.sh`) validates against hand-derived answers: the
exponential at four logits plus clamping, the activation at five points
(including its clamp-zero), the inverse norm on derived heads, matmul dot
products, a full QKV anchor (identity weights, derived f16/e4m3 chain), group
sums (uniform and sparse), single-token attention with the padding
subtraction derived end to end, a 2-head × 128-token case exercising
multi-head offsets and multi-group accumulation, the FFN expand and residual
linear chains through f16 and e4m3, rejection classes, sentinel guards and
determinism. An independent read-only review re-derived every anchor and the
whole `vit.py` mapping; its findings (runner failure masking, NaN undefined
behaviour, rounding-mode dependence, bounds, comment derivations) are fixed.

## Non-vendor statement

This is not vendor code and not a reimplementation of any vendor's protected
network. It follows the MIT reference's layer semantics only and never
corresponds to NVIDIA's implementation. No proprietary weights, captures,
binaries or proprietary runtime are used or redistributed; the tree performs no
driver-level runtime/capture probing.
