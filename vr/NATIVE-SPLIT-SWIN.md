# Native CPU "split Swin" layer family (encoder blocks 23–30, decoder blocks 40–47)

`vr/native-split-swin.hpp` implements the 512-channel "split Swin" layers of the
encoder bottom and decoder start — grouped feed-forward, gained 1×1
projections, the pooled projection, the 512→1024 final head and shifted-window
multi-head attention — as typed, deterministic CPU kernels. The windowed
transformer families live in the `native-swin-*` slices, the bottleneck
transformer in `native-vit`, the boundary families in `native-pre-post`/
`native-input-lift`/`native-blend`/`native-compose`. CPU-only clean C++
re-derivation of published open-source semantics — **not vendor code**.

## Contents

* `split_fma_half` — `a·b + c` with the exact product and sum in binary64 and
  a single rounding to float16 (the reference's `_fma_half`); the reference
  computes its float16 fused multiply-adds this way.
* `split_store` — the network store `_store`: float32 → float16 → e4m3, the
  double rounding of every stored result (direct e4m3 rounding differs at
  float16 half-steps).
* `split_ffn_activation` — the cubic SiLU of `SplitFfwd.activation` computed in
  float16 arithmetic (a rounding after every operation): `x` rounds to
  float16, `t = clamp(x, ±4)`, `u = fma(−0.055908203125, |t|, 0.447265625)`,
  `v = fma(t, u, 0.89453125)`, result `f16(x·v)` with the unclamped `x`.
* `split_exp` — the attention's float16 exponential: `y = clamp(fma(f16(logit),
  slope, offset), 1.03125, 1.5693359375)`; the result is the float16 whose bit
  pattern is the ten mantissa bits of `y` shifted left by five (an affine map
  into the exponent field). The clamp keeps results in normal float16 range; a
  NaN logit maps deterministically to the clamp floor.
* `split_normalise` — per-head division by the vector length with the network's
  float16 reduction (square-sums `d` with `d + 16`, then a butterfly over
  distances 8, 4, 2, 1), the f16 floor `0.000062` (6.198883056640625e-05), a
  float32 reciprocal square root rounded to float16, and float16 per-channel
  products.
* `split_softmax_denominator` — the pinned 64-leaf denominator reduction
  (tiles sequential `((0+1)+2)+3` per (half, c), adjacent c pairs,
  `(0+1)+(2+3)` per half, then the halves). The reference accumulates these
  sums in float32; over the exponential's leaf range every intermediate is
  exact in float32, so the pinned order and the exact sum agree.
* `split_matmul` — `out[t][n] = Σ_k in[t][k]·W[n][k]` with e4m3 weights held
  exactly in float32 and float32 accumulation in key order (the family's
  `int8.mm` with its default plain-product semantics).
* `split_projection_value` — the family's 1×1 projection value
  `f16(W·x + gain·residual)` per channel (`_Projection.project`): float32
  product and gained residual sum, one rounding to float16.
* `split_ffwd` — `SplitFfwd`: eight groups of 512 → 64 → 256 (cubic SiLU in
  float16) → 64, every stage stored; the block's residual input is the
  feed-forward's input (returned unchanged by the reference).
* `split_ffwd_proj` / `split_proj` — `SplitFfwdProj`/`SplitProj`: the stored
  projection (`e4m3` of the float16 value), the residual being the block input
  or the feed-forward projection's output.
* `split_proj_pool` — `SplitProjPool`: the projection plus the 2×2-mean pooled
  copy of the float16 projection values (pairwise column sums then row sums in
  float16, × 0.25, e4m3), extent `ceil(extent/2)` zero-padded to a multiple of
  4. The reference executes even extents only; the native form defines a
  trailing odd row/column pairing with an implicit zero as an explicit
  extension.
* `split_final_head` — `SplitFinalHead`: the residual-free 512 → 1024
  projection of the pooled copy (the transformer stage's input).
* `split_qkv_attn` — `SplitQKVAttn`: window attention over 8×8 pixels of a grid
  shifted by 0 or −1 tiles (`shift_x` along columns, `shift_y` along rows). A
  window's 64 tokens run over its four 4×4 tiles (top-left, top-right,
  bottom-left, bottom-right), each tile row-major — the order the position bias
  is stored in. Q and K are normalised per 32-channel head, Q is scaled by the
  head's float32 gain (rounded to float16), both are rounded to e4m3; logits
  are `Q·Kᵀ + bias` in float32 with the e4m3 grid read back exactly; the
  softmax uses `split_exp`, the pinned denominator and e4m3 probabilities
  (`e4m3(f16(leaf·inverse))`); the context accumulates in float32 and is stored
  (float16 → e4m3). Tiles of a shifted window outside the map are zero tokens:
  they take part as keys (their logits still carry the bias) and their outputs
  are dropped.

The reference's experimental int4/int8 quantisation hooks are out of scope;
this slice executes the e4m3/float16 dataflow (`int8.mm` with the default empty
site set is a plain float32 matrix product). Weight-record decoding (the packed
`ffwd_a`/`ffwd_b` scatter maps and the accumulator-fragment `attn_pos_bias`
unpack) is a weight-file concern outside this slice: the kernels take the
decoded matrices and unpacked per-head biases directly.

## Reference and provenance

Semantics re-derived from the pinned MIT reference `mauri870/dlssnr-pytorch`
commit `3beef7935a8ef976b47ab78b906e75600c52d2de`
(`dlssnr/layers/split_swin.py`, cross-checked against `quant.py`/`int8.py`);
the pinned copy's content was re-verified against the upstream commit. No
network weights, captures, raw payloads or binaries are used or reproduced.

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

`vr/tests/native-split-swin-test.cpp` (34,267 checks at -O2 and -O0 via
`vr/test-native-split-swin.sh`) validates against hand-derived answers: the
fused multiply-add (including a derived float16 tie), the double-rounding
store, the activation at six points (including its clamp zero and sign), the
exponential at seven logits (clamps and overflow), the normalisation butterfly
(rounding swallow, floor, uniform head), the denominator (uniform and an
order-sensitive sparse case), matmul dot products, projection values with a
derived tie, the stored projections, the pooled projection (even map, odd map
with the implicit-zero extension and its stride-4 padded layout, pairwise
rounding), the final head, the grouped feed-forward through identity weights,
windowed attention with sparse weights (single-window softmax, position-bias
orientation with a transposed-reading control, the window-token order anchored
at token 16, head gain, one-live-token geometry, single- and double-shifted
grids with padded windows), rejection classes, sentinel guards and
determinism. An independent read-only review re-derived the anchors and the
`split_swin.py` mapping; findings are fixed (two transcription errors in
hand-derived literals/layout, the pooled padding zero-fill count for
multi-channel maps, comment derivations).

## Non-vendor statement

This is not vendor code and not a reimplementation of any vendor's protected
network. It follows the MIT reference's layer semantics only and never
corresponds to NVIDIA's implementation. No proprietary weights, captures,
binaries or proprietary runtime are used or redistributed; the tree performs no
driver-level runtime/capture probing.
