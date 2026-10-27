# Native CPU decoder input stage (block 39): `DecInputUpsample`

`vr/native-decoder.hpp` implements the decoder's input layer — the stage where
the bottleneck's output meets the encoder's full-resolution skip — as typed,
deterministic CPU kernels: the e4m3 projection kept in binary16, the nearest-2×
replication cropped to the skip's extent, the per-channel gated skip and the q8
store. The windowed transformer families live in the `native-swin-*` slices,
the bottleneck transformer in `native-vit`, the split Swin family in
`native-split-swin`, the boundary families in `native-pre-post`/
`native-input-lift`/`native-blend`/`native-compose`. CPU-only clean C++
re-derivation of published open-source semantics — **not vendor code**.

Not to be confused with [NATIVE-DECODER.md](NATIVE-DECODER.md), the
milestone-2 proprietary-route decoder launch-contract investigation record;
this document covers only the independent synthetic implementation path.

## Contents

* `DecInputWeight` — the layer's weights: the e4m3 projection `[c_out, c_in]`
  in the `_Matmul`/weight-file representation (row-major raw e4m3 codes; the
  codes `0x7f`/`0xff` are the e4m3 NaN encodings and are rejected) and the
  per-channel f16 `skip_weight` gain row. All operands are passed explicitly;
  no learned weights are used or implied (weight-record decode stays the
  declared weight-file boundary outside this slice).
* `dec_gated_element` — the gated skip term `f16(f16(skip) · gain)`: the skip
  value rounds to binary16 first (the reference's explicit `f16(skip)`), then
  the product with the binary16 gain is computed in float32 (exact: at most 22
  significant bits) and rounded to binary16 once, ties to even. Total for any
  float32 inputs (pre-rounding overflow yields ±infinity; `infinity · 0` yields
  NaN canonicalised by the binary16 rounding — a quiet NaN with zero payload,
  the sign being the raw product's platform default).
* `dec_store` — the network store `q8(f16(value))`: float32 → binary16 →
  e4m3, saturating at ±448 with ties to even, NaN canonicalised to the e4m3
  NaN code `0x7f` exactly as the reference's q8 ("NaN is not preserved"; the
  e4m3 store drops the NaN sign, so the store of any NaN is the positive
  canonical quiet NaN `0x7fc00000`). Direct e4m3 rounding differs at binary16
  half-steps (68.00390625 stores as 64 through the binary16 hop but rounds to
  72 directly).
* `validate_decoder_map` — the shared map geometry/stride envelope
  (`kDecMaxExtent` 16448 follows the split family's padded-frame bound;
  `kDecMaxTokens` 2^29; the channel envelope is the vit family's
  `kVitMaxChannels`).
* `dec_input_upsample` — `DecInputUpsample.forward(x, skip)`: `x` is the
  bottleneck output map `[c_in, rows, cols]` on the FP8 boundary (e4m3-grid
  values, e4m3 NaN codes rejected), `skip` the encoder's full-resolution skip
  `[c_out, skip.rows, skip.cols]` (any finite values). The projection
  `f16(matmul(to_tokens(x)))` is the vit family's matmul: `[rows·cols][c_in]`
  row-major tokens, `out[t][n] = sum_k in[t][k] · W[n][k]`, e4m3 operands
  exact in float32, float32 accumulation in key order, every result rounded to
  binary16 once; the projected map is written in full into the caller's tight
  scratch mirror. `upsampled` repeats each projected value over a 2×2 block
  (output `(y, x)` reads projected `(y >> 1, x >> 1)`) and is **cropped** to
  the skip's extent — not the resample family's clamp-replicate padding: a
  skip larger than the replicated map is a shape error in the reference and is
  rejected with `Status::kInvalidDimensions`. The gated skip is added and the
  result stored `q8(f16(upsampled + gated))` tight at the skip's extent.
  All validation (nulls, maps, crop extent, weight/gain shapes, value classes,
  output and scratch capacities) completes before the first write: on any
  error `out` and `scratch` are both sentinel-untouched. Stride-aware reads
  honour `row_stride`/`plane_stride`; the kernel is allocation-free.

## Reference and provenance

Semantics re-derived from the pinned MIT reference `mauri870/dlssnr-pytorch`
commit `3beef7935a8ef976b47ab78b906e75600c52d2de`
(`dlssnr/layers/decoder.py` "DecInputUpsample", with its imported
`dlssnr/layers/vit.py` helpers `_Matmul`/`to_tokens`/`f16` and
`dlssnr/quant.py`'s `q8`). No network weights, captures, raw payloads or
binaries are used or reproduced.

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

`vr/tests/native-decoder-test.cpp` (5,477 checks at -O2 and -O0 via
`vr/test-native-decoder.sh`) validates against hand-derived answers: the
gated element (the pre-rounding discriminator `1 + 511·2^-20`, a product tie
resolved to even, overflow to ±infinity, `infinity · 0` as a canonical quiet
NaN with the platform sign left open), the store (the binary16-hop
double-rounding discriminator 68.00390625 → 64 versus direct e4m3 rounding →
72, e4m3 ties 9.5/10.5/11.5, ±448 saturation, the subnormal `0.001 → 2^-9`,
NaN canonicalisation), an end-to-end known answer (K1) with e4m3 ties in the
store and the scratch layout, the double-rounding discriminator through the
kernel (K2), a zero gain (K3), padded strides with poisoned padding (K4: the
x padding holds off-e4m3-grid values and the skip padding NaN in **both**
channels, catching kernel over-reads and validation over-scans), the
crop-to-skip-extent mapping at odd extents and exactly-2× extents (K5/K5b,
with the scratch mirror checked in full and its tail untouched), overflow/NaN
edges through the kernel (K6: the f16 overflow hop, ±448 saturation, the
canonical quiet NaN through the store), the projection's binary16 rounding in
the finite range (K7: sums that round down and up at the binary16 midpoint,
discriminating an unrounded, a truncating and a round-up projection), and
three-term accumulation (K8: term completeness; a tiny-term absorption pair
that pins float32 key-order accumulation against both reassociation and
reversal — the store cannot see the difference, so the scratch pins it). The
rejection matrix drives every validation class (nulls, dimensions, strides,
weight/gain shapes and value classes, buffer capacities, skip larger than the
replicated map) with sentinel-checked untouched `out` **and** `scratch`;
determinism and zero-allocation accounting close the suite. The discriminating
controls were mutation-verified: an unrounded/truncating/round-up projection,
a reversed or shortened accumulation loop each fail exactly at their designed
checks. An independent read-only review re-derived every literal (all correct)
and found the original suite missing the finite-range projection-rounding and
c_in ≥ 3 accumulation controls plus overstating K4's channel-1 poison and
K5's crop claims; all findings are fixed and the new controls are
mutation-verified as above. The sanitised build is skipped on this toolchain
(missing `/usr/lib64/libubsan.so.1.0.0`), a recorded environment gap; the
suite runs at -O2 and -O0 with `-Wall -Wextra -Werror -ffp-contract=off`.

## Non-vendor statement

This is not vendor code and not a reimplementation of any vendor's protected
network. It follows the MIT reference's layer semantics only and never
corresponds to NVIDIA's implementation. No proprietary weights, captures,
binaries or proprietary runtime are used or redistributed; the tree performs no
runtime or capture probes. Synthetic known answers establish synthetic
correctness only — never vendor parity.
