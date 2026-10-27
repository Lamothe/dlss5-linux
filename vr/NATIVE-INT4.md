# Native int4 mixed-network operand path

`vr/native-int4.hpp` implements the int4 mixed network's runtime operand
path: the Walsh-Hadamard input quantiser, the exact-integer accumulator with
its float64 affine, the mixed int4 + e4m3-tail contraction, and the Swin
site's hard-swish and fold-constant chain. CPU-only, allocation-free; no
GPU, headset, proprietary payload, Python or PyTorch runtime required. No
vendor code and no vendor correspondence.

## Contents

* `Int4MatrixView` — the decoded matrix record (the host's weight-file
  boundary, in the same declared style as `native-eval.hpp`'s `ffwd`
  matrices): `q [rows, cols]` int8, `scale`/`constant [rows]`, `aq [cols, 2]`
  (1/step, zero), the Hadamard `block`, and the Swin sites' `half` form. The
  `.fix`/`.mu`/`.aq` recovery tables and their file decode stay the host's
  duty: those tables hold no weight value (the int4 weights are GPTQ
  solutions that cannot ship) and no table data appears in-tree.
* `int4_fwht` — the unnormalised Walsh-Hadamard of every `block` consecutive
  values (float32, the ascending butterfly — the shaders' order).
* `int4_quantiser_params` / `int4_quantise` —
  `code = clamp(rint(H(x) * gain + zero), 0, 15) - 8`: `gain = (1/step) *
  (1/sqrt(block))` in float32 (the `half` form keeps `1/step` at its
  binary16 rounding with no sqrt factor), `zero` the zero point rounded to an
  integer by the `1.5 * 2^23` trick (ties to even for |value| < 2^22; the
  `half` form keeps its binary16 rounding), and the rint is
  round-half-to-even of the fused product-plus-sum taken exactly in float64
  (the pinned form), computed here mode-independently. The clamp precedes the
  -8; NaN maps to code 0 and +inf saturates to 15.
* `int4_accumulate` — `y = s * (q . x_int4) + c`: the integer dot is exact in
  float32 in any order under the enforced bounds (`|q * x| <= 1016`,
  `cols <= 2^14` keep every partial sum below 2^24); the affine is evaluated
  in float64 and stored once (the `half` form rounds `s` and `c` through
  binary16 first). `q` is accepted as int8 as delivered — the pinned's escape
  corrections wrap on its int8 cast, so a [-8, 7] enforcement would reject
  legitimate records.
* `int4_mixed_contraction` — the mixed ViT contraction: the int4 channels'
  product from the quantiser's codes plus the e4m3 tail's product (exact
  e4m3 values through the permutation), summed left to right in float32 (the
  native deterministic contract; the pinned's matmul order is unspecified).
* `hard_swish` / `swin_fold_constant` — the Swin site's activation
  (`x * clamp(f32(f64(x * A) + B), 0, 1)`, (A, B) = (0.187367, 0.500139) at
  C=32 and (0.182412, 0.500465) wide; the double product is exact and the sum
  is exact wherever the gate does not saturate, so after the clamp this
  equals the single-rounding fma form) and the settings chain
  `NR_HS_FOLDPP` (blocks 0 and 70) / `NR_HS_FOLD32` (C=32) / `NR_HS_FOLD`
  (wide), each level defaulting to the wide value below it, gated by
  `NR_HS_FOLD_BLOCKS` (absent or empty = fold everywhere, as the pinned
  parsing does; an unlisted block drops to 0.0 even for blocks 0/70).

## Reference and provenance

Semantics re-derived from the MIT reference `mauri870/dlssnr-pytorch`
commit `3beef7935a8ef976b47ab78b906e75600c52d2de` (`dlssnr/int4.py`),
itself following DLSSNR-AMD's `i4_obtain` and shaders operation for
operation. No network weights, recovery tables, captures, raw payloads or
binaries are used or reproduced.

## MIT licences (reference-derived sections)

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

The int4 semantics follow DLSSNR-AMD (MIT licensed), Copyright (c) 2026
mochizuki0323, whose licence text accompanies that project.

## Known-answer test

`vr/tests/native-int4-test.cpp` (83 checks at -O2 and again at -O0 via
`vr/test-native-int4.sh`) pins the path with hand-derived answers: the
Hadamard butterflies (`fwht([1,2,3,4]) = [10,-2,-4,0]` and a second
independent group), the 1.5 * 2^23 ties-to-even (2.5 -> 2, 3.5 -> 4,
-1.5 -> -2, -0.5 -> 0), the gain/zero derivation in all three forms
(non-half block 1 / block 4, the binary16 half form with its absent sqrt and
magic), the clamp-before-minus-8 code form (including NaN -> -8 and
+inf -> 7), the exact integer dot with the float64 affine (-50 * 0.5 + 0.25 =
-24.75), the half form's observable binary16 hop (hand-derived
-4.94976806640625 from f16(0.1f)/f16(0.049f) where the non-half form gives
about -4.951), the permuted mixed int4 + e4m3-tail contraction (6.5
hand-derived), the hard-swish clamp directions, and the fold-constant chain
including the gating beating the first/last selection, the empty listed set
and the null-list-as-absent form; plus the rejection matrix with
sentinel-untouched outputs, determinism and zero-allocation accounting with
the repo's full allocation-override block. Independent read-only review
traced the butterfly order against the pinned `fwht` verbatim (an 8-value
trace), confirmed the gain/zero/quantise/accumulate/contraction/fold
semantics against the pinned code and its own tests, and raised ten finding
groups, all fixed and reverified: the wrong exactness bound (2^18, now
enforced as the 2^14 column bound with the int8-q premise), the magic's
stated range (2^22), the hard-swish exactness claim, the UB float-to-int cast
on non-finite values (now clamped on the double with a mode-independent
half-even rounding), the fold chain's null-deref/empty-string semantics, the
missing sibling-style bounds, the q-range enforcement that would reject the
pinned's escaped records (dropped; q is int8 as delivered), a tautological
half-form test (now a hand-derived literal), stale comments and the
decode-side CLIPS table (moved to the host boundary's description), and the
reference copyright line. The sanitised build is skipped on this toolchain
(missing `/usr/lib64/libubsan.so.1.0.0`), a recorded environment gap.

## Non-vendor statement

This is not vendor code and not a reimplementation of any vendor's protected
quantised network. It follows the MIT reference's int4 contracts only and
never corresponds to NVIDIA's or AMD's implementation. No proprietary
weights, recovery tables, captures, binaries or proprietary runtime are used
or redistributed; the tree performs no runtime or capture probes. Synthetic
known answers establish synthetic correctness only — never vendor parity.
