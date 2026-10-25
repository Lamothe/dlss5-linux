# Native CPU input-lift slice — specification and evidence

Date: 26/10/2026 (Australia/Sydney). Implements the first executable deliverable of
[vr/NATIVE-RECOVERY-ASSIGNMENT.md](NATIVE-RECOVERY-ASSIGNMENT.md): a CPU-only, typed,
native Linux RGB → 16-feature → 32-channel input lift with original synthetic frames and
weights and independently derived arithmetic controls.

**Qualification: this slice establishes synthetic arithmetic correctness of a defined
tensor pipeline only. It is not vendor correspondence, not a claim of DLSS-NR bit-exact
parity, and not evidence about NVIDIA's model or weights. The structure below is adapted
from a public MIT-licensed reimplementation's documentation of the network's input stage;
a software licence grants no rights to NVIDIA model data, and none of NVIDIA's weights,
captures, binaries or proprietary payloads are used or redistributed here.**

## Files

- `vr/native-input-lift.hpp` — the typed CPU implementation (header-only, allocation-free).
- `vr/tests/native-input-lift-test.cpp` — independent regression with known-answer,
  permutation, sensitivity, determinism and negative controls.
- `vr/test-native-input-lift.sh` — strict build + test runner (sanitised build when a
  sanitizer runtime can be linked).

```bash
bash vr/test-native-input-lift.sh
```

## Attribution

The feature composition, conditioning derivation, deterministic noise generator, stored
16→32 input-lift permutation and constants are adapted from `mauri870/dlssnr-pytorch`,
commit `3beef7935a8ef976b47ab78b906e75600c52d2de` (`dlssnr/layers/pre_post.py`,
`dlssnr/io.py`, `dlssnr/runtime.py`, `dlssnr/weights.py`), reviewed at that pin on
26/10/2026. Licence: MIT, Copyright (c) 2026 Mauri de Souza Meneguzzo. The MIT licence
text is reproduced below. Test frames and test weights are original synthetic data and
carry no third-party content. All arithmetic code here is written natively in C++; no
Python/PyTorch runtime is introduced or relied upon.

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

The pinned source documents a single-frame path only; its current-colour duplication is a
no-history stub. Nothing here implements or claims temporal history/reset behaviour.

## Tensor layout and bounds

- Input frame: interleaved RGB float32, row-major, natural orientation (`x` = column,
  `y` = row, origin top-left), values in `[0, 1]` (the reference path feeds 8-bit frames
  as `k/255`). `row_stride` is in floats between rows and must be `>= 3 * cols`; padded
  rows are accepted and ignored.
- Output: 32 channel planes, contiguous, `lifted[c * rows * cols + y * cols + x]`.
- Working extent: `1 <= rows <= 16384`, `1 <= cols <= 16384`, `rows * cols <= 2^24`;
  all products are checked in widened arithmetic before use. Mirror padding to a network
  working extent and window cropping belong to later stages, not this slice.
- Allocations: none. The caller owns every buffer; validation completes before any output
  write, so a rejected call leaves the output untouched.

## Input features (per pixel, in order)

| k | feature | source |
|---|---------|--------|
| 0..2 | three noise fields `n0, n1, n2` | `f16(noise_gain * gaussian(x, y, seed))`, `noise_gain = 1.0` |
| 3 | constant | `1.0` |
| 4..6 | centred colour R, G, B | `f16(f16(f16(rgb) - 0.5) * 0.125)` |
| 7..9 | centred colour R, G, B again | the reference's no-history stand-in where the previous frame's colour goes in temporal mode; with no history it is the current colour |
| 10..14 | conditioning (style, tone, structure, skin, other) | see below |
| 15 | zero pad | `0.0` |

The duplication at k = 7..9 is a **single-frame stub, not a temporal implementation**.

## Conditioning

Mirrors the reference `Controls` subset consumed by the input stage: `style` ∈ {0, 1, 2},
`local_tone` ∈ [0, 2], `local_structure` ∈ [0, 2], `skin_structure` ∈ [−1, 2] (negative
values follow `local_structure`), `auto_mask`, `seed` (uint32).

- With `auto_mask`: `(style/128, local_tone, 1.0, skin, local_structure)` where `skin` is
  `skin_structure` or `local_structure` when `skin_structure < 0`.
- Without `auto_mask`: `(style/128, local_tone, local_structure, -1, -1)`.

Each of the five values is rounded to the f16 grid; `style/128` is exact.

## Deterministic noise

Per pixel `(x, y, seed)`, with uint32 wraparound arithmetic (bit-exact, portable):

```
base = (x * 0x8DA6B343) ^ (seed * 0x9E3779B9) ^ (y * 0xD8163841) ^ 0x243F6A88
t    = pcg(base)                     # v = (v >> ((v >> 28) + 4)) ^ v; v * 0x108EF2D9
h    = (t >> 22) ^ t
u_i  = unit(h * multiplier_i + offset_i)   # u = (((t >> 30) ^ (t >> 8)) + 1) * 2^-24
```

with multiplier/offset pairs `(0x2C9277B5, 0xAC564B05)`, `(0xFA6DC5F9, 0x4712A88E)`,
`(0xCAA5B80D, 0x21DD796B)`, `(0x83232C31, 0x3463E0AC)`. The uniform mapping is exact in
float32 (`u ∈ (0, 1]`). The gaussian fields are Box–Muller pairs:

```
radius_a = sqrt(-2 * log(u0));  radius_c = sqrt(-2 * log(u2))
n0 = radius_a * cos(u1 * 2pi);  n1 = radius_a * sin(u1 * 2pi);  n2 = radius_c * cos(u3 * 2pi)
```

Each operation is evaluated in float32 and rounded to float32; `2pi` is the float32
nearest of 6.28318530718. **The integer hashing and uniform mapping are bit-exact
portable; the log/sqrt/sin/cos stage uses the platform libm and is deterministic per
platform but is NOT claimed bit-exact across platforms or libms.** The test suite
therefore compares noise known-answers with a small tolerance and cross-checks against an
independent scalar oracle exactly on the host.

## Stored lift permutation and matrix product

The learned 16→32 lift is stored as 512 f16 values arranged for the matrix hardware: rows
in groups of 16 channels; inside a group the channel's low three bits select a 32-element
stride and the 16 inputs are split into halves and pairs. Decoding:

```
offset(c, k) = (c / 16) * 256 + 32 * (c % 8) + 8 * ((k % 8) / 2)
             + 4 * ((c / 8) % 2) + (k % 2) + 2 * (k / 8)
lift[c][k] = stored[offset(c, k)]
```

`offset` is a bijection over 0..511 (checked exhaustively by the test). Weights must be
f16-representable and finite (validated). The lifted output is

```
lifted[c][y][x] = f16( sum_{k=0..15} lift[c][k] * features[k][y][x] )
```

with the sum accumulated left to right in float32 and one final rounding to the f16 grid
(ties to even). The e4m3 grid, the Swin body and downstream stages are not part of this
slice.

## Rounding semantics

`f16()` rounds a float32 to the nearest IEEE 754 binary16 (ties to even) in software
(no F16C/AVX512FP16 requirement): subnormals on the 2^-24 grid, values `>= 65520` round to
infinity, `2^-25` rounds to zero, NaN maps to a canonical quiet NaN. `is_f16()` validates
f16-representability. Every rounding point listed above is exercised against literals
derived independently from the IEEE 754 definition.

The 16-term lift sum is a plain `sum += a*b` float32 accumulation with one final rounding.
The regression runner compiles with `-ffp-contract=off`, which pins the build contract;
the tests themselves cannot detect FMA contraction. Note also that every product in the
sum is exact in float32 — both operands are f16-grid values (at most 11-bit mantissas:
validated for the weights, f16-rounded for the features), so the product needs at most
22 mantissa bits — which makes fused multiply-add a provable no-op for this slice
regardless of the flag. What would change results is reassociation of the 16-term sum
(`-ffast-math`), which the runner's flags exclude.

## Test evidence (26/10/2026)

`vr/test-native-input-lift.sh` — 5316 checks in the strict `-O2` build and again in the
`-O0` build (both compiled with `-ffp-contract=off`). The sanitised build is currently
skipped because this workstation has no
linkable sanitizer runtime (libasan/libubsan RPMs absent; the runner verifies the skip
reason is the missing runtime and reports the exact
link failure — observed: `cannot find /usr/lib64/libubsan.so.1.0.0`). Coverage includes:

- 30 binary16 rounding literals derived from the IEEE 754 definition (Fraction-based) and
  cross-checked with numpy float16 and the compiler's `_Float16`; canonical NaN; f16
  representability validation.
- Round-trip of every binary16 pattern (65 536 scanned; NaN payloads excluded by policy),
  the production decode pinned bit-for-bit against an independent `ldexpf`-construction
  decode oracle over all finite patterns, 200 000 pseudorandom float32 draws (NaN bit
  patterns, about 1 in 256 of the draws, skipped) against the `_Float16` oracle, and
  structured rounding points across exponents −24..16.
- Lift permutation: 16 hand-computed `offset` anchors, exhaustive bijection coverage of
  all 512 stored slots, and a full 512-entry decode-mapping check.
- Noise: 5 anchor points × 4 uniform integers derived independently with Python (exact),
  gaussian feature literals from an independent scalar C derivation (tolerance ±2^-9 for
  libm variance), exact in-file scalar oracle comparison, seed/position sensitivity.
- Centred colour chain literals (including values whose f16 rounding is non-trivial).
- Hand-computed end-to-end cases: grey 1×1 select-lift case, exact arithmetic lift
  (0.671875), f16 tie-rounding cases (`1 + 2^-11 → 1.0`, `1 + 3*2^-11 → 1.001953125`),
  conditioning/auto-mask/skin known answers, stored-weights decode path.
- Changed-input sensitivity with spatial locality (one changed colour sample moves only
  that pixel), row-stride independence, run-to-run determinism (bit-identical).
- Negative controls (null pointers, invalid dimensions/strides/extents, NaN and
  out-of-domain pixel values, invalid controls, non-f16/NaN/infinite weights, wrong buffer
  counts):
  every rejected `input_lift` call with a non-null output buffer and the rejected
  `decode_lift` call are verified to leave the output untouched (validation-before-write);
  the null-output rejection and stage-API status negatives verify status codes.
- Allocation discipline: zero allocations across decode + lift + stage calls.

## What this does NOT establish

- Not temporal: no history, no reset, no previous-frame input; the second colour copy is
  the documented single-frame stub.
- Not the network: no Swin body, attention, e4m3 quantisation, down/up-sampling, decoder,
  output conversion or image output.
- Not vendor equivalence: no vendor weights, captures or binaries were used; synthetic
  tests cannot establish correspondence to NVIDIA's implementation, and no such claim is
  made.
- Not a rights grant: the MIT licence covers the adapted software structure only.

## Next stages

1. 2×2 f16 pooling (`pool2x2`: column pairs added in float32 and narrowed to f16, the two
   row sums added in f16, the quarter exact) and the e4m3 rounding boundary.
2. Remaining pre/post-block elementwise arithmetic, then the learned-weight execution
   graph with original synthetic weights, temporal history/reset and output conversion,
   continuing toward live-input image evaluation per the assignment.
