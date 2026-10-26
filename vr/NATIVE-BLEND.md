# Native post-block blend stage

CPU-only, typed, allocation-free C++20 arithmetic for the post-block input blend of the
native DLSS5-NR-style network: the 2x replication of the half-resolution decoder output
with replicate padding, and the per-channel f16 mix with the full-resolution skip. No
GPU, headset, proprietary payload, Python or PyTorch runtime is required.

This is an independent implementation of explicitly documented arithmetic semantics,
adapted with attribution from the MIT-licensed reference implementation
[`mauri870/dlssnr-pytorch`](https://github.com/mauri870/dlssnr-pytorch) at commit
`3beef7935a8ef976b47ab78b906e75600c52d2de` (`dlssnr/layers/pre_post.py`, the `_mix`
function and `PostBlock.blend`). All synthetic frames, gains and test weights are
original. **This is not vendor code.** Passing these tests establishes synthetic
arithmetic correctness of this implementation against independently derived controls
only — never correspondence to NVIDIA's implementation. No proprietary weights, captures
or binaries are used or implied.

## Source attribution and licence

Semantics pinned from the reference at the commit above:

> `f16(skip * skip_gain + f16(up * main_gain))`, the skip product fused into the add
> (float64 holds it exactly).

> Replicate `x` 2x (repeating the last row/column if the skip is larger) and mix in the
> skip.

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

## Elementwise mix (`mix_element`)

Per element, in this exact order:

1. `main = f16(up * main_gain)` — the product computed in float32 (the reference
   multiplies two float32 tensors), then one rounding to the binary16 grid.
2. `mix = f16(skip * skip_gain + main)` — the skip product computed **exactly** in
   float64 (two float32 operands need at most 48 significand bits) and fused into a
   single float64 add with `main`, exactly as the reference's float64 form does.

The float64 path matters: a float32 form of the fused sum is **not** rounding-equivalent.
For example `1037.5 - 1.5·2^-23` rounds to 1038 through float32 (the float32 step lands
exactly on the binary16 midpoint `1037.5`, whose tie-to-even goes up) but to 1037
directly. The final double → binary16 rounding (`f16_from_double`) is therefore done
directly in double arithmetic as well: explicit ties-to-even rounding on the
`2^-24` subnormal grid below `2^-14` and on the 11-bit normal grid up to 65504, with
the overflow band `(65504, 65520)` rounding to 65504 and `>= 65520` to infinity (the
65520 tie overflows, matching the reference's IEEE conversion and `f16_bits`). NaN maps
to the canonical binary16 NaN with the sign dropped; the reference's conversion
preserves NaN signs and payloads, but the stage validates its inputs finite so this
policy difference is unreachable through `blend`. `mix_element` itself is a total pure
function for any float arguments.

The product pre-rounding is observable. With `up = 821·2^-9`, `main_gain = 5·2^-3` the
product `4105·2^-12 = 1 + 2^-9 + 2^-12` sits a quarter ulp above the binary16 grid
point `1 + 2^-9` (the grid in `[1, 2)` is `2^-10`) and rounds down to it; adding the
exact skip term `2·2^-12` then lands exactly on the midpoint `4106·2^-12 =
1 + 2.5·2^-10`, which ties to even down to `1 + 2^-9 = 1.001953125`. A semantics that
skipped the product's f16 rounding would sum `4107·2^-12`, above that midpoint, and
round up to `1 + 3·2^-10 = 1.0029296875`. The tests pin `1.001953125`.

## Blend (`blend`)

`FeatureBatchView` describes a channels-first batch: one plane of `rows × cols` floats
per channel, natural orientation (x = column, y = row), `row_stride` in floats between
rows (`cols <= row_stride <= kMaxCols`) and `plane_stride` in floats between planes
(`rows * row_stride <= plane_stride <= kMaxRows * kMaxCols`; `kMaxRows = kMaxCols =
16384`, shared with the input lift). Any stride outside those bounds is rejected with
`Status::kInvalidStride` (dimension failures take precedence) before any output write.
`up` (half resolution) and `skip` (full resolution) must share the channel count
(1..32; the pipeline uses 32) but their spatial extents need not match. `main_gain` and
`skip_gain` hold exactly one f16-representable, finite gain per channel.

The reference's three steps — 2x2 expansion (each value repeated over a 2x2 block),
replicate padding of the last row/column where the skip is larger than 2x, crop where
the skip is smaller — reduce exactly to:

```
up2x(y, x)    = up(min(y / 2, up_rows - 1), min(x / 2, up_cols - 1))
dst(c, y, x)  = mix_element(up2x(c, y, x), skip(c, y, x), main_gain[c], skip_gain[c])
```

for `y` in `[0, skip_rows)`, `x` in `[0, skip_cols)`, `c` in `[0, channels)`. The output
is `channels × skip_rows × skip_cols` floats, planes in channel order, row-major within
a plane, in the caller's buffer.

Validation completes before any output write (dimensions, strides, gain count and
f16-representability, finite-value scans of both batches, output size), so every
rejected call leaves the output buffer untouched. The stage is allocation-free. The
output buffer must not overlap either input batch; overlapping writes are never
rejected (all access stays in bounds) and can silently corrupt the mix.

## Test evidence (26/10/2026)

`vr/test-native-blend.sh` — 403 checks in the strict `-O2` build and again in the
`-O0` build (both compiled with `-ffp-contract=off`). The sanitised build is skipped on
this workstation for the same environment reason as the earlier stages (no linkable
sanitizer runtime; the runner verifies the skip reason is the missing runtime and
reports the tail of the linker error). Coverage includes:

- `f16_from_double` hand-derived literals (zero signs, the 65504/65520 overflow band
  including the guard-sensitive `65519.999999999`, the `2^-25` subnormal tie to even
  zero and its sign, and the binade-top rounding carry `32759/32760/32761 -> 32752,
  32768, 32768`) plus 200 000 pseudorandom double bit patterns and targeted
  boundary neighbourhoods (including the `32760` binade top) against the independent
  compiler `_Float16` oracle (verified
  IEEE RNE with overflow to infinity on this toolchain).
- `mix_element` hand-derived literals: exact path, the product pre-rounding tie
  discriminator (`1.001953125`, not `1.0029296875`), a sum-rounding literal
  (`2.00390625`, where the pre-rounding is not observable in the result), the subnormal
  `3·2^-25` tie to `2^-23`, the fused-sum overflow
  band (65504 stays finite, 65520 overflows to signed infinity), main-product overflow
  and the canonical-NaN policy — plus 200 000 pseudorandom f16-grid quadruples against
  the independent `_Float16`/double oracle (this fuzz is what caught a float32-shortcut
  double-rounding defect in `f16_from_double` during development).
- `blend` known answers: 1x1 replication, exact-2x block replication, crop of a larger
  up, replicate-pad + crop of a smaller up, a dual-clamp case (2x2 up over a 6x5 skip,
  where output rows `y >= 4` must repeat the last up row and column `x = 4` the last up
  column — hand-computed values distinguish clamp-to-last from clamp-to-first or
  zero-fill in both axes), two-channel distinct-gain mixing, and the
  tie discriminator through the full stage.
- An oracle re-deriving the closed-form spatial reduction from the reference's three
  explicit steps (2x2 expansion, replicate padding, crop) with `_Float16` mixing, over
  2000 randomised configurations (1–4 channels, mismatched extents, padded strides,
  random f16-grid values and gains), with determinism (bit-identical reruns) checked
  each iteration. The oracle shares the closed form with production rather than
  materialising the expansion/pad/crop intermediates; that reduction is pinned
  independently by the hand-computed crop, replicate-pad+crop and dual-clamp known
  answers above.
- 32-channel permutation control with channel-coded inputs and distinct per-channel
  gains (no channel may observe another channel's inputs or gains), changed-input
  sensitivity with exact expected difference sets (one up element feeds exactly its 2x2
  replicated block; one skip element feeds exactly its own output), stride independence
  (padded equals tight), and zero-allocation accounting.
- Negative controls: every rejection path (null data/gains/output, zero or oversized or
  mismatched channels, invalid extents and pixel counts, invalid row/plane strides —
  both lower and upper bounds, for both batches — wrong gain count, non-f16 or
  non-finite gains, NaN/infinite batch values, short
  output) verified to return the documented `Status` and leave the output untouched.

No external binary16 or float64-rounding reference beyond the compiler `_Float16` path
is claimed; the `_Float16` oracle's overflow behaviour was probed before use.

## What this does NOT establish

- No vendor correspondence. The arithmetic matches the pinned MIT reference semantics
  as documented; it is not evidence about NVIDIA's implementation.
- No temporal behaviour: the skip is consumed as-is; history/reset is a later stage.
- No output conversion: the post-block's `compose` (residual projection, intensity
  blend and clamp) is the later output-conversion stage.
- The reference's learned gains come from trained weights; here all gains are original
  synthetic values.

## Next stages

1. Output conversion (`compose`: projected residual added with weight 1/4, clamp, the
   Intensity blend and the bounded luminance-ratio extrapolation).
2. The learned-weight execution graph (Swin body) with original synthetic weights.
3. Temporal history/reset, then live-input/image native evaluation per the assignment.
