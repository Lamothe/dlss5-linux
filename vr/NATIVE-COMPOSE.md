# Native output-conversion stage

CPU-only, typed, allocation-free C++20 arithmetic for the post-block output conversion
of the native DLSS5-NR-style network: the learned 32-channel → 3-colour residual
projection and the output compose (residual add with weight 1/4, clamp, intensity
blend, and the intensity > 1 luminance-ratio amplification). No GPU, headset,
proprietary payload, Python or PyTorch runtime is required.

This is an independent implementation of explicitly documented arithmetic semantics,
adapted with attribution from the MIT-licensed reference implementation
[`mauri870/dlssnr-pytorch`](https://github.com/mauri870/dlssnr-pytorch) at commit
`3beef7935a8ef976b47ab78b906e75600c52d2de` (`dlssnr/layers/pre_post.py`, the
module-level `compose` and `PostBlock.forward`'s projection). All synthetic frames,
residuals and test weights are original. **This is not vendor code.** Passing these
tests establishes synthetic arithmetic correctness of this implementation against
independently derived controls only — never correspondence to NVIDIA's
implementation. No proprietary weights, captures or binaries are used or implied.

## Source attribution and licence

Semantics pinned from the reference at the commit above:

> Its 32 f16 output channels are projected to three colour residuals with the learned
> `out_project` (accumulated in f32); the residual is added to the input frame with
> weight 1/4, clamped to [0, 1], and blended toward the input frame by the Intensity
> control.

> `nr = clamp(frame + residual / 4)`; for intensity up to 1 the output is `frame +
> intensity * (nr - frame)`. Past 1 a plain extrapolation would push the colour
> channels apart faster than the brightness moves, so the luminance ratio `nr / frame`
> is raised to the intensity instead and the colour is rescaled to that brightness,
> which keeps the hue; the amplification is bounded to a factor of 2 either way.

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

## Residual projection (`project_residual`)

Per output channel `r` in 0..2 and pixel `(y, x)`:

```
dst_r(y, x) = sum over c = 0..31 of weights[32 * r + c] * wide(c, y, x)
```

The weights are the stored `out_project` record's first three rows: 96
f16-representable, finite values widened to float32 (the same record contract as the
lift and blend stages). Each product is a float32 multiply and the 32 terms accumulate
sequentially in ascending channel order in float32. **The reference's einsum does not
specify a contraction order**; ascending sequential accumulation is this
implementation's explicit semantics, and the tests pin it with an order-discriminating
literal (terms `1.0, 2^-24, 2^-24` with unit weights give `1.0` ascending — each
`2^-24` lands on a tie at the float32 grid point 1.0 and ties to even — while
small-terms-first would give `1 + 2^-23`). Other orders can round differently; no
claim of agreement with any particular torch backend's contraction order is made. The
build contract pins these roundings: every build compiles with `-ffp-contract=off`, so
no expression below fuses a multiply and an add.

`wide` is a channels-first `FeatureBatchView` with exactly `kLiftedChannels` (32)
planes; `dst` receives `3 * rows * cols` floats, planes in channel order, row-major
within a plane.

## Compose (`compose_pixel`, `compose`)

Per pixel, in this exact order (float32 throughout unless noted; the build contract is
`-ffp-contract=off`, so no expression below fuses):

1. `network[c] = clamp01(frame[c] + residual[c] * 0.25)` — one float32 multiply, one
   float32 add, clamp.
2. `t = (float)min(intensity, 1.0)` — the min taken in double on the double
   `intensity` input, range-guarded to `[−FLT_MAX, FLT_MAX]` (narrowing an
   out-of-float32-range double is undefined in C++), narrowed once to float32.
3. `out[c] = clamp01(frame[c] + t * (network[c] - frame[c]))` — exactly as written:
   the difference is rounded before the multiply and the add (the expression is **not**
   algebraically simplified to `network`, even at `t = 1`).
4. For `intensity <= 1` the result is `out`.
5. For `intensity > 1`: luminance `l = (out_r * 0.2126 + out_g * 0.7152) + out_b *
   0.0722` and likewise `lf` for the frame, the three products and two adds in
   float32, **accumulated left-to-right** (an explicit choice; the reference's sum
   order is not specified). Then `ratio = (l + 2^-9) / (lf + 2^-9)` in float32,
   `amplification = (float)clamp(pow((double)ratio, intensity), 0.5, 2.0)` — the pow
   evaluated in double and the clamp applied in double **before** the single narrowing
   to float32, so a double result beyond the float32 range is never converted (0.5 and
   2.0 are exact, so the narrowing after the clamp is exact) — and
   `result[c] = clamp01((out[c] * amplification) / ratio)` — product then quotient,
   left-to-right in float32.

Explicit choices where the reference is silent are marked above. The pow is evaluated
in double precision (platform libm) and narrowed once to float32: this is a **disclosed
deviation** from the reference, whose torch pow computes in float32 — it is not a
gap-fill for missing reference semantics — and like the input lift's gaussian stage the
result is **not claimed bit-exact across platforms** or against any torch backend. The
tests pin the amplification **clamp band** exactly — clamping makes the outcome
libm-independent (strong brightening at intensity 100 gives amplification 2.0 exactly,
strong darkening 0.5 exactly) — and check interior amplification cases by exact,
bit-level agreement with a scalar re-derivation of the documented formula. That
re-derivation shares no code with the production routines but runs in the same process
against the same libm, so exact agreement is expected and checked exactly; the genuinely
independent pins are the hand-derived literals above. The reference's `ratio.clamp_min(1e-6)` cannot
bind under this stage's validated inputs: with frame and blended colours in `[0, 1]`
the ratio lies in `[2^-9 / (1 + 2^-9), (1 + 2^-9) / 2^-9]` (the luminance is at most
the weight sum ≈ 1), far from `1e-6`, so that comparison's rounding semantics are
unobservable through this stage. `pow(1, y) == 1` is exact for any `y`, so a zero
residual preserves the frame exactly at every intensity.

Validation completes before any output write (dimensions, strides, weight count and
f16-representability, `intensity` finiteness, the frame's documented `[0, 1]` range
and finite-value scans of both inputs, output size), so every rejected call leaves the
output buffer untouched. The frame's `row_stride` contract is the input lift's
(`3 * cols <= row_stride <= 3 * kMaxCols`); the residual's is the blend's
(`cols <= row_stride <= kMaxCols`, `rows * row_stride <= plane_stride <= kMaxRows *
kMaxCols`, rejected as `kInvalidStride` when the dimensions are valid). Both stages
are allocation-free. The output buffer must not overlap either input; overlapping
writes are never rejected (all access stays in bounds) and can silently corrupt the
result. Any finite `intensity` has defined semantics — negative values extrapolate
away from the network colour. `compose` rejects non-finite `intensity` as
`kInvalidControls`; `compose_pixel` performs no validation and **requires** its
documented preconditions — frame values in `[0, 1]`, finite residuals and finite
`intensity` (a violated precondition has no specified outcome; through `compose` the
preconditions are always enforced).

## Test evidence (26/10/2026)

`vr/test-native-compose.sh` — 60 137 checks in the strict `-O2` build and again in the
`-O0` build (both compiled with `-ffp-contract=off`). The sanitised build is skipped on
this workstation for the same environment reason as the earlier stages (no linkable
sanitizer runtime; the runner verifies the skip reason is the missing runtime and
reports the tail of the linker error). Coverage includes:

- `compose_pixel` hand-derived literals: the zero-residual identity at intensities
  0/0.5/1/100, intensity 0, intensity 1 with clamping at both ends, exact half- and
  full-intensity blends (`0.375`, `0.75`), negative-intensity extrapolation (`0.0`),
  the amplification identity at ratio 1 (intensities 1.0001 and 37.5), and the
  amplification clamp band pinned exactly (brightening → 2.0, darkening → 0.5).
- `compose_pixel` fuzz: 20 000 randomised pixels across intensities 0, (0, 1), 1 and
  (1, 4) against the scalar re-derivation — exact agreement on every path (the
  re-derivation mirrors the documented clamp-in-double-then-narrow pow, so bit-exact
  agreement is expected; the independent pins are the hand literals above) —
  plus changed-input sensitivity (one channel's residual moves only that channel) and
  bit-identical determinism reruns.
- `project_residual` hand-derived literals: identity rows, the ascending-accumulation
  order discriminator (`1.0`, not `1 + 2^-23`) with the discarded alternative proven
  distinct, and a mixed-gain known answer (`4 * 0.5 + 0.25 * -2 = 1.5`).
- `project_residual` oracle: 2x2 batches of pseudorandom f16-grid weights and values
  against a separately written scalar re-derivation (same-process, as above),
  bit-identical determinism reruns, and
  a channel-permutation control (permuted weight rows permute the residuals exactly).
- Batch-level compose known answers (hand-computed pixel with a padded-stride residual,
  frame preservation under zero residual, and batch == pixel path at intensity 2.5),
  interleaved-frame plumbing, and padded-stride independence.
- Negative controls for both entry points: every rejection path (null data/weights/
  output, wrong channel counts, invalid extents and pixel counts, invalid row/plane
  strides — both lower and upper bounds where enforced — wrong weight count, non-f16
  or non-finite weights, non-finite intensity, out-of-range or non-finite frame
  values, non-finite residual values, short output) verified to return the documented
  `Status`, with sentinel-checked output-untouched assertions on representative paths
  of each rejection class (the structural guarantee is that validation completes
  before the first output write).
- Zero-allocation accounting across both entry points.

## What this does NOT establish

- No vendor correspondence. The arithmetic matches the pinned MIT reference semantics
  as documented; it is not evidence about NVIDIA's implementation.
- No cross-platform bit-exactness for the amplification's pow (platform libm), and no
  agreement is claimed with any particular torch backend's float32 reduction orders.
- No temporal behaviour and no network body: history/reset and the Swin
  body/decoder stages remain later stages. This completes only the output-conversion
  tail of the post-block.
