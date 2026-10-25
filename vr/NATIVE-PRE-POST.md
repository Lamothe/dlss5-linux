# Native pre/post stages — e4m3 rounding boundary and 2×2 f16 pooling

Status: implemented and tested (CPU-only, typed C++20) in `vr/native-pre-post.hpp`,
tested by `vr/tests/native-pre-post-test.cpp` and `vr/test-native-pre-post.sh`. This is
the second executable stage of the recovery work order
([NATIVE-RECOVERY-ASSIGNMENT.md](NATIVE-RECOVERY-ASSIGNMENT.md)); the input-lift slice is
in [NATIVE-INPUT-LIFT.md](NATIVE-INPUT-LIFT.md).

This is an independent implementation of explicitly documented arithmetic semantics.
It is **not** vendor code. Passing synthetic tests establishes synthetic correctness of
this implementation against independently derived controls only — never correspondence
to NVIDIA's implementation. No proprietary weights, captures or binaries are used or
implied; test data is original synthetic material.

## Attribution

The semantics are adapted from `mauri870/dlssnr-pytorch`, commit
`3beef7935a8ef976b47ab78b906e75600c52d2de` (`dlssnr/quant.py` `q8`,
`dlssnr/layers/swin.py` `_pool`/`pool2x2`), reviewed at that pin on 26/10/2026. Licence:
MIT, Copyright (c) 2026 Mauri de Souza Meneguzzo. The MIT licence text is reproduced
below. All arithmetic code here is written natively in C++; no Python/PyTorch runtime is
introduced or relied upon. The upstream source's own documentation is:

> The 8-bit floating point format the network computes in. The network is FP8 e4m3 at
> every operation boundary: a value is rounded to the nearest e4m3 number (ties to
> even), saturating at ±448, and the next operation reads it back exactly.

> 2x2 mean of a block's unrounded output … Pixel pairs along the columns are added in
> float32 and narrowed to f16, the two row sums are added in f16, and the quarter is
> exact.

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

## e4m3 rounding boundary

The network computes in FP8 **e4m3fn**: 1 sign bit, 4 exponent bits (bias 7), 3
mantissa bits; no infinities — the codes `0x7f`/`0xff` are NaN. The grid is
`(1 + m/8) · 2^(e−7)` for `e` in 1..15 (`m` in 0..7 except `m = 7` only at `e = 15`,
which is the NaN code), and a fixed `2^-9` subnormal step below `2^-6` (`m · 2^-9`,
`m` in 0..7). Maximum magnitude `448 = 1.75 · 2^8`. Every e4m3 value is exactly
representable in float32.

`e4m3_bits(value)` rounds a float32 to the nearest e4m3 code (ties to even) and
`e4m3(value)`/`e4m3_from_bits(raw)` map between codes and float32 values:

- The reference **clamps to ±448 before rounding**, so anything with magnitude `>= 448`
  saturates to ±448 (`0x7e`/`0xfe`) — including ±infinity, which saturates rather than
  becoming NaN. Values just below 448 round on the normal grid (`432` is the midpoint
  of `416`/`448` and ties to the even mantissa, 448).
- Subnormal results use the `2^-9` step with ties to even (`2^-10 → 0`,
  `3 · 2^-10 → 2^-9`); `2^-6 − 2^-10` ties up to the min normal `2^-6`.
- NaN maps to the canonical code `0x7f` regardless of sign ("NaN is not preserved" in
  the reference); `0x7f`/`0xff` decode to a canonical quiet NaN.

## 2×2 f16 pooling (`pool2x2`)

A 2×2 mean of a plane's **unrounded** float32 values (the block accumulator), producing
`rows/2 × cols/2` values **on the e4m3 grid**. For a block

```
a b
c d
```

the exact arithmetic is, in order:

1. `s0 = f16(a + b)`, `s1 = f16(c + d)` — pixel pairs across the columns (along each
   row) added in float32 and narrowed to f16;
2. `total = f16(s0 + s1)` — the two row sums added in f16 (computed in float32 and
   narrowed; for f16 addends the float32→f16 double rounding is benign and equals a
   strict binary16 add);
3. `out = e4m3(total * 0.25)` — the quarter is exact (a power-of-two scale in float32),
   then the result is rounded to the e4m3 grid.

The grouping order is observable: for `[4.25, 2^-9; 0, 2^-9]` the documented order gives
`f16(4.25 + 2^-9) = 4.25` twice over and `e4m3(1.0625) = 1.0` (both ties to even), where
a column-first grouping would give `1.125`; the test pins `1.0`. Extents must be even
(the pipeline pads to even working extents first); the row stride is in floats and must
be `>= cols` and `<= kMaxCols`. Validation completes before any output write and the
stage is allocation-free.

## Test evidence (26/10/2026)

`vr/test-native-pre-post.sh` — 198 checks in the strict `-O2` build and again in the
`-O0` build (both compiled with `-ffp-contract=off`). The sanitised build is skipped on
this workstation for the same environment reason as the lift suite (no linkable
sanitizer runtime; the runner verifies the skip reason is the missing runtime and
reports the exact linker error). Coverage includes:

- e4m3 hand-derived encode literals (34 cases: zeros, subnormal ties, min normal,
  normal-grid ties, saturation, infinity saturation) and hand-derived decode literals,
  each also checked against the independent oracle; canonical NaN encode/decode;
  `is_e4m3` positive and negative cases.
- Exhaustive code sweep: all 256 codes decode bit-identically through production and
  the independent oracle, and every non-NaN code round-trips through production encode;
  structured boundary probes and 200 000 pseudorandom floats across a wide exponent
  range against the brute-force oracle.
- Independent oracles: the e4m3 oracle is a brute-force nearest-code search over the
  grid decoded straight from the format definition with `ldexpf` (ties to even code
  parity, saturating), and the pooling oracle narrows through the compiler's `_Float16`
  conversion path — no production code is shared. No external float8 library is
  installed on this workstation (numpy has no float8 dtype; no ml_dtypes/torch), so the
  definition-derived oracle and hand literals are the reference.
- Pooling known answers: exact block (`2.5`), the grouping discriminator (`1.0`, not
  `1.125`), an f16 tie in the column-pair sum (`0.5`), negatives (`-2.5`), overflow
  saturation (`±448`), NaN propagation, and a hand-computed 2×4 plane (`3.5`, `5.5`).
- Full-plane oracle comparison on a padded-stride plane, run-to-run determinism
  (bit-identical) and row-stride independence.
- Negative controls: null pointers, zero/negative/odd extents, oversized extents,
  `rows*cols` beyond `kMaxPixels`, bad strides, null output and short output buffer —
  each rejected call with an output buffer is verified untouched
  (validation-before-write); a valid call still succeeds afterwards.
- Allocation discipline: zero allocations across `e4m3`/`pool2x2` calls.

## What this does NOT establish

- Not the network: no Swin body, attention, learned projections, decoder, temporal
  history/reset, output conversion or image output. The pool reads a block's unrounded
  accumulator; the Swin bodies that produce such accumulators are later stages.
- Not vendor equivalence: synthetic correctness only; no correspondence to NVIDIA's
  implementation is claimed or implied by these tests.
- Not a rights grant: the MIT licence covers the adapted software structure only.
