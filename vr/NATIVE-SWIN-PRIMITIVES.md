# Native Swin body scalar primitives

CPU-only, typed, allocation-free C++20 arithmetic for the scalar primitives of the
native DLSS5-NR-style network's Swin body: the cubic SiLU activation, the
double-computed fused multiply-add, the deterministic reciprocal square root and the
head-dimension reciprocal L2 norm. This is the first executable slice of the
learned-weight execution graph (the Swin body); the windowed attention, the e4m3
linears, the position-bias deswizzle and the window partitioning are later slices. No
GPU, headset, proprietary payload, Python or PyTorch runtime is required.

This is an independent implementation of explicitly documented arithmetic semantics,
adapted with attribution from the MIT-licensed reference implementation
[`mauri870/dlssnr-pytorch`](https://github.com/mauri870/dlssnr-pytorch) at commit
`3beef7935a8ef976b47ab78b906e75600c52d2de` (`dlssnr/layers/swin.py`, the `_fma`,
`activation`, `_rsqrt` and `_norm_scale` functions). All synthetic vectors and test
values are original. **This is not vendor code.** Passing these tests establishes
synthetic arithmetic correctness of this implementation against independently derived
controls only — never correspondence to NVIDIA's implementation. No proprietary
weights, captures or binaries are used or implied.

## Source attribution and licence

Semantics pinned from the reference at the commit above:

> `y = rs * x + Wc . q(mid(q(act(We . q(x)))))` … `act` = cubic SiLU … the
> two nested multiply-adds are fused (one rounding each) and the whole expression
> stays in float32.

> `_fma(a, b, c)`: `a * b + c` with a single rounding to float32 (the product is
> exact in float64).

> `_rsqrt(x)`: float32 `1 / sqrt(x)` with the same bits on every device … the square
> root rounded to float32 (taken in float64, where it is exact on any device)
> followed by an exact division is the CPU's result.

> `_norm_scale(values)`: reciprocal L2 norm over the 32 head dimensions … the squares
> are accumulated by fused multiply-add in float32, in four interleaved chains over
> each of the two halves of the dimension set (even and odd dimensions); the halves
> are narrowed to f16 and added in f16. A sum that overflows f16 gives a scale of
> zero. The reciprocal root itself is not narrowed.

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

## The fused multiply-add (`swin_fma`)

`swin_fma(a, b, c)` = `a * b + c` computed exactly as the reference's `_fma`: the
product and the sum are taken in **float64** and the result is rounded **once** to
float32. This is deliberately **not** `std::fma` (which fuses the expression into a
single float32-precision rounding): the reference's float64 intermediate is part of
the semantics. For float32 `a` and the f16-exact constants `b` and `c` the product is
exact in float64; the sum is rounded to float64 and then to float32, a double
rounding this routine reproduces. The build contract is `-ffp-contract=off` (as for
every earlier stage) and the explicit double casts pin the operations regardless.

## The cubic SiLU activation (`swin_activation`)

Per value, in this exact order (float32 throughout):

1. `t = clamp(x, -4, 4)`.
2. `inner = swin_fma(|t|, -0.055908203125, 0.447265625)`.
3. `out = x * swin_fma(t, inner, 0.89453125)` — a plain float32 multiply (one
   rounding) on the **unclamped** `x`: beyond the clamp the ramp saturates but the
   outer factor does not (`activation(10) = 17.890625`, not `4 * 1.7890625`).

Sign follows `x`: `activation(-0.0) = -0.0`, and at `x = -4` the ramp fma (the second
of the two) cancels to `+0.0` (since `t * inner = -kActLinear` exactly there) so the
outer multiply carries `x`'s sign (`activation(-4) = -0.0`). Precondition:
`x` is finite. The constants are exactly `-229/4096`, `229/512` and `229/256`; every
test literal is a hand-derived dyadic rational of them.

## The reciprocal square root (`swin_rsqrt`)

float32 `1 / sqrt(x)` with the CPU bits the reference documents: the square root is
taken in float64 (correctly rounded, as IEEE 754 requires of `sqrt`), rounded to
float32, then a correctly rounded float32 division. No transcendentals are involved,
so these bits are the same on every IEEE 754 conforming platform; a GPU's rsqrt
approximation is **not** reproduced and is not claimed. Precondition: `x` is positive
and finite or `+inf` (the norm can feed an overflowed sum, which maps to a zero
scale); zero and negative inputs are outside the contract.

## The head-dimension reciprocal L2 norm (`swin_norm_scale`)

The attention head's normalisation scale from one vector of 32 head dimensions
(`kNormDim`), in this exact order:

1. Each value is squared in float64 (exact for float32 operands). The 32 slots decode
   to `(fragment f, component group g, chain c, half h)` as
   `i = f * 16 + g * 8 + c * 2 + h` (the reference's fragment layout).
2. For each `(c, h)` the four squares of that chain accumulate in the `(g, f)` order
   `(g=0, f=0), (g=0, f=1), (g=1, f=0), (g=1, f=1)`, interleaved across the chains;
   after every step the running total is **rounded to float32 and widened again**
   (the reference's `.float().double()` at each fragment accumulation).
3. Each half is `((c0 + c1) + (c2 + c3))` in float32 and narrowed to binary16; the
   two halves are added in binary16 and clamped at `kNormFloor` =
   `f16(0.000062)` = `6.198883056640625e-05` (exactly `0x1.04p-14`).
4. The scale is `swin_rsqrt` of that binary16 sum. A sum that overflows binary16 (or
   float32 in the chains) becomes an infinity and the scale is zero.

The chain/half decomposition above is this implementation's reading of the reference's
documented "four interleaved chains over each of the two halves" fragment-accumulator
order; the tests pin this reading. Precondition: values are finite (NaN is outside the
contract).

## Explicit choices and disclosed limitations

- The (g, f) loop order within a chain is pinned only where it is observable at the
  binary16 output: the accumulation discriminator below separates the documented
  stepwise-float32 rounding (with its ties to even) from a double-accumulation
  alternative, but no tested case separates a swapped (g, f) order at binary16 output
  precision — the tie-to-even grid makes the tested orders coincide. The order is
  nevertheless implemented exactly as documented.
- Random corpora do **not** separate the accumulation semantics at the binary16
  output (development probing found no randomised case distinguishing stepwise from
  double accumulation): the hand-derived discriminator is the pin, not corpus
  statistics. The test corpus checks only the documented semantics against the
  re-derivation, not the alternative.
- The oracles are scalar re-derivations written from the specification. They call no
  production routine except the shared f16 rounding helper (`f16_from_double`) and
  the shared constants, so their binary16 steps are not independent of production —
  those steps are pinned by the hand-derived literals. Otherwise they run in the same
  process against the same float32/float64 arithmetic, so bit-exact agreement is
  expected and checked exactly; the genuinely independent pins are the hand-derived
  literals.

## Test evidence (26/10/2026)

`vr/test-native-swin-primitives.sh` — 141 056 checks in the strict `-O2` build and
again in the `-O0` build (both compiled with `-ffp-contract=off`). The sanitised build
is skipped on this workstation for the same environment reason as the earlier stages
(no linkable sanitizer runtime; the runner verifies the skip reason is the missing
runtime and reports the tail of the linker error). Coverage includes:

- Activation hand-derived dyadic literals: `0`/`-0.0` sign pins, `±1`, `±2`, the
  clamp at `±4` (including the `-4` cancellation to `-0.0`), beyond-clamp values
  (`±4.5`, `±10`) pinning the unclamped outer multiply, the interior dyadic `3.75`
  and `0.125`, and a large finite saturation.
- The fma double-rounding discriminator: `(1 + 2^-23)(1 - 2^-23) + (-1)` gives
  `-2^-46` through the documented float64 path while a float32 multiply-add gives
  `0` (both asserted), plus exact cases and the cancellation sign pin.
- Reciprocal square root literals: exact-root cases (`4 → 0.5`, `1 → 1`, `0.25 → 2`,
  `6.25 → 0.4000000059604645`), definition-derived CPU bits (`0.5`, `2`, `8`, `2.5`,
  `kNormFloor → 127.01156616210938`) and the overflowed-sum `+inf → 0` path.
- Norm literals: the zero vector clamping at `kNormFloor` (`127.01156616210938`), the
  single-slot scale (`1.0`) with monotone sensitivity (`0.5 → 2.0`, `2.0 → 0.5`), the
  uniform vector (`rsqrt(8)`), below-midpoint rounding in the halves add (scale `1.0`
  versus the distinct `rsqrt(1 + 2^-10) = 0.9995120763778687`), the non-tie contrast
  (`1 + 2^-10`), the **accumulation discriminator** — two `2^-24` squares tied away
  against a `1.0` base with the half sum at the binary16 midpoint `1 + 2^-11`: the
  documented stepwise order gives scale `1.0`, the double-accumulation alternative
  `0.9995120763778687` (both asserted, proven distinct), f16-overflow vectors mapping
  to a zero scale, sign-insensitivity and bit-identical determinism reruns.
- Oracles: 100 000 randomised norm vectors (sign-randomised mantissas, exponents
  drawn from `2^-12 … 2^5`) checked
  bit-exactly against the scalar re-derivation with 1000 determinism reruns, 20 000
  randomised activation inputs across and beyond the clamp band, and 20 000
  randomised reciprocal-square-root inputs.
- Zero-allocation accounting across all four primitives.

## What this does NOT establish

- No vendor correspondence. The arithmetic matches the pinned MIT reference semantics
  as documented; it is not evidence about NVIDIA's implementation.
- Not the body: no windowed attention, window partitioning, e4m3 linears, position
  bias, softmax or MLP assembly, and no learned weights — every value tested here is
  an original synthetic scalar. The learned-weight execution graph remains later
  slices.
- No temporal behaviour, no live-input evaluation and no GPU-rsqrt parity.
