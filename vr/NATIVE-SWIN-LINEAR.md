# Native Swin e4m3 linears

CPU-only, typed, allocation-free C++20 arithmetic for the Swin body's MLP matrix
products on the e4m3 grid: the matrix product itself (`swin_linear`, the reference's
`_linear` default path) and the e4m3 hidden (`swin_hidden`, the reference's `_hidden`
non-int4 path — expansion followed by the e4m3-rounded cubic SiLU). The `_mlp`
assembly (grouped mid projection with its f16 narrowing, contract projection and the
residual scale) is a later slice. No GPU, headset, proprietary payload, Python or
PyTorch runtime is required.

This is an independent implementation of explicitly documented arithmetic semantics,
adapted with attribution from the MIT-licensed reference implementation
[`mauri870/dlssnr-pytorch`](https://github.com/mauri870/dlssnr-pytorch) at commit
`3beef7935a8ef976b47ab78b906e75600c52d2de` (`dlssnr/layers/swin.py` `_linear` /
`_hidden`, `dlssnr/int8.py` `mm`). All weights and test values are original
synthetic values. **This is not vendor code.** Passing these tests establishes
synthetic arithmetic correctness of this implementation against independently derived
controls only — never correspondence to NVIDIA's implementation. No proprietary
weights, captures or binaries are used or implied.

## Source attribution and licence

Semantics pinned from the reference at the commit above:

> `_linear(x, weight, activate)`: `x @ weight.T` for e4m3 operands; with `activate`
> the e4m3-rounded cubic SiLU of it. On a GPU this is one float16 matrix kernel with
> the float32 accumulator rounded in its epilogue.

> `_hidden(params, quantised)`: the e4m3 hidden of the MLP, `[n, 64, 4C]`: expansion
> and activation … `return _linear(quantised, params.expand, activate=True)` (the
> non-int4 path).

> `int8.mm(a, b, site)`: `a @ b` … Off by default: with `sites` empty `mm` is a plain
> matrix product. (The INT8 W8A8 simulation is an optional experiment mode; it is not
> the default and is not implemented here.)

The full MIT licence notice (Copyright (c) 2026 Mauri de Souza Meneguzzo), also
retained in [NATIVE-INPUT-LIFT.md](NATIVE-INPUT-LIFT.md), applies to the adapted
semantics here:

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

The licence covers the referenced software's **code**; it grants no rights to
any NVIDIA model data, weights or captures, none of which are used here.

## The matrix product (`swin_linear`)

`out [rows, n] = x [rows, k] @ weight [n, k]^T`, row-major matrices with explicit
strides (element `(r, c)` at `data + r * row_stride + c`). The contract:

- **Operands**: every value must be finite and exactly on the e4m3 grid
  (`is_e4m3`), pinned by validation (`kInvalidValues` otherwise). This matches the
  reference's "e4m3 operands" and makes every product exact in float32 (e4m3
  significands give at most 8 significant bits; the exponent range cannot underflow
  or overflow float32).
- **Accumulator**: products are added to a float32 accumulator in increasing `k`
  order starting at `+0.0`, rounding to float32 at every step (round to nearest, ties
  to even). The reference's default path (`mm` with an empty `sites` set is the
  plain product `a @ b`) states no summation-order contract; the GPU path's note
  mentions a float32 accumulator rounded in the kernel epilogue, and the phrase "up
  to float32 summation order" in `int8.py` qualifies the optional INT8 W8A8
  simulation mode, not the default path. The k-order convention is this
  implementation's explicit choice and is pinned by the accumulation-order test. No
  claim is made that any particular device kernel sums in this order.
- **Bounded layout**: `rows` in `1 .. kMaxLinearRows` (2^24), `cols` in
  `1 .. kMaxLinearDim` (1024, = 4C at C = 256), `cols <= row_stride <= kMaxLinearDim`.
- **Untouched outputs**: dimensions, strides, the operand grid and the output
  capacity are all validated before the first output write; on any error `out` is
  untouched. `out_capacity` (floats accessible at `out->data`) must be at least
  `(rows - 1) * out.row_stride + n`.
- Buffer aliasing is caller discipline (never rejected; all access stays in bounds).

## The e4m3 hidden (`swin_hidden`)

`swin_hidden` computes the reference `_hidden`'s non-int4 path:
`out = e4m3(swin_activation(swin_linear(x, expand)))` — the cubic SiLU
(`swin_activation`, pinned in [NATIVE-SWIN-PRIMITIVES.md](NATIVE-SWIN-PRIMITIVES.md))
of each product, rounded to the e4m3 grid (`e4m3`, pinned in
[NATIVE-PRE-POST.md](NATIVE-PRE-POST.md)), stored as float32 e4m3 values. `out` must
be `rows x expand.rows`; the validation and untouched-output contract is that of
`swin_linear`. The int4 mixed-network variants of `_hidden` (hard-swish, folded
scales) are out of scope for this synthetic pipeline and are not implemented.

## Explicit choices and disclosed limitations

- The summation order is this implementation's explicit convention, not a claim
  about any device. The accumulation-order test pins it (see below).
- The INT8 W8A8 experiment mode of `int8.mm` and the GPU kernel path are not
  implemented; the default plain-product semantics are.
- The `_mlp` assembly (grouped mid projection, f16 narrowing `store`, contract
  projection, residual scale) is a later slice; this slice stops at the hidden.
- Operand validation is exact (bit-level grid membership) and linear in the operand
  count; the allocation-free discipline is enforced by tests.
- The sanitised build is skipped on this workstation (no linkable sanitizer runtime;
  the runner verifies the skip reason is the missing runtime and reports the linker
  error) — an environment gap, not a code result.

## Test evidence (26/10/2026)

`vr/test-native-swin-linear.sh` — 290 checks in the strict `-O2` build and again in
the `-O0` build (both compiled with `-ffp-contract=off`). Every known answer is a
hand-derived dyadic rational (the corpus lives on the e4m3 grid, so products and the
expected sums are exact or hand-rounded). Coverage includes:

- Hand-known answers: `K = 1` and `K = 2` products, sign cases, a `2 x 2 x 2`
  multi-row/multi-output-channel case with padded strides and sentinel-checked
  untouched stride padding.
- **The accumulation-order discriminator**: products `[2^16 x 128, 0.5, 0.5]` (all
  e4m3 operands: `256 * 256`, `1 * 0.5`) reach exactly `2^23` after the 128 leading
  terms; each trailing `0.5` then hits the tie-to-even midpoint (ulp 1 at `2^23`)
  and is dropped, pinning the sequential k-order float32 accumulator at `8388608`.
  Summing the two tail products first, reverse k-order or exact (float64) summation
  each give the provably distinct `8388609` (both values asserted distinct). The
  discriminator's scope is pinned too: moving a single trailing `0.5` to the front
  or transposing the two trailing products still hits the recurring tie-to-even
  midpoint and gives `8388608` (asserted in the test), so the case discriminates
  the named alternatives, not every reordering.
- Hand-known `swin_hidden` activations as exact rationals of the cubic constants
  (`-229/4096`, `229/512`, `229/256`): product `1 -> 5267/4096 -> e4m3 1.25`,
  product `-1 -> -2061/4096 -> e4m3 -0.5`, product `4 -> 229/32 = 7.15625 -> e4m3
  7.0`, product `200704` (448·448) `-> 359072 ->` saturation `448`, and product
  `0 -> 0`.
- Permutation controls on an exact corpus whose weight rows and sensitivity deltas
  are pairwise distinct (with explicit corpus guards and a hand-derived anchor row
  so the controls cannot be vacuous): row permutation permutes output rows,
  weight-row permutation permutes output channels, and a K-permutation applied to
  both operands leaves the exact sums bit-identical.
- Changed-input sensitivity with an exact delta (`+1` at one operand moves each
  output of that row by exactly the paired weight — distinct per channel — and other
  rows bit-untouched), determinism reruns (bit-identical outputs).
- Negative controls for every validation rejection class of both entry points (null
  pointers, dimensions incl. the `kMaxLinearDim`/`kMaxLinearRows` bounds and the K
  and row-match constraints, strides, short capacity, and the e4m3 operand-grid class
  with off-grid (`0.3`, `1.0625`, `2^-10`), out-of-range (`±500`) and non-finite
  (`NaN`, `±inf`) operands on both the x and weight sides), each with sentinel-checked
  untouched outputs; plus the valid boundary class (`±448`, `2^-9`, `-0.0`).
- Zero-allocation accounting across repeated `swin_linear`/`swin_hidden` calls.

## What this does NOT establish

- No vendor correspondence. The arithmetic matches the pinned MIT reference semantics
  as documented (plus this slice's explicit summation convention); it is not evidence
  about NVIDIA's implementation.
- No learned weights: every operand tested is an original synthetic e4m3 value. The
  learned-weight execution graph assembly remains later slices.
- No `_mlp` assembly, attention, temporal behaviour, live-input evaluation or GPU
  behaviour of any kind.
