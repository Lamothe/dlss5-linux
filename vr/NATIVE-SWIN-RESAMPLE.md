# Native Swin resampling layers

CPU-only, typed, allocation-free C++20 implementation of the resampling layers
around the Swin block body: the `ds` variant's pooled learned `C -> 2C`
projection (`swin_downsample`), the `upsample` variant's learned `2C -> C`
projected nearest-2x gated blend (`swin_upsample_blend`) and the thin `SwinLayer`
compositions (`swin_layer_plain`, `swin_layer_upsample`), with the elementwise
gated sum (`gated_sum_element`), the 1x1 projection
(`swin_project_dot`/`swin_project_narrow`/`swin_project_store`) and the block
scratch mirror (`swin_block_scratch_needed`). The stages come from the completed
slices: the e4m3 boundary and pooling ([NATIVE-PRE-POST.md](NATIVE-PRE-POST.md)),
the post-block blend primitives ([NATIVE-BLEND.md](NATIVE-BLEND.md)) and the
Swin block composition ([NATIVE-SWIN-BLOCK.md](NATIVE-SWIN-BLOCK.md)). The
encoder-decoder network assembly, temporal history/reset and live-input
evaluation remain later slices. No GPU, headset, proprietary payload, Python or
PyTorch runtime is required.

This is an independent implementation of explicitly documented arithmetic
semantics, adapted with attribution from the MIT-licensed reference
implementation
[`mauri870/dlssnr-pytorch`](https://github.com/mauri870/dlssnr-pytorch) at
commit `3beef7935a8ef976b47ab78b906e75600c52d2de` (`dlssnr/layers/swin.py`, the
`downsample`, `upsample_blend`, `_project`, `_pad_to`, `_gated_sum`, `pool2x2`
and `SwinLayer` sections, using the previously pinned and adapted `swin_block`,
`store`, `round8`, `narrow` and `q8` semantics of the earlier slices). All test
shapes and values are original. **This is not vendor code.** Passing these
tests establishes synthetic arithmetic correctness of this implementation
against independently derived controls only — never correspondence to NVIDIA's
implementation. No proprietary weights, captures or binaries are used or
implied.

## Source attribution and licence

Semantics pinned from the reference at the commit above (`dlssnr/layers/swin.py`).
The resampling section and the layer wrapper:

> ```python
> @fuse
> def _pool(accumulator: torch.Tensor) -> torch.Tensor:
>     pairs = (accumulator[..., 0::2] + accumulator[..., 1::2]).to(torch.float16)
>     total = pairs[..., 0::2, :] + pairs[..., 1::2, :]
>     return q8(total.float() * 0.25)
>
> def _project(weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
>     """1x1 convolution ``[N, K] . [1, K, h, w]`` in float32, narrowed to f16 (still float32)."""
>     return narrow(torch.einsum("nk,bkhw->bnhw", weight, x))
>
> def _pad_to(x: torch.Tensor, rows: int, columns: int) -> torch.Tensor:
>     return F.pad(x, (0, columns - x.shape[-1], 0, rows - x.shape[-2]))
>
> def downsample(params: SwinParams, resample: torch.Tensor, x: torch.Tensor, shift_x: int, shift_y: int,
>                rows: int, columns: int) -> tuple[torch.Tensor, torch.Tensor]:
>     """A block followed by pooling and the learned ``C -> 2C`` projection.
>
>     Returns ``(next, body)``: the projected pooled map zero padded to ``rows x columns`` and the block's own
>     e4m3 output.
>     """
>     wide, acc = swin_block(params, x, shift_x, shift_y, accumulator=True)
>     pooled = store(torch.einsum("nk,bkhw->bnhw", resample, pool2x2(acc)))
>     return _pad_to(pooled, rows, columns), round8(wide)
>
> @fuse
> def _gated_sum(skip: torch.Tensor, gain: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
>     """``f16(skip * gain + other)`` with the product and sum taken in float64 (one rounding to f16)."""
>     return (skip.double() * gain.double() + other.double()).to(torch.float16).float()
>
> def upsample_blend(project: torch.Tensor, gain: torch.Tensor, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
>     """``f16(skip * gain + P(x))`` on the skip's grid: ``P`` is the learned ``2C -> C`` projection of the
>     coarser level, replicated 2x in both directions. Returns the f16-exact blend ``[1, C, rows, columns]``."""
>     projected = _project(project, x)
>     rows, columns = skip.shape[-2:]
>     index_r = (torch.arange(rows, device=x.device) // 2).clamp(max=projected.shape[-2] - 1)
>     index_c = (torch.arange(columns, device=x.device) // 2).clamp(max=projected.shape[-1] - 1)
>     replicated = projected[:, :, index_r][:, :, :, index_c]
>     return _gated_sum(skip, gain.reshape(1, -1, 1, 1), replicated)
> ```
>
> ```python
>     @torch.no_grad()
>     def forward(self, x: torch.Tensor, skip=None):
>         spec = self.spec
>         if spec.variant == "ds":
>             return downsample(self.params, self.resample, x, spec.shift_x, spec.shift_y, spec.out_w, spec.out_h)
>         if spec.variant == "upsample":
>             skip = skip[1] if isinstance(skip, tuple) else skip
>             blend = upsample_blend(self.resample, self.gain, x, skip)
>             if spec.channels == 32:
>                 return round8(swin_block(self.params, blend, spec.shift_x, spec.shift_y, x_wide=blend))
>             return round8(swin_block(self.params, round8(blend), spec.shift_x, spec.shift_y))
>         return round8(swin_block(self.params, x, spec.shift_x, spec.shift_y))
> ```

`pool2x2` (pinned to `_pool`) is the earlier slice documented in
[NATIVE-PRE-POST.md](NATIVE-PRE-POST.md); this slice calls it unchanged. The
`upsample` variant's `resample` buffer is the `project` matrix here and its
`upsample_gain` row is the `gain` row; the reference's gain fallback merge from
the residual table is a weight-loading concern and is out of scope (the operand
is passed explicitly). The int4 mixed-network path (`params.i4`) is out of
scope: this slice implements the eager default path only.

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

The MIT licence covers the referenced software's **code**; it grants no rights
to any NVIDIA model data, weights or captures, none of which are used here.

## Arithmetic semantics

All operands are passed explicitly (original synthetic values in the tests; no
learned weights are used or implied). Feature maps are
`FeatureBatchView` (`[C, rows, cols]`, plane/row strides); the weight matrices
are `MatrixView` on the e4m3 grid (`resample` `[2C, C]`, `project` `[C, 2C]`)
or exactly binary16 (`gain` `[1, C]`).

- **`gated_sum_element(skip, gain, other)`** (`_gated_sum`): `f16(skip*gain +
  other)` with the product and the sum computed in float64 and the whole
  expression rounded to binary16 **once** (ties to even, `f16_from_double`'s
  documented zones). A float32 form of the sum is not rounding-equivalent (the
  tests pin the 1037-not-1038 discriminator). Extreme operand sets round to
  ±infinity exactly as the reference's float16 conversion (the (65504, 65520)
  band rounds to 65504, |sum| >= 65520 to infinity); the result is never NaN
  for finite f16 operands.
- **`swin_project_dot`/`_narrow`/`_store`** (`_project`/`narrow`/`store`):
  `sum_k weight[k] * map[k, y, x]` in float32 with the products exact (e4m3
  operands) accumulated in the explicit k-ascending sequential float32
  accumulator this campaign fixes as a convention — the reference's
  `torch.einsum` states no summation-order contract, so no order
  correspondence is claimed. `_narrow` rounds the sum to binary16 once;
  `_store` applies the stored-result double rounding f16 -> e4m3 (the pinned
  367.875 -> 384 case, where a direct e4m3 rounding gives 352).
- **`swin_downsample`** (`downsample`'s `ds` variant): the block in accumulator
  mode, `pool2x2` per plane, the stored-result projected pooled map placed
  top-left in `next` resized to `out_rows x out_cols` (zero filled where the
  pooled extent is smaller — `_pad_to`'s zero padding — cropped bottom/right
  where it is larger — the reference's negative `_pad_to` padding crops),
  and `body = round8(wide)`. The input extents are even (pool2x2's contract).
  The scratch holds the acc map, the pooled map and the block's scratch in
  that order.
- **`swin_upsample_blend`** (`upsample_blend`): the projected map
  `narrow(sum_k project[k] * x[k])` on the coarser grid, replicated 2x on the
  skip's grid with the reference's clamp (`min(i / 2, hx - 1)`), then
  `f16(skip * gain[c] + replicated)` per element. The x reads honour x's plane
  and row strides.
- **`swin_layer_plain`** (`plain`/`inpview`/`outview`): `round8(swin_block(x,
  shifts))` — the variants only name a neighbouring buffer layout.
- **`swin_layer_upsample`** (`upsample`): the blend on the skip's grid, then
  the `C == 32` path runs the block on the f16 blend with the blend doubling as
  the MLP residual (`swin_core`'s `x_wide = x` rule) and every other width
  rounds the blend first (`round8(blend)`) and takes the quantised residual
  path; the output is `round8(wide)`.

Disclosed non-reproductions: the reference's `SwinLayer.forward` passes
`spec.out_w, spec.out_h` into `downsample`'s `(rows, columns)` slots — the plan
metadata naming that makes that correct is out of scope, so `swin_downsample`'s
output extents are explicit parameters. The windows are processed one at a
time; the reference chunks windows only for scheduling and windows are
arithmetically independent. Validation of every entry (nulls, dimensions,
strides, pairings, operand classes, capacities) completes before the first
write; on any error the outputs and scratch are untouched, with one documented
exception: `swin_layer_upsample` reports `kInvalidValues` after the blend stage
has written its scratch regions (outputs untouched) when an extreme operand set
rounds a blend element outside the finite binary16 range — the reference
returns infinity there and the block's finite-map class would otherwise reject
mid-write. The entry rejects that class before the block call.

## Validation and tests

`vr/tests/native-swin-resample-test.cpp`, run by `vr/test-native-swin-resample.sh`
(strict `-O2` and `-O0` builds, both `-ffp-contract=off`; a sanitised build when
the toolchain can link a sanitizer runtime): **5 278 checks pass in both
builds**. The suite is centred on:

- **Hand-derived dyadic known answers** (exact decompositions beside each
  case): the gated-sum single-rounding discriminator (`25 * 41.5 - 3*2^-24`
  rounds to 1037, not the float32-sum alternative 1038), its overflow band
  (65504 + 15.5 -> 65504, 65504 + 16 -> infinity, mirrored negatives) and
  subnormal ties-to-even; the projection's k-order accumulator discriminator
  (200704, not 200704.015625) and the stored-result double rounding
  (367.875 -> 384 where direct e4m3 gives 352); the replication index
  arithmetic with its clamp edge (the 3x5-into-2x2 map pinned cell by cell);
  the downsample resize semantics (top-left kept, zero padded, cropped); and
  the `C == 32` f16-residual upsample path discriminator (384 versus the
  `C == 64` quantised-residual 352).
- **Bit-exact staged-chain equivalence for every composed entry** over
  deterministic corpora, with the chain written out directly where the entry's
  own reads are under test: `swin_downsample` against the block + `pool2x2` +
  a manual projection/resize chain; `swin_upsample_blend` against a manual
  projection + clamped replication + gated-sum chain; `swin_layer_plain`
  against `swin_block` + `e4m3`; `swin_layer_upsample` against a manual blend
  chain + the `C == 32`/`C != 32` block staging + `e4m3`. The corpora cover
  zero-weight and non-trivial randomised blocks (`BlockFixture`/`WideFixture`),
  all four accepted shift pairs, pad and crop output extents, clamped
  replication extents (skip larger than the replicated map), and poisoned
  padded input strides (map plane/row strides and matrix row strides, gaps at
  `-224`, exactly binary16 and on the e4m3 grid). The block's own x strides
  are pinned in the block suite and are not re-shared as evidence here.
- **Negative controls** with sentinel-checked untouched outputs and scratch on
  every validation rejection class of the four entries (entry-operand and
  weight nulls, odd/zero extents, stride bounds, pairings — heads and the mid
  projection —, off-grid/non-finite operand values, the weight grids, f16
  scale rows, head-scale magnitude and bias table, off-grid/non-finite input
  maps, shift classes, the downsample output extents and exact capacity
  boundaries — the layer scratch boundary is blend + projected + block). The
  layer weight classes are pre-validated in `swin_layer_upsample` in the
  block's own order, so each weight rejection is sentinel-checked to prove it
  precedes the blend stage's first write. The documented blend-overflow
  scratch exception is pinned explicitly (outputs untouched, blend/projected
  scratch regions written). Plus determinism repeats and zero-allocation
  accounting.

The deterministic corpus draws arbitrary sign/exponent/mantissa f16 patterns
(full 16-bit draws, exponent field 0..30 only), modest exactly-binary16
gains/skips for the layer blend corpora (clear of the disclosed overflow
class; the standalone blend-surface chains draw full-range f16 gains/skips and
may round to infinity, computed bit-identically by both chains) and
e4m3-grid values elsewhere. Development of the padded-stride equivalence
exposed a real stride defect in `swin_upsample_blend`'s projection read (the
plane offset used `r * x.cols` instead of `r * x.row_stride`, wrong for any
padded row stride); it is fixed and now pinned by both blend-surface chains.

The sanitised build is skipped on this workstation because no sanitizer
runtime can be linked (libasan/libubsan RPMs absent; the runner verifies the
skip reason is the missing runtime and reports the exact linker error) — an
environment gap, not a code failure. Existing CPU regressions (slot replay,
ladder report/names, input layout) pass unchanged. Synthetic correctness only:
this establishes no vendor correspondence.

Run: `bash vr/test-native-swin-resample.sh`.
