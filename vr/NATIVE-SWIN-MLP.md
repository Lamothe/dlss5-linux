# Native Swin MLP assembly

CPU-only, typed, allocation-free C++20 arithmetic for the `_mlp` body of the
windowed-attention blocks on the e4m3 grid: the e4m3 hidden (`swin_hidden`, from the
linears slice), the grouped middle projection narrowed through the stored-result
double rounding (`swin_store`), the contract projection and the f16 residual scale,
composed in `swin_mlp`. Windowed attention (position-bias deswizzle, softmax
numerators, head scale), the int4 mixed-network variants, temporal history/reset and
live-input evaluation remain later slices. No GPU, headset, proprietary payload,
Python or PyTorch runtime is required.

This is an independent implementation of explicitly documented arithmetic semantics,
adapted with attribution from the MIT-licensed reference implementation
[`mauri870/dlssnr-pytorch`](https://github.com/mauri870/dlssnr-pytorch) at commit
`3beef7935a8ef976b47ab78b906e75600c52d2de` (`dlssnr/layers/swin.py`, the `_mlp` and
`store` functions and the `_hidden`/`_linear` they compose). All test shapes and
values are original. **This is not vendor code.** Passing these tests establishes
synthetic arithmetic correctness of this implementation against independently
derived controls only — never correspondence to NVIDIA's implementation. No
proprietary weights, captures or binaries are used or implied.

## Source attribution and licence

Semantics pinned from the reference at the commit above:

> `HEAD_DIM = 32`

> ``mid`` ``[C, 4C/heads]`` (block diagonal over heads, ``None`` for a single head),
> ``contract`` ``[C, 4C]`` or ``[C, C]`` … and the f16-exact residual scales
> ``mlp_scale`` and ``attn_scale``, ``[C]``.

> ```python
> @fuse
> def store(x: torch.Tensor) -> torch.Tensor:
>     """float32 -> f16 -> e4m3, the double rounding of a stored result, as one kernel."""
>     return q8(_f16(x))
> ```

> ```python
> def _mlp(params: SwinParams, quantised: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
>     """``rs * x + Wc . hidden`` for ``[n, 64, C]`` tokens, still in float32 (rounded by the caller). The activation goes straight from the float32 accumulator to e4m3; the grouped middle projection is narrowed to f16 first. """
>     hidden = _hidden(params, quantised)
>     if params.mid is not None:
>         groups = hidden.float().reshape(*hidden.shape[:-1], params.heads, -1)
>         mid = params.mid.reshape(params.heads, HEAD_DIM, -1)
>         hidden = store(torch.einsum("ntgj,gdj->ntgd", groups, mid).reshape(*hidden.shape[:-1], -1))
>     return residual * params.mlp_scale + _linear(hidden, params.contract)
> ```

> `_hidden` … if `site is None`: `_linear(quantised, params.expand, activate=True)`;
> the int4 mixed network changes both (int4 expansion and the hard-swish).

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

The MIT licence covers the referenced software's **code**; it grants no rights to any
NVIDIA model data, weights or captures, none of which are used here.

## Operand classes and layout

The matrix view conventions (`rows` x `cols`, float strides, `cols <= row_stride`)
and validation helpers come from the linears slice. `x`, `expand`, `mid` and
`contract` are e4m3-grid operands (every value finite and exactly on the e4m3 grid),
so every product is exact in float32. `residual` is arbitrary finite float32 (the
block feature maps). `scale` is one row of finite, exactly binary16-representable
values (the reference's f16-exact `mlp_scale`). `mid` is the block-diagonal per-head
weight laid out flat `[heads * kHeadDim, group_width]` — the reference's
`params.mid.reshape(heads, HEAD_DIM, -1)` — where `group_width = expand.rows /
heads`; it is `null` for the single-head case.

## The stored-result double rounding (`swin_store`)

`swin_store(value) = e4m3(f16(value))`: the float32 accumulator is first narrowed to
binary16 (nearest, ties to even), then the f16 value is rounded to the e4m3 grid
(nearest, ties to even, saturating at ±448; the f16 infinities saturate through the
e4m3 clamp, and NaN gives the canonical quiet NaN with the sign dropped). This
double rounding is the reference's own and is observable: `367.875` stores as `384`
(the f16 tie lands on 368 and the e4m3 tie on the even code 384) where a direct e4m3
rounding would give `352`.

## The grouped middle projection (`swin_mid`)

`swin_mid(hidden, mid, heads, out)` maps `hidden [rows, heads * group_width]` to
`out [rows, heads * kHeadDim]`:

```
out_row[g * kHeadDim + d] = swin_store(sum_j hidden_row[g * group_width + j] * mid[g * kHeadDim + d][j])
```

per token and head — the reference's `einsum("ntgj,gdj->ntgd", groups, mid)` with
`store` applied per element. The products are added to a float32 accumulator in
increasing `j` order starting at `+0.0`; each product is exact and the sums are
rounded to float32 at each step (see the accumulation note below). `swin_mid_row`
is the shared row primitive so the assembly uses one arithmetic path.

## The contract projection and residual scale (`swin_mlp_tail`)

`swin_mlp_tail(in, contract, residual, scale, out)` maps `in [rows, k]` to
`out [rows, n]`:

```
out_row[c] = residual_row[c] * scale[c] + sum_j line[j] * contract[c][j]
```

— the reference's eager `residual * mlp_scale + _linear(hidden, params.contract)`.
The residual-scale multiply and the add are **two separate float32 operations with
one rounding each**, not a fused multiply-add. The pin: `8388609 * 1.5` rounds
(ties to even) to `12582914` and `12582914 - 0.5` ties to even again to `12582914`,
where a fused multiply-add would round the exact `12582913.0` once to `12582913`.
The contract product uses the linears slice's k-order float32 accumulator.

Extreme residual magnitudes can overflow the float32 product or sum to the IEEE
±infinity; that passes through the output unchanged (no rejection, no NaN).

## The composed assembly (`swin_mlp`)

`swin_mlp(x, expand, mid, heads, contract, residual, scale, out, out_capacity,
scratch, scratch_capacity)` computes `hidden = swin_hidden(x, expand)` into the
caller-owned `scratch [rows, expand.rows]`, then either feeds `hidden` straight to
the tail (single-head: `mid` null, `contract.cols == expand.rows`) or its
`store`-narrowed grouped mid projection (`contract.cols == heads * kHeadDim`) into
one fixed internal row buffer. Every parameter — dimensions, strides, the e4m3
operand grids, the residual/scale classes and both capacities — is validated before
the first write: on any error `out` and `scratch` are both untouched. The buffers
must not overlap (caller discipline, never rejected). `heads` is ignored when `mid`
is null.

## Explicit choices and disclosed limitations

The reference's default `_linear` path states no summation-order contract; as in the
linears slice, the explicit ascending-k float32 accumulation is this implementation's
documented convention, pinned by permutation controls that are bit-exact only because
the test corpora are exactly representable at every partial sum. No vendor
correspondence is claimed for the order.

- The int4 mixed-network path (`site.ex` int4 expansion and the hard-swish with its
  folded scale) is **not** implemented; this slice covers the non-int4 `_hidden`
  path only, matching the linears slice's scope.
- The reference's `_mlp` return is "still in float32 (rounded by the caller)"; the
  caller-side rounding belongs to later composition slices and is out of scope here.
- `swin_mid`/`swin_mlp_tail`'s output-validator dimension bounds are unreachable
  through those APIs (`mid.rows = heads * kHeadDim` is confined to 32..1024 and the
  tail's `contract.rows` is confined by the residual/scale stride bound), and
  `swin_mlp`'s `out` is likewise confined; the cols bound **is** reachable through
  `swin_mlp`'s scratch validation (a mid path with `expand.rows > kMaxLinearDim`),
  and that case is exercised. No other rejection branch is claimed unexercised.

## Test evidence (26/10/2026)

`vr/tests/native-swin-mlp-test.cpp` with runner `vr/test-native-swin-mlp.sh`:
**2780 checks** in the strict `-O2` build and again at `-O0` (both
`-ffp-contract=off`). Every known answer is a hand-derived dyadic rational. The
corpus covers:

- the store pins (the 367.875 → 384 double-rounding discriminator vs the direct
  e4m3 352, the 65519/65530 f16 saturation chain to 448, signed zero, canonical
  NaN);
- hand-derived mid known answers including the head-grouping discriminator (a
  mis-grouped implementation gives 15 instead of 3), the store-through-projection
  discriminator (367.875 → 384 vs 352) and padded strides whose padding is never
  read or written;
- hand-derived tail known answers including the two-rounding discriminator
  (12582914 where a fused multiply-add gives 12582913);
- composed known answers through the pinned `swin_hidden` activations on both the
  single-head and mid paths;
- permutation and changed-input-sensitivity controls on exact corpora with guarded
  pairwise-distinct rows and per-cell exact deltas (row, head-block and
  group-internal-K permutations for the mid path; row, output-channel and K
  permutations plus in and residual sensitivity for the tail);
- a bit-exact equivalence control of the composed assembly against the staged chain
  (`swin_hidden` → `swin_mid` → `swin_mlp_tail`) over a deterministic pseudorandom
  e4m3 corpus on both paths, with padded strides and untouched-padding checks;
- negative controls for every rejection class of all three entry points (nulls,
  dimension pairings, the heads bound, stride bounds, capacity, the e4m3 operand
  grid on every grid side, the residual finiteness and f16-exact scale classes,
  the over-wide scratch branch) with sentinel-checked untouched outputs — and
  untouched scratch in the assembly — after every rejection group;
- the documented float32 overflow passthrough (finite residual scaled past the
  float32 range gives ±inf), a valid boundary-operand success case (±448 and the
  subnormal grid), determinism reruns and zero-allocation accounting.

The development history includes one corrected flaw in an earlier draft of the
two-rounding discriminator (values `8388607`/`8388608`, whose product is exact at
half-ulp below 2^23 and therefore does not discriminate at all); the asserted pair
is the corrected `12582914`/`12582913` above.

The sanitised build is skipped on this workstation because no sanitizer runtime can
be linked (libubsan/libasan RPMs absent; the runner verifies the skip reason is the
missing runtime and reports the exact linker error) — an environment gap, not a code
failure.

## What this does NOT establish

- No vendor correspondence. The arithmetic matches the pinned MIT reference semantics
  as documented (plus this slice's explicit summation and rounding conventions); it is
  not evidence about NVIDIA's implementation.
- No learned weights: every operand tested is an original synthetic e4m3 value. The
  learned-weight execution graph assembly remains later slices.
- No windowed attention (position-bias deswizzle, softmax numerators, head scale or
  exp-bias ramp), temporal behaviour, decoder stages, live-input/image evaluation or
  GPU behaviour of any kind.
