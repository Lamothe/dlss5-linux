# Native Swin block composition

CPU-only, typed, allocation-free C++20 composition of one windowed-attention
block (`swin_block`) and its f16-data entry (`swin_core`): the e4m3 input
rounding (`round8`), the shifted window passes (`_to_windows`), the per-window
MLP -> stored-result double rounding -> attention chain with the one-head
unrounded-skip rule, the map reassembly (`_from_windows`) and the f16 `wide`
output, with the accumulator mode of the reference. The stages come from the
completed slices: the e4m3 boundary and pooling ([NATIVE-PRE-POST.md](NATIVE-PRE-POST.md)),
window partition ([NATIVE-SWIN-WINDOWS.md](NATIVE-SWIN-WINDOWS.md)), e4m3
linears ([NATIVE-SWIN-LINEAR.md](NATIVE-SWIN-LINEAR.md)), the MLP assembly
([NATIVE-SWIN-MLP.md](NATIVE-SWIN-MLP.md)) and windowed attention
([NATIVE-SWIN-ATTENTION.md](NATIVE-SWIN-ATTENTION.md)). The int4 mixed-network
path, the wider network assembly, temporal history/reset and live-input
evaluation remain later slices. No GPU, headset, proprietary payload, Python or
PyTorch runtime is required.

This is an independent implementation of explicitly documented arithmetic
semantics, adapted with attribution from the MIT-licensed reference
implementation
[`mauri870/dlssnr-pytorch`](https://github.com/mauri870/dlssnr-pytorch) at
commit `3beef7935a8ef976b47ab78b906e75600c52d2de` (`dlssnr/layers/swin.py`, the
`swin_block`/`swin_core` functions and the `SwinParams` shape contract, using
the previously pinned and adapted `_mlp`/`store`/`round8`/`_to_windows`/
`_from_windows`/`_attention` semantics of the earlier slices). All test shapes
and values are original. **This is not vendor code.** Passing these tests
establishes synthetic arithmetic correctness of this implementation against
independently derived controls only — never correspondence to NVIDIA's
implementation. No proprietary weights, captures or binaries are used or
implied.

## Source attribution and licence

Semantics pinned from the reference at the commit above (`dlssnr/layers/swin.py`).
The block body and its f16-data entry:

> ```python
> def swin_block(params: SwinParams, x: torch.Tensor, shift_x: int, shift_y: int,
>                x_wide: torch.Tensor | None = None, accumulator: bool = False):
>     """Run one block on ``[1, C, rows, columns]``.
>
>     Returns ``wide``, the f16-exact output before its final e4m3 rounding (the layer's output is
>     ``q8(wide)``); with ``accumulator=True`` returns ``(wide, acc)`` where ``acc`` is the same output
>     before the f16 rounding, which the 2x2 pooling of a downsample reads.
>
>     ``x`` is what the matrix products read (rounded to e4m3); ``x_wide``, when given, is the f16 value
>     the MLP residual uses instead (the first and last blocks feed f16 data). ``shift_x`` / ``shift_y`` are
>     the plan's window shifts in tiles: ``shift_x`` moves the grid along the columns of the map (its last
>     dimension), ``shift_y`` along the rows.
>     """
>     _, channels, height, width = x.shape
>     ...
>     quantised, grid = _to_windows(round8(x), shift_y, shift_x)
>     residual = quantised if x_wide is None else _to_windows(x_wide, shift_y, shift_x)[0]
>     per_window = max(1, _CHUNK_ELEMENTS // (params.heads * TOKENS * TOKENS))
>     out = torch.empty_like(quantised)
>     inside = None
>     if params.i4 is not None and params.i4.qkv is not None:
>         inside = _to_windows(torch.ones(1, 1, height, width, device=x.device), shift_y, shift_x)[0]
>     for start in range(0, quantised.shape[0], per_window):
>         chunk = slice(start, start + per_window)
>         y = _mlp(params, quantised[chunk], residual[chunk])
>         y_quantised = store(y)
>         # one head keeps the unrounded MLP output as the attention skip; several heads use its e4m3 form
>         skip = y if params.heads == 1 else y_quantised
>         out[chunk] = _attention(params, skip, y_quantised, None if inside is None else inside[chunk])
>         drain(out.device)
>     acc = _from_windows(out, grid, height, width)
>     wide = _f16(acc)
>     return (wide, acc) if accumulator else wide
>
>
> def swin_core(params: SwinParams, x: torch.Tensor, shift_x: int, shift_y: int, accumulator: bool = False):
>     """``swin_block`` on f16-exact data: ``x`` doubles as the MLP residual."""
>     return swin_block(params, x, shift_x, shift_y, x_wide=x, accumulator=accumulator)
> ```

The weight-shape contract of one block, from the `SwinParams` docstring at the
same pin:

> Buffers (shapes, all float32): ``expand`` ``[4C, C]``, ``mid`` ``[C, 4C/heads]``
> (block diagonal over heads, ``None`` for a single head), ``contract`` ``[C, 4C]``
> or ``[C, C]``, ``qkv`` ``[3C, C]`` (per head: q rows, k rows, v rows), ``out_proj``
> ``[C, C]`` (all e4m3 values); ``exp_bias`` ``[heads, 64, 64]``, the position bias
> already carried through the exponent ramp; ``head_scale`` ``[heads]`` as stored;
> and the f16-exact residual scales ``mlp_scale`` and ``attn_scale``, ``[C]``.

`round8` is the reference's `q8` on the matrix-product operands (`round8 =
fuse(q8)` at the pin), `store` its `q8(_f16(x))` stored-result double rounding,
`_mlp` the MLP body adapted in the MLP slice ([NATIVE-SWIN-MLP.md](NATIVE-SWIN-MLP.md)),
`_attention` the attention body adapted in the attention slice
([NATIVE-SWIN-ATTENTION.md](NATIVE-SWIN-ATTENTION.md)), and `_to_windows`/
`_from_windows`/`_window_grid` the window partition adapted in the windows slice
([NATIVE-SWIN-WINDOWS.md](NATIVE-SWIN-WINDOWS.md)); each of those documents
quotes and adapts its own pinned fragment. The int4 branch of the body
(`params.i4`, the `inside` mask, the fused-kernel path and `drain`) is out of
scope here: this slice implements the eager default path only.

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

## Operand classes and layout

`SwinBlockParams` passes every learned operand explicitly (original synthetic
values in the tests; no learned weights are used or implied):

| operand | shape | class |
| --- | --- | --- |
| `expand` | `[expand.rows, C]` | e4m3 grid |
| `mid` | `[heads * 32, mid.cols]` or null | e4m3 grid (grouped middle projection) |
| `contract` | `[C, expand.rows]` (mid null) or `[C, mid.rows]` | e4m3 grid |
| `mlp_scale` | `[1, C]` | finite, exactly binary16-representable |
| `qkv` | `[3C, C]` | e4m3 grid |
| `head_scale` | `[1, heads]` | finite float32, magnitude at most `kMaxHeadScale` = 65504 |
| `exp_bias` | `[heads, 4096]` | finite float32 (post exponent-ramp) |
| `attn_scale` | `[1, C]` | finite, exactly binary16-representable |
| `out_proj` | `[C, C]` | e4m3 grid |

`C = heads * kHeadDim` with `kHeadDim = 32`; `heads` must satisfy
`heads <= kMaxLinearDim / (3 * kHeadDim)` (10 at `kMaxLinearDim = 1024`), the
same qkv-width bound as the attention assembly. The `mid`/`expand`/`contract`
pairings mirror the MLP slice exactly (including `mid = null` on the single-head
path, matching the reference's `mid ... if heads > 1 else None`).

The input maps are channels-first `FeatureBatchView`s `[C, rows, cols]`:
`x` holds finite float32 (the matrix products read `round8(x)`), and `x_wide`,
when given, has `x`'s dimensions and holds finite, exactly
binary16-representable values (the f16 data the MLP residual uses; the
reference's `swin_core` passes f16-exact `x` here). `wide_dst` receives the
f16-rounded output map; `acc_dst`, when given, receives the pre-f16 `acc` map
(accumulator mode). Both outputs are tight channels-first buffers of
`C * rows * cols` floats.

The scratch buffer is caller-owned and holds, in order (all tightly packed):
the e4m3 map (`C*rows*cols`), the quantised token batch (`windows*64*C`), the
residual token batch (only when `x_wide` is given), the out token batch, one
window's `y` and `y_quantised` (`64*C` each), the MLP scratch
(`64*expand.rows`), the attention scratch (`64*3C + 4*64*C + 2*heads*4096`),
and the `acc` map (only when `acc_dst` is null). The buffers must not overlap
each other or the input maps (caller discipline, never rejected).

## Composition semantics

The body follows the pinned `swin_block` exactly:

1. `quantised, grid = _to_windows(round8(x), shift_rows, shift_cols)` — the
   input map is rounded to the e4m3 grid elementwise (nearest, ties to even,
   saturating at 448), then windowed with the zero front/remainder padding.
   `shift_rows`/`shift_cols` map the reference's `shift_y`/`shift_x` (rows and
   columns of the map); only 0 and -1 tiles exist (`swin_window_grid` rejects
   anything else with `kInvalidControls`).
2. `residual = quantised` when no `x_wide` is given, else
   `_to_windows(x_wide, ...)` — the residual-source rule of the pinned body.
   When the residual is the quantised batch it is the same windowed copy
   (`r_tokens = q_tokens`); `x_wide` windows it separately.
3. Per window: `y = _mlp(quantised_w, residual_w)` (float32), then
   `y_quantised = store(y)` = `e4m3(f16(y))`, the stored-result double
   rounding. The attention skip is `y` when `heads == 1` and `y_quantised`
   otherwise — the pinned comment: "one head keeps the unrounded MLP output as
   the attention skip; several heads use its e4m3 form". The attention input is
   always `y_quantised`; `out_w = _attention(skip, y_quantised)`.
4. `acc = _from_windows(out, grid, rows, cols)` — the tight `[C, rows, cols]`
   map, `wide = _f16(acc)` — one binary16 rounding of `acc` (ties to even;
   finite `acc` beyond the binary16 range gives +/-inf exactly as the
   reference's `_f16`). `acc_dst`, when given, receives `acc` before that
   rounding; the layer-level `q8(wide)` output rounding is a later stage.

`swin_core(x, ...)` is `swin_block` with `x_wide = x`, exactly as the pinned
entry: `x` doubles as the MLP residual and therefore must hold f16-exact
values. The reference chunks windows only for scheduling
(`_CHUNK_ELEMENTS`); windows are arithmetically independent, so this
implementation processes them one at a time with no observable difference.

## Validation and write contract

Validation completes before the first write, in this order: nulls; `x`
dimensions and strides; the window grids (shift classes); `x_wide` pairing,
dimensions and strides; the block pairings (mirroring the staged entries'
pairings exactly, and mirroring `swin_mlp`'s scratch width bound —
`expand.rows <= kMaxLinearDim` — so the staged call cannot reject after the
first write); the weight operand grids and value classes (including the
head-scale row's magnitude bound); the `x` value scan (finite) and the `x_wide`
value scan (finite and f16-exact); and finally the output and scratch
capacities. On any error `wide_dst`, `acc_dst` and `scratch` are all untouched.

Every staged call in the composition is then guaranteed `kOk`: the validated
operand classes keep every intermediate within float32 range (e4m3 operands and
values bound each product; the activations are clamped-polynomial times the
input and re-stored on the e4m3 grid; the residual products are at most
`65504 * 65504`, far below the float32 limit) or on finite stored grids
(`store` saturates infinities to 448; the softmax ramp clamps to
`[EXP_LOW, EXP_HIGH]` before the bit transplant). The norm-overflow path of
`_norm_scale` (a half-sum that overflows f16 gives a scale of zero) multiplies
the raw finite float32 accumulator, producing exact `0.0` Q/K rows — not NaN —
so its `store` stays on the e4m3 grid. The only remaining NaN hazard is the
per-head scale product `_norm_scale(query) * head_scale`: `_norm_scale` is at
most ~127.02 (the `kNormFloor` clamp), so the magnitude bound
`|head_scale| <= 65504` keeps that product below `8.4e6` — an unbounded finite
row (e.g. `1e38`) would overflow the product to infinity and a zero query
dimension would then form `0 * inf = NaN`, failing `swin_logits`' e4m3 operand
checks after the writes. With the bound every product is finite (large ones
saturate on the e4m3 grid) and the claim holds. This is the substantiation of
the "guaranteed `kOk`" claim; it is not a claim about unvalidated operands, and
the bound is this implementation's accepted-domain restriction (the reference's
`head_scale` is float32), not an arithmetic change.

## Explicit choices and disclosed limitations

- The composition passes each window through the staged entries one at a time;
  the reference's chunking (`per_window`) is scheduling only and this
  implementation does not reproduce it (no observable difference).
- The map outputs are float32; `wide` is one binary16 rounding of `acc`. The
  reference returns torch tensors whose `wide` is "the f16-exact output before
  its final e4m3 rounding"; `q8(wide)` (the layer output) and the 2x2 pooling
  of `acc` are later stages (the pooling is pinned in
  [NATIVE-PRE-POST.md](NATIVE-PRE-POST.md)).
- The int4 mixed-network path (`params.i4`, the `inside` mask, hard-swish),
  the fused-kernel path (`swin_fused`/`window_attention`/`gemm`) and `drain`
  are not implemented; this slice is the eager default path. The `inside`
  mask exists only on the int4 path, so the composition here has no
  equivalent operand.
- The `heads` bound is enforced through the same observable accept/reject
  boundary as the attention assembly (the `3C` qkv width inside
  `kMaxLinearDim`); the token-channel bound (`kMaxTokenChannels = 256`) binds
  first for `heads * 32` maps, and the pairings' heads bound itself is pinned
  directly (`heads = 11` with an in-bound 32-channel map, through both the
  entry and the pairing helper).
- No weight loading, runtime binding, GPU execution or capture is involved;
  the weights are caller-provided operands. Original synthetic values in the
  tests only.

## Test evidence (26/10/2026)

`vr/test-native-swin-block.sh` builds `vr/tests/native-swin-block-test.cpp`
strictly (`-std=c++20 -Wall -Wextra -Werror -ffp-contract=off`) at `-O2` and
`-O0`: **633 646 checks pass in both builds**. The suite is centred on:

- Closed forms of a zero-weight block with unit scales (every stage collapses
  to a documented closed form): the residual-source rule (`e4m3(1 + 2^-11) =
  1.0` for the quantised residual versus the `x_wide` value `1 + 2^-10` passing
  through as f16 data, and `swin_core` doubling `x` as its residual), the
  one-head unrounded-skip rule and the `wide = _f16(acc)` tie-to-even
  narrowing pinned together as one discriminator (residual `1.5` with the f16
  scale `1 + 2^-10` gives the exact float32 product `1.50146484375`, which
  ties on the binary16 grid and rounds to even `1.501953125`: heads == 1 keeps
  `acc = 1.50146484375` and `wide = 1.501953125`, while several heads skip
  through `store(y) = 1.5` and both maps hold `1.5`), including heads == 2
  fixtures on `C = 64` — the null-`mid` single-stage path and the grouped
  non-null-`mid` path (`mid [64, 16]`, `expand.rows = heads * mid.cols`).
- A bit-exact equivalence control of the composed entry against the pinned
  staged chain written out directly (the same reference composition against
  the independently tested stage entries), over deterministic xorshift32
  finite-binary16 corpora (any sign, exponent field 0..30): both residual
  paths, the single-head null-`mid` and the multi-head grouped-`mid` paths,
  zero-weight and non-trivial random weight corpora, all four shift pairs for
  both paths and both MLP shapes, multi-window grids with the front zero
  padding, and input maps with padded strides whose padding cells are poisoned;
  output sentinels verify the writes stay within the map. The same matrix pins
  the non-accumulator mode: with `acc_dst` null the `acc` map is kept in
  scratch (`acc_count` is ignored) and `wide` is bit-identical. A boundary run
  with `head_scale = ±65504` and the mid-path width edge (`expand.rows ==
  kMaxLinearDim`, heads = 2 with `mid.cols = 512`) match the staged chain
  end to end.
- Negative controls covering every rejection class: null pointers (including
  a null `x_wide` data pointer and a null `mid` table), the `x`
  dimension/stride bounds, the shift classes (`kInvalidControls`, forwarded
  from `swin_window_grid`), the `x_wide` pairing/stride/value classes (an
  off-grid value, NaN, and the f16-exactness rule through `swin_core`), the
  mirrored pairing classes for every weight operand (including the mid-path
  width bound: heads = 2 with `mid.cols = 513` makes `expand.rows = 1026 >
  kMaxLinearDim` and is rejected before any write), one weight value class
  per operand class (off-e4m3 `expand`, off-f16 `mlp_scale`, non-finite
  `head_scale`, and finite `head_scale` values beyond the `kMaxHeadScale`
  magnitude bound — the `1e38` case that would overflow the norm*head scale
  product — NaN `exp_bias`), the exact scratch boundary (the in-scratch `acc`
  map accounting with `acc_dst` null, one float short rejected, exactly there
  accepted), and the output capacity classes — each group followed by sentinel
  checks that `wide_dst`, `acc_dst` and `scratch` are untouched.
- Determinism (bit-exact repeats within each residual path) and
  zero-allocation accounting over 64 alternating calls.

The sanitised build is skipped on this workstation because no sanitizer
runtime can be linked (libasan/libubsan RPMs absent; the runner verifies the
skip reason is the missing runtime and reports the exact linker error) — an
environment gap, not a code failure. The existing CPU regressions
(slot-replay self-test 834 known answers, ladder-report 19 controls,
ladder-names 14 controls, native input-layout with 9 invalid-layout controls)
pass unchanged. An independent read-only review of this slice raised two
defects (the unbounded `head_scale` product and the missing mid-path width
bound) and three honesty/coverage gaps (the misdescribed heads == 2 fixture,
untested non-accumulator mode and thin equivalence arithmetic); all five are
fixed in this change set, with the review outcome recorded in the campaign
handover. A follow-up independent review (26/10/2026) returned **PASS** with
no defects and two findings, both fixed here: a mislabelled negative-test
comment (the `heads = 11` case observes the token-channel bound, not the
pairings' heads bound — the comment now says so and the heads bound is pinned
directly) and the deliberately write-only staged out-params
(`front_h`/`front_w`, the pass grids), now annotated in the header. The full
output of an earlier PASS review round was lost to a context-rotation
truncation; this fresh review covered the same criteria before commit.

## What this does NOT establish

This is synthetic arithmetic correctness of one composed block against
independently derived controls. It is **not** vendor correspondence, uses no
learned weights (original synthetic values only) and establishes no model
behaviour. The int4 mixed-network path, the encoder/decoder network assembly
and its learned-weight execution graph, temporal history/reset, and
live-input image evaluation all remain; the migration milestone requires those
before any native input-to-output claim.
