# Native Swin windowed attention

CPU-only, typed, allocation-free C++20 arithmetic for the `_attention` body of the
windowed-attention blocks on the e4m3 grid: the position-bias deswizzle
(`swin_deswizzle_index`/`swin_deswizzle_bias`) and the exponent-ramp bias
(`swin_exp_bias`), the exponent-ramp softmax numerators (`swin_softmax_numerator`),
the f16 denominator tree (`swin_softmax_denominator_row`) and the softmax rows
(`swin_softmax_row`/`swin_softmax`), the QKV normalisation with the head scale
(`swin_normalise_head_row`/`swin_normalise_qkv`), the attention logits
(`swin_logit_dot`/`swin_logits`) and the context einsum through the stored-result
double rounding (`swin_context`), composed in `swin_attention`. The residual output
line reuses the MLP slice's `swin_mlp_tail`. The int4 mixed-network variants,
temporal history/reset and live-input evaluation remain later slices. No GPU,
headset, proprietary payload, Python or PyTorch runtime is required.

This is an independent implementation of explicitly documented arithmetic semantics,
adapted with attribution from the MIT-licensed reference implementation
[`mauri870/dlssnr-pytorch`](https://github.com/mauri870/dlssnr-pytorch) at commit
`3beef7935a8ef976b47ab78b906e75600c52d2de` (`dlssnr/layers/swin.py`, the
`_deswizzle_bias`/`_normalise_qkv`/`_softmax_numerators`/`_denominator`/`_softmax`/
`_attention` functions and the constants they use, with `dlssnr/attention.py` and
`tests/test_triton_kernels.py` from the same pin as corroborating shape/rounding
evidence). All test shapes and values are original. **This is not vendor code.**
Passing these tests establishes synthetic arithmetic correctness of this
implementation against independently derived controls only — never correspondence
to NVIDIA's implementation. No proprietary weights, captures or binaries are used
or implied.

## Source attribution and licence

Semantics pinned from the reference at the commit above (`dlssnr/layers/swin.py`):

> ```python
> WINDOW = 8                   # pixels along each side of an attention window
> TOKENS = WINDOW * WINDOW
> TILE = 4                     # pixels along each side of a tile; a window is 2x2 tiles
> HEAD_DIM = 32
>
> EXP_SLOPE = 0.044921875
> EXP_OFFSET = 1.30078125
> EXP_LOW, EXP_HIGH = 1.03125, 1.5693359375   # clamp of the exponent ramp (both exact in f16)
> NORM_FLOOR = float(torch.tensor(0.000062, dtype=torch.float16))   # lower bound of the squared norm (f16-exact)
> SUM_FLOOR = 2.0 ** -14                      # lower bound of the softmax denominator
> ```

> ```python
> def _deswizzle_bias(raw: torch.Tensor) -> torch.Tensor:
>     """One head's 64x64 position bias from the fragment order it is stored in to ``[query, key]``.
>
>     The table is stored as sixteen 16x16 (query block, key block) tiles, each tile in the lane order of an
>     accumulator fragment: 32 lanes of eight values.
>     """
>     i = np.arange(TOKENS)[:, None]
>     j = np.arange(TOKENS)[None, :]
>     lane = 4 * ((i % 16) % 8) + ((j % 16) % 8) // 2
>     slot = (j % 2) | (((i % 16) // 8) << 1) | (((j % 16) // 8) << 2)
>     index = (4 * (i // 16) + (j // 16)) * 256 + lane * 8 + slot
>     return raw[torch.from_numpy(index.reshape(-1))].reshape(TOKENS, TOKENS)
> ```

> ```python
> def _softmax_numerators(logits: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
>     """exp(logit + bias) as the network builds it: an affine ramp rounded to f16 and clamped, whose bit
>     pattern is shifted into the exponent field. ``[n, heads, q, k]`` -> f16 tensor of the same shape."""
>     ramp = _fma(logits, EXP_SLOPE, bias).to(torch.float16).clamp(EXP_LOW, EXP_HIGH)
>     return ((ramp.view(torch.int16) & 0x3FF) << 5).view(torch.float16)
> ```

> ```python
> def _denominator(numerators: torch.Tensor) -> torch.Tensor:
>     """Sum over the 64 keys in the order the original adds them, in f16.
>
>     The keys of a query sit in two lane halves, each holding eight of every 16-key tile block as four
>     pairs: key = 16 block + 4 pair + 2 member + half. Blocks are added first, then pairs, then the two
>     halves, then the two members of a pair.
>     """
>     keys = numerators.reshape(*numerators.shape[:-1], 4, 4, 2, 2)           # block, pair, member, half
>     blocks = ((keys[..., 0, :, :, :] + keys[..., 1, :, :, :]) + keys[..., 2, :, :, :]) + keys[..., 3, :, :, :]
>     pairs = ((blocks[..., 0, :, :] + blocks[..., 1, :, :]) + blocks[..., 2, :, :]) + blocks[..., 3, :, :]
>     halves = pairs[..., 0] + pairs[..., 1]
>     return halves[..., 0] + halves[..., 1]
> ```

> ```python
> @fuse
> def _normalise_qkv(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, head_scale: torch.Tensor):
>     return (q8(query * (_norm_scale(query) * head_scale).unsqueeze(-1)),
>             q8(key * _norm_scale(key).unsqueeze(-1)),
>             q8(_f16(value)))
> ```

> ```python
> @fuse
> def _softmax(logits: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
>     """e4m3 probabilities from ``[n, heads, q, k]`` logits and the exponent-ramp bias."""
>     numerators = _softmax_numerators(logits, bias)
>     reciprocal = 1.0 / _denominator(numerators).float().clamp(min=SUM_FLOOR)
>     return q8(_f16(numerators.float() * reciprocal.unsqueeze(-1)))
> ```

> ```python
>     query, key, value = qkv[..., 0, :], qkv[..., 1, :], qkv[..., 2, :]
>     query, key, value = _normalise_qkv(query, key, value, params.head_scale)
>     logits = torch.einsum("ntgd,nsgd->ngts", query, key)                     # [n, head, query, key]
>     probability = _softmax(logits, params.exp_bias)
>     context = store(torch.einsum("ngts,nsgd->ntgd", probability, value).reshape(count, TOKENS, -1))
>     return skip * params.attn_scale + _linear(context, params.out_proj)
> ```

(the last two blocks are the body of `_attention`, whose docstring states "Q and K
are scaled straight from the float32 accumulator (Q also by the head's scale); V,
the probabilities and the context are narrowed to f16 before they reach the e4m3
grid"). `store` is the reference's `q8(_f16(x))` stored-result double rounding,
`_fma` the double-product/double-add float32-rounding fused multiply-add, `q8` the
e4m3 rounding and `_norm_scale` the 32-dimension reciprocal L2 norm with the
`NORM_FLOOR` clamp — all pinned and adapted in the primitives slice
([NATIVE-SWIN-PRIMITIVES.md](NATIVE-SWIN-PRIMITIVES.md)). The `dlssnr/attention.py`
fused kernel from the same pin states it reproduces `swin._attention`'s rounding
points and f16 summation orders (the squared-norm chains and the softmax
denominator tree), with only the float32 sums inside its two `tl.dot` products
taken in matrix-unit order; `tests/test_triton_kernels.py` pins the eager call
shapes, including the single `store` after the full context sum.

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

Matrix views (`rows` x `cols`, float strides) and the e4m3/f16 grid validators come
from the linears slice; the attention tensor (`[windows, heads, 64, 64]` with
window/head strides) is introduced here. Quantised, qkv, out_proj, query, key,
value and probability operands are e4m3-grid values; the stored/deswizzled bias
tables (`raw` of `swin_deswizzle_bias`/`swin_exp_bias`) are `[heads, 4096]`
binary16-grid values, while the exponent-ramp bias table (`exp_bias`) is
`[heads, 4096]` finite float32; logits and the raw residual `skip` are arbitrary
finite float32;
`attn_scale` is one row of finite, exactly binary16-representable values (the
reference's f16-exact `attn_scale`); `head_scale` is one row of finite float32.
Token rows are `window * 64 + token`; the per-head dimension slice is
`[h * 32, (h + 1) * 32)`.

## The exponent-ramp numerator (`swin_softmax_numerator`)

`swin_softmax_numerator(logit, bias)` is the reference's `_softmax_numerators` core:
one `swin_fma(logit, EXP_SLOPE, bias)` — one float32 rounding from the double
intermediate — narrowed to binary16 (ties to even), clamped to
`[EXP_LOW, EXP_HIGH]` = `[1 + 32/1024, 1 + 583/1024]` (both exactly
binary16-representable), then the mantissa bits are shifted into the exponent field
(`(bits & 0x3FF) << 5`). The clamp forces the ramp into `[1, 2)`, so the pattern is
`0_01111_mmmmmmmmmm` and the result is `2^((m >> 5) - 15) * (1 + (m & 31) / 32)`
for mantissa `m` — an affine approximation of exp carried in the exponent field,
strictly increasing over the clamp range and spanning `[2^-14, 9.75]`. Float32
overflow of the ramp gives ±inf and the clamp saturates it exactly as the
reference's `torch.clamp` on inf; NaN is excluded by the finite-operand validation,
matching the reference's float32 data path.

## The position-bias deswizzle and exponent-ramp bias

`swin_deswizzle_index(i, j)` is the reference's `_deswizzle_bias` index map
character-for-character (`lane`/`slot`/`index` above); the map is a permutation of
`[0, 4096)` (pinned by exhaustive tests) and out-of-range `i`/`j` returns `-1`.
`swin_deswizzle_bias(raw, out, out_capacity)` copies `raw [heads, 4096]` to
`out [heads, 4096]` at the deswizzled positions — index arithmetic only, since
`raw` holds binary16-grid values. `swin_exp_bias(raw, out, out_capacity)` composes
the deswizzle with the ramp: `swin_fma(deswizzled, EXP_SLOPE, EXP_OFFSET)` per
entry, one float32 rounding from the double intermediate — the reference's
`_fma(torch.stack([_deswizzle_bias(row) for row in bias]), EXP_SLOPE, EXP_OFFSET)`.
Both entries validate (tables, the binary16 operand grid, the output pairing and
capacity) before the first write; on any error `out` is untouched.

## The softmax denominator (`swin_softmax_denominator_row`)

The sum over a query row's 64 keys **in the order the reference adds them**, each
add a binary16 add (the sum of two binary16 values is exact in float32, so
`f16(a + b)` is the correctly rounded binary16 add). The keys of a query sit in two
lane halves, each holding eight of every 16-key tile block as four pairs:
`key = 16 block + 4 pair + 2 member + half` (the reference's `reshape(4, 4, 2, 2)`);
blocks first `((b0 + b1) + b2) + b3`, then pairs `((p0 + p1) + p2) + p3`, then the
two halves of a member, then the two members of a pair. Binary16 overflow gives
±inf, which passes through. Every intermediate node is a binary16 value, so the
sum-order discriminators (2050 vs 2048 at the `2048 + 1` ties) are observable
exactly as in the reference. Through the full softmax the denominator is at least
`64 * 2^-14 = 2^-8` (every numerator is ≥ `2^-14`), so the `SUM_FLOOR` clamp in
`swin_softmax_row` provably never binds there; it is kept for exactness with the
reference.

## The softmax rows (`swin_softmax_row`, `swin_softmax`)

`swin_softmax_row` is the reference's `_softmax` per row: numerators, denominator,
`1.0f / clamped` as one float32 division, then each
`swin_store(numerator * reciprocal)` — one float32 product rounding followed by the
stored f16 → e4m3 double rounding (`q8(_f16(...))`), matching the reference's
`q8(_f16(numerators.float() * reciprocal.unsqueeze(-1)))`. `swin_softmax` is the
batched tensor entry over `[windows, heads, 64, 64]` logits and per-head exp-bias
rows, writing e4m3-grid probabilities.

## The QKV normalisation (`swin_normalise_head_row`, `swin_normalise_qkv`)

One (token, head) slice of 32 dimensions, the reference's `_normalise_qkv`: Q and K
scale straight from the float32 accumulator to e4m3 in **one** rounding — the
per-head scale product `_norm_scale(query) * head_scale` is one float32 product
(head_scale folded into the query scale only; the key is scaled by its own
`_norm_scale`), each scaled dimension one float32 product, `q8` rounding that
product directly. V takes the stored double rounding (`q8(_f16(value))`,
`swin_store`). The q/v paths therefore differ observably (367.875 scales to 352 on
the q path and stores to 384 on the v path; pinned by the tests). The norm-scale
grouping (scale computed once, then scaled dimensions) is pinned by the 0.025390625
discriminator where per-dimension regrouping gives 0.0234375.

## The attention logits and context

`swin_logit_dot` is the 32-dimension dot product of one (query, key) head pair;
both operands are e4m3-grid so every product is exact in float32, added to a
float32 accumulator in increasing `d` order from `+0.0` (the linears slice's
explicit convention). `swin_logits` maps `query/key [rows, heads * 32]` to
`[windows, heads, 64, 64]` float32 logits (`einsum("ntgd,nsgd->ngts")`).
`swin_context` maps `probabilities [windows, heads, 64, 64]` and normalised
`value [rows, heads * 32]` to `out [rows, heads * 32]` with
`swin_store(sum_j prob[...] * value[...])` — the products summed over **all 64
keys** in one float32 accumulator in increasing `j` order and one `store`
(`einsum("ngts,nsgd->ntgd")` through `store`), matching the eager `_attention` and
its single `store` after the full context sum; the fused kernel's tolerated
deviation is its matrix-unit summation order inside `tl.dot`, not the store count.

## The assembled `swin_attention`

`swin_attention(quantised, qkv, head_scale, exp_bias, skip, attn_scale, out_proj,
out, out_capacity, scratch, scratch_capacity)` runs the non-int4 `_attention` body
per window: the e4m3 QKV projection into `scratch`, the per-head normalisation, the
logits, softmax and context per window, and the residual output line through the
MLP slice's `swin_mlp_tail` (`skip * attn_scale + _linear(context, out_proj)`, the
residual-scale multiply and add two separate float32 roundings). Every parameter —
dimensions, pairings, operand grids, strides and both capacities — is validated
before the first write: on any error `out` and `scratch` are both untouched. The
scratch layout is `rows * 3C + 4 * rows * C + 2 * heads * 4096` floats (the
projection out, the normalised q, k, v and the context, and one window's logits
and probabilities workspaces), disjoint regions; the buffers must not overlap
(caller discipline, never rejected). The effective width bound `3 * C <=
kMaxLinearDim` caps the assembly at `heads <= 10` (see the attribution limitation
below).

## Explicit choices and disclosed limitations

- The reference's einsums state no summation-order contract; as in the linears
  slice, the explicit ascending-d/j float32 accumulations are this
  implementation's documented convention. No vendor correspondence is claimed for
  the order.
- The int4 mixed-network `_attention` path (`params.i4.qkv`, the `inside` mask) is
  **not** implemented; this slice covers the non-int4 path only, matching the
  linears/MLP slices' scope.
- The tests pin the `heads <= 10` accept/reject boundary (heads = 10 accepted with
  the full 960-wide projection; heads = 11 rejected with `kInvalidDimensions`),
  but no status can attribute the rejection to a specific check — the entry's
  heads bound, the qkv pairing against `3 * channels` and the qkv linear width cap
  all enforce the same limit and return the same status. No single rejecting
  check or check order is claimed.
- `validate_bias_table_output`'s own rows bounds are unreachable through
  `swin_deswizzle_bias`/`swin_exp_bias`: the entries' `out->rows != raw.rows`
  pairing check runs first and confines `out->rows` to the raw table's
  `1..kMaxHeads`, exactly the validator's bounds. The pairing class itself is
  exercised (both a larger and a smaller rows view).
- `swin_softmax_row`'s `SUM_FLOOR` clamp is unreachable through a full 64-key
  softmax row (denominator ≥ 2^-8); it is retained for reference exactness, and
  the tests pin its non-binding bound (`2^-8 > SUM_FLOOR`) rather than executing
  the clamp branch.

## Test evidence (26/10/2026)

`vr/tests/native-swin-attention-test.cpp` with runner
`vr/test-native-swin-attention.sh`: **305 967 checks** in the strict `-O2` build
and again at `-O0` (both `-ffp-contract=off`). Every known answer is a hand-derived
dyadic rational (the exact decompositions are written next to each case). The
corpus covers:

- deswizzle anchors plus an exhaustive 4096-slot bijection and full decode
  mapping; the exp-bias ramp anchors plus an exhaustive sweep of all 63 488 finite
  binary16 raw patterns pinning the single-rounding `swin_fma` against a
  two-rounding re-derivation (they agree everywhere over binary16 raw inputs at
  `c = EXP_OFFSET`);
- the numerator mantissa-to-exponent bit transplant (interior pins, both clamp
  boundaries, two ramp ties-to-even) and a `swin_fma`-vs-two-roundings
  discriminator that survives the binary16 ramp rounding (0.00262451171875 where
  the two-rounding path gives 0.002685546875);
- the denominator sum-order discriminators (block, pair and member/half nesting
  orders, 2050 vs 2048 and 2048 vs 2050 at the `2048 + 1` ties), binary16
  overflow passthrough and the `SUM_FLOOR` clamp bound;
- the head_scale/norm_scale grouping discriminator (0.025390625 where
  per-dimension regrouping gives 0.0234375) and the q-vs-v rounding asymmetry
  (367.875 scales to 352 on the q path and stores to 384 on the v path);
- the logit accumulation-order discriminator (200704 where a smalls-first order
  gives 200704.125) and the store double rounding through the context (367.875 →
  384); permutation and changed-input sensitivity controls; bit-exact equivalence
  of every batch entry against its row/staged chain and of `swin_attention`
  against the full staged chain on a deterministic pseudorandom corpus with
  padded strides;
- sentinel-checked untouched outputs (and untouched scratch in the assembly) on
  every validation rejection class — including the non-finite value classes on
  every scanned operand, the output-rows pairing classes of both bias entries and
  the single-head-scale-row pairing — determinism and zero-allocation accounting
  (all nine entries × 64 iterations).

Development review history: an independent read-only review of the delivered slice
(26/10/2026) re-derived the arithmetic against the pinned source and found no
arithmetic or memory-safety defects; its honesty/coverage findings (this missing
document and its MIT notice, an invalid-value coverage overclaim, a mislabelled
denominator-grouping comment, an untestable heads-bound attribution, a dead
temporary and the output/head-scale row negatives above) were corrected in this
change set and reverified by the suite.

The sanitised build is skipped on this workstation because no sanitizer runtime can
be linked (libubsan/libasan RPMs absent; the runner verifies the skip reason is the
missing runtime and reports the exact linker error) — an environment gap, not a
code failure.

## What this does NOT establish

- No vendor correspondence. The arithmetic matches the pinned MIT reference
  semantics as documented (plus this slice's explicit summation conventions); it
  is not evidence about NVIDIA's implementation.
- No learned weights: every operand tested is an original synthetic value. The
  learned-weight execution graph assembly and its weight bindings remain later
  slices.
- No int4 mixed network, temporal history/reset behaviour, decoder stages,
  live-input/image evaluation or GPU behaviour of any kind.
