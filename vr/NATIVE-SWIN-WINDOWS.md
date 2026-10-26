# Native Swin window partition

CPU-only, typed, allocation-free C++20 layout arithmetic for the windowed-attention
blocks' window partition: the window grid of the shifted window layout and the planar
<-> window-token transforms. These are **pure layout transforms** — values pass
through bit-exactly, no arithmetic is performed on them. This slice supplies the
window partitioning of the Swin body; the e4m3 linears, the attention itself
(position-bias deswizzle, softmax numerators, head scale) and the temporal
history/reset remain later slices. No GPU, headset, proprietary payload, Python or
PyTorch runtime is required.

This is an independent implementation of explicitly documented layout semantics,
adapted with attribution from the MIT-licensed reference implementation
[`mauri870/dlssnr-pytorch`](https://github.com/mauri870/dlssnr-pytorch) at commit
`3beef7935a8ef976b47ab78b906e75600c52d2de` (`dlssnr/layers/swin.py`, the
`_window_grid`, `_to_windows` and `_from_windows` functions). All test shapes and
values are original. **This is not vendor code.** Passing these tests establishes
synthetic layout correctness of this implementation against independently derived
controls only — never correspondence to NVIDIA's implementation. No proprietary
weights, captures or binaries are used or implied.

## Source attribution and licence

Semantics pinned from the reference at the commit above:

> `WINDOW = 8` … `TOKENS = WINDOW * WINDOW` … `TILE = 4` … Attention runs inside
> 8x8 pixel windows (2x2 tiles of 4x4 pixels, 64 tokens). Alternate layers move
> the window grid by half a window; pixels the shifted grid hangs over the frame
> edge read as zero and are not written back.

> `_window_grid(extent, shift)`: (windows along one axis, zero pixels in front) for
> a grid moved by `shift` tiles.

> `_to_windows(x, shift_rows, shift_cols)`: `[1, C, rows, columns]` ->
> `[windows, 64, C]` with tokens in tile order (tile row, tile column, y, x).

> `_from_windows(tokens, grid, height, width)`: the inverse of `_to_windows` for a
> grid it produced, cropped back to `height` x `width`.

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

## The window grid (`swin_window_grid`)

For one axis of extent `extent` and a grid moved by `shift` tiles (tile = 4 pixels):

- `front = -4 * shift` zero pixels are placed in front of the axis;
- `windows = ceil((extent + front) / 8)` windows cover the padded axis.

Only `shift = 0` (unshifted layers) and `shift = -1` (alternate layers, whose grid is
moved by half a window = one tile = 4 pixels, giving `front = 4`) are supported; the
reference raises for anything else and this API returns `kInvalidControls`. The
extent must lie in `1 .. kMaxRows` (`kInvalidDimensions` otherwise).

## Planar batch -> window tokens (`swin_to_windows`)

The channels-first `FeatureBatchView` batch is zero padded to the grid: `front_h` /
`front_w` zero rows/columns in front, the remainder of the `windows * 8` padded plane
behind. The padded plane is split into `8 x 8` pixel windows, and each window's 64
pixels are written in **tile order** (tile row, tile column, y, x):

```
t = (py / 4) * 32 + (px / 4) * 16 + (py % 4) * 4 + (px % 4)
```

for window-local pixel `(py, px)`; window `(wy, wx)` is batch index
`w = wy * window_cols + wx`. Element `(w, t, c)` lives at
`data + w * window_stride + t * token_stride + c`. A token pixel whose source
coordinate `padded - front` falls outside the frame reads as **`+0.0`** (the C literal
`0.0f`), matching the reference's zero pad.

Layout contract (matching the sibling stages' conventions):

- The input batch is channels-first planes with `cols <= row_stride <= kMaxCols` and
  `rows * row_stride <= plane_stride <= kMaxRows * kMaxCols`; token channels must
  equal batch channels (`1 .. kMaxTokenChannels`).
- `token_stride` is in `channels .. kMaxTokenChannels` floats; `window_stride` is in
  `kTokens * token_stride .. kTokens * kMaxTokenChannels` floats.
- `token_capacity` (floats accessible at the token buffer) must be at least
  `(windows - 1) * window_stride + 63 * token_stride + channels` (the last written
  element's offset plus one).
- Only elements `(w, t, c)` for `c < channels` are written; the token- and
  window-stride padding is never touched.
- All validation (dimensions, strides, controls, capacity) completes before the first
  output write: on any error both outputs (`grid_out`, `tokens` including its window
  count) are untouched.
- The token buffer must not overlap the input batch (overlapping writes can silently
  corrupt the copy; all access stays in bounds so this is never rejected — caller
  discipline).

## Window tokens -> planar output (`swin_from_windows`)

The exact inverse mapping of `swin_to_windows` for a grid it produced: each output
pixel `(py, px)` maps to padded coordinate `(py + front_h, px + front_w)`, window
`(padded_y / 8, padded_x / 8)`, token index `t` from the formula above with the
window-local coordinates, and is read from channel slot `c`. The output `dst` is
tightly packed channels-first (`channels * height * width` floats, row-major within a
plane). Pixels the grid hangs over the frame edge are **dropped** — their tokens are
never read and the region is not written back.

Validation (output dimensions, grid shape, crop inside the padded plane, window
count, channels, strides, output capacity) completes before the first output write;
on any error `dst` is untouched. The grid must carry `front` values of `0` or `4` per
axis and a crop inside the `window_rows * 8` x `window_cols * 8` padded plane. Like
aliasing, the token buffer's extent is caller discipline (there is no capacity
parameter): it must be readable over `(windows - 1) * window_stride + 63 *
token_stride + channels` floats. The token buffer must not overlap `dst` (same
caller-discipline caveat as above).

## Arithmetic semantics

There are none: this slice is a pure layout transform. Values are loaded and stored
as plain `float`, so they pass through **bit-exactly**, including signed zeroes,
infinities and NaN payloads. The pad constant is `+0.0`. No rounding, no
transcendentals and no `-ffp-contract` sensitivity is involved; the runner still
builds with `-ffp-contract=off` like every other stage.

## Explicit choices and disclosed limitations

- Only shifts `0` and `-1` are supported (the reference raises for anything else).
  "Half a window" shifting is `shift = -1` tiles of 4 pixels, `front = 4`.
- Window grids are bounded by the sibling conventions (`kMaxRows`/`kMaxCols` =
  16384 per axis, so at most `kMaxWindows` = 2049 x 2049 windows; token channels at
  most `kMaxTokenChannels` = 256). Overlap between buffers is a documented caller
  discipline, not a detected error.
- `swin_from_windows` validates the grid it is given; it does not verify that the
  grid was produced by `swin_to_windows` (it checks shape consistency only).
- The sanitised build is skipped on this workstation (no linkable sanitizer runtime;
  the runner verifies the skip reason is the missing runtime and reports the linker
  error) — an environment gap, not a code result.

## Test evidence (26/10/2026)

`vr/test-native-swin-windows.sh` — 4 360 checks in the strict `-O2` build and again
in the `-O0` build (both compiled with `-ffp-contract=off`). Coverage includes:

- Hand-derived grid literals for both shifts, including the boundary cases
  (`extent 8/9/12` unshifted, `4/5/12/13/16` shifted) and the maximum axis
  (`kMaxRows` -> 2049 windows, `front 4`, pinning `kMaxWindows`), plus the invalid
  shift and extent classes and null outputs — every grid rejection verified to leave
  both outputs untouched.
- Token-index pins: the full 64-token x 2-channel tile-order decode of a `2 x 5 x 3`
  batch (unshifted) with stride-padding invariants (sentinel-checked) — this fixture
  is a re-derivation of the documented index formula, not an independent oracle —
  and genuinely hand-derived pins of the shifted grids: the `2 x 1` window grid
  (`front 4/4`) with zero-pad tokens, boundary tokens crossing the crop and interior
  tokens (including the `x = 1` and `y = 1` decode cells), a `2 x 2` window grid
  (`9 x 7`, both shifts) pinning the window-batch index order `w = wy *
  window_cols + wx` absolutely (a transposed order fails), and an asymmetric-shift
  fixture (`5 x 3`, `shift_rows = -1`, `shift_cols = 0`) pinning the shift-axis
  assignment. The pad is pinned as `+0.0` by bit comparison, not `== 0`.
- A bit-exact round-trip oracle: 500 randomised shapes/shifts/strides with special
  values (signed zeroes, infinities, several NaN payloads, ordinary dyadic values),
  `memcmp`-exact round trips and determinism reruns (identical inputs give
  bit-identical tokens).
- Locality and permutation controls: a single-pixel change moves exactly one token
  element; swapping two input channels swaps exactly those channel slots in every
  token.
- Negative controls for every validation rejection class of all three routines (null
  pointers, dimensions, strides, controls, view mismatches, short capacity), each
  with sentinel-checked untouched outputs (the null-output rejection cannot write by
  construction; null-input rejections with live outputs are checked untouched too).
  The capacity boundary is pinned from both sides: 317 floats are accepted and 316
  rejected for the single-window fixture ((windows - 1) * window_stride + 63 *
  token_stride + channels).
- Zero-allocation accounting across repeated transforms (no heap use in either
  direction).

## What this does NOT establish

- No vendor correspondence. The layout matches the pinned MIT reference semantics as
  documented; it is not evidence about NVIDIA's implementation.
- Not the body: no e4m3 linears, attention, position bias, softmax, MLP assembly and
  no learned weights — every value moved here is an original synthetic value. The
  learned-weight execution graph remains later slices.
- No temporal behaviour, no live-input evaluation, no GPU behaviour of any kind.
