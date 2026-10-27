# Native C++ network graph plan (geometry layer)

`vr/native-graph-plan.hpp` is the deterministic, allocation-free C++ geometry
plan for the whole DLSSnR-style generator network: the typed layer table
(152 layers / 71 blocks for the "ht" model, 76 layers / 31 blocks for "small"),
per-layer extent walking, the padding rule, grid/shift derivations and the
residual skip wiring. It is the map later native layer families and the full
encoder-decoder assembly follow. Like the other slices it is a clean C++
re-derivation of published open-source semantics — **not vendor code**.

## Reference and provenance

The geometry semantics are re-derived from the pinned MIT reference
`mauri870/dlssnr-pytorch` commit
`3beef7935a8ef976b47ab78b906e75600c52d2de` (`dlssnr/plan.py`, `dlssnr/model.py`
`_rows`/`_skip_sources`/`build_plan`). The pinned snapshot used is
`/tmp/plan-pin.py` (sha256
`8d7b353e931af5a8de98d96e711a3f3ed26f71bf7af5bb451805cfb71b2887c35`); the
pinned docstring's "77" is the number of weight-record positions (both models
share the first 77 records), **not** a layer count — the real counts are 152
layers / 71 blocks ("ht") and 76 layers / 31 blocks ("small"), derived
independently in C++ by the test (the pinned Python is never imported).

This slice publishes no network weights, captures, raw payloads or binaries and
reproduces no proprietary routing or addressing.

## MIT licence (reference-derived sections)

Copyright (c) 2026 Mauri de Souza Meneguzzo

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Layer types

* `VitPatchify` — patch embedding (stride 4, `4*4*ch` tokens per patch).
* `VitQKV` / `VitAttention` — pre-norm ViT self-attention (head count from the
  table; window rows use 1×w×h windows).
* `VitProj` — attention output projection + residual.
* `Conv` — `k×k` convolution with explicit output channels, no residual.
* `SplitProj` — decoder-side linear that splits into `q*kv` head groups plus an
  independent `k*q` skip token projection (the skip tensor keeps its own token
  count; the merged attention stream is `q*kv`).
* `Upsample` — nearest-neighbour 2× width/height upsampling (channels unchanged).
* `InputBlock` / `OutputBlock` — the two pixel-shuffle boundary blocks: the
  input block folds a 4×4 pixel neighbourhood into channels (`16*in_channels`
  out, stride 4, `in_token_count` in); the output block unfolds channels into
  4×4 RGB pixels (`3*16` out, stride ¼, `out_token_count` out). Tokens move per
  input pixel at the input edge and per output pixel at the output edge.

## Geometry rules

* **Padding** — the walk starts at the raw input extent and pads both dimensions
  to the next multiple of the model alignment (32 for "ht", 64 for "small" —
  the small model uses the wider alignment). Tokens are `w*h` per row; the
  aligned dimensions are the canonical extent for all later layer families.
* **Extent walk** — Conv/attention rows keep the extent, Upsample rows double
  both dimensions, InputBlock divides by 4, OutputBlock multiplies by 4. Halving
  rows are a network topology detail of later slices, not a plan-level extent
  change here.
* **Grid/shift** — Window rows derive `(gx, gy)` = `(w % 2, h % 2)` and
  `(sx, sy)` = `(gx, gy)`; ViT rows are `(0,0)`. Grid meaning is normalised as
  `grid.w = w / 2`, `grid.h = h / 2`.
* **Skip wiring** — every block has `skip_source` (an earlier row index) and
  `skip_join` (which head group receives the residual); `is_skip_source` marks
  rows that later rows read. The final OutputBlock's `skip_source` is 0
  (no skip).

## Known-answer test

`vr/tests/native-graph-plan-test.cpp` validates the table against independently
hand-derived answers (no pinned Python execution):

* layer/block counts for both models (152/71 and 76/31);
* the complete padding table (raw input → aligned extent) including the
  32/64-alignment split;
* extent walk (stride-4, upsample 2×, stride-¼) and token counts;
* type-sequence anchors at the raw rows (input edge, first block boundary,
  centre block, the row where ht grows past small, output edge, every anchor
  carrying its exact `channels`);
* grid/shift for normal and shifted windows;
* skip wiring (source indices and join groups, including the final-block
  source of 0);
* extent coherence (every layer's output extent is consistent with its kind),
  rejection of malformed tables and fully deterministic enumeration;
* the high-width boundary: `w=16384` stays `int`-safe end-to-end
  (`tokens = 269,484,032` at the output edge);
* zero-heap allocation of the whole plan walk (accounting new/delete).

`vr/test-native-graph-plan.sh` builds the test like the other runners
(`-std=c++20 -Wall -Wextra -Werror -ffp-contract=off`, both `-O2` and `-O0`).

## Non-vendor statement

This is not vendor code and not a reimplementation of any vendor's protected
network layout. It follows the MIT reference's model geometry only and never
corresponds to NVIDIA's implementation. No proprietary weights, captures,
binaries or proprietary runtime are used or redistributed, and the tree
performs no driver-level runtime/capture probing.
