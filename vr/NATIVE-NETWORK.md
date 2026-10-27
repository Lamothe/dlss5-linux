# Native CPU network dispatch (the plan-row run loop)

`vr/native-network.hpp` implements the whole-network dispatch: which tensor
every one of the plan rows reads, the tuple semantics of the multi-output
layers and the last-use release of finished tensors — the wiring that turns
the typed plan of `native-graph-plan.hpp` into an executable run. The layer
kernels stay in their own families (`native-pre-post`/`native-blend`/
`native-compose`, `native-swin-*`, `native-split-swin`, `native-vit`,
`native-decoder`); this header routes them, it does not reimplement them, and
the learned-weight binding and the synthetic input-to-image evaluator are the
next slices. CPU-only, allocation-free; no GPU, headset, proprietary payload,
Python or PyTorch runtime required. CPU-only clean C++ re-derivation of
published open-source semantics — **not vendor code**.

## Contents

* `LayerValue` — one row's produced tensors: `main` (the pinned tuple element
  0, what the next layer reads) and `aside` (element 1). The producers of
  `(main, aside)` are the pre block `(pooled, full-resolution body)`, the `ds`
  Swin layers `(projected pooled map, body)`, `SplitFfwd` `(out, x)` and
  `SplitProjPool` `(projection, pooled)`; every other kind returns one tensor.
  `SplitFfwd`'s aside is its input's own tensor (the unchanged block input),
  so it aliases the previous row's main — the release schedule accounts for
  that alias.
* `LayerCall` — the exact inputs of one layer call, selected per the pinned
  layer modules' forwards: `x` (the main input; the pre block has none and
  reads the image), `aux` (the kind's second input) and `image` (the padded
  frame; the pre and post blocks only). `aux` is the skip source's main for
  `SplitProj`/`SplitProjPool`/`VitFfnContract`/`VitProjection`/
  `DecInputUpsample` (`skip[0]`/`_main`), the skip source's aside for the
  Swin `upsample` variant (`skip[1]`, falling back to the whole skip) and for
  `PostBlock` (`skip_value[1]`), the previous row's aside for
  `SplitFfwdProj` (the raw pair; `skip` stays None when the previous is a
  single tensor), and the previous row's main for `SplitFinalHead` when its
  skip source is not a pair (`pooled = skip[1] if isinstance(skip, tuple)
  else x`). The plain bodies read only `x` (the pinned passes their skip value
  positionally; their forwards ignore it).
* `layer_produces_pair` / `layer_reads_image` — the per-kind output and image
  contracts.
* `network_last_uses` — the pinned `NRNet._last_uses`: every output is read by
  its sequential reader and by every layer naming it as a skip source; the
  maximum is the row's last reader.
* `network_run` (`NetworkRun`) — the pinned `NRNet.run` loop over the plan
  rows: the skip source of every row (the pinned `NRNet._skip_sources`, via
  the plan family's `plan_skip_source`), the per-kind call contract above, the
  produced-value contract check before the value is routed on, and the
  last-use release (a row's buffer is released exactly once, at its last
  reader; the final row's output never leaves the run). A `SplitFfwd` row's
  input buffer is released at the later of its own release and the
  `SplitFfwd` row's release (the pinned tuple keeps the tensor alive past its
  dict entry). A null `ReleaseFn` keeps every output (the pinned
  `capture=True`). Upfront validation (nulls, row indices in plan order,
  kind/variant ranges, output extents, image geometry, skip sources earlier
  than their readers) completes before the first write: on an upfront error
  every output stays untouched. Mid-run structural rejections fail fast:
  completed rows keep their outputs, the failing row's entry is whatever its
  layer last wrote (or untouched when rejected before its layer runs), and
  rows not reached stay untouched.

## Reference and provenance

Semantics re-derived from the pinned MIT reference `mauri870/dlssnr-pytorch`
commit `3beef7935a8ef976b47ab78b906e75600c52d2de` (`dlssnr/model.py`
`NRNet.run`/`NRNet._skip_sources`/`NRNet._last_uses` and the forward contracts
of `layers/pre_post.py`, `layers/swin.py`, `layers/split_swin.py`,
`layers/vit.py`, `layers/decoder.py` at the same pin). No network weights,
captures, raw payloads or binaries are used or reproduced.

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

## Known-answer test

`vr/tests/native-network-test.cpp` (3,564 checks at -O2 and -O0 via
`vr/test-native-network.sh`) validates the wiring against hand-derived answers
re-derived from the pinned rules independently of the header: an exhaustive
pair/image contract check over every (kind, variant) cell; full-plan walks of
the real "ht" plan (152 rows, two frame sizes) and "small" (76 rows) asserting
every row's `x`/`aux`/`image` pointer identities (the previous row's main, the
exact skip element, the padded frame), the produced pair forms, the skip
source and last-use schedules and the release order, all against test-side
re-derivations of `_skip_sources`, `_last_uses`, the del-loop and the tuple
lifetimes; mini plans for the edge contracts (a lone pre block, a two-row
chain, the skip span with the pool/final-head/decoder element readers, the
upsample and final-head element fallbacks, the `SplitFfwd` pair with its
buffer alias, capture mode); a rejection matrix covering every upfront class
(nulls, count bounds, row order, negative and oversized enums, output
extents, image geometry) with outputs and scratch verified untouched, every
mid-run class (layer error, pair-contract and extent violations, missing
second inputs for the post/upsample/skip-main groups, missing previous), and
self/future skip sources; a buffer-poison control (any call touching a
released buffer is flagged — this is the control for the `SplitFfwd` alias
lifetime); determinism reruns and zero-allocation accounting. An independent
read-only review walked every dispatch arm against the pinned source (routing
found faithful) and raised five findings, all fixed and reverified: the
`SplitFfwd` alias could be released before its reader under a freeing
`ReleaseFn` (now deferred to the aliasing row's release; the poison control
and the release-schedule re-derivation pin it, and a seeded removal of the
deferral fails exactly at the poison control), an overstated mid-run comment,
the missing earlier-row requirement on skip sources (self/future sources now
rejected before the first call; a seeded removal fails exactly at that case),
four rejection-matrix gaps, and this document. The sanitised build is skipped
on this toolchain (missing `/usr/lib64/libubsan.so.1.0.0`), a recorded
environment gap; the suite runs at -O2 and -O0 with
`-Wall -Wextra -Werror -ffp-contract=off`.

## Non-vendor statement

This is not vendor code and not a reimplementation of any vendor's protected
network. It follows the MIT reference's dispatch semantics only and never
corresponds to NVIDIA's implementation. No proprietary weights, captures,
binaries or proprietary runtime are used or redistributed; the tree performs no
runtime or capture probes. Synthetic known answers establish synthetic
correctness only — never vendor parity.
