# Native temporal history carrier

`vr/native-history.hpp` implements the temporal recurrence state machine that
wraps a network run: one FP16 colour-history map per feature, the
frames-since-reset counter, and the reset contract. CPU-only,
allocation-free; no GPU, headset, proprietary payload, Python or PyTorch
runtime required. No vendor code and no vendor correspondence.

## Contents

* `TemporalCarrier` — the per-feature state over caller-provided storage:
  the interleaved colour history map (`rows * cols` pixels of three floats,
  `map_capacity` in **float elements**), the counter and the history flag.
* `temporal_init` / `temporal_reset` — the reset value is all zeros (the
  capture's `history0-init.bin` is uniform `0x00`) with the counter at 0 and
  the flag dropped.
* `temporal_history_view` / `temporal_counter` / `temporal_has_history` —
  the inputs of the frame about to be evaluated: the history colour (the zero
  map at start-up and after a reset) and the frames-since-reset counter.
* `temporal_commit` — `history_{N+1} = output_N` **bit-exactly** (values must
  be finite and binary16-representable; anything the FP16 texture could not
  hold is rejected rather than rounded), then the counter advances. The
  source is read through its own row stride (padding never read); the copy is
  `memmove` per row, so a self-commit through `temporal_history_view` is
  legal. The counter refuses to advance past `UINT32_MAX` rather than
  wrapping into the reset-frame value 0.

## The contract and its evidence

The behavioural contract is this repository's own capture evidence
(`TEMPORAL-HISTORY.md`, gate stage 13, Proton-side; tested regime: eye0,
720x720, stationary gradient inputs, 8 frames):

```
output_N = M(inputs, history_N, counter_N)   history_N = output_{N-1} (bit-exact)
counter_N = N since the last reset
Reset -> behave as if history = 0, counter = 0
```

The evidence cannot distinguish "reset zeroes the texture" from "the model
ignores the history when its counter is 0"; this stage implements the
strongest form consistent with both readings — the behavioural form, with the
carrier bytes zeroed at reset (so staged history content is ignored, the
capture's `injectreset` control) and the flag dropped. The captured counter
sequence across a mid-stream reset at frame 4 is `[0,1,2,3,0,1,2,3]`, pinned
as a literal in the suite.

### A naming correction

Earlier handover text referred to pinned temporal stages "History / Reset2 /
Reset1 / ResetNone" and "their weight records". No such stages or records
exist in the pinned reference: `model.py`'s run loop is frame -> 152 layers ->
output with no temporal stage, `Controls.history`/`Controls.reset` are
documented as "accepted ... and are ignored" in the single-frame path, and
`plan.py`/`extract.py`/`weights.py` contain no history/reset/counter names.
The pinned reference is single-frame by construction. This slice therefore
implements the capture-evidence contract above, not reference code; the
weight-record framing was a handover error and is withdrawn.

### Scope limits (what this does NOT establish)

* The input lift's feature slots 7..9 (`native-input-lift.hpp`) keep their
  documented single-frame stub (the current-colour copy) — deliberately
  untouched here per instruction; binding the history colour into those slots
  is the live-input stage's integration step.
* Which stage consumes the frame counter is an unestablished boundary of the
  proprietary route (`TEMPORAL-HISTORY.md`'s "Next"); the counter is tracked
  here and not injected anywhere.
* The history's exact numeric domain at the network input (the capture's
  carrier is the FP16 output texture; the lift centres the colour) is the
  live-input stage's binding decision, not established here.
* No temporal quality, motion handling, or vendor behaviour is established.

## Reference and provenance

The `FrameView` type and the `is_f16` grid check come from
`native-input-lift.hpp`; the `Controls.history`/`Controls.reset` field
semantics ("accepted ... and are ignored") are from the MIT reference
`mauri870/dlssnr-pytorch` commit `3beef7935a8ef976b47ab78b906e75600c52d2de`
(`dlssnr/io.py`). No network weights, captures, raw payloads or binaries are
used or reproduced.

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

`vr/tests/native-history-test.cpp` (281 checks at -O2 and again at -O0 via
`vr/test-native-history.sh`) pins the contract: the all-zero init value with
the documented failure form (map fields only; the caller's buffer
byte-untouched; a failed re-init leaves the counter and flag untouched); the
pinned counter sequence `[0,1,2,3,0,1,2,3]` across a mid-stream reset at
frame 4; commit bit-exactness including the grid edges (0.0, -0.0, 2^-24,
65504.0) compared with `memcmp` so a rounding hop or a sign-of-zero flip is
visible; `history_N = output_{N-1}` read through the view's own stride on
non-square maps; a stride-poison control (NaN and wild constants in a padded
source's padding, which an `r*cols`-style bug would bring into the carrier);
the reset-ignores-staged-history control (`injectreset`) with idempotence;
the rejection matrix (null/geometry/stride/value classes, a shrunk capacity
(`kInvalidBuffer`), and counter exhaustion (`kInvalidValues`)) with
sentinel-checked untouched state; determinism across a reset+recommit; and
zero-allocation accounting with the repo's full allocation-override block.
Independent read-only review verified the contract ordering against
`TEMPORAL-HISTORY.md`, the `is_f16`/`isfinite` semantics, stride discipline
and validation-before-write, and raised seven findings, all fixed and
reverified: the undocumented counter wrap (now a refusal), the overstated
untouched-on-error claim for `temporal_init` (now true: the failure form
touches the map fields only), the missing upper bounds (now the input-lift
`kMaxRows`/`kMaxCols`/`kMaxPixels` pattern with a stride bound), the
incomplete allocation override (now the full block), bit-exactness under-pinned
by `==` alone (now memcmp at the grid edges), three untested rejection
classes (now tested), and `memcpy` UB on a legal self-commit (now `memmove`).
The sanitised build is skipped on this toolchain (missing
`/usr/lib64/libubsan.so.1.0.0`), a recorded environment gap.

## Non-vendor statement

This is not vendor code and not a reimplementation of any vendor's protected
temporal machinery. It follows this repository's capture-evidence contract
only and never corresponds to NVIDIA's implementation. No proprietary
weights, captures, binaries or proprietary runtime are used or redistributed;
the tree performs no runtime or capture probes. Synthetic known answers
establish synthetic correctness only — never vendor parity.
