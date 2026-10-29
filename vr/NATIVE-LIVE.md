# Native sustained live-input evaluation

`vr/tests/native-live-test.cpp` (runner `vr/test-native-live.sh`) is the
sustained-run suite for the native evaluator: it feeds full-network frames
through the real consumer (`temporal_image_evaluation`) in multi-frame
sequences and asserts the sustained-run contracts — deterministic replay,
input responsiveness, balanced allocations, constant arena high-water and
bounded resident-set growth — with a measured throughput report. CPU-only;
no GPU, headset, proprietary payload, Python or PyTorch runtime required.
No vendor code and no vendor correspondence.

## What the suite runs

Three sequences over the small plan (76 rows, 64x72 frames, 320x320 working
extent, synthetic seeded records):

1. **Sustain + state** — `depth` consecutive frames with a reset before frame
   `depth/2`; the per-frame counter must follow the capture's schedule
   (`0..reset_at-1`, then `0..` again).
2. **Deterministic replay** — the identical sequence on a fresh carrier must
   be bit-identical frame for frame (memcmp of the full output frames) with
   the same counter schedule.
3. **Input responsiveness** — a different input stream (different salt) must
   produce different output frames everywhere, with the no-reset counter
   `0,1,2,..`.

Per frame the harness also asserts:

* **Balanced allocations** — the live-allocation count (operator new minus
  operator delete, the `native-vulkan-stress-test` pattern) is back at its
  baseline after every frame. The parallel layer's documented spawn-and-join
  design allocates transient thread state per `parallel_for` call, so the
  contract is balance, not zero churn; the churn count is measured and
  reported (~89,320 per frame at 8 threads).
* **Constant arena high-water** — `eval_arena_used` is identical after every
  frame (the arena rewinds per run).
* **Bounded RSS** — the resident set (honouring the host page size) is read
  after warm-up (the end of the first frame, so first-fetch memory does not
  count) and at the session's end; growth after warm-up across the remaining
  frames measured 76-276 KiB.

## Measured (RTX 4070 Ti SUPER host, -O2, 8 threads, 9 frames)

| metric | value |
|---|---|
| mean frame | **8,260-8,440 ms** (0.119-0.121 frames/s) |
| decode (`frame_from_u8`) | 0.24 ms |
| pipeline (derived) | 8,260 ms |
| output (`frame_to_u8`) | 0.03 ms |
| allocation churn | 89,320/frame (balanced) |
| arena high-water | constant |
| RSS growth | 76-276 KiB over 9 frames |

Speed numbers are real measurements of the synthetic-weight pipeline; the
output content is an arbitrary transform — no image-quality claim. A fresh
verification run on 29/10/2026 (host under concurrent load) measured
8,921 ms mean frame at the same level.

## Suite levels

* `-O2`, 8 threads, depth 3 (three 3-frame sequences): the full soak; 87
  checks.
* `-O0`, 8 threads, depth 2 (six frames across the three sequences): **runs
  the assertion classes** (reinstated 29/10/2026 with the uninitialised-read
  investigation's closure below; it was compile-only while that record was
  open). Measured 36-48 s/frame under load.
* sanitised: skipped on this toolchain (missing sanitizer runtime), the
  recorded environment gap shared by the family's suites.

## The uninitialised-read investigation (closed 29/10/2026 — not reproduced; the runner's execution cap was the confound; content independence pinned by regression)

The record this closes (kept verbatim below) described an intermittent stall
on the second network run of long unoptimised runs, attributed to an
uninitialised heap read proven by `MALLOC_PERTURB_=165` flipping a clean
4-run probe into stalling on run 2. The 29/10/2026 localisation could not
reproduce any part of the stall and identified the confound that produces
exactly the recorded signatures:

* **The workspace's command runner kills foreground commands at ~120 s**
  (recorded in `.tools/job-cap-ticks.log`: a 10-second-tick heartbeat
  reaching tick 13 — 120 s of wall time — then killed mid-tick; a 100 s
  sleep survives, anything longer dies). An -O0 network frame is 36-94 s
  (8-1 threads), so any -O0 probe of several runs straddles the cap and is
  cut mid-run — which is the recorded "stall on run 2" shape, with the
  completed run's hash unchanged. The overnight investigation's own logs
  show the same shape (`.tools/stall-a1.log` ends mid-frame-3,
  `.tools/watch2.log` ends at ~120 s without its footer). -O2 runs are
  ~8 s/frame and never reach the cap ("O2 immune even under
  MALLOC_PERTURB_"). The original probe's per-run cost was not recorded,
  so the exact cap boundary of that observation cannot be reconstructed;
  what is established is that every long-run failure observed in this
  workspace — including this investigation's own first attempts — ended at
  ~119-125 s of work. Detached execution (`setsid nohup`) escapes the cap;
  every observation below is a detached run.
* **`MALLOC_PERTURB_` sweeps at -O0**: seven runs (six at 6 frames, one at
  3 frames) across fills {165, 255, 1, 127, 128, 254} and 1 thread —
  **39 frame hashes identical**, no stall, uniform per-frame timings.
* **Arena poisoning at -O0/-O2** (the whole 512 MiB arena painted before
  every frame with NaN `0xFF`, denormal `0x01`, `0xA5`, and per-frame
  pseudorandom bytes): frame hashes **bit-identical** to the unpainted
  baseline. This probes what `MALLOC_PERTURB_` cannot: the fresh-malloc
  fill bytes always pair to finite floats, never NaN.
* **ASan+UBSan at -O0, 4 full frames: clean** (no heap overflow, no UB).
  **valgrind memcheck `--track-origins=yes`, 2 full -O0 frames** (the
  fresh-run and the leftover-state run; `.tools/job-memcheck.log`): both
  frames `DONE status=0` with hashes identical to the clean runs, heap
  balanced (268,241 allocs/frees), **ERROR SUMMARY: 0 errors from 0
  contexts**.
* **Live suite at -O0: 87 checks pass**, unperturbed and under
  `MALLOC_PERTURB_=165`, with bounded RSS in both.
* **The recorded candidates are pinned directly** by
  `vr/tests/native-uninit-test.cpp`: the swin scratch partitions and the
  per-row decode staging, poisoned with 5 poisons (denormal, the perturb
  fill byte, 0xA5, NaN, random) against a zeroed baseline (plus the
  zeroed-control repaint), produce **bit-identical** `swin_core`,
  `swin_mlp` and `swin_attention` outputs for all four head families
  (246,186 checks at -O2), plus a full-network arena sweep and a literal
  cross-process `MALLOC_PERTURB_` sweep in `vr/test-native-uninit.sh`
  (every perturb fill prints identical output at -O2 for
  1/127/128/165/254/255 and at -O0 for 165/255).

Conclusion: no uninitialised read with any observable output, control-flow
or timing effect exists in the exercised paths; the swin scratch partitions
and the decode staging are write-before-read before any use. The recorded
stall did not reproduce in roughly ninety evaluated -O0 frames across 8
heap-fill configurations and 2 thread counts. A hypothetical read whose
garbage affects only the timing of dead computation cannot be excluded by
an output regression, but nothing in the timing data supports one (frame
times were uniform across all fills) and no such read was found by memcheck.
The defect record is retired into this investigation;
`vr/test-native-uninit.sh` (the MALLOC_PERTURB_ regression) and the
reinstated -O0 runs are the standing guard.

### The original record (historical, 28/10/2026)

During development an intermittent stall appeared on the *second* network
run of long unoptimised runs. The investigation established:

* **The evaluator is deterministic and correct**: four consecutive runs on
  one process produced **identical output hashes** and identical per-row
  timings; the -O2 suite's replay memcmp passes repeatedly.
* **An uninitialised heap read exists**: `MALLOC_PERTURB_=165` flips a clean
  probe (4 clean runs) into stalling on run 2, without changing run 1's
  output hash. Auto-initialised locals (`-ftrivial-auto-var-init=zero`) do
  not change the behaviour, so the read is past a buffer, not an uninitialised
  local.
* **The stall is a slowdown of normal work** (stack samples land in the swin
  softmax/MLP kernels and move through the pipeline; no spin, no paging, no
  kernel time, calibration loops stay flat) — consistent with garbage values
  driving pathological per-op costs or loop extents.
* **Exposure**: -O0 codegen only. -O2 runs are clean at both thread counts,
  including with `MALLOC_PERTURB_` set. The stall is not data-dependent
  (identical inputs reproduce it) and not thread-count-dependent at -O0.
* Ruled out: host contention/throttling (wall == CPU at full boost; the
  host's `powersave` governor was checked), buffering artefacts (stderr is
  unbuffered), denormal assists alone (FTZ/DAZ shifts but does not clear
  it), the record store, the arena allocator, the consumer phases.

The exact site is not yet localised. It does not affect the -O2 deliverable
builds or any output in fifteen observed runs; it is the recorded next
action for the follow-up slice (localise the read — candidates are the swin
scratch partitions and the per-row decode staging in `native-eval.hpp` —
then add the regression that reproduces under `MALLOC_PERTURB_`).

## Review

Independent read-only review: **FAIL with 6 findings (4 MINOR, 2 NIT)** —
all fixed and reverified (87 checks at -O2). The findings: the reset-frame
counter assertion was tautological (the reset frame's "observed" counter was
hard-coded, so `counters[reset_at] == 0` asserted 0 == 0) — now the carrier's
POST-call counter is asserted directly (1 after a reset frame, recorded+1
otherwise), which observes the reset at any depth; the fixed 320x320 probe
maps assumed the pinned working floor without asserting it (now a
`padding.width == 320 && padding.height == 320` guard fails loudly if the
padding rule ever yields more); stale comments claiming the -O0 level runs
the assertions (it is compile-only) and the serial configuration was covered
(8 threads are pinned at every level) were corrected; the RSS claim was
wrong (the baseline is taken after warm-up, not before the session) and is
now precise; `read_rss_kib` hard-coded 4 KiB pages and now honours
`sysconf(_SC_PAGESIZE)`. The review hand-derived the 320x320 working extent
and every buffer size, verified all API call orders against the headers, and
confirmed the replay/responsiveness/allocation-balance checks are
non-vacuous.

## Provenance

Independent implementation in typed C++ following this repository's pinned
capture-evidence contract (`TEMPORAL-HISTORY.md`, `vr/native-history.hpp`
carrier semantics) and the network semantics of the pinned reference
`mauri870/dlssnr-pytorch` at `3beef7935a8ef976b47ab78b906e75600c52d2de`.
The MIT licence text for that reference follows.

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

## Non-vendor statement

This is not vendor code and not a reimplementation of any vendor's protected
machinery. It follows this repository's capture-evidence contract only and
never corresponds to NVIDIA's implementation. No proprietary weights,
captures, binaries or proprietary runtime are used or redistributed; the
tree performs no runtime or capture probes. Synthetic known answers and
synthetic speed measurements establish synthetic correctness and measured
throughput only — never vendor parity.
