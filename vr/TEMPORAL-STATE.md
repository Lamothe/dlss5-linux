# Temporal-state contract (gate stage 12)

The model is a *history blend*
(`CC_Control_History_Blend_Quantize_With_Teacher...`, see
[REFERENCE-CAPTURE.md](REFERENCE-CAPTURE.md)): outputs evolve across frames
despite stationary inputs, and a `dlssnr_prev_output` 4 MiB texture per eye is
part of its allocations. This stage establishes the **contract of that temporal
state** — what `DLSSNR.Reset` clears, whether anything else persists across
frames, and which buffers carry the history — before any native temporal
reconstruction is attempted.

## The experiment: mid-stream reset, two unequal offsets

`feature-probe.exe --capture-reset-mid DIR [FRAME]` (`capture-reset-mid-v1`)
repeats the controlled reference capture — identical stationary
gradient/depth/motion inputs, fresh NaN output each evaluation,
`DLSSNR.Reset=1` at frame 0 — but adds a **second reset at the given frame**.

Two precise predictions, from (P1) reset clears the effective cross-frame state
and (P2) the runtime is time-homogeneous (no absolute frame index in the
computation):

1. **Restart equivalence** — frames after the mid-stream reset must replay the
   run's first frames bit-for-bit (same inputs, same cleared state,
   deterministic runtime).
2. **Mid-reset divergence** — those frames must *differ* from the unreset
   reference run's corresponding frames (which keep evolving).

**Why two offsets:** a single run with segments of equal length (4+4) cannot
distinguish "full reset + time-homogeneous" from a frame-index-dependent
behaviour that is periodic with period 2 or 4 (a surviving frame counter
consumed mod 4, for example). The second run resets at frame 5 (segments 5+3).
A behaviour can only pass both runs if it is periodic with a period dividing
both restart strides — **no period in 2..7 survives**, so small frame-counter
aliasing is excluded.

Runs `temporal-a`/`temporal-b` (reset at 4) and `temporal-c`/`temporal-d`
(reset at 5):

| Relation | Result |
|---|---|
| restart at offset 4: frames 4–7 vs 0–3 | **bit-identical** (4/4 pairs) |
| restart at offset 5: frames 5–7 vs 0–2 | **bit-identical** (3/3 pairs) |
| prefix: frames before the reset vs `run-a` | **bit-identical** (4 + 5 pairs per run kind; the shared protocol) |
| divergence: post-reset frames vs `run-a` | differ (4/4 and 3/3 pairs) |
| pairs: `temporal-a`=`temporal-b`, `temporal-c`=`temporal-d` | **bit-identical** (11 files per pair; 22 comparisons) |

### What this establishes (scoped to the evidence)

- **No observable cross-frame state distinguishes the post-reset frames from
  the run's first frames** under stationary, zero-motion, eye0 inputs. The
  restart is exact at both unequal offsets, so no frame-index dependence with
  period 2–7 and no other residual state (accumulated counters, RNG) survives
  `DLSSNR.Reset` in any way visible at the output. This is consistent with
  reset clearing the *complete* effective state; state that is invisible under
  these inputs (e.g. motion-history state) is not excluded.
- **Within that scope the recurrence is time-homogeneous**: the observed
  behaviour is `state' = F(inputs, state)`, `output = G(inputs, state)`. A
  native reimplementation matching these conditions needs the history
  recurrence and the reset value of the state; frame indices beyond the reset
  offsets' exclusion are not ruled out (e.g. a period-8 alias), and unequal
  offsets in *every* future sequence comparison remain mandatory.
- **Deterministic**: identical sequences across independent runs (consistent
  with the established reference-capture determinism).

Consequence for the native work: the temporal stage reduces to recovering the
one-step recurrence and the reset value of the state — not a family of
frame-indexed behaviours — under the scope above.

## Boundary carriers (where the history flows)

Changed bytes between consecutive frames of the retained decoder-boundary
capture ([DECODER-BOUNDARY.md](DECODER-BOUNDARY.md) run `a`; identical runs
`a`/`b` were pinned there):

| Window | 0→1 | 1→2 | Classification |
|---|---:|---:|---|
| `w0` main-input | 184,079 | 162,524 | **temporal carrier** |
| `w1` skip-input | 1,780,260 | 1,522,698 | **temporal carrier** |
| `w2` out0 | 1,891,782 | 1,716,572 | **temporal carrier** |
| `w4` counter region `[0,512)` | 0 | 0 | static (values ∈ {0, 3}) |
| `w6` partial-sums | 1,626,476 | 1,389,748 | **temporal carrier** |
| `w7` weights | 0 | 0 | static |

The `w4` window beyond `[0,512)` churns (pool reuse); only the counter region
is classified. The history therefore reaches the decoder **through its inputs**
(main and skip), consistent with the earlier finding that temporal state enters
before the decoder; `out0`/`partial-sums` are the decoder's own per-frame
writes. The `w7` weights window being bit-static across frames is the
in-window control.

(Note: the boundary capture's `output-frameNN.bin` files all share digest
`0xa0aa045f6a5e4325` — the untouched NaN sentinel; the post-decoder truncation
held for every frame.)

## Output dynamics (report-only)

Per consecutive pair of the unreset reference run, changed samples of 2,073,600
and mean absolute difference / max in FP16 value space (bit-reinterpreted IEEE
half):

| Pair | changed | MAD | max |
|---|---:|---:|---:|
| 0–1 | 69.2% | 0.000603 | 0.0098 |
| 1–2 | 69.2% | 0.000526 | 0.0063 |
| 2–3 | 66.6% | 0.000450 | 0.0083 |
| 3–4 | 66.7% | 0.000425 | 0.0073 |
| 4–5 | 69.0% | 0.000537 | 0.0083 |
| 5–6 | 69.0% | 0.000537 | 0.0078 |
| 6–7 | 66.5% | 0.000416 | 0.0068 |

Nearly every sample changes each frame by a small amount; the changed fraction
alternates high/low in frame pairs rather than decaying, so under stationary
inputs the history does **not** converge to a fixed point within 7 transitions.
The reset-mid runs' transition at the reset frame is the reset snap (offset 4:
1,465,993 changed / MAD 0.000688; offset 5: 1,456,437 / 0.000676) and their
dynamics then repeat the runs' first pairs exactly — further confirmation of
the restart. (MAD convention: per channel value over all values; the
changed-sample percentages agree with REFERENCE-CAPTURE.md's 66–69%.)

## Verification

- Probe: `zig cc ... -ld3d12 -ldxguid -ld3dcompiler_47` (strict `-Wall -Wextra
  -Werror`); runs via `vr/run-feature-probe.sh capture-reset <label> [frame]`,
  retained under `vr/logs/temporal/<label>`.
- `vr/temporal-state-report.py validate` — pinned checks over five capture runs
  and the boundary run: per-run manifests (digests, sizes, name sets, exactly
  one protocol line each) with **externally anchored digests** for run-a's 11
  files and the boundary NaN sentinel, pair identity, input identity, prefix,
  restart (per-run offsets 4 and 5), divergence, weights/counter static,
  carriers evolving. **PASS** (report:
  `vr/logs/temporal/temporal-state-report.txt`).
- `vr/temporal-state-report.py self-test` — 23 controls (baseline + 22 named
  corruptions, most *consistently* re-sealing manifests so only cross-run,
  relational or anchored checks can catch them, a word-format manifest
  truncation control, and a size-table guard asserting a capture-only **and** a
  boundary-only rejection so neither table's failures can mask the other). All
  reject with their pinned messages.

Evidence discipline: retained evidence and reports carry counts, digests and
derived aggregates only. The captures themselves (input/output tensors) remain
local under `vr/logs/`; nothing was pushed to GitHub.

## Limitations

- **The state's identity is not yet established**: the restart proves *that*
  reset removes every output-visible cross-frame influence, not *what* the
  state is. Aliasing `dlssnr_prev_output` into the window/binding graph — and
  whether it is the *only* carrier — is the next question.
- Stationary inputs, eye0 only, 720×720, 8 frames; motion/second-eye
  behaviour untested here, and state invisible under these inputs is not
  excluded. Frame-index periods ≥ 8 are not excluded either (offsets 4+4 and
  5+3 break periods 2–7).
- Boundary windows come from post-decoder-truncated runs (`DLSS_NVAPI_CUT`);
  the temporal dynamics of a truncated run are not claimed to equal an
  untruncated run's beyond the static-input relationships checked here.
- This stage is behavioural evidence from the vendor runtime; no native
  temporal computation ran, and no neural-rendered image was produced. Rights
  and data-handling caveats from earlier stages are unchanged.

## Next

The state's identity is now established in
[TEMPORAL-HISTORY.md](TEMPORAL-HISTORY.md) (gate stage 13): the model's
`dlssnr_prev_output` texture, written post-decoder with the frame's output,
read during evaluation, reset to zero by `DLSSNR.Reset`, plus a frame counter
(slot 1 word 25) that resets with it. Remaining:

1. Native one-step recurrence replay: feed a captured frame's inputs, the
   previous captured output as history and counter `N` to a native pipeline and
   compare against the vendor's next frame — the temporal analogue of the
   bit-exact decoder replay ([NATIVE-VENDOR-COMPARE.md](NATIVE-VENDOR-COMPARE.md)).
2. Pin the read/write kernels in the 156-slot sequence to complete the state
   map.
