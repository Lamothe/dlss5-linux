# Temporal-history carrier — gate stage 13 (14/10/2026)

The stage-12 work established *that* the runtime is a history blend with an
exact restart at `DLSSNR.Reset`, but not *what the state is*. This stage binds
the state to a concrete allocation, proves the read and write sides by
intervention, and completes the temporal contract **for the tested regime**
(eye0, 720×720, stationary gradient inputs, 8 frames):

**`output_N = M(inputs, history_N, counter_N)` where `history_N = output_{N-1}`
(bit-exact), `counter_N` = frames since the last reset, and `DLSSNR.Reset`
makes the model behave as if both were zero.**

Everything here ran on the RTX 4070 Ti SUPER under Proton/Wine via the existing
`vr/run-feature-probe.sh` harness (`capture-history*` modes); the native
side of the project is untouched by this stage.

## The carrier: `dlssnr_prev_output`, one per feature

The model's own NGX log names two 4.0 MB `CreateTexture2DResourceShared`
allocations (`dlssnr_prev_output`). The new resource-map dump
(`resource-map.txt`, recorded by the device hook with a creation-phase tag)
shows the model's phase-1 allocations as two parallel sets — `CG2RWeightHeap`,
`CG2RWeightUpload`, `CG2RPool`, history texture — one set per feature (eye),
with the two sets' buffers exactly `0x1CD00000` apart.

`capture-history` mode reads the model-owned phase-1 textures back after
creation (`stage=init`) and after every evaluation (`stage=frameNN`). Findings:

| Finding | Evidence |
|---|---|
| The history texture holds the frame's output | `history0-frameNN == output-frameNN` byte-exact, 8/8 frames in the 9 runs whose post-decoder stages ran (`hist-cut` is the withheld-write control) |
| The reset value is **all zeros** | `history0-init.bin` / `history1-init.bin` uniform `0x00` |
| One history per feature; the other is untouched | `history1-*` == init in every run (only feature 0 is evaluated) |
| The write is **post-decoder** | cut run (`DLSS_NVAPI_CUT`): history stays at init all 8 frames |

The cut run doubles as the in-process canary: its output keeps the untouched
FP16 NaN sentinel (`0x7e00`) in all 8 frames, so the truncated command stream
really never reached the output/history writes.

## The read side: intervention proves the model consumes the history

`capture-history-inject` overwrites the evaluated feature's history with a
synthetic pattern (uniform FP16 1.0 at frame 2; 16-pixel 1.0/0.0 checkerboard at
frame 5) before evaluation. `capture-history-sham` runs the same
upload→texture copy but stages the history's **own bytes**.

| Run | Result |
|---|---|
| `hist-inj` / `hist-inj-b` | outputs diverge from the reference at f2..f7 (pair bit-identical) |
| `hist-sham` | reproduces `hist-a` bit-for-bit — the copy machinery is not the cause |
| `hist-inj` prefix f0,f1 | equals the reference — only the intervention frames move |

So the model reads the history during evaluation, and the causal factor is its
**content**.

## Not the whole state: a frame counter

`capture-history-replay` stages frame 0's history (= `output-00`) and re-injects
it before frames 2 and 5. If the history were the sole state, `output-02` would
equal `output-01` (same history, stationary inputs). It does not
(`replay-counter-absent`): some frame-indexed state also matters.

The parameter dumps locate it. The `scan` analysis in
`vr/history-carrier-report.py` classifies every word position of the 120-eval
parameter dump (`vr/logs/capture/nvapi-trace-eval-args-1.log`) into an
exhaustive partition (`SCAN_PINS`, verified by `scan-mismatch` in `validate`):
1453 positions — 552 constant, 896 two-valued, and **5 multi-valued**, each
multi-valued position pinned as (position, distinct count, FNV of its full
120-eval value sequence). Of the multi-valued positions only **slot 1,
word 25** varies *per frame*: the frame counter (60 values across the 120
evals — the sequence is digest-frozen, and this reading of it is cross-checked
by the 8-eval counter check below). The other four — slot 1 words 1–2 and the
identical slot 154 words 11–12 (the duplication is machine-checked as
`multi_shared`) — are one lazily initialised parameter pair **shared verbatim
between the two launches**, read from the frozen sequences as: zero at evals
0–1, then per-eye constants from eval 2 (the two eyes' values differ by 2).
A traced `capture-reset` run shows the counter sequence
**`[0,1,2,3,0,1,2,3]`** across a mid-stream reset at frame 4: the counter is
the frame index since the last `DLSSNR.Reset`.

`capture-history-replay2` perturbs frame 1 (uniform 1.0) and then re-injects the
same `output-00` history at f2/f5. **All** later outputs (f2..f7) match
`hist-replay` exactly despite the f1 perturbation (`replay2-trace`) — no
content-dependent state beyond the history survives across frames.
(Frame-indexed *parameter* state is excluded by the exhaustive slot scan above,
not by replay2 alone; state carried in buffer contents — the history texture
itself — is what the carrier experiments cover.)

## Reset frames ignore the history

`capture-history-injectreset` injects uniform 1.0 immediately **before** a
`DLSSNR.Reset` frame (4). The garbage has no effect: `output-04 == output-00`
(zero-history result) and frames 5–7 replay 1–3. Reset therefore behaves as if
the history were zero at its frame — the evidence cannot distinguish "reset
zeroes the texture" from "the model ignores the history when its counter is
0", and the contract is stated in the behavioural form. Together with the
counter restart:

```
output_N = M(inputs, history_N, counter_N)   (tested regime: eye0, 720x720,
history_N = output_{N-1}                     stationary inputs, 8 frames)
counter_N = N                                (slot 1 word 25, since last reset)
DLSSNR.Reset -> behave as if history = 0, counter = 0
```

## The stage-9 "two memory halves" are the two features

The earlier parameter analysis found 881 two-valued word positions (values
`0x1CD00000` apart) in an `ABBA` pattern and hypothesised a rotating working
set. The resource map now shows the two features' parallel allocations at
exactly that delta. The `scan` analysis (pins verified by `scan-mismatch`)
counts **896 two-valued word positions** in the 120-eval parameter dump:
**881 are address-shaped** (matching NVAPI-BINDINGS.md's two-addr count), of
which all track the evaluation's eye exactly; of the 15 non-address two-valued
positions, 5 also track the eye and **10 are one-time lazy scalars** that switch
at eval 2 and stay. So 886/896 track the eye order, and the `ABBA` pattern is
the alternating submission order of the two eyes — not a rotation within one
context. Each feature holds a static resource set.

The exhaustive classifier also corrects one earlier claim: the first two-bucket
scan reported "exactly one varying word position" while silently dropping
within-eye toggles with disjoint value pairs. The four lazily initialised words
above were found that way (the review that caught the blind spot is recorded in
the stage notes); the self-test now plants that exact shape
(`write_scan_fixture`) so the class scheme cannot regress. The parameter pair
shared between slot 1 and slot 154 is recorded as a lead for pinning the
history read/write kernels (the "Next" item), not yet interpreted.

## Validation

`vr/history-carrier-report.py`:

- `validate` walks 10 runs (`hist-a/b`, `hist-r4`, `hist-cut`, `hist-inj/-b`,
  `hist-sham`, `hist-replay`, `hist-replay2`, `hist-injreset`), recomputes every
  FNV-1a-64 digest from the bytes on disk, and checks 28 pinned relationships:
  protocol lines, intervention markers, write identity, zero init, static
  control, prefix/restart/no-divergence at the mid-reset, cut-run invariants,
  sham identity, intervention effectiveness, the counter sequence, and the
  reset-frame replay. **PASS (10 runs, 28 checks).**
- `self-test` builds a 4×4 synthetic fixture with the same relationships and
  asserts **35 corruption controls** each fail with their pinned check (one is
  deliberately non-isolated: `no-restart-divergence` also trips the anchors and
  the sham check by construction). The scan machinery, both external anchors,
  the anchor walk's missing-file path and the cross-run input check each have
  dedicated controls (including all three scan error paths), a coverage
  assertion requires every check name to fire somewhere, and the scan fixture
  plants a hand-written ground-truth classification (`SCAN_FIXTURE_PINS`) that
  includes the blind-spot toggle shape and a verbatim shared pair — the pins
  for the real trace are tool-generated, so this synthetic ground truth is
  what validates the classifier itself. **PASS.**
- hist-a's digests are externally anchored in the tool (below) — all payloads
  *and* its manifest and resource map — so a coordinated change to both files
  and manifest is still detectable.
- Provenance: the 10 runs were captured with the final probe build
  (`zig cc … -Werror`, PE sha256 prefix `609233eaf09f1988`). The earlier
  pre-rebuild runs are retained under `vr/logs/history-superseded-pre-rebuild/`
  and are bit-identical (290/290 `.bin` payloads, all manifests and resource
  maps), so the final-source runs independently reproduce the pinned anchors;
  the `hist-cut` NvAPI trace likewise reproduces structurally (448 cut lines,
  identical kernel names/handles/counts; only run-time pointer values differ)
  and is retained as `nvapi-trace-hist-cut-superseded-pre-rebuild.log`.
- The self-test's `reseal` helper previously rewrote only the last two manifest
  fields, which never matches history lines (`file=` sits between `bytes=` and
  `fnv=`): stale digests leaked past the helper and 17 controls failed
  isolation. It now rewrites the full tail and fails loudly on an unexpected
  line shape; the `bad-digest` control corrupts a manifest digest entry (not
  file bytes, which would also trip the cross-run input check).

Anchored hist-a digest highlights (full table in the tool):
`output-frame00.bin = 0x839de3f30038880a`, `history0-init.bin = 0x0b709f104affc325`,
`output-frame07.bin = 0xbbba397b954e15eb`; `history1-*` all `0x0b709f104affc325`.

## Limitations

- Capture runs evaluate feature 0 (eye0) only, 720×720, stationary gradient
  inputs, 8 frames; the second feature's history is observed only as a static
  control. Motion/second-eye behaviour is untested here.
- The counter is identified by its per-frame variation and reset behaviour
  (slot 1 word 25); its exact semantics inside the model (jitter index, blend
  schedule, …) are not decoded — only that it is a frame index since reset.
- The write is localised post-decoder only (cut run); the specific kernel is
  not pinned (candidates: `cg2r_post_process_kernel`/`cg2r_copy_kernel`, slots
  154–155). The read point in the 156-slot sequence is likewise not pinned.
- Intervention runs overwrite the history *texture*; if the model caches the
  history elsewhere at creation, the cache is not exercised differently here.
- The `dlssnr_prev_output` identification rests on reading the resource map's
  records; the map bytes are now anchored, but the name-to-allocation link and
  the manifests' history metadata (`idx`/`rec`/`stage`/`w`/`h`/`fmt`, all
  `va=0x0`) are shape-checked only, not cross-pinned to the map.
- The f5 checkerboard's effect is not independently isolated: the required
  divergence at f2..f7 is fully explained by the f2 uniform injection, so a
  no-op f5 write would pass (an f2-only run would isolate it). Likewise the
  injectreset write-landing has no in-run positive control (it relies on
  command-list ordering before `evaluate()` and on the shared injection path
  proven effective in `hist-inj`/`hist-replay2`), and `replay-counter-absent`
  is meaningful only jointly with `replay2-trace`.
- The readback copies are driver-lenient D3D12: `CopyTextureRegion` with a UAV
  barrier only, no `COPY_SOURCE`/`COPY_DEST` state transitions (noted in the
  code). The bit-exact identities and intervention effects are the empirical
  validation.
- The real-trace scan pins are generated by the validated classifier (the
  multi-valued positions additionally pin full-sequence digests, so their
  readings are machine-checkable against the anchored dump); the classifier
  itself is checked against hand-written synthetic ground truth and corruption
  controls, which is the limit of the anti-circularity argument.
- This stage is behavioural evidence from the vendor runtime; no native
  temporal computation ran and no neural-rendered image was produced. Rights
  and data-handling caveats from earlier stages are unchanged; captures remain
  local under `vr/logs/`.

## Next

1. Native one-step recurrence replay: feed a captured frame's inputs,
   `output_{N-1}` as history and counter `N` to a native pipeline and compare
   against the vendor's `output_N` — the temporal analogue of the bit-exact
   decoder replay (NATIVE-VENDOR-COMPARE.md). The decoder stage alone cannot
   do this: the history enters *before* the decoder, so this needs the
   pre-decoder stages' native implementations.
2. Pin the read/write kernels (e.g. by a cut at finer granularity or
   per-kernel injection) to complete the 156-slot state map.
