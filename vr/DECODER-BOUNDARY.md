# Decoder-boundary intermediate capture — 13/10/2026

## Result

**The model's decoder boundary is now captured: for each evaluation, the exact
input, output, counter, partial-sum and weight buffers crossing the decoder
launch, read back from the live runtime on the RTX 4070 Ti SUPER under
Proton/Wine.** This is the stage-2 deliverable of the phase-3 plan
([NVAPI-BINDINGS.md](NVAPI-BINDINGS.md) next steps): the tensors the
native-vs-vendor comparison needs, in the format the native decoder work can
consume directly. It is **not** a neural-rendered image — the capture runs
deliberately truncated before the image-producing stages.

Two findings are milestones in their own right:

1. **The stored decoder-weight record is confirmed by live capture.** The 2 MiB
   window at the decoder's weight pointer (word 7) begins with the
   `block39.layer0.layer` payload **byte-for-byte** (525,312 bytes; checked by
   `weights-record-match`). Until now the record was a *bound candidate* on
   numerical compatibility ([NATIVE-DECODER-BINDING.md](NATIVE-DECODER-BINDING.md));
   the live runtime now agrees that this is the region the decoder reads.
2. **The counter region reads the recovered protocol's final value.** All eight
   counter slots hold `3` after the decoder runs — the exact publication value
   recovered by disassembly and exercised natively
   ([NATIVE-DECODER.md](NATIVE-DECODER.md)). A stale readback would show the
   `-1` reset values, so this doubles as the synchronisation canary.

## Method: truncate the evaluation at the decoder

Mid-stream GPU buffers cannot be read at "CPU call time": the launches are
recorded, not executed, when the harness sees them. The capture therefore
**removes everything after the decoder from the command stream** and reads the
boundary at the end of the frame:

1. **The forwarder cuts** (`DLSS_NVAPI_CUT=1` in
   `vr/capture/nvapi-trace.c`): a name-anchored state machine forwards
   launches up to and including
   `cc_dec_input_upsample_1024_512_tilesync_fp8`, withholds every subsequent
   launch (the stub returns `NVAPI_OK` without calling the real entry point),
   and closes the window at the next `cc_cb_clear` — or at a bounded budget
   of 56 withheld launches, so no failure can truncate beyond one window.
   Each evaluation therefore executes exactly slots 0–99 of the recovered
   156-launch sequence ([NVAPI-ARGS.md](NVAPI-ARGS.md)), so nothing
   overwrites the decoder's buffers. Each armed decoder launch appends one
   `boundary eval=.. psize=.. pmem=` line to `DLSS_NVAPI_CUT_FILE` with the
   80-byte parameter block. Fail-closed at source: an unknown handle can
   never *trigger* a cut; arming additionally requires a clear-kernel name to
   be currently mapped and the sidecar line written in full (otherwise
   `cut-clear-unknown` / `cut-sidecar-write-failed`, and the run stays
   uncut); the window withholds at most the recovered 56 post-decoder
   launches (`cut-window-exhausted` closes it and forwards the current
   launch); and a decoder-named launch inside an open window is withheld and
   flagged (`cut-anomaly`), never re-armed. An incomplete parameter dump
   never arms the cut (`cut-pmem-incomplete`).
2. **The harness maps addresses** (`vr/boundary-probe.h`): the device vtable
   is wrapped (`CreateCommittedResource`/`CreatePlacedResource`) so every
   resource the model creates through the probe's device is recorded with its
   GPU virtual address and size. The model allocates through that device: all
   six boundary words resolved.
3. **The harness copies windows** (`feature-probe.exe --capture-boundary`):
   per frame, after one `evaluate` call produced exactly one new sidecar line,
   words 0/1/2/4/6/7 are resolved through the map and a window of up to 2 MiB
   is copied into readback buffers after a UAV barrier, then dumped with
   geometry and digests. The output texture is checked to still hold the
   uploaded NaN sentinel — with stages 100–155 absent, nothing can have
   written it.

## Findings

### The cut is exact

From `nvapi-trace-boundary-a.log`: **3 `cut-armed`, 168 `cut-drop`
(= 56 × 3 evaluations), 2 `cut-release`** — the drop count per evaluation
matches the 156-launch inventory exactly (slots 100–155), and the trace
contains none of the failure-mode records (`cut-anomaly`, `cut-unknown`,
`cut-window-exhausted`, `cut-clear-unknown`, `cut-pmem-incomplete`,
`cut-sidecar-write-failed`, `cut-name-table-full`), which the report now
rejects. All three frames report `untouched=1` (518,400 NaN pixels, 0
other).

### The allocator picture from stage 9 is now visible as bytes

All boundary words resolve into two resources, whose sizes match the model's
own NGX allocation logs exactly:

| Resource | Size | Words |
|---|---:|---|
| working pool | 62,684,160 (59.8 MiB) | 0 main-input, 1 skip-input, 2 out0, 4 counter, 6 partial-sums |
| weight heap | 147,719,680 (140.9 MiB) | 7 weights |

Frame 0 geometry (later frames identical VAs in these runs): main-input at
pool+62,487,552 (window clamped to 196,608 bytes by the resource end — so the
main-input extent is at most that), skip-input at +52,313,088, out0 at
+20,610,048, counter at +97,280, partial-sums at +35,994,624, weights at
heap+123,736,576.

### The live constants match the recovered host ABI

The parameter block's packed pairs are `dimension=12` (offset 64),
**`width=16` (offset 68)** and **output-axis overrides `(24, 28)`** (offsets
72/76) — the latter two matching `NVAPI-BINDINGS.md`'s live binding analysis
and `NATIVE-DECODER-IO.md`'s constant-bank census slots. `dimension=12` is the
runtime's live input-axis value (the native probes chose their own guarded
test values, e.g. `INPUT_AXES=(8,8)`).

### The boundary is temporal

With stationary inputs and `DLSSNR.Reset` only on frame 0, the boundary
buffers still change substantially between frames (changed bytes of the
captured windows — 2 MiB except the clamped 196,608-byte main-input):
main-input ≈ 183–184k, skip-input ≈ 1.78M, out0 ≈ 1.89M. A temporal
history-dependent value therefore enters *before* the decoder, not only at the
final blend — a concrete constraint for temporal reconstruction. The evidence
establishes the behavioural role of a history texture, but does not yet prove
that it is the allocation named `dlssnr_prev_output` in
[REFERENCE-CAPTURE.md](REFERENCE-CAPTURE.md); the resource map does not link
that name to its record. The per-window carrier classification and the reset
contract are documented in [TEMPORAL-STATE.md](TEMPORAL-STATE.md).

### Determinism holds at the boundary

Two captures (`a`, `b`) are **bit-identical**: all 18 windows, the parameter
blocks and every input/output file (`runs-identical`), consistent with the
11-tensor determinism established for the reference capture.

## Evidence

Runs on 13/10/2026 (final source; the self-test's fixture bug and the report's
`repin`/pair-comparison defects were fixed before these runs):

- `vr/logs/boundary/a` and `vr/logs/boundary/b` — captures (manifest + 36
  window files + inputs/outputs + `boundary-va.txt` sidecar).
- `vr/logs/capture/nvapi-trace-boundary-{a,b}.log` — forwarder traces.
- `vr/logs/reference/regress1` — reference-capture regression after the
  harness changes; `nvapi-trace-regress-create.log` — plain-forwarder
  regression.
- `python3 vr/decoder-boundary-report.py self-test` — 19 corruption controls,
  each rejected with its pinned message (`manifest-protocol`,
  `sidecar-records`, `manifest-consistent`, `words-complete`,
  `boundary-constants`, `counter-published`, `output-untouched`, `cut-drops`,
  `runs-identical`), plus 2 positive record-comparator asserts.
- `report vr/logs/boundary/a --trace … --record-dll vr/probe/nvngx_dlssnr.dll`
  passes all pinned checks including `weights-record-match` (record payload
  sha256 `e7fa32bd…c4fb8`, extracted through the hash-pinned DLL of
  [NATIVE-DECODER-BINDING.md](NATIVE-DECODER-BINDING.md)); `pair a b` passes.

Evidence contains digests, counts and geometry only — never tensor or weight
bytes. The retained `.bin` windows are local (gitignored) and carry the
proprietary data-handling caveats of the earlier stages: do not publish
proprietary implementation material without resolving rights.

## Limitations

- Windows are **2 MiB bounds, not tensor extents**. Where the window clamps
  (main-input: 196,608 bytes) the extent is bounded above; elsewhere the
  tensor extent within the window is not yet derived.
- The truncated run produces no image, and stages 100–155 are absent by
  design: this validates capture mechanics, not the full pipeline.
- The device hook wraps the base `ID3D12Device` vtable only; empirically
  sufficient (all six words resolved), but an allocation through an extended
  device interface would appear as unmapped and fail the run closed.
- The drop path returns `NVAPI_OK`; the model's evaluate calls all returned
  `0x1` with the cut active, but the model's internal handling of the
  unresolved later counter waits is unverified beyond that.
- `weights-record-match` proves the decoder reads this record's bytes from
  this region; the record's exact position in the Windows graph and the scale
  tail's semantics (`NATIVE-DECODER-BINDING.md`) remain as documented there.

## Reproduction

Read `README.md` first; from the repository root:

```bash
python3 vr/decoder-boundary-report.py self-test
bash vr/run-feature-probe.sh boundary a
bash vr/run-feature-probe.sh boundary b
python3 vr/decoder-boundary-report.py report vr/logs/boundary/a \
  --trace vr/logs/capture/nvapi-trace-boundary-a.log \
  --record-dll vr/probe/nvngx_dlssnr.dll
python3 vr/decoder-boundary-report.py pair vr/logs/boundary/a vr/logs/boundary/b
```

Builds (see `EVALUATION.md`/`NVAPI-DISPATCH.md`):

```bash
tools/zig-x86_64-linux-0.15.2/zig cc -target x86_64-windows-gnu -std=c11 -O2 -shared \
  -Wall -Wextra vr/capture/nvapi-trace.c -o vr/capture/nvapi64-trace.dll
tools/zig-x86_64-linux-0.15.2/zig cc -target x86_64-windows-gnu -std=c11 -Wall -Wextra \
  -Werror -O2 vr/feature-probe.c -ld3d12 -ldxguid -ld3dcompiler_47 -o vr/probe/feature-probe.exe
```

## Next

**Completed 14/10/2026:** the stage-3 comparison ran and matches byte-for-byte
— see [NATIVE-VENDOR-COMPARE.md](NATIVE-VENDOR-COMPARE.md). (Original plan
text follows.) The stage-3 comparison is now unblocked and fully specified: run the native
decoder (`native-decoder-launch-probe.py` machinery, real weights now
confirmed) with **captured words 0/1 as inputs and the captured weights**, and
compare against **captured word 2**. A bit-exact or tolerance-bounded match
validates the native reconstruction numerically against the vendor runtime —
the first true native-vs-vendor result. The frame-evolution finding above also
tells that stage which buffers must be held fixed (feed frame N's words 0/1).
