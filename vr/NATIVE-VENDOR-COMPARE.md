# Native-vs-vendor decoder comparison — 14/10/2026

## Result

**The native replay reproduces the vendor's captured decoder output
byte-for-byte.** For each of three captured evaluations, the decoder launch was
replayed through the **native Linux CUDA driver** on the RTX 4070 Ti SUPER —
no Windows code, no DirectX, no Proton — from the decoder-boundary windows of
[DECODER-BOUNDARY.md](DECODER-BOUNDARY.md), and the native out0 matched the
vendor's captured out0 exactly: **12/12 comparison entries, 0 differing bytes
of 344,064 per entry** (both kernel variants × three frames × both partials
pre-states). This is the stage-3 deliverable of the phase-3 plan: the first
numerical validation of the reconstruction against the vendor runtime itself,
rather than against structural predictions.

Two tensor channels validate, not one:

| Channel | Scope | Result |
|---|---|---|
| out0 (word 2) | `[0, 344064)` = 24×28×512 | **bit-exact** in 12/12 entries (enforced) |
| partial sums (word 6) | `[0, 196608)` write extent | **bit-exact from a zero pre-state** in 6/6 such entries; **unchanged** from the captured pre-state in 6/6 |

The partials channel is what addresses the pre-state question. The decoder's
partial-sums write-extent content is **identical across the two tested
pre-states** (zeros, and the captured window): from a *zero* pre-state the
kernel writes `[0, 196608)` with content identical to the vendor's captured
window, and from the captured window it leaves the window byte-identical. No
writes occur beyond the retained 2 MiB window (`beyond_window_change` false in
all entries). Unconditional rewrite versus conditional (pre-state-dependent)
writes is *not* distinguished by two sample states; what is established is
that **out0 matched the vendor under both tested pre-states**, so the match
does not require the unrecovered pre-decoder state.

## Method: replay the captured boundary

`native-vendor-compare-probe.py` loads a frame's six captured windows
(digest-verified against the capture manifest) and rebuilds the launch:

- **main input (w0)**, **skip input (w1)**, **weights (w7)**: uploaded
  byte-identically. The weights window begins with the bound
  `block39.layer0.layer` record ([NATIVE-DECODER-BINDING.md](NATIVE-DECODER-BINDING.md));
  the captured window covers whatever the kernel reads.
- **counters (w4)**: reset through the recovered `cc_cb_clear` contract at the
  vendor's own live count — **40,704 slots, grid (159,1,1)**, taken from the
  level-2 launch capture (slot 0's parameter block points at the pool base).
  This is `reset_geometry(40704)` of the recovered contract
  ([NATIVE-DECODER.md](NATIVE-DECODER.md)) at the vendor's own scale.
- **scalars**: the captured constant-bank bytes copied verbatim:
  `dimension=12, width=16, output_axis0=24, output_axis1=28` — the same
  `boundary-constants` values pinned by
  [DECODER-BOUNDARY.md](DECODER-BOUNDARY.md).
- **argument block**: the recovered 80-byte `DecoderArguments`
  ([NATIVE-DECODER-LAUNCH.md](NATIVE-DECODER-LAUNCH.md)) with our own buffer
  addresses; grid **(8,3,4)** = `launch_geometry(12, 16)`, cross-pinned to the
  level-2 live capture (slot 99 `grid=8,3,4`), block (32,2,1).
- **partials (w6)**: replayed at two pre-states (`pcap` = captured window,
  `pzero` = zeros) to test sensitivity. **out0 (w2)**: pattern-filled (0xA5) —
  its pre-decoder contents are not captured, and
  [NATIVE-DECODER-BINDING.md](NATIVE-DECODER-BINDING.md) established pure
  stores for the recovered variant.

Both decoder kernels in the SM89 module are replayed: the **vendor's own**
`cc_dec_input_upsample_1024_512_tilesync_fp8` (the slot-99 launch of
[NVAPI-ARGS.md](NVAPI-ARGS.md)) and the recovered/bound sibling
`cc_dec_input_upsample_1024_512_fp8` (the kernel of the permutation and
binding work). **Both match the vendor output byte-for-byte** — the variants
are numerically equivalent on these inputs.

## Findings (recorded; no pass/fail semantics)

1. **The recovered output-dimension rule holds at the live geometry — enforced
   for both variants.** Both variants write **exactly** `[0, 344063]` =
   `output_axis0 × output_axis1 × 512` bytes, the rule recovered from the 8×8
   synthetic test ([NATIVE-DECODER-LAUNCH.md](NATIVE-DECODER-LAUNCH.md)) now
   confirmed against the vendor's own 24×28 run; the run fails if any entry's
   extent differs or writes outside the scope.
2. **The live variant writes the out2 slot; the recovered variant does not.**
   `tilesync_fp8` writes exactly **96 bytes** (`[0, 95]` = 24×4) to out2 in
   every entry; `..._fp8` leaves out2 untouched, consistent with the
   [NATIVE-DECODER-IO.md](NATIVE-DECODER-IO.md) census. out3 is never written
   by either. The census claim is therefore variant-scoped, and the live
   variant publishes a 96-byte status/counter block the recovery had marked
   unused.
3. **The partial-sums write extent is 192 KiB** (`[0, 196607]`) = 12 × 16,384
   bytes, matching the
   [NATIVE-DECODER-LAYOUT.md](NATIVE-DECODER-LAYOUT.md) 16,384-byte tile
   derivation (12 tiles for this geometry).
4. **Counter publication matches the capture exactly**: 24 slots of `3`
   (matching `2 × rows × columns` = 2×(12/4)×(16/4)), with the window
   `[0, 512)` — the decoder-private, clear-preserved region — identical to the
   captured w4 window.

## Verified

Final-source runs on 14/10/2026 (RTX 4070 Ti SUPER, native CUDA):

- `vendor-compare-execute-HxkABdYb` / `-CuOIbN8H` — 12 entries each (3 frames ×
  2 kernels × 2 partials pre-states); 24 kernel launches, 96 readbacks per run;
  `out0_matches: 12`, `partials_scope_matches: 12`. **Pair-identical**: all 12
  entry results and all **36 retained evidence windows** bit-identical
  (oracle `pair`).
- `vendor-compare-skip-RloyywW0` — reset only: no decoder launch (1 kernel),
  counters `[−1]×24` in every slot (non-publication enforced exactly), out0
  untouched, all inputs immutable; correctly *not* matching the vendor window.
- `vendor-compare-wrong-wkPEM5dL` — matrix rotated by 1,024 bytes
  (the [NATIVE-DECODER-BINDING.md](NATIVE-DECODER-BINDING.md) precedent):
  output **rejected** (338,493 of 344,064 bytes differ), extent still exactly
  `[0, 344063]`, counters still `[3]×24`. The rotated output digest
  (`0xd6cdbc4b7837de0a`) is identical across all four wrong-mode runs, so the
  corrupted computation is deterministic and the comparison discriminates.

Integrity (enforced, run fails otherwise): guards, input immutability
(main/skip/weights everywhere; out3/out2 on the recovered variant per its
census), counter publication `[3]×24`, counter window `[0,512)`, out0 extent
**exactly** `[0,344064)` on every launched entry, out0 match on **every**
launched entry (execute), and non-match/non-publication for skip and non-match
for wrong.

Tooling:

- `python3 vr/native-vendor-compare-probe.py self-test` — 3 geometry known
  answers (incl. the live grid and clear geometry), 3 diff + 3 extent known
  answers, 4 malformed-input controls, 6 verdict-consistency controls, 4
  capture-loader controls (digest, size, scalars, content), each rejected.
- `python3 vr/native-vendor-compare-oracle.py self-test` — 4 known answers, 4
  malformed-input controls and **35 corruption controls**, each rejected with
  its pinned message: retained content/size/digest and recorded fields for
  every channel (out0, partials incl. extent/scope/schema, counters incl.
  head), entry plan/order/kernel/frame/prestate/rotation/launch arithmetic,
  integrity flags (guards, immutable, out3/out2 consistency),
  enforced-failures pinning, and capture digests/key set/scalars corrupted in
  exactly one frame (discriminating the per-frame scalar loop). The oracle
  recomputes every verdict derivable from the retained windows from raw bytes
  with independent arithmetic (big-integer XOR, deliberately a different
  algorithm from the probe's blockwise comparison) and pins the
  probe-enforced flag claims for internal consistency; guard zones and bytes
  beyond the retained windows remain probe-enforced, fail-closed, outside the
  oracle's recomputation scope. It verifies all four runs
  (`verified: true` each).

Development evidence retained: `vendor-compare-wrong-lNkqivuV` (exit 1,
`KeyError: 'decoder'` — the function-key renaming bug caught by the wrong-mode
smoke run; retained windows absent because the failure predated the
retain-first fix) and `vendor-compare-{execute,skip,wrong}-*` runs superseded
by review-driven schema, enforcement and scope changes (three generations:
`EWV9THrE`/`IBB6n8kP`/`sQm60C7j`, `e4zWpCmv`/`7P91RTE4`/`6FbetCW0`/`gQmWVwoX`,
`Ze3D1orY`/`NzaQgHCi`/`EBxNZJIT`/`EnkE9AQy`). Out0 results were identical
throughout every generation.

## What this does and does not establish

Established: the recovered decoder contract — argument block, launch geometry,
counter protocol, weight region, input/output addressing — is **numerically
correct for the decoder stage**, reproducible natively on Linux from captured
buffers, and equivalent across both kernel variants on these inputs. The
vendor's learned weights execute correctly through the recovered interfaces.

Not established: anything beyond this stage. Slots 0–98 (encoder, transformer,
ViT) and 100–155 are not replayed; temporal history-dependent information is
present in the boundary captured inputs before the decoder. The temporal
history-carrier behaviour is not linked by the resource map to the allocation
labelled `dlssnr_prev_output` in the runtime log; that allocation identity
remains an inference (see [TEMPORAL-HISTORY.md](TEMPORAL-HISTORY.md)). **No
complete native pipeline and no neural-rendered image** — and this remains an
execution of NVIDIA's proprietary kernel and weights: the result validates our
interfaces, not our freedom to use them. Reuse/publication rights remain
unresolved. Read-only out-of-bounds access is undetectable by the guard
discipline; skip-window contents beyond 2 MiB are unreproduced (no observable
effect here).

## Reproduction

Read `README.md` first; from the repository root:

```bash
python3 vr/native-vendor-compare-probe.py self-test
bash vr/run-native-vendor-compare-probe.sh execute a   # ×2
bash vr/run-native-vendor-compare-probe.sh skip a
bash vr/run-native-vendor-compare-probe.sh wrong a
python3 vr/native-vendor-compare-oracle.py self-test
python3 vr/native-vendor-compare-oracle.py verify research/native-linux/runs/<run> vr/logs/boundary/a
python3 vr/native-vendor-compare-oracle.py pair <run-a> <run-b>
```

The launcher uses the existing common lock, home/network isolation, private
tmp, pinned source snapshots and a 300-second timeout; the capture directory is
bound read-only and the DLL is read as data. NVIDIA device nodes are exposed
only to this sandbox. Core dumps are disabled; swap, driver persistence and
memory erasure remain uncontrolled. Evidence retains digests, counts and the
bounded windows listed above (out0 `[0,344064)`, partials `[0,2 MiB)`,
counters `[0,512)` per entry) locally under
`research/native-linux/runs/`; they carry the proprietary data-handling
caveats of the earlier stages and are not committed.

## Next

Phase-3 stage 3 is complete. The remaining stages of the plan:

1. **Temporal/history reconstruction** — the pivot point. The model is
   `CC_Control_History_Blend_Quantize_With_Teacher...`
   ([REFERENCE-CAPTURE.md](REFERENCE-CAPTURE.md)); a history-dependent value
   enters before this stage. The behavioural history-carrier evidence does not
   link the carrier to a resource-map record corresponding to the allocation
   labelled `dlssnr_prev_output` in the runtime log, so that allocation identity
   remains an inference. The boundary capture's
   frame-evolution data and the now-validated decoder give concrete
   constraints, but the history feedback loop is the open question that decides
   reconstruction versus an open neural backend.
2. **Native VR integration into the existing Genesis project** —
   `/home/michael/Projects/Genesis` is already the Linux OpenXR application
   (native Vulkan 1.3, WiVRn + Quest 3) this plan once anticipated building.
   Do not build a second VR host. Bind the validated stages to its
   `neural-seam.h` per-eye evaluator contexts (`NEURAL-SEAM.md` conventions)
   once stage 1 settles the pipeline shape.

Either path ends in native Linux neural rendering; the difference is whether
the pipeline runs NVIDIA's weights or a model we own.
