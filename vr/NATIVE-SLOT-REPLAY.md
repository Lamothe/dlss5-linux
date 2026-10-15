# Native pre-decoder kernel replay (slots 24-26) — 15/10/2026

## Result

**Three pre-decoder kernels of the vendor's 156-launch sequence now replay
natively on Linux and reproduce the vendor's own captured state
byte-for-byte.** Each kernel's launch is rebuilt from its captured parameter
block and executed through the **native Linux CUDA driver** on the RTX 4070
Ti SUPER — no Windows code, DirectX or Proton — and the native pool state
after the launch equals the vendor's ladder capture at the matching cut point
([LADDER.md](LADDER.md)) over the **whole 62,684,160-byte pool: 0 differing
bytes on all three frames, in both execute runs, for all three slots**.

| Slot | Kernel (NVAPI-ARGS.md) | Contract | Cut | Grid |
|---|---|---|---|---|
| 24 | `cc_split_swin_16h_ffwd_inpview_512_tilesync_fp8` | reads w0; writes output w1 (344,064 B) + sync w6 (96 B) | s23→s24 | (4,3,2) |
| 25 | `cc_split_swin_16h_ffwd_proj_inpview_512_chained_fp8` | reads w0 + residual w1 + previous sync w7; writes output w2 (344,064 B) + sync w6 (96 B) | s24→s25 | (8,3,1) |
| 26 | `cc_split_swin_16h_qkv_512_chained_fp8` | reads output-of-slot-25 w0 + previous sync w5; writes output w1 (344,064 B) + sync w6 (192 B) | s25→s26 | (4,3,4) |

Slot 25 is the `ffwd→proj` pair's second half and the first kernel with the
**chained sync handoff**: it consumes slot 24's sync record (w7 at pool
offset 48,640) and writes its own at w6 (49,152) — which the attribution
shows is exactly slot 26's w5 input ([names-s24-s25.txt](logs/ladder/names-s24-s25.txt)).
Its w1 is slot 24's w0 buffer (the residual input; the two kernels swap the
same two 344,064-byte regions, the `inpview` pattern), and its w2 feeds
slot 26's qkv w0.

Slot 26 is the transformer block's qkv half. It consumes slot 25's sync
record (w5 at 49,152) and writes its own at w6 (49,664) — slot 27's w7
input — and its output at w1 (21,298,176) is slot 27's w0. The sync write's
**changed extent is 192 bytes** (not 96), stable across all three frames in
the independent write-set attribution ([names-s25-s26.txt](logs/ladder/names-s25-s26.txt));
the replay's sibling divergence matches that measurement exactly (below).
The kernel handle is shared with slot 30 at a **different geometry** (grid
4,3,4 here, 4,4,4 there — and the same cubin serves both), so the grid is
resolved at the **slot's own launch position** and pinned across evaluations
and runs; per-handle geometry is not unique in this inventory.

Everything each launch needs was resolved from retained evidence — no new
capture semantics were invented:

| Element | Source |
|---|---|
| grid/block/smem | forwarder trace created/launch records, pinned equal across runs and evals |
| parameter block | trace slot dump (56 B / 72 B), pinned equal across evals, runs and the capture sidecar |
| input pool state | the capture after slot n−1 (exactly what slot n reads) |
| reference pool state | the capture after slot n |
| weight bytes | `block23.layer0.layer` (524,288 B) / `block23.layer1.layer` (263,168 B) / `block23.layer2.layer` (917,568 B) from the DLL's WEIGHTS_HT resource |

The slot-25 word map (72-byte block, 9 words): w0 output-of-slot-24, w1
residual input, w2 output, w3 weights, w4 the recovered `(24, 28)` geometry
overrides constant, w5 reserved 0, w6 sync out, w7 sync in, w8 reserved 0.
The slot-26 word map (56-byte block, 7 words): w0 output-of-slot-25, w1
output, w2 weights, w3 the `(24, 28)` constant, w4 reserved 0, w5 sync in
(slot 25's record at 49,152), w6 sync out (49,664 — the same 512-byte
stride).

## The weight-heap layout

The 140.9 MiB weight heap is the **WEIGHTS_HT record payloads concatenated at
512-byte alignment**. The layout is pinned by the decoder's live weights
(word 7 of the decoder boundary, [DECODER-BOUNDARY.md](DECODER-BOUNDARY.md))
at `block39.layer0.layer`'s payload, and slot 24's word 2, slot 25's word 3
and slot 26's word 2 all land **exactly at their record payload starts**
(`block23.layer0.layer` at heap offset 6,733,312; `block23.layer1.layer` at
7,257,600 — directly after layer0's 524,288-byte payload; `block23.layer2.layer`
at 7,520,768 — directly after layer1). layer2's 917,568-byte payload is not
512-aligned: the following record starts 448 bytes later at the alignment
boundary (arithmetic from the probe's live DLL walk, which enforces exactly
one record at that boundary), so the tail control's following-record lookup
is alignment-aware. Beyond the anchors, the probe
computes and enforces the structural claim itself (`setup.heap_scan`, retained
in every run's evidence): every heap-pointing parameter word of every scanned
trace lands at a record payload start — **459 of 459 in each scan record,
zero mid-record** (the scans cover the s23, s24, s25 and s26 traces; s24 and
s25 are each scanned by two contracts' setups).

## Variant finding: the tilesync/chained sync record

All three contracts' fp8 siblings (`..._512_fp8`) are **created but never
launched by the vendor** (pinned: 0 launches in every trace), so they have no
launch contract of their own; each is replayed at the live variant's geometry
as a variant probe. In every case the sibling's **output tensor matches the
reference byte-for-byte** but its **sync record is left at the input state**
(`sync_from_input` true on every entry) while the live variant (tilesync at
slot 24, chained at slots 25 and 26) writes it. That divergence confinement
is **enforced** (`sibling-divergence`): the differences inside the sync
record must equal the total difference count over the whole pool — which
proves every differing byte lies within the record — and the record itself
must differ in **every byte of its changed extent**. The extent is **per
slot, not a constant**: 96 B at slots 24 and 25, **192 B at slot 26**
(predicted from the independent ladder write-set and confirmed exactly by
the replay's sibling divergence — two sources, one number). The tilesync-only
sync record is now observed at four sites — the decoder's out2, and slots 24,
25 and 26 — a family trait of the `*_tilesync_*` / `*_chained_*` kernels'
chain handoff.

## Controls

| Mode | Slot 24 | Slot 25 | Slot 26 |
|---|---|---|---|
| `execute` ×2 | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** |
| `skip` (no launch) | pool bit-identical to input and non-matching — diff counts **344,085 / 344,104 / 344,110**, exactly the ladder write-set counts | same, diffs **340,602 / 341,160 / 341,167** = the s24→s25 ladder counts | same, diffs **342,905 / 342,801 / 342,841** = the s25→s26 ladder counts |
| `wrong` (weights rotated inwards 1,024 B) | rejects in all entries (319,425–319,537 bytes differ), pool changed from input (dead-launch detection) | rejects (319,545–319,658 bytes differ), pool changed from input | rejects (341,086–342,203 bytes differ), pool changed from input |
| `tail` (bytes beyond the record replaced 0x5A vs 0xA5) | **output unchanged** — weight reads bounded by the record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 917,568-byte record |

Two cross-tool agreements pin the numbers: the skip diff counts equal the
independent `ladder-report diff` changed-byte counts to the byte, and the
execute entries' change counts equal the summed `ladder-names` segment counts
of the same interval (slots 25→26: 342,713/342,609/342,649 tensor + 192
sync; the fp8 siblings show the tensor-only counts, exactly 192 less).

## Method

`vr/native-slot-replay-probe.py` now carries a **per-slot contract table**
(`CONTRACTS`, `--slot {24,25,26}`): parameter-block size and word classes
(pool / weights / constant / zero), output and sync word indices and the
sync record's changed extent (`sync_bytes` — measured per slot in the ladder
write-sets, enforced as the sibling's divergence), weight record binding,
and the input/reference capture pair. Grid geometry is resolved at the
slot's own launch position (handles are shared across slots at different
geometries). The replay substitutes our own device addresses at the vendor's
own pool offsets; guards sit around every allocation and the weight buffer is
immutable-checked. Evidence
(`vr/logs/slot-replay/s{24,25,26}/{a,b,skip,wrong,tail}`) retains digests and
bounded windows only (windows are written before verdicts on any mismatch);
the `pair` command persists its result as `pair.json` beside the first run.

Self-test: **25 corruption controls** (12 verdict, 6 consistency, 5 pair, 1
metadata, 1 grid) each rejected with its pinned label or message, plus **19
known answers** (3 diff, 4 layout, 4 following-record, 1 metadata, 1 grid, 6
contract-table shape per contract). The contract table's shape is itself
pinned (class-table length = parameter size, exactly one weights word,
pool-class region words, sync extent within the 512-byte sync stride).

## Evidence

Runs on 15/10/2026 against the final source of this stage:

- `vr/logs/ladder/{s26,s26b}` — the new slot-26 reference captures
  (validated: cut accounting, pool digests, the slot-aware output canary,
  input digests; `ladder-report.py pair` enforces byte-equal pools with a
  fail-closed exit — its passing report is retained as `pair-s26-s26b.json`).
  Traces `vr/logs/capture/nvapi-trace-ladder-s26{,b}.log`.
- `vr/logs/ladder/diff-s25-s26.json`, `vr/logs/ladder/names-s25-s26.txt` —
  the measured slot-26 write-set (2 segments per frame, zero unattributed:
  342,713/342,609/342,649 tensor + 192 sync).
- `vr/logs/slot-replay/s26/…` — the five replay runs plus the persisted pair
  result (`pair.json`, bit-identical, run-digest bound). One earlier attempt
  was killed mid-run by a host-side command timeout while a second run
  overlapped it; the partial directory is retained as
  `a-superseded-killed-by-host-timeout` and the completed runs re-taken
  cleanly (`execute-a.log`, `controls.log` retained).
- `vr/logs/ladder/{s25,s25b}`, `vr/logs/ladder/diff-s24-s25.json`,
  `vr/logs/ladder/names-s24-s25.txt`, `vr/logs/slot-replay/s{24,25}/…` —
  the slot-24/25 stage's equivalent evidence (ten replay runs plus two
  persisted pair results, all pair bit-identical).
- The slot-24 runs were **re-taken against the contract-table source** and
  reproduce the stage-16 evidence (`*-superseded-pre-contract-table`) digests
  **exactly** across every recorded entry field.
- `python3 vr/native-slot-replay-probe.py self-test`, `python3
  vr/ladder-report.py self-test` (19 controls) and `python3
  vr/ladder-names.py self-test` all pass.

## Scope and limitations

- The claim covers **three kernels of the 156** (slots 24–26), one resolution
  (720×720, eye0), three frames under the stationary-input protocol. The
  method is generic — the parameter block, geometry, word classes and weight
  record are all contract fields now — but each remaining kernel still needs
  its own reference capture at sN+1 before it can be replayed.
- The tail control bounds weight reads to the record **for these kernels**;
  read-only out-of-bounds accesses beyond the guard window are not detected
  by canaries (same caveat as earlier stages).
- The `wrong`-mode rotation follows the decoder record's 1,024-byte
  alternative-split convention; no split semantics are claimed for these
  records.
- The write-set does not separate reads from writes: slot 25's w0, w1 and w7
  reads (previous output, residual, previous sync) and slot 26's w0 and w5
  reads are inferred from the attribution (those words never change) and the
  word map, not from a measured read-set.
- The sync extents (96/96/192 B) are the records' **changed** extents under
  this protocol; records whose unchanged bytes extend further are not
  excluded.
- This stage uses stored learned weight bytes through three extracted records
  (bounded model-weight computation, [NATIVE-FP8.md](NATIVE-FP8.md)
  precedent) and is **not** a neural-rendered image. Proprietary
  reuse/publication rights remain unresolved; evidence retains digests and
  bounded 8 KiB windows (written only on mismatch) and stays local.

## Next

Repeat the replay for the remaining named kernels of the interval (the
`qkv`/`proj`/`ffwd` pairs of [LADDER-NAMES.md](LADDER-NAMES.md)) and the
decoder-pyramid slots — each needs a reference capture at its own cut point
(`run-feature-probe.sh ladder sN N`) plus one contract entry, and
occasionally a small probe generalisation (slot 26 needed per-slot grid
resolution and an alignment-aware record lookup). The
temporal/history state (`prev_output`, counter) then connects the kernels
into a full native evaluation for the native-vs-vendor comparison of the
whole pipeline.
