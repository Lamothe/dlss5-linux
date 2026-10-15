# Native pre-decoder kernel replay (slots 24-28) — 15/10/2026

## Result

**Five pre-decoder kernels of the vendor's 156-launch sequence now replay
natively on Linux and reproduce the vendor's own captured state
byte-for-byte.** Each kernel's launch is rebuilt from its captured parameter
block and executed through the **native Linux CUDA driver** on the RTX 4070
Ti SUPER — no Windows code, DirectX or Proton — and the native pool state
after the launch equals the vendor's ladder capture at the matching cut point
([LADDER.md](LADDER.md)) over the **whole 62,684,160-byte pool: 0 differing
bytes on all three frames, in both execute runs, for all five slots**.

| Slot | Kernel (NVAPI-ARGS.md) | Contract | Cut | Grid |
|---|---|---|---|---|
| 24 | `cc_split_swin_16h_ffwd_inpview_512_tilesync_fp8` | reads w0; writes output w1 (344,064 B) + sync w6 (96 B) | s23→s24 | (4,3,2) |
| 25 | `cc_split_swin_16h_ffwd_proj_inpview_512_chained_fp8` | reads w0 + residual w1 + previous sync w7; writes output w2 (344,064 B) + sync w6 (96 B) | s24→s25 | (8,3,1) |
| 26 | `cc_split_swin_16h_qkv_512_chained_fp8` | reads output-of-slot-25 w0 + previous sync w5; writes output w1 (344,064 B) + sync w6 (192 B) | s25→s26 | (4,3,4) |
| 27 | `cc_split_swin_16h_proj_512_chained_fp8` | reads output-of-slot-26 w0 + residual w1 + previous sync w7; writes output w2 (344,064 B) + sync w6 (96 B) | s26→s27 | (8,3,1) |
| 28 | `cc_split_swin_16h_ffwd_512_chained_fp8` | reads output-of-slot-27 w0 + previous sync w4; writes output w1 (344,064 B) + sync w6 (96 B) | s27→s28 | (4,3,2) |

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

Slot 27 is the transformer block's **proj half** — the `qkv→proj` pair's
second half, and the `proj` counterpart to slot 25's `ffwd_proj`. It reads
slot 26's output (w0 at 21,298,176), a residual input (w1 at 20,954,112 —
the buffer slot 26 itself read at w0, i.e. slot 25's output) and consumes
slot 26's sync record (w7 at 49,664 — address-equal to slot 26's w6, the
chained handoff confirmed by address identity), and writes its output tensor
at w2 (344,064 B at 21,642,240) plus a 96-byte sync record at w6 (50,176).
The write-set attribution pins exactly those two words
([names-s26-s27.txt](logs/ladder/names-s26-s27.txt): w2 + w6, zero
unattributed, stable across all three frames). Its 96-byte sync extent
returns to the slot-24/25 norm after slot 26's 192-byte record.

Slot 28 is the **next block's ffwd half** — the first kernel outside
`block23` and the first to bind a `block24` weight record. It reads slot
27's output (w0 at 21,642,240 — address-equal to slot 27's w2) and consumes
slot 27's sync record (w4 at 50,176 — address-equal to slot 27's w6, so the
chained handoff's sync-in word moves to w4 on this 56-byte block), and
writes its output tensor at w1 (344,064 B at 21,986,304) plus a 96-byte sync
record at w6 (50,688 — the 512-byte stride continuing: 49,664 → 50,176 →
50,688). The write-set attribution pins exactly those two words
([names-s27-s28.txt](logs/ladder/names-s27-s28.txt): w1 + w6, zero
unattributed, stable across all three frames). Its w1 region is the span
slot 29 (`ffwd_proj_512_chained_fp8`) reads at w0, and its w0 is slot 29's
w1 — the pair swaps the same two regions like slots 24/25.

Everything each launch needs was resolved from retained evidence — no new
capture semantics were invented:

| Element | Source |
|---|---|
| grid/block/smem | forwarder trace created/launch records, pinned equal across runs and evals |
| parameter block | trace slot dump (56 B / 72 B), pinned equal across evals, runs and the capture sidecar |
| input pool state | the capture after slot n−1 (exactly what slot n reads) |
| reference pool state | the capture after slot n |
| weight bytes | `block23.layer0.layer` (524,288 B) / `block23.layer1.layer` (263,168 B) / `block23.layer2.layer` (917,568 B) / `block23.layer3.layer` (263,168 B) / `block24.layer0.layer` (524,288 B) from the DLL's WEIGHTS_HT resource |

The slot-25 word map (72-byte block, 9 words): w0 output-of-slot-24, w1
residual input, w2 output, w3 weights, w4 the recovered `(24, 28)` geometry
overrides constant, w5 reserved 0, w6 sync out, w7 sync in, w8 reserved 0.
The slot-26 word map (56-byte block, 7 words): w0 output-of-slot-25, w1
output, w2 weights, w3 the `(24, 28)` constant, w4 reserved 0, w5 sync in
(slot 25's record at 49,152), w6 sync out (49,664 — the same 512-byte
stride). The slot-27 word map (72-byte block, 9 words) is slot 25's shape
again: w0 output-of-slot-26, w1 residual input, w2 output, w3 weights, w4
the `(24, 28)` constant, w5 reserved 0, w6 sync out (50,176), w7 sync in
(slot 26's record at 49,664), w8 reserved 0. The slot-28 word map (56-byte
block, 7 words) is slot 24's shape with the chained sync-in moved to w4: w0
output-of-slot-27, w1 output, w2 weights, w3 the `(24, 28)` constant, w4
sync in (slot 27's record at 50,176), w5 reserved 0, w6 sync out (50,688).

## The weight-heap layout

The 140.9 MiB weight heap is the **WEIGHTS_HT record payloads concatenated at
512-byte alignment**. The layout is pinned by the decoder's live weights
(word 7 of the decoder boundary, [DECODER-BOUNDARY.md](DECODER-BOUNDARY.md))
at `block39.layer0.layer`'s payload, and slot 24's word 2, slot 25's word 3,
slot 26's word 2, slot 27's word 3 and slot 28's word 2 all land **exactly at
their record
payload starts** (`block23.layer0.layer` at heap offset 6,733,312;
`block23.layer1.layer` at 7,257,600 — directly after layer0's 524,288-byte
payload; `block23.layer2.layer` at 7,520,768 — directly after layer1;
`block23.layer3.layer` at 8,438,784 — directly after layer2 at the
alignment boundary below; `block24.layer0.layer` at 8,701,952 — directly
after layer3's 263,168-byte payload, so the record order is the DLL's
declaration order and block24 repeats block23's layer0-3 shape). layer2's
917,568-byte payload is not
512-aligned: the following record starts 448 bytes later at the alignment
boundary (arithmetic from the probe's live DLL walk, which enforces exactly
one record at that boundary), so the tail control's following-record lookup
is alignment-aware. Beyond the anchors, the probe
computes and enforces the structural claim itself (`setup.heap_scan`, retained
in every run's evidence): every heap-pointing parameter word of every scanned
trace lands at a record payload start — **459 of 459 in each scan record,
zero mid-record** (the scans cover the s23–s28 traces; s24, s25, s26 and s27
are each scanned by two adjacent contracts' setups). The binding's *identity*
(`block24.layer0.layer`, 524,288 B here) is enforced at runtime and named in
the source contract table; what the evidence retains is the binding's
*content* and *position* — `setup.record_sha256` and `setup.heap_offset` —
plus the enforced equality to the layout walk, so a record rename would not
disturb any run's verdicts.

## Variant finding: the tilesync/chained sync record

All five contracts' fp8 siblings (`..._512_fp8`) are **created but never
launched by the vendor** (pinned: 0 launches in every trace), so they have no
launch contract of their own; each is replayed at the live variant's geometry
as a variant probe. In every case the sibling's **output tensor matches the
reference byte-for-byte** but its **sync record is left at the input state**
(`sync_from_input` true on every entry) while the live variant (tilesync at
slot 24, chained at slots 25, 26, 27 and 28) writes it. That divergence confinement
is **enforced** (`sibling-divergence`): the differences inside the sync
record must equal the total difference count over the whole pool — which
proves every differing byte lies within the record — and the record itself
must differ in **every byte of its changed extent**. The extent is **per
slot, not a constant**: 96 B at slots 24, 25, 27 and 28, **192 B at slot 26**
(predicted from the independent ladder write-set and confirmed exactly by
the replay's sibling divergence — two sources, one number). The tilesync-only
sync record is now observed at six sites — the decoder's out2, and slots 24,
25, 26, 27 and 28 — a family trait of the `*_tilesync_*` / `*_chained_*`
kernels' chain handoff.

## Controls

| Mode | Slot 24 | Slot 25 | Slot 26 | Slot 27 | Slot 28 |
|---|---|---|---|---|---|
| `execute` ×2 | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** |
| `skip` (no launch) | pool bit-identical to input and non-matching — diff counts **344,085 / 344,104 / 344,110**, exactly the ladder write-set counts | same, diffs **340,602 / 341,160 / 341,167** = the s24→s25 ladder counts | same, diffs **342,905 / 342,801 / 342,841** = the s25→s26 ladder counts | same, diffs **340,726 / 341,250 / 341,150** = the s26→s27 ladder counts | same, diffs **344,027 / 344,077 / 344,069** = the s27→s28 ladder counts |
| `wrong` (weights rotated inwards 1,024 B) | rejects in all entries (319,425–319,537 bytes differ), pool changed from input (dead-launch detection) | rejects (319,545–319,658 bytes differ), pool changed from input | rejects (341,086–342,203 bytes differ), pool changed from input | rejects (342,666–342,849 bytes differ), pool changed from input | rejects (343,654–343,784 bytes differ), pool changed from input |
| `tail` (bytes beyond the record replaced 0x5A vs 0xA5) | **output unchanged** — weight reads bounded by the record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 917,568-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 524,288-byte record |

Two cross-tool agreements pin the numbers: the skip diff counts equal the
independent `ladder-report diff` changed-byte counts to the byte, and the
execute entries' change counts equal the summed `ladder-names` segment counts
of the same interval (slots 25→26: 342,713/342,609/342,649 tensor + 192
sync, slots 26→27: 340,630/341,154/341,054 tensor + 96 sync, slots 27→28:
343,931/343,981/343,973 tensor + 96 sync; the fp8
siblings show the tensor-only counts, exactly the sync extent less).

## Method

`vr/native-slot-replay-probe.py` now carries a **per-slot contract table**
(`CONTRACTS`, `--slot {24,25,26,27,28}`): parameter-block size and word classes
(pool / weights / constant / zero), output and sync word indices and the
sync record's changed extent (`sync_bytes` — measured per slot in the ladder
write-sets, enforced as the sibling's divergence), weight record binding,
and the input/reference capture pair. Grid geometry is resolved at the
slot's own launch position (handles are shared across slots at different
geometries). The replay substitutes our own device addresses at the vendor's
own pool offsets; guards sit around every allocation and the weight buffer is
immutable-checked. Evidence
(`vr/logs/slot-replay/s{24,25,26,27,28}/{a,b,skip,wrong,tail}`) retains digests and
bounded windows only (windows are written before verdicts on any mismatch);
the `pair` command persists its result as `pair.json` beside the first run.

Self-test: **25 corruption controls** (12 verdict, 6 consistency, 5 pair, 1
metadata, 1 grid) each rejected with its pinned label or message, plus **48
known answers** (3 diff, 4 layout, 4 following-record, 1 metadata, 1 grid,
and 7 contract-table shape checks per contract — 35 across the five
contracts, counted per contract rather than per table). The contract table's shape is itself
pinned (slot key equals the table key, class-table length = parameter size,
exactly one weights word,
pool-class region words, sync extent within the 512-byte sync stride,
output and sync words distinct, six distinct plan entries — every shape
check mutation-tested at development time against a corrupted contract
entry, each rejection with its pinned message; the corruptions are
development-time in-memory mutations, not retained self-test fixtures).

## Evidence

Runs on 15/10/2026 against the final source of this stage:

- `vr/logs/ladder/{s28,s28b}` — the new slot-28 reference captures
  (validated: cut accounting, pool digests, the slot-aware output canary,
  input digests; `ladder-report.py pair` enforces byte-equal pools with a
  fail-closed exit — its passing report is retained as `pair-s28-s28b.json`,
  pool digests equal across runs). Traces
  `vr/logs/capture/nvapi-trace-ladder-s28{,b}.log`.
- `vr/logs/ladder/diff-s27-s28.json`, `vr/logs/ladder/names-s27-s28.txt` —
  the measured slot-28 write-set (2 segments per frame, zero unattributed:
  343,931/343,981/343,973 tensor + 96 sync).
- `vr/logs/slot-replay/s28/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s). The
  review-driven self-test fix (a vacuous entry-plan shape check, replaced by
  two discriminating ones; all seven shape checks then mutation-tested at
  development time) touches only `self_test()`;
  the five runs and the four rechecks were nevertheless re-taken against the
  final source and pair **bit-identical** with the pre-fix runs (retained as
  `*-superseded-pre-selftest-fix`, each with a persisted reproduction
  `pair.json`), so the verdicts do not depend on the fix.
- `vr/logs/ladder/{s27,s27b}` — the slot-27 stage's reference captures
  (validated: cut accounting, pool digests, the slot-aware output canary,
  input digests; `ladder-report.py pair` enforces byte-equal pools with a
  fail-closed exit — its passing report is retained as `pair-s27-s27b.json`).
  Traces `vr/logs/capture/nvapi-trace-ladder-s27{,b}.log`.
- `vr/logs/ladder/diff-s26-s27.json`, `vr/logs/ladder/names-s26-s27.txt` —
  the measured slot-27 write-set (2 segments per frame, zero unattributed:
  340,630/341,154/341,054 tensor + 96 sync).
- `vr/logs/slot-replay/s27/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, run-digest bound). Two earlier
  attempts were killed mid-run by host-side command timeouts and the partial
  directories are retained as `wrong-superseded-killed-by-host-timeout` (no
  files — killed before its first window) and
  `tail-superseded-killed-by-host-timeout` (mismatch windows only); neither
  has a `run.json`. The completed runs were re-taken cleanly
  (`execute-a.log`, `execute-b.log`, `controls.log`, `tail-run.log`
  retained). The first attempt also used a flat `s27-<mode>` layout outside
  the per-slot namespace; it is retained as
  `s27-<mode>-superseded-flat-layout` and **all five** pair **bit-identical**
  with the re-taken runs — one persisted `pair.json` per mode, each
  digest-bound with equal `run.json` SHA-256s.
- `vr/logs/ladder/{s26,s26b}` — the slot-26 stage's reference captures
  (`pair-s26-s26b.json`); `vr/logs/ladder/diff-s25-s26.json`,
  `vr/logs/ladder/names-s25-s26.txt` — its measured write-set.
- `vr/logs/slot-replay/s26/…` — the slot-26 stage's five replay runs plus
  the persisted pair result; `vr/logs/slot-replay/s{24,25}/…` and
  `vr/logs/ladder/{s25,s25b}`, `diff-s24-s25.json`, `names-s24-s25.txt` —
  the slot-24/25 stage's equivalent evidence (all pair bit-identical).
- **Source neutrality:** the slot 24–27 execute runs were re-taken against
  **this** stage's source (`s{24,25,26,27}/recheck-stage20-final-source`) and
  pair **bit-identical** with the retained runs (the comparator covers every
  entry field except the `retained` window filenames, and the persisted
  `run.json` SHA-256s are equal — byte-equality of the whole file), so all
  retained evidence stands under the final source. The pre-fix recheck runs
  are retained as `recheck-stage20-superseded-pre-selftest-fix`; their
  persisted `pair.json` labels carry the directory's pre-rename name (the
  records travelled with the rename), and the digests they record equal the
  fresh runs' `run.json` SHA-256s, so pre-fix ≡ fresh ≡ retained
  byte-for-byte. The slot 24/25/26 runs
  were likewise re-taken against the stage-19 source
  (`s{24,25,26}/recheck-final-source`) and against the contract-table source
  earlier (`a-superseded-pre-contract-table`, `b-superseded-pre-contract-table`),
  each time reproducing the retained digests exactly.
- `python3 vr/native-slot-replay-probe.py self-test` (48 known answers), `python3
  vr/ladder-report.py self-test` (19 controls) and `python3
  vr/ladder-names.py self-test` (14 controls) all pass.

## Scope and limitations

- The claim covers **five kernels of the 156** (slots 24–28), one resolution
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
  reads (previous output, residual, previous sync), slot 26's w0 and w5
  reads and slot 28's w0 and w4 reads are inferred from the attribution (those words never change) and the
  word map, not from a measured read-set.
- The sync extents (96/96/192/96/96 B for slots 24–28) are the records'
  **changed** extents under
  this protocol; records whose unchanged bytes extend further are not
  excluded.
- This stage uses stored learned weight bytes through five extracted records
  (bounded model-weight computation, [NATIVE-FP8.md](NATIVE-FP8.md)
  precedent) and is **not** a neural-rendered image. Proprietary
  reuse/publication rights remain unresolved; evidence retains digests and
  bounded 8 KiB windows (written only on mismatch) and stays local.

## Next

Repeat the replay for the remaining named kernels of
[LADDER-NAMES.md](LADDER-NAMES.md) — the encoder (slots 1–23), the
transformer interval's further `qkv`/`proj`/`ffwd` pairs (29–56), the ViT
stage (57–98), transformer-2 (100–131) and the decoder pyramid (132–153);
slot 99 (the decoder) is already covered by
[NATIVE-VENDOR-COMPARE.md](NATIVE-VENDOR-COMPARE.md) and slot 154 (the final
blend) writes only outside the working pool. Each remaining kernel needs a
reference capture at its
own cut point (`run-feature-probe.sh ladder sN N`) plus one contract entry,
and occasionally a small probe generalisation (slot 26 needed per-slot grid
resolution and an alignment-aware record lookup). Slots 24–28 now cover one
complete `ffwd→proj` pair and one `qkv→proj` pair plus the first kernel of a
second block, so the pair pattern, the chained sync handoff (with its
per-slot word position: w5 at slot 26, w7 at slots 25/27/29, w4 at slot 28)
and the record-ordering pattern are established: block24 repeats block23's
layer0-3 record shape in declaration order, so each remaining slot's weight
record is predictable from its parameter block's weights word plus the
layout walk. The natural next candidate is slot 29
(`cc_split_swin_16h_ffwd_proj_512_chained_fp8` — the second pair's second
half, whose weights word already lands at `block24.layer1.layer`). The
temporal/history state (`prev_output`, counter) then connects the kernels
into a full native evaluation for the native-vs-vendor comparison of the
whole pipeline.
