# Native pre-decoder kernel replay (slots 24, 25) — 15/10/2026

## Result

**Two pre-decoder kernels of the vendor's 156-launch sequence now replay
natively on Linux and reproduce the vendor's own captured state
byte-for-byte.** Each kernel's launch is rebuilt from its captured parameter
block and executed through the **native Linux CUDA driver** on the RTX 4070
Ti SUPER — no Windows code, DirectX or Proton — and the native pool state
after the launch equals the vendor's ladder capture at the matching cut point
([LADDER.md](LADDER.md)) over the **whole 62,684,160-byte pool: 0 differing
bytes on all three frames, in both execute runs, for both slots**.

| Slot | Kernel (NVAPI-ARGS.md) | Contract | Cut | Grid |
|---|---|---|---|---|
| 24 | `cc_split_swin_16h_ffwd_inpview_512_tilesync_fp8` | reads w0; writes output w1 (344,064 B) + sync w6 (96 B) | s23→s24 | (4,3,2) |
| 25 | `cc_split_swin_16h_ffwd_proj_inpview_512_chained_fp8` | reads w0 + residual w1 + previous sync w7; writes output w2 (344,064 B) + sync w6 (96 B) | s24→s25 | (8,3,1) |

Slot 25 is the `ffwd→proj` pair's second half and the first kernel with the
**chained sync handoff**: it consumes slot 24's sync record (w7 at pool
offset 48,640) and writes its own at w6 (49,152) — which the attribution
shows is exactly slot 26's w5 input ([names-s24-s25.txt](logs/ladder/names-s24-s25.txt)).
Its w1 is slot 24's w0 buffer (the residual input; the two kernels swap the
same two 344,064-byte regions, the `inpview` pattern), and its w2 feeds
slot 26's qkv w0.

Everything each launch needs was resolved from retained evidence — no new
capture semantics were invented:

| Element | Source |
|---|---|
| grid/block/smem | forwarder trace created/launch records, pinned equal across runs and evals |
| parameter block | trace slot dump (56 B / 72 B), pinned equal across evals, runs and the capture sidecar |
| input pool state | the capture after slot n−1 (exactly what slot n reads) |
| reference pool state | the capture after slot n |
| weight bytes | `block23.layer0.layer` (524,288 B) / `block23.layer1.layer` (263,168 B) from the DLL's WEIGHTS_HT resource |

The slot-25 word map (72-byte block, 9 words): w0 output-of-slot-24, w1
residual input, w2 output, w3 weights, w4 the recovered `(24, 28)` geometry
overrides constant, w5 reserved 0, w6 sync out, w7 sync in, w8 reserved 0.

## The weight-heap layout

The 140.9 MiB weight heap is the **WEIGHTS_HT record payloads concatenated at
512-byte alignment**. The layout is pinned by the decoder's live weights
(word 7 of the decoder boundary, [DECODER-BOUNDARY.md](DECODER-BOUNDARY.md))
at `block39.layer0.layer`'s payload, and slot 24's word 2 and slot 25's word
3 both land **exactly at their record payload starts** (`block23.layer0.layer`
at heap offset 6,733,312; `block23.layer1.layer` at 7,257,600 — directly
after layer0's 524,288-byte payload). Beyond the anchors, the probe computes
and enforces the structural claim itself (`setup.heap_scan`, retained in
every run's evidence): every heap-pointing parameter word of every scanned
trace lands at a record payload start — **459 of 459 in each scan record,
zero mid-record** (the scans cover the s23, s24 and s25 traces; s24 is
scanned by both contracts' setups).

## Variant finding: the tilesync/chained sync record

Both contracts' fp8 siblings (`..._512_fp8`) are **created but never
launched by the vendor** (pinned: 0 launches in every trace), so they have no
launch contract of their own; each is replayed at the live variant's geometry
as a variant probe. In both cases the sibling's **output tensor matches the
reference byte-for-byte** but its **96-byte sync record is left at the input
state** (`sync_from_input` true on every entry) while the live variant
(tilesync at slot 24, chained at slot 25) writes it. That divergence
confinement is **enforced** (`sibling-divergence`): the differences inside
the sync record must equal the total difference count over the whole pool —
which proves every differing byte lies within the record — and the record
itself must differ in all 96 bytes. The tilesync-only 96-byte sync record is
now observed at three sites — the decoder's out2, and slots 24 and 25 — a
family trait of the `*_tilesync_*` / `*_chained_*` kernels' chain handoff.

## Controls

| Mode | Slot 24 | Slot 25 |
|---|---|---|
| `execute` ×2 | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** |
| `skip` (no launch) | pool bit-identical to input and non-matching — diff counts **344,085 / 344,104 / 344,110**, exactly the ladder write-set counts | same, diffs **340,602 / 341,160 / 341,167** = the s24→s25 ladder counts |
| `wrong` (weights rotated inwards 1,024 B) | rejects in all entries (319,425–319,537 bytes differ), pool changed from input (dead-launch detection) | rejects (319,545–319,658 bytes differ), pool changed from input |
| `tail` (bytes beyond the record replaced 0x5A vs 0xA5) | **output unchanged** — weight reads bounded by the record | **output unchanged** — bounded by the 263,168-byte record |

Two cross-tool agreements pin the numbers: the skip diff counts equal the
independent `ladder-report diff` changed-byte counts to the byte, and the
execute entries' change counts (340,506/341,064/341,071 tensor + 96 sync)
equal the summed `ladder-names` segment counts of the same interval.

## Method

`vr/native-slot-replay-probe.py` now carries a **per-slot contract table**
(`CONTRACTS`, `--slot {24,25}`): parameter-block size and word classes
(pool / weights / constant / zero), output and sync word indices, weight
record binding, and the input/reference capture pair. The replay substitutes
our own device addresses at the vendor's own pool offsets; guards sit around
every allocation and the weight buffer is immutable-checked. Evidence
(`vr/logs/slot-replay/s{24,25}/{a,b,skip,wrong,tail}`) retains digests and
bounded windows only (windows are written before verdicts on any mismatch);
the `pair` command persists its result as `pair.json` beside the first run.

Self-test: **24 corruption controls** (12 verdict, 6 consistency, 5 pair, 1
metadata) each rejected with its pinned label or message, plus **9 known
answers** (3 diff, 3 layout, 1 metadata, 2 contract-table shape). The
contract table's shape is itself pinned (class-table length = parameter
size, exactly one weights word, pool-class region words).

## Evidence

Runs on 15/10/2026 against the final source of this stage:

- `vr/logs/ladder/{s25,s25b}` — the new slot-25 reference captures
  (validated: cut accounting, pool digests, the slot-aware output canary,
  input digests; `ladder-report.py pair` enforces byte-equal pools with a
  fail-closed exit — its passing report is retained as `pair-s25-s25b.json`).
  Traces `vr/logs/capture/nvapi-trace-ladder-s25{,b}.log`.
- `vr/logs/ladder/diff-s24-s25.json`, `vr/logs/ladder/names-s24-s25.txt` —
  the measured slot-25 write-set (2 segments per frame, zero unattributed).
- `vr/logs/slot-replay/s{24,25}/…` — the ten replay runs plus the two
  persisted pair results (`pair.json`, bit-identical; each records the
  sha256 of the two `run.json` files it compared, so a re-taken run
  invalidates a stale pairing).
- The slot-24 runs were **re-taken against the contract-table source** and
  reproduce the stage-16 evidence (`*-superseded-pre-contract-table`) digests
  **exactly** across every recorded entry field.
- `python3 vr/native-slot-replay-probe.py self-test`, `python3
  vr/ladder-report.py self-test` (19 controls) and `python3
  vr/ladder-names.py self-test` all pass.

## Scope and limitations

- The claim covers **two kernels of the 156** (slots 24–25), one resolution
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
  reads (previous output, residual, previous sync) are inferred from the
  attribution (those words never change) and the word map, not from a
  measured read-set.
- This stage uses stored learned weight bytes through two extracted records
  (bounded model-weight computation, [NATIVE-FP8.md](NATIVE-FP8.md)
  precedent) and is **not** a neural-rendered image. Proprietary
  reuse/publication rights remain unresolved; evidence retains digests and
  bounded 8 KiB windows (written only on mismatch) and stays local.

## Next

Repeat the replay for the remaining named kernels of the interval (the
`qkv`/`proj`/`ffwd` pairs of [LADDER-NAMES.md](LADDER-NAMES.md)) and the
decoder-pyramid slots — each needs a reference capture at its own cut point
(`run-feature-probe.sh ladder sN N`) plus one contract entry. The
temporal/history state (`prev_output`, counter) then connects the kernels
into a full native evaluation for the native-vs-vendor comparison of the
whole pipeline.
