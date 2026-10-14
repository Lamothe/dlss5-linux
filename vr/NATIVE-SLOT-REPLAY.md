# Native pre-decoder kernel replay (slot 24) — 15/10/2026

## Result

**A pre-decoder kernel of the vendor's 156-launch sequence now replays
natively on Linux and reproduces the vendor's own captured state
byte-for-byte.** The vendor's slot-24 launch (`cc_split_swin_16h_ffwd_inpview
_512_tilesync_fp8`, [NVAPI-ARGS.md](NVAPI-ARGS.md)) was rebuilt from its
captured parameter block and executed through the **native Linux CUDA
driver** on the RTX 4070 Ti SUPER — no Windows code, DirectX or Proton — and
the native pool state after the launch equals the vendor's s24 ladder capture
([LADDER.md](LADDER.md)) over the **whole 62,684,160-byte pool: 0 differing
bytes on all three frames, in both execute runs**. This generalises the
decoder replay ([NATIVE-VENDOR-COMPARE.md](NATIVE-VENDOR-COMPARE.md)) from
one boundary kernel to a kernel in the middle of the pipeline.

Everything the launch needs was resolved from retained evidence — no new
capture semantics were invented:

| Element | Source |
|---|---|
| grid (4,3,2), block (32,4,1), smem 0 | forwarder trace created/launch records, pinned equal across both runs and all evals |
| 56-byte parameter block | trace slot-24 dump, pinned equal to the capture sidecar |
| input pool state | s23 capture (after slot 23 = exactly what slot 24 reads) |
| reference pool state | s24 capture (after slot 24) |
| weight bytes | `block23.layer0.layer` payload from the DLL's WEIGHTS_HT resource |

## The weight-heap layout (new structural result)

The 140.9 MiB weight heap is the **WEIGHTS_HT record payloads concatenated at
512-byte alignment**. Two independent live anchors pin the layout exactly:

- the decoder's live weights (word 7 of the decoder boundary,
  [DECODER-BOUNDARY.md](DECODER-BOUNDARY.md)) begin at
  `block39.layer0.layer`'s payload — heap offset **123,736,576**, matching
  the layout exactly;
- slot 24's own word 2 points at **`block23.layer0.layer`'s payload start**
  (heap offset 6,733,312) — an exact record-start landing.

Beyond the two anchors, the probe now computes and enforces the structural
claim itself (`setup.heap_scan`, retained in every run's evidence): every
heap-pointing parameter word of both traces is classified against the
layout — **459 of 459 land exactly at record payload starts in each trace,
zero mid-record.** Every kernel's weight pointer addresses a whole record.
The packed layout totals 147,719,232 bytes against the live allocation's
147,719,680 (448 bytes of tail padding, enforced bound `< 512`).

## Variant finding: the tilesync sync record

The fp8 sibling (`cc_split_swin_16h_ffwd_inpview_512_fp8`) is **created but
never launched by the vendor** (pinned: 0 launches in both traces), so it has
no launch contract of its own; it is replayed at the tilesync geometry as a
variant probe. Its **output tensor matches the reference byte-for-byte**, but
its pool diverges in exactly **96 bytes at pool offset 48,640** — the `w6`
binding: the tilesync variant writes a 96-byte sync record there (matching
the reference), the fp8 sibling **leaves the buffer at its input state**
(`sync_from_input` true on every frame). The same tilesync-only 96-byte
record was observed at the decoder's out2 slot
([NATIVE-VENDOR-COMPARE.md](NATIVE-VENDOR-COMPARE.md)); it is becoming a
family trait of the `*_tilesync_*` kernels. In `wrong` mode the tilesync
record is still written identically (`sync_matches_reference` true with
rotated weights) — its content is unchanged under this rotation.

## Controls

| Mode | Outcome |
|---|---|
| `execute` ×2 | tilesync pool match exact on 3/3 frames each; runs **pair bit-identical** |
| `skip` (no launch) | pool bit-identical to input and non-matching — and its diff against the reference equals **344,085 / 344,104 / 344,110 bytes**, exactly the s23→s24 write-set counts measured by the independent `ladder-report diff` |
| `wrong` (weights rotated inwards 1,024 bytes) | rejects the reference in all entries (319,425–319,537 bytes differ), with the pool enforced to have changed from the input state (dead-launch detection) |
| `tail` (bytes beyond the record replaced by 0x5A vs execute's 0xA5) | **output unchanged** — weight reads are bounded by the record's 524,288 bytes |

The skip cross-check is a genuine independent-tool agreement: the probe's
whole-pool diff counts equal the ladder tool's changed-byte counts to the
byte.

## Method

`vr/native-slot-replay-probe.py` parses the traces, captures and DLL records
it consumes and fails closed on every mismatch (parameter dumps pinned equal
across evals, runs and the sidecar; slot-24 identity pinned to the tilesync
kernel's created handle in each trace; creation geometry pinned across runs;
pool identity pinned across captures; the two heap anchors and the
record-start property of every heap-pointing word pinned against the layout).
The launch rebuilds the parameter block with our own device addresses at the
vendor's own pool offsets; guards sit around every allocation and the weight
buffer is immutable-checked. Evidence
(`vr/logs/slot-replay/{a,b,skip,wrong,tail}`) retains digests and bounded
windows only (windows are written before verdicts on any mismatch).

Self-test: **21 corruption controls** (10 verdict, 5 consistency, 5 pair, 1
metadata) each rejected with its pinned message, plus 7 known answers (3
diff, 3 layout, 1 metadata). One independent review round ran before the
commit: it **failed** the first version with 10 findings — 2 must-fix (an
unevidenced structural claim, now computed and enforced as `setup.heap_scan`;
a false "digests only" statement about retained evidence), 4 should-fix
(including wrong-mode dead-launch detection and slot-24 identity pinning),
4 advisory — all fixed, with all five evidence runs re-taken against the
corrected source.

## Scope and limitations

- The primary claim covers **one kernel of the 156**: slot 24 of the
  pre-decoder transformer stage, one resolution (720×720, eye0), three
  frames under the stationary-input protocol. Other slots' contracts are not
  yet replayed, though the method is now generic (the parameter block, grid
  and weights are all trace/DLL-derived).
- The tail control bounds weight reads to the record **for this kernel**;
  read-only out-of-bounds accesses beyond the guard window are not detected
  by canaries (same caveat as the decoder stage).
- The `wrong`-mode rotation follows the decoder record's 1,024-byte
  alternative-split convention; for `block23` no split semantics are claimed.
- This stage uses stored learned weight bytes through one extracted record
  (bounded model-weight computation, [NATIVE-FP8.md](NATIVE-FP8.md)
  precedent) and is **not** a neural-rendered image. Proprietary
  reuse/publication rights remain unresolved; evidence retains digests and
  bounded 8 KiB windows (written only on mismatch) and stays local.

## Next

Repeat the replay for the remaining named kernels of the interval (the
`qkv`/`proj`/`ffwd` pairs of [LADDER-NAMES.md](LADDER-NAMES.md)) and the
decoder-pyramid slots — each one turns a named write-set into a validated
native kernel. The temporal/history state (`prev_output`, counter) then
connects the kernels into a full native evaluation for the
native-vs-vendor comparison of the whole pipeline.
