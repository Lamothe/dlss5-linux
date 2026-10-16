# Native pre-decoder kernel replay (slots 24-36) — 15-16/10/2026

## Result

**Thirteen pre-decoder kernel launches (slots 24–36; six distinct kernel
names — slots 26/30/34 share one kernel across two geometries, slots
27/31/35, 28/32/36 and 29/33 reuse kernels at the same geometry) of the
vendor's 156-launch sequence now replay
natively on Linux and reproduce the vendor's own captured state
byte-for-byte.** Each kernel's launch is rebuilt from its captured parameter
block and executed through the **native Linux CUDA driver** on the RTX 4070
Ti SUPER — no Windows code, DirectX or Proton — and the native pool state
after the launch equals the vendor's ladder capture at the matching cut point
([LADDER.md](LADDER.md)) over the **whole 62,684,160-byte pool: 0 differing
bytes on all three frames, in both execute runs, for all thirteen slots**.

| Slot | Kernel (NVAPI-ARGS.md) | Contract | Cut | Grid |
|---|---|---|---|---|
| 24 | `cc_split_swin_16h_ffwd_inpview_512_tilesync_fp8` | reads w0; writes output w1 (344,064 B) + sync w6 (96 B) | s23→s24 | (4,3,2) |
| 25 | `cc_split_swin_16h_ffwd_proj_inpview_512_chained_fp8` | reads w0 + residual w1 + previous sync w7; writes output w2 (344,064 B) + sync w6 (96 B) | s24→s25 | (8,3,1) |
| 26 | `cc_split_swin_16h_qkv_512_chained_fp8` | reads output-of-slot-25 w0 + previous sync w5; writes output w1 (344,064 B) + sync w6 (192 B) | s25→s26 | (4,3,4) |
| 27 | `cc_split_swin_16h_proj_512_chained_fp8` | reads output-of-slot-26 w0 + residual w1 + previous sync w7; writes output w2 (344,064 B) + sync w6 (96 B) | s26→s27 | (8,3,1) |
| 28 | `cc_split_swin_16h_ffwd_512_chained_fp8` | reads output-of-slot-27 w0 + previous sync w4; writes output w1 (344,064 B) + sync w6 (96 B) | s27→s28 | (4,3,2) |
| 29 | `cc_split_swin_16h_ffwd_proj_512_chained_fp8` | reads output-of-slot-28 w0 + residual w1 + previous sync w7; writes output w2 (344,064 B) + sync w6 (96 B) | s28→s29 | (8,3,1) |
| 30 | `cc_split_swin_16h_qkv_512_chained_fp8` | reads output-of-slot-29 w0 + previous sync w5; writes output w1 (344,064 B) + sync w6 (256 B) | s29→s30 | (4,4,4) |
| 31 | `cc_split_swin_16h_proj_512_chained_fp8` | reads output-of-slot-30 w0 + residual w1 + previous sync w7; writes output w2 (344,064 B) + sync w6 (96 B) | s30→s31 | (8,3,1) |
| 32 | `cc_split_swin_16h_ffwd_512_chained_fp8` | reads output-of-slot-31 w0 + previous sync w4; writes output w1 (344,064 B) + sync w6 (96 B) | s31→s32 | (4,3,2) |
| 33 | `cc_split_swin_16h_ffwd_proj_512_chained_fp8` | reads output-of-slot-32 w0 + residual w1 + previous sync w7; writes output w2 (344,064 B) + sync w6 (96 B) | s32→s33 | (8,3,1) |
| 34 | `cc_split_swin_16h_qkv_512_chained_fp8` | reads output-of-slot-33 w0 + previous sync w5; writes output w1 (344,064 B) + sync w6 (192 B) | s33→s34 | (4,3,4) |
| 35 | `cc_split_swin_16h_proj_512_chained_fp8` | reads output-of-slot-34 w0 + residual w1 + previous sync w7; writes output w2 (344,064 B) + sync w6 (96 B) | s34→s35 | (8,3,1) |
| 36 | `cc_split_swin_16h_ffwd_512_chained_fp8` | reads output-of-slot-35 w0 + previous sync w4; writes output w1 (344,064 B) + sync w6 (96 B) | s35→s36 | (4,3,2) |

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
The kernel handle is the `qkv_512` family's shared handle — all sixteen
`qkv` launches of both transformer intervals use it (slots
26/30/34/38/42/46/50/54 and
102/106/110/114/118/122/126/130) at exactly two geometries (grid 4,3,4 at
slots 26/34/42/50/102/110/118/126; 4,4,4 at slots
30/38/46/54/106/114/122/130), so the grid is
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

Slot 29 is the **block24 ffwd-proj pair's second half** — the `ffwd_proj`
counterpart to slot 25's, same 72-byte block and grid (8,3,1). It reads slot
28's output (w0 at 21,986,304 — address-equal to slot 28's w1) and a
residual input (w1 at 21,642,240 — the buffer slot 27 wrote its output to),
consumes slot 28's sync record (w7 at 50,688 — address-equal to slot 28's
w6, the chained handoff) and writes its output tensor at w2 (344,064 bytes
at 22,330,368) plus a 96-byte sync record at w6 (51,200 — the 512-byte
stride continuing: 49,664 → 50,176 → 50,688 → 51,200). The write-set
attribution pins exactly those two words
([names-s28-s29.txt](logs/ladder/names-s28-s29.txt): w2 + w6, zero
unattributed, stable across all three frames). It binds `block24.layer1.layer`
(263,168 B) — measured at its parameter block's weights word (heap offset
9,226,240), the record directly following slot 28's, so the heap-layout walk
holds beyond block23.

Slot 30 is the **block24 qkv half** — slot 26's counterpart, at the shared
handle's second geometry (grid 4,4,4, resolved at the slot's own launch
position as the slot-26 note predicted). It reads slot 29's output (w0 at
22,330,368 — address-equal to slot 29's w2) and consumes slot 29's sync
record (w5 at 51,200 — address-equal to slot 29's w6, the chained handoff)
and writes its output tensor at w1 (344,064 bytes at 22,674,432) plus a sync
record at w6 (51,712 — the 512-byte stride continuing: 50,688 → 51,200 →
51,712). The write-set attribution pins exactly those two words
([names-s29-s30.txt](logs/ladder/names-s29-s30.txt): w1 + w6, zero
unattributed, stable across all three frames), and the sync write's changed
extent is **256 bytes** — a third per-slot extent after slot 26's 192,
predicted from that write-set and confirmed exactly by the replay's sibling
divergence. It binds `block24.layer2.layer` (917,568 B — the record directly
following `block24.layer1.layer`, measured at its parameter block's weights
word at heap offset 9,489,408). Its parameter block also carries the
contract table's first **two-valued constant block**: w3 is the (24, 28)
geometry constant as usual, but w4 is `0xFFFFFFFCFFFFFFFC` — where the
same-shape slot 26 block reserves its w4 as zero. The per-word constant
values are pinned in the contract table (`constant_values`) and passed
through to the replay unchanged.

Slot 31 is the **block24 qkv-proj pair's second half** — the `proj`
counterpart to slot 27's, same 72-byte block and grid (8,3,1). It reads slot
30's output (w0 at 22,674,432 — address-equal to slot 30's w1) and a
residual input (w1 at 22,330,368 — the buffer slot 29 wrote its output to),
consumes slot 30's sync record (w7 at 51,712 — address-equal to slot 30's
w6, the chained handoff) and writes its output tensor at w2 (344,064 bytes
at 23,018,496) plus a 96-byte sync record at w6 (52,224 — the 512-byte
stride continuing: 50,688 → 51,200 → 51,712 → 52,224). The write-set
attribution pins exactly those two words
([names-s30-s31.txt](logs/ladder/names-s30-s31.txt): w2 + w6, zero
unattributed, stable across all three frames). It binds `block24.layer3.layer`
(263,168 B) — measured at its parameter block's weights word (heap offset
10,407,424), the record directly following slot 30's at the alignment
boundary below layer2's unaligned payload (the block23 layer2→3 padding
pattern repeating), so the record-ordering walk holds through the pair. Its
parameter block carries two constant words like its pair partner slot 30:
w4 the (24, 28) geometry constant and w8 `0xFFFFFFFCFFFFFFFC` — where the
same-shape slot 27 block reserves its w8 as zero. The constant therefore
appears in **both halves of block24's qkv→proj pair** (slot 30's w4, slot
31's w8) and in neither half of block23's (slots 26/27) nor block24's
ffwd→proj pair (slot 29's w8 is reserved zero) — a per-pair parameter
difference; slots 34/35 show the pattern holds in block25's pair at that
pair's own value.

Slot 32 is the **block25 ffwd half** — the next `ffwd→proj` pair's first
half, slot 28's counterpart. It reads slot 31's output (w0 at 23,018,496 —
address-equal to slot 31's w2) and consumes slot 31's sync record (w4 at
52,224 — address-equal to slot 31's w6, the chained handoff at the ffwd
shape's w4 position, like slot 28) and writes its output tensor at w1
(344,064 bytes at 23,362,560) plus a 96-byte sync record at w6 (52,736 —
the 512-byte stride continuing: 51,200 → 51,712 → 52,224 → 52,736). The
write-set attribution pins exactly those two words
([names-s31-s32.txt](logs/ladder/names-s31-s32.txt): w1 + w6, zero
unattributed, stable across all three frames). It binds `block25.layer0.layer`
(524,288 B) — measured at its parameter block's weights word (heap offset
10,670,592), the record directly following slot 31's `block24.layer3.layer`
263,168-byte payload at the alignment boundary (that payload is 512-aligned,
so no padding intervenes), so the record-ordering walk continues into
block25. Its parameter block is the slot-28 word map exactly (56-byte block,
7 words: w0 output-of-slot-31, w1 output, w2 weights, w3 the (24, 28)
constant, w4 sync in, w5 reserved 0, w6 sync out) — no second constant here
(slots 34/35 below show the qkv→proj pair carries its second constant in
both halves at one value per pair — block24's at `0xFFFFFFFCFFFFFFFC`,
block25's at `0x00000000FFFFFFFC`; block23's pair and the ffwd→proj halves
reserve the word). The kernel name and grid (4,3,2) are shared with slot 28 (the
`ffwd_512` handle at the same launch shape — the same-geometry reuse slots
27/31 established at the `proj` handle), so slot 32 is a different parameter
block and weight record at an identical launch shape to slot 28.

Slot 33 is the **block25 ffwd-proj pair's second half** — the `ffwd_proj`
counterpart to slots 25 and 29, same 72-byte block and grid (8,3,1). It
reads slot 32's output (w0 at 23,362,560 — address-equal to slot 32's w1)
and a residual input (w1 at 23,018,496 — the buffer slot 31 wrote its
output to), consumes slot 32's sync record (w7 at 52,736 — address-equal
to slot 32's w6, the chained handoff) and writes its output tensor at w2
(344,064 bytes at 23,706,624) plus a 96-byte sync record at w6 (53,248 —
the 512-byte stride continuing: 50,688 → 51,200 → 51,712 → 52,224 →
52,736 → 53,248). The write-set attribution pins exactly those two words
([names-s32-s33.txt](logs/ladder/names-s32-s33.txt): w2 + w6, zero
unattributed, stable across all three frames). It binds
`block25.layer1.layer` (263,168 B — measured at its parameter block's
weights word at heap offset 11,194,880, the record directly following
slot 32's `block25.layer0.layer` 524,288-byte payload), so the
record-ordering walk holds through block25's pair. Its parameter block is
the slot-25 word map exactly: w0 output-of-slot-32 (23,362,560), w1
residual input (23,018,496), w2 output (23,706,624), w3 weights, w4 the
(24, 28) constant, w5 reserved 0, w6 sync out (53,248), w7 sync in (slot
32's record at 52,736), w8 reserved 0 — so `0xFFFFFFFCFFFFFFFC` does
**not** recur in block25's ffwd→proj pair (like slots 25/29, unlike slot
31's w8); slots 34/35 (below) show the qkv→proj pair carries its second
constant in both halves at one value per pair — block24's at
`0xFFFFFFFCFFFFFFFC`, block25's at `0x00000000FFFFFFFC`. The kernel
name and grid (8,3,1) are shared with slot 29 (the `ffwd_proj` handle at
the same launch shape — the same-geometry reuse slots 27/31 and 28/32
established), so slot 33 is a different parameter block and weight record
at an identical launch shape to slot 29.

Slot 34 is the **block25 qkv half** — slot 26's counterpart, the
`qkv→proj` pair's first half. It reads slot 33's output (w0 at
23,706,624 — address-equal to slot 33's w2) and consumes slot 33's sync
record (w5 at 53,248 — address-equal to slot 33's w6, the chained handoff
at the qkv shape's w5 position, like slots 26 and 30) and writes its
output tensor at w1 (344,064 bytes at 24,050,688) plus a sync record at
w6 (53,760 — the 512-byte stride continuing: 51,200 → 51,712 → 52,224 →
52,736 → 53,248 → 53,760). The write-set attribution pins exactly those
two words ([names-s33-s34.txt](logs/ladder/names-s33-s34.txt): w1 + w6,
zero unattributed, stable across all three frames), and the sync write's
changed extent is **192 bytes** — the slot-26 extent returning after slot
30's 256 (predicted from that write-set and confirmed exactly by the
replay's sibling divergence). It binds `block25.layer2.layer` (917,568 B
— measured at its parameter block's weights word at heap offset
11,458,048, the record directly following slot 33's `block25.layer1.layer`
263,168-byte payload — 512-aligned, so no padding intervenes), so the
record-ordering walk holds through block25's pair. The kernel handle is
the shared `qkv_512` handle (the sixteen qkv launches at the two
geometries above), so the grid is resolved at the slot's own launch
position (4,3,4 here — the slot-26 geometry; slot 30's was 4,4,4). Its
parameter block carries two constant words: w3 the (24, 28) geometry
constant and w4
**`0x00000000FFFFFFFC`** — the low dword only, where slot 30's same-shape
w4 is `0xFFFFFFFCFFFFFFFC` and slot 26's is reserved zero. Slot 35's w8
(below) carries the same value, so the pair's story is complete: the
second constant lives in **both halves of a `qkv→proj` pair at one value
per pair** (block24's pair at `0xFFFFFFFCFFFFFFFC`, block25's at
`0x00000000FFFFFFFC`); block23's pair and every `ffwd→proj` half reserve
the word zero. What the dword pattern encodes is not yet established.

Slot 35 is the **block25 qkv-proj pair's second half** — the `proj`
counterpart to slots 27 and 31, same 72-byte block and grid (8,3,1). It
reads slot 34's output (w0 at 24,050,688 — address-equal to slot 34's
w1) and a residual input (w1 at 23,706,624 — the buffer slot 33 wrote
its output to), consumes slot 34's sync record (w7 at 53,760 —
address-equal to slot 34's w6, the chained handoff) and writes its
output tensor at w2 (344,064 bytes at 24,394,752) plus a 96-byte sync
record at w6 (54,272 — the 512-byte stride continuing: 51,200 → 51,712
→ 52,224 → 52,736 → 53,248 → 53,760 → 54,272). The write-set
attribution pins exactly those two words
([names-s34-s35.txt](logs/ladder/names-s34-s35.txt): w2 + w6, zero
unattributed, stable across all three frames). It binds
`block25.layer3.layer` (263,168 B at heap offset 12,376,064 — the record
directly following slot 34's `block25.layer2.layer` 917,568-byte payload
at the alignment boundary below it (448 bytes of padding), the
block23/24 layer2→3 pattern repeating), so the record-ordering walk
holds through block25's pair. Its parameter block is slot 31's word map
with w8 at block25's pair value: w4 the (24, 28) constant and w8
`0x00000000FFFFFFFC` — the same value as its pair partner slot 34's w4,
closing the per-pair constant question above.

Slot 36 is the **block26 ffwd half** — the next `ffwd→proj` pair's first
half. It reads slot 35's output (w0 at 24,394,752 — address-equal to slot
35's w2) and consumes slot 35's sync record (w4 at 54,272 — address-equal
to slot 35's w6, the chained handoff at the ffwd shape's w4 position, like
slots 28 and 32), and writes its output tensor at w1 (344,064 bytes at
24,738,816) plus a 96-byte sync record at w6 (54,784 — the 512-byte
stride continuing: 52,224 → 52,736 → 53,248 → 53,760 → 54,272 → 54,784).
The write-set
attribution pins exactly those two words
([names-s35-s36.txt](logs/ladder/names-s35-s36.txt): w1 + w6, zero
unattributed, stable across all three frames). It binds
`block26.layer0.layer` (524,288 B at heap offset 12,639,232 — the record
directly after `block25.layer3.layer`, the record-ordering walk continuing
into block26). Its parameter block is the slot-28/32 word map exactly (w5
reserved zero — an ffwd half carries no pair second constant). The kernel
handle is the `ffwd_512` family's shared handle; the s35/s36 traces measure
its full enumeration — all fifteen ffwd launches of both transformer
intervals (slots 28/32/36/40/44/48/52 and
100/104/108/112/116/120/124/128) use one handle at **exactly one
geometry** (grid 4,3,2), unlike the qkv family's two — the same-geometry
reuse pattern measured at `proj` (27/31/35) and `ffwd_proj` (29/33) now
extends to three covered `ffwd` launches (28/32/36). The same enumeration
reconciles every documented family in both traces with zero mismatches
(`ffwd_proj_512`: fifteen launches at grid 8,3,1; `proj_512`: fourteen at
grid 8,3,1; `qkv_512`: sixteen at grids 4,3,4/4,4,4), and singles out the
interval-edge variants: the inpview entry pair (24/25), `proj_pool_512` at
slot 55, `final_head_512_wait` at slot 56 and `proj_512_outview_wait` at
slot 131.

Everything each launch needs was resolved from retained evidence — no new
capture semantics were invented:

| Element | Source |
|---|---|
| grid/block/smem | forwarder trace created/launch records, pinned equal across runs and evals |
| parameter block | trace slot dump (56 B / 72 B), pinned equal across evals, runs and the capture sidecar |
| input pool state | the capture after slot n−1 (exactly what slot n reads) |
| reference pool state | the capture after slot n |
| weight bytes | `block23.layer0.layer` (524,288 B) / `block23.layer1.layer` (263,168 B) / `block23.layer2.layer` (917,568 B) / `block23.layer3.layer` (263,168 B) / `block24.layer0.layer` (524,288 B) / `block24.layer1.layer` (263,168 B) / `block24.layer2.layer` (917,568 B) / `block24.layer3.layer` (263,168 B) / `block25.layer0.layer` (524,288 B) / `block25.layer1.layer` (263,168 B) / `block25.layer2.layer` (917,568 B) / `block25.layer3.layer` (263,168 B) / `block26.layer0.layer` (524,288 B) from the DLL's WEIGHTS_HT resource |

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
The slot-30 word map (56-byte block, 7 words) is slot 26's shape with w4
carrying a second pinned constant instead of reserved zero: w0
output-of-slot-29, w1 output (22,674,432), w2 weights, w3 the `(24, 28)`
constant, w4 `0xFFFFFFFCFFFFFFFC`, w5 sync in (slot 29's record at 51,200),
w6 sync out (51,712). The slot-31 word map (72-byte block, 9 words) is slot
27's shape with w8 carrying that same second constant instead of reserved
zero: w0 output-of-slot-30 (22,674,432), w1 residual input (22,330,368), w2
output (23,018,496), w3 weights, w4 the `(24, 28)` constant, w5 reserved 0,
w6 sync out (52,224), w7 sync in (slot 30's record at 51,712), w8
`0xFFFFFFFCFFFFFFFC`. The slot-33 word map (72-byte block, 9 words) is
slot 25's shape exactly (w8 reserved zero again): w0 output-of-slot-32
(23,362,560), w1 residual input (23,018,496), w2 output (23,706,624), w3
weights, w4 the `(24, 28)` constant, w5 reserved 0, w6 sync out (53,248),
w7 sync in (slot 32's record at 52,736), w8 reserved 0. The slot-34 word
map (56-byte block, 7 words) is slot 26's shape with w4 carrying the
pair's second constant: w0 output-of-slot-33 (23,706,624), w1
output (24,050,688), w2 weights, w3 the `(24, 28)` constant, w4
`0x00000000FFFFFFFC`, w5 sync in (slot 33's record at 53,248), w6 sync
out (53,760). The slot-35 word map (72-byte block, 9 words) is slot 31's
shape with w8 carrying block25's pair value: w0 output-of-slot-34
(24,050,688), w1 residual input (23,706,624), w2 output (24,394,752),
w3 weights, w4 the `(24, 28)` constant, w5 reserved 0, w6 sync out
(54,272), w7 sync in (slot 34's record at 53,760), w8
`0x00000000FFFFFFFC` — the same value as its pair partner slot 34's w4.
The slot-36 word map (56-byte block, 7 words) is slot 28's shape exactly
(w5 reserved zero): w0 output-of-slot-35 (24,394,752), w1 output
(24,738,816), w2 weights, w3 the `(24, 28)` constant, w4 sync in (slot
35's record at 54,272), w5 reserved 0, w6 sync out (54,784).

## The weight-heap layout

The 140.9 MiB weight heap is the **WEIGHTS_HT record payloads concatenated at
512-byte alignment**. The layout is pinned by the decoder's live weights
(word 7 of the decoder boundary, [DECODER-BOUNDARY.md](DECODER-BOUNDARY.md))
at `block39.layer0.layer`'s payload, and slot 24's word 2, slot 25's word 3,
slot 26's word 2, slot 27's word 3, slot 28's word 2, slot 29's word 3,
slot 30's word 2, slot 31's word 3, slot 32's word 2, slot 33's word 3,
slot 34's word 2, slot 35's word 3 and slot 36's word 2
all land **exactly at
their record
payload starts** (`block23.layer0.layer` at heap offset 6,733,312;
`block23.layer1.layer` at 7,257,600 — directly after layer0's 524,288-byte
payload; `block23.layer2.layer` at 7,520,768 — directly after layer1;
`block23.layer3.layer` at 8,438,784 — directly after layer2 at the
alignment boundary below; `block24.layer0.layer` at 8,701,952 — directly
after layer3's 263,168-byte payload; `block24.layer1.layer` at 9,226,240 —
directly after layer0's 524,288-byte payload; `block24.layer2.layer` at
9,489,408 — directly after layer1's 263,168-byte payload; `block24.layer3.layer`
at 10,407,424 — directly after layer2 at the same alignment boundary
pattern; `block25.layer0.layer` at 10,670,592 — directly after layer3's
263,168-byte payload; `block25.layer1.layer` at 11,194,880 — directly
after layer0's 524,288-byte payload; `block25.layer2.layer` at 11,458,048
— directly after layer1's 263,168-byte payload; `block25.layer3.layer` at
12,376,064 — directly after layer2 at the same alignment boundary
pattern; `block26.layer0.layer` at 12,639,232 — directly after layer3's
263,168-byte payload, so the record
order is the DLL's declaration order, block24 repeats block23's
layer0-3 shape, and blocks 25 and 26 are repeating that shape again). layer2's
917,568-byte payload is not
512-aligned: the following record starts 448 bytes later at the alignment
boundary (arithmetic from the probe's live DLL walk, which enforces exactly
one record at that boundary), so the tail control's following-record lookup
is alignment-aware. Beyond the anchors, the probe
computes and enforces the structural claim itself (`setup.heap_scan`, retained
in every run's evidence): every heap-pointing parameter word of every scanned
trace lands at a record payload start — **459 of 459 in each scan record,
zero mid-record** (the scans cover the s23–s36 traces; s24, s25, s26, s27,
s28, s29, s30, s31, s32, s33, s34 and s35 are each scanned by two adjacent contracts' setups). The binding's *identity*
(`block26.layer0.layer`, 524,288 B at slot 36) is enforced at runtime and named in
the source contract table; what the evidence retains is the binding's
*content* and *position* — `setup.record_sha256` and `setup.heap_offset` —
plus the enforced equality to the layout walk, so a record rename would not
disturb any run's verdicts.

## Variant finding: the tilesync/chained sync record

All thirteen contracts' fp8 siblings (`..._512_fp8`) are **created but never
launched by the vendor** (pinned: 0 launches in every trace), so they have no
launch contract of their own; each is replayed at the live variant's geometry
as a variant probe. In every case the sibling's **output tensor matches the
reference byte-for-byte** but its **sync record is left at the input state**
(`sync_from_input` true on every sibling entry) while the live variant (tilesync at
slot 24, chained at slots 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35 and 36) writes it. That divergence confinement
is **enforced** (`sibling-divergence`): the differences inside the sync
record must equal the total difference count over the whole pool — which
proves every differing byte lies within the record — and the record itself
must differ in **every byte of its changed extent**. The extent is **per
slot, not a constant**: 96 B at slots 24, 25, 27, 28, 29, 31, 32, 33, 35 and 36,
**192 B at slots 26 and 34**
and **256 B at slot 30** (each predicted from the independent ladder
write-set and confirmed exactly by
the replay's sibling divergence — two sources, one number). The tilesync-only
sync record is now observed at fourteen sites — the decoder's out2, and slots 24,
25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35 and 36 — a family trait of the `*_tilesync_*` / `*_chained_*`
kernels' chain handoff.

## Controls

| Mode | Slot 24 | Slot 25 | Slot 26 | Slot 27 | Slot 28 | Slot 29 | Slot 30 | Slot 31 | Slot 32 | Slot 33 | Slot 34 | Slot 35 | Slot 36 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `execute` ×2 | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** |
| `skip` (no launch) | pool bit-identical to input and non-matching — diff counts **344,085 / 344,104 / 344,110**, exactly the ladder write-set counts | same, diffs **340,602 / 341,160 / 341,167** = the s24→s25 ladder counts | same, diffs **342,905 / 342,801 / 342,841** = the s25→s26 ladder counts | same, diffs **340,726 / 341,250 / 341,150** = the s26→s27 ladder counts | same, diffs **344,027 / 344,077 / 344,069** = the s27→s28 ladder counts | same, diffs **341,165 / 341,559 / 341,526** = the s28→s29 ladder counts | same, diffs **343,293 / 343,253 / 343,275** = the s29→s30 ladder counts | same, diffs **340,970 / 341,345 / 341,358** = the s30→s31 ladder counts | same, diffs **342,117 / 342,432 / 342,376** = the s31→s32 ladder counts | same, diffs **340,754 / 341,134 / 341,144** = the s32→s33 ladder counts | same, diffs **342,020 / 341,782 / 341,773** = the s33→s34 ladder counts | same, diffs **340,617 / 340,872 / 340,885** = the s34→s35 ladder counts | same, diffs **342,288 / 341,759 / 341,735** = the s35→s36 ladder counts |
| `wrong` (weights rotated inwards 1,024 B) | rejects in all entries (319,425–319,537 bytes differ), pool changed from input (dead-launch detection) | rejects (319,545–319,658 bytes differ), pool changed from input | rejects (341,086–342,203 bytes differ), pool changed from input | rejects (342,666–342,849 bytes differ), pool changed from input | rejects (343,654–343,784 bytes differ), pool changed from input | rejects (343,429–343,791 bytes differ), pool changed from input | rejects (341,008–341,728 bytes differ), pool changed from input | rejects (343,233–343,398 bytes differ), pool changed from input | rejects (340,696–340,891 bytes differ), pool changed from input | rejects (342,186–342,308 bytes differ), pool changed from input | rejects (339,481–340,740 bytes differ), pool changed from input | rejects (342,832–343,015 bytes differ), pool changed from input | rejects (340,920–341,551 bytes differ), pool changed from input |
| `tail` (bytes beyond the record replaced 0x5A vs 0xA5) | **output unchanged** — bounded by the 524,288-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 917,568-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 524,288-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 917,568-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 524,288-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 917,568-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 524,288-byte record |

Two cross-tool agreements pin the numbers: the skip diff counts equal the
independent `ladder-report diff` changed-byte counts to the byte, and the
execute entries' change counts equal the summed `ladder-names` segment counts
of the same interval (slots 24→25: 340,506/341,064/341,071 tensor + 96
sync, slots 25→26: 342,713/342,609/342,649 tensor + 192
sync, slots 26→27: 340,630/341,154/341,054 tensor + 96 sync, slots 27→28:
343,931/343,981/343,973 tensor + 96 sync, slots 28→29:
341,069/341,463/341,430 tensor + 96 sync, slots 29→30:
343,037/342,997/343,019 tensor + 256 sync, slots 30→31:
340,874/341,249/341,262 tensor + 96 sync, slots 31→32:
342,021/342,336/342,280 tensor + 96 sync, slots 32→33:
340,658/341,038/341,048 tensor + 96 sync, slots 33→34:
341,828/341,590/341,581 tensor + 192 sync, slots 34→35:
340,521/340,776/340,789 tensor + 96 sync, slots 35→36:
342,192/341,663/341,639 tensor + 96 sync — every interval with a names
file, complete; the s23→s24 attribution predates the names tool and is
stated in the slot-24 section above; the fp8
siblings show the tensor-only counts, exactly the sync extent less).

## Method

`vr/native-slot-replay-probe.py` now carries a **per-slot contract table**
(`CONTRACTS`, `--slot {24..36}`): parameter-block size and word classes
(pool / weights / constant / zero; constant words pinned per word in
`constant_values` — a block can carry more than one, at more than one
value), output and sync word indices and the
sync record's changed extent (`sync_bytes` — measured per slot in the ladder
write-sets, enforced as the sibling's divergence), weight record binding,
and the input/reference capture pair. Grid geometry is resolved at the
slot's own launch position (handles are shared across slots at different
geometries). The replay substitutes our own device addresses at the vendor's
own pool offsets; guards sit around every allocation and the weight buffer is
immutable-checked. Evidence
(`vr/logs/slot-replay/s{24..36}/{a,b,skip,wrong,tail}`) retains digests and
bounded windows only (windows are written before verdicts on any mismatch);
the `pair` command persists its result as `pair.json` beside the first run.

Self-test: **25 corruption controls** (12 verdict, 6 consistency, 5 pair, 1
metadata, 1 grid) each rejected with its pinned label or message, plus **117
known answers** (3 diff, 4 layout, 4 following-record, 1 metadata, 1 grid,
and 8 contract-table shape checks per contract — 104 across the thirteen
contracts, counted per contract rather than per table). The contract table's shape is itself
pinned (slot key equals the table key, class-table length = parameter size,
constant words carry a pinned entry for exactly the constant-class indices
(values enforced against the captured block at setup),
exactly one weights word,
pool-class region words, sync extent within the 512-byte sync stride,
output and sync words distinct, six distinct plan entries — every shape
check mutation-tested at development time against a corrupted contract
entry, each rejection with its pinned message; the corruptions are
development-time in-memory mutations, not retained self-test fixtures).

## Evidence

Runs on 16/10/2026 against the final source of the slot-36 stage (the
contract-36 entry, docstrings and usage text all predate the runs):

- `vr/logs/ladder/{s36,s36b}` — the new slot-36 reference captures
  (validated: cut accounting, pool digests, the slot-aware output canary,
  input digests; `ladder-report.py pair` enforces byte-equal pools with a
  fail-closed exit — its passing report is retained as `pair-s36-s36b.json`,
  pool digests equal across runs). Traces
  `vr/logs/capture/nvapi-trace-ladder-s36{,b}.log` (the slot-36 parameter
  dump is stable across evaluations and identical between the runs; the
  capture sidecar's three records match it byte-for-byte).
- `vr/logs/ladder/diff-s35-s36.json`, `vr/logs/ladder/names-s35-s36.txt` —
  the measured slot-36 write-set (2 segments per frame, zero unattributed:
  342,192/341,663/341,639 tensor + 96 sync).
- `vr/logs/slot-replay/s36/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s
  b6e98939…). Disclosed: the first `execute` run's console log raced the
  probe's evidence-directory creation (`tee` opened first and failed); the
  run itself completed normally (zero enforced failures, `run.json` and the
  six mismatch windows intact) and the console log was written from the
  probe's own summary afterwards — every subsequent run pre-created its
  directory.
- **Additivity regression control:** the contract-36 entry is a new table
  key; a fresh slot-35 `execute` run under the final source
  (`vr/logs/slot-replay/s35/recheck-stage28-final-source`) pairs
  **bit-identical** with the retained `s35/a` (the pairing persisted beside
  the recheck run as `pair.json`, equal `run.json` SHA-256s b0564ee2… — and
  `s35/a`'s retained a-vs-b `pair.json` is untouched), so the addition
  changes no earlier verdict.
- **Metadata reconciliation:** the s35/s36 traces enumerate every
  `cc_split_swin_16h_*` family identically (zero mismatches against the
  documented slot-26 note): `ffwd_512` one shared handle, fifteen launches
  at grid 4,3,2; `ffwd_proj_512` fifteen at 8,3,1; `proj_512` fourteen at
  8,3,1; `qkv_512` sixteen at grids 4,3,4/4,4,4; single-launch variants at
  the interval edges (inpview pair 24/25, `proj_pool_512` at 55,
  `final_head_512_wait` at 56, `proj_512_outview_wait` at 131).
- **Review round:** independent review FAIL → fixed (5 must-fix: a stale
  "109 known answers" Method count (now 117 — the same file's other count
  already read 117), the variant-finding section's live-variant and
  sync-extent enumerations missing slot 36, the weight-bytes table row
  broken across two physical lines (GFM one-row-per-line), and the
  weight-heap-layout section's anchor list, scan coverage and
  binding-identity claims not yet extended to slot 36; 1 advisory: the
  slot-36 stride chain spelled out in full like its neighbours). All fixes
  are `vr/NATIVE-SLOT-REPLAY.md` prose (plus one Controls-table delimiter
  cell, found in post-review verification: the column added at the header
  and rows had not been added at the delimiter row), so the retained runs
  stand unchanged — the probe source is byte-identical to the source the
  five runs and the additivity recheck were taken under. The reviewer
  independently re-verified the contract entry, both s36 captures, all six
  `run.json` files, both `pair.json` files and the traces/names/diff
  artefacts against every numeric claim; self-test re-run green after the
  fixes (117 known answers, 25 controls, plus ladder-report's 19 and
  ladder-names' 14).

Runs on 16/10/2026 against the final source of the slot-35 stage:

- `vr/logs/ladder/{s35,s35b}` — the new slot-35 reference captures
  (validated: cut accounting, pool digests, the slot-aware output canary,
  input digests; `ladder-report.py pair` enforces byte-equal pools with a
  fail-closed exit — its passing report is retained as `pair-s35-s35b.json`,
  pool digests equal across runs). Traces
  `vr/logs/capture/nvapi-trace-ladder-s35{,b}.log` (the slot-35 parameter
  dump is stable across evaluations and identical between the runs).
- `vr/logs/ladder/diff-s34-s35.json`, `vr/logs/ladder/names-s34-s35.txt` —
  the measured slot-35 write-set (2 segments per frame, zero unattributed:
  340,521/340,776/340,789 tensor + 96 sync).
- `vr/logs/slot-replay/s35/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s b0564ee2…),
  taken after the stage's functional probe change (the additive contract-35
  entry, docstrings and usage text). The first `execute` attempt was killed
  by a host command timeout in its tail phase (all six windows written, no
  `run.json`; the partial directory was removed before the clean re-take) and
  the runs were then taken strictly one at a time — disclosed here.
- **Additivity regression control:** the contract-35 entry is a new table
  key; a fresh slot-34 `execute` run under the final source
  (`vr/logs/slot-replay/s34/recheck-stage27-final-source`) pairs
  **bit-identical** with the retained `s34/a` (the pairing persisted beside
  the recheck run as `pair.json`, equal `run.json` SHA-256s 3a7ec4bf… — and
  `s34/a`'s retained a-vs-b `pair.json` is untouched), so the addition
  changes no earlier verdict.
- **Review round:** independent review FAIL → fixed (2 must-fix: a stale
  "101 known answers" self-test count carried over from the eleven-contract
  stage (now 109), and two stale "value varies per pair" claims about the
  second constant in the slot-32/34 narratives — both now state the measured
  finding (one value per pair, in both halves of a `qkv→proj` pair;
  block23's pair and the `ffwd→proj` halves reserve the word); 1 advisory:
  the tail row's slot-24 cell now follows the row convention). The fixes are
  docstrings and prose only (no functional code) but postdate the five runs,
  so a fresh `execute` run was taken under the final source
  (`s35/recheck-stage27-final-source`) and pairs **bit-identical** with the
  retained `s35/a` (persisted `pair.json`, equal `run.json` SHA-256s
  b0564ee2… — `s35/a`'s retained a-vs-b `pair.json` untouched), so all
  retained evidence stands under the final source.

Runs on 16/10/2026 against the final source of the slot-34 stage:

- `vr/logs/ladder/{s34,s34b}` — the new slot-34 reference captures
  (validated: cut accounting, pool digests, the slot-aware output canary,
  input digests; `ladder-report.py pair` enforces byte-equal pools with a
  fail-closed exit — its passing report is retained as `pair-s34-s34b.json`,
  pool digests equal across runs). Traces
  `vr/logs/capture/nvapi-trace-ladder-s34{,b}.log` (the slot-34 parameter
  block is byte-identical across all three evaluations in both traces).
- `vr/logs/ladder/diff-s33-s34.json`, `vr/logs/ladder/names-s33-s34.txt` —
  the measured slot-34 write-set (2 segments per frame, zero unattributed:
  341,828/341,590/341,581 tensor + 192 sync).
- `vr/logs/slot-replay/s34/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s
  3a7ec4bf…), all taken after the stage's functional probe change (the
  additive contract-34 entry and usage text). The first `execute` attempt
  was killed by a host command timeout mid-run (it had written two fp8
  windows, no `run.json`; those partial windows were removed before the
  clean re-take) and the five retained runs were then taken strictly one
  at a time — disclosed here.
- **Additivity regression control:** the contract-34 entry is a new table
  key; a fresh slot-33 `execute` run under the final source
  (`vr/logs/slot-replay/s33/recheck-stage26-final-source`) pairs
  **bit-identical** with the retained `s33/a` (the pairing persisted
  beside the recheck run as `pair.json`, equal `run.json` SHA-256s
  e6ac5c6f…), so the addition changes no earlier verdict.
- **Review round:** independent review FAIL → fixed (3 must-fix: the
  slot-26 and slot-34 handle-sharing enumerations were inconsistent with
  each other and incomplete — both now state the measured full set (the
  shared `qkv_512` handle serves all sixteen qkv launches of both
  transformer intervals at exactly two geometries), and the wrong
  "observed three times" count is gone; this stage's first additivity
  pairing had been run with `s33/a` as its first path and so overwrote the
  retained s33 a-vs-b `pair.json` — the record is restored by re-pairing
  `s33/a` with `s33/b` (bit-identical, equal `run.json` SHA-256s) and the
  additivity pairing re-taken with the recheck directory as first path,
  persisted beside it; the pre-review recheck run is retained as
  `s33/recheck-stage26-superseded-pre-reviewfix`; 1 advisory: an ambiguous
  "there" in the slot-26 geometry clause). The review-round fixes are
  docstrings and prose only (no functional code) but postdate the five
  runs, so fresh `execute` runs were taken under the final source for both
  slots — `s34/recheck-stage26-final-source` pairing **bit-identical** with
  `s34/a` (3a7ec4bf…) and `s33/recheck-stage26-final-source` with `s33/a`
  (e6ac5c6f…), pairings persisted beside the recheck runs — so all
  retained evidence stands under the final source.

Runs on 16/10/2026 against the final source of the slot-33 stage:

- `vr/logs/ladder/{s33,s33b}` — the new slot-33 reference captures
  (validated: cut accounting, pool digests, the slot-aware output canary,
  input digests; `ladder-report.py pair` enforces byte-equal pools with a
  fail-closed exit — its passing report is retained as `pair-s33-s33b.json`,
  pool digests equal across runs). Traces
  `vr/logs/capture/nvapi-trace-ladder-s33{,b}.log`.
- `vr/logs/ladder/diff-s32-s33.json`, `vr/logs/ladder/names-s32-s33.txt` —
  the measured slot-33 write-set (2 segments per frame, zero unattributed:
  340,658/341,038/341,048 tensor + 96 sync).
- `vr/logs/slot-replay/s33/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s), taken
  after the stage's functional probe change (the additive contract-33
  entry and usage text). One `wrong` attempt was killed by a host command
  timeout mid-run when the three controls were first chained in one host
  command (its directory was empty — no `run.json`, no windows) and was
  cleanly re-taken alone; disclosed here.
- **Review round:** independent review FAIL → fixed (2 must-fix: the
  `wrong`-row diff range now spans all six entries (342,186–342,308, the
  fp8 siblings included) as the row convention demands, and a stale
  nine-slot sync-extent enumeration in the probe's contract-schema comment;
  1 advisory: the stage-28-era "seven shape checks" wording scoped to the
  then-seven). The probe docstring/comment fixes postdate the five runs, so
  a fresh `execute` run was taken under the final source
  (`s33/recheck-stage25-final-source`) and pairs **bit-identical** with the
  retained `s33/a` (persisted `pair.json`, equal `run.json` SHA-256s).
- **Additivity regression control:** the contract-33 entry is a new table
  key; a fresh slot-32 `execute` run under the final source
  (`vr/logs/slot-replay/s32/recheck-stage25-final-source`) pairs
  **bit-identical** with the retained `s32/a` (persisted `pair.json`, equal
  `run.json` SHA-256s), so the addition changes no earlier verdict.

Runs on 16/10/2026 against the final source of the slot-32 stage:

- `vr/logs/ladder/{s32,s32b}` — the new slot-32 reference captures
  (validated: cut accounting, pool digests, the slot-aware output canary,
  input digests; `ladder-report.py pair` enforces byte-equal pools with a
  fail-closed exit — its passing report is retained as `pair-s32-s32b.json`,
  pool digests equal across runs). Traces
  `vr/logs/capture/nvapi-trace-ladder-s32{,b}.log`.
- `vr/logs/ladder/diff-s31-s32.json`, `vr/logs/ladder/names-s31-s32.txt` —
  the measured slot-32 write-set (2 segments per frame, zero unattributed:
  342,021/342,336/342,280 tensor + 96 sync).
- `vr/logs/slot-replay/s32/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s), taken
  after the stage's functional probe change (the additive contract-32
  entry and usage text). One earlier `wrong` attempt was killed by a host
  command timeout after its windows were written but before `run.json`
  (it raced a concurrent `tail` run on the one GPU) and is
  retained as `wrong-superseded-killed-by-host-timeout` (mismatch windows
  only, no `run.json`); the completed run was re-taken cleanly and alone.
- **Source neutrality:** the docstring fixes of the review round (slot-32
  bullet addresses, scope wording — no functional code) postdate the five
  runs, so a fresh `execute` run was taken under the final source
  (`s32/recheck-stage24-final-source`) and pairs **bit-identical** with the
  retained `s32/a` (persisted `pair.json`, equal `run.json` SHA-256s), so
  all retained evidence stands under the final source.
- **Additivity regression control:** the contract-32 entry is a new table
  key; a fresh slot-31 `execute` run under the final source
  (`vr/logs/slot-replay/s31/recheck-stage24-final-source`) pairs
  **bit-identical** with the retained `s31/a` (persisted `pair.json`, equal
  `run.json` SHA-256s), so the addition changes no earlier verdict. The
  pre-docfix recheck is retained as `recheck-stage24-superseded-pre-docfix`.

Runs on 16/10/2026 against the final source of the slot-31 stage:

- `vr/logs/ladder/{s31,s31b}` — the new slot-31 reference captures
  (validated: cut accounting, pool digests, the slot-aware output canary,
  input digests; `ladder-report.py pair` enforces byte-equal pools with a
  fail-closed exit — its passing report is retained as `pair-s31-s31b.json`,
  pool digests equal across runs). Traces
  `vr/logs/capture/nvapi-trace-ladder-s31{,b}.log`.
- `vr/logs/ladder/diff-s30-s31.json`, `vr/logs/ladder/names-s30-s31.txt` —
  the measured slot-31 write-set (2 segments per frame, zero unattributed:
  340,874/341,249/341,262 tensor + 96 sync).
- `vr/logs/slot-replay/s31/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s), all taken
  after the stage's probe changes (the additive contract-31 entry, docstring
  and usage text).
- **Additivity regression control:** the contract-31 entry is a new table
  key; a fresh slot-30 `execute` run under the final source
  (`vr/logs/slot-replay/s30/recheck-stage23-final-source`) pairs
  **bit-identical** with the retained `s30/a` (persisted `pair.json`, equal
  `run.json` SHA-256s), so the addition changes no earlier verdict.

Runs on 16/10/2026 against the final source of the slot-30 stage:

- `vr/logs/ladder/{s30,s30b}` — the new slot-30 reference captures
  (validated: cut accounting, pool digests, the slot-aware output canary,
  input digests; `ladder-report.py pair` enforces byte-equal pools with a
  fail-closed exit — its passing report is retained as `pair-s30-s30b.json`,
  pool digests equal across runs). Traces
  `vr/logs/capture/nvapi-trace-ladder-s30{,b}.log`.
- `vr/logs/ladder/diff-s29-s30.json`, `vr/logs/ladder/names-s29-s30.txt` —
  the measured slot-30 write-set (2 segments per frame, zero unattributed:
  343,037/342,997/343,019 tensor + 256 sync).
- `vr/logs/slot-replay/s30/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s). One
  intermediate attempt at run `b` hung in post-run teardown after completing
  all six entries (host-side command timeout; no `run.json`) and its partial
  windows were overwritten by the clean re-take — disclosed here; the
  transient hang was not reproducible in the re-take.
- **Schema regression control:** the stage's probe changes (the per-word
  `constant_values` schema replacing the single `constant_value`, the
  contract-30 entry, docstrings and usage text) were validated against the
  retained slot-29 evidence: a fresh slot-29 `execute` run under the final
  source (`vr/logs/slot-replay/s29/recheck-stage22-final-source`) pairs
  **bit-identical** with the retained `s29/a` (persisted `pair.json`, equal
  `run.json` SHA-256s), so the schema generalisation changes no verdict.

Runs on 16/10/2026 against the final source of the slot-29 stage:

- `vr/logs/ladder/{s29,s29b}` — the new slot-29 reference captures
  (validated: cut accounting, pool digests, the slot-aware output canary,
  input digests; `ladder-report.py pair` enforces byte-equal pools with a
  fail-closed exit — its passing report is retained as `pair-s29-s29b.json`,
  pool digests equal across runs). Traces
  `vr/logs/capture/nvapi-trace-ladder-s29{,b}.log`.
- `vr/logs/ladder/diff-s28-s29.json`, `vr/logs/ladder/names-s28-s29.txt` —
  the measured slot-29 write-set (2 segments per frame, zero unattributed:
  341,069/341,463/341,430 tensor + 96 sync).
- `vr/logs/slot-replay/s29/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s). The
  review-driven fixes (two stale documentation counts and three provenance
  clarifications; no functional code) touch only docstrings, comments and
  usage text — the five runs were nevertheless re-taken against the final
  source and pair **bit-identical** with the pre-fix runs (retained as
  `*-superseded-pre-reviewfix`, `run.json` SHA-256s and window digests
  equal — byte-equality of every retained artefact), so the verdicts do not
  depend on the fixes.

Runs on 15/10/2026 against the final sources of the slot-24–28 stages:

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
  `pair.json`), so the verdicts do not depend on the fix (the then-seven
  shape checks; the eighth, the constant-word pin, arrived with the
  stage-22 `constant_values` schema).
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
  the stage-20 source (`s{24,25,26,27}/recheck-stage20-final-source`) and
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
- `python3 vr/native-slot-replay-probe.py self-test` (117 known answers), `python3
  vr/ladder-report.py self-test` (19 controls) and `python3
  vr/ladder-names.py self-test` (14 controls) all pass.

## Scope and limitations

- The claim covers **thirteen launches of the 156** (slots 24–36; six distinct
  kernel names, one shared across two geometries (26/30/34) and three reused at
  the same geometry (27/31/35, 28/32/36, 29/33)), one resolution
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
- The write-set does not separate reads from writes: slot 24's w0 read,
  slot 25's w0, w1 and w7
  reads (previous output, residual, previous sync), slot 26's w0 and w5
  reads, slot 27's w0, w1 and w7 reads (output, residual, previous sync),
  slot 30's w0 and w5 reads, slot 31's w0, w1 and w7 reads, slot
  32's w0 and w4 reads, slot 33's w0, w1 and w7 reads (output, residual,
  previous sync), slot 34's w0 and w5 reads (output of slot 33, previous
  sync), slot 35's w0, w1 and w7 reads (output, residual, previous sync),
  slot 36's w0 and w4 reads (output of slot 35, previous sync)
  and slots 28/29's w0 and (w4/w7) reads — slot 29
  additionally its residual w1 — are inferred from the attribution (those words never change) and the
  word map, not from a measured read-set.
- The sync extents (96/96/192/96/96/96/256/96/96/96/192/96/96 B for slots 24–36) are the records'
  **changed** extents under
  this protocol; records whose unchanged bytes extend further are not
  excluded.
- This stage uses stored learned weight bytes through thirteen extracted records
  (bounded model-weight computation, [NATIVE-FP8.md](NATIVE-FP8.md)
  precedent) and is **not** a neural-rendered image. Proprietary
  reuse/publication rights remain unresolved; evidence retains digests and
  bounded 8 KiB windows (written only on mismatch) and stays local.

## Next

Repeat the replay for the remaining named kernels of
[LADDER-NAMES.md](LADDER-NAMES.md) — the encoder (slots 1–23), the
transformer interval's further `qkv`/`proj`/`ffwd` pairs (37–56), the ViT
stage (57–98), transformer-2 (100–131) and the decoder pyramid (132–153);
slot 99 (the decoder) is already covered by
[NATIVE-VENDOR-COMPARE.md](NATIVE-VENDOR-COMPARE.md) and slot 154 (the final
blend) writes only outside the working pool. Each remaining kernel needs a
reference capture at its
own cut point (`run-feature-probe.sh ladder sN N`) plus one contract entry,
and occasionally a small probe generalisation (slot 26 needed per-slot grid
resolution and an alignment-aware record lookup). Slots 24–36 now cover
three complete `ffwd→proj` pairs (block23, block24 and block25) and three
complete `qkv→proj`
pairs (block23: slots 26/27, block24: 30/31, block25: 34/35), plus
block26's ffwd half (slot 36), so
the pair pattern, the chained sync handoff (with its
per-slot word position: w5 at slots 26/30/34, w7 at slots 25/27/29/31/33/35, w4 at
slots 28/32/36)
and the record-ordering pattern are established: each block repeats
block23's layer0-3 record shape in declaration order, so each remaining
slot's weight
record is predictable from its parameter block's weights word plus the
layout walk — and slots 29's, 30's, 31's, 32's, 33's, 34's, 35's and 36's
bindings measured exactly those
predictions (`block24.layer1.layer` at heap offset 9,226,240,
`block24.layer2.layer` at 9,489,408, `block24.layer3.layer` at 10,407,424,
`block25.layer0.layer` at 10,670,592, `block25.layer1.layer` at 11,194,880,
`block25.layer2.layer` at 11,458,048, `block25.layer3.layer` at
12,376,064, `block26.layer0.layer` at 12,639,232). The qkv→proj pair's
second constant is now measured in both
halves of two pairs (block24 at `0xFFFFFFFCFFFFFFFC`, block25 at
`0x00000000FFFFFFFC`) and absent from block23's pair and every `ffwd→proj`
half — one value per pair, meaning not yet established.
The natural next
candidate is slot 37
(`cc_split_swin_16h_ffwd_proj_512_chained_fp8` — block26's `ffwd→proj`
pair's second half): its parameter block is already
retained in the s36 traces' dropped-launch records (identical across all
three evaluations in both runs), its weights word already lands on
the layout walk's next record start (`block26.layer1.layer`, 263,168 B at
heap offset 13,163,520) and w5 is reserved zero (the `ffwd→proj` half
carries no pair second constant). One measured oddity to pin at that
stage: its w2 output sits at 25,426,944 — two tensor strides past its w0
(24,738,816), where slots 25/29/33's w2 sat one stride past their w0; the
intervening region (25,082,880) is the encoder-residual buffer the names
tool tracks as slot 23's w9 / slot 24's w0. Meaning not yet established.
The slot-37 stage
will pin and replay it.
The
temporal/history state (`prev_output`, counter) then connects the kernels
into a full native evaluation for the native-vs-vendor comparison of the
whole pipeline.
