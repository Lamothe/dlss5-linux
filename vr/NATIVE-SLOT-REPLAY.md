# Native pre-decoder kernel replay (slots 24-52) — 15-18/10/2026

## Result

**Twenty-nine pre-decoder kernel launches (slots 24–52; six distinct kernel
names — slots 26/30/34/38/42/46/50 share one kernel across two geometries, slots
27/31/35/39/43/47/51, 28/32/36/40/44/48/52 and 29/33/37/41/45/49 reuse kernels at the same geometry) of the
vendor's 156-launch sequence now replay
natively on Linux and reproduce the vendor's own captured state
byte-for-byte.** Each kernel's launch is rebuilt from its captured parameter
block and executed through the **native Linux CUDA driver** on the RTX 4070
Ti SUPER — no Windows code, DirectX or Proton — and the native pool state
after the launch equals the vendor's ladder capture at the matching cut point
([LADDER.md](LADDER.md)) over the **whole 62,684,160-byte pool: 0 differing
bytes on all three frames, in both execute runs, for all twenty-nine slots**.

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
| 37 | `cc_split_swin_16h_ffwd_proj_512_chained_fp8` | reads output-of-slot-36 w0 + residual w1 + previous sync w7; writes output w2 (344,064 B) + sync w6 (96 B) | s36→s37 | (8,3,1) |
| 38 | `cc_split_swin_16h_qkv_512_chained_fp8` | reads output-of-slot-37 w0 + previous sync w5; writes output w1 (344,064 B) + sync w6 (256 B) | s37→s38 | (4,4,4) |
| 39 | `cc_split_swin_16h_proj_512_chained_fp8` | reads output-of-slot-38 w0 + residual w1 + previous sync w7; writes output w2 (344,064 B) + sync w6 (96 B) | s38→s39 | (8,3,1) |
| 40 | `cc_split_swin_16h_ffwd_512_chained_fp8` | reads output-of-slot-39 w0 + previous sync w4; writes output w1 (344,064 B) + sync w6 (96 B) | s39→s40 | (4,3,2) |
| 41 | `cc_split_swin_16h_ffwd_proj_512_chained_fp8` | reads output-of-slot-40 w0 + residual w1 + previous sync w7; writes output w2 (344,064 B) + sync w6 (96 B) | s40→s41 | (8,3,1) |
| 42 | `cc_split_swin_16h_qkv_512_chained_fp8` | reads output-of-slot-41 w0 + previous sync w5; writes output w1 (344,064 B) + sync w6 (192 B) | s41→s42 | (4,3,4) |
| 43 | `cc_split_swin_16h_proj_512_chained_fp8` | reads output-of-slot-42 w0 + residual w1 + previous sync w7; writes output w2 (344,064 B) + sync w6 (96 B) | s42→s43 | (8,3,1) |
| 44 | `cc_split_swin_16h_ffwd_512_chained_fp8` | reads output-of-slot-43 w0 + previous sync w4; writes output w1 (344,064 B) + sync w6 (96 B) | s43→s44 | (4,3,2) |
| 45 | `cc_split_swin_16h_ffwd_proj_512_chained_fp8` | reads output-of-slot-44 w0 + residual w1 + previous sync w7; writes output w2 (344,064 B) + sync w6 (96 B) | s44→s45 | (8,3,1) |
| 46 | `cc_split_swin_16h_qkv_512_chained_fp8` | reads output-of-slot-45 w0 + previous sync w5; writes output w1 (344,064 B) + sync w6 (256 B) | s45→s46 | (4,4,4) |
| 47 | `cc_split_swin_16h_proj_512_chained_fp8` | reads output-of-slot-46 w0 + residual w1 + previous sync w7; writes output w2 (344,064 B) + sync w6 (96 B) | s46→s47 | (8,3,1) |
| 48 | `cc_split_swin_16h_ffwd_512_chained_fp8` | reads output-of-slot-47 w0 + previous sync w4; writes output w1 (344,064 B) + sync w6 (96 B) | s47→s48 | (4,3,2) |
| 49 | `cc_split_swin_16h_ffwd_proj_512_chained_fp8` | reads output-of-slot-48 w0 + residual w1 + previous sync w7; writes output w2 (344,064 B) + sync w6 (96 B) | s48→s49 | (8,3,1) |
| 50 | `cc_split_swin_16h_qkv_512_chained_fp8` | reads output-of-slot-49 w0 + previous sync w5; writes output w1 (344,064 B) + sync w6 (192 B) | s49→s50 | (4,3,4) |
| 51 | `cc_split_swin_16h_proj_512_chained_fp8` | reads output-of-slot-50 w0 + residual w1 + previous sync w7; writes output w2 (344,064 B) + sync w6 (96 B) | s50→s51 | (8,3,1) |
| 52 | `cc_split_swin_16h_ffwd_512_chained_fp8` | reads output-of-slot-51 w0 + previous sync w4; writes output w1 (344,064 B) + sync w6 (96 B) | s51→s52 | (4,3,2) |

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
reuse pattern measured at `proj` (27/31/35/39/43/47/51) and `ffwd_proj` (29/33/37/41/45/49) now
extends to seven covered `ffwd` launches (28/32/36/40/44/48/52). The same enumeration
reconciles every documented family in both traces with zero mismatches
(`ffwd_proj_512`: fifteen launches at grid 8,3,1; `proj_512`: fourteen at
grid 8,3,1; `qkv_512`: sixteen at grids 4,3,4/4,4,4), and singles out the
interval-edge variants: the inpview entry pair (24/25), `proj_pool_512` at
slot 55, `final_head_512_wait` at slot 56 and `proj_512_outview_wait` at
slot 131.

Slot 37 is the **block26 ffwd-proj pair's second half** — the `ffwd_proj`
counterpart to slots 25, 29 and 33 (same 72-byte block and grid 8,3,1),
completing block26's `ffwd→proj` pair. It reads slot 36's output (w0 at
24,738,816 — address-equal to slot 36's w1) and a residual input (w1 at
24,394,752 — the buffer slot 35 wrote its output to), consumes slot 36's
sync record (w7 at 54,784 — address-equal to slot 36's w6, the chained
handoff) and writes its output tensor at w2 (344,064 bytes at 25,426,944)
plus a 96-byte sync record at w6 (55,296 — the 512-byte stride continuing:
52,736 → 53,248 → 53,760 → 54,272 → 54,784 → 55,296). The write-set
attribution pins exactly those two words
([names-s36-s37.txt](logs/ladder/names-s36-s37.txt): w2 + w6, zero
unattributed, stable across all three frames). It binds
`block26.layer1.layer` (263,168 B at heap offset 13,163,520 — the record
directly after `block26.layer0.layer`'s 524,288-byte payload, 512-aligned
so no padding intervenes), so the record-ordering walk holds through
block26's pair. Its parameter block is the slot-25/29/33 word map exactly:
w0 output-of-slot-36 (24,738,816), w1 residual input (24,394,752), w2
output (25,426,944), w3 weights, w4 the `(24, 28)` constant, w5 reserved 0,
w6 sync out (55,296), w7 sync in (slot 36's record at 54,784), w8 reserved
0 — so the pair second constant stays absent from block26's `ffwd→proj`
pair (like slots 25/29/33); slot 38's retained block shows block26's
`qkv→proj` pair does carry it — at a third value (`0xFFFFFFFC00000000`,
the high-dword-only form). The one measured oddity is its
output spacing: w2 sits **two** tensor strides past its w0 (25,426,944 vs
24,738,816), where slots 25/29/33's w2 sat one stride past their w0; the
intervening region (25,082,880) is the encoder-residual buffer the names
tool tracks as slot 23's w9 / slot 24's w0. What that spacing encodes is
not yet established; the replay reproduces the captured placement exactly
and the output tensor is slot 38's w0.

Slot 38 is the **block26 qkv half** — the `qkv→proj` pair's first half at
the shared handle's second geometry (grid 4,4,4, the slot-30/34
counterpart). It reads slot 37's output (w0 at 25,426,944 — address-equal
to slot 37's w2) and consumes slot 37's sync record (w5 at 55,296 —
address-equal to slot 37's w6, the chained handoff) and writes its output
tensor at w1 (344,064 bytes at 25,771,008) plus a sync record at w6
(55,808 — the 512-byte stride continuing: 54,784 → 55,296 → 55,808). The
write-set attribution pins exactly those two words
([names-s37-s38.txt](logs/ladder/names-s37-s38.txt): w1 + w6, zero
unattributed, stable across all three frames), and the sync write's changed
extent is **256 bytes** — slot 30's extent recurring at the second
geometry (slots 26/34 at the first geometry carry 192). It binds
`block26.layer2.layer` (917,568 B — the record directly after
`block26.layer1.layer`, measured at its parameter block's weights word at
heap offset 13,426,688). Its parameter block is the slot-30/34 word map
exactly, with w4 carrying block26's pair second constant at
**`0xFFFFFFFC00000000`** — the high-dword-only form, the third geometry of
the marker dword after block24's double (`0xFFFFFFFCFFFFFFFC`) and
block25's low-dword-only (`0x00000000FFFFFFFC`) forms. The stage-29
prediction (from the s37 traces' retained dropped-launch record) is
confirmed exactly by the s38 capture's own parameter block; the s38
traces' retained block already showed slot 39's w8 carrying the same value
(the pair sharing now observed at all three measured pairs), re-confirmed
from the s39 capture's own block.

Slot 39 is the **block26 proj half** — the `qkv→proj` pair's second half at
grid (8,3,1), block (32,8,1) (the slot-27/31/35 counterpart), completing
block26's `qkv→proj` pair and block26's layer0-3 record shape. It reads slot
38's output (w0 at 25,771,008 — address-equal to slot 38's w1) and a
residual input (w1 at 25,426,944 — the buffer slot 37 wrote its output to),
consumes slot 38's sync record (w7 at 55,808 — address-equal to slot 38's
w6, the chained handoff) and writes its output tensor at w2 (344,064 bytes
at 26,115,072) plus a 96-byte sync record at w6 (56,320 — the 512-byte
stride continuing: 55,296 → 55,808 → 56,320). The write-set attribution pins
exactly those two words
([names-s38-s39.txt](logs/ladder/names-s38-s39.txt): w2 + w6, zero
unattributed, stable across all three frames). It binds
`block26.layer3.layer` (263,168 B — the record directly after
`block26.layer2.layer`'s 917,568-byte payload at the 448-byte alignment
boundary, completing the record-ordering walk through block26). Its
parameter block is the slot-27/31/35 word map exactly, with w8 carrying
block26's pair second constant at **`0xFFFFFFFC00000000`** — identical to
its pair partner slot 38's w4 (the high-dword-only form), so the pair's
second constant is now seen in both halves of all three measured pairs
(block24, block25, block26).

Everything each launch needs was resolved from retained evidence — no new
capture semantics were invented:

| Element | Source |
|---|---|
| grid/block/smem | forwarder trace created/launch records, pinned equal across runs and evals |
| parameter block | trace slot dump (56 B / 72 B), pinned equal across evals, runs and the capture sidecar |
| input pool state | the capture after slot n−1 (exactly what slot n reads) |
| reference pool state | the capture after slot n |
| weight bytes | `block23.layer0.layer` (524,288 B) / `block23.layer1.layer` (263,168 B) / `block23.layer2.layer` (917,568 B) / `block23.layer3.layer` (263,168 B) / `block24.layer0.layer` (524,288 B) / `block24.layer1.layer` (263,168 B) / `block24.layer2.layer` (917,568 B) / `block24.layer3.layer` (263,168 B) / `block25.layer0.layer` (524,288 B) / `block25.layer1.layer` (263,168 B) / `block25.layer2.layer` (917,568 B) / `block25.layer3.layer` (263,168 B) / `block26.layer0.layer` (524,288 B) / `block26.layer1.layer` (263,168 B) / `block26.layer2.layer` (917,568 B) / `block26.layer3.layer` (263,168 B) / `block27.layer0.layer` (524,288 B) / `block27.layer1.layer` (263,168 B) / `block27.layer2.layer` (917,568 B) / `block27.layer3.layer` (263,168 B) / `block28.layer0.layer` (524,288 B) / `block28.layer1.layer` (263,168 B) / `block28.layer2.layer` (917,568 B) / `block28.layer3.layer` (263,168 B) / `block29.layer0.layer` (524,288 B) / `block29.layer1.layer` (263,168 B) / `block29.layer2.layer` (917,568 B) / `block29.layer3.layer` (263,168 B) / `block3.layer0.layer` (20,672 B) / `block30.layer0.layer` (524,288 B) from the DLL's WEIGHTS_HT resource |

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
35's record at 54,272), w5 reserved 0, w6 sync out (54,784). The slot-37
word map (72-byte block, 9 words) is slot 25's shape exactly (w5/w8
reserved zero): w0 output-of-slot-36 (24,738,816), w1 residual input
(24,394,752), w2 output (25,426,944), w3 weights, w4 the `(24, 28)`
constant, w5 reserved 0, w6 sync out (55,296), w7 sync in (slot 36's
record at 54,784), w8 reserved 0. The slot-38 word map (56-byte block, 7
words) is slot 30's shape exactly (the qkv block's w4 second constant): w0
output-of-slot-37 (25,426,944), w1 output (25,771,008), w2 weights, w3
the `(24, 28)` constant, w4 `0xFFFFFFFC00000000` — block26's pair value at
the high dword only — w5 sync in (slot 37's record at 55,296), w6 sync out
(55,808). The slot-39 word map (72-byte block, 9 words) is slot 35's shape
with w8 carrying block26's pair value: w0 output-of-slot-38 (25,771,008),
w1 residual input (25,426,944), w2 output (26,115,072), w3 weights, w4 the
`(24, 28)` constant, w5 reserved 0, w6 sync out (56,320), w7 sync in (slot
38's record at 55,808), w8 `0xFFFFFFFC00000000` — the same value as its
pair partner slot 38's w4. The slot-40 word map (56-byte block, 7 words) is
slot 28's shape exactly: w0 output-of-slot-39 (26,115,072), w1 output
(26,459,136), w2 weights, w3 the `(24, 28)` constant, w4 sync in (slot 39's
record at 56,320), w5 reserved 0, w6 sync out (56,832). The slot-41 word map
(72-byte block, 9 words) is slot 25's shape exactly (w5/w8 reserved zero —
the ffwd-proj half carries no pair second constant): w0 output-of-slot-40
(26,459,136), w1 residual input (26,115,072), w2 output (30,833,664), w3
weights, w4 the `(24, 28)` constant, w5 reserved 0, w6 sync out (57,344), w7
sync in (slot 40's record at 56,832), w8 reserved 0. The slot-42 word map
(56-byte block, 7 words) is slot 26's shape exactly (w4 reserved zero —
block27's `qkv` half carries no second constant; its `proj` half's w8 is
measured zero in the s43 capture): w0 output-of-slot-41
(30,833,664), w1 output (31,177,728), w2 weights, w3 the `(24, 28)`
constant, w4 reserved 0, w5 sync in (slot 41's record at 57,344), w6 sync
out (57,856). The slot-43 word map
(72-byte block, 9 words) is slot 25's shape exactly (w5/w8 reserved zero —
block27's `qkv`-proj pair carries no second constant): w0
output-of-slot-42 (31,177,728), w1 residual input (30,833,664), w2 output
(31,521,792), w3 weights, w4 the `(24, 28)` constant, w5 reserved 0, w6
sync out (58,368), w7 sync in (slot 42's record at 57,856), w8 reserved 0.
The slot-44 word map (56-byte block, 7 words) is slot 28's shape exactly:
w0 output-of-slot-43 (31,521,792), w1 output (31,865,856), w2 weights, w3
the `(24, 28)` constant, w4 sync in (slot 43's record at 58,368), w5
reserved 0, w6 sync out (58,880). The slot-45 word map (72-byte block, 9
words) is slot 25's shape exactly (w5/w8 reserved zero — block28's
`ffwd`-proj pair carries no second constant): w0 output-of-slot-44
(31,865,856), w1 residual input (31,521,792), w2 output (32,209,920), w3
weights, w4 the `(24, 28)` constant, w5 reserved 0, w6 sync out (59,392),
w7 sync in (slot 44's record at 58,880), w8 reserved 0. The slot-46 word
map (56-byte block, 7 words) is slot 30's shape exactly (the qkv block's w4
second constant): w0 output-of-slot-45 (32,209,920), w1 output
(32,553,984), w2 weights, w3 the `(24, 28)` constant, w4
`0xFFFFFFFCFFFFFFFC` — block28's pair value at the double form — w5 sync
in (slot 45's record at 59,392), w6 sync out (59,904). The slot-47 word
map (72-byte block, 9 words) is slot 27's shape exactly (the proj block,
its w8 carrying a second constant where slot 27 reserves zero): w0
output-of-slot-46 (32,553,984), w1 residual input (32,209,920), w2 output
(32,898,048), w3 weights, w4 the `(24, 28)` constant, w5 reserved 0, w6
sync out (60,416), w7 sync in (slot 46's record at 59,904), w8
`0xFFFFFFFCFFFFFFFC` — block28's pair value at the double form. The slot-48
word map (56-byte block, 7 words) is slot 28's shape exactly (w5 reserved
zero — every `ffwd` half reserves the word): w0 output-of-slot-47
(32,898,048), w1 output (33,242,112), w2 weights, w3 the `(24, 28)`
constant, w4 sync in (slot 47's record at 60,416), w5 reserved 0, w6 sync
out (60,928). The slot-49 word map (72-byte block, 9 words) is slot 25's
shape exactly (w5/w8 reserved zero — block29's `ffwd`-proj pair carries no
second constant): w0 output-of-slot-48 (33,242,112), w1 residual input
(32,898,048), w2 output (33,586,176), w3 weights, w4 the `(24, 28)`
constant, w5 reserved 0, w6 sync out (61,440), w7 sync in (slot 48's
record at 60,928), w8 reserved 0. The slot-50 word map (56-byte block, 7
words) is the qkv shape with w4 a second constant (the slot-30/34/38/46
form; slots 26/42 reserve w4 as zero): w0 output-of-slot-49 (33,586,176),
w1 output (33,930,240), w2 weights, w3 the `(24, 28)` constant, w4 the
pair's second constant `0x00000000FFFFFFFC` (the low-dword form), w5 sync
in (slot 49's record at 61,440), w6 sync out (61,952). The slot-51 word map
(72-byte block, 9 words) is slot 27's shape with w8 carrying block29's pair
value (the slot-31/35/39/47 carrying form; slots 27/43 reserve w8 as zero):
w0 output-of-slot-50 (33,930,240), w1 residual input (33,586,176), w2
output (34,274,304), w3 weights, w4 the `(24, 28)` constant, w5 reserved 0,
w6 sync out (62,464), w7 sync in (slot 50's record at 61,952), w8 the
pair's second constant `0x00000000FFFFFFFC` (the low-dword form — the same
value as its pair partner slot 50's w4). The slot-52 word map (56-byte block,
7 words) is slot 28's shape exactly (w5 reserved zero — every `ffwd` half
reserves the word, the `ffwd→proj` pair carrying no second constant): w0
output-of-slot-51 (34,274,304), w1 output (34,618,368), w2 weights, w3 the
`(24, 28)` constant, w4 sync in (slot 51's record at 62,464), w5 reserved 0,
w6 sync out (62,976).

## The weight-heap layout

The 140.9 MiB weight heap is the **WEIGHTS_HT record payloads concatenated at
512-byte alignment**. The layout is pinned by the decoder's live weights
(word 7 of the decoder boundary, [DECODER-BOUNDARY.md](DECODER-BOUNDARY.md))
at `block39.layer0.layer`'s payload, and slot 24's word 2, slot 25's word 3,
slot 26's word 2, slot 27's word 3, slot 28's word 2, slot 29's word 3,
slot 30's word 2, slot 31's word 3, slot 32's word 2, slot 33's word 3,
slot 34's word 2, slot 35's word 3, slot 36's word 2, slot 37's word 3,
slot 38's word 2, slot 39's word 3, slot 40's word 2, slot 41's word 3,
slot 42's word 2, slot 43's word 3, slot 44's word 2, slot 45's word 3,
slot 46's word 2, slot 47's word 3, slot 48's word 2, slot 49's word 3,
slot 50's word 2, slot 51's word 3 and slot 52's word 2
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
263,168-byte payload; `block26.layer1.layer` at 13,163,520 — directly
after layer0's 524,288-byte payload; `block26.layer2.layer` at 13,426,688
— directly after layer1's 263,168-byte payload; `block26.layer3.layer` at
14,344,704 — directly after layer2's 917,568-byte payload at the same
448-byte alignment boundary (13,426,688 + 917,568 + 448); `block27.layer0.layer`
at 14,607,872 — directly after layer3's 263,168-byte payload at the 512
boundary (14,344,704 + 263,168); `block27.layer1.layer` at 15,132,160 —
directly after layer0's 524,288-byte payload (14,607,872 + 524,288);
`block27.layer2.layer` at 15,395,328 — directly after layer1's 263,168-byte
payload (15,132,160 + 263,168); `block27.layer3.layer` at 16,313,344 —
directly after layer2 at the 448-byte alignment boundary (15,395,328 +
917,568 + 448); `block28.layer0.layer` at 16,576,512 — directly after
layer3's 263,168-byte payload at the 512 boundary (16,313,344 + 263,168);
`block28.layer1.layer` at 17,100,800 — directly after layer0's 524,288-byte
payload (16,576,512 + 524,288); `block28.layer2.layer` at 17,363,968 —
directly after layer1's 263,168-byte payload (17,100,800 + 263,168);
`block28.layer3.layer` at 18,281,984 — directly after layer2 at the
448-byte alignment boundary (17,363,968 + 917,568 + 448); `block29.layer0.layer`
at 18,545,152 — directly after layer3's 263,168-byte payload at the 512
boundary (18,281,984 + 263,168); `block29.layer1.layer` at 19,069,440 —
directly after layer0's 524,288-byte payload (18,545,152 + 524,288); and
`block29.layer2.layer` at 19,332,608 — directly after layer1's 263,168-byte
payload (19,069,440 + 263,168); `block29.layer3.layer` at 20,250,624 —
directly after layer2 at the 448-byte alignment boundary (19,332,608 +
917,568 + 448); `block3.layer0.layer` at 20,513,792 — directly after
layer3's 263,168-byte payload at the 512 boundary (20,250,624 + 263,168);
and `block30.layer0.layer` at 20,534,784 — directly after that 20,672-byte
record at its 320-byte alignment boundary (20,513,792 + 20,672 + 320), so the
record
order is the DLL's declaration order, block24 repeats block23's
layer0-3 shape, blocks 25, 26, 27, 28 and 29 are repeating that shape again, and one auxiliary record (`block3.layer0.layer`, 20,672 B) is declared between `block29.layer3.layer` and `block30.layer0.layer`). layer2's
917,568-byte payload is not
512-aligned: the following record starts 448 bytes later at the alignment
boundary (arithmetic from the probe's live DLL walk, which enforces exactly
one record at that boundary), so the tail control's following-record lookup
is alignment-aware. Beyond the anchors, the probe
computes and enforces the structural claim itself (`setup.heap_scan`, retained
in every run's evidence): every heap-pointing parameter word of every scanned
trace lands at a record payload start — **459 of 459 in each scan record,
zero mid-record** (the scans cover the s23–s52 traces; s24, s25, s26, s27,
s28, s29, s30, s31, s32, s33, s34, s35, s36, s37, s38, s39, s40, s41, s42,
s43, s44, s45, s46, s47, s48, s49, s50 and s51 are each
scanned
by two
adjacent contracts' setups). The binding's *identity*
(`block30.layer0.layer`, 524,288 B at slot 52) is enforced at runtime and
named in the source contract table; what the evidence retains is the
binding's *content* and *position* — `setup.record_sha256` and
`setup.heap_offset` — plus the enforced equality to the layout walk, so a
record rename would not disturb any run's verdicts.

## Variant finding: the tilesync/chained sync record

All twenty-nine contracts' fp8 siblings (`..._512_fp8`) are **created but never
launched by the vendor** (pinned: 0 launches in every trace), so they have no
launch contract of their own; each is replayed at the live variant's geometry
as a variant probe. In every case the sibling's **output tensor matches the
reference byte-for-byte** but its **sync record is left at the input state**
(`sync_from_input` true on every sibling entry) while the live variant (tilesync at
slot 24, chained at slots 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51 and 52) writes it. That divergence confinement
is **enforced** (`sibling-divergence`): the differences inside the sync
record must equal the total difference count over the whole pool — which
proves every differing byte lies within the record — and the record itself
must differ in **every byte of its changed extent**. The extent is **per
slot, not a constant**: 96 B at slots 24, 25, 27, 28, 29, 31, 32, 33, 35, 36, 37, 39, 40, 41, 43, 44, 45, 47, 48, 49, 51 and 52,
**192 B at slots 26, 34, 42 and 50**
and **256 B at slots 30, 38 and 46** (each predicted from the independent ladder
write-set and confirmed exactly by
the replay's sibling divergence — two sources, one number). The tilesync-only
sync record is now observed at thirty sites — the decoder's out2, and slots
24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51 and 52 — a family trait of the `*_tilesync_*` / `*_chained_*`
kernels' chain handoff.

## Controls

| Mode | Slot 24 | Slot 25 | Slot 26 | Slot 27 | Slot 28 | Slot 29 | Slot 30 | Slot 31 | Slot 32 | Slot 33 | Slot 34 | Slot 35 | Slot 36 | Slot 37 | Slot 38 | Slot 39 | Slot 40 | Slot 41 | Slot 42 | Slot 43 | Slot 44 | Slot 45 | Slot 46 | Slot 47 | Slot 48 | Slot 49 | Slot 50 | Slot 51 | Slot 52 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---| --- |
| `execute` ×2 | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** |
| `skip` (no launch) | pool bit-identical to input and non-matching — diff counts **344,085 / 344,104 / 344,110**, exactly the ladder write-set counts | same, diffs **340,602 / 341,160 / 341,167** = the s24→s25 ladder counts | same, diffs **342,905 / 342,801 / 342,841** = the s25→s26 ladder counts | same, diffs **340,726 / 341,250 / 341,150** = the s26→s27 ladder counts | same, diffs **344,027 / 344,077 / 344,069** = the s27→s28 ladder counts | same, diffs **341,165 / 341,559 / 341,526** = the s28→s29 ladder counts | same, diffs **343,293 / 343,253 / 343,275** = the s29→s30 ladder counts | same, diffs **340,970 / 341,345 / 341,358** = the s30→s31 ladder counts | same, diffs **342,117 / 342,432 / 342,376** = the s31→s32 ladder counts | same, diffs **340,754 / 341,134 / 341,144** = the s32→s33 ladder counts | same, diffs **342,020 / 341,782 / 341,773** = the s33→s34 ladder counts | same, diffs **340,617 / 340,872 / 340,885** = the s34→s35 ladder counts | same, diffs **342,288 / 341,759 / 341,735** = the s35→s36 ladder counts | same, diffs **340,998 / 342,180 / 342,179** = the s36→s37 ladder counts | same, diffs **341,836 / 342,467 / 342,490** = the s37→s38 ladder counts | same, diffs **340,575 / 341,602 / 341,525** = the s38→s39 ladder counts | same, diffs **342,483 / 342,991 / 342,940** = the s39→s40 ladder counts | same, diffs **341,529 / 341,682 / 341,833** = the s40→s41 ladder counts | same, diffs **341,857 / 342,616 / 342,726** = the s41→s42 ladder counts | same, diffs **341,031 / 341,654 / 341,826** = the s42→s43 ladder counts | same, diffs **342,613 / 343,155 / 343,140** = the s43→s44 ladder counts | same, diffs **340,586 / 341,494 / 341,395** = the s44→s45 ladder counts | same, diffs **341,677 / 342,212 / 342,214** = the s45→s46 ladder counts | same, diffs **340,810 / 340,607 / 340,618** = the s46→s47 ladder counts | same, diffs **341,033 / 341,572 / 341,529** = the s47→s48 ladder counts | same, diffs **340,692 / 340,683 / 340,636** = the s48→s49 ladder counts | same, diffs **340,830 / 341,096 / 341,110** = the s49→s50 ladder counts | same, diffs **340,652 / 340,704 / 340,674** = the s50→s51 ladder counts | same, diffs **340,904/341,369/341,391** = the s51→s52 ladder counts |
| `wrong` (weights rotated inwards 1,024 B) | rejects in all entries (319,425–319,537 bytes differ), pool changed from input (dead-launch detection) | rejects (319,545–319,658 bytes differ), pool changed from input | rejects (341,086–342,203 bytes differ), pool changed from input | rejects (342,666–342,849 bytes differ), pool changed from input | rejects (343,654–343,784 bytes differ), pool changed from input | rejects (343,429–343,791 bytes differ), pool changed from input | rejects (341,008–341,728 bytes differ), pool changed from input | rejects (343,233–343,398 bytes differ), pool changed from input | rejects (340,696–340,891 bytes differ), pool changed from input | rejects (342,186–342,308 bytes differ), pool changed from input | rejects (339,481–340,740 bytes differ), pool changed from input | rejects (342,832–343,015 bytes differ), pool changed from input | rejects (340,920–341,551 bytes differ), pool changed from input | rejects (341,331–342,654 bytes differ), pool changed from input | rejects (339,569–340,249 bytes differ), pool changed from input | rejects (342,908–343,043 bytes differ), pool changed from input | rejects (341,039–341,235 bytes differ), pool changed from input | rejects (342,327–342,472 bytes differ), pool changed from input | rejects (340,161–340,730 bytes differ), pool changed from input | rejects (342,490–342,698 bytes differ), pool changed from input | rejects (340,856–341,150 bytes differ), pool changed from input | rejects (342,399–342,648 bytes differ), pool changed from input | rejects (340,296–341,393 bytes differ), pool changed from input | rejects (342,899–343,156 bytes differ), pool changed from input | rejects (341,024–341,224 bytes differ), pool changed from input | rejects (342,682–342,841 bytes differ), pool changed from input | rejects (340,007–340,822 bytes differ), pool changed from input | rejects (342,432–342,542 bytes differ), pool changed from input | rejects (340,929–341,109 bytes differ), pool changed from input |
| `tail` (bytes beyond the record replaced 0x5A vs 0xA5) | **output unchanged** — bounded by the 524,288-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 917,568-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 524,288-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 917,568-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 524,288-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 917,568-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 524,288-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 917,568-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 524,288-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 917,568-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 524,288-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 917,568-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 524,288-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 917,568-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 524,288-byte record |

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
342,192/341,663/341,639 tensor + 96 sync, slots 36→37:
340,902/342,084/342,083 tensor + 96 sync, slots 37→38:
341,580/342,211/342,234 tensor + 256 sync, slots 38→39:
340,479/341,506/341,429 tensor + 96 sync, slots 39→40:
342,387/342,895/342,844 tensor + 96 sync, slots 40→41:
341,433/341,586/341,737 tensor + 96 sync, slots 41→42:
341,665/342,424/342,534 tensor + 192 sync, slots 42→43:
340,935/341,558/341,730 tensor + 96 sync, slots 43→44:
342,517/343,059/343,044 tensor + 96 sync, slots 44→45:
340,490/341,398/341,299 tensor + 96 sync, slots 45→46:
341,421/341,956/341,958 tensor + 256 sync, slots 46→47:
340,714/340,511/340,522 tensor + 96 sync, slots 47→48:
340,937/341,476/341,433 tensor + 96 sync, slots 48→49:
340,596/340,587/340,540 tensor + 96 sync, slots 49→50:
340,638/340,904/340,918 tensor + 192 sync, slots 50→51:
340,556/340,608/340,578 tensor + 96 sync, slots 51→52:
340,808/341,273/341,295 tensor + 96 sync — every execute interval with a
names file, complete; the s23→s24 attribution predates the names tool and is
stated in the slot-24 section above; the fp8
siblings show the tensor-only counts, exactly the sync extent less).

## Method

`vr/native-slot-replay-probe.py` now carries a **per-slot contract table**
(`CONTRACTS`, `--slot {24..52}`): parameter-block size and word classes
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
(`vr/logs/slot-replay/s{24..52}/{a,b,skip,wrong,tail}`) retains digests and
bounded windows only (windows are written before verdicts on any mismatch);
the `pair` command persists its result as `pair.json` beside the first run.

Self-test: **25 corruption controls** (12 verdict, 6 consistency, 5 pair, 1
metadata, 1 grid) each rejected with its pinned label or message, plus **245
known answers** (3 diff, 4 layout, 4 following-record, 1 metadata, 1 grid,
and 8 contract-table shape checks per contract — 232 across the twenty-nine
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

Runs on 18/10/2026 against the final source of the slot-52 stage (the
contract-52 entry, usage text and docstring bullets predate the runs; the
only post-run probe-source change is the module-summary contract count
word, "Twenty-eight" → "Twenty-nine" — a docstring count with no functional
effect):

- `vr/logs/ladder/{s52,s52b}` — the new slot-52 reference captures
  (validated: cut accounting, pool digests, the slot-aware output canary,
  input digests; `ladder-report.py pair` enforces byte-equal pools with a
  fail-closed exit — its passing report is retained as `pair-s52-s52b.json`,
  pool digests `0x920f35ad…/0x81498de4…/0x7d377c56…` equal across runs, all
  three pool frames byte-identical). Traces
  `vr/logs/capture/nvapi-trace-ladder-s52{,b}.log` (the slot-52 parameter
  dump is stable across evaluations and identical between the runs; the
  capture sidecar's three records match it byte-for-byte).
- `vr/logs/ladder/diff-s51-s52.json`, `vr/logs/ladder/names-s51-s52.txt` —
  the measured slot-52 write-set (2 segments per frame, zero unattributed:
  340,808/341,273/341,295 tensor + 96 sync).
- `vr/logs/slot-replay/s52/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s
  `c6e24122…`); the skip/wrong/tail signatures match the mode contracts
  (skip pool bit-identical to input and non-matching — diffs
  340,904/341,369/341,391 = the s51→s52 ladder counts, wrong changed from
  input and non-matching — 340,929–341,109 bytes differ, tail output
  unchanged beyond the record).
- **Additivity/source neutrality:** the slot-51 execute run was re-taken
  under the slot-52 source (`s51/recheck-stage44`) and pairs
  **bit-identical** with the retained `s51/a` (equal `run.json` SHA-256s,
  `f6ff439e…` — the same digest as stage 43's retained pair, the pairing
  persisted beside the recheck run as `pair.json`, and `s51/a`'s retained
  a-vs-b `pair.json` is untouched), so the contract-52 addition changes no
  prior verdict.
- **Review round:** independent review FAIL → fixed (one pass). The pass
  found 2 must-fix — the probe's module-summary contract count left at
  "Twenty-eight per-slot contracts" (the table now holds twenty-nine) and
  the slot-36 narrative's `ffwd` "covered launches" count left at six
  (28/32/36/40/44/48) while its own surrounding enumeration already carried
  slot 52 — and 1 advisory (the same sentence's `proj` reuse list omitting
  slot 51, a stale list the stage-43 review consciously left as rolling
  history, extended here since the sentence was being corrected anyway).
  All numeric claims, the contract-table shape, every enumeration and the
  self-test accounting were verified correct against the retained
  evidence. The fixes are prose-only in `vr/NATIVE-SLOT-REPLAY.md` plus the
  named docstring count word in the probe — no functional change — so every
  retained verdict stands unchanged (the stage's runs were taken against the
  functional probe source; the slot-51 additivity recheck pairs `f6ff439e…`
  under it).
- `python3 vr/native-slot-replay-probe.py self-test` (245 known answers),
  `python3 vr/ladder-report.py self-test` (19 controls) and `python3
  vr/ladder-names.py self-test` (14 controls) all pass.

Runs on 18/10/2026 against the final source of the slot-51 stage (the
contract-51 entry, usage text and docstring bullets predate the runs; the
only post-run probe-source change is comment-only — the contract-51 comment
and docstring bullet's word-map phrasing, below):

- `vr/logs/ladder/{s51,s51b}` — the new slot-51 reference captures
  (validated: cut accounting, pool digests, the slot-aware output canary,
  input digests; `ladder-report.py pair` enforces byte-equal pools with a
  fail-closed exit — its passing report is retained as `pair-s51-s51b.json`,
  pool digests `0x1f08f83e…/0x3b8aa9f9…/0xadb3fae8…` equal across runs, all
  three pool frames byte-identical). Traces
  `vr/logs/capture/nvapi-trace-ladder-s51{,b}.log` (the slot-51 parameter
  dump is stable across evaluations and identical between the runs; the
  capture sidecar's three records match it byte-for-byte).
- `vr/logs/ladder/diff-s50-s51.json`, `vr/logs/ladder/names-s50-s51.txt` —
  the measured slot-51 write-set (2 segments per frame, zero unattributed:
  340,556/340,608/340,578 tensor + 96 sync).
- `vr/logs/slot-replay/s51/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s
  `f6ff439e…`); the skip/wrong/tail signatures match the mode contracts
  (skip pool bit-identical to input and non-matching — diffs
  340,652/340,704/340,674 = the s50→s51 ladder counts, wrong changed from
  input and non-matching — 342,432–342,542 bytes differ, tail output
  unchanged beyond the record).
- **Additivity/source neutrality:** the slot-50 execute run was re-taken
  under the slot-51 source (`s50/recheck-stage43`) and pairs
  **bit-identical** with the retained `s50/a` (equal `run.json` SHA-256s,
  `c496ee13…` — the same digest as stage 42's retained pair, the pairing
  persisted beside the recheck run as `pair.json`, and `s50/a`'s retained
  a-vs-b `pair.json` is untouched), so the contract-51 addition changes no
  prior verdict.
- **Review round:** independent review FAIL → fixed (one pass). The pass
  found 2 must-fix — the Controls table's four data rows taking their new
  slot-51 cell without the `|` separator (each row one cell short of the
  29-column header, a scripted-edit error) and the slot-52 next-candidate
  paragraph attributing the 62,976 sync-out landing to the w4 sync-in word
  (the s51 trace's slot-52 parameter block pins w4 at 62,464 —
  address-equal to slot 51's w6 — and w6 at 62,976) — and 2 advisory (this
  block's post-run-change forward reference, now named concretely, and one
  stale slot-36 narrative reuse list left untouched as rolling history per
  the stage-42 precedent). All numeric claims, the contract-table shape,
  every enumeration and the self-test accounting were verified correct
  against the retained evidence. The fixes are `vr/NATIVE-SLOT-REPLAY.md`
  prose only (plus the named comment-only probe change), so every retained
  verdict stands unchanged (the stage's runs were taken against the
  functional probe source; the slot-50 additivity recheck pairs `c496ee13…`
  under it).
- `python3 vr/native-slot-replay-probe.py self-test` (237 known answers),
  `python3 vr/ladder-report.py self-test` (19 controls) and `python3
  vr/ladder-names.py self-test` (14 controls) all pass.

Runs on 18/10/2026 against the final source of the slot-50 stage (the
contract-50 entry, usage text and docstring bullets predate the runs; the
only post-run probe-source change is comment-only — the constant-schema
note, the binding list and the word-map/count enumerations, below):

- `vr/logs/ladder/{s50,s50b}` — the new slot-50 reference captures
  (validated: cut accounting, pool digests, the slot-aware output canary,
  input digests; `ladder-report.py pair` enforces byte-equal pools with a
  fail-closed exit — its passing report is retained as `pair-s50-s50b.json`,
  pool digests `0x45be794a…/0xef6641da…/0xa30aa44c…` equal across runs, all
  three pool frames byte-identical). Traces
  `vr/logs/capture/nvapi-trace-ladder-s50{,b}.log` (the slot-50 parameter
  dump is stable across evaluations and identical between the runs).
- `vr/logs/ladder/diff-s49-s50.json`, `vr/logs/ladder/names-s49-s50.txt` —
  the measured slot-50 write-set (2 segments per frame, zero unattributed:
  340,638/340,904/340,918 tensor + 192 sync).
- `vr/logs/slot-replay/s50/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s
  `c496ee13…`); the skip/wrong/tail signatures match the mode contracts
  (skip pool bit-identical to input and non-matching — diffs
  340,830/341,096/341,110 = the s49→s50 ladder counts, wrong changed from
  input and non-matching — 340,007–340,822 bytes differ, tail output
  unchanged beyond the record).
- **Additivity/source neutrality:** the slot-49 execute run was re-taken
  under the slot-50 source (`s49/recheck-stage42`) and pairs
  **bit-identical** with the retained `s49/a` (equal `run.json` SHA-256s,
  `2b8d0235…` — the same digest as stage 41's retained pair, the pairing
  persisted beside the recheck run as `pair.json`, and `s49/a`'s retained
  a-vs-b `pair.json` is untouched), so the contract-50 addition changes no
  prior verdict.
- **Review round:** independent review FAIL → fixed (two passes). The first
  pass found 4 must-fix — the slot-50 write-set "tensor" figures double-counting
  the sync bytes (the cross-tool paragraph and narrative quoting the skip
  totals 340,830/341,096/341,110 where the tensor-only counts are
  340,638/340,904/340,918), the slot-50 narrative quoting block (32,8,1)
  where the traces' created records pin (32,4,1) for `qkv` kernels, the
  missing stage-42 Evidence block (every prior stage had one), and
  several stale counts/enumerations not extended to slot 50 (the title and
  Result summary still "twenty-six/24-49", the Result per-slot table missing
  its slot-50 row, the weight-heap and payload-anchor lists not extended
  through `block29.layer2.layer` at 19,332,608, stale scan-range text
  s23–s49, the slot-48 Evidence block's self-test count carried over as 221)
  and the `## Next` section still listing slot 50 as remaining work — and 2
  advisory (the "word map exactly" parenthetical over-claiming slots 26/42,
  which reserve w4 as zero, and stale block29 "to be confirmed" predictions).
  A second review pass confirmed all of those and found 3 more of the same
  stale-enumeration class — the consolidated word-map list missing its
  slot-50 entry, the `## Next` bindings list not extended through slot 50's
  `block29.layer2.layer` at 19,332,608, and one stale block29 future-tense
  prediction ("tests the cycle's low-dword step at slots 50/51") — plus the
  review-round line undercounting its own findings, all now fixed. The fixes
  are `vr/NATIVE-SLOT-REPLAY.md` prose plus comment-only probe changes, so
  every retained verdict stands unchanged (the stage's runs were taken
  against the functional probe source; the slot-49 additivity recheck pairs
  `2b8d0235…` under it).
- `python3 vr/native-slot-replay-probe.py self-test` (229 known answers),
  `python3 vr/ladder-report.py self-test` (19 controls) and `python3
  vr/ladder-names.py self-test` (14 controls) all pass.

Runs on 18/10/2026 against the final source of the slot-49 stage (the
contract-49 entry, usage text and docstring bullets predate the runs; the
only post-run probe-source change is comment-only — the contract-schema
sync-extent enumeration, below):

- `vr/logs/ladder/{s49,s49b}` — the new slot-49 reference captures
  (validated: cut accounting, pool digests, the slot-aware output canary,
  input digests; `ladder-report.py pair` enforces byte-equal pools with a
  fail-closed exit — its passing report is retained as `pair-s49-s49b.json`,
  pool digests equal across runs, all three pool frames byte-identical).
  Traces `vr/logs/capture/nvapi-trace-ladder-s49{,b}.log` (the slot-49
  parameter dump is stable across evaluations and identical between the
  runs; the capture sidecar's three records match it byte-for-byte).
- `vr/logs/ladder/diff-s48-s49.json`, `vr/logs/ladder/names-s48-s49.txt` —
  the measured slot-49 write-set (2 segments per frame, zero unattributed:
  340,596/340,587/340,540 tensor + 96 sync).
- `vr/logs/slot-replay/s49/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s
  `2b8d0235…`); the skip/wrong/tail signatures match the mode contracts
  (skip pool bit-identical to input and non-matching — diffs
  340,692/340,683/340,636 = the s48→s49 ladder counts, wrong changed from
  input and non-matching — 342,682–342,841 bytes differ, tail output
  unchanged beyond the record).
- **Additivity/source neutrality:** the slot-48 execute run was re-taken
  under the slot-49 source (`s48/recheck-stage41`) and pairs
  **bit-identical** with the retained `s48/a` (equal `run.json` SHA-256s,
  `5460b105…` — the same digest as stage 40's retained pair, the pairing
  persisted beside the recheck run as `pair.json`, and `s48/a`'s retained
  a-vs-b `pair.json` is untouched), so the contract-49 addition changes no
  prior verdict.
- **Review round:** independent review FAIL → fixed (one pass). The pass
  found 5 must-fix — the Method count's stale **213 known answers** (the
  breakdown and `self_test()` both total 221 after the 26th contract), the
  contract-schema sync-extent enumeration not extended to slot 49, the
  slot-49 narrative (and the pre-existing slot-45 one) quoting block
  (32,8,1) where the traces' created records pin (32,4,1) for
  `ffwd_proj`/`qkv` kernels, the weight-heap weights-word and payload-anchor
  lists not extended through `block29.layer1.layer` at 19,069,440, and stale
  scan-range/binding-identity text (s23–s48 → s23–s49) — and 1 advisory (the
  covered-launch reuse enumerations stale since the slot-44 stage), all now
  fixed. The fixes are `vr/NATIVE-SLOT-REPLAY.md` prose plus one comment-only
  probe change, so every retained verdict stands unchanged (the stage's runs
  were taken against the functional probe source; the slot-48 additivity
  recheck pairs `5460b105…` under it).
- `python3 vr/native-slot-replay-probe.py self-test` (221 known answers),
  `python3 vr/ladder-report.py self-test` (19 controls) and `python3
  vr/ladder-names.py self-test` (14 controls) all pass.

Runs on 18/10/2026 against the final source of the slot-48 stage (the
contract-48 entry, usage text and docstring bullets predate the runs; no
post-run probe-source changes):

- `vr/logs/ladder/{s48,s48b}` — the new slot-48 reference captures
  (validated: cut accounting, pool digests, the slot-aware output canary,
  input digests; `ladder-report.py pair` enforces byte-equal pools with a
  fail-closed exit — its passing report is retained as `pair-s48-s48b.json`,
  pool digests equal across runs, all three pool frames byte-identical).
  Traces `vr/logs/capture/nvapi-trace-ladder-s48{,b}.log` (the slot-48
  parameter dump is stable across evaluations and identical between the
  runs; the capture sidecar's three records match it byte-for-byte).
- `vr/logs/ladder/diff-s47-s48.json`, `vr/logs/ladder/names-s47-s48.txt` —
  the measured slot-48 write-set (2 segments per frame, zero unattributed:
  340,937/341,476/341,433 tensor + 96 sync).
- `vr/logs/slot-replay/s48/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s
  `5460b105…`); the skip/wrong/tail signatures match the mode contracts
  (skip pool bit-identical to input and non-matching — diffs
  341,033/341,572/341,529 = the s47→s48 ladder counts, wrong changed from
  input and non-matching — 341,024–341,224 bytes differ, tail output
  unchanged beyond the record).
- **Additivity/source neutrality:** the slot-47 execute run was re-taken
  under the slot-48 source (`s47/recheck-stage40`) and pairs
  **bit-identical** with the retained `s47/a` (equal `run.json` SHA-256s,
  `3d7abdee…` — the same digest as stage 39's retained pair, the pairing
  persisted beside the recheck run as `pair.json`, and `s47/a`'s retained
  a-vs-b `pair.json` is untouched), so the contract-48 addition changes no
  prior verdict.
- `python3 vr/native-slot-replay-probe.py self-test` (213 known answers),
  `python3 vr/ladder-report.py self-test` (19 controls) and `python3
  vr/ladder-names.py self-test` (14 controls) all pass.

Runs on 18/10/2026 against the final source of the slot-47 stage (the
`Gate stage 39` commit; the contract-47 entry, usage text and docstring
bullets predate the runs, the post-run touch-ups are comment-only, below):

- `vr/logs/ladder/{s47,s47b}` — the new slot-47 reference captures
  (validated: cut accounting, pool digests, the slot-aware output canary,
  input digests; `ladder-report.py pair` enforces byte-equal pools with a
  fail-closed exit — its passing report is retained as `pair-s47-s47b.json`,
  pool digests equal across runs, all three pool frames byte-identical).
  Traces `vr/logs/capture/nvapi-trace-ladder-s47{,b}.log` (the slot-47
  parameter dump is stable across evaluations and identical between the
  runs; the capture sidecar's three records match it byte-for-byte).
- `vr/logs/ladder/diff-s46-s47.json`, `vr/logs/ladder/names-s46-s47.txt` —
  the measured slot-47 write-set (2 segments per frame, zero unattributed:
  340,714/340,511/340,522 tensor + 96 sync).
- `vr/logs/slot-replay/s47/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s
  `3d7abdee…`); the skip/wrong/tail signatures match the mode contracts
  (skip pool bit-identical to input and non-matching, wrong changed from
  input and non-matching, tail output unchanged beyond the record).
- **Source neutrality:** the only post-run probe-source changes are
  comment-only documentation (the cycle-confirmation passage and the
  cumulative enumerations) — no functional probe code changed after the
  runs.
- `python3 vr/native-slot-replay-probe.py self-test` (205 known answers),
  `python3 vr/ladder-report.py self-test` (19 controls) and `python3
  vr/ladder-names.py self-test` (14 controls) all pass.

Runs on 17-18/10/2026 across the slot-44, slot-45 and slot-46 stages (the
`Gate stage 36`–`Gate stage 38` commits; the stages ran without their own
evidence blocks, so this block documents the retained artefacts for all
three; the contract-46 entry, usage text and docstring bullets predate the
slot-46 runs, the post-run touch-ups are comment-only, below):

- `vr/logs/ladder/{s44,s44b,s45,s45b,s46,s46b}` — the slot-44/45/46
  reference captures (validated: cut accounting, pool digests, the
  slot-aware output canary, input digests; `ladder-report.py pair` enforces
  byte-equal pools with a fail-closed exit — its passing reports are
  retained as `pair-s44-s44b.json`, `pair-s45-s45b.json` and
  `pair-s46-s46b.json`, pool digests equal across runs, all three pool
  frames byte-identical). Traces
  `vr/logs/capture/nvapi-trace-ladder-s{44,45,46}{,b}.log` (each slot's
  parameter dump is stable across evaluations and identical between the
  runs; the capture sidecars' three records match it byte-for-byte —
  enforced by every replay run's setup).
- `vr/logs/ladder/diff-s43-s44.json`, `diff-s44-s45.json`,
  `diff-s45-s46.json` and `vr/logs/ladder/names-s43-s44.txt`,
  `names-s44-s45.txt`, `names-s45-s46.txt` — the measured write-sets (2
  segments per frame, zero unattributed: 342,517/343,059/343,044 tensor +
  96 sync at s43→s44, 340,490/341,398/341,299 tensor + 96 sync at
  s44→s45, 341,421/341,956/341,958 tensor + 256 sync at s45→s46).
- `vr/logs/slot-replay/s{44,45,46}/…` — the five replay runs per slot
  (execute ×2, skip, wrong, tail) plus the persisted pair results
  (`a/pair.json`, bit-identical, equal `run.json` SHA-256s `d0825bd5…`
  (s44), `c9223c37…` (s45), `888e0e71…` (s46)); the skip/wrong/tail
  signatures match the mode contracts (skip pool bit-identical to input
  and non-matching, wrong changed from input and non-matching, tail output
  unchanged beyond the record).
- **Post-run source-neutrality:** the only post-run probe-source changes
  are comment-only documentation (the cycle-prediction passages, the
  cumulative enumerations and the weight-bytes list, all fixed in the
  review round) — no functional probe code changed after the runs; the
  slot-46 execute run was re-taken under the post-review source
  (`s46/recheck-under-slot46-final-source`) and pairs bit-identical with
  the retained `s46/a` (equal `run.json` SHA-256 `888e0e71…`).
- **Review round:** independent review FAIL → fixed (one pass). The pass
  found 17 must-fix — cumulative enumerations not extended through slot 46
  (the contract counts and slot ranges, the sync-extent lists, the
  weight-bytes table and word maps, the weight-heap anchors/payload
  list/scan range/binding identity, the variant-finding lists, the
  Controls table and cross-tool intervals, the scope bullets and
  read-inference list), two stale cycle-prediction passages (slot 46's w4
  is measured now), one stale usage range and two write-set citations
  without retained artefacts (the s43→s44 and s44→s45 names/diff files are
  generated and retained) — and 3 advisory (the missing evidence block,
  the per-stage self-test counts and a stale phrase), all now fixed. The
  fixes are documentation in `vr/NATIVE-SLOT-REPLAY.md` and comment-only
  documentation in `vr/native-slot-replay-probe.py`; no functional probe
  code changed after the runs.
- `python3 vr/native-slot-replay-probe.py self-test` (197 known answers),
  `python3 vr/ladder-report.py self-test` (19 controls) and `python3
  vr/ladder-names.py self-test` (14 controls) all pass.

Runs on 17/10/2026 against the final source of the slot-43 stage (the
contract-43 entry, usage text and docstrings predate the runs; the post-run
touch-ups are comment-only, below):

- `vr/logs/ladder/{s43,s43b}` — the new slot-43 reference captures
  (validated: cut accounting, pool digests, the slot-aware output canary,
  input digests; `ladder-report.py pair` enforces byte-equal pools with a
  fail-closed exit — its passing report is retained as `pair-s43-s43b.json`,
  pool digests equal across runs, all three pool frames byte-identical).
  Traces `vr/logs/capture/nvapi-trace-ladder-s43{,b}.log` (the slot-43
  parameter dump is stable across evaluations and identical between the
  runs; the capture sidecar's three records match it byte-for-byte).
- `vr/logs/ladder/diff-s42-s43.json`, `vr/logs/ladder/names-s42-s43.txt` —
  the measured slot-43 write-set (2 segments per frame, zero unattributed:
  340,935/341,558/341,730 tensor + 96 sync).
- `vr/logs/slot-replay/s43/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s
  `0a0ee5fc…`).
- **Additivity/source neutrality:** the slot-42 execute run was re-taken
  under the slot-43 source (`s42/recheck-under-slot43-source`) and
  pairs **bit-identical** with the retained `s42/a` (equal `run.json`
  SHA-256s `200f1e07…` — the same digest as s42's own retained execute run,
  so the contract-43 addition changes no prior verdict; the pairing
  persisted beside the recheck run as `pair.json`, and `s42/a`'s retained
  a-vs-b `pair.json` is untouched).
- **Post-run source-neutrality:** the only post-run probe-source changes are
  comment-only documentation (the contract-schema sync-extent enumeration
  and the slot-42 contract note, two stale relative-time phrases, the cycle
  framing and the layer2→3 pattern phrase, all fixed in the review round) —
  no functional code changed after the runs. The slot-43 execute run was
  re-taken under the post-review source
  (`s43/recheck-under-slot43-final-source`) and pairs **bit-identical** with
  the retained `s43/a` (equal `run.json` SHA-256s `0a0ee5fc…`), and the
  slot-42 recheck was repeated under the same source
  (`s42/recheck-under-slot43-final-source`) — again bit-identical with
  `s42/a` (`200f1e07…`) — so every retained verdict stands unchanged.
- **Review round:** independent review FAIL → fixed (one pass). The pass
  found 4 must-fix — the contract-schema sync-extent enumeration not
  extended to slot 43, the weight-bytes source table row missing
  `block27.layer3.layer`, the word-map section still labelling block27's
  `proj` half's w8 a prediction and missing its slot-43 word map, and a
  dangling "below" in this block's header — and 3 advisory (two stale "next
  stage" phrases, the cycle framing, the layer2→3 pattern phrase), all now
  fixed. The fixes are prose and comment-only documentation in
  `vr/NATIVE-SLOT-REPLAY.md` and `vr/native-slot-replay-probe.py`; no
  functional probe code changed after the runs (the post-review rechecks
  pair `0a0ee5fc…` and `200f1e07…` under the final source).
- `python3 vr/native-slot-replay-probe.py self-test` (173 known answers),
  `python3 vr/ladder-report.py self-test` (19 controls) and `python3
  vr/ladder-names.py self-test` (14 controls) all pass.

Runs on 17/10/2026 against the final source of the slot-42 stage (the
contract-42 entry and usage text predate the runs; the docstrings'
post-run touch-ups are comment-only, below):

- `vr/logs/ladder/{s42,s42b}` — the new slot-42 reference captures
  (validated: cut accounting, pool digests, the slot-aware output canary,
  input digests; `ladder-report.py pair` enforces byte-equal pools with a
  fail-closed exit — its passing report is retained as `pair-s42-s42b.json`,
  pool digests equal across runs, all three pool frames byte-identical).
  Traces `vr/logs/capture/nvapi-trace-ladder-s42{,b}.log` (the slot-42
  parameter dump is stable across evaluations and identical between the
  runs; the capture sidecar's three records match it byte-for-byte).
- `vr/logs/ladder/diff-s41-s42.json`, `vr/logs/ladder/names-s41-s42.txt` —
  the measured slot-42 write-set (2 segments per frame, zero unattributed:
  341,665/342,424/342,534 tensor + 192 sync).
- `vr/logs/slot-replay/s42/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s
  `200f1e07…`).
- **Additivity/source neutrality:** the slot-41 execute run was re-taken
  under the committed slot-42 source (`s41/recheck-under-slot42-source`) and
  pairs **bit-identical** with the retained `s41/a` (equal `run.json`
  SHA-256s `67f6826b…` — the same digest as s41's own retained execute run,
  so the contract-42 addition changes no prior verdict; the pairing
  persisted beside the recheck run as `pair.json`, and `s41/a`'s retained
  a-vs-b `pair.json` is untouched).
- **Post-run source-neutrality:** the only post-run probe-source changes are
  comment-only documentation (the module summary line's slot range and the
  block27 pair-constant phrasing, both fixed in the review round) — no
  functional code changed after the runs; the slot-41 execute run was
  re-taken once more under the post-review source
  (`s41/recheck-under-slot42-final-source`) and again pairs **bit-identical**
  with `s41/a` (equal `run.json` SHA-256 `67f6826b…`, the pairing persisted
  beside the recheck run as `pair.json`), so every retained verdict stands
  unchanged.
- **Review round:** independent review FAIL → fixed (one pass). The pass
  found 4 must-fix — the weight-bytes source table and the retained-evidence
  path range not extended to slot 42, the probe module summary line still
  reading slots 24-41, and pair-level phrasing for block27's second constant
  where only slot 42's w4 is measured (its proj half's w8 is a prediction,
  now labelled as such everywhere) — and 1 advisory (the display guard's
  numeric argument parsing), all now fixed. The fixes are
  `vr/NATIVE-SLOT-REPLAY.md` prose, comment-only
  `vr/native-slot-replay-probe.py` documentation and `vr/x11-keep-on-primary.py`
  (outside the verdict path); no functional probe code changed after the
  runs, so every retained verdict stands unchanged (the additivity rechecks
  pair `67f6826b…` under the pre- and post-review sources alike).
- `python3 vr/native-slot-replay-probe.py self-test` (165 known answers),
  `python3 vr/ladder-report.py self-test` (19 controls) and `python3
  vr/ladder-names.py self-test` (14 controls) all pass.

Runs on 17/10/2026 against the final source of the slot-41 stage (the
contract-41 entry, docstrings and usage text all predate the runs):

- `vr/logs/ladder/{s41,s41b}` — the new slot-41 reference captures
  (validated: cut accounting, pool digests, the slot-aware output canary,
  input digests; `ladder-report.py pair` enforces byte-equal pools with a
  fail-closed exit — its passing report is retained as `pair-s41-s41b.json`,
  pool digests equal across runs, all three pool frames byte-identical).
  Traces `vr/logs/capture/nvapi-trace-ladder-s41{,b}.log` (the slot-41
  parameter dump is stable across evaluations and identical between the
  runs; the capture sidecar's three records match it byte-for-byte).
- `vr/logs/ladder/diff-s40-s41.json`, `vr/logs/ladder/names-s40-s41.txt` —
  the measured slot-41 write-set (2 segments per frame, zero unattributed:
  341,433/341,586/341,737 tensor + 96 sync).
- `vr/logs/slot-replay/s41/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s
  `67f6826b…`).
- **Additivity/source neutrality:** the slot-40 execute run was re-taken
  under the committed slot-41 source (`s40/recheck-under-slot41-source`,
  named distinctly because stage 32's post-run recheck already occupies
  `s40/recheck-final-source`) and pairs **bit-identical** with the retained
  `s40/a` (equal `run.json` SHA-256s, `b05765cd…` — the same digest as
  `s40/recheck-final-source`, so all three slot-40 execute runs under the
  post-contract-40 sources are byte-identical; the pairing persisted beside
  the recheck run as `pair.json`, and `s40/a`'s retained a-vs-b `pair.json`
  is untouched), so the contract-41 addition changes no prior verdict.
- **Review round:** independent review FAIL → fixed (one pass). The pass
  found 2 must-fix — the slot-41 narrative and the slot-42 prediction quoted
  block (32,8,1), copy-pasted from the `proj`/`ffwd` narratives, where the
  trace's created records pin (32,4,1) for `ffwd_proj`/`qkv` kernels — and 1
  advisory (slot 41's unusual output spacing unremarked beside the prose
  claiming every prediction confirmed), all now fixed. The fixes are
  `vr/NATIVE-SLOT-REPLAY.md` prose only; no probe source changed after the
  runs, so every retained verdict stands unchanged (the stage's runs were
  taken against the final probe source; the slot-40 additivity recheck pairs
  `b05765cd…` under it).
- `python3 vr/native-slot-replay-probe.py self-test` (157 known answers),
  `python3 vr/ladder-report.py self-test` (19 controls) and `python3
  vr/ladder-names.py self-test` (14 controls) all pass.

Runs on 17/10/2026 against the committed source of the slot-40 stage (the
contract-40 entry and usage text finalise before the primary runs; the
post-run probe-source changes are comment-only documentation, re-verified
verdict-neutral below):

- `vr/logs/ladder/{s40,s40b}` — the new slot-40 reference captures
  (validated: cut accounting, pool digests, the slot-aware output canary,
  input digests; `ladder-report.py pair` enforces byte-equal pools with a
  fail-closed exit — its passing report is retained as `pair-s40-s40b.json`,
  pool digests equal across runs, all three pool frames byte-identical).
  Traces `vr/logs/capture/nvapi-trace-ladder-s40{,b}.log` (the slot-40
  parameter dump is stable across evaluations and identical between the
  runs; the capture sidecar's three records match it byte-for-byte).
- `vr/logs/ladder/diff-s39-s40.json`, `vr/logs/ladder/names-s39-s40.txt` —
  the measured slot-40 write-set (2 segments per frame, zero unattributed:
  342,387/342,895/342,844 tensor + 96 sync).
- `vr/logs/slot-replay/s40/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s
  `b05765cd…`).
- **Post-run source-neutrality:** the only post-run probe-source changes are
  comment-only documentation (the module docstring's slot-40 coverage and the
  contract-schema comment's slot-40 extent/slot lists) — no functional code
  changed; the `s40/recheck-final-source` execute run against the committed
  source pairs **bit-identical** with the retained `s40/a` (equal `run.json`
  SHA-256s, `b05765cd…`, the pairing persisted beside the recheck run), so
  every retained verdict holds for the committed
  source (`s40/a`'s retained a-vs-b `pair.json` is untouched).
- **Additivity/source neutrality:** the slot-39 execute run was re-taken
  under the committed slot-40 source (`s39/recheck-final-source`) and pairs
  **bit-identical** with the retained `s39/a` (equal `run.json` SHA-256s,
  `2e374132…`; the pairing persisted beside the recheck run as `pair.json`,
  and `s39/a`'s retained a-vs-b `pair.json` is untouched), so the
  contract-40 addition changes no prior verdict.
- **Review round:** independent review FAIL → fixed (two passes). The first
  pass found 5 must-fix (the probe docstring's contract count and missing
  slot-40 bullet, the contract-schema comment's 16-slot extent/slot
  enumerations, the MD weight-bytes table row and per-slot word-map paragraph
  not yet extended to slot 40, and a source-neutrality wording contradiction)
  plus 2 advisory (the stale "candidate" label on the confirmed slot 40); the
  re-review pass found 3 residual (the weight-bytes row, the `ffwd`
  "covered launches" count and the "sole source change" provenance phrase),
  all now fixed. The probe-source fixes are comment-only (module docstring +
  contract-schema comment), so the retained primary runs stand unchanged — a
  fresh `s40/recheck-final-source` execute under the committed source pairs
  **bit-identical** with the retained `s40/a` (`b05765cd…`) and the s39
  additivity recheck under the same source pairs `2e374132…`, so every
  retained verdict holds for the committed source; the remaining fixes are
  `vr/NATIVE-SLOT-REPLAY.md` prose. Self-test re-run green after the fixes
  (149 known answers, 25 controls, plus ladder-report's 19 and
  ladder-names' 14).
- `python3 vr/native-slot-replay-probe.py self-test` (149 known answers),
  `python3 vr/ladder-report.py self-test` (19 controls) and `python3
  vr/ladder-names.py self-test` (14 controls) all pass.

Runs on 17/10/2026 against the final source of the slot-39 stage (the
contract-39 entry, docstrings and usage text all predate the runs):

- `vr/logs/ladder/{s39,s39b}` — the new slot-39 reference captures
  (validated: cut accounting, pool digests, the slot-aware output canary,
  input digests; `ladder-report.py pair` enforces byte-equal pools with a
  fail-closed exit — its passing report is retained as `pair-s39-s39b.json`,
  pool digests equal across runs, all three pool frames byte-identical).
  Traces `vr/logs/capture/nvapi-trace-ladder-s39{,b}.log` (the slot-39
  parameter dump is stable across evaluations and identical between the
  runs; the capture sidecar's three records match it byte-for-byte).
- `vr/logs/ladder/diff-s38-s39.json`, `vr/logs/ladder/names-s38-s39.txt` —
  the measured slot-39 write-set (2 segments per frame, zero unattributed:
  340,479/341,506/341,429 tensor + 96 sync).
- `vr/logs/slot-replay/s39/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s
  2e374132…).
- **Additivity/source neutrality:** the slot-38 execute run was re-taken
  under the stage-31 source (`s38/recheck-final-source`) and pairs
  **bit-identical** with the retained `s38/a` (equal `run.json` SHA-256s,
  `4c047527…`; the pairing persisted beside the recheck run as `pair.json`,
  and `s38/a`'s retained a-vs-b `pair.json` is untouched), so the
  contract-39 addition changes no prior verdict.
- `python3 vr/native-slot-replay-probe.py self-test` (141 known answers),
  `python3 vr/ladder-report.py self-test` (19 controls) and `python3
  vr/ladder-names.py self-test` (14 controls) all pass.

Runs on 17/10/2026 against the final source of the slot-38 stage (the
contract-38 entry, docstrings and usage text all predate the runs):

- `vr/logs/ladder/{s38,s38b}` — the new slot-38 reference captures
  (validated: cut accounting, pool digests, the slot-aware output canary,
  input digests; `ladder-report.py pair` enforces byte-equal pools with a
  fail-closed exit — its passing report is retained as `pair-s38-s38b.json`,
  pool digests equal across runs). Traces
  `vr/logs/capture/nvapi-trace-ladder-s38{,b}.log` (the slot-38 parameter
  dump is stable across evaluations and identical between the runs and the
  s37 traces' dropped-launch record; the capture sidecar's three records
  match it byte-for-byte).
- `vr/logs/ladder/diff-s37-s38.json`, `vr/logs/ladder/names-s37-s38.txt` —
  the measured slot-38 write-set (2 segments per frame, zero unattributed:
  341,580/342,211/342,234 tensor + 256 sync).
- `vr/logs/slot-replay/s38/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s
  4c047527…).
- **Additivity/source neutrality:** the slot-37 execute run was re-taken
  under the stage-30 source (`s37/recheck-final-source`) and pairs
  **bit-identical** with the retained `s37/a` (equal `run.json` SHA-256s,
  `9088c6ff…`; the pairing persisted beside the recheck run as `pair.json`,
  and `s37/a`'s retained a-vs-b `pair.json` is untouched), so the
  contract-38 addition changes no prior verdict.
- `python3 vr/native-slot-replay-probe.py self-test` (133 known answers),
  `python3 vr/ladder-report.py self-test` (19 controls) and `python3
  vr/ladder-names.py self-test` (14 controls) all pass.

Runs on 17/10/2026 against the final source of the slot-37 stage (the
contract-37 entry, docstrings and usage text all predate the runs):

- `vr/logs/ladder/{s37,s37b}` — the new slot-37 reference captures
  (validated: cut accounting, pool digests, the slot-aware output canary,
  input digests; `ladder-report.py pair` enforces byte-equal pools with a
  fail-closed exit — its passing report is retained as `pair-s37-s37b.json`,
  pool digests equal across runs). Traces
  `vr/logs/capture/nvapi-trace-ladder-s37{,b}.log` (the slot-37 parameter
  dump is stable across evaluations and identical between the runs and the
  s36 traces' dropped-launch record; the capture sidecar's three records
  match it byte-for-byte).
- `vr/logs/ladder/diff-s36-s37.json`, `vr/logs/ladder/names-s36-s37.txt` —
  the measured slot-37 write-set (2 segments per frame, zero unattributed:
  340,902/342,084/342,083 tensor + 96 sync).
- `vr/logs/slot-replay/s37/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s). One
  earlier attempt at the `wrong` control was killed mid-run by a host-side
  command timeout before its first window; the empty directory is retained
  as `wrong-superseded-killed-by-host-timeout` (no `run.json`) and the
  control was re-taken cleanly.
- **Additivity/source neutrality:** the slot-36 execute run was re-taken
  under the stage-29 source (`s36/recheck-stage29-final-source`) and pairs
  **bit-identical** with the retained `s36/a` (equal `run.json` SHA-256s,
  `b6e98939…`), so the contract-37 addition changes no prior verdict.
- `python3 vr/native-slot-replay-probe.py self-test` (125 known answers),
  `python3 vr/ladder-report.py self-test` (19 controls) and `python3
  vr/ladder-names.py self-test` (14 controls) all pass.

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
- `python3 vr/native-slot-replay-probe.py self-test` (117 known answers — the
  stage-28 final source's own count, thirteen contracts of the then table),
  `python3
  vr/ladder-report.py self-test` (19 controls) and `python3
  vr/ladder-names.py self-test` (14 controls) all pass.

## Scope and limitations

- The claim covers **twenty-nine launches of the 156** (slots 24–52; six distinct
  kernel names, one shared across two geometries (26/30/34/38/42/46/50) and three reused at
  the same geometry (27/31/35/39/43/47/51, 28/32/36/40/44/48/52, 29/33/37/41/45/49)), one resolution
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
  slot 36's w0 and w4 reads (output of slot 35, previous sync),
  slot 37's w0, w1 and w7 reads (output, residual, previous sync),
  slot 38's w0 and w5 reads (output of slot 37, previous sync),
  slot 39's w0, w1 and w7 reads (output, residual, previous sync),
  slot 40's w0 and w4 reads (output of slot 39, previous sync),
  slot 41's w0, w1 and w7 reads (output, residual, previous sync),
  slot 42's w0 and w5 reads (output of slot 41, previous sync),
  slot 43's w0, w1 and w7 reads (output of slot 42, residual, previous sync),
  slot 44's w0 and w4 reads (output of slot 43, previous sync),
  slot 45's w0, w1 and w7 reads (output, residual, previous sync),
  slot 46's w0 and w5 reads (output of slot 45, previous sync),
  slot 47's w0, w1 and w7 reads (output of slot 46, residual, previous sync),
  slot 48's w0 and w4 reads (output of slot 47, previous sync),
  slot 49's w0, w1 and w7 reads (output, residual, previous sync),
  slot 50's w0 and w5 reads (output of slot 49, previous sync),
  slot 51's w0, w1 and w7 reads (output, residual, previous sync),
  slot 52's w0 and w4 reads (output of slot 51, previous sync)
  and slots 28/29's w0 and (w4/w7) reads — slot 29
  additionally its residual w1 — are inferred from the attribution (those words never change) and the
  word map, not from a measured read-set.
- The sync extents (96/96/192/96/96/96/256/96/96/96/192/96/96/96/256/96/96/96/192/96/96/96/256/96/96/96/192/96/96 B for slots 24–52) are the records'
  **changed** extents under
  this protocol; records whose unchanged bytes extend further are not
  excluded.
- This stage uses stored learned weight bytes through twenty-nine extracted records
  (bounded model-weight computation, [NATIVE-FP8.md](NATIVE-FP8.md)
  precedent) and is **not** a neural-rendered image. Proprietary
  reuse/publication rights remain unresolved; evidence retains digests and
  bounded 8 KiB windows (written only on mismatch) and stays local.

## Next

Repeat the replay for the remaining named kernels of
[LADDER-NAMES.md](LADDER-NAMES.md) — the encoder (slots 1–23), the
transformer interval's further `qkv`/`proj`/`ffwd` pairs (53–56), the ViT
stage (57–98), transformer-2 (100–131) and the decoder pyramid (132–153);
slot 99 (the decoder) is already covered by
[NATIVE-VENDOR-COMPARE.md](NATIVE-VENDOR-COMPARE.md) and slot 154 (the final
blend) writes only outside the working pool. Each remaining kernel needs a
reference capture at its
own cut point (`run-feature-probe.sh ladder sN N`) plus one contract entry,
and occasionally a small probe generalisation (slot 26 needed per-slot grid
resolution and an alignment-aware record lookup). Slots 24–52 now cover
seven complete `ffwd→proj` pairs (block23: slots 24/25, block24: 28/29,
block25: 32/33, block26: 36/37, block27: 40/41, block28: 44/45, block29:
48/49), block30's `ffwd` half (slot 52, its pair's second half at slot 53
still open) and seven complete `qkv→proj`
pairs (block23: slots 26/27, block24: 30/31, block25: 34/35, block26: 38/39,
block27: 42/43, block28: 46/47, block29: 50/51), so
the pair pattern, the chained sync handoff (with its
per-slot word position: w5 at slots 26/30/34/38/42/46/50, w7 at slots 25/27/29/31/33/35/37/39/41/43/45/47/49/51, w4 at
slots 28/32/36/40/44/48/52)
and the record-ordering pattern are established: each block repeats
block23's layer0-3 record shape in declaration order, so each remaining
slot's weight
record is predictable from its parameter block's weights word plus the
layout walk — and slots 29's, 30's, 31's, 32's, 33's, 34's, 35's, 36's, 37's,
38's, 39's, 40's, 41's, 42's, 43's, 44's, 45's, 46's, 47's, 48's, 49's, 50's,
51's and 52's
bindings measured exactly those
predictions (`block24.layer1.layer` at heap offset 9,226,240,
`block24.layer2.layer` at 9,489,408, `block24.layer3.layer` at 10,407,424,
`block25.layer0.layer` at 10,670,592, `block25.layer1.layer` at 11,194,880,
`block25.layer2.layer` at 11,458,048, `block25.layer3.layer` at
12,376,064, `block26.layer0.layer` at 12,639,232, `block26.layer1.layer`
at 13,163,520, `block26.layer2.layer` at 13,426,688, `block26.layer3.layer`
at 14,344,704, `block27.layer0.layer` at 14,607,872, `block27.layer1.layer`
at 15,132,160, `block27.layer2.layer` at 15,395,328,
`block27.layer3.layer` at 16,313,344, `block28.layer0.layer` at 16,576,512,
`block28.layer1.layer` at 17,100,800,
`block28.layer2.layer` at 17,363,968,
`block28.layer3.layer` at 18,281,984,
`block29.layer0.layer` at 18,545,152,
`block29.layer1.layer` at 19,069,440,
`block29.layer2.layer` at 19,332,608,
`block29.layer3.layer` at 20,250,624 and `block30.layer0.layer` at
20,534,784) — slot 52's landing one interloper past the block23-style shape
(`block3.layer0.layer`, 20,672 B at 20,513,792, declared between
`block29.layer3.layer` and `block30.layer0.layer`, so the "each block
repeats block23's layer0-3 shape" claim now carries that one-record
exception), the weights-word-plus-layout-walk prediction still exact. The qkv→proj pair's
second constant is now measured in both halves of all five fully-measured
carrying pairs
(block24
at `0xFFFFFFFCFFFFFFFC`, block25 at `0x00000000FFFFFFFC`, block26 at
`0xFFFFFFFC00000000` — slot 38's w4 and slot 39's w8 alike — block28 at
`0xFFFFFFFCFFFFFFFC`, slot 46's w4 and slot 47's w8 alike, and block29 at
`0x00000000FFFFFFFC`, slot 50's w4 and slot 51's w8 alike);
it is absent from block23's pair (both halves measured), every `ffwd→proj`
half and block27's pair (slot 42's w4 and slot 43's w8 both measured
reserved zero) — one value per
pair where present, shared by both halves wherever measured (block24,
block25, block26, block28 and block29). The carrying pairs'
values are the double, low-dword-only and high-dword-only forms of the same
marker dword (`0xFFFFFFFC`) on a repeating absent/double/low/high cycle
(block23 absent, block24 double, block25 low, block26 high, block27 absent,
block28 double, block29 low — now measured in both halves), block29's
pair at the
low-dword form.
The
slot 40
(`cc_split_swin_16h_ffwd_512_chained_fp8` — block27's ffwd half, the next
`ffwd→proj` pair's first half at grid (4,3,2), block (32,8,1)) is now
replayed byte-for-byte and confirms every prediction. Its parameter block
(psize 56) carries the ffwd word shape (pool, pool, weights, constant, pool,
zero, pool): w1 output (pool+26,459,136), w3 constant **`0x1C00000018`**,
w6 sync (pool+56,832) and no pair second constant (absent from every
`ffwd→proj` half). Its weights
word lands on the
layout walk's next record start (`block27.layer0.layer`, 524,288 B at heap
offset 14,607,872 — 14,344,704 + 263,168 at the 512 boundary, confirmed by the
probe's live DLL walk), its w0 is slot
39's w2 output (26,115,072) and its w4 sync-in is slot
39's w6 record (56,320). The sync
extent measured **96 B** (the ffwd norm), pinned from the s39→s40 write-set
and enforced as the sibling's divergence (the sync record at w6, 56,832 —
the 512-byte stride continuing), stable across all three frames. Slot
37's odd output spacing (w2 two tensor strides past w0, the encoder-residual
buffer between) is pinned as captured in the slot-37 note; what it encodes
is not yet established. Slot 41
(`cc_split_swin_16h_ffwd_proj_512_chained_fp8` — block27's `ffwd→proj`
pair's second half at grid (8,3,1), block (32,4,1)) is now
replayed byte-for-byte and confirms every word-map, record-binding and
sync-extent prediction. Its parameter block
(psize 72) is the slot-25/29/33/37 word map exactly (pool, pool, pool,
weights, constant, zero, pool, pool, zero): w2 output (pool+30,833,664), w4
constant **`0x1C00000018`**, w6 sync (pool+57,344), w7 sync-in (slot 40's
record at 56,832) and w5/w8 reserved zero (no pair second constant in any
ffwd-proj half). Its weights
word lands on the
layout walk's next record start (`block27.layer1.layer`, 263,168 B at heap
offset 15,132,160 — 14,607,872 + 524,288 at the 512 boundary), its w0 is slot
40's w1 output (26,459,136) and its w1 residual is the buffer slot 39 wrote
its output to (26,115,072 — the input slot 40 itself reads). The sync
extent measured **96 B** (the ffwd norm), pinned from the s40→s41 write-set
and enforced as the sibling's divergence (the sync record at w6, 57,344 —
the 512-byte stride continuing), stable across all three frames. Slot 41's
output placement is another spacing oddity, larger than slot 37's: its w2
(30,833,664) is 12.7 tensor strides (344,064 B) past its w0 — not a whole
stride from either input (its two inputs are one stride apart) — unlike the
one-stride output spacing of slots 25/29/33 and the two-stride spacing of
slot 37. It is pinned as captured; what that placement encodes is not yet
established. The
slot 42
(`cc_split_swin_16h_qkv_512_chained_fp8` — block27's `qkv→proj` pair's
first half at grid (4,3,4), block (32,4,1)) is now
replayed byte-for-byte and confirms every prediction. Its parameter block
(psize 56) is the slot-26/34 qkv word map exactly (pool, pool, weights,
constant, zero, pool, pool): w1 output (pool+31,177,728 — one stride past
its input, the regular output spacing resuming after slot 41's offset
placement), w3 constant **`0x1C00000018`**, w5 sync-in (slot 41's record at
57,344), w6 sync (pool+57,856) and **w4 reserved zero** (no pair second
constant). Its weights
word lands on the
layout walk's next record start (`block27.layer2.layer`, 917,568 B at heap
offset 15,395,328 — 15,132,160 + 263,168 at the 512 boundary, confirmed by
the probe's live DLL walk), its w0 is slot
41's w2 output (30,833,664). The sync
extent measured **192 B** (the grid-4,3,4 qkv norm of slots 26/34), pinned
from the s41→s42 write-set
and enforced as the sibling's divergence (the sync record at w6, 57,856 —
the 512-byte stride continuing), stable across all three frames. With
block23's pair carrying no second constant in either half, block27's pair
reserving both words zero (slot 42's w4 and slot 43's w8, both measured) and
blocks
24/25/26 each carrying one (double, low-dword, high-dword forms of the same
marker dword), the five measured pairs complete one four-step
absent/double/low/high cycle across blocks 23-27 — measured over those
blocks, a hypothesis beyond; the cycle
predicted
block28's pair at the double form (slot 46's w4 and slot 47's w8, both
since measured at
`0xFFFFFFFCFFFFFFFC`) and
block29's at the low-dword form (slot 50's w4, since measured at
`0x00000000FFFFFFFC`). The
slot 43
(`cc_split_swin_16h_proj_512_chained_fp8` — block27's `qkv→proj` pair's
second half at grid (8,3,1), block (32,8,1)) is now
replayed byte-for-byte and confirms every word-map, record-binding,
sync-extent and constant prediction (the one miss is the creation block:
predicted (32,4,1), the s43 trace measures (32,8,1) — the same created
handle as slots 27/31/35/39). Its parameter block
(psize 72) is the slot-27/31/35/39 word map exactly (pool, pool, pool,
weights, constant, zero, pool, pool, zero): w2 output (pool+31,521,792 — one
stride past its input, the regular output spacing continuing), w4 constant
**`0x1C00000018`**, w6 sync (pool+58,368), w7 sync-in (slot 42's record at
57,856) and **w8 measured reserved zero** (block27's pair second constant
absent, its `qkv` half's w4 likewise). Its weights
word lands on the
layout walk's next record start (`block27.layer3.layer`, 263,168 B at heap
offset 16,313,344 — 15,395,328 + 917,568 + 448 at the
alignment boundary, the layer2→3 alignment-boundary pattern of blocks 23–26
repeating), its w0
is slot
42's w1 output (31,177,728) and its w1 residual is the buffer slot 41 wrote
its output to (30,833,664 — the input slot 42 itself reads). The sync
extent measured **96 B** (the proj norm), pinned
from the s42→s43 write-set
and enforced as the sibling's divergence (the sync record at w6, 58,368 —
the 512-byte stride continuing), stable across all three frames. The
slot-44 `ffwd` half (block28's, `cc_split_swin_16h_ffwd_512_chained_fp8` at
grid (4,3,2), block (32,8,1)) is now replayed byte-for-byte and confirms
every word-map, record-binding, sync-extent and constant prediction. Its
parameter block (psize 56) is the slot-28/32/36/40 `ffwd` word map exactly
(POOL, POOL, WEIGHTS, CONSTANT, POOL, ZERO, POOL): w0 input (31,521,792 —
address-equal to slot 43's w2 output), w1 output (31,865,856 — one stride
past its input, the regular output spacing), w2 weights, w3 the (24, 28)
geometry constant `0x1C00000018`, w4 sync-in (58,368 — address-equal to
slot 43's w6, the chained handoff), w5 reserved zero (every `ffwd` half
reserves the word; the `ffwd→proj` pair carries no second constant) and w6
sync-out (58,880 — the 512-byte stride continuing: 57,856 → 58,368 → 58,880).
Its weight record `block28.layer0.layer` (524,288 B at heap offset 16,576,512
— 16,313,344 + 263,168 at the 512 boundary, the block24/26 layer3→next-block
pattern repeating) is confirmed by the probe's live DLL walk. The sync extent
measured **96 B** (the `ffwd` norm), pinned from the s43→s44 write-set and
enforced as the sibling's divergence (the fp8 control's pool differs from the
reference by exactly 96 bytes — the sync record — on all three frames), stable
across all three frames. The execute pair is bit-identical (the a/b runs share
one run_sha256). The
slot-45 `ffwd_proj` half (block28's,
`cc_split_swin_16h_ffwd_proj_512_chained_fp8` at grid (8,3,1), block (32,4,1))
is now replayed byte-for-byte and confirms the slot-25/29/33/37/41 word map,
record-binding, sync-extent and constant prediction. Its parameter block
(psize 72) is (POOL, POOL, POOL, WEIGHTS, CONSTANT, ZERO, POOL, POOL, ZERO):
w0 input (31,865,856 — address-equal to slot 44's w1), w1 residual
(31,521,792 — the buffer slot 43 wrote its output to, the input slot 44
itself read), w2 output, w4 the (24, 28) geometry constant `0x1C00000018`,
w5/w8 reserved zero (the `ffwd-proj` pair carries no second constant —
confirmed in the s45 capture), w6 sync-out (59,392 — the 512-byte stride
continuing: 58,880 → 59,392) and w7 sync-in (58,880 — address-equal to slot
44's w6, the chained handoff). Its weight record `block28.layer1.layer`
(263,168 B at heap offset 17,100,800 — 16,576,512 + 524,288 at the 512
boundary) is confirmed by the probe's live DLL walk. The sync extent measured
**96 B** (the `ffwd-proj` norm), pinned from the s44→s45 write-set and enforced
as the sibling's divergence (the fp8 control's pool differs from the reference
by exactly 96 bytes on all three frames). The execute pair is bit-identical.
The slot-46 `qkv` half (block28's,
`cc_split_swin_16h_qkv_512_chained_fp8` at grid (4,4,4), the shared qkv
handle's second geometry — the slot-30/38 counterpart) is now replayed
byte-for-byte and confirms the slot-26/30/34/38/42 word map,
record-binding, sync-extent and constant prediction. Its parameter block
(psize 56) is (POOL, POOL, WEIGHTS, CONSTANT, CONSTANT, POOL, POOL):
w0 input (32,209,920 — address-equal to slot 45's w2), w1 output
(32,553,984 — one stride past its input, the regular output spacing), w3
the (24, 28) geometry constant `0x1C00000018`, w4 the pair second constant
at **`0xFFFFFFFCFFFFFFFC`** — the double form, the absent/double/low/high
cycle's block28 step measured in the qkv half (slot 47's w8, the
pair's second half, is measured below), w5 sync-in (59,392 — address-equal to slot
45's w6, the chained handoff) and w6 sync-out (59,904 — the 512-byte
stride continuing: 58,880 → 59,392 → 59,904). Its weight record
`block28.layer2.layer` (917,568 B at heap offset 17,363,968 — 17,100,800
+ 263,168, already 512-aligned so no padding intervenes) is confirmed by
the probe's live DLL walk. The sync extent measured **256 B** (the
grid-4,4,4 qkv norm of slots 30/38 — slot 30's extent recurring at the
second geometry, where the grid-4,3,4 slots 26/34/42 carry 192), pinned
from the s45→s46 write-set (the first changed range, 59,904–60,160, on all
three frames) and enforced as the sibling's divergence (the fp8 control's
pool differs from the reference by exactly 256 bytes on all three frames).
The execute pair is bit-identical.
The
slot-47 `proj` half (block28's, `cc_split_swin_16h_proj_512_chained_fp8`
at grid (8,3,1), block (32,8,1)) is now replayed byte-for-byte and confirms
every word-map, record-binding, sync-extent and constant prediction. Its
parameter block (psize 72) is the slot-27/31/35/39/43 word map exactly
(POOL, POOL, POOL, WEIGHTS, CONSTANT, ZERO, POOL, POOL, CONSTANT): w0
output-of-slot-46 (pool+32,553,984 — address-equal to slot 46's w1), w1
residual (pool+32,209,920 — the buffer slot 46 itself read at w0), w2
output (pool+32,898,048 — one stride past its input, the regular output
spacing; slot 48's w0 is address-equal), w4 the (24, 28) geometry constant
`0x1C00000018`, **w5 reserved zero** (every proj half reserves the word),
w6 sync-out (pool+60,416 — the 512-byte stride continuing: 59,904 →
60,416), w7 sync-in (pool+59,904 — address-equal to slot 46's w6, the
chained handoff) and **w8 the pair second constant at
`0xFFFFFFFCFFFFFFFC`** — the double form. The absent/double/low/high
cycle's predicted block28 step is now measured in **both halves** (slot
46's w4 carries the same value), so the pair shares one constant; block29's
pair then took the cycle's low-dword step at slot 50 (w4 since measured at
`0x00000000FFFFFFFC`), slot 51 testing the proj half. Its weights word
lands on the layout walk's next record start (`block28.layer3.layer`,
263,168 B at heap offset 18,281,984 — 17,363,968 + 917,568 + 448 at the
alignment boundary, the layer2→3 pattern of blocks 23–27 repeating),
confirmed by the probe's live DLL walk. The sync extent measured **96 B**
(the proj norm, returning from slot 46's 256-byte qkv record), pinned from
the s46→s47 write-set (two segments per frame, zero unattributed:
340,714/340,511/340,522 tensor + 96 sync) and enforced as the sibling's
divergence (the fp8 control's pool differs from the reference by exactly
96 bytes on all three frames). The execute pair is bit-identical.
The slot-48 `ffwd` half (block29's,
`cc_split_swin_16h_ffwd_512_chained_fp8` at grid (4,3,2), block (32,8,1)) is
now replayed byte-for-byte and confirms every word-map, record-binding and
sync-extent prediction. Its parameter block (psize 56) is the
slot-28/32/36/40/44 `ffwd` word map exactly (POOL, POOL, WEIGHTS, CONSTANT,
POOL, ZERO, POOL): w0 input (pool+32,898,048 — address-equal to slot 47's
w2), w1 output (pool+33,242,112 — one stride past its input, the regular
output spacing), w2 weights, w3 the (24, 28) geometry constant
`0x1C00000018`, w4 sync-in (pool+60,416 — address-equal to slot 47's w6,
the chained handoff), w5 reserved zero (every `ffwd` half reserves the word;
the `ffwd→proj` pair carries no second constant) and w6 sync-out
(pool+60,928 — the 512-byte stride continuing: 59,904 → 60,416 → 60,928).
Its weights word lands on the layout walk's next record start
(`block29.layer0.layer`, 524,288 B at heap offset 18,545,152 — 18,281,984 +
263,168, block28's layer3 closing exactly at the 512 boundary so no padding
intervenes), confirmed by the probe's live DLL walk. The sync extent
measured **96 B** (the `ffwd` norm), pinned from the s47→s48 write-set (two
segments per frame, zero unattributed: 340,937/341,476/341,433 tensor + 96
sync) and enforced as the sibling's divergence (the fp8 control's pool
differs from the reference by exactly 96 bytes on all three frames). The
execute pair is bit-identical (run_sha256 `5460b105…`).
The slot-49 `ffwd_proj` half (block29's,
`cc_split_swin_16h_ffwd_proj_512_chained_fp8` at grid (8,3,1), block
(32,4,1)) is now replayed byte-for-byte and confirms every word-map,
record-binding and sync-extent prediction. Its parameter block (psize 72)
is the slot-25/29/33/37/41/45 `ffwd_proj` word map exactly (POOL, POOL,
POOL, WEIGHTS, CONSTANT, ZERO, POOL, POOL, ZERO): w0 input
(pool+33,242,112 — address-equal to slot 48's w1), w1 residual
(pool+32,898,048 — the buffer slot 47 wrote its output to, the input slot
48 itself read), w2 output (pool+33,586,176 — one stride past its input,
the regular output spacing; slot 50's w0 is address-equal), w4 the (24, 28)
geometry constant `0x1C00000018`, **w5/w8 reserved zero** (the
`ffwd→proj` pair carries no second constant), w6 sync-out (pool+61,440 —
the 512-byte stride continuing: 59,904 → 60,416 → 60,928 → 61,440) and w7
sync-in (pool+60,928 — address-equal to slot 48's w6, the chained
handoff). Its weights word lands on the layout walk's next record start
(`block29.layer1.layer`, 263,168 B at heap offset 19,069,440 — 18,545,152 +
524,288, block29's layer0 closing exactly at the 512 boundary so no padding
intervenes), confirmed by the probe's live DLL walk. The sync extent
measured **96 B** (the proj norm), pinned from the s48→s49 write-set (two
segments per frame, zero unattributed: 340,596/340,587/340,540 tensor + 96
sync) and enforced as the sibling's divergence (the fp8 control's pool
differs from the reference by exactly 96 bytes on all three frames). The
execute pair is bit-identical (run_sha256 `2b8d0235…`).
The slot-50 `qkv` half (block29's,
`cc_split_swin_16h_qkv_512_chained_fp8` at grid (4,3,4), block
(32,4,1)) is now replayed byte-for-byte and confirms every word-map,
record-binding and sync-extent prediction. Its parameter block (psize 56)
is the qkv family's word map exactly (POOL, POOL, WEIGHTS, CONSTANT,
CONSTANT, POOL, POOL — the qkv shape's w4 as a second constant, the
slot-30/34/38/46 form; slots 26/42 reserve w4 as zero): w0 input
(pool+33,586,176 — address-equal to slot 49's w2), w1 output (pool+33,930,240 — one stride past its input,
the regular output spacing), w2 weights, w3 the (24, 28) geometry constant
`0x1C00000018` and **w4 the pair's second constant at
`0x00000000FFFFFFFC`** — the low-dword form the stage-41 note predicted
(the absent/double/low/high cycle step: block23 absent / 24 double / 25 low
/ 26 high / 27 absent / 28 double / **29 low** now measured), w5 sync-in
(pool+61,440 — address-equal to slot 49's w6, the chained handoff) and w6
sync-out (pool+61,952 — the 512-byte stride continuing: 59,904 → 60,416 →
60,928 → 61,440 → 61,952). Its weights word lands on the layout walk's next
record start (`block29.layer2.layer`, 917,568 B at heap offset 19,332,608 —
the qkv record size),
confirmed by the probe's live DLL walk. The sync extent measured **192 B**
(span 61,952 → 62,144, the qkv 192/256 alternation continuing: 26/34/42/50
at 192, 30/38/46 at 256), pinned from the s49→s50 write-set (two segments
per frame, zero unattributed: 340,638/340,904/340,918 tensor + 192 sync)
and enforced as the sibling's divergence (the fp8 control's pool differs
from the reference by exactly 192 bytes on all three frames). The execute
pair is bit-identical (run_sha256 `c496ee13…`).
The
slot-51 `proj` half (block29's,
`cc_split_swin_16h_proj_512_chained_fp8` at grid (8,3,1), block (32,8,1)) is
now replayed byte-for-byte and confirms every word-map, record-binding,
sync-extent and constant prediction. Its parameter block (psize 72) is the
slot-31/35/39/47 carrying `proj` word map exactly (POOL, POOL, POOL, WEIGHTS,
CONSTANT, ZERO, POOL, POOL, CONSTANT — slot 27/43's shape with w8 a second
constant where those reserve the word zero): w0 input (33,930,240 —
address-equal to slot 50's w1), w1 residual (33,586,176 — the buffer slot
49 wrote its output to, the input slot 50 itself read), w2 output
(34,274,304 — one stride past its input, the regular output spacing), w4
the (24, 28) geometry constant `0x1C00000018`, **w5 reserved zero** and
**w8 the pair's second constant at `0x00000000FFFFFFFC`** — the low-dword
form the stage-42 note predicted, now measured in both halves of block29's
pair (slot 50's w4 and slot 51's w8 alike), w6 sync-out (62,464 — the
512-byte stride continuing: 60,928 → 61,440 → 61,952 → 62,464) and w7
sync-in (61,952 — address-equal to slot 50's w6, the chained handoff). Its
weights word lands on the layout walk's next record start
(`block29.layer3.layer`, 263,168 B at heap offset 20,250,624 —
19,332,608 + 917,568 + 448 at the alignment boundary, layer2's 448-byte
tail padding), confirmed by the probe's live DLL walk. The sync extent
measured **96 B** (the proj norm), pinned from the s50→s51 write-set (two
segments per frame, zero unattributed: 340,556/340,608/340,578 tensor + 96
sync) and enforced as the sibling's divergence (the fp8 control's pool
differs from the reference by exactly 96 bytes on all three frames). The
execute pair is bit-identical (run_sha256 `f6ff439e…`). With block29's
pair closing at the same low-dword value in both halves, the
absent/double/low/high cycle is now measured complete over blocks 23–29
(23 absent / 24 double / 25 low / 26 high / 27 absent / 28 double / 29 low
— the repeating four-step pattern over seven pairs: absent 23/27, double
24/28, low 25/29, high 26).
The
slot-52 `ffwd` half (block30's,
`cc_split_swin_16h_ffwd_512_chained_fp8` at grid (4,3,2), block (32,8,1)) is
now replayed byte-for-byte and confirms every word-map, record-binding and
sync-extent prediction. Its parameter block (psize 56) is the
slot-28/32/36/40/44/48 `ffwd` word map exactly (POOL, POOL, WEIGHTS,
CONSTANT, POOL, ZERO, POOL — **w5 reserved zero**: every `ffwd` half
reserves the word, the `ffwd→proj` pair carrying no second constant): w0
input (34,274,304 — address-equal to slot 51's w2), w1 output (34,618,368 —
one stride past its input, the regular output spacing), w2 weights, w3 the
(24, 28) geometry constant `0x1C00000018`, w4 sync-in (62,464 —
address-equal to slot 51's w6, the chained handoff) and w6 sync-out
(62,976 — the 512-byte stride continuing: 60,928 → 61,440 → 61,952 →
62,464 → 62,976). Its weights word lands on the layout walk's
`block30.layer0.layer` record start (524,288 B at heap offset 20,534,784)
— past the intervening `block3.layer0.layer` record (20,672 B at
20,513,792, 320 bytes of tail padding to the alignment boundary), so the
stage-51 note's naive +263,168 successor arithmetic lands on that small
record, not block30's first; confirmed by the probe's live DLL walk. The
sync extent measured **96 B** (the ffwd norm), pinned from the s51→s52
write-set (two segments per frame, zero unattributed:
340,808/341,273/341,295 tensor + 96 sync) and enforced as the sibling's
divergence (the fp8 control's pool differs from the reference by exactly 96
bytes on all three frames). The execute pair is bit-identical (run_sha256
`c6e24122…`).
The natural next candidate is slot 53
(`cc_split_swin_16h_ffwd_proj_512_chained_fp8` — block30's `ffwd→proj`
pair's second half at grid (8,3,1), the slot-25/29/33/37/41/45/49 word
map); slot 53 tests the layout walk's next record start
(`block30.layer1.layer`, 263,168 B at heap offset 21,059,072 —
20,534,784 + 524,288 at the 512 boundary) and the chained handoff's next
step (w7 sync-in at 62,976 — address-equal to slot 52's w6; w6 sync-out at
63,488, the 512-byte stride continuing), with no pair second constant
(every `ffwd→proj` half reserves the words). The interval's tail then
departs from the four-kernel block shape: the inventory's slot 55 is a
`proj_pool` variant (`cc_split_swin_16h_proj_pool_512_chained_fp8`) and
slot 56 the `final_head` (`cc_split_swin_16h_final_head_512_wait_fp8`)
closing the transformer interval at the s56 cut. The cycle's continuation
would predict the slots 54/55 pair at the high-dword form
`0xFFFFFFFC00000000` (block30 the cycle's "high" step) — measured over
blocks 23–29, a hypothesis beyond. The
temporal/history state (`prev_output`, counter) then connects the kernels
into a full native evaluation for the native-vs-vendor comparison of the
whole pipeline.
