#!/usr/bin/env python3
"""Native pre-decoder kernel replay: slots 24-84 against ladder references.

Generalises the decoder replay (NATIVE-VENDOR-COMPARE.md) to pre-decoder
kernels: vendor launches of the 156-launch inventory (NVAPI-ARGS.md) are
replayed natively on Linux (CUDA driver; no Windows code, DirectX or Proton)
from their captured parameter blocks, and the native pool state is compared
byte-for-byte against the vendor's own captured state at the matching ladder
cut point (LADDER.md). Sixty-one per-slot contracts are covered:

- slot 24 `cc_split_swin_16h_ffwd_inpview_512_tilesync_fp8` (s23 -> s24):
  reads w0, writes its output tensor at w1 (344,064 bytes) and a 96-byte
  sync record at w6 — the stage-16 claim;
- slot 25 `cc_split_swin_16h_ffwd_proj_inpview_512_chained_fp8`
  (s24 -> s25): the ffwd-proj pair's second half. It reads slot 24's
  output (w0) and a residual input (w1), consumes slot 24's sync record
  (w7 — the chained handoff), writes its output tensor at w2 (344,064
  bytes) and a 96-byte sync record at w6. The write-set attribution pins
  exactly those two words (names-s24-s25.txt: w2 + w6, zero unattributed);
- slot 26 `cc_split_swin_16h_qkv_512_chained_fp8` (s25 -> s26): the
  transformer block's qkv half. It reads slot 25's output (w0), consumes
  slot 25's sync record (w5), writes its output tensor at w1 (344,064
  bytes) and a sync record at w6 whose changed extent is 192 bytes
  (names-s23-s56.txt, stable across all three frames). The kernel handle
  is the `qkv_512` family's shared handle — all sixteen qkv launches of
  both transformer intervals (slots 26/30/34/38/42/46/50/54 and
  102/106/110/114/118/122/126/130) use it at exactly two geometries (grid
  4,3,4 here and at slots 34/42/50/102/110/118/126, 4,4,4 at slots
  30/38/46/54/106/114/122/130), so the grid is pinned at the slot's own launch position;
- slot 27 `cc_split_swin_16h_proj_512_chained_fp8` (s26 -> s27): the
  transformer block's proj half (the qkv-proj pair's second half). It
  reads slot 26's output (w0) and a residual input (w1, the address slot
  25 wrote its output to), consumes slot 26's sync record (w7 — the
  chained handoff, address-equal to slot 26's w6), writes its output
  tensor at w2 (344,064 bytes) and a 96-byte sync record at w6. The
  write-set attribution pins exactly those two words (names-s26-s27.txt:
  w2 + w6, zero unattributed, stable across all three frames);
- slot 28 `cc_split_swin_16h_ffwd_512_chained_fp8` (s27 -> s28): the next
  block's ffwd half — the first kernel outside `block23`. It reads slot
  27's output (w0 at 21,642,240 — address-equal to slot 27's w2),
  consumes slot 27's sync record (w4 at 50,176 — address-equal to slot
  27's w6), writes its output tensor at w1 (344,064 bytes) and a 96-byte
  sync record at w6. The write-set attribution pins exactly those two
  words (names-s27-s28.txt: w1 + w6, zero unattributed, stable across all
  three frames);
- slot 29 `cc_split_swin_16h_ffwd_proj_512_chained_fp8` (s28 -> s29): the
  block24 ffwd-proj pair's second half (the `ffwd_proj` counterpart to
  slot 25's, same 72-byte block and grid 8,3,1). It reads slot 28's
  output (w0 at 21,986,304 — address-equal to slot 28's w1) and a
  residual input (w1 at 21,642,240 — the buffer slot 27 wrote its output
  to), consumes slot 28's sync record (w7 at 50,688 — address-equal to
  slot 28's w6, the chained handoff), writes its output tensor at w2
  (344,064 bytes at 22,330,368) and a 96-byte sync record at w6 (51,200
  — the 512-byte stride continuing). The write-set attribution pins
  exactly those two words (names-s28-s29.txt: w2 + w6, zero unattributed,
  stable across all three frames). It binds `block24.layer1.layer`.
- slot 30 `cc_split_swin_16h_qkv_512_chained_fp8` (s29 -> s30): the
  block24 qkv half — slot 26's counterpart at grid 4,4,4, the shared
  handle's second geometry now exercised. It reads slot 29's output (w0
  at 22,330,368 — address-equal to slot 29's w2) and consumes slot 29's
  sync record (w5 at 51,200 — address-equal to slot 29's w6), writes its
  output tensor at w1 (344,064 bytes at 22,674,432) and a sync record at
  w6 (51,712 — the 512-byte stride continuing) whose changed extent is
  256 bytes (names-s29-s30.txt, stable across all three frames). The
  write-set attribution pins exactly those two words (w1 + w6, zero
  unattributed). It binds `block24.layer2.layer` (917,568 B, the record
  immediately after `block24.layer1.layer`). Its parameter block carries
  two constant words at two distinct values: w3 the (24, 28) geometry
  constant and w4 0xFFFFFFFCFFFFFFFC, where the same-shape slot 26
  reserves its w4 as zero.
- slot 31 `cc_split_swin_16h_proj_512_chained_fp8` (s30 -> s31): the
  block24 qkv-proj pair's second half (the `proj` counterpart to slot
  27's, same 72-byte block). It reads slot 30's output (w0 at 22,674,432
  — address-equal to slot 30's w1) and a residual input (w1 at
  22,330,368 — the buffer slot 29 wrote its output to), consumes slot
  30's sync record (w7 at 51,712 — address-equal to slot 30's w6, the
  chained handoff), writes its output tensor at w2 (344,064 bytes at
  23,018,496) and a 96-byte sync record at w6 (52,224 — the 512-byte
  stride continuing). The write-set attribution pins exactly those two
  words (names-s30-s31.txt: w2 + w6, zero unattributed, stable across
  all three frames). It binds `block24.layer3.layer` (263,168 B at heap
  offset 10,407,424 — the record immediately after `block24.layer2.layer`,
  the record-ordering walk holding through the pair). Its parameter block
  carries two constant words at two distinct values like its pair partner
  slot 30: w4 the (24, 28) geometry constant and w8 0xFFFFFFFCFFFFFFFC,
  where the same-shape slot 27 reserves its w8 as zero — the block24
  qkv-proj pair carries that constant in both halves, block23's pair
  carries neither.
- slot 32 `cc_split_swin_16h_ffwd_512_chained_fp8` (s31 -> s32): the
  block25 ffwd half — the next `ffwd->proj` pair's first half. It reads
  slot 31's output (w0 at 23,018,496 — address-equal to slot 31's w2) and
  consumes slot 31's sync record (w4 at 52,224 — address-equal to slot
  31's w6, the chained handoff at the ffwd shape's w4 position, like slot
  28), writes its output tensor at w1 (344,064 bytes at 23,362,560) and a
  96-byte sync record at w6 (52,736 — the 512-byte stride continuing).
  The write-set attribution pins exactly those two words
  (names-s31-s32.txt: w1 + w6, zero unattributed, stable across all three
  frames). It binds `block25.layer0.layer` (524,288 B — the record
  directly after `block24.layer3.layer`, the record-ordering walk
  continuing into block25). Its parameter block is the slot-28 word
  map exactly (no second constant: w5 is reserved zero).
- slot 33 `cc_split_swin_16h_ffwd_proj_512_chained_fp8` (s32 -> s33):
  the block25 ffwd-proj pair's second half — the `ffwd_proj` counterpart
  to slots 25 and 29 (same 72-byte block and grid 8,3,1). It reads slot
  32's output (w0 at 23,362,560) and a residual input (w1 at 23,018,496,
  the buffer slot 31 wrote its output to), consumes slot 32's sync record
  (w7 at 52,736 — address-equal to slot 32's w6, the chained handoff) and
  writes its output tensor at w2 (344,064 bytes at 23,706,624) plus a
  96-byte sync record at w6 (53,248 — the 512-byte stride continuing).
  The write-set attribution pins exactly those two words
  (names-s32-s33.txt: w2 + w6, zero unattributed, stable across all
  three frames). It binds `block25.layer1.layer` (263,168 B at heap
  offset 11,194,880 — the record directly after `block25.layer0.layer`,
  the record-ordering walk continuing through block25's pair). Its
  parameter block is slot 25's word map exactly (as at slot 29; w8
  reserved zero: the
  0xFFFFFFFCFFFFFFFC value does not recur in block25's ffwd->proj
  pair; slots 34/35 below show the qkv-proj pair carries its second
  constant in both halves at one value per pair — block24's at
  0xFFFFFFFCFFFFFFFC, block25's at 0x00000000FFFFFFFC).
- slot 34 `cc_split_swin_16h_qkv_512_chained_fp8` (s33 -> s34): the
  block25 qkv half — slot 26's counterpart, the `qkv->proj` pair's first
  half. It reads slot 33's output (w0 at 23,706,624 — address-equal to
  slot 33's w2) and consumes slot 33's sync record (w5 at 53,248 —
  address-equal to slot 33's w6, the chained handoff at the qkv shape's
  w5 position), writes its output tensor at w1 (344,064 bytes at
  24,050,688) and a sync record at w6 (53,760 — the 512-byte stride
  continuing) whose changed extent is 192 bytes (names-s33-s34.txt,
  stable across all three frames). The write-set attribution pins exactly
  those two words (w1 + w6, zero unattributed). It binds
  `block25.layer2.layer` (917,568 B at heap offset 11,458,048 — the
  record directly after `block25.layer1.layer`, the record-ordering walk
  holding through block25's pair). The kernel handle is the shared
  `qkv_512` handle (the sixteen qkv launches at the two geometries in the
  slot-26 note), so the grid is resolved at the slot's own launch
  position (4,3,4 here, the slot-26 geometry; slot 30's was 4,4,4). Its
  parameter block carries two constant words: w3 the (24, 28) geometry
  constant and w4 0x00000000FFFFFFFC — the low dword only, where slot
  30's same-shape w4 is 0xFFFFFFFCFFFFFFFC and slot 26's is zero. This
  pair's second constant takes one value per pair, differing between
  block24's and block25's pairs (slot 35's w8 below carries the same
  value as this w4).
- slot 35 `cc_split_swin_16h_proj_512_chained_fp8` (s34 -> s35): the
  block25 qkv-proj pair's second half — the `proj` counterpart to slots
  27 and 31 (same 72-byte block and grid 8,3,1). It reads slot 34's
  output (w0 at 24,050,688 — address-equal to slot 34's w1) and a
  residual input (w1 at 23,706,624 — the buffer slot 33 wrote its output
  to), consumes slot 34's sync record (w7 at 53,760 — address-equal to
  slot 34's w6, the chained handoff) and writes its output tensor at w2
  (344,064 bytes at 24,394,752) plus a 96-byte sync record at w6
  (54,272 — the 512-byte stride continuing: 52,224 -> 52,736 -> 53,248
  -> 53,760 -> 54,272). The write-set attribution pins exactly those two
  words (names-s34-s35.txt: w2 + w6, zero unattributed, stable across all
  three frames). It binds `block25.layer3.layer` (263,168 B at heap
  offset 12,376,064 — the record directly after `block25.layer2.layer`,
  at the alignment boundary below layer2's unaligned payload (448 bytes
  of padding), the block23/24 layer2->3 pattern repeating), so the
  record-ordering walk holds through block25's pair. Its parameter block
  is slot 31's word map with w8 at block25's pair value: w4 the (24, 28)
  constant and w8 0x00000000FFFFFFFC — the same value as its pair
  partner slot 34's w4, so the qkv-proj pair carries the second constant
  in both halves at one value per pair (block24's pair at
  0xFFFFFFFCFFFFFFFC, block25's at 0x00000000FFFFFFFC); block23's pair
  and every ffwd-proj half reserve the word zero.
- slot 36 `cc_split_swin_16h_ffwd_512_chained_fp8` (s35 -> s36): the
  block26 ffwd half — the next `ffwd->proj` pair's first half. It reads
  slot 35's output (w0 at 24,394,752 — address-equal to slot 35's w2) and
  consumes slot 35's sync record (w4 at 54,272 — address-equal to slot
  35's w6, the chained handoff at the ffwd shape's w4 position, like slots
  28 and 32), writes its output tensor at w1 (344,064 bytes at
  24,738,816) and a 96-byte sync record at w6 (54,784 — the 512-byte
  stride continuing). The write-set attribution pins exactly those two
  words (names-s35-s36.txt: w1 + w6, zero unattributed, stable across all
  three frames). It binds `block26.layer0.layer` (524,288 B at heap
  offset 12,639,232 — the record directly after `block25.layer3.layer`,
  the record-ordering walk continuing into block26). Its parameter block
  is the slot-28/32 word map exactly (w5 reserved zero: an ffwd half
  carries no pair second constant). The kernel handle is the `ffwd_512`
  family's shared handle — all fifteen ffwd launches of both transformer
  intervals (slots 28/32/36/40/44/48/52 and
  100/104/108/112/116/120/124/128) use it at exactly one geometry (grid
  4,3,2 — unlike the qkv family's two), so per-handle geometry is unique
  within this family but the grid is still resolved at the slot's own
  launch position.
- slot 37 `cc_split_swin_16h_ffwd_proj_512_chained_fp8` (s36 -> s37):
  the block26 ffwd-proj pair's second half — the `ffwd_proj` counterpart
  to slots 25, 29 and 33 (same 72-byte block and grid 8,3,1). It reads
  slot 36's output (w0 at 24,738,816 — address-equal to slot 36's w1) and
  a residual input (w1 at 24,394,752 — the buffer slot 35 wrote its output
  to), consumes slot 36's sync record (w7 at 54,784 — address-equal to
  slot 36's w6, the chained handoff) and writes its output tensor at w2
  (344,064 bytes at 25,426,944) plus a 96-byte sync record at w6 (55,296
  — the 512-byte stride continuing). The write-set attribution pins
  exactly those two words (names-s36-s37.txt: w2 + w6, zero unattributed,
  stable across all three frames). It binds `block26.layer1.layer`
  (263,168 B at heap offset 13,163,520 — the record immediately after
  `block26.layer0.layer`, the record-ordering walk holding through
  block26's pair). Its parameter block is the slot-25/29/33 word map
  exactly (w5 and w8 reserved zero — an ffwd-proj half carries no pair
  second constant). The one measured oddity is its output spacing: w2 sits
  two tensor strides past its w0 (25,426,944 vs 24,738,816), where slots
  25/29/33's w2 sat one stride past their w0; the intervening region
  (25,082,880) is the encoder-residual buffer the names tool tracks as
  slot 23's w9 / slot 24's w0. What that spacing encodes is not yet
  established; the replay reproduces it as captured.
- slot 38 `cc_split_swin_16h_qkv_512_chained_fp8` (s37 -> s38): the
  block26 qkv half — the qkv-proj pair's first half at the shared handle's
  second geometry (grid 4,4,4, the slot-30/34 counterpart). It reads slot
  37's output (w0 at 25,426,944 — address-equal to slot 37's w2) and
  consumes slot 37's sync record (w5 at 55,296 — address-equal to slot
  37's w6, the chained handoff), writes its output tensor at w1 (344,064
  bytes at 25,771,008) plus a sync record at w6 (55,808 — the 512-byte
  stride continuing) whose changed extent is 256 bytes
  (names-s37-s38.txt, stable across all three frames). The write-set
  attribution pins exactly those two words (w1 + w6, zero unattributed).
  It binds `block26.layer2.layer` (917,568 B at heap offset 13,426,688 —
  the record immediately after `block26.layer1.layer`, the record-ordering
  walk continuing through block26's qkv-proj pair). Its parameter block is
  the slot-30/34 word map exactly (the qkv shape's w4 as a second constant)
  and answers the pair-constant geometry question: w4 is
  0xFFFFFFFC00000000 — block26's pair value at the high dword only, the
  third geometry of the marker dword after block24's double
  (0xFFFFFFFCFFFFFFFC) and block25's low-dword-only (0x00000000FFFFFFFC)
  forms, one value per pair.

- slot 39 `cc_split_swin_16h_proj_512_chained_fp8` (s38 -> s39): the
  block26 proj half — the qkv-proj pair's second half at grid (8,3,1), block
  (32,8,1) (the slot-27/31/35 counterpart). It reads slot 38's output (w0 at
  25,771,008 — address-equal to slot 38's w1) and the residual input (w1 at
  25,426,944 — the buffer slot 37 wrote its output to), consumes slot 38's
  sync record (w7 at 55,808 — address-equal to slot 38's w6, the chained
  handoff), writes its output tensor at w2 (344,064 bytes at 26,115,072)
  plus a sync record at w6 (56,320 — the 512-byte stride continuing) whose
  changed extent is 96 bytes
  (names-s38-s39.txt, stable across all three frames). The write-set
  attribution pins exactly those two words (w2 + w6, zero unattributed).
  It binds `block26.layer3.layer` (263,168 B at heap offset 14,344,704 —
  the record immediately after `block26.layer2.layer` at the 448-byte
  alignment boundary, the record-ordering walk completing block26). Its
  parameter block is the slot-27/31/35 word map exactly and confirms the
  pair-constant sharing in both halves: w8 is
  0xFFFFFFFC00000000 — identical to its pair partner slot 38's w4 (block26's
  pair value, the high-dword-only form), so the sharing is now seen in both
  halves of block24's, block25's and block26's pairs.

- slot 40 `cc_split_swin_16h_ffwd_512_chained_fp8` (s39 -> s40): the
  block27 ffwd half — the ffwd-proj pair's first half at grid (4,3,2), block
  (32,8,1) (the slot-28/32/36 counterpart). It reads slot 39's output (w0 at
  26,115,072 — address-equal to slot 39's w2) and consumes slot 39's sync
  record (w4 at 56,320 — address-equal to slot 39's w6, the chained handoff),
  writes its output tensor at w1 (344,064 bytes at 26,459,136) plus a sync
  record at w6 (56,832 — the 512-byte stride continuing) whose changed
  extent is 96 bytes (names-s39-s40.txt, stable across all three frames).
  The write-set attribution pins exactly those two words (w1 + w6, zero
  unattributed). It binds `block27.layer0.layer` (524,288 B at heap offset
  14,607,872 — the record immediately after `block26.layer3.layer` at the
  512 boundary, the record-ordering walk opening block27). Its parameter
  block is the slot-28/32/36 word map exactly (pool, pool, weights, constant,
  pool, zero, pool) with w3 = 0x1C00000018 and no pair second constant
  (absent from every ffwd-proj half).
- slot 41 `cc_split_swin_16h_ffwd_proj_512_chained_fp8` (s40 -> s41): the
  block27 ffwd-proj pair's second half (the `ffwd_proj` counterpart to
  slots 25/29/33/37, same 72-byte block and grid 8,3,1). It reads slot
  40's output (w0 at 26,459,136 — address-equal to slot 40's w1) and a
  residual input (w1 at 26,115,072 — the buffer slot 39 wrote its output
  to, address-equal to slot 40's w0), consumes slot 40's sync record (w7
  at 56,832 — address-equal to slot 40's w6, the chained handoff), writes
  its output tensor at w2 (344,064 bytes at 30,833,664) plus a sync
  record at w6 (57,344 — the 512-byte stride continuing) whose changed
  extent is 96 bytes (names-s40-s41.txt, stable across all three frames).
  The write-set attribution pins exactly those two words (w2 + w6, zero
  unattributed). It binds `block27.layer1.layer` (263,168 B at heap
  offset 15,132,160 — the record immediately after `block27.layer0.layer`
  at the 512 boundary, the record-ordering walk completing block27's
  pair). Its parameter block is the slot-25/29/33/37 word map exactly
  (pool, pool, pool, weights, constant, zero, pool, pool, zero) with
  w4 = 0x1C00000018 and w5/w8 reserved zero (the pair second constant is
  absent from every ffwd-proj half).
- slot 42 `cc_split_swin_16h_qkv_512_chained_fp8` (s41 -> s42): the
  block27 qkv half — the `qkv->proj` pair's first half. It reads slot 41's
  output (w0 at 30,833,664 — address-equal to slot 41's w2) and consumes
  slot 41's sync record (w5 at 57,344 — address-equal to slot 41's w6, the
  chained handoff at the qkv shape's w5 position), writes its output tensor
  at w1 (344,064 bytes at 31,177,728 — one stride past its input, the
  regular spacing resuming after slot 41's offset placement) and a sync
  record at w6 (57,856 — the 512-byte stride continuing) whose changed
  extent is 192 bytes (names-s41-s42.txt, stable across all three frames).
  The write-set attribution pins exactly those two words (w1 + w6, zero
  unattributed). It binds `block27.layer2.layer` (917,568 B at heap offset
  15,395,328 — the record immediately after `block27.layer1.layer`, the
  record-ordering walk completing block27). The kernel handle is the shared
  `qkv_512` handle (the sixteen qkv launches at the two geometries in the
  slot-26 note), so the grid is resolved at the slot's own launch position
  (4,3,4 here, the slot-26/34 geometry). Its parameter block is the slot-26
  word map exactly (pool, pool, weights, constant, zero, pool, pool): w3
  the (24, 28) geometry constant and w4 reserved zero — block27's qkv
  half carries no second constant (its proj half's w8 measured zero in the
  s43 capture), where block24's,
  block25's and block26's pairs each carry one value per pair in both
  halves. The four pairs measured by that stage suggested a repeating
  absent/double/low-dword/high-dword cycle (block23 absent, block24 double,
  block25 low, block26 high) — block27's measured pair (slot 42's w4 and
  slot 43's w8 both reserved zero) completes the cycle's absent step; the cycle
  predicted block28's pair at the
  double-dword form (slot 46's w4 and slot 47's w8, both since measured at
  0xFFFFFFFCFFFFFFFC) and
  block29's at the low-dword form (slot 50's w4 since measured at
  0x00000000FFFFFFFC).

- slot 43 `cc_split_swin_16h_proj_512_chained_fp8` (s42 -> s43): the
  block27 proj half — the qkv-proj pair's second half at grid (8,3,1), block
  (32,8,1) (the slot-27/31/35/39 counterpart). It reads slot 42's output (w0
  at 31,177,728 — address-equal to slot 42's w1) and the residual input (w1
  at 30,833,664 — the buffer slot 41 wrote its output to), consumes slot
  42's sync record (w7 at 57,856 — address-equal to slot 42's w6, the
  chained handoff), writes its output tensor at w2 (344,064 bytes at
  31,521,792) plus a sync record at w6 (58,368 — the 512-byte stride
  continuing) whose changed extent is 96 bytes
  (names-s42-s43.txt, stable across all three frames). The write-set
  attribution pins exactly those two words (w2 + w6, zero unattributed).
  It binds `block27.layer3.layer` (263,168 B at heap offset 16,313,344 —
  the record immediately after `block27.layer2.layer` at the 448-byte
  alignment boundary, the layer2-to-3 alignment-boundary pattern of blocks
  23-26 repeating and
  the record-ordering walk completing block27). Its parameter block is the
  slot-27/31/35/39 word map exactly: w8 is measured reserved zero (its pair
  partner slot 42's w4 likewise), so block27's qkv-proj pair carries no
  second constant — the cycle hypothesis' absent step now measured in both
  halves of block23's and block27's pairs.
- slot 44 `cc_split_swin_16h_ffwd_512_chained_fp8` (s43 -> s44): the
  block28 ffwd half — the next `ffwd->proj` pair's first half. It reads
  slot 43's output (w0 at 31,521,792 — address-equal to slot 43's w2),
  consumes slot 43's sync record (w4 at 58,368 — address-equal to slot
  43's w6, the chained handoff at the ffwd shape's w4 position), writes
  its output tensor at w1 (344,064 bytes at 31,865,856) and a 96-byte
  sync record at w6 (58,880 — the 512-byte stride continuing). The
  write-set attribution pins exactly those two words (names-s43-s44.txt:
  w1 + w6, zero unattributed, stable across all three frames). It binds
  `block28.layer0.layer` (524,288 B at heap offset 16,576,512 — the
  record immediately after `block27.layer3.layer`). Its parameter block
  is the slot-28/32/36/40 word map exactly (w5 reserved zero).
- slot 45 `cc_split_swin_16h_ffwd_proj_512_chained_fp8` (s44 -> s45):
  the block28 ffwd-proj pair's second half — the `ffwd_proj` counterpart
  to slots 25/29/33/37/41 (same 72-byte block and grid 8,3,1). It reads
  slot 44's output (w0 at 31,865,856) and a residual input (w1 at
  31,521,792, the buffer slot 43 wrote its output to), consumes slot
  44's sync record (w7 at 58,880 — address-equal to slot 44's w6, the
  chained handoff) and writes its output tensor at w2 (344,064 bytes at
  32,209,920) plus a 96-byte sync record at w6 (59,392 — the 512-byte
  stride continuing). The write-set attribution pins exactly those two
  words (names-s44-s45.txt: w2 + w6, zero unattributed). It binds
  `block28.layer1.layer` (263,168 B at heap offset 17,100,800). Its
  parameter block is the slot-25/29/33/37/41 word map exactly (w5/w8
  reserved zero: the `ffwd->proj` pair carries no second constant).
- slot 46 `cc_split_swin_16h_qkv_512_chained_fp8` (s45 -> s46): the
  block28 qkv half — the `qkv->proj` pair's first half at the shared
  handle's second geometry (grid 4,4,4, the slot-30/38 counterpart). It
  reads slot 45's output (w0 at 32,209,920 — address-equal to slot 45's
  w2) and consumes slot 45's sync record (w5 at 59,392 — address-equal
  to slot 45's w6, the chained handoff at the qkv shape's w5 position),
  writes its output tensor at w1 (344,064 bytes at 32,553,984) and a
  sync record at w6 (59,904 — the 512-byte stride continuing) whose
  changed extent is 256 bytes (names-s45-s46.txt: w1 + w6, zero
  unattributed — the slot-30/38 extent recurring at the second
  geometry). It binds `block28.layer2.layer` (917,568 B at heap offset
  17,363,968). Its parameter block is the slot-26/30/34/38/42 word map
  exactly, with w4 carrying block28's pair second constant at
  0xFFFFFFFCFFFFFFFC — the double form, the cycle's block28 step measured
  in the qkv half (slot 47's w8, the pair's second half, is measured below).

- slot 47 `cc_split_swin_16h_proj_512_chained_fp8` (s46 -> s47): the
  block28 proj half — the qkv-proj pair's second half at grid (8,3,1), block
  (32,8,1) (the slot-27/31/35/39/43 counterpart). It reads slot 46's output
  (w0 at 32,553,984 — address-equal to slot 46's w1) and the residual input
  (w1 at 32,209,920 — the buffer slot 46 itself read at w0), consumes slot
  46's sync record (w7 at 59,904 — address-equal to slot 46's w6, the
  chained handoff at the proj shape's w7 position), writes its output tensor
  at w2 (344,064 bytes at 32,898,048) and a sync record at w6 (60,416 — the
  512-byte stride continuing) whose changed extent is 96 bytes
  (names-s46-s47.txt: w2 + w6, zero unattributed). It binds
  `block28.layer3.layer` (263,168 B at heap offset 18,281,984). Its
  parameter block is the slot-27/31/35/39/43 word map exactly, with w8
  carrying block28's pair second constant at 0xFFFFFFFCFFFFFFFC — the double
  form, the cycle's block28 step now measured in both halves (slot 46's w4
  alike); block29's pair at the low-dword form (slot 50's w4 since measured
  at 0x00000000FFFFFFFC).
- slot 48 `cc_split_swin_16h_ffwd_512_chained_fp8` (s47 -> s48): the
  block29 ffwd half — the next `ffwd->proj` pair's first half at grid
  (4,3,2), block (32,8,1) (the slot-28/32/36/40/44 counterpart). It reads
  slot 47's output (w0 at 32,898,048 — address-equal to slot 47's w2) and
  consumes slot 47's sync record (w4 at 60,416 — address-equal to slot
  47's w6, the chained handoff at the ffwd shape's w4 position), writes
  its output tensor at w1 (344,064 bytes at 33,242,112 — one stride past
  its input, the regular output spacing) and a sync record at w6 (60,928
  — the 512-byte stride continuing) whose changed extent is 96 bytes
  (names-s47-s48.txt: w1 + w6, zero unattributed, stable across all three
  frames). It binds `block29.layer0.layer` (524,288 B at heap offset
  18,545,152 — the record immediately after `block28.layer3.layer`, which
  closes exactly at the 512 boundary so no padding intervenes, the
  record-ordering walk opening block29). Its parameter block is the
  slot-28/32/36/40/44 word map exactly (pool, pool, weights, constant,
  pool, zero, pool) with w3 = 0x1C00000018 and no pair second constant
  (absent from every ffwd-proj half).
- slot 49 `cc_split_swin_16h_ffwd_proj_512_chained_fp8` (s48 -> s49): the
  block29 ffwd-proj pair's second half (the slot-25/29/33/37/41/45 word
  map, same 72-byte block and grid (8,3,1)). It reads slot 48's output
  (w0 at 33,242,112 — address-equal to slot 48's w1) and a residual input
  (w1 at 32,898,048 — the buffer slot 47 wrote its output to, the input
  slot 48 itself read), consumes slot 48's sync record (w7 at 60,928 —
  address-equal to slot 48's w6, the chained handoff), writes its output
  tensor at w2 (344,064 bytes at 33,586,176) and a sync record at w6
  (61,440 — the 512-byte stride continuing: 60,416 -> 60,928 -> 61,440)
  whose changed extent is 96 bytes (names-s48-s49.txt: w2 + w6, zero
  unattributed, stable across all three frames). It binds
  `block29.layer1.layer` (263,168 B at heap offset 19,069,440 — the
  record immediately after `block29.layer0.layer`, which closes exactly at
  the 512 boundary so no padding intervenes, the record-ordering walk
  continuing through block29's pair). Its parameter block is the
  slot-25/29/33/37/41/45 word map exactly (pool, pool, pool, weights,
  constant, zero, pool, pool, zero) with w5/w8 reserved zero (the
  ffwd-proj pair carries no second constant).
- slot 50 `cc_split_swin_16h_qkv_512_chained_fp8` (s49 -> s50): the
  block29 qkv half — the qkv-proj pair's first half at grid (4,3,4), the
  slot-26/30/34/38/42/46 word map. It reads slot 49's output (w0 at
  33,586,176 — address-equal to slot 49's w2), consumes slot 49's sync
  record (w5 at 61,440 — address-equal to slot 49's w6, the chained
  handoff), writes its output tensor at w1 (344,064 bytes at 33,930,240 —
  one stride past its input) and a sync record at w6 (61,952 — the
  512-byte stride continuing: 60,928 -> 61,440 -> 61,952) whose changed
  extent is 192 bytes (the qkv 192/256 alternation: 26/34/42/50 at 192,
  30/38/46 at 256). It binds `block29.layer2.layer` (917,568 B — the qkv
  record size). Its parameter block is the slot-26/30/34/38/42/46 word map
  exactly (pool, pool, weights, constant, constant, pool, pool) with w4 the
  pair's second constant at `0x00000000FFFFFFFC` — the low-dword form of
  the absent/double/low/high cycle (block23 absent / 24 double / 25 low / 26
  high / 27 absent / 28 double / 29 low).
- slot 51 `cc_split_swin_16h_proj_512_chained_fp8` (s50 -> s51): the
  block29 proj half — the qkv-proj pair's second half, the
  slot-31/35/39/47 carrying word map (slot 27/43's shape with w8 a second
  constant where those reserve the word zero). It reads slot 50's output (w0 at
  33,930,240 — address-equal to slot 50's w1) and a residual input (w1 at
  33,586,176 — the buffer slot 49 wrote its output to, the input slot 50
  itself read), consumes slot 50's sync record (w7 at 61,952 —
  address-equal to slot 50's w6, the chained handoff), writes its output
  tensor at w2 (344,064 bytes at 34,274,304 — one stride past its input,
  the regular output spacing) and a sync record at w6 (62,464 — the
  512-byte stride continuing: 60,928 -> 61,440 -> 61,952 -> 62,464) whose
  changed extent is 96 bytes (the proj norm). It binds `block29.layer3.layer`
  (263,168 B at heap offset 20,250,624 — 19,332,608 + 917,568 rounded up
  to the 512 boundary, the qkv record's tail padding). Its parameter block
  is the slot-31/35/39/47 word map exactly (pool, pool, pool, weights,
  constant, zero, pool, pool, constant) with w8 the pair's second constant
  at `0x00000000FFFFFFFC` — the low-dword form, now measured in both halves
  of block29's pair (slot 50's w4 and slot 51's w8 alike, one value per
  pair shared across its halves).
- slot 52 `cc_split_swin_16h_ffwd_512_chained_fp8` (s51 -> s52): the
  block30 ffwd half — the next `ffwd->proj` pair's first half. It reads
  slot 51's output (w0 at 34,274,304 — address-equal to slot 51's w2),
  consumes slot 51's sync record (w4 at 62,464 — address-equal to slot
  51's w6, the chained handoff at the ffwd shape's w4 position, like slots
  28/32/36/40/44/48), writes its output tensor at w1 (344,064 bytes at
  34,618,368) and a sync record at w6 (62,976 — the 512-byte stride
  continuing: 60,928 -> 61,440 -> 61,952 -> 62,464 -> 62,976) whose
  changed extent is 96 bytes (the ffwd norm). It binds `block30.layer0.layer`
  (524,288 B at heap offset 20,534,784 — past the intervening
  `block3.layer0.layer` record, 20,672 B at 20,513,792 with 320 bytes of
  tail padding to the boundary, so block29.layer3.layer's naive +263,168
  successor is that small record, not block30's first). Its parameter
  block is the slot-28/32/36/40/44/48 word map exactly (pool, pool,
  weights, constant, pool, zero, pool) with w5 reserved zero — an ffwd
  half carries no pair second constant.
- slot 53 `cc_split_swin_16h_ffwd_proj_512_chained_fp8` (s52 -> s53): the
  block30 ffwd-proj pair's second half (the slot-25/29/33/37/41/45/49 word
  map, same 72-byte block and grid 8,3,1). It reads slot 52's output (w0 at
  34,618,368 — address-equal to slot 52's w1) and a residual input (w1 at
  34,274,304 — the buffer slot 51 wrote its output to, the input slot 52
  itself read), consumes slot 52's sync record (w7 at 62,976 —
  address-equal to slot 52's w6, the chained handoff), writes its output
  tensor at w2 (344,064 bytes at 34,962,432 — one stride past its input,
  the regular output spacing) and a sync record at w6 (63,488 — the
  512-byte stride continuing: 61,952 -> 62,464 -> 62,976 -> 63,488) whose
  changed extent is 96 bytes (the proj norm). It binds `block30.layer1.layer`
  (263,168 B at heap offset 21,059,072 — 20,534,784 + 524,288 at the 512
  boundary, block30's layer0 closing exactly at the boundary so no padding
  intervenes). Its parameter block is the slot-25/29/33/37/41/45/49 word map
  exactly (pool, pool, pool, weights, constant, zero, pool, pool, zero)
  with w5/w8 reserved zero — every ffwd->proj half reserves the words (the
  ffwd->proj pair carries no second constant).
- slot 54 `cc_split_swin_16h_qkv_512_chained_fp8` (s53 -> s54): the
  block30 qkv-proj pair's first half (the slot-26/30/34/38/42/46/50 word
  map, same 56-byte block; grid 4,4,4 — the slot-30/38/46 geometry). It
  reads slot 53's output (w0 at 34,962,432 — address-equal to slot 53's
  w2), consumes slot 53's sync record (w5 at 63,488 — address-equal to
  slot 53's w6, the chained handoff), writes its output tensor at w1
  (344,064 bytes at 35,306,496 — one stride past its input, the regular
  output spacing) and a sync record at w6 (64,000 — the 512-byte stride
  continuing: 62,464 -> 62,976 -> 63,488 -> 64,000) whose changed extent
  is 256 bytes (the qkv 192/256 alternation: 26/34/42/50 at 192 B,
  30/38/46/54 at 256 B). It binds `block30.layer2.layer` (917,568 B at
  heap offset 21,322,240 — 21,059,072 + 263,168 at the 512 boundary, the
  record immediately after `block30.layer1.layer`). Its parameter block is
  the slot-26/30/34/38/42/46/50 word map exactly (pool, pool, weights,
  constant, constant, pool, pool) with w4 the pair's second constant at
  0xFFFFFFFC00000000 — the high-dword form, the absent/double/low/high
  cycle's block30 step (block23 absent / 24 double / 25 low / 26 high /
  27 absent / 28 double / 29 low / 30 high).
- slot 55 `cc_split_swin_16h_proj_pool_512_chained_fp8` (s54 -> s55): the
  block30 `qkv`-proj pair's second half in the `proj_pool` variant — the
  first 80-byte block (10 words; every earlier block is 56 or 72 bytes) and
  the first kernel with a second output tensor. It reads slot 54's output
  (w0 at 35,306,496 — address-equal to slot 54's w1) and a residual input
  (w1 at 34,962,432 — the buffer slot 53 wrote its output to, the input
  slot 54 itself read), consumes slot 54's sync record (w5 at 64,000 —
  address-equal to slot 54's w6, the chained handoff), writes its output
  tensor at w2 (344,064 bytes at 52,313,088), the pooled output tensor at
  w3 (98,304 bytes at 26,803,200 — the `proj_pool` name's pool expansion)
  and a sync record at w6 (64,512 — the 512-byte stride continuing:
  63,488 -> 64,000 -> 64,512) whose changed extent is 96 bytes. The
  write-set attribution pins exactly those three words (names-s54-s55.txt:
  w2 + w3 + w6, zero unattributed, stable across all three frames). It
  binds `block30.layer3.layer` (263,168 B at heap offset 22,240,256 — the
  record immediately after `block30.layer2.layer`). Its parameter block is
  a new ten-word shape (pool, pool, pool, pool, weights, pool, pool,
  constant, constant, constant): w7 the pair's second constant at
  0xFFFFFFFC00000000 — block30's high-dword step at the proj half's own
  word position, now measured in both halves of the pair — w8 the (24, 28)
  geometry constant and w9 a (12, 16) constant (the pooled tensor's
  geometry: 12x16x512 = 98,304 bytes).
- slot 56 `cc_split_swin_16h_final_head_512_wait_fp8` (s55 -> s56): the
  transformer interval's final head and the first `wait`-family kernel —
  the first slot that consumes a sync record without writing one — in the
  smallest of the transformer interval's parameter blocks (40 bytes, five
  words; every earlier contract block is 56, 72 or 80 bytes). It reads
  slot 55's pooled output
  (w0 at 26,803,200 — address-equal to slot 55's w3, the `proj_pool`
  expansion) and waits on slot 55's sync record (w3 at 64,512 —
  address-equal to slot 55's w6, the chained handoff) without writing a
  sync record of its own: the write-set attribution pins exactly one word
  (names-s55-s56.txt: w1 only, zero unattributed, stable across all three
  frames) and the sync record's extent is byte-unchanged across the cut.
  It writes its output tensor at w1 (196,608 bytes at 52,657,152 — 384
  512-wide rows, twice the pooled input's 192; immediately past slot 55's
  w2 output, 52,313,088 + 344,064, and closing exactly at slot 57's output
  binding w1 at 52,853,760 — the `cc_vit_1d_repack_2d_to_1d_fp8` launch
  that opens the ViT interval reads that extent back as its own input).
  It binds `block30.layer4.layer` (524,304 B at heap offset 22,503,424 —
  the record immediately after `block30.layer3.layer`, block30's fifth
  record: 22,240,256 + 263,168 at the 512 boundary; the transformer
  interval's last weight record, `block31.layer0.layer` at 23,028,224
  opening the ViT records). Its parameter block is a new five-word shape
  (pool, pool, weights, pool, constant): w4 the (12, 16) pooled geometry
  constant 0x000000100000000C — slot 55's w9 value recurring at the head.
- slot 57 `cc_vit_1d_repack_2d_to_1d_fp8` (s56 -> s57): the ViT interval's
  opening repack — the first single-variant contract (no fp8 sibling
  exists: the repack family has one kernel per direction, this 2d-to-1d
  here and the `cc_vit_1d_repack_1d_to_2d_fp8` at slot 98, rather than the
  tilesync/chained/wait variant pairs of every earlier family) and the
  first weights-less one (psize 24 — three words, the smallest of the
  replayed contracts' parameter blocks; every earlier block carries a
  weights word). It reads slot 56's final-head tensor back (w0 at
  52,657,152 — address-equal to slot 56's w1) and writes the repacked
  tensor at w1 (196,608 bytes at 52,853,760 — the head's own 384x512
  geometry, closing exactly at the ViT interval's 53,050,368 boundary
  where slot 97's w2 and slot 98's w0 bind). w2 is the (12, 16) constant
  0x000000100000000C (slot 55's w9 and slot 56's w4 recurring). The
  write-set attribution pins exactly one word (names-s56-s57.txt: w1
  only, zero unattributed, stable across all three frames) and there is
  no sync word at all — the repack family is outside the sync chain. The
  repack is a pure byte permutation of the head tensor, not an arithmetic
  kernel: both write-set spans have all-zero pre-states (measured over the
  retained capture pools, vr/logs/ladder/permutation-s56-s57.json), so each
  write-set's changed count is its output tensor's non-zero byte count and
  the two match frame for frame (171,969/171,970/171,962); the sorted byte
  multisets of the head tensor and the repacked tensor are equal per frame
  (same artefact). Its control set is
  correspondingly re-based (see Controls below), and its kernel image is
  extracted from DLL fatbin container 5 — each family is created from its
  own cubin (the traces pin one per family) and container 4 carries the
  transformer interval's kernels only.
- slot 58 `cc_vit_1d_ffn_expand_publish_fp8` (s57 -> s58): the ViT
  interval's first weighted kernel — the opening `ffn_expand` of the ViT
  block pattern (`ffn_expand` -> `ffn_contract` -> `qkv` -> `attention` ->
  `projection`, a five-slot cycle) and the first contract to bind a weight
  record of the ViT block records. Measured from the s58 capture: word
  classes (POOL, ZERO, POOL, WEIGHTS, ZERO, ZERO, ZERO, POOL, CONSTANT),
  w0 input (52,853,760 — address-equal to slot 57's w1, the repacked head
  tensor read back as the ViT block's input), w2 output (34,372,608 —
  786,432 bytes, the 384x2048 fp8 expansion of the 384x512 input: the
  extent closes exactly at slot 63's w2 binding at 35,159,040, the next
  `ffn_expand` output's base, and the next block's extent repeats the same
  786,432), w3 weights on the layout walk's next record start
  (`block31.layer0.layer`, 4,194,320 B at heap offset 23,028,224 —
  22,503,424 + 524,304 rounded up at the 512 boundary, block31's first
  record and the first ViT record), w7 sync-out (65,024 — the 512-byte
  stride continuing: 64,512 -> 65,024) and w8 the (12, 16) constant
  0x000000100000000C (slot 55's w9 / slot 56's w4 / slot 57's w2
  recurring); w1, w4, w5 and w6 are reserved zero — no sync-in word is
  bound at all (unlike every `chained` transformer half). The sync extent
  measured 256 bytes (span 65,024 -> 65,280) and the write-set is two
  segments per frame (names-s57-s58.txt: w2 with 779,794/779,827/779,876
  changed bytes inside the 786,432-byte binding, w7 256 B changed,
  zero unattributed). The `publish` variant is the live one at the
  interval's opening slot (the later `ffn_expand` launches use the
  `chained` variant at their own slots); the fp8 variant is created but
  never launched — pinned sibling.
- slot 59 `cc_vit_1d_ffn_contract_chained_fp8` (s58 -> s59): the ViT
  block's second slot — the `ffn_contract` half of the five-slot cycle
  (`ffn_expand` -> `ffn_contract` -> `qkv` -> `attention` ->
  `projection`) and the first contract with three written output regions.
  Measured from the s59 capture: word classes (POOL, POOL, POOL, WEIGHTS,
  POOL, POOL, POOL, POOL, CONSTANT) — every word bound, no reserved zero
  word (slot 58's nine-word block reserves four of its nine; slots 55, 56
  and 57 also bind every word at their own shapes).
  w0 input (34,372,608 — address-equal to slot 58's w2, the fp8
  expansion read back as the contraction's input), w1 residual
  (52,853,760 — slot 57's repacked head tensor bound again, the skip
  connection), w2 output (53,246,976 — 196,608 bytes, the 384x512 fp8
  contraction closing exactly at slot 60's w1 binding at 53,443,584), w3
  weights (`block31.layer1.layer`, 4,196,352 B at heap offset 27,223,040 —
  the layout walk's next record start after `block31.layer0.layer`), w5
  second output (49,167,360 — 393,216 bytes, closing exactly at slot 62's
  w5 binding at 49,560,576; the `projection` half of the cycle shares this
  whole word map), w4 the third output (65,536 — 64 bytes, the record
  walk's 512-byte stride continuing: 65,024 -> 65,536), w6 sync in
  (65,024 — slot 58's record), w7 the sync record (66,048 — 64 bytes, the
  next stride) and w8 the (12, 16) constant 0x000000100000000C recurring.
  The write-set is four segments per frame (names-s58-s59.txt: w5 with
  342,690/342,703/342,652 changed bytes inside the 393,216-byte binding,
  w2 with 171,959/171,966/171,970 inside the 196,608-byte binding — both
  changed boxes closing 4 bytes short of their bindings, the final word
  static in every frame — w4 and w7 64 B changed each, zero unattributed).
  The `chained` variant is live (the fp8 variant is created but never
  launched — pinned sibling). The sibling run re-classifies the two
  record-walk publications: w7 is the chain handoff the plain variant
  leaves at the input state (the 64 B sibling divergence, pinned), while
  w4 is variant-independent output data — both variants write it
  byte-identically, the first contract where the sibling shares an output
  beyond the main tensors (a record-walk-only prediction of 128 B of
  divergence is wrong here: the retained sibling run pins 64 B). Slot 60's
  parameter block binds w7's
  record as its sync-in; no launch in the 156-launch sequence binds w4's
  record (scanned from the same retained traces the replay reads).
- slot 60 `cc_vit_1d_qkv_chained_fp8` (s59 -> s60): the ViT block's
  third slot — the `qkv` half of the five-slot cycle (`ffn_expand` ->
  `ffn_contract` -> `qkv` -> `attention` -> `projection`) and the first
  contract with five written output regions (the q/k/v triplet plus w6
  and the w5 publication) and a six-segment write-set. Measured from the
  s60 capture: word classes (POOL, POOL, POOL, POOL, WEIGHTS, POOL,
  POOL, POOL, POOL, CONSTANT). w0 input (53,246,976 — address-equal to
  slot 59's w2, the contraction output read back as the qkv input),
  w1/w2/w3 the q/k/v triplet (196,608 bytes each, 384x512 — slot 61's
  attention binds them as its w0/w1/w2, and the triplet tiles
  [53,443,584, 54,033,408), closing exactly at the attention output's
  base), w4 weights (`block31.layer2.layer`, 3,145,856 B at heap offset
  31,419,392 — the layout walk's next record start after
  `block31.layer1.layer`, closing exactly at its end), w6 a fourth
  written region (33,192,960 — 1,179,648 bytes = 384x3,072, closing
  exactly at slot 58's w2 binding at 34,372,608; one per ViT block —
  block32's qkv binds its w6 at pool 20,610,048), w5 the first
  record-walk publication (66,560 — 128 bytes, the 512-byte stride
  continuing: 66,048 -> 66,560), w7 sync in (66,048 — slot 59's record),
  w8 the sync record (67,072 — 128 bytes, the next stride; slot 61 binds
  its record at the attention block's w5 sync-in), w9 the (12, 16)
  constant 0x000000100000000C recurring. The write-set is six segments
  per frame (names-s59-s60.txt: w6 with 1,172,971/1,172,523/1,172,573
  changed bytes inside the 1,179,648-byte binding — the changed box
  covering the whole binding, w3 with 171,974/171,993/171,981 inside the
  196,608-byte binding closing 2 bytes short, w1 with
  171,959/171,944/171,950 closing 4 bytes short, w2 with
  171,661/171,667/171,669 closing 8 bytes short — the final words static
  in every frame — and w5/w8 128 B changed each, zero unattributed).
  The `chained` variant is live (the fp8 variant is created but never
  launched — pinned sibling). The two record-walk publications follow
  slot 59's re-classification: w8 is the chain handoff (its record is
  bound by slot 61 as the attention block's sync-in — the predicted 128 B
  sibling divergence), while w5 is bound by no other launch in the
  156-launch sequence (scanned from the same retained traces the replay
  reads) and is variant-independent output data.
- slot 61 `cc_vit_1d_attention_chained_fp8` (s60 -> s61): the ViT block's
  fourth slot — the `attention` half of the five-slot cycle — and the
  first eight-word block (psize 64). Measured from the s61 capture: word
  classes (POOL, POOL, POOL, POOL, WEIGHTS, POOL, POOL, CONSTANT).
  w0/w1/w2 the q/k/v triplet (53,443,584 / 53,836,800 / 53,640,192 —
  address-equal to slot 60's w1/w2/w3), w3 output (54,033,408 — 196,608
  bytes, closing exactly at slot 62's w2 binding at 54,230,016), w4
  weights (`block31.layer3.layer`, a 2-byte payload at heap offset
  34,565,632 — the layout walk's next record start after
  `block31.layer2.layer`, whose end at 34,565,248 rounds up at the 512
  boundary — a placeholder record and a dead binding: the kernel's output
  is invariant under whole-record complementation as well as the wrong
  control's rotation, measured in both variants, so the wrong control
  re-bases onto the query tensor at w0 — `wrong_word`, the probe's first
  control re-targeting), w5 sync in (67,072 — slot 60's record), w6 the
  sync record (67,584 — 256 bytes, the 512-byte stride continuing:
  67,072 -> 67,584; slot 62 binds its record at the projection block's w6
  sync-in), w7 the (12, 16) constant 0x000000100000000C recurring. The
  write-set is two segments per frame (names-s60-s61.txt: w3 with
  196,535/196,524/196,496 changed bytes inside the 196,608-byte binding —
  the changed box covering the whole binding, 73/84/112 interior bytes
  static in the three frames — and w6 256/256/256, the whole publication,
  zero unattributed). The `chained` variant is live (the fp8 variant is
  created but never launched — pinned sibling). Slot 61 is the first
  contract with a single record-walk publication: the family rule
  classifies it by consumption alone (slot 62 binds w6's record as its
  sync-in, so w6 is the chain handoff — the predicted 256 B sibling
  divergence), with no sibling-shared publication at this block.
- slot 62 `cc_vit_1d_projection_chained_fp8` (s61 -> s62): the ViT block's
  fifth slot — the `projection` half closing block31's five-slot cycle
  (`ffn_expand` -> `ffn_contract` -> `qkv` -> `attention` -> `projection`).
  It shares slot 59's nine-word contraction shape (psize 72, every word
  bound, three written output regions). Measured from the s62 capture: word
  classes (POOL, POOL, POOL, WEIGHTS, POOL, POOL, POOL, POOL, CONSTANT).
  w0 input (54,033,408 — address-equal to slot 61's w3, the attention
  output read back as the projection's input), w1 residual (53,246,976 —
  address-equal to slot 59's w2, the contraction output from three slots
  earlier bound again as the attention sub-block's skip connection), w2
  output (54,230,016 — 196,608 bytes, 384x512, closing at 54,426,624;
  slot 63's `ffn_expand` binds it as w0 and slot 64's `ffn_contract` as its
  w1 residual, opening block32's cycle), w3 weights (`block31.layer4.layer`,
  1,050,624 B at heap offset 34,566,144 — the layout walk's next record
  start after `block31.layer3.layer`, whose 2-byte payload at 34,565,632
  rounds up across 510 bytes of padding to the 512 boundary, closing
  block31's five-record set exactly at `block32.layer0.layer`'s start at
  35,616,768), w5 second output (49,560,576 — 393,216 bytes, immediately
  past slot 59's w5 [49,167,360, 49,560,576) and closing at slot 64's w5
  binding at 49,953,792), w4 third output (68,096 — 64 bytes, the record
  walk's 512-byte stride continuing: 67,584 -> 68,096), w6 sync in (67,584
  — slot 61's record), w7 the sync record (68,608 — 64 bytes, the next
  stride; slot 63 binds its record at `ffn_expand`'s w6 sync-in) and w8 the
  (12, 16) constant 0x000000100000000C recurring. The write-set is four
  segments per frame (names-s61-s62.txt: w5 with 391,486/391,452/391,349
  changed bytes inside the 393,216-byte binding and w2 with
  196,500/196,524/196,534 inside the 196,608-byte binding — both changed
  boxes covering their whole bindings, unlike slot 59's 4-byte-short boxes
  — and w4/w7 64 B changed each, zero unattributed). Three variants exist
  in the cubin (`chained`, `wait`, `fp8`): `chained` is live at slots
  62/67/72/77/82/87/92 (`wait` runs at slot 97, the ViT interval's closing
  projection), while `fp8` is created but never launched (pinned sibling).
  The two record-walk publications follow the slot-59/60 family rule: w7 is
  the chain handoff (slot 63 binds its record as sync-in — the predicted
  64 B sibling divergence), while w4 is bound by no other launch in the
  156-launch sequence (scanned from the same retained traces the replay
  reads) and is variant-independent output data.
- slot 63 `cc_vit_1d_ffn_expand_chained_fp8` (s62 -> s63): the second ViT
  block's opening slot — block32's five-slot cycle starting at the
  `ffn_expand` family's `chained` variant's first launch (slot 58's
  `publish` opened block31's cycle, writing the chain's first record
  without consuming one). It shares slot 58's nine-word expansion shape
  (psize 72) with w6 bound as the sync-in that `publish` leaves zero.
  Measured from the s63 capture: word classes (POOL, ZERO, POOL, WEIGHTS,
  ZERO, ZERO, POOL, POOL, CONSTANT). w0 input (54,230,016 — address-equal
  to slot 62's w2, the projection output read back as the next block's
  input), w2 output (35,159,040 — 786,432 bytes, the 384x2048 fp8
  expansion of the 384x512 input, closing at 35,945,472 where slot 68's
  `chained` expansion output chains one block-stride later and slot 64's
  `ffn_contract` binds the region as w0), w3 weights
  (`block32.layer0.layer`, 4,194,320 B at heap offset 35,616,768 —
  block31's five-record set ends there exactly, no padding intervenes,
  block32's first record; block32 repeats block31's five-record shape), w6
  sync in (68,608 — slot 62's record), w7 the sync record (69,120 — 256
  bytes, the record walk's 512-byte stride continuing: 68,608 -> 69,120;
  slot 64 binds its record at `ffn_contract`'s w6 sync-in) and w8 the
  (12, 16) constant 0x000000100000000C recurring; w1, w4 and w5 are
  reserved zero. The write-set is two segments per frame
  (names-s62-s63.txt: w2 with 778,353/777,687/777,881 changed bytes
  inside the 786,432-byte binding, the span covering the whole binding,
  and w7 256/256/256, the whole publication, zero unattributed). Three
  variants exist in the cubin (`chained`, `publish`, `fp8`): `chained` is
  live at slots 63/68/73/78/83/88/93 (`publish` ran once at slot 58),
  while `fp8` is created but never launched (pinned sibling). The single
  record-walk publication is classified by consumption alone: w7 is the
  chain handoff (slot 64 binds its record as sync-in — the predicted 256 B
  sibling divergence).
- slot 64 `cc_vit_1d_ffn_contract_chained_fp8` (s63 -> s64): the second
  ViT block's second slot — block32's `ffn_contract` half of the five-slot
  cycle (`ffn_expand` -> `ffn_contract` -> `qkv` -> `attention` ->
  `projection`) — sharing slot 59's nine-word contraction shape exactly
  (psize 72, the same (12, 16) constant at w8). Measured from the s64
  capture: word classes (POOL, POOL, POOL, WEIGHTS, POOL, POOL, POOL,
  POOL, CONSTANT) — every word bound, no reserved zero word. w0 input
  (35,159,040 — address-equal to slot 63's w2, the fp8 expansion read
  back as the contraction's input), w1 residual (54,230,016 — slot 62's
  w2, the projection output from two slots earlier bound again as the
  skip connection; slot 59 bound slot 57's head tensor the same way), w2
  output (54,426,624 — 196,608 bytes, the 384x512 fp8 contraction closing
  exactly at slot 65's w1 binding at 54,623,232; slot 67's projection
  re-reads the region as its w1 residual), w3 weights
  (`block32.layer1.layer`, 4,196,352 B at heap offset 39,811,584 — the
  layout walk's next record start after `block32.layer0.layer`, its
  4,194,320-byte payload at 35,616,768 plus 496 bytes of padding to the
  512 boundary), w5 second output (49,953,792 — 393,216 bytes, closing
  exactly at 50,347,008 where slot 69's w5 chains one block-stride
  later), w4 the third output (69,632 — 64 bytes, the record walk's
  stride continuing: 69,120 -> 69,632), w6 sync in (69,120 — slot 63's
  record) and w7 the sync record (70,144 — 64 bytes, the next stride;
  slot 65 binds the record at the `qkv` block's w7 sync-in) plus w8 the
  (12, 16) constant 0x000000100000000C recurring. The write-set is four
  segments per frame (names-s63-s64.txt: w5 with
  391,734/391,648/391,716 changed bytes inside the 393,216-byte binding
  and w2 with 196,523/196,553/196,503 inside the 196,608-byte binding —
  both changed boxes covering their whole bindings exactly with static
  interior bytes (1,482/1,568/1,500 and 85/55/105), w4 and w7 64 B
  changed each, zero unattributed). The `chained` variant is live (the
  fp8 variant is created but never launched — pinned sibling). The
  record-walk re-classification holds a fourth time: w7 is the chain
  handoff (the plain variant leaves it at the input state — the predicted
  64 B sibling divergence), while w4 is variant-independent output data
  (both variants write it byte-identically — the fourth sibling-shared
  publication beyond the main tensors; no launch in the 156-launch
  sequence binds w4's record, scanned from the same retained traces the
  replay reads).
- slot 65 `cc_vit_1d_qkv_chained_fp8` (s64 -> s65): the second ViT
  block's third slot — block32's `qkv` half of the five-slot cycle —
  sharing slot 60's ten-word qkv shape exactly (psize 80, the same (12, 16)
  constant at w9). Measured from the s65 capture: word classes (POOL,
  POOL, POOL, POOL, WEIGHTS, POOL, POOL, POOL, POOL, CONSTANT). w0 input
  (54,426,624 — address-equal to slot 64's w2, the contraction output read
  back as the qkv input), w1/w2/w3 the q/k/v triplet (196,608 bytes each,
  384x512 — slot 66's attention binds them as its w0/w1/w2, and the
  triplet tiles [54,623,232, 55,213,056), closing exactly at the
  attention output's base at 55,213,056), w4 weights
  (`block32.layer2.layer`, 3,145,856 B at heap offset 44,007,936 — the
  layout walk's next record start after `block32.layer1.layer`, closing
  exactly at its end), w6 a fourth written region (20,610,048 —
  1,179,648 bytes = 384x3,072, the block32 instance of the one-per-block
  region slot 60's comment predicts, closing exactly at 21,789,696 where
  block33's `qkv` w6 chains immediately), w5 the first record-walk
  publication (70,656 — 128 bytes, the 512-byte stride continuing: 69,632
  -> 70,144 -> 70,656), w7 sync in (70,144 — slot 64's record), w8 the
  sync record (71,168 — 128 bytes, the next stride; slot 66 binds its
  record at the attention block's w5 sync-in), w9 the (12, 16) constant
  0x000000100000000C recurring. The write-set is six segments per frame
  (names-s64-s65.txt: w6 with 1,173,080/1,172,617/1,172,341 changed bytes
  inside the 1,179,648-byte binding — the changed box covering the whole
  binding, w3 with 196,552/196,517/196,546, w1 with
  196,549/196,526/196,538 and w2 with 196,218/196,171/196,181 inside the
  196,608-byte bindings — all three covering their whole bindings with
  static interior bytes (56/91/62, 59/82/70 and 390/437/427) — and w5/w8
  128 B changed each, zero unattributed). The `chained` variant is live
  (the fp8 variant is created but never launched — pinned sibling). The
  two record-walk publications follow the established re-classification:
  w8 is the chain handoff (its record is bound by slot 66 as the attention
  block's w5 sync-in — the predicted 128 B sibling divergence), while w5
  is bound by no other launch in the 156-launch sequence (scanned from the
  same retained traces the replay reads) and is variant-independent output
  data.
- slot 66 `cc_vit_1d_attention_chained_fp8` (s65 -> s66): the second ViT
  block's fourth slot — block32's `attention` half of the five-slot cycle
  — sharing slot 61's eight-word attention shape exactly (psize 64, the
  (12, 16) constant at w7 — slot 55 binds ten words, slot 56 five, slot 57
  three, slots 58 and 59 nine, slot 60 ten, slot 61 eight). Measured from
  the s66 capture: word classes (POOL, POOL, POOL, POOL, WEIGHTS, POOL,
  POOL, CONSTANT). w0/w1/w2 the q/k/v triplet (54,623,232 / 54,819,840 /
  55,016,448 — address-equal to slot 65's w1/w2/w3, the three
  196,608-byte tensors consumed as query, key and value), w3 output
  (55,213,056 — 196,608 bytes, the attention output closing exactly at
  slot 67's w2 binding at 55,409,664), w4 weights
  (`block32.layer3.layer`, a 2-byte payload at heap offset 47,154,176 —
  the layout walk's next record start after `block32.layer2.layer`, whose
  end at 47,153,792 rounds up across 384 bytes at the 512 boundary; the
  ViT attention blocks' layer3 records all measure a 2-byte payload —
  block32's `f3 b3`) — a dead binding like slot 61's: with the wrong
  control aimed at the record (the query re-targeting removed for the
  experiment), both an inward rotation and whole-record complementation
  reproduce every execute pool byte-for-byte in the retained
  `wrong-record-rotation-experiment` and `wrong-complement-experiment`
  runs, so the weight-free attention family reads nothing from its
  placeholder record and the wrong control re-bases onto the input tensor
  the kernel functionally reads (the query at w0) — the family's second
  re-targeting, pinned by `wrong_word`), w5 sync in (71,168 — slot 65's
  record, the chained handoff continuing), w6 the sync record (71,680 —
  256 bytes, the 512-byte stride continuing: 71,168 -> 71,680; slot 67
  binds this record at the projection block's w6 sync-in), w7 the (12, 16)
  constant 0x000000100000000C recurring. The write-set is two segments per
  frame (names-s65-s66.txt: w3 with 196,547/196,544/196,547 changed bytes
  inside the 196,608-byte binding — the changed box covering the whole
  binding, 61/64/61 interior bytes static in the three frames — and w6
  256/256/256, the whole publication, zero unattributed). The `chained`
  variant is live (the fp8 variant is created but never launched — pinned
  sibling). As at slot 61 the block has a single record-walk publication,
  classified by consumption alone: slot 67 binds w6's record as its
  sync-in, so w6 is the chain handoff and the sibling divergence is the
  whole 256-byte publication (the plain variant leaves it at the input
  state) — there is no sibling-shared publication here.
- slot 67 `cc_vit_1d_projection_chained_fp8` (s66 -> s67): the second ViT
  block's fifth slot — block32's `projection` closing the five-slot cycle
  — sharing slot 62's nine-word projection shape exactly (psize 72,
  grid (16,1,4), every word bound, three written output regions). Measured
  from the s67 capture: word classes (POOL, POOL, POOL, WEIGHTS, POOL,
  POOL, POOL, POOL, CONSTANT). w0 input (55,213,056 — address-equal to
  slot 66's w3, the attention output read back as the projection's input),
  w1 residual (54,426,624 — address-equal to slot 64's w2, the contraction
  output from three slots earlier bound again as the attention sub-block's
  skip connection), w2 output (55,409,664 — 196,608 bytes, closing at
  55,606,272; slot 68's `ffn_expand` binds it as w0), w3 weights
  (`block32.layer4.layer`, 1,050,624 B at heap offset 47,154,688 — the
  layout walk's next record start after `block32.layer3.layer`, whose
  2-byte payload at 47,154,176 rounds up across 510 bytes of padding to
  the 512 boundary, closing block32's five-record set exactly at
  `block33.layer0.layer`'s start at 48,205,312 — the block31
  layer3-to-layer4 pair repeating exactly), w5 second output (48,774,144 —
  393,216 bytes, closing at 49,167,360 where slot 59's w5 binds the
  region's next extent), w4 third output (72,192 — 64 bytes, the record
  walk's 512-byte stride continuing: 71,680 -> 72,192), w6 sync in
  (71,680 — slot 66's record), w7 the sync record (72,704 — 64 bytes, the
  next stride; slot 68 binds its record at `ffn_expand`'s w6 sync-in) and
  w8 the (12, 16) constant 0x000000100000000C recurring. The write-set is
  four segments per frame (names-s66-s67.txt: w5 with
  391,543/391,442/391,443 changed bytes inside the 393,216-byte binding
  and w2 with 196,545/196,547/196,536 inside the 196,608-byte binding —
  both changed boxes covering their whole bindings — and w4/w7 64 B
  changed each, zero unattributed). Three variants exist in the cubin (`chained`, `wait`,
  `fp8`): `chained` is live at slots 62/67/72/77/82/87/92 (`wait` runs at
  slot 97), while `fp8` is created but never launched (pinned sibling).
  The two record-walk publications follow the slot-59/60 family rule: w7
  is the chain handoff (slot 68 binds its record as sync-in — the
  predicted 64 B sibling divergence), while w4 is bound by no other launch
  in the 156-launch sequence (scanned from the same retained traces the
  replay reads) and is variant-independent output data — the
  re-classification's sixth confirmation.

- slot 68 `cc_vit_1d_ffn_expand_chained_fp8` (s67 -> s68): block33's
  expansion — the third ViT block's opening slot, the expansion family's
  `chained` variant's second launch (slots 63/68/73/78/83/88/93;
  `publish` opened block31's cycle at slot 58). Shares slot 63's
  nine-word expansion shape (psize 72, grid (64,1,1)) exactly. Measured
  from the s68 capture: word classes (POOL, ZERO, POOL, WEIGHTS, ZERO,
  ZERO, POOL, POOL, CONSTANT). w0 input (55,409,664 — address-equal to
  slot 67's w2, the projection output read back as the next block's
  input), w2 output (35,945,472 — 786,432 bytes, closing at 36,731,904;
  slot 63's output closes at 35,945,472 exactly, the block-stride chain,
  and slot 69's `ffn_contract` binds the region as w0), w3 weights
  (`block33.layer0.layer`, 4,194,320 B at heap offset 48,205,312 —
  block32's five-record set ends there exactly and block33's first record
  begins; 496 bytes of padding close the gap to slot 69's record at
  52,400,128), w6 sync in (72,704 — slot 67's record), w7 the sync record
  (73,216 — 256 bytes, the record walk's 512-byte stride continuing:
  72,704 -> 73,216; slot 69 binds its record at `ffn_contract`'s w6
  sync-in) and w8 the (12, 16) constant 0x000000100000000C recurring; w1,
  w4 and w5 are reserved zero. The write-set is two segments per frame
  (names-s67-s68.txt: w2 with 779,454/778,805/778,812 changed bytes inside
  the 786,432-byte binding, the span covering the whole binding, and w7
  256/256/256, the whole publication, zero unattributed). Three variants
  exist in the cubin (`chained`, `publish`, `fp8`): `chained` is live at
  slots 63/68/73/78/83/88/93 (`publish` ran once at slot 58), while `fp8`
  is created but never launched (pinned sibling). The single record-walk
  publication is classified by consumption alone: w7 is the chain handoff
  (slot 69 binds its record as sync-in — the predicted 256 B sibling
  divergence), the single-publication rule's fourth confirmation.

- slot 69 `cc_vit_1d_ffn_contract_chained_fp8` (s68 -> s69): block33's
  contraction — the third ViT block's second slot — sharing slot 59's
  nine-word contraction shape exactly (psize 72, grid (16,1,4), the same
  (12, 16) constant at w8). Measured from the s69 capture (manifest and
  sidecar agree byte-for-byte): word classes (POOL, POOL, POOL, WEIGHTS,
  POOL, POOL, POOL, POOL, CONSTANT) — every word bound, no reserved zero
  word. w0 input (35,945,472 — address-equal to slot 68's w2, the fp8
  expansion read back as the contraction's input), w1 residual
  (55,409,664 — slot 67's w2, the projection output from two slots
  earlier bound again as the skip connection; slot 64 bound slot 62's w2
  the same way), w2 output (55,606,272 — 196,608 bytes, the 384x512 fp8
  contraction closing exactly at 55,802,880; slot 72's projection
  re-reads the region as its w1 residual), w3 weights
  (`block33.layer1.layer`, 4,196,352 B at heap offset 52,400,128 — the
  layout walk's next record start after `block33.layer0.layer`, its
  4,194,320-byte payload at 48,205,312 plus 496 bytes of padding to the
  512 boundary), w5 second output (50,347,008 — 393,216 bytes, closing
  exactly at 50,740,224 where slot 72's w5 chains one block-stride
  later), w4 the third output (73,728 — 64 bytes, the record walk's
  stride continuing: 73,216 -> 73,728), w6 sync in (73,216 — slot 68's
  record) and w7 the sync record (74,240 — 64 bytes, the next stride;
  slot 70 binds the record at the `qkv` block's w7 sync-in) plus w8 the
  (12, 16) constant 0x000000100000000C recurring. The write-set is four
  segments per frame (names-s68-s69.txt: w5 with
  391,762/391,784/391,712 changed bytes inside the 393,216-byte binding
  and w2 with 196,555/196,553/196,521 inside the 196,608-byte binding —
  both changed boxes covering their whole bindings exactly with static
  interior bytes (1,454/1,432/1,504 and 53/55/87), w4 and w7 64 B changed
  each, zero unattributed). The `chained` variant is live (the fp8
  variant is created but never launched — pinned sibling). The
  record-walk re-classification holds a seventh time: w7 is the chain
  handoff (the plain variant leaves it at the input state — the predicted
  64 B sibling divergence), while w4 is variant-independent output data
  (both variants write it byte-identically — the seventh sibling-shared
  publication beyond the main tensors; no launch in the 156-launch
  sequence binds w4's record, scanned from the same retained traces the
  replay reads).

- slot 70 `cc_vit_1d_qkv_chained_fp8` (s69 -> s70): block33's
  q/k/v, the third ViT block's third slot — block33's `qkv` half of the
  five-slot cycle (`ffn_expand` -> `ffn_contract` -> `qkv` ->
  `attention` -> `projection`), sharing slot 60's ten-word qkv shape
  exactly (psize 80, grid (32,1,2), the same (12, 16) constant at w9).
  w0 input (55,606,272 — address-equal to slot 69's w2, the contraction
  output read back as the qkv input), w1/w2/w3 the q/k/v triplet
  (55,802,880 / 55,999,488 / 56,196,096 — 196,608 bytes each, the
  384x512 fp8 tensors tiling [55,802,880, 56,392,704) and closing
  exactly at the attention output's base at 56,392,704 where slot 71's
  w3 lands; slot 71's attention binds the three as its w0/w1/w2), w4
  weights (`block33.layer2.layer`, 3,145,856 B at heap offset
  56,596,480 — the layout walk's next record start after
  `block33.layer1.layer`, its 4,196,352-byte payload at 52,400,128
  closing exactly there with no padding — the stage-61 prediction
  pinned), w6 the fourth written region (21,789,696 — 1,179,648 bytes =
  384x3,072, the block33 instance of the one-per-block region and the
  exact address slot 65's comment predicted the block33 `qkv` w6 would
  chain immediately at), w5 the first record-walk publication (74,752 —
  128 bytes, the stride continuing: 74,240 -> 74,752), w7 sync in
  (74,240 — slot 69's record) and w8 the sync record (75,264 — 128
  bytes, the next stride; slot 71 binds the record at the `attention`
  block's w5 sync-in) plus w9 the (12, 16) constant. The write-set is
  six segments per frame (names-s69-s70.txt: w6 with
  1,173,143/1,172,569/1,172,385 changed bytes inside the 1,179,648-byte
  binding — the changed box covering the whole binding — and w1/w2/w3
  with 196,558/196,550/196,546, 196,254/196,178/196,178 and
  196,549/196,547/196,549 inside the 196,608-byte bindings — all three
  covering their whole bindings with static interior bytes (50/58/62,
  354/430/430 and 59/61/59), w5 and w8 128 B changed each, zero
  unattributed). The `chained` variant is live (the fp8 variant is
  created but never launched — pinned sibling). The record-walk
  re-classification holds an eighth time: w8 is the chain handoff (the
  plain variant leaves it at the input state — the predicted 128 B
  sibling divergence), while w5 is variant-independent output data (both
  variants write it byte-identically — the eighth sibling-shared
  publication beyond the main tensors; no launch in the 156-launch
  sequence binds w5's record, scanned from the same retained traces the
  replay reads).

- slot 71 `cc_vit_1d_attention_chained_fp8` (s70 -> s71): block33's
  attention, the third ViT block's fourth slot and the attention family's
  third launch — sharing slot 61's eight-word attention shape exactly
  (psize 64, grid (32,1,1), the (12, 16) constant at w7). w0/w1/w2 the
  q/k/v triplet (55,802,880 / 55,999,488 / 56,196,096 — address-equal to
  slot 70's w1/w2/w3, consumed as query, key and value), w3 output
  (56,392,704 — 196,608 bytes, closing at 56,589,312 where slot 72's
  projection w2 lands; the base slot 70's contract predicted), w4
  weights (`block33.layer3.layer`, a 2-byte payload at heap offset
  59,742,720 — the layout walk's next record start after
  `block33.layer2.layer`, whose end at 59,742,336 rounds up across 384
  bytes at the 512 boundary; the ViT attention blocks' layer3 records all
  measure a 2-byte payload) — a dead binding like slots 61 and 66's: the
  weight-free attention family reads nothing from its placeholder record,
  so the wrong control re-bases onto the query tensor at w0, the family's
  third re-targeting pinned by `wrong_word` (the rotation and
  complementation experiments reproduce the execute pools
  byte-for-byte), w5 sync in (75,264 — slot 70's record) and w6 the sync
  record (75,776 — 256 bytes, the stride continuing: 75,264 -> 75,776;
  slot 72 binds the record at the `projection` block's w6 sync-in) plus
  w7 the (12, 16) constant. The write-set is two segments per frame
  (names-s70-s71.txt: w3 with 196,523/196,527/196,549 changed bytes
  inside the 196,608-byte binding — the changed box covering the whole
  binding, 85/81/59 interior bytes static — and w6 256/256/256, the
  whole publication, zero unattributed). The `chained` variant is live
  (the fp8 variant is created but never launched — pinned sibling). As at
  slots 61 and 66 the block has a single record-walk publication,
  classified by consumption alone: slot 72 binds w6's record as its
  sync-in, so w6 is the chain handoff and the sibling divergence is the
  whole 256-byte publication — there is no sibling-shared publication at
  this contract.
- slot 72 `cc_vit_1d_projection_chained_fp8` (s71 -> s72): block33's
  projection, the third ViT block's fifth slot and the block-cycle
  closing slot — sharing slot 62's nine-word projection shape exactly
  (psize 72, grid (16,1,4), the (12, 16) constant at w8). w0 input
  (56,392,704 — the attention output of slot 71 read back as the
  projection's input), w1 residual (55,606,272 — the contraction output
  of slot 69, three slots earlier, bound again as the attention
  sub-block's skip connection), w2 output (56,589,312 — 196,608 bytes,
  closing at 56,785,920 where slot 74's w2 binds the region's next
  extent), w3 weights
  (`block33.layer4.layer`, 1,050,624 B at heap offset 59,743,232 — the
  layout walk's next record start after `block33.layer3.layer`, whose
  2-byte payload at 59,742,720 rounds up across 510 bytes of padding to
  the 512 boundary, closing block33's five-record set exactly at
  `block34.layer0.layer`'s start at 60,793,856) — live like slots 62/67's
  projection records, the wrong control rotating it normally with no
  re-targeting, w5 second output (50,740,224 — 393,216 bytes, closing at
  51,133,440 where the region's next extent begins), w4 third
  output (76,288 — 64 bytes, the record walk's 512-byte stride
  continuing: 75,776 -> 76,288), w6 sync in (75,776 — slot 71's
  record), w7 the sync record (76,800 — 64 bytes, the next stride; slot
  73 binds its record at `ffn_expand`'s w6 sync-in). The write-set is
  four segments per frame (names-s71-s72.txt: w5 with
  391,671/391,507/391,719 changed bytes inside the 393,216-byte binding
  and w2 with 196,536/196,555/196,543 inside the 196,608-byte binding —
  both changed boxes covering their whole bindings — and w4/w7 64 B
  changed each, zero unattributed). Following the slot-59/60 family
  rule, the block has
  two record-walk publications: w7 is the chain handoff (slot 73 binds its
  record as sync-in — the predicted 64-byte sibling divergence), while w4
  is variant-independent output data shared by both kernel variants, bound
  by no other launch in the 156-launch sequence — the re-classification's
  ninth confirmation.
- slot 73 `cc_vit_1d_ffn_expand_chained_fp8` (s72 -> s73): block34's
  `ffn_expand`, opening its five-slot cycle and the expansion family's third
  `chained` launch (after slots 63 and 68; slot 58's `publish` opened
  block31's cycle). It shares slot 63's nine-word expansion shape (psize 72,
  grid (64,1,1), block (32,4,1), w8 the (12, 16) constant). The s73
  manifest and sidecar agree on every parameter word across three frames:
  w0 input (56,589,312 — slot 72's projection output), w2 output
  (36,731,904 — 786,432 bytes, the 384x2048 fp8 expansion, closing at
  37,518,336), w3 weights (`block34.layer0.layer`, 4,194,320 B at heap
  offset 60,793,856 — the record begins exactly where block33's five-record
  set closes), w6 sync-in (76,800 — slot 72's w7 record), and w7 the single
  256-byte publication (77,312 — the 512-byte stride continuing:
  76,800 -> 77,312; slot 74 binds it as w6 sync-in); w1, w4 and w5 are
  reserved zero. The write-set is two segments per frame
  (names-s72-s73.txt: w2 with 779,475/778,866/778,660 changed bytes inside
  the 786,432-byte binding, the changed box covering the whole binding,
  plus w7 256/256/256, the whole publication, zero unattributed). `chained`
  is live (the fp8 sibling is created but never launched — pinned sibling);
  the single-publication rule resolves w7 by consumption alone as the chain
  handoff (slot 74 binds that record at w6, so the predicted sibling
  divergence is exactly 256 B).
- slot 74 `cc_vit_1d_ffn_contract_chained_fp8` (s73 -> s74): block34's
  contraction, the second launch in its five-slot cycle, reusing slot 64's
  nine-word map (psize 72, grid (16,1,4), block (32,4,1), w8 the (12,16)
  constant). The s74 manifest and sidecar agree on every parameter word
  across three frames: w0 expansion input (36,731,904 — slot 73's output),
  w1 residual (56,589,312 — slot 72's projection output, bound again as the
  cycle's residual), w2 contraction output (56,785,920 — 196,608 B, closing
  at 56,982,528), w3 weights (the DLL-record identity is pinned by the
  runtime layout walk and the slot's heap offset), w4 third output
  (77,824 — 64 B), w5 second output (51,133,440 — 393,216 B, closing at
  51,526,656), w6 sync-in (77,312 — slot 73's 256-B publication), w7
  publication (78,336 — 64 B, next in the 512-byte stride; downstream
  consumption is left for slot 75 to test), and w8 the
  (12,16) constant. The measured write-set has four segments per frame,
  zero unattributed: w5 391,689/391,742/391,717 changed bytes, w2
  196,552/196,550/196,552, and 64 B each at w4 and w7; frame totals
  588,369/588,420/588,397. The wrong control rotates the live weight record
  inwards 1,024 B; tail poisoning after the record reproduces the live execute pool.
  The chained variant is live and its fp8 sibling is created but never
  launched; the sibling matches all three outputs, including the shared w4 publication,
  while leaving the 64-byte w7 handoff at input state.

- slot 75 `cc_vit_1d_qkv_chained_fp8` (s74 -> s75): block34 qkv,
  sharing slot 70's ten-word map. Reads slot 74's contraction at w0
  and its 64-byte w7 handoff, binds live block34.layer2.layer weights
  (3,145,856 B at heap 69,185,024), and writes three 196,608-byte
  q/k/v tensors, a 1,179,648-byte fourth region, 128-byte output data
  at w5 and a 128-byte sync handoff at w8. Six measured segments total
  1,760,497/1,759,938/1,760,032 changed bytes, zero unattributed.

- slot 82 `cc_vit_1d_projection_chained_fp8` (s81 -> s82): block35
  projection repeats the nine-word contraction/projection map at grid
  (16,1,4), block (32,4,1). It reads attention output at 58,752,000 and
  residual at 57,965,568; w3 binds `block35.layer4.layer` (1,050,624 B at
  heap offset 84,920,320); w6 consumes slot81's 256-byte sync at 83,968.
  Outputs: w2 at 58,948,608 (196,608 B), w5 at 26,508,288 (3,932,160 B),
  w4 data at 58,358,784 (64 B), w7 sync at 84,992 (64 B); constant w8 is
  `(12,16)`. The s81→s82 write-set has four attributed regions/frame,
  zero unattributed: w5 391,232/391,180/391,210 B; w2
  196,504/196,446/196,447 B; w4 64 B; w7 64 B. Totals
  587,884/587,690/587,721 B. Sibling should match all output regions and
  diverge only across the 64-byte sync handoff.

- slot 81 `cc_vit_1d_attention_chained_fp8` (s80 -> s81): block35
  attention, sharing slot76's eight-word map. Reads slot80's q/k/v w0/w1/w2
  at 58,358,784 / 58,161,024 / 58,557,824 and consumes its 128-byte w8
  handoff at 83,456. w4 binds the two-byte `block35.layer3.layer` placeholder
  at heap offset 84,919,808. Writes w3 output at 58,752,000 (196,608 B)
  and w6 sync at 83,968 (256 B). The s80→s81 write-set has two regions/frame,
  zero unattributed: w3 changes 196,519/196,547/196,532 B and w6 256 B each;
  totals 196,775/196,803/196,788. Sibling should match output and diverge
  only across its 256-byte w6 sync handoff.

- slot 80 `cc_vit_1d_qkv_chained_fp8` (s79 -> s80): block35 qkv,
  sharing slot 75's ten-word map. Reads slot79 contraction output at w0
  (57,965,568) and its 64-byte w7 handoff (82,432); w4 binds live
  `block35.layer2.layer` (3,145,856 B at heap 81,773,568). Writes q/k/v
  regions at w1/w2/w3 (58,358,784 / 58,161,024 / 58,557,824; 196,608 B
  each), fourth region w6 (33,192,960; 1,179,648 B), w5 output data
  (82,944; 128 B), and w8 handoff (83,456; 128 B) consumed by slot81 w5.
  Six s79→s80 segments per frame, all bytes attributed: frame totals
  1,760,641/1,759,249/1,759,573; w6 1,171,121/1,169,755/1,170,057;
  w3 196,552/196,495/196,543; w1 196,533/196,561/196,552; w2
  196,179/196,182/196,165; w5 and w8 128 each. The first same-grid qkv
  dispatch in each trace is an earlier slot; slot80 is the dispatch directly
  preceding the `cut-boundary slot=80` marker. Replay verifies live/sibling
  behaviour and controls below.

- slot 76 `cc_vit_1d_attention_chained_fp8` (s75 -> s76): block34
  attention, sharing slot 71's eight-word map. Reads the slot-75 q/k/v
  triplet and its 128-byte handoff; w4 binds the two-byte placeholder
  block34.layer3.layer at heap 72,331,264. Writes 196,608 B at w3
  (57,572,352) and a 256-byte w6 handoff at 79,872, consumed by slot 77.
  Two measured segments: w3 196,530/196,532/196,520 changed bytes +
  256 B at w6; totals 196,786/196,788/196,776, zero unattributed.
  The wrong control rotates the query at w0, as for earlier attention gates.

Inputs come from retained evidence only:

- the launch specification (grid, block, shared memory, parameter block) is
  parsed from the forwarder traces' slot records and pinned equal across
  evaluations, runs and the capture sidecar;
- the input pool state is the ladder capture after slot n-1 (exactly the
  state slot n reads), the reference pool state the capture after slot n;
- the weight bytes are extracted from the model DLL's WEIGHTS_HT resource
  through the measured heap layout (record payloads concatenated at 512-byte
  alignment): slot 24 binds `block23.layer0.layer`, slot 25 binds
  `block23.layer1.layer`, slot 26 binds `block23.layer2.layer`, slot 27
  binds `block23.layer3.layer`, slot 28 binds `block24.layer0.layer` (the
  record immediately after `block23.layer3.layer`), slot 29 binds
  `block24.layer1.layer` (the record immediately after `block24.layer0.layer`
  — the record-ordering pattern repeats in block24), slot 30 binds
  `block24.layer2.layer` (the record immediately after `block24.layer1.layer`),
  slot 31 binds `block24.layer3.layer` (the record immediately after
  `block24.layer2.layer`, at heap offset 10,407,424), slot 32 binds
  `block25.layer0.layer` (the record immediately after
  `block24.layer3.layer`), slot 33 binds `block25.layer1.layer` (the
  record immediately after `block25.layer0.layer`, at heap offset
  11,194,880), slot 34 binds `block25.layer2.layer` (the record
  immediately after `block25.layer1.layer`, at heap offset 11,458,048),
  slot 35 binds `block25.layer3.layer` (the record immediately after
  `block25.layer2.layer`, at heap offset 12,376,064), slot 36 binds
  `block26.layer0.layer` (the record immediately after
  `block25.layer3.layer`, at heap offset 12,639,232), slot 37 binds
  `block26.layer1.layer` (the record immediately after
  `block26.layer0.layer`, at heap offset 13,163,520), slot 38 binds
  `block26.layer2.layer` (the record immediately after
  `block26.layer1.layer`, at heap offset 13,426,688), slot 39 binds
  `block26.layer3.layer` (the record immediately after
  `block26.layer2.layer`, at heap offset 14,344,704), slot 40 binds
  `block27.layer0.layer` (the record immediately after
  `block26.layer3.layer`, at heap offset 14,607,872), slot 41 binds
  `block27.layer1.layer` (the record immediately after
  `block27.layer0.layer`, at heap offset 15,132,160), slot 42 binds
  `block27.layer2.layer` (the record immediately after
  `block27.layer1.layer`, at heap offset 15,395,328), slot 43 binds
  `block27.layer3.layer` (the record immediately after
  `block27.layer2.layer`, at heap offset 16,313,344), slot 44 binds
  `block28.layer0.layer` (the record immediately after
  `block27.layer3.layer`, at heap offset 16,576,512), slot 45 binds
  `block28.layer1.layer` (the record immediately after
  `block28.layer0.layer`, at heap offset 17,100,800), slot 46 binds
  `block28.layer2.layer` (the record immediately after
  `block28.layer1.layer`, at heap offset 17,363,968), slot 47 binds
  `block28.layer3.layer` (the record immediately after
  `block28.layer2.layer`, at heap offset 18,281,984), slot 48 binds
  `block29.layer0.layer` (the record immediately after
  `block28.layer3.layer`, at heap offset 18,545,152), slot 49 binds
  `block29.layer1.layer` (the record immediately after
  `block29.layer0.layer`, at heap offset 19,069,440), slot 50 binds
  `block29.layer2.layer` (the record immediately after
  `block29.layer1.layer`, at heap offset 19,332,608), slot 51 binds
  `block29.layer3.layer` (the record immediately after
  `block29.layer2.layer`, at heap offset 20,250,624), slot 52 binds
  `block30.layer0.layer` (the record after the intervening
  `block3.layer0.layer` — 20,672 B at 20,513,792 — at heap offset
  20,534,784), slot 53 binds `block30.layer1.layer` (the record
  immediately after `block30.layer0.layer`, at heap offset 21,059,072),
  slot 54 binds `block30.layer2.layer` (the record immediately after
  `block30.layer1.layer`, at heap offset 21,322,240), slot 55 binds
  `block30.layer3.layer` (the record immediately after
  `block30.layer2.layer`, at heap offset 22,240,256), slot 56 binds
  `block30.layer4.layer` (the record immediately after
  `block30.layer3.layer`, at heap offset 22,503,424), slot 58 binds
  `block31.layer0.layer` (the record immediately after
  `block30.layer4.layer`, at heap offset 23,028,224 — the ViT records'
  first; slot 57's repack block is weights-less), slot 59 binds
  `block31.layer1.layer` (the record immediately after
  `block31.layer0.layer`, at heap offset 27,223,040), slot 60 binds
  `block31.layer2.layer` (the record immediately after
  `block31.layer1.layer`, at heap offset 31,419,392 — closing exactly at
  its end, no padding intervenes), slot 61 binds `block31.layer3.layer`
  (the record immediately after `block31.layer2.layer`, at heap offset
  34,565,632 — `block31.layer2.layer`'s end at 34,565,248 rounded up at
  the 512 boundary; the record carries a 2-byte payload, the ViT attention
  blocks' layer3 placeholder shape — a dead binding, see the contract-61
  entry), slot 62 binds `block31.layer4.layer` (the record immediately
  after `block31.layer3.layer`, at heap offset 34,566,144 —
  `block31.layer3.layer`'s 2-byte payload at 34,565,632 rounded up across
  510 bytes of padding to the 512 boundary, closing block31's five-record
  set), slot 63 binds `block32.layer0.layer` (the record immediately after
  `block31.layer4.layer`, at heap offset 35,616,768 — block31's
  five-record set ends there exactly, no padding intervenes, block32's
  first record; the lookup walks block32's five-record set
  `block32.layer0.layer` (4,194,320 B), `block32.layer1.layer` (4,196,352
  B), `block32.layer2.layer` (3,145,856 B), `block32.layer3.layer` (2 B)
  and `block32.layer4.layer` (1,050,624 B) — block31's shape repeating
  exactly), slot 64 binds `block32.layer1.layer` (the record immediately
  after `block32.layer0.layer`, at heap offset 39,811,584 —
  `block32.layer0.layer`'s 4,194,320-byte payload at 35,616,768 rounded
  up across 496 bytes of padding to the 512 boundary), slot 65 binds
  `block32.layer2.layer` (the record immediately after
  `block32.layer1.layer`, at heap offset 44,007,936 —
  `block32.layer1.layer`'s 4,196,352-byte payload at 39,811,584 ends on
  the 512 boundary exactly, no padding intervenes), slot 66 binds
  `block32.layer3.layer` (the record immediately after
  `block32.layer2.layer`, at heap offset 47,154,176 —
  `block32.layer2.layer`'s 3,145,856-byte payload at 44,007,936 rounded
  up across 384 bytes of padding to the 512 boundary; the ViT attention
  blocks' placeholder record — a 2-byte payload), slot 67 binds
  `block32.layer4.layer` (the record immediately after
  `block32.layer3.layer`, at heap offset 47,154,688 — the 2-byte payload
  at 47,154,176 rounded up across 510 bytes of padding to the 512
  boundary, closing block32's five-record set exactly at
  `block33.layer0.layer`'s start at 48,205,312 — the block31
  layer3-to-layer4 pair repeating exactly), slot 68 binds
  `block33.layer0.layer` (block33's first record, at heap offset
  48,205,312 — block32's layer4's 1,050,624-byte payload closing at that
  boundary with no padding; 496 bytes of padding then close the gap to
  slot 69's `block33.layer1.layer` at 52,400,128), slot 69 binds
  `block33.layer1.layer` (the record immediately after
  `block33.layer0.layer`, at heap offset 52,400,128 — the 4,194,320-byte
  payload at 48,205,312 plus 496 bytes of padding to the 512 boundary),
  slot 70 binds
  `block33.layer2.layer` (the record immediately after
  `block33.layer1.layer`, at heap offset 56,596,480 — the 4,196,352-byte
  payload at 52,400,128 closing exactly there with no padding), slot 71 binds
  `block33.layer3.layer` (block33's 2-byte placeholder record, at heap
  offset 59,742,720 — the layout walk's next record start after
  `block33.layer2.layer`, whose 3,145,856-byte payload at 56,596,480 ends
  at 59,742,336 and rounds up across 384 bytes at the 512 boundary),
  slot 72 binds
  `block33.layer4.layer` (block33's closing projection record, at heap
  offset 59,743,232 — the layout walk's next record start after
  `block33.layer3.layer`, whose 2-byte payload at 59,742,720 rounds up
  across 510 bytes of padding to the 512 boundary, closing block33's
  five-record set exactly at `block34.layer0.layer`'s start at
  60,793,856), slot 73 binds `block34.layer0.layer` (the first record
  of block34, at heap offset 60,793,856 — exactly after block33's closing
  `block33.layer4.layer` payload; its own 4,194,320-byte payload is followed
  by 496 bytes of padding before `block34.layer1.layer`), slot 74 binds
  `block34.layer1.layer` (4,196,352 B at heap 64,988,672), slot 75 binds
  `block34.layer2.layer` (3,145,856 B at heap 69,185,024), and slot 76
  binds `block34.layer3.layer` (2 B at heap 72,331,264)
  — all exact
  record-start landings under the decoder-anchored layout
  (DECODER-BOUNDARY.md word 7 at `block39.layer0.layer`).

Primary claim (execute mode): for each slot's vendor-launched variant the
native pool readback equals the vendor's captured pool byte-for-byte over
the whole 62,684,160-byte pool. Each contract's fp8 sibling (created but
never launched by the vendor — pinned) is replayed at the live geometry as a
variant probe: its output tensor must match, and its sync record is recorded
per entry (the decoder stage found the same family record at out2; the
record's changed extent is per slot — 96 B at the decoder and slots 24, 25,
27, 28, 29, 31, 32, 33, 35, 36, 37, 39, 40, 41, 43, 44, 45, 47, 48, 49, 51, 52, 53 and 55, 192 B at slots 26, 34,
42 and 50, 256 B at slots 30, 38, 46, 54, 58, 61, 63, 66, 68, 71, 73 and 76, 64 B at slots 59, 62, 64, 67, 69, 72 and 74, 128 B at
slots 60, 65, 70, 75 and 80, 256 B at slots 78, 81 and 83, 64 B at slots 77, 79, 82 and 84 — and 0 B at slot 56, the
first `wait`-family kernel: it waits on w3's record without writing one,
so its sibling's divergence pin of 0 enforces whole-pool sibling equality,
the first contract to need it). Single-variant contracts (slot 57: the
kernel family has no fp8 sibling at all) drop the sibling replay; the
vendor launches the only variant and the whole-pool compare alone carries
the claim. Controls: skip (no launch), wrong (weights rotated
inwards — one byte for records no longer than the rotation extent; the full
rotation would be a no-op, and a rotation that leaves the record unchanged is
rejected outright), tail
(bytes beyond the record replaced by a different pattern —
bounding weight reads to the record). A contract whose weights record is a
dead binding re-bases the wrong rotation onto the input tensor the kernel
functionally reads (pinned per contract as `wrong_word`): the weight-free
attention family (slots 61, 66, 71, 76 and 81) is invariant under whole-record
complementation (measured in both variants at the captured conditions), so
its wrong control rotates the query tensor at w0 and the record stays at
its live content. Weights-less contracts (slot 57: a
pure data-movement kernel with no weights word in the block) re-base the
same three controls onto the input tensor: wrong rotates the input tensor
inwards (the output must change — the repack derives from its input), and
the tail bytes go into the pool at the output extent's pre-launch bytes —
the bytes immediately beyond the input tensor — in both patterns (execute
0xA5 / tail 0x5A), so unchanged outputs under two different tail patterns
bound the kernel's reads to the input tensor. The tail window sits inside
the output extent because that is the only pool state the kernel
rewrites, and the pattern stages for the launched modes only (skip has no
launch to bound; its pool is the pristine input state, keeping its diff
counts comparable with the ladder write-sets).

This stage uses stored learned weight bytes through extracted records
(bounded model-weight computation, NATIVE-FP8.md precedent) and is not a
neural-rendered image. Evidence retains digests and bounded windows only;
reuse/publication rights remain unresolved.

"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import ctypes as ct
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import struct
import sys
from typing import Any


HERE = Path(__file__).resolve().parent


def load_helper(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError("missing helper")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


base = load_helper("base", "native-runtime-probe.py")
utility = load_helper("utility", "native-kernel-probe.py")
launch = load_helper("launch", "native-decoder-launch-probe.py")
compare = load_helper("compare", "native-vendor-compare-probe.py")
names = load_helper("names", "ladder-names.py")

# ---------------------------------------------------------------------------
# Pinned constants
# ---------------------------------------------------------------------------

CONTAINER = 4
FRAMES = 3
ANCHOR_RECORD = "block39.layer0.layer"
HEAP_SIZE = 147719680
OUTPUT_BYTES = 344064   # binding extent 24x28x512 (LADDER-NAMES.md segments)
ROTATE_BYTES = 1024
TAIL_BYTES = 4096
EXEC_TAIL = 0xA5
TAIL_TAIL = 0x5A
GUARD_WORDS = launch.GUARD_WORDS
CLS_POOL = "pool"
CLS_WEIGHTS = "weights"
CLS_CONSTANT = "constant"
CLS_ZERO = "zero"

# Per-slot replay contracts. word_classes classifies every 8-byte word of the
# slot's parameter block in index order: pool (working-pool address, replayed
# at the same pool offset), weights (weight-heap record start, the extracted
# record replayed at our allocation), constant (passed through from the
# captured block, value pinned per word in constant_values), zero (reserved,
# must be 0). extra_outputs lists (word, span) pairs of further written
# output tensors beyond the primary output_word binding (slot 55's pooled
# tensor at w3, 98,304 bytes; slots 59, 62, 64, 67, 69, 72, 74, 79 and 84's second and third outputs at
# w5, 393,216 bytes, and w4, 64 bytes; slot 77's second output at w5
# (3,932,160 bytes), and w4 (64 bytes); slots 60, 65, 70, 75 and 80's q/k/v siblings at w2
# and w3,
# 196,608 bytes each, their w6 region at 1,179,648 bytes and their w5
# publication at 128 bytes; slots 61, 66, 71, 76 and 81's attention blocks write no extra
# output — their only other written region is the sync record); the primary
# binding's own extent is
# OUTPUT_BYTES unless the contract pins output_bytes (slots 56 and 57's
# 196,608-byte tensors, slots 58, 63, 68, 73, 78 and 83's 786,432-byte expansions, slots 59, 62, 64, 67, 69, 72, 74, 77 and 79/82/84's
# contractions/projections, slots 60, 65, 70, 75 and 80's q/k/v and slots 61, 66, 71, 76 and 81's attention outputs
# likewise). input_word/input_bytes pin the input tensor
# binding for weights-less contracts (the wrong control's rotation target
# and the read-bounding claim's subject). A contract with no weights word
# (CLS_WEIGHTS absent: slot 57's data-movement block) carries
# record_name=None and record_bytes=0; its controls re-base onto the input
# tensor (module docstring, Controls). sync_bytes is the sync record's
# changed extent,
# measured per
# slot in the ladder write-sets (LADDER-NAMES.md segments;
# 96/96/192/96/96/96/256/96/96/96/192/96/96/96/256/96/96/96/192/96/96/96/256/96/96/96/192/96/96/96/256/96/0/0/256/64/128/256/64/256/64/128/256/64/256/64/128/256/64/256/64/128/256/64 for
# slots 24/25/26/27/28/29/30/31/32/33/34/35/36/37/38/39/40/41/42/43/44/45/46/47/48/49/50/51/52/53/54/55/56/57/58/59/60/61/62/63/64/65/66/67/68/69/70/71/72/73/74/75/76/77/78/79/80/81/82/83/84) — the fp8
# sibling's divergence from the reference is enforced to equal it exactly
# (0 at slot 56: the wait family waits on w3's record without writing one,
# so the pin enforces whole-pool sibling equality; 0 at slot 57 with no
# sync word at all — the repack family is outside the sync chain, so
# sync_word is None and the compare is empty by construction).
# constant_values maps constant-word index to its pinned value (a block can
# carry more than one, at more than one value: slot 30's w4 and slot 31's w8
# are 0xFFFFFFFCFFFFFFFC where the same-shape slots 26/27 reserve those
# words as zero; slot 34's w4 and slot 35's w8 are 0x00000000FFFFFFFC — the
# low dword only; slot 38's w4 and slot 39's w8 are 0xFFFFFFFC00000000 — the
# high dword only; slots 42's w4 and 43's w8 are reserved zero like slots
# 26's/27's — block27's qkv-proj pair carries no second constant (block23's
# pair likewise), and slot 46's w4 and slot 47's w8 are both
# 0xFFFFFFFCFFFFFFFC — the double form, the absent/double/low/high cycle's
# block28 step measured in both halves, and slot 50's w4 and slot 51's w8
# are both 0x00000000FFFFFFFC — the low-dword form, block29's step now
# measured in both halves too, and slot 54's w4 is 0xFFFFFFFC00000000 —
# the high-dword form, block30's step measured in the qkv half — slot 55's
# w7 carries the same value at the proj_pool half's own word position, the
# pair's step now measured in both halves).
# One value per qkv-proj pair where
# present, shared by both halves).
CONTRACTS: dict[int, dict[str, Any]] = {
    24: dict(
        slot=24,
        psize=56,
        name_live="cc_split_swin_16h_ffwd_inpview_512_tilesync_fp8",
        name_sibling="cc_split_swin_16h_ffwd_inpview_512_fp8",
        tag_live="tilesync", tag_sibling="fp8",
        record_name="block23.layer0.layer", record_bytes=524288,
        constant_values={3: 0x1C00000018},   # recovered (24, 28) geometry overrides
        word_classes=(CLS_POOL, CLS_POOL, CLS_WEIGHTS, CLS_CONSTANT,
                      CLS_ZERO, CLS_ZERO, CLS_POOL),
        output_word=1, sync_word=6, sync_bytes=96,
        input_label=23, reference_label=24),
    25: dict(
        slot=25,
        psize=72,
        name_live="cc_split_swin_16h_ffwd_proj_inpview_512_chained_fp8",
        name_sibling="cc_split_swin_16h_ffwd_proj_inpview_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        record_name="block23.layer1.layer", record_bytes=263168,
        constant_values={4: 0x1C00000018},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS,
                      CLS_CONSTANT, CLS_ZERO, CLS_POOL, CLS_POOL, CLS_ZERO),
        output_word=2, sync_word=6, sync_bytes=96,
        input_label=24, reference_label=25),
    26: dict(
        slot=26,
        psize=56,
        name_live="cc_split_swin_16h_qkv_512_chained_fp8",
        name_sibling="cc_split_swin_16h_qkv_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        record_name="block23.layer2.layer", record_bytes=917568,
        constant_values={3: 0x1C00000018},
        word_classes=(CLS_POOL, CLS_POOL, CLS_WEIGHTS, CLS_CONSTANT,
                      CLS_ZERO, CLS_POOL, CLS_POOL),
        output_word=1, sync_word=6, sync_bytes=192,
        input_label=25, reference_label=26),
    27: dict(
        slot=27,
        psize=72,
        name_live="cc_split_swin_16h_proj_512_chained_fp8",
        name_sibling="cc_split_swin_16h_proj_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        record_name="block23.layer3.layer", record_bytes=263168,
        constant_values={4: 0x1C00000018},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS,
                      CLS_CONSTANT, CLS_ZERO, CLS_POOL, CLS_POOL, CLS_ZERO),
        output_word=2, sync_word=6, sync_bytes=96,
        input_label=26, reference_label=27),
    28: dict(
        slot=28,
        psize=56,
        name_live="cc_split_swin_16h_ffwd_512_chained_fp8",
        name_sibling="cc_split_swin_16h_ffwd_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        record_name="block24.layer0.layer", record_bytes=524288,
        constant_values={3: 0x1C00000018},
        word_classes=(CLS_POOL, CLS_POOL, CLS_WEIGHTS, CLS_CONSTANT,
                      CLS_POOL, CLS_ZERO, CLS_POOL),
        output_word=1, sync_word=6, sync_bytes=96,
        input_label=27, reference_label=28),
    29: dict(
        slot=29,
        psize=72,
        name_live="cc_split_swin_16h_ffwd_proj_512_chained_fp8",
        name_sibling="cc_split_swin_16h_ffwd_proj_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        record_name="block24.layer1.layer", record_bytes=263168,
        constant_values={4: 0x1C00000018},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS,
                      CLS_CONSTANT, CLS_ZERO, CLS_POOL, CLS_POOL, CLS_ZERO),
        output_word=2, sync_word=6, sync_bytes=96,
        input_label=28, reference_label=29),
    30: dict(
        slot=30,
        psize=56,
        name_live="cc_split_swin_16h_qkv_512_chained_fp8",
        name_sibling="cc_split_swin_16h_qkv_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        record_name="block24.layer2.layer", record_bytes=917568,
        constant_values={3: 0x1C00000018, 4: 0xFFFFFFFCFFFFFFFC},
        word_classes=(CLS_POOL, CLS_POOL, CLS_WEIGHTS, CLS_CONSTANT,
                      CLS_CONSTANT, CLS_POOL, CLS_POOL),
        output_word=1, sync_word=6, sync_bytes=256,
        input_label=29, reference_label=30),
    31: dict(
        slot=31,
        psize=72,
        name_live="cc_split_swin_16h_proj_512_chained_fp8",
        name_sibling="cc_split_swin_16h_proj_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        record_name="block24.layer3.layer", record_bytes=263168,
        constant_values={4: 0x1C00000018, 8: 0xFFFFFFFCFFFFFFFC},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS,
                      CLS_CONSTANT, CLS_ZERO, CLS_POOL, CLS_POOL,
                      CLS_CONSTANT),
        output_word=2, sync_word=6, sync_bytes=96,
        input_label=30, reference_label=31),
    32: dict(
        slot=32,
        psize=56,
        name_live="cc_split_swin_16h_ffwd_512_chained_fp8",
        name_sibling="cc_split_swin_16h_ffwd_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        record_name="block25.layer0.layer", record_bytes=524288,
        constant_values={3: 0x1C00000018},
        word_classes=(CLS_POOL, CLS_POOL, CLS_WEIGHTS, CLS_CONSTANT,
                      CLS_POOL, CLS_ZERO, CLS_POOL),
        output_word=1, sync_word=6, sync_bytes=96,
        input_label=31, reference_label=32),
    33: dict(
        slot=33,
        psize=72,
        name_live="cc_split_swin_16h_ffwd_proj_512_chained_fp8",
        name_sibling="cc_split_swin_16h_ffwd_proj_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        record_name="block25.layer1.layer", record_bytes=263168,
        constant_values={4: 0x1C00000018},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS,
                      CLS_CONSTANT, CLS_ZERO, CLS_POOL, CLS_POOL, CLS_ZERO),
        output_word=2, sync_word=6, sync_bytes=96,
        input_label=32, reference_label=33),
    34: dict(
        slot=34,
        psize=56,
        name_live="cc_split_swin_16h_qkv_512_chained_fp8",
        name_sibling="cc_split_swin_16h_qkv_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        record_name="block25.layer2.layer", record_bytes=917568,
        constant_values={3: 0x1C00000018, 4: 0x00000000FFFFFFFC},
        word_classes=(CLS_POOL, CLS_POOL, CLS_WEIGHTS, CLS_CONSTANT,
                      CLS_CONSTANT, CLS_POOL, CLS_POOL),
        output_word=1, sync_word=6, sync_bytes=192,
        input_label=33, reference_label=34),
    35: dict(
        slot=35,
        psize=72,
        name_live="cc_split_swin_16h_proj_512_chained_fp8",
        name_sibling="cc_split_swin_16h_proj_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        record_name="block25.layer3.layer", record_bytes=263168,
        # Both halves of block25's qkv-proj pair carry the second constant at
        # the same value (the low dword only), like block24's pair at its own
        # value; block23's pair and every ffwd-proj half keep the word zero.
        constant_values={4: 0x1C00000018, 8: 0x00000000FFFFFFFC},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS,
                      CLS_CONSTANT, CLS_ZERO, CLS_POOL, CLS_POOL,
                      CLS_CONSTANT),
        output_word=2, sync_word=6, sync_bytes=96,
        input_label=34, reference_label=35),
    36: dict(
        slot=36,
        psize=56,
        name_live="cc_split_swin_16h_ffwd_512_chained_fp8",
        name_sibling="cc_split_swin_16h_ffwd_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        record_name="block26.layer0.layer", record_bytes=524288,
        constant_values={3: 0x1C00000018},
        word_classes=(CLS_POOL, CLS_POOL, CLS_WEIGHTS, CLS_CONSTANT,
                      CLS_POOL, CLS_ZERO, CLS_POOL),
        output_word=1, sync_word=6, sync_bytes=96,
        input_label=35, reference_label=36),
    37: dict(
        slot=37,
        psize=72,
        name_live="cc_split_swin_16h_ffwd_proj_512_chained_fp8",
        name_sibling="cc_split_swin_16h_ffwd_proj_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        record_name="block26.layer1.layer", record_bytes=263168,
        # An ffwd-proj half reserves the pair's second constant (like slots
        # 25/29/33): w5 and w8 are zero here.
        constant_values={4: 0x1C00000018},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS,
                      CLS_CONSTANT, CLS_ZERO, CLS_POOL, CLS_POOL, CLS_ZERO),
        output_word=2, sync_word=6, sync_bytes=96,
        input_label=36, reference_label=37),
    38: dict(
        slot=38,
        psize=56,
        name_live="cc_split_swin_16h_qkv_512_chained_fp8",
        name_sibling="cc_split_swin_16h_qkv_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        record_name="block26.layer2.layer", record_bytes=917568,
        # Block26's qkv-proj pair carries its second constant at the
        # high-dword-only value (w4 here, slot 39's w8) — the third geometry
        # of the same marker dword after block24's double and block25's
        # low-dword-only forms; one value per pair, shared across the halves.
        constant_values={3: 0x1C00000018, 4: 0xFFFFFFFC00000000},
        word_classes=(CLS_POOL, CLS_POOL, CLS_WEIGHTS, CLS_CONSTANT,
                      CLS_CONSTANT, CLS_POOL, CLS_POOL),
        output_word=1, sync_word=6, sync_bytes=256,
        input_label=37, reference_label=38),
    39: dict(
        slot=39,
        psize=72,
        name_live="cc_split_swin_16h_proj_512_chained_fp8",
        name_sibling="cc_split_swin_16h_proj_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        record_name="block26.layer3.layer", record_bytes=263168,
        # Block26's proj half is the qkv-proj pair's second half and carries
        # the pair's second constant at w8 — the high-dword-only form shared
        # with slot 38's w4 (one value per pair, both halves). Recovered from
        # the s39 capture's own boundary words (w8 = 0xFFFFFFFC00000000).
        constant_values={4: 0x1C00000018, 8: 0xFFFFFFFC00000000},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS,
                      CLS_CONSTANT, CLS_ZERO, CLS_POOL, CLS_POOL,
                      CLS_CONSTANT),
        output_word=2, sync_word=6, sync_bytes=96,
        input_label=38, reference_label=39),
    40: dict(
        slot=40,
        psize=56,
        name_live="cc_split_swin_16h_ffwd_512_chained_fp8",
        name_sibling="cc_split_swin_16h_ffwd_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        # Block27's ffwd half opens the next ffwd-proj pair and repeats
        # block23-26's layer0 shape (524,288 B). Measured from the s40
        # capture: word classes (POOL, POOL, WEIGHTS, CONSTANT, POOL, ZERO,
        # POOL), w1 output (pool+26,459,136), w3 constant 0x1C00000018, w6
        # sync (pool+56,832, 96 B — the ffwd norm, 512-stride continuing).
        # The weights word lands on the layout walk's next record start
        # (block27.layer0.layer, 524,288 B at heap offset 14,607,872 =
        # 14,344,704 + 263,168, confirmed by the probe's live DLL walk).
        record_name="block27.layer0.layer", record_bytes=524288,
        constant_values={3: 0x1C00000018},
        word_classes=(CLS_POOL, CLS_POOL, CLS_WEIGHTS, CLS_CONSTANT,
                      CLS_POOL, CLS_ZERO, CLS_POOL),
        output_word=1, sync_word=6, sync_bytes=96,
        input_label=39, reference_label=40),
    41: dict(
        slot=41,
        psize=72,
        name_live="cc_split_swin_16h_ffwd_proj_512_chained_fp8",
        name_sibling="cc_split_swin_16h_ffwd_proj_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        # Block27's ffwd-proj pair's second half (the ffwd_proj counterpart
        # to slots 25/29/33/37, same 72-byte block and grid 8,3,1).
        # Measured from the s41 capture: word classes (POOL, POOL, POOL,
        # WEIGHTS, CONSTANT, ZERO, POOL, POOL, ZERO), w0 input
        # (pool+26,459,136 — address-equal to slot 40's w1), w1 residual
        # (pool+26,115,072 — the buffer slot 39 wrote its output to),
        # w2 output (pool+30,833,664), w4 constant 0x1C00000018, w6 sync
        # (pool+57,344, 96 B — the ffwd norm, 512-stride continuing), w7
        # sync-in (pool+56,832 — address-equal to slot 40's w6, the chained
        # handoff), w5/w8 reserved zero (the pair second constant is absent
        # from every ffwd-proj half). The weights word lands on the layout
        # walk's next record start (block27.layer1.layer, 263,168 B at heap
        # offset 15,132,160 = 14,607,872 + 524,288 at the 512 boundary).
        record_name="block27.layer1.layer", record_bytes=263168,
        constant_values={4: 0x1C00000018},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS,
                      CLS_CONSTANT, CLS_ZERO, CLS_POOL, CLS_POOL, CLS_ZERO),
        output_word=2, sync_word=6, sync_bytes=96,
        input_label=40, reference_label=41),
    42: dict(
        slot=42,
        psize=56,
        name_live="cc_split_swin_16h_qkv_512_chained_fp8",
        name_sibling="cc_split_swin_16h_qkv_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        # Block27's qkv-proj pair's first half (the qkv counterpart to slots
        # 26/30/34/38, same 56-byte block and grid 4,3,4). Measured from the
        # s42 capture: word classes (POOL, POOL, WEIGHTS, CONSTANT, ZERO,
        # POOL, POOL) — the slot-26 map exactly: w0 output-of-slot-41
        # (30,833,664), w1 output (31,177,728 — one stride past its input),
        # w2 weights, w3 the (24, 28) constant, w4 reserved zero (block27's
        # qkv half carries no second constant; its proj half's w8 measured
        # zero in the s43 capture), w5 sync-in
        # (slot 41's record at 57,344), w6 sync-out (57,856).
        record_name="block27.layer2.layer", record_bytes=917568,
        constant_values={3: 0x1C00000018},
        word_classes=(CLS_POOL, CLS_POOL, CLS_WEIGHTS, CLS_CONSTANT,
                      CLS_ZERO, CLS_POOL, CLS_POOL),
        output_word=1, sync_word=6, sync_bytes=192,
        input_label=41, reference_label=42),
    43: dict(
        slot=43,
        psize=72,
        name_live="cc_split_swin_16h_proj_512_chained_fp8",
        name_sibling="cc_split_swin_16h_proj_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        # Block27's proj half — the qkv-proj pair's second half (the
        # slot-27/31/35/39 word map: POOL, POOL, POOL, WEIGHTS, CONSTANT,
        # ZERO, POOL, POOL, ZERO). Measured from the s43 capture: w0
        # output-of-slot-42 (31,177,728), w1 residual (30,833,664), w2
        # output (31,521,792 — one stride past its input, the regular output
        # spacing), w4 the (24, 28) constant, w6 sync-out (58,368), w7
        # sync-in (slot 42's record at 57,856) and w8 measured reserved zero
        # — block27's pair second constant absent (slot 42's w4 likewise),
        # the cycle hypothesis' absent step confirmed in both halves.
        record_name="block27.layer3.layer", record_bytes=263168,
        constant_values={4: 0x1C00000018},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS,
                      CLS_CONSTANT, CLS_ZERO, CLS_POOL, CLS_POOL, CLS_ZERO),
        output_word=2, sync_word=6, sync_bytes=96,
        input_label=42, reference_label=43),
    44: dict(
        slot=44,
        psize=56,
        name_live="cc_split_swin_16h_ffwd_512_chained_fp8",
        name_sibling="cc_split_swin_16h_ffwd_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        # Block28's ffwd half opens the next ffwd-proj pair and repeats
        # block24-27's layer0 shape (524,288 B at the 512 boundary). Measured
        # from the s44 capture's own parameter block: word classes (POOL,
        # POOL, WEIGHTS, CONSTANT, POOL, ZERO, POOL) — the slot-28/32/36/40
        # map exactly: w0 input (slot 43's output region), w1 output, w2
        # weights, w3 the (24, 28) geometry constant 0x1C00000018, w4
        # sync-in (pool+58,368 — address-equal to slot 43's w6, the chained
        # handoff), w5 reserved zero (every ffwd half reserves the word; the
        # ffwd-proj pair carries no second constant), w6 sync-out
        # (pool+58,880 — the 512-byte stride continuing: 57,856 → 58,368 →
        # 58,880). Its weight record block28.layer0.layer (524,288 B at heap
        # offset 16,576,512 = 16,313,344 + 263,168 at the 512 boundary, the
        # block24/26 layer3→next-block pattern repeating) is confirmed by the
        # probe's live DLL walk.
        record_name="block28.layer0.layer", record_bytes=524288,
        constant_values={3: 0x1C00000018},
        word_classes=(CLS_POOL, CLS_POOL, CLS_WEIGHTS, CLS_CONSTANT,
                      CLS_POOL, CLS_ZERO, CLS_POOL),
        output_word=1, sync_word=6, sync_bytes=96,
        input_label=43, reference_label=44),
    45: dict(
        slot=45,
        psize=72,
        name_live="cc_split_swin_16h_ffwd_proj_512_chained_fp8",
        name_sibling="cc_split_swin_16h_ffwd_proj_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        # Block28's ffwd-proj pair's second half (the slot-25/29/33/37/41
        # word map, same 72-byte block and grid 8,3,1). Measured from the s45
        # capture: word classes (POOL, POOL, POOL, WEIGHTS, CONSTANT, ZERO,
        # POOL, POOL, ZERO), w0 input (31,865,856 — address-equal to slot
        # 44's w1), w1 residual (31,521,792 — the buffer slot 43 wrote its
        # output to, the input slot 44 itself read), w2 output, w4 the
        # (24, 28) geometry constant 0x1C00000018, w5/w8 reserved zero (the
        # ffwd-proj pair carries no second constant), w6 sync-out (59,392 —
        # the 512-byte stride continuing) and w7 sync-in (58,880 —
        # address-equal to slot 44's w6, the chained handoff). Its weight
        # record block28.layer1.layer (263,168 B at heap offset 17,100,800 —
        # 16,576,512 + 524,288 at the 512 boundary) follows the established
        # pattern, confirmed by the probe's live DLL walk.
        record_name="block28.layer1.layer", record_bytes=263168,
        constant_values={4: 0x1C00000018},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS,
                      CLS_CONSTANT, CLS_ZERO, CLS_POOL, CLS_POOL, CLS_ZERO),
        output_word=2, sync_word=6, sync_bytes=96,
        input_label=44, reference_label=45),
    46: dict(
        slot=46,
        psize=56,
        name_live="cc_split_swin_16h_qkv_512_chained_fp8",
        name_sibling="cc_split_swin_16h_qkv_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        # Block28's qkv-proj pair's first half (the slot-26/30/34/38/42
        # word map, same 56-byte block) at the shared qkv handle's second
        # geometry (grid 4,4,4 — the slot-30/38 counterpart). Measured
        # from the s46 capture: word classes (POOL, POOL, WEIGHTS,
        # CONSTANT, CONSTANT, POOL, POOL): w0 input (pool+32,209,920 —
        # address-equal to slot 45's w2), w1 output (pool+32,553,984 —
        # one stride past its input, the regular output spacing), w2
        # weights, w3 the (24, 28) geometry constant, w4 the pair second
        # constant at 0xFFFFFFFCFFFFFFFC (the double form — the cycle's
        # block28 prediction, confirmed in the qkv half), w5 sync-in
        # (pool+59,392 — address-equal to slot 45's w6, the chained
        # handoff) and w6 sync-out (pool+59,904 — the 512-byte stride
        # continuing: 58,880 -> 59,392 -> 59,904). Its weights word lands
        # on the layout walk's next record start (block28.layer2.layer,
        # 917,568 B at heap offset 17,363,968 = 17,100,800 + 263,168,
        # already 512-aligned so no padding intervenes).
        record_name="block28.layer2.layer", record_bytes=917568,
        constant_values={3: 0x1C00000018, 4: 0xFFFFFFFCFFFFFFFC},
        word_classes=(CLS_POOL, CLS_POOL, CLS_WEIGHTS, CLS_CONSTANT,
                      CLS_CONSTANT, CLS_POOL, CLS_POOL),
        output_word=1, sync_word=6, sync_bytes=256,
        input_label=45, reference_label=46),
    47: dict(
        slot=47,
        psize=72,
        name_live="cc_split_swin_16h_proj_512_chained_fp8",
        name_sibling="cc_split_swin_16h_proj_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        # Block28's proj half — the qkv-proj pair's second half (the
        # slot-27/31/35/39/43 word map: POOL, POOL, POOL, WEIGHTS, CONSTANT,
        # ZERO, POOL, POOL, CONSTANT). Measured from the s47 capture: w0
        # output-of-slot-46 (pool+32,553,984 — address-equal to slot 46's
        # w1), w1 residual (pool+32,209,920 — the buffer slot 46 itself read
        # at w0), w2 output (pool+32,898,048 — one stride past its input, the
        # regular output spacing; slot 48's w0 is address-equal), w4 the
        # (24, 28) geometry constant 0x1C00000018, w5 reserved zero (every
        # proj half reserves the word), w6 sync-out (pool+60,416 — the
        # 512-byte stride continuing: 59,904 -> 60,416), w7 sync-in
        # (pool+59,904 — address-equal to slot 46's w6, the chained handoff)
        # and w8 the pair second constant at 0xFFFFFFFCFFFFFFFC — the double
        # form. The absent/double/low/high cycle's predicted block28 step is
        # now measured in BOTH halves (slot 46's w4 carries the same value),
        # so the pair shares one constant; block29's pair then took the
        # cycle's low-dword step at slot 50 (w4 since measured at
        # 0x00000000FFFFFFFC), slot 51 testing the proj half. Its weights word lands on
        # the layout walk's next record start (block28.layer3.layer,
        # 263,168 B at heap offset 18,281,984 = 17,363,968 + 917,568 + 448
        # at the alignment boundary, the layer2->3 pattern of blocks 23-27
        # repeating). The sync extent returns to the 96-byte proj norm after
        # slot 46's 256-byte qkv record.
        record_name="block28.layer3.layer", record_bytes=263168,
        constant_values={4: 0x1C00000018, 8: 0xFFFFFFFCFFFFFFFC},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS,
                      CLS_CONSTANT, CLS_ZERO, CLS_POOL, CLS_POOL, CLS_CONSTANT),
        output_word=2, sync_word=6, sync_bytes=96,
        input_label=46, reference_label=47),
    48: dict(
        slot=48,
        psize=56,
        name_live="cc_split_swin_16h_ffwd_512_chained_fp8",
        name_sibling="cc_split_swin_16h_ffwd_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        # Block29's ffwd half opens block29 and the next ffwd-proj pair.
        # Measured from the s48 capture's own parameter block: word classes
        # (POOL, POOL, WEIGHTS, CONSTANT, POOL, ZERO, POOL) — the
        # slot-28/32/36/40/44 map exactly: w0 output-of-slot-47 (pool
        # +32,898,048), w1 output (pool+33,242,112 — one stride past its
        # input, the regular output spacing), w2 weights, w3 the (24, 28)
        # geometry constant 0x1C00000018, w4 sync-in (pool+60,416 —
        # address-equal to slot 47's w6, the chained handoff), w5 reserved
        # zero (every ffwd half reserves the word; the ffwd-proj pair
        # carries no second constant) and w6 sync-out (pool+60,928 — the
        # 512-byte stride continuing: 59,904 -> 60,416 -> 60,928). Its
        # weights word lands on the layout walk's next record start
        # (block29.layer0.layer, 524,288 B at heap offset 18,545,152 =
        # 18,281,984 + 263,168, already 512-aligned so no padding
        # intervenes — block28.layer3 closes exactly at the boundary),
        # confirmed by the probe's live DLL walk. The sync extent measured
        # 96 B (the ffwd norm) pinned from the s47->s48 write-set (two
        # segments per frame, zero unattributed: 340,937/341,476/341,433
        # tensor + 96 sync) and enforced as the fp8 sibling's divergence.
        record_name="block29.layer0.layer", record_bytes=524288,
        constant_values={3: 0x1C00000018},
        word_classes=(CLS_POOL, CLS_POOL, CLS_WEIGHTS, CLS_CONSTANT,
                      CLS_POOL, CLS_ZERO, CLS_POOL),
        output_word=1, sync_word=6, sync_bytes=96,
        input_label=47, reference_label=48),
    49: dict(
        slot=49,
        psize=72,
        name_live="cc_split_swin_16h_ffwd_proj_512_chained_fp8",
        name_sibling="cc_split_swin_16h_ffwd_proj_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        # Block29's ffwd-proj pair's second half (the slot-25/29/33/37/41/45
        # word map, same 72-byte block and grid 8,3,1). Measured from the s49
        # capture: word classes (POOL, POOL, POOL, WEIGHTS, CONSTANT, ZERO,
        # POOL, POOL, ZERO), w0 input (33,242,112 — address-equal to slot
        # 48's w1), w1 residual (32,898,048 — the buffer slot 47 wrote its
        # output to, the input slot 48 itself read), w2 output, w4 the
        # (24, 28) geometry constant 0x1C00000018, w5/w8 reserved zero (the
        # ffwd-proj pair carries no second constant), w6 sync-out (61,440 —
        # the 512-byte stride continuing) and w7 sync-in (60,928 —
        # address-equal to slot 48's w6, the chained handoff). Its weight
        # record block29.layer1.layer (263,168 B at heap offset 19,069,440 —
        # 18,545,152 + 524,288 at the 512 boundary) follows the established
        # pattern, confirmed by the probe's live DLL walk.
        record_name="block29.layer1.layer", record_bytes=263168,
        constant_values={4: 0x1C00000018},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS,
                      CLS_CONSTANT, CLS_ZERO, CLS_POOL, CLS_POOL, CLS_ZERO),
        output_word=2, sync_word=6, sync_bytes=96,
        input_label=48, reference_label=49),
    50: dict(
        slot=50,
        psize=56,
        name_live="cc_split_swin_16h_qkv_512_chained_fp8",
        name_sibling="cc_split_swin_16h_qkv_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        # Block29's qkv half — the qkv-proj pair's first half at grid (4,3,4),
        # the slot-26/30/34/38/42/46 qkv word map. Measured from the s50
        # capture: word classes (POOL, POOL, WEIGHTS, CONSTANT, CONSTANT,
        # POOL, POOL), w0 input (33,586,176 — address-equal to slot 49's
        # w2), w1 output (one stride past its input, the regular output
        # spacing), w2 weights, w3 the (24, 28) geometry constant
        # 0x1C00000018 and w4 the pair's second constant at
        # 0x00000000FFFFFFFC — the low-dword form the stage-41 note predicted
        # (block23 absent / 24 double / 25 low / 26 high / 27 absent / 28
        # double / 29 low), w5 sync-in (61,440 — address-equal to slot 49's
        # w6, the chained handoff) and w6 sync-out (61,952 — the 512-byte
        # stride continuing). Its weight record block29.layer2.layer
        # (917,568 B, the qkv record size) follows the established layout
        # walk, confirmed by the probe's live DLL walk. The sync extent
        # measured 192 B (span 61,952 -> 62,144), the qkv 192/256
        # alternation continuing (26/34/42/50 at 192, 30/38/46 at 256).
        record_name="block29.layer2.layer", record_bytes=917568,
        constant_values={3: 0x1C00000018, 4: 0x00000000FFFFFFFC},
        word_classes=(CLS_POOL, CLS_POOL, CLS_WEIGHTS, CLS_CONSTANT,
                      CLS_CONSTANT, CLS_POOL, CLS_POOL),
        output_word=1, sync_word=6, sync_bytes=192,
        input_label=49, reference_label=50),
    51: dict(
        slot=51,
        psize=72,
        name_live="cc_split_swin_16h_proj_512_chained_fp8",
        name_sibling="cc_split_swin_16h_proj_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        # Block29's proj half — the qkv-proj pair's second half at grid (8,3,1),
        # the slot-31/35/39/47 carrying proj word map (slot 27/43's shape with
        # w8 a second constant where those reserve the word zero). Measured
        # from the s51
        # capture: word classes (POOL, POOL, POOL, WEIGHTS, CONSTANT, ZERO,
        # POOL, POOL, CONSTANT), w0 input (33,930,240 — address-equal to
        # slot 50's w1), w1 residual (33,586,176 — the buffer slot 49 wrote
        # its output to, the input slot 50 itself read), w2 output
        # (34,274,304 — one stride past its input, the regular output
        # spacing), w4 the (24, 28) geometry constant 0x1C00000018, w5
        # reserved zero (every 72-byte block reserves the word), w8 the
        # pair's second constant at 0x00000000FFFFFFFC — the
        # low-dword form the stage-42 note predicted, now measured in both
        # halves of block29's pair (the absent/double/low/high cycle:
        # block23 absent / 24 double / 25 low / 26 high / 27 absent / 28
        # double / 29 low), w6 sync-out (62,464 — the 512-byte stride
        # continuing) and w7 sync-in (61,952 — address-equal to slot 50's
        # w6, the chained handoff). Its weight record block29.layer3.layer
        # (263,168 B at heap offset 20,250,624 — the record after the qkv
        # record's 448-byte tail padding) follows the established layout
        # walk, confirmed by the probe's live DLL walk. The sync extent
        # measured 96 B (span 62,464 -> 62,560), the proj norm.
        record_name="block29.layer3.layer", record_bytes=263168,
        constant_values={4: 0x1C00000018, 8: 0x00000000FFFFFFFC},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS,
                      CLS_CONSTANT, CLS_ZERO, CLS_POOL, CLS_POOL,
                      CLS_CONSTANT),
        output_word=2, sync_word=6, sync_bytes=96,
        input_label=50, reference_label=51),
    52: dict(
        slot=52,
        psize=56,
        name_live="cc_split_swin_16h_ffwd_512_chained_fp8",
        name_sibling="cc_split_swin_16h_ffwd_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        # Block30's ffwd half opens block30 and the next ffwd-proj pair.
        # Measured from the s52 capture's own parameter block: word classes
        # (POOL, POOL, WEIGHTS, CONSTANT, POOL, ZERO, POOL) — the
        # slot-28/32/36/40/44/48 map exactly at grid (4,3,2, the ffwd_512
        # family's single geometry): w0 input (34,274,304 — address-equal
        # to slot 51's w2), w1 output (34,618,368 — one stride past its
        # input, the regular output spacing), w2 weights, w3 the (24, 28)
        # geometry constant 0x1C00000018, w4 sync-in (62,464 —
        # address-equal to slot 51's w6, the chained handoff), w5 reserved
        # zero (every ffwd half reserves the word; the ffwd-proj pair
        # carries no second constant) and w6 sync-out (62,976 — the
        # 512-byte stride continuing: 60,928 -> 61,440 -> 61,952 ->
        # 62,464 -> 62,976). Its weights word lands on the layout walk's
        # block30.layer0.layer record start (524,288 B at heap offset
        # 20,534,784) — past the intervening block3.layer0.layer record
        # (20,672 B at 20,513,792, 320 bytes of tail padding to the
        # boundary), so block29.layer3.layer's naive +263,168 successor is
        # that small record, not block30's first; confirmed by the probe's
        # live DLL walk. The sync extent measured 96 B (the ffwd norm)
        # pinned from the s51->s52 write-set (two segments per frame, zero
        # unattributed: 340,808/341,273/341,295 tensor + 96 sync) and
        # enforced as the fp8 sibling's divergence.
        record_name="block30.layer0.layer", record_bytes=524288,
        constant_values={3: 0x1C00000018},
        word_classes=(CLS_POOL, CLS_POOL, CLS_WEIGHTS, CLS_CONSTANT,
                      CLS_POOL, CLS_ZERO, CLS_POOL),
        output_word=1, sync_word=6, sync_bytes=96,
        input_label=51, reference_label=52),
    53: dict(
        slot=53,
        psize=72,
        name_live="cc_split_swin_16h_ffwd_proj_512_chained_fp8",
        name_sibling="cc_split_swin_16h_ffwd_proj_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        # Block30's ffwd-proj pair's second half (the slot-25/29/33/37/41/45/49
        # word map, same 72-byte block and grid 8,3,1). Measured from the s53
        # capture: word classes (POOL, POOL, POOL, WEIGHTS, CONSTANT, ZERO,
        # POOL, POOL, ZERO), w0 input (34,618,368 — address-equal to slot
        # 52's w1), w1 residual (34,274,304 — the buffer slot 51 wrote its
        # output to, the input slot 52 itself read), w2 output (34,962,432 —
        # one stride past its input, the regular output spacing), w4 the
        # (24, 28) geometry constant 0x1C00000018, w5/w8 reserved zero (the
        # ffwd-proj pair carries no second constant), w6 sync-out (63,488 —
        # the 512-byte stride continuing: 61,952 -> 62,464 -> 62,976 ->
        # 63,488) and w7 sync-in (62,976 — address-equal to slot 52's w6,
        # the chained handoff). Its weight record block30.layer1.layer
        # (263,168 B at heap offset 21,059,072 — 20,534,784 + 524,288 at the
        # 512 boundary, block30's layer0 closing exactly at the boundary so
        # no padding intervenes) follows the established pattern, confirmed
        # by the probe's live DLL walk. The sync extent measured 96 B (span
        # 63,488 -> 63,584), the proj norm.
        record_name="block30.layer1.layer", record_bytes=263168,
        constant_values={4: 0x1C00000018},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS,
                      CLS_CONSTANT, CLS_ZERO, CLS_POOL, CLS_POOL, CLS_ZERO),
        output_word=2, sync_word=6, sync_bytes=96,
        input_label=52, reference_label=53),
    54: dict(
        slot=54,
        psize=56,
        name_live="cc_split_swin_16h_qkv_512_chained_fp8",
        name_sibling="cc_split_swin_16h_qkv_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        # Block30's qkv half — the qkv-proj pair's first half at grid (4,4,4),
        # the slot-30/38/46 geometry and the slot-26/30/34/38/42/46/50 qkv
        # word map. Measured from the s54 capture: word classes (POOL, POOL,
        # WEIGHTS, CONSTANT, CONSTANT, POOL, POOL), w0 input (34,962,432 —
        # address-equal to slot 53's w2), w1 output (one stride past its
        # input, the regular output spacing), w2 weights, w3 the (24, 28)
        # geometry constant 0x1C00000018 and w4 the pair's second constant at
        # 0xFFFFFFFC00000000 — the high-dword form the stage-45 note predicted
        # (block23 absent / 24 double / 25 low / 26 high / 27 absent / 28
        # double / 29 low / 30 high), w5 sync-in (63,488 — address-equal to
        # slot 53's w6, the chained handoff) and w6 sync-out (64,000 — the
        # 512-byte stride continuing: 62,464 -> 62,976 -> 63,488 -> 64,000).
        # Its weight record block30.layer2.layer (917,568 B, the qkv record
        # size) follows the established layout walk at heap offset 21,322,240
        # (21,059,072 + 263,168 at the 512 boundary), confirmed by the
        # probe's live DLL walk. The sync extent measured 256 B (span 64,000
        # -> 64,256), the qkv 192/256 alternation continuing (26/34/42/50 at
        # 192 B, 30/38/46/54 at 256 B).
        record_name="block30.layer2.layer", record_bytes=917568,
        constant_values={3: 0x1C00000018, 4: 0xFFFFFFFC00000000},
        word_classes=(CLS_POOL, CLS_POOL, CLS_WEIGHTS, CLS_CONSTANT,
                      CLS_CONSTANT, CLS_POOL, CLS_POOL),
        output_word=1, sync_word=6, sync_bytes=256,
        input_label=53, reference_label=54),
    55: dict(
        slot=55,
        psize=80,
        name_live="cc_split_swin_16h_proj_pool_512_chained_fp8",
        name_sibling="cc_split_swin_16h_proj_pool_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        # Block30's proj_pool half — the qkv-proj pair's second half in the
        # `proj_pool` variant (the first 80-byte block, 10 words, and the
        # first kernel with a second output tensor). Measured from the s55
        # capture: word classes (POOL, POOL, POOL, POOL, WEIGHTS, POOL,
        # POOL, CONSTANT, CONSTANT, CONSTANT), w0 input (35,306,496 —
        # address-equal to slot 54's w1), w1 residual (34,962,432 — the
        # buffer slot 53 wrote its output to), w2 output (52,313,088 —
        # 24x28x512 = 344,064 bytes), w3 pooled output (26,803,200 —
        # 12x16x512 = 98,304 bytes, the `proj_pool` name's pool expansion),
        # w4 weights on the layout walk's next record start
        # (`block30.layer3.layer`, 263,168 B at heap offset 22,240,256),
        # w5 sync-in (64,000 — address-equal to slot 54's w6, the chained
        # handoff) and w6 sync-out (64,512 — the 512-byte stride
        # continuing: 63,488 -> 64,000 -> 64,512), w7 the pair's second
        # constant at 0xFFFFFFFC00000000 (block30's high-dword step at the
        # proj half's own word position, now measured in both halves), w8
        # the (24, 28) constant 0x1C00000018 and w9 the (12, 16) pooled
        # geometry constant 0x000000100000000C. The sync extent measured
        # 96 B (span 64,512 -> 64,608), and the write-set is three segments
        # per frame (w2 344,064 B, w3 98,304 B, w6 96 B), zero unattributed.
        record_name="block30.layer3.layer", record_bytes=263168,
        constant_values={7: 0xFFFFFFFC00000000, 8: 0x1C00000018,
                         9: 0x000000100000000C},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS,
                      CLS_POOL, CLS_POOL, CLS_CONSTANT, CLS_CONSTANT,
                      CLS_CONSTANT),
        output_word=2, extra_outputs=((3, 98304),),
        sync_word=6, sync_bytes=96,
        input_label=54, reference_label=55),
    56: dict(
        slot=56,
        psize=40,
        name_live="cc_split_swin_16h_final_head_512_wait_fp8",
        name_sibling="cc_split_swin_16h_final_head_512_fp8",
        tag_live="wait", tag_sibling="fp8",
        # The transformer interval's final head — the first `wait`-family
        # kernel (no sync record written) and the first five-word contract
        # block (40 bytes). Measured from the s56 capture: word classes (POOL,
        # POOL, WEIGHTS, POOL, CONSTANT), w0 input (26,803,200 —
        # address-equal to slot 55's w3, the pooled tensor), w1 output
        # (52,657,152 — immediately past slot 55's w2 output:
        # 52,313,088 + 344,064), w2 weights, w3 the sync record it waits
        # on (64,512 — address-equal to slot 55's w6, the chained handoff;
        # read-only, the write-set attribution pins w1 alone and the
        # record's extent is byte-unchanged across the cut) and w4 the
        # (12, 16) pooled geometry constant 0x000000100000000C (slot 55's
        # w9 value recurring at the head). Its weight record
        # block30.layer4.layer (524,304 B, block30's fifth record)
        # follows the established layout walk at heap offset 22,503,424
        # (22,240,256 + 263,168 at the 512 boundary), confirmed by the
        # probe's live DLL walk. The output tensor measures 196,608 bytes
        # (384 512-wide rows — twice the pooled input's 192, closing
        # exactly at slot 57's output binding at 52,853,760), and the
        # write-set is one segment per frame (w1 196,608 B), zero
        # unattributed.
        record_name="block30.layer4.layer", record_bytes=524304,
        constant_values={4: 0x000000100000000C},
        word_classes=(CLS_POOL, CLS_POOL, CLS_WEIGHTS, CLS_POOL,
                      CLS_CONSTANT),
        output_word=1, output_bytes=196608,
        sync_word=3, sync_bytes=0,
        input_label=55, reference_label=56),
    57: dict(
        slot=57,
        psize=24,
        name_live="cc_vit_1d_repack_2d_to_1d_fp8",
        name_sibling=None,
        tag_live="fp8", tag_sibling=None,
        # The ViT interval's opening repack — the first single-variant
        # contract (the vendor creates and launches the only variant; the
        # repack family has one kernel per direction, the `1d_to_2d`
        # counterpart living at slot 98) and the first weights-less one
        # (psize 24 — three words, the smallest block replayed; no weights
        # word exists). Measured from the s57 capture: word classes (POOL,
        # POOL, CONSTANT), w0 input (52,657,152 — address-equal to slot
        # 56's w1, the final-head tensor read back as the ViT interval's
        # input), w1 output (52,853,760 — the head's output extent closes
        # exactly here: 52,657,152 + 196,608) and w2 the (12, 16) constant
        # 0x000000100000000C (slot 55's w9 / slot 56's w4 recurring). The
        # output tensor measures 196,608 bytes (the head's own 384x512
        # geometry) closing exactly at the ViT interval's 53,050,368
        # boundary (slot 97's w2 / slot 98's w0). The write-set is one
        # segment per frame (names-s56-s57.txt: w1 alone, zero
        # unattributed) and there is no sync word: the repack family is
        # outside the sync chain. The kernel is a pure byte permutation of
        # the head tensor — with all-zero pre-states in both write-set
        # spans the changed counts match the s55 -> s56 head write-set
        # frame for frame (171,969/171,970/171,962) and the sorted byte
        # multisets are equal per frame (measured over the retained capture
        # pools: vr/logs/ladder/permutation-s56-s57.json) — so the controls
        # re-base onto the input tensor (the
        # wrong control rotates it inwards, the read-bounding tail pattern
        # goes into the output extent's pre-launch bytes). The ViT
        # interval's kernels are created from their own cubin and extracted
        # from DLL fatbin container 5 (container 4 carries the transformer
        # interval's).
        container=5,
        record_name=None, record_bytes=0,
        constant_values={2: 0x000000100000000C},
        word_classes=(CLS_POOL, CLS_POOL, CLS_CONSTANT),
        output_word=1, output_bytes=196608,
        input_word=0, input_bytes=196608,
        sync_word=None, sync_bytes=0,
        input_label=56, reference_label=57),
    58: dict(
        slot=58,
        psize=72,
        name_live="cc_vit_1d_ffn_expand_publish_fp8",
        name_sibling="cc_vit_1d_ffn_expand_fp8",
        tag_live="publish", tag_sibling="fp8",
        # The ViT interval's first weighted kernel — the opening ffn_expand
        # of the ViT block pattern (ffn_expand -> ffn_contract -> qkv ->
        # attention -> projection, a five-slot cycle) and the first contract
        # to bind one of the ViT block records. Measured from the s58
        # capture: word classes (POOL, ZERO, POOL, WEIGHTS, ZERO, ZERO,
        # ZERO, POOL, CONSTANT), w0 input (52,853,760 — address-equal to
        # slot 57's w1, the repacked head tensor), w2 output
        # (34,372,608 — 786,432 bytes, the 384x2048 fp8 expansion of the
        # 384x512 input; the extent closes exactly at slot 63's w2 binding
        # at 35,159,040, the next ffn_expand output's base), w3 weights on
        # the layout walk's next record start (`block31.layer0.layer`,
        # 4,194,320 B at heap offset 23,028,224 — 22,503,424 + 524,304
        # rounded up at the 512 boundary, block31's first record and the
        # first ViT record), w7 sync-out (65,024 — the 512-byte stride
        # continuing: 64,512 -> 65,024) and w8 the (12, 16) constant
        # 0x000000100000000C (slot 55's w9 / slot 56's w4 / slot 57's w2
        # recurring); w1, w4, w5 and w6 are reserved zero — no sync-in
        # word is bound (the interval's opening publish writes the chain's
        # first record without consuming one). The sync extent measured
        # 256 B (span 65,024 -> 65,280) and the write-set is two segments
        # per frame (names-s57-s58.txt: w2 with 779,794/779,827/779,876
        # changed bytes inside the 786,432-byte binding, w7 256 B changed,
        # zero unattributed). The `publish` variant is live at the
        # interval's opening slot — the later ffn_expand launches use the
        # `chained` variant at their own slots — and the fp8 variant is
        # created but never launched (pinned sibling).
        container=5,
        record_name="block31.layer0.layer", record_bytes=4194320,
        constant_values={8: 0x000000100000000C},
        word_classes=(CLS_POOL, CLS_ZERO, CLS_POOL, CLS_WEIGHTS, CLS_ZERO,
                      CLS_ZERO, CLS_ZERO, CLS_POOL, CLS_CONSTANT),
        output_word=2, output_bytes=786432,
        sync_word=7, sync_bytes=256,
        input_label=57, reference_label=58),
    59: dict(
        slot=59,
        psize=72,
        name_live="cc_vit_1d_ffn_contract_chained_fp8",
        name_sibling="cc_vit_1d_ffn_contract_fp8",
        tag_live="chained", tag_sibling="fp8",
        # The ViT block's second slot — the `ffn_contract` half of the
        # five-slot cycle (`ffn_expand` -> `ffn_contract` -> `qkv` ->
        # `attention` -> `projection`) and the first contract with three
        # written output regions. Measured from the s59 capture: word
        # classes (POOL, POOL, POOL, WEIGHTS, POOL, POOL, POOL, POOL,
        # CONSTANT) — every word bound, no reserved zero word (slot 58's
        # nine-word block reserves four of its nine; slots 55, 56 and 57
        # also bind every word at their own shapes). w0
        # input (34,372,608 — address-equal to slot 58's w2, the fp8
        # expansion read back as the contraction's input), w1 residual
        # (52,853,760 — slot 57's repacked head tensor bound again, the
        # skip connection), w2 output (53,246,976 — 196,608 bytes, the
        # 384x512 fp8 contraction closing exactly at slot 60's w1 binding
        # at 53,443,584), w3 weights (`block31.layer1.layer`, 4,196,352 B
        # at heap offset 27,223,040 — the layout walk's next record start
        # after `block31.layer0.layer`), w5 second output (49,167,360 —
        # 393,216 bytes, closing exactly at slot 62's w5 binding at
        # 49,560,576; the `projection` half of the cycle shares this whole
        # word map), w4 the third output (65,536 — 64 bytes, the record
        # walk's 512-byte stride continuing: 65,024 -> 65,536), w6 sync in
        # (65,024 — slot 58's record) and w7 the sync record (66,048 — 64
        # bytes, the next stride) plus w8 the (12, 16) constant
        # 0x000000100000000C recurring. The write-set is four segments per
        # frame (names-s58-s59.txt: w5 with 342,690/342,703/342,652 changed
        # bytes inside the 393,216-byte binding, w2 with
        # 171,959/171,966/171,970 inside the 196,608-byte binding — both
        # changed boxes closing 4 bytes short of their bindings, the final
        # word static in every frame — w4 and w7 64 B changed each, zero
        # unattributed). The `chained` variant is live (the fp8 variant is
        # created but never launched — pinned sibling). The variant finding
        # re-classifies the two record-walk publications: w7 is the chain
        # handoff (the plain variant leaves it at the input state — the
        # predicted 64 B sibling divergence), while w4 is variant-independent
        # output data (both variants write it byte-identically — the first
        # contract where the sibling shares an output beyond the main
        # tensors; a record-walk-only prediction of 128 B of divergence is
        # wrong here: the retained sibling run pins 64 B). Slot 60's parameter block
        # binds w7's record as its sync-in; no launch in the 156-launch
        # sequence binds w4's record (scanned from the same retained traces
        # the replay reads).
        container=5,
        record_name="block31.layer1.layer", record_bytes=4196352,
        constant_values={8: 0x000000100000000C},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS, CLS_POOL,
                      CLS_POOL, CLS_POOL, CLS_POOL, CLS_CONSTANT),
        output_word=2, output_bytes=196608,
        extra_outputs=((5, 393216), (4, 64)),
        sync_word=7, sync_bytes=64,
        input_label=58, reference_label=59),
    60: dict(
        slot=60,
        psize=80,
        name_live="cc_vit_1d_qkv_chained_fp8",
        name_sibling="cc_vit_1d_qkv_fp8",
        tag_live="chained", tag_sibling="fp8",
        # The ViT block's third slot — the `qkv` half of the five-slot
        # cycle (`ffn_expand` -> `ffn_contract` -> `qkv` -> `attention` ->
        # `projection`) and the first contract with five written output
        # regions and a six-segment write-set. Measured from the s60
        # capture: word classes (POOL, POOL, POOL, POOL, WEIGHTS, POOL,
        # POOL, POOL, POOL, CONSTANT). w0 input (53,246,976 — address-equal
        # to slot 59's w2, the contraction output read back as the qkv
        # input), w1/w2/w3 the q/k/v triplet (196,608 bytes each, 384x512 —
        # slot 61's attention binds them as its w0/w1/w2, and the triplet
        # tiles [53,443,584, 54,033,408), closing exactly at the attention
        # output's base), w4 weights (`block31.layer2.layer`, 3,145,856 B
        # at heap offset 31,419,392 — the layout walk's next record start
        # after `block31.layer1.layer`, closing exactly at its end), w6 a
        # fourth written region (33,192,960 — 1,179,648 bytes = 384x3,072,
        # closing exactly at slot 58's w2 binding at 34,372,608; one per
        # ViT block — block32's qkv binds its w6 at pool 20,610,048), w5
        # the first record-walk publication (66,560 — 128 bytes, the
        # 512-byte stride continuing: 66,048 -> 66,560), w7 sync in
        # (66,048 — slot 59's record), w8 the sync record (67,072 — 128
        # bytes, the next stride; slot 61 binds its record at the
        # attention block's w5 sync-in), w9 the (12, 16) constant
        # 0x000000100000000C recurring. The write-set is six segments per
        # frame (names-s59-s60.txt: w6 with 1,172,971/1,172,523/1,172,573
        # changed bytes inside the 1,179,648-byte binding — the changed
        # box covering the whole binding, w3 with
        # 171,974/171,993/171,981 inside the 196,608-byte binding closing
        # 2 bytes short, w1 with 171,959/171,944/171,950 closing 4 bytes
        # short, w2 with 171,661/171,667/171,669 closing 8 bytes short —
        # the final words static in every frame — and w5/w8 128 B changed
        # each, zero unattributed). The `chained` variant is live (the fp8
        # variant is created but never launched — pinned sibling). The two
        # record-walk publications follow slot 59's re-classification: w8
        # is the chain handoff (its record is bound by slot 61 as the
        # attention block's sync-in — the predicted 128 B sibling
        # divergence), while w5 is bound by no other launch in the
        # 156-launch sequence (scanned from the same retained traces the
        # replay reads) and is variant-independent output data.
        container=5,
        record_name="block31.layer2.layer", record_bytes=3145856,
        constant_values={9: 0x000000100000000C},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS,
                      CLS_POOL, CLS_POOL, CLS_POOL, CLS_POOL, CLS_CONSTANT),
        output_word=1, output_bytes=196608,
        extra_outputs=((2, 196608), (3, 196608), (6, 1179648), (5, 128)),
        sync_word=8, sync_bytes=128,
        input_label=59, reference_label=60),
    61: dict(
        slot=61,
        psize=64,
        name_live="cc_vit_1d_attention_chained_fp8",
        name_sibling="cc_vit_1d_attention_fp8",
        tag_live="chained", tag_sibling="fp8",
        # The ViT block's fourth slot — the `attention` half of the five-slot
        # cycle (`ffn_expand` -> `ffn_contract` -> `qkv` -> `attention` ->
        # `projection`) and the first eight-word block (psize 64 — slot 55
        # binds ten words, slot 56 five, slot 57 three, slots 58 and 59
        # nine, slot 60 ten). Measured from the s61 capture: word classes (POOL, POOL,
        # POOL, POOL, WEIGHTS, POOL, POOL, CONSTANT). w0/w1/w2 the q/k/v
        # triplet (53,443,584 / 53,836,800 / 53,640,192 — address-equal to
        # slot 60's w1/w2/w3, the three 196,608-byte tensors consumed as the
        # attention's query, key and value), w3 output (54,033,408 — 196,608
        # bytes, the attention output closing exactly at slot 62's w2 binding
        # at 54,230,016), w4 weights (`block31.layer3.layer`, a 2-byte
        # payload at heap offset 34,565,632 — the layout walk's next record
        # start after `block31.layer2.layer`, whose end at 34,565,248 is
        # rounded up at the 512 boundary — and by far the smallest record in
        # the walk: the ViT attention blocks' layer3 records all measure a
        # 2-byte payload (block32's `f3 b3`, block33's `49 03`), unlike the
        # transformer interval's 263,168-byte layer3 records) — a dead
        # binding: the kernel's output is invariant under whole-record
        # complementation as well as under the wrong control's rotation,
        # measured in both variants (the retained wrong run and the
        # complementation experiment reproduce the execute pools
        # byte-for-byte), so the weight-free attention family reads nothing
        # from its placeholder record and the wrong control re-bases onto the
        # input tensor the kernel functionally reads (the query at w0) —
        # the probe's first control re-targeting, pinned by `wrong_word`),
        # w5 sync in (67,072 — slot 60's record, the chained handoff
        # continuing), w6 the sync record (67,584 — 256
        # bytes, the 512-byte stride continuing: 67,072 -> 67,584; slot 62
        # binds this record at the projection block's w6 sync-in), w7 the
        # (12, 16) constant 0x000000100000000C recurring. The write-set is
        # two segments per frame (names-s60-s61.txt: w3 with
        # 196,535/196,524/196,496 changed bytes inside the 196,608-byte
        # binding — the changed box covering the whole binding, 73/84/112
        # interior bytes static in the three frames — and w6 256/256/256,
        # the whole publication, zero unattributed). The `chained` variant
        # is live (the fp8 variant is created but never launched — pinned
        # sibling). Slot 61 is the first contract with a single record-walk
        # publication, and the family rule classifies it by consumption
        # alone: slot 62 binds w6's record as its sync-in, so w6 is the
        # chain handoff and the predicted sibling divergence is the whole
        # 256-byte publication (the plain variant leaves it at the input
        # state) — there is no sibling-shared publication here.
        container=5,
        record_name="block31.layer3.layer", record_bytes=2,
        constant_values={7: 0x000000100000000C},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS,
                      CLS_POOL, CLS_POOL, CLS_CONSTANT),
        output_word=3, output_bytes=196608,
        wrong_word=0, wrong_bytes=196608,
        sync_word=6, sync_bytes=256,
        input_label=60, reference_label=61),
    62: dict(
        slot=62,
        psize=72,
        name_live="cc_vit_1d_projection_chained_fp8",
        name_sibling="cc_vit_1d_projection_fp8",
        tag_live="chained", tag_sibling="fp8",
        # The ViT block's fifth slot — the `projection` half closing
        # block31's five-slot cycle (`ffn_expand` -> `ffn_contract` -> `qkv`
        # -> `attention` -> `projection`). It shares slot 59's nine-word
        # contraction shape (psize 72, every word bound, three written
        # output regions). Measured from the s62 capture: word classes (POOL,
        # POOL, POOL, WEIGHTS, POOL, POOL, POOL, POOL, CONSTANT). w0 input
        # (54,033,408 — address-equal to slot 61's w3, the attention output
        # read back as the projection's input), w1 residual (53,246,976 —
        # address-equal to slot 59's w2, the contraction output from three
        # slots earlier bound again as the attention sub-block's skip
        # connection), w2 output (54,230,016 — 196,608 bytes, 384x512,
        # closing at 54,426,624; slot 63's `ffn_expand` binds it as w0 and
        # slot 64's `ffn_contract` as its w1 residual, opening block32's
        # cycle), w3 weights (`block31.layer4.layer`, 1,050,624 B at heap
        # offset 34,566,144 — the layout walk's next record start after
        # `block31.layer3.layer`, whose 2-byte payload at 34,565,632 rounds
        # up across 510 bytes of padding to the 512 boundary, closing
        # block31's five-record set exactly at `block32.layer0.layer`'s
        # start at 35,616,768), w5 second output (49,560,576 — 393,216 bytes,
        # immediately past slot 59's w5 [49,167,360, 49,560,576) and closing
        # at slot 64's w5 binding at 49,953,792), w4 third output (68,096 —
        # 64 bytes, the record walk's 512-byte stride continuing: 67,584 ->
        # 68,096), w6 sync in (67,584 — slot 61's record), w7 the sync
        # record (68,608 — 64 bytes, the next stride; slot 63 binds its
        # record at `ffn_expand`'s w6 sync-in) and w8 the (12, 16) constant
        # 0x000000100000000C recurring. The write-set is four segments per
        # frame (names-s61-s62.txt: w5 with 391,486/391,452/391,349 changed
        # bytes inside the 393,216-byte binding and w2 with
        # 196,500/196,524/196,534 inside the 196,608-byte binding — both
        # changed boxes covering their whole bindings, unlike slot 59's
        # 4-byte-short boxes — and w4/w7 64 B changed each, zero
        # unattributed). Three variants exist in the cubin (`chained`,
        # `wait`, `fp8`): `chained` is live at slots 62/67/72/77/82/87/92
        # (`wait` runs at slot 97, the ViT interval's closing projection),
        # while `fp8` is created but never launched (pinned sibling). The
        # two record-walk publications follow the slot-59/60 family rule: w7
        # is the chain handoff (slot 63 binds its record as sync-in — the
        # predicted 64 B sibling divergence), while w4 is bound by no other
        # launch in the 156-launch sequence (scanned from the same retained
        # traces the replay reads) and is variant-independent output data.
        container=5,
        record_name="block31.layer4.layer", record_bytes=1050624,
        constant_values={8: 0x000000100000000C},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS, CLS_POOL,
                      CLS_POOL, CLS_POOL, CLS_POOL, CLS_CONSTANT),
        output_word=2, output_bytes=196608,
        extra_outputs=((5, 393216), (4, 64)),
        sync_word=7, sync_bytes=64,
        input_label=61, reference_label=62),
    63: dict(
        slot=63,
        psize=72,
        name_live="cc_vit_1d_ffn_expand_chained_fp8",
        name_sibling="cc_vit_1d_ffn_expand_fp8",
        tag_live="chained", tag_sibling="fp8",
        # The second ViT block's opening slot — block32's five-slot cycle
        # starting at the `ffn_expand` family's `chained` variant's first
        # launch (slot 58's `publish` opened block31's cycle, writing the
        # chain's first record without consuming one). It shares slot 58's
        # nine-word expansion shape (psize 72) with w6 bound as the sync-in
        # that `publish` leaves zero. Measured from the s63 capture: word
        # classes (POOL, ZERO, POOL, WEIGHTS, ZERO, ZERO, POOL, POOL,
        # CONSTANT). w0 input (54,230,016 — address-equal to slot 62's w2,
        # the projection output read back as the next block's input), w2
        # output (35,159,040 — 786,432 bytes, the 384x2048 fp8 expansion
        # of the 384x512 input, closing at 35,945,472 where slot 68's
        # `chained` expansion output chains one block-stride later and slot
        # 64's `ffn_contract` binds the region as w0), w3 weights
        # (`block32.layer0.layer`, 4,194,320 B at heap offset 35,616,768 —
        # block31's five-record set ends there exactly, no padding
        # intervenes, block32's first record; block32 repeats block31's
        # five-record shape), w6 sync in (68,608 — slot 62's record), w7
        # the sync record (69,120 — 256 bytes, the record walk's 512-byte
        # stride continuing: 68,608 -> 69,120; slot 64 binds its record at
        # `ffn_contract`'s w6 sync-in) and w8 the (12, 16) constant
        # 0x000000100000000C recurring; w1, w4 and w5 are reserved zero.
        # The write-set is two segments per frame (names-s62-s63.txt: w2
        # with 778,353/777,687/777,881 changed bytes inside the
        # 786,432-byte binding, the span covering the whole binding, and
        # w7 256/256/256, the whole publication, zero unattributed). Three
        # variants exist in the cubin (`chained`, `publish`, `fp8`):
        # `chained` is live at slots 63/68/73/78/83/88/93 (`publish` ran
        # once at slot 58), while `fp8` is created but never launched
        # (pinned sibling). The single record-walk publication is
        # classified by consumption alone: w7 is the chain handoff (slot 64
        # binds its record as sync-in — the predicted 256 B sibling
        # divergence).
        container=5,
        record_name="block32.layer0.layer", record_bytes=4194320,
        constant_values={8: 0x000000100000000C},
        word_classes=(CLS_POOL, CLS_ZERO, CLS_POOL, CLS_WEIGHTS, CLS_ZERO,
                      CLS_ZERO, CLS_POOL, CLS_POOL, CLS_CONSTANT),
        output_word=2, output_bytes=786432,
        sync_word=7, sync_bytes=256,
        input_label=62, reference_label=63),
    64: dict(
        slot=64,
        psize=72,
        name_live="cc_vit_1d_ffn_contract_chained_fp8",
        name_sibling="cc_vit_1d_ffn_contract_fp8",
        tag_live="chained", tag_sibling="fp8",
        # The second ViT block's second slot — block32's `ffn_contract`
        # half of the five-slot cycle (`ffn_expand` -> `ffn_contract` ->
        # `qkv` -> `attention` -> `projection`) — sharing slot 59's
        # nine-word contraction shape exactly (psize 72, the same (12, 16)
        # constant at w8). Measured from the s64 capture: word classes
        # (POOL, POOL, POOL, WEIGHTS, POOL, POOL, POOL, POOL, CONSTANT) —
        # every word bound, no reserved zero word. w0 input (35,159,040 —
        # address-equal to slot 63's w2, the fp8 expansion read back as the
        # contraction's input), w1 residual (54,230,016 — slot 62's w2,
        # the projection output from two slots earlier bound again as the
        # skip connection; slot 59 bound slot 57's head tensor the same
        # way), w2 output (54,426,624 — 196,608 bytes, the 384x512 fp8
        # contraction closing exactly at slot 65's w1 binding at
        # 54,623,232; slot 67's projection re-reads the region as its w1
        # residual), w3 weights (`block32.layer1.layer`, 4,196,352 B at
        # heap offset 39,811,584 — the layout walk's next record start
        # after `block32.layer0.layer`, its 4,194,320-byte payload at
        # 35,616,768 plus 496 bytes of padding to the 512 boundary), w5
        # second output (49,953,792 — 393,216 bytes, closing exactly at
        # 50,347,008 where slot 69's w5 chains one block-stride later), w4
        # the third output (69,632 — 64 bytes, the record walk's stride
        # continuing: 69,120 -> 69,632), w6 sync in (69,120 — slot 63's
        # record) and w7 the sync record (70,144 — 64 bytes, the next
        # stride; slot 65 binds the record at the `qkv` block's w7
        # sync-in) plus w8 the (12, 16) constant 0x000000100000000C
        # recurring. The write-set is four segments per frame
        # (names-s63-s64.txt: w5 with 391,734/391,648/391,716 changed
        # bytes inside the 393,216-byte binding and w2 with
        # 196,523/196,553/196,503 inside the 196,608-byte binding — both
        # changed boxes covering their whole bindings exactly with static
        # interior bytes (1,482/1,568/1,500 and 85/55/105), w4 and w7 64 B
        # changed each, zero unattributed). The `chained` variant is live
        # (the fp8 variant is created but never launched — pinned
        # sibling). The record-walk re-classification holds a fourth time:
        # w7 is the chain handoff (the plain variant leaves it at the
        # input state — the predicted 64 B sibling divergence), while w4
        # is variant-independent output data (both variants write it
        # byte-identically — the fourth sibling-shared publication beyond
        # the main tensors; no launch in the 156-launch sequence binds
        # w4's record, scanned from the same retained traces the replay
        # reads).
        container=5,
        record_name="block32.layer1.layer", record_bytes=4196352,
        constant_values={8: 0x000000100000000C},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS, CLS_POOL,
                      CLS_POOL, CLS_POOL, CLS_POOL, CLS_CONSTANT),
        output_word=2, output_bytes=196608,
        extra_outputs=((5, 393216), (4, 64)),
        sync_word=7, sync_bytes=64,
        input_label=63, reference_label=64),
    65: dict(
        slot=65,
        psize=80,
        name_live="cc_vit_1d_qkv_chained_fp8",
        name_sibling="cc_vit_1d_qkv_fp8",
        tag_live="chained", tag_sibling="fp8",
        # The second ViT block's third slot — block32's `qkv` half of the
        # five-slot cycle (`ffn_expand` -> `ffn_contract` -> `qkv` ->
        # `attention` -> `projection`), sharing slot 60's ten-word qkv
        # shape exactly (psize 80, the same (12, 16) constant at w9) and
        # the first contract with five written output regions and a
        # six-segment write-set since slot 60. Measured from the s65
        # capture: word classes (POOL, POOL, POOL, POOL, WEIGHTS, POOL,
        # POOL, POOL, POOL, CONSTANT). w0 input (54,426,624 — address-equal
        # to slot 64's w2, the contraction output read back as the qkv
        # input), w1/w2/w3 the q/k/v triplet (196,608 bytes each, 384x512
        # — slot 66's attention binds them as its w0/w1/w2, and the
        # triplet tiles [54,623,232, 55,213,056), closing exactly at the
        # attention output's base at 55,213,056), w4 weights
        # (`block32.layer2.layer`, 3,145,856 B at heap offset 44,007,936
        # — the layout walk's next record start after
        # `block32.layer1.layer`, closing exactly at its end), w6 a fourth
        # written region (20,610,048 — 1,179,648 bytes = 384x3,072, the
        # block32 instance of the one-per-block region slot 60's comment
        # predicts, closing exactly at 21,789,696 where block33's `qkv`
        # w6 chains immediately), w5 the first record-walk publication
        # (70,656 — 128 bytes, the 512-byte stride continuing: 69,632 ->
        # 70,144 -> 70,656), w7 sync in (70,144 — slot 64's record), w8
        # the sync record (71,168 — 128 bytes, the next stride; slot 66
        # binds its record at the attention block's w5 sync-in), w9 the
        # (12, 16) constant 0x000000100000000C recurring. The write-set is
        # six segments per frame (names-s64-s65.txt: w6 with
        # 1,173,080/1,172,617/1,172,341 changed bytes inside the
        # 1,179,648-byte binding — the changed box covering the whole
        # binding, w3 with 196,552/196,517/196,546, w1 with
        # 196,549/196,526/196,538 and w2 with 196,218/196,171/196,181
        # inside the 196,608-byte bindings — all three covering their
        # whole bindings with static interior bytes (56/91/62, 59/82/70 and
        # 390/437/427) — and w5/w8 128 B changed each, zero unattributed).
        # The `chained` variant is live (the fp8 variant is created but
        # never launched — pinned sibling). The two record-walk
        # publications follow the established re-classification: w8 is the
        # chain handoff (its record is bound by slot 66 as the attention
        # block's w5 sync-in — the predicted 128 B sibling divergence),
        # while w5 is bound by no other launch in the 156-launch sequence
        # (scanned from the same retained traces the replay reads) and is
        # variant-independent output data.
        container=5,
        record_name="block32.layer2.layer", record_bytes=3145856,
        constant_values={9: 0x000000100000000C},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS,
                      CLS_POOL, CLS_POOL, CLS_POOL, CLS_POOL, CLS_CONSTANT),
        output_word=1, output_bytes=196608,
        extra_outputs=((2, 196608), (3, 196608), (6, 1179648), (5, 128)),
        sync_word=8, sync_bytes=128,
        input_label=64, reference_label=65),
    66: dict(
        slot=66,
        psize=64,
        name_live="cc_vit_1d_attention_chained_fp8",
        name_sibling="cc_vit_1d_attention_fp8",
        tag_live="chained", tag_sibling="fp8",
        # The second ViT block's fourth slot — block32's `attention` half
        # of the five-slot cycle and the attention family's second launch,
        # sharing slot 61's eight-word attention shape exactly (psize 64,
        # the (12, 16) constant at w7 — slot 55 binds ten words, slot 56
        # five, slot 57 three, slots 58 and 59 nine, slot 60 ten, slot 61
        # eight). Measured from the s66 capture: word classes (POOL, POOL,
        # POOL, POOL, WEIGHTS, POOL, POOL, CONSTANT). w0/w1/w2 the q/k/v
        # triplet (54,623,232 / 54,819,840 / 55,016,448 — address-equal to
        # slot 65's w1/w2/w3, the three 196,608-byte tensors consumed as
        # query, key and value), w3 output (55,213,056 — 196,608 bytes, the
        # attention output closing exactly at slot 67's w2 binding at
        # 55,409,664), w4 weights (`block32.layer3.layer`, a 2-byte payload
        # at heap offset 47,154,176 — the layout walk's next record start
        # after `block32.layer2.layer`, whose end at 47,153,792 rounds up
        # across 384 bytes at the 512 boundary; the ViT attention blocks'
        # layer3 records all measure a 2-byte payload — block32's `f3 b3`)
        # — a dead binding like slot 61's: the weight-free attention family
        # reads nothing from its placeholder record, so the wrong control
        # re-bases onto the input tensor the kernel functionally reads (the
        # query at w0), the family's second re-targeting pinned by
        # `wrong_word` (the rotation and complementation experiments at
        # slot 66 reproduce the execute pools byte-for-byte), w5 sync in
        # (71,168 — slot 65's record, the chained handoff continuing), w6
        # the sync record (71,680 — 256 bytes, the 512-byte stride
        # continuing: 71,168 -> 71,680; slot 67 binds this record at the
        # projection block's w6 sync-in), w7 the (12, 16) constant
        # 0x000000100000000C recurring. The write-set is two segments per
        # frame (names-s65-s66.txt: w3 with 196,547/196,544/196,547 changed
        # bytes inside the 196,608-byte binding — the changed box covering
        # the whole binding, 61/64/61 interior bytes static in the three
        # frames — and w6 256/256/256, the whole publication, zero
        # unattributed). The `chained` variant is live (the fp8 variant is
        # created but never launched — pinned sibling). As at slot 61 the
        # block has a single record-walk publication, classified by
        # consumption alone: slot 67 binds w6's record as its sync-in, so
        # w6 is the chain handoff and the sibling divergence is the whole
        # 256-byte publication (the plain variant leaves it at the input
        # state) — there is no sibling-shared publication at this contract.
        container=5,
        record_name="block32.layer3.layer", record_bytes=2,
        constant_values={7: 0x000000100000000C},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS,
                      CLS_POOL, CLS_POOL, CLS_CONSTANT),
        output_word=3, output_bytes=196608,
        wrong_word=0, wrong_bytes=196608,
        sync_word=6, sync_bytes=256,
        input_label=65, reference_label=66),

    67: dict(
        slot=67,
        psize=72,
        name_live="cc_vit_1d_projection_chained_fp8",
        name_sibling="cc_vit_1d_projection_fp8",
        tag_live="chained", tag_sibling="fp8",
        # The second ViT block's fifth slot — block32's `projection` half
        # closing the five-slot cycle. It shares slot 62's nine-word
        # projection shape exactly (psize 72, every word bound, three
        # written output regions; grid (16,1,4)). Measured from the s67
        # capture: word classes (POOL, POOL, POOL, WEIGHTS, POOL, POOL,
        # POOL, POOL, CONSTANT). w0 input (55,213,056 — address-equal to
        # slot 66's w3, the attention output read back as the projection's
        # input), w1 residual (54,426,624 — address-equal to slot 64's w2,
        # the contraction output from three slots earlier bound again as
        # the attention sub-block's skip connection), w2 output
        # (55,409,664 — 196,608 bytes, 384x512, closing at 55,606,272;
        # slot 68's `ffn_expand` binds it as w0), w3 weights
        # (`block32.layer4.layer`, 1,050,624 B at heap offset 47,154,688 —
        # the layout walk's next record start after `block32.layer3.layer`,
        # whose 2-byte payload at 47,154,176 rounds up across 510 bytes of
        # padding to the 512 boundary, closing block32's five-record set
        # exactly at `block33.layer0.layer`'s start at 48,205,312), w5
        # second output (48,774,144 — 393,216 bytes, closing at
        # 49,167,360 where slot 59's w5 binds the region's next extent), w4
        # third output (72,192 — 64 bytes, the record walk's 512-byte
        # stride continuing: 71,680 -> 72,192), w6 sync in (71,680 — slot
        # 66's record), w7 the sync record (72,704 — 64 bytes, the next
        # stride; slot 68 binds its record at `ffn_expand`'s w6 sync-in)
        # and w8 the (12, 16) constant 0x000000100000000C recurring. The
        # write-set is four segments per frame (names-s66-s67.txt: w5 with
        # 391,543/391,442/391,443 changed bytes inside the 393,216-byte
        # binding and w2 with 196,545/196,547/196,536 inside the
        # 196,608-byte binding — both changed boxes covering their whole
        # bindings — and w4/w7 64 B changed each, zero unattributed). Three variants exist in the
        # cubin (`chained`, `wait`, `fp8`): `chained` is live at slots
        # 62/67/72/77/82/87/92 (`wait` runs at slot 97, the ViT interval's
        # closing projection), while `fp8` is created but never launched
        # (pinned sibling). The two record-walk publications follow the
        # slot-59/60 family rule: w7 is the chain handoff (slot 68 binds
        # its record as sync-in — the predicted 64 B sibling divergence),
        # while w4 is bound by no other launch in the 156-launch sequence
        # (scanned from the same retained traces the replay reads) and is
        # variant-independent output data — the re-classification's sixth
        # confirmation.
        container=5,
        record_name="block32.layer4.layer", record_bytes=1050624,
        constant_values={8: 0x000000100000000C},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS, CLS_POOL,
                      CLS_POOL, CLS_POOL, CLS_POOL, CLS_CONSTANT),
        output_word=2, output_bytes=196608,
        extra_outputs=((5, 393216), (4, 64)),
        sync_word=7, sync_bytes=64,
        input_label=66, reference_label=67),
    68: dict(
        slot=68,
        psize=72,
        name_live="cc_vit_1d_ffn_expand_chained_fp8",
        name_sibling="cc_vit_1d_ffn_expand_fp8",
        tag_live="chained", tag_sibling="fp8",
        # block33's expansion — the third ViT block's opening slot, the
        # expansion family's `chained` variant's second launch (slots
        # 63/68/73/78/83/88/93; `publish` opened block31's cycle at slot
        # 58). Shares slot 63's nine-word expansion shape (psize 72)
        # exactly. Measured from the s68 capture: word classes (POOL, ZERO,
        # POOL, WEIGHTS, ZERO, ZERO, POOL, POOL, CONSTANT). w0 input
        # (55,409,664 — address-equal to slot 67's w2, the projection
        # output read back as the next block's input), w2 output
        # (35,945,472 — 786,432 bytes, closing at 36,731,904; slot 63's
        # output closes at 35,945,472 exactly, the block-stride chain, and
        # slot 69's `ffn_contract` binds the region as w0), w3 weights
        # (`block33.layer0.layer`, 4,194,320 B at heap offset 48,205,312 —
        # block32's five-record set ends there exactly and block33's first
        # record begins; 496 bytes of padding close the gap to slot 69's
        # record at 52,400,128), w6 sync in (72,704 — slot 67's record),
        # w7 the sync record (73,216 — 256 bytes, the record walk's
        # 512-byte stride continuing: 72,704 -> 73,216; slot 69 binds its
        # record at `ffn_contract`'s w6 sync-in) and w8 the (12, 16)
        # constant 0x000000100000000C recurring; w1, w4 and w5 are
        # reserved zero. The write-set is two segments per frame
        # (names-s67-s68.txt: w2 with 779,454/778,805/778,812 changed
        # bytes inside the 786,432-byte binding, the span covering the
        # whole binding, and w7 256/256/256, the whole publication, zero
        # unattributed). Three variants exist in the cubin (`chained`,
        # `publish`, `fp8`): `chained` is live at slots 63/68/73/78/83/88/93
        # (`publish` ran once at slot 58), while `fp8` is created but never
        # launched (pinned sibling). The single record-walk publication is
        # classified by consumption alone: w7 is the chain handoff (slot 69
        # binds its record as sync-in — the predicted 256 B sibling
        # divergence).
        container=5,
        record_name="block33.layer0.layer", record_bytes=4194320,
        constant_values={8: 0x000000100000000C},
        word_classes=(CLS_POOL, CLS_ZERO, CLS_POOL, CLS_WEIGHTS, CLS_ZERO,
                      CLS_ZERO, CLS_POOL, CLS_POOL, CLS_CONSTANT),
        output_word=2, output_bytes=786432,
        sync_word=7, sync_bytes=256,
        input_label=67, reference_label=68),
    69: dict(
        slot=69,
        psize=72,
        name_live="cc_vit_1d_ffn_contract_chained_fp8",
        name_sibling="cc_vit_1d_ffn_contract_fp8",
        tag_live="chained", tag_sibling="fp8",
        # The third ViT block's second slot — block33's `ffn_contract`
        # half of the five-slot cycle (`ffn_expand` -> `ffn_contract` ->
        # `qkv` -> `attention` -> `projection`) — sharing slot 59's
        # nine-word contraction shape exactly (psize 72, the same (12, 16)
        # constant at w8). Measured from the s69 capture (manifest and
        # sidecar agree byte-for-byte): word classes (POOL, POOL, POOL,
        # WEIGHTS, POOL, POOL, POOL, POOL, CONSTANT) — every word bound, no
        # reserved zero word. w0 input (35,945,472 — address-equal to slot
        # 68's w2, the fp8 expansion read back as the contraction's
        # input), w1 residual (55,409,664 — slot 67's w2, the projection
        # output from two slots earlier bound again as the skip connection;
        # slot 64 bound slot 62's w2 the same way), w2 output
        # (55,606,272 — 196,608 bytes, the 384x512 fp8 contraction closing
        # exactly at 55,802,880; slot 72's projection re-reads the region
        # as its w1 residual, slot 67's w1 binding the same extent for
        # block32), w3 weights (`block33.layer1.layer`, 4,196,352 B at
        # heap offset 52,400,128 — the layout walk's next record start
        # after `block33.layer0.layer`, its 4,194,320-byte payload at
        # 48,205,312 plus 496 bytes of padding to the 512 boundary), w5
        # second output (50,347,008 — 393,216 bytes, closing exactly at
        # 50,740,224 where slot 72's w5 chains one block-stride later), w4
        # the third output (73,728 — 64 bytes, the record walk's stride
        # continuing: 73,216 -> 73,728), w6 sync in (73,216 — slot 68's
        # record) and w7 the sync record (74,240 — 64 bytes, the next
        # stride; slot 70 binds the record at the `qkv` block's w7
        # sync-in) plus w8 the (12, 16) constant 0x000000100000000C
        # recurring. The write-set is four segments per frame
        # (names-s68-s69.txt: w5 with 391,762/391,784/391,712 changed
        # bytes inside the 393,216-byte binding and w2 with
        # 196,555/196,553/196,521 inside the 196,608-byte binding — both
        # changed boxes covering their whole bindings exactly with static
        # interior bytes (1,454/1,432/1,504 and 53/55/87), w4 and w7 64 B
        # changed each, zero unattributed). The `chained` variant is live
        # (the fp8 variant is created but never launched — pinned
        # sibling). The record-walk re-classification holds a seventh time:
        # w7 is the chain handoff (the plain variant leaves it at the
        # input state — the predicted 64 B sibling divergence), while w4
        # is variant-independent output data (both variants write it
        # byte-identically — the seventh sibling-shared publication beyond
        # the main tensors; no launch in the 156-launch sequence binds
        # w4's record, scanned from the same retained traces the replay
        # reads).
        container=5,
        record_name="block33.layer1.layer", record_bytes=4196352,
        constant_values={8: 0x000000100000000C},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS, CLS_POOL,
                      CLS_POOL, CLS_POOL, CLS_POOL, CLS_CONSTANT),
        output_word=2, output_bytes=196608,
        extra_outputs=((5, 393216), (4, 64)),
        sync_word=7, sync_bytes=64,
        input_label=68, reference_label=69),
    70: dict(
        slot=70,
        psize=80,
        name_live="cc_vit_1d_qkv_chained_fp8",
        name_sibling="cc_vit_1d_qkv_fp8",
        tag_live="chained", tag_sibling="fp8",
        # The third ViT block's third slot — block33's `qkv` half of the
        # five-slot cycle (`ffn_expand` -> `ffn_contract` -> `qkv` ->
        # `attention` -> `projection`), sharing slot 60's ten-word qkv
        # shape exactly (psize 80, the same (12, 16) constant at w9) and
        # the first contract with five written output regions and a
        # six-segment write-set since slot 65. Measured from the s70
        # capture: word classes (POOL, POOL, POOL, POOL, WEIGHTS, POOL,
        # POOL, POOL, POOL, CONSTANT). w0 input (55,606,272 — address-equal
        # to slot 69's w2, the contraction output read back as the qkv
        # input), w1/w2/w3 the q/k/v triplet (196,608 bytes each, 384x512
        # — slot 71's attention binds them as its w0/w1/w2, and the
        # triplet tiles [55,802,880, 56,392,704), closing exactly at the
        # attention output's base at 56,392,704), w4 weights
        # (`block33.layer2.layer`, 3,145,856 B at heap offset 56,596,480
        # — the layout walk's next record start after
        # `block33.layer1.layer`, closing exactly at its end with no
        # padding — the stage-61 prediction pinned), w6 a fourth
        # written region (21,789,696 — 1,179,648 bytes = 384x3,072, the
        # block33 instance of the one-per-block region slot 60's comment
        # predicts and the exact address slot 65's comment predicted the
        # block33 `qkv` w6 would chain immediately at), w5 the first
        # record-walk publication
        # (74,752 — 128 bytes, the 512-byte stride continuing: 74,240 ->
        # 74,752), w7 sync in (74,240 — slot 69's record), w8
        # the sync record (75,264 — 128 bytes, the next stride; slot 71
        # binds its record at the attention block's w5 sync-in), w9 the
        # (12, 16) constant 0x000000100000000C recurring. The write-set is
        # six segments per frame (names-s69-s70.txt: w6 with
        # 1,173,143/1,172,569/1,172,385 changed bytes inside the
        # 1,179,648-byte binding — the changed box covering the whole
        # binding, w1 with 196,558/196,550/196,546, w2 with
        # 196,254/196,178/196,178 and w3 with 196,549/196,547/196,549
        # inside the 196,608-byte bindings — all three covering their
        # whole bindings with static interior bytes (50/58/62, 354/430/430
        # and 59/61/59) — and w5/w8 128 B changed each, zero
        # unattributed).
        # The `chained` variant is live (the fp8 variant is created but
        # never launched — pinned sibling). The two record-walk
        # publications follow the established re-classification: w8 is the
        # chain handoff (its record is bound by slot 71 as the attention
        # block's w5 sync-in — the predicted 128 B sibling divergence),
        # while w5 is bound by no other launch in the 156-launch sequence
        # (scanned from the same retained traces the replay reads) and is
        # variant-independent output data.
        container=5,
        record_name="block33.layer2.layer", record_bytes=3145856,
        constant_values={9: 0x000000100000000C},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS,
                      CLS_POOL, CLS_POOL, CLS_POOL, CLS_POOL, CLS_CONSTANT),
        output_word=1, output_bytes=196608,
        extra_outputs=((2, 196608), (3, 196608), (6, 1179648), (5, 128)),
        sync_word=8, sync_bytes=128,
        input_label=69, reference_label=70),
    71: dict(
        slot=71,
        psize=64,
        name_live="cc_vit_1d_attention_chained_fp8",
        name_sibling="cc_vit_1d_attention_fp8",
        tag_live="chained", tag_sibling="fp8",
        # The third ViT block's fourth slot — block33's `attention` half
        # of the five-slot cycle and the attention family's third launch,
        # sharing slot 61's eight-word attention shape exactly (psize 64,
        # the (12, 16) constant at w7). Measured from the s71 capture
        # (manifest + sidecar, three records byte-identical): word classes
        # (POOL, POOL, POOL, POOL, WEIGHTS, POOL, POOL, CONSTANT).
        # w0/w1/w2 the q/k/v triplet (55,802,880 / 55,999,488 /
        # 56,196,096 — address-equal to slot 70's w1/w2/w3, the three
        # 196,608-byte tensors consumed as query, key and value), w3
        # output (56,392,704 — 196,608 bytes, the attention output closing
        # exactly at 56,589,312 where slot 72's projection w2 lands; the
        # base slot 70's contract predicted), w4 weights
        # (`block33.layer3.layer`, a 2-byte payload at heap offset
        # 59,742,720 — the layout walk's next record start after
        # `block33.layer2.layer`, whose end at 59,742,336 rounds up
        # across 384 bytes at the 512 boundary; the ViT attention blocks'
        # layer3 records all measure a 2-byte payload — block32's `f3 b3`)
        # — a dead binding like slot 61's and slot 66's: the
        # weight-free attention family reads nothing from its placeholder
        # record, so the wrong control re-bases onto the input tensor the
        # kernel functionally reads (the query at w0), the family's third
        # re-targeting pinned by `wrong_word` (the rotation and
        # complementation experiments reproduce the execute pools
        # byte-for-byte), w5 sync in (75,264 — slot 70's record, the
        # chained handoff continuing), w6 the sync record (75,776 — 256
        # bytes, the 512-byte stride continuing: 75,264 -> 75,776; slot
        # 72 binds this record at the projection block's w6 sync-in), w7
        # the (12, 16) constant 0x000000100000000C recurring. The
        # write-set is two segments per frame (names-s70-s71.txt: w3 with
        # 196,523/196,527/196,549 changed bytes inside the 196,608-byte
        # binding — the changed box covering the whole binding, 85/81/59
        # interior bytes static in the three frames — and w6 256/256/256,
        # the whole publication, zero unattributed; frame totals
        # 196,779/196,783/196,805). The `chained` variant is live (the fp8
        # variant is created but never launched — pinned sibling). As at
        # slots 61 and 66 the block has a single record-walk publication,
        # classified by consumption alone: slot 72 binds w6's record as
        # its sync-in, so w6 is the chain handoff and the sibling
        # divergence is the whole 256-byte publication (the plain variant
        # leaves it at the input state) — there is no sibling-shared
        # publication at this contract.
        container=5,
        record_name="block33.layer3.layer", record_bytes=2,
        constant_values={7: 0x000000100000000C},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS,
                      CLS_POOL, CLS_POOL, CLS_CONSTANT),
        output_word=3, output_bytes=196608,
        wrong_word=0, wrong_bytes=196608,
        sync_word=6, sync_bytes=256,
        input_label=70, reference_label=71),
    72: dict(
        slot=72,
        psize=72,
        name_live="cc_vit_1d_projection_chained_fp8",
        name_sibling="cc_vit_1d_projection_fp8",
        tag_live="chained", tag_sibling="fp8",
        # The third ViT block's fifth slot — block33's `projection` half
        # closing the five-slot cycle. It shares slot 62's nine-word
        # projection shape exactly (psize 72, every word bound, three
        # written output regions; grid (16,1,4)). Measured from the s72
        # capture: word classes (POOL, POOL, POOL, WEIGHTS, POOL, POOL,
        # POOL, POOL, CONSTANT). w0 input (56,392,704 — address-equal to
        # slot 71's w3, the attention output read back as the projection's
        # input), w1 residual (55,606,272 — address-equal to slot 69's w2,
        # the contraction output from three slots earlier bound again as
        # the attention sub-block's skip connection), w2 output
        # (56,589,312 — 196,608 bytes, 384x512, closing at 56,785,920;
        # slot 73's `ffn_expand` binds it as w0), w3 weights
        # (`block33.layer4.layer`, 1,050,624 B at heap offset 59,743,232 —
        # the layout walk's next record start after `block33.layer3.layer`,
        # whose 2-byte payload at 59,742,720 rounds up across 510 bytes of
        # padding to the 512 boundary, closing block33's five-record set
        # exactly at `block34.layer0.layer`'s start at 60,793,856), w5
        # second output (50,740,224 — 393,216 bytes, closing at
        # 51,133,440 where the region's next extent begins),
        # w4 third output (76,288 — 64 bytes, the record walk's 512-byte
        # stride continuing: 75,776 -> 76,288), w6 sync in (75,776 — slot
        # 71's record), w7 the sync record (76,800 — 64 bytes, the next
        # stride; slot 73 binds its record at `ffn_expand`'s w6 sync-in)
        # and w8 the (12, 16) constant 0x000000100000000C recurring. The
        # write-set is four segments per frame (names-s71-s72.txt: w5 with
        # 391,671/391,507/391,719 changed bytes inside the 393,216-byte
        # binding and w2 with 196,536/196,555/196,543 inside the
        # 196,608-byte binding — both changed boxes covering their whole
        # bindings — and w4/w7 64 B changed each, zero unattributed). Three
        # variants exist in the cubin (`chained`, `wait`, `fp8`): `chained`
        # is live at slots 62/67/72/77/82/87/92 (`wait` runs at slot 97,
        # the ViT interval's closing projection), while `fp8` is created
        # but never launched (pinned sibling). The two record-walk
        # publications follow the slot-59/60 family rule: w7 is the chain
        # handoff (slot 73 binds its record as sync-in — the predicted 64 B
        # sibling divergence), while w4 is bound by no other launch in the
        # 156-launch sequence (scanned from the same retained traces the
        # replay reads) and is variant-independent output data — the
        # re-classification's ninth confirmation.
        container=5,
        record_name="block33.layer4.layer", record_bytes=1050624,
        constant_values={8: 0x000000100000000C},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS, CLS_POOL,
                      CLS_POOL, CLS_POOL, CLS_POOL, CLS_CONSTANT),
        output_word=2, output_bytes=196608,
        extra_outputs=((5, 393216), (4, 64)),
        sync_word=7, sync_bytes=64,
        input_label=71, reference_label=72),
    73: dict(
        slot=73,
        psize=72,
        name_live="cc_vit_1d_ffn_expand_chained_fp8",
        name_sibling="cc_vit_1d_ffn_expand_fp8",
        tag_live="chained", tag_sibling="fp8",
        # block34's opening `ffn_expand` — the expansion family's third
        # `chained` launch (slots 63/68/73/78/83/88/93; `publish` opened
        # block31's cycle at slot 58). Shares slot 63's nine-word expansion
        # shape (psize 72, grid (64,1,1), block (32,4,1)). Measured from the
        # s73 capture: word classes (POOL, ZERO, POOL, WEIGHTS, ZERO, ZERO,
        # POOL, POOL, CONSTANT). w0 input (56,589,312 — address-equal to
        # slot 72's w2, the projection output read back as the next block's
        # input), w2 output (36,731,904 — 786,432 bytes, the 384x2048 fp8
        # expansion, closing at 37,518,336 where slot 78's expansion binds
        # the region's next extent), w3 weights (`block34.layer0.layer`,
        # 4,194,320 B at heap offset 60,793,856 — the first record of
        # block34, exactly where block33's five-record set closes; the
        # record's own 4,194,320-byte payload is followed by 496 bytes of
        # padding before `block34.layer1.layer`), w6 sync-in (76,800 — slot
        # 72's w7 record), w7 the single 256-byte publication (77,312 — the
        # 512-byte stride continuing: 76,800 -> 77,312; slot 74 binds its
        # record as w6 sync-in) and w8 the (12, 16) constant
        # 0x000000100000000C; w1, w4 and w5 are reserved zero. The write-set
        # is two segments per frame (names-s72-s73.txt: w2 with
        # 779,475/778,866/778,660 changed bytes inside the 786,432-byte
        # binding, the changed box covering the whole binding, plus w7
        # 256/256/256, the whole publication, zero unattributed). Three
        # variants exist in the cubin (`chained`, `publish`, `fp8`): `chained`
        # is live at slots 63/68/73/78/83/88/93 (`publish` ran once at slot
        # 58), while `fp8` is created but never launched (pinned sibling).
        # The single-publication rule classifies w7 by consumption alone:
        # slot 74 binds it as w6 sync-in, so the predicted sibling
        # divergence is exactly 256 B.
        container=5,
        record_name="block34.layer0.layer", record_bytes=4194320,
        constant_values={8: 0x000000100000000C},
        word_classes=(CLS_POOL, CLS_ZERO, CLS_POOL, CLS_WEIGHTS, CLS_ZERO,
                      CLS_ZERO, CLS_POOL, CLS_POOL, CLS_CONSTANT),
        output_word=2, output_bytes=786432,
        sync_word=7, sync_bytes=256,
        input_label=72, reference_label=73),
    74: dict(
        slot=74,
        psize=72,
        name_live="cc_vit_1d_ffn_contract_chained_fp8",
        name_sibling="cc_vit_1d_ffn_contract_fp8",
        tag_live="chained", tag_sibling="fp8",
        # block34's second slot, reusing slot 64's contraction map exactly.
        # Measured at the s74 cut: w0 input 36,731,904 (slot 73's output),
        # w1 residual 56,589,312 (slot 72's projection output), w2 primary
        # output 56,785,920 (196,608 B, closing at 56,982,528), w3 live
        # weights `block34.layer1.layer` (4,196,352 B at heap offset
        # 64,988,672 — the record after block34.layer0.layer, whose
        # 4,194,320-byte payload ends at 64,988,176 and rounds up 496 B to
        # the 512-byte boundary), w4 third output 77,824 (64 B), w5 second
        # output 51,133,440 (393,216 B, closing at 51,526,656), w6 sync-in
        # 77,312 (slot 73's 256-byte w7 publication), w7 the 64-byte sync
        # record at 78,336 (slot 75 consumes it at qkv w7), and w8 the
        # (12,16) constant 0x000000100000000C. Four segments per frame in
        # names-s73-s74.txt: w5 with 391,689/391,742/391,717 changed bytes,
        # w2 with 196,552/196,550/196,552, and 64 B each at w4/w7; totals
        # 588,369/588,420/588,397, zero unattributed. The fp8 sibling matches
        # both primary and second/third outputs but leaves w7 at input state.
        container=5,
        record_name="block34.layer1.layer", record_bytes=4196352,
        constant_values={8: 0x000000100000000C},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS, CLS_POOL,
                      CLS_POOL, CLS_POOL, CLS_POOL, CLS_CONSTANT),
        output_word=2, output_bytes=196608,
        extra_outputs=((5, 393216), (4, 64)),
        sync_word=7, sync_bytes=64,
        input_label=73, reference_label=74),
    75: dict(
        slot=75,
        psize=80,
        name_live="cc_vit_1d_qkv_chained_fp8",
        name_sibling="cc_vit_1d_qkv_fp8",
        tag_live="chained", tag_sibling="fp8",
        # block34's qkv launch, sharing slot 70's ten-word map and
        # grid (32,1,2), block (32,4,1). Manifest, sidecar and both
        # traces pin every parameter word across all three frames.
        # w0 input = slot 74's w2 at 56,785,920; w1/w2/w3 the
        # 196,608-byte q/k/v triplet at 56,982,528 / 57,179,136 /
        # 57,375,744, closing at 57,572,352 (slot 76's output base).
        # w4 binds block34.layer2.layer: 3,145,856 B at heap offset
        # 69,185,024, exactly after layer1's 4,196,352-byte payload.
        # w6 writes 1,179,648 B at 22,969,344, immediately after
        # slot 70's corresponding region. w5 publishes 128 B at 78,848;
        # no other launch binds that address, so it is output data.
        # w7 consumes slot 74's 64-byte w7 record at 78,336; w8
        # publishes 128 B at 79,360, consumed by slot 76 as w5.
        # w9 is the recurring (12,16) constant. Six measured segments,
        # zero unattributed: w6 1,170,970/1,170,459/1,170,492;
        # w1 196,556/196,546/196,556; w2 196,207/196,142/196,173;
        # w3 196,508/196,535/196,555; w5/w8 128 B each. Totals
        # 1,760,497/1,759,938/1,760,032. The sibling is never vendor-
        # launched; its predicted divergence is the 128-byte w8 handoff.
        container=5,
        record_name="block34.layer2.layer", record_bytes=3145856,
        constant_values={9: 0x000000100000000C},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS,
                      CLS_POOL, CLS_POOL, CLS_POOL, CLS_POOL, CLS_CONSTANT),
        output_word=1, output_bytes=196608,
        extra_outputs=((2, 196608), (3, 196608), (6, 1179648), (5, 128)),
        sync_word=8, sync_bytes=128,
        input_label=74, reference_label=75),
    76: dict(
        slot=76,
        psize=64,
        name_live="cc_vit_1d_attention_chained_fp8",
        name_sibling="cc_vit_1d_attention_fp8",
        tag_live="chained", tag_sibling="fp8",
        # block34's attention launch, slot 71's eight-word shape:
        # grid (32,1,1), block (32,4,1), w7 the (12,16) constant.
        # w0/w1/w2 consume slot 75's q/k/v tensors at 56,982,528 /
        # 57,179,136 / 57,375,744 (196,608 B each). w3 writes
        # 196,608 B at 57,572,352, closing at slot 77's output base.
        # w4 binds block34.layer3.layer: 2 B (59 a1), heap 72,331,264,
        # after layer2's payload end 72,330,880 plus 384 B alignment.
        # As with earlier attention contracts, independently test the
        # placeholder by record rotation and whole-record complementation;
        # the wrong control targets the functionally live query at w0.
        # w5 consumes slot 75's 128-byte w8 record at 79,360; w6
        # publishes 256 B at 79,872, consumed by slot 77 as w6. No
        # sibling-shared publication: its predicted divergence is 256 B.
        # Two measured segments, zero unattributed: w3 changes
        # 196,530/196,532/196,520 B, plus 256 B at w6; totals
        # 196,786/196,788/196,776.
        container=5,
        record_name="block34.layer3.layer", record_bytes=2,
        constant_values={7: 0x000000100000000C},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS,
                      CLS_POOL, CLS_POOL, CLS_CONSTANT),
        output_word=3, output_bytes=196608,
        wrong_word=0, wrong_bytes=196608,
        sync_word=6, sync_bytes=256,
        input_label=75, reference_label=76),
    77: dict(
        slot=77,
        psize=72,
        name_live="cc_vit_1d_projection_chained_fp8",
        name_sibling="cc_vit_1d_projection_fp8",
        tag_live="chained", tag_sibling="fp8",
        # block34's closing projection; slot 72's nine-word layout,
        # grid (16,1,4), block (32,4,1), w8 the (12,16) constant.
        # w0 consumes slot76's attention output at 57,572,352; w1
        # consumes slot74's contraction residual at 56,785,920.
        # w2 writes 196,608 B at 57,768,960, immediately before slot78's
        # next activation output; w5 writes the second output region
        # 3,932,160 B at 51,526,656, ending 128 B before slot79's
        # second-output base. w3 binds block34.layer4.layer:1,050,624 B
        # at heap offset 72,331,776, after the 512-byte aligned placeholder.
        # w4 publishes 64 B at80,384 (unique output data); w6 consumes
        # slot76's 256-byte handoff at79,872; w7 publishes 64 B at80,896,
        # consumed by slot78 as w6. w8 is the (12,16) constant.
        # Four measured segments per frame, zero unattributed: w5 changes
        # 391,426/391,474/391,505 B; w2 196,550/196,533/196,528;
        # w4/w7 64 B each. Totals 588,104/588,135/588,161.
        container=5,
        record_name="block34.layer4.layer", record_bytes=1050624,
        constant_values={8: 0x000000100000000C},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS, CLS_POOL,
                      CLS_POOL, CLS_POOL, CLS_POOL, CLS_CONSTANT),
        output_word=2, output_bytes=196608,
        extra_outputs=((5, 3932160), (4, 64)),
        sync_word=7, sync_bytes=64,
        input_label=76, reference_label=77),
    78: dict(
        slot=78,
        psize=72,
        name_live="cc_vit_1d_ffn_expand_chained_fp8",
        name_sibling="cc_vit_1d_ffn_expand_fp8",
        tag_live="chained", tag_sibling="fp8",
        # Block35's opening expansion, reusing slot 73's nine-word shape:
        # grid (64,1,1), block (32,4,1), w8 (12,16) constant.
        # w0 reads slot 77's projection output at 57,768,960; w2 writes
        # 786,432 B at 37,518,336. w3 binds block35.layer0.layer,
        # 4,194,320 B at heap offset 73,382,400; w6 consumes slot77 w7
        # at 80,896. w7 publishes 256 B at 81,408, consumed by slot 79 w6.
        # The two write-set regions are w2 (779,577/779,094/779,059 B
        # changed within 786,432 B) and w7 (256 B each), zero unattributed;
        # totals 779,833/779,350/779,315. The fp8 sibling is not vendor-
        # launched; its predicted divergence is the 256-byte sync handoff.
        container=5,
        record_name="block35.layer0.layer", record_bytes=4194320,
        constant_values={8: 0x000000100000000C},
        word_classes=(CLS_POOL, CLS_ZERO, CLS_POOL, CLS_WEIGHTS, CLS_ZERO,
                      CLS_ZERO, CLS_POOL, CLS_POOL, CLS_CONSTANT),
        output_word=2, output_bytes=786432,
        sync_word=7, sync_bytes=256,
        input_label=77, reference_label=78),
    83: dict(
        slot=83,
        psize=72,
        name_live="cc_vit_1d_ffn_expand_chained_fp8",
        name_sibling="cc_vit_1d_ffn_expand_fp8",
        tag_live="chained", tag_sibling="fp8",
        # Block36 opening expansion, same map as slot78. Capture sidecar
        # pins grid (64,1,1), block (32,4,1), w0 input 58,948,608,
        # w2 output 38,304,768, w3 weight at heap offset 85,970,944
        # (block36.layer0.layer, 4,194,320 B), w6 sync-in 84,992,
        # w7 sync-out 85,504 and constant w8=(12,16). s82->s83 has two
        # fully attributed regions/frame, zero unattributed: w2 changes
        # 778,505/779,858/779,861 B; w7 256 B each; totals
        # 778,761/780,114/780,117 B. Sibling expected to match output
        # tensor and diverge only across the 256-byte w7 sync handoff.
        container=5,
        record_name="block36.layer0.layer", record_bytes=4194320,
        constant_values={8: 0x000000100000000C},
        word_classes=(CLS_POOL, CLS_ZERO, CLS_POOL, CLS_WEIGHTS,
                      CLS_ZERO, CLS_ZERO, CLS_POOL, CLS_POOL, CLS_CONSTANT),
        output_word=2, output_bytes=786432,
        sync_word=7, sync_bytes=256,
        input_label=82, reference_label=83),
    79: dict(
        slot=79,
        psize=72,
        name_live="cc_vit_1d_ffn_contract_chained_fp8",
        name_sibling="cc_vit_1d_ffn_contract_fp8",
        tag_live="chained", tag_sibling="fp8",
        # Block35's contraction, matching slot 74's nine-word map; grid
        # (16,1,4), block (32,4,1), w8 (12,16) constant. w0 consumes slot
        # 78's expansion at 37,518,336; w1 consumes slot 77's projection
        # output at 57,768,960. w2 writes 196,608 B at 57,965,568; w5
        # writes 393,216 B at 51,919,872 and w4 writes 64 B at 81,920.
        # w3 binds block35.layer1.layer (4,196,352 B at heap offset
        # 77,577,216); w6 consumes slot 78's 256-byte handoff at 81,408;
        # w7 publishes 64 B at 82,432, consumed by slot 80 as w7. Four
        # write-set segments per frame, zero unattributed: w5 changes
        # 391,846/391,724/391,816 B, w2 196,531/196,543/196,545 B, and
        # w4/w7 64 B each; totals 588,505/588,395/588,489. The fp8 sibling
        # should retain reference main outputs and leave its w7 handoff at
        # input state (64-byte predicted sync divergence).
        container=5,
        record_name="block35.layer1.layer", record_bytes=4196352,
        constant_values={8: 0x000000100000000C},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS, CLS_POOL,
                      CLS_POOL, CLS_POOL, CLS_POOL, CLS_CONSTANT),
        output_word=2, output_bytes=196608,
        extra_outputs=((5, 393216), (4, 64)),
        sync_word=7, sync_bytes=64,
        input_label=78, reference_label=79),
    80: dict(
        slot=80,
        psize=80,
        name_live="cc_vit_1d_qkv_chained_fp8",
        name_sibling="cc_vit_1d_qkv_fp8",
        tag_live="chained", tag_sibling="fp8",
        # Block35 qkv: slot 75's ten-word map and grid (32,1,2),
        # block (32,4,1). Use the launch immediately before each
        # cut-boundary slot=80 marker (the trace also contains earlier
        # same-grid qkv dispatches for other slots). w0 reads slot79's
        # contraction output at 57,965,568; w1/w2/w3 are the q/k/v
        # triplet at 58,358,784 / 58,161,024 / 58,557,824. w4 binds
        # block35.layer2.layer, 3,145,856 B at heap offset 81,773,568.
        # w6 is a 1,179,648-byte output at 33,192,960; w5 is 128 B of
        # output data at 82,944; w7 consumes slot79's 64-byte handoff at
        # 82,432; w8 publishes 128 B at 83,456, consumed by slot81 w5;
        # w9 is (12,16). Six write-set regions, zero unattributed:
        # w6 1,171,121/1,169,755/1,170,057; w3
        # 196,552/196,495/196,543; w1 196,533/196,561/196,552;
        # w2 196,179/196,182/196,165; w5/w8 128 B each; totals
        # 1,760,641/1,759,249/1,759,573. Sibling sync divergence is
        # predicted 128 B and must be confirmed by execute evidence.
        container=5,
        record_name="block35.layer2.layer", record_bytes=3145856,
        constant_values={9: 0x000000100000000C},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS,
                      CLS_POOL, CLS_POOL, CLS_POOL, CLS_POOL, CLS_CONSTANT),
        output_word=1, output_bytes=196608,
        extra_outputs=((2, 196608), (3, 196608), (6, 1179648), (5, 128)),
        sync_word=8, sync_bytes=128,
        input_label=79, reference_label=80),
    81: dict(
        slot=81,
        psize=64,
        name_live="cc_vit_1d_attention_chained_fp8",
        name_sibling="cc_vit_1d_attention_fp8",
        tag_live="chained", tag_sibling="fp8",
        # Block35 attention repeats slot76's eight-word map at grid
        # (32,1,1), block (32,4,1). Sidecar establishes w0/w1/w2 at
        # 58,358,784 / 58,161,024 / 58,557,824, w4 record at
        # heap+84,919,808, w5 sync-in at 83,456, w3 output at 58,752,000,
        # w6 sync-out at 83,968, and constant w7=(12,16). Contract's
        # s80→s81 measured write-set is two regions: w3 changes
        # 196,519/196,547/196,532 bytes; w6 changes 256 B each; totals
        # 196,775/196,803/196,788, zero unattributed. Sibling should keep
        # tensor output exact with 256-byte sync divergence.
        container=5,
        record_name="block35.layer3.layer", record_bytes=2,  # fifth observed dead attention placeholder
        constant_values={7: 0x000000100000000C},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS,
                      CLS_POOL, CLS_POOL, CLS_CONSTANT),
        output_word=3, output_bytes=196608,
        sync_word=6, sync_bytes=256,
        input_label=80, reference_label=81,
        wrong_word=0, wrong_bytes=196608),
    82: dict(
        slot=82,
        psize=72,
        name_live="cc_vit_1d_projection_chained_fp8",
        name_sibling="cc_vit_1d_projection_fp8",
        tag_live="chained", tag_sibling="fp8",
        # Block35 projection repeats slot79's nine-word map at grid
        # (16,1,4), block (32,4,1). Sidecar confirms w0/w1 at
        # 58,752,000/57,965,568, w3 weight at heap+84,920,320, w6 sync-in
        # at 83,968, w2 output at 58,948,608, w5 output at 26,508,288,
        # w4 output data at 58,358,784, w7 sync-out at 84,992 and w8=(12,16).
        # s81→s82 has four fully attributed regions/frame, zero unattributed:
        # w5 391,232/391,180/391,210 B, w2 196,504/196,446/196,447 B,
        # w4/w7 64 B each, totals 587,884/587,690/587,721 B.
        container=5,
        record_name="block35.layer4.layer", record_bytes=1050624,
        constant_values={8: 0x000000100000000C},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS, CLS_POOL,
                      CLS_POOL, CLS_POOL, CLS_POOL, CLS_CONSTANT),
        output_word=2, output_bytes=196608,
        extra_outputs=((5, 3932160), (4, 64)),
        sync_word=7, sync_bytes=64,
        input_label=81, reference_label=82),
    84: dict(
        slot=84,
        psize=72,
        name_live="cc_vit_1d_ffn_contract_chained_fp8",
        name_sibling="cc_vit_1d_ffn_contract_fp8",
        tag_live="chained", tag_sibling="fp8",
        # Block36 contraction follows slot83's expansion. The s83/s84
        # captures and parameter trace pin w0=38,304,768, w1=57,768,960,
        # w2=59,145,216, w3=weights, w4=86,016, w5=40,664,064,
        # w6=85,504, w7=86,528 and w8=(12,16). The s83→s84 named
        # write-set attributes all changes to w2/w4/w5/w7 with no
        # unattributed bytes; weight-layout verification is documented
        # alongside the slot evidence.
        container=5,
        record_name="block36.layer1.layer", record_bytes=4196352,
        constant_values={8: 0x000000100000000C},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS, CLS_POOL,
                      CLS_POOL, CLS_POOL, CLS_POOL, CLS_CONSTANT),
        output_word=2, output_bytes=196608,
        extra_outputs=((5, 393216), (4, 64)),
        sync_word=7, sync_bytes=64,
        input_label=83, reference_label=84),
}


DLL = HERE.parent / "model" / "nvngx_dlssnr.dll"
BOUNDARY_MANIFEST = HERE / "logs" / "boundary" / "a" / "manifest.txt"


def capture_dir(label: int) -> Path:
    return HERE / "logs" / "ladder" / f"s{label}"


def trace_path(label: int) -> Path:
    return HERE / "logs" / "capture" / f"nvapi-trace-ladder-s{label}.log"

CREATED_RE = re.compile(
    r"created .*block=(\d+),(\d+),(\d+) smem=(\d+) flags=(\d+) "
    r"name=(\S+) handle=(0x[0-9a-f]+)")
GRID_RE = re.compile(
    r"launch id=0x5c52bb86 .*handle=(0x[0-9a-f]+) grid=(\d+),(\d+),(\d+)")
BOUNDARY_RE = re.compile(r"^boundary eval=(\d+) psize=(\d+) pmem=([0-9a-f]+)$")
WORD7_RE = re.compile(r"^word=7 .* off=(\d+) ")
RES_RE = re.compile(
    r"^res idx=(\d+) va=0x([0-9a-f]+) size=(\d+) ")

# Mode entry plans: every mode exercises all variants on all frames (the
# single-variant contracts — name_sibling None — have one tag).
def entry_plan(contract: dict[str, Any]) -> tuple[str, ...]:
    tags = ((contract["tag_live"],) if contract["tag_sibling"] is None
            else (contract["tag_live"], contract["tag_sibling"]))
    return tuple(f"{tag}-f{frame}" for frame in range(FRAMES) for tag in tags)


MODE_TAIL = {"execute": EXEC_TAIL, "tail": TAIL_TAIL, "skip": EXEC_TAIL,
             "wrong": EXEC_TAIL}


# ---------------------------------------------------------------------------
# Evidence parsing
# ---------------------------------------------------------------------------

def weight_layout(weights: bytes) -> dict[str, dict[str, int]]:
    """Walk WEIGHTS_HT; offsets follow the measured heap layout.

    Record payloads are concatenated at 512-byte alignment (the alignment
    model validated by two independent live anchors, see module docstring).
    """
    base.require(base.unpack("<Q", weights, 0)[0] == len(weights),
                 "weights: container size mismatch")
    cursor = 8
    heap = 0
    layout: dict[str, dict[str, int]] = {}
    while cursor < len(weights):
        name_length = base.unpack("<Q", weights, cursor)[0]
        cursor += 8
        base.require(0 < name_length <= 256, "weights: name length")
        name = base.span(weights, cursor, name_length).decode("ascii")
        cursor += name_length
        size = base.unpack("<Q", weights, cursor)[0]
        cursor += 8
        record = base.span(weights, cursor, size)
        cursor += size
        declared, payload, _tag = base.unpack("<QQI", record, 0)
        base.require(declared == size, "weights: record size mismatch")
        base.require(name not in layout, "weights: duplicate name")
        heap = (heap + 511) // 512 * 512
        layout[name] = dict(heap_offset=heap, payload_bytes=payload,
                            payload_at=cursor - size + 20)
        heap += payload
    base.require(cursor == len(weights), "weights: overrun")
    layout["<heap-total>"] = dict(heap_offset=heap)
    return layout


def parameter_metadata(info: bytes) -> dict[str, Any]:
    """Parse bounded CUDA metadata forms; never skip an unknown record form."""
    position = 0
    sizes: list[int] = []
    parameters: list[tuple[int, int, int]] = []
    while position < len(info):
        form, attribute, length = base.unpack("<BBH", info, position)
        position += 4
        if form == 4:
            value = base.span(info, position, length)
            position += length
            if attribute == 0x17:
                base.require(length == 12, "abi: parameter record length")
                index, ordinal, packed = struct.unpack("<III", value)
                parameters.append((index, ordinal, packed >> 18))
        elif form in (1, 2, 3):
            if attribute == 0x19:
                base.require(form == 3, "abi: parameter size requires HVAL")
                sizes.append(length)
        else:
            raise ValueError("abi: unsupported metadata form")
    return dict(sizes=sizes, parameters=parameters)


def read_created(text: str, name: str) -> dict[str, Any]:
    records = [m for m in CREATED_RE.finditer(text) if m.group(6) == name]
    base.require(len(records) == 1, f"created: expected one record for {name}")
    record = records[0]
    return dict(block=(int(record.group(1)), int(record.group(2)), int(record.group(3))),
                smem=int(record.group(4)), flags=int(record.group(5)),
                handle=record.group(7))


def addressed_heaps(traces: dict[int, Any], heap_vas: list[int],
                    heap_size: int) -> set[int]:
    """Heap instances the traces' parameter words actually address.

    The captured resource map carries several same-size heap instances (the
    model binds duplicated views); the vendor's parameter words address
    exactly one of them, and weights-less contracts — which have no weights
    word to resolve the heap through — pin it from the traces instead.
    """
    found: set[int] = set()
    for trace in traces.values():
        for evaluations in trace["evals"]:
            for _handle, _psize, _pcap, pmem, _grid in evaluations:
                block = bytes.fromhex(pmem)
                for position in range(0, len(block) // 8 * 8, 8):
                    word = struct.unpack_from("<Q", block, position)[0]
                    found.update(va for va in heap_vas
                                 if va <= word < va + heap_size)
    return found


def heap_word_scan(trace: dict[str, Any], heap_va: int, heap_size: int,
                   layout: dict[str, dict[str, int]]) -> dict[str, int]:
    """Classify every heap-pointing parameter word: record start or mid-record."""
    starts = {entry["heap_offset"] for name, entry in layout.items()
              if name != "<heap-total>"}
    total = at_start = mid = 0
    for evaluations in trace["evals"]:
        for _handle, psize, _pcap, pmem, _grid in evaluations:
            block = bytes.fromhex(pmem)
            for position in range(0, len(block) // 8 * 8, 8):
                word = struct.unpack_from("<Q", block, position)[0]
                if heap_va <= word < heap_va + heap_size:
                    total += 1
                    if word - heap_va in starts:
                        at_start += 1
                    else:
                        mid += 1
    return dict(total=total, at_record_start=at_start, mid_record=mid)


def parse_grid(raw: str) -> tuple[int, int, int]:
    """The launch grid of one slot record (three decimal components)."""
    parts = raw.split(",")
    base.require(len(parts) == 3 and
                 all(p.isascii() and p.isdigit() for p in parts),
                 "grid: malformed geometry")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def following_records(layout: dict[str, dict[str, int]], record_name: str,
                      record_bytes: int) -> list[str]:
    """Records starting at this record's payload end, rounded up to the
    heap's 512-byte record alignment (payloads need not be aligned:
    `block23.layer2.layer` is followed by 448 bytes of padding)."""
    record_end = layout[record_name]["heap_offset"] + record_bytes
    return [name for name, entry in layout.items()
            if entry["heap_offset"] == (record_end + 511) // 512 * 512]


def setup(data: bytes, inventory: dict[str, Any],
          contract: dict[str, Any]) -> dict[str, Any]:
    """Resolve the launch, bindings and weights from retained evidence."""
    slot = contract["slot"]
    psize = contract["psize"]
    live, sibling = contract["tag_live"], contract["tag_sibling"]
    kernel_tags = [(live, contract["name_live"])]
    if contract["name_sibling"] is not None:
        kernel_tags.append((sibling, contract["name_sibling"]))
    kernel_tags = tuple(kernel_tags)
    in_label, ref_label = contract["input_label"], contract["reference_label"]
    trace_text = {label: trace_path(label).read_text(encoding="utf-8",
                                                     errors="replace")
                  for label in (in_label, ref_label)}
    traces = {label: names.read_trace(trace_path(label))
              for label in (in_label, ref_label)}

    # Creation geometry (pinned equal across runs) and per-trace handles.
    created: dict[str, dict[str, Any]] = {}
    handles: dict[int, dict[str, str]] = {label: {} for label in (in_label, ref_label)}
    for tag, name in kernel_tags:
        per_trace = {label: read_created(trace_text[label], name)
                     for label in (in_label, ref_label)}
        base.require({key: value for key, value in per_trace[in_label].items()
                      if key != "handle"} ==
                     {key: value for key, value in per_trace[ref_label].items()
                      if key != "handle"},
                     f"created: {name} geometry differs between runs")
        created[tag] = per_trace[in_label]
        for label in (in_label, ref_label):
            handles[label][tag] = per_trace[label]["handle"]

    # Parameter block: the slot's launch must be the live kernel's (identity
    # pinned per trace and frame), a full dump, stable across evals and runs.
    dumps: dict[int, str] = {}
    for label, trace in traces.items():
        base.require(len(trace["evals"]) >= FRAMES,
                     f"slot{slot}-dump: eval count")
        pmems = []
        for evaluation in trace["evals"]:
            handle, seen_size, _pcap, pmem, _grid = evaluation[slot]
            base.require(handle == handles[label][live],
                         f"slot{slot}-identity: slot {slot} is not the live launch")
            base.require(seen_size == psize and len(pmem) == 2 * psize,
                         f"slot{slot}-dump: truncated parameter dump")
            pmems.append(pmem)
        base.require(len(set(pmems)) == 1,
                     f"slot{slot}-dump: parameters vary across evals")
        dumps[label] = pmems[0]
    base.require(dumps[in_label] == dumps[ref_label],
                 f"slot{slot}-dump: parameters differ between runs")
    words = struct.unpack(f"<{psize // 8}Q", bytes.fromhex(dumps[in_label]))
    base.require(len(words) == len(contract["word_classes"]),
                 f"slot{slot}-words: class table length")
    for index, cls in enumerate(contract["word_classes"]):
        if cls == CLS_ZERO:
            base.require(words[index] == 0,
                         f"slot{slot}-words: reserved word non-zero")
        elif cls == CLS_CONSTANT:
            base.require(words[index] == contract["constant_values"][index],
                         f"slot{slot}-words: constant word mismatch")

    # Per-slot grid, read at the slot's own launch position: handles are
    # shared across slots at different geometries (the slot-26 qkv kernel runs
    # at 4,3,4 here and 4,4,4 at slot 30), so the position — not the handle —
    # is the geometry's identity. Pinned across evaluations and runs. The
    # vendor launches only the live variant at this slot; the fp8 sibling is
    # replayed at the same geometry (NATIVE-VENDOR-COMPARE.md precedent).
    slot_grids = {label: {parse_grid(evaluation[slot][4])
                          for evaluation in traces[label]["evals"]}
                  for label in (in_label, ref_label)}
    base.require(len(slot_grids[in_label]) == 1,
                 f"grid: {in_label} launch geometry varies")
    base.require(len(slot_grids[ref_label]) == 1,
                 f"grid: {ref_label} launch geometry varies")
    base.require(slot_grids[in_label] == slot_grids[ref_label],
                 "grid: geometry differs between runs")
    grids = {live: slot_grids[in_label].pop()}
    if sibling is not None:
        grids[sibling] = grids[live]
    # The fp8 sibling is created but never launched by the vendor: it has no
    # launch contract of its own (pinned per trace, each with its own
    # handle). Single-variant contracts have no sibling to pin.
    if sibling is not None:
        sibling_launches: Any = sum(1 for label in (in_label, ref_label)
                                    for m in GRID_RE.finditer(trace_text[label])
                                    if m.group(1) == handles[label][sibling])
        base.require(sibling_launches == 0, "sibling: unexpected vendor launches")
    else:
        sibling_launches = None

    # Captures: input state (after slot-1) and reference state (after slot).
    capture = {label: names.read_capture(capture_dir(label))
               for label in (in_label, ref_label)}
    base.require(capture[in_label]["slot"] == in_label and
                 capture[ref_label]["slot"] == ref_label, "capture: slot pin")
    base.require(capture[in_label]["pool_size"] == capture[ref_label]["pool_size"],
                 "capture: pool size differs")
    base.require(capture[in_label]["pool_va"] == capture[ref_label]["pool_va"],
                 "capture: pool address differs")
    base.require(capture[in_label]["frames"] == capture[ref_label]["frames"] == FRAMES,
                 "capture: frame count")

    # The capture sidecar carries the same parameter block (independent copy).
    sidecar = [BOUNDARY_RE.match(line) for line in
               (capture_dir(ref_label) / "boundary-va.txt").read_text().splitlines()
               if line]
    base.require(all(sidecar) and len(sidecar) == FRAMES, "sidecar: record count")
    base.require(all(m.group(2) == str(psize) for m in sidecar if m),
                 "sidecar: parameter size")
    base.require(all(m.group(3) == dumps[in_label] for m in sidecar if m),
                 f"slot{slot}-dump: sidecar parameters differ")

    # Pool bindings: every pool-class word points into the working pool.
    pool_va = capture[in_label]["pool_va"]
    pool_size = capture[in_label]["pool_size"]
    offsets: dict[int, int] = {}
    for index, cls in enumerate(contract["word_classes"]):
        if cls == CLS_POOL:
            base.require(pool_va <= words[index] < pool_va + pool_size,
                         f"slot{slot}-bind: word {index} outside pool")
            offsets[index] = words[index] - pool_va

    # The weights word resolves into the weight heap through the captured
    # resource map. Weights-less contracts (no CLS_WEIGHTS word) resolve the
    # heap by its pinned resource size instead: the layout anchors and the
    # heap-word scan below still apply to every contract.
    weighted = contract["word_classes"].count(CLS_WEIGHTS) == 1
    res_lines = [RES_RE.match(line) for line in
                 (capture_dir(ref_label) / "resource-map.txt").read_text().splitlines()
                 if line.startswith("res ")]
    base.require(all(res_lines), f"slot{slot}-bind: resource map format")
    if weighted:
        weight_index = contract["word_classes"].index(CLS_WEIGHTS)
        containing = [m for m in res_lines if m
                      and int(m.group(2), 16) <= words[weight_index]
                      < int(m.group(2), 16) + int(m.group(3))]
        base.require(len(containing) == 1, f"slot{slot}-bind: weight resource not unique")
        weight_res = containing[0]
        base.require(int(weight_res.group(3)) == HEAP_SIZE,
                     f"slot{slot}-bind: unexpected weight resource size")
        heap_va = int(weight_res.group(2), 16)
        heap_offset: Any = words[weight_index] - heap_va
    else:
        sized = [m for m in res_lines if m and int(m.group(3)) == HEAP_SIZE]
        base.require(len(sized) >= 1, f"slot{slot}-bind: weight heap missing")
        addressed = addressed_heaps(traces,
                                    [int(m.group(2), 16) for m in sized],
                                    HEAP_SIZE)
        base.require(len(addressed) == 1,
                     f"slot{slot}-bind: weight heap not unique")
        heap_va = addressed.pop()
        heap_offset = None

    # Heap layout anchors from the DLL's WEIGHTS_HT resource.
    _resource_offset, weights = base.PE(data).resource("WEIGHTS_HT")
    layout = weight_layout(weights)
    heap_total = layout.pop("<heap-total>")["heap_offset"]
    # The live allocation is the packed heap rounded up (448 bytes of tail
    # padding observed); the exact pins are the two anchors below.
    base.require(0 <= HEAP_SIZE - heap_total < 512,
                 "weight-heap: layout size mismatch")
    base.require(layout[ANCHOR_RECORD]["heap_offset"] ==
                 boundary_word7_offset(), "weight-heap: decoder anchor mismatch")
    record_name = contract["record_name"]
    record_bytes = contract["record_bytes"]
    if weighted:
        base.require(layout[record_name]["heap_offset"] == heap_offset,
                     f"slot{slot}-weight: record binding mismatch")
        base.require(layout[record_name]["payload_bytes"] == record_bytes,
                     f"slot{slot}-weight: record size mismatch")
    # Every heap-pointing parameter word of both traces must address a whole
    # record payload start (the structural claim, computed and enforced).
    heap_scan = {str(label): heap_word_scan(traces[label], heap_va,
                                            HEAP_SIZE, layout)
                 for label in (in_label, ref_label)}
    for label in (in_label, ref_label):
        scan = heap_scan[str(label)]
        base.require(scan["mid_record"] == 0 and
                     scan["total"] == scan["at_record_start"],
                     "weight-heap: weight pointer off record start")
    if weighted:
        record = base.span(weights, layout[record_name]["payload_at"],
                           record_bytes)
        following = following_records(layout, record_name, record_bytes)
        base.require(len(following) == 1, f"slot{slot}-weight: no following record")
        tail_live = base.span(weights, layout[following[0]]["payload_at"],
                              min(TAIL_BYTES, layout[following[0]]["payload_bytes"]))
        base.require(tail_live != bytes([EXEC_TAIL]) * len(tail_live) and
                     tail_live != bytes([TAIL_TAIL]) * len(tail_live),
                     "weight-tail: tail pattern coincides with live bytes")
    else:
        record = None

    # Kernel ABI from the SM89 cubin: one parameter of the contract's size.
    # Each kernel family is created from its own cubin (the traces pin one
    # cubin per family) and extracted from its own DLL fatbin container —
    # the transformer contracts from container 4, the ViT interval's from
    # container 5 (contract["container"], default CONTAINER).
    modules = [e for e in inventory["fatbins"][contract.get("container", CONTAINER)]["entries"]
               if e["kind"] == 2 and e["arch"] == 89]
    base.require(len(modules) == 1, "abi: unique SM89 module")
    image = base.decompress(data, modules[0])
    for tag, name in kernel_tags:
        meta = parameter_metadata(launch.kernel_metadata(image, name))
        base.require(meta["sizes"] == [psize] and
                     meta["parameters"] == [(0, 0, psize)],
                     f"abi: unexpected parameter metadata for {name}")
    base.require(all(created[tag]["smem"] == 0 for tag in dict(kernel_tags)),
                 "abi: unexpected shared memory")

    return dict(
        words=list(words), pool_offsets=offsets, heap_offset=heap_offset,
        heap_total=heap_total, heap_allocation=HEAP_SIZE,
        sibling_launches=sibling_launches, heap_scan=heap_scan,
        record_sha256=(hashlib.sha256(record).hexdigest()
                       if record is not None else None),
        block=created[live]["block"], smem=created[live]["smem"],
        grids={tag: list(grids[tag]) for tag in dict(kernel_tags)},
        pool_va=pool_va, pool_size=pool_size,
        trace_sha256={str(label): hashlib.sha256(
            trace_path(label).read_bytes()).hexdigest()
            for label in (in_label, ref_label)},
        capture_fnv={str(label): {str(p["frame"]): f"0x{p['fnv']:016x}"
                                  for p in capture[label]["pools"]}
                     for label in (in_label, ref_label)},
        record=record, image=image, capture=capture)


def boundary_word7_offset() -> int:
    for line in BOUNDARY_MANIFEST.read_text().splitlines():
        if match := WORD7_RE.match(line):
            return int(match.group(1))
    raise ValueError("weight-heap: decoder anchor word 7 missing")


# ---------------------------------------------------------------------------
# GPU replay
# ---------------------------------------------------------------------------

def rotation_extent(size: int) -> int:
    """The wrong control's inwards-rotation extent for a payload of `size`.

    The full ROTATE_BYTES rotation needs headroom (the weights-less input
    binding pins input_bytes > ROTATE_BYTES for the same reason); payloads no
    longer than the rotation — the ViT attention blocks' two-byte placeholder
    records — rotate by one byte instead, since the full rotation would be a
    no-op.
    """
    return ROTATE_BYTES if size > ROTATE_BYTES else 1


def wrong_rotation(payload: bytes) -> bytes:
    """Wrong-control corruption: rotate the payload inwards.

    A rotation that leaves the content unchanged is rejected outright — an
    inert corruption would make the wrong control's verdict meaningless.
    """
    rotate = rotation_extent(len(payload))
    corrupted = payload[rotate:] + payload[:rotate]
    base.require(corrupted != payload, "wrong control: rotation is a no-op")
    return corrupted


def run_entry(driver: Any, functions: dict[str, ct.c_void_p], plan: dict[str, Any],
              entry_id: str, frame: int, tag: str, mode: str,
              guard_words: list[int], evidence: Path) -> dict[str, Any]:
    setup_data = plan["setup"]
    contract = plan["contract"]
    capture = setup_data["capture"]
    input_pool = capture[contract["input_label"]]["pools"][frame]["data"]
    reference = capture[contract["reference_label"]]["pools"][frame]["data"]
    record = setup_data["record"]
    record_bytes = contract["record_bytes"]
    weighted = contract["word_classes"].count(CLS_WEIGHTS) == 1
    tail = bytes([MODE_TAIL[mode]]) * TAIL_BYTES
    pool_content = input_pool
    if weighted:
        weights_content = record + tail
        if mode == "wrong":
            if contract.get("wrong_word") is None:
                weights_content = wrong_rotation(record) + tail
            else:
                # Dead-record contracts (slot 61: the weight-free attention
                # family's placeholder record is invariant under
                # complementation — the retained runs measure it) re-base the
                # wrong control onto the input tensor the kernel functionally
                # reads; the weights record stays at its live content.
                wrong_start = setup_data["pool_offsets"][contract["wrong_word"]]
                wrong_bytes = contract["wrong_bytes"]
                staged = bytearray(input_pool)
                staged[wrong_start:wrong_start + wrong_bytes] = wrong_rotation(
                    bytes(staged[wrong_start:wrong_start + wrong_bytes]))
                pool_content = bytes(staged)
    else:
        # Weights-less controls (module docstring, Controls): the tail
        # pattern goes into the pool at the bytes immediately beyond the
        # input tensor — the output extent's pre-launch bytes — and the
        # wrong control rotates the input tensor inwards.
        input_start = setup_data["pool_offsets"][contract["input_word"]]
        input_bytes = contract["input_bytes"]
        tail_start = input_start + input_bytes
        base.require(tail_start + TAIL_BYTES <=
                     setup_data["pool_offsets"][contract["output_word"]] +
                     contract.get("output_bytes", OUTPUT_BYTES),
                     "input-tail: tail window not covered by the output extent")
        staged = bytearray(input_pool)
        if mode != "skip":
            # The pattern stages where reads can occur (the launched modes);
            # skip's pool is the pristine input state, keeping its diff
            # counts comparable with the ladder write-sets.
            staged[tail_start:tail_start + TAIL_BYTES] = tail
            if mode == "wrong":
                tensor = staged[input_start:input_start + input_bytes]
                staged[input_start:input_start + input_bytes] = \
                    wrong_rotation(tensor)
        pool_content = bytes(staged)

    launches = readbacks = 0
    with ExitStack() as allocations:
        pool_alloc = launch.allocate(driver, allocations, setup_data["pool_size"])
        if weighted:
            weight_alloc = launch.allocate(driver, allocations,
                                           record_bytes + TAIL_BYTES)
        initial_pool = compare.upload_region(driver, pool_alloc, pool_content,
                                             0x00, guard_words)
        if weighted:
            initial_weights = compare.upload_region(driver, weight_alloc,
                                                    weights_content,
                                                    0x00, guard_words)
        launched = mode != "skip"
        if launched:
            words = setup_data["words"]
            offsets = setup_data["pool_offsets"]
            values = []
            for index, cls in enumerate(contract["word_classes"]):
                if cls == CLS_POOL:
                    values.append(pool_alloc[1] + offsets[index])
                elif cls == CLS_WEIGHTS:
                    values.append(weight_alloc[1])
                elif cls == CLS_CONSTANT:
                    values.append(words[index])
                else:
                    values.append(0)
            arguments = (ct.c_uint64 * (contract["psize"] // 8))(*values)
            kernel_params = (ct.c_void_p * 1)(ct.addressof(arguments))
            grid = setup_data["grids"][tag]
            driver.check("cuLaunchKernel", functions[tag], *grid,
                         *setup_data["block"], setup_data["smem"],
                         None, kernel_params, None)
            launches += 1
            driver.check("cuCtxSynchronize")

        current_pool = compare.readback_region(driver, pool_alloc)
        if weighted:
            current_weights = compare.readback_region(driver, weight_alloc)
            readbacks += 2
        else:
            readbacks += 1

    pool_region = compare.region_slice(current_pool, pool_alloc)
    if weighted:
        weights_region = compare.region_slice(current_weights, weight_alloc)
    input_region = compare.region_slice(initial_pool, pool_alloc)

    # Evidence first: a failing comparison must not discard its explanation.
    retained: dict[str, str] = {}
    match, diff_count, first_diff = compare.diff_stats(reference, pool_region)
    if not match:
        window_start = max(0, first_diff - 4096)
        retained["pool-window"] = f"{entry_id}-pool-window.bin"
        (evidence / retained["pool-window"]).write_bytes(
            pool_region[window_start:window_start + 8192])
        retained["reference-window"] = f"{entry_id}-reference-window.bin"
        (evidence / retained["reference-window"]).write_bytes(
            reference[window_start:window_start + 8192])

    unchanged, unchanged_count, _ = compare.diff_stats(input_region, pool_region)
    if not unchanged:
        extent = compare.change_extent(input_region, pool_region)
    else:
        extent = (-1, -1, 0)

    # Region verdicts: every output tensor binding (the primary word plus any
    # extra output words — slot 55's pooled tensor writes a second one, slots
    # 59 and 62's contractions/projections a second and third, slot 60's qkv
    # four more beyond its primary q tensor) and
    # the sync record (its changed extent per slot, contract["sync_bytes"]).
    # main_match and main_diff_count aggregate every output region;
    # main_first_diff stays the primary output region's own first-difference
    # offset (region-relative, diagnostic only) so the field keeps one
    # unambiguous coordinate system under multi-region contracts.
    sync_bytes = contract["sync_bytes"]
    # No sync word (weights-less data-movement contracts): the compare is
    # empty by construction, matching the wait family's zero-extent shape.
    sync_start = (setup_data["pool_offsets"][contract["sync_word"]]
                  if contract["sync_word"] is not None else 0)
    main_match, main_diffs, main_first = True, 0, -1
    for index, (word, span) in enumerate(
            ((contract["output_word"],
              contract.get("output_bytes", OUTPUT_BYTES)),
             *contract.get("extra_outputs", ()))):
        output_start = setup_data["pool_offsets"][word]
        region_match, region_diffs, region_first = compare.diff_stats(
            reference[output_start:output_start + span],
            pool_region[output_start:output_start + span])
        main_match = main_match and region_match
        main_diffs += region_diffs
        if index == 0:
            main_first = region_first
    sync = pool_region[sync_start:sync_start + sync_bytes]
    if sync_bytes:
        _sync_match, sync_diff_count, _sync_first = compare.diff_stats(
            reference[sync_start:sync_start + sync_bytes], sync)
    else:
        # The wait family writes no sync record: the compare is empty by
        # measurement (sync_bytes is the write-set's changed extent) and the
        # whole-pool comparison carries the claim instead.
        sync_diff_count = 0

    entry = {
        "entry": entry_id, "frame": frame, "kernel": plan["kernels"][tag],
        "mode": mode, "launches": launches, "readbacks": readbacks,
        "pool_match": match, "pool_diff_count": diff_count,
        "pool_first_diff": first_diff,
        "pool_fnv": f"0x{names.fnv1a64(pool_region):016x}",
        "reference_fnv": f"0x{names.fnv1a64(reference):016x}",
        "input_fnv": f"0x{names.fnv1a64(input_region):016x}",
        "main_match": main_match, "main_diff_count": main_diffs,
        "main_first_diff": main_first,
        "sync_matches_reference": sync == reference[sync_start:sync_start + sync_bytes],
        "sync_diff_count": sync_diff_count,
        "sync_from_input": sync == input_region[sync_start:sync_start + sync_bytes],
        "sync_fnv": f"0x{names.fnv1a64(sync):016x}",
        "pool_unchanged": unchanged, "pool_input_diff_count": unchanged_count,
        "change_first": extent[0], "change_last": extent[1],
        "change_blocks": extent[2],
        "guards_intact": (compare.guards_intact(current_pool, guard_words) and
                          (not weighted or
                           compare.guards_intact(current_weights, guard_words))),
        "weights_immutable": (not weighted or
                              (weights_region ==
                               compare.region_slice(initial_weights, weight_alloc))),
        "retained": retained,
    }
    validate_entry(entry)
    return entry


def validate_entry(entry: dict[str, Any]) -> None:
    base.require(entry["pool_match"] == (entry["pool_diff_count"] == 0),
                 "entry: inconsistent verdict")
    base.require(entry["main_match"] == (entry["main_diff_count"] == 0),
                 "entry: inconsistent main verdict")
    base.require(entry["pool_unchanged"] == (entry["pool_input_diff_count"] == 0),
                 "entry: inconsistent input verdict")
    base.require(entry["pool_match"] == (entry["pool_fnv"] == entry["reference_fnv"]),
                 "entry: digest disagrees with comparison")
    base.require(entry["pool_unchanged"] == (entry["pool_fnv"] == entry["input_fnv"]),
                 "entry: input digest disagrees")
    base.require(entry["sync_matches_reference"] == (entry["sync_diff_count"] == 0),
                 "entry: inconsistent sync verdict")


def enforced_failures(mode: str, entries: list[dict[str, Any]],
                      live_name: str, sync_bytes: int,
                      input_targeted_wrong: bool = False) -> list[str]:
    labels: list[str] = []
    for entry in entries:
        name = entry["entry"]
        if not entry["guards_intact"]:
            labels.append(f"guards:{name}")
        if not entry["weights_immutable"]:
            labels.append(f"weights-mutated:{name}")
        if mode in ("execute", "tail"):
            check = "pool-mismatch" if mode == "execute" else "tail-mismatch"
            if entry["kernel"] == live_name:
                if not entry["pool_match"]:
                    labels.append(f"{check}:{name}")
            else:
                if not entry["main_match"]:
                    labels.append(f"{'main-mismatch' if mode == 'execute' else 'tail-main-mismatch'}:{name}")
                elif (entry["pool_diff_count"] != sync_bytes or
                      entry["sync_diff_count"] != sync_bytes):
                    # The fp8 sibling's divergence from the reference must be
                    # exactly the sync record's changed extent (the
                    # tilesync/chained family trait, measured per slot in the
                    # ladder write-sets). Equal difference counts inside the
                    # record and over the whole pool prove confinement: every
                    # differing byte lies within the record.
                    labels.append(f"sibling-divergence:{name}")
        elif mode == "skip":
            if not entry["pool_unchanged"]:
                labels.append(f"skip-changed:{name}")
            if entry["pool_match"]:
                labels.append(f"skip-matched:{name}")
        elif mode == "wrong":
            if entry["pool_match"]:
                labels.append(f"wrong-matched:{name}")
            if entry["pool_unchanged"]:
                labels.append(f"wrong-unchanged:{name}")
            if input_targeted_wrong and entry["main_match"]:
                # Input corruption itself changes the pool. Require changed
                # outputs too, rather than accepting that injected difference.
                labels.append(f"wrong-output-unchanged:{name}")
        else:
            raise ValueError(f"unknown mode {mode}")
    return labels


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def execute(data: bytes, inventory: dict[str, Any], mode: str,
            evidence: Path, progress: dict[str, Any],
            contract: dict[str, Any]) -> dict[str, Any]:
    kernel_tags = [(contract["tag_live"], contract["name_live"])]
    if contract["name_sibling"] is not None:
        kernel_tags.append((contract["tag_sibling"], contract["name_sibling"]))
    kernel_tags = tuple(kernel_tags)
    plan = dict(setup=setup(data, inventory, contract), contract=contract,
                kernels=dict(kernel_tags))
    image = plan["setup"]["image"]
    driver = utility.Driver()
    device, version = utility.select_device(driver)
    guard_words = [0x12340000 + i for i in range(GUARD_WORDS)]
    entries: list[dict[str, Any]] = []
    plan_entries = entry_plan(contract)
    with ExitStack() as resources:
        context = ct.c_void_p()
        driver.check("cuCtxCreate_v2", ct.byref(context), 0, device)
        resources.callback(driver.check, "cuCtxDestroy_v2", context)
        module = ct.c_void_p()
        driver.check("cuModuleLoadData", ct.byref(module), ct.create_string_buffer(image))
        resources.callback(driver.check, "cuModuleUnload", module)
        functions: dict[str, ct.c_void_p] = {}
        for tag, name in kernel_tags:
            function = ct.c_void_p()
            driver.check("cuModuleGetFunction", ct.byref(function), module, name.encode())
            functions[tag] = function
        for entry_id in plan_entries:
            tag, frame_text = entry_id.split("-")
            frame = int(frame_text[1:])
            entries.append(run_entry(driver, functions, plan, entry_id, frame,
                                     tag, mode, guard_words, evidence))
            progress["entries"] = entries

    weighted = contract["word_classes"].count(CLS_WEIGHTS) == 1
    failures = enforced_failures(
        mode, entries, contract["name_live"], contract["sync_bytes"],
        input_targeted_wrong=(not weighted or contract.get("wrong_word") is not None))
    return dict(
        native=True, device="NVIDIA GeForce RTX 4070 Ti SUPER",
        driver_api_version=version, mode=mode, slot=contract["slot"],
        entries_plan=list(plan_entries),
        entries=entries,
        kernels_launched=sum(e["launches"] for e in entries),
        readbacks=sum(e["readbacks"] for e in entries),
        enforced_failures=failures,
        checks_enforced=[
            ("guards_intact and weights_immutable on every entry"
             if weighted else
             "guards_intact on every entry (weights-less contract: "
             "no weight record is bound)"),
            f"whole-pool byte match against the s{contract['reference_label']} "
            f"vendor capture (execute, tail: {contract['tag_live']} variant)",
            ((f"main-tensor{'' if not contract.get('extra_outputs') else 's'} "
              "byte match for the fp8 sibling, with its divergence "
              + (f"at zero everywhere — it matches the reference over the "
                 f"whole pool (the wait family: w{contract['sync_word']} is "
                 f"the read-only handoff it waits on, changed extent 0)"
                 if contract["sync_bytes"] == 0 else
                 f"confined to the {contract['sync_bytes']}-byte sync "
                 f"record (w{contract['sync_word']})"))
              if contract["name_sibling"] is not None else
              "single-variant kernel — no fp8 sibling exists (the repack "
              "family has one kernel per direction), so the whole-pool "
              "compare alone carries the claim"),
            f"pool bit-identical to input and non-match against "
            f"s{contract['reference_label']} (skip)",
            f"non-match against s{contract['reference_label']} and pool changed "
            f"from input (wrong: "
            + (f"input tensor at w{contract['wrong_word']} rotated inwards "
               f"{rotation_extent(contract['wrong_bytes'])} bytes)"
               if weighted and contract.get("wrong_word") is not None else
               f"weights rotated inwards "
               f"{rotation_extent(contract['record_bytes'])} bytes)"
               if weighted else
               f"input tensor rotated inwards "
               f"{rotation_extent(contract['input_bytes'])} bytes)"),
        ],
        findings_recorded=[
            f"change extent of the pool against the s{contract['input_label']} "
            f"input state",
            (f"tail bytes beyond the record replaced (execute 0x{EXEC_TAIL:02X} / "
             f"tail 0x{TAIL_TAIL:02X})"
             if weighted else
             f"tail bytes beyond the input tensor replaced at the output "
             f"extent's pre-launch bytes (execute 0x{EXEC_TAIL:02X} / "
             f"tail 0x{TAIL_TAIL:02X})"),
            ("fp8 sibling replayed at the live variant's geometry "
             "(the vendor creates but never launches it: sibling_launches pinned 0)"
             if contract["name_sibling"] is not None else
             "single-variant kernel: the vendor creates and launches the only "
             "variant (pinned at the slot's own launch position)"),
            (f"{contract['sync_bytes']}-byte sync record at "
             f"w{contract['sync_word']} per entry: "
             f"matches reference / matches input"
             if contract["sync_bytes"] > 0 else
             (f"read-only sync record at w{contract['sync_word']} per entry "
              f"(the wait family writes none — changed extent 0): the "
              f"compare is empty and the whole-pool verdict carries the "
              f"claim"
              if contract["sync_word"] is not None else
              f"no sync word in the parameter block (the repack family is "
              f"outside the sync chain): the compare is empty and the "
              f"whole-pool verdict carries the claim")),
            *([f"the weights record is a dead binding: the kernel's output is "
               f"invariant under whole-record complementation (measured in "
               f"both variants at the captured conditions), so the wrong "
               f"control rotates the input tensor at w{contract['wrong_word']} "
               f"instead"]
              if weighted and contract.get("wrong_word") is not None else []),
        ],
        setup={key: value for key, value in plan["setup"].items()
               if key not in ("record", "image", "capture")},
        learned_weights_used=True, neural_image_produced=False,
        complete_native_pipeline=False,
        canaries_do_not_detect_read_only_out_of_bounds=True,
        deliberate_proprietary_payload_files_written=False)


def pair(first: Path, second: Path) -> dict[str, Any]:
    run_a = json.loads((first / "run.json").read_text())
    run_b = json.loads((second / "run.json").read_text())
    labels = pair_runs(run_a, run_b)
    return dict(pair_failures=labels, entries=len(run_a["entries"]),
                bit_identical=not labels)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

SYNC_FIXTURE = 48640   # fixture pool offset for the sync record
FIXTURE_SYNC_BYTES = 96   # fixture sync record extent
FIXTURE_LIVE = "fixture-live-kernel"
FIXTURE_SIBLING = "fixture-sibling-kernel"
SHAPE_CHECKS = 12   # contract_shape requirements per contract


def contract_shape(slot: int, contract: dict[str, Any]) -> None:
    """Contract-table invariants (self-test target): one call per contract."""
    base.require(contract["slot"] == slot, "self-test: contract slot key")
    base.require(len(contract["word_classes"]) == contract["psize"] // 8,
                 f"self-test: contract {slot} class table length")
    base.require(set(contract["constant_values"]) ==
                 {index for index, cls in enumerate(contract["word_classes"])
                  if cls == CLS_CONSTANT},
                 f"self-test: contract {slot} constant words")
    weights_words = contract["word_classes"].count(CLS_WEIGHTS)
    weighted = weights_words == 1
    base.require(weights_words <= 1 and
                 weighted == (contract["record_name"] is not None) and
                 (not weighted or contract["record_bytes"] > 0),
                 f"self-test: contract {slot} weights word count")
    base.require((contract["name_sibling"] is None) ==
                 (contract["tag_sibling"] is None),
                 f"self-test: contract {slot} sibling pair")
    extras = contract.get("extra_outputs", ())
    sync_word = contract["sync_word"]
    base.require(contract["word_classes"][contract["output_word"]] == CLS_POOL and
                 (sync_word is None or
                  contract["word_classes"][sync_word] == CLS_POOL) and
                 len({word for word, _span in extras}) == len(extras) and
                 all(contract["word_classes"][word] == CLS_POOL and
                     word not in (contract["output_word"], sync_word) and
                     span > 0
                     for word, span in extras),
                 f"self-test: contract {slot} region word class")
    base.require(0 <= contract["sync_bytes"] <= 512 and
                 (sync_word is not None or contract["sync_bytes"] == 0),
                 f"self-test: contract {slot} sync extent")
    base.require(sync_word is None or
                 contract["output_word"] != sync_word,
                 f"self-test: contract {slot} output/sync words distinct")
    base.require(len(set(entry_plan(contract))) ==
                 (1 if contract["tag_sibling"] is None else 2) * FRAMES,
                 f"self-test: contract {slot} plan distinct entries")
    base.require(contract.get("output_bytes", OUTPUT_BYTES) > 0,
                 f"self-test: contract {slot} output extent")
    input_word = contract.get("input_word")
    base.require((not weighted and input_word is not None and
                  contract["word_classes"][input_word] == CLS_POOL and
                  contract["input_bytes"] > ROTATE_BYTES) or
                 (weighted and input_word is None),
                 f"self-test: contract {slot} input binding")
    # The wrong control's rotation target: the weights record by default; a
    # contract whose record is a dead binding (slot 61) pins the input tensor
    # the kernel functionally reads instead.
    wrong_word = contract.get("wrong_word")
    base.require((wrong_word is None) == (contract.get("wrong_bytes") is None) and
                 (wrong_word is None or
                  (weighted and
                   contract["word_classes"][wrong_word] == CLS_POOL and
                   contract["wrong_bytes"] > ROTATE_BYTES)),
                 f"self-test: contract {slot} wrong target")


def make_entry(**overrides: Any) -> dict[str, Any]:
    entry = dict(entry="e0", frame=0, kernel=FIXTURE_LIVE, mode="execute",
                 launches=1, readbacks=2,
                 pool_match=True, pool_diff_count=0, pool_first_diff=-1,
                 pool_fnv="0x1", reference_fnv="0x1", input_fnv="0x2",
                 main_match=True, main_diff_count=0, main_first_diff=-1,
                 sync_matches_reference=True, sync_from_input=False,
                 sync_diff_count=0, sync_fnv="0x5",
                 pool_unchanged=False, pool_input_diff_count=5,
                 change_first=10, change_last=20, change_blocks=1,
                 guards_intact=True, weights_immutable=True, retained={})
    entry.update(overrides)
    return entry


def self_test() -> dict[str, Any]:
    # diff_stats known answers.
    base.require(compare.diff_stats(b"abc", b"abc") == (True, 0, -1),
                 "self-test: unchanged diff")
    match, count, first = compare.diff_stats(b"abc", b"aXc")
    base.require(match is False and count == 1 and first == 1,
                 "self-test: middle diff")
    match, count, first = compare.diff_stats(b"abcd", b"abcX")
    base.require(match is False and count == 1 and first == 3,
                 "self-test: last diff")
    known_answers = 3

    # Verdict controls: each corruption must raise exactly its pinned label.
    controls = 0
    base.require(enforced_failures("execute", [make_entry()],
                             FIXTURE_LIVE, FIXTURE_SYNC_BYTES) == [],
                 "self-test: clean execute rejected")
    for name, mode, overrides, expected in (
            ("execute mismatch", "execute", dict(pool_match=False, pool_diff_count=1,
                                                 pool_fnv="0x3"),
             "pool-mismatch:e0"),
            ("tail mismatch", "tail", dict(mode="tail", pool_match=False,
                                           pool_diff_count=1, pool_fnv="0x3"),
             "tail-mismatch:e0"),
            ("guards", "execute", dict(guards_intact=False), "guards:e0"),
            ("weights", "execute", dict(weights_immutable=False), "weights-mutated:e0"),
            ("skip changed", "skip", dict(mode="skip", pool_match=False,
                                          pool_diff_count=1, pool_fnv="0x3",
                                          pool_unchanged=False,
                                          pool_input_diff_count=1),
             "skip-changed:e0"),
            ("skip matched", "skip", dict(mode="skip", pool_unchanged=True,
                                          pool_input_diff_count=0,
                                          input_fnv="0x1"),
             "skip-matched:e0"),
            ("wrong matched", "wrong", dict(mode="wrong"), "wrong-matched:e0"),
            ("wrong unchanged", "wrong",
             dict(mode="wrong", pool_match=False, pool_diff_count=1, pool_fnv="0x2",
                  input_fnv="0x2", pool_unchanged=True, pool_input_diff_count=0),
             "wrong-unchanged:e0"),
            ("sibling main mismatch", "execute",
             dict(kernel=FIXTURE_SIBLING, pool_match=False, pool_diff_count=96,
                  pool_fnv="0x3", pool_first_diff=SYNC_FIXTURE,
                  main_match=False, main_diff_count=1,
                  sync_matches_reference=False, sync_from_input=True,
                  sync_diff_count=FIXTURE_SYNC_BYTES),
             "main-mismatch:e0"),
            ("sibling tail mismatch", "tail",
             dict(kernel=FIXTURE_SIBLING, mode="tail", pool_match=False,
                  pool_diff_count=96, pool_fnv="0x3",
                  pool_first_diff=SYNC_FIXTURE, main_match=False,
                  main_diff_count=1, sync_matches_reference=False,
                  sync_from_input=True, sync_diff_count=FIXTURE_SYNC_BYTES),
             "tail-main-mismatch:e0"),
            ("sibling divergence extent", "execute",
             dict(kernel=FIXTURE_SIBLING, pool_match=False,
                  pool_diff_count=FIXTURE_SYNC_BYTES + 4, pool_fnv="0x3",
                  pool_first_diff=SYNC_FIXTURE,
                  sync_matches_reference=False, sync_diff_count=FIXTURE_SYNC_BYTES),
             "sibling-divergence:e0"),
            ("sibling sync divergence", "execute",
             dict(kernel=FIXTURE_SIBLING, pool_match=False,
                  pool_diff_count=FIXTURE_SYNC_BYTES, pool_fnv="0x3",
                  pool_first_diff=SYNC_FIXTURE,
                  sync_matches_reference=False, sync_diff_count=3),
             "sibling-divergence:e0")):
        labels = enforced_failures(mode, [make_entry(**overrides)],
                                FIXTURE_LIVE, FIXTURE_SYNC_BYTES)
        base.require(expected in labels, f"self-test: {name} not detected")
        base.require(len(labels) == 1, f"self-test: {name} over-detected")
        controls += 1

    # A corrupted input can differ from the reference without affecting any
    # output: whole-pool non-equality alone must not pass this wrong control.
    input_only = make_entry(mode="wrong", pool_match=False, pool_diff_count=1,
                            pool_fnv="0x3", main_match=True, main_diff_count=0)
    base.require(enforced_failures("wrong", [input_only], FIXTURE_LIVE,
                                   FIXTURE_SYNC_BYTES, True) ==
                 ["wrong-output-unchanged:e0"],
                 "self-test: input-only corruption accepted")
    controls += 1
    sensitive = dict(input_only, main_match=False, main_diff_count=1)
    base.require(enforced_failures("wrong", [sensitive], FIXTURE_LIVE,
                                   FIXTURE_SYNC_BYTES, True) == [],
                 "self-test: sensitive input-targeted wrong rejected")

    # The wait family's zero sync extent: the sibling's divergence pin is
    # whole-pool equality, so a single differing byte anywhere is flagged.
    base.require(enforced_failures("execute", [make_entry(kernel=FIXTURE_SIBLING)],
                                   FIXTURE_LIVE, 0) == [],
                 "self-test: clean wait-family sibling rejected")
    labels = enforced_failures(
        "execute", [make_entry(kernel=FIXTURE_SIBLING, pool_match=False,
                               pool_diff_count=1, pool_fnv="0x3",
                               pool_first_diff=SYNC_FIXTURE)],
        FIXTURE_LIVE, 0)
    base.require(labels == ["sibling-divergence:e0"],
                 "self-test: wait-family sibling divergence not detected")
    controls += 1

    # validate_entry consistency controls, each with its pinned message.
    for name, overrides, expected in (
            ("match vs count", dict(pool_match=True, pool_diff_count=2, pool_fnv="0x3"),
             "entry: inconsistent verdict"),
            ("main match vs count", dict(main_match=True, main_diff_count=2),
             "entry: inconsistent main verdict"),
            ("unchanged vs count", dict(pool_unchanged=True, pool_input_diff_count=2),
             "entry: inconsistent input verdict"),
            ("digest vs match", dict(pool_match=True, pool_diff_count=0,
                                     pool_fnv="0x3", reference_fnv="0x4"),
             "entry: digest disagrees with comparison"),
            ("input digest vs unchanged", dict(pool_unchanged=True,
                                               pool_input_diff_count=0),
             "entry: input digest disagrees"),
            ("sync verdict vs count", dict(sync_matches_reference=True,
                                           sync_diff_count=2),
             "entry: inconsistent sync verdict")):
        try:
            validate_entry(make_entry(**overrides))
        except ValueError as error:
            base.require(str(error) == expected, f"self-test: {name} wrong message")
            controls += 1
        else:
            raise ValueError(f"fixture {name} corruption accepted")

    # Pair controls on synthetic runs.
    run = dict(mode="execute", setup=dict(x=1),
               entries=[make_entry()])
    clean = json.loads(json.dumps(run))
    base.require(pair_runs(run, clean) == [], "self-test: clean pair rejected")
    for name, mutate, expected in (
            ("mode", lambda r: r.update(mode="wrong"), "pair: mode"),
            ("setup", lambda r: r["setup"].update(x=2), "pair: setup"),
            ("pool digest", lambda r: r["entries"][0].update(pool_fnv="0x9",
                                                             pool_match=False,
                                                             pool_diff_count=1),
             "pair:pool-digest:e0"),
            ("verdict", lambda r: r["entries"][0].update(change_first=99),
             "pair:verdict:e0"),
            ("entries", lambda r: r["entries"].append(make_entry(entry="e1")),
             "pair: entries")):
        other = json.loads(json.dumps(run))
        mutate(other)
        labels = pair_runs(run, other)
        base.require(expected in labels, f"self-test: pair {name} not detected")
        controls += 1

    # Weight-layout known answer: three payloads force the 512-byte alignment
    # (rec0 300 B at heap 0, rec1 700 B at 512, rec2 100 B at 1536).
    container = bytearray(struct.pack("<Q", 0))
    expected_offsets = []
    heap = 0
    for index, payload in enumerate((300, 700, 100)):
        name = f"rec{index}".encode()
        container += struct.pack("<Q", len(name)) + name
        record = struct.pack("<QQI", 40 + payload, payload, 1) + bytes(payload) + \
            struct.pack("<IIQ", 0, 0, 1) + struct.pack("<I", 4)
        container += struct.pack("<Q", len(record)) + record
        heap = (heap + 511) // 512 * 512
        expected_offsets.append((heap, payload))
        heap += payload
    struct.pack_into("<Q", container, 0, len(container))
    layout = weight_layout(bytes(container))
    base.require(layout.pop("<heap-total>")["heap_offset"] == heap,
                 "self-test: layout heap total")
    for index, (offset, payload) in enumerate(expected_offsets):
        base.require(layout[f"rec{index}"]["heap_offset"] == offset and
                     layout[f"rec{index}"]["payload_bytes"] == payload,
                     f"self-test: layout record {index}")
    known_answers += 4

    # following_records known answers: the lookup matches the payload end
    # rounded up to the 512-byte record alignment — exactly. In this fixture
    # rec0's unaligned 300-byte payload is followed by 212 bytes of padding
    # before rec1 (at 512), and rec1's 700-byte payload by 324 bytes before
    # rec2 (at 1536). The fourth case pins exactness: with a 200-byte
    # hypothetical end (rounded to 1024) nothing is returned even though rec2
    # exists beyond it — the lookup is not "first record at or past the end".
    base.require(following_records(layout, "rec0", 300) == ["rec1"],
                 "self-test: following record across padding")
    base.require(following_records(layout, "rec1", 700) == ["rec2"],
                 "self-test: following record across second padding")
    base.require(following_records(layout, "rec2", 100) == [],
                 "self-test: following record at heap end")
    base.require(following_records(layout, "rec1", 200) == [],
                 "self-test: following record past rounded end")
    known_answers += 4

    # addressed_heaps known answers: the resolution returns exactly the
    # heap instances the traces' words address, among several same-size
    # resource-map instances (weights-less contracts pin the heap this way).
    def param(value: int) -> tuple[str, int, int, str, str]:
        return ("h", 8, 0, struct.pack("<Q", value).hex(), "1,1,1")

    fixture_traces = {56: dict(evals=[[param(0x1000), param(0x2000)]]),
                      57: dict(evals=[[param(0x1008), param(0x0),
                                       param(0x1FFF)]])}
    base.require(addressed_heaps(fixture_traces, [0x1000, 0x3000], 0x1000) ==
                 {0x1000}, "self-test: heap instance resolution")
    known_answers += 1
    base.require(addressed_heaps(fixture_traces, [0x1000, 0x2000], 0x1000) ==
                 {0x1000, 0x2000}, "self-test: heap instance multiplicity")
    known_answers += 1

    # parse_grid known answer and malformed-form control.
    base.require(parse_grid("4,3,4") == (4, 3, 4), "self-test: grid parse")
    known_answers += 1
    try:
        parse_grid("4,3")
    except ValueError as error:
        base.require(str(error) == "grid: malformed geometry",
                     "self-test: grid form wrong message")
        controls += 1
    else:
        raise ValueError("fixture grid corruption accepted")

    # Metadata known answer and malformed-form control.
    info = struct.pack("<BBH", 3, 0x19, 56) + \
        struct.pack("<BBH", 4, 0x17, 12) + struct.pack("<III", 0, 0, 56 << 18)
    parsed = parameter_metadata(info)
    base.require(parsed == dict(sizes=[56], parameters=[(0, 0, 56)]),
                 "self-test: metadata parse")
    known_answers += 1
    try:
        parameter_metadata(struct.pack("<BBH", 9, 0, 0))
    except ValueError as error:
        base.require(str(error) == "abi: unsupported metadata form",
                     "self-test: metadata form wrong message")
        controls += 1
    else:
        raise ValueError("fixture metadata form corruption accepted")

    # Wrong-control rotation known answers: the full rotation for payloads
    # with headroom, one byte at and below the ROTATE_BYTES boundary (slot
    # 61's two-byte scale record among them), plus the inert-corruption
    # control.
    long_payload = bytes(range(256)) * 8 + b"\xfb\x2e"   # 2,050 bytes
    base.require(wrong_rotation(long_payload) ==
                 long_payload[ROTATE_BYTES:] + long_payload[:ROTATE_BYTES],
                 "self-test: rotation extent headroom")
    known_answers += 1
    boundary = bytes(range(256)) * 4   # exactly ROTATE_BYTES bytes
    base.require(wrong_rotation(boundary) == boundary[1:] + boundary[:1],
                 "self-test: rotation extent boundary")
    known_answers += 1
    base.require(wrong_rotation(b"\xfb\x2e") == b"\x2e\xfb",
                 "self-test: rotation extent two-byte")
    known_answers += 1
    try:
        wrong_rotation(b"\xaa\xaa")
    except ValueError as error:
        base.require(str(error) == "wrong control: rotation is a no-op",
                     "self-test: rotation no-op wrong message")
        controls += 1
    else:
        raise ValueError("fixture rotation no-op accepted")

    # Contract-table shape: the class table covers the parameter block
    # exactly, the region words are pool-resident, weights/sibling/sync
    # presence is internally consistent and the weights-less input binding
    # is pinned (contract_shape, SHAPE_CHECKS requirements per contract).
    contract_checks = 0
    for slot, contract in sorted(CONTRACTS.items()):
        contract_shape(slot, contract)
        contract_checks += SHAPE_CHECKS
    known_answers += contract_checks

    # Contract-shape corruption controls: each schema violation must raise
    # exactly its pinned message (deep copies of one weighted and one
    # weights-less real contract).
    for name, template, mutate, expected in (
            ("sibling pair", 57,
             lambda c: c.update(tag_sibling="fp8"),
             "self-test: contract 57 sibling pair"),
            ("weights word count", 57,
             lambda c: c.update(record_name="block31.layer0.layer",
                                record_bytes=1),
             "self-test: contract 57 weights word count"),
            ("two weights words", 57,
             lambda c: c.update(word_classes=[CLS_WEIGHTS, CLS_WEIGHTS,
                                              CLS_CONSTANT]),
             "self-test: contract 57 weights word count"),
            ("sync extent", 57,
             lambda c: c.update(sync_bytes=96),
             "self-test: contract 57 sync extent"),
            ("input binding", 57,
             lambda c: c.update(input_bytes=1),
             "self-test: contract 57 input binding"),
            ("plan entries", 57,
             lambda c: c.update(tag_sibling="fp8",
                                name_sibling="cc_vit_1d_repack_2d_to_1d_fp8"),
             "self-test: contract 57 plan distinct entries"),
            ("sibling pair weighted", 55,
             lambda c: c.update(name_sibling=None),
             "self-test: contract 55 sibling pair"),
            ("input binding weighted", 55,
             lambda c: c.update(input_word=0),
             "self-test: contract 55 input binding"),
            ("wrong target pair", 61,
             lambda c: c.update(wrong_bytes=None),
             "self-test: contract 61 wrong target"),
            ("wrong target class", 61,
             lambda c: c.update(wrong_word=7),
             "self-test: contract 61 wrong target"),
            ("wrong target extent", 61,
             lambda c: c.update(wrong_bytes=1),
             "self-test: contract 61 wrong target"),
            ("wrong target weights-less", 57,
             lambda c: c.update(wrong_word=0, wrong_bytes=196608),
             "self-test: contract 57 wrong target")):
        fixture = copy.deepcopy(CONTRACTS[template])
        mutate(fixture)
        try:
            contract_shape(template, fixture)
        except ValueError as error:
            base.require(str(error) == expected,
                         f"self-test: {name} wrong message")
            controls += 1
        else:
            raise ValueError(f"fixture {name} corruption accepted")

    return dict(diff_known_answers=3, verdict_controls=14,
                consistency_controls=6, pair_controls=5,
                layout_known_answers=4, following_known_answers=4,
                metadata_known_answers=1,
                grid_known_answers=1,
                rotation_known_answers=3,
                contract_known_answers=contract_checks,
                corruption_controls=controls,
                known_answers=known_answers,
                gpu_kernels_launched=0)


def pair_runs(first: dict[str, Any], second: dict[str, Any]) -> list[str]:
    """pair() core on in-memory runs (self-test target)."""
    labels: list[str] = []
    if first["mode"] != second["mode"]:
        labels.append("pair: mode")
    if first["setup"] != second["setup"]:
        labels.append("pair: setup")
    entries_a = {entry["entry"]: entry for entry in first["entries"]}
    entries_b = {entry["entry"]: entry for entry in second["entries"]}
    if entries_a.keys() != entries_b.keys():
        labels.append("pair: entries")
    else:
        for name, entry in entries_a.items():
            other = entries_b[name]
            if entry["pool_fnv"] != other["pool_fnv"]:
                labels.append(f"pair:pool-digest:{name}")
            if {key: value for key, value in entry.items() if key != "retained"} != \
               {key: value for key, value in other.items() if key != "retained"}:
                labels.append(f"pair:verdict:{name}")
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=[
        "self-test", "execute", "skip", "wrong", "tail", "pair"])
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--slot", type=int, choices=sorted(CONTRACTS),
                        help="replay contract to execute (execute-family modes)")
    args = parser.parse_args()
    if args.command == "self-test":
        print(json.dumps(self_test(), indent=2))
        return 0
    if args.command == "pair":
        if len(args.paths) != 2:
            print("usage: native-slot-replay-probe pair FIRST SECOND")
            return 2
        result = pair(Path(args.paths[0]), Path(args.paths[1]))
        result["with"] = args.paths[1]
        result["run_sha256"] = {
            args.paths[i]: hashlib.sha256(
                (Path(args.paths[i]) / "run.json").read_bytes()).hexdigest()
            for i in (0, 1)}
        (Path(args.paths[0]) / "pair.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 1 if result["pair_failures"] else 0
    if len(args.paths) != 1 or args.slot is None:
        print("usage: native-slot-replay-probe "
              "{execute|skip|wrong|tail} OUTDIR --slot {24..84}")
        return 2
    evidence = Path(args.paths[0])
    evidence.mkdir(parents=True, exist_ok=True)
    data = DLL.read_bytes()
    inventory = base.inventory(data)
    progress: dict[str, Any] = {}
    result = execute(data, inventory, args.command, evidence, progress,
                     CONTRACTS[args.slot])
    (evidence / "run.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(
        mode=result["mode"], entries=len(result["entries"]),
        kernels_launched=result["kernels_launched"],
        readbacks=result["readbacks"],
        enforced_failures=result["enforced_failures"]), indent=2))
    return 1 if result["enforced_failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
