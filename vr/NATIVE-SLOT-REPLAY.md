# Native pre-decoder kernel replay (slots 24-74) — 15-23/10/2026

## Result

**Fifty-one pre-decoder kernel launches (slots 24–74; fifteen distinct kernel
names — slots 26/30/34/38/42/46/50/54 share one kernel across two geometries, slots
27/31/35/39/43/47/51, 28/32/36/40/44/48/52 and 29/33/37/41/45/49/53 reuse kernels at the same geometry, slots
24 and 25 are single-use `inpview` kernels, slots
55 and 56 are the `proj_pool` and `final_head` variants' own kernels, slot
57 is the single-use `repack` kernel opening the ViT interval, slot 58 is
the `ffn_expand` family's single-use `publish` variant — the later
`ffn_expand` launches use the `chained` variant at their own slots — slot
59 is the `ffn_contract` family's `chained` variant, the contraction
half of the ViT block's five-slot cycle, slot 60 is the `qkv`
family's `chained` variant — the cycle's third slot — its kernel shared
with the seven later blocks' qkv slots (65/70/75/80/85/90/95), and slot 61
is the `attention` family's `chained` variant — the cycle's fourth slot —
its kernel shared with the seven later blocks' attention slots
(66/71/76/81/86/91/96), and slot 62 is the `projection` family's
`chained` variant — the cycle's fifth slot closing block31's five-slot
cycle — its kernel shared with the six later blocks' chained projection
slots (67/72/77/82/87/92), the closing projection at slot 97 using the
family's `wait` variant, and slot 63 is the `ffn_expand` family's
`chained` variant's first launch — block32's cycle opening (slot 58's
`publish` opened block31's) — its kernel shared with the six later blocks'
chained expansion slots (68/73/78/83/88/93), slot 64 is the
`ffn_contract` family's `chained` variant again — slot 59's kernel reused,
the contraction half of the second block's cycle (shared with the six
later blocks' contraction slots 69/74/79/84/89/94), and slot 73 opens
block34's cycle — the `ffn_expand` chained launch reuses slot 63's kernel,
shared with later expansion slots 78/83/88/93 — slot 74 is its
`ffn_contract` chained launch reusing slot 64's kernel) of the
vendor's 156-launch sequence (**51/156**) now replay
natively on Linux and reproduce the vendor's own captured state
byte-for-byte.** Each kernel's launch is rebuilt from its captured parameter
block and executed through the **native Linux CUDA driver** on the RTX 4070
Ti SUPER — no Windows code, DirectX or Proton — and the native pool state
after the launch equals the vendor's ladder capture at the matching cut point
([LADDER.md](LADDER.md)) over the **whole 62,684,160-byte pool: 0 differing
bytes on all three frames, in both execute runs, for all fifty-one slots**.

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
| 53 | `cc_split_swin_16h_ffwd_proj_512_chained_fp8` | reads output-of-slot-52 w0 + residual w1 + previous sync w7; writes output w2 (344,064 B) + sync w6 (96 B) | s52→s53 | (8,3,1) |
| 54 | `cc_split_swin_16h_qkv_512_chained_fp8` | reads output-of-slot-53 w0 + previous sync w5; writes output w1 (344,064 B) + sync w6 (256 B) | s53→s54 | (4,4,4) |
| 55 | `cc_split_swin_16h_proj_pool_512_chained_fp8` | reads output-of-slot-54 w0 + residual w1 + previous sync w5; writes output w2 (344,064 B) + pooled output w3 (98,304 B) + sync w6 (96 B) | s54→s55 | (8,3,1) |
| 56 | `cc_split_swin_16h_final_head_512_wait_fp8` | reads pooled output w0 + previous sync w3 (waits on it — no sync written); writes output w1 (196,608 B) | s55→s56 | (8,2,1) |
| 57 | `cc_vit_1d_repack_2d_to_1d_fp8` | reads head tensor w0 (196,608 B); writes repacked tensor w1 (196,608 B) — **no weights word, no sync word, no fp8 sibling** | s56→s57 | (192,1,1) |
| 58 | `cc_vit_1d_ffn_expand_publish_fp8` | reads input w0 (196,608 B) + weights `block31.layer0.layer`; writes output w2 (786,432 B) + sync w7 (256 B) | s57→s58 | (64,1,1) |
| 59 | `cc_vit_1d_ffn_contract_chained_fp8` | reads input w0 (786,432 B) + residual w1 (196,608 B) + weights `block31.layer1.layer` + previous sync w6; writes output w2 (196,608 B) + second output w5 (393,216 B) + third output w4 (64 B) + sync w7 (64 B) | s58→s59 | (16,1,4) |
| 60 | `cc_vit_1d_qkv_chained_fp8` | reads input w0 (196,608 B) + weights `block31.layer2.layer` + previous sync w7; writes q/k/v w1, w2, w3 (196,608 B each) + output w6 (1,179,648 B) + output w5 (128 B) + sync w8 (128 B) | s59→s60 | (32,1,2) |
| 61 | `cc_vit_1d_attention_chained_fp8` | reads q/k/v w0, w1, w2 (196,608 B each) + weights `block31.layer3.layer` (2 B placeholder — a dead binding) + previous sync w5; writes output w3 (196,608 B) + sync w6 (256 B) | s60→s61 | (32,1,1) |
| 62 | `cc_vit_1d_projection_chained_fp8` | reads input w0 (196,608 B) + residual w1 (196,608 B) + weights `block31.layer4.layer` + previous sync w6; writes output w2 (196,608 B) + second output w5 (393,216 B) + third output w4 (64 B) + sync w7 (64 B) | s61→s62 | (16,1,4) |
| 63 | `cc_vit_1d_ffn_expand_chained_fp8` | reads input w0 (196,608 B) + weights `block32.layer0.layer` + previous sync w6 (64 B); writes output w2 (786,432 B) + sync w7 (256 B) | s62→s63 | (64,1,1) |
| 64 | `cc_vit_1d_ffn_contract_chained_fp8` | reads input w0 (786,432 B) + residual w1 (196,608 B) + weights `block32.layer1.layer` + previous sync w6 (256 B); writes output w2 (196,608 B) + second output w5 (393,216 B) + third output w4 (64 B) + sync w7 (64 B) | s63→s64 | (16,1,4) |
| 65 | `cc_vit_1d_qkv_chained_fp8` | reads input w0 (196,608 B) + weights `block32.layer2.layer` + previous sync w7 (64 B); writes q/k/v w1, w2, w3 (196,608 B each) + output w6 (1,179,648 B) + output w5 (128 B) + sync w8 (128 B) | s64→s65 | (32,1,2) |
| 66 | `cc_vit_1d_attention_chained_fp8` | reads q/k/v w0, w1, w2 (196,608 B each) + weights `block32.layer3.layer` (2 B placeholder — a dead binding) + previous sync w5 (128 B); writes output w3 (196,608 B) + sync w6 (256 B) | s65→s66 | (32,1,1) |
| 67 | `cc_vit_1d_projection_chained_fp8` | reads input w0 (196,608 B) + residual w1 (196,608 B) + weights `block32.layer4.layer` + previous sync w6 (256 B); writes output w2 (196,608 B) + second output w5 (393,216 B) + third output w4 (64 B) + sync w7 (64 B) | s66→s67 | (16,1,4) |
| 68 | `cc_vit_1d_ffn_expand_chained_fp8` | reads input w0 (196,608 B) + weights `block33.layer0.layer` + previous sync w6 (64 B); writes output w2 (786,432 B) + sync w7 (256 B) | s67→s68 | (64,1,1) |
| 69 | `cc_vit_1d_ffn_contract_chained_fp8` | reads input w0 (786,432 B) + residual w1 (196,608 B) + weights `block33.layer1.layer` + previous sync w6 (256 B); writes output w2 (196,608 B) + second output w5 (393,216 B) + third output w4 (64 B) + sync w7 (64 B) | s68→s69 | (16,1,4) |
| 70 | `cc_vit_1d_qkv_chained_fp8` | reads input w0 (196,608 B) + weights `block33.layer2.layer` + previous sync w7 (64 B); writes q/k/v w1, w2, w3 (196,608 B each) + output w6 (1,179,648 B) + output w5 (128 B) + sync w8 (128 B) | s69→s70 | (32,1,2) |
| 71 | `cc_vit_1d_attention_chained_fp8` | reads q/k/v w0, w1, w2 (196,608 B each) + weights `block33.layer3.layer` (2 B placeholder — a dead binding) + previous sync w5 (128 B); writes output w3 (196,608 B) + sync w6 (256 B) | s70→s71 | (32,1,1) |
| 72 | `cc_vit_1d_projection_chained_fp8` | reads input w0 (196,608 B) + residual w1 (196,608 B) + weights `block33.layer4.layer` + previous sync w6 (256 B); writes output w2 (196,608 B) + second output w5 (393,216 B) + third output w4 (64 B) + sync w7 (64 B) | s71→s72 | (16,1,4) |
| 73 | `cc_vit_1d_ffn_expand_chained_fp8` | reads input w0 (196,608 B) + weights `block34.layer0.layer` + previous sync w6 (64 B); writes output w2 (786,432 B) + sync w7 (256 B) | s72→s73 | (64,1,1) |
| 74 | `cc_vit_1d_ffn_contract_chained_fp8` | reads input w0 (786,432 B) + residual w1 (196,608 B) + weights `block34.layer1.layer` + previous sync w6 (256 B); writes output w2 (196,608 B) + second output w5 (393,216 B) + third output w4 (64 B) + sync w7 (64 B) | s73→s74 | (16,1,4) |

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
reuse pattern measured at `proj` (27/31/35/39/43/47/51) and `ffwd_proj` (29/33/37/41/45/49/53) now
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
| parameter block | trace slot dump (24 B / 40 B / 56 B / 72 B / 80 B), pinned equal across evals, runs and the capture sidecar |
| input pool state | the capture after slot n−1 (exactly what slot n reads) |
| reference pool state | the capture after slot n |
| weight bytes | `block23.layer0.layer` (524,288 B) / `block23.layer1.layer` (263,168 B) / `block23.layer2.layer` (917,568 B) / `block23.layer3.layer` (263,168 B) / `block24.layer0.layer` (524,288 B) / `block24.layer1.layer` (263,168 B) / `block24.layer2.layer` (917,568 B) / `block24.layer3.layer` (263,168 B) / `block25.layer0.layer` (524,288 B) / `block25.layer1.layer` (263,168 B) / `block25.layer2.layer` (917,568 B) / `block25.layer3.layer` (263,168 B) / `block26.layer0.layer` (524,288 B) / `block26.layer1.layer` (263,168 B) / `block26.layer2.layer` (917,568 B) / `block26.layer3.layer` (263,168 B) / `block27.layer0.layer` (524,288 B) / `block27.layer1.layer` (263,168 B) / `block27.layer2.layer` (917,568 B) / `block27.layer3.layer` (263,168 B) / `block28.layer0.layer` (524,288 B) / `block28.layer1.layer` (263,168 B) / `block28.layer2.layer` (917,568 B) / `block28.layer3.layer` (263,168 B) / `block29.layer0.layer` (524,288 B) / `block29.layer1.layer` (263,168 B) / `block29.layer2.layer` (917,568 B) / `block29.layer3.layer` (263,168 B) / `block3.layer0.layer` (20,672 B — the record walk's interloper, not contract-bound) / `block30.layer0.layer` (524,288 B) / `block30.layer1.layer` (263,168 B) / `block30.layer2.layer` (917,568 B) / `block30.layer3.layer` (263,168 B) / `block30.layer4.layer` (524,304 B) / `block31.layer0.layer` (4,194,320 B) / `block31.layer1.layer` (4,196,352 B) / `block31.layer2.layer` (3,145,856 B) / `block31.layer3.layer` (2 B — the attention family's placeholder) / `block31.layer4.layer` (1,050,624 B) / `block32.layer0.layer` (4,194,320 B) / `block32.layer1.layer` (4,196,352 B) / `block32.layer2.layer` (3,145,856 B) / `block32.layer3.layer` (2 B — the second attention placeholder) / `block32.layer4.layer` (1,050,624 B) / `block33.layer0.layer` (4,194,320 B) / `block33.layer1.layer` (4,196,352 B) / `block33.layer2.layer` (3,145,856 B) / `block33.layer3.layer` (2 B — the third attention placeholder) / `block33.layer4.layer` (1,050,624 B) / `block34.layer0.layer` (4,194,320 B) / `block34.layer1.layer` (4,196,352 B) from the DLL's WEIGHTS_HT resource |

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
w6 sync out (62,976). The slot-53 word map (72-byte block, 9 words) is slot
25's shape exactly (w5/w8 reserved zero — every `ffwd→proj` half reserves
the words): w0 output-of-slot-52 (34,618,368), w1 residual input
(34,274,304), w2 output (34,962,432), w3 weights, w4 the `(24, 28)`
constant, w5 reserved 0, w6 sync out (63,488), w7 sync in (slot 52's record
at 62,976), w8 reserved 0. The slot-54 word map (56-byte block, 7 words) is
slot 30's shape exactly (the qkv block's w4 second constant): w0
output-of-slot-53 (34,962,432), w1 output (35,306,496), w2 weights, w3 the
`(24, 28)` constant, w4 the pair's second constant `0xFFFFFFFC00000000`
(the high-dword form), w5 sync in (slot 53's record at 63,488), w6 sync out
(64,000). The slot-55 word map (80-byte block, 10 words — the first block
outside the 56/72-byte shapes, and the first with two output tensors): w0
output-of-slot-54 (35,306,496), w1 residual input (34,962,432 — the buffer
slot 53 wrote its output to), w2 output (52,313,088 — 24x28x512 =
344,064 bytes), w3 pooled output (26,803,200 — 12x16x512 = 98,304 bytes,
the `proj_pool` name's pool expansion), w4 weights, w5 sync in (slot 54's
record at 64,000), w6 sync out (64,512), w7 the pair's second constant
`0xFFFFFFFC00000000` (the high-dword form — the same value as its pair
partner slot 54's w4), w8 the (24, 28) constant and w9 the (12, 16)
constant (the pooled tensor's geometry). The slot-56 word map (40-byte
block, 5 words — the smallest of the transformer interval's blocks, the
first
contract below the
56-byte shape) is a new form (pool, pool, weights, pool, constant): w0
pooled output-of-slot-55 (26,803,200 — the `proj_pool` expansion), w1
output (52,657,152 — 196,608 bytes, 384 512-wide rows, twice the pooled
input's 192), w2 weights, w3 sync in (slot 55's record at 64,512 —
read-only: the `wait` family waits on the record and writes none), w4 the
(12, 16) constant (slot 55's w9 value recurring at the head). The slot-57
word map (24-byte block, 3 words — the new smallest of the replayed
contracts' blocks, the first below slot
56's 40, and the first with neither a weights word nor a sync word) is a
new form (pool, pool, constant): w0 the final-head tensor (52,657,152 —
address-equal to slot 56's w1, read back as the ViT interval's input), w1
output (52,853,760 — 196,608 bytes, the head's own 384×512 geometry,
closing exactly at the ViT interval's 53,050,368 boundary where slot 97's
w2 and slot 98's w0 bind) and w2 the (12, 16) constant
`0x000000100000000C` (slot 55's w9 and slot 56's w4 recurring). The slot-58
word map (72-byte block, 9 words — the ViT interval's first weighted block,
its first 72-byte shape) is a new form (pool, zero, pool, weights, zero,
zero, zero, pool, constant): w0 the repacked head tensor (52,853,760 —
address-equal to slot 57's w1, read back as the ViT block's input), w2
output (34,372,608 — 786,432 bytes, the 384×2048 fp8 expansion of the
384×512 input, closing exactly at slot 63's w2 binding at 35,159,040 — the
next `ffn_expand` output's base, whose extent repeats the same 786,432), w3
weights, w7 sync out (65,024 — the 512-byte stride continuing:
64,512 → 65,024) and w8 the (12, 16) constant `0x000000100000000C`
recurring; w1, w4, w5 and w6 are reserved zero — **no sync-in word is bound
at all** (unlike every `chained` transformer half: the interval's opening
`publish` writes the chain's first record without consuming one). The slot-59
word map (72-byte block, 9 words — the same shape as slot 58's but with
every word bound, **no reserved zero word**: eight pool words and the
constant) is a new form (pool, pool, pool, weights, pool, pool, pool,
pool, constant): w0 the fp8 expansion (34,372,608 — address-equal to slot
58's w2, read back as the contraction's input), w1 residual (52,853,760 —
slot 57's repacked head tensor bound again, the skip connection), w2
output (53,246,976 — 196,608 bytes, the 384×512 fp8 contraction closing
exactly at slot 60's w1 binding at 53,443,584), w3 weights
(`block31.layer1.layer`, 4,196,352 B at heap offset 27,223,040 — the
layout walk's next record start after `block31.layer0.layer`), w5 second
output (49,167,360 — 393,216 bytes, closing exactly at slot 62's w5
binding at 49,560,576; the `projection` half of the cycle shares this whole
word map), w4 third output (65,536 — 64 bytes, the record walk's 512-byte
stride continuing: 65,024 → 65,536), w6 sync in (65,024 — slot 58's
record), w7 the sync record (66,048 — 64 bytes, the next stride) and w8
the (12, 16) constant `0x000000100000000C` recurring. The slot-60 word
map (80-byte block, 10 words) is the `qkv` shape: w0 the input
(53,246,976 — address-equal to slot 59's w2, the contraction output read
back), w1/w2/w3 the q/k/v triplet (196,608 bytes each, 384×512 — slot
61's `attention` binds them as its w0/w1/w2, and the triplet tiles
[53,443,584, 54,033,408), closing exactly at the attention output's
base), w4 weights (`block31.layer2.layer`), w6 a fourth written region
(33,192,960 — 1,179,648 bytes = 384×3,072, closing exactly at slot 58's
w2 binding at 34,372,608; one per ViT block — block32's qkv binds its w6
at pool 20,610,048), w5 output data (66,560 — 128 bytes, the record
walk's 512-byte stride continuing: 66,048 → 66,560), w7 sync in (66,048
— slot 59's record), w8 the sync record (67,072 — 128 bytes, the next
stride; slot 61 binds its record at the attention block's w5 sync-in) and
w9 the (12, 16) constant `0x000000100000000C` recurring.

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
slot 50's word 2, slot 51's word 3, slot 52's word 2,
slot 53's word 3, slot 54's word 2, slot 55's word 4, slot 56's word 2,
slot 58's word 3, slot 59's word 3, slot 60's word 4, slot 61's word 4,
slot 62's word 3, slot 63's word 3, slot 64's word 3, slot 65's word 4, slot 66's word 4, slot 67's word 3, slot 68's word 3, slot 69's word 3, slot 70's word 4, slot 71's word 4, slot 72's word 3, slot 73's word 3, and slot 74's word 3
(all contracts through slot 74 — slot 57 carries no weights word at all,
the first weights-less block)
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
record at its 320-byte alignment boundary (20,513,792 + 20,672 + 320);
`block30.layer1.layer` at 21,059,072 — directly after that 524,288-byte
record at the 512 boundary (20,534,784 + 524,288); `block30.layer2.layer`
at 21,322,240 — directly after layer1's 263,168-byte payload at the 512
boundary (21,059,072 + 263,168); `block30.layer3.layer` at 22,240,256 —
directly after layer2's 917,568-byte payload at the 448-byte alignment
boundary (21,322,240 + 917,568 + 448); `block30.layer4.layer` at
22,503,424 — directly after layer3's 263,168-byte payload at the 512
boundary (22,240,256 + 263,168), so the
record
order is the DLL's declaration order, block24 repeats block23's
layer0-3 shape, blocks 25, 26, 27, 28 and 29 are repeating that shape again, block30 adds a fifth
record (`block30.layer4.layer`, 524,304 B — the final head's weights,
slot 56's binding), the ViT's block31 carries its own five-record layer0-4
set (bound by slots 58–62, closing at `block32.layer0.layer`'s start at
35,616,768 — directly after `block31.layer4.layer`'s 1,050,624-byte
payload, no padding intervenes), block32 repeats block31's five-record
layer0-4 shape exactly (its `block32.layer0.layer`, 4,194,320 B at 35,616,768,
is slot 63's binding and `block32.layer1.layer`, 4,196,352 B at 39,811,584,
is slot 64's — directly after layer0's 4,194,320-byte payload at the 512
boundary across 496 bytes of padding, the block31 layer0-to-layer1 pair
repeating exactly — and `block32.layer2.layer`, 3,145,856 B at 44,007,936,
is slot 65's, directly after layer1's 4,196,352-byte payload with no
padding (the block31 layer1-to-layer2 pair repeating exactly); then
`block32.layer3.layer`, 2 B at 47,154,176 — slot 66's placeholder, 384
bytes of padding after layer2's payload rounding to the 512 boundary, the
block31 layer2-to-layer3 pair repeating exactly — and `block32.layer4.layer`,
1,050,624 B at 47,154,688, slot 67's projection record (the block31
layer3-to-layer4 pair repeating exactly); the set
closes at `block33.layer0.layer`'s start at
48,205,312 — directly after `block32.layer4.layer`'s 1,050,624-byte
payload), and one auxiliary record (`block3.layer0.layer`, 20,672 B) is declared between `block29.layer3.layer` and `block30.layer0.layer`). layer2's
917,568-byte payload is not
512-aligned: the following record starts 448 bytes later at the alignment
boundary (arithmetic from the probe's live DLL walk, which enforces exactly
one record at that boundary), so the tail control's following-record lookup
is alignment-aware. Beyond the anchors, the probe
computes and enforces the structural claim itself (`setup.heap_scan`, retained
in every run's evidence): every heap-pointing parameter word of every scanned
trace lands at a record payload start — **459 of 459 in each scan record,
zero mid-record** (the scans cover s23–s74: endpoint traces s23 and s74 are
scanned once each, while every interior trace s24–s73 is scanned by both
adjacent contract setups). The binding's *identity*
(`block30.layer3.layer`, 263,168 B at slot 55; `block30.layer4.layer`,
524,304 B at slot 56; `block31.layer0.layer`, 4,194,320 B at slot 58;
`block31.layer1.layer`, 4,196,352 B at slot 59; `block31.layer2.layer`,
3,145,856 B at slot 60; `block31.layer3.layer`, 2 B at slot 61 — the
attention family's placeholder record, its payload two bytes and a dead
binding; `block31.layer4.layer`, 1,050,624 B at slot 62 — the projection
family's live record, closing block31's five-record set; `block32.layer0.layer`,
4,194,320 B at slot 63 — the second ViT block's first record and live;
`block32.layer1.layer`, 4,196,352 B at slot 64 — the contraction
family's live record; `block32.layer2.layer`, 3,145,856 B at slot 65 — the
qkv family's live record, the wrong control rotating it normally;
`block32.layer3.layer`, 2 B at slot 66 — the attention family's
placeholder, a dead binding again: the wrong control re-bases onto the
query tensor as at slot 61, and the record rotation and whole-record
complementation experiments both reproduce the execute pools
byte-for-byte; `block32.layer4.layer`, 1,050,624 B at slot 67 — the
projection family's live record, the wrong control rotating it normally,
closing block32's five-record set exactly at `block33.layer0.layer`'s
start at 48,205,312; `block33.layer0.layer`, 4,194,320 B at slot 68 —
block33's first record and live, the expansion family's record, its
start pinned exactly by slot 68's own weights word; `block33.layer1.layer`,
4,196,352 B at slot 69 — block33's second record and live, the
contraction family's record, its start pinned exactly by slot 69's own
weights word; `block33.layer2.layer`,
3,145,856 B at slot 70 — block33's third record and live, the
qkv family's record, its start pinned exactly by slot 70's own
weights word; `block33.layer3.layer`, 2 B at slot 71 —
block33's placeholder record, the attention family's third dead binding;
`block33.layer4.layer`, 1,050,624 B at slot 72 —
block33's closing record and live, the projection family's record, its
start pinned exactly by slot 72's own weights word, closing block33's
five-record set exactly at `block34.layer0.layer`'s start at 60,793,856;
`block34.layer0.layer`, 4,194,320 B at slot 73 — block34's first record
and live, the expansion family's record, its start pinned exactly by
slot 73's own weights word; `block34.layer1.layer`, 4,196,352 B at slot 74 —
block34's live contraction record at heap offset 64,988,672, pinned by
slot 74's own weights word; blocks 31, 32 and 33's five-record sets end exactly
at their successor's first record) is enforced at runtime and
named in the source contract table; what the evidence retains is the
binding's *content* and *position* — `setup.record_sha256` and
`setup.heap_offset` — plus the enforced equality to the layout walk, so a
record rename would not disturb any run's verdicts.

## Variant finding: the tilesync/chained sync record

All fifty variant-bearing contracts' fp8 siblings (`..._fp8`) are
**created but never
launched by the vendor** (pinned: 0 launches in every trace), so they have no
launch contract of their own; each is replayed at the live variant's geometry
as a variant probe. (Slot 57 is the first **single-variant** contract: the
repack family has one kernel per direction — the `2d_to_1d` here, the
`1d_to_2d` at slot 98 — with no fp8 sibling created at all, so no sibling
replay applies and the whole-pool compare alone carries the claim.) In every case the sibling's **output tensor matches the
reference byte-for-byte** (both output tensors at slot 55, all three at
slot 59, all five at slot 60, the single attention output at slot 61, all
three at slot 62, the single expansion output at slot 63, all three at slot 64, the q/k/v triplet and fourth region at slot 65, the single attention output at slot 66, all three at slot 67, single expansion outputs at slots 68 and 73, three-output contracts at slots 59, 64, 69, 72 and 74, the q/k/v triplet and fourth region at slot 70, the single attention output at slot 71) but its **sync record is left at the input state**
(`sync_from_input` true on every sibling entry) while the live variant (tilesync at
slot 24, chained at slots 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73 and 74, publish at slot 58) writes it —
except at slot 56, the first `wait`-family kernel, where **neither variant
writes a sync record** and the sibling therefore matches the reference over
the **whole pool** (the first zero-divergence sibling: `pool_diff_count` 0
on every entry, the sibling identical to the live variant's output — the
retained `sync_*` verdict fields are vacuously true at this contract, the
compare being empty by measurement; the whole-pool verdict carries the
claim). That divergence confinement
is **enforced** (`sibling-divergence`): the differences inside the sync
record must equal the total difference count over the whole pool — which
proves every differing byte lies within the record — and the record itself
must differ in **every byte of its changed extent**. The extent is **per
slot, not a constant**: 96 B at slots 24, 25, 27, 28, 29, 31, 32, 33, 35, 36, 37, 39, 40, 41, 43, 44, 45, 47, 48, 49, 51, 52, 53 and 55,
**192 B at slots 26, 34, 42 and 50**
and **256 B at slots 30, 38, 46, 54, 58, 61, 63, 66, 68, 71 and 73** (each predicted from the independent ladder
write-set and confirmed exactly by
the replay's sibling divergence — two sources, one number), **64 B at
slots 59, 62, 64, 67, 69, 72 and 74**, **128 B at
slots 60, 65 and 70** and **0 B at
slot 56** where the wait family writes no record and the sibling's whole-pool
match confirms the same zero from the replay side (slot 57 then carries **no
sync word in its parameter block at all** — the repack family is outside the
sync chain, so there is no record to compare and the compare is empty by
construction). Slot 59 re-classifies one publication along the way: its
block writes **two** 64-byte records in the chain's 512-byte stride walk
(w4 and w7), and only the w7 handoff is variant-dependent — the w4
publication is **output data**, written byte-identically by both variants
(the first sibling-shared output beyond the main tensors), so the sibling
divergence is 64 B, not the 128 B a record-walk-only prediction gives
(that prediction is measured wrong here — the retained sibling run pins
the classification). A record-walk
address alone therefore does not make a sync record. Slot 60 confirms the
classification is the family rule rather than a one-off: its block's two
publications (w5 and w8, 128 bytes each) split the same way — w8 is the
chain handoff (slot 61 binds its record at the attention block's w5
sync-in; the predicted 128 B divergence, confirmed exactly by the
retained sibling run) while w5 is output data, written byte-identically
by both variants (bound by no other launch in the sequence). Slot 61 is
the first contract with a **single** record-walk publication (w6, 256 B),
so the rule resolves it by consumption alone — slot 62 binds the record at
the projection block's w6 sync-in, hence w6 is the chain handoff and the
sibling run pins the predicted 256 B divergence exactly (`sync_from_input`
true) — with no sibling-shared publication at this block. Slot 62 returns
to the two-publication shape and confirms the classification a third time:
of its w4 (68,096) and w7 (68,608) record-walk publications only the w7
handoff is variant-dependent (slot 63 binds its record at the
`ffn_expand` block's w6 sync-in; the predicted 64 B divergence, pinned),
while w4 (bound by no other launch in the sequence) is output data written
byte-identically by both variants — the third sibling-shared publication
beyond the main tensors. Slot 63 returns to the single-publication shape
(w7, 256 B) and resolves it by consumption alone a second time: slot 64
binds the record at the `ffn_contract` block's w6 sync-in (the predicted
256 B divergence, pinned — the handoff's family shape repeating at the
second block's cycle opening), with no sibling-shared publication at this
block. Slot 64 returns to the two-publication shape and confirms the
classification a fourth time: of its w4 (69,632) and w7 (70,144)
record-walk publications only the w7 handoff is variant-dependent (slot 65
binds its record at the `qkv` block's w7 sync-in; the predicted 64 B
divergence, pinned), while w4 (bound by no other launch in the sequence)
is output data written byte-identically by both variants — the fourth
sibling-shared publication beyond the main tensors. Slot 65 is the `qkv`
shape again and confirms the classification a fifth time: of its w5
(70,656) and w8 (71,168) record-walk publications only the w8 handoff is
variant-dependent (slot 66 binds its record at the `attention` block's w5
sync-in; the predicted 128 B divergence, pinned), while w5 (bound by no
other launch in the sequence) is output data written byte-identically by
both variants — the fifth sibling-shared publication beyond the main
tensors. Slot 66 returns to the single-publication shape (w6, 256 B) and
resolves it by consumption alone a third time: slot 67 binds the record at
the `projection` block's w6 sync-in (the predicted 256 B divergence,
pinned), with no sibling-shared publication at this block. Slot 67 returns
to the two-publication shape and confirms the classification a sixth time:
of its w4 (72,192) and w7 (72,704) record-walk publications only the w7
handoff is variant-dependent (slot 68 binds its record at the `ffn_expand`
block's w6 sync-in; the predicted 64 B divergence, pinned), while w4 (bound
by no other launch in the sequence) is output data written byte-identically
by both variants — the sixth sibling-shared publication beyond the main
tensors. Slot 68 returns to the single-publication shape (w7, 256 B) and
resolves it by consumption alone a fourth time: slot 69 binds the record
at the `ffn_contract` block's w6 sync-in (the predicted 256 B divergence,
pinned), with no sibling-shared publication at this block. Slot 69
returns to the two-publication shape and confirms the classification a
seventh time: of its w4 (73,728) and w7 (74,240) record-walk
publications only the w7 handoff is variant-dependent (slot 70 binds its
record at the `qkv` block's w7 sync-in; the predicted 64 B divergence,
pinned), while w4 (bound by no other launch in the sequence) is output
data written byte-identically by both variants — the seventh
sibling-shared publication beyond the main tensors. Slot 70 repeats the
qkv family's two-publication shape and confirms the classification an
eighth time: of its w5 (74,752) and w8 (75,264) record-walk
publications only the w8 handoff is variant-dependent (slot 71 binds its
record at the attention block's w5 sync-in; the predicted 128 B
divergence,
pinned), while w5 (bound by no other launch in the sequence) is output
data written byte-identically by both variants — the eighth
sibling-shared publication beyond the main tensors. The tilesync-only
sync record is now observed at fifty sites — the decoder's out2, and slots
24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73 and 74 — a family trait of the `*_tilesync_*` / `*_chained_*` / `*_publish_*`
kernels' chain handoff (slot 58's `publish` is the wait family's mirror: it
publishes the ViT chain's first record at w7 without consuming one — its
parameter block binds no sync-in word); slot 56 (`*_wait_*`) is the first kernel outside
that family, consuming the handoff without publishing one.

## Controls

| Mode | Slot 24 | Slot 25 | Slot 26 | Slot 27 | Slot 28 | Slot 29 | Slot 30 | Slot 31 | Slot 32 | Slot 33 | Slot 34 | Slot 35 | Slot 36 | Slot 37 | Slot 38 | Slot 39 | Slot 40 | Slot 41 | Slot 42 | Slot 43 | Slot 44 | Slot 45 | Slot 46 | Slot 47 | Slot 48 | Slot 49 | Slot 50 | Slot 51 | Slot 52 | Slot 53 | Slot 54 | Slot 55 | Slot 56 | Slot 57 | Slot 58 | Slot 59 | Slot 60 | Slot 61 | Slot 62 | Slot 63 | Slot 64 | Slot 65 |  Slot 66 |  Slot 67 |  Slot 68 |  Slot 69 |  Slot 70 |  Slot 71 |  Slot 72 | Slot 73 | Slot 74 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `execute` ×2 | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** |  whole pool exact 3/3 frames; runs **pair bit-identical** |  whole pool exact 3/3 frames; runs **pair bit-identical** |  whole pool exact 3/3 frames; runs **pair bit-identical** |  whole pool exact 3/3 frames; runs **pair bit-identical** |  whole pool exact 3/3 frames; runs **pair bit-identical** |  whole pool exact 3/3 frames; runs **pair bit-identical** |  whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** | whole pool exact 3/3 frames; runs **pair bit-identical** |
| `skip` (no launch) | pool bit-identical to input and non-matching — diff counts **344,085 / 344,104 / 344,110**, exactly the ladder write-set counts | same, diffs **340,602 / 341,160 / 341,167** = the s24→s25 ladder counts | same, diffs **342,905 / 342,801 / 342,841** = the s25→s26 ladder counts | same, diffs **340,726 / 341,250 / 341,150** = the s26→s27 ladder counts | same, diffs **344,027 / 344,077 / 344,069** = the s27→s28 ladder counts | same, diffs **341,165 / 341,559 / 341,526** = the s28→s29 ladder counts | same, diffs **343,293 / 343,253 / 343,275** = the s29→s30 ladder counts | same, diffs **340,970 / 341,345 / 341,358** = the s30→s31 ladder counts | same, diffs **342,117 / 342,432 / 342,376** = the s31→s32 ladder counts | same, diffs **340,754 / 341,134 / 341,144** = the s32→s33 ladder counts | same, diffs **342,020 / 341,782 / 341,773** = the s33→s34 ladder counts | same, diffs **340,617 / 340,872 / 340,885** = the s34→s35 ladder counts | same, diffs **342,288 / 341,759 / 341,735** = the s35→s36 ladder counts | same, diffs **340,998 / 342,180 / 342,179** = the s36→s37 ladder counts | same, diffs **341,836 / 342,467 / 342,490** = the s37→s38 ladder counts | same, diffs **340,575 / 341,602 / 341,525** = the s38→s39 ladder counts | same, diffs **342,483 / 342,991 / 342,940** = the s39→s40 ladder counts | same, diffs **341,529 / 341,682 / 341,833** = the s40→s41 ladder counts | same, diffs **341,857 / 342,616 / 342,726** = the s41→s42 ladder counts | same, diffs **341,031 / 341,654 / 341,826** = the s42→s43 ladder counts | same, diffs **342,613 / 343,155 / 343,140** = the s43→s44 ladder counts | same, diffs **340,586 / 341,494 / 341,395** = the s44→s45 ladder counts | same, diffs **341,677 / 342,212 / 342,214** = the s45→s46 ladder counts | same, diffs **340,810 / 340,607 / 340,618** = the s46→s47 ladder counts | same, diffs **341,033 / 341,572 / 341,529** = the s47→s48 ladder counts | same, diffs **340,692 / 340,683 / 340,636** = the s48→s49 ladder counts | same, diffs **340,830 / 341,096 / 341,110** = the s49→s50 ladder counts | same, diffs **340,652 / 340,704 / 340,674** = the s50→s51 ladder counts | same, diffs **340,904/341,369/341,391** = the s51→s52 ladder counts | same, diffs **340,381/340,630/340,634** = the s52→s53 ladder counts | same, diffs **341,044/341,703/341,653** = the s53→s54 ladder counts | same, diffs **441,562/441,807/441,778** = the s54→s55 ladder counts | same, diffs **171,969/171,970/171,962** = the s55→s56 ladder counts | same, diffs **171,969/171,970/171,962** = the s56→s57 ladder counts | same, diffs **780,050/780,083/780,132** = the s57→s58 ladder counts | same, diffs **514,777/514,797/514,750** = the s58→s59 ladder counts | same, diffs **1,688,821/1,688,383/1,688,429** = the s59→s60 ladder counts | same, diffs **196,791/196,780/196,752** = the s60→s61 ladder counts | same, diffs **588,114/588,104/588,011** = the s61→s62 ladder counts | same, diffs **778,609/777,943/778,137** = the s62→s63 ladder counts | same, diffs **588,385/588,329/588,347** = the s63→s64 ladder counts | same, diffs **1,762,655/1,762,087/1,761,862** = the s64→s65 ladder counts |  same, diffs **196,803/196,800/196,803** = the s65→s66 ladder counts |  same, diffs **588,216/588,117/588,107** = the s66→s67 ladder counts |  same, diffs **779,710/779,061/779,068** = the s67→s68 ladder counts |  same, diffs **588,445/588,465/588,361** = the s68→s69 ladder counts |  same, diffs **1,762,760/1,762,100/1,761,914** = the s69→s70 ladder counts |  same, diffs **196,779/196,783/196,805** = the s70→s71 ladder counts |  same, diffs **588,335/588,190/588,390** = the s71→s72 ladder counts | same, diffs **779,731 / 779,122 / 778,916** = the s72→s73 ladder counts | same, diffs **588,369 / 588,420 / 588,397** = the s73→s74 ladder counts |
| `wrong` (weights rotated inwards 1,024 B; input tensor rotated at the weights-less slot 57 and at slots 61, 66 and 71, whose records are dead bindings) | rejects in all entries (319,425–319,537 bytes differ), pool changed from input (dead-launch detection) | rejects (319,545–319,658 bytes differ), pool changed from input | rejects (341,086–342,203 bytes differ), pool changed from input | rejects (342,666–342,849 bytes differ), pool changed from input | rejects (343,654–343,784 bytes differ), pool changed from input | rejects (343,429–343,791 bytes differ), pool changed from input | rejects (341,008–341,728 bytes differ), pool changed from input | rejects (343,233–343,398 bytes differ), pool changed from input | rejects (340,696–340,891 bytes differ), pool changed from input | rejects (342,186–342,308 bytes differ), pool changed from input | rejects (339,481–340,740 bytes differ), pool changed from input | rejects (342,832–343,015 bytes differ), pool changed from input | rejects (340,920–341,551 bytes differ), pool changed from input | rejects (341,331–342,654 bytes differ), pool changed from input | rejects (339,569–340,249 bytes differ), pool changed from input | rejects (342,908–343,043 bytes differ), pool changed from input | rejects (341,039–341,235 bytes differ), pool changed from input | rejects (342,327–342,472 bytes differ), pool changed from input | rejects (340,161–340,730 bytes differ), pool changed from input | rejects (342,490–342,698 bytes differ), pool changed from input | rejects (340,856–341,150 bytes differ), pool changed from input | rejects (342,399–342,648 bytes differ), pool changed from input | rejects (340,296–341,393 bytes differ), pool changed from input | rejects (342,899–343,156 bytes differ), pool changed from input | rejects (341,024–341,224 bytes differ), pool changed from input | rejects (342,682–342,841 bytes differ), pool changed from input | rejects (340,007–340,822 bytes differ), pool changed from input | rejects (342,432–342,542 bytes differ), pool changed from input | rejects (340,929–341,109 bytes differ), pool changed from input | rejects (342,329–342,499 bytes differ), pool changed from input | rejects (340,577–341,053 bytes differ), pool changed from input | rejects (428,917–429,168 bytes differ), pool changed from input | rejects (169,884–169,992 bytes differ), pool changed from input | rejects (342,948–343,184 bytes differ) — the **input tensor** rotated inwards (weights-less contract), pool changed from input | rejects (636,844–637,271 bytes differ), pool changed from input | rejects (500,508–501,272 bytes differ), pool changed from input | rejects (1,539,533–1,539,856 bytes differ), pool changed from input | rejects (331,565–331,890 bytes differ), pool changed from input — the **query tensor at w0** rotated inwards (the weights record is a 2-byte dead binding) | rejects (577,032–577,587 bytes differ), pool changed from input | rejects (730,753–733,258 bytes differ), pool changed from input | rejects (576,052–576,607 bytes differ), pool changed from input  | rejects (1,744,940–1,745,632 bytes differ), pool changed from input |  rejects (381,748–382,857 bytes differ), pool changed from input — the **query tensor at w0** rotated inwards (the weights record is a 2-byte dead binding) |  rejects (574,479–574,631 bytes differ), pool changed from input |  rejects (725,392–726,875 bytes differ), pool changed from input |  rejects (573,813–574,284 bytes differ), pool changed from input |  rejects (1,744,678–1,745,066 bytes differ), pool changed from input |  rejects (382,682–383,440 bytes differ), pool changed from input — the **query tensor at w0** rotated inwards (the weights record is a 2-byte dead binding) |  rejects (573,283–573,762 bytes differ), pool changed from input | rejects (720,842–725,565 bytes differ), pool changed from input; live weights record rotated inwards 1,024 B | rejects (see retained stage-66 run evidence) |
| `tail` (bytes beyond the record / input tensor replaced 0x5A vs 0xA5) | **output unchanged** — bounded by the 524,288-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 917,568-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 524,288-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 917,568-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 524,288-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 917,568-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 524,288-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 917,568-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 524,288-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 917,568-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 524,288-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 917,568-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 524,288-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 917,568-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 524,288-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 917,568-byte record | **output unchanged** — bounded by the 263,168-byte record | **output unchanged** — bounded by the 524,304-byte record | **output unchanged** — reads bounded by the 196,608-byte input tensor (pattern at the output extent's pre-launch bytes) | **output unchanged** — bounded by the 4,194,320-byte record (the fp8 sibling entries carry only the 256-byte sync divergence) | **output unchanged** — bounded by the 4,196,352-byte record (the fp8 sibling entries carry only the 64-byte sync divergence) | **output unchanged** — bounded by the 3,145,856-byte record (the fp8 sibling entries carry only the 128-byte sync divergence) | **output unchanged** — the weights record is a dead binding (whole-record complementation leaves the output unchanged too; the fp8 sibling entries carry only the 256-byte sync divergence) | **output unchanged** — bounded by the 1,050,624-byte record (the fp8 sibling entries carry only the 64-byte sync divergence) | **output unchanged** — bounded by the 4,194,320-byte record (the fp8 sibling entries carry only the 256-byte sync divergence) | **output unchanged** — bounded by the 4,196,352-byte record (the fp8 sibling entries carry only the 64-byte sync divergence) | **output unchanged** — bounded by the 3,145,856-byte record (the fp8 sibling entries carry only the 128-byte sync divergence) |  **output unchanged** — the weights record is a dead binding (record rotation and whole-record complementation both leave the output unchanged, measured in the retained experiment runs; the fp8 sibling entries carry only the 256-byte sync divergence) |  **output unchanged** — bounded by the 1,050,624-byte record (the fp8 sibling entries carry only the 64-byte sync divergence) |  **output unchanged** — bounded by the 4,194,320-byte record (the fp8 sibling entries carry only the 256-byte sync divergence) |  **output unchanged** — bounded by the 4,196,352-byte record (the fp8 sibling entries carry only the 64-byte sync divergence) |  **output unchanged** — bounded by the 3,145,856-byte record (the fp8 sibling entries carry only the 128-byte sync divergence) |  **output unchanged** — the weights record is a dead binding (record rotation and whole-record complementation both leave the output unchanged, measured in the retained experiment runs; the fp8 sibling entries carry only the 256-byte sync divergence) | **output unchanged** — bounded by the 1,050,624-byte record | **output unchanged** — bounded by the 4,194,320-byte record (fp8 sibling entries carry only the 256-byte sync divergence) | output unchanged beyond the 4,196,352-byte record |

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
340,808/341,273/341,295 tensor + 96 sync, slots 52→53:
340,285/340,534/340,538 tensor + 96 sync, slots 53→54:
340,788/341,447/341,397 tensor + 256 sync, slots 54→55:
441,466/441,711/441,682 tensors (343,913/343,909/343,907 output +
97,553/97,802/97,775 pooled) + 96 sync, slots 55→56:
171,969/171,970/171,962 tensor and no sync record, slots 56→57:
171,969/171,970/171,962 tensor and no sync record, slots 57→58:
779,794/779,827/779,876 tensor + 256 sync, slots 58→59:
514,649/514,669/514,622 tensors (342,690/342,703/342,652 second output +
171,959/171,966/171,970 output) + 64 sync (w7) + 64 output record (w4 — the
re-classified publication, variant-independent output data), slots 59→60:
1,688,565/1,688,127/1,688,173 tensors (1,172,971/1,172,523/1,172,573 w6
region + 171,974/171,993/171,981 + 171,959/171,944/171,950 +
171,661/171,667/171,669 q/k/v) + 128 sync (w8) + 128 output record (w5 —
the re-classified publication, variant-independent output data), slots 60→61:
196,535/196,524/196,496 tensor + 256 sync (w6 — the block's single
publication, the chain handoff), slots 61→62:
588,114/588,104/588,011 (391,486/391,452/391,349 second output +
196,500/196,524/196,534 output) + 64 sync (w7 — the chain handoff) + 64
output record (w4 — the
re-classified publication, variant-independent output data), slots 62→63:
778,353/777,687/777,881 tensor + 256 sync (w7 — the chain handoff; the
block's single publication), slots 63→64:
588,385/588,329/588,347 (391,734/391,648/391,716 second output +
196,523/196,553/196,503 output) + 64 sync (w7 — the chain handoff) + 64
output record (w4 — the
re-classified publication, variant-independent output data), slots 64→65:
1,762,655/1,762,087/1,761,862 (1,173,080/1,172,617/1,172,341 fourth
region + 196,552/196,517/196,546 v + 196,549/196,526/196,538 q +
196,218/196,171/196,181 k) + 128 sync (w8 — the chain handoff) + 128
output record (w5 — the
re-classified publication, variant-independent output data), slots 65→66:
196,803/196,800/196,803 (196,547/196,544/196,547 output) + 256 sync (w6 —
the chain handoff; the block's single publication), slots 66→67:
588,216/588,117/588,107 (391,543/391,442/391,443 second output +
196,545/196,547/196,536 output) + 64 sync (w7 — the chain handoff) + 64
output record (w4 — the
re-classified publication, variant-independent output data), slots 67→68:
779,710/779,061/779,068 (779,454/778,805/778,812 output) + 256 sync (w7 —
the chain handoff; the block's single publication), slots 68→69:
588,445/588,465/588,361 (391,762/391,784/391,712 second output +
196,555/196,553/196,521 output) + 64 sync (w7 — the chain handoff) + 64
output record (w4 — the
re-classified publication, variant-independent output data), slots 69→70:
1,762,760/1,762,100/1,761,914 (1,173,143/1,172,569/1,172,385 fourth
region + 196,549/196,547/196,549 v + 196,558/196,550/196,546 q +
196,254/196,178/196,178 k) + 128 sync (w8 — the chain handoff) + 128
output record (w5 — the
re-classified publication, variant-independent output data), slots 70→71:
196,779/196,783/196,805 (196,523/196,527/196,549 output) + 256 sync (w6 —
the chain handoff; the block's single publication), slots 71→72:
588,335/588,190/588,390 (391,671/391,507/391,719 second output +
196,536/196,555/196,543 output) + 64 sync (w7 — the chain handoff) + 64
output record (w4 — the
re-classified publication, variant-independent output data), slots 72→73:
779,731/779,122/778,916 (779,475/778,866/778,660 tensor) + 256 sync (w7 —
the single chain-handoff publication), slots 73→74:
588,369/588,420/588,397 (391,689/391,742/391,717 second output +
196,552/196,550/196,552 primary output) + 64 sync (w7 — the chain
handoff) + 64 output record (w4 — variant-independent output data) —
every execute interval with a
names file, complete, with one staging caveat: the weights-less slot 57's
execute entries count 172,457/172,461/172,456 against their tail-staged
input state (the 4,096-byte read-bounding window at the output extent's
pre-launch bytes, +488/+491/+494 over the names sums), while its skip
entries — pristine input — carry exactly the names counts; the s23→s24
attribution predates the names tool and is
stated in the slot-24 section above; the fp8
siblings show the tensor-only counts, exactly the sync extent less).

## Method

`vr/native-slot-replay-probe.py` now carries a **per-slot contract table**
(`CONTRACTS`, `--slot {24..74}`): parameter-block size and word classes
(pool / weights / constant / zero; constant words pinned per word in
`constant_values` — a block can carry more than one, at more than one
value), output and sync word indices, the primary output tensor's extent
(`output_bytes` — `OUTPUT_BYTES` where the measured binding is the standard
344,064, per contract where it is not) and the
sync record's changed extent (`sync_bytes` — measured per slot in the ladder
write-sets, enforced as the sibling's divergence; 0 at the wait family,
which waits on the record without writing one), weight record binding,
the input/reference capture pair and, per kernel family, the DLL fatbin
container the kernel image is extracted from (`container` — 4 for the
transformer interval, 5 for the ViT interval's; each family is created
from its own cubin). Three schema extensions accompany slot 57: a contract
may be **single-variant** (`name_sibling`/`tag_sibling` None — no fp8
sibling exists), **weights-less** (`CLS_WEIGHTS` absent,
`record_name`/`record_bytes` None/0 — a pure data-movement block, its
`input_word`/`input_bytes` pinning the input tensor the controls re-base
onto) and **sync-less** (`sync_word` None — no sync record to compare), and
slot 61 adds the wrong-control **re-targeting** (`wrong_word`/`wrong_bytes` —
a contract whose weights record is a dead binding pins the input tensor the
wrong rotation targets instead of the record). Grid
geometry is resolved at the
slot's own launch position (handles are shared across slots at different
geometries). The replay substitutes our own device addresses at the vendor's
own pool offsets; guards sit around every allocation and the weight buffer is
immutable-checked. Evidence
(`vr/logs/slot-replay/s{24..74}/{a,b,skip,wrong,tail}`) retains digests and
bounded windows only (windows are written before verdicts on any mismatch);
the `pair` command persists its result as `pair.json` beside the first run.

Self-test: **39 corruption controls** (13 verdict, 6 consistency, 5 pair, 1
metadata, 1 grid, 1 rotation, 12 contract-shape) each rejected with its pinned label or message, plus **630
known answers** (3 diff, 4 layout, 4 following-record, 2 heap-instance, 1 metadata, 1 grid,
3 rotation, and 12 contract-table shape checks per contract — 612 across the fifty-one
contracts, counted per contract rather than per table). The contract table's shape is itself
pinned (slot key equals the table key, class-table length = parameter size,
constant words carry a pinned entry for exactly the constant-class indices
(values enforced against the captured block at setup),
exactly one weights word (or none — the weights-less contracts bind no
record and pin their input tensor instead), the wrong target paired
(`wrong_word` with `wrong_bytes`), pool-class and beyond the rotation
extent where it is pinned (a weights-less contract must not pin one),
pool-class region words (primary, extra-output and sync words distinct,
extra-output spans positive, the primary output span positive), sync extent within the 512-byte sync stride
(a measured 0 is admitted — the wait family writes no record; a contract
with no sync word must pin 0),
output and sync words distinct, sibling name/tag presence consistent, six distinct plan entries (two
variants) or three (single-variant) — every shape
check mutation-tested at development time against a corrupted contract
entry, each rejection with its pinned message; the corruptions are
development-time in-memory mutations, not retained self-test fixtures).

## Evidence

Runs on 23/10/2026 against the final source of the slot-74 stage (the
contract-74 entry and docstring precede the runs; every later touch is
self-test, comment or text only and proven run-neutral: see
the source-neutrality note below):

- `vr/logs/ladder/{s73,s73b,s74,s74b}` — slot-73 and slot-74 reference
  captures at their cuts (validated this stage: cut accounting, pool digests, the slot-aware
  output canary, input digests;
  `ladder-report.py pair` enforces byte-equal pools with a fail-closed exit
  — passing reports are retained as `pair-s73-s73b.json` and
  `pair-s74-s74b.json`). Traces
  `vr/logs/capture/nvapi-trace-ladder-s{73,74}{,b}.log` (both parameter
  blocks are stable across evaluations and identical between paired runs;
  manifest and sidecar agree byte-for-byte).
- `vr/logs/ladder/diff-s72-s73.json`, `vr/logs/ladder/names-s72-s73.txt` —
  the measured slot-73 write-set (2 segments per frame, zero unattributed:
  779,475/778,866/778,660 changed bytes at w2 inside the 786,432-byte
  binding — the changed box covers the whole binding — plus 256 B at w7
  [77,312, 77,568), the whole publication; frame totals
  779,731/779,122/778,916).
- `vr/logs/ladder/{validate-s74.json,validate-s74b.json,pair-s74-s74b.json}`
  and `vr/logs/ladder/{diff-s73-s74.json,names-s73-s74.txt}` — both captures
  validate and pair bit-identically; the measured four-segment write-set has
  588,369/588,420/588,397 changed bytes (w5 391,689/391,742/391,717; w2
  196,552/196,550/196,552; w4 and w7 64 B each; zero unattributed).
- `vr/logs/slot-replay/s74/{a,b,skip,wrong,tail}` — six-entry runs, pair
  bit-identical (`run.json` SHA-256 `83ba0c358858f220dc2281006483b2758eeb973a83dbb70a3f0e50ffd4bb703f`);
  skip diffs equal the ladder write-set, wrong mode rejects (574,153–574,546
  bytes differ), and tail poisoning reproduces the live execute pool byte-for-byte. The fp8 sibling
  matches all three outputs (including the shared 64-byte w4 data) while differing
  exactly across the 64-byte w7 sync handoff.
- Fresh corrected-source replays `vr/logs/slot-replay/s74/{verified-a,
  verified-skip,verified-wrong-retry,verified-tail}` pair bit-identically
  with the retained execute/skip/wrong/tail runs. Fresh ladder validation,
  pairing, diff and names attribution are retained under `vr/logs/stage66/`;
  the names output is byte-identical to `names-s73-s74.txt`. A combined
  control command timed out during the wrong run; no replay process remained,
  and a new wrong-run directory was used for the successful rerun.
- `vr/logs/slot-replay/s73/recheck-stage66` and
  `vr/logs/slot-replay/s72/recheck-stage66` — source-neutrality rechecks paired
  bit-identically with their retained `a` runs (`4b4171a3…` and `4099a395…`).
- `vr/logs/slot-replay/s73/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s
  `4b4171a3…`); the skip/wrong/tail signatures match the mode contracts
  (skip pool bit-identical to input and non-matching — diffs
  779,731/779,122/778,916 = the s72→s73 ladder counts; wrong changed from
  input and non-matching — 720,842–725,565 bytes differ from the reference
  with the live `block34.layer0.layer` record rotated inwards 1,024 B; tail
  output unchanged — bounded by the 4,194,320-byte record). The fp8
  sibling's output entries match reference byte-for-byte with divergence
  exactly the 256-byte sync record in every frame (`sync_from_input` true),
  confirming the single-publication handoff prediction.
- **Additivity/source neutrality:** the slot-73 and slot-72 execute runs were
  re-taken under the slot-74 source (`s73/recheck-stage66`,
  `s72/recheck-stage66`; the pairings persisted beside the recheck runs as
  `pair.json`, and the original a-vs-b pair results are
  untouched) and pair **bit-identical** with the retained `s73/a` and
  `s72/a` (equal `run.json` SHA-256s, `4b4171a3…` and `4099a395…`), so the
  contract-74 entry changes no prior verdict.
- `python3 vr/native-slot-replay-probe.py self-test` (630 known answers / 39
  controls, `vr/logs/stage66/selftest-verified.json`), `python3 vr/ladder-report.py
  self-test` (19 controls) and `python3 vr/ladder-names.py self-test` (14
  controls) all pass; fresh outputs retained at
  `vr/logs/stage66/{ladder-selftest-verified.txt,names-selftest-verified.txt}`.

Runs on 21/10/2026 against the final source of the slot-71 stage (the
contract-71 entry and docstring precede the runs; every later touch is
self-test, comment or text only and proven run-neutral: see
the source-neutrality note below):

- `vr/logs/ladder/{s71,s71b}` — the slot-71 reference captures at the s71
  cut (validated this stage: cut accounting, pool digests, the slot-aware
  output canary, input digests;
  `ladder-report.py pair` enforces byte-equal pools with a fail-closed exit
  — its passing report is retained as `pair-s71-s71b.json`). Traces
  `vr/logs/capture/nvapi-trace-ladder-s71{,b}.log` (the slot-71 parameter
  block is stable across evaluations and identical between the runs —
  manifest and sidecar agree byte-for-byte).
- `vr/logs/ladder/diff-s70-s71.json`, `vr/logs/ladder/names-s70-s71.txt` —
  the measured slot-71 write-set (2 segments per frame, zero unattributed:
  196,523/196,527/196,549 changed bytes at w3 inside the 196,608-byte
  binding — the changed box covering the whole binding, 85/81/59 interior
  bytes static — plus 256 B changed at w6 [75,776, 76,032), the whole
  publication; frame totals 196,779/196,783/196,805).
- `vr/logs/slot-replay/s71/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s
  `50ce1b9d…`); the skip/wrong/tail signatures match the mode contracts
  (skip pool bit-identical to input and non-matching — diffs
  196,779/196,779/196,783/196,783/196,805/196,805 = the
  s70→s71 ladder counts, wrong changed from input and non-matching —
  382,682–383,440 bytes differ from the reference with the query tensor
  at w0 rotated inwards 1,024 B, the family's third dead-binding
  re-targeting, tail output unchanged — the weights record is a 2-byte
  dead binding). The fp8 sibling's entries match the reference's
  output byte-for-byte with divergence exactly the 256-byte sync record in
  every frame (`sync_from_input` true) — the attention family's third
  single-publication handoff (w6 the chain handoff, no sibling-shared
  publication at this contract). The dead-binding claim has its own
  slot-71 evidence: the retained `wrong-record-rotation-experiment` and
  `wrong-complement-experiment` runs (temporary probe variants with the
  query re-targeting removed — record rotation and whole-record
  complementation respectively) both reproduce every execute pool
  byte-for-byte (live entries 0 diffs / `pool_match` true), proving the
  kernel reads nothing from the placeholder record.
- **Additivity/source neutrality:** the slot-70 and slot-69 execute runs were
  re-taken under the slot-71 source (`s70/recheck-stage63`,
  `s69/recheck-stage63`; the pairings persisted beside the recheck runs as
  `pair.json`, and `s70/a`'s and `s69/a`'s retained a-vs-b `pair.json` are
  untouched) and pair **bit-identical** with the retained `s70/a` and
  `s69/a` (equal `run.json` SHA-256s, `575a7e6b…` and `16cb2374…`), so the
  contract-71 entry changes no prior verdict.
- `python3 vr/native-slot-replay-probe.py self-test` (594 known answers / 39
  controls, `selftest-stage63.json`), `python3 vr/ladder-report.py
  self-test` (19 controls) and `python3 vr/ladder-names.py self-test` (14
  controls) all pass.

Runs on 21/10/2026 against the final source of the slot-70 stage (the
contract-70 entry and docstring precede the runs; every later touch is
self-test, comment or text only and proven run-neutral: see
the source-neutrality note below):

- `vr/logs/ladder/{s70,s70b}` — the slot-70 reference captures at the s70
  cut (validated this stage: cut accounting, pool digests, the slot-aware
  output canary, input digests;
  `ladder-report.py pair` enforces byte-equal pools with a fail-closed exit
  — its passing report is retained as `pair-s70-s70b.json`). Traces
  `vr/logs/capture/nvapi-trace-ladder-s70{,b}.log` (the slot-70 parameter
  block is stable across evaluations and identical between the runs —
  manifest and sidecar agree byte-for-byte).
- `vr/logs/ladder/diff-s69-s70.json`, `vr/logs/ladder/names-s69-s70.txt` —
  the measured slot-70 write-set (6 segments per frame, zero unattributed:
  1,173,143/1,172,569/1,172,385 changed bytes at w6 inside the 1,179,648-byte
  binding — the changed box covering the whole binding — and 196,558/196,550/196,546
  at w1, 196,254/196,178/196,178 at w2, 196,549/196,547/196,549 at w3
  inside the 196,608-byte bindings — all three covering their whole
  bindings — plus 128 B changed at w5 [74,752, 74,880) and 128 B at w8
  [75,264, 75,392), the whole publications).
- `vr/logs/slot-replay/s70/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s
  `575a7e6b…`); the skip/wrong/tail signatures match the mode contracts
  (skip pool bit-identical to input and non-matching — diffs
  1,762,760/1,762,760/1,762,100/1,762,100/1,761,914/1,761,914 = the
  s69→s70 ladder counts, wrong changed from input and non-matching —
  1,744,678–1,745,066 bytes differ from the reference with the weights
  record rotated inwards 1,024 B, tail output unchanged — bounded by the
  3,145,856-byte record). The fp8 sibling's entries match the reference's
  output byte-for-byte with divergence exactly the 128-byte sync record in
  every frame (`sync_from_input` true) — the record-walk re-classification's
  eighth confirmation (w8 the handoff, w5 variant-independent output data
  shared by both variants).
- **Additivity/source neutrality:** the slot-69 and slot-68 execute runs were
  re-taken under the slot-70 source (`s69/recheck-stage62`,
  `s68/recheck-stage62`; the pairings persisted beside the recheck runs as
  `pair.json`, and `s69/a`'s and `s68/a`'s retained a-vs-b `pair.json` are
  untouched) and pair **bit-identical** with the retained `s69/a` and
  `s68/a` (equal `run.json` SHA-256s, `16cb2374…` and `852ce980…`), so the
  contract-70 entry changes no prior verdict.
- `python3 vr/native-slot-replay-probe.py self-test` (582 known answers / 39
  controls, `selftest-stage62.json`), `python3 vr/ladder-report.py
  self-test` (19 controls) and `python3 vr/ladder-names.py self-test` (14
  controls) all pass.

Runs on 21/10/2026 against the final source of the slot-69 stage (the
contract-69 entry and docstring precede the runs; every later touch is
self-test, comment or text only and proven run-neutral: see
the source-neutrality note below):

- `vr/logs/ladder/{s69,s69b}` — the slot-69 reference captures at the s69
  cut (validated this stage: cut accounting, pool digests, the slot-aware
  output canary, input digests;
  `ladder-report.py pair` enforces byte-equal pools with a fail-closed exit
  — its passing report is retained as `pair-s69-s69b.json`). Traces
  `vr/logs/capture/nvapi-trace-ladder-s69{,b}.log` (the slot-69 parameter
  block is stable across evaluations and identical between the runs —
  manifest and sidecar agree byte-for-byte).
- `vr/logs/ladder/diff-s68-s69.json`, `vr/logs/ladder/names-s68-s69.txt` —
  the measured slot-69 write-set (4 segments per frame, zero unattributed:
  391,762/391,784/391,712 changed bytes at w5 inside the 393,216-byte
  binding and 196,555/196,553/196,521 at w2 inside the 196,608-byte
  binding — both spans covering their whole bindings — plus 64 B changed
  at w4 [73,728, 73,792) and 64 B at w7 [74,240, 74,304), the whole
  publications).
- `vr/logs/slot-replay/s69/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s
  `16cb2374…`); the skip/wrong/tail signatures match the mode contracts
  (skip pool bit-identical to input and non-matching — diffs
  588,445/588,445/588,465/588,465/588,361/588,361 = the s68→s69 ladder
  counts, wrong changed from input and non-matching — 573,813–574,284
  bytes differ from the reference with the weights record rotated inwards
  1,024 B, tail output unchanged — bounded by the 4,196,352-byte record).
  The fp8 sibling's entries match the reference's output byte-for-byte
  with divergence exactly the 64-byte sync record in every frame
  (`sync_from_input` true) — the record-walk re-classification's seventh
  confirmation (w7 the handoff, w4 variant-independent output data shared
  by both variants).
- **Additivity/source neutrality:** the slot-68 and slot-67 execute runs were
  re-taken under the slot-69 source (`s68/recheck-stage61`,
  `s67/recheck-stage61`; the pairings persisted beside the recheck runs as
  `pair.json`, and `s68/a`'s and `s67/a`'s retained a-vs-b `pair.json` are
  untouched) and pair **bit-identical** with the retained `s68/a` and
  `s67/a` (equal `run.json` SHA-256s, `852ce980…` and `02704ec7…`), so the
  contract-69 entry changes no prior verdict.
- `python3 vr/native-slot-replay-probe.py self-test` (570 known answers / 39
  controls, `selftest-stage61.json`), `python3 vr/ladder-report.py
  self-test` (19 controls) and `python3 vr/ladder-names.py self-test` (14
  controls) all pass.

Runs on 21/10/2026 against the final source of the slot-68 stage (the
contract-68 entry and docstring precede the runs; every later touch is
self-test, comment or text only and proven run-neutral: see
the source-neutrality note below):

- `vr/logs/ladder/{s68,s68b}` — the slot-68 reference captures at the s68
  cut (validated this stage: cut accounting, pool digests, the slot-aware
  output canary, input digests;
  `ladder-report.py pair` enforces byte-equal pools with a fail-closed exit
  — its passing report is retained as `pair-s68-s68b.json`). Traces
  `vr/logs/capture/nvapi-trace-ladder-s68{,b}.log` (the slot-68 parameter
  dump is stable across evaluations and identical between the runs; the
  capture sidecar's three records match it byte-for-byte).
- `vr/logs/ladder/diff-s67-s68.json`, `vr/logs/ladder/names-s67-s68.txt` —
  the measured slot-68 write-set (2 segments per frame, zero unattributed:
  779,454/778,805/778,812 changed bytes at w2 inside the 786,432-byte
  binding, the span [35,945,472, 36,731,904) covering the whole binding;
  and 256 B changed at w7 [73,216, 73,472) — the whole publication).
- `vr/logs/slot-replay/s68/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s
  `852ce980…`); the skip/wrong/tail signatures match the mode contracts
  (skip pool bit-identical to input and non-matching — diffs
  779,710/779,061/779,068 = the s67→s68 ladder counts, wrong changed from
  input and non-matching — 725,392–726,875 bytes differ from the reference
  with the weights record rotated inwards 1,024 B, tail output unchanged
  — bounded by the 4,194,320-byte record). The fp8 sibling's entries match
  the reference's output byte-for-byte with divergence exactly the 256-byte
  sync record in every frame (`sync_from_input` true) — the
  single-publication rule's fourth confirmation (w7 the handoff,
  the block's only record-walk publication, classified by consumption alone).
- **Additivity/source neutrality:** the slot-67 and slot-66 execute runs were
  re-taken under the slot-68 source (`s67/recheck-stage60`,
  `s66/recheck-stage60`; the pairings persisted beside the recheck runs as
  `pair.json`, and `s67/a`'s and `s66/a`'s retained a-vs-b `pair.json` are
  untouched) and pair **bit-identical** with the retained `s67/a` and
  `s66/a` (equal `run.json` SHA-256s, `02704ec7…` and `d0131a22…`), so the
  contract-68 entry changes no prior verdict.
- `python3 vr/native-slot-replay-probe.py self-test` (558 known answers / 39
  controls, `selftest-stage60.json`), `python3 vr/ladder-report.py
  self-test` (19 controls) and `python3 vr/ladder-names.py self-test` (14
  controls) all pass.

Runs on 21/10/2026 against the final source of the slot-67 stage (the
contract-67 entry and docstring precede the runs; every later touch is
self-test, comment or text only and proven run-neutral: see
the source-neutrality note below):

- `vr/logs/ladder/{s67,s67b}` — the slot-67 reference captures at the s67
  cut (validated this stage: cut accounting, pool digests, the slot-aware
  output canary, input digests;
  `ladder-report.py pair` enforces byte-equal pools with a fail-closed exit
  — its passing report is retained as `pair-s67-s67b.json`, pool digests
  `0xc1980769…/0x4e788235…/0xc98d82e7…` equal across runs, all three pool
  frames byte-identical, input digests equal to the earlier captures'). Traces
  `vr/logs/capture/nvapi-trace-ladder-s67{,b}.log` (the slot-67 parameter
  dump is stable across evaluations and identical between the runs; the
  capture sidecar's three records match it byte-for-byte).
- `vr/logs/ladder/diff-s66-s67.json`, `vr/logs/ladder/names-s66-s67.txt` —
  the measured slot-67 write-set (4 segments per frame, zero unattributed:
  391,543/391,442/391,443 changed bytes at w5 inside the 393,216-byte
  binding, the span [48,774,144, 49,167,360) covering the whole binding;
  196,545/196,547/196,536 at w2 inside the 196,608-byte binding
  [55,409,664, 55,606,272); and 64 B changed at each of w4
  [72,192, 72,256) and w7 [72,704, 72,768) — the whole publications).
- `vr/logs/slot-replay/s67/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s
  `02704ec7…`); the skip/wrong/tail signatures match the mode contracts
  (skip pool bit-identical to input and non-matching — diffs
  588,216/588,117/588,107 = the s66→s67 ladder counts, wrong changed from
  input and non-matching — 574,479–574,631 bytes differ from the reference
  with the weights record rotated inwards 1,024 B, tail output unchanged
  — bounded by the 1,050,624-byte record). The fp8 sibling's entries match
  the reference's output byte-for-byte with divergence exactly the 64-byte
  sync record in every frame (`sync_from_input` true) — the
  record-walk re-classification's sixth confirmation (w7 the handoff, w4
  output data).
- **Additivity/source neutrality:** the slot-66 and slot-65 execute runs were
  re-taken under the slot-67 source (`s66/recheck-stage59`,
  `s65/recheck-stage59`; the pairings persisted beside the recheck runs as
  `pair.json`, and `s66/a`'s and `s65/a`'s retained a-vs-b `pair.json` are
  untouched) and pair **bit-identical** with the retained `s66/a` and
  `s65/a` (equal `run.json` SHA-256s, `d0131a22…` and `922949f3…`), so the
  contract-67 entry changes no prior verdict.
- `python3 vr/native-slot-replay-probe.py self-test` (546 known answers / 39
  controls, `selftest-stage59.json`), `python3 vr/ladder-report.py
  self-test` (19 controls) and `python3 vr/ladder-names.py self-test` (14
  controls) all pass.

Runs on 21/10/2026 against the final source of the slot-66 stage (the
contract-66 entry and the wrong-control re-targeting precede the runs; every
later touch is self-test, comment or text only and proven run-neutral: see
the source-neutrality note below):

- `vr/logs/ladder/{s66,s66b}` — the slot-66 reference captures at the s66
  cut (validated this stage: cut accounting, pool digests, the slot-aware
  output canary, input digests;
  `ladder-report.py pair` enforces byte-equal pools with a fail-closed exit
  — its passing report is retained as `pair-s66-s66b.json`, pool digests
  `0xcd00572d…/0x96952417…/0x87c0f1b1…` equal across runs, all three pool
  frames byte-identical, input digests equal to the earlier captures'). Traces
  `vr/logs/capture/nvapi-trace-ladder-s66{,b}.log` (the slot-66 parameter
  dump is stable across evaluations and identical between the runs; the
  capture sidecar's three records match it byte-for-byte).
- `vr/logs/ladder/diff-s65-s66.json`, `vr/logs/ladder/names-s65-s66.txt` —
  the measured slot-66 write-set (2 segments per frame, zero unattributed:
  196,547/196,544/196,547 changed bytes at w3 inside the 196,608-byte
  binding, the span [55,213,056, 55,409,664) covering the whole binding,
  61/64/61 interior bytes static in the three frames; and 256/256/256 at
  w6 [71,680, 71,936) — the whole publication).
- `vr/logs/slot-replay/s66/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s
  `d0131a22…`); the skip/wrong/tail signatures match the mode contracts
  (skip pool bit-identical to input and non-matching — diffs
  196,803/196,800/196,803 = the s65→s66 ladder counts, wrong changed from
  input and non-matching — 381,748–382,857 bytes differ from the reference
  with the **query tensor at w0** rotated inwards (the wrong control
  re-bases there because the weights record is a 2-byte dead binding),
  tail output unchanged). The fp8 sibling's entries match the reference's
  output byte-for-byte with divergence exactly the 256-byte sync record in
  every frame (`sync_from_input` true) — the single-publication handoff
  prediction pinned a third time.
- `vr/logs/slot-replay/s66/wrong-record-rotation-experiment`,
  `…/wrong-complement-experiment` — the dead-binding proof at slot 66: with
  the wrong control aimed at the weights record (the query re-targeting
  removed for the run), both content-changing mutations — an inward
  rotation of the 2-byte record and whole-record complementation —
  reproduce every execute pool byte-for-byte (the chained entries' digests
  `0xcd00572d…/0x96952417…/0x87c0f1b1…` and the fp8 entries'
  `0x0f7b0853…/0xd90fd53d…/0xca3ba2d7…` identical to the execute run's),
  the wrong mode's expected `wrong-matched` enforcement failures retained
  as the invariance evidence. The weight-free attention family therefore
  reads nothing from its placeholder record at this block too.
- **Additivity/source neutrality:** the slot-65 and slot-64 execute runs were
  re-taken under the slot-66 source (`s65/recheck-stage58`,
  `s64/recheck-stage58`; the pairings persisted beside the recheck runs as
  `pair.json`, and `s65/a`'s and `s64/a`'s retained a-vs-b `pair.json` are
  untouched) and pair **bit-identical** with the retained `s65/a` and
  `s64/a` (equal `run.json` SHA-256s, `922949f3…` and `ab1fda10…`), so the
  contract-66 entry changes no prior verdict.
- `python3 vr/native-slot-replay-probe.py self-test` (534 known answers / 39
  controls, `selftest-stage58.json`), `python3 vr/ladder-report.py
  self-test` (19 controls) and `python3 vr/ladder-names.py self-test` (14
  controls) all pass.

Runs on 20/10/2026 against the final source of the slot-65 stage (the
contract-65 entry precedes the runs; every later touch is self-test,
comment or text only and proven run-neutral: see the source-neutrality
note below):

- `vr/logs/ladder/{s65,s65b}` — the slot-65 reference captures at the s65
  cut (validated this stage: cut accounting, pool digests, the slot-aware
  output canary, input digests;
  `ladder-report.py pair` enforces byte-equal pools with a fail-closed exit
  — its passing report is retained as `pair-s65-s65b.json`, pool digests
  `0x8fc0688f…/0x3ae8aef0…/0x8724a704…` equal across runs, all three pool
  frames byte-identical, input digests equal to the earlier captures'). Traces
  `vr/logs/capture/nvapi-trace-ladder-s65{,b}.log` (the slot-65 parameter
  dump is stable across evaluations and identical between the runs; the
  capture sidecar's three records match it byte-for-byte).
- `vr/logs/ladder/diff-s64-s65.json`, `vr/logs/ladder/names-s64-s65.txt` —
  the measured slot-65 write-set (6 segments per frame, zero unattributed:
  1,173,080/1,172,617/1,172,341 changed bytes at w6 inside the
  1,179,648-byte binding, the span [20,610,048, 21,789,696) covering the
  whole binding, 6,568/7,031/7,307 interior bytes static in the three
  frames; 196,552/196,517/196,546 at w3, 196,549/196,526/196,538 at w1 and
  196,218/196,171/196,181 at w2 inside the 196,608-byte q/k/v bindings, the
  spans [55,016,448, 55,213,056), [54,623,232, 54,819,840) and
  [54,819,840, 55,016,448) covering their whole bindings,
  56/91/62, 59/82/70 and 390/437/427 interior bytes static; and 128/128/128
  at w5 [70,656, 70,784) and 128/128/128 at w8 [71,168, 71,296) — the
  whole record-walk publications).
- `vr/logs/slot-replay/s65/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s
  `922949f3…`); the skip/wrong/tail signatures match the mode contracts
  (skip pool bit-identical to input and non-matching — diffs
  1,762,655/1,762,087/1,761,862 = the s64→s65 ladder counts, wrong changed
  from input and non-matching — 1,744,940–1,745,632 bytes differ from the
  reference with the `block32.layer2.layer` record rotated inwards 1,024 B
  — the record is live, so no wrong-control re-targeting is needed here —,
  tail output unchanged). The
  fp8 sibling's entries match the reference's q/k/v and fourth-region
  outputs byte-for-byte with divergence exactly the 128-byte sync record in
  every frame (`sync_from_input` true), confirming the two-publication
  handoff prediction (w8 the chain handoff — slot 66 binds its record as
  the `attention` block's w5 sync-in — and w5 variant-independent output
  data, the fifth sibling-shared publication).
- **Additivity/source neutrality:** the slot-64 and slot-63 execute runs were
  re-taken under the slot-65 source (`s64/recheck-stage57`,
  `s63/recheck-stage57`; the pairings persisted beside the recheck runs as
  `pair.json`, and `s64/a`'s and `s63/a`'s retained a-vs-b `pair.json` are
  untouched) and pair **bit-identical** with the retained `s64/a` and
  `s63/a` (equal `run.json` SHA-256s, `ab1fda10…` and `d0018066…`), so the
  contract-65 entry changes no prior verdict.
- `python3 vr/native-slot-replay-probe.py self-test` (522 known answers / 39
  controls, `selftest-stage57.json`), `python3 vr/ladder-report.py
  self-test` (19 controls) and `python3 vr/ladder-names.py self-test` (14
  controls) all pass.

Runs on 20/10/2026 against the final source of the slot-64 stage (the
contract-64 entry precedes the runs; every later touch is self-test,
comment or text only and proven run-neutral: see the source-neutrality
note below):

- `vr/logs/ladder/{s64,s64b}` — the slot-64 reference captures at the s64
  cut (validated this stage: cut accounting, pool digests, the slot-aware
  output canary, input digests;
  `ladder-report.py pair` enforces byte-equal pools with a fail-closed exit
  — its passing report is retained as `pair-s64-s64b.json`, pool digests
  `0x9e98165d…/0x3b417ae3…/0x354a666b…` equal across runs, all three pool
  frames byte-identical, input digests equal to the earlier captures'). Traces
  `vr/logs/capture/nvapi-trace-ladder-s64{,b}.log` (the slot-64 parameter
  dump is stable across evaluations and identical between the runs; the
  capture sidecar's three records match it byte-for-byte).
- `vr/logs/ladder/diff-s63-s64.json`, `vr/logs/ladder/names-s63-s64.txt` —
  the measured slot-64 write-set (4 segments per frame, zero unattributed:
  391,734/391,648/391,716 changed bytes at w5 inside the 393,216-byte
  binding, the span [49,953,792, 50,347,008) covering the whole binding,
  1,482/1,568/1,500 interior bytes static in the three frames;
  196,523/196,553/196,503 at w2 inside the 196,608-byte binding, the span
  [54,426,624, 54,623,232) covering the whole binding,
  85/55/105 interior bytes static; and 64/64/64 at w4 [69,632, 69,696) and
  64/64/64 at w7 [70,144, 70,208) — the whole record-walk publications).
- `vr/logs/slot-replay/s64/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s
  `ab1fda10…`); the skip/wrong/tail signatures match the mode contracts
  (skip pool bit-identical to input and non-matching — diffs
  588,385/588,329/588,347 = the s63→s64 ladder counts, wrong changed from
  input and non-matching — 576,052–576,607 bytes differ from the reference
  with the `block32.layer1.layer` record rotated inwards 1,024 B — the
  record is live, so no wrong-control re-targeting is needed here —, tail
  output unchanged). The
  fp8 sibling's entries match the reference's three output regions
  byte-for-byte with divergence exactly the 64-byte sync record in every
  frame (`sync_from_input` true), confirming the two-publication handoff
  prediction (w7 the chain handoff — slot 65 binds its record as the
  `qkv` block's w7 sync-in — and w4 variant-independent output data, the
  fourth sibling-shared publication).
- **Additivity/source neutrality:** the slot-63 and slot-62 execute runs were
  re-taken under the slot-64 source (`s63/recheck-stage56`,
  `s62/recheck-stage56`; the pairings persisted beside the recheck runs as
  `pair.json`, and `s63/a`'s and `s62/a`'s retained a-vs-b `pair.json` are
  untouched) and pair **bit-identical** with the retained `s63/a` and
  `s62/a` (equal `run.json` SHA-256s, `d0018066…` and `7c319752…`), so the
  contract-64 entry changes no prior verdict.
- `python3 vr/native-slot-replay-probe.py self-test` (510 known answers / 39
  controls, `selftest-stage56.json`), `python3 vr/ladder-report.py
  self-test` (19 controls) and `python3 vr/ladder-names.py self-test` (14
  controls) all pass.

Runs on 20/10/2026 against the final source of the slot-63 stage (the
contract-63 entry precedes the runs; every later touch is self-test,
comment or text only and proven run-neutral: see the source-neutrality
note below):

- `vr/logs/ladder/{s63,s63b}` — the slot-63 reference captures at the s63
  cut (validated this stage: cut accounting, pool digests, the slot-aware
  output canary, input digests;
  `ladder-report.py pair` enforces byte-equal pools with a fail-closed exit
  — its passing report is retained as `pair-s63-s63b.json`, pool digests
  `0x90a70c16…/0x6fdde73c…/0x5f388175…` equal across runs, all three pool
  frames byte-identical, input digests equal to the earlier captures'). Traces
  `vr/logs/capture/nvapi-trace-ladder-s63{,b}.log` (the slot-63 parameter
  dump is stable across evaluations and identical between the runs; the
  capture sidecar's three records match it byte-for-byte).
- `vr/logs/ladder/diff-s62-s63.json`, `vr/logs/ladder/names-s62-s63.txt` —
  the measured slot-63 write-set (2 segments per frame, zero unattributed:
  778,353/777,687/777,881 changed bytes at w2 inside the 786,432-byte
  binding, the span [35,159,040, 35,945,472) covering the whole binding,
  8,079/8,745/8,551 interior bytes static in the three frames, and
  256/256/256 at w7, the span [69,120, 69,376) — the whole publication).
- `vr/logs/slot-replay/s63/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s
  `d0018066…`); the skip/wrong/tail signatures match the mode contracts
  (skip pool bit-identical to input and non-matching — diffs
  778,609/777,943/778,137 = the s62→s63 ladder counts, wrong changed from
  input and non-matching — 730,753–733,258 bytes differ from the reference
  with the `block32.layer0.layer` record rotated inwards 1,024 B — the
  record is live, so no wrong-control re-targeting is needed here, tail
  output unchanged). The
  fp8 sibling's entries match the reference's output tensor
  byte-for-byte with divergence exactly the 256-byte sync record in every
  frame (`sync_from_input` true), confirming the single-publication handoff
  prediction (w7 the chain handoff — slot 64 binds its record as the
  `ffn_contract` block's w6 sync-in).
- **Additivity/source neutrality:** the slot-62 and slot-61 execute runs were
  re-taken under the slot-63 source (`s62/recheck-stage55`,
  `s61/recheck-stage55`; the pairings persisted beside the recheck runs as
  `pair.json`, and `s62/a`'s and `s61/a`'s retained a-vs-b `pair.json` are
  untouched) and pair **bit-identical** with the retained `s62/a` and
  `s61/a` (equal `run.json` SHA-256s, `7c319752…` and `77c696d2…`), so the
  contract-63 entry changes no prior verdict.
- `python3 vr/native-slot-replay-probe.py self-test` (498 known answers / 39
  controls, `selftest-stage55.json`), `python3 vr/ladder-report.py
  self-test` (19 controls) and `python3 vr/ladder-names.py self-test` (14
  controls) all pass.

Runs on 20/10/2026 against the final source of the slot-62 stage (the
contract-62 entry and its docstrings precede the runs; every later touch is
self-test, comment or text only and proven run-neutral: see
the source-neutrality note below):

- `vr/logs/ladder/{s62,s62b}` — the slot-62 reference captures at the s62
  cut (validated this stage: cut accounting, pool digests, the slot-aware
  output canary, input digests;
  `ladder-report.py pair` enforces byte-equal pools with a fail-closed exit
  — its passing report is retained as `pair-s62-s62b.json`, pool digests
  `0xac706dae…/0x9f66b274…/0xaa2755d4…` equal across runs, all three pool
  frames byte-identical, input digests equal to the s61 captures'). Traces
  `vr/logs/capture/nvapi-trace-ladder-s62{,b}.log` (the slot-62 parameter
  dump is stable across evaluations and identical between the runs; the
  capture sidecar's three records match it byte-for-byte).
- `vr/logs/ladder/diff-s61-s62.json`, `vr/logs/ladder/names-s61-s62.txt` —
  the measured slot-62 write-set (4 segments per frame, zero unattributed:
  391,486/391,452/391,349 changed bytes at w5 inside the 393,216-byte
  binding, the span [49,560,576, 49,953,792) covering the whole binding,
  196,500/196,524/196,534 at w2 inside the 196,608-byte binding, the span
  [54,230,016, 54,426,624) covering the whole binding, and 64/64/64 at each
  of w4, the span [68,096, 68,160), and w7, the span [68,608, 68,672) —
  the whole publications).
- `vr/logs/slot-replay/s62/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s
  `7c319752…`); the skip/wrong/tail signatures match the mode contracts
  (skip pool bit-identical to input and non-matching — diffs
  588,114/588,104/588,011 = the s61→s62 ladder counts, wrong changed from
  input and non-matching — 577,032–577,587 bytes differ from the reference
  with the `block31.layer4.layer` record rotated inwards 1,024 B, tail
  output unchanged). The
  fp8 sibling's entries match the reference's three output regions
  byte-for-byte with divergence exactly the 64-byte sync record in every
  frame (`sync_from_input` true), confirming the two-publication handoff
  prediction (w7 the chain handoff, w4 variant-independent output data).
- **Additivity/source neutrality:** the slot-61 and slot-60 execute runs were
  re-taken under the slot-62 source (`s61/recheck-stage54`,
  `s60/recheck-stage54`; the pairings persisted beside the recheck runs as
  `pair.json`, and `s61/a`'s and `s60/a`'s retained a-vs-b `pair.json` are
  untouched) and pair **bit-identical** with the retained `s61/a` and
  `s60/a` (equal `run.json` SHA-256s, `77c696d2…` and `d9dd3a21…`), so the
  contract-62 entry changes no prior verdict.
- `python3 vr/native-slot-replay-probe.py self-test` (486 known answers / 39
  controls, `selftest-stage54.json`), `python3 vr/ladder-report.py
  self-test` (19 controls) and `python3 vr/ladder-names.py self-test` (14
  controls) all pass.

Runs on 20/10/2026 against the final source of the slot-61 stage (the
contract-61 entry and the wrong-control re-targeting precede the runs; every
later touch is self-test, comment or text only and proven run-neutral: see
the source-neutrality note below):

- `vr/logs/ladder/{s61,s61b}` — the slot-61 reference captures at the s61
  cut (validated this stage: cut accounting, pool digests, the slot-aware
  output canary, input digests;
  `ladder-report.py pair` enforces byte-equal pools with a fail-closed exit
  — its passing report is retained as `pair-s61-s61b.json`, pool digests
  `0xc8919a73…/0x88306956…/0x47f1b9bf…` equal across runs, all three pool
  frames byte-identical, input digests equal to the s60 captures'). Traces
  `vr/logs/capture/nvapi-trace-ladder-s61{,b}.log` (the slot-61 parameter
  dump is stable across evaluations and identical between the runs; the
  capture sidecar's three records match it byte-for-byte).
- `vr/logs/ladder/diff-s60-s61.json`, `vr/logs/ladder/names-s60-s61.txt` —
  the measured slot-61 write-set (2 segments per frame, zero unattributed:
  196,535/196,524/196,496 changed bytes at w3 inside the 196,608-byte
  binding, the span [54,033,408, 54,230,016) covering the whole binding,
  73/84/112 interior bytes static in the three frames, and 256/256/256 at
  w6, the span [67,584, 67,840) — the whole publication).
- `vr/logs/slot-replay/s61/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s
  `77c696d2…`); the skip/wrong/tail signatures match the mode contracts
  (skip pool bit-identical to input and non-matching — diffs
  196,791/196,780/196,752 = the s60→s61 ladder counts, wrong changed from
  input and non-matching — 331,565–331,890 bytes differ from the reference
  with the query tensor at w0 rotated inwards, tail output unchanged). The
  fp8 sibling's entries match the reference's output tensor byte-for-byte
  with divergence exactly the 256-byte sync record in every frame
  (`sync_from_input` true), confirming the single-publication handoff
  prediction.
- **Dead binding (the stage's finding):** `block31.layer3.layer`'s 2-byte
  payload is not read by the kernel — the retained
  `s61/wrong-superseded-record-swap` run (the record rotated one byte) and
  `s61/wrong-complement-experiment` (every record byte complemented)
  reproduce the execute pools byte-for-byte in both variants (`pool_fnv`
  equal to `s61/a`'s per entry), so the wrong control re-bases onto the
  query tensor (`wrong_word` — the probe's first control re-targeting) and
  the pre-retarget runs are retained as
  `s61/{a,b,skip}-superseded-pre-retarget`.
- **Additivity/source neutrality:** the slot-60 and slot-59 execute runs were
  re-taken under the slot-61 source (`s60/recheck-stage53`,
  `s59/recheck-stage53`; the pairings persisted beside the recheck runs as
  `pair.json`, and `s60/a`'s and `s59/a`'s retained a-vs-b `pair.json` are
  untouched) and pair **bit-identical** with the retained `s60/a` and
  `s59/a` (equal `run.json` SHA-256s, `d9dd3a21…` and `b9b759b7…`), so the
  contract-61 entry and the control re-targeting change no prior verdict.
- `python3 vr/native-slot-replay-probe.py self-test` (474 known answers / 39
  controls, `selftest-stage53.json`), `python3 vr/ladder-report.py
  self-test` (19 controls) and `python3 vr/ladder-names.py self-test` (14
  controls) all pass.

Runs on 20/10/2026 against the final source of the slot-60 stage (the
contract-60 entry predates the runs; every later touch is self-test, comment
or text only and proven run-neutral: see the source-neutrality note below):

- `vr/logs/ladder/{s60,s60b}` — the slot-60 reference captures at the s60
  cut (validated this stage: cut accounting, pool digests, the slot-aware
  output canary, input digests;
  `ladder-report.py pair` enforces byte-equal pools with a fail-closed exit
  — its passing report is retained as `pair-s60-s60b.json`, pool digests
  `0xf0730266…/0x890912e5…/0x4a3a848c…` equal across runs, all three pool
  frames byte-identical, input digests equal to the s59 captures'). Traces
  `vr/logs/capture/nvapi-trace-ladder-s60{,b}.log` (the slot-60 parameter
  dump is stable across evaluations and identical between the runs; the
  capture sidecar's three records match it byte-for-byte).
- `vr/logs/ladder/diff-s59-s60.json`, `vr/logs/ladder/names-s59-s60.txt` —
  the measured slot-60 write-set (6 segments per frame, zero unattributed:
  1,172,971/1,172,523/1,172,573 changed bytes at w6 inside the
  1,179,648-byte binding, the span [33,192,960, 34,372,608) covering the
  whole binding,
  171,974/171,993/171,981 at w3 inside the 196,608-byte binding, the span
  [53,640,192, 53,836,798),
  171,959/171,944/171,950 at w1 (the span [53,443,584, 53,640,188)),
  171,661/171,667/171,669 at w2 (the span [53,836,800, 54,033,400)), plus
  128/128/128 at w5 (the span [66,560, 66,688)) and 128/128/128 at w8 (the
  span [67,072, 67,200))).
- `vr/logs/slot-replay/s60/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s
  `d9dd3a21…`); the skip/wrong/tail signatures match the mode contracts
  (skip pool bit-identical to input and non-matching — diffs
  1,688,821/1,688,383/1,688,429 = the s59→s60 ladder counts, wrong changed
  from input and non-matching — 1,539,675/1,539,533/1,539,728 bytes at the
  chained entries and 1,539,803/1,539,661/1,539,856 at the fp8 entries (the
  sibling's 128-byte sync divergence added) differ from the reference, the
  chained entries' first difference at the w6 region's base (33,192,960) and
  the fp8 entries' at the 128-byte sync record (67,072), tail output
  unchanged — bounded
  by the 3,145,856-byte weight record). The fp8 sibling's entries match the
  reference's five output regions byte-for-byte with divergence exactly the
  128-byte sync record in every frame (`sync_from_input` true — the record is
  left at the input state) while sharing the w5 publication with the live
  variant byte-for-byte (the variant finding's family rule confirmed).
- **Additivity/source neutrality:** the slot-59 and slot-58 execute runs
  were re-taken under the slot-60 source (`s59/recheck-stage52`,
  `s58/recheck-stage52`; the pairings persisted beside the recheck runs as
  `pair.json`, and `s59/a`'s and `s58/a`'s retained a-vs-b `pair.json` are
  untouched) and pair **bit-identical** with the retained `s59/a` and
  `s58/a` (equal `run.json` SHA-256s, `b9b759b7…` and `4fb03d2f…`), so the
  contract-60 entry changes no prior verdict.
- `python3 vr/native-slot-replay-probe.py self-test` (422 known answers / 34
  controls, `selftest-stage52.json`), `python3 vr/ladder-report.py
  self-test` (19 controls) and `python3 vr/ladder-names.py self-test` (14
  controls) all pass.

Runs on 19/10/2026 against the final source of the slot-59 stage (the
contract-59 entry predates the runs; every later touch is self-test, comment
or text only and proven run-neutral: see the source-neutrality note below):

- `vr/logs/ladder/{s59,s59b}` — the slot-59 reference captures at the s59
  cut (validated this stage: cut accounting, pool digests, the slot-aware
  output canary, input digests;
  `ladder-report.py pair` enforces byte-equal pools with a fail-closed exit
  — its passing report is retained as `pair-s59-s59b.json`, pool digests
  `0xf071b5c7…/0xf8e480d3…/0xbade8c88…` equal across runs, all three pool
  frames byte-identical, input digests equal to the s58 captures'). Traces
  `vr/logs/capture/nvapi-trace-ladder-s59{,b}.log` (the slot-59 parameter
  dump is stable across evaluations and identical between the runs; the
  capture sidecar's three records match it byte-for-byte).
- `vr/logs/ladder/diff-s58-s59.json`, `vr/logs/ladder/names-s58-s59.txt` —
  the measured slot-59 write-set (4 segments per frame, zero unattributed:
  342,690/342,703/342,652 changed bytes at w5 inside the 393,216-byte
  binding, the span [49,167,360, 49,560,572),
  171,959/171,966/171,970 at w2 inside the 196,608-byte binding, the span
  [53,246,976, 53,443,580), plus 64/64/64 at w4 (the span
  [65,536, 65,600)) and 64/64/64 at w7 (the span [66,048, 66,112))).
- `vr/logs/slot-replay/s59/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s
  `b9b759b7…`); the skip/wrong/tail signatures match the mode contracts
  (skip pool bit-identical to input and non-matching — diffs
  514,777/514,797/514,750 = the s58→s59 ladder counts, wrong changed from
  input and non-matching — 501,208/500,739/500,508 bytes at the chained
  entries and 501,272/500,803/500,572 at the fp8 entries (the sibling's
  64-byte sync divergence added) differ from the reference, the chained
  entries' first difference at the second output's base (49,167,360) and
  the fp8 entries' at the 64-byte sync record (66,048), tail output
  unchanged — bounded
  by the 4,196,352-byte weight record). The fp8 sibling's entries match the
  reference's three output tensors byte-for-byte with divergence exactly the
  64-byte sync record in every frame (`sync_from_input` true — the record is
  left at the input state) while sharing the w4 publication with the live
  variant byte-for-byte (the variant finding's re-classification).
- **Additivity/source neutrality:** the slot-58 and slot-57 execute runs
  were re-taken under the slot-59 source (`s58/recheck-stage51`,
  `s57/recheck-stage51`; the pairings persisted beside the recheck runs as
  `pair.json`, and `s58/a`'s and `s57/a`'s retained a-vs-b `pair.json` are
  untouched) and pair **bit-identical** with the retained `s58/a` and
  `s57/a` (equal `run.json` SHA-256s, `4fb03d2f…` and `6b75cafd…`), so the
  contract-59 entry changes no prior verdict.
- `python3 vr/native-slot-replay-probe.py self-test` (411 known answers / 34
  controls, `selftest-stage51.json`), `python3 vr/ladder-report.py
  self-test` (19 controls) and `python3 vr/ladder-names.py self-test` (14
  controls) all pass.

Runs on 19/10/2026 against the final source of the slot-58 stage (the
contract-58 entry predates the runs; every later touch is self-test, comment
or text only and proven run-neutral: see the source-neutrality note below):

- `vr/logs/ladder/{s58,s58b}` — the slot-58 reference captures at the s58
  cut (validated this stage: cut accounting, pool digests, the slot-aware
  output canary, input digests;
  `ladder-report.py pair` enforces byte-equal pools with a fail-closed exit
  — its passing report is retained as `pair-s58-s58b.json`, pool digests
  `0x1d6d5719…/0x28cc05a6…/0xf34cf6ba…` equal across runs, all three pool
  frames byte-identical, input digests equal to the s57 captures'). Traces
  `vr/logs/capture/nvapi-trace-ladder-s58{,b}.log` (the slot-58 parameter
  dump is stable across evaluations and identical between the runs; the
  capture sidecar's three records match it byte-for-byte).
- `vr/logs/ladder/diff-s57-s58.json`, `vr/logs/ladder/names-s57-s58.txt` —
  the measured slot-58 write-set (2 segments per frame, zero unattributed:
  779,794/779,827/779,876 changed bytes at w2 inside the 786,432-byte
  binding, the span [34,372,608, 35,159,040), plus 256/256/256 at w7, the
  span [65,024, 65,280)).
- `vr/logs/slot-replay/s58/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s
  `4fb03d2f…`); the skip/wrong/tail signatures match the mode contracts
  (skip pool bit-identical to input and non-matching — diffs
  780,050/780,083/780,132 = the s57→s58 ladder counts, wrong changed from
  input and non-matching — 637,015/636,844/636,967 bytes at the publish
  entries and 637,271/637,100/637,223 at the fp8 entries (the sibling's
  256-byte sync divergence added) differ from the reference, the publish
  entries' first difference at the output tensor's base (34,372,608) and
  the fp8 entries' at the 256-byte sync record (65,024), tail output
  unchanged — bounded
  by the 4,194,320-byte weight record). The fp8 sibling's entries match the
  reference's main tensor byte-for-byte with divergence exactly the 256-byte
  sync record in every frame (`sync_from_input` true — the record is left at
  the input state, the variant finding's shape at the publish family).
- **Additivity/source neutrality:** the slot-57 and slot-56 execute runs
  were re-taken under the slot-58 source (`s57/recheck-stage50`,
  `s56/recheck-stage50`; the pairings persisted beside the recheck runs as
  `pair.json`, and `s57/a`'s and `s56/a`'s retained a-vs-b `pair.json` are
  untouched) and pair **bit-identical** with the retained `s57/a` and
  `s56/a` (equal `run.json` SHA-256s, `6b75cafd…` and `323bfa2a…`), so the
  contract-58 entry changes no prior verdict.
- `python3 vr/native-slot-replay-probe.py self-test` (400 known answers / 34
  controls, `selftest-stage50.json`), `python3 vr/ladder-report.py
  self-test` (19 controls) and `python3 vr/ladder-names.py self-test` (14
  controls) all pass.

Runs on 19/10/2026 against the final source of the slot-57 stage (the
contract-57 entry, the single-variant / weights-less / sync-less contract
shapes, the input-tensor control set, the addressed-heap resolution and the
per-family kernel containers predate the runs; every later touch — usage
and docstring wording, one probe touch at 14:48 and the review's advisory
fixes — is self-test, comment or text only and proven run-neutral: see the
source-neutrality note below):

- `vr/logs/ladder/{s57,s57b}` — the slot-57 reference captures at the s57
  cut (validated this stage: cut accounting, pool digests, the slot-aware
  output canary, input digests;
  `ladder-report.py pair` enforces byte-equal pools with a fail-closed exit
  — its passing report is retained as `pair-s57-s57b.json`, pool digests
  `0x1dbc66c3…/0x374e1d48…/0x6e06afdb…` equal across runs, all three pool
  frames byte-identical). Traces
  `vr/logs/capture/nvapi-trace-ladder-s57{,b}.log` (the slot-57 parameter
  dump is stable across evaluations and identical between the runs; the
  capture sidecar's three records match it byte-for-byte).
- `vr/logs/ladder/diff-s56-s57.json`, `vr/logs/ladder/names-s56-s57.txt` —
  the measured slot-57 write-set (1 segment per frame, zero unattributed:
  171,969/171,970/171,962 changed bytes at w1 inside the 196,608-byte
  binding, the span [52,853,760, 53,050,364)); **no sync segment** — the
  parameter block carries no sync word at all.
- `vr/logs/slot-replay/s57/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s
  `6b75cafd…`); the skip/wrong/tail signatures match the mode contracts
  (skip pool bit-identical to input and non-matching — diffs
  171,969/171,970/171,962 = the s56→s57 ladder counts, wrong changed from
  input and non-matching — 343,184/343,038/342,948 bytes differ from the
  reference, first difference at the input tensor, tail output unchanged —
  the first tail control bounding input reads rather than weight reads). The
  contract is single-variant: no fp8 sibling replay applies and the
  whole-pool compare alone carries the claim.
- **Additivity/source neutrality:** the slot-56 and slot-55 execute runs
  were re-taken under the slot-57 source (`s56/recheck-stage49`,
  `s55/recheck-stage49`; the pairings persisted beside the recheck runs as
  `pair.json`) and pair **bit-identical** with the retained `s56/a` and
  `s55/a` (equal `run.json` SHA-256s, `323bfa2a…` and `fb5d5362…`), so the
  contract-57 additions and the three schema extensions change no prior
  verdict. One probe touch postdates the five s57 runs (source mtime
  14:48, after the last run at 14:39) and the review's advisory fixes
  postdate the rechecks; all are self-test, comment or text only and
  proven run-neutral: all five modes and both recheck executes were
  re-executed under the final source (`vr/logs/slot-replay/retake-stage49/…`
  — every take's `run.json` digest recorded in
  `retake-stage49/sha256sums.txt`) and every `run.json` reproduces
  byte-identically — `6b75cafd…` (a/b),
  `aeb17cbf…`/`ca2778b2…`/`5f2b43e7…` (skip/wrong/tail), `323bfa2a…` and
  `fb5d5362…` (the rechecks). The only self-test change is the review's
  added weights-word corruption control (33 → 34 corruption controls; 389
  known answers unchanged, `selftest-stage49.json` re-recorded).
- `python3 vr/native-slot-replay-probe.py self-test` (389 known answers / 34
  controls), `python3 vr/ladder-report.py self-test` (19 controls) and
  `python3 vr/ladder-names.py self-test` (14 controls) all pass.
- **Review round:** independent review PASS with three advisory findings —
  all three resolved: the weights-word count invariant tightened to reject
  a second weights word (its new contract-shape corruption control is the
  34th; 389 known answers unchanged), the `pure byte permutation`
  measurement backed by the retained
  `vr/logs/ladder/permutation-s56-s57.json` (the ambiguous "against zero
  pre-states" phrasing replaced by the measured zero-pre-state fact), and
  the re-execution digests persisted in
  `vr/logs/slot-replay/retake-stage49/sha256sums.txt`.

Runs on 19/10/2026 against the final source of the slot-56 stage (the
contract-56 entry, the `output_bytes` primary-extent generalisation, the
wait-family zero-sync handling, usage text and docstring bullets predate the
runs; no post-run probe change):

- `vr/logs/ladder/{s56,s56b}` — the slot-56 reference captures at the s56
  cut (the original interval-cut takes, pre-fix variants retained as
  `s56{,b}-superseded-pre-*`; validated this stage: cut accounting, pool
  digests, the slot-aware output canary, input digests;
  `ladder-report.py pair` enforces byte-equal pools with a fail-closed exit
  — its passing report is retained as `pair-s56-s56b.json`, pool digests
  `0x41aab191…/0xdd845176…/0x09293e91…` equal across runs, all three pool
  frames byte-identical). Traces
  `vr/logs/capture/nvapi-trace-ladder-s56{,b}.log` (the slot-56 parameter
  dump is stable across evaluations and identical between the runs; the
  capture sidecar's three records match it byte-for-byte).
- `vr/logs/ladder/diff-s55-s56.json`, `vr/logs/ladder/names-s55-s56.txt` —
  the measured slot-56 write-set (1 segment per frame, zero unattributed:
  171,969/171,970/171,962 changed bytes at w1 inside the 196,608-byte
  binding, the tail written zeros over unchanged zeros; and **no sync
  segment** — w3's record extent is byte-unchanged across the cut, the wait
  family's read-only handoff).
- `vr/logs/slot-replay/s56/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s
  `323bfa2a…`); the skip/wrong/tail signatures match the mode contracts
  (skip pool bit-identical to input and non-matching — diffs
  171,969/171,970/171,962 = the s55→s56 ladder counts, wrong changed from
  input and non-matching — 169,884–169,992 bytes differ from the reference,
  tail output unchanged beyond the 524,304-byte record). The fp8 sibling
  matches the reference over the **whole pool** on every entry — the first
  zero-divergence sibling, the wait family confirming that the variant pair
  differs only in sync-record handling.
- **Additivity/source neutrality:** the slot-55 execute run was re-taken
  under the slot-56 source (`s55/recheck-stage48`) and pairs
  **bit-identical** with the retained `s55/a` (equal `run.json` SHA-256s,
  `fb5d5362…` — the same digest as stage 47's retained pair, the pairing
  persisted beside the recheck run as `pair.json`), so the contract-56
  additions — the `output_bytes` primary-extent generalisation and the
  wait-family zero-sync handling — change no prior verdict.
- `python3 vr/native-slot-replay-probe.py self-test` (310 known answers / 26
  controls), `python3 vr/ladder-report.py self-test` (19 controls) and
  `python3 vr/ladder-names.py self-test` (14 controls) all pass.
- **Review round:** independent review FAIL → fixed (one pass). The pass
  found 2 must-fix — the "smallest parameter block of the inventory"
  over-claim (the 156-launch inventory carries 16-byte and 24-byte blocks;
  the claim is now scoped to the replayed contracts' blocks) and the
  recheck pairing's persistence place (the `pair` command persists beside
  its **first** path — the stage-48 pairing was re-taken recheck-first as
  `s55/recheck-stage48/pair.json`, and the stage-47 a↔b pairing it had
  overwritten at `s55/a/pair.json` restored byte-identically) — and 2
  advisory (the wait-branch `checks_enforced` wording, self-contradictory
  about "divergence matching the reference", and the vacuous sync verdict
  fields at `sync_bytes` 0 recorded as measurements; both resolved in the
  claims text, the vacuity now stated in the Variant finding). The five
  slot-56 runs and the slot-55 additivity recheck were re-taken under the
  final source: the s56 pair is bit-identical at `323bfa2a…` and the
  recheck still pairs with `s55/a` at `fb5d5362…`. Every numeric claim of
  the stage itself — the contract-56 word map, binding, output extent,
  write-set counts, pool and `run.json` digests and the control signatures
  — was verified correct against the retained evidence.

Runs on 19/10/2026 against the final source of the slot-55 stage (the
contract-55 entry, the `extra_outputs` region generalisation, usage text and
docstring bullets predate the runs; the only post-run probe change is the
review's advisory fix — the `main_first_diff` diagnostic semantics plus a
self-test shape check, no run-affecting behaviour — and both the slot-55
execute run and the slot-54 additivity recheck were re-taken under it,
pairing bit-identically with the retained runs):

- `vr/logs/ladder/{s55,s55b}` — the new slot-55 reference captures
  (validated: cut accounting, pool digests, the slot-aware output canary,
  input digests; `ladder-report.py pair` enforces byte-equal pools with a
  fail-closed exit — its passing report is retained as `pair-s55-s55b.json`,
  pool digests `0xd604dc3b…/0x1abe90f4…/0xa1ad1647…` equal across runs, all
  three pool frames byte-identical). Traces
  `vr/logs/capture/nvapi-trace-ladder-s55{,b}.log` (the slot-55 parameter
  dump is stable across evaluations and identical between the runs; the
  capture sidecar's three records match it byte-for-byte).
- `vr/logs/ladder/diff-s54-s55.json`, `vr/logs/ladder/names-s54-s55.txt` —
  the measured slot-55 write-set (3 segments per frame, zero unattributed:
  343,913/343,909/343,907 output tensor + 97,553/97,802/97,775 pooled
  tensor + 96 sync).
- `vr/logs/slot-replay/s55/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s
  `fb5d5362…`); the skip/wrong/tail signatures match the mode contracts
  (skip pool bit-identical to input and non-matching — diffs
  441,562/441,807/441,778 = the s54→s55 ladder counts, wrong changed from
  input and non-matching — 428,917–429,168 bytes differ from the reference,
  tail output unchanged beyond the 263,168-byte record). The fp8 sibling's
  two output tensors match the reference byte-for-byte on every entry and
  its divergence is exactly the 96-byte sync record (`sync_from_input`).
- **Additivity/source neutrality:** the slot-54 execute run was re-taken
  under the slot-55 source (`s54/recheck-stage47`) and pairs
  **bit-identical** with the retained `s54/a` (equal `run.json` SHA-256s,
  `f9dbd8d2…` — the same digest as stage 46's retained pair, the pairing
  persisted beside the recheck run as `pair.json`), so the contract-55
  addition — including the multi-region output-verdict generalisation —
  changes no prior verdict. After the review's advisory fix both takes were
  repeated under the final source: `s55/recheck-stage47` pairs
  **bit-identical** with the retained `s55/a` (`fb5d5362…`) and the re-taken
  `s54/recheck-stage47` with `s54/a` (`f9dbd8d2…`), the pre-advisory take
  retained as `s54/recheck-stage47-superseded-pre-advisory-fix`.
- **Review round:** independent review FAIL → fixed (one pass). The pass
  found 3 must-fix, all stale documentation enumerations — the weight-bytes
  source row's record list two records behind (`block30.layer2.layer`,
  `block30.layer3.layer` missing), the parameter-block source row missing
  the 80 B form, and the Result/Scope kernel-name breakdowns enumerating
  only five of the seven names (the single-use `inpview` pair, slots 24/25,
  unlisted) — and 1 advisory (the multi-region `main_first_diff` diagnostic,
  resolved as the primary output region's own region-relative offset;
  duplicate extra-output words are now rejected by the contract-shape
  check). Every numeric claim of the stage itself — the contract-55 word
  map, binding, sync offsets and extent, write-set counts, pool and
  `run.json` digests, the control signatures and the slot-56 decode — was
  verified correct against the retained evidence. The fixes are
  `vr/NATIVE-SLOT-REPLAY.md` prose plus the probe's diagnostic semantic and
  self-test shape check (self-tests unchanged: 269 known answers / 25
  controls), re-verified bit-identical as above.
- `python3 vr/native-slot-replay-probe.py self-test` (269 known answers),
  `python3 vr/ladder-report.py self-test` (19 controls) and `python3
  vr/ladder-names.py self-test` (14 controls) all pass.

Runs on 18/10/2026 against the final source of the slot-54 stage (the
contract-54 entry, usage text and docstring bullets predate the runs; no
post-run probe-source change):

- `vr/logs/ladder/{s54,s54b}` — the new slot-54 reference captures
  (validated: cut accounting, pool digests, the slot-aware output canary,
  input digests; `ladder-report.py pair` enforces byte-equal pools with a
  fail-closed exit — its passing report is retained as `pair-s54-s54b.json`,
  pool digests `0x38b23dce…/0x826ac349…/0xfabd133c…` equal across runs, all
  three pool frames byte-identical). Traces
  `vr/logs/capture/nvapi-trace-ladder-s54{,b}.log` (the slot-54 parameter
  dump is stable across evaluations and identical between the runs; the
  capture sidecar's three records match it byte-for-byte).
- `vr/logs/ladder/diff-s53-s54.json`, `vr/logs/ladder/names-s53-s54.txt` —
  the measured slot-54 write-set (2 segments per frame, zero unattributed:
  340,788/341,447/341,397 tensor + 256 sync).
- `vr/logs/slot-replay/s54/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s
  `f9dbd8d2…`); the skip/wrong/tail signatures match the mode contracts
  (skip pool bit-identical to input and non-matching — diffs
  341,044/341,703/341,653 = the s53→s54 ladder counts, wrong changed from
  input and non-matching — 340,577–341,053 bytes differ from the reference,
  tail output unchanged beyond the record).
- **Additivity/source neutrality:** the slot-53 execute run was re-taken
  under the slot-54 source (`s53/recheck-stage46`) and pairs
  **bit-identical** with the retained `s53/a` (equal `run.json` SHA-256s,
  `9154b1ce…` — the same digest as stage 45's retained pair). The pairing is
  persisted beside the recheck run as `pair.json` and `s53/a`'s a-vs-b
  `pair.json` regenerated beside it with the same verdict and digests (the
  first take landed beside `s53/a` — `pair` writes beside its FIRST
  argument — and both pairings are now in their established places), so the
  contract-54 addition changes no prior verdict.
- **Review round:** independent review FAIL → fixed (one pass). The pass
  found 8 must-fix — the recheck pairing's persistence place (both pairings
  re-taken in their established places), the Method section's stale
  `--slot {24..53}` usage range and `s{24..53}` evidence path, the
  weight-heap section's unextended word-anchor list and record walk (slot
  54's word 2 / `block30.layer2.layer` at 21,322,240), the heap-scan
  coverage and binding-identity anchors left at their slot-53 values, the
  Scope read-set enumeration missing slot 54's w0/w5 reads, and the Next
  section's remaining-work range left at (54–56) — and 1 advisory (a
  dangling `block29` fragment left by the cycle edit, dropped). Every
  numeric claim of the stage itself — the contract-54 word map, binding,
  sync offsets and extent, write-set counts, pool and `run.json` digests,
  the control signatures and the slot-55/56 decode — was verified correct
  against the retained evidence. The fixes are `vr/NATIVE-SLOT-REPLAY.md`
  prose only (the `pair` re-takes re-established existing verdicts) — no
  functional change — so every retained verdict stands unchanged (the
  stage's runs were taken against the functional probe source and the
  slot-53 additivity recheck pairs `9154b1ce…` under it).
- `python3 vr/native-slot-replay-probe.py self-test` (261 known answers),
  `python3 vr/ladder-report.py self-test` (19 controls) and `python3
  vr/ladder-names.py self-test` (14 controls) all pass.

Runs on 18/10/2026 against the final source of the slot-53 stage (the
contract-53 entry, usage text and docstring bullets predate the runs; no
post-run probe-source change):

- `vr/logs/ladder/{s53,s53b}` — the new slot-53 reference captures
  (validated: cut accounting, pool digests, the slot-aware output canary,
  input digests; `ladder-report.py pair` enforces byte-equal pools with a
  fail-closed exit — its passing report is retained as `pair-s53-s53b.json`,
  pool digests `0x9bf657c2…/0xd36d15c2…/0x5f1ef454…` equal across runs, all
  three pool frames byte-identical). Traces
  `vr/logs/capture/nvapi-trace-ladder-s53{,b}.log` (the slot-53 parameter
  dump is stable across evaluations and identical between the runs; the
  capture sidecar's three records match it byte-for-byte).
- `vr/logs/ladder/diff-s52-s53.json`, `vr/logs/ladder/names-s52-s53.txt` —
  the measured slot-53 write-set (2 segments per frame, zero unattributed:
  340,285/340,534/340,538 tensor + 96 sync).
- `vr/logs/slot-replay/s53/…` — the five replay runs plus the persisted pair
  result (`a/pair.json`, bit-identical, equal `run.json` SHA-256s
  `9154b1ce…`); the skip/wrong/tail signatures match the mode contracts
  (skip pool bit-identical to input and non-matching — diffs
  340,381/340,630/340,634 = the s52→s53 ladder counts, wrong changed from
  input and non-matching — 342,329–342,499 bytes differ, tail output
  unchanged beyond the record).
- **Additivity/source neutrality:** the slot-52 execute run was re-taken
  under the slot-53 source (`s52/recheck-stage45`) and pairs
  **bit-identical** with the retained `s52/a` (equal `run.json` SHA-256s,
  `c6e24122…` — the same digest as stage 44's retained pair, the pairing
  persisted beside the recheck run as `pair.json`, and `s52/a`'s retained
  a-vs-b `pair.json` is untouched), so the contract-53 addition changes no
  prior verdict.
- **Review round:** independent review PASS with 1 advisory → fixed (one
  pass). The advisory — the weight-bytes source row listing the record
  walk's unbound interloper `block3.layer0.layer` beside the thirty
  contract-bound records, readable as a thirty-first extracted record — is
  fixed here by annotating the entry (the interloper is not replayed as
  weights; its bytes are never extracted). Every numeric claim, the
  contract-table shape, all enumerations and the self-test accounting were
  verified correct against the retained evidence, so every retained verdict
  stands unchanged (the fix is `vr/NATIVE-SLOT-REPLAY.md` prose only; the
  stage's runs were taken against the functional probe source and the
  slot-52 additivity recheck pairs `c6e24122…` under it).
- `python3 vr/native-slot-replay-probe.py self-test` (253 known answers),
  `python3 vr/ladder-report.py self-test` (19 controls) and `python3
  vr/ladder-names.py self-test` (14 controls) all pass.

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

- The claim covers **fifty-one launches of the 156** (slots 24–74; fifteen distinct
  kernel names, one shared across two geometries (26/30/34/38/42/46/50/54), three reused at
  the same geometry (27/31/35/39/43/47/51, 28/32/36/40/44/48/52, 29/33/37/41/45/49/53), two
  single-use `inpview` (24/25), two
  the `proj_pool` and `final_head` variants' own (55/56), one single-use
  `repack` (57), one single-use `ffn_expand` `publish` variant (58), the
  `ffn_contract` `chained` variant (59/64/69/74), the `qkv` `chained` variant
  (60/65/70), the `attention` `chained` variant (61/66/71), the `projection`
  `chained` variant (62/67/72) and the `ffn_expand` `chained` variant (63/68/73)), one resolution
  (720×720, eye0), three frames under the stationary-input protocol. The
  method is generic — the parameter block, geometry, word classes and weight
  record are all contract fields now (with single-variant, weights-less and
  sync-less shapes) — but each remaining kernel still needs
  its own reference capture at sN+1 before it can be replayed.
- The tail control bounds weight reads to the record **for these kernels**
  (at the weights-less slot 57 it bounds input reads to the input tensor
  instead, the pattern staged at the output extent's pre-launch bytes —
  the only pool state the kernel rewrites);
  read-only out-of-bounds accesses beyond the guard window are not detected
  by canaries (same caveat as earlier stages).
- The `wrong`-mode rotation follows the decoder record's 1,024-byte
  alternative-split convention; no split semantics are claimed for these
  records (the rotation targets the input tensor at four contracts: slot 57,
  the weights-less contract's computation input, and slots 61, 66 and 71, whose
  2-byte placeholder records are dead bindings — the query tensor at w0
  carries the sensitivity claim there).
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
  slot 52's w0 and w4 reads (output of slot 51, previous sync),
  slot 53's w0, w1 and w7 reads (output of slot 52, residual, previous
  sync),
  slot 54's w0 and w5 reads (output of slot 53, previous sync),
  slot 55's w0, w1 and w5 reads (output of slot 54, residual, previous
  sync),
  slot 56's w0 and w3 reads (pooled output of slot 55, previous sync),
  slot 57's w0 read (the final-head tensor),
  slot 58's w0 read (the repacked head tensor),
  slot 59's w0, w1 and w6 reads (fp8 expansion of slot 58, residual head
  tensor of slot 57, previous sync),
  slot 60's w0 and w7 reads (the qkv input — slot 59's contraction output
  — and the previous sync record),
  slot 61's w0, w1, w2 and w5 reads (the q/k/v triplet of slot 60 and the
  previous sync record),
  slot 62's w0, w1 and w6 reads (the attention output of slot 61, the
  residual contraction output of slot 59 and the previous sync record),
  slot 63's w0 and w6 reads (the projection output of slot 62 and the
  previous sync record),
  slot 64's w0, w1 and w6 reads (the expansion output of slot 63, the
  residual projection output of slot 62 and the previous sync record),
  slot 65's w0 and w7 reads (the contraction output of slot 64 and the
  previous sync record), slot 66's w0, w1, w2 and w5 reads (the q/k/v
  triplet of slot 65 and the previous sync record), slot 67's w0, w1, w3
  and w6 reads (the attention output of slot 66, the residual contraction
  output of slot 64, the weights record and the previous sync record),
  slot 68's w0, w3 and w6 reads (the projection output of slot 67, the
  weights record and the previous sync record), slot 69's w0, w1, w3 and
  w6 reads (the expansion output of slot 68, the projection output of
  slot 67 as the residual skip, the weights record and the previous sync
  record), slot 70's w0, w4 and w7 reads (the contraction output of
  slot 69, the weights record and the previous sync record), slot 71's
  w0, w1, w2, w4 and w5 reads (the q/k/v triplet of slot 70, the weights
  record and the previous sync record), slot 72's
  w0, w1, w3 and w6 reads (the attention output of slot 71, the residual
  = slot 69's w2, the weights record and the previous sync record), and
  slot 73's w0, w3 and w6 reads (the projection output of slot 72, the
  weights record and the previous sync record), and slot 74's w0, w1, w3
  and w6 reads (the expansion output of slot 73, the residual from slot 72,
  the weights record and the previous sync record); slots 28/29's w0 and (w4/w7) reads — slot 29
  additionally its residual w1 — are inferred from the attribution (those words never change) and the
  word map, not from a measured read-set.
- The sync extents (96/96/192/96/96/96/256/96/96/96/192/96/96/96/256/96/96/96/192/96/96/96/256/96/96/96/192/96/96/96/256/96/0/0/256/64/128/256/64/256/64/128/256/64/256/64/128/256/64/256/64 B for slots 24–74) are the records'
  **changed** extents under
  this protocol; records whose unchanged bytes extend further are not
  excluded (slot 56's 0 is the wait family's read-only handoff; slot 57's is
  the empty compare — its block carries no sync word; slot 58's 256 is the
  ViT chain's first record, written whole by the publish family; slot 59's
  64 is the w7 handoff alone — its w4 record-walk publication is
  variant-independent output data, not a sync record; slot 60's 128 is the
  w8 handoff alone — its w5 record-walk publication is likewise output
  data, the re-classification now confirmed twice; slot 61's 256 is its
  single w6 publication — the first block with one record-walk publication,
  classified as the chain handoff by consumption alone; slot 62's 64 is the
  w7 handoff alone — its w4 record-walk publication is likewise output
  data, the re-classification now confirmed three times; slot 63's 256 is
  its single w7 publication — classified as the chain handoff by
  consumption alone (slot 64 binds the record), the single-publication rule
  now confirmed twice; slot 64's 64 is the
  w7 handoff alone — its w4 record-walk publication is likewise output
  data, the re-classification now confirmed four times; slot 65's 128 is the
  w8 handoff alone — its w5 record-walk publication is likewise output
  data, the re-classification now confirmed five times; slot 66's 256 is the
  w6 handoff alone — the block's single publication, classified by
  consumption as at slots 61 and 63, with no sibling-shared publication at
  this contract; slot 67's 64 is the
  w7 handoff alone — its w4 record-walk publication is likewise output
  data, the re-classification now confirmed six times; slot 68's 256 is
  its single w7 publication — classified by consumption as at slots 61, 63
  and 66, the single-publication rule now confirmed four times; slot 69's
  64 is the
  w7 handoff alone — its w4 record-walk publication is likewise output
  data, the re-classification now confirmed seven times; slot 70's
  128 is the
  w8 handoff alone — its w5 record-walk publication is likewise output
  data, the re-classification now confirmed eight times; slot 71's
  256 is the
  w6 handoff alone — the attention family's third single-publication
  block, no sibling-shared publication at this contract; slot 72's
  64 is the
  w7 handoff alone — its w4 record-walk publication is likewise output
  data, the re-classification now confirmed nine times; slot 73's 256 is the
   single w7 publication, classified by consumption as the chain handoff
   because slot 74 binds it as w6 sync-in, the predicted 256-byte sibling
   divergence — confirmed by the sibling replay; slot 74's 64-byte w7 is
   the next handoff candidate, awaiting confirmation by slot 75).
- This stage uses stored learned weight bytes through fifty extracted records
  (slots 24–56, 58–74; slot 57 binds none — the first weights-less contract;
  slots 61, 66 and 71 bind 2-byte attention placeholders that are read by no
  variant — the dead-binding finding, measured three times). This is
  bounded model-weight computation ( [NATIVE-FP8.md](NATIVE-FP8.md)
  precedent) and is **not** a neural-rendered image. Proprietary
  reuse/publication rights remain unresolved; evidence retains digests and
  bounded 8 KiB windows (written only on mismatch) and stays local.

## Next

Slot 74 is replayed; continue with slot 75. Repeat the replay for the remaining named kernels of
[LADDER-NAMES.md](LADDER-NAMES.md) — the encoder (slots 1–23), the
ViT
stage (75–98), transformer-2 (100–131) and the decoder pyramid (132–153);
slot 99 (the decoder) is already covered by
[NATIVE-VENDOR-COMPARE.md](NATIVE-VENDOR-COMPARE.md) and slot 154 (the final
blend) writes only outside the working pool. Each remaining kernel needs a
reference capture at its
own cut point (`run-feature-probe.sh ladder sN N`) plus one contract entry,
and occasionally a small probe generalisation (slot 26 needed per-slot grid
resolution and an alignment-aware record lookup; slot 56 needed the
`output_bytes` primary-extent field and the wait family's zero-sync
handling; slot 57 needed the single-variant, weights-less and sync-less
contract shapes with the input-tensor control set, the addressed-heap
resolution for weights-less blocks and per-family kernel containers; slot
58 needed no probe generalisation at all — its shape composes the existing
fields, a new word map and the ViT container's second record; slot 59
likewise needed none — its three outputs sit in the existing `extra_outputs`
list and the re-classified w4 publication is an output word, not a new
sync shape; slot 60 again needed none — its five outputs compose
`output_bytes` plus four `extra_outputs` entries and the re-classified w5
publication is an output word like slot 59's w4; slot 61 needed the
wrong-control **re-targeting** (`wrong_word`/`wrong_bytes` — its weights
record is a 2-byte dead binding, so the wrong rotation targets the query
tensor at w0 instead of the record) plus the tiny-record rotation extent;
slot 62 needed none — its three outputs and two record-walk publications
compose the slot-59/60 shapes exactly, the re-classification rule
resolving w7 as the chain handoff and w4 as output data; slot 63 needed
none either — its expansion shape composes slot 58's word map with the
sync-in bound and a new weight record, and its single publication
resolves by consumption alone; slot 64 needed none either — its
contraction shape is slot 59's word map exactly (the same (12, 16)
constant at w8), the two-publication re-classification confirmed a fourth
time; slot 65 needed none either — it shares slot 60's ten-word qkv shape
exactly (the same (12, 16) constant at w9), the two-publication
re-classification confirmed a fifth time; slot 66 needed none either — it
shares slot 61's eight-word attention shape exactly (the same (12, 16)
constant at w7), its single publication resolving by consumption as at
slot 61, and the dead-binding wrong-control re-targeting repeats at the
family's placeholder record (`wrong_word` — the query at w0 again);
slot 67 needed none either — it shares slot 62's nine-word projection
shape exactly (the same (12, 16) constant at w8, the same three output
regions and two record-walk publications), the two-publication
re-classification confirmed a sixth time; slot 68 needed none either — it
shares slot 63's nine-word expansion shape exactly (the same (12, 16)
constant at w8, the same single publication), the single-publication rule
confirmed a fourth time; slot 69 needed none either — it
shares slot 59's nine-word contraction shape exactly (the same (12, 16)
constant at w8, the same two-publication split), the re-classification
confirmed a seventh time; slot 70 needed none either — it
shares slot 60's ten-word qkv shape exactly (the same (12, 16)
constant at w9, the same two-publication split), the re-classification
confirmed an eighth time; slot 71 needed none either — it
shares slot 61's eight-word attention shape exactly (the same (12, 16)
constant at w7, the same single-publication handoff), the family's third
dead-binding re-targeting; slot 72 needed none either — it
shares slot 62's nine-word projection shape exactly (the same (12, 16)
constant at w8, the same three output regions and two record-walk
publications), the two-publication re-classification confirmed a ninth
time). Slots
24–72 cover the complete transformer interval (24–56) plus the ViT
interval's opening repack and first block's complete five-slot cycle
(57–62) plus the second block's complete five-slot cycle (63–67) and the
third block's complete five-slot cycle (68–72) —
eight complete `ffwd→proj` pairs (block23: slots 24/25, block24: 28/29,
block25: 32/33, block26: 36/37, block27: 40/41, block28: 44/45, block29:
48/49, block30: 52/53) and eight complete `qkv`-proj
pairs (block23: slots 26/27, block24: 30/31, block25: 34/35, block26: 38/39,
block27: 42/43, block28: 46/47, block29: 50/51 and block30: 54/55 — block30's
second half in the `proj_pool` variant, a distinct ten-word kernel rather
than block23's `proj` shape), so
the pair pattern, the chained sync handoff (with its
per-slot word position: w5 at slots 26/30/34/38/42/46/50/54/55, w7 at slots 25/27/29/31/33/35/37/39/41/43/45/47/49/51/53, w4 at
slots 28/32/36/40/44/48/52, w6 at
slot 59 (consumed at w6, published at w7), w7 at
slot 60 (consumed at w7, published at w8), w5 at
slot 61 (consumed at w5, published at w6), w6 at
slot 62 (consumed at w6, published at w7), w6 at
slot 63 (consumed at w6, published at w7), w6 at
slot 64 (consumed at w6, published at w7), w7 at
slot 65 (consumed at w7, published at w8), w5 at
slot 66 (consumed at w5, published at w6), w6 at
slot 67 (consumed at w6, published at w7), w6 at
slot 68 (consumed at w6, published at w7), w6 at
slot 69 (consumed at w6, published at w7), w7 at
slot 70 (consumed at w7, published at w8), w5 at
slot 71 (consumed at w5, published at w6), w6 at
slot 72 (consumed at w6, published at w7), w3 at
slot 56 — the wait family's read-only handoff)
and the record-ordering pattern are established: each block repeats
block23's layer0-3 record shape in declaration order, so each remaining
slot's weight
record is predictable from its parameter block's weights word plus the
layout walk — and slots 29's, 30's, 31's, 32's, 33's, 34's, 35's, 36's, 37's,
38's, 39's, 40's, 41's, 42's, 43's, 44's, 45's, 46's, 47's, 48's, 49's, 50's,
51's, 52's, 53's, 54's, 55's and 56's
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
20,534,784, `block30.layer1.layer` at
21,059,072, `block30.layer2.layer` at 21,322,240, `block30.layer3.layer` at 22,240,256 and `block30.layer4.layer` at
22,503,424 — block30's fifth record, the final head's weights)
— slot 52's landing one interloper past the block23-style shape
(`block3.layer0.layer`, 20,672 B at 20,513,792, declared between
`block29.layer3.layer` and `block30.layer0.layer`, so the "each block
repeats block23's layer0-3 shape" claim now carries that one-record
exception), the weights-word-plus-layout-walk prediction still exact. The qkv→proj pair's
second constant is now measured in both halves of all six fully-measured
carrying pairs
(block24
at `0xFFFFFFFCFFFFFFFC`, block25 at `0x00000000FFFFFFFC`, block26 at
`0xFFFFFFFC00000000` — slot 38's w4 and slot 39's w8 alike — block28 at
`0xFFFFFFFCFFFFFFFC`, slot 46's w4 and slot 47's w8 alike, block29 at
`0x00000000FFFFFFFC`, slot 50's w4 and slot 51's w8 alike, and block30 at
`0xFFFFFFFC00000000`, slot 54's w4 and slot 55's w7 alike);
it is absent from block23's pair (both halves measured), every `ffwd→proj`
half and block27's pair (slot 42's w4 and slot 43's w8 both measured
reserved zero) — one value per
pair where present, shared by both halves wherever measured (block24,
block25, block26, block28, block29 and block30). The carrying pairs'
values are the double, low-dword-only and high-dword-only forms of the same
marker dword (`0xFFFFFFFC`) on a repeating absent/double/low/high cycle
(block23 absent, block24 double, block25 low, block26 high, block27 absent,
block28 double, block29 low — now measured in both halves — and block30
high, the second turn's closing step the stage-45 note predicted, now
measured in both halves of the pair — slot 54's w4, slot 55's w7).
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
The
slot-53 `ffwd_proj` half (block30's,
`cc_split_swin_16h_ffwd_proj_512_chained_fp8` at grid (8,3,1), block
(32,4,1)) is now replayed byte-for-byte and confirms every word-map,
record-binding, sync-extent and constant prediction. Its parameter block
(psize 72) is the slot-25/29/33/37/41/45/49 `ffwd_proj` word map exactly
(POOL, POOL, POOL, WEIGHTS, CONSTANT, ZERO, POOL, POOL, ZERO): w0 input
(34,618,368 — address-equal to slot 52's w1), w1 residual (34,274,304 —
the buffer slot 51 wrote its output to, the input slot 52 itself read), w2
output (34,962,432 — one stride past its input, the regular output
spacing), w4 the (24, 28) geometry constant `0x1C00000018`, **w5/w8
reserved zero** (the `ffwd→proj` pair carries no second constant), w6
sync-out (pool+63,488 — the 512-byte stride continuing: 61,952 → 62,464 →
62,976 → 63,488) and w7 sync-in (pool+62,976 — address-equal to slot 52's
w6, the chained handoff). Its weights word lands on the layout walk's next
record start (`block30.layer1.layer`, 263,168 B at heap offset 21,059,072 —
20,534,784 + 524,288, block30's layer0 closing exactly at the 512 boundary
so no padding intervenes), confirmed by the probe's live DLL walk. The sync
extent measured **96 B** (the proj norm), pinned from the s52→s53 write-set
(two segments per frame, zero unattributed:
340,285/340,534/340,538 tensor + 96 sync) and enforced as the sibling's
divergence (the fp8 control's pool differs from the reference by exactly 96
bytes on all three frames). The execute pair is bit-identical (run_sha256
`9154b1ce…`).
The
slot 54 `qkv` half (block30's `qkv→proj` pair's first half at grid (4,4,4),
the slot-30/38/46 geometry) is now
replayed byte-for-byte and confirms every word-map, record-binding,
sync-extent and constant prediction. Its parameter block
(psize 56) is the slot-26/30/34/38/42/46/50 qkv word map exactly (pool,
pool, weights, constant, constant, pool, pool): w0 input (34,962,432 —
address-equal to slot 53's w2), w1 output (35,306,496 — one stride past its
input, the regular output spacing), w2 weights, w3 the (24, 28) geometry
constant, w4 the pair's second constant at `0xFFFFFFFC00000000` — the
high-dword form the stage-45 note predicted (blocks 23–30: absent / double /
low / high / absent / double / low / high), w5 sync-in (63,488 —
address-equal to slot 53's w6, the chained handoff) and w6 sync-out (64,000
— the 512-byte stride continuing: 62,464 -> 62,976 -> 63,488 -> 64,000).
Its weights word lands on the layout walk's next record start
(`block30.layer2.layer`, 917,568 B at heap offset 21,322,240 — 21,059,072 +
263,168 at the 512 boundary), confirmed by the probe's live DLL walk. The
sync extent measured **256 B** (span 64,000 -> 64,256), the qkv 192/256
alternation continuing (26/34/42/50 at 192, 30/38/46/54 at 256), pinned
from the s53→s54 write-set (two segments per frame, zero unattributed:
340,788/341,447/341,397 tensor + 256 sync) and enforced as the sibling's
divergence. The execute pair is bit-identical (run_sha256 `f9dbd8d2…`).
The
natural next candidate is slot 55 (`cc_split_swin_16h_proj_pool_512_chained_fp8`),
where the interval's tail departs from the four-kernel block shape; its
parameter block is 80 bytes (10 words — decoded from the s54 capture's own
retained trace, to be verified at its stage) with w0 input (35,306,496 —
address-equal to slot 54's w1), w1 residual (34,962,432 — the buffer slot
53 wrote its output to), a second output word w2 (52,313,088) and a fourth
pool word w3 (26,803,200) — the `proj_pool` name's pool expansion — w4
weights on the layout walk's next record start (`block30.layer3.layer`,
263,168 B at heap offset 22,240,256 — 21,322,240 + 917,568 past the qkv
record's 448-byte tail padding to the 512 boundary), w5 sync-in (64,000 —
address-equal to slot 54's w6) and w6 sync-out (64,512, the stride
continuing); w7 carries the pair's second constant at
`0xFFFFFFFC00000000` — block30's high-dword step now seen at the proj
half's own word position too — w8 the (24, 28) constant and w9 a further
constant `0x000000100000000C` (dword pair (12, 16)). Slot 56 is then the
`final_head` (`cc_split_swin_16h_final_head_512_wait_fp8`, 40-byte block,
grid (8,2,1)) closing the transformer interval at the s56 cut. The
temporal/history state (`prev_output`, counter) then connects the kernels
into a full native evaluation for the native-vs-vendor comparison of the
whole pipeline.

The
slot 55 `proj_pool` half (block30's `qkv`-proj pair's second half in the
`proj_pool` variant, `cc_split_swin_16h_proj_pool_512_chained_fp8` at grid
(8,3,1), block (32,4,1)) is now replayed byte-for-byte and confirms every
word-map, record-binding, write-set and constant prediction. Its parameter
block (psize 80) is the first ten-word block (pool, pool, pool, pool,
weights, pool, pool, constant, constant, constant): w0 input (35,306,496 —
address-equal to slot 54's w1), w1 residual (34,962,432 — the buffer slot
53 wrote its output to, the input slot 54 itself read), w2 output
(52,313,088 — 344,064 bytes, the (24, 28) geometry), w3 the pooled output
(26,803,200 — 98,304 bytes, the (12, 16) geometry w9 describes), w4
weights, w5 sync-in (64,000 — address-equal to slot 54's w6, the chained
handoff) and w6 sync-out (64,512 — the 512-byte stride continuing: 63,488
-> 64,000 -> 64,512), w7 the pair's second constant at
`0xFFFFFFFC00000000` — block30's high-dword step now measured in both
halves of the pair (slot 54's w4, slot 55's w7) — w8 the (24, 28) constant
`0x1C00000018` and w9 the (12, 16) constant `0x000000100000000C`. Its
weights word lands on the layout walk's next record start
(`block30.layer3.layer`, 263,168 B at heap offset 22,240,256 — 21,322,240 +
917,568 past the qkv record's 448-byte tail padding to the 512 boundary),
confirmed by the probe's live DLL walk. The write-set is three segments per
frame (names-s54-s55.txt, zero unattributed: 343,913/343,909/343,907 at w2
+ 97,553/97,802/97,775 at w3 + 96 sync) and the sync extent measured 96 B
(span 64,512 -> 64,608). The execute pair is bit-identical (run_sha256
`fb5d5362…`); the fp8 sibling writes both output tensors byte-for-byte and
leaves the sync record at the input state (the family trait). Block30's
pair is therefore the first `qkv`-proj pair whose second half is a distinct
kernel rather than block23's `proj` shape, and the interval's four-kernel
block pattern ends with it. The
natural next candidate is slot 56 (`cc_split_swin_16h_final_head_512_wait_fp8`),
closing the transformer interval at the s56 cut; the temporal/history state
(`prev_output`, counter) then connects the kernels into a full native
evaluation for the native-vs-vendor comparison of the whole pipeline. The
slot 56 (`cc_split_swin_16h_final_head_512_wait_fp8` — the transformer
interval's final head at grid (8,2,1), block (32,8,1)) is now replayed
byte-for-byte and confirms every word-map, record-binding and write-set
prediction. Its parameter block (psize 40) is a new five-word shape (pool,
pool, weights, pool, constant): w0 input (26,803,200 — address-equal to
slot 55's w3, the pooled `proj_pool` tensor), w1 output (52,657,152 —
196,608 bytes, 384 512-wide rows, twice the pooled input's 192), w2
weights, w3 sync-in (64,512 — address-equal to slot 55's w6, the chained
handoff) and w4 the (12, 16) constant `0x000000100000000C` — slot 55's w9
value recurring at the head. Its weights word lands on the layout walk's
next record start (`block30.layer4.layer`, 524,304 B at heap offset
22,503,424 — 22,240,256 + 263,168 at the 512 boundary, block30's fifth
record), confirmed by the probe's live DLL walk. The write-set is one
segment per frame (names-s55-s56.txt, zero unattributed:
171,969/171,970/171,962 at w1) and there is **no sync segment** — w3's
record extent is byte-unchanged across the cut: the first `wait`-family
kernel waits on the handoff without publishing one. The execute pair is
bit-identical (run_sha256 `323bfa2a…`); the fp8 sibling matches the
reference over the **whole pool** — the first zero-divergence sibling,
showing the variant pair differs only in sync-record handling, invisible at
this family. The output extent closes exactly at slot 57's output binding
(w1 at 52,853,760) and slot 57's w0 reads the head's output back as the ViT
interval's input, so the interval boundary is contiguous in the pool.

The slot 57 `cc_vit_1d_repack_2d_to_1d_fp8` (s56 -> s57) — the ViT
interval's opening repack — is now replayed byte-for-byte and opens three
new contract shapes at once: the first **single-variant** kernel (no fp8
sibling exists — the repack family has one kernel per direction, this
2d-to-1d and slot 98's `cc_vit_1d_repack_1d_to_2d_fp8`, not the
tilesync/chained/wait variant pairs of every earlier family), the first
**weights-less** contract (psize 24, three words — the smallest block
replayed; every earlier block carries a weights word) and the first with
**no sync word** at all (the repack family is outside the sync chain). Its
parameter block is w0 input (52,657,152 — address-equal to slot 56's w1,
the final-head tensor read back as the ViT interval's input), w1 output
(52,853,760 — the head's output extent closes exactly here: 52,657,152 +
196,608) and w2 the (12, 16) constant `0x000000100000000C` (slot 55's w9
and slot 56's w4 recurring). The output tensor measures 196,608 bytes
(the head's own 384×512 geometry) closing exactly at the ViT interval's
53,050,368 boundary (slot 97's w2 / slot 98's w0 bind there). The
write-set is one segment per frame (names-s56-s57.txt: w1 alone, zero
unattributed, changed 171,969/171,970/171,962 in the span
[52,853,760, 53,050,364)). The kernel is a **pure byte permutation** of
the head tensor, not an arithmetic kernel: both write-set spans have
all-zero pre-states (measured over the retained capture pools:
vr/logs/ladder/permutation-s56-s57.json), so each write-set's changed
count is its output tensor's non-zero byte count and the two match frame
for frame (171,969/171,970/171,962); the sorted byte multisets of the head
tensor and the repacked tensor are equal on all three frames (same
artefact). With no weights word the
control set re-bases onto the input tensor: `wrong` rotates the input
tensor inwards 1,024 bytes (the output must change — 343,184/343,038/
342,948 bytes differ against the reference, pool changed from input) and
the read-bounding tail pattern goes into the output extent's pre-launch
bytes — the bytes immediately beyond the input tensor — in both patterns
(execute 0xA5 / tail 0x5A), outputs unchanged under both (**the first
tail control to bound input reads rather than weight reads**; the window
sits inside the output extent because that is the only pool state the
kernel rewrites, and `skip` stays pristine so its diff counts remain the
ladder write-set counts exactly). The ViT interval's kernels are created
from their own cubin and extracted from DLL fatbin container 5 (container
4 carries the transformer interval's), and the weights-less heap
resolution pins the addressed heap instance from the traces themselves —
the captured resource map carries four same-size heap instances, the
traces' 459 heap words address exactly one of them, all on record starts
(mid-record 0, both traces). The execute pair is bit-identical
(run_sha256 `6b75cafd…`) and the slot-56 and slot-55 additivity rechecks
pair at `323bfa2a…` and `fb5d5362…` — the schema extensions render every
earlier contract byte-identically (probe self-test 389 known answers / 34
corruption controls, 11 shape checks per contract across thirty-four). The
natural next candidate is slot 58 (`cc_vit_1d_ffn_expand_publish_fp8`), the
ViT interval's first arithmetic kernel and the `ffn_expand` family's
publish variant; the temporal/history state (`prev_output`, counter) then
connects the kernels into a full native evaluation for the native-vs-vendor
comparison of the whole pipeline.

The slot 58 `cc_vit_1d_ffn_expand_publish_fp8` (s57 -> s58) — the ViT
interval's first arithmetic kernel, first weighted contract and the
`ffn_expand` family's single-use `publish` variant (the later `ffn_expand`
launches use the `chained` variant at their own slots) — is now replayed
byte-for-byte and opens the ViT block's five-slot cycle (`ffn_expand` ->
`ffn_contract` -> `qkv` -> `attention` -> `projection`): it binds the first
ViT weight record (`block31.layer0.layer`, 4,194,320 B at heap offset
23,028,224 — block31's first record, the layout walk's next record start
after `block30.layer4.layer`), and it is the first `publish`-family kernel:
its parameter block binds **no sync-in word at all** — the interval's
opening publish writes the ViT chain's first sync record (256 B at w7, span
65,024 -> 65,280; the 512-byte stride continuing: 64,512 -> 65,024) without
consuming one, the slot-56 `wait` kernel's mirror. The write-set is two
segments per frame (names-s57-s58.txt: w2 with 779,794/779,827/779,876
changed bytes inside the 786,432-byte binding and w7 256 B changed, zero
unattributed) and `skip`'s diff counts remain the ladder write-set counts
exactly (780,050/780,083/780,132). The controls re-base on the weights
record (slot 57's weights-less input detour is over): `wrong` rotates the
4,194,320-byte record inwards 1,024 bytes (the output must change —
636,844–637,271 bytes differ against the reference, pool changed from
input) and the read-bounding tail pattern goes into the bytes immediately
beyond the record (execute 0xA5 / tail 0x5A), outputs unchanged under both.
The fp8 sibling (`cc_vit_1d_ffn_expand_fp8`) is created but never launched
(pinned: 0 launches) and matches the reference over the whole pool except
the sync record, left at the input state: 256 bytes, first difference at
65,024 — the sibling divergence equals the ladder's measured extent
exactly (each predicted from the independent ladder write-set and confirmed
by the replay's sibling divergence — two sources, one number). The execute
pair is bit-identical (run_sha256 `4fb03d2f…`) and the slot-57 and
slot-56 additivity rechecks pair at `6b75cafd…` and `323bfa2a…` — the
schema extensions render every
earlier contract byte-identically (probe self-test 400 known answers / 34
corruption controls, 11 shape checks per contract across thirty-five). The
natural next candidate is slot 59 (the ViT block's `ffn_contract`, the
five-slot cycle's second slot), which needs its s58->s59 reference capture
before the contract; the temporal/history state (`prev_output`, counter)
then connects the kernels into a full native evaluation for the
native-vs-vendor comparison of the whole pipeline.

The slot 59 `cc_vit_1d_ffn_contract_chained_fp8` (s58 -> s59) — the
contraction half of the ViT block's five-slot cycle (`ffn_expand` ->
`ffn_contract` -> `qkv` -> `attention` -> `projection`; the `projection`
half at slot 62 shares this whole word map) — is now replayed
byte-for-byte and opens the contraction's three-output shape: a block with
**no reserved zero word** (eight pool words and the constant — every word
bound), the first with **three written output regions** (w2's
196,608-byte contraction and w5's 393,216-byte second output — both
changed boxes closing 4 bytes short of their bindings, the final word
static in every frame — plus w4's 64-byte record-walk publication) and it
binds the **residual stream** directly (w1 re-binds slot 57's repacked
head tensor — produced two slots earlier — as the skip connection, rather
than the pair's swapped buffer the earlier halves' residual w1s carry). It binds
`block31.layer1.layer` (4,196,352 B at heap offset 27,223,040 — the layout
walk's next record start), and its sibling run re-classifies the variant
finding: of the block's two record-walk publications only the w7 handoff
(66,048 — slot 60's sync-in) is variant-dependent, while w4 (65,536 —
bound by no other launch in the 156-launch sequence) is output data
written byte-identically by both variants — the first sibling-shared
output beyond the main tensors, so the pinned sibling divergence is 64 B
where a record-walk-only reading of the ladder write-set predicts 128 B
(the write-set sees both publications change; only the replay's sibling
run separates them — the variant finding's two sources again, now across
classification as well as extent). The execute pair is bit-identical
(run_sha256 `b9b759b7…`) and the slot-58 and slot-57 additivity rechecks
pair at `4fb03d2f…` and `6b75cafd…` — the contract-59 entry (no probe
generalisation needed) renders every
earlier contract byte-identically (probe self-test 411 known answers / 34
corruption controls, 11 shape checks per contract across thirty-six). The
natural next candidate is slot 60 (the ViT block's `qkv`, the five-slot
cycle's third slot, its block already visible in the traces at a 10-word
shape); the temporal/history state (`prev_output`, counter)
then connects the kernels into a full native evaluation for the
native-vs-vendor comparison of the whole pipeline.

The slot 60 `cc_vit_1d_qkv_chained_fp8` (s59 -> s60) — the ViT block's
third slot, the `qkv` half of the five-slot cycle (`ffn_expand` ->
`ffn_contract` -> `qkv` -> `attention` -> `projection`) — is now replayed
byte-for-byte and opens the first **five-output** contract (and the first
six-segment write-set): the q/k/v triplet at w1/w2/w3 (196,608 bytes each,
384×512 — slot 61's `attention` binds them as its w0/w1/w2, and the
triplet tiles [53,443,584, 54,033,408), closing exactly at the attention
output's base), w6's 1,179,648-byte region (384×3,072, closing exactly at
slot 58's w2 binding at 34,372,608 — one per ViT block, block32's qkv
binding its own at pool 20,610,048) and the 128-byte w5 publication. Its
w0 re-reads slot 59's contraction output (53,246,976) and its w7 consumes
slot 59's sync record (66,048 — the chained handoff continuing at w8,
67,072: slot 61 binds that record at the attention block's w5 sync-in). It
binds `block31.layer2.layer` (3,145,856 B at heap offset 31,419,392 — the
layout walk's next record start after `block31.layer1.layer`, closing
exactly at its end). The write-set is six segments per frame
(names-s59-s60.txt: w6 covering its whole binding at
1,172,971/1,172,523/1,172,573 changed bytes, w3/w1/w2 at
171,974/171,993/171,981, 171,959/171,944/171,950 and
171,661/171,667/171,669 inside their 196,608-byte bindings — the changed
boxes closing 2/4/8 bytes short of their bindings, the final words static
in every frame — and 128/128/128 at each of w5 and w8, zero unattributed).
The sibling run confirms the variant finding's re-classification is the
family rule rather than a one-off: of the block's two record-walk
publications only the w8 handoff is variant-dependent (the predicted 128 B
divergence, pinned), while w5 (bound by no other launch in the
156-launch sequence) is output data written byte-identically by both
variants — the second sibling-shared publication beyond the main tensors.
The execute pair is bit-identical (run_sha256 `d9dd3a21…`) and the slot-59
and slot-58 additivity rechecks
pair at `b9b759b7…` and `4fb03d2f…` — the contract-60 entry (no probe
generalisation needed) renders every
earlier contract byte-identically (probe self-test 422 known answers / 34
corruption controls, 11 shape checks per contract across thirty-seven). The
natural next candidate is slot 61 (the ViT block's `attention`, the
five-slot cycle's fourth slot, its 8-word block already visible in the
traces); the temporal/history state (`prev_output`, counter)
then connects the kernels into a full native evaluation for the
native-vs-vendor comparison of the whole pipeline.

The slot 61 `cc_vit_1d_attention_chained_fp8` (s60 -> s61) — the ViT
block's fourth slot, the `attention` half of the five-slot cycle
(`ffn_expand` -> `ffn_contract` -> `qkv` -> `attention` -> `projection`)
— is now replayed byte-for-byte at the first **eight-word** block (psize
64): the q/k/v triplet read at w0/w1/w2 (53,443,584 / 53,836,800 /
53,640,192 — address-equal to slot 60's w1/w2/w3, the three 196,608-byte
tensors the attention consumes as query, key and value), the attention
output at w3 (54,033,408 — 196,608 bytes, closing exactly at slot 62's w2
binding at 54,230,016) and the 256-byte w6 publication (67,584 — the
512-byte stride continuing: 67,072 -> 67,584; slot 62 binds the record at
the projection block's w6 sync-in). Its w5 consumes slot 60's sync record
(67,072 — the chained handoff continuing). It binds
`block31.layer3.layer` (a 2-byte payload at heap offset 34,565,632 — the
layout walk's next record start after `block31.layer2.layer`, whose end at
34,565,248 rounds up at the 512 boundary) — and the stage's finding is
that this record is a **dead binding**: the ViT attention family is
weight-free and the kernel's output is invariant under whole-record
complementation as well as under the wrong control's one-byte rotation,
measured in both variants (the retained `wrong-superseded-record-swap` and
`wrong-complement-experiment` runs reproduce the execute pools
byte-for-byte). The record therefore cannot carry the wrong control's
sensitivity claim, so the control **re-bases onto the query tensor at w0**
— the probe's first control re-targeting, pinned per contract as
`wrong_word`/`wrong_bytes` with a shape check of its own (paired,
pool-class, rotation headroom, weighted contracts only), while the
rotation helper keeps the one-byte extent and no-op rejection the 2-byte
record would have needed for any future live record of that shape. The
write-set is two segments per frame (names-s60-s61.txt: w3 with
196,535/196,524/196,496 changed bytes inside the 196,608-byte binding —
the changed box covering the whole binding, 73/84/112 interior bytes
static in the three frames — and w6 256/256/256, the whole publication,
zero unattributed). Slot 61 is the first contract with a **single**
record-walk publication, and the family rule classifies it by consumption
alone: slot 62 binds w6's record as its sync-in, so w6 is the chain
handoff and the sibling run pins the predicted 256 B divergence exactly
(`sync_from_input` true — the record is left at the input state); there is
no sibling-shared publication at this block. The execute pair is
bit-identical (run_sha256 `77c696d2…`) and the slot-60 and slot-59
additivity rechecks pair at `d9dd3a21…` and `b9b759b7…` — the contract-61
entry and the control re-targeting render every earlier contract
byte-identically (probe self-test 474 known answers / 39 corruption
controls, 12 shape checks per contract across thirty-eight). The natural
next candidate is slot 62 (the ViT block's `projection`, the five-slot
cycle's fifth slot, its block already visible at a 9-word shape with both
`projection` variants and a `wait` variant created); the temporal/history
state (`prev_output`, counter) then connects the kernels into a full
native evaluation for the native-vs-vendor comparison of the whole
pipeline.

The slot 62 `cc_vit_1d_projection_chained_fp8` (s61 -> s62) — the ViT
block's fifth slot, the `projection` half closing block31's five-slot cycle
(`ffn_expand` -> `ffn_contract` -> `qkv` -> `attention` -> `projection`) —
is now replayed byte-for-byte and shares slot 59's nine-word contraction
shape (psize 72, every word bound, three written output regions): the
projection input read at w0 (54,033,408 — address-equal to slot 61's w3,
the attention output read back), the residual at w1 (53,246,976 —
address-equal to slot 59's w2, the contraction output from three slots
earlier bound again as the attention sub-block's skip connection), the
projection output at w2 (54,230,016 — 196,608 bytes, 384x512, closing
exactly at 54,426,624; slot 63's `ffn_expand` binds the region as w0 and slot
64's `ffn_contract` as its w1 residual, opening block32's cycle), the
second output at w5 (49,560,576 — 393,216 bytes, immediately past slot
59's w5 [49,167,360, 49,560,576) and closing exactly at slot 64's w5
binding at 49,953,792) and the 64-byte w4 publication (68,096 — the record
walk's 512-byte stride continuing: 67,584 -> 68,096). Its w6 consumes slot
61's sync record (67,584 — the chained handoff continuing) and its w7
publishes the next (68,608 — the next stride; slot 63 binds the record at
the `ffn_expand` block's w6 sync-in). It binds `block31.layer4.layer`
(1,050,624 B at heap offset 34,566,144 — the layout walk's next record
start after `block31.layer3.layer`, whose 2-byte payload at 34,565,632
rounds up across 510 bytes of padding to the 512 boundary, closing
block31's five-record set exactly at `block32.layer0.layer`'s start at
35,616,768) — the attention block's placeholder record skipped in between,
so this is the first live weight record since `block31.layer2.layer` (slot
60's): the wrong control rotates the record itself here, no re-targeting
needed, and the output breaks in every entry (577,032–577,587 bytes differ
from the reference), confirming the projection family reads its weights —
the live counterpart to slot 61's dead placeholder. The write-set is four
segments per frame (names-s61-s62.txt: w5 with 391,486/391,452/391,349
changed bytes inside the 393,216-byte binding and w2 with
196,500/196,524/196,534 inside the 196,608-byte binding — both changed
boxes covering their whole bindings, unlike slot 59's 4-byte-short boxes —
and 64/64/64 at each of w4 and w7, the whole publications, zero
unattributed). The sibling run confirms the variant finding's
re-classification is the family rule rather than a one-off: of the block's
two record-walk publications only the w7 handoff is variant-dependent (the
predicted 64 B divergence, pinned), while w4 (bound by no other launch in
the 156-launch sequence) is output data written byte-identically by both
variants — the third sibling-shared publication beyond the main tensors.
The execute pair is bit-identical (run_sha256 `7c319752…`) and the slot-61
and slot-60 additivity rechecks pair at `77c696d2…` and `d9dd3a21…` — the
contract-62 entry (no probe generalisation needed) renders every earlier
contract byte-identically (probe self-test 486 known answers / 39
corruption controls, 12 shape checks per contract across thirty-nine). The
natural next candidate is slot 63 (the second ViT block's `ffn_expand`,
the `chained` variant's first launch — slot 58's `publish` opened
block31's cycle — its block already visible at the 9-word shape); the
temporal/history state (`prev_output`, counter) then connects the kernels
into a full native evaluation for the native-vs-vendor comparison of the
whole pipeline.

The slot 63 `cc_vit_1d_ffn_expand_chained_fp8` (s62 -> s63) — the second
ViT block's opening slot, block32's five-slot cycle starting at the
`ffn_expand` family's `chained` variant's first launch (slot 58's
`publish` opened block31's cycle, writing the chain's first record without
consuming one) — is now replayed byte-for-byte and shares slot 58's
nine-word expansion shape (psize 72) with w6 bound as the sync-in that
`publish` leaves zero: the expansion input read at w0 (54,230,016 —
address-equal to slot 62's w2, the projection output read back as the next
block's input), the expansion output at w2 (35,159,040 — 786,432 bytes,
the 384x2048 fp8 expansion of the 384x512 input, closing exactly at
35,945,472 where slot 68's `chained` expansion output chains one
block-stride later and slot 64's `ffn_contract` binds the region as w0) and
the 256-byte w7 publication (69,120 — the record walk's 512-byte stride
continuing: 68,608 -> 69,120; slot 64 binds the record at the
`ffn_contract` block's w6 sync-in). Its w6 consumes slot 62's sync record
(68,608 — the chained handoff continuing). It binds `block32.layer0.layer`
(4,194,320 B at heap offset 35,616,768 — block31's five-record set ends
there exactly, no padding intervenes, block32's first record; block32
repeats block31's five-record shape exactly) — a live record again: the
wrong control rotates it and the output breaks in every entry
(730,753–733,258 bytes differ from the reference), so no wrong-control
re-targeting is needed at this contract. The write-set is two segments per
frame (names-s62-s63.txt: w2 with 778,353/777,687/777,881 changed bytes
inside the 786,432-byte binding — the changed box covering the whole
binding, 8,079/8,745/8,551 interior bytes static in the three frames —
and 256/256/256 at w7, the whole publication, zero unattributed). The
sibling run returns to the single-publication shape and resolves it by
consumption alone a second time: w7 is the chain handoff (the predicted
256 B divergence, pinned), with no sibling-shared publication at this
block. The execute pair is bit-identical (run_sha256 `d0018066…`) and the
slot-62 and slot-61 additivity rechecks pair at `7c319752…` and
`77c696d2…` — the contract-63 entry (no probe generalisation needed)
renders every earlier contract byte-identically (probe self-test 498 known
answers / 39 corruption controls, 12 shape checks per contract across
forty). The natural next candidate is slot 64 (the second ViT block's
`ffn_contract`, the `chained` variant continuing the cycle — its block
already visible at slot 59's 9-word contraction shape); the
temporal/history state (`prev_output`, counter) then connects the kernels
into a full native evaluation for the native-vs-vendor comparison of the
whole pipeline.

The slot 64 `cc_vit_1d_ffn_contract_chained_fp8` (s63 -> s64) — the
second ViT block's second slot, block32's `ffn_contract` half of the
five-slot cycle — is now replayed byte-for-byte and shares slot 59's
nine-word contraction shape exactly (psize 72, the same (12, 16) constant
at w8): the contraction input read at w0 (35,159,040 — address-equal to
slot 63's w2, the expansion output read back), the residual at w1
(54,230,016 — address-equal to slot 62's w2, the projection output from
two slots earlier bound again as the contraction's skip connection, slot
59 binding slot 57's head tensor the same way), the contraction output at
w2 (54,426,624 — 196,608 bytes, 384x512, closing exactly at 54,623,232
where slot 65's `qkv` binds the region as its q input and slot 67's
`projection` re-reads it as its w1 residual), the second output at w5
(49,953,792 — 393,216 bytes, immediately past slot 62's w5
[49,560,576, 49,953,792) and closing exactly at 50,347,008 where slot
69's w5 chains one block-stride later) and the 64-byte w4 publication
(69,632 — the record walk's 512-byte stride continuing: 69,120 ->
69,632). Its w6 consumes slot 63's sync record (69,120 — the chained
handoff continuing) and its w7 publishes the next (70,144 — the next
stride; slot 65 binds the record at the `qkv` block's w7 sync-in). It
binds `block32.layer1.layer` (4,196,352 B at heap offset 39,811,584 — the
layout walk's next record start after `block32.layer0.layer`, whose
4,194,320-byte payload at 35,616,768 ends 496 bytes below the 512
boundary; the block31 layer0-to-layer1 pair repeats exactly) — a live
record again: the wrong control rotates it and the output breaks in every
entry (576,052–576,607 bytes differ from the reference), so no
wrong-control re-targeting is needed at this contract either. The
write-set is four segments per frame (names-s63-s64.txt: w5 with
391,734/391,648/391,716 changed bytes inside the 393,216-byte binding and
w2 with 196,523/196,553/196,503 inside the 196,608-byte binding — both
changed boxes covering their whole bindings exactly, 1,482/1,568/1,500 and
85/55/105 interior bytes static in the three frames — plus w4 and w7 64 B
each, the whole publications, zero unattributed). The two record-walk
publications split by the family rule a fourth time: w7 is the chain
handoff (the predicted 64 B sibling divergence, pinned) while w4 is
variant-independent output data (the fourth sibling-shared publication;
bound by no other launch in the sequence). The execute pair is
bit-identical (run_sha256 `ab1fda10…`) and the slot-63 and slot-62
additivity rechecks pair at `d0018066…` and `7c319752…` — the
contract-64 entry (no probe generalisation needed)
renders every earlier contract byte-identically (probe self-test 510 known
answers / 39 corruption controls, 12 shape checks per contract across
forty-one). The natural next candidate is slot 65 (block32's `qkv`
`chained`, the cycle's third slot continuing on slot 60's 80-byte word
map); the temporal/history state (`prev_output`, counter) then connects
the kernels into a full native evaluation for the native-vs-vendor
comparison of the whole pipeline.

The slot 65 `cc_vit_1d_qkv_chained_fp8` (s64 -> s65) — the second ViT
block's third slot, block32's `qkv` half of the five-slot cycle — is now
replayed byte-for-byte and shares slot 60's ten-word qkv shape exactly
(psize 80, the same (12, 16) constant at w9): the contraction input read
at w0 (54,426,624 — address-equal to slot 64's w2, the contraction output
read back as the qkv input), the q/k/v triplet written at w1/w2/w3
(196,608 bytes each at 54,623,232 / 54,819,840 / 55,016,448 — slot 66's
attention binds them as its w0/w1/w2, the triplet tiling [54,623,232,
55,213,056) and closing exactly at the attention output's base at
55,213,056), the fourth written region at w6 (20,610,048 — 1,179,648
bytes = 384x3,072, the block32 instance of the one-per-block region slot
60's comment predicts, closing exactly at 21,789,696 where block33's
`qkv` w6 chains immediately) and the 128-byte w5 publication (70,656 —
the record walk's 512-byte stride continuing: 69,632 -> 70,144 ->
70,656). Its w7 consumes slot 64's sync record (70,144 — the chained
handoff continuing) and its w8 publishes the next (71,168 — the next
stride; slot 66 binds the record at the `attention` block's w5 sync-in).
It binds `block32.layer2.layer` (3,145,856 B at heap offset 44,007,936 —
the layout walk's next record start after `block32.layer1.layer`, whose
4,196,352-byte payload at 39,811,584 ends on the 512 boundary exactly
with no padding; the block31 layer1-to-layer2 pair repeats exactly) — a
live record again: the wrong control rotates it and the output breaks in
every entry (1,744,940–1,745,632 bytes differ from the reference), so no
wrong-control re-targeting is needed at this contract either. The
write-set is six segments per frame (names-s64-s65.txt: w6 with
1,173,080/1,172,617/1,172,341 changed bytes inside the 1,179,648-byte
binding and w3/w1/w2 with 196,552/196,517/196,546,
196,549/196,526/196,538 and 196,218/196,171/196,181 inside the three
196,608-byte bindings — all four changed boxes covering their whole
bindings exactly, 6,568/7,031/7,307, 56/91/62, 59/82/70 and
390/437/427 interior bytes static in the three frames — plus w5 and w8
128 B each, the whole publications, zero unattributed). The two
record-walk publications split by the family rule a fifth time: w8 is the
chain handoff (the predicted 128 B sibling divergence, pinned) while w5
is variant-independent output data (the fifth sibling-shared publication;
bound by no other launch in the sequence). The execute pair is
bit-identical (run_sha256 `922949f3…`) and the slot-64 and slot-63
additivity rechecks pair at `ab1fda10…` and `d0018066…` — the
contract-65 entry (no probe generalisation needed)
renders every earlier contract byte-identically (probe self-test 522 known
answers / 39 corruption controls, 12 shape checks per contract across
forty-two). The natural next candidate is slot 66 (block32's `attention`
`chained` — its 8-word block already visible at psize 64, binding
`block32.layer3.layer`'s 2-byte placeholder, so the dead-binding
wrong-control re-targeting returns there); the temporal/history state
(`prev_output`, counter) then connects the kernels into a full native
evaluation for the native-vs-vendor comparison of the whole pipeline.

The slot 66 `cc_vit_1d_attention_chained_fp8` (s65 -> s66) — the second
ViT block's fourth slot, block32's `attention` half of the five-slot
cycle — is now replayed byte-for-byte and shares slot 61's eight-word
attention shape exactly (psize 64, the same (12, 16) constant at w7): the
q/k/v triplet read at w0/w1/w2 (54,623,232 / 54,819,840 / 55,016,448 —
address-equal to slot 65's w1/w2/w3), the attention output written at w3
(55,213,056 — 196,608 bytes, closing exactly at slot 67's w2 binding at
55,409,664) and the 256-byte w6 publication (71,680 — the 512-byte stride
continuing: 71,168 -> 71,680; slot 67 binds the record at the projection
block's w6 sync-in). Its w5 consumes slot 65's sync record (71,168 — the
chained handoff continuing). It binds `block32.layer3.layer` (a 2-byte
payload at heap offset 47,154,176 — the layout walk's next record start
after `block32.layer2.layer`, whose end at 47,153,792 rounds up across 384
bytes at the 512 boundary; the ViT attention blocks' placeholder records
all measure a 2-byte payload — block32's `f3 b3`) — and the stage's
finding is slot 61's repeating exactly: this record too is a **dead
binding**. With the wrong control aimed at the record (the query
re-targeting removed for the experiment), both content-changing mutations
— an inward rotation and whole-record complementation — reproduce every
execute pool byte-for-byte in the retained
`wrong-record-rotation-experiment` and `wrong-complement-experiment` runs
(all six digests per run identical to the execute run's, the wrong mode's
`wrong-matched` enforcement failures retained as the invariance
evidence), so the weight-free attention family reads nothing from its
placeholder record at block32 either and the wrong control **re-bases
onto the query tensor at w0** — the family's second re-targeting, pinned
per contract as `wrong_word`/`wrong_bytes`. The standard wrong run then
rejects as it must (381,748–382,857 bytes differ from the reference, the
rotated query propagating through the attention to the output on all six
entries). The write-set is two segments per frame (names-s65-s66.txt: w3
with 196,547/196,544/196,547 changed bytes inside the 196,608-byte
binding — the changed box covering the whole binding, 61/64/61 interior
bytes static in the three frames — and w6 256/256/256, the whole
publication, zero unattributed). As at slot 61 the block has a **single**
record-walk publication, classified by consumption alone: slot 67 binds
w6's record as its sync-in, so w6 is the chain handoff and the sibling
run pins the predicted 256-byte divergence exactly (the plain variant
leaves the record at the input state; there is no sibling-shared
publication at this contract). The execute pair is bit-identical
(run_sha256 `d0131a22…`) and the slot-65 and slot-64 additivity rechecks
pair at `922949f3…` and `ab1fda10…` — the contract-66 entry (no probe
generalisation needed)
renders every earlier contract byte-identically (probe self-test 534 known
answers / 39 corruption controls, 12 shape checks per contract across
forty-three). The natural next candidate is slot 67 (block32's
`projection` `chained` — closing block32's cycle on the known projection
word map, with `block32.layer4.layer`'s live record at heap 47,154,688);
the temporal/history state (`prev_output`, counter) then connects the
kernels into a full native evaluation for the native-vs-vendor comparison
of the whole pipeline.

The slot 67 `cc_vit_1d_projection_chained_fp8` (s66 -> s67) — the second
ViT block's fifth slot, block32's `projection` closing the five-slot cycle
— is now replayed byte-for-byte and shares slot 62's nine-word projection
shape exactly (psize 72, grid (16,1,4), the same (12, 16) constant at w8):
the input read at w0 (55,213,056 — address-equal to slot 66's w3, the
attention output read back as the projection's input), the residual read
at w1 (54,426,624 — address-equal to slot 64's w2, the contraction output
from three slots earlier bound again as the attention sub-block's skip
connection), the output written at w2 (55,409,664 — 196,608 bytes,
closing at 55,606,272 where slot 68's `ffn_expand` binds it as w0), the
second output at w5 (48,774,144 — 393,216 bytes, closing at 49,167,360
where slot 59's w5 binds the region's next extent) and the two record-walk
publications at w4 (72,192 — 64 bytes, the 512-byte stride continuing:
71,680 -> 72,192) and w7 (72,704 — 64 bytes, the next stride; slot 68
binds the record at the `ffn_expand` block's w6 sync-in). Its w6 consumes
slot 66's sync record (71,680 — the chained handoff continuing). It binds
`block32.layer4.layer` (1,050,624 B at heap offset 47,154,688 — the
layout walk's next record start after `block32.layer3.layer`, whose 2-byte
payload at 47,154,176 rounds up across 510 bytes of padding to the 512
boundary, closing block32's five-record set exactly at
`block33.layer0.layer`'s start at 48,205,312 — the block31
layer3-to-layer4 pair repeating exactly) and the record is **live**: the
wrong control's inward 1,024-byte rotation breaks the output everywhere
(574,479–574,631 bytes differ from the reference across the six entries),
no re-targeting needed here as at slot 62. The write-set is four segments
per frame (names-s66-s67.txt: w5 with 391,543/391,442/391,443 changed
bytes inside the 393,216-byte binding and w2 with
196,545/196,547/196,536 inside the 196,608-byte binding — both changed
boxes covering their whole bindings — and w4/w7 64 B changed each, zero
unattributed). The two record-walk publications confirm the
re-classification a sixth time: w7 is the chain handoff (its record is
bound by slot 68 as the `ffn_expand` block's w6 sync-in — the predicted
64-byte sibling divergence, pinned exactly) while w4 is bound by no other
launch in the sequence and is variant-independent output data written
byte-identically by both variants. The execute pair is bit-identical
(run_sha256 `02704ec7…`) and the slot-66 and slot-65 additivity rechecks
pair at `d0131a22…` and `922949f3…` — the contract-67 entry (no probe
generalisation needed)
renders every earlier contract byte-identically (probe self-test 546 known
answers / 39 corruption controls, 12 shape checks per contract across
forty-four). The natural next candidate is slot 68 (block33's
`ffn_expand` `chained` — the third block's cycle opening on the expansion
word map, with `block33.layer0.layer`'s record at heap 48,205,312); the
temporal/history state (`prev_output`, counter) then connects the
kernels into a full native evaluation for the native-vs-vendor comparison
of the whole pipeline.

The slot 68 `cc_vit_1d_ffn_expand_chained_fp8` (s67 -> s68) — block33's
expansion, the third ViT block's opening slot — is now replayed
byte-for-byte and shares slot 63's nine-word expansion shape exactly
(psize 72, grid (64,1,1), the same (12, 16) constant at w8, w1/w4/w5
reserved zero): the input read at w0 (55,409,664 — address-equal to slot
67's w2, the projection output read back as the next block's input), the
output written at w2 (35,945,472 — 786,432 bytes, the 384x2048 fp8
expansion, closing at 36,731,904; slot 63's expansion output closes at
35,945,472 exactly, the expansion outputs chaining one block-stride apart,
and slot 69's `ffn_contract` binds the region as w0), and the single
record-walk publication at w7 (73,216 — 256 bytes, the 512-byte stride
continuing: 72,704 -> 73,216). Its w6 consumes slot 67's sync record
(72,704 — the chained handoff continuing). It binds `block33.layer0.layer`
(4,194,320 B at heap offset 48,205,312 — block32's five-record set ends
there exactly, its layer4's 1,050,624-byte payload closing at that
boundary with no padding, and block33's first record begins; 496 bytes of
padding close the gap to slot 69's record at 52,400,128) and the record is
**live**: the wrong control's inward 1,024-byte rotation breaks the output
everywhere (725,392–726,875 bytes differ from the reference across the six
entries), no re-targeting needed as at slots 62 and 67. The write-set is
two segments per frame (names-s67-s68.txt: w2 with
779,454/778,805/778,812 changed bytes inside the 786,432-byte binding —
the changed box covering the whole binding — and w7 256/256/256, the whole
publication, zero unattributed). The block's single record-walk
publication resolves the classification by consumption alone a fourth
time: w7 is the chain handoff (slot 69 binds the record at its w6
sync-in — the predicted 256-byte sibling divergence, pinned exactly), with
no sibling-shared publication at this block. The execute pair is
bit-identical (run_sha256 `852ce980…`) and the slot-67 and slot-66
additivity rechecks pair at `02704ec7…` and `d0131a22…` — the contract-68
entry (no probe generalisation needed) renders every earlier contract
byte-identically (probe self-test 558 known answers / 39 corruption
controls, 12 shape checks per contract across forty-five). The natural
next candidate is slot 69 (block33's `ffn_contract` `chained` — the
contraction on slot 59's word map, its record `block33.layer1.layer` at
heap 52,400,128 and its residual skip binding slot 67's output at w1);
the temporal/history state (`prev_output`, counter) then connects the
kernels into a full native evaluation for the native-vs-vendor comparison
of the whole pipeline.

The slot 69 `cc_vit_1d_ffn_contract_chained_fp8` (s68 -> s69) — block33's
contraction, the third ViT block's second slot — is now replayed
byte-for-byte and shares slot 59's nine-word contraction shape exactly
(psize 72, grid (16,1,4), the same (12, 16) constant at w8, every word
bound): the input read at w0 (35,945,472 — address-equal to slot 68's w2,
the expansion output read back as the contraction's input), the residual
skip at w1 (55,409,664 — slot 67's w2, the projection output from two
slots earlier, slot 64 binding slot 62's w2 the same way), the output
written at w2 (55,606,272 — 196,608 bytes, the 384x512 fp8 contraction
closing at 55,802,880; slot 72's projection re-reads the region as its w1
residual), the second output at w5 (50,347,008 — 393,216 bytes, closing
at 50,740,224, the second-output chain stepping one block-stride from
slot 64's w5 as predicted at stage 56) and the record walk's two
publications at w4 (73,728 — 64 bytes) and w7 (74,240 — 64 bytes, the
512-byte stride continuing: 73,216 -> 73,728 -> 74,240). Its w6 consumes
slot 68's sync record (73,216 — the chained handoff continuing, exactly
as the stage-60 contract predicted). It binds `block33.layer1.layer`
(4,196,352 B at heap offset 52,400,128 — the layout walk's next record
start after `block33.layer0.layer`'s 4,194,320-byte payload at
48,205,312 plus 496 bytes of padding to the 512 boundary) and the record
is **live**: the wrong control's inward 1,024-byte rotation breaks the
output everywhere (573,813–574,284 bytes differ from the reference
across the six entries), no re-targeting needed as at slots 62, 64 and
67. The write-set is four segments per frame (names-s68-s69.txt: w5 with
391,762/391,784/391,712 changed bytes inside the 393,216-byte binding,
w2 with 196,555/196,553/196,521 inside the 196,608-byte binding — both
changed boxes covering their whole bindings — and w4/w7 64 B each, the
whole publications, zero unattributed). The block's two record-walk
publications resolve the re-classification a seventh time: only w7 is the
chain handoff (slot 70 binds the record at the `qkv` block's w7 sync-in —
the predicted 64-byte sibling divergence, pinned exactly), while w4 is
variant-independent output data shared by both variants (the seventh
sibling-shared publication; no other launch in the sequence binds w4's
record). The execute pair is bit-identical (run_sha256 `16cb2374…`) and
the slot-68 and slot-67 additivity rechecks pair at `852ce980…` and
`02704ec7…` — the contract-69 entry (no probe generalisation needed)
renders every earlier contract byte-identically (probe self-test 570
known answers / 39 corruption controls, 12 shape checks per contract
across forty-six). The natural
next candidate is slot 70 (block33's `qkv` `chained` — the q/k/v triplet
on slot 60's ten-word map, its record `block33.layer2.layer` at heap
56,596,480 and its triplet landing right after slot 69's output at
55,802,880);
the temporal/history state (`prev_output`, counter) then connects the
kernels into a full native evaluation for the native-vs-vendor comparison
of the whole pipeline.

The slot 70 `cc_vit_1d_qkv_chained_fp8` (s69 -> s70) — block33's q/k/v,
the third ViT block's third slot — is now replayed byte-for-byte and
shares slot 60's ten-word qkv shape exactly (psize 80, grid (32,1,2),
the same (12, 16) constant at w9, every word bound): the input read at
w0 (55,606,272 — address-equal to slot 69's w2, the contraction output
read back as the qkv input), the q/k/v triplet at w1/w2/w3
(55,802,880 / 55,999,488 / 56,196,096 — 196,608 bytes each, the
384x512 fp8 tensors tiling [55,802,880, 56,392,704) and closing exactly
at the attention output's base at 56,392,704 where slot 71's w3 lands,
slot 65's block32 triplet closing at 55,213,056 the same way), the
fourth written region at w6 (21,789,696 — 1,179,648 bytes = 384x3,072,
the block33 instance of the one-per-block region and the exact address
slot 65's comment predicted the block33 `qkv` w6 would chain
immediately at), and the record walk's two publications at w5 (74,752 —
128 bytes) and w8 (75,264 — 128 bytes, the 512-byte stride continuing:
74,240 -> 74,752 -> 75,264). Its w7 consumes slot 69's sync record
(74,240 — the chained handoff continuing, exactly as the stage-61
contract predicted). It binds `block33.layer2.layer` (3,145,856 B at
heap offset 56,596,480 — the layout walk's next record start after
`block33.layer1.layer`'s 4,196,352-byte payload at 52,400,128 closing
exactly there with no padding, the stage-61 prediction pinned exactly)
and the record is **live**: the wrong control's inward 1,024-byte
rotation breaks the output everywhere (1,744,678–1,745,066 bytes differ
from the reference across the six entries), no re-targeting needed as at
slots 62, 64, 67 and 69. The write-set is six segments per frame
(names-s69-s70.txt: w6 with 1,173,143/1,172,569/1,172,385 changed bytes
inside the 1,179,648-byte binding — the changed box covering the whole
binding — and w1/w2/w3 with 196,558/196,550/196,546,
196,254/196,178/196,178 and 196,549/196,547/196,549 inside the
196,608-byte bindings — all three covering their whole bindings — and
w5/w8 128 B each, the whole publications, zero unattributed). The
block's two record-walk publications resolve the re-classification an
eighth time: only w8 is the chain handoff (slot 71 binds the record at
the attention block's w5 sync-in — the predicted 128-byte sibling
divergence, pinned exactly), while w5 is variant-independent output
data shared by both variants (the eighth sibling-shared publication; no
other launch in the sequence binds w5's record). The execute pair is
bit-identical (run_sha256 `575a7e6b…`) and the slot-69 and slot-68
additivity rechecks pair at `16cb2374…` and `852ce980…` — the
contract-70 entry (no probe generalisation needed) renders every earlier
contract byte-identically (probe self-test 582 known answers / 39
corruption controls, 12 shape checks per contract across forty-seven).
The natural
next candidate is slot 71 (block33's `attention` `chained` — the
attention shape on slot 61's eight-word map, the third dead-binding
case: its block binds `block33.layer3.layer`, the block33 placeholder,
so the wrong control will need re-targeting onto the query tensor);
the temporal/history state (`prev_output`, counter) then connects the
kernels into a full native evaluation for the native-vs-vendor comparison
of the whole pipeline.

The slot 71 `cc_vit_1d_attention_chained_fp8` (s70 -> s71) — block33's
attention, the third ViT block's fourth slot and the attention family's
third launch — is now replayed byte-for-byte and shares slot 61's
eight-word attention shape exactly (psize 64, grid (32,1,1), the (12, 16)
constant at w7, every word bound): the q/k/v triplet at w0/w1/w2
(55,802,880 / 55,999,488 / 56,196,096 — address-equal to slot 70's
w1/w2/w3, consumed as query, key and value), the output at w3
(56,392,704 — 196,608 bytes, closing at 56,589,312 where slot 72's
projection w2 lands; the base slot 70's contract predicted), and the
record walk's single publication at w6 (75,776 — 256 bytes, the
512-byte stride continuing: 75,264 -> 75,776). Its w5 consumes slot
70's sync record (75,264 — the chained handoff continuing). It binds
`block33.layer3.layer` (a 2-byte payload at heap offset 59,742,720 — the
layout walk's next record start after `block33.layer2.layer`, whose
3,145,856-byte payload at 56,596,480 ends
at 59,742,336 and rounds up across 384 bytes at the 512 boundary; the
ViT attention blocks' layer3 records all measure a 2-byte payload) and
the record is the family's third **dead binding**: the wrong control
re-bases onto the query tensor at w0 (the family's third re-targeting,
pinned by `wrong_word`), where the inward 1,024-byte rotation breaks the
output everywhere (382,682–383,440 bytes differ from the reference
across the six entries). The write-set is two segments per frame
(names-s70-s71.txt: w3 with 196,523/196,527/196,549 changed bytes inside
the 196,608-byte binding — the changed box covering the whole binding,
85/81/59 interior bytes static — and w6 256 B, the whole publication,
zero unattributed; frame totals 196,779/196,783/196,805). The block has
a single record-walk publication, classified by consumption alone: slot
72 binds w6's record as its sync-in, so w6 is the chain handoff and the
sibling divergence is the whole 256-byte publication (the plain variant
leaves it at the input state) — the attention family's third instance of
the single-publication rule, no sibling-shared publication at this
contract. The execute pair is bit-identical (run_sha256 `50ce1b9d…`) and
the slot-70 and slot-69 additivity rechecks pair at `575a7e6b…` and
`16cb2374…` — the contract-71 entry (no probe generalisation needed)
renders every earlier contract byte-identically (probe self-test 594
known answers / 39 corruption controls, 12 shape checks per contract
across forty-eight). The natural
next candidate is slot 72 (block33's `projection` `chained` — the
projection shape on slot 62's nine-word map, closing block33's cycle;
its skip connection binds slot 69's contraction output two slots earlier
and its w4 record-walk publication re-joins the two-publication shape);
the temporal/history state (`prev_output`, counter) then connects the
kernels into a full native evaluation for the native-vs-vendor comparison
of the whole pipeline.

The slot 73 `cc_vit_1d_ffn_expand_chained_fp8` (s72 -> s73) — block34's
opening expansion in the ViT five-slot cycle — is now replayed byte-for-byte.
It shares slot 63's nine-word shape (psize 72, grid (64,1,1), w8 the
(12,16) constant): w0 reads slot 72's output at 56,589,312; w2 writes the
786,432-byte expansion at 36,731,904, closing at 37,518,336; w3 binds the
live `block34.layer0.layer` record (4,194,320 B at heap offset 60,793,856,
exactly where block33's five-record set ends); w6 consumes slot 72's
64-byte record at 76,800, and w7 publishes 256 B at 77,312, consumed as
w6 by slot 74. Words w1, w4 and w5 are reserved zero. The measured write-set
has two segments per frame and zero unattributed bytes: 779,475/778,866/778,660
changed bytes at w2 and 256 at w7, totals 779,731/779,122/778,916. The wrong
control rotates the live record inwards 1,024 B and rejects (720,842–725,565
bytes differ); the tail control leaves output unchanged beyond the
4,194,320-byte record. The fp8 sibling matches the output tensor but leaves
the entire 256-byte sync publication at input state, confirming the
single-publication handoff classification. Execute runs are bit-identical
(run SHA-256 `4b4171a3…`); slot-72 and slot-71 rechecks under this contract are
bit-identical to their retained runs (`4099a395…` and `50ce1b9d…`). The
contract shape is 12 checks, with 618 known answers and 39 corruption
controls. The next gate is slot 74, `cc_vit_1d_ffn_contract_chained_fp8`,
continuing block34's cycle. Temporal/history state (`prev_output`, counter)
will then be needed to connect the kernels into a full native evaluation
for comparison against the vendor pipeline.

The slot 74 `cc_vit_1d_ffn_contract_chained_fp8` (s73 -> s74) —
block34's contraction, second slot in its five-launch cycle — is now replayed
byte-for-byte. It shares slot 64's nine-word map (psize 72, grid (16,1,4),
w8 the (12,16) constant): w0 reads the slot-73 expansion at 36,731,904; w1
residual reads slot 72's projection output at 56,589,312; w2 writes 196,608 B
at 56,785,920; w3 binds live `block34.layer1.layer` (4,196,352 B at heap
offset 64,988,672, after block34.layer0.layer's 4,194,320 B and 496 B
alignment padding); w4 third output is 64 B at 77,824; w5 second output is
393,216 B at 51,133,440; w6 consumes slot 73's 256-byte publication at
77,312; w7 publishes 64 B at 78,336. Four measured write-set segments per
frame: w5 391,689/391,742/391,717 changed bytes, w2
196,552/196,550/196,552, and 64 B at each w4/w7; totals
588,369/588,420/588,397 with zero unattributed. The wrong-weight control
rejects (574,153–574,546 bytes differ); tail poisoning reproduces the live execute pool
byte-for-byte. The fp8 sibling matches all three outputs (w2, w5 and w4), with exactly 64 B divergence at w7. Execute pair
SHA-256 is `83ba0c358858f220dc2281006483b2758eeb973a83dbb70a3f0e50ffd4bb703f`;
slot-73 and slot-72 source-neutrality rechecks match their prior digests
(`4b4171a3…` and `4099a395…`). The 12-check contract yields 630 known
answers / 39 controls. Next is slot 75, block34's qkv launch; its capture will
test whether it consumes slot 74's w7 publication.
