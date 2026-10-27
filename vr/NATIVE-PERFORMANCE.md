# NATIVE-PERFORMANCE — pipeline performance campaign

Status: living document. The CPU path stays the deterministic reference
throughout; every optimisation keeps known-answer parity (bit-identical for
the CPU work; documented per-family tolerance only where GPU float arithmetic
allows contraction). Speed numbers are real measurements; with synthetic
weight presets the OUTPUT is arbitrary, so no image-quality claim is made.

## P1 — the profile (28/10/2026)

Method: `demo-native --profile` times each plan row at the `NetworkRun`
dispatch seam (a forwarding wrapper — no arithmetic added) and accumulates
per-kind wall time. Machine: AMD Ryzen 7 7800X3D (8 cores / 16 threads),
single-threaded build `-O2 -ffp-contract=off`, 64x72 frame, 'small' plan
(76 rows, padded 320x320 extent), `seeded` preset, one frame.

Baseline: **38,966 ms/frame** (the demo's earlier 4-frame run: 37.4 s/frame
mean; run-to-run noise ~4%).

| Kind | Rows | ms | share |
|---|---|---|---|
| VitFfnExpand | 8 | 8,739 | 22.4% |
| VitFfnContract | 8 | 7,767 | 19.9% |
| VitQKV | 8 | 7,389 | 19.0% |
| VitProjection | 8 | 2,014 | 5.2% |
| SplitQKVAttn | 4 | 1,976 | 5.1% |
| PostBlock | 1 | 1,718 | 4.4% |
| Swin1H | 4 | 1,480 | 3.8% |
| Swin2H | 4 | 1,123 | 2.9% |
| Swin4H | 4 | 977 | 2.5% |
| Swin8H | 4 | 1,187 | 3.1% |
| SplitFfwd | 4 | 1,213 | 3.1% |
| SplitFfwdProj | 4 | 569 | 1.5% |
| SplitProj | 3 | 426 | 1.1% |
| SplitProjPool | 1 | 146 | 0.4% |
| DecInputUpsample | 1 | 157 | 0.4% |
| SplitFinalHead | 1 | 129 | 0.3% |
| VitAttention | 8 | 51 | 0.1% |
| PreBlock | 1 | ~1,000 | ~2.6% |

Findings: the three ViT projection rows (Expand + Contract + QKV) are
**61.3%** of the frame; the projection-matmul class (those three +
VitProjection + SplitQKVAttn + the swin rows' inner linears) is ~72%.
The attention kernels themselves are cheap (VitAttention 0.1%). The plan's
cost is dominated by `tokens @ W^T` projections over the padded 320x320
token grid.

Consequence for the optimisation order: (P2) thread the projection matmul
first (output-row parallel with fixed inner accumulation order = bit-identical
parity), (P3) then SIMD/vectorisation where P2 leaves headroom, (P4+) then
GPU compute per family with measured parity and speedup.

## P2 — deterministic CPU threading (28/10/2026)

`vr/native-parallel.hpp`: opt-in process thread count (default 1 = the serial
reference) and a spawn-and-join `parallel_for` with an inline fallback on
spawn failure. Threaded sites split the independent outer index (tokens or
rows) while every element keeps its single-threaded key-ordered accumulation,
so results are **bit-identical at every thread count** — proven by memcmp
parity tests at 1/2/3/4/8/16 threads (2,479 checks at -O2/-O0) and by
byte-identical demo artefacts. Sites: `matmul_e4m3`, `vit_ffn_expand`,
`vit_residual_linear`, `vit_qkv`'s raw projection, `swin_linear`,
`split_projection_value` and `split_ffwd`. Serial remainder (documented):
`split_ffwd_proj`/`split_qkv_attn`'s status-returning inner calls and the
small per-element families.

Measured (same run conditions as P1, `seeded` preset, 64x72 frame, one
frame; 7800X3D = 8 cores / 16 threads; SMT adds contention here):

| Threads | ms/frame | speedup |
|---|---|---|
| 1 (reference) | 36,268 | 1.00x |
| 8 | 16,306 | **2.22x** |
| 16 | 16,909 | 2.15x |

Output bit-parity verified (cmp of the artefacts). The scaling is capped by
memory bandwidth: the projection rows stream ~100 MiB of e4m3 weight records
per frame, so the matmuls are bandwidth-bound well before 8 cores saturate
compute. This is the honest CPU result; the GPU stage (VRAM bandwidth) is
where the larger win lives.

The demo exposes this as `--threads N` (default 1 = the reference; 0 =
hardware concurrency) and `--profile` (the per-kind breakdown at the dispatch
seam).
