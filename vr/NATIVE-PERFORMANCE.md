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

## G1 — Vulkan projection matmul (29/10/2026)

The first GPU slice (vr/NATIVE-VULKAN.md): the hot projection family on
Vulkan compute with `precise` accumulation. CPU-vs-GPU results are
**bit-identical** on normal-range intermediates (contractual; proven across
dispatch padding boundaries and a full-size run) with one measured,
driver-dependent divergence class (denormal intermediates flush to zero;
both shapes pinned in the test). Measured on the RTX 4070 Ti SUPER at
100x1024x4096: cpu-1t 1,578.8 ms, cpu-mt 201.1 ms, **gpu kernel 2.54 ms =
622x vs the single-thread reference** (cold-clock runs measure up to 15 ms;
both are reported). Staging upload 2.0 ms / download 5.6 ms measured
separately (a later slice keeps weights resident).

Next GPU slices: wire the projection into the eval dispatch as an optional
backend (CPU reference intact), then the split/attention and swin families,
weight residency, and end-to-end per-frame GPU timing against the CPU
baselines above.

## G2 — the eval-dispatch GPU backend (30/10/2026)

The optional ProjectionExecutor seam (CPU reference intact and default):
the vit rows' raw projections run on the GPU; all post chains and rows stay
on the CPU. Complete-network artefacts are byte-identical (the runner's G2
gate; it also asserts the GPU actually ran). 71 checks at -O2/-O0.
End-to-end at 8 CPU threads: **16,303 -> 14,242 ms/frame (1.14x)** — the
GPU row cost is staging-dominated (kernel 223 ms vs decode+staging 509 ms
per frame), which G3 weight residency removes. G1's per-kernel result
stands (622x vs cpu-1t on the projection).

## G3 — the split/attention GPU family (30/10/2026)

The SplitProjectionSite (one hoisted executor call per row; CPU post
chains) covers SplitFfwdProj/SplitProj/SplitProjPool/SplitFinalHead — 44%
of the frame profile. 86 checks at -O2/-O0; the complete-network CPU-vs-GPU
gate stays byte-identical with these rows on the GPU. Staging still
dominates the GPU row cost; G4 residency is next, then the swin family and
the end-to-end frame numbers.

## G4 — the swin projection seam (30/10/2026)

The SwinProjectionSite + ProjectionExecutor::matmul_f32 seam (swin_linear's
expand/QKV projections; strided gather/scatter; bit-identical on
grid-exact products) threads through the swin composition with CPU
defaults everywhere. Eval wiring is deliberately deferred to G5 weight
residency: per-window GPU calls (16 tokens) would thrash under per-call
staging. 95 checks at -O2/-O0. Measured G3 numbers stand; the complete
network is unchanged by this slice.

## G5 — weight residency (30/10/2026)

Persistent weight buffers with content-hash verification (the pointer key
alone is unsound for recycled per-call record storage). First sight uploads
once; verified hits bind the resident buffer and skip the e4m3 decode.
Measured (64x72, kSmall, 8 threads, RTX 4070 Ti SUPER):

| Path | ms/frame | vs cpu-mt |
|---|---|---|
| cpu-mt reference | 16,303 | 1.00x |
| G2 per-call staging | 14,242 | 1.14x |
| G5 weight residency | 12,380 | 1.32x |

GPU totals per frame (G5): kernel 723 ms, decode 885 ms (first-sight AND
hash-mismatch re-decodes), staging upload 89 ms, download 177 ms. 116
checks at -O2/-O0.

Honest attribution: the demo's record store uses 16 round-robin slots that
recycle per-call storage, so most fetches are content-hash MISMATCHES and
re-decode/re-upload — the resident path therefore measures only 1.15x over
G2's per-call staging here. The full residency benefit requires stable
pointer-to-content record storage (a per-key record store); that is part of
the G6 wiring slice along with the swin/split eval rows. A stable-key
provider pays the hash per hit and never re-uploads.

## G6 — full swin wiring + the stable record store (28/10/2026)

Two deliverables, one measured surprise.

1. **The swin eval rows are WIREABLE and parity-verified** — eval_pre_block,
   the swin_downsample row, the swin_layer_upsample row and the swin_core
   row each construct a SwinProjectionSite (fixed per-window sizing: the
   swin windows are always 64 rows, so the site scratch is 320*channels
   floats) and route their swin_linear projections through the executor.
   The complete-network CPU-vs-GPU parity gate stays byte-identical with
   the wiring enabled.

2. **The per-key stable record store** (native-demo-support.hpp): one
   lazily-allocated buffer per record key, stable after first fill, so the
   Vulkan weight cache serves hits without re-upload and the e4m3 decode
   collapses: 885 -> 454 ms/frame.

**Measured surprise (honest):** with the swin rows' GPU site ENABLED the
end-to-end frame time REGRESSES: 14,418 ms/frame vs 12,380 (G5) — the
per-window calls (64-token windows; ~5k matmul calls per frame across the
swin rows) thrash the per-call input/output staging: 3,337 ms staging
upload + 5,549 ms download per frame versus the 1,229 ms of kernel time.
The per-window transfer overhead dominates even with resident weights.
The swin rows are therefore OPT-IN (`EvalContext::swin_gpu`, default
false): the wiring and its parity tests land, the CPU kernels stay the
default for these rows until a batched seam exists.

**The recorded fix (G7):** an offset-dispatch seam — upload each swin row's
map ONCE, run the per-window matmuls with device-local in/out offsets (two
extra push constants), download ONCE. The products per window are
unchanged (bit-parity preserved; only the transfer granularity changes).
Until then the measured configuration is: vit + split rows on GPU (hoisted,
few large calls), swin rows on CPU, stable record store resident —
11,717 ms/frame end-to-end (1.39x vs cpu-mt; 3.22x vs cpu-1t), parity
gate byte-identical, 615,365-check eval suite green.

| Path | ms/frame | note |
|---|---|---|
| cpu-mt (8 threads) | 16,303 | P2 reference |
| G5 (vit+split GPU, staging) | 12,380 | per-call weight staging |
| G6 swin-wired (measured) | 14,418 | REGRESSION: per-window transfer thrash |
| G6 default (swin on CPU) | 11,717 | stable store resident (decode 885->455 ms) |

## G7 — CPU: branchless decode, bit-exact SIMD, split-QKV hoist, threading
(28/10/2026)

Four changes, all bit-exact on the CPU reference (verified by the suites at
-O2/-O0 and the complete-network CPU-vs-GPU parity gate, byte-identical):

1. **Branchless e4m3 decode** (native-pre-post.hpp): the ldexp/branch form
   is replaced by exact bit construction (normal: sign | (e+120)<<23 |
   m<<20; subnormal: bits((float)m) - (9<<23); signless canonical NaN).
   Exhaustive: all 256 codes match two independently derived oracles.
   Measured: 2.05x on the decode-bound matmul loops.
2. **Bit-exact AVX2 dot8 matmul** (native-matmul-simd.hpp): eight output
   rows per key-loop pass, each lane keeping the scalar key order (mul,
   then add) — results bit-identical at any width. Runtime-gated
   (__builtin_cpu_supports) with target attributes (no -march needed);
   size-gated at c_in >= 256 (measured: 1.9x at 512, 1.8x at 1024; SLOWER
   below 256, hence the gate). decode8 is swept exhaustively (all 256
   codes) against e4m3_from_bits.
3. **Split-QKV projection hoist** (native-eval.hpp): the row's per-token
   projections become one batched [tokens, 512] x [1536, 512] matmul (the
   kernel's existing products branch; the padded zero-input projection
   through an e4m3 matrix is exactly +0). Bit-exact with the CPU fallback
   (split_matmul, the split family's 2^29 token envelope); the executor
   path carries the documented normal-range tolerance.
4. **Threading the P2 serial remainder** (native-split-swin.hpp):
   split_ffwd_proj's token loop and split_qkv_attn's window loop now run
   parallel_for with a recorded failure status (the earlier serial loops
   returned mid-flight). Bit-identical at any thread count — pinned by the
   1-vs-{2,3,8,16}-thread memcmp parity tests in native-parallel-test.cpp
   (NaN-poisoned outputs).

Measured (RTX 4070 Ti SUPER, 8 CPU threads, 64x72 frame, kSmall):

| Path | ms/frame | vs cpu-1t (37,700) | vs cpu-mt (16,303) |
|---|---|---|---|
| G6 default | 11,717 | 3.22x | 1.39x |
| G7 | 7,855 | 4.80x | 2.08x |

Row-level: SplitQKVAttn 1,828 -> 89 ms (hoist + threading), SplitFfwd
199 -> 73 ms. The remaining CPU profile is the swin family (79%: pre/post
blocks + the four swin row kinds stay on the CPU per G6's opt-in finding).
Check counts: pre-post 454, vit 662 (incl. the 256-code decode8 sweep),
split-swin 34,272, parallel 2,479, eval 615,365 — all at -O2 and -O0.
