# NATIVE-VULKAN — G1: the Vulkan projection matmul

Status: G1 published (the first GPU slice). The CPU path in the family
headers stays the deterministic reference; every GPU slice is parity-tested
against it. With synthetic weight presets the OUTPUT is arbitrary — the
speed numbers below are real measurements, image-quality claims are not
made.

## Scope

`vr/native-vulkan.hpp` (compute-capable device selection, device-local
buffers with staging transfers, timestamp queries, one-shot dispatches) and
`vr/shaders/matmul_e4m3.comp` (the projection matmul `out[t][n] =
sum_k in[t][k] * w[n][k]` with `precise` key-ordered accumulation). The
shader source is GLSL compiled by the runner with glslc; no compiled
artefacts are stored in the repository. Plain Vulkan 1.1 compute, no vendor
extensions.

## CPU-vs-GPU parity (the tolerance, stated exactly)

Scoped to FINITE operands (NaN payloads are not bit-stable across devices):

- **Contractual: bit-identical.** When every intermediate product and
  partial sum stays within the fp32 normal range, the GPU result is
  bit-identical to the CPU reference (`matmul_e4m3`'s key-ordered
  accumulation; `precise`/NoContraction on the GPU, `-ffp-contract=off` on
  the CPU). Proven at tokens 1/2/255/256/257/300 including the dispatch
  padding boundaries, and on a 100x1024x4096 run.
- **Measured (driver-dependent, NOT contractual): denormal flushing.**
  Vulkan leaves denormal handling unspecified unless DenormPreserve is
  requested. On the dev host's RTX 4070 Ti SUPER driver, every denormal
  INTERMEDIATE is flushed to zero, so the GPU result can diverge from the
  CPU reference even when the CPU's final value is normal: a sum of eight
  2^-129 products gives CPU 2^-126 (normal) but GPU +0. Both divergence
  shapes are pinned in the test so the behaviour is visible, not assumed.
  A driver update may change it (full parity would follow).

## Measured speed (RTX 4070 Ti SUPER; shape 100 x 1024 x 4096)

| Path | ms | vs cpu-1t |
|---|---|---|
| cpu-1t (`matmul_e4m3`, the reference) | 1,578.8 | 1.0x |
| cpu-mt (hardware threads) | 201.1 | 7.8x |
| **gpu kernel** (timestamped) | **2.54** | **622x** |
| gpu staging upload | 2.0 | |
| gpu staging download | 5.6 | |

The GPU kernel time varies with clock state (15.2 ms in a cold-clock run
vs 2.54 ms warm); both are reported by the suite. Upload/download are
staging transfers measured separately — a later slice keeps the weights
resident (they are static per run) so only the output download remains in
the frame path.

## G2 — the eval-dispatch GPU backend (30/10/2026)

The eval dispatch accepts an optional `ProjectionExecutor`
(`EvalContext::projection`): the vit projection rows'
raw projection (`out[t][n] = sum_k in[t][k] * w[n][k]`) runs through the
executor — the Vulkan backend above — while ALL post chains
(`vit_ffn_expand_post` / `vit_residual_post` / `vit_qkv_post`) and every
other row remain the CPU reference's operations. Null (the default) keeps
the pure CPU path. Wired rows: VitFfnExpand, VitFfnContract/VitProjection
and VitQKV (three projections assembled into the reference's interleaved
layout).

**Complete-network parity: byte-identical.** The runner's G2 gate runs the
whole network twice through image_evaluation — CPU reference and --gpu —
and requires identical artefacts (bit-identical where every intermediate is
in the fp32 normal range; the denormal tolerance above applies). The gate
also asserts the GPU actually ran (no silent CPU fallback). Per-row seam
tests cover the three rows independently (71 checks at -O2/-O0).

**Measured end-to-end (RTX 4070 Ti SUPER; 64x72 frame, kSmall plan, 8 CPU
threads):** 16,303 ms/frame CPU vs **14,242 ms/frame GPU-backend = 1.14x**.
The GPU row cost per frame: kernel 223 ms, e4m3 decode 323 ms, staging
upload 135 ms, download 51 ms — per-call staging dominates (G3 weight
residency removes the decode+upload of the static weights). The remaining
frame time is the CPU rows (split/swin families) and the pre/post blocks;
the GPU families grow one slice at a time.

## Provenance

No proprietary code, payload, capture or binary is used or redistributed;
no vendor runtime or capture probe exists in this implementation. The
kernels are original work derived from the public reference below.

MIT License

Copyright (c) 2026 Mauri de Souza Meneguzzo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Reference: https://github.com/mauri870/dlssnr-pytorch at commit
3beef7935a8ef976b47ab78b906e75600c52d2de (MIT, Copyright (c) 2026 Mauri de
Souza Meneguzzo). Pinned public reference used for the parameter semantics
and contracts mirrored by the native implementation; no source is copied.

Non-vendor statement: this is an independent native implementation. It does
not implement, wrap or interoperate with DLSS/DLSSNR runtime code; no
vendor parity, equivalence or compatibility claim is made. The synthetic
weights make the visual output arbitrary; the measured speedup demonstrates
the pipeline, not image quality.
