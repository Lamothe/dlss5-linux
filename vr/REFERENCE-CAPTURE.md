# Proton reference capture — 13/10/2026

## Result

**Vendor reference evaluations are now captured and the runtime is proven
deterministic.** Two independent capture runs (`run-a`, `run-b`) of the same
controlled protocol are **bit-identical across all 11 tensors** (manifest
digests equal and independently recomputed). The working VR PoC is untouched.

```sh
# From testbed root.
tools/zig-x86_64-linux-0.15.2/zig cc -target x86_64-windows-gnu -std=c11 -Wall -Wextra -Werror -O2 \
  vr/feature-probe.c -ld3d12 -ldxguid -ld3dcompiler_47 -o vr/probe/feature-probe.exe
bash vr/run-feature-probe.sh capture run-a     # repeat with run-b, run-c, ...
python3 vr/reference-capture-report.py --a vr/logs/reference/run-a --b vr/logs/reference/run-b \
  --report vr/logs/reference/determinism-report.json
```

Sources: `feature-probe.c` (`--capture DIR` mode), `evaluation-probe.h`
(`capture_dump`/`capture_write_file`), `run-feature-probe.sh` (`capture`
mode with label validation and retention under `vr/logs/reference/<label>`),
`reference-capture-report.py` (strict/float comparison). Captures:
`vr/logs/reference/run-a/`, `run-b/` (payloads stay local — `*.bin` is
excluded from git; the digests below are the committed evidence).

## Protocol

Single context (eye 0), 720×720, the deterministic stationary gradient inputs
from `prepare_eye` (opposing-channel FP16×4 colour gradient, R32F depth
gradient, zero motion), `DLSSNR.Reset=1` on frame 0 only, then 7 frames with
`Reset=0`. Eight evaluations per capture. Output is reset to the NaN sentinel
before every evaluation; every captured RGB value is finite (validated by the
60-frame `evaluate` mode, which passes 60,60 with the capture code present).

| File | Bytes | FNV-1a-64 (run-a = run-b) |
|---|---|---|
| input-colour.bin | 4,147,200 | `71236363ed0b69b3` |
| input-depth.bin | 2,073,600 | `d9b9cd9791df8223` |
| input-motion.bin | 2,073,600 | `fb01942564bcb383` |
| output-frame00.bin | 4,147,200 | `865e5c4d2e787b60` |
| output-frame01.bin | 4,147,200 | `bd7192c090cad4cf` |
| output-frame02.bin | 4,147,200 | `d01848c49144f93e` |
| output-frame03.bin | 4,147,200 | `f74c825006b9ef6d` |
| output-frame04.bin | 4,147,200 | `4f7c3081ad7580a1` |
| output-frame05.bin | 4,147,200 | `efec5456ae7b90d8` |
| output-frame06.bin | 4,147,200 | `c2dc1c2f42a9ae73` |
| output-frame07.bin | 4,147,200 | `78ab395e94624add` |

Layout: tight-packed rows (RowPitch padding stripped). Formats: colour/output
FP16×4, depth F32, motion FP16×2. `manifest.txt` in each capture records
protocol, sizes and digests; the report tool recomputes digests from payloads.

## Temporal behaviour (why the frame index matters)

Outputs differ per frame despite static inputs and zero motion — the model
runs a history blend, consistent with its network name (below). Consecutive
frames differ in 66–69% of samples with mean absolute difference ≈
0.0004–0.0006 (FP16 scale). Output vs input colour: mean absolute difference
≈ 0.0066 at frame 0 and frame 7. **Any future comparison must pin the frame
index, the input set and the reset semantics** — the reference is the sequence,
not a single image.

## Dispatch investigation — how the model runs its kernels under Proton

The capture was preceded by instrumentation of the runtime itself:

1. **Host CUDA is present but interposition failed.** `/proc` maps attribution
   shows `/run/host/usr/lib64/libcuda.so.615.71.09` mapped into the probe
   process (the OpenCL/browser mappings on the desktop were excluded by PID
   attribution). A host-side tracer (`vr/capture/cuda-shim.c`) was built and
   injected via `LD_PRELOAD` + `LD_LIBRARY_PATH` — it loaded (28 processes,
   transparent forwarding: the 60-frame evaluation passed unchanged) but
   **intercepted zero CUDA calls and zero `dlsym` lookups**. The runtime's
   entry-point table ("LoadCUDAFunctionTable", resolved via `cuGetProcAddress`
   per its strings) therefore bypasses interposable dynamic-symbol binding
   (PE-level resolution and glibc-versioned `dlsym` defeat the unversioned
   hook). Recorded as failed interception evidence; the shim source is kept.

2. **The model's own logging** (enabled via `__NGX_LOG_LEVEL`,
   `__NGX_LOG_PATH_OVERRIDE`, `__NGX_ENABLE_OVERRIDE_LOG_PATH`) is now
   harnessed (`vr/logs/capture/ngxlog/`). Findings:
   - Network: `CC_Control_History_Blend_Quantize_With_Teacher_honest_tench_
     2026_07_04_22_30_weights`, backbone `crazy-cuckoo`, build v310.8.0;
     `texmode_raw` enabled; `m_gpuArch = 0x190`.
   - Resource model (per feature,720×720): `CG2RWeightHeap` 140.9 MB
     (153 raw tensors), `CG2RWeightUpload` 140.9 MB staging, `CG2RPool`
     59.8 MB, `dlssnr_prev_output` 4.0 MB **2D history texture** — the
     temporal state our native work has not yet modelled.
   - Per evaluation it logs resource pointers, intensity and reset; the
     captured runs show `reset=1` on frame 0 and `reset=0` afterwards.

3. **The NVAPI CuModule family is unavailable** in DXVK-NVAPI:
   `NvAPI_D3D12_CreateCuModule` "Not implemented", four unknown QueryInterface
   IDs, and `NvAPI_D3D12_GetCudaIndependentDescriptorObject` returns
   "Invalid" (status −5 in the model log) — **yet evaluations succeed**. The
   kernel dispatch inside `NGXCubinD3D12` is therefore still *directly
   unobserved*: it is neither the NVAPI CuModule path (unimplemented) nor an
   interposed host-CUDA call path. This is the main open mechanism question.

## What this establishes — and what it does not

Established: a repeatable, deterministic vendor reference dataset (inputs,
per-frame outputs, digests) for the controlled protocol; the temporal/state
contract of that reference; the model's network identity and resource model;
and a precise negative result for symbol-level CUDA interposition.

Not established: kernel dispatch (see above), any intermediate tensor capture,
any **native-vs-vendor comparison** (the gate item's comparison half — it
needs either native input→output coverage or decoder-boundary intermediates),
and any native neural-rendered image. Proprietary reuse/publication rights
remain unresolved.

## Next stage

1. **Pin the dispatch** with a PE-level `nvapi64.dll` QueryInterface forwarder
   (the `nvngx.dll_nrfwd.dll` pattern): log every resolved/called NvAPI method
   around CreateFeature/EvaluateFeature. This is the cleanest interception
   point for the NGXCubinD3D12 path and would unlock intermediate capture.
   Higher `__NGX_LOG_LEVEL` values are a cheap complement.
2. **Run the comparison half of the gate**: with dispatch pinned (or native
   input→output coverage), compare one native evaluation against this
   reference using `reference-capture-report.py --mode float` (tolerance
   reporting per sample), pinning frame index and reset semantics.

Containment unchanged: bwrap + Proton boundaries; captures and logs stay
local (`*.bin`, `vr/logs/` excluded from git); committed evidence is
digests/counts/verdicts only.
