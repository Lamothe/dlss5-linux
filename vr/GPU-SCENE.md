# GPU scene generation — 04/10/2026

## Completed

`gpu-scene.hlsl` computes the same shared-world two-view ray scene as the CPU renderer. `gpu-scene.h` creates a D3D12 compute pipeline, per-eye UAV descriptor tables and input readbacks. `feature-probe.c --gpu-scene` runs12frames per eye through the existing independent neural contexts.

Build from project root:

```sh
tools/zig-x86_64-linux-0.15.2/zig cc -target x86_64-windows-gnu -std=c11 -Wall -Wextra -Werror -O2 vr/feature-probe.c -ld3d12 -ldxguid -ld3dcompiler_47 -o vr/probe/feature-probe.exe
bash vr/run-feature-probe.sh gpu-scene
```

All feature-probe builds now require `-ld3dcompiler_47`, including older modes. Shader source is loaded relative to the executable from`../gpu-scene.hlsl`; no external compiler/package installed. Zig's C D3D12 aggregate-return wrappers enabled via`WIDL_C_INLINE_WRAPPERS`.

## Verification

Two tightened launches passed48neural evaluations and48validated outputs total. Every GPU colour/depth/motion component checked finite/in range; moving frames have positive X and zero Y motion at all pixels. Sparse depth/motion samples checked against independent CPU intersection oracle, tolerances1e-5 normal-depth and0.002pixels motion. NaN poisoning before EVERY dispatch prevents stale/partial writes passing. Upload poison resources are separate immutable allocations, not the host comparison mirrors.

Negative control skipped scene dispatch while still calling neural evaluation. Both API calls returned Success, but the input validator rejected2,073,600invalid colour components and suppressed PASS. Source/executable restored, strict compilation and subsequent run passed. Create-only and60-frame-per-eye synthetic modes rerun successfully. Input/output snapshot PPM dimensions/lengths validated; CPU/GPU source images numerically compared (mean8bit error below1/255 each frame0/11 view).

Logs/control/PPM snapshots:`logs/gpu-scene/`. No headset or system configuration changed. Independent review initially found stale-input validation gap; poisoned inputs and independent sampled geometry checks fixed it. Follow-up read-only review PASS. No probe/cube process remains; GPU responds normally.

## Limits / next milestone

This is GPU scene generation and neural evaluation, NOT an OpenXR frontend or headset test. No GPU timings claimed: diagnostic input/output readbacks, CPU validation and per-batch fences remain and must not be taken as production VR performance. Fixed square FOV/camera trajectory, no runtime poses. Sparse checks are not full-image numerical equivalence or proof of proprietary motion semantics/temporal isolation.

Next implement runtime-selected DXGI adapter (OpenXR-required LUID), tracked per-eye poses/FOV and GPU composition into supported OpenXR swapchains. In particular display-encoded FP16 proxy output is not directly interchangeable with a linear FP16 compositor texture. Need explicit format/colour handling, acquire/wait/release and correct resource state restoration. Test frontend without neural first, then neural toggle; never assume this diagnostic loop is comfortable interactive VR.

## Update — OpenXR frontend

See `OPENXR-SCENE.md`. Frontend built and tracked offscreen control passed; runtime bootstrap currently blocks both it and official sample, so no new headset pass.
