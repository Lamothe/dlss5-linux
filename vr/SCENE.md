# Rendered two-view scene — 04/10/2026

## Implemented and verified

`scene-probe.h` adds an offscreen CPU ray renderer of a fixed shared world: two coloured spheres, checkerboard floor, lit back wall. Geometry uses metres. Parallel camera views are separated by64mm and translate in X between frames. Each720×720 view has90-degree square FOV.

`feature-probe.c --scene` processes12 frames per eye through existing separate neural contexts. The scene supplies display-referred sRGB-encoded FP16 colour, normal D3D projected depth (near0.1m/far20m) and previous-minus-current pixel motion vectors. Initial reset only; no jitter. Separate per-eye resource/parameter state and fenced uploads/evaluation/readback remain in place. CPU rendering is intentional for input correctness testing, NOT a performance architecture.

Build from project root with existing local compiler:

```sh
tools/zig-x86_64-linux-0.15.2/zig cc -target x86_64-windows-gnu -std=c11 -Wall -Wextra -Werror -O2 vr/feature-probe.c -ld3d12 -ldxguid -o vr/probe/feature-probe.exe
bash vr/run-feature-probe.sh scene
```

Scene launcher is bounded300seconds; original modes remain60seconds. Earlier60-frame attempts exceeded their timeouts (no PASS); shortened12-frame run completed with checked teardown. Slow CPU rendering/readback means NO FPS or GPU-pass timing inference is valid.

## Evidence

`logs/scene/run-1.txt`:24successful evaluations and24finite, spatially varying, changed RGB readbacks. Both parameter destruction calls and core shutdown succeeded, zero live devices. At frame11, mean absolute input/output difference≈0.010965(left) and0.011572(right). All518,400pixels per eye had nonzero computed motion after initial frame. Frame0 zero motion.

`logs/scene/scene-eye{0,1}-frame{0,11}-{input,output}.ppm`: binary720×720PPM snapshots. File headers and pixel lengths checked. Input/output differs in1,142,595–1,394,258of1,555,2008bit components across the four comparisons. Left/right inputs differ in161,698components atframe0 and162,092atframe11. These are diagnostic comparisons, not visual-quality scores or stereo consistency proof. No model visual assessment has been performed by the assistant.

Original create-only and60-frame synthetic evaluation modes were rerun successfully; evidence in`logs/scene/*regression*.txt`. Analytical checks verify normal-depth near/far endpoints and previous-minus-current motion sign/magnitude. Float-to-half conversion now handles signed values/subnormals rather than only positive synthetic values.

Independent read-only review PASS: projection, motion convention, depth, signed-half conversion, fence/map lifetimes and limited claims checked. No probe/cube processes remain; GPU responds normally.

## Still outstanding

No headset presentation, tracked poses/controllers, runtime eye FOV or real GPU renderer yet. No independent temporal-history or stereo consistency proof. NR evaluation on moving inputs succeeds, but that is not validation of the proprietary model's exact motion/depth semantics or visual correctness. It remains Windows/DX12 through Proton.

Next: a minimal standalone C OpenXR frontend (rather than building upstream hello_xr, whose DirectX CMake path assumes MSVC). Requirements-first DXGI adapter selection must honour runtime LUID and minimum feature level before creating device/contexts. Eye poses/FOV at predictedDisplayTime replace test camera. XR images are runtime-owned: acquire/wait, transition RENDER_TARGET to COPY_DEST, submit work on bound queue, restore RENDER_TARGET before release. FP16 proxy output must be converted to a supported compositor format (copy alone cannot convert/gamma encode). Follow shouldRender/session-state handling and strict ownership cleanup.

Official state/device requirements verified from:
https://raw.githubusercontent.com/KhronosGroup/OpenXR-Docs/main/specification/sources/chapters/extensions/khr/khr_d3d12_enable.adoc

Do not put this CPU/readback diagnostic loop directly in a headset: it is not latency suitable. A GPU renderer or pre-rendered bounded presentation control is needed before comfortable tracked interactive testing.

## Update — GPU renderer

See `GPU-SCENE.md` for the completed GPU scene stage and current build command (now also links `d3dcompiler_47`). OpenXR frontend remains outstanding.
