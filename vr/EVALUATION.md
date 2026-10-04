# Two-context synthetic evaluation — 04/10/2026

## Result

Two corrected independent launches passed: 60 evaluations per context per launch, 240 successful neural evaluations and 240 validated RGB readbacks total. Existing create-only regression passed. No headset needed; no ReShade or DLSS Super Resolution involved.

```sh
# From testbed root. Build uses locally unpacked verified Zig, not system packages.
tools/zig-x86_64-linux-0.15.2/zig cc -target x86_64-windows-gnu -std=c11 -Wall -Wextra -Werror -O2 vr/feature-probe.c -ld3d12 -ldxguid -o vr/probe/feature-probe.exe
bash vr/run-feature-probe.sh evaluate
# Existing lifecycle check:
bash vr/run-feature-probe.sh create
```

Sources: `feature-probe.c`, `evaluation-probe.h`, `run-feature-probe.sh`. Saved outputs and Proton traces: `logs/evaluation/`.

## Controls and validation

- Two distinct feature handles and allocated parameter blocks; separate 720×720 FP16 colour/output, R32 depth, RG16 motion, uploads/readbacks for each context.
- Stationary synthetic colour gradients: first red-dominant, second blue-dominant; depth gradient, zero motion. These are deliberately NOT geometrically related eye images.
- Alternating evaluation order each batch. Reset both contexts on first batch only.
- Explicit transitions and GPU fence completion before every allocator reuse, readback or resource destruction. Bounded launcher and fence waits.
- Output filled with NaN sentinel immediately before EVERY evaluation. Every RGB component must become finite; dominant channel must vary spatially; mean red/blue dominance must match context. Mean absolute change from original quantised colour must exceed0.00001. Alpha not validated.
- Final sampled means at frame59: context0 RGB≈(0.514899,0.192359,0.134347), mean absolute input change≈0.009092; context1≈(0.130406,0.197725,0.500844), change≈0.002868. These are diagnostic model outputs, not quality scores.
- Negative control compiled from same probe with evaluation call replaced by synthetic Success return. Launcher correctly rejected it:1,555,200 invalid RGB components, no PASS. Saved generated source`probe/negative-control.c` is for audit only; final executable restored from production probe and repeated successfully.
- Parameter destruction statuses and core shutdown checked before PASS; shutdown reports zero live devices. Launcher uses lock/fresh evidence and rejects unhandled exceptions/assertions.

## ABI finding

First attempt used official header's typed D3D12-resource setter, vtable slot6, and evaluation returned0xBAD00005. Existing Linux add-on uses ULL setter slot0 for resource pointers; adopting that exact runtime ABI produced successful evaluations. Float setter slot1 and uint slot3 work here. Public header reference: https://raw.githubusercontent.com/NVIDIA/DLSS/main/include/nvsdk_ngx_params.h ; experimental runtime is NOT guaranteed to match public SDK ABI.

## Limits and next work

This proves repeated direct evaluation and separate input/output resource handling only. It does NOT prove independent temporal history (dominance can survive history leakage), stereo consistency, moving-camera correctness, visual quality, native Linux execution, long-term stability or VR-suitable GPU timing. No timings reported: CPU-fenced synthetic readback is not a VR performance benchmark.

Next stage is minimal per-eye renderer integration: real colour/depth/motion/jitter/reset data, independent contexts, output composition and swapchain ownership/state handling. Neural paths stay in Windows/DX12 through Proton. Existing headset cube executable remains unchanged.

Independent review initially found pooled-channel variation could accept flat colours and ignored parameter-destruction failure. Both fixed; reran strict build, negative control, creation regression and two evaluation launches. Follow-up review recorded below.

Follow-up independent read-only review: PASS; no remaining concrete defects within this limited milestone. GPU responded normally and no probe/cube processes remained after verification.

## Update — rendered scene

See `SCENE.md` for the completed offscreen shared-world two-view stage. Headset hand-off remains outstanding.
