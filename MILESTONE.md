# DLSS5 Linux migration milestones

Updated 26/10/2026. This document defines the Linux project, independently of the later VR integration project. Status and recorded evidence remain unchanged: L1 is in progress; L2 and L3 are not yet demonstrated. See `vr/NATIVE-MIGRATION-GATE.md` and `vr/CAMPAIGN-HANDOVER.md` for the latest evidence-gate review.

## L1 — Native execution contracts (in progress)

Recover and validate kernel interfaces, weight bindings, memory layouts and data flow against repeatable vendor captures with positive/negative controls. The published pre-decoder contract baseline is slots 24–91 (68/156 launches), through commit `33fa071`. Slot 91's raw captures, traces and replay run JSONs remain local ignored evidence. Decoder investigations are recorded separately. Per-slot replay is not an end-to-end evaluator.

## L2 — Complete native evaluator (not yet demonstrated)

Assemble preprocessing, the full graph, decoder/output conversion and the temporal feedback/reset state into a native Linux evaluator. Drive it with live inputs, not captured intermediate activations. Windows/Proton may supply reference evidence but must not be required by the evaluator.

## L3 — Migration acceptance (not yet demonstrated)

Demonstrate native neural-rendered image output and validate it against the reference across repeated frames, reset/history transitions and changed inputs. Record supported hardware, resolutions, numerical/visual differences, performance, stability and remaining limitations. Do not declare completion from API success, kernel counts or a Proton VR demonstration. Proprietary runtime provenance and reuse/distribution rights remain separate unresolved constraints.

## Later, separate project — VR integration

Only after the native migration is validated, integrate the evaluator into the existing Genesis Linux Vulkan/OpenXR host through its neural seam. Per-eye state, stereo correctness, headset behaviour and VR performance belong to [dlss5-vr](https://github.com/Lamothe/dlss5-vr), not Linux migration acceptance.

## Archived Proton VR demonstration

The following is historical evidence from the predecessor project, not a completed Linux migration milestone.

# Milestone 1 — interactive neural-rendered PC VR

Recorded 05/10/2026. Proposed tag: `milestone-1-neural-vr`.

## Demonstrated

- RTX 4070 Ti SUPER on Linux, using an unofficial Windows neural runtime through Proton (DX12 translated to Vulkan).
- OpenXR stereo rendering through WiVRn to a Meta Quest 3 over Wi-Fi; no SteamVR required.
- Separate neural feature handles and resources for each eye; actual headset views/poses used to render inputs.
- Procedural room, static textured GLB character, adjustable lighting and live neural off/on toggle.
- Left-stick horizontal movement; right-stick horizontal 30-degree snap turning and vertical inspection movement (up to 0.5 m/s, virtual eye height bounded 0.25–2.95 m).
- Checkerboard footprint filtering and final spatial edge anti-aliasing.
- User confirmed scene, stereo/head tracking, controls, model placement and vertical movement. User prefers 1440 per eye: sharper, apparently responsive; neural pass visibly improves skin appearance. Hair aliasing remains noticeable.

Preferred launch with WiVRn connected and headset worn:

```bash
bash vr/run-xr-scene.sh on 1440 model
```

## Evidence and limits

Earlier simple-scene benchmark: 1080 per-eye stereo GPU mean 8.03 ms, ~89.6 submissions/s; 1440 mean 11.64 ms, ~76.3 submissions/s. These are NOT current model/AA measurements and NOT headset latency. See `vr/TIMING-1080-RESULTS.md` and `vr/TIMING-RESULTS.md`.

Latest changes passed strict Windows builds at 720/1080/1440, offscreen GPU mesh checks, all-pixel numerical AA checks/negative controls, host vertical-movement tests and independent reviews. Successful feature evaluation and user observation do not establish independent temporal history, broad stereo correctness or long-term stability.

Materials are simplified: shared atlas reduces texture resolution; no normal maps, mipmaps, skeletal animation, mesh shadows or physically accurate skin shader. Hair uses hard alpha cutout. Added neural detail may be inferred rather than recovered from source textures. Movement bounds the room but does not collide with furniture.

This is the experimental neural component commonly referred to as DLSS 5, not an official NVIDIA-supported RTX40/VR integration or a complete DLSS feature suite. Proprietary patched-runtime provenance remains unverified. Containment is not complete security isolation.

## Repository boundary

Source, scripts, test source and result summaries only. Excluded: proprietary/experimental DLLs, model and texture assets, generated binaries, downloaded tools/SDK headers, Proton prefixes/copies, runtime payloads, raw logs and caches. Original Test.blend/Test.glb remain untouched and outside this repository.

A clone alone is not runnable yet: see CHECKPOINT.md. Keep the existing local testbed for dependency state. Do not upload the excluded payloads, even to a private repository, without appropriate rights and a separate review. No licence grant for original source is selected at this milestone.
