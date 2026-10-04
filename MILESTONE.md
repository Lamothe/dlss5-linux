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
