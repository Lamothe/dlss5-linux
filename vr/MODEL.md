# Static GLB model and locomotion — 04/10/2026

Launch with WiVRn connected and headset worn:

```bash
bash vr/run-xr-scene.sh on 1080 model
```

Left stick: head-relative horizontal movement, maximum 1 m/s. Right stick left/right: continuous head-centred turning, up to 60 degrees/s. Right stick up/down: vertical inspection movement at up to 0.5 m/s, eye height bounded 0.25–2.95 m. Either trigger: neural toggle for both eyes. A/X: lighting. Bounds only, no furniture collision. Keep a clear physical play space; stop if uncomfortable. Existing sphere/room modes remain available.

Test.glb imported read-only in fresh Blender with automatic scripts disabled. Evaluated meshes bake current armature/morph pose; no animation. Skipped unmaterialled Icosphere helper. 83,773 triangles, 24 materials, 4096 RGBA atlas. Character placed with feet at floor, centred x=0,z=2.35, clear of table, height1.835m. GLB and original blend unchanged. Conversion scripts export-model.py and pack-model.py, assets under assets/test-model, runtime probe/model.bin. Unique nonempty glTF material names required; unsupported mappings rejected. Base-colour factors and alpha modes sourced directly from GLB metadata. Texture sampler repeat/clamp/mirror respected. Unused materials skipped.

Limitations: BLEND approximated by alpha cutout at0.5; MASK retains source cutoff. No texture normal/metallic/roughness maps, specialised skin/eye shaders, animation, texture mipmaps or mesh shadows. Atlas downsamples textures to roughly810pixels/material (more reduction for repeated ranges). Not a general-purpose full glTF implementation, nor an anti-aliasing improvement. Do not expect Blender-identical appearance.

Mesh raster pass writes source colour/depth/motion before neural evaluation. Procedural depth snapshot clips occluded geometry; independent mesh D32 depth handles overlapping mesh. Combined virtual/physical camera history drives motion; snap/recenter/focus reset neural history, continuous movement preserves it. Physical OpenXR compositor poses remain unmodified.

Verified: strict Windows C builds720/1080/1440; Python compilation and shell syntax. Offscreen mesh GPU readbacks:12412static/12416moving covered pixels, finite source values, motion max error0.0172px; synthetic alpha discard, procedural occlusion and near/far mesh order passed. Disabled-mesh negative control rejected. Existing sphere GPU neural regression passed24evaluations/outputs. Native locomotion tests validated direction atmultipleheadings, bounds, snap/rearm and deterministic smoke with adversarial input. Independent reviews found and corrected movement basis, smoke input coupling, material alpha metadata, GPU upload failure lifetime and importer/atlas edge cases; final focusedreviewPASS.

Headset attempt loaded model and pipelines but trackinginvalid; later retry blockedbootstrap and boundedtimeout. No model headset frame pass claimed. Existing1080sphere timings DO NOT establish model performance. No leftover test processes;GPUresponsive. Logs under logs/room/mesh-* and model-*.

Smoke mode `model-smoke` schedules movement frames100–129, lighting300, snap350, neural toggles200/400;600submittedframes required. No headsetsmokepassyet. To rerun offscreen controls: `bash vr/run-feature-probe.sh mesh` / `mesh-negative`.

Placement/vertical fix: actual thumbstick response confirmed by user. Moved model forward of table; rear model bound Z2.523 versus table front Z3.0 (0.477m gap). Host tests of actual locomotion functions pass up/down, speed, height bounds, inactive/reactivation and diagonal suppression; all three strict builds and offscreen mesh GPU regression pass. Independent review PASS. Headset verification of these latest changes remains outstanding. Test source probe/vertical-test.c.
