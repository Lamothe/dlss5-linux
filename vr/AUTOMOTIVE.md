# Complete automotive scene — 06/10/2026

Separate `automotive` mode uses the complete user-exported `Downloads/VR/Vehicles/Modern+Minimalist+Automotive+Scene.glb`. Original file is untouched. No downloads. Source and derived assets remain Git-ignored; redistribution rights are not established.

```bash
blender -b -t 2 --factory-startup --python vr/build-city.py -- --automotive
python3 vr/pack-city.py --automotive
bash vr/run-xr-scene.sh on 1440 automotive
# Or start off and toggle with a trigger for a fair source comparison.
```

Left stick moves; right stick smooth-turns/up/down; triggers toggle neural; A/X change lighting; B exits. Interactive mode has no timer. `automotive-smoke` remains bounded. Initial virtual offset is zero; physical tracking remains active. Coarse collisions/navigation use 176 generated envelopes including perimeter walls, not mesh-accurate physics or ground following.

## Import and fidelity

Full evaluated scene: 960 mesh instances, 2,755,306 triangles, 58 materials including one neutral fallback for unassigned primitives. Earlier raw GLB counts did not include every instantiated/evaluated object. No decimation is applied in this import. Extent approximately 38 × 33 m horizontally, 9.8 m vertically; authored transforms/scale retained.

Base-colour textures use each material's selected UV layer and transform (five materials select nonzero UV sets). Neutral fallback materials prevent unassigned geometry being lost. Existing base-colour/factors, supported normals, roughness/metallic scalar values and sorted glass path retained. Four compatible normal maps retained; incompatible UV/transform/scale normal maps are omitted rather than sampled incorrectly.

This is NOT a full physically based renderer: metallic/roughness/AO texture channels, clearcoat, transmission/refraction, realistic reflections and mesh shadows are not implemented. Imported texture atlas resolution/filtering and existing spatial AA still limit fidelity. Scene cameras/lights are not imported into runtime; existing tracked camera and adjustable test lighting are used. Better source geometry does not guarantee photorealism or eliminate aliasing.

## Verification

Strict warning-as-error 720/1080/1440 XR builds and mesh probe build pass. Automotive static/moving GPU readbacks validate visible colour, finite depth and projected camera motion (~499k visible pixels per view); disabled-mesh negative control rejected. Original model and supplied-city regressions, locomotion/collision tests, Python/script syntax and diff whitespace pass. These offscreen tests do not run neural processing or OpenXR.

Independent review caught an omitted interactive timeout exemption. Fixed: automotive sessions are unbounded; automotive-smoke remains timed. Launcher routing check passes.

Headset smoke attempt did not reach application output: Proton logged repeated `VR state wait timeout` during bootstrap, and the bounded attempt timed out. No headset rendering/quality/performance pass claimed. No remaining test processes; NVIDIA responds normally. Results in local ignored `vr/logs/automotive-*.txt`.

Original city/model scenes remain available. Nothing committed or pushed in this round.
