# Supplied city — 05/10/2026

Separate scene from the assembled `city` and original `model`.

```bash
blender -b -t 2 --factory-startup --python vr/build-city.py -- --supplied
python3 vr/pack-city.py --supplied
bash vr/run-xr-scene.sh on 1440 supplied-city
# Trigger switches neural processing; B exits. No interactive timer.
```

Source: local `Downloads/VR/Scenes/street_city_7_for_games_free.glb`. Original untouched. Embedded metadata credits dasy444, https://sketchfab.com/dasy444 and https://sketchfab.com/3d-models/street-city-7-for-games-free-493a69b451284ff88346c7b3e4e1b5a7 ; licence field says SKETCHFAB Standard, https://sketchfab.com/licenses . User-supplied for local testing; no redistribution permission established. Source, textures and derived payload remain Git-ignored. No new downloads.

111 mesh objects, 92,967 triangles, 12 materials. Full exported geometry preserved, no decimation. Authored scale retained: approximately 54 × 53 m horizontal extent, height ~14 m including below-ground geometry. Payload renderer bounds X -38.32..15.39, Y -1.75..12.43, Z -26.41..26.49. Initial virtual offset X=10 m at street ground near Blender X=-10; original head height remains tracked. Left stick moves, right stick smooth-turns / raises or lowers, trigger toggles neural, A/X changes lighting, B exits. Uses sky/fallback-ground shader, imported source surfaces in depth-tested mesh path.

Coarse collisions: 52 compact tall mesh AABBs plus four perimeter envelopes. Broad ground and roof planes excluded. Not mesh-accurate collisions or ground-following; source objects split by material can cause coarse blocking. Physical head movement is never constrained. Virtual navigation limits are derived from the four generated perimeter slabs, not the assembled street bounds.

Materials reuse current atlas/shading pipeline: base colour/factors, supported normal maps, roughness/metallic scalar values. No full PBR, texture-channel roughness/metallic, AO, mesh shadows/reflections. This asset is not more geometrically detailed than our 3-million-triangle assembled city. Replacing the environment does not promise to eliminate aliasing. Ground-specific mip filtering in the assembled street does not automatically apply to these imported materials.

## Verification

Strict warning-as-error 720/1080/1440 builds. Dedicated `supplied-city` offscreen mesh probe validates actual colour coverage, finite outputs, depth and projected camera motion at the initial street position; ~447k static/~445k moving pixels. Disabled-mesh negative rejected. Existing city and original model regressions retain synthetic alpha/glass/depth oracles. Locomotion tests pass. Script/Python syntax and diff whitespace checks pass.

```bash
bash vr/run-feature-probe.sh supplied-city
bash vr/run-feature-probe.sh supplied-city-negative
bash vr/run-feature-probe.sh city
bash vr/run-feature-probe.sh mesh
bash vr/test-locomotion.sh
```

No new headset submission/visual-quality/performance claim. Existing neural renderer and per-eye contexts retained. Headset confirmation needed; interactive launch deliberately left to user to avoid disrupting their session.

Independent review caught shared old-street navigation limits blocking the new spawn. Fixed with per-scene limits derived from perimeter boxes; tests cover movement from X=10, beyond old bounds and perimeter stopping. All resolution builds and supplied positive/negative probes rerun successfully.
