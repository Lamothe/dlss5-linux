# City street test — 05/10/2026

A static, metre-scale street for a larger neural VR test. Five parked cars, six pedestrians (two avatar sources repeated), 54 apartment façade bays including transverse end frontages, road markings and footpaths. Roughly 64 m long, 18 m wide; about 1 million triangles, 37 materials, 8192² atlas. Separate from the user's character/room asset: nothing overwrites Test.blend, Test.glb or model.bin.

## Run

Connect WiVRn and wear the Quest with tracking active:

```bash
bash vr/run-xr-scene.sh on 1440 city
# Automated 600-frame toggle/lighting/movement smoke:
bash vr/run-xr-scene.sh on 1440 city-smoke
```

1080 and 720 remain available. Left stick moves, right stick left/right smooth-turns at up to 60°/s, right stick up/down changes height. Trigger toggles neural rendering; A/X changes daylight/warm lighting. Bounds are X ±10 m, Z −20..40 m, eye height 0.25..15 m. No building/car/pedestrian collision, physics, walking animation or moving traffic. No room status board in this mode; toggles are logged. Interactive room/model/city launches run until you press **right B** to exit or end the application through the runtime. No frame/time cutoff. Smoke tests and sphere benchmarks remain bounded. Interactive timings stream into `vr/probe/xr-output.txt` rather than accumulating in RAM.

## Free assets and attribution

No account/login needed. Downloaded files and generated assets remain ignored by Git.

- [Modular Urban Apartments Facade](https://polyhaven.com/a/modular_urban_apartments_facade), James Ray Cock, Poly Haven, **CC0**. Selected façade pieces assembled/repeated and colour textures repacked.
- [Covered Car](https://polyhaven.com/a/covered_car), MP, Poly Haven, **CC0**. Scaled and placed; repacked textures.
- [Car Concept](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/CarConcept), model and textures copyright 2024 DGG, **CC BY 4.0**. Source began with Unity Fan's CC0 concept car. Our changes: scale/placement/repetition, atlas repacking, simplified glass and shading; not the original appearance. Full attribution/licence downloaded to `vr/assets/city/downloads/car-concept/LICENSE.md`. [Licence](https://creativecommons.org/licenses/by/4.0/).
- [Microsoft Rocketbox](https://github.com/microsoft/Microsoft-Rocketbox), copyright 2020 Microsoft, **MIT**. Male_Adult_01 and Female_Adult_03, mid-poly meshes, source colour/opacity textures. Rest-pose arms lowered, duplicate placement, textures repacked. Full notice downloaded to `vr/assets/city/downloads/rocketbox/LICENSE.md`; retain when redistributing models/derived payloads.

These are realistic-source assets, not a photorealistic renderer. Geometry/material rights do not grant endorsement or trademark rights.

## Prepare locally

Read repository README and existing dependency/toolchain setup first. Python requires NumPy and Pillow; Blender was available on this workstation (no packages installed). Scripts download models/data, not installers. All destinations are inside this repository.

```bash
python3 vr/fetch-city-assets.py
blender -b -t 2 --factory-startup --python vr/build-city.py
python3 vr/pack-city.py
```

Fetcher requires HTTPS and validates redirect destinations against Poly Haven/raw GitHub hosts, restricts glTF relative paths, limits file sizes and records SHA-256 in `vr/assets/city/downloads/download-hashes.json`. Upstream GitHub branches are mutable: recorded hashes identify this download, not pinned immutable future inputs. Assembly saves `city.blend`, geometry/material metadata and textures under `vr/assets/city`; packer outputs `vr/probe/city.bin`. Blender assembly runs in a new empty process, not the user's open scene.

Compile XR builds as in OPENXR-SCENE.md, with `-DPROBE_SIZE=1080` or `1440` and respective executable suffix. Compile `mesh-probe.c` with the same strict toolchain for tests:

```bash
bash vr/run-feature-probe.sh city
bash vr/run-feature-probe.sh city-negative
bash vr/run-feature-probe.sh mesh
```

## Verification and limits

- Strict warning-as-error compilation: mesh probe and XR at 720/1080/1440.
- City offscreen readback passes static/moving coverage, finite colour, projected depth and camera-motion oracles; alpha discard, depth order, far-plane clipping, in-frustum ground occlusion with a visible positive control, and off-axis geometry at 80 m view depth. Static/moving coverage about 182k/180k pixels at probe resolution.
- Disabled-mesh negative control rejected (zero coverage). Disabling the shader procedural-depth rejection was also correctly rejected; restored shader passes. Previous character/room mesh regression passes.
- Host locomotion test runs extracted production functions; room regression and expanded city bounds pass. Python/script syntax and HTTPS/redirect rejection checks pass.
- Independent review caught incorrect radial sky occlusion, linked material-factor defaults, redirect validation and a room-only occlusion test; corrected with new far-geometry coverage, effective source glTF factors/transforms and scene-specific tests.
- Headset smoke **not passed**: Proton/bootstrap stalled with empty application output and tool timeout; subsequent inspection found no test/Proton/wineserver processes and GPU responded normally. WiVRn was left running. No city neural frame submission or current city performance claim yet.

Renderer still omits mipmaps, normal/roughness/metallic texture maps, mesh shadows, AO, true transmission and full PBR extensions. Hair uses cutout alpha; shader spatial AA remains enabled. Atlas resolution is shared and repeated UV ranges reduce effective detail. Static geometry has camera-motion vectors only. Two-source repeated cars/pedestrians are deliberate first-test limitations. Benchmark actual city mode before assuming earlier sphere timings apply.

Saved local evidence: `vr/assets/city/{probe-output.txt,negative-output.txt,model-regression.txt,locomotion-output.txt,headset-smoke.log}` and `city-static.ppm`. No proprietary runtimes or downloaded/model payloads should be committed.

## Closed-street / session update

Both street ends now have transverse building frontages, with side façades and pavements extended to meet them. Still bounds-only locomotion, not collision. Your headset feedback confirms the earlier city rendered, but these latest end closures and right-B exit need headset confirmation. Cars/buildings still use simplified materials; no claim of improved photorealism. Test.glb remains available unchanged.

## Detailed assets and smooth turning update (05/10/2026)

- Interactive right-stick turning is now continuous, up to 60°/s, with a 0.2 deadzone and head-centred pivot. Turning takes priority over vertical movement. Scheduled smoke-test snap turns remain for deterministic tests.
- Cars now use the user-supplied Volvo from `Downloads/VR/Vehicles/Modern+Minimalist+Automotive+Scene.blend`, exported separately to `vr/assets/city/downloads/user-volvo.glb`. Original files remain untouched. For reproducibility, export only the vehicle meshes (no display platform/scene), apply their evaluated static transforms and materials, preserving GLB textures. Assembly requires this local export; the automatic fetcher does not supply it. Asset redistribution rights have not been established; do not publish it.
- Five copies have a target length of 4.95 m; assembly asserts their lowest geometry is on road Z=-0.02 m. Detailed meshes use an 8% decimation ratio for display; smaller parts retain their geometry. City total: 1,688,139 triangles.
- Façade origins are rebased to continuous 3 m storeys. Cornices have their own wall-top placement rather than the wall-base position.
- Source glTF base-colour factors are preserved alongside textures instead of using linked Blender socket defaults. City normal maps use dedicated atlas tiles and derivative-based tangent frames. Glass transmission is not implemented; the Volvo main glass uses an explicitly dark/smooth opaque approximation rather than white rough metal.
- Atlas: 8192², 11×11 cells, 744-pixel tiles; colour and normal maps share it. This limits texture resolution, and mipmaps, mesh shadows, reflections and full transmission remain absent. More realistic assets do not by themselves produce photorealistic rendering.
- Strict 720/1080/1440 builds, actual locomotion host tests (including diagonal arbitration), city GPU colour/depth/motion/occlusion checks, disabled-mesh negative control and original model regression pass. New car placement, floor connections, materials and smooth turning still require visual headset confirmation. No updated city performance benchmark.

Review follow-up: normal data now uses per-texel sRGB reversal before bilinear interpolation (not reversal after filtered sampling). Unsupported independent normal UV/transform/strength/addressing is explicitly reported and that map is omitted; the Volvo turn-signal material uses this fallback. This is deliberately limited material support, not full glTF compliance. Updated city and original-model GPU regressions pass.

## Detail, surfaces, transparency and collision update

Supersedes the earlier detail/glass/bounds-only limitations above:
- Car decimation increased from 8% to 20% (2.5× retained proportion for affected meshes). Assembled city has **3,019,935 triangles / 9,059,805 flattened vertices**. No fabricated detail or subdivision of every asset.
- Corrected coordinate conversion from mirrored (X,Z,Y) to proper rotation (-X,Z,Y), for positions, normals and collision metadata. This fixes reflected geometry/lettering, rather than flipping texture UVs. Original model.bin is unchanged.
- Added CC0 Poly Haven `asphalt_02` and `concrete_floor_02` diffuse/normal surfaces: https://polyhaven.com/a/asphalt_02 and https://polyhaven.com/a/concrete_floor_02. Run `python3 vr/fetch-city-surfaces.py` before assembly. URLs and SHA-256 are saved locally in downloads/surfaces/sources.json. HTTPS redirects are restricted to Poly Haven API/download hosts. Payloads remain ignored.
- Road and pavement consist of short UV-mapped tiles to avoid compressing the entire street into one atlas cell. UV rounding tolerance prevents tiny export errors from tripling texture-repeat spans. Building materials retain the imported colour/normal maps; the atlas and simplified lighting still limit realism.
- Sorted per-eye glass triangles blend in **linear light** between explicit decode/encode GPU passes. Glass depth-tests against opaque objects but does not write neural depth/motion; those describe the nearest opaque surface. This is alpha transparency, **not refraction, reflection or physical transmission**. Triangle-centre distance sorting can still fail for intersecting glass. Normal opaque rendering remains display-encoded as expected by the neural runtime.
- City controller locomotion uses swept 20 cm head-radius collision against 15 coarse bounds (five cars, six people, four building envelopes). Sliding and vertical movement are supported; ground/room bounds retained. This is not a full body/mesh physics solver. Physical headset movement remains unblocked for tracking safety and can enter geometry; virtual controls allow escape. No real-world safety guarantee.

Verification: strict 720/1080/1440 builds; city and original-model GPU regression; disabled mesh/glass negative controls; zero/25%/100% glass linear-light colour oracle and unchanged depth/motion; collision metadata malformed-input, thin-wall tunnelling, corner, sliding, vertical and escape tests; locomotion regressions. Independent review identified encoded-space blending and unrestricted surface-fetch redirects, both fixed and retested. New headset appearance/performance remain unverified.

## Ground shimmer filtering

User confirmed the city improvements, but reported road/pavement random-noise shimmer with neural **off**. Those surfaces previously used an unfiltered single mip for both fine diffuse and normal-map detail.

Ground diffuse/normal tiles now contain nine tile-local mip levels (368² down to 1²). The pixel footprint conservatively selects/interpolates levels using the Frobenius norm of UV derivatives. Colours are downsampled in linear light; normal data is averaged without gamma conversion or vector renormalisation. Periodic gutters and texel-edge mapping preserve continuity across repeated 2 m surface tiles. Other meshes/materials keep their existing sampling path. Filtered flags require an 8192² atlas and are rejected for incompatible dimensions.

This deliberately trades some near-ground texture resolution (368 vs 736 pixels per tile) and grazing-angle sharpness for stability; it is not temporal AA or a whole-scene blur. Pyramid layout/linear-light/data filtering, periodic boundary and anisotropic-footprint numerical tests pass (`python3 vr/tests/surface-filter-test.py`). Three resolution builds, city/glass and original-model GPU regressions pass. Independent review caught periodic seams, insufficient footprint bounds and incompatible-atlas acceptance; corrected and retested. Actual headset shimmer reduction and updated GPU cost remain to be checked.
