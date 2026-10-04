# Controller bindings and output AA fix

User reported dead thumbsticks and heavy aliasing in model mode. Root cause: six binding slots/actions existed but only four trigger/button bindings were suggested. Added left/right Touch `/input/thumbstick` bindings for move/snap vector actions and count derived from array. Logs now record suggested six bindings and active stick states every60frame attempts. Actual headset input remains to be confirmed; synthetic locomotion tests alone cannot verify runtime bindings.

All XR resolutions now enable directional spatial edge filtering in final composition, both neural modes. Room bit0 and AA bit1 are separated. Filtering operates on selected source's encoded colour before linear compositor conversion, uses clamped bilinear taps and luminance contrast threshold. No history/additional image buffers or increased neural resolution. Can soften detail; not MSAA/TAA, cannot guarantee elimination of temporal/hair/texture shimmer. GPU performance cost not yet measured.

Strict720/1080/1440Windows builds passed. Actual D3D compilation and offscreen AA GPU runs passed all16cases (twoeyes,flat/diagonal,neuraloff/on,AAoff/on): all-pixel independent CPU reference based on uploaded half values, finaldecode, finiteRGB exactalpha, flatunchanged and edge-localchanges. DisabledAA negative rejected615reference mismatches; corruptreadback+0.05 negative rejected1mismatch. Mesh GPU regression passed. Additional sphereGPU regression timedout afterframe11; no fullsphere pass claimed for this round. FinalindependentreviewPASS after strengthening numericalAAoracle. No test processes remain, GPUresponsive.

Run with headset worn and WiVRn connected:
`bash vr/run-xr-scene.sh on 1080 model`

Restart required if an old process is running. Leftstickmove,rightsticksnap,triggerneural,A/Xlighting. Verify both stick response and edgequality in headset. Offscreen commands: `bash vr/run-feature-probe.sh aa`, `aa-negative`, `aa-corrupt`. Logs logs/room/aa-* and mesh-after-aa-*.
