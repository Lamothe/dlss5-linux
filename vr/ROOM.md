# Interactive neural VR demonstration room

## Launch

From this testbed in the graphical session, with WiVRn connected and Quest worn:

```bash
bash vr/run-xr-scene.sh on 1080 room
```

Either trigger (pull/release): toggle neural processing for both eyes together.
Right A or left X: toggle cool daylight/warm lighting. Board ahead reads NEURAL ON/OFF; lettering is composed after neural processing in a world-anchored position with geometric occlusion. Controls use trigger hysteresis and rising edges; focus recovery initialises latches without false presses. Room loads both neural contexts even when started off, to permit activation. Each activation/lighting change resets both neural histories.

Tracked physical movement only; no joystick movement or virtual collision. Stay in your clear physical play space. Automatically closes after6000submittedframes or120s (launcher150s bound). Same unofficial runtime/containment limitations as README.md. Neural off bypasses evaluation but does not unload the room's runtime.

## Scene

Procedural6×7×3.2m room with window/exterior, wood table and legs, glossy/metal objects, fabric sofa/cushions, shelving/books and contrasting material panels. Material-specific highlights, filtered procedural detail and coarse bounded shadows. Not imported textured assets, a physically based path tracer or artist-tuned game integration. Source lighting already includes shading; neural appearance must be judged through A/B, not assumed better.

Old sphere scene remains default (third argument spheres). Same per-eye resolution choices720/1080/1440. ShaderPad0 selects room, Pad1 lighting. Status lettering uses post-neural geometry retrace only inside board bounds, not reading depth UAV in SRV state.

## Verification 04/10/2026

- Strict720/1080/1440C cross-compilation and shell syntax pass.
- Existing GPU sphere regression:12evaluations/eye,24validatedoutputs; numerical depth/motion/floor oracle remains passing. This also compiles shader through actual D3DCompilecs_5_1.
- Independent HLSL compilation/spirvvalidation passed in focused build agent; parent verified DX12 execution.
- Room smoke:600stereo submissions, auto toggles at200/400 and lighting300, 400neural-onframes/800neural evaluations and200offframes. Per-row state sequence checked. All timestamps complete. Smoke ignores controller changes and validates scheduled transitions.
- Interactive room:6000submittedframes;7logged actual controller neural toggles and4lighting toggles, clean completion. This verifies live input and neural/bypass switching, NOT user visual quality judgement.
- Independent review initially found smoke/liveinput coupling and focusrearm; corrected andstrictrebuilt, smoke runtime retested. Follow-up reviewPASS.
- No remaining testprocess;GPUresponsive. Logs underlogs/room/.

Timing rows retain their actual per-frame neural mode. Mixed room runs are NOT valid inputs for fixed-mode600frame benchmark analyser. Do not compare smoke-mode means as controlled benchmarks (headmovement/lightstate/warmup differ). Earlier sphere1080benchmark does not establish same performance for richerroom.
