# Tracked OpenXR frontend — 04/10/2026

## Implemented, not yet headset-verified

`xr-scene.c`: standalone C Windows D3D12 frontend using official OpenXR1.1.63 headers. Runtime-required adapter LUID/minimum feature level selected before graphics/NR setup; LOCAL tracked poses/FOV at predicted display time; fixed720×720per eye. Shared world rotated180degreesY and floor1.4m below LOCAL origin. GPU scene projection uses quaternion rays and previous-view reprojection. Normal D3D depth, pixel previous-minus-current motion. Separate features/resources per eye.

Composition converts display-encoded neural input/output to linear FP16, copies into runtime FP16 swapchains, restores RENDER_TARGET state before release. Runtime must support FP16; no fallback. Frame/session events and shouldRender handled; resets on missing tracking, suspended rendering and effective LOCAL recenter. GPU completion fence precedes allocator reuse/release. Test bounds120seconds or600submittedframes, external launcher150seconds plus5second kill grace. No controllers, desktop preview or live toggle yet. Separate launches select neural mode.

```sh
# From project root
tools/zig-x86_64-linux-0.15.2/zig cc -target x86_64-windows-gnu -std=c11 -Wall -Wextra -Werror -Wno-missing-field-initializers -O2 -Ivr/include vr/xr-scene.c -ld3d12 -ldxguid -ldxgi -ld3dcompiler_47 -o vr/probe/xr-scene.exe
bash vr/run-xr-scene.sh off
bash vr/run-xr-scene.sh on
```

Launcher keeps existing GPU/XWayland containment and additionally shares only WiVRn compositor socket. No network/session bus/home data. This is not complete GPU/desktop/IPC security isolation. Uses same prefix lock as feature probe; concurrent-launch negative control rejects before executable launch. Official sample launcher is older/uncontained and used only for bounded comparison, no neural DLL.

## Verification

Strict builds pass. Offscreen `--tracked-control` exercises shader's actual quaternion/FOV branch (180Y, known squareFOV translating camera).24successful neural evals/readbacks; sparse normal-depth/motion samples agree with independent mirrored CPU intersection oracle. Finite/initialzero motion tolerance0.001pixels allows projection roundoff, magnitude samples tolerance0.002pixels. All input textures poisoned every frame. This tests known poses, not arbitrary runtime tracking or stereo correctness.

GPU baseline24eval regression, create-only and60frame-per-eye synthetic regression pass after shader/frontend changes. Logs in`logs/xr-scene/`. GPU remains responsive; no sample/probe process remains.

Independent review found/fixed aliased-handle double release, wrong/omitted createparams, recenter history gap, behind-eye floor/inside-sphere intersection and different prefixlock paths. Follow-up confirmed firstfour fixes; final lock fix manually tested by holding common lock and observing rejection. No review-based claim of runtime success.

## Actual blocker

Custom off-mode launch stalled in Proton get_vulkan_extensions VR state wait before program output/frame loop. Bounded run timed out; not PASS. After inspection confirmed no residual sample processes, previously working official hello_xr baseline was run bounded25seconds and stalled at the SAME VR-state bootstrap. Therefore blocker is reproducible independently of new renderer and outside its entered frame loop; exact root cause (runtime/headset connection versus bootstrap state) remains undiagnosed. Do not claim headset is disconnected merely from this evidence.

`logs/xr-scene/blocked-bootstrap.log` and`official-baseline-bootstrap.log` retain evidence. No neural headset mode executed while baseline blocked. Prepared program can be tested after WiVRn/Quest active connection is re-established and official cube baseline works. Need actual headset image/headtracking assessment, fresh frame submission evidence, then neural mode, temporal/stereo checks and GPU timing. Current implementation not yet known safe/comfortable for VR: CPU readback removed from interactive rendering but perframe GPU fence waits remain conservative, not latency optimised.

Headers fetched directly from pinned KhronosGroup/OpenXR-SDK release-1.1.63 include/openxr paths; copyright/licence retained. Hashes:`logs/xr-scene/header-sha256.txt`.

## Reconnection follow-up

WiVRn reconnection cleared initial bootstrap. Builtin-DXGI neural-off run completed600submittedframes and checked teardown (`logs/xr-scene/reconnected-off*`). Neural-on revealed incorrect `_Init_Ext` export and builtin-DXGI NGX failure0xbad00002. Corrected to already-tested `_Init` export/appID, UICorrection1 and nativeDXGI=n,b. Added export/init and early frame/view diagnostics; timeout launcher now surfaces captured output before propagating failure.

Native-DXGI neural initialisation now succeeds, with two distinct features. Native off/on did NOT submit frames before bounded termination. Short diagnostic confirms shouldRender1 but xrLocateViews count2 flags0x0 (neither position nor orientation valid), so frontend correctly refuses to render from invalid poses. This may reflect headset wear/tracking/runtime state; exact reason unverified. Need active headset tracking before rerun. No neural evaluations or neural headset pass claimed. Logs retain earlier offpass, native init and tracking diagnostics separately.

Strict compilation and shell syntax pass. Independent review confirmed corrected ABI/settings, found timeout diagnostics bypass and that was fixed. Follow-up review requested. One force-terminated diagnostic briefly left a defunct process; it is not GPU work.

## Headset worn — neural frame submission passed

User confirmed connected and wearing Quest. `run-xr-scene.sh on` completed exit0: view flags0xf (valid/tracked position+orientation), two distinct feature handles,600neural-enabled stereo frame submissions. Each submitted frame requires successful evaluation for both eyes:1200successful neural API evaluations inferred from checked loop. Both parameter destructions succeeded and checked runtime shutdown passed. Launcher checked PASS and absence of recognised fatal Proton diagnostics. Evidence:`logs/xr-scene/worn-on.txt`, `worn-on-launch.txt`, `worn-on-proton.log`. No remaining sample/probe process; GPU responsive.

This establishes live tracked stereo submission with neural evaluation enabled, NOT visual confirmation, independent temporal history, stereo quality, latency or useful VR performance. User assessment of scene/head motion/discomfort and controlled off/on timing still outstanding.

## Filtered checkerboard

Floor source now uses analytically box-filtered checker colour. Explicit ray differentials compute floor-plane projected pixel extent; signed period-two square-wave integrals average each checker axis, preserving .2/.5 near interiors and converging to .35 when minified. Axis-aligned footprint approximation, not full anisotropic or scene anti-aliasing. Does not change depth/motion, sphere edges,720square resolution or neural settings.

Strict probe/frontend builds pass. GPU and tracked-control each24successful evaluations/readbacks; visiblefloor sample colour agrees within0.0015encoded with CPU expectations. Negative control substituting original point-sampling rejected (invalid3floorRGBchecks) then restoredfiltered regressionpass.63signedphase/width integral cases matched independently integratedpiecewisewave (summarylog only). Independent read-only review PASS; flagged distant/generalrotatedFOV coverage limits.

Bounded filtered neural headset attempt initialised successfully but no submittedframePASS before timeout. Logs `logs/floor-filter/headset*`; not visual confirmation or performance evidence. Prior unfiltered600frame headsetpass remains valid separately. No assumption that this change eliminates allshimmer; headset comparison remains outstanding. No shader/GPUbenchmark claimed.

## Filtered floor — tracked headset execution passed

User confirmed wearing headset. Filtered neural-on run exit0, trackedviewflags0xf,600stereo submissions (1200checkedneural evaluations), normal teardown/PASS. Logs:`logs/floor-filter/worn-headset*`. No remaining sampleprocess; GPUresponsive. Visual judgement still awaits user; no timing/qualityclaim inferred from execution.

## Higher-resolution comparison

User found filteredfloor smoother but whole720scene blurry. Added guardeddefaultPROBE_SIZE and built independent1080/1440XR variants;720probe/default preserved. Launcher `[off|on] [720|1080|1440]` allowlists resolution/executable, rejects extraargs and missingbinary. All3strictbuilds andbashsyntaxpass; malicious resolutionargumentrejectedbeforelaunch. Read-onlyreviewPASS dimension propagation/argument safety.

`bash vr/run-xr-scene.sh on 1440` completedexit0:1440squareeach eye (4×720pixelcount), validtrackedposes,600neuralstereo submissions/1200checked evaluations, teardownpass. Logs:`logs/resolution/1440-on*`. No remainingtestprocessGPUresponsive. Visual sharpness/headmotion judgement pending; no GPUtimingorlatencyclaim.1080builtbutnotheadsettested. Rebuild higher variant with same documentedcommand plus `-DPROBE_SIZE=1440u -o vr/probe/xr-scene-1440.exe` (or1080).
