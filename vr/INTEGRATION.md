# Neural stereo integration findings — 04/10/2026

## Verified next milestone

`feature-probe.c` calls the prefix's NVIDIA driver-core exports and existing checked forwarder directly. No ReShade, DLSS SR or OpenXR is loaded. It creates two simultaneous distinct feature handles at720×720, submits the initialisation command list, waits for a GPU fence, releases features and shuts down. Two consecutive corrected runs passed. This is feature creation only: NO neural evaluations, independent history verification, stereo images or performance measurement.

Build (local Zig0.15.2 verified against official SHA256; no system compiler changes):

```sh
tools/zig-x86_64-linux-0.15.2/zig cc -target x86_64-windows-gnu -std=c11 -Wall -Wextra -Werror -O2 vr/feature-probe.c -ld3d12 -ldxguid -o vr/probe/feature-probe.exe
bash vr/run-feature-probe.sh
```

Run from project root. Uses desktop containment, isolated VR prefix, bounded60second execution. Copies of existing verified proprietary model/forwarder live in`vr/probe`. This is not a full security sandbox. Host home/session bus/network hidden; GPU/X11 accessible. Steam SDK symlinks expose already-mounted libraries only, not `.steam` IPC files/token. No headset required. Output:`vr/probe/probe-output.txt`; Proton trace:`vr/logs/steam-424243.log`. Launcher validates PASS marker and rejects assertion/unhandled-exception logs, since Proton/cmd can report exit0 after application failures.

## Failures corrected

- First probe launch displayed Wine assertion in steam.exe: native Steam compatibility library unavailable inside containment. Added only `.steam/sdk32` and`sdk64` links pointing into already-exposed Steam library tree. No host IPC/token exposed.
- Builtin DXGI prevented DXVK-NVAPI initialisation. Native DXGI now selected (`dxgi=n,b`), without local ReShade DLL.
- Driver-core Shutdown1 export differs from public SDK wrapper: requires device and pointer to live-device count. Disassembly at0x180060690 forwards incoming RDX as live-device count pointer; one-argument call caused access violation during teardown. Corrected signature; EAX returns NGX status, additional output reports live-device count. Final validation checks successful status and zero remaining devices. Final repeated runs have no assertion or unhandled exception.

## What must change for actual eye rendering

Source: `research/source/addon-dlssnr-linux-0.2.2/src`.

- `nr_runner.hpp:550–650`: requires colour, depth and motion-vector textures; carries jitter, MV scale, reset/subrect data and inverted-depth flag. It colour-encodes/composites via compute. Not just an image-filter DLL.
- `nr_runner.hpp:~120` and`nr_compose.hpp:215–235`: each uses global singleton mutable state. Alternating eyes through unchanged code would mix feature/resources/white-point state. Distinct feature creation now proven, but independent evaluation contexts remain to implement/verify.
- `nr_runner.hpp:~420–450`: internal textures explicitly have array size1 and UAV access. Do not feed an OpenXR array swapchain directly. Copy each acquired eye subresource to dedicated 2D intermediates, preserving OpenXR image ownership and resource states.
- `nr_compose.hpp:~237–252` and`nr_runner.hpp:610–645`: assumptions about COMMON/UAV states come from intercepted DLSS output. Explicit transitions needed for render targets, depth, motion and OpenXR images in our renderer.
- Existing add-on's retired-resource frame countdown is not GPU fence synchronisation. A new path must fence resource/feature lifetimes; avoid live resizing initially.

Next real test: synthetic colour/depth/motion inputs evaluated repeatedly in two contexts with GPU fences and readback checks. Only after sustained valid outputs add to Windows D3D12 hello_xr renderer. Need genuine per-eye depth/motion/jitter/reset data for moving scene, not claim that zero motion is correct under head movement. Current cube sample binary is unmodified. Basic visible WiVRn/Proton headset baseline passed; neural headset rendering still NOT implemented.

Independent review: first pass identified stale-evidence/missing-log acceptance, alias double-release and unchecked shutdown. Fixed all three; reran strict build and validated probe successfully (shutdown0x1, live_devices0). Second read-only review PASS. Evidence is removed before each run under an exclusive lock; final PASS follows checked teardown. Baseline sample and probe processes no longer running; GPU responds normally.

## Update — repeated evaluation

The next milestone is now complete; see `EVALUATION.md` for two-context evaluation/readback results and limitations. Original feature-creation-only findings above describe the earlier stage.
