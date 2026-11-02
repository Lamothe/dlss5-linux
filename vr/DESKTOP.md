# DESKTOP — a windowed DLSSNR application on Linux through Proton

Status: first visible desktop milestone, 02/11/2026. Not a product, not a parity or
quality claim, and not a benchmark.

## What this is

`vr/nr-desktop.c` is a small D3D12 application that owns its own scene, evaluates
NVIDIA's DLSS Neural Rendering model (NGX feature 18) on each frame it renders, and
presents the raw render beside the neural output in one window. It runs as a Windows
PE under Proton, because the model (`nvngx_dlssnr.dll`) is a Windows binary and no
native Linux NR library exists.

Unlike the add-on route, nothing is injected into someone else's process: no ReShade,
no Detours, no hooking of a game's NGX calls. The application calls the model itself,
so the layers are Wine/Proton, vkd3d-proton and dxvk-nvapi only:

1. `_nvngx.dll` (dxvk-nvapi in the prefix) provides `NVSDK_NGX_D3D12_Init` and the
   capability parameter block.
2. `nvngx.dll_nrfwd.dll` loads `nvngx_dlssnr.dll` and forwards Init/Create/Evaluate/
   Release from a module the snippet's caller gate accepts (its name contains
   `nvngx.dll`); the same convention the add-on source documents.
3. The application writes the `DLSSNR.*` creation and per-frame parameters through the
   driver core's own parameter block (uint setter is vtable slot 3, resource setter
   slot 0, float slot 1).

## Scene

The window is 1440×720: two 720×720 panels, 1:1 with no filtering. The left panel is
the application's own render; the right is the same frame after the model. A two-pixel
white seam separates them. The scene is the procedural room from `vr/gpu-scene.hlsl`
(`Pad0 & 1`), rendered by its compute entry with `Eye = 0` and a camera that pans with
the frame number, so the model receives non-zero motion vectors.

F10 cycles side-by-side → neural only → raw only (single-source modes are letterboxed
at 1:1, not stretched). Escape closes.

## Build and run

```sh
tools/zig-x86_64-linux-0.15.2/zig cc -target x86_64-windows-gnu -std=c11 \
  -Wall -Wextra -Werror -Wno-unused-function -O2 vr/nr-desktop.c \
  -ld3d12 -ldxgi -ldxguid -ld3dcompiler_47 -o vr/probe/nr-desktop.exe

bash vr/run-nr-desktop.sh                    # window stays up until Escape
bash vr/run-nr-desktop.sh --frames 100 --dump nrdemo
```

`vr/run-nr-desktop.sh` uses the same containment boundary as `run-feature-probe.sh`
(bubblewrap, XWayland only, no session bus, no host home) and the same Proton copy and
prefix. Proton detaches the child's stdio, so the launcher runs the exe through
`cmd /c` with a redirection into `vr/probe/nr-desktop-output.txt`.

The model and forwarder must sit beside the executable (`vr/probe/`); neither is in
Git.

## Verified 02/11/2026

- Clean build with the warnings-as-errors line above.
- `CoreInit`, `GetCapabilityParameters`, `NeuralInit` and feature creation all return
  `0x1`, and `CoreShutdown` reports `live_devices=0`.
- 100 presented frames with one evaluation each; every evaluation returned success.
- Frame 90 dumped the raw texture, the neural texture and, in the same frame, the
  backbuffer that was about to be presented. Comparing the presented image against the
  two dumps: every row of both halves matches except the two seam columns (mean
  |difference| 0.20/255 left, 0.08/255 right; the only pixels over 2/255 are the 720
  seam pixels). The composition is therefore pixel-exact to what was dumped.
- Raw versus neural on that frame: mean |difference| 7.44/255, 78.6% of channels moved
  by more than 2/255, 8.3% unchanged, horizontal gradient energy 11.07 → 12.03. The
  model is changing the image and is not simply blurring it.

## Not verified, and not claimed

- **No quality claim.** Whether the neural frame looks better than the raw one is a
  human judgement; these numbers only establish that the model ran and changed the
  image in a structured way.
- No comparison against NVIDIA's Windows NR output, no vendor reference frame.
- One scene, one resolution (720×720 per panel), one model build
  (`sha256 4b8d19bc…`, 165,840,496 bytes — not the build the upstream add-on documents
  as tested).
- ~~No frame-time measurement.~~ Measured separately: [NR-COST.md](NR-COST.md) reports
  the serialised cost of an evaluation across a resolution ladder and concludes that a
  native-resolution stereo pair does not fit a 72–90 Hz budget on a 4070 Ti SUPER. The
  demonstration loop itself is still a diagnostic loop, not a timing harness.
- No HDR, no tone mapping, no UI, no temporal reset policy beyond frame 0.
- The model is NVIDIA's and is not redistributed by this repository; the demonstration
  consumes a locally present copy.