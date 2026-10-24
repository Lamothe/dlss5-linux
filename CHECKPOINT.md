# Linux migration workspace split — 24/10/2026

The active native migration workspace is `/home/michael/Projects/dlss5-linux`, remote `git@github.com:Lamothe/dlss5-linux.git`. It preserves the source history; the Linux migration commits `a5fb96d` and `d166448` are published to its `main`. The original `dlss-testbed` workspace and `dlss5-vr` remote remain intact.

The complete local working tree (including ignored evidence and dependencies) was copied using Btrfs copy-on-write reflinks. The workspaces have independent files; do not run GPU/reference probes concurrently because their locks are independent. Local-only runtime manifests/prefixes can retain old absolute paths; inspect them before running a reference harness. No payloads or generated assets are added to Git by this split.

The historical checkpoint below describes the inherited layout and dependency boundary; its statement that no remote upload has occurred is historical. The current project scope is [MILESTONE.md](MILESTONE.md).

# Source checkpoint and local dependencies

This is a milestone snapshot, not a turnkey distribution. Launchers currently contain workstation-specific paths, UID 1000 and Steam layout. No remote upload has been performed.

## Preserved source

Standalone C/D3D12/OpenXR renderer/probes, HLSL shaders, Blender GLB import/atlas packaging scripts, launchers, timing analysis, host movement tests and chronological engineering notes. Older documents describe the state at their original test; MILESTONE.md is the current summary.

## Deliberately local-only

- `tools/zig-x86_64-linux-0.15.2`, Proton/Steam runtime copies and prefixes.
- `vr/include/openxr` Khronos SDK headers, local WiVRn/xrizer payloads and workstation runtime manifests.
- `vr/probe/` compiled programs and third-party neural/forwarder/OpenXR DLLs.
- `vr/assets/test-model/` exported character geometry/materials/textures and generated atlas; `vr/probe/model.bin`.
- Desktop NVIDIA sample, downloaded archives, raw logs and timing CSVs.

Do not delete these when creating the repository: they remain necessary for current local execution. Component version/hash notes are in README.md and vr/README.md. A new workstation needs independently obtained dependencies, authorised assets and adapted paths; the repository does not recreate them automatically.

## Checks

Portable host movement test (C compiler and Python required):

```bash
bash vr/test-locomotion.sh
```

Strict Windows build on the existing workstation, repeated for 720, 1080 and 1440:

```bash
tools/zig-x86_64-linux-0.15.2/zig cc -target x86_64-windows-gnu -std=c11 -Wall -Wextra -Werror -Wno-missing-field-initializers -O2 -DPROBE_SIZE=1440u -Ivr/include vr/xr-scene.c -ld3d12 -ldxguid -ldxgi -ld3dcompiler_47 -o vr/probe/xr-scene-1440.exe
```

Existing local offscreen checks: `bash vr/run-feature-probe.sh mesh`, `aa`, `aa-negative`, `aa-corrupt`. Require the excluded dependencies and graphical-session containment setup; do not mistake host checks for GPU/headset checks.

## GitHub

Prefer an empty private repository initially (no generated README/licence/gitignore). Set a remote only after confirming its URL, and push only the audited source snapshot/tag. Decide redistribution/licensing and portable dependency setup before making it public.
