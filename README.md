# DLSS5 Linux — native neural-rendering migration

**Current goal: migrate the neural-rendering pipeline to native Linux. VR is a separate, later project.**

Repository: [Lamothe/dlss5-linux](https://github.com/Lamothe/dlss5-linux). Local workspace: `/home/michael/Projects/dlss5-linux`.

## Milestones and project boundary

1. **Native Linux migration (this project):** recover and validate the complete pipeline, including preprocessing, learned-weight bindings, the full execution graph, temporal history/reset behaviour and output conversion. Demonstrate end-to-end neural-rendered images from live inputs using native Linux execution, without Windows host code, DirectX or Proton in the evaluator. Isolated kernel replays are intermediate evidence, not completion.
2. **VR integration ([dlss5-vr](https://github.com/Lamothe/dlss5-vr), later):** consume the validated native evaluator in the existing Genesis host; then validate per-eye state, stereo correctness, performance and headset behaviour. This is not a prerequisite for the Linux migration. Do not build another VR host.

The earlier Proton-based desktop/VR demonstrations are reference experiments, **not proof of a native Linux migration**. Proton may still be used to capture vendor reference data; the final native evaluator must not depend on it.

## Current migration status — 25/10/2026

- Published reference baseline: 68 pre-decoder slot contracts (slots 24–91) out of the archival 156-launch inventory, through commit `33fa071`.
- Slot 91 is qualified with paired captures, native replay and controls, and has passed independent review; its contract and summary are published in commit `33fa071`. Published contract coverage is 68/156. Raw captures, traces and replay run JSONs remain ignored local evidence and are not included in Git.
- These counts describe the pre-decoder replay campaign, not percentage completion of the entire migration. Decoder and temporal investigations have their own bounded evidence.
- A complete native input-to-output pipeline and native neural-rendered image are **not yet established**. VR/Genesis integration is out of scope for the current campaign.

Historical handovers still name the predecessor workspace/remote: for all continuing migration work, use this workspace and its `dlss5-linux` origin instead. Do not follow their old publishing destination.

Start with [MILESTONE.md](MILESTONE.md), [vr/CAMPAIGN-HANDOVER.md](vr/CAMPAIGN-HANDOVER.md), [vr/NATIVE-SLOT-REPLAY.md](vr/NATIVE-SLOT-REPLAY.md), and [CHECKPOINT.md](CHECKPOINT.md).

## Layout, dependencies and checks

The existing `vr/` directory name is retained to preserve imports, capture paths and historical evidence. Native migration probes live there alongside the inherited reference harness; the directory name does not make VR part of this milestone. The Git history is preserved from `dlss5-vr`; inherited VR source is historical/reference material, not the active deliverable.

This workstation's excluded dependencies and evidence were copied using copy-on-write reflinks, not shared writable directories. They remain local and are **not uploaded**: model/runtime DLLs, assets, captures, prefixes, SDKs, generated binaries and raw logs. A fresh clone is not turnkey. See [CHECKPOINT.md](CHECKPOINT.md) for the dependency boundary. Existing machine-specific runtime manifests may need adaptation before launching a reference application from this workspace.

CPU-only migration regression checks (no GPU/headset or proprietary payload required):

```bash
PYTHONDONTWRITEBYTECODE=1 python3 vr/native-slot-replay-probe.py self-test
PYTHONDONTWRITEBYTECODE=1 python3 vr/ladder-report.py self-test
PYTHONDONTWRITEBYTECODE=1 python3 vr/ladder-names.py self-test
PYTHONDONTWRITEBYTECODE=1 python3 vr/tests/native-input-layout-test.py
```

Keep proprietary payloads out of Git. Native replay currently uses NVIDIA's proprietary kernels/weights; it is not a clean-room replacement or a redistribution licence. Preserve the existing containment, rights and evidence qualifications. Never run probes concurrently across the two workspaces: their local locks are now independent.

## Historical Proton experiment notes (archived, not the current milestone)

The text below records the earlier experiment and its original dates/limits. The migration scope and status above supersede its current-milestone wording.

# DLSS neural-rendering experiment

**Historical Proton milestone (05/10/2026): interactive neural-rendered VR was demonstrated.** See [MILESTONE.md](MILESTONE.md) for current capabilities, user confirmation and limits; [CHECKPOINT.md](CHECKPOINT.md) for the source-only repository boundary. Earlier notes below retain desktop-test history.

Prepared 03/10/2026 for Linux/Wayland, RTX 4070 Ti SUPER, NVIDIA 615.71.09. The desktop experiment uses a Windows DirectX12 NVIDIA sample under Proton. The standalone VR prototype now renders neural stereo through Proton/OpenXR/WiVRn; neither is native Linux neural rendering. See `vr/ROOM.md` for the interactive demonstration room and `vr/TIMING-1080-RESULTS.md` for the earlier sphere benchmark.

## City street test

See [vr/CITY.md](vr/CITY.md) for the static city scene, free asset attribution, preparation, controls and validation limits. With dependencies prepared: `bash vr/run-xr-scene.sh on 1440 city`.

## Run

From a terminal in the graphical session:

```bash
cd /home/michael/Projects/dlss-testbed
# Ordinary DLSS baseline; bypasses ReShade entirely:
bash run-contained.sh
# Experimental neural renderer:
DLSS_TEST_NEURAL=1 bash run-contained.sh
```

Close the window to finish. In the neural run, F10 toggles the neural pass; Home opens ReShade and its DLSSNR Linux Add-ons panel. Avoid changing resolution or repeatedly pressing Apply initially: independent source review found resource lifetime and cross-thread settings risks. The sample may maximise its window despite requested 1280x720; recorded neural textures were 2560x1358. Do not interpret log-derived rates as isolated neural-pass GPU timing.

## Controlled benchmark

See `benchmark/RESULTS.md` for three repeated off/on captures at two verified resolutions. Measured mean throughput: 425.7 → 81.2 FPS at2560×1358; 954.1 → 211.7 FPS at1280×720. These are whole-app presentation intervals, not isolated GPU timings. Launcher now correctly uses Windows -width/-height arguments. User also visually confirmed F10 changes lighting.

## Results

- Official demo ordinary DLSS reached NVIDIA CUDA/NGX execution through Proton after Microsoft's VC++ x64 runtime was installed in prefix-0 only.
- RTX40 community model version310.8.0.0 successfully created NGX feature18.
- 45-second run reached 3,000 successful neural evaluations.
- 120-second bounded run reached 9,600 successful neural evaluations; no logged neural error or observed process crash before termination.
- No remaining demo/Proton/wineserver processes after verification; NVIDIA responded normally afterwards.
- User has confirmed visible desktop/VR A/B effects; broad quality and extended stability remain UNVERIFIED. A success API return alone does not prove useful visual output.

## Containment and risk

run-contained.sh uses bubblewrap namespaces: no network; host home and session bus hidden; /usr,/etc,/sys and Steam read-only; only this testbed writable. Runtime and Proton copies avoid writes to original Steam tool installations. `bash run-contained.sh --check` checks hidden home/session paths and network reachability (101 = network unreachable). GPU devices and authenticated XWayland are exposed; therefore this is NOT complete security isolation and cannot protect against GPU/driver vulnerabilities or all X11 session attacks. No root commands, driver changes, game modifications or system package installation were performed.

The proprietary RTX40 patch has no independently established/reproducible patch provenance. Hashes identify the community release bytes, not safety. Independent source review found no obvious explicit runtime networking/process launch in the add-on, but flagged GPU lifetime, shutdown and settings races. These were not patched; this experiment is not a production endorsement.

## Pinned components

- NVIDIA official demo v310.9.1 Windows archive: sha256 21c0511c6b45e80c9d0338f504f4f986d261d52050f7fe2af480e3c42581fc35
- https://github.com/NapXDD/addon-dlssnr-linux/releases/tag/v0.2.2
  - dlssnr-linux.addon64: bd62ae37a55a0e8b4698301dd862378e61454acf4e1083b8da018f70273bb6a7
  - nvngx.dll_nrfwd.dll: 940150545d7e1439d81d0711fc596b44d3f5b8101168bfbbd8f2b62eb77231d3
- https://github.com/RankFTW/rhi-repo/releases/tag/dlssnr-310.8.0-RTX40
  - ZIP: 46124cfaef532ad5f6da07494772ea8c1b3e719f934e254385697f38d1289e3f
  - DLL: 4b8d19bc3eff58a084f5eca7489c921501c203450169fb82ff4f649a4482ba05
- ReShade6.8.0 full-addon setup fetched from https://reshade.me/downloads/ReShade_Setup_6.8.0_Addon.exe; ZIP payload extracted without running installer.
- Microsoft VC++ redistributable fetched from https://aka.ms/vs/17/release/vc_redist.x64.exe and installed into prefix-0.

The model hash deliberately differs from the add-on's RTX50 tested hash; warning is retained in logs. RTX40 evidence: https://github.com/NapXDD/addon-dlssnr-linux/issues/8

Saved logs are in logs/, including neural-45s-reshade.log and neural-120s-reshade.log. Research metadata/source and unexecuted downloaded archives are retained under research/ and downloads/.
