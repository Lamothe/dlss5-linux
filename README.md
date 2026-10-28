# DLSS5 Linux — native neural-rendering migration

**Current goal: migrate the neural-rendering pipeline to native Linux. VR is a separate, later project.**

Repository: [Lamothe/dlss5-linux](https://github.com/Lamothe/dlss5-linux). Local workspace: `/home/michael/Projects/dlss5-linux`.

## Milestones and project boundary

1. **Native Linux migration (this project):** recover and validate the complete pipeline, including preprocessing, learned-weight bindings, the full execution graph, temporal history/reset behaviour and output conversion. Demonstrate end-to-end neural-rendered images from live inputs using native Linux execution, without Windows host code, DirectX or Proton in the evaluator. Isolated kernel replays are intermediate evidence, not completion.
2. **VR integration ([dlss5-vr](https://github.com/Lamothe/dlss5-vr), later):** consume the validated native evaluator in the existing Genesis host; then validate per-eye state, stereo correctness, performance and headset behaviour. This is not a prerequisite for the Linux migration. Do not build another VR host.

The earlier Proton-based desktop/VR demonstrations are reference experiments, **not proof of a native Linux migration**. Proton may still be used to capture vendor reference data; the final native evaluator must not depend on it.

## Current migration status — 26/10/2026

Published source checks remain bounded: `vk_gltf_renderer` v2026.11 release notes do not announce a DLSS-NR SDK/header availability change; that does not establish whether the SDK changed elsewhere. Its guide still requires a compatible beta NGX SDK. Official NVIDIA/DLSS `v310.9.16` tracked source lacks `nvsdk_ngx_helpers_dlssnr_vk.h`; its Linux demo archive was not inspected, and separately distributed restricted packages are not ruled out. Official Streamline 2.14.0 changelog lists `sl.dlss_nr`, but its Vulkan support entry is for Reflex, not NR; its current programming guide specifies a 64-bit Windows requirement. These sources do not document a native Linux/Vulkan NR ABI. Target availability, authorisation, ABI compatibility and interop remain unverified; no capture or GPU probe is authorised. The bounded source finding and its qualifications are recorded in `a30bee1`; the Streamline Linux-support issue response is summarised in the latest handover update. Streamline 2.14.16 was published 16/10/2026 UTC (17/10/2026 Australia/Sydney); official release metadata lists `streamline-sdk-v2.14.16.zip`, `streamline-sdk-v2.14.16-aarch64.zip`, and `streamline-sdk-v2.14.16-arm64ec.zip`, none Linux-named. Asset names do not establish package contents or rule out separate/restricted distribution. A bounded staged-download inventory is recorded in `bf4fdbf` and the gate note: the workspace contains an NR archive and a Windows demo archive, but no Linux demo archive or Streamline NR header; the proprietary files were not inspected. NVIDIA/DLSS release `v310.9.16` does publish a Linux demo archive, verified from official release metadata, but its contents remain uninspected; see the gate note.

- Published reference baseline: 68 pre-decoder slot contracts (slots 24–91) out of the archival 156-launch inventory, through commit `33fa071`.
- Slot 91 is qualified with paired captures, native replay and controls, and has passed independent review; its contract and summary are published in commit `33fa071`. Published contract coverage is 68/156. Raw captures, traces and replay run JSONs remain ignored local evidence and are not included in Git.
- These counts describe the pre-decoder replay campaign, not percentage completion of the entire migration. Decoder and temporal investigations have their own bounded evidence.
- The documented input-plane layout and capture-manifest integrity are validated against two local reference captures; see [NATIVE-INPUT-LAYOUT.md](vr/NATIVE-INPUT-LAYOUT.md). This does not establish DLSS preprocessing or model bindings. The bounded static DLL inspection recorded in `b894c93` was run on 25/10/2026; its ignored report SHA-256 is `e01956d0dd6afaab7b540db9f76a7fe8d70f193ddaa4ce0f9e9960622c72b0e2` (details in [NATIVE-GRAPH-NEXT.md](vr/NATIVE-GRAPH-NEXT.md)). It confirms only the inspected descriptor/routing logic and explicitly reports full graph recovery, dynamic bindings and native image inference false. Existing traces lack the raw-input-to-launch identity edge. A CPU-only closed-schema sketch in `vr/native-resource-events.py` tests metadata record shapes, bounded values and fail-closed controls only (including bounded JSON size/depth, duplicate-key rejection, unique launch join keys and duplicate binding detection while allowing multiple mappings per launch, including distinct mappings over the same opaque object ID); it does not prove runtime binding or validate captures. The source-grounded, independently reviewed resource-identity and provenance investigation is documented in [NATIVE-HOOK-PLAN.md](vr/NATIVE-HOOK-PLAN.md); current evidence does not justify capture. The existing `vr/native-ngx-probe.cpp` is only a native Vulkan/NGX capability and feature-creation probe; it submits setup commands and initialises the proprietary runtime but does no feature evaluation, inference or image output. Do not repeat it absent fresh explicit runtime/capture authorization. The decoder binding doc now specifies a distinct synthetic multi-term/FP16-tail experiment plan (`vr/NATIVE-DECODER-ACCUMULATION-PLAN.md`) and a static output-store envelope prerequisite audit (`vr/NATIVE-DECODER-STORE-ENVELOPE.md`); the experiment requires a complete runtime address/CFG bound, explicit operator authorisation and current safety preflight, and is not vendor-equivalence evidence. The probe's 256 MiB output allocation is not a proven bound; legacy launch/IO/weight notes distinguish the one tested output-change extent from the maximum effective address. No capture is authorised by the hook plan. See also [NATIVE-GRAPH-NEXT.md](vr/NATIVE-GRAPH-NEXT.md). A complete native input-to-output pipeline and native neural-rendered image are **not yet established**. VR/Genesis integration is out of scope for the current campaign.
- **Runnable demo and performance phase (28-29/10/2026):** the pipeline runs end to end from a built-in synthetic image through `image_evaluation` to host frames via [README-demo.md](README-demo.md) (`vr/demo-native.cpp`; synthetic weight presets — the visual transformation is arbitrary, this proves the pipeline runs, not DLSS quality). Performance work is tracked in [NATIVE-PERFORMANCE.md](vr/NATIVE-PERFORMANCE.md): the per-kind profile (P1), deterministic CPU threading with bit-identical output (P2, 2.22x at 8 threads), and the Vulkan GPU stage ([NATIVE-VULKAN.md](vr/NATIVE-VULKAN.md), G1: the projection matmul 622x vs the single-thread CPU reference on the RTX 4070 Ti SUPER, CPU-vs-GPU bit-identity contractual for normal-range intermediates with the denormal divergence pinned and attributed). The GPU stage now routes the vit, split and swin projection rows (G2-G8) with byte-identical complete-network artefacts: weight residency (G5), the stable record store and the batched swin seam (G8, one executor call per 32 windows instead of per window — the swin rows measure 6,850 vs 8,301 ms/frame on CPU and default ON with an executor attached) plus CPU branchless e4m3 decode, bit-exact AVX2 matmul and threading (G7). Measured end-to-end at 8 threads: 3.22x vs the single-thread reference; the speed numbers are real measurements and the synthetic weights make the visual output arbitrary — no image-quality or vendor-parity claim.
- **Independent native implementation path (26/10/2026):** the proprietary-route blockers above bind the replay/capture route only; they do not prohibit implementing explicitly defined native tensor stages with original synthetic data and weights. The executable deliverables of the recovery work order ([vr/NATIVE-RECOVERY-ASSIGNMENT.md](vr/NATIVE-RECOVERY-ASSIGNMENT.md)) so far are typed CPU slices with explicit arithmetic semantics and independently derived controls, adapted with attribution from a pinned MIT-licensed public source: the input lift (RGB → 16 features → 32 channels, [NATIVE-INPUT-LIFT.md](vr/NATIVE-INPUT-LIFT.md)), the e4m3 boundary and 2x2 pooling ([NATIVE-PRE-POST.md](vr/NATIVE-PRE-POST.md)), the post-block blend ([NATIVE-BLEND.md](vr/NATIVE-BLEND.md)), the output conversion ([NATIVE-COMPOSE.md](vr/NATIVE-COMPOSE.md)) and the Swin body's scalar primitives ([NATIVE-SWIN-PRIMITIVES.md](vr/NATIVE-SWIN-PRIMITIVES.md)), window partition ([NATIVE-SWIN-WINDOWS.md](vr/NATIVE-SWIN-WINDOWS.md)), e4m3 linears ([NATIVE-SWIN-LINEAR.md](vr/NATIVE-SWIN-LINEAR.md)), the MLP assembly ([NATIVE-SWIN-MLP.md](vr/NATIVE-SWIN-MLP.md)), windowed attention ([NATIVE-SWIN-ATTENTION.md](vr/NATIVE-SWIN-ATTENTION.md)), the block composition (`swin_block`/`swin_core`, [NATIVE-SWIN-BLOCK.md](vr/NATIVE-SWIN-BLOCK.md)), the resampling layers (`swin_downsample`/`swin_upsample_blend` and the `SwinLayer` compositions, [NATIVE-SWIN-RESAMPLE.md](vr/NATIVE-SWIN-RESAMPLE.md)), the residual block core (LayerNorm/im2col conv/pixel shuffle, [NATIVE-BLOCK.md](vr/NATIVE-BLOCK.md)), the typed network graph plan (152 layers/71 blocks ht, 76 layers/31 blocks small; the pinned docstring's 77 is weight-file records, [NATIVE-GRAPH-PLAN.md](vr/NATIVE-GRAPH-PLAN.md)), the bottleneck transformer family ([NATIVE-VIT.md](vr/NATIVE-VIT.md)) and the split Swin family (`SplitFfwd`/`SplitQKVAttn`/`SplitProjPool`/`SplitFinalHead`, [NATIVE-SPLIT-SWIN.md](vr/NATIVE-SPLIT-SWIN.md)), the decoder input stage (`DecInputUpsample`, [NATIVE-DECODER-INPUT.md](vr/NATIVE-DECODER-INPUT.md)), the network dispatch (the plan-row run loop wiring all of the above per the pinned `model.py`, [NATIVE-NETWORK.md](vr/NATIVE-NETWORK.md)) the layer binding/synthetic evaluator (all 18 kinds bound to the kernels over synthetic weight records, [NATIVE-EVAL.md](vr/NATIVE-EVAL.md)) the temporal history carrier ([NATIVE-HISTORY.md](vr/NATIVE-HISTORY.md)) the host-image boundary/image evaluation ([NATIVE-IO.md](vr/NATIVE-IO.md)) and the int4 mixed-network path (the operand kernels, the mixed ViT feed-forward miring, both QKV sites, the SwinSite activation/fold changes and the binary16 side channel with the expansion side-stream, all over the Int4Site record boundary, [NATIVE-INT4.md](vr/NATIVE-INT4.md)). Their synthetic correctness is **not** vendor correspondence, and no software licence grants rights to NVIDIA model data; no proprietary payloads are used. These slices alone are not the migration: the learned-weight file preparation (the `.fix`/`.mu`/`.aq` tables), the temporal counter's network consumer and any game/runtime binding remain outside the tree — the counter's consumer is the documented proprietary-route boundary and the weight tables are the declared host boundary.

Historical handovers still name the predecessor workspace/remote: for all continuing migration work, use this workspace and its `dlss5-linux` origin instead. Do not follow their old publishing destination.

Start with [MILESTONE.md](MILESTONE.md), [vr/CAMPAIGN-HANDOVER.md](vr/CAMPAIGN-HANDOVER.md), [vr/NATIVE-SLOT-REPLAY.md](vr/NATIVE-SLOT-REPLAY.md), and [CHECKPOINT.md](CHECKPOINT.md).

The current externally gated evidence path and resumption criteria are summarised in [NATIVE-MIGRATION-GATE.md](vr/NATIVE-MIGRATION-GATE.md). The audit state was documented at `1f03478`, then clarified in `ffd808c`; subsequent read-only local evidence/artifact checks recorded in `a5a52f6` and `37c251a` found no new qualifying input. The earlier bounded 26/10/2026 source-check checkpoint began at `f4eca7a`; the latest maintenance review began at `0e4c8c0`. A subsequent safe CPU-regression check and handover update at `a060431` still found no authorised target ABI/capture scope or producer-to-launch identity evidence. See the campaign handover and migration gate for the current status and qualifications. This is a blocker record for the proprietary evidence path, not a prohibition on the independent synthetic implementation path and not a completion claim.

## Layout, dependencies and checks

The existing `vr/` directory name is retained to preserve imports, capture paths and historical evidence. Native migration probes live there alongside the inherited reference harness; the directory name does not make VR part of this milestone. The Git history is preserved from `dlss5-vr`; inherited VR source is historical/reference material, not the active deliverable.

This workstation's excluded dependencies and evidence were copied using copy-on-write reflinks, not shared writable directories. They remain local and are **not uploaded**: model/runtime DLLs, assets, captures, prefixes, SDKs, generated binaries and raw logs. A fresh clone is not turnkey. See [CHECKPOINT.md](CHECKPOINT.md) for the dependency boundary. Existing machine-specific runtime manifests may need adaptation before launching a reference application from this workspace.

CPU-only migration regression checks (no GPU/headset or proprietary payload required):

```bash
PYTHONDONTWRITEBYTECODE=1 python3 vr/native-slot-replay-probe.py self-test
PYTHONDONTWRITEBYTECODE=1 python3 vr/ladder-report.py self-test
PYTHONDONTWRITEBYTECODE=1 python3 vr/ladder-names.py self-test
PYTHONDONTWRITEBYTECODE=1 python3 vr/tests/native-input-layout-test.py  # reference plane integrity and row packing
PYTHONDONTWRITEBYTECODE=1 python3 vr/native-input-layout.py vr/logs/reference/run-a  # local size/layout/manifest validation (optional)
bash vr/test-native-input-lift.sh  # typed CPU input-lift slice: strict builds + sanitised build when linkable
bash vr/test-native-pre-post.sh  # typed CPU e4m3 boundary + 2x2 pooling slice
bash vr/test-native-blend.sh  # typed CPU post-block blend slice
bash vr/test-native-compose.sh  # typed CPU output-conversion slice (projection + compose)
bash vr/test-native-swin-primitives.sh  # typed CPU Swin body scalar primitives slice
bash vr/test-native-swin-windows.sh  # typed CPU Swin window-partition slice
bash vr/test-native-swin-linear.sh  # typed CPU Swin e4m3 linears slice
bash vr/test-native-swin-mlp.sh  # typed CPU Swin MLP assembly slice (store, mid projection, residual scale)
bash vr/test-native-swin-attention.sh  # typed CPU Swin windowed-attention slice (deswizzle, softmax, normalise, logits, context)
bash vr/test-native-swin-block.sh  # typed CPU Swin block composition slice (round8, window passes, MLP -> store -> attention, reassembly, f16 wide)
bash vr/test-native-swin-resample.sh  # typed CPU Swin resampling slice (downsample, upsample blend, SwinLayer compositions)
bash vr/test-native-block.sh  # typed CPU residual block core slice (LayerNorm, im2col conv, pixel shuffle, GELU, ConvNormAct)
bash vr/test-native-graph-plan.sh  # typed CPU network graph plan slice (layer table, extent walk, padding, skip wiring)
bash vr/test-native-vit.sh  # typed CPU vit bottleneck family slice (FFN, QKV, grouped global attention, projection)
bash vr/test-native-split-swin.sh  # typed CPU split Swin family slice (grouped FFN, projections, pooled head, shifted-window attention)
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
