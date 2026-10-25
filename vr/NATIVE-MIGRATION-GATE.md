# Native migration external dependency gate — 25/10/2026

## Scope

The active deliverable remains a native Linux live-input→image evaluator, including preprocessing, learned-weight binding, complete graph execution, temporal history/reset and output conversion. This note records the dependency status; it is not a claim of migration completion. VR/Genesis remains a later, separate project.

## Current repository state checked

The evidence audit underlying this gate was performed at `1f03478` (`Update native migration gate handover`); the gate was subsequently clarified and published at `ffd808c` (`Clarify native migration evidence gate`). At the audit, `main` and `origin/main` were aligned and the working tree was clean. A follow-up search of local ignored run manifests and supplied SDK/artifact listings found no later producer→consumer identity evidence or version-matched NR ABI/integration material. That bounded check was read-only: no probe, archive download, capture or GPU execution was performed; its result was recorded in maintenance commit `a5a52f6`. The current `README.md`, `vr/CAMPAIGN-HANDOVER.md`, and `vr/NATIVE-GRAPH-NEXT.md` document the same unresolved boundary.

A later local staged-package inventory was published at `bf4fdbf`: the workspace has the 310.8 NR archive and Windows demo archive, but no downloaded Linux demo archive or Streamline NR header. Only filenames/presence were checked; proprietary archive contents were not opened. This is a local inventory statement, not proof of global unavailability or lack of access rights. In particular, the separate staged `dlssnr-linux.addon64` was not treated as a Linux SDK/header, and its contents were not inspected. The public NVIDIA/DLSS GitHub release API for `v310.9.16` lists a Linux demo asset `ngx_dlss_demo_linux.zip` (96,907,020 bytes; SHA-256 `985019f1307c0367d0d4dff35f06b2615c5c3ad6b469ffec5b0c16e28758610d`) alongside the Windows demo. The release notes say only bug fixes and stability improvements. Asset metadata establishes that the Linux archive is published, not what it contains; it has not been downloaded or inspected here. The release tag's tracked-source audit and `vk_gltf_renderer`'s beta NR SDK requirement remain as separately bounded evidence; neither proves or disproves NR headers inside the Linux demo archive.

- Existing harness evidence does not join raw input producers to launch-readable resources. No additional schema, static routing assertion, or address coincidence can supply that missing observed edge.
- Version-matched target NR ABI/integration material and explicit authority to perform the bounded reference capture have not been supplied in the project evidence. Local binary/add-on presence does not substitute for either.
- Decoder output-store/CFG/address bounds are incomplete; the proposed stored-weight accumulation experiment also requires separate explicit approval and safety preflight.
- No complete native live-input→image evaluation/inference or neural-rendered image output is established. Bounded native kernel/decoder checks exist, and the native NGX probe performs capability/feature-creation only.

The bounded official Streamline source check was recorded in `a30bee1`: changelog entry 2.14.0 lists `sl.dlss_nr`, but its Vulkan bullet is specifically for Reflex (`sl.reflex`/`VK_NV_low_latency2`), while the current programming guide describes a 64-bit Windows requirement. Neither establishes native Linux/Vulkan NR support or provides a Linux NR ABI. [Streamline issue #105](https://github.com/NVIDIA-RTX/Streamline/issues/105) states “Currently Proton is supported” and avoids commenting on future plans; that is not evidence of native Linux support. These bounded public-source observations do not rule out separately distributed or restricted target materials.

## Decision

There is no evidence-backed implementation step to take on the blocked producer→consumer/runtime path at this time. Do not fabricate ABI bindings, convert synthetic event schemas into evidence, inspect proprietary archives, or launch the gated capture/decoder experiment without the required authorisation and safety gates. Reassess that path if either (a) authorised, version-matched target ABI/integration materials and explicit capture scope are supplied, or (b) genuinely new local producer→consumer evidence becomes available and can be inspected within existing authority and safety boundaries. Keep all proprietary payloads and captures excluded from Git.

This dependency is a blocker on this evidence path, not an assertion that the whole project has no possible future work and not acceptance of the migration. Overall L1/L2/L3 status remains incomplete.

## Resumption checklist

1. If new local producer→consumer evidence appears, inspect its provenance and whether its use is within existing authority; do not treat historical harness captures or synthetic fixtures as new evidence.
2. Otherwise, confirm provenance/version and permitted use of supplied NR target headers/integration materials, record explicit capture scope/authorisation, and run the documented safety preflight before any reference capture.
3. Trace only metadata identities across producer, SetResource, upload/copy/interop and consumer launch; retain missing identity links as blockers rather than inferring them.
4. Implement a validator only after a concrete edge is observed; preserve independent captures and negative controls locally.
5. Continue the full native pipeline, including temporal/reset and live-input image validation; do not call isolated replay completion.
