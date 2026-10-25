# Native migration external dependency gate — 25/10/2026

## Scope

The active deliverable remains a native Linux live-input→image evaluator, including preprocessing, learned-weight binding, complete graph execution, temporal history/reset and output conversion. This note records the dependency status; it is not a claim of migration completion. VR/Genesis remains a later, separate project.

## Current repository state checked

The evidence audit underlying this gate was performed at `1f03478` (`Update native migration gate handover`); the gate was subsequently clarified and published at `ffd808c` (`Clarify native migration evidence gate`). At the audit, `main` and `origin/main` were aligned and the working tree was clean. A follow-up search of local ignored run manifests and supplied SDK/artifact listings found no later producer→consumer identity evidence or version-matched NR ABI/integration material. That bounded check was read-only: no probe, archive download, capture or GPU execution was performed; its result was recorded in maintenance commit `a5a52f6`. The current `README.md`, `vr/CAMPAIGN-HANDOVER.md`, and `vr/NATIVE-GRAPH-NEXT.md` document the same unresolved boundary:

- Existing harness evidence does not join raw input producers to launch-readable resources. No additional schema, static routing assertion, or address coincidence can supply that missing observed edge.
- The target NR-specific ABI/integration materials and explicit authority to perform the bounded reference capture are absent from the supplied repository evidence.
- Decoder output-store/CFG/address bounds are incomplete; the proposed stored-weight accumulation experiment also requires a separate explicit approval and safety preflight.
- No native evaluation/inference or neural-rendered image output is established. The existing native NGX probe is capability/feature-creation only.

The locally available public Linux NGX SDK headers do not by themselves establish an NR integration ABI. Local proprietary payloads or historical harness captures are not treated as authorisation or as the missing target materials. No capture, GPU probe, or archive download was performed for this status update.

## Decision

There is no evidence-backed implementation step to take on the blocked producer→consumer/runtime path at this time. Do not fabricate ABI bindings, convert synthetic event schemas into evidence, or launch the gated capture/decoder experiment. Reassess that path if either (a) an authorised, version-matched target ABI/integration package and capture scope are supplied, or (b) genuinely new local producer→consumer evidence becomes available and can be inspected within the existing authority and safety boundaries. Keep all proprietary payloads and captures excluded from Git.

This dependency is a blocker on this evidence path, not an assertion that the whole project has no possible future work and not acceptance of the migration. Overall L1/L2/L3 status remains incomplete.

## Resumption checklist

1. If new local producer→consumer evidence appears, inspect its provenance and whether its use is within existing authority; do not treat historical harness captures or synthetic fixtures as new evidence.
2. Otherwise, confirm provenance/version and permitted use of supplied NR target headers/integration materials, record explicit capture scope/authorisation, and run the documented safety preflight before any reference capture.
3. Trace only metadata identities across producer, SetResource, upload/copy/interop and consumer launch; retain missing identity links as blockers rather than inferring them.
4. Implement a validator only after a concrete edge is observed; preserve independent captures and negative controls locally.
5. Continue the full native pipeline, including temporal/reset and live-input image validation; do not call isolated replay completion.
