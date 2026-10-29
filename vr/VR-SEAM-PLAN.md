# VR/Genesis seam integration plan — milestones only (29/10/2026)

Scope of this document: the milestone skeleton for wiring the validated
native evaluator into the Genesis VR/AR scene editor through Genesis's
neural seam. Milestones only — no schedule, no implementation. The headset
work happens with Michael at the machine that holds the Genesis checkout
(platinum; this workspace does not have it). Per `MILESTONE.md`, per-eye
state, stereo correctness, headset behaviour and VR performance are
[dlss5-vr](https://github.com/Lamothe/dlss5-vr) concerns, not Linux
migration acceptance.

Constraints carried forward: Genesis itself is not modified from this
workspace; synthetic weights only (the in-band label travels with any
presented output); the CPU evaluator is seconds-per-frame and can never be
presented as real-time — in VR it is an inspect/slow-compare mode, not a
frame-rate path; no proprietary payloads, captures or binaries in Git.

## Milestone 0 — Preconditions (delivered here; re-verify at the session)

* Native evaluator validated end to end (the `vr/test-*.sh` families).
* Temporal history/reset semantics pinned (`NATIVE-IO.md`, `NATIVE-LIVE.md`).
* The desktop delivery boundary proven by a real host consumer (the Centre
  Stage branch `feature/dlss5-native-seam`, the C mock host at -O2 and -O0).
* Scratch-content independence pinned (`vr/test-native-uninit.sh`) — the
  regression any GPU/VR seam must also satisfy.

## Milestone 1 — Contract alignment (with Michael, at the Genesis machine)

* Read Genesis's in-repo neural seam contract (`vr/NEURAL-SEAM.md` in the
  Genesis checkout) and map its per-eye feature handles/resources to the
  evaluator's plan and input layout.
* **The key open question**: this boundary's arena refuses frames whose
  padded extent exceeds its 320x320 working floor (the documented bound),
  and any headset per-eye extent is larger. Decide: extend the arena bound,
  or target a bounded region (foveated/upscale window) per eye. The plan
  and padding rules (`native-graph-plan.hpp`) are the constraint to design
  against.
* Decide the ownership split with dlss5-vr (stereo/per-eye state lives
  there; this workspace supplies the evaluator and its contracts).

## Milestone 2 — Desktop stereo dry run (no headset)

* Two evaluator handles (per-eye temporal history; the counter schedule is
  per handle), fed from the desktop renderer's low-res per-eye targets.
* Side-by-side presentation of the two processed frames with the in-band
  label; frame-by-frame stepping with visible latency (an inspect mode).
* Determinism and reset verified per eye (the mock host's contracts, stereo
  shape).

## Milestone 3 — Headset session (with Michael; WiVRn/Quest)

* Present per-eye results through the seam in-headset in the slow inspect
  mode; verify passthrough/horizon-lock interplay does not perturb the
  captured inputs.
* Record the latency budget honestly (seconds/frame on CPU); agree what
  belongs to the dlss5-vr GPU path for anything frame-rate bound.

## Milestone 4 — Acceptance

* The seam's own smoke set extended (Genesis's `--smoke-*` family) with the
  stereo dry run; stereo correctness checks; the label present on every
  presented frame; docs updated in both repositories.
