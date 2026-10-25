# DLSS5 native Linux migration — current handover

Updated 25/10/2026 (Australia/Sydney). Active repository `/home/michael/Projects/dlss5-linux`, remote `git@github.com:Lamothe/dlss5-linux.git`, branch `main`. Verified pre-edit HEAD: `e8e3437ee68ea446f772a45aabfdcc083a149ce1` (`main`, `origin/main`, `origin/HEAD` aligned; clean). This handover replaces older pointers; historical evidence remains in the linked reports.

## Goal and boundaries

Build and validate a native Linux evaluator from live inputs through preprocessing, learned-weight bindings, full execution graph, temporal history/reset and output conversion, producing neural-rendered images without Windows host code, DirectX or Proton in the evaluator. Proton reference capture is allowed only when authorised. Kernel replay alone is intermediate evidence. Do not modify Genesis or build another VR host. Keep proprietary payloads, weights, captures, binaries and assets out of Git; preserve containment/rights qualifications. Use only this workspace and remote.

## Current verified migration state

- Replay campaign: 68 contracts, slots 24–91, out of archival 156-launch inventory; not overall completion. Slot 91 remains latest qualified replay; no new contract absent paired captures, attribution and controls.
- Bounded native GPU results include one stored learned-weight record's single-product decoder checks, byte-exact replay of captured decoder-boundary outputs for tested variants/frames, and a constructed projection→bridge→repack chain. They are not the proprietary graph or live-input neural image output. See `NATIVE-DECODER-BINDING.md`, `NATIVE-VENDOR-COMPARE.md`, `NATIVE-CHAIN.md`.
- Source-only raw-input identity investigation is exhausted against current authorised material. Available traces do not join raw producer resources to consuming launch. Do not repeat schemas, fixtures, hash scaffolds or decoder comparisons. That path requires explicit authorisation for version-matched capture and target ABI/materials, or genuinely new local producer→consumer evidence (`NATIVE-GRAPH-NEXT.md`, `NATIVE-HOOK-PLAN.md`).
- Existing NGX native probe checks capability and feature creation only; it performs no feature evaluation, inference or image output. No new capture or GPU probe is authorised. Accumulation/FP16-tail experiment requires explicit operator approval, current safety preflight and complete output-store/CFG/address/launch bounds first.
- The overall native live-input→image evaluator, full semantic layouts/weights, temporal behaviour and reset remain unestablished. VR/Genesis integration is later and untouched.

## Static graph and decoder evidence

`NATIVE-GRAPH.md` reports conditional outer source indices `[38,30]` and selectors `[0,1]` feeding an upsampler candidate. Its append schedule `[9,6,8,8,8]` yields conditional candidate slots 23–30 (split-Swin) and 31–38 (primary transformer), not global runtime descriptor IDs. The checker explicitly leaves global numbering, actual tensor indices and CUDA addresses unverified. Do not resume broad global-numbering/extra append-anchor work as if that limitation were merely missing tests.

A fresh authorised-source, no-GPU graph inspect ran this session through `vr/run-native-graph-probe.sh inspect`, evidence `research/native-linux/runs/graph-inspect-42qiEUZK` (local, do not commit). It succeeded with `static_contract_verified=true`, conditional indices 38/30, `global_descriptor_numbering_verified=false`, `actual_tensor_indices_verified=false`, `cuda_addresses_verified=false`, `full_graph_recovered=false`, and zero kernels launched. This reproduces existing results; it is not new graph progress. The graph CPU self-test passed (1 parser and 1 configuration known answer; 2 and 3 negative controls rejected).

`NATIVE-DECODER.md` and `native-decoder-probe.py` already recover the bounded host decoder-argument/variant/backend contract, allocation planner consumer and part of binder pointer construction. The consumer reads `state+0x238` as a locator/reference, resolves through subsequent table lookups, then reads signed 32-bit feature indices and scales those by 16 into planner entries. The *writer/publication of the +0x238 field*, the locator tables' ownership, and record→group→layer association remain explicitly unverified. The extensive decoder contract (938 earlier corrupted-anchor controls plus arena and temporal extensions) must not be repeated.

### Next narrow static question

Investigate only whether a source-grounded writer/publication of the binder-read `state+0x238` field can be identified in already authorised pinned source. Trace the field write, owner/construction and checked caller path; keep the resolved feature-index table distinct from the field itself. Preserve unresolved table ownership and record/group/layer association. If existing disassembly does not include the writer, extend only to its immediate proven caller/callee; stop rather than broad speculative scanning. Do not infer runtime binding from the consumer, add a synthetic fixture instead of evidence, or run capture/GPU probes. Update `NATIVE-DECODER.md` and the handover only if a genuinely source-backed finding is established; otherwise record the precise unavailable writer boundary without fabricating progress.

## This session's checks and verification

The slot-replay self-test passed 834 known answers and rejected 40 corruption controls, with zero kernels launched. Graph self-test passed. Decoder CPU self-test passed: 9 sizing answers, 616 arithmetic comparisons, 9 invalid-domain and 3 incorrect-execution rejections, arena arithmetic controls, 4 grid answers and malformed metadata controls. `git diff --check` passed. An independent reviewer checked the preceding handover diff, flagged and received corrections for commit provenance and exact `+0x238` semantics, then passed the final wording. No source edits or probes beyond the graph inspect above were made.

## References

`NATIVE-SLOT-REPLAY.md`, `NATIVE-GRAPH.md`, `NATIVE-GRAPH-NEXT.md`, `NATIVE-HOOK-PLAN.md`, `NATIVE-CHAIN.md`, `NATIVE-DECODER.md`, `NATIVE-DECODER-BINDING.md`, `NATIVE-VENDOR-COMPARE.md`, `NATIVE-DECODER-STORE-ENVELOPE.md`, `NATIVE-DECODER-IO.md`, `TEMPORAL-HISTORY.md`. Recheck branch, worktree and origin before future edits/pushes.
