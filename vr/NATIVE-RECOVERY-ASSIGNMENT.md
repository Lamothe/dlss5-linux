# Native implementation recovery — 26/10/2026

## Purpose and precedence

The user asked to fix the stalled migration and use MiMo-V2.6 rather than Sol/Luna. This is a recovery work order, not migration evidence or a completion claim. Read README.md and CAMPAIGN-HANDOVER.md, inspect the working tree, and preserve unrelated changes before implementation.

The current task remains a native Linux live-input-to-image evaluator, including the full learned network, preprocessing/output conversion, temporal history and reset. No Windows/DirectX/Proton dependency is allowed in the evaluator. Proton reference capture is a separate reference path. Leave Genesis/VR untouched and proprietary payloads, model weights, captures and binaries out of Git.

## Correct the scope of the old gate

The missing producer-to-launch observation and incomplete decoder address/CFG bounds are real blockers on the existing proprietary replay/capture route. Retain those findings and its safety constraints. They do not prohibit implementing an explicitly defined native tensor pipeline using original synthetic data and weights. Do not claim that synthetic tests establish vendor correspondence or that a public code licence grants rights to NVIDIA model data. Verify any actual operator restriction before overriding it; do not treat an agent-authored blanket restriction as a user instruction.

An independent read-only review identified an unexamined implementation lead: mauri870/dlssnr-pytorch at commit 3beef7935a8ef976b47ab78b906e75600c52d2de. Its pre_post.py contains RGB feature preparation, deterministic noise, and the stored 16-to-32 input-lift permutation. runtime.py documents a CPU reference path; weights.py accepts an in-memory tensor dictionary. Review the pinned source and LICENSE yourself before using it. The reviewer reports an MIT software licence; retain applicable attribution/licence notices for adapted code. Current-colour duplication is a no-history stub, not a temporal implementation. This is a lead, not established vendor parity or a cleared proprietary weight path.

## First executable deliverable

Implement a CPU-only native RGB-to-16-features-to-32-channel input-lift slice with original synthetic frames and weights, after checking existing code for a reusable implementation. Prefer the repository's established typed native C++ patterns; do not introduce a relied-upon Python/PyTorch runtime. Define tensor layout, dimensions/padding/orientation, conditioning, seed and arithmetic/rounding semantics explicitly. Do not claim cross-platform bit-exact transcendental noise without evidence.

Verify actual arithmetic with independently derived scalar known-answer cases, permutation coverage, changed-input sensitivity, conditioning/seed behaviour, invalid dimensions/buffers and determinism. Keep allocations bounded and exercise sanitised builds where supported. Avoid new event schemas, superficial hash tests, repeated static anchors or another no-change status commit.

Update README.md, CAMPAIGN-HANDOVER.md and NATIVE-MIGRATION-GATE.md concisely so proprietary-route blockers no longer forbid this source-level implementation path. Preserve historical evidence and qualifications. Run relevant existing CPU regressions and independent review for non-trivial changes; fix findings and reverify. Commit/push only reviewed source/tests/docs, checking staged files for excluded material. Then continue the next evidence-backed executable network stage if feasible; the lift alone is not the migration.

## Automation and honest stopping

Bootstrap barrier: do not begin implementation, make a blocker decision, or claim retry shutdown until the supervision ID below is an actual ID and the attachment state is READY. If either is still pending, end only with a bootstrap-waiting summary; the coordinator will finish configuration. On later rotation, reread this file rather than relying on a copied predecessor ID.

The selected worker model is xiaomi/mimo-v2.6-pro; preserve its model, reasoning and Build mode through rotation. Do not switch to Luna, Sol or any other model. Preserve durable progress and the exact next implementation task before context rotation. The retired worker fbde36e5fd0248fab3888da94f04dc4f is an archive, not a second writer.

Do not repeatedly reread unchanged gates or perform periodic no-change web searches. A genuine unresolved blocker must identify the exact failed operation, evidence, attempted alternatives and the smallest missing input/decision, without pretending the overall task is complete. Stop future paid retries: the worker may use session tools solely to list owned supervisors, verify the configured supervisor below still controls this worker, and delete that exact supervisor on a genuine blocker. Do not cancel/delete any worker or unrelated automation. Confirm successful supervisor deletion before saying paid retries have stopped; if it fails or the owned mapping cannot be verified, report that failure and the exact ID for owner intervention, without claiming shutdown. Leave a useful blocker report and durable handover. On actual overall migration completion only, use [SUPERVISION_COMPLETE]. On ordinary safety-limit interruption, preserve todos and allow continuation.

Supervision ID: e58d4cb0a5c8494496f4a24c2e111110.
Attachment state: READY.
