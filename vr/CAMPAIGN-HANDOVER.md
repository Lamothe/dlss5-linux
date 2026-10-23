# DLSS5 native slot replay campaign handover

Updated 24/10/2026 (Australia/Sydney). Repository `/home/michael/Projects/dlss-testbed`, remote `git@github.com:Lamothe/dlss5-vr.git`, branch `main`.

## Published state

- Slot86 was independently reviewed, published in `77ce2c1c9c07a4a9221be4b277cc920dddd34fc2`, and handover corrected in `b7699dc24f109c9e74067083e059fa25a813d1a3`. Both were pushed; later slot87 work is uncommitted. Genesis was not modified.
- Published replay coverage is 63/156, slots 24–86. Probe tests: 774 known answers / 40 corruption controls (756 contract shape checks); ladder-report 19 controls; ladder-names 14 controls. Python compilation and diff check passed after slot86 final edits.
- Slot86 contract: `cc_vit_1d_attention_chained_fp8`, psize64/grid(32,1,1)/block(32,4,1); q/k/v w0/w1/w2; output w3; w4 is `block36.layer3.layer`, 2 bytes at heap offset 97,508,352, SHA-256 `45ddf4c3a24603c25bef714a04df5aafa24c168c553c9375b020cab502d5e64a`; sync-in w5, sync-out w6 (256 B), constant w7=(12,16). Details in `vr/NATIVE-SLOT-REPLAY.md`; source in `vr/native-slot-replay-probe.py`.
- Slot86 capture validator passes and its s85→s86 diff is fully attributed to w3 and w6 (196789/196790/196798 changed bytes). Native execute pair at `/tmp/dlss86-exec-{a,b}` bit-matches, SHA-256 `8a78e5ff1777437f210af110a45638f5fc84c2419dc35b3760a43272cb449350`; six chained entries match full pools. Skip/wrong/tail each covered six entries. These `/tmp` artefacts are ephemeral.

## Slot86 qualification

The wrong control rotates query input w0 and does not mutate the 2-byte weight record. `vr/logs/ladder/s86/record-mutation-experiment.py` says `NOT_RUN_NO_MUTATION_HOOK`; no slot86 weight-insensitivity/dead-binding conclusion is supported. Historical scope named four 2-byte placeholders (slots61/66/71/76). Later placeholders must be qualified by actual evidence; query corruption or tail poisoning does not establish weights unread.

## Slot87 live status (not yet replayed)

- Captures `vr/logs/ladder/s87` and `s87b` were independently produced with traces `vr/logs/capture/nvapi-trace-ladder-s87.log` and `...s87b.log`. Both validate at slot87; pair passes bit-identically across all pools and input sidecars: pool FNVs `0x74044d753ed21000`, `0x773ecb4a9070a80a`, `0xdf844c16948598f4`; inputs match s86 (`fb51a175911c82f5`, `33533d64f7d84205`, `227ea3796d10f325`).
- `ladder-report.py diff s86 s87` changed totals: 586723, 586934, 586810 bytes; first changed range [88576,88640), last ends at 60324864. `ladder-names.py` attributes four regions/frame, zero unattributed: w5 changed 390042/390271/390150 bytes, w2 196553/196535/196532 bytes, w4 64 each, w7 64 each. These sum exactly to the per-frame totals.
- `ladder-names.py names s86 s87 ...` attributes all changed bytes, zero unattributed, to four regions per frame: slot87 `cc_vit_1d_projection_chained_fp8` w5 at 46,808,064 (3,932,160 B; changed 390042/390271/390150), w2 at 60,128,256 (196,608 B; changed 196553/196535/196532), w4 at 88,576 (64 B) and w7 at 89,088 (64 B). The ladder-names report shows a slot88 w6 alias at the w7 region; this is only a candidate handoff until slot88 read/role analysis establishes its meaning. No contract or native execution is implemented yet.
- Slot87 parameter trace reports `psize=72`, grid `(16,1,4)`, block `(32,4,1)` and words w0–w8. Interpret addresses, extents, and the two 64-byte outputs only after reconciling sidecar, captured resource map, downstream aliases and the DLL weight layout; do not assume the sync/data classification.
- `ladder-report pair s86 s87` was correctly rejected as different cut points; use same-slot s87/s87b pair (already passes). Do not treat cross-slot pair rejection as failed reproducibility.

## Next steps

1. Inspect slot87 trace/manifest/parameter block/resource map and DLL record walk; establish exact psize, grid/block, word classes, weights record identity, all reads/writes, sync extents, and control target before adding contract87.
2. Implement minimal contract from the established pattern. Run native execute twice, pair, full-pool/primary-output checks, plus skip/wrong/tail controls; retain reproducible repository evidence according to campaign conventions, not only `/tmp`.
3. Run probe/report/names self-tests, Python compilation, diff check; reconcile coverage/docs to 64/156 only after replay succeeds. Fresh independent review; commit/push and verify clean remote state.
4. Continue autonomously through subsequent slots by capture→validate/pair→write-set/names→contract→execute pair/controls→verification→review→publication. The authorised campaign is incomplete.
