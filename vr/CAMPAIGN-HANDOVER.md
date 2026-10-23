# DLSS5 native slot replay campaign handover

Updated 24/10/2026 (Australia/Sydney). Repository `/home/michael/Projects/dlss-testbed`, remote `git@github.com:Lamothe/dlss5-vr.git`, branch `main`.

## Current verified position

- Inspection began at clean, synchronised HEAD `780c3d9335f21cf3907938f5c2ebea3bb70e0694`. No Genesis files were touched.
- Slots 24–85 are previously published. Slot86 implementation is now present as an uncommitted contract in `vr/native-slot-replay-probe.py`; documentation changes are also pending. No commit/push has yet been made for slot86.
- Slot86 contract maps attention chained/fp8 at psize64, grid (32,1,1), block (32,4,1); q/k/v at w0/w1/w2; output w3 (196,608 B); weights `block36.layer3.layer` at w4 (2-byte payload, heap offset 97,508,352; reported SHA-256 `45ddf4c3a24603c25bef714a04df5aafa24c168c553c9375b020cab502d5e64a`); sync-in w5; sync-out w6 (256 B); constant w7=(12,16). Contract code validates parameter trace, capture sidecar, resource map, PE weight layout and whole-record placement before launching.
- Capture validator succeeded for `vr/logs/ladder/s86`; write-set diff s85→s86 was 196789/196790/196798 changed bytes. `ladder-names.py names vr/logs/ladder/s85 vr/logs/ladder/s86 vr/logs/capture/nvapi-trace-ladder-s85.log vr/logs/capture/nvapi-trace-ladder-s86.log` completed with all changes attributed to w3 and w6 and zero unattributed.
- Two fresh native execute runs `/tmp/dlss86-exec-a` and `/tmp/dlss86-exec-b`: six entries each, no enforced failures, bit-identical pair, run SHA-256 `8a78e5ff1777437f210af110a45638f5fc84c2419dc35b3760a43272cb449350` each. Chained entries match the full reference pool; all six main-output comparisons match. Skip, wrong-input and tail modes each completed six entries with no enforced failures at `/tmp/dlss86-{skip,wrong,tail}`. These are temporary evidence paths and must be recreated before depending on them after this session.
- Final verification after source and documentation edits: probe self-test 774 known answers / 40 corruption controls (756 contract-shape checks across 63 contracts); ladder-report 19 controls; ladder-names 14 controls; Python compilation across all three scripts; and `git diff --check` all pass.

## Important qualification

The wrong control for slot86 rotates query input w0; it leaves the 2-byte weight record unchanged. `vr/logs/ladder/s86/record-mutation-experiment.py` declares `NOT_RUN_NO_MUTATION_HOOK`; it did not inject bit flips. Therefore no weight-insensitivity or dead-binding conclusion is established for slot86. Historical claims for slots 61/66/71/76 are the original four-placeholder scope; later claims (e.g. slot81) remain finite-condition claims/hypotheses, not universal sensitivity conclusions. Avoid describing any record as dead based only on query corruption or tail poisoning.

## Remaining work

1. Obtain fresh independent review of the current full slot86 diff; address any concrete findings. The current docs and verification record state 63 contracts, 63/156 coverage, 774 known answers / 40 controls, ladder-report 19 and ladder-names 14 controls; tests, compilation and diff check were rerun after edits.
2. Commit and push only after the review passes; confirm the published commit and clean/synchronised working tree.
3. Only after publication proceed to slot87, following capture→validate/pair→write-set/names→contract→execute-pair/controls→tests→review→publish. Slot87 is a projection by inventory, but inspect live evidence; do not assume its contract. Campaign remains incomplete.
