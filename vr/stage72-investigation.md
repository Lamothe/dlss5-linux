# Slot 80 capture and weight-binding investigation (24/10/2026)

## Captures and slot selection

Generated with `bash vr/run-feature-probe.sh ladder s80 80` and `s80b 80`. Both traces/captures retained locally; both validate at slot80 and pair bit-identically, same pool/input digests. Pool is 62,684,160 bytes at `0x1cae0000`. `ladder-report diff s79 s80` totals 1,760,641 / 1,759,249 / 1,759,573 changed bytes. `vr/logs/ladder/names-s79-s80.txt` attributes all changes to six slot80 qkv bindings, none unattributed. Frame0: w6 1,171,121, w3 196,552, w1 196,533, w2 196,179, w5 128, w8 128 bytes.

**Important trace indexing:** the trace contains repeated same-geometry qkv launches. The first grid `(32,1,2)` launch at log line 534 is not slot80; its w4 is `0x0cef6c00` and maps to block31.layer2. Slot counting in `vr/capture/nvapi-trace.c` opens at clear-kernel slot0 and increments for each launch; the selected slot80 launch is the fifth QKV at line 574, immediately before `cut-boundary eval=0 slot=80` at line 575. It has w4 `0x0fefc400`. The same cut-adjacent launch appears at lines 969/1359 for evals1/2 and in s80b.

## Verified actual slot80 binding

Resource map entry 0 has VA `0x0b100000`, size 147,719,680. For the actual slot80 w4:

`0x0fefc400 - 0x0b100000 = 0x04dfc400 = 81,773,568`.

The DLL WEIGHTS_HT layout parsed with `weight_layout` reports `block35.layer2.layer` at heap offset 81,773,568, payload 3,145,856 B. The other same-family record `block31.layer2.layer` is at offset 31,419,392; its pointer was from the earlier dispatch, not the slot80 boundary. Therefore the initially suspected mismatch was a trace-selection error, not a capture/contract discrepancy. Boundary sidecar at `vr/logs/ladder/s80/boundary-va.txt` matches the selected cut parameter block, confirming this interpretation.

Proceed with slot80 contract using slot75 qkv as structural template, but check all actual slot80 bindings against the s79/s80 traces and write attribution; retain execute pair and sibling/skip/wrong/tail controls. Preserve these captures. Genesis remains untouched/out of scope.
