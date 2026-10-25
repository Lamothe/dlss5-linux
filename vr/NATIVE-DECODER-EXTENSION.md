# Decoder structural extension — (o,i,m) closed form — 13/10/2026

## Scope and result

`native-decoder-extension-probe.py` extends the byte-granularity permutation
map of [NATIVE-DECODER-PERMUTATION.md](NATIVE-DECODER-PERMUTATION.md) from
"23 measured input bytes" to a **validated closed form**, launching the pinned
SM89 `cc_dec_input_upsample_1024_512_fp8` through **native Linux CUDA** on the
RTX 4070 Ti SUPER — no Windows code, DirectX or Proton. This is next-stage
item 1 of that document (the stage gate for `matrix_permutation_recovered`).

**Result: `matrix_permutation_recovered: true`**, scope *"validated on 21 of
4×4096 input bytes (12 single-bit fits, 8 held-out multi-bit exact map matches,
1 identity regression, 21 matrix-byte-0 preimage checks, 3 spot triples);
closed form extrapolates over the remainder by additivity"*. The two execute
runs are bit-identical (405 records, 42,988 triples). **No learned weights
were used and no neural-rendered image is produced.**

## The model

For input byte `n = u*16 + l`, the feeding matrix byte of output byte `o` is
`m(n, o) = B(n) + S(o)`: a fixed bit shuffle `S` (the established
`{0,1,2,3,8,9,10,11}` within-unit pattern plus unit/blocks terms) plus effects
`B(n)` that **compose additively over the set bits of `n`**. Each of the 12
index bits of `n` carries a 4-tuple `(base, row, block_shift, byte_shift)`,
fitted from single-bit training columns:

| Bit | Effect (base, row, block, byte) | | Bit | Effect (base, row, block, byte) |
|-----|---------------------------------|-|-----|---------------------------------|
| 0 | `+1, 0, 0, 0` | | 6 | `0, 0, +8, 0` |
| 1 | `+2, 0, 0, 0` | | 7 | `0, +1, 0, 0` |
| 2 | `0, +4, 0, 0` | | 8 | `0, 0, 0, +4` |
| 3 | `+4, 0, 0, 0` | | 9 | `+16384, 0, 0, 0` |
| 4 | `+16, 0, 0, 0` | | 10 | `+32768, 0, 0, 0` |
| 5 | `+32, 0, 0, 0` | | 11 | `+65536, 0, 0, 0` |

Index bits map: 0,1,3,4,5,9,10,11 → base `+1,+2,+4,+16,+32,+16384,+32768,+65536`;
2 → row `+4`; 7 → row `+1`; 6 → block `+8`; 8 → **byte shift `+4`** (within-unit
bytes `{0,1,2,3,8,9,10,11}` → `{4,5,6,7,12,13,14,15}`).

## The byte-shift discovery (findings honesty)

The first GPU run (`decoder-extension-execute-gcsmSslQ`, **retained as failed
evidence**) failed at training bit 8 (unit 16): the 3-effect model could not
express a response whose within-unit byte pattern was shifted. That run caught
what the stage-4 cross-check had missed — 4 of the 23 prior columns' bit fits
were silent defaults. After extending the model to the fourth (byte-shift)
dimension, **all 8 held-out multi-bit columns matched exactly**, including
`n = 4095` (all 12 bits) across all four impulse partitions. Hold-out inputs
were never used in fitting.

## Verified results (runs `decoder-extension-execute-ZeB4QueU`, `-f0fqeJWC`)

Each run: **810 kernel launches, 3,242 readbacks**, 405 records (381 phase-1
signature/direct, 24 phase-2 extension); both runs **bit-identical**
(oracle-pair) and oracle-valid with `matrix_permutation_recovered: true`.
Checks: 12 training fits self-predict; **8 held-out predictions exact** (2,048
mapped entries each); 1 identity regression (`n = 0`); **21 matrix-byte-0
preimage checks**; 3 spot triples; impulse-only leakage absent; decoupling and
the coupled 512-unit reference enforced; counters `[3]×8`; no changed bytes were
observed outside `[0, 131072)` within the monitored out0 allocation; the
full-region final check was clean. This finite scan cannot detect writes beyond
the allocation or writes matching the poison pattern, and does not establish a
maximum effective address (see `NATIVE-DECODER-STORE-ENVELOPE.md`). The skip control
(`decoder-extension-skip-GKGUc26c`) leaves output untouched and reports
`matrix_permutation_recovered: false` as required.

Oracle (`native-decoder-extension-oracle.py`): re-derives every decode from raw
change ranges with independent inline arithmetic; self-test 25 corruption
controls, 2 pair-reject controls (sample bytes, sample offsets), 3 pair
tolerances, 8 shuffle cross-checks. Probe self-test: 43 model/plan checks, 15
corruption controls, helper-chain regression. Both self-tests pass against the
final source; evidence was re-run after each review-fix round (superseded and
failed runs retained).

## Limitations

* Validation covers **21 of 4×4096 input bytes**; the remainder is additivity
  extrapolation, bounded strongly (not proved) by the all-bits column `n=4095`.
* `m = 0` entries are invisible to signature decoding (zero product); covered
  indirectly by the direct single-byte preimage runs.
* The map is structural: **no learned weights, no semantic channel naming
  beyond the declared layout hypothesis, no native neural-rendered image.**
  Proprietary reuse/publication rights remain unresolved. Swap, driver
  persistence and memory erasure remain uncontrolled.

## Next stage

1. ~~Bind the stored decoder-weight candidate record through this map (decode
   stored weight bytes into the recovered (o, i) structure).~~ **Done:** see
   [NATIVE-DECODER-BINDING.md](NATIVE-DECODER-BINDING.md) —
   `block39.layer0.layer` uploaded byte-identically; 159,296 single-product
   byte-identity checks per run against the stored weight bytes, rotated
   control rejecting the alternative split.
2. Capture Proton reference evaluations for numerical comparison of one native
   evaluation against the working VR PoC.

Containment unchanged: bubblewrap, no network, no home, pinned source
snapshots, hash-verified DLL read as data, 480 s bound, disabled core dumps.
No proprietary payload files are deliberately created.

Evidence: `research/native-linux/runs/decoder-extension-execute-ZeB4QueU`,
`-f0fqeJWC` (each with `oracle.json`, `oracle-pair.json`), skip
`decoder-extension-skip-GKGUc26c`. Retained as failed/superseded:
`decoder-extension-execute-gcsmSslQ` (bit-8 model failure — the defect that
motivated the byte-shift dimension) and earlier review-fix re-runs. Source:
`vr/native-decoder-extension-probe.py`, `vr/native-decoder-extension-oracle.py`,
`vr/run-native-decoder-extension-probe.sh`.
