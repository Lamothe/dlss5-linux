# Decoder weight-record binding — 13/10/2026

## Scope and result

`native-decoder-binding-probe.py` binds the stored record **`block39.layer0.layer`**
to the pinned SM89 `cc_dec_input_upsample_1024_512_fp8`, launching it through
**native Linux CUDA** on the RTX 4070 Ti SUPER — no Windows code, DirectX or
Proton. The record's payload is uploaded **byte-identically** as the weights
region and the kernel's single-product responses are checked against the
product prediction of the validated `(o, i) -> matrix-byte` closed form
([NATIVE-DECODER-EXTENSION.md](NATIVE-DECODER-EXTENSION.md)) applied to the
stored weight bytes.

**Result: `weight_record_binding_verified: true`.** Across 80 single-byte
impulses (`x = +1.0` / `−1.0`, 10 input bytes × 4 partitions × 2 signs) plus a
zero-impulse base run, **159,296 nonzero product checks per run matched the
stored-weight prediction byte-for-byte**, and every zero-class position held a
zero encoding. Both execute runs are bit-identical (81 entries,
oracle-pair). This stage **uses stored learned weight bytes** (bounded
model-weight computation, following the [NATIVE-FP8.md](NATIVE-FP8.md)
stored-record precedent) and is **not** a neural-rendered image.

## Record layout (data-side evidence)

The record declares tag 1, dimensions `[262656]`, payload 525,312 bytes =
262,656 × 2. The payload splits as:

| Span | Content | Evidence |
|---|---|---|
| `[0, 524288)` | E4M3 matrix (the kernel's coefficient region) | **0** E4M3-NaN bytes (like the FP8 projection matrix); 15,560 zero-class bytes; 68 distinct byte values (trained alphabet) |
| `[524288, 525312)` | 512 FP16 scales at the kernel's word region `+0x80000` | 512 finite FP16 values (0 NaN/Inf, 395 distinct); **9** bytes that would be E4M3-NaN — so the tail is not matrix data |

The dimension count reads as 262,144 matrix FP16-pairs + 512 FP16 scales,
matching the FP8 projection records' convention (scales follow the matrix).
Under byte-identical upload the tail lands exactly at `+0x80000`, where
`launch.WEIGHTS_BYTES`' established word region reads. **The tail's semantics
are unverified** (`tail_semantics_verified: false`): word tables do not affect
the tested zero-skip contract.

## Product semantics under test

With a single nonzero FP8 input byte `x` at `(p, u, l)`, each influenced output
byte receives exactly one product `x · w` (the
[NATIVE-DECODER-PERMUTATION.md](NATIVE-DECODER-PERMUTATION.md) single-feeder
premise), where `w` is the stored byte at `p*131072 + m(u*16+l, o)`. With
`x = ±1.0` the product is exact in E4M3, so the expected output byte is the
stored weight byte itself (sign bit flipped for `−1.0`) — a **byte-identity**
check with no rounding ambiguity. Zero-class weight bytes (`0x00`/`0x80`) and
non-fed positions expect a zero encoding; `0x80` signed-zero variants are
recorded as findings (none were observed).

## Verified results (runs `decoder-binding-execute-R0jFiSdr`, `-8uFILFfH`)

Each run: **162 kernel launches** (81 decoder + 81 counter resets), **649
readbacks**, 81 entries (1 zero-impulse base + 80 impulses); both runs
**bit-identical** (oracle-pair, 81 entries) and oracle-valid.

- **159,296 nonzero product-identity checks per run, all exact.** Predicted
  counts per entry (1,960–2,020 of the 2,048 fed positions; the remainder are
  zero-class weight bytes) are independently re-derived by the oracle from the
  hash-verified DLL; the oracle also pins the entry name plan (81/17 entries
  per mode) and the launch/readback arithmetic (2 launches and 8 readbacks per
  entry, +1 final full-region readback).
- **Zero-class discipline:** every zero-class position held `0x00` (zero
  `0x80` signed-zero variants across all runs); the 4,096-byte poison margin
  stayed intact in every entry (enforced); counters `[3]×8` everywhere
  (enforced); main, skip, weights, partials, out3, out2 immutable and all
  guards intact (enforced); writes confined to `[0, 131072)` with a final
  full-region check (enforced).
- The checks cover every fed position of the closed form per entry — including
  entries whose feeding matrix byte is `m = 0` (read from the partition base),
  whose invisibility to bit-signature decoding was a stage-4/5 evidence gap.

## Controls

- **`skip`** (`decoder-binding-skip-Ouv9RfJe`): one counter-reset launch only
  (`kernels_launched: 1`, `decoder_launches: 0`); counters `[−1]×8`, every
  buffer byte-identical to its upload, all guards intact (enforced).
- **`wrong`** (`decoder-binding-wrong-flPv3uf3`): the matrix partitions are
  rotated circularly **inwards by 1,024 bytes** (the record's alternative
  `k = 1,024` split). 17 entries: the run matches the rotated reference
  byte-identically **and** the intended `k = 0` result is rejected in every
  **impulse** entry (16 of the 17 — the zero-impulse base entry has no distinct
  intended result), both prediction-level (`intended_k0_prediction_differs`)
  and observationally (`intended_k0_observed_rejected`, zero-class tolerant).
  The top-level `intended_k0_rejected` aggregates the impulse entries. The
  control reports `rotated_control_passed: true` and
  **`weight_record_binding_verified: false`** — only the byte-identical
  execute runs claim binding. The rotation stays inside the NaN-free matrix
  region, so classification remains well-defined. The oracle pins the upload
  rotation to the mode (0 for execute, 1,024 for wrong) in every record.
- **Oracle** (`native-decoder-binding-oracle.py`): verifies the DLL bytes
  against the pinned SHA-256 before deriving anything, re-derives the closed
  form with inline arithmetic (no import of the probe or extension modules),
  re-extracts the record, recomputes every entry's prediction digest and the
  metadata aggregates, and re-checks them against the evidence. Evidence
  equality is attested by SHA-256 over sorted `(offset, expected byte)` pairs —
  **no stored weight byte values are written to evidence**, including failure
  paths (mismatch messages report offsets only)
  (`payload_byte_values_written_to_evidence: 0`, checked in every mode). The
  oracle pins the mode and sign values to their allowed sets, pins the entry
  name plan, and binds each record's self-declared fields to that plan before
  any prediction is re-derived, with type-pinned counts and counters (malformed
  records fail closed). Oracle self-test: 8 model
  known answers, 3 digest controls, **38 corruption controls rejected**
  (record fields, sign/mode claims, plan names, name↔field swaps, type
  impostors, rotation claims, launch arithmetic, metadata, skip and wrong-mode
  verdicts) over full-plan fixtures.
  Probe self-test: launch regression, 8 model known answers, 6 expected-map
  known answers, rotation sensitivity on a varied payload, plan coverage
  (80 entries, 10 keys × 4 × 2), digest controls.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 vr/native-decoder-binding-probe.py self-test
PYTHONDONTWRITEBYTECODE=1 python3 vr/native-decoder-binding-oracle.py self-test
bash vr/run-native-decoder-binding-probe.sh execute
bash vr/run-native-decoder-binding-probe.sh skip
bash vr/run-native-decoder-binding-probe.sh wrong
```

## Retained failed / superseded runs

- `decoder-binding-execute-38faNb3j` (exit 1): all 81 entry identity checks
  passed, but the **final full-region guard check failed** — the probe omitted
  the initial full 256 MiB out0 poison upload that establishes both guard
  zones (only per-entry window uploads were performed). Fixed; retained as
  failed evidence. The entry-level successes in this run are superseded.
- `decoder-binding-execute-2Immlo7r` (exit 1): the probe passed; the runner's
  oracle pass could not find `output.json` inside the sandbox (it was captured
  on the host). Fixed by publishing the report into the evidence directory
  before the oracle pass. Retained as failed evidence.
- `decoder-binding-skip-p8jNUZaQ` (exit 1): the probe passed; the oracle's
  skip check demanded zero kernel launches, contradicting the established
  convention that `kernels_launched` includes the counter reset. Fixed with an
  explicit `decoder_launches` field (skip: 0; runs: one per entry). Retained as
  failed evidence.
- Oracle self-test defect (no run): the self-test's periodic payload
  (period 256) made the 1,024-byte rotation invisible (1024 ≡ 0 mod 256), and
  a probe self-test distinctness check was vacuous for uniform payloads (keys
  sharing a fed set collide). Both fixed before the final runs.
- Superseded evidence (re-run so evidence matches the current source exactly):
  cycle 1 (report-field fix `decoder_launches`):
  `decoder-binding-execute-blGSZArx`, `-1KM3rdSz`, `-EMSCTNbP`,
  `decoder-binding-skip-p8jNUZaQ`, `decoder-binding-wrong-c6XWLKAW`
  (`skip-p8jNUZaQ` also appears under retained failures — the run is both a
  failed run and a superseded generation);
  cycle 2 (review-1 fixes: offsets-only failure messages, mode-pinned
  rotation, corrected verdict fields, oracle DLL-hash verification):
  `decoder-binding-execute-SyBgpb9M`, `-GZf7wjFD`, `-skip-B18PeYF1`,
  `-wrong-7Zqk7dL5`; cycle 3 (review-2 fixes: full-plan oracle fixtures,
  mode/sign/plan pinning, containment check in every mode, metadata flag
  rename): `decoder-binding-execute-RSsPk6kM`, `-5H242Ll0`,
  `-skip-arbsHobv`, `-wrong-uufgbT1A`; cycle 4 (review-3 fixes: name↔field
  binding, type pinning, pair provenance):
  `decoder-binding-execute-mGvAiGdR`, `-8udYFa7O`, `-skip-epwiOrtP`,
  `-wrong-SoDX5uAl`. Current evidence: `decoder-binding-execute-R0jFiSdr`,
  `-8uFILFfH`, `-skip-Ouv9RfJe`, `-wrong-flPv3uf3`, each with `oracle.json`,
  plus `oracle-pair.json` (with run-name provenance) on the `R0jFiSdr`
  evidence directory.

## Review

Independent review round 1: **FAIL (3 must-fix, 7 advisory)** — the oracle did
not hash-verify the DLL bytes it derived predictions from, the upload rotation
was not pinned to the mode (leaving the byte-identical-upload claim
unverified), and the per-entry count range was misstated. Advisory findings
included weight byte values in failure-path messages (containment leak),
prediction-vs-prediction `intended_k0_rejected` semantics, wrong mode claiming
binding, an overstated skip-path `metadata_checked`, a misnamed helper, control
gaps and doc/runner nits. All were addressed; the oracle self-test grew from 9
to 23 corruption controls, and all evidence was re-run against the corrected
source.

Independent review round 2: **FAIL (1 must-fix, 4 advisory)** — all round-1
findings verified resolved; the new findings were an overstated wrong-mode
rejection claim (the zero-impulse base entry has no distinct intended result),
the containment flag unchecked in skip mode, unpinned mode/sign values,
self-referential launch pins (entry plan unpinned) and an evidence-internal
flag contradiction (`learned_weights_used` true in metadata, false at report
level). All addressed: the wrong-mode claim is scoped to the 16 impulse
entries, the oracle pins mode/sign/plan and checks containment in every mode,
the metadata flag was renamed `candidate_record_contains_learned_weights`,
and the self-test fixtures now span the full experiment plan (31 corruption
controls). All evidence was re-run against the corrected source.

Independent review round 3: **FAIL (1 must-fix, 5 advisory)** — all round-2
findings verified resolved; the must-fix was a documentation error (the
`oracle-pair.json` location was misattributed), plus advisories on name↔field
binding, self-test coverage gaps, type pinning, pair provenance and a doc
footnote. All addressed: the oracle binds each record's self-declared fields
to its plan name before re-deriving predictions, type-pins counts/counters/
containment flags with fail-closed malformed-record handling, the self-test
grew to 38 corruption controls (including name↔field swap and type-impostor
controls), and pair artifacts now carry run-name provenance. All evidence was
re-run against the corrected source. Round 4 was skipped as unnecessary: the
remaining changes are provenance strings and documentation.

## What this establishes — and what it does not

Established: the stored record's bytes serve as the decoder's weight matrix
under the recovered `(o, i, m)` map — the kernel's single-product responses
equal the stored weight bytes exactly (sign-aware) over 159k checked positions
per run, with the alternative split rejected by a rotated control. Numerical
compatibility makes the record a **bound candidate** for this operation; as in
the FP8 stage, it does not independently prove the record's exact position in
the Windows runtime's graph.

Not established: multi-term accumulation semantics (only single products were
tested), the 512 FP16 scales' semantics, magnitude linearity for non-`±1.0`
inputs, the model's graph position, and **any native neural-rendered image**.
Proprietary reuse/publication rights remain unresolved. Swap, driver
persistence and memory erasure remain uncontrolled.

## Next stage

1. Capture Proton reference evaluations (controlled colour/depth/motion
   inputs from the working VR PoC) and compare one native evaluation against
   the vendor reference — the gate item that converts structural recovery into
   a validated pipeline.
2. Establish multi-term accumulation and the FP16 scales' role if the
   comparison requires them.

Containment unchanged: bubblewrap, no network, no home, pinned source
snapshots, hash-verified DLL read as data, 480 s probe bound (120 s oracle),
disabled core dumps. No proprietary payload files are deliberately created;
evidence contains digests, counts and verdicts only.

Evidence: `research/native-linux/runs/decoder-binding-*`. Source:
`vr/native-decoder-binding-probe.py`, `vr/native-decoder-binding-oracle.py`,
`vr/run-native-decoder-binding-probe.sh`.
