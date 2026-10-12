# Decoder byte-granularity permutation resolution — 12/10/2026

## Scope and result

`native-decoder-permutation-probe.py` resolves **(output byte, input byte,
matrix byte) triples** for the decoder weight region by launching the pinned
SM89 `cc_dec_input_upsample_1024_512_fp8` through **native Linux CUDA** on the
RTX 4070 Ti SUPER — no Windows code, DirectX or Proton. This is the stage the
weight-binding gate calls for ("resolve (o, i) per physical byte", [NATIVE-
DECODER-SWEEP.md](NATIVE-DECODER-SWEEP.md) next-stage item 1).

**Result: 47,076 triples resolved and validated across 23 probed input bytes,
with complete matrix-byte coverage per probed input byte.** It does **not** use
learned weights and produces **no neural-rendered image**; the semantic naming
of the input positions/channels and the extension to unprobed input bytes
remain open.

## Method

With a single nonzero FP8 input byte, each influenced output byte receives
exactly one product from one matrix byte, so:

* **Bit-signature runs** (17 per input byte; `PARTITION_BYTES == 2**17`): code
  matrix byte `m` iff bit `b` of `m` is set; output byte `o` responds in run
  `b` iff bit `b` of its feeding `m(o)` is set. The response pattern decodes
  `m(o)` exactly. Every nonzero matrix byte has at least one bit, so coverage is
  complete per probed input byte **within its coupled matrix partition**.
* **Direct runs** re-measure decoded pairs independently: single matrix byte +
  single input byte must change exactly the decoded preimage (8 own pairs, 2
  negative-code sign pairs requiring exact FP8 sign-bit mirroring, 8 cross-input
  pairs predicted from the *other* input's decode — including empty
  predictions).

Two checks support the single-feeder premise: per input byte every responding
output byte must return the **identical FP8 value in every responding signature
run** (detects sums of distinct feeding matrix bytes), and every direct run is a
true single product whose response must **equal the signature response** for the
same (input, output) byte, sign-adjusted. A matrix byte feeding one output byte
through two identical paths decodes correctly (same `m`) although doubled;
magnitude linearity is not claimed.

## Verified results (runs `decoder-permutation-execute-t099lU4g`, `-46jXxhRO`)

Each run: **870 kernel launches, 3,482 readbacks**, 435 entries
(1 base, 23 impulse-base, 2 controls, 391 signature, 18 direct);
`kernels_launched` counts include the per-entry counter resets. Both runs are
**identical** (oracle-pair verified) and oracle-valid; the skip control
(`decoder-permutation-skip-dFlcacXr`) leaves output untouched with counters at
−1. Checks: 47,076 decode entries re-derived from raw ranges, **265,216
response-constancy checks**, **40 direct/signature value-agreement checks**,
18/18 direct predictions exact, 2/2 sign mirrors exact, impulse-only leakage
absent, coupled reference exactly the canonical 512-unit set with the
established `{0,1,2,3,8,9,10,11}` within-unit byte pattern, decoupled control
unchanged, counters `[3]×8`, guards and immutability clean, writes inside the
established extent `[0, 131072)`.

## Measured structure (findings)

1. **Every matrix byte feeds exactly 4 output bytes** (preimage size 4 for all
   511–512 weights of every probed column; 2048 outputs ÷ 512 weights). The
   decoder's weight region is therefore strongly shared/packed — consistent
   with a 2×2 upsample phase applying one weight at four output bytes.
2. **`m = 0` is invisible** (zero product): columns that include it measure
   511 weights / 2044 outputs instead of 512 / 2048. This explains the
   511-count asymmetry, not a missing weight.
3. **Input lanes pair up sharing weights.** The 16 lanes of a unit fall into 8
   weight-set groups — pairs `(0,4) (1,5) (2,6) (3,7) (8,12) (9,13) (10,14)
   (11,15)` — with **identical matrix-byte sets but disjoint output sets**.
   Some sets are further shared across units 0/4/8 and across matrix partitions
   1–3 (partition-local offsets; 28 overlapping input pairs total).
4. **The input unit selects a weight block**: unit 0/1/4/8 columns live in
   partition-local `[0 … 16344)`, unit 32's column in `[16384 … 32712)`.
5. **Within-cell packing is bit-shuffled** (example, unit 0 lane 1, cell
   `rt=0,col=0,blk=0`): output bytes `0,1,2,3,8,9,10,11` map to matrix bytes
   `+1, +65, +9, +73, +513, +577, +521, +585` — i.e. byte-index bits select
   matrix strides of 8/64/512, the signature of a tensor-core-style layout.
6. Output naming (against [NATIVE-DECODER-NAMING.md](NATIVE-DECODER-NAMING.md)
   cells): each probed column covers **256 cells** of the 16×16×512 extent,
   each cell contributing the established 8 byte offsets — row-term/column/
   block fields measured per column and retained in `cell_histogram`.

## Controls and review

* Probe self-test: full helper-chain regression, builder known answers,
  signature-pattern checks, decode round-trips, 5 decode corruption rejections,
  sign-mirror controls, plan invariants, selection-rule known answers.
* Oracle self-test: synthetic 435-record report validated (both cross paths
  exercised), **28 message-pinned corruption controls + 7 pair controls**
  rejected at their labelled checks.
* Independent review: **4 rounds**. Rounds 1–3 returned FAIL (evidence-first
  gaps, undetected multi-feeder failure mode, inconsistent sign accounting,
  untyped/unclean evidence rejection, vacuous controls, unimplemented claim);
  all findings fixed and evidence re-run against the changed source each time.
  **Round 4: PASS.** The oracle docstring discloses its verification scope
  (byte-value evidence beyond 8 samples per record is enforced at probe runtime
  and carried by retained flags).

## What this establishes — and what it does not

Established: the decoder's **(o, i) → physical matrix byte map is measurable
and validated through native Linux execution**, complete per probed input byte,
with the weight-sharing structure above. This is the packing information that
stored-weight binding needs.

Not established: the semantic (position, channel) names of the input bytes,
coverage of the other ~16,000 input bytes (extension of the measured structure
is a hypothesis until validated), exact vendor-instruction equivalence of the
decode assumptions beyond the tested domain, learned-weight binding, or any
neural-rendering quality. **No learned weights were used; no neural-rendered
image was produced.** `matrix_permutation_recovered` was **false** at this
stage; the structural extension
([NATIVE-DECODER-EXTENSION.md](NATIVE-DECODER-EXTENSION.md)) subsequently
validated it.

## Next stage

1. ~~Validate the structural extension~~ — **done**:
   [NATIVE-DECODER-EXTENSION.md](NATIVE-DECODER-EXTENSION.md) (`m = B(n) + S(o)`
   closed form, 8 held-out columns exact, `matrix_permutation_recovered: true`).
2. Bind the stored decoder-weight candidate record through this map.
3. Capture Proton reference evaluations for numerical comparison.

Containment unchanged: bubblewrap, no network, no home, pinned source
snapshots, hash-verified DLL read as data, 300 s bound, disabled core dumps.
Per-entry out0 traffic is bounded to the established write extent plus margin,
with one full-region poison upload per run and a final full-region check (a
write whose bytes coincide with the 0xA5 poison is undetectable). Run manifests
pin the probe chain; the oracle is retained beside the evidence
(`oracle-source.py`, `oracle-self-test.json`) but is not hashed by the run
manifests. Swap, driver persistence and memory erasure remain uncontrolled;
proprietary reuse/publication rights remain unresolved. No proprietary payload
files are deliberately created.

Evidence: `research/native-linux/runs/decoder-permutation-execute-t099lU4g`,
`-46jXxhRO` (each with `oracle.json`, `oracle-pair.json` on the first),
`decoder-permutation-skip-dFlcacXr`. Retained as failed/superseded:
`decoder-permutation-execute-Ufgv1gTv` (guards not initialised by the bounded
window upload — the defect that produced the one-time full poison upload),
`-W7fTKrTB`, `-Nh7nFNyb`, `-YlfiYrLm`, `-3SJGGXm6` (evidence re-run after each
review-fix round). Source: `vr/native-decoder-permutation-probe.py`,
`vr/native-decoder-permutation-oracle.py`,
`vr/run-native-decoder-permutation-probe.sh`.
