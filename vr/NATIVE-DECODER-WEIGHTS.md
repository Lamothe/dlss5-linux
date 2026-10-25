# Decoder weight influence map — 12/10/2026

## Scope and result

`native-decoder-weights-probe.py` is **stage one of pinning the decoder's dense
matrix permutation** (the weight-binding gate in [NATIVE-DECODER-LAUNCH.md](NATIVE-DECODER-LAUNCH.md)).
It launches the pinned SM89 `cc_dec_input_upsample_1024_512_fp8` through native
Linux CUDA on the RTX 4070 Ti SUPER — no Windows code, DirectX or Proton — with
a **single 16-byte FP8 main-input impulse** and **region-coded synthetic
matrices** in the 524,288-byte slot-56 coefficient region. The probe retains
the **full unit-offset list** of every measured footprint (bounded by
`UNIT_LIST_BOUND`), which is what makes output-channel naming possible from
retained evidence; see [NATIVE-DECODER-NAMING.md](NATIVE-DECODER-NAMING.md).

Two structural predictions are enforced:

1. **Decoupling:** a matrix partition whose input partition is all zero cannot
   change the output. Result must be bit-identical to the zero-matrix base.
2. **Coupling:** each impulse's coupled matrix partition must change the output.

Both held in every tested configuration, covering **every (matrix partition q,
input partition p) cell, with every matrix partition appearing in both the
coupled and decoupled roles**. This is an influence map with
synthetic weights — **not learned-weight inference, not the recovered
permutation and not a neural-rendered image**.

## Method

For each word-table variant (`f32-one`: `0x3F800000` words; `zeros`) and impulse
partition (`p = 0..3`; one 16-byte main-input unit at byte `p*4096`, value +1.0
E4M3, all other input zero):

| Experiment | Coded matrix partition | Code byte |
|---|---|---|
| `base` | none (all zero) | `0x00` |
| `coupled` | `p` (the impulse's partition) | `0x38` (+1.0) |
| `negative` | `p` | `0xB8` (−1.0) |
| `decoupled-qN` (three per group) | `N` (each `N ≠ p`) | `0x38` (+1.0) |

Matrix partition `q` occupies bytes `[q*131072, (q+1)*131072)` of the slot-56
region, per the recovered coefficient-read formula. Every matrix partition is
coded as `coupled` once and as a `decoupled` variant under the three other
impulse partitions: 8 groups × 6 experiments = 48 decoder launches. Skip and
partial buffers are zero; geometry is the tested 8×8 → 16×16 contract; counters
reset to −1 through `cc_cb_clear` before each launch.

## Verified results (runs `decoder-weights-execute-IUvcUu7c`, `-JvrNtvNL`)

Each run: **96 kernel launches, 384 readbacks**; both runs report identical
values at every anchor. (Earlier pairs `-pTSuyNDt`/`-aaHtBUZE` and
`-52a9i4Ap`/`-azQk2Plh` are superseded: probe source changed — full unit-offset
capture, then a field-semantics clarification — and evidence is always re-run
against the current source.)

- **Decoupling held for all 24 decoupled cells** (3 per group × 8 groups):
  every `decoupled-qN` output was bit-identical to `base` (enforced; the run
  fails at the first differing byte otherwise). Combined with the coupled
  results, every `(matrix q, input p)` cell behaves as the recovered model
  predicts on the tested contract.
- **Coupling held in all 8 groups** (enforced per group): `coupled` differed
  from `base` with a **4,096-byte influence footprint** spanning `[0, 40827]`
  in **512 distinct 16-byte units** (8 changed bytes per unit) — identical for
  every impulse partition and both word-table variants. The full 512-unit
  offset list is retained in the evidence and matches across runs; its naming
  is in [NATIVE-DECODER-NAMING.md](NATIVE-DECODER-NAMING.md).
- **Sign symmetry in the matrix code**: `+1.0` produced `+8.0`, `−1.0`
  produced `−8.0` at the sampled influenced bytes. Only these two magnitudes
  were tested; magnitude linearity is not established. The observed gain of 8
  under a single 16-byte impulse unit is recorded, not interpreted.
- **Base output is all zero in the output extent** (enforced; `16×16×512 =
  131072` bytes), with zero matrix and zero skip: no bias/offset contribution
  was observed under either word-table variant, and the two variants produced
  identical coupling values. This does not establish what the `+0x80000` word
  tables mean.
- No detectable changes were observed outside `[0, 131072)` in the monitored
  out0 region for these experiments. Writes equal to the `0xA5` poison byte are
  undetectable; addresses beyond the finite monitored allocation are not
  observed. This does not establish a maximum effective output address or a
  general safe allocation bound (see `NATIVE-DECODER-STORE-ENVELOPE.md`). Counters
  finished at `[3]×8` everywhere (enforced); main, skip, weights and
  census-unused out3/out2 stayed byte-identical (enforced); all guard zones
  intact (enforced).

The responding 4,096 bytes are 1/32 of the extent. Their structure (512 units ×
8 bytes) was the raw material for the subsequent permutation work. Mapping these
positions onto output channels and finer impulse/code sweeps were the historical
next stage; their bounded results and extrapolation limits are documented in
the permutation and extension reports below.

## Controls

- **`skip` mode** (`decoder-weights-skip-VAbcjrLZ`): reset kernel only, no
  decoder launch — counters read `[−1]×8`, every buffer byte-identical to its
  upload (including out0 poison), all guards intact on all seven regions; the
  run enforces all of these and rejects a changed output.
- **Comparator oracle** (CPU and GPU-captured data): identical buffers accepted,
  one-byte perturbations rejected.
- **Failure paths retain evidence**: the decoupling-rejection, per-group
  coupling, extent and guard-violation paths serialise the accumulated
  group/experiment measurements to stdout before failing, so a failed run
  retains its diagnostics. Counter-protocol and counter-guard mismatches
  report the observed values in the failure message.
- **Self-test** (CPU): launch-probe regression plus matrix-region tiling,
  three experiment-plan known answers including full-cell coverage, 25 buffer
  non-vacuity pairs (intended pairs differ exactly in the intended partition),
  impulse placement known answers and isolation checks for all four partitions,
  eight E4M3 decode known answers, four `changed_summary` unit-capture known
  answers (full unit list, byte-pattern union, uniformity flag, retention
  bound) and two comparator oracle controls.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 vr/native-decoder-weights-probe.py self-test
bash vr/run-native-decoder-weights-probe.sh skip
bash vr/run-native-decoder-weights-probe.sh execute
```

## Retained failed / superseded runs

- `decoder-weights-skip-LJvFJ98U` (exit 1): the reset-kernel argument block was
  a `CounterArguments` temporary freed before `cuLaunchKernel` read it; counters
  stayed at zero. Fixed by binding the struct to a local. Retained as failed
  evidence.
- `decoder-weights-execute-mKbNxMTn` (no exit status): stalled in a host-side
  `bytes(ctypes-array-slice)` conversion that materialised 268 million Python
  integers per 256 MiB readback; killed when the invoking command timed out.
  Fixed with `ct.string_at`. No GPU processes remained afterwards; `nvidia-smi`
  responded normally and subsequent runs complete in well under a minute.
- `decoder-weights-skip-ldVLPbvB`, `-vUezRJha`, `-V8FyxMJu`;
  `decoder-weights-execute-ZyRBWE2D`, `-yUpKzgGH`, `-dzMmJHTs`, `-dVnVPTt6`,
  `-68GYXcVU` (all exit 0): superseded by later source corrections (host-side
  slicing fix, influence-footprint reporting, then the review fixes:
  full-coverage experiment plan, enforced per-group verdicts,
  evidence-preserving failure paths, full skip-mode guard checks, strict-JSON
  sample values and counter-failure routing). Re-run so evidence snapshots
  match the current source exactly.
- `decoder-weights-execute-hZkGgn8k` (no exit status): interrupted when the
  invoking command's window closed mid-run (the probe's own 300 s bound was not
  reached). No processes remained; the final pair was re-run afterwards.
- `decoder-weights-execute-pTSuyNDt`, `-aaHtBUZE`, `-52a9i4Ap`, `-azQk2Plh`,
  `decoder-weights-skip-fWDJhitn`, `-6WYUMpqb` (all exit 0): earlier evidence
  pairs and skip controls, superseded as the probe source moved — full
  unit-offset capture (`unit_offsets`, `unit_byte_offsets`,
  `uniform_unit_pattern`, `unit_offsets_truncated`), then a field-semantics
  clarification in `changed_summary`. Their anchors match the current runs,
  but the pre-capture pair cannot support channel naming because it retained
  only eight sample offsets per footprint. Current evidence:
  `decoder-weights-execute-IUvcUu7c`/`-JvrNtvNL`, `decoder-weights-skip-VAbcjrLZ`.

## What this establishes — and what it does not

Established: on the tested contract, the matrix region couples to the output
only through the matching input partition — validated by GPU-executed
bit-identity across all 24 decoupled cells — and the response is symmetric in
the matrix code's sign. A bounded, repeatable influence footprint is measurable
through native Linux execution. The gate's coupling question is answered
positively.

Not established by this influence-sweep stage: the word-table semantics, the
numeric upsample computation, magnitude linearity, or any neural-rendering
quality. Subsequent stages establish a bounded structural permutation map
(validated on 21 of 4×4096 input bytes and extrapolated by additivity; see
[NATIVE-DECODER-EXTENSION.md](NATIVE-DECODER-EXTENSION.md)) and stored-record
binding for tested single-product cases (see
[NATIVE-DECODER-BINDING.md](NATIVE-DECODER-BINDING.md)); neither establishes full
numerical decoding. Learned-weight binding
is established separately for the tested single-product cases in
[NATIVE-DECODER-BINDING.md](NATIVE-DECODER-BINDING.md), but multi-term
accumulation, FP16-tail semantics and runtime graph position remain unresolved.
The bounded positive-domain group-selector reconstruction is documented in
[NATIVE-DECODER-SELECTOR.md](NATIVE-DECODER-SELECTOR.md); exact vendor-SASS
equivalence is not established. **No learned weights were used in this
influence-sweep stage; no neural-rendered image was produced.**

## Next stage

1. ~~Fine-grained influence sweeps: code matrix sub-partitions (slab/warp/lane
   groups from the coefficient-read formula) and sweep impulse positions to
   resolve `(o, i)` per physical byte.~~ **Superseded:** a structural permutation
   map was validated on 21 of 4×4096 input bytes and extrapolated over the
   remainder by additivity; see
   [NATIVE-DECODER-EXTENSION.md](NATIVE-DECODER-EXTENSION.md) for scope and
   limitations. The stored-record single-product binding is documented in
   [NATIVE-DECODER-BINDING.md](NATIVE-DECODER-BINDING.md). Multi-term accumulation
   and FP16-tail behavior remain gated by
   [NATIVE-DECODER-ACCUMULATION-PLAN.md](NATIVE-DECODER-ACCUMULATION-PLAN.md).
2. ~~Decode the output addressing (`index*4 + unit*16 + slot`) far enough to
   name the influenced output channels.~~ **Done:** see
   [NATIVE-DECODER-NAMING.md](NATIVE-DECODER-NAMING.md) — the influenced cells
   name as rows `y ∈ {0, 4}` × all `x` × channel blocks `{0-7, 16-23}` under
   the contiguous-channel hypothesis, locked by
   `native-decoder-influence-names.py`.
3. ~~Map the stored decoder-weight candidate record through the recovered
   permutation.~~ **Done for the tested single-product cases:** see
   [NATIVE-DECODER-BINDING.md](NATIVE-DECODER-BINDING.md). Proton reference
   evaluations remain unperformed and require a separately authorised,
   safety-reviewed capture; they are not a substitute for completing the
   native arithmetic and input-binding stages.

The launcher retains the established containment (bubblewrap, no network, no
home, pinned source snapshots, hash-verified DLL read as data, 300 s bound).
Core dumps are disabled; swap, driver persistence and memory erasure remain
uncontrolled. Proprietary reuse/publication rights remain unresolved. No
proprietary payload files are deliberately created.

Evidence: `research/native-linux/runs/decoder-weights-*`. Two independent
review rounds drove the corrections listed above — the first found 3 must-fix
and 4 advisory defects, the follow-up passed with 4 advisory notes, all
addressed — and the evidence was re-run against the final source. The
unit-capture extension and naming stage are reviewed in
[NATIVE-DECODER-NAMING.md](NATIVE-DECODER-NAMING.md).
