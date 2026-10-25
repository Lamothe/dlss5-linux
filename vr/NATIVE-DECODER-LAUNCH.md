# Guarded native decoder launch — 11/10/2026

## Scope and result

`native-decoder-launch-probe.py` performs the **first native Linux execution** of
the pinned SM89 `cc_dec_input_upsample_1024_512_fp8` kernel. It launches through
the native CUDA driver on the RTX 4070 Ti SUPER — no Windows code, no DirectX,
no Proton, no VR. Synthetic fill is used for every input; **learned weights are
not read from the runtime and no neural-rendered image is produced**.

This is a guarded structural test — protocol completion, allocation guard/canary
integrity for the tested run, and format sanity. It does **not** establish a
maximum effective output address or general safe allocation bound, and is not
numerical decoder validation.

## What ran

One device context; the decoder module loaded from the pinned DLL as data; the
`cc_cb_clear` reset kernel (previously validated) initialises the counter region
to −1; then one decoder launch:

```text
input axes 8x8 (padded); grid (4, 2, 4) = 2*ceil(8/4) x ceil(8/4) x 4
block (32, 2, 1); 80-byte argument block via kernelParams
```

Argument block fields follow the recovered constant-bank slots: main, skip,
out0, out3, counters, out2, partials, weights, dimension, width,
output_axis0=16, output_axis1=16 (twice input, per the recovered host rule).
Buffers: 256 MiB out0, 4 MiB main/skip/partials, 8 MiB weights, 2 MiB out3/out2
(the slots the census shows unused; sized at the nominal output extent plus
margin for this tested launch only. This allocation does not prove the maximum
effective output address for other dimensions or unmodelled index/control-flow
paths), 8
counter words; 1 KiB guard zones around every region; distinct fill patterns
per role.

## Verified (mode `execute`, run `decoder-launch-execute-ualSxfj5`)

- The kernel **launched and completed** natively; `cuCtxSynchronize` returned
  normally. No hang, no CUDA fault.
- **Counter protocol reached its recovered final value**: all 8 counter slots
  read **3** — consistent with the recovered ordered-publication protocol
  (publication order itself is not directly observed by a single readback).
- **Output change extent**: first changed byte 0, last changed byte 131071,
  exactly one changed 8 MiB scan chunk — i.e. the changed bytes fall within
  `[0, 131072)`, precisely `output_axis0 × output_axis1 × 512 = 16 × 16 × 512 =
  128 KiB`. No changed chunks elsewhere within the guarded 256 MiB region and
  the out0 canary is intact. Byte-completeness of the interval is not counted;
  writes beyond the guarded window are undetectable by design. The extent is
  consistent with the recovered output-dimension rule and out0 slot wiring; it
  does not by itself discriminate the per-index addressing model.
- Inputs (main, skip, weights) byte-identical after the run; the census-unused
  slots out3/out2 untouched — corroborating the wiring census claims.
- All guard zones intact (counters, partials, every region boundary). Guard,
  canary and immutability results are **enforced**: the run fails if any is
  violated.
- No `0x7F`/`0xFF` (NaN-encoding) bytes among the written output (advisory
  content observation, not a correctness requirement).

## Negative control (mode `reduced-partition`, run `decoder-launch-reduced-partition-FgPKc6Xv`)

Grid z=2 (partitions 2/3 absent): the kernel completed safely, counters ended at
**1** — the partial-protocol value predicted by the recovered publication model
— and the run asserts exactly that plus zero output writes; the full-protocol
expectation is therefore genuinely discriminated, not merely "not 3". Guards
intact; out0 untouched — consistent with the output stores belonging to the
final partition.

## What this does and does not establish

Established: the recovered argument contract and tested launch geometry/counter
protocol execute on native Linux. For the tested synthetic input, changed bytes
were observed across `[0, 131072)` within the finite monitored out0 allocation
(byte-completeness was not counted); numerical content is undetermined. This
observation does not prove a maximum effective output address or generally safe
allocation bound (see `NATIVE-DECODER-STORE-ENVELOPE.md`).

Not established: the numerical meaning of that output — weight packing
semantics, activation semantics, the upsampling computation and temporal state
remain unrecovered. **No claim of neural-rendering quality or equivalence with
the Proton reference is made.** Index-register origins beyond the pinned
arithmetic remain conditional.

## Reproduce

Read the repository `README.md`, then:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 vr/native-decoder-launch-probe.py self-test
bash vr/run-native-decoder-launch-probe.sh reduced-partition
bash vr/run-native-decoder-launch-probe.sh execute
```

The launcher uses the existing common lock, home/network isolation, private tmp,
pinned source snapshots and a 180-second timeout; NVIDIA device nodes are
exposed only to this sandbox. Core dumps are disabled; swap, driver persistence
and memory erasure remain uncontrolled. No proprietary payload is written to
disk; the DLL is read as data. Reuse/publication rights are unresolved.

## Evidence and next work

Final-source evidence (parent read the recorded exit statuses and report values;
source snapshots retained in each run):

- `decoder-launch-execute-ualSxfj5` and `decoder-launch-reduced-partition-FgPKc6Xv`:
  exit 0; counters `[3]×8` / `[1]×8`; extent `[0, 131072)` / none; all enforced
  guard, canary and immutability checks passed; 2 kernels launched per run
  (reset + decoder), 8 readbacks.

Superseded runs:

- `decoder-launch-execute-ySuGpCHB` / `decoder-launch-reduced-partition-gcdKvI1n`
  (exit 0): identical behaviour to the final pair; superseded only by
  comment-only source corrections (allocation rationale and `allocate`
  docstring), re-run so evidence snapshots match the current source exactly.

- `decoder-launch-execute-VKpxziIp` / `decoder-launch-reduced-partition-bZ9O1i01`
  (exit 0): first passing pair; superseded by the review fixes that made guard,
  canary and immutability results enforced rather than advisory, pinned the
  reduced-partition expectation to `[1]×8` with zero writes, corrected
  `scan_changes` later-chunk first-offset handling, enlarged the out3/out2
  allocations and tightened these documentation claims.
- `decoder-launch-reduced-partition-hKgOKGvZ` (exit 1): superseded failed run;
  the shared `verify_kernel_abi` helper rejected the decoder's metadata form.
  Fixed by using the decoder's own `parameter_contract`, which asserts the
  80-byte single-parameter ABI.

Independent review: one round found seven defects (advisory verification
outcomes, a `scan_changes` first-offset bug masked by a vacuous known answer, a
vacuous-capable negative control, undersized out3/out2, three documentation
overclaims, a missing counter-ABI assertion and a vestigial `objdump`
dependency); all corrected above and re-verified.

## Decision gate: weight-binding feasibility (11/10/2026)

The gate question from the plan: can the runtime's **learned weight binding** be
made concrete within bounded effort, or should reconstruction stop and pivot to
an open neural backend?

Assessment: **continue, one more bounded stage — the gate is not yet closed.**

**Stage one outcome (12/10/2026):** the dense-matrix permutation probe now
exists and its coupling question is answered positively — see
[NATIVE-DECODER-WEIGHTS.md](NATIVE-DECODER-WEIGHTS.md). Across every tested
(matrix q, input p) cell — four coupled and twelve decoupled pairs per
word-table variant — matrix partitions couple to the output only through the
matching input partition (GPU-executed bit-identity), with a consistent
4,096-byte influence footprint. The permutation itself remains unrecovered;
fine-grained sweeps are the next stage.

What is known: the decoder reads weights from argument slot 56 — a 524,288-byte
FP8 (E4M3) matrix (1024×512) with 4-byte word tables at `+0x80000`; the address
formula is recovered and now executable (this launch). A stored decoder-weight
candidate exists in the DLL.

What is missing and concrete:

1. **Dense matrix permutation** — which stored bytes fill the slot-56 matrix, in
   what element order. The FP8 projection work solved the analogous problem for
   its kernel with one-hot channel probes; the same method applies here.
2. **Word-table semantics** — the `+0x80000` words (scales/biases?) and their
   interpretation.
3. **Reference capture** — numerical comparison needs decoder input/output
   evaluations from the working Proton VR PoC (instrument the runtime to record
   the buffers), since no isolated reference exists yet.

Each is an established-methods task, not an unknown-unknown. Pivot trigger: if
the packing permutation cannot be pinned by one-hot/impulse probes within that
stage, stop reconstruction and adopt an open neural backend using the recovered
interface contract (argument slots, geometry, counter protocol and tested I/O observations —
validated only for tested configurations; maximum effective output address and a general safe allocation bound remain unverified, see
[`NATIVE-DECODER-STORE-ENVELOPE.md`](NATIVE-DECODER-STORE-ENVELOPE.md)).

Strategic note: even a successful reconstruction reuses NVIDIA's proprietary
kernels and weights — native execution removes Windows/Proton, not the
proprietary dependency. The recovered contract is what transfers to an open
backend either way.
