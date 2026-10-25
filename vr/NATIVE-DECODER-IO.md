# Decoder skip/output addressing — 11/10/2026

## Scope and result

`native-decoder-io-probe.py` recovers bounded **physical byte addressing** for the
final-partition skip reads, partial-sum reads, output stores and weight word reads
of the pinned SM89 `cc_dec_input_upsample_1024_512_fp8` kernel. Pointer chains are
verified **contiguous**: no intervening register writer between each decoded
formation stage and its consumer load. Constant-bank slot usage is fingerprinted
per path region and tied to the full kernel extent. No CUDA context is created;
no decoder or other GPU computation is launched. The working Windows/DX12/Proton
VR PoC and existing probes remain unchanged.

This is not a complete decoder contract, a captured runtime binding, semantic
channel proof, output buffer extent proof or a native neural-rendered image.

## Constant-bank wiring census

The 80-byte parameter bank (`EIATTR_PARAM_CBANK` base `0x160`, size 80) is
referenced by three path regions tiling the complete kernel (`0x6680` bytes).
Reference counts per slot are pinned exactly:

| Region | Slot references (offset: count) |
|---|---|
| initial `[0x0000,0x14d0)` | `0x0`:1 `0x4`:1 `0x28`:1 `0x118`:1 `0x160`:2 `0x164`:2 `0x180`:1 `0x198`:2 `0x1a0`:2 `0x1a4`:3 |
| final `[0x14d0,0x5da0)` | `0x118`:1 `0x168`:16 `0x170`:2 `0x190`:8 `0x198`:16 `0x1a8`:6 `0x1ac`:6 |
| partial `[0x5da0,0x6680)` | `0x118`:3 `0x180`:1 `0x190`:9 `0x194`:8 `0x1a4`:1 |

Combined with the previously recovered host slot provenance and observed
load/store behaviour, the roles are:

| Constant slot | Argument offset | Observed role |
|---|---|---|
| `0x160` | 0 | main input (initial-path activation staging) |
| `0x168` | 8 | skip input (16 slot references in the final path; four decoded as pointer formations) |
| `0x170` | 16 | output 0 (all sixteen final-partition stores) |
| `0x180` | 32 | counter wait/publication (previously recovered) |
| `0x190` | 48 | partial sums (non-final writes, final-path reads) |
| `0x198` | 56 | weight matrix reads plus 4-byte words at `+0x80000` |
| `0x1a4` | 68 | width |
| `0x1a8`/`0x1ac` | 72/76 | output axis overrides |

Slots `0x178`/`0x188` (host outputs 3/2) are **never referenced** by this
variant's decoded instructions (the census covers the full kernel extent); the
host-side two-input/four-output minimum does not imply kernel use.

## Derived addressing, conditional domain

Read and store pointers form in **two stages** where a unit term applies:

```text
output(index, unit, slot) = c[0x0][0x170] + index*4 + unit*16 + slot
    slot ∈ {0x0000, 0x0200, 0x0400, 0x0600, 0x2000, 0x2200, 0x2400, 0x2600}
    unit = SR_LANEID (pinned S2R at 0x4c30)
read(index, unit)         = slot_base + index*4 + unit*16      (16-byte reads)
    slot_base = c[0x0][0x168] (skip) or c[0x0][0x190] (partial sums)
    unit = R119 or R129 (16-byte step registers; origins unmodelled)
weight_word(index)        = c[0x0][0x198] + index*4 + 0x80000  (4-byte reads)
```

The sixteen output stores are `STG.E.NA.128 [...], R4` in four groups: bases
`R48`/`R2` cover slots `0x0..0x600`, bases `R46`/`R10` cover slots
`0x2000..0x2600`. Two decoded indices feed the groups:

```text
index_A (R127) = (R8*R23a + R9)*2048 + 4*R2
index_B (R0)   = (R8*R23a + r23a + R9)*2048 + 4*R2
```

from the pinned subsequences `0x46d0,0x46e0,0x4860` and
`0x4620,0x4660,0x46e0,0x4750,0x4970` (`R23a` is `R23` before its `0x46e0`
redefinition). The pinned `0x4620` result is consumed by interleaved `SHFL`
instructions and clobbered at `0x4660` in this straight-line reading; **other
CFG paths are unmodelled and its contribution is not asserted**. The origins of
`R8`, `R9`, `R23a` and `R2` from loop coordinates are likewise **not** asserted;
the index domain is conditional.

Stride/lane provenance windows are enforced by bounded writer audits:
`R10 = 4` on `[0x12b0,0x21b0)`, `R144 = 4` on `[0x20a0,0x31b0)`, `R44 = 4` on
`[0x4ca0,0x5550)`, `R45 = SR_LANEID` on `[0x4c30,0x5b90)`. Wide/matrix write
families without width markers (`HMMA`, `IMMA`, `LDSM`, …) **fail closed** if
they appear in any audited window.

Per index and unit, one 16-byte store lands at each of the eight slots; per
index the 256 unit/slot addresses are distinct and each 32-unit slot block
covers 512 contiguous bytes; the greatest store-start offset from the first
slot's base is exactly `0x27f0` (the corresponding stipulated byte extent,
including the final 16-byte store, is `0x2800`).
Per execution of the pinned store group that is `2 indices × 8 slots × 32 units
× 16 B = 8192 B`; **per-CTA totals depend on unmodelled warp and loop counts**.
The **maximum output index and therefore the output buffer extent remain
unverified**. The existing launch's margin and canaries describe one tested finite allocation only; they do not establish a safe output size or maximum effective address (see `NATIVE-DECODER-STORE-ENVELOPE.md`).

`IMAD.SHL.U32` is interpreted as `d = a*b + c` with power-of-two `b`, per the
published disassembler convention; `IMAD.WIDE` forms exact 64-bit addresses.
The documented formulas and checks cover only stipulated non-negative
synthetic inputs; they do not establish general operand signedness, runtime
register origins, or reachable address semantics across CFG paths. The helper
rejects test-vector indices above `2^30 - 1`; this is not a runtime/kernel index
bound. Its CPU self-test separately asserts the *conditional* unsigned-32-bit
formula envelope (`0x4000027fc`) and excess over the launch's 256 MiB output
allocation. These are arithmetic checks under stipulated inputs, not evidence
of reachable runtime addresses.

## Validation

- Nine inspected instruction slices are compared with source-pinned SHA-256
  fingerprints (index derivation, partial/skip chains, weight word sites, all
  store groups). Fingerprints reject altered instructions but are **not**
  semantic proof.
- 68 decoded-arithmetic comparisons against separately expressed formulas across
  four synthetic grid/unit extremes: both output index derivations, both output
  pointer groups, five two-stage read chains and two weight word bases. The six
  `CHAINS` entries (four 16-byte read chains with pinned consumer loads, two
  weight word chains) additionally have their formation-to-consumer spans
  checked free of intervening writers; the fifth read chain (`0x31b0 → 0x3370`)
  is validated arithmetically only, with no pinned consumer.
- 12 per-index coverage oracles (no unit/slot alias, contiguous slot blocks,
  exact `0x27f0` difference between minimum and maximum store-start offsets (the stipulated byte extent including the final 16-byte store is `0x2800`) at four indices, plus six literal address answers.
- The full-kernel constant-bank census (three regions, exact counts) and four
  register provenance windows.
- Four wrong-formula controls fail **numerically** (wrong index multiplier,
  wrong output unit stride, wrong read unit strides on two chains); two consumer
  form controls reject shifted/widened loads; eight invalid addressing domains
  are rejected; four writer-model controls exercised (unmarked wide writes and
  atomic return writes fail closed; stores write no registers).
- Negative mode: all **51** pinned instruction corruptions (35 arithmetic/form
  pins, 16 stores) rejected; source/evidence hashes match.

The interpreter is deliberately small (IMAD family, IADD3, IMAD.WIDE), assumes
selected predicates true and does not emulate pointer carry assembly or CFG
liveness. The writer model is bounded: unmarked multi-register write families
and atomics with register returns fail closed. No GPU numerical result is claimed.

## Reproduce

Read the repository `README.md`, then:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 vr/native-decoder-io-probe.py self-test
bash vr/run-native-decoder-io-probe.sh inspect
bash vr/run-native-decoder-io-probe.sh negative
```

The launcher snapshots/hashes its six source inputs and uses the existing common
lock, home/network isolation, private tmp and 90-second timeout. It exposes no
NVIDIA devices. DLL/cubin/disassembly are process data/sealed memfd only; no new
proprietary payload file is deliberately saved. Core dumps are disabled, but
exec-time dumpability, same-UID `/proc`, swap, memory erasure and old retained
research disassembly remain uncontrolled. Reuse/publication rights are unresolved.

## Evidence and next work

Final-source evidence (both exit 0; parent re-checked snapshot hashes against
the current files and read the recorded exit statuses):

- `decoder-io-inspect-uSqKqjAj` and `decoder-io-negative-zx9DiTQ5`: 68
  arithmetic comparisons, 12 coverage checks, 6 chains, 51 corruptions
  rejected, zero GPU launches.

Superseded runs, retained as the defect history this probe's contiguity checks
exposed (all exit 1 unless noted):

- `decoder-io-inspect-Qv9aMgln` / `decoder-io-negative-j9Hmk6hD` (exit 0):
  first version; skipped the second pointer stage, so the skip formula and
  read-width labelling were wrong, and its third wrong-formula control was not
  numerical. Superseded by the first review's findings.
- `decoder-io-inspect-gjPbOXaU` / `decoder-io-negative-UHNJoSMT`: the new
  contiguity check rejected an R80 rewrite at `0x1e00`, exposing the missing
  `+ unit*16` stage.
- `decoder-io-inspect-XGi5acyn` / `decoder-io-negative-4ZwBIUQj`: rejected the
  stage-2 instruction's own register rewrite at `0x1700` before the chain scan
  excluded it by design.
- `decoder-io-inspect-FdcWOx6a` / `decoder-io-negative-ipWWPyoZ` (exit 0):
  first fully passing pair; superseded by the second review's comparison-count,
  evidence and atomic writer-model corrections.

Independent review: two rounds. Round one (v1) found the skip-formula/labelling
defect and six control-strength issues; round two confirmed the two-stage model
internally consistent and found the comparison miscount, evidence misattribution
and an atomic-return fail-open, all corrected above and re-verified.

The guarded launch predicted here is done; see `NATIVE-DECODER-LAUNCH.md`. It validates launch/protocol behaviour; changed bytes were observed within `[0,131072)` at 8 MiB scan-chunk granularity, consistent with the nominal region but not a byte-complete extent measurement. It does not bound the maximum effective output address or prove a generally safe allocation size. The later audit in `NATIVE-DECODER-STORE-ENVELOPE.md` establishes that the complete store-path bound remains unproven: pointer carries, CFG liveness, runtime loop/coordinate limits, operand signedness and exact host launch ranges are unresolved. Do not use the observed changed-byte envelope or helper `MAX_INDEX` as an allocation bound, and do not launch the decoder until that proof is independently reviewed. Weight-record binding is addressed for tested single-product cases in `NATIVE-DECODER-BINDING.md`; unresolved learned-record questions include the 512-element FP16 tail's semantics, multi-term accumulation and vendor numerical reference.
