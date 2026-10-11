# Decoder main-input and partial-sum layout — 11/10/2026

## Scope and result

`native-decoder-layout-probe.py` recovers bounded **physical byte addressing** for
initial/prefetched main-input reads and the non-final partition partial-sum paths
of the pinned SM89 `cc_dec_input_upsample_1024_512_fp8` kernel. It creates no CUDA
context and launches no decoder or other GPU computation. The working
Windows/DX12/Proton VR PoC and existing probes remain unchanged.

This is not a complete decoder contract, a captured runtime binding, or a native
neural-rendered image. Skip-input addressing, final output addressing, suffix
weight semantics and dynamic buffer association remain unverified.

## Derived layout, conditional domain

For already padded positive axes <=4096, **at most 524,288 spatial tiles**, and
the existing conditional grid:

```text
columns = axis1/4; rows = axis0/4
group = floor(CTAID.X / columns)       # assumed ordinary quotient, group 0/1
column = CTAID.X - group*columns
tile = CTAID.Y*columns + column
```

The preceding selector experiment tested our arithmetic reconstruction, not exact
vendor SASS equivalence. That distinction remains a condition here.

### Main input at argument offset 0

An initial/prefetch 16-byte main read has the candidate relative byte address:

```text
tile*16384 + partition*4096 + slab*16 + warp*512 + lane*16
```

`partition=0..3`, `slab=0,64,128,192`, `warp=0..1`, `lane=0..31`.
The initial read uses slab0; loop prefetches cover the other three slabs. For one
spatial tile, these 1,024 reads cover **16,384 bytes exactly once**. Both channel
groups reuse the same main-input tile: group is absent from this physical address.
The FP8 E4M3 datatype is a candidate supported by the shared-memory matrix input
operation, not a recovered pixel/channel semantic layout or scale convention.

Selected decoded arithmetic subsequences implement the initial and prefetch
address origins. The high-word address instruction shifts its first input by 27,
adds the second input as a high word, then extracts the high 32 bits; this matters
for distinguishing partition and warp strides. The interpreter is deliberately
small, supports only the selected positive-domain u32 forms, and assumes the
selected predicates true. Pointer carry assembly and complete predicate/CFG
liveness are not emulated.

### Partial sums at argument offset 48

A non-final-partition scratch word has candidate relative byte address:

```text
tile*16384 + group*8192 + warp*4096 + vector*512 + lane*16 + word*4
```

`group=0..1`, `warp=0..1`, `vector=0..7`, `lane=0..31`, `word=0..3`.
The observed partition-zero path writes 16-byte vectors. The observed paths for
partitions one and two use four-byte **packed FP16-pair additions**. Together,
the physical locations form 4,096 four-byte words per tile, again 16,384 bytes.
The final partition follows a separate skip/upsampling/output path outside this
bounded audit. This does **not** prove scratch lifetime, safe nullability, complete
partition scheduling, final output format or semantic channels.

The conditional extent of each of these two buffers is:

```text
rows * columns * 16384 bytes = axis0 * axis1 * 1024 bytes
```

For axes 8×20 that is 163,840 bytes. Word-address origins are sign-extended to 64
bits by the vendor instructions, so the helper rejects shapes above 524,288 tiles:
axes 4096×4096 are **rejected**, not assigned a safe 17 GB device span. The boundary
case 4096×2048 gives 8,589,934,592 bytes and is only a CPU arithmetic boundary,
**not a feasible GPU allocation or launch recommendation**. Existing host allocation queries use
caller axes whose mapping to launch axes remains unproven; these byte extents
must not silently be substituted into the original arena planner.

## Validation

Full inspected instruction slices `[0,0x14d0)` and `[0x5da0,0x6430)` are compared
with source-pinned SHA-256 fingerprints. All438 instruction positions are
individually corrupted in negative inspection. This prevents silent changes in
unlisted intermediate instructions, but fingerprints alone are **not mathematical
inference validation or exhaustive semantic/provenance proof**.

The independent checks are:

- 24,912 comparisons of selected decoded arithmetic with separately expressed
  address formulas, across seven square/non-square shapes, both groups and warps.
- Contiguous-set coverage oracles for all 1,024 main reads and all 4,096 scratch
  words in a tile, plus six literal address/extent answers.
- Three wrong-arithmetic controls, comparing values rather than relying on hashes.
- Ten invalid dimension/coordinate controls, including signed-origin overflow.

The main interpreter comparisons use slab starts; the lane/vector offsets and
pointer-assembly sequence are manually interpreted from the fully fingerprinted
slices and checked by independent physical coverage, not a general SASS emulator.
No GPU numerical result is claimed.

## Reproduce

Read the repository `README.md`, then:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 vr/native-decoder-layout-probe.py self-test
bash vr/run-native-decoder-layout-probe.sh inspect
bash vr/run-native-decoder-layout-probe.sh negative
```

The launcher snapshots/hashes its six source inputs and uses the existing common
lock, home/network isolation, private tmp and 90-second timeout. It exposes no
NVIDIA devices. DLL/cubin/disassembly are process data/sealed memfd only; no new
proprietary payload file is deliberately saved. Core dumps are disabled, but
exec-time dumpability, same-UID `/proc`, swap, memory erasure and old retained
research disassembly remain uncontrolled. Reuse/publication rights are unresolved.

## Evidence and next work

Final-source verification:

- `decoder-layout-inspect-hNNyqzb5` and `decoder-layout-inspect-ATZdPwyL`:
  byte-identical positive reports, each with 24,912 decoded arithmetic comparisons.
- `decoder-layout-negative-BYdJzeCs`: all 438 individually corrupted instruction
  positions rejected, plus three numerical wrong-arithmetic and ten invalid-domain
  controls. These are CPU/static controls, not GPU numerical inference controls.
- `decoder-address-inspect-jZQ9PBfn` and `graph-inspect-AMoBBV3j`: unchanged
  existing static regressions pass. Decoder CPU sizing/arena/ABI self-tests pass.
- Parent separately recomputed all five manifests' immutable snapshot and current
  source hashes, checked zero exit statuses, byte-identical repeats, Python AST,
  shell syntax and whitespace/Git diff checks. No surviving probe/disassembler;
  NVIDIA driver query responded normally. No GPU kernels launched this round.

Initial reports `UssyeFef`, `6GeI4v52` and `iBsJObn7` permitted too-large shapes.
During parent verification, the signed word-origin risk was identified and fixed:
shapes exceeding 524,288 tiles are now rejected. Those earlier reports are
superseded. Independent follow-up read-only review **passed** for the corrected
source/reports. The reviewer inspected code/evidence but did not execute commands
or independently disassemble vendor pointer assembly; the parent performed the
hash/test reproduction above. Only these three new layout probe/launcher/document
files were created; existing PoC/native/scene files were not changed.

Next: recover skip-input and final output/work addressing, suffix coefficients,
and dynamic arena associations before any guarded decoder launch. This round
narrows two buffer extents; it does not close all launch requirements. Nothing
committed or pushed.
