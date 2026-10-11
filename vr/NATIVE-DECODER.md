# Decoder launch contract — 11/10/2026

## Result and limits

`native-decoder-probe.py` connects the inspected concrete decoder class's host
argument construction to the embedded **SM89** kernel parameter metadata. This
is a bounded, hash-pinned **static contract**, not a captured dynamic binding,
a decoder GPU execution test, or a native neural-rendered image.

The successful Windows/DX12/Proton VR PoC is untouched. The existing native
projection/bridge/repacking chain remains a constructed reference chain, not a
substitute for the original model. Nothing is committed or pushed here.

## Recovered contract

The concrete `CCDecInputUpsample` constructor checks the descriptor identity
and specialises to **1024 input / 512 output channels**. Its vtable associates
this class with the inspected launch, dimension-query and kernel-initialisation
methods. Minimum range checks require **two inputs and four outputs**; a fifth
output is optional. Error-message wording alone was not used as proof of count.

All four matching SM89 kernels independently specify **one 80-byte argument**:

- `cc_dec_input_upsample_1024_512`
- `cc_dec_input_upsample_1024_512_fp8`
- `cc_dec_input_upsample_1024_512_tilesync`
- `cc_dec_input_upsample_1024_512_tilesync_fp8`

The host's 80-byte stack argument is composed of eight 64-bit slots followed by
four signed 32-bit dimension slots. Slot provenance is checked, but calling
something a pointer does **not** establish its element type, size or layout.

| Offset | Host-side slot provenance |
| ---: | --- |
| 0 | input 0, main candidate |
| 8 | input 1, skip candidate |
| 16 | output 0 |
| 24 | output 3 |
| 32 | output 1 |
| 40 | output 2 if tile-sync selected, otherwise null |
| 48 | output 4 if supplied, otherwise null |
| 56 | concrete layer state slot 0, weight-pointer candidate |
| 64 | input spatial axis 0, padded to a multiple of four |
| 68 | input spatial axis 1, padded to a multiple of four |
| 72 | output axis 0: positive override or twice input axis 0, then padded |
| 76 | output axis 1: positive override or twice input axis 1, then padded |

Axis formulas apply to positive signed dimensions without integer overflow.
We deliberately leave width/height labels unverified until the backend and
kernel addressing corroborate them. Output-dimension overrides initialise to
zero in the inspected constructor, but later mutation is not excluded.

The launch method supplies three integers to a backend virtual method:
`2*ceil(input_axis1/4)`, `ceil(input_axis0/4)`, and `4`, with an argument pointer
and byte count 80. Their mapping is now recovered **conditionally on the inspected generic wrapper
and concrete CUDA backend**, as detailed below. The actual decoder's dynamic
backend binding is not captured, and no `cuLaunchKernel` decoder call is attempted.

Kernel initialisation selects FP8-named kernels when the layer name contains
`_fp8`; it retains both ordinary and tile-sync handles. The launch selects the
tile-sync handle only with an enabled global flag, non-null output 2 and
non-null tile-sync handle. This does not prove the datatype of every slot or
which variant the actual configured model selects.

## Candidate weights, not a proven binding

The stored record `block39.layer0.layer` has **525,312 payload bytes**. That is
arithmetically compatible with 512×1024 byte coefficients plus 512 two-byte
values. This remains a size-based hypothesis: datatype, scales/bias semantics,
channel packing, transformations and graph-index-to-record correspondence are
**not verified**. No decoder learned bytes are uploaded to a GPU by this probe.

## Reproduce

Read `README.md` first. From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 vr/native-decoder-probe.py self-test
bash vr/run-native-decoder-probe.sh inspect
bash vr/run-native-decoder-probe.sh negative
```

The launcher follows the existing source snapshot/manifest convention, locks
against other native probes, disables core dumps, isolates network/home/tmp,
and allows 90 seconds. **No NVIDIA device nodes are exposed.** The Python probe
never creates a CUDA context or executes Windows host code.

The graph disassembler now accepts explicitly supplied bounded ranges, retaining
its original default ranges and behaviour. The decoder reads additional bounded host
ranges. Its checker verifies **223 decoder and 158 backend instruction anchors**,
the decoder and generic-wrapper vtables, selected CUDA backend slots,
four kernel-name strings, and SM89 parameter metadata. The anchors were populated
from the pinned binary and manually interpreted: they are an instruction
fingerprint/checker, **not an independent mathematical decoder oracle**.

The metadata parser distinguishes NVAL/BVAL/HVAL header values from SVAL
following-byte payloads, rejects unknown forms, wrong ABI attribute forms,
truncation and duplicates. CPU tests include an independent literal metadata
record, all twelve host field offsets and **eight malformed metadata controls**.
Negative inspection replaces each of the 223 host anchors independently and
requires all corruptions to be rejected. Backend anchors are independently corrupted
and checked too. These are checker controls, not GPU
negative controls or exhaustive control-flow validation.

## Verification

Initial successful `decoder-inspect-bam13wRC` used an earlier source that labelled
the axes width/height without proving the labels; it is not final-source evidence.
Those labels were removed and the concrete vtable check added before final runs.

Current evidence:

- `decoder-inspect-usQVdxL4`: host packing and four SM89 parameter contracts.
- `decoder-inspect-4Zldl8qy`: repeated identical inspection.
- `decoder-negative-3ucK6ynh`: 223 host-anchor corruptions rejected.
- `graph-inspect-jKJ9S3gG`: existing graph recovery regression.
- `chain-execute-UmnTz9EJ`: existing native chain with candidate block38; **16
  launches / 36 readbacks**, 4,194,304 stage-output bytes matching references;
  all reported input/guard checks and teardown passed.

Parent verification recomputed each run's immutable snapshot and current-source
SHA-256 hashes against its manifest; all five match. The repeated decoder JSON
reports are identical. CPU self-tests, Python AST, shell syntax and tracked/new
file whitespace checks passed. Process inspection found no surviving probe or
disassembler; the NVIDIA driver query succeeded.

Independent read-only review found no concrete defects in the bounded contract,
metadata parser, launcher, source/evidence consistency or documented limits.
The reviewer could inspect source/reports but could not execute commands or
recompute hashes; the parent separately completed those checks. No decoder
kernel ran. Do not read static inspections as numerical inference.

## Data handling and remaining work

Pinned DLL bytes, decoded host instructions and decompressed GPU image remain
in process memory or a sealed anonymous memfd; output records contain metadata
only. No deliberate proprietary payload file writes occur in this probe.
However, objdump exec resets dumpability and has a descriptor for the entire DLL;
same-UID host `/proc` access is **not prevented**. Swap, memory erasure and driver
persistence remain uncontrolled. Older retained research disassembly remains
separate. Proprietary reuse/publication rights remain unresolved.

Next: confirm the conditional backend binding, the two activation address layouts,
output/work-buffer allocation and synchronisation, and actual learned-weight
lookup/packing. Only then attempt a guarded, synthetic-weight **numerical decoder
GPU test**, before connecting it to a recovered original graph or evaluating an
image. The kernel names and matching byte size are necessary evidence, not a
sufficient launch contract.


## Backend geometry recovery — 11/10/2026

A generic wrapper explains the earlier apparent mismatch: decoder dispatch at
wrapper vtable `+0x28` is not CUDA backend vtable `+0x28`. The wrapper table at
RVA `0xb28e0` selects `0x1800449a0`, which forwards the three dimensions to
backend vtable **`+0x140`**. In the concrete CUDA table at RVA `0xbc9e8`, that is
`0x1800784d0`. The constructor and selected init/create/bind/launch table slots
are checked against the pinned bytes. This establishes a conditional static
path, **not the dynamic backend actually supplied to the model**.

On this path:

- CUDA grid: `(2*ceil(input_axis1/4), ceil(input_axis0/4), 4)`.
- CUDA block on fresh kernel-descriptor creation: **`(32, 2, 1)`**, 64 threads.
- Dynamic shared memory: **0 bytes** (not a claim of no static shared memory).
- Stream: concrete CUDA backend field `+0x3f8`; its actual value is unknown.
- Arguments: `kernelParams=null` with CUDA's `extra` buffer-pointer,
  buffer-size and end tags; the wrapper supplies the **80-byte** decoder argument.

The dimension order follows register/stack forwarding, not names or hardware
assumptions. The decoder's kernel initialiser supplies `(32,2,1)` to wrapper
creation; the wrapper forwards these to CUDA create. CUDA create stores them in
kernel descriptor `+0x40/+0x44/+0x48`; launch uses those as block X/Y/Z.
Descriptor `+0x50` supplies dynamic shared memory, initialised here to zero.
The create path can return a cached descriptor without revalidating its block
sizes, so the block claim is explicitly scoped to fresh creation.

The CPU shape test contains four literal grid answers, including transposed
non-square inputs, plus five rejected invalid/overflow-domain inputs. These
validate our formula implementation, not the decoder's numerical computation.

### Current evidence

- `decoder-inspect-fYoOOgZE` and `decoder-inspect-Xyco9m2w`: identical reports.
- `decoder-negative-IlFlLtKh`: **223 decoder + 158 backend** corruptions rejected.
- `graph-inspect-Itl3AsUg`: existing graph regression passes.

Earlier runs `decoder-inspect-sbfzVhKa`, `decoder-inspect-LneSwDSv` and
`decoder-negative-453QwZ83` used an intermediate source with fewer provenance
anchors and an insufficiently qualified grid-order flag. They are not final
source evidence. `decoder-inspect-37oq9XaJ` failed because a disassembly range
ended before the last expected instruction; the range was corrected.

Independent review found a missing CUDA-create shared-memory provenance check.
The callee frame maps its ninth argument to `[rbp+0x8f]`; we now check its load
into `rbx` and the branch directly to descriptor population, including the final
`rbx` store to `+0x50`. All positive/negative tests were repeated after the fix.
The earlier `DdqcYy1u`, `iFsZzBi7` and `IlFlLtKh` runs omit these three anchors
and are superseded. Follow-up read-only review passed. The reviewer could inspect source/reports
but could not execute commands or independently recompute hashes. Parent
verification recomputed all snapshot/current-source hashes against manifests
for the final three decoder runs and graph regression; all match. The two
positive JSON reports are identical. Decoder/graph CPU self-tests, AST, shell
syntax and whitespace checks passed. Process inspection found no surviving
probe/disassembler, and the NVIDIA driver query succeeded.
No decoder kernel has run. The next unresolved launch requirements are activation
and output/work-buffer layouts/sizes, synchronisation and weight binding/packing.
The rights and data-handling limitations above still apply. No commit or push.

## Sizing methods and allocation ledgers — 11/10/2026

The decoder vtable's slots `+0x40` and `+0x48` select two small leaf queries:
`0x180074710` and `0x1800747c0`. Both use unsigned query indices: indices 0 and
1 share the same result; greater indices return zero. On positive,
overflow-free dimensions, define:

```text
half_axis0 = align4(floor(query_axis0 / 2))
half_axis1 = align4(floor(query_axis1 / 2))
counter_return = 2 * (half_axis0 / 4) * (half_axis1 / 4)
feature_return = 2 * 512 * half_axis0 * half_axis1
```

**These are the query caller's axes, not proven decoder launch-argument axes.**
In particular, odd dimensions use floor division before alignment. A dimension
of one can yield zero. This is not permission to launch such a shape.

The bounded consumer around `0x180038d5e` calls slot `+0x40`, computes
`align512(4 * counter_return)` and adds this to an arena-offset accumulator.
That establishes four-byte reservation units and 512-byte reservation padding
on this inspected path. It does not prove the arena's base alignment, counter
initial value, reset policy or which decoder pointer receives that reservation.

The consumer around `0x1800391b2` calls slot `+0x48` and copies its result,
unchanged on the inspected normal-return path, into the first eight bytes of a
16-byte allocation ledger record. Both caller branches are checked through the
store. The optional ledger-index growth helper saves/restores `r14`; this
provenance assumes normal return and ABI-conforming nested calls.
The formula is compatible with two-byte 512-channel spatial data, but **its
activation datatype, element addressing, final allocator and exact argument
slot mapping are not established**. The report therefore calls this quantity a
`feature_ledger_value`, rather than asserting a final GPU allocation size.
Neither query accounts for every decoder input, output and workspace slot.

### Validation and limits

The probe now checks **153 additional sizing/consumer instruction anchors**,
on top of 223 decoder and 158 backend anchors. A bounded integer interpreter
implements the actual two leaf instruction sequences with 32-/64-bit register
semantics and compares their returns with separately expressed formulas:

- Nine literal known-answer cases, including odd/non-square dimensions, zero
  returns, large unsigned indices and the 512-byte reservation boundary.
- **616 instruction-interpreter/formula comparisons** across 308 cases.
- Nine rejected domain controls, including signed intermediate/product overflow.
- Three deliberately incorrect leaf computations rejected by value comparison.

Inspection runs apply these tests to the **decoded pinned binary instructions**,
not just the source anchor table. Neither the interpreter nor these controls
execute Windows code or GPU kernels; this remains conditional static evidence.
The interpreter is specialised to the two leaves, not a general x86 emulator.
There is no numerical decoder oracle yet.

Final verification run identifiers and independent-review outcome are recorded
below. Intermediate `decoder-inspect-SftWrvss`
checks the first version only; its tests omit decoded-instruction execution
checks and are superseded.

Next: trace the ledger values into actual device allocations and pointer slots,
then recover activation addressing, counter reset/synchronisation and weight
packing. No guarded decoder GPU launch is justified until those contracts close.
Existing rights, swap, memory-erasure and same-UID disassembler exposure limits
remain unchanged. The successful VR PoC is not modified.

### Final sizing evidence

- `decoder-inspect-A3ZFJHFO` and `decoder-inspect-nE1YSYZW`: repeated inspections.
- `decoder-negative-ccJXdWsw`: **153 sizing + 223 decoder + 158 backend**
  instruction-anchor corruptions rejected.
- `graph-inspect-4VEJO4zT`: existing graph regression.

Independent review caught an omitted caller interval between the feature query
return and its ledger store. The checker now covers both caller branches and
save/restore provenance in the optional index-vector growth helper. The earlier
`7ku7KjG6`, `RgyhV7KJ`, `6cOdAW93`, `Zcv95HLY`, `CH0Ix4EX`, and `tgYr0Tri`
runs are intermediate-source evidence, superseded by the runs above.

Follow-up independent read-only review **passed** after that fix. The reviewer
checked source/reports but did not run commands or independently recompute
hashes. Parent verification independently recomputed snapshot and current-source
SHA-256 values against all four manifests; all match, and the two decoder JSON
reports are identical. Decoder/graph CPU tests, Python AST, launcher shell
syntax, and whitespace checks passed. Process inspection found no surviving
probe/disassembler; the driver query succeeded. **No decoder GPU kernel ran.**
Nothing committed or pushed.


## Allocation-ledger to arena-pointer recovery — 11/10/2026

The next inspected consumer of the temporary 16-byte feature records is the
host planner at `0x18002ddf0`, called at `0x18003934c`. It reserves each
non-zero record quantity rounded to 512, and produces a 16-byte entry containing
an **arena offset and reserved quantity**. The records' two dwords are used as
interval endpoints; their complete lifetime semantics and the planner's full
sorting/placement algorithm are not yet independently recovered. Additional
anchors check that the selected index is multiplied by 16 for the destination
slot, and that its offset starts at zero and advances past selected occupied
ranges. The selected-index list's origin/order remains unverified; this does not
prove all original-index entries are correctly populated.

The caller moves the resulting host entry vector into **state+0x178**, copies
its primary aggregate to **state+0x168**, and copies a separate metric to
**state+0x170**. The required-arena getter at `0x18003d790` adds **state+0x208**
(the counter reservation prefix) to **state+0x168**. These tables are allocation
metadata, not already allocated device pointers.

The inspected binder at `0x18003e4e0` receives an externally supplied arena base
in `rdx` and capacity in `r8`, checks a virtual `+0x60` query against that
capacity, and records the supplied values in `+0x250/+0x258`. The precise
virtual target and the external arena's actual allocation are **not verified**.
Its consumer-side pointer construction distinguishes two regions:

```text
counter pointer = supplied base + recorded counter offset
feature pointer = supplied base + counter-prefix quantity
                              + feature-entry[record index].offset
```

The counter offsets use an eight-byte stride; feature record indices use a
four-byte stride, selecting 16-byte planner entries. The latter lookup reads
**state+0x238**; the intermediate publication linking this consumer to the earlier
feature-index ledger is **not checked**. The corresponding counter-table and
group-table publications are also unverified. Consequently, end-to-end association
of original records with a specific model group/layer remains unproven.
On the inspected nonnegative-input branch, the binder appends input-feature
pointers, then counter pointers, then feature work pointers to temporary per-layer
pointer vectors. Checked definitions establish the zeroed temporary-vector fields
and descriptor base. It publishes those through
`0x18002c1e0` into the group object's **+0x140** vector. This is conditional
static provenance, **not captured launch pointers or a decoder GPU test**.

### Checks and limits

The probe adds 305 instruction anchors for the planner output, caller transfer,
required-size getter, binder arguments/base arithmetic, indices, strides and
pointer-vector publication. Corrupting each anchor individually must fail.
Anchors are a fingerprint of our interpreted path, not an independent numerical
oracle or exhaustive control-flow validation.

A separate guarded CPU arithmetic helper tests three literal pointer answers,
six invalid/overflow/out-of-capacity cases, three literal 16-byte table entries,
and missing-prefix/wrong-field/wrong-stride controls. These are **our own helper
checks**, not emulation of the Windows binder. They do not validate actual model
indices or prove that the original runtime performs equivalent per-entry safety
checks. The known pointer values are artificial host integers, not GPU addresses.

No decoder kernel or new GPU computation ran. No device nodes are exposed by the
static launcher. Activation element types/addressing, counter initialisation,
exact weight binding, query-to-launch axis correspondence and dynamic backend
binding remain unresolved. No complete native neural-rendered image exists.

Initial run `decoder-inspect-5MCVeRiI` failed because the new checker used the
wrong helper name. It is failed evidence, not a successful contract run. The
working VR PoC and other native probes remain untouched. New data handling is
unchanged: no deliberate new proprietary payload files, but memory/swap and
same-UID disassembler access remain uncontrolled; older retained disassembly
exists. Reuse/publication rights remain unresolved.


Independent review initially failed three provenance gaps: omitted input-pointer
stage/temporary-vector definitions, unverified bookkeeping-table publication,
and missing selected-output-slot/offset definitions. The input stage and bounded
slot/placement definitions were added; claims about end-to-end table publication
and original record/group/layer association were explicitly narrowed. No unverified
publication is treated as recovered. Intermediate pre-review successful reports
`ROij3ivZ`, `q8e86Kvo` and `qibZ0z3Q` are superseded by final-source runs below.


### Final-source verification

- `decoder-inspect-kMwugEkQ` and `decoder-inspect-Hm8qLRwh`: identical reports.
- `decoder-negative-KUaiFcKz`: **305 arena + 153 sizing + 158 backend + 223
  decoder = 839** independently corrupted instruction anchors rejected.
- `graph-inspect-qJ7OLTq5`: existing graph recovery regression passed.
- CPU tests retain 616 sizing interpreter/formula comparisons and all earlier
  known-answer/malformed-input controls, plus the new arena arithmetic controls.

Parent verification recomputed immutable source snapshots and current-source
hashes against all four run manifests and checked successful exit statuses.
Python AST, shell syntax and whitespace checks passed. Follow-up independent
read-only review **passed** after the fixes and narrower claims above; execution
and hash recomputation were separately done by the parent. No decoder kernels
or new GPU work ran, no surviving probe/disassembler was found, and the NVIDIA
driver query succeeded. The task modified only this document and the decoder
probe; the existing uncommitted scene/native work was not reverted or changed.

Next: connect the remaining consumer-side bookkeeping to its publications;
recover activation addressing, counter reset and actual weight binding before
attempting a guarded numerical decoder test. This stage removes an allocation
ambiguity, but does not claim native image inference.

## Shared counter-reset path and native test — 11/10/2026

The inspected initialiser at `0x180036250` requests `cc_cb_clear` through virtual
slot `+0x20`, with dimensions `(256,1,1)`, and records the returned handle at
state **+0x230**. The binder publishes state **+0x228** as the supplied arena base
when the counter prefix is nonzero, otherwise null. These connect the shared
counter-prefix allocation to a concrete reset operation, not to the decoder's
individual counter slots or complete tile-synchronisation protocol.

The range-dispatch method at `0x18003cc80` submits that reset only on the inspected
nonempty, valid range starting at zero, with nonzero prefix, handle, pointer and
wrapper. It packs **16 bytes**: the arena-prefix pointer at offset zero and the
low 32 bits of `prefix_bytes >> 2` at offset eight. The requested grid is
`(ceil(count/256),1,1)` within the nonoverflowing test domain. Reset submission
precedes the checked range-loop comparison. **The continuation through actual
group dispatch is not anchored**, and the returned descriptor's actual block
sizes are not verified (including cache reuse). These are requested dimensions,
not captured dynamic backend state.

`native-counter-probe.py` invokes only the SM89 `cc_cb_clear` through native Linux
CUDA, explicitly supplying block `(256,1,1)`. It checks opaque 16-byte cubin ABI,
pinned host anchors, wrapper vtable identity and kernel-name pointer. It does not
execute Windows host code. The numerical reset value is **−1**, confirmed by
GPU readback, not inferred from host instructions.

### Reproduce

```bash
PYTHONDONTWRITEBYTECODE=1 python3 vr/native-counter-probe.py self-test
bash vr/run-native-counter-probe.sh execute
bash vr/run-native-counter-probe.sh skip-reset
bash vr/run-native-counter-probe.sh short-reset
```

The launcher retains source snapshots/manifests using the existing native-probe
pattern, uses the shared probe lock and bounded sandbox, and exposes the NVIDIA
device nodes for this numerical test. Only source and metadata reports are
deliberately persisted; existing swap, driver-cache, memory-erasure and same-UID
objdump exposure caveats still apply. Canaries detect writes, not read-only
out-of-bounds access. Learned weights are not uploaded and no decoder kernel runs.

### Final-source evidence

- `counter-execute-k24RyzkI`, `counter-execute-CmN6XgsK`: identical reports,
  **24 launches / 28 readbacks** combined. Counts 0, 1, 255, 256, 257, 512 and 640
  each tested twice on re-poisoned/reused allocations. All **7,684 counter words**
  reset exactly to −1; all **19,740 guard/neighbouring sentinel words** unchanged.
  Zero count is a host-side no-launch control, not a zero-grid GPU dispatch.
- `counter-skip-reset-u1o7THGb`: no launch, precisely unchanged payload rejected.
- `counter-short-reset-yR00fuQ7`: one launch, precisely one unreset word rejected.
- `decoder-inspect-jy3Lbc3x`, `decoder-inspect-6v8Ee2p6`: identical static reports.
- `decoder-negative-97aXow1s`: **99 new reset + 839 existing = 938** independently
  corrupted instruction anchors rejected.
- `graph-inspect-4wLmKrBk`: existing graph regression passed.

Earlier `kfJ3vS4A`, `7yaUCWlb`, `IYpq46zS`, `nGT2R79y`, `QgyCbDMB` and
`1653rvTL` runs passed numerical/checker tests but have stronger intermediate
host-ordering/descriptor claims; they are superseded by the final-source runs.
An exploratory string lookup used an incorrect RVA and failed before any GPU
work or code change; it is not successful evidence.

Independent review found two overclaims: requested block dimensions were labelled
as verified descriptor dimensions, and reset-before-group-dispatch ordering lacked
continuation anchors. The report now explicitly marks those unverified and all
positive/negative runs were repeated. Final follow-up review and parent
hash/cleanup checks are recorded after completion below.

**Remaining blocker:** recover concrete decoder activation addressing, exact
weight binding/packing, per-slot arena associations and decoder counter-consumer
protocol. Reset is a bounded utility computation already demonstrated in earlier
research; this round connects it to the original shared-arena host path and adds
stronger boundary/reuse controls. It is **not a new weighted stage, a launched
decoder, or a native neural-rendered image**. The working VR PoC is untouched;
no commit or push occurred. Proprietary reuse/publication rights remain unresolved.

Follow-up independent read-only review **passed**, checking the narrowed contract,
CUDA numerical checks/negative controls, launcher and evidence arithmetic. The
parent separately recomputed all eight immutable snapshot/current source hashes,
confirmed successful exit statuses and identical repeated positive reports, and
ran CPU self-tests, AST, shell syntax and whitespace checks. Process inspection
found no surviving probe/disassembler, and the NVIDIA driver query responded
normally. Changes in this round are limited to the decoder probe/documentation
and the new counter probe/launcher; existing scene/native work was preserved.
