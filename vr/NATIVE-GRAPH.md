# Static transformer descriptor bindings — 10/10/2026

## Scope

`native-graph-probe.py` checks a **bounded original host descriptor-construction contract** in the pinned RTX40 DLL. It does not execute Windows code, access a GPU, reconstruct the entire graph, or evaluate an image. This is progress beyond compatible kernel names and the [constructed chain](NATIVE-CHAIN.md), not a new native inference result.

The pinned bytes are read once after dump protection, hash-verified, and supplied to `objdump` using a sealed anonymous memfd. Only the explicitly listed bounded host ranges are decoded (currently 25, including the larger builder prefix). Disassembly stays in process memory; evidence reports contain stage names, addresses/counts and edges, not weights, binaries or intermediate activations. The disassembler child resets dumpability on exec and holds a readable descriptor of the **entire** input DLL. Same-UID host access to its `/proc` descriptors is **not prevented**; sandboxing and sealing do not prevent that exposure. Dump protection covers the Python parent, not this child. The memfd is not a durable extracted file; **swap, memory erasure and driver persistence remain uncontrolled**. Existing research disassembly remains separate. Do not publish proprietary implementation material without resolving rights.

## Recovered static structure

Descriptor factory `0x1800439d0` creates five layer descriptors, in this order:

| Index | Type | Constructor | Factory call |
| --- | --- | --- | --- |
| 0 | `CCVit1DFfnExpand` | `0x180071b90` | `0x180043c42` |
| 1 | `CCVit1DFfnContract` | `0x1800719b0` | `0x180043c87` |
| 2 | `CCVit1DQKV` | `0x180072110` | `0x1800441bd` |
| 3 | `CCVit1DAttention` | `0x1800714a0` | `0x180044310` |
| 4 | `CCVit1DProjection` | `0x180071ed0` | `0x1800444a4` |

The block constructor at `0x180043260` requires five descriptors, compares their type strings, constructs concrete layer objects, and advances the descriptor iterator by **0x170 bytes**. It does not assume the order from nearby string placement.

Descriptor vectors have distinct namespaces:

- **+0x90:** previous-layer indices supplying inputs.
- **+0xa8:** external block-input indices (not layer indices).
- **+0xc0:** output-port selector paired with each previous-layer index.

Descriptor copy helper `0x18001ea60` preserves these fields. The generic block resolver uses previous-layer indices to fetch output vectors from block state at **+0x140**, then uses port selectors to fetch the actual output handle. The checker verifies bounded instruction anchors for this behaviour at `0x18002b080..0x18002b0b5`, as well as the external-input path at `0x18002aff0..0x18002b00c`.

### Explicit edges checked

| Producer | Output port | Consumer | Input slot |
| --- | ---: | --- | ---: |
| FFN contract (1) | 0 | QKV (2) | 0 |
| QKV (2) | 0 | Attention (3) | 0 |
| QKV (2) | 1 | Attention (3) | 1 |
| QKV (2) | 2 | Attention (3) | 2 |
| Attention (3) | 0 | Projection (4) | 0 |
| FFN contract (1) | 0 | Projection (4) | 1 |

Projection is therefore fed by **attention plus an earlier residual**, not merely any 1,024-channel buffer. Attention consumes three separate QKV output ports. The checker extracts the short vector initialisation sequences and verifies their copy size, source, destination field and end-pointer update.

This is a fail-closed checker for inspected offsets and instruction contracts, **not a general x86 interpreter**. It does not exhaustively validate all control flow, reallocation helpers, concrete GPU bindings or runtime mutation. The hash pin ties all findings to the one inspected build.

## Additional inspected findings, not complete automated graph recovery

The main graph builder has an eight-iteration descriptor-building loop (`0x18003aa60..0x18003ab09`). It passes two boundary-dependent flags to this factory. The first selects the FFN expansion's `repack_input` descriptor spelling; the second selects projection's `repack_output` spelling for the final iteration. **Descriptor spelling does not establish an exported standalone GPU repack launch.** This still does not prove the block31..38 weight namespace/placement or the concrete consumer after projection. These loop/flag findings are now checked by the boundary contract below; they still do not prove concrete GPU bindings.

The original FFN expansion has 1024→4096 channels; contraction has 4096→1024. Projection is 1024→1024. The next useful target is the **first/next block's FFN expansion and its weight/backend contract**, or the final projection's decoder boundary—not assuming our previous spatial repacker is the next original operation.

## Reproduction

Read `README.md` first; from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 vr/native-graph-probe.py self-test
bash vr/run-native-graph-probe.sh inspect
bash vr/run-native-graph-probe.sh negative
```

The launcher follows the existing immutable source snapshot/manifest convention, serialises with the native-probe lock, disables core dumps, isolates network/home/tmp, and bounds execution to 90 seconds. **No NVIDIA device nodes are exposed.** No original renderer/probe sources or assets are changed.

Controls change six semantic anchors independently: projection producer index, attention output-port selection, projection constructor target output-handle lookup, QKV end-pointer arithmetic and projection descriptor identity. Each must be rejected. These are checker-corruption controls, **not GPU negative controls or an independent numerical oracle**. The parser also checks a literal known answer and rejects empty/duplicate instruction input.

## Evidence and verification

Final-source evidence under `research/native-linux/runs/`:

- `graph-inspect-KAkVhAU6`: six original descriptor edges verified.
- `graph-inspect-Nk9quHVX`: repeated result.
- `graph-negative-thKXpnnP`: all six semantic corruptions rejected.

Earlier `graph-inspect-4Thz9hhy` failed an incorrect checker anchor (r14 versus r15); **no Windows code or GPU kernels ran**. It remains failed evidence. `b6PoQyjO` / `q3bMFm1a` and pre-review `789q6vI6` / `0FOOnNjT` / `oBFpbujZ` were successful earlier-source checks, not substituted for current-source evidence.

CPU tests, AST/shell syntax, retained-source/current-source hashes and existing native chain regression passed. Regression `chain-execute-w9dMvQLP` has 16 launches/36 readbacks with successful teardown. Independent review identified missing QKV end-pointer and constructor/descriptor identity anchors; these were added with two extra corruption controls and the inspection was reproduced twice. Review also identified same-UID `/proc` exposure in the disassembler child; this remains uncontrolled and is now explicitly reported. Follow-up independent read-only review passed with no concrete remaining defects in the bounded scope. It verified source/report consistency but could not execute shell commands or recompute hashes; the parent separately performed those checks.

## Still missing

- Concrete following weighted consumer and actual projection/FFN layout transition.
- Proven original learned-weight namespace bindings, transformations and launch sequence.
- QKV/attention execution, dense activation rounding, complete decoder/preprocessing/temporal state.
- A native input-to-neural-output image and Windows intermediate oracle.

The successful Windows/Proton VR PoC remains unchanged. No commit or push is made by this investigation.

## Outer transformer/decoder boundary — 11/10/2026

The checker now verifies the eight-iteration builder loop and the descriptor
immediately constructed after it, plus the shared vector assignment helper.
This is **static construction evidence, not a new GPU computation**.

- `0x18003aa4f..0x18003ab09`: zero-based loop with eight iterations. With the
  inspected configuration flag zero, it calls the previously verified
  five-stage factory. A nonzero flag selects a different factory at
  `0x1800425a0`; that alternative has not been reconstructed. First/last
  iteration booleans are supplied explicitly to the primary factory.
- `0x18003ab0f..0x18003abbd`: immediately after the loop, a single-layer outer
  descriptor is built using constructor `0x180073db0`. The constructor assigns
  the type **`CCDecInputUpsample`**, input channels **1024**, output channels
  **512**. Caller argument registers and constructor field writes are checked.
- The caller assigns two outer vectors: **raw source indices `[38, 30]`** at
  outer-descriptor `+0x38`, and **raw selectors `[0, 1]`** at `+0x50`. The
  descriptor base is `rbp-0x20`, distinct from the intra-layer descriptor
  namespace. Both source ranges have an explicitly checked 16-byte length.
- `0x180032650..0x1800326be`: the shared assignment helper copies `end-begin`
  bytes from the supplied range and retains the resulting destination end
  pointer. Source, destination, memcpy and end-pointer anchors are checked.

**Do not equate source index 38 with the learned-weight name `block38`.** We
have not yet proved the global descriptor numbering, outer resolver's selector
semantics, actual tensor handles, or the correspondence to the candidate
stored-weight records previously tested. This identifies a decoder-boundary
candidate in the original construction sequence, not a validated direct
projection-to-upsampler GPU edge. Likewise, the consumer between non-final
transformer blocks is not yet verified.

Fourteen semantic corruption controls now cover the previous six intra-block
anchors plus loop count/branch, outer source index/selector destination,
upsample identity, copy source, range length and channel argument mapping.
All must fail closed. These remain checker controls, not independent numerical
oracles or a complete control-flow audit.

Final source evidence:

- `graph-inspect-F290coTp`: six intra-block edges and new boundary contract.
- `graph-negative-yy4lglJx`: fourteen semantic corruptions rejected.

- `graph-inspect-yERlmtcq`: repeated boundary and intra-block inspection.

All three runs exited successfully. Parent verification recomputed retained
snapshot and current-source SHA-256 hashes against each run's manifest; all
match. No probe/disassembler processes remained. CPU parser tests, Python AST
parsing and shell syntax passed. Independent read-only
review found no concrete defects in the bounded source/evidence scope; the
reviewer could read the snapshots/reports but could not run shell commands or
recompute hashes. The parent performs those checks separately.

**Next:** prove the outer resolver and global descriptor numbering, then
recover the upsampling operation's actual inputs, weight bindings, tensor
layouts and launch contract. No GPU launches, Windows execution, new extracted
weight/module files, commit or push were performed in this static round. The
original PoC and other uncommitted renderer work are untouched. The memory and
publication limitations above still apply.


## Outer exported-output routing — 11/10/2026

This round resolves the **namespace semantics**, not the concrete decoder GPU
inputs. The bounded checker now includes the original outer routing pass at
`0x1800380c9..0x18003852d` (including zero-register provenance) and export helper
`0x18003dfd0..0x18003e1ef`. These instructions are part of **tensor lifetime
analysis**, not a captured inference launch.

### Newly checked contract

- Outer descriptors are **0xe0 bytes**, distinct from 0x170-byte internal layer
  descriptors. Fast append move/copy paths preserve their source and selector
  vectors. Reallocation helpers are recognised by their call targets, but are
  not exhaustively interpreted by this checker.
- A `+0x38` entry selects a source outer block. The corresponding `+0x50`
  entry chooses one of that block's **exported outputs**. A missing selector
  defaults to zero; an out-of-range export selection also falls back to zero
  in this inspected pass. The pass assumes a usable source/export vector;
  it is not a general bounds-safe graph loader.
- Export helper `0x18003dfd0` constructs eight-byte pairs: **low 32 bits are
  internal layer index, high 32 bits are internal output port**. Explicit
  exports use outer fields `+0xb0` (layers) and `+0xc8` (ports). With no explicit
  exports it enumerates the final internal layer's outputs; an empty output
  metadata range implies one output. A primary five-stage factory initially
  leaves explicit export vectors empty. Subsequent mutation is not ruled out.
- The routing pass follows that pair into the host tensor-index table at
  `+0x1a8`, using 24-byte block/layer vector entries and four-byte signed tensor
  indices. For nonnegative indices it updates a 16-byte lifetime record's
  `+0xc` endpoint. **This is not a CUDA pointer table.**

Consequently, decoder source **38/export 0** is not established merely by
seeing raw indices as "internal layer 38/port 0". Likewise source
**30/export 1** is not automatically internal port 1. This fixes an important
interpretation hazard before reconstructing actual tensor bindings.

### Reproduction and evidence

Use the existing self-test/inspect/negative commands above. Eight new
corruptions cover the outer stride, selector slot lookup, export fallback,
packed port shift, tensor-index element width, explicit-export source, final
layer selection and default output count. Four additional review-driven controls cover explicit-port branch
fall-through, port-vector element width, initial zero and a later zero reset.
A follow-up exchange-register control brings the total to **27 semantic
corruptions which must be rejected**. These checks are instruction
anchor controls, not a mathematical inference oracle or dynamic trace.

Final corrected-source evidence:

- `graph-inspect-th2Rb4Oc`: new routing contract and existing contracts pass.
- `graph-inspect-9eGbNuOx`: repeated inspection passes.
- `graph-negative-l9vLTmZ7`: all 27 corruptions rejected.

Earlier `eOCOQQLq` / `8hW7yYWp` / `nSRbvVve` were successful pre-review
checks with 22 controls, not final-source evidence. Independent review found
an unchecked explicit-port jump and incomplete selector-zero provenance.
These were fixed by checking the port branch/length and the preceding zero
initialisation/reset path. A bounded register audit rejects calls, unexpected
nonzero fallback-register writes and branches that escape its temporary
pointer interval before the reset. Follow-up review caught a second-operand
`xchg` write missed by that audit; `xchg`/`xadd` now check both operands, with an
additional negative control. The earlier corrected `lErIo4RR` / `1vtzhx9z` /
`t7ioKOB5` runs had 26 controls and are superseded by the final runs above.
This remains **not an exhaustive CFG or instruction-side-effect proof**. Intermediate `RxisFJCY` / `LYHyAUPC` / `vbSE9dUr` / `ORhUytd6`
checks failed overly restrictive audit assumptions; all are retained as failed
static evidence and did not execute Windows or GPU code.

CPU parser tests, Python AST and shell syntax pass. Parent verification
recomputed every retained/current source hash for the three final manifests;
all match, and repeated inspection JSON is identical. Final independent
read-only follow-up review **passed with no concrete remaining defects in the
bounded scope**. The reviewer checked source, retained evidence and declared
hash consistency but did not run commands or recompute hashes; the parent
performed those checks. No probe/disassembler processes remain.
No GPU kernels or Windows code ran in this round. A preliminary narrow
function lookup returned `StopIteration` for a leaf without a `.pdata` entry;
that exploratory lookup was not used as passing evidence. An attempted
research delegation could not inspect the workspace under its mode limits;
all findings above were inspected and implemented by the parent.

**Still missing:** complete global block numbering; the actual export maps for
source blocks 38 and 30 in the configured runtime; concrete activation layouts,
upsampler weights/arguments and kernel launches; a native neural output image.
The working VR PoC and unrelated uncommitted renderer work remain unchanged.
Same-UID disassembler exposure, swap/erasure limitations and proprietary reuse
and publication rights remain unresolved. Nothing committed or pushed.


## Conditional decoder source exports — 11/10/2026

The checker now inspects the builder prefix and split-Swin factory, with
**conditional static routing**, not captured runtime tensor bindings:

- The outer vector is initialised empty at graph `+0x70`. The inspected
  successful append schedule is nine straight-line descriptors, then loops of
  six, eight, eight and eight appends. Capacity copy/reallocation paths are
  alternatives, not two append counts. This places the split-Swin loop at
  conditional slots **23–30**, and the primary transformer loop at **31–38**.
  This is an inspected schedule, **not an exhaustive CFG, callee-side-effect or
  reallocation-helper proof**. The checker leaves global runtime numbering
  unverified. It does not establish the learned-weight namespace.
- The third loop traverses eight signed configuration pairs. Its final
  iteration passes both final-head and explicit-export flags to factory
  `0x180040e20`. The table is checked in place as metadata, not extracted weights.
- That factory's final branch explicitly exports **(layer 4, port 0)** and
  **(layer 3, port 0)**, in that order. The assignment ranges are 16 bytes, with
  independent layer/port vectors. Constructor call/type anchors identify layer
  4 as **`CCSplitSwin16HFinalHead`** and layer 3 as **`CCSplitSwin16HProjPool`**.
- The primary transformer's projection initialises output metadata empty;
  its inspected fast move preserves the range endpoints. Together with the
  existing default-export helper, this implies **(layer 4, port 0)** on the
  inspected construction path. No dynamic output shape is established.
- Consequently, the decoder's **source 38/export 0** conditionally resolves to
  the primary transformer's projection, and **source 30/export 1** to the
  preceding split-Swin projection/pooling layer's **port 0**, not port 1.
  This is evidence for a second, earlier-model feature input (a skip-path
  candidate), not simply two copies of the final projection.

Fast outer move/copy export fields and endpoints are checked. Later descriptor
mutation, alternate transformer configuration, full helper effects and actual
GPU handles remain unresolved. No operation was launched because of this new
routing; no native neural image was produced.

### Verification

- `graph-inspect-hrbbBnzw` and `graph-inspect-nV9xzQIf`: repeatable inspection.
- `graph-negative-PiTJ4ur4`: **51** independent instruction-anchor corruptions
  rejected, retaining all previous 27 controls.
- CPU parser/table known-answer tests and three malformed configuration tables
  pass/reject as expected. These are checker controls, not an independent
  mathematical inference oracle.
- Native constructed-chain regression `chain-execute-jOFOKecw`: **16 launches,
  36 readbacks**, 4,194,304 stage-output bytes matching references. The chain
  still explicitly reports original model linkage and image evaluation false.

Initial `graph-inspect-80Ti0gQb` failed because a disassembly range truncated
an instruction; intermediate `graph-negative-T7im9fGI` failed a stale expected
control count. Final-head extension attempts before correction failed an incorrectly entered
call address (see retained run stderr/manifest); none executed Windows or GPU code. These are not passing evidence.
The previous successful 41/44-control snapshots remain earlier-source evidence,
not substitutes for the final source.

**Next:** validate runtime descriptor/tensor bindings for these conditional
edges, then recover the decoder's two activation layouts, weight selection,
argument ABI and launch sequence. The VR PoC and unrelated renderer changes
remain untouched. Same-UID disassembler exposure, swap/erasure limitations and
proprietary reuse/publication restrictions remain unresolved. Nothing committed
or pushed.

Independent review initially found an unchecked split-Swin branch predicate and
missing local append-count/identity anchors. Both were fixed: the final-head
predicate, initial empty local vector, three preceding append/capacity paths,
constructor calls and descriptor source identities now have explicit checks.
Four additional corruptions exercise the predicate, local append stride, append
join and initial local count. Final-source inspections/controls above were
rerun after those changes. An initial review-fix check failed a copy-instruction
address calculation; it is retained as failed static evidence, not a GPU result.

Follow-up independent read-only review passed with no concrete remaining defects
in the declared bounded scope. The reviewer inspected current/retained source
and reports but could not run commands or recompute hashes. Parent verification
recomputed all final inspection/control/regression snapshot/current source hashes,
confirmed identical repeated inspection JSON, and reran parser/table tests,
Python AST, shell syntax and tracked whitespace checks. Process inspection found
no surviving graph/chain probe or disassembler; NVIDIA driver query succeeded.

## Concrete decoder launch argument recovery — 11/10/2026

See [NATIVE-DECODER.md](NATIVE-DECODER.md) for the next bounded step: the concrete
upsampling class's input/output slot provenance, 80-byte host argument packing,
kernel-selection predicates and four independently inspected SM89 parameter
contracts. It does not establish activation layouts, learned-weight bindings,
backend CUDA launch geometry or a native image. The graph disassembler accepts
caller-supplied bounded ranges; its existing defaults/regression remain intact.
