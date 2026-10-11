# Native FP8 projection and stored-weight test — 10/10/2026

## Result

The pinned runtime's **SM89 `cc_vit_1d_projection_fp8`** now executes through Linux CUDA on the RTX 4070 Ti SUPER. Synthetic channel-specific tests and tests using stored records `block31.layer4.layer` and `block38.layer4.layer` pass bit-exact comparisons. No Windows host code, DirectX, Wine, Proton, headset or image renderer runs.

**This is a bounded model-weight computation, not a complete neural-rendered image or reconstructed inference graph.** The records are candidates for this operation: numerical compatibility does not independently prove their exact position in the Windows runtime's graph.

## Why investigate FP8?

The validated FP16 projection needs 2,099,200 weight bytes (1024² FP16 values + 1024 FP16 residual scales). None of the 153 stored records has that size. Eight records, `block31..38.layer4.layer`, instead have exactly **1,050,624 payload bytes** (1,050,664 bytes including record metadata): 1024² E4M3 bytes + 1024 FP16 values. The container declares tag 1 and a flat dimension `[525312]`; **that metadata alone does not establish the payload's mixed datatype**.

Static host inspection provides additional, bounded evidence:

- Projection descriptor constructor `0x180071ed0` identifies `cc_vit_1d_projection` / `cc_vit_1d_projection_repack_output`, type `CCVit1DProjection`, and 1024 input/output channels.
- Backend preparation `0x1800738b0` selects `cc_vit_1d_projection_fp8` when its descriptor name contains `_fp8`; helper `0x180073b50` performs that name check.
- Shared weight loader `0x180073340` looks up local key `layer` through `0x180060110`, validates the weight object through `0x180061810`, and assigns its data pointer to the backend state. No weight permutation/conversion is visible **in this bounded loader**; upstream preparation has not been fully traced.
- Projection backend vtable entries at `0x1800bbb28` and `0x1800bbb30` link that shared weight loader and FP8-aware backend preparation.
- The inspected SM89 cubin uses E4M3 Tensor Core operations, FP16 residual scales at matrix byte offset 1,048,576, FP16 intermediate accumulation and E4M3 output conversion.

The FP8 matrix contains no E4M3 NaN encodings; its tail contains finite FP16 values. This and the size match are consistent with the hypothesis, not a standalone proof of graph identity. Only block 31 and 38 were executed; no claim is made about the six other candidates.

## Tested contract

One opaque **72-byte** argument, checked against the SM89 ELF metadata:

| Byte offset | Tested meaning |
| --- | --- |
| 0 | E4M3 source, MMA-input tiled layout |
| 8 | E4M3 residual/skip, output tiled layout |
| 16 | E4M3 quantised destination, output tiled layout |
| 24 | 1024² E4M3 matrix + 1024 FP16 residual scales |
| 32 | Signed 32-bit completion counters |
| 40 | **Required FP16 accumulation scratch buffer** |
| 48, 56 | Zero; untested completion-related fields |
| 64, 68 | H, W; tested 16×16 only |

Grid `(16, 1, 4)`, block `(32, 4, 1)`, no dynamic shared memory. Six active allocations: source/residual/output each 262,144 bytes, weights 1,050,624 bytes, counters 64 bytes, scratch 524,288 bytes. All receive distinct 1024-byte guards before/after. Counters start at −1 and finish at 3. Source, residual, complete weights/scales and all guards are checked unchanged.

The scratch contains the residual plus contributions from input-channel partitions **0..2**. Partition 3 merges its contribution and writes quantised output, **without storing the final sum back to scratch**. Treating scratch as the full FP16 output produces a genuine reference failure.

### Layouts

All indices below are byte indices, except the FP16 scratch index (see `activation_index` in `NATIVE-PROJECTION.md`). The FP8 source and output/skip layouts are **different**.

```text
matrix(o,i) = (i//32)*32768 + (o//16)*512 + (o%8)*64
              + ((i%16)//4)*16 + ((o%16)//8)*8
              + ((i%32)//16)*4 + i%4

source(t,c) = (t//16)*16384 + (c//32)*512 + ((t%16)//8)*4
              + ((c%32)//16)*8 + (t%8)*64 + ((c%16)//4)*16 + c%4

output(t,c) = (t//16)*16384 + (c//32)*512 + ((t%16)//8)*4
              + ((c%32)//16)*8 + (t%8)*64
              + ((c%8)//2)*16 + ((c%16)//8)*2 + c%2
```

Each permutation is exhaustively checked as a bijection, with literal boundary answers. Public fragment reference: [PTX m16n8k32](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16832). SM120 PTX is a hypothesis source; the GPU tests execute **SM89 cubin**, not that PTX.

## Mathematical reference and controls

Four 256-token batches cover all 1024 one-hot input channels, and every output channel. Source amplitudes are 1, 0.5, −1 and 2. Residuals vary by token/channel. Synthetic matrices contain asymmetric E4M3-exact patterns; scales include 1, 0.5, −1 and 2.

The independent synthetic reference is the logical one-hot product plus scaled residual, not a physical-index lookup. E4M3 conversion is an exhaustive nearest-even finite-encoding search, constrained with subnormal, signed-zero, tie and saturation known answers. It is not a dense-activation rounding reference.

For stored records, the matrix is decoded with the **synthetically validated** permutation, then the mathematical reference uses the logical matrix. Upload is asserted byte-identical to the original record; no weight conversion or reordering is performed. This checks execution against the recovered interpretation, **not independently against a Windows layer capture or the author's intended matrix orientation**. Learned-scale products round to FP16 before merging, and the final result rounds to E4M3.

- `skip-fp8` / `skip-candidate`: zero launches; destination, scratch and counters remain poisoned/initial; expected output is specifically rejected.
- `wrong-fp8` / `wrong-candidate`: an in-bounds matrix transpose is launched; output and scratch must match the transposed reference and reject the intended result. CUDA errors do not count as passing controls.

## Reproduction

Read the project README first. From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 vr/native-projection-probe.py self-test
bash vr/run-native-projection-probe.sh execute-fp8
bash vr/run-native-projection-probe.sh skip-fp8
bash vr/run-native-projection-probe.sh wrong-fp8
bash vr/run-native-projection-probe.sh execute-candidate 31
bash vr/run-native-projection-probe.sh execute-candidate 38
bash vr/run-native-projection-probe.sh skip-candidate
bash vr/run-native-projection-probe.sh wrong-candidate
```

Optional candidate block is restricted to 31..38. The existing launcher retains immutable source snapshots, source hashes, Python/kernel version, block selection, JSON output, stderr and exit status, and serialises/bounds GPU use to 90 seconds. Candidate mismatch messages suppress payload/derived output values. Reports contain counts and metadata only, not images, learned values or extracted binaries.

## Evidence

Final source evidence (independently reviewed):

| Run directory under `research/native-linux/runs/` | Launches/readbacks | Checked output |
| --- | --- | --- |
| `projection-execute-fp8-RD0IVYl1` | 8 / 48 | 2,097,152 E4M3 bytes + 2,097,152 FP16 scratch values |
| `projection-execute-candidate-tm9ogcdM` | 4 / 24 | 1,048,576 E4M3 bytes + 1,048,576 FP16 scratch values (block 31) |
| `projection-execute-candidate-eYaim6OM` | 4 / 24 | Same; repeated block 31 |
| `projection-execute-candidate-KHpyWdWx` | 4 / 24 | Same; block 38 |
| `projection-skip-fp8-WYbKhKde` | 0 / 6 | Skipped output rejected |
| `projection-wrong-fp8-YzVsLiAa` | 1 / 6 | Transposed result validated; intended result rejected |
| `projection-skip-candidate-zGjDv9zf` | 0 / 6 | Skipped output rejected |
| `projection-wrong-candidate-1kYj18n3` | 1 / 6 | Transposed result validated; intended result rejected |

Earlier successful exploratory records `projection-execute-fp8-x0BVBQWz` and `projection-execute-candidate-lowEz6QF` use earlier source; not substituted for final-source evidence. Original FP16 projection regression: `projection-execute-cGCI3aZ5` (4/20). Tensor regression: `tensor-execute-AtitkmmS` (24/48).

### Failed attempts remain failed

- `projection-execute-fp8-1ETEN3eQ`: missing required scratch pointer caused CUDA illegal address (700). Allocation/module cleanup also returned 700; context teardown/process exit occurred. This is **failed evidence**, not a passing control.
- `6pkQodmx`, `7RmVKaUo`: numerical failure from the initial skip layout hypothesis.
- `KyaNvfwG`, `0p3gFGu6`, `dBC3B3UM`: output-layout hypothesis failures.
- `7XfBRqYA`: temporary diagnostic syntax error, no kernel launch.
- `rJli6HOa`, `qiHhyw5x`, `EkNde9ET`: scratch reference incorrectly included partition 3.
- `ZQ9iGKpc`: scratch signed-zero reference discrepancy.
- `AhdjE8Fc`: CPU conversion reference exceeded the 90-second bound; no success report. Bounded caching made the exact reference practical.

Later corrected runs pass; successful cleanup flags are emitted only after teardown. Finite canaries do not prove absence of read-only out-of-bounds access.

## Verification and review

Independent read-only review passed with no concrete defects. The parent independently recomputed **every retained snapshot and current-source hash** against all eight final FP8/candidate manifests and the FP16 regression manifest; all matched. All had exit status 0, empty stderr, expected control/count flags and successful cleanup flags. CPU self-tests, Python AST validation, launcher shell syntax and tracked whitespace checks passed again. The NVIDIA driver responded normally; no matching probe/disassembler processes remained. No tracked PoC source was changed this round.

## Limits and next step

No complete inference graph, learned-weight layer capture from the Proton reference, preprocessing, temporal state or native image evaluation has been recovered. The next useful task is mapping the actual block descriptors and producer/consumer links, particularly attention → projection → following stage, then obtaining a reference intermediate where practical.

This retains NVIDIA's proprietary kernels/weights; it is not an independently owned/trained model. Reuse/publication rights are unresolved. Core dumps are disabled and no proprietary payload files are deliberately created, but **swap, driver caches and memory erasure are not controlled**. Existing research disassembly and bounded instruction excerpts in tool/chat output must be considered separately. Do not upload extracted weights, modules or intermediate outputs. The working VR PoC is unchanged; nothing committed or pushed.
