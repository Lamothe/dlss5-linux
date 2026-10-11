# Native synthetic projection — 10/10/2026

## Result and limits

`native-projection-probe.py` executes `cc_vit_1d_projection` from the pinned runtime's SM89 GPU module directly through the Linux CUDA driver on the RTX 4070 Ti SUPER. It performs weighted FP16 computation with **synthetic weights**, not learned model weights or a neural image evaluation. No Windows host code, Wine, Proton, DirectX, VR runtime or Python GPU framework runs.

The narrow tested contract is a 1024-input/1024-output-channel projection plus a scaled residual, on full 128-token blocks. Activations are tiled using the independently tested repack layout. Synthetic matrices have a different uniform coefficient in each 256-input-channel partition: `1/1024, 1/512, -1/1024, 1/2048`. This exercises all four partial computations, including negative weights, but intentionally avoids claiming a recovered dense matrix permutation. In the initial tests the input varies by token and partition, residual by token and channel, and scale cycles through `1, 0.5, -1, 2`. Zero-matrix cases isolate scaled-residual behaviour.

All dyadic fixtures are chosen so products, partial sums and merges are exactly representable in FP16. The CPU reference uses mathematical 256-element partition sums; two literal answers independently constrain it. This validates these fixtures, **not** the rounding behaviour of arbitrary dense inputs, a complete inference graph or arbitrary dimensions.

## Candidate host contract, experimentally checked

The SM89 ELF parameter metadata specifies one opaque **72-byte** argument. Its internal fields are inferred, not published ABI:

| Byte offset | Tested interpretation |
| --- | --- |
| 0 | Tiled source FP16 activation buffer |
| 8 | Tiled residual activation buffer |
| 16 | Tiled destination activation buffer |
| 24 | Packed 1024×1024 FP16 matrix, followed at byte 2097152 by 1024 FP16 residual scales |
| 32 | Signed 32-bit completion-counter buffer |
| 40–63 | Zeroed; semantics not recovered |
| 64, 68 | Height and width; tested through their token-count product |

Only shapes H×W = 8×16 and 16×16 are allowed. Grid = `(8 * tokens/128, 1, 4)`, block = `(32, 4, 1)`, no dynamic shared memory. The four grid-z partitions each handle 256 input channels. Counter count is `8 * tokens/128`, **initial value −1**, final value 3. Initialising to zero allows partition 1 to proceed before partition 0 has completed and caused genuine numerical failure. This is an important recovered synchronisation requirement.

SM120 PTX provided hypotheses, but the SM89 cubin was also inspected in memory with `nvdisasm` via a memfd; it confirms FP16 Tensor Core instructions, the tested pointer/dimension offsets, residual-scale accesses and counter synchronisation. No new cubin or disassembly file was deliberately created. That initial round did not resolve the internal dense matrix mapping; see the channel-specific follow-up below.

## Reproducible checks

Read the project README, then from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 vr/native-projection-probe.py self-test
bash vr/run-native-projection-probe.sh execute
bash vr/run-native-projection-probe.sh skip-projection
```

The launcher snapshots and hashes all five source files, verifies the pinned DLL hash before CUDA use, disables core dumps/compilation caching, hides host home/session/network, serialises with other native probes and bounds execution to 90 seconds. The actual GPU device is exposed: this is **not GPU security isolation**. Process memory is not locked or guaranteed erased; swap/driver persistence remains uncontrolled. Existing disassembly in ignored research files remains outside the source checkpoint and needs separate publication review. Proprietary reuse/publication rights remain unresolved.

### Evidence

- `research/native-linux/runs/projection-execute-74kIODC9`
- `research/native-linux/runs/projection-execute-zEgvsmBQ`
- `research/native-linux/runs/projection-skip-projection-6zqtifP5`

Each execute run records **4 launches, 20 readbacks**, and bit-exact comparison of **786432 FP16 output values** across two shapes and weighted/zero cases. Entire source, residual and synthetic weight allocations remain unchanged, including their distinct 1024-byte pre/post guards. Output/counter guards remain unchanged and every counter finishes at 3. The skipped run records zero launches, five readbacks, unchanged poison/counters, and rejection of the output. Unrelated CUDA errors do not count as a passing negative control.

Initial attempts are retained, not relabelled as successes: `projection-execute-dYabx6d1` rejected unfamiliar metadata before launch; `projection-execute-kzql42VA` and `projection-execute-uQ10f8td` exposed the incorrect zero-initialised counter; `projection-execute-rUeNH8vo` used an incorrect exploratory eight-warp block and failed with CUDA illegal address, including subsequent allocation/module cleanup errors. No success report was emitted for these failures. Context teardown/process exit occurred; later correct-contract runs succeed and `nvidia-smi` responds normally. Finite canaries cannot prove absence of read-only out-of-bounds access or all memory faults.

Independent read-only review passed with no substantive defects. The parent then recomputed every current-source and retained-snapshot hash against all three manifests, checked zero exit status/empty stderr, output counts and cleanup flags; all matched. CPU projection checks and launcher/Python syntax passed again. The existing tensor GPU regression also passed after projection testing (`research/native-linux/runs/tensor-execute-S8UcNV2o`: 24 launches, 48 readbacks). No matching probe/disassembler processes remained, and the NVIDIA driver responded normally. Nothing was committed or pushed.

## Next question

The follow-up below tests channel-specific weight packing; then trace actual learned-weight transformation and neighbouring kernels. A native weighted supporting operation is now demonstrated on the tested contract; **native neural rendering is still not demonstrated**. Keep the working Proton/VR PoC unchanged.

## Channel-specific packing follow-up — 10/10/2026

**The candidate 1024×1024 FP16 matrix packing now passes channel-specific native GPU checks.** This is still a supporting operation with synthetic weights, not learned-weight inference or an image evaluation. The original partition-uniform tests remain as regressions.

### Reconstructed mapping

For logical output channel `o` and input channel `i`, the FP16-element index is:

```text
(i // 16) * 16384 + (o // 16) * 256
+ (o % 8) * 32 + ((i % 8) // 2) * 8
+ ((o % 16) // 8) * 4 + ((i % 16) // 8) * 2 + i % 2
```

This is a bijection across all 1,048,576 matrix elements. It was derived from input-slab strides, output-tile offsets, lane-vector weight loads and the documented MMA B-fragment coordinates. The SM120 PTX was a hypothesis source; the numerical validation executes the **SM89 cubin**, not that PTX. Public instruction reference: [PTX floating-point m16n8k16 fragments](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float).

### Independent mathematical validation

Each of four 256-token batches activates a different input channel for each token, covering all 1024 input channels. Two asymmetric dyadic synthetic matrices distinguish channel permutations. Every output channel is checked: **2,097,152 FP16 outputs per execute run**. Source amplitudes include positive, negative and scaled values; residual is zero. The reference is the logical one-hot dot product `0 + amplitude * W[o,i]`, with no use of the candidate physical packing formula. Exact zeros are positive because of addition to a zero accumulator. This does not establish arbitrary dense-activation accumulation/rounding behaviour.

CPU checks constrain the physical index with 11 independently enumerated boundary answers, prove its bijection and retain the existing argument/activation-layout checks.

Two controls:

- `skip-packed`: zero launches, poisoned destination and initial counters unchanged, expected output rejected.
- `wrong-packed`: a valid in-bounds transpose of the matrix is launched. Its result must match the separately calculated transposed mathematical reference and **fail** the intended reference. This is a real computational negative control, not a CUDA error or unsafe launch.

### Reproduction and evidence

```bash
PYTHONDONTWRITEBYTECODE=1 python3 vr/native-projection-probe.py self-test
bash vr/run-native-projection-probe.sh execute-packed
bash vr/run-native-projection-probe.sh skip-packed
bash vr/run-native-projection-probe.sh wrong-packed
```

Successful execute evidence (each **8 launches, 40 readbacks**):

- `research/native-linux/runs/projection-execute-packed-osf0fSze`
- `research/native-linux/runs/projection-execute-packed-EhxsL4yH`

Controls:

- `research/native-linux/runs/projection-skip-packed-n0CUhlTt` (0 launches / 5 readbacks)
- `research/native-linux/runs/projection-wrong-packed-eEWDTRnr` (1 launch / 5 readbacks)

The two successful runs compare **4,194,304 outputs bit-for-bit** in total. Full source, residual and matrix/scale buffers remain unchanged; all allocation guards remain unchanged and counters finish at 3. Original projection regression: `projection-execute-RtfmCuty` (4 launches / 20 readbacks). Tensor regression: `tensor-execute-NsYVZaVs` (24 launches / 48 readbacks). NVIDIA responds normally afterwards; no matching probe processes remain.

Initial failures `projection-execute-packed-BN7o0TuQ`, `projection-execute-packed-2nm6Cucc` and `projection-execute-packed-iGam0sQI` remain failed evidence. More precise mismatch diagnostics showed a signed-zero reference discrepancy (`−0` expected versus `+0` produced), not a demonstrated index permutation or illegal address. Correcting the mathematical zero-accumulator reference led to the successful runs. All evidence retains source snapshots; failed runs are not success claims.

Independent read-only review passed with no substantive defects. The parent then recomputed every snapshot/current-source hash against all five projection manifests, verified zero exit statuses, empty stderr and successful cleanup flags. CPU self-tests, Python AST/shell syntax and whitespace checks passed. The data-handling/licensing limits above still apply. No proprietary payload files were deliberately created, but chat/tool output includes bounded instruction excerpts and process memory/driver persistence is not controlled. Nothing was committed or pushed.

**Follow-up:** [NATIVE-FP8.md](NATIVE-FP8.md) records validated native FP8 projection and direct use of two candidate stored-weight records, including the different source/output layouts and required scratch buffer. Exact model-layer linkage, the full inference graph and native neural image evaluation remain unresolved.
