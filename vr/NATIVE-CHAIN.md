# Constructed native FP8 stage chain — 10/10/2026

## Result and scope

The native Linux CUDA driver on the RTX 4070 Ti SUPER executes this **constructed** chain:

```text
stored candidate projection weights + controlled activation/residual
    -> cc_vit_1d_projection_fp8
    -> our own GPU channel-layout bridge
    -> cc_vit_1d_repack_1d_to_2d_fp8
    -> cc_vit_1d_repack_2d_to_1d_fp8
    -> validation readbacks
```

All intermediate tensors stay on the GPU. The four launches use the same default stream; synchronisation/readback happens only after the final launch. This establishes compatible stage composition, including a separately authored bridge, **not the proprietary runtime's actual graph, next weighted layer, or native neural image inference**. No Windows host code, DirectX, Proton, headset or renderer executes.

The projection keeps the validated 72-byte contract from [NATIVE-FP8.md](NATIVE-FP8.md). Only **16 × 16 positions and 1,024 channels** are tested. Source fixtures remain one-hot: this is not a dense-activation rounding oracle.

## Why a bridge is necessary

Projection's quantised output and MMA-source tensors have different channel permutations. The FP8 spatial repackers use the MMA-source permutation, not projection's output permutation. Merely sharing the output pointer with a repacker would preserve scrambled channel meanings even if a spatial round trip returned identical bytes.

`native-chain-probe.py` contains our own short PTX entry `layout_bridge`. One thread copies one logical `(position, channel)` byte from `fp8_output_index` to `fp8_activation_index`. It does not contain extracted proprietary instructions or weights. An intentionally incorrect identity-copy mode executes in bounds and must fail the intended-chain comparison.

The `cc_vit_1d_projection_repack_output` string recorded in host-descriptor analysis is **not an exported SM89 kernel name**. It must not be confused with the exported FP8 spatial repackers. Both FP8 repacks have a 24-byte opaque argument, verified from SM89 ELF parameter metadata:

| Offset | Meaning in the tested contract |
| --- | --- |
| 0 | source pointer |
| 8 | destination pointer |
| 16 | height, signed 32-bit |
| 20 | width, signed 32-bit |

Each tensor is 262,144 bytes. Forward/reverse repacks each launch grid `(64, 1, 1)`, block `(256, 1, 1)`, no dynamic shared memory. The bridge launches `(1024, 1, 1)` blocks of 256 threads. Projection retains grid `(16, 1, 4)`, block `(32, 4, 1)` and required scratch/counters. Eight guarded tensor/weight allocations plus one guarded counter allocation are used.

Spatial tile address: tile `(y//4)*(W//4)+x//4`, within-tile position `(y%4)*4+x%4`, then the validated FP8 MMA fragment coordinates. All 262,144 spatial byte addresses are checked as a bijection, along with nine literal tile/fragment boundary answers. These are complete 4×4 tiles only; padding, broadcasting and arbitrary dimensions are untested.

## Reference and controls

Stored candidate payloads `block31.layer4.layer` and `block38.layer4.layer` are uploaded byte-identically; scale bytes are unaltered. Their placement in the actual graph remains unverified. Four batches cover all 1,024 one-hot input channels and all output channels.

Validation does **not** rely solely on a round trip:

1. Projection output and FP16 scratch match the independently expressed logical one-hot matrix/residual reference from the projection probe.
2. Bridge output matches the CPU output-to-MMA channel mapping.
3. Forward spatial output matches a CPU spatial-tile reference **before** treating the inverse as confirmation.
4. Inverse output matches the intended bridged tensor.
5. Source, residual, complete stored weights/scales and all allocation guards are unchanged; counters finish at 3.

Per successful run: **16 launches, 36 readbacks, 4,194,304 stage-output byte checks**, plus projection scratch, counters, source and guard checks. These byte counts are four observations of each logical projection output across stages, not four different model evaluations.

Controls omit projection, bridge or forward repack separately, or execute an incorrect identity bridge. They validate the resulting intermediate tensors and counters against their specific incorrect/skipped references, then require rejection of the intended final result. CUDA errors do not qualify as a successful negative control. CPU tests also reject omitted/wrong bridge data and check literal payload addresses.

## Reproduction

Read README.md first; from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 vr/native-chain-probe.py self-test
bash vr/run-native-chain-probe.sh execute 31
bash vr/run-native-chain-probe.sh execute 38
bash vr/run-native-chain-probe.sh skip-projection
bash vr/run-native-chain-probe.sh skip-bridge
bash vr/run-native-chain-probe.sh skip-repack
bash vr/run-native-chain-probe.sh wrong-bridge
```

The launcher follows the established sandbox/manifest pattern: pinned DLL verification, no network, private home/tmp, minimal GPU devices, immutable source snapshots and hashes, serialised GPU use and a 90-second process bound. It disables core dumps and CUDA's disk cache. **Swap, driver persistence and memory erasure are not controlled.** Writable evidence contains metadata/counts only; candidate mismatch values are suppressed. Original assets and the working VR prototype remain unchanged.

## Evidence under `research/native-linux/runs/`

| Directory | Result |
| --- | --- |
| `chain-execute-hfsGIgee` | Failed before CUDA context creation: initial source used the wrong Python argument-class name. Retained as failed evidence. |
| `chain-execute-fP9ds5mw` | Block 31: 16 launches/36 readbacks; 4,194,304 stage-output bytes checked. |
| `chain-execute-1OpPzoqS` | Block 31 repeat: same counts. |
| `chain-execute-ebweD2lN` | Block 38: same counts. |
| `chain-skip-projection-rOqQXU10` | 3 launches/9 readbacks; omitted projection rejected. |
| `chain-skip-bridge-VNrf2OYZ` | 3 launches/9 readbacks; omitted bridge rejected. |
| `chain-skip-repack-UDTQCL2c` | 3 launches/9 readbacks; omitted forward repack rejected. |
| `chain-wrong-bridge-xH6t70Pk` | 4 launches/9 readbacks; in-bounds wrong bridge rejected. |

Across three successful runs: **48 launches, 108 readbacks and 12,582,912 stage-output byte checks**. All seven successful/control manifests match retained snapshots and current source; exit status 0, stderr empty and teardown flags true.

Projection regression `projection-execute-candidate-gPITtTxe`: 4 launches/24 readbacks. Existing tensor regression `tensor-execute-Fss2Wm9U`: 24 launches/48 readbacks. CPU tests, Python AST validation, launcher shell syntax and tracked whitespace checks passed. Independent read-only review passed with no concrete defects, checking ABI, bridge arithmetic, references, controls, cleanup and claim boundaries. The parent separately recomputed snapshot/current-source hashes against all seven successful/control manifests and the two regression manifests. The driver responded normally and process inspection found no surviving probe/disassembler processes.

## Remaining graph work

No validated original-runtime producer/consumer tensor bindings, following weighted stage, decoder sequence, preprocessing, temporal state or native output image have been recovered. Descriptor strings and neighbouring record names are not sufficient proof. Our bridge is an owned implementation of a useful boundary; it is not evidence that NVIDIA launches an equivalent standalone bridge.

The next graph milestone needs actual descriptor/launch binding evidence (or a Windows intermediate oracle), identifying the original projection output consumer, its tensor layout, learned weights and launch contract. Proprietary reuse/publication rights remain unresolved; source manifests do not establish those rights. Nothing is committed or pushed by this investigation.

Follow-on static descriptor investigation: [NATIVE-GRAPH.md](NATIVE-GRAPH.md) checks original QKV → attention → projection bindings and the residual edge. This does not validate our constructed chain as the original sequence or prove projection’s following consumer.
