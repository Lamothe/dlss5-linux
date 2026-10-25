# Decoder multi-term and FP16-tail semantics — bounded next-stage plan

## Purpose and gate

The decoder candidate record is bound to the recovered byte map for **single-product** tests (`NATIVE-DECODER-BINDING.md`). That does not establish accumulation, FP16-tail use, or numerical equivalence to a vendor evaluation. This plan defines a bounded next experiment using synthetic data only. It is not capture authorization, not a claim that the kernel is additive, and not permission to run a GPU probe: confirm current local policy/authorization and preflight before any execution.

## Evidence to reuse

- `native-decoder-launch-probe.py` establishes the bounded launch argument/counter/output contract and negative controls.
- `native-decoder-permutation-probe.py` and `native-decoder-extension-probe.py` supply a validated `(output,input,matrix-byte)` map.
- `native-decoder-binding-probe.py` binds the stored record for isolated single-product cases. It records the 512-element FP16 tail at bytes `[524288,525312)` and requires finite values, but does not prove those values are read or define their semantics.
- The graph notes explicitly leave raw-input/pre-decoder dataflow, full decoder evaluation, output conversion and history unestablished. No synthetic decoder test can close those gaps.

Do not inspect/export learned payload values outside the existing bounded probe and retention rules. Use synthetic matrix/activation/tail buffers for the planned arithmetic test; compare against an independently implemented CPU oracle. Keep raw evidence local and ignored.

## Experiment A — multi-term accumulation

1. Pin the exact entry point, arguments, strides and counter protocol from the existing launch contract. The documented `[0,131072)` writes are observed, **not a semantic maximum**; current source explicitly leaves loop/coordinate origins and maximum index unverified. Do not treat the observed window as a safe allocation bound. A possible bounded-memory experiment is a CUDA virtual-memory arena reserved across the complete arithmetic envelope of every pinned 32-bit index expression and final store (base-pointer addition, all `IMAD.WIDE` carries, lane/slot offsets, the complete 16-byte store width, signedness, and pointer-width overflow). This requires a proof for every reachable store path; the IO interpreter's current non-emulation of pointer-carry assembly and CFG liveness means its present report alone is insufficient. Map only the expected small prefix; any mapping-granularity padding beyond it must be explicitly identified, initialised with canaries and checked, since it is not protected by the unmapped guard. Validate VMM granularity, device/context support and access for the launching device; prove no overlapping allocation lies in the reservation. VMM failure means stop, never fall back to a conventional undersized allocation. Run the actual kernel in an isolated process/context, synchronise immediately after launch, and treat any guard fault as fatal: discard outputs/counters and terminate the process without further CUDA work, since partial writes may precede the asynchronous fault. If this strategy cannot be independently proven to contain every store, derive a conservative bound from authoritative ABI/source or stop with no GPU run. Such a guard bounds memory exposure but does not establish semantic output extent. Only after the gate, select one output and a small set of mapped input/matrix positions; do not sweep the 256 MiB output region.
2. Use at least two distinct, finite, nonzero FP8 activation terms and simple synthetic FP8 weights. Run each term alone, then both together. Include same-sign, opposite-sign cancellation, and a third term with a distinct magnitude; choose values whose expected results differ between FP16 and FP32 accumulation/rounding hypotheses.
3. Preserve all unrelated input, weight, tail, scratch, output and guard regions; poison only after the output-allocation gate in step 1 has established a safe bound. Read back only the bounded expected output window and required counters. Reject unexpected writes, NaN/Inf, counter drift, or changed guards.
4. The CPU oracle must define exact term order and parenthesisation. If static evidence does not establish either, enumerate every tested order/association for the chosen small term set (including left-to-right and alternative groupings), and compute each sequential-FP16 result with explicit round-to-FP16 after each operation; separately compute FP32 accumulation followed by one final cast, plus any instruction-derived alternative justified by static evidence. Select only among these predeclared predictions and report ambiguity when hypotheses collide; no post-hoc ordering. The GPU result may support only tested arithmetic cases, not prove production graph use or all-input equivalence.

## Experiment B — tail influence

1. After A is stable, choose a minimal, independently checked set of mapped products and set the full 512-value synthetic FP16 tail to distinct finite patterns. Compare all-zero, identity, per-index impulse, and non-unit patterns; change only one tail element per influence trial.
2. Measure bounded output deltas and counters; ensure full tails and dimensions are correct, controls are paired, and zero/identity are not treated as equivalent without measurement.
3. Report exactly which outputs change for tested indices and values. No response means only “no observed influence in this tested path”; a response does not identify scale semantics or graph location without stronger controls.

## Acceptance / stop rules

- Require two bit-identical independent executions and one skip/wrong-input or equivalent negative control before accepting any selected arithmetic relation.
- Any parser/oracle fixture must reject missing fields, wrong dimensions/types, unexpected changed-output sets, extra output bytes, control leakage and inconsistent paired results.
- Stop if the kernel/argument contract cannot be pinned from tracked source, if protection/containment preflight fails, if one bounded run could affect unrelated proprietary state, or if explicit execution authorization is absent. Do not fall back to the Vulkan NGX capability probe; it enters the proprietary runtime and is not this experiment.
- Keep report language limited to synthetic kernel arithmetic. It cannot establish learned-scale meaning, vendor evaluation equivalence, preprocessor/encoder connections, temporal recurrence, live-input image quality, or migration completion.

## Status

Plan only; no GPU execution performed. As of 25/10/2026, the output extent is still not authoritatively bounded, so this experiment cannot safely execute. Once that bound/guard gate is resolved, explicit project/operator authorization and successful current safety preflight are also required. Official Feature 18 ABI/interop provenance remains a separate blocker for connecting this component to raw native inputs.
