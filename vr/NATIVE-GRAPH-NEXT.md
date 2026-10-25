# Native graph reconstruction: next concrete gate

## Current evidence boundary

`NATIVE-GRAPH.md` checks substantial static descriptor construction, export routing and conditional source assignment in one pinned DLL build. It does not establish concrete runtime tensor handles/layouts or launch dataflow for the pre-decoder graph. `native-chain-probe.py` executes a constructed native projection/repack chain with controlled activations and candidate weights; it is not the proprietary runtime's graph. Separately, `NATIVE-VENDOR-COMPARE.md` replays the decoder from captured vendor boundary buffers.

The raw reference planes are byte-validated in `NATIVE-INPUT-LAYOUT.md`, but no checked operation maps them into the decoder-boundary activations. Temporal recurrence/reset remains vendor-side evidence in `TEMPORAL-HISTORY.md`. No native live-input image result exists.

## Available local evidence and its limit

Ignored captures `vr/logs/ladder/s0`, `s98` and `s99` each contain three 62,684,160-byte pool snapshots, resource maps and manifest entries for identical harness-generated 720×720 colour/depth/motion planes. In `evaluation-probe.h`, the harness generates gradients/zero motion, uploads them through mapped upload buffers, and later writes those upload-buffer bytes to the named files; the files are dumps, not inputs loaded by the capture harness. Their local trace files are `vr/logs/capture/nvapi-trace-ladder-s0.log`, `nvapi-trace-ladder-s98.log` and `nvapi-trace-ladder-s99.log`. They pin the respective cuts; slot 99 is the decoder, with an 80-byte parameter block. Cut 98 has a 24-byte later-graph launch block. `NVAPI-ARGS.md` and `NVAPI-BINDINGS.md` give the launch inventory and decoded decoder binding contract.

These captures expose pool snapshots and per-cut launch-boundary blocks (slot 0 at `s0`, slot 98 at `s98`, decoder slot 99 at `s99`), but their 24-record resource maps do **not** join the raw input files to model-arena addresses. The harness-created colour/depth/motion textures are recorded as zero-VA texture resources; the named dump files are captured by mapping separate upload buffers. The tracer logs launch parameter bytes, including GPU-address values, but never dereferences those pointers or captures tensor contents. A matching raw-plane digest across manifests therefore does not identify the input's kernel-readable resource.

## Next bounded investigation

Do not repeat captured decoder comparison, byte-search opaque pools, launch GPU probes, or add more input-layout scaffolding. First close the host-side identity gap for these harness-generated reference inputs, while keeping their relation to any production/vendor caller inputs explicitly unestablished:

1. Inspect `evaluation-probe.h` generated-plane/upload/copy and NGX-parameter code, `boundary-probe.h` resource mapping, `feature-probe.c` hooks, and the tracer's create/launch records. Determine which texture resource pointer is placed under each `DLSSNR.Color/Depth/MVec` key, how each generated plane is copied from its mapped upload buffer, and whether a stable pointer/address/identity is already logged.
2. Compare that texture identity with the first launch's parameter blocks in a harness-generated-plane trace (`cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8`, slot 1) and follow only joins directly supported by the capture. Keep D3D resource identity distinct from the CUDA device addresses in launch parameter blocks; do not infer causality from address equality alone.
3. If the current artifacts lack the join, define the minimum additional reference instrumentation: local metadata correlating named upload, D3D texture/resource, NGX parameter key and relevant kernel input address, with explicit copy/transform boundary. It must not dump tensor contents; any new capture remains reference evidence, not an evaluator dependency.
4. Only after a concrete producer→consumer edge can be stated should a fail-closed validator be implemented. Require bounds/uniqueness checks, independently reproduced input/capture pairs and synthetic corruptions; keep raw bytes and proprietary payloads local/ignored. If the missing join cannot be observed safely, record that concrete blocker rather than claim progress from another synthetic replay.

## Acceptance distinction

Even a validated edge remains intermediate evidence. End-to-end acceptance still requires raw/live input preprocessing, complete native graph execution and learned-weight bindings, native temporal history/reset recurrence, output conversion, and image comparison against repeated vendor references—all without Windows, DirectX or Proton in the evaluator.