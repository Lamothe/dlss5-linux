# Native graph reconstruction: next concrete gate

## Current evidence boundary

`NATIVE-GRAPH.md` now checks substantial static descriptor construction, export routing and a conditional source assignment for the decoder. This is bounded static analysis of one pinned DLL build. It does not establish concrete runtime tensor handles/layouts or launch dataflow for the pre-decoder graph. The existing `native-chain-probe.py` executes a constructed native projection/repack chain with controlled activations and candidate weights; it is not the proprietary runtime's graph. Separately, `NATIVE-VENDOR-COMPARE.md` replays the decoder from captured vendor boundary buffers.

The raw reference planes in `REFERENCE-CAPTURE.md` and `NATIVE-INPUT-LAYOUT.md` are documented as packed input, but no checked operation maps them into the captured main/skip decoder windows. Temporal history enters before the decoder and the vendor-side recurrence is described in `TEMPORAL-HISTORY.md`; native recurrence therefore also depends on recovering/executing this missing pre-decoder path. No native live-input image result exists.

## Next bounded investigation

Do **not** launch another captured decoder comparison. The next high-value question is whether existing contained static/native artifacts can identify and validate the earliest runtime graph segment that consumes raw input planes and produces one of the decoder-boundary activation windows.

1. Inventory current graph-builder/runtime probes and their local ignored evidence. Start from `NATIVE-GRAPH.md`, `NATIVE-DECODER.md`, `NVAPI-ARGS.md`, `DECODER-BOUNDARY.md`, and the pinned source/evidence manifests; inspect current tree and code rather than assuming archival notes are current.
2. Select one falsifiable edge (producer output → consumer input), and state what evidence can distinguish its tensor identity, shape/stride, conversion, weight binding and launch geometry. Prefer an already captured runtime descriptor/launch trace, since a new Proton capture is permitted but may only supply reference evidence and is not native inference.
3. Add a fail-closed static checker or bounded native replay only if it can validate that edge from independent evidence, with corruption controls and independently reproducible inputs and checks. Keep all tensor bytes, weights and DLL payloads local and excluded. Do not infer semantic plane formats from storage byte widths.
4. If existing evidence cannot establish the edge, record the precise missing artifact/observation and the smallest controlled capture needed; do not build further input-layout scaffolding or claim migration progress from another synthetic replay.

## Acceptance distinction

A bounded graph edge or replay would be intermediate evidence. End-to-end acceptance still requires raw/live input preprocessing, complete native graph execution and learned-weight bindings, native temporal state/reset recurrence, output conversion, and image comparison to repeated vendor references—all without Windows/DirectX/Proton in the evaluator.