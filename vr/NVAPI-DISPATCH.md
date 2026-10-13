# NvAPI dispatch trace — 13/10/2026

## Result

**The kernel dispatch mechanism is pinned.** The DLSSNR runtime executes its
neural kernels as CUDA cubins on the D3D12 command list through
**`NvAPI_D3D12_LaunchCubinShader`** (interface ID `0x5C52BB86`), resolved via
`nvapi_QueryInterface` and executed by DXVK-NVAPI (experimental build) on the
Vulkan path. Three traced `evaluate` runs (two pre-review, one against the
final source below) show the same fixed structure:

| Event | ID | Count per run (identical in all three runs) |
|---|---|---|
| **LaunchCubinShader** | `0x5C52BB86` | **18,720 calls = 120 evaluations x 156 launches** |
| CreateCubinComputeShaderExV2 | `0x299F5FDC` | 97 (per-kernel handle creation at feature create) |
| DestroyCubinComputeShader | `0x7FB785BA` | 96 (per-kernel teardown) |
| QueryInterface resolves | various | 41 (27 distinct IDs) |

Every evaluation dispatches **exactly 156 kernel launches**. Both runs have
identical sequence structure (120 evaluations, 18,720 launches; the two
traced logs match line-for-line in layout and differ only in tick values);
tick-level burst grouping varies with scheduling and a few evaluations
interleave other NvAPI calls. All launches return to
`nvngx_dlssnr.dll+0x2406`; handle creation (97x) to `+0x1fa3`; teardown
(96x) to `+0x24dc`.

This closes the main open question from `REFERENCE-CAPTURE.md`: the dispatch
is neither the NVAPI CuModule family (probed and rejected: `CreateCuModule`
`0xAD1A677D` resolves "Not implemented", plus four unknown IDs `0xad298d3f`,
`0x33c7358c`, `0x593e8644`, `0xa782ea46` all resolve NULL) nor host-CUDA
interposition as the dispatch path. It also explains the stage-7a negative
result: `cuda-shim.c` intercepted zero libcuda calls because **the kernel
dispatch makes no host CUDA calls** — cubin execution goes through
DXVK-NVAPI's interop inside the Vulkan driver. This explains the observed
zero intercepts; it does not prove the absence of every host-CUDA call in
the process (see the interposition caveat in `REFERENCE-CAPTURE.md`).

**156 launches per evaluation is the complete per-frame inference sequence**,
the pipeline inventory the native work has been reconstructing piece by piece.

## Method: the nvapi64.dll QueryInterface forwarder

`vr/capture/nvapi-trace.c` builds to `vr/capture/nvapi64-trace.dll`, a PE
forwarder following the `nvngx.dll_nrfwd.dll` pattern:

- Exports `nvapi_QueryInterface`, forwarding to the real DXVK-NVAPI,
  located through `DLSS_NVAPI_REAL` (explicit override) first, then
  module-adjacent `nvapi64_dxvk.dll`, then the pristine copy next to the
  probe executable (the fallback the harness relies on).
- Logs every resolve (`result`/`resolve` lines) and, via per-ID trampolines,
  every call to a resolved entry point, each with caller **module + RVA**
  attribution (this is what pinned the snippet's call sites).
- Trampolines save all volatile registers and flags and tail-jump to the real
  function: arguments are never read and returns pass straight through, so
  the wrapper is behaviour-preserving for the callee.

Environment: `DLSS_NVAPI_LOG` (trace file; absent = pure passthrough),
`DLSS_NVAPI_WRAP=0` (log resolves only). If the log cannot be opened the
forwarder **fails closed** (returns NULL) so no untraced run can pass as
evidence.

### Deployment (two failed approaches retained as evidence)

1. **Prefix system32 install failed:** Proton's launcher `try_copy` force-
   redeploys its own `nvapi64.dll` into the prefix on every run, clobbering
   any replacement (observed: our install was overwritten before the probe
   started; the run then passed with DXVK-NVAPI's own log lines).
2. **App-directory install failed:** both `_nvngx.dll` and `nvngx_dlssnr.dll`
   reference the wide string `nvapi64.dll` and the core imports
   `GetSystemDirectoryA` — the load is by explicit system32 path, so the
   Wine app-dir search never applied (observed: only
   `C:\windows\system32\nvapi64.dll` loaded).

Final mechanism: when `DLSS_NVAPI_LABEL` is set, `run-feature-probe.sh`
prepends a `cmd /c copy` of the forwarder over the deployed system32
`nvapi64.dll` **inside the Wine session**, before the probe starts. Nothing
persistent is modified (Proton and the prefix stay pristine; `proton-copy`,
`runtime-copy`, `vr/prefix`, `vr/probe` are gitignored). The harness retains
the trace at `vr/logs/capture/nvapi-trace-<label>.log` **before** marker
validation so failed runs keep evidence, and fails hard when a requested
trace is missing (which caught both failed deployment attempts).

## Controls

- `evaluate` with tracing and wrap enabled passes the unchanged marker
  (`frames=60 evaluations=60,60 validated_rgb_outputs=60,60`) — **the probe's
  own output validation is the transparency control** over 18,720 wrapped
  calls. All three traced runs are count-identical (the last one generated
  against the final committed source after review fixes).
- `create` passes with tracing (feature creation path).
- Harness rejects a missing/empty trace (development failures recorded above).
- The four unknown IDs and `CreateCuModule` resolve NULL exactly matching
  DXVK-NVAPI's own "Unknown function ID"/"Not implemented" log lines
  (cross-check between two independent observers).

## Cheap complement: NGX log level (negative result)

`DLSS_NGX_LOG=1` in the harness (level via `DLSS_NGX_LOG_LEVEL`; the override
directory must exist — creating it in the harness was required). Level 4
still logs only the two `CG2RKernelManager` kernels (`PostProcess`, `Copy`);
**the 156 network kernels are not named in the model's own log at any tested
level**, so launch identity must come from the forwarder path, not NGX
logging.

## ID naming

Interface IDs are named from the public `nvapi_interface.h` table (release
535 list, `jNizM/AHK_NVIDIA_NvAPI` `NvAPI_IDs.txt`) plus a web lookup for the
newer `CreateCubinComputeShaderExV2`. The core dispatch IDs are unambiguous
(matches: `0x5C52BB86`, `0x299F5FDC`, `0x7FB785BA`, `0xAD1A677D`,
`0x3151211B`, `0x1DC7261F`, `0x6E5EA9DB`, plus the DRS/initialisation family).
Unresolved names (queried and used, but not in the table): `0x0ddac234`,
`0x329fe6e0`, `0xd8265d24`, `0xadd604d1`, `0x842b066e`, `0x2926aaad` and the
four NULL-resolving IDs. Naming was done offline; the trace records raw IDs.

## What this does and does not establish

Established: the dispatch entry point, its call volume and structure (156
fixed launches per evaluation), kernel handle lifecycle, the caller module
and call sites, why host-CUDA interposition saw nothing, and a transparent
tracing harness for future capture.

Not established: **launch arguments** (the trampolines deliberately do not
read them), per-launch kernel identity (which of the 97 created handles each
of the 156 launches uses; note 97 created vs 96 destroyed — one handle is
never destroyed, unexplained), intermediate tensors, and any native-vs-vendor
comparison. Traces contain IDs, pointers and code addresses only — no tensor
or weight contents. Single machine, single Proton copy, deterministic runs.
Binary hashes are not used as evidence: rebuilds of `nvapi64-trace.dll` are
not byte-identical (nondeterministic linker import-thunk ordering; verified
same section sizes with differences confined to import-thunk displacements),
so the committed evidence is source, counts and verdicts only.

## Next stage

1. **Launch-argument capture** (opt-in `DLSS_NVAPI_ARGS` mode): record
   `LaunchCubinShader` arguments (handle, grid/block dims, shared memory,
   parameter pointer target) per call. That yields the complete per-frame
   kernel sequence — the contract the native reconstruction needs, and the
   mapping from the 96 handles to our recovered kernel inventory.
2. With that, capture **decoder-boundary intermediates** and run the
   comparison half of the gate:
   `reference-capture-report.py --mode float --tolerance T` against
   `vr/logs/reference/` (pinning frame index and reset semantics).

Containment unchanged: bwrap + Proton boundaries; traces and logs stay local
(gitignored); committed evidence is source, counts and verdicts only.
