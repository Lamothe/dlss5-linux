# NvAPI launch-argument capture — 13/10/2026

## Result

**The complete per-frame kernel sequence is recovered, named, and the decoder
launch contract recovered natively in stage 5 is confirmed by the vendor's own
call.** Captured through the `DLSS_NVAPI_ARGS` mode of
`vr/capture/nvapi-trace.c`, running on the RTX 4070 Ti SUPER under
Proton/Wine — no changes to the model or to Proton.

Every evaluation dispatches the same **156 launches**, and each one is now
fully specified:

| Field | Source |
|---|---|
| kernel name | `NVAPI_D3D12_CREATE_CUBIN_SHADER_PARAMS::pShaderName` (create call) |
| block dims + dynamic shared memory | same struct (`blockX/Y/Z`, `dynSharedMemBytes`) |
| grid dims | `NvAPI_D3D12_LaunchCubinShader` arguments |
| parameter block size + pointer | last two launch arguments |
| shader handle | `NVAPI_D3D12_CREATE_CUBIN_SHADER_PARAMS::hShader` (create call, read post-call) |

### Stage-5 confirmation (the headline)

Slot 99 of every evaluation is

```
cc_dec_input_upsample_1024_512_tilesync_fp8  grid=8,3,4  block=32,2,1  psize=80
```

**`block = (32, 2, 1)` and an 80-byte parameter block** — exactly the launch
contract recovered independently from disassembly in `NATIVE-DECODER-LAUNCH.md`
and exercised natively in `NATIVE-DECODER-BINDING.md`. The vendor's live call
agrees with the native reconstruction on both values. The grid (8,3,4) is new
information and replaces the earlier "derived from padded spatial dimensions"
inference with a measurement.

### The sequence

120 evaluations were captured; **all 120 have identical handle, name, grid,
parameter-size and block-dimension sequences** (slot-by-slot), and two runs
under different capture levels agree exactly. 43 distinct kernels are used of
the 96 created. The pipeline is visible in the names:

| Slots | Stage | Kernels |
|---|---|---|
| 0 | counter/buffer clear | `cc_cb_clear` (block 256,1,1) |
| 1 | pre-block | `cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8` |
| 2–23 | encoder, 1h → 2h → 4h → 8h | `cc_tinlayout_fused_swin_*` (block 32,h,1 for h = 1,2,4,8) |
| 24–56 | transformer stage | `cc_split_swin_16h_{ffwd,ffwd_proj,qkv,proj}_512_*`, ending `proj_pool`, `final_head` |
| 57–98 | ViT stage | `cc_vit_1d_{repack_2d_to_1d,ffn_expand,ffn_contract,qkv,attention,projection,repack_1d_to_2d}` |
| 99 | **decoder upsample** | `cc_dec_input_upsample_1024_512_tilesync_fp8` |
| 100–131 | transformer stage | `cc_split_swin_16h_{ffwd,ffwd_proj,qkv,proj}_512_*` (8 quadruplets) |
| 132–153 | decoder, 8h → 4h → 2h → 1h | `cc_tinlayout_fused_swin_*_{upsample,outview}_*` |
| 154 | final blend | `cc_tinlayout_fused_post_block_swin_1h_32_fp8` (psize 184) |
| 155 | output copy | `cg2r_copy_kernel` (block 16,16,1) |

Both transformer stages repeat a `qkv → proj → ffwd → ffwd_proj` quadruplet
eight times (**16 `cc_split_swin_16h_qkv` launches per evaluation**), and the
ViT stage runs **8 `cc_vit_1d` attention blocks** — consistent with the block
structure recovered statically in `NATIVE-GRAPH.md`. The `cg2r_post_process`
and `cg2r_copy` kernels are the two the model's own NGX log names
(`CG2RKernelManager`), which independently corroborates the naming.

The full 156-slot table is in `vr/logs/capture/args-report-1.json`.

## Method

### The launch ABI is seven arguments, not twelve

The first decode attempt assumed the twelve-argument form from a search summary
of `nvapi.h` (`pCommandList, hShader, pShaderName, gridDimX…, pParams,
paramSize`). It produced garbage: empty names and implausible dimensions.
Two independent sources settled it:

1. **DXVK-NVAPI's implementation** (`src/nvapi_d3d12.cpp`):
   `NvAPI_D3D12_LaunchCubinShader(ID3D12GraphicsCommandList*, NVDX_ObjectHandle,
   NvU32 blockX, blockY, blockZ, const void* params, NvU32 paramSize)` —
   seven arguments.
2. **The caller's own marshalling**, disassembled at `nvngx_dlssnr.dll+0x2404`
   (`objdump` on `model/nvngx_dlssnr.dll`): `mov %r12,%rcx; mov %rbp,%rdx;
   mov %esi,%r8d; mov %edi,%r9d` then three stack stores at
   `[rsp+0x20], [rsp+0x28], [rsp+0x30]` before `call *%rbx` at `+0x2404`
   (returning to `+0x2406`, the site recorded in the trace).

There is **no shader-name argument at launch**. Block dimensions are fixed at
creation, which is why the create struct carries `blockX/Y/Z`.

The trampoline's saved-register block is indexed against this layout: `[2]`=r9,
`[3]`=r8, `[4]`=rdx, `[5]`=rcx, `[14..16]`=stack args 5..7. The full 22-slot
array is logged with every capture (`s=`) so the mapping stays checkable
offline; the four shadow-space slots (`[10..13]`) are not part of the ABI and
are deliberately left uninterpreted.

### Creation needs a post-call read

`NVAPI_D3D12_CREATE_CUBIN_SHADER_PARAMS` (0x50 bytes; `structSizeIn` observed
as 80) is passed as a single pointer in rcx. `pShaderName` is an input, but
**`hShader` is an output written by the callee** — reading the struct before the
call yields names but never handles. The first attempt deferred the read to the
next traced call and failed: the model reuses one struct for every creation, so
by the next call the struct already held the following kernel's name with
`hShader` still zero (`*-superseded-deferred-read.log`, 96 names captured,
handles all zero, names shifted by one).

The working mechanism is a **call-style stub** for this interface only: it
drops the saved return addresses (the original is captured into `g_saved_ret`),
restores the stack to the caller's pre-call value, **calls** the real entry
point (its arguments line up again because the frame is exactly what the
original caller set up), reads the struct in `tramp_post_logger`, and then
returns to the original caller with the real return value. The probe's own
output validation (`PASS: two distinct feature handles created…`) over 97
creations is the transparency control: a stub that corrupted the call would
fail the marker.

All other interfaces keep the tail-jumping stub from `NVAPI-DISPATCH.md`, which
still reads no arguments unless `DLSS_NVAPI_ARGS` is set.

### Captured fields and bounds

`DLSS_NVAPI_ARGS=1` records the launch ABI and the creation struct (name, block
dims, shared memory, flags, cubin pointer and size, handle). `=2` additionally
hex-dumps the first 512 bytes of each launch's parameter block (`pcap=3` records that no dump was
requested at level 1). Reads are VirtualQuery-guarded and stop at the first
unreadable byte; names are capped at 128 bytes and rendered whitespace-free
(`\xNN` escapes). The cubin pointer is recorded but **never dereferenced**;
parameter-block contents stay as hex in the local trace only — the report
emits counts, names and dimensions, never parameter bytes. Invalid
`DLSS_NVAPI_ARGS` values, or ARGS without a trace or with `DLSS_NVAPI_WRAP=0`,
fail closed (QueryInterface returns NULL); `DLSS_NVAPI_ARGS=0` is accepted and
reads nothing, as documented.

## Evidence

Under `vr/logs/capture/` (gitignored: traces and report JSON stay local.
Committed artefacts are source and this document; the verdicts below are
reproducible from them):

| Run | Label | Level | Purpose |
|---|---|---|---|
| create | `create-args-1` | 1 | creation struct, names + handles |
| evaluate ×2 | `eval-args-1`, `eval-args-2` | 2 | main inventory + parameter dumps |
| evaluate | `eval-args-l1` | 1 | cross-check of level 1 vs level 2 |

`vr/nvapi-args-report.py` validates the evidence (`--trace`, `--a/--b`,
`--self-test`). Verdicts on this machine:

- `eval-args-1`: **inventory-consistent** — all 17 checks pass (97 create
  calls, 96 kernels + 1 null-parameter create, 18,720 launches = 120 × 156,
  call-record counts launch=18,720 / create=97 / destroy=96 matching the
  parsed records, every captured call on one thread, every launch handle
  mapped to a created kernel, per-slot name/grid/psize/block stable, decoder
  block `(32,2,1)` and `psize=80` as pinned).
- `eval-args-1` vs `eval-args-2`: **identical-inventory**. The verdict
  compares name, grid, block dims, parameter size and shared memory per slot;
  handle values are compared separately and were identical in both runs
  (`args-pair.json`).
- `eval-args-1` vs `eval-args-l1`: **identical-inventory** (the level-2 dump
  does not perturb the capture; `args-pair-l1.json`).
- `--self-test`: **20/20 corruption controls** behave as expected — dropped
  and duplicated launches, a tail-truncated trace, sequence gaps, unknown
  record kinds, consistently forged parameter sizes, changed grid and handle,
  altered decoder block dims, a null created handle, a renamed kernel, a
  launch handle no create owns, malformed records, unaligned parameter hex,
  out-of-range sizes, a `pcap=3` record carrying a dump, and two pair-mode
  controls.

## Retained failures

- **Twelve-argument ABI assumption** (wrong decode): the traces were taken
  under labels `eval-args-1/2` and later **overwritten when those labels were
  reused** for the corrected capture — an evidence-retention slip. Their
  findings are recorded here; the create-mode counterpart of the same mistake
  is retained as `nvapi-trace-create-args-smoke.log`.
- **Deferred struct read**: `nvapi-trace-create-args-1-superseded-deferred-read.log`.
- **`created-null` over-reporting** (a flush logged a record whenever no create
  was pending, 18,818 spurious records): retained as
  `*-superseded-null-bug.log`; fixed with an explicit pending flag and the
  stale in-logger flush removed.
- **`pcap=0` ambiguity** (level-1 runs claimed a complete dump while carrying
  none): superseded by the `pcap=3` "not requested" value above; the affected
  captures were re-taken against the final source.

## What this does and does not establish

Established: the dispatch contract for every launch in a frame (kernel
identity, grid, block dims, shared memory, parameter size), the handle→kernel
mapping, the fixed 156-slot sequence and its stages, and — by measurement — the
decoder launch contract that the native reconstruction had derived by hand.

Not established: the *contents* of the parameter blocks as semantics (level 2
records them as hex; interpreting the buffer bindings is the next stage), any
intermediate tensor, the temporal/history wiring, and any native-vs-vendor
numerical comparison. Single machine, single Proton copy, deterministic runs;
the call-style stub assumes single-threaded dispatch (observed: every call in
one thread). Proprietary reuse and publication rights remain unresolved, and
the captured names/argument data are vendor material kept local.

## Next stage

1. **Decode the level-2 parameter dumps into buffer bindings** — for each of the
   156 slots, which device buffers it reads and writes. That turns the named
   sequence into a data-flow graph and is the input to…
2. **decoder-boundary intermediates** (the intermediate-tensor capture) and the
   native-vs-vendor comparison against `vr/logs/reference/`, using
   `reference-capture-report.py --mode float --tolerance`.

Containment unchanged: bwrap + Proton boundaries; traces and report JSON stay
local (gitignored); committed artefacts are source and documentation only.
