# Native Vulkan stress and sustained-operation hardening

## Scope

Stage 4 of the performance brief: sustained-operation and stability
testing of the GPU path. The CPU reference path is unchanged; every check
below requires bit-identical results against an independently accumulated
scalar reference (the shader's `precise` key-order accumulation).

## What is covered (native-vulkan-stress-test.cpp; 1,965 checks at -O2/-O0)

1. **Init/teardown cycles (50).** Full `VulkanMatmul` create -> resident
   run -> destroy cycles: every result bit-identical, and the CPU-side
   allocation count (operator-new accounting) back to baseline after each
   cycle. No degradation across cycles.
2. **Long mixed-shape resident run (200 runs: 50 x 4 shapes).** Four
   shapes (17x256x8, 64x512x16, 9x768x384, 128x1024x24) through one cache
   and one matmul: every result bit-identical. Repeat-content determinism
   is covered by the cycle check (the same workload 50 times).
3. **Weight-cache content hash and capacity.** (a) The same pointer with
   CHANGED content re-uploads (the G5 content hash) and stays correct
   across 5 rounds. (b) 520 DISTINCT pointer keys (511 accepted with one
   entry pre-cached): the first key past capacity is rejected
   (`cache.count <= kCapacity` asserted, no silent wrap) and entries
   already cached remain usable and correct.
4. **Failure paths.** Null in/out, zero tokens, entry-shape mismatch and
   oversized tokens/rows reject with `Status`, leave the output sentinel
   untouched, and a subsequent good run remains correct.

## Sustained operation (the 50-frame demo run)

The demo runs 50 frames with a mid-run reset (`--reset-at 20`) on the GPU
projection backend. Checked: the arena high-water mark is flat (the demo
prints it per frame; it must equal the single-frame value), the
frames-since-reset counter walks 0..19 then resets to 0, and the process
RSS (sampled every 5 s) is flat across the run. The measured numbers are
in NATIVE-PERFORMANCE.md.

## What is NOT covered — device-loss behaviour (honest gap)

Forcing `VK_ERROR_DEVICE_LOST` requires driver fault injection
(`VK_EXT_device_fault` fault submission or a faulting layer) which we
deliberately do not add: no proprietary runtime probes and no fragile
driver shims. What IS established:

- every Vulkan call site whose failure can affect a run checks its
  `VkResult` and returns a `Status` (never crashes, never writes outputs
  on a failed transfer/dispatch — the G1 review verified the failure-path
  cleanup; the two `vkEnumeratePhysicalDevices` results are ignored by
  design — failure leaves the count zero and the run skips cleanly);
- the weight cache's stale-set rewrite is guarded by the
  `written_in`/`written_out` handle check (the entry's generation field is
  currently inert: `vulkan_matmul_create` resets the struct before
  incrementing, so `written_gen` never distinguishes a rebuild — recorded
  here rather than claimed as protection);
- a failed run leaves the CPU reference path fully functional (the GPU
  backend is optional at every seam).

What is untested: actual device-loss recovery (re-creation of the device,
in-flight command buffer behaviour and driver-specific teardown ordering).
A real device loss mid-run is expected to surface as a failed `VkResult`
on the next submission -> `Status` -> the row's caller rejects; the demo
prints the failing frame. This expectation is NOT verified against a real
device loss and must not be read as a claim.

## Provenance and honesty

The Vulkan kernels are original work in this repository (see
NATIVE-VULKAN.md for the pinned-reference relationship and the full MIT
notice). Speed numbers are real measurements on the stated hardware; with
synthetic weights the OUTPUT is arbitrary — this hardening shows the GPU
path runs stably and deterministically, not image quality and not vendor
equivalence. No proprietary weights, captures, payloads or binaries; no
runtime or capture probes.
