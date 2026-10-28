# Native Vulkan stress and sustained-operation hardening

## Scope

Stage 4 of the performance brief: sustained-operation and stability
testing of the GPU path. The CPU reference path is unchanged; every check
below requires bit-identical results against an independently accumulated
scalar reference (the shader's `precise` key-order accumulation).

## What is covered (native-vulkan-stress-test.cpp; 1,997 checks at -O2/-O0)

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
5. **Forced failures (fault injection) + the live generation counter.**
   The test-only `VulkanFaultPlan` seam makes the next `vkQueueSubmit` or
   `vkMapMemory` fail synthetically WITHOUT touching the driver (armed for
   exactly one failure; inert at -1 — production behaviour unchanged).
   Each armed failure surfaces as `Status` with the output sentinel
   untouched, and the next unarmed run is correct. The countdown
   semantics are checked deterministically at the seam (one run issues
   several submits, so run-level "Nth submit" expectations are avoided).
   The generation counter is now LIVE: `vulkan_matmul_destroy` preserves
   it (it is the matmul's identity, not a resource) and create advances
   it, so a grow-only rebuild (destroy + create) moves 1 -> 2 and the
   cache hit rewrites the entry's descriptor against the new buffers
   (`written_gen`, `written_in` asserted). The ensure-insert path records
   `written_gen` (previously left 0).

## Sustained operation (the 50-frame demo run)

The demo runs 50 frames with a mid-run reset (`--reset-at 20`) on the GPU
projection backend. Checked: the arena high-water mark is flat (the demo
prints it per frame; it must equal the single-frame value), the
frames-since-reset counter walks 0..19 then resets to 0, and the process
RSS (sampled every 5 s) is flat across the run. The measured numbers are
in NATIVE-PERFORMANCE.md.

## What is NOT covered — real device loss (honest gap)

The forced-failure seam injects failures at the CALL RESULT level (our
code's `VkResult` handling); it cannot reproduce a driver's behaviour
after `VK_ERROR_DEVICE_LOST`. Forcing that needs driver fault injection
(`VK_EXT_device_fault` fault submission or a faulting layer) which we
deliberately do not add: no proprietary runtime probes and no fragile
driver shims. What IS established:

- every Vulkan call site whose failure can affect a run checks its
  `VkResult` and returns a `Status` (never crashes, never writes outputs
  on a failed transfer/dispatch — the G1 review verified the failure-path
  cleanup; the two `vkEnumeratePhysicalDevices` results are ignored by
  design — failure leaves the count zero and the run skips cleanly);
- the weight cache's stale-set rewrite is guarded by the LIVE generation
  counter (verified: a rebuild moves the generation and the next hit
  rewrites the descriptor) together with the `written_in`/`written_out`
  handle check;
- a failed run leaves the CPU reference path fully functional (the GPU
  backend is optional at every seam).

What is recovered after a FORCED failure (tested): nothing needs recovery
— the run returns `Status`, outputs stay sentinel-untouched, the CPU
reference path is unaffected, and the next unarmed run is correct (the
entry cache and matmul stay usable).

What is NOT implemented: real device-loss recovery. A `VK_ERROR_DEVICE_LOST`
mid-run is expected to surface as a failed `VkResult` on the next
submission -> `Status` -> the row's caller rejects; the demo prints the
failing frame. What is NOT recovered and NOT claimed: the device, queues
and in-flight command buffers (after device loss the driver's teardown
ordering is undefined and the caller must discard the whole Vulkan context
— `vulkan_close` + recreate — before retrying). This expectation is NOT
verified against a real device loss and must not be read as a claim.

## Provenance and honesty

The Vulkan kernels are original work in this repository (see
NATIVE-VULKAN.md for the pinned-reference relationship and the full MIT
notice). Speed numbers are real measurements on the stated hardware; with
synthetic weights the OUTPUT is arbitrary — this hardening shows the GPU
path runs stably and deterministically, not image quality and not vendor
equivalence. No proprietary weights, captures, payloads or binaries; no
runtime or capture probes.
