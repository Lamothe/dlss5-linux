# Decoder output-store envelope — static prerequisite audit

## Result

**No usable output allocation bound is established.** The IO probe fixes the observed SASS forms for 16 stores and models selected arithmetic for test vectors, but it expressly does not model pointer-carry assembly, complete CFG liveness, or runtime loop/coordinate limits. Therefore neither the observed output window nor the helper's accepted index range is a proof of the set of effective addresses the live kernel can generate.

## Address arithmetic from tracked source

`vr/native-decoder-io-probe.py` models each selected output pointer as

`out0_base + 4 * index + 16 * lane + slot`

for `lane ∈ [0,31]` and `slot ∈ {0,0x200,0x400,0x600,0x2000,0x2200,0x2400,0x2600}`. Each observed `STG.E.NA.128` writes 16 bytes. For a *stipulated* unsigned 32-bit index bit-pattern, the raw byte offset is bounded arithmetically by

`4*(2^32 - 1) + 16*31 + 0x2600 + 16 = 0x4000027fc` bytes

from the base (exclusive endpoint; the IO probe's CPU self-test now independently asserts this arithmetic and the difference from 256 MiB). This is only a mathematical envelope for the displayed formula under the stated interpretation. It exceeds the launch probe's 256 MiB `out0` allocation by 16,911,443,964 bytes; that existing allocation is therefore not a safe allocation for this conditional full-domain envelope. It is **not** a proven kernel envelope: the probe's `MAX_INDEX = 2^30 - 1` is a helper-domain restriction, not evidence about all runtime values; pointer formation also contains wide operations and carry-sensitive address formation which the probe explicitly does not emulate. The reachable set of paths, actual operand signedness, base transformations and address-width behaviour have not been established.

The native launch note records one tested 8×8-input configuration, 16×16 output axes and grid (4,2,4), plus an observed output interval. It does not prove the envelope for that run because byte-completeness and out-of-window writes were not observable beyond the finite allocation. The input row/index expressions in the selected straight-line interpreter are test vectors, not loop or grid bounds. Host-wrapper launch sizing and a conditional generic CUDA dispatch do not, without an end-to-end proof tying the launched grid and all control paths to each store, establish tighter bounds.

## Bounded structural CFG inventory (25/10/2026)

The static IO probe now builds direct-branch reachability from the pinned, contiguous SASS map and reports the immediate predecessors of each of the 16 output stores. Conditional branch targets and fall-through are both retained; predicates are unconstrained. Predicated `EXIT`/`RET` instructions retain the fall-through path, while unpredicated termination does not. Recognised direct branches are traversed conservatively. BSSY/BSYNC pairs include the sequential and declared reconvergence successors (conservative over-approximation); other reconvergence/synchronisation and listed indirect transfers fail closed. Missing direct/reconvergence targets, unmatched or non-dominating BSSY/BSYNC barriers, and malformed instruction maps also fail closed. CPU fixtures cover direct/conditional taken and fall-through edges, a backedge, BSSY/BSYNC join alternatives, bad/missing targets, unmatched barriers, unsupported transfer, and predicated/unpredicated exit behaviour. This bounded opcode classification is not a general SASS control-flow decoder.

This is a bounded graph-shape inventory only. It does not establish predicate values, complete path semantics, address-register liveness/ranges, loop bounds or a safe allocation envelope. It does not qualify decoder execution; the required proof remains open.

## Required next proof before any decoder GPU run

1. Extend the static audit beyond structural reachability to cover every store's relevant predicate/loop semantics and every writer/use of address and base registers; reject unknown opcodes or paths.
2. Emulate exact unsigned/signed arithmetic widths, carry propagation and pointer additions; derive the full effective byte interval from the exact supplied `out0` base, including store width. Validate the emulator against hand-derived edge cases and mutation controls.
3. Tie operand ranges to the exact host launch grid, device ABI and argument values used by the experiment. If those ranges cannot be justified from authorised/version-matched source, retain the full representable envelope or stop.
4. Independently review this proof and any CUDA VMM reservation/mapping design (including granularity, device access, overlap and fault isolation) before considering a synthetic run. A guard fault invalidates the run; it is not a successful bounds test.

Until this gate is closed, do not submit the decoder kernel. No proprietary values are needed for this static proof; no GPU execution or payload extraction was performed for this audit. Feature 18 API/interop provenance is a separate blocker for the native live-input pipeline.