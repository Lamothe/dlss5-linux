# Decoder group-selector reconstruction — 11/10/2026

## Result and scope

`native-decoder-selector-probe.py` investigates the **CTA-X channel-group
selector** left assumed by `NATIVE-DECODER-ADDRESS.md`. A bounded static slice
matches the pinned SM89 decoder, and **our own PTX arithmetic translation** is
checked natively against ordinary integer division. **The vendor decoder is not
launched**, no learned weights or vendor GPU modules are uploaded by this probe,
and no complete native neural image is produced. The Windows/Proton/VR PoC and
all existing probes remain unchanged.

For **positive, already padded input axis1 in 4..4096**, the inspected host grid
is `2*(axis1/4)` along X. The reconstructed positive-domain selector is:

```text
divisor = input_axis1 / 4
group = CTAID.X / divisor    # integer division
```

The first `divisor` workgroups select group 0, and the next `divisor` select
group 1. This strengthens the earlier coefficient-read footprint **conditionally
on the reconstructed instruction semantics and inspected backend path**; it is
not a captured tensor binding or proof of every device access.

## Independent checks

- All 1,024 divisors (padded axes 4..4096) and all their valid CTA-X positions:
  **1,049,600 coordinate cases** per GPU run, compared with Python `x // divisor`.
- The owned PTX exports its actual reciprocal bits as well as its group. The CPU
  reconstruction consumes those bits and must independently reproduce each GPU
  result. No reciprocal/output buffer is deliberately saved in the reports.
- A CPU sweep checks five reciprocal bit patterns around the nearest rounded
  reciprocal for every coordinate. **4,920,094 safe-seed comparisons** pass;
  **327,906 overestimating-seed cases** are explicitly rejected rather than
  silently treated as valid. This is not an ISA-wide reciprocal error guarantee.
- Six literal answers and seven invalid-domain controls.
- **130 SASS anchors**, including the complete straight-line arithmetic slice
  and prior address/provenance checks, plus existing host/backend contracts.
  Independently corrupting each anchor must fail. These are fingerprint/checker
  controls, not neural numerical validation.
- GPU source and destination guards remain unchanged. Source buffers are checked
  in their entirety. Destination payload is checked against independent division.
- Skipped execution and intentionally increased divisors must cause numerical
  disagreement while preserving the tested allocation guards. These finite guards do not establish a maximum effective output address or generally safe allocation bound.

### Important arithmetic detail

The reciprocal refinement uses a high-half multiply-add whose addend is a
register pair (zero low word, old reciprocal high word). Treating it as an
ordinary 32-bit addend gives a different calculation. The owned translation
expresses this as `seed + high32(seed * product)`, modulo 2^32.

Float-to-u32 conversion clamps rather than wraps. An exploratory CPU check also
showed that simply assuming all nearby reciprocal seeds are safe is wrong: some
seeds overshoot the integer inverse. The current checks reject those cases, and
all actual successful GPU cases must satisfy the safe-seed condition. No exact
vendor-SASS execution or equivalence of every MUFU/conversion edge case is
claimed. PTX is a virtual ISA, not documentation for all SASS encodings; see
[NVIDIA's PTX ISA documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).

## Reproduce

Read `README.md` first, then from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 vr/native-decoder-selector-probe.py self-test
bash vr/run-native-decoder-selector-probe.sh inspect
bash vr/run-native-decoder-selector-probe.sh negative
bash vr/run-native-decoder-selector-probe.sh execute
bash vr/run-native-decoder-selector-probe.sh skip
bash vr/run-native-decoder-selector-probe.sh wrong-divisor
```

The launcher uses the existing exclusive native-probe lock, immutable source
snapshots/manifests, private temporary paths, hidden home/network and a 90-second
limit. Only numerical modes expose NVIDIA device nodes. Static disassembly uses
the existing hash-pinned tool and sealed anonymous memfd. No new vendor module,
weight or disassembly file is deliberately produced. Our own source/metadata
are retained. Core dumps are disabled, but swap, exec-time dumpability, same-UID
`/proc`, driver persistence and erasure remain uncontrolled. Older retained
vendor disassembly remains untouched. Proprietary reuse/publication rights are
unresolved.

## Verification

Pre-review positive runs `decoder-selector-execute-j7NTz9fA` and
`decoder-selector-execute-8yhgsebi` each passed **4 launches / 8 readbacks** and
1,049,600 coordinates. Reports are identical. Controls
`decoder-selector-skip-Y6ClZEbL`, `decoder-selector-wrong-divisor-SunMJAnL` and
`decoder-selector-negative-9yTdCheo` passed. Existing address regression
`decoder-address-inspect-ouTQzdpn` passed unchanged.

Exploratory CPU seed-range checks and two report-key lookups failed before any
GPU work; they are not successful evidence. One earlier owned-PTX run passed,
but was superseded by stronger host/backend report checks and is not final-source
evidence. Independent review and final evidence will be recorded after completion.

## Remaining work

The selected positive-domain group-selector question is closed by this bounded
owned-PTX translation test. Later stages document bounded main/skip and partial-
sum layouts (`NATIVE-DECODER-LAYOUT.md`), output-store arithmetic (`NATIVE-DECODER-IO.md`),
coefficient permutation and tested single-product record binding
(`NATIVE-DECODER-PERMUTATION.md`, `NATIVE-DECODER-BINDING.md`). These are still
conditional/static or bounded experiments: they do not establish exact vendor-
SASS selector semantics, complete device-access coverage, runtime arena
associations, suffix coefficient meaning, multi-term accumulation, complete
control flow, or a native rendered image. A synthetic decoder numerical test
remains gated by the complete output-address proof in
`NATIVE-DECODER-STORE-ENVELOPE.md`, current authorization and safety preflight;
it cannot substitute for authorised Feature 18 ABI/interop evidence and the
raw-input-to-launch binding needed for full native inference.
Nothing is committed or pushed.

### Final-source evidence and independent review

Independent read-only review **passed**, finding no concrete arithmetic,
validation, bounds, cleanup, containment or documentation defect. It explicitly
confirmed the limitation: this tests our PTX reconstruction, not exact vendor
SASS semantics. Reviewer execution tools were unavailable, so parent execution
remains the numerical evidence; the reviewer inspected sources and reports.

Post-review run `decoder-selector-execute-EWrNhN5p` passed unchanged. Across the
three final-source successful positive GPU runs: **12 launches / 24 readbacks**,
**3,148,800 coordinate comparisons** against integer division and reconstructed
actual reciprocal bits, with all source/guard bytes unchanged. The negative
GPU run adds four launches, and the skipped control launches none; these are
separate control evidence, not positive computation counts.

`decoder-selector-inspect-fJjWkQkb` passed static checks. Unchanged native reset
regression `counter-execute-YadtLBjj` passed **12 launches / 14 readbacks**, 3,842
counters reset to -1 with 9,870 guard/feature sentinel words unchanged. No vendor
decoder kernel ran. Existing address CPU tests, AST/shell syntax and Git diff
whitespace checks passed.

Parent independently recomputed all immutable snapshots and current-source
hashes against **nine final-source manifests**, verified empty stderr and zero
exit statuses, and confirmed the two repeat positive reports were identical.
Process inspection found no surviving probe or disassembler. NVIDIA responded
normally afterwards. Only this document and the two new selector-probe/launcher
files were created this round; no earlier source files were changed, committed
or pushed.
