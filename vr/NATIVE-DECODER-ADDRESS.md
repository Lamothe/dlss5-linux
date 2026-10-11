# FP8 decoder coefficient reads and counter protocol — 11/10/2026

## Scope

`native-decoder-address-probe.py` adds a bounded **static GPU-instruction audit**
for `cc_dec_input_upsample_1024_512_fp8` only. It never creates a CUDA context,
loads a GPU module or launches a decoder. The Windows/DX12/Proton VR PoC and all
existing native probes remain unchanged. No complete native image exists.

The parser reads raw SM89 instruction bytes through a sealed anonymous memfd
and the existing installed CUDA 13.4 `nvdisasm`, whose executable hash is checked.
Only metadata and our source snapshots are deliberately saved. No new cubin,
weights or proprietary disassembly file is deliberately produced. Tool stdout
and diagnostics remain in memory and are not printed on failure. Core dumps are
disabled, but exec-time dumpability, same-UID `/proc`, swap and memory erasure
remain uncontrolled. Existing older retained disassembly is not removed.
Proprietary reuse/publication rights remain unresolved.

## New evidence

### Coefficient read footprint, conditional on the group selector

The parameter constant-bank metadata establishes a base of `0x160`, size 80.
This ties constant-bank references `0x198` and `0x180` to argument offsets **56**
(weight-state candidate) and **32** (output1/counter candidate), respectively,
rather than assuming the compiler's usual parameter placement. Existing host
anchors check the 80-byte argument field provenance separately.

Initial coefficient loads and loop prefetches use sixteen 16-byte reads per
lane. A candidate byte-address expression relative to slot56 is:

```text
4 * (partition*32768 + (group*256 + warp*128)*8 + slab*128)
+ lane*16 + vector_offset
```

Here `partition=CTAID.Z` ranges 0–3, `warp=TID.Y` ranges 0–1, `lane` ranges
0–31, slabs are 0,64,128,192 and vector offsets are 0,0x200,...,0xe00 and
0x4000,0x4200,...,0x4e00. **Group=0/1 is an assumption**: the reciprocal-based
CTA-X quotient calculation has not been independently emulated or validated.
The coefficient-read result is conditional on that domain, not a complete proof
of every reachable device read.

Within that conditional domain, all **32,768 16-byte reads** cover the first
**524,288 bytes** exactly once, without gaps or aliases. A sorted contiguous-set
oracle is independent of the candidate physical address expression. Three
literal answers and deliberately missing/duplicated ranges strengthen the CPU
check. This recovers a coefficient footprint, **not input/output channel weight
packing, suffix scale semantics or learned-record binding**. Other reads beyond
this coefficient region remain outside this bound.

### Ordinary (non-tile-sync) decoder completion protocol

For already padded positive dimensions, the inspected counter address is:

```text
slot = 2*CTAID.Y*(input_axis1/4) + CTAID.X
byte_address = argument_slot32 + 4*slot
```

The numerical domain is restricted to padded axes ≤4096, valid launch X/Y,
so the tested address arithmetic cannot overflow. CPU enumeration covers five
square/non-square shapes and a literal `(8,20,x=9,y=1) → (slot19,count20)` answer.

The inspected code contains a wait path comparing the counter against
`CTAID.Z−1`, followed by GPU-wide memory ordering and a single-thread publication
of `CTAID.Z`. This is consistent with the existing native reset value −1 and
sequential partition completion 0,1,2,3. **This is static path evidence, not proof
that every CTA takes the wait path, that a launch cannot deadlock, or that the
original scheduler supplies the assumed buffers.** The early branch and
intermediate predicates remain outside full control-flow validation.

A bounded register-writer audit rejects unexpected redefinitions of selected
address/protocol registers between their producers and consumers, including
wide paired-register writes and predicate-first data destinations. Eight deliberate
register-clobber controls check these cases. It is not a general SASS emulator or exhaustive CFG
proof. Do not infer a safe decoder launch from these checks.

## Reproduce

Read `README.md` first, then from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 vr/native-decoder-address-probe.py self-test
bash vr/run-native-decoder-address-probe.sh inspect
bash vr/run-native-decoder-address-probe.sh negative
```

The launcher retains immutable input source snapshots/hashes, uses the existing
probe lock, network/home isolation, private temporary directories and a 90-second
limit. It exposes **no NVIDIA device nodes**. CUDA 13.4's existing disassembler
must be present; no toolkit download/install is performed.

The negative run corrupts each of 102 interpreted SASS anchors independently.
Those checks validate the fingerprint/checker, **not the numerical inference**.
CPU tests also reject malformed SASS/constant-bank metadata, invalid address
domains, register clobbers and wrong coverage/row-stride formulas.

## Verification and remaining work

Final-source positive/negative evidence and independent-review outcome will be
recorded below after completion. Exploratory command section selection failed
before GPU work, and early launcher runs failed due to duplicated disassembly
ranges or a missing Python import; none is successful inference evidence.

Remaining: prove group-selector arithmetic, resolve main/skip activation layouts,
all output/scratch extents, suffix coefficient semantics, actual learned-weight
binding and dynamic arena associations. Then build a guarded synthetic numerical
decoder test, with independent references and skipped/wrong execution controls.
Full native neural rendering is still unproven; no commit or push occurs here.


Independent review found two holes in the initial register-writer audit: paired
`IMAD.WIDE` destinations and predicate-first `LOP3` data destinations were missed.
Both were fixed and eight clobber controls now cover scalar, wide, predicate-first,
64-bit and 128-bit writes. A subsequent check initially mistook predicate-only
comparisons for data writers; these are now correctly distinguished. Those
intermediate runs are not final-source evidence. Final repeat tests and follow-up
review are recorded below after completion.


### Final-source evidence

- `decoder-address-inspect-lXUataV6` and `decoder-address-inspect-QyWNoFBA`:
  identical metadata reports. 102 SASS anchors and host argument provenance pass.
- `decoder-address-negative-T3D0cOPC`: 102 independently corrupted SASS anchors
  rejected. These are checker controls, not GPU numerical controls.
- `decoder-inspect-CzDfV2o6`: unchanged original decoder static regression passes.
- `counter-execute-5CZ0n25X`: unchanged existing native counter GPU regression,
  **12 launches / 14 readbacks**, all 3,842 counter values reset to −1 and 9,870
  guard/feature sentinel values unchanged. No decoder kernel launched.

CPU tests check conditional coverage of 32,768 16-byte coefficient reads,
three literal coefficient addresses, five counter shapes, five malformed SASS
controls, four bank-metadata controls, eight register-clobber controls, three
wrong-coverage/stride controls and ten invalid addressing domains.

Follow-up independent read-only review **passed** after the fixes, checking code,
limits and final reports. The parent separately recomputed immutable snapshot
and current-source hashes against all five manifests, confirmed zero exit
statuses and identical repeat reports, and passed CPU tests, AST, shell syntax,
new-file whitespace and existing Git diff checks. Process inspection found no
surviving probe/disassembler; the NVIDIA driver query responded normally.

Only this document and the two new address-probe/launcher files were created.
Earlier PoC/native/scene files were not modified or reverted. Nothing committed
or pushed. Next remains activation address layouts, group-selector proof and all
buffer/weight associations before a guarded decoder computation.
