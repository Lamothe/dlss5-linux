# NvAPI parameter-block buffer bindings — 13/10/2026

## Result

**The level-2 parameter dumps are now decoded into a buffer-binding reference
graph: for each of the 156 launch slots, which device buffers its parameter
block references, which slots alias the same buffer, and how every buffer
rotates across the captured evaluations.** Derived by
`vr/nvapi-params-flow.py` from the `DLSS_NVAPI_ARGS=2` captures in
`vr/logs/capture/`, under Proton/Wine on the RTX 4070 Ti SUPER — no changes
to the model or to Proton. This is the input the next stage (decoder-boundary
intermediates) needs: the named sequence of `NVAPI-ARGS.md` is now a data-flow
graph.

The headline finding is that the vendor's own live bindings **corroborate the
statically recovered decoder contract end to end** — including the skip path
that `NATIVE-GRAPH.md` could only hypothesise.

## Findings

### The allocator runs two memory halves, and rotates them in lockstep

882 of the 1,453 word positions are address-shaped; **881 of them carry
exactly two values** across the 120 captured evaluations, and the two values
differ by exactly **`0x1CD00000`** (483 MB) — one fixed gap, with no
exceptions among those 881. The remaining address-shaped position carries a
single constant value (the weights-region address). The model allocates two
parallel copies of its rotating working set and swaps between them; the 25
constant bindings (weights region, dimension constants) do not rotate. The
rotation follows an `ABBAABBA…` period-4 pattern in 451 of the bindings; 25
bindings are constant across all evaluations; and 6 scalar positions switch
once after the first two evaluations and stay (recorded, not interpreted).
Each binding's pattern is recorded in the report, so this partition is
checkable from the tool output.

### Word typing is mechanical, not semantic

| Type | Rule | Count |
|---|---|---|
| `two-addr` | exactly 2 values, both address-shaped | 881 |
| `addr-candidate` | address-shaped, single value | 1 |
| `addr-multi` | address-shaped, 3+ values (rejected by `half-delta`) | 0 |
| `scalar` | everything else non-zero | 285 |
| `zero` | all-evaluations zero | 286 |

"Address-shaped" = `ADDR_MIN ≤ v < 2^32` and 512-byte aligned. **Read/write
direction is not inferred from the trace** — a parameter block records which
buffers are bound, not how the kernel touches them. Direction claims below
come from disassembly or kernel semantics and are attributed as such.

### 482 distinct bindings, 266 of them shared

Bindings are grouped by **exact per-evaluation value sequence** — two
positions alias only if they carry the same buffer in the same evaluation,
whatever the rotation phase — and labelled with an FNV-1a-64 id over that
sequence (the report emits these ids and the reference lists, never raw
parameter bytes — same discipline as `nvapi-args-report.py`; the hash is a
label, while the grouping itself is exact sequence equality). 266 bindings
are referenced from more than one `(slot, word)` position. A shared binding
means the same device address is bound at both positions in the same
evaluation: **either a data-flow edge (producer output bound as consumer
input) or arena reuse** (scratch recycled after a consumer finishes). The
graph records the aliasing; distinguishing the two cases needs the
intermediate capture of the next stage.

### The decoder slot contract is confirmed live

Slot 99 (`cc_dec_input_upsample_1024_512_tilesync_fp8`, 80-byte block) —
argument offsets 0–64 are address words, 64–80 are two packed u32 dimension
pairs. Mapping the constant-bank roles from `NATIVE-DECODER-IO.md` onto the
live bindings:

| Word | Arg offset | Recovered role | Live binding | Aliases |
|---|---|---|---|---|
| 0 | 0 | main input | private pair | slot 98 word 1 (`cc_vit_1d_repack_1d_to_2d_fp8`) |
| 1 | 8 | skip input | private pair | slot 55 word 2 (`cc_split_swin_16h_proj_pool_512_chained_fp8`) |
| 2 | 16 | output 0 | shared | 22 positions incl. slot 100 word 0 (`…ffwd_512_chained_fp8`) |
| 3 | 24 | output 3 (unused by kernel) | shared | slots 55 word 3, 56 word 0 (`…final_head_512_wait_fp8`) |
| 4 | 32 | counter wait/publication | **private** | — |
| 5 | 40 | output 2 (unused by kernel) | shared | slot 100 word 4 |
| 6 | 48 | partial sums | **private** | — |
| 7 | 56 | weight matrix reads | **private** | — |
| 8 | 64–72 | width pair | constant | 45 positions |
| 9 | 72–80 | output-axis overrides | constant | 66 positions |

Three points where independent evidence streams agree:

1. **The skip path is real.** `NATIVE-GRAPH.md` recovered statically that the
   decoder's second input comes from an earlier *projection/pooling* output —
   a skip-path candidate. The live binding for word 1 is the same address the
   first transformer stage binds at its `proj_pool` kernel. The static
   hypothesis is confirmed by the vendor's own binding.
2. **The counter and partial-sum buffers are decoder-private** — consistent
   with the counter protocol recovered in `NATIVE-DECODER.md` (a per-stage
   wait/publication region) and with the partial-sum role in
   `NATIVE-DECODER-IO.md`.
3. **The width dimension is 16**, bound at argument offset 68 — the `width`
   slot of the constant-bank census — matching the 16×16 output tile the
   native work derives for this kernel. The second pair (24, 28) occupies the
   two output-axis override slots (offsets 72/76).

Words 3 and 5 are bound although `NATIVE-DECODER-IO.md` shows this kernel
variant never references them: the host binds a wider output set than the
kernel uses. That the unused slots alias neighbouring-stage buffers is
recorded as observed, not interpreted.

## Method

`vr/nvapi-params-flow.py` parses the level-2 traces (the `pmem=` hex of each
`launch` record), splits parameter blocks into 64-bit words, types each word
position by its value set across evaluations, groups equal value sets into
bindings, and emits a report JSON (local, gitignored) with the binding table,
per-binding reference lists, rotation patterns and the pinned checks.

Modes: `--trace` (single capture), `--a/--b` (pair comparison), `--self-test`
(corruption controls over generated fixtures).

Pinned checks, all passing on both captures:

| Check | Verdict |
|---|---|
| `shape` | 18,720 launches = 120 × 156 |
| `complete-dumps` | every dump full length, `pcap=0` |
| `param-bounds` | every `psize` in `(0, 512]` (the trace's dump cap) |
| `word-shape` | every parameter block is whole 64-bit words |
| `slot-structure-stable` | word counts fixed per slot across evaluations |
| `half-delta` | all 881 two-addr positions differ by exactly `0x1CD00000`; no `addr-multi` |
| `decoder-psize` | slot 99 parameter block is 80 bytes (every evaluation) |
| `decoder-pointer-words` | offsets 0–64 address words, 64–80 scalars |
| `decoder-counter-pad` | word 5 − word 4 = 512 (`NATIVE-DECODER-LAYOUT.md` padding, every evaluation) |

Provenance note (not a check, because it compares two constants rather than
reading the trace): the decoder's `+0x80000` weight-word offset from
`NATIVE-DECODER-IO.md` equals the scale-tail offset 524288 recorded in
`NATIVE-DECODER-BINDING.md`, and lies inside the bound record's 525,312-byte
extent. This is checked where it can be — in `NATIVE-DECODER-BINDING.md`'s
own evidence — not re-asserted here.

Pair mode (`--a eval-args-1 --b eval-args-2`): verdict **`identical-bindings`**
— the two runs' parameter bytes are byte-identical at every `(evaluation,
slot)`. Combined with the per-evaluation rotation this pins the allocator
behaviour precisely: **deterministic across runs, two-half rotating within a
run.** (`reference-capture-report.py` independently found the reference
tensors bit-identical across runs; this extends determinism to the binding
addresses.)

Self-test: **15/15 corruption controls** fail for the reason their name
states — dropped launch, `pcap` mismatch, truncated hex (evaluation 0 and a
later evaluation, the crash direction a naive implementation would miss),
broken word shape, zero-size parameter block, corrupted rotation delta,
three-valued address position, corrupted decoder parameter size, corrupted
counter padding, scalar word promoted to an address pair, pair divergence,
malformed record, missing evidence file, and same-value-set/opposite-phase
positions wrongly merging into one binding.

## Evidence

Local under `vr/logs/capture/` (gitignored, like the traces): `params-flow-2.json`
(single-capture analysis of `eval-args-2`), `params-flow-pair.json` (pair
comparison). Committed artefacts are the tool and this document. No raw
parameter bytes appear in either.

## What this does and does not establish

Established: the complete buffer-binding table for every launch slot, the
two-half rotation geometry (`0x1CD00000`, `ABBAABBA`), the binding aliases
that connect the decoder to its neighbouring stages, and — by live vendor
bindings — the decoder's skip path, private counter/partial-sum buffers and
width dimension.

Not established: read/write direction per binding (the trace does not carry
it), which shared bindings are data flow versus arena reuse, any intermediate
tensor value, the temporal/history wiring beyond the two-half rotation, and
any native-vs-vendor numerical comparison. Single machine, single Proton
copy, deterministic runs. Bindings are grouped by exact sequence equality, so
grouping cannot merge distinct buffers; the FNV-1a-64 id is only a label and
is not collision-free in principle, but the report's per-binding reference
lists make any such collision visible. Proprietary reuse and publication
rights remain unresolved; captured addresses and argument data are vendor
material kept local.

## Next stage

Decoder-boundary intermediates: use this binding table to identify which
buffers cross the slot-99 boundary (words 0, 1 in; word 2 out), capture their
contents per evaluation, and run the native-vs-vendor comparison against
`vr/logs/reference/` with `reference-capture-report.py --mode float --tolerance`.
The two-half rotation pins the capture window: sample one A-half and one
B-half evaluation.
