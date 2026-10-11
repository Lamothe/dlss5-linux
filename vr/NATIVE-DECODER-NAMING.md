# Influenced output naming — 12/10/2026

## Scope and result

`native-decoder-influence-names.py` names the decoder's influenced output cells
from the retained footprint evidence, and extends the influence-footprint
constraint lock from eight sample offsets to the **full 512-unit set**. It is a
CPU-side analysis of evidence produced by
`native-decoder-weights-probe.py`; it launches no GPU work.

Under the declared **contiguous-channel layout hypothesis**
`offset = (y*16 + x)*512 + c` for the tested `16×16×512 = 131072`-byte output
extent, the coupled footprint of every (variant, impulse-partition) group names
exactly:

| Dimension | Named values |
|---|---|
| Output rows `y` | **{0, 4}** |
| Output columns `x` | **0–15** (full width) |
| 16-channel blocks | **{0–7, 16–23}** → channels **[0–127] ∪ [256–383]** |
| Channels per touched block | `c0 + {0, 1, 2, 3, 8, 9, 10, 11}` |

So each influenced 16-byte unit is one 16-channel group at one output position,
and the touched channels inside it are two of its four 4-byte words (offsets
`0–3` and `8–11`). Totals: **32 positions × 16 blocks = 512 units**;
**128 influenced channels per position**; **4,096 (position, channel) pairs**,
matching the canonical `changed_count` exactly. The footprint is **identical
across all four impulse partitions and both word-table variants** (enforced).

This is a naming **conditional on the layout hypothesis** — the semantic tensor
layout is not proven. Hypothesis-independently, the same set factorises as
`offset = row_term*8192 + column*512 + channel_block*16` with
`row_term ∈ {0, 4}`, `column ∈ 0–15`, `channel_block ∈ {0–7, 16–23}`; this
lossless factorisation is the canonical constraint any future permutation or
layout claim must reproduce. Under the channel-major alternative
(`offset = c*256 + y*16 + x`) the same set names 64 channels
(`{0–31, 128–159}`) with 8 positions each.

## Method

The probe now retains `unit_offsets` (full list), `unit_byte_offsets` (pattern
union), `uniform_unit_pattern` and `unit_offsets_truncated` in every
`changed_summary`; previously only eight sample offsets survived, which made
naming impossible from retained evidence. The naming tool then:

1. Enforces every coupled/negative footprint in every group against the
   canonical 512-unit set (strict element-typed list equality — floats and
   booleans are rejected element-wise), plus unit count, byte pattern
   `[0, 1, 2, 3, 8, 9, 10, 11]`, uniformity, span `[0, 40827]`, the sample
   anchors with sign symmetry, the `truncated` flags and the no-overclaim
   flags.
2. Derives the named structure under both layout views and reports it with
   `semantic_layout_proven=False`.

## Verified results (runs `decoder-weights-execute-IUvcUu7c`, `-JvrNtvNL`)

- Both runs pass the naming tool and the existing influence oracle (16 anchors
  each) with **identical unit lists** across all 16 coupled/negative footprints
  per run.
- Structure: `named_rows = [0, 4]`, 32 named positions, 16 channel blocks,
  4,096 influenced pairs, factorisation check true — consistent across runs.
- **Self-test:** 13 known answers (contiguous and channel-major mappings,
  influenced-channel derivation, report structure, canonical-set
  self-consistency) and **42 corruption controls** — shifted/unaligned/dropped/
  duplicated/empty unit lists, value-correct float and boolean-in-element type
  corruptions (rejected element-wise; Python list equality alone would accept
  them), truncation and uniformity flags, count and span fields, sample anchors
  and sign symmetry, missing and malformed evidence objects, kernel/mode/
  overclaim flags, group key corruption, and non-unit-start offsets passed to
  `influenced_channels` — all rejected; baseline accepted.
- Mapping inputs reject non-int, boolean and out-of-range offsets.
- Regression: the probe self-test (including four unit-capture known answers),
  the skip control (`decoder-weights-skip-VAbcjrLZ`) and the influence oracle
  all pass against the current source.

Independent review of this stage found 1 must-fix and 6 advisory findings: the
must-fix was that type-corrupted unit lists (floats, `False == 0`) passed
value-based equality — fixed with element-wise int typing plus dedicated
controls; the advisory findings (flag-semantics documentation, `truncated` and
sample-anchor enforcement as defence in depth, `influenced_channels`
alignment/range validation, sign-correct fixture values, derived channel-range
strings) were all addressed and the evidence re-run against the final source.

## What this establishes — and what it does not

Established: the influenced output cells are **named and locked** under a
declared layout hypothesis, with a hypothesis-independent factorisation that
any competing naming must reproduce. The rows `{0, 4}` and the
first/third-channel-quarter structure are now concrete targets the recovered
upsample model must explain.

Not established: the semantic tensor layout (`semantic_layout_proven=False`),
which logical `(output channel, input channel)` pair each matrix byte serves
(the permutation itself), the word-table semantics, magnitude linearity,
learned-weight binding, or any neural-rendering quality. **No GPU work ran in
this stage; no learned weights were used; no neural-rendered image was
produced.**

## Next stage

1. Fine-grained influence sweeps (code matrix sub-partitions; sweep impulse
   positions) to resolve `(o, i)` per physical byte against these named cells.
2. Map the stored decoder-weight candidate record through the recovered
   permutation.
3. Capture Proton reference evaluations for numerical comparison.

Containment is unchanged: analysis runs on retained evidence only; the
launcher retains bubblewrap, no network, no home, pinned source snapshots,
hash-verified DLL read as data, 300 s bound and disabled core dumps. Swap,
driver persistence and memory erasure remain uncontrolled; proprietary
reuse/publication rights remain unresolved. No proprietary payload files are
deliberately created.

Evidence: `research/native-linux/runs/decoder-weights-execute-IUvcUu7c`,
`-JvrNtvNL` (full unit capture), `decoder-weights-skip-VAbcjrLZ`. Source:
`vr/native-decoder-influence-names.py`, `vr/native-decoder-weights-probe.py`.
