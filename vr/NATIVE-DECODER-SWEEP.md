# Decoder fine-grained influence sweep — 12/10/2026

## Scope and result

`native-decoder-sweep-probe.py` is the **fine-grained influence sweep** that the
weight-binding gate calls for (see [NATIVE-DECODER-LAUNCH.md](NATIVE-DECODER-LAUNCH.md)
and [NATIVE-DECODER-WEIGHTS.md](NATIVE-DECODER-WEIGHTS.md)). It launches the pinned
SM89 `cc_dec_input_upsample_1024_512_fp8` through native Linux CUDA on the RTX
4070 Ti SUPER — no Windows code, DirectX or Proton — and sweeps two axes at finer
granularity than the whole-partition coding used before:

* **Matrix axis** (weight -> output): hold the impulse fixed at unit 0 and code
  only a sub-block of the coupled matrix partition; observe which output cells
  respond.
* **Impulse axis** (input -> output): code the whole coupled partition and move
  the single 16-byte main-input impulse within its partition; observe which
  output cells respond.

This resolves **bounded slices of the (o, i) reach structure** against the named
output cells of [NATIVE-DECODER-NAMING.md](NATIVE-DECODER-NAMING.md). It does
**not** recover the full matrix permutation, does **not** use learned weights and
produces **no neural-rendered image**.

## Method

Output cells are named with the general hypothesis-independent factorisation
`unit_offset = row_term*8192 + column*512 + channel_block*16` (any 16-byte-aligned
unit inside the `16×16×512 = 131072`-byte output extent). The **canonical set**
(the locked unit-0 footprint) is `row_term ∈ {0,4}`, `column ∈ 0–15`,
`channel_block ∈ {0–7, 16–23}` — 512 units.

Matrix sub-blocks are coded inside coupled partition 0 (`[0, 131072)` of the
slot-56 coefficient region) at 512-byte and 16-byte granularity. Impulse
positions sweep the 16-byte units of main-input partition 0. Plan: `base`,
`coupled-ref` (whole partition 0 coded, impulse unit 0), `decoupled-ctrl`
(partition 1 coded under impulse 0), 20 matrix sub-blocks and 12 impulse
positions — 35 sweep entries per run (35 decoder launches; 70 kernel launches
with the per-entry counter resets).

## Verified results (runs `decoder-sweep-execute-qf4GSDp3`, `-3rTSN1Na`)

Each run: **70 kernel launches, 280 readbacks**; `unit_offsets` **identical**
across both runs and aggregate metrics match (oracle-verified; oracle output
retained as `oracle.json` / `oracle-pair.json` beside the run reports). Skip
control `decoder-sweep-skip-HvNnG4lx`: counters `[−1]×8`, output untouched,
guards intact.

**Matrix axis.** Coding fewer weights influences a subset of the canonical
footprint (enforced). Structure (the live weight region is **sparse**, not
contiguous):

* A **512-byte** matrix sub-block influences exactly **32** output cells where it
  is live — one full column (rows `{0,4}` × 16 channel blocks). Sub-block offset
  selects the column (e.g. `@0` → column 0, `@8192` → column 8). Sub-blocks at
  `≥16384` influence nothing (dead weight region).
* A **16-byte** matrix sub-block influences **8** cells where it is live (rows
  `{0,4}` × 4 blocks, e.g. `@0` → blocks `{0,4,16,20}`, `@256` → `{2,6,18,22}`,
  `@4096` → column 4 / blocks `{0,4,16,20}`), but **influences nothing at other
  offsets** (`@16`, `@16384`, `@49152`, `@81920`, `@131056`). So live 16-byte
  weight units are interleaved with dead ones — the influence is **structured /
  sparse**, and the "first ~8704 bytes" live region contains dead units.

**Impulse axis (the (o, i) reach map).** Every impulse position fans out to
exactly **512** output cells, and the footprint **shifts** with the impulse
unit `u`:

| Reach-class `(row_shift, block_shift)` | Rows | Blocks | Impulse units tested |
|---|---|---|---|
| `(0, 0)` | `{0,4}` | `{0–7,16–23}` | 0, 1, 2, 3, 32, 128 |
| `(0, 8)` | `{0,4}` | `{8–15,24–31}` | 4, 5 |
| `(1, 0)` | `{1,5}` | `{0–7,16–23}` | 8, 24 |
| `(1, 8)` | `{1,5}` | `{8–15,24–31}` | 12, 255 |

The reach-class is predicted by the **bit model** `row_shift = (u>>3)&1`,
`block_shift = ((u>>2)&1)*8`; for **all 12 impulse positions** the footprint
equals the predicted shifted canonical set **exactly** (not just its shift). The four reach-classes are **pairwise disjoint** (512
cells each; 2048 distinct cells across the 12 samples). So **each reach-class
selects one of four disjoint 512-cell output blocks** (multiple input positions
share a reach-class), and the selection has a regular two-bit structure — direct
evidence of the (o, i) mapping that a full permutation recovery must reproduce.

`native-decoder-sweep-oracle.py` locks these invariants (coupled-ref == canonical
set; matrix-axis footprints ⊆ canonical; each impulse footprint is exactly the
**shifted canonical set** predicted by `reach_class(u)`; reach-classes pairwise
disjoint as defence-in-depth). It runs no GPU work.

## Controls and review

* **Self-test (probe):** 5 factor known answers, 4 canonical-membership answers,
  12 reach-class answers, 512 factor round-trips, 4 canonical-subset rejections,
  4 factor rejections, 5 upload-isolation answers, 3 sweep-cell answers, 2
  comparator-oracle controls.
* **Self-test (oracle):** synthetic 4-reach-class report accepted, identical pair
  accepted, 4 model cross-checks, 2 `_require_disjoint` controls, **10 corruption
  controls** rejected (non-canonical coupled-ref, matrix escape, wrong footprint
  size, impulse composition mismatch, reach-class mismatch, cross-class
  contamination, float unit offset, bool unit offset, missing coupled-ref, wrong
  mode) — each pinned to a `ValueError` rejection.
* **Regression:** the probe self-test re-runs the launch and weights self-tests.

## What this establishes — and what it does not

Established: the decoder's input->output **reach structure** is measurable and
reproducible through native Linux execution. Input positions fan out to disjoint
512-cell output blocks selected by two impulse bits; weight sub-blocks select
output columns/block-groups. Both are concrete constraints on the permutation.

Not established: which logical `(o, i)` pair each physical matrix byte serves (the
full permutation), the word-table semantics, magnitude linearity, exact
vendor-instruction equivalence, learned-weight binding, or any neural-rendering
quality. **No learned weights were used; no neural-rendered image was produced.**

## Next stage

1. Sweep the full impulse-partition x matrix-sub-block grid to completion and
   resolve `(o, i)` per physical byte against these named cells.
2. Map the stored decoder-weight candidate record through the recovered
   reach/permutation structure.
3. Capture Proton reference evaluations for numerical comparison.

Containment is unchanged: bubblewrap, no network, no home, pinned source
snapshots, hash-verified DLL read as data, 300 s bound, disabled core dumps.
Swap, driver persistence and memory erasure remain uncontrolled; proprietary
reuse/publication rights remain unresolved. No proprietary payload files are
deliberately created.

Evidence: `research/native-linux/runs/decoder-sweep-execute-qf4GSDp3`,
`-3rTSN1Na` (each 35 records including `base`; retained `oracle.json` /
`oracle-pair.json`), `decoder-sweep-skip-HvNnG4lx`. Source:
`vr/native-decoder-sweep-probe.py`, `vr/native-decoder-sweep-oracle.py`.
(Earlier runs `decoder-sweep-execute-CN4gEHzH` — failed, subset-invariant bug —
and `-6GRNDlk4`, `-UyYxCQDB`, `-19fphxCE`, `-9NasuatR`, `-VVfhVDcQ`, `-J17ryK0K`,
`decoder-sweep-skip-bbaqYxsp`, `-vbXs5Uri` are superseded by the review fixes:
evidence-first failure paths with base retention, scoped docstring, exact-set
lock, oracle hardening; evidence is re-run against the current source.)
