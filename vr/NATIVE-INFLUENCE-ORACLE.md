# Decoder influence-footprint constraint oracle — 12/10/2026

`native-decoder-influence-oracle.py` locks the measured influence footprint of
the decoder kernel as a **regression anchor** for the upcoming output-channel
naming stage. It runs no GPU work: it reads the retained evidence written by
`native-decoder-weights-probe.py` and asserts each footprint matches a single
canonical pattern.

## Why

Gate stage one ([NATIVE-DECODER-WEIGHTS.md](NATIVE-DECODER-WEIGHTS.md)) showed
the matrix region couples to the output only through the matching input
partition. The next stage must name the influenced output positions — i.e.
recover which logical `(output channel, input channel)` each matrix byte serves.
Any such claim must reproduce these exact anchors; the oracle makes that check
mechanical instead of a visual diff.

## The canonical pattern

Enforced across all **8 experiment groups** (2 word-table variants × 4 impulse
partitions), for both the `coupled` (+1.0) and `negative` (−1.0) footprints:

| Field | Canonical value |
|---|---|
| `changed_count` | 4096 |
| `changed_units16` | 512 |
| `changed_first` / `changed_last` | 0 / 40827 |
| `sample_offsets` | `[0, 1, 2, 3, 8, 9, 10, 11]` |
| `sample_values` (coupled / negative) | `+8.0` ×8 / `−8.0` ×8 |
| `changed` / `truncated` | `True` / `False` |

Group flags `base_zero_in_extent`, `coupled_differs`, `decoupled_identical`
and `negative_differs` must all be `True`. `decoupled_checked` is
**partition-relative**: for impulse partition `p` it lists the *other*
partitions `q ≠ p` as `decoupled-qN`.

Top-level claims are pinned to the contract this evidence supports:
`mode = execute`, `kernel = cc_dec_input_upsample_1024_512_fp8`,
`learned_weights_used = False`, `matrix_permutation_recovered = False`,
`native_image_inference = False`. Evidence that *over*-claims is rejected.

## Structural fact pinned by the anchors

**The influence footprint is identical for every impulse partition and both
word-table variants** — same span `[0, 40827]`, same 512 units, same 8 changed
bytes per unit at offsets `[0,1,2,3,8,9,10,11]`, same `±8.0` magnitudes. The
response is symmetric in the matrix code's sign and independent of the
word-table pattern and of which input partition carries the impulse. This is
the invariant a recovered addressing model must explain.

## Verified

- **Self-test (CPU):** 39 mutation corruptions rejected, baseline accepted.
  Mutations cover all four group flags, the partition-relative
  `decoupled_checked` rule (wrong set, wrong length, mixed element types),
  every influence field plus sign symmetry, the pinned kernel, execute mode,
  the no-overclaim flags, key membership/duplicates, exact-`int` partition
  typing, and clean rejection of missing keys and non-object containers.
- **Both agreed evidence runs** (`decoder-weights-execute-pTSuyNDt`,
  `-aaHtBUZE`) pass all 16 anchors each.
- **Independent review — two rounds.** Round one found 2 must-fix (raw
  `KeyError`/`TypeError` on malformed evidence; self-test coverage gap with an
  overclaiming docstring) and 5 advisories; all addressed. Round two found 3
  residual must-fix (raw indexing of the influence objects, a `TypeError` on
  mixed-type `decoupled_checked`, residual docstring overclaim); all fixed and
  re-verified. Malformed JSON and missing fields now fail with a diagnostic
  and exit 1 rather than a traceback.

## What this does NOT establish

This is a **constraint lock, not a permutation recovery**. It does not run the
GPU, launch a kernel, recover the `(o, i)` matrix permutation, bind stored
learned weights, or produce a neural-rendered image. The 4096 changed bytes
spanning `[0, 40827]` in 512 sixteen-byte units remain raw material for the
permutation; mapping output positions onto output channels and sweeping
impulse/code positions at finer granularity is still the next stage.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 vr/native-decoder-influence-oracle.py --self-test
PYTHONDONTWRITEBYTECODE=1 python3 vr/native-decoder-influence-oracle.py \
    research/native-linux/runs/decoder-weights-execute-pTSuyNDt \
    research/native-linux/runs/decoder-weights-execute-aaHtBUZE
```

Proprietary reuse/publication rights and documented data-exposure limits
(swap, driver persistence) are unchanged and unresolved. Nothing pushed at the
time of writing.
