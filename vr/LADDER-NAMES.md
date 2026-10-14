# Named ladder write-sets — 15/10/2026

## Result

**The ladder write-sets are now named tensors.** Every changed byte between
two cut points is attributed to the `(slot, kernel, word)` parameter-block
binding whose buffer contains it, so the byte ranges of
[LADDER.md](LADDER.md) become concrete buffers of the recovered 156-launch
inventory. Produced by `vr/ladder-names.py` from the ladder captures and
their `DLSS_NVAPI_ARGS=2` traces, on the RTX 4070 Ti SUPER under Proton/Wine —
the analysis itself is offline; only the captures were taken through the GPU.

Seven intervals cover the whole evaluation. Counts are segment counts of the
frame-0 report (`names-*.txt`, full lists retained); every changed byte of
every interval is attributed — **125,618,796 bytes at frame 0**
(376,939,340 across all three frames):

| Interval | Slots | Changed (frame 0) | Named content (frame 0 segments) |
|---|---|---:|---|
| s0→s23 | 1–23 encoder | 48,513,382 | 35 segments: a 20.4 MB `pre_block_swin_1h` w27 buffer + the `1h→2h→4h→8h` pyramid (`*_chained_fp8`, 2.5–5.1 MB) |
| s23→s56 | 24–56 transformer | 11,200,631 | 66 segments: 16 `qkv`, 14 `proj`, 14 `ffwd`, 14 `ffwd_proj`, 3 `proj_pool`, 4 `*_inpview`, 1 `final_head`; extents ≈ 344 KB |
| s56→s98 | 57–98 ViT | 31,519,899 | 145 segments: 48 `cc_vit_1d_qkv` (28 × 192 KiB, 8 × 1,179,648 B, 16 × 128 B), 32 `ffn_contract`, 28 `projection`, 16 `attention`, 14 `ffn_expand` |
| s98→s99 | 99 decoder | 536,208 | 4 segments: the exact decoder contract (below) |
| s99→s131 | 100–131 transformer-2 | 10,920,551 | 63 segments: the same `qkv`/`proj`/`ffwd`/`ffwd_proj` pattern |
| s131→s153 | 132–153 decoder pyramid | 22,928,125 | 33 segments: `8h→4h→2h→1h` chained + upsample buffers (2.5–5.1 MB) |
| s153→s154 | 154 final blend | **0** | writes only outside the working pool (the output texture) |

### The decoder interval is the cross-validation

`names-s98-s99.txt` names four segments that match the independently
recovered decoder contract ([NATIVE-DECODER-IO.md](NATIVE-DECODER-IO.md),
[NVAPI-BINDINGS.md](NVAPI-BINDINGS.md)):

| Segment (pool offset) | Changed | Extent | Name |
|---|---:|---:|---|
| +20610048 | 340,588 | 344,064 | `w2` = **out0** |
| +35994624 | 195,428 | 196,608 | `w6` = **partial sums** (192 KiB) |
| +97280 | 96 | — | `w4` = **counter**, first bank |
| +97792 | 96 | — | labelled `w5`, but see below |

The `w5` label is a nearest-base label, **not** a contract role: the second
96-byte counter bank sits exactly 512 bytes past `w4` — the documented
counter padding (`decoder-counter-pad`, NVAPI-BINDINGS.md) — and the contract
marks word 5 as *"output 2 … never referenced by this variant's decoded
instructions"*. The bytes are written through `w4` at a +512 offset. This is
also a live case of the attribution model's aliasing caveat: a buffer
spanning a later binding base is split at that base.

## Attribution model

Stated exactly; the tool fails closed wherever it cannot hold.

- The writer of a changed byte is a slot in the interval `(cut_a, cut_b]`:
  earlier slots run in both captures and cancel (runs are deterministic),
  later slots never run in the later capture.
- Writes reach bytes through the writing launch's own parameter words at
  non-negative buffer offsets, so the writer's binding base is at or below
  the byte.
- The interval's own binding bases tile the pool into buffer extents.
  Bindings of *other* slots can sit inside those extents as aliased
  sub-views — measured: the s23→s56 base pair `20954112/21298176` spans
  exactly 344,064 bytes, the out0 tensor shape, with slots 17/18/133/134
  bound at interior offsets of it (recorded as `alias` lines in the reports)
  — so segments are cut at the interval's bases only.
- Changed bytes in `[b_i, b_i+1)` name the bindings at `b_i`. Aliased
  `(slot, word)` pairs at one base are reported as a candidate set: the
  parameter block does not separate writer from reader.
- **Fail closed:** a changed byte below every interval base is rejected
  rather than mis-named.

## Cross-validation

`vr/ladder-names.py` and `vr/ladder-report.py diff` compute changed bytes
with different algorithms (big-int XOR + pattern matching vs bytewise scan)
and agree exactly on every interval and frame. The two committed stage-14
diffs (`diff-s23-s56.json`, `diff-s56-s98.json`) also reproduce bit-for-bit
from the re-taken captures.

## Self-test

`python3 vr/ladder-names.py self-test` — **14 corruption controls**, each
tripping exactly the check named in its control string: frame count, pool
digest, pool bytes, pool address varies, pool file missing, pmem length,
word shape, no pool bindings, no interval bindings, clear at slot 0, launches
per eval, cross-run binding mismatch, change below first binding, bindings
vary across evals. The fixture is a two-eval, four-launch arena whose word
VAs satisfy the real address-shape rule (`0x01000000 ≤ v < 2^32`,
512-aligned); the eval-stability control mutates one trace only, so both
halves of the cross-eval check are exercised.

## Companion fixes (same stage)

Three capture-path defects surfaced while re-taking the evidence and are
fixed in the same commit:

1. **Forwarder slot-0 re-arm** (`nvapi-trace.c`): a cut at slot 0 is the
   clear kernel itself, and the clear that closed the previous window never
   armed the next one — every evaluation after the first ran uncut and the
   capture died on missing sidecar lines. The release now falls through to
   the boundary check; name-anchored (decoder) mode is unaffected because a
   clear is never the decoder boundary.
2. **Pool word resolver** (`boundary-probe.h`): scalar parameter words can
   coincide with a device VA range — the s154 cut point's `0x3ab60b61`
   (an f32 ≈ 0.00135) landed inside the second per-eye pool and the
   single-pool rule failed closed on a false ambiguity. Only address-shaped
   words are now considered, exactly the recovered word-typing rule
   ([NVAPI-BINDINGS.md](NVAPI-BINDINGS.md)).
3. **Output-canary slot awareness** (`boundary-probe.h`): the NaN-sentinel
   expectation only holds for cuts before the output writer (measured: s153
   leaves the sentinel, s154 writes it). The state is now recorded as
   evidence in ladder mode and only *rejected* on the decoder-boundary path,
   whose cut is before the writer.

## Limitations

- Attribution is by binding base; where several `(slot, word)` pairs alias
  one base the writer is one of the listed candidates and is not singled
  out. A buffer spanning a later base is split there (the counter case
  above). Extents come from the observed change spans and segment bounds,
  not from decoded tensor shapes.
- Slot 155 (`cg2r_copy_kernel`) is a principled negative: its parameter
  block binds no working-pool buffer at all, so the ladder pool resolver
  correctly refuses it ("resolve into no single working pool"). Its writes
  target the output texture only.
- The three-frame captures exercise one eye at one resolution with
  stationary inputs; slot-154's out-of-pool write (the output texture) is
  observed through the canary, not through pool diffs.
- Evidence under `vr/logs/` remains local; committed artefacts are digests,
  offsets, counts and names only. Proprietary-reuse caveats from the earlier
  stages are unchanged.

## Reproduction

```bash
python3 vr/ladder-names.py self-test
python3 vr/ladder-names.py names vr/logs/ladder/s23 vr/logs/ladder/s56 \
  vr/logs/capture/nvapi-trace-ladder-s23.log \
  vr/logs/capture/nvapi-trace-ladder-s56.log
```

Builds (unchanged from [LADDER.md](LADDER.md) reproduction):

```bash
tools/zig-x86_64-linux-0.15.2/zig cc -target x86_64-windows-gnu -std=c11 -O2 \
  -shared -Wall -Wextra vr/capture/nvapi-trace.c -o vr/capture/nvapi64-trace.dll
tools/zig-x86_64-linux-0.15.2/zig cc -target x86_64-windows-gnu -std=c11 \
  -Wall -Wextra -Werror -O2 vr/feature-probe.c -ld3d12 -ldxguid \
  -ld3dcompiler_47 -o vr/probe/feature-probe.exe
```
