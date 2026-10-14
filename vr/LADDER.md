# Intermediate-ladder capture — 14/10/2026

## Result

**The evaluation can now be truncated at any recovered slot, and the model's
whole working pool read back at that point.** Consecutive cut points are then
diffable, which turns the 156-launch sequence into measured write-sets instead
of an inference from kernel names. Runs on the RTX 4070 Ti SUPER under
Proton/Wine — no changes to the model or Proton, and no Windows/DirectX in the
analysis path.

This is the enabling stage for the full-pipeline native replay
([NATIVE-VENDOR-COMPARE.md](NATIVE-VENDOR-COMPARE.md) proved the method on the
decoder alone): before any pre-decoder kernel can be reimplemented natively and
checked, its reference input and output tensors must exist. The ladder supplies
them.

Two measured write-sets, from the first cut points:

| Slots | Stage (NVAPI-ARGS.md) | Changed pool bytes | Ranges | Span (pool offsets) |
|---|---|---:|---:|---|
| 24–56 | transformer (`cc_split_swin_16h_*`) | ≈ 11.21 M | ≈ 72–83 k | 48,640 → 52,853,632 |
| 57–98 | ViT stage (`cc_vit_1d_*`) | ≈ 31.52 M | ≈ 211 k | 65,024 → 62,684,160 |

Frame-to-frame the **changed-byte** counts agree to within 0.1 % (worst case
0.099 %, s23–s56 frame 0→1; the range counts vary much more — 72–83 k —
because coalescing is sensitive to where the last differing byte falls), as
expected under the stationary-input protocol.
**The ViT stage is by far the heavier writer: about three times the transformer
stage's bytes, and reaching the very end of the 62.7 MB pool** (the span ends
are exclusive offsets). The first changed region in both cases sits at offset
48,640 / 65,024 — well past the harness's own allocations, consistent with
model-owned working memory.

The two write-set tables are retained as
`vr/logs/ladder/diff-s23-s56.json` and `vr/logs/ladder/diff-s56-s98.json`.

## Method

### Truncate at an arbitrary slot (`DLSS_NVAPI_CUT_AT=<n>`)

The existing cut ([DECODER-BOUNDARY.md](DECODER-BOUNDARY.md)) arms on the
decoder kernel's *name*. The index-based mode instead counts launches from the
clear kernel — recovered slot 0 of the 156-launch inventory — and arms at slot
*n*, for any *n* in **0..155** (the endpoints included: *n* = 0 reads the pool
just after `cc_cb_clear`, *n* = 155 just after the final `cg2r_copy_kernel`).
It keeps every fail-closed rule of the original:

- it never arms without a mapped clear kernel (no evaluation boundary to index
  from), a complete parameter dump, or a fully written sidecar line;
- the window withholds at most `155 - n` launches, so no failure can truncate
  past one evaluation;
- the name-anchored mode is untouched and still rejects a decoder name inside
  an open window as an anomaly. In index mode that rule is dropped (a decoder
  name inside the window is simply the case `n < 99`).

### Read the whole pool at the cut point

Mid-stream GPU buffers still cannot be read at call time, so the forwarder
withholds everything after the cut point and the harness reads at end-of-frame
as before. The new part is *what* it reads: the model's entire working pool
(62,684,160 bytes) instead of six 2 MiB windows, so every cut point is directly
comparable.

The pool is identified by the cut point's own bindings, not by size alone: two
per-eye pools of identical size exist (the `0x1CD00000` pair of
[TEMPORAL-HISTORY.md](TEMPORAL-HISTORY.md)), and the words of one evaluation
touch exactly one of them. If the words resolve into no pool, or into more than
one, the run fails closed rather than capturing a plausible but wrong buffer.

## Evidence

Runs on 14/10/2026 against the final source in this stage:

- `vr/logs/ladder/{s23,s56,s56b,s98}` — captures (manifest + 3 pool files +
  inputs/outputs + `boundary-va.txt` + `slot.txt`).
- `vr/logs/capture/nvapi-trace-ladder-*.log` — forwarder traces.
- `vr/logs/boundary/c` — the decoder-boundary **regression**: the name-anchored
  path re-run with the new forwarder. It reproduces the retained pre-change
  capture `a` **bit-for-bit across all 18 windows**
  (`decoder-boundary-report.py pair a c`), passes every pinned check including
  `weights-record-match` (record payload sha256 `e7fa32bd…c4fb8`), and its
  trace accounting is exactly the documented 3 armed / 168 drops / 2 releases.
- `python3 vr/ladder-report.py self-test` — **16 corruption controls**, each
  rejected with its pinned message, plus **3 pair-comparator controls**.
- `validate` passes for `s23` (slot 23), `s56` (slot 56) and `s98` (slot 98):
  cut accounting (396 drops at slot 23 = 132 × 3, etc.), pool digests, the
  slot-aware output canary and the input digests. The three captures share
  their input digests exactly, which is what makes the diffs comparable.
- `pair s56 s56b` — **bit-identical**, so the ladder path is deterministic.
- `diff s23 s56` and `diff s56 s98` — the write-set table above.

Evidence contains digests, counts and geometry only — never pool or tensor
bytes. The retained `.bin` pools are local (gitignored) and carry the
proprietary data-handling caveats of the earlier stages.

## Limitations

- A write-set is the **net** change between two cut points. Anything written and
  then overwritten within the same stage is invisible to it, and it does not
  separate reads from writes: a slot's read-set still has to come from its
  parameter block.
- All seven stage intervals are now measured and named (see
  [LADDER-NAMES.md](LADDER-NAMES.md)), including the encoder (0–23) and both
  post-decoder tails; slot 155 is a principled negative (its block binds no
  working-pool buffer). The only write-set that cannot be captured directly
  is slot 154's, which is empty in the pool by measurement — it writes the
  output texture only.
- Pairs exist at two points (`s56`/`s56b`, `s131`/`s131b`); the remaining
  single runs rest on the determinism established for the ladder path and
  repeated elsewhere (their digests also reproduce exactly across the binary
  changes made mid-stage), not on their own pair.
- The validator now pins pool identity and shared inputs before diffing:
  every frame of a capture must read the same pool, and `pair`/`diff` reject
  different pools or differing inputs. Without that, a diff across the two
  per-eye pools would report a confident and meaningless write-set. The
  retained evidence is consistent (`res_va=0x1cae0000` throughout, identical
  input digests across all four captures).
- The pool is 62,684,160 bytes of mostly model working memory; the ranges above
  are byte ranges, not tensor extents. Turning them into named tensors is the
  next analysis, not something this stage establishes.
- The two per-eye pools are disambiguated by the cut point's bindings; only eye0
  is captured, as in every earlier stage.
- **No native computation ran this stage.** This is reference capture for the
  replay work; it executes the proprietary kernel chain as before, and the
  rights and data-handling caveats are unchanged.

## Reproduction

Read `README.md` first; from the repository root:

```bash
python3 vr/ladder-report.py self-test
bash vr/run-feature-probe.sh ladder s23 23
bash vr/run-feature-probe.sh ladder s56 56
python3 vr/ladder-report.py validate vr/logs/ladder/s23 \
  --trace vr/logs/capture/nvapi-trace-ladder-s23.log --slot 23
python3 vr/ladder-report.py diff vr/logs/ladder/s23 vr/logs/ladder/s56
```

Builds (unchanged from [DECODER-BOUNDARY.md](DECODER-BOUNDARY.md)):

```bash
tools/zig-x86_64-linux-0.15.2/zig cc -target x86_64-windows-gnu -std=c11 -O2 -shared \
  -Wall -Wextra vr/capture/nvapi-trace.c -o vr/capture/nvapi64-trace.dll
tools/zig-x86_64-linux-0.15.2/zig cc -target x86_64-windows-gnu -std=c11 -Wall -Wextra \
  -Werror -O2 vr/feature-probe.c -ld3d12 -ldxguid -ld3dcompiler_47 -o vr/probe/feature-probe.exe
```

## Next

The write-sets are named in [LADDER-NAMES.md](LADDER-NAMES.md)
(`vr/ladder-names.py`): every changed byte of every interval is attributed to
the `(slot, kernel, word)` binding whose buffer contains it, including the
decoder interval's exact match to the recovered decoder contract. The tail
is closed: slot 154 writes only the output texture (zero pool change) and
slot 155 binds no working-pool buffer (principled negative). What remains
for the full-pipeline replay is the reverse direction — taking each named
tensor as a replay input/output pair and reimplementing the kernel that
produced it, starting from the transformer `qkv→proj` pairs now pinned at
their pool offsets.
