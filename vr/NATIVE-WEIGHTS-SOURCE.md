# NATIVE-WEIGHTS-SOURCE — the source-tensor-dictionary → NWF1 converter

Status: complete (02/11/2026, the L2 weight-ingestion boundary).
Tests: `vr/test-native-weights-source.sh` (169 checks at -O2 and -O0, plus a
sanitised build when the toolchain can link a sanitizer runtime).
Code: `vr/native-weights-source.hpp`; tests `vr/tests/native-weights-source-test.cpp`.

## What this is

The declared weight-ingestion boundary: a source-tensor-dictionary → NWF1
converter (`vr/native-weights-source.hpp`). It accepts an in-memory
dictionary `name → tensor` as supplied by **any legitimate decoder** of a
source-format weight file, and produces the NWF1 container
(`vr/native-weights-file.hpp`) the native evaluator already consumes. Until
this boundary existed, a legitimately obtained DLSS5 weight file could not
have been loaded at all; now the ingestion path is complete and verified
against synthetic fixtures.

**Data-free by construction.** The converter ships no weight content,
touches no NVIDIA assets, performs no runtime or capture probes, and
contains no proprietary payload, capture or binary. Its tests use synthetic
fixtures only.

**This establishes READINESS only.** It is not a vendor parity claim and
not a DLSS5 quality claim. Synthetic tests prove the plumbing — the
transforms, the identity discipline and the byte-identical round-trip — and
say nothing about how real weights would behave.

## Rights statement

The converter **accepts a dictionary from a legitimate decoder**; we
neither ship nor extract weights. Nothing in this repository extracts,
derives or contains NVIDIA weight data: the converter consumes whatever
tensors a caller's own legitimate decoder produces, in memory, and the
tests generate every byte they consume. No proprietary payload, weight,
capture or binary is committed or required. This work establishes
readiness to ingest legitimately obtained weights — never vendor parity,
never DLSS5 quality.

## The pinned MIT reference

The source packing is re-derived from the MIT-licensed reference
**mauri870/dlssnr-pytorch**, pinned at commit
**`3beef7935a8ef976b47ab78b906e75600c52d2de`**, files `dlssnr/weights.py`,
`dlssnr/layers/split_swin.py`, `dlssnr/layers/vit.py`,
`dlssnr/layers/decoder.py`, `dlssnr/model.py`. A local copy is kept outside
Git at `.tools/dlssnr-ref/` (reference code, never shipped). Provenance
verified 02/11/2026: all five named files byte-identical to the pinned
commit fetched from `raw.githubusercontent.com`.

The reference's decoding conventions (from `dlssnr/weights.py`): every
tensor is keyed `block{B}.layer{L}.{name}` and holds the exact bytes a
typed accessor reads — `e4m3` views raw u8 codes, `f16` views raw binary16
halves (`raw.view(torch.float16).to(torch.float32)`), `f32` views raw
float32. The converter's dictionary semantics mirror this exactly.

### MIT notice (retained with attribution)

```
MIT License

Copyright (c) 2026 Mauri de Souza Meneguzzo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Attribution: the packing definitions (`_slice_sources`,
`_grouped_ffwd_matrices`, `unpack_bias`) and the `Weights` keying semantics
are re-derived from `dlssnr-pytorch` by Mauri de Souza Meneguzzo, MIT
licensed, used here with the notice retained. The converter and its tests
are independent C++ implementations of the recorded arithmetic; the test
oracles are direct arithmetic transcriptions, the hand literals the
independent pins.

## Source dictionary semantics

`weightssource::SourceStore` is the in-memory dictionary. Each tensor is
the raw byte payload keyed `block{B}.layer{L}.{name}`; the element encoding
is fixed per accessor (the pinned convention):

| Accessor | Payload bytes | Served to the evaluator as |
|---|---|---|
| `e4m3` | raw u8 e4m3 codes (1 byte/element) | e4m3 u8 codes (NWF1 kind 1) |
| `f16` | raw binary16 halves (2 bytes/element) | binary16-grid float32 (NWF1 kind 2) |
| `f32` | raw float32 (4 bytes/element) | float32 (NWF1 kind 3) |
| `ffwd` (split feed-forward) | the two packed `ffwd_a`/`ffwd_b` u8 tensors (512×512 codes each) | the three grouped e4m3 matrices via the scatter below (NWF1 kind 4 records at their `which` 0/1/2 identities) |

Wildcard tables (`residual_scale`, `attn_residual_scale`, `scalars_b`,
`upsample_gain` — the pinned `reshape(-1)` records) are served at their
plan table lengths: a short table is zero-padded to the table length, the
exact length answers, an over-long table is named and rejected, and
`count == -1` and exact-count discovery probes answer per the evaluator's
`record_length_*` contract (see `NATIVE-WEIGHTS-FILE.md`'s provider
contract).

## The generated identity set — never a hand-copied table

The record set and its exact identities (name, kind, block, layer, which,
rows, cols, count) are **generated from the plan**: `convert_source` runs
the network with the recorder (`native-weights-file.hpp`) over the source
provider bound underneath. The plan requests what it needs at its plan
identity, the recorder captures each request, and the container records
fall out exactly. The dictionary must match that generated set exactly —
every rejection names the tensor. The identity validator refuses exact
duplicate identities and one source key requested under two element
encodings; the serialiser refuses a capture set holding one identity twice;
the container reader is the downstream backstop.

## Packing transform 1: the split feed-forward grouped scatter

Source: the pinned `split_swin.py` `_slice_sources` + `_grouped_ffwd_matrices`.
The two packed halves (`ffwd_a`, `ffwd_b`, each 512×512 codes) are NOT two
matrices: they are a byte permutation of the combined 524,288-code record
holding, bit-scattered, for each of the eight groups, `A_g [64, 512]`
(first half) then `Q0_g [256, 64]` and `Q2_g [64, 256]` (second half).
Undoing the permutation and regrouping yields the evaluator's three
grouped matrices in natural row-major `[out, in]` orientation
(`native-split-swin.hpp`'s `SplitFfwdWeight`):

| Output record | Shape | Row identity | Contents |
|---|---|---|---|
| `expand_in` | [512, 512] | row `g*64 + j` | `A_g` row `j` (512 inputs) |
| `expand_hidden` | [2048, 64] | row `g*256 + j` | `Q0_g` row `j` |
| `contract` | [512, 256] | row `g*64 + n` | `Q2_g` row `n` |

Each of the 524,288 output codes is a distinct byte of the packed record,
so the scatter is a **permutation** of the input codes — a property the
tests verify exhaustively (every code claimed exactly once, in both
directions), not by spot checks.

### The derivation record (hand tracing)

A packed region is a sequence of 512-byte slices, 16 output rows by 32
inputs, ordered input-group-major (`row_group = slice % 32`,
`col_group = slice / 32`); inside a slice, `offset = slice*512 + lane*16 +
half*8 + byte`, `row = row_group*16 + half*8 + lane/4`,
`col = col_group*32 + 8*(byte/2) + 2*(lane%4) + byte%2` (the pinned
`_slice_sources`). The record is `record[slice_sources(i,j)] = ffwd_a[i,j]`
(and `record[262144 + ...] = ffwd_b[i,j]`); the scatter maps then index the
record by group:

```
expand_in(g,j,k)     = bit(k,0→0) (k,1→4) (k,2→5) (k,3→1) (k,4→2) (k,5→14)
                       (k,6→15) (k,7→16) (k,8→17) (j,0→3) (j,1→6) (j,2→7)
                       (j,3→8) (j,4→9) (j,5→10) (g,0→11) (g,1→12) (g,2→13)
expand_hidden(g,j,r) = 262144 + g*16384 + (r,0→1) (r,1→0) (r,2→4) (r,3→5)
                       (r,4→2) (r,5→13) (j,0→6) (j,1→3) (j,2→9) (j,3→7)
                       (j,4→8) (j,5→10) (j,6→11) (j,7→12)
contract(g,n,h)      = 393216 + g*16384 + (h,0→0) (h,1→1) (h,2→2) (h,3→4)
                       (h,4→5) (h,5→11) (h,6→12) (h,7→13) (n,0→6) (n,1→7)
                       (n,2→8) (n,3→3) (n,4→9) (n,5→10)
```

(`(i→b)` means "bit `i` of the index contributes to offset bit `b`", the
pinned `_bit` transcription.) Worked anchors, traced end to end:

- `expand_in[1][0]` = `a[g=0,j=1,k=0]` = `record[8]` = `ffwd_a[8*512+0]` =
  `ffwd_a[4096]` — row bit 0 puts `j=1` at offset 8, and `record[8]` is
  pack offset 8 = slice 0, half 1, lane 0, byte 0 = `first[8, 0]`.
- `expand_in[0][64]` = `a[g=0,j=0,k=64]` = `record[32768]` =
  `ffwd_a[64]` — k bit 6 puts `k=64` at offset 32768 = slice 64 =
  `first[0, 64]`.
- `expand_hidden[0][1]` = `record[262146]` = `ffwd_b[8]`; `contract[0][1]` =
  `record[393217]` = `ffwd_b[257]`.

The test battery pins the maps with hand-derived anchors, an independent
oracle restatement compared full-space (all 262,144 + 262,144 outputs per
matrix family), coverage (a bijection), and mutation-verified
discriminators: a seeded wrong permutation, a wrong fragment order and a
dropped group each fail **exactly** at its designed check.

## Packing transform 2: the `attn_pos_bias` fragment order

Source: the pinned `split_swin.py` `unpack_bias` / `swin.py`
`_deswizzle_bias`. The source's `attn_pos_bias` is the
`[heads, 4096]` accumulator-fragment-order binary16 table exactly as the
evaluator fetches it. The converter **transfers the fragment-order table
faithfully** (binary16 → binary16-grid float32 decode, no reordering): the
deswizzle to the `[64, 64]` `[query, key]` bias is the evaluator's
downstream step (`swin_deswizzle_index`), carried through the exponent ramp
by the swin rows (`swin_exp_bias`) and used raw by the split family's
no-ramp form (`swin_deswizzle_bias`). The fragment-order index map is
exported (`attn_pos_bias_fragment_index`) so the tests pin **both** consumer
forms against hand-derived answers and catch a wrong fragment order:

```
lane  = 4*((i%16)%8) + ((j%16)%8)/2
slot  = (j%2) | (((i%16)/8)<<1) | (((j%16)/8)<<2)
index = (4*(i/16) + (j/16))*256 + lane*8 + slot
```

(`i` = query row, `j` = key column; the table is per head.) Anchors:
`(0,0)→0`, `(0,2)→8`, `(1,0)→32`, `(8,0)→2`, `(15,15)→255`, `(16,0)→1024`.

## The named rejection matrix

Every rejection names the tensor; each class is tested:

| Class | Rejection |
|---|---|
| Wrong packed shapes | `ffwd_a`/`ffwd_b` named with `512*512` required |
| Truncated fragments | `attn_pos_bias` named with both counts |
| Duplicate identities | `duplicate source tensor`/`duplicate record identity`, tensor named (dictionary `set`, serialiser, container reader, identity validator) |
| Count/shape mismatch | e4m3 `codes`, f16 `binary16 entries`, f32 `float32 entries` — both counts named; odd binary16 byte count; non-float32 byte count |
| Non-finite values | binary16/float32 non-finite at its index; e4m3 NaN codes 0x7f/0xff at its index |
| Unknown names | `unknown source tensor` (a dictionary entry the plan never requests); `missing source tensor` (a fetch the dictionary cannot serve) |
| Two element encodings | one source key requested as two kinds, both kinds named |

## Disclosed strictness

Two disciplines are stricter than "any bytes a decoder might produce", and
both are deliberate and disclosed:

1. **Finite-code discipline.** e4m3fn's two NaN codes (0x7f/0xff) and
   non-finite binary16/float32 values are rejected by name and index. The
   code spaces are otherwise dense (every byte a valid code). A legitimate
   decoder producing non-finite weight content is refused — a real
   incompatibility worth naming, not silent garbage into the evaluator.
2. **Wildcard pad convention.** Short wildcard tables zero-pad to the plan
   table length (the pinned constructors slice these records); the length
   discovery probes keep their exact semantics.

## Verification

`vr/test-native-weights-source.sh` (strict `-Wall -Wextra -Werror`
builds, `-ffp-contract=off` pinned):

- hand-derived known answers for both packing transforms (anchor tables +
  worked chains above), including the independent oracle restatements
  compared full-space and the non-vacuity check that the oracle itself
  passes the battery;
- **full permutation coverage** of the scatter (every one of the 524,288
  codes claimed exactly once, both directions);
- the fragment map pinned in both consumer forms (swin ramp / split
  no-ramp);
- the **whole-network round-trip**: synthetic dictionary → converter →
  NWF1 → evaluator, artefact byte-identical (`cmp`-identical values) to
  the direct synthetic path, with conversion determinism (two conversions
  byte-equal) and the container record set exactly the generated identity
  set;
- mutation-verified discriminators (seeded wrong permutation, wrong
  fragment order, dropped group — each fails exactly at its designed
  check);
- the named negative matrix above, at the fetch boundary and end-to-end
  through `convert_source`.

Runs at -O2 and -O0; a sanitised build is attempted and skipped only when
the toolchain cannot link a sanitizer runtime (recorded as an environment
gap in the runner output — never faked). Synthetic fixtures only: no
weights, no proprietary payloads, no vendor correspondence is claimed.
