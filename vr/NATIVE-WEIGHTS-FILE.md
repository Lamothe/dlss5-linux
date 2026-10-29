# NATIVE-WEIGHTS-FILE — the data-free weights interface

Status: complete (29/10/2026, delivery phase "scope pivot", item 1).
Tests: `vr/test-native-weights-file.sh` (72 checks at -O2 and -O0, plus the
full-network byte-identity round-trip and the rejection paths through the
demo). Resume path: `vr/validate-weights.sh <file>` — one command.

## What this is

A weight-file loader and compatibility-verification harness for the
evaluator's `EvalWeights` tensor layer (`vr/native-weights-file.hpp`,
`vr/weights-verify.cpp`, `vr/validate-weights.sh`). It accepts weight
files **when one is ever legitimately obtained**; until then it is
exercised entirely with synthetic fixtures.

**Data-free by construction.** The interface ships no weight content,
touches no NVIDIA assets, includes no proprietary payloads, captures or
binaries, and performs no runtime probes. Nothing in it knows any
proprietary format's bytes: it defines and verifies a documented
container, and the verification logic consumes *tensor records*
(metadata + content), not file encodings.

Every artefact, tool and label in this area carries the honesty label;
this interface makes **no DLSS5 quality claim** — it proves that a
weight file can be accepted, verified and evaluated through the native
pipeline end to end.

## The NWF1 container

Little-endian throughout. Full byte layout in
`vr/native-weights-file.hpp`'s header comment. Summary:

- header (40 bytes): magic `"NWF1\x0d\x0a\x1a\x0a"`, version (1),
  flags (bit 0 = payload hash present), record count, reserved,
  payload bytes, payload hash (FNV-1a 64 over records + payload);
- `count` tensor records (96 bytes each): name[48], kind
  (1 = e4m3 u8 codes, 2 = binary16-grid float32, 3 = free float32,
  4 = int4 packed u8 codes), block, layer, which, rows, cols, count,
  payload offset (8-byte aligned), payload length;
- the payload blob.

A `count != rows*cols` on a matrix record, an unaligned or out-of-range
payload slice, a duplicate record, a hash mismatch or an unknown kind
rejects the file with the record named. Content verification rejects
non-finite float32 and off-binary16-grid values in binary16-grid
records; the e4m3 and int4 code spaces are dense (every byte valid).

## The provider contract (the subtle part)

The `EvalWeights` fetch contract has two mechanisms the loader must
mirror exactly:

1. **Wildcard fetches** (`count == -1`) return the record whatever its
   length; the caller reads it at the plan's known table length
   (`native-demo-support.hpp`'s `wildcard_length`).
2. **Length discovery** (`record_length_f16/f32` in `native-eval.hpp`)
   probes exact counts 0, 1, 2, ... and takes the first count the
   provider answers; a provider that answers *any* count breaks it.

The file-backed provider therefore serves exact-count fetches **only**
at the record's true length and answers `count == -1` as the wildcard.
Count mismatches reject *silently* (that is normal length discovery);
identity misses (a name/which the file does not have) are recorded and
named — a real incompatibility. Shape mismatches and every other
mismatch class are named by the verification harness below, which runs
before any evaluation.

The recorder (`Recorder` in the header) captures requests and bytes
through the same contract. Its `length_for` hook supplies the plan's
table lengths so wildcard probes capture at the true length; an unknown
probe length records an error and the dump refuses to write rather than
record a wrong-length record. The extension logic (a later longer
request at the same identity keeps the first bytes) mirrors
`DemoStore`'s documented contract and is pinned by tests.

## The compatibility-verification harness

`weights-verify <reference.nwf> <candidate.nwf>`: the reference file
carries the plan's **request manifest** (generated from the plan itself
by `demo-native --dump-weights` — there is no static table to drift);
the candidate is accepted or rejected with every problem named:
missing tensors, unexpected tensors, shape mismatches, structural
corruption, content problems. Exit 0 = accepted.

## The reference-validation resume path — one documented step

    vr/validate-weights.sh <weights.nwf>

regenerates the request manifest from the plan, verifies the file
against it, then runs the reference validation: the full network
through the file-backed provider on the CPU and on the GPU projection
path, requiring **byte-identical artefacts between the two** (the G2
property with these weights), and checks the mandatory label. Exit 0 =
accepted and reference-validated; the evidence directory is printed.

## Accepting a source-format file

The verification consumes tensor records, not file encodings. When a
weight file is legitimately obtained in any source format, one decoder
maps its tensors to `TensorRecord`s + payload bytes (the format's shapes
and dtypes become the records' metadata); everything downstream —
structural checks, content checks, the manifest comparison, the
reference validation — is unchanged. The built-in NWF1 reader is the
reference decoder. No decoder for any proprietary format is included or
implied.

## Scope note

The loader serves the evaluator's matrices and scalars (e4m3,
binary16-grid, float32, int4). The structured site records
(`swin`/`int4` sites) are absent in a file — that is the evaluator's
documented "run the e4m3 dataflow" case, not an error. The test family
compares the DUMP run's artefacts with the FILE run's (both through the
recorder/file providers, the same dataflow and bytes — an exact
identity); the plain synthetic run is kept as a smoke test.

## Honest limits

- The container is *our* documented format. "Accepts DLSS5-format
  weight files" means the plumbing is format-agnostic and ready for a
  legitimate decoder (above); it does not mean any proprietary format
  knowledge is present or claimed.
- The reference validation proves shape/metadata compatibility and
  CPU-vs-GPU byte identity. It proves nothing about the *quality* of a
  file's contents — quality is the separate demonstration-transform
  work ("not DLSS5" everywhere).
- `validate-weights.sh` runs the GPU reference only where the Vulkan
  toolchain exists; elsewhere it reports the skip rather than failing.
