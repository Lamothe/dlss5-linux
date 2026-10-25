# Reference input plane integrity and packing

## Scope

`native-input-layout.py` is CPU-only. It validates the three documented raw reference planes against their capture manifest, then exercises their declared tightly packed row layout. It can strip row padding from a caller-supplied contiguous byte buffer without changing active sample bytes.

This is **not** a DLSS preprocessor, colour/depth/motion semantic decoder, coordinate-system definition, GPU upload path, runtime binding, or native inference. The capture manifest's FNV-1a-64 is a local corruption check, not a cryptographic authenticity claim. No proprietary payload or captured tensor is included by this tool.

## Documented plane contract

| File | Components | Bytes/component | Layout used by validator |
|---|---:|---:|---|
| `input-colour.bin` | 4 | 2 (FP16-sized) | interleaved four-component samples |
| `input-depth.bin` | 1 | 4 (F32-sized) | scalar samples |
| `input-motion.bin` | 2 | 2 (FP16-sized) | interleaved two-component samples |

For width `W`, height `H`, components `C`, and component size `S`, each active row is `W×C×S` bytes. `pack_rows` requires the source to contain exactly `row_pitch×H` contiguous bytes with `row_pitch >= W×C×S`; it copies only each active row and discards padding. Tight rows are returned unchanged as bytes. Unsupported filenames, non-positive/non-integer dimensions, short pitch and wrong total input size are rejected.

The sample formats above are the capture protocol's documented storage widths; this helper does not convert endianness, decode values, or assert that the byte ordering/meaning matches the model's expectations.

## Manifest/file validation

`validate_capture` reads each of the three files once, verifies expected dimensions and exact tight byte counts, and computes FNV-1a-64 on those same bytes. It requires exactly one well-formed manifest entry per required plane and rejects malformed or duplicate required entries, wrong byte counts, or digest mismatches. Other manifest record types (including the protocol header and output frames) are ignored. It then routes each payload through the tight-row packer and checks its length and byte identity.

The optional local check is:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 vr/native-input-layout.py vr/logs/reference/run-a
PYTHONDONTWRITEBYTECODE=1 python3 vr/native-input-layout.py vr/logs/reference/run-b
```

Current ignored local run-a/run-b files both pass at 720×720: colour 4,147,200 bytes; depth and motion 2,073,600 bytes each. The captures and their manifests are not committed. CPU synthetic controls are in `vr/tests/native-input-layout-test.py` and require no GPU or proprietary runtime.

## Migration gap

The result establishes local capture-file integrity and a storage-layout adapter primitive only. It does not explain how colour, depth, motion or temporal state are preprocessed, converted, scaled, normalised or bound to the original graph. No connection from these raw planes to the captured decoder inputs has been demonstrated. The native live-input-to-image evaluator remains unimplemented.