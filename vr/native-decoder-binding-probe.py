#!/usr/bin/env python3
"""Bind the stored decoder-weight candidate record to the native SM89 decoder
kernel through native Linux CUDA; NOT neural rendering or image inference.

Launches the pinned SM89 `cc_dec_input_upsample_1024_512_fp8` on the RTX
4070 Ti SUPER through native Linux CUDA (no Windows code, DirectX or Proton)
with the stored record `block39.layer0.layer` uploaded BYTE-IDENTICALLY as the
weights region, and checks the kernel's single-product responses against the
product prediction of the validated (o, i) -> matrix-byte closed form of
[NATIVE-DECODER-EXTENSION.md](NATIVE-DECODER-EXTENSION.md), applied to the
stored weight bytes.

This stage USES stored learned weight bytes (bounded model-weight computation,
following the [NATIVE-FP8.md](NATIVE-FP8.md) stored-record precedent). It is
NOT a neural-rendered image and does NOT establish the record's exact position
in the Windows runtime's graph: numerical compatibility makes the record a
bound candidate, not a proven graph node.

Record layout under test (data-side evidence, see `binding_metadata`):

  payload[0, 524288)      E4M3 matrix (the kernel's 524,288-byte coefficient
                          region; E4M3-NaN-free like the FP8 projection matrix)
  payload[524288, 525312) 512 FP16 values (finite; the kernel's word region at
                          +0x80000 reads here under byte-identical upload)

Product semantics under test: with a single nonzero FP8 input byte x at
(p, u, l), every influenced output byte o receives exactly one product
x * w where w is the stored matrix byte at partition-local m(u*16+l, o)
(NATIVE-DECODER-PERMUTATION.md single-feeder premise). With x = +1.0 / -1.0
the product is exact in E4M3 and the expected output byte is the stored
weight byte itself (sign-flipped for -1.0), so the check is BYTE IDENTITY at
every nonzero-fed position. Positions with zero-class weight bytes (0x00 /
0x80) and non-fed positions expect a zero encoding (0x00 observed in base
runs; 0x80 signed-zero variants are recorded as findings, not failures).

Evidence containment: no stored weight byte values are written to evidence.
Equality is attested by SHA-256 over the sorted (offset, expected byte) pairs,
which the companion oracle recomputes independently from the hash-verified
DLL. Zero-variant offsets are retained as hashes and counts only.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import ctypes as ct
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
from typing import Any


def load_helper(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    if spec is None or spec.loader is None:
        raise RuntimeError("missing helper")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


permutation = load_helper("permutation", "native-decoder-permutation-probe.py")
extension = load_helper("extension", "native-decoder-extension-probe.py")
runtime = load_helper("runtime", "native-runtime-probe.py")

base = permutation.base
launch = permutation.launch
weights = permutation.weights
utility = permutation.utility
decoder = permutation.decoder
OutputMismatch = permutation.OutputMismatch
NAME = permutation.NAME
RESET = permutation.RESET
BLOCK = permutation.BLOCK
INPUT_AXES = permutation.INPUT_AXES
OUTPUT_AXES = permutation.OUTPUT_AXES
DecoderArguments = permutation.DecoderArguments
GUARD_WORDS = permutation.GUARD_WORDS
allocate = permutation.allocate
kernel_metadata = permutation.kernel_metadata
reset_geometry = permutation.reset_geometry
input_key = permutation.input_key
key_name = permutation.key_name
e4m3 = permutation.e4m3
finite = permutation.finite
MATRIX_BYTES = permutation.MATRIX_BYTES
PARTITION_BYTES = permutation.PARTITION_BYTES
MAIN_PARTITION_BYTES = permutation.MAIN_PARTITION_BYTES
IMPULSE_BYTES = permutation.IMPULSE_BYTES
E4M3_POS = permutation.E4M3_POS
E4M3_NEG = permutation.E4M3_NEG
E4M3_ZERO = permutation.E4M3_ZERO
PROTOCOL_FINAL = permutation.PROTOCOL_FINAL
RESET_FINAL = permutation.RESET_FINAL
EXTENT_BYTES = permutation.EXTENT_BYTES
OUT0_WINDOW = permutation.OUT0_WINDOW

# ---------------------------------------------------------------------------
# The validated closed form (NATIVE-DECODER-EXTENSION.md), pinned as constants.
# ---------------------------------------------------------------------------

BIT_EFFECTS: dict[int, tuple[int, int, int, int]] = {
    0: (1, 0, 0, 0), 1: (2, 0, 0, 0), 2: (0, 4, 0, 0), 3: (4, 0, 0, 0),
    4: (16, 0, 0, 0), 5: (32, 0, 0, 0), 6: (0, 0, 8, 0), 7: (0, 1, 0, 0),
    8: (0, 0, 0, 4), 9: (16384, 0, 0, 0), 10: (32768, 0, 0, 0),
    11: (65536, 0, 0, 0),
}

# Candidate record and its split.
CANDIDATE_RECORD = "block39.layer0.layer"
RECORD_PAYLOAD_BYTES = 525312
RECORD_DIMENSIONS = [262656]
RECORD_TAG = 1
MATRIX_SLOTS = 4 * PARTITION_BYTES                 # 524,288 kernel coefficient region
SCALE_BYTES = 1024                                  # 512 FP16 values at +0x80000
POISON = launch.OUT0_PATTERN

# Entry plan: single-byte impulses x = +1.0 / -1.0 across partitions and keys.
PLAN_KEYS = ((0, 1), (0, 2), (0, 3), (3, 5), (2, 15), (18, 0), (34, 9),
             (100, 7), (200, 13), (255, 15))
PLAN_PARTITIONS = (0, 1, 2, 3)
PLAN_SIGNS = ("pos", "neg")
WRONG_KEYS = ((0, 1), (255, 15))
WRONG_ROTATION = 1024


def sign_code(sign: str) -> int:
    base.require(sign in PLAN_SIGNS, "sign name")
    return E4M3_POS if sign == "pos" else E4M3_NEG


def main_code(partition: int, unit: int, lane: int, code: int) -> bytes:
    """Zero main input with one nonzero byte `code` at (partition, unit, lane)."""
    key = input_key(partition, unit, lane)
    base.require(type(code) is int and 0 <= code <= 255, "impulse code byte")
    region = bytearray(launch.LARGE_BYTES)
    region[partition * MAIN_PARTITION_BYTES + unit * IMPULSE_BYTES + lane] = code
    return bytes(region)


def word_fill() -> bytes:
    word = struct.pack("<I", weights.F32_ONE)
    return (word * (launch.WEIGHTS_BYTES // 4))[:launch.WEIGHTS_BYTES]


def candidate_payload(data: bytes) -> tuple[bytes, dict[str, Any]]:
    """Extract the candidate record payload from the hash-verified DLL."""
    pe = runtime.PE(data)
    offset, blob = pe.resource("WEIGHTS_HT")
    base.require(type(blob) is bytes and len(blob) > 8, "weights blob")
    position = 8
    names: set[str] = set()
    found: list[tuple[bytes, dict[str, Any]]] = []
    while position < len(blob):
        length = runtime.unpack("<Q", blob, position)[0]
        position += 8
        base.require(0 < length <= 256, "weight name length")
        name = runtime.span(blob, position, length).decode("ascii")
        base.require(name not in names, "duplicate weight name")
        names.add(name)
        position += length
        size = runtime.unpack("<Q", blob, position)[0]
        position += 8
        record = runtime.span(blob, position, size)
        position += size
        base.require(size >= 40, "short weight record")
        declared, payload_bytes, tag = runtime.unpack("<QQI", record, 0)
        base.require(declared == size, "weight record size mismatch")
        payload = runtime.span(record, 20, payload_bytes)
        metadata_a, metadata_b, dimension_count = runtime.unpack("<IIQ", record, 20 + payload_bytes)
        base.require(dimension_count <= 16, "excessive weight dimension count")
        dimensions = list(runtime.unpack("<" + "I" * dimension_count, record, 36 + payload_bytes))
        base.require(size == 36 + payload_bytes + 4 * dimension_count,
                     "weight record trailing bytes")
        if name == CANDIDATE_RECORD:
            found.append((payload, dict(name=name, record_bytes=size, payload_bytes=payload_bytes,
                                        tag=tag, metadata_a=metadata_a, metadata_b=metadata_b,
                                        dimensions=dimensions)))
    base.require(position == len(blob), "weight records overrun container")
    base.require(len(found) == 1, "unique candidate record")
    payload, meta = found[0]
    base.require(len(payload) == RECORD_PAYLOAD_BYTES, "candidate payload size")
    return payload, meta


def binding_metadata(payload: bytes, meta: dict[str, Any]) -> dict[str, Any]:
    """Data-side layout evidence; aggregates only, no payload byte values."""
    matrix, tail = payload[:MATRIX_SLOTS], payload[MATRIX_SLOTS:]
    base.require(len(matrix) == MATRIX_SLOTS and len(tail) == SCALE_BYTES,
                 "candidate split sizes")
    matrix_nan = sum(1 for byte in matrix if byte in (0x7F, 0xFF))
    tail_e4m3_nan = sum(1 for byte in tail if byte in (0x7F, 0xFF))
    matrix_zeros = sum(1 for byte in matrix if byte in (0x00, 0x80))
    tail_fp16 = struct.unpack("<512H", tail)
    tail_nan = sum(1 for value in tail_fp16 if (value & 0x7C00) == 0x7C00 and (value & 0x03FF))
    tail_inf = sum(1 for value in tail_fp16 if (value & 0x7C00) == 0x7C00 and not (value & 0x03FF))
    tail_zeros = sum(1 for value in tail_fp16 if (value & 0x7FFF) == 0)
    dims_ok = (meta["dimensions"] == RECORD_DIMENSIONS
               and meta["tag"] == RECORD_TAG
               and meta["payload_bytes"] == RECORD_PAYLOAD_BYTES
               and RECORD_DIMENSIONS[0] * 2 == RECORD_PAYLOAD_BYTES
               and RECORD_DIMENSIONS[0] == MATRIX_SLOTS // 2 + SCALE_BYTES // 2)
    return dict(record=CANDIDATE_RECORD, record_bytes=meta["record_bytes"],
                payload_bytes=meta["payload_bytes"], tag=meta["tag"],
                dimensions=meta["dimensions"], dimensions_arithmetic_valid=bool(dims_ok),
                split=dict(matrix_bytes=MATRIX_SLOTS, scale_bytes=SCALE_BYTES),
                matrix_e4m3_nan_bytes=matrix_nan,
                matrix_zero_class_bytes=matrix_zeros,
                matrix_distinct_byte_values=len(set(matrix)),
                tail_e4m3_nan_bytes=tail_e4m3_nan,
                tail_fp16=dict(count=512, nan=tail_nan, inf=tail_inf,
                               zeros=tail_zeros, distinct=len(set(tail_fp16))),
                layout_hypothesis=("payload[0:524288] = E4M3 matrix (kernel coefficient region); "
                                   "payload[524288:525312] = 512 FP16 scales at the kernel's "
                                   "word region +0x80000 under byte-identical upload"),
                layout_evidence=("matrix region is E4M3-NaN-free (like the FP8 projection matrix); "
                                 "the 1,024-byte tail is 512 finite FP16 values and contains "
                                 "E4M3-NaN byte encodings, so it is not matrix data; the record "
                                 "dimensions count 262,144 matrix FP16-pairs + 512 FP16 scales"),
                tail_semantics_verified=False,
                learned_weights_used=True, native_image_inference=False,
                deliberate_proprietary_payload_files_written=False,
                payload_byte_values_written_to_evidence=0)


def matrix_upload(payload: bytes, rotation: int) -> bytes:
    """Weights region: F32_ONE word fill, payload byte-identical at offset 0.

    With `rotation`, each 131,072-byte matrix partition is rotated circularly
    inwards by `rotation` bytes (wrong-mode control); the 1,024-byte scale tail
    stays at +0x80000 and bytes outside the record span keep the word fill.
    """
    base.require(type(rotation) is int and 0 <= rotation < PARTITION_BYTES, "matrix rotation")
    region = bytearray(word_fill())
    if rotation == 0:
        region[:len(payload)] = payload
    else:
        matrix = bytearray(MATRIX_SLOTS)
        for partition in range(4):
            chunk = payload[partition * PARTITION_BYTES:(partition + 1) * PARTITION_BYTES]
            matrix[partition * PARTITION_BYTES:(partition + 1) * PARTITION_BYTES] = \
                chunk[rotation:] + chunk[:rotation]
        region[:MATRIX_SLOTS] = matrix
        region[MATRIX_SLOTS:len(payload)] = payload[MATRIX_SLOTS:]
    return bytes(region)


def expected_map(payload: bytes, partition: int, unit: int, lane: int,
                 sign: str, rotation: int) -> dict[int, int]:
    """Predicted nonzero output bytes {(offset): expected byte} for one impulse.

    Non-fed and zero-class-weight positions are predicted as zero-class and are
    not returned here; classification checks cover them.
    """
    code = sign_code(sign)
    flip = 0x80 if code == E4M3_NEG else 0x00
    out: dict[int, int] = {}
    for offset, matrix_byte in extension.model_map(BIT_EFFECTS, unit, lane).items():
        local = (matrix_byte + rotation) % PARTITION_BYTES if rotation else matrix_byte
        weight = payload[partition * PARTITION_BYTES + local]
        if weight in (0x00, 0x80):
            continue
        out[offset] = weight ^ flip
    base.require(len(out) <= 2048, "expected map bound")
    return out


def zero_class_positions(unit: int, lane: int) -> set[int]:
    fed = set(extension.model_map(BIT_EFFECTS, unit, lane))
    return (set(range(EXTENT_BYTES)) - fed)


def nonzero_digest(expected: dict[int, int]) -> tuple[str, int]:
    packed = b"".join(struct.pack("<IB", offset, byte)
                      for offset, byte in sorted(expected.items()))
    return hashlib.sha256(packed).hexdigest(), len(expected)


def offsets_digest(offsets: list[int]) -> str:
    packed = b"".join(struct.pack("<I", offset) for offset in sorted(offsets))
    return hashlib.sha256(packed).hexdigest()


def self_test() -> dict[str, Any]:
    regression = launch.self_test()
    rejected = 0

    def check(condition: Any, label: str) -> None:
        if not condition:
            raise OutputMismatch(f"self-test: {label}")

    def reject(action: Any, label: str) -> None:
        nonlocal rejected
        try:
            action()
        except (OutputMismatch, ValueError, TypeError, AssertionError):
            rejected += 1
            return
        raise OutputMismatch(f"self-test: corrupt control accepted ({label})")

    # Pinned closed-form constants vs the extension model (cross-helper).
    check(set(BIT_EFFECTS) == set(range(12)), "bit effects cover 12 bits")
    check(all(len(effect) == 4 and all(type(value) is int for value in effect)
              for effect in BIT_EFFECTS.values()), "bit effect tuple shapes")
    known_effects = {0: (1, 0, 0, 0), 8: (0, 0, 0, 4), 11: (65536, 0, 0, 0),
                     2: (0, 4, 0, 0), 6: (0, 0, 8, 0), 7: (0, 1, 0, 0)}
    for bit, effect in known_effects.items():
        check(BIT_EFFECTS[bit] == effect, f"bit effect known answer {bit}")
    for unit, lane in ((0, 1), (255, 15), (34, 9)):
        inline = extension.model_map(BIT_EFFECTS, unit, lane)
        check(len(inline) == 2048, "model map size")
        check(all(0 <= offset < EXTENT_BYTES for offset in inline), "model map bounds")
        check(all(0 <= matrix_byte < PARTITION_BYTES for matrix_byte in inline.values()),
              "matrix byte bounds")
    # Identity key: n = 0 must be the pure shuffle.
    identity = extension.model_map(BIT_EFFECTS, 0, 0)
    check(all(matrix_byte == extension.shuffle(offset)
              for offset, matrix_byte in identity.items()), "identity map is pure shuffle")
    # Sign flip and zero-class classification known answers.
    check(sign_code("pos") == 0x38 and sign_code("neg") == 0xB8, "sign codes")
    check((0x38 ^ 0x80) == 0xB8 and (0x00 ^ 0x80) == 0x80, "sign flip known answers")
    check({0x00, 0x80} == {byte for byte in (E4M3_ZERO, 0x80)}, "zero-class set")
    check(finite(e4m3(0x38)) == 1.0 and finite(e4m3(0xB8)) == -1.0, "e4m3 decode answers")
    check(finite(e4m3(0x00)) == 0.0 and e4m3(0x7F) != e4m3(0x7F), "e4m3 zero/nan answers")
    # Nonzero digest is order-stable and detects single-byte corruption.
    sample = {7: 0x38, 3: 0x01, 100: 0xB8}
    digest_a, count_a = nonzero_digest(sample)
    digest_b, _ = nonzero_digest(dict(reversed(list(sample.items()))))
    check(digest_a == digest_b and count_a == 3, "digest order stability")
    check(digest_a != nonzero_digest({7: 0x39, 3: 0x01, 100: 0xB8})[0], "digest detects mutation")
    check(offsets_digest([5, 1, 9]) == offsets_digest([1, 5, 9]), "offset digest order stability")
    check(offsets_digest([5, 1, 9]) != offsets_digest([1, 5]), "offset digest length sensitivity")
    reject(lambda: check(nonzero_digest({7: 0x38})[0] == nonzero_digest({7: 0x39})[0],
                         "mismatching digests accepted"), "digest equality corruption")
    # Entry plan: full cell coverage (key x partition x sign) and non-vacuity.
    plan = [(unit, lane, partition, sign) for unit, lane in PLAN_KEYS
            for partition in PLAN_PARTITIONS for sign in PLAN_SIGNS]
    check(len(plan) == len(PLAN_KEYS) * 4 * 2 == 80, "plan entry count")
    check(len({(unit, lane) for unit, lane, _, _ in plan}) == len(PLAN_KEYS), "plan key coverage")
    check({partition for _, _, partition, _ in plan} == set(PLAN_PARTITIONS), "plan partitions")
    check({sign for _, _, _, sign in plan} == set(PLAN_SIGNS), "plan signs")
    # Expected-map non-vacuity against synthetic payloads.
    # Uniform payload: every fed position answers; digests vary only through the
    # fed sets, so distinctness is checked against a periodic payload whose
    # bytes depend on the matrix offset (keys with equal fed sets but distinct
    # B(n) terms must still produce distinct maps).
    synthetic = bytes([0x38]) * RECORD_PAYLOAD_BYTES
    varied = bytes(range(256)) * (RECORD_PAYLOAD_BYTES // 256)
    distinct_maps = len({nonzero_digest(expected_map(varied, 0, unit, lane, "pos", 0))[0]
                         for unit, lane in PLAN_KEYS})
    check(distinct_maps == len(PLAN_KEYS), "plan expected maps distinct per key")
    full = expected_map(synthetic, 0, 0, 1, "pos", 0)
    check(len(full) == 2048, "uniform payload fills every fed position")
    mirrored = expected_map(synthetic, 0, 0, 1, "neg", 0)
    check(all(mirrored[offset] == (byte ^ 0x80) for offset, byte in full.items()),
          "neg map is sign mirror")
    zeroed = expected_map(bytes(RECORD_PAYLOAD_BYTES), 0, 0, 1, "pos", 0)
    check(len(zeroed) == 0, "zero payload predicts silence")
    rotated = expected_map(synthetic, 0, 0, 1, "pos", WRONG_ROTATION)
    check(rotated == full, "uniform payload rotation-invariant")
    shifted = expected_map(bytes([0x38]) * MATRIX_SLOTS + bytes(SCALE_BYTES), 0, 0, 1, "pos", 0)
    check(shifted == full, "scale tail unused by predictions")
    # Zero-class position accounting.
    check(len(zero_class_positions(0, 1)) == EXTENT_BYTES - 2048, "zero-class position count")
    return dict(launch_regression=regression, model_known_answers=len(known_effects) + 2,
                plan_entries=len(plan), plan_keys=len(PLAN_KEYS),
                digest_controls=4, corrupt_controls_rejected=rejected,
                expected_map_known_answers=6, gpu_kernels_launched=0)


def execute(data: bytes, inventory: dict[str, Any], mode: str) -> dict[str, Any]:
    entries = [entry for entry in inventory["fatbins"][6]["entries"]
               if entry["kind"] == 2 and entry["arch"] == 89]
    base.require(len(entries) == 1, "unique SM89 decoder module")
    image = base.decompress(data, entries[0])
    utility.verify_kernel_abi(image, RESET, 16)
    abi = decoder.parameter_contract(kernel_metadata(image, NAME))
    base.require(abi["bytes"] == 80, "decoder ABI size")
    payload, meta = candidate_payload(data)
    metadata = binding_metadata(payload, meta)
    rotation = WRONG_ROTATION if mode == "wrong" else 0
    driver = utility.Driver()
    device, version = utility.select_device(driver)
    launches = readbacks = 0
    guard_words = [0x12340000 + i for i in range(GUARD_WORDS)]
    poison_window = bytes([POISON]) * OUT0_WINDOW
    poison_full = bytes([POISON]) * launch.OUT0_BYTES
    records: list[dict[str, Any]] = []
    weights_snapshot = matrix_upload(payload, rotation)

    def fail(message: str) -> None:
        print(json.dumps(dict(failure=message, binding_metadata=metadata,
                              records=records), indent=2))
        raise OutputMismatch(message)

    with ExitStack() as resources:
        context = ct.c_void_p()
        driver.check("cuCtxCreate_v2", ct.byref(context), 0, device)
        resources.callback(driver.check, "cuCtxDestroy_v2", context)
        module = ct.c_void_p()
        driver.check("cuModuleLoadData", ct.byref(module), ct.create_string_buffer(image))
        resources.callback(driver.check, "cuModuleUnload", module)
        decoder_function, reset_function = ct.c_void_p(), ct.c_void_p()
        driver.check("cuModuleGetFunction", ct.byref(decoder_function), module, NAME.encode())
        driver.check("cuModuleGetFunction", ct.byref(reset_function), module, RESET.encode())
        with ExitStack() as allocations:
            addresses = {}
            for key, size in (("main", launch.LARGE_BYTES), ("skip", launch.LARGE_BYTES),
                              ("weights", launch.WEIGHTS_BYTES), ("partials", launch.LARGE_BYTES),
                              ("out0", launch.OUT0_BYTES), ("out3", launch.SMALL_BYTES),
                              ("out2", launch.SMALL_BYTES)):
                addresses[key] = allocate(driver, allocations, size)
            counter_base, counter_region, _ = allocate(driver, allocations, launch.COUNTER_WORDS * 4)

            def upload(key: str, content: bytes) -> bytes:
                base_address, region, size = addresses[key]
                base.require(len(content) == size, "upload size")
                host = (ct.c_uint8 * (size + 2 * GUARD_WORDS * 4))()
                ct.memmove(ct.byref(host, GUARD_WORDS * 4), content, size)
                for index, word in enumerate(guard_words):
                    ct.cast(ct.byref(host, index * 4), ct.POINTER(ct.c_uint32)).contents.value = word
                    ct.cast(ct.byref(host, GUARD_WORDS * 4 + size + index * 4),
                            ct.POINTER(ct.c_uint32)).contents.value = word
                driver.check("cuMemcpyHtoD_v2", ct.c_uint64(base_address), host, ct.sizeof(host))
                return content

            def download(key: str) -> tuple[bytes, bool]:
                base_address, region, size = addresses[key]
                host = (ct.c_uint8 * (size + 2 * GUARD_WORDS * 4))()
                driver.check("cuMemcpyDtoH_v2", host, ct.c_uint64(base_address), ct.sizeof(host))
                region_bytes = ct.string_at(ct.byref(host, GUARD_WORDS * 4), size)
                intact = all(ct.cast(ct.byref(host, index * 4),
                                     ct.POINTER(ct.c_uint32)).contents.value == word
                             for index, word in enumerate(guard_words))
                intact &= all(ct.cast(ct.byref(host, GUARD_WORDS * 4 + size + index * 4),
                                      ct.POINTER(ct.c_uint32)).contents.value == word
                              for index, word in enumerate(guard_words))
                return region_bytes, bool(intact)

            def upload_window(content: bytes) -> None:
                base.require(len(content) == OUT0_WINDOW, "window upload size")
                base_address, region, size = addresses["out0"]
                host = (ct.c_uint8 * (GUARD_WORDS * 4 + OUT0_WINDOW))()
                for index, word in enumerate(guard_words):
                    ct.cast(ct.byref(host, index * 4), ct.POINTER(ct.c_uint32)).contents.value = word
                ct.memmove(ct.byref(host, GUARD_WORDS * 4), content, OUT0_WINDOW)
                driver.check("cuMemcpyHtoD_v2", ct.c_uint64(base_address), host, ct.sizeof(host))

            def download_window() -> tuple[bytes, bool]:
                base_address, region, size = addresses["out0"]
                host = (ct.c_uint8 * (GUARD_WORDS * 4 + OUT0_WINDOW))()
                driver.check("cuMemcpyDtoH_v2", host, ct.c_uint64(base_address), ct.sizeof(host))
                region_bytes = ct.string_at(ct.byref(host, GUARD_WORDS * 4), OUT0_WINDOW)
                intact = all(ct.cast(ct.byref(host, index * 4),
                                     ct.POINTER(ct.c_uint32)).contents.value == word
                             for index, word in enumerate(guard_words))
                return region_bytes, bool(intact)

            def reset_counters() -> None:
                nonlocal launches
                counters = (ct.c_uint8 * (launch.COUNTER_WORDS * 4 + 2 * GUARD_WORDS * 4))()
                for index, word in enumerate(guard_words):
                    ct.cast(ct.byref(counters, index * 4), ct.POINTER(ct.c_uint32)).contents.value = word
                    ct.cast(ct.byref(counters, GUARD_WORDS * 4 + launch.COUNTER_WORDS * 4 + index * 4),
                            ct.POINTER(ct.c_uint32)).contents.value = word
                counter_values = (ct.c_int32 * launch.COUNTER_WORDS)(*([0] * launch.COUNTER_WORDS))
                ct.memmove(ct.byref(counters, GUARD_WORDS * 4), counter_values, ct.sizeof(counter_values))
                driver.check("cuMemcpyHtoD_v2", ct.c_uint64(counter_base), counters, ct.sizeof(counters))
                grid = reset_geometry(launch.COUNTER_WORDS)
                reset_args = utility.CounterArguments(counter_region, launch.COUNTER_WORDS)
                params = (ct.c_void_p * 1)(ct.addressof(reset_args))
                driver.check("cuLaunchKernel", reset_function, *grid, 256, 1, 1, 0, None, params, None)
                launches += 1
                driver.check("cuCtxSynchronize")

            def read_counters() -> tuple[list[int], bool]:
                nonlocal readbacks
                counters = (ct.c_uint8 * (launch.COUNTER_WORDS * 4 + 2 * GUARD_WORDS * 4))()
                driver.check("cuMemcpyDtoH_v2", counters, ct.c_uint64(counter_base), ct.sizeof(counters))
                readbacks += 1
                values = ct.cast(ct.byref(counters, GUARD_WORDS * 4),
                                 ct.POINTER(ct.c_int32 * launch.COUNTER_WORDS)).contents
                intact = all(ct.cast(ct.byref(counters, index * 4),
                                     ct.POINTER(ct.c_uint32)).contents.value == word
                             for index, word in enumerate(guard_words))
                intact &= all(ct.cast(ct.byref(counters, GUARD_WORDS * 4 + launch.COUNTER_WORDS * 4 + index * 4),
                                      ct.POINTER(ct.c_uint32)).contents.value == word
                              for index, word in enumerate(guard_words))
                return list(values), bool(intact)

            def run_entry(name: str, key: tuple[int, int, int] | None,
                          sign: str | None, expected: dict[int, int],
                          intended: dict[int, int] | None) -> dict[str, Any]:
                nonlocal launches, readbacks
                impulse = bytes(launch.LARGE_BYTES) if key is None else \
                    main_code(key[0], key[1], key[2], sign_code(sign or "pos"))
                snapshots = {
                    "main": upload("main", impulse),
                    "skip": upload("skip", bytes(launch.LARGE_BYTES)),
                    "weights": upload("weights", weights_snapshot),
                    "partials": upload("partials", bytes(launch.LARGE_BYTES)),
                    "out3": upload("out3", bytes([launch.IMMUTABLE_PATTERN]) * launch.SMALL_BYTES),
                    "out2": upload("out2", bytes([launch.IMMUTABLE_PATTERN]) * launch.SMALL_BYTES),
                }
                upload_window(poison_window)
                reset_counters()
                arguments = DecoderArguments(
                    main=addresses["main"][1], skip=addresses["skip"][1],
                    out0=addresses["out0"][1], out3=addresses["out3"][1],
                    counters=counter_region, out2=addresses["out2"][1],
                    partials=addresses["partials"][1], weights=addresses["weights"][1],
                    dimension=INPUT_AXES[0], width=INPUT_AXES[1],
                    output_axis0=OUTPUT_AXES[0], output_axis1=OUTPUT_AXES[1])
                grid_x, grid_y, grid_z = decoder.launch_geometry(*INPUT_AXES)
                params = (ct.c_void_p * 1)(ct.addressof(arguments))
                driver.check("cuLaunchKernel", decoder_function, grid_x, grid_y, grid_z,
                             *BLOCK, 0, None, params, None)
                launches += 1
                driver.check("cuCtxSynchronize")
                observed_counters, counters_intact = read_counters()
                immutable = True
                guards_intact = True
                for region_key in ("main", "skip", "weights", "out3", "out2"):
                    region_bytes, intact = download(region_key)
                    readbacks += 1
                    immutable &= region_bytes == snapshots[region_key]
                    guards_intact &= intact
                partials_bytes, partials_intact = download("partials")
                readbacks += 1
                guards_intact &= partials_intact
                out0_window, window_intact = download_window()
                readbacks += 1
                guards_intact &= window_intact
                if not counters_intact:
                    fail(f"counter guard violation ({name})")
                if observed_counters != [PROTOCOL_FINAL] * launch.COUNTER_WORDS:
                    fail(f"counter protocol mismatch ({name}): {observed_counters}")
                if not (immutable and guards_intact):
                    fail(f"guard/canary/immutability violation ({name})")
                if out0_window[EXTENT_BYTES:] != poison_window[EXTENT_BYTES:]:
                    fail(f"write outside the extent ({name})")
                # Nonzero-fed positions: strict byte identity with the prediction.
                mismatches = [[offset, out0_window[offset], expected[offset]]
                              for offset in sorted(expected)
                              if out0_window[offset] != expected[offset]]
                if mismatches:
                    fail(f"product identity mismatch ({name}): "
                         f"{len(mismatches)} of {len(expected)}; first {mismatches[:8]}")
                # Zero-class positions: zero encodings accepted; 0x80 recorded.
                zero_variants = [offset for offset in range(EXTENT_BYTES)
                                 if offset not in expected and out0_window[offset] == 0x80]
                bad_zero = [offset for offset in range(EXTENT_BYTES)
                            if offset not in expected and out0_window[offset] not in (0x00, 0x80)]
                if bad_zero:
                    fail(f"nonzero output at zero-class position ({name}): first {bad_zero[:8]}")
                record = dict(name=name, kind="impulse" if key is not None else "base",
                              impulse_partition=None if key is None else key[0],
                              impulse_unit=None if key is None else key[1],
                              impulse_lane=None if key is None else key[2],
                              sign=sign, rotation=rotation,
                              expected_nonzero_sha256=nonzero_digest(expected)[0],
                              expected_nonzero_count=len(expected),
                              zero_class_count=EXTENT_BYTES - len(expected),
                              zero_0x80_variant_count=len(zero_variants),
                              zero_0x80_variant_offsets_sha256=offsets_digest(zero_variants),
                              product_identity_match=True, margin_poison_intact=True,
                              counters_observed=observed_counters,
                              counters_guard_intact=bool(counters_intact),
                              immutable_regions_intact=bool(immutable),
                              guards_intact=bool(guards_intact))
                if intended is not None:
                    record["intended_k0_nonzero_sha256"] = nonzero_digest(intended)[0]
                    record["intended_k0_rejected"] = bool(nonzero_digest(intended)[0]
                                                          != nonzero_digest(expected)[0])
                    if out0_window[:EXTENT_BYTES] == b"".join(
                            bytes([intended.get(offset, 0x00)]) for offset in range(EXTENT_BYTES)):
                        fail(f"rotated run reproduced the intended k0 result ({name})")
                return record

            if mode == "skip":
                initial = {key: upload(key, content) for key, content in
                           (("main", bytes(launch.LARGE_BYTES)), ("skip", bytes(launch.LARGE_BYTES)),
                            ("weights", weights_snapshot),
                            ("partials", bytes(launch.LARGE_BYTES)),
                            ("out0", poison_full),
                            ("out3", bytes([launch.IMMUTABLE_PATTERN]) * launch.SMALL_BYTES),
                            ("out2", bytes([launch.IMMUTABLE_PATTERN]) * launch.SMALL_BYTES))}
                reset_counters()
                observed, counters_intact = read_counters()
                if not counters_intact:
                    fail("skip counter guard violation")
                if observed != [RESET_FINAL] * launch.COUNTER_WORDS:
                    fail(f"skip counters not the reset value: {observed}")
                guards_intact = True
                immutable = True
                out0: bytes | None = None
                for key in ("main", "skip", "weights", "partials", "out0", "out3", "out2"):
                    region_bytes, intact = download(key)
                    readbacks += 1
                    guards_intact &= intact
                    immutable &= region_bytes == initial[key]
                    if key == "out0":
                        out0 = region_bytes
                base.require(out0 is not None, "skip out0 readback")
                if out0 != poison_full or not (guards_intact and immutable):
                    fail("skip launch changed output or guards")
                return dict(native=True, mode=mode, kernel=NAME, kernels_launched=launches,
                            decoder_launches=0,
                            readbacks=readbacks, counters_observed=observed,
                            counter_reset_value=RESET_FINAL, out0_changed=False,
                            all_guards_intact=guards_intact, immutable_regions_checked=list(initial),
                            binding_metadata=metadata, negative_control_rejected=True,
                            weight_record_binding_verified=False,
                            learned_weights_used=False, native_image_inference=False,
                            candidate_weight_payload_uploaded=True,
                            deliberate_proprietary_payload_files_written=False)

            # Entries: base (zero impulse) then the impulse plan. The initial
            # full out0 upload sets both guard zones and poisons the whole
            # region; per-entry window uploads then refresh the leading guard.
            upload("out0", poison_full)
            keys = WRONG_KEYS if mode == "wrong" else PLAN_KEYS
            entries_plan = [(unit, lane, partition, sign) for unit, lane in keys
                            for partition in PLAN_PARTITIONS for sign in PLAN_SIGNS]
            base_record = run_entry("base-real", None, None, {}, None)
            records.append(base_record)
            for unit, lane, partition, sign in entries_plan:
                expected = expected_map(payload, partition, unit, lane, sign, rotation)
                intended = None
                if mode == "wrong":
                    intended = expected_map(payload, partition, unit, lane, sign, 0)
                records.append(run_entry(f"{key_name((partition, unit, lane))}-{sign}",
                                         (partition, unit, lane), sign, expected, intended))

            # Final full-region check on the last entry's output.
            full, full_intact = download("out0")
            readbacks += 1
            if not full_intact:
                fail("final full-region guard violation")
            if full[EXTENT_BYTES:] != poison_full[EXTENT_BYTES:]:
                fail("out0 written outside the measurement window during the run")

        identity_ok = all(record["product_identity_match"] for record in records)
        margin_ok = all(record["margin_poison_intact"] for record in records)
        intended_rejected = (all(record.get("intended_k0_rejected", True) for record in records)
                             if mode == "wrong" else True)
        binding = bool(identity_ok and margin_ok and intended_rejected)
        return dict(native=True, device="NVIDIA GeForce RTX 4070 Ti SUPER",
                    driver_api_version=version, mode=mode, kernel=NAME,
                    block=list(BLOCK), kernels_launched=launches,
                    decoder_launches=len(records), readbacks=readbacks,
                    candidate_weight_record=CANDIDATE_RECORD,
                    matrix_upload_rotation=rotation,
                    records=records,
                    impulse_entries=len(entries_plan), base_entries=1,
                    product_identity_passed=bool(identity_ok),
                    margin_poison_passed=bool(margin_ok),
                    intended_k0_rejected=bool(intended_rejected),
                    zero_0x80_variant_total=sum(record["zero_0x80_variant_count"]
                                                for record in records),
                    out0_full_region_final_check=True,
                    weight_record_binding_verified=binding,
                    binding_scope=("stored record bytes uploaded byte-identically; single-product "
                                   "byte-identity validated on 80 single-byte impulses across "
                                   "4 partitions of the tested 8x8 -> 16x16 contract"
                                   if mode == "execute" else
                                   "wrong-mode rotated-matrix control; rejects the intended k0 result"
                                   if mode == "wrong" else "skip control"),
                    learned_weights_used=(mode != "skip"), native_image_inference=False,
                    candidate_weight_payload_uploaded=True,
                    deliberate_proprietary_payload_files_written=False,
                    payload_byte_values_written_to_evidence=0,
                    binding_metadata=metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("self-test", "execute", "skip", "wrong"))
    parser.add_argument("dll", type=Path, nargs="?")
    args = parser.parse_args()
    if args.mode == "self-test":
        print(json.dumps(self_test(), indent=2))
        return
    base.require(args.dll is not None, "DLL required")
    policy = base.protect_process_memory()
    policy["disassembler_same_uid_proc_exposure_controlled"] = False
    data = base.read_bounded(args.dll)
    inventory = base.inventory(data)
    print(json.dumps(dict(dll_sha256=base.PIN, memory_policy=policy,
                          cpu_self_test=self_test(), cuda=execute(data, inventory, args.mode)),
                     indent=2))


if __name__ == "__main__":
    main()
