#!/usr/bin/env python3
"""Byte-granularity (matrix byte, input byte) -> output byte resolution for the
decoder weight region; NOT neural inference.

Launches the pinned SM89 `cc_dec_input_upsample_1024_512_fp8` on native Linux
CUDA (no Windows code, DirectX or Proton) with a SINGLE nonzero FP8 main-input
byte and bit-coded synthetic matrix partitions. With one nonzero input byte,
each influenced output byte receives exactly one product, so two probe kinds
resolve the (o, i) mapping per physical matrix byte for the probed input bytes:

  * Bit-signature runs: for input byte i, code matrix byte m iff bit b of m is
    set (17 runs; PARTITION_BYTES == 2**17). Output byte o responds in run b
    iff bit b of its feeding matrix byte m(o) is set, so the response pattern
    decodes m(o) exactly — CONSISTENT with one feeding matrix byte per output
    byte. Two checks support that premise: (a) per input byte, every responding
    output byte must return the identical FP8 value in every responding
    signature run, which detects sums of DISTINCT feeding matrix bytes (their
    bit patterns disagree somewhere, so the summed value would differ by run);
    (b) every direct run codes exactly one matrix byte, so its response is a
    true single product and must equal the signature response for the same
    (input, output) byte, sign-adjusted. A matrix byte feeding one output byte
    through two identical paths remains correctly decoded (same m) although its
    response doubles; magnitude linearity is NOT claimed. Coverage is complete
    per probed input byte within its coupled matrix partition: every nonzero
    matrix byte of that partition has at least one bit set.
  * Direct runs: code one matrix byte alone under one input byte and require
    the changed output set to equal the decoded preimage of that matrix byte —
    positive code, negative code for exact sign mirroring, and cross-input
    pairs whose prediction comes from the other input's decode. Direct runs
    cover 16 of the 23 probe inputs (own/sign pairs); the remainder are
    signature-only.

Retained evidence: per-entry change ranges and sign ranges, the column decode,
direct predictions vs measurements and named output cells. It does NOT use
learned weights and produces no neural-rendered image.

Bounded-transfer note: per-entry out0 traffic is limited to [0, EXTENT_BYTES +
4096), the established write extent plus margin. The full 256 MiB region is
poisoned once per run and verified after the final entry, so any write outside
the window persists in device memory and fails the final full-region check —
except one whose byte values coincide with the 0xA5 poison pattern, which is
not detectable. Guard zones of every region are checked on every download.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import ctypes as ct
import importlib.util
import json
from pathlib import Path
from typing import Any


def load_helper(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    if spec is None or spec.loader is None:
        raise RuntimeError("missing helper")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


sweep = load_helper("sweep", "native-decoder-sweep-probe.py")
weights = sweep.weights
launch = sweep.launch
base = sweep.base
utility = sweep.utility
decoder = sweep.decoder
OutputMismatch = sweep.OutputMismatch
NAME = sweep.NAME
RESET = sweep.RESET
BLOCK = sweep.BLOCK
INPUT_AXES = sweep.INPUT_AXES
OUTPUT_AXES = sweep.OUTPUT_AXES
DecoderArguments = sweep.DecoderArguments
GUARD_WORDS = sweep.GUARD_WORDS
allocate = sweep.allocate
kernel_metadata = sweep.kernel_metadata
reset_geometry = sweep.reset_geometry
canonical_units = sweep.canonical_units
factor_cell = sweep.factor_cell
e4m3 = weights.e4m3
finite = weights.finite
differing_ranges = weights.differing_ranges
MATRIX_BYTES = weights.MATRIX_BYTES
PARTITION_BYTES = weights.PARTITION_BYTES
MAIN_PARTITION_BYTES = weights.MAIN_PARTITION_BYTES
IMPULSE_BYTES = weights.IMPULSE_BYTES
E4M3_POS = weights.E4M3_POS
E4M3_NEG = weights.E4M3_NEG
E4M3_ZERO = weights.E4M3_ZERO
PROTOCOL_FINAL = weights.PROTOCOL_FINAL
RESET_FINAL = weights.RESET_FINAL
EXTENT_BYTES = weights.EXTENT_BYTES

SIGNATURE_BITS = 17          # PARTITION_BYTES == 1 << 17
UNITS_PER_PARTITION = MAIN_PARTITION_BYTES // IMPULSE_BYTES
OUT0_WINDOW = EXTENT_BYTES + 4096   # Established extent plus margin (per-entry traffic).

base.require(PARTITION_BYTES == 1 << SIGNATURE_BITS, "signature bits cover one matrix partition")
base.require(UNITS_PER_PARTITION == 256, "impulse units per partition slice")

# Probed input bytes as (partition, unit, lane); lane None = whole 16-byte unit.
# Lanes 0-15 of (p0, u0) resolve lane semantics; units 1/4/8/32 span the reach
# classes of NATIVE-DECODER-SWEEP.md; partitions 1-3 re-check coupling.
PROBE_INPUTS = ([(0, 0, lane) for lane in range(16)]
                + [(0, 1, 0), (0, 4, 0), (0, 8, 0), (0, 32, 0)]
                + [(1, 0, 0), (2, 0, 0), (3, 0, 0)])
OWN_CHECKS = 8
SIGN_CHECKS = 2
CROSS_CHECKS = 8
CROSS_STRIDE = 12


def input_key(partition: int, unit: int, lane: int | None) -> tuple[int, int, int | None]:
    base.require(type(partition) is int and 0 <= partition < 4, "impulse partition")
    base.require(type(unit) is int and 0 <= unit < UNITS_PER_PARTITION, "impulse unit")
    base.require(lane is None or (type(lane) is int and 0 <= lane < IMPULSE_BYTES), "impulse lane")
    return partition, unit, lane


def key_name(key: tuple[int, int, int | None]) -> str:
    partition, unit, lane = key
    return f"p{partition}-u{unit}-l{lane}" if lane is not None else f"p{partition}-u{unit}-unit16"


def main_byte(key: tuple[int, int, int | None]) -> bytes:
    """Zero main input with one nonzero byte (or one 16-byte unit) in one slice."""
    partition, unit, lane = key
    region = bytearray(launch.LARGE_BYTES)
    if lane is None:
        region[partition * MAIN_PARTITION_BYTES + unit * IMPULSE_BYTES:
               partition * MAIN_PARTITION_BYTES + (unit + 1) * IMPULSE_BYTES] = bytes([E4M3_POS]) * IMPULSE_BYTES
    else:
        region[partition * MAIN_PARTITION_BYTES + unit * IMPULSE_BYTES + lane] = E4M3_POS
    return bytes(region)


def _word_region() -> bytearray:
    import struct as _struct
    word = _struct.pack("<I", weights.F32_ONE)
    return bytearray((word * (launch.WEIGHTS_BYTES // 4))[:launch.WEIGHTS_BYTES])


_SIGNATURE_CACHE: dict[tuple[int, int], bytes] = {}


def weights_signature(partition: int, bit: int) -> bytes:
    """Weights region: word overlay plus matrix bytes coded iff bit `bit` of the
    byte's partition-local offset is set."""
    base.require(type(partition) is int and 0 <= partition < 4, "matrix partition")
    base.require(type(bit) is int and 0 <= bit < SIGNATURE_BITS, "signature bit")
    cached = _SIGNATURE_CACHE.get((partition, bit))
    if cached is not None:
        return cached
    region = _word_region()
    matrix = bytearray(MATRIX_BYTES)
    start = partition * PARTITION_BYTES
    matrix[start:start + PARTITION_BYTES] = bytes(
        E4M3_POS if (offset >> bit) & 1 else E4M3_ZERO for offset in range(PARTITION_BYTES))
    region[:MATRIX_BYTES] = matrix
    built = bytes(region)
    _SIGNATURE_CACHE[(partition, bit)] = built
    return built


def weights_single(partition: int, offset: int, code: int) -> bytes:
    """Weights region: word overlay plus a single coded matrix byte."""
    base.require(type(partition) is int and 0 <= partition < 4, "matrix partition")
    base.require(type(offset) is int and 0 <= offset < PARTITION_BYTES, "matrix byte offset")
    base.require(type(code) is int and 0 <= code <= 255, "code byte")
    region = _word_region()
    matrix = bytearray(MATRIX_BYTES)
    matrix[partition * PARTITION_BYTES + offset] = code
    region[:MATRIX_BYTES] = matrix
    return bytes(region)


def expand_ranges(ranges: Any, bound: int) -> set[int]:
    """Expanded byte offsets from retained half-open [start, end) ranges."""
    base.require(isinstance(ranges, list), "ranges list")
    out: set[int] = set()
    for pair in ranges:
        base.require(isinstance(pair, list) and len(pair) == 2, "range pair")
        start, end = pair
        base.require(type(start) is int and type(end) is int and 0 <= start < end, "range bounds")
        base.require(end <= bound, "range inside bound")
        out.update(range(start, end))
    return out


def decode_column(presence: dict[int, set[int]]) -> dict[int, int]:
    """Signature decode: presence[bit] = output bytes changed in signature run
    `bit`; returns {output byte offset: feeding matrix byte}."""
    base.require(isinstance(presence, dict) and presence, "signature presence")
    offsets: set[int] = set()
    for bit, changed in presence.items():
        base.require(type(bit) is int and 0 <= bit < SIGNATURE_BITS, "signature bit")
        base.require(isinstance(changed, set), "changed set")
        offsets |= changed
    out: dict[int, int] = {}
    for offset in sorted(offsets):
        matrix_byte = 0
        for bit, changed in presence.items():
            if offset in changed:
                matrix_byte |= 1 << bit
        base.require(matrix_byte > 0, "decoded matrix byte is zero")
        out[offset] = matrix_byte
    base.require(out, "empty column decode")
    return out


def preimages(decode: dict[int, int]) -> dict[int, list[int]]:
    """Invert a column decode: {matrix byte: sorted output byte offsets}."""
    out: dict[int, list[int]] = {}
    for offset, matrix_byte in decode.items():
        base.require(type(offset) is int and type(matrix_byte) is int and matrix_byte > 0,
                     "decode entry types")
        out.setdefault(matrix_byte, []).append(offset)
    return {matrix_byte: sorted(offsets) for matrix_byte, offsets in out.items()}


def mirror_samples(pos_window: bytes, neg_window: bytes, offsets: Any) -> tuple[bool, list[list[int]]]:
    """Sign-mirror check: every changed byte must flip exactly the FP8 sign bit."""
    base.require(isinstance(offsets, set) and offsets, "mirror offsets")
    base.require(len(pos_window) == len(neg_window), "mirror window sizes")
    samples: list[list[int]] = []
    matched = True
    for offset in sorted(offsets):
        base.require(0 <= offset < len(pos_window), "mirror offset bound")
        if neg_window[offset] != (pos_window[offset] ^ 0x80):
            matched = False
        if len(samples) < 8:
            samples.append([offset, pos_window[offset], neg_window[offset]])
    return matched, samples


def phase1_plan() -> list[dict[str, Any]]:
    """Fixed measurement plan: controls, per-input impulse bases, signatures."""
    plan: list[dict[str, Any]] = [dict(name="base", kind="control", impulse=None,
                                       matrix=None, code=E4M3_ZERO)]
    for key in PROBE_INPUTS:
        plan.append(dict(name=f"impulse-base-{key_name(key)}", kind="impulse-base",
                         impulse=key, matrix=None, code=E4M3_ZERO))
    plan.append(dict(name="coupled-ref", kind="control", impulse=(0, 0, None), matrix=("whole", 0, E4M3_POS),
                     code=E4M3_POS))
    plan.append(dict(name="decoupled-ctrl", kind="control", impulse=(0, 0, 0), matrix=("whole", 1, E4M3_POS),
                     code=E4M3_POS))
    for key in PROBE_INPUTS:
        for bit in range(SIGNATURE_BITS):
            plan.append(dict(name=f"sig-{key_name(key)}-b{bit}", kind="signature",
                             impulse=key, matrix=("sig", key[0], bit), code=E4M3_POS))
    return plan


def phase2_plan(decode_by_input: dict[tuple[int, int, int | None], dict[int, int]]) -> list[dict[str, Any]]:
    """Deterministic direct checks selected from the decoded columns.

    Own pairs: smallest decoded matrix byte per probe input (first OWN_CHECKS
    inputs), re-measured directly. Sign pairs: the first SIGN_CHECKS own pairs
    with the negative code. Cross pairs: the own matrix byte of PROBE_INPUTS[j]
    measured under PROBE_INPUTS[(j + CROSS_STRIDE) % len(PROBE_INPUTS)], with
    the expectation taken from that input's decode (empty or not).
    """
    plan: list[dict[str, Any]] = []
    own: list[tuple[int, tuple[int, int, int | None], int, list[int]]] = []
    for j in range(OWN_CHECKS):
        key = PROBE_INPUTS[j]
        decode = decode_by_input[key]
        matrix_byte = min(decode.values())
        expected = preimages(decode)[matrix_byte]
        plan.append(dict(name=f"direct-own-j{j}-{key_name(key)}-m{matrix_byte}", kind="direct",
                         sub="own", impulse=key, matrix=("single", key[0], matrix_byte, E4M3_POS),
                         code=E4M3_POS, expected=expected))
        own.append((j, key, matrix_byte, expected))
    for j in range(SIGN_CHECKS):
        index, key, matrix_byte, expected = own[j]
        plan.append(dict(name=f"direct-sign-j{j}-{key_name(key)}-m{matrix_byte}", kind="direct",
                         sub="sign", impulse=key, matrix=("single", key[0], matrix_byte, E4M3_NEG),
                         code=E4M3_NEG, expected=expected))
    for j in range(CROSS_CHECKS):
        key1 = PROBE_INPUTS[j]
        key2 = PROBE_INPUTS[(j + CROSS_STRIDE) % len(PROBE_INPUTS)]
        matrix_byte = min(decode_by_input[key1].values())
        expected = preimages(decode_by_input[key2]).get(matrix_byte, [])
        plan.append(dict(name=f"direct-cross-j{j}-{key_name(key1)}-{key_name(key2)}-m{matrix_byte}",
                         kind="direct", sub="cross", impulse=key2,
                         matrix=("single", key2[0], matrix_byte, E4M3_POS),
                         code=E4M3_POS, expected=expected, source=key1))
    return plan


def self_test() -> dict[str, Any]:
    regression = sweep.self_test()
    # Builder known answers: main-byte placement and value.
    single = main_byte((0, 0, 0))
    base.require(single[0] == E4M3_POS and single[1] == 0, "main byte value")
    other = main_byte((3, 12, 5))
    other_offset = 3 * MAIN_PARTITION_BYTES + 12 * IMPULSE_BYTES + 5
    base.require(differing_ranges(single, other) == [(0, 1), (other_offset, other_offset + 1)],
                 "main byte placement known answer")
    unit16 = main_byte((0, 0, None))
    base.require(unit16[:IMPULSE_BYTES] == bytes([E4M3_POS]) * IMPULSE_BYTES
                 and unit16[IMPULSE_BYTES] == 0, "main unit16 placement")
    # Signature builder: exact coded pattern and isolation.
    sig0 = weights_signature(0, 0)
    zero = weights_single(0, 0, E4M3_ZERO)
    coded_offsets = expand_ranges([[start, end] for start, end in differing_ranges(zero, sig0)],
                                  launch.WEIGHTS_BYTES)
    base.require(coded_offsets == {offset for offset in range(PARTITION_BYTES) if offset & 1},
                 "signature bit 0 coded exactly the odd offsets")
    matrix0 = sig0[:MATRIX_BYTES]
    base.require(matrix0[0] == E4M3_ZERO and matrix0[1] == E4M3_POS
                 and matrix0[2] == E4M3_ZERO, "signature bit 0 pattern")
    base.require(matrix0[PARTITION_BYTES] == E4M3_ZERO, "signature bit 0 bounded to partition 0")
    sig16 = weights_signature(0, 16)
    matrix16 = sig16[:MATRIX_BYTES]
    base.require(matrix16[65535] == E4M3_ZERO and matrix16[65536] == E4M3_POS
                 and matrix16[PARTITION_BYTES - 1] == E4M3_POS, "signature bit 16 pattern")
    coded = sum(1 for byte in matrix16[:PARTITION_BYTES] if byte == E4M3_POS)
    base.require(coded == PARTITION_BYTES // 2, "signature half-coded")
    matrix_q1 = weights_signature(1, 0)[:MATRIX_BYTES]
    base.require(matrix_q1[0] == E4M3_ZERO and matrix_q1[PARTITION_BYTES] == E4M3_ZERO
                 and matrix_q1[PARTITION_BYTES + 1] == E4M3_POS, "signature partition 1 pattern")
    # Single-byte builder isolation.
    one = weights_single(2, 4711, E4M3_POS)
    one_offset = 2 * PARTITION_BYTES + 4711
    base.require(differing_ranges(zero, one) == [(one_offset, one_offset + 1)], "single byte isolation")
    # Range expansion known answers.
    base.require(expand_ranges([[0, 3], [5, 7]], 16) == {0, 1, 2, 5, 6}, "expand ranges")
    base.require(expand_ranges([], 16) == set(), "expand empty ranges")
    # Decode known answers and round-trip.
    presence = {0: {10, 20}, 1: {20}, 2: set(), 3: {10}}
    decoded = decode_column(presence)
    base.require(decoded == {10: 9, 20: 3}, "decode known answer")
    base.require(preimages(decoded) == {3: [20], 9: [10]}, "preimage known answer")
    rebuilt = {offset: {bit for bit, changed in presence.items() if offset in changed}
               for offset in decoded}
    base.require(all(decoded[offset] == sum(1 << bit for bit in rebuilt[offset])
                     for offset in decoded), "decode round-trip")
    # Decode corruption controls: empty presence and vacuous decode are rejected.
    decode_rejections = 0
    for bad in ({}, {0: set()}, {-1: {1}}, {SIGNATURE_BITS: {1}}, {0: "x"}):
        try:
            decode_column(bad)  # type: ignore[arg-type]
        except ValueError:
            pass
        except Exception as error:  # noqa: BLE001 - a non-ValueError crash is not a clean rejection
            raise ValueError(f"bad presence {bad} rejected by unexpected {type(error).__name__}: {error}")
        else:
            raise ValueError(f"bad presence accepted: {bad}")
        decode_rejections += 1
    # Sign-mirror known answer, mismatch detection, and structural rejection.
    pos = bytes([0x38, 0x38, 0x38])
    neg = bytes([0xB8, 0xB8, 0xB8])
    matched, samples = mirror_samples(pos, neg, {0, 1, 2})
    base.require(matched and samples == [[0, 0x38, 0xB8], [1, 0x38, 0xB8], [2, 0x38, 0xB8]],
                 "sign mirror known answer")
    mismatched, _ = mirror_samples(pos, bytes([0xB8, 0x38, 0xB8]), {0, 1, 2})
    base.require(not mismatched, "sign mirror mismatch detected")
    mirrored, _ = mirror_samples(pos, bytes([0xB8, 0x38, 0xB8]), {0, 2})
    base.require(mirrored, "sign mirror subset accepted")
    try:
        mirror_samples(pos, neg, "x")  # type: ignore[arg-type]
    except Exception:
        pass
    else:
        raise ValueError("bad mirror offsets accepted")
    # Phase-1 plan invariants.
    plan = phase1_plan()
    names = [entry["name"] for entry in plan]
    base.require(len(names) == len(set(names)), "unique phase-1 names")
    base.require(plan[0]["name"] == "base", "base entry first")
    signatures = [entry for entry in plan if entry["kind"] == "signature"]
    base.require(len(signatures) == len(PROBE_INPUTS) * SIGNATURE_BITS, "signature coverage")
    base.require(len([e for e in plan if e["kind"] == "impulse-base"]) == len(PROBE_INPUTS),
                 "impulse-base coverage")
    base.require({"coupled-ref", "decoupled-ctrl"} <= set(names), "control entries present")
    base.require(all(entry["impulse"] is None or input_key(*entry["impulse"]) == entry["impulse"]
                     for entry in plan), "impulse keys valid")
    # Phase-2 selection rule known answers on a synthetic decode.
    synthetic = {key: {offset: 1000 + index * 3 + (offset % 7)
                       for offset in range(16)} for index, key in enumerate(PROBE_INPUTS)}
    for key, decode in synthetic.items():
        base.require(min(decode.values()) == 1000 + PROBE_INPUTS.index(key) * 3,
                     "synthetic decode minimum")
    direct = phase2_plan(synthetic)
    base.require(len(direct) == OWN_CHECKS + SIGN_CHECKS + CROSS_CHECKS, "direct plan size")
    base.require([e["sub"] for e in direct].count("own") == OWN_CHECKS, "own checks")
    base.require([e["sub"] for e in direct].count("sign") == SIGN_CHECKS, "sign checks")
    base.require([e["sub"] for e in direct].count("cross") == CROSS_CHECKS, "cross checks")
    cross = [e for e in direct if e["sub"] == "cross"]
    base.require(all(e["impulse"] == PROBE_INPUTS[(j + CROSS_STRIDE) % len(PROBE_INPUTS)]
                     for j, e in enumerate(cross)), "cross pairing rule")
    base.require(all(e["code"] == E4M3_NEG for e in direct if e["sub"] == "sign"), "sign codes")
    return dict(launch_regression=regression.get("launch_regression", regression),
                sweep_regression=True, probe_inputs=len(PROBE_INPUTS),
                main_builder_known_answers=3, signature_builder_known_answers=6,
                single_builder_known_answers=1, expand_known_answers=2,
                decode_known_answers=2, decode_round_trips=len(decoded),
                decode_rejections=decode_rejections,
                sign_mirror_known_answers=2, sign_mirror_controls=2,
                phase1_entries=len(plan), signature_entries=len(signatures),
                direct_entries=len(direct), gpu_kernels_launched=0)


def execute(data: bytes, inventory: dict[str, Any], mode: str) -> dict[str, Any]:
    entries = [e for e in inventory["fatbins"][6]["entries"] if e["kind"] == 2 and e["arch"] == 89]
    base.require(len(entries) == 1, "unique SM89 decoder module")
    image = base.decompress(data, entries[0])
    utility.verify_kernel_abi(image, RESET, 16)
    abi = decoder.parameter_contract(kernel_metadata(image, NAME))
    base.require(abi["bytes"] == 80, "decoder ABI size")
    base.require(decoder.launch_geometry(*INPUT_AXES)[2] == 4, "four partitions required")
    driver = utility.Driver()
    device, version = utility.select_device(driver)
    launches = readbacks = 0
    guard_words = [0x12340000 + i for i in range(GUARD_WORDS)]
    poison = launch.OUT0_PATTERN
    poison_window = bytes([poison]) * OUT0_WINDOW
    poison_full = bytes([poison]) * launch.OUT0_BYTES
    records: list[dict[str, Any]] = []

    def fail(message: str) -> None:
        print(json.dumps(dict(failure=message, records=records), indent=2))
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

            def matrix_upload(spec: tuple[Any, ...] | None) -> bytes:
                if spec is None:
                    return sweep.weights_subblock(None, E4M3_ZERO, 0, 0)
                kind = spec[0]
                if kind == "whole":
                    return sweep.weights_subblock(spec[1], spec[2], 0, PARTITION_BYTES)
                if kind == "sig":
                    return weights_signature(spec[1], spec[2])
                if kind == "single":
                    return weights_single(spec[1], spec[2], spec[3])
                fail(f"unknown matrix spec {spec}")
                raise AssertionError("unreachable")

            def run_entry(entry: dict[str, Any], base_region: bytes | None) -> tuple[dict[str, Any], bytes, set[int]]:
                nonlocal launches, readbacks
                name = entry["name"]
                key = entry["impulse"]
                snapshots = {
                    "main": upload("main", main_byte(key) if key is not None else bytes(launch.LARGE_BYTES)),
                    "skip": upload("skip", bytes(launch.LARGE_BYTES)),
                    "weights": upload("weights", matrix_upload(entry["matrix"])),
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
                observed, counters_intact = read_counters()
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
                if base_region is None:
                    ranges = []
                else:
                    ranges = differing_ranges(base_region[:OUT0_WINDOW], out0_window)
                changed = expand_ranges([[start, end] for start, end in ranges], OUT0_WINDOW)
                if changed:
                    first, last = min(changed), max(changed)
                else:
                    first, last = -1, -1
                sample = sorted(changed)[:8]
                negative = [offset for offset in sorted(changed) if out0_window[offset] & 0x80]
                negative_ranges: list[list[int]] = []
                for offset in negative:
                    if negative_ranges and negative_ranges[-1][1] == offset:
                        negative_ranges[-1][1] = offset + 1
                    else:
                        negative_ranges.append([offset, offset + 1])
                record = dict(name=name, kind=entry["kind"], sub=entry.get("sub"),
                              impulse_partition=None if key is None else key[0],
                              impulse_unit=None if key is None else key[1],
                              impulse_lane=None if key is None else key[2],
                              matrix_spec=None if entry["matrix"] is None else list(entry["matrix"]),
                              code_byte=entry["code"],
                              source=None if entry.get("source") is None else list(entry["source"]),
                              counters_observed=observed,
                              counters_guard_intact=bool(counters_intact),
                              changed_ranges=[[start, end] for start, end in ranges],
                              changed_count=len(changed), changed_first=first, changed_last=last,
                              negative_ranges=negative_ranges, negative_count=len(negative),
                              positive_sign_all=(len(negative) == 0),
                              sample_offsets=sample,
                              sample_bytes=[out0_window[index] for index in sample],
                              sample_values=[finite(e4m3(out0_window[index])) for index in sample],
                              expected_offsets=entry.get("expected"),
                              guards_intact=bool(guards_intact), immutable=bool(immutable),
                              prediction_match=None if entry.get("expected") is None
                              else (sorted(changed) == sorted(entry["expected"])))
                records.append(record)
                # Invariant checks run after the record is retained so a failure
                # preserves the measurements that explain it (evidence-first).
                if changed and last >= EXTENT_BYTES:
                    fail(f"output write outside the established extent ({name}): {last}")
                if not counters_intact:
                    fail(f"counter guard violation ({name})")
                if observed != [PROTOCOL_FINAL] * launch.COUNTER_WORDS:
                    fail(f"counter protocol mismatch ({name}): {observed}")
                if not (immutable and guards_intact):
                    fail(f"guard/canary/immutability violation ({name})")
                return record, out0_window, changed

            if mode == "skip":
                initial = {key: upload(key, content) for key, content in
                           (("main", main_byte(PROBE_INPUTS[0])), ("skip", bytes(launch.LARGE_BYTES)),
                            ("weights", sweep.weights_subblock(None, E4M3_ZERO, 0, 0)),
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
                            readbacks=readbacks, counters_observed=observed,
                            counter_reset_value=RESET_FINAL, out0_changed=False,
                            all_guards_intact=guards_intact, immutable_regions_checked=list(initial),
                            negative_control_rejected=True,
                            learned_weights_used=False, native_image_inference=False,
                            matrix_permutation_recovered=False,
                            deliberate_proprietary_payload_files_written=False)

            # Phase 1: base capture, impulse bases, controls, signature runs.
            # One full-region upload per run establishes the run-start poison and
            # BOTH guard zones; per-entry traffic then touches only the window.
            upload("out0", poison_full)
            base_region: bytes | None = None
            phase1 = phase1_plan()
            response_values: dict[tuple[int, int, int | None], dict[int, tuple[int, str]]] = {}
            inconsistencies: list[list[Any]] = []
            response_checks = 0
            for entry in phase1:
                record, out0_window, changed = run_entry(entry, base_region)
                if entry["name"] == "base":
                    full, full_intact = download("out0")
                    readbacks += 1
                    if not full_intact:
                        fail("base full-region guard violation")
                    if full[:EXTENT_BYTES] != bytes(EXTENT_BYTES):
                        first_nz = next(index for index in range(EXTENT_BYTES) if full[index] != 0)
                        fail(f"base output not zero in extent (first non-zero byte {first_nz})")
                    if full[EXTENT_BYTES:OUT0_WINDOW] != poison_window[EXTENT_BYTES:]:
                        fail("base writes inside the measurement-window margin")
                    if full[OUT0_WINDOW:] != poison_full[OUT0_WINDOW:]:
                        fail("base writes outside the measurement window")
                    base_region = full
                    continue
                if base_region is None:
                    fail("base region not captured")
                if entry["kind"] == "impulse-base" and changed:
                    fail(f"impulse-only leakage ({entry['name']})")
                if entry["kind"] == "signature":
                    key = entry["impulse"]
                    seen = response_values.setdefault(key, {})
                    for offset in changed:
                        value = out0_window[offset]
                        response_checks += 1
                        prior = seen.get(offset)
                        if prior is None:
                            seen[offset] = (value, entry["name"])
                        elif prior[0] != value and len(inconsistencies) < 16:
                            inconsistencies.append([key_name(key), offset, prior[1], prior[0],
                                                    entry["name"], value])
                if entry["name"] == "coupled-ref":
                    units = sorted({offset // IMPULSE_BYTES * IMPULSE_BYTES for offset in changed})
                    byte_offsets = sorted({offset % IMPULSE_BYTES for offset in changed})
                    if units != canonical_units():
                        fail("coupled-ref footprint is not the canonical set")
                    if byte_offsets != [0, 1, 2, 3, 8, 9, 10, 11]:
                        fail(f"coupled-ref unit byte pattern changed: {byte_offsets}")
                if entry["name"] == "decoupled-ctrl" and changed:
                    fail("decoupling rejected: decoupled-ctrl changed the output")

            assert base_region is not None
            # Single-feeder premise: a responding output byte must return the
            # identical FP8 value in every responding signature run of its input
            # byte (constant coded byte, constant input byte). A multi-feeder
            # sum would differ where the feeders' bit patterns disagree.
            if inconsistencies:
                fail(f"response-value inconsistency across signature runs "
                     f"({len(inconsistencies)}+ output bytes; e.g. {inconsistencies[:3]})")
            # Column decode from the signature runs.
            decode_by_input: dict[tuple[int, int, int | None], dict[int, int]] = {}
            for key in PROBE_INPUTS:
                try:
                    presence: dict[int, set[int]] = {}
                    for record in records:
                        if record["kind"] != "signature":
                            continue
                        if (record["impulse_partition"], record["impulse_unit"],
                                record["impulse_lane"]) != key:
                            continue
                        bit = record["matrix_spec"][2]
                        presence[bit] = expand_ranges(record["changed_ranges"], OUT0_WINDOW)
                    base.require(len(presence) == SIGNATURE_BITS, "signature runs present for input")
                    decode_by_input[key] = decode_column(presence)
                except (ValueError, TypeError) as error:
                    fail(f"decode failed for {key_name(key)}: {error}")

            # Phase 2: direct checks selected from the decode.
            windows: dict[str, bytes] = {}
            changed_by_name: dict[str, set[int]] = {}
            value_mismatches: list[list[Any]] = []
            value_checks = 0
            direct_plan = phase2_plan(decode_by_input)
            for entry in direct_plan:
                record, out0_window, changed = run_entry(entry, base_region)
                windows[entry["name"]] = out0_window
                changed_by_name[entry["name"]] = changed
                if record["prediction_match"] is not True:
                    fail(f"direct prediction mismatch ({entry['name']}): "
                         f"expected {entry['expected']}, observed {sorted(changed)}")
                # A direct run codes exactly one matrix byte, so its response at
                # each changed byte is a true single product; it must equal the
                # signature-run response for the same (input, output) byte,
                # sign-adjusted for negative codes.
                flip = 0x80 if entry["code"] == E4M3_NEG else 0
                prior_by_offset = response_values.get(entry["impulse"], {})
                for offset in sorted(changed):
                    value_checks += 1
                    prior = prior_by_offset.get(offset)
                    if prior is None or (out0_window[offset] ^ flip) != prior[0]:
                        if len(value_mismatches) < 16:
                            value_mismatches.append([entry["name"], offset,
                                                     None if prior is None else prior[0],
                                                     out0_window[offset]])

            if value_mismatches:
                fail(f"direct/signature response-value disagreement "
                     f"({len(value_mismatches)}+ bytes; e.g. {value_mismatches[:3]})")

            # Sign-mirror checks against their own pairs.
            own_records = [entry for entry in direct_plan if entry["sub"] == "own"]
            sign_records = [entry for entry in direct_plan if entry["sub"] == "sign"]
            mirror_passed = True
            for sign_entry, own_entry in zip(sign_records, own_records):
                matched, samples = mirror_samples(windows[own_entry["name"]],
                                                  windows[sign_entry["name"]],
                                                  changed_by_name[own_entry["name"]])
                mirror_passed &= matched
                for record in records:
                    if record["name"] == sign_entry["name"]:
                        record["mirror_samples"] = samples
                        record["mirror_match"] = bool(matched)
            if not mirror_passed:
                fail("sign-mirror mismatch between positive and negative direct pairs")

            # Final full-region check: everything beyond the per-entry window is
            # still the run-start poison; guard zones intact.
            final_full, final_intact = download("out0")
            readbacks += 1
            if not final_intact:
                fail("final out0 guard violation")
            if final_full[OUT0_WINDOW:] != poison_full[OUT0_WINDOW:]:
                fail("out0 written outside the measurement window during the run")

        # Aggregate findings.
        column_records = []
        total_triples = 0
        all_matrix_bytes: dict[tuple[int, int, int | None], set[int]] = {}
        shared_within_input = 0
        for key in PROBE_INPUTS:
            decode = decode_by_input[key]
            preimage = preimages(decode)
            shared_within_input += sum(1 for offsets in preimage.values() if len(offsets) > 1)
            total_triples += len(decode)
            all_matrix_bytes[key] = set(decode.values())
            histogram: dict[str, int] = {}
            byte_histogram: dict[int, int] = {}
            for offset in decode:
                cell = factor_cell(offset // IMPULSE_BYTES * IMPULSE_BYTES)
                label = f"{cell['row_term']},{cell['column']},{cell['channel_block']}"
                histogram[label] = histogram.get(label, 0) + 1
                byte_histogram[offset % IMPULSE_BYTES] = byte_histogram.get(offset % IMPULSE_BYTES, 0) + 1
            column_records.append(dict(impulse_partition=key[0], impulse_unit=key[1], impulse_lane=key[2],
                                       influenced_outputs=len(decode),
                                       distinct_matrix_bytes=len(set(decode.values())),
                                       matrix_byte_min=min(decode.values()),
                                       matrix_byte_max=max(decode.values()),
                                       shared_matrix_bytes=sum(1 for offsets in preimage.values()
                                                               if len(offsets) > 1),
                                       cell_histogram=dict(sorted(histogram.items())),
                                       byte_in_unit_histogram=dict(sorted(byte_histogram.items())),
                                       decode=[[offset, matrix_byte]
                                               for offset, matrix_byte in sorted(decode.items())]))
        overlaps: list[list[Any]] = []
        keys = list(PROBE_INPUTS)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                shared = all_matrix_bytes[keys[i]] & all_matrix_bytes[keys[j]]
                if shared:
                    overlaps.append([key_name(keys[i]), key_name(keys[j]), len(shared),
                                     sorted(shared)[:8]])
        direct_records = [record for record in records if record["kind"] == "direct"]
        direct_ok = all(record["prediction_match"] is True for record in direct_records)
        return dict(native=True, device="NVIDIA GeForce RTX 4070 Ti SUPER",
                    driver_api_version=version, mode=mode, kernel=NAME,
                    block=list(BLOCK), kernels_launched=launches, readbacks=readbacks,
                    signature_bits=SIGNATURE_BITS, probe_inputs=len(PROBE_INPUTS),
                    phase1_entries=len(phase1), phase2_entries=len(direct_records),
                    records=records, column_records=column_records,
                    permutation_triples_resolved=total_triples,
                    permutation_input_bytes_probed=len(PROBE_INPUTS),
                    matrix_bytes_shared_across_inputs=len(overlaps),
                    matrix_byte_overlaps=overlaps,
                    matrix_bytes_shared_within_input=shared_within_input,
                    direct_validation_passed=bool(direct_ok),
                    sign_mirror_passed=bool(mirror_passed),
                    response_value_constant=True, response_value_checks=response_checks,
                    direct_value_agrees=True, direct_value_checks=value_checks,
                    impulse_leakage_detected=False,
                    coupled_ref_enforced=True, decoupling_enforced=True,
                    out0_full_region_final_check=True,
                    matrix_permutation_recovered=False,
                    learned_weights_used=False, native_image_inference=False,
                    deliberate_proprietary_payload_files_written=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("self-test", "execute", "skip"))
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
