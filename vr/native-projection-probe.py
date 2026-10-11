#!/usr/bin/env python3
"""Bounded native FP16/FP8 projection tests, NOT full image inference.

Read the pinned DLL as data and load its SM89 cubin, never Windows host code.
Deliberately persist no proprietary payloads; swap/driver persistence is not
controlled. Only full 128-token blocks are tested. Internal ABI is experimental.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from functools import lru_cache
import ctypes as ct
import importlib.util
import json
import math
from pathlib import Path
import struct
from typing import Any

spec = importlib.util.spec_from_file_location("tensor", Path(__file__).with_name("native-tensor-probe.py"))
if spec is None or spec.loader is None:
    raise RuntimeError("missing tensor probe")
tensor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tensor)
utility = tensor.utility
base = utility.base
GUARD_BYTES = 1024
CHANNELS = 1024
WEIGHT_BYTES = CHANNELS * CHANNELS * 2
SHAPES = ((8, 16), (16, 16))


class ProjectionArguments(ct.Structure):
    # Candidate fields: source, scaled residual, output, packed matrix+scales,
    # per-output-tile completion counters, three unused 64-bit slots, H and W.
    _fields_ = [("source", ct.c_uint64), ("residual", ct.c_uint64),
                ("destination", ct.c_uint64), ("weights", ct.c_uint64),
                ("counters", ct.c_uint64), ("unused", ct.c_uint64 * 3),
                ("height", ct.c_int32), ("width", ct.c_int32)]


def half_bits(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", value))[0]


def half(value: float) -> float:
    return struct.unpack("<e", struct.pack("<e", value))[0]


def activation_index(token: int, channel: int) -> int:
    return ((token // 16) * 8192 + tensor.within_tile(token % 16, channel // 2)) * 2 + channel % 2


def source_value(token: int, channel: int) -> float:
    return (token % 7 + 1) / 16 + (channel // 256) / 32


def residual_value(token: int, channel: int) -> float:
    return ((token * 3 + channel * 5) % 31 - 15) / 256


def scale_value(channel: int) -> float:
    return (1.0, 0.5, -1.0, 2.0)[channel % 4]


def coefficients(zero: bool) -> tuple[float, ...]:
    # Each 256-input-channel partition is uniform, so its matrix permutation
    # cannot affect the dot product. This does NOT recover dense weight layout.
    return (0.0,) * 4 if zero else (1 / 1024, 1 / 512, -1 / 1024, 1 / 2048)


def fixtures(height: int, width: int, zero: bool) -> tuple[list[bytes], bytes, bytes]:
    count = height * width
    base.require((height, width) in SHAPES and count % 128 == 0, "unsupported projection shape")
    source = [0] * (count * CHANNELS)
    residual = source.copy()
    expected = source.copy()
    wrong_scale = source.copy()
    coeff = coefficients(zero)
    for token in range(count):
        # Complete dyadic fixtures: each partial sum and merge is FP16-exact.
        projected = 0.0
        for partition, weight in enumerate(coeff):
            partial = half(256 * source_value(token, partition * 256) * weight)
            projected = half(projected + partial)
        for channel in range(CHANNELS):
            i = activation_index(token, channel)
            r = residual_value(token, channel)
            source[i] = half_bits(source_value(token, channel))
            residual[i] = half_bits(r)
            expected[i] = half_bits(half(projected + half(r * scale_value(channel))))
            wrong_scale[i] = half_bits(half(projected + r))
    pack = lambda values: struct.pack("<" + "H" * len(values), *values)
    weights = b"".join(struct.pack("<e", c) * (CHANNELS * 256) for c in coeff)
    weights += pack([half_bits(scale_value(c)) for c in range(CHANNELS)])
    counters = struct.pack("<i", -1) * (8 * (count // 128))
    uploads = [pack(source), pack(residual), b"\x00\x7e" * len(source), weights, counters]
    return uploads, pack(expected), pack(wrong_scale)



def weight_index(output: int, source: int) -> int:
    """Candidate FP16 packing from weight loads and MMA B-fragment coordinates.

    Each 16-input slab spans all output channels; 16x16 tiles are lane-swizzled.
    The formula is independently tested using logical one-hot dot products.
    """
    return ((source // 16) * 16384 + (output // 16) * 256
            + (output % 8) * 32 + (source % 8 // 2) * 8
            + (output % 16 // 8) * 4 + (source % 16 // 8) * 2 + source % 2)


def channel_weight(output: int, source: int, seed: int) -> float:
    # Asymmetric dyadic patterns expose input/output transposition and swizzles.
    a, b, modulus = ((37, 19, 127), (23, 41, 113))[seed]
    return ((output * a + source * b) % modulus - modulus // 2) / 64


def packed_fixtures(batch: int, seed: int, transpose: bool = False) -> tuple[list[bytes], bytes, bytes]:
    base.require(0 <= batch < 4 and seed in (0, 1), "unsupported packed fixture")
    count = 256
    source = [0] * (count * CHANNELS)
    residual = source.copy()
    expected = source.copy()
    transposed = source.copy()
    matrix = [0] * (CHANNELS * CHANNELS)
    for output in range(CHANNELS):
        for channel in range(CHANNELS):
            index = weight_index(channel, output) if transpose else weight_index(output, channel)
            matrix[index] = half_bits(channel_weight(output, channel, seed))
    for token in range(count):
        channel = batch * count + token
        amplitude = (1.0, 0.5, -1.0, 2.0)[token % 4]
        source[activation_index(token, channel)] = half_bits(amplitude)
        for output in range(CHANNELS):
            index = activation_index(token, output)
            # A one-hot dot product is a literal mathematical matrix lookup,
            # independent of the candidate physical weight_index() permutation.
            # Addition to the zero accumulator makes exact zeros positive.
            expected[index] = half_bits(0.0 + amplitude * channel_weight(output, channel, seed))
            transposed[index] = half_bits(0.0 + amplitude * channel_weight(channel, output, seed))
    pack = lambda values: struct.pack("<" + "H" * len(values), *values)
    weights = pack(matrix) + struct.pack("<e", 1.0) * CHANNELS
    uploads = [pack(source), pack(residual), b"\x00\x7e" * len(source), weights,
               struct.pack("<i", -1) * 16]
    return uploads, pack(expected), pack(transposed)


def fp8_value(bits: int) -> float:
    """E4M3 finite encoding, including subnormals; reject the two NaN encodings."""
    base.require(0 <= bits < 256 and bits & 127 != 127, "invalid E4M3 value")
    exponent, fraction = (bits >> 3) & 15, bits & 7
    magnitude = fraction / 512 if exponent == 0 else (1 + fraction / 8) * 2 ** (exponent - 7)
    return -magnitude if bits & 128 else magnitude


def fp8_bits(value: float) -> int:
    base.require(math.isfinite(value), "non-finite E4M3 conversion")
    return fp8_magnitude_bits(min(abs(value), 448.0)) | (128 if math.copysign(1.0, value) < 0 else 0)


@lru_cache(maxsize=16384)
def fp8_magnitude_bits(magnitude: float) -> int:
    """Nearest-even finite E4M3, saturating at 448; preserve signed zero."""
    # Exhaustive reference over 127 positive encodings, independent of bit shifts.
    nearest = min(range(127), key=lambda b: (abs(fp8_value(b) - magnitude), b & 1))
    return nearest


def fp8_weight_index(output: int, source: int) -> int:
    # Byte address from 32-input slabs and m16n8k32 B fragments, NOT FP16 packing.
    return ((source // 32) * 32768 + (output // 16) * 512 + (output % 8) * 64
            + (source % 16 // 4) * 16 + (output % 16 // 8) * 8
            + (source % 32 // 16) * 4 + source % 4)


def fp8_activation_index(token: int, channel: int) -> int:
    # Byte address: 16 positions x 1024 E4M3 channels, four bytes per word.
    return ((token // 16) * 16384 + (channel // 32) * 512 + (token % 16 // 8) * 4
            + (channel % 32 // 16) * 8 + (token % 8) * 64
            + (channel % 16 // 4) * 16 + channel % 4)


def fp8_output_index(token: int, channel: int) -> int:
    # Quantised output/skip layout differs from the MMA input layout.
    return ((token // 16) * 16384 + (channel // 32) * 512 + (token % 16 // 8) * 4
            + (channel % 32 // 16) * 8 + (token % 8) * 64
            + (channel % 8 // 2) * 16 + (channel % 16 // 8) * 2 + channel % 2)


def projection_record(data: bytes, block: int) -> bytes:
    """Bounded candidate record, validated container first; never print its values."""
    base.require(block in range(31, 39), "unsupported candidate block")
    _, resource = base.PE(data).resource("WEIGHTS_HT")
    summary = base.weight_inventory(resource)
    target = f"block{block}.layer4.layer"
    record = next(r for r in summary["records"] if r["name"] == target)
    base.require(record["tag"] == 1 and record["dimensions"] == [525312]
                 and record["payload_bytes"] == 1050624
                 and record["metadata_a"] == record["metadata_b"] == 0,
                 "candidate record metadata changed")
    position = 8
    while position < len(resource):
        length = base.unpack("<Q", resource, position)[0]
        name = base.span(resource, position + 8, length).decode("ascii")
        position += 8 + length
        size = base.unpack("<Q", resource, position)[0]
        position += 8
        if name == target:
            payload = base.span(resource, position + 20, record["payload_bytes"])
            base.require(all(b & 127 != 127 for b in payload[:CHANNELS * CHANNELS]),
                         "candidate matrix contains E4M3 NaN")
            scales = struct.unpack("<1024e", payload[CHANNELS * CHANNELS:])
            base.require(all(math.isfinite(v) for v in scales), "non-finite candidate scales")
            return payload
        position += size
    raise ValueError("candidate record missing")


def fp8_fixtures(batch: int, seed: int, transpose: bool = False,
                 learned: bytes | None = None) -> tuple[list[bytes], bytes, bytes, bytes, bytes]:
    base.require(0 <= batch < 4 and seed in (0, 1), "unsupported FP8 fixture")
    count = 256
    # Distinct asymmetric matrices; choose exactly representable E4M3 values.
    values = (0x00, 0x08, 0x18, 0x28, 0x38, 0x40, 0x88, 0x98, 0xA8, 0xB8, 0xC0)
    matrix = bytearray(CHANNELS * CHANNELS)
    scales = tuple(scale_value(c) for c in range(CHANNELS))
    logical = bytearray(len(matrix))
    if learned is None:
        for output in range(CHANNELS):
            for source in range(CHANNELS):
                logical[output * CHANNELS + source] = values[(output * (37 + seed * 4)
                                                             + source * (19 + seed * 4)) % len(values)]
    else:
        base.require(len(learned) == 1050624, "bad candidate payload length")
        scales = struct.unpack("<1024e", learned[CHANNELS * CHANNELS:])
        for output in range(CHANNELS):
            for source in range(CHANNELS):
                logical[output * CHANNELS + source] = learned[fp8_weight_index(output, source)]
    for output in range(CHANNELS):
        for source in range(CHANNELS):
            physical = fp8_weight_index(source, output) if transpose else fp8_weight_index(output, source)
            matrix[physical] = logical[output * CHANNELS + source]
    if learned is not None and not transpose:
        base.require(matrix == learned[:CHANNELS * CHANNELS], "candidate payload altered")
    src = bytearray(count * CHANNELS)
    residual = bytearray(len(src))
    expected = [0] * len(src)
    wrong = expected.copy()
    output_bytes = bytearray(len(src))
    wrong_bytes = bytearray(len(src))
    for token in range(count):
        source = batch * count + token
        amplitude_bits = (0x38, 0x30, 0xB8, 0x40)[token % 4]  # 1, 0.5, -1, 2
        amplitude = fp8_value(amplitude_bits)
        src[fp8_activation_index(token, source)] = amplitude_bits
        for output in range(CHANNELS):
            residual_bits = (0x28, 0xA8, 0x30, 0x00)[(token + output) % 4]
            residual[fp8_output_index(token, output)] = residual_bits
            r = half(fp8_value(residual_bits) * scales[output])
            index = activation_index(token, output)
            # One-hot product is exact; model scales/residual merge round to FP16.
            expected[index] = half_bits(half(r + half(0.0 + amplitude * fp8_value(logical[output * CHANNELS + source]))))
            wrong[index] = half_bits(half(r + half(0.0 + amplitude * fp8_value(logical[source * CHANNELS + output]))))
            destination = fp8_output_index(token, output)
            output_bytes[destination] = fp8_bits(struct.unpack("<e", struct.pack("<H", expected[index]))[0])
            wrong_bytes[destination] = fp8_bits(struct.unpack("<e", struct.pack("<H", wrong[index]))[0])
            # Scratch holds partitions 0..2; partition 3 merges without storing
            # its final FP16 value back to scratch, then quantises the output.
            if source >= 768:
                expected[index] = wrong[index] = half_bits(0.0 + r)
    pack = lambda v: struct.pack("<" + "H" * len(v), *v)
    weights = bytes(matrix) + struct.pack("<1024e", *scales)
    return ([bytes(src), bytes(residual), b"\x7f" * len(src), weights,
             struct.pack("<i", -1) * 16, b"\x00\x7e" * len(src)],
            bytes(output_bytes), bytes(wrong_bytes), pack(expected), pack(wrong))


def fp8_self_test() -> dict[str, Any]:
    base.require(sorted(fp8_weight_index(o, i) for o in range(CHANNELS) for i in range(CHANNELS))
                 == list(range(CHANNELS * CHANNELS)), "non-bijective FP8 matrix")
    base.require(sorted(fp8_activation_index(t, c) for t in range(256) for c in range(CHANNELS))
                 == list(range(256 * CHANNELS)), "non-bijective FP8 activations")
    base.require(sorted(fp8_output_index(t, c) for t in range(256) for c in range(CHANNELS))
                 == list(range(256 * CHANNELS)), "non-bijective FP8 output")
    for t, c, answer in ((0, 0, 0), (0, 2, 16), (0, 8, 2), (0, 16, 8), (8, 0, 4), (255, 1023, 262143)):
        base.require(fp8_output_index(t, c) == answer, "FP8 output known answer mismatch")
    weight_answers = ((0, 0, 0), (1, 0, 64), (8, 0, 8), (0, 1, 1), (0, 4, 16),
                      (0, 16, 4), (0, 32, 32768), (16, 0, 512), (1023, 1023, 1048575))
    activation_answers = ((0, 0, 0), (1, 0, 64), (8, 0, 4), (16, 0, 16384),
                          (0, 4, 16), (0, 16, 8), (0, 32, 512), (255, 1023, 262143))
    for o, i, answer in weight_answers:
        base.require(fp8_weight_index(o, i) == answer, "FP8 weight known answer mismatch")
    for t, c, answer in activation_answers:
        base.require(fp8_activation_index(t, c) == answer, "FP8 activation known answer mismatch")
    for bits, value in ((0, 0), (1, 1/512), (0x08, 1/64), (0x38, 1), (0xB8, -1), (0x7E, 448)):
        base.require(fp8_value(bits) == value, "E4M3 known answer mismatch")
    for bits in (0x7F, 0xFF, -1, 256):
        try:
            fp8_value(bits)
        except ValueError:
            pass
        else:
            raise ValueError("invalid E4M3 control accepted")
    for value, answer in ((1.0625, 0x38), (1.1875, 0x3A), (500.0, 0x7E),
                          (-0.0, 0x80), (1/1024, 0), (3/1024, 2)):
        base.require(fp8_bits(value) == answer, "E4M3 rounding known answer mismatch")
    return dict(fp8_output_known_answers=6, fp8_output_layout_bijective=True,
                fp8_rounding_known_answers=6, fp8_weight_layout_bijective=True, fp8_activation_layout_bijective=True,
                fp8_weight_known_answers=len(weight_answers), fp8_activation_known_answers=len(activation_answers),
                e4m3_known_answers=6, e4m3_invalid_controls_rejected=4)


def packed_self_test() -> dict[str, Any]:
    indices = [weight_index(o, i) for o in range(CHANNELS) for i in range(CHANNELS)]
    base.require(sorted(indices) == list(range(CHANNELS * CHANNELS)), "non-bijective weight packing")
    # Manually enumerated lane/fragment boundaries, not formula-generated values.
    answers = ((0, 0, 0), (1, 0, 32), (8, 0, 4), (0, 1, 1), (0, 2, 8),
               (0, 8, 2), (0, 16, 16384), (0, 32, 32768), (16, 0, 256),
               (128, 0, 2048), (1023, 1023, 1048575))
    for output, source, physical in answers:
        base.require(weight_index(output, source) == physical, "weight packing known answer mismatch")
    for seed in (0, 1):
        base.require(channel_weight(0, 1, seed) != channel_weight(1, 0, seed), "symmetric control")
    return dict(weight_packing_bijective=True, weight_index_known_answers=len(answers),
                all_input_channels_covered=4 * 256, synthetic_weights_only=True)


def exact(label: str, actual: bytes, expected: bytes, redact: bool = False) -> dict[str, Any]:
    if actual != expected:
        if len(actual) != len(expected):
            raise utility.OutputMismatch(f"{label}: byte length mismatch")
        i = next(i for i, (a, b) in enumerate(zip(actual, expected)) if a != b)
        detail = "values suppressed" if redact else f"{actual[i]:02x} != {expected[i]:02x}"
        raise utility.OutputMismatch(f"{label}: byte {i}: {detail}")
    return dict(operation=label, checked_bytes=len(expected), bit_exact=True)


def self_test() -> dict[str, Any]:
    base.require(ct.sizeof(ProjectionArguments) == 72 and ProjectionArguments.weights.offset == 24
                 and ProjectionArguments.counters.offset == 32 and ProjectionArguments.height.offset == 64
                 and ProjectionArguments.width.offset == 68, "projection host packing mismatch")
    # Fixed dyadic known answers, not generated by the expected-output loop.
    # token0: sum = 1/64 + 3/64 - 1/32 + 5/256 = 13/256.
    uploads, output, wrong = fixtures(8, 16, False)
    exact("literal-projection-answer", output[activation_index(0, 0) * 2:activation_index(0, 0) * 2 + 2],
          struct.pack("<e", -2 / 256))
    exact("literal-scaled-residual-answer", output[activation_index(0, 2) * 2:activation_index(0, 2) * 2 + 2],
          struct.pack("<e", 18 / 256))
    rejected = 0
    for label, bad in (("missing-scale", wrong), ("skipped-output", uploads[2])):
        try:
            exact(label, bad, output)
        except utility.OutputMismatch:
            rejected += 1
        else:
            raise ValueError(f"{label} control passed")
    for height, width in SHAPES:
        n = height * width * CHANNELS
        base.require(sorted(activation_index(t, c) for t in range(height * width)
                            for c in range(CHANNELS)) == list(range(n)), "non-bijective projection tensor")
    return dict(host_argument_bytes=72, known_answers=2, cpu_negative_controls_rejected=rejected,
                activation_layout_bijective=True, synthetic_weights_only=True)


def verify_projection_abi(image: bytes, name: str = "cc_vit_1d_projection") -> None:
    # This entry additionally contains valueless and byte-valued attributes,
    # which the utility-only probe intentionally does not support.
    table = base.unpack("<Q", image, 40)[0]
    stride, count, names_index = base.unpack("<HHH", image, 58)
    base.require(stride == 64 and 0 < count <= 4096 and names_index < count, "bad ELF sections")
    sections = [base.unpack("<IIQQQQIIQQ", image, table + i * stride) for i in range(count)]
    strings = base.span(image, sections[names_index][4], sections[names_index][5])
    matches = [s for s in sections if base.string_at(strings, s[0]) == ".nv.info." + name]
    base.require(len(matches) == 1, "missing unique projection metadata")
    info = base.span(image, matches[0][4], matches[0][5])
    sizes: list[int] = []
    parameters: list[tuple[int, int, int]] = []
    position = 0
    while position < len(info):
        form, attribute, length = base.unpack("<BBH", info, position)
        position += 4
        if form == 3:
            if attribute == 0x19:
                sizes.append(length)
        elif form == 4:
            value = base.span(info, position, length)
            position += length
            if attribute == 0x17:
                base.require(length == 12, "bad projection parameter")
                index, ordinal_offset, packed = struct.unpack("<III", value)
                parameters.append((index, ordinal_offset, packed >> 18))
        else:
            base.require((form, attribute, length) in ((1, 4, 0), (2, 0x4c, 1)),
                         "unexpected projection metadata form")
    base.require(sizes == [72] and parameters == [(0, 0, 72)], "projection opaque ABI mismatch")


def execute(data: bytes, inventory: dict[str, Any], skip: bool, packed: bool = False,
            wrong_layout: bool = False, fp8: bool = False, learned_block: int | None = None) -> dict[str, Any]:
    cpu_checks = self_test()
    if packed:
        cpu_checks.update(packed_self_test())
    if fp8:
        cpu_checks.update(fp8_self_test())
    learned = projection_record(data, learned_block) if learned_block is not None else None
    base.require(learned is None or fp8, "learned candidate requires FP8 mode")
    def checked(label: str, actual: bytes, expected: bytes | None) -> dict[str, Any]:
        base.require(expected is not None, "missing reference")
        return exact(label, actual, expected, redact=learned is not None)

    driver = utility.Driver()
    device, version = utility.select_device(driver)
    checks: list[dict[str, Any]] = []
    launches = readbacks = 0
    rejected = False
    with ExitStack() as stack:
        context = ct.c_void_p()
        driver.check("cuCtxCreate_v2", ct.byref(context), 0, device)
        stack.callback(driver.check, "cuCtxDestroy_v2", context)
        base.require(bool(context.value), "null CUDA context")
        entry = next(e for e in inventory["fatbins"][5]["entries"] if e["kind"] == 2 and e["arch"] == 89)
        image = base.decompress(data, entry)
        kernel_name = "cc_vit_1d_projection_fp8" if fp8 else "cc_vit_1d_projection"
        verify_projection_abi(image, kernel_name)
        module, function = ct.c_void_p(), ct.c_void_p()
        driver.check("cuModuleLoadData", ct.byref(module), ct.create_string_buffer(image))
        stack.callback(driver.check, "cuModuleUnload", module)
        base.require(bool(module.value), "null module")
        driver.check("cuModuleGetFunction", ct.byref(function), module, kernel_name.encode("ascii"))
        base.require(bool(function.value), "null function")
        if packed or fp8:
            cases = ((16, 16, f"packed-seed{seed}-inputs{batch * 256}-{batch * 256 + 255}",
                      batch, seed) for seed in ((0,) if learned is not None else (0, 1)) for batch in range(4))
        else:
            cases = ((h, w, f"{h}x{w}-{'zero' if zero else 'weighted'}", int(zero), 0)
                     for h, w in SHAPES for zero in (False, True))
        for height, width, label, fixture, seed in cases:
            scratch_expected = scratch_wrong = None
            if fp8:
                uploads, expected, wrong, scratch_expected, scratch_wrong = fp8_fixtures(fixture, seed, wrong_layout, learned)
            else:
                uploads, expected, wrong = (packed_fixtures(fixture, seed, wrong_layout) if packed
                                            else fixtures(height, width, bool(fixture)))
            with ExitStack() as allocations:
                pointers: list[ct.c_uint64] = []
                guarded: list[bytes] = []
                for index, uploaded in enumerate(uploads):
                    pre = bytes((i * 13 + index * 37 + 1) % 256 for i in range(GUARD_BYTES))
                    post = bytes((i * 17 + index * 41 + 7) % 256 for i in range(GUARD_BYTES))
                    raw = pre + uploaded + post
                    pointer = ct.c_uint64()
                    driver.check("cuMemAlloc_v2", ct.byref(pointer), len(raw))
                    allocations.callback(driver.check, "cuMemFree_v2", pointer)
                    base.require(pointer.value != 0, "null allocation")
                    driver.check("cuMemcpyHtoD_v2", pointer, ct.create_string_buffer(raw), len(raw))
                    pointers.append(pointer)
                    guarded.append(raw)
                args = ProjectionArguments()
                for field, pointer in zip(("source", "residual", "destination", "weights", "counters"), pointers):
                    setattr(args, field, pointer.value + GUARD_BYTES)
                if fp8:
                    args.unused[0] = pointers[5].value + GUARD_BYTES
                args.height, args.width = height, width
                if not skip:
                    arguments = (ct.c_void_p * 1)(ct.addressof(args))
                    driver.check("cuLaunchKernel", function, 8 * (height * width // 128), 1, 4,
                                 32, 4, 1, 0, None, arguments, None)
                    launches += 1
                    driver.check("cuCtxSynchronize")
                actual: list[bytes] = []
                for pointer, original in zip(pointers, guarded):
                    host = ct.create_string_buffer(len(original))
                    driver.check("cuMemcpyDtoH_v2", host, pointer, len(original))
                    readbacks += 1
                    actual.append(host.raw)
                for index in (0, 1, 3):
                    checks.append(checked(f"{label}-input-{index}-and-guards", actual[index], guarded[index]))
                for index in ((2, 4, 5) if fp8 else (2, 4)):
                    checks.append(checked(f"{label}-pre-{index}", actual[index][:GUARD_BYTES], guarded[index][:GUARD_BYTES]))
                    checks.append(checked(f"{label}-post-{index}", actual[index][-GUARD_BYTES:], guarded[index][-GUARD_BYTES:]))
                output = actual[2][GUARD_BYTES:-GUARD_BYTES]
                counters = actual[4][GUARD_BYTES:-GUARD_BYTES]
                if fp8:
                    scratch = actual[5][GUARD_BYTES:-GUARD_BYTES]
                    if not skip and not wrong_layout and scratch != scratch_expected and learned is None:
                        samples = []
                        for t in range(height * width):
                            for c in range(CHANNELS):
                                i = activation_index(t, c) * 2
                                if scratch[i:i+2] != scratch_expected[i:i+2]:
                                    samples.append(dict(token=t, channel=c,
                                                        actual=struct.unpack("<e", scratch[i:i+2])[0],
                                                        expected=struct.unpack("<e", scratch_expected[i:i+2])[0]))
                                    if len(samples) == 12:
                                        break
                            if len(samples) == 12:
                                break
                        raise utility.OutputMismatch(label + ": " + json.dumps(samples))
                    checks.append(checked(label + "-fp16-scratch", scratch,
                                        uploads[5] if skip else scratch_wrong if wrong_layout else scratch_expected))
                if skip:
                    checks.append(checked("skipped-destination", output, uploads[2]))
                    checks.append(checked("skipped-counters", counters, uploads[4]))
                    try:
                        checked("skipped-projection", output, expected)
                    except utility.OutputMismatch:
                        rejected = True
                    base.require(rejected, "skipped projection accepted")
                elif wrong_layout:
                    checks.append(checked(label + "-transposed-mathematical-reference", output, wrong))
                    checks.append(checked(label + "-completion-counters", counters,
                                        struct.pack("<I", 3) * (len(counters) // 4)))
                    try:
                        checked("wrong-weight-layout", output, expected)
                    except utility.OutputMismatch:
                        rejected = True
                    base.require(rejected, "incorrect weight layout accepted")
                else:
                    if output != expected:
                        if fp8:
                            checked(label + "-fp8-projection", output, expected)
                        samples = []
                        for token in range(height * width):
                            for channel in range(CHANNELS):
                                offset = activation_index(token, channel) * 2
                                if output[offset:offset + 2] != expected[offset:offset + 2]:
                                    samples.append(dict(token=token, channel=channel,
                                                        actual_bits=output[offset:offset + 2].hex(),
                                                        expected_bits=expected[offset:offset + 2].hex()))
                                    if len(samples) == 8:
                                        break
                            if len(samples) == 8:
                                break
                        raise utility.OutputMismatch(json.dumps(dict(case=label, samples=samples if learned is None else "candidate mismatch values suppressed",
                                                                     counters=list(struct.unpack("<" + "I" * (len(counters) // 4), counters)))))
                    checks.append(checked(label + "-projection", output, expected))
                    checks.append(checked(label + "-completion-counters", counters,
                                        struct.pack("<I", 3) * (len(counters) // 4)))
                if skip or wrong_layout:
                    break
        driver.check("cuCtxSynchronize")
    return dict(native=True, device="NVIDIA GeForce RTX 4070 Ti SUPER", driver_api_version=version,
                cpu_checks=cpu_checks, kernels_launched=launches, readbacks=readbacks, checks=checks,
                skipped_projection=skip, skipped_projection_rejected=skip and rejected,
                synthetic_weights_used=learned is None, learned_weights_used=learned is not None, evaluations=0,
                 fp8_projection=fp8, fp16_scratch_verified=fp8 and not skip and not wrong_layout, candidate_record=f"block{learned_block}.layer4.layer" if learned is not None else None,
                 actual_model_layer_linkage_verified=False,
                 fp8_channel_packing_verified=fp8 and not skip and not wrong_layout,
                channel_specific_weight_packing_verified=packed and not skip and not wrong_layout,
                incorrect_weight_layout=wrong_layout,
                incorrect_weight_layout_rejected=wrong_layout and rejected,
                dense_weight_layout_recovered=packed and not skip and not wrong_layout,
                context_destroyed=True, resources_destroyed=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("self-test", "execute", "skip-projection", "execute-packed", "skip-packed", "wrong-packed", "execute-fp8", "skip-fp8", "wrong-fp8", "execute-candidate", "skip-candidate", "wrong-candidate"))
    parser.add_argument("dll", nargs="?", type=Path)
    parser.add_argument("--block", type=int, choices=range(31, 39), default=31)
    args = parser.parse_args()
    if args.mode == "self-test":
        print(json.dumps({**self_test(), **packed_self_test(), **fp8_self_test()}, indent=2))
        return
    base.require(args.dll is not None, "DLL required")
    protection = base.protect_process_memory()
    data = base.read_bounded(args.dll)
    report = execute(data, base.inventory(data), args.mode in ("skip-projection", "skip-packed", "skip-fp8", "skip-candidate"),
                     packed=args.mode in ("execute-packed", "skip-packed", "wrong-packed"),
                     wrong_layout=args.mode in ("wrong-packed", "wrong-fp8", "wrong-candidate"),
                      fp8=args.mode.endswith(("fp8", "candidate")),
                      learned_block=args.block if args.mode.endswith("candidate") else None)
    print(json.dumps(dict(sha256=base.PIN, memory_protection=protection, cuda=report), indent=2))


if __name__ == "__main__":
    main()
