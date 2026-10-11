#!/usr/bin/env python3
"""Influence-map probe for the decoder weight region; NOT neural inference.

Launches the pinned SM89 `cc_dec_input_upsample_1024_512_fp8` on native Linux
CUDA with a single-unit FP8 main-input impulse and region-coded synthetic
weight matrices. It measures which output bytes respond to which matrix
partition and enforces two structural predictions:

1. Decoupling: a matrix partition whose input partition is all zero cannot
   change the output (bit-identical results required).
2. Coupling: at least one coupled matrix partition must change the output.

This is the first stage of pinning the dense matrix permutation. It does not
recover the permutation itself, does not use learned weights and produces no
neural-rendered image. No Windows code, DirectX or Proton runs.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import ctypes as ct
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


launch = load_helper("launch", "native-decoder-launch-probe.py")
base = launch.base
utility = launch.utility
decoder = launch.decoder
OutputMismatch = utility.OutputMismatch
NAME = launch.NAME
RESET = launch.RESET
BLOCK = launch.BLOCK
INPUT_AXES = launch.INPUT_AXES
OUTPUT_AXES = launch.OUTPUT_AXES
DecoderArguments = launch.DecoderArguments
GUARD_WORDS = launch.GUARD_WORDS
scan_changes = launch.scan_changes
allocate = launch.allocate
kernel_metadata = launch.kernel_metadata
reset_geometry = launch.reset_geometry

MATRIX_BYTES = 524_288                      # Slot-56 coefficient region: 1024x512 FP8.
PARTITION_BYTES = MATRIX_BYTES // 4         # Per CTAID.Z partition, from the address audit.
MAIN_PARTITION_BYTES = 4_096                # Per-partition main-input slice, tile 0.
MAIN_TILE_BYTES = 16_384                    # Main-input tile stride (unused tiles stay zero).
IMPULSE_BYTES = 16                          # One 16-byte read unit (slab0/warp0/lane0).
PROTOCOL_FINAL = 3
RESET_FINAL = -1
EXTENT_BYTES = 16 * 16 * 512                # Observed output write extent: [0, 131072).
E4M3_ZERO = 0x00
E4M3_POS = 0x38                             # +1.0
E4M3_NEG = 0xB8                             # -1.0
F32_ONE = 0x3F800000
VARIANTS = ("f32-one", "zeros")
IMPULSE_PARTITIONS = (0, 1, 2, 3)           # Full coverage: every (matrix, input) cell.
PARTITIONS = (0, 1, 2, 3)


def matrix_region(partition: int) -> tuple[int, int]:
    base.require(type(partition) is int and 0 <= partition < 4, "partition range")
    return partition * PARTITION_BYTES, (partition + 1) * PARTITION_BYTES


def experiment_plan(impulse_partition: int) -> dict[str, tuple[int | None, int]]:
    """Per experiment: (coded matrix partition or None, code byte).

    Every matrix partition is coded as `coupled` once (under its own impulse
    partition) and as a decoupled variant under every other impulse partition,
    so every (matrix q, input p) cell is exercised and every matrix partition
    appears in both the coupled and decoupled roles.
    """
    base.require(impulse_partition in PARTITIONS, "impulse partition")
    plan: dict[str, tuple[int | None, int]] = {
        "base": (None, E4M3_ZERO),
        "coupled": (impulse_partition, E4M3_POS),
        "negative": (impulse_partition, E4M3_NEG),
    }
    for partition in PARTITIONS:
        if partition != impulse_partition:
            plan[f"decoupled-q{partition}"] = (partition, E4M3_POS)
    return plan


def weights_upload(variant: str, coded_partition: int | None, code: int) -> bytes:
    """Full weights region: deterministic word pattern overlaid by the coded matrix."""
    base.require(variant in VARIANTS, "variant")
    base.require(type(code) is int and 0 <= code <= 255, "code byte")
    word = struct.pack("<I", F32_ONE) if variant == "f32-one" else b"\x00\x00\x00\x00"
    region = bytearray((word * (launch.WEIGHTS_BYTES // 4))[:launch.WEIGHTS_BYTES])
    matrix = bytearray(MATRIX_BYTES)
    if coded_partition is not None:
        start, end = matrix_region(coded_partition)
        matrix[start:end] = bytes([code]) * (end - start)
    region[:MATRIX_BYTES] = matrix
    return bytes(region)


def main_upload(impulse_partition: int) -> bytes:
    """Zero main input with one 16-byte impulse unit in one partition slice."""
    base.require(impulse_partition in IMPULSE_PARTITIONS, "impulse partition")
    region = bytearray(launch.LARGE_BYTES)
    start = impulse_partition * MAIN_PARTITION_BYTES
    base.require(start + IMPULSE_BYTES <= MAIN_TILE_BYTES, "impulse inside tile 0")
    region[start:start + IMPULSE_BYTES] = bytes([E4M3_POS]) * IMPULSE_BYTES
    return bytes(region)


def differing_ranges(left: bytes, right: bytes) -> list[tuple[int, int]]:
    """Half-open [start, end) ranges where the buffers differ.

    Block comparison first (memcmp speed), then bytewise within differing
    4 KiB blocks, merging ranges across block boundaries.
    """
    base.require(len(left) == len(right) and len(left) > 0, "differing_ranges size")
    block_size = 4096
    ranges: list[tuple[int, int]] = []
    for chunk_start in range(0, len(left), block_size):
        chunk_end = min(chunk_start + block_size, len(left))
        if left[chunk_start:chunk_end] == right[chunk_start:chunk_end]:
            continue
        index = chunk_start
        while index < chunk_end:
            if left[index] == right[index]:
                index += 1
                continue
            start = index
            while index < chunk_end and left[index] != right[index]:
                index += 1
            if ranges and ranges[-1][1] == start:
                ranges[-1] = (ranges[-1][0], index)
            else:
                ranges.append((start, index))
    return ranges


def e4m3(byte: int) -> float:
    """OCP FP8 E4M3 nearest decode (reporting only)."""
    base.require(type(byte) is int and 0 <= byte <= 255, "e4m3 byte")
    sign = -1.0 if byte & 0x80 else 1.0
    exponent = (byte >> 3) & 0xF
    mantissa = byte & 0x7
    if exponent == 0xF and mantissa == 0x7:
        return float("nan")
    if exponent == 0:
        return sign * (mantissa / 8.0) * 2.0 ** -6
    return sign * (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)


def require_same_output(label: str, left: bytes, right: bytes) -> None:
    if left != right:
        first = next(index for index in range(len(left)) if left[index] != right[index])
        raise OutputMismatch(f"{label}: outputs differ at byte {first}")


def self_test() -> dict[str, Any]:
    regression = launch.self_test()
    # Matrix partition ranges tile the coefficient region exactly once.
    covered = bytearray(MATRIX_BYTES)
    for partition in range(4):
        start, end = matrix_region(partition)
        base.require((start, end) == (partition * PARTITION_BYTES, (partition + 1) * PARTITION_BYTES),
                     "matrix region known answer")
        covered[start:end] = b"\x01" * (end - start)
    base.require(bytes(covered) == b"\x01" * MATRIX_BYTES, "matrix partitions tile region")
    base.require(matrix_region(0) == (0, 131072) and matrix_region(3) == (393216, 524288),
                 "matrix region boundary answers")
    # Experiment plans: coupled/decoupled partitions are exactly as predicted,
    # and every matrix partition appears in both the coupled and decoupled roles.
    plan0 = experiment_plan(0)
    base.require(plan0 == {"base": (None, E4M3_ZERO), "coupled": (0, E4M3_POS),
                           "negative": (0, E4M3_NEG), "decoupled-q1": (1, E4M3_POS),
                           "decoupled-q2": (2, E4M3_POS), "decoupled-q3": (3, E4M3_POS)},
                 "experiment plan p=0")
    plan3 = experiment_plan(3)
    base.require(plan3["coupled"] == (3, E4M3_POS) and plan3["decoupled-q0"] == (0, E4M3_POS)
                 and "decoupled-q3" not in plan3, "experiment plan p=3")
    for partition in PARTITIONS:
        coded = {q for q, _ in experiment_plan(partition).values() if q is not None}
        base.require(coded == set(PARTITIONS), f"plan covers every matrix partition (p={partition})")
    # Non-vacuity of the synthetic buffers: intended pairs differ exactly in the
    # intended partition; the identity case is empty.
    zero_weights = weights_upload("f32-one", None, E4M3_ZERO)
    base.require(differing_ranges(zero_weights, zero_weights) == [], "identity weights")
    non_vacuity_pairs = 1
    for partition in PARTITIONS:
        for name, (coded, code) in experiment_plan(partition).items():
            if name == "base":
                continue
            built = weights_upload("f32-one", coded, code)
            base.require(differing_ranges(zero_weights, built) == [matrix_region(coded)],
                         f"weights differ only in coded partition ({name}, p={partition})")
            non_vacuity_pairs += 1
        coupled = weights_upload("f32-one", partition, E4M3_POS)
        negative = weights_upload("f32-one", partition, E4M3_NEG)
        base.require(differing_ranges(coupled, negative) == [matrix_region(partition)],
                     "negative weights differ only in coded partition")
        non_vacuity_pairs += 1
    variant_zero = weights_upload("zeros", None, E4M3_ZERO)
    variant_ranges = differing_ranges(zero_weights, variant_zero)
    base.require(variant_ranges, "variant difference non-vacuous")
    base.require(all(start >= MATRIX_BYTES and end <= launch.WEIGHTS_BYTES
                     for start, end in variant_ranges), "variant isolation beyond the matrix")
    base.require(zero_weights[:MATRIX_BYTES] == variant_zero[:MATRIX_BYTES],
                 "matrix identical across variants")
    # Impulse placement: partitions differ only in their 16-byte unit.
    impulse0, impulse3 = main_upload(0), main_upload(3)
    base.require(differing_ranges(impulse0, main_upload(0)) == [], "identity main")
    base.require(differing_ranges(impulse0, impulse3) == [(0, IMPULSE_BYTES),
                 (3 * MAIN_PARTITION_BYTES, 3 * MAIN_PARTITION_BYTES + IMPULSE_BYTES)],
                 "impulse units known answer")
    base.require(impulse0[0:IMPULSE_BYTES] == bytes([E4M3_POS]) * IMPULSE_BYTES, "impulse value")
    for partition in PARTITIONS:
        impulse = main_upload(partition)
        start = partition * MAIN_PARTITION_BYTES
        base.require(impulse[start:start + IMPULSE_BYTES] == bytes([E4M3_POS]) * IMPULSE_BYTES
                     and impulse[:start] + impulse[start + IMPULSE_BYTES:] == bytes(launch.LARGE_BYTES - IMPULSE_BYTES),
                     f"impulse isolated to its unit (p={partition})")
    # E4M3 decode known answers (reporting helper).
    answers = {0x00: 0.0, 0x38: 1.0, 0xB8: -1.0, 0x30: 0.5, 0x40: 2.0,
               0x7E: 448.0, 0x01: 2.0 ** -6 / 8.0}
    for byte, expected in answers.items():
        base.require(e4m3(byte) == expected, f"e4m3 known answer {byte:#04x}")
    base.require(e4m3(0x7F) != e4m3(0x7F), "e4m3 NaN encoding")
    # Comparator oracle: identical accepted, perturbed rejected.
    require_same_output("oracle-identity", zero_weights[:4096], zero_weights[:4096])
    perturbed = bytearray(zero_weights[:4096])
    perturbed[777] = 0x38
    try:
        require_same_output("oracle-perturbed", zero_weights[:4096], bytes(perturbed))
    except OutputMismatch:
        pass
    else:
        raise ValueError("perturbed outputs accepted")
    return dict(launch_regression=regression, matrix_regions=4, plan_known_answers=3,
                buffer_non_vacuity_pairs=non_vacuity_pairs, impulse_known_answers=3,
                impulse_isolation_checks=len(IMPULSE_PARTITIONS), e4m3_known_answers=8,
                comparator_oracle_controls=2, gpu_kernels_launched=0)


def finite(value: float) -> float | None:
    """Strict-JSON-safe decode: E4M3 NaN encodings report as None."""
    return None if value != value else value


def changed_summary(before: bytes, after: bytes) -> dict[str, Any]:
    """Change footprint of one output region; details bounded to a 4 MiB span."""
    first, last, chunks = scan_changes(before, after)
    if chunks == 0:
        return dict(changed=False, changed_count=0, changed_first=-1, changed_last=-1,
                    changed_units16=0, sample_offsets=[], sample_values=[], truncated=False)
    span = last - first + 1
    if span > 4 * 1024 * 1024:
        return dict(changed=True, changed_count=-1, changed_first=first, changed_last=last,
                    changed_units16=-1, sample_offsets=[], sample_values=[], truncated=True)
    changed = [first + offset for offset in range(span)
               if before[first + offset] != after[first + offset]]
    units = {index // IMPULSE_BYTES for index in changed}
    sample = changed[:8]
    return dict(changed=True, changed_count=len(changed), changed_first=first, changed_last=last,
                changed_units16=len(units), sample_offsets=sample,
                sample_values=[finite(e4m3(after[index])) for index in sample], truncated=False)


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
    groups: list[dict[str, Any]] = []
    experiment_reports: list[dict[str, Any]] = []

    def fail(message: str) -> None:
        """Emit the accumulated evidence, then fail the run.

        The decoupling-rejection, per-group coupling, extent and guard failure
        paths must not discard the measurements that explain the failure.
        """
        print(json.dumps(dict(failure=message, groups=groups,
                              experiments=experiment_reports), indent=2))
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
                """Guard + region + guard upload; returns the region content snapshot."""
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
                """Read back region + guards; returns (region bytes, guards intact)."""
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
                return region_bytes, intact

            def reset_counters() -> None:
                nonlocal launches
                counters = (ct.c_uint8 * (launch.COUNTER_WORDS * 4 + 2 * GUARD_WORDS * 4))()
                for index, word in enumerate(guard_words):
                    ct.cast(ct.byref(counters, index * 4), ct.POINTER(ct.c_uint32)).contents.value = word
                    ct.cast(ct.byref(counters, GUARD_WORDS * 4 + launch.COUNTER_WORDS * 4 + index * 4),
                            ct.POINTER(ct.c_uint32)).contents.value = word
                driver.check("cuMemcpyHtoD_v2", ct.c_uint64(counter_base), counters, ct.sizeof(counters))
                grid = reset_geometry(launch.COUNTER_WORDS)
                reset_args = utility.CounterArguments(counter_region, launch.COUNTER_WORDS)
                params = (ct.c_void_p * 1)(ct.addressof(reset_args))
                driver.check("cuLaunchKernel", reset_function, *grid, 256, 1, 1, 0, None, params, None)
                launches += 1
                driver.check("cuCtxSynchronize")

            def read_counters() -> list[int]:
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
                if not intact:
                    fail("counter guard violation")
                return list(values)

            if mode == "skip":
                # Negative control: reset only, no decoder launch. Nothing moves.
                initial = {key: upload(key, content) for key, content in
                           (("main", main_upload(0)), ("skip", bytes(launch.LARGE_BYTES)),
                            ("weights", weights_upload("f32-one", None, E4M3_ZERO)),
                            ("partials", bytes(launch.LARGE_BYTES)),
                            ("out0", bytes([poison]) * launch.OUT0_BYTES),
                            ("out3", bytes([launch.IMMUTABLE_PATTERN]) * launch.SMALL_BYTES),
                            ("out2", bytes([launch.IMMUTABLE_PATTERN]) * launch.SMALL_BYTES))}
                reset_counters()
                observed = read_counters()
                if observed != [RESET_FINAL] * launch.COUNTER_WORDS:
                    fail(f"skip counters not the reset value: {observed}")
                out0: bytes | None = None
                guards_intact = True
                immutable = True
                for key in ("main", "skip", "weights", "partials", "out0", "out3", "out2"):
                    region_bytes, intact = download(key)
                    readbacks += 1
                    guards_intact &= intact
                    immutable &= region_bytes == initial[key]
                    if key == "out0":
                        out0 = region_bytes
                base.require(out0 is not None, "skip out0 readback")
                first, last, chunks = scan_changes(initial["out0"], out0)
                base.require((first, last, chunks) == (-1, -1, 0), "skip launch changed output")
                base.require(guards_intact and immutable, "skip guards/immutability")
                return dict(native=True, mode=mode, kernel=NAME, kernels_launched=launches,
                            readbacks=readbacks, counters_observed=observed,
                            counter_reset_value=RESET_FINAL, out0_changed=False,
                            all_guards_intact=guards_intact, immutable_regions_checked=list(initial),
                            negative_control_rejected=True,
                            learned_weights_used=False, native_image_inference=False,
                            deliberate_proprietary_payload_files_written=False)

            for variant in VARIANTS:
                for impulse_partition in IMPULSE_PARTITIONS:
                    plan = experiment_plan(impulse_partition)
                    base_region: bytes | None = None
                    decoupled_checked: list[str] = []
                    group: dict[str, Any] = dict(variant=variant, impulse_partition=impulse_partition,
                                                 experiments={})
                    for name, (coded_partition, code) in plan.items():
                        snapshots = {
                            "main": upload("main", main_upload(impulse_partition)),
                            "skip": upload("skip", bytes(launch.LARGE_BYTES)),
                            "weights": upload("weights", weights_upload(variant, coded_partition, code)),
                            "partials": upload("partials", bytes(launch.LARGE_BYTES)),
                            "out0": upload("out0", bytes([poison]) * launch.OUT0_BYTES),
                            "out3": upload("out3", bytes([launch.IMMUTABLE_PATTERN]) * launch.SMALL_BYTES),
                            "out2": upload("out2", bytes([launch.IMMUTABLE_PATTERN]) * launch.SMALL_BYTES),
                        }
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
                        observed = read_counters()
                        if observed != [PROTOCOL_FINAL] * launch.COUNTER_WORDS:
                            fail(f"counter protocol mismatch ({name}, variant {variant}, "
                                 f"p={impulse_partition}): {observed}")
                        immutable = True
                        guards_intact = True
                        for key in ("main", "skip", "weights", "out3", "out2"):
                            region_bytes, intact = download(key)
                            readbacks += 1
                            immutable &= region_bytes == snapshots[key]
                            guards_intact &= intact
                        partials_bytes, partials_intact = download("partials")
                        readbacks += 1
                        guards_intact &= partials_intact
                        out0, out0_intact = download("out0")
                        readbacks += 1
                        guards_intact &= out0_intact
                        summary = changed_summary(snapshots["out0"], out0)
                        record = dict(experiment=name, variant=variant,
                                      impulse_partition=impulse_partition,
                                      coded_partition=coded_partition, code_byte=code,
                                      counters_observed=observed, **summary)
                        experiment_reports.append(record)
                        group["experiments"][name] = record
                        if summary["changed"] and summary["changed_last"] >= EXTENT_BYTES:
                            fail(f"output write outside the observed extent ({name}, "
                                 f"variant {variant}, p={impulse_partition})")
                        if not (immutable and guards_intact):
                            fail(f"guard/canary/immutability violation ({name}, "
                                 f"variant {variant}, p={impulse_partition})")
                        if name == "base":
                            base_region = out0
                            group["base_zero_in_extent"] = out0[:EXTENT_BYTES] == bytes(EXTENT_BYTES)
                        elif name == "coupled":
                            group["coupled_differs"] = out0 != base_region
                            group["coupled_influence"] = changed_summary(base_region, out0)
                        elif name == "negative":
                            group["negative_differs"] = out0 != base_region
                            group["negative_influence"] = changed_summary(base_region, out0)
                        else:  # decoupled-qN: structurally predicted bit-identity with base.
                            if out0 != base_region:
                                first = changed_summary(base_region, out0)["changed_first"]
                                fail(f"decoupling rejected: variant {variant} "
                                     f"p={impulse_partition} {name} differs at byte {first}")
                            decoupled_checked.append(name)
                    base.require(base_region is not None, "base region captured")
                    group["decoupled_checked"] = decoupled_checked
                    group["decoupled_identical"] = len(decoupled_checked) == 3
                    groups.append(group)
                    # Per-group structural results are enforced, not advisory.
                    if not group["base_zero_in_extent"]:
                        fail(f"base output not zero in extent (variant {variant}, p={impulse_partition})")
                    if not group.get("coupled_differs", False):
                        fail(f"coupled partition produced no change (variant {variant}, p={impulse_partition})")
                    if not group["decoupled_identical"]:
                        fail(f"decoupled identity incomplete (variant {variant}, p={impulse_partition})")

    # Comparator non-vacuity on GPU-captured data: a perturbed copy must be rejected.
    perturbed = bytearray(base_region[:4096])
    perturbed[17] = (perturbed[17] + 1) & 0xFF
    try:
        require_same_output("gpu-captured oracle", base_region[:4096], bytes(perturbed))
    except OutputMismatch:
        pass
    else:
        raise ValueError("perturbed GPU output accepted")
    # Coupling is enforced per group above; this aggregate is report-only.
    any_coupling = any(group["coupled_differs"] for group in groups)
    return dict(native=True, device="NVIDIA GeForce RTX 4070 Ti SUPER",
                driver_api_version=version, mode=mode, kernel=NAME,
                block=list(BLOCK), kernels_launched=launches, readbacks=readbacks,
                groups=groups, experiments=experiment_reports,
                decoupling_enforced=True, any_coupling_observed=any_coupling,
                coupling_enforced=True, comparator_oracle_rejected=True,
                output_extent_bytes=EXTENT_BYTES, learned_weights_used=False,
                matrix_permutation_recovered=False, native_image_inference=False,
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
