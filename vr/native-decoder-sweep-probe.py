#!/usr/bin/env python3
"""Fine-grained influence sweep for the decoder; NOT neural inference.

Launches the pinned SM89 `cc_dec_input_upsample_1024_512_fp8` on native Linux
CUDA and sweeps two axes at finer granularity than the whole-partition coding of
`native-decoder-weights-probe.py`:

  * Matrix sub-blocks: code only part of the coupled matrix partition and see
    which output cells respond (weight -> output mapping).
  * Impulse positions: move the single 16-byte main-input impulse within its
    partition and see which output cells respond (input -> output mapping).

Every matrix-axis footprint (and the coupled reference) must be a subset of the
locked canonical 512-unit set (NATIVE-DECODER-NAMING.md); swept impulse positions
deliberately reach non-canonical output cells, which is part of the finding. This
resolves bounded slices of the (o, i) reach structure; it does NOT recover the
full permutation, use learned weights, or produce a neural-rendered image. No
Windows code, DirectX or Proton runs.
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


weights = load_helper("weights", "native-decoder-weights-probe.py")
launch = weights.launch
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

changed_summary = weights.changed_summary
e4m3 = weights.e4m3
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
PARTITIONS = weights.PARTITIONS
VARIANT = "f32-one"

# Canonical influenced-unit set (hypothesis-independent factorisation):
#   unit_offset = row_term*8192 + column*512 + channel_block*16
#   row_term in {0,4}; column in 0..15; channel_block in {0..7, 16..23}
ROW_TERMS = (0, 4)
COLUMNS = tuple(range(16))
CHANNEL_BLOCKS = tuple(list(range(0, 8)) + list(range(16, 24)))


def canonical_units() -> list[int]:
    """Locked 512-unit byte-offset set (impulse at unit 0); matches NAMING."""
    units = sorted({rt * 8192 + col * 512 + blk * 16
                    for rt in ROW_TERMS for col in COLUMNS for blk in CHANNEL_BLOCKS})
    return units


def factor_cell(unit_offset: int) -> dict[str, int]:
    """General (row_term, column, channel_block) for any 16-byte-aligned unit in extent.

    Unlike the locked canonical set (impulse unit 0), a swept impulse position
    reaches other output cells, so naming must accept any unit inside the
    observed output extent — not only canonical members.
    """
    base.require(type(unit_offset) is int and unit_offset >= 0, "unit offset int")
    base.require(unit_offset % IMPULSE_BYTES == 0, "unit offset aligned")
    base.require(unit_offset < EXTENT_BYTES, "unit offset inside output extent")
    index = unit_offset // IMPULSE_BYTES
    channel_block = index % 32
    column = (index // 32) % 16
    row_term = index // 512
    return dict(row_term=row_term, column=column, channel_block=channel_block)


def is_canonical(unit_offset: int) -> bool:
    return unit_offset in set(canonical_units())


def reach_class(impulse_unit: int) -> tuple[int, int]:
    """Measured input->output reach shift for an impulse unit (row_shift, block_shift).

    Verified against the sweep: the impulse unit's bit 3 selects a +1 row shift
    and bit 2 a +8 channel-block shift of the 512-cell output footprint. This
    describes the reach-class (which output block an input position fans out
    to); it is not the full (o, i) permutation.
    """
    base.require(type(impulse_unit) is int and impulse_unit >= 0, "impulse unit int")
    row_shift = (impulse_unit >> 3) & 1
    block_shift = ((impulse_unit >> 2) & 1) * 8
    return row_shift, block_shift


def require_canonical_subset(label: str, unit_offsets: list[int]) -> list[dict[str, int]]:
    """Matrix-axis invariant: every influenced unit must be inside the canonical
    unit-0 set (coding fewer weights can only influence fewer cells). Returns
    factored cells. Does NOT apply to swept impulse positions.
    """
    canonical = set(canonical_units())
    cells: list[dict[str, int]] = []
    for offset in unit_offsets:
        base.require(type(offset) is int, "unit offset type")
        base.require(offset in canonical, f"{label}: unit {offset} outside canonical set")
        cells.append(factor_cell(offset))
    return cells


def weights_subblock(partition: int | None, code: int, sub_offset: int, sub_len: int) -> bytes:
    """Weights region: word overlay plus a single coded matrix sub-block."""
    base.require(type(code) is int and 0 <= code <= 255, "code byte")
    base.require(type(sub_offset) is int and sub_offset >= 0, "sub_offset")
    base.require(type(sub_len) is int and sub_len >= 0, "sub_len")
    if partition is not None:
        base.require(type(partition) is int and 0 <= partition < 4, "partition range")
        base.require(sub_len > 0, "coded sub-block length")
        base.require(sub_offset + sub_len <= PARTITION_BYTES, "sub-block inside partition")
    import struct as _struct
    word = _struct.pack("<I", weights.F32_ONE)
    region = bytearray((word * (launch.WEIGHTS_BYTES // 4))[:launch.WEIGHTS_BYTES])
    matrix = bytearray(MATRIX_BYTES)
    if partition is not None:
        start = partition * PARTITION_BYTES + sub_offset
        matrix[start:start + sub_len] = bytes([code]) * sub_len
    region[:MATRIX_BYTES] = matrix
    return bytes(region)


def main_at(partition: int, unit: int) -> bytes:
    """Zero main input with one 16-byte impulse unit at (partition, unit)."""
    base.require(type(partition) is int and partition in PARTITIONS, "impulse partition")
    base.require(type(unit) is int and 0 <= unit < MAIN_PARTITION_BYTES // IMPULSE_BYTES, "impulse unit")
    region = bytearray(launch.LARGE_BYTES)
    start = partition * MAIN_PARTITION_BYTES + unit * IMPULSE_BYTES
    region[start:start + IMPULSE_BYTES] = bytes([E4M3_POS]) * IMPULSE_BYTES
    return bytes(region)


# Sweep sample points (bounded, spanning each domain at increasing stride).
MATRIX_SUB_OFFSETS_512 = (0, 512, 1024, 2048, 4096, 8192, 16384, 32768, 49152, 65536, 98304, 130560)
MATRIX_SUB_OFFSETS_16 = (0, 16, 256, 4096, 16384, 49152, 81920, 131056)
IMPULSE_UNITS = (0, 1, 2, 3, 4, 5, 8, 12, 24, 32, 128, 255)


def sweep_plan() -> list[dict[str, Any]]:
    """Ordered (name, main config, weights config) sweep; deterministic."""
    plan: list[dict[str, Any]] = []
    # Reference + controls (impulse at partition 0, unit 0).
    plan.append(dict(name="base", axis="control", impulse=(0, 0), matrix=None,
                     sub_offset=0, sub_len=0, code=E4M3_ZERO))
    plan.append(dict(name="coupled-ref", axis="control", impulse=(0, 0), matrix=0,
                     sub_offset=0, sub_len=PARTITION_BYTES, code=E4M3_POS))
    plan.append(dict(name="decoupled-ctrl", axis="control", impulse=(0, 0), matrix=1,
                     sub_offset=0, sub_len=PARTITION_BYTES, code=E4M3_POS))
    # Axis B (coarse): matrix sub-blocks of 512 bytes inside coupled partition 0.
    for off in MATRIX_SUB_OFFSETS_512:
        plan.append(dict(name=f"matrix-512@{off}", axis="matrix", impulse=(0, 0), matrix=0,
                         sub_offset=off, sub_len=512, code=E4M3_POS))
    # Axis B (fine): matrix sub-blocks of 16 bytes.
    for off in MATRIX_SUB_OFFSETS_16:
        plan.append(dict(name=f"matrix-16@{off}", axis="matrix", impulse=(0, 0), matrix=0,
                         sub_offset=off, sub_len=16, code=E4M3_POS))
    # Axis A: impulse positions within partition 0, whole coupled partition coded.
    for unit in IMPULSE_UNITS:
        plan.append(dict(name=f"impulse-unit@{unit}", axis="impulse", impulse=(0, unit), matrix=0,
                         sub_offset=0, sub_len=PARTITION_BYTES, code=E4M3_POS))
    return plan


def self_test() -> dict[str, Any]:
    regression = weights.self_test()
    units = canonical_units()
    base.require(len(units) == 512 and len(set(units)) == 512, "canonical set size")
    base.require(units[0] == 0 and units[-1] == 40816, "canonical set span")
    # Factorisation known answers (general; canonical and non-canonical).
    base.require(factor_cell(0) == dict(row_term=0, column=0, channel_block=0), "factor 0")
    base.require(factor_cell(40816) == dict(row_term=4, column=15, channel_block=23), "factor max canonical")
    base.require(factor_cell(32768) == dict(row_term=4, column=0, channel_block=0), "factor row4")
    base.require(factor_cell(512) == dict(row_term=0, column=1, channel_block=0), "factor col1")
    base.require(factor_cell(128) == dict(row_term=0, column=0, channel_block=8), "factor non-canonical block8")
    base.require(is_canonical(0) and is_canonical(40816) and not is_canonical(128)
                 and not is_canonical(40832), "canonical membership known answers")
    # Reach-class bit model known answers (row_shift, block_shift).
    reach_answers = {0: (0, 0), 1: (0, 0), 2: (0, 0), 3: (0, 0), 32: (0, 0), 128: (0, 0),
                     4: (0, 8), 5: (0, 8), 8: (1, 0), 24: (1, 0), 12: (1, 8), 255: (1, 8)}
    for unit, expected in reach_answers.items():
        base.require(reach_class(unit) == expected, f"reach-class known answer {unit}")
    # The four reach-classes are distinct and cover the (row_shift, block_shift) grid.
    base.require(len({reach_class(u) for u in reach_answers}) == 4, "four reach-classes")
    base.require({reach_class(u) for u in reach_answers} == {(0, 0), (0, 8), (1, 0), (1, 8)},
                 "reach-class grid coverage")
    # Round-trip: every canonical unit factors and re-composes to itself.
    for unit in units:
        cell = factor_cell(unit)
        recomposed = cell["row_term"] * 8192 + cell["column"] * 512 + cell["channel_block"] * 16
        base.require(recomposed == unit, f"factor round-trip {unit}")
    # Canonical-subset checker accepts canonical members, rejects non-members.
    require_canonical_subset("subset-ok", [units[0], units[7], units[-1]])
    subset_rejections = 0
    for bad in (40832, 40960, 128, -16):
        try:
            require_canonical_subset("subset-bad", [bad])
        except Exception:
            pass
        else:
            raise ValueError(f"non-canonical unit accepted: {bad}")
        subset_rejections += 1
    # Misaligned / out-of-extent offsets are rejected by factor_cell itself.
    factor_rejections = 0
    for bad in (8, 15, -16, EXTENT_BYTES):
        try:
            factor_cell(bad)
        except Exception:
            pass
        else:
            raise ValueError(f"bad factor offset accepted: {bad}")
        factor_rejections += 1
    # Sweep plan: control invariants and full span coverage.
    plan = sweep_plan()
    names = [entry["name"] for entry in plan]
    base.require(len(names) == len(set(names)), "unique sweep names")
    base.require({"base", "coupled-ref", "decoupled-ctrl"} <= set(names), "control entries present")
    matrix_entries = [e for e in plan if e["axis"] == "matrix"]
    impulse_entries = [e for e in plan if e["axis"] == "impulse"]
    base.require(len(matrix_entries) == len(MATRIX_SUB_OFFSETS_512) + len(MATRIX_SUB_OFFSETS_16),
                 "matrix sweep coverage")
    base.require(len(impulse_entries) == len(IMPULSE_UNITS), "impulse sweep coverage")
    base.require(all(e["sub_offset"] + e["sub_len"] <= PARTITION_BYTES for e in matrix_entries),
                 "matrix sub-blocks bounded")
    base.require(all(e["impulse"][1] < MAIN_PARTITION_BYTES // IMPULSE_BYTES for e in plan),
                 "impulse units bounded")
    # Upload builders: isolation and non-vacuity known answers.
    import struct as _struct
    zeros = weights_subblock(None, E4M3_ZERO, 0, 0)
    coded = weights_subblock(0, E4M3_POS, 0, 512)
    diff = weights.differing_ranges(zeros, coded)
    base.require(diff == [(0, 512)], "512-byte sub-block isolation")
    fine = weights_subblock(0, E4M3_POS, 16, 16)
    base.require(weights.differing_ranges(zeros, fine) == [(16, 32)], "16-byte sub-block isolation")
    cross = weights_subblock(1, E4M3_POS, 0, PARTITION_BYTES)
    base.require(weights.differing_ranges(zeros, cross) == [(PARTITION_BYTES, 2 * PARTITION_BYTES)],
                 "partition-1 sub-block isolation")
    impulse0 = main_at(0, 0)
    impulse_last = main_at(0, 255)
    base.require(weights.differing_ranges(impulse0, impulse_last) ==
                 [(0, IMPULSE_BYTES),
                  (255 * IMPULSE_BYTES, 255 * IMPULSE_BYTES + IMPULSE_BYTES)],
                 "impulse sweep placement known answer")
    base.require(impulse0[:IMPULSE_BYTES] == bytes([E4M3_POS]) * IMPULSE_BYTES, "impulse value")
    # changed_summary subset plumbing: a canonical-subset footprint is accepted,
    # and its factored cells match the expected decomposition.
    before = bytes(launch.CHUNK)
    after = bytearray(launch.CHUNK)
    sample_units = [units[0], units[5], units[511]]
    for unit in sample_units:
        for offset in (0, 1, 2, 3, 8, 9, 10, 11):
            after[unit + offset] = 0x38
    summary = changed_summary(before, bytes(after))
    base.require(summary["unit_offsets"] == sample_units, "sweep unit offsets known answer")
    cells = require_canonical_subset("sweep-cells", summary["unit_offsets"])
    base.require(cells == [factor_cell(u) for u in sample_units], "sweep cells known answer")
    # Comparator oracle.
    require = weights.require_same_output
    require("sweep-oracle-identity", zeros[:4096], zeros[:4096])
    perturbed = bytearray(zeros[:4096])
    perturbed[17] = 0x38
    try:
        require("sweep-oracle-perturbed", zeros[:4096], bytes(perturbed))
    except OutputMismatch:
        pass
    else:
        raise ValueError("perturbed outputs accepted")
    return dict(launch_regression=regression.get("launch_regression", regression),
                weights_regression=True, canonical_units=len(units),
                factor_known_answers=5, canonical_membership_known_answers=4,
                reach_class_known_answers=len(reach_answers),
                factor_round_trips=len(units),
                subset_rejections=subset_rejections, factor_rejections=factor_rejections,
                sweep_plan_entries=len(plan),
                matrix_sweep_entries=len(matrix_entries), impulse_sweep_entries=len(impulse_entries),
                upload_isolation_checks=5, sweep_cell_known_answers=len(sample_units),
                comparator_oracle_controls=2, gpu_kernels_launched=0)


def execute(data: bytes, inventory: dict[str, Any], mode: str) -> dict[str, Any]:
    entries = [e for e in inventory["fatbins"][6]["entries"] if e["kind"] == 2 and e["arch"] == 89]
    base.require(len(entries) == 1, "unique SM89 decoder module")
    image = base.decompress(data, entries[0])
    utility.verify_kernel_abi(image, RESET, 16)
    abi = decoder.parameter_contract(kernel_metadata(image, NAME))
    base.require(abi["bytes"] == 80, "decoder ABI size")
    base.require(decoder.launch_geometry(*INPUT_AXES)[2] == 4, "four partitions required")
    canonical = canonical_units()
    driver = utility.Driver()
    device, version = utility.select_device(driver)
    launches = readbacks = 0
    guard_words = [0x12340000 + i for i in range(GUARD_WORDS)]
    poison = launch.OUT0_PATTERN
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
                initial = {key: upload(key, content) for key, content in
                           (("main", main_at(0, 0)), ("skip", bytes(launch.LARGE_BYTES)),
                            ("weights", weights_subblock(None, E4M3_ZERO, 0, 0)),
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
                            matrix_permutation_recovered=False,
                            deliberate_proprietary_payload_files_written=False)

            base_region: bytes | None = None
            for entry in sweep_plan():
                name = entry["name"]
                imp_partition, imp_unit = entry["impulse"]
                snapshots = {
                    "main": upload("main", main_at(imp_partition, imp_unit)),
                    "skip": upload("skip", bytes(launch.LARGE_BYTES)),
                    "weights": upload("weights", weights_subblock(entry["matrix"], entry["code"],
                                                                  entry["sub_offset"], entry["sub_len"])),
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
                    fail(f"counter protocol mismatch ({name}): {observed}")
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
                if name == "base":
                    base_region = out0
                    records.append(dict(name="base", axis="control",
                                        impulse_partition=imp_partition, impulse_unit=imp_unit,
                                        reach_class=None, matrix_partition=entry["matrix"],
                                        sub_offset=entry["sub_offset"], sub_len=entry["sub_len"],
                                        code_byte=entry["code"], counters_observed=observed,
                                        unit_offsets=[], named_cells=[]))
                    if not (immutable and guards_intact):
                        fail("guard/canary/immutability violation (base)")
                    if out0[:EXTENT_BYTES] != bytes(EXTENT_BYTES):
                        first_nz = next(index for index in range(EXTENT_BYTES) if out0[index] != 0)
                        fail(f"base output not zero in extent (first non-zero byte {first_nz})")
                    continue
                if base_region is None:
                    fail("base region not captured")
                summary = changed_summary(base_region, out0)
                # Matrix axis (and the coupled-ref) hold the impulse at unit 0 and
                # code a sub-block of the coupled partition: coding fewer weights can
                # only influence a subset of the canonical unit-0 footprint. The
                # impulse axis sweeps the input position, which reaches other output
                # cells, so only extent/alignment apply and cells are named generally.
                # Cell naming is defensive: a rejection is retained, not raised, so
                # the run keeps the measurements that explain it.
                try:
                    if entry["axis"] == "matrix" or name == "coupled-ref":
                        cells = require_canonical_subset(name, summary["unit_offsets"])
                    else:
                        cells = [factor_cell(u) for u in summary["unit_offsets"]]
                except Exception as error:  # noqa: BLE001 - retain evidence then fail()
                    cells = []
                    cell_error = str(error)
                else:
                    cell_error = None
                record = dict(name=name, axis=entry["axis"],
                              impulse_partition=imp_partition, impulse_unit=imp_unit,
                              reach_class=list(reach_class(imp_unit)) if entry["axis"] == "impulse" else None,
                              matrix_partition=entry["matrix"], sub_offset=entry["sub_offset"],
                              sub_len=entry["sub_len"], code_byte=entry["code"],
                              counters_observed=observed, **summary,
                              named_cells=cells)
                records.append(record)
                # Invariant checks run after the record is retained so a failure
                # preserves the measurements that explain it (evidence-first).
                if cell_error is not None:
                    fail(f"cell naming rejected ({name}): {cell_error}")
                if not (immutable and guards_intact):
                    fail(f"guard/canary/immutability violation ({name})")
                if summary["changed"] and summary["changed_last"] >= EXTENT_BYTES:
                    fail(f"output write outside the observed extent ({name})")
                if name == "coupled-ref":
                    if summary["unit_offsets"] != canonical:
                        fail(f"coupled-ref footprint is not the canonical set ({name})")
                elif name == "decoupled-ctrl":
                    if summary["changed"]:
                        fail(f"decoupling rejected: decoupled-ctrl changed the output")
                # axis entries are measurements; their emptiness/structure is the finding.

    matrix_distinct = sorted({tuple(record["unit_offsets"])
                              for record in records if record["axis"] == "matrix"})
    impulse_distinct = sorted({tuple(record["unit_offsets"])
                               for record in records if record["axis"] == "impulse"})
    impulse_units_seen = sorted({u for record in records if record["axis"] == "impulse"
                                 for u in record["unit_offsets"]})
    return dict(native=True, device="NVIDIA GeForce RTX 4070 Ti SUPER",
                driver_api_version=version, mode=mode, kernel=NAME,
                block=list(BLOCK), kernels_launched=launches, readbacks=readbacks,
                canonical_units=len(canonical), records=records,
                matrix_distinct_footprints=len(matrix_distinct),
                impulse_distinct_footprints=len(impulse_distinct),
                impulse_units_seen=len(impulse_units_seen),
                impulse_reaches_noncanonical=any(not is_canonical(u) for u in impulse_units_seen),
                matrix_subset_enforced=True, coupled_ref_enforced=True, decoupling_enforced=True,
                matrix_permutation_recovered=False, learned_weights_used=False,
                native_image_inference=False,
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
