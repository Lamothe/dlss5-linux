#!/usr/bin/env python3
"""Guarded native decoder launch: protocol/bounds/format, NOT neural inference.

Launches the pinned SM89 `cc_dec_input_upsample_1024_512_fp8` on synthetic
buffers through native Linux CUDA, with no Windows code, DirectX or Proton.
Verifies counter protocol completion, buffer immutability, guard/canary
integrity, and reports the observed output write extent and FP8 pattern
sanity. This is the first native execution of this kernel; it is NOT a
numerical decoder validation and produces no neural-rendered image. Learned
weights are NOT used (synthetic fill only).
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


decoder = load_helper("decoder", "native-decoder-probe.py")
utility = load_helper("utility", "native-kernel-probe.py")
base = utility.base
NAME = "cc_dec_input_upsample_1024_512_fp8"
RESET = "cc_cb_clear"
BLOCK = (32, 2, 1)
INPUT_AXES = (8, 8)          # Padded positive input axes for the guarded run.
OUTPUT_AXES = (16, 16)       # Twice input axes, per the recovered host rule.
GUARD_WORDS = 256            # 1 KiB guard zones around every allocation region.
OUT0_BYTES = 256 * 1024 * 1024   # Generous: output index domain is unverified.
LARGE_BYTES = 4 * 1024 * 1024    # main, skip, partials.
WEIGHTS_BYTES = 8 * 1024 * 1024  # 512 KiB matrix + word region at +0x80000.
SMALL_BYTES = 2 * 1024 * 1024     # Census-unused slots (out3/out2), sized at the
                                  # nominal output extent plus margin so a violated
                                  # census keeps bounded writes inside the allocation
                                  # instead of crossing guard zones.
COUNTER_WORDS = 8            # 2 * rows * columns for 8x8 input.
CHUNK = 8 * 1024 * 1024
OUT0_PATTERN = 0xA5
IMMUTABLE_PATTERN = 0x5A
PARTIAL_PATTERN = 0x33


class DecoderArguments(ct.Structure):
    """Recovered 80-byte argument block; offsets match constant-bank slots."""
    _fields_ = [("main", ct.c_uint64), ("skip", ct.c_uint64), ("out0", ct.c_uint64),
                ("out3", ct.c_uint64), ("counters", ct.c_uint64), ("out2", ct.c_uint64),
                ("partials", ct.c_uint64), ("weights", ct.c_uint64),
                ("dimension", ct.c_uint32), ("width", ct.c_uint32),
                ("output_axis0", ct.c_uint32), ("output_axis1", ct.c_uint32)]


def counter_slots() -> int:
    rows, columns = (axis // 4 for axis in INPUT_AXES)
    base.require(all(axis % 4 == 0 for axis in INPUT_AXES), "unpadded input axes")
    return 2 * rows * columns


def reset_geometry(count: int) -> tuple[int, int, int]:
    base.require(type(count) is int and 0 <= count <= 0x7fffffff - 255, "invalid reset count")
    return ((count + 255) // 256, 1, 1)


def scan_changes(before: bytes, after: bytes) -> tuple[int, int, int]:
    """Return (first, last, changed_chunks) of differing bytes, or (-1, -1, 0).

    Chunk comparisons are memcmp-speed; only boundary chunks are scanned bytewise
    so large written regions stay cheap to measure.
    """
    base.require(len(before) == len(after) and len(before) % CHUNK == 0 and len(before) > 0,
                 "scan size mismatch")
    bad = [offset for offset in range(0, len(before), CHUNK)
           if before[offset:offset + CHUNK] != after[offset:offset + CHUNK]]
    if not bad:
        return -1, -1, 0
    first = bad[0] + next(index for index in range(CHUNK)
                          if before[bad[0] + index] != after[bad[0] + index])
    last = bad[-1] + next(index for index in reversed(range(CHUNK))
                          if before[bad[-1] + index] != after[bad[-1] + index])
    return first, last, len(bad)


def kernel_metadata(image: bytes, name: str) -> bytes:
    """Extract one kernel's .nv.info section from the SM89 cubin ELF."""
    table = base.unpack("<Q", image, 40)[0]
    stride, count, names_index = base.unpack("<HHH", image, 58)
    base.require(stride == 64 and 0 < count <= 4096 and names_index < count, "bad ELF sections")
    sections = [base.unpack("<IIQQQQIIQQ", image, table + index * stride) for index in range(count)]
    strings = base.span(image, sections[names_index][4], sections[names_index][5])
    matches = [section for section in sections
               if base.string_at(strings, section[0]) == ".nv.info." + name]
    base.require(len(matches) == 1, "missing unique kernel metadata")
    return base.span(image, matches[0][4], matches[0][5])


def self_test() -> dict[str, Any]:
    base.require(ct.sizeof(DecoderArguments) == 80, "decoder host ABI size")
    offsets = {name: getattr(DecoderArguments, name).offset for name, _ in DecoderArguments._fields_}
    base.require(offsets == dict(main=0, skip=8, out0=16, out3=24, counters=32, out2=40,
                                 partials=48, weights=56, dimension=64, width=68,
                                 output_axis0=72, output_axis1=76), "decoder host ABI offsets")
    base.require(decoder.launch_geometry(8, 8) == (4, 2, 4), "launch geometry 8x8")
    base.require(decoder.launch_geometry(4, 4) == (2, 1, 4), "launch geometry 4x4")
    base.require(decoder.launch_geometry(16, 20) == (10, 4, 4), "launch geometry 16x20")
    base.require(counter_slots() == COUNTER_WORDS, "counter slot count")
    base.require(reset_geometry(8) == (1, 1, 1), "reset geometry")
    base.require(reset_geometry(257) == (2, 1, 1), "reset geometry rounding")
    pattern = b"\x5a" * CHUNK
    base.require(scan_changes(pattern, pattern) == (-1, -1, 0), "unchanged scan")
    mutated = bytearray(pattern * 2)
    mutated[7] = 0x00
    mutated[CHUNK + 5] = 0x01
    base.require(scan_changes(pattern * 2, bytes(mutated)) == (7, CHUNK + 5, 2), "changed scan")
    later = bytearray(pattern * 2)
    later[CHUNK + 9] = 0x01
    base.require(scan_changes(pattern * 2, bytes(later)) == (CHUNK + 9, CHUNK + 9, 1),
                 "later-chunk first-change scan")
    base.require(ct.sizeof(utility.CounterArguments) == 16
                 and utility.CounterArguments.pointer.offset == 0
                 and utility.CounterArguments.count.offset == 8, "counter host ABI")
    # Expected-counter oracle: the wrong final value must be rejected.
    try:
        exact_counters([3] * COUNTER_WORDS, 2)
    except utility.OutputMismatch:
        pass
    else:
        raise ValueError("wrong counter expectation accepted")
    exact_counters([3] * COUNTER_WORDS, 3)
    return dict(abi_fields=len(offsets), geometry_known_answers=3, scan_known_answers=3,
                counter_oracle_controls=1, gpu_kernels_launched=0)


def exact_counters(observed: list[int], expected_final: int) -> None:
    if observed != [expected_final] * len(observed):
        raise utility.OutputMismatch("counter protocol final value mismatch")


def allocate(driver: Any, stack: Any, total_bytes: int) -> tuple[int, int, int]:
    """One device allocation: guard + region + guard; returns (base, region, region bytes)."""
    guard = GUARD_WORDS * 4
    pointer = ct.c_uint64()
    driver.check("cuMemAlloc_v2", ct.byref(pointer), total_bytes + 2 * guard)
    stack.callback(driver.check, "cuMemFree_v2", pointer)
    base.require(bool(pointer.value), "null device allocation")
    return pointer.value, pointer.value + guard, total_bytes


def execute(data: bytes, inventory: dict[str, Any], mode: str) -> dict[str, Any]:
    entries = [e for e in inventory["fatbins"][6]["entries"] if e["kind"] == 2 and e["arch"] == 89]
    base.require(len(entries) == 1, "unique SM89 decoder module")
    image = base.decompress(data, entries[0])
    utility.verify_kernel_abi(image, RESET, 16)
    abi = decoder.parameter_contract(kernel_metadata(image, NAME))
    base.require(abi["bytes"] == 80, "decoder ABI size")
    driver = utility.Driver()
    device, version = utility.select_device(driver)
    launches = readbacks = 0
    guard_words = [0x12340000 + i for i in range(GUARD_WORDS)]
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
            for key, size in (("main", LARGE_BYTES), ("skip", LARGE_BYTES),
                              ("weights", WEIGHTS_BYTES), ("partials", LARGE_BYTES),
                              ("out0", OUT0_BYTES), ("out3", SMALL_BYTES),
                              ("out2", SMALL_BYTES)):
                addresses[key] = allocate(driver, allocations, size)
            counter_base, counter_region, _ = allocate(driver, allocations, COUNTER_WORDS * 4)

            def upload(key: str, pattern: int) -> ct.Array:
                base_address, region, size = addresses[key]
                host = (ct.c_uint8 * (size + 2 * GUARD_WORDS * 4))()
                ct.memset(host, pattern, ct.sizeof(host))
                for index, word in enumerate(guard_words):
                    ct.cast(ct.byref(host, index * 4), ct.POINTER(ct.c_uint32)).contents.value = word
                    ct.cast(ct.byref(host, GUARD_WORDS * 4 + size + index * 4),
                            ct.POINTER(ct.c_uint32)).contents.value = word
                driver.check("cuMemcpyHtoD_v2", ct.c_uint64(base_address), host, ct.sizeof(host))
                return host

            initial = {key: upload(key, pattern) for key, pattern in
                       (("main", IMMUTABLE_PATTERN), ("skip", IMMUTABLE_PATTERN),
                        ("weights", IMMUTABLE_PATTERN), ("partials", PARTIAL_PATTERN),
                        ("out0", OUT0_PATTERN), ("out3", IMMUTABLE_PATTERN),
                        ("out2", IMMUTABLE_PATTERN))}
            counters = (ct.c_uint8 * (COUNTER_WORDS * 4 + 2 * GUARD_WORDS * 4))()
            for index, word in enumerate(guard_words):
                ct.cast(ct.byref(counters, index * 4), ct.POINTER(ct.c_uint32)).contents.value = word
                ct.cast(ct.byref(counters, GUARD_WORDS * 4 + COUNTER_WORDS * 4 + index * 4),
                        ct.POINTER(ct.c_uint32)).contents.value = word
            driver.check("cuMemcpyHtoD_v2", ct.c_uint64(counter_base), counters, ct.sizeof(counters))

            # Reset the counter region through the previously validated reset kernel.
            grid = reset_geometry(COUNTER_WORDS)
            reset_args = utility.CounterArguments(counter_region, COUNTER_WORDS)
            params = (ct.c_void_p * 1)(ct.addressof(reset_args))
            driver.check("cuLaunchKernel", reset_function, *grid, 256, 1, 1, 0, None, params, None)
            launches += 1
            driver.check("cuCtxSynchronize")

            arguments = DecoderArguments(
                main=addresses["main"][1], skip=addresses["skip"][1],
                out0=addresses["out0"][1], out3=addresses["out3"][1],
                counters=counter_region, out2=addresses["out2"][1],
                partials=addresses["partials"][1], weights=addresses["weights"][1],
                dimension=INPUT_AXES[0], width=INPUT_AXES[1],
                output_axis0=OUTPUT_AXES[0], output_axis1=OUTPUT_AXES[1])
            grid_x, grid_y, grid_z = decoder.launch_geometry(*INPUT_AXES)
            if mode == "reduced-partition":
                grid_z = 2  # Partitions 2/3 absent; protocol must be detected incomplete.
            kernel_params = (ct.c_void_p * 1)(ct.addressof(arguments))
            driver.check("cuLaunchKernel", decoder_function, grid_x, grid_y, grid_z,
                         *BLOCK, 0, None, kernel_params, None)
            launches += 1
            driver.check("cuCtxSynchronize")

            # Counters and guards.
            driver.check("cuMemcpyDtoH_v2", counters, ct.c_uint64(counter_base), ct.sizeof(counters))
            readbacks += 1
            counter_values = ct.cast(ct.byref(counters, GUARD_WORDS * 4),
                                     ct.POINTER(ct.c_int32 * COUNTER_WORDS)).contents
            observed = list(counter_values)
            counter_guards = [ct.cast(ct.byref(counters, index * 4),
                                      ct.POINTER(ct.c_uint32)).contents.value
                              for index in range(GUARD_WORDS)]
            counter_guards += [ct.cast(ct.byref(counters, GUARD_WORDS * 4 + COUNTER_WORDS * 4 + index * 4),
                                       ct.POINTER(ct.c_uint32)).contents.value
                               for index in range(GUARD_WORDS)]
            guards_intact = counter_guards == guard_words * 2

            # Immutable and unused-slot regions.
            immutable_intact = True
            out0_canary = True
            for key in ("main", "skip", "weights", "out3", "out2"):
                base_address, region, size = addresses[key]
                host = initial[key]
                current = (ct.c_uint8 * ct.sizeof(host))()
                driver.check("cuMemcpyDtoH_v2", current, ct.c_uint64(base_address), ct.sizeof(host))
                readbacks += 1
                immutable_intact &= bytes(current) == bytes(host)
            for key in ("partials", "out0"):
                base_address, region, size = addresses[key]
                host = initial[key]
                current = (ct.c_uint8 * ct.sizeof(host))()
                driver.check("cuMemcpyDtoH_v2", current, ct.c_uint64(base_address), ct.sizeof(host))
                readbacks += 1
                boundary_intact = (bytes(current[:GUARD_WORDS * 4]) == bytes(host[:GUARD_WORDS * 4]) and
                                   bytes(current[GUARD_WORDS * 4 + size:]) == bytes(host[GUARD_WORDS * 4 + size:]))
                immutable_intact &= boundary_intact
                if key == "out0":
                    out0_canary = boundary_intact
                    region_bytes = bytes(current[GUARD_WORDS * 4:GUARD_WORDS * 4 + size])
                    expected = bytes([OUT0_PATTERN]) * size
                    first, last, changed_chunks = scan_changes(expected, region_bytes)
                    nan_like = region_bytes.count(b"\x7f") + region_bytes.count(b"\xff")
            driver.check("cuCtxSynchronize")

    normal = [3] * COUNTER_WORDS
    # Integrity is enforced, not advisory: guards, canaries and immutability must
    # hold for the run to be accepted. NaN-like counts remain advisory content.
    if not (guards_intact and immutable_intact and out0_canary):
        raise utility.OutputMismatch("guard/canary/immutability violation detected")
    protocol_ok = observed == normal
    negative_rejected = False
    if mode == "reduced-partition":
        # Pin the intended partial-protocol outcome, not merely "not full".
        base.require(observed == [1] * COUNTER_WORDS,
                     "reduced-partition counters not the partial-protocol value")
        base.require((first, last, changed_chunks) == (-1, -1, 0),
                     "reduced-partition launch wrote output")
        negative_rejected = True
    else:
        exact_counters(observed, 3)
    return dict(native=True, device="NVIDIA GeForce RTX 4070 Ti SUPER",
                driver_api_version=version, mode=mode, kernel=NAME,
                launch_grid=[grid_x, grid_y, grid_z], block=list(BLOCK),
                kernels_launched=launches, readbacks=readbacks,
                counter_words=COUNTER_WORDS, counters_observed=observed,
                counter_protocol_final=3, protocol_ok=protocol_ok,
                negative_control_rejected=negative_rejected,
                all_guards_intact=guards_intact and immutable_intact,
                immutable_regions_checked=["main", "skip", "weights", "out3", "out2"],
                out0_change_first=first, out0_change_last=last, out0_changed_chunks=changed_chunks,
                out0_nan_like_bytes=nan_like, out0_canary_intact=out0_canary,
                learned_weights_used=False, native_image_inference=False,
                numerical_decoder_validation=False,
                canaries_do_not_detect_read_only_out_of_bounds=True,
                deliberate_proprietary_payload_files_written=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("self-test", "execute", "reduced-partition"))
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
