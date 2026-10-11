#!/usr/bin/env python3
"""Guarded native shared-counter reset, NOT decoder execution or image inference.

Reads pinned DLL as data; executes only cc_cb_clear's SM89 GPU code. Core dumps
are disabled; swap, driver persistence and memory erasure remain uncontrolled.
No deliberate proprietary payload/disassembly output files are produced.
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
COUNTS = (0, 1, 255, 256, 257, 512, 640)
GUARD_WORDS = 256
FEATURE_WORDS = 193  # Distinct neighbouring feature-region sentinel, not a tensor.


def reset_geometry(prefix_bytes: int) -> tuple[int, tuple[int, int, int]]:
    # Narrow safe domain: host count truncation and count+255 must not overflow.
    base.require(type(prefix_bytes) is int and 0 <= prefix_bytes <= (0x7fffffff - 255) * 4
                 and prefix_bytes % 4 == 0, "invalid counter-prefix domain")
    count = prefix_bytes // 4
    return count, ((count + 255) // 256, 1, 1)


def exact(actual: list[int], expected: list[int]) -> None:
    if actual != expected:
        raise utility.OutputMismatch("counter reset output/guard/feature region mismatch")


def self_test() -> dict[str, Any]:
    base.require(ct.sizeof(utility.CounterArguments) == 16
                 and utility.CounterArguments.pointer.offset == 0
                 and utility.CounterArguments.count.offset == 8, "counter host ABI")
    known = ((0, 0, (0, 1, 1)), (4, 1, (1, 1, 1)),
             (1020, 255, (1, 1, 1)), (1024, 256, (1, 1, 1)),
             (1028, 257, (2, 1, 1)), (2560, 640, (3, 1, 1)))
    for prefix, count, grid in known:
        base.require(reset_geometry(prefix) == (count, grid), "counter known answer")
    rejected = 0
    for bad in (-4, True, 1.0, 1, 0x80000000 * 4):
        try:
            reset_geometry(bad)
        except ValueError:
            rejected += 1
    base.require(rejected == 5, "invalid counter domain accepted")
    wrong = 0
    for actual, expected in (([0] * 257, [-1] * 257),
                             ([-1] * 256 + [123], [-1] * 257),
                             ([-1] * 258, [-1] * 257 + [456])):
        try:
            exact(actual, expected)
        except utility.OutputMismatch:
            wrong += 1
    base.require(wrong == 3, "incorrect counter reset accepted")
    return dict(known_answers=len(known), invalid_domains_rejected=rejected,
                wrong_value_short_reset_and_overwrite_controls_rejected=wrong)


def execute(data: bytes, inventory: dict[str, Any], mode: str) -> dict[str, Any]:
    self_test()
    # Check original host reset dispatch and binder pointer provenance before GPU use.
    code = decoder.graph.disassemble(data, decoder.COUNTER_RESET_RANGES +
                                    ((0x18003e4e0, 0x18003e57c),))
    contract = decoder.counter_reset_contract(code)
    pe = base.PE(data)
    base.require(base.string_at(data, pe.offset(0xb1af0, 12)) == "cc_cb_clear", "reset name")
    base.require(base.unpack("<6Q", data, pe.offset(0xb28e0, 48)) ==
                 (0x180044710, 0x180044b70, 0x180044b60, 0x180044c60,
                  0x180044830, 0x1800449a0), "generic wrapper vtable")
    entries = [e for e in inventory["fatbins"][6]["entries"] if e["kind"] == 2 and e["arch"] == 89]
    base.require(len(entries) == 1, "unique counter SM89 module")
    image = base.decompress(data, entries[0])
    utility.verify_kernel_abi(image, "cc_cb_clear", 16)
    driver = utility.Driver()
    device, version = utility.select_device(driver)
    launches = readbacks = 0
    checks: list[dict[str, Any]] = []
    negative = mode != "execute"
    rejected = False
    counts = (257,) if negative else COUNTS
    with ExitStack() as resources:
        context = ct.c_void_p()
        driver.check("cuCtxCreate_v2", ct.byref(context), 0, device)
        resources.callback(driver.check, "cuCtxDestroy_v2", context)
        base.require(bool(context.value), "null CUDA context")
        module, function = ct.c_void_p(), ct.c_void_p()
        driver.check("cuModuleLoadData", ct.byref(module), ct.create_string_buffer(image))
        resources.callback(driver.check, "cuModuleUnload", module)
        base.require(bool(module.value), "null CUDA module")
        driver.check("cuModuleGetFunction", ct.byref(function), module, b"cc_cb_clear")
        base.require(bool(function.value), "null CUDA function")
        for count in counts:
            _, grid = reset_geometry(count * 4)
            with ExitStack() as allocations:
                # Separate, distinguishable guards on both sides. Feature region follows
                # the counter prefix immediately; it must not be touched by reset.
                prefix_guard = [0x12340000 + i for i in range(GUARD_WORDS)]
                suffix_guard = [0x34560000 + i for i in range(GUARD_WORDS)]
                feature = [0x56780000 + i for i in range(FEATURE_WORDS)]
                total = GUARD_WORDS * 2 + count + FEATURE_WORDS
                pointer = ct.c_uint64()
                driver.check("cuMemAlloc_v2", ct.byref(pointer), total * 4)
                allocations.callback(driver.check, "cuMemFree_v2", pointer)
                base.require(pointer.value != 0, "null device allocation")
                # Reuse the allocation after re-poisoning; detects one-time-only reset.
                for cycle in range(2):
                    initial = prefix_guard + [7654321 + cycle * 97 + i for i in range(count)] + feature + suffix_guard
                    host = (ct.c_int32 * total)(*initial)
                    driver.check("cuMemcpyHtoD_v2", pointer, host, ct.sizeof(host))
                    launched_count = count - 1 if mode == "short-reset" else count
                    if count and mode != "skip-reset":
                        args = utility.CounterArguments(pointer.value + GUARD_WORDS * 4, launched_count)
                        params = (ct.c_void_p * 1)(ct.addressof(args))
                        driver.check("cuLaunchKernel", function, *grid, 256, 1, 1, 0, None, params, None)
                        launches += 1
                    driver.check("cuCtxSynchronize")
                    driver.check("cuMemcpyDtoH_v2", host, pointer, ct.sizeof(host))
                    readbacks += 1
                    actual = list(host)
                    expected = prefix_guard + [-1] * count + feature + suffix_guard
                    if negative:
                        # Must match precisely the intended failure, not arbitrary corruption.
                        intended = initial if mode == "skip-reset" else prefix_guard + [-1] * (count - 1) + [initial[GUARD_WORDS + count - 1]] + feature + suffix_guard
                        exact(actual, intended)
                        try:
                            exact(actual, expected)
                        except utility.OutputMismatch:
                            rejected = True
                        base.require(rejected, "negative reset unexpectedly passed")
                        checks.append(dict(count=count, cycle=cycle, intended_failure_verified=True))
                        break
                    exact(actual, expected)
                    checks.append(dict(count=count, cycle=cycle, checked_counter_words=count,
                                       guard_and_feature_words=GUARD_WORDS * 2 + FEATURE_WORDS,
                                       max_absolute_error=0))
        driver.check("cuCtxSynchronize")
    return dict(native=True, device="NVIDIA GeForce RTX 4070 Ti SUPER", driver_api_version=version,
                mode=mode, kernel="cc_cb_clear", kernels_launched=launches, readbacks=readbacks,
                checks=checks, negative_control_rejected=rejected,
                host_contract=contract, context_destroyed=True, resources_destroyed=True,
                decoder_kernels_launched=0, native_image_inference=False, learned_weights_used=False,
                canaries_do_not_detect_read_only_out_of_bounds=True,
                deliberate_proprietary_payload_files_written=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("self-test", "execute", "skip-reset", "short-reset"))
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
                          cpu_self_test=self_test(), cuda=execute(data, inventory, args.mode)), indent=2))


if __name__ == "__main__":
    main()
