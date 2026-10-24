#!/usr/bin/env python3
"""Bounded native carried-pool replay of the captured slot24 -> slot25 edge.

The vendor DLL is read as data; only its pinned SM89 cubins are launched via
CUDA Driver API. This is a constructed two-kernel replay, not image inference.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import ctypes as ct
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("slot_replay", HERE / "native-slot-replay-probe.py")
if spec is None or spec.loader is None:
    raise RuntimeError("missing native slot replay probe")
slot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(slot)
base, utility, launch, compare, names = slot.base, slot.utility, slot.launch, slot.compare, slot.names
DLL = slot.DLL
CONTRACTS = slot.CONTRACTS
GUARD_WORDS = launch.GUARD_WORDS


def require_chain(plans: dict[int, dict[str, Any]]) -> None:
    first, second = plans[24], plans[25]
    c24, c25 = CONTRACTS[24], CONTRACTS[25]
    base.require(c24["reference_label"] == c25["input_label"], "chain: capture edge")
    base.require(first["pool_size"] == second["pool_size"], "chain: pool size")
    base.require(first["pool_va"] == second["pool_va"], "chain: pool base")
    base.require(first["capture_fnv"]["24"] == second["capture_fnv"]["24"],
                 "chain: intermediate capture digest")
    base.require(first["pool_offsets"][c24["output_word"]] ==
                 second["pool_offsets"][0], "chain: output/input offset")
    base.require(first["record_sha256"] != second["record_sha256"],
                 "chain: expected distinct stage weight records")
    base.require(first["image"] == second["image"], "chain: module image differs")
    base.require(first["pool_offsets"][c24["sync_word"]] ==
                 second["pool_offsets"][7], "chain: sync handoff offset")
    base.require(first["pool_offsets"][c24["sync_word"]] + c24["sync_bytes"] <=
                 first["pool_size"] and
                 second["pool_offsets"][7] + c24["sync_bytes"] <= second["pool_size"],
                 "chain: sync handoff bounds")


def pool_arguments(setup: dict[str, Any], contract: dict[str, Any],
                   pool_alloc: tuple[int, int, int], weight_alloc: tuple[int, int, int]) -> tuple[Any, Any]:
    values = []
    words, offsets = setup["words"], setup["pool_offsets"]
    for index, cls in enumerate(contract["word_classes"]):
        if cls == slot.CLS_POOL:
            values.append(pool_alloc[1] + offsets[index])
        elif cls == slot.CLS_WEIGHTS:
            values.append(weight_alloc[1])
        elif cls == slot.CLS_CONSTANT:
            values.append(words[index])
        else:
            values.append(0)
    args = (ct.c_uint64 * (contract["psize"] // 8))(*values)
    return args, (ct.c_void_p * 1)(ct.addressof(args))


def upload_weight(driver: Any, alloc: tuple[int, int, int], record: bytes,
                  guard: list[int], corrupt: bool = False) -> bytes:
    payload = slot.wrong_rotation(record) if corrupt else record
    content = payload + bytes([slot.EXEC_TAIL]) * slot.TAIL_BYTES
    return compare.upload_region(driver, alloc, content, 0, guard)


def execute(mode: str, out: Path) -> dict[str, Any]:
    base.require(mode in ("execute", "skip-first", "wrong-first"), "chain: mode")
    base.protect_process_memory()
    data = base.read_bounded(DLL)
    dll_hash = hashlib.sha256(data).hexdigest()
    base.require(dll_hash == base.PIN, "input hash does not match the pinned local payload")
    inventory = base.inventory(data)
    plans = {}
    for number in (24, 25):
        print(f"prepare slot {number}", file=sys.stderr, flush=True)
        plans[number] = slot.setup(data, inventory, CONTRACTS[number])
    require_chain(plans)
    print("contracts and captures validated", file=sys.stderr, flush=True)
    print("initialise CUDA driver", file=sys.stderr, flush=True)
    driver = utility.Driver()
    device, version = utility.select_device(driver)
    print("CUDA driver ready", file=sys.stderr, flush=True)
    guard = [0x12340000 + i for i in range(GUARD_WORDS)]
    runs = []
    with ExitStack() as stack:
        print("create CUDA context", file=sys.stderr, flush=True)
        context = ct.c_void_p()
        driver.check("cuCtxCreate_v2", ct.byref(context), 0, device)
        stack.callback(driver.check, "cuCtxDestroy_v2", context)
        module = ct.c_void_p()
        driver.check("cuModuleLoadData", ct.byref(module),
                     ct.create_string_buffer(plans[24]["image"]))
        stack.callback(driver.check, "cuModuleUnload", module)
        functions = {}
        for number in (24, 25):
            function = ct.c_void_p()
            driver.check("cuModuleGetFunction", ct.byref(function), module,
                         CONTRACTS[number]["name_live"].encode())
            functions[number] = function
        for frame in range(slot.FRAMES):
            print(f"chain frame {frame}", file=sys.stderr, flush=True)
            input_pool = plans[24]["capture"][23]["pools"][frame]["data"]
            mid_ref = plans[24]["capture"][24]["pools"][frame]["data"]
            final_ref = plans[25]["capture"][25]["pools"][frame]["data"]
            pool_alloc = launch.allocate(driver, stack, plans[24]["pool_size"])
            weights24 = launch.allocate(driver, stack, len(plans[24]["record"]) + slot.TAIL_BYTES)
            weights25 = launch.allocate(driver, stack, len(plans[25]["record"]) + slot.TAIL_BYTES)
            before24 = upload_weight(driver, weights24, plans[24]["record"], guard)
            before25 = upload_weight(driver, weights25, plans[25]["record"], guard)
            initial = compare.upload_region(driver, pool_alloc, input_pool, 0, guard)
            print(f"frame {frame}: initial pool uploaded", file=sys.stderr, flush=True)
            base.require(compare.region_slice(initial, pool_alloc) == input_pool,
                         "chain: initial pool upload")
            if mode == "skip-first":
                print(f"frame {frame}: skip first stage", file=sys.stderr, flush=True)
                after_first = compare.readback_region(driver, pool_alloc)
            else:
                active_record = plans[24]["record"]
                if mode == "wrong-first":
                    upload_weight(driver, weights24, active_record, guard, corrupt=True)
                args24, params24 = pool_arguments(plans[24], CONTRACTS[24], pool_alloc, weights24)
                driver.check("cuLaunchKernel", functions[24], *plans[24]["grids"]["tilesync"],
                             *plans[24]["block"], plans[24]["smem"], None, params24, None)
                driver.check("cuCtxSynchronize")
                print(f"frame {frame}: stage24 launched", file=sys.stderr, flush=True)
                after_first = compare.readback_region(driver, pool_alloc)
            print(f"frame {frame}: stage24 boundary read", file=sys.stderr, flush=True)
            mid = compare.region_slice(after_first, pool_alloc)
            mid_guard = compare.guards_intact(after_first, guard)
            mid_match = mid == mid_ref
            base.require(mid_guard, "chain: stage24 pool guards")
            if mode == "execute":
                base.require(mid_match, f"chain: stage24 boundary frame {frame}")
            else:
                base.require(not mid_match, f"chain: control unexpectedly matched intermediate frame {frame}")
            if mode == "skip-first":
                after_second = after_first
                final = mid
                final_match = final == final_ref
            else:
                print(f"frame {frame}: launch stage25", file=sys.stderr, flush=True)
                args25, params25 = pool_arguments(plans[25], CONTRACTS[25], pool_alloc, weights25)
                driver.check("cuLaunchKernel", functions[25], *plans[25]["grids"]["chained"],
                             *plans[25]["block"], plans[25]["smem"], None, params25, None)
                driver.check("cuCtxSynchronize")
                after_second = compare.readback_region(driver, pool_alloc)
                final = compare.region_slice(after_second, pool_alloc)
                final_match = final == final_ref
            final_guard = compare.guards_intact(after_second, guard)
            weights_after24 = compare.readback_region(driver, weights24)
            weights_after25 = compare.readback_region(driver, weights25)
            weight_guards_intact = (compare.guards_intact(weights_after24, guard) and
                                    compare.guards_intact(weights_after25, guard))
            weights_unchanged = (compare.region_slice(weights_after24, weights24) ==
                                 compare.region_slice(before24, weights24) and
                                 compare.region_slice(weights_after25, weights25) ==
                                 compare.region_slice(before25, weights25))
            base.require(final_guard and weight_guards_intact,
                         "chain: pool or weight guards changed")
            if mode != "wrong-first":
                base.require(weights_unchanged, "chain: weights changed")
            if mode == "execute":
                base.require(final_match, f"chain: final boundary frame {frame}")
            else:
                base.require(not final_match,
                             f"chain: negative control unexpectedly matched final frame {frame}")
            mid_diffs = (0 if mid_match else compare.diff_stats(mid, mid_ref)[1])
            final_diffs = (0 if final_match else compare.diff_stats(final, final_ref)[1])
            runs.append(dict(frame=frame, input_sha256=hashlib.sha256(input_pool).hexdigest(),
                             intermediate_sha256=hashlib.sha256(mid).hexdigest(),
                             intermediate_reference_sha256=hashlib.sha256(mid_ref).hexdigest(),
                             intermediate_matches=mid_match,
                             intermediate_diff_count=mid_diffs,
                             final_sha256=hashlib.sha256(final).hexdigest(),
                             final_reference_sha256=hashlib.sha256(final_ref).hexdigest(),
                             final_matches=final_match,
                             final_diff_count=final_diffs,
                             pool_guard_intact=final_guard,
                             weight_guards_intact=weight_guards_intact,
                             weights_unchanged=weights_unchanged,
                             weight_mutation_expected=mode == "wrong-first"))
            (out / f"frame{frame}.json").write_text(json.dumps(runs[-1], indent=2))
    return dict(native=True, mode=mode, device="NVIDIA GeForce RTX 4070 Ti SUPER",
                driver_api_version=version, dll_sha256=dll_hash, stages=[24, 25], frames=runs,
                launches=6 if mode != "skip-first" else 0,
                carry_semantics="one device pool per frame; no vendor intermediate re-upload between launches",
                scope="captured pre-decoder stages only; not live-input evaluator or image inference",
                proprietary_payload_published=False, enforced_failures=[])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("execute", "skip-first", "wrong-first"))
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result = execute(args.mode, args.out)
    (args.out / "run.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
