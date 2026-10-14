#!/usr/bin/env python3
"""Native pre-decoder kernel replay: slot 24 against the ladder s24 reference.

Generalises the decoder replay (NATIVE-VENDOR-COMPARE.md) to a pre-decoder
kernel: the vendor's slot-24 launch of the 156-launch inventory
(`cc_split_swin_16h_ffwd_inpview_512_tilesync_fp8`, NVAPI-ARGS.md) is replayed
natively on Linux (CUDA driver; no Windows code, DirectX or Proton) from its
captured parameter block, and the native pool state is compared byte-for-byte
against the vendor's own captured state after slot 24 (the s24 cut point of
LADDER.md).

Inputs come from retained evidence only:

- the launch specification (grid, block, shared memory, 56-byte parameter
  block) is parsed from the forwarder traces' slot-24 records and pinned
  equal across evaluations, runs and the capture sidecar;
- the input pool state is the s23 ladder capture (after slot 23, so exactly
  the state slot 24 reads), the reference pool state is the s24 capture
  (after slot 24; s23->s24 changes exactly slot 24's two named bindings and
  s24->s56 confirms later slots never touch them);
- the weight bytes are extracted from the model DLL's WEIGHTS_HT resource
  through a measured heap layout: record payloads concatenated at 512-byte
  alignment. Two independent anchors pin that layout: the decoder's live
  weights (DECODER-BOUNDARY.md word 7) at `block39.layer0.layer`, and slot
  24's own word 2 exactly at `block23.layer0.layer`'s payload start.

Primary claim (execute mode): for the vendor's own tilesync variant the
native pool readback equals the vendor's s24 pool byte-for-byte over the
whole 62,684,160-byte pool. The fp8 sibling (created but never launched by
the vendor — pinned) is replayed at the same geometry as a variant probe:
its output tensor must match, and its 96-byte sync record at w6 is recorded
per entry (the decoder stage found the same tilesync-only 96-byte record at
out2). Controls: skip (no launch), wrong (weights rotated inwards), tail
(bytes beyond the record replaced by a different pattern — bounding weight
reads to the record).

This stage uses stored learned weight bytes through one extracted record
(bounded model-weight computation, NATIVE-FP8.md precedent) and is not a
neural-rendered image. Evidence retains digests and bounded windows only;
reuse/publication rights remain unresolved.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import ctypes as ct
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import struct
import sys
from typing import Any


HERE = Path(__file__).resolve().parent


def load_helper(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError("missing helper")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


base = load_helper("base", "native-runtime-probe.py")
utility = load_helper("utility", "native-kernel-probe.py")
launch = load_helper("launch", "native-decoder-launch-probe.py")
compare = load_helper("compare", "native-vendor-compare-probe.py")
names = load_helper("names", "ladder-names.py")

# ---------------------------------------------------------------------------
# Pinned constants
# ---------------------------------------------------------------------------

SLOT = 24
CONTAINER = 4
FRAMES = 3
PSIZE = 56
WORDS = PSIZE // 8
NAME_LIVE = "cc_split_swin_16h_ffwd_inpview_512_tilesync_fp8"
NAME_SIBLING = "cc_split_swin_16h_ffwd_inpview_512_fp8"
KERNEL_TAGS = (("tilesync", NAME_LIVE), ("fp8", NAME_SIBLING))
RECORD_NAME = "block23.layer0.layer"
RECORD_BYTES = 524288
ANCHOR_RECORD = "block39.layer0.layer"
HEAP_SIZE = 147719680
OUTPUT_BYTES = 344064   # w1 binding extent: 24x28x512 (LADDER-NAMES.md segment)
SYNC_BYTES = 96         # w6 sync record extent (stage-15 segment)
ROTATE_BYTES = 1024
TAIL_BYTES = 4096
EXEC_TAIL = 0xA5
TAIL_TAIL = 0x5A
GUARD_WORDS = launch.GUARD_WORDS

DLL = HERE.parent / "model" / "nvngx_dlssnr.dll"
DIR_S23 = HERE / "logs" / "ladder" / "s23"
DIR_S24 = HERE / "logs" / "ladder" / "s24"
TRACE_S23 = HERE / "logs" / "capture" / "nvapi-trace-ladder-s23.log"
TRACE_S24 = HERE / "logs" / "capture" / "nvapi-trace-ladder-s24.log"
BOUNDARY_MANIFEST = HERE / "logs" / "boundary" / "a" / "manifest.txt"

CREATED_RE = re.compile(
    r"created .*block=(\d+),(\d+),(\d+) smem=(\d+) flags=(\d+) "
    r"name=(\S+) handle=(0x[0-9a-f]+)")
GRID_RE = re.compile(
    r"launch id=0x5c52bb86 .*handle=(0x[0-9a-f]+) grid=(\d+),(\d+),(\d+)")
BOUNDARY_RE = re.compile(r"^boundary eval=(\d+) psize=(\d+) pmem=([0-9a-f]+)$")
WORD7_RE = re.compile(r"^word=7 .* off=(\d+) ")
RES_RE = re.compile(
    r"^res idx=(\d+) va=0x([0-9a-f]+) size=(\d+) ")

# Mode entry plans: every mode exercises both variants on all frames.
PLAN = tuple(f"{tag}-f{frame}" for frame in range(FRAMES)
             for tag, _ in KERNEL_TAGS)

MODE_TAIL = {"execute": EXEC_TAIL, "tail": TAIL_TAIL, "skip": EXEC_TAIL,
             "wrong": EXEC_TAIL}


# ---------------------------------------------------------------------------
# Evidence parsing
# ---------------------------------------------------------------------------

def weight_layout(weights: bytes) -> dict[str, dict[str, int]]:
    """Walk WEIGHTS_HT; offsets follow the measured heap layout.

    Record payloads are concatenated at 512-byte alignment (the alignment
    model validated by two independent live anchors, see module docstring).
    """
    base.require(base.unpack("<Q", weights, 0)[0] == len(weights),
                 "weights: container size mismatch")
    cursor = 8
    heap = 0
    layout: dict[str, dict[str, int]] = {}
    while cursor < len(weights):
        name_length = base.unpack("<Q", weights, cursor)[0]
        cursor += 8
        base.require(0 < name_length <= 256, "weights: name length")
        name = base.span(weights, cursor, name_length).decode("ascii")
        cursor += name_length
        size = base.unpack("<Q", weights, cursor)[0]
        cursor += 8
        record = base.span(weights, cursor, size)
        cursor += size
        declared, payload, _tag = base.unpack("<QQI", record, 0)
        base.require(declared == size, "weights: record size mismatch")
        base.require(name not in layout, "weights: duplicate name")
        heap = (heap + 511) // 512 * 512
        layout[name] = dict(heap_offset=heap, payload_bytes=payload,
                            payload_at=cursor - size + 20)
        heap += payload
    base.require(cursor == len(weights), "weights: overrun")
    layout["<heap-total>"] = dict(heap_offset=heap)
    return layout


def parameter_metadata(info: bytes) -> dict[str, Any]:
    """Parse bounded CUDA metadata forms; never skip an unknown record form."""
    position = 0
    sizes: list[int] = []
    parameters: list[tuple[int, int, int]] = []
    while position < len(info):
        form, attribute, length = base.unpack("<BBH", info, position)
        position += 4
        if form == 4:
            value = base.span(info, position, length)
            position += length
            if attribute == 0x17:
                base.require(length == 12, "abi: parameter record length")
                index, ordinal, packed = struct.unpack("<III", value)
                parameters.append((index, ordinal, packed >> 18))
        elif form in (1, 2, 3):
            if attribute == 0x19:
                base.require(form == 3, "abi: parameter size requires HVAL")
                sizes.append(length)
        else:
            raise ValueError("abi: unsupported metadata form")
    return dict(sizes=sizes, parameters=parameters)


def read_created(text: str, name: str) -> dict[str, Any]:
    records = [m for m in CREATED_RE.finditer(text) if m.group(6) == name]
    base.require(len(records) == 1, f"created: expected one record for {name}")
    record = records[0]
    return dict(block=(int(record.group(1)), int(record.group(2)), int(record.group(3))),
                smem=int(record.group(4)), flags=int(record.group(5)),
                handle=record.group(7))


def heap_word_scan(trace: dict[str, Any], heap_va: int, heap_size: int,
                   layout: dict[str, dict[str, int]]) -> dict[str, int]:
    """Classify every heap-pointing parameter word: record start or mid-record."""
    starts = {entry["heap_offset"] for name, entry in layout.items()
              if name != "<heap-total>"}
    total = at_start = mid = 0
    for evaluations in trace["evals"]:
        for _handle, psize, _pcap, pmem in evaluations:
            block = bytes.fromhex(pmem)
            for position in range(0, len(block) // 8 * 8, 8):
                word = struct.unpack_from("<Q", block, position)[0]
                if heap_va <= word < heap_va + heap_size:
                    total += 1
                    if word - heap_va in starts:
                        at_start += 1
                    else:
                        mid += 1
    return dict(total=total, at_record_start=at_start, mid_record=mid)


def read_grid(text: str, handle: str, label: str) -> tuple[int, int, int]:
    grids = {(int(m.group(2)), int(m.group(3)), int(m.group(4)))
             for m in GRID_RE.finditer(text) if m.group(1) == handle}
    base.require(grids, f"grid: {label} has no launches")
    base.require(len(grids) == 1, f"grid: {label} launch geometry varies")
    return grids.pop()


def setup(data: bytes, inventory: dict[str, Any]) -> dict[str, Any]:
    """Resolve the launch, bindings and weights from retained evidence."""
    trace_text = {23: TRACE_S23.read_text(encoding="utf-8", errors="replace"),
                  24: TRACE_S24.read_text(encoding="utf-8", errors="replace")}
    traces = {23: names.read_trace(TRACE_S23), 24: names.read_trace(TRACE_S24)}

    # Creation geometry (pinned equal across runs) and per-trace handles.
    created: dict[str, dict[str, Any]] = {}
    handles: dict[int, dict[str, str]] = {23: {}, 24: {}}
    for tag, name in KERNEL_TAGS:
        per_trace = {label: read_created(trace_text[label], name) for label in (23, 24)}
        base.require({key: value for key, value in per_trace[23].items() if key != "handle"} ==
                     {key: value for key, value in per_trace[24].items() if key != "handle"},
                     f"created: {name} geometry differs between runs")
        created[tag] = per_trace[23]
        for label in (23, 24):
            handles[label][tag] = per_trace[label]["handle"]

    # Slot-24 parameter block: the slot's launch must be the tilesync kernel's
    # (identity pinned per trace and frame), a full dump, stable across evals
    # and runs.
    dumps: dict[int, str] = {}
    for label, trace in traces.items():
        pmems = []
        for frame in range(FRAMES):
            handle, psize, _pcap, pmem = trace["evals"][frame][SLOT]
            base.require(handle == handles[label]["tilesync"],
                         "slot24-identity: slot 24 is not the tilesync launch")
            base.require(psize == PSIZE and len(pmem) == 2 * PSIZE,
                         "slot24-dump: truncated parameter dump")
            pmems.append(pmem)
        base.require(len(set(pmems)) == 1, "slot24-dump: parameters vary across evals")
        dumps[label] = pmems[0]
    base.require(dumps[23] == dumps[24], "slot24-dump: parameters differ between runs")
    words = struct.unpack(f"<{WORDS}Q", bytes.fromhex(dumps[23]))
    base.require(words[4] == 0 and words[5] == 0, "slot24-words: reserved word non-zero")

    # Per-run grid, resolved through each trace's own created handle. The
    # vendor launches only the tilesync variant at slot 24; the fp8 sibling is
    # replayed at the same geometry (NATIVE-VENDOR-COMPARE.md precedent).
    grids = {"tilesync": read_grid(trace_text[23], handles[23]["tilesync"], "tilesync")}
    base.require(grids["tilesync"] == read_grid(trace_text[24], handles[24]["tilesync"],
                                                "tilesync"),
                 "grid: tilesync differs between runs")
    grids["fp8"] = grids["tilesync"]
    # The fp8 sibling is created but never launched by the vendor: it has no
    # launch contract of its own (pinned per trace, each with its own handle).
    sibling_launches = sum(1 for label in (23, 24)
                           for m in GRID_RE.finditer(trace_text[label])
                           if m.group(1) == handles[label]["fp8"])
    base.require(sibling_launches == 0, "sibling: unexpected vendor launches")

    # Captures: input state (s23) and reference state (s24).
    capture = {23: names.read_capture(DIR_S23), 24: names.read_capture(DIR_S24)}
    base.require(capture[23]["slot"] == 23 and capture[24]["slot"] == 24,
                 "capture: slot pin")
    base.require(capture[23]["pool_size"] == capture[24]["pool_size"],
                 "capture: pool size differs")
    base.require(capture[23]["pool_va"] == capture[24]["pool_va"],
                 "capture: pool address differs")
    base.require(capture[23]["frames"] == capture[24]["frames"] == FRAMES,
                 "capture: frame count")

    # The capture sidecar carries the same slot-24 block (independent copy).
    sidecar = [BOUNDARY_RE.match(line) for line in
               (DIR_S24 / "boundary-va.txt").read_text().splitlines() if line]
    base.require(all(sidecar) and len(sidecar) == FRAMES, "sidecar: record count")
    base.require(all(m.group(2) == str(PSIZE) for m in sidecar if m),
                 "sidecar: parameter size")
    base.require(all(m.group(3) == dumps[23] for m in sidecar if m),
                 "slot24-dump: sidecar parameters differ")

    # Pool bindings: words 0, 1 and 6 point into the working pool.
    pool_va = capture[23]["pool_va"]
    pool_size = capture[23]["pool_size"]
    offsets = {}
    for index in (0, 1, 6):
        base.require(pool_va <= words[index] < pool_va + pool_size,
                     f"slot24-bind: word {index} outside pool")
        offsets[index] = words[index] - pool_va

    # Word 2 resolves into the weight heap through the captured resource map.
    res_lines = [RES_RE.match(line) for line in
                 (DIR_S24 / "resource-map.txt").read_text().splitlines()
                 if line.startswith("res ")]
    base.require(all(res_lines), "slot24-bind: resource map format")
    containing = [m for m in res_lines if m
                  and int(m.group(2), 16) <= words[2] < int(m.group(2), 16) + int(m.group(3))]
    base.require(len(containing) == 1, "slot24-bind: weight resource not unique")
    weight_res = containing[0]
    base.require(int(weight_res.group(3)) == HEAP_SIZE,
                 "slot24-bind: unexpected weight resource size")
    heap_offset = words[2] - int(weight_res.group(2), 16)

    # Heap layout anchors from the DLL's WEIGHTS_HT resource.
    _resource_offset, weights = base.PE(data).resource("WEIGHTS_HT")
    layout = weight_layout(weights)
    heap_total = layout.pop("<heap-total>")["heap_offset"]
    # The live allocation is the packed heap rounded up (448 bytes of tail
    # padding observed); the exact pins are the two anchors below.
    base.require(0 <= HEAP_SIZE - heap_total < 512,
                 "weight-heap: layout size mismatch")
    base.require(layout[ANCHOR_RECORD]["heap_offset"] ==
                 boundary_word7_offset(), "weight-heap: decoder anchor mismatch")
    base.require(layout[RECORD_NAME]["heap_offset"] == heap_offset,
                 "slot24-weight: record binding mismatch")
    base.require(layout[RECORD_NAME]["payload_bytes"] == RECORD_BYTES,
                 "slot24-weight: record size mismatch")
    # Every heap-pointing parameter word of both traces must address a whole
    # record payload start (the structural claim, computed and enforced).
    heap_scan = {str(label): heap_word_scan(traces[label],
                                            int(weight_res.group(2), 16),
                                            HEAP_SIZE, layout)
                 for label in (23, 24)}
    for label in (23, 24):
        scan = heap_scan[str(label)]
        base.require(scan["mid_record"] == 0 and
                     scan["total"] == scan["at_record_start"],
                     "weight-heap: weight pointer off record start")
    record = base.span(weights, layout[RECORD_NAME]["payload_at"], RECORD_BYTES)
    following = [name for name, entry in layout.items()
                 if entry["heap_offset"] == layout[RECORD_NAME]["heap_offset"] + RECORD_BYTES]
    base.require(len(following) == 1, "slot24-weight: no following record")
    tail_live = base.span(weights, layout[following[0]]["payload_at"],
                          min(TAIL_BYTES, layout[following[0]]["payload_bytes"]))
    base.require(tail_live != bytes([EXEC_TAIL]) * len(tail_live) and
                 tail_live != bytes([TAIL_TAIL]) * len(tail_live),
                 "weight-tail: tail pattern coincides with live bytes")

    # Kernel ABI from the SM89 cubin: one 56-byte parameter.
    modules = [e for e in inventory["fatbins"][CONTAINER]["entries"]
               if e["kind"] == 2 and e["arch"] == 89]
    base.require(len(modules) == 1, "abi: unique SM89 module")
    image = base.decompress(data, modules[0])
    for tag, name in KERNEL_TAGS:
        meta = parameter_metadata(launch.kernel_metadata(image, name))
        base.require(meta["sizes"] == [PSIZE] and meta["parameters"] == [(0, 0, PSIZE)],
                     f"abi: unexpected parameter metadata for {name}")
    base.require(all(created[tag]["smem"] == 0 for tag in dict(KERNEL_TAGS)),
                 "abi: unexpected shared memory")

    return dict(
        words=list(words), pool_offsets=offsets, heap_offset=heap_offset,
        heap_total=heap_total, heap_allocation=HEAP_SIZE,
        sibling_launches=sibling_launches, heap_scan=heap_scan,
        record_sha256=hashlib.sha256(record).hexdigest(),
        block=created["tilesync"]["block"], smem=created["tilesync"]["smem"],
        grids={tag: list(grids[tag]) for tag in dict(KERNEL_TAGS)},
        pool_va=pool_va, pool_size=pool_size,
        trace_sha256={str(label): hashlib.sha256(
            (TRACE_S23 if label == 23 else TRACE_S24).read_bytes()).hexdigest()
            for label in (23, 24)},
        capture_fnv={str(label): {str(p["frame"]): f"0x{p['fnv']:016x}"
                                  for p in capture[label]["pools"]}
                     for label in (23, 24)},
        record=record, image=image, capture=capture)


def boundary_word7_offset() -> int:
    for line in BOUNDARY_MANIFEST.read_text().splitlines():
        if match := WORD7_RE.match(line):
            return int(match.group(1))
    raise ValueError("weight-heap: decoder anchor word 7 missing")


# ---------------------------------------------------------------------------
# GPU replay
# ---------------------------------------------------------------------------

def run_entry(driver: Any, functions: dict[str, ct.c_void_p], plan: dict[str, Any],
              entry_id: str, frame: int, tag: str, mode: str,
              guard_words: list[int], evidence: Path) -> dict[str, Any]:
    setup_data = plan["setup"]
    capture = setup_data["capture"]
    input_pool = capture[23]["pools"][frame]["data"]
    reference = capture[24]["pools"][frame]["data"]
    record = setup_data["record"]
    tail = bytes([MODE_TAIL[mode]]) * TAIL_BYTES
    weights_content = record + tail
    if mode == "wrong":
        weights_content = (record[ROTATE_BYTES:] + record[:ROTATE_BYTES]) + tail

    launches = readbacks = 0
    with ExitStack() as allocations:
        pool_alloc = launch.allocate(driver, allocations, setup_data["pool_size"])
        weight_alloc = launch.allocate(driver, allocations, RECORD_BYTES + TAIL_BYTES)
        initial_pool = compare.upload_region(driver, pool_alloc, input_pool,
                                             0x00, guard_words)
        initial_weights = compare.upload_region(driver, weight_alloc, weights_content,
                                                0x00, guard_words)
        launched = mode != "skip"
        if launched:
            words = setup_data["words"]
            offsets = setup_data["pool_offsets"]
            arguments = (ct.c_uint64 * WORDS)(
                pool_alloc[1] + offsets[0], pool_alloc[1] + offsets[1],
                weight_alloc[1], words[3], 0, 0, pool_alloc[1] + offsets[6])
            kernel_params = (ct.c_void_p * 1)(ct.addressof(arguments))
            grid = setup_data["grids"][tag]
            driver.check("cuLaunchKernel", functions[tag], *grid,
                         *setup_data["block"], setup_data["smem"],
                         None, kernel_params, None)
            launches += 1
            driver.check("cuCtxSynchronize")

        current_pool = compare.readback_region(driver, pool_alloc)
        current_weights = compare.readback_region(driver, weight_alloc)
        readbacks += 2

    pool_region = compare.region_slice(current_pool, pool_alloc)
    weights_region = compare.region_slice(current_weights, weight_alloc)
    input_region = compare.region_slice(initial_pool, pool_alloc)

    # Evidence first: a failing comparison must not discard its explanation.
    retained: dict[str, str] = {}
    match, diff_count, first_diff = compare.diff_stats(reference, pool_region)
    if not match:
        window_start = max(0, first_diff - 4096)
        retained["pool-window"] = f"{entry_id}-pool-window.bin"
        (evidence / retained["pool-window"]).write_bytes(
            pool_region[window_start:window_start + 8192])
        retained["reference-window"] = f"{entry_id}-reference-window.bin"
        (evidence / retained["reference-window"]).write_bytes(
            reference[window_start:window_start + 8192])

    unchanged, unchanged_count, _ = compare.diff_stats(input_region, pool_region)
    if not unchanged:
        extent = compare.change_extent(input_region, pool_region)
    else:
        extent = (-1, -1, 0)

    # Region verdicts: the output tensor binding and the 96-byte sync record.
    output_start = setup_data["pool_offsets"][1]
    sync_start = setup_data["pool_offsets"][6]
    main_match, main_diffs, main_first = compare.diff_stats(
        reference[output_start:output_start + OUTPUT_BYTES],
        pool_region[output_start:output_start + OUTPUT_BYTES])
    sync = pool_region[sync_start:sync_start + SYNC_BYTES]

    entry = {
        "entry": entry_id, "frame": frame, "kernel": dict(KERNEL_TAGS)[tag],
        "mode": mode, "launches": launches, "readbacks": readbacks,
        "pool_match": match, "pool_diff_count": diff_count,
        "pool_first_diff": first_diff,
        "pool_fnv": f"0x{names.fnv1a64(pool_region):016x}",
        "reference_fnv": f"0x{names.fnv1a64(reference):016x}",
        "input_fnv": f"0x{names.fnv1a64(input_region):016x}",
        "main_match": main_match, "main_diff_count": main_diffs,
        "main_first_diff": main_first,
        "sync_matches_reference": sync == reference[sync_start:sync_start + SYNC_BYTES],
        "sync_from_input": sync == input_region[sync_start:sync_start + SYNC_BYTES],
        "sync_fnv": f"0x{names.fnv1a64(sync):016x}",
        "pool_unchanged": unchanged, "pool_input_diff_count": unchanged_count,
        "change_first": extent[0], "change_last": extent[1],
        "change_blocks": extent[2],
        "guards_intact": (compare.guards_intact(current_pool, guard_words) and
                          compare.guards_intact(current_weights, guard_words)),
        "weights_immutable": (weights_region ==
                              compare.region_slice(initial_weights, weight_alloc)),
        "retained": retained,
    }
    validate_entry(entry)
    return entry


def validate_entry(entry: dict[str, Any]) -> None:
    base.require(entry["pool_match"] == (entry["pool_diff_count"] == 0),
                 "entry: inconsistent verdict")
    base.require(entry["main_match"] == (entry["main_diff_count"] == 0),
                 "entry: inconsistent main verdict")
    base.require(entry["pool_unchanged"] == (entry["pool_input_diff_count"] == 0),
                 "entry: inconsistent input verdict")
    base.require(entry["pool_match"] == (entry["pool_fnv"] == entry["reference_fnv"]),
                 "entry: digest disagrees with comparison")
    base.require(entry["pool_unchanged"] == (entry["pool_fnv"] == entry["input_fnv"]),
                 "entry: input digest disagrees")


def enforced_failures(mode: str, entries: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for entry in entries:
        name = entry["entry"]
        if not entry["guards_intact"]:
            labels.append(f"guards:{name}")
        if not entry["weights_immutable"]:
            labels.append(f"weights-mutated:{name}")
        if mode in ("execute", "tail"):
            check = "pool-mismatch" if mode == "execute" else "tail-mismatch"
            if entry["kernel"] == NAME_LIVE:
                if not entry["pool_match"]:
                    labels.append(f"{check}:{name}")
            else:
                if not entry["main_match"]:
                    labels.append(f"{'main-mismatch' if mode == 'execute' else 'tail-main-mismatch'}:{name}")
        elif mode == "skip":
            if not entry["pool_unchanged"]:
                labels.append(f"skip-changed:{name}")
            if entry["pool_match"]:
                labels.append(f"skip-matched:{name}")
        elif mode == "wrong":
            if entry["pool_match"]:
                labels.append(f"wrong-matched:{name}")
            if entry["pool_unchanged"]:
                labels.append(f"wrong-unchanged:{name}")
        else:
            raise ValueError(f"unknown mode {mode}")
    return labels


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def execute(data: bytes, inventory: dict[str, Any], mode: str,
            evidence: Path, progress: dict[str, Any]) -> dict[str, Any]:
    plan = dict(setup=setup(data, inventory))
    image = plan["setup"]["image"]
    record = plan["setup"]["record"]
    driver = utility.Driver()
    device, version = utility.select_device(driver)
    guard_words = [0x12340000 + i for i in range(GUARD_WORDS)]
    entries: list[dict[str, Any]] = []
    with ExitStack() as resources:
        context = ct.c_void_p()
        driver.check("cuCtxCreate_v2", ct.byref(context), 0, device)
        resources.callback(driver.check, "cuCtxDestroy_v2", context)
        module = ct.c_void_p()
        driver.check("cuModuleLoadData", ct.byref(module), ct.create_string_buffer(image))
        resources.callback(driver.check, "cuModuleUnload", module)
        functions: dict[str, ct.c_void_p] = {}
        for tag, name in KERNEL_TAGS:
            function = ct.c_void_p()
            driver.check("cuModuleGetFunction", ct.byref(function), module, name.encode())
            functions[tag] = function
        for entry_id in PLAN:
            tag, frame_text = entry_id.split("-")
            frame = int(frame_text[1:])
            entries.append(run_entry(driver, functions, plan, entry_id, frame,
                                     tag, mode, guard_words, evidence))
            progress["entries"] = entries

    failures = enforced_failures(mode, entries)
    return dict(
        native=True, device="NVIDIA GeForce RTX 4070 Ti SUPER",
        driver_api_version=version, mode=mode, entries_plan=list(PLAN),
        entries=entries,
        kernels_launched=sum(e["launches"] for e in entries),
        readbacks=sum(e["readbacks"] for e in entries),
        enforced_failures=failures,
        checks_enforced=[
            "guards_intact and weights_immutable on every entry",
            "whole-pool byte match against the s24 vendor capture "
            "(execute, tail: tilesync variant)",
            "main-tensor byte match for the fp8 sibling "
            "(its 96-byte sync record is recorded, not enforced)",
            "pool bit-identical to input and non-match against s24 (skip)",
            "non-match against s24 and pool changed from input "
            "(wrong: weights rotated inwards 1024 bytes)",
        ],
        findings_recorded=[
            "change extent of the pool against the s23 input state",
            "tail bytes beyond the record replaced (execute 0xA5 / tail 0x5A)",
            "fp8 sibling replayed at the tilesync variant's geometry "
            "(the vendor creates but never launches it: sibling_launches pinned 0)",
            "96-byte sync record at w6 per entry: matches reference / matches input",
        ],
        setup={key: value for key, value in plan["setup"].items()
               if key not in ("record", "image", "capture")},
        learned_weights_used=True, neural_image_produced=False,
        complete_native_pipeline=False,
        canaries_do_not_detect_read_only_out_of_bounds=True,
        deliberate_proprietary_payload_files_written=False)


def pair(first: Path, second: Path) -> dict[str, Any]:
    run_a = json.loads((first / "run.json").read_text())
    run_b = json.loads((second / "run.json").read_text())
    labels = pair_runs(run_a, run_b)
    return dict(pair_failures=labels, entries=len(run_a["entries"]),
                bit_identical=not labels)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def make_entry(**overrides: Any) -> dict[str, Any]:
    entry = dict(entry="e0", frame=0, kernel=NAME_LIVE, mode="execute",
                 launches=1, readbacks=2,
                 pool_match=True, pool_diff_count=0, pool_first_diff=-1,
                 pool_fnv="0x1", reference_fnv="0x1", input_fnv="0x2",
                 main_match=True, main_diff_count=0, main_first_diff=-1,
                 sync_matches_reference=True, sync_from_input=False,
                 sync_fnv="0x5",
                 pool_unchanged=False, pool_input_diff_count=5,
                 change_first=10, change_last=20, change_blocks=1,
                 guards_intact=True, weights_immutable=True, retained={})
    entry.update(overrides)
    return entry


def self_test() -> dict[str, Any]:
    # diff_stats known answers.
    base.require(compare.diff_stats(b"abc", b"abc") == (True, 0, -1),
                 "self-test: unchanged diff")
    match, count, first = compare.diff_stats(b"abc", b"aXc")
    base.require(match is False and count == 1 and first == 1,
                 "self-test: middle diff")
    match, count, first = compare.diff_stats(b"abcd", b"abcX")
    base.require(match is False and count == 1 and first == 3,
                 "self-test: last diff")
    known_answers = 3

    # Verdict controls: each corruption must raise exactly its pinned label.
    controls = 0
    base.require(enforced_failures("execute", [make_entry()]) == [],
                 "self-test: clean execute rejected")
    for name, mode, overrides, expected in (
            ("execute mismatch", "execute", dict(pool_match=False, pool_diff_count=1,
                                                 pool_fnv="0x3"),
             "pool-mismatch:e0"),
            ("tail mismatch", "tail", dict(mode="tail", pool_match=False,
                                           pool_diff_count=1, pool_fnv="0x3"),
             "tail-mismatch:e0"),
            ("guards", "execute", dict(guards_intact=False), "guards:e0"),
            ("weights", "execute", dict(weights_immutable=False), "weights-mutated:e0"),
            ("skip changed", "skip", dict(mode="skip", pool_match=False,
                                          pool_diff_count=1, pool_fnv="0x3",
                                          pool_unchanged=False,
                                          pool_input_diff_count=1),
             "skip-changed:e0"),
            ("skip matched", "skip", dict(mode="skip", pool_unchanged=True,
                                          pool_input_diff_count=0),
             "skip-matched:e0"),
            ("wrong matched", "wrong", dict(mode="wrong"), "wrong-matched:e0"),
            ("wrong unchanged", "wrong",
             dict(mode="wrong", pool_match=False, pool_diff_count=1, pool_fnv="0x2",
                  input_fnv="0x2", pool_unchanged=True, pool_input_diff_count=0),
             "wrong-unchanged:e0"),
            ("sibling main mismatch", "execute",
             dict(kernel=NAME_SIBLING, pool_match=False, pool_diff_count=96,
                  pool_fnv="0x3", main_match=False, main_diff_count=1,
                  sync_matches_reference=False, sync_from_input=True),
             "main-mismatch:e0"),
            ("sibling tail mismatch", "tail",
             dict(kernel=NAME_SIBLING, mode="tail", pool_match=False,
                  pool_diff_count=96, pool_fnv="0x3", main_match=False,
                  main_diff_count=1, sync_matches_reference=False,
                  sync_from_input=True),
             "tail-main-mismatch:e0")):
        labels = enforced_failures(mode, [make_entry(**overrides)])
        base.require(expected in labels, f"self-test: {name} not detected")
        base.require(len(labels) == 1, f"self-test: {name} over-detected")
        controls += 1

    # validate_entry consistency controls, each with its pinned message.
    for name, overrides, expected in (
            ("match vs count", dict(pool_match=True, pool_diff_count=2, pool_fnv="0x3"),
             "entry: inconsistent verdict"),
            ("main match vs count", dict(main_match=True, main_diff_count=2),
             "entry: inconsistent main verdict"),
            ("unchanged vs count", dict(pool_unchanged=True, pool_input_diff_count=2),
             "entry: inconsistent input verdict"),
            ("digest vs match", dict(pool_match=True, pool_diff_count=0,
                                     pool_fnv="0x3", reference_fnv="0x4"),
             "entry: digest disagrees with comparison"),
            ("input digest vs unchanged", dict(pool_unchanged=True,
                                               pool_input_diff_count=0),
             "entry: input digest disagrees")):
        try:
            validate_entry(make_entry(**overrides))
        except ValueError as error:
            base.require(str(error) == expected, f"self-test: {name} wrong message")
            controls += 1
        else:
            raise ValueError(f"fixture {name} corruption accepted")

    # Pair controls on synthetic runs.
    run = dict(mode="execute", setup=dict(x=1),
               entries=[make_entry()])
    clean = json.loads(json.dumps(run))
    base.require(pair_runs(run, clean) == [], "self-test: clean pair rejected")
    for name, mutate, expected in (
            ("mode", lambda r: r.update(mode="wrong"), "pair: mode"),
            ("setup", lambda r: r["setup"].update(x=2), "pair: setup"),
            ("pool digest", lambda r: r["entries"][0].update(pool_fnv="0x9",
                                                             pool_match=False,
                                                             pool_diff_count=1),
             "pair:pool-digest:e0"),
            ("verdict", lambda r: r["entries"][0].update(change_first=99),
             "pair:verdict:e0"),
            ("entries", lambda r: r["entries"].append(make_entry(entry="e1")),
             "pair: entries")):
        other = json.loads(json.dumps(run))
        mutate(other)
        labels = pair_runs(run, other)
        base.require(expected in labels, f"self-test: pair {name} not detected")
        controls += 1

    # Weight-layout known answer: two payloads force the 512-byte alignment.
    container = bytearray(struct.pack("<Q", 0))
    expected_offsets = []
    heap = 0
    for index, payload in enumerate((300, 700)):
        name = f"rec{index}".encode()
        container += struct.pack("<Q", len(name)) + name
        record = struct.pack("<QQI", 40 + payload, payload, 1) + bytes(payload) + \
            struct.pack("<IIQ", 0, 0, 1) + struct.pack("<I", 4)
        container += struct.pack("<Q", len(record)) + record
        heap = (heap + 511) // 512 * 512
        expected_offsets.append((heap, payload))
        heap += payload
    struct.pack_into("<Q", container, 0, len(container))
    layout = weight_layout(bytes(container))
    base.require(layout.pop("<heap-total>")["heap_offset"] == heap,
                 "self-test: layout heap total")
    for index, (offset, payload) in enumerate(expected_offsets):
        base.require(layout[f"rec{index}"]["heap_offset"] == offset and
                     layout[f"rec{index}"]["payload_bytes"] == payload,
                     f"self-test: layout record {index}")
    known_answers += 3

    # Metadata known answer and malformed-form control.
    info = struct.pack("<BBH", 3, 0x19, 56) + \
        struct.pack("<BBH", 4, 0x17, 12) + struct.pack("<III", 0, 0, 56 << 18)
    parsed = parameter_metadata(info)
    base.require(parsed == dict(sizes=[56], parameters=[(0, 0, 56)]),
                 "self-test: metadata parse")
    known_answers += 1
    try:
        parameter_metadata(struct.pack("<BBH", 9, 0, 0))
    except ValueError:
        controls += 1
    else:
        raise ValueError("fixture metadata form corruption accepted")

    return dict(diff_known_answers=3, verdict_controls=10,
                consistency_controls=5, pair_controls=5,
                layout_known_answers=3, metadata_known_answers=1,
                corruption_controls=controls,
                known_answers=known_answers,
                gpu_kernels_launched=0)


def pair_runs(first: dict[str, Any], second: dict[str, Any]) -> list[str]:
    """pair() core on in-memory runs (self-test target)."""
    labels: list[str] = []
    if first["mode"] != second["mode"]:
        labels.append("pair: mode")
    if first["setup"] != second["setup"]:
        labels.append("pair: setup")
    entries_a = {entry["entry"]: entry for entry in first["entries"]}
    entries_b = {entry["entry"]: entry for entry in second["entries"]}
    if entries_a.keys() != entries_b.keys():
        labels.append("pair: entries")
    else:
        for name, entry in entries_a.items():
            other = entries_b[name]
            if entry["pool_fnv"] != other["pool_fnv"]:
                labels.append(f"pair:pool-digest:{name}")
            if {key: value for key, value in entry.items() if key != "retained"} != \
               {key: value for key, value in other.items() if key != "retained"}:
                labels.append(f"pair:verdict:{name}")
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=[
        "self-test", "execute", "skip", "wrong", "tail", "pair"])
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()
    if args.command == "self-test":
        print(json.dumps(self_test(), indent=2))
        return 0
    if args.command == "pair":
        if len(args.paths) != 2:
            print("usage: native-slot-replay-probe pair FIRST SECOND")
            return 2
        result = pair(Path(args.paths[0]), Path(args.paths[1]))
        print(json.dumps(result, indent=2))
        return 1 if result["pair_failures"] else 0
    if len(args.paths) != 1:
        print("usage: native-slot-replay-probe {execute|skip|wrong|tail} OUTDIR")
        return 2
    evidence = Path(args.paths[0])
    evidence.mkdir(parents=True, exist_ok=True)
    data = DLL.read_bytes()
    inventory = base.inventory(data)
    progress: dict[str, Any] = {}
    result = execute(data, inventory, args.command, evidence, progress)
    (evidence / "run.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(
        mode=result["mode"], entries=len(result["entries"]),
        kernels_launched=result["kernels_launched"],
        readbacks=result["readbacks"],
        enforced_failures=result["enforced_failures"]), indent=2))
    return 1 if result["enforced_failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
