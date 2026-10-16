#!/usr/bin/env python3
"""Native pre-decoder kernel replay: slots 24-32 against ladder references.

Generalises the decoder replay (NATIVE-VENDOR-COMPARE.md) to pre-decoder
kernels: vendor launches of the 156-launch inventory (NVAPI-ARGS.md) are
replayed natively on Linux (CUDA driver; no Windows code, DirectX or Proton)
from their captured parameter blocks, and the native pool state is compared
byte-for-byte against the vendor's own captured state at the matching ladder
cut point (LADDER.md). Nine per-slot contracts are covered:

- slot 24 `cc_split_swin_16h_ffwd_inpview_512_tilesync_fp8` (s23 -> s24):
  reads w0, writes its output tensor at w1 (344,064 bytes) and a 96-byte
  sync record at w6 — the stage-16 claim;
- slot 25 `cc_split_swin_16h_ffwd_proj_inpview_512_chained_fp8`
  (s24 -> s25): the ffwd-proj pair's second half. It reads slot 24's
  output (w0) and a residual input (w1), consumes slot 24's sync record
  (w7 — the chained handoff), writes its output tensor at w2 (344,064
  bytes) and a 96-byte sync record at w6. The write-set attribution pins
  exactly those two words (names-s24-s25.txt: w2 + w6, zero unattributed);
- slot 26 `cc_split_swin_16h_qkv_512_chained_fp8` (s25 -> s26): the
  transformer block's qkv half. It reads slot 25's output (w0), consumes
  slot 25's sync record (w5), writes its output tensor at w1 (344,064
  bytes) and a sync record at w6 whose changed extent is 192 bytes
  (names-s23-s56.txt, stable across all three frames). The kernel handle
  is shared with slot 30 at a different geometry (grid 4,3,4 here, 4,4,4
  there), so the grid is pinned at the slot's own launch position;
- slot 27 `cc_split_swin_16h_proj_512_chained_fp8` (s26 -> s27): the
  transformer block's proj half (the qkv-proj pair's second half). It
  reads slot 26's output (w0) and a residual input (w1, the address slot
  25 wrote its output to), consumes slot 26's sync record (w7 — the
  chained handoff, address-equal to slot 26's w6), writes its output
  tensor at w2 (344,064 bytes) and a 96-byte sync record at w6. The
  write-set attribution pins exactly those two words (names-s26-s27.txt:
  w2 + w6, zero unattributed, stable across all three frames);
- slot 28 `cc_split_swin_16h_ffwd_512_chained_fp8` (s27 -> s28): the next
  block's ffwd half — the first kernel outside `block23`. It reads slot
  27's output (w0 at 21,642,240 — address-equal to slot 27's w2),
  consumes slot 27's sync record (w4 at 50,176 — address-equal to slot
  27's w6), writes its output tensor at w1 (344,064 bytes) and a 96-byte
  sync record at w6. The write-set attribution pins exactly those two
  words (names-s27-s28.txt: w1 + w6, zero unattributed, stable across all
  three frames);
- slot 29 `cc_split_swin_16h_ffwd_proj_512_chained_fp8` (s28 -> s29): the
  block24 ffwd-proj pair's second half (the `ffwd_proj` counterpart to
  slot 25's, same 72-byte block and grid 8,3,1). It reads slot 28's
  output (w0 at 21,986,304 — address-equal to slot 28's w1) and a
  residual input (w1 at 21,642,240 — the buffer slot 27 wrote its output
  to), consumes slot 28's sync record (w7 at 50,688 — address-equal to
  slot 28's w6, the chained handoff), writes its output tensor at w2
  (344,064 bytes at 22,330,368) and a 96-byte sync record at w6 (51,200
  — the 512-byte stride continuing). The write-set attribution pins
  exactly those two words (names-s28-s29.txt: w2 + w6, zero unattributed,
  stable across all three frames). It binds `block24.layer1.layer`.
- slot 30 `cc_split_swin_16h_qkv_512_chained_fp8` (s29 -> s30): the
  block24 qkv half — slot 26's counterpart at grid 4,4,4, the shared
  handle's second geometry now exercised. It reads slot 29's output (w0
  at 22,330,368 — address-equal to slot 29's w2) and consumes slot 29's
  sync record (w5 at 51,200 — address-equal to slot 29's w6), writes its
  output tensor at w1 (344,064 bytes at 22,674,432) and a sync record at
  w6 (51,712 — the 512-byte stride continuing) whose changed extent is
  256 bytes (names-s29-s30.txt, stable across all three frames). The
  write-set attribution pins exactly those two words (w1 + w6, zero
  unattributed). It binds `block24.layer2.layer` (917,568 B, the record
  immediately after `block24.layer1.layer`). Its parameter block carries
  two constant words at two distinct values: w3 the (24, 28) geometry
  constant and w4 0xFFFFFFFCFFFFFFFC, where the same-shape slot 26
  reserves its w4 as zero.
- slot 31 `cc_split_swin_16h_proj_512_chained_fp8` (s30 -> s31): the
  block24 qkv-proj pair's second half (the `proj` counterpart to slot
  27's, same 72-byte block). It reads slot 30's output (w0 at 22,674,432
  — address-equal to slot 30's w1) and a residual input (w1 at
  22,330,368 — the buffer slot 29 wrote its output to), consumes slot
  30's sync record (w7 at 51,712 — address-equal to slot 30's w6, the
  chained handoff), writes its output tensor at w2 (344,064 bytes at
  23,018,496) and a 96-byte sync record at w6 (52,224 — the 512-byte
  stride continuing). The write-set attribution pins exactly those two
  words (names-s30-s31.txt: w2 + w6, zero unattributed, stable across
  all three frames). It binds `block24.layer3.layer` (263,168 B at heap
  offset 10,407,424 — the record immediately after `block24.layer2.layer`,
  the record-ordering walk holding through the pair). Its parameter block
  carries two constant words at two distinct values like its pair partner
  slot 30: w4 the (24, 28) geometry constant and w8 0xFFFFFFFCFFFFFFFC,
  where the same-shape slot 27 reserves its w8 as zero — the block24
  qkv-proj pair carries that constant in both halves, block23's pair
  carries neither.
- slot 32 `cc_split_swin_16h_ffwd_512_chained_fp8` (s31 -> s32): the
  block25 ffwd half — the next `ffwd->proj` pair's first half. It reads
  slot 31's output (w0 at 23,018,496 — address-equal to slot 31's w2) and
  consumes slot 31's sync record (w4 at 52,224 — address-equal to slot
  31's w6, the chained handoff at the ffwd shape's w4 position, like slot
  28), writes its output tensor at w1 (344,064 bytes at 23,362,560) and a
  96-byte sync record at w6 (52,736 — the 512-byte stride continuing).
  The write-set attribution pins exactly those two words
  (names-s31-s32.txt: w1 + w6, zero unattributed, stable across all three
  frames). It binds `block25.layer0.layer` (524,288 B — the record
  directly after `block24.layer3.layer`, the record-ordering walk
  continuing into block25). Its parameter block is the slot-28 word
  map exactly (no second constant: w5 is reserved zero).

Inputs come from retained evidence only:

- the launch specification (grid, block, shared memory, parameter block) is
  parsed from the forwarder traces' slot records and pinned equal across
  evaluations, runs and the capture sidecar;
- the input pool state is the ladder capture after slot n-1 (exactly the
  state slot n reads), the reference pool state the capture after slot n;
- the weight bytes are extracted from the model DLL's WEIGHTS_HT resource
  through the measured heap layout (record payloads concatenated at 512-byte
  alignment): slot 24 binds `block23.layer0.layer`, slot 25 binds
  `block23.layer1.layer`, slot 26 binds `block23.layer2.layer`, slot 27
  binds `block23.layer3.layer`, slot 28 binds `block24.layer0.layer` (the
  record immediately after `block23.layer3.layer`), slot 29 binds
  `block24.layer1.layer` (the record immediately after `block24.layer0.layer`
  — the record-ordering pattern repeats in block24), slot 30 binds
  `block24.layer2.layer` (the record immediately after `block24.layer1.layer`),
  slot 31 binds `block24.layer3.layer` (the record immediately after
  `block24.layer2.layer`, at heap offset 10,407,424), slot 32 binds
  `block25.layer0.layer` (the record immediately after
  `block24.layer3.layer`)
  — all exact
  record-start landings under the decoder-anchored layout
  (DECODER-BOUNDARY.md word 7 at `block39.layer0.layer`).

Primary claim (execute mode): for each slot's vendor-launched variant the
native pool readback equals the vendor's captured pool byte-for-byte over
the whole 62,684,160-byte pool. Each contract's fp8 sibling (created but
never launched by the vendor — pinned) is replayed at the live geometry as a
variant probe: its output tensor must match, and its sync record is recorded
per entry (the decoder stage found the same family record at out2; the
record's changed extent is per slot — 96 B at the decoder and slots 24, 25,
27, 28, 29, 31 and 32, 192 B at slot 26, 256 B at slot 30). Controls: skip (no launch), wrong (weights rotated
inwards), tail (bytes beyond the record replaced by a different pattern —
bounding weight reads to the record).

This stage uses stored learned weight bytes through extracted records
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

CONTAINER = 4
FRAMES = 3
ANCHOR_RECORD = "block39.layer0.layer"
HEAP_SIZE = 147719680
OUTPUT_BYTES = 344064   # binding extent 24x28x512 (LADDER-NAMES.md segments)
ROTATE_BYTES = 1024
TAIL_BYTES = 4096
EXEC_TAIL = 0xA5
TAIL_TAIL = 0x5A
GUARD_WORDS = launch.GUARD_WORDS
CLS_POOL = "pool"
CLS_WEIGHTS = "weights"
CLS_CONSTANT = "constant"
CLS_ZERO = "zero"

# Per-slot replay contracts. word_classes classifies every 8-byte word of the
# slot's parameter block in index order: pool (working-pool address, replayed
# at the same pool offset), weights (weight-heap record start, the extracted
# record replayed at our allocation), constant (passed through from the
# captured block, value pinned per word in constant_values), zero (reserved,
# must be 0). sync_bytes is the sync record's changed extent, measured per
# slot in the ladder write-sets (LADDER-NAMES.md segments;
# 96/96/192/96/96/96/256/96/96 for slots 24/25/26/27/28/29/30/31/32) — the fp8
# sibling's divergence from the reference is enforced to equal it exactly.
# constant_values maps constant-word index to its pinned value (a block can
# carry more than one, at more than one value: slot 30's w4 and slot 31's w8
# are 0xFFFFFFFCFFFFFFFC where the same-shape slots 26/27 reserve those
# words as zero).
CONTRACTS: dict[int, dict[str, Any]] = {
    24: dict(
        slot=24,
        psize=56,
        name_live="cc_split_swin_16h_ffwd_inpview_512_tilesync_fp8",
        name_sibling="cc_split_swin_16h_ffwd_inpview_512_fp8",
        tag_live="tilesync", tag_sibling="fp8",
        record_name="block23.layer0.layer", record_bytes=524288,
        constant_values={3: 0x1C00000018},   # recovered (24, 28) geometry overrides
        word_classes=(CLS_POOL, CLS_POOL, CLS_WEIGHTS, CLS_CONSTANT,
                      CLS_ZERO, CLS_ZERO, CLS_POOL),
        output_word=1, sync_word=6, sync_bytes=96,
        input_label=23, reference_label=24),
    25: dict(
        slot=25,
        psize=72,
        name_live="cc_split_swin_16h_ffwd_proj_inpview_512_chained_fp8",
        name_sibling="cc_split_swin_16h_ffwd_proj_inpview_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        record_name="block23.layer1.layer", record_bytes=263168,
        constant_values={4: 0x1C00000018},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS,
                      CLS_CONSTANT, CLS_ZERO, CLS_POOL, CLS_POOL, CLS_ZERO),
        output_word=2, sync_word=6, sync_bytes=96,
        input_label=24, reference_label=25),
    26: dict(
        slot=26,
        psize=56,
        name_live="cc_split_swin_16h_qkv_512_chained_fp8",
        name_sibling="cc_split_swin_16h_qkv_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        record_name="block23.layer2.layer", record_bytes=917568,
        constant_values={3: 0x1C00000018},
        word_classes=(CLS_POOL, CLS_POOL, CLS_WEIGHTS, CLS_CONSTANT,
                      CLS_ZERO, CLS_POOL, CLS_POOL),
        output_word=1, sync_word=6, sync_bytes=192,
        input_label=25, reference_label=26),
    27: dict(
        slot=27,
        psize=72,
        name_live="cc_split_swin_16h_proj_512_chained_fp8",
        name_sibling="cc_split_swin_16h_proj_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        record_name="block23.layer3.layer", record_bytes=263168,
        constant_values={4: 0x1C00000018},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS,
                      CLS_CONSTANT, CLS_ZERO, CLS_POOL, CLS_POOL, CLS_ZERO),
        output_word=2, sync_word=6, sync_bytes=96,
        input_label=26, reference_label=27),
    28: dict(
        slot=28,
        psize=56,
        name_live="cc_split_swin_16h_ffwd_512_chained_fp8",
        name_sibling="cc_split_swin_16h_ffwd_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        record_name="block24.layer0.layer", record_bytes=524288,
        constant_values={3: 0x1C00000018},
        word_classes=(CLS_POOL, CLS_POOL, CLS_WEIGHTS, CLS_CONSTANT,
                      CLS_POOL, CLS_ZERO, CLS_POOL),
        output_word=1, sync_word=6, sync_bytes=96,
        input_label=27, reference_label=28),
    29: dict(
        slot=29,
        psize=72,
        name_live="cc_split_swin_16h_ffwd_proj_512_chained_fp8",
        name_sibling="cc_split_swin_16h_ffwd_proj_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        record_name="block24.layer1.layer", record_bytes=263168,
        constant_values={4: 0x1C00000018},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS,
                      CLS_CONSTANT, CLS_ZERO, CLS_POOL, CLS_POOL, CLS_ZERO),
        output_word=2, sync_word=6, sync_bytes=96,
        input_label=28, reference_label=29),
    30: dict(
        slot=30,
        psize=56,
        name_live="cc_split_swin_16h_qkv_512_chained_fp8",
        name_sibling="cc_split_swin_16h_qkv_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        record_name="block24.layer2.layer", record_bytes=917568,
        constant_values={3: 0x1C00000018, 4: 0xFFFFFFFCFFFFFFFC},
        word_classes=(CLS_POOL, CLS_POOL, CLS_WEIGHTS, CLS_CONSTANT,
                      CLS_CONSTANT, CLS_POOL, CLS_POOL),
        output_word=1, sync_word=6, sync_bytes=256,
        input_label=29, reference_label=30),
    31: dict(
        slot=31,
        psize=72,
        name_live="cc_split_swin_16h_proj_512_chained_fp8",
        name_sibling="cc_split_swin_16h_proj_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        record_name="block24.layer3.layer", record_bytes=263168,
        constant_values={4: 0x1C00000018, 8: 0xFFFFFFFCFFFFFFFC},
        word_classes=(CLS_POOL, CLS_POOL, CLS_POOL, CLS_WEIGHTS,
                      CLS_CONSTANT, CLS_ZERO, CLS_POOL, CLS_POOL,
                      CLS_CONSTANT),
        output_word=2, sync_word=6, sync_bytes=96,
        input_label=30, reference_label=31),
    32: dict(
        slot=32,
        psize=56,
        name_live="cc_split_swin_16h_ffwd_512_chained_fp8",
        name_sibling="cc_split_swin_16h_ffwd_512_fp8",
        tag_live="chained", tag_sibling="fp8",
        record_name="block25.layer0.layer", record_bytes=524288,
        constant_values={3: 0x1C00000018},
        word_classes=(CLS_POOL, CLS_POOL, CLS_WEIGHTS, CLS_CONSTANT,
                      CLS_POOL, CLS_ZERO, CLS_POOL),
        output_word=1, sync_word=6, sync_bytes=96,
        input_label=31, reference_label=32),
}

DLL = HERE.parent / "model" / "nvngx_dlssnr.dll"
BOUNDARY_MANIFEST = HERE / "logs" / "boundary" / "a" / "manifest.txt"


def capture_dir(label: int) -> Path:
    return HERE / "logs" / "ladder" / f"s{label}"


def trace_path(label: int) -> Path:
    return HERE / "logs" / "capture" / f"nvapi-trace-ladder-s{label}.log"

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
def entry_plan(contract: dict[str, Any]) -> tuple[str, ...]:
    return tuple(f"{tag}-f{frame}" for frame in range(FRAMES)
                 for tag in (contract["tag_live"], contract["tag_sibling"]))


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
        for _handle, psize, _pcap, pmem, _grid in evaluations:
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


def parse_grid(raw: str) -> tuple[int, int, int]:
    """The launch grid of one slot record (three decimal components)."""
    parts = raw.split(",")
    base.require(len(parts) == 3 and
                 all(p.isascii() and p.isdigit() for p in parts),
                 "grid: malformed geometry")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def following_records(layout: dict[str, dict[str, int]], record_name: str,
                      record_bytes: int) -> list[str]:
    """Records starting at this record's payload end, rounded up to the
    heap's 512-byte record alignment (payloads need not be aligned:
    `block23.layer2.layer` is followed by 448 bytes of padding)."""
    record_end = layout[record_name]["heap_offset"] + record_bytes
    return [name for name, entry in layout.items()
            if entry["heap_offset"] == (record_end + 511) // 512 * 512]


def setup(data: bytes, inventory: dict[str, Any],
          contract: dict[str, Any]) -> dict[str, Any]:
    """Resolve the launch, bindings and weights from retained evidence."""
    slot = contract["slot"]
    psize = contract["psize"]
    live, sibling = contract["tag_live"], contract["tag_sibling"]
    kernel_tags = ((live, contract["name_live"]),
                   (sibling, contract["name_sibling"]))
    in_label, ref_label = contract["input_label"], contract["reference_label"]
    trace_text = {label: trace_path(label).read_text(encoding="utf-8",
                                                     errors="replace")
                  for label in (in_label, ref_label)}
    traces = {label: names.read_trace(trace_path(label))
              for label in (in_label, ref_label)}

    # Creation geometry (pinned equal across runs) and per-trace handles.
    created: dict[str, dict[str, Any]] = {}
    handles: dict[int, dict[str, str]] = {label: {} for label in (in_label, ref_label)}
    for tag, name in kernel_tags:
        per_trace = {label: read_created(trace_text[label], name)
                     for label in (in_label, ref_label)}
        base.require({key: value for key, value in per_trace[in_label].items()
                      if key != "handle"} ==
                     {key: value for key, value in per_trace[ref_label].items()
                      if key != "handle"},
                     f"created: {name} geometry differs between runs")
        created[tag] = per_trace[in_label]
        for label in (in_label, ref_label):
            handles[label][tag] = per_trace[label]["handle"]

    # Parameter block: the slot's launch must be the live kernel's (identity
    # pinned per trace and frame), a full dump, stable across evals and runs.
    dumps: dict[int, str] = {}
    for label, trace in traces.items():
        base.require(len(trace["evals"]) >= FRAMES,
                     f"slot{slot}-dump: eval count")
        pmems = []
        for evaluation in trace["evals"]:
            handle, seen_size, _pcap, pmem, _grid = evaluation[slot]
            base.require(handle == handles[label][live],
                         f"slot{slot}-identity: slot {slot} is not the live launch")
            base.require(seen_size == psize and len(pmem) == 2 * psize,
                         f"slot{slot}-dump: truncated parameter dump")
            pmems.append(pmem)
        base.require(len(set(pmems)) == 1,
                     f"slot{slot}-dump: parameters vary across evals")
        dumps[label] = pmems[0]
    base.require(dumps[in_label] == dumps[ref_label],
                 f"slot{slot}-dump: parameters differ between runs")
    words = struct.unpack(f"<{psize // 8}Q", bytes.fromhex(dumps[in_label]))
    base.require(len(words) == len(contract["word_classes"]),
                 f"slot{slot}-words: class table length")
    for index, cls in enumerate(contract["word_classes"]):
        if cls == CLS_ZERO:
            base.require(words[index] == 0,
                         f"slot{slot}-words: reserved word non-zero")
        elif cls == CLS_CONSTANT:
            base.require(words[index] == contract["constant_values"][index],
                         f"slot{slot}-words: constant word mismatch")

    # Per-slot grid, read at the slot's own launch position: handles are
    # shared across slots at different geometries (the slot-26 qkv kernel runs
    # at 4,3,4 here and 4,4,4 at slot 30), so the position — not the handle —
    # is the geometry's identity. Pinned across evaluations and runs. The
    # vendor launches only the live variant at this slot; the fp8 sibling is
    # replayed at the same geometry (NATIVE-VENDOR-COMPARE.md precedent).
    slot_grids = {label: {parse_grid(evaluation[slot][4])
                          for evaluation in traces[label]["evals"]}
                  for label in (in_label, ref_label)}
    base.require(len(slot_grids[in_label]) == 1,
                 f"grid: {in_label} launch geometry varies")
    base.require(len(slot_grids[ref_label]) == 1,
                 f"grid: {ref_label} launch geometry varies")
    base.require(slot_grids[in_label] == slot_grids[ref_label],
                 "grid: geometry differs between runs")
    grids = {live: slot_grids[in_label].pop(), sibling: None}
    grids[sibling] = grids[live]
    # The fp8 sibling is created but never launched by the vendor: it has no
    # launch contract of its own (pinned per trace, each with its own handle).
    sibling_launches = sum(1 for label in (in_label, ref_label)
                           for m in GRID_RE.finditer(trace_text[label])
                           if m.group(1) == handles[label][sibling])
    base.require(sibling_launches == 0, "sibling: unexpected vendor launches")

    # Captures: input state (after slot-1) and reference state (after slot).
    capture = {label: names.read_capture(capture_dir(label))
               for label in (in_label, ref_label)}
    base.require(capture[in_label]["slot"] == in_label and
                 capture[ref_label]["slot"] == ref_label, "capture: slot pin")
    base.require(capture[in_label]["pool_size"] == capture[ref_label]["pool_size"],
                 "capture: pool size differs")
    base.require(capture[in_label]["pool_va"] == capture[ref_label]["pool_va"],
                 "capture: pool address differs")
    base.require(capture[in_label]["frames"] == capture[ref_label]["frames"] == FRAMES,
                 "capture: frame count")

    # The capture sidecar carries the same parameter block (independent copy).
    sidecar = [BOUNDARY_RE.match(line) for line in
               (capture_dir(ref_label) / "boundary-va.txt").read_text().splitlines()
               if line]
    base.require(all(sidecar) and len(sidecar) == FRAMES, "sidecar: record count")
    base.require(all(m.group(2) == str(psize) for m in sidecar if m),
                 "sidecar: parameter size")
    base.require(all(m.group(3) == dumps[in_label] for m in sidecar if m),
                 f"slot{slot}-dump: sidecar parameters differ")

    # Pool bindings: every pool-class word points into the working pool.
    pool_va = capture[in_label]["pool_va"]
    pool_size = capture[in_label]["pool_size"]
    offsets: dict[int, int] = {}
    for index, cls in enumerate(contract["word_classes"]):
        if cls == CLS_POOL:
            base.require(pool_va <= words[index] < pool_va + pool_size,
                         f"slot{slot}-bind: word {index} outside pool")
            offsets[index] = words[index] - pool_va

    # The weights word resolves into the weight heap through the captured
    # resource map.
    weight_index = contract["word_classes"].index(CLS_WEIGHTS)
    res_lines = [RES_RE.match(line) for line in
                 (capture_dir(ref_label) / "resource-map.txt").read_text().splitlines()
                 if line.startswith("res ")]
    base.require(all(res_lines), f"slot{slot}-bind: resource map format")
    containing = [m for m in res_lines if m
                  and int(m.group(2), 16) <= words[weight_index]
                  < int(m.group(2), 16) + int(m.group(3))]
    base.require(len(containing) == 1, f"slot{slot}-bind: weight resource not unique")
    weight_res = containing[0]
    base.require(int(weight_res.group(3)) == HEAP_SIZE,
                 f"slot{slot}-bind: unexpected weight resource size")
    heap_offset = words[weight_index] - int(weight_res.group(2), 16)

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
    record_name = contract["record_name"]
    record_bytes = contract["record_bytes"]
    base.require(layout[record_name]["heap_offset"] == heap_offset,
                 f"slot{slot}-weight: record binding mismatch")
    base.require(layout[record_name]["payload_bytes"] == record_bytes,
                 f"slot{slot}-weight: record size mismatch")
    # Every heap-pointing parameter word of both traces must address a whole
    # record payload start (the structural claim, computed and enforced).
    heap_scan = {str(label): heap_word_scan(traces[label],
                                            int(weight_res.group(2), 16),
                                            HEAP_SIZE, layout)
                 for label in (in_label, ref_label)}
    for label in (in_label, ref_label):
        scan = heap_scan[str(label)]
        base.require(scan["mid_record"] == 0 and
                     scan["total"] == scan["at_record_start"],
                     "weight-heap: weight pointer off record start")
    record = base.span(weights, layout[record_name]["payload_at"], record_bytes)
    following = following_records(layout, record_name, record_bytes)
    base.require(len(following) == 1, f"slot{slot}-weight: no following record")
    tail_live = base.span(weights, layout[following[0]]["payload_at"],
                          min(TAIL_BYTES, layout[following[0]]["payload_bytes"]))
    base.require(tail_live != bytes([EXEC_TAIL]) * len(tail_live) and
                 tail_live != bytes([TAIL_TAIL]) * len(tail_live),
                 "weight-tail: tail pattern coincides with live bytes")

    # Kernel ABI from the SM89 cubin: one parameter of the contract's size.
    modules = [e for e in inventory["fatbins"][CONTAINER]["entries"]
               if e["kind"] == 2 and e["arch"] == 89]
    base.require(len(modules) == 1, "abi: unique SM89 module")
    image = base.decompress(data, modules[0])
    for tag, name in kernel_tags:
        meta = parameter_metadata(launch.kernel_metadata(image, name))
        base.require(meta["sizes"] == [psize] and
                     meta["parameters"] == [(0, 0, psize)],
                     f"abi: unexpected parameter metadata for {name}")
    base.require(all(created[tag]["smem"] == 0 for tag in dict(kernel_tags)),
                 "abi: unexpected shared memory")

    return dict(
        words=list(words), pool_offsets=offsets, heap_offset=heap_offset,
        heap_total=heap_total, heap_allocation=HEAP_SIZE,
        sibling_launches=sibling_launches, heap_scan=heap_scan,
        record_sha256=hashlib.sha256(record).hexdigest(),
        block=created[live]["block"], smem=created[live]["smem"],
        grids={tag: list(grids[tag]) for tag in dict(kernel_tags)},
        pool_va=pool_va, pool_size=pool_size,
        trace_sha256={str(label): hashlib.sha256(
            trace_path(label).read_bytes()).hexdigest()
            for label in (in_label, ref_label)},
        capture_fnv={str(label): {str(p["frame"]): f"0x{p['fnv']:016x}"
                                  for p in capture[label]["pools"]}
                     for label in (in_label, ref_label)},
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
    contract = plan["contract"]
    capture = setup_data["capture"]
    input_pool = capture[contract["input_label"]]["pools"][frame]["data"]
    reference = capture[contract["reference_label"]]["pools"][frame]["data"]
    record = setup_data["record"]
    record_bytes = contract["record_bytes"]
    tail = bytes([MODE_TAIL[mode]]) * TAIL_BYTES
    weights_content = record + tail
    if mode == "wrong":
        weights_content = (record[ROTATE_BYTES:] + record[:ROTATE_BYTES]) + tail

    launches = readbacks = 0
    with ExitStack() as allocations:
        pool_alloc = launch.allocate(driver, allocations, setup_data["pool_size"])
        weight_alloc = launch.allocate(driver, allocations,
                                       record_bytes + TAIL_BYTES)
        initial_pool = compare.upload_region(driver, pool_alloc, input_pool,
                                             0x00, guard_words)
        initial_weights = compare.upload_region(driver, weight_alloc, weights_content,
                                                0x00, guard_words)
        launched = mode != "skip"
        if launched:
            words = setup_data["words"]
            offsets = setup_data["pool_offsets"]
            values = []
            for index, cls in enumerate(contract["word_classes"]):
                if cls == CLS_POOL:
                    values.append(pool_alloc[1] + offsets[index])
                elif cls == CLS_WEIGHTS:
                    values.append(weight_alloc[1])
                elif cls == CLS_CONSTANT:
                    values.append(words[index])
                else:
                    values.append(0)
            arguments = (ct.c_uint64 * (contract["psize"] // 8))(*values)
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

    # Region verdicts: the output tensor binding and the sync record (its
    # changed extent per slot, contract["sync_bytes"]).
    sync_bytes = contract["sync_bytes"]
    output_start = setup_data["pool_offsets"][contract["output_word"]]
    sync_start = setup_data["pool_offsets"][contract["sync_word"]]
    main_match, main_diffs, main_first = compare.diff_stats(
        reference[output_start:output_start + OUTPUT_BYTES],
        pool_region[output_start:output_start + OUTPUT_BYTES])
    sync = pool_region[sync_start:sync_start + sync_bytes]
    _sync_match, sync_diff_count, _sync_first = compare.diff_stats(
        reference[sync_start:sync_start + sync_bytes], sync)

    entry = {
        "entry": entry_id, "frame": frame, "kernel": plan["kernels"][tag],
        "mode": mode, "launches": launches, "readbacks": readbacks,
        "pool_match": match, "pool_diff_count": diff_count,
        "pool_first_diff": first_diff,
        "pool_fnv": f"0x{names.fnv1a64(pool_region):016x}",
        "reference_fnv": f"0x{names.fnv1a64(reference):016x}",
        "input_fnv": f"0x{names.fnv1a64(input_region):016x}",
        "main_match": main_match, "main_diff_count": main_diffs,
        "main_first_diff": main_first,
        "sync_matches_reference": sync == reference[sync_start:sync_start + sync_bytes],
        "sync_diff_count": sync_diff_count,
        "sync_from_input": sync == input_region[sync_start:sync_start + sync_bytes],
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
    base.require(entry["sync_matches_reference"] == (entry["sync_diff_count"] == 0),
                 "entry: inconsistent sync verdict")


def enforced_failures(mode: str, entries: list[dict[str, Any]],
                      live_name: str, sync_bytes: int) -> list[str]:
    labels: list[str] = []
    for entry in entries:
        name = entry["entry"]
        if not entry["guards_intact"]:
            labels.append(f"guards:{name}")
        if not entry["weights_immutable"]:
            labels.append(f"weights-mutated:{name}")
        if mode in ("execute", "tail"):
            check = "pool-mismatch" if mode == "execute" else "tail-mismatch"
            if entry["kernel"] == live_name:
                if not entry["pool_match"]:
                    labels.append(f"{check}:{name}")
            else:
                if not entry["main_match"]:
                    labels.append(f"{'main-mismatch' if mode == 'execute' else 'tail-main-mismatch'}:{name}")
                elif (entry["pool_diff_count"] != sync_bytes or
                      entry["sync_diff_count"] != sync_bytes):
                    # The fp8 sibling's divergence from the reference must be
                    # exactly the sync record's changed extent (the
                    # tilesync/chained family trait, measured per slot in the
                    # ladder write-sets). Equal difference counts inside the
                    # record and over the whole pool prove confinement: every
                    # differing byte lies within the record.
                    labels.append(f"sibling-divergence:{name}")
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
            evidence: Path, progress: dict[str, Any],
            contract: dict[str, Any]) -> dict[str, Any]:
    kernel_tags = ((contract["tag_live"], contract["name_live"]),
                   (contract["tag_sibling"], contract["name_sibling"]))
    plan = dict(setup=setup(data, inventory, contract), contract=contract,
                kernels=dict(kernel_tags))
    image = plan["setup"]["image"]
    driver = utility.Driver()
    device, version = utility.select_device(driver)
    guard_words = [0x12340000 + i for i in range(GUARD_WORDS)]
    entries: list[dict[str, Any]] = []
    plan_entries = entry_plan(contract)
    with ExitStack() as resources:
        context = ct.c_void_p()
        driver.check("cuCtxCreate_v2", ct.byref(context), 0, device)
        resources.callback(driver.check, "cuCtxDestroy_v2", context)
        module = ct.c_void_p()
        driver.check("cuModuleLoadData", ct.byref(module), ct.create_string_buffer(image))
        resources.callback(driver.check, "cuModuleUnload", module)
        functions: dict[str, ct.c_void_p] = {}
        for tag, name in kernel_tags:
            function = ct.c_void_p()
            driver.check("cuModuleGetFunction", ct.byref(function), module, name.encode())
            functions[tag] = function
        for entry_id in plan_entries:
            tag, frame_text = entry_id.split("-")
            frame = int(frame_text[1:])
            entries.append(run_entry(driver, functions, plan, entry_id, frame,
                                     tag, mode, guard_words, evidence))
            progress["entries"] = entries

    failures = enforced_failures(mode, entries, contract["name_live"],
                                contract["sync_bytes"])
    return dict(
        native=True, device="NVIDIA GeForce RTX 4070 Ti SUPER",
        driver_api_version=version, mode=mode, slot=contract["slot"],
        entries_plan=list(plan_entries),
        entries=entries,
        kernels_launched=sum(e["launches"] for e in entries),
        readbacks=sum(e["readbacks"] for e in entries),
        enforced_failures=failures,
        checks_enforced=[
            "guards_intact and weights_immutable on every entry",
            f"whole-pool byte match against the s{contract['reference_label']} "
            f"vendor capture (execute, tail: {contract['tag_live']} variant)",
            "main-tensor byte match for the fp8 sibling, with its "
            f"divergence confined to the {contract['sync_bytes']}-byte sync "
            f"record (w{contract['sync_word']})",
            f"pool bit-identical to input and non-match against "
            f"s{contract['reference_label']} (skip)",
            f"non-match against s{contract['reference_label']} and pool changed "
            f"from input (wrong: weights rotated inwards {ROTATE_BYTES} bytes)",
        ],
        findings_recorded=[
            f"change extent of the pool against the s{contract['input_label']} "
            f"input state",
            f"tail bytes beyond the record replaced (execute 0x{EXEC_TAIL:02X} / "
            f"tail 0x{TAIL_TAIL:02X})",
            "fp8 sibling replayed at the live variant's geometry "
            "(the vendor creates but never launches it: sibling_launches pinned 0)",
            f"{contract['sync_bytes']}-byte sync record at "
            f"w{contract['sync_word']} per entry: "
            f"matches reference / matches input",
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

SYNC_FIXTURE = 48640   # fixture pool offset for the sync record
FIXTURE_SYNC_BYTES = 96   # fixture sync record extent
FIXTURE_LIVE = "fixture-live-kernel"
FIXTURE_SIBLING = "fixture-sibling-kernel"


def make_entry(**overrides: Any) -> dict[str, Any]:
    entry = dict(entry="e0", frame=0, kernel=FIXTURE_LIVE, mode="execute",
                 launches=1, readbacks=2,
                 pool_match=True, pool_diff_count=0, pool_first_diff=-1,
                 pool_fnv="0x1", reference_fnv="0x1", input_fnv="0x2",
                 main_match=True, main_diff_count=0, main_first_diff=-1,
                 sync_matches_reference=True, sync_from_input=False,
                 sync_diff_count=0, sync_fnv="0x5",
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
    base.require(enforced_failures("execute", [make_entry()],
                             FIXTURE_LIVE, FIXTURE_SYNC_BYTES) == [],
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
                                          pool_input_diff_count=0,
                                          input_fnv="0x1"),
             "skip-matched:e0"),
            ("wrong matched", "wrong", dict(mode="wrong"), "wrong-matched:e0"),
            ("wrong unchanged", "wrong",
             dict(mode="wrong", pool_match=False, pool_diff_count=1, pool_fnv="0x2",
                  input_fnv="0x2", pool_unchanged=True, pool_input_diff_count=0),
             "wrong-unchanged:e0"),
            ("sibling main mismatch", "execute",
             dict(kernel=FIXTURE_SIBLING, pool_match=False, pool_diff_count=96,
                  pool_fnv="0x3", pool_first_diff=SYNC_FIXTURE,
                  main_match=False, main_diff_count=1,
                  sync_matches_reference=False, sync_from_input=True,
                  sync_diff_count=FIXTURE_SYNC_BYTES),
             "main-mismatch:e0"),
            ("sibling tail mismatch", "tail",
             dict(kernel=FIXTURE_SIBLING, mode="tail", pool_match=False,
                  pool_diff_count=96, pool_fnv="0x3",
                  pool_first_diff=SYNC_FIXTURE, main_match=False,
                  main_diff_count=1, sync_matches_reference=False,
                  sync_from_input=True, sync_diff_count=FIXTURE_SYNC_BYTES),
             "tail-main-mismatch:e0"),
            ("sibling divergence extent", "execute",
             dict(kernel=FIXTURE_SIBLING, pool_match=False,
                  pool_diff_count=FIXTURE_SYNC_BYTES + 4, pool_fnv="0x3",
                  pool_first_diff=SYNC_FIXTURE,
                  sync_matches_reference=False, sync_diff_count=FIXTURE_SYNC_BYTES),
             "sibling-divergence:e0"),
            ("sibling sync divergence", "execute",
             dict(kernel=FIXTURE_SIBLING, pool_match=False,
                  pool_diff_count=FIXTURE_SYNC_BYTES, pool_fnv="0x3",
                  pool_first_diff=SYNC_FIXTURE,
                  sync_matches_reference=False, sync_diff_count=3),
             "sibling-divergence:e0")):
        labels = enforced_failures(mode, [make_entry(**overrides)],
                                FIXTURE_LIVE, FIXTURE_SYNC_BYTES)
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
             "entry: input digest disagrees"),
            ("sync verdict vs count", dict(sync_matches_reference=True,
                                           sync_diff_count=2),
             "entry: inconsistent sync verdict")):
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

    # Weight-layout known answer: three payloads force the 512-byte alignment
    # (rec0 300 B at heap 0, rec1 700 B at 512, rec2 100 B at 1536).
    container = bytearray(struct.pack("<Q", 0))
    expected_offsets = []
    heap = 0
    for index, payload in enumerate((300, 700, 100)):
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
    known_answers += 4

    # following_records known answers: the lookup matches the payload end
    # rounded up to the 512-byte record alignment — exactly. In this fixture
    # rec0's unaligned 300-byte payload is followed by 212 bytes of padding
    # before rec1 (at 512), and rec1's 700-byte payload by 324 bytes before
    # rec2 (at 1536). The fourth case pins exactness: with a 200-byte
    # hypothetical end (rounded to 1024) nothing is returned even though rec2
    # exists beyond it — the lookup is not "first record at or past the end".
    base.require(following_records(layout, "rec0", 300) == ["rec1"],
                 "self-test: following record across padding")
    base.require(following_records(layout, "rec1", 700) == ["rec2"],
                 "self-test: following record across second padding")
    base.require(following_records(layout, "rec2", 100) == [],
                 "self-test: following record at heap end")
    base.require(following_records(layout, "rec1", 200) == [],
                 "self-test: following record past rounded end")
    known_answers += 4

    # parse_grid known answer and malformed-form control.
    base.require(parse_grid("4,3,4") == (4, 3, 4), "self-test: grid parse")
    known_answers += 1
    try:
        parse_grid("4,3")
    except ValueError as error:
        base.require(str(error) == "grid: malformed geometry",
                     "self-test: grid form wrong message")
        controls += 1
    else:
        raise ValueError("fixture grid corruption accepted")

    # Metadata known answer and malformed-form control.
    info = struct.pack("<BBH", 3, 0x19, 56) + \
        struct.pack("<BBH", 4, 0x17, 12) + struct.pack("<III", 0, 0, 56 << 18)
    parsed = parameter_metadata(info)
    base.require(parsed == dict(sizes=[56], parameters=[(0, 0, 56)]),
                 "self-test: metadata parse")
    known_answers += 1
    try:
        parameter_metadata(struct.pack("<BBH", 9, 0, 0))
    except ValueError as error:
        base.require(str(error) == "abi: unsupported metadata form",
                     "self-test: metadata form wrong message")
        controls += 1
    else:
        raise ValueError("fixture metadata form corruption accepted")

    # Contract-table shape: the class table covers the parameter block exactly,
    # the region words are pool-resident, and exactly one weights word exists.
    contract_checks = 0
    for slot, contract in sorted(CONTRACTS.items()):
        base.require(contract["slot"] == slot, "self-test: contract slot key")
        base.require(len(contract["word_classes"]) == contract["psize"] // 8,
                     f"self-test: contract {slot} class table length")
        base.require(set(contract["constant_values"]) ==
                     {index for index, cls in enumerate(contract["word_classes"])
                      if cls == CLS_CONSTANT},
                     f"self-test: contract {slot} constant words")
        base.require(contract["word_classes"].count(CLS_WEIGHTS) == 1,
                     f"self-test: contract {slot} weights word count")
        base.require(contract["word_classes"][contract["output_word"]] == CLS_POOL and
                     contract["word_classes"][contract["sync_word"]] == CLS_POOL,
                     f"self-test: contract {slot} region word class")
        base.require(0 < contract["sync_bytes"] <= 512,
                     f"self-test: contract {slot} sync extent")
        base.require(contract["output_word"] != contract["sync_word"],
                     f"self-test: contract {slot} output/sync words distinct")
        base.require(len(set(entry_plan(contract))) == 2 * FRAMES,
                     f"self-test: contract {slot} plan distinct entries")
        contract_checks += 8
    known_answers += contract_checks

    return dict(diff_known_answers=3, verdict_controls=12,
                consistency_controls=6, pair_controls=5,
                layout_known_answers=4, following_known_answers=4,
                metadata_known_answers=1,
                grid_known_answers=1,
                contract_known_answers=contract_checks,
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
    parser.add_argument("--slot", type=int, choices=sorted(CONTRACTS),
                        help="replay contract to execute (execute-family modes)")
    args = parser.parse_args()
    if args.command == "self-test":
        print(json.dumps(self_test(), indent=2))
        return 0
    if args.command == "pair":
        if len(args.paths) != 2:
            print("usage: native-slot-replay-probe pair FIRST SECOND")
            return 2
        result = pair(Path(args.paths[0]), Path(args.paths[1]))
        result["with"] = args.paths[1]
        result["run_sha256"] = {
            args.paths[i]: hashlib.sha256(
                (Path(args.paths[i]) / "run.json").read_bytes()).hexdigest()
            for i in (0, 1)}
        (Path(args.paths[0]) / "pair.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 1 if result["pair_failures"] else 0
    if len(args.paths) != 1 or args.slot is None:
        print("usage: native-slot-replay-probe "
              "{execute|skip|wrong|tail} OUTDIR --slot {24,25,26,27,28,29,30,31,32}")
        return 2
    evidence = Path(args.paths[0])
    evidence.mkdir(parents=True, exist_ok=True)
    data = DLL.read_bytes()
    inventory = base.inventory(data)
    progress: dict[str, Any] = {}
    result = execute(data, inventory, args.command, evidence, progress,
                     CONTRACTS[args.slot])
    (evidence / "run.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(
        mode=result["mode"], entries=len(result["entries"]),
        kernels_launched=result["kernels_launched"],
        readbacks=result["readbacks"],
        enforced_failures=result["enforced_failures"]), indent=2))
    return 1 if result["enforced_failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
