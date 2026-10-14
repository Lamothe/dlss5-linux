#!/usr/bin/env python3
"""Name the ladder write-sets from the slots' parameter blocks.

The ladder captures (LADDER.md) measure what each stage of the recovered
156-launch inventory writes into the model's working pool, as the changed
ranges between two cut points. This tool turns those byte ranges into
*tensors*: every pool-resident word of a launch's parameter block is the base
of a buffer the kernel binds (NVAPI-BINDINGS.md), so each binding base starts
a buffer segment and every changed byte is attributed to the segment that
contains it.

Attribution model, stated exactly:

  - The writer of a changed byte is one of the slots in the interval
    (cut_a, cut_b]: slots 0..cut_a run in both captures and cancel in the
    diff (runs are deterministic - established in LADDER.md), and slots above
    cut_b do not run in the later capture.
  - Writes travel through the writing launch's own parameter words at a
    non-negative buffer offset, so the writer's binding base is at or below
    the byte it writes.
  - The interval's own binding bases tile the pool into buffer extents
    (e.g. the s23-s56 interval's base pair 20954112/21298176 spans exactly
    344064 bytes, the out0 tensor shape). Bindings of slots outside the
    interval can sit *inside* those extents - they are aliased sub-views of
    the same region - so they must not split the attribution: segments are
    cut at the interval's bases only.
  - Changed bytes in the segment [b_i, b_i+1) are therefore attributed to
    the interval bindings at base b_i. Where several (slot, word) pairs share
    that base the parameter block does not distinguish writer from reader, so
    all are reported as candidates and no single writer is named.
  - Fail closed: a changed byte below every interval base cannot have been
    written through one, so the run is rejected rather than mis-named.

Every self-test corruption control must trip exactly the check named in
its control string - no more, no less - so each pinned failure mode is
proven reachable and correctly attributed.
"""

from __future__ import annotations

import os
import re
import struct
import sys
from bisect import bisect_right
from pathlib import Path
from typing import Any

NAME = "ladder-names"

FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211

ADDR_MIN = 0x01000000
ADDR_ALIGN = 512
CLEAR_NAME = "cc_cb_clear"
INVENTORY = 156

HEADER_RE = re.compile(
    r"^protocol ladder-capture-v1 frames=(\d+) size=(\d+) context=(\w+) "
    r"pool=(\d+)$")
POOL_RE = re.compile(
    r"^pool-frame=(\d+) bytes=(\d+) file=(\S+) fnv=0x([0-9a-f]{16}) "
    r"va=0x([0-9a-f]+) res_va=0x([0-9a-f]+) res_size=(\d+)$")
LAUNCH_RE = re.compile(
    r"launch id=0x5c52bb86 .*handle=(0x[0-9a-f]+) grid=\S+ params=\S+ "
    r"psize=(\d+) pn=(\d+) pcap=(\d+) pmem=([0-9a-f]*)")
CREATED_RE = re.compile(r"created .*name=(\S+) handle=(0x[0-9a-f]+)")


class Reject(Exception):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Reject(message)


def fnv1a64(data: bytes) -> int:
    hash_value = FNV_OFFSET
    for byte in data:
        hash_value ^= byte
        hash_value = (hash_value * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return hash_value


def replace_bytes(path: Path, data: bytes) -> None:
    """Swap in a new file rather than truncating: self-test fixtures share
    hardlinks with their base captures, so a mutation must not overwrite the
    shared inode."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def read_capture(root: Path) -> dict[str, Any]:
    """Parse one ladder capture: header, pool records, pool bytes."""
    manifest = root / "manifest.txt"
    require(manifest.is_file(), "capture: manifest missing")
    lines = manifest.read_text(encoding="utf-8", errors="replace").splitlines()
    require(len(lines) >= 2, "capture: manifest truncated")
    header = HEADER_RE.match(lines[0])
    require(header is not None, "capture: header format")
    capture: dict[str, Any] = {
        "frames": int(header.group(1)),
        "pool_size": int(header.group(4)),
        "pools": [],
    }
    require(capture["frames"] > 0, "capture: frame count")
    for line in lines[2:]:
        if not line or line.startswith(("boundary-frame=", "sidecar=")):
            continue
        if pool := POOL_RE.match(line):
            capture["pools"].append({
                "frame": int(pool.group(1)),
                "bytes": int(pool.group(2)),
                "file": pool.group(3),
                "fnv": int(pool.group(4), 16),
                "va": int(pool.group(5), 16),
                "res_va": int(pool.group(6), 16),
                "res_size": int(pool.group(7)),
            })
    require(len(capture["pools"]) == capture["frames"],
            "capture: pool record count")
    for index, pool in enumerate(capture["pools"]):
        require(pool["frame"] == index, "capture: pool frame indexing")
        require(pool["bytes"] == capture["pool_size"], "capture: pool bytes")
        require(pool["res_size"] == capture["pool_size"],
                "capture: pool resource size")
        require(pool["va"] == pool["res_va"], "capture: pool copy base")
        path = root / pool["file"]
        require(path.is_file(), "capture: pool file missing")
        data = path.read_bytes()
        require(len(data) == pool["bytes"], "capture: pool file size")
        require(fnv1a64(data) == pool["fnv"], "capture: pool digest")
        pool["data"] = data
    va = capture["pools"][0]["res_va"]
    require(all(p["res_va"] == va for p in capture["pools"]),
            "capture: pool address varies")
    capture["pool_va"] = va
    slot_path = root / "slot.txt"
    require(slot_path.is_file(), "capture: slot.txt missing")
    match = re.match(r"^slot=(\d+)$", slot_path.read_text().strip())
    require(match is not None, "capture: slot.txt format")
    capture["slot"] = int(match.group(1))
    return capture


def read_trace(path: Path, inventory: int = INVENTORY) -> dict[str, Any]:
    """Parse one forwarder trace into per-eval slot records.

    Every launch is logged with its parameter dump even when the cut drops it,
    so one trace carries all `inventory` slots' words for every eval.
    """
    require(path.is_file(), "trace: trace missing")
    names: dict[str, str] = {}
    launches: list[tuple[str, int, int, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := CREATED_RE.search(line):
            names[match.group(2)] = match.group(1)
        if match := LAUNCH_RE.search(line):
            handle, psize, pn, pcap, pmem = match.groups()
            launches.append((handle, int(psize), int(pcap), pmem))
    require(len(names) > 0, "trace: no created names")
    clears = [h for h, n in names.items() if n == CLEAR_NAME]
    require(len(clears) == 1, "trace: clear kernel handle")
    clear = clears[0]
    evals: list[list[tuple[str, int, int, str]]] = []
    current: list[tuple[str, int, int, str]] = []
    for launch in launches:
        if launch[0] == clear and current:
            evals.append(current)
            current = []
        current.append(launch)
    if current:
        evals.append(current)
    require(len(evals) > 0, "trace: eval count")
    for group in evals:
        require(group[0][0] == clear, "trace: clear at slot 0")
        require(len(group) == inventory, "trace: launches per eval")
    return {"names": names, "evals": evals, "clear": clear}


def pool_bindings(trace: dict[str, Any], eval_index: int,
                  pool_va: int, pool_size: int) -> list[dict[str, Any]]:
    """Every pool-resident (slot, kernel, word, offset) of one eval."""
    bindings: list[dict[str, Any]] = []
    for slot, (handle, psize, pcap, pmem) in enumerate(
            trace["evals"][eval_index]):
        require(pcap == 0 and len(pmem) == 2 * psize, "words: pmem length")
        # Words are read 8 bytes at a time: a non-multiple size would raise
        # struct.error past the Reject boundary (NVAPI-BINDINGS word shape).
        require(psize > 0 and psize % 8 == 0, "words: word shape")
        data = bytes.fromhex(pmem)
        for word in range(0, len(data), 8):
            value = struct.unpack_from("<Q", data, word)[0]
            if (ADDR_MIN <= value < 0x100000000 and value % ADDR_ALIGN == 0
                    and pool_va <= value < pool_va + pool_size):
                bindings.append({
                    "slot": slot,
                    "kernel": trace["names"].get(handle, "<unknown>"),
                    "word": word // 8,
                    "offset": value - pool_va,
                })
    require(len(bindings) > 0, "words: no pool bindings")
    return bindings


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def changed_ranges(a: bytes, b: bytes) -> tuple[int, list[tuple[int, int]]]:
    """Total changed bytes and coalesced [start, end) ranges of two pools.

    The two pools are compared as big integers (C-speed) and the differing
    bytes located by pattern matching over the XOR image: a byte differs
    exactly where the XOR image is non-zero, so its maximal non-zero runs are
    the coalesced changed ranges.
    """
    require(len(a) == len(b), "ranges: pool size mismatch")
    length = len(a)
    image = (int.from_bytes(a, "big") ^ int.from_bytes(b, "big")).to_bytes(
        length, "big")
    total = length - image.count(0)
    spans = [(m.start(), m.end()) for m in re.finditer(rb"[^\x00]+", image)]
    return total, spans


def name_interval(first: dict[str, Any], second: dict[str, Any],
                  trace_a: dict[str, Any], trace_b: dict[str, Any]) \
        -> dict[str, Any]:
    """Attribute each frame's changed bytes to slot/kernel/word bindings."""
    require(first["pool_size"] == second["pool_size"],
            "interval: pool size mismatch")
    require(first["pool_va"] == second["pool_va"], "interval: pool mismatch")
    require(first["frames"] == second["frames"], "interval: frame count")
    require(first["slot"] < second["slot"], "interval: cut order")
    require(len(trace_a["evals"]) == first["frames"],
            "interval: trace eval count")
    require(len(trace_b["evals"]) == first["frames"],
            "interval: trace eval count")
    pool_size = first["pool_size"]
    pool_va = first["pool_va"]
    low, high = first["slot"], second["slot"]

    frames: list[dict[str, Any]] = []
    binding_sets_a: list[frozenset[tuple[int, int, int]]] = []
    binding_sets_b: list[frozenset[tuple[int, int, int]]] = []
    for frame in range(first["frames"]):
        binds_a = pool_bindings(trace_a, frame, pool_va, pool_size)
        binds_b = pool_bindings(trace_b, frame, pool_va, pool_size)
        keys_a = frozenset(
            (b["slot"], b["word"], b["offset"]) for b in binds_a)
        keys_b = frozenset(
            (b["slot"], b["word"], b["offset"]) for b in binds_b)
        binding_sets_a.append(keys_a)
        binding_sets_b.append(keys_b)
        # The interval's word table must agree across the two runs: both runs
        # execute the same slots with the same arguments (determinism).
        interval_a = frozenset(k for k in keys_a if low < k[0] <= high)
        interval_b = frozenset(k for k in keys_b if low < k[0] <= high)
        require(interval_a == interval_b,
                "interval: cross-run binding mismatch")
        binds = [b for b in binds_b if low < b["slot"] <= high]

        changed, ranges = changed_ranges(first["pools"][frame]["data"],
                                         second["pools"][frame]["data"])
        # Segments are cut at the *interval's* binding bases only. Bindings of
        # out-of-interval slots can sit inside an interval buffer as aliased
        # sub-views (the s23-s56 case: slots 17/18/133/134 bind interior
        # offsets of the 344064-byte out0 extent), so cutting on every binding
        # would shred one buffer into many. Writes reach their bytes through
        # the writer's own binding at a non-negative offset, so a changed byte
        # below every interval base cannot have been written by the interval
        # and the run is rejected rather than mis-named.
        bases = sorted(set(b["offset"] for b in binds))
        require(len(bases) > 0, "words: no interval bindings")
        if ranges:
            require(ranges[0][0] >= bases[0],
                    "attribution: change below first binding")
        by_base: dict[int, list[dict[str, Any]]] = {}
        for b in binds:
            by_base.setdefault(b["offset"], []).append(b)
        alias_at: dict[int, list[dict[str, Any]]] = {}
        for b in binds_b:
            if low >= b["slot"] or b["slot"] > high:
                alias_at.setdefault(b["offset"], []).append(b)

        segments: dict[int, dict[str, Any]] = {}
        for start, end in ranges:
            pos = start
            while pos < end:
                index = bisect_right(bases, pos) - 1
                base = bases[index]
                seg_end = bases[index + 1] if index + 1 < len(bases) \
                    else pool_size
                stop = min(end, seg_end)
                entry = segments.get(base)
                if entry is None:
                    entry = segments[base] = {
                        "offset": base, "changed": 0,
                        "first": pos, "last": stop,
                        "base_anchored": pos == base,
                    }
                entry["changed"] += stop - pos
                entry["first"] = min(entry["first"], pos)
                entry["last"] = max(entry["last"], stop)
                if pos == base:
                    entry["base_anchored"] = True
                pos = stop
        for base, entry in segments.items():
            entry["slots"] = sorted(
                (b["slot"], b["kernel"], b["word"])
                for b in by_base.get(base, []))
            # Out-of-interval aliases are informational: they name sub-views
            # of the same region, not separate writers. Scan the full segment
            # [base, seg_end) - the unchanged tail included - so the recorded
            # alias set covers the buffer, not just its changed part.
            index = bases.index(base)
            seg_end = bases[index + 1] if index + 1 < len(bases) else pool_size
            entry["aliases"] = sorted(
                (b["slot"], b["kernel"], b["word"], b["offset"])
                for b in binds_b
                if base <= b["offset"] < seg_end
                and not (low < b["slot"] <= high))
        attributed = sum(e["changed"] for e in segments.values())
        frames.append({
            "frame": frame,
            "changed": changed,
            "ranges": len(ranges),
            "attributed": attributed,
            "segments": sorted(segments.values(),
                               key=lambda e: -e["changed"]),
        })

    require(len(set(binding_sets_a)) == 1 and len(set(binding_sets_b)) == 1,
            "words: bindings vary across evals")
    return {
        "interval": [low, high],
        "pool_va": pool_va,
        "pool_size": pool_size,
        "frames": frames,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

CONTROLS = [
    "capture: frame count",
    "capture: pool digest",
    "capture: pool bytes",
    "capture: pool address varies",
    "capture: pool file missing",
    "words: pmem length",
    "words: word shape",
    "words: no pool bindings",
    "words: no interval bindings",
    "trace: clear at slot 0",
    "trace: launches per eval",
    "interval: cross-run binding mismatch",
    "attribution: change below first binding",
    "words: bindings vary across evals",
]

# Fixture geometry: a 4096-byte pool at VA 0x01000000 with 512-byte-aligned
# binding bases at offsets 0 (clear), 512 (writer, the interval slot) and
# 1024 (other, outside the interval). The pool VA sits at ADDR_MIN so the
# words pass the address-shape rule exactly as real device VAs do. The second
# capture changes 64 bytes at the writer's base.
POOL_BYTES_FIXTURE = 4096
POOL_VA_FIXTURE = 0x01000000
BASE_CLEAR = POOL_VA_FIXTURE + 0
BASE_WRITER = POOL_VA_FIXTURE + 512
BASE_OTHER = POOL_VA_FIXTURE + 1024
BASE_ALT = POOL_VA_FIXTURE + 1536
BASE_ALT2 = POOL_VA_FIXTURE + 2048


def write_fixture(root: Path, slot: int, written: bool) -> None:
    """One capture of the tiny two-segment arena (2 frames).

    `written` captures change bytes 512..576 - inside the interval slot's
    buffer at base 512.
    """
    root.mkdir(parents=True, exist_ok=True)
    pool = bytearray(b"\x00" * POOL_BYTES_FIXTURE)
    if written:
        pool[512:576] = b"\x11" * 64
    pool_bytes = bytes(pool)
    lines = ["protocol ladder-capture-v1 frames=2 size=8 context=eye0 "
             f"pool={POOL_BYTES_FIXTURE}",
             "sidecar=nvapi-ladder-x.txt",
             "boundary-frame=0 eval=0 psize=8 pmem=" + "00" * 8,
             "boundary-frame=1 eval=1 psize=8 pmem=" + "00" * 8]
    for frame in range(2):
        (root / f"pool-frame{frame:02d}.bin").write_bytes(pool_bytes)
        lines.append(f"pool-frame={frame} bytes={POOL_BYTES_FIXTURE} "
                     f"file=pool-frame{frame:02d}.bin "
                     f"fnv=0x{fnv1a64(pool_bytes):016x} va={POOL_VA_FIXTURE:#x} "
                     f"res_va={POOL_VA_FIXTURE:#x} res_size={POOL_BYTES_FIXTURE}")
    (root / "manifest.txt").write_text("\n".join(lines) + "\n")
    (root / "slot.txt").write_text(f"slot={slot}\n")


def launch_line(handle: str, base: int) -> str:
    pmem = struct.pack("<QQ", base, 0).hex()
    return (f"000001 709714463 352 launch id=0x5c52bb86 cmd=0x0 "
            f"handle={handle} grid=1,1,1 params=0x1 psize=16 pn=16 "
            f"pcap=0 pmem={pmem}")


def write_trace(path: Path) -> None:
    """Two identical evals over a 4-launch inventory.

    Slots: 0 = clear (binds pool+0), 1 = writer (binds pool+512),
    2 = other (binds pool+1024), 3 = writer again (binds pool+512).
    """
    lines = ["000000 709714463 352 created name=cc_cb_clear handle=0x10",
             "000000 709714463 352 created name=writer handle=0x20",
             "000000 709714463 352 created name=other handle=0x30"]
    for _ in range(2):
        lines += [launch_line("0x10", BASE_CLEAR),
                  launch_line("0x20", BASE_WRITER),
                  launch_line("0x30", BASE_OTHER),
                  launch_line("0x20", BASE_WRITER)]
    path.write_text("\n".join(lines) + "\n")


def repin_pool(root: Path, frame: int, payload: bytes) -> None:
    """Replace a pool file and repin its manifest digest."""
    replace_bytes(root / f"pool-frame{frame:02d}.bin", payload)
    text = (root / "manifest.txt").read_text()
    text = re.sub(
        rf"(^pool-frame={frame} bytes=\d+ file=\S+ fnv=)0x[0-9a-f]{{16}}",
        rf"\g<1>0x{fnv1a64(payload):016x}", text, count=1,
        flags=re.MULTILINE)
    replace_bytes(root / "manifest.txt", text.encode())


def mutate(root: Path, first: Path, control: str) -> None:
    """Corrupt exactly one input so exactly the named check must fire."""
    writer_hex = struct.pack("<QQ", BASE_WRITER, 0).hex()
    alt_hex = struct.pack("<QQ", BASE_ALT, 0).hex()
    other_hex = struct.pack("<QQ", BASE_OTHER, 0).hex()
    if control == "capture: frame count":
        text = (root / "manifest.txt").read_text()
        replace_bytes(root / "manifest.txt",
                      text.replace("frames=2", "frames=0", 1).encode())
    elif control == "capture: pool digest":
        data = bytearray((root / "pool-frame00.bin").read_bytes())
        data[2048] ^= 0x01
        replace_bytes(root / "pool-frame00.bin", bytes(data))
    elif control == "capture: pool bytes":
        text = (root / "manifest.txt").read_text()
        replace_bytes(root / "manifest.txt",
                      re.sub(rf"(^pool-frame=0 bytes=){POOL_BYTES_FIXTURE}",
                             r"\g<1>512", text, count=1,
                             flags=re.MULTILINE).encode())
    elif control == "capture: pool address varies":
        # Keep va == res_va on the mutated frame (copy-base check) but move
        # frame 1 to a different address, as a per-eye flip would.
        text = (root / "manifest.txt").read_text()
        text = re.sub(
            r"(^pool-frame=1 \S+ \S+ \S+ va=)0x[0-9a-f]+( res_va=)0x[0-9a-f]+",
            r"\g<1>0x01002000\g<2>0x01002000", text, count=1, flags=re.MULTILINE)
        replace_bytes(root / "manifest.txt", text.encode())
    elif control == "capture: pool file missing":
        (root / "pool-frame00.bin").unlink()
    elif control == "words: pmem length":
        text = (root / "trace-b.log").read_text()
        index = text.index(f"pmem={writer_hex}")
        line_start = text.rfind("\n", 0, index) + 1
        line_end = text.index("\n", index)
        line = text[line_start:line_end]
        shortened = line[:line.index("pmem=") + 5] + writer_hex[:2]
        replace_bytes(root / "trace-b.log",
                      (text[:line_start] + shortened + text[line_end:])
                      .encode())
    elif control == "trace: clear at slot 0":
        # Move the writer launch ahead of the clear in eval 0: the eval is
        # still complete but no longer starts with the clear kernel.
        text = (root / "trace-b.log").read_text()
        lines = text.splitlines(keepends=True)
        index = next(i for i, line in enumerate(lines)
                     if "handle=0x10 " in line)
        lines[index], lines[index + 1] = lines[index + 1], lines[index]
        replace_bytes(root / "trace-b.log", "".join(lines).encode())
    elif control == "trace: launches per eval":
        text = (root / "trace-b.log").read_text()
        lines = text.splitlines(keepends=True)
        index = next(i for i, line in enumerate(lines)
                     if "handle=0x30 " in line)
        del lines[index]
        replace_bytes(root / "trace-b.log", "".join(lines).encode())
    elif control == "interval: cross-run binding mismatch":
        # Shift the writer's binding in trace-a only; the interval word table
        # then disagrees across the runs.
        text = (first / "trace-a.log").read_text()
        replace_bytes(first / "trace-a.log",
                      text.replace(f"pmem={writer_hex}",
                                   f"pmem={alt_hex}", 1).encode())
    elif control == "attribution: change below first binding":
        # Change a byte below every interval binding base: no interval slot
        # can reach it at a non-negative buffer offset. Repinned so the digest
        # check cannot fire first.
        data = bytearray((root / "pool-frame00.bin").read_bytes())
        data[100] ^= 0x01
        repin_pool(root, 0, bytes(data))
    elif control == "words: word shape":
        # Odd psize with a matching pmem length: the length check passes and
        # only the 8-byte word-shape rule can fire.
        text = (root / "trace-b.log").read_text()
        marker = f"psize=16 pn=16 pcap=0 pmem={writer_hex}"
        replacement = f"psize=15 pn=15 pcap=0 pmem={writer_hex[:30]}"
        require(marker in text, "fixture: writer launch marker")
        replace_bytes(root / "trace-b.log",
                      text.replace(marker, replacement, 1).encode())
    elif control == "words: no pool bindings":
        # Zero every word of trace-b: no pool address survives the
        # address-shape filter and the binding table comes back empty. The
        # hex length is preserved, so the pmem-length check cannot fire first.
        text = (root / "trace-b.log").read_text()
        text = re.sub(r"pmem=[0-9a-f]+",
                      lambda m: "pmem=" + "0" * (len(m.group(0)) - 5),
                      text)
        replace_bytes(root / "trace-b.log", text.encode())
    elif control == "words: no interval bindings":
        # Unalign the interval slot's word in both traces: it drops out of the
        # binding tables symmetrically (cross-run still agrees) and the
        # interval ends up with no binding at all. The pmem hex length is
        # unchanged, so the pmem-length check cannot fire first.
        for target in (first / "trace-a.log", root / "trace-b.log"):
            text = target.read_text()
            odd = struct.pack("<QQ", BASE_WRITER + 1, 0).hex()
            text = text.replace(f"pmem={writer_hex}", f"pmem={odd}")
            replace_bytes(target, text.encode())
    elif control == "words: bindings vary across evals":
        # Rebind the "other" slot's launch in eval 1 of trace-b only: the two
        # runs still agree on the interval (cross-run holds), and the changed
        # bytes beneath the writer stay attributed, but trace-b's own binding
        # set now differs between its evals. Mutating one trace exercises the
        # keys_b half of the stability check. "other" is outside the interval
        # and writes nothing, so no other check can fire first.
        other_hex = struct.pack("<QQ", BASE_OTHER, 0).hex()
        moved_hex = struct.pack("<QQ", BASE_ALT2, 0).hex()
        target = root / "trace-b.log"
        text = target.read_text()
        lines = text.splitlines(keepends=True)
        other_lines = [i for i, line in enumerate(lines)
                       if f"pmem={other_hex}" in line]
        require(len(other_lines) == 2, "fixture: other launch count")
        line = lines[other_lines[1]]
        lines[other_lines[1]] = line.replace(f"pmem={other_hex}",
                                             f"pmem={moved_hex}")
        replace_bytes(target, "".join(lines).encode())
    else:
        raise AssertionError(f"unknown control {control}")


def self_test() -> None:
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        good = base / "first"
        other = base / "second"
        write_fixture(good, slot=0, written=False)
        write_fixture(other, slot=1, written=True)
        write_trace(good / "trace-a.log")
        write_trace(good / "trace-b.log")
        write_trace(other / "trace-a.log")
        write_trace(other / "trace-b.log")

        def check(first: Path, second: Path) -> dict[str, Any]:
            return name_interval(
                read_capture(first), read_capture(second),
                read_trace(first / "trace-a.log", inventory=4),
                read_trace(second / "trace-b.log", inventory=4))

        result = check(good, other)
        require(result["frames"][0]["changed"] == 64,
                "self-test: baseline changed bytes")
        require(result["frames"][0]["segments"][0]["offset"] == 512,
                "self-test: baseline segment base")
        require(result["frames"][0]["segments"][0]["slots"][0][0] == 1,
                "self-test: baseline named slot")
        for control in CONTROLS:
            target = base / control.replace(" ", "_").replace(":", "-")
            shutil.copytree(other, target, copy_function=os.link)
            shutil.copytree(good, target / "_a", copy_function=os.link)
            mutate(target, target / "_a", control)
            try:
                check(target / "_a", target)
            except Reject as error:
                require(str(error) == control,
                        f"self-test: control {control} tripped {error}")
            else:
                raise Reject(f"self-test: control {control} was not rejected")
    print(f"PASS: {NAME} self-test controls={len(CONTROLS)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def report(result: dict[str, Any]) -> None:
    """Print every segment: these reports are the evidence the write-up
    cites, so nothing is elided."""
    low, high = result["interval"]
    print(f"interval {low}-{high} pool=0x{result['pool_va']:x} "
          f"size={result['pool_size']}")
    for frame in result["frames"]:
        print(f" frame {frame['frame']}: changed={frame['changed']} "
              f"ranges={frame['ranges']} attributed={frame['attributed']} "
              f"segments={len(frame['segments'])}")
        for entry in frame["segments"]:
            who = ", ".join(f"slot {s} {k} w{w}" for s, k, w in entry["slots"])
            print(f"   +{entry['offset']:<8} changed={entry['changed']:<8} "
                  f"span=[{entry['first']},{entry['last']}) "
                  f"anchored={int(entry['base_anchored'])} {who}")
            for slot, kernel, word, offset in entry["aliases"]:
                print(f"       alias +{offset} slot {slot} {kernel} w{word}")


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "self-test":
        self_test()
        return 0
    if args and args[0] == "names" and len(args) == 5:
        try:
            result = name_interval(read_capture(Path(args[1])),
                                   read_capture(Path(args[2])),
                                   read_trace(Path(args[3])),
                                   read_trace(Path(args[4])))
        except Reject as error:
            print(f"REJECT: {error}")
            return 1
        report(result)
        return 0
    print("usage: ladder-names {self-test|names FIRST SECOND TRACE-A TRACE-B}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
