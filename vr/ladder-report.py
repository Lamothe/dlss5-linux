#!/usr/bin/env python3
"""Validate, compare and diff intermediate-ladder captures.

An intermediate-ladder capture (vr/LADDER.md) truncates the model's evaluation
at one recovered slot of the 156-launch sequence (NVAPI-ARGS.md) and reads the
model's entire working pool at that point. Consecutive cut points can then be
diffed into the write-set of the slots between them.

Subcommands
  validate DIR [--trace T] [--slot N]   pinned checks over one capture
  pair A B                              two captures at the same slot are identical
  diff A B                              changed pool bytes between two cut points
  self-test                             corruption controls, each rejected

Evidence carries digests, counts and geometry only - never tensor bytes. The
retained .bin windows stay local (gitignored) and carry the proprietary
data-handling caveats of the earlier stages.
"""

from __future__ import annotations

import argparse
import os
import re
import struct
import sys
from pathlib import Path
from typing import Any

NAME = "ladder-report"
FRAMES = 3
PROBE_SIZE = 720
NAN_HALF = 0x7E00
POOL_SIZE = 62684160
INVENTORY = 156
# The output-texture writer is a measured slot of the recovered inventory: the
# s153 capture leaves the FP16 NaN sentinel intact and the s154 capture writes
# it (both retained under vr/logs/ladder, gate stage 15). A cut at slot n runs
# slots 0..n, so the expected output state is "written" exactly for n >= this
# slot. The probe records the state; this constant pins the expectation.
LADDER_OUTPUT_WRITER_SLOT = 154

FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211

HEADER_RE = re.compile(
    r"^protocol ladder-capture-v1 frames=(\d+) size=(\d+) context=(\w+) "
    r"pool=(\d+)$")
BOUNDARY_RE = re.compile(
    r"^boundary-frame=(\d+) eval=(\d+) psize=(\d+) pmem=([0-9a-f]+)$")
POOL_RE = re.compile(
    r"^pool-frame=(\d+) bytes=(\d+) file=(\S+) fnv=0x([0-9a-f]{16}) "
    r"va=0x([0-9a-f]+) res_va=0x([0-9a-f]+) res_size=(\d+)$")
SIDE_RE = re.compile(r"^boundary eval=(\d+) psize=(\d+) pmem=([0-9a-f]+)$")
FILE_RE = re.compile(r"^(\S+) bytes=(\d+) fnv=0x([0-9a-f]{16})$")
# Trace lines carry a "seq time thread " prefix from log_line, so this must not
# anchor at the start of the line. The fixture emits the same prefix format.
CUT_BOUNDARY_RE = re.compile(r"\bcut-boundary eval=(\d+) slot=(\d+) psize=(\d+)\b",
                            re.MULTILINE)
CUT_ANOMALY_RE = re.compile(
    r"cut-(anomaly|window-exhausted|unknown|name-table-full|clear-unknown|"
    r"pmem-incomplete|sidecar-write-failed) ")


class Reject(Exception):
    """A pinned check failed. The message is the check name + what broke."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Reject(message)


def replace_bytes(path: Path, data: bytes) -> None:
    """Replace a fixture file's content without truncating it in place.

    Self-test fixtures share hardlinks with the good capture to avoid copying
    ~190 MB per control, so a mutation must swap in a new file rather than
    overwrite the shared inode.
    """
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def fnv1a64(data: bytes) -> int:
    hash_value = FNV_OFFSET
    for byte in data:
        hash_value ^= byte
        hash_value = (hash_value * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return hash_value


# Digests are memoised by file identity (device, inode, size, mtime). Self-test
# fixtures share hardlinked pool files, so this turns ~12 full rehashes into
# one; a mutation swaps in a new inode via replace_bytes and is never a hit.
_DIGEST_CACHE: dict[tuple[int, int, int, int], int] = {}


def file_digest(path: Path, message: str) -> tuple[bytes, int]:
    data = read_capture_file(path.parent, path.name, message)
    stat = path.stat()
    key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    cached = _DIGEST_CACHE.get(key)
    if cached is None:
        cached = fnv1a64(data)
        _DIGEST_CACHE[key] = cached
    return data, cached


def nan_sentinel_bytes() -> bytes:
    return struct.pack("<4H", NAN_HALF, NAN_HALF, NAN_HALF, NAN_HALF) * (
        PROBE_SIZE * PROBE_SIZE)


def read_capture_file(root: Path, name: str, message: str) -> bytes:
    path = root / name
    require(path.is_file(), message)
    return path.read_bytes()


# ---------------------------------------------------------------------------
# Manifest parsing
# ---------------------------------------------------------------------------


def read_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.txt"
    require(path.is_file(), "manifest-protocol: manifest missing")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    require(len(lines) >= 2, "manifest-protocol: manifest truncated")
    header = HEADER_RE.match(lines[0])
    require(header is not None, "manifest-protocol: header format")
    require(lines[1].startswith("sidecar="), "manifest-protocol: sidecar line")

    manifest: dict[str, Any] = {
        "frames": int(header.group(1)),
        "size": int(header.group(2)),
        "context": header.group(3),
        "pool_size": int(header.group(4)),
        "boundary": [],
        "pools": [],
        "files": {},
    }
    for line in lines[2:]:
        if not line:
            continue
        if boundary := BOUNDARY_RE.match(line):
            manifest["boundary"].append({
                "frame": int(boundary.group(1)),
                "eval": int(boundary.group(2)),
                "psize": int(boundary.group(3)),
                "pmem": boundary.group(4),
            })
            continue
        if pool := POOL_RE.match(line):
            manifest["pools"].append({
                "frame": int(pool.group(1)),
                "bytes": int(pool.group(2)),
                "file": pool.group(3),
                "fnv": int(pool.group(4), 16),
                "va": int(pool.group(5), 16),
                "res_va": int(pool.group(6), 16),
                "res_size": int(pool.group(7)),
            })
            continue
        if file := FILE_RE.match(line):
            manifest["files"][file.group(1)] = {
                "bytes": int(file.group(2)),
                "fnv": int(file.group(3), 16),
            }
            continue
        require(False, f"manifest-protocol: unrecognised line {line[:40]!r}")
    return manifest


def read_sidecar(root: Path) -> list[dict[str, int | str]]:
    path = root / "boundary-va.txt"
    require(path.is_file(), "sidecar-records: sidecar missing")
    records: list[dict[str, int | str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line:
            continue
        match = SIDE_RE.match(line)
        require(match is not None, "sidecar-records: sidecar line format")
        records.append({
            "eval": int(match.group(1)),
            "psize": int(match.group(2)),
            "pmem": match.group(3),
        })
    return records


def read_slot(root: Path) -> int:
    path = root / "slot.txt"
    require(path.is_file(), "slot-pinned: slot.txt missing")
    match = re.match(r"^slot=(\d+)$", path.read_text(encoding="utf-8").strip())
    require(match is not None, "slot-pinned: slot.txt format")
    return int(match.group(1))


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_capture(root: Path, trace: Path | None, slot: int | None) -> dict[str, Any]:
    """Pinned checks over one capture. Every failure names the check."""
    manifest = read_manifest(root)
    require(manifest["frames"] == FRAMES, "manifest-protocol: frame count")
    require(manifest["size"] == PROBE_SIZE, "manifest-protocol: probe size")
    require(manifest["context"] == "eye0", "manifest-protocol: context")
    require(manifest["pool_size"] == POOL_SIZE, "manifest-protocol: pool size")

    recorded = read_slot(root)
    if slot is not None:
        require(recorded == slot, "slot-pinned: slot.txt vs --slot")
    require(0 <= recorded <= INVENTORY - 1, "slot-pinned: slot out of range")

    sidecar = read_sidecar(root)
    require(len(sidecar) == FRAMES, "sidecar-records: record count")
    require(len(manifest["boundary"]) == FRAMES, "sidecar-records: manifest count")
    for frame, (entry, boundary) in enumerate(zip(sidecar, manifest["boundary"])):
        require(boundary["frame"] == frame, "sidecar-records: frame indexing")
        require(boundary["eval"] == entry["eval"], "sidecar-records: eval vs sidecar")
        require(boundary["psize"] == entry["psize"], "sidecar-records: psize")
        require(boundary["pmem"] == entry["pmem"], "sidecar-records: pmem")
        require(len(boundary["pmem"]) == 2 * boundary["psize"],
                "sidecar-records: pmem length")

    require(len(manifest["pools"]) == FRAMES, "pool-records: record count")
    result: dict[str, Any] = {"slot": recorded, "digests": {}}
    pool_va: int | None = None
    for frame, pool in enumerate(manifest["pools"]):
        require(pool["frame"] == frame, "pool-records: frame indexing")
        require(pool["bytes"] == POOL_SIZE, "pool-records: pool bytes")
        require(pool["res_size"] == POOL_SIZE, "pool-records: resource size")
        require(pool["va"] == pool["res_va"], "pool-records: copy from resource base")
        # Every frame of one capture must have read the same pool: the two
        # per-eye pools are the same size, so a flip between them would pass
        # every check above and make the capture internally inconsistent.
        if pool_va is None:
            pool_va = pool["res_va"]
        else:
            require(pool["res_va"] == pool_va, "pool-records: pool address varies")
        require(pool["file"] == f"pool-frame{frame:02d}.bin",
                "pool-records: file naming")
        data, digest = file_digest(root / pool["file"], "pool-file: pool file missing")
        require(len(data) == pool["bytes"], "pool-file: pool file size")
        require(digest == pool["fnv"], "pool-digest: pool digest")
        result["digests"][f"pool-frame{frame}"] = f"0x{pool['fnv']:016x}"
    result["pool_va"] = pool_va

    nan_digest = fnv1a64(nan_sentinel_bytes())
    # The output canary is slot-aware (LADDER_OUTPUT_WRITER_SLOT): a cut before
    # the writer must leave the NaN sentinel (proving the cut held); a cut at
    # or past it must show a written output (proving the writer ran). Either
    # mismatch means the capture does not describe its own cut point.
    writer = recorded >= LADDER_OUTPUT_WRITER_SLOT
    for frame in range(FRAMES):
        record = manifest["files"].get(f"output-frame{frame:02d}.bin")
        require(record is not None, "output-record: output record missing")
        data, digest = file_digest(root / f"output-frame{frame:02d}.bin",
                                   "output-record: output missing")
        require(len(data) == record["bytes"], "output-record: output size")
        require(digest == record["fnv"], "output-record: output digest")
        if writer:
            require(record["fnv"] != nan_digest,
                    "output-state: untouched at the writer slot")
        else:
            require(record["fnv"] == nan_digest,
                    "output-state: written before the writer slot")
    for name in ("input-colour.bin", "input-depth.bin", "input-motion.bin"):
        record = manifest["files"].get(name)
        require(record is not None, "input-records: input record missing")
        data, digest = file_digest(root / name, "input-records: input missing")
        require(len(data) == record["bytes"] and digest == record["fnv"],
                "input-records: input digest")
        result["digests"][name] = f"0x{record['fnv']:016x}"

    if trace is not None:
        require(trace.is_file(), "cut-accounting: trace missing")
        text = trace.read_text(encoding="utf-8", errors="replace")
        require(CUT_ANOMALY_RE.search(text) is None, "cut-accounting: anomaly records")
        armed = len(re.findall(r"cut-armed ", text))
        drops = len(re.findall(r"cut-drop ", text))
        releases = len(re.findall(r"cut-release ", text))
        require(armed == FRAMES, "cut-accounting: armed evaluations")
        # Everything after the cut point is withheld: slots recorded+1 .. 155.
        require(drops == (INVENTORY - 1 - recorded) * armed,
                "cut-accounting: withheld launches")
        require(releases == armed - 1, "cut-accounting: releases")
        boundaries = [int(m.group(2)) for m in CUT_BOUNDARY_RE.finditer(text)]
        require(len(boundaries) == armed, "cut-boundary-slot: boundary records")
        require(all(s == recorded for s in boundaries),
                "cut-boundary-slot: slot mismatch")
    return result


def input_digests(digests: dict[str, str]) -> dict[str, str]:
    """The controlled inputs, separate from the pool content."""
    return {k: v for k, v in digests.items() if not k.startswith("pool-")}


def check_pair(first: Path, second: Path, trace_a: Path | None,
               trace_b: Path | None) -> dict[str, Any]:
    """Two captures of the same cut point must be bit-identical.

    Three independent conditions, checked in the order they establish things:
    the same pool, the same inputs, then the same bytes. There is deliberately
    no separate digest-summary comparison - check_capture already pins every
    recorded digest to its file bytes, so the summaries can only differ when
    the bytes already did.
    """
    left = check_capture(first, trace_a, None)
    right = check_capture(second, trace_b, None)
    require(left["slot"] == right["slot"], "runs-identical: different cut points")
    require(left["pool_va"] == right["pool_va"], "runs-identical: different pools")
    require(input_digests(left["digests"]) == input_digests(right["digests"]),
            "runs-identical: inputs differ")
    for frame in range(FRAMES):
        a = read_capture_file(first, f"pool-frame{frame:02d}.bin",
                              "runs-identical: pool missing")
        b = read_capture_file(second, f"pool-frame{frame:02d}.bin",
                              "runs-identical: pool missing")
        require(a == b, "runs-identical: pool bytes differ")
    return {"slot": left["slot"], "pool_va": left["pool_va"],
            "digests": left["digests"]}


def check_diff(first: Path, second: Path) -> dict[str, Any]:
    """Changed pool bytes between two cut points: the intervening write-set.

    The two captures must share their inputs and their temporal state, so only
    a pair of cut points from the same controlled protocol is comparable.
    """
    left = check_capture(first, None, None)
    right = check_capture(second, None, None)
    low, high = sorted((left["slot"], right["slot"]))
    require(low != high, "write-set: identical cut points (use pair)")
    # Without these, a diff across the two per-eye pools (same size, different
    # content) or across differing inputs would report a confident, enormous
    # and meaningless write-set.
    require(left["pool_va"] == right["pool_va"], "write-set: different pools")
    require(input_digests(left["digests"]) == input_digests(right["digests"]),
            "write-set: inputs differ")
    report: dict[str, Any] = {"slots": [low, high], "pool_va": left["pool_va"],
                              "frames": []}
    for frame in range(FRAMES):
        a = read_capture_file(first, f"pool-frame{frame:02d}.bin", "write-set: pool missing")
        b = read_capture_file(second, f"pool-frame{frame:02d}.bin", "write-set: pool missing")
        require(len(a) == len(b), "write-set: pool size mismatch")
        require(len(a) == POOL_SIZE, "write-set: pool size")
        ranges: list[tuple[int, int]] = []
        changed = 0
        start = -1
        # Exact, but skips identical chunks at C speed: the byte scan only runs
        # inside chunks that actually differ, so a sparse write-set stays cheap.
        chunk = 8192
        for base in range(0, len(a), chunk):
            chunk_a = a[base:base + chunk]
            chunk_b = b[base:base + chunk]
            if chunk_a == chunk_b:
                if start >= 0:
                    ranges.append((start, base))
                    start = -1
                continue
            for offset, (x, y) in enumerate(zip(chunk_a, chunk_b)):
                if x != y:
                    changed += 1
                    if start < 0:
                        start = base + offset
                elif start >= 0:
                    ranges.append((start, base + offset))
                    start = -1
        if start >= 0:
            ranges.append((start, len(a)))
        report["frames"].append({
            "frame": frame,
            "changed": changed,
            "ranges": len(ranges),
            "first": ranges[0] if ranges else None,
            "last": ranges[-1] if ranges else None,
            "largest": max((e - s for s, e in ranges), default=0),
        })
    return report


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


CONTROLS = [
    "manifest-protocol: header format",
    "manifest-protocol: pool size",
    "slot-pinned: slot.txt vs --slot",
    "sidecar-records: record count",
    "sidecar-records: pmem length",
    "pool-records: pool bytes",
    "pool-records: pool address varies",
    "pool-file: pool file missing",
    "pool-digest: pool digest",
    "output-record: output digest",
    "output-state: written before the writer slot",
    "output-state: untouched at the writer slot",
    "input-records: input digest",
    "cut-accounting: withheld launches",
    "cut-accounting: anomaly records",
    "cut-boundary-slot: slot mismatch",
]

# Controls for check_pair: each must be reachable only through the pair
# comparator's own checks, not through a per-capture check firing first.
PAIR_CONTROLS = [
    "runs-identical: different pools",
    "runs-identical: inputs differ",
    "runs-identical: pool bytes differ",
]


def write_fixture(root: Path, slot: int = 99, written: bool = False) -> None:
    """Minimal capture shaped exactly like a real run. `written=True` replaces
    the NaN sentinel outputs with a non-sentinel payload, as a capture at or
    past LADDER_OUTPUT_WRITER_SLOT must show."""
    root.mkdir(parents=True, exist_ok=True)
    pool = bytes(POOL_SIZE)
    nan = nan_sentinel_bytes()
    # A written output is any non-sentinel payload of the output texture's
    # size; zeros are enough (720*720*4 FP16 = 4147200 bytes).
    output = b"\x00\x00" * (4147200 // 2) if written else nan
    lines = [
        f"protocol ladder-capture-v1 frames={FRAMES} size={PROBE_SIZE} "
        f"context=eye0 pool={POOL_SIZE}",
        "sidecar=nvapi-ladder-x.txt",
    ]
    for frame in range(FRAMES):
        lines.append(f"boundary-frame={frame} eval={frame} psize=80 pmem={'00' * 80}")
    for frame in range(FRAMES):
        (root / f"pool-frame{frame:02d}.bin").write_bytes(pool)
        lines.append(
            f"pool-frame={frame} bytes={POOL_SIZE} file=pool-frame{frame:02d}.bin "
            f"fnv=0x{fnv1a64(pool):016x} va=0x1000 res_va=0x1000 res_size={POOL_SIZE}")
    for name in ("input-colour.bin", "input-depth.bin", "input-motion.bin"):
        (root / name).write_bytes(nan)
        lines.append(f"{name} bytes={len(nan)} fnv=0x{fnv1a64(nan):016x}")
    for frame in range(FRAMES):
        (root / f"output-frame{frame:02d}.bin").write_bytes(output)
        lines.append(
            f"output-frame{frame:02d}.bin bytes={len(output)} "
            f"fnv=0x{fnv1a64(output):016x}")
    (root / "manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "slot.txt").write_text(f"slot={slot}\n", encoding="utf-8")
    side = [f"boundary eval={i} psize=80 pmem={'00' * 80}" for i in range(FRAMES)]
    (root / "boundary-va.txt").write_text("\n".join(side) + "\n", encoding="utf-8")
    trace = ["000001 709714463 352 cut-init armed=1 evals=0 drops=0"]
    for frame in range(FRAMES):
        trace.append(
            f"000002 709714463 352 cut-boundary eval={frame} slot={slot} psize=80")
        trace.append(f"000003 709714463 352 cut-armed eval={frame} handle=0x1")
        trace += ["000004 709714463 352 cut-drop handle=0x2 name=x"] * (
            INVENTORY - 1 - slot)
        if frame < FRAMES - 1:
            trace.append(f"000005 709714463 352 cut-release eval={frame + 1} window=0")
    (root / "trace.log").write_text("\n".join(trace) + "\n", encoding="utf-8")


# Controls that corrupt manifest.txt rather than a side file or the trace.
MANIFEST_CONTROLS = {
    "manifest-protocol: header format",
    "manifest-protocol: pool size",
    "pool-records: pool bytes",
    "pool-records: pool address varies",
    "sidecar-records: pmem length",
    "output-state: written before the writer slot",
    "output-state: untouched at the writer slot",
}

# Controls whose base fixture sits at the output writer slot, where the
# expected output state is the opposite of the default fixture's.
WRITER_BASE_CONTROLS = {
    "output-state: untouched at the writer slot",
}


def repin_output(root: Path, text: str, frame: int, payload: bytes) -> str:
    """Point the manifest's output record at `payload` so the digest check
    holds and only the semantic output-state check can fire. Uses replace_bytes
    because the fixture shares inodes with the good capture."""
    replace_bytes(root / f"output-frame{frame:02d}.bin", payload)
    return re.sub(
        rf"^(output-frame{frame:02d}\.bin bytes=\d+ fnv=)0x[0-9a-f]{{16}}$",
        rf"\g<1>0x{fnv1a64(payload):016x}", text, count=1, flags=re.MULTILINE)


def mutate(root: Path, control: str) -> None:
    """Corrupt the fixture so exactly the named check must fire."""
    manifest_path = root / "manifest.txt"
    text = manifest_path.read_text(encoding="utf-8")
    if control == "manifest-protocol: header format":
        text = text.replace("protocol ladder-capture-v1", "protocol ladder-capture-v0", 1)
    elif control == "manifest-protocol: pool size":
        text = text.replace(f"pool={POOL_SIZE}", f"pool={POOL_SIZE + 8}", 1)
    elif control == "pool-records: pool bytes":
        text = re.sub(r"^pool-frame=0 bytes=\d+", "pool-frame=0 bytes=1",
                      text, count=1, flags=re.MULTILINE)
    elif control == "slot-pinned: slot.txt vs --slot":
        replace_bytes(root / "slot.txt", b"slot=98\n")
    elif control == "sidecar-records: record count":
        replace_bytes(root / "boundary-va.txt", b"boundary eval=0 psize=80 pmem=00\n")
    elif control == "pool-file: pool file missing":
        (root / "pool-frame01.bin").unlink()
    elif control == "pool-digest: pool digest":
        data = bytearray((root / "pool-frame02.bin").read_bytes())
        data[17] ^= 0x01
        replace_bytes(root / "pool-frame02.bin", bytes(data))
    elif control == "output-record: output digest":
        # Corrupt the output payload but keep the manifest digest stale, so
        # only the digest check can fire.
        data = bytearray((root / "output-frame00.bin").read_bytes())
        data[0] ^= 0x01
        replace_bytes(root / "output-frame00.bin", bytes(data))
    elif control == "output-state: written before the writer slot":
        # Base fixture is below the writer slot: outputs must stay the NaN
        # sentinel. Repin the manifest so the digest check holds and only the
        # semantic state check can fire.
        payload = b"\x00\x00" * (4147200 // 2)
        text = repin_output(root, text, 0, payload)
    elif control == "output-state: untouched at the writer slot":
        # Base fixture sits at the writer slot with written outputs: replace
        # one with the NaN sentinel and repin, leaving only the state check.
        payload = nan_sentinel_bytes()
        text = repin_output(root, text, 0, payload)
    elif control == "input-records: input digest":
        data = bytearray((root / "input-depth.bin").read_bytes())
        data[5] ^= 0x01
        replace_bytes(root / "input-depth.bin", bytes(data))
    elif control == "sidecar-records: pmem length":
        # Reduce psize in both the manifest and the sidecar so their equality
        # holds and the length check is the one that fires.
        text = text.replace("psize=80 pmem=", "psize=40 pmem=")
        sidecar = (root / "boundary-va.txt").read_text(encoding="utf-8")
        replace_bytes(root / "boundary-va.txt",
                      sidecar.replace("psize=80 pmem=", "psize=40 pmem=").encode())
    elif control == "pool-records: pool address varies":
        # Keep va == res_va (so "copy from resource base" holds) but move frame
        # 1 to a different pool address, as a per-eye flip would.
        text = re.sub(r"^(pool-frame=1 \S+ \S+ \S+ va=)0x[0-9a-f]+( res_va=)0x[0-9a-f]+",
                      r"\g<1>0x2000\g<2>0x2000", text, count=1, flags=re.MULTILINE)
    elif control == "cut-accounting: withheld launches":
        trace = (root / "trace.log").read_text(encoding="utf-8")
        replace_bytes(root / "trace.log",
                      trace.replace("cut-drop ", "cut-keep ", 1).encode())
    elif control == "cut-accounting: anomaly records":
        trace = (root / "trace.log").read_text(encoding="utf-8")
        replace_bytes(root / "trace.log",
                      trace.replace("cut-drop ", "cut-anomaly ", 1).encode())
    elif control == "cut-boundary-slot: slot mismatch":
        trace = (root / "trace.log").read_text(encoding="utf-8")
        replace_bytes(root / "trace.log",
                      trace.replace("cut-boundary eval=0 slot=99",
                                    "cut-boundary eval=0 slot=98", 1).encode())
    else:
        raise AssertionError(f"unknown control {control}")
    if control in MANIFEST_CONTROLS:
        replace_bytes(manifest_path, text.encode())


def mutate_pair(root: Path, control: str) -> None:
    """Corrupt a copy so exactly the named check_pair check must fire.

    Each mutation keeps the copy internally self-consistent - its own manifest
    records its own bytes - so check_capture passes and it is the pair
    comparator that rejects it.
    """
    manifest_path = root / "manifest.txt"
    text = manifest_path.read_text(encoding="utf-8")
    if control == "runs-identical: different pools":
        # A per-eye pool flip: all frames consistent within the capture (so
        # "pool address varies" does not fire), but a different pool overall.
        text = re.sub(r"(pool-frame=\d+ \S+ \S+ \S+ va=)0x[0-9a-f]+( res_va=)0x[0-9a-f]+",
                      r"\g<1>0x2000\g<2>0x2000", text)
        replace_bytes(manifest_path, text.encode())
        return
    if control == "runs-identical: inputs differ":
        path = root / "input-depth.bin"
        pattern = r"^(input-depth\.bin bytes=\d+ fnv=)0x[0-9a-f]{16}"
    elif control == "runs-identical: pool bytes differ":
        path = root / "pool-frame01.bin"
        pattern = r"^(pool-frame=1 bytes=\d+ file=pool-frame01\.bin fnv=)0x[0-9a-f]{16}"
    else:
        raise AssertionError(f"unknown pair control {control}")
    data = bytearray(path.read_bytes())
    data[0] ^= 0x01
    payload = bytes(data)
    replace_bytes(path, payload)
    text = re.sub(pattern, rf"\g<1>0x{fnv1a64(payload):016x}", text, count=1,
                  flags=re.MULTILINE)
    replace_bytes(manifest_path, text.encode())


def run_self_test() -> None:
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        good = base / "good"
        write_fixture(good)
        check_capture(good, good / "trace.log", 99)
        check_pair(good, good, None, None)
        # Base fixture at the output writer slot: written outputs are the
        # expected state there, so its unmutated form must also pass.
        writer = base / "writer"
        write_fixture(writer, slot=LADDER_OUTPUT_WRITER_SLOT, written=True)
        check_capture(writer, writer / "trace.log", LADDER_OUTPUT_WRITER_SLOT)
        for control in CONTROLS:
            source = writer if control in WRITER_BASE_CONTROLS else good
            slot = LADDER_OUTPUT_WRITER_SLOT if control in WRITER_BASE_CONTROLS \
                else 99
            target = base / control.replace(" ", "_").replace(":", "-")
            shutil.copytree(source, target, copy_function=os.link)
            mutate(target, control)
            try:
                check_capture(target, target / "trace.log", slot)
            except Reject as error:
                require(str(error) == control,
                        f"self-test: control {control} tripped {error}")
            else:
                raise Reject(f"self-test: control {control} was not rejected")
        # The pair comparator must reject each of its own conditions, and must
        # reject it as itself - not as an earlier per-capture check.
        for control in PAIR_CONTROLS:
            other = base / control.replace(" ", "_").replace(":", "-")
            shutil.copytree(good, other, copy_function=os.link)
            mutate_pair(other, control)
            try:
                check_pair(good, other, None, None)
            except Reject as error:
                require(str(error) == control,
                        f"self-test: pair control {control} tripped {error}")
            else:
                raise Reject(f"self-test: pair control {control} was not rejected")
    print(f"PASS: {NAME} self-test controls={len(CONTROLS) + len(PAIR_CONTROLS)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(prog=NAME)
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("path", type=Path)
    validate.add_argument("--trace", type=Path)
    validate.add_argument("--slot", type=int)

    pair = sub.add_parser("pair")
    pair.add_argument("first", type=Path)
    pair.add_argument("second", type=Path)
    pair.add_argument("--trace-a", type=Path)
    pair.add_argument("--trace-b", type=Path)

    diff = sub.add_parser("diff")
    diff.add_argument("first", type=Path)
    diff.add_argument("second", type=Path)

    sub.add_parser("self-test")

    args = parser.parse_args()
    try:
        if args.command == "validate":
            result = check_capture(args.path, args.trace, args.slot)
        elif args.command == "pair":
            result = check_pair(args.first, args.second, args.trace_a, args.trace_b)
        elif args.command == "diff":
            result = check_diff(args.first, args.second)
        else:
            run_self_test()
            return 0
    except Reject as error:
        print(f"REJECT: {error}", file=sys.stderr)
        return 1
    import json
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
