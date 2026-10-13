#!/usr/bin/env python3
"""Independent oracle for native-vs-vendor decoder comparison evidence.

Recomputes every verdict derivable from the retained windows of
`native-vendor-compare-probe.py` runs — out0, partial-sums and counter
comparisons, change extents, write scopes and digests — from raw bytes with
independent inline arithmetic (no shared helper code with the probe), then
pins the recorded results, formulas, entry plan, integrity flags and mode
semantics. `pair` proves two runs bit-identical.

Scope: recomputation is bounded by the retained windows (out0 [0, 344064),
partials [0, 2 MiB), counters [0, 512)). Claims about bytes beyond them (the
`extent_matches_rule` confinement half) are enforced fail-closed by the
probe's own guarded run rather than re-derived here; the flags carrying those
claims are pinned for internal consistency. Guard zones, allocation
immutability beyond the retained windows and out3/out2 contents are likewise
probe-enforced and only consistency-pinned. Evidence carries digests, counts
and bounded windows only. Reuse/publication rights are unresolved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import struct
import tempfile
from typing import Any


FNV_OFFSET = 0xcbf29ce484222325
FNV_PRIME = 0x100000001b3
OUTPUT_BYTES = 24 * 28 * 512
COUNTER_SLOTS = 2 * (12 // 4) * (16 // 4)
COUNTER_SCOPE = 512
WINDOW = 2 * 1024 * 1024
DIMENSION, WIDTH, AXIS0, AXIS1 = 12, 16, 24, 28
KERNELS = {"tilesync": "cc_dec_input_upsample_1024_512_tilesync_fp8",
           "fp8": "cc_dec_input_upsample_1024_512_fp8"}
PARTIALS = {"pcap": "capture", "pzero": "zero"}
EXECUTE_PLAN = tuple(
    f"{tag}-f{frame}-{pm}" for frame in range(3) for pm in PARTIALS for tag in KERNELS)
PLANS = {"execute": EXECUTE_PLAN, "skip": ("tilesync-f0-pcap",),
         "wrong": ("tilesync-f0-pcap",)}
WORD_RE = re.compile(
    r"^word=(\d+) name=([\w-]+) frame=(\d+) eval=(\d+) va=(0x[0-9a-f]+) "
    r"res_va=(0x[0-9a-f]+) res_size=(\d+) off=(\d+) bytes=(\d+) "
    r"file=([\w.\-]+) fnv=(0x[0-9a-f]+)$")
BOUNDARY_RE = re.compile(r"^boundary-frame=(\d+) eval=(\d+) psize=(\d+) pmem=([0-9a-f]+)$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def fnv1a64(data: bytes) -> int:
    value = FNV_OFFSET
    for index in range(len(data)):
        value ^= data[index]
        value = (value * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return value


def diff_stats(left: bytes, right: bytes, block: int = 65536) -> tuple[bool, int, int]:
    """Independent arithmetic: big-integer XOR with byte-level zero counts.

    Deliberately a different algorithm from the probe's blockwise comparison
    (which counts per byte inside differing chunks), so agreement between the
    two is cross-method, not shared-code.
    """
    require(type(left) is bytes and type(right) is bytes, "recompute: diff type")
    require(len(left) == len(right) and len(left) > 0 and block > 0, "recompute: diff size")
    if left == right:
        return (True, 0, -1)
    raw = (int.from_bytes(left, "big") ^ int.from_bytes(right, "big")).to_bytes(
        len(left), "big")
    first = len(raw) - len(raw.lstrip(b"\x00"))
    last = len(raw.rstrip(b"\x00")) - 1
    require(0 <= first <= last < len(raw), "recompute: diff internal")
    return (False, len(raw) - raw.count(0), first)


def change_extent(before: bytes, after: bytes, block: int = 65536) -> tuple[int, int, int]:
    """Independent arithmetic over XOR-derived change positions."""
    require(type(before) is bytes and type(after) is bytes, "recompute: extent type")
    require(len(before) == len(after) and len(before) > 0 and block > 0,
            "recompute: extent size")
    if before == after:
        return (-1, -1, 0)
    raw = (int.from_bytes(before, "big") ^ int.from_bytes(after, "big")).to_bytes(
        len(before), "big")
    first = len(raw) - len(raw.lstrip(b"\x00"))
    last = len(raw.rstrip(b"\x00")) - 1
    changed_blocks = sum(1 for offset in range(0, len(raw), block)
                         if raw[offset:offset + block].strip(b"\x00"))
    return (first, last, changed_blocks)


_CACHE: dict[str, tuple[int, int, int, bytes, int]] = {}


def cached_capture_window(path: Path) -> tuple[bytes, int]:
    """Capture windows are re-read per entry and per digest pin; cache them.

    Validated by (inode, mtime_ns, size) so mutated copies are never stale.
    Run evidence is deliberately NOT cached: controls mutate it, and each
    verify reads it exactly once.
    """
    key = str(path)
    info = path.stat()
    hit = _CACHE.get(key)
    if hit is not None and hit[:3] == (info.st_ino, info.st_mtime_ns, info.st_size):
        return hit[3], hit[4]
    data = path.read_bytes()
    digest = fnv1a64(data)
    _CACHE[key] = (info.st_ino, info.st_mtime_ns, info.st_size, data, digest)
    return data, digest


def parse_capture(root: Path) -> dict[str, Any]:
    text = (root / "manifest.txt").read_text(encoding="utf-8").splitlines()
    words: dict[tuple[int, int], dict[str, Any]] = {}
    boundary: dict[int, str] = {}
    for line in text:
        entry = WORD_RE.match(line)
        if entry:
            words[(int(entry.group(3)), int(entry.group(1)))] = {
                "file": entry.group(10), "bytes": int(entry.group(9)),
                "fnv": int(entry.group(11), 16)}
            continue
        frame = BOUNDARY_RE.match(line)
        if frame:
            boundary[int(frame.group(1))] = frame.group(4)
    require(len(words) == 18 and len(boundary) == 3, "capture: manifest shape")
    return {"words": words, "boundary": boundary}


def load_capture_window(capture: dict[str, Any], root: Path, frame: int,
                        word: int) -> bytes:
    entry = capture["words"].get((frame, word))
    require(entry is not None, "capture: window entry missing")
    data, digest = cached_capture_window(root / entry["file"])
    require(len(data) == entry["bytes"], "capture: window size")
    require(digest == entry["fnv"], "capture: window digest")
    return data


def retained(run: Path, name: str) -> bytes:
    path = run / "evidence" / name
    require(path.is_file(), "evidence: retained window missing")
    return path.read_bytes()


def verify_entry(entry: dict[str, Any], run: Path, capture: dict[str, Any],
                 capture_root: Path, mode: str) -> None:
    entry_id = entry["entry"]
    require(entry_id in PLANS[mode], f"plan: unknown entry {entry_id}")
    tag, frame_text, partials_tag = entry_id.split("-")
    frame = int(frame_text[1:])
    require(entry["frame"] == frame, "entry-consistency: frame")
    require(entry["kernel"] == KERNELS[tag], "entry-consistency: kernel")
    require(entry["partials_prestate"] == PARTIALS[partials_tag],
            "entry-consistency: partials prestate")
    require(entry["weights_rotated"] == (mode == "wrong"), "entry-consistency: rotation")
    require(entry["decoder_launched"] == (mode != "skip"), "entry-consistency: launched")
    require(entry["launches"] == (2 if mode != "skip" else 1), "entry-arithmetic: launches")
    require(entry["readbacks"] == 8, "entry-arithmetic: readbacks")

    out0 = retained(run, entry["retained"]["out0"])
    partials = retained(run, entry["retained"]["partials"])
    counters = retained(run, entry["retained"]["counters"])
    require(len(out0) == OUTPUT_BYTES, "evidence: out0 retained size")
    require(len(partials) == WINDOW, "evidence: partials retained size")
    require(len(counters) == COUNTER_SCOPE, "evidence: counters retained size")

    vendor_out0 = load_capture_window(capture, capture_root, frame, 2)[:OUTPUT_BYTES]
    recorded = entry["out0"]
    require(recorded["scope_bytes"] == OUTPUT_BYTES, "recompute: out0 scope")
    require(recorded["native_fnv"] == f"0x{fnv1a64(out0):016x}",
            "recompute: out0 native digest")
    require(recorded["vendor_fnv"] == f"0x{fnv1a64(vendor_out0):016x}",
            "recompute: out0 vendor digest")
    match, count, first = diff_stats(vendor_out0, out0)
    require((recorded["match"], recorded["diff_count"], recorded["first_diff"])
            == (match, count, first), "recompute: out0 comparison")
    fill = bytes([0xA5]) * OUTPUT_BYTES
    extent_o = change_extent(fill, out0)
    require((recorded["change_first"], recorded["change_last"], recorded["change_blocks"])
            == extent_o, "recompute: out0 extent")
    require(recorded["extent_matches_rule"]
            == ((extent_o[0], extent_o[1]) == (0, OUTPUT_BYTES - 1)),
            "recompute: out0 extent flag")

    capture_partials = load_capture_window(capture, capture_root, frame, 6)
    prestate = capture_partials if PARTIALS[partials_tag] == "capture" else bytes(WINDOW)
    extent = change_extent(prestate, partials)
    recorded_p = entry["partials"]
    require(type(recorded_p["beyond_window_change"]) is bool,
            "recompute: partials schema")
    require((recorded_p["change_first"], recorded_p["change_last"],
             recorded_p["change_blocks"]) == extent, "recompute: partials extent")
    match_p, count_p, first_p = diff_stats(capture_partials, partials)
    require((recorded_p["window_match"], recorded_p["window_diff_count"],
             recorded_p["window_first_diff"]) == (match_p, count_p, first_p),
            "recompute: partials window")
    extent_end = extent[1] + 1 if extent[1] >= 0 else 0
    scope = min(extent_end, WINDOW) if extent_end else WINDOW
    require(recorded_p["write_extent_end"] == extent_end, "recompute: partials extent end")
    require(recorded_p["write_scope_bytes"] == scope, "recompute: partials scope")
    match_s, count_s, first_s = diff_stats(capture_partials[:scope], partials[:scope])
    require((recorded_p["write_scope_match"], recorded_p["write_scope_diff_count"],
             recorded_p["write_scope_first_diff"]) == (match_s, count_s, first_s),
            "recompute: partials write scope")

    capture_counters = load_capture_window(capture, capture_root, frame, 4)
    recorded_c = entry["counters"]
    require(recorded_c["scope_bytes"] == COUNTER_SCOPE, "recompute: counters scope")
    head = list(struct.unpack(f"<{COUNTER_SLOTS}i", counters[:COUNTER_SLOTS * 4]))
    require(recorded_c["head"] == head, "recompute: counter head")
    match_c, count_c, first_c = diff_stats(capture_counters[:COUNTER_SCOPE], counters)
    require((recorded_c["window_match"], recorded_c["window_diff_count"],
             recorded_c["window_first_diff"]) == (match_c, count_c, first_c),
            "recompute: counters window")


def verify(run: Path, capture_root: Path) -> dict[str, Any]:
    result = json.loads((run / "output.json").read_text(encoding="utf-8"))
    require("cuda" in result, "run: no cuda result")
    cuda = result["cuda"]
    mode = cuda["mode"]
    require(mode in PLANS, "run: unknown mode")
    require(tuple(cuda["entries_plan"]) == PLANS[mode], "plan: entry plan mismatch")
    require([e["entry"] for e in cuda["entries"]] == list(PLANS[mode]),
            "plan: entry order")
    capture = parse_capture(capture_root)
    expected_frames = range(3) if mode == "execute" else (0,)
    expected_keys = {f"frame{frame}-w{word}" for frame in expected_frames
                     for word in (0, 1, 2, 4, 6, 7)}
    require(set(result["capture_digests"]) == expected_keys,
            "capture: digest key set mismatch")
    for frame in expected_frames:
        for word in (0, 1, 2, 4, 6, 7):
            window = load_capture_window(capture, capture_root, frame, word)
            recorded = result["capture_digests"][f"frame{frame}-w{word}"]
            require(recorded == f"0x{fnv1a64(window):016x}",
                    "capture: recorded digest mismatch")
        scalars = struct.unpack("<4I", bytes.fromhex(capture["boundary"][frame])[64:80])
        require(scalars == (DIMENSION, WIDTH, AXIS0, AXIS1), "capture: boundary scalars")
    require(OUTPUT_BYTES == AXIS0 * AXIS1 * 512, "formula: output extent")
    require(COUNTER_SLOTS == 2 * (DIMENSION // 4) * (WIDTH // 4), "formula: counter slots")

    for entry in cuda["entries"]:
        verify_entry(entry, run, capture, capture_root, mode)

    # Integrity flags: probe-enforced outcomes pinned for consistency with the
    # recorded no-failure verdict (raw guard zones are outside retained scope).
    for entry in cuda["entries"]:
        require(entry["guards_intact"], "flags: guards")
        require(all(entry["immutable"][key] for key in ("main", "skip", "weights")),
                "flags: immutable")
        if entry["kernel"] == KERNELS["fp8"]:
            require(all(entry["immutable"][key] for key in ("out3", "out2")),
                    "flags: immutable")
        for key in ("out3", "out2"):
            changed = entry["out3_out2_extent"][key]["change_first"] != -1
            require(changed == (not entry["immutable"][key]),
                    "flags: out3_out2 consistency")
        if entry["decoder_launched"]:
            require(entry["out0"]["extent_matches_rule"], "flags: out0 extent")
            require(entry["counters"]["head"] == [3] * COUNTER_SLOTS, "flags: counter head")
            require(entry["counters"]["window_match"], "flags: counter window")

    require(cuda["kernels_launched"] == sum(e["launches"] for e in cuda["entries"]),
            "entry-arithmetic: total launches")
    require(cuda["readbacks"] == 8 * len(cuda["entries"]), "entry-arithmetic: readbacks")
    launched = [e for e in cuda["entries"] if e["decoder_launched"]]
    if mode == "execute":
        require(all(e["out0"]["match"] for e in launched),
                "semantics: all launched entries must match")
    elif mode == "wrong":
        require(not cuda["entries"][0]["out0"]["match"], "semantics: wrong must not match")
    else:
        entry = cuda["entries"][0]
        require(not entry["out0"]["match"], "semantics: skip must not match")
        require(entry["counters"]["head"] == [-1] * COUNTER_SLOTS,
                "semantics: skip must not publish")
    require(cuda["enforced_failures"] == [], "run: enforced failures recorded")
    return dict(verified=True, mode=mode, entries=len(cuda["entries"]),
                launched_entries=len(launched),
                out0_matches=sum(1 for e in cuda["entries"] if e["out0"]["match"]),
                partials_scope_matches=sum(1 for e in cuda["entries"]
                                           if e["partials"]["write_scope_match"]))


def pair(run_a: Path, run_b: Path) -> dict[str, Any]:
    result_a = json.loads((run_a / "output.json").read_text(encoding="utf-8"))
    result_b = json.loads((run_b / "output.json").read_text(encoding="utf-8"))
    require(result_a["cuda"]["entries"] == result_b["cuda"]["entries"],
            "pair: entry results differ")
    require(result_a["cuda"]["entries_plan"] == result_b["cuda"]["entries_plan"],
            "pair: entry plans differ")
    files = sorted((run_a / "evidence").glob("*.bin"))
    require(len(files) > 0, "pair: no evidence files")
    for path in files:
        other = run_b / "evidence" / path.name
        require(other.is_file(), "pair: evidence file missing")
        require(path.read_bytes() == other.read_bytes(),
                f"pair: evidence differs {path.name}")
    return dict(pair_identical=True, entries=len(result_a["cuda"]["entries"]),
                evidence_files=len(files))


# ---------------------------------------------------------------------------
# Self-test: corruption controls, each rejected for the reason in its name.
# ---------------------------------------------------------------------------

def self_test() -> dict[str, Any]:
    require(diff_stats(b"\x01" * 10, b"\x01" * 10) == (True, 0, -1), "known: diff equal")
    require(diff_stats(b"\x01" * 10, b"\x02\x01\x01\x01\x01\x01\x01\x01\x01\x02")
            == (False, 2, 0), "known: diff")
    require(change_extent(b"\x00" * 10, b"\x00" * 10) == (-1, -1, 0), "known: extent")
    require(change_extent(bytes(10), b"\x01" + bytes(8) + b"\x01") == (0, 9, 1),
            "known: extent changed")
    malformed = 0
    for call in (lambda: diff_stats(b"\x00", b"\x00\x00"),
                 lambda: diff_stats("x", "x"),
                 lambda: change_extent(b"\x00", b"\x00\x00"),
                 lambda: change_extent(b"", b"")):
        try:
            call()
        except ValueError:
            malformed += 1
        else:
            raise ValueError("malformed recompute input accepted")

    here = Path(__file__).resolve()
    source = here.parents[1] / "research" / "native-linux" / "runs"
    runs = [path.parent for path in sorted(source.glob("vendor-compare-execute-*/output.json"))
            if "beyond_window_change" in path.read_text(encoding="utf-8")]
    if not runs:
        return dict(known_answers=4, malformed_controls=0, controls={},
                    note="no run evidence present")
    template = runs[-1]
    capture_root = here.parent / "logs" / "boundary" / "a"
    controls: dict[str, bool] = {}

    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        capture_copy = root / "capture"
        shutil.copytree(capture_root, capture_copy)

        def fresh(name: str) -> Path:
            target = root / name
            shutil.copytree(template, target)
            return target

        def expect_reject(name: str, target: Path, message: str,
                          capture: Path = capture_root) -> None:
            try:
                verify(target, capture)
            except ValueError as error:
                require(str(error) == message,
                        f"control {name}: wrong rejection: {error}")
                controls[name] = True
            else:
                raise ValueError(f"control {name}: corruption accepted")

        def rewrite(name: str, target: Path, mutate: Any) -> None:
            data = json.loads((target / "output.json").read_text(encoding="utf-8"))
            mutate(data)
            (target / "output.json").write_text(json.dumps(data), encoding="utf-8")

        def flip_file(name: str, target: Path, entry_index: int, key: str,
                      offset: int, size: int | None = None) -> None:
            data = json.loads((target / "output.json").read_text(encoding="utf-8"))
            file_name = data["cuda"]["entries"][entry_index]["retained"][key]
            path = target / "evidence" / file_name
            blob = path.read_bytes()
            if size is not None:
                path.write_bytes(blob[:size])
                return
            changed = bytearray(blob)
            changed[offset] ^= 0x01
            path.write_bytes(bytes(changed))

        require(verify(fresh("clean"), capture_root)["verified"], "baseline run rejected")

        flip_file("out0-content", fresh("out0-content"), 0, "out0", 5)
        expect_reject("out0-content", root / "out0-content",
                      "recompute: out0 native digest")
        flip_file("out0-size", fresh("out0-size"), 0, "out0", 0, size=10)
        expect_reject("out0-size", root / "out0-size", "evidence: out0 retained size")
        for name, field, value, message in (
                ("out0-digest", "native_fnv", "0x0000000000000000",
                 "recompute: out0 native digest"),
                ("out0-diff_count", "diff_count", 1, "recompute: out0 comparison"),
                ("out0-extent", "change_last", 1, "recompute: out0 extent"),
                ("out0-extent-flag", "extent_matches_rule", False,
                 "recompute: out0 extent flag")):
            target = fresh(name)
            rewrite(name, target, lambda d, f=field, v=value:
                    d["cuda"]["entries"][0]["out0"].__setitem__(f, v))
            expect_reject(name, target, message)
        target = fresh("out0-match")
        rewrite("out0-match", target,
                lambda d: d["cuda"]["entries"][0]["out0"].__setitem__(
                    "match", not d["cuda"]["entries"][0]["out0"]["match"]))
        expect_reject("out0-match", target, "recompute: out0 comparison")

        # Partial-sums suite: pcap entry 0 (no writes) and pzero entry 2 (writes).
        flip_file("partials-extent", fresh("partials-extent"), 0, "partials", 100)
        expect_reject("partials-extent", root / "partials-extent",
                      "recompute: partials extent")
        flip_file("partials-window", fresh("partials-window"), 2, "partials", 100)
        expect_reject("partials-window", root / "partials-window",
                      "recompute: partials window")
        flip_file("partials-size", fresh("partials-size"), 0, "partials", 0, size=10)
        expect_reject("partials-size", root / "partials-size",
                      "evidence: partials retained size")
        for name, field, value, message in (
                ("partials-schema", "beyond_window_change", "yes",
                 "recompute: partials schema"),
                ("partials-extent-field", "change_last", 1, "recompute: partials extent"),
                ("partials-extent-end", "write_extent_end", 7, "recompute: partials extent end"),
                ("partials-scope", "write_scope_bytes", 7, "recompute: partials scope"),
                ("partials-scope-diff", "write_scope_diff_count", 1,
                 "recompute: partials write scope")):
            target = fresh(name)
            rewrite(name, target, lambda d, f=field, v=value:
                    d["cuda"]["entries"][2]["partials"].__setitem__(f, v))
            expect_reject(name, target, message)

        # Counter suite: head bytes, tail bytes, size, recorded head.
        flip_file("counters-head", fresh("counters-head"), 0, "counters", 0)
        expect_reject("counters-head", root / "counters-head", "recompute: counter head")
        flip_file("counters-window", fresh("counters-window"), 0, "counters", 100)
        expect_reject("counters-window", root / "counters-window",
                      "recompute: counters window")
        flip_file("counters-size", fresh("counters-size"), 0, "counters", 0, size=10)
        expect_reject("counters-size", root / "counters-size",
                      "evidence: counters retained size")
        target = fresh("counters-head-field")
        rewrite("counters-head-field", target,
                lambda d: d["cuda"]["entries"][0]["counters"]["head"].__setitem__(0, 2))
        expect_reject("counters-head-field", target, "recompute: counter head")

        # Plan, entry and arithmetic pins.
        target = fresh("plan")
        rewrite("plan", target, lambda d: d["cuda"]["entries_plan"].__setitem__(0, "fp8-f0-pcap"))
        expect_reject("plan", target, "plan: entry plan mismatch")
        target = fresh("entries-reorder")
        def reorder(d: dict[str, Any]) -> None:
            entries = d["cuda"]["entries"]
            entries[0], entries[1] = entries[1], entries[0]
        rewrite("entries-reorder", target, reorder)
        expect_reject("entries-reorder", target, "plan: entry order")
        for name, path, value, message in (
                ("kernel", ("entries", 0, "kernel"), "cc_cb_clear",
                 "entry-consistency: kernel"),
                ("frame", ("entries", 0, "frame"), 2, "entry-consistency: frame"),
                ("partials-prestate", ("entries", 0, "partials_prestate"), "zero",
                 "entry-consistency: partials prestate"),
                ("rotation", ("entries", 0, "weights_rotated"), True,
                 "entry-consistency: rotation"),
                ("launches", ("entries", 0, "launches"), 3,
                 "entry-arithmetic: launches"),
                ("readbacks", ("entries", 0, "readbacks"), 9,
                 "entry-arithmetic: readbacks"),
                ("guards", ("entries", 0, "guards_intact"), False, "flags: guards"),
                ("total-launches", ("kernels_launched",), 1,
                 "entry-arithmetic: total launches")):
            target = fresh(name)
            def mutate(d: dict[str, Any], p=path, v=value) -> None:
                node = d["cuda"]
                for part in p[:-1]:
                    node = node[part]
                node[p[-1]] = v
            rewrite(name, target, mutate)
            expect_reject(name, target, message)
        target = fresh("immutable")
        rewrite("immutable", target,
                lambda d: d["cuda"]["entries"][0]["immutable"].__setitem__("main", False))
        expect_reject("immutable", target, "flags: immutable")
        target = fresh("out3out2-consistency")
        def inconsistent(d: dict[str, Any]) -> None:
            d["cuda"]["entries"][0]["out3_out2_extent"]["out2"] = {
                "change_first": -1, "change_last": -1, "change_blocks": 0}
        rewrite("out3out2-consistency", target, inconsistent)
        expect_reject("out3out2-consistency", target, "flags: out3_out2 consistency")
        target = fresh("enforced-failures")
        rewrite("enforced-failures", target,
                lambda d: d["cuda"].__setitem__("enforced_failures", ["x"]))
        expect_reject("enforced-failures", target, "run: enforced failures recorded")

        # Capture pins (digests, key set, per-frame scalars).
        target = fresh("capture-digest")
        rewrite("capture-digest", target,
                lambda d: d["capture_digests"].__setitem__("frame0-w2", "0x0"))
        expect_reject("capture-digest", target, "capture: recorded digest mismatch")
        target = fresh("capture-keys")
        rewrite("capture-keys", target,
                lambda d: d["capture_digests"].pop("frame2-w2"))
        expect_reject("capture-keys", target, "capture: digest key set mismatch")
        manifest = capture_copy / "manifest.txt"
        good = struct.pack("<2I", DIMENSION, WIDTH).hex()
        lines = manifest.read_text(encoding="utf-8").splitlines(keepends=True)
        for index, line in enumerate(lines):
            if line.startswith("boundary-frame=2"):
                lines[index] = line.replace(
                    good, struct.pack("<2I", DIMENSION, 15).hex())
        manifest.write_text("".join(lines), encoding="utf-8")
        expect_reject("capture-scalars", root / "clean", "capture: boundary scalars",
                      capture=capture_copy)

    return dict(known_answers=4, malformed_controls=malformed, controls=controls,
                controls_passed=len(controls),
                baseline_accepted=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("verify")
    check.add_argument("run", type=Path)
    check.add_argument("capture", type=Path)
    combine = sub.add_parser("pair")
    combine.add_argument("run_a", type=Path)
    combine.add_argument("run_b", type=Path)
    sub.add_parser("self-test")
    args = parser.parse_args()
    if args.command == "verify":
        print(json.dumps(verify(args.run, args.capture), indent=2))
    elif args.command == "pair":
        print(json.dumps(pair(args.run_a, args.run_b), indent=2))
    else:
        print(json.dumps(self_test(), indent=2))


if __name__ == "__main__":
    main()
