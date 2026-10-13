#!/usr/bin/env python3
"""Validate decoder-boundary capture evidence (vr/DECODER-BOUNDARY.md).

The capture is produced by `run-feature-probe.sh boundary <label>`: the
nvapi64 forwarder truncates every evaluation after the model's decoder launch
(the `cut`), and the harness reads the boundary buffers back through its own
device. This tool checks the retained evidence without touching the GPU.

Modes:
  report <dir>            validate one capture directory
  pair <dir-a> <dir-b>    compare two captures (determinism)
  self-test               corruption controls over generated fixtures

Pinned checks (each with its own rejection message):
  manifest-protocol      protocol/geometry header is exactly as recorded
  sidecar-records        retained sidecar reproduces the parameter blocks
  manifest-consistent    word records agree with their files and geometry
  words-complete         every frame carries exactly the six boundary words
  boundary-constants     dimension/width/output-axis constants as recovered
  counter-published      decoder counter region reads the final value 3
  output-untouched       output texture is still the NaN sentinel (the cut ran)
  cut-drops              trace shows 56 withheld launches per evaluation
  weights-record-match   live weights equal the stored block39.layer0.layer
  runs-identical         two captures are bit-identical (pair mode)

Only digests, counts and geometry are reported: never tensor or weight bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import struct
import sys
from pathlib import Path
from typing import Any

NAME = "decoder-boundary-report"
FRAMES = 3
WINDOW = 2097152
PSIZE = 80
WORDS = {0: "main-input", 1: "skip-input", 2: "out0", 4: "counter",
         6: "partial-sums", 7: "weights"}
DIMENSION = 12
WIDTH = 16
OVERRIDES = (24, 28)
COUNTER_VALUE = 3
COUNTER_SLOTS = 8
CUT_DROPS_PER_EVAL = 56
RECORD_NAME = "block39.layer0.layer"
RECORD_BYTES = 525312
PROBE_SIZE = 720
NAN_HALF = 0x7E00

FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211


class Reject(Exception):
    """A pinned check failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Reject(message)


def fnv1a64(data: bytes) -> int:
    hash_value = FNV_OFFSET
    for byte in data:
        hash_value ^= byte
        hash_value = (hash_value * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return hash_value


def nan_sentinel_bytes() -> bytes:
    return struct.pack("<4H", NAN_HALF, NAN_HALF, NAN_HALF, NAN_HALF) * (PROBE_SIZE * PROBE_SIZE)


# ---------------------------------------------------------------------------
# Manifest parsing
# ---------------------------------------------------------------------------

HEADER_RE = re.compile(
    r"^protocol boundary-capture-v1 frames=(\d+) size=(\d+) context=(\w+) "
    r"window=(\d+) words=(\d+)$")
BOUNDARY_RE = re.compile(
    r"^boundary-frame=(\d+) eval=(\d+) psize=(\d+) pmem=([0-9a-f]+)$")
BOUNDARY_SIDE_RE = re.compile(
    r"^boundary eval=(\d+) psize=(\d+) pmem=([0-9a-f]+)$")
WORD_RE = re.compile(
    r"^word=(\d+) name=([\w-]+) frame=(\d+) eval=(\d+) va=(0x[0-9a-f]+) "
    r"res_va=(0x[0-9a-f]+) res_size=(\d+) off=(\d+) bytes=(\d+) "
    r"file=([\w.\-]+) fnv=(0x[0-9a-f]+)$")
FILE_RE = re.compile(r"^([\w.\-]+) bytes=(\d+) fnv=(0x[0-9a-f]+)$")


def read_capture_file(root: Path, name: str, message: str) -> bytes:
    path = root / name
    require(path.is_file(), message)
    return path.read_bytes()


def parse_manifest(path: Path) -> dict[str, Any]:
    require(path.is_file(), "manifest-protocol: manifest missing")
    text = path.read_text(encoding="utf-8").splitlines()
    require(len(text) >= 2, "manifest-protocol: manifest too short")
    header = HEADER_RE.match(text[0])
    require(header is not None, "manifest-protocol: unparsable header")
    require(text[1].startswith("sidecar="), "manifest-protocol: missing sidecar line")
    state: dict[str, Any] = {
        "frames": int(header.group(1)), "size": int(header.group(2)),
        "context": header.group(3), "window": int(header.group(4)),
        "word_slots": int(header.group(5)), "boundary": [], "words": [],
        "files": {}}
    require(state["frames"] == FRAMES, "manifest-protocol: frame count")
    require(state["size"] == PROBE_SIZE, "manifest-protocol: texture size")
    require(state["context"] == "eye0", "manifest-protocol: context")
    require(state["window"] == WINDOW, "manifest-protocol: window")
    require(state["word_slots"] == len(WORDS), "manifest-protocol: word slots")
    for line in text[2:]:
        if not line:
            continue
        boundary = BOUNDARY_RE.match(line)
        if boundary:
            state["boundary"].append({
                "frame": int(boundary.group(1)), "eval": int(boundary.group(2)),
                "psize": int(boundary.group(3)), "hex": boundary.group(4)})
            continue
        word = WORD_RE.match(line)
        if word:
            state["words"].append({
                "word": int(word.group(1)), "name": word.group(2),
                "frame": int(word.group(3)), "eval": int(word.group(4)),
                "va": int(word.group(5), 16), "res_va": int(word.group(6), 16),
                "res_size": int(word.group(7)), "off": int(word.group(8)),
                "bytes": int(word.group(9)), "file": word.group(10),
                "fnv": int(word.group(11), 16)})
            continue
        file_line = FILE_RE.match(line)
        require(file_line is not None, "manifest-consistent: unparsable line")
        state["files"][file_line.group(1)] = {
            "bytes": int(file_line.group(2)), "fnv": int(file_line.group(3), 16)}
    return state


def boundary_words(hexdata: str) -> tuple[int, ...]:
    require(len(hexdata) == PSIZE * 2, "words-complete: pmem length")
    return struct.unpack("<10Q", bytes.fromhex(hexdata))


# ---------------------------------------------------------------------------
# Single-capture checks
# ---------------------------------------------------------------------------

def check_capture(root: Path, trace: Path | None) -> dict[str, Any]:
    require(root.is_dir(), "manifest-consistent: capture directory missing")
    manifest = parse_manifest(root / "manifest.txt")
    result: dict[str, Any] = {"dir": str(root), "digests": {}, "observations": {}}

    # The retained sidecar is part of the validated evidence chain: its lines
    # must reproduce the manifest's boundary parameter blocks exactly.
    sidecar = root / "boundary-va.txt"
    require(sidecar.is_file(), "sidecar-records: missing sidecar")
    parsed_sidecar = []
    for line in sidecar.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        match = BOUNDARY_SIDE_RE.match(line)
        require(match is not None, "sidecar-records: sidecar mismatch")
        parsed_sidecar.append({"eval": int(match.group(1)), "psize": int(match.group(2)),
                               "hex": match.group(3)})
    require(parsed_sidecar == [{k: b[k] for k in ("eval", "psize", "hex")}
                               for b in manifest["boundary"]],
            "sidecar-records: sidecar mismatch")

    # Every retained window file must be accounted for by the manifest.
    listed = {w["file"] for w in manifest["words"]} | set(manifest["files"])
    for path in sorted(root.glob("*.bin")):
        require(path.name in listed, "manifest-consistent: untracked file")

    require(len(manifest["boundary"]) == FRAMES, "words-complete: boundary-frame count")
    constants_seen: set[tuple[int, ...]] = set()
    nan_digest = fnv1a64(nan_sentinel_bytes())
    for frame in range(FRAMES):
        entries = [w for w in manifest["words"] if w["frame"] == frame]
        require(len(entries) == len(WORDS), "words-complete: frame word count")
        require({w["word"] for w in entries} == set(WORDS), "words-complete: frame word slots")
        require(all(w["name"] == WORDS[w["word"]] for w in entries),
                "words-complete: word naming")
        boundary = manifest["boundary"][frame]
        require(boundary["frame"] == frame and boundary["eval"] == frame,
                "words-complete: frame/eval indexing")
        require(boundary["psize"] == PSIZE, "words-complete: psize")
        words = boundary_words(boundary["hex"])
        constants_seen.add(words[8:10])
        for entry in entries:
            require(entry["eval"] == boundary["eval"], "manifest-consistent: word eval")
            require(entry["va"] == words[entry["word"]],
                    "manifest-consistent: word va vs pmem")
            require(entry["off"] == entry["va"] - entry["res_va"],
                    "manifest-consistent: window offset")
            expected = min(WINDOW, entry["res_size"] - entry["off"])
            require(entry["bytes"] == expected, "manifest-consistent: window bytes")
            data = read_capture_file(root, entry["file"], "manifest-consistent: file missing")
            require(len(data) == entry["bytes"], "manifest-consistent: file size")
            require(fnv1a64(data) == entry["fnv"], "manifest-consistent: file digest")
            result["digests"][f"frame{frame}-w{entry['word']}"] = f"0x{entry['fnv']:016x}"
        output = manifest["files"].get(f"output-frame{frame:02d}.bin")
        require(output is not None, "output-untouched: output record missing")
        data = read_capture_file(root, f"output-frame{frame:02d}.bin",
                                 "output-untouched: output missing")
        require(len(data) == output["bytes"], "output-untouched: output size")
        require(fnv1a64(data) == output["fnv"], "output-untouched: output digest")
        require(output["fnv"] == nan_digest, "output-untouched: output not the NaN sentinel")
        result["digests"][f"output-frame{frame}"] = f"0x{output['fnv']:016x}"
    for name in ("input-colour.bin", "input-depth.bin", "input-motion.bin"):
        record = manifest["files"].get(name)
        require(record is not None, "manifest-consistent: input record missing")
        data = read_capture_file(root, name, "manifest-consistent: file missing")
        require(len(data) == record["bytes"] and fnv1a64(data) == record["fnv"],
                "manifest-consistent: input digest")
        result["digests"][name] = f"0x{record['fnv']:016x}"

    # Constants: the recovered decoder host ABI (NATIVE-DECODER-IO.md slots and
    # NVAPI-BINDINGS.md live bindings) fixes width=16 and the (24,28) pair.
    require(len(constants_seen) == 1, "boundary-constants: not constant across frames")
    constant = next(iter(constants_seen))
    dimension, width = struct.unpack("<2I", struct.pack("<Q", constant[0]))
    overrides = struct.unpack("<2I", struct.pack("<Q", constant[1]))
    require(width == WIDTH, "boundary-constants: width")
    require(overrides == OVERRIDES, "boundary-constants: output-axis overrides")
    require(dimension == DIMENSION, "boundary-constants: dimension")
    result["constants"] = {"dimension": dimension, "width": width,
                           "overrides": list(overrides)}

    # Counter protocol: the decoder publishes its final value in every slot.
    counters: set[tuple[int, ...]] = set()
    for frame in range(FRAMES):
        entry = next(w for w in manifest["words"] if w["frame"] == frame and w["word"] == 4)
        data = read_capture_file(root, entry["file"], "counter-published: counter missing")
        require(len(data) >= COUNTER_SLOTS * 4, "counter-published: counter window short")
        slots = struct.unpack(f"<{COUNTER_SLOTS}i", data[:COUNTER_SLOTS * 4])
        counters.add(slots)
        require(all(value == COUNTER_VALUE for value in slots),
                "counter-published: counter values")
    require(len(counters) == 1, "counter-published: not constant across frames")
    result["counter"] = list(next(iter(counters)))

    # Frame evolution is an observation: the boundary feeds the temporal path.
    for word in (0, 1, 2):
        first = next(w for w in manifest["words"] if w["frame"] == 0 and w["word"] == word)
        a = (root / first["file"]).read_bytes()
        for frame in range(1, FRAMES):
            entry = next(w for w in manifest["words"] if w["frame"] == frame and w["word"] == word)
            b = (root / entry["file"]).read_bytes()
            changed = sum(1 for x, y in zip(a, b) if x != y)
            result["observations"][f"w{word}-frame{frame}-changed-bytes"] = changed

    require(trace is not None and trace.is_file(), "cut-drops: trace required")
    text = trace.read_text(encoding="utf-8", errors="replace")
    require(re.search(r"cut-(anomaly|window-exhausted|unknown|name-table-full|"
                      r"pmem-incomplete|clear-unknown|sidecar-write-failed) ",
                      text) is None, "cut-drops: anomaly records")
    armed = len(re.findall(r"cut-armed ", text))
    drops = len(re.findall(r"cut-drop ", text))
    releases = len(re.findall(r"cut-release ", text))
    require(armed == FRAMES, "cut-drops: armed evaluations")
    require(drops == CUT_DROPS_PER_EVAL * armed, "cut-drops: withheld launches")
    require(releases == armed - 1, "cut-drops: releases")
    result["cut"] = {"armed": armed, "drops": drops, "releases": releases}
    return result


def check_weights_record(root: Path, dll: Path) -> dict[str, Any]:
    """The live decoder weights must equal the stored candidate record."""
    helper_dir = Path(__file__).resolve().parent

    def load(name: str, filename: str) -> Any:
        spec = importlib.util.spec_from_file_location(name, helper_dir / filename)
        require(spec is not None and spec.loader is not None, "weights-record-match: helper")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    binding = load("binding", "native-decoder-binding-probe.py")
    data = dll.read_bytes()
    require(hashlib.sha256(data).hexdigest() == binding.base.PIN,
            "weights-record-match: DLL hash pin")
    payload, _meta = binding.candidate_payload(data)
    require(len(payload) == RECORD_BYTES, "weights-record-match: record size")
    manifest = parse_manifest(root / "manifest.txt")
    entry = next(w for w in manifest["words"] if w["frame"] == 0 and w["word"] == 7)
    window = (root / entry["file"]).read_bytes()
    require(len(window) >= RECORD_BYTES, "weights-record-match: window short")
    require(window[:RECORD_BYTES] == payload, "weights-record-match: live vs record bytes")
    return {"record": RECORD_NAME, "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "window_tail_fnv": f"0x{fnv1a64(window[RECORD_BYTES:]):016x}"}


def check_pair(a: Path, b: Path) -> dict[str, Any]:
    first = parse_manifest(a / "manifest.txt")
    second = parse_manifest(b / "manifest.txt")
    require(first["boundary"] == second["boundary"], "runs-identical: parameter blocks")
    require(sorted((w["word"], w["frame"], w["bytes"], w["file"]) for w in first["words"]) ==
            sorted((w["word"], w["frame"], w["bytes"], w["file"]) for w in second["words"]),
            "runs-identical: window geometry")
    for entry in first["words"]:
        other = next(w for w in second["words"]
                     if w["frame"] == entry["frame"] and w["word"] == entry["word"])
        require(read_capture_file(a, entry["file"], "runs-identical: file missing") ==
                read_capture_file(b, other["file"], "runs-identical: file missing"),
                "runs-identical: window content")
    require(set(first["files"]) == set(second["files"]), "runs-identical: file set")
    for name in sorted(first["files"]):
        require(read_capture_file(a, name, "runs-identical: file missing") ==
                read_capture_file(b, name, "runs-identical: file missing"),
                "runs-identical: file content")
    return {"pair": [str(a), str(b)], "windows": len(first["words"])}


# ---------------------------------------------------------------------------
# Self-test fixtures
# ---------------------------------------------------------------------------

def write_fixture(root: Path, trace: bool = True) -> None:
    root.mkdir(parents=True, exist_ok=True)
    window_bytes = 4096  # clamped window: res_size - off == 4096
    lines = [f"protocol boundary-capture-v1 frames={FRAMES} size={PROBE_SIZE} "
             f"context=eye0 window={WINDOW} words={len(WORDS)}",
             "sidecar=fixture-va.txt"]
    sidecar_lines: list[str] = []
    words = [0] * 10
    words[8] = struct.unpack("<Q", struct.pack("<2I", DIMENSION, WIDTH))[0]
    words[9] = struct.unpack("<Q", struct.pack("<2I", *OVERRIDES))[0]
    for frame in range(FRAMES):
        base_va = 0x10000000 + frame * 0x1000000
        for index, slot in enumerate(WORDS):
            words[slot] = base_va + index * 0x10000
        words[4] = base_va + 0x10000 * 3  # counter slot position preserved
        hexdata = struct.pack("<10Q", *words).hex()
        lines.append(f"boundary-frame={frame} eval={frame} psize={PSIZE} pmem={hexdata}")
        sidecar_lines.append(f"boundary eval={frame} psize={PSIZE} pmem={hexdata}")
        for index, slot in enumerate(WORDS):
            va = words[slot]
            name = f"frame{frame:02d}-w{slot}.bin"
            if slot == 4:
                payload = struct.pack(f"<{COUNTER_SLOTS}i", *([COUNTER_VALUE] * COUNTER_SLOTS))
                payload += bytes(window_bytes - len(payload))
            else:
                payload = bytes([frame * 16 + index]) * window_bytes
            (root / name).write_bytes(payload)
            offset = va - base_va
            lines.append(f"word={slot} name={WORDS[slot]} frame={frame} eval={frame} "
                         f"va=0x{va:x} res_va=0x{base_va:x} res_size={offset + window_bytes} "
                         f"off={offset} bytes={window_bytes} file={name} "
                         f"fnv=0x{fnv1a64(payload):016x}")
        if frame == 0:
            for name in ("input-colour.bin", "input-depth.bin", "input-motion.bin"):
                payload = bytes(64)
                (root / name).write_bytes(payload)
                lines.append(f"{name} bytes=64 fnv=0x{fnv1a64(payload):016x}")
        output = nan_sentinel_bytes()
        (root / f"output-frame{frame:02d}.bin").write_bytes(output)
        lines.append(f"output-frame{frame:02d}.bin bytes={len(output)} "
                     f"fnv=0x{fnv1a64(output):016x}")
    (root / "manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "boundary-va.txt").write_text("\n".join(sidecar_lines) + "\n", encoding="utf-8")
    if trace:
        trace_lines = ["cut-init armed=1 evals=0 drops=0"]
        for frame in range(FRAMES):
            trace_lines.append(f"cut-armed eval={frame} handle=0x1")
            trace_lines += ["cut-drop handle=0x2 name=x"] * CUT_DROPS_PER_EVAL
            if frame < FRAMES - 1:
                trace_lines.append(f"cut-release eval={frame + 1} drops=0")
        (root / "trace.log").write_text("\n".join(trace_lines) + "\n", encoding="utf-8")


def repin(root: Path, name: str) -> None:
    """Recompute a window's recorded digest so a later semantic check is the
    one that fires (an un-pinned digest would trip manifest-consistent first).
    Word records read `file=<name> fnv=`, capture files `<name> bytes=.. fnv=`."""
    path = root / name
    digest = fnv1a64(path.read_bytes())
    manifest = (root / "manifest.txt").read_text(encoding="utf-8")
    pattern = rf"((?:file=)?{re.escape(name)}(?: bytes=\d+)? fnv=)0x[0-9a-f]+"
    patched = re.sub(pattern, rf"\g<1>0x{digest:016x}", manifest, count=1)
    require(patched != manifest, "self-test: repin target missing")
    (root / "manifest.txt").write_text(patched, encoding="utf-8")


def mutate(root: Path, name: str) -> None:
    manifest = (root / "manifest.txt").read_text(encoding="utf-8")
    sidecar_path = root / "boundary-va.txt"

    def rewrite_both(transform: Any) -> None:
        """Mutate manifest and sidecar consistently so deeper checks remain
        reachable (an inconsistent sidecar would trip sidecar-records first)."""
        (root / "manifest.txt").write_text(transform(manifest), encoding="utf-8")
        sidecar_path.write_text(
            transform(sidecar_path.read_text(encoding="utf-8")), encoding="utf-8")

    if name == "manifest-protocol: frame count":
        manifest = manifest.replace("frames=3 size=", "frames=4 size=", 1)
        (root / "manifest.txt").write_text(manifest, encoding="utf-8")
    elif name == "sidecar-records: sidecar mismatch":
        sidecar = sidecar_path.read_text(encoding="utf-8")
        sidecar_path.write_text(sidecar.replace("pmem=0", "pmem=1", 1), encoding="utf-8")
    elif name == "manifest-consistent: untracked file":
        (root / "frame99-w9.bin").write_bytes(bytes(8))
    elif name == "manifest-consistent: file missing":
        (root / "frame00-w0.bin").unlink()
    elif name == "manifest-consistent: file digest":
        data = bytearray((root / "frame00-w0.bin").read_bytes())
        data[0] ^= 0xFF
        (root / "frame00-w0.bin").write_bytes(bytes(data))
    elif name == "manifest-consistent: window offset":
        manifest = manifest.replace("off=65536 bytes", "off=65544 bytes", 1)
        (root / "manifest.txt").write_text(manifest, encoding="utf-8")
    elif name == "manifest-consistent: file size":
        (root / "frame00-w1.bin").write_bytes(bytes(100))
    elif name == "words-complete: frame word count":
        manifest = "\n".join(l for l in manifest.splitlines() if "word=6 " not in l) + "\n"
        (root / "manifest.txt").write_text(manifest, encoding="utf-8")
        for frame in range(FRAMES):
            (root / f"frame{frame:02d}-w6.bin").unlink()
    elif name == "words-complete: frame word slots":
        manifest = manifest.replace("word=6 name=partial-sums", "word=5 name=partial-sums", 1)
        (root / "manifest.txt").write_text(manifest, encoding="utf-8")
    elif name == "words-complete: pmem length":
        rewrite_both(lambda text: re.sub(r"pmem=[0-9a-f]+", "pmem=abcd", text, count=1))
    elif name == "boundary-constants: width":
        rewrite_both(lambda text: text.replace(
            struct.pack("<2I", DIMENSION, WIDTH).hex(),
            struct.pack("<2I", DIMENSION, 15).hex()))
    elif name == "boundary-constants: output-axis overrides":
        rewrite_both(lambda text: text.replace(
            struct.pack("<2I", *OVERRIDES).hex(), struct.pack("<2I", 24, 29).hex()))
    elif name == "boundary-constants: dimension":
        rewrite_both(lambda text: text.replace(
            struct.pack("<2I", DIMENSION, WIDTH).hex(),
            struct.pack("<2I", 13, WIDTH).hex()))
    elif name == "counter-published: counter values":
        data = bytearray((root / "frame01-w4.bin").read_bytes())
        data[8] = 2
        (root / "frame01-w4.bin").write_bytes(bytes(data))
        repin(root, "frame01-w4.bin")
    elif name == "output-untouched: output not the NaN sentinel":
        data = bytearray((root / "output-frame02.bin").read_bytes())
        data[7] ^= 0x01
        (root / "output-frame02.bin").write_bytes(bytes(data))
        repin(root, "output-frame02.bin")
    elif name == "cut-drops: withheld launches":
        trace = (root / "trace.log").read_text(encoding="utf-8")
        trace = trace.replace("cut-drop ", "cut-keep ", 1)
        (root / "trace.log").write_text(trace, encoding="utf-8")
    elif name == "cut-drops: anomaly records":
        trace = (root / "trace.log").read_text(encoding="utf-8")
        trace = trace.replace("cut-drop ", "cut-anomaly ", 1)
        (root / "trace.log").write_text(trace, encoding="utf-8")
    elif name == "cut-drops: armed evaluations":
        trace = (root / "trace.log").read_text(encoding="utf-8")
        (root / "trace.log").write_text(trace.replace("cut-armed ", "cut-xarmed ", 1),
                                        encoding="utf-8")
    elif name == "runs-identical: window content":
        data = bytearray((root / "frame00-w2.bin").read_bytes())
        data[3] ^= 0x20
        (root / "frame00-w2.bin").write_bytes(bytes(data))
        repin(root, "frame00-w2.bin")
    else:
        raise AssertionError(f"unknown mutation {name}")


def self_test() -> dict[str, Any]:
    capture_controls = [
        "manifest-protocol: frame count",
        "sidecar-records: sidecar mismatch",
        "manifest-consistent: untracked file",
        "manifest-consistent: file missing",
        "manifest-consistent: file digest",
        "manifest-consistent: window offset",
        "manifest-consistent: file size",
        "words-complete: frame word count",
        "words-complete: frame word slots",
        "words-complete: pmem length",
        "boundary-constants: width",
        "boundary-constants: output-axis overrides",
        "boundary-constants: dimension",
        "counter-published: counter values",
        "output-untouched: output not the NaN sentinel",
        "cut-drops: withheld launches",
        "cut-drops: anomaly records",
        "cut-drops: armed evaluations",
    ]
    pair_controls = ["runs-identical: window content"]
    import shutil
    import tempfile
    checked = 0
    with tempfile.TemporaryDirectory(prefix="boundary-selftest-") as tmp:
        base = Path(tmp)
        good = base / "good"
        write_fixture(good)
        result = check_capture(good, good / "trace.log")
        require(result["constants"]["width"] == WIDTH, "self-test: baseline constants")
        require(check_pair(good, good)["windows"] == FRAMES * len(WORDS),
                "self-test: baseline pair")
        for control in capture_controls:
            target = base / control.replace(": ", "-").replace(" ", "-")
            shutil.copytree(good, target)
            mutate(target, control)
            try:
                check_capture(target, target / "trace.log")
            except Reject as error:
                require(str(error) == control, f"self-test: wrong message {error}")
                checked += 1
                continue
            raise AssertionError(f"control not rejected: {control}")
        # Pair rejection: mutate one run and require the pair check to fail.
        for control in pair_controls:
            other = base / "pair-other"
            shutil.copytree(good, other)
            mutate(other, control)
            try:
                check_pair(good, other)
            except Reject as error:
                require(str(error) == control, f"self-test: wrong pair message {error}")
                checked += 1
                continue
            raise AssertionError(f"pair control not rejected: {control}")
        # Direct record comparator with synthetic payload/window bytes.
        payload = bytes(range(256)) * 4
        window = payload + bytes(64)
        require(window[:len(payload)] == payload, "self-test: record comparator")
        window_corrupt = bytearray(window)
        window_corrupt[17] ^= 0x08
        require(bytes(window_corrupt)[:len(payload)] != payload,
                "self-test: record comparator corruption")
        checked += 2
    return {"mode": "self-test",
            "controls": len(capture_controls) + len(pair_controls) + 2,
            "checked": checked}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(prog=NAME)
    parser.add_argument("mode", choices=("report", "pair", "self-test"))
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--record-dll", type=Path)
    args = parser.parse_args()
    output: dict[str, Any]
    if args.mode == "self-test":
        output = self_test()
    elif args.mode == "report":
        require(len(args.paths) == 1, "report: one capture directory")
        output = check_capture(args.paths[0], args.trace)
        if args.record_dll is not None:
            output["weights"] = check_weights_record(args.paths[0], args.record_dll)
    else:
        require(len(args.paths) == 2, "pair: two capture directories")
        output = check_pair(*args.paths)
    print(json.dumps({"tool": NAME, **output}, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Reject as rejected:
        print(f"REJECT: {rejected}")
        sys.exit(1)
