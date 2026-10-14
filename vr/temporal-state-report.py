#!/usr/bin/env python3
"""Validate the temporal-state contract from retained capture evidence.

Evidence sources (all retained; reports carry counts, digests and derived
aggregates only):
  vr/logs/reference/run-a          capture-v1, DLSSNR.Reset at frame 0 only
  vr/logs/temporal/temporal-a/b     capture-reset-mid-v1, reset at frames 0+4
  vr/logs/temporal/temporal-c/d     capture-reset-mid-v1, reset at frames 0+5
  vr/logs/boundary/a               boundary-capture-v1 decoder-boundary windows

Two unequal reset offsets are pinned per run (temporal-a/b: 4, segments 4+4;
temporal-c/d: 5, segments 5+3) so that a frame-index-dependent behaviour can
only pass if it is periodic with a period dividing BOTH restart strides — i.e.
no period in 2..7 survives (see vr/TEMPORAL-STATE.md).

Contract checked:
  - manifest-digests / size-mismatch / protocol-mismatch / missing-evidence /
    unexpected-evidence / unlisted-file: per-run structure vs pinned tables
  - anchor-mismatch:        run-a digests and the boundary NaN sentinel are
                            pinned in this module (closes coordinated file+
                            manifest corruption for the anchored files)
  - pair-mismatch:          temporal-a == temporal-b, temporal-c == temporal-d
  - input-mismatch:         capture inputs identical across all runs
  - prefix-mismatch:        temporal frames before the reset == run-a frames
  - restart-mismatch:       temporal frame(n) == frame(n - offset)
  - no-divergence:          temporal frames from the reset differ from run-a
  - weights-changed:        boundary weights window static across frames
  - counter-changed:        boundary counter region [0,512) static
  - carrier-stalled:        boundary main/skip/out0/partial windows evolve

Evidence discipline: this tool reports counts, digests and derived aggregates
(changed-sample counts, MAD) only — never payload content.
"""
import argparse
import re
import shutil
import struct
import sys
import tempfile
from pathlib import Path

MANIFEST_LINE = re.compile(r"^(?P<name>\S+) bytes=(?P<bytes>\d+) fnv=0x(?P<fnv>[0-9a-f]{16})$")
WORD_LINE = re.compile(r"^word=\d+ \S+ .*? bytes=(?P<bytes>\d+) file=(?P<name>\S+) fnv=0x(?P<fnv>[0-9a-f]{16})$")
PROTOCOL_LINE = re.compile(r"^protocol (?P<protocol>.+)$")

CAPTURE_NAMES = ["input-colour.bin", "input-depth.bin", "input-motion.bin"] + \
                [f"output-frame{n:02d}.bin" for n in range(8)]
BOUNDARY_WORDS = ["w0", "w1", "w2", "w4", "w6", "w7"]
BOUNDARY_NAMES = ["input-colour.bin", "input-depth.bin", "input-motion.bin"] + \
                 [f"output-frame{n:02d}.bin" for n in range(3)] + \
                 [f"frame{n:02d}-{w}.bin" for n in range(3) for w in BOUNDARY_WORDS]
NON_EVIDENCE_FILES = {"manifest.txt", "boundary-va.txt"}

# Pinned per-run reset offsets: pairs (label, offset). Segments are 4+4 and
# 5+3, so no period in 2..7 divides both restart strides.
RESET_MID_RUNS = {"temporal-a": 4, "temporal-b": 4, "temporal-c": 5, "temporal-d": 5}
PAIRS = [("temporal-a", "temporal-b"), ("temporal-c", "temporal-d")]

PROTOCOL_REFERENCE = "capture-v1 frames=8 size=720 context=eye0 inputs=gradient-stationary reset=frame0"
PROTOCOL_BOUNDARY = "boundary-capture-v1 frames=3 size=720 context=eye0 window=2097152 words=6"

COUNTER_REGION = 512
CARRIERS = ["w0", "w1", "w2", "w6"]

# External anchors: run-a's digest table (as published in REFERENCE-CAPTURE.md
# and pinned there) and the boundary NaN-sentinel digest. Closing the chain for
# the anchored files; unanchored boundary windows are pinned by their manifests.
REAL_REFERENCE_ANCHORS = {
    "input-colour.bin": "fb51a175911c82f5",
    "input-depth.bin": "33533d64f7d84205",
    "input-motion.bin": "227ea3796d10f325",
    "output-frame00.bin": "839de3f30038880a",
    "output-frame01.bin": "65cba335fdd167ed",
    "output-frame02.bin": "1f5df6063c506f74",
    "output-frame03.bin": "9997e15cd2957237",
    "output-frame04.bin": "c49fb39331ef808f",
    "output-frame05.bin": "be6ebbf0d7da3676",
    "output-frame06.bin": "61050c5427b0ea49",
    "output-frame07.bin": "bbba397b954e15eb",
}
REAL_BOUNDARY_SENTINEL = "a0aa045f6a5e4325"


def protocol_reset_mid(frame: int) -> str:
    return ("capture-reset-mid-v1 frames=8 size=720 context=eye0 "
            f"inputs=gradient-stationary reset=frame0+frame{frame}")


# Pinned capture sizes (bytes) per name. Fixture mode uses a disjoint table so
# fixture evidence can never validate against real pins unnoticed.
def capture_sizes(sample: int, depth: int) -> dict:
    sizes = {"input-colour.bin": sample, "input-depth.bin": depth, "input-motion.bin": depth}
    sizes.update({f"output-frame{n:02d}.bin": sample for n in range(8)})
    return sizes


def boundary_sizes(sample: int, depth: int, window: int, clamped: int) -> dict:
    sizes = {"input-colour.bin": sample, "input-depth.bin": depth, "input-motion.bin": depth}
    sizes.update({f"output-frame{n:02d}.bin": sample for n in range(3)})
    for n in range(3):
        for w in BOUNDARY_WORDS:
            # The main-input window clamps to the resource end in every frame.
            sizes[f"frame{n:02d}-{w}.bin"] = clamped if w == "w0" else window
    return sizes


REAL_CAPTURE_SIZES = capture_sizes(4147200, 2073600)
REAL_BOUNDARY_SIZES = boundary_sizes(4147200, 2073600, 2097152, 196608)
FIXTURE_CAPTURE_SIZES = capture_sizes(64, 64)
FIXTURE_BOUNDARY_SIZES = boundary_sizes(64, 64, 600, 600)


def fnv1a64(data: bytes) -> str:
    value = 14695981039346656037
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def read_manifest(directory: Path):
    protocols = []
    files = {}
    for line in (directory / "manifest.txt").read_text().splitlines():
        match = PROTOCOL_LINE.match(line)
        if match:
            protocols.append(match.group("protocol"))
            continue
        match = MANIFEST_LINE.match(line) or WORD_LINE.match(line)
        if match:
            files[match.group("name")] = {"bytes": int(match.group("bytes")), "fnv": match.group("fnv")}
    return protocols, files


def check_run(directory: Path, required_names, size_table: dict, expected_protocol: str,
              label: str, anchors: dict):
    """Per-run structural checks. Returns pinned failure messages."""
    failures = []
    manifest_path = directory / "manifest.txt"
    if not manifest_path.is_file():
        return [f"missing-evidence: {label}/manifest.txt"]
    protocols, files = read_manifest(directory)
    if protocols != [expected_protocol]:
        failures.append(f"protocol-mismatch: {label}")
    for name in required_names:
        if name not in files:
            failures.append(f"missing-evidence: {name}")
    for name, entry in sorted(files.items()):
        if name not in required_names:
            failures.append(f"unexpected-evidence: {name}")
            continue
        path = directory / name
        if not path.is_file():
            failures.append(f"missing-evidence: {name}")
            continue
        data = path.read_bytes()
        if len(data) != entry["bytes"] or entry["bytes"] != size_table.get(name):
            failures.append(f"size-mismatch: {name}")
            continue
        digest = fnv1a64(data)
        if digest != entry["fnv"]:
            failures.append(f"digest-mismatch: {name}")
            continue
        anchor = anchors.get(name)
        if anchor is not None and digest != anchor:
            failures.append(f"anchor-mismatch: {name}")
    present = {p.name for p in directory.iterdir()} - NON_EVIDENCE_FILES
    for name in sorted(present - set(files)):
        failures.append(f"unlisted-file: {name}")
    return failures


def pair_equal(dir_a: Path, dir_b: Path, names):
    failures = []
    for name in names:
        path_a, path_b = dir_a / name, dir_b / name
        if not (path_a.is_file() and path_b.is_file()):
            continue  # reported by structural checks
        if path_a.read_bytes() != path_b.read_bytes():
            failures.append(f"pair-mismatch: {name}")
    return failures


def check_temporal_contract(reference: Path, temporal: Path, offset: int):
    """prefix / restart / divergence relations for one reset-mid run."""
    failures = []
    for n in range(8):
        name = f"output-frame{n:02d}.bin"
        if not ((temporal / name).is_file() and (reference / name).is_file()):
            continue
        same = (temporal / name).read_bytes() == (reference / name).read_bytes()
        if n < offset and not same:
            failures.append(f"prefix-mismatch: {name}")
        if n >= offset and same:
            failures.append(f"no-divergence: {name}")
    for n in range(offset, 8):
        name = f"output-frame{n:02d}.bin"
        base = f"output-frame{n - offset:02d}.bin"
        if not all((temporal / x).is_file() for x in (name, base)):
            continue
        if (temporal / name).read_bytes() != (temporal / base).read_bytes():
            failures.append(f"restart-mismatch: {name}")
    return failures


def check_inputs_match(reference: Path, temporal: Path):
    failures = []
    for name in ("input-colour.bin", "input-depth.bin", "input-motion.bin"):
        if not ((reference / name).is_file() and (temporal / name).is_file()):
            continue
        if (reference / name).read_bytes() != (temporal / name).read_bytes():
            failures.append(f"input-mismatch: {name}")
    return failures


def check_boundary_temporal(boundary: Path):
    failures = []
    frames = [f"frame{n:02d}" for n in range(3)]
    for n in (1, 2):
        name = f"{frames[n]}-w7.bin"
        base = f"{frames[0]}-w7.bin"
        if (boundary / name).is_file() and (boundary / base).is_file():
            if (boundary / name).read_bytes() != (boundary / base).read_bytes():
                failures.append(f"weights-changed: {frames[n]}")
        name = f"{frames[n]}-w4.bin"
        base = f"{frames[0]}-w4.bin"
        if (boundary / name).is_file() and (boundary / base).is_file():
            if (boundary / name).read_bytes()[:COUNTER_REGION] != (boundary / base).read_bytes()[:COUNTER_REGION]:
                failures.append(f"counter-changed: {frames[n]}")
    for word in CARRIERS:
        diffs = []
        for n in (1, 2):
            name = f"{frames[n]}-{word}.bin"
            base = f"{frames[n - 1]}-{word}.bin"
            if (boundary / name).is_file() and (boundary / base).is_file():
                diffs.append((boundary / name).read_bytes() != (boundary / base).read_bytes())
        if diffs and not all(diffs):
            failures.append(f"carrier-stalled: {word}")
    return failures


def temporal_report(directory: Path, label: str):
    """Report-only dynamics: per-transition changed samples, FP16 MAD and max."""
    def f16(bits):
        return struct.unpack("<e", struct.pack("<H", bits & 0xFFFF))[0]

    rows = []
    prev = None
    for n in range(8):
        name = f"output-frame{n:02d}.bin"
        path = directory / name
        if not path.is_file():
            continue
        data = path.read_bytes()
        if len(data) % 2:
            continue  # structural checks report the malformation
        cur = struct.unpack(f"<{len(data) // 2}H", data)
        if prev is not None:
            diffs = [abs(f16(x) - f16(y)) for x, y in zip(cur, prev)]
            rows.append({"pair": f"{n - 1}-{n}",
                         "changed": sum(1 for x, y in zip(cur, prev) if x != y),
                         "samples": len(cur),
                         "mad": round(sum(diffs) / len(diffs), 6),
                         "max": round(max(diffs), 4)})
        prev = cur
    return {"label": label, "transitions": rows}


def boundary_report(directory: Path):
    rows = {}
    frames = [f"frame{n:02d}" for n in range(3)]
    for word in BOUNDARY_WORDS:
        counts = []
        size = 0
        for n in range(3):
            path = directory / f"{frames[n]}-{word}.bin"
            if not path.is_file():
                continue
            size = path.stat().st_size
            if n > 0:
                prev = (directory / f"{frames[n - 1]}-{word}.bin").read_bytes()
                counts.append(sum(1 for x, y in zip(path.read_bytes(), prev) if x != y))
        rows[word] = {"changed": counts, "bytes": size}
    return rows


def run_validation(root: Path, capture_sizes_table: dict, boundary_sizes_table: dict,
                   anchors: dict, sentinel: str, quiet: bool = False):
    reference = root / "vr/logs/reference/run-a"
    temporal_runs = {label: root / f"vr/logs/temporal/{label}" for label in RESET_MID_RUNS}
    boundary = root / "vr/logs/boundary/a"
    for path in [reference, boundary] + list(temporal_runs.values()):
        if not path.is_dir():
            raise SystemExit(f"evidence directory missing: {path}")
    failures = []
    failures += check_run(reference, CAPTURE_NAMES, capture_sizes_table,
                          PROTOCOL_REFERENCE, "run-a", anchors)
    for label, offset in RESET_MID_RUNS.items():
        failures += check_run(temporal_runs[label], CAPTURE_NAMES, capture_sizes_table,
                              protocol_reset_mid(offset), label, {})
    boundary_anchors = {f"output-frame{n:02d}.bin": sentinel for n in range(3)}
    failures += check_run(boundary, BOUNDARY_NAMES, boundary_sizes_table,
                          PROTOCOL_BOUNDARY, "boundary-a", boundary_anchors)
    for label_a, label_b in PAIRS:
        failures += pair_equal(temporal_runs[label_a], temporal_runs[label_b], CAPTURE_NAMES)
    for label in RESET_MID_RUNS:
        failures += check_inputs_match(reference, temporal_runs[label])
    for label, offset in RESET_MID_RUNS.items():
        failures += check_temporal_contract(reference, temporal_runs[label], offset)
    failures += check_boundary_temporal(boundary)
    report = {}
    if not failures:  # dynamics reports only for evidence that passed structure
        report = {
            "reference": temporal_report(reference, "run-a"),
            "temporal": {label: temporal_report(path, label) for label, path in temporal_runs.items()},
            "boundary": boundary_report(boundary),
        }
        if not quiet:
            for row in report["reference"]["transitions"]:
                print(f"reference d({row['pair']}): changed={row['changed']}/{row['samples']} "
                      f"mad={row['mad']} max={row['max']}")
            for label, run in report["temporal"].items():
                for row in run["transitions"]:
                    print(f"{label} d({row['pair']}): changed={row['changed']}/{row['samples']} "
                          f"mad={row['mad']} max={row['max']}")
            for word, row in sorted(report["boundary"].items()):
                print(f"boundary {word}: changed={row['changed']} bytes={row['bytes']}")
    return failures, report


# --- self-test fixtures -----------------------------------------------------

def write_manifest(directory: Path, protocol: str, names, sizes: dict, payloads, words=()):
    lines = [f"protocol {protocol}"]
    for name in names:
        if name in words:
            frame = name[5:7]
            index = BOUNDARY_WORDS.index(name[8:-4])
            lines.append(f"word={index} name=fixture frame={int(frame)} eval={int(frame)} "
                         f"va=0x0 res_va=0x0 res_size=0 off=0 bytes={sizes[name]} "
                         f"file={name} fnv=0x{fnv1a64(payloads[name])}")
        else:
            lines.append(f"{name} bytes={sizes[name]} fnv=0x{fnv1a64(payloads[name])}")
    (directory / "manifest.txt").write_text("\n".join(lines) + "\n")


def build_fixture(root: Path):
    """Tiny consistent evidence tree satisfying every check."""
    gen = lambda seed, size: bytes((seed * 31 + i * 7) % 256 for i in range(size))
    reference = root / "vr/logs/reference/run-a"
    boundary = root / "vr/logs/boundary/a"
    temporal_runs = {label: root / f"vr/logs/temporal/{label}" for label in RESET_MID_RUNS}
    for path in [reference, boundary] + list(temporal_runs.values()):
        path.mkdir(parents=True)
    cap = FIXTURE_CAPTURE_SIZES
    base = {f"output-frame{n:02d}.bin": gen(n + 1, cap["output-frame00.bin"]) for n in range(8)}
    inputs = {"input-colour.bin": gen(40, cap["input-colour.bin"]),
              "input-depth.bin": gen(41, cap["input-depth.bin"]),
              "input-motion.bin": gen(42, cap["input-motion.bin"])}
    ref_payloads = dict(inputs, **base)
    write_manifest(reference, PROTOCOL_REFERENCE, CAPTURE_NAMES, cap, ref_payloads)
    for name, payload in ref_payloads.items():
        (reference / name).write_bytes(payload)
    for label, offset in RESET_MID_RUNS.items():
        frames = {}
        for n in range(8):
            # before the reset: shared protocol; from the reset: exact replay
            source = n if n < offset else n - offset
            frames[f"output-frame{n:02d}.bin"] = base[f"output-frame{source:02d}.bin"]
        payloads = dict(inputs, **frames)
        write_manifest(temporal_runs[label], protocol_reset_mid(offset), CAPTURE_NAMES, cap, payloads)
        for name, payload in payloads.items():
            (temporal_runs[label] / name).write_bytes(payload)
    bnd = FIXTURE_BOUNDARY_SIZES
    bnd_payloads = {
        "input-colour.bin": gen(40, bnd["input-colour.bin"]),
        "input-depth.bin": gen(41, bnd["input-depth.bin"]),
        "input-motion.bin": gen(42, bnd["input-motion.bin"]),
    }
    for n in range(3):
        bnd_payloads[f"output-frame{n:02d}.bin"] = gen(0, bnd["output-frame00.bin"])  # constant sentinel
        for w in BOUNDARY_WORDS:
            name = f"frame{n:02d}-{w}.bin"
            if w == "w7":
                bnd_payloads[name] = gen(70, bnd[name])  # static weights
            elif w == "w4":
                payload = bytearray(gen(71 + n, bnd[name]))  # pool churn beyond region
                payload[:COUNTER_REGION] = gen(72, COUNTER_REGION)  # static counter region
                bnd_payloads[name] = bytes(payload)
            elif w == "w0":
                bnd_payloads[name] = gen(120 + n, bnd[name])  # evolves every pair
            else:
                bnd_payloads[name] = gen(80 + n * 3 + BOUNDARY_WORDS.index(w), bnd[name])
    words = {f"frame{n:02d}-{w}.bin" for n in range(3) for w in BOUNDARY_WORDS}
    write_manifest(boundary, PROTOCOL_BOUNDARY, BOUNDARY_NAMES, bnd, bnd_payloads, words=words)
    for name, payload in bnd_payloads.items():
        (boundary / name).write_bytes(payload)
    anchors = {name: fnv1a64(payload) for name, payload in ref_payloads.items()}
    sentinel = fnv1a64(bnd_payloads["output-frame00.bin"])
    return anchors, sentinel


def reseal(directory: Path, name: str, kind: str):
    """Rewrite one manifest entry to match the file on disk (consistent corruption)."""
    path = directory / "manifest.txt"
    data = (directory / name).read_bytes()
    out = []
    for line in path.read_text().splitlines():
        if kind == "word" and f" file={name} " in f" {line} ":
            line = re.sub(r" bytes=\d+ file=", f" bytes={len(data)} file=", line)
            line = re.sub(r"fnv=0x[0-9a-f]{16}", f"fnv=0x{fnv1a64(data)}", line)
        elif kind == "manifest" and line.startswith(f"{name} bytes="):
            line = f"{name} bytes={len(data)} fnv=0x{fnv1a64(data)}"
        out.append(line)
    path.write_text("\n".join(out) + "\n")


def self_test() -> int:
    controls = []
    state = {}

    def build():
        tmp = Path(tempfile.mkdtemp(prefix="temporal-selftest-"))
        anchors, sentinel = build_fixture(tmp)
        state["anchors"], state["sentinel"] = anchors, sentinel
        return tmp

    def run(tmp: Path):
        failures, _ = run_validation(tmp, FIXTURE_CAPTURE_SIZES, FIXTURE_BOUNDARY_SIZES,
                                     state["anchors"], state["sentinel"], quiet=True)
        return failures

    def control(label, mutate, expected, exact=False):
        tmp = build()
        try:
            mutate(tmp)
            failures = run(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        ok = failures == [expected] if exact else expected in failures
        controls.append((label, ok, failures if not ok else None))
        return ok

    def flip(path: Path, byte=-1):
        data = bytearray(path.read_bytes())
        data[byte] ^= 0xFF
        path.write_bytes(bytes(data))

    ta = Path("vr/logs/temporal/temporal-a")
    tb = Path("vr/logs/temporal/temporal-b")
    tc = Path("vr/logs/temporal/temporal-c")
    td = Path("vr/logs/temporal/temporal-d")
    br = Path("vr/logs/boundary/a")
    ra = Path("vr/logs/reference/run-a")

    tmp = build()
    try:
        baseline_failures = run(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    controls.append(("baseline-clean", baseline_failures == [], baseline_failures))

    control("file-flip",
            lambda t: flip(t / ta / "output-frame00.bin"),
            "digest-mismatch: output-frame00.bin")
    control("file-truncated",
            lambda t: (t / ta / "output-frame01.bin").write_bytes((t / ta / "output-frame01.bin").read_bytes()[:16]),
            "size-mismatch: output-frame01.bin")
    control("manifest-line-dropped",
            lambda t: (t / ta / "manifest.txt").write_text(
                "\n".join(l for l in (t / ta / "manifest.txt").read_text().splitlines()
                          if not l.startswith("output-frame07.bin")) + "\n"),
            "missing-evidence: output-frame07.bin")
    control("extra-file",
            lambda t: (t / ra / "mystery.bin").write_bytes(b"\x01\x02"),
            "unlisted-file: mystery.bin")
    control("extra-txt-file",
            lambda t: (t / ra / "notes.txt").write_bytes(b"\x01\x02"),
            "unlisted-file: notes.txt")
    control("protocol-flip",
            lambda t: (t / tc / "manifest.txt").write_text(
                (t / tc / "manifest.txt").read_text().replace(protocol_reset_mid(5), PROTOCOL_REFERENCE)),
            "protocol-mismatch: temporal-c")
    control("protocol-duplicate",
            lambda t: (t / ra / "manifest.txt").write_text(
                (t / ra / "manifest.txt").read_text().replace(
                    PROTOCOL_REFERENCE, PROTOCOL_REFERENCE + "\nprotocol bogus-line")),
            "protocol-mismatch: run-a")
    control("anchor-flip-consistent",
            lambda t: (flip(t / ra / "output-frame05.bin"), reseal(t / ra, "output-frame05.bin", "manifest")),
            "anchor-mismatch: output-frame05.bin")
    control("sentinel-flip-consistent",
            lambda t: (flip(t / br / "output-frame01.bin"), reseal(t / br, "output-frame01.bin", "manifest")),
            "anchor-mismatch: output-frame01.bin")
    control("pair-flip-consistent",
            lambda t: (flip(t / tb / "output-frame00.bin"), reseal(t / tb, "output-frame00.bin", "manifest")),
            "pair-mismatch: output-frame00.bin")
    control("pair2-flip-consistent",
            lambda t: (flip(t / td / "output-frame00.bin"), reseal(t / td, "output-frame00.bin", "manifest")),
            "pair-mismatch: output-frame00.bin")
    control("input-flip-consistent",
            lambda t: (flip(t / ta / "input-colour.bin"), reseal(t / ta, "input-colour.bin", "manifest")),
            "input-mismatch: input-colour.bin")
    control("prefix-flip-consistent",
            lambda t: (flip(t / tc / "output-frame03.bin"), reseal(t / tc, "output-frame03.bin", "manifest")),
            "prefix-mismatch: output-frame03.bin")
    control("restart-flip-consistent",
            lambda t: (flip(t / ta / "output-frame04.bin"), reseal(t / ta, "output-frame04.bin", "manifest")),
            "restart-mismatch: output-frame04.bin")
    control("restart5-flip-consistent",
            lambda t: (flip(t / tc / "output-frame07.bin"), reseal(t / tc, "output-frame07.bin", "manifest")),
            "restart-mismatch: output-frame07.bin")
    control("no-divergence-consistent",
            lambda t: [((t / ta / f"output-frame{n:02d}.bin").write_bytes((t / ra / f"output-frame{n:02d}.bin").read_bytes()),
                        (t / tb / f"output-frame{n:02d}.bin").write_bytes((t / ra / f"output-frame{n:02d}.bin").read_bytes()),
                        reseal(t / ta, f"output-frame{n:02d}.bin", "manifest"),
                        reseal(t / tb, f"output-frame{n:02d}.bin", "manifest")) for n in range(4, 8)],
            "no-divergence: output-frame04.bin")
    control("no-divergence5-consistent",
            lambda t: [((t / tc / f"output-frame{n:02d}.bin").write_bytes((t / ra / f"output-frame{n:02d}.bin").read_bytes()),
                        (t / td / f"output-frame{n:02d}.bin").write_bytes((t / ra / f"output-frame{n:02d}.bin").read_bytes()),
                        reseal(t / tc, f"output-frame{n:02d}.bin", "manifest"),
                        reseal(t / td, f"output-frame{n:02d}.bin", "manifest")) for n in range(5, 8)],
            "no-divergence: output-frame05.bin")
    # exact=True pins the word-reseal branch: without a working manifest
    # rewrite the digest check would add digest-mismatch and this fails.
    control("weights-flip-consistent",
            lambda t: (flip(t / br / "frame01-w7.bin"), reseal(t / br, "frame01-w7.bin", "word")),
            "weights-changed: frame01", exact=True)
    control("counter-flip-consistent",
            lambda t: (lambda d: ((t / br / "frame01-w4.bin").write_bytes(d[:8] + b"\xee" + d[9:]),
                                  reseal(t / br, "frame01-w4.bin", "word")))((t / br / "frame01-w4.bin").read_bytes()),
            "counter-changed: frame01")
    control("carrier-stall-consistent",
            lambda t: ((t / br / "frame01-w0.bin").write_bytes((t / br / "frame00-w0.bin").read_bytes()),
                       reseal(t / br, "frame01-w0.bin", "word")),
            "carrier-stalled: w0")
    control("word-truncated-reseal",
            lambda t: ((t / br / "frame01-w6.bin").write_bytes((t / br / "frame01-w6.bin").read_bytes()[:100]),
                       reseal(t / br, "frame01-w6.bin", "word")),
            "size-mismatch: frame01-w6.bin")

    # size pins themselves: fixture evidence must not validate against the real
    # size tables. Assert a capture-only AND a boundary-only rejection so the
    # guard cannot pass via the other table's failures.
    tmp = build()
    try:
        real_failures, _ = run_validation(tmp, REAL_CAPTURE_SIZES, REAL_BOUNDARY_SIZES,
                                          state["anchors"], state["sentinel"], quiet=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    guard_ok = ("size-mismatch: output-frame07.bin" in real_failures
                and "size-mismatch: frame00-w0.bin" in real_failures)
    controls.append(("size-table-guard", guard_ok, None if guard_ok else real_failures))

    ok = True
    for label, passed, detail in controls:
        print(f"{'PASS' if passed else 'FAIL'} control {label}" + ("" if passed else f" failures={detail}"))
        ok = ok and passed
    print(f"{'PASS' if ok else 'FAIL'}: temporal-state self-test controls={len(controls)}")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["validate", "self-test"], nargs="?", default="validate")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    try:
        if args.mode == "self-test":
            return self_test()
        failures, report = run_validation(args.root, REAL_CAPTURE_SIZES, REAL_BOUNDARY_SIZES,
                                          REAL_REFERENCE_ANCHORS, REAL_BOUNDARY_SENTINEL)
        if failures:
            for message in failures:
                print(f"REJECT: {message}")
            print(f"REJECT: temporal-state contract failures={len(failures)}")
            return 1
        transitions = len(report["reference"]["transitions"])
        restart_checks = sum(8 - offset for offset in RESET_MID_RUNS.values())
        print("PASS: temporal-state contract validated "
              f"(runs=5 boundary=1 restart-checks={restart_checks} transitions={transitions})")
        return 0
    except SystemExit:
        raise
    except Exception as error:  # fail closed with a diagnostic, never a traceback
        print(f"REJECT: temporal-state report error: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
