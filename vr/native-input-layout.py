#!/usr/bin/env python3
"""Validate and pack documented reference input planes for native adapters.

Strips row padding while preserving sample bytes. This utility does not define
DLSS preprocessing, colour space, coordinate conventions, or runtime bindings.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

REFERENCE_INPUT_LAYOUTS = {
    "input-colour.bin": (4, 2),
    "input-depth.bin": (1, 4),
    "input-motion.bin": (2, 2),
}
MANIFEST_LINE = re.compile(r"^(?P<name>\S+) bytes=(?P<bytes>\d+) fnv=0x(?P<fnv>[0-9a-f]{16})$")


def fnv1a64(data: bytes | bytearray | memoryview) -> str:
    """Return the capture manifest's FNV-1a-64 representation."""
    value = 14695981039346656037
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def pack_rows(data: bytes | bytearray | memoryview, *, width: int, height: int,
              components: int, bytes_per_component: int, row_pitch: int) -> bytes:
    """Return tight-packed rows; reject invalid dimensions, pitch, or buffer size."""
    values = (width, height, components, bytes_per_component, row_pitch)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("dimensions, component sizes and row pitch must be positive integers")
    row_bytes = width * components * bytes_per_component
    if row_pitch < row_bytes:
        raise ValueError("row pitch is smaller than the active row")
    view = memoryview(data)
    if view.ndim != 1 or view.itemsize != 1 or not view.contiguous:
        raise ValueError("input must be a contiguous byte buffer")
    expected = row_pitch * height
    if view.nbytes != expected:
        raise ValueError(f"buffer size {view.nbytes} does not match pitched size {expected}")
    if row_pitch == row_bytes:
        return bytes(view)
    packed = bytearray(row_bytes * height)
    for row in range(height):
        source = row * row_pitch
        target = row * row_bytes
        packed[target:target + row_bytes] = view[source:source + row_bytes]
    return bytes(packed)


def pack_reference_input(name: str, data: bytes | bytearray | memoryview, *,
                         width: int, height: int, row_pitch: int) -> bytes:
    """Pack a documented reference plane by its capture filename and format."""
    try:
        components, bytes_per_component = REFERENCE_INPUT_LAYOUTS[name]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unsupported reference input plane: {name!r}") from error
    return pack_rows(data, width=width, height=height, components=components,
                     bytes_per_component=bytes_per_component, row_pitch=row_pitch)


def read_input_manifest(directory: Path) -> dict[str, tuple[int, str]]:
    """Read required input entries from a capture manifest; reject ambiguity."""
    manifest: dict[str, tuple[int, str]] = {}
    for line in (directory / "manifest.txt").read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        candidate_name = fields[0] if fields else ""
        match = MANIFEST_LINE.fullmatch(line)
        if candidate_name in REFERENCE_INPUT_LAYOUTS:
            if match is None:
                raise ValueError(f"{directory / 'manifest.txt'}: malformed entry for {candidate_name}")
            if candidate_name in manifest:
                raise ValueError(f"{directory / 'manifest.txt'}: duplicate entry for {candidate_name}")
            manifest[candidate_name] = (int(match.group("bytes")), match.group("fnv"))
    missing = REFERENCE_INPUT_LAYOUTS.keys() - manifest.keys()
    if missing:
        raise ValueError(f"{directory / 'manifest.txt'}: missing input entries {sorted(missing)}")
    return manifest


def validate_capture(directory: Path, *, width: int = 720, height: int = 720) -> list[dict[str, object]]:
    """Validate actual input files against tight layout and their local manifest."""
    expected_manifest = read_input_manifest(directory)
    reports: list[dict[str, object]] = []
    for name, (components, bytes_per_component) in REFERENCE_INPUT_LAYOUTS.items():
        path = directory / name
        row_pitch = width * components * bytes_per_component
        expected = row_pitch * height
        payload = path.read_bytes()
        actual = len(payload)
        if actual != expected:
            raise ValueError(f"{path}: size {actual} does not match expected {expected}")
        manifest_bytes, manifest_fnv = expected_manifest[name]
        digest = fnv1a64(payload)
        if (manifest_bytes, manifest_fnv) != (actual, digest):
            raise ValueError(f"{path}: size/digest does not match manifest")
        packed = pack_reference_input(name, payload, width=width, height=height,
                                      row_pitch=row_pitch)
        if len(packed) != expected:
            raise ValueError(f"{path}: packed size {len(packed)} does not match expected {expected}")
        reports.append({"file": name, "bytes": actual, "fnv1a64": digest,
                        "row_pitch": row_pitch, "tight_rows": True,
                        "unchanged": packed == payload})
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="local reference-capture directory")
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    try:
        reports = validate_capture(args.capture, width=args.width, height=args.height)
    except (OSError, ValueError) as error:
        parser.exit(1, f"FAIL: {error}\n")
    for report in reports:
        print(f"PASS {report['file']}: {report['bytes']} bytes; manifest digest matches; tight rows unchanged")


if __name__ == "__main__":
    main()
