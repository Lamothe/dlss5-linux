#!/usr/bin/env python3
"""Compare two reference captures (or a native run against a vendor capture).

Strict mode (default) compares byte-for-byte and via manifest digests: the
expected result for two vendor runs with identical inputs and protocol.

Float mode treats .bin payloads as little-endian IEEE half (2-byte) or float
(4-byte) samples -- selected by file name (input-depth is float32, the rest
fp16) -- and reports per-file differing-sample counts and max absolute
difference. With --tolerance T, float mode's verdict becomes `within-tolerance`
when every differing sample is within T (NaN equals NaN; NaN vs a number is
differing and reported as infinite distance). This is for future native-vs-
vendor comparisons where bitwise identity is not required.

Evidence discipline: this tool reports counts and digests only.
"""
import argparse
import json
import math
import re
import struct
import sys
from pathlib import Path

MANIFEST_LINE = re.compile(r"^(?P<name>\S+) bytes=(?P<bytes>\d+) fnv=0x(?P<fnv>[0-9a-f]{16})$")
PROTOCOL_LINE = re.compile(r"^protocol (?P<protocol>.+)$")


def read_manifest(directory: Path):
    files = {}
    protocol = None
    for line in (directory / "manifest.txt").read_text().splitlines():
        match = MANIFEST_LINE.match(line)
        if match:
            files[match.group("name")] = {
                "bytes": int(match.group("bytes")),
                "fnv": match.group("fnv"),
            }
            continue
        match = PROTOCOL_LINE.match(line)
        if match:
            protocol = match.group("protocol")
    return protocol, files


def fnv1a64(data: bytes) -> str:
    value = 14695981039346656037
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def sample_format(name: str):
    if name == "input-depth.bin":
        return "f", 4
    return "e", 2  # IEEE half


def compare_float(name: str, payload_a: bytes, payload_b: bytes):
    fmt, width = sample_format(name)
    if len(payload_a) % width != 0 or len(payload_b) % width != 0:
        return {
            "samples": 0,
            "differing_samples": 0,
            "nan_mismatches": 0,
            "max_abs_diff": math.inf,
            "partial_sample_bytes": True,
        }
    count = len(payload_a) // width
    values_a = struct.unpack(f"<{count}{fmt}", payload_a[: count * width])
    values_b = struct.unpack(f"<{count}{fmt}", payload_b[: count * width])
    differing = 0
    max_abs = 0.0
    nan_mismatch = 0
    for left, right in zip(values_a, values_b):
        if left == right:
            continue
        left_nan = math.isnan(left)
        right_nan = math.isnan(right)
        if left_nan and right_nan:
            continue  # NaN equals NaN for comparison purposes
        differing += 1
        if left_nan != right_nan:
            nan_mismatch += 1
            max_abs = math.inf
            continue
        difference = abs(left - right)
        if difference > max_abs:
            max_abs = difference
    return {
        "samples": count,
        "differing_samples": differing,
        "nan_mismatches": nan_mismatch,
        "max_abs_diff": max_abs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, type=Path)
    parser.add_argument("--b", required=True, type=Path)
    parser.add_argument("--mode", choices=("strict", "float"), default="strict")
    parser.add_argument("--tolerance", type=float, default=None,
                        help="float mode: accept differences up to this absolute value")
    parser.add_argument("--report", type=Path, help="write JSON report here")
    args = parser.parse_args()

    protocol_a, files_a = read_manifest(args.a)
    protocol_b, files_b = read_manifest(args.b)
    report = {
        "mode": args.mode,
        "tolerance": args.tolerance,
        "a": str(args.a),
        "b": str(args.b),
        "protocol_a": protocol_a,
        "protocol_b": protocol_b,
        "protocol_match": protocol_a == protocol_b,
        "files": {},
    }
    if set(files_a) != set(files_b):
        report["file_sets_match"] = False
        report["only_a"] = sorted(set(files_a) - set(files_b))
        report["only_b"] = sorted(set(files_b) - set(files_a))
    else:
        report["file_sets_match"] = True

    common = sorted(set(files_a) & set(files_b))
    report["files_compared"] = len(common)
    all_consistent = report["protocol_match"] and report["file_sets_match"] and bool(common)
    all_within_tolerance = all_consistent
    for name in common:
        payload_a = (args.a / name).read_bytes()
        payload_b = (args.b / name).read_bytes()
        entry = {
            "bytes_a": len(payload_a),
            "bytes_b": len(payload_b),
            "manifest_fnv_a": files_a[name]["fnv"],
            "manifest_fnv_b": files_b[name]["fnv"],
            "fnv_computed_a": fnv1a64(payload_a),
            "fnv_computed_b": fnv1a64(payload_b),
        }
        entry["manifest_consistent"] = (
            entry["fnv_computed_a"] == entry["manifest_fnv_a"]
            and entry["fnv_computed_b"] == entry["manifest_fnv_b"]
        )
        entry["bit_identical"] = payload_a == payload_b
        if args.mode == "float":
            if len(payload_a) != len(payload_b):
                entry["length_mismatch"] = True
                entry["within_tolerance"] = False
                all_within_tolerance = False
            else:
                entry["float"] = compare_float(name, payload_a, payload_b)
                within = (
                    entry["float"]["max_abs_diff"] <= args.tolerance
                    if args.tolerance is not None
                    else entry["bit_identical"]
                )
                entry["within_tolerance"] = within
                if not within:
                    all_within_tolerance = False
        report["files"][name] = entry
        if not entry["manifest_consistent"]:
            all_consistent = False
            all_within_tolerance = False
        if args.mode == "strict" and not entry["bit_identical"]:
            all_within_tolerance = False

    if not all_consistent:
        report["verdict"] = "divergent"
    elif args.mode == "float" and args.tolerance is not None:
        report["verdict"] = "within-tolerance" if all_within_tolerance else "beyond-tolerance"
    else:
        report["verdict"] = "bit-identical" if all_within_tolerance else "divergent"

    text = json.dumps(report, indent=2)
    if args.report:
        args.report.write_text(text + "\n")
    print(text)
    return 0 if report["verdict"] in ("bit-identical", "within-tolerance") else 1


if __name__ == "__main__":
    sys.exit(main())
