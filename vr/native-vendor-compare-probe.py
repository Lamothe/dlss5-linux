#!/usr/bin/env python3
"""Native-vs-vendor decoder comparison: replay captured boundary buffers.

Replays the model's decoder launch natively on Linux (CUDA driver; no Windows
code, DirectX or Proton) using the buffers captured from the live runtime at
the decoder boundary (DECODER-BOUNDARY.md), then compares the native output
against the vendor's captured output byte-for-byte. The vendor's own kernel
(`cc_dec_input_upsample_1024_512_tilesync_fp8`, the slot-99 launch of
NVAPI-ARGS.md) and the recovered/bound sibling
(`cc_dec_input_upsample_1024_512_fp8`, NATIVE-DECODER-BINDING.md) are both
replayed.

Inputs are the captured windows verbatim: main input (word 0), skip input
(word 1) and weights (word 7) are fed byte-identically; the counter region is
reset through the recovered `cc_cb_clear` contract at the vendor's own live
count (40,704 slots, grid 159); the 80-byte argument block carries the
captured scalar constants (dimension=12, width=16, output axes 24x28).

The pre-decoder contents of the partial-sum (word 6) and output (word 2)
buffers are NOT captured (the capture reads end-of-frame state), so two
partials pre-states are replayed (zeroed, and the captured window) to test
sensitivity, and out0 is pattern-filled (NATIVE-DECODER-BINDING.md established
pure stores for the recovered variant). Findings are recorded per entry with
scope notes and carry no pass/fail semantics; enforced checks are listed in
`checks_enforced`.

This uses stored learned weight bytes through the captured window (bounded
model-weight computation, NATIVE-FP8.md precedent). Evidence retains digests
and bounded windows only; it is not a neural-rendered image and does not claim
a complete native pipeline. Reuse/publication rights are unresolved.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import ctypes as ct
import importlib.util
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from typing import Any


def load_helper(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    if spec is None or spec.loader is None:
        raise RuntimeError("missing helper")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


decoder = load_helper("decoder", "native-decoder-probe.py")
utility = load_helper("utility", "native-kernel-probe.py")
launch = load_helper("launch", "native-decoder-launch-probe.py")
report = load_helper("report", "decoder-boundary-report.py")
base = utility.base

NAME_LIVE = "cc_dec_input_upsample_1024_512_tilesync_fp8"
NAME_RECOVERED = "cc_dec_input_upsample_1024_512_fp8"
RESET = "cc_cb_clear"
BLOCK = (32, 2, 1)
DIMENSION = 12
WIDTH = 16
OUTPUT_AXES = (24, 28)
GRID = (8, 3, 4)                 # launch_geometry(12, 16); cross-pinned to the
                                 # level-2 live capture (NVAPI-ARGS.md grid=8,3,4).
OUTPUT_BYTES = OUTPUT_AXES[0] * OUTPUT_AXES[1] * 512   # 344,064: recovered rule.
COUNTER_SLOTS = 2 * (DIMENSION // 4) * (WIDTH // 4)    # 24: 2 * rows * columns.
COUNTER_SCOPE = 512              # Decoder-private clear-preserved window (bytes).
CLEAR_COUNT = 40704              # Vendor's live cc_cb_clear count (level-2 capture).
WINDOW = 2 * 1024 * 1024
MAIN_BYTES = 196_608             # Captured w0 window: clamped at the pool end.
LARGE_BYTES = 8 * 1024 * 1024    # skip, weights, partials, out0 live here.
UNUSED_BYTES = 2 * 1024 * 1024   # Census-unused slots out3/out2.
COUNTER_BYTES = WINDOW
GUARD_WORDS = launch.GUARD_WORDS
OUT0_PATTERN = 0xA5
SKIP_PATTERN = 0x6B
WEIGHTS_PATTERN = 0xC3
UNUSED_PATTERN = 0x5A
ZERO_PATTERN = 0x00
MATRIX_BYTES = 524288            # Record layout: NATIVE-DECODER-BINDING.md.
ROTATE_BYTES = 1024              # Wrong-mode matrix rotation (binding precedent).
FRAMES = (0, 1, 2)
KERNEL_TAGS = (("tilesync", NAME_LIVE), ("fp8", NAME_RECOVERED))
PARTIALS_MODES = (("pcap", "capture"), ("pzero", "zero"))

EXECUTE_PLAN = tuple(
    f"{tag}-f{frame}-{pm}" for frame in FRAMES for pm, _ in PARTIALS_MODES
    for tag, _ in KERNEL_TAGS)
SKIP_PLAN = ("tilesync-f0-pcap",)
WRONG_PLAN = ("tilesync-f0-pcap",)
PLANS = {"execute": EXECUTE_PLAN, "skip": SKIP_PLAN, "wrong": WRONG_PLAN}


# ---------------------------------------------------------------------------
# Pure comparisons (CPU)
# ---------------------------------------------------------------------------

def diff_stats(left: bytes, right: bytes, block: int = 65536) -> tuple[bool, int, int]:
    """(match, differing byte count, first differing offset); -1 offset if none.

    Byte counting happens only inside differing blocks, so large equal regions
    stay memcmp-speed.
    """
    base.require(type(left) is bytes and type(right) is bytes, "diff: type")
    base.require(len(left) == len(right) and len(left) > 0 and block > 0, "diff: size")
    count = 0
    first = -1
    for offset in range(0, len(left), block):
        chunk_left = left[offset:offset + block]
        chunk_right = right[offset:offset + block]
        if chunk_left == chunk_right:
            continue
        if first < 0:
            first = offset + next(index for index in range(len(chunk_left))
                                  if chunk_left[index] != chunk_right[index])
        count += sum(1 for x, y in zip(chunk_left, chunk_right) if x != y)
    return (count == 0, count, first)


def change_extent(before: bytes, after: bytes, block: int = 65536) -> tuple[int, int, int]:
    """(first, last, changed blocks) of differing bytes, or (-1, -1, 0)."""
    base.require(len(before) == len(after) and len(before) > 0 and block > 0,
                 "extent: size")
    changed = [offset for offset in range(0, len(before), block)
               if before[offset:offset + block] != after[offset:offset + block]]
    if not changed:
        return (-1, -1, 0)
    head_left = before[changed[0]:changed[0] + block]
    head_right = after[changed[0]:changed[0] + block]
    first = changed[0] + next(index for index in range(len(head_left))
                              if head_left[index] != head_right[index])
    tail_left = before[changed[-1]:changed[-1] + block]
    tail_right = after[changed[-1]:changed[-1] + block]
    last = changed[-1] + next(index for index in reversed(range(len(tail_left)))
                              if tail_left[index] != tail_right[index])
    return (first, last, len(changed))


class Capture:
    """Loaded, digest-verified decoder-boundary capture (selected frames)."""

    def __init__(self, root: Path, frames: tuple[int, ...]) -> None:
        base.require(root.is_dir(), "capture: directory missing")
        self.root = root
        self.manifest = report.parse_manifest(root / "manifest.txt")
        self.frames = frames
        self.windows: dict[tuple[int, int], bytes] = {}
        for frame in frames:
            for word in (0, 1, 2, 4, 6, 7):
                self.windows[(frame, word)] = self._load_window(frame, word)
        self.hexes = {int(item["frame"]): item["hex"] for item in self.manifest["boundary"]}
        for frame in frames:
            words = report.boundary_words(self.hexes[frame])
            entries = {w["word"]: w for w in self.manifest["words"] if w["frame"] == frame}
            for word in (0, 1, 2, 4, 6, 7):
                base.require(entries[word]["va"] == words[word],
                             "capture: word va vs pmem")
            base.require(struct.unpack("<4I", self.scalar_bytes(frame))
                         == (DIMENSION, WIDTH, *OUTPUT_AXES),
                         "capture: boundary scalars")

    def _load_window(self, frame: int, word: int) -> bytes:
        entries = [w for w in self.manifest["words"]
                   if w["frame"] == frame and w["word"] == word]
        base.require(len(entries) == 1, "capture: window entry")
        entry = entries[0]
        data = report.read_capture_file(self.root, entry["file"], "capture: window missing")
        base.require(len(data) == entry["bytes"], "capture: window size")
        base.require(report.fnv1a64(data) == entry["fnv"], "capture: window digest")
        return data

    def scalar_bytes(self, frame: int) -> bytes:
        """The captured 16 constant-bank bytes (argument offsets 64..80), verbatim."""
        words = report.boundary_words(self.hexes[frame])
        return struct.pack("<2Q", *words[8:10])

    def window(self, frame: int, word: int) -> bytes:
        return self.windows[(frame, word)]

    def digest_summary(self) -> dict[str, str]:
        return {f"frame{frame}-w{word}": f"0x{report.fnv1a64(data):016x}"
                for (frame, word), data in sorted(self.windows.items())}


def validate_entry_verdicts(entry: dict[str, Any]) -> None:
    """Consistency of one entry's recorded verdicts (probe-side re-check)."""
    out0 = entry["out0"]
    base.require(out0["match"] == (out0["diff_count"] == 0), "verdict: match vs diff count")
    if out0["match"]:
        base.require(out0["first_diff"] == -1, "verdict: match with diff offset")
    else:
        base.require(0 <= out0["first_diff"] < OUTPUT_BYTES, "verdict: diff offset range")
    partials = entry["partials"]
    base.require(partials["write_scope_match"] == (partials["write_scope_diff_count"] == 0),
                 "verdict: partials scope match vs diff count")
    if partials["write_scope_match"]:
        base.require(partials["write_scope_first_diff"] == -1,
                     "verdict: partials scope match with diff offset")


# ---------------------------------------------------------------------------
# Self-test (CPU only; no CUDA context)
# ---------------------------------------------------------------------------

def write_fixture(root: Path) -> bytes:
    """Minimal valid capture fixture for loader controls; returns the fill bytes."""
    fill = bytes((index * 7 + 3) & 0xFF for index in range(4096))
    lines = ["protocol boundary-capture-v1 frames=3 size=720 context=eye0 "
             "window=2097152 words=6", "sidecar=nvapi-boundary-fixture.txt"]
    for frame in range(3):
        words = [0x1000 + frame * 0x100 + index for index in range(10)]
        words[8] = struct.unpack("<Q", struct.pack("<2I", DIMENSION, WIDTH))[0]
        words[9] = struct.unpack("<Q", struct.pack("<2I", *OUTPUT_AXES))[0]
        hexdata = struct.pack("<10Q", *words).hex()
        lines.append(f"boundary-frame={frame} eval={frame} psize=80 pmem={hexdata}")
        for word in (0, 1, 2, 4, 6, 7):
            name = {0: "main-input", 1: "skip-input", 2: "out0",
                    4: "counter", 6: "partial-sums", 7: "weights"}[word]
            filename = f"frame{frame:02d}-w{word}.bin"
            (root / filename).write_bytes(fill)
            lines.append(f"word={word} name={name} frame={frame} eval={frame} "
                         f"va=0x{words[word]:x} res_va=0x1000 res_size=62684160 "
                         f"off=0 bytes=4096 file={filename} "
                         f"fnv=0x{report.fnv1a64(fill):016x}")
    (root / "manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fill


def self_test() -> dict[str, Any]:
    base.require(ct.sizeof(launch.DecoderArguments) == 80, "decoder host ABI size")
    offsets = {name: getattr(launch.DecoderArguments, name).offset
               for name, _ in launch.DecoderArguments._fields_}
    base.require(offsets == dict(main=0, skip=8, out0=16, out3=24, counters=32, out2=40,
                                 partials=48, weights=56, dimension=64, width=68,
                                 output_axis0=72, output_axis1=76), "decoder host ABI offsets")
    base.require(decoder.launch_geometry(DIMENSION, WIDTH) == GRID, "live launch geometry")
    base.require(launch.reset_geometry(CLEAR_COUNT) == (159, 1, 1), "live clear geometry")
    base.require(OUTPUT_BYTES == 344064, "output extent rule")
    base.require(COUNTER_SLOTS == 24 and COUNTER_SCOPE == 512, "counter rules")
    base.require(len(EXECUTE_PLAN) == 12 and len(set(EXECUTE_PLAN)) == 12, "execute plan")
    base.require(EXECUTE_PLAN[0] == "tilesync-f0-pcap" and EXECUTE_PLAN[-1] == "fp8-f2-pzero",
                 "execute plan order")

    equal = b"\x01" * 1000
    base.require(diff_stats(equal, equal) == (True, 0, -1), "diff equal")
    mutated = bytearray(equal)
    mutated[0] = 0x02
    mutated[700] = 0x03
    base.require(diff_stats(equal, bytes(mutated)) == (False, 2, 0), "diff mutated")
    tail = bytearray(equal)
    tail[999] = 0x04
    base.require(diff_stats(equal, bytes(tail)) == (False, 1, 999), "diff tail")
    base.require(change_extent(equal, equal) == (-1, -1, 0), "extent unchanged")
    base.require(change_extent(equal, bytes(mutated)) == (0, 700, 1), "extent changed")
    base.require(change_extent(equal, bytes(tail)) == (999, 999, 1), "extent tail")
    for bad_call in (lambda: diff_stats(equal, equal[:-1]),
                     lambda: diff_stats(b"", b""),
                     lambda: change_extent(equal, equal + b"\x00"),
                     lambda: change_extent(b"", b"")):
        try:
            bad_call()
        except Exception:
            pass
        else:
            raise ValueError("malformed diff/extent input accepted")

    verdict = {"out0": {"match": True, "diff_count": 0, "first_diff": -1},
               "partials": {"write_scope_match": True, "write_scope_diff_count": 0,
                            "write_scope_first_diff": -1}}
    validate_entry_verdicts(verdict)
    verdict_bad = 0
    for bad in ({"out0": {"match": True, "diff_count": 1, "first_diff": 0},
                 "partials": {"write_scope_match": True, "write_scope_diff_count": 0,
                              "write_scope_first_diff": -1}},
                {"out0": {"match": False, "diff_count": 0, "first_diff": -1},
                 "partials": {"write_scope_match": True, "write_scope_diff_count": 0,
                              "write_scope_first_diff": -1}},
                {"out0": {"match": False, "diff_count": 1, "first_diff": -5},
                 "partials": {"write_scope_match": True, "write_scope_diff_count": 0,
                              "write_scope_first_diff": -1}},
                {"out0": {"match": False, "diff_count": 1, "first_diff": OUTPUT_BYTES},
                 "partials": {"write_scope_match": True, "write_scope_diff_count": 0,
                              "write_scope_first_diff": -1}},
                {"out0": {"match": True, "diff_count": 0, "first_diff": -1},
                 "partials": {"write_scope_match": True, "write_scope_diff_count": 1,
                              "write_scope_first_diff": 0}},
                {"out0": {"match": True, "diff_count": 0, "first_diff": -1},
                 "partials": {"write_scope_match": False, "write_scope_diff_count": 0,
                              "write_scope_first_diff": -1}}):
        try:
            validate_entry_verdicts(bad)
        except Exception:
            verdict_bad += 1
        else:
            raise ValueError("inconsistent entry verdict accepted")

    capture_controls = 0
    with TemporaryDirectory() as scratch:
        root = Path(scratch)
        fill = write_fixture(root)
        capture = Capture(root, (0,))
        base.require(len(capture.window(0, 0)) == 4096, "fixture window")
        base.require(capture.scalar_bytes(0) ==
                     struct.pack("<2I", DIMENSION, WIDTH) +
                     struct.pack("<2I", *OUTPUT_AXES), "fixture scalars")
        manifest_text = (root / "manifest.txt").read_text(encoding="utf-8")
        fill_digest = report.fnv1a64(fill)
        for name, old, new in (
                ("digest", f"fnv=0x{fill_digest:016x}", f"fnv=0x{fill_digest ^ 1:016x}"),
                ("size", "bytes=4096", "bytes=4095"),
                ("scalars", struct.pack("<2I", DIMENSION, WIDTH).hex(),
                 struct.pack("<2I", DIMENSION, 15).hex())):
            (root / "manifest.txt").write_text(
                manifest_text.replace(old, new, 1), encoding="utf-8")
            try:
                Capture(root, (0,))
            except Exception:
                capture_controls += 1
            else:
                raise ValueError(f"fixture {name} corruption accepted")
            (root / "manifest.txt").write_text(manifest_text, encoding="utf-8")
        (root / "frame00-w0.bin").write_bytes(b"\x00" * 4096)
        try:
            Capture(root, (0,))
        except Exception:
            capture_controls += 1
        else:
            raise ValueError("fixture window content corruption accepted")

    return dict(abi_fields=len(offsets), geometry_known_answers=3,
                diff_known_answers=3, extent_known_answers=3,
                malformed_controls=4, verdict_controls=verdict_bad,
                capture_controls=capture_controls, gpu_kernels_launched=0)


# ---------------------------------------------------------------------------
# GPU replay
# ---------------------------------------------------------------------------

def upload_region(driver: Any, alloc: tuple[int, int, int], content: bytes,
                  fill: int, guard_words: list[int]) -> bytes:
    """Upload content at the region start, pattern beyond; return the host image."""
    base_address, _region, size = alloc
    base.require(len(content) <= size, "upload: content exceeds region")
    host = (ct.c_uint8 * (size + 2 * GUARD_WORDS * 4))()
    ct.memset(host, fill, ct.sizeof(host))
    if content:
        ct.memmove(ct.byref(host, GUARD_WORDS * 4), content, len(content))
    for index, word in enumerate(guard_words):
        ct.cast(ct.byref(host, index * 4), ct.POINTER(ct.c_uint32)).contents.value = word
        ct.cast(ct.byref(host, GUARD_WORDS * 4 + size + index * 4),
                ct.POINTER(ct.c_uint32)).contents.value = word
    driver.check("cuMemcpyHtoD_v2", ct.c_uint64(base_address), host, ct.sizeof(host))
    return bytes(host)


def readback_region(driver: Any, alloc: tuple[int, int, int]) -> bytes:
    base_address, _region, size = alloc
    host = (ct.c_uint8 * (size + 2 * GUARD_WORDS * 4))()
    driver.check("cuMemcpyDtoH_v2", host, ct.c_uint64(base_address), ct.sizeof(host))
    return bytes(host)


def region_slice(image: bytes, alloc: tuple[int, int, int]) -> bytes:
    return image[GUARD_WORDS * 4:GUARD_WORDS * 4 + alloc[2]]


def guards_intact(image: bytes, guard_words: list[int]) -> bool:
    head = list(struct.unpack(f"<{GUARD_WORDS}I", image[:GUARD_WORDS * 4]))
    tail = list(struct.unpack(f"<{GUARD_WORDS}I", image[-GUARD_WORDS * 4:]))
    return head == guard_words and tail == guard_words


def run_entry(driver: Any, reset_function: ct.c_void_p, decoder_function: ct.c_void_p,
              capture: Capture,
              entry_id: str, frame: int, kernel_name: str, partials_kind: str,
              weights_rotate: bool, launch_decoder: bool,
              guard_words: list[int], evidence: Path) -> dict[str, Any]:
    main_window = capture.window(frame, 0)
    skip_window = capture.window(frame, 1)
    out0_window = capture.window(frame, 2)
    counter_window = capture.window(frame, 4)
    partials_window = capture.window(frame, 6)
    weights_window = capture.window(frame, 7)
    base.require(len(main_window) == MAIN_BYTES, "entry: main window size")
    base.require(len(skip_window) == len(out0_window) == len(partials_window)
                 == len(weights_window) == WINDOW, "entry: window size")
    if weights_rotate:
        matrix = weights_window[:MATRIX_BYTES]
        weights_window = (matrix[ROTATE_BYTES:] + matrix[:ROTATE_BYTES]
                          + weights_window[MATRIX_BYTES:])
    partials_content = partials_window if partials_kind == "capture" else b""

    launches = readbacks = 0
    with ExitStack() as allocations:
        allocs = {
            "main": launch.allocate(driver, allocations, MAIN_BYTES),
            "skip": launch.allocate(driver, allocations, LARGE_BYTES),
            "weights": launch.allocate(driver, allocations, LARGE_BYTES),
            "partials": launch.allocate(driver, allocations, LARGE_BYTES),
            "out0": launch.allocate(driver, allocations, LARGE_BYTES),
            "out3": launch.allocate(driver, allocations, UNUSED_BYTES),
            "out2": launch.allocate(driver, allocations, UNUSED_BYTES),
            "counters": launch.allocate(driver, allocations, COUNTER_BYTES),
        }
        initial = {
            "main": upload_region(driver, allocs["main"], main_window,
                                  UNUSED_PATTERN, guard_words),
            "skip": upload_region(driver, allocs["skip"], skip_window,
                                  SKIP_PATTERN, guard_words),
            "weights": upload_region(driver, allocs["weights"], weights_window,
                                     WEIGHTS_PATTERN, guard_words),
            "partials": upload_region(driver, allocs["partials"], partials_content,
                                      ZERO_PATTERN, guard_words),
            "out0": upload_region(driver, allocs["out0"], b"",
                                  OUT0_PATTERN, guard_words),
            "out3": upload_region(driver, allocs["out3"], b"",
                                  UNUSED_PATTERN, guard_words),
            "out2": upload_region(driver, allocs["out2"], b"",
                                  UNUSED_PATTERN, guard_words),
            "counters": upload_region(driver, allocs["counters"], b"",
                                      ZERO_PATTERN, guard_words),
        }

        # Counter reset through the recovered contract at the vendor's live count.
        reset_args = utility.CounterArguments(allocs["counters"][1], CLEAR_COUNT)
        reset_params = (ct.c_void_p * 1)(ct.addressof(reset_args))
        grid = launch.reset_geometry(CLEAR_COUNT)
        base.require(grid == (159, 1, 1), "entry: clear geometry")
        driver.check("cuLaunchKernel", reset_function, *grid, 256, 1, 1,
                     0, None, reset_params, None)
        launches += 1
        driver.check("cuCtxSynchronize")

        if launch_decoder:
            dim, width, axis0, axis1 = struct.unpack("<4I", capture.scalar_bytes(frame))
            arguments = launch.DecoderArguments(
                main=allocs["main"][1], skip=allocs["skip"][1],
                out0=allocs["out0"][1], out3=allocs["out3"][1],
                counters=allocs["counters"][1], out2=allocs["out2"][1],
                partials=allocs["partials"][1], weights=allocs["weights"][1],
                dimension=dim, width=width, output_axis0=axis0, output_axis1=axis1)
            kernel_params = (ct.c_void_p * 1)(ct.addressof(arguments))
            driver.check("cuLaunchKernel", decoder_function, *GRID, *BLOCK,
                         0, None, kernel_params, None)
            launches += 1
            driver.check("cuCtxSynchronize")

        current = {key: readback_region(driver, alloc) for key, alloc in allocs.items()}
        readbacks += len(allocs)

    # Retain the raw readback windows before computing verdicts: a failing
    # comparison must not discard the evidence that explains it.
    out0_region = region_slice(current["out0"], allocs["out0"])
    partials_region = region_slice(current["partials"], allocs["partials"])
    counters_region = region_slice(current["counters"], allocs["counters"])
    retained = {"out0": f"{entry_id}-out0.bin", "partials": f"{entry_id}-partials.bin",
                "counters": f"{entry_id}-counters.bin"}
    (evidence / retained["out0"]).write_bytes(out0_region[:OUTPUT_BYTES])
    (evidence / retained["partials"]).write_bytes(partials_region[:WINDOW])
    (evidence / retained["counters"]).write_bytes(counters_region[:COUNTER_SCOPE])

    guards_ok = all(guards_intact(current[key], guard_words) for key in allocs)
    immutable = {key: region_slice(current[key], allocs[key])
                 == region_slice(initial[key], allocs[key])
                 for key in ("main", "skip", "weights", "out3", "out2")}

    out0_initial = region_slice(initial["out0"], allocs["out0"])
    extent = change_extent(out0_initial, out0_region)
    beyond_clean = out0_region[OUTPUT_BYTES:] == out0_initial[OUTPUT_BYTES:]
    vendor_out0 = out0_window[:OUTPUT_BYTES]
    out0_match, out0_diffs, out0_first = diff_stats(vendor_out0, out0_region[:OUTPUT_BYTES])

    partials_initial = region_slice(initial["partials"], allocs["partials"])
    # Extent is recorded within the retained window (the oracle-recomputable
    # scope); writes beyond it are a separately recorded finding.
    partials_extent = change_extent(partials_initial[:WINDOW], partials_region[:WINDOW])
    beyond_window_change = partials_initial[WINDOW:] != partials_region[WINDOW:]
    partials_match, partials_diffs, partials_first = diff_stats(
        partials_window, partials_region[:WINDOW])
    write_extent_end = partials_extent[1] + 1 if partials_extent[1] >= 0 else 0
    write_scope = min(write_extent_end, WINDOW) if write_extent_end else WINDOW
    scope_match, scope_diffs, scope_first = diff_stats(
        partials_window[:write_scope], partials_region[:write_scope])

    counter_head = list(struct.unpack(f"<{COUNTER_SLOTS}i",
                                      counters_region[:COUNTER_SLOTS * 4]))
    counter_match, counter_diffs, counter_first = diff_stats(
        counter_window[:COUNTER_SCOPE], counters_region[:COUNTER_SCOPE])

    entry = {
        "entry": entry_id, "frame": frame, "kernel": kernel_name,
        "partials_prestate": partials_kind, "weights_rotated": weights_rotate,
        "decoder_launched": launch_decoder, "launches": launches, "readbacks": readbacks,
        "out0": {"match": out0_match, "diff_count": out0_diffs, "first_diff": out0_first,
                 "scope_bytes": OUTPUT_BYTES,
                 "change_first": extent[0], "change_last": extent[1],
                 "change_blocks": extent[2],
                 "extent_matches_rule": (extent[0], extent[1]) == (0, OUTPUT_BYTES - 1)
                 and beyond_clean,
                 "vendor_fnv": f"0x{report.fnv1a64(vendor_out0):016x}",
                 "native_fnv": f"0x{report.fnv1a64(out0_region[:OUTPUT_BYTES]):016x}"},
        "partials": {"window_match": partials_match, "window_diff_count": partials_diffs,
                     "window_first_diff": partials_first,
                     "change_first": partials_extent[0], "change_last": partials_extent[1],
                     "change_blocks": partials_extent[2],
                     "beyond_window_change": beyond_window_change,
                     "write_extent_end": write_extent_end,
                     "write_scope_bytes": write_scope,
                     "write_scope_match": scope_match,
                     "write_scope_diff_count": scope_diffs,
                     "write_scope_first_diff": scope_first},
        "counters": {"head": counter_head, "window_match": counter_match,
                     "window_diff_count": counter_diffs, "window_first_diff": counter_first,
                     "scope_bytes": COUNTER_SCOPE},
        "immutable": immutable,
        "out3_out2_extent": {
            key: dict(zip(("change_first", "change_last", "change_blocks"),
                          change_extent(region_slice(initial[key], allocs[key]),
                                        region_slice(current[key], allocs[key]))))
            for key in ("out3", "out2")},
        "guards_intact": guards_ok,
        "retained": retained,
    }
    validate_entry_verdicts(entry)
    return entry


def execute(data: bytes, inventory: dict[str, Any], mode: str,
            capture: Capture, evidence: Path, progress: dict[str, Any]) -> dict[str, Any]:
    entries_plan = PLANS[mode]
    modules = [e for e in inventory["fatbins"][6]["entries"]
               if e["kind"] == 2 and e["arch"] == 89]
    base.require(len(modules) == 1, "unique SM89 decoder module")
    image = base.decompress(data, modules[0])
    utility.verify_kernel_abi(image, RESET, 16)
    for name in (NAME_LIVE, NAME_RECOVERED):
        abi = decoder.parameter_contract(launch.kernel_metadata(image, name))
        base.require(abi["bytes"] == 80, "decoder ABI size")
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
        for key, name in (("reset", RESET), ("tilesync", NAME_LIVE),
                          ("fp8", NAME_RECOVERED)):
            function = ct.c_void_p()
            driver.check("cuModuleGetFunction", ct.byref(function), module, name.encode())
            functions[key] = function
        for entry_id in entries_plan:
            tag, frame_text, partials_tag = entry_id.split("-")
            frame = int(frame_text[1:])
            kernel_name = dict(KERNEL_TAGS)[tag]
            partials_kind = dict(PARTIALS_MODES)[partials_tag]
            entries.append(run_entry(
                driver, functions["reset"], functions[tag], capture, entry_id, frame,
                kernel_name,
                partials_kind, weights_rotate=(mode == "wrong"),
                launch_decoder=(mode != "skip"),
                guard_words=guard_words, evidence=evidence))
            progress["entries"] = entries

    # Enforced outcomes (integrity and the primary claim). Findings are recorded
    # per entry above and carry no pass/fail semantics.
    failures = [f"guards:{e['entry']}" for e in entries if not e["guards_intact"]]
    failures += [f"immutable:{e['entry']}:{key}" for e in entries
                 for key, ok in e["immutable"].items()
                 if not ok and (key in ("main", "skip", "weights") or
                                e["kernel"] == NAME_RECOVERED)]
    decoder_entries = [e for e in entries if e["decoder_launched"]]
    failures += [f"counters:{e['entry']}" for e in decoder_entries
                 if e["counters"]["head"] != [3] * COUNTER_SLOTS]
    failures += [f"counter-window:{e['entry']}" for e in decoder_entries
                 if not e["counters"]["window_match"]]
    failures += [f"extent:{e['entry']}" for e in decoder_entries
                 if not e["out0"]["extent_matches_rule"]]
    if mode == "execute":
        failures += [f"out0-mismatch:{e['entry']}" for e in decoder_entries
                     if not e["out0"]["match"]]
    elif mode == "wrong":
        failures += [f"wrong-matched:{e['entry']}" for e in entries if e["out0"]["match"]]
    elif mode == "skip":
        failures += [f"skip-matched:{e['entry']}" for e in entries if e["out0"]["match"]]
        failures += [f"skip-counters:{e['entry']}" for e in entries
                     if e["counters"]["head"] != [-1] * COUNTER_SLOTS]

    return dict(
        native=True, device="NVIDIA GeForce RTX 4070 Ti SUPER",
        driver_api_version=version, mode=mode,
        entries_plan=list(entries_plan), entries=entries,
        kernels_launched=sum(e["launches"] for e in entries),
        readbacks=sum(e["readbacks"] for e in entries),
        enforced_failures=failures,
        checks_enforced=["guards_intact", "immutable(main,skip,weights)",
                         "immutable(out3,out2) on recovered variant",
                         "counter_head=[3]*24 on launched entries",
                         "counter_window[0,512) on launched entries",
                         "out0 extent exactly [0,344064) on launched entries",
                         "out0 match on all launched entries (execute)",
                         "out0 non-match and full non-publication [−1]×24 (skip)",
                         "out0 non-match (wrong)"],
        findings_recorded=["partial-sums change extent, window and write-scope match",
                           "out3/out2 change extents per variant"],
        partials_prestate_unrecovered=True, out0_prestate_unrecovered=True,
        skip_window_out_of_scope_reads_possible=True,
        learned_weights_used=True, neural_image_produced=False,
        complete_native_pipeline=False,
        canaries_do_not_detect_read_only_out_of_bounds=True,
        deliberate_proprietary_payload_files_written=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("self-test", "execute", "skip", "wrong"))
    parser.add_argument("dll", type=Path, nargs="?")
    parser.add_argument("capture", type=Path, nargs="?")
    args = parser.parse_args()
    if args.mode == "self-test":
        print(json.dumps(self_test(), indent=2))
        return
    base.require(args.dll is not None and args.capture is not None,
                 "DLL and capture directory required")
    policy = base.protect_process_memory()
    policy["disassembler_same_uid_proc_exposure_controlled"] = False
    data = base.read_bounded(args.dll)
    inventory = base.inventory(data)
    frames = FRAMES if args.mode == "execute" else (0,)
    capture = Capture(args.capture, frames)
    progress: dict[str, Any] = {"entries": []}
    result = dict(dll_sha256=base.PIN, memory_policy=policy,
                  capture_digests=capture.digest_summary())
    try:
        result["cuda"] = execute(data, inventory, args.mode, capture, Path.cwd(), progress)
        result["cpu_self_test"] = self_test()
    except Exception as error:
        result["rejected"] = f"{type(error).__name__}: {error}"
        result["entries_so_far"] = progress["entries"]
        print(json.dumps(result, indent=2))
        raise SystemExit(1)
    print(json.dumps(result, indent=2))
    if result["cuda"]["enforced_failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
