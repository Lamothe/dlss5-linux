#!/usr/bin/env python3
"""Output-position/channel naming for the decoder influence footprint.

Reads retained evidence from the native-decoder-weights-probe.py execute runs
(which retain the full unit-offset list) and:

1. Extends the influence-footprint constraint lock to the full 512-unit set:
   every coupled/negative footprint in every group must reproduce the canonical
   unit-offset set measured in the two agreed runs.
2. Names the influenced output positions and channels under the declared
   contiguous-channel layout hypothesis `offset = (y*16 + x)*512 + c` for the
   16x16x512 output extent, alongside a hypothesis-independent physical
   residue description (row terms, column terms, channel residues).

What it does NOT do: run the GPU, launch a kernel, prove the semantic tensor
layout, recover the (o, i) matrix permutation, bind stored learned weights, or
produce a neural-rendered image. The naming is conditional on the declared
layout hypothesis.

Usage:
    python3 vr/native-decoder-influence-names.py <runs-dir> [<runs-dir> ...]
    python3 vr/native-decoder-influence-names.py --self-test
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


CANONICAL_KERNEL = "cc_dec_input_upsample_1024_512_fp8"

# Tested output contract: 8x8 input upsampled to 16x16 with 512 output
# channels (kernel name cc_dec_input_upsample_1024_512_fp8).
OUTPUT_ROWS = 16
OUTPUT_COLS = 16
OUTPUT_CHANNELS = 512
EXTENT_BYTES = OUTPUT_ROWS * OUTPUT_COLS * OUTPUT_CHANNELS
UNIT_BYTES = 16

# Canonical influence footprint, taken from the two agreed execute runs
# (decoder-weights-execute-IUvcUu7c, -JvrNtvNL): 512 sixteen-byte units, each
# with 8 changed bytes, spanning [0, 40827].
CANONICAL_UNIT_BYTE_OFFSETS = (0, 1, 2, 3, 8, 9, 10, 11)
CANONICAL_SAMPLE_OFFSETS = (0, 1, 2, 3, 8, 9, 10, 11)
CANONICAL_COUPLED_VALUES = (8.0,) * 8
CANONICAL_NEGATIVE_VALUES = (-8.0,) * 8
CANONICAL_CHANGED_COUNT = 4096
CANONICAL_UNITS16 = 512
CANONICAL_FIRST = 0
CANONICAL_LAST = 40827

# The measured unit-offset set factorises exactly as
#   offset = row_term*8192 + column*512 + channel_block*16
# with the parameters below (2*16*16 = 512 units). This is a lossless encoding
# of the measured offsets, not a claim about tensor semantics.
CANONICAL_ROW_TERMS = (0, 4)
CANONICAL_COLUMNS = tuple(range(16))
CANONICAL_CHANNEL_BLOCKS = (0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23)

CANONICAL_VARIANTS = ("f32-one", "zeros")
CANONICAL_PARTITIONS = (0, 1, 2, 3)

NO_OVERCLAIM_FLAGS = (
    "learned_weights_used",
    "matrix_permutation_recovered",
    "native_image_inference",
)


class NamingError(Exception):
    """Evidence or input violates the canonical influence-naming contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NamingError(message)


def _field(mapping: Any, key: str, kind: type, path: str) -> Any:
    """Fetch a required field, raising NamingError (not KeyError/TypeError) on bad evidence."""
    _require(isinstance(mapping, dict), f"{path}: expected an object, got {type(mapping).__name__}")
    _require(key in mapping, f"{path}: missing required field {key!r}")
    value = mapping[key]
    if kind is int:
        _require(type(value) is int, f"{path}.{key}: expected an int, got {value!r}")
    else:
        _require(isinstance(value, kind), f"{path}.{key}: expected {kind.__name__}, "
                                          f"got {type(value).__name__}")
    return value


def canonical_units() -> tuple[int, ...]:
    """The measured 512-unit offset set, from its factorised encoding."""
    units = tuple(sorted({row * 8192 + column * 512 + block * UNIT_BYTES
                          for row in CANONICAL_ROW_TERMS
                          for column in CANONICAL_COLUMNS
                          for block in CANONICAL_CHANNEL_BLOCKS}))
    _require(len(units) == CANONICAL_UNITS16, "canonical unit count")
    _require(len(set(units)) == CANONICAL_UNITS16, "canonical units not distinct")
    _require(units[0] == CANONICAL_FIRST and units[-1] == CANONICAL_LAST - max(CANONICAL_UNIT_BYTE_OFFSETS),
             "canonical unit span")
    return units


def _checked_offset(offset: Any) -> int:
    _require(type(offset) is int, f"offset must be an int, got {offset!r}")
    _require(0 <= offset < EXTENT_BYTES, f"offset {offset} outside [0, {EXTENT_BYTES})")
    return offset


def contiguous_name(offset: int) -> tuple[int, int, int]:
    """(y, x, c) under the contiguous-channel hypothesis offset = (y*16 + x)*512 + c."""
    value = _checked_offset(offset)
    position, channel = divmod(value, OUTPUT_CHANNELS)
    return position // OUTPUT_COLS, position % OUTPUT_COLS, channel


def channel_major_name(offset: int) -> tuple[int, int, int]:
    """(c, y, x) under the channel-major alternative offset = c*256 + y*16 + x."""
    value = _checked_offset(offset)
    channel, yx = divmod(value, OUTPUT_ROWS * OUTPUT_COLS)
    return channel, yx // OUTPUT_COLS, yx % OUTPUT_COLS


def influenced_channels(offset: int) -> tuple[int, ...]:
    """Channels touched inside one unit's named contiguous-channel block.

    Requires a unit-start offset (16-byte aligned) whose named block lies
    inside the 512-channel extent.
    """
    value = _checked_offset(offset)
    _require(value % UNIT_BYTES == 0, f"offset {value} is not a unit start")
    _, _, channel0 = contiguous_name(value)
    _require(channel0 + max(CANONICAL_UNIT_BYTE_OFFSETS) < OUTPUT_CHANNELS,
             f"unit at offset {value} extends past the channel extent")
    return tuple(channel0 + byte for byte in CANONICAL_UNIT_BYTE_OFFSETS)


def _block_ranges(blocks: list[int]) -> list[str]:
    """Human-readable channel ranges for runs of consecutive 16-channel blocks."""
    _require(blocks and all(type(block) is int for block in blocks), "block list")
    ranges: list[str] = []
    start = previous = blocks[0]
    for block in blocks[1:] + [None]:  # type: ignore[list-item]
        if block is not None and block == previous + 1:
            previous = block
            continue
        ranges.append(f"{start * UNIT_BYTES}-{(previous + 1) * UNIT_BYTES - 1} "
                      f"(blocks {start}-{previous})")
        if block is not None:
            start = previous = block
    return ranges


def naming_report(unit_offsets: tuple[int, ...]) -> dict[str, Any]:
    """Derive the named structure of the canonical unit set under both views."""
    _require(tuple(sorted(unit_offsets)) == canonical_units(), "report units not canonical")
    positions = sorted({contiguous_name(unit)[:2] for unit in unit_offsets})
    rows = sorted({position[0] for position in positions})
    columns = sorted({position[1] for position in positions})
    blocks = sorted({contiguous_name(unit)[2] // UNIT_BYTES for unit in unit_offsets})
    channels = sorted({contiguous_name(unit)[2] for unit in unit_offsets})
    factorises = (len(positions) == len(rows) * len(columns)
                  and len(unit_offsets) == len(positions) * len(blocks))
    major_channels = sorted({channel_major_name(unit)[0] for unit in unit_offsets})
    major_positions = {channel_major_name(unit)[0]: set() for unit in unit_offsets}
    for unit in unit_offsets:
        major_positions[channel_major_name(unit)[0]].add(channel_major_name(unit)[1:])
    per_channel_identical = len({tuple(sorted(v)) for v in major_positions.values()}) == 1
    return dict(
        layout_hypothesis="offset = (y*16 + x)*512 + c (channels contiguous, 16x16x512)",
        semantic_layout_proven=False,
        named_positions=[list(position) for position in positions],
        named_rows=rows,
        named_columns=columns,
        named_channel_blocks=blocks,
        named_channel_residues=channels,
        named_channel_ranges=_block_ranges(blocks),
        influenced_channels_per_position=len(blocks) * len(CANONICAL_UNIT_BYTE_OFFSETS),
        influenced_pairs=len(unit_offsets) * len(CANONICAL_UNIT_BYTE_OFFSETS),
        within_unit_byte_offsets=list(CANONICAL_UNIT_BYTE_OFFSETS),
        factorises_as_positions_times_blocks=factorises,
        physical_row_terms=sorted({unit // 8192 for unit in unit_offsets}),
        physical_column_terms=sorted({(unit % 8192) // 512 for unit in unit_offsets}),
        physical_channel_residues=sorted({unit % 512 for unit in unit_offsets}),
        channel_major_alternative=dict(
            distinct_channels=major_channels,
            positions_per_channel=len(next(iter(major_positions.values()))),
            per_channel_position_sets_identical=per_channel_identical,
        ),
    )


def _check_influence(label: str, influence: Any, values: tuple[float, ...]) -> None:
    """Enforce one measured influence footprint against the canonical unit set."""
    _require(_field(influence, "changed", bool, label) is True,
             f"{label}: changed flag is not True")
    _require(_field(influence, "truncated", bool, label) is False,
             f"{label}: footprint was truncated; cannot enforce the full span")
    _require(_field(influence, "unit_offsets_truncated", bool, label) is False,
             f"{label}: unit list was truncated; cannot enforce the full set")
    _require(_field(influence, "uniform_unit_pattern", bool, label) is True,
             f"{label}: unit byte pattern is not uniform")
    byte_offsets = _field(influence, "unit_byte_offsets", list, label)
    _require(byte_offsets == list(CANONICAL_UNIT_BYTE_OFFSETS),
             f"{label}: unit_byte_offsets {byte_offsets} != {list(CANONICAL_UNIT_BYTE_OFFSETS)}")
    sample_offsets = _field(influence, "sample_offsets", list, label)
    _require(sample_offsets == list(CANONICAL_SAMPLE_OFFSETS),
             f"{label}: sample_offsets {sample_offsets} != {list(CANONICAL_SAMPLE_OFFSETS)}")
    observed_values = _field(influence, "sample_values", list, label)
    _require(observed_values == list(values),
             f"{label}: sample_values {observed_values} != {list(values)}")
    units = _field(influence, "unit_offsets", list, label)
    _require(all(type(unit) is int for unit in units),
             f"{label}: unit_offsets elements must all be ints")
    _require(units == list(canonical_units()),
             f"{label}: unit_offsets do not match the canonical 512-unit set")
    _require(_field(influence, "changed_units16", int, label) == CANONICAL_UNITS16,
             f"{label}: changed_units16 {influence['changed_units16']} != {CANONICAL_UNITS16}")
    _require(_field(influence, "changed_count", int, label) == CANONICAL_CHANGED_COUNT,
             f"{label}: changed_count {influence['changed_count']} != {CANONICAL_CHANGED_COUNT}")
    _require(_field(influence, "changed_first", int, label) == CANONICAL_FIRST,
             f"{label}: changed_first {influence['changed_first']} != {CANONICAL_FIRST}")
    _require(_field(influence, "changed_last", int, label) == CANONICAL_LAST,
             f"{label}: changed_last {influence['changed_last']} != {CANONICAL_LAST}")


def check_run(payload: dict[str, Any]) -> dict[str, Any]:
    """Enforce one execute-run payload; return the derived naming summary."""
    cuda = _field(payload, "cuda", dict, "root")
    _require(_field(cuda, "mode", str, "cuda") == "execute", "not an execute run")
    _require(_field(cuda, "kernel", str, "cuda") == CANONICAL_KERNEL,
             f"unexpected kernel {cuda.get('kernel')!r}; anchors are kernel-specific")
    for flag in NO_OVERCLAIM_FLAGS:
        _require(_field(cuda, flag, bool, "cuda") is False,
                 f"evidence claims {flag}; naming tool does not")

    groups = _field(cuda, "groups", list, "cuda")
    _require(len(groups) == 8, f"expected 8 experiment groups, found {len(groups)}")
    seen: set[tuple[str, int]] = set()
    for index, group in enumerate(groups):
        label = f"groups[{index}]"
        variant = _field(group, "variant", str, label)
        partition = _field(group, "impulse_partition", int, label)
        key = (variant, partition)
        _require(key not in seen, f"duplicate group {key}")
        seen.add(key)
        _require(variant in CANONICAL_VARIANTS, f"unknown variant {variant}")
        _require(partition in CANONICAL_PARTITIONS, f"unknown partition {partition}")
        _check_influence(f"{label}/coupled", _field(group, "coupled_influence", dict, label),
                         CANONICAL_COUPLED_VALUES)
        _check_influence(f"{label}/negative", _field(group, "negative_influence", dict, label),
                         CANONICAL_NEGATIVE_VALUES)

    expected = {(variant, partition) for variant in CANONICAL_VARIANTS
                for partition in CANONICAL_PARTITIONS}
    _require(seen == expected, f"missing groups: {sorted(expected - seen)}")

    report = naming_report(canonical_units())
    report.update(kernel=CANONICAL_KERNEL, groups=len(groups),
                  anchors_per_group=2, anchors_total=len(groups) * 2)
    return report


def self_test() -> dict[str, Any]:
    """Known answers for the naming functions plus corruption controls on evidence."""
    known = 0
    controls = 0

    def expect_reject(mutate: Any, label: str) -> None:
        nonlocal controls
        payload = _fixture()
        mutate(payload)
        try:
            check_run(payload)
        except NamingError:
            controls += 1
            return
        raise NamingError(f"self-test: corruption not rejected ({label})")

    def set_influence(field: str, value: Any, which: str = "coupled_influence") -> Any:
        def mutate(payload: dict[str, Any]) -> None:
            payload["cuda"]["groups"][0][which][field] = value
        return mutate

    # Baseline fixture must pass.
    check_run(_fixture())

    # Naming-function known answers (contiguous-channel hypothesis).
    answers = {0: (0, 0, 0), 511: (0, 0, 511), 512: (0, 1, 0), 8192: (1, 0, 0),
               32768: (4, 0, 0), 40816: (4, 15, 368)}
    for offset, expected in answers.items():
        _require(contiguous_name(offset) == expected, f"contiguous known answer {offset}")
        known += 1
    _require(channel_major_name(0) == (0, 0, 0), "channel-major known answer 0")
    _require(channel_major_name(255) == (0, 15, 15), "channel-major known answer 255")
    _require(channel_major_name(256) == (1, 0, 0), "channel-major known answer 256")
    _require(influenced_channels(0) == CANONICAL_UNIT_BYTE_OFFSETS, "influenced channels 0")
    _require(influenced_channels(16) == (16, 17, 18, 19, 24, 25, 26, 27),
             "influenced channels 16")
    known += 5

    # Mapping inputs must reject non-int, bool, and out-of-range offsets.
    for bad in (-1, EXTENT_BYTES, 1.5, "0", True, None):
        try:
            contiguous_name(bad)  # type: ignore[arg-type]
        except NamingError:
            controls += 1
        else:
            raise NamingError(f"self-test: bad offset accepted ({bad!r})")
    # influenced_channels additionally requires unit-start alignment.
    for bad in (5, 511):
        try:
            influenced_channels(bad)
        except NamingError:
            controls += 1
        else:
            raise NamingError(f"self-test: non-unit-start offset accepted ({bad!r})")

    # Report structure known answers on the canonical set.
    report = naming_report(canonical_units())
    _require(report["named_rows"] == [0, 4], "report rows known answer")
    _require(len(report["named_positions"]) == 32, "report position count")
    _require(report["named_channel_blocks"] == list(CANONICAL_CHANNEL_BLOCKS),
             "report channel blocks known answer")
    _require(report["influenced_pairs"] == CANONICAL_CHANGED_COUNT, "report pair count")
    _require(report["factorises_as_positions_times_blocks"] is True, "report factorisation")
    _require(report["semantic_layout_proven"] is False, "report overclaims layout")
    known += 1

    # Canonical-set self-consistency.
    units = canonical_units()
    _require(units == tuple(sorted(set(units))), "canonical units sorted and distinct")
    _require(len(units) == 512 and units[0] == 0 and units[-1] == 40816, "canonical span")
    known += 1

    # Corruption controls over evidence.
    expect_reject(lambda p: p["cuda"]["groups"][0]["coupled_influence"]["unit_offsets"]
                  .__setitem__(0, 16), "shifted unit offset")
    expect_reject(lambda p: p["cuda"]["groups"][0]["coupled_influence"]["unit_offsets"]
                  .__setitem__(0, 1), "unaligned unit offset")
    expect_reject(lambda p: p["cuda"]["groups"][0]["coupled_influence"]["unit_offsets"].pop(),
                  "dropped unit offset")
    expect_reject(lambda p: p["cuda"]["groups"][0]["coupled_influence"]["unit_offsets"]
                  .append(0), "duplicate unit offset")
    expect_reject(set_influence("unit_offsets", []), "empty unit list")
    expect_reject(set_influence("unit_offsets", [0.0] * 512), "non-int unit list")
    # Value-correct but type-corrupted lists must be rejected element-wise
    # (Python list equality would otherwise accept floats and False == 0).
    expect_reject(set_influence("unit_offsets", [float(unit) for unit in canonical_units()]),
                  "value-correct float unit list")
    expect_reject(set_influence("unit_offsets",
                                [False] + list(canonical_units())[1:]),
                  "bool in unit list")
    expect_reject(set_influence("unit_offsets_truncated", True), "unit truncation flag")
    expect_reject(set_influence("uniform_unit_pattern", False), "uniformity flag")
    expect_reject(set_influence("truncated", True), "footprint truncation flag")
    expect_reject(set_influence("unit_byte_offsets", [0] * 8), "byte offset pattern")
    expect_reject(set_influence("sample_offsets", [0] * 8), "sample offsets")
    expect_reject(set_influence("sample_values", [0.0] * 8), "sample values")
    expect_reject(set_influence("sample_values", [-8.0] * 8), "coupled sign flip")
    expect_reject(set_influence("sample_values", [8.0] * 8, "negative_influence"),
                  "negative sign flip")
    expect_reject(set_influence("changed_units16", CANONICAL_UNITS16 - 1), "unit count")
    expect_reject(set_influence("changed_count", CANONICAL_CHANGED_COUNT + 1), "changed count")
    expect_reject(set_influence("changed_first", 1), "changed first")
    expect_reject(set_influence("changed_last", CANONICAL_LAST - 1), "changed last")
    expect_reject(set_influence("changed", False), "changed flag")
    expect_reject(set_influence("unit_offsets", "not-a-list", "negative_influence"),
                  "negative non-list units")
    expect_reject(lambda p: p["cuda"]["groups"][0]["coupled_influence"].pop("unit_offsets", None),
                  "missing unit_offsets")
    expect_reject(lambda p: p["cuda"]["groups"][0].__setitem__("coupled_influence", "bad"),
                  "non-object influence")
    expect_reject(lambda p: p["cuda"].__setitem__("kernel", "other_kernel"), "wrong kernel")
    expect_reject(lambda p: p["cuda"].__setitem__("mode", "skip"), "non-execute mode")
    for flag in NO_OVERCLAIM_FLAGS:
        expect_reject(lambda p, f=flag: p["cuda"].__setitem__(f, True), f"{flag} claim")
    expect_reject(lambda p: p["cuda"]["groups"].__setitem__(slice(0, 1), []), "missing group")
    expect_reject(lambda p: p["cuda"]["groups"][1].__setitem__("impulse_partition", 0),
                  "duplicate group key")
    expect_reject(lambda p: p["cuda"]["groups"][0].__setitem__("variant", "unknown-variant"),
                  "unknown variant")
    expect_reject(lambda p: p["cuda"]["groups"][0].__setitem__("impulse_partition", 9),
                  "unknown partition")
    expect_reject(lambda p: p["cuda"].pop("groups", None), "missing groups")

    return dict(self_test="passed", known_answers=known,
                corruption_controls=controls, baseline_accepted=True)


def _fixture() -> dict[str, Any]:
    """Canonical execute-run payload matching the agreed evidence."""
    def influence(values: tuple[float, ...]) -> dict[str, Any]:
        return dict(
            changed=True,
            changed_count=CANONICAL_CHANGED_COUNT,
            changed_first=CANONICAL_FIRST,
            changed_last=CANONICAL_LAST,
            changed_units16=CANONICAL_UNITS16,
            sample_offsets=list(CANONICAL_SAMPLE_OFFSETS),
            sample_values=list(values),
            truncated=False,
            unit_offsets=list(canonical_units()),
            unit_byte_offsets=list(CANONICAL_UNIT_BYTE_OFFSETS),
            uniform_unit_pattern=True,
            unit_offsets_truncated=False,
        )

    groups = []
    for variant in CANONICAL_VARIANTS:
        for partition in CANONICAL_PARTITIONS:
            groups.append(dict(
                variant=variant,
                impulse_partition=partition,
                base_zero_in_extent=True,
                coupled_differs=True,
                negative_differs=True,
                decoupled_identical=True,
                decoupled_checked=[f"decoupled-q{q}" for q in CANONICAL_PARTITIONS
                                   if q != partition],
                coupled_influence=influence(CANONICAL_COUPLED_VALUES),
                negative_influence=influence(CANONICAL_NEGATIVE_VALUES),
            ))
    return dict(cuda=dict(
        mode="execute",
        kernel=CANONICAL_KERNEL,
        learned_weights_used=False,
        matrix_permutation_recovered=False,
        native_image_inference=False,
        groups=groups,
    ))


def main(argv: list[str]) -> int:
    if any(arg == "--self-test" for arg in argv):
        if len(argv) != 1:
            print("names: --self-test takes no other arguments", file=sys.stderr)
            return 2
        print(json.dumps(self_test(), indent=2, sort_keys=True))
        return 0
    if any(arg.startswith("--") for arg in argv):
        print("names: unknown flag", file=sys.stderr)
        return 2
    if not argv:
        print(__doc__.strip(), file=sys.stderr)
        return 2

    reports = []
    for arg in argv:
        run_dir = Path(arg)
        output = run_dir / "output.json"
        if not output.is_file():
            print(f"names: no output.json under {run_dir}", file=sys.stderr)
            return 1
        try:
            payload = json.loads(output.read_text())
        except (OSError, json.JSONDecodeError) as error:
            print(f"names: cannot read {output}: {error}", file=sys.stderr)
            return 1
        try:
            summary = check_run(payload)
        except NamingError as error:
            print(f"names: {run_dir.name}: {error}", file=sys.stderr)
            return 1
        summary["run"] = run_dir.name
        reports.append(summary)
        print(json.dumps(summary, sort_keys=True))

    print(json.dumps(dict(names="passed", runs=reports), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
