#!/usr/bin/env python3
"""Influence-footprint constraint oracle for the decoder weight probe.

Reads retained evidence from the native-decoder-weights-probe.py runs and
enforces the invariants any recovered output-channel naming must reproduce.

This is a *constraint lock*, not a permutation recovery. It takes the
measured coupled/negative influence footprints already on disk and asserts
they match a single canonical pattern across every impulse partition and
word-table variant. A future naming stage that claims to explain the output
addressing must be checked against these same anchors.

What it does NOT do: run the GPU, launch a kernel, recover the (o, i)
matrix permutation, bind stored learned weights, or produce a
neural-rendered image.

Usage:
    python3 vr/native-decoder-influence-oracle.py <runs-dir> [<runs-dir> ...]
    python3 vr/native-decoder-influence-oracle.py --self-test
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


# Canonical influence footprint, taken from the two agreed execute runs.
# changed_count is the number of bytes that differ from the base experiment's
# output; changed_units16 counts touched 16-byte units; sample_offsets are the
# first eight changed byte offsets in absolute region coordinates.
CANONICAL_KERNEL = "cc_dec_input_upsample_1024_512_fp8"
CANONICAL_COUNT = 4096
CANONICAL_UNITS16 = 512
CANONICAL_FIRST = 0
CANONICAL_LAST = 40827
CANONICAL_SAMPLE_OFFSETS = [0, 1, 2, 3, 8, 9, 10, 11]
CANONICAL_COUPLED_VALUES = [8.0] * 8
CANONICAL_NEGATIVE_VALUES = [-8.0] * 8

# The eight experiment groups cover four impulse partitions under two
# word-table variants (f32-one, zeros).
CANONICAL_VARIANTS = ("f32-one", "zeros")
CANONICAL_PARTITIONS = (0, 1, 2, 3)

# Group-level verdicts required for the coupling model to hold.
REQUIRED_GROUP_FLAGS = (
    "base_zero_in_extent",
    "coupled_differs",
    "decoupled_identical",
    "negative_differs",
)


class OracleError(Exception):
    """A measured footprint violates the canonical influence contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OracleError(message)


def _field(mapping: Any, key: str, kind: type, path: str) -> Any:
    """Fetch a required field, raising OracleError (not KeyError/TypeError) on bad evidence."""
    _require(isinstance(mapping, dict), f"{path}: expected an object, got {type(mapping).__name__}")
    _require(key in mapping, f"{path}: missing required field {key!r}")
    value = mapping[key]
    if kind is float:
        _require(
            isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{path}.{key}: expected a number, got {value!r}",
        )
    elif kind is int:
        _require(
            type(value) is int,
            f"{path}.{key}: expected an int, got {value!r}",
        )
    else:
        _require(
            isinstance(value, kind),
            f"{path}.{key}: expected {kind.__name__}, got {type(value).__name__}",
        )
    return value


def _check_influence(label: str, influence: Any, values: list[float]) -> None:
    """Enforce one measured influence footprint against the canonical pattern."""
    _require(
        _field(influence, "changed", bool, label) is True,
        f"{label}: changed flag is not True",
    )
    _require(
        _field(influence, "changed_count", int, label) == CANONICAL_COUNT,
        f"{label}: changed_count {influence['changed_count']} != {CANONICAL_COUNT}",
    )
    _require(
        _field(influence, "changed_units16", int, label) == CANONICAL_UNITS16,
        f"{label}: changed_units16 {influence['changed_units16']} != {CANONICAL_UNITS16}",
    )
    _require(
        _field(influence, "changed_first", int, label) == CANONICAL_FIRST,
        f"{label}: changed_first {influence['changed_first']} != {CANONICAL_FIRST}",
    )
    _require(
        _field(influence, "changed_last", int, label) == CANONICAL_LAST,
        f"{label}: changed_last {influence['changed_last']} != {CANONICAL_LAST}",
    )
    offsets = _field(influence, "sample_offsets", list, label)
    _require(
        offsets == CANONICAL_SAMPLE_OFFSETS,
        f"{label}: sample_offsets {offsets} != {CANONICAL_SAMPLE_OFFSETS}",
    )
    observed_values = _field(influence, "sample_values", list, label)
    _require(
        observed_values == values,
        f"{label}: sample_values {observed_values} != {values}",
    )
    _require(
        _field(influence, "truncated", bool, label) is False,
        f"{label}: footprint was truncated; cannot enforce full span",
    )


def _check_group(label: str, group: Any) -> None:
    """Enforce one variant/partition group: flags plus coupled/negative footprints."""
    variant = _field(group, "variant", str, label)
    partition = _field(group, "impulse_partition", int, label)
    for flag in REQUIRED_GROUP_FLAGS:
        _require(
            _field(group, flag, bool, label) is True,
            f"{label}: group flag {flag} is not True",
        )
    expected_decoupled = [
        f"decoupled-q{q}" for q in CANONICAL_PARTITIONS if q != partition
    ]
    checked = _field(group, "decoupled_checked", list, label)
    for element in checked:
        _require(
            isinstance(element, str),
            f"{label}: decoupled_checked elements must be strings, got {element!r}",
        )
    _require(
        sorted(checked) == sorted(expected_decoupled),
        f"{label}: unexpected decoupled experiment set {checked} "
        f"(expected {expected_decoupled})",
    )
    _check_influence(
        f"{label}/coupled", _field(group, "coupled_influence", dict, label),
        CANONICAL_COUPLED_VALUES,
    )
    _check_influence(
        f"{label}/negative", _field(group, "negative_influence", dict, label),
        CANONICAL_NEGATIVE_VALUES,
    )


def check_run(payload: dict[str, Any]) -> dict[str, Any]:
    """Enforce one execute-run payload; return a compact summary of anchors checked."""
    cuda = _field(payload, "cuda", dict, "root")
    _require(
        _field(cuda, "mode", str, "cuda") == "execute",
        "not an execute run",
    )
    _require(
        _field(cuda, "kernel", str, "cuda") == CANONICAL_KERNEL,
        f"unexpected kernel {cuda.get('kernel')!r}; anchors are kernel-specific",
    )
    _require(
        _field(cuda, "matrix_permutation_recovered", bool, "cuda") is False,
        "evidence claims the matrix permutation was recovered; oracle does not",
    )
    _require(
        _field(cuda, "learned_weights_used", bool, "cuda") is False,
        "evidence claims learned weights were used; oracle does not",
    )
    _require(
        _field(cuda, "native_image_inference", bool, "cuda") is False,
        "evidence claims native image inference; oracle does not",
    )

    groups = _field(cuda, "groups", list, "cuda")
    _require(len(groups) == 8, f"expected 8 experiment groups, found {len(groups)}")
    seen: set[tuple[str, int]] = set()
    for index, group in enumerate(groups):
        label = f"groups[{index}]"
        variant = _field(group, "variant", str, label)
        partition = _field(group, "impulse_partition", int, label)
        _require(type(partition) is int, f"{label}: impulse_partition must be int")
        key = (variant, partition)
        _require(key not in seen, f"duplicate group {key}")
        seen.add(key)
        _require(variant in CANONICAL_VARIANTS, f"unknown variant {variant}")
        _require(partition in CANONICAL_PARTITIONS, f"unknown partition {partition}")
        _check_group(f"{variant}/p{partition}", group)

    # Defence in depth: 8 distinct valid keys over an 8-element expected set
    # already implies equality, but assert it explicitly.
    expected = {(v, p) for v in CANONICAL_VARIANTS for p in CANONICAL_PARTITIONS}
    _require(seen == expected, f"missing groups: {sorted(expected - seen)}")

    return dict(
        kernel=CANONICAL_KERNEL,
        groups=len(groups),
        anchors_per_group=2,
        anchors_total=len(groups) * 2,
    )


def self_test() -> dict[str, Any]:
    """Mutation tests: representative corruptions of each enforced invariant are rejected.

    Coverage: all four group flags, the partition-relative decoupled_checked
    rule, both influence footprints (every field plus sign symmetry), the
    pinned kernel, the execute-mode and no-overclaim flags, key
    membership/duplicates, exact-int partition typing, and clean rejection of
    missing keys and non-object containers. Type violations of scalar fields
    (e.g. changed_count as a string) are enforced by _field and exercised
    indirectly through the missing-key cases.
    """
    checked = 0

    def expect_reject(mutate: Any, label: str) -> None:
        nonlocal checked
        payload = _fixture()
        mutate(payload)
        try:
            check_run(payload)
        except OracleError:
            checked += 1
            return
        raise OracleError(f"self-test: corruption not rejected ({label})")

    def set_group_flag(flag: str) -> Any:
        return lambda p: p["cuda"]["groups"][0].__setitem__(flag, False)

    def set_group_field(field: str, value: Any) -> Any:
        return lambda p: p["cuda"]["groups"][0].__setitem__(field, value)

    def set_influence(field: str, value: Any, which: str) -> Any:
        def mutate(p: dict[str, Any]) -> None:
            p["cuda"]["groups"][0][which][field] = value
        return mutate

    def drop_key(field: str) -> Any:
        def mutate(p: dict[str, Any]) -> None:
            p["cuda"]["groups"][0]["coupled_influence"].pop(field, None)
        return mutate

    # Baseline fixture must pass.
    check_run(_fixture())
    checked += 1

    # Group flags: all four enforced flags must be True.
    for flag in REQUIRED_GROUP_FLAGS:
        expect_reject(set_group_flag(flag), f"{flag} false")

    # Partition-relative decoupled set: own partition must not appear, q3 must.
    expect_reject(
        set_group_field("decoupled_checked", ["decoupled-q0", "decoupled-q1", "decoupled-q2"]),
        "decoupled_checked includes own partition and omits q3",
    )
    expect_reject(
        set_group_field("decoupled_checked", ["decoupled-q0"]),
        "decoupled_checked wrong length",
    )
    expect_reject(
        set_group_field("decoupled_checked", [1, "decoupled-q2", "decoupled-q3"]),
        "decoupled_checked mixed element types",
    )

    expect_reject(lambda p: p["cuda"]["groups"].__setitem__(slice(0, 1), []), "missing group")
    expect_reject(
        lambda p: p["cuda"]["groups"][1].__setitem__("impulse_partition", 0),
        "duplicate group key",
    )
    expect_reject(
        lambda p: p["cuda"]["groups"][2].__setitem__("impulse_partition", True),
        "bool impulse_partition",
    )
    expect_reject(
        set_group_field("variant", "unknown-variant"),
        "unknown variant",
    )
    expect_reject(
        lambda p: p["cuda"]["groups"][0].__setitem__("impulse_partition", 9),
        "unknown partition",
    )
    expect_reject(
        lambda p: p["cuda"]["groups"][0].pop("coupled_influence", None),
        "missing coupled_influence object",
    )
    expect_reject(
        lambda p: p["cuda"]["groups"][0].pop("negative_influence", None),
        "missing negative_influence object",
    )
    expect_reject(lambda p: p["cuda"].__setitem__("mode", "skip"), "non-execute mode")
    expect_reject(lambda p: p["cuda"].__setitem__("kernel", "other_kernel"), "wrong kernel")
    expect_reject(lambda p: p["cuda"].__setitem__("learned_weights_used", True), "learned-weight claim")
    expect_reject(lambda p: p["cuda"].__setitem__("matrix_permutation_recovered", True), "permutation claim")
    expect_reject(lambda p: p["cuda"].__setitem__("native_image_inference", True), "image-inference claim")

    # Malformed evidence must raise OracleError (counted), never KeyError/TypeError.
    expect_reject(drop_key("changed_count"), "missing changed_count")
    expect_reject(lambda p: p["cuda"].pop("groups", None), "missing groups")

    for which in ("coupled_influence", "negative_influence"):
        values = CANONICAL_COUPLED_VALUES if which == "coupled_influence" else CANONICAL_NEGATIVE_VALUES
        expect_reject(set_influence("changed", False, which), f"{which} changed flag")
        expect_reject(set_influence("changed_count", CANONICAL_COUNT + 1, which), f"{which} count")
        expect_reject(set_influence("changed_units16", CANONICAL_UNITS16 - 1, which), f"{which} units")
        expect_reject(set_influence("changed_first", 1, which), f"{which} first")
        expect_reject(set_influence("changed_last", CANONICAL_LAST - 1, which), f"{which} last")
        expect_reject(set_influence("sample_offsets", [0] * 8, which), f"{which} offsets")
        expect_reject(set_influence("sample_values", [0.0] * 8, which), f"{which} values")
        expect_reject(set_influence("truncated", True, which), f"{which} truncated")
        # Sign symmetry: the coupled/negative values must be exact opposites.
        flipped = [-v for v in values]
        expect_reject(set_influence("sample_values", flipped, which), f"{which} sign")

    return dict(self_test="passed", mutations_rejected=checked - 1, baseline_accepted=True)


def _fixture() -> dict[str, Any]:
    """Canonical execute-run payload matching the agreed evidence."""
    groups = []
    for variant in CANONICAL_VARIANTS:
        for partition in CANONICAL_PARTITIONS:
            def influence(values: list[float]) -> dict[str, Any]:
                return dict(
                    changed=True,
                    changed_count=CANONICAL_COUNT,
                    changed_first=CANONICAL_FIRST,
                    changed_last=CANONICAL_LAST,
                    changed_units16=CANONICAL_UNITS16,
                    sample_offsets=list(CANONICAL_SAMPLE_OFFSETS),
                    sample_values=list(values),
                    truncated=False,
                )

            groups.append(dict(
                variant=variant,
                impulse_partition=partition,
                base_zero_in_extent=True,
                coupled_differs=True,
                negative_differs=True,
                decoupled_identical=True,
                decoupled_checked=[
                    f"decoupled-q{q}" for q in CANONICAL_PARTITIONS if q != partition
                ],
                coupled_influence=influence(CANONICAL_COUPLED_VALUES),
                negative_influence=influence(CANONICAL_NEGATIVE_VALUES),
            ))
    return dict(cuda=dict(
        mode="execute",
        kernel="cc_dec_input_upsample_1024_512_fp8",
        learned_weights_used=False,
        matrix_permutation_recovered=False,
        native_image_inference=False,
        groups=groups,
    ))


def main(argv: list[str]) -> int:
    if any(arg == "--self-test" for arg in argv):
        if len(argv) != 1:
            print("oracle: --self-test takes no other arguments", file=sys.stderr)
            return 2
        print(json.dumps(self_test(), indent=2, sort_keys=True))
        return 0
    if any(arg.startswith("--") for arg in argv):
        print("oracle: unknown flag", file=sys.stderr)
        return 2
    if not argv:
        print(__doc__.strip(), file=sys.stderr)
        return 2

    reports = []
    for arg in argv:
        run_dir = Path(arg)
        output = run_dir / "output.json"
        if not output.is_file():
            print(f"oracle: no output.json under {run_dir}", file=sys.stderr)
            return 1
        try:
            payload = json.loads(output.read_text())
        except (OSError, json.JSONDecodeError) as error:
            print(f"oracle: cannot read {output}: {error}", file=sys.stderr)
            return 1
        try:
            summary = check_run(payload)
        except OracleError as error:
            print(f"oracle: {run_dir.name}: {error}", file=sys.stderr)
            return 1
        summary["run"] = run_dir.name
        reports.append(summary)
        print(json.dumps(summary, sort_keys=True))

    print(json.dumps(dict(oracle="passed", runs=reports), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
