#!/usr/bin/env python3
"""Constraint lock for the decoder influence-sweep evidence; NOT neural inference.

Validates retained `native-decoder-sweep-probe.py` execute reports against the
structural findings of the fine-grained sweep:

  1. The coupled reference footprint is exactly the canonical 512-unit set.
  2. Every matrix-axis footprint is a subset of the canonical set (coding fewer
     weights can only influence fewer cells under a fixed impulse).
  3. Every impulse-axis footprint is exactly 512 units and its measured
     (row_shift, block_shift) equals `reach_class(impulse_unit)`; footprints of
     distinct reach-classes are pairwise disjoint.

Any future (o, i) permutation or layout claim must reproduce these invariants.
It launches no GPU work and reads only retained evidence.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def load_helper(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    if spec is None or spec.loader is None:
        raise RuntimeError("missing helper")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


sweep = load_helper("sweep", "native-decoder-sweep-probe.py")
base = sweep.base
canonical_units = sweep.canonical_units
reach_class = sweep.reach_class
factor_cell = sweep.factor_cell
EXTENT_BYTES = sweep.EXTENT_BYTES
IMPULSE_BYTES = sweep.IMPULSE_BYTES
CANONICAL = set(canonical_units())


def _typed_ints(values: Any, label: str) -> list[int]:
    base.require(isinstance(values, list), f"{label}: unit_offsets not a list")
    out: list[int] = []
    for value in values:
        base.require(type(value) is int, f"{label}: non-int unit offset")
        out.append(value)
    return out


def _shifted_canonical(row_shift: int, block_shift: int) -> set[int]:
    """Exact 512-unit footprint predicted for a reach-class.

    Rows are the canonical {0,4} pair shifted by row_term (0 -> {0,4},
    1 -> {1,5}); channel blocks are the canonical {0-7,16-23} shifted by
    block_shift (0 or 8). Verified to match every measured impulse footprint.
    """
    base.require(row_shift in (0, 1) and block_shift in (0, 8), "reach-class range")
    rows = (0, 4) if row_shift == 0 else (1, 5)
    blocks = [b + block_shift for b in list(range(0, 8)) + list(range(16, 24))]
    return {rt * 8192 + col * 512 + blk * 16 for rt in rows for col in range(16) for blk in blocks}


def _require_disjoint(reach_sets: dict[Any, set[int]]) -> None:
    """Defence-in-depth: distinct reach-classes must not share output units.

    With exact shifted-canonical enforcement this is automatic; kept as an
    independent, unit-testable invariant so a permutation claim cannot quietly
    reintroduce overlap. Raises ValueError on any pairwise intersection.
    """
    values = list(reach_sets.values())
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            overlap = values[i] & values[j]
            if overlap:
                raise ValueError(f"reach-classes overlap at unit {min(overlap)}")


def validate_report(report: dict[str, Any]) -> dict[str, Any]:
    cuda = report["cuda"] if "cuda" in report else report
    base.require(cuda.get("mode") == "execute", "report is not an execute run")
    records = cuda.get("records")
    base.require(isinstance(records, list) and records, "records present")
    by_name = {}
    for record in records:
        base.require(isinstance(record, dict), "record entry not an object")
        name = record.get("name")
        base.require(isinstance(name, str), "record name")
        base.require(name not in by_name, "duplicate record name")
        by_name[name] = record
    # (1) coupled reference == canonical set.
    base.require("coupled-ref" in by_name, "coupled-ref present")
    ref_units = _typed_ints(by_name["coupled-ref"].get("unit_offsets"), "coupled-ref")
    base.require(set(ref_units) == CANONICAL and len(ref_units) == len(CANONICAL),
                 "coupled-ref footprint is not the canonical set")
    # (2) matrix axis subset; (3) impulse axis reach-class + disjointness.
    matrix_count = impulse_count = 0
    reach_sets: dict[tuple[int, int], set[int]] = {}
    for name, record in by_name.items():
        axis = record.get("axis")
        units = _typed_ints(record.get("unit_offsets"), name)
        if axis == "matrix":
            matrix_count += 1
            base.require(set(units) <= CANONICAL, f"{name}: matrix footprint not canonical-subset")
        elif axis == "impulse":
            impulse_count += 1
            base.require(len(units) == len(set(units)) and len(units) == 512,
                         f"{name}: impulse footprint not 512 distinct units")
            unit = record.get("impulse_unit")
            base.require(type(unit) is int, f"{name}: impulse_unit not int")
            expected = _shifted_canonical(*reach_class(unit))
            base.require(set(units) == expected,
                         f"{name}: footprint != shifted canonical for reach_class({unit})")
            reach_sets.setdefault(reach_class(unit), set()).update(units)
    base.require(matrix_count > 0 and impulse_count > 0, "both axes present")
    # Reach-classes pairwise disjoint (defence-in-depth; automatic under exact
    # shifted-canonical enforcement, kept as an independent invariant).
    _require_disjoint(reach_sets)
    classes = list(reach_sets.values())
    return dict(valid=True, matrix_records=matrix_count, impulse_records=impulse_count,
                reach_classes=len(classes), canonical_units=len(CANONICAL),
                reach_class_sizes=sorted(len(c) for c in classes))


def validate_pair(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Two execute runs must record identical unit_offsets per named record."""
    ra = {r["name"]: r for r in (first["cuda"] if "cuda" in first else first)["records"]}
    rb = {r["name"]: r for r in (second["cuda"] if "cuda" in second else second)["records"]}
    base.require(set(ra) == set(rb), "run record names differ")
    for name in ra:
        base.require(_typed_ints(ra[name].get("unit_offsets"), name) ==
                     _typed_ints(rb[name].get("unit_offsets"), name),
                     f"cross-run unit_offsets differ for {name}")
    return dict(identical=True, records=len(ra))


def self_test() -> dict[str, Any]:
    import copy
    canon = sorted(CANONICAL)

    # Cross-check: an independent construction of a reach-class footprint must
    # equal the model's _shifted_canonical (guards against a divergent model).
    def independent(rs: int, bs: int) -> set[int]:
        rows = (0, 4) if rs == 0 else (1, 5)
        base_units = [rt * 8192 + col * 512 + blk * 16
                      for rt in rows for col in range(16)
                      for blk in (list(range(0, 8)) + list(range(16, 24)))]
        return {u + bs * 16 for u in base_units}

    for rs, bs in ((0, 0), (0, 8), (1, 0), (1, 8)):
        base.require(independent(rs, bs) == _shifted_canonical(rs, bs),
                     f"independent construction matches model ({rs},{bs})")

    def imp_records() -> list[dict[str, Any]]:
        recs = []
        for unit in (0, 4, 8, 12):
            rs, bs = reach_class(unit)
            recs.append(dict(name=f"impulse-unit@{unit}", axis="impulse", impulse_unit=unit,
                             unit_offsets=sorted(_shifted_canonical(rs, bs))))
        return recs

    valid = dict(cuda=dict(mode="execute", records=[
        dict(name="coupled-ref", axis="control", unit_offsets=canon),
        dict(name="matrix-512@0", axis="matrix", unit_offsets=canon[:32]),
        *imp_records(),
    ]))
    ok = validate_report(valid)
    base.require(ok["reach_classes"] == 4, "synthetic reach-classes")
    pair = validate_pair(valid, json.loads(json.dumps(valid)))
    base.require(pair["identical"], "identical pair")
    # _require_disjoint direct controls (defence-in-depth branch coverage).
    _require_disjoint({"a": {1, 2, 3}, "b": {4, 5, 6}})
    try:
        _require_disjoint({"a": {1, 2, 3}, "b": {3, 4, 5}})
    except ValueError:
        pass
    else:
        raise ValueError("overlapping reach-classes accepted by _require_disjoint")
    disjoint_controls = 2
    # Corruption controls: each must be rejected with ValueError (pinned).
    corruptions = 0

    def reject(mutated: dict[str, Any], label: str) -> None:
        nonlocal corruptions
        try:
            validate_report(mutated)
        except ValueError:
            pass
        except Exception as error:  # noqa: BLE001 - a non-ValueError crash is not a clean rejection
            raise ValueError(f"corruption {label} rejected by unexpected {type(error).__name__}: {error}")
        else:
            raise ValueError(f"corruption accepted: {label}")
        corruptions += 1

    # coupled-ref not canonical.
    m = copy.deepcopy(valid); m["cuda"]["records"][0]["unit_offsets"] = canon[:-1]
    reject(m, "coupled-ref missing unit")
    # matrix footprint escapes canonical subset.
    m = copy.deepcopy(valid); m["cuda"]["records"][1]["unit_offsets"] = canon[:32] + [128]
    reject(m, "matrix non-canonical unit")
    # impulse footprint wrong size.
    m = copy.deepcopy(valid); m["cuda"]["records"][2]["unit_offsets"] = canon[:100]
    reject(m, "impulse wrong footprint size")
    # impulse footprint right size but wrong composition (exact-set check):
    # swap one (0,8) unit for a canonical (0,0) unit, keeping 512 distinct units.
    m = copy.deepcopy(valid)
    bad = sorted(_shifted_canonical(0, 8))
    bad[0] = sorted(CANONICAL - _shifted_canonical(0, 8))[0]
    m["cuda"]["records"][3]["unit_offsets"] = sorted(bad)  # unit 4 = class (0,8)
    reject(m, "impulse footprint composition mismatch")
    # impulse reach-class label mismatch.
    m = copy.deepcopy(valid); m["cuda"]["records"][2]["impulse_unit"] = 8
    reject(m, "impulse reach-class mismatch")
    # cross-class contamination (defence-in-depth; caught by exact-set/shift).
    m = copy.deepcopy(valid); m["cuda"]["records"][3]["unit_offsets"][0] = m["cuda"]["records"][2]["unit_offsets"][0]
    reject(m, "cross-class contamination")
    # type corruption: float unit offset.
    m = copy.deepcopy(valid); m["cuda"]["records"][0]["unit_offsets"][0] = float(canon[0])
    reject(m, "float unit offset")
    # type corruption: bool unit offset (False == 0 trap).
    m = copy.deepcopy(valid); m["cuda"]["records"][0]["unit_offsets"][0] = False
    reject(m, "bool unit offset")
    # missing coupled-ref.
    m = copy.deepcopy(valid); m["cuda"]["records"] = m["cuda"]["records"][1:]
    reject(m, "missing coupled-ref")
    # wrong mode.
    m = copy.deepcopy(valid); m["cuda"]["mode"] = "skip"
    reject(m, "wrong mode")
    return dict(synthetic_reach_classes=ok["reach_classes"], pair_records=pair["records"],
                model_cross_checks=4, disjoint_controls=disjoint_controls,
                corruption_controls=corruptions, gpu_kernels_launched=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("self-test", "validate"))
    parser.add_argument("reports", type=Path, nargs="*")
    args = parser.parse_args()
    if args.mode == "self-test":
        print(json.dumps(self_test(), indent=2))
        return
    base.require(1 <= len(args.reports) <= 2, "one or two execute output.json reports")
    loaded = [json.loads(path.read_text()) for path in args.reports]
    results = [validate_report(report) for report in loaded]
    out: dict[str, Any] = dict(reports=[str(p) for p in args.reports], per_report=results)
    if len(loaded) == 2:
        out["pair"] = validate_pair(loaded[0], loaded[1])
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
