#!/usr/bin/env python3
"""Constraint lock for the decoder permutation-resolution evidence; NOT neural
inference.

Validates retained `native-decoder-permutation-probe.py` execute reports. From
the raw per-entry change ranges it re-derives the column decode, the direct-run
predictions and the control footprints, and it pins the probe-computed verdict
flags that were evaluated over the full device buffers at run time:

  1. The column decode (which matrix byte feeds each influenced output byte)
     must equal the decode recomputed from the raw bit-signature runs.
  2. The set of direct runs must match `phase2_plan` applied to the recomputed
     decode exactly (selection rule, impulses, matrix specs, expectations); each
     direct run must change exactly the predicted preimage (change sets compared
     in full), and sign pairs must cover the same change set as their own pairs.
  3. Controls must hold: base and impulse-base runs unchanged, the coupled
     reference exactly the canonical unit set with the established within-unit
     byte pattern, the decoupled control unchanged, counter protocol final,
     guards/immutability flags set, sign ranges consistent with the change
     ranges, and writes inside the established extent.

Scope of verification: change-set equality, predictions and the negative-byte
set cover every changed byte (ranges are retained in full). Byte-VALUE evidence
beyond the first eight changed offsets per record — sign-bit membership and the
exact FP8 sign-bit mirroring — is enforced at probe runtime over the complete
buffers and carried by the retained `negative_ranges`, `negative_count`,
`positive_sign_all`, `mirror_match` fields and `mirror_samples`; this oracle
checks those fields for internal consistency and verifies byte values only on
the retained samples, not independently for every byte.

Any future (o, i) permutation or weight-binding claim must reproduce these
invariants. It launches no GPU work and reads only retained evidence.
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


permutation = load_helper("permutation", "native-decoder-permutation-probe.py")
base = permutation.base
decode_column = permutation.decode_column
preimages = permutation.preimages
expand_ranges = permutation.expand_ranges
canonical_units = permutation.canonical_units
PROBE_INPUTS = permutation.PROBE_INPUTS
SIGNATURE_BITS = permutation.SIGNATURE_BITS
EXTENT_BYTES = permutation.EXTENT_BYTES
OUT0_WINDOW = permutation.OUT0_WINDOW
PROTOCOL_FINAL = permutation.PROTOCOL_FINAL
IMPULSE_BYTES = permutation.IMPULSE_BYTES
CANONICAL = set(canonical_units())
UNIT_BYTES = [0, 1, 2, 3, 8, 9, 10, 11]


def _typed_int(value: Any, label: str) -> int:
    base.require(type(value) is int, f"{label}: not an int")
    return value


def _typed_offsets(ranges: Any, label: str) -> set[int]:
    base.require(isinstance(ranges, list), f"{label}: ranges not a list")
    for pair in ranges:
        base.require(isinstance(pair, list) and len(pair) == 2, f"{label}: bad range pair")
        _typed_int(pair[0], f"{label} start")
        _typed_int(pair[1], f"{label} end")
    return expand_ranges(ranges, OUT0_WINDOW)


def _impulse_key(record: dict[str, Any], label: str) -> tuple[int, int, int | None]:
    partition = _typed_int(record.get("impulse_partition"), f"{label} partition")
    unit = _typed_int(record.get("impulse_unit"), f"{label} unit")
    lane = record.get("impulse_lane")
    base.require(lane is None or type(lane) is int, f"{label} lane")
    return partition, unit, lane


def _recompute_decodes(records: dict[str, dict[str, Any]]) -> dict[tuple[int, int, int | None], dict[int, int]]:
    """Independent column decode from raw signature change ranges."""
    out: dict[tuple[int, int, int | None], dict[int, int]] = {}
    for key in PROBE_INPUTS:
        presence: dict[int, set[int]] = {}
        for record in records.values():
            if record.get("kind") != "signature":
                continue
            if _impulse_key(record, "signature") != key:
                continue
            spec = record.get("matrix_spec")
            base.require(isinstance(spec, list) and len(spec) == 3 and spec[0] == "sig",
                         "signature matrix spec")
            base.require(spec[1] == key[0], "signature matrix partition matches impulse partition")
            bit = _typed_int(spec[2], "signature bit")
            base.require(bit not in presence, "duplicate signature bit")
            presence[bit] = _typed_offsets(record.get("changed_ranges"), "signature")
        base.require(len(presence) == SIGNATURE_BITS, f"{key}: signature runs missing")
        out[key] = decode_column(presence)
    return out


def _expected_direct(record: dict[str, Any],
                     decodes: dict[tuple[int, int, int | None], dict[int, int]]) -> set[int]:
    """Uniform prediction rule: preimage of the matrix byte in the decode of the
    record's own input byte (empty set if the byte is not in that column)."""
    key = _impulse_key(record, "direct")
    spec = record.get("matrix_spec")
    base.require(isinstance(spec, list) and len(spec) == 4 and spec[0] == "single", "direct matrix spec")
    matrix_byte = _typed_int(spec[2], "direct matrix byte")
    base.require(spec[1] == key[0], "direct matrix partition matches impulse partition")
    return set(preimages(decodes[key]).get(matrix_byte, []))


def validate_report(report: dict[str, Any]) -> dict[str, Any]:
    base.require(isinstance(report, dict), "report root not an object")
    cuda = report["cuda"] if "cuda" in report else report
    base.require(isinstance(cuda, dict), "report payload not an object")
    base.require(cuda.get("mode") == "execute", "report is not an execute run")
    base.require(cuda.get("matrix_permutation_recovered") is False, "permutation flag must stay scoped")
    base.require(cuda.get("learned_weights_used") is False, "learned weights flag")
    base.require(cuda.get("native_image_inference") is False, "native image flag")
    base.require(cuda.get("direct_validation_passed") is True, "direct validation flag")
    base.require(cuda.get("sign_mirror_passed") is True, "sign mirror flag")
    base.require(cuda.get("response_value_constant") is True, "response-value flag")
    base.require(_typed_int(cuda.get("response_value_checks"), "response checks") > 0, "response checks")
    base.require(cuda.get("direct_value_agrees") is True, "direct value agreement flag")
    base.require(_typed_int(cuda.get("direct_value_checks"), "direct value checks") > 0, "direct value checks")
    base.require(cuda.get("impulse_leakage_detected") is False, "impulse leakage flag")
    base.require(cuda.get("out0_full_region_final_check") is True, "final full-region check flag")
    raw = cuda.get("records")
    base.require(isinstance(raw, list) and raw, "records present")
    records: dict[str, dict[str, Any]] = {}
    for record in raw:
        base.require(isinstance(record, dict), "record entry not an object")
        name = record.get("name")
        base.require(isinstance(name, str) and name not in records, "unique record name")
        records[name] = record
        counters = record.get("counters_observed")
        base.require(isinstance(counters, list)
                     and all(type(value) is int for value in counters)
                     and counters == [PROTOCOL_FINAL] * 8, f"{name}: counter protocol")
        base.require(record.get("guards_intact") is True, f"{name}: guards flag")
        base.require(record.get("counters_guard_intact") is True, f"{name}: counter guard flag")
        base.require(record.get("immutable") is True, f"{name}: immutability flag")
        changed = _typed_offsets(record.get("changed_ranges"), name)
        base.require(_typed_int(record.get("changed_count"), name) == len(changed), f"{name}: changed_count")
        first = min(changed) if changed else -1
        last = max(changed) if changed else -1
        base.require(record.get("changed_first") == first and record.get("changed_last") == last,
                     f"{name}: changed_first/last not derived from ranges")
        base.require(last == -1 or last < EXTENT_BYTES, f"{name}: extent bound")
        negative = _typed_offsets(record.get("negative_ranges"), name)
        base.require(negative <= changed, f"{name}: negative bytes outside changed set")
        base.require(_typed_int(record.get("negative_count"), name) == len(negative), f"{name}: negative_count")
        base.require(record.get("positive_sign_all") is (len(negative) == 0), f"{name}: positive_sign_all")
        sample_offsets = record.get("sample_offsets")
        sample_bytes = record.get("sample_bytes")
        base.require(isinstance(sample_offsets, list) and isinstance(sample_bytes, list)
                     and len(sample_offsets) == len(sample_bytes), f"{name}: sample lists")
        for offset, byte in zip(sample_offsets, sample_bytes):
            base.require(type(offset) is int and type(byte) is int, f"{name}: sample types")
            base.require((byte & 0x80 != 0) == (offset in negative), f"{name}: sample sign mismatch")
    for name in ("base", "coupled-ref", "decoupled-ctrl"):
        base.require(name in records, f"{name} record present")
    # (3) Controls.
    base.require(records["base"]["changed_count"] == 0, "base output changed")
    for record in records.values():
        if record.get("kind") == "impulse-base":
            base.require(record["changed_count"] == 0, f"{record['name']}: impulse-only leakage")
    ref = records["coupled-ref"]
    ref_offsets = _typed_offsets(ref.get("changed_ranges"), "coupled-ref")
    units = sorted({offset // IMPULSE_BYTES * IMPULSE_BYTES for offset in ref_offsets})
    unit_bytes = sorted({offset % IMPULSE_BYTES for offset in ref_offsets})
    base.require(set(units) == CANONICAL and len(units) == len(CANONICAL), "coupled-ref not canonical")
    base.require(unit_bytes == UNIT_BYTES, "coupled-ref within-unit byte pattern")
    base.require(records["decoupled-ctrl"]["changed_count"] == 0, "decoupled-ctrl changed output")
    # (1) Column decode recomputation.
    decodes = _recompute_decodes(records)
    columns = cuda.get("column_records")
    base.require(isinstance(columns, list) and len(columns) == len(PROBE_INPUTS), "column records")
    seen: set[tuple[int, int, int | None]] = set()
    for column in columns:
        base.require(isinstance(column, dict), "column record")
        key = _impulse_key(column, "column")
        base.require(key in decodes and key not in seen, "column key coverage")
        seen.add(key)
        decode_pairs = column.get("decode")
        base.require(isinstance(decode_pairs, list), "column decode list")
        retained: dict[int, int] = {}
        for pair in decode_pairs:
            base.require(isinstance(pair, list) and len(pair) == 2, "decode pair")
            offset = _typed_int(pair[0], "decode offset")
            base.require(offset not in retained, "duplicate decode offset")
            retained[offset] = _typed_int(pair[1], "decode matrix byte")
        base.require(retained == decodes[key], f"{key}: retained decode differs from recomputed")
        base.require(_typed_int(column.get("influenced_outputs"), "column") == len(decodes[key]),
                     f"{key}: influenced_outputs")
    base.require(seen == set(PROBE_INPUTS), "column record coverage")
    # (2) Direct runs: full plan coverage, recomputed predictions, sign mirroring.
    plan = permutation.phase2_plan(decodes)
    plan_by_name = {entry["name"]: entry for entry in plan}
    direct_records = [r for r in records.values() if r.get("kind") == "direct"]
    base.require({r["name"] for r in direct_records} == set(plan_by_name),
                 "direct records do not match the selection-rule plan")
    own_by_pair: dict[tuple[Any, Any], dict[str, Any]] = {}
    sign_pairs = 0
    for name, entry in plan_by_name.items():
        record = records[name]
        key = _impulse_key(record, name)
        base.require(key == entry["impulse"], f"{name}: impulse differs from plan")
        base.require(record.get("sub") == entry["sub"], f"{name}: sub differs from plan")
        base.require(record.get("matrix_spec") == list(entry["matrix"]), f"{name}: matrix spec differs from plan")
        base.require(_typed_int(record.get("code_byte"), name) == entry["code"], f"{name}: code differs from plan")
        source = entry.get("source")
        base.require(record.get("source") == (list(source) if source is not None else None),
                     f"{name}: source differs from plan")
        expected = _expected_direct(record, decodes)
        retained_expected = record.get("expected_offsets")
        base.require(isinstance(retained_expected, list), f"{name}: expected_offsets")
        expected_list = [_typed_int(value, name) for value in retained_expected]
        base.require(expected_list == sorted(expected), f"{name}: expected_offsets differs from recompute")
        observed = _typed_offsets(record.get("changed_ranges"), name)
        base.require(observed == expected, f"{name}: changed != predicted preimage")
        base.require(record.get("prediction_match") is True, f"{name}: prediction_match")
        spec = record["matrix_spec"]
        if entry["sub"] == "sign":
            base.require(spec[3] == permutation.E4M3_NEG, f"{name}: sign spec code")
            own = own_by_pair.get((key, spec[2]))
            base.require(own is not None, f"{name}: missing own pair")
            # Defence-in-depth: both changed sets already equal the recomputed
            # prediction above; this comparison pins them to each other too.
            base.require(_typed_offsets(own.get("changed_ranges"), name) == observed,
                         f"{name}: sign offsets differ from own pair")
            samples = record.get("mirror_samples")
            base.require(isinstance(samples, list) and samples, f"{name}: mirror samples")
            for row in samples:
                base.require(isinstance(row, list) and len(row) == 3, f"{name}: mirror row")
                offset, pos_byte, neg_byte = (_typed_int(value, f"{name} mirror") for value in row)
                base.require(neg_byte == (pos_byte ^ 0x80), f"{name}: mirror sample not sign-flipped")
                base.require(offset in observed, f"{name}: mirror sample outside changed set")
            base.require(record.get("mirror_match") is True, f"{name}: mirror_match")
            sign_pairs += 1
        else:
            base.require(spec[3] == permutation.E4M3_POS, f"{name}: positive spec code")
            own_by_pair[(key, spec[2])] = record
    base.require(sign_pairs == 2, "sign pairs count")
    return dict(valid=True, records=len(records), columns=len(columns),
                direct_records=len(direct_records), sign_pairs=sign_pairs,
                decode_entries=sum(len(decode) for decode in decodes.values()),
                canonical_units=len(CANONICAL))


def validate_pair(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Two execute runs must record identical change ranges and decodes."""
    tables: list[dict[str, dict[str, Any]]] = []
    payloads: list[dict[str, Any]] = []
    for index, report in enumerate((first, second)):
        label = ("first", "second")[index]
        base.require(isinstance(report, dict), f"pair {label} root not an object")
        cuda = report["cuda"] if "cuda" in report else report
        base.require(isinstance(cuda, dict), f"pair {label} payload not an object")
        raw = cuda.get("records")
        base.require(isinstance(raw, list) and raw, f"pair {label} records present")
        table: dict[str, dict[str, Any]] = {}
        for record in raw:
            base.require(isinstance(record, dict), f"pair {label} record not an object")
            name = record.get("name")
            base.require(isinstance(name, str) and name not in table,
                         f"pair {label} unique record name")
            table[name] = record
        base.require(isinstance(cuda.get("column_records"), list), f"pair {label} column records")
        tables.append(table)
        payloads.append(cuda)
    base.require(set(tables[0]) == set(tables[1]), "run record names differ")
    for name in tables[0]:
        left = _typed_offsets(tables[0][name].get("changed_ranges"), f"pair first {name}")
        right = _typed_offsets(tables[1][name].get("changed_ranges"), f"pair second {name}")
        base.require(left == right, f"cross-run changed_ranges differ for {name}")
    base.require(payloads[0].get("column_records") == payloads[1].get("column_records"),
                 "cross-run decodes differ")
    return dict(identical=True, records=len(tables[0]))


def self_test() -> dict[str, Any]:
    import copy

    # Build a complete synthetic execute report from synthetic column decodes.
    # Keys 12-15 borrow key (i-12)'s minimum matrix byte so cross pairs j=0..3
    # have non-empty predictions while j=4..7 stay empty: both paths exercised.
    decodes: dict[tuple[int, int, int | None], dict[int, int]] = {}
    for index, key in enumerate(PROBE_INPUTS):
        if 12 <= index <= 15:
            decodes[key] = {0: 16 + (index - 12), 3: 16 + index, 6: 1016 + index, 9: 2016 + index}
        else:
            decodes[key] = {0: 16 + index, 3: 1016 + index, 6: 2016 + index, 9: 3016 + index}
    records: list[dict[str, Any]] = []

    def control_record(name: str, kind: str, impulse: tuple[int, int, int | None] | None,
                       matrix_spec: Any, code: int, ranges: list[list[int]]) -> dict[str, Any]:
        changed_offsets = expand_ranges(ranges, OUT0_WINDOW)
        first = min(changed_offsets) if changed_offsets else -1
        last = max(changed_offsets) if changed_offsets else -1
        return dict(name=name, kind=kind, sub=None,
                    impulse_partition=None if impulse is None else impulse[0],
                    impulse_unit=None if impulse is None else impulse[1],
                    impulse_lane=None if impulse is None else impulse[2],
                    matrix_spec=matrix_spec, code_byte=code, source=None,
                    counters_observed=[3] * 8, counters_guard_intact=True,
                    changed_ranges=ranges,
                    changed_count=len(changed_offsets), changed_first=first, changed_last=last,
                    negative_ranges=[], negative_count=0, positive_sign_all=True,
                    sample_offsets=sorted(changed_offsets)[:8], sample_bytes=[0x50] * min(8, len(changed_offsets)),
                    sample_values=[8.0] * min(8, len(changed_offsets)),
                    expected_offsets=None, guards_intact=True, immutable=True)

    records.append(control_record("base", "control", None, None, 0, []))
    for key in PROBE_INPUTS:
        records.append(control_record(f"impulse-base-{key}", "impulse-base", key, None, 0, []))
    canon = sorted(CANONICAL)
    canon_ranges = [[unit + offset, unit + offset + 1] for unit in canon for offset in UNIT_BYTES]
    records.append(control_record("coupled-ref", "control", (0, 0, None), ["whole", 0, 0x38], 0x38, canon_ranges))
    records.append(control_record("decoupled-ctrl", "control", (0, 0, 0), ["whole", 1, 0x38], 0x38, []))
    for key in PROBE_INPUTS:
        decode = decodes[key]
        for bit in range(SIGNATURE_BITS):
            changed = sorted(offset for offset, matrix_byte in decode.items() if (matrix_byte >> bit) & 1)
            records.append(control_record(f"sig-{key}-b{bit}", "signature", key,
                                          ["sig", key[0], bit], 0x38,
                                          [[offset, offset + 1] for offset in changed]))
    for entry in permutation.phase2_plan(decodes):
        code = entry["code"]
        expected = sorted(entry["expected"])
        key = entry["impulse"]
        record = control_record(entry["name"], "direct", key, list(entry["matrix"]), code,
                                [[offset, offset + 1] for offset in expected])
        record["sub"] = entry["sub"]
        record["source"] = list(entry["source"]) if entry.get("source") is not None else None
        record["expected_offsets"] = expected
        record["prediction_match"] = True
        if code == permutation.E4M3_NEG:
            record["negative_ranges"] = [[offset, offset + 1] for offset in expected]
            record["negative_count"] = len(expected)
            record["positive_sign_all"] = len(expected) == 0
            record["sample_bytes"] = [0xD0] * min(8, len(expected))
            record["sample_values"] = [-8.0] * min(8, len(expected))
            record["mirror_samples"] = [[offset, 0x50, 0xD0] for offset in expected[:8]]
            record["mirror_match"] = True
        records.append(record)
    columns = [dict(impulse_partition=key[0], impulse_unit=key[1], impulse_lane=key[2],
                    influenced_outputs=len(decodes[key]),
                    distinct_matrix_bytes=len(set(decodes[key].values())),
                    matrix_byte_min=min(decodes[key].values()),
                    matrix_byte_max=max(decodes[key].values()),
                    shared_matrix_bytes=0, cell_histogram={}, byte_in_unit_histogram={},
                    decode=[[offset, matrix_byte]
                            for offset, matrix_byte in sorted(decodes[key].items())])
               for key in PROBE_INPUTS]
    valid = dict(cuda=dict(mode="execute", matrix_permutation_recovered=False,
                           learned_weights_used=False, native_image_inference=False,
                           direct_validation_passed=True, sign_mirror_passed=True,
                           response_value_constant=True, response_value_checks=1,
                           direct_value_agrees=True, direct_value_checks=1,
                           impulse_leakage_detected=False, out0_full_region_final_check=True,
                           records=records, column_records=columns))
    ok = validate_report(valid)
    base.require(ok["direct_records"] == 18 and ok["sign_pairs"] == 2, "synthetic report validated")
    non_empty_crosses = sum(1 for r in records if r.get("sub") == "cross" and r["changed_count"])
    base.require(non_empty_crosses == 4, "synthetic exercises both cross paths")
    pair = validate_pair(json.loads(json.dumps(valid)), json.loads(json.dumps(valid)))
    base.require(pair["identical"], "identical pair")

    # Corruption controls: each must be rejected with ValueError (pinned).
    corruptions = 0

    def reject(mutated: Any, label: str, expect: str) -> None:
        nonlocal corruptions
        try:
            validate_report(mutated)
        except ValueError as error:
            base.require(expect in str(error),
                         f"corruption {label} rejected by unexpected message: {error}")
        except Exception as error:  # noqa: BLE001 - a non-ValueError crash is not a clean rejection
            raise ValueError(f"corruption {label} rejected by unexpected {type(error).__name__}: {error}")
        else:
            raise ValueError(f"corruption accepted: {label}")
        corruptions += 1

    reject("not-a-report", "non-dict root", "report root not an object")
    m = copy.deepcopy(valid); m["cuda"]["column_records"][0]["decode"][0][1] += 1
    reject(m, "tampered retained decode", "retained decode differs")
    m = copy.deepcopy(valid); m["cuda"]["column_records"][0]["decode"].append([0, 1])
    reject(m, "duplicate decode offset", "duplicate decode offset")
    m = copy.deepcopy(valid); m["cuda"]["records"][-1]["expected_offsets"] = [0]
    reject(m, "tampered direct expectation", "expected_offsets differs")

    def rederive(record: dict[str, Any]) -> None:
        """Keep the convenience fields consistent with the mutated change ranges
        so the mutation reaches its intended semantic check, not the derived
        first/last/count guards."""
        offsets = sorted(expand_ranges(record["changed_ranges"], OUT0_WINDOW))
        record["changed_count"] = len(offsets)
        record["changed_first"] = offsets[0] if offsets else -1
        record["changed_last"] = offsets[-1] if offsets else -1
        record["negative_ranges"] = []
        record["negative_count"] = 0
        record["positive_sign_all"] = True
        record["sample_offsets"] = offsets[:8]
        record["sample_bytes"] = [0x50] * min(8, len(offsets))
        record["sample_values"] = [8.0] * min(8, len(offsets))

    direct_index = next(i for i, r in enumerate(records)
                        if r["kind"] == "direct" and r["sub"] != "sign" and r["changed_count"])
    m = copy.deepcopy(valid)
    m["cuda"]["records"][direct_index]["changed_ranges"] = [[5, 6]]
    rederive(m["cuda"]["records"][direct_index])
    reject(m, "direct changed != predicted preimage", "changed != predicted preimage")
    m = copy.deepcopy(valid)
    cross_index = next(i for i, r in enumerate(records) if r.get("sub") == "cross" and r["changed_count"])
    del m["cuda"]["records"][cross_index]
    reject(m, "deleted direct plan entry", "selection-rule plan")
    m = copy.deepcopy(valid); m["cuda"]["records"][direct_index]["matrix_spec"] = ["single", 0, 16, 0xB8]
    reject(m, "matrix spec inconsistent with plan/code", "matrix spec differs")
    impulse_index = next(i for i, r in enumerate(records) if r["kind"] == "impulse-base")
    m = copy.deepcopy(valid); m["cuda"]["records"][impulse_index]["changed_ranges"] = [[0, 1]]
    rederive(m["cuda"]["records"][impulse_index])
    reject(m, "impulse-base leakage", "impulse-only leakage")
    m = copy.deepcopy(valid); m["cuda"]["records"][0]["counters_observed"] = [3.0] * 8
    reject(m, "counter protocol type violation", "counter protocol")
    m = copy.deepcopy(valid); m["cuda"]["records"][0]["counters_observed"] = [2] * 8
    reject(m, "counter protocol violation", "counter protocol")
    m = copy.deepcopy(valid); m["cuda"]["records"][0]["counters_guard_intact"] = False
    reject(m, "counter guard violation flagged", "counter guard flag")
    m = copy.deepcopy(valid); del m["cuda"]["records"][0]
    reject(m, "missing base record", "base record present")
    ref_index = next(i for i, r in enumerate(records) if r["name"] == "coupled-ref")
    m = copy.deepcopy(valid); del m["cuda"]["records"][ref_index]
    reject(m, "missing coupled-ref record", "coupled-ref record present")
    decoupled_index = next(i for i, r in enumerate(records) if r["name"] == "decoupled-ctrl")
    m = copy.deepcopy(valid); del m["cuda"]["records"][decoupled_index]
    reject(m, "missing decoupled-ctrl record", "decoupled-ctrl record present")
    first_unit = canon[0]
    kept = [r for r in canon_ranges if not (first_unit <= r[0] < first_unit + IMPULSE_BYTES)]
    m = copy.deepcopy(valid)
    m["cuda"]["records"][ref_index]["changed_ranges"] = kept
    rederive(m["cuda"]["records"][ref_index])
    reject(m, "coupled-ref missing unit", "coupled-ref not canonical")
    without_byte11 = [r for r in canon_ranges if r[0] % IMPULSE_BYTES != 11]
    m = copy.deepcopy(valid)
    m["cuda"]["records"][ref_index]["changed_ranges"] = without_byte11
    rederive(m["cuda"]["records"][ref_index])
    reject(m, "coupled-ref within-unit byte pattern", "within-unit byte pattern")
    sign_index = next(i for i, r in enumerate(records) if r.get("sub") == "sign" and r["changed_count"])
    m = copy.deepcopy(valid); m["cuda"]["records"][sign_index]["mirror_samples"] = [[0, 0x50, 0x51]]
    reject(m, "tampered mirror sample", "mirror sample not sign-flipped")
    m = copy.deepcopy(valid); m["cuda"]["records"][sign_index]["negative_count"] += 1
    reject(m, "negative_count inconsistent", "negative_count")
    m = copy.deepcopy(valid); m["cuda"]["records"][sign_index]["positive_sign_all"] = True
    reject(m, "positive_sign_all inconsistent", "positive_sign_all")
    m = copy.deepcopy(valid); m["cuda"]["records"][sign_index]["changed_ranges"] = []
    m["cuda"]["records"][sign_index]["changed_count"] = 0
    m["cuda"]["records"][sign_index]["changed_first"] = -1
    m["cuda"]["records"][sign_index]["changed_last"] = -1
    reject(m, "sign record changed set tampered", "negative bytes outside changed set")
    sig_index = next(i for i, r in enumerate(records) if r["kind"] == "signature")
    m = copy.deepcopy(valid); del m["cuda"]["records"][sig_index]
    reject(m, "missing signature record", "signature runs missing")
    m = copy.deepcopy(valid); m["cuda"]["records"].append(copy.deepcopy(m["cuda"]["records"][0]))
    reject(m, "duplicate record name", "unique record name")
    m = copy.deepcopy(valid)
    beyond = EXTENT_BYTES + 16
    m["cuda"]["records"][sig_index]["changed_ranges"] = [[beyond, beyond + 1]]
    rederive(m["cuda"]["records"][sig_index])
    reject(m, "write outside established extent", "extent bound")
    m = copy.deepcopy(valid)
    m["cuda"]["records"][sig_index]["changed_ranges"] = [[EXTENT_BYTES - 2, EXTENT_BYTES]]
    m["cuda"]["records"][sig_index]["changed_count"] = 2
    m["cuda"]["records"][sig_index]["changed_first"] = EXTENT_BYTES - 2
    m["cuda"]["records"][sig_index]["changed_last"] = -1
    reject(m, "changed_last inconsistent with ranges", "changed_first/last not derived")
    m = copy.deepcopy(valid); m["cuda"]["records"][0]["guards_intact"] = False
    reject(m, "guard violation flagged", "guards flag")
    m = copy.deepcopy(valid); m["cuda"]["direct_validation_passed"] = False
    reject(m, "direct validation flag cleared", "direct validation flag")
    m = copy.deepcopy(valid); m["cuda"]["response_value_constant"] = False
    reject(m, "response-value flag cleared", "response-value flag")
    m = copy.deepcopy(valid); m["cuda"]["direct_value_agrees"] = False
    reject(m, "direct value agreement flag cleared", "direct value agreement flag")

    # Pair-mode corruption controls: validate_pair must reject cleanly too.
    pair_corruptions = 0

    def reject_pair(mutated_a: Any, mutated_b: Any, label: str, expect: str) -> None:
        nonlocal pair_corruptions
        try:
            validate_pair(mutated_a, mutated_b)
        except ValueError as error:
            base.require(expect in str(error),
                         f"pair corruption {label} rejected by unexpected message: {error}")
        except Exception as error:  # noqa: BLE001 - a non-ValueError crash is not a clean rejection
            raise ValueError(f"pair corruption {label} rejected by unexpected {type(error).__name__}: {error}")
        else:
            raise ValueError(f"pair corruption accepted: {label}")
        pair_corruptions += 1

    def snapshot() -> dict[str, Any]:
        return json.loads(json.dumps(valid))

    reject_pair("not-a-report", snapshot(), "non-dict root", "root not an object")
    b = snapshot(); del b["cuda"]["records"][0]
    reject_pair(snapshot(), b, "record name sets differ", "run record names differ")
    b = snapshot(); b["cuda"]["records"][5]["changed_ranges"] = [[7, 8]]
    reject_pair(snapshot(), b, "changed_ranges differ", "changed_ranges differ")
    b = snapshot(); b["cuda"]["column_records"][0]["decode"][0][1] += 1
    reject_pair(snapshot(), b, "decodes differ", "cross-run decodes differ")
    b = snapshot(); b["cuda"]["records"].append(copy.deepcopy(b["cuda"]["records"][0]))
    reject_pair(snapshot(), b, "duplicate record names", "unique record name")
    b = snapshot(); del b["cuda"]["column_records"]
    reject_pair(snapshot(), b, "missing column records", "column records")
    b = snapshot(); b2 = snapshot()
    del b["cuda"]["records"][5]["changed_ranges"]
    del b2["cuda"]["records"][5]["changed_ranges"]
    reject_pair(b2, b, "missing changed_ranges on both sides", "ranges not a list")
    return dict(synthetic_report_records=len(records), synthetic_decodes=sum(len(d) for d in decodes.values()),
                corruption_controls=corruptions, pair_controls=pair_corruptions, gpu_kernels_launched=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("self-test", "validate", "pair"))
    parser.add_argument("reports", type=Path, nargs="*")
    args = parser.parse_args()
    if args.mode == "self-test":
        print(json.dumps(self_test(), indent=2))
        return
    if args.mode == "validate":
        base.require(len(args.reports) == 1, "one report required")
        report = json.loads(args.reports[0].read_text())
        print(json.dumps(validate_report(report), indent=2))
        return
    base.require(len(args.reports) == 2, "two reports required")
    first = json.loads(args.reports[0].read_text())
    second = json.loads(args.reports[1].read_text())
    print(json.dumps(validate_pair(first, second), indent=2))


if __name__ == "__main__":
    main()
