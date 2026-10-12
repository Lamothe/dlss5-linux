#!/usr/bin/env python3
"""Constraint lock for the decoder structural-extension evidence; NOT neural
inference.

Validates retained `native-decoder-extension-probe.py` execute reports. From the
raw per-entry change ranges it independently re-derives the column decodes,
re-fits the 12 single-bit effects, re-composes the hold-out/identity predictions
and re-derives every direct-run expectation, pinning the probe-computed verdict
flags:

  1. Training: each single-bit column must fit a constant base term under the
     fixed bit shuffle, and the fitted effect must reproduce its own column.
  2. Hold-out and identity: the composed closed-form map must equal the
     measured column EXACTLY (every (output byte, matrix byte) pair).
  3. Direct runs must change exactly the model-predicted preimage: the
     matrix-byte-0 preimages (empty when the composed base is nonzero) and the
     spot triples.
  4. Controls must hold: base/impulse-base unchanged, coupled reference exactly
     the canonical unit set with the established byte pattern, decoupled
     control unchanged, counter protocol final, guards/immutability set, and
     writes inside the established extent.

Scope of verification: change-set equality is enforced from the retained full
ranges. The model primitives (bit shuffle, composition) are imported from the
probe under validation and cross-checked here against independent inline
arithmetic on every training entry; the raw evidence remains the source of
truth. Byte-VALUE evidence (sample bytes/values) is recorded per record and
used for identification only; full-buffer FP8 sign and value consistency is
enforced at probe runtime and is NOT re-derived here.

Any future `matrix_permutation_recovered` claim must reproduce these
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


extension = load_helper("extension", "native-decoder-extension-probe.py")
base = extension.base
decode_column = extension.decode_column
expand_ranges = extension.expand_ranges
canonical_units = extension.canonical_units
shuffle = extension.shuffle
model_map = extension.model_map
signature_map = extension.signature_map
zero_preimage = extension.zero_preimage
fit_effect = extension.fit_effect
bit_key = extension.bit_key
index_bits = extension.index_bits
TRAIN_INPUTS = extension.TRAIN_INPUTS
HOLDOUT_INPUTS = extension.HOLDOUT_INPUTS
IDENTITY_INPUTS = extension.IDENTITY_INPUTS
SPOT_INPUTS = extension.SPOT_INPUTS
ALL_INPUTS = extension.ALL_INPUTS
phase1_plan = extension.phase1_plan
phase2_plan = extension.phase2_plan
SIGNATURE_BITS = extension.SIGNATURE_BITS
EXTENT_BYTES = extension.EXTENT_BYTES
OUT0_WINDOW = extension.OUT0_WINDOW
PROTOCOL_FINAL = extension.PROTOCOL_FINAL
IMPULSE_BYTES = extension.IMPULSE_BYTES
BIT_COUNT = extension.BIT_COUNT
CANONICAL = set(canonical_units())
UNIT_BYTES = list(extension.UNIT_BYTES)

# Independent inline re-derivation of the shuffle (must agree with the probe's).
_INLINE_STRIDES = {0: 64, 1: 8, 3: 512, 4: 128, 5: 256,
                   9: 1024, 10: 2048, 11: 4096, 12: 8192}


def _inline_shuffle(offset: int) -> int:
    return sum(stride * ((offset >> bit) & 1) for bit, stride in _INLINE_STRIDES.items())


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


def _ranges(offsets: Any) -> list[list[int]]:
    """Merge sorted offsets into retained-style half-open ranges."""
    out: list[list[int]] = []
    for offset in sorted(offsets):
        if out and out[-1][1] == offset:
            out[-1][1] = offset + 1
        else:
            out.append([offset, offset + 1])
    return out


def _recompute_decodes(records: dict[str, dict[str, Any]]) -> dict[tuple[int, int, int], dict[int, int]]:
    """Independent column decode from raw signature change ranges."""
    out: dict[tuple[int, int, int], dict[int, int]] = {}
    for key in ALL_INPUTS:
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


def _fit_inline(decode: dict[int, int]) -> tuple[int, int, int, int]:
    """Independent fit via inline shuffle arithmetic (not the probe's fit)."""
    terms = {matrix_byte - _inline_shuffle(offset) for offset, matrix_byte in decode.items()}
    base.require(len(terms) == 1, "inline fit: base term not constant")
    base_value = terms.pop()
    rows = {(offset >> 13) for offset in decode}
    blocks = {(offset // 16) % 32 for offset in decode}
    byte_offsets = {offset % 16 for offset in decode}
    base.require(len(rows) == 1, "inline fit: row term not unique")
    base.require(len(byte_offsets) == 8, "inline fit: byte pattern size")
    row = rows.pop()
    byte_shift = min(byte_offsets)
    base.require(byte_shift in (0, 4), "inline fit: byte shift not measured")
    base.require(byte_offsets == {byte + byte_shift for byte in UNIT_BYTES},
                 "inline fit: byte pattern not established set")
    shift = 8 if min(blocks) >= 8 else 0
    expected = set(range(shift, shift + 8)) | set(range(shift + 16, shift + 24))
    base.require(blocks == expected, "inline fit: block set not canonical")
    return base_value, row, shift, byte_shift


def _expected_direct(record: dict[str, Any],
                     full_maps: dict[tuple[int, int, int], dict[int, int]]) -> set[int]:
    """Model rule: matrix byte 0 -> zero preimage; else visible preimage."""
    key = _impulse_key(record, "direct")
    base.require(key in full_maps, "direct impulse key known")
    spec = record.get("matrix_spec")
    base.require(isinstance(spec, list) and len(spec) == 4 and spec[0] == "single", "direct matrix spec")
    base.require(spec[1] == key[0], "direct matrix partition matches impulse partition")
    matrix_byte = _typed_int(spec[2], "direct matrix byte")
    if matrix_byte == 0:
        return set(zero_preimage(full_maps[key]))
    visible = signature_map(full_maps[key])
    return {offset for offset, value in visible.items() if value == matrix_byte}


def validate_report(report: dict[str, Any]) -> dict[str, Any]:
    base.require(isinstance(report, dict), "report root not an object")
    cuda = report["cuda"] if "cuda" in report else report
    base.require(isinstance(cuda, dict), "report payload not an object")
    base.require(cuda.get("mode") == "execute", "report is not an execute run")
    base.require(cuda.get("learned_weights_used") is False, "learned weights flag")
    base.require(cuda.get("native_image_inference") is False, "native image flag")
    base.require(cuda.get("training_self_prediction_passed") is True, "training flag")
    base.require(cuda.get("holdout_model_prediction_passed") is True, "hold-out flag")
    base.require(cuda.get("direct_validation_passed") is True, "direct flag")
    base.require(cuda.get("impulse_leakage_detected") is False, "impulse leakage flag")
    base.require(cuda.get("out0_full_region_final_check") is True, "final check flag")
    raw = cuda.get("records")
    base.require(isinstance(raw, list) and raw, "records present")
    records: dict[str, dict[str, Any]] = {}
    for record in raw:
        base.require(isinstance(record, dict), "record entry not an object")
        name = record.get("name")
        base.require(isinstance(name, str) and name not in records, "unique record name")
        records[name] = record
        counters = record.get("counters_observed")
        base.require(isinstance(counters, list) and all(type(v) is int for v in counters),
                     f"{name}: counters not int list")
        base.require(record.get("counters_guard_intact") is True, f"{name}: counter guard")
        base.require(record.get("guards_intact") is True, f"{name}: guards")
        base.require(record.get("immutable") is True, f"{name}: immutability")
        base.require(counters == [PROTOCOL_FINAL] * len(counters), f"{name}: counter protocol")
        changed = _typed_offsets(record.get("changed_ranges"), name)
        base.require(_typed_int(record.get("changed_count"), f"{name} count") == len(changed),
                     f"{name}: changed_count mismatch")
        first = min(changed) if changed else -1
        last = max(changed) if changed else -1
        base.require(_typed_int(record.get("changed_first"), f"{name} first") == first,
                     f"{name}: changed_first mismatch")
        base.require(_typed_int(record.get("changed_last"), f"{name} last") == last,
                     f"{name}: changed_last mismatch")
        base.require(not (changed and last >= EXTENT_BYTES), f"{name}: write outside extent")

    # Phase-1 plan identities.
    planned = {entry["name"] for entry in phase1_plan()}
    base.require(planned <= set(records), "phase-1 plan entries present")
    base_record = records["base"]
    base.require(not _typed_offsets(base_record.get("changed_ranges"), "base"), "base run unchanged")
    for key in ALL_INPUTS:
        name = f"impulse-base-{extension.key_name(key)}"
        base.require(name in records, f"{name} present")
        base.require(not _typed_offsets(records[name].get("changed_ranges"), name),
                     f"{name}: impulse-only leakage")
    coupled = records["coupled-ref"]
    coupled_changed = _typed_offsets(coupled.get("changed_ranges"), "coupled-ref")
    expected_coupled = {unit + byte for unit in CANONICAL for byte in UNIT_BYTES}
    base.require(coupled_changed == expected_coupled,
                 "coupled-ref canonical unit set")
    decoupled = records["decoupled-ctrl"]
    base.require(not _typed_offsets(decoupled.get("changed_ranges"), "decoupled-ctrl"),
                 "decoupled control changed the output")

    # Independent decodes, fits, composed predictions.
    decodes = _recompute_decodes(records)
    fitted: dict[int, tuple[int, int, int, int]] = {}
    training_ok = True
    for bit in range(BIT_COUNT):
        key = bit_key(bit)
        effect = _fit_inline(decodes[key])
        fitted[bit] = effect
        reported = cuda.get("bit_effects", {}).get(str(bit))
        base.require(isinstance(reported, list) and len(reported) == 4, f"bit {bit}: reported effect")
        base.require(tuple(reported) == effect, f"bit {bit}: reported effect mismatch")
        # Self-prediction from the single-bit effect alone.
        single = {other: (0, 0, 0, 0) for other in range(BIT_COUNT)}
        single[bit] = effect
        if signature_map(model_map(single, *key[1:])) != decodes[key]:
            training_ok = False
    # Training self-prediction is collected into the re-derived verdict below.

    full_maps = {key: model_map(fitted, key[1], key[2]) for key in ALL_INPUTS}
    holdout_ok = True
    for kind, keys in (("holdout", HOLDOUT_INPUTS), ("identity", IDENTITY_INPUTS)):
        for key in keys:
            predicted = signature_map(full_maps[key])
            if predicted != decodes[key]:
                holdout_ok = False
    # Hold-out/identity prediction is collected into the re-derived verdict below.

    # Reported training/hold-out records must agree with re-derivation.
    training_records = cuda.get("training_records")
    base.require(isinstance(training_records, list) and len(training_records) == BIT_COUNT,
                 "training record count")
    seen_bits: set[int] = set()
    for record in training_records:
        bit = _typed_int(record.get("bit"), "training bit")
        base.require(0 <= bit < BIT_COUNT, "training bit range")
        base.require(bit not in seen_bits, "training bit unique")
        seen_bits.add(bit)
        base.require(tuple(record.get("fitted_effect", [])) == fitted[bit], "training effect mismatch")
        base.require(record.get("self_prediction_match") is True, "training self-match flag")
    base.require(seen_bits == set(range(BIT_COUNT)), "training bits cover all index bits")
    holdout_records = cuda.get("holdout_records")
    base.require(isinstance(holdout_records, list)
                 and len(holdout_records) == len(HOLDOUT_INPUTS) + len(IDENTITY_INPUTS),
                 "hold-out record count")
    for record in holdout_records:
        base.require(record.get("model_match") is True, "hold-out model-match flag")
        base.require(record.get("mismatches") == [], "hold-out mismatches empty")

    # Reported column records must match the re-derived decodes exactly.
    column_records = cuda.get("column_records")
    base.require(isinstance(column_records, list) and len(column_records) == len(ALL_INPUTS),
                 "column record count")
    seen_keys: set[tuple[int, int, int]] = set()
    for record in column_records:
        base.require(isinstance(record, dict), "column record not an object")
        key = _impulse_key(record, "column record")
        base.require(key in decodes and key not in seen_keys, "column record key known/unique")
        seen_keys.add(key)
        raw_decode = record.get("decode")
        base.require(isinstance(raw_decode, list), "column decode list")
        pairs = sorted(tuple(pair) for pair in raw_decode)
        base.require(pairs == sorted(tuple(pair) for pair in decodes[key].items()),
                     "column record decode mismatch")
    base.require(seen_keys == set(ALL_INPUTS), "column records cover all inputs")

    # Direct runs: exact change sets against the model rule, collected for the
    # re-derived verdict along with the training and hold-out axes.
    direct_names = {entry["name"] for entry in phase2_plan(full_maps)}
    direct_records = {name: record for name, record in records.items()
                      if record.get("kind") == "direct"}
    base.require(set(direct_records) == direct_names, "direct plan identities")
    direct_ok = True
    for name, record in direct_records.items():
        expected = _expected_direct(record, full_maps)
        changed = _typed_offsets(record.get("changed_ranges"), name)
        reported = record.get("expected_offsets")
        base.require(isinstance(reported, list)
                     and all(type(v) is int for v in reported), f"{name}: expected not int list")
        base.require(sorted(reported) == sorted(expected), f"{name}: expected offsets mismatch")
        base.require(record.get("prediction_match") is True, f"{name}: prediction flag")
        if changed != expected:
            direct_ok = False
    zero_checks = len([n for n in direct_records if n.startswith("direct-zero-")])
    base.require(zero_checks == len(ALL_INPUTS), "zero-preimage check count")
    # Re-derived verdict: the conjunction of the three independent axes. A
    # report is valid only if all three hold, and then the recorded flag must
    # be true (a valid report cannot carry a false verdict).
    recovered = bool(training_ok and holdout_ok and direct_ok)
    base.require(recovered, "re-derived verdict failed: "
                            f"training={training_ok} holdout={holdout_ok} direct={direct_ok}")
    base.require(cuda.get("matrix_permutation_recovered") is True,
                 "permutation flag must be true for a valid report")
    return dict(valid=True, records=len(records),
                signature_decodes=sum(len(d) for d in decodes.values()),
                training_bits=fitted, holdout_inputs=len(HOLDOUT_INPUTS),
                identity_inputs=len(IDENTITY_INPUTS), direct_entries=len(direct_records),
                zero_preimage_checks=zero_checks, matrix_permutation_recovered=recovered,
                gpu_kernels_launched=0)


def validate_pair(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Two execute runs must record identical evidence: same record identities,
    per-record change sets (compared as expanded offset sets, order-insensitive)
    and the same sampled bytes. Change sets and decodes are additionally pinned
    to the model by validate_report; these comparisons are cross-run identity
    checks over the raw retained evidence."""
    a = validate_report(first)
    b = validate_report(second)
    base.require(a["signature_decodes"] == b["signature_decodes"], "cross-run decode totals differ")
    base.require(a["training_bits"] == b["training_bits"], "cross-run fitted effects differ")
    base.require(a["direct_entries"] == b["direct_entries"], "cross-run direct counts differ")
    fa = first["cuda"] if "cuda" in first else first
    fb = second["cuda"] if "cuda" in second else second
    left_records = {record["name"]: record for record in fa["records"]}
    right_records = {record["name"]: record for record in fb["records"]}
    base.require(set(left_records) == set(right_records), "run record names differ")
    for name in left_records:
        left = _typed_offsets(left_records[name].get("changed_ranges"), f"pair first {name}")
        right = _typed_offsets(right_records[name].get("changed_ranges"), f"pair second {name}")
        base.require(left == right, f"cross-run changed_ranges differ for {name}")
        base.require(left_records[name].get("sample_offsets") == right_records[name].get("sample_offsets"),
                     f"cross-run sample offsets differ for {name}")
        base.require(left_records[name].get("sample_bytes") == right_records[name].get("sample_bytes"),
                     f"cross-run sample bytes differ for {name}")
    left_columns = {(record.get("impulse_partition"), record.get("impulse_unit"),
                     record.get("impulse_lane")): record
                    for record in fa["column_records"]}
    right_columns = {(record.get("impulse_partition"), record.get("impulse_unit"),
                      record.get("impulse_lane")): record
                     for record in fb["column_records"]}
    base.require(set(left_columns) == set(right_columns), "cross-run column identities differ")
    for key in left_columns:
        base.require(left_columns[key].get("decode") == right_columns[key].get("decode"),
                     f"cross-run decodes differ for {key}")
    return dict(identical=True, records=len(left_records),
                signature_decodes=a["signature_decodes"], gpu_kernels_launched=0)


def self_test() -> dict[str, Any]:
    # Model primitives must agree with the independent inline arithmetic.
    for offset in (0, 1, 3, 8, 64, 320, 8059, 131071):
        base.require(shuffle(offset) == _inline_shuffle(offset), "shuffle cross-check")
    # Synthetic evidence: exact maps under known effects, converted to ranges.
    known = {0: (1, 0, 0, 0), 1: (2, 0, 0, 0), 2: (0, 4, 0, 0), 3: (4, 0, 0, 0),
             4: (16, 0, 0, 0), 5: (32, 0, 0, 0), 6: (0, 0, 8, 0), 7: (0, 1, 0, 0),
             8: (0, 0, 0, 4), 9: (16384, 0, 0, 0), 10: (32768, 0, 0, 0),
             11: (65536, 0, 0, 0)}
    decodes = {key: signature_map(model_map(known, key[1], key[2])) for key in ALL_INPUTS}
    full_maps = {key: model_map(known, key[1], key[2]) for key in ALL_INPUTS}

    def synthetic() -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        records.append(dict(name="base", kind="control", impulse_partition=None,
                            impulse_unit=None, impulse_lane=None, matrix_spec=None,
                            counters_observed=[PROTOCOL_FINAL] * 8, counters_guard_intact=True,
                            guards_intact=True, immutable=True,
                            changed_ranges=[], changed_count=0,
                            changed_first=-1, changed_last=-1))
        for key in ALL_INPUTS:
            records.append(dict(name=f"impulse-base-{extension.key_name(key)}", kind="impulse-base",
                                impulse_partition=key[0], impulse_unit=key[1], impulse_lane=key[2],
                                matrix_spec=None, counters_observed=[PROTOCOL_FINAL] * 8,
                                counters_guard_intact=True, guards_intact=True, immutable=True,
                                changed_ranges=[], changed_count=0, changed_first=-1,
                                changed_last=-1))
        coupled_offsets = [offset for unit in canonical_units() for offset in
                           (unit + b for b in UNIT_BYTES)]
        changed = sorted(coupled_offsets)
        records.append(dict(name="coupled-ref", kind="control", impulse_partition=0,
                            impulse_unit=0, impulse_lane=None, matrix_spec=["whole", 0, 248],
                            counters_observed=[PROTOCOL_FINAL] * 8, counters_guard_intact=True,
                            guards_intact=True, immutable=True,
                            changed_ranges=_ranges(changed), changed_count=len(changed),
                            changed_first=changed[0], changed_last=changed[-1]))
        records.append(dict(name="decoupled-ctrl", kind="control", impulse_partition=0,
                            impulse_unit=0, impulse_lane=0, matrix_spec=["whole", 1, 248],
                            counters_observed=[PROTOCOL_FINAL] * 8, counters_guard_intact=True,
                            guards_intact=True, immutable=True,
                            changed_ranges=[], changed_count=0, changed_first=-1, changed_last=-1))
        for key in ALL_INPUTS:
            full = full_maps[key]
            for bit in range(SIGNATURE_BITS):
                changed = sorted(offset for offset, value in decodes[key].items()
                                 if (value >> bit) & 1)
                records.append(dict(name=f"sig-{extension.key_name(key)}-b{bit}", kind="signature",
                                    impulse_partition=key[0], impulse_unit=key[1],
                                    impulse_lane=key[2], matrix_spec=["sig", key[0], bit],
                                    counters_observed=[PROTOCOL_FINAL] * 8,
                                    counters_guard_intact=True, guards_intact=True, immutable=True,
                                    changed_ranges=_ranges(changed), changed_count=len(changed),
                                    changed_first=changed[0] if changed else -1,
                                    changed_last=changed[-1] if changed else -1))
        for entry in phase2_plan(full_maps):
            expected = sorted(entry["expected"])
            spec = list(entry["matrix"])
            key = entry["impulse"]
            records.append(dict(name=entry["name"], kind="direct", sub=entry["sub"],
                                impulse_partition=key[0], impulse_unit=key[1], impulse_lane=key[2],
                                matrix_spec=spec, counters_observed=[PROTOCOL_FINAL] * 8,
                                counters_guard_intact=True, guards_intact=True, immutable=True,
                                changed_ranges=_ranges(expected), changed_count=len(expected),
                                changed_first=expected[0] if expected else -1,
                                changed_last=expected[-1] if expected else -1,
                                expected_offsets=expected, prediction_match=True))
        # Sample evidence fields (probe records carry these; pair checks compare them).
        for record in records:
            sample = sorted(_typed_offsets(record["changed_ranges"], "sample"))[:8]
            record["sample_offsets"] = sample
            record["sample_bytes"] = [0x38 for _ in sample]
            record["sample_values"] = [1.0 for _ in sample]
        report: dict[str, Any] = dict(mode="execute", learned_weights_used=False,
                                      native_image_inference=False,
                                      training_self_prediction_passed=True,
                                      holdout_model_prediction_passed=True,
                                      direct_validation_passed=True,
                                      impulse_leakage_detected=False,
                                      out0_full_region_final_check=True,
                                      matrix_permutation_recovered=True,
                                      bit_effects={str(bit): list(known[bit])
                                                   for bit in range(BIT_COUNT)},
                                      training_records=[dict(bit=bit, fitted_effect=list(known[bit]),
                                                             self_prediction_match=True)
                                                        for bit in range(BIT_COUNT)],
                                      holdout_records=[dict(kind=kind, model_match=True, mismatches=[])
                                                       for kind in ("holdout", "holdout", "holdout",
                                                                    "holdout", "holdout", "holdout",
                                                                    "holdout", "holdout", "identity")],
                                      records=records,
                                      column_records=[dict(impulse_partition=key[0],
                                                           impulse_unit=key[1],
                                                           impulse_lane=key[2],
                                                           decode=sorted(decodes[key].items()))
                                                      for key in ALL_INPUTS])
        return report

    result = validate_report(synthetic())
    base.require(result["valid"] and result["matrix_permutation_recovered"], "synthetic accepted")
    corruptions = 0

    def reject(label: str, mutate: Any, expect: str) -> None:
        nonlocal corruptions
        report = synthetic()
        mutate(report)
        try:
            validate_report(report)
        except ValueError as error:
            if type(error) is not ValueError:  # OutputMismatch is not a clean rejection
                raise extension.OutputMismatch(
                    f"corruption {label} rejected by unexpected {type(error).__name__}: {error}")
            if expect not in str(error):
                raise extension.OutputMismatch(
                    f"corruption {label} rejected at wrong check: {error}")
            corruptions += 1
            return
        except Exception as error:  # noqa: BLE001 - a non-ValueError crash is not a clean rejection
            raise extension.OutputMismatch(
                f"corruption {label} rejected by unexpected {type(error).__name__}: {error}")
        raise extension.OutputMismatch(f"corruption not rejected: {label}")

    def records_of(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {record["name"]: record for record in report["records"]}

    def stamp(record: dict[str, Any], changed: set[int]) -> None:
        """Write a consistent changed-range bookkeeping block (probe style)."""
        record["changed_ranges"] = _ranges(changed)
        record["changed_count"] = len(changed)
        record["changed_first"] = min(changed) if changed else -1
        record["changed_last"] = max(changed) if changed else -1

    def flip_first(report: dict[str, Any]) -> None:
        target = records_of(report)["sig-p0-u0-l1-b0"]
        changed = _typed_offsets(target["changed_ranges"], "flip")
        stamp(target, changed ^ {min(changed)})

    reject("signature change flip", flip_first, "column record decode mismatch")
    reject("bit effect mismatch",
           lambda r: r["bit_effects"].__setitem__("0", [5, 0, 0, 0]),
           "reported effect mismatch")
    reject("training flag false",
           lambda r: r.__setitem__("training_self_prediction_passed", False), "training flag")
    reject("hold-out flag false",
           lambda r: r.__setitem__("holdout_model_prediction_passed", False), "hold-out flag")
    reject("direct flag false",
           lambda r: r.__setitem__("direct_validation_passed", False), "direct flag")
    reject("learned weights true",
           lambda r: r.__setitem__("learned_weights_used", True), "learned weights flag")
    reject("recovered flag false",
           lambda r: r.__setitem__("matrix_permutation_recovered", False),
           "permutation flag must be true")
    reject("impulse leakage true",
           lambda r: r.__setitem__("impulse_leakage_detected", True), "impulse leakage flag")
    reject("duplicate record name",
           lambda r: r["records"].append(dict(r["records"][0])), "unique record name")
    reject("counter protocol wrong",
           lambda r: records_of(r)["base"].__setitem__("counters_observed", [2] * 8),
           "counter protocol")
    reject("guards false", lambda r: records_of(r)["base"].__setitem__("guards_intact", False),
           "guards")
    reject("immutable false", lambda r: records_of(r)["base"].__setitem__("immutable", False),
           "immutability")
    def spot_name(report: dict[str, Any]) -> str:
        names = [name for name in records_of(report) if name.startswith("direct-spot-p0-u3-l5-")]
        base.require(len(names) == 1, "synthetic spot record name")
        return names[0]

    reject("impulse-base leakage",
           lambda r: stamp(records_of(r)["impulse-base-p0-u0-l0"], {1}), "impulse-only leakage")
    reject("decoupled changed",
           lambda r: stamp(records_of(r)["decoupled-ctrl"], {1}),
           "decoupled control changed the output")
    reject("coupled-ref wrong units",
           lambda r: stamp(records_of(r)["coupled-ref"], {2}), "coupled-ref canonical unit set")
    reject("direct change set wrong",
           lambda r: stamp(records_of(r)[spot_name(r)], {9}), "re-derived verdict failed")
    reject("direct expected wrong",
           lambda r: records_of(r)[spot_name(r)].__setitem__("expected_offsets", [9]),
           "expected offsets mismatch")
    reject("direct prediction flag",
           lambda r: records_of(r)[spot_name(r)].__setitem__("prediction_match", False),
           "prediction flag")
    reject("missing training record", lambda r: r["training_records"].pop(), "training record count")
    reject("hold-out mismatches non-empty",
           lambda r: r["holdout_records"][0].__setitem__("mismatches", [[1, 2, 3]]),
           "hold-out mismatches empty")
    reject("training effect mismatch",
           lambda r: r["training_records"][0].__setitem__("fitted_effect", [9, 0, 0, 0]),
           "training effect mismatch")
    reject("changed_count wrong",
           lambda r: records_of(r)["base"].__setitem__("changed_count", 1), "changed_count mismatch")
    reject("write outside extent",
           lambda r: stamp(records_of(r)["sig-p0-u0-l1-b0"], {0, EXTENT_BYTES + 1}),
           "write outside extent")
    reject("non-int counters",
           lambda r: records_of(r)["base"].__setitem__("counters_observed", ["x"] * 8),
           "counters not int list")
    def drop_signature_kind(report: dict[str, Any]) -> None:
        # The record stays present (name in plan) but stops counting as a
        # signature run, so the decode re-derivation is the rejecting check.
        records_of(report)["sig-p0-u0-l1-b1"]["kind"] = "not-a-signature"

    reject("missing signature run", drop_signature_kind, "signature runs missing")
    # Pair controls: mutations keep each report individually valid so the
    # labelled cross-run comparison is the rejecting check. Change sets and
    # column decodes are pinned to the model by validate_report, so no
    # individually-valid mutation can alter them; those pair comparisons are
    # defence-in-depth and the tolerance paths get positive controls instead.
    pair_corruptions = 0
    pair_tolerances = 0

    def reject_pair(label: str, mutate: Any, expect: str) -> None:
        nonlocal pair_corruptions
        first, second = synthetic(), synthetic()
        mutate(second)
        try:
            validate_pair(first, second)
        except ValueError as error:
            if type(error) is not ValueError:
                raise extension.OutputMismatch(
                    f"pair corruption {label} rejected by unexpected {type(error).__name__}: {error}")
            if expect not in str(error):
                raise extension.OutputMismatch(
                    f"pair corruption {label} rejected at wrong check: {error}")
            pair_corruptions += 1
            return
        except Exception as error:  # noqa: BLE001 - a non-ValueError crash is not a clean rejection
            raise extension.OutputMismatch(
                f"pair corruption {label} rejected by unexpected {type(error).__name__}: {error}")
        raise extension.OutputMismatch(f"pair corruption not rejected: {label}")

    def accept_pair(label: str, mutate: Any) -> None:
        nonlocal pair_tolerances
        first, second = synthetic(), synthetic()
        mutate(second)
        result = validate_pair(first, second)
        base.require(result["identical"] is True, f"pair tolerance {label} rejected")
        pair_tolerances += 1

    def resplit_ranges(report: dict[str, Any]) -> None:
        """Same expanded offsets, different range encoding."""
        target = records_of(report)["sig-p0-u0-l1-b0"]
        ranges = target["changed_ranges"]
        base.require(ranges and ranges[0][1] - ranges[0][0] >= 2, "synthetic first range splittable")
        start, end = ranges[0]
        target["changed_ranges"] = [[start, start + 1], [start + 1, end]] + ranges[1:]

    def flip_sample(report: dict[str, Any]) -> None:
        target = records_of(report)["sig-p0-u0-l1-b0"]
        base.require(target.get("sample_bytes"), "synthetic samples present")
        target["sample_bytes"][0] ^= 0x01

    def flip_sample_offsets(report: dict[str, Any]) -> None:
        target = records_of(report)["sig-p0-u0-l1-b0"]
        base.require(target.get("sample_offsets"), "synthetic sample offsets present")
        target["sample_offsets"][0] = target["sample_offsets"][0] + 1

    def reorder_columns(report: dict[str, Any]) -> None:
        report["column_records"] = list(reversed(report["column_records"]))

    def reorder_records(report: dict[str, Any]) -> None:
        report["records"] = list(reversed(report["records"]))

    reject_pair("cross-run sample evidence differs", flip_sample,
                "cross-run sample bytes differ")
    reject_pair("cross-run sample offsets differ", flip_sample_offsets,
                "cross-run sample offsets differ")
    accept_pair("change-set encoding resplit", resplit_ranges)
    accept_pair("column record order", reorder_columns)
    accept_pair("record order", reorder_records)
    return dict(synthetic_report_records=len(synthetic()["records"]),
                synthetic_decodes=sum(len(d) for d in decodes.values()),
                shuffle_cross_checks=8, corruption_controls=corruptions,
                pair_controls=pair_corruptions, pair_tolerances=pair_tolerances,
                gpu_kernels_launched=0)


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
