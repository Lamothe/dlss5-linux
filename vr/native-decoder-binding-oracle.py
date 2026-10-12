#!/usr/bin/env python3
"""Independent oracle for the stored decoder-weight binding probe.

Re-derives the (o, i) -> matrix-byte closed form of
[NATIVE-DECODER-EXTENSION.md](NATIVE-DECODER-EXTENSION.md) with inline
arithmetic (no import of the binding or extension probes), re-extracts the
candidate weight record from the DLL — whose bytes are themselves verified
against the pinned SHA-256 before any prediction is derived — recomputes every
entry's nonzero product prediction and its SHA-256 digest, and checks them
against the binding probe's evidence. The probe attests kernel equality
in-sandbox; this oracle pins what "the prediction" means, independently.

Modes:

  self-test                     CPU controls with synthetic payloads.
  validate <dll> <output.json>  Re-derive and check one run; writes oracle.json
                                next to the evidence.
  pair <outputA> <outputB>      Bit-identity across two runs; writes
                                oracle-pair.json next to the first evidence.

Evidence containment: the oracle reads the DLL's weight bytes but writes only
digests, counts and verdicts — never payload byte values.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
from typing import Any


def load_helper(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    if spec is None or spec.loader is None:
        raise RuntimeError("missing helper")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


runtime = load_helper("runtime", "native-runtime-probe.py")
require = runtime.require

# ---------------------------------------------------------------------------
# Independent inline expression of the validated closed form.
# ---------------------------------------------------------------------------

SHUFFLE_STRIDES = {0: 64, 1: 8, 3: 512, 4: 128, 5: 256,
                   9: 1024, 10: 2048, 11: 4096, 12: 8192}
BIT_EFFECTS = {0: (1, 0, 0, 0), 1: (2, 0, 0, 0), 2: (0, 4, 0, 0), 3: (4, 0, 0, 0),
               4: (16, 0, 0, 0), 5: (32, 0, 0, 0), 6: (0, 0, 8, 0), 7: (0, 1, 0, 0),
               8: (0, 0, 0, 4), 9: (16384, 0, 0, 0), 10: (32768, 0, 0, 0),
               11: (65536, 0, 0, 0)}
UNIT_BYTES = (0, 1, 2, 3, 8, 9, 10, 11)
ROW_STRIDE = 8192
BLOCK_BASES = (0, 16)
EXTENT_BYTES = 16 * 16 * 512
PARTITION_BYTES = 131072
MATRIX_SLOTS = 4 * PARTITION_BYTES
SCALE_BYTES = 1024
RECORD_PAYLOAD_BYTES = 525312
CANDIDATE_RECORD = "block39.layer0.layer"
ZERO_CLASS = (0x00, 0x80)
WRONG_ROTATION = 1024
PLAN_KEYS = ((0, 1), (0, 2), (0, 3), (3, 5), (2, 15), (18, 0), (34, 9),
             (100, 7), (200, 13), (255, 15))
WRONG_MODE_KEYS = ((0, 1), (255, 15))


def shuffle(output_offset: int) -> int:
    return sum(stride * ((output_offset >> bit) & 1)
               for bit, stride in SHUFFLE_STRIDES.items())


def fed_map(unit: int, lane: int) -> dict[int, int]:
    """{output offset: partition-local matrix byte} for input byte unit*16+lane."""
    n = unit * 16 + lane
    base_value = row = block_shift = byte_shift = 0
    for bit in range(12):
        if (n >> bit) & 1:
            effect = BIT_EFFECTS[bit]
            base_value += effect[0]
            row += effect[1]
            block_shift += effect[2]
            byte_shift += effect[3]
    out: dict[int, int] = {}
    for block_base in BLOCK_BASES:
        for inner in range(8):
            block = block_shift + block_base + inner
            for column in range(16):
                for byte in UNIT_BYTES:
                    offset = row * ROW_STRIDE + column * 512 + block * 16 + byte + byte_shift
                    out[offset] = base_value + shuffle(offset)
    return out


def nonzero_digest(expected: dict[int, int]) -> tuple[str, int]:
    packed = b"".join(struct.pack("<IB", offset, byte)
                      for offset, byte in sorted(expected.items()))
    return hashlib.sha256(packed).hexdigest(), len(expected)


def offsets_digest(offsets: list[int]) -> str:
    packed = b"".join(struct.pack("<I", offset) for offset in sorted(offsets))
    return hashlib.sha256(packed).hexdigest()


def candidate_payload(data: bytes) -> tuple[bytes, dict[str, Any]]:
    pe = runtime.PE(data)
    offset, blob = pe.resource("WEIGHTS_HT")
    require(type(blob) is bytes and len(blob) > 8, "weights blob")
    position = 8
    names: set[str] = set()
    found: list[tuple[bytes, dict[str, Any]]] = []
    while position < len(blob):
        length = runtime.unpack("<Q", blob, position)[0]
        position += 8
        require(0 < length <= 256, "weight name length")
        name = runtime.span(blob, position, length).decode("ascii")
        require(name not in names, "duplicate weight name")
        names.add(name)
        position += length
        size = runtime.unpack("<Q", blob, position)[0]
        position += 8
        record = runtime.span(blob, position, size)
        position += size
        require(size >= 40, "short weight record")
        declared, payload_bytes, tag = runtime.unpack("<QQI", record, 0)
        require(declared == size, "weight record size mismatch")
        payload = runtime.span(record, 20, payload_bytes)
        metadata_a, metadata_b, dimension_count = runtime.unpack("<IIQ", record, 20 + payload_bytes)
        require(dimension_count <= 16, "excessive weight dimension count")
        dimensions = list(runtime.unpack("<" + "I" * dimension_count, record, 36 + payload_bytes))
        require(size == 36 + payload_bytes + 4 * dimension_count, "weight record trailing bytes")
        if name == CANDIDATE_RECORD:
            found.append((payload, dict(tag=tag, payload_bytes=payload_bytes,
                                        dimensions=dimensions)))
    require(position == len(blob), "weight records overrun container")
    require(len(found) == 1, "unique candidate record")
    return found[0]


def predicted(payload: bytes, partition: int, unit: int, lane: int,
              sign: str, rotation: int) -> dict[int, int]:
    flip = 0x80 if sign == "neg" else 0x00
    out: dict[int, int] = {}
    for offset, matrix_byte in fed_map(unit, lane).items():
        local = (matrix_byte + rotation) % PARTITION_BYTES if rotation else matrix_byte
        weight = payload[partition * PARTITION_BYTES + local]
        if weight in ZERO_CLASS:
            continue
        out[offset] = weight ^ flip
    return out


def recompute_metadata(payload: bytes, meta: dict[str, Any]) -> dict[str, Any]:
    matrix, tail = payload[:MATRIX_SLOTS], payload[MATRIX_SLOTS:]
    tail_fp16 = struct.unpack("<512H", tail)
    return dict(payload_bytes=meta["payload_bytes"], tag=meta["tag"],
                dimensions=meta["dimensions"],
                dimensions_arithmetic_valid=bool(
                    meta["dimensions"] == [262656] and meta["tag"] == 1
                    and meta["payload_bytes"] == RECORD_PAYLOAD_BYTES
                    and 262656 * 2 == RECORD_PAYLOAD_BYTES),
                matrix_e4m3_nan_bytes=sum(1 for byte in matrix if byte in (0x7F, 0xFF)),
                matrix_zero_class_bytes=sum(1 for byte in matrix if byte in ZERO_CLASS),
                matrix_distinct_byte_values=len(set(matrix)),
                tail_e4m3_nan_bytes=sum(1 for byte in tail if byte in (0x7F, 0xFF)),
                tail_fp16=dict(count=512,
                               nan=sum(1 for v in tail_fp16
                                       if (v & 0x7C00) == 0x7C00 and (v & 0x03FF)),
                               inf=sum(1 for v in tail_fp16
                                       if (v & 0x7C00) == 0x7C00 and not (v & 0x03FF)),
                               zeros=sum(1 for v in tail_fp16 if (v & 0x7FFF) == 0),
                               distinct=len(set(tail_fp16))))


def check_evidence(report: dict[str, Any], payload: bytes, meta: dict[str, Any]) -> dict[str, Any]:
    cuda = report["cuda"]
    problems: list[str] = []
    mode = cuda["mode"]
    if mode not in ("execute", "skip", "wrong"):
        return dict(oracle_valid=False, mode=mode, entries_checked=0,
                    metadata_checked=False, problems=[f"unknown mode {mode!r}"])
    if cuda.get("payload_byte_values_written_to_evidence") != 0:
        problems.append("evidence claims payload byte values were written")
    # Metadata is checked in every mode (the skip report carries it too).
    metadata = cuda.get("binding_metadata")
    metadata_checked = isinstance(metadata, dict)
    if not metadata_checked:
        problems.append("missing binding metadata")
    else:
        expected_metadata = recompute_metadata(payload, meta)
        for key, value in expected_metadata.items():
            if metadata.get(key) != value:
                problems.append(f"metadata mismatch for {key}")
    if mode == "skip":
        if cuda.get("out0_changed") is not False:
            problems.append("skip run reports a changed output")
        if cuda.get("decoder_launches") != 0:
            problems.append("skip run launched the decoder kernel")
        if cuda.get("kernels_launched") != 1:
            problems.append("skip run must run exactly the counter reset")
        return dict(oracle_valid=not problems, mode=mode, entries_checked=0,
                    metadata_checked=metadata_checked, problems=problems)
    records = cuda.get("records")
    if not isinstance(records, list):
        problems.append("missing records")
        records = []
    keys = WRONG_MODE_KEYS if mode == "wrong" else PLAN_KEYS
    expected_names = ["base-real"] + [f"p{partition}-u{unit}-l{lane}-{sign}"
                                      for unit, lane in keys
                                      for partition in (0, 1, 2, 3)
                                      for sign in ("pos", "neg")]
    if [record.get("name") for record in records] != expected_names:
        problems.append("entry names/order do not match the plan")
    if cuda.get("decoder_launches") != len(records):
        problems.append("decoder launch count does not match the entries")
    if cuda.get("kernels_launched") != 2 * len(records):
        problems.append("kernel launch count does not match the entries")
    if cuda.get("readbacks") != 8 * len(records) + 1:
        problems.append("readback count does not match the entries")
    rotation_mode = WRONG_ROTATION if mode == "wrong" else 0
    if cuda.get("matrix_upload_rotation") != rotation_mode:
        problems.append("matrix upload rotation does not match the mode")
    entries_checked = 0
    for record in records:
        partition, unit, lane = (record["impulse_partition"], record["impulse_unit"],
                                 record["impulse_lane"])
        if record.get("sign") not in (None, "pos", "neg") or \
                (partition is None) != (record.get("sign") is None):
            problems.append(f"invalid sign/key combination ({record['name']})")
        if type(record.get("rotation")) is not int or record["rotation"] != rotation_mode:
            problems.append(f"record rotation does not match the mode ({record['name']})")
        expect = (dict() if partition is None else
                  predicted(payload, partition, unit, lane, record["sign"], rotation_mode))
        digest, count = nonzero_digest(expect)
        if record["expected_nonzero_sha256"] != digest:
            problems.append(f"digest mismatch ({record['name']})")
        if record["expected_nonzero_count"] != count:
            problems.append(f"count mismatch ({record['name']})")
        if record["zero_class_count"] != EXTENT_BYTES - count:
            problems.append(f"zero-class count mismatch ({record['name']})")
        if record["product_identity_match"] is not True:
            problems.append(f"identity not attested ({record['name']})")
        if record["margin_poison_intact"] is not True:
            problems.append(f"margin not intact ({record['name']})")
        if record["counters_observed"] != [3] * 8:
            problems.append(f"counter protocol ({record['name']})")
        if mode == "wrong" and partition is not None:
            intended = predicted(payload, partition, unit, lane, record["sign"], 0)
            expected_differs = digest != nonzero_digest(intended)[0]
            differs_flag = record.get("intended_k0_prediction_differs")
            if type(differs_flag) is not bool or differs_flag != expected_differs:
                problems.append(f"intended prediction-differs flag mismatch ({record['name']})")
            if record.get("intended_k0_observed_rejected") is not True:
                problems.append(f"intended result not observationally rejected ({record['name']})")
        entries_checked += 1
    if mode == "execute":
        if cuda.get("weight_record_binding_verified") is not True:
            problems.append("binding verdict not true")
        if cuda.get("learned_weights_used") is not True:
            problems.append("learned-weights flag incorrect")
        if cuda.get("intended_k0_rejected") is not None or cuda.get("rotated_control_passed") is not None:
            problems.append("execute mode must not claim control verdicts")
    if mode == "wrong":
        if cuda.get("weight_record_binding_verified") is not False:
            problems.append("wrong mode must not claim binding")
        if cuda.get("rotated_control_passed") is not True:
            problems.append("rotated control not passed")
        if cuda.get("intended_k0_rejected") is not True:
            problems.append("wrong mode did not reject the intended k0 result")
    return dict(oracle_valid=not problems, mode=mode, entries_checked=entries_checked,
                metadata_checked=metadata_checked, problems=problems)


def self_test() -> dict[str, Any]:
    rejected = 0
    problems: list[str] = []

    def check(condition: Any, label: str) -> None:
        if not condition:
            problems.append(label)

    # Inline model known answers.
    identity = fed_map(0, 0)
    check(len(identity) == 2048, "identity map size")
    check(all(matrix_byte == shuffle(offset) for offset, matrix_byte in identity.items()),
          "identity map is pure shuffle")
    check(all(matrix_byte == 1 + shuffle(offset) for offset, matrix_byte in fed_map(0, 1).items()),
          "bit-0 map shifted by one")
    full = fed_map(255, 15)
    check(max(full.values()) == 114743 + max(shuffle(offset) for offset in full),
          "all-bits base composition")
    check(all(0 <= offset < EXTENT_BYTES for offset in full), "composed offsets in extent")
    check(all(0 <= matrix_byte < PARTITION_BYTES for matrix_byte in full.values()),
          "composed matrix bytes in partition")
    # Prediction helpers on synthetic payloads.
    uniform = bytes([0x38]) * RECORD_PAYLOAD_BYTES
    expect = predicted(uniform, 0, 0, 1, "pos", 0)
    check(len(expect) == 2048, "uniform prediction size")
    check(all(byte == 0x38 for byte in expect.values()), "uniform prediction values")
    mirrored = predicted(uniform, 0, 0, 1, "neg", 0)
    check(all(mirrored[offset] == (byte ^ 0x80) for offset, byte in expect.items()),
          "negative sign mirror")
    check(predicted(bytes(RECORD_PAYLOAD_BYTES), 0, 0, 1, "pos", 0) == {},
          "zero payload predicts silence")
    varied = bytes(((g * 131 + 7) % 255) + 1 for g in range(RECORD_PAYLOAD_BYTES))
    check(predicted(varied, 0, 0, 1, "pos", 0) != predicted(varied, 0, 0, 2, "pos", 0),
          "distinct input bases give distinct predictions")
    check(predicted(varied, 0, 0, 1, "pos", WRONG_ROTATION)
          != predicted(varied, 0, 0, 1, "pos", 0),
          "rotation changes predictions")
    # Digest controls.
    digest_a = nonzero_digest({3: 0x01, 7: 0x38})[0]
    check(digest_a == nonzero_digest({7: 0x38, 3: 0x01})[0], "digest order stability")
    check(digest_a != nonzero_digest({3: 0x01, 7: 0x39})[0], "digest value sensitivity")
    check(offsets_digest([2, 1]) == offsets_digest([1, 2]), "offset digest order stability")
    # Evidence checker controls with synthetic evidence built over the FULL
    # plan (plan pinning rejects any truncated report, so single-record
    # fixtures would make every control vacuous).
    synthetic_meta = dict(tag=1, payload_bytes=RECORD_PAYLOAD_BYTES, dimensions=[262656])
    metadata_block = recompute_metadata(varied, synthetic_meta)

    def build_records(payload_: bytes, mode: str) -> list[dict[str, Any]]:
        keys = WRONG_MODE_KEYS if mode == "wrong" else PLAN_KEYS
        rotation = WRONG_ROTATION if mode == "wrong" else 0
        records = [dict(name="base-real", impulse_partition=None, impulse_unit=None,
                        impulse_lane=None, sign=None, rotation=rotation,
                        expected_nonzero_sha256=nonzero_digest({})[0],
                        expected_nonzero_count=0, zero_class_count=EXTENT_BYTES,
                        product_identity_match=True, margin_poison_intact=True,
                        counters_observed=[3] * 8)]
        for unit, lane in keys:
            for partition in (0, 1, 2, 3):
                for sign in ("pos", "neg"):
                    expect = predicted(payload_, partition, unit, lane, sign, rotation)
                    record = dict(name=f"p{partition}-u{unit}-l{lane}-{sign}",
                                  impulse_partition=partition, impulse_unit=unit,
                                  impulse_lane=lane, sign=sign, rotation=rotation,
                                  expected_nonzero_sha256=nonzero_digest(expect)[0],
                                  expected_nonzero_count=len(expect),
                                  zero_class_count=EXTENT_BYTES - len(expect),
                                  product_identity_match=True, margin_poison_intact=True,
                                  counters_observed=[3] * 8)
                    if mode == "wrong":
                        intended = predicted(payload_, partition, unit, lane, sign, 0)
                        record["intended_k0_prediction_differs"] = \
                            nonzero_digest(expect)[0] != nonzero_digest(intended)[0]
                        record["intended_k0_observed_rejected"] = True
                    records.append(record)
        return records

    def must_reject(report: dict[str, Any], label: str) -> None:
        nonlocal rejected
        verdict = check_evidence(report, varied, synthetic_meta)
        if verdict["oracle_valid"]:
            problems.append(f"corrupt control accepted ({label})")
        else:
            rejected += 1

    def execute_report(mutate: Any = None, **overrides: Any) -> dict[str, Any]:
        records = build_records(varied, "execute")
        if mutate is not None:
            mutate(records[1])
        cuda = dict(mode="execute", binding_metadata=metadata_block, records=records,
                    decoder_launches=len(records), kernels_launched=2 * len(records),
                    readbacks=8 * len(records) + 1, matrix_upload_rotation=0,
                    weight_record_binding_verified=True, learned_weights_used=True,
                    intended_k0_rejected=None, rotated_control_passed=None,
                    payload_byte_values_written_to_evidence=0)
        cuda.update(overrides)
        return dict(cuda=cuda)

    check(check_evidence(execute_report(), varied, synthetic_meta)["oracle_valid"],
          "synthetic good evidence accepted")
    mutations = (
        ("digest", lambda r: r.update(expected_nonzero_sha256="0" * 64)),
        ("count", lambda r: r.update(expected_nonzero_count=r["expected_nonzero_count"] + 1)),
        ("zero-class count", lambda r: r.update(zero_class_count=0)),
        ("identity flag", lambda r: r.update(product_identity_match=False)),
        ("margin flag", lambda r: r.update(margin_poison_intact=False)),
        ("counters", lambda r: r.update(counters_observed=[3] * 7)),
        ("sign", lambda r: r.update(sign="neg")),
        ("invalid sign", lambda r: r.update(sign="foo")),
        ("record rotation", lambda r: r.update(rotation=WRONG_ROTATION)),
        ("entry name", lambda r: r.update(name="bogus")),
    )
    for label, mutate in mutations:
        must_reject(execute_report(mutate), label)
    must_reject(execute_report(matrix_upload_rotation=WRONG_ROTATION),
                "upload rotation claim")
    must_reject(execute_report(decoder_launches=82), "decoder launch count")
    must_reject(execute_report(kernels_launched=163), "kernel launch count")
    must_reject(execute_report(readbacks=650), "readback count")
    must_reject(execute_report(payload_byte_values_written_to_evidence=1),
                "payload byte-value claim")
    must_reject(execute_report(intended_k0_rejected=True),
                "execute claims intended verdict")
    must_reject(execute_report(rotated_control_passed=True),
                "execute claims rotated verdict")
    must_reject(execute_report(
        binding_metadata=dict(metadata_block, matrix_e4m3_nan_bytes=1)),
        "metadata mutation")
    must_reject(dict(cuda=dict(execute_report()["cuda"], mode="bogus")), "unknown mode")
    # Wrong-mode controls (rotation pinned to 1,024).
    def wrong_report(mutate: Any = None, **overrides: Any) -> dict[str, Any]:
        records = build_records(varied, "wrong")
        if mutate is not None:
            mutate(records[1])
        cuda = dict(mode="wrong", binding_metadata=metadata_block, records=records,
                    decoder_launches=len(records), kernels_launched=2 * len(records),
                    readbacks=8 * len(records) + 1, matrix_upload_rotation=WRONG_ROTATION,
                    weight_record_binding_verified=False, rotated_control_passed=True,
                    intended_k0_rejected=True, learned_weights_used=True,
                    payload_byte_values_written_to_evidence=0)
        cuda.update(overrides)
        return dict(cuda=cuda)

    check(check_evidence(wrong_report(), varied, synthetic_meta)["oracle_valid"],
          "synthetic wrong-mode evidence accepted")
    must_reject(wrong_report(lambda r: r.update(intended_k0_prediction_differs=False)),
                "wrong prediction-differs flag")
    must_reject(wrong_report(lambda r: r.update(intended_k0_observed_rejected=False)),
                "wrong observed-rejection flag")
    must_reject(wrong_report(intended_k0_rejected=None), "wrong intended verdict missing")
    must_reject(wrong_report(intended_k0_rejected=False), "wrong intended verdict false")
    must_reject(wrong_report(weight_record_binding_verified=True), "wrong claims binding")
    must_reject(wrong_report(rotated_control_passed=False), "wrong control not passed")
    must_reject(wrong_report(matrix_upload_rotation=0), "wrong upload rotation claim")
    # Skip controls (metadata present and checked).
    skip_cuda = dict(mode="skip", out0_changed=False, kernels_launched=1,
                     decoder_launches=0, binding_metadata=metadata_block,
                     payload_byte_values_written_to_evidence=0)
    check(check_evidence(dict(cuda=skip_cuda), varied, synthetic_meta)["oracle_valid"],
          "skip evidence accepted")
    must_reject(dict(cuda=dict(skip_cuda, out0_changed=True)), "skip with output change")
    must_reject(dict(cuda=dict(skip_cuda, decoder_launches=1)), "skip with decoder launch")
    must_reject(dict(cuda=dict(skip_cuda, kernels_launched=0)), "skip without reset")
    must_reject(dict(cuda=dict(skip_cuda, payload_byte_values_written_to_evidence=1)),
                "skip payload byte-value claim")
    must_reject(dict(cuda=dict(skip_cuda, binding_metadata=dict(
        metadata_block, matrix_zero_class_bytes=0))), "skip metadata mutation")
    return dict(model_known_answers=8, digest_controls=3,
                corrupt_controls_rejected=rejected, problems=problems,
                oracle_valid=not problems, gpu_kernels_launched=0)


def validate(dll: Path, evidence: Path) -> dict[str, Any]:
    data = dll.read_bytes()
    require(hashlib.sha256(data).hexdigest() == runtime.PIN,
            "DLL bytes do not match the pinned hash")
    report = json.loads(evidence.read_text())
    require(report.get("dll_sha256") == runtime.PIN,
            "evidence must pin the verified DLL hash")
    payload, meta = candidate_payload(data)
    verdict = check_evidence(report, payload, meta)
    out = evidence.parent / "oracle.json"
    out.write_text(json.dumps(dict(oracle="binding", dll=dll.name,
                                   evidence=evidence.name, **verdict), indent=2) + "\n")
    return verdict


def pair(first: Path, second: Path) -> dict[str, Any]:
    left = json.loads(first.read_text())
    right = json.loads(second.read_text())
    problems: list[str] = []
    if left["cuda"]["mode"] != right["cuda"]["mode"]:
        problems.append("mode mismatch")
    anchor_keys = ("name", "expected_nonzero_sha256", "expected_nonzero_count",
                   "zero_class_count", "zero_0x80_variant_count",
                   "zero_0x80_variant_offsets_sha256", "counters_observed",
                   "product_identity_match", "margin_poison_intact",
                   "counters_guard_intact", "immutable_regions_intact",
                   "guards_intact", "rotation", "sign")
    anchors_left = [{k: r.get(k) for k in anchor_keys}
                    for r in left["cuda"].get("records", [])]
    anchors_right = [{k: r.get(k) for k in anchor_keys}
                     for r in right["cuda"].get("records", [])]
    if anchors_left != anchors_right:
        problems.append("entry anchors differ")
    verdict = dict(oracle="binding-pair", pair_identical=not problems, problems=problems,
                   entries_compared=len(anchors_left))
    out = first.parent / "oracle-pair.json"
    out.write_text(json.dumps(verdict, indent=2) + "\n")
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("self-test", "validate", "pair"))
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    if args.mode == "self-test":
        print(json.dumps(self_test(), indent=2))
        return
    if args.mode == "validate":
        require(len(args.paths) == 2, "validate requires <dll> <output.json>")
        verdict = validate(*args.paths)
    else:
        require(len(args.paths) == 2, "pair requires <outputA> <outputB>")
        verdict = pair(*args.paths)
    print(json.dumps(verdict, indent=2))
    require(verdict.get("oracle_valid", verdict.get("pair_identical")) is True,
            "oracle verdict failed")


if __name__ == "__main__":
    main()
