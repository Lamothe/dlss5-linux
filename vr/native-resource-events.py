#!/usr/bin/env python3
"""Validate metadata-only resource-event chains for a future authorised capture.

This validates schema and explicit process-local IDs only. It cannot establish
runtime GPU dataflow; it never reads tensor bytes or infers identity from pointer
values, dimensions, formats, or addresses.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = "native-resource-events-v1"
_ADDRESS_TOKEN = re.compile(r"(?:0x[0-9a-f]{6,}(?![a-z0-9])|(?<![a-z0-9])[0-9a-f]{12,}(?![a-z0-9]))", re.IGNORECASE)
EVENT_FIELDS: dict[str, set[str]] = {
    "resource_create": {"type", "resource_id", "kind", "dimension", "format"},
    "set_resource": {"type", "parameter_set_id", "key", "resource_id"},
    "copy": {"type", "command_id", "source_id", "destination_id", "subresource"},
    "interop_map": {"type", "mapping_id", "resource_id", "object_id", "mapping_type"},
    "launch": {"type", "slot", "kernel", "parameter_capture_id"},
    "launch_binding": {"type", "slot", "parameter_capture_id", "mapping_id"},
}

ID_FIELDS = {"resource_id", "parameter_set_id", "command_id", "mapping_id", "object_id", "parameter_capture_id", "source_id", "destination_id"}
VALUE_FIELDS: dict[str, dict[str, set[str]]] = {
    "resource_create": {"kind": {"upload", "texture", "buffer", "unknown"}, "dimension": {"buffer", "1d", "2d", "3d", "unknown"}, "format": {"unknown", "r16f", "r32f", "rg16f", "rgba16f", "r32u", "other"}},
    "set_resource": {"key": {"Color", "Depth", "MVec", "Output", "other"}},
    "interop_map": {"mapping_type": {"opaque", "registered", "mapped", "staging", "unknown"}},
}


def _valid_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= 96
        and value[0].isalpha()
        and _ADDRESS_TOKEN.search(value) is None
        and not all(char.isdigit() or char in "._:-" for char in value)
        and all(char.isalnum() or char in "_.:-" for char in value)
    )


def validate(document: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(document, Mapping) or document.get("schema") != SCHEMA:
        return ["schema-mismatch"]
    if document.keys() != {"schema", "content_policy", "events"}:
        errors.append("document-fields-invalid")
    if document.get("content_policy") != "metadata-only-no-tensor-bytes":
        errors.append("content-policy-mismatch")
    events = document.get("events")
    if not isinstance(events, list) or not events:
        return errors + ["events-missing"]

    # Closed event shapes and enumerated free-form values prevent this schema
    # sketch from becoming a sink for captured payloads or address-like values.
    seen: dict[str, set[str]] = {name: set() for name in ("resource_id", "mapping_id")}
    launches: set[tuple[int, str]] = set()
    bindings: set[tuple[int, str, str]] = set()
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            errors.append(f"event-{index}-invalid")
            continue
        kind = event.get("type")
        if not isinstance(kind, str) or kind not in EVENT_FIELDS:
            errors.append(f"event-{index}-invalid")
            continue
        expected_fields = EVENT_FIELDS[kind]
        if event.keys() - expected_fields:
            errors.append(f"event-{index}-unknown-fields")
        if expected_fields - event.keys():
            errors.append(f"event-{index}-missing-fields")
        for field in ID_FIELDS & event.keys():
            if not _valid_id(event[field]):
                errors.append(f"event-{index}-invalid-id")
        for field, allowed in VALUE_FIELDS.get(kind, {}).items():
            if not isinstance(event.get(field), str) or event[field] not in allowed:
                errors.append(f"event-{index}-invalid-{field}")
        if kind == "copy" and (type(event.get("subresource")) is not int or not 0 <= event["subresource"] <= 65535):
            errors.append(f"event-{index}-invalid-subresource")
        if any(error.startswith(f"event-{index}-") for error in errors):
            continue
        if kind == "resource_create":
            rid = event["resource_id"]
            if not isinstance(rid, str) or not rid or rid in seen["resource_id"]:
                errors.append(f"event-{index}-resource-id-invalid-or-duplicate")
            else:
                seen["resource_id"].add(rid)
        elif kind == "set_resource":
            if event["resource_id"] not in seen["resource_id"]:
                errors.append(f"event-{index}-resource-unseen")
        elif kind == "copy":
            if any(not isinstance(event[name], str) or event[name] not in seen["resource_id"] for name in ("source_id", "destination_id")):
                errors.append(f"event-{index}-copy-resource-unseen")
        elif kind == "interop_map":
            rid, mid = event["resource_id"], event["mapping_id"]
            if not isinstance(rid, str) or rid not in seen["resource_id"]:
                errors.append(f"event-{index}-interop-resource-unseen")
            if not isinstance(mid, str) or not mid or mid in seen["mapping_id"]:
                errors.append(f"event-{index}-mapping-id-invalid-or-duplicate")
            else:
                seen["mapping_id"].add(mid)
        elif kind == "launch":
            slot, capture = event["slot"], event["parameter_capture_id"]
            if type(slot) is not int or slot < 0 or slot > 10000 or not isinstance(event["kernel"], str) or not event["kernel"] or len(event["kernel"]) > 96 or not all(c.isalnum() or c in "_.:-" for c in event["kernel"]) or not isinstance(capture, str) or not capture:
                errors.append(f"event-{index}-launch-invalid")
            else:
                key = (slot, capture)
                if key in launches:
                    errors.append(f"event-{index}-launch-duplicate")
                else:
                    launches.add(key)
        elif kind == "launch_binding":
            slot, capture, mapping = event["slot"], event["parameter_capture_id"], event["mapping_id"]
            if type(slot) is not int or slot < 0 or slot > 10000 or not isinstance(capture, str) or not capture:
                errors.append(f"event-{index}-binding-invalid")
                continue
            launch_key = (slot, capture)
            binding_key = (slot, capture, mapping)
            if binding_key in bindings:
                errors.append(f"event-{index}-binding-duplicate")
            bindings.add(binding_key)
            if launch_key not in launches:
                errors.append(f"event-{index}-launch-unseen")
            if not isinstance(mapping, str) or mapping not in seen["mapping_id"]:
                errors.append(f"event-{index}-mapping-unseen")
    if not launches:
        errors.append("launch-missing")
    if not bindings:
        errors.append("launch-binding-missing")
    return errors


def _fixture() -> dict[str, Any]:
    return {"schema": SCHEMA, "content_policy": "metadata-only-no-tensor-bytes", "events": [
        {"type": "resource_create", "resource_id": "r-upload", "kind": "upload", "dimension": "buffer", "format": "unknown"},
        {"type": "resource_create", "resource_id": "r-texture", "kind": "texture", "dimension": "2d", "format": "rgba16f"},
        {"type": "set_resource", "parameter_set_id": "p1", "key": "Color", "resource_id": "r-texture"},
        {"type": "copy", "command_id": "c1", "source_id": "r-upload", "destination_id": "r-texture", "subresource": 0},
        {"type": "interop_map", "mapping_id": "m1", "resource_id": "r-texture", "object_id": "o1", "mapping_type": "opaque"},
        {"type": "launch", "slot": 0, "kernel": "fixture-kernel", "parameter_capture_id": "b1"},
        {"type": "launch_binding", "slot": 0, "parameter_capture_id": "b1", "mapping_id": "m1"},
    ]}


def self_test() -> int:
    good = _fixture()
    assert validate(good) == []
    cases = []
    bad = json.loads(json.dumps(good)); bad["events"][2]["resource_id"] = "missing"; cases.append((bad, "event-2-resource-unseen"))
    bad = json.loads(json.dumps(good)); bad["events"][4].pop("object_id"); cases.append((bad, "event-4-missing-fields"))
    bad = json.loads(json.dumps(good)); bad["events"][4]["mapping_id"] = "absent"; bad["events"][5]["parameter_capture_id"] = "absent"; cases.append((bad, "event-6-mapping-unseen"))
    bad = json.loads(json.dumps(good)); bad["events"][1]["device_address"] = "0x1234"; cases.append((bad, "event-1-unknown-fields"))
    bad = json.loads(json.dumps(good)); bad["events"][2]["resource_id"] = ["r-texture"]; cases.append((bad, "event-2-invalid-id"))
    bad = json.loads(json.dumps(good)); bad["events"][5]["slot"] = True; cases.append((bad, "event-5-launch-invalid"))
    bad = json.loads(json.dumps(good)); bad["events"][0]["type"] = []; cases.append((bad, "event-0-invalid"))
    bad = json.loads(json.dumps(good)); bad["events"][0]["payload"] = "forbidden"; cases.append((bad, "event-0-unknown-fields"))
    bad = json.loads(json.dumps(good)); bad["events"][0]["kind"] = {}; cases.append((bad, "event-0-invalid-kind"))
    bad = json.loads(json.dumps(good)); bad["events"][3]["subresource"] = []; cases.append((bad, "event-3-invalid-subresource"))
    bad = json.loads(json.dumps(good)); bad["events"][3]["subresource"] = 65536; cases.append((bad, "event-3-invalid-subresource"))
    bad = json.loads(json.dumps(good)); bad["unexpected"] = "value"; cases.append((bad, "document-fields-invalid"))
    bad = json.loads(json.dumps(good)); bad["events"][0]["resource_id"] = "0x123456789abcdef0"; cases.append((bad, "event-0-invalid-id"))
    bad = json.loads(json.dumps(good)); bad["events"][0]["resource_id"] = "deadbeef12345678"; cases.append((bad, "event-0-invalid-id"))
    bad = json.loads(json.dumps(good)); bad["events"][0]["resource_id"] = "resource-0x7f1234567890"; cases.append((bad, "event-0-invalid-id"))
    bad = json.loads(json.dumps(good)); bad["events"][0]["resource_id"] = "resource-deadbeef12345678"; cases.append((bad, "event-0-invalid-id"))
    bad = json.loads(json.dumps(good)); bad["events"][0]["record_id"] = ["rec-1"]; cases.append((bad, "event-0-unknown-fields"))
    bad = json.loads(json.dumps(good)); bad["events"].pop(); cases.append((bad, "launch-binding-missing"))
    bad = json.loads(json.dumps(good)); bad["events"].insert(6, json.loads(json.dumps(bad["events"][5]))); cases.append((bad, "event-6-launch-duplicate"))
    bad = json.loads(json.dumps(good)); bad["events"].append({"type": "interop_map", "mapping_id": "m1", "resource_id": "r-texture", "object_id": "o2", "mapping_type": "opaque"}); cases.append((bad, "event-7-mapping-id-invalid-or-duplicate"))
    bad = json.loads(json.dumps(good)); bad["events"].append(json.loads(json.dumps(bad["events"][6]))); cases.append((bad, "event-7-binding-duplicate"))
    valid_long_identifier = json.loads(json.dumps(good))
    valid_long_identifier["events"][2]["parameter_set_id"] = "p-0xabcdefg"
    assert validate(valid_long_identifier) == []
    multi_mapping = json.loads(json.dumps(good))
    multi_mapping["events"][4]["mapping_id"] = "m2"
    multi_mapping["events"][4]["object_id"] = "o1"
    multi_mapping["events"].insert(5, {"type": "interop_map", "mapping_id": "m1", "resource_id": "r-texture", "object_id": "o1", "mapping_type": "opaque"})
    multi_mapping["events"].append({"type": "launch_binding", "slot": 0, "parameter_capture_id": "b1", "mapping_id": "m2"})
    assert validate(multi_mapping) == []
    for case, expected in cases:
        found = validate(case)
        assert expected in found, (expected, found)
    try:
        _parse_json("[" * 2000 + "0" + "]" * 2000)
    except ValueError as exc:
        assert "nesting exceeds" in str(exc)
    else:
        raise AssertionError("deep JSON did not fail with bounded parser error")
    try:
        _parse_json("{" + "\"n\":" + "9" * 5000 + "}")
    except ValueError:
        pass
    else:
        raise AssertionError("oversized JSON integer unexpectedly accepted")
    for duplicate in ('{"schema":"x","schema":"y"}', '{"outer":{"x":1,"x":2}}'):
        try:
            _parse_json(duplicate)
        except ValueError as exc:
            assert "duplicate JSON object key" in str(exc)
        else:
            raise AssertionError("duplicate JSON keys unexpectedly accepted")
    script = Path(__file__).resolve()
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        bad_utf8 = Path(directory) / "bad-utf8.json"
        bad_utf8.write_bytes(bytes([0xFF]))
        malformed = Path(directory) / "malformed.json"
        malformed.write_text("{broken", encoding="utf-8")
        for path in (bad_utf8, malformed):
            result = subprocess.run([sys.executable, str(script), "validate", str(path)], capture_output=True, text=True, check=False)
            assert result.returncode == 2 and "input-error:" in result.stderr, (path.name, result.returncode, result.stderr)
    print(f"PASS: metadata event schema; 1 known-answer, {len(cases)} corruption controls, parser bounds/duplicate/CLI controls")
    return 0


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _parse_json(raw: str) -> Any:
    if len(raw) > 4_000_000:
        raise ValueError("input exceeds 4,000,000 characters")
    depth = 0
    in_string = False
    escaped = False
    for char in raw:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
            if depth > 128:
                raise ValueError("JSON nesting exceeds 128 levels")
        elif char in "]}":
            depth -= 1
    return json.loads(raw, object_pairs_hook=_reject_duplicate_keys)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate", "self-test"))
    parser.add_argument("path", nargs="?", type=Path)
    args = parser.parse_args()
    if args.mode == "self-test":
        return self_test()
    if args.path is None:
        parser.error("validate requires a JSON path")
    try:
        with args.path.open("rb") as source:
            raw_bytes = source.read(4_000_001)
        if len(raw_bytes) > 4_000_000:
            raise ValueError("input exceeds 4,000,000 bytes")
        result = _parse_json(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        print(f"input-error: {exc}", file=sys.stderr)
        return 2
    errors = validate(result)
    if errors:
        print("invalid: " + ", ".join(errors))
        return 1
    print("schema-valid-only: event chain is syntactically joined; no runtime dataflow is established")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
