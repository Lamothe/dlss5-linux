"""CPU-only tests for row-pitch packing; no model or GPU semantics implied."""
import importlib.util
from pathlib import Path

source = Path(__file__).resolve().parents[1] / "native-input-layout.py"
spec = importlib.util.spec_from_file_location("native_input_layout", source)
assert spec is not None and spec.loader is not None
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)

# Two 3-pixel rows, RGBA8, with two padding bytes per row.
source_bytes = bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0xEE, 0xEE,
                      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 0xDD, 0xDD])
expected = source_bytes[:12] + source_bytes[14:26]
assert layout.pack_rows(source_bytes, width=3, height=2, components=4,
                        bytes_per_component=1, row_pitch=14) == expected
assert layout.pack_rows(expected, width=3, height=2, components=4,
                        bytes_per_component=1, row_pitch=12) == expected

# Reference capture protocol: synthetic 720×720 tightly packed planes; check byte sizes.
for name, components, bytes_per_component, size in (
        ("input-colour.bin", 4, 2, 4_147_200),
        ("input-depth.bin", 1, 4, 2_073_600),
        ("input-motion.bin", 2, 2, 2_073_600)):
    assert layout.REFERENCE_INPUT_LAYOUTS[name] == (components, bytes_per_component)
    assert 720 * 720 * components * bytes_per_component == size
    packed_plane = layout.pack_reference_input(name, bytes(size), width=720,
                                                height=720, row_pitch=720 * components * bytes_per_component)
    assert len(packed_plane) == size
try:
    layout.pack_reference_input("output-frame00.bin", b"", width=1, height=1, row_pitch=8)
except ValueError as error:
    assert "unsupported" in str(error)
else:
    raise AssertionError("unrecognised plane accepted")

cases = [
    ({"width": 0}, "positive"),
    ({"height": -1}, "positive"),
    ({"components": True}, "positive"),
    ({"bytes_per_component": 0}, "positive"),
    ({"row_pitch": 11}, "smaller"),
    ({"row_pitch": 14}, "does not match"),  # too short for two pitched rows
]
base = dict(width=3, height=2, components=4, bytes_per_component=1, row_pitch=14)
for change, message in cases:
    args = base | change
    payload = source_bytes[:-1] if change == {"row_pitch": 14} else source_bytes
    try:
        layout.pack_rows(payload, **args)
    except ValueError as error:
        assert message in str(error), (change, error)
    else:
        raise AssertionError(f"invalid layout accepted: {change}")

for malformed in (memoryview(bytearray(28))[::2], memoryview(bytearray(56)).cast("H")):
    try:
        layout.pack_rows(malformed, **base)
    except ValueError:
        pass
    else:
        raise AssertionError("non-byte contiguous buffer accepted")
print("PASS: reference input schemas, pitched-row packing, tight plane sizes and 9 invalid-layout controls")
