#!/usr/bin/env python3
"""Validate and pack established reference input planes for native adapters.

This host-side layout utility strips row padding while preserving sample bytes.
It does not define DLSS preprocessing, colour space, coordinate conventions,
or runtime/model bindings.
"""
from __future__ import annotations

REFERENCE_INPUT_LAYOUTS = {
    "input-colour.bin": (4, 2),
    "input-depth.bin": (1, 4),
    "input-motion.bin": (2, 2),
}


def pack_rows(data: bytes | bytearray | memoryview, *, width: int, height: int,
              components: int, bytes_per_component: int, row_pitch: int) -> bytes:
    """Return tight-packed rows; reject invalid dimensions, pitch, or buffer size."""
    values = (width, height, components, bytes_per_component, row_pitch)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("dimensions, component sizes and row pitch must be positive integers")
    row_bytes = width * components * bytes_per_component
    if row_pitch < row_bytes:
        raise ValueError("row pitch is smaller than the active row")
    view = memoryview(data)
    if view.ndim != 1 or view.itemsize != 1 or not view.contiguous:
        raise ValueError("input must be a contiguous byte buffer")
    expected = row_pitch * height
    if view.nbytes != expected:
        raise ValueError(f"buffer size {view.nbytes} does not match pitched size {expected}")
    if row_pitch == row_bytes:
        return bytes(view)
    packed = bytearray(row_bytes * height)
    for row in range(height):
        source = row * row_pitch
        target = row * row_bytes
        packed[target:target + row_bytes] = view[source:source + row_bytes]
    return bytes(packed)


def pack_reference_input(name: str, data: bytes | bytearray | memoryview, *,
                         width: int, height: int, row_pitch: int) -> bytes:
    """Pack a documented reference plane by its capture filename and format."""
    try:
        components, bytes_per_component = REFERENCE_INPUT_LAYOUTS[name]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unsupported reference input plane: {name!r}") from error
    return pack_rows(data, width=width, height=height, components=components,
                     bytes_per_component=bytes_per_component, row_pitch=row_pitch)
