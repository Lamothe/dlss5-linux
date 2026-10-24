#!/usr/bin/env python3
"""Validate and pack bounded row-pitched image planes for native input adapters.

This host-side layout utility does not define DLSS preprocessing, colour space,
coordinate conventions, or model bindings. It only strips row padding while
preserving each row's bytes, for formats and dimensions established separately.
"""
from __future__ import annotations


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
