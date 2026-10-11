#!/usr/bin/env python3
"""Local-only PE inventory / native CUDA module-load experiment, never inference.

Does not execute PE host code, launch kernels or deliberately write payload files.
Core dumps are disabled; swap, driver caching and memory erasure are not controlled.
Only the previously tested, hash-pinned community DLL is accepted by the CLI.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import os
import resource
from pathlib import Path
import struct
import stat
from typing import Any

PIN = "4b8d19bc3eff58a084f5eca7489c921501c203450169fb82ff4f649a4482ba05"
MAX_FILE = 200 * 1024 * 1024
MAX_MODULE = 32 * 1024 * 1024


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def protect_process_memory() -> dict[str, Any]:
    """Disable core dumps before input; this does not lock or erase process memory."""
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    libc = ct.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ct.c_int, ct.c_ulong, ct.c_ulong, ct.c_ulong, ct.c_ulong]
    libc.prctl.restype = ct.c_int
    require(libc.prctl(4, 0, 0, 0, 0) == 0, "PR_SET_DUMPABLE failed")
    require(libc.prctl(3, 0, 0, 0, 0) == 0, "process still dumpable")
    require(resource.getrlimit(resource.RLIMIT_CORE) == (0, 0), "core limit not disabled")
    return dict(core_soft_limit=0, core_hard_limit=0, process_dumpable=False,
                swap_prevented=False, memory_erasure_guaranteed=False,
                driver_persistence_controlled=False)



def read_bounded(path: Path, maximum: int = MAX_FILE) -> bytes:
    """One regular-file descriptor; bounded even if the path/file changes.

    Nonblocking open allows rejecting a FIFO without waiting for a writer.
    Call protect_process_memory before using this for proprietary input.
    """
    require(maximum > 0, "invalid input limit")
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    with os.fdopen(descriptor, "rb") as source:
        metadata = os.fstat(source.fileno())
        require(stat.S_ISREG(metadata.st_mode), "input is not a regular file")
        require(metadata.st_size <= maximum, "file too large")
        data = source.read(maximum + 1)
        require(len(data) <= maximum, "file grew beyond input limit")
        return data


def span(data: bytes, offset: int, size: int) -> bytes:
    require(offset >= 0 and size >= 0 and offset <= len(data) - size, "range outside input")
    return data[offset:offset + size]


def unpack(fmt: str, data: bytes, offset: int) -> tuple[Any, ...]:
    return struct.unpack(fmt, span(data, offset, struct.calcsize(fmt)))


def string_at(data: bytes, offset: int) -> str:
    require(0 <= offset < len(data), "string offset outside table")
    end = data.find(b"\0", offset)
    require(end >= offset, "unterminated string")
    return data[offset:end].decode("ascii")


class PE:
    def __init__(self, data: bytes):
        self.data = data
        require(span(data, 0, 2) == b"MZ", "missing MZ header")
        pe = unpack("<I", data, 60)[0]
        require(span(data, pe, 4) == b"PE\0\0", "missing PE header")
        machine, count = unpack("<HH", data, pe + 4)
        require(machine == 0x8664 and 0 < count <= 96, "not bounded AMD64 PE")
        optional_size = unpack("<H", data, pe + 20)[0]
        optional = pe + 24
        require(optional_size >= 136 and unpack("<H", data, optional)[0] == 0x20b,
                "not PE32+ with resource directory")
        span(data, optional, optional_size)
        require(unpack("<I", data, optional + 108)[0] >= 3, "missing resource directory entry")
        self.image_base = unpack("<Q", data, optional + 24)[0]
        self.resource_rva, self.resource_size = unpack("<II", data, optional + 128)
        self.sections: list[dict[str, Any]] = []
        for index in range(count):
            values = unpack("<8sIIIIIIHHI", data, optional + optional_size + 40 * index)
            name, virtual_size, rva, raw_size, offset = values[:5]
            span(data, offset, raw_size)
            self.sections.append(dict(name=name.rstrip(b"\0").decode("ascii"),
                                      virtual_size=virtual_size, rva=rva,
                                      size=raw_size, offset=offset))

    def offset(self, rva: int, size: int) -> int:
        matches = [section for section in self.sections
                   if section["rva"] <= rva and rva + size <= section["rva"] + section["size"]]
        require(len(matches) == 1, "RVA not backed by exactly one section")
        section = matches[0]
        return section["offset"] + rva - section["rva"]

    def resource(self, target: str) -> tuple[int, bytes]:
        root = self.offset(self.resource_rva, self.resource_size)
        data = span(self.data, root, self.resource_size)
        found: list[tuple[int, bytes]] = []
        visited: set[int] = set()

        def walk(relative: int, path: list[str | int]) -> None:
            require(len(path) < 4 and relative not in visited, "cyclic/deep resource directory")
            visited.add(relative)
            named, ids = unpack("<HH", data, relative + 12)
            require(named + ids <= 4096, "excessive resource directory")
            for index in range(named + ids):
                name, child = unpack("<II", data, relative + 16 + index * 8)
                label: str | int = name
                if name & 0x80000000:
                    location = name & 0x7fffffff
                    length = unpack("<H", data, location)[0]
                    label = span(data, location + 2, length * 2).decode("utf-16-le")
                next_path = path + [label]
                if child & 0x80000000:
                    walk(child & 0x7fffffff, next_path)
                else:
                    rva, size, _, reserved = unpack("<IIII", data, child)
                    require(reserved == 0, "reserved resource field is nonzero")
                    if next_path == [10, target, 1033]:
                        offset = self.offset(rva, size)
                        found.append((offset, span(self.data, offset, size)))

        walk(0, [])
        require(len(found) == 1, "expected unique English RCDATA resource")
        return found[0]


def weight_inventory(data: bytes) -> dict[str, Any]:
    require(unpack("<Q", data, 0)[0] == len(data), "weight container size mismatch")
    position = 8
    names: set[str] = set()
    payload_total = 0
    records: list[dict[str, Any]] = []
    while position < len(data):
        length = unpack("<Q", data, position)[0]
        position += 8
        require(0 < length <= 256, "invalid weight name length")
        name = span(data, position, length).decode("ascii")
        require(name not in names, "duplicate weight name")
        names.add(name)
        position += length
        size = unpack("<Q", data, position)[0]
        position += 8
        record = span(data, position, size)
        position += size
        require(size >= 40, "short weight record")
        declared, payload, tag = unpack("<QQI", record, 0)
        require(declared == size, "weight record size mismatch")
        span(record, 20, payload)
        a, b, dimension_count = unpack("<IIQ", record, 20 + payload)
        require(dimension_count <= 16, "excessive weight dimension count")
        dimensions = list(unpack("<" + "I" * dimension_count, record, 36 + payload))
        require(size == 36 + payload + 4 * dimension_count, "weight record trailing bytes")
        # Observed tag 1 and half-width counts are metadata, not proven datatype/layout.
        records.append(dict(name=name, bytes=size, payload_bytes=payload, tag=tag,
                            metadata_a=a, metadata_b=b, dimensions=dimensions))
        payload_total += payload
        require(len(records) <= 4096, "excessive weight record count")
    require(position == len(data), "weight records overrun container")
    return dict(resource_bytes=len(data), record_count=len(records),
                payload_bytes=payload_total, records=records)


def fatbins(data: bytes, base: int = 0) -> list[dict[str, Any]]:
    position = 0
    bins: list[dict[str, Any]] = []
    while True:
        start = data.find(b"\x50\xed\x55\xba", position)
        if start < 0:
            break
        magic, version, header, size = unpack("<IHHQ", data, start)
        require(magic == 0xba55ed50 and version == 1 and header == 16,
                "unsupported fatbinary container header")
        require(size <= MAX_FILE, "excessive fatbinary size")
        span(data, start + header, size)
        end = start + header + size
        cursor = start + header
        entries: list[dict[str, Any]] = []
        while cursor < end:
            kind, entry_version, entry_header, padded = unpack("<HHIQ", data, cursor)
            require(entry_version == 0x101 and kind in (1, 2), "unsupported fatbinary entry")
            require(64 <= entry_header <= 256, "invalid fatbinary entry header size")
            require(cursor + entry_header + padded <= end, "fatbinary entry overrun")
            # uint32 size at +16; +20 contains independent flags, not high size bits.
            compressed = unpack("<I", data, cursor + 16)[0]
            arch = unpack("<I", data, cursor + 28)[0]
            flags = unpack("<Q", data, cursor + 40)[0]
            expanded = unpack("<Q", data, cursor + 56)[0]
            require(flags & 0x8000 != 0, "expected compressed entry")
            require(0 < compressed <= padded and 0 < expanded <= MAX_MODULE,
                    "invalid compressed module bounds")
            require(span(data, cursor + entry_header, 4) == b"\x28\xb5\x2f\xfd",
                    "expected Zstandard frame")
            entries.append(dict(offset=base + cursor, kind=kind, arch=arch,
                                header_bytes=entry_header, padded_bytes=padded,
                                compressed_bytes=compressed, expanded_bytes=expanded))
            cursor += entry_header + padded
            require(len(entries) <= 32, "excessive fatbinary entries")
        require(cursor == end and len(entries) > 0, "fatbinary boundary mismatch")
        bins.append(dict(offset=base + start, bytes=header + size, entries=entries))
        position = end
        require(len(bins) <= 128, "excessive fatbinary containers")
    return bins


def decompress(data: bytes, entry: dict[str, Any]) -> bytes:
    source = span(data, entry["offset"] + entry["header_bytes"], entry["compressed_bytes"])
    zstd = ct.CDLL("libzstd.so.1")
    zstd.ZSTD_decompress.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t]
    zstd.ZSTD_decompress.restype = ct.c_size_t
    zstd.ZSTD_isError.argtypes = [ct.c_size_t]
    zstd.ZSTD_isError.restype = ct.c_uint
    output = ct.create_string_buffer(entry["expanded_bytes"])
    result = zstd.ZSTD_decompress(output, len(output), source, len(source))
    require(not zstd.ZSTD_isError(result) and result == len(output), "Zstandard decompression failed")
    return output.raw


def elf_functions(data: bytes) -> list[str]:
    require(span(data, 0, 6) == b"\x7fELF\x02\x01", "not little-endian ELF64")
    require(unpack("<H", data, 18)[0] == 190, "not EM_CUDA GPU ELF")
    table = unpack("<Q", data, 40)[0]
    stride, count, _ = unpack("<HHH", data, 58)
    require(stride == 64 and 0 < count <= 4096, "invalid ELF section table")
    sections = [unpack("<IIQQQQIIQQ", data, table + index * stride) for index in range(count)]
    functions: list[str] = []
    for section in sections:
        if section[1] != 2:
            continue
        _, _, _, _, offset, size, link, _, _, symbol_stride = section
        require(symbol_stride == 24 and size % 24 == 0 and link < count, "invalid ELF symbols")
        strings_section = sections[link]
        require(strings_section[1] == 3, "symbols do not link to string table")
        strings = span(data, strings_section[4], strings_section[5])
        symbols = span(data, offset, size)
        for index in range(0, size, 24):
            name, info, _, section_index, _, _ = unpack("<IBBHQQ", symbols, index)
            if info & 15 == 2 and info >> 4 in (1, 2) and section_index != 0:
                functions.append(string_at(strings, name))
    require(len(functions) > 0 and len(set(functions)) == len(functions), "missing/duplicate GPU functions")
    return functions


def inventory(data: bytes) -> dict[str, Any]:
    require(hashlib.sha256(data).hexdigest() == PIN, "DLL hash does not match tested RTX40 build")
    pe = PE(data)
    sections = [section for section in pe.sections if section["name"] == ".data"]
    require(len(sections) == 1, "expected unique .data section")
    section = sections[0]
    containers = fatbins(span(data, section["offset"], section["size"]), section["offset"])
    require(len(containers) == 15, "unexpected pinned fatbinary count")
    for container in containers:
        require(sorted((entry["kind"], entry["arch"]) for entry in container["entries"])
                == [(1, 120), (2, 89), (2, 120)], "unexpected pinned architectures")
    resource_offset, weights = pe.resource("WEIGHTS_HT")
    summary = weight_inventory(weights)
    require(summary["record_count"] == 153 and summary["payload_bytes"] == 147683778,
            "unexpected pinned weight counts")
    return dict(sha256=PIN, file_bytes=len(data), image_base=pe.image_base,
                sections=pe.sections, weight_resource_offset=resource_offset,
                weights=summary, fatbins=containers)


class CUDA:
    def __init__(self):
        self.library = ct.CDLL("libcuda.so.1")
        declarations = {
            "cuInit": [ct.c_uint], "cuDriverGetVersion": [ct.POINTER(ct.c_int)],
            "cuDeviceGetCount": [ct.POINTER(ct.c_int)],
            "cuDeviceGet": [ct.POINTER(ct.c_int), ct.c_int],
            "cuDeviceGetName": [ct.c_void_p, ct.c_int, ct.c_int],
            "cuDeviceGetAttribute": [ct.POINTER(ct.c_int), ct.c_int, ct.c_int],
            "cuCtxCreate_v2": [ct.POINTER(ct.c_void_p), ct.c_uint, ct.c_int],
            "cuCtxDestroy_v2": [ct.c_void_p],
            "cuModuleLoadData": [ct.POINTER(ct.c_void_p), ct.c_void_p],
            "cuModuleUnload": [ct.c_void_p],
            "cuModuleGetFunction": [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p],
            "cuGetErrorName": [ct.c_int, ct.POINTER(ct.c_char_p)],
        }
        for name, arguments in declarations.items():
            function = getattr(self.library, name)
            function.argtypes = arguments
            function.restype = ct.c_int

    def check(self, operation: str, *arguments: Any) -> None:
        result = getattr(self.library, operation)(*arguments)
        if result != 0:
            name = ct.c_char_p()
            self.library.cuGetErrorName(result, ct.byref(name))
            raise RuntimeError(f"{operation}: CUDA {result} {name.value!r}")


def load_modules(data: bytes, report: dict[str, Any], negative: bool) -> dict[str, Any]:
    cuda = CUDA()
    cuda.check("cuInit", 0)
    version, count = ct.c_int(), ct.c_int()
    cuda.check("cuDriverGetVersion", ct.byref(version))
    cuda.check("cuDeviceGetCount", ct.byref(count))
    matches: list[int] = []
    for ordinal in range(count.value):
        device, major, minor = ct.c_int(), ct.c_int(), ct.c_int()
        name = ct.create_string_buffer(256)
        cuda.check("cuDeviceGet", ct.byref(device), ordinal)
        cuda.check("cuDeviceGetName", name, len(name), device)
        cuda.check("cuDeviceGetAttribute", ct.byref(major), 75, device)
        cuda.check("cuDeviceGetAttribute", ct.byref(minor), 76, device)
        if name.value == b"NVIDIA GeForce RTX 4070 Ti SUPER" and (major.value, minor.value) == (8, 9):
            matches.append(device.value)
    require(len(matches) == 1, "expected unique RTX 4070 Ti SUPER compute capability 8.9")
    context = ct.c_void_p()
    cuda.check("cuCtxCreate_v2", ct.byref(context), 0, matches[0])
    require(bool(context.value), "CUDA success with null context")
    modules: list[dict[str, Any]] = []
    try:
        for container in report["fatbins"]:
            entry = next(entry for entry in container["entries"] if entry["kind"] == 2 and entry["arch"] == 89)
            image = decompress(data, entry)
            names = elf_functions(image)
            if negative:
                image = b"\0" * 4 + image[4:]
            buffer = ct.create_string_buffer(image)
            module = ct.c_void_p()
            result = cuda.library.cuModuleLoadData(ct.byref(module), buffer)
            if negative:
                # Only INVALID_IMAGE qualifies; no-driver/device/resource errors are not a pass.
                try:
                    require(result == 200 and not module.value,
                            f"corrupt ELF not specifically rejected as INVALID_IMAGE: {result}, {module.value}")
                finally:
                    if module.value:
                        cuda.check("cuModuleUnload", module)
                modules.append(dict(offset=entry["offset"], corrupt_rejected=True, result=result))
                break
            require(result == 0 and bool(module.value), f"native module load failed: {result}")
            try:
                for name in names:
                    function = ct.c_void_p()
                    cuda.check("cuModuleGetFunction", ct.byref(function), module, name.encode("ascii"))
                    require(bool(function.value), "CUDA success with null function")
                missing = ct.c_void_p()
                missing_result = cuda.library.cuModuleGetFunction(ct.byref(missing), module,
                                                                  b"__dlss_testbed_definitely_missing_kernel__")
                require(missing_result == 500 and not missing.value, "missing symbol not rejected as NOT_FOUND")
                modules.append(dict(offset=entry["offset"], arch=89, elf_bytes=len(image),
                                    elf_sha256=hashlib.sha256(image).hexdigest(),
                                    resolved_functions=len(names), missing_symbol_rejected=True,
                                    example_names=names[:4]))
            finally:
                cuda.check("cuModuleUnload", module)
    finally:
        cuda.check("cuCtxDestroy_v2", context)
    return dict(native=True, driver_api_version=version.value, module_count=len(modules),
                resolved_functions=sum(module.get("resolved_functions", 0) for module in modules),
                modules=modules, kernels_launched=0, evaluations=0, weight_files_written=False,
                module_image_files_written=False, context_destroyed=True, negative=negative)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inventory", "module-load", "negative"))
    parser.add_argument("dll", type=Path)
    args = parser.parse_args()
    protection = protect_process_memory()
    data = read_bounded(args.dll)
    report = inventory(data)
    if args.mode != "inventory":
        report = dict(sha256=PIN, cuda=load_modules(data, report, args.mode == "negative"))
    report["memory_protection"] = protection
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
