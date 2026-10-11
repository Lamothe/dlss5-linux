#!/usr/bin/env python3
"""Bounded native Linux execution of three embedded utility kernels, NOT inference.

The pinned PE is read as data only. Only SM89 GPU ELF images are loaded; model
payloads are not deliberately written to files. Core dumps are disabled, but
swap/driver persistence and memory erasure are not controlled. No PE host code,
Wine or Proton runs.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import ctypes as ct
import importlib.util
import json
from pathlib import Path
import struct
from typing import Any

spec = importlib.util.spec_from_file_location("native_inventory", Path(__file__).with_name("native-runtime-probe.py"))
if spec is None or spec.loader is None:
    raise RuntimeError("missing inventory module")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


class ArrayDescriptor(ct.Structure):
    _fields_ = [("width", ct.c_size_t), ("height", ct.c_size_t), ("depth", ct.c_size_t),
                ("format", ct.c_int), ("channels", ct.c_uint), ("flags", ct.c_uint)]


class ResourceUnion(ct.Union):
    _fields_ = [("array", ct.c_void_p), ("reserved", ct.c_int * 32)]


class ResourceDescriptor(ct.Structure):
    _fields_ = [("type", ct.c_int), ("resource", ResourceUnion), ("flags", ct.c_uint)]


class TextureDescriptor(ct.Structure):
    _fields_ = [("address", ct.c_int * 3), ("filter", ct.c_int), ("flags", ct.c_uint),
                ("anisotropy", ct.c_uint), ("mipmap_filter", ct.c_int),
                ("bias", ct.c_float), ("min_level", ct.c_float), ("max_level", ct.c_float),
                ("border", ct.c_float * 4), ("reserved", ct.c_int * 12)]


class Copy2D(ct.Structure):
    _fields_ = [("sx", ct.c_size_t), ("sy", ct.c_size_t), ("stype", ct.c_int),
                ("shost", ct.c_void_p), ("sdevice", ct.c_uint64), ("sarray", ct.c_void_p),
                ("spitch", ct.c_size_t), ("dx", ct.c_size_t), ("dy", ct.c_size_t),
                ("dtype", ct.c_int), ("dhost", ct.c_void_p), ("ddevice", ct.c_uint64),
                ("darray", ct.c_void_p), ("dpitch", ct.c_size_t),
                ("width_bytes", ct.c_size_t), ("height", ct.c_size_t)]


class ClearArguments(ct.Structure):
    _fields_ = [("minimum", ct.c_uint * 2), ("maximum", ct.c_uint * 2),
                ("surface", ct.c_uint64), ("value", ct.c_float * 4)]


class CopyArguments(ct.Structure):
    _fields_ = [("texture", ct.c_uint64), ("surface", ct.c_uint64),
                ("source_start", ct.c_float * 2), ("source_span", ct.c_float * 2),
                ("coordinate_scale", ct.c_float * 2), ("destination", ct.c_float * 2),
                ("unused", ct.c_float * 4), ("extent", ct.c_int * 2)]


class CounterArguments(ct.Structure):
    _fields_ = [("pointer", ct.c_uint64), ("count", ct.c_int)]


class OutputMismatch(ValueError):
    """Only numerical output mismatches qualify for the skipped-launch control."""


def verify_abi() -> None:
    # Measured against the installed CUDA 13.4 cuda.h; all reserved fields zero.
    for kind, size in ((ArrayDescriptor, 40), (ResourceDescriptor, 144),
                       (TextureDescriptor, 104), (Copy2D, 128),
                       (ClearArguments, 40), (CopyArguments, 72), (CounterArguments, 16)):
        base.require(ct.sizeof(kind) == size, f"unexpected CUDA ABI: {kind.__name__}")
    base.require(ResourceDescriptor.resource.offset == 8 and ResourceDescriptor.flags.offset == 136,
                 "resource union alignment mismatch")
    base.require(ClearArguments.surface.offset == 16 and CopyArguments.extent.offset == 64,
                 "kernel argument offsets mismatch")


def verify_kernel_abi(image: bytes, name: str, expected_size: int) -> None:
    """Check SM89 cubin parameter metadata rather than assuming SM120 PTX ABI."""
    table = base.unpack("<Q", image, 40)[0]
    stride, count, names_index = base.unpack("<HHH", image, 58)
    base.require(stride == 64 and 0 < count <= 4096 and names_index < count, "bad ELF sections")
    sections = [base.unpack("<IIQQQQIIQQ", image, table + index * stride) for index in range(count)]
    strings = base.span(image, sections[names_index][4], sections[names_index][5])
    matches = [section for section in sections if base.string_at(strings, section[0]) == ".nv.info." + name]
    base.require(len(matches) == 1, "missing unique SM89 kernel metadata")
    info = base.span(image, matches[0][4], matches[0][5])
    position = 0
    sizes: list[int] = []
    parameters: list[tuple[int, int, int]] = []
    while position < len(info):
        form, attribute, length = base.unpack("<BBH", info, position)
        position += 4
        if form == 3:  # EIFMT_HVAL, immediate half-word.
            if attribute == 0x19:
                sizes.append(length)
        elif form == 4:  # EIFMT_SVAL, sized data.
            value = base.span(info, position, length)
            position += length
            if attribute == 0x17:
                base.require(length == 12, "unexpected kernel parameter record")
                index, ordinal_offset, packed = struct.unpack("<III", value)
                parameters.append((index, ordinal_offset, packed >> 18))
        else:
            raise ValueError("unsupported CUDA metadata format")
    base.require(sizes == [expected_size] and parameters == [(0, 0, expected_size)],
                 f"unexpected SM89 parameter layout for {name}")


class Driver(base.CUDA):
    def __init__(self) -> None:
        super().__init__()
        declarations = {
            "cuCtxSynchronize": [],
            "cuArray3DCreate_v2": [ct.POINTER(ct.c_void_p), ct.POINTER(ArrayDescriptor)],
            "cuArrayDestroy": [ct.c_void_p],
            "cuSurfObjectCreate": [ct.POINTER(ct.c_uint64), ct.POINTER(ResourceDescriptor)],
            "cuSurfObjectDestroy": [ct.c_uint64],
            "cuTexObjectCreate": [ct.POINTER(ct.c_uint64), ct.POINTER(ResourceDescriptor),
                                  ct.POINTER(TextureDescriptor), ct.c_void_p],
            "cuTexObjectDestroy": [ct.c_uint64],
            "cuMemcpy2D_v2": [ct.POINTER(Copy2D)],
            "cuMemAlloc_v2": [ct.POINTER(ct.c_uint64), ct.c_size_t],
            "cuMemFree_v2": [ct.c_uint64],
            "cuMemcpyHtoD_v2": [ct.c_uint64, ct.c_void_p, ct.c_size_t],
            "cuMemcpyDtoH_v2": [ct.c_void_p, ct.c_uint64, ct.c_size_t],
            "cuLaunchKernel": [ct.c_void_p] + [ct.c_uint] * 7 + [ct.c_void_p, ct.c_void_p, ct.c_void_p],
        }
        for name, arguments in declarations.items():
            function = getattr(self.library, name)
            function.argtypes = arguments
            function.restype = ct.c_int


def select_device(driver: Driver) -> tuple[int, int]:
    driver.check("cuInit", 0)
    version, count = ct.c_int(), ct.c_int()
    driver.check("cuDriverGetVersion", ct.byref(version))
    driver.check("cuDeviceGetCount", ct.byref(count))
    matches: list[int] = []
    for ordinal in range(count.value):
        device, major, minor = ct.c_int(), ct.c_int(), ct.c_int()
        name = ct.create_string_buffer(256)
        driver.check("cuDeviceGet", ct.byref(device), ordinal)
        driver.check("cuDeviceGetName", name, len(name), device)
        driver.check("cuDeviceGetAttribute", ct.byref(major), 75, device)
        driver.check("cuDeviceGetAttribute", ct.byref(minor), 76, device)
        if name.value == b"NVIDIA GeForce RTX 4070 Ti SUPER" and (major.value, minor.value) == (8, 9):
            matches.append(device.value)
    base.require(len(matches) == 1, "expected unique RTX 4070 Ti SUPER SM89")
    return matches[0], version.value


def pixel_data(width: int, height: int) -> list[float]:
    return [value for y in range(height) for x in range(width)
            for value in (x / 32 + 1 / 16, y / 32 + 1 / 8, ((x * 3 + y * 7) % 29) / 32, 1.0)]


def compare(label: str, actual: list[float] | list[int], expected: list[float] | list[int]) -> dict[str, Any]:
    if len(actual) != len(expected) or actual != expected:
        raise OutputMismatch(f"{label}: GPU readback differs from CPU reference")
    return dict(operation=label, checked_components=len(expected), max_absolute_error=0)


def execute(data: bytes, inventory: dict[str, Any], skip_copy: bool) -> dict[str, Any]:
    verify_abi()
    driver = Driver()
    device, version = select_device(driver)
    checks: list[dict[str, Any]] = []
    launches = 0
    readbacks = 0
    rejected = False
    rejection = ""
    with ExitStack() as stack:
        context = ct.c_void_p()
        driver.check("cuCtxCreate_v2", ct.byref(context), 0, device)
        stack.callback(driver.check, "cuCtxDestroy_v2", context)
        base.require(bool(context.value), "null CUDA context")

        def kernel(container_index: int, name: str, size: int) -> ct.c_void_p:
            container = inventory["fatbins"][container_index]
            entry = next(entry for entry in container["entries"] if entry["kind"] == 2 and entry["arch"] == 89)
            image = base.decompress(data, entry)
            base.require(name in base.elf_functions(image), "missing kernel ELF symbol")
            verify_kernel_abi(image, name, size)
            module, function = ct.c_void_p(), ct.c_void_p()
            driver.check("cuModuleLoadData", ct.byref(module), ct.create_string_buffer(image))
            stack.callback(driver.check, "cuModuleUnload", module)
            base.require(bool(module.value), "null CUDA module")
            driver.check("cuModuleGetFunction", ct.byref(function), module, name.encode("ascii"))
            base.require(bool(function.value), "null CUDA function")
            return function

        clear = kernel(12, "cuda_clear_view_kernel", 40)
        copy = kernel(14, "cg2r_copy_kernel", 72)
        counter = kernel(6, "cc_cb_clear", 16)

        def launch(function: ct.c_void_p, args: ct.Structure, grid: tuple[int, int, int],
                   block: tuple[int, int, int]) -> None:
            nonlocal launches
            pointers = (ct.c_void_p * 1)(ct.addressof(args))
            driver.check("cuLaunchKernel", function, *grid, *block, 0, None, pointers, None)
            launches += 1
            driver.check("cuCtxSynchronize")

        width, height = 35, 27
        values = pixel_data(width, height)
        source, destination = ct.c_void_p(), ct.c_void_p()
        descriptor = ArrayDescriptor(width, height, 0, 32, 4, 2)  # float4, SURFACE_LDST
        for array in (source, destination):
            driver.check("cuArray3DCreate_v2", ct.byref(array), ct.byref(descriptor))
            stack.callback(driver.check, "cuArrayDestroy", array)
            base.require(bool(array.value), "null CUDA array")

        def transfer(array: ct.c_void_p, host: Any, to_gpu: bool) -> None:
            args = Copy2D()
            args.width_bytes = width * 16
            args.height = height
            if to_gpu:
                args.stype, args.dtype = 1, 3  # host -> array
                args.shost, args.spitch, args.darray = ct.addressof(host), width * 16, array
            else:
                args.stype, args.dtype = 3, 1
                args.sarray, args.dhost, args.dpitch = array, ct.addressof(host), width * 16
            driver.check("cuMemcpy2D_v2", ct.byref(args))

        def read(array: ct.c_void_p) -> list[float]:
            nonlocal readbacks
            host = (ct.c_float * (width * height * 4))()
            transfer(array, host, False)
            readbacks += 1
            return list(host)

        transfer(source, (ct.c_float * len(values))(*values), True)
        resource = ResourceDescriptor()
        resource.resource.array = destination
        surface = ct.c_uint64()
        driver.check("cuSurfObjectCreate", ct.byref(surface), ct.byref(resource))
        stack.callback(driver.check, "cuSurfObjectDestroy", surface)
        base.require(surface.value != 0, "null surface object")
        resource.resource.array = source
        texture_desc = TextureDescriptor()
        texture_desc.address[:] = (1, 1, 1)  # clamp
        texture_desc.flags = 2  # normalized coordinates, nearest filtering
        texture = ct.c_uint64()
        driver.check("cuTexObjectCreate", ct.byref(texture), ct.byref(resource), ct.byref(texture_desc), None)
        stack.callback(driver.check, "cuTexObjectDestroy", texture)
        base.require(texture.value != 0, "null texture object")

        sentinel = [-3.5, -4.5, -5.5, -6.5] * (width * height)
        transfer(destination, (ct.c_float * len(sentinel))(*sentinel), True)
        clear_args = ClearArguments((ct.c_uint * 2)(3, 5), (ct.c_uint * 2)(31, 23), surface,
                                    (ct.c_float * 4)(0.125, 0.25, 0.5, 1.0))
        launch(clear, clear_args, ((width + 7) // 8, (height + 7) // 8, 1), (8, 8, 1))
        expected = sentinel.copy()
        for y in range(5, 23):
            for x in range(3, 31):
                expected[(y * width + x) * 4:(y * width + x) * 4 + 4] = list(clear_args.value)
        checks.append(compare("surface-clear-and-boundary-guards", read(destination), expected))

        count, guard = 267, 37
        host_counters = (ct.c_int * (count + guard * 2))(*([1234567] * (count + guard * 2)))
        pointer = ct.c_uint64()
        driver.check("cuMemAlloc_v2", ct.byref(pointer), ct.sizeof(host_counters))
        stack.callback(driver.check, "cuMemFree_v2", pointer)
        base.require(pointer.value != 0, "null device allocation")
        driver.check("cuMemcpyHtoD_v2", pointer, host_counters, ct.sizeof(host_counters))
        launch(counter, CounterArguments(pointer.value + guard * 4, count), (2, 1, 1), (256, 1, 1))
        driver.check("cuMemcpyDtoH_v2", host_counters, pointer, ct.sizeof(host_counters))
        readbacks += 1
        checks.append(compare("tile-counter-clear-and-boundary-guards", list(host_counters),
                              [1234567] * guard + [-1] * count + [1234567] * guard))

        # Two independent source/target mappings, including subrectangle guards.
        for output_width, output_height, dx, dy in ((17, 13, 2, 3), (35, 27, 0, 0)):
            transfer(destination, (ct.c_float * len(sentinel))(*sentinel), True)
            args = CopyArguments(texture, surface, (ct.c_float * 2)(0, 0), (ct.c_float * 2)(1, 1),
                                 (ct.c_float * 2)(1, 1), (ct.c_float * 2)(dx, dy),
                                 (ct.c_float * 4)(0, 0, 0, 0), (ct.c_int * 2)(output_width, output_height))
            if not skip_copy:
                launch(copy, args, ((output_width + 7) // 8, (output_height + 7) // 8, 1), (8, 8, 1))
            expected = sentinel.copy()
            for y in range(output_height):
                sy = ((2 * y + 1) * height) // (2 * output_height)
                for x in range(output_width):
                    sx = ((2 * x + 1) * width) // (2 * output_width)
                    start = ((y + dy) * width + x + dx) * 4
                    expected[start:start + 4] = values[(sy * width + sx) * 4:(sy * width + sx) * 4 + 4]
            actual = read(destination)
            label = f"texture-copy-{output_width}x{output_height}-and-guards"
            if skip_copy:
                # Only an unchanged uploaded destination is the intended control.
                # Corruption, failed reads and unexpected writes must not qualify.
                checks.append(compare("skipped-copy-sentinel-and-boundary-guards", actual, sentinel))
                try:
                    compare(label, actual, expected)
                except OutputMismatch as error:
                    rejected, rejection = True, str(error)
                base.require(rejected, "skipped copy unexpectedly passed")
                break
            checks.append(compare(label, actual, expected))
        base.require(not skip_copy or rejected, "skipped copy unexpectedly passed")
        driver.check("cuCtxSynchronize")
    # ExitStack must complete every destructor before a success JSON is emitted.
    return dict(native=True, driver_api_version=version, device="NVIDIA GeForce RTX 4070 Ti SUPER",
                kernels_launched=launches, readbacks=readbacks, checks=checks,
                skipped_copy=skip_copy, skipped_copy_rejected=rejected, rejection=rejection,
                context_destroyed=True, resources_destroyed=True, evaluations=0,
                weights_used=False, weight_files_written=False, module_image_files_written=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("execute", "skip-copy"))
    parser.add_argument("dll", type=Path)
    args = parser.parse_args()
    protection = base.protect_process_memory()
    data = base.read_bounded(args.dll)
    inventory = base.inventory(data)
    print(json.dumps(dict(sha256=base.PIN, memory_protection=protection,
                          cuda=execute(data, inventory, args.mode == "skip-copy")), indent=2))


if __name__ == "__main__":
    main()
