#!/usr/bin/env python3
"""Native FP8 projection -> owned layout bridge -> spatial repack -> inverse.

This is a constructed compatible chain, NOT the recovered model graph or image
inference. Read the pinned DLL as data only. Never persist proprietary payloads;
swap/driver persistence and read-only out-of-bounds access are not controlled.
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

spec = importlib.util.spec_from_file_location("projection", Path(__file__).with_name("native-projection-probe.py"))
if spec is None or spec.loader is None:
    raise RuntimeError("missing projection probe")
projection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(projection)
utility = projection.utility
base = projection.base
CHANNELS = 1024
GUARD_BYTES = 1024
HEIGHT = WIDTH = 16
TOKENS = HEIGHT * WIDTH
MODES = ("execute", "skip-projection", "skip-bridge", "skip-repack", "wrong-bridge")

# Our own PTX, not extracted vendor instructions. One byte per logical (token,
# channel). Map projection-output packing to MMA-input packing. Wrong-layout
# control copies physical bytes instead, in bounds, through the same entry.
BRIDGE_PTX = b""".version 8.0
.target sm_89
.address_size 64
.visible .entry layout_bridge(
 .param .u64 source, .param .u64 destination,
 .param .u32 elements, .param .u32 identity_control)
{
 .reg .pred %p<3>;
 .reg .b32 %r<30>;
 .reg .b64 %rd<6>;
 ld.param.u64 %rd1, [source];
 ld.param.u64 %rd2, [destination];
 ld.param.u32 %r1, [elements];
 ld.param.u32 %r2, [identity_control];
 mov.u32 %r3, %ctaid.x;
 mov.u32 %r4, %ntid.x;
 mov.u32 %r5, %tid.x;
 mad.lo.u32 %r6, %r3, %r4, %r5;
 setp.ge.u32 %p1, %r6, %r1;
 @%p1 bra DONE;
 // token = index/1024, channel = index%1024
 shr.u32 %r7, %r6, 10;
 and.b32 %r8, %r6, 1023;
 // shared base: (t/16)*16384 + (c/32)*512 + ((t%16)/8)*4
 //              + ((c%32)/16)*8 + (t%8)*64
 shr.u32 %r9, %r7, 4;
 shl.b32 %r9, %r9, 14;
 shr.u32 %r10, %r8, 5;
 shl.b32 %r10, %r10, 9;
 add.u32 %r9, %r9, %r10;
 and.b32 %r11, %r7, 8;
 shr.u32 %r11, %r11, 1;
 add.u32 %r9, %r9, %r11;
 and.b32 %r12, %r8, 16;
 shr.u32 %r12, %r12, 1;
 add.u32 %r9, %r9, %r12;
 and.b32 %r13, %r7, 7;
 shl.b32 %r13, %r13, 6;
 add.u32 %r9, %r9, %r13;
 // source: ((c%8)/2)*16 + ((c%16)/8)*2 + c%2
 and.b32 %r14, %r8, 6;
 shl.b32 %r14, %r14, 3;
 and.b32 %r15, %r8, 8;
 shr.u32 %r15, %r15, 2;
 and.b32 %r16, %r8, 1;
 add.u32 %r17, %r9, %r14;
 add.u32 %r17, %r17, %r15;
 add.u32 %r17, %r17, %r16;
 // destination: ((c%16)/4)*16 + c%4
 and.b32 %r18, %r8, 12;
 shl.b32 %r18, %r18, 2;
 and.b32 %r19, %r8, 3;
 add.u32 %r20, %r9, %r18;
 add.u32 %r20, %r20, %r19;
 setp.ne.u32 %p2, %r2, 0;
 selp.u32 %r17, %r6, %r17, %p2;
 selp.u32 %r20, %r6, %r20, %p2;
 cvt.u64.u32 %rd3, %r17;
 cvt.u64.u32 %rd4, %r20;
 add.u64 %rd3, %rd1, %rd3;
 add.u64 %rd4, %rd2, %rd4;
 ld.global.u8 %r21, [%rd3];
 st.global.u8 [%rd4], %r21;
 DONE:
 ret;
}
"""


def spatial_index(y: int, x: int, channel: int, width: int) -> int:
    """FP8 MMA fragment in a 4x4 spatial tile, byte address."""
    tile = (y // 4) * (width // 4) + x // 4
    position = (y % 4) * 4 + x % 4
    # Explicit fragment coordinates; spatial addressing differs from linear t.
    return (tile * 16384 + (channel // 32) * 512 + (position // 8) * 4
            + ((channel % 32) // 16) * 8 + (position % 8) * 64
            + ((channel % 16) // 4) * 16 + channel % 4)


def bridge_reference(projected: bytes) -> bytes:
    result = bytearray(len(projected))
    for token in range(TOKENS):
        for channel in range(CHANNELS):
            result[projection.fp8_activation_index(token, channel)] = projected[projection.fp8_output_index(token, channel)]
    return bytes(result)


def spatial_reference(linear: bytes) -> bytes:
    result = bytearray(len(linear))
    for y in range(HEIGHT):
        for x in range(WIDTH):
            for channel in range(CHANNELS):
                result[spatial_index(y, x, channel, WIDTH)] = linear[projection.fp8_activation_index(y * WIDTH + x, channel)]
    return bytes(result)


def self_test() -> dict[str, Any]:
    tensor = projection.tensor
    base.require(ct.sizeof(tensor.RepackArguments) == 24
                 and tensor.RepackArguments.source.offset == 0
                 and tensor.RepackArguments.destination.offset == 8
                 and tensor.RepackArguments.height.offset == 16
                 and tensor.RepackArguments.width.offset == 20, "repack host ABI mismatch")
    base.require(sorted(spatial_index(y, x, c, WIDTH) for y in range(HEIGHT)
                        for x in range(WIDTH) for c in range(CHANNELS))
                 == list(range(TOKENS * CHANNELS)), "non-bijective spatial map")
    # Literal coordinates independently enumerated across tile/fragment edges.
    answers = ((0, 0, 0, 0), (0, 0, 2, 2), (0, 0, 4, 16),
               (0, 1, 0, 64), (1, 0, 0, 256), (2, 0, 0, 4),
               (0, 4, 0, 16384), (4, 0, 0, 65536), (15, 15, 1023, 262143))
    for y, x, c, expected in answers:
        base.require(spatial_index(y, x, c, WIDTH) == expected, "spatial known answer mismatch")
    # Asymmetric channel/position pattern; not a round-trip-only oracle.
    fixture = bytes((i * 37 + i // 256 * 19) % 256 for i in range(TOKENS * CHANNELS))
    bridged = bridge_reference(fixture)
    spatial = spatial_reference(bridged)
    base.require(bridged != fixture and spatial != bridged, "negative controls indistinguishable")
    base.require(bridged[2] == fixture[16] and bridged[16] == fixture[32], "bridge literal answers mismatch")
    base.require(spatial[16384] == bridged[256], "spatial literal payload mismatch")
    rejected = 0
    for wrong in (fixture, b"\x7f" * len(fixture)):
        try:
            projection.exact("CPU wrong bridge", wrong, bridged)
        except utility.OutputMismatch:
            rejected += 1
    base.require(rejected == 2, "CPU controls accepted")
    return dict(spatial_layout_bijective=True, spatial_known_answers=len(answers),
                bridge_known_answers=2, spatial_payload_known_answers=1,
                cpu_negative_controls_rejected=rejected)


def execute(data: bytes, mode: str, block: int) -> dict[str, Any]:
    cpu = {**projection.fp8_self_test(), **self_test()}
    learned = projection.projection_record(data, block)
    driver = utility.Driver()
    device, version = utility.select_device(driver)
    checks: list[dict[str, Any]] = []
    launches = readbacks = checked_outputs = 0
    rejection = False
    with ExitStack() as stack:
        context = ct.c_void_p()
        driver.check("cuCtxCreate_v2", ct.byref(context), 0, device)
        stack.callback(driver.check, "cuCtxDestroy_v2", context)
        base.require(bool(context.value), "null context")
        inventory = base.inventory(data)
        entry = next(e for e in inventory["fatbins"][5]["entries"] if e["kind"] == 2 and e["arch"] == 89)
        image = base.decompress(data, entry)
        projection.verify_projection_abi(image, "cc_vit_1d_projection_fp8")
        forward = "cc_vit_1d_repack_1d_to_2d_fp8"
        backward = "cc_vit_1d_repack_2d_to_1d_fp8"
        for name in (forward, backward):
            utility.verify_kernel_abi(image, name, 24)
        functions: dict[str, ct.c_void_p] = {}
        for module_image, names in ((image, ("cc_vit_1d_projection_fp8", forward, backward)),
                                    (BRIDGE_PTX, ("layout_bridge",))):
            module = ct.c_void_p()
            driver.check("cuModuleLoadData", ct.byref(module), ct.create_string_buffer(module_image))
            stack.callback(driver.check, "cuModuleUnload", module)
            base.require(bool(module.value), "null module")
            for name in names:
                function = ct.c_void_p()
                driver.check("cuModuleGetFunction", ct.byref(function), module, name.encode("ascii"))
                base.require(bool(function.value), "null function")
                functions[name] = function
        for batch in range(4 if mode == "execute" else 1):
            uploads, expected, _, scratch, _ = projection.fp8_fixtures(batch, 0, learned=learned)
            intended_bridge = bridge_reference(expected)
            intended_spatial = spatial_reference(intended_bridge)
            uploads += [b"\x7f" * len(expected)] * 3  # bridge, spatial, returned linear
            with ExitStack() as allocations:
                pointers: list[ct.c_uint64] = []
                originals: list[bytes] = []
                for index, upload in enumerate(uploads):
                    pre = bytes((i * 13 + index * 37 + 1) % 256 for i in range(GUARD_BYTES))
                    post = bytes((i * 17 + index * 41 + 7) % 256 for i in range(GUARD_BYTES))
                    raw = pre + upload + post
                    pointer = ct.c_uint64()
                    driver.check("cuMemAlloc_v2", ct.byref(pointer), len(raw))
                    allocations.callback(driver.check, "cuMemFree_v2", pointer)
                    base.require(pointer.value != 0, "null allocation")
                    driver.check("cuMemcpyHtoD_v2", pointer, ct.create_string_buffer(raw), len(raw))
                    pointers.append(pointer)
                    originals.append(raw)
                active = [p.value + GUARD_BYTES for p in pointers]
                args = projection.ProjectionArguments()
                for field, address in zip(("source", "residual", "destination", "weights", "counters"), active):
                    setattr(args, field, address)
                args.unused[0] = active[5]
                args.height, args.width = HEIGHT, WIDTH
                if mode != "skip-projection":
                    params = (ct.c_void_p * 1)(ct.addressof(args))
                    driver.check("cuLaunchKernel", functions["cc_vit_1d_projection_fp8"], 16, 1, 4,
                                 32, 4, 1, 0, None, params, None)
                    launches += 1
                if mode != "skip-bridge":
                    values = (ct.c_uint64(active[2]), ct.c_uint64(active[6]),
                              ct.c_uint32(TOKENS * CHANNELS), ct.c_uint32(mode == "wrong-bridge"))
                    params = (ct.c_void_p * 4)(*(ct.addressof(v) for v in values))
                    driver.check("cuLaunchKernel", functions["layout_bridge"], TOKENS * CHANNELS // 256, 1, 1,
                                 256, 1, 1, 0, None, params, None)
                    launches += 1
                # All launches share the default stream; there is no intermediate
                # GPU->CPU->GPU tensor copy or per-stage context synchronisation.
                for name, src, dst in ((forward, 6, 7), (backward, 7, 8)):
                    if name == forward and mode == "skip-repack":
                        continue
                    repack_args = projection.tensor.RepackArguments(active[src], active[dst], HEIGHT, WIDTH)
                    params = (ct.c_void_p * 1)(ct.addressof(repack_args))
                    driver.check("cuLaunchKernel", functions[name], 64, 1, 1, 256, 1, 1, 0, None, params, None)
                    launches += 1
                driver.check("cuCtxSynchronize")
                actual = []
                for pointer, original in zip(pointers, originals):
                    host = ct.create_string_buffer(len(original))
                    driver.check("cuMemcpyDtoH_v2", host, pointer, len(original))
                    readbacks += 1
                    actual.append(host.raw)
                def check(label: str, got: bytes, wanted: bytes) -> None:
                    checks.append(projection.exact(f"batch{batch}-{label}", got, wanted, redact=True))
                for i in (0, 1, 3):
                    check(f"input-{i}-unchanged", actual[i], originals[i])
                for i in (2, 4, 5, 6, 7, 8):
                    check(f"pre-{i}", actual[i][:GUARD_BYTES], originals[i][:GUARD_BYTES])
                    check(f"post-{i}", actual[i][-GUARD_BYTES:], originals[i][-GUARD_BYTES:])
                payload = [raw[GUARD_BYTES:-GUARD_BYTES] for raw in actual]
                projection_expected = uploads[2] if mode == "skip-projection" else expected
                bridge_expected = (uploads[6] if mode == "skip-bridge" else projection_expected
                                   if mode == "wrong-bridge" else bridge_reference(projection_expected))
                spatial_expected = uploads[7] if mode == "skip-repack" else spatial_reference(bridge_expected)
                returned_expected = uploads[8] if mode in ("skip-projection", "skip-bridge", "skip-repack") else bridge_expected
                check("projected-output", payload[2], projection_expected)
                check("projection-counters", payload[4], uploads[4] if mode == "skip-projection" else struct.pack("<16i", *([3] * 16)))
                check("projection-scratch", payload[5], uploads[5] if mode == "skip-projection" else scratch)
                check("bridge-output", payload[6], bridge_expected)
                check("spatial-output", payload[7], spatial_expected)
                check("returned-linear-output", payload[8], returned_expected)
                if mode == "execute":
                    check("spatial-independent-intended", payload[7], intended_spatial)
                    checked_outputs += len(expected) * 4
                else:
                    try:
                        projection.exact("intended-chain", payload[8], intended_bridge, redact=True)
                    except utility.OutputMismatch:
                        rejection = True
                    base.require(rejection, "incorrect/skipped stage accepted")
    return dict(native=True, device="NVIDIA GeForce RTX 4070 Ti SUPER", driver_api_version=version,
                mode=mode, candidate_record=f"block{block}.layer4.layer", cpu_checks=cpu,
                kernels_launched=launches, readbacks=readbacks, checks=checks,
                checked_stage_output_bytes=checked_outputs, candidate_payload_uploaded_unchanged=True,
                own_gpu_layout_bridge=True, intermediate_host_tensor_transfers=0,
                constructed_chain_verified=mode == "execute", control_rejected=rejection,
                actual_model_graph_linkage_verified=False, neural_image_evaluations=0,
                context_destroyed=True, resources_destroyed=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("self-test", *MODES))
    parser.add_argument("dll", nargs="?", type=Path)
    parser.add_argument("--block", type=int, choices=range(31, 39), default=31)
    args = parser.parse_args()
    if args.mode == "self-test":
        print(json.dumps(self_test(), indent=2))
        return
    base.require(args.dll is not None, "DLL required")
    protection = base.protect_process_memory()
    data = base.read_bounded(args.dll)
    print(json.dumps(dict(sha256=base.PIN, memory_protection=protection,
                         cuda=execute(data, args.mode, args.block)), indent=2))


if __name__ == "__main__":
    main()
