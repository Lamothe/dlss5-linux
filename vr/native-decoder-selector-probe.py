#!/usr/bin/env python3
"""Bounded decoder group-selector reconstruction; NOT decoder execution.

Static anchors tie a positive-domain reciprocal division to the pinned SM89
kernel. An owned PTX translation runs with synthetic coordinates only. No vendor
module or learned weights are uploaded. This is not an exact SASS execution test.
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


def helper(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    if spec is None or spec.loader is None:
        raise RuntimeError("missing helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


address = helper("address", "native-decoder-address-probe.py")
utility = helper("utility", "native-kernel-probe.py")
base = address.base
require = base.require
MASK = (1 << 32) - 1
GUARD = 1024
# The R2:R3 addend at 0x1a0 is a register pair: low zero, high old R3.
# Positive padded axes/CTA-X remove sign and divide-by-zero alternatives.
ANCHORS = {
    0x00c0: "IABS R5, R117.reuse", 0x00d0: "IABS R9, R8",
    0x00e0: "I2F.RP R4, R5", 0x00f0: "IABS R10, R117",
    0x0120: "MUFU.RCP R4, R4", 0x0130: "IADD3 R2, R4, 0xffffffe, RZ",
    0x0140: "IMAD.MOV R4, RZ, RZ, -R10",
    0x0150: "F2I.FTZ.U32.TRUNC.NTZ R3, R2",
    0x0160: "IMAD.MOV.U32 R2, RZ, RZ, RZ",
    0x0170: "IMAD.MOV R6, RZ, RZ, -R3", 0x0180: "IMAD R7, R6, R5, RZ",
    0x0190: "IMAD.MOV.U32 R6, RZ, RZ, R9",
    0x01a0: "IMAD.HI.U32 R3, R3, R7, R2",
    0x01b0: "S2R R9, SR_TID.X", 0x01d0: "IMAD.HI.U32 R3, R3, R6, RZ",
    0x01e0: "IMAD R2, R3, R4, R6",
    0x01f0: "ISETP.GT.U32.AND P1, PT, R5, R2, PT",
    0x0200: "@!P1 IMAD.IADD R2, R2, 0x1, -R5",
    0x0210: "@!P1 IADD3 R3, R3, 0x1, RZ",
    0x0220: "ISETP.NE.AND P1, PT, R117, RZ, PT",
    0x0230: "ISETP.GE.U32.AND P0, PT, R2, R5, PT",
    0x0250: "LOP3.LUT R2, R8, R117, RZ, 0x3c, !PT",
    0x0260: "ISETP.GE.AND P2, PT, R2, RZ, PT",
    0x0270: "@P0 IADD3 R3, R3, 0x1, RZ",
    0x0280: "LOP3.LUT P0, R9, R9, RZ, R120, 0xfa, !PT",
    0x0290: "IMAD.MOV.U32 R15, RZ, RZ, R3",
    0x02a0: "@!P2 IMAD.MOV R15, RZ, RZ, -R15",
    0x02b0: "@!P1 LOP3.LUT R15, RZ, R117, RZ, 0x33, !PT",
}

# Our arithmetic reconstruction, NOT extracted vendor code. All stores are
# index-bounded; inputs are (divisor, CTA-X). Output is (group, reciprocal bits).
PTX = b""".version 8.0
.target sm_89
.address_size 64
.visible .entry selector(.param .u64 source, .param .u64 output,
 .param .u32 count, .param .u32 wrong_divisor)
{
 .reg .pred %p<4>;
 .reg .b32 %r<25>;
 .reg .f32 %f<3>;
 .reg .b64 %rd<6>;
 ld.param.u64 %rd1, [source];
 ld.param.u64 %rd2, [output];
 ld.param.u32 %r1, [count];
 ld.param.u32 %r2, [wrong_divisor];
 mov.u32 %r3, %ctaid.x;
 mov.u32 %r4, %ntid.x;
 mov.u32 %r5, %tid.x;
 mad.lo.u32 %r6, %r3, %r4, %r5;
 setp.ge.u32 %p1, %r6, %r1;
 @%p1 bra DONE;
 mul.wide.u32 %rd3, %r6, 8;
 add.u64 %rd4, %rd1, %rd3;
 ld.global.v2.u32 {%r7, %r8}, [%rd4];
 add.u32 %r7, %r7, %r2;
 cvt.rp.f32.u32 %f1, %r7;
 rcp.approx.ftz.f32 %f2, %f1;
 mov.b32 %r9, %f2;
 add.u32 %r10, %r9, 0x0ffffffe;
 mov.b32 %f1, %r10;
 cvt.rzi.u32.f32 %r11, %f1;
 neg.s32 %r12, %r11;
 mul.lo.u32 %r13, %r12, %r7;
 mul.hi.u32 %r14, %r11, %r13;
 add.u32 %r11, %r14, %r11;
 mul.hi.u32 %r15, %r11, %r8;
 neg.s32 %r16, %r7;
 mad.lo.u32 %r17, %r15, %r16, %r8;
 setp.ge.u32 %p2, %r17, %r7;
 @%p2 sub.u32 %r17, %r17, %r7;
 @%p2 add.u32 %r15, %r15, 1;
 setp.ge.u32 %p3, %r17, %r7;
 @%p3 add.u32 %r15, %r15, 1;
 add.u64 %rd5, %rd2, %rd3;
 st.global.v2.u32 [%rd5], {%r15, %r9};
 DONE:
 ret;
}
"""


def nearest_reciprocal_bits(divisor: int) -> int:
    return struct.unpack("<I", struct.pack("<f", 1.0 / divisor))[0]


def candidate_group(divisor: int, x: int, reciprocal_bits: int) -> int:
    require(type(divisor) is int and 1 <= divisor <= 1024, "invalid divisor")
    require(type(x) is int and 0 <= x < 2 * divisor, "invalid CTA-X")
    require(type(reciprocal_bits) is int and abs(reciprocal_bits - nearest_reciprocal_bits(divisor)) <= 2,
            "reciprocal outside tested two-ULP bit interval")
    seed_float = struct.unpack("<f", struct.pack("<I", reciprocal_bits + 0x0ffffffe))[0]
    require(0 <= seed_float <= 2**32, "invalid seed conversion")
    # Float-to-u32 conversion clamps at UINT32_MAX (not wraparound).
    seed = min(int(seed_float), MASK)
    require(seed * divisor <= 2**32, "reciprocal seed overestimates inverse")
    product = ((-seed) * divisor) & MASK
    refined = (seed + ((seed * product) >> 32)) & MASK
    quotient = (refined * x) >> 32
    remainder = (x - quotient * divisor) & MASK
    if remainder >= divisor:
        remainder -= divisor
        quotient += 1
    if remainder >= divisor:
        quotient += 1
    return quotient


def check_static(code: dict[int, str]) -> None:
    address.check_anchors(code)
    address.check_register_provenance(code)
    for offset, instruction in ANCHORS.items():
        require(code.get(offset) == instruction, f"selector anchor mismatch at {offset:#x}")
    # Full straight-line slice: no unexamined branch or register writer can be
    # inserted amongst arithmetic instructions; all gaps are prior anchors.
    require(all(offset in ANCHORS or offset in address.ANCHORS
                for offset in range(0x00c0, 0x02c0, 16)), "selector slice has unchecked instruction")


def self_test() -> dict[str, Any]:
    comparisons = unsafe = 0
    for divisor in range(1, 1025):
        nearest = nearest_reciprocal_bits(divisor)
        for x in range(2 * divisor):
            for bits in range(nearest - 2, nearest + 3):
                try:
                    candidate = candidate_group(divisor, x, bits)
                except ValueError as error:
                    require(str(error) == "reciprocal seed overestimates inverse", "unexpected seed rejection")
                    unsafe += 1
                    continue
                require(candidate == x // divisor, "reciprocal reference mismatch")
                comparisons += 1
    known = ((1,0,0),(1,1,1),(3,2,0),(3,3,1),(1024,1023,0),(1024,2047,1))
    for divisor, x, expected in known:
        require(candidate_group(divisor,x,nearest_reciprocal_bits(divisor)) == expected, "literal answer")
    invalid = ((0,0,0),(1025,0,0),(True,0,0),(4,8,0),(4,-1,0),(4,True,0),
               (4,0,nearest_reciprocal_bits(4)+3))
    for args in invalid:
        try:
            candidate_group(*args)
        except ValueError:
            pass
        else:
            raise ValueError("invalid selector input accepted")
    return dict(reciprocal_interval_comparisons=comparisons, unsafe_seed_cases_rejected=unsafe, padded_axes_tested=1024,
                coordinate_cases=1024*1025, literal_answers=len(known), invalid_domains_rejected=len(invalid))


def inspect(data: bytes, negative: bool) -> dict[str, Any]:
    inventory = base.inventory(data)
    host = address.decoder.graph.disassemble(data, address.decoder.RANGES)
    contract = address.decoder.host_contract(host)
    backend = address.decoder.backend_contract(host)
    address.decoder.module_contract(data)
    entries = [e for e in inventory["fatbins"][6]["entries"] if e["kind"] == 2 and e["arch"] == 89]
    require(len(entries) == 1, "unique SM89 decoder module")
    code = address.disassemble(base.decompress(data, entries[0]))
    check_static(code)
    offsets = sorted(set(ANCHORS) | set(address.ANCHORS))
    if negative:
        for offset in offsets:
            bad = dict(code)
            bad[offset] = "NOP"
            try:
                check_static(bad)
            except ValueError:
                pass
            else:
                raise ValueError("corrupted selector contract accepted")
    require(contract["argument_bytes"] == 80 and backend["argument_bytes"] == 80
            and backend["cuda_block"] == [32, 2, 1]
            and backend["cuda_grid"] == ["2*ceil(input_axis1/4)", "ceil(input_axis0/4)", "4"]
            and backend["dynamic_shared_bytes"] == 0 and not backend["captured_runtime_binding"],
            "unexpected host/backend contract")
    return dict(sass_anchors_checked=len(offsets), corruptions_rejected=len(offsets) if negative else 0,
                positive_padded_axis1_domain=[4,4096], divisor="input_axis1/4",
                host_grid_x="2*(input_axis1/4)", group_reference="CTAID.X // (input_axis1/4)",
                group_domain=[0,1], conditional_on_positive_domain_and_reconstructed_instruction_semantics=True,
                reciprocal_seed_not_overestimating_required=True, exact_vendor_sass_executed=False,
                dynamic_backend_binding_verified=False, all_decoder_buffer_bounds_verified=False)


def execute(skip: bool, wrong: bool) -> dict[str, Any]:
    driver = utility.Driver()
    device, version = utility.select_device(driver)
    launches = readbacks = cases = 0
    rejection = False
    with ExitStack() as stack:
        context = ct.c_void_p()
        driver.check("cuCtxCreate_v2", ct.byref(context), 0, device)
        stack.callback(driver.check, "cuCtxDestroy_v2", context)
        require(bool(context.value), "null context")
        module, function = ct.c_void_p(), ct.c_void_p()
        driver.check("cuModuleLoadData", ct.byref(module), ct.create_string_buffer(PTX))
        stack.callback(driver.check, "cuModuleUnload", module)
        driver.check("cuModuleGetFunction", ct.byref(function), module, b"selector")
        require(bool(module.value) and bool(function.value), "null module/function")
        for first in range(1,1025,256):
            coordinates = [(d,x) for d in range(first,first+256) for x in range(2*d)]
            source = b"".join(struct.pack("<II",d,x) for d,x in coordinates)
            sentinel = b"\xa5" * len(source)
            prefix, suffix = b"\x37"*GUARD, b"\x6b"*GUARD
            uploaded = (prefix+source+suffix, prefix+sentinel+suffix)
            with ExitStack() as allocations:
                pointers = []
                for raw in uploaded:
                    pointer = ct.c_uint64()
                    driver.check("cuMemAlloc_v2", ct.byref(pointer), len(raw))
                    allocations.callback(driver.check, "cuMemFree_v2", pointer)
                    require(pointer.value != 0, "null allocation")
                    driver.check("cuMemcpyHtoD_v2", pointer, ct.create_string_buffer(raw), len(raw))
                    pointers.append(pointer)
                parameters = (ct.c_uint64(pointers[0].value+GUARD), ct.c_uint64(pointers[1].value+GUARD),
                              ct.c_uint(len(coordinates)), ct.c_uint(int(wrong)))
                arguments = (ct.c_void_p*4)(*(ct.addressof(p) for p in parameters))
                if not skip:
                    driver.check("cuLaunchKernel",function,(len(coordinates)+255)//256,1,1,256,1,1,0,None,arguments,None)
                    launches += 1
                    driver.check("cuCtxSynchronize")
                actual = []
                for pointer, raw in zip(pointers,uploaded):
                    host = ct.create_string_buffer(len(raw))
                    driver.check("cuMemcpyDtoH_v2",host,pointer,len(raw))
                    readbacks += 1
                    actual.append(host.raw)
                require(actual[0] == uploaded[0], "source modified")
                require(actual[1][:GUARD] == prefix and actual[1][-GUARD:] == suffix, "guard modified")
                mismatches = 0
                for (d,x),(q,bits) in zip(coordinates,struct.iter_unpack("<II",actual[1][GUARD:-GUARD])):
                    if q != x//d:
                        mismatches += 1
                    if not skip and not wrong:
                        require(candidate_group(d,x,bits) == q, "GPU/reconstructed reciprocal disagreement")
                if skip or wrong:
                    require(mismatches > 0, "negative control indistinguishable")
                    rejection = True
                else:
                    require(mismatches == 0, "GPU quotient differs from integer division")
                cases += len(coordinates)
    return dict(driver_version=version, gpu_launches=launches, gpu_readbacks=readbacks,
                coordinate_cases_checked=cases, numerical_negative_control_rejected=rejection,
                source_and_guards_unchanged=True, context_cleanup_completed=True,
                module="owned PTX arithmetic translation", vendor_decoder_kernels_launched=0,
                learned_weights_uploaded=False, complete_native_neural_image=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("self-test","inspect","negative","execute","skip","wrong-divisor"))
    parser.add_argument("dll",type=Path,nargs="?")
    args = parser.parse_args()
    policy = base.protect_process_memory()
    cpu = self_test()
    report: dict[str, Any] = dict(mode=args.mode,cpu_self_test=cpu,memory_policy=policy)
    if args.mode != "self-test":
        require(args.dll is not None,"DLL required")
        report["static"] = inspect(base.read_bounded(args.dll),args.mode == "negative")
        report["dll_sha256"] = base.PIN
        report["disassembler_sha256"] = address.TOOL_PIN
    if args.mode in ("execute","skip","wrong-divisor"):
        report["cuda"] = execute(args.mode == "skip",args.mode == "wrong-divisor")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
