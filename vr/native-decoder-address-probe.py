#!/usr/bin/env python3
"""Bounded FP8 decoder coefficient-read/counter audit; no GPU execution.

Pinned DLL is data only. Disassembly remains in memory/sealed memfd, never in
reports. Swap, exec-time dumpability and same-UID /proc exposure are uncontrolled.
This does not recover activation layouts or the complete decoder calling contract.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import struct
import subprocess
from typing import Any

spec = importlib.util.spec_from_file_location("decoder", Path(__file__).with_name("native-decoder-probe.py"))
if spec is None or spec.loader is None:
    raise RuntimeError("missing decoder helper")
decoder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decoder)
base = decoder.runtime
require = base.require
TOOL = Path("/usr/local/cuda-13.4/bin/nvdisasm")
TOOL_PIN = "fa719d636e2dcebdbce466e07ab69f144feabc2fdb3eda52380fbd4a5c189faf"
NAME = "cc_dec_input_upsample_1024_512_fp8"
MATRIX_BYTES = 512 * 1024
# Bounded manually interpreted instructions, not independent semantic proof.
ANCHORS = {
    0x0010: "IMAD.MOV.U32 R14, RZ, RZ, c[0x0][0x1a4]",
    0x0020: "S2R R8, SR_CTAID.X",
    0x0050: "IADD3 R0, R14, -0x1, RZ",
    0x0060: "S2R R120, SR_TID.Y",
    0x0070: "SHF.R.S32.HI R3, RZ, 0x1f, R0",
    0x0080: "S2R R13, SR_LANEID",
    0x0090: "LEA.HI R3, R3, R0, RZ, 0x2",
    0x00a0: "SHF.R.S32.HI R0, RZ, 0x2, R3",
    0x00b0: "IADD3 R117, R0, 0x1, RZ",
    0x0100: "S2UR UR5, SR_CTAID.Y",
    0x0110: "IMAD.SHL.U32 R12, R120, 0x80, RZ",
    0x01c0: "S2R R7, SR_CTAID.Z",
    0x0240: "IMAD.MOV.U32 R5, RZ, RZ, 0x4",
    0x02d0: "IMAD R2, R15, 0x100, R12",
    0x02f0: "IMAD.SHL.U32 R6, R2, 0x8, RZ",
    0x0310: "IMAD R4, R7, 0x8000, R6",
    0x0330: "IMAD.WIDE R4, R4, R5, c[0x0][0x198]",
    0x0360: "IMAD.WIDE R4, R13, 0x10, R4",
    0x05f0: "USHF.L.U32 UR4, UR5, 0x1, URZ",
    0x08d0: "IMAD.MOV.U32 R121, RZ, RZ, RZ",
    0x0a10: "S2R R123, SR_LANEID",
    0x0a60: "IMAD.MOV.U32 R126, RZ, RZ, R121",
    0x0a90: "IADD3 R121, R121, 0x40, RZ",
    0x0b00: "ISETP.GE.U32.AND P2, PT, R126, 0x80, PT",
    0x0c30: "IMAD R47, R7, 0x100, R121",
    0x0f10: "IMAD R40, R47, 0x80, R6",
    0x0f20: "IMAD.MOV.U32 R41, RZ, RZ, 0x4",
    0x0f30: "IMAD.WIDE R40, R40, R41, c[0x0][0x198]",
    0x0f40: "IMAD.WIDE R106, R123, 0x10, R40",
    0x1130: "@!P2 BRA 0xa10",
    0x11a0: "IMAD R117, R117, UR4, R8",
    0x12b0: "IMAD.MOV.U32 R10, RZ, RZ, 0x4",
    0x13c0: "@!P0 BRA 0x6290",
    0x13d0: "S2R R8, SR_TID.Z",
    0x13e0: "LOP3.LUT P0, RZ, R9, R8, RZ, 0xfc, !PT",
    0x13f0: "@P0 BRA 0x1490",
    0x1400: "IMAD.WIDE R8, R117, R10, c[0x0][0x180]",
    0x1410: "LDG.E.NA.STRONG.GPU R12, [R8.64]",
    0x1420: "IADD3 R11, R7, -0x1, RZ",
    0x1430: "ISETP.GE.AND P0, PT, R12, R11, PT",
    0x1440: "@P0 BRA 0x1490",
    0x1450: "NANOSLEEP 0x40",
    0x1460: "LDG.E.NA.STRONG.GPU R12, [R8.64]",
    0x1470: "ISETP.GE.AND P0, PT, R12, R11, PT",
    0x1480: "@!P0 BRA 0x1450",
    0x1490: "ISETP.GE.U32.AND P0, PT, R7, 0x3, PT",
    0x14a0: "WARPSYNC 0xffffffff",
    0x14b0: "BAR.SYNC.DEFER_BLOCKING 0x0",
    0x14c0: "@!P0 BRA 0x5da0",
    0x6430: "S2R R0, SR_TID.Z",
    0x6440: "S2R R3, SR_TID.X",
    0x6450: "S2R R2, SR_TID.Y",
    0x6460: "BAR.SYNC.DEFER_BLOCKING 0x0",
    0x6470: "LOP3.LUT P0, RZ, R0, R3, R2, 0xfe, !PT",
    0x6480: "@P0 EXIT",
    0x6490: "S2R R0, SR_CTAID.Y",
    0x64a0: "ULDC UR4, c[0x0][0x1a4]",
    0x64b0: "IMAD.MOV.U32 R7, RZ, RZ, 0x4",
    0x64c0: "UIADD3 UR4, UR4, -0x1, URZ",
    0x64d0: "S2R R3, SR_CTAID.X",
    0x64e0: "USHF.R.S32.HI UR5, URZ, 0x1f, UR4",
    0x64f0: "S2R R5, SR_CTAID.Z",
    0x6500: "ULEA.HI UR5, UR5, UR4, URZ, 0x2",
    0x6510: "USHF.R.S32.HI UR5, URZ, 0x2, UR5",
    0x6520: "IMAD R0, R0, UR5, R0",
    0x6540: "IMAD R0, R0, 0x2, R3",
    0x6550: "IMAD.WIDE R2, R0, R7, c[0x0][0x180]",
    0x65a0: "MEMBAR.ALL.GPU",
    0x65b0: "ERRBAR",
    0x65c0: "STG.E.NA.STRONG.GPU [R2.64], R5",
}
OFFSETS = tuple(range(0, 0x1000, 0x200)) + tuple(range(0x4000, 0x5000, 0x200))
for i, offset in enumerate(OFFSETS):
    operand = "[R4.64]" if offset == 0 else f"[R4.64+{hex(offset)}]"
    address = (0x400, 0x420, *range(0x440, 0x520, 0x10))[i]
    ANCHORS[address] = f"LDG.E.128.STRONG.SM R{100-i*4}, {operand}"
    operand = "[R106.64]" if offset == 0 else f"[R106.64+{hex(offset)}]"
    ANCHORS[0x0f50+i*0x10] = f"LDG.E.128.STRONG.SM R{100-i*4}, {operand}"


def parse_sass(text: str) -> dict[int, str]:
    code: dict[int, str] = {}
    for line in text.splitlines():
        if "/*" not in line:
            require(not line.strip() or line.lstrip().startswith(".headerflags"), "unexpected SASS line")
            continue
        match = re.fullmatch(r"\s*/\*([0-9a-f]+)\*/\s+(.+?)\s*;\s*", line)
        require(match is not None, "malformed SASS instruction")
        address = int(match[1], 16)
        require(address not in code and address % 16 == 0, "duplicate/unaligned SASS instruction")
        code[address] = " ".join(match[2].split())
    require(bool(code) and list(code) == list(range(0, len(code)*16, 16)), "incomplete SASS stream")
    return code


def check_anchors(code: dict[int, str]) -> None:
    for address, expected in ANCHORS.items():
        require(code.get(address) == expected, f"SASS contract mismatch at {address:#x}")


def parameter_bank(info: bytes) -> None:
    # EIATTR_PARAM_CBANK supplies packed u16 base and extent, not a guessed
    # compiler convention: constant offsets 0x180/0x198 map to slots32/56.
    position = 0
    banks: list[tuple[int, int]] = []
    while position < len(info):
        form, attribute, length = base.unpack("<BBH", info, position)
        position += 4
        require(form in (1,2,3,4), "unknown metadata form")
        if form == 4:
            value = base.span(info, position, length)
            position += length
            if attribute == 0x0a:
                require(length == 8, "invalid constant bank metadata")
                _, packed = struct.unpack("<II", value)
                banks.append((packed & 0xffff, packed >> 16))
        else:
            require(attribute != 0x0a, "wrong bank attribute form")
    require(banks == [(0x160,80)], "unexpected parameter bank base/size")


def check_register_provenance(code: dict[int, str]) -> None:
    # Bounded writer audit: prevents silent register redefinitions between
    # manually inspected producers and consumers. NOT a general CFG emulator.
    intervals = (("R117",0x00b0,0x11a0,set()), ("R117",0x11a0,0x1400,set()),
                 ("UR4",0x05f0,0x11a0,set()), ("UR5",0x0100,0x11a0,set()),
                 ("R8",0x0020,0x11a0,set()), ("R7",0x01c0,0x1420,set()),
                 ("R6",0x02f0,0x0f10,set()), ("R121",0x08d0,0x0c30,{0x0a90}),
                 ("R126",0x0a60,0x0b00,set()))
    for register, start, end, allowed in intervals:
        for address, instruction in code.items():
            if not start < address < end or address in allowed:
                continue
            parts = instruction.split()
            if parts[0].startswith("@"):
                parts = parts[1:]
            if len(parts) >= 2:
                opcode, operands = parts[0], " ".join(parts[1:]).split(",")
                if opcode.split(".")[0] in ("ISETP", "FSETP", "PLOP3", "PSETP"):
                    continue  # Predicate-only writers; data operands are read-only.
                destination = operands[0].strip()
                # Some instructions have predicate output(s) before the data
                # destination: LOP3, LEA, IADD3, IMAD carry variants. Treat the
                # first register following those predicates as a destination.
                if re.fullmatch(r"!?U?P(?:\d+|T)", destination):
                    destination = next((operand.strip() for operand in operands[1:]
                                        if not re.fullmatch(r"!?U?P(?:\d+|T)", operand.strip())), "")
                written = {destination}
                if re.fullmatch(r"(?:U?R)\d+", destination):
                    width = (4 if ".128" in opcode else 2 if ".64" in opcode or
                             ".WIDE" in opcode or opcode == "CS2R" else 1)
                    prefix = "UR" if destination.startswith("UR") else "R"
                    first = int(destination[len(prefix):])
                    written = {f"{prefix}{i}" for i in range(first,first+width)}
                require(register not in written, "unexpected address register writer")


def disassemble(image: bytes) -> dict[int, str]:
    table = base.unpack("<Q", image, 40)[0]
    stride, count, names_index = base.unpack("<HHH", image, 58)
    require(stride == 64 and 0 < count <= 4096 and names_index < count, "invalid ELF sections")
    sections = [base.unpack("<IIQQQQIIQQ", image, table+i*stride) for i in range(count)]
    names = base.span(image, sections[names_index][4], sections[names_index][5])
    matches = [s for s in sections if base.string_at(names, s[0]) == ".text."+NAME]
    require(len(matches) == 1, "missing unique decoder text")
    require(matches[0][1] == 1 and matches[0][2] & 4, "decoder text is not executable PROGBITS")
    metadata = [s for s in sections if base.string_at(names, s[0]) == ".nv.info."+NAME]
    require(len(metadata) == 1, "missing unique kernel metadata")
    parameter_bank(base.span(image, metadata[0][4], metadata[0][5]))
    raw = base.span(image, matches[0][4], matches[0][5])
    require(0 < len(raw) <= base.MAX_MODULE and len(raw) % 16 == 0, "bad instruction extent")
    require(hashlib.sha256(base.read_bounded(TOOL)).hexdigest() == TOOL_PIN, "nvdisasm hash mismatch")
    fd = os.memfd_create("decoder-address-input", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        remaining = memoryview(raw)
        while remaining:
            written = os.write(fd, remaining)
            require(written > 0, "short memfd write")
            remaining = remaining[written:]
        fcntl.fcntl(fd, fcntl.F_ADD_SEALS, fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW |
                    fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
        result = subprocess.run([str(TOOL), "-b", "SM89", f"/proc/self/fd/{fd}"],
                                pass_fds=(fd,), capture_output=True, text=True, timeout=25)
        # Never emit vendor disassembly/stderr even when the tool fails.
        require(result.returncode == 0 and not result.stderr, "nvdisasm failed or produced diagnostics")
        require(len(result.stdout) <= 4*1024*1024, "excessive disassembly")
        code = parse_sass(result.stdout)
        require(len(code)*16 == len(raw), "disassembly extent mismatch")
        return code
    finally:
        os.close(fd)


def coefficient_address(partition: int, group: int, warp: int, slab: int,
                        lane: int, vector: int) -> int:
    # group=0/1 is a CONDITION, not proof of the reciprocal-division path.
    domains = ((partition, range(4)), (group, range(2)), (warp, range(2)),
               (slab, (0, 64, 128, 192)), (lane, range(32)), (vector, range(16)))
    require(all(type(v) is int and v in domain for v, domain in domains), "invalid coefficient domain")
    return 4*(partition*32768 + (group*256+warp*128)*8 + slab*128) + lane*16 + OFFSETS[vector]


def counter_slot(axis0: int, axis1: int, x: int, y: int) -> tuple[int, int]:
    # Already padded positive dimensions, bounded to prevent 32-bit arithmetic overflow.
    require(all(type(v) is int for v in (axis0, axis1, x, y)), "invalid counter types")
    require(0 < axis0 <= 4096 and 0 < axis1 <= 4096 and axis0 % 4 == axis1 % 4 == 0,
            "invalid padded counter dimensions")
    columns, rows = axis1//4, axis0//4
    require(0 <= x < 2*columns and 0 <= y < rows, "invalid grid coordinate")
    return 2*y*columns+x, 2*rows*columns


def self_test() -> dict[str, Any]:
    require(parse_sass(' /*0000*/ MOV R0, RZ ;\n /*0010*/ @P0 EXIT ;\n') ==
            {0: "MOV R0, RZ", 16: "@P0 EXIT"}, "SASS parser known answer")
    bad_parser = ('garbage', '/*0010*/ EXIT ;', '/*0000*/ EXIT ;\n/*0000*/ EXIT ;',
                  '/*0001*/ EXIT ;', '/*0000*/ EXIT')
    parameter_bank(bytes.fromhex("04 0a 08 00 0e 00 00 00 60 01 50 00"))
    bank_bad = (b"", bytes.fromhex("03 0a 60 01"),
                bytes.fromhex("04 0a 08 00 0e 00 00 00 70 01 50 00"),
                bytes.fromhex("04 0a 08 00 0e 00 00 00 60 01 50"))
    bank_rejected = 0
    for info in bank_bad:
        try:
            parameter_bank(info)
        except ValueError:
            bank_rejected += 1
    require(bank_rejected == len(bank_bad), "invalid bank metadata accepted")
    rejected = 0
    for text in bad_parser:
        try:
            parse_sass(text)
        except ValueError:
            rejected += 1
    require(rejected == len(bad_parser), "invalid SASS accepted")
    known = (((0,0,0,0,0,0),0), ((0,1,1,192,31,15),131056),
             ((3,1,1,192,31,15),524272))
    for args, expected in known:
        require(coefficient_address(*args) == expected, "coefficient literal answer")
    # Independent set oracle: sorted 16-byte reads must cover every aligned word
    # exactly once; no expected address is constructed using the layout formula.
    addresses = [coefficient_address(p,g,w,s,l,v) for p in range(4) for g in range(2)
                 for w in range(2) for s in (0,64,128,192) for l in range(32) for v in range(16)]
    require(sorted(addresses) == list(range(0, MATRIX_BYTES, 16)), "coefficient coverage/gap/alias")
    for axis0, axis1 in ((4,4),(4,12),(12,4),(8,20),(20,8)):
        slots = [counter_slot(axis0, axis1, x, y)[0] for y in range(axis0//4) for x in range(axis1//2)]
        require(slots == list(range(axis0*axis1//8)), "counter slot coverage")
    require(counter_slot(8, 20, 9, 1) == (19,20), "non-square counter literal")
    require(sorted(addresses[:-1]) != list(range(0,MATRIX_BYTES,16)), "short coverage control")
    require(sorted(addresses[1:]+[addresses[1]]) != list(range(0,MATRIX_BYTES,16)), "alias coverage control")
    require(1*5+9 != counter_slot(8,20,9,1)[0], "missing counter row factor control")
    clobbers = ((0x1000,"MOV R117, RZ"), (0x1000,"MOV UR4, RZ"),
                (0x0300,"MOV R6, RZ"), (0x1000,"IMAD.WIDE R116, RZ, RZ, RZ"),
                (0x1000,"LOP3.LUT P0, R117, RZ, RZ, RZ, 0xfa, !PT"),
                (0x1000,"CS2R R116, SRZ"), (0x1000,"LDG.E.128 R114, [R4.64]"),
                (0x1000,"IADD3 R117, P0, RZ, RZ, RZ"))
    for position, instruction in clobbers:
        try:
            check_register_provenance({position:instruction})
        except ValueError:
            pass
        else:
            raise ValueError("register clobber accepted")
    invalid = ((4,0,0,0,0,0), (0,2,0,0,0,0), (0,0,0,256,0,0), (True,0,0,0,0,0))
    invalid_count = 0
    for args in invalid:
        try:
            coefficient_address(*args)
        except ValueError:
            invalid_count += 1
    for args in ((0,4,0,0),(4,5,0,0),(4,4,2,0),(8,20,0,2),(True,4,0,0),(8192,4,0,0)):
        try:
            counter_slot(*args)
        except ValueError:
            invalid_count += 1
    require(invalid_count == 10, "invalid addressing domain accepted")
    return dict(parser_negative_controls_rejected=rejected, bank_metadata_controls_rejected=bank_rejected,
                register_clobber_controls_rejected=len(clobbers), coverage_formula_controls_rejected=3,
                invalid_domains_rejected=invalid_count,
                conditional_coefficient_reads=len(addresses), coefficient_bytes_covered=MATRIX_BYTES,
                coefficient_literal_answers=len(known), counter_shape_checks=5,
                gpu_kernels_launched=0)


def inspect(data: bytes, negative: bool) -> dict[str, Any]:
    inventory = base.inventory(data)  # Includes DLL hash validation.
    host = decoder.graph.disassemble(data, decoder.RANGES)
    host_contract = decoder.host_contract(host)  # Original 80-byte slot provenance.
    entries = [e for e in inventory["fatbins"][6]["entries"] if e["kind"] == 2 and e["arch"] == 89]
    require(len(entries) == 1, "unique SM89 decoder module")
    decoder.module_contract(data)
    code = disassemble(base.decompress(data, entries[0]))
    check_anchors(code)
    check_register_provenance(code)
    if negative:
        rejected = 0
        for address in ANCHORS:
            bad = dict(code)
            bad[address] = "NOP"
            try:
                check_anchors(bad)
            except ValueError:
                rejected += 1
        require(rejected == len(ANCHORS), "corrupted SASS accepted")
        return dict(sass_corruptions_rejected=rejected, gpu_kernels_launched=0)
    return dict(kernel=NAME, sass_anchors_checked=len(ANCHORS), argument_bytes=host_contract["argument_bytes"],
                conditional_matrix_read_bytes=MATRIX_BYTES,
                group_selector_domain_assumed=[0,1], group_division_semantics_verified=False,
                coefficient_prefetch_slabs=[0,64,128,192],
                matrix_channel_semantics_verified=False, stored_decoder_weight_binding_verified=False,
                parameter_constant_base=0x160, counter_argument_offset=32, counter_bytes_per_slot=4,
                counter_index="2*CTAID.Y*(input_axis1/4)+CTAID.X",
                counter_wait="on inspected wait path, nonzero CTAID.Z waits for value >= CTAID.Z-1",
                all_ctas_take_wait_path_verified=False,
                counter_publication="one thread per CTA writes CTAID.Z after GPU membar/errbar",
                scheduling_deadlock_freedom_verified=False, all_buffer_bounds_verified=False,
                activation_layouts_verified=False, numerical_decoder_validation=False, gpu_kernels_launched=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("self-test", "inspect", "negative"))
    parser.add_argument("dll", type=Path, nargs="?")
    args = parser.parse_args()
    cpu = self_test()
    if args.mode == "self-test":
        print(json.dumps(cpu, indent=2))
        return
    require(args.dll is not None, "DLL required")
    policy = base.protect_process_memory()
    data = base.read_bounded(args.dll)
    report = inspect(data, args.mode == "negative")
    print(json.dumps(dict(mode=args.mode, dll_sha256=base.PIN, disassembler_sha256=TOOL_PIN,
                         memory_policy=policy, cpu_self_test=cpu, addressing=report), indent=2))


if __name__ == "__main__":
    main()
