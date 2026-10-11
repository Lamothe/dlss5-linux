#!/usr/bin/env python3
"""Bounded main-input/scratch physical layout audit; never launches a decoder.

Full inspected slices are hash-pinned in memory. Arithmetic slices are interpreted
with synthetic coordinates and compared with independently expressed byte layouts.
Channel/pixel semantics, skip/final output paths and dynamic binding remain unknown.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from typing import Any

spec = importlib.util.spec_from_file_location("address", Path(__file__).with_name("native-decoder-address-probe.py"))
if spec is None or spec.loader is None:
    raise RuntimeError("missing address helper")
address = importlib.util.module_from_spec(spec)
spec.loader.exec_module(address)
base = address.base
require = base.require
TILE_BYTES = 16384
MAX_TILES = (1 << 31) // (TILE_BYTES // 4) # Word origins are sign-extended to 64 bits.
# Complete slices, not proprietary disassembly saved to a file. Fingerprints
# reject altered instructions, but are NOT an independent numerical oracle.
SLICES = ((0, 0x14d0, "f13e26955f06551cb33dee2c070386ebbb978566924f2a347e2fcd1fe7e94b56"),
          (0x5da0, 0x6430, "3bed264b4c9ca83a4ac7ef0d7a0ac13ecb14e6b8685f3391ed9c69582c2ebdc7"))
# Small arithmetic-only subsequences. Predicates/branch reachability are scoped
# separately; unknown forms fail closed. No Windows host code is executed.
ARITHMETIC = {
    0x0110: "IMAD.SHL.U32 R12, R120, 0x80, RZ",
    0x02d0: "IMAD R2, R15, 0x100, R12",
    0x02f0: "IMAD.SHL.U32 R6, R2, 0x8, RZ",
    0x0520: "IMAD R114, R116, UR5, RZ",
    0x0550: "IMAD R5, R0, R15, R15",
    0x0560: "IMAD.IADD R5, R8, 0x1, -R5",
    0x0600: "@P1 IMAD R11, R7, 0x400, R12",
    0x0620: "@P1 IMAD R12, R0, 0x1000, R11",
    0x0c30: "IMAD R47, R7, 0x100, R121",
    0x0c50: "@P3 LEA.HI R41, R47, R120, RZ, 0x1b",
    0x0c60: "@P3 IMAD.IADD R40, R3, 0x1, R118",
    0x0c70: "@P3 IMAD R40, R40, 0x20, R41",
    0x0c90: "@P3 IMAD.SHL.U32 R43, R40, 0x80, RZ",
    0x5de0: "IMAD.IADD R5, R5, 0x1, R114",
    0x5e00: "IMAD R6, R5, 0x1000, R6",
    0x6300: "IMAD R2, R5, 0x1000, R6",
}


def padded_shape(axis0: int, axis1: int) -> tuple[int, int]:
    require(all(type(v) is int and 0 < v <= 4096 and v % 4 == 0 for v in (axis0, axis1)),
            "invalid padded shape")
    require((axis0 // 4) * (axis1 // 4) <= MAX_TILES, "signed address origin overflow")
    return axis0 // 4, axis1 // 4


def tile_index(axis0: int, axis1: int, x: int, y: int) -> tuple[int, int, int]:
    rows, columns = padded_shape(axis0, axis1)
    require(type(x) is int and type(y) is int and 0 <= x < 2 * columns and 0 <= y < rows,
            "invalid CTA coordinate")
    group, column = divmod(x, columns)
    return y * columns + column, group, rows * columns * TILE_BYTES


def main_address(tile: int, partition: int, warp: int, slab: int, lane: int) -> int:
    require(type(tile) is int and 0 <= tile < MAX_TILES, "invalid tile")
    require(all(type(v) is int and v in domain for v, domain in
                ((partition, range(4)), (warp, range(2)), (slab, (0,64,128,192)), (lane, range(32)))),
            "invalid main coordinate")
    return tile * TILE_BYTES + partition * 4096 + slab * 16 + warp * 512 + lane * 16


def scratch_address(tile: int, group: int, warp: int, lane: int, vector: int, word: int) -> int:
    require(type(tile) is int and 0 <= tile < MAX_TILES, "invalid tile")
    require(all(type(v) is int and v in domain for v, domain in
                ((group,range(2)),(warp,range(2)),(lane,range(32)),(vector,range(8)),(word,range(4)))),
            "invalid scratch coordinate")
    return tile * TILE_BYTES + group * 8192 + warp * 4096 + vector * 512 + lane * 16 + word * 4


def interpret(code: dict[int, str], offsets: tuple[int, ...], registers: dict[str, int]) -> dict[str, int]:
    """Positive-domain u32 arithmetic only, not a general SASS/CFG emulator.

    LEA.HI extracts the high word of (a << shift) + (b << 32) + c.
    All predicates on selected subsequences are assumed true here; reachability
    and pointer carry assembly are NOT simulated by this arithmetic helper.
    """
    registers = dict(registers)
    mask = (1 << 32) - 1

    def value(token: str) -> int:
        if token == "RZ":
            return 0
        if token.startswith("-"):
            return -value(token[1:])
        if re.fullmatch(r"U?R\d+", token):
            require(token in registers, "uninitialised arithmetic register")
            return registers[token]
        require(re.fullmatch(r"0x[0-9a-f]+", token) is not None, "unsupported arithmetic operand")
        return int(token, 16)

    for offset in offsets:
        require(offset in code, "missing arithmetic instruction")
        instruction = code[offset]
        if instruction.startswith("@"):
            predicate, instruction = instruction.split(" ", 1)
            require(predicate in ("@P1", "@P3"), "unsupported arithmetic predicate")
        opcode, operand_text = instruction.split(" ", 1)
        operands = operand_text.split(", ")
        require(re.fullmatch(r"R\d+", operands[0]) is not None, "invalid destination")
        if opcode in ("IMAD", "IMAD.SHL.U32", "IMAD.IADD"):
            require(len(operands) == 4, "bad multiply-add arity")
            result = value(operands[1]) * value(operands[2]) + value(operands[3])
        elif opcode == "LEA.HI":
            require(len(operands) == 5, "bad high-address arity")
            result = ((value(operands[1]) << value(operands[4])) +
                      (value(operands[2]) << 32) + value(operands[3])) >> 32
        else:
            raise ValueError("unsupported arithmetic opcode")
        registers[operands[0]] = result & mask
    return registers


def check_static(code: dict[int, str]) -> None:
    address.check_anchors(code)
    address.check_register_provenance(code)
    for lo, hi, digest in SLICES:
        require(all(offset in code for offset in range(lo, hi, 16)), "missing slice instruction")
        text = "\n".join(f"{offset:04x}:{code[offset]}" for offset in range(lo,hi,16)).encode()
        require(hashlib.sha256(text).hexdigest() == digest, "layout instruction fingerprint mismatch")
    for offset, expected in ARITHMETIC.items():
        require(code.get(offset) == expected, "layout arithmetic mismatch")


def self_test(code: dict[int, str] | None = None) -> dict[str, Any]:
    actual = ARITHMETIC if code is None else code
    comparisons = 0
    shapes = ((4,4),(4,12),(12,4),(8,20),(20,8),(28,36),(64,64))
    for a0,a1 in shapes:
        rows, columns = padded_shape(a0,a1)
        for y in range(rows):
            for x in range(2*columns):
                tile, group, extent = tile_index(a0,a1,x,y)
                registers = interpret(actual,(0x0520,0x0550,0x0560),
                                      dict(R116=columns,UR5=y,R0=columns-1,R15=group,R8=x))
                column = registers["R5"]
                require(column == x % columns and registers["R114"] == y*columns,
                        "decoded CTA arithmetic mismatch")
                row = registers["R114"] if a0 != 4 else 0
                for p in range(4):
                    for w in range(2):
                        initial = interpret(actual,(0x0110,0x0600,0x0620),
                                            dict(R120=w,R7=p,R0=row+column))
                        require(4*initial["R12"] == main_address(tile,p,w,0,0), "decoded initial main mismatch")
                        for slab in (64,128,192):
                            nxt = interpret(actual,(0x0c30,0x0c50,0x0c60,0x0c70,0x0c90),
                                            dict(R7=p,R121=slab,R120=w,R3=row,R118=column))
                            require(4*nxt["R43"] == main_address(tile,p,w,slab,0), "decoded prefetch mismatch")
                            comparisons += 1
                        require(main_address(tile,p,w,192,31)+16 <= extent, "main extent overrun")
                        comparisons += 1
                for w in range(2):
                    origin = interpret(actual,(0x0110,0x02d0,0x02f0),dict(R120=w,R15=group))
                    for store in (0x6300,0x5e00):
                        r = interpret(actual,(0x5de0,store),
                                      dict(R5=column,R114=y*columns,R6=origin["R6"]))
                        dst = "R2" if store == 0x6300 else "R6"
                        require(4*r[dst] == scratch_address(tile,group,w,0,0,0), "decoded scratch mismatch")
                        comparisons += 1
                    require(scratch_address(tile,group,w,31,7,3)+4 <= extent, "scratch extent overrun")
    # Independent contiguous-set oracles; no expected layout expression used.
    main = sorted(main_address(0,p,w,s,l) for p in range(4) for w in range(2)
                  for s in (0,64,128,192) for l in range(32))
    scratch = sorted(scratch_address(0,g,w,l,v,j) for g in range(2) for w in range(2)
                     for l in range(32) for v in range(8) for j in range(4))
    require(main == list(range(0,TILE_BYTES,16)), "main tile gap/alias")
    require(scratch == list(range(0,TILE_BYTES,4)), "scratch tile gap/alias")
    known = ((main_address(0,0,0,0,0),0),(main_address(0,3,1,192,31),16368),
             (main_address(1,0,0,0,0),16384),(scratch_address(0,1,1,31,7,3),16380))
    for observed, expected in known:
        require(observed == expected, "literal address mismatch")
    require(tile_index(8,20,9,1) == (9,1,163840), "non-square extent literal")
    require(tile_index(4096,2048,1023,1023) == (524287,1,8589934592), "maximum extent literal")
    # Synthetic arithmetic corruption must fail the numerical oracle, not merely a hash.
    wrong = ((0x0620,"@P1 IMAD R12, R0, 0x800, R11"),
             (0x0c50,"@P3 LEA.HI R41, R47, R120, RZ, 0x1a"),
             (0x02f0,"IMAD.SHL.U32 R6, R2, 0x4, RZ"))
    for offset,instruction in wrong:
        altered = dict(actual)
        altered[offset] = instruction
        if offset == 0x0620:
            r = interpret(altered,(0x0110,0x0600,0x0620),dict(R120=1,R7=2,R0=1))
            require(4*r["R12"] != main_address(1,2,1,0,0), "wrong initial arithmetic accepted")
        elif offset == 0x0c50:
            r = interpret(altered,(0x0c30,0x0c50,0x0c60,0x0c70,0x0c90),
                          dict(R7=2,R121=64,R120=1,R3=1,R118=0))
            require(4*r["R43"] != main_address(1,2,1,64,0), "wrong prefetch arithmetic accepted")
        else:
            r = interpret(altered,(0x0110,0x02d0,0x02f0),dict(R120=1,R15=1))
            require(4*r["R6"] != scratch_address(0,1,1,0,0,0), "wrong scratch arithmetic accepted")
    invalid = (lambda:tile_index(4,5,0,0),lambda:tile_index(True,4,0,0),
               lambda:tile_index(4,4,2,0),lambda:tile_index(4,4,0,1),
               lambda:main_address(0,4,0,0,0),lambda:main_address(0,0,0,256,0),
               lambda:scratch_address(0,2,0,0,0,0),lambda:scratch_address(0,0,0,0,0,4),lambda:tile_index(4096,4096,0,0),
               lambda:main_address(MAX_TILES,0,0,0,0))
    for call in invalid:
        try:
            call()
        except ValueError:
            pass
        else:
            raise ValueError("invalid layout domain accepted")
    return dict(arithmetic_comparisons=comparisons,shapes_checked=len(shapes),
                main_16_byte_reads_per_tile=len(main),scratch_4_byte_words_per_tile=len(scratch),
                literal_answers=len(known)+2,wrong_arithmetic_controls_rejected=len(wrong),
                invalid_domains_rejected=len(invalid),gpu_kernels_launched=0)


def inspect(data: bytes, negative: bool) -> dict[str, Any]:
    inventory = base.inventory(data)
    host = address.decoder.graph.disassemble(data,address.decoder.RANGES)
    address.decoder.host_contract(host)
    address.decoder.backend_contract(host)
    address.decoder.module_contract(data)
    entries = [e for e in inventory["fatbins"][6]["entries"] if e["kind"] == 2 and e["arch"] == 89]
    require(len(entries) == 1, "unique SM89 decoder module")
    code = address.disassemble(base.decompress(data,entries[0]))
    check_static(code)
    cpu = self_test(code)
    offsets = sorted({x for lo,hi,_ in SLICES for x in range(lo,hi,16)})
    if negative:
        for offset in offsets:
            altered = dict(code)
            altered[offset] = "NOP" if code[offset] != "NOP" else "EXIT"
            try:
                check_static(altered)
            except ValueError:
                pass
            else:
                raise ValueError("corrupted layout slice accepted")
        return dict(instruction_corruptions_rejected=len(offsets),decoded_arithmetic=cpu,gpu_kernels_launched=0)
    return dict(kernel=address.NAME,physical_tile_bytes=TILE_BYTES,
                conditional_main_input_extent="(axis0/4)*(axis1/4)*16384 bytes",
                conditional_scratch_extent="(axis0/4)*(axis1/4)*16384 bytes",
                main_argument_offset=0,scratch_argument_offset=48,
                main_input_datatype_candidate="FP8 E4M3",scratch_observed_access_type="packed FP16 pairs",
                scratch_initial_store_partition=0,scratch_addition_path_partitions=[1,2],
                coordinate_domain="positive padded axes <=4096; tiles<=524288 for signed word origins; group=x//(axis1/4) in {0,1}",
                quotient_instruction_equivalence_verified=False,
                scope="initial and loop main reads; non-final partition scratch stores/reductions only",
                contiguous_main_and_scratch_tiles_checked=True,decoded_arithmetic=cpu,
                intermediate_predicate_liveness_fully_verified=False,channel_pixel_semantics_verified=False,
                dynamic_buffer_binding_verified=False,skip_and_final_output_bounds_verified=False,
                all_buffer_bounds_verified=False,numerical_decoder_validation=False,gpu_kernels_launched=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("self-test","inspect","negative"))
    parser.add_argument("dll",type=Path,nargs="?")
    args = parser.parse_args()
    cpu = self_test()
    if args.mode == "self-test":
        print(json.dumps(cpu,indent=2))
        return
    require(args.dll is not None,"DLL required")
    policy = base.protect_process_memory()
    report = inspect(base.read_bounded(args.dll),args.mode == "negative")
    print(json.dumps(dict(mode=args.mode,dll_sha256=base.PIN,disassembler_sha256=address.TOOL_PIN,
                         memory_policy=policy,cpu_self_test=cpu,layout=report),indent=2))


if __name__ == "__main__":
    main()
