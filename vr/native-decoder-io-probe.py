#!/usr/bin/env python3
"""Bounded static decoder skip/output addressing audit; never launches a decoder.

Recovers physical byte addressing for the final-partition skip reads, output
stores, partial-sum reads and weight word reads of the pinned SM89
`cc_dec_input_upsample_1024_512_fp8` kernel. Constant-bank slot usage is
fingerprinted per path region and tied to the full kernel extent. Pointer
formation chains are checked contiguous: no intervening writer between the
decoded formation stages and their consumer load/store. No CUDA context is
created. The working Windows/DX12/Proton VR PoC and existing probes remain
unchanged.

This is not a complete decoder contract, a captured runtime binding, semantic
channel proof, output buffer extent proof or a native neural-rendered image.
Index register origins are validated as decoded arithmetic only; loop and
coordinate bounds, and multi-path control flow, remain conditional.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

spec = importlib.util.spec_from_file_location("address", Path(__file__).with_name("native-decoder-address-probe.py"))
if spec is None or spec.loader is None:
    raise RuntimeError("missing address helper")
address = importlib.util.module_from_spec(spec)
spec.loader.exec_module(address)
base = address.base
require = base.require

KERNEL_BYTES = 0x6680
# Complete inspected slices over the index derivation, store groups and each
# pointer-formation chain. Fingerprints reject altered instructions; they are
# NOT semantic validation.
SLICES = ((0x1620, 0x1750, "b7819bd2a7e6271ead7abd69cd73cf60acdeb1488b1e17c511d2afee856d71c7"),
          (0x1d20, 0x1e60, "92ac2b461d42ba449c69dd74f864d389d6bca661d20d2f2c452844e5d27ff317"),
          (0x2140, 0x21b0, "b804d7d3cec59d2220a93652977db25e3c497b9ad6ef5c0f8879ac0057a650db"),
          (0x22b0, 0x2400, "beea8be406b605ed2a90808ea493d8b1fae87a66b776b04770e9750ae80be81f"),
          (0x27b0, 0x2840, "415f839eba876ca08d54bff9407f878576d7ebe49a8058ef41b817a3883b241a"),
          (0x2e20, 0x2f40, "70d23b0c519a1f694058d413cfd6033a574adc9774c83ff541d8c1bf9ba3f1a9"),
          (0x31a0, 0x31c0, "a54510ccf6ba2f5e5a97a4b86c35d3023b596350fb3516c1a13fca2394aa81af"),
          (0x45c0, 0x4a00, "49fd3c96aaa85de708184661c07ab992eb900409316b7a56093d244c80961787"),
          (0x4c00, 0x5da0, "faff4e62d69ba182cc8936cbe387bcf5e535d2311b103f8c5f1a76fd36af12d4"))
# Bounded manually interpreted subsequences; predicates assumed true where the
# decoded stream runs them. Unknown forms fail closed. No Windows code executes.
ARITHMETIC = {
    0x12b0: "IMAD.MOV.U32 R10, RZ, RZ, 0x4",
    0x20a0: "IMAD.MOV.U32 R144, RZ, RZ, 0x4",
    0x4ca0: "IMAD.MOV.U32 R44, RZ, RZ, 0x4",
    0x4c30: "S2R R45, SR_LANEID",
    0x4620: "IADD3 R0, R9, 0x1, RZ",
    0x4660: "IMAD R0, R8, R23, R23",
    0x46d0: "IMAD R20, R8, R23, R9",
    0x46e0: "IMAD.SHL.U32 R23, R2, 0x4, RZ",
    0x4750: "IMAD.IADD R0, R9, 0x1, R0",
    0x4860: "IMAD R127, R20, 0x800, R23",
    0x4970: "IMAD R0, R0, 0x800, R23",
    0x4d70: "IMAD.WIDE R46, R127, R44, c[0x0][0x170]",
    0x4f10: "IMAD.WIDE R48, R45, 0x10, R46",
    0x5340: "IMAD.WIDE R46, R45, 0x10, R46",
    0x5540: "IMAD.WIDE R10, R0, R44, c[0x0][0x170]",
    0x57a0: "IMAD.WIDE R2, R45, 0x10, R10",
    0x5b80: "IMAD.WIDE R10, R45, 0x10, R10",
    0x16b0: "@P3 IMAD.WIDE R6, R7, R10, c[0x0][0x190]",
    0x1700: "@P3 IMAD.WIDE R6, R119, 0x10, R6",
    0x1740: "@P3 LDG.E.128.STRONG.SM R40, [R6.64]",
    0x1d30: "@P2 IMAD.WIDE R80, R81, R10, c[0x0][0x168]",
    0x1e00: "@P2 IMAD.WIDE R80, R119, 0x10, R80",
    0x1e50: "@P2 LDG.E.128.STRONG.SM R72, [R80.64]",
    0x2150: "IMAD.WIDE R98, R99, R144.reuse, c[0x0][0x198]",
    0x21a0: "LDG.E R13, [R98.64+0x80000]",
    0x22c0: "@P4 IMAD.WIDE R98, R99, R144, c[0x0][0x168]",
    0x2390: "@P4 IMAD.WIDE R98, R129, 0x10, R98",
    0x23f0: "@P4 LDG.E.128.STRONG.SM R84, [R98.64]",
    0x27c0: "IMAD.WIDE R134, R3, R144.reuse, c[0x0][0x198]",
    0x2830: "LDG.E R10, [R134.64+0x80000]",
    0x2e30: "@P0 IMAD.WIDE R146, R147, R144, c[0x0][0x168]",
    0x2f00: "@P0 IMAD.WIDE R146, R129, 0x10, R146",
    0x2f30: "@P0 LDG.E.128.STRONG.SM R40, [R146.64]",
    0x31b0: "@P0 IMAD.WIDE R144, R145, R144, c[0x0][0x168]",
    0x3370: "@P0 IMAD.WIDE R144, R129, 0x10, R144",
}
# Pointer chains: (stage1 offset, base register, stage2 offset or None, load
# offset, kind). kind fixes the consumer's form: read128 = 16-byte read at
# pointer+0; word32 = 4-byte read at pointer+0x80000. The chain must be
# contiguous: no writer of the base register between stages and the load.
CHAINS = ((0x16b0, "R6", 0x1700, 0x1740, "read128"),
          (0x1d30, "R80", 0x1e00, 0x1e50, "read128"),
          (0x22c0, "R98", 0x2390, 0x23f0, "read128"),
          (0x2e30, "R146", 0x2f00, 0x2f30, "read128"),
          (0x2150, "R98", None, 0x21a0, "word32"),
          (0x27c0, "R134", None, 0x2830, "word32"))
LOAD_FORMS = {
    "read128": r"@P[0-6] LDG\.E\.128\.STRONG\.SM R\d+, \[{base}\.64\]",
    "word32": r"LDG\.E R\d+, \[{base}\.64\+0x80000\]",
}
# The sixteen final-partition output stores: base register -> slot.
STORE_ORDER = (0x4f60, 0x4ff0, 0x5040, 0x5090, 0x53a0, 0x5490, 0x54e0, 0x5530,
               0x5800, 0x58c0, 0x5910, 0x5960, 0x5bf0, 0x5ce0, 0x5d30, 0x5d80)
STORES = {
    0x4f60: ("R48", 0x0000), 0x4ff0: ("R48", 0x0200),
    0x5040: ("R48", 0x0400), 0x5090: ("R48", 0x0600),
    0x53a0: ("R46", 0x2000), 0x5490: ("R46", 0x2200),
    0x54e0: ("R46", 0x2400), 0x5530: ("R46", 0x2600),
    0x5800: ("R2", 0x0000), 0x58c0: ("R2", 0x0200),
    0x5910: ("R2", 0x0400), 0x5960: ("R2", 0x0600),
    0x5bf0: ("R10", 0x2000), 0x5ce0: ("R10", 0x2200),
    0x5d30: ("R10", 0x2400), 0x5d80: ("R10", 0x2600),
}
STORE_TEXT = "STG.E.NA.128 [{base}.64{slot}], R4"
# Constant-bank slot reference counts per path region; a wiring fingerprint.
CENSUS = {
    (0x0000, 0x14d0): {"0x0": 1, "0x4": 1, "0x28": 1, "0x118": 1, "0x160": 2,
                       "0x164": 2, "0x180": 1, "0x198": 2, "0x1a0": 2, "0x1a4": 3},
    (0x14d0, 0x5da0): {"0x118": 1, "0x168": 16, "0x170": 2, "0x190": 8,
                       "0x198": 16, "0x1a8": 6, "0x1ac": 6},
    (0x5da0, 0x6680): {"0x118": 3, "0x180": 1, "0x190": 9, "0x194": 8, "0x1a4": 1},
}
# Bounded writer audits: stride/lane registers hold these values on these windows.
PROVENANCE = (("R10", 0x12b0, 0x21b0), ("R144", 0x20a0, 0x31b0),
              ("R44", 0x4ca0, 0x5550), ("R45", 0x4c30, 0x5b90))
# Multi-register destinations without width markers; fail closed if present.
UNAUDITED_WIDE = ("HMMA", "IMMA", "LDSM", "TEX", "TLD", "TLD4", "TMA", "LDGSTS")
OUTPUT_SLOTS = (0x0000, 0x0200, 0x0400, 0x0600, 0x2000, 0x2200, 0x2400, 0x2600)
OUTPUT_START_OFFSET_DELTA = 0x27f0  # max-minus-min start offset over the pinned store set.
MAX_INDEX = (1 << 30) - 1  # Helper/test-vector domain only; not a runtime index bound.
U32_MAX = (1 << 32) - 1
OUT0_TEST_ALLOCATION_BYTES = 256 * 1024 * 1024


def output_address(index: int, unit: int, slot: int) -> int:
    require(type(index) is int and 0 <= index <= MAX_INDEX, "invalid output index")
    require(type(unit) is int and 0 <= unit < 32, "invalid output unit")
    require(type(slot) is int and slot in OUTPUT_SLOTS, "invalid output slot")
    return index * 4 + unit * 16 + slot


def read_address(index: int, unit: int) -> int:
    """Skip and partial-sum reads: 16 bytes at index*4 + unit*16."""
    require(type(index) is int and 0 <= index <= MAX_INDEX, "invalid read index")
    require(type(unit) is int and 0 <= unit < 32, "invalid read unit")
    return index * 4 + unit * 16


def weight_word_address(index: int) -> int:
    require(type(index) is int and 0 <= index <= MAX_INDEX, "invalid weight index")
    return 0x80000 + index * 4


def written_registers(instruction: str) -> set[str] | None:
    """Registers written by one decoded instruction, or None if unaudited.

    Bounded writer model, not a general ISA emulator: width markers (.128/.64/
    .WIDE/CS2R) expand the destination span; predicate-only and store/control
    instructions write nothing; UNAUDITED_WIDE families fail closed.
    """
    parts = instruction.split()
    if parts[0].startswith("@"):
        parts = parts[1:]
    if len(parts) < 2:
        return set()
    opcode, operands = parts[0], " ".join(parts[1:]).split(",")
    if opcode.split(".")[0] in ("ISETP", "FSETP", "PLOP3", "PSETP"):
        return set()
    if opcode.startswith(("STG", "STL", "STS", "BRA", "BAR", "WARPSYNC", "MEMBAR",
                          "ERRBAR", "NANOSLEEP", "DEPBAR", "LDGDEPBAR", "EXIT", "RET",
                          "BSYNC", "BSSY", "ST", "RED", "GRED", "SUST", "SURED", "R2P")):
        return set()
    # Atomics return values into registers; fail closed rather than assume none.
    if opcode.split(".")[0] in (*UNAUDITED_WIDE, "ATOM", "ATOMS", "SUATOM", "SUATOMS"):
        return None
    destination = operands[0].strip()
    if re.fullmatch(r"!?U?P(?:\d+|T)", destination):
        destination = next((o.strip() for o in operands[1:]
                            if not re.fullmatch(r"!?U?P(?:\d+|T)", o.strip())), "")
    destination = re.sub(r"\.(reuse|reuse2?)$", "", destination)
    if not re.fullmatch(r"(?:U?R)\d+", destination):
        return set()
    width = (4 if ".128" in opcode else 2 if ".64" in opcode or
             ".WIDE" in opcode or opcode == "CS2R" else 1)
    prefix = "UR" if destination.startswith("UR") else "R"
    first = int(destination[len(prefix):])
    return {f"{prefix}{i}" for i in range(first, first + width)}


def interpret(code: dict[int, str], offsets: tuple[int, ...], registers: dict[str, int],
              bases: dict[str, int]) -> dict[str, int]:
    """Small positive-domain interpreter, not a general SASS/CFG emulator.

    Supports the decoded IMAD family (IMAD, IMAD.SHL.U32, IMAD.IADD,
    IMAD.MOV.U32), IADD3 and IMAD.WIDE with constant-bank base operands.
    IMAD.SHL.U32 is d = a*b + c with power-of-two b (disassembler convenience
    form). Predicates on selected subsequences are assumed true; pointer carry
    assembly and CFG liveness are NOT simulated. Non-wide results are masked to
    32 bits; IMAD.WIDE results are exact Python integers.
    """
    registers = dict(registers)
    mask = (1 << 32) - 1

    def value(token: str) -> int:
        token = re.sub(r"\.(reuse|reuse2?)$", "", token)
        if token == "RZ":
            return 0
        if token.startswith("-"):
            return -value(token[1:])
        if re.fullmatch(r"U?R\d+", token):
            require(token in registers, "uninitialised arithmetic register")
            return registers[token]
        bank = re.fullmatch(r"c\[0x0\]\[(0x[0-9a-f]+)\]", token)
        if bank is not None:
            require(bank[1] in bases, "unmapped constant-bank operand")
            return bases[bank[1]]
        require(re.fullmatch(r"0x[0-9a-f]+", token) is not None, "unsupported arithmetic operand")
        return int(token, 16)

    for offset in offsets:
        require(offset in code, "missing arithmetic instruction")
        instruction = code[offset]
        if instruction.startswith("@"):
            predicate, instruction = instruction.split(" ", 1)
            require(re.fullmatch(r"@P[0-6]", predicate) is not None, "unsupported arithmetic predicate")
        opcode, operand_text = instruction.split(" ", 1)
        operands = operand_text.split(", ")
        require(re.fullmatch(r"R\d+", operands[0]) is not None, "invalid destination")
        wide = False
        if opcode in ("IMAD", "IMAD.SHL.U32", "IMAD.IADD", "IMAD.MOV.U32", "IMAD.WIDE"):
            require(len(operands) == 4, "bad multiply-add arity")
            wide = opcode == "IMAD.WIDE"
            result = value(operands[1]) * value(operands[2]) + value(operands[3])
        elif opcode == "IADD3":
            require(len(operands) == 4, "bad three-input add arity")
            result = sum(value(operand) for operand in operands[1:])
        else:
            raise ValueError("unsupported arithmetic opcode")
        registers[operands[0]] = result if wide else result & mask
    return registers


def check_static(code: dict[int, str]) -> None:
    require(len(code) * 16 == KERNEL_BYTES, "unexpected kernel extent")
    address.check_anchors(code)
    address.check_register_provenance(code)
    for offset, expected in ARITHMETIC.items():
        require(code.get(offset) == expected, f"io arithmetic mismatch at {offset:#x}")
    require(set(STORES) == set(STORE_ORDER) and len(STORE_ORDER) == 16, "store site list mismatch")
    for offset, (base_register, slot) in STORES.items():
        suffix = f"+{slot:#x}" if slot else ""
        require(code.get(offset) == STORE_TEXT.format(base=base_register, slot=suffix),
                f"store mismatch at {offset:#x}")
    for stage1, register, stage2, load, kind in CHAINS:
        pattern = LOAD_FORMS[kind].format(base=re.escape(register))
        require(re.fullmatch(pattern, code.get(load, "")) is not None,
                f"load form mismatch at {load:#x}")
        for offset in range(stage1 + 16, load, 16):
            if offset == stage2:
                continue  # The second stage rewrites the register by design.
            written = written_registers(code[offset])
            require(written is not None, f"unaudited wide write at {offset:#x}")
            require(register not in written, f"unexpected {register} writer at {offset:#x}")
    for (lo, hi), expected in CENSUS.items():
        counts: dict[str, int] = {}
        for offset in range(lo, hi, 16):
            require(offset in code, "missing census instruction")
            for slot in re.findall(r"c\[0x0\]\[(0x[0-9a-f]+)\]", code[offset]):
                counts[slot] = counts.get(slot, 0) + 1
        require(counts == expected, f"constant-bank census mismatch in [{lo:#x},{hi:#x})")
    for register, start, end in PROVENANCE:
        for offset in range(start + 16, end, 16):
            written = written_registers(code[offset])
            require(written is not None, f"unaudited wide write at {offset:#x}")
            require(register not in written, f"unexpected {register} writer at {offset:#x}")


def self_test(code: dict[int, str] | None = None) -> dict[str, Any]:
    actual = ARITHMETIC if code is None else code
    arithmetic = 0
    coverage = 0
    base_value = 0x7F00000000
    bases = {"0x170": base_value, "0x168": base_value + 0x100000,
             "0x190": base_value + 0x200000, "0x198": base_value + 0x300000}
    # Path A: decoded index derivation then pointer formation, one synthetic case
    # per grid corner and unit extreme.
    for r8, r23a, r9, r2, unit in ((0, 0, 0, 0, 0), (3, 5, 7, 11, 31),
                                   (1, 2, 4, 8, 15), (255, 511, 1023, 2047, 1)):
        r = interpret(actual, (0x46d0, 0x46e0, 0x4860), dict(R8=r8, R23=r23a, R9=r9, R2=r2), bases)
        index_a = (r8 * r23a + r9) * 2048 + 4 * r2
        require(r["R20"] == r8 * r23a + r9, "decoded output row mismatch")
        require(r["R127"] == index_a and r["R23"] == 4 * r2, "decoded output index A mismatch")
        r = interpret(actual, (0x4d70, 0x4f10), dict(R127=index_a, R44=4, R45=unit), bases)
        require(r["R46"] == bases["0x170"] + index_a * 4, "decoded output base A mismatch")
        require(r["R48"] == bases["0x170"] + output_address(index_a, unit, 0),
                "decoded output pointer A mismatch")
        r = interpret(actual, (0x4d70, 0x5340), dict(R127=index_a, R44=4, R45=unit), bases)
        require(r["R46"] == bases["0x170"] + output_address(index_a, unit, 0),
                "decoded output pointer A2 mismatch")
        arithmetic += 6
        # Path B: its own index derivation shares R8/R9/R23/R2 inputs. The
        # pinned 0x4620 result is consumed by interleaved SHFL instructions and
        # clobbered at 0x4660 in this straight-line reading; other CFG paths are
        # unmodelled and its contribution is therefore not asserted.
        r = interpret(actual, (0x4620, 0x4660, 0x46e0, 0x4750, 0x4970),
                      dict(R8=r8, R23=r23a, R9=r9, R2=r2), bases)
        index_b = (r8 * r23a + r23a + r9) * 2048 + 4 * r2
        require(r["R0"] == index_b, "decoded output index B mismatch")
        r = interpret(actual, (0x5540, 0x57a0), dict(R0=index_b, R44=4, R45=unit), bases)
        require(r["R10"] == bases["0x170"] + index_b * 4, "decoded output base B mismatch")
        require(r["R2"] == bases["0x170"] + output_address(index_b, unit, 0),
                "decoded output pointer B mismatch")
        r = interpret(actual, (0x5540, 0x5b80), dict(R0=index_b, R44=4, R45=unit), bases)
        require(r["R10"] == bases["0x170"] + output_address(index_b, unit, 0),
                "decoded output pointer B2 mismatch")
        arithmetic += 4
        # Two-stage read chains (partial and skip) with supplied indices.
        for stage1, dst, idx_reg, stride_reg, stage2, unit_reg in (
                (0x16b0, "R6", "R7", "R10", 0x1700, "R119"),
                (0x1d30, "R80", "R81", "R10", 0x1e00, "R119"),
                (0x22c0, "R98", "R99", "R144", 0x2390, "R129"),
                (0x2e30, "R146", "R147", "R144", 0x2f00, "R129"),
                (0x31b0, "R144", "R145", "R144", 0x3370, "R129")):
            index = r8 * 256 + r2
            slot = "0x190" if stage1 == 0x16b0 else "0x168"
            r = interpret(actual, (stage1, stage2),
                          {idx_reg: index, stride_reg: 4, unit_reg: unit}, bases)
            require(r[dst] == bases[slot] + read_address(index, unit),
                    "decoded two-stage read pointer mismatch")
            arithmetic += 1
        # Weight word bases have no second stage in the pinned chains.
        for stage1, dst, idx_reg in ((0x2150, "R98", "R99"), (0x27c0, "R134", "R3")):
            index = r9 * 64 + unit
            r = interpret(actual, (stage1,), {idx_reg: index, "R144": 4}, bases)
            require(r[dst] == bases["0x198"] + weight_word_address(index) - 0x80000,
                    "decoded weight word base mismatch")
            arithmetic += 1
    # Conditional arithmetic only: if every runtime index were an unsigned
    # 32-bit bit-pattern and the displayed formula applied on every store path,
    # this would be its exclusive byte endpoint. It is not a kernel bound.
    conditional_u32_end = U32_MAX * 4 + 31 * 16 + 0x2600 + 16
    require(conditional_u32_end == 0x4000027fc, "conditional u32 envelope mismatch")
    require(conditional_u32_end - OUT0_TEST_ALLOCATION_BYTES == 16_911_443_964,
            "conditional envelope/allocation comparison mismatch")
    # Independent per-index coverage oracles built without the layout formula.
    for index in (0, 1, 1000, MAX_INDEX):
        addresses = sorted(output_address(index, unit, slot) for unit in range(32)
                           for slot in OUTPUT_SLOTS)
        require(len(addresses) == len(set(addresses)) == 256, "output unit/slot alias")
        by_slot: dict[int, list[int]] = {}
        for unit in range(32):
            for slot in OUTPUT_SLOTS:
                by_slot.setdefault(slot, []).append(output_address(index, unit, slot))
        for slot, values in by_slot.items():
            require(sorted(values) == list(range(index * 4 + slot, index * 4 + slot + 512, 16)),
                    "output slot block gap/alias")
        require(output_address(index, 31, 0x2600) - output_address(index, 0, 0) == OUTPUT_START_OFFSET_DELTA,
                "output per-index start-offset delta mismatch")
        coverage += 3
    known = ((output_address(0, 0, 0), 0), (output_address(0, 31, 0x2600), 0x27f0),
             (output_address(1000, 3, 0x200), 0x11d0), (read_address(3, 2), 44),
             (weight_word_address(0), 0x80000), (read_address(2, 1), 24))
    for observed, expected in known:
        require(observed == expected, "literal address answer mismatch")
    # Wrong-formula controls must fail numerically, not merely by hash.
    wrong = ((0x4860, "IMAD R127, R20, 0x400, R23"),
             (0x4f10, "IMAD.WIDE R48, R45, 0x20, R46"),
             (0x2390, "@P4 IMAD.WIDE R98, R129, 0x8, R98"),
             (0x1e00, "@P2 IMAD.WIDE R80, R119, 0x20, R80"))
    for offset, instruction in wrong:
        altered = dict(actual)
        altered[offset] = instruction
        if offset == 0x4860:
            r = interpret(altered, (0x46d0, 0x46e0, 0x4860), dict(R8=3, R23=5, R9=7, R2=11), bases)
            require(r["R127"] != (3 * 5 + 7) * 2048 + 44, "wrong index arithmetic accepted")
        elif offset == 0x4f10:
            r = interpret(altered, (0x4d70, 0x4f10), dict(R127=9, R44=4, R45=5), bases)
            require(r["R48"] != bases["0x170"] + output_address(9, 5, 0), "wrong pointer arithmetic accepted")
        elif offset == 0x2390:
            r = interpret(altered, (0x22c0, 0x2390), dict(R99=9, R144=4, R129=5), bases)
            require(r["R98"] != bases["0x168"] + read_address(9, 5), "wrong skip unit arithmetic accepted")
        else:
            r = interpret(altered, (0x1d30, 0x1e00), dict(R81=9, R10=4, R119=5), bases)
            require(r["R80"] != bases["0x168"] + read_address(9, 5), "wrong read unit arithmetic accepted")
    # Consumer-form controls: widened/immediate-shifted loads must not match.
    require(re.fullmatch(LOAD_FORMS["read128"].format(base="R80"),
                         "@P2 LDG.E.128.STRONG.SM R72, [R80.64+0x10]") is None,
            "wrong load form accepted")
    require(re.fullmatch(LOAD_FORMS["word32"].format(base="R134"),
                         "LDG.E R10, [R134.64+0x80004]") is None,
            "wrong weight load form accepted")
    # This is only the helper's narrower domain rejection; the U32 arithmetic
    # envelope above is asserted independently and is not exercised by it.
    invalid = (lambda: output_address(-1, 0, 0), lambda: output_address(0, 32, 0),
               lambda: output_address(0, 0, 0x100), lambda: output_address(True, 0, 0),
               lambda: read_address(-1, 0), lambda: read_address(0, -1),
               lambda: weight_word_address(MAX_INDEX + 1), lambda: output_address(0, 0, True))
    rejected = 0
    for call in invalid:
        try:
            call()
        except ValueError:
            rejected += 1
    require(rejected == len(invalid), "invalid io addressing domain accepted")
    require(written_registers("HMMA.M16N8K16.F16 R0, R1, R2, R3") is None,
            "unaudited wide opcode accepted")
    require(written_registers("ATOM.E.ADD R4, [R2], R6") is None,
            "atomic return write accepted")
    require(written_registers("STG.E.NA.128 [R48.64], R4") == set(),
            "store treated as register write")
    require(written_registers("@P3 LDG.E.128.STRONG.SM R72, [R80.64]") == {"R72", "R73", "R74", "R75"},
            "wide load span wrong")
    return dict(arithmetic_comparisons=arithmetic, coverage_checks=coverage,
                literal_answers=len(known), store_sites=len(STORES), chains=len(CHAINS),
                load_form_controls_rejected=2,
                wrong_formula_controls_rejected=len(wrong), invalid_domains_rejected=rejected,
                writer_model_controls=4, conditional_u32_envelope=conditional_u32_end,
                conditional_excess_over_test_allocation=conditional_u32_end - OUT0_TEST_ALLOCATION_BYTES,
                gpu_kernels_launched=0)


def inspect(data: bytes, negative: bool) -> dict[str, Any]:
    inventory = base.inventory(data)  # Includes DLL hash validation.
    entries = [e for e in inventory["fatbins"][6]["entries"] if e["kind"] == 2 and e["arch"] == 89]
    require(len(entries) == 1, "unique SM89 decoder module")
    code = address.disassemble(base.decompress(data, entries[0]))
    check_static(code)
    for lo, hi, digest in SLICES:
        require(digest.startswith("PLACEHOLDER") is False and len(digest) == 64,
                "unpinned slice")
        require(all(offset in code for offset in range(lo, hi, 16)), "missing slice instruction")
        text = "\n".join(f"{offset:04x}:{code[offset]}" for offset in range(lo, hi, 16)).encode()
        require(hashlib.sha256(text).hexdigest() == digest, "io instruction fingerprint mismatch")
    if negative:
        rejected = 0
        for offset in (*ARITHMETIC, *STORES):
            bad = dict(code)
            bad[offset] = "NOP"
            try:
                check_static(bad)
            except ValueError:
                rejected += 1
        require(rejected == len(ARITHMETIC) + len(STORES), "corrupted SASS accepted")
        return dict(sass_corruptions_rejected=rejected, gpu_kernels_launched=0)
    cpu = self_test()
    return dict(kernel=address.NAME, cpu_self_test=cpu,
                output_slot_count=len(OUTPUT_SLOTS), output_stores=len(STORES),
                output_pointer="argument slot 16 (c[0x0][0x170]) + index*4 + unit*16 + slot",
                output_unit="SR_LANEID (pinned S2R)", output_bytes_per_store=16,
                output_stores_per_index_per_unit=len(OUTPUT_SLOTS),
                output_store_slots=list(OUTPUT_SLOTS), output_per_index_start_offset_delta=OUTPUT_START_OFFSET_DELTA,
                read_pointer="stage1 index*4 then + unit*16; 16-byte reads pinned at four sites",
                read_slots="argument slot 8 (c[0x0][0x168]) skip, slot 48 (c[0x0][0x190]) partial",
                read_unit="R119/R129 16-byte step registers; origins unmodelled",
                weight_words="argument slot 56 (c[0x0][0x198]) + index*4 + 0x80000; 4-byte reads pinned at two sites",
                constant_census_regions=len(CENSUS), kernel_extent=KERNEL_BYTES,
                provenance_windows=len(PROVENANCE),
                index_semantics="decoded arithmetic only; loop/coordinate bounds and CFG paths conditional",
                output_buffer_extent_verified=False, channel_semantics_verified=False,
                complete_decoder_contract=False, numerical_decoder_validation=False,
                gpu_kernels_launched=0)


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
    print(json.dumps(dict(mode=args.mode, dll_sha256=base.PIN, disassembler_sha256=address.TOOL_PIN,
                          memory_policy=policy, addressing=report), indent=2))


if __name__ == "__main__":
    main()
