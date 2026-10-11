#!/usr/bin/env python3
"""Check a bounded static host descriptor contract; never execute PE/CUDA code.

Retains metadata only, not disassembly or weights. This is not a dynamic trace,
full x86 emulator, or evidence that the tested GPU chain is the runtime graph.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import fcntl
import os
from pathlib import Path
import re
import subprocess
from typing import Any

spec = importlib.util.spec_from_file_location("runtime", Path(__file__).with_name("native-runtime-probe.py"))
if spec is None or spec.loader is None:
    raise RuntimeError("missing runtime inventory helper")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
require = runtime.require

# Only these bounded functions are disassembled, in memory. No extracted GPU code.
RANGES = ((0x1800439D0, 0x180044665), (0x180043260, 0x1800437A0),
          (0x180071B90, 0x180071C3D), (0x1800719B0, 0x180071A39),
          (0x180072110, 0x1800721AE), (0x1800714A0, 0x180071534),
          (0x180071ED0, 0x180071F7A), (0x18002AF33, 0x18002B19D),
          (0x18001EA60, 0x18001EB44),
          (0x180039780, 0x18003ABC2), (0x180073DB0, 0x180074469),
          (0x180032650, 0x1800326BE),
           (0x1800380C9, 0x18003852D), (0x18003DFD0, 0x18003E1EF),
           (0x18003DB60, 0x18003DBE0), (0x180023970, 0x180023993),
           (0x1800319A0, 0x180031B66), (0x18001E8A0, 0x18001EA5D),
           (0x180040E20, 0x180041B28), (0x18005FD60, 0x18005FE78),
           (0x180031DF0, 0x180031FE9), (0x18001EB44, 0x18001EB8A),
           (0x18006BA54, 0x18006BA66), (0x18006BCCC, 0x18006BCDE),
           (0x18006BFBF, 0x18006BFD1))


def instructions(text: str) -> dict[int, tuple[str, str]]:
    result: dict[int, tuple[str, str]] = {}
    for line in text.splitlines():
        match = re.fullmatch(r"\s*([0-9a-f]+):\s+(?:[0-9a-f]{2}\s+)+([a-z][a-z0-9]*)\s*(.*?)\s*", line)
        if match is None:
            continue  # labels, continuation bytes and section headers
        address, mnemonic, operands = match.groups()
        require(int(address, 16) not in result, "duplicate instruction address")
        result[int(address, 16)] = (mnemonic, operands.split("#", 1)[0].strip())
    require(bool(result), "no decoded instructions")
    return result


def expect(code: dict[int, tuple[str, str]], address: int, mnemonic: str, operands: str) -> None:
    require(code.get(address) == (mnemonic, operands), f"static evidence mismatch at {address:#x}")


def vector(code: dict[int, tuple[str, str]], assignments: tuple[tuple[int, str], ...],
           length_address: int, length: int, destination_address: int, rbp_field: int,
           reference_address: int, end_address: int, end_field: int) -> list[int]:
    """Decode a short, inspected memcpy sequence; register constants checked separately.

    This is deliberately fail-closed at known addresses, not generic control-flow
    emulation. Constructor/reallocation paths have been inspected separately.
    """
    constants = {"r13": 0, "rsi": 1, "r14": 2}
    values = []
    for index, (address, register) in enumerate(assignments):
        target = f"QWORD PTR [rsp+{0x30 + 8 * index:#x}]"
        expect(code, address, "mov", f"{target},{register}")
        require(register in constants or register == "0x3", "unknown vector constant")
        values.append(3 if register == "0x3" else constants[register])
    require(len(values) == length, "vector length mismatch")
    expect(code, length_address, "mov", f"r8d,{8 * length:#x}")
    expect(code, destination_address, "mov", f"{'rbx' if length > 1 else 'rcx'},QWORD PTR [rbp+{rbp_field:#x}]")
    if length > 1:
        expect(code, reference_address - 3, "mov", "rcx,rbx")
    expect(code, reference_address, "call", "0x18007e550")
    expect(code, length_address + 6, "lea", "rdx,[rsp+0x30]")
    expect(code, reference_address + 5, "lea", f"rax,[rbx+{length * 8:#x}]")
    expect(code, end_address, "mov", f"QWORD PTR [rbp+{end_field:#x}],rax")
    return values


def contract(code: dict[int, tuple[str, str]]) -> dict[str, Any]:
    # Constants are callee-saved registers. These anchors constrain interpretation.
    for address, mnemonic, operands in (
        (0x180043A0F, "xor", "r13d,r13d"), (0x180043EDA, "mov", "esi,0x1"),
        (0x180044316, "mov", "r14d,0x2"),
        (0x180043C67, "jmp", "0x180043c76"),
        (0x180044463, "mov", "rax,QWORD PTR [r15+0x28]"),
        (0x180044499, "movzx", "edx,r12b"),
        (0x18004331C, "cmp", "rdx,0x5"), (0x1800436BA, "add", "r15,0x170"),
        (0x1800436C1, "add", "r14,0x170"),
    ):
        expect(code, address, mnemonic, operands)

    stages = [
        ("ffn-expand", 0x180043C42, 0x180071B90, 0x180071C10, 0x1800B2818),
        ("ffn-contract", 0x180043C87, 0x1800719B0, 0x180071A0D, 0x1800B2830),
        ("qkv", 0x1800441BD, 0x180072110, 0x180072171, 0x1800B2848),
        ("attention", 0x180044310, 0x1800714A0, 0x180071501, 0x1800B2858),
        ("projection", 0x1800444A4, 0x180071ED0, 0x180071F4D, 0x1800B2870),
    ]
    for _, call, constructor, type_address, string_address in stages:
        expect(code, call, "call", hex(constructor))
        # RIP target recomputed by objdump from the instruction, not a string-name guess.
        mnemonic, operands = code[type_address]
        require(mnemonic == "lea" and operands.startswith("rdx,[rip+0x"), "missing type assignment")
        displacement = int(operands.split("+", 1)[1].split("]", 1)[0], 16)
        require(type_address + 7 + displacement == string_address, "wrong stage type target")
    # Tie constructed objects to copied descriptor identities, not Python literals.
    for address, mnemonic, operands in (
        (0x180043C3B, "lea", "rcx,[rbp+0x510]"),
        (0x180043C47, "mov", "r8,rax"),
        (0x180043C5A, "call", "0x180031df0"),
        (0x180043C5F, "add", "QWORD PTR [r15+0x28],0x170"),
        (0x180043C70, "call", "0x18002f150"),
        (0x180043C82, "lea", "rcx,[rsp+0x50]"),
        (0x18004418E, "lea", "rdx,[rsp+0x50]"),
        (0x180044196, "call", "0x18001ea60"),
        (0x18004419B, "add", "QWORD PTR [r15+0x28],0x170"),
        (0x1800441A5, "lea", "r8,[rsp+0x50]"),
        (0x1800441B1, "call", "0x180040070"),
        (0x1800441B6, "lea", "rcx,[rbp+0x3a0]"),
        (0x1800442DD, "lea", "rdx,[rbp+0x3a0]"),
        (0x1800442E7, "call", "0x18001ea60"),
        (0x1800442EC, "add", "QWORD PTR [r15+0x28],0x170"),
        (0x1800442F6, "lea", "r8,[rbp+0x3a0]"),
        (0x180044304, "call", "0x180040070"),
        (0x180044309, "lea", "rcx,[rbp+0x230]"),
        (0x18004446D, "lea", "rdx,[rbp+0x230]"),
        (0x180044477, "call", "0x18001ea60"),
        (0x18004447C, "add", "QWORD PTR [r15+0x28],0x170"),
        (0x180044486, "lea", "r8,[rbp+0x230]"),
        (0x180044494, "call", "0x180040070"),
        (0x18004449D, "lea", "rcx,[rbp+0xc0]"),
        (0x1800445F5, "lea", "rdx,[rbp+0xc0]"),
        (0x1800445FF, "call", "0x18001ea60"),
        (0x180044604, "add", "QWORD PTR [r15+0x28],0x170"),
        (0x18004460E, "lea", "r8,[rbp+0xc0]"),
        (0x18004461C, "call", "0x180040070"),
    ):
        expect(code, address, mnemonic, operands)
    # Generic input resolver: +0x90 selects earlier layer, +0xa8 external input,
    # +0xc0 selects earlier layer's output port. These are distinct namespaces.
    for address, mnemonic, operands in (
        (0x18002AFD4, "mov", "rbx,QWORD PTR [rdi+0xa8]"),
        (0x18002B05A, "mov", "rcx,QWORD PTR [rdi+0x90]"),
        (0x18002AFF0, "mov", "rsi,QWORD PTR [rbx]"),
        (0x18002AFF3, "mov", "rcx,QWORD PTR [r12]"),
        (0x18002B00C, "lea", "r8,[rcx+rsi*8]"),
        (0x18002B080, "mov", "r8,QWORD PTR [rcx+rbx*8]"),
        (0x18002B084, "mov", "rdx,QWORD PTR [rdi+0xc0]"),
        (0x18002B0A6, "lea", "rcx,[r8+r8*2]"),
        (0x18002B0AA, "mov", "rax,QWORD PTR [r15+0x140]"),
        (0x18002B0B1, "mov", "rcx,QWORD PTR [rax+rcx*8]"),
        (0x18002B0B5, "mov", "rax,QWORD PTR [rcx+rdx*8]"),
        # Descriptor copies preserve each of the three vectors at the same offsets.
        (0x18001EB08, "lea", "rcx,[rbx+0x90]"),
        (0x18001EB0F, "lea", "rdx,[rdi+0x90]"),
        (0x18001EB1C, "lea", "rcx,[rbx+0xa8]"),
        (0x18001EB23, "lea", "rdx,[rdi+0xa8]"),
        (0x18001EB30, "lea", "rcx,[rbx+0xc0]"),
        (0x18001EB37, "lea", "rdx,[rdi+0xc0]"),
    ):
        expect(code, address, mnemonic, operands)

    qkv_sources = vector(code, 0x1800441F9, ((0x1800441C3, "rsi"),),
                         0, 1, 0, 0, 0, 0, 0) if False else []
    # Single-element vectors use a store rather than memcpy.
    for address, mnemonic, operands in (
        (0x1800441F9, "mov", "QWORD PTR [rcx],rsi"),
        (0x1800441F2, "mov", "rcx,QWORD PTR [rbp+0x430]"),
        (0x1800441FC, "lea", "rax,[rcx+0x8]"),
        (0x180044200, "mov", "QWORD PTR [rbp+0x438],rax"),
        (0x18004423D, "mov", "QWORD PTR [rcx],r13"),
        (0x180044236, "mov", "rcx,QWORD PTR [rbp+0x460]"),
        (0x180044240, "lea", "rax,[rcx+0x8]"),
        (0x180044244, "mov", "QWORD PTR [rbp+0x468],rax"),
    ):
        expect(code, address, mnemonic, operands)
    qkv_sources, qkv_ports = [1], [0]
    attention_sources = vector(code,
        ((0x18004431C, "r14"), (0x180044321, "r14"), (0x180044326, "r14")),
        0x18004435D, 3, 0x180044356, 0x2C0, 0x18004436B, 0x180044374, 0x2C8)
    attention_ports = vector(code,
        ((0x18004437B, "r13"), (0x180044380, "rsi"), (0x180044385, "r14")),
        0x1800443BD, 3, 0x1800443B6, 0x2F0, 0x1800443CB, 0x1800443D4, 0x2F8)
    projection_sources = vector(code,
        ((0x1800444AA, "0x3"), (0x1800444B3, "rsi")),
        0x1800444E9, 2, 0x1800444E2, 0x150, 0x1800444F7, 0x180044500, 0x158)
    projection_ports = vector(code,
        ((0x180044507, "r13"), (0x18004450C, "r13")),
        0x180044542, 2, 0x18004453B, 0x180, 0x180044550, 0x180044559, 0x188)
    for base, sources_field, ports_field in ((0x3A0, 0x430, 0x460),
                                            (0x230, 0x2C0, 0x2F0), (0xC0, 0x150, 0x180)):
        require(sources_field - base == 0x90 and ports_field - base == 0xC0,
                "descriptor field identity mismatch")
    edges = []
    for consumer, sources, ports in (("qkv", qkv_sources, qkv_ports),
                                    ("attention", attention_sources, attention_ports),
                                    ("projection", projection_sources, projection_ports)):
        require(len(sources) == len(ports), "edge vectors disagree")
        for slot, (source, port) in enumerate(zip(sources, ports)):
            edges.append(dict(producer=stages[source][0], output_port=port,
                              consumer=consumer, input_slot=slot))
    return dict(stage_order=[stage[0] for stage in stages], edges=edges,
                descriptor_stride_bytes=0x170, static_contract_verified=True,
                 boundary=boundary_contract(code),
                 outer_routing=outer_routing_contract(code),
                 source_exports=source_export_contract(code),
                full_graph_recovered=False, dynamic_bindings_verified=False,
                native_image_inference=False, gpu_kernels_launched=0)


def boundary_contract(code: dict[int, tuple[str, str]]) -> dict[str, Any]:
    """Check builder loop and following descriptor, not GPU tensor bindings.

    The two vectors remain raw outer-descriptor indices/selectors. They must not
    be substituted for the intra-layer namespaces checked by contract().
    """
    for address, mnemonic, operands in (
        # Eight iterations; nonzero configuration selects a different factory.
        (0x18003AA4F, "xor", "r15d,r15d"), (0x18003AA52, "mov", "edi,r15d"),
        (0x18003AA55, "mov", "r14d,DWORD PTR [rsp+0x60]"),
        (0x18003AA60, "lea", "rcx,[rbp-0x20]"),
        (0x18003AA64, "test", "r14d,r14d"), (0x18003AA67, "jne", "0x18003aab9"),
        (0x18003AA69, "cmp", "edi,0x7"), (0x18003AA6C, "sete", "r8b"),
        (0x18003AA70, "test", "edi,edi"), (0x18003AA72, "sete", "dl"),
        (0x18003AA75, "call", "0x1800439d0"),
        (0x18003AA7B, "mov", "rdx,rax"), (0x18003AA7E, "mov", "rcx,rbx"),
        (0x18003AA81, "call", "0x18003db60"),
        (0x18003AAB7, "jmp", "0x18003aafb"),
        (0x18003AAB9, "call", "0x1800425a0"),
        (0x18003AABF, "mov", "rdx,rax"), (0x18003AAC2, "mov", "rcx,rbx"),
        (0x18003AAC5, "call", "0x18003db60"),
        (0x18003AAFB, "lea", "rcx,[rbp-0x50]"),
        (0x18003AAFF, "call", "0x1800081a0"),
        (0x18003AB04, "inc", "edi"), (0x18003AB06, "cmp", "edi,0x8"),
        (0x18003AB09, "jl", "0x18003aa60"),
        # The following single-layer descriptor and its authored channel sizes.
        (0x18003AB0F, "lea", "rcx,[rbp-0x20]"),
        (0x18003AB13, "call", "0x180031b70"),
        (0x18003AB19, "lea", "rdx,[rip+0x77838]"),
        (0x18003AB20, "lea", "rcx,[rbp-0x20]"),
        (0x18003AB24, "call", "0x180008cc0"),
        (0x18003AB29, "mov", "edx,0x400"),
        (0x18003AB2E, "mov", "r8d,0x200"),
        (0x18003AB34, "lea", "rcx,[rbp+0x140]"),
        (0x18003AB3B, "call", "0x180073db0"),
        (0x18003AB41, "mov", "rdx,rax"),
        (0x18003AB44, "lea", "rcx,[rbp+0x0]"),
        (0x18003AB48, "call", "0x18003dbe0"),
        (0x18003AB4E, "lea", "rcx,[rbp+0x140]"),
        (0x18003AB55, "call", "0x180016ef0"),
        (0x18003ABB6, "lea", "rdx,[rbp-0x20]"),
        (0x18003ABBA, "mov", "rcx,rbx"),
        (0x18003ABBD, "call", "0x18003dba0"),
        # Constructor maps dimensions to +0x50 / +0x54 and sets its type.
        (0x180073DE0, "mov", "DWORD PTR [rsp+0x38],r8d"),
        (0x180073DE7, "mov", "DWORD PTR [rsp+0x48],edx"),
        (0x180073DEB, "mov", "rbx,rcx"),
        (0x180073E0A, "lea", "rsi,[rbx+0x20]"),
        (0x18007429E, "mov", "r8d,0x12"),
        (0x1800742A4, "lea", "rdx,[rip+0x3bd8d]"),
        (0x1800742AB, "mov", "rcx,rsi"),
        (0x1800742AE, "call", "0x180008ce0"),
        (0x1800742C1, "mov", "eax,DWORD PTR [rsp+0x48]"),
        (0x1800742C5, "mov", "DWORD PTR [rbx+0x50],eax"),
        (0x1800742C8, "mov", "eax,DWORD PTR [rsp+0x38]"),
        (0x1800742CC, "mov", "DWORD PTR [rbx+0x54],eax"),
        # Shared vector assignment: end-begin bytes copied; end pointer retained.
        (0x180032664, "mov", "rsi,QWORD PTR [rdx+0x8]"),
        (0x180032668, "mov", "rdi,rcx"),
        (0x18003266B, "mov", "rbp,QWORD PTR [rdx]"),
        (0x180032672, "sub", "rsi,rbp"),
        (0x18003268D, "mov", "rbx,QWORD PTR [rdi]"),
        (0x180032690, "mov", "r8,rsi"),
        (0x180032693, "mov", "rcx,rbx"),
        (0x180032696, "mov", "rdx,rbp"),
        (0x180032699, "call", "0x18007e550"),
        (0x18003269E, "lea", "rax,[rsi+rbx*1]"),
        (0x1800326A2, "mov", "QWORD PTR [rdi+0x8],rax"),
    ):
        expect(code, address, mnemonic, operands)

    # Two stack ranges, each exactly 16 bytes, copied into distinct outer fields.
    for addresses, values, field in (
        ((0x18003AB5A, 0x18003AB62, 0x18003AB6A, 0x18003AB6E,
          0x18003AB73, 0x18003AB77, 0x18003AB7C, 0x18003AB81, 0x18003AB85),
         ("0x26", "0x1e"), "0x18"),
        ((0x18003AB8A, 0x18003AB8E, 0x18003AB96, 0x18003AB9A,
          0x18003AB9F, 0x18003ABA3, 0x18003ABA8, 0x18003ABAD, 0x18003ABB1),
         ("r15", "0x1"), "0x30"),
    ):
        for address, mnemonic, operands in zip(addresses,
            ("mov", "mov", "lea", "mov", "lea", "mov", "lea", "lea", "call"),
            (f"QWORD PTR [rbp-0x70],{values[0]}", f"QWORD PTR [rbp-0x68],{values[1]}",
             "rax,[rbp-0x70]", "QWORD PTR [rsp+0x60],rax", "rax,[rbp-0x60]",
             "QWORD PTR [rsp+0x68],rax", "rdx,[rsp+0x60]", f"rcx,[rbp+{field}]",
             "0x180032650")):
            expect(code, address, mnemonic, operands)
    return dict(transformer_iterations=8, primary_factory="CCVit1D-five-stage",
                alternate_factory_present=True, first_block_input_flag=True,
                last_block_output_flag=True, following_descriptor="CCDecInputUpsample",
                descriptor_input_channels=1024, descriptor_output_channels=512,
                outer_descriptor_source_indices=[38, 30], outer_descriptor_port_selectors=[0, 1],
                original_weight_namespace_verified=False, resolved_gpu_consumer_verified=False)



def outer_routing_contract(code: dict[int, tuple[str, str]]) -> dict[str, Any]:
    """Check static exported-output routing, not actual CUDA addresses.

    +0x38/+0x50 select a source block and its exported output. Exports are
    (internal layer, internal port) pairs, not direct internal port selectors.
    The checked consumer here is lifetime/tensor-index analysis. It must not
    be described as a recovered kernel launch or a dynamic handle capture.
    """
    for address, mnemonic, operands in (
        # The primary factory initialises its outer export vectors empty.
        (0x180043A07, "mov", "r15,rcx"),
        (0x180043A83, "mov", "QWORD PTR [rcx+0xb0],r13"),
        (0x180043A8A, "mov", "QWORD PTR [rcx+0xb8],r13"),
        (0x180043A98, "mov", "QWORD PTR [rcx+0xc8],r13"),
        (0x180043A9F, "mov", "QWORD PTR [rcx+0xd0],r13"),
        (0x180044653, "mov", "rax,r15"),
        # Outer descriptors have their own 0xe0 stride, not 0x170 layer stride.
        (0x18003DB66, "mov", "rax,QWORD PTR [rcx+0x8]"),
        (0x18003DB71, "je", "0x18003db89"),
        (0x18003DB76, "call", "0x1800319a0"),
        (0x18003DB7B, "add", "QWORD PTR [rbx+0x8],0xe0"),
        (0x18003DB94, "jmp", "0x18002ef80"),
        (0x18003DBA6, "mov", "rax,QWORD PTR [rcx+0x8]"),
        (0x18003DBB1, "je", "0x18003dbc9"),
        (0x18003DBB6, "call", "0x18001e8a0"),
        (0x18003DBBB, "add", "QWORD PTR [rbx+0x8],0xe0"),
        (0x18003DBD4, "jmp", "0x18002eca0"),
        # Fast move and copy paths preserve source/selector vectors.
        (0x180031A05, "mov", "rax,QWORD PTR [r8+0x38]"),
        (0x180031A0D, "mov", "QWORD PTR [r9+0x38],rax"),
        (0x180031A29, "mov", "rax,QWORD PTR [r8+0x50]"),
        (0x180031A31, "mov", "QWORD PTR [r9+0x50],rax"),
        (0x18001E9B7, "lea", "rcx,[r15+0x38]"),
        (0x18001E9BB, "lea", "rdx,[rbp+0x38]"),
        (0x18001E9BF, "call", "0x18001e7f0"),
        (0x18001E9C5, "lea", "rcx,[r15+0x50]"),
        (0x18001E9C9, "lea", "rdx,[rbp+0x50]"),
        (0x18001E9CD, "call", "0x18001e7f0"),
        # Zero reaches the first selector through the preceding non-calling pass.
        (0x1800380C9, "xor", "r10d,r10d"),
        (0x1800380F6, "jle", "0x18003831c"),
        (0x180038316, "jl", "0x180038110"),
        (0x18003814D, "jle", "0x1800382a6"),
        (0x18003817A, "je", "0x180038257"),
        (0x1800381C2, "mov", "r10,QWORD PTR [rsi+rax*1]"),
        (0x180038254, "xor", "r10d,r10d"),
        # Source vector iteration and selector fallback to export zero.
        (0x1800383A8, "mov", "rcx,QWORD PTR [r14+r13*1+0x38]"),
        (0x1800383AD, "mov", "rax,QWORD PTR [r14+r13*1+0x40]"),
        (0x1800383B5, "sar", "rax,0x3"),
        (0x1800383D0, "mov", "r15,QWORD PTR [rcx+rbx*8]"),
        (0x1800383D4, "mov", "rdi,QWORD PTR [r14+r13*1+0x50]"),
        (0x1800383D9, "mov", "rax,QWORD PTR [r14+r13*1+0x58]"),
        (0x1800383E1, "sar", "rax,0x3"),
        (0x1800383E5, "cmp", "rbx,rax"),
        (0x1800383E8, "jae", "0x1800383f0"),
        (0x1800383EA, "mov", "rdi,QWORD PTR [rdi+rbx*8]"),
        (0x1800383EE, "jmp", "0x1800383f3"),
        (0x1800383F0, "mov", "rdi,r10"),
        (0x1800383F3, "imul", "rdx,r15,0xe0"),
        (0x1800383FA, "add", "rdx,QWORD PTR [r12+0x70]"),
        (0x1800383FF, "lea", "rcx,[rbp+0x128]"),
        (0x180038406, "call", "0x18003dfd0"),
        (0x18003840B, "mov", "rcx,QWORD PTR [rbp+0x130]"),
        (0x180038412, "mov", "r9,QWORD PTR [rbp+0x128]"),
        (0x180038419, "sub", "rcx,r9"),
        (0x18003841C, "sar", "rcx,0x3"),
        (0x180038420, "xor", "r10d,r10d"),
        (0x180038423, "mov", "eax,r10d"),
        (0x180038426, "cmp", "rdi,rcx"),
        (0x180038429, "cmovb", "rax,rdi"),
        (0x18003842D, "mov", "rdi,QWORD PTR [r9+rax*8]"),
        # Packed export: low32 layer, high32 port -> signed tensor-index table.
        (0x180038431, "lea", "rcx,[r15+r15*2]"),
        (0x180038435, "mov", "rax,QWORD PTR [rsp+0x70]"),
        (0x18003843A, "mov", "rax,QWORD PTR [rax+0x1a8]"),
        (0x180038441, "lea", "rdx,[rax+rcx*8]"),
        (0x180038445, "movsxd", "rax,edi"),
        (0x180038448, "lea", "rcx,[rax+rax*2]"),
        (0x18003844C, "mov", "rax,QWORD PTR [rdx]"),
        (0x18003844F, "lea", "rax,[rax+rcx*8]"),
        (0x180038453, "shr", "rdi,0x20"),
        (0x180038457, "movsxd", "rcx,edi"),
        (0x18003845A, "mov", "rax,QWORD PTR [rax]"),
        (0x18003845D, "movsxd", "rdx,DWORD PTR [rax+rcx*4]"),
        (0x180038461, "test", "edx,edx"),
        (0x180038463, "js", "0x180038482"),
        (0x180038465, "mov", "rcx,rdx"),
        (0x180038468, "shl", "rcx,0x4"),
        (0x18003846C, "add", "rcx,0xc"),
        (0x180038470, "add", "rcx,QWORD PTR [rbp-0x10]"),
        (0x180038478, "cmp", "DWORD PTR [rcx],esi"),
        (0x18003847A, "cmovge", "rax,rcx"),
        (0x180038480, "mov", "DWORD PTR [rcx],eax"),
        (0x1800384A7, "inc", "rbx"),
        (0x1800384BE, "jb", "0x1800383d0"),
        (0x1800384E6, "add", "r13,0xe0"),
        # Export helper explicit map: +0xb0 layer indices, +0xc8 ports.
        (0x18003DFE5, "mov", "rdi,rdx"),
        (0x18003DFE8, "mov", "rsi,rcx"),
        (0x18003DFEB, "xor", "ebx,ebx"),
        (0x18003DFFC, "mov", "rax,QWORD PTR [rdx+0xb0]"),
        (0x18003E003, "mov", "rdx,QWORD PTR [rdx+0xb8]"),
        (0x18003E00A, "cmp", "rax,rdx"),
        (0x18003E00D, "je", "0x18003e0fc"),
        (0x18003E042, "mov", "r9,QWORD PTR [rdi+0xb0]"),
        (0x18003E070, "mov", "r10,QWORD PTR [rdi+0xc8]"),
        (0x18003E077, "mov", "rax,QWORD PTR [rdi+0xd0]"),
        (0x18003E07E, "sub", "rax,r10"),
        (0x18003E081, "sar", "rax,0x3"),
        (0x18003E085, "lea", "rdx,[r14*8+0x0]"),
        (0x18003E08D, "cmp", "r14,rax"),
        (0x18003E090, "jae", "0x18003e098"),
        (0x18003E092, "mov", "eax,DWORD PTR [r10+rdx*1]"),
        (0x18003E096, "jmp", "0x18003e09a"),
        (0x18003E09A, "mov", "DWORD PTR [rbp+0x28],eax"),
        (0x18003E098, "mov", "eax,ebx"),
        (0x18003E09D, "mov", "r9d,DWORD PTR [r9+rdx*1]"),
        (0x18003E0AA, "mov", "DWORD PTR [rcx],r9d"),
        (0x18003E0AD, "mov", "DWORD PTR [rcx+0x4],eax"),
        (0x18003E0B0, "add", "rcx,0x8"),
        # With no explicit exports: expose final layer's output ports.
        (0x18003E0FC, "mov", "rax,QWORD PTR [rdi+0x20]"),
        (0x18003E100, "mov", "r10,QWORD PTR [rdi+0x28]"),
        (0x18003E104, "cmp", "rax,r10"),
        (0x18003E107, "jne", "0x18003e119"),
        (0x18003E119, "mov", "rcx,r10"),
        (0x18003E11C, "sub", "rcx,rax"),
        (0x18003E11F, "movabs", "rax,0xb21642c8590b2165"),
        (0x18003E129, "imul", "rcx"),
        (0x18003E12C, "add", "rdx,rcx"),
        (0x18003E12F, "sar", "rdx,0x8"),
        (0x18003E133, "mov", "r9,rdx"),
        (0x18003E136, "shr", "r9,0x3f"),
        (0x18003E13A, "add", "r9,rdx"),
        (0x18003E13D, "dec", "r9d"),
        (0x18003E144, "mov", "rax,QWORD PTR [r10-0x98]"),
        (0x18003E14B, "mov", "rcx,QWORD PTR [r10-0x90]"),
        (0x18003E152, "cmp", "rax,rcx"),
        (0x18003E155, "jne", "0x18003e15e"),
        (0x18003E157, "mov", "edi,0x1"),
        (0x18003E15C, "jmp", "0x18003e180"),
        (0x18003E15E, "sub", "rcx,rax"),
        (0x18003E161, "movabs", "rax,0x2aaaaaaaaaaaaaab"),
        (0x18003E16B, "imul", "rcx"),
        (0x18003E16E, "mov", "rdi,rdx"),
        (0x18003E171, "sar", "rdi,1"),
        (0x18003E174, "mov", "rax,rdi"),
        (0x18003E177, "shr", "rax,0x3f"),
        (0x18003E17B, "add", "rdi,rax"),
        (0x18003E18C, "mov", "DWORD PTR [rcx],r9d"),
        (0x18003E18F, "mov", "DWORD PTR [rcx+0x4],ebx"),
        (0x18003E192, "add", "rcx,0x8"),
        (0x18003E1BC, "inc", "rbx"),
        (0x18003E1BF, "cmp", "rbx,rdi"),
        (0x18003E1C2, "jb", "0x18003e184"),
        (0x18003E1D2, "mov", "QWORD PTR [rsi+0x8],rcx"),
        (0x18003E1D6, "mov", "QWORD PTR [rsi],rax"),
    ):
        expect(code, address, mnemonic, operands)
    # The preceding pass temporarily uses r10 as a pointer, then resets it.
    # Check that its dirty interval cannot branch past the reset; otherwise
    # no call/nonzero r10 write is accepted before first outer selection.
    # This is a register-preservation audit, not an exhaustive CFG proof.
    for address, (mnemonic, operands) in code.items():
        if 0x1800380C9 <= address < 0x1800383F3:
            require(mnemonic != "call", "zero-register path contains a call")
            if 0x1800381C2 <= address < 0x180038254 and mnemonic.startswith("j"):
                require(0x180038192 <= int(operands, 16) <= 0x180038254,
                        "temporary pointer can bypass zero reset")
            if address == 0x1800381C2:
                continue  # checked pointer load; all exits must pass reset above
            operand_list = operands.split(",")
            destination = operand_list[0]
            fallback_aliases = ("r10", "r10d", "r10w", "r10b")
            if mnemonic in ("xchg", "xadd"):
                require(not any(operand in fallback_aliases for operand in operand_list),
                        "exchange writes selector fallback register")
            if destination in fallback_aliases:
                require(mnemonic in ("test", "cmp") or
                        (mnemonic == "xor" and operands == "r10d,r10d"),
                        "nonzero write to selector fallback register")
    return dict(outer_descriptor_stride_bytes=0xE0,
                exported_pair_bytes=8, exported_pair_low32="internal-layer-index",
                exported_pair_high32="internal-output-port",
                explicit_export_layers_field=0xB0, explicit_export_ports_field=0xC8,
                default_exports="last-layer-output-ports",
                missing_selector_fallback=0, out_of_range_export_fallback=0,
                tensor_index_table_field=0x1A8, tensor_index_element_bytes=4,
                routing_context="static tensor lifetime analysis",
                global_descriptor_numbering_verified=False,
                actual_tensor_indices_verified=False, cuda_addresses_verified=False)


def split_swin_configuration(table: tuple[int, ...]) -> None:
    require(table == (-1, -1, 1, 1, 1, 0, 0, 1) * 2,
            "split-Swin configuration table mismatch")


def source_export_contract(code: dict[int, tuple[str, str]]) -> dict[str, Any]:
    """Bounded append schedule and pre-transformer exports, not a CFG proof.

    Slot numbers are conditional on the inspected successful construction path
    and helper append semantics; no runtime descriptor or CUDA tensor is read.
    """
    for address, mnemonic, operands in (
        (0x1800397C8, "xor", "r8d,r8d"),
        (0x18003981B, "lea", "rbx,[rcx+0x70]"),
        (0x18003981F, "mov", "QWORD PTR [rbx],r8"),
        (0x180039822, "mov", "QWORD PTR [rbx+0x8],r8"),
        (0x180039826, "mov", "QWORD PTR [rbx+0x10],r8"),
        (0x180039AF3, "xor", "esi,esi"),
        (0x18003A7B6, "inc", "rsi"), (0x18003A7B9, "cmp", "rsi,0x6"),
        (0x18003A7BD, "jl", "0x18003a600"),
        (0x18003A7C3, "xor", "esi,esi"),
        (0x18003A98D, "inc", "rsi"), (0x18003A990, "cmp", "rsi,0x8"),
        (0x18003A994, "jl", "0x18003a7d0"),
        # A third eight-entry configuration table drives the split-Swin factory.
        (0x18003A99A, "xor", "r15d,r15d"),
        (0x18003A99D, "lea", "rsi,[rip+0x778ac]"),
        (0x18003A9A4, "mov", "r14,rsi"),
        (0x18003A9A7, "lea", "r13,[rip+0x778e2]"),
        (0x18003A9B0, "mov", "r9d,DWORD PTR [r14+0x4]"),
        (0x18003A9B4, "mov", "ecx,DWORD PTR [r14]"),
        (0x18003A9B7, "mov", "r8d,r9d"),
        (0x18003A9BA, "shr", "r8d,0x1f"),
        (0x18003A9BE, "xor", "r8b,0x1"),
        (0x18003A9C2, "cmp", "r15d,0x7"),
        (0x18003A9C6, "sete", "dil"),
        (0x18003A9CA, "test", "r15d,r15d"),
        (0x18003A9CD, "sete", "al"),
        (0x18003A9D0, "mov", "BYTE PTR [rsp+0x40],0x0"),
        (0x18003A9D5, "mov", "BYTE PTR [rsp+0x38],dil"),
        (0x18003A9DA, "mov", "BYTE PTR [rsp+0x30],dil"),
        (0x18003A9DF, "mov", "BYTE PTR [rsp+0x28],al"),
        (0x18003A9E3, "mov", "DWORD PTR [rsp+0x20],ecx"),
        (0x18003A9E7, "mov", "edx,0x200"),
        (0x18003A9EC, "lea", "rcx,[rbp-0x20]"),
        (0x18003A9F0, "call", "0x180040e20"),
        (0x18003A9F6, "mov", "rdx,rax"),
        (0x18003A9F9, "mov", "rcx,rbx"),
        (0x18003A9FC, "call", "0x18003db60"),
        (0x18003AA3F, "inc", "r15d"),
        (0x18003AA42, "add", "r14,0x8"),
        (0x18003AA46, "cmp", "r14,r13"),
        (0x18003AA49, "jl", "0x18003a9b0"),
        # Windows x64 stack argument locations after pushes/frame allocation.
        (0x180040E2D, "lea", "rbp,[rsp-0x828]"),
        (0x180040E35, "sub", "rsp,0x928"),
        (0x180040E52, "movzx", "r14d,r8b"),
        (0x180040E59, "mov", "rsi,rcx"),
        (0x180040E7D, "xor", "eax,eax"),
        (0x180040E94, "mov", "QWORD PTR [rcx+0x20],rax"),
        (0x180040E98, "mov", "QWORD PTR [rcx+0x28],rax"),
        (0x180040E9C, "mov", "QWORD PTR [rcx+0x30],rax"),
        (0x180040F3F, "call", "0x18006ad60"),
        (0x180040F44, "mov", "r8,rax"),
        (0x180040F57, "call", "0x180031df0"),
        (0x180040F5C, "add", "QWORD PTR [rsi+0x28],0x170"),
        (0x180040F64, "jmp", "0x180040f73"),
        (0x180040F6D, "call", "0x18002f150"),
        (0x180040F8B, "call", "0x18006b3a0"),
        (0x180041294, "call", "0x18001ea60"),
        (0x180041299, "add", "QWORD PTR [rsi+0x28],0x170"),
        (0x1800412A1, "jmp", "0x1800412b4"),
        (0x1800412AF, "call", "0x180040070"),
        (0x1800412CF, "call", "0x18006c130"),
        (0x180041505, "call", "0x18001ea60"),
        (0x18004150A, "add", "QWORD PTR [rsi+0x28],0x170"),
        (0x180041512, "jmp", "0x180041527"),
        (0x180041522, "call", "0x180040070"),
        (0x180040EEE, "mov", "QWORD PTR [rcx+0xb0],rax"),
        (0x180040EF5, "mov", "QWORD PTR [rcx+0xb8],rax"),
        (0x180040F03, "mov", "QWORD PTR [rcx+0xc8],rax"),
        (0x180040F0A, "mov", "QWORD PTR [rcx+0xd0],rax"),
        (0x180041527, "movzx", "r14d,BYTE PTR [rbp+0x8a0]"),
        (0x180041532, "test", "r14b,r14b"),
        (0x180041535, "je", "0x18004154b"),
        (0x18004153E, "call", "0x18006bdf0"),
        (0x180041549, "jmp", "0x180041562"),
        (0x180041558, "call", "0x18006bbc0"),
        (0x180041566, "mov", "rdx,rax"),
        (0x180041569, "lea", "rcx,[rbp+0x250]"),
        (0x180041570, "call", "0x180031df0"),
        (0x180041688, "lea", "rdx,[rbp+0x250]"),
        (0x180041692, "call", "0x18001ea60"),
        (0x180041697, "add", "QWORD PTR [rsi+0x28],0x170"),
        (0x18004169F, "jmp", "0x1800416b4"),
        (0x1800416A1, "lea", "r8,[rbp+0x250]"),
        (0x1800416AF, "call", "0x180040070"),
        (0x1800416C7, "call", "0x18006b890"),
        (0x18004177A, "lea", "rdx,[rbp+0x3c0]"),
        (0x180041784, "call", "0x18001ea60"),
        (0x180041789, "add", "QWORD PTR [rsi+0x28],0x170"),
        (0x180041791, "jmp", "0x1800417a6"),
        (0x180041793, "lea", "r8,[rbp+0x3c0]"),
        (0x1800417A1, "call", "0x180040070"),
        (0x1800416B4, "test", "r14b,r14b"),
        (0x1800416B7, "je", "0x180041826"),
        (0x1800417A6, "cmp", "BYTE PTR [rbp+0x8a8],0x0"),
        (0x1800417AD, "je", "0x180041818"),
        (0x1800417AF, "mov", "QWORD PTR [rsp+0x30],0x4"),
        (0x1800417B8, "mov", "QWORD PTR [rsp+0x38],0x3"),
        (0x1800417C1, "lea", "rax,[rsp+0x30]"),
        (0x1800417C6, "mov", "QWORD PTR [rsp+0x40],rax"),
        (0x1800417CB, "lea", "rax,[rsp+0x40]"),
        (0x1800417D0, "mov", "QWORD PTR [rsp+0x48],rax"),
        (0x1800417D5, "lea", "rdx,[rsp+0x40]"),
        (0x1800417DA, "lea", "rcx,[rsi+0xb0]"),
        (0x1800417E1, "call", "0x180032650"),
        (0x1800417E6, "xor", "eax,eax"),
        (0x1800417E8, "mov", "QWORD PTR [rsp+0x30],rax"),
        (0x1800417ED, "mov", "QWORD PTR [rsp+0x38],rax"),
        (0x1800417F2, "lea", "rax,[rsp+0x30]"),
        (0x1800417F7, "mov", "QWORD PTR [rsp+0x40],rax"),
        (0x1800417FC, "lea", "rax,[rsp+0x40]"),
        (0x180041801, "mov", "QWORD PTR [rsp+0x48],rax"),
        (0x180041806, "lea", "rdx,[rsp+0x40]"),
        (0x18004180B, "lea", "rcx,[rsi+0xc8]"),
        (0x180041812, "call", "0x180032650"),
        (0x180041B02, "mov", "rax,rsi"),
        # Fast outer append paths preserve export range endpoints, not just bases.
        (0x1800319E9, "mov", "QWORD PTR [r9+0x20],rax"),
        (0x1800319ED, "mov", "QWORD PTR [r9+0x28],rcx"),
        (0x180031AED, "mov", "rcx,QWORD PTR [r8+0xb8]"),
        (0x180031B10, "mov", "QWORD PTR [r9+0xb8],rcx"),
        (0x180031B2C, "mov", "rcx,QWORD PTR [r8+0xd0]"),
        (0x180031B4F, "mov", "QWORD PTR [r9+0xd0],rcx"),
        (0x18001EA09, "lea", "rcx,[r15+0xb0]"),
        (0x18001EA10, "lea", "rdx,[rbp+0xb0]"),
        (0x18001EA1D, "lea", "rcx,[r15+0xc8]"),
        (0x18001EA24, "lea", "rdx,[rbp+0xc8]"),
        # Empty projection metadata is transferred on its fast move path.
        (0x180071F09, "call", "0x18005fd60"),
        (0x18005FD60, "xor", "eax,eax"),
        (0x18005FDEF, "mov", "QWORD PTR [rcx+0xd8],rax"),
        (0x18005FDF6, "mov", "QWORD PTR [rcx+0xe0],rax"),
        (0x18005FDFD, "mov", "QWORD PTR [rcx+0xe8],rax"),
        (0x180031DF0, "mov", "r8,rdx"),
        (0x180031DF3, "mov", "r9,rcx"),
        (0x180031FB8, "mov", "rcx,QWORD PTR [r8+0xe0]"),
        (0x180031FC6, "mov", "rax,QWORD PTR [r8+0xd8]"),
        (0x180031FD4, "mov", "QWORD PTR [r9+0xd8],rax"),
        (0x180031FDB, "mov", "QWORD PTR [r9+0xe0],rcx"),
        (0x18001EB5C, "mov", "rcx,QWORD PTR [rdi+0xe0]"),
        (0x18001EB63, "sub", "rcx,QWORD PTR [rdi+0xd8]"),
        (0x18001EB84, "je", "0x18001ec42"),
    ):
        expect(code, address, mnemonic, operands)

    # The first three local layers append exactly one descriptor per capacity
    # alternative before projection(layer 3)/optional final-head(layer 4).
    for load, branch, copy, source, fallback, end in (
        (0x180040F47, 0x180040F4F, 0x180040F57, None, 0x180040F66, 0x180040F73),
        (0x180041282, 0x18004128A, 0x180041294, "[rsp+0x70]", 0x1800412A3, 0x1800412B4),
        (0x1800414F1, 0x1800414F9, 0x180041505, "[rbp+0xe0]", 0x180041514, 0x180041527),
    ):
        expect(code, load, "mov", "rax,QWORD PTR [rsi+0x28]")
        expect(code, load + 4, "cmp", "rax,QWORD PTR [rsi+0x30]")
        expect(code, branch, "je", hex(fallback))
        if source is None:
            expect(code, copy - 6, "mov", "rdx,r8")
            expect(code, fallback, "mov", "rdx,rax")
            expect(code, fallback + 3, "lea", "rcx,[rsi+0x20]")
        else:
            expect(code, copy - (8 if source.startswith("[rsp") else 10), "lea", f"rdx,{source}")
            expect(code, fallback, "lea", f"r8,{source}")
            lea_size = 5 if source.startswith("[rsp") else 7
            expect(code, fallback + lea_size, "mov", "rdx,rax")
            expect(code, fallback + lea_size + 3, "lea", "rcx,[rsi+0x20]")
        expect(code, copy - 3, "mov", "rcx,rax")
        expect(code, copy + 13, "jmp", hex(end))

    for address, target in ((0x18006BA54, 0x1800B2530),
                            (0x18006BCCC, 0x1800B2500),
                            (0x18006BFBF, 0x1800B2518)):
        mnemonic, operands = code[address]
        require(mnemonic == "lea" and operands.startswith("rdx,[rip+0x"),
                "missing split-Swin type assignment")
        displacement = int(operands.split("+", 1)[1].split("]", 1)[0], 16)
        require(address + 7 + displacement == target, "wrong split-Swin type target")

    # Nine straight-line append sites before the 6/8/8/8 loops. Capacity
    # branches are alternatives, not two different descriptors.
    for copy_call in (0x180039BC8, 0x180039DF0, 0x180039F01,
                      0x18003A00E, 0x18003A111):
        expect(code, copy_call - 17, "mov", "rax,QWORD PTR [rbx+0x8]")
        expect(code, copy_call - 13, "cmp", "rax,QWORD PTR [rbx+0x10]")
        expect(code, copy_call - 9, "je", hex(copy_call + 15))
        expect(code, copy_call - 7, "lea", "rdx,[rbp-0x20]")
        expect(code, copy_call - 3, "mov", "rcx,rax")
        expect(code, copy_call, "call", "0x18001e8a0")
        expect(code, copy_call + 5, "add", "QWORD PTR [rbx+0x8],0xe0")
        expect(code, copy_call + 13, "jmp", hex(copy_call + 30))
        expect(code, copy_call + 15, "lea", "r8,[rbp-0x20]")
        expect(code, copy_call + 19, "mov", "rdx,rax")
        expect(code, copy_call + 22, "mov", "rcx,rbx")
        expect(code, copy_call + 25, "call", "0x18002eca0")
    for call in (0x18003A2F6, 0x18003A3A7, 0x18003A454, 0x18003A4F7,
                 0x18003A6C0, 0x18003A897):
        expect(code, call - 7, "lea", "rdx,[rbp-0x20]")
        expect(code, call - 3, "mov", "rcx,rbx")
        expect(code, call, "call", "0x18003dba0")
    return dict(inspected_append_schedule=[9, 6, 8, 8, 8],
                conditional_split_swin_slots=[23, 30],
                conditional_primary_transformer_slots=[31, 38],
                split_swin_final_explicit_exports=[dict(internal_layer=4, output_port=0, type="CCSplitSwin16HFinalHead"),
                                                  dict(internal_layer=3, output_port=0, type="CCSplitSwin16HProjPool")],
                conditional_decoder_sources=[dict(source=38, export=0, internal_layer=4, output_port=0),
                                             dict(source=30, export=1, internal_layer=3, output_port=0)],
                numbering_basis="inspected successful append schedule; not exhaustive CFG/helper proof",
                final_split_swin_export_flags=True,
                subsequent_descriptor_mutation_excluded=False,
                runtime_source_exports_verified=False, gpu_tensor_bindings_verified=False)


def disassemble(data: bytes, ranges: tuple[tuple[int, int], ...] = RANGES) -> dict[int, tuple[str, str]]:
    # objdump reads the very same hash-verified bytes, not a reopened mutable path.
    # memfd is anonymous RAM-backed storage; it still does not prevent swapping.
    descriptor = os.memfd_create("graph-input", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        remaining = memoryview(data)
        while remaining:
            written = os.write(descriptor, remaining)
            require(written > 0, "short memfd write")
            remaining = remaining[written:]
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS,
                    fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
        code = {}
        for start, end in ranges:
            completed = subprocess.run(["objdump", "-d", "-M", "intel", "-j", ".text",
                f"--start-address={start}", f"--stop-address={end}", f"/proc/self/fd/{descriptor}"],
                pass_fds=(descriptor,), capture_output=True, text=True, check=True, timeout=10)
            decoded = instructions(completed.stdout)
            require(not code.keys() & decoded.keys(), "overlapping evidence ranges")
            code.update(decoded)
        return code
    finally:
        os.close(descriptor)


def self_test() -> dict[str, Any]:
    require(instructions("  180000000:\t48 89 01\tmov QWORD PTR [rcx],rax\n") ==
            {0x180000000: ("mov", "QWORD PTR [rcx],rax")}, "instruction parser known answer")
    rejected = 0
    for text in ("not disassembly", "  1: 90 nop\n  1: 90 nop\n"):
        try:
            instructions(text)
        except ValueError:
            rejected += 1
    require(rejected == 2, "parser negative controls failed")
    split_swin_configuration((-1, -1, 1, 1, 1, 0, 0, 1) * 2)
    table_rejected = 0
    for table in ((), (-1, -1, 1, 1, 1, 0, 0, 1), (-1, -1, 1, 1, 1, 0, 0, -1) * 2):
        try:
            split_swin_configuration(table)
        except ValueError:
            table_rejected += 1
    require(table_rejected == 3, "configuration table controls failed")
    return dict(parser_known_answers=1, parser_negative_controls_rejected=rejected,
                configuration_known_answers=1, configuration_negative_controls_rejected=table_rejected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("self-test", "inspect", "negative"))
    parser.add_argument("dll", type=Path, nargs="?")
    args = parser.parse_args()
    if args.mode == "self-test":
        print(json.dumps(self_test(), indent=2))
        return
    require(args.dll is not None, "DLL required")
    policy = runtime.protect_process_memory()
    policy["disassembler_same_uid_proc_exposure_controlled"] = False
    data = runtime.read_bounded(args.dll)
    require(hashlib.sha256(data).hexdigest() == runtime.PIN, "unpinned DLL")
    code = disassemble(data)
    pe = runtime.PE(data)
    for address, expected in ((0x1800B2818, "CCVit1DFfnExpand"),
                              (0x1800B2830, "CCVit1DFfnContract"),
                              (0x1800B2848, "CCVit1DQKV"),
                              (0x1800B2858, "CCVit1DAttention"),
                              (0x1800B2870, "CCVit1DProjection"),
                              (0x1800B0038, "CCDecInputUpsample"),
                              (0x1800B2358, "single_layer_block"),
                               (0x1800B2530, "CCSplitSwin16HFinalHead"),
                               (0x1800B2500, "CCSplitSwin16HProj"),
                               (0x1800B2518, "CCSplitSwin16HProjPool")):
        require(runtime.string_at(data, pe.offset(address - pe.image_base, len(expected) + 1)) == expected,
                "unexpected stage type string")
    # Metadata table is not a weight payload. Check eight signed pairs in place.
    table = runtime.unpack("<16i", data, pe.offset(0xB2250, 64))
    split_swin_configuration(table)
    report = contract(code)
    if args.mode == "negative":
        rejected = 0
        for address, replacement in ((0x1800444AA, ("mov", "QWORD PTR [rsp+0x30],0x2")),
                                     (0x180044385, ("mov", "QWORD PTR [rsp+0x40],rsi")),
                                     (0x1800444A4, ("call", "0x180071b90")),
                                     (0x18002B0B5, ("mov", "rax,QWORD PTR [rcx]")),
                                     (0x1800441FC, ("lea", "rax,[rcx+0x10]")),
                                     (0x18004449D, ("lea", "rcx,[rbp+0x230]")),
                                      (0x18003AB06, ("cmp", "edi,0x7")),
                                      (0x18003AA67, ("je", "0x18003aab9")),
                                      (0x18003AB62, ("mov", "QWORD PTR [rbp-0x68],0x26")),
                                      (0x18003ABAD, ("lea", "rcx,[rbp+0x18]")),
                                      (0x18003AB3B, ("call", "0x180071b90")),
                                      (0x180032696, ("mov", "rdx,rbx")),
                                      (0x18003AB73, ("lea", "rax,[rbp-0x68]")),
                                      (0x1800742C8, ("mov", "eax,DWORD PTR [rsp+0x48]")),
                                      (0x18003DB7B, ("add", "QWORD PTR [rbx+0x8],0x170")),
                                      (0x1800383EA, ("mov", "rdi,QWORD PTR [rdi]")),
                                      (0x180038429, ("cmovae", "rax,rdi")),
                                      (0x180038453, ("shr", "rdi,0x10")),
                                      (0x18003845D, ("movsxd", "rdx,DWORD PTR [rax+rcx*8]")),
                                      (0x18003E09D, ("mov", "r9d,DWORD PTR [r10+rdx*1]")),
                                      (0x18003E13D, ("inc", "r9d")),
                                      (0x18003E157, ("mov", "edi,0x0")),
                                      (0x18003E096, ("jmp", "0x18003e098")),
                                      (0x18003E081, ("sar", "rax,0x2")),
                                      (0x1800380C9, ("mov", "r10d,0x1")),
                                      (0x180038254, ("mov", "r10d,0x1")),
                                      (0x180038350, ("xchg", "rax,r10")),
                                      (0x180039822, ("mov", "QWORD PTR [rbx+0x8],0xe0")),
                                      (0x18003A7B9, ("cmp", "rsi,0x5")),
                                      (0x18003A990, ("cmp", "rsi,0x7")),
                                      (0x18003A9FC, ("call", "0x18003dba0")),
                                      (0x18003A9DA, ("mov", "BYTE PTR [rsp+0x30],al")),
                                      (0x18003A9D5, ("mov", "BYTE PTR [rsp+0x38],al")),
                                      (0x18003AA42, ("add", "r14,0x10")),
                                      (0x1800417AD, ("jne", "0x180041818")),
                                      (0x1800417B8, ("mov", "QWORD PTR [rsp+0x38],0x4")),
                                      (0x1800417CB, ("lea", "rax,[rsp+0x38]")),
                                      (0x18004180B, ("lea", "rcx,[rsi+0xb0]")),
                                      (0x18005FDF6, ("mov", "QWORD PTR [rcx+0xe0],rcx")),
                                      (0x180031FDB, ("mov", "QWORD PTR [r9+0xe0],rax")),
                                      (0x180039BCD, ("add", "QWORD PTR [rbx+0x8],0x170")),
                                      (0x180039BBF, ("je", "0x180039be6")),
                                      (0x180031B10, ("mov", "QWORD PTR [r9+0xb8],rax")),
                                      (0x180031B4F, ("mov", "QWORD PTR [r9+0xd0],rax")),
                                      (0x18004153E, ("call", "0x18006bbc0")),
                                      (0x1800416C7, ("call", "0x18006bdf0")),
                                      (0x18006BFBF, ("lea", "rdx,[rip+0x4f040]")),
                                      (0x180041532, ("test", "r15b,r15b")),
                                      (0x180041299, ("add", "QWORD PTR [rsi+0x28],0x2e0")),
                                      (0x180041512, ("jmp", "0x1800416b4")),
                                      (0x180040E98, ("mov", "QWORD PTR [rcx+0x28],rcx"))):
            modified = dict(code)
            modified[address] = replacement
            try:
                contract(modified)
            except ValueError:
                rejected += 1
        require(rejected == 51, "semantic corruption not rejected")
        report = dict(static_semantic_corruptions_rejected=rejected, gpu_kernels_launched=0)
    print(json.dumps(dict(mode=args.mode, dll_sha256=runtime.PIN, memory_policy=policy,
                         cpu_self_test=self_test(), graph=report), indent=2))


if __name__ == "__main__":
    main()
