#!/usr/bin/env python3
"""Bounded static decoder launch-argument recovery; no CUDA or Windows execution.

Reads pinned bytes as data; emits metadata only. Not a verified activation
layout, numerical decoder test or complete native neural-rendered image.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
from typing import Any

spec = importlib.util.spec_from_file_location("graph", Path(__file__).with_name("native-graph-probe.py"))
if spec is None or spec.loader is None:
    raise RuntimeError("missing graph helper")
graph = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph)
runtime = graph.runtime
require = runtime.require
RANGES = ((0x180073BF0, 0x180073D43), (0x1800744B0, 0x18007470F),
          (0x180074770, 0x1800747B1), (0x1800749F0, 0x180074B62),
          (0x1800447B0, 0x1800448AB), (0x1800449A0, 0x180044A8C),
          (0x180076690, 0x1800766B7), (0x180078ED0, 0x180078EEA),
          (0x1800793D0, 0x1800793DE), (0x180079792, 0x1800797B4), (0x18004D1B0, 0x18004D1F9),
          (0x1800776D0, 0x18007771B), (0x18007785E, 0x180077872), (0x18007791E, 0x18007793B),
          (0x1800784D0, 0x1800785FA),
          (0x180074710, 0x180074770), (0x1800747c0, 0x180074814), (0x180038d5e, 0x180038d87), (0x180038d87, 0x180038dde), (0x1800391b2, 0x1800391cf), (0x180039217, 0x180039258),
          (0x1800391cf, 0x180039217), (0x18002eba0, 0x18002ebb4), (0x18002ec7a, 0x18002ec8d))
ARENA_RANGES = ((6442638874, 6442638968), (6442639336, 6442639360), (6442639600, 6442639649), (6442639869, 6442639882), (6442640176, 6442640217), (6442685249, 6442685360), (6442702736, 6442702751), (6442706144, 6442706300), (6442707105, 6442707151), (6442707519, 6442707536), (6442707665, 6442707684), (6442707716, 6442707789), (6442707925, 6442708111), (6442708205, 6442708235), (6442708445, 6442708481), (6442631648, 6442631679))
ARENA_RANGES += ((6442639325, 6442639336), (6442639556, 6442639600), (6442707368, 6442707519), (6442707536, 6442707665))
RANGES += ARENA_RANGES
KERNELS = (
    (0xBC3D0, "cc_dec_input_upsample_1024_512_fp8"),
    (0xBC3F8, "cc_dec_input_upsample_1024_512"),
    (0xBC418, "cc_dec_input_upsample_1024_512_tilesync_fp8"),
    (0xBC448, "cc_dec_input_upsample_1024_512_tilesync"),
)


class DecoderArguments(ct.Structure):
    # Slot provenance only: GPU element datatypes/layouts are NOT established.
    _fields_ = [("main", ct.c_uint64), ("skip", ct.c_uint64),
                ("output0", ct.c_uint64), ("output3", ct.c_uint64),
                ("output1", ct.c_uint64), ("conditional_output2", ct.c_uint64),
                ("optional_output4", ct.c_uint64), ("weight_state0", ct.c_uint64),
                ("input_axis0", ct.c_int32), ("input_axis1", ct.c_int32),
                ("output_axis0", ct.c_int32), ("output_axis1", ct.c_int32)]


# Bounded instruction anchors are populated below; no general control-flow claim.
ANCHORS: tuple[tuple[int, str, str], ...] = (
    (0x180073c30, 'lea', 'rax,[rip+0x48671]'),
    (0x180073c37, 'mov', 'QWORD PTR [rdi],rax'),
    (0x180073c3a, 'mov', 'ebp,DWORD PTR [rbx+0x50]'),
    (0x180073c3d, 'mov', 'DWORD PTR [rdi+0x178],ebp'),
    (0x180073c43, 'mov', 'esi,DWORD PTR [rbx+0x54]'),
    (0x180073c46, 'mov', 'DWORD PTR [rdi+0x17c],esi'),
    (0x180073c4c, 'xor', 'r14d,r14d'),
    (0x180073c4f, 'mov', 'QWORD PTR [rdi+0x180],r14'),
    (0x180073c56, 'mov', 'QWORD PTR [rdi+0x188],r14'),
    (0x180073c70, 'cmp', 'r8,0x12'),
    (0x180073c74, 'jne', '0x180073d0e'),
    (0x180073c7a, 'lea', 'rdx,[rip+0x3c3b7]'),
    (0x180073c81, 'call', '0x18007e450'),
    (0x180073c86, 'test', 'eax,eax'),
    (0x180073c88, 'jne', '0x180073d0e'),
    (0x180073c8e, 'cmp', 'ebp,0x400'),
    (0x180073c94, 'jne', '0x180073ceb'),
    (0x180073c96, 'cmp', 'esi,0x200'),
    (0x180073c9c, 'jne', '0x180073ceb'),
    (0x1800744b0, 'mov', 'r11,rsp'),
    (0x1800744b3, 'push', 'rbx'),
    (0x1800744b4, 'sub', 'rsp,0xd0'),
    (0x1800744bb, 'mov', 'rbx,QWORD PTR [rdx]'),
    (0x1800744be, 'mov', 'r9,r8'),
    (0x1800744c1, 'mov', 'rax,QWORD PTR [rdx+0x8]'),
    (0x1800744c5, 'mov', 'r10,rcx'),
    (0x1800744c8, 'sub', 'rax,rbx'),
    (0x1800744cb, 'sar', 'rax,0x3'),
    (0x1800744cf, 'cmp', 'rax,0x2'),
    (0x1800744d3, 'jb', '0x1800746ec'),
    (0x1800744d9, 'mov', 'r8,QWORD PTR [r8]'),
    (0x1800744dc, 'mov', 'rcx,QWORD PTR [r9+0x8]'),
    (0x1800744e0, 'sub', 'rcx,r8'),
    (0x1800744e3, 'sar', 'rcx,0x3'),
    (0x1800744e7, 'cmp', 'rcx,0x4'),
    (0x1800744eb, 'jb', '0x1800746c9'),
    (0x1800744f1, 'mov', 'r9,QWORD PTR [r10+0x188]'),
    (0x180074504, 'mov', 'edi,DWORD PTR [rsp+0x108]'),
    (0x18007450b, 'mov', 'QWORD PTR [r11-0x18],r14'),
    (0x18007450f, 'mov', 'r11d,DWORD PTR [rsp+0x100]'),
    (0x180074517, 'lea', 'eax,[r11+0x3]'),
    (0x18007451b, 'cdq', ''),
    (0x18007451c, 'and', 'edx,0x3'),
    (0x18007451f, 'lea', 'esi,[rdx+rax*1]'),
    (0x180074522, 'lea', 'eax,[rdi+0x3]'),
    (0x180074525, 'and', 'esi,0xfffffffc'),
    (0x180074528, 'cdq', ''),
    (0x180074529, 'and', 'edx,0x3'),
    (0x18007452c, 'lea', 'ebp,[rdx+rax*1]'),
    (0x18007452f, 'mov', 'eax,DWORD PTR [r10+0x180]'),
    (0x180074536, 'and', 'ebp,0xfffffffc'),
    (0x180074539, 'test', 'eax,eax'),
    (0x18007453b, 'jle', '0x180074542'),
    (0x18007453d, 'add', 'eax,0x3'),
    (0x180074540, 'jmp', '0x18007454a'),
    (0x180074542, 'lea', 'eax,[r11*2+0x3]'),
    (0x18007454a, 'cdq', ''),
    (0x18007454b, 'and', 'edx,0x3'),
    (0x18007454e, 'add', 'eax,edx'),
    (0x180074550, 'sar', 'eax,0x2'),
    (0x180074553, 'lea', 'r14d,[rax*4+0x0]'),
    (0x18007455b, 'mov', 'eax,DWORD PTR [r10+0x184]'),
    (0x180074562, 'test', 'eax,eax'),
    (0x180074564, 'jle', '0x18007456b'),
    (0x180074566, 'add', 'eax,0x3'),
    (0x180074569, 'jmp', '0x180074572'),
    (0x18007456b, 'lea', 'eax,[rdi*2+0x3]'),
    (0x180074572, 'mov', 'r10,QWORD PTR [r8+0x10]'),
    (0x180074576, 'cdq', ''),
    (0x180074577, 'mov', 'rdi,QWORD PTR [rsp+0xc8]'),
    (0x18007457f, 'and', 'edx,0x3'),
    (0x180074582, 'add', 'eax,edx'),
    (0x180074584, 'sar', 'eax,0x2'),
    (0x180074587, 'cmp', 'BYTE PTR [rip+0x6c143],0x0'),
    (0x18007458e, 'lea', 'r11d,[rax*4+0x0]'),
    (0x180074596, 'je', '0x1800745a8'),
    (0x180074598, 'test', 'r10,r10'),
    (0x18007459b, 'je', '0x1800745a8'),
    (0x18007459d, 'cmp', 'QWORD PTR [r9+0x10],0x0'),
    (0x1800745a2, 'je', '0x1800745a8'),
    (0x1800745a4, 'mov', 'dl,0x1'),
    (0x1800745a6, 'jmp', '0x1800745aa'),
    (0x1800745a8, 'xor', 'dl,dl'),
    (0x1800745aa, 'mov', 'rax,QWORD PTR [rbx]'),
    (0x1800745ad, 'mov', 'QWORD PTR [rsp+0x70],rax'),
    (0x1800745b2, 'mov', 'rax,QWORD PTR [rbx+0x8]'),
    (0x1800745b6, 'mov', 'QWORD PTR [rsp+0x78],rax'),
    (0x1800745bb, 'mov', 'rax,QWORD PTR [r8]'),
    (0x1800745be, 'mov', 'QWORD PTR [rsp+0x80],rax'),
    (0x1800745c6, 'mov', 'rax,QWORD PTR [r8+0x8]'),
    (0x1800745ca, 'mov', 'QWORD PTR [rsp+0x90],rax'),
    (0x1800745d2, 'xor', 'eax,eax'),
    (0x1800745d4, 'test', 'dl,dl'),
    (0x1800745d6, 'cmovne', 'rax,r10'),
    (0x1800745da, 'mov', 'QWORD PTR [rsp+0x98],rax'),
    (0x1800745e2, 'mov', 'rax,QWORD PTR [r8+0x18]'),
    (0x1800745e6, 'mov', 'QWORD PTR [rsp+0x88],rax'),
    (0x1800745ee, 'cmp', 'rcx,0x4'),
    (0x1800745f2, 'jbe', '0x180074602'),
    (0x1800745f4, 'mov', 'rax,QWORD PTR [r8+0x20]'),
    (0x1800745f8, 'mov', 'QWORD PTR [rsp+0xa0],rax'),
    (0x180074600, 'jmp', '0x18007460e'),
    (0x180074602, 'mov', 'QWORD PTR [rsp+0xa0],0x0'),
    (0x18007460e, 'mov', 'rax,QWORD PTR [r9]'),
    (0x180074611, 'mov', 'DWORD PTR [rsp+0xb8],r14d'),
    (0x180074619, 'mov', 'r14,QWORD PTR [rsp+0xc0]'),
    (0x180074621, 'mov', 'QWORD PTR [rsp+0xa8],rax'),
    (0x180074629, 'mov', 'DWORD PTR [rsp+0xb0],esi'),
    (0x180074630, 'mov', 'DWORD PTR [rsp+0xb4],ebp'),
    (0x180074637, 'mov', 'DWORD PTR [rsp+0xbc],r11d'),
    (0x18007463f, 'test', 'dl,dl'),
    (0x180074641, 'je', '0x180074649'),
    (0x180074643, 'mov', 'r11,QWORD PTR [r9+0x10]'),
    (0x180074647, 'jmp', '0x18007464d'),
    (0x180074649, 'mov', 'r11,QWORD PTR [r9+0x8]'),
    (0x18007464d, 'lea', 'rax,[rsp+0x70]'),
    (0x180074652, 'mov', 'BYTE PTR [rsp+0x38],0x0'),
    (0x180074657, 'mov', 'QWORD PTR [rsp+0xe8],rax'),
    (0x18007465f, 'lea', 'eax,[rsi+0x3]'),
    (0x180074662, 'mov', 'rcx,QWORD PTR [r9+0x18]'),
    (0x180074666, 'cdq', ''),
    (0x180074667, 'and', 'edx,0x3'),
    (0x18007466a, 'mov', 'QWORD PTR [rsp+0x30],0x50'),
    (0x180074673, 'mov', 'r10,QWORD PTR [rcx]'),
    (0x180074676, 'lea', 'r9d,[rdx+rax*1]'),
    (0x18007467a, 'lea', 'eax,[rbp+0x3]'),
    (0x18007467d, 'sar', 'r9d,0x2'),
    (0x180074681, 'cdq', ''),
    (0x180074682, 'and', 'edx,0x3'),
    (0x180074685, 'add', 'eax,edx'),
    (0x180074687, 'lea', 'rdx,[rsp+0xe8]'),
    (0x18007468f, 'sar', 'eax,0x2'),
    (0x180074692, 'mov', 'QWORD PTR [rsp+0x28],rdx'),
    (0x180074697, 'mov', 'rdx,r11'),
    (0x18007469a, 'mov', 'DWORD PTR [rsp+0x20],0x4'),
    (0x1800746a2, 'lea', 'r8d,[rax+rax*1]'),
    (0x1800746a6, 'mov', 'rax,QWORD PTR [r10+0x28]'),
    (0x1800746aa, 'call', 'QWORD PTR [rip+0x37d78]'),
    (0x180074770, 'mov', 'rax,QWORD PTR [rsp+0x30]'),
    (0x180074775, 'mov', 'DWORD PTR [rax],edx'),
    (0x180074777, 'lea', 'eax,[r8+r8*1]'),
    (0x18007477b, 'mov', 'edx,DWORD PTR [rcx+0x180]'),
    (0x180074781, 'test', 'edx,edx'),
    (0x180074783, 'cmovle', 'edx,eax'),
    (0x180074786, 'mov', 'rax,QWORD PTR [rsp+0x38]'),
    (0x18007478b, 'mov', 'DWORD PTR [rax],edx'),
    (0x18007478d, 'lea', 'eax,[r9+r9*1]'),
    (0x180074791, 'mov', 'edx,DWORD PTR [rcx+0x184]'),
    (0x180074797, 'test', 'edx,edx'),
    (0x180074799, 'cmovle', 'edx,eax'),
    (0x18007479c, 'mov', 'rax,QWORD PTR [rsp+0x40]'),
    (0x1800747a1, 'mov', 'DWORD PTR [rax],edx'),
    (0x1800747a3, 'mov', 'rax,QWORD PTR [rsp+0x48]'),
    (0x1800747a8, 'mov', 'ecx,DWORD PTR [rcx+0x17c]'),
    (0x1800747ae, 'mov', 'DWORD PTR [rax],ecx'),
    (0x1800747b0, 'ret', ''),
    (0x1800749f0, 'push', 'r14'),
    (0x1800749f2, 'sub', 'rsp,0x60'),
    (0x1800749f6, 'mov', 'r14,rdx'),
    (0x1800749f9, 'test', 'rdx,rdx'),
    (0x1800749fc, 'je', '0x180074b3f'),
    (0x180074a02, 'cmp', 'QWORD PTR [rcx+0x20],0x10'),
    (0x180074a07, 'mov', 'rax,QWORD PTR [rcx+0x18]'),
    (0x180074a0b, 'mov', 'QWORD PTR [rsp+0x70],rbx'),
    (0x180074a10, 'mov', 'QWORD PTR [rsp+0x78],rbp'),
    (0x180074a15, 'mov', 'rbp,QWORD PTR [rcx+0x188]'),
    (0x180074a24, 'lea', 'rdi,[rcx+0x8]'),
    (0x180074a28, 'jb', '0x180074a2d'),
    (0x180074a2a, 'mov', 'rdi,QWORD PTR [rdi]'),
    (0x180074a2d, 'mov', 'QWORD PTR [rsp+0x80],rsi'),
    (0x180074a35, 'cmp', 'rax,0x4'),
    (0x180074a39, 'jb', '0x180074a8b'),
    (0x180074a3b, 'lea', 'rsi,[rax+rdi*1]'),
    (0x180074a3f, 'mov', 'edx,0x5f'),
    (0x180074a44, 'lea', 'r8,[rsi-0x3]'),
    (0x180074a48, 'mov', 'rcx,rdi'),
    (0x180074a4b, 'sub', 'r8,rdi'),
    (0x180074a4e, 'call', '0x18007e3d0'),
    (0x180074a53, 'mov', 'rbx,rax'),
    (0x180074a56, 'test', 'rax,rax'),
    (0x180074a59, 'je', '0x180074a8b'),
    (0x180074a60, 'cmp', 'DWORD PTR [rbx],0x3870665f'),
    (0x180074a66, 'je', '0x180074b37'),
    (0x180074a6c, 'inc', 'rbx'),
    (0x180074a6f, 'lea', 'r8,[rsi-0x3]'),
    (0x180074a73, 'sub', 'r8,rbx'),
    (0x180074a76, 'mov', 'rcx,rbx'),
    (0x180074a79, 'mov', 'edx,0x5f'),
    (0x180074a7e, 'call', '0x18007e3d0'),
    (0x180074a83, 'mov', 'rbx,rax'),
    (0x180074a86, 'test', 'rax,rax'),
    (0x180074a89, 'jne', '0x180074a60'),
    (0x180074a8b, 'mov', 'rbx,0xffffffffffffffff'),
    (0x180074a92, 'mov', 'rax,QWORD PTR [r14]'),
    (0x180074a95, 'lea', 'rcx,[rip+0x47934]'),
    (0x180074a9c, 'cmp', 'rbx,0xffffffffffffffff'),
    (0x180074aa0, 'lea', 'rdx,[rip+0x47951]'),
    (0x180074aa7, 'cmovne', 'rdx,rcx'),
    (0x180074aab, 'xor', 'edi,edi'),
    (0x180074aad, 'mov', 'rax,QWORD PTR [rax+0x20]'),
    (0x180074ab1, 'mov', 'rcx,r14'),
    (0x180074ab4, 'mov', 'QWORD PTR [rsp+0x28],rdi'),
    (0x180074ab9, 'mov', 'DWORD PTR [rsp+0x20],0x1'),
    (0x180074ac1, 'lea', 'r9d,[rdi+0x2]'),
    (0x180074ac5, 'lea', 'r8d,[rdi+0x20]'),
    (0x180074ac9, 'call', 'QWORD PTR [rip+0x37959]'),
    (0x180074acf, 'cmp', 'rbx,0xffffffffffffffff'),
    (0x180074ad3, 'mov', 'QWORD PTR [rsp+0x28],rdi'),
    (0x180074ad8, 'mov', 'QWORD PTR [rbp+0x8],rax'),
    (0x180074adc, 'lea', 'rcx,[rip+0x47935]'),
    (0x180074ae3, 'mov', 'rax,QWORD PTR [r14]'),
    (0x180074ae6, 'lea', 'rdx,[rip+0x4795b]'),
    (0x180074aed, 'cmovne', 'rdx,rcx'),
    (0x180074af1, 'mov', 'DWORD PTR [rsp+0x20],0x1'),
    (0x180074af9, 'lea', 'r9d,[rdi+0x2]'),
    (0x180074afd, 'mov', 'rcx,r14'),
    (0x180074b00, 'lea', 'r8d,[rdi+0x20]'),
    (0x180074b04, 'mov', 'rax,QWORD PTR [rax+0x20]'),
    (0x180074b08, 'call', 'QWORD PTR [rip+0x3791a]'),
    (0x180074b23, 'mov', 'QWORD PTR [rbp+0x10],rax'),
    (0x180074b27, 'mov', 'QWORD PTR [rbp+0x18],r14'),
    (0x180074b37, 'sub', 'rbx,rdi'),
    (0x180074b3a, 'jmp', '0x180074a92'),
)


# Corroborate the wrapper -> concrete CUDA backend -> cuLaunchKernel path.
# Conditional on these vtables; not a captured decoder/backend dynamic binding.
BACKEND_ANCHORS: tuple[tuple[int, str, str], ...] = (
    (0x1800447b6, "mov", "rbx,rcx"),
    (0x1800447d2, "mov", "QWORD PTR [rax+0x8],rbx"),
    (0x1800447d6, "lea", "rcx,[rip+0x6e103]"),
    (0x1800447dd, "mov", "QWORD PTR [rax],rcx"),
    (0x180044830, "rex", "push rbx"),
    (0x180044832, "push", "rdi"),
    (0x180044833, "sub", "rsp,0x158"),
    (0x18004484c, "mov", "rdi,rdx"),
    (0x18004484f, "mov", "rbx,rcx"),
    (0x18004485b, "mov", "rcx,QWORD PTR [rcx+0x8]"),
    (0x18004485f, "mov", "rax,QWORD PTR [rcx]"),
    (0x180044862, "mov", "rdx,QWORD PTR [rsp+0x198]"),
    (0x18004486a, "mov", "QWORD PTR [rsp+0x48],rdx"),
    (0x18004486f, "mov", "edx,DWORD PTR [rsp+0x190]"),
    (0x180044876, "mov", "DWORD PTR [rsp+0x40],edx"),
    (0x18004487a, "mov", "DWORD PTR [rsp+0x38],r9d"),
    (0x18004487f, "mov", "DWORD PTR [rsp+0x30],r8d"),
    (0x180044884, "lea", "rdx,[rsp+0x60]"),
    (0x180044889, "mov", "QWORD PTR [rsp+0x28],rdx"),
    (0x18004488e, "mov", "DWORD PTR [rsp+0x20],0x1"),
    (0x180044896, "mov", "r9,rdi"),
    (0x180044899, "mov", "r8d,DWORD PTR [rbx+0x20]"),
    (0x18004489d, "mov", "rdx,QWORD PTR [rbx+0x18]"),
    (0x1800448a1, "mov", "rax,QWORD PTR [rax+0x60]"),
    (0x1800448a5, "call", "QWORD PTR [rip+0x67b7d]"),
    (0x1800449a0, "rex", "push rbx"),
    (0x1800449a2, "push", "rbp"),
    (0x1800449a3, "push", "rsi"),
    (0x1800449a4, "push", "rdi"),
    (0x1800449a5, "sub", "rsp,0xe8"),
    (0x1800449be, "mov", "ebp,r9d"),
    (0x1800449c1, "mov", "esi,r8d"),
    (0x1800449c4, "mov", "rdi,rdx"),
    (0x1800449c7, "mov", "rbx,rcx"),
    (0x1800449ca, "mov", "rcx,QWORD PTR [rcx+0x8]"),
    (0x1800449d1, "cmp", "BYTE PTR [rsp+0x148],0x0"),
    (0x1800449d9, "je", "0x1800449fc"),
    (0x1800449fc, "xor", "r9d,r9d"),
    (0x1800449ff, "xor", "r8d,r8d"),
    (0x180044a02, "mov", "rdx,QWORD PTR [rbx+0x10]"),
    (0x180044a06, "mov", "rax,QWORD PTR [rax+0x150]"),
    (0x180044a0d, "call", "QWORD PTR [rip+0x67a15]"),
    (0x180044a13, "mov", "rcx,QWORD PTR [rbx+0x8]"),
    (0x180044a17, "mov", "rax,QWORD PTR [rcx]"),
    (0x180044a1a, "mov", "rdx,rdi"),
    (0x180044a1d, "mov", "rax,QWORD PTR [rax+0xd8]"),
    (0x180044a24, "call", "QWORD PTR [rip+0x679fe]"),
    (0x180044a2a, "test", "eax,eax"),
    (0x180044a2c, "jne", "0x180044b04"),
    (0x180044a32, "mov", "rax,QWORD PTR [rsp+0x138]"),
    (0x180044a3a, "mov", "rcx,QWORD PTR [rax]"),
    (0x180044a3d, "lea", "rax,[rip+0x6de74]"),
    (0x180044a44, "mov", "QWORD PTR [rsp+0x48],rax"),
    (0x180044a49, "mov", "QWORD PTR [rsp+0x50],rcx"),
    (0x180044a4e, "mov", "eax,DWORD PTR [rsp+0x140]"),
    (0x180044a55, "mov", "DWORD PTR [rsp+0x58],eax"),
    (0x180044a59, "mov", "rcx,QWORD PTR [rbx+0x8]"),
    (0x180044a5d, "mov", "rax,QWORD PTR [rcx]"),
    (0x180044a60, "mov", "edx,DWORD PTR [rsp+0x130]"),
    (0x180044a67, "mov", "DWORD PTR [rsp+0x28],edx"),
    (0x180044a6b, "mov", "DWORD PTR [rsp+0x20],ebp"),
    (0x180044a6f, "mov", "r9d,esi"),
    (0x180044a72, "mov", "r8,QWORD PTR [rbx+0x10]"),
    (0x180044a76, "lea", "rdx,[rsp+0x48]"),
    (0x180044a7b, "mov", "rax,QWORD PTR [rax+0x140]"),
    (0x180044a82, "call", "QWORD PTR [rip+0x679a0]"),
    (0x1800766a4, "mov", "rdi,rcx"),
    (0x1800766ad, "lea", "rax,[rip+0x46334]"),
    (0x1800766b4, "mov", "QWORD PTR [rdi],rax"),
    (0x180078ed6, "mov", "rbx,rcx"),
    (0x180078ede, "lea", "rcx,[rbx+0x408]"),
    (0x180078ee5, "call", "0x1800793d0"),
    (0x1800793d9, "mov", "rsi,rcx"),
    (0x180079792, "xor", "r9d,r9d"),
    (0x180079795, "lea", "rdx,[rsi+0x68]"),
    (0x18007979f, "lea", "rcx,[rip+0x43a2a]"),
    (0x1800797a6, "mov", "rax,rdi"),
    (0x1800797a9, "call", "QWORD PTR [rip+0x32c79]"),
    (0x18004d1e6, "mov", "rcx,QWORD PTR [rax+0x18]"),
    (0x18004d1ec, "mov", "QWORD PTR [rbx+0x140],rcx"),
    (0x1800776d5, "push", "rbp"),
    (0x1800776d6, "push", "rsi"),
    (0x1800776d7, "push", "rdi"),
    (0x1800776d8, "push", "r12"),
    (0x1800776da, "push", "r13"),
    (0x1800776dc, "push", "r14"),
    (0x1800776de, "push", "r15"),
    (0x1800776e0, "lea", "rbp,[rsp-0x7]"),
    (0x1800776e5, "sub", "rsp,0x90"),
    (0x1800776fa, "mov", "r15,r9"),
    (0x1800776fd, "mov", "r14,rcx"),
    (0x180077704, "mov", "r13,QWORD PTR [rbp+0x6f]"),
    (0x180077708, "mov", "r9,r13"),
    (0x18007770b, "mov", "r8,r15"),
    (0x18007770e, "call", "0x18004f480"),
    (0x180077713, "test", "al,al"),
    (0x180077715, "jne", "0x1800779e9"),
    (0x18007785e, "mov", "rbx,QWORD PTR [rbp+0x8f]"),
    (0x180077865, "cmp", "rbx,0xc000"),
    (0x18007786c, "jbe", "0x18007791e"),
    (0x18007791e, "mov", "eax,DWORD PTR [rbp+0x77]"),
    (0x180077921, "mov", "DWORD PTR [rdi+0x40],eax"),
    (0x180077924, "mov", "eax,DWORD PTR [rbp+0x7f]"),
    (0x180077927, "mov", "DWORD PTR [rdi+0x44],eax"),
    (0x18007792a, "mov", "eax,DWORD PTR [rbp+0x87]"),
    (0x180077930, "mov", "DWORD PTR [rdi+0x48],eax"),
    (0x180077933, "mov", "QWORD PTR [rdi+0x50],rbx"),
    (0x1800784d0, "rex", "push rbx"),
    (0x1800784d2, "push", "rsi"),
    (0x1800784d3, "push", "rdi"),
    (0x1800784d4, "sub", "rsp,0xa0"),
    (0x1800784f5, "mov", "esi,r9d"),
    (0x1800784f8, "mov", "rdi,rdx"),
    (0x1800784fb, "mov", "rbx,rcx"),
    (0x18007851d, "mov", "rcx,QWORD PTR [rbx+0x140]"),
    (0x180078524, "cmp", "eax,DWORD PTR [rcx]"),
    (0x180078526, "jne", "0x180078658"),
    (0x18007852c, "mov", "rax,QWORD PTR [rdi]"),
    (0x18007852f, "mov", "rcx,rdi"),
    (0x180078532, "mov", "rax,QWORD PTR [rax]"),
    (0x180078535, "call", "QWORD PTR [rip+0x33eed]"),
    (0x18007853b, "mov", "rax,QWORD PTR [rdi]"),
    (0x18007853e, "mov", "rcx,rdi"),
    (0x180078541, "mov", "rax,QWORD PTR [rax+0x18]"),
    (0x180078545, "call", "QWORD PTR [rip+0x33edd]"),
    (0x18007854b, "mov", "rcx,rdi"),
    (0x18007854e, "mov", "QWORD PTR [rsp+0x70],0x1"),
    (0x180078557, "mov", "eax,eax"),
    (0x180078559, "mov", "QWORD PTR [rsp+0x60],rax"),
    (0x18007855e, "mov", "rax,QWORD PTR [rdi]"),
    (0x180078561, "mov", "rax,QWORD PTR [rax+0x10]"),
    (0x180078565, "call", "QWORD PTR [rip+0x33ebd]"),
    (0x18007856b, "mov", "r8,QWORD PTR [rbx+0x3f8]"),
    (0x180078572, "lea", "rdx,[rsp+0x70]"),
    (0x180078577, "mov", "QWORD PTR [rsp+0x78],rax"),
    (0x18007857c, "xor", "edi,edi"),
    (0x18007857e, "mov", "QWORD PTR [rsp+0x50],rdx"),
    (0x180078583, "lea", "rax,[rsp+0x60]"),
    (0x180078588, "mov", "QWORD PTR [rsp+0x88],rax"),
    (0x180078590, "mov", "edx,esi"),
    (0x180078592, "mov", "rax,QWORD PTR [rbx+0x140]"),
    (0x180078599, "mov", "QWORD PTR [rsp+0x80],0x2"),
    (0x1800785a5, "mov", "QWORD PTR [rsp+0x90],rdi"),
    (0x1800785ad, "mov", "QWORD PTR [rsp+0x48],rdi"),
    (0x1800785b2, "mov", "r9d,DWORD PTR [rax+0x50]"),
    (0x1800785b6, "mov", "ecx,DWORD PTR [rax+0x40]"),
    (0x1800785b9, "mov", "r10d,DWORD PTR [rax+0x48]"),
    (0x1800785bd, "mov", "r11d,DWORD PTR [rax+0x44]"),
    (0x1800785c1, "mov", "QWORD PTR [rsp+0x40],r8"),
    (0x1800785c6, "mov", "r8d,DWORD PTR [rsp+0xe0]"),
    (0x1800785ce, "mov", "DWORD PTR [rsp+0x38],r9d"),
    (0x1800785d3, "mov", "r9d,DWORD PTR [rsp+0xe8]"),
    (0x1800785db, "mov", "DWORD PTR [rsp+0x30],r10d"),
    (0x1800785e0, "mov", "DWORD PTR [rsp+0x28],r11d"),
    (0x1800785e5, "mov", "DWORD PTR [rsp+0x20],ecx"),
    (0x1800785e9, "mov", "rcx,QWORD PTR [rax+0x38]"),
    (0x1800785ed, "mov", "rax,QWORD PTR [rbx+0x470]"),
    (0x1800785f4, "call", "QWORD PTR [rip+0x33e2e]"),
)


# Exact sizing methods and bounded allocation-ledger consumers; not tensor addressing.
SIZING_ANCHORS: tuple[tuple[int, str, str], ...] = (
    (0x180038d5e, 'mov', 'rax,QWORD PTR [rcx]'),
    (0x180038d61, 'mov', 'r9d,DWORD PTR [rsp+0x64]'),
    (0x180038d66, 'mov', 'r8d,DWORD PTR [rsp+0x7c]'),
    (0x180038d6b, 'mov', 'rdx,rdi'),
    (0x180038d6e, 'mov', 'rax,QWORD PTR [rax+0x40]'),
    (0x180038d72, 'call', 'QWORD PTR [rip+0x736b0]'),
    (0x180038d78, 'lea', 'rbx,[rax*4+0x1ff]'),
    (0x180038d80, 'and', 'rbx,0xfffffffffffffe00'),
    (0x180038d87, 'mov', 'r9,QWORD PTR [rsp+0x70]'),
    (0x180038d8c, 'mov', 'rax,QWORD PTR [r9+0x210]'),
    (0x180038d93, 'mov', 'r11,QWORD PTR [rbp+0x118]'),
    (0x180038d9a, 'mov', 'rcx,QWORD PTR [r11+rax*1]'),
    (0x180038d9e, 'add', 'rcx,r13'),
    (0x180038da1, 'mov', 'rdx,QWORD PTR [rcx+0x8]'),
    (0x180038da5, 'cmp', 'rdx,QWORD PTR [rcx+0x10]'),
    (0x180038da9, 'je', '0x180038dbc'),
    (0x180038dab, 'mov', 'rax,QWORD PTR [r9+0x208]'),
    (0x180038db2, 'mov', 'QWORD PTR [rdx],rax'),
    (0x180038db5, 'add', 'QWORD PTR [rcx+0x8],0x8'),
    (0x180038dba, 'jmp', '0x180038dcd'),
    (0x180038dbc, 'lea', 'r8,[r9+0x208]'),
    (0x180038dc3, 'call', '0x18001e110'),
    (0x180038dc8, 'mov', 'r9,QWORD PTR [rsp+0x70]'),
    (0x180038dcd, 'add', 'QWORD PTR [r9+0x208],rbx'),
    (0x180038dd4, 'inc', 'rdi'),
    (0x180038dd7, 'mov', 'rbx,QWORD PTR [rbp-0x70]'),
    (0x180038ddb, 'rex', '.WB'),
    (0x1800391b2, 'mov', 'rax,QWORD PTR [rcx]'),
    (0x1800391b5, 'mov', 'r9d,DWORD PTR [rsp+0x64]'),
    (0x1800391ba, 'mov', 'r8d,DWORD PTR [rsp+0x7c]'),
    (0x1800391bf, 'mov', 'rdx,rbx'),
    (0x1800391c2, 'mov', 'rax,QWORD PTR [rax+0x48]'),
    (0x1800391c6, 'call', 'QWORD PTR [rip+0x7325c]'),
    (0x1800391cc, 'mov', 'r14,rax'),
    (0x180039217, 'mov', 'QWORD PTR [rbp+0x128],r14'),
    (0x18003921e, 'mov', 'DWORD PTR [rbp+0x130],edi'),
    (0x180039224, 'mov', 'eax,DWORD PTR [rbp-0x6c]'),
    (0x180039227, 'mov', 'DWORD PTR [rbp+0x134],eax'),
    (0x18003922d, 'mov', 'rdx,QWORD PTR [rbp-0x8]'),
    (0x180039231, 'cmp', 'rdx,QWORD PTR [rbp+0x0]'),
    (0x180039235, 'je', '0x180039248'),
    (0x180039237, 'movups', 'xmm0,XMMWORD PTR [rbp+0x128]'),
    (0x18003923e, 'movups', 'XMMWORD PTR [rdx],xmm0'),
    (0x180039241, 'add', 'QWORD PTR [rbp-0x8],0x10'),
    (0x180039246, 'jmp', '0x180039258'),
    (0x180039248, 'lea', 'r8,[rbp+0x128]'),
    (0x18003924f, 'lea', 'rcx,[rbp-0x10]'),
    (0x180039253, 'call', '0x18002ee80'),
    (0x180074710, 'cmp', 'rdx,0x1'),
    (0x180074714, 'jbe', '0x180074719'),
    (0x180074716, 'xor', 'eax,eax'),
    (0x180074718, 'ret', ''),
    (0x180074719, 'mov', 'eax,r8d'),
    (0x18007471c, 'cdq', ''),
    (0x18007471d, 'sub', 'eax,edx'),
    (0x18007471f, 'sar', 'eax,1'),
    (0x180074721, 'add', 'eax,0x3'),
    (0x180074724, 'cdq', ''),
    (0x180074725, 'and', 'edx,0x3'),
    (0x180074728, 'add', 'eax,edx'),
    (0x18007472a, 'sar', 'eax,0x2'),
    (0x18007472d, 'lea', 'eax,[rax*4+0x3]'),
    (0x180074734, 'cdq', ''),
    (0x180074735, 'and', 'edx,0x3'),
    (0x180074738, 'add', 'eax,edx'),
    (0x18007473a, 'sar', 'eax,0x2'),
    (0x18007473d, 'movsxd', 'r8,eax'),
    (0x180074740, 'mov', 'eax,r9d'),
    (0x180074743, 'cdq', ''),
    (0x180074744, 'sub', 'eax,edx'),
    (0x180074746, 'sar', 'eax,1'),
    (0x180074748, 'add', 'eax,0x3'),
    (0x18007474b, 'cdq', ''),
    (0x18007474c, 'and', 'edx,0x3'),
    (0x18007474f, 'add', 'eax,edx'),
    (0x180074751, 'sar', 'eax,0x2'),
    (0x180074754, 'lea', 'eax,[rax*4+0x3]'),
    (0x18007475b, 'cdq', ''),
    (0x18007475c, 'and', 'edx,0x3'),
    (0x18007475f, 'add', 'eax,edx'),
    (0x180074761, 'sar', 'eax,0x2'),
    (0x180074764, 'movsxd', 'rcx,eax'),
    (0x180074767, 'imul', 'r8,rcx'),
    (0x18007476b, 'lea', 'rax,[r8+r8*1]'),
    (0x18007476f, 'ret', ''),
    (0x1800747c0, 'mov', 'r10,rdx'),
    (0x1800747c3, 'mov', 'eax,r8d'),
    (0x1800747c6, 'cdq', ''),
    (0x1800747c7, 'mov', 'r11,rcx'),
    (0x1800747ca, 'sub', 'eax,edx'),
    (0x1800747cc, 'sar', 'eax,1'),
    (0x1800747ce, 'add', 'eax,0x3'),
    (0x1800747d1, 'cdq', ''),
    (0x1800747d2, 'and', 'edx,0x3'),
    (0x1800747d5, 'lea', 'r8d,[rdx+rax*1]'),
    (0x1800747d9, 'mov', 'eax,r9d'),
    (0x1800747dc, 'cdq', ''),
    (0x1800747dd, 'and', 'r8d,0xfffffffc'),
    (0x1800747e1, 'sub', 'eax,edx'),
    (0x1800747e3, 'sar', 'eax,1'),
    (0x1800747e5, 'add', 'eax,0x3'),
    (0x1800747e8, 'cdq', ''),
    (0x1800747e9, 'and', 'edx,0x3'),
    (0x1800747ec, 'lea', 'ecx,[rdx+rax*1]'),
    (0x1800747ef, 'and', 'ecx,0xfffffffc'),
    (0x1800747f2, 'cmp', 'r10,0x1'),
    (0x1800747f6, 'ja', '0x180074811'),
    (0x1800747f8, 'movsxd', 'rax,DWORD PTR [r11+0x17c]'),
    (0x1800747ff, 'movsxd', 'rcx,ecx'),
    (0x180074802, 'imul', 'rax,rcx'),
    (0x180074806, 'movsxd', 'rcx,r8d'),
    (0x180074809, 'imul', 'rax,rcx'),
    (0x18007480d, 'add', 'rax,rax'),
    (0x180074810, 'ret', ''),
    (0x180074811, 'xor', 'eax,eax'),
    (0x180074813, 'ret', ''),
    (0x18002eba0, 'mov', 'QWORD PTR [rsp+0x8],rcx'),
    (0x18002eba5, 'push', 'rbx'),
    (0x18002eba6, 'push', 'rsi'),
    (0x18002eba7, 'push', 'rdi'),
    (0x18002eba8, 'push', 'r12'),
    (0x18002ebaa, 'push', 'r13'),
    (0x18002ebac, 'push', 'r14'),
    (0x18002ebae, 'push', 'r15'),
    (0x18002ebb0, 'sub', 'rsp,0x20'),
    (0x18002ec7a, 'mov', 'rax,r14'),
    (0x18002ec7d, 'add', 'rsp,0x20'),
    (0x18002ec81, 'pop', 'r15'),
    (0x18002ec83, 'pop', 'r14'),
    (0x18002ec85, 'pop', 'r13'),
    (0x18002ec87, 'pop', 'r12'),
    (0x18002ec89, 'pop', 'rdi'),
    (0x18002ec8a, 'pop', 'rsi'),
    (0x18002ec8b, 'pop', 'rbx'),
    (0x18002ec8c, 'ret', ''),
    (0x1800391cf, 'mov', 'r8,QWORD PTR [rbp-0x8]'),
    (0x1800391d3, 'sub', 'r8,QWORD PTR [rbp-0x10]'),
    (0x1800391d7, 'sar', 'r8,0x4'),
    (0x1800391db, 'mov', 'DWORD PTR [rbp-0x60],r8d'),
    (0x1800391df, 'mov', 'rax,QWORD PTR [rsp+0x70]'),
    (0x1800391e4, 'mov', 'rcx,QWORD PTR [rax+0x238]'),
    (0x1800391eb, 'mov', 'rax,QWORD PTR [rbp+0x28]'),
    (0x1800391ef, 'mov', 'rdx,QWORD PTR [rax+rcx*1]'),
    (0x1800391f3, 'mov', 'rcx,QWORD PTR [rbp-0x58]'),
    (0x1800391f7, 'add', 'rcx,rdx'),
    (0x1800391fa, 'mov', 'rdx,QWORD PTR [rcx+0x8]'),
    (0x1800391fe, 'cmp', 'rdx,QWORD PTR [rcx+0x10]'),
    (0x180039202, 'je', '0x18003920e'),
    (0x180039204, 'mov', 'DWORD PTR [rdx],r8d'),
    (0x180039207, 'add', 'QWORD PTR [rcx+0x8],0x4'),
    (0x18003920c, 'jmp', '0x180039217'),
    (0x18003920e, 'lea', 'r8,[rbp-0x60]'),
    (0x180039212, 'call', '0x18002eba0'),
)


# Feature planner output and arena pointer construction; static provenance only.
ARENA_ANCHORS: tuple[tuple[int, str, str], ...] = (
    (0x18002de1a, 'mov', 'rsi,rdx'),
    (0x18002de1d, 'mov', 'QWORD PTR [rbp-0x31],rdx'),
    (0x18002de21, 'mov', 'QWORD PTR [rbp-0x29],rcx'),
    (0x18002de25, 'mov', 'QWORD PTR [rbp-0x19],rcx'),
    (0x18002de29, 'mov', 'DWORD PTR [rsp+0x28],0x0'),
    (0x18002de31, 'add', 'rcx,0x10'),
    (0x18002de35, 'mov', 'QWORD PTR [rcx],0x0'),
    (0x18002de3c, 'mov', 'QWORD PTR [rcx+0x8],0x0'),
    (0x18002de44, 'mov', 'QWORD PTR [rcx+0x10],0x0'),
    (0x18002de4c, 'mov', 'DWORD PTR [rsp+0x28],0x1'),
    (0x18002de54, 'mov', 'QWORD PTR [rbp-0x11],0x0'),
    (0x18002de5c, 'mov', 'QWORD PTR [rbp-0x9],0x0'),
    (0x18002de64, 'mov', 'rdx,QWORD PTR [rdx+0x8]'),
    (0x18002de68, 'sub', 'rdx,QWORD PTR [rsi]'),
    (0x18002de6b, 'sar', 'rdx,0x4'),
    (0x18002de6f, 'lea', 'r8,[rbp-0x11]'),
    (0x18002de73, 'call', '0x18002dc00'),
    (0x18002dfe8, 'mov', 'r12,QWORD PTR [rsi]'),
    (0x18002dfeb, 'add', 'r12,rcx'),
    (0x18002dfee, 'mov', 'r13,QWORD PTR [r12]'),
    (0x18002dff2, 'add', 'r13,0x1ff'),
    (0x18002dff9, 'and', 'r13,0xfffffffffffffe00'),
    (0x18002e0f0, 'mov', 'rax,QWORD PTR [rbp-0x29]'),
    (0x18002e0f4, 'mov', 'rax,QWORD PTR [rax+0x10]'),
    (0x18002e0f8, 'mov', 'rcx,QWORD PTR [rbp-0x21]'),
    (0x18002e0fc, 'mov', 'QWORD PTR [rax+rcx*1],rdx'),
    (0x18002e100, 'mov', 'QWORD PTR [rax+rcx*1+0x8],r13'),
    (0x18002e105, 'mov', 'QWORD PTR [rbp-0x1],rdx'),
    (0x18002e109, 'lea', 'rdi,[rdx+r13*1]'),
    (0x18002e10d, 'mov', 'QWORD PTR [rbp+0x7],rdi'),
    (0x18002e111, 'mov', 'eax,DWORD PTR [r12+0x8]'),
    (0x18002e116, 'mov', 'DWORD PTR [rbp+0xf],eax'),
    (0x18002e119, 'mov', 'eax,DWORD PTR [r12+0xc]'),
    (0x18002e11e, 'mov', 'DWORD PTR [rbp+0x13],eax'),
    (0x18002e1fd, 'mov', 'r14,QWORD PTR [rbp-0x29]'),
    (0x18002e201, 'mov', 'QWORD PTR [r14],rcx'),
    (0x18002e204, 'xor', 'ebx,ebx'),
    (0x18002e206, 'mov', 'DWORD PTR [rsp+0x20],ebx'),
    (0x18002e330, 'mov', 'rax,r14'),
    (0x18002e333, 'mov', 'rcx,QWORD PTR [rbp+0x17]'),
    (0x18002e337, 'xor', 'rcx,rsp'),
    (0x18002e33a, 'call', '0x18007c7d0'),
    (0x18002e33f, 'mov', 'rbx,QWORD PTR [rsp+0x140]'),
    (0x18002e347, 'add', 'rsp,0xf0'),
    (0x18002e34e, 'pop', 'r15'),
    (0x18002e350, 'pop', 'r14'),
    (0x18002e352, 'pop', 'r13'),
    (0x18002e354, 'pop', 'r12'),
    (0x18002e356, 'pop', 'rdi'),
    (0x18002e357, 'pop', 'rsi'),
    (0x18002e358, 'pop', 'rbp'),
    (0x180039341, 'lea', 'rdx,[rbp-0x10]'),
    (0x180039345, 'lea', 'rcx,[rbp+0xb8]'),
    (0x18003934c, 'call', '0x18002ddf0'),
    (0x180039351, 'mov', 'rcx,QWORD PTR [rax]'),
    (0x180039354, 'mov', 'rdx,QWORD PTR [rsp+0x70]'),
    (0x180039359, 'mov', 'QWORD PTR [rdx+0x168],rcx'),
    (0x180039360, 'mov', 'rcx,QWORD PTR [rax+0x8]'),
    (0x180039364, 'mov', 'QWORD PTR [rdx+0x170],rcx'),
    (0x18003936b, 'lea', 'rdi,[rax+0x10]'),
    (0x18003936f, 'lea', 'rbx,[rdx+0x178]'),
    (0x180039376, 'cmp', 'rbx,rdi'),
    (0x180039379, 'je', '0x1800393a4'),
    (0x18003937b, 'mov', 'rcx,rbx'),
    (0x18003937e, 'call', '0x18002d810'),
    (0x180039383, 'mov', 'rax,QWORD PTR [rdi]'),
    (0x180039386, 'mov', 'QWORD PTR [rbx],rax'),
    (0x180039389, 'mov', 'rax,QWORD PTR [rdi+0x8]'),
    (0x18003938d, 'mov', 'QWORD PTR [rbx+0x8],rax'),
    (0x180039391, 'mov', 'rax,QWORD PTR [rdi+0x10]'),
    (0x180039395, 'mov', 'QWORD PTR [rbx+0x10],rax'),
    (0x180039399, 'mov', 'QWORD PTR [rdi],r12'),
    (0x18003939c, 'mov', 'QWORD PTR [rdi+0x8],r12'),
    (0x1800393a0, 'mov', 'QWORD PTR [rdi+0x10],r12'),
    (0x1800393a4, 'lea', 'rcx,[rbp+0xb8]'),
    (0x1800393ab, 'call', '0x18002d870'),
    (0x18003d790, 'mov', 'rax,QWORD PTR [rcx+0x208]'),
    (0x18003d797, 'add', 'rax,QWORD PTR [rcx+0x168]'),
    (0x18003d79e, 'ret', ''),
    (0x18003e4e0, 'mov', 'QWORD PTR [rsp+0x10],rbx'),
    (0x18003e4e5, 'mov', 'QWORD PTR [rsp+0x18],rsi'),
    (0x18003e4ea, 'mov', 'QWORD PTR [rsp+0x20],rdi'),
    (0x18003e4ef, 'push', 'rbp'),
    (0x18003e4f0, 'push', 'r12'),
    (0x18003e4f2, 'push', 'r13'),
    (0x18003e4f4, 'push', 'r14'),
    (0x18003e4f6, 'push', 'r15'),
    (0x18003e4f8, 'lea', 'rbp,[rsp-0x110]'),
    (0x18003e500, 'sub', 'rsp,0x210'),
    (0x18003e507, 'mov', 'rax,QWORD PTR [rip+0x1102032]'),
    (0x18003e50e, 'xor', 'rax,rsp'),
    (0x18003e511, 'mov', 'QWORD PTR [rbp+0x100],rax'),
    (0x18003e518, 'mov', 'rbx,r8'),
    (0x18003e51b, 'mov', 'rdi,rdx'),
    (0x18003e51e, 'mov', 'QWORD PTR [rsp+0x78],rdx'),
    (0x18003e523, 'mov', 'rsi,rcx'),
    (0x18003e526, 'test', 'rdx,rdx'),
    (0x18003e529, 'je', '0x18003f2a7'),
    (0x18003e52f, 'test', 'rbx,rbx'),
    (0x18003e532, 'je', '0x18003f2a7'),
    (0x18003e538, 'mov', 'rax,QWORD PTR [rcx]'),
    (0x18003e53b, 'mov', 'rax,QWORD PTR [rax+0x60]'),
    (0x18003e53f, 'call', 'QWORD PTR [rip+0x6dee3]'),
    (0x18003e545, 'cmp', 'rax,rbx'),
    (0x18003e548, 'ja', '0x18003f2ca'),
    (0x18003e54e, 'mov', 'QWORD PTR [rsi+0x250],rdi'),
    (0x18003e555, 'mov', 'QWORD PTR [rsi+0x258],rbx'),
    (0x18003e55c, 'mov', 'rcx,QWORD PTR [rsi+0x208]'),
    (0x18003e563, 'lea', 'rax,[rcx+rdi*1]'),
    (0x18003e567, 'mov', 'QWORD PTR [rsp+0x38],rax'),
    (0x18003e56c, 'xor', 'eax,eax'),
    (0x18003e56e, 'test', 'rcx,rcx'),
    (0x18003e571, 'cmovne', 'rax,rdi'),
    (0x18003e575, 'mov', 'QWORD PTR [rsi+0x228],rax'),
    (0x18003e8a1, 'xor', 'r14d,r14d'),
    (0x18003e8a4, 'mov', 'QWORD PTR [rsp+0x68],r14'),
    (0x18003e8a9, 'xor', 'r12d,r12d'),
    (0x18003e8ac, 'mov', 'QWORD PTR [rsp+0x60],r12'),
    (0x18003e8b1, 'xor', 'r13d,r13d'),
    (0x18003e8b4, 'mov', 'QWORD PTR [rsp+0x70],r13'),
    (0x18003e8b9, 'movabs', 'rcx,0xaaaaaaaaaaaaaaa'),
    (0x18003e8c0, 'aa', '0a'),
    (0x18003e8c3, 'lea', 'r11,[r8+r13*1]'),
    (0x18003e8c7, 'mov', 'QWORD PTR [rsp+0x28],r11'),
    (0x18003e8cc, 'xor', 'r9b,r9b'),
    (0x18003ea3f, 'mov', 'rdi,r8'),
    (0x18003ea42, 'mov', 'r13,r8'),
    (0x18003ea45, 'mov', 'r14,r8'),
    (0x18003ea48, 'mov', 'QWORD PTR [rsp+0x30],r8'),
    (0x18003ea4d, 'nop', 'DWORD PTR [rax]'),
    (0x18003ead1, 'add', 'rbx,0x4'),
    (0x18003ead5, 'cmp', 'rbx,r14'),
    (0x18003ead8, 'jne', '0x18003ea84'),
    (0x18003eada, 'mov', 'r12,QWORD PTR [rsp+0x60]'),
    (0x18003eadf, 'mov', 'r14,QWORD PTR [rsp+0x30]'),
    (0x18003eb04, 'mov', 'r14,QWORD PTR [rsp+0x78]'),
    (0x18003eb09, 'nop', 'DWORD PTR [rax+0x0]'),
    (0x18003eb10, 'mov', 'rax,QWORD PTR [rsi+0x210]'),
    (0x18003eb17, 'mov', 'rcx,QWORD PTR [r12+rax*1]'),
    (0x18003eb1b, 'mov', 'rax,QWORD PTR [rdi+rcx*1]'),
    (0x18003eb1f, 'mov', 'rax,QWORD PTR [rax+rbx*8]'),
    (0x18003eb23, 'mov', 'r9,QWORD PTR [rbp-0x80]'),
    (0x18003eb27, 'add', 'r9,rdi'),
    (0x18003eb2a, 'add', 'rax,r14'),
    (0x18003eb2d, 'mov', 'QWORD PTR [rsp+0x20],rax'),
    (0x18003eb32, 'mov', 'rdx,QWORD PTR [r9+0x8]'),
    (0x18003eb36, 'cmp', 'rdx,QWORD PTR [r9+0x10]'),
    (0x18003eb3a, 'je', '0x18003ebc8'),
    (0x18003eb40, 'mov', 'QWORD PTR [rdx],rax'),
    (0x18003eb43, 'add', 'QWORD PTR [r9+0x8],0x8'),
    (0x18003eb48, 'jmp', '0x18003ebd5'),
    (0x18003ebd5, 'inc', 'rbx'),
    (0x18003ebd8, 'mov', 'rax,QWORD PTR [r15+r13*1+0xf8]'),
    (0x18003ebe0, 'sub', 'rax,QWORD PTR [r15+r13*1+0xf0]'),
    (0x18003ebe8, 'sar', 'rax,0x3'),
    (0x18003ebec, 'cmp', 'rbx,rax'),
    (0x18003ebef, 'jb', '0x18003eb10'),
    (0x18003ebf5, 'mov', 'r14,QWORD PTR [rsp+0x30]'),
    (0x18003ebfa, 'xor', 'r8d,r8d'),
    (0x18003ebfd, 'mov', 'rbx,r8'),
    (0x18003ec00, 'mov', 'rax,QWORD PTR [r15+r13*1+0x160]'),
    (0x18003ec08, 'sub', 'rax,QWORD PTR [r15+r13*1+0x158]'),
    (0x18003ec10, 'sar', 'rax,0x3'),
    (0x18003ec14, 'test', 'rax,rax'),
    (0x18003ec17, 'je', '0x18003ec8f'),
    (0x18003ec19, 'nop', 'DWORD PTR [rax+0x0]'),
    (0x18003ec20, 'mov', 'rax,QWORD PTR [rsi+0x238]'),
    (0x18003ec27, 'mov', 'rcx,QWORD PTR [r12+rax*1]'),
    (0x18003ec2b, 'mov', 'rax,QWORD PTR [rdi+rcx*1]'),
    (0x18003ec2f, 'movsxd', 'rax,DWORD PTR [rax+rbx*4]'),
    (0x18003ec33, 'mov', 'r9,QWORD PTR [rbp-0x80]'),
    (0x18003ec37, 'add', 'r9,rdi'),
    (0x18003ec3a, 'shl', 'rax,0x4'),
    (0x18003ec3e, 'add', 'rax,QWORD PTR [rsi+0x178]'),
    (0x18003ec45, 'mov', 'rcx,QWORD PTR [rsp+0x38]'),
    (0x18003ec4a, 'add', 'rcx,QWORD PTR [rax]'),
    (0x18003ec4d, 'mov', 'QWORD PTR [rsp+0x20],rcx'),
    (0x18003ec52, 'mov', 'rdx,QWORD PTR [r9+0x8]'),
    (0x18003ec56, 'cmp', 'rdx,QWORD PTR [r9+0x10]'),
    (0x18003ec5a, 'je', '0x18003ec66'),
    (0x18003ec5c, 'mov', 'QWORD PTR [rdx],rcx'),
    (0x18003ec5f, 'add', 'QWORD PTR [r9+0x8],0x8'),
    (0x18003ec64, 'jmp', '0x18003ec73'),
    (0x18003ec66, 'lea', 'r8,[rsp+0x20]'),
    (0x18003ec6b, 'mov', 'rcx,r9'),
    (0x18003ec6e, 'call', '0x18001e110'),
    (0x18003ec73, 'inc', 'rbx'),
    (0x18003ec76, 'mov', 'rax,QWORD PTR [r15+r13*1+0x160]'),
    (0x18003ec7e, 'sub', 'rax,QWORD PTR [r15+r13*1+0x158]'),
    (0x18003ec86, 'sar', 'rax,0x3'),
    (0x18003ec8a, 'cmp', 'rbx,rax'),
    (0x18003ec8d, 'jb', '0x18003ec20'),
    (0x18003eced, 'mov', 'r14,QWORD PTR [rsp+0x68]'),
    (0x18003ecf2, 'mov', 'r13,QWORD PTR [rsp+0x70]'),
    (0x18003ecf7, 'mov', 'rax,QWORD PTR [rsi+0xf8]'),
    (0x18003ecfe, 'lea', 'rdx,[rbp-0x80]'),
    (0x18003ed02, 'mov', 'rcx,QWORD PTR [r14+rax*1]'),
    (0x18003ed06, 'call', '0x18002c1e0'),
    (0x18003eddd, 'movabs', 'r15,0x2aaaaaaaaaaaaaab'),
    (0x18003ede4, 'aa', '2a'),
    (0x18003ede7, 'mov', 'ebx,DWORD PTR [rsp+0x5c]'),
    (0x18003edeb, 'inc', 'ebx'),
    (0x18003eded, 'mov', 'DWORD PTR [rsp+0x5c],ebx'),
    (0x18003edf1, 'add', 'r13,0xe0'),
    (0x18003edf8, 'mov', 'QWORD PTR [rsp+0x70],r13'),
    (0x18003edfd, 'add', 'r12,0x18'),
    (0x18002c1e0, 'add', 'rcx,0x140'),
    (0x18002c1e7, 'cmp', 'rcx,rdx'),
    (0x18002c1ea, 'je', '0x18002c1fe'),
    (0x18002c1ec, 'mov', 'r8,QWORD PTR [rdx+0x8]'),
    (0x18002c1f0, 'mov', 'rdx,QWORD PTR [rdx]'),
    (0x18002c1f3, 'movzx', 'r9d,BYTE PTR [rsp+0x8]'),
    (0x18002c1f9, 'jmp', '0x1800296d0'),
    (0x18002c1fe, 'ret', ''),
    (0x18002dfdd, 'mov', 'rcx,QWORD PTR [rax]'),
    (0x18002dfe0, 'shl', 'rcx,0x4'),
    (0x18002dfe4, 'mov', 'QWORD PTR [rbp-0x21],rcx'),
    (0x18002e0c4, 'xor', 'edx,edx'),
    (0x18002e0c6, 'mov', 'rcx,rbx'),
    (0x18002e0c9, 'cmp', 'rbx,rdi'),
    (0x18002e0cc, 'je', '0x18002e0f0'),
    (0x18002e0ce, 'xchg', 'ax,ax'),
    (0x18002e0d0, 'lea', 'rax,[rdx+r13*1]'),
    (0x18002e0d4, 'cmp', 'rax,QWORD PTR [rcx]'),
    (0x18002e0d7, 'jbe', '0x18002e0f0'),
    (0x18002e0d9, 'mov', 'rax,QWORD PTR [rcx+0x8]'),
    (0x18002e0dd, 'add', 'rcx,0x10'),
    (0x18002e0e1, 'cmp', 'rax,rdx'),
    (0x18002e0e4, 'cmovbe', 'rax,rdx'),
    (0x18002e0e8, 'mov', 'rdx,rax'),
    (0x18002e0eb, 'cmp', 'rcx,rdi'),
    (0x18002e0ee, 'jne', '0x18002e0d0'),
    (0x18003e9a8, 'mov', 'r11,QWORD PTR [rsp+0x28]'),
    (0x18003e9ad, 'xor', 'r8d,r8d'),
    (0x18003e9b0, 'jmp', '0x18003e9da'),
    (0x18003e9b2, 'test', 'rdi,rdi'),
    (0x18003e9b5, 'je', '0x18003e9d0'),
    (0x18003e9b7, 'mov', 'rcx,rdi'),
    (0x18003e9ba, 'call', '0x18007c058'),
    (0x18003e9bf, 'lea', 'rcx,[rax+rdi*1]'),
    (0x18003e9c3, 'mov', 'rdx,rax'),
    (0x18003e9c6, 'mov', 'r11,QWORD PTR [rsp+0x28]'),
    (0x18003e9cb, 'xor', 'r8d,r8d'),
    (0x18003e9ce, 'jmp', '0x18003e9da'),
    (0x18003e9d0, 'mov', 'rax,r8'),
    (0x18003e9d3, 'lea', 'rcx,[r8+rdi*1]'),
    (0x18003e9d7, 'mov', 'rdx,r8'),
    (0x18003e9da, 'mov', 'QWORD PTR [rbp-0x80],rdx'),
    (0x18003e9de, 'mov', 'QWORD PTR [rbp-0x70],rcx'),
    (0x18003e9e2, 'nop', 'DWORD PTR [rax+0x0]'),
    (0x18003e9e6, 'data16', 'nop WORD PTR [rax+rax*1+0x0]'),
    (0x18003e9f0, 'mov', 'QWORD PTR [rax],r8'),
    (0x18003e9f3, 'mov', 'QWORD PTR [rax+0x8],r8'),
    (0x18003e9f7, 'mov', 'QWORD PTR [rax+0x10],r8'),
    (0x18003e9fb, 'add', 'rax,0x18'),
    (0x18003e9ff, 'sub', 'rbx,0x1'),
    (0x18003ea03, 'jne', '0x18003e9f0'),
    (0x18003ea05, 'mov', 'QWORD PTR [rbp-0x78],rax'),
    (0x18003ea09, 'mov', 'DWORD PTR [rsp+0x58],r8d'),
    (0x18003ea0e, 'mov', 'r15,QWORD PTR [r11+0x20]'),
    (0x18003ea12, 'mov', 'rcx,QWORD PTR [r11+0x28]'),
    (0x18003ea16, 'sub', 'rcx,r15'),
    (0x18003ea19, 'movabs', 'rax,0xb21642c8590b2165'),
    (0x18003ea20, 'b2', ''),
    (0x18003ea23, 'imul', 'rcx'),
    (0x18003ea26, 'add', 'rdx,rcx'),
    (0x18003ea29, 'sar', 'rdx,0x8'),
    (0x18003ea2d, 'mov', 'rax,rdx'),
    (0x18003ea30, 'shr', 'rax,0x3f'),
    (0x18003ea34, 'add', 'rdx,rax'),
    (0x18003ea37, 'test', 'edx,edx'),
    (0x18003ea39, 'jle', '0x18003ecf7'),
    (0x18003ea50, 'mov', 'rax,QWORD PTR [rsi+0x190]'),
    (0x18003ea57, 'mov', 'rcx,QWORD PTR [r12+rax*1]'),
    (0x18003ea5b, 'cmp', 'DWORD PTR [r14+rcx*1],0x0'),
    (0x18003ea60, 'jl', '0x18003eb4d'),
    (0x18003ea66, 'mov', 'rax,QWORD PTR [rsi+0x1a8]'),
    (0x18003ea6d, 'mov', 'rcx,QWORD PTR [r12+rax*1]'),
    (0x18003ea71, 'mov', 'r14,QWORD PTR [rdi+rcx*1+0x8]'),
    (0x18003ea76, 'mov', 'rbx,QWORD PTR [rdi+rcx*1]'),
    (0x18003ea7a, 'cmp', 'rbx,r14'),
    (0x18003ea7d, 'je', '0x18003eadf'),
    (0x18003ea7f, 'mov', 'r12,QWORD PTR [rsp+0x38]'),
    (0x18003ea84, 'movsxd', 'rax,DWORD PTR [rbx]'),
    (0x18003ea87, 'test', 'eax,eax'),
    (0x18003ea89, 'js', '0x18003ea9e'),
    (0x18003ea8b, 'shl', 'rax,0x4'),
    (0x18003ea8f, 'add', 'rax,QWORD PTR [rsi+0x178]'),
    (0x18003ea96, 'mov', 'rdx,QWORD PTR [rax]'),
    (0x18003ea99, 'add', 'rdx,r12'),
    (0x18003ea9c, 'jmp', '0x18003eaa1'),
    (0x18003ea9e, 'mov', 'rdx,r8'),
    (0x18003eaa1, 'mov', 'QWORD PTR [rsp+0x20],rdx'),
    (0x18003eaa6, 'mov', 'rcx,QWORD PTR [rbp-0x80]'),
    (0x18003eaaa, 'add', 'rcx,rdi'),
    (0x18003eaad, 'mov', 'rax,QWORD PTR [rcx+0x8]'),
    (0x18003eab1, 'cmp', 'rax,QWORD PTR [rcx+0x10]'),
    (0x18003eab5, 'je', '0x18003eac1'),
    (0x18003eab7, 'mov', 'QWORD PTR [rax],rdx'),
    (0x18003eaba, 'add', 'QWORD PTR [rcx+0x8],0x8'),
    (0x18003eabf, 'jmp', '0x18003ead1'),
    (0x18003eac1, 'lea', 'r8,[rsp+0x20]'),
    (0x18003eac6, 'mov', 'rdx,rax'),
    (0x18003eac9, 'call', '0x180027370'),
    (0x18003eace, 'xor', 'r8d,r8d'),
)

COUNTER_RESET_RANGES = ((6442672720, 6442672895), (6442699904, 6442700170))
RANGES += COUNTER_RESET_RANGES
COUNTER_RESET_ANCHORS: tuple[tuple[int, str, str], ...] = (
    (0x180036250, 'rex', 'push rdi'),
    (0x180036252, 'sub', 'rsp,0x40'),
    (0x180036256, 'cmp', 'QWORD PTR [rcx+0x270],0x0'),
    (0x18003625e, 'mov', 'rdi,rcx'),
    (0x180036261, 'je', '0x1800362f9'),
    (0x180036267, 'mov', 'QWORD PTR [rsp+0x50],rbx'),
    (0x18003626c, 'mov', 'rbx,QWORD PTR [rcx+0xf8]'),
    (0x180036273, 'mov', 'QWORD PTR [rsp+0x58],rsi'),
    (0x180036278, 'mov', 'rsi,QWORD PTR [rcx+0x100]'),
    (0x18003627f, 'cmp', 'rbx,rsi'),
    (0x180036282, 'je', '0x1800362a8'),
    (0x180036290, 'mov', 'rdx,QWORD PTR [rdi+0x270]'),
    (0x180036297, 'mov', 'rcx,QWORD PTR [rbx]'),
    (0x18003629a, 'call', '0x18002c000'),
    (0x18003629f, 'add', 'rbx,0x8'),
    (0x1800362a3, 'cmp', 'rbx,rsi'),
    (0x1800362a6, 'jne', '0x180036290'),
    (0x1800362a8, 'mov', 'rsi,QWORD PTR [rsp+0x58]'),
    (0x1800362ad, 'xor', 'edx,edx'),
    (0x1800362af, 'mov', 'rbx,QWORD PTR [rsp+0x50]'),
    (0x1800362b4, 'cmp', 'QWORD PTR [rdi+0x208],rdx'),
    (0x1800362bb, 'jbe', '0x1800362f2'),
    (0x1800362bd, 'mov', 'rcx,QWORD PTR [rdi+0x270]'),
    (0x1800362c4, 'lea', 'r9d,[rdx+0x1]'),
    (0x1800362c8, 'mov', 'QWORD PTR [rsp+0x28],rdx'),
    (0x1800362cd, 'mov', 'r8d,0x100'),
    (0x1800362d3, 'lea', 'rdx,[rip+0x7b816]'),
    (0x1800362da, 'mov', 'DWORD PTR [rsp+0x20],0x1'),
    (0x1800362e2, 'mov', 'rax,QWORD PTR [rcx]'),
    (0x1800362e5, 'mov', 'rax,QWORD PTR [rax+0x20]'),
    (0x1800362e9, 'call', 'QWORD PTR [rip+0x76139]'),
    (0x1800362ef, 'mov', 'rdx,rax'),
    (0x1800362f2, 'mov', 'QWORD PTR [rdi+0x230],rdx'),
    (0x1800362f9, 'add', 'rsp,0x40'),
    (0x1800362fd, 'pop', 'rdi'),
    (0x1800362fe, 'ret', ''),
    (0x18003cc80, 'mov', 'QWORD PTR [rsp+0x18],rbx'),
    (0x18003cc85, 'push', 'rbp'),
    (0x18003cc86, 'push', 'rsi'),
    (0x18003cc87, 'push', 'rdi'),
    (0x18003cc88, 'push', 'r12'),
    (0x18003cc8a, 'push', 'r13'),
    (0x18003cc8c, 'push', 'r14'),
    (0x18003cc8e, 'push', 'r15'),
    (0x18003cc90, 'lea', 'rbp,[rsp-0x1f]'),
    (0x18003cc95, 'sub', 'rsp,0x100'),
    (0x18003cc9c, 'mov', 'rax,QWORD PTR [rip+0x110389d]'),
    (0x18003cca3, 'xor', 'rax,rsp'),
    (0x18003cca6, 'mov', 'QWORD PTR [rbp+0xf],rax'),
    (0x18003ccaa, 'mov', 'rsi,r9'),
    (0x18003ccad, 'mov', 'QWORD PTR [rbp-0x71],r9'),
    (0x18003ccb1, 'mov', 'rbx,r8'),
    (0x18003ccb4, 'mov', 'QWORD PTR [rbp-0x39],rbx'),
    (0x18003ccb8, 'mov', 'r13,rdx'),
    (0x18003ccbb, 'mov', 'r15,rcx'),
    (0x18003ccbe, 'mov', 'rax,QWORD PTR [rcx+0x100]'),
    (0x18003ccc5, 'mov', 'rcx,QWORD PTR [rcx+0xf8]'),
    (0x18003cccc, 'cmp', 'rcx,rax'),
    (0x18003cccf, 'je', '0x18003d414'),
    (0x18003ccd5, 'cmp', 'rdx,rbx'),
    (0x18003ccd8, 'ja', '0x18003d3f3'),
    (0x18003ccde, 'sub', 'rax,rcx'),
    (0x18003cce1, 'sar', 'rax,0x3'),
    (0x18003cce5, 'cmp', 'rbx,rax'),
    (0x18003cce8, 'ja', '0x18003d3f3'),
    (0x18003ccee, 'cmp', 'rdx,rbx'),
    (0x18003ccf1, 'je', '0x18003d3cc'),
    (0x18003ccf7, 'test', 'rdx,rdx'),
    (0x18003ccfa, 'jne', '0x18003cd81'),
    (0x18003cd00, 'mov', 'rax,QWORD PTR [r15+0x208]'),
    (0x18003cd07, 'test', 'rax,rax'),
    (0x18003cd0a, 'je', '0x18003cd81'),
    (0x18003cd0c, 'mov', 'rdx,QWORD PTR [r15+0x230]'),
    (0x18003cd13, 'test', 'rdx,rdx'),
    (0x18003cd16, 'je', '0x18003cd81'),
    (0x18003cd18, 'mov', 'r8,QWORD PTR [r15+0x228]'),
    (0x18003cd1f, 'test', 'r8,r8'),
    (0x18003cd22, 'je', '0x18003cd81'),
    (0x18003cd24, 'mov', 'rcx,QWORD PTR [r15+0x270]'),
    (0x18003cd2b, 'test', 'rcx,rcx'),
    (0x18003cd2e, 'je', '0x18003cd81'),
    (0x18003cd30, 'mov', 'QWORD PTR [rsp+0x50],r8'),
    (0x18003cd35, 'shr', 'rax,0x2'),
    (0x18003cd39, 'mov', 'DWORD PTR [rsp+0x58],eax'),
    (0x18003cd3d, 'lea', 'r8,[rsp+0x50]'),
    (0x18003cd42, 'mov', 'QWORD PTR [rbp-0x79],r8'),
    (0x18003cd46, 'lea', 'r8d,[rax+0xff]'),
    (0x18003cd4d, 'shr', 'r8d,0x8'),
    (0x18003cd51, 'mov', 'rax,QWORD PTR [rcx]'),
    (0x18003cd54, 'mov', 'BYTE PTR [rsp+0x38],r13b'),
    (0x18003cd59, 'mov', 'QWORD PTR [rsp+0x30],0x10'),
    (0x18003cd62, 'lea', 'r9,[rbp-0x79]'),
    (0x18003cd66, 'mov', 'QWORD PTR [rsp+0x28],r9'),
    (0x18003cd6b, 'mov', 'DWORD PTR [rsp+0x20],0x1'),
    (0x18003cd73, 'lea', 'r9d,[r13+0x1]'),
    (0x18003cd77, 'mov', 'rax,QWORD PTR [rax+0x28]'),
    (0x18003cd7b, 'call', 'QWORD PTR [rip+0x6f6a7]'),
    (0x18003cd81, 'cmp', 'r13,rbx'),
    (0x18003cd84, 'jae', '0x18003d3cc'),
)


def counter_reset_contract(code: dict[int, tuple[str, str]]) -> dict[str, Any]:
    for address, mnemonic, operands in COUNTER_RESET_ANCHORS:
        graph.expect(code, address, mnemonic, operands)
    # Binder provenance for +0x228; constructor/query provenance stays conditional.
    for address in (0x18003e51b, 0x18003e523, 0x18003e55c, 0x18003e563,
                    0x18003e56c, 0x18003e56e, 0x18003e571, 0x18003e575):
        anchor = next(item for item in ARENA_ANCHORS if item[0] == address)
        graph.expect(code, *anchor)
    return dict(
        conditional_static_evidence=True,
        initialiser="0x180036250: virtual +0x20 requests cc_cb_clear with dimensions (256,1,1)",
        kernel_handle_state_field="+0x230; returned by wrapper call",
        counter_pointer_state_field="+0x228 = supplied arena base when prefix != 0, otherwise null",
        range_dispatch="0x18003cc80; only nonempty range starting at zero attempts reset",
        reset_conditions=["nonempty group vector", "valid nonempty range", "start group = 0",
                          "prefix, kernel handle, counter pointer and wrapper nonzero"],
        counter_word_count="uint32(state[+0x208] >> 2)",
        reset_grid="(ceil(count/256),1,1) within nonoverflowing count domain",
        requested_reset_block=[256, 1, 1],
        returned_descriptor_block_dimensions_verified=False, argument_bytes=16,
        argument_fields={"0": "counter pointer u64", "8": "counter count low u32"},
        counter_reset_value=-1,
        reset_value_provenance="existing native utility execution; not established by host anchors",
        reset_submission_precedes_inspected_range_loop_comparison=True,
        continuation_to_group_dispatch_verified=False,
        captured_wrapper_binding=False,
        decoder_counter_consumption_and_tilesync_protocol_verified=False,
        decoder_kernels_launched=0)


def arena_pointer(base: int, counter_bytes: int, feature_offset: int,
                  reserved_bytes: int, supplied_bytes: int) -> int:
    """Our guarded arithmetic helper, not an emulator of the Windows binder.

    Windows does not perform these per-entry bounds checks on the inspected path.
    Units are arena address offsets; element types and tensor layout are unknown.
    """
    values = (base, counter_bytes, feature_offset, reserved_bytes, supplied_bytes)
    require(all(type(v) is int and 0 <= v <= 0xffffffffffffffff for v in values),
            "invalid arena value")
    require(base > 0 and reserved_bytes > 0, "empty arena binding")
    end = counter_bytes + feature_offset + reserved_bytes
    require(end <= supplied_bytes and base + end <= 0xffffffffffffffff,
            "arena binding out of range")
    return base + counter_bytes + feature_offset


def arena_self_test() -> dict[str, int]:
    # Literal non-zero prefix/offset answers, deliberately non-GPU host addresses.
    known = ((0x100000, 512, 0, 1024, 4096, 0x100200),
             (0x100000, 512, 1536, 512, 4096, 0x100800),
             (0x200000, 1024, 512, 1024, 2560, 0x200600))
    for base, prefix, offset, size, supplied, expected in known:
        require(arena_pointer(base, prefix, offset, size, supplied) == expected,
                "arena pointer known answer")
    invalid = ((0, 512, 0, 512, 1024), (True, 512, 0, 512, 1024),
               (0x100000, -1, 0, 512, 1024), (0x100000, 512, 0, 0, 1024),
               (0x100000, 512, 512, 1024, 1536),
               (0xfffffffffffffc00, 512, 0, 512, 1024))
    rejected = 0
    for values in invalid:
        try:
            arena_pointer(*values)
        except ValueError:
            rejected += 1
    require(rejected == len(invalid), "invalid arena pointer accepted")
    # Literal 16-byte planner entries exercise index stride and two qword fields.
    entries = bytes.fromhex(
        "00 06 00 00 00 00 00 00 00 02 00 00 00 00 00 00 "
        "00 00 00 00 00 00 00 00 00 04 00 00 00 00 00 00 "
        "00 04 00 00 00 00 00 00 00 02 00 00 00 00 00 00")
    pointers = []
    for index in (0, 1, 2):
        offset, reserved = struct.unpack_from("<QQ", entries, 16 * index)
        pointers.append(arena_pointer(0x100000, 512, offset, reserved, 4096))
    require(pointers == [0x100800, 0x100200, 0x100600], "arena table stride")
    # Wrong prefix, wrong field and wrong 8-byte stride must disagree.
    require(0x100000 + 1536 != pointers[0], "missing prefix control")
    require(0x100000 + 512 + 512 != pointers[0], "size-as-offset control")
    require(0x100000 + 512 + struct.unpack_from("<Q", entries, 8)[0] != pointers[1],
            "wrong table stride control")
    return dict(arena_known_answers=len(known), arena_invalid_values_rejected=rejected,
                arena_table_entries_checked=3, arena_wrong_formula_controls=3)


def arena_contract(code: dict[int, tuple[str, str]]) -> dict[str, Any]:
    for address, mnemonic, operands in ARENA_ANCHORS:
        graph.expect(code, address, mnemonic, operands)
    return dict(
        decoder_gpu_execution=False, dynamic_arena_binding_verified=False,
        concrete_activation_layouts_verified=False,
        bounded_instruction_anchors=len(ARENA_ANCHORS),
        feature_planner_call="0x18003934c -> 0x18002ddf0, input temporary 16-byte records",
        planner_record_input_fields=["qword requested quantity", "dword interval start candidate",
                                     "dword interval end candidate"],
        planner_record_reserved_quantity="align512(record.qword0)",
        planner_output_entry_stride=16,
        planner_output_entry_fields=["qword arena offset", "qword reserved quantity"],
        placement_offset_rule="starts at zero, advances over selected occupied intervals to their maximum end",
        output_slot_rule="selected index qword << 4; stores offset then reservation",
        selected_index_origin_and_sort_order_verified=False,
        planner_output_destination="state+0x178 vector storage; records remain host metadata",
        planner_total_destination="state+0x168",
        planner_secondary_metric_destination="state+0x170 (not the required-arena getter)",
        planner_full_placement_algorithm_verified=False,
        arena_size_getter="0x18003d790: state[+0x208] + state[+0x168]",
        binder="0x18003e4e0; supplied base in rdx, available quantity in r8",
        binder_capacity_check="vtable+0x60 result <= supplied quantity (target binding unverified)",
        supplied_arena_state_fields=["base -> +0x250", "quantity -> +0x258"],
        counter_prefix_quantity="state+0x208",
        feature_arena_base="supplied_base + state[+0x208], binder stack+0x38",
        counter_pointer="supplied_base + recorded counter offset, 8-byte counter-offset stride",
        feature_pointer="feature_arena_base + planner_entries[feature_index].qword0",
        feature_index_lookup="consumer state+0x238 table -> 4-byte record index (publication unverified)",
        counter_and_feature_bookkeeping_publications_verified=False,
        group_lookup_table_publication_verified=False,
        end_to_end_record_group_layer_association_verified=False,
        launch_vector_order="on inspected nonnegative-input branch: feature inputs, counters, feature work",
        feature_work_pointer_storage="temporary per-layer 24-byte pointer vectors",
        published_pointer_vectors="0x18003ed06 -> 0x18002c1e0: move into group+0x140",
        guarded_helper_checks_address_arithmetic_not_live_indices=True,
        actual_feature_index_domain_verified=False,
        qualification="conditional static path; not an actual GPU allocation, decoder run, or tensor layout")


def sizing_values(index: int, axis0: int, axis1: int) -> dict[str, int]:
    """Positive overflow-free query subset; axes are NOT proven launch axes."""
    maximum = 0x7fffffff - 6
    require(type(index) is int and 0 <= index <= 0xffffffffffffffff, "invalid sizing index")
    require(type(axis0) is int and type(axis1) is int and
            0 < axis0 <= maximum and 0 < axis1 <= maximum, "invalid sizing dimension domain")
    if index > 1:
        return dict(counter_words=0, counter_reserved_bytes=0, feature_ledger_value=0)
    half0 = ((axis0 // 2 + 3) // 4) * 4
    half1 = ((axis1 // 2 + 3) // 4) * 4
    words = 2 * (half0 // 4) * (half1 // 4)
    require(2 * 512 * half0 * half1 <= 0x7fffffffffffffff, "feature sizing product overflow")
    return dict(counter_words=words, counter_reserved_bytes=((words * 4 + 511) // 512) * 512,
                feature_ledger_value=2 * 512 * half0 * half1)


def emulate_sizing(code: dict[int, tuple[str, str]], start: int,
                   index: int, axis0: int, axis1: int) -> int:
    """Bounded integer interpreter for two leaves; never executes Windows code.

    Machine-width arithmetic independently checks the formula. Only the exact
    operand forms in the leaves are supported; all other forms fail closed.
    """
    registers = {"rax": 0, "rdx": index, "r8": axis0, "r9": axis1,
                 "rcx": 0, "r10": 0, "r11": 0}
    aliases = {"eax": "rax", "edx": "rdx", "r8d": "r8", "r9d": "r9", "ecx": "rcx"}
    def read(operand: str) -> int:
        if operand == "DWORD PTR [r11+0x17c]":
            return 512  # Constructor and concrete vtable independently checked.
        if operand in aliases:
            return registers[aliases[operand]] & 0xffffffff
        if operand in registers:
            return registers[operand]
        return int(operand, 0)
    def signed(value: int, bits: int) -> int:
        value &= (1 << bits) - 1
        return value - (1 << bits) if value >= 1 << (bits - 1) else value
    def write(operand: str, value: int) -> None:
        require(operand in registers or operand in aliases, "unknown sizing destination")
        registers[aliases.get(operand, operand)] = value & ((1 << (32 if operand in aliases else 64)) - 1)
    addresses = sorted(a for a in code if start <= a < (0x180074770 if start == 0x180074710 else 0x180074814))
    successors = dict(zip(addresses, addresses[1:]))
    pc = start
    above = False
    for _ in range(128):
        mnemonic, operands = code[pc]
        parts = operands.split(",")
        if mnemonic == "ret":
            return registers["rax"]
        if mnemonic == "mov":
            write(parts[0], read(parts[1]))
        elif mnemonic == "movsxd":
            write(parts[0], signed(read(parts[1]), 32))
        elif mnemonic == "cdq":
            write("edx", -1 if signed(read("eax"), 32) < 0 else 0)
        elif mnemonic == "cmp":
            above = read(parts[0]) > read(parts[1])
        elif mnemonic in ("ja", "jbe"):
            if (above if mnemonic == "ja" else not above):
                pc = int(operands, 0)
                continue
        elif mnemonic in ("add", "sub", "and", "xor", "imul", "sar"):
            left, right = read(parts[0]), read(parts[1])
            if mnemonic == "add": value = left + right
            elif mnemonic == "sub": value = left - right
            elif mnemonic == "and": value = left & right
            elif mnemonic == "xor": value = left ^ right
            elif mnemonic == "imul": value = left * right
            else: value = signed(left, 32 if parts[0] in aliases else 64) >> right
            write(parts[0], value)
        elif mnemonic == "lea":
            expression = parts[1]
            if expression == "[rdx+rax*1]": value = read("rdx") + read("rax")
            elif expression == "[rax*4+0x3]": value = read("rax") * 4 + 3
            elif expression == "[r8+r8*1]": value = read("r8") * 2
            else: raise ValueError("unsupported sizing address expression")
            write(parts[0], value)
        else:
            raise ValueError("unsupported sizing instruction")
        require(pc in successors, "sizing leaf missing return")
        pc = successors[pc]
    raise ValueError("sizing interpreter instruction bound")


def sizing_contract(code: dict[int, tuple[str, str]]) -> dict[str, Any]:
    for address, mnemonic, operands in SIZING_ANCHORS:
        graph.expect(code, address, mnemonic, operands)
    return dict(conditional_on_concrete_decoder_vtable=True,
                query_methods={"counter": "0x180074710", "feature": "0x1800747c0"},
                query_slots={"counter": "0x40", "feature": "0x48"},
                index_domain="unsigned index 0 or 1; greater indices return zero",
                query_axes="caller-provided axes; relation to decoder launch axes unverified",
                padded_half_axes="align4(floor(query_axis/2))",
                counter_return="2 * (half_axis0/4) * (half_axis1/4)",
                counter_reservation="align512(4 * counter_return)",
                feature_return="2 * 512 * half_axis0 * half_axis1",
                feature_consumer="return stored unchanged in 16-byte ledger on inspected normal-return path",
                ledger_provenance="r14 retained on both caller paths; growth helper saves/restores r14",
                ledger_call_preservation_scope="normal return, ABI-conforming nested calls",
                allocation_sizes_not_all_recovered=True,
                counter_initial_value_verified=False,
                activation_element_type_and_addressing_verified=False,
                decoder_kernels_launched=0)


def sizing_self_test(code: dict[int, tuple[str, str]] | None = None) -> dict[str, int]:
    known = ((0, 8, 8, (2, 512, 16384)), (1, 9, 17, (4, 512, 32768)),
             (0, 17, 9, (4, 512, 32768)), (0, 16, 32, (16, 512, 131072)),
             (1, 64, 64, (128, 512, 1048576)), (0, 72, 64, (144, 1024, 1179648)),
             (0, 1, 9, (0, 0, 0)), (2, 8, 8, (0, 0, 0)),
             (0xffffffffffffffff, 8, 8, (0, 0, 0)))
    if code is None:
        code = {a: (m, o) for a, m, o in SIZING_ANCHORS}
    for index, axis0, axis1, expected in known:
        require(tuple(sizing_values(index, axis0, axis1).values()) == expected, "sizing known answer mismatch")
        require(emulate_sizing(code, 0x180074710, index, axis0, axis1) == expected[0], "counter machine known answer")
        require(emulate_sizing(code, 0x1800747C0, index, axis0, axis1) == expected[2], "feature machine known answer")
    cases = 0
    for index in (0, 1, 2, 0xffffffffffffffff):
        for axis0 in (1, 2, 7, 8, 9, 15, 16, 17, 72, 1080, 0x7fffffff - 6):
            for axis1 in (1, 3, 8, 9, 31, 64, 1440):
                values = sizing_values(index, axis0, axis1)
                require(emulate_sizing(code, 0x180074710, index, axis0, axis1) == values["counter_words"], "counter formula mismatch")
                require(emulate_sizing(code, 0x1800747C0, index, axis0, axis1) == values["feature_ledger_value"], "feature formula mismatch")
                cases += 1
    invalid = ((True, 8, 8), (-1, 8, 8), (1 << 64, 8, 8), (0, 0, 8),
               (0, 8, -1), (0, 8.0, 8), (0, 8, True), (0, 0x7fffffff, 8),
               (0, 0x7fffffff - 6, 0x7fffffff - 6))
    rejected = 0
    for case in invalid:
        try:
            sizing_values(*case)
        except ValueError:
            rejected += 1
    require(rejected == len(invalid), "invalid sizing domain accepted")
    mutations = ((0x180074767, ("xor", "r8,r8"), 0x180074710, 2),
                 (0x180074809, ("xor", "rax,rax"), 0x1800747C0, 16384),
                 (0x1800747F2, ("cmp", "r10,0x0"), 0x1800747C0, 16384))
    wrong_rejected = 0
    for address, replacement, start, expected in mutations:
        bad = dict(code)
        bad[address] = replacement
        require(emulate_sizing(bad, start, 1, 8, 8) != expected, "incorrect sizing execution accepted")
        wrong_rejected += 1
    return dict(sizing_known_answers=len(known), sizing_machine_comparisons=cases * 2,
                invalid_sizing_domains_rejected=rejected,
                incorrect_sizing_executions_rejected=wrong_rejected)


def launch_geometry(axis0: int, axis1: int) -> tuple[int, int, int]:
    # Both host padding and later +3 must fit signed int32.
    maximum = 0x7fffffff - 6
    require(type(axis0) is int and type(axis1) is int and
            0 < axis0 <= maximum and 0 < axis1 <= maximum,
            "invalid launch dimension domain")
    return (2 * ((axis1 + 3) // 4), (axis0 + 3) // 4, 4)


def backend_contract(code: dict[int, tuple[str, str]]) -> dict[str, Any]:
    for address, mnemonic, operands in BACKEND_ANCHORS:
        graph.expect(code, address, mnemonic, operands)
    return dict(conditional_on_generic_wrapper_and_cuda_backend=True,
                conditional_cuda_grid_order_verified=True,
                wrapper_vtable_rva="0xb28e0", cuda_backend_vtable_rva="0xbc9e8",
                wrapper_launch_method="0x1800449a0", cuda_launch_method="0x1800784d0",
                cuda_grid=["2*ceil(input_axis1/4)", "ceil(input_axis0/4)", "4"],
                cuda_block=[32, 2, 1], dynamic_shared_bytes=0,
                block_contract_scope="fresh descriptor, cache-hit dimensions not revalidated",
                argument_submission="kernelParams=null; extra=buffer-pointer/size/end",
                argument_bytes=80, stream_provenance="cuda_backend[0x3f8]",
                captured_runtime_binding=False, decoder_kernels_launched=0)


def parameter_contract(info: bytes) -> dict[str, Any]:
    """Parse bounded CUDA attribute forms; never skip an unknown record form.

    NVAL/BVAL/HVAL encode their value within the four-byte header; only SVAL
    consumes a subsequent byte range. Non-ABI attributes are not interpreted.
    """
    position = 0
    sizes: list[int] = []
    parameters: list[tuple[int, int, int]] = []
    while position < len(info):
        form, attribute, length = runtime.unpack("<BBH", info, position)
        position += 4
        if form == 4:
            value = runtime.span(info, position, length)
            position += length
            if attribute == 0x17:
                require(length == 12, "unexpected parameter record length")
                index, ordinal_offset, packed = struct.unpack("<III", value)
                parameters.append((index, ordinal_offset, packed >> 18))
            require(attribute != 0x19, "parameter size in wrong metadata form")
        elif form in (1, 2, 3):
            require(attribute != 0x17, "parameter record in wrong metadata form")
            if attribute == 0x19:
                require(form == 3, "parameter size requires HVAL")
                sizes.append(length)
        else:
            raise ValueError("unsupported CUDA metadata form")
    require(sizes == [80] and parameters == [(0, 0, 80)], "unexpected decoder parameter ABI")
    return dict(parameter_count=1, ordinal=0, offset=0, bytes=80)


def module_contract(data: bytes) -> dict[str, Any]:
    inventory = runtime.inventory(data)
    container = inventory["fatbins"][6]
    entries = [entry for entry in container["entries"] if entry["kind"] == 2 and entry["arch"] == 89]
    require(len(entries) == 1, "missing unique decoder SM89 image")
    image = runtime.decompress(data, entries[0])
    functions = runtime.elf_functions(image)
    require(set(functions) == {name for _, name in KERNELS} | {"cc_cb_clear"}, "unexpected decoder symbols")
    table = runtime.unpack("<Q", image, 40)[0]
    stride, count, names_index = runtime.unpack("<HHH", image, 58)
    require(stride == 64 and 0 < count <= 4096 and names_index < count, "bad section table")
    sections = [runtime.unpack("<IIQQQQIIQQ", image, table + index * stride) for index in range(count)]
    require(sections[names_index][1] == 3, "section names not STRTAB")
    strings = runtime.span(image, sections[names_index][4], sections[names_index][5])
    contracts = []
    for _, name in KERNELS:
        matches = [section for section in sections if runtime.string_at(strings, section[0]) == ".nv.info." + name]
        require(len(matches) == 1, "missing unique decoder metadata")
        info = runtime.span(image, matches[0][4], matches[0][5])
        contracts.append(dict(kernel=name, **parameter_contract(info)))
    records = [record for record in inventory["weights"]["records"] if record["name"] == "block39.layer0.layer"]
    require(len(records) == 1 and records[0]["payload_bytes"] == 525312,
            "unexpected candidate decoder record")
    return dict(container_index=6, arch=89, kernel_parameters=contracts,
                candidate_weight_record=records[0]["name"], candidate_weight_bytes=525312,
                weight_record_to_decoder_binding_verified=False,
                weight_packing_verified=False, proprietary_payload_files_written=False)


def host_contract(code: dict[int, tuple[str, str]]) -> dict[str, Any]:
    for address, mnemonic, operands in ANCHORS:
        graph.expect(code, address, mnemonic, operands)
    require(bool(ANCHORS), "empty decoder host contract")
    return dict(input_slots_minimum=2, output_slots_minimum=4,
                channels=[1024, 512], argument_bytes=80,
                pointer_slot_provenance=["input[0]", "input[1]", "output[0]", "output[3]",
                                         "output[1]", "tile_sync ? output[2] : null",
                                         "output_count > 4 ? output[4] : null", "state[0]"],
                dimension_slots=["align4(input_axis0)", "align4(input_axis1)",
                                 "align4(override_axis0 > 0 ? override_axis0 : 2*input_axis0)",
                                 "align4(override_axis1 > 0 ? override_axis1 : 2*input_axis1)"],
                fp8_selection="layer name contains _fp8",
                tile_sync_selection="global flag && output[2] != null && tile_kernel != null",
                backend_launch_integer_arguments=["2*ceil(input_axis1/4)", "ceil(input_axis0/4)", "4"],
                dimension_domain="positive signed dimensions, without integer overflow",
                spatial_axis_names_verified=False,
                cuda_grid_order_verified=False, concrete_activation_layouts_verified=False,
                original_model_linkage_verified=False, native_image_inference=False,
                gpu_kernels_launched=0)


def self_test() -> dict[str, Any]:
    require(ct.sizeof(DecoderArguments) == 80, "decoder host structure size")
    for index, (name, _) in enumerate(DecoderArguments._fields_):
        expected = index * 8 if index < 8 else 64 + (index - 8) * 4
        require(getattr(DecoderArguments, name).offset == expected, "decoder host field offset")
    # Independent literal bytes including all three immediate forms.
    info = bytes.fromhex("03 19 50 00 04 17 0c 00 00 00 00 00 00 00 00 00 00 f0 41 01 02 4c 01 00 01 04 00 00")
    parameter_contract(info)
    rejected = 0
    malformed = (b"", info[:-1], info + info, b"\x05" + info[1:],
                 info[:2] + b"\x48\x00" + info[4:],
                 info[:12] + b"\x01\x00\x00\x00" + info[16:],
                 b"\x02" + info[1:], info[:4] + b"\x03" + info[5:])
    for bad in malformed:
        try:
            parameter_contract(bad)
        except ValueError:
            rejected += 1
    require(rejected == len(malformed), "ABI malformed control accepted")
    known = (((4, 4), (2, 1, 4)), ((5, 9), (6, 2, 4)),
             ((9, 5), (4, 3, 4)), ((32, 68), (34, 8, 4)))
    for shape, expected in known:
        require(launch_geometry(*shape) == expected, "grid known answer mismatch")
    bad_shapes = ((0, 4), (4, -1), (True, 4), (4, 1.0), (0x7fffffff, 4))
    shape_rejected = 0
    for shape in bad_shapes:
        try:
            launch_geometry(*shape)
        except ValueError:
            shape_rejected += 1
    require(shape_rejected == len(bad_shapes), "invalid dimension accepted")
    return dict(**sizing_self_test(), **arena_self_test(), grid_known_answers=len(known), invalid_dimensions_rejected=shape_rejected,
                host_structure_bytes=80, host_offsets_checked=12,
                metadata_known_answers=1, malformed_metadata_rejected=rejected)


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
    pe = runtime.PE(data)
    require(runtime.string_at(data, pe.offset(0xB1AF0, 12)) == "cc_cb_clear",
            "unexpected counter-reset kernel name")
    for address, name in KERNELS:
        require(runtime.string_at(data, pe.offset(address, len(name) + 1)) == name, "wrong kernel-name target")
    # Tie the concrete constructor vtable to inspected launch/init methods.
    require(runtime.unpack("<10Q", data, pe.offset(0xBC2A8, 80)) ==
            (0x180073D50, 0x1800744B0, 0x1800612A0, 0x180074770,
             0x180061100, 0x180060CD0, 0x180074820, 0x1800749F0,
             0x180074710, 0x1800747C0), "unexpected decoder vtable")
    require(runtime.unpack("<6Q", data, pe.offset(0xB28E0, 48)) ==
            (0x180044710, 0x180044B70, 0x180044B60, 0x180044C60,
             0x180044830, 0x1800449A0), "unexpected generic wrapper vtable")
    for offset, target in ((0x08, 0x180078ED0), (0x60, 0x1800776D0),
                           (0xD8, 0x18004D1B0), (0x140, 0x1800784D0)):
        require(runtime.unpack("<Q", data, pe.offset(0xBC9E8 + offset, 8))[0] == target,
                "unexpected CUDA backend vtable")
    require(runtime.string_at(data, pe.offset(0xBD1D0, 15)) == "cuLaunchKernel",
            "unexpected loaded CUDA symbol")
    code = graph.disassemble(data, RANGES)
    report = host_contract(code)
    report["module"] = module_contract(data)
    report["backend"] = backend_contract(code)
    report["arena"] = arena_contract(code)
    report["counter_reset"] = counter_reset_contract(code)
    report["sizing"] = sizing_contract(code)
    report["sizing"]["decoded_instruction_checks"] = sizing_self_test(code)
    if args.mode == "negative":
        rejected = 0
        for address, _, _ in ANCHORS:
            bad = dict(code)
            bad[address] = ("nop", "")
            try:
                host_contract(bad)
            except ValueError:
                rejected += 1
        require(rejected == len(ANCHORS), "host anchor corruption accepted")
        backend_rejected = 0
        for address, _, _ in BACKEND_ANCHORS:
            bad = dict(code)
            bad[address] = ("nop", "")
            try:
                backend_contract(bad)
            except ValueError:
                backend_rejected += 1
        require(backend_rejected == len(BACKEND_ANCHORS), "backend anchor corruption accepted")
        sizing_rejected = 0
        for address, _, _ in SIZING_ANCHORS:
            bad = dict(code)
            bad[address] = ("nop", "")
            try:
                sizing_contract(bad)
            except ValueError:
                sizing_rejected += 1
        require(sizing_rejected == len(SIZING_ANCHORS), "sizing anchor corruption accepted")
        arena_rejected = 0
        for address, _, _ in ARENA_ANCHORS:
            bad = dict(code)
            bad[address] = ("nop", "")
            try:
                arena_contract(bad)
            except ValueError:
                arena_rejected += 1
        require(arena_rejected == len(ARENA_ANCHORS), "arena anchor corruption accepted")
        counter_rejected = 0
        for address, _, _ in COUNTER_RESET_ANCHORS:
            bad = dict(code)
            bad[address] = ("nop", "")
            try:
                counter_reset_contract(bad)
            except ValueError:
                counter_rejected += 1
        require(counter_rejected == len(COUNTER_RESET_ANCHORS), "counter reset corruption accepted")
        report = dict(counter_reset_anchor_corruptions_rejected=counter_rejected,
                      arena_anchor_corruptions_rejected=arena_rejected,
                      sizing_anchor_corruptions_rejected=sizing_rejected,
                      backend_anchor_corruptions_rejected=backend_rejected,
                       host_anchor_corruptions_rejected=rejected, gpu_kernels_launched=0,
                      numerical_decoder_validation=False)
    print(json.dumps(dict(mode=args.mode, dll_sha256=runtime.PIN, memory_policy=policy,
                         cpu_self_test=self_test(), decoder=report), indent=2))


if __name__ == "__main__":
    main()
