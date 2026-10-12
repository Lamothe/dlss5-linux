#!/usr/bin/env python3
"""Train/hold-out validation of the decoder (o, i) -> matrix-byte structural
model through native Linux CUDA; NOT neural inference.

Launches the pinned SM89 `cc_dec_input_upsample_1024_512_fp8` on the RTX
4070 Ti SUPER through native Linux CUDA (no Windows code, DirectX or Proton)
and validates the structural extension that
[NATIVE-DECODER-PERMUTATION.md](NATIVE-DECODER-PERMUTATION.md) leaves open:
a CLOSED FORM that predicts, for any single-nonzero input byte (p, u, l),
which output bytes respond and which physical matrix byte feeds each one.

Model (the hypothesis under test):

  * Let `n = u*16 + l` be the input byte index inside its main-input slice.
    The matrix byte feeding output byte `o` is `m = B + S(o)` where `S` is
    the FIXED bit shuffle `SHUFFLE_STRIDES` and `B` is the sum of the per-bit
    base effects of the set bits of `n` (additivity under test).
  * Each index bit also contributes a row term, a channel-block shift and a
    within-unit byte shift to the influenced output cells (the reach rule of
    [NATIVE-DECODER-SWEEP.md](NATIVE-DECODER-SWEEP.md) generalised per bit;
    the byte shift was discovered by the training phase).
  * Outputs fed by matrix byte 0 are invisible to bit-signature decoding
    (offset 0 is never coded), so the decoded map omits exactly the model's
    `m == 0` entries.

Training/hold-out split: all 12 index bits are measured singly (training) and
their per-bit effects fitted; 8 multi-bit input bytes never measured before
(hold-out) must then match composed predictions EXACTLY (full change maps).
Direct runs re-measure selected triples and the matrix-byte-0 preimages.
`matrix_permutation_recovered` flips true only if every check passes; its
scope is the validated model (12 single-bit fits plus exact matches on 21 of
the 4*4096 input bytes; the closed form extrapolates over the rest).

It does NOT use learned weights and produces NO neural-rendered image.
Bounded-transfer note: per-entry out0 traffic is limited to [0, EXTENT_BYTES +
4096); the full region is poisoned once per run and verified after the final
entry (a write whose bytes coincide with the 0xA5 poison is not detectable).
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import ctypes as ct
import importlib.util
import json
from pathlib import Path
from typing import Any


def load_helper(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    if spec is None or spec.loader is None:
        raise RuntimeError("missing helper")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


permutation = load_helper("permutation", "native-decoder-permutation-probe.py")
base = permutation.base
sweep = permutation.sweep
launch = permutation.launch
weights = permutation.weights
utility = permutation.utility
decoder = permutation.decoder
OutputMismatch = permutation.OutputMismatch
NAME = permutation.NAME
RESET = permutation.RESET
BLOCK = permutation.BLOCK
INPUT_AXES = permutation.INPUT_AXES
OUTPUT_AXES = permutation.OUTPUT_AXES
DecoderArguments = permutation.DecoderArguments
GUARD_WORDS = permutation.GUARD_WORDS
allocate = permutation.allocate
kernel_metadata = permutation.kernel_metadata
reset_geometry = permutation.reset_geometry
factor_cell = permutation.factor_cell
canonical_units = permutation.canonical_units
input_key = permutation.input_key
key_name = permutation.key_name
main_byte = permutation.main_byte
weights_signature = permutation.weights_signature
weights_single = permutation.weights_single
expand_ranges = permutation.expand_ranges
decode_column = permutation.decode_column
differing_ranges = permutation.differing_ranges
e4m3 = permutation.e4m3
finite = permutation.finite
MATRIX_BYTES = permutation.MATRIX_BYTES
PARTITION_BYTES = permutation.PARTITION_BYTES
MAIN_PARTITION_BYTES = permutation.MAIN_PARTITION_BYTES
IMPULSE_BYTES = permutation.IMPULSE_BYTES
UNITS_PER_PARTITION = permutation.UNITS_PER_PARTITION
E4M3_POS = permutation.E4M3_POS
E4M3_NEG = permutation.E4M3_NEG
E4M3_ZERO = permutation.E4M3_ZERO
PROTOCOL_FINAL = permutation.PROTOCOL_FINAL
RESET_FINAL = permutation.RESET_FINAL
EXTENT_BYTES = permutation.EXTENT_BYTES
SIGNATURE_BITS = permutation.SIGNATURE_BITS
OUT0_WINDOW = permutation.OUT0_WINDOW

# ---------------------------------------------------------------------------
# The structural model under test.
# ---------------------------------------------------------------------------

UNIT_BYTES = (0, 1, 2, 3, 8, 9, 10, 11)      # established within-unit byte pattern
BYTE_SHIFTS = (0, 4)                          # measured byte-pattern shifts
ROW_STRIDE = 8192                             # row_term*8192 + column*512 + block*16
CELL_COLUMNS = 16
BLOCK_BASES = (0, 16)
BIT_COUNT = 12                                # n = u*16 + l spans 12 bits
LANE_BITS = 4
GEOM_MAX_ROW = 15                             # offset decomposition stays in extent

# Fixed within-cell bit shuffle: output byte bit -> matrix byte stride
# (output bits 2, 6, 7, 8 and 13+ contribute nothing; bit 2 never varies).
SHUFFLE_STRIDES = {0: 64, 1: 8, 3: 512, 4: 128, 5: 256,
                   9: 1024, 10: 2048, 11: 4096, 12: 8192}


def shuffle(output_offset: int) -> int:
    """S(o): fixed bit shuffle from output byte offset to matrix byte offset."""
    base.require(type(output_offset) is int and 0 <= output_offset < EXTENT_BYTES,
                 "shuffle offset range")
    return sum(stride * ((output_offset >> bit) & 1)
               for bit, stride in SHUFFLE_STRIDES.items())


def bit_key(bit: int) -> tuple[int, int, int]:
    """Input key whose byte index n = u*16 + l is exactly 2**bit."""
    base.require(type(bit) is int and 0 <= bit < BIT_COUNT, "index bit")
    if bit < LANE_BITS:
        return 0, 0, 1 << bit
    return 0, 1 << (bit - LANE_BITS), 0


def index_bits(unit: int, lane: int) -> int:
    base.require(type(unit) is int and 0 <= unit < UNITS_PER_PARTITION, "index unit")
    base.require(type(lane) is int and 0 <= lane < IMPULSE_BYTES, "index lane")
    return unit * IMPULSE_BYTES + lane


def compose(effects: dict[int, tuple[int, int, int, int]], unit: int, lane: int) -> tuple[int, int, int, int]:
    """Additive composition of per-bit effects (base, row, block, byte) over n."""
    base.require(isinstance(effects, dict) and set(effects) == set(range(BIT_COUNT)),
                 "effects cover all index bits")
    for effect in effects.values():
        base.require(isinstance(effect, tuple) and len(effect) == 4, "effect tuple")
        for value in effect:
            base.require(type(value) is int, "effect int")
    total_base = total_row = total_block = total_byte = 0
    for bit in range(BIT_COUNT):
        if (index_bits(unit, lane) >> bit) & 1:
            effect = effects[bit]
            total_base += effect[0]
            total_row += effect[1]
            total_block += effect[2]
            total_byte += effect[3]
    base.require(0 <= total_row <= GEOM_MAX_ROW, "composed row term")
    base.require(total_block in (0, 8), "composed block shift")
    base.require(total_byte in BYTE_SHIFTS, "composed byte shift")
    return total_base, total_row, total_block, total_byte


def model_map(effects: dict[int, tuple[int, int, int, int]], unit: int, lane: int) -> dict[int, int]:
    """Full predicted {output byte: matrix byte} map, INCLUDING m == 0 entries."""
    base_value, row, block_shift, byte_shift = compose(effects, unit, lane)
    bytes_used = tuple(byte + byte_shift for byte in UNIT_BYTES)
    base.require(len(set(bytes_used)) == len(UNIT_BYTES), "byte shift keeps pattern distinct")
    out: dict[int, int] = {}
    for block_base in BLOCK_BASES:
        for inner in range(8):
            block = block_shift + block_base + inner
            for column in range(CELL_COLUMNS):
                for byte in bytes_used:
                    offset = row * ROW_STRIDE + column * 512 + block * 16 + byte
                    base.require(0 <= offset < EXTENT_BYTES, "model output bound")
                    out[offset] = base_value + shuffle(offset)
    base.require(len(out) == 2048, "model map size")
    return out


def signature_map(full: dict[int, int]) -> dict[int, int]:
    """Visible part of a model map: matrix byte 0 is never signature-coded."""
    return {offset: matrix_byte for offset, matrix_byte in full.items() if matrix_byte != 0}


def zero_preimage(full: dict[int, int]) -> list[int]:
    """Outputs fed by matrix byte 0 (invisible to signature runs)."""
    return sorted(offset for offset, matrix_byte in full.items() if matrix_byte == 0)


def fit_effect(decode: dict[int, int]) -> tuple[int, int, int, int]:
    """Fit one bit effect (base, row, block shift, byte shift) from a training
    column.

    The decode must be constant in `m - S(o)`, cover the established 8 byte
    offsets (possibly uniformly shifted), one row term, and the shifted
    canonical block set.
    """
    base.require(isinstance(decode, dict) and decode, "fit decode non-empty")
    offsets: set[int] = set()
    for offset, matrix_byte in decode.items():
        base.require(type(offset) is int and type(matrix_byte) is int, "fit entry types")
        base.require(0 <= offset < EXTENT_BYTES, "fit output bound")
        base.require(type(matrix_byte) is int and matrix_byte > 0, "fit matrix byte visible")
        offsets.add(offset)
    terms = {matrix_byte - shuffle(offset) for offset, matrix_byte in decode.items()}
    base.require(len(terms) == 1, "constant base term across the column")
    base_value = terms.pop()
    cells = [factor_cell(offset // IMPULSE_BYTES * IMPULSE_BYTES) for offset in offsets]
    rows = {cell["row_term"] for cell in cells}
    blocks = {cell["channel_block"] for cell in cells}
    byte_offsets = {offset % IMPULSE_BYTES for offset in offsets}
    base.require(len(byte_offsets) == len(UNIT_BYTES), "byte pattern size")
    byte_shift = min(byte_offsets)
    base.require(byte_shift in BYTE_SHIFTS, "byte shift in measured set")
    base.require(byte_offsets == {byte + byte_shift for byte in UNIT_BYTES},
                 "established within-unit byte pattern")
    base.require(len(rows) == 1, "single row term")
    row = rows.pop()
    base.require(type(row) is int and 0 <= row <= GEOM_MAX_ROW, "row term range")
    block_shift = 8 if min(blocks) >= 8 else 0
    expected_blocks = set(range(block_shift, block_shift + 8)) | \
        set(range(block_shift + 16, block_shift + 24))
    base.require(blocks == expected_blocks, "channel blocks match shifted canonical set")
    return base_value, row, block_shift, byte_shift


# Training inputs: one per index bit (12); 8 re-measure columns of
# NATIVE-DECODER-PERMUTATION.md as regression, 4 (units 2/16/64/128) are new.
# Hold-out inputs: multi-bit bytes never probed before, spanning all 12 bits,
# all four partitions and the reach combinations; disjoint from TRAIN_INPUTS
# and from the 23 permutation columns. Identity control: n = 0 (the empty
# composition; re-measures a known column as a regression anchor).
TRAIN_INPUTS = tuple(bit_key(bit) for bit in range(BIT_COUNT))
HOLDOUT_INPUTS = ((0, 3, 5), (0, 18, 0), (0, 210, 15), (1, 100, 7),
                  (2, 200, 13), (3, 255, 15), (0, 34, 9), (1, 2, 15))
IDENTITY_INPUTS = ((0, 0, 0),)
SPOT_INPUTS = ((0, 3, 5), (2, 200, 13), (0, 32, 0))
ALL_INPUTS = TRAIN_INPUTS + HOLDOUT_INPUTS + IDENTITY_INPUTS


def phase1_plan() -> list[dict[str, Any]]:
    """Controls, per-input impulse bases, then signature runs for every input."""
    plan: list[dict[str, Any]] = [dict(name="base", kind="control", impulse=None,
                                       matrix=None, code=E4M3_ZERO)]
    for key in ALL_INPUTS:
        plan.append(dict(name=f"impulse-base-{key_name(key)}", kind="impulse-base",
                         impulse=key, matrix=None, code=E4M3_ZERO))
    plan.append(dict(name="coupled-ref", kind="control", impulse=(0, 0, None),
                     matrix=("whole", 0, E4M3_POS), code=E4M3_POS))
    plan.append(dict(name="decoupled-ctrl", kind="control", impulse=(0, 0, 0),
                     matrix=("whole", 1, E4M3_POS), code=E4M3_POS))
    for key in ALL_INPUTS:
        for bit in range(SIGNATURE_BITS):
            plan.append(dict(name=f"sig-{key_name(key)}-b{bit}", kind="signature",
                             impulse=key, matrix=("sig", key[0], bit), code=E4M3_POS))
    return plan


def phase2_plan(full_maps: dict[tuple[int, int, int], dict[int, int]]) -> list[dict[str, Any]]:
    """Direct checks: matrix-byte-0 preimages for every input, plus spot triples.

    The zero checks are the model's strongest structural claim per input: the
    outputs fed by matrix byte 0 (none when the composed base is nonzero).
    Spot checks re-measure one decoded triple per selected input directly.
    """
    plan: list[dict[str, Any]] = []
    for key in ALL_INPUTS:
        expected = zero_preimage(full_maps[key])
        plan.append(dict(name=f"direct-zero-{key_name(key)}", kind="direct", sub="zero",
                         impulse=key, matrix=("single", key[0], 0, E4M3_POS),
                         code=E4M3_POS, expected=expected))
    for key in SPOT_INPUTS:
        visible = signature_map(full_maps[key])
        matrix_byte = min(visible.values())
        expected = sorted(offset for offset, value in visible.items() if value == matrix_byte)
        plan.append(dict(name=f"direct-spot-{key_name(key)}-m{matrix_byte}", kind="direct",
                         sub="spot", impulse=key,
                         matrix=("single", key[0], matrix_byte, E4M3_POS),
                         code=E4M3_POS, expected=expected))
    return plan


def self_test() -> dict[str, Any]:
    regression = permutation.self_test()
    checks = 0
    # Shuffle known answers (unit 0 lane 1 column head of the retained evidence).
    for offset, answer in ((0, 0), (1, 64), (2, 8), (3, 72), (8, 512), (9, 576),
                           (10, 520), (11, 584), (16, 128), (24, 640)):
        base.require(shuffle(offset) == answer, f"shuffle known answer {offset}")
        checks += 1
    # Bit-key known answers and plan invariants (each counted via `check`).
    def check(condition: Any, label: str) -> None:
        nonlocal checks
        base.require(condition, label)
        checks += 1

    check(bit_key(0) == (0, 0, 1) and bit_key(3) == (0, 0, 8), "lane bit keys")
    check(bit_key(4) == (0, 1, 0) and bit_key(11) == (0, 128, 0), "unit bit keys")
    check(len(set(TRAIN_INPUTS)) == BIT_COUNT, "training inputs distinct")
    check(set(TRAIN_INPUTS).isdisjoint(HOLDOUT_INPUTS), "hold-out disjoint from training")
    check(set(IDENTITY_INPUTS).isdisjoint(TRAIN_INPUTS + HOLDOUT_INPUTS),
          "identity control distinct")
    # Hold-out inputs must be new measurements: disjoint from both the training
    # set and the 23 columns of NATIVE-DECODER-PERMUTATION.md. Training inputs
    # legitimately re-measure 8 of those columns (regression) plus 4 new ones;
    # the identity control re-measures (0,0,0) deliberately.
    check(set(HOLDOUT_INPUTS).isdisjoint(permutation.PROBE_INPUTS),
          "hold-out inputs disjoint from the 23 permutation columns")
    check(set(IDENTITY_INPUTS) <= set(permutation.PROBE_INPUTS),
          "identity control is a re-measured permutation column")
    check(all(key[0] == 0 for key in TRAIN_INPUTS), "training partition 0")
    covered = 0
    for key in HOLDOUT_INPUTS:
        n = index_bits(key[1], key[2])
        covered |= n
        check(bin(n).count("1") >= 2, "hold-out input is multi-bit")
    check(covered == (1 << BIT_COUNT) - 1, "hold-out covers all index bits")
    check({key[0] for key in HOLDOUT_INPUTS} == {0, 1, 2, 3}, "hold-out partitions")
    check(all(index_bits(key[1], key[2]) == 0 for key in IDENTITY_INPUTS),
          "identity input is n=0")
    check(len(ALL_INPUTS) == 21 and len(set(ALL_INPUTS)) == 21, "input count")
    # Model known answers: zero-effects map for (0, 0) is the measured head.
    effects = {bit: (0, 0, 0, 0) for bit in range(BIT_COUNT)}
    zero_map = model_map(effects, 0, 0)
    check(len(zero_map) == 2048, "model map size")
    check(signature_map(zero_map)[1] == 64, "model head o=1")
    check(signature_map(zero_map)[2] == 8, "model head o=2")
    check(signature_map(zero_map)[8] == 512, "model head o=8")
    check(zero_preimage(zero_map) == [0, 64, 256, 320], "model zero preimage")
    # Byte-shift known answer: bit 8's measured effect shifts the byte pattern.
    shifted = dict(effects)
    shifted[8] = (0, 0, 0, 4)
    shifted_map = model_map(shifted, 16, 0)
    check(sorted({offset % IMPULSE_BYTES for offset in shifted_map}) ==
          [4, 5, 6, 7, 12, 13, 14, 15], "byte shift pattern")
    # Fit round-trip through the model for every effect shape.
    for effect in ((0, 0, 0, 0), (16, 0, 0, 0), (8, 4, 0, 0), (0, 0, 8, 0),
                   (16384, 1, 8, 0), (16, 1, 8, 4), (32, 0, 0, 4)):
        single = {bit: (0, 0, 0, 0) for bit in range(BIT_COUNT)}
        probe_bit = 9
        single[probe_bit] = effect
        full = model_map(single, *bit_key(probe_bit)[1:])
        decoded = signature_map(full)
        fitted = fit_effect(decoded)
        check(fitted == effect, f"fit round trip {effect}")
    # Corruption controls: each must be rejected at its LABELLED check (the
    # exception message is pinned) with m - S(o) constancy preserved where the
    # target check lies beyond the constancy gate.
    corruptions = 0

    def reject(action: Any, label: str, expect: str) -> None:
        nonlocal corruptions
        try:
            action()
        except ValueError as error:
            if type(error) is not ValueError:  # OutputMismatch is not a clean rejection
                raise OutputMismatch(
                    f"corruption {label} rejected by unexpected {type(error).__name__}: {error}")
            if expect not in str(error):
                raise OutputMismatch(
                    f"corruption {label} rejected at wrong check: {error}")
            corruptions += 1
            return
        except Exception as error:  # noqa: BLE001 - a non-ValueError crash is not a clean rejection
            raise OutputMismatch(
                f"corruption {label} rejected by unexpected {type(error).__name__}: {error}")
        raise OutputMismatch(f"corruption not rejected: {label}")

    good = signature_map(model_map({bit: (0, 0, 0, 0) for bit in range(BIT_COUNT)}, 0, 0))
    bad = dict(good)
    first = min(bad)
    bad[first] = bad[first] + 1
    reject(lambda: fit_effect(bad), "varying base term", "constant base term")
    # Byte pattern 8 -> 12 across all units: exactly 8 byte offsets remain,
    # byte_shift 0 is in BYTE_SHIFTS, but the set is not the established one.
    # Bit 2 has no shuffle stride, so S(o+4) == S(o) and the base term is kept.
    bad = {(offset + 4 if offset % IMPULSE_BYTES == 8 else offset): value
           for offset, value in good.items()}
    reject(lambda: fit_effect(bad), "byte pattern not established set",
           "established within-unit byte pattern")
    # Uniform +1 offset shift with values rebuilt as 1 + S(o): base term stays
    # constant at 1, all bytes visible, min byte offset 1 outside BYTE_SHIFTS.
    bad = {offset + 1: 1 + shuffle(offset + 1) for offset in good}
    reject(lambda: fit_effect(bad), "byte shift not measured", "byte shift in measured set")
    bad = {offset: value for offset, value in good.items() if offset % IMPULSE_BYTES != 8}
    reject(lambda: fit_effect(bad), "missing byte offset 8", "byte pattern size")
    bad = {offset: value for offset, value in good.items() if offset % IMPULSE_BYTES != 0}
    reject(lambda: fit_effect(bad), "missing byte offset 0", "byte pattern size")
    reject(lambda: fit_effect({}), "empty decode", "fit decode non-empty")
    reject(lambda: fit_effect({0: 0}), "zero matrix byte", "fit matrix byte visible")
    reject(lambda: shuffle(-1), "shuffle negative", "shuffle offset range")
    reject(lambda: bit_key(BIT_COUNT), "bit key overflow", "index bit")
    reject(lambda: compose({bit: (0, 0, 0, 0) for bit in range(BIT_COUNT - 1)}, 0, 0),
           "incomplete effects", "effects cover all index bits")
    reject(lambda: compose({bit: (0, 0, 0, 0) for bit in range(BIT_COUNT)} |
                           {0: (0, 0, 4, 0)}, 0, 1), "bad block shift", "composed block shift")
    reject(lambda: compose({bit: (0, 0, 0, 0) for bit in range(BIT_COUNT)} |
                           {0: (0, 0, 0, 2)}, 0, 1), "bad byte shift", "composed byte shift")
    reject(lambda: compose({bit: (0, 0, 0, 0) for bit in range(BIT_COUNT)} |
                           {11: (0, 0, 0)}, 0, 1), "malformed unused effect", "effect tuple")
    reject(lambda: compose({bit: (0, 0, 0, 0) for bit in range(BIT_COUNT)} |
                           {11: (0, 0, 0, "x")}, 0, 1), "non-int unused effect", "effect int")
    reject(lambda: model_map({bit: (0, 0, 0, 0) for bit in range(BIT_COUNT)},
                             UNITS_PER_PARTITION, 0), "unit overflow", "index unit")
    return dict(model_and_plan_checks=checks, corruption_controls=corruptions,
                train_inputs=len(TRAIN_INPUTS), holdout_inputs=len(HOLDOUT_INPUTS),
                spot_inputs=len(SPOT_INPUTS),
                helper_chain_regression=(regression.get("probe_inputs") == 23
                                         and regression.get("signature_entries") == 391),
                gpu_kernels_launched=0)


def execute(data: bytes, inventory: dict[str, Any], mode: str) -> dict[str, Any]:
    entries = [entry for entry in inventory["fatbins"][6]["entries"]
               if entry["kind"] == 2 and entry["arch"] == 89]
    base.require(len(entries) == 1, "unique SM89 decoder module")
    image = base.decompress(data, entries[0])
    utility.verify_kernel_abi(image, RESET, 16)
    abi = decoder.parameter_contract(kernel_metadata(image, NAME))
    base.require(abi["bytes"] == 80, "decoder ABI size")
    base.require(decoder.launch_geometry(*INPUT_AXES)[2] == 4, "four partitions required")
    driver = utility.Driver()
    device, version = utility.select_device(driver)
    launches = readbacks = 0
    guard_words = [0x12340000 + i for i in range(GUARD_WORDS)]
    poison = launch.OUT0_PATTERN
    poison_window = bytes([poison]) * OUT0_WINDOW
    poison_full = bytes([poison]) * launch.OUT0_BYTES
    records: list[dict[str, Any]] = []

    def fail(message: str) -> None:
        print(json.dumps(dict(failure=message, records=records), indent=2))
        raise OutputMismatch(message)

    with ExitStack() as resources:
        context = ct.c_void_p()
        driver.check("cuCtxCreate_v2", ct.byref(context), 0, device)
        resources.callback(driver.check, "cuCtxDestroy_v2", context)
        module = ct.c_void_p()
        driver.check("cuModuleLoadData", ct.byref(module), ct.create_string_buffer(image))
        resources.callback(driver.check, "cuModuleUnload", module)
        decoder_function, reset_function = ct.c_void_p(), ct.c_void_p()
        driver.check("cuModuleGetFunction", ct.byref(decoder_function), module, NAME.encode())
        driver.check("cuModuleGetFunction", ct.byref(reset_function), module, RESET.encode())
        with ExitStack() as allocations:
            addresses = {}
            for key, size in (("main", launch.LARGE_BYTES), ("skip", launch.LARGE_BYTES),
                              ("weights", launch.WEIGHTS_BYTES), ("partials", launch.LARGE_BYTES),
                              ("out0", launch.OUT0_BYTES), ("out3", launch.SMALL_BYTES),
                              ("out2", launch.SMALL_BYTES)):
                addresses[key] = allocate(driver, allocations, size)
            counter_base, counter_region, _ = allocate(driver, allocations, launch.COUNTER_WORDS * 4)

            def upload(key: str, content: bytes) -> bytes:
                base_address, region, size = addresses[key]
                base.require(len(content) == size, "upload size")
                host = (ct.c_uint8 * (size + 2 * GUARD_WORDS * 4))()
                ct.memmove(ct.byref(host, GUARD_WORDS * 4), content, size)
                for index, word in enumerate(guard_words):
                    ct.cast(ct.byref(host, index * 4), ct.POINTER(ct.c_uint32)).contents.value = word
                    ct.cast(ct.byref(host, GUARD_WORDS * 4 + size + index * 4),
                            ct.POINTER(ct.c_uint32)).contents.value = word
                driver.check("cuMemcpyHtoD_v2", ct.c_uint64(base_address), host, ct.sizeof(host))
                return content

            def download(key: str) -> tuple[bytes, bool]:
                base_address, region, size = addresses[key]
                host = (ct.c_uint8 * (size + 2 * GUARD_WORDS * 4))()
                driver.check("cuMemcpyDtoH_v2", host, ct.c_uint64(base_address), ct.sizeof(host))
                region_bytes = ct.string_at(ct.byref(host, GUARD_WORDS * 4), size)
                intact = all(ct.cast(ct.byref(host, index * 4),
                                     ct.POINTER(ct.c_uint32)).contents.value == word
                             for index, word in enumerate(guard_words))
                intact &= all(ct.cast(ct.byref(host, GUARD_WORDS * 4 + size + index * 4),
                                      ct.POINTER(ct.c_uint32)).contents.value == word
                              for index, word in enumerate(guard_words))
                return region_bytes, bool(intact)

            def upload_window(content: bytes) -> None:
                base.require(len(content) == OUT0_WINDOW, "window upload size")
                base_address, region, size = addresses["out0"]
                host = (ct.c_uint8 * (GUARD_WORDS * 4 + OUT0_WINDOW))()
                for index, word in enumerate(guard_words):
                    ct.cast(ct.byref(host, index * 4), ct.POINTER(ct.c_uint32)).contents.value = word
                ct.memmove(ct.byref(host, GUARD_WORDS * 4), content, OUT0_WINDOW)
                driver.check("cuMemcpyHtoD_v2", ct.c_uint64(base_address), host, ct.sizeof(host))

            def download_window() -> tuple[bytes, bool]:
                base_address, region, size = addresses["out0"]
                host = (ct.c_uint8 * (GUARD_WORDS * 4 + OUT0_WINDOW))()
                driver.check("cuMemcpyDtoH_v2", host, ct.c_uint64(base_address), ct.sizeof(host))
                region_bytes = ct.string_at(ct.byref(host, GUARD_WORDS * 4), OUT0_WINDOW)
                intact = all(ct.cast(ct.byref(host, index * 4),
                                     ct.POINTER(ct.c_uint32)).contents.value == word
                             for index, word in enumerate(guard_words))
                return region_bytes, bool(intact)

            def reset_counters() -> None:
                nonlocal launches
                counters = (ct.c_uint8 * (launch.COUNTER_WORDS * 4 + 2 * GUARD_WORDS * 4))()
                for index, word in enumerate(guard_words):
                    ct.cast(ct.byref(counters, index * 4), ct.POINTER(ct.c_uint32)).contents.value = word
                    ct.cast(ct.byref(counters, GUARD_WORDS * 4 + launch.COUNTER_WORDS * 4 + index * 4),
                            ct.POINTER(ct.c_uint32)).contents.value = word
                counter_values = (ct.c_int32 * launch.COUNTER_WORDS)(*([0] * launch.COUNTER_WORDS))
                ct.memmove(ct.byref(counters, GUARD_WORDS * 4), counter_values, ct.sizeof(counter_values))
                driver.check("cuMemcpyHtoD_v2", ct.c_uint64(counter_base), counters, ct.sizeof(counters))
                grid = reset_geometry(launch.COUNTER_WORDS)
                reset_args = utility.CounterArguments(counter_region, launch.COUNTER_WORDS)
                params = (ct.c_void_p * 1)(ct.addressof(reset_args))
                driver.check("cuLaunchKernel", reset_function, *grid, 256, 1, 1, 0, None, params, None)
                launches += 1
                driver.check("cuCtxSynchronize")

            def read_counters() -> tuple[list[int], bool]:
                nonlocal readbacks
                counters = (ct.c_uint8 * (launch.COUNTER_WORDS * 4 + 2 * GUARD_WORDS * 4))()
                driver.check("cuMemcpyDtoH_v2", counters, ct.c_uint64(counter_base), ct.sizeof(counters))
                readbacks += 1
                values = ct.cast(ct.byref(counters, GUARD_WORDS * 4),
                                 ct.POINTER(ct.c_int32 * launch.COUNTER_WORDS)).contents
                intact = all(ct.cast(ct.byref(counters, index * 4),
                                     ct.POINTER(ct.c_uint32)).contents.value == word
                             for index, word in enumerate(guard_words))
                intact &= all(ct.cast(ct.byref(counters, GUARD_WORDS * 4 + launch.COUNTER_WORDS * 4 + index * 4),
                                      ct.POINTER(ct.c_uint32)).contents.value == word
                              for index, word in enumerate(guard_words))
                return list(values), bool(intact)

            def matrix_upload(spec: tuple[Any, ...] | None) -> bytes:
                if spec is None:
                    return sweep.weights_subblock(None, E4M3_ZERO, 0, 0)
                kind = spec[0]
                if kind == "whole":
                    return sweep.weights_subblock(spec[1], spec[2], 0, PARTITION_BYTES)
                if kind == "sig":
                    return weights_signature(spec[1], spec[2])
                if kind == "single":
                    return weights_single(spec[1], spec[2], spec[3])
                fail(f"unknown matrix spec {spec}")
                raise AssertionError("unreachable")

            def run_entry(entry: dict[str, Any], base_region: bytes | None) -> tuple[dict[str, Any], bytes, set[int]]:
                nonlocal launches, readbacks
                name = entry["name"]
                key = entry["impulse"]
                snapshots = {
                    "main": upload("main", main_byte(key) if key is not None else bytes(launch.LARGE_BYTES)),
                    "skip": upload("skip", bytes(launch.LARGE_BYTES)),
                    "weights": upload("weights", matrix_upload(entry["matrix"])),
                    "partials": upload("partials", bytes(launch.LARGE_BYTES)),
                    "out3": upload("out3", bytes([launch.IMMUTABLE_PATTERN]) * launch.SMALL_BYTES),
                    "out2": upload("out2", bytes([launch.IMMUTABLE_PATTERN]) * launch.SMALL_BYTES),
                }
                upload_window(poison_window)
                reset_counters()
                arguments = DecoderArguments(
                    main=addresses["main"][1], skip=addresses["skip"][1],
                    out0=addresses["out0"][1], out3=addresses["out3"][1],
                    counters=counter_region, out2=addresses["out2"][1],
                    partials=addresses["partials"][1], weights=addresses["weights"][1],
                    dimension=INPUT_AXES[0], width=INPUT_AXES[1],
                    output_axis0=OUTPUT_AXES[0], output_axis1=OUTPUT_AXES[1])
                grid_x, grid_y, grid_z = decoder.launch_geometry(*INPUT_AXES)
                params = (ct.c_void_p * 1)(ct.addressof(arguments))
                driver.check("cuLaunchKernel", decoder_function, grid_x, grid_y, grid_z,
                             *BLOCK, 0, None, params, None)
                launches += 1
                driver.check("cuCtxSynchronize")
                observed, counters_intact = read_counters()
                immutable = True
                guards_intact = True
                for region_key in ("main", "skip", "weights", "out3", "out2"):
                    region_bytes, intact = download(region_key)
                    readbacks += 1
                    immutable &= region_bytes == snapshots[region_key]
                    guards_intact &= intact
                partials_bytes, partials_intact = download("partials")
                readbacks += 1
                guards_intact &= partials_intact
                out0_window, window_intact = download_window()
                readbacks += 1
                guards_intact &= window_intact
                if base_region is None:
                    ranges = []
                else:
                    ranges = differing_ranges(base_region[:OUT0_WINDOW], out0_window)
                changed = expand_ranges([[start, end] for start, end in ranges], OUT0_WINDOW)
                if changed:
                    first, last = min(changed), max(changed)
                else:
                    first, last = -1, -1
                sample = sorted(changed)[:8]
                record = dict(name=name, kind=entry["kind"], sub=entry.get("sub"),
                              impulse_partition=None if key is None else key[0],
                              impulse_unit=None if key is None else key[1],
                              impulse_lane=None if key is None else key[2],
                              matrix_spec=None if entry["matrix"] is None else list(entry["matrix"]),
                              code_byte=entry["code"],
                              counters_observed=observed,
                              counters_guard_intact=bool(counters_intact),
                              changed_ranges=[[start, end] for start, end in ranges],
                              changed_count=len(changed), changed_first=first, changed_last=last,
                              sample_offsets=sample,
                              sample_bytes=[out0_window[index] for index in sample],
                              sample_values=[finite(e4m3(out0_window[index])) for index in sample],
                              expected_offsets=entry.get("expected"),
                              guards_intact=bool(guards_intact), immutable=bool(immutable),
                              prediction_match=None if entry.get("expected") is None
                              else (sorted(changed) == sorted(entry["expected"])))
                records.append(record)
                # Invariant checks run after the record is retained so a failure
                # preserves the measurements that explain it (evidence-first).
                if changed and last >= EXTENT_BYTES:
                    fail(f"output write outside the established extent ({name}): {last}")
                if not counters_intact:
                    fail(f"counter guard violation ({name})")
                if observed != [PROTOCOL_FINAL] * launch.COUNTER_WORDS:
                    fail(f"counter protocol mismatch ({name}): {observed}")
                if not (immutable and guards_intact):
                    fail(f"guard/canary/immutability violation ({name})")
                return record, out0_window, changed

            if mode == "skip":
                initial = {key: upload(key, content) for key, content in
                           (("main", main_byte(ALL_INPUTS[0])), ("skip", bytes(launch.LARGE_BYTES)),
                            ("weights", sweep.weights_subblock(None, E4M3_ZERO, 0, 0)),
                            ("partials", bytes(launch.LARGE_BYTES)),
                            ("out0", poison_full),
                            ("out3", bytes([launch.IMMUTABLE_PATTERN]) * launch.SMALL_BYTES),
                            ("out2", bytes([launch.IMMUTABLE_PATTERN]) * launch.SMALL_BYTES))}
                reset_counters()
                observed, counters_intact = read_counters()
                if not counters_intact:
                    fail("skip counter guard violation")
                if observed != [RESET_FINAL] * launch.COUNTER_WORDS:
                    fail(f"skip counters not the reset value: {observed}")
                guards_intact = True
                immutable = True
                out0: bytes | None = None
                for key in ("main", "skip", "weights", "partials", "out0", "out3", "out2"):
                    region_bytes, intact = download(key)
                    readbacks += 1
                    guards_intact &= intact
                    immutable &= region_bytes == initial[key]
                    if key == "out0":
                        out0 = region_bytes
                base.require(out0 is not None, "skip out0 readback")
                if out0 != poison_full or not (guards_intact and immutable):
                    fail("skip launch changed output or guards")
                return dict(native=True, mode=mode, kernel=NAME, kernels_launched=launches,
                            readbacks=readbacks, counters_observed=observed,
                            counter_reset_value=RESET_FINAL, out0_changed=False,
                            all_guards_intact=guards_intact, immutable_regions_checked=list(initial),
                            negative_control_rejected=True,
                            matrix_permutation_recovered=False,
                            learned_weights_used=False, native_image_inference=False,
                            deliberate_proprietary_payload_files_written=False)

            # Phase 1: base capture, impulse bases, controls, signature runs.
            upload("out0", poison_full)
            base_region: bytes | None = None
            for entry in phase1_plan():
                record, out0_window, changed = run_entry(entry, base_region)
                if entry["name"] == "base":
                    full, full_intact = download("out0")
                    readbacks += 1
                    if not full_intact:
                        fail("base full-region guard violation")
                    if full[:EXTENT_BYTES] != bytes(EXTENT_BYTES):
                        first_nz = next(index for index in range(EXTENT_BYTES) if full[index] != 0)
                        fail(f"base output not zero in extent (first non-zero byte {first_nz})")
                    if full[EXTENT_BYTES:OUT0_WINDOW] != poison_window[EXTENT_BYTES:]:
                        fail("base writes inside the measurement-window margin")
                    if full[OUT0_WINDOW:] != poison_full[OUT0_WINDOW:]:
                        fail("base writes outside the measurement window")
                    base_region = full
                    continue
                if base_region is None:
                    fail("base region not captured")
                if entry["kind"] == "impulse-base" and changed:
                    fail(f"impulse-only leakage ({entry['name']})")
                if entry["name"] == "coupled-ref":
                    expected_coupled = {unit + byte for unit in canonical_units()
                                        for byte in UNIT_BYTES}
                    if changed != expected_coupled:
                        first_bad = sorted(changed ^ expected_coupled)[:8]
                        fail(f"coupled-ref footprint is not the exact canonical set: {first_bad}")
                if entry["name"] == "decoupled-ctrl" and changed:
                    fail("decoupling rejected: decoupled-ctrl changed the output")

            assert base_region is not None
            # Column decode per input from the signature runs.
            decode_by_input: dict[tuple[int, int, int], dict[int, int]] = {}
            for key in ALL_INPUTS:
                try:
                    presence: dict[int, set[int]] = {}
                    for record in records:
                        if record["kind"] != "signature":
                            continue
                        if (record["impulse_partition"], record["impulse_unit"],
                                record["impulse_lane"]) != key:
                            continue
                        bit = record["matrix_spec"][2]
                        presence[bit] = expand_ranges(record["changed_ranges"], OUT0_WINDOW)
                    base.require(len(presence) == SIGNATURE_BITS, "signature runs present for input")
                    decode_by_input[key] = decode_column(presence)
                except (ValueError, TypeError) as error:
                    fail(f"decode failed for {key_name(key)}: {error}")

            # Training: fit one effect per index bit from its single-bit column,
            # then require the fitted effect to reproduce its own column.
            effects: dict[int, tuple[int, int, int, int]] = {}
            training_records: list[dict[str, Any]] = []
            for bit in range(BIT_COUNT):
                key = bit_key(bit)
                decode = decode_by_input[key]
                try:
                    effect = fit_effect(decode)
                    effects[bit] = effect
                    # Compose with ONLY this bit set (all other effects zeroed).
                    single = {other: (0, 0, 0, 0) for other in range(BIT_COUNT)}
                    single[bit] = effect
                    self_map = signature_map(model_map(single, *key[1:]))
                    self_match = self_map == decode
                except (ValueError, TypeError) as error:
                    fail(f"training fit failed for bit {bit} ({key_name(key)}): {error}")
                training_records.append(dict(bit=bit, impulse=list(key),
                                             fitted_effect=list(effect),
                                             decode_entries=len(decode),
                                             matrix_byte_min=min(decode.values()),
                                             matrix_byte_max=max(decode.values()),
                                             self_prediction_match=bool(self_match)))
                if not self_match:
                    fail(f"training self-prediction mismatch for bit {bit} ({key_name(key)})")

            # Hold-out + identity: composed predictions must match exactly.
            # Composition is guarded so a model violation in a new way (the class
            # of event that produced the byte-shift discovery) retains evidence.
            holdout_records: list[dict[str, Any]] = []
            full_maps: dict[tuple[int, int, int], dict[int, int]] = {}
            try:
                for key in ALL_INPUTS:
                    full_maps[key] = model_map(effects, key[1], key[2])
            except ValueError as error:
                fail(f"composed model map failed: {error}; "
                     f"effects={ {bit: effects.get(bit) for bit in range(BIT_COUNT)} }")
            for kind, keys in (("holdout", HOLDOUT_INPUTS), ("identity", IDENTITY_INPUTS)):
                for key in keys:
                    predicted = signature_map(full_maps[key])
                    measured = decode_by_input[key]
                    match = predicted == measured
                    mismatches: list[list[Any]] = []
                    for offset in sorted(set(predicted) | set(measured)):
                        if predicted.get(offset) != measured.get(offset):
                            mismatches.append([offset, measured.get(offset), predicted.get(offset)])
                            if len(mismatches) >= 8:
                                break
                    holdout_records.append(dict(kind=kind, impulse=list(key),
                                                predicted_entries=len(predicted),
                                                measured_entries=len(measured),
                                                model_match=bool(match),
                                                mismatches=mismatches,
                                                predicted_zero_preimage=zero_preimage(full_maps[key])))
                    if not match:
                        fail(f"{kind} model mismatch for {key_name(key)}: {mismatches}")

            # Phase 2: direct checks derived from the fitted model.
            try:
                direct_plan = phase2_plan(full_maps)
            except ValueError as error:
                fail(f"direct plan derivation failed: {error}")
            for entry in direct_plan:
                record, _window, changed = run_entry(entry, base_region)
                if record["prediction_match"] is not True:
                    fail(f"direct prediction mismatch ({entry['name']}): "
                         f"expected {entry['expected']}, observed {sorted(changed)}")

            # Final full-region check: everything beyond the per-entry window is
            # still the run-start poison; guard zones intact.
            final_full, final_intact = download("out0")
            readbacks += 1
            if not final_intact:
                fail("final out0 guard violation")
            if final_full[OUT0_WINDOW:] != poison_full[OUT0_WINDOW:]:
                fail("out0 written outside the measurement window during the run")

        # Aggregate findings.
        column_records = []
        total_triples = 0
        for key in ALL_INPUTS:
            decode = decode_by_input[key]
            total_triples += len(decode)
            column_records.append(dict(impulse_partition=key[0], impulse_unit=key[1],
                                       impulse_lane=key[2],
                                       influenced_outputs=len(decode),
                                       distinct_matrix_bytes=len(set(decode.values())),
                                       matrix_byte_min=min(decode.values()),
                                       matrix_byte_max=max(decode.values()),
                                       decode=[[offset, matrix_byte]
                                               for offset, matrix_byte in sorted(decode.items())]))
        direct_records = [record for record in records if record["kind"] == "direct"]
        direct_ok = all(record["prediction_match"] is True for record in direct_records)
        holdout_ok = all(record["model_match"] for record in holdout_records)
        training_ok = all(record["self_prediction_match"] for record in training_records)
        recovered = bool(direct_ok and holdout_ok and training_ok)
        return dict(native=True, device="NVIDIA GeForce RTX 4070 Ti SUPER",
                    driver_api_version=version, mode=mode, kernel=NAME,
                    block=list(BLOCK), kernels_launched=launches, readbacks=readbacks,
                    signature_bits=SIGNATURE_BITS,
                    train_inputs=len(TRAIN_INPUTS), holdout_inputs=len(HOLDOUT_INPUTS),
                    identity_inputs=len(IDENTITY_INPUTS), spot_inputs=len(SPOT_INPUTS),
                    phase1_entries=len(phase1_plan()), phase2_entries=len(direct_records),
                    records=records, column_records=column_records,
                    training_records=training_records, holdout_records=holdout_records,
                    bit_effects={str(bit): list(effects[bit]) for bit in range(BIT_COUNT)},
                    extension_triples_total=total_triples,
                    training_self_prediction_passed=bool(training_ok),
                    holdout_model_prediction_passed=bool(holdout_ok),
                    direct_validation_passed=bool(direct_ok),
                    zero_preimage_checks=len([r for r in direct_records if r["sub"] == "zero"]),
                    impulse_leakage_detected=False,
                    coupled_ref_enforced=True, decoupling_enforced=True,
                    out0_full_region_final_check=True,
                    matrix_permutation_recovered=recovered,
                    permutation_scope=("validated on 21 of 4*4096 input bytes (12 single-bit "
                                       "fits, 8 held-out multi-bit exact map matches, 1 "
                                       "identity regression, 21 matrix-byte-0 preimage "
                                       "checks, 3 spot triples); closed form extrapolates "
                                       "over the remainder by additivity"),
                    learned_weights_used=False, native_image_inference=False,
                    deliberate_proprietary_payload_files_written=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("self-test", "execute", "skip"))
    parser.add_argument("dll", type=Path, nargs="?")
    args = parser.parse_args()
    if args.mode == "self-test":
        print(json.dumps(self_test(), indent=2))
        return
    base.require(args.dll is not None, "DLL required")
    policy = base.protect_process_memory()
    policy["disassembler_same_uid_proc_exposure_controlled"] = False
    data = base.read_bounded(args.dll)
    inventory = base.inventory(data)
    print(json.dumps(dict(dll_sha256=base.PIN, memory_policy=policy,
                          cpu_self_test=self_test(), cuda=execute(data, inventory, args.mode)),
                     indent=2))


if __name__ == "__main__":
    main()
