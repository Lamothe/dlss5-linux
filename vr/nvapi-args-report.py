#!/usr/bin/env python3
"""Analyse and validate an NvAPI launch-argument capture (DLSS_NVAPI_ARGS).

Parses a trace produced by vr/capture/nvapi-trace.c and derives the per-frame
kernel inventory: for every one of the 156 launches in an evaluation, the
kernel name (joined from the creation records), grid, block dimensions, shared
memory and parameter-block size. It verifies the structural claims the capture
was taken for and cross-checks the decoder launch contract recovered by the
native work (block 32,2,1 and an 80-byte parameter block) against the vendor's
own call.

Evidence discipline: this tool reports counts, names and dimensions only -
never parameter-block contents. The report JSON is retained locally
(gitignored), like the traces it is derived from.

Modes:
  --trace A.log              single-capture analysis
  --a A.log --b B.log        compare two captures (determinism / pair check)
  --self-test                run the corruption controls against fixtures

Verdicts: `inventory-consistent` (all pinned checks hold), `divergent` (a pin
failed), `malformed-evidence` (the trace could not be parsed), and
`identical-inventory` / `divergent` for pair mode. Exit code 0 only for the
passing verdicts.
"""
import argparse
import json
import re
import sys

# --- pinned expectations -------------------------------------------------
# The real capture (`evaluate` mode) is pinned to the observed run shape; the
# self-test uses a smaller fixture profile over the same checks.
PROFILES = {
    "evaluate": {
        "evaluations": 120,
        "per_eval": 156,
        "launch_calls": 18720,
        "create_calls": 97,
        "created": 96,
        "created_null": 1,
        "destroy_calls": 96,
    },
    "fixture": {
        "evaluations": 2,
        "per_eval": 156,
        "launch_calls": 312,
        "create_calls": 5,
        "created": 4,
        "created_null": 1,
        "destroy_calls": 0,
    },
}
# The decoder launch contract recovered natively in vr/NATIVE-DECODER-LAUNCH.md
# and vr/NATIVE-DECODER-BINDING.md, to be confirmed against the live call.
DECODER_KERNEL = "cc_dec_input_upsample_1024_512_tilesync_fp8"
DECODER_BLOCK = (32, 2, 1)
DECODER_PSIZE = 80
MAX_GRID_DIM = 65536
ARGS_PMEM_MAX = 512  # must match ARGS_PMEM_MAX in nvapi-trace.c
MAX_PSIZE = 4096

LINE = re.compile(r"^(?P<seq>\d+) (?P<tick>\d+) (?P<tid>\d+) (?P<rest>.*)$")
HEX = re.compile(r"^0x[0-9a-f]+$")

LAUNCH_RE = re.compile(
    r"^launch id=(?P<id>0x[0-9a-f]+) cmd=(?P<cmd>0x[0-9a-f]+) "
    r"handle=(?P<handle>0x[0-9a-f]+) grid=(?P<grid>\d+,\d+,\d+) "
    r"params=(?P<params>0x[0-9a-f]+) psize=(?P<psize>\d+) "
    r"pn=(?P<pn>\d+) pcap=(?P<pcap>\d+) pmem=(?P<pmem>[0-9a-f]*) s=(?P<slots>.+)$"
)
CREATED_RE = re.compile(
    r"^created struct=(?P<struct>0x[0-9a-f]+) struct_size=(?P<struct_size>\d+) "
    r"device=(?P<device>0x[0-9a-f]+) cubin=(?P<cubin>0x[0-9a-f]+) "
    r"cubin_size=(?P<cubin_size>\d+) block=(?P<block>\d+,\d+,\d+) "
    r"smem=(?P<smem>\d+) flags=(?P<flags>\d+) name=(?P<name>\S+) "
    r"handle=(?P<handle>0x[0-9a-f]+)$"
)
CREATEARGS_RE = re.compile(
    r"^createargs id=(?P<id>0x[0-9a-f]+) struct=(?P<struct>0x[0-9a-f]+) "
    r"s=(?P<slots>.+)$"
)
CALL_RE = re.compile(r"^call id=(?P<id>0x[0-9a-f]+) ret=(?P<ret>[0-9A-Fa-f]+) site=\S+$")
RESULT_RE = re.compile(r"^result id=(?P<id>0x[0-9a-f]+) ptr=(?P<ptr>[0-9A-Fa-f]+)$")
RESOLVE_RE = re.compile(
    r"^resolve id=(?P<id>0x[0-9a-f]+) ret=(?P<ret>[0-9A-Fa-f]+) site=\S+$"
)
INIT_RE = re.compile(r"^init real=\S+ wrap=\d+( args=\d+)?$")
SLOTS_RE = re.compile(r"^(0x[0-9a-f]{16})(,0x[0-9a-f]{16})*$")


class EvidenceError(Exception):
    """The trace is not parseable evidence; nothing may be claimed from it."""


def parse_trace(path):
    """Strictly parse a trace. Raises EvidenceError on any malformed line."""
    launches = []
    created = {}  # handle -> record
    order = []  # (kind, payload) in trace order
    counts = {}
    call_ids = {}
    tids = set()
    created_null = 0
    prev_seq = None
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except OSError as exc:
        raise EvidenceError(f"unreadable trace: {exc}") from exc

    for lineno, raw in enumerate(text.splitlines(), 1):
        if not raw:
            raise EvidenceError(f"line {lineno}: empty line in trace")
        m = LINE.match(raw)
        if not m:
            raise EvidenceError(f"line {lineno}: unprefixed line")
        seq = int(m.group("seq"))
        if prev_seq is not None and seq != prev_seq + 1:
            raise EvidenceError(
                f"line {lineno}: sequence gap ({prev_seq} -> {seq})"
            )
        prev_seq = seq
        rest = m.group("rest")
        if rest.endswith(" trunc=1"):
            # The tracer truncates rather than drops an over-long record; the
            # missing tail must fail loudly, never pass as complete evidence.
            raise EvidenceError(f"line {lineno}: truncated record")
        kind = rest.split(" ", 1)[0]
        counts[kind] = counts.get(kind, 0) + 1
        if kind in ("launch", "createargs", "created", "created-null"):
            tids.add(int(m.group("tid")))

        if kind == "launch":
            lm = LAUNCH_RE.match(rest)
            if not lm:
                raise EvidenceError(f"line {lineno}: malformed launch record")
            grid = tuple(int(v) for v in lm.group("grid").split(","))
            if any(v < 1 or v > MAX_GRID_DIM for v in grid):
                raise EvidenceError(f"line {lineno}: grid out of range {grid}")
            psize = int(lm.group("psize"))
            if psize < 1 or psize > MAX_PSIZE:
                raise EvidenceError(f"line {lineno}: psize out of range {psize}")
            pn = int(lm.group("pn"))
            pcap = int(lm.group("pcap"))
            pmem = lm.group("pmem")
            # pcap: 0 = complete dump, 1 = capped at 512, 2 = short read,
            # 3 = no dump requested (ARGS level 1).
            if pcap not in (0, 1, 2, 3):
                raise EvidenceError(f"line {lineno}: unknown pcap value")
            if len(pmem) % 2 != 0:
                raise EvidenceError(f"line {lineno}: pmem hex not byte-aligned")
            if pn != len(pmem) // 2:
                raise EvidenceError(f"line {lineno}: pn does not match pmem length")
            # The dump is exactly min(psize, 512) bytes unless a short read
            # (pcap=2) was reported; these must all agree or the record lies.
            if pcap == 3 and (pn != 0 or pmem != ""):
                raise EvidenceError(f"line {lineno}: pcap=3 but a dump is present")
            if pcap == 0 and (psize > ARGS_PMEM_MAX or pn != psize):
                raise EvidenceError(f"line {lineno}: pn inconsistent with psize")
            if pcap == 1 and (pn != ARGS_PMEM_MAX or psize <= ARGS_PMEM_MAX):
                raise EvidenceError(f"line {lineno}: pcap=1 inconsistent with psize")
            if pcap == 2 and pn >= min(psize, ARGS_PMEM_MAX):
                raise EvidenceError(f"line {lineno}: pcap=2 inconsistent with pn")
            if not SLOTS_RE.match(lm.group("slots")):
                raise EvidenceError(f"line {lineno}: malformed slot array")
            rec = {
                "handle": int(lm.group("handle"), 16),
                "cmd": int(lm.group("cmd"), 16),
                "grid": grid,
                "params": int(lm.group("params"), 16),
                "psize": psize,
                "pn": pn,
                "pcap": pcap,
            }
            launches.append(rec)
            order.append(("launch", rec))
        elif kind == "created":
            cm = CREATED_RE.match(rest)
            if not cm:
                raise EvidenceError(f"line {lineno}: malformed created record")
            handle = int(cm.group("handle"), 16)
            if handle == 0:
                raise EvidenceError(f"line {lineno}: created record with null handle")
            if handle in created:
                raise EvidenceError(f"line {lineno}: duplicate created handle")
            rec = {
                "struct_size": int(cm.group("struct_size")),
                "block": tuple(int(v) for v in cm.group("block").split(",")),
                "smem": int(cm.group("smem")),
                "flags": int(cm.group("flags")),
                "name": cm.group("name"),
                "handle": handle,
                "cubin_size": int(cm.group("cubin_size")),
            }
            created[handle] = rec
            order.append(("created", rec))
        elif kind == "created-null":
            if rest != "created-null":
                raise EvidenceError(f"line {lineno}: malformed created-null record")
            created_null += 1
            order.append(("created-null", None))
        elif kind == "created-invalid":
            raise EvidenceError(f"line {lineno}: created-invalid in evidence")
        elif kind == "createargs":
            am = CREATEARGS_RE.match(rest)
            if not am:
                raise EvidenceError(f"line {lineno}: malformed createargs record")
            if not SLOTS_RE.match(am.group("slots")):
                raise EvidenceError(f"line {lineno}: malformed slot array")
            order.append(("createargs", {"struct": int(am.group("struct"), 16)}))
        elif kind == "call":
            cm = CALL_RE.match(rest)
            if not cm:
                raise EvidenceError(f"line {lineno}: malformed call record")
            cid = int(cm.group("id"), 16)
            call_ids[cid] = call_ids.get(cid, 0) + 1
            order.append(("call", None))
        elif kind == "result":
            if not RESULT_RE.match(rest):
                raise EvidenceError(f"line {lineno}: malformed result record")
            order.append(("result", None))
        elif kind == "resolve":
            if not RESOLVE_RE.match(rest):
                raise EvidenceError(f"line {lineno}: malformed resolve record")
            order.append(("resolve", None))
        elif kind == "init":
            if not INIT_RE.match(rest):
                raise EvidenceError(f"line {lineno}: malformed init record")
            order.append(("init", None))
        elif kind in ("wrap-mismatch", "wrap-exhausted", "wrap-failed"):
            # Diagnostics emitted by the forwarder; not failures of the capture.
            order.append((kind, None))
        else:
            raise EvidenceError(f"line {lineno}: unknown record kind {kind!r}")

    return {
        "launches": launches,
        "created": created,
        "created_null": created_null,
        "counts": counts,
        "call_ids": call_ids,
        "tids": tids,
        "order": order,
    }


def fnv1a64(data: bytes) -> str:
    value = 14695981039346656037
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def slot_sequence(parsed, per_eval):
    """Split the launch stream into complete evaluations.

    Any trailing partial evaluation is ignored here; the launch-count and
    evaluation-count checks report the mismatch, so a dropped or duplicated
    launch is rejected as a pin failure rather than a parse failure.
    """
    launches = parsed["launches"]
    complete = len(launches) // per_eval
    return [launches[i * per_eval : (i + 1) * per_eval] for i in range(complete)]


def analyse(parsed, profile):
    """Return (report, checks) where checks maps check name -> (ok, detail)."""
    expect = PROFILES[profile]
    checks = {}
    counts = parsed["counts"]
    created = parsed["created"]
    launches = parsed["launches"]

    checks["create-call-count"] = (
        counts.get("createargs", 0) == expect["create_calls"],
        f"{counts.get('createargs', 0)} createargs records "
        f"(expected {expect['create_calls']})",
    )
    checks["created-count"] = (
        len(created) == expect["created"],
        f"{len(created)} kernels created (expected {expect['created']})",
    )
    checks["created-null-count"] = (
        parsed["created_null"] == expect["created_null"],
        f"{parsed['created_null']} null-parameter creates",
    )
    checks["struct-size"] = (
        all(r["struct_size"] == 80 for r in created.values()),
        "every create struct reported 80 bytes",
    )
    call_ids = parsed["call_ids"]
    checks["call-id-crosscheck"] = (
        call_ids.get(0x5C52BB86, 0) == expect["launch_calls"]
        and call_ids.get(0x299F5FDC, 0) == expect["create_calls"]
        and call_ids.get(0x7FB785BA, 0) == expect["destroy_calls"],
        f"call records: launch={call_ids.get(0x5C52BB86, 0)} "
        f"create={call_ids.get(0x299F5FDC, 0)} "
        f"destroy={call_ids.get(0x7FB785BA, 0)}",
    )
    checks["single-thread"] = (
        len(parsed["tids"]) == 1,
        f"captured calls on {len(parsed['tids'])} thread(s)",
    )

    evals = slot_sequence(parsed, expect["per_eval"])
    checks["launch-count"] = (
        len(launches) == expect["launch_calls"],
        f"{len(launches)} launches (expected {expect['launch_calls']})",
    )
    checks["evaluation-count"] = (
        len(evals) == expect["evaluations"],
        f"{len(evals)} evaluations of {expect['per_eval']} launches",
    )
    checks["launch-handle-mapped"] = (
        all(l["handle"] in created for l in launches),
        "every launch handle maps to a created kernel",
    )

    # Per-slot stability across evaluations: the pipeline is a fixed sequence.
    first = evals[0] if evals else []
    for field in ("handle", "grid", "psize"):
        ok = all(
            len({e[i][field] for e in evals}) == 1 for i in range(len(first))
        )
        checks[f"slot-{field}-stable"] = (ok, f"{field} constant per slot")
    def kernel_of(launch):
        return created.get(launch["handle"], {"name": "<unknown>", "block": ()})

    ok = all(
        len({kernel_of(e[i])["name"] for e in evals}) == 1
        for i in range(len(first))
    )
    checks["slot-name-stable"] = (ok, "kernel name constant per slot")
    ok = all(
        len({kernel_of(e[i])["block"] for e in evals}) == 1
        for i in range(len(first))
    )
    checks["slot-block-stable"] = (ok, "block dims constant per slot")

    # The decoder contract recovered natively in stage 5.
    decoder_slots = [
        i
        for i, l in enumerate(first)
        if kernel_of(l).get("name") == DECODER_KERNEL
    ]
    checks["decoder-present"] = (
        len(decoder_slots) >= 1,
        f"{DECODER_KERNEL} appears at slot(s) {decoder_slots}",
    )
    if decoder_slots:
        slot = decoder_slots[0]
        block = kernel_of(first[slot])["block"]
        checks["decoder-block"] = (
            block == DECODER_BLOCK,
            f"block dims {block} (expected {DECODER_BLOCK})",
        )
        checks["decoder-psize"] = (
            first[slot]["psize"] == DECODER_PSIZE,
            f"parameter block {first[slot]['psize']} bytes (expected {DECODER_PSIZE})",
        )
    else:
        checks["decoder-block"] = (False, "decoder kernel not launched")
        checks["decoder-psize"] = (False, "decoder kernel not launched")

    summary = []
    for i, l in enumerate(first):
        rec = kernel_of(l)
        summary.append(
            {
                "slot": i,
                "name": rec.get("name", "<unknown>"),
                "grid": list(l["grid"]),
                "block": list(rec.get("block", ())),
                "smem": rec.get("smem"),
                "psize": l["psize"],
                "handle": f"0x{l['handle']:x}",
            }
        )

    report = {
        "mode": profile,
        "counts": counts,
        "created": len(created),
        "launches": len(launches),
        "evaluations": len(evals),
        "kernels": sorted(r["name"] for r in created.values()),
        "launched_kernels": sorted({kernel_of(l)["name"] for l in launches}),
        "slots": summary,
    }
    return report, checks


def verdict_for(checks, report):
    failed = [name for name, (ok, _) in checks.items() if not ok]
    report["checks"] = {name: {"ok": ok, "detail": detail} for name, (ok, detail) in checks.items()}
    report["failed_checks"] = failed
    report["verdict"] = "inventory-consistent" if not failed else "divergent"
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=str)
    parser.add_argument("--a", type=str)
    parser.add_argument("--b", type=str)
    parser.add_argument("--report", type=str)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="evaluate")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    if args.a and args.b:
        return pair_mode(args)
    if not args.trace:
        parser.error("need --trace, or --a/--b, or --self-test")
    try:
        parsed = parse_trace(args.trace)
        report, checks = analyse(parsed, args.profile)
        report = verdict_for(checks, report)
    except EvidenceError as exc:
        report = {"verdict": "malformed-evidence", "error": str(exc)}
    emit(report, args.report)
    return 0 if report["verdict"] == "inventory-consistent" else 1


def pair_mode(args):
    """Compare two captures slot by slot (determinism check)."""
    profile = getattr(args, "profile", "evaluate")
    try:
        pa = parse_trace(args.a)
        pb = parse_trace(args.b)
        ra, ca = analyse(pa, profile)
        rb, cb = analyse(pb, profile)
        ra = verdict_for(ca, ra)
        rb = verdict_for(cb, rb)
        slots_a, slots_b = ra["slots"], rb["slots"]
        same = len(slots_a) == len(slots_b) and all(
            (x["name"], tuple(x["grid"]), tuple(x["block"]), x["psize"], x["smem"])
            == (y["name"], tuple(y["grid"]), tuple(y["block"]), y["psize"], y["smem"])
            for x, y in zip(slots_a, slots_b)
        )
        handles_same = [x["handle"] for x in slots_a] == [y["handle"] for y in slots_b]
        report = {
            "mode": "pair",
            "a": args.a,
            "b": args.b,
            "verdict_a": ra["verdict"],
            "verdict_b": rb["verdict"],
            "slot_sequences_identical": same,
            "handle_values_identical": handles_same,
            "failed_checks_a": ra["failed_checks"],
            "failed_checks_b": rb["failed_checks"],
        }
        if ra["verdict"] != "inventory-consistent" or rb["verdict"] != "inventory-consistent":
            report["verdict"] = "divergent"
        else:
            report["verdict"] = "identical-inventory" if same else "divergent"
    except EvidenceError as exc:
        report = {"verdict": "malformed-evidence", "error": str(exc)}
    emit(report, args.report)
    return 0 if report["verdict"] == "identical-inventory" else 1


def emit(report, path):
    text = json.dumps(report, indent=2)
    if path:
        open(path, "w", encoding="utf-8").write(text + "\n")
    print(text)


# --- corruption controls --------------------------------------------------
def make_fixture(evals=2, per_eval=156):
    """Synthesise a well-formed trace satisfying the fixture profile."""
    kernels = [
        ("cc_cb_clear", (256, 1, 1), 16),
        (DECODER_KERNEL, DECODER_BLOCK, DECODER_PSIZE),
        ("cc_split_swin_16h_qkv_512_chained_fp8", (32, 4, 1), 56),
        ("cg2r_copy_kernel", (16, 16, 1), 72),
    ]
    handles = [0x22B0000 + i * 0x10 for i in range(len(kernels))]
    lines = []
    seq = 0

    def emit_line(text):
        nonlocal seq
        lines.append(f"{seq:06d} 1000 352 {text}")
        seq += 1

    emit_line("init real=00006FFFFBB77DF0 wrap=1 args=2")
    slots = ",".join(f"0x{i:016x}" for i in range(22))
    for (name, block, psize), handle in zip(kernels, handles):
        emit_line("call id=0x299f5fdc ret=00006FFFF14E1FA3 site=nvngx_dlssnr.dll+0x1fa3")
        emit_line(f"createargs id=0x299f5fdc struct=0x000000000111e850 s={slots}")
        emit_line(
            f"created struct=0x000000000111e850 struct_size=80 device=0x0000000002390080 "
            f"cubin=0x00006ffff15c06e0 cubin_size=1000 block={block[0]},{block[1]},{block[2]} "
            f"smem=0 flags=0 name={name} handle=0x{handle:016x}"
        )
    emit_line("call id=0x299f5fdc ret=00006FFFF14E1FA3 site=nvngx_dlssnr.dll+0x1fa3")
    emit_line(f"createargs id=0x299f5fdc struct=0x0000000000000000 s={slots}")
    emit_line("created-null")
    for _ in range(evals):
        for slot in range(per_eval):
            name, block, psize = kernels[slot % len(kernels)]
            handle = handles[slot % len(kernels)]
            pmem = "ab" * psize
            emit_line("call id=0x5c52bb86 ret=00006FFFF14E2406 site=nvngx_dlssnr.dll+0x2406")
            emit_line(
                f"launch id=0x5c52bb86 cmd=0x0000000002316b60 handle=0x{handle:016x} "
                f"grid=8,3,1 params=0x{0x1000 + slot:016x} psize={psize} "
                f"pn={psize} pcap=0 pmem={pmem} s={slots}"
            )
    return lines


def renumber(lines):
    for i, line in enumerate(lines):
        parts = line.split(" ", 3)
        parts[0] = f"{i:06d}"
        lines[i] = " ".join(parts)


def run_selftest_case(mutate, expect_ok, expect_substr, do_renumber=True):
    lines = make_fixture()
    if mutate:
        mutate(lines)
    if do_renumber:
        renumber(lines)
    import os
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
        fh.write("\n".join(lines) + "\n")
        path = fh.name
    try:
        parsed = parse_trace(path)
        report, checks = analyse(parsed, "fixture")
        report = verdict_for(checks, report)
        verdict = report["verdict"]
        detail = "; ".join(report["failed_checks"])
        ok = (verdict == "inventory-consistent") == expect_ok
        if expect_substr:
            ok = ok and expect_substr in detail
    except EvidenceError as exc:
        detail = str(exc)
        verdict = "malformed-evidence"
        ok = (not expect_ok) and (expect_substr in detail if expect_substr else True)
    finally:
        os.unlink(path)
    return ok, verdict, detail


def self_test() -> int:
    def first_index(lines, needle):
        for i, line in enumerate(lines):
            if needle in line:
                return i
        raise AssertionError(f"fixture has no line matching {needle!r}")

    def drop_first_launch(lines):
        del lines[first_index(lines, " launch id=")]

    def drop_last_launch(lines):
        for i in range(len(lines) - 1, -1, -1):
            if " launch id=" in lines[i]:
                del lines[i]
                return

    def duplicate_launch(lines):
        i = first_index(lines, " launch id=")
        lines.insert(i + 1, lines[i])

    def break_seq(lines):
        del lines[first_index(lines, " launch id=") - 1]

    def unknown_kind(lines):
        lines.append("999999 1000 352 mystery id=0x1")

    def mutate_grid(lines):
        i = first_index(lines, " launch id=")
        lines[i] = re.sub(r"grid=\d+,\d+,\d+", "grid=9999,3,1", lines[i])

    def mutate_psize(lines):
        """Forge a launch's parameter size consistently across all four fields."""
        i = first_index(lines, " launch id=")
        line = lines[i]
        line = re.sub(r"psize=\d+", "psize=64", line)
        line = re.sub(r"pn=\d+", "pn=64", line)
        line = re.sub(r"pmem=[0-9a-f]*", "pmem=" + "ab" * 64, line)
        lines[i] = line

    def mutate_handle(lines):
        i = first_index(lines, " launch id=")
        lines[i] = re.sub(r"handle=0x[0-9a-f]+", "handle=0x0000000000deadc0", lines[i])

    def mutate_decoder_block(lines):
        i = first_index(lines, f"name={DECODER_KERNEL}")
        lines[i] = lines[i].replace("block=32,2,1", "block=32,4,1")

    def mutate_struct_size(lines):
        i = first_index(lines, " created struct=")
        lines[i] = lines[i].replace("struct_size=80", "struct_size=72")

    def mutate_null_handle(lines):
        i = first_index(lines, " created struct=")
        lines[i] = re.sub(r"handle=0x[0-9a-f]+", "handle=0x0000000000000000", lines[i])

    def mutate_decoder_name(lines):
        i = first_index(lines, f"name={DECODER_KERNEL}")
        lines[i] = re.sub(r"name=\S+", "name=cc_renamed_kernel", lines[i])

    def mutate_malformed_launch(lines):
        i = first_index(lines, " launch id=")
        lines[i] = lines[i].replace("psize=", "psizee=")

    def mutate_pmem_odd(lines):
        i = first_index(lines, " launch id=")
        lines[i] = re.sub(r"pmem=[0-9a-f]*", "pmem=abc", lines[i])

    def mutate_truncated(lines):
        i = first_index(lines, " launch id=")
        lines[i] = lines[i][: len(lines[i]) - 20] + " trunc=1"

    def mutate_pcap3_with_dump(lines):
        """Claim 'no dump requested' while keeping the parameter bytes."""
        i = first_index(lines, " launch id=")
        lines[i] = re.sub(r"pcap=\d+", "pcap=3", lines[i])

    def mutate_psize_range(lines):
        i = first_index(lines, " launch id=")
        lines[i] = re.sub(r"psize=\d+", "psize=99999", lines[i])

    def mutate_orphan_launch(lines):
        """Point one launch at a handle no create record owns."""
        i = first_index(lines, " launch id=")
        lines[i] = re.sub(r"handle=0x[0-9a-f]+", "handle=0x0000000000f00d00", lines[i])

    cases = [
        ("clean baseline", None, True, None, True),
        ("launch dropped", drop_first_launch, False, "launch-count", True),
        ("launch duplicated", duplicate_launch, False, "launch-count", True),
        ("sequence gap", break_seq, False, "sequence gap", False),
        ("unknown record kind", unknown_kind, False, "unknown record kind", True),
        ("grid changed in one slot", mutate_grid, False, "slot-grid-stable", True),
        ("psize changed in one slot", mutate_psize, False, "slot-psize-stable", True),
        ("handle changed in one slot", mutate_handle, False, "slot-handle-stable", True),
        ("decoder block dims altered", mutate_decoder_block, False, "decoder-block", True),
        ("create struct size altered", mutate_struct_size, False, "struct-size", True),
        ("created handle null", mutate_null_handle, False, "null handle", True),
        ("decoder kernel renamed", mutate_decoder_name, False, "decoder-present", True),
        ("launch dropped from tail", drop_last_launch, False, "launch-count", True),
        ("launch handle not created", mutate_orphan_launch, False, "launch-handle-mapped", True),
        ("malformed launch record", mutate_malformed_launch, False, "malformed launch", True),
        ("pmem hex not aligned", mutate_pmem_odd, False, "pmem hex", True),
        ("psize out of range", mutate_psize_range, False, "psize out of range", True),
        ("pcap=3 with a dump present", mutate_pcap3_with_dump, False, "pcap=3", True),
        ("truncated record", mutate_truncated, False, "truncated record", True),
    ]

    failures = []
    for name, mutate, expect_ok, expect_substr, do_renumber in cases:
        ok, verdict, detail = run_selftest_case(mutate, expect_ok, expect_substr, do_renumber)
        print(f"{'PASS' if ok else 'FAIL'} {name}: verdict={verdict} {detail}")
        if not ok:
            failures.append(name)

    # Pair-mode controls: a rename is invisible inside one capture but must
    # diverge when compared against a run whose kernel set differs.
    import os
    import tempfile

    def write_fixture(lines):
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
            fh.write("\n".join(lines) + "\n")
            return fh.name

    base = make_fixture()
    other = make_fixture()
    for i, line in enumerate(other):
        if " created struct=" in line:
            other[i] = re.sub(r"name=cc_cb_clear", "name=cc_renamed_kernel", line)
            break
    path_a, path_b = write_fixture(base), write_fixture(other)
    try:
        rc = pair_mode(argparse.Namespace(a=path_a, b=path_b, report=None, profile="fixture"))
        pair_divergent = rc == 1
        print(f"{'PASS' if pair_divergent else 'FAIL'} pair detects renamed kernel: rc={rc}")
        if not pair_divergent:
            failures.append("pair-rename")
        rc = pair_mode(argparse.Namespace(a=path_a, b=path_a, report=None, profile="fixture"))
        pair_same = rc == 0
        print(f"{'PASS' if pair_same else 'FAIL'} pair accepts identical captures: rc={rc}")
        if not pair_same:
            failures.append("pair-identical")
    finally:
        os.unlink(path_a)
        os.unlink(path_b)

    print(f"self-test: {len(cases) + 2 - len(failures)}/{len(cases) + 2} controls behaved as expected")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
