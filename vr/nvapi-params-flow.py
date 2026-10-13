#!/usr/bin/env python3
"""Decode level-2 NvAPI parameter dumps into buffer bindings (stage-2 input).

Parses traces produced by vr/capture/nvapi-trace.c under DLSS_NVAPI_ARGS=2 and
turns the 156 per-evaluation parameter blocks into a buffer-binding reference
graph: which device buffers each launch slot references, which slots alias the
same buffer, and how the buffers rotate across the 120 captured evaluations.

Evidence discipline (same as vr/nvapi-args-report.py): this tool reports
counts, dimensions, deltas and pseudonymous binding ids (FNV-1a-64 over each
binding's per-evaluation value sequence) - never raw parameter-block bytes.
The report JSON stays local (gitignored), like the traces it is derived from.

Word typing is mechanical and pinned, not semantic: `two-addr` is a word
position carrying exactly two distinct 64-bit values across the evaluation
sequence, both at least ADDR_MIN, both below ADDR_MAX and both ADDR_ALIGN-
aligned; the `half-delta` check then pins their difference to HALF_DELTA (the
observed two-half allocation geometry). `addr-candidate` is a single
address-shaped value (ambiguous by design); `addr-multi` is three or more
address-shaped values (never observed; the half-delta check rejects it).
Everything else is `zero` or `scalar`.

Bindings are grouped by exact per-evaluation value sequence, not by value set:
two positions alias only if they carry the same buffer in the same evaluation,
whatever the rotation phase. Read/write direction is NOT inferred here - the
trace records references, not access mode. Direction claims live in the stage
documents where disassembly or kernel semantics provide them.

Modes:
  --trace A.log              single-capture binding analysis
  --a A.log --b B.log        compare two captures (allocator determinism)
  --self-test                corruption controls against generated fixtures

Verdicts: `bindings-decoded` (all pinned checks hold), `divergent` (a pin
failed), `malformed-evidence` (the trace could not be parsed); pair mode adds
`identical-bindings`. Exit code 0 only for the passing verdicts.
"""
import argparse
import json
import re
import sys

# --- pinned expectations -------------------------------------------------
PROFILES = {
    "evaluate": {"evaluations": 120, "per_eval": 156, "launch_calls": 18720},
    "fixture": {"evaluations": 2, "per_eval": 156, "launch_calls": 312},
}

HALF_DELTA = 0x1CD00000  # observed gap between the two address halves
ADDR_MIN = 0x01000000
ADDR_MAX = 1 << 32  # every observed device address fits in 32 bits
ADDR_ALIGN = 512
ARGS_PMEM_MAX = 512  # must match ARGS_PMEM_MAX in nvapi-trace.c

# Decoder slot contract (vr/NATIVE-DECODER-IO.md constant-bank roles and
# vr/NATIVE-DECODER-LAUNCH.md 80-byte bank), confirmed against the live call.
DECODER_SLOT = 99
DECODER_PSIZE = 80
DECODER_PTR_WORDS = 8  # argument offsets 0..64 are pointer-forming slots
DECODER_COUNTER_PAD = 512  # vr/NATIVE-DECODER-LAYOUT.md counter padding

LAUNCH_RE = re.compile(
    r"handle=(0x[0-9a-f]+) grid=([0-9,]+) params=(0x[0-9a-f]+) "
    r"psize=(\d+) pn=(\d+) pcap=(\d+) pmem=([0-9a-f]*)"
)


def fnv64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def binding_id(sequence) -> str:
    """Pseudonymous label for a per-evaluation value sequence."""
    blob = b"".join(int(v).to_bytes(8, "little") for v in sequence)
    return "b%016x" % fnv64(blob)


def sequence_pattern(sequence):
    """AB rotation string over the sequence, or `constant`."""
    vals = sorted(set(sequence))
    if len(vals) != 2:
        return "constant"
    return "".join("AB"[vals.index(v)] for v in sequence)


def parse_lines(lines, source="trace"):
    """Parse launch records from trace lines; raise ValueError if malformed."""
    launches = []
    for lineno, line in enumerate(lines, 1):
        if " launch " not in line:
            continue
        m = LAUNCH_RE.search(line)
        if not m:
            raise ValueError(f"{source}:{lineno}: unparsable launch record")
        handle, grid, params, psize, pn, pcap, pmem = m.groups()
        try:
            blob = bytes.fromhex(pmem)
        except ValueError as exc:
            raise ValueError(f"{source}:{lineno}: bad parameter hex") from exc
        launches.append({
            "handle": handle,
            "grid": tuple(int(x) for x in grid.split(",")),
            "psize": int(psize),
            "pn": int(pn),
            "pcap": int(pcap),
            "pmem": blob,
        })
    if not launches:
        raise ValueError(f"{source}: no launch records")
    return launches


def parse_trace(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return parse_lines(f.read().splitlines(), source=path)
    except OSError as exc:
        raise ValueError(f"{path}: unreadable evidence ({exc.strerror})") from exc


def word_type(values):
    """Classify one word position by its value sequence across evaluations."""
    s = set(values)
    if s == {0}:
        return "zero"
    if all(ADDR_MIN <= v < ADDR_MAX and v % ADDR_ALIGN == 0 for v in s):
        return {1: "addr-candidate", 2: "two-addr"}.get(len(s), "addr-multi")
    return "scalar"


def analyse(launches, profile):
    """Pinned structural analysis over one capture."""
    checks = []

    def check(name, ok, detail=""):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        return ok

    per_eval = profile["per_eval"]
    n_eval = profile["evaluations"]

    check(
        "shape",
        len(launches) == profile["launch_calls"] and len(launches) == n_eval * per_eval,
        f"launches={len(launches)} expected={profile['launch_calls']}",
    )
    complete = all(
        l["pcap"] == 0 and l["pn"] == l["psize"] == len(l["pmem"]) for l in launches
    )
    check("complete-dumps", complete, "every dump full length, pcap=0")
    check(
        "param-bounds",
        all(0 < l["psize"] <= ARGS_PMEM_MAX for l in launches),
        f"0 < psize <= {ARGS_PMEM_MAX}",
    )
    check(
        "word-shape",
        all(l["psize"] % 8 == 0 for l in launches),
        "all parameter blocks are whole 64-bit words",
    )

    def report(chk, wds=None, tys=None, binds=None):
        flat = [t for row in (tys or []) for t in row]
        entries = {}
        for seq, b in sorted((binds or {}).items()):
            entries[binding_id(seq)] = {
                "type": b["type"],
                "pattern": sequence_pattern(seq),
                "refs": b["refs"],
            }
        patterns = sorted({e["pattern"] for e in entries.values()})
        return {
            "mode": "single",
            "profile": profile,
            "checks": chk,
            "failed_checks": [c["name"] for c in chk if not c["ok"]],
            "summary": {
                "launches": len(launches),
                "evaluations": n_eval,
                "per_eval": per_eval,
                "words_total": sum(len(w) for ev in (wds or []) for w in ev),
                "type_counts": {t: flat.count(t) for t in sorted(set(flat))},
                "bindings": len(entries),
                "shared_bindings": sum(
                    1 for e in entries.values() if len(e["refs"]) > 1
                ),
                "rotation_patterns": patterns,
            },
            "bindings": entries,
        }

    # From here on the analysis indexes evaluations positionally; refuse to do
    # so over a misshapen capture and report the shape failure instead.
    if not checks[0]["ok"]:
        return report(checks)

    evals = [launches[i * per_eval:(i + 1) * per_eval] for i in range(n_eval)]
    words = [  # words[e][slot][widx]
        [[int.from_bytes(l["pmem"][i:i + 8], "little")
          for i in range(0, len(l["pmem"]), 8)]
         for l in ev]
        for ev in evals
    ]

    # Per-slot word typing over the minimum word count of the slot (ragged
    # evidence must fail `slot-structure-stable`, never crash), and structure
    # stability across evaluations.
    types = []  # types[slot][widx]
    stable = True
    for slot in range(per_eval):
        counts = [len(words[e][slot]) for e in range(n_eval)]
        if min(counts) != max(counts):
            stable = False
        types.append([
            word_type([words[e][slot][widx] for e in range(n_eval)])
            for widx in range(min(counts))
        ])
    check("slot-structure-stable", stable, "word counts fixed per slot")

    # half-delta: every two-addr position must differ by exactly HALF_DELTA,
    # and no position may rotate over three or more address-shaped values.
    two_addr = [
        (slot, widx)
        for slot in range(per_eval)
        for widx in range(len(types[slot]))
        if types[slot][widx] == "two-addr"
    ]
    multi = [
        (slot, widx)
        for slot in range(per_eval)
        for widx in range(len(types[slot]))
        if types[slot][widx] == "addr-multi"
    ]
    deltas_ok = all(
        max(words[e][slot][widx] for e in range(n_eval))
        - min(words[e][slot][widx] for e in range(n_eval)) == HALF_DELTA
        for slot, widx in two_addr
    )
    check(
        "half-delta",
        len(two_addr) > 0 and deltas_ok and not multi,
        f"two-addr positions={len(two_addr)} addr-multi={len(multi)}",
    )

    # Binding graph over non-zero words: alias = identical value sequence.
    bindings = {}  # seq -> {"type": ..., "refs": [[slot, widx], ...]}
    for slot in range(per_eval):
        for widx in range(len(types[slot])):
            if types[slot][widx] == "zero":
                continue
            seq = tuple(words[e][slot][widx] for e in range(n_eval))
            bindings.setdefault(seq, {"type": types[slot][widx], "refs": []})["refs"].append(
                [slot, widx]
            )

    # Decoder slot contract, pinned over every evaluation.
    if per_eval > DECODER_SLOT:
        dwords = [words[e][DECODER_SLOT] for e in range(n_eval)]
        psizes = [evals[e][DECODER_SLOT]["psize"] for e in range(n_eval)]
        enough = all(len(w) >= DECODER_PTR_WORDS + 2 for w in dwords)
        check(
            "decoder-psize",
            all(p == DECODER_PSIZE for p in psizes),
            f"psize={psizes[0]} (all evaluations)",
        )
        check(
            "decoder-pointer-words",
            enough
            and all(
                types[DECODER_SLOT][i] == "two-addr" for i in range(DECODER_PTR_WORDS)
            )
            and all(
                types[DECODER_SLOT][i] == "scalar"
                for i in range(DECODER_PTR_WORDS, DECODER_PTR_WORDS + 2)
            ),
            "offsets 0..64 two-addr, 64..80 scalar",
        )
        check(
            "decoder-counter-pad",
            enough and all(w[5] - w[4] == DECODER_COUNTER_PAD for w in dwords),
            "arg 40 - arg 32 == 512 (all evaluations)",
        )

    return report(checks, words, types, bindings)


def verdict_for(report):
    report["verdict"] = (
        "bindings-decoded" if not report.get("failed_checks") else "divergent"
    )
    return report


def analyse_pair(launches_a, launches_b, profile):
    """Compare two captures: identical per-evaluation parameter bytes."""
    ra = verdict_for(analyse(launches_a, profile))
    rb = verdict_for(analyse(launches_b, profile))
    pmem_a = [l["pmem"] for l in launches_a]
    pmem_b = [l["pmem"] for l in launches_b]
    identical = pmem_a == pmem_b
    checks = [{
        "name": "pair-identical",
        "ok": identical,
        "detail": "parameter bytes identical" if identical else "parameter bytes differ",
    }]
    failed = [c["name"] for c in checks if not c["ok"]]
    verdict = (
        "identical-bindings"
        if not failed and ra["verdict"] == rb["verdict"] == "bindings-decoded"
        else "divergent"
    )
    return {
        "mode": "pair",
        "checks": checks,
        "failed_checks": failed,
        "verdict_a": ra["verdict"],
        "verdict_b": rb["verdict"],
        "failed_a": ra["failed_checks"],
        "failed_b": rb["failed_checks"],
        "verdict": verdict,
    }


# --- fixtures and corruption controls ------------------------------------
def make_fixture(profile):
    """Synthesise a well-formed level-2 trace with the pinned structure."""
    per_eval = profile["per_eval"]
    n_eval = profile["evaluations"]
    lines = []
    seq = 0

    def emit(handle, grid, pmem):
        nonlocal seq
        ps = len(pmem)
        lines.append(
            f"{seq:06d} 100 352 launch id=0x5c52bb86 cmd=0x1 handle={handle} "
            f"grid={grid} params=0x2 psize={ps} pn={ps} pcap=0 "
            f"pmem={pmem.hex()} s=0x0"
        )
        seq += 1

    for e in range(n_eval):
        half = bool(e % 2)  # the two halves alternate in the fixture
        for slot in range(per_eval):
            if slot == DECODER_SLOT:
                words = [
                    0x20000000 + (HALF_DELTA if half else 0) + i * 0x200
                    for i in range(DECODER_PTR_WORDS)
                ]
                words[4] = 0x10000000 + (HALF_DELTA if half else 0)
                words[5] = words[4] + DECODER_COUNTER_PAD
                words += [0x000000100000000C, 0x0000001C00000018]
            elif slot == 0:
                words = [0x10000000 + (HALF_DELTA if half else 0), 0x9F00]
            else:
                words = [
                    0x30000000 + (HALF_DELTA if half else 0) + (slot % 7) * 0x200,
                    0,
                    0x000001A000000180,
                    0x3F800000,
                ]
            emit("0x1", "1,1,1", b"".join(int(w).to_bytes(8, "little") for w in words))
    return lines


def self_test():
    """Each control must fail for the reason its name states."""
    profile = PROFILES["fixture"]
    results = []

    def run(name, lines, expect_check, lines_b=None):
        try:
            if lines_b is not None:
                rep = analyse_pair(
                    parse_lines(lines, name), parse_lines(lines_b, name), profile
                )
            else:
                rep = verdict_for(analyse(parse_lines(lines, name), profile))
            failed = rep.get("failed_checks", [])
            results.append({"control": name, "ok": expect_check in failed,
                            "failed": failed})
        except (ValueError, KeyError, IndexError) as exc:
            results.append({
                "control": name,
                "ok": expect_check == "malformed-evidence",
                "failed": [f"exception:{type(exc).__name__}"],
            })

    def assert_control(name, ok, detail):
        results.append({"control": name, "ok": bool(ok), "failed": [detail]})

    def mutate(lines, line_index, fn):
        """Rewrite one line's parameter hex through fn(bytearray) -> None."""
        m = LAUNCH_RE.search(lines[line_index])
        pm = bytearray(bytes.fromhex(m.group(7)))
        fn(pm)
        lines[line_index] = (
            lines[line_index][: m.start(7)] + pm.hex() + lines[line_index][m.end(7):]
        )

    def get_word(lines, line_index, widx):
        m = LAUNCH_RE.search(lines[line_index])
        blob = bytes.fromhex(m.group(7))
        return int.from_bytes(blob[widx * 8:widx * 8 + 8], "little")

    def set_word(lines, line_index, widx, value):
        mutate(lines, line_index,
               lambda pm: pm.__setitem__(
                   slice(widx * 8, widx * 8 + 8),
                   int(value).to_bytes(8, "little")))

    base = make_fixture(profile)
    per = profile["per_eval"]
    dec_lines = [i for i, ln in enumerate(base) if "psize=80 " in ln]
    assert len(dec_lines) == profile["evaluations"], "fixture decoder lines"

    # 1. dropped launch
    run("drop-launch", base[:-1], "shape")
    # 2. pcap says "not requested" but a dump is present
    bad = list(base)
    bad[5] = bad[5].replace("pcap=0", "pcap=3", 1)
    run("pcap-mismatch", bad, "complete-dumps")
    # 3. truncated parameter hex in evaluation 0
    bad = list(base)
    bad[5] = bad[5][: bad[5].rindex("pmem=") + 5] + "00"
    run("pmem-truncated", bad, "complete-dumps")
    # 4. truncated parameter hex in a later evaluation (crash direction)
    bad = list(base)
    i = per + 5
    bad[i] = bad[i][: bad[i].rindex("pmem=") + 5] + "00"
    run("pmem-truncated-late", bad, "slot-structure-stable")
    # 5. word shape broken (parameter size not a whole number of words)
    bad = list(base)
    bad[5] = re.sub(r"psize=\d+ pn=\d+", "psize=33 pn=33", bad[5], count=1)
    run("word-shape", bad, "word-shape")
    # 6. empty parameter block must not pass the size bounds
    bad = list(base)
    bad[5] = re.sub(r"psize=\d+ pn=\d+ pcap=\d+ pmem=[0-9a-f]*",
                    "psize=0 pn=0 pcap=0 pmem=", bad[5], count=1)
    run("psize-zero", bad, "param-bounds")
    # 7. rotation delta corrupted (values stay aligned, difference wrong)
    bad = list(base)
    set_word(bad, 1, 0, get_word(bad, 1, 0) + 0x200)
    run("half-delta", bad, "half-delta")
    # 8. three-valued address position must not pass as two-addr
    a, b_, c = (0x30000000, 0x30000000 + HALF_DELTA, 0x30000000 + 2 * HALF_DELTA)
    assert_control(
        "addr-multi",
        word_type([a, b_, c]) == "addr-multi" and word_type([a, b_]) == "two-addr",
        f"word_type([3 values]) == {word_type([a, b_, c])}",
    )
    # 9. decoder parameter size corrupted
    bad = list(base)
    bad[dec_lines[0]] = re.sub(r"psize=\d+ pn=\d+", "psize=72 pn=72",
                               bad[dec_lines[0]], count=1)
    run("decoder-psize", bad, "decoder-psize")
    # 10. decoder counter padding corrupted in both evaluations (the words stay
    # aligned and rotating, so only the pad relation breaks)
    bad = list(base)
    for idx in dec_lines:
        set_word(bad, idx, 5, get_word(bad, idx, 5) + 512)
    run("decoder-counter-pad", bad, "decoder-counter-pad")
    # 11. decoder scalar word turned into a two-addr pair
    bad = list(base)
    for idx, half in ((dec_lines[0], False), (dec_lines[1], True)):
        set_word(bad, idx, 8, 0x20000000 + (HALF_DELTA if half else 0))
    run("decoder-pointer-words", bad, "decoder-pointer-words")
    # 12. pair divergence
    bad_b = list(base)
    mutate(bad_b, 5, lambda pm: pm.__setitem__(0, pm[0] ^ 1))
    run("pair-divergent", base, "pair-identical", lines_b=bad_b)
    # 13. malformed record (carries the launch marker but not the fields)
    run("malformed-record", base[:5] + ["000005 100 352 launch garbage"],
        "malformed-evidence")
    # 14. missing evidence file must be malformed-evidence, not a traceback
    try:
        parse_trace("/nonexistent/nvapi-trace.log")
        assert_control("missing-file", False, "no exception raised")
    except ValueError:
        assert_control("missing-file", True, "ValueError as expected")
    except OSError as exc:
        assert_control("missing-file", False, f"unwrapped {type(exc).__name__}")
    # 15. same value set with opposite rotation phase must NOT merge bindings
    bad = list(base)
    x = 0x30000000 + (2 % 7) * 0x200  # slot 2 word 0, evaluation 0
    set_word(bad, 3, 0, x + HALF_DELTA)     # slot 3: sequence (x+D, x)
    set_word(bad, per + 3, 0, x)
    rep = verdict_for(analyse(parse_lines(bad, "phase-split"), profile))
    ids = {}
    for bid, e in rep["bindings"].items():
        for ref in e["refs"]:
            ids[tuple(ref)] = bid
    ok = ids.get((2, 0)) is not None and ids.get((2, 0)) != ids.get((3, 0))
    assert_control("phase-split", ok, "positions (2,0)/(3,0) share a binding id")

    all_ok = all(r["ok"] for r in results)
    return {
        "mode": "self-test",
        "controls": results,
        "verdict": "controls-pass" if all_ok else "controls-fail",
    }


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
        report = self_test()
    elif args.trace:
        try:
            report = verdict_for(analyse(parse_trace(args.trace), PROFILES[args.profile]))
        except (ValueError, KeyError, IndexError) as exc:
            report = {"mode": "single", "verdict": "malformed-evidence", "error": str(exc)}
    elif args.a and args.b:
        try:
            report = analyse_pair(
                parse_trace(args.a), parse_trace(args.b), PROFILES[args.profile]
            )
        except (ValueError, KeyError, IndexError) as exc:
            report = {"mode": "pair", "verdict": "malformed-evidence", "error": str(exc)}
    else:
        parser.error("need --trace, --a/--b or --self-test")

    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))

    good = {"bindings-decoded", "identical-bindings", "controls-pass"}
    return 0 if report.get("verdict") in good else 1


if __name__ == "__main__":
    sys.exit(main())
