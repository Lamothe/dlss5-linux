#!/usr/bin/env python3
"""History-carrier validation (gate stage 13) — see vr/TEMPORAL-HISTORY.md.

Validates the capture-history evidence set under vr/logs/history/: the
history==output write identity, the all-zero reset value, the static
second-history control, the mid-reset restart and prefix, the withheld-write
cut run, the read-side intervention (pattern vs sham vs replay) and the
frame-counter result. Usage:

  python3 vr/history-carrier-report.py validate [root]
  python3 vr/history-carrier-report.py self-test

Validate walks the nine runs, recomputes every digest from the bytes on disk,
and checks the pinned relationships. Self-test builds a small synthetic fixture
(same relationships, 4x4 files) and asserts that each named corruption control
fails with its pinned check (exactly, where isolation is achievable).
"""

import os
import random
import re
import shutil
import struct
import sys
import tempfile

# ---------------------------------------------------------------------------
# Digests: standard FNV-1a-64. A non-standard constant here would silently
# break every cross-check (a real defect once found in a sibling tool).
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
FNV_MASK = 0xFFFFFFFFFFFFFFFF


def fnv64(data):
    """Standard FNV-1a-64 over bytes; returns an int in [0, 2**64)."""
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("fnv64 requires bytes")
    h = FNV_OFFSET
    for b in data:
        h = ((h ^ b) * FNV_PRIME) & FNV_MASK
    return h


# ---------------------------------------------------------------------------
# File classes. Real runs are 720x720; the self-test fixture uses 4x4.
def sizes_for(n):
    return {"flat": n * n * 8, "wide": n * n * 4}


REAL_SIZES = sizes_for(720)
FIXTURE_SIZES = sizes_for(4)

FLAT_NAMES = ["input-colour.bin"]
WIDE_NAMES = ["input-depth.bin", "input-motion.bin"]
OUTPUT_NAMES = ["output-frame%02u.bin" % n for n in range(8)]
HISTORY_NAMES = (
    ["history0-init.bin", "history1-init.bin"]
    + ["history0-frame%02u.bin" % n for n in range(8)]
    + ["history1-frame%02u.bin" % n for n in range(8)]
)
BIN_NAMES = FLAT_NAMES + WIDE_NAMES + OUTPUT_NAMES + HISTORY_NAMES


def name_size(name, sizes):
    return sizes["wide"] if name in WIDE_NAMES else sizes["flat"]


# ---------------------------------------------------------------------------
# Run table. `interventions` is the exact ordered list of
# (frame, pattern) pairs the run's manifest must carry.
PROTOCOL_PLAIN = ("protocol capture-history-v1 frames=8 size=%d context=eye0 "
                  "inputs=gradient-stationary reset=frame0")
PROTOCOL_RESET4 = ("protocol capture-history-v1 frames=8 size=%d context=eye0 "
                   "inputs=gradient-stationary reset=frame0+frame4")

RUNS = {
    "hist-a": {"protocol": PROTOCOL_PLAIN, "interventions": [], "cut": False},
    "hist-b": {"protocol": PROTOCOL_PLAIN, "interventions": [], "cut": False},
    "hist-r4": {"protocol": PROTOCOL_RESET4, "interventions": [], "cut": False},
    "hist-cut": {"protocol": PROTOCOL_PLAIN, "interventions": [], "cut": True},
    "hist-inj": {"protocol": PROTOCOL_PLAIN, "cut": False,
                 "interventions": [(2, "uniform-1.0"),
                                   (5, "checkerboard-1.0/0.0")]},
    "hist-inj-b": {"protocol": PROTOCOL_PLAIN, "cut": False,
                   "interventions": [(2, "uniform-1.0"),
                                     (5, "checkerboard-1.0/0.0")]},
    "hist-sham": {"protocol": PROTOCOL_PLAIN, "cut": False,
                  "interventions": [(2, "sham"), (5, "sham")]},
    "hist-replay": {"protocol": PROTOCOL_PLAIN, "cut": False,
                    "interventions": [(0, "stage-output00"),
                                      (2, "replay-output00"),
                                      (5, "replay-output00")]},
    "hist-replay2": {"protocol": PROTOCOL_PLAIN, "cut": False,
                     "interventions": [(0, "stage-output00"),
                                       (1, "uniform-1.0"),
                                       (2, "replay-output00"),
                                       (5, "replay-output00")]},
    "hist-injreset": {"protocol": PROTOCOL_RESET4, "cut": False,
                      "interventions": [(4, "uniform-1.0-at-reset")]},
}
RUN_ORDER = ["hist-a", "hist-b", "hist-r4", "hist-cut", "hist-inj",
             "hist-inj-b", "hist-sham", "hist-replay", "hist-replay2",
             "hist-injreset"]

# Externally anchored digests for hist-a (published in TEMPORAL-HISTORY.md),
# so a coordinated change to both the files and their manifest is still
# detectable without trusting the run itself.
ANCHORS = {
    "history0-frame00.bin": (4147200, 0x839de3f30038880a),
    "history0-frame01.bin": (4147200, 0x65cba335fdd167ed),
    "history0-frame02.bin": (4147200, 0x1f5df6063c506f74),
    "history0-frame03.bin": (4147200, 0x9997e15cd2957237),
    "history0-frame04.bin": (4147200, 0xc49fb39331ef808f),
    "history0-frame05.bin": (4147200, 0xbe6ebbf0d7da3676),
    "history0-frame06.bin": (4147200, 0x61050c5427b0ea49),
    "history0-frame07.bin": (4147200, 0xbbba397b954e15eb),
    "history0-init.bin": (4147200, 0x0b709f104affc325),
    "history1-frame00.bin": (4147200, 0x0b709f104affc325),
    "history1-frame01.bin": (4147200, 0x0b709f104affc325),
    "history1-frame02.bin": (4147200, 0x0b709f104affc325),
    "history1-frame03.bin": (4147200, 0x0b709f104affc325),
    "history1-frame04.bin": (4147200, 0x0b709f104affc325),
    "history1-frame05.bin": (4147200, 0x0b709f104affc325),
    "history1-frame06.bin": (4147200, 0x0b709f104affc325),
    "history1-frame07.bin": (4147200, 0x0b709f104affc325),
    "history1-init.bin": (4147200, 0x0b709f104affc325),
    "input-colour.bin": (4147200, 0xfb51a175911c82f5),
    "input-depth.bin": (2073600, 0x33533d64f7d84205),
    "input-motion.bin": (2073600, 0x227ea3796d10f325),
    "output-frame00.bin": (4147200, 0x839de3f30038880a),
    "output-frame01.bin": (4147200, 0x65cba335fdd167ed),
    "output-frame02.bin": (4147200, 0x1f5df6063c506f74),
    "output-frame03.bin": (4147200, 0x9997e15cd2957237),
    "output-frame04.bin": (4147200, 0xc49fb39331ef808f),
    "output-frame05.bin": (4147200, 0xbe6ebbf0d7da3676),
    "output-frame06.bin": (4147200, 0x61050c5427b0ea49),
    "output-frame07.bin": (4147200, 0xbbba397b954e15eb),
}

# Externally anchored digests for the two traces (see TEMPORAL-HISTORY.md):
# the counter trace is the only evidence for the counter reset, and the eval
# trace carries the scan pins, so both must not be substitutable.
COUNTER_TRACE_ANCHOR = 0xa098b0e2f552e779  # nvapi-trace-counter-reset.log
SCAN_TRACE_ANCHOR = 0x48a6bc869fa1196b     # nvapi-trace-eval-args-1.log

# Pinned check names; the self-test asserts each fires for its corruption.
CHECKS = [
    "protocol-mismatch",
    "manifest-format",
    "file-missing",
    "file-extra",
    "digest-mismatch",
    "intervention-mismatch",
    "anchor-mismatch",
    "pair-mismatch",
    "inject-pair-mismatch",
    "history-output-mismatch",
    "init-nonzero",
    "control-drift",
    "prefix-mismatch",
    "restart-mismatch",
    "no-divergence",
    "cut-history-written",
    "cut-output-written",
    "sham-mismatch",
    "inject-ineffective",
    "inject-prefix-mismatch",
    "replay-counter-absent",
    "replay2-trace",
    "injectreset-prefix-mismatch",
    "injectreset-not-ignored",
    "counter-reset-mismatch",
    "counter-anchor-mismatch",
    "input-mismatch",
    "scan-mismatch",
]


# ---------------------------------------------------------------------------
# Manifest parsing. Accepted line shapes (all produced by the probe):
#   protocol capture-history-v1 frames=8 size=%d context=eye0 ... reset=...
#   history idx=%u rec=%u stage=%s va=0x%llx w=%llu h=%u fmt=%u bytes=%zu file=%s fnv=0x%016llx
#   history-inject frame=%u idx=0 pattern=%s [bytes=%llu]
#   history-stage frame=0 source=history-readback
#   %s bytes=%zu fnv=0x%016llx
RE_PROTOCOL = re.compile(r"^protocol \S+ frames=(\d+) size=(\d+) .+$")
RE_HISTORY = re.compile(
    r"^history idx=\d+ rec=\d+ stage=\S+ va=0x[0-9a-fA-F]+ w=\d+ h=\d+ fmt=\d+ "
    r"bytes=(\d+) file=(\S+) fnv=0x([0-9a-fA-F]+)$")
RE_FILE = re.compile(r"^(\S+) bytes=(\d+) fnv=0x([0-9a-fA-F]+)$")
RE_INJECT = re.compile(r"^history-inject frame=(\d+) idx=0 pattern=(\S+?)(?: bytes=\d+)?$")
RE_STAGE = re.compile(r"^history-stage frame=0 source=history-readback$")
RE_LAUNCH = re.compile(
    r"launch id=0x[0-9a-f]+ cmd=0x[0-9a-f]+ handle=0x[0-9a-f]+ grid=[0-9,]+ "
    r"params=0x[0-9a-f]+ psize=(\d+) pn=\d+ pcap=\d+ pmem=([0-9a-f]*)")

COUNTER_SLOT = 1      # second launch of every evaluation
COUNTER_WORD = 25     # 64-bit word index inside the parameter block
COUNTER_EVALS = 8     # capture-history runs evaluate 8 frames
COUNTER_SLOTS = 156   # launches per evaluation (the recovered inventory)


def counter_sequence(trace_path):
    """Per-evaluation values of the frame-counter word (slot 1, word 25)."""
    launches = launch_blobs(trace_path)
    if len(launches) != COUNTER_EVALS * COUNTER_SLOTS:
        raise ValueError("%s: %d launches, expected %d"
                         % (trace_path, len(launches), COUNTER_EVALS * COUNTER_SLOTS))
    out = []
    for e in range(COUNTER_EVALS):
        blob = launches[e * COUNTER_SLOTS + COUNTER_SLOT]
        off = COUNTER_WORD * 8
        if len(blob) < off + 8:
            raise ValueError("%s: eval %d slot %d too short"
                             % (trace_path, e, COUNTER_SLOT))
        out.append(int.from_bytes(blob[off:off + 8], "little"))
    return out


def launch_blobs(trace_path):
    """All launch parameter blocks in trace order (one per launch line)."""
    launches = []
    with open(trace_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RE_LAUNCH.search(line)
            if not m:
                continue
            psize = int(m.group(1))
            blob = bytes.fromhex(m.group(2))
            if len(blob) < psize:
                raise ValueError("%s: truncated pmem" % trace_path)
            launches.append(blob)
    return launches


# Binding-scan pins (computed by `scan` over the 120-eval parameter dump; see
# TEMPORAL-HISTORY.md "The stage-9 'two memory halves' are the two features").
SCAN_EVALS = 120
SCAN_EYE = lambda i: ((i // 2) + (i % 2)) % 2  # submission order (frame+position)%2
SCAN_PINS = {
    "two_valued": 896,       # word positions taking exactly two values
    "tracks_eye": 886,       # value is a pure function of the evaluation's eye
    "one_time": 10,          # remaining two-valued positions (lazy scalars)
    "addr_shaped": 881,      # two-valued AND 512-aligned in [0x10000, 2**32)
    "addr_track": 881,       # address-shaped AND eye-tracking (all 881)
    "addr_lazy": 0,
    "nonaddr_track": 5,
    "nonaddr_lazy": 10,
    "per_eye_varying": [(1, 25)],  # positions varying per frame within one eye
    "lazy_switch_evals": [2],      # eval index where the lazy scalars switch
}


def scan_trace(trace_path, evals_expected, eye_fn, slots=COUNTER_SLOTS):
    """Classify every word position of every slot across a parameter dump.

    Returns the counts pinned in SCAN_PINS: two-valued positions, how many
    track the eye order, how many are one-time lazy scalars (and where they
    switch), how many two-valued positions are address-shaped, and which
    positions vary per frame within a single eye.
    """
    launches = launch_blobs(trace_path)
    if len(launches) != evals_expected * slots:
        raise ValueError("%s: %d launches, expected %d"
                         % (trace_path, len(launches), evals_expected * slots))
    evals = [launches[i * slots:(i + 1) * slots] for i in range(evals_expected)]
    two = tracks = onetime = addr_shaped = 0
    addr_track = addr_lazy = nonaddr_track = nonaddr_lazy = 0
    per_eye = []
    lazy_switches = set()
    for slot in range(slots):
        width = len(evals[0][slot])
        if width % 8:
            continue
        for w in range(width // 8):
            seq = [int.from_bytes(ev[slot][w * 8:w * 8 + 8], "little")
                   for ev in evals]
            values = set(seq)
            by_eye = {e: {seq[i] for i in range(evals_expected) if eye_fn(i) == e}
                      for e in (0, 1)}
            if len(values) > 2 and any(len(s) > 2 for s in by_eye.values()):
                per_eye.append((slot, w))
            if len(values) != 2:
                continue
            two += 1
            lo, hi = min(values), max(values)
            addr = all(0x10000 <= v < (1 << 32) and v % 512 == 0 for v in values)
            if addr:
                addr_shaped += 1
            track = all((v == hi) == (eye_fn(i) == 1) for i, v in enumerate(seq)) or \
                    all((v == hi) == (eye_fn(i) == 0) for i, v in enumerate(seq))
            if track:
                tracks += 1
            else:
                onetime += 1
                lazy_switches.update(
                    i for i in range(1, evals_expected) if seq[i] != seq[i - 1])
            if addr and track:
                addr_track += 1
            elif addr:
                addr_lazy += 1
            elif track:
                nonaddr_track += 1
            else:
                nonaddr_lazy += 1
    return {
        "two_valued": two,
        "tracks_eye": tracks,
        "one_time": onetime,
        "addr_shaped": addr_shaped,
        "addr_track": addr_track,
        "addr_lazy": addr_lazy,
        "nonaddr_track": nonaddr_track,
        "nonaddr_lazy": nonaddr_lazy,
        "per_eye_varying": per_eye,
        "lazy_switch_evals": sorted(lazy_switches),
    }


def parse_manifest(path):
    """Returns (protocol_line, size_hint, file_table, interventions)."""
    files = {}
    interventions = []
    protocol_line = None
    size_hint = None
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            m = RE_PROTOCOL.match(line)
            if m:
                if protocol_line is not None:
                    raise ValueError("%s:%d: duplicate protocol line" % (path, lineno))
                protocol_line = line
                size_hint = int(m.group(2))
                continue
            m = RE_HISTORY.match(line)
            if m:
                name = m.group(2)
                if name in files:
                    raise ValueError("%s:%d: duplicate entry %s" % (path, lineno, name))
                files[name] = (int(m.group(1)), int(m.group(3), 16))
                continue
            m = RE_FILE.match(line)
            if m:
                name = m.group(1)
                if name in files:
                    raise ValueError("%s:%d: duplicate entry %s" % (path, lineno, name))
                files[name] = (int(m.group(2)), int(m.group(3), 16))
                continue
            m = RE_INJECT.match(line)
            if m:
                interventions.append((int(m.group(1)), m.group(2)))
                continue
            if RE_STAGE.match(line):
                interventions.append((0, "stage-output00"))
                continue
            raise ValueError("%s:%d: unparsable manifest line" % (path, lineno))
    if protocol_line is None:
        raise ValueError("%s: no protocol line" % path)
    return protocol_line, size_hint, files, interventions


# ---------------------------------------------------------------------------
def validate(root, sizes, anchors, quiet=True, counter_trace=None, scan_trace_path=None,
             counter_anchor=None, scan_anchor=None):
    """Runs every pinned check; returns (failed_names, notes)."""
    failed = []
    notes = []

    def fail(name, detail):
        if name not in failed:
            failed.append(name)
        notes.append("%s: %s" % (name, detail))

    expected_size = 720 if sizes == REAL_SIZES else 4
    table = {}

    def blob(run, name):
        with open(os.path.join(root, run, name), "rb") as f:
            return f.read()

    def have(run, name):
        return os.path.isfile(os.path.join(root, run, name))

    for run in RUN_ORDER:
        rdir = os.path.join(root, run)
        mpath = os.path.join(rdir, "manifest.txt")
        if not os.path.isfile(mpath):
            fail("file-missing", "%s/manifest.txt" % run)
            continue
        spec = RUNS[run]
        try:
            protocol_line, size_hint, files, interventions = parse_manifest(mpath)
        except ValueError as exc:
            fail("manifest-format", str(exc))
            continue
        table[run] = files
        # 1. Protocol line exactly as pinned for this run and profile.
        want_protocol = spec["protocol"] % expected_size
        if protocol_line != want_protocol:
            fail("protocol-mismatch", "%s: %r" % (run, protocol_line))
        if size_hint != expected_size:
            fail("protocol-mismatch", "%s: size=%d" % (run, size_hint))
        # 2. Interventions: exact ordered (frame, pattern) list.
        if list(interventions) != list(spec["interventions"]):
            fail("intervention-mismatch",
                 "%s: %r != %r" % (run, list(interventions), list(spec["interventions"])))
        # 3. File set and per-file digests, recomputed from the bytes on disk.
        on_disk = set(f for f in os.listdir(rdir) if f.endswith(".bin"))
        expected_names = set(BIN_NAMES)
        for missing in sorted(expected_names - on_disk):
            fail("file-missing", "%s/%s" % (run, missing))
        for extra in sorted(on_disk - expected_names):
            fail("file-extra", "%s/%s" % (run, extra))
        for name in sorted(expected_names & on_disk):
            want = name_size(name, sizes)
            entry = files.get(name)
            if entry is None:
                fail("digest-mismatch", "%s/%s not in manifest" % (run, name))
                continue
            got_size, got_fnv = entry
            if type(got_size) is not int or type(got_fnv) is not int:
                fail("digest-mismatch", "%s/%s manifest types" % (run, name))
                continue
            try:
                data = blob(run, name)
            except OSError as exc:
                fail("digest-mismatch", "%s/%s unreadable (%s)" % (run, name, exc))
                continue
            if got_size != want or len(data) != want:
                fail("digest-mismatch", "%s/%s size %d/%d want %d" % (
                    run, name, got_size, len(data), want))
                continue
            if fnv64(data) != got_fnv:
                fail("digest-mismatch", "%s/%s fnv" % (run, name))

    # 4. External anchors on hist-a.
    for name, (asize, afnv) in sorted(anchors.items()):
        if not have("hist-a", name):
            continue
        data = blob("hist-a", name)
        if len(data) != asize or fnv64(data) != afnv:
            fail("anchor-mismatch", "hist-a/%s" % name)

    # 5. Pair identity: hist-b == hist-a, hist-inj-b == hist-inj.
    for a, b, name in (("hist-a", "hist-b", "pair-mismatch"),
                       ("hist-inj", "hist-inj-b", "inject-pair-mismatch")):
        if a not in table or b not in table:
            continue
        for bn in BIN_NAMES:
            if not have(a, bn) or not have(b, bn):
                continue
            if blob(a, bn) != blob(b, bn):
                fail(name, "%s vs %s/%s" % (a, b, bn))

    # 6. Write identity: history0 after frame N == output N in every run whose
    #    post-decoder stages actually ran. The cut run's write is withheld by
    #    construction (see check 10), so it is excluded here.
    for run in RUN_ORDER:
        if run not in table or RUNS[run]["cut"]:
            continue
        for n in range(8):
            hn = "history0-frame%02u.bin" % n
            on = "output-frame%02u.bin" % n
            if not have(run, hn) or not have(run, on):
                continue
            if blob(run, hn) != blob(run, on):
                fail("history-output-mismatch", "%s frame %u" % (run, n))

    # 7. Reset value: both init bins are all zero in every run.
    for run in RUN_ORDER:
        if run not in table:
            continue
        for name in ("history0-init.bin", "history1-init.bin"):
            if not have(run, name):
                continue
            if any(blob(run, name)):
                fail("init-nonzero", "%s/%s" % (run, name))

    # 8. Static control: history1 (the never-evaluated feature's history)
    #    never changes in any run.
    for run in RUN_ORDER:
        if run not in table or not have(run, "history1-init.bin"):
            continue
        init = blob(run, "history1-init.bin")
        for n in range(8):
            name = "history1-frame%02u.bin" % n
            if have(run, name) and blob(run, name) != init:
                fail("control-drift", "%s/%s" % (run, name))

    # 9. Mid-reset restart: hist-r4's frames 4+k replay hist-a's frames k, and
    #    the pre-reset prefix follows the reference.
    if have("hist-r4", "history0-frame00.bin") and have("hist-a", "history0-frame00.bin"):
        for n in range(4):
            name = "history0-frame%02u.bin" % n
            if have("hist-r4", name) and have("hist-a", name) and \
                    blob("hist-r4", name) != blob("hist-a", name):
                fail("prefix-mismatch", "history f%u" % n)
        for k in range(4):
            rn = "history0-frame%02u.bin" % (4 + k)
            an = "history0-frame%02u.bin" % k
            sn = "history0-frame%02u.bin" % (4 + k)
            if not have("hist-r4", rn) or not have("hist-a", an):
                continue
            if blob("hist-r4", rn) != blob("hist-a", an):
                fail("restart-mismatch", "history f%u" % (4 + k))
            if have("hist-a", sn) and blob("hist-r4", rn) == blob("hist-a", sn):
                fail("no-divergence", "history f%u" % (4 + k))

    # 10. Cut run: the history is never written (stays at init) and the output
    #     texture keeps the untouched FP16 NaN sentinel (0x7e00).
    if "hist-cut" in table and have("hist-cut", "history0-init.bin"):
        init = blob("hist-cut", "history0-init.bin")
        for n in range(8):
            name = "history0-frame%02u.bin" % n
            if have("hist-cut", name) and blob("hist-cut", name) != init:
                fail("cut-history-written", name)
    for n in range(8):
        name = "output-frame%02u.bin" % n
        if not have("hist-cut", name):
            continue
        data = blob("hist-cut", name)
        if len(data) % 2:
            fail("cut-output-written", "%s odd length" % name)
            continue
        halfs = struct.unpack("<%dH" % (len(data) // 2), data)
        if any(h != 0x7E00 for h in halfs):
            fail("cut-output-written", name)

    # 11. Sham control: content-preserving copies reproduce hist-a exactly.
    if "hist-sham" in table and "hist-a" in table:
        for name in BIN_NAMES:
            if not have("hist-sham", name) or not have("hist-a", name):
                continue
            if blob("hist-sham", name) != blob("hist-a", name):
                fail("sham-mismatch", name)

    # 12. Intervention effect: pattern runs diverge from the reference at and
    #     after the intervention frames (the divergence chains through the
    #     history), with an untouched prefix.
    for run in ("hist-inj", "hist-inj-b"):
        if run not in table:
            continue
        for n in (0, 1):
            name = "output-frame%02u.bin" % n
            if have(run, name) and have("hist-a", name) and \
                    blob(run, name) != blob("hist-a", name):
                fail("inject-prefix-mismatch", "%s f%u" % (run, n))
        for n in range(2, 8):
            name = "output-frame%02u.bin" % n
            if have(run, name) and have("hist-a", name) and \
                    blob(run, name) == blob("hist-a", name):
                fail("inject-ineffective", "%s f%u" % (run, n))

    # 13. Replay: with the history reset to output-00 at f2, the f2 output
    #     differs from f1's — a frame-indexed state (the counter) exists.
    if have("hist-replay", "output-frame02.bin") and have("hist-replay", "output-frame01.bin"):
        if blob("hist-replay", "output-frame02.bin") == \
                blob("hist-replay", "output-frame01.bin"):
            fail("replay-counter-absent", "f2 == f1")

    # 14. Replay2: the f1 perturbation took effect but leaves no trace at the
    #     shared-history frames (f2/f5 match hist-replay exactly).
    if "hist-replay2" in table and "hist-replay" in table:
        for n in (1,):
            name = "output-frame%02u.bin" % n
            if have("hist-replay2", name) and have("hist-replay", name) and \
                    blob("hist-replay2", name) == blob("hist-replay", name):
                fail("inject-ineffective", "replay2 f1 did not perturb")
        for n in (2, 5):
            name = "output-frame%02u.bin" % n
            if not have("hist-replay2", name) or not have("hist-replay", name):
                continue
            if blob("hist-replay2", name) != blob("hist-replay", name):
                fail("replay2-trace", "f%u" % n)

    # 15. Reset-frame injection: garbage injected before a reset frame must be
    #     ignored. The prefix follows the reference; frame 4 equals the
    #     zero-history frame-0 result; frames 5-7 replay 1-3 (counter restart).
    if "hist-injreset" in table and "hist-a" in table:
        for n in range(4):
            name = "output-frame%02u.bin" % n
            if not have("hist-injreset", name) or not have("hist-a", name):
                continue
            if blob("hist-injreset", name) != blob("hist-a", name):
                fail("injectreset-prefix-mismatch", "f%u" % n)
        for k in range(4):
            name = "output-frame%02u.bin" % (4 + k)
            want = "output-frame%02u.bin" % k
            if not have("hist-injreset", name) or not have("hist-a", want):
                continue
            if blob("hist-injreset", name) != blob("hist-a", want):
                fail("injectreset-not-ignored", "f%u == ref f%u" % (4 + k, k))

    # 16. Counter: the traced reset run's frame-counter word (slot 1 word 25)
    #     must restart exactly at the reset frame (0,1,2,3,0,1,2,3), and the
    #     trace itself is externally anchored (it is the only evidence for the
    #     counter reset, so a substituted trace must not pass).
    if counter_trace is not None:
        if counter_anchor is not None:
            try:
                with open(counter_trace, "rb") as f:
                    if fnv64(f.read()) != counter_anchor:
                        fail("counter-anchor-mismatch", counter_trace)
            except OSError as exc:
                fail("counter-anchor-mismatch", str(exc))
        try:
            seq = counter_sequence(counter_trace)
        except (ValueError, OSError) as exc:
            fail("counter-reset-mismatch", str(exc))
        else:
            if seq != [0, 1, 2, 3, 0, 1, 2, 3]:
                fail("counter-reset-mismatch", "counter=%r" % seq)

    # 16b. Cross-run inputs: every run must use the same protocol inputs as the
    #      reference (only hist-a's are externally anchored).
    if "hist-a" in table:
        for run in RUN_ORDER:
            if run not in table or run == "hist-a":
                continue
            for name in FLAT_NAMES + WIDE_NAMES:
                if not have(run, name) or not have("hist-a", name):
                    continue
                if blob(run, name) != blob("hist-a", name):
                    fail("input-mismatch", "%s/%s" % (run, name))

    # 17. Binding scan: the per-frame/eye classification of the 120-eval
    #     parameter dump must match the pinned scan (this is what rules out
    #     additional frame-indexed words beyond slot 1 word 25).
    if scan_trace_path is not None:
        if scan_anchor is not None:
            try:
                with open(scan_trace_path, "rb") as f:
                    if fnv64(f.read()) != scan_anchor:
                        fail("scan-mismatch", "trace anchor: %s" % scan_trace_path)
            except OSError as exc:
                fail("scan-mismatch", str(exc))
        try:
            got = scan_trace(scan_trace_path, SCAN_EVALS, SCAN_EYE)
        except (ValueError, OSError) as exc:
            fail("scan-mismatch", str(exc))
        else:
            for key, want in SCAN_PINS.items():
                if got.get(key) != want:
                    fail("scan-mismatch", "%s: %r != %r" % (key, got.get(key), want))

    if not quiet:
        for note in notes:
            print("  " + note)
    return failed, notes


# ---------------------------------------------------------------------------
# Synthetic fixture for self-test: the same relationships at 4x4.
def build_fixture(root, sizes):
    rnd = random.Random(20261014)
    flat, wide = sizes["flat"], sizes["wide"]
    sizes_side = int((flat // 8) ** 0.5)  # 720 for real, 4 for fixture

    def buf(n):
        return bytes(rnd.randrange(256) for _ in range(n))

    frames = [buf(flat) for _ in range(8)]
    init = bytes(flat)
    h1 = init
    # Inputs are protocol-constant across runs (the probe uploads the same
    # gradient/depth/motion in every run).
    shared_inputs = {}
    for name in FLAT_NAMES:
        shared_inputs[name] = buf(flat)
    for name in WIDE_NAMES:
        shared_inputs[name] = buf(wide)

    def write_run(run, outputs, h0, interventions, protocol):
        rdir = os.path.join(root, run)
        os.makedirs(rdir)
        files = {}
        for name in FLAT_NAMES + WIDE_NAMES:
            files[name] = shared_inputs[name]
        for n in range(8):
            files["output-frame%02u.bin" % n] = outputs[n]
            files["history0-frame%02u.bin" % n] = h0[n]
            files["history1-frame%02u.bin" % n] = h1
        files["history0-init.bin"] = init
        files["history1-init.bin"] = init
        lines = [protocol % 4]
        for name in sorted(files):
            data = files[name]
            with open(os.path.join(rdir, name), "wb") as f:
                f.write(data)
            if name.startswith("history"):
                idx = int(name[len("history")])
                stage = "init" if name.endswith("init.bin") else \
                    "frame" + name[len("history0-frame"):-len(".bin")]
                lines.append(
                    "history idx=%u rec=%u stage=%s va=0x0 w=%d h=%d fmt=10 "
                    "bytes=%d file=%s fnv=0x%016x"
                    % (idx, 3 + 4 * idx, stage, sizes_side, sizes_side, len(data),
                       name, fnv64(data)))
            else:
                lines.append("%s bytes=%d fnv=0x%016x" % (name, len(data), fnv64(data)))
        for frame, pattern in interventions:
            if pattern == "stage-output00":
                lines.append("history-stage frame=0 source=history-readback")
            elif pattern in ("sham", "replay-output00", "uniform-1.0-at-reset"):
                # The probe writes these byte-less (see feature-probe.c).
                lines.append("history-inject frame=%d idx=0 pattern=%s" % (frame, pattern))
            else:
                lines.append("history-inject frame=%d idx=0 pattern=%s bytes=%d"
                             % (frame, pattern, flat))
        with open(os.path.join(rdir, "manifest.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    plain = RUNS["hist-a"]["protocol"]
    reset4 = RUNS["hist-r4"]["protocol"]

    write_run("hist-a", list(frames), list(frames), [], plain)
    write_run("hist-b", list(frames), list(frames), [], plain)
    write_run("hist-sham", list(frames), list(frames), [(2, "sham"), (5, "sham")], plain)

    write_run("hist-r4", frames[:4] + frames[:4], frames[:4] + frames[:4], [], reset4)

    nan = b"\x00\x7e" * (flat // 2)
    write_run("hist-cut", [nan] * 8, [init] * 8, [], plain)

    inj_out = frames[:2] + [buf(flat) for _ in range(6)]
    write_run("hist-inj", inj_out, list(inj_out),
              [(2, "uniform-1.0"), (5, "checkerboard-1.0/0.0")], plain)
    write_run("hist-inj-b", inj_out, list(inj_out),
              [(2, "uniform-1.0"), (5, "checkerboard-1.0/0.0")], plain)

    # Replay: f2/f5 take output-00 as history yet differ from f1 (counter).
    rep_out = frames[:2] + [buf(flat) for _ in range(6)]
    write_run("hist-replay", rep_out, list(rep_out),
              [(0, "stage-output00"), (2, "replay-output00"), (5, "replay-output00")],
              plain)
    # Replay2: f1 perturbed (differs from replay's f1); f2/f5 identical to
    # replay's despite the earlier perturbation.
    rep2_out = [rep_out[0], buf(flat), rep_out[2], buf(flat), buf(flat),
                rep_out[5], buf(flat), buf(flat)]
    write_run("hist-replay2", rep2_out, list(rep2_out),
              [(0, "stage-output00"), (1, "uniform-1.0"), (2, "replay-output00"),
               (5, "replay-output00")], plain)

    # Reset-frame injection: the pattern is ignored at the reset frame, so the
    # run replays the reference (prefix equal, f4..f7 == ref f0..f3).
    write_run("hist-injreset", frames[:4] + frames[:4], frames[:4] + frames[:4],
              [(4, "uniform-1.0-at-reset")], reset4)


def write_counter_trace(path, values):
    """Minimal trace with the pinned launch structure (156 launches/eval) and
    the frame-counter word at slot 1 word 25 of every evaluation."""
    lines = []
    for counter in values:
        for slot in range(COUNTER_SLOTS):
            pmem = bytearray(26 * 8)
            if slot == COUNTER_SLOT:
                pmem[COUNTER_WORD * 8:COUNTER_WORD * 8 + 8] = \
                    int(counter).to_bytes(8, "little")
            lines.append("launch id=0x5c52bb86 cmd=0x1 handle=0x2 grid=1,1,1 "
                         "params=0x3 psize=%d pn=%d pcap=0 pmem=%s"
                         % (len(pmem), len(pmem), pmem.hex()))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def anchors_for(root):
    out = {}
    rdir = os.path.join(root, "hist-a")
    for name in sorted(os.listdir(rdir)):
        if not name.endswith(".bin"):
            continue
        with open(os.path.join(rdir, name), "rb") as f:
            data = f.read()
        out[name] = (len(data), fnv64(data))
    return out


def reseal(root, run):
    """Rewrite a run's manifest digests from the bytes currently on disk, so
    corruption controls isolate relational/anchored checks."""
    rdir = os.path.join(root, run)
    mpath = os.path.join(rdir, "manifest.txt")
    lines = []
    with open(mpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            m = RE_HISTORY.match(line)
            if m:
                name = m.group(2)
                with open(os.path.join(rdir, name), "rb") as bf:
                    data = bf.read()
                # History lines carry file=<name> between bytes= and fnv=, so
                # rewrite the whole tail (bytes=.. file=.. fnv=..); fail loud
                # if the line ever stops matching that shape.
                tail = "bytes=%d file=%s fnv=0x%016x" % (len(data), name, fnv64(data))
                new_line, nsub = re.subn(r"bytes=\d+ file=\S+ fnv=0x[0-9a-fA-F]+$",
                                         lambda _: tail, line)
                if nsub != 1:
                    raise ValueError("reseal: cannot rewrite %r" % (line,))
                lines.append(new_line)
                continue
            m = RE_FILE.match(line)
            if m:
                name = m.group(1)
                with open(os.path.join(rdir, name), "rb") as bf:
                    data = bf.read()
                lines.append("%s bytes=%d fnv=0x%016x" % (name, len(data), fnv64(data)))
                continue
            lines.append(line)
    with open(mpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def mutate(root, run, name, blob_fn):
    path = os.path.join(root, run, name)
    with open(path, "rb") as f:
        data = f.read()
    with open(path, "wb") as f:
        f.write(blob_fn(data))


def flip(data):
    b = bytearray(data)
    b[len(b) // 2] ^= 0xFF
    return bytes(b)


def copy_file(root, src_run, src_name, dst_run, dst_name):
    with open(os.path.join(root, src_run, src_name), "rb") as f:
        data = f.read()
    with open(os.path.join(root, dst_run, dst_name), "wb") as f:
        f.write(data)


# ---------------------------------------------------------------------------
def self_test():
    tmp = tempfile.mkdtemp(prefix="history-selftest-")
    problems = []
    try:
        build_fixture(tmp, FIXTURE_SIZES)
        fixture_anchors = anchors_for(tmp)
        counter_path = os.path.join(tmp, "counter-trace.log")
        write_counter_trace(counter_path, [0, 1, 2, 3, 0, 1, 2, 3])
        failed, _ = validate(tmp, FIXTURE_SIZES, fixture_anchors,
                             counter_trace=counter_path)
        if failed:
            print("FAIL: baseline fixture rejected: %r" % failed)
            return 1
        print("baseline accepted")

        controls_run = []

        def control(name, mutate_fn, expect, isolated=True):
            controls_run.append(name)
            root = os.path.join(tmp, name)
            shutil.copytree(tmp, root, ignore=shutil.ignore_patterns(name))
            trace = os.path.join(root, "counter-trace.log")
            mutate_fn(root, trace)
            got, _ = validate(root, FIXTURE_SIZES, fixture_anchors,
                              counter_trace=trace)
            ok = expect in got and (got == [expect] if isolated else True)
            if not ok:
                problems.append(name)
                print("FAIL %-26s expected %s%s got %r"
                      % (name, expect, "" if isolated else " (any)", got))
            else:
                print("ok   %-26s -> %s%s" % (name, expect,
                                              "" if isolated else " in %r" % got))
            shutil.rmtree(root)

        # -- manifest-shape corruptions ---------------------------------
        def c_protocol(root, trace):
            p = os.path.join(root, "hist-b", "manifest.txt")
            s = open(p).read().replace("frames=8", "frames=9", 1)
            open(p, "w").write(s)

        def c_format(root, trace):
            with open(os.path.join(root, "hist-b", "manifest.txt"), "a") as f:
                f.write("garbage line\n")

        def c_missing(root, trace):
            os.remove(os.path.join(root, "hist-b", "history0-frame03.bin"))

        def c_extra(root, trace):
            with open(os.path.join(root, "hist-b", "junk.bin"), "wb") as f:
                f.write(b"x")

        def c_digest(root, trace):
            # Corrupt a manifest digest entry; the bytes on disk stay intact,
            # so only the digest check can fire (changing the bytes would also
            # trip the cross-run input check).
            p = os.path.join(root, "hist-replay", "manifest.txt")
            s = open(p, encoding="utf-8").read()
            i = s.index("input-colour.bin bytes=")
            j = s.index("fnv=0x", i) + len("fnv=0x")
            s = s[:j] + ("0" if s[j] != "0" else "1") + s[j + 1:]
            open(p, "w", encoding="utf-8").write(s)

        def c_intervention(root, trace):
            p = os.path.join(root, "hist-inj", "manifest.txt")
            s = open(p).read().replace("frame=2 idx=0 pattern=uniform-1.0",
                                       "frame=3 idx=0 pattern=uniform-1.0")
            open(p, "w").write(s)

        # -- relational corruptions (resealed manifests) ----------------
        def c_anchor(root, trace):
            # Same change across hist-a, hist-b and hist-sham keeps every
            # relational check passing; only the external anchors can catch it
            # (late frames: the restart compares early frames only).
            for run in ("hist-a", "hist-b", "hist-sham"):
                mutate(root, run, "history0-frame04.bin", flip)
                mutate(root, run, "output-frame04.bin", flip)
                reseal(root, run)

        def c_pair(root, trace):
            mutate(root, "hist-b", "history0-frame03.bin", flip)
            mutate(root, "hist-b", "output-frame03.bin", flip)
            reseal(root, "hist-b")

        def c_inject_pair(root, trace):
            mutate(root, "hist-inj-b", "history0-frame04.bin", flip)
            mutate(root, "hist-inj-b", "output-frame04.bin", flip)
            reseal(root, "hist-inj-b")

        def c_hist_out(root, trace):
            mutate(root, "hist-replay", "history0-frame02.bin", flip)
            reseal(root, "hist-replay")

        def c_init(root, trace):
            mutate(root, "hist-replay", "history0-init.bin", flip)
            reseal(root, "hist-replay")

        def c_control_drift(root, trace):
            mutate(root, "hist-replay", "history1-frame03.bin", flip)
            reseal(root, "hist-replay")

        def c_prefix(root, trace):
            mutate(root, "hist-r4", "history0-frame01.bin", flip)
            mutate(root, "hist-r4", "output-frame01.bin", flip)
            reseal(root, "hist-r4")

        def c_restart(root, trace):
            mutate(root, "hist-r4", "history0-frame05.bin", flip)
            mutate(root, "hist-r4", "output-frame05.bin", flip)
            reseal(root, "hist-r4")

        def c_divergence(root, trace):
            # Make hist-r4's late frames equal the pair's late frames: restart
            # still passes (it compares hist-a's early frames) but the required
            # divergence disappears. The pair's late frames change, so the
            # external anchors fire as well — accepted as non-isolated.
            for k in range(4):
                copy_file(root, "hist-r4", "history0-frame%02u.bin" % (4 + k),
                          "hist-a", "history0-frame%02u.bin" % (4 + k))
                copy_file(root, "hist-r4", "output-frame%02u.bin" % (4 + k),
                          "hist-a", "output-frame%02u.bin" % (4 + k))
            for run in ("hist-a", "hist-b"):
                for k in range(4):
                    copy_file(root, "hist-a", "history0-frame%02u.bin" % (4 + k),
                              run, "history0-frame%02u.bin" % (4 + k))
                    copy_file(root, "hist-a", "output-frame%02u.bin" % (4 + k),
                              run, "output-frame%02u.bin" % (4 + k))
                reseal(root, run)

        def c_cut_hist(root, trace):
            mutate(root, "hist-cut", "history0-frame01.bin", flip)
            reseal(root, "hist-cut")

        def c_cut_out(root, trace):
            mutate(root, "hist-cut", "output-frame01.bin", flip)
            reseal(root, "hist-cut")

        def c_sham(root, trace):
            mutate(root, "hist-sham", "history0-frame02.bin", flip)
            mutate(root, "hist-sham", "output-frame02.bin", flip)
            reseal(root, "hist-sham")

        def c_inject_effect(root, trace):
            # Make the f2 intervention frame identical to the reference in
            # both pattern runs: the required divergence disappears.
            for run in ("hist-inj", "hist-inj-b"):
                copy_file(root, "hist-a", "history0-frame02.bin",
                          run, "history0-frame02.bin")
                copy_file(root, "hist-a", "output-frame02.bin",
                          run, "output-frame02.bin")
                reseal(root, run)

        def c_inject_prefix(root, trace):
            for run in ("hist-inj", "hist-inj-b"):
                mutate(root, run, "history0-frame01.bin", flip)
                mutate(root, run, "output-frame01.bin", flip)
                reseal(root, run)

        def c_replay_counter(root, trace):
            # Force the replay run's f2 to equal its f1 (both files).
            for name in ("output-frame02.bin", "history0-frame02.bin"):
                copy_file(root, "hist-replay", "output-frame01.bin",
                          "hist-replay", name)
            reseal(root, "hist-replay")
            for name in ("output-frame02.bin", "history0-frame02.bin"):
                copy_file(root, "hist-replay", "output-frame02.bin",
                          "hist-replay2", name)
            reseal(root, "hist-replay2")

        def c_replay2(root, trace):
            mutate(root, "hist-replay2", "history0-frame02.bin", flip)
            mutate(root, "hist-replay2", "output-frame02.bin", flip)
            reseal(root, "hist-replay2")

        def c_injreset_prefix(root, trace):
            mutate(root, "hist-injreset", "history0-frame01.bin", flip)
            mutate(root, "hist-injreset", "output-frame01.bin", flip)
            reseal(root, "hist-injreset")

        def c_injreset_effect(root, trace):
            # Break the post-reset replay: f5 must equal ref f1.
            mutate(root, "hist-injreset", "history0-frame05.bin", flip)
            mutate(root, "hist-injreset", "output-frame05.bin", flip)
            reseal(root, "hist-injreset")

        def c_counter(root, trace):
            write_counter_trace(trace, [0, 1, 2, 3, 0, 1, 2, 4])

        def c_counter_short(root, trace):
            write_counter_trace(trace, [0, 1, 2, 3])

        control("bad-protocol", c_protocol, "protocol-mismatch")
        control("bad-manifest-line", c_format, "manifest-format")
        control("missing-bin", c_missing, "file-missing")
        control("extra-bin", c_extra, "file-extra")
        control("bad-digest", c_digest, "digest-mismatch")
        control("bad-intervention", c_intervention, "intervention-mismatch")
        control("broken-anchor", c_anchor, "anchor-mismatch")
        control("broken-pair", c_pair, "pair-mismatch")
        control("broken-inject-pair", c_inject_pair, "inject-pair-mismatch")
        control("history-not-output", c_hist_out, "history-output-mismatch")
        control("nonzero-init", c_init, "init-nonzero")
        control("control-drifted", c_control_drift, "control-drift")
        control("broken-prefix", c_prefix, "prefix-mismatch")
        control("broken-restart", c_restart, "restart-mismatch")
        control("no-restart-divergence", c_divergence, "no-divergence", isolated=False)
        control("cut-history-written", c_cut_hist, "cut-history-written")
        control("cut-output-written", c_cut_out, "cut-output-written")
        control("sham-diverged", c_sham, "sham-mismatch")
        control("inject-ineffective", c_inject_effect, "inject-ineffective")
        control("inject-prefix-broken", c_inject_prefix, "inject-prefix-mismatch")
        control("replay-counter-gone", c_replay_counter, "replay-counter-absent")
        control("replay2-leaves-trace", c_replay2, "replay2-trace")
        control("injreset-prefix-broken", c_injreset_prefix, "injectreset-prefix-mismatch")
        control("injreset-not-ignored", c_injreset_effect, "injectreset-not-ignored")
        control("counter-not-reset", c_counter, "counter-reset-mismatch")
        control("counter-trace-short", c_counter_short, "counter-reset-mismatch")

        if problems:
            print("FAIL: %d self-test control(s) missed" % len(problems))
            return 1
        print("PASS: history-carrier checks validated (%d controls, %d checks)"
              % (len(controls_run), len(CHECKS)))
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
def main(argv):
    if len(argv) >= 2 and argv[1] == "self-test":
        return self_test()
    if len(argv) >= 2 and argv[1] == "validate":
        root = argv[2] if len(argv) > 2 else "vr/logs/history"
        trace = argv[3] if len(argv) > 3 else "vr/logs/capture/nvapi-trace-counter-reset.log"
        scan = argv[4] if len(argv) > 4 else "vr/logs/capture/nvapi-trace-eval-args-1.log"
        failed, _ = validate(root, REAL_SIZES, ANCHORS, quiet=False,
                             counter_trace=trace, counter_anchor=COUNTER_TRACE_ANCHOR,
                             scan_trace_path=scan, scan_anchor=SCAN_TRACE_ANCHOR)
        if failed:
            print("REJECT: history-carrier evidence (%s)" % ", ".join(failed))
            return 1
        print("PASS: history-carrier evidence validated (runs=%d checks=%d)"
              % (len(RUN_ORDER), len(CHECKS)))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
