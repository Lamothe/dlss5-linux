#!/usr/bin/env bash
# Native-vs-vendor decoder comparison: replay captured boundary buffers natively.
set -euo pipefail
ulimit -c 0 # Defence in depth; the probe also sets PR_SET_DUMPABLE=0 before reading.
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
OUT="$ROOT/research/native-linux/runs"
DLL="$ROOT/model/nvngx_dlssnr.dll"
mode="${1:-execute}"
capture="${2:-a}"
case "$mode" in execute|skip|wrong) ;; *) echo 'Usage: run-native-vendor-compare-probe.sh [execute|skip|wrong] [a|b]' >&2; exit 1 ;; esac
CAPTURE="$ROOT/vr/logs/boundary/$capture"
for command in python3 bwrap timeout flock; do command -v "$command" >/dev/null; done
[[ -f "$DLL" && -f "$CAPTURE/manifest.txt" ]]
# The Python inventory verifies the pinned hash AFTER disabling dumpability.
mkdir -p "$OUT"
exec 9>"$OUT/probe.lock"
flock -n 9 || { echo 'Another native probe is running' >&2; exit 1; }
run="$(mktemp -d "$OUT/vendor-compare-$mode-XXXXXXXX")"
# Freeze the actual source supplied to the sandbox; retain hashes and snapshots.
mkdir "$run/source" "$run/evidence"
cp -- "$ROOT/vr/run-native-vendor-compare-probe.sh" "$run/source/run-native-vendor-compare-probe.sh"
cp -- "$ROOT/vr/native-runtime-probe.py" "$run/source/native-runtime-probe.py"
cp -- "$ROOT/vr/native-kernel-probe.py" "$run/source/native-kernel-probe.py"
cp -- "$ROOT/vr/native-graph-probe.py" "$run/source/native-graph-probe.py"
cp -- "$ROOT/vr/native-decoder-probe.py" "$run/source/native-decoder-probe.py"
cp -- "$ROOT/vr/native-decoder-launch-probe.py" "$run/source/native-decoder-launch-probe.py"
cp -- "$ROOT/vr/decoder-boundary-report.py" "$run/source/decoder-boundary-report.py"
cp -- "$ROOT/vr/native-vendor-compare-probe.py" "$run/source/native-vendor-compare-probe.py"
chmod a-w "$run/source/"*
python3 - "$run/source" "$mode" "$capture" <<'PY_MANIFEST'
from datetime import datetime, timezone
import hashlib, json, platform, sys
from pathlib import Path
source = Path(sys.argv[1])
manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), mode=sys.argv[2],
                capture=sys.argv[3], python=platform.python_version(),
                kernel=platform.release(),
                source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(source.iterdir())})
(source.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY_MANIFEST
args=(--unshare-all --die-with-parent --new-session
  --ro-bind /usr /usr --ro-bind /etc /etc --proc /proc --dev /dev
  --ro-bind /sys /sys --tmpfs /tmp --tmpfs /run --tmpfs /var --tmpfs /home
  --ro-bind "$run/source/native-runtime-probe.py" /native-runtime-probe.py
  --ro-bind "$run/source/native-kernel-probe.py" /native-kernel-probe.py
  --ro-bind "$run/source/native-graph-probe.py" /native-graph-probe.py
  --ro-bind "$run/source/native-decoder-probe.py" /native-decoder-probe.py
  --ro-bind "$run/source/native-decoder-launch-probe.py" /native-decoder-launch-probe.py
  --ro-bind "$run/source/decoder-boundary-report.py" /decoder-boundary-report.py
  --ro-bind "$run/source/native-vendor-compare-probe.py" /probe.py
  --ro-bind "$DLL" /model.dll --ro-bind "$CAPTURE" /capture
  --bind "$run/evidence" /work --chdir /work
  --clearenv --setenv PATH /usr/bin --setenv HOME /tmp
  --setenv CUDA_MODULE_LOADING EAGER --setenv CUDA_CACHE_DISABLE 1
  --setenv PYTHONDONTWRITEBYTECODE 1)
for path in /dev/nvidiactl /dev/nvidia0 /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
  [[ ! -e "$path" ]] || args+=(--dev-bind "$path" "$path")
done
for path in /bin /sbin /lib /lib64; do
  if [[ -L "$path" ]]; then args+=(--symlink "$(readlink "$path")" "$path"); else args+=(--ro-bind "$path" "$path"); fi
done
printf 'Run evidence: %s\n' "$run"
set +e
timeout --signal=TERM --kill-after=5s 300s bwrap "${args[@]}" \
  python3 /probe.py "$mode" /model.dll /capture >"$run/output.json" 2>"$run/stderr.txt"
status=$?
set -e
printf '%s\n' "$status" >"$run/exit-status.txt"
cat "$run/stderr.txt" >&2
# Report best-effort: never let a malformed output.json change the exit status.
if [[ -s "$run/output.json" ]]; then
  python3 - "$run/output.json" <<'PY' || true
import json,sys
from pathlib import Path
try:
    report=json.loads(Path(sys.argv[1]).read_text())
except Exception as error:
    print(f"output.json unreadable: {error}")
else:
    print(json.dumps(report.get("cuda", report),indent=2))
PY
fi
exit "$status"
