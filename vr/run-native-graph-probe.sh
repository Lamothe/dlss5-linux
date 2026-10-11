#!/usr/bin/env bash
# Bounded static graph contract inspection, no Windows host code or GPU access.
set -euo pipefail
ulimit -c 0 # Defence in depth; the probe also sets PR_SET_DUMPABLE=0 before reading.
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
OUT="$ROOT/research/native-linux/runs"
DLL="$ROOT/model/nvngx_dlssnr.dll"
mode="${1:-inspect}"
[[ $# -le 1 ]] || { echo 'Expected at most one mode' >&2; exit 1; }
case "$mode" in inspect|negative) ;; *) echo 'Usage: run-native-graph-probe.sh [inspect|negative]' >&2; exit 1 ;; esac
for command in python3 objdump bwrap timeout flock; do command -v "$command" >/dev/null; done
[[ -f "$DLL" ]]
# The Python inventory verifies the pinned hash AFTER disabling dumpability.
mkdir -p "$OUT"
# Same lock as the native NGX probe; do not overlap GPU probes.
exec 9>"$OUT/probe.lock"
flock -n 9 || { echo 'Another native probe is running' >&2; exit 1; }
run="$(mktemp -d "$OUT/graph-$mode-XXXXXXXX")"
# Freeze the actual source supplied to the sandbox; retain hashes and snapshots.
mkdir "$run/source" "$run/evidence"
cp -- "$ROOT/vr/run-native-graph-probe.sh" "$run/source/run-native-graph-probe.sh"
cp -- "$ROOT/vr/native-runtime-probe.py" "$run/source/native-runtime-probe.py"
cp -- "$ROOT/vr/native-graph-probe.py" "$run/source/native-graph-probe.py"
chmod a-w "$run/source/"*
python3 - "$run/source" "$mode" <<'PY_MANIFEST'
from datetime import datetime, timezone
import hashlib, json, platform, sys
from pathlib import Path
source = Path(sys.argv[1])
manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), mode=sys.argv[2],
                python=platform.python_version(), kernel=platform.release(),
                source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(source.iterdir())})
(source.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY_MANIFEST
args=(--unshare-all --die-with-parent --new-session
  --ro-bind /usr /usr --ro-bind /etc /etc --proc /proc --dev /dev
  --ro-bind /sys /sys --tmpfs /tmp --tmpfs /run --tmpfs /var --tmpfs /home
  --ro-bind "$run/source/native-runtime-probe.py" /native-runtime-probe.py
  --ro-bind "$run/source/native-graph-probe.py" /probe.py
  --ro-bind "$DLL" /model.dll --bind "$run/evidence" /work --chdir /work
  --clearenv --setenv PATH /usr/bin --setenv HOME /tmp
  --setenv CUDA_MODULE_LOADING EAGER --setenv CUDA_CACHE_DISABLE 1
  --setenv PYTHONDONTWRITEBYTECODE 1)
# No NVIDIA device nodes are exposed to this static inspection.
for path in /bin /sbin /lib /lib64; do
  if [[ -L "$path" ]]; then args+=(--symlink "$(readlink "$path")" "$path"); else args+=(--ro-bind "$path" "$path"); fi
done
printf 'Run evidence: %s\n' "$run"
set +e
timeout --signal=TERM --kill-after=5s 90s bwrap "${args[@]}" \
  python3 /probe.py "$mode" /model.dll >"$run/output.json" 2>"$run/stderr.txt"
status=$?
set -e
printf '%s\n' "$status" >"$run/exit-status.txt"
cat "$run/stderr.txt" >&2
if [[ "$status" == 0 ]]; then
  python3 - "$run/output.json" <<'PY'
import json,sys
from pathlib import Path
report=json.loads(Path(sys.argv[1]).read_text())
print(json.dumps(report['graph'],indent=2))
PY
fi
exit "$status"
