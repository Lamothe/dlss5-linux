#!/usr/bin/env bash
# Bounded native FP16/FP8 projection execution; NOT full neural image inference.
set -euo pipefail
ulimit -c 0 # Defence in depth; the probe also sets PR_SET_DUMPABLE=0 before reading.
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
OUT="$ROOT/research/native-linux/runs"
DLL="$ROOT/model/nvngx_dlssnr.dll"
mode="${1:-execute}"
block="${2:-31}"
[[ $# -le 2 && "$block" =~ ^(31|32|33|34|35|36|37|38)$ ]] || { echo 'Expected optional candidate block 31..38' >&2; exit 1; }
case "$mode" in execute|skip-projection|execute-packed|skip-packed|wrong-packed|execute-fp8|skip-fp8|wrong-fp8|execute-candidate|skip-candidate|wrong-candidate) ;; *) echo 'Usage: run-native-projection-probe.sh [execute|skip-projection|execute-packed|skip-packed|wrong-packed|execute-fp8|skip-fp8|wrong-fp8|execute-candidate|skip-candidate|wrong-candidate]' >&2; exit 1 ;; esac
for command in python3 bwrap timeout flock; do command -v "$command" >/dev/null; done
[[ -f "$DLL" ]]
# The Python inventory verifies the pinned hash AFTER disabling dumpability.
mkdir -p "$OUT"
exec 9>"$OUT/probe.lock"
flock -n 9 || { echo 'Another native probe is running' >&2; exit 1; }
run="$(mktemp -d "$OUT/projection-$mode-XXXXXXXX")"
# Freeze the actual source supplied to the sandbox; retain hashes and snapshots.
mkdir "$run/source" "$run/evidence"
cp -- "$ROOT/vr/run-native-projection-probe.sh" "$run/source/run-native-projection-probe.sh"
cp -- "$ROOT/vr/native-runtime-probe.py" "$run/source/native-runtime-probe.py"
cp -- "$ROOT/vr/native-kernel-probe.py" "$run/source/native-kernel-probe.py"
cp -- "$ROOT/vr/native-tensor-probe.py" "$run/source/native-tensor-probe.py"
cp -- "$ROOT/vr/native-projection-probe.py" "$run/source/native-projection-probe.py"
chmod a-w "$run/source/"*
python3 - "$run/source" "$mode" "$block" <<'PY_MANIFEST'
from datetime import datetime, timezone
import hashlib, json, platform, sys
from pathlib import Path
source = Path(sys.argv[1])
manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), mode=sys.argv[2], candidate_block=int(sys.argv[3]),
                python=platform.python_version(), kernel=platform.release(),
                source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(source.iterdir())})
(source.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY_MANIFEST
args=(--unshare-all --die-with-parent --new-session
  --ro-bind /usr /usr --ro-bind /etc /etc --proc /proc --dev /dev
  --ro-bind /sys /sys --tmpfs /tmp --tmpfs /run --tmpfs /var --tmpfs /home
  --ro-bind "$run/source/native-runtime-probe.py" /native-runtime-probe.py
  --ro-bind "$run/source/native-kernel-probe.py" /native-kernel-probe.py
  --ro-bind "$run/source/native-tensor-probe.py" /native-tensor-probe.py
  --ro-bind "$run/source/native-projection-probe.py" /probe.py
  --ro-bind "$DLL" /model.dll --bind "$run/evidence" /work --chdir /work
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
timeout --signal=TERM --kill-after=5s 90s bwrap "${args[@]}" \
  python3 /probe.py "$mode" /model.dll --block "$block" >"$run/output.json" 2>"$run/stderr.txt"
status=$?
set -e
printf '%s\n' "$status" >"$run/exit-status.txt"
cat "$run/stderr.txt" >&2
if [[ "$status" == 0 ]]; then
  python3 - "$run/output.json" <<'PY'
import json,sys
from pathlib import Path
report=json.loads(Path(sys.argv[1]).read_text())
print(json.dumps(report['cuda'],indent=2))
PY
fi
exit "$status"
