#!/usr/bin/env bash
# Headless native Linux only. No Windows host code and no CUDA kernel launch.
set -euo pipefail
ulimit -c 0 # Defence in depth; the probe also sets PR_SET_DUMPABLE=0 before reading.
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
OUT="$ROOT/research/native-linux/runs"
DLL="$ROOT/model/nvngx_dlssnr.dll"
mode="${1:-inventory}"
case "$mode" in inventory|module-load|negative) ;; *) echo 'Usage: run-native-runtime-probe.sh [inventory|module-load|negative]' >&2; exit 1 ;; esac
for command in python3 bwrap timeout flock; do command -v "$command" >/dev/null; done
[[ -f "$DLL" ]]
# The Python inventory verifies the pinned hash AFTER disabling dumpability.
mkdir -p "$OUT"
# Same lock as the native NGX probe; do not overlap GPU probes.
exec 9>"$OUT/probe.lock"
flock -n 9 || { echo 'Another native probe is running' >&2; exit 1; }
run="$(mktemp -d "$OUT/runtime-$mode-XXXXXXXX")"
# Freeze the actual source supplied to the sandbox; retain hashes and snapshots.
mkdir "$run/source" "$run/evidence"
cp -- "$ROOT/vr/run-native-runtime-probe.sh" "$run/source/run-native-runtime-probe.sh"
cp -- "$ROOT/vr/native-runtime-probe.py" "$run/source/native-runtime-probe.py"
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
  --ro-bind "$run/source/native-runtime-probe.py" /probe.py
  --ro-bind "$DLL" /model.dll --bind "$run/evidence" /work --chdir /work
  --clearenv --setenv PATH /usr/bin --setenv HOME /tmp
  --setenv CUDA_MODULE_LOADING EAGER --setenv CUDA_CACHE_DISABLE 1
  --setenv PYTHONDONTWRITEBYTECODE 1)
# Inventory requires no GPU. Only native CUDA module-load modes expose NVIDIA nodes.
if [[ "$mode" != inventory ]]; then
  for path in /dev/nvidiactl /dev/nvidia0 /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
    [[ ! -e "$path" ]] || args+=(--dev-bind "$path" "$path")
  done
fi
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
if 'cuda' in report:
    print(json.dumps(report['cuda'],indent=2))
else:
    print(f"Inventory: {report['weights']['record_count']} weight records, {report['weights']['payload_bytes']} payload bytes; {len(report['fatbins'])} CUDA containers")
PY
fi
exit "$status"
