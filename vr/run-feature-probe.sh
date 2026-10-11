#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
if [[ "${DLSS_PROBE_CONTAINED:-0}" != 1 ]]; then
  # Reuse the verified desktop containment boundary; no headset socket needed.
  STEAM=/home/michael/.local/share/Steam
  : "${XAUTHORITY:?XWayland authorisation file is required}"
  args=(--unshare-all --die-with-parent --new-session --ro-bind /usr /usr --ro-bind /etc /etc --proc /proc --dev-bind /dev /dev --ro-bind /sys /sys --tmpfs /tmp --tmpfs /run --tmpfs /var --dir /home/michael --ro-bind "$STEAM" "$STEAM" --bind "$ROOT" "$ROOT" --ro-bind /tmp/.X11-unix /tmp/.X11-unix --ro-bind "$XAUTHORITY" /tmp/xauthority --setenv XAUTHORITY /tmp/xauthority --setenv HOME /home/michael --unsetenv DBUS_SESSION_BUS_ADDRESS --unsetenv WAYLAND_DISPLAY --setenv XDG_RUNTIME_DIR /run/user/1000 --dir /run/user/1000 --chdir "$ROOT")
  for path in /bin /sbin /lib /lib64; do
    if [[ -L "$path" ]]; then args+=(--symlink "$(readlink "$path")" "$path"); else args+=(--ro-bind "$path" "$path"); fi
  done
  # Proton's Steam shim locates its native library through these links. Expose
  # only SDK directories, not host Steam IPC pipes/tokens or other home files.
  args+=(--dir /home/michael/.steam --symlink /home/michael/.local/share/Steam/linux64 /home/michael/.steam/sdk64 --symlink /home/michael/.local/share/Steam/linux32 /home/michael/.steam/sdk32)
  exec bwrap "${args[@]}" env DLSS_PROBE_CONTAINED=1 bash "$ROOT/vr/run-feature-probe.sh" "$@"
fi
mode="${1:-create}"
case "$mode" in
  aa-corrupt) command='aa-probe.exe --bad-aa-output > probe-output.txt 2>&1'; marker='^REJECT:' ;;
  aa) command='aa-probe.exe > probe-output.txt 2>&1'; marker='^PASS: offscreen composition AA' ;;
  aa-negative) command='aa-probe.exe --skip-aa > probe-output.txt 2>&1'; marker='^REJECT:' ;;
  city-glass-negative) command='mesh-probe.exe --city --skip-glass > probe-output.txt 2>&1'; marker='^REJECT: glass blend' ;;
  automotive) command='mesh-probe.exe --automotive > probe-output.txt 2>&1'; marker='^PASS: offscreen mesh colour/depth/motion,' ;;
  automotive-negative) command='mesh-probe.exe --automotive --skip-mesh > probe-output.txt 2>&1'; marker='^REJECT: static coverage' ;;
  supplied-city) command='mesh-probe.exe --supplied-city > probe-output.txt 2>&1'; marker='^PASS: offscreen mesh colour/depth/motion,' ;;
  supplied-city-negative) command='mesh-probe.exe --supplied-city --skip-mesh > probe-output.txt 2>&1'; marker='^REJECT: static coverage' ;;
  city) command='mesh-probe.exe --city > probe-output.txt 2>&1'; marker='^PASS: offscreen mesh colour/depth/motion,' ;;
  city-negative) command='mesh-probe.exe --city --skip-mesh > probe-output.txt 2>&1'; marker='^REJECT: static coverage' ;;
  mesh) command='mesh-probe.exe > probe-output.txt 2>&1'; marker='^PASS: offscreen mesh colour/depth/motion,' ;;
  mesh-negative) command='mesh-probe.exe --skip-mesh > probe-output.txt 2>&1'; marker='^REJECT: static coverage' ;;
  create) command='feature-probe.exe > probe-output.txt 2>&1'; marker='^PASS: two distinct feature handles created and GPU initialization completed\.' ;;
  tracked-control) command='feature-probe.exe --tracked-control > probe-output.txt 2>&1'; marker='^PASS: GPU scene evaluation frames=12 evaluations=12,12 validated_rgb_outputs=12,12\.' ;;
  gpu-scene) command='feature-probe.exe --gpu-scene > probe-output.txt 2>&1'; marker='^PASS: GPU scene evaluation frames=12 evaluations=12,12 validated_rgb_outputs=12,12\.' ;;
  scene) command='feature-probe.exe --scene > probe-output.txt 2>&1'; marker='^PASS: rendered scene evaluation frames=12 evaluations=12,12 validated_rgb_outputs=12,12\.' ;;
  evaluate) command='feature-probe.exe --evaluate > probe-output.txt 2>&1'; marker='^PASS: repeated two-context evaluation frames=60 evaluations=60,60 validated_rgb_outputs=60,60\.' ;;
  *) echo 'Usage: run-feature-probe.sh [create|evaluate|scene|gpu-scene]' >&2; exit 1 ;;
esac
export STEAM_COMPAT_CLIENT_INSTALL_PATH=/home/michael/.local/share/Steam
export STEAM_COMPAT_DATA_PATH="$ROOT/vr/prefix"
export PROTON_ENABLE_NVAPI=1
export WINEDLLOVERRIDES='dxgi=n,b'
export SteamGameId=424243 PROTON_LOG=1 PROTON_LOG_DIR="$ROOT/vr/logs"
unset VK_INSTANCE_LAYERS MANGOHUD
cd "$ROOT/vr/probe"
exec 9> "$ROOT/vr/probe/run.lock"
flock -n 9 || { echo "Probe already running" >&2; exit 1; }
rm -f probe-output.txt "$ROOT/vr/logs/steam-424243.log"
limit=60
if [[ "$mode" == scene ]]; then limit=300; fi
timeout --kill-after=5s "$limit"s "$ROOT/runtime-copy/run" -- "$ROOT/proton-copy/proton" waitforexitandrun cmd /c "$command"

# cmd/Proton can return zero after an application failure. Validate evidence too.
test -s probe-output.txt
test -s "$ROOT/vr/logs/steam-424243.log"
grep -q "$marker" probe-output.txt
if grep -Eq "Unhandled exception:|_wassert|unable to load native steamclient" "$ROOT/vr/logs/steam-424243.log"; then
  echo "Probe failed: exception/assertion in Proton log" >&2
  exit 1
fi

# Reject read errors as well as matching diagnostic errors.
status=0
grep -Eq "Unhandled exception:|_wassert|unable to load native steamclient" "$ROOT/vr/logs/steam-424243.log" || status=$?
[[ "$status" == 1 ]]
