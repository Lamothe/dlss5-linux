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
WINROOT="Z:${ROOT//\//\\}"
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
  capture)
    label="${2:-run}"
    [[ "$label" =~ ^[a-z0-9][a-z0-9-]{0,31}$ ]] || { echo 'capture label must match [a-z0-9-]{1,32}' >&2; exit 2; }
    command="feature-probe.exe --capture capture-$label > probe-output.txt 2>&1"
    marker='^PASS: reference capture'
    ;;
  *) echo 'Usage: run-feature-probe.sh [create|evaluate|scene|gpu-scene]' >&2; exit 1 ;;
esac
export STEAM_COMPAT_CLIENT_INSTALL_PATH=/home/michael/.local/share/Steam
export STEAM_COMPAT_DATA_PATH="$ROOT/vr/prefix"
export PROTON_ENABLE_NVAPI=1
export WINEDLLOVERRIDES='dxgi=n,b'
# Optional tracing (see vr/NVAPI-DISPATCH.md): DLSS_NVAPI_LABEL enables the
# nvapi64.dll QueryInterface forwarder trace (retained under
# vr/logs/capture/nvapi-<label>.log); DLSS_NGX_LOG=1 enables the model's own
# logging (level via DLSS_NGX_LOG_LEVEL, default 2).
if [[ -n "${DLSS_NVAPI_LABEL:-}" ]]; then
  [[ "$DLSS_NVAPI_LABEL" =~ ^[a-z0-9][a-z0-9-]{0,31}$ ]] || { echo 'DLSS_NVAPI_LABEL must match [a-z0-9-]{1,32}' >&2; exit 2; }
  export DLSS_NVAPI_LOG="nvapi-trace-${DLSS_NVAPI_LABEL}.log"
  # Proton redeploys its own nvapi64.dll into the prefix at startup, so the
  # forwarder replaces it inside the Wine session as the first command.
  command="copy /y ${WINROOT}\\vr\\capture\\nvapi64-trace.dll C:\\windows\\system32\\nvapi64.dll >nul && $command"
fi
# Optional launch-argument capture (see vr/NVAPI-ARGS.md): level 1 decodes the
# LaunchCubinShader ABI, level 2 also dumps each parameter block. Level 0 (or
# unset) reads no arguments. Needs the forwarder trace above and trampolines;
# the DLL fails closed without them as well.
if [[ -n "${DLSS_NVAPI_ARGS:-}" ]]; then
  [[ "$DLSS_NVAPI_ARGS" == 0 || "$DLSS_NVAPI_ARGS" == 1 || "$DLSS_NVAPI_ARGS" == 2 ]] || { echo 'DLSS_NVAPI_ARGS must be 0, 1 or 2' >&2; exit 2; }
  if [[ "$DLSS_NVAPI_ARGS" != 0 ]]; then
    [[ -n "${DLSS_NVAPI_LABEL:-}" ]] || { echo 'DLSS_NVAPI_ARGS requires DLSS_NVAPI_LABEL' >&2; exit 2; }
    [[ "${DLSS_NVAPI_WRAP:-1}" != 0 ]] || { echo 'DLSS_NVAPI_ARGS requires DLSS_NVAPI_WRAP=1' >&2; exit 2; }
    export DLSS_NVAPI_ARGS
  fi
fi
if [[ -n "${DLSS_NGX_LOG:-}" ]]; then
  mkdir -p "$ROOT/vr/logs/capture/ngxlog"
  export __NGX_LOG_LEVEL="${DLSS_NGX_LOG_LEVEL:-2}"
  export __NGX_LOG_PATH_OVERRIDE="${WINROOT}\\vr\\logs\\capture\\ngxlog"
  export __NGX_ENABLE_OVERRIDE_LOG_PATH=1
fi
export SteamGameId=424243 PROTON_LOG=1 PROTON_LOG_DIR="$ROOT/vr/logs"
unset VK_INSTANCE_LAYERS MANGOHUD
cd "$ROOT/vr/probe"
exec 9> "$ROOT/vr/probe/run.lock"
flock -n 9 || { echo "Probe already running" >&2; exit 1; }
rm -f probe-output.txt "$ROOT/vr/logs/steam-424243.log"
if [[ -n "${DLSS_NVAPI_LABEL:-}" ]]; then
  rm -f "nvapi-trace-${DLSS_NVAPI_LABEL}.log"
fi
limit=60
if [[ "$mode" == scene ]]; then limit=300; fi
run_status=0
timeout --kill-after=5s "$limit"s "$ROOT/runtime-copy/run" -- "$ROOT/proton-copy/proton" waitforexitandrun cmd /c "$command" || run_status=$?

# Retain the NvAPI trace before any validation so failed runs keep evidence
# (crashes and timeouts included), then propagate the run status.
if [[ -n "${DLSS_NVAPI_LABEL:-}" ]]; then
  if [[ -s "nvapi-trace-${DLSS_NVAPI_LABEL}.log" ]]; then
    mkdir -p "$ROOT/vr/logs/capture"
    mv "nvapi-trace-${DLSS_NVAPI_LABEL}.log" "$ROOT/vr/logs/capture/"
    echo "NvAPI trace stored at vr/logs/capture/nvapi-trace-${DLSS_NVAPI_LABEL}.log"
  else
    echo 'NvAPI trace requested but missing or empty' >&2
    exit 1
  fi
fi
if [[ "$run_status" != 0 ]]; then
  echo "Probe run failed with status $run_status" >&2
  exit "$run_status"
fi

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

# Accepted capture runs are retained under vr/logs/reference/<label>.
if [[ "$mode" == capture ]]; then
  mkdir -p "$ROOT/vr/logs/reference"
  rm -rf "$ROOT/vr/logs/reference/$label"
  mv "capture-$label" "$ROOT/vr/logs/reference/$label"
  echo "Reference capture stored at vr/logs/reference/$label"
fi
