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
  capture-reset)
    # Temporal-state probe (see vr/TEMPORAL-STATE.md): identical inputs with a
    # second DLSSNR.Reset at the given frame (default 4); retained under
    # vr/logs/temporal/<label>. Staged under resetmid-* to avoid colliding with
    # a `capture` run of the same label.
    label="${2:-run}"
    [[ "$label" =~ ^[a-z0-9][a-z0-9-]{0,31}$ ]] || { echo 'capture-reset label must match [a-z0-9-]{1,32}' >&2; exit 2; }
    frame="${3:-4}"
    [[ "$frame" =~ ^[1-7]$ ]] || { echo 'capture-reset frame must be an integer in [1,7]' >&2; exit 2; }
    command="feature-probe.exe --capture-reset-mid resetmid-$label $frame > probe-output.txt 2>&1"
    marker='^PASS: reset-mid capture'
    ;;
  capture-history)
    # History-carrier capture (see vr/TEMPORAL-HISTORY.md): the reference
    # protocol plus readback of the model's own history textures ("init" and
    # per frame) and a full resource map. Optional frame arg adds a second
    # DLSSNR.Reset at that frame. Retained under vr/logs/history/<label>.
    label="${2:-run}"
    [[ "$label" =~ ^[a-z0-9][a-z0-9-]{0,31}$ ]] || { echo 'capture-history label must match [a-z0-9-]{1,32}' >&2; exit 2; }
    if [[ -n "${3:-}" ]]; then
      frame="$3"
      [[ "$frame" =~ ^[1-7]$ ]] || { echo 'capture-history frame must be an integer in [1,7]' >&2; exit 2; }
      command="feature-probe.exe --capture-history capture-history-$label $frame > probe-output.txt 2>&1"
    else
      command="feature-probe.exe --capture-history capture-history-$label > probe-output.txt 2>&1"
    fi
    marker='^PASS: history capture'
    ;;
  capture-history-inject|capture-history-sham|capture-history-replay|capture-history-replay2|capture-history-injectreset)
    # Read-side intervention (see vr/TEMPORAL-HISTORY.md): the capture-history
    # protocol plus a history overwrite before frames 2 and 5 — synthetic
    # patterns (inject), the history's own bytes (sham control), or frame 0's
    # history bytes (replay: the one-step recurrence test). Retained under
    # vr/logs/history/<label>.
    label="${2:-run}"
    [[ "$label" =~ ^[a-z0-9][a-z0-9-]{0,31}$ ]] || { echo 'history label must match [a-z0-9-]{1,32}' >&2; exit 2; }
    command="feature-probe.exe --$mode capture-history-$label > probe-output.txt 2>&1"
    marker='^PASS: history capture'
    ;;
  boundary)
    # Decoder-boundary intermediate capture (see vr/DECODER-BOUNDARY.md): the
    # forwarder truncates every evaluation after the decoder launch and writes
    # the boundary words to DLSS_NVAPI_CUT_FILE, which the harness reads back.
    label="${2:-run}"
    [[ "$label" =~ ^[a-z0-9][a-z0-9-]{0,31}$ ]] || { echo 'boundary label must match [a-z0-9-]{1,32}' >&2; exit 2; }
    command="feature-probe.exe --capture-boundary capture-boundary-$label > probe-output.txt 2>&1"
    marker='^PASS: boundary capture'
    export DLSS_NVAPI_LABEL="${DLSS_NVAPI_LABEL:-boundary-$label}"
    export DLSS_NVAPI_ARGS=2
    export DLSS_NVAPI_CUT=1
    export DLSS_NVAPI_CUT_FILE="nvapi-boundary-$label.txt"
    ;;
  ladder)
    # Intermediate-ladder capture (see vr/LADDER.md): truncate the evaluation
    # at recovered slot $slot and read back the model's whole working pool, so
    # consecutive cut points can be diffed into the intervening write-sets.
    label="${2:-run}"
    slot="${3:-}"
    [[ "$label" =~ ^[a-z0-9][a-z0-9-]{0,31}$ ]] || { echo 'ladder label must match [a-z0-9-]{1,32}' >&2; exit 2; }
    # Reject leading zeros outright: bash arithmetic would read 018 as octal
    # while the DLL's strtol parses base 10, so the two bounds could disagree.
    [[ "$slot" =~ ^(0|[1-9][0-9]{0,2})$ ]] || { echo 'ladder slot must be decimal without leading zeros' >&2; exit 2; }
    (( 10#$slot >= 0 && 10#$slot <= 155 )) || { echo 'ladder slot must be 0..155 (recovered 156-launch inventory)' >&2; exit 2; }
    command="feature-probe.exe --capture-ladder capture-ladder-$label > probe-output.txt 2>&1"
    marker='^PASS: ladder capture'
    export DLSS_NVAPI_LABEL="${DLSS_NVAPI_LABEL:-ladder-$label}"
    export DLSS_NVAPI_ARGS=2
    # Index mode arms the cut itself; DLSS_NVAPI_CUT stays unset so the two
    # modes can never both be active.
    export DLSS_NVAPI_CUT_AT="$slot"
    export DLSS_NVAPI_CUT_FILE="nvapi-ladder-$label.txt"
    ;;
  *) echo 'Usage: run-feature-probe.sh [create|evaluate|scene|gpu-scene|capture|capture-reset|capture-history|boundary|ladder LABEL SLOT]' >&2; exit 1 ;;
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
if [[ "$mode" == boundary ]]; then
  rm -f "nvapi-boundary-$label.txt"
fi
if [[ "$mode" == ladder ]]; then
  rm -f "nvapi-ladder-$label.txt"
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

# Reset-mid temporal captures are retained under vr/logs/temporal/<label>.
if [[ "$mode" == capture-reset ]]; then
  mkdir -p "$ROOT/vr/logs/temporal"
  rm -rf "$ROOT/vr/logs/temporal/$label"
  mv "resetmid-$label" "$ROOT/vr/logs/temporal/$label"
  echo "Reset-mid capture stored at vr/logs/temporal/$label"
fi

# History-carrier captures are retained under vr/logs/history/<label>.
if [[ "$mode" == capture-history || "$mode" == capture-history-inject || "$mode" == capture-history-sham || "$mode" == capture-history-replay || "$mode" == capture-history-replay2 || "$mode" == capture-history-injectreset ]]; then
  mkdir -p "$ROOT/vr/logs/history"
  rm -rf "$ROOT/vr/logs/history/$label"
  mv "capture-history-$label" "$ROOT/vr/logs/history/$label"
  echo "History capture stored at vr/logs/history/$label"
fi

# Boundary captures are retained under vr/logs/boundary/<label> together with
# the forwarder's sidecar (the boundary words the harness resolved).
if [[ "$mode" == boundary ]]; then
  mkdir -p "$ROOT/vr/logs/boundary"
  rm -rf "$ROOT/vr/logs/boundary/$label"
  mv "capture-boundary-$label" "$ROOT/vr/logs/boundary/$label"
  if [[ -s "nvapi-boundary-$label.txt" ]]; then
    mv "nvapi-boundary-$label.txt" "$ROOT/vr/logs/boundary/$label/boundary-va.txt"
  else
    echo 'Boundary sidecar missing or empty' >&2
    exit 1
  fi
  echo "Boundary capture stored at vr/logs/boundary/$label"
fi

# Ladder captures are retained under vr/logs/ladder/<label> with the forwarder's
# sidecar (the cut point's parameter block) and the recorded slot.
if [[ "$mode" == ladder ]]; then
  mkdir -p "$ROOT/vr/logs/ladder"
  rm -rf "$ROOT/vr/logs/ladder/$label"
  mv "capture-ladder-$label" "$ROOT/vr/logs/ladder/$label"
  if [[ -s "nvapi-ladder-$label.txt" ]]; then
    mv "nvapi-ladder-$label.txt" "$ROOT/vr/logs/ladder/$label/boundary-va.txt"
  else
    echo 'Ladder sidecar missing or empty' >&2
    exit 1
  fi
  echo "slot=$slot" > "$ROOT/vr/logs/ladder/$label/slot.txt"
  echo "Ladder capture stored at vr/logs/ladder/$label (slot=$slot)"
fi
