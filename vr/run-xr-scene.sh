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
  # Expose only WiVRn compositor IPC, not the session bus or other runtime IPC.
  : "${XDG_RUNTIME_DIR:?}"
  [[ -S "$XDG_RUNTIME_DIR/wivrn/comp_ipc" ]] || { echo 'WiVRn compositor socket unavailable' >&2; exit 1; }
  args+=(--dir /run/user/1000/wivrn --bind "$XDG_RUNTIME_DIR/wivrn/comp_ipc" /run/user/1000/wivrn/comp_ipc)
  exec bwrap "${args[@]}" env DLSS_PROBE_CONTAINED=1 bash "$ROOT/vr/run-xr-scene.sh" "$@"
fi
mode="${1:-off}"
resolution="${2:-720}"
scene="${3:-spheres}"
if (( $# > 3 )); then echo 'Usage: run-xr-scene.sh [off|on] [720|1080|1440] [spheres|room|room-smoke]' >&2; exit 1; fi
case "$resolution" in
  720) executable='xr-scene.exe' ;;
  1080|1440) executable="xr-scene-${resolution}.exe" ;;
  *) echo 'Resolution must be 720, 1080 or 1440 pixels per eye axis' >&2; exit 1 ;;
esac
case "$mode" in off) option='' ;; on) option='--neural' ;; *) echo 'Usage: run-xr-scene.sh [off|on] [720|1080|1440] [spheres|room|room-smoke]'  >&2; exit 1 ;; esac
case "$scene" in
  spheres) scene_option='' ;;
  room) scene_option='--room' ;;
  room-smoke) scene_option='--room-smoke' ;;
  model) scene_option='--room --model' ;;
  model-smoke) scene_option='--room-smoke --model' ;;
  *) echo 'Scene must be spheres, room, room-smoke, model or model-smoke' >&2; exit 1 ;;
esac
export STEAM_COMPAT_CLIENT_INSTALL_PATH=/home/michael/.local/share/Steam
export STEAM_COMPAT_DATA_PATH="$ROOT/vr/prefix"
export PROTON_ENABLE_NVAPI=1
# NGX driver core requires native DXGI/DXVK-NVAPI; no ReShade DLL here.
export WINEDLLOVERRIDES='dxgi=n,b'
export SteamGameId=424244 PROTON_LOG=1 PROTON_LOG_DIR="$ROOT/vr/logs"
export PRESSURE_VESSEL_IMPORT_OPENXR_1_RUNTIMES=1 XR_RUNTIME_JSON="$ROOT/vr/runtime.json"
export VR_PATHREG_OVERRIDE="$ROOT/vr/openvrpaths.vrpath" VR_OVERRIDE="$ROOT/vr/xrizer/xrizer-v0.5"
export WINEDEBUG='+timestamp,+pid,+tid,+seh,+openxr,+vrclient'
unset VK_INSTANCE_LAYERS MANGOHUD
exec 9>"$ROOT/vr/probe/run.lock"
flock -n 9 || { echo 'Another probe uses this prefix' >&2; exit 1; }
cd "$ROOT/vr/probe"
mkdir -p "$ROOT/vr/logs"
[[ -f "$executable" ]] || { echo "Missing executable: $executable" >&2; exit 1; }
echo "XR test mode=$mode resolution=${resolution}x${resolution} per eye"
rm -f xr-output.txt "$ROOT/vr/logs/steam-424244.log"
status=0
timeout --kill-after=5s 150s "$ROOT/runtime-copy/run" -- "$ROOT/proton-copy/proton" waitforexitandrun cmd /c "$executable $option $scene_option > xr-output.txt 2>&1" || status=$?
if [[ -f xr-output.txt ]]; then cat xr-output.txt; fi
if (( status != 0 )); then
  if (( status == 124 || status == 137 )); then echo "XR test timed out (status=$status)" >&2; else echo "XR test failed (status=$status)" >&2; fi
  exit "$status"
fi
if ! grep -Eq '^PASS: XR submitted frames=[1-9][0-9]* neural=[01]\.' xr-output.txt; then echo 'No validated XR completion' >&2; exit 1; fi
if [[ ! -s "$ROOT/vr/logs/steam-424244.log" ]] || grep -Eq 'Unhandled (exception|page fault)|Assertion failed|wine: Assertion' "$ROOT/vr/logs/steam-424244.log"; then echo 'Proton diagnostic failure' >&2; exit 1; fi
