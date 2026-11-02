#!/usr/bin/env bash
# Launch the windowed DLSSNR desktop demo under Proton inside the same containment
# boundary the probes use. The window appears on the host display through XWayland.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
if [[ "${NR_DESKTOP_CONTAINED:-0}" != 1 ]]; then
  STEAM=/home/michael/.local/share/Steam
  : "${XAUTHORITY:?XWayland authorisation file is required}"
  args=(--unshare-all --die-with-parent --new-session --ro-bind /usr /usr --ro-bind /etc /etc --proc /proc --dev-bind /dev /dev --ro-bind /sys /sys --tmpfs /tmp --tmpfs /run --tmpfs /var --dir /home/michael --ro-bind "$STEAM" "$STEAM" --bind "$ROOT" "$ROOT" --ro-bind /tmp/.X11-unix /tmp/.X11-unix --ro-bind "$XAUTHORITY" /tmp/xauthority --setenv XAUTHORITY /tmp/xauthority --setenv HOME /home/michael --unsetenv DBUS_SESSION_BUS_ADDRESS --unsetenv WAYLAND_DISPLAY --setenv XDG_RUNTIME_DIR /run/user/1000 --dir /run/user/1000 --chdir "$ROOT")
  for path in /bin /sbin /lib /lib64; do
    if [[ -L "$path" ]]; then args+=(--symlink "$(readlink "$path")" "$path"); else args+=(--ro-bind "$path" "$path"); fi
  done
  args+=(--dir /home/michael/.steam --symlink /home/michael/.local/share/Steam/linux64 /home/michael/.steam/sdk64 --symlink /home/michael/.local/share/Steam/linux32 /home/michael/.steam/sdk32)
  exec bwrap "${args[@]}" env NR_DESKTOP_CONTAINED=1 bash "$ROOT/vr/run-nr-desktop.sh" "$@"
fi
export STEAM_COMPAT_CLIENT_INSTALL_PATH=/home/michael/.local/share/Steam
export STEAM_COMPAT_DATA_PATH="$ROOT/vr/prefix"
export PROTON_ENABLE_NVAPI=1
export WINEDLLOVERRIDES='dxgi=n,b'
export SteamGameId=424243 PROTON_LOG=1 PROTON_LOG_DIR="$ROOT/vr/logs"
unset VK_INSTANCE_LAYERS MANGOHUD
cd "$ROOT/vr/probe"
exec 9> "$ROOT/vr/probe/run.lock"
flock -n 9 || { echo "Probe already running" >&2; exit 1; }
# Proton detaches the child's stdio, so the app's own output is captured through a
# cmd redirection into the working directory, exactly as run-feature-probe.sh does.
arguments=""
for argument in "$@"; do arguments+=" $argument"; done
rm -f nr-desktop-output.txt
exec timeout --kill-after=5s "${NR_DESKTOP_TIMEOUT:-300}"s \
  "$ROOT/runtime-copy/run" -- "$ROOT/proton-copy/proton" waitforexitandrun \
  cmd /c "nr-desktop.exe$arguments > nr-desktop-output.txt 2>&1"