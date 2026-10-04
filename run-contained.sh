#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "$0")" && pwd)"
STEAM="/home/michael/.local/share/Steam"
: "${XAUTHORITY:?XWayland authorisation file is required}"
args=(--unshare-all --die-with-parent --new-session --ro-bind /usr /usr --ro-bind /etc /etc --proc /proc --dev-bind /dev /dev --ro-bind /sys /sys --tmpfs /tmp --tmpfs /run --tmpfs /var --dir /home/michael --ro-bind "$STEAM" "$STEAM" --bind "$HERE" "$HERE" --ro-bind /tmp/.X11-unix /tmp/.X11-unix --ro-bind "$XAUTHORITY" /tmp/xauthority --setenv XAUTHORITY /tmp/xauthority --setenv HOME /home/michael --unsetenv DBUS_SESSION_BUS_ADDRESS --unsetenv WAYLAND_DISPLAY --setenv XDG_RUNTIME_DIR /run/user/1000 --dir /run/user/1000 --chdir "$HERE")
for path in /bin /sbin /lib /lib64; do
  if [[ -L "$path" ]]; then args+=(--symlink "$(readlink "$path")" "$path"); else args+=(--ro-bind "$path" "$path"); fi
done
if [[ "${1:-}" == '--check' ]]; then
  exec bwrap "${args[@]}" bash -c 'test ! -e /home/michael/.ssh && test ! -e /home/michael/Projects/atomic && test ! -e /run/user/1000/bus && test -f ./run-baseline.sh && echo "Host home and session bus hidden"; python3 -c "import socket; s=socket.socket(); s.settimeout(2); print(s.connect_ex((\"1.1.1.1\",443)))"'
fi
exec bwrap "${args[@]}" bash ./run-baseline.sh "$@"
