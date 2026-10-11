#!/usr/bin/env bash
# Native-only, headless capability probe. Official SDK dependencies remain local.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
SDK="$ROOT/research/native-linux/sdk"
OUT="$ROOT/research/native-linux/runs"
ICD=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
check=0
if [[ "${1:-}" == --check ]]; then check=1; shift; fi
mode="${1:-with-sr}"
if (( $# > 1 )); then echo 'Too many arguments' >&2; exit 1; fi
case "$mode" in with-sr|without-sr) ;; *) echo 'Usage: run-native-ngx-probe.sh [--check] [with-sr|without-sr]' >&2; exit 1 ;; esac
for command in g++ pkg-config bwrap timeout flock sha256sum cmp; do command -v "$command" >/dev/null; done
[[ -f "$ICD" ]]
# Pins transcribed from the seven-line research manifest. Never trust a mutable
# manifest as the authority for the local proprietary dependency bytes.
pins=$(cat <<'PINS'
7561f16cab74e2b7ccf791bf44bb9cc43abb44bcf78f33c5663520ad4e857e02  research/native-linux/sdk/lib/libnvidia-ngx-dlss.so.310.9.1
3321c4ca9adf71f345550d10bf2148ed38003151b03ec5f81c99d00949ab8202  research/native-linux/sdk/lib/libnvsdk_ngx.a
ea23f33497cd274860d1c25a97644fce807dcb0037c594547203343103fad03e  research/native-linux/sdk/include/nvsdk_ngx_defs.h
0a24d0861ace7d6b9362a67b7f08bea1b33ea123c5ce730b19133de8a891d031  research/native-linux/sdk/include/nvsdk_ngx_defs_vk.h
dc38e7467cf415379c9d12ae1b6e4a494c453ed92720fb53e92aecb523e7b848  research/native-linux/sdk/include/nvsdk_ngx.h
943bc8cc5cdae03b6303016fbad3183636f2335ae27a2d18776798c3b4efabbc  research/native-linux/sdk/include/nvsdk_ngx_params.h
2d364ce7132881eb669e9498fd570d74cba563b1fbebc82c235f8e1ad2dd8b6d  research/native-linux/sdk/include/nvsdk_ngx_vk.h
PINS
)
if ! cmp -s <(printf '%s\n' "$pins") "$ROOT/research/native-linux/sdk-sha256.txt"; then
  echo 'SDK hash manifest differs from reviewed pins' >&2; exit 1
fi
(cd "$ROOT"; printf '%s\n' "$pins" | sha256sum --check --strict)
mkdir -p "$OUT"
exec 9>"$OUT/probe.lock"
flock -n 9 || { echo 'Native probe already running' >&2; exit 1; }
run="$(mktemp -d "$OUT/$mode-XXXXXXXX")"
# Per-run executable and control directory: no stale snippets in a shared folder.
mkdir "$run/empty"
if [[ "$mode" == with-sr ]]; then
  # A mountpoint placeholder is required before the parent directory is read-only.
  : >"$run/empty/libnvidia-ngx-dlss.so.310.9.1"
fi
g++ -std=c++17 -Wall -Wextra -Werror -Wno-missing-field-initializers -O2 \
  $(pkg-config --cflags vulkan) -I"$SDK/include" "$ROOT/vr/native-ngx-probe.cpp" \
  "$SDK/lib/libnvsdk_ngx.a" $(pkg-config --libs vulkan) -ldl -pthread \
  -o "$run/native-ngx-probe"
{
  printf 'UTC=%s\nmode=%s\ncontainment_check=%s\n' "$(date -u +%FT%TZ)" "$mode" "$check"
  printf 'Compiler: '; g++ --version | head -1
  printf 'Vulkan pkg-config: '; pkg-config --modversion vulkan
  printf 'Reviewed SDK dependency hashes:\n%s\n' "$pins"
  printf 'Run source/executable/launcher/ICD hashes:\n'
  sha256sum "$ROOT/vr/native-ngx-probe.cpp" "$ROOT/vr/run-native-ngx-probe.sh" "$run/native-ngx-probe" "$ICD"
  printf 'Dynamic dependencies:\n'; ldd "$run/native-ngx-probe"
  # Driver dependencies dynamically loaded by Vulkan/NGX are not in ldd output.
  for library in libnvidia-ngx.so.1 libGLX_nvidia.so.0 libcuda.so.1 libvulkan.so.1; do
    path=$(ldconfig -p | awk -v name="$library" '$1 == name && /x86-64/ {print $NF; exit}')
    if [[ -n "$path" ]]; then printf 'Driver/link dependency %s -> %s\n' "$library" "$(readlink -f "$path")"; sha256sum "$path"; fi
  done
} >"$run/metadata.txt"
args=(--unshare-all --die-with-parent --new-session
  --ro-bind /usr /usr --ro-bind /etc /etc --proc /proc --dev /dev
  --ro-bind /sys /sys --tmpfs /tmp --tmpfs /run --tmpfs /var --tmpfs /home
  --ro-bind "$run/native-ngx-probe" /probe
  --ro-bind "$run/empty" /sdk --bind "$run" /work --chdir /work
  --clearenv --setenv PATH /usr/bin --setenv HOME /tmp
  --setenv VK_DRIVER_FILES "$ICD")
if [[ "$mode" == with-sr ]]; then
  args+=(--ro-bind "$SDK/lib/libnvidia-ngx-dlss.so.310.9.1" /sdk/libnvidia-ngx-dlss.so.310.9.1)
fi
# Synthetic /dev hides host shared memory and all unrelated host devices.
# This single-GPU probe needs only NVIDIA compute/control nodes, not DRM/input.
for device in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm; do
  [[ -c "$device" ]] || { echo "Missing NVIDIA device $device" >&2; exit 1; }
  args+=(--dev-bind "$device" "$device")
  stat -c 'Device %n major/minor=%t:%T' "$device" >>"$run/metadata.txt"
done
for path in /bin /sbin /lib /lib64; do
  if [[ -L "$path" ]]; then args+=(--symlink "$(readlink "$path")" "$path"); else args+=(--ro-bind "$path" "$path"); fi
done
# No host home, display, session bus, Steam or network. GPU/driver security
# isolation is NOT guaranteed. --check inspects paths/network; never opens a GPU.
command=(/probe /sdk "$mode")
if (( check )); then
  command=(/usr/bin/python3 -c '
import errno, os, socket, stat, sys
assert not os.path.exists("/home/michael")
assert not os.path.exists("/run/user")
assert not os.path.exists("/tmp/.X11-unix")
assert os.listdir("/dev/shm") == [], "shared memory must be fresh"
assert set(os.listdir("/dev")) <= {"core", "fd", "full", "null", "ptmx", "pts", "random", "shm", "stderr", "stdin", "stdout", "tty", "urandom", "zero", "nvidia0", "nvidiactl", "nvidia-uvm"}
for path in ("/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm"):
    assert stat.S_ISCHR(os.stat(path).st_mode)
assert os.listdir("/sdk") == (["libnvidia-ngx-dlss.so.310.9.1"] if sys.argv[1] == "with-sr" else [])
s = socket.socket()
s.settimeout(2)
try:
    s.connect(("1.1.1.1", 443))
except OSError as error:
    assert error.errno == errno.ENETUNREACH, str(error)
else:
    raise AssertionError("network reachable")
finally:
    s.close()
print("CONTAINMENT CHECK PASS; no GPU API calls")
' "$mode")
fi
printf 'Run evidence: %s\n' "$run"
set +e
timeout --signal=TERM --kill-after=5s 90s bwrap "${args[@]}" \
  "${command[@]}" >"$run/output.txt" 2>&1
status=$?
set -e
cat "$run/output.txt"
printf '%s\n' "$status" >"$run/exit-status.txt"
exit "$status"
