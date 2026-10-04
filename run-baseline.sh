#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "$0")" && pwd)"
STEAM="/home/michael/.local/share/Steam"
export STEAM_COMPAT_CLIENT_INSTALL_PATH="$STEAM"
export STEAM_COMPAT_DATA_PATH="$HERE/prefix-0"
export PROTON_ENABLE_NVAPI=1
if [[ "${DLSS_TEST_NEURAL:-0}" == 1 ]]; then
  export WINEDLLOVERRIDES='dxgi=n,b;concrt140,msvcp140,vcruntime140,vcruntime140_1=n,b'
else
  export WINEDLLOVERRIDES='dxgi=b;concrt140,msvcp140,vcruntime140,vcruntime140_1=n,b'
fi
export PROTON_LOG=1
export PROTON_LOG_DIR="$HERE/logs"
export DXVK_NVAPI_LOG_PATH="$HERE/logs"
export DXVK_NVAPI_LOG_LEVEL="${DLSS_TEST_LOG_LEVEL:-info}"
if [[ "${DLSS_BENCHMARK:-0}" == 1 ]]; then
  export MANGOHUD=1
  export VK_LAYER_PATH="$HERE/tools/mangohud/layers"
  export VK_INSTANCE_LAYERS=VK_LAYER_MANGOHUD_overlay_x86_64
fi
mkdir -p "$HERE/logs" "$HERE/prefix-0"
cd "$HERE/demo/DLSS_Sample_App/bin/ngx_dlss_demo"
exec "$HERE/runtime-copy/run" -- "$HERE/proton-copy/proton" waitforexitandrun ./ngx_dlss_demo.exe -d3d12 -width 1280 -height 720 "$@"
