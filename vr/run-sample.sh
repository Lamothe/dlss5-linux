#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "$0")" && pwd)"
ROOT="$(dirname -- "$HERE")"
export STEAM_COMPAT_CLIENT_INSTALL_PATH=/home/michael/.local/share/Steam
export STEAM_COMPAT_DATA_PATH="$HERE/prefix"
export PRESSURE_VESSEL_IMPORT_OPENXR_1_RUNTIMES=1
export XR_RUNTIME_JSON="$HERE/runtime.json"
export VR_PATHREG_OVERRIDE="$HERE/openvrpaths.vrpath"
export VR_OVERRIDE="$HERE/xrizer/xrizer-v0.5"
export WINEDLLOVERRIDES='dxgi=b;concrt140,msvcp140,vcruntime140,vcruntime140_1=n,b'
export SteamGameId=424242
export PROTON_LOG=1 PROTON_LOG_DIR="$HERE/logs"
export WINEDEBUG='+timestamp,+pid,+tid,+seh,+openxr,+vrclient'
unset VK_INSTANCE_LAYERS MANGOHUD PROTON_ENABLE_NVAPI
mkdir -p "$HERE/logs"
# Redirect inside Windows: this GUI-subsystem sample otherwise loses console output.
APP="Z:${HERE//\//\\}\\sample\\hello_xr.exe"
LOG="Z:${HERE//\//\\}\\logs\\headset-sample.txt"
exec "$ROOT/runtime-copy/run" -- "$ROOT/proton-copy/proton" waitforexitandrun cmd /c "$APP -g D3D12 -vc Stereo -s Local -v > $LOG 2>&1"
