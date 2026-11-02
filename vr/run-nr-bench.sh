#!/usr/bin/env bash
# Cost measurement for DLSSNR under Proton (vr/nr-bench.c) through the same
# containment boundary and prefix as the desktop demo.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
export NR_EXE=nr-bench.exe
export NR_OUT="${NR_OUT:-nr-bench-output.txt}"
export NR_DESKTOP_TIMEOUT="${NR_DESKTOP_TIMEOUT:-1800}"
exec bash "$ROOT/vr/run-nr-desktop.sh" "$@"