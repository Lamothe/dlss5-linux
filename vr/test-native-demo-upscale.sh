#!/bin/bash
# test-native-demo-upscale.sh — the demonstration transform's quality
# family (NOT DLSS5).
#
#   Usage: vr/test-native-demo-upscale.sh
#
# Builds and runs vr/tests/native-demo-upscale-test.cpp: the measured
# quality claim (PSNR/SSIM of the demonstration transform against the
# generated source AND against a bilinear baseline, with the parameter
# sweep printed as the tuning record), determinism, the unsharp
# semantics and the rejection classes. Runs at -O2 and -O0.
#
# DATA-FREE: the test image is generated deterministically in the test;
# no external assets, no NVIDIA data, no runtime probes.
set -u
cd "$(dirname "$0")/.."

CC=${CXX:-c++}
FLAGS="-std=c++20 -ffp-contract=off -Wall -Wextra -Ivr"
failed=0

for opt in -O2 -O0; do
  echo "=== demo-upscale at $opt ==="
  if ! $CC $FLAGS $opt vr/tests/native-demo-upscale-test.cpp \
       -o /tmp/native-demo-upscale-test 2>&1; then
    echo "FAILED to build at $opt" >&2
    failed=1
    continue
  fi
  if ! /tmp/native-demo-upscale-test; then
    echo "FAILED at $opt" >&2
    failed=1
  fi
done

if [ $failed -ne 0 ]; then
  echo "test-native-demo-upscale: FAILED" >&2
  exit 1
fi
echo "test-native-demo-upscale: OK"
