#!/bin/bash
# validate-weights.sh — the single documented reference-validation step.
#
#   Usage: vr/validate-weights.sh <weights.nwf>
#
# Accepts a weight file (the NWF1 container; a source-format file is
# converted first - see NATIVE-WEIGHTS-FILE.md, "accepting a source-format
# file") and, in one command:
#
#   1. regenerates the plan's request manifest from the plan itself
#      (demo-native --dump-weights; no static table to drift),
#   2. verifies the file's structure, content and compatibility against
#      that manifest (weights-verify: every mismatch named),
#   3. runs the reference validation: the full network through the
#      file-backed provider on the CPU and on the GPU projection path,
#      requiring byte-identical artefacts between the two (the G2
#      property with these weights), and checks the mandatory label.
#
# Exit 0 = the file is accepted and reference-validated. Exit non-zero =
# rejected or invalid, with every problem named on stderr.
#
# DATA-FREE: no NVIDIA assets, no proprietary payloads, no runtime
# probes. The manifest comes from the synthetic plan; the candidate's
# content is never compared to any reference values - the validation is
# shape/metadata compatibility plus CPU-vs-GPU byte identity.
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)

if [ $# -ne 1 ]; then
  echo "usage: vr/validate-weights.sh <weights.nwf>" >&2
  exit 2
fi
FILE=$1
if [ ! -s "$FILE" ]; then
  echo "validate-weights: no such file: $FILE" >&2
  exit 2
fi

TMP=$(mktemp -d)

echo "validate-weights: building the harness..."
# The demo is built with the Vulkan projection path when the toolchain is
# present (the GPU reference run below), else CPU-only. The two builds are
# distinct so the GPU-skip reporting can tell them apart.
GPU_BUILD=0
if c++ -std=c++20 -pthread -O2 -ffp-contract=off -Ivr \
     -DDLSS5_DEMO_VULKAN vr/demo-native.cpp -o "$TMP/demo" -lvulkan \
     2> /dev/null; then
  GPU_BUILD=1
else
  c++ -std=c++20 -pthread -O2 -ffp-contract=off -Ivr \
      vr/demo-native.cpp -o "$TMP/demo" || exit 1
fi
c++ -std=c++20 -O2 -ffp-contract=off -Ivr \
    vr/weights-verify.cpp -o "$TMP/verify" || exit 1

echo "validate-weights: regenerating the plan's request manifest..."
"$TMP/demo" --out "$TMP/manifest-run" --frames 1 --reset-at -1 --threads 2 \
    --dump-weights "$TMP/manifest.nwf" > "$TMP/manifest.log" 2>&1
if [ $? -ne 0 ]; then
  echo "validate-weights: manifest generation FAILED (see $TMP/manifest.log)" >&2
  exit 1
fi

echo "validate-weights: verifying the file against the manifest..."
if ! "$TMP/verify" "$TMP/manifest.nwf" "$FILE"; then
  echo "validate-weights: REJECTED" >&2
  exit 1
fi

echo "validate-weights: reference validation (CPU)..."
"$TMP/demo" --out "$TMP/cpu" --frames 1 --reset-at -1 --threads 2 \
    --weights "$FILE" > "$TMP/cpu.log" 2>&1
if [ $? -ne 0 ]; then
  echo "validate-weights: CPU reference run FAILED (see $TMP/cpu.log)" >&2
  exit 1
fi

GPU_NOTE="GPU path skipped"
if [ $GPU_BUILD -eq 0 ]; then
  GPU_NOTE="GPU path skipped (no Vulkan toolchain for the build)"
elif ! command -v glslc > /dev/null 2>&1; then
  GPU_NOTE="GPU path skipped (no shader compiler on this host)"
elif ! glslc --target-env=vulkan1.1 -O vr/shaders/matmul_e4m3.comp \
     -o "$TMP/matmul.spv" 2> "$TMP/glslc.log"; then
  GPU_NOTE="GPU path skipped (shader compile failed - see $TMP/glslc.log)"
else
  echo "validate-weights: reference validation (GPU projection)..."
  "$TMP/demo" --out "$TMP/gpu" --frames 1 --reset-at -1 --threads 2 \
      --weights "$FILE" --gpu "$TMP/matmul.spv" > "$TMP/gpu.log" 2>&1
  if [ $? -ne 0 ]; then
    if grep -q "no Vulkan support\|no compute-capable Vulkan device" \
        "$TMP/gpu.log"; then
      GPU_NOTE="GPU path skipped (no usable Vulkan device at run time)"
    else
      echo "validate-weights: GPU reference run FAILED (see $TMP/gpu.log)" >&2
      exit 1
    fi
  else
    for art in before after contact; do
      if ! cmp -s "$TMP/cpu/$art.ppm" "$TMP/gpu/$art.ppm"; then
        echo "validate-weights: CPU-vs-GPU $art.ppm DIFFERS - REJECTED" >&2
        exit 1
      fi
    done
    GPU_NOTE="GPU projection path: byte-identical to the CPU path"
  fi
fi

if ! grep -q "synthetic weights" "$TMP/cpu.log" 2> /dev/null; then
  echo "validate-weights: label check FAILED (the mandatory label is missing)" >&2
  exit 1
fi

echo "validate-weights: ACCEPTED"
echo "  file:      $FILE"
echo "  manifest:  $TMP/manifest.nwf (the plan request set)"
echo "  reference: CPU run completed; $GPU_NOTE"
echo "  label:     present (synthetic-weights honesty label)"
echo "  evidence:  $TMP"
