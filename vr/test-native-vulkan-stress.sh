#!/usr/bin/env bash
# GPU-path hardening regression (performance stage 4): init/teardown cycles,
# a long mixed-shape resident run, the weight cache's content-hash mismatch
# and capacity exhaustion paths, and failure paths that must not corrupt
# later runs. Every result is checked bit-identical to the CPU reference.
# Clean SKIP without a Vulkan device or glslc (recorded environment gaps).
# The GLSL shader is compiled at run time (no compiled artefacts in the
# repository). Speed numbers are real; with synthetic weights the OUTPUT is
# arbitrary (no image-quality claim). No headset, proprietary payload or
# Python required.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
CXX="${CXX:-c++}"
SRC="$ROOT/vr/tests/native-vulkan-stress-test.cpp"
temp="$(mktemp -d)"
trap 'rm -rf "$temp"' EXIT
COMMON=(-std=c++20 -pthread -Wall -Wextra -Werror -ffp-contract=off -I"$ROOT/vr")

if ! command -v glslc >/dev/null 2>&1; then
    echo "native-vulkan-stress: SKIP (no glslc shader compiler; recorded environment gap)"
    exit 0
fi
SPIRV="$temp/matmul_e4m3.spv"
glslc --target-env=vulkan1.1 -O "$ROOT/vr/shaders/matmul_e4m3.comp" -o "$SPIRV"

"${CXX}" "${COMMON[@]}" -O2 "$SRC" -o "$temp/test-o2" -lvulkan
"$temp/test-o2" "$SPIRV"
"${CXX}" "${COMMON[@]}" -O0 "$SRC" -o "$temp/test-o0" -lvulkan
"$temp/test-o0" "$SPIRV"

sanitised=0
for flags in "-fsanitize=address,undefined" "-fsanitize=undefined"; do
    if "${CXX}" "${COMMON[@]}" -O1 -g -fno-omit-frame-pointer $flags \
            "$SRC" -o "$temp/test-sanitised" -lvulkan 2>"$temp/sanitised-build.log"; then
        "$temp/test-sanitised" "$SPIRV"
        sanitised=1
        break
    fi
    if ! grep -Eq "cannot find .*lib(asan|ubsan)|cannot find -l(asan|ubsan)|unable to find library -l(asan|ubsan)|libasan_preinit\.o" "$temp/sanitised-build.log"; then
        echo "error: sanitised build failed for a reason other than a missing sanitizer runtime:" >&2
        cat "$temp/sanitised-build.log" >&2
        exit 1
    fi
done
if [[ "$sanitised" -eq 0 ]]; then
    echo "note: sanitised build skipped; the toolchain cannot link a sanitizer runtime:" >&2
    tail -n 2 "$temp/sanitised-build.log" >&2
fi
echo "native-vulkan-stress: suite passed"
