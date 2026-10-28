#!/usr/bin/env bash
# G1 GPU stage regression: CPU-vs-GPU bit-parity for the Vulkan projection
# matmul (with the documented denormal flush-to-zero tolerance), GPU
# determinism, rejection classes and per-kernel timing. The binary name is
# derived per run; the suite prints SKIP when no Vulkan device or no glslc
# shader compiler exists (a recorded environment gap). The GLSL shader is
# compiled here at run time (no compiled artefacts are stored in the
# repository). Speed numbers are real measurements; with synthetic weights
# the OUTPUT is arbitrary (no image-quality claim). Strict builds plus a
# sanitised build when the toolchain can link a sanitizer runtime. No
# headset, proprietary payload or Python required.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
CXX="${CXX:-c++}"
SRC="$ROOT/vr/tests/native-vulkan-test.cpp"
temp="$(mktemp -d)"
trap 'rm -rf "$temp"' EXIT
# -ffp-contract=off pins the documented arithmetic semantics for every stage build.
COMMON=(-std=c++20 -pthread -Wall -Wextra -Werror -ffp-contract=off -I"$ROOT/vr")

# The GLSL shader compiles first (no compiled artefacts in the repository);
# every run receives the SPIR-V path. No glslc is a recorded environment gap.
if ! command -v glslc >/dev/null 2>&1; then
    echo "native-vulkan: SKIP (no glslc shader compiler; recorded environment gap)"
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
    # Only a missing sanitizer runtime is a skippable environment gap; any other build
    # failure (compile error, -Werror diagnostic) is a real failure of the test build.
    # Match GNU ld's full-path and -l forms plus lld's wording for the missing runtime.
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

echo "native-vulkan: suite passed"
