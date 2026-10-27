#!/usr/bin/env bash
# CPU-only typed native residual block core regression (LayerNorm, im2col
# convolution with depthwise specialisation, pixel shuffle/unshuffle, GELU,
# ConvNormAct chain, residual split/join), with hand-derived known answers,
# sentinel guards and zero-allocation accounting.
# Strict builds plus a sanitised build when the toolchain can link a sanitizer
# runtime. No GPU, headset, proprietary payload or Python required.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
CXX="${CXX:-c++}"
SRC="$ROOT/vr/tests/native-block-test.cpp"
temp="$(mktemp -d)"
trap 'rm -rf "$temp"' EXIT
# -ffp-contract=off pins the documented arithmetic semantics for every stage build.
run_build() {
    local name="$1"; shift
    "$CXX" "$@" -std=c++20 -Wall -Wextra -Werror -ffp-contract=off \
        -I"$ROOT" "$SRC" -o "$temp/$name"
    "$temp/$name"
}
run_build O2 -O2
run_build O0 -O0
if run_build asan -O1 -fsanitize=address,undefined -fno-omit-frame-pointer 2>"$temp/asan-build.log"; then
    run_build asan -O1 -fsanitize=address,undefined -fno-omit-frame-pointer >/dev/null
else
    echo "note: sanitised build skipped; the toolchain cannot link a sanitizer runtime:" >&2
    sed -n '1,4p' "$temp/asan-build.log" >&2
fi
