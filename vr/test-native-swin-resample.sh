#!/usr/bin/env bash
# CPU-only typed native Swin resampling regression (the pooled learned C -> 2C
# projection (`downsample`), the learned 2C -> C projected nearest-2x gated
# blend (`upsample_blend`) and the thin SwinLayer compositions, with
# hand-derived dyadic known answers, staged-chain equivalence, sentinel
# rejection classes and zero-allocation accounting).
# Strict builds plus a sanitised build when the toolchain can link a sanitizer
# runtime. No GPU, headset, proprietary payload or Python required.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
CXX="${CXX:-c++}"
SRC="$ROOT/vr/tests/native-swin-resample-test.cpp"
temp="$(mktemp -d)"
trap 'rm -rf "$temp"' EXIT
# -ffp-contract=off pins the documented arithmetic semantics for every stage build.
COMMON=(-std=c++20 -Wall -Wextra -Werror -ffp-contract=off -I"$ROOT/vr")

"${CXX}" "${COMMON[@]}" -O2 "$SRC" -o "$temp/test-o2"
"$temp/test-o2"
"${CXX}" "${COMMON[@]}" -O0 "$SRC" -o "$temp/test-o0"
"$temp/test-o0"

sanitised=0
for flags in "-fsanitize=address,undefined" "-fsanitize=undefined"; do
    if "${CXX}" "${COMMON[@]}" -O1 -g -fno-omit-frame-pointer $flags \
            "$SRC" -o "$temp/test-sanitised" 2>"$temp/sanitised-build.log"; then
        "$temp/test-sanitised"
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
