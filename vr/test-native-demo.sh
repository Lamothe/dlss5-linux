#!/usr/bin/env bash
# CPU-only demo support regression and runnable smoke: known-answer tests for
# the deterministic LCG, the synthetic image generators, the binary PPM (P6)
# codec, the contact sheet, the synthetic weight presets and the FP16
# output-texture store, then a real demo run through the complete network
# (one bounded smoke frame: the demo is the CPU baseline and its artefacts
# are checked for existence and exact size; the README-demo.md commands show
# the full temporal sequence). Strict builds plus a
# sanitised build when the toolchain can link a sanitizer runtime. No GPU,
# headset, proprietary payload or Python required.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
CXX="${CXX:-c++}"
SRC="$ROOT/vr/tests/native-demo-test.cpp"
TOOL="$ROOT/vr/demo-native.cpp"
temp="$(mktemp -d)"
trap 'rm -rf "$temp"' EXIT
# -ffp-contract=off pins the documented arithmetic semantics for every stage build.
COMMON=(-std=c++20 -pthread -Wall -Wextra -Werror -ffp-contract=off -I"$ROOT/vr")

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

# ---- the runnable demo smoke ------------------------------------------------
# The complete pipeline run is the deliverable: build the tool at -O2 and run
# one frame (the smoke keeps the runner bounded; the README-demo.md commands
# show the full temporal sequence), then check the artefacts.
"${CXX}" "${COMMON[@]}" -O2 "$TOOL" -o "$temp/demo-native"
"$temp/demo-native" --out "$temp/out" --frames 1 --reset-at -1 > "$temp/demo.log"
for artefact in before.ppm after.ppm contact.ppm; do
    test -s "$temp/out/$artefact"
done
# Exact artefact sizes: 64x72 P6 (13-byte header) and the 128x72 contact
# sheet (14-byte header).
test "$(stat -c %s "$temp/out/before.ppm")" -eq 13837
test "$(stat -c %s "$temp/out/after.ppm")" -eq 13837
test "$(stat -c %s "$temp/out/contact.ppm")" -eq 27662
grep -q "frames-since-reset=0 history=no" "$temp/demo.log"
grep -q "arena=" "$temp/demo.log"
echo "native-demo: demo smoke passed (artefacts exact, temporal line printed)"
