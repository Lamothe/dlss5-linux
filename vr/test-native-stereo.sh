#!/usr/bin/env bash
# The Milestone 2 desktop stereo dry run (vr/VR-SEAM-M1-ALIGNMENT.md): two
# evaluator handles as the two per-eye contexts (neural-seam.h's ABI shape),
# synthetic stereo pairs, per-eye counters/history, the discontinuity rule's
# reset composition, per-eye determinism and a side-by-side evidence
# artefact with the in-band label. Consumes vr/native-host.h ONLY through
# the C test vr/tests/native-stereo-test.c (compiled as C, linked against
# the C++ implementation). This is a CONTRACT test: the codegen dimension
# belongs to vr/test-native-host.sh's -O0 run; a sanitised build is
# attempted here (tolerated skip where the toolchain lacks a sanitizer
# runtime). No GPU, headset, proprietary payload or Python required.
# Roughly a dozen full-network frames (~8 s each at -O2), so the family
# runs for about three minutes.
#
# Usage: bash vr/test-native-stereo.sh [evidence-dir]
# The evidence dir (default: a fresh temp dir) receives
# stereo-side-by-side.ppm (left | grey | right) and report.txt (label,
# geometry, result). The path is printed on success; artefacts are evidence,
# not source, and are never committed.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
CXX="${CXX:-c++}"
CC="${CC:-cc}"
SRC="$ROOT/vr/native-host.cpp"
TEST="$ROOT/vr/tests/native-stereo-test.c"
temp="$(mktemp -d)"
trap 'rm -rf "$temp"' EXIT
out="${1:-$(mktemp -d)}"
mkdir -p "$out"

# -ffp-contract=off pins the documented arithmetic semantics.
COMMON=(-std=c++20 -pthread -Wall -Wextra -Werror -ffp-contract=off -I"$ROOT/vr")
CCOMMON=(-std=c11 -Wall -Wextra -Werror -I"$ROOT/vr")

"${CXX}" "${COMMON[@]}" -O2 -c "$SRC" -o "$temp/host.o"
"${CC}" "${CCOMMON[@]}" -O2 -c "$TEST" -o "$temp/test.o"
"${CXX}" "${COMMON[@]}" -O2 "$temp/host.o" "$temp/test.o" -o "$temp/stereo"

"$temp/stereo" "$out"

# A sanitised build when the toolchain can link a sanitizer runtime (the
# manual malloc/free paths are exactly what it should police; the skip is a
# recorded environment gap, not a pass).
for flags in "-fsanitize=address,undefined" "-fsanitize=undefined"; do
    if "${CXX}" "${COMMON[@]}" -O1 -g -fno-omit-frame-pointer $flags \
            -c "$SRC" -o "$temp/host-san.o" 2>"$temp/sanitised-build.log" \
        && "${CC}" "${CCOMMON[@]}" -O1 -g -fno-omit-frame-pointer $flags \
            -c "$TEST" -o "$temp/test-san.o" 2>>"$temp/sanitised-build.log" \
        && "${CXX}" "${COMMON[@]}" -O1 -g -fno-omit-frame-pointer $flags \
            "$temp/host-san.o" "$temp/test-san.o" -o "$temp/stereo-san" \
            2>>"$temp/sanitised-build.log"; then
        "$temp/stereo-san" "$out"
        break
    else
        echo "note: sanitised build skipped; the toolchain cannot link a sanitizer runtime:"
        tail -2 "$temp/sanitised-build.log"
    fi
done

# The artefacts must agree with the run: the side-by-side image carries the
# expected PPM header (shape) and the report carries the label and the PASS
# line the C test wrote (the binary already exited 0 on any failure).
head -c 32 "$out/stereo-side-by-side.ppm" | grep -q "P6" \
    || { echo "error: artefact is not a P6 PPM" >&2; exit 1; }
test "$(head -2 "$out/stereo-side-by-side.ppm" | tail -1)" = "134 72" \
    || { echo "error: artefact shape is not 134x72" >&2; exit 1; }
test -s "$out/report.txt"
grep -q "synthetic weights" "$out/report.txt"
grep -q "arbitrary transform" "$out/report.txt"
grep -q "result: PASS" "$out/report.txt"
echo "test-native-stereo: OK (evidence: $out)"
