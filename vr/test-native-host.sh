#!/usr/bin/env bash
# The stable desktop-boundary suite (vr/native-host.h + vr/native-host.cpp,
# consumed by the C mock host vr/tests/native-host-test.c): the label, the
# create/evaluate/destroy lifecycle, the counter schedule with DLSSNR.Reset,
# the pipeline's determinism, the arena report and the rejection matrix
# with the untouched-output contract. The C test is compiled as C and links
# the C++ implementation — the ABI exactly as a host renderer sees it.
# Strict builds plus a sanitised build when the toolchain can link a
# sanitizer runtime. No GPU, headset, proprietary payload or Python
# required. The full mock runs full-network frames (~8 s each at -O2).
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
CXX="${CXX:-c++}"
CC="${CC:-cc}"
SRC="$ROOT/vr/native-host.cpp"
TEST="$ROOT/vr/tests/native-host-test.c"
temp="$(mktemp -d)"
trap 'rm -rf "$temp"' EXIT
# -ffp-contract=off pins the documented arithmetic semantics for every stage build.
COMMON=(-std=c++20 -pthread -Wall -Wextra -Werror -ffp-contract=off -I"$ROOT/vr")
CCOMMON=(-std=c11 -Wall -Wextra -Werror -I"$ROOT/vr")

build_pair() {
    local out="$1"
    shift
    "${CXX}" "${COMMON[@]}" "$@" -c "$SRC" -o "$temp/host.o"
    "${CC}" "${CCOMMON[@]}" -c "$TEST" -o "$temp/test.o"
    "${CXX}" "${COMMON[@]}" "$@" "$temp/host.o" "$temp/test.o" -o "$out"
}

build_pair "$temp/test-o2" -O2
"$temp/test-o2"
# The -O0 level runs the full mock too (it was compile-only while the
# recorded uninitialised-read gap was open; NATIVE-LIVE.md now pins that
# investigation and vr/test-native-uninit.sh the content-independence
# regression, so the run is reinstated). Seven full-network evaluations at
# -O0 (three frames over two handles plus the reset frame).
build_pair "$temp/test-o0" -O0
"$temp/test-o0"

sanitised=0
for flags in "-fsanitize=address,undefined" "-fsanitize=undefined"; do
    if "${CXX}" "${COMMON[@]}" -O1 -g -fno-omit-frame-pointer $flags \
            -c "$SRC" -o "$temp/host.o" 2>"$temp/sanitised-build.log" \
        && "${CC}" "${CCOMMON[@]}" -O1 -g -fno-omit-frame-pointer $flags \
            -c "$TEST" -o "$temp/test.o" 2>>"$temp/sanitised-build.log" \
        && "${CXX}" "${COMMON[@]}" $flags "$temp/host.o" "$temp/test.o" \
            -o "$temp/test-sanitised" 2>>"$temp/sanitised-build.log"; then
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
