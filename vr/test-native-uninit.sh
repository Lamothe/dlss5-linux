#!/usr/bin/env bash
# The scratch-content independence suite (the MALLOC_PERTURB_ regression,
# vr/tests/native-uninit-test.cpp): the evaluator's outputs must not depend
# on the content of memory the kernels overwrite before reading. The swin
# scratch partitions and the per-row decode staging are poisoned with
# NaN/denormal/random patterns at -O2 and -O0 (bit-identical outputs), the
# full-network arena is painted per pattern (bit-identical frames, painted
# run and leftover run both compared, at both levels), and a cross-process
# MALLOC_PERTURB_ sweep asserts identical output across perturb fills (the
# full value set at -O2; the recorded 165/255 at -O0). This is the
# regression for the recorded -O0 uninitialised-read investigation
# (NATIVE-LIVE.md): it pins every observable output; a timing-only dead read
# is documented there separately. CPU-only; no GPU, headset, proprietary
# payload or Python required.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
CXX="${CXX:-c++}"
SRC="$ROOT/vr/tests/native-uninit-test.cpp"
temp="$(mktemp -d)"
trap 'rm -rf "$temp"' EXIT
COMMON=(-std=c++20 -pthread -Wall -Wextra -Werror -ffp-contract=off -I"$ROOT/vr")

"${CXX}" "${COMMON[@]}" -O2 "$SRC" -o "$temp/test-o2"
"$temp/test-o2"

# The literal MALLOC_PERTURB_ contract: glibc's fresh-malloc and free fills
# must not change any output. The suite is deterministic, so each perturb
# value's whole console output must equal the unperturbed run's (the hash
# lines are the sensitive part). The reduced pattern set and single frame
# keep the sweep bounded; the full pattern set runs above. The sweep covers
# both levels: the full value set at -O2, and the two values the record
# names (165, 255) at -O0 — the codegen class the defect was recorded under.
sweep() {
    DLSS5_UNINIT_PATTERNS=reduced DLSS5_UNINIT_NET_FRAMES=1 "$@"
}
sweep "$temp/test-o2" > "$temp/sweep-baseline.log"
for perturb in 1 127 128 165 254 255; do
    sweep env MALLOC_PERTURB_="$perturb" "$temp/test-o2" \
        > "$temp/sweep-$perturb.log"
    if ! diff -u "$temp/sweep-baseline.log" "$temp/sweep-$perturb.log"; then
        echo "error: MALLOC_PERTURB_=$perturb changed the suite output" >&2
        exit 1
    fi
done
echo "native-uninit: -O2 MALLOC_PERTURB_ sweep (1 127 128 165 254 255) output-identical"

# The -O0 level: the recorded defect's codegen class. The reduced pattern
# set bounds the run; the -O2 run above covers the full set.
"${CXX}" "${COMMON[@]}" -O0 "$SRC" -o "$temp/test-o0"
DLSS5_UNINIT_PATTERNS=reduced "$temp/test-o0"

sweep "$temp/test-o0" > "$temp/sweep-o0-baseline.log"
for perturb in 165 255; do
    sweep env MALLOC_PERTURB_="$perturb" "$temp/test-o0" \
        > "$temp/sweep-o0-$perturb.log"
    if ! diff -u "$temp/sweep-o0-baseline.log" "$temp/sweep-o0-$perturb.log"; then
        echo "error: MALLOC_PERTURB_=$perturb changed the -O0 suite output" >&2
        exit 1
    fi
done
echo "native-uninit: -O0 MALLOC_PERTURB_ sweep (165 255) output-identical"

sanitised=0
for flags in "-fsanitize=address,undefined" "-fsanitize=undefined"; do
    if "${CXX}" "${COMMON[@]}" -O1 -g -fno-omit-frame-pointer $flags \
            "$SRC" -o "$temp/test-sanitised" 2>"$temp/sanitised-build.log"; then
        DLSS5_UNINIT_PATTERNS=reduced DLSS5_UNINIT_NET_FRAMES=1 \
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

echo "native-uninit: suite passed"
