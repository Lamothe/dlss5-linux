#!/usr/bin/env bash
# The sustained live-input suite (vr/tests/native-live-test.cpp): multi-frame
# feeding of the full evaluator through the temporal consumer with the
# sustained-run contracts (determinism, input responsiveness, balanced
# allocations, constant arena high-water, bounded RSS) and a measured
# throughput report. This is the slow suite by design (full-network frames at
# the plan's 320x320 working floor, ~8.4 s/frame at -O2 and ~36 s/frame at
# -O0 on the dev host): the full soak runs at -O2 (three 3-frame sequences);
# the -O0 level runs the same assertion classes at reduced depth (the
# recorded-codegen level, reinstated now that the uninitialised-read
# investigation is pinned by vr/test-native-uninit.sh and NATIVE-LIVE.md);
# any sanitised build runs at reduced depth too. Cross-optimisation
# bit-identity is pinned by the eval/io suites' known answers, not
# re-measured here.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
CXX="${CXX:-c++}"
SRC="$ROOT/vr/tests/native-live-test.cpp"
temp="$(mktemp -d)"
trap 'rm -rf "$temp"' EXIT
COMMON=(-std=c++20 -pthread -Wall -Wextra -Werror -ffp-contract=off -I"$ROOT/vr")

"${CXX}" "${COMMON[@]}" -O2 "$SRC" -o "$temp/test-o2"
DLSS5_LIVE_DEPTH=3 DLSS5_LIVE_THREADS=8 "$temp/test-o2"
# The -O0 level RUNS the assertion classes at depth 2 (six frames across the
# three sequences). The earlier compile-only gap recorded here was closed by
# the scratch-content independence suite (vr/test-native-uninit.sh): its
# arena/scratch poison runs are output-identical at -O0 and its
# MALLOC_PERTURB_ sweep covers both levels (the full value set at -O2, the
# recorded 165/255 at -O0), and the sustained -O0 run completes with the
# same contracts (NATIVE-LIVE.md carries the evidence).
"${CXX}" "${COMMON[@]}" -O0 "$SRC" -o "$temp/test-o0"
DLSS5_LIVE_DEPTH=2 DLSS5_LIVE_THREADS=8 "$temp/test-o0"

sanitised=0
for flags in "-fsanitize=address,undefined" "-fsanitize=undefined"; do
    if "${CXX}" "${COMMON[@]}" -O1 -g -fno-omit-frame-pointer $flags \
            "$SRC" -o "$temp/test-sanitised" 2>"$temp/sanitised-build.log"; then
        DLSS5_LIVE_DEPTH=1 DLSS5_LIVE_THREADS=8 "$temp/test-sanitised"
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

echo "native-live: suite passed"
