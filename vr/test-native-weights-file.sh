#!/bin/bash
# test-native-weights-file.sh — the data-free weights interface family.
#
#   1. The library tests (container round-trip, the structural/content
#      rejection matrix with named errors, the compatibility verifier's
#      mismatch classes, recorder <-> file-provider fetch fidelity).
#   2. The full-network byte-identity round-trip through the demo: dump
#      the request set from a synthetic run, reload it through the
#      file-backed provider, and require byte-identical artefacts - the
#      property the reference-validation resume path relies on.
#   3. The rejection path through the demo: a corrupted container must
#      exit non-zero with the error named on stderr.
#
# NOT DLSS5. No NVIDIA assets, no proprietary payloads; the fixtures are
# synthetic. Runs -O2 and -O0 (the codegen class the hygiene phase pinned
# a regression for).
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)

fails=0
check() {
  local desc=$1; shift
  if "$@" > /dev/null 2>&1; then
    echo "  PASS $desc"
  else
    echo "  FAIL $desc"
    fails=$((fails + 1))
  fi
}

TMP=$(mktemp -d)
# Evidence is kept for inspection on failure (the report references it);
# only a fully green run cleans up.

for OPT in -O2 -O0; do
  echo "native-weights-file ($OPT):"
  c++ -std=c++20 -ffp-contract=off -Wall -Wextra $OPT -Ivr \
      vr/tests/native-weights-file-test.cpp -o "$TMP/wf-test" || exit 1
  "$TMP/wf-test" || fails=$((fails + 1))
  c++ -std=c++20 -pthread -ffp-contract=off -Wall -Wextra $OPT -Ivr \
      vr/demo-native.cpp -o "$TMP/demo" || exit 1
  c++ -std=c++20 -ffp-contract=off -Wall -Wextra $OPT -Ivr \
      vr/weights-verify.cpp -o "$TMP/verify" || exit 1

  # The full-network byte-identity round-trip. The DUMP run is the
  # reference: the recorder and the file provider run the same dataflow
  # over the same bytes, so the identity is exact. The plain synthetic
  # run is kept as a smoke test.
  "$TMP/demo" --out "$TMP/ref" --frames 1 --reset-at -1 --threads 2 > "$TMP/ref.log" 2>&1
  check "synthetic smoke run" test $? -eq 0
  "$TMP/demo" --out "$TMP/dumpdir" --frames 1 --reset-at -1 --threads 2 \
      --dump-weights "$TMP/manifest.nwf" > "$TMP/dump.log" 2>&1
  check "request-set dump" test $? -eq 0 -a -s "$TMP/manifest.nwf"
  "$TMP/demo" --out "$TMP/file" --frames 1 --reset-at -1 --threads 2 \
      --weights "$TMP/manifest.nwf" > "$TMP/file.log" 2>&1
  check "file-provider run" test $? -eq 0
  check "artefacts byte-identical (after.ppm)" \
      cmp -s "$TMP/dumpdir/after.ppm" "$TMP/file/after.ppm"
  check "artefacts byte-identical (before.ppm)" \
      cmp -s "$TMP/dumpdir/before.ppm" "$TMP/file/before.ppm"

  # The verification harness accepts the matching file...
  check "harness accepts the matching file" \
      "$TMP/verify" "$TMP/manifest.nwf" "$TMP/manifest.nwf"
  # ...and rejects a shape-edited one with the tensor named.
  cp "$TMP/manifest.nwf" "$TMP/bad.nwf"
  printf 'X' | dd of="$TMP/bad.nwf" bs=1 seek=90 conv=notrunc > /dev/null 2>&1
  check "harness rejects a corrupted file" \
      bash -c "! $TMP/verify $TMP/manifest.nwf $TMP/bad.nwf"
  "$TMP/verify" "$TMP/manifest.nwf" "$TMP/bad.nwf" > "$TMP/bad.log" 2>&1
  check "rejection names the problem" \
      grep -q "hash mismatch\|rejected" "$TMP/bad.log"

  # The demo's rejection path: a corrupt file must fail loudly.
  "$TMP/demo" --out "$TMP/baddir" --frames 1 --reset-at -1 \
      --weights "$TMP/bad.nwf" > "$TMP/baddemo.log" 2>&1
  check "demo rejects a corrupt weights file" test $? -ne 0
  check "demo rejection is named" \
      grep -q "hash mismatch\|rejected" "$TMP/baddemo.log"
done

if [ $fails -ne 0 ]; then
  echo "test-native-weights-file: $fails check(s) FAILED (evidence kept: $TMP)"
  exit 1
fi
rm -rf "$TMP"
echo "test-native-weights-file: OK"
