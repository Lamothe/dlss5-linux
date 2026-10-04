#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
temp="$(mktemp -d)"
trap 'rm -rf "$temp"' EXIT
python3 - "$ROOT" "$temp" <<'PY'
from pathlib import Path
import sys
root, output = map(Path, sys.argv[1:])
source = (root / 'vr/xr-scene.c').read_text()
functions = source[source.index('typedef struct Locomotion'):source.index('static void view_constants')]
test = (root / 'vr/tests/vertical-test.c').read_text()
marker = '// Locomotion functions are inserted from xr-scene.c by the host test runner.'
if test.count(marker) != 1:
    raise SystemExit('Test insertion marker missing or ambiguous')
(output / 'test.c').write_text(test.replace(marker, functions))
PY
"${CC:-cc}" -std=c11 -Wall -Wextra -Werror "$temp/test.c" -lm -o "$temp/test"
"$temp/test"
