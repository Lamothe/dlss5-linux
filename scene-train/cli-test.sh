#!/bin/sh
set -eu
exe="$1"
dir=$(mktemp -d "${PWD}/cli-test-XXXXXX")
trap 'rm -rf "$dir"' EXIT HUP INT TERM
# Tiny constant test buffers, not scene imagery.
for file in cheap full vcheap vfull; do
    { printf 'P6\n4 4\n255\n'; i=0; while [ "$i" -lt 48 ]; do printf '\100'; i=$((i+1)); done; } > "$dir/$file.ppm"
done
printf 'cheap.ppm\tfull.ppm\n' > "$dir/train.tsv"
printf 'vcheap.ppm\tvfull.ppm\n' > "$dir/val.tsv"
"$exe" --train "$dir/train.tsv" --val "$dir/val.tsv" --checkpoint "$dir/model.pt" --patch 4 --epochs 1 --patches-per-image 1 > "$dir/log"
grep -q '^best_epoch=' "$dir/log"
test -s "$dir/model.pt"
if "$exe" --train "$dir/train.tsv" --val "$dir/val.tsv" --checkpoint "$dir/invalid.pt" --patch 4 --epochs 1 --lr nan > /dev/null 2>&1; then exit 1; fi
test ! -e "$dir/invalid.pt"
if "$exe" --train "$dir/train.tsv" --val "$dir/train.tsv" --checkpoint "$dir/invalid.pt" --patch 4 > /dev/null 2>&1; then exit 1; fi
test ! -e "$dir/invalid.pt"
if "$exe" --train "$dir/train.tsv" --val "$dir/val.tsv" --checkpoint "$dir/model.pt" --patch 4 > /dev/null 2>&1; then exit 1; fi
if "$exe" --help unexpected > /dev/null 2>&1; then exit 1; fi
