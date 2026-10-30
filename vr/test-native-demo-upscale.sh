#!/usr/bin/env bash
# The demonstration-transform test family: the quality harness (the
# classical kernels' measured claims at the x4 and seam geometries) and
# the neural runtime smokes (loader verification, determinism,
# finiteness). NOT DLSS5 - a demonstration transform; the neural rows run
# only when the real open weights are present (fetched by
# vr/fetch-demo-model.sh; no binaries are ever bundled).
#
# Every present model is measured (EDSR-base x2 and Real-ESRGAN x2), each
# under its own honest label.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BUILD="$ROOT/.build"
mkdir -p "$BUILD"

# Collect every present model (the order is the report's order).
MODELS=()
if [ -n "${DLSS5_DEMO_MODEL:-}" ]; then
    MODELS+=("$DLSS5_DEMO_MODEL")
else
    for cand in "$ROOT/.tools/models/edsr-base-x2.pt" \
                "$HOME/.local/share/dlss5-demo/edsr-base-x2.pt" \
                "$ROOT/.tools/models/RealESRGAN_x2plus.safetensors" \
                "$HOME/.local/share/dlss5-demo/RealESRGAN_x2plus.safetensors"; do
        [ -f "$cand" ] && MODELS+=("$cand")
    done
fi
echo "test-native-demo-upscale: models: ${#MODELS[@]} present (${MODELS[*]:-none})"
fails=0

echo "== quality harness (classical kernels + the seam geometry)"
g++ -O2 -std=c++17 -pthread -Ivr vr/tests/native-demo-upscale-test.cpp \
    -o "$BUILD/demo-quality" || exit 1
"$BUILD/demo-quality" ${MODELS[@]+"${MODELS[@]}"} || fails=$((fails + 1))

for MODEL in ${MODELS[@]+"${MODELS[@]}"}; do
    echo "== neural runtime smoke: $(basename "$MODEL")"
    case "$MODEL" in
        *.pt)
            g++ -O2 -std=c++17 -pthread -Ivr vr/tests/native-demo-edsr-smoke.cpp \
                -o "$BUILD/demo-edsr" || exit 1
            "$BUILD/demo-edsr" "$MODEL" 64 48 8 || fails=$((fails + 1))
            ;;
        *)
            g++ -O2 -std=c++17 -pthread -Ivr vr/tests/native-demo-rrdbnet-smoke.cpp \
                -o "$BUILD/demo-rrdb" || exit 1
            "$BUILD/demo-rrdb" "$MODEL" 64 48 8 || fails=$((fails + 1))
            ;;
    esac
done
[ ${#MODELS[@]} -eq 0 ] && echo "== neural runtime smokes: SKIPPED (no model; run vr/fetch-demo-model.sh)"

if [ "$fails" -ne 0 ]; then
    echo "test-native-demo-upscale: $fails suite(s) FAILED"
    exit 1
fi
echo "test-native-demo-upscale: OK"
