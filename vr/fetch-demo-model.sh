#!/usr/bin/env bash
# Fetch the demonstration models: real, openly licensed weights for the
# demonstration transform. This is NOT DLSS5 - the models demonstrate the
# seam end to end with real weights.
#
# Models (both openly obtained and openly licensed; no NVIDIA assets):
#
#  1. EDSR-base x2 (the default - classical super-resolution, trained on
#     clean degradation; like every learned upscaler it adds plausible
#     structure and is MEASURED, not assumed: see NATIVE-DEMO-UPSCALE.md)
#       source:  huggingface.co/eugenesiow/edsr-base (pytorch_model_2x.pt)
#       lineage: super-image (eugenesiow), trained on DIV2K
#       license: Apache-2.0 (Lim et al., "Enhanced Deep Residual Networks
#                for Single Image Super-Resolution", CVPRW 2017)
#       format:  torch-save (zip of a pickle stream + raw f32 storages,
#                16 ResBlocks x 64 feats, 1,369,883 parameters)
#
#  2. Real-ESRGAN x2 (kept for comparison: a blind-restoration model;
#     measured to underperform the classical baselines against a pristine
#     ground truth - see the docs)
#       source:  huggingface.co/mlx-community/Real-ESRGAN-x2plus (model.safetensors)
#       lineage: converted from the official xinntao/Real-ESRGAN release
#                checkpoint (RealESRGAN_x2plus.pth); the conversion's documented
#                parity vs PyTorch is 2.4e-6 full-forward diff (CPU fp32)
#       license: BSD-3-Clause (Wang et al., "Real-ESRGAN: Training
#                Real-World Blind Super-Resolution with Pure Synthetic
#                Data", ICCVW 2021)
#       format:  safetensors (F16, 16,703,171 parameters, 33,472,058 bytes)
#
# The weights are downloaded, never committed (this repository carries no
# model binaries). Checksums are pinned: a mismatch is a hard failure.
set -u

fetch() {
    local dest="$1" url="$2" sha="$3" name="$4" license="$5"
    mkdir -p "$(dirname "$dest")"
    if [ -f "$dest" ]; then
        if echo "$sha  $dest" | sha256sum -c --quiet - 2>/dev/null; then
            echo "present and verified: $dest"
            return 0
        fi
        echo "checksum mismatch on the existing file; re-downloading: $dest" >&2
        rm -f "$dest"
    fi
    echo "downloading $name: $url"
    curl -fsSL -o "$dest" "$url" || {
        echo "error: download failed: $url" >&2
        rm -f "$dest"
        return 1
    }
    if ! echo "$sha  $dest" | sha256sum -c --quiet -; then
        echo "error: checksum mismatch after download: $dest" >&2
        rm -f "$dest"
        return 1
    fi
    echo "verified: $dest"
    echo "license: $license"
}

BASE="${1:-$HOME/.local/share/dlss5-demo}"
rc=0

fetch "$BASE/edsr-base-x2.pt" \
    "https://huggingface.co/eugenesiow/edsr-base/resolve/main/pytorch_model_2x.pt" \
    "b709bea0d3e3a7a1c84eb1eaa6e5e777f176edcfca00b3b5acde5320eb630211" \
    "EDSR-base x2" \
    "Apache-2.0 (EDSR, Lim et al., CVPRW 2017; DIV2K)" || rc=1

fetch "$BASE/RealESRGAN_x2plus.safetensors" \
    "https://huggingface.co/mlx-community/Real-ESRGAN-x2plus/resolve/main/model.safetensors" \
    "93c875ee5637ca326bae14420fbbb5be8a19f0eb7d3b0c6af08a8ef9c0180020" \
    "Real-ESRGAN x2" \
    "BSD-3-Clause (Real-ESRGAN, Wang et al., ICCVW 2021)" || rc=1

exit $rc
