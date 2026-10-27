# Native CPU residual block core (foundation slice)

`vr/native-block.hpp` provides the typed, deterministic, allocation-free core
primitives shared by the residual blocks of the DLSSnR-style generator — the
convolutional and normalisation core the remaining layer families and the
full-network assembly build on. The windowed-transformer families live in the
`native-swin-*` slices; this slice adds what the conv layers and the
pixel-shuffle boundary blocks need. CPU-only clean C++ re-derivation of
published open-source semantics — **not vendor code**.

## Contents

* `LayerNorm` — channel-axis normalisation (token-major input, channel-major
  output as the Swin MLP family consumes), optional per-channel gamma/beta and
  a half-ReLU variant; biased variance with `eps = 1e-5`.
* `conv2d_kxk` — k×k convolution by explicit im2col + per-group GEMM with a
  depthwise specialisation (`group == in_channels`, one output channel per
  input channel). Symmetric padding, stride 1, implicit zero fill outside;
  grouped weights are packed `[out][per_group_in*k*k]`.
* `pixel_shuffle` / `pixel_unshuffle` — depth-to-space / space-to-depth exactly
  per PyTorch `nn.PixelShuffle` (`output[c, h·r+i, w·r+j] = input[c·r²+i·r+j, h, w]`
  in channels-last token layout); `pixel_unshuffle` is the exact inverse.
* `gelu` / `gelu_apply` — exact erf formulation, matching the Swin MLP family.
* `conv_norm_act` — conv + LayerNorm + GELU chain with both output layouts.
* `residual_split` / `residual_project` / `residual_join` — additive residual
  wiring with optional channel projection.

## Reference and provenance

Semantics re-derived from the pinned MIT reference `mauri870/dlssnr-pytorch`
commit `3beef7935a8ef976b47ab78b906e75600c52d2de` (conv/residual block
composition in `dlssnr/model.py`; shuffle helpers per the PyTorch operator
semantics its docstrings name). No network weights, captures, raw payloads or
binaries are used or reproduced.

## MIT licence (reference-derived sections)

Copyright (c) 2026 Mauri de Souza Meneguzzo

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Known-answer test

`vr/tests/native-block-test.cpp` (93 checks at -O2 and -O0 via
`vr/test-native-block.sh`) validates against hand-derived answers: LayerNorm
mean/variance/affine/half-ReLU anchors and channel-major layout, the
`nn.PixelShuffle` channel mapping anchor, shuffle/unshuffle inversion (r=2 and
r=4 round trips at asymmetric extents), im2col convolution anchors (3×3
identity, hand-computed 2×2 case with bias, depthwise passthrough, grouped
2-group case with packed weights), GELU anchors, the ConvNormAct chain,
residual split/project/join values, sentinel guards and zero-allocation
accounting.

## Non-vendor statement

This is not vendor code and not a reimplementation of any vendor's protected
network. It follows the MIT reference's operator semantics only and never
corresponds to NVIDIA's implementation. No proprietary weights, captures,
binaries or proprietary runtime are used or redistributed; the tree performs no
driver-level runtime/capture probing.
