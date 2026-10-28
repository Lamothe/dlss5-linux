# Native desktop host boundary

`vr/native-host.h` + `vr/native-host.cpp` are the stable C ABI for desktop
renderers: a host renders its scene at a LOW resolution, hands the RGB
bytes to the boundary and presents the upscaled result. CPU-only over the
temporal frame step (`temporal_image_evaluation` — reset, history feed,
run, FP16 output-texture store, read-back, commit, counter); no GPU,
headset, proprietary payload, Python or PyTorch runtime required. No vendor
code and no vendor correspondence.

## The ABI

* `dlss5_nr_create(width, height, threads, &handle)` — one pipeline per
  handle over the small plan at the host's frame extent. The internal
  working extent comes from the plan's padding; frames whose padded extent
  exceeds the demo geometry's floor (320x320) are refused
  (`DLSS5_NR_INVALID_DIMENSIONS`, the documented bound of the fixed
  512 MiB arena).
* `dlss5_nr_evaluate(handle, rgb, row_stride, out_rgb, out_stride, reset)` —
  one frame in/out at the same extent (the host presents the upscaled
  result itself). A non-zero `reset` runs DLSSNR.Reset first (history = 0,
  counter = 0). **The output buffer is untouched unless the call returns
  `DLSS5_NR_OK`.**
* `dlss5_nr_counter(handle)` — the frames-since-reset counter.
* `dlss5_nr_arena_high_water(handle, &floats)` — the evaluator arena's
  high-water (constant across frames).
* `dlss5_nr_label()` — the string a host must display alongside any
  presented output: **"synthetic weights — arbitrary transform"**. The
  synthetic DemoStore records make the transformation arbitrary; the
  boundary is honest about it in-band, not just in documentation.
* `dlss5_nr_status_name(status)` — stable status strings for logs and
  overlays. The status values are pinned to `Status`'s exact numbers.

## The mock host

`vr/tests/native-host-test.c` is a pure C consumer (compiled as C11,
linked against the C++ implementation) — the ABI exactly as a host sees
it: the label's content, the lifecycle, the counter schedule with resets,
the pipeline's determinism (two handles over the same stream produce
bit-identical frames), the arena report's stability, and the rejection
matrix with the untouched-output contract. 45 checks at -O2 via
`vr/test-native-host.sh`; the -O0 build is compile-only (the recorded
NATIVE-LIVE.md gap) and the sanitised build follows the family's
missing-runtime rule.

## Centre Stage interop (investigated; the tested boundary is the delivery)

The Centre Stage app (/home/michael/Projects/CentreStage) is a C# GLB/glTF
viewer with a Silk.NET OpenGL 3.3 renderer (`src/CentreStage/Renderer.cs`,
`Program.cs`). The interop seam is mechanically clean: .NET P/Invoke over
this C ABI, and the app already reads framebuffers (`gl.ReadPixels` in
`Program.cs` for its screenshot path). The integration shape is: render
the scene into a low-resolution offscreen framebuffer, read the RGB bytes
back, call `dlss5_nr_evaluate`, and present the upscaled result as a
fullscreen texture, with `dlss5_nr_label()` shown in the app's status
overlay (`StatusOverlay.cs`).

The full app integration is NOT delivered here: the evaluator costs
~8 s/frame at the demo geometry (a slideshow, explicitly labelled), the
renderer has no test harness (present-correctness verification needs GUI
runs), and the brief's stated valid outcome is the tested boundary with a
mock host when interop is not clean end to end. The boundary + mock host
is that outcome; the app branch remains the recorded next step for a
session with display access.

## Provenance

Independent implementation in typed C++ following this repository's
capture-evidence contract and the network semantics of the pinned
reference `mauri870/dlssnr-pytorch` at
`3beef7935a8ef976b47ab78b906e75600c52d2de`. The MIT licence text for that
reference follows.

MIT License

Copyright (c) 2026 Mauri de Souza Meneguzzo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Non-vendor statement

This is not vendor code and not a reimplementation of any vendor's
protected machinery. It follows this repository's capture-evidence
contract only and never corresponds to NVIDIA's implementation. No
proprietary weights, captures, binaries or proprietary runtime are used or
redistributed; the tree performs no runtime or capture probes. The
synthetic weights make the transformation arbitrary — the label says so
in-band — and no synthetic vendor-parity or image-quality claim is made.
