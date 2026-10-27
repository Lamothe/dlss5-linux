# Native Linux pipeline demo

A runnable, zero-asset demo of the complete typed native network in this
repository: host frame in -> `image_evaluation` (the full plan, all 76 rows of
the 'small' network) -> host frame out, on the CPU, with deterministic
synthetic weight records, a short temporal sequence with a mid-stream reset,
and per-frame CPU timings plus the arena high-water mark (the CPU baseline for
a later GPU stage).

**Honest scope:** the weight records are synthetic and deterministic. The
visual transformation is arbitrary — the demo proves the complete pipeline
runs on Linux, **not** DLSS quality and **not** vendor correspondence. No
proprietary weights, captures, payloads or binaries are used or produced; no
runtime/capture probes are performed.

## Build and run

From the repository root (any recent g++/clang with C++20; no GPU, no Python,
no external assets):

```sh
c++ -std=c++20 -Wall -Wextra -Werror -ffp-contract=off -Ivr -O2 \
    vr/demo-native.cpp -o /tmp/demo-native
/tmp/demo-native --out demo-output
```

That is all: with no `--input` a built-in checkerboard is generated, so the
demo runs out of the box. Expected console output (timings vary by machine;
this CPU baseline was recorded on the development host):

```
dlss5 native demo: 64x72 frame, plan 76 rows (small), padded extent 320x320
preset: seeded (seed 20261028) - synthetic weights: the visual transformation is arbitrary; this proves the pipeline runs, not DLSS quality
frame 0: frames-since-reset=0 history=no   37476.47 ms  mean= 94.09  arena=48531972 floats  float-delta=n/a
frame 1: frames-since-reset=1 history=yes  37269.25 ms  mean= 94.10  arena=48531972 floats  float-delta=
           max |out_1 - out_0| = 0.433105 float over the padded map; 0.433105 over the visible crop (pre-u8)
frame 2: RESET (history cleared, counter -> 0)
frame 2: frames-since-reset=0 history=no   37447.27 ms  mean= 94.09  arena=48531972 floats  float-delta=
           max |out_2 - out_1| = 0.433105 float over the padded map; 0.433105 over the visible crop (pre-u8)
frame 3: frames-since-reset=1 history=yes  37394.19 ms  mean= 94.10  arena=48531972 floats  float-delta=
           max |out_3 - out_2| = 0.433105 float over the padded map; 0.433105 over the visible crop (pre-u8)
total: 149587.18 ms for 4 frames (mean 37396.80 ms/frame); arena high-water 48531972 floats (185.1 MiB)
artefacts:
  demo-output/before.ppm
  demo-output/after.ppm
  demo-output/contact.ppm
```

The `frames-since-reset`/`history` columns are the temporal carrier's
capture-evidence behaviour: the counter walks 0, 1, ... and a reset returns it
to 0 with the history cleared (frames-since-reset=0 and history=no on the
reset frame). The `float-delta` line reports `max |out_N - out_{N-1}|` at the FP16
store precision (before the 8-bit read-back), over the visible crop and over
the full padded map: the u8 images can hide a history influence smaller than
1/255, and the temporal signal can land in the padded border that the read-back
cropped away with an earlier store design (the per-key record slots fixed
that: with the default preset the delta 0.433105 appears in the visible crop
too). The state machine is exercised regardless of the preset.

## Artefacts

Written to `--out` (default `demo-output/`):

| File          | Size (bytes) | Contents                                        |
|---------------|--------------|-------------------------------------------------|
| `before.ppm`  | 13,837       | the 64x72 input frame (built-in or `--input`)   |
| `after.ppm`   | 13,837       | the final frame's 64x72 pipeline output         |
| `contact.ppm` | 27,662       | side-by-side sheet: before (left) \| after (right), 128x72 |

They are binary PPM (P6) images. Open them in the GNOME image viewer
(`eog`/Image Viewer) directly, or convert to PNG first:

```sh
eog demo-output/contact.ppm            # GNOME Image Viewer opens P6 PPM
# or: pnmtopng demo-output/contact.ppm > contact.png   (netpbm, if installed)
# or: magick demo-output/contact.ppm contact.png        (ImageMagick, if installed)
```

The generated images are git-ignored (the root `.gitignore`'s catch-all and
`*.ppm` rules cover `demo-output/`).

## Options

```
--input PATH       host PPM (P6) image, exactly 64x72 RGB (default: built-in)
--synthetic NAME   built-in image: checker | gradient
--out DIR          output directory (default: demo-output)
--preset NAME      weight preset: seeded | zero
--seed N           preset seed (default: 20261028)
--frames N         frames to evaluate (default: 4)
--reset-at K       temporal reset before frame K (default: 2; -1 = never)
```

### Weight presets

Both presets serve **every** record the plan asks for, deterministically from
`(preset, seed, block, layer, name, shape)` — repeated runs are bit-identical.

* `seeded` (default): small-magnitude pseudo-random records (e4m3 codes
  12..28, which the tree's bias-7 decode gives 0.0234375..0.09375, and
  1/1024-grid values in [-1, 1)) so multi-layer activations stay off the e4m3
  ceiling. Arbitrary but structured and repeatable; each record is keyed by
  (block, layer, name) so the consumers' concurrently-live record pointers
  stay independent.
* `zero`: every record is zero. The residual path dominates; the demo prints
  the observed output rather than promising any particular image.

No preset is a trained network: any visible structure is an artefact of the
synthetic records, not an enhancement.

## Tests

The demo's own components have known-answer tests (the LCG constants, the PPM
codec byte layout, the checker/gradient pixels, the contact-sheet placement,
the preset determinism and ranges, the FP16 store rounding) and the runner
also executes a real one-frame demo smoke, checking the artefacts' exact
sizes:

```sh
bash vr/test-native-demo.sh
```

The suite runs at -O2 and -O0 (it prints its own check count — 351 on the
development host); a sanitised build runs when the toolchain can link a
sanitizer runtime (it skips on this toolchain's known missing
`/usr/lib64/libubsan.so.1.0.0`, recorded as an environment gap).

## Boundaries

CPU-only; the GPU stage is later work. No proprietary payloads, captures or
binaries are read, written or embedded; the demo performs no runtime/capture
probes and asserts no vendor parity. The temporal commit models the FP16
output-texture store of the hardware route (the carrier holds FP16 texture
content bit-exactly), using the tree's verified binary16 rounding; see
`vr/NATIVE-DEMO.md` for the full semantics and the MIT attribution notice.
