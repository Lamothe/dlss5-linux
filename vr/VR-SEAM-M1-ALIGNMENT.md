# VR seam — Milestone 1 contract alignment (29/10/2026)

Milestone 1 of `VR-SEAM-PLAN.md`, executed against the real contract: the
Genesis checkout on platinum (repo at `be98965`, worktree
`feature/dlss5-native-seam` — main untouched) carries `NEURAL-SEAM.md` +
`neural-seam.h` at its root (the plan's `vr/NEURAL-SEAM.md` path was
approximate; `vr/SCENE.md` and `vr/INTEGRATION.md` are referenced by the
contract but live in the dlss5-vr history, not in Genesis).

Sources pinned: `NEURAL-SEAM.md` (16/10/2026, provisional), `neural-seam.h`
(82 lines, the ABI), `vr/native-host.h` (this workspace's host boundary),
`vr/native-graph-plan.hpp` (padding/extent rules), `vr/native-host.cpp`
(arena bound), demo evidence `CentreStage-dlss5/.evidence/dlss-demo-final2/
report.txt` (arena high-water measurement).

## 1. Field mapping — `NeuralSeamEye` to the native host boundary

| Genesis (`neural-seam.h`) | Native (`native-host.h`) | Mapping |
| --- | --- | --- |
| `colour_srgb_f16` (RGBA FP16, display-referred sRGB) | `rgb` (RGB8, row_stride) | Adapter-side FP16 → 8-bit sRGB round-to-nearest; alpha dropped (the evaluator is a colour transform). Output converts RGB8 → FP16 RGBA (alpha 1.0) into the caller's buffer. |
| `render_width/height`, `output_width/height` | `dlss5_nr_create(width, height, threads)` | The evaluator returns the same extent it is given; "upscaling" is presentation-side. The seam adapter creates one handle at the render extent and lets the compositor present at `output_width/height`. |
| `depth` (D3D projected, 0.1–20 m) | — | **Accepted and ignored.** The synthetic evaluator consumes colour only (swin/decode on the colour plane). The field stays in the contract so the real network can bind it later without a shape change. |
| `motion` (prev-minus-current, pixels) | — | Accepted and ignored (same reason). |
| `jitter_x/jitter_y` | — | Accepted and ignored. The evaluator has no jitter model; the adapter records the values in its trace for the dlss5-vr session. |
| `reset` (int32) | `reset` (int, per call) | Direct. `reset != 0` runs the history reset (history = 0, counter = 0) exactly as the contract's rule 1/3 require. |
| `discontinuity_id` (uint64) | — | Adapter-side per rule 2: the adapter tracks the last evaluated id and ORs 1 into both eyes' `reset` when it differs. |

The contract's Status section asks for exactly this conformance ("the
structure of the contract should not change, only the convention table"):
the convention table becomes **colour = display-referred sRGB, FP16 in the
contract / RGB8 across the native boundary (rounded, alpha dropped); depth
and motion are carried but not consumed by the synthetic evaluator; jitter
is carried for traceability.**

## 2. Two contexts — the ABI already matches

`neural_seam_evaluate(NeuralSeamContext* contexts[2], ...)` requires one
context per eye and forbids sharing/reuse across features (the Proton-era
global-state finding). The native boundary is one `dlss5_nr_handle` per
pipeline with **per-handle temporal state** — history, counter and arena:
**two handles (left/right) are the two contexts, 1:1** — and the counter
schedule (`dlss5_nr_counter`) is per handle as the contract expects. One
shared setting must be noted: `dlss5_nr_create` sets the evaluator's
parallelism process-wide (last create wins for both handles; the knob is
documented as not thread-safe against its own use), so the two handles
should be created with the same `threads` value. A context must never
outlive its handle pair or be rebound across features; the adapter owns
exactly two and destroys them together.

## 3. The key open question: arena bound vs per-eye extent — **recommendation: bounded region (foveated/upscale window) per eye**

Measured and computed:

* The plan (`native-graph-plan.hpp`) is extent-parametric: `0 <
  width,height <= 16384` with every padded axis floored at 320
  (`kPlanFloorExtent`). The 320x320 refusal is **not** a kernel limit: it is
  `kMaxPaddedSide` in `native-host.cpp`, guarding a fixed 512 MiB arena
  (`1 << 27` floats) sized for the demo geometry.
* Arena high-water at the demo floor (64x72 render, padded 320x320):
  **48,531,972 floats = 185 MiB** (measured, `report.txt`), inside the
  512 MiB bound with 2.8x headroom.
* A headset per-eye extent (e.g. Quest 3, ~2064x2208) pads to 2112x2240
  under the plan's own padding rules (axis alignment 64 at these extents;
  4,730,880 padded pixels) = **46x the floor's token count**. The arena
  high-water scales with tokens: **~8.5 GiB** (~17x the documented
  512 MiB bound) per eye. Runtime scales the same way: ~8.7 s x 46 ≈
  **6.7 minutes per eye per frame on CPU** — unusable even for the slow
  inspect mode.

Therefore: **each eye feeds the evaluator a bounded region of at most
320x320 padded** — a foveated/upscale window (crop or downsampled eye
view), presented/composited by the caller. This keeps the documented arena
bound (194 MB high-water), keeps the inspect-mode latency at seconds per
eye, and needs no plan or kernel change. Extending `kMaxPaddedSide` is a
one-line guard relaxation plus arena re-sizing and is *only* sensible for
the dlss5-vr GPU path, where the frame-rate questions live anyway.

## 4. Ownership split with dlss5-vr

* **dlss5-vr** owns: stereo/per-eye state in the headset (context lifetime,
  per-eye history semantics under reprojection), the compositor/handover
  path, horizon-lock/passthrough interplay on the captured inputs, and
  anything frame-rate bound (the GPU path).
* **dlss5-linux (this workspace)** owns: the evaluator, `native-host.h`
  (the stable C ABI and its documented bound), the adapter pattern that
  implements `neural_seam_evaluate` over two handles (including the
  discontinuity rule and the FP16/RGB8 conversions), and the dry-run
  evidence families (the `vr/test-*` suites).
* **The editor (Genesis)** stays exactly as its contract states: it
  produces `NeuralSeamFrame` and consumes `NeuralSeamOutput`; it never
  touches evaluator internals. The seam adapter is the "native evaluation
  stream" the contract defers binding to.

## 5. Milestone 2 shape (desktop stereo dry run, no headset) — defined here

Two evaluator handles fed with synthetic stereo pairs at demo-ish geometry
(<= 320x320 padded per eye); per-eye histories with the counter schedule per
handle; the reset semantics the rule-2 discontinuity composition relies on
(the composition itself is pinned by the seam adapter suite below);
side-by-side presentation of both processed frames with the
`dlss5_nr_label()` string recorded beside the artefact; per-eye history
independence (a shared history must fail) and determinism/reset verified
against the mock-host contracts. Evidence: `vr/test-native-stereo.sh` (this
workspace, also runnable on platinum for the Vulkan/RADV portability leg).

## 6. Readiness state — stopping at the headset boundary

Done without the headset: contract mapping and conformance (sections 1–2),
the bounded-window decision (section 3), the ownership split (section 4),
the evaluator-side stereo plumbing (`vr/test-native-stereo.sh`: two
handles, per-handle counters, per-eye history independence, history
consumption, reset determinism, a side-by-side PPM with the label beside
it), the portability leg
on platinum's 8060S (RADV — see below), and the seam adapter binding
`neural_seam_evaluate` to the native ABI in the Genesis worktree branch
`feature/dlss5-native-seam` (`src/neural-seam-native.cc` + the factory
surface `neural-seam-native.h`; the contract header is unmodified; 29
checks pin rules 1–3 (including history accumulation and the
aliasing/extent rejections), per-eye independence, the FP16/RGB8
conventions and the output-untouched-on-error contract, and `run-tests.sh`'s core suite
stays green at 859 checks).

**Portability leg (platinum, Radeon 8060S, RADV — logs in
`~/dlss5-port-logs/` there, source mirror `~/Projects/dlss5-linux-port`):**
`vr/test-native-vulkan.sh` **suite passed** — G1 projection parity, **G2
complete-network parity byte-identical**, **G8 batched swin parity
byte-identical** (the GPU really ran: 427 batched swin calls), and the
G8 six-frame soak; `vr/test-native-stereo.sh` 42 checks; `vr/test-native-host.sh`
45 checks at -O2 and -O0 (the sanitised build is skipped there — the
toolchain lacks libubsan, a recorded environment gap). The byte-identical
CPU-vs-GPU artefact contract therefore holds on RADV as well as on the
NVIDIA reference; no divergence to attribute.

Needs Michael + the headset (Milestone 3, recorded in the plan): in-headset
presentation of the per-eye results in the slow inspect mode; the
passthrough/horizon-lock interplay check on the captured inputs; the honest
in-headset latency budget; and the decision of what transfers to the
dlss5-vr GPU path for anything frame-rate bound. Nothing in sections 1–5
assumes those answers.
