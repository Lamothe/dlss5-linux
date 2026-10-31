# Offline scene trainer (proof of plumbing)

A small native C++20/libtorch trainer for **existing aligned pairs**, not a scene renderer or an image-quality claim. Input is cheap RGB; target is full RGB. The fixed model is 3×3 convolutions 3→8→8→3, ReLU after the first two, plus an unclamped RGB identity skip. It has no temporal input, augmentation or correspondence checking. Test fixtures are tiny synthetic numeric buffers solely for software tests; no Teen Kitty data is supplied or simulated.

## Dependencies and CPU build

- Linux, CMake ≥3.20, a C++20 compiler and matching libstdc++ ABI, and libtorch headers and shared libraries. Tested here with GCC 16.2.1 and CMake 4.3.0.
- The installed libtorch is PyTorch **2.11.0**, at `/home/michael/Projects/ComfyUI/.venv/lib/python3.12/site-packages/torch`. The existing installation is a CUDA-enabled wheel, whose shared-library dependencies include CUDA 12 runtime components installed alongside it in `site-packages/nvidia/`. These must remain accessible to the dynamic loader even for CPU use. Python and its interpreter are **not used** to build or run the trainer.
- This build links the wheel's `torch`, `torch_cpu` and `c10` directly instead of its TorchConfig, which would demand a matching CUDA development toolkit. It assumes the wheel's current C++11 libstdc++ ABI; a different libtorch build may need ABI flags adjusted. Linking explicitly places libstdc++ before libtorch to avoid this wheel's conflicting filesystem symbols; preserve this order for other native consumers. CUDA training requires a compatible driver, CUDA-enabled libtorch and its runtime libraries; it has not been tested here.

From this directory:

```sh
cmake -S . -B build -DTORCH_ROOT=/home/michael/Projects/ComfyUI/.venv/lib/python3.12/site-packages/torch -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 1
ctest --test-dir build --output-on-failure
```

## Input and execution

Each manifest is UTF-8-compatible plain text, one pair per line: `cheap.ppm<TAB>full.ppm`. Paths are relative to that manifest's directory or absolute; tabs and newlines in paths are unsupported. Empty lines and lines beginning `#` are ignored. Both manifests must be nonempty and different files. Every input file — both manifests and every image across them — must be a distinct file identity, including via hard links and symlinks. Identity checks use `fstat` on the same open descriptor that supplies the bytes, so replacing a path after it is opened cannot defeat them; a path resolved for reading is never re-opened. This prevents file-identity overlap, **not** copied content, correlated frames, or other leakage; split at capture/scene level yourself. **Maintain exclusive control of the data directory and its paths (manifests and images) for the entire run**: it must not be writable by untrusted parties, and no other process, including a trusted concurrent process, may modify its entries or change the paths. Relative manifest paths resolve against the manifest's canonical directory, which this constraint pins.

Only binary P6 RGB PPM with maxval 255, exact raster length and matching dimensions throughout is accepted; headers may contain standard whitespace/comments before tokens, including a comment immediately after maxval and before its separator; comments end at CR or LF. A `#` after the separator is raster data. When a CR following maxval is followed by LF, exact raster length distinguishes a CRLF separator from a CR separator followed by an LF raster byte; otherwise exactly one whitespace byte separates header and raster. Maximum 16,384 per dimension and 67,108,864 pixels. Input bytes are divided by 255 without gamma or colour-space conversion. All images are loaded into memory at once (plus one transient raw-file copy per load), which caps realistic scale at modest datasets; streaming or epoch-wise reload is not implemented. Semantic alignment, exposure, colour space and distinct frame provenance are the caller's responsibility.

**Log-target safety:** the trainer refuses to run (exit code 2) when `stdout` or `stderr` names any input file. Every path the invocation can name — argv values and every manifest-listed image, resolved the same way as at load — is checked against the log streams before any work, and the check is repeated by `fstat` identity against the loaded input descriptors before any log write; error and usage output never reaches a stream that may name an input. A regular-file log stream that cannot be proven free of input names (manifests missing or unparsable) is treated as unsafe: errors and usage go to the controlling terminal where available, and the trainer stays silent in preference to corrupting a file. Do not redirect logs onto dataset paths. This defeats stable aliasing; a hostile concurrent path swap during the run is excluded by the exclusive data-directory constraint above.

```sh
./build/scene-train --train /path/train.tsv --val /path/val.tsv \
  --checkpoint /path/new-model.pt --device cpu --epochs 10 --patch 32 \
  --patches-per-image 4 --lr 0.001 --seed 1
```

Checkpoint must not already exist. New models initialise the final convolution weights and bias to zero, so the initial unclamped float output is the cheap input (identity), including at image borders. The epoch-zero validation loss is therefore the full-image cheap-versus-full MSE, and its checkpoint remains selected unless training strictly improves it. The output layer can receive gradients immediately; earlier layers initially have zero gradients and can receive gradients after the output layer updates. The parameter names and checkpoint format are unchanged, and loading a checkpoint replaces these initial values. The model is evaluated on every **full** validation image before training and after each epoch, reporting two mean per-image RGB criteria: `validation_mse` (unclamped float model output, matching the training objective) and `deployed_mse` (after exactly the host inference conversion — clamp to [0,1], multiply by 255, round to nearest ties-to-even, quantise to 8-bit — matching deployed output). **Checkpoint selection uses `deployed_mse`** (ties keep the earlier checkpoint), so selection sees what inference will produce; both criteria are logged per epoch. Training uses Adam, batch size one and seeded aligned random supervised patches in manifest order. Each patch includes up to a three-pixel input halo, clipped at actual image borders; loss is computed only on the selected interior. There is no gradient contribution from validation. Training loss uses unclamped model output. Logs begin with a `not DLSS5` label line, then show the initial/epoch losses and the selected epoch. The checkpoint contains only CPU model parameters, not the optimiser, provenance or resume state. The output name is reserved exclusively, including against dangling symlinks. Each complete checkpoint is written to a separate temporary file and atomically replaces the reserved name; failures preserve any earlier complete checkpoint. A process interruption can leave a reserved empty file or temporary file. Use a new output path per run. Maintain exclusive control of the checkpoint directory and its path for the entire run: it must not be writable by untrusted parties, and no other process, including a trusted concurrent process, may modify its entries or change the path. This is required for reservation, replacement and cleanup ownership assumptions. Atomic replacement does not guarantee durability across power loss.

**Checkpoint security:** Load only trusted, locally-produced scene-train model checkpoints from a directory not writable by untrusted parties, and do not modify the checkpoint while it is being loaded. Libtorch archive deserialisation is not safe for untrusted input. The host API accepts only regular files and rejects files over 1 MiB (ample for this fixed model), but this file-size cap cannot bound decoded archive storage or make malicious archives safe. Trust is a security prerequisite, including for direct `torch::load` calls.

For native inference compile against `model.h`/`model.cpp` and the same libtorch, construct `scene_train::ResidualNet model;`, `torch::load(model, "new-model.pt");`, call `model->eval()` under `torch::NoGradGuard`, then `model->forward(input)` with float32 NCHW RGB values scaled to [0,1]. Model output is unconstrained. Use the same model definition/version as the checkpoint; this is a C++ libtorch archive, not TorchScript/ONNX.

Initialisation uses the CPU random generator only; no CUDA device discovery is performed in CPU mode. Seeded initialisation and patch selection are deterministic for repeat runs with the same software/device; deterministic libtorch algorithms are requested and CPU uses one thread. Cross-version/platform/device bit identity is not promised. CUDA may reject unsupported deterministic operations or require runtime configuration (for example `CUBLAS_WORKSPACE_CONFIG`); CUDA behaviour is unverified and no GPU was used for these tests. Validation of small synthetic fixtures establishes software behaviour only, not scene performance.

## Offline host-ABI inference

Apply a **trusted** locally produced checkpoint to one P6 image on the CPU:

```sh
./build/scene-infer --input /path/cheap.ppm --output /path/new-output.ppm --checkpoint /path/model.pt
```

All three paths are required: the host ABI needs a checkpoint and provides no default model. This uses `scene_host_load` and `scene_host_infer`, not training or a GPU. The input has the P6/maxval/raster rules and dimension limits above; its total header is additionally limited to 65,536 bytes and it must be a regular file. The output is a same-resolution P6 RGB8 image: input bytes are scaled by 255, with no colour-space conversion; finite model results are clamped to [0,1], multiplied by 255 and rounded to nearest, ties to even. Nonfinite results fail without creating an output.

Output creation is exclusive: existing files, input/checkpoint paths, hard links and symlinks (including dangling links) are never overwritten. Maintain exclusive control of the output directory and its path throughout inference: it must not be writable by untrusted parties, and no other process, including a trusted concurrent process, may modify its entries or change the path. Failure cleanup removes the created name by path, and concurrent directory changes can invalidate that ownership assumption. A write/close failure removes the newly created output where possible; interruption may leave a partial file, and power-loss durability is not guaranteed. Use a new output name per run. Checkpoint deserialisation has the trust and size caveats above. This is stateless RGB processing, not a scene reconstruction or image-quality claim.

## Scene experiment requirements

This is a **neural rendering demonstration, not DLSS5**. It has now been trained and evaluated once on the Teen Kitty scene-pair capture (31/10/2026, `pairs-20261031-b`; run report `../.scratch/scene-runs/pairs-20261031-b/report.md`, ignored local evidence): on the locked test frames the scene-trained model improved PSNR/SSIM and the lighting proxy over the cheap baseline and slightly regressed the detail-gradient proxy. That single-asset run establishes nothing beyond itself. Its RGB-only, 7×7 receptive-field model cannot guarantee recovery of missing texture or unseen lighting, and has no motion, exposure, history or temporal consistency inputs.

Capture aligned, same-resolution full and cheap render pairs from identical scene, pose, camera, lighting, exposure and colour state. Record the renderer/build, asset hash, capture IDs, material settings, camera, pose/time, lights, exposure, colour pipeline, resolution, temporal reset and image hashes. Verify repeat-capture stability and inspect edge alignment before training. A post-process blur is not evidence of cheaper rendering; distinguish it from a genuinely cheap render.

Predefine disjoint training, validation and locked test groups by entire camera/pose trajectories and lighting conditions rather than random adjacent frames. Check duplicates and correlated captures across splits; keep test data out of model selection. Asset-specific results do not establish generalisation to unseen assets. Archive the commands, seed, manifests, logs, build and checkpoint hash.

Run `scene-infer` on every locked test cheap frame, then compare full, cheap and neural images using `scene-integration-metrics` built by CMake or the standalone `scene-demo/metrics.cpp`. Report per-frame and grouped results for **both** cheap and neural against full, including regressions. PSNR and SSIM use encoded RGB/luma; gradient and tile-mean errors are detail and lighting proxies, not physical lighting measures. Inspect labelled images and sequences for artefacts and flicker. Report full rendering, cheap rendering, inference and combined cheap-plus-inference timing at the same resolution; offline CPU timing is not real-time evidence.

The CTest suite includes one end-to-end CPU software integration test. It creates
separate tiny synthetic aligned P6 training and validation pairs in an isolated
temporary directory, runs the real training and inference CLIs and the
`scene-demo/metrics.cpp` executable, checks the report and compares inference
against the saved checkpoint's direct model output and host conversion. These
numeric fixtures are **not Teen Kitty** and are not evidence of scene quality.
The metrics tool is built from the adjacent `scene-demo` source as part of this
build; no Python or GPU is used. Run its separate CPU metrics tests too:

```sh
../scene-demo/test-metrics.sh
```
