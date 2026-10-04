# Controlled neural-pass comparison — 03/10/2026

Capture: `capture-20261003-161658`. RTX4070 Ti SUPER, Linux NVIDIA615.71.09, Proton Experimental, NVIDIA Windows DX12 sample. MangoHud0.8.4 Vulkan layer measures CPU-observed presentation intervals; NOT GPU timestamp duration or headset latency.

## Method

- Fresh application per run, unchanged initial camera/scene; no camera/controller input or setting changes.
- Both modes load the identical ReShade/add-on/model. Off uses F10, confirmed by logged DISABLED; on remains enabled.
- 30-second MangoHud autostart delay; off toggled around9seconds, leaving >15seconds in requested condition before capture.
- Three20-second per-present captures per mode per resolution; pair order off/on, on/off, off/on. Resolutions tested sequentially, not randomised.
- Verified every logged swapchain creation/resize and neural output matches requested size. NVIDIA sample initial VSync flag changes to UI default false during rendering; ordinary DLSS remains enabled.
- Corrected launcher Windows arguments from ignored -w/-h to -width/-height. Previous runs unintentionally used desktop-sized2560x1358.
- Local MangoHud unpacked; no system install. Same FPS-only overlay and error-level NVAPI logging both modes. Add-on heartbeat logging remains enabled.

## Pooled results

| Resolution | Neural pass | Mean FPS | Mean interval ms | P95 ms | P99 ms | Across-run FPS range |
|---|---|---:|---:|---:|---:|---:|
|2560×1358|Off|425.7|2.349|2.670|2.786|423.6–428.3|
|2560×1358|On|81.2|12.310|13.090|13.548|80.0–82.1|
|1280×720|Off|954.1|1.048|1.180|1.238|951.3–957.5|
|1280×720|On|211.7|4.723|5.147|5.328|211.4–212.1|

Mean FPS =1000/pooled mean interval, not mean instantaneous FPS. P95/P99 use nearest-rank whole-frame presentation intervals. Rough net enabled overhead9.961ms at2560×1358,3.675ms at1280×720; not isolated neural-pass timing. Pooled repeat agreement is descriptive, not a formal confidence interval.

## Validation and limitations

All12 captures passed `python3 benchmark/validate.py benchmark/capture-20261003-161658`: dimensions, >15s off warmup, capture duration, no logged failure/automatic disable, heartbeat gaps<=10s including capture edges. Sparse heartbeat logs CANNOT exclude unlogged transient evaluation failures. Independent read-only review initially found validation/warmup weaknesses; corrected postvalidation passed final review. No benchmark/Proton/demo/wineserver processes remained; GPU responded normally after completion.

Desktop compositor/background workloads, telemetry overhead, thermals/clocks, and native driver vs Proton contributions are not isolated. GPU telemetry reports high utilisation; on-run temperature means around76–77.5°C vs off around73–75°C. No headset or stereo rendering/encoding/compositor integration tested. FPS is application throughput, not physical display refresh. Smaller image renders less detail;720p result is not proof of acceptable Quest3 resolution/VR performance.

## Repeat

Close existing demo and avoid using the PC during capture (script focuses demo and sends F10):

```bash
cd /home/michael/Projects/dlss-testbed
python3 benchmark/benchmark.py
python3 benchmark/validate.py benchmark/capture-YYYYMMDD-HHMMSS
```

Allow approximately12minutes. New captures do not overwrite old results. CSV/result/validation JSON and ReShade logs retained per run. Runtime proprietary patch provenance and upstream resource/settings risks remain unchanged.
