# 1440 × 1440 per-eye VR timing results

Six complete 600-frame captures, order off/on/on/off/off/on. First 120 submitted frames discarded per run; 1440 measured frames per mode. All launchers exit 0 and complete PASS; retained Proton diagnostics checked by launcher. Head motion was user-controlled, not a fixed camera path.

| Metric (ms) | Off mean | On mean | Off pooled p99 | On pooled p99 |
|---|---:|---:|---:|---:|
| scene_ms | 0.113 | 0.111 | 0.138 | 0.128 |
| neural_ms | 0.000 | 11.441 | 0.001 | 12.268 |
| compose_ms | 0.141 | 0.084 | 0.148 | 0.119 |
| gpu_ms | 0.254 | 11.636 | 0.283 | 12.480 |
| frame_wall_ms | 1.052 | 13.048 | 2.717 | 14.340 |
| cadence_ms | 11.114 | 13.105 | 12.812 | 14.399 |

Runtime predicted display period: 11.111111 ms (90 Hz). Neural-on GPU span exceeded this period in all 1440 measured samples; neural-off in none. Average neural-on submission cadence corresponds to approximately 76.3 submissions/s, versus 90.0 off. This is not physical display FPS. No measured cadence sample exceeded 1.5 predicted periods; this does not establish zero compositor drops.

All timing scope/overhead limitations from TIMING.md apply. GPU span measures both eye pipelines, excludes query resolve; neural stage includes transition; frame wall includes GPU/runtime waits and is not CPU execution cost. No transport/display latency or reprojection telemetry. Simple scene costs ~0.11 ms, not representative of a complex app.

Conclusion: visually promising 1440 per-eye configuration is not meeting the 90 Hz application rendering budget; runtime presentation may mask this perceptually. Next test: same instrumentation at 1080 per eye, or retain 1440 quality and investigate reducing neural cost. Do not infer comfort or full-rate rendering from subjective responsiveness.

Raw logs and per-run JSON: vr/logs/timing/.
