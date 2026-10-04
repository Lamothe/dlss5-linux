# 1080 × 1080 per-eye timing results

Six complete captures in off/on/on/off/off/on order; exit0/PASS all runs. First120 of600frames discarded each:1440 measured frames/mode. User head motion uncontrolled. Same instrumentation and limitations as TIMING.md.

| Metric (ms) | Off mean | On mean | Off pooled p99 | On pooled p99 |
|---|---:|---:|---:|---:|
| scene_ms | 0.066 | 0.061 | 0.070 | 0.065 |
| neural_ms | 0.000 | 7.918 | 0.001 | 8.251 |
| compose_ms | 0.082 | 0.052 | 0.086 | 0.053 |
| gpu_ms | 0.148 | 8.031 | 0.167 | 8.364 |
| frame_wall_ms | 0.942 | 9.145 | 2.600 | 10.688 |
| cadence_ms | 11.104 | 11.156 | 12.782 | 13.379 |

Neural-on maximum GPU span: 8.900 ms.
Mode neural=0: GPU span exceeds display period 0/1440; frame wall exceeds period 0/1440; cadence above1.5periods 1/1440. Display periods perrun: 11.111111, 11.111111, 11.111111 ms.
Mode neural=1: GPU span exceeds display period 0/1440; frame wall exceeds period 4/1440; cadence above1.5periods 6/1440. Display periods perrun: 11.111111, 11.111111, 11.111111 ms.

Budget flags are not confirmed compositor drops or headset latency. Rates inferred from cadence are application submissions, not physical display FPS. User visual judgement at1080 pending. Complex scenes will cost more than this minimal scene. Rawlogs vr/logs/timing-1080/.
