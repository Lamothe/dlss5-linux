# Captured consecutive-stage chain feasibility — 25/10/2026

## Scope and method

This bounded experiment tests the smallest evidence-supported consecutive pair, slots 24→25. It is not a live-input evaluator, an image inference result, or a performance claim. Three retained frames were used; each s23 pool was uploaded once, the vendor-launched slot24 `tilesync` kernel ran, then slot25's `chained` kernel consumed the same still-resident pool. The intermediate readback was diagnostic only; it was never re-uploaded. Both vendor weight records were bound separately. The DLL was read as data after disabling core dumps/process dumpability and checked against the pinned SHA-256; only its native SM89 cubins were loaded through CUDA Driver API. No Windows host code, DirectX, Proton, or proprietary payload was added to Git. Swap, driver persistence and memory erasure remain uncontrolled; use rights remain unresolved.

## Evidence and result

The retained s23/s24/s24b/s25/s25b ladder captures validate. Independent pairs s24/s24b and s25/s25b passed, and the s24→s25 write-set attribution has zero unattributed bytes. The contracts establish that slot24 writes its output at w1 (pool offset 20,610,048, 344,064 bytes) and sync at w6; slot25 consumes slot24's output at w0 and sync at w7, reads its additional residual from w1, and writes output at w2. The captures are the s23 input/s24 intermediate/s25 final boundaries.

`vr/native-captured-chain-probe.py execute` passed on an NVIDIA GeForce RTX 4070 Ti SUPER (driver API 13040; pinned DLL SHA-256 `4b8d19bc3eff58a084f5eca7489c921501c203450169fb82ff4f649a4482ba05`): six kernel launches across three frames. Every slot24 intermediate and slot25 final whole-pool comparison matched its corresponding vendor capture bit-for-bit (zero differing bytes); pool and weight allocation guards remained intact and clean-run weights were unchanged. Per-frame hashes and run summary are retained only in ignored `/tmp/native-chain-feasibility-run/run.json`.

Controls:

- `skip-first` omitted both launches, preserving the s23 pool exactly. Its intermediate differed from s24 by 344,085/344,104/344,110 pool bytes; final (still s23) differed from s25 by 684,687/685,264/685,277 bytes. The second launch is intentionally omitted: slot25's residual w1 binding aliases bytes slot24 ordinarily writes, so this control does not provide a valid slot25 input. The successful control result is the intact unchanged input and its non-match to both later boundaries—not a claimed stage25 dependency test.
- `wrong-first` rotated slot24's learned record before its launch. Intermediate differences from s24 were 319,427/319,441/319,425 bytes; final differences from s25 were 635,487/636,181/636,124 bytes. Both allocation guards stayed intact; weight content was intentionally corrupted, so this does not assert weight immutability. Run summary is ignored/local at `/tmp/native-chain-wrong-first/run.json`.

The evidence supports that native consecutive execution can carry the actual slot24 pool output into slot25 and reproduce these captured boundaries on these inputs. It does not establish general scheduling, full model graph execution, preprocessing, decoder/output conversion, temporal/reset correctness or neural image generation. The first broader blocker remains assembling the unimplemented full graph and its live input/output path. This experiment is limited to this pair and should not be presented as general graph support.

## Reproduction and verification

Run serially; the program invokes the CUDA Driver API and uses the pinned local DLL under `vr/model/`:

```bash
python3 vr/native-captured-chain-probe.py execute /tmp/native-chain-feasibility-run
python3 vr/native-captured-chain-probe.py skip-first /tmp/native-chain-skip-first
python3 vr/native-captured-chain-probe.py wrong-first /tmp/native-chain-wrong-first
```

Run summaries contain hashes/verdicts only, not pool/weight bytes. The CPU-only replay/ladder suites and Python compilation are separate regressions; they do not substitute for this GPU result. The existing isolated per-slot replay remains unchanged.
