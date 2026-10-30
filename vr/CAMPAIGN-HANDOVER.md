# DLSS5 native Linux migration — current handover

## Recovery implementation — 26/10/2026

The user authorised recovery of the stalled migration under [NATIVE-RECOVERY-ASSIGNMENT.md](NATIVE-RECOVERY-ASSIGNMENT.md) (supervision `e58d4cb0a5c8494496f4a24c2e111110`, attachment READY). Scope corrections per that order: the missing producer-to-launch observation and incomplete decoder address/CFG bounds remain real blockers on the proprietary replay/capture route with all existing safety/rights constraints; they do **not** prohibit implementing explicitly defined native tensor pipelines with original synthetic data and weights. Synthetic tests are not vendor correspondence, and no public code licence grants rights to NVIDIA model data.

First executable deliverable implemented: a typed CPU input-lift slice (RGB → 16 features → 32 channels) in `vr/native-input-lift.hpp`, with `vr/tests/native-input-lift-test.cpp` and runner `vr/test-native-input-lift.sh`; full specification, attribution and qualifications in [NATIVE-INPUT-LIFT.md](NATIVE-INPUT-LIFT.md). The feature composition, conditioning derivation, deterministic noise generator, stored 16→32 lift permutation and constants are adapted from the pinned MIT-licensed `mauri870/dlssnr-pytorch` commit `3beef7935a8ef976b47ab78b906e75600c52d2de` (licence text retained in NATIVE-INPUT-LIFT.md); the pinned `pre_post.py`, `runtime.py`, `weights.py`, `io.py`, `layers/swin.py` and LICENSE were fetched and reviewed at that pin. No weights, runtime assets or proprietary payloads were fetched or used; the extraction-based weight route remains uncleared and untouched. Tensor layout, padding/orientation, conditioning, seed, f16 rounding (ties to even, software conversion), float32 accumulation order and uint32 noise arithmetic are defined explicitly; the gaussian noise stage is documented as platform-libm-dependent and is not claimed bit-exact across platforms.

Verification (26/10/2026): `vr/test-native-input-lift.sh` passes 5316 checks in the strict `-O2` build and again at `-O0` (both with `-ffp-contract=off`; see the Rounding semantics section of NATIVE-INPUT-LIFT.md for exactly what is and is not enforced by the tests) — independently derived binary16 literals (from the IEEE 754 definition, cross-checked with numpy float16 and the compiler `_Float16`), round-trip of every f16 pattern (65 536 scanned; NaN payloads excluded by policy) plus 200 000 pseudorandom float32 draws (NaN bit patterns, about 1 in 256 of the draws, skipped) against the `_Float16` oracle, noise integer literals derived independently with Python (exact), gaussian literals from an independent scalar C derivation (±2^-9 tolerance for libm variance), lift permutation anchors + exhaustive 512-slot bijection coverage + full decode mapping, hand-computed end-to-end cases including f16 tie-rounding (`1+2^-11 → 1.0`, `1+3*2^-11 → 1.001953125`), changed-input sensitivity with spatial locality, row-stride independence, determinism, negative controls where every rejected `input_lift` call with an output buffer and the rejected `decode_lift` call are verified to leave the output untouched (the null-output rejection and stage-API status negatives verify status codes), and zero-allocation accounting. The sanitised build is skipped on this workstation because no sanitizer runtime can be linked (libasan/libubsan RPMs absent; the runner verifies the skip reason is the missing runtime and reports the exact linker error) — an environment gap, not a code failure. Existing CPU regressions (slot replay, ladder report/names, input layout) pass unchanged. Independent read-only review (two passes, 26/10/2026): arithmetic, memory safety, upstream fidelity and attribution were verified correct in both passes (feature order, colour chain, noise literals, conditioning on both branches, lift bijection and anchors, f16 edge cases, and a static recount matching the suite's reported check total); pass one raised documentation-accuracy and oracle-independence findings, pass two confirmed those fixes and raised four residual defects (sanitiser-skip linker-error matching, a comparison-count estimate, review-outcome wording, a comment typo) plus an infinite-weight validation hole — all fixed (weights are now validated f16-representable and finite) and reverified by the 5316-check suite in this change set.

Second executable deliverable implemented (26/10/2026): the pre/post-block arithmetic boundary in `vr/native-pre-post.hpp` — the e4m3 rounding boundary (`e4m3_bits`/`e4m3_from_bits`/`e4m3`/`is_e4m3`, nearest with ties to even, saturating ±448, canonical NaN, subnormal grid `m·2^-9` below `2^-6`) and the 2x2 f16 pooling `pool2x2` (column pairs added in float32 and narrowed to f16, row sums added in f16 — computed in float32 and narrowed, where for f16 addends the double rounding is benign and equals a strict binary16 add — quarter exact in float32, result rounded to the e4m3 grid), pinned to `quant.py` "q8" and `layers/swin.py` "_pool" at the same MIT commit with the full licence notice in [NATIVE-PRE-POST.md](NATIVE-PRE-POST.md). Tests in `vr/tests/native-pre-post-test.cpp` with runner `vr/test-native-pre-post.sh`: 198 checks in the strict `-O2` build and again at `-O0` (both `-ffp-contract=off`), centred on a brute-force definition-based e4m3 oracle over the full code/value space and 200 000 pseudorandom probes, a `_Float16` pooling oracle with padded-stride layout checks, hand-derived literals including an observable grouping discriminator (`[4.25, 2^-9; 0, 2^-9] → 1.0`, where column-first grouping would give `1.125`), full 256-code sweep round-trips, negative controls with untouched-output verification on every rejection path, and zero-allocation accounting. No external e4m3 reference exists on this workstation (numpy 2.4.6 has no float8 dtype; `ml_dtypes`/`torch` are not installed), so the oracle is definition-based brute force plus hand literals only — no external validation is claimed. Independent read-only review (26/10/2026) verified arithmetic fidelity against the pinned `quant.py`, memory safety, validation totality, honest check counting, attribution and style consistency, and raised four documentation/comment inaccuracies (encode-literal case count 34 not 33, an incorrect "exact sum" justification for the row-sum narrowing, a wrong post-rounding range in an encoding comment, a plane-size test comment) — all fixed and reverified by the 198-check suite in this change set. The sanitiser skip is the same recorded environment gap. Existing CPU regressions (ladder-report 19 controls, ladder-names 14, input-layout schemas/packing + 9 invalid-layout controls) pass unchanged.

Third executable deliverable implemented (26/10/2026): the post-block input blend in `vr/native-blend.hpp` — the elementwise f16 mix `mix_element` (`f16(skip·skip_gain + f16(up·main_gain))`, the float32 product with one binary16 rounding and the skip product exact in a fused float64 add rounded once, directly double→binary16) and `blend` (2x replication with replicate padding/crop and per-channel gains), pinned to `pre_post.py` `_mix`/`PostBlock.blend` at the same MIT commit with the full licence notice in [NATIVE-BLEND.md](NATIVE-BLEND.md). Tests in `vr/tests/native-blend-test.cpp` with runner `vr/test-native-blend.sh`: 403 checks in the strict `-O2` build and again at `-O0` (both `-ffp-contract=off`), centred on hand-derived binary16 literals (overflow band 65504/65520, subnormal ties, binade-top rounding carry), a product pre-rounding tie discriminator (`1.001953125`, not `1.0029296875`), 200 000-pair `_Float16`/double oracles for the rounding and the mix, a spatial oracle over 2000 randomised configurations with determinism checks, hand-computed crop/replicate-pad/dual-clamp known answers pinning the spatial reduction, 32-channel permutation and changed-input-sensitivity controls, every stride-bound negative with untouched-output verification, and zero-allocation accounting. Development fuzz exposed a real double-rounding defect in a float32 shortcut inside `f16_from_double` (f16 subnormal zone and midpoints); it now rounds directly from double. Independent read-only review (two passes, 26/10/2026): the first pass verified arithmetic fidelity against the pinned `_mix`, memory safety and validation totality, and raised false tie claims in the discriminator documentation, an overstated second-literal comment and an incomplete stride contract; the confirming pass verified those fixes and raised four residual honesty/coverage defects (a false 'unreachable' carry comment with unpinned carry tests, overstated blend-oracle independence, untested skip-side upper stride bounds, an unstated dst-aliasing precondition) — all fixed and reverified by the 403-check suite in this change set. The sanitiser skip is the same recorded environment gap. Existing CPU regressions (ladder-report 19 controls, ladder-names 14, input-layout schemas/packing + 9 invalid-layout controls) pass unchanged. Synthetic correctness only: this establishes no vendor correspondence.

Fourth executable deliverable implemented (26/10/2026, commit `d3930b4`): the output conversion in `vr/native-compose.hpp` — `project_residual` (32-channel wide batch to 3-channel residual, `sum += weight*value` with sequential float32 accumulation in ascending channel order) and `compose_pixel`/`compose` (`network = clamp01(frame + residual·0.25)`, the Intensity blend and the `intensity > 1` bounded luminance-ratio amplification), pinned to `pre_post.py`'s compose semantics at the same MIT commit with the full licence notice in [NATIVE-COMPOSE.md](NATIVE-COMPOSE.md). Tests in `vr/tests/native-compose-test.cpp` with runner `vr/test-native-compose.sh`: 60 137 checks in the strict `-O2` build and again at `-O0` (both `-ffp-contract=off`), centred on hand-derived literals (the ascending-accumulation order discriminator `1.0`, not `1 + 2^-23`; the amplification clamp band pinned exactly at 2.0/0.5), a separately written same-process scalar re-derivation checked bit-exactly (the hand literals are the independent pins), permutation/sensitivity/determinism controls, negative controls covering every rejection path including the `kMaxPixels` pixel-count branch for both entry points, and zero-allocation accounting. The double-precision pow with clamp-in-double-then-narrow is a disclosed deviation from the reference's float32 torch pow, not a gap-fill. Independent read-only review (26/10/2026) verified the five earlier findings fixed (double→float narrowing guards, `kMaxPixels` test coverage, the build contract, oracle-independence wording, doc corrections) and raised five residual items (untouched-output claim strength, an "independent" wording slip, an incomplete `compose_pixel` precondition, a dead constant, build-contract placement) — all fixed in this change set. The sanitiser skip is the same recorded environment gap. Synthetic correctness only: no vendor correspondence.

Fifth executable deliverable implemented (26/10/2026): the first learned-weight-execution-graph slice — the Swin body's scalar primitives in `vr/native-swin-primitives.hpp` (`swin_fma`, the double-computed fused multiply-add with its float64 intermediate; `swin_activation`, the cubic SiLU with the unclamped outer multiply; `swin_rsqrt`, the CPU deterministic reciprocal square root; and `swin_norm_scale`, the 32-dimension head reciprocal L2 norm with the interleaved four-chain float32 accumulation, binary16 halves and `f16(0.000062)` floor), pinned to `swin.py` `_fma`/`activation`/`_rsqrt`/`_norm_scale` at the same MIT commit with the full licence notice in [NATIVE-SWIN-PRIMITIVES.md](NATIVE-SWIN-PRIMITIVES.md). Tests in `vr/tests/native-swin-primitives-test.cpp` with runner `vr/test-native-swin-primitives.sh`: 141 056 checks in the strict `-O2` build and again at `-O0` (both `-ffp-contract=off`), centred on hand-derived dyadic activation literals (clamp boundary, unclamped outer multiply, `-0.0` sign pins), the fma double-rounding discriminator (`-2^-46`, where a float32 multiply-add gives `0`), exact-root reciprocal-square-root literals, and the norm's accumulation discriminator (stepwise float32 rounding with ties to even at the binary16 midpoint gives scale `1.0`; a double-accumulation alternative gives `0.9995120763778687`, both asserted), plus 100 000-vector/20 000-activation/20 000-rsqrt randomised same-process oracles checked bit-exactly, determinism reruns and zero-allocation accounting. Disclosed honestly: random corpora do not separate the accumulation semantics at the binary16 output, and no tested case separates a swapped (g, f) chain order at binary16 precision — the hand-derived discriminator is the pin. The oracles reuse the shared f16 helper and constants, so the binary16 steps are pinned only by the hand literals. The sanitiser skip is the same recorded environment gap. Synthetic correctness only: no vendor correspondence.

Sixth executable deliverable implemented (26/10/2026): the Swin body's window partition in `vr/native-swin-windows.hpp` — `swin_window_grid` (the shifted window grid: `front = -4·shift`, `windows = ceil((extent + front)/8)`, shifts 0 and −1 only, matching the reference's raise on anything else), `swin_to_windows` (channels-first planar batch → `[windows, 64, C]` tokens in tile order with the documented zero pad, exact capacity formula and untouched outputs on error) and `swin_from_windows` (the inverse crop; edge pixels dropped, never written back), pinned to `swin.py` `_window_grid`/`_to_windows`/`_from_windows` at the same MIT commit with the full licence notice in [NATIVE-SWIN-WINDOWS.md](NATIVE-SWIN-WINDOWS.md). Tests in `vr/tests/native-swin-windows-test.cpp` with runner `vr/test-native-swin-windows.sh`: 4 360 checks in the strict `-O2` build and again at `-O0` (both `-ffp-contract=off`), centred on hand-derived grid literals (including the maximum axis pinning `kMaxWindows`), token-index pins — a full 64-token tile-order decode re-derivation (unshifted) plus genuinely hand-derived shifted pins including a 2x2-window fixture pinning the window-batch index order absolutely and an asymmetric-shift fixture pinning the shift-axis assignment — with stride-padding invariants, a 500-case bit-exact round-trip oracle over special values (signed zeroes, infinities, NaN payloads) with determinism reruns, single-pixel locality and channel-permutation controls, negative controls for every validation rejection class with sentinel-checked untouched outputs and the exact capacity boundary (317 accepted, 316 rejected for the single-window fixture), and zero-allocation accounting. Pure layout arithmetic: values pass through bit-exactly (the pad is pinned as `+0.0` by bit comparison); no vendor correspondence is claimed. The sanitiser skip is the same recorded environment gap. Independent read-only review (26/10/2026, two passes): pass one verified the layout arithmetic, memory safety, validation order, attribution and check counts against the pinned reference and raised eight findings — the window-batch index order and shift-axis assignment were unpinned (fixed with absolute hand pins in a 2x2-window fixture and an asymmetric-shift fixture), plus independence overclaims in fixture wording, missing untouched-output checks for the grid API, a wrong null-pointer parenthetical, a garbled stride-contract comment, an unstated read-side precondition and an unpinned pad sign — all fixed and reverified by the 4 360-check suite; pass two independently re-derived every pin (including catching and correcting one mis-derived development pin before commit), confirmed the fixtures discriminate a transposed window order and swapped shift axes, reproduced the 4 360 check count and passed, with one capacity-phrasing wording nit also fixed.

Seventh executable deliverable implemented (26/10/2026): the Swin body's e4m3 linears in `vr/native-swin-linear.hpp` — `swin_linear` (the `_linear` default path: `x @ weight.T` on e4m3-grid operands, each product exact in float32, accumulated in the explicit sequential k-order float32 accumulator this slice convention adds) and `swin_hidden` (the `_hidden` non-int4 path: the e4m3-rounded cubic SiLU of the product), pinned to `swin.py` `_linear`/`_hidden` and `int8.py` `mm` (whose default is a plain matrix product) at the same MIT commit with the licence notice in [NATIVE-SWIN-LINEAR.md](NATIVE-SWIN-LINEAR.md). Tests in `vr/tests/native-swin-linear-test.cpp` with runner `vr/test-native-swin-linear.sh`: 290 checks in the strict `-O2` build and again at `-O0` (both `-ffp-contract=off`), centred on hand-derived dyadic known answers, the accumulation-order discriminator (products `[2^16 x 128, 0.5, 0.5]` pin the k-order float32 accumulator at `2^23` = 8388608, where summing the two tail products first or exact summation gives the provably distinct 8388609; the discriminator's scope is pinned too — moving a single trailing 0.5 to the front still gives 8388608 via the recurring tie-to-even midpoint), hand-known hidden activations as exact rationals of the cubic constants (including the +448 saturation; the −448 end is unreachable through `swin_hidden`: the cubic SiLU bottoms out near −0.53 and the ≤ −4 ramp factor cancels to zero, giving −0.0 only via the outer multiply with negative x), permutation/sensitivity/determinism controls on an exact corpus with pairwise-distinct weight rows and sensitivity deltas (corpus guards plus a hand-derived anchor row), negative controls for every rejection class — including the e4m3 operand-grid class on both operand sides — with sentinel-checked untouched outputs, and zero-allocation accounting. The reference's default plain-product path states no summation-order contract (the `int8.py` phrase "up to float32 summation order" qualifies its optional INT8 W8A8 mode); the k-order convention is this implementation's explicit choice and no vendor correspondence is claimed. Independent read-only review (26/10/2026, two passes): pass one re-derived the arithmetic, memory safety and check counts as correct and raised five findings — an untested untouched-output gap in the negative fixture (fixed with an added sentinel check), a vacuous weight-row permutation corpus (fixed with pairwise-distinct rows/deltas, corpus guards and a hand-derived anchor row), an over-broad accumulation-discriminator claim (fixed by naming only the discriminated alternatives and pinning the non-discriminated reorderings as explicit counterexample checks), a mis-attributed `int8.py` summation-order quote and the ±448 saturation overclaim — all fixed and reverified; pass two confirmed those fixes and the licence retention, per-entry-point negative coverage (now exercised through `swin_hidden` as well), validation-order and sign-of-zero wording residuals it raised, and its one remaining coverage nuance (the zero-K-pair and over-wide-weight rejection branches) was closed with dedicated negative cases; the 290-check suite passes after every fix. The `_mlp` assembly remains the next slice. The sanitiser skip is the same recorded environment gap.

Eighth executable deliverable implemented (26/10/2026): the `_mlp` body assembly in `vr/native-swin-mlp.hpp` — `swin_store` (the stored-result double rounding f32→f16→e4m3), `swin_mid` (the grouped middle projection matching the reference's `einsum("ntgj,gdj->ntgd", groups, mid)` with per-element store narrowing), `swin_mlp_tail` (the contract projection plus the f16 residual scale, `residual·scale + product` rounded as two separate float32 roundings, never one fused multiply-add) and the composed `swin_mlp` (`swin_hidden` → optional grouped mid path or single-head path → tail), pinned to `swin.py` `_mlp`/`store` at the same MIT commit with the licence notice in [NATIVE-SWIN-MLP.md](NATIVE-SWIN-MLP.md). Tests in `vr/tests/native-swin-mlp-test.cpp` with runner `vr/test-native-swin-mlp.sh`: 2 780 checks in the strict `-O2` build and again at `-O0` (both `-ffp-contract=off`), centred on hand-derived dyadic known answers, the store double-rounding discriminator (`367.875` = 1.875·192 + 1.125·7 stores as `384` where a direct e4m3 rounding gives `352`), the tail two-rounding discriminator (`8388609·1.5 = 12582913.5` rounds to `12582914` and ±0.5 leaves it there, where a fused multiply-add would give `12582913`), head-grouping, permutation and changed-input-sensitivity controls on exact corpora, a bit-exact composed-vs-staged-chain equivalence control over a pseudorandom e4m3 corpus, negative controls for every rejection class of all three entry points (including the reachable `kMaxLinearDim` scratch-cols branch through `swin_mlp`) with sentinel-checked untouched outputs and untouched scratch, determinism reruns and zero-allocation accounting. Development history is disclosed in the doc: an earlier draft discriminator pair (`8388607`/`8388608`) was itself flawed — its product is exact at half-ulp below 2^23 and discriminates nothing — and the asserted pair is the corrected one. Independent read-only review (26/10/2026) re-derived the pinned literals, both discriminators, validation order and the static check count, and raised three findings — an overstated exactness claim in the test header (scoped to the e4m3 operand products, with the residual-scale product disclosed as deliberately hand-rounded through the pinned tail roundings), a grammar slip, and headings diverging from the sibling docs (aligned to "Explicit choices and disclosed limitations" / "Test evidence (26/10/2026)" / "What this does NOT establish", the last completed in this change set) — all fixed and the 2 780-check suite passes after every fix. The sanitiser skip is the same recorded environment gap. Synthetic correctness only: no vendor correspondence.

Ninth executable deliverable implemented (26/10/2026, continued under supervision `3d08a7b9869b4bdc8813b3795947398e`): the `_attention` body of the windowed-attention blocks in `vr/native-swin-attention.hpp` — `swin_deswizzle_index`/`swin_deswizzle_bias` (the `_deswizzle_bias` fragment-order permutation), `swin_exp_bias` (the deswizzle through `_fma(…, EXP_SLOPE, EXP_OFFSET)`), `swin_softmax_numerator` (the exponent-ramp softmax numerator: f16 ramp → clamp `[EXP_LOW, EXP_HIGH]` → mantissa-bits-into-exponent transplant), `swin_softmax_denominator_row` (the f16 64-key add tree, blocks → pairs → halves → members), `swin_softmax_row`/`swin_softmax` (numerator → denominator → reciprocal → `store`), `swin_normalise_head_row`/`swin_normalise_qkv` (Q/K one rounding from the float32 accumulator with the head scale folded into the query scale, V the stored double rounding), `swin_logit_dot`/`swin_logits` and `swin_context` (the two einsums, the context summed over all 64 keys with a single `store`, matching the eager reference semantics) and the composed `swin_attention` (residual line through the MLP slice's `swin_mlp_tail`), pinned to `swin.py` `_deswizzle_bias`/`_normalise_qkv`/`_softmax_numerators`/`_denominator`/`_softmax`/`_attention` and the constants at the same MIT commit, with `dlssnr/attention.py` and `tests/test_triton_kernels.py` as corroborating shape/rounding evidence and the full licence notice in [NATIVE-SWIN-ATTENTION.md](NATIVE-SWIN-ATTENTION.md). Tests in `vr/tests/native-swin-attention-test.cpp` with runner `vr/test-native-swin-attention.sh`: 305 967 checks in the strict `-O2` build and again at `-O0` (both `-ffp-contract=off`), centred on hand-derived dyadic known answers: deswizzle anchors plus the exhaustive 4096-slot bijection/decode map, the exhaustive 63 488-pattern finite f16 sweep pinning the single-rounding `swin_fma` against a two-rounding re-derivation, the numerator bit-transplant pins (clamp boundaries, ramp ties-to-even) and the fma-vs-two-roundings discriminator (0.00262451171875 vs 0.002685546875), the denominator sum-order discriminators (2050 vs 2048 at the `2048 + 1` ties, block/pair/member-half nesting), the head_scale/norm_scale grouping discriminator (0.025390625 vs 0.0234375) and the q-vs-v rounding asymmetry (367.875 → 352 vs → 384), the logit accumulation-order discriminator (200704 vs 200704.125) and the store double rounding through the context (367.875 → 384), permutation/sensitivity controls, bit-exact batch-entry and assembly-vs-staged-chain equivalence on deterministic pseudorandom corpora with padded strides, sentinel-checked untouched outputs and assembly scratch on every rejection class (including the non-finite classes on every scanned operand, the bias-entry output-rows pairing and the single-head-scale-row pairing), determinism and zero-allocation accounting. An independent read-only review of the delivered slice (26/10/2026) re-derived the arithmetic against the pinned source and found no arithmetic or memory-safety defects; its six honesty/coverage findings (the missing attribution document and MIT notice, an invalid-value coverage overclaim, a mislabelled denominator-grouping comment, an untestable heads-bound attribution — the tests now claim only the observable accept/reject boundary — a dead temporary and output/head-scale row negatives) were all corrected and reverified in this change set. The int4 `_attention` path remains unimplemented (the documented scope boundary). The sanitiser skip is the same recorded environment gap. Synthetic correctness only: no vendor correspondence.

Tenth executable deliverable implemented (26/10/2026): the block composition of the windowed-attention blocks in `vr/native-swin-block.hpp` — `swin_block` (the `round8` input rounding, the shifted `_to_windows` passes with the residual-source rule, the per-window MLP -> `store` -> attention chain with the one-head unrounded-skip rule, the `_from_windows` reassembly and the `wide = _f16(acc)` narrowing with the accumulator mode) and `swin_core` (`x` doubling as the f16 residual), pinned to `swin.py` `swin_block`/`swin_core` and the `SwinParams` shape contract at the same MIT commit, with the full licence notice in [NATIVE-SWIN-BLOCK.md](NATIVE-SWIN-BLOCK.md). Tests in `vr/tests/native-swin-block-test.cpp` with runner `vr/test-native-swin-block.sh`: 633 646 checks in the strict `-O2` build and again at `-O0` (both `-ffp-contract=off`), centred on zero-weight closed forms (the residual-source discriminator `e4m3(1 + 2^-11) = 1.0` versus the f16 `1 + 2^-10` pass-through and `swin_core`; the one-head unrounded-skip rule and the f16 tie-to-even `wide` narrowing pinned together at `1.50146484375 -> 1.501953125` versus the stored-skip `1.5`; heads = 2 fixtures on `C = 64`, the null-`mid` single-stage and the grouped non-null-`mid` paths), a bit-exact equivalence control of the composed entry against the pinned staged chain over deterministic finite-f16 xorshift corpora (both residual paths, single-head null-`mid` and multi-head grouped-`mid` paths, zero-weight and non-trivial weight corpora, all four shift pairs for both, multi-window grids with front padding, poisoned padded input strides and output sentinels; the non-accumulator mode — `acc_dst` null keeps the `acc` map in scratch, ignores `acc_count` and gives a bit-identical `wide` — and boundary runs at `head_scale = ±65504` and `expand.rows == kMaxLinearDim`), every rejection class with sentinel-checked untouched outputs (including the forwarded `kInvalidControls` shift rejection, the head-scale magnitude bound and the mid-path width bound with their accepting/rejecting edges, and the exact scratch/`acc`-map accounting), determinism and zero-allocation accounting. Development of the slice fixed real defects in the unverified predecessor test draft (two out-of-scope references, three ill-formed temporary-address expressions, a broken xorshift f16 corpus generator that emitted NaN/Inf patterns, a mis-dimensioned multi-head `contract` fixture, a scale/expectation mismatch in the closed forms, a garbage cross-path comparison and unmet expectations for the shift-rejection status) and one type mismatch in the header (`MutableMatrixView` passed where `const MatrixView&` is required); the pinned reference at the same commit was re-fetched to verify the composition semantics line by line. The validation/write contract is stated with its substantiation (the validated operand classes and the `kMaxHeadScale` = 65504 head-scale magnitude bound keep every intermediate within float32 range or on the finite stored grids, so every staged call is guaranteed `kOk`). The int4 mixed-network path and the fused kernels remain out of scope (the documented boundary). The sanitiser skip is the same recorded environment gap. Existing CPU regressions pass unchanged. Independent read-only review of the delivered slice (26/10/2026) returned FAIL on two defects — the unbounded `head_scale` norm-product (finite rows up to `1e38` overflow `_norm_scale * head_scale` to infinity; a zero query dimension then forms `0 * inf = NaN` and `swin_logits`' e4m3 operand check rejects it after `swin_block` has written scratch/outputs, falsifying the write contract) and the missing mid-path width bound (`expand.rows == heads * mid.cols` up to 8192 passed the block pairings while `swin_mlp`'s scratch validation rejects `cols > kMaxLinearDim`) — plus three honesty/coverage gaps (the misdescribed heads = 2 fixture, untested non-accumulator mode, thin equivalence arithmetic). All five findings are fixed in this change set: `swin_normalise_qkv`/`swin_attention`/`swin_block` validate `head_scale` through `validate_head_scale` (finite, magnitude `<= kMaxHeadScale` — an accepted-domain restriction, not an arithmetic change), the pairings mirror the `kMaxLinearDim` scratch width bound, both bounds carry accepting and rejecting controls (including the `1e38` regression case, evidenced: `e4m3(0 * inf) = NaN` and `swin_logits` rejects it), the documented heads = 2 non-null-`mid` closed-form fixture now exists, and the non-accumulator mode and enriched equivalence matrix are pinned. The attention suite gained five checks for the bound class (305 972 total; its earlier 305 967 figure predates the change). Follow-up independent review (26/10/2026) returned PASS with no defects and two findings, both fixed in this change set: a mislabelled negative-test comment (the `heads = 11` case observes the token-channel bound; the pairings' heads bound is now pinned directly through both the entry and the pairing helper — +2 checks, 633 644 -> 633 646) and the deliberately write-only staged out-params (`front_h`/`front_w`, the pass grids), now annotated in the header. The full output of an earlier PASS review round (four nits) was lost to a context-rotation truncation; this fresh review covered the same criteria before commit. Synthetic correctness only: no vendor correspondence.

Eleventh executable deliverable implemented (27/10/2026): the resampling layers in `vr/native-swin-resample.hpp` — `gated_sum_element` (the fused `f16(skip·gain + other)` in one float64 rounding), `swin_project_dot`/`swin_project_narrow`/`swin_project_store` (the 1x1 `_project` einsum, k-order float32 accumulation narrowed to f16), `swin_downsample` (the `_pool`-based pooled learned `C -> 2C` projection with explicit output extents), `swin_upsample_blend` (the learned `2C -> C` projected nearest-2x clamped-replication gated blend), `swin_layer_plain` (round8 of `swin_block`) and `swin_layer_upsample` (blend → `C == 32` f16-residual `swin_core` / quantised-residual staging → round8), plus the `swin_block_scratch_needed` mirror, pinned to `swin.py` `_pool`/`_project`/`_pad_to`/`_gated_sum`/`downsample`/`upsample_blend` and the `SwinLayer` sections at the same MIT commit with the full licence notice in [NATIVE-SWIN-RESAMPLE.md](NATIVE-SWIN-RESAMPLE.md). Tests in `vr/tests/native-swin-resample-test.cpp` with runner `vr/test-native-swin-resample.sh`: 5 278 checks in the strict `-O2` build and again at `-O0` (both `-ffp-contract=off`), centred on hand-derived dyadic known answers (the gated-sum single-rounding discriminator 1037 not 1038, the 65504/65520 overflow band and subnormal ties-to-even; the projection k-order discriminator 200704 and the stored double rounding 367.875 -> 384; the replication clamp edge; the downsample resize semantics top-left kept/zero padded/cropped; the `C == 32` f16-residual discriminator 384 versus the `C == 64` 352), bit-exact staged-chain equivalence for every composed entry over deterministic corpora (zero-weight and non-trivial `BlockFixture`/`WideFixture` blocks, all four shift pairs, pad/crop extents, clamped replication extents, poisoned padded map and matrix strides with `-224` gaps), sentinel-checked untouched outputs and full scratch on every validation rejection class of the four entries (the layer weight classes are pre-validated in `swin_layer_upsample` in the block's own order and sentinel-proven to precede the blend stage's first write; the documented blend-overflow scratch exception is pinned explicitly), determinism and zero-allocation accounting. Development fixed a real padded-stride defect in `swin_upsample_blend`'s projection read (the plane offset used `r * x.cols` instead of `r * x.row_stride`) and a corpus sign/comment discrepancy (`rnd_f16` now draws the full 16-bit pattern, sign bit included). Independent read-only review (27/10/2026) initially returned FAIL on the layer weight-rejection ordering (classes rechecked only inside the later block call, after blend scratch writes), a false poisoned-matrix-row-stride claim for the downsample chains and a "every validation rejection class" overclaim — all fixed in this change set (entry-level weight pre-validation with layer weight-rejection sentinel tests, padded `resample.row_stride` poisoning in the downsample chains, and a full rewrite of the negative layer controls with full-scratch sentinels, per-group untouched checks, the exact blend + projected + block scratch boundaries and `swin_layer_plain` value classes); the follow-up review returned PASS with two minor documentation findings, both fixed and reverified here (the modest-gain claim scoped to the layer corpora and an explicit projected-region write assertion in the overflow pin). The sanitiser skip is the same recorded environment gap. Existing CPU regressions pass unchanged. Synthetic correctness only: no vendor correspondence.

Twelfth executable deliverable implemented (27/10/2026, commit `d59df6d`): the residual block core in `vr/native-block.hpp` — LayerNorm, im2col convolution with depthwise/grouped specialisation, PyTorch-exact pixel shuffle/unshuffle, GELU, the ConvNormAct chain and residual split/project/join, pinned to the upstream model/block semantics at the same MIT commit with the full licence notice in [NATIVE-BLOCK.md](vr/NATIVE-BLOCK.md). Tests in `vr/tests/native-block-test.cpp` with runner `vr/test-native-block.sh`: 93 hand-derived checks at `-O2` and `-O0`; the shuffle anchor caught and corrected a wrong channel-interleaving assumption during rebuild (semantics now pinned to `nn.PixelShuffle`). Independent read-only review re-derived the semantics and found no defects; findings fixed.

Thirteenth executable deliverable implemented (27/10/2026, commit `d59df6d`): the typed native network graph plan in `vr/native-graph-plan.hpp` — the layer table (152 layers/71 blocks for ht, 76 layers/31 blocks for small; the pinned small docstring's 77 is weight-file records, not layer count), the extent walk, padding (ht align 32, small align 64), grid/shift geometry and skip wiring, pinned to the upstream `plan.py`/`model.py` at the same MIT commit with the full licence notice in [NATIVE-GRAPH-PLAN.md](vr/NATIVE-GRAPH-PLAN.md). Tests in `vr/tests/native-graph-plan-test.cpp` with runner `vr/test-native-graph-plan.sh`: 20 447 hand-derived checks at `-O2` and `-O0`, zero-allocation accounting. Independent read-only review re-derived the plan semantics from the pinned upstream plan/model source and found no semantic defects; all findings fixed (channels anchors, `operator new` alignment via `std::aligned_alloc`, NDEBUG-safe `CHECK`, the `w = 16384` → 269 484 032 token bound, halvings wording, walk guard).

Fourteenth executable deliverable implemented (27/10/2026, commit `f2c5920`): the bottleneck transformer family in `vr/native-vit.hpp` — the e4m3-grid matmul, the clipped cubic SiLU FFN, the network's piecewise-linear exponential, the butterfly per-head inverse norm, QKV with per-head temperature, 64-key grouped global attention with padding subtraction, the FFN expand/contract and the residual projection, pinned to the upstream `layers/vit.py` at the same MIT commit with the full licence notice in [NATIVE-VIT.md](vr/NATIVE-VIT.md). Tests in `vr/tests/native-vit-test.cpp` with runner `vr/test-native-vit.sh`: 404 hand-derived checks at `-O2` and `-O0`. Independent read-only review re-derived every anchor and the full `vit.py` mapping; findings fixed (runner missing-sanitiser-gap masking, NaN logit undefined behaviour, deterministic ties-to-even rounding, token/channel bounds, three comment derivations, plus a 2-head × 128-token multi-head/multi-group case). The reference's experimental int4 hooks remain out of scope.

Fifteenth executable deliverable implemented (27/10/2026): the split Swin family in `vr/native-split-swin.hpp` — `split_fma_half`/`split_store` (the double-rounded store), `split_ffn_activation` (float16 cubic SiLU), `split_exp` (the mantissa-bit exponential), `split_normalise` (the float16 butterfly norm), `split_softmax_denominator` (the pinned 64-leaf reduction), `split_matmul`, `split_projection_value` and the `SplitFfwd`/`SplitFfwdProj`/`SplitProj`/`SplitProjPool`/`SplitFinalHead`/`SplitQKVAttn` kernels (8×8 shifted-window attention, 16 heads of 32, tile-major token order, zero-token key participation), pinned to the upstream `layers/split_swin.py` at the same MIT commit with the full licence notice in [NATIVE-SPLIT-SWIN.md](vr/NATIVE-SPLIT-SWIN.md). Tests in `vr/tests/native-split-swin-test.cpp` with runner `vr/test-native-split-swin.sh`: 34 267 hand-derived checks at `-O2` and `-O0`. Development fixed two predecessor hand-derived transcription defects (the fma tie literal, the pooled stride-4 layout) and added the tile-order anchor (window token 16), double-shift geometry and multi-channel pooled/sentinel coverage; the independent read-only review re-derived the anchors and the `split_swin.py` mapping and found the pooled padding zero-fill miscount for `c_out > 1` (its HIGH finding) plus two comment/doc inaccuracies — all fixed and reverified. Weight-record decoding (the packed `ffwd_a`/`ffwd_b` scatter maps and the `attn_pos_bias` fragment unpack) is the declared weight-file boundary of this slice. The sanitiser skip is the same recorded environment gap. Synthetic correctness only: no vendor correspondence.

Sixteenth executable deliverable implemented (27/10/2026, continued under supervision `446027da7cb64b579d43c5ac134e5c46`): the decoder input stage in `vr/native-decoder.hpp` — `dec_gated_element` (`f16(f16(skip)·gain)`, the reference's gated skip), `dec_store` (`q8(f16(value))`, the double-rounded network store with ±448 saturation and sign-dropped NaN canonicalisation), `validate_decoder_map` and `dec_input_upsample` (`DecInputUpsample.forward`: the e4m3 projection kept in binary16 over row-major `to_tokens` with float32 key-order accumulation, the nearest-2× replication **cropped** to the skip's extent — a skip larger than the replicated map is the reference's shape error and is rejected, not clamp-replicated — the per-channel gated skip and the q8 store at the skip's extent), pinned to `dlssnr/layers/decoder.py` "DecInputUpsample" with its imported `vit.py` helpers and `quant.py`'s `q8` at the same MIT commit with the full licence notice in [NATIVE-DECODER-INPUT.md](NATIVE-DECODER-INPUT.md) (distinct from the milestone-2 launch-contract record [NATIVE-DECODER.md](NATIVE-DECODER.md); the README's proprietary-route/independent-path separation is preserved). Tests in `vr/tests/native-decoder-test.cpp` with runner `vr/test-native-decoder.sh`: 5 477 checks in the strict `-O2` build and again at `-O0` (both `-ffp-contract=off`), centred on hand-derived literals (the gated pre-rounding discriminator and product tie, the binary16-hop double-rounding discriminator 68.00390625 → 64 versus direct 72 through the kernel, e4m3 store ties, ±448 saturation, the subnormal hop and NaN canonicalisation), the crop/replicate-extent discriminator at odd and exact extents with the scratch mirror checked in full, poisoned padded strides in **both** channels (catching kernel over-reads and validation over-scans), the projection's binary16 rounding discriminators (K7: round-down and round-up sums distinguishing unrounded, truncating and round-up projections), a three-term accumulation group (K8: term completeness and a tiny-term absorption pair pinning float32 key order against both reassociation and reversal — the e4m3 store cannot see the difference, so the scratch pins it), the full rejection matrix with sentinel-checked untouched `out` and `scratch`, determinism and zero-allocation accounting. An independent read-only review re-derived every literal (all correct) and raised four findings — the missing finite-range projection-rounding and c_in ≥ 3 accumulation controls, K4's channel-1 poison coverage and two false K5 crop claims — all fixed; the discriminating controls are mutation-verified (each seeded bug fails exactly at its designed check). The sanitiser skip is the same recorded environment gap. Synthetic correctness only: no vendor correspondence.

Seventeenth executable deliverable implemented (27/10/2026): the network dispatch in `vr/native-network.hpp` — `LayerValue`/`LayerCall` (the pinned tuple semantics and the per-kind input selection: which of the main previous, the skip main, the skip aside, the raw previous pair and the padded frame each layer reads), `layer_produces_pair`/`layer_reads_image`, `network_last_uses` (the pinned `NRNet._last_uses`) and `network_run` (the pinned `NRNet.run` loop over `native-graph-plan.hpp` rows: `_skip_sources` wiring via `plan_skip_source`, the call contracts of all 18 layer kinds, the produced-value contract check and the last-use release including the `SplitFfwd` buffer alias — its `(out, x)` tuple keeps the input's tensor alive past its dict entry), pinned to `dlssnr/model.py` and the five layer families' forwards at the same MIT commit with the full licence notice in [NATIVE-NETWORK.md](NATIVE-NETWORK.md). Tests in `vr/tests/native-network-test.cpp` with runner `vr/test-native-network.sh`: 3 564 checks in the strict `-O2` build and again at `-O0` (both `-ffp-contract=off`), centred on independent re-derivations of `_skip_sources`, `_last_uses`, the del-loop and the tuple lifetimes over the full "ht" (152 rows) and "small" (76 rows) plans (every row's `x`/`aux`/`image` pointer identities, pair forms and release order), mini plans for the edge contracts (the skip span, the element fallbacks, the `SplitFfwd` pair and its alias, capture mode), a rejection matrix covering every upfront class with untouched outputs/scratch and every mid-run class (including self/future skip sources), a buffer-poison control that flags any call touching a released buffer, determinism and zero-allocation accounting. Independent read-only review walked every dispatch arm against the pinned source (routing faithful) and raised five findings, all fixed and reverified (the alias lifetime is now exact and its control mutation-verified, the mid-run comment corrected, skip sources required to be earlier rows, four matrix gaps filled). The layer kernels are not yet bound to real feature-map buffers — that binding and the synthetic input-to-image evaluator are the next slice. The sanitiser skip is the same recorded environment gap. Synthetic correctness only: no vendor correspondence.

Still not established: the counter's network consumer (an open boundary of the proprietary route), the remaining int4 sites (the expansion side-stream, the QKV sites, the SwinSite hard-swish/fold), and any vendor correspondence. The mixed ViT feed-forward miring is bound (NATIVE-INT4.md). The live-input/image evaluation and the temporal history feed are bound (NATIVE-IO.md).

Eighteenth executable deliverable implemented (27/10/2026): the layer binding and synthetic evaluator in `vr/native-eval.hpp` — `eval_layer` (a `LayerFn` for `network_run`) binding all 18 plan-row kinds to the family kernels over the `EvalWeights` record interface (the pinned `Weights` keying; the split feed-forward's packed `ffwd_a`/`ffwd_b` scatter stays the declared weight-file boundary), the per-kind record derivations (`_residual_table` content rule, `scalars_b`/`attn_residual_scale` prefixes, the `attn_pos_bias` deswizzle + exponent ramp; the split bias uses the same deswizzle without the ramp), the wide-gain tail fallback, the planar/token/interleaved-frame transposes, the planar frame boundary, and the arena honouring the dispatch's last-use releases (`eval_reset`/`eval_release`/`eval_arena_used`), pinned to `dlssnr/model.py` and the five layer families' constructors at the same MIT commit with the full licence notice in [NATIVE-EVAL.md](NATIVE-EVAL.md). Tests in `vr/tests/native-eval-test.cpp` with runner `vr/test-native-eval.sh`: 615,354 checks at -O2 (including a bit-for-bit determinism rerun of the whole-network run) and 308,151 at -O0 (the determinism rerun is an optimised-build control), centred on hand-derived answers (the transposes and derivations; `DecInputUpsample` K1; a permuted-identity `VitFfnContract` known answer as an operand-swap control with off-grid store pins), independent oracles (the pre-block `input_lift` decode against a direct pinned decomposition — with a non-symmetric stored record; `VitQKV` distinct q/k/v routing against a direct `vit_qkv` call; the shift-order mapping), record-consumption sets for every heavy kind, the whole-network zero-signal end-to-end over a real `build_plan` (black frame → black frame read through the view's own strides), determinism/zero-allocation accounting, the wildcard record bound, release idempotence and a rejection matrix. Independent read-only review cross-checked every kind's operand mapping against the pinned constructors (Swin body derivations, split/vit/decoder mappings, transposes and arena found correct) and raised five findings, all fixed and reverified (the pre-block record decode, the post-block output strides, a scratch-view stride swap, the vit-attention channel count, status-code nits) together with the six missing controls it identified, now in the suite. Synthetic correctness only: no vendor correspondence.

Next executable stages: temporal history/reset (the second colour copy is the single-frame stub; the pinned `History`/`Reset2`/`Reset1`/`ResetNone` semantics and their weight records), then live-input/image native evaluation, then the int4 mixed-network path as feasible. The host-side frame padding/cropping and the weight-file loading (including the `ffwd` boundary decode) remain the documented host/weight-file boundaries.


## Latest maintenance — 26/10/2026

NVIDIA's DLSS 5 article's “Sign up to be notified” call to action leads to an official “DLSS Notify Me Form” page. This confirms the notification destination but not its form fields or any SDK application/access process; the page was read without submitting it. The official DLSS GitHub releases listing still shows v310.9.16 as latest among displayed releases, described as available to all developers; that release note alone does not identify DLSS-NR content. Sources: [notify page](https://developer.nvidia.com/rtx/dlss/notify-me), [DLSS releases](https://github.com/NVIDIA/DLSS/releases), [v310.9.16](https://github.com/NVIDIA/DLSS/releases/tag/v310.9.16). Read only, no signup or SDK download.

A follow-up read of NVIDIA's 22/09/2026 DLSS 5 developer article verifies that its developer call to action is “Sign up to be notified for DLSS updates for developers here”; it is a notification signup, not documentation of a DLSS-NR SDK request/application route, platform ABI, or distribution entitlement. The same article distinguishes DLSS 5 details from RTX Kit's NTC 0.10 beta update. Thus, this article provides a notification route but no NR package access process. Source: [NVIDIA developer article](https://developer.nvidia.com/blog/whats-new-for-game-developers-dlss-5-with-3d-guided-neural-rendering-nvidia-ace-updates-and-new-rtx-kit-capabilities/). Read only; no signup/contact/download or capture.

Checked NVIDIA's current DLSS product page and public RTX Neural Texture Compression (NTC) SDK repository after NTC appeared in the DLSS 5 developer article. NTC 0.10 beta publicly documents Linux x64 and Vulkan support, but it is a distinct neural texture compression product, not DLSS-NR; it supplies no DLSS-NR ABI, weights, or evidence of DLSS 5 Linux availability. This distinction prevents conflating an unrelated usable SDK with the blocked target. The official DLSS page describes currently available DLSS 4.5 resources and the September 2026 blog describes DLSS 5 inputs/architecture, but neither reviewed page publishes an NR Linux SDK. Sources: [RTX NTC SDK](https://github.com/NVIDIA-RTX/RTXNTC), [NVIDIA DLSS page](https://developer.nvidia.com/rtx/dlss), [DLSS 5 article](https://www.nvidia.com/en-us/geforce/news/dlss-5-3d-guided-neural-rendering/). Read-only checks; no NTC download/build, contact, capture, archive inspection, or GPU probe.

Read two NVIDIA Developer Forum threads where developers ask about authorised DLSS 5 Neural Rendering access: [custom Windows/Blender-Iray request](https://forums.developer.nvidia.com/t/authorized-dlss-5-neural-rendering-sdk-for-a-personal-blender-iray-windows-integration/385118) (05/10/2026; zero replies shown) and [UE 5.8.2 plugin request](https://forums.developer.nvidia.com/t/official-dlss-5-neural-rendering-plugin-access-for-unreal-engine-5-8-2/382731) (08/09/2026; discussion through 16/10/2026, without a visible NVIDIA staff answer or access-process confirmation). These are forum users' requests, not statements of NVIDIA policy, and do not establish availability, eligibility, or Linux support. They reinforce that these pages do not answer the request-route question; restricted or separately distributed materials may still exist. Pages read only; no reply/post, contact, signup or download.

The published NVIDIA DLSS v310.9.16 release asset listing was checked directly. It lists Linux and Windows DLSS demo archives plus source archives, but no separately named NR SDK/header/runtime asset. This is metadata only: the Linux demo archive remains uninspected, and its name/size/hash cannot establish whether it contains NR materials. It neither clears the beta ABI/authority gate nor proves all separately distributed materials unavailable. Source: [expanded v310.9.16 assets](https://github.com/NVIDIA/DLSS/releases/expanded_assets/v310.9.16). No downloads or archive inspection.

A primary-source follow-up read the current `vk_gltf_renderer` releases listing (which shows v2026.11), its v2026.9.1 NR introduction, current user guide, NVIDIA DLSS SDK v310.9.16 release page, and current NVIDIA NGX programming guide. The renderer's NR route remains gated on a beta SDK/header; the general SDK release announces availability to developers but does not state it contains the NR header or offer Linux NR ABI details. The generic NGX programming guide is explicitly Windows/D3D/CUDA oriented and does not document Vulkan NR or a request address for this restricted beta; prior suggestion to email its listed support contact is not grounded there and is withdrawn. The current renderer guide documents feature usage but does not specify how to obtain the NR beta SDK. No official SDK request path for this beta was identified in these pages; this bounded search does not establish that no path exists. NVIDIA Developer Forum threads were also read: a custom Windows/Blender-Iray access request dated 05/10/2026 shows zero replies; a UE 5.8.2 plugin request dated 08/09/2026 has discussion through 16/10/2026 but no visible NVIDIA staff answer or access-process confirmation. These are users' requests, not NVIDIA policy, and do not establish availability, eligibility or Linux support. See direct links above. Sources: [renderer releases](https://github.com/nvpro-samples/vk_gltf_renderer/releases), [v2026.9.1](https://github.com/nvpro-samples/vk_gltf_renderer/releases/tag/v2026.9.1), [renderer guide](https://github.com/nvpro-samples/vk_gltf_renderer/blob/main/docs/user-guide.md), [DLSS SDK v310.9.16 release](https://github.com/NVIDIA/DLSS/releases/tag/v310.9.16), [NGX programming guide](https://docs.nvidia.com/ngx/programming-guide/index.html). Page reads only; no request/contact sent, downloads, source checkout, build, capture or GPU probe.

A focused check of NVIDIA's 22/09/2026 DLSS 5 developer blog found it describes DLSS 5's one-frame-in/one-frame-out architecture and game-engine frame/motion-vector inputs, but announces no Linux/Vulkan SDK, beta access or native NR ABI. Its beta SDK mention is RTX Neural Texture Compression, a distinct feature, not DLSS-NR. This is useful distinction for evaluating public leads, but does not establish any substitute target ABI or clear the existing integration gate. Source: [NVIDIA developer blog](https://developer.nvidia.com/blog/whats-new-for-game-developers-dlss-5-with-3d-guided-neural-rendering-nvidia-ace-updates-and-new-rtx-kit-capabilities). Page read only; no SDK, source or artifact downloaded, no build/capture/GPU probe.

A source/doc check of NVIDIA `vk_gltf_renderer` refreshed the external dependency assessment. Its v2026.9.1 release notes describe DLSS-NR as limited-availability beta requiring an NGX SDK with `nvsdk_ngx_helpers_dlssnr_vk.h`; the ordinary public 310.7.0 SDK omits that header and auto-disables the feature. The current user guide describes the NR feature pipeline but does not document the beta package's Linux availability or provide its contents. The general NVIDIA DLSS Linux support announcement concerns DLSS 2.2.1 and does not establish NR availability. v2026.11 release notes checked 23/10/2026 do not announce an NR SDK change. These public docs do not clear the version-matched ABI or authorisation gate and do not prove restricted/separate materials unavailable. Sources: [v2026.9.1 release](https://github.com/nvpro-samples/vk_gltf_renderer/releases/tag/v2026.9.1), [v2026.11 release](https://github.com/nvpro-samples/vk_gltf_renderer/releases/tag/v2026.11), [user guide](https://github.com/nvpro-samples/vk_gltf_renderer/blob/main/docs/user-guide.md), and [NVIDIA DLSS 2.2.1 Linux announcement](https://developer.nvidia.com/blog/?p=34582). Searches/page reads only; no source checkout/download, build, capture, GPU probe or proprietary material was used. This reinforces the existing gate assessment, not a migration milestone.

Commit `78ab3e4` (`Reject embedded address-like metadata IDs`) tightened the metadata-only `native-resource-events.py` validator: ID validation now rejects embedded pointer-shaped hexadecimal tokens as well as whole-value forms, with negative controls for prefixed and bare-hex tokens and a positive boundary case (`p-0xabcdefg`). This prevents raw-address-shaped strings from being accepted as process-local metadata IDs; it does not establish runtime identity or GPU dataflow. CPU verification passed for the resource-event self-test (1 known answer, 21 corruption controls, parser/duplicate/CLI controls), slot replay (834 known answers, 40 corruption controls; zero GPU kernels launched), ladder-report (19 controls), ladder-names (14 controls), and native input layout (9 invalid-layout controls). Independent review found and prompted a regex boundary fix; the corrected patch then passed review. `git diff --check` passed. No capture, archive inspection, or GPU probe was performed. The external ABI/authority/producer-to-consumer and decoder-bound gates remain unresolved; overall native evaluation is incomplete.

Updated 26/10/2026 (Australia/Sydney). Active repository `/home/michael/Projects/dlss5-linux`, remote `git@github.com:Lamothe/dlss5-linux.git`, branch `main`. The foundational handover body is from `6786a40`; an earlier 26/10/2026 checkpoint began at `f4eca7a` (`main`, `origin/main` and `origin/HEAD` clean/aligned). The latest maintenance review began from clean/aligned `0e4c8c0`; its bounded findings and publication are recorded below. Subsequent bounded public-lead assessments (OpenDLSS-NR, DLSSNR-AMD, DLSS5VKLayer and dlssnr-pytorch) do not resolve missing lawful weight provenance, producer→launch identity, target ABI, or native temporal/live-image evaluation. A public-source-only review confirms that the NVIDIA v310.9.16 Linux demo asset's public metadata does not reveal contents; the archive remains uninspected and no NR ABI can be inferred from the metadata. Public NVIDIA Vulkan headers/libraries document general DLSS, while the NR Vulkan header is identified in NVIDIA's sample docs as limited-availability beta. In the project evidence reviewed, no authorised target materials, explicit capture authority, or new producer→consumer evidence were supplied. No archive contents or target implementation source were inspected; no capture or GPU work was performed. The earlier evidence audit was performed at `1f03478`; later maintenance and official-source checks are recorded in the sections below. This handover replaces older pointers; historical evidence remains in the linked reports.

## Goal and boundaries

Build and validate a native Linux evaluator from live inputs through preprocessing, learned-weight bindings, full execution graph, temporal history/reset and output conversion, producing neural-rendered images without Windows host code, DirectX or Proton in the evaluator. Proton reference capture is allowed only when authorised. Kernel replay alone is intermediate evidence. Do not modify Genesis or build another VR host. Keep proprietary payloads, weights, captures, binaries and assets out of Git; preserve containment/rights qualifications. Use only this workspace and remote.

## Current verified migration state

- Replay campaign: 68 contracts, slots 24–91, out of archival 156-launch inventory; not overall completion. Slot 91 remains latest qualified replay; no new contract absent paired captures, attribution and controls.
- Earlier checkout/evidence-gate checkpoint (26/10/2026): `main` began clean/aligned at `f4eca7a`; latest maintenance review began at `0e4c8c0` and is recorded in the maintenance section below. No newly supplied version-matched NR target material, capture authority, or producer→consumer identity evidence is documented. Existing local graph reports are static-only: the newest available report, `graph-inspect-42qiEUZK` (25/10/2026), has `static_contract_verified=true`, `full_graph_recovered=false`, `dynamic_bindings_verified=false`, `native_image_inference=false`, and zero kernels launched. Official-source review of NVIDIA's 22/09/2026 DLSS 5 developer article finds no SDK availability statement; this is not evidence that restricted or separately distributed SDK materials do not exist. Public beta-renderer documentation requires a limited-availability NR SDK header, without establishing Linux availability. Existing ignored decoder address/IO probe reports are historical bounded audits, not authority to launch the separately gated GPU experiment. No runtime probe, archive inspection, capture or GPU work was performed in this continuation.
- Bounded native GPU results include one stored learned-weight record's single-product decoder checks, byte-exact replay of captured decoder-boundary outputs for tested variants/frames, and a constructed projection→bridge→repack chain. They are not the proprietary graph or live-input neural image output. See `NATIVE-DECODER-BINDING.md`, `NATIVE-VENDOR-COMPARE.md`, `NATIVE-CHAIN.md`.
- Source-only raw-input identity investigation is exhausted against current authorised material. Available traces do not join raw producer resources to consuming launch. Do not repeat schemas, fixtures, hash scaffolds or decoder comparisons. That path requires explicit authorisation for version-matched capture and target ABI/materials, or genuinely new local producer→consumer evidence (`NATIVE-GRAPH-NEXT.md`, `NATIVE-HOOK-PLAN.md`).
- Existing NGX native probe checks capability and feature creation only; it performs no feature evaluation, inference or image output. No new capture or GPU probe is authorised. Accumulation/FP16-tail experiment requires explicit operator approval, current safety preflight and complete output-store/CFG/address/launch bounds first.
- The overall native live-input→image evaluator, full semantic layouts/weights, temporal behaviour and reset remain unestablished. VR/Genesis integration is later and untouched.

## Current external dependency gate

A current status note in `NATIVE-MIGRATION-GATE.md` records why the producer→consumer path cannot safely advance from available materials: there is no concrete input-resource-to-launch identity edge, no supplied version-matched NR ABI/integration material plus capture authorisation, and the separate decoder experiment still lacks store/CFG/address bounds and approval. Official Streamline 2.14.0 notes identify `sl.dlss_nr`, but distinguish Vulkan Reflex support from NR; the current programming guide specifies a Windows requirement and supplies no evidence of a native Linux/Vulkan NR ABI. No GPU/capture work was run for this source check. The overall migration remains incomplete. Reassess only if authorised, version-matched target materials arrive or genuinely new local producer→consumer evidence becomes available; do not repeat schema/static-anchor work or invent bindings.

## Static graph and decoder evidence

`NATIVE-GRAPH.md` reports conditional outer source indices `[38,30]` and selectors `[0,1]` feeding an upsampler candidate. Its append schedule `[9,6,8,8,8]` yields conditional candidate slots 23–30 (split-Swin) and 31–38 (primary transformer), not global runtime descriptor IDs. The checker explicitly leaves global numbering, actual tensor indices and CUDA addresses unverified. Do not resume broad global-numbering/extra append-anchor work as if that limitation were merely missing tests.

A fresh authorised-source, no-GPU graph inspect ran this session through `vr/run-native-graph-probe.sh inspect`, evidence `research/native-linux/runs/graph-inspect-42qiEUZK` (local, do not commit). It succeeded with `static_contract_verified=true`, conditional indices 38/30, `global_descriptor_numbering_verified=false`, `actual_tensor_indices_verified=false`, `cuda_addresses_verified=false`, `full_graph_recovered=false`, and zero kernels launched. This reproduces existing results; it is not new graph progress. The graph CPU self-test passed (1 parser and 1 configuration known answer; 2 and 3 negative controls rejected).

`NATIVE-DECODER.md` and `native-decoder-probe.py` already recover the bounded host decoder-argument/variant/backend contract, allocation planner consumer and part of binder pointer construction. The consumer reads `state+0x238` as a locator/reference, resolves through subsequent table lookups, then reads signed 32-bit feature indices and scales those by 16 into planner entries. The decoder probe's current anchored source ranges include the binder tail at `0x18003ec76..0x18003ed06` and the immediate callee `0x18002c1e0`. That path finishes the feature-pointer loop and publishes a temporary pointer vector into `group+0x140`; checked instructions do not write the binder-read `state+0x238` field. Consequently, the previously proposed *writer search* is not presently justified as a new task: that field's producer lies outside the proven consumer/caller chain and identifying an upstream writer would require a separate, bounded source-authority decision rather than assuming a direct caller. The locator tables' ownership and record→group→layer association remain explicitly unverified. The decoder contract's 938 negative instruction-anchor controls already include arena and reset anchors; do not repeat them.

### Next concrete gate

No unblocked static writer or decoder experiment remains identified: the `state+0x238` producer is outside the checked consumer/binder chain and needs a separate scope/authority decision; activation addressing and all-store address bounds remain unresolved; the decoder GPU experiment is not authorised. Do not start synthetic scaffolding, repeat the binder anchors, or launch capture/GPU work. Reassess the producer→consumer path if either genuinely new local evidence becomes available within existing authority and safety boundaries, or explicit capture authorisation plus version-matched NR target ABI/integration materials are supplied to identify an allowed bounded trace. Do not infer bindings from the current evidence. The separate stored-weight multi-term/FP16-tail experiment also requires explicit operator approval, current preflight and the complete store-envelope gate; it cannot establish the full evaluator alone.

The repository does contain local, ignored probe runs demonstrating the SM89 upsampling kernel's structural influence mapping and single-product use of one stored record (`NATIVE-DECODER-EXTENSION.md`, `NATIVE-DECODER-BINDING.md`). Treat those as bounded existing evidence, not general layer or native image validation. The next stage written in those reports is Proton reference evaluation, which requires explicit capture authorisation and still cannot close the raw-input/interop gap. Likewise the temporal recurrence/reset contract is Proton-side evidence only (`TEMPORAL-HISTORY.md`, `TEMPORAL-STATE.md`), not native temporal execution. No further CPU-only reformulation of those findings would advance the missing pipeline edges.

## Latest maintenance update — 26/10/2026

This notification-page follow-up was performed at `3346908` and published at `a8617e1`; the current checkout is clean and aligned there. I reread the official NVIDIA DLSS product page, which lists DLSS 4.5 resources and a notification link but no DLSS-NR integration package or Linux ABI. This confirms the bounded public-page observation, not the absence of separately distributed materials. No downloads, signup, capture or GPU probe were made.

The prior local-evidence maintenance check was performed at `a10318e`, the official Streamline source review was recorded in `a30bee1`, and its wording/chronology updates followed in `a6f8576` and `26dd922`. The preceding documentation update began from clean, aligned `main` at `26dd922`; the subsequent local package inventory began at `5a2f587` and was published as `bf4fdbf`; the subsequent scoped-status wording fix was published at `5e3614e`, with handover provenance corrected in `6786a40`; this maintenance check began from clean, aligned `main` at `6786a40`. I rechecked the official Streamline changelog, current programming guide, public include listing, and [Streamline Linux-support issue #105](https://github.com/NVIDIA-RTX/Streamline/issues/105). The issue response says “Currently Proton is supported” and avoids future plans; that is not native Linux support. Public docs list `sl.dlss_nr`, but no NR-specific header was present in the inspected include listing, and the Windows-oriented guide does not establish a native Linux/Vulkan NR ABI. These bounded public-source observations do not rule out separate or restricted target materials. Official GitHub release metadata for Streamline 2.14.16 (published 16/10/2026 UTC, 17/10/2026 Australia/Sydney) lists `streamline-sdk-v2.14.16.zip`, `streamline-sdk-v2.14.16-aarch64.zip`, and `streamline-sdk-v2.14.16-arm64ec.zip`; it lists no Linux-named release asset. The release body says the SDK is available to all developers and describes bug fixes/stability improvements; neither statement establishes Linux NR support. This is a check of public asset names/metadata, not package contents, and does not rule out separately distributed/restricted Linux material. I did not download or inspect those assets. I also checked the locally staged SDK/demo materials: this workspace has the 310.8 NR archive and Windows demo archive, but no downloaded Linux demo archive or Streamline NR header; I did not inspect archive contents. Separately, the official NVIDIA/DLSS `v310.9.16` release API lists `ngx_dlss_demo_linux.zip` (96,907,020 bytes; SHA-256 `985019f1307c0367d0d4dff35f06b2615c5c3ad6b469ffec5b0c16e28758610d`), while its notes mention only bug fixes/stability. This metadata establishes publication, not archive contents; it was not downloaded or inspected. No capture or GPU work was run. The local evidence inventory and existing decoder IO/store-envelope notes still expose no new producer→consumer identity evidence or proven reachable output-address bound. No proprietary evidence changed in this pass. I also checked the official `vk_gltf_renderer` v2026.11 release (published 23/10/2026): its release notes cover OIDN, glTF 2.1, path-tracer and other fixes, and do not mention DLSS-NR. The v2026.9.1 notes describe DLSS-NR as limited availability and require `nvsdk_ngx_helpers_dlssnr_vk.h`, absent from the ordinary public SDK; these public notes do not establish whether a version-matched Linux beta header/material is available. This was a metadata/documentation check only: no SDK assets or archives were downloaded or inspected, and it does not clear the project’s authority, ABI, or producer→launch evidence gates. This bounded release-metadata update and its date/filename corrections were published in `0e4bf45` (`Record Streamline Linux asset metadata check`). `git diff --check` and the graph, resource-event, and slot-replay CPU self-tests passed; slot replay reported 834 known answers and 40 corruption controls, with zero GPU kernels launched. Independent documentation review passed after correcting timezone labelling, full asset names, release-body wording, and this section's maintenance date. The checkout after publication was clean and aligned with `origin/main`. Follow-up review clarified that the ARM-labelled Streamline asset names alone do not establish package platform/content, separated the 25/10 source review from the 26/10 release-metadata check in the hook plan, and pinned the third-party Linux-bridge citations to commit `89e85fb7ac8c36d817f77eb671b1e64bd09890c1` (timestamp 01/09/2026 UTC). The corrected doc changes passed `git diff --check` and independent review. Final repository state after that publication is commit `0782544` (`Pin public release evidence and source dates`), clean and aligned with `origin/main`. I made a bounded search-result check for newer Streamline 2.14.16 Linux/NR announcements; it returned only the already recorded release/changelog facts and no new official Linux NR availability announcement. This search did not inspect SDK contents or alter the target-material/authorisation/producer-identity blockers.

Earlier provenance: the audit state from `1f03478` was clarified in `ffd808c`; local evidence/artifact checks were recorded in `a5a52f6` and `37c251a`; status handover refreshed in `a10318e`; official Streamline-source interpretation was documented in `a30bee1`; chronology/wording were corrected in `9d75fbc` and `a6f8576`; the Proton-support distinction was recorded in `26dd922`; the local staged-package inventory was recorded in `bf4fdbf`; a gate wording correction followed in `5e3614e`, and handover chronology was corrected in `6786a40`. The current update records official release-asset metadata only, not archive contents. These are documentation checkpoints, not native inference progress. The final maintenance-only evidence review began from clean/aligned commit `0e4c8c0` on 26/10/2026. It found no newly available authorised target materials or producer→consumer evidence; the decoder output-address safety bound remains unresolved, so the gated GPU experiment was not run. The gate and milestone current-state pointers were refreshed only; no project probe was run, and no proprietary archive, binary or capture was inspected. `git diff --check` and independent review passed. The native migration remains incomplete. The latest status-pointer publication is `bb1cac4` (`Refresh native migration status pointers`), clean and aligned with `origin/main`. At a follow-up maintenance pass starting from `0174b65` on 26/10/2026, I reran only the documented CPU self-tests: metadata events (1 known answer, 19 corruption controls), slot replay (834 known answers, 40 corruption controls; zero GPU kernels), ladder report (19 controls), ladder names (14 controls), and input-layout tests (9 invalid-layout controls). All passed. The worktree was clean; no proprietary material, capture or GPU work was used. This verifies the documented safe regression suite but supplies no new native execution evidence.

## Prior checks and verification

At the gate documentation update, the slot-replay self-test passed 834 known answers and rejected 40 corruption controls, with zero kernels launched. Graph self-test passed. Decoder CPU self-test passed: 9 sizing answers, 616 arithmetic comparisons, 9 invalid-domain and 3 incorrect-execution rejections, arena arithmetic controls, 4 grid answers and malformed metadata controls. The `ac58577` provenance update passed `git diff --check`; no new source edits, captures or GPU probes were made.

### Follow-up worker check — 26/10/2026

I re-read `README.md` and this handover, then confirmed the active `main` checkout was clean and aligned with `origin/main` at `a060431` (`Record safe native migration regression results`). The gate still records no authorised version-matched NR target ABI/capture scope or new producer→launch identity evidence; the decoder-store bounds and approval are also unresolved. A bounded web search found no newer official Linux NR SDK announcement; this is not proof that restricted materials do not exist. I ran the CPU-only slot replay self-test (834 known answers, 40 corruption controls, zero GPU kernels), ladder report (19 controls), ladder names (14 controls), native input-layout (9 invalid-layout controls), and resource-event schema self-test (1 known answer, 19 corruption controls); all passed. These checks validate tooling only and do not advance the missing runtime evidence edges. No archive, binary, capture, GPU probe or proprietary payload was inspected or run; no implementation change was justified, and the native migration remains incomplete. No commit was made in this check.

## Safe CPU regression refresh (26/10/2026)

After commit `9817c85`, I re-read the current gate and confirmed `main` was clean and aligned with `origin/main`; no newer local run manifests or supplied gate-clearing materials were present. The documented blocked-path decision remains: no synthetic binding scaffolding, proprietary archive inspection, capture, or GPU/decoder experiment without the required authority, version-matched target ABI and safety bounds. I ran the documented CPU-only checks: slot-replay self-test (834 known answers, 40 corruption controls, zero GPU kernels), ladder-report (19 controls), ladder-names (14 controls), and native input-layout (9 invalid-layout controls). I also ran `surface-filter-test.py`, which passed its tile/mip, colour/normal averaging, gutter/high-frequency, periodic-boundary, rotated-footprint, original2048 source-filtering, and continuous-boundary-derivative checks. These are tooling regressions only and add no producer→launch, native inference, temporal execution, or image-output evidence. No archives, binaries, captures, or proprietary payloads were inspected or run. The checkout was not modified by these tests; native migration remains incomplete.

## References

`NATIVE-SLOT-REPLAY.md`, `NATIVE-GRAPH.md`, `NATIVE-GRAPH-NEXT.md`, `NATIVE-HOOK-PLAN.md`, `NATIVE-CHAIN.md`, `NATIVE-DECODER.md`, `NATIVE-DECODER-BINDING.md`, `NATIVE-VENDOR-COMPARE.md`, `NATIVE-DECODER-STORE-ENVELOPE.md`, `NATIVE-DECODER-IO.md`, `TEMPORAL-HISTORY.md`. Recheck branch, worktree and origin before future edits/pushes.

## New public lead assessed — OpenDLSS-NR (25/10/2026)

A public lead appeared in `maanHimself/OpenDLSS-NR` (MIT-licensed repository, public metadata checked 25/10/2026; upstream README and demo docs read). It describes a substantial 310.8.0 DLSS-NR network reimplementation, a separate WebGPU port, a live Filament temporal demo, and author-claimed byte-exact block-boundary parity. This is useful design/reference material, not evidence for the proprietary NVIDIA runtime and not an established Linux backend. The upstream demo docs explicitly say Windows-only; its kernels rely on NVIDIA-specific Vulkan extensions (`VK_NV_cuda_kernel_launch` or cooperative matrix + float8). The CLI is single-frame/no-history; history/reprojection/reset is in the Windows Filament demo. Required model weights are not supplied, upstream says nothing in the repository produces the model directory, and the authors disclaim NVIDIA affiliation/IP grants and place model-data rights responsibility on users. The WebGPU port is browser-based and still requires the same model directory. No upstream code, weights, archive, or fixtures were downloaded or imported; no build or GPU work was run.

This does not clear this project's runtime/weight provenance, Linux execution, or input-capture gates and does not establish that upstream's parity claims have been independently verified here. It may inform future network and temporal design work only if lawful compatible weights are separately available and an authorised Linux backend is demonstrated. No change to L1/L2/L3 status: the native live-input→image evaluator and temporal/reset acceptance remain incomplete. References: https://github.com/maanHimself/OpenDLSS-NR and its `demo/README.md`, `docs/frame.md`.

## New public Linux Vulkan implementation lead — DLSSNR-AMD (25/10/2026)

Public documentation for `mochizuki0323/DLSSNR-AMD` describes Vulkan-compute GLSL reimplementations of the 310.8.0 and 310.9.16 NR models, targeting RDNA4 cooperative-matrix/FP8 extensions. The author reports comparative moving-frame and single-frame results, but these are upstream claims on one RX 9070 XT, not tests in this repository; the README describes the project as early and Linux testing is via Steam/Proton. Its game paths are not evidence of a standalone native Linux evaluator: architecture docs describe an NGX core as a Windows DLL under Proton, and game integrations using Windows ReShade/OptiScaler components. The README now advertises v0.0.7 (19/10/2026), the 310.9.16 model by default and temporal moving-sequence comparisons, but also limits testing to one RX 9070 XT/Linux desktop and warns of crashes/artifacts. This does not establish Linux-native game input support or remove the Windows-process dependency. Source-level isolation/build validation would be necessary before any reuse claim.

This is a potentially useful shader/network and temporal-pipeline reference, but does not clear the project's standalone-host, lawful weight provenance, hardware, or capture/input identity gates. The model is not included; the upstream installer extracts it from a user-provided NVIDIA DLL. Upstream's software licence is not a grant to NVIDIA model data, and does not establish our authority to extract/use it. No upstream source, DLL, weights, fixtures, or binaries were downloaded/imported; no build, capture, or GPU test was run. Do not port or depend on its code until its licence/NOTICE and target components are separately reviewed and a Linux-only host boundary plus lawful model path is demonstrated. References: https://github.com/mochizuki0323/DLSSNR-AMD and `docs/ARCHITECTURE.md`.

## Native-game Vulkan layer lead checked — DLSS5VKLayer (25/10/2026)

The public `bmitch87/DLSS5VKLayer` README was checked after it surfaced as a possible native-Vulkan route. It explicitly describes a Linux Vulkan layer that forwards presented frames to a **Windows NGX neural-rendering helper running under Wine/Proton**. Thus “native Linux games” describes the intercepted game's rendering API/process, not the evaluator: NR inference still crosses to a Windows helper. The README requires an NVIDIA GPU/driver and user-imported `nvngx_dlssnr.dll`; NVIDIA NGX binaries are not in the public package, and personal-package redistribution is conditioned on rights. It also labels the project experimental/research. This cannot satisfy our strict evaluator constraint (no Windows host code/Proton), and it supplies no new authorised weights or producer→launch evidence. No code, binaries, or model files were fetched or run. Reference: https://github.com/bmitch87/DLSS5VKLayer.

The associated Linux Swapper fork is an installer/manager whose README offers DLSS5VKLayer for native Vulkan games and separately lists Steam with Proton as a requirement; the Windows-helper dependency described above is stated by DLSS5VKLayer's README, not independently established by the Swapper README. The fork is not evidence of a separate native inference implementation. No package or source was downloaded. Reference: https://github.com/Nicholasvoador/DLSS5-Swapper-Linux-Support.

## Standalone single-frame inference lead — dlssnr-pytorch (25/10/2026)

The public `mauri870/dlssnr-pytorch` README describes a Python/PyTorch implementation that runs the NR network on one supplied colour frame, with CPU/CUDA/ROCm device selection and reported single-frame comparisons to NVIDIA; its documented setup suggests a Linux-capable workflow but does not expressly establish Linux support/testing. The README explicitly labels its reference comparisons single-frame/no-history. It describes extraction of model data from a user-supplied `nvngx_dlssnr.dll`; that DLL is a technical input, not a weight licence or evidence of authority to extract/use those weights. README says the CUDA build has not been tried on NVIDIA hardware; reported GPU figures are ROCm/RX 7900 XTX. This is useful network/evaluation reference, but not a temporal implementation, live-input acquisition path, or end-to-end evaluator. Do not infer temporal/reset parity or NVIDIA-Linux performance from it. The README was reviewed; no implementation code, DLL, weights, frame fixtures, or binaries were downloaded or run. Reference: https://github.com/mauri870/dlssnr-pytorch.

## Safe regression rerun (26/10/2026)

From clean commit `3b0b489`, I reran the bounded CPU checks: resource-event schema (1 known answer, 19 corruption controls), slot replay (834 known answers, 40 corruption controls; zero GPU kernels), ladder report (19 controls), ladder names (14 controls), input-layout (9 invalid-layout controls), and surface filtering (nine tile-local levels; linear-light colour and normal averaging; gutters/high-frequency rejection; periodic boundaries across mip levels and rotated footprints; original2048 filtering and continuous boundary derivatives). One parallel invocation containing the ladder and input-layout checks timed out; rerunning those checks synchronously completed and all passed. These are tooling regressions only: they establish no new producer→launch edge, native inference, temporal execution, or live image output. No proprietary materials, archives, captures or GPU work were used; native migration remains incomplete.

## Current worker check (26/10/2026)

Revalidated the checkout at clean `75c8b3a` (`main`, `origin/main`, and `origin/HEAD` aligned; origin is the `dlss5-linux` repository). The current handover, migration gate and milestone still identify the same evidence blocker: no newly supplied version-matched NR target ABI/integration material, explicit capture authority, or raw-producer-to-launch identity evidence; the decoder experiment remains gated on a complete store/CFG/address envelope, approval and safety preflight. Bounded metadata-only inventories show no newer ignored run than the existing static graph/decoder inspection and no later slot replay or reference capture than slot 91. I did not inspect proprietary payloads or capture content, download archives, or run capture/GPU probes.

Repeated the CPU checks synchronously: resource-event schema (1 known answer/19 corruption controls), slot replay (834 known answers/40 corruption controls; zero GPU kernels), ladder report (19), ladder names (14), input-layout (9 invalid-layout controls), and `vr/tests/surface-filter-test.py` (all three test groups). The first attempt to run the surface-filter test used a nonexistent path (`vr/tests/native-surface-filter-test.py`) and exited 2; locating the actual test and rerunning it passed. This check adds regression confidence only, no new migration evidence. The checkout remains incomplete at L1; L2/L3 are not demonstrated. No code path to the native evaluator is unblocked by these checks, so work pauses at the external evidence/authorisation gate rather than inventing bindings or repeating exhausted investigations.

## Follow-up worker check (26/10/2026)

Re-read the current gate, graph-next and decoder-store-envelope notes from clean/aligned `245d992`. A fresh bounded public-source search found no evidence of a generally available native Linux/Vulkan DLSS-NR SDK/ABI; the result is consistent with the existing qualified gate, and does not rule out separately distributed or restricted materials. No new version-matched NR ABI/integration materials, producer→consumer identity evidence or decoder launch-bound facts were supplied. A read-only inspection of the decoder IO audit confirmed its own stated limits: the reaching-definition classifier and CFG are bounded structural analyses, not a full SASS value/path proof; no safe output envelope or address range follows. The independent planning review likewise found no useful CPU-only implementation change at the present evidence boundary; adding further schemas, vectors or static anchors would not advance the missing proof. No proprietary files/capture content were inspected, no archive was downloaded, and no capture/GPU probe was run. There is no evidence-backed implementation step to take now. Native L1 remains in progress; L2/L3, temporal/reset evaluation and live-input image output remain unproven.

Nineteenth executable deliverable implemented (27/10/2026): the temporal history carrier in `vr/native-history.hpp` — the per-feature FP16 colour-history map, the frames-since-reset counter and the reset contract in the behavioural form established by this repository's capture evidence (`TEMPORAL-HISTORY.md`, gate stage 13): `history_N = output_{N-1}` bit-exactly (finite binary16-grid values stored unchanged; anything else rejected, never rounded), `counter_N` = frames since the last reset (the pinned trace sequence `[0,1,2,3,0,1,2,3]` across a mid-stream reset), and a reset behaving as if history = 0 and counter = 0 while ignoring staged history content (the `injectreset` control). The carrier stores zero at init/reset (the capture's `history0-init.bin`), the commit reads strided sources through their own row stride (padding never read) with `memmove` rows (a self-commit is legal), and the counter refuses to advance past UINT32_MAX rather than wrapping into the reset-frame value. Tests in `vr/tests/native-history-test.cpp` with runner `vr/test-native-history.sh`: 281 checks at -O2 and -O0 — the pinned counter sequence, bit-exactness at the grid edges compared with memcmp (sign of zero included), the stride-poison control, the reset-ignores-staged-history control, the rejection matrix (including a shrunk capacity and counter exhaustion) with sentinel-checked untouched state, determinism and zero-allocation accounting. Independent read-only review verified the contract ordering against the capture evidence and raised seven findings (counter wrap, an overstated untouched-on-error claim for init, missing upper bounds, an incomplete allocation override, bit-exactness under-pinned by `==`, three untested rejection classes, memcpy UB on a legal self-commit), all fixed and reverified. Naming correction recorded in [NATIVE-HISTORY.md](NATIVE-HISTORY.md): the earlier handover's "pinned History/Reset2/Reset1/ResetNone stages and their weight records" do not exist in the pinned reference (single-frame by construction; `Controls.history`/`reset` are accepted-and-ignored) — that framing was a handover error and is withdrawn; the capture-evidence contract is what was implemented. Synthetic correctness only: no vendor correspondence.

Twentieth executable deliverable implemented (27/10/2026): the host-image boundary and image evaluation in `vr/native-io.hpp` — `frame_from_u8`/`frame_to_u8` (the pinned `io.py` `levels` table, `_mirror` edge padding and the literal `tensor_to_frame` rounding rule, including the `fl(0.5/255)` midpoint where `floor(scaled + 0.5f)` would diverge from the pin), the chain's axis convention (rows = height; `out_w` is the tensor row count) and `image_evaluation` (host frame in/out over `network_run` with the temporal carrier's history colour fed to the input lift's slots 7..9 via the new `input_lift(frame, history, ...)` primary; the old entry delegates with the current frame — the documented single-frame stub, behaviour-preserving per the unchanged 5,316-check lift suite). Tests in `vr/tests/native-io-test.cpp` with runner `vr/test-native-io.sh`: 2,045 checks at -O2/-O0 — hand-derived mirror tables, a non-square map case, the exact integer-derived rounding table with the tie and the divergent midpoint, the 256-code round-trip, clamp/NaN edges, the lift history wiring, a stub-layer `image_evaluation` seam pinning the work extent/crop/orientation/rounding together, the precondition rejections and zero-allocation accounting. Independent read-only review cross-checked the boundary against the pinned `io.py` and the plan's axis convention and raised eight findings, all fixed and reverified — including a fatal transposed-axis defect (`image_evaluation` rejected every landscape frame before the fix) and the rounding divergence. Disclosed gap: the non-null history path through `eval_pre_block` is exercised at the lift level only (the eval suite covers the stub path). The eval suite (615,354/308,151) and the input-lift suite (5,316) pass unchanged after the slice. Synthetic correctness only: no vendor correspondence.

Twenty-first executable deliverable implemented (28/10/2026): the int4 mixed-network operand path in `vr/native-int4.hpp` — the unnormalised Walsh-Hadamard quantiser (`clamp(rint(H(x) * gain + zero), 0, 15) - 8` with the 1.5 * 2^23 integer trick, the binary16 half form and the clamp-before-minus-8), the exact-integer accumulator with its float64 affine (the binary16 hop observable), the mixed int4 + e4m3-tail ViT contraction (deterministic left-to-right float32 sum), and the Swin site's hard-swish and fold-constant chain (NR_HS_FOLD/FOLD32/FOLDPP with the NR_HS_FOLD_BLOCKS gating), pinned to the MIT reference `dlssnr/int4.py` (itself following DLSSNR-AMD's i4_obtain/shaders, MIT mochizuki0323) with both licence notices in [NATIVE-INT4.md](NATIVE-INT4.md). The `.fix`/`.mu`/`.aq` recovery tables and their file decode remain the declared host weight-file boundary (like `ffwd`); no table data in-tree. Tests in `vr/tests/native-int4-test.cpp` with runner `vr/test-native-int4.sh`: 83 checks at -O2/-O0 — hand-derived butterflies, the ties-to-even table, the three gain/zero forms, the clamp code form with NaN/inf, the exact dot + affine, the hand-derived binary16 hop (-4.94976806640625), the permuted mixed contraction (6.5), the hard-swish clamps, the fold chain incl. gating-beats-first/last and empty/null forms, the rejection matrix and zero-allocation accounting. Independent read-only review traced the butterfly against the pinned verbatim and raised ten finding groups (the exactness bound now enforced as the 2^14 column bound; the magic 2^22 range; the hard-swish exactness claim; the UB non-finite cast now clamped on the double with mode-independent half-even rounding; the fold null/empty semantics; sibling-style bounds; the q-range enforcement that would reject the pinned's escaped int8 records — dropped; a tautological half test — now a hand literal; stale comments and the decode-side CLIPS table; the reference copyright line), all fixed and reverified. Synthetic correctness only: no vendor correspondence.

Twenty-second executable deliverable implemented (28/10/2026): the int4 network miring in `vr/native-eval.hpp` + `vr/native-int4.hpp`'s `Int4Site` record — the mixed ViT feed-forward now runs over the record boundary exactly as the pinned `VitFfn.contraction` (the int4 channels from the unrounded activation through the quantiser, the e4m3 tail from the rounded copy through the solver-table decode with the NaN codes reading 0, the pinned's single float32 affine+product add, and the residual arithmetic byte-identical to `vit_residual_linear`), gated on the "vit_ffn" site record's presence as the pinned's table-presence gating (`kVitProjection` never takes the site). Tests in `vr/tests/native-eval-int4-test.cpp` with runner `vr/test-native-eval-int4.sh`: 36 checks at -O2/-O0 — the hand-derived two-token mixed answers ([1,-22,3,-32]), the gain hop ([3,-20,5,-30]), the tail NaN decode, the gating discriminators (null site and projection), the site-validation rejections with sentinel outputs, determinism and zero-allocation accounting. Independent review re-derived the answers independently and confirmed the residual equivalence line-by-line; its seven findings (the tail NaN decode, the association, the perm-check comment, the missing alloc checks, a wrong rejection case, the untested projection gate, docs) all fixed and reverified. The eval suite (615,354/308,151) passes unchanged. Remaining int4 sites: the expansion side-stream (an open device-cache boundary), the kSplitQKVAttn attention site and the SwinSite hard-swish/fold. The ViT QKV site is bound (NATIVE-INT4.md). Synthetic correctness only: no vendor correspondence.

Twenty-third executable deliverable implemented (28/10/2026): the ViT QKV int4 site in `vr/native-eval.hpp` (`eval_vit_qkv_int4`, the "vit_qkv" gate before the e4m3 "q"/"k"/"v" fetches) over the pinned `vit_matrix` form (one [3C, C] matrix through the quantiser + accumulator), plus `vr/native-vit.hpp`'s new `vit_qkv_post` — the per-head projection/inv/norm/temperature chain shared byte-identically with the e4m3 path (`vit_qkv` refactored to use it; its 404-check suite unchanged). The site rejects half-form and permutation-style records rather than silently mis-evaluating. Tests: the suite is now 429 checks at -O2/-O0, including the hand-derived two-head QKV case (the delta-input Hadamard, codes 1/0, products at rows 0/64, sqrt_head = f16(sqrt(32)) = 5.65625 -> e4m3 store 5.5, head 1's zeros through the floored inverse), the record-consumption pin (the run succeeds with no q/k/v records at all), the null-site fall-through and the shape/half/perm rejections with sentinel outputs. Independent review verified the refactor's byte-equivalence line-by-line and re-derived the case; its five findings (the sqrt_head literal, the record-contract gap, the silent half acceptance, the fetch ordering, a header-comment drift) all fixed and reverified. Eval (615,354/308,151) and vit (404) suites pass unchanged. Remaining: the expansion side-stream (open device-cache boundary), the kSplitQKVAttn attention site, the SwinSite hard-swish/fold. Synthetic correctness only: no vendor correspondence.

Twenty-fourth executable deliverable implemented (28/10/2026): the split attention QKV int4 site — `vr/native-split-swin.hpp`'s `SplitQkvWeight` products form (the pinned `attention_matrix` `[1536, 512]` block 16 through quantise+accumulate into precomputed per-token products plus the zero-input projection the padded slots take) and `vr/native-eval.hpp`'s "attn_qkv" gate before the e4m3 "qkv" fetch. The kernel's e4m3 path is untouched (34,267 checks unchanged). Tests: the suite is now 458 checks at -O2/-O0 — the exact-quantisation equivalence (both paths bit-identical, including padded slots and a shifted-window row), the quantisation-divergence discriminator (2.5 -> code 2 vs a 2.5 contribution), the record-consumption pin (no "qkv" fetch on the mixed path), the padded-product discrimination and the null-products rejection, plus the shape/half/permuted rejections. Independent review verified the row order/token-index/round placement, the scratch arithmetic and the equivalence derivation; its six findings fixed and reverified (the f16-stash seam divergence now documented; the null-base UB; the padded overclaim; stale comments; bounded coverage; pointer hygiene). The review also caught two real test bugs the discriminators exposed (arena aliasing across eval_reset; products saturating the binary16 norm chain). Recorded seam-wide divergence: both mixed sites quantise the run's e4m3-stored map where the pinned quantises the producer's f16 stash (int4.side) — documented in NATIVE-INT4.md; delivering the side channel is the open integration boundary shared with the expansion cache. Synthetic correctness only: no vendor correspondence.

Twenty-fifth executable deliverable implemented (28/10/2026): the int4 SwinSite — `vr/native-swin-linear.hpp`'s `SwinSiteOverlay`/`swin_linear_int4` (the per-row quantise+accumulate expand/QKV seam and the hard-swish activation swap), threaded through `swin_mlp`/`swin_attention`/`SwinBlockParams`, with `vr/native-eval.hpp`'s `SwinSiteRecord` + `build_swin_overlay` reproducing the pinned gating (hs iff fold != 0 with the (A,B) by channel count; the block-1 half-form ex/qkv matrices only for heads > 1 with both records). The e4m3 paths are unchanged (the four swin suites pass: 290/2,780/305,972/633,646; eval 615,354/308,151). Tests: 485 checks at -O2/-O0 — the hand-derived swin_hidden seam ([0, 7] through the half-form quantiser dots and the hard-swish gate), the builder-gating unit pins and the eval-level kSwin1H no-op indistinguishability. Independent review verified the hard-swish arithmetic operation-for-operation against the pinned fma form and the gating against the pinned constructor; its eight findings fixed and reverified — three real (the QKV threading had silently not landed; a stack bound unchecked; the int4 output validation weaker than the e4m3 path's). Remaining: the expansion side-stream and the f16-stash side channel (one open integration boundary). Synthetic correctness only: no vendor correspondence.

## Binary16 side channel + int4 expansion (29/10/2026)

The last int4 boundary closed (`18a8a9a`): the pinned `int4.side` stash pairs
are bound end to end — the expand's unrounded activation ("hidden"), the
contract/projection rows' binary16 value ("c"/"stream"), the split projection's
value ("attn") — and the expansion's mixed side-stream runs (stream stash only,
`out[:, perm] = permuted`, non-half ex). 518 checks at -O2/-O0; an independent
review raised eleven findings including three silently unwired pairs, all fixed.
The four regressions (eval/split-swin/vit/network) were re-run against the fixed
tree; see NATIVE-RECOVERY-ASSIGNMENT.md's latest continuation for the recorded
results and the remaining bounded items (the eval_pre_block non-null-history
addition).

## Runnable native Linux demo (28/10/2026)

The complete pipeline now runs as a zero-asset CLI demo (`vr/demo-native.cpp`
+ `vr/native-demo-support.hpp` + `vr/tests/native-demo-test.cpp` +
`vr/test-native-demo.sh` + `README-demo.md` + `vr/NATIVE-DEMO.md`): built-in
checkerboard/gradient or P6 PPM input, the full 76-row 'small' plan through
`image_evaluation` with deterministic synthetic weight presets (zero/seeded,
keyed per record), before/after/contact PPM artefacts, a temporal sequence
with a mid-stream reset (frames-since-reset printed), per-frame CPU timings
and the arena high-water mark (the CPU baseline for the GPU stage). 351
known-answer checks at -O2/-O0 + a bounded demo smoke. An independent review
raised 14 findings, all fixed — the significant one: the demo store's
single-buffer-per-kind clobbered the consumers' concurrently-live record
pointers (per-key slots now); plus the temporal_init axis order, the
FP16-store helper's write-before-reject, an unchecked CLI argument and the
missing MIT notice file. Honest scope: synthetic weights make the visual
transformation arbitrary — the demo proves the pipeline runs on Linux, not
DLSS quality and not vendor parity.

## Performance phase P1-P2: profile + deterministic CPU threading (28/10/2026)

P1 (`ce5bc1f`): the per-kind profile at the dispatch seam (`demo-native
--profile`) — the three ViT projection rows are 61.3% of the frame and the
projection-matmul class ~72%; attention kernels are cheap. P2: deterministic
CPU threading (`vr/native-parallel.hpp`, 2,479 checks at -O2/-O0 incl.
memcmp bit-parity at 1/2/3/4/8/16 threads): 36.3 s -> 16.3 s at 8 threads =
**2.22x** with byte-identical artefacts (16 threads = 2.15x; SMT adds
contention; the matmuls are memory-bandwidth-bound). Independent review
raised 7 findings, all fixed (spawn-failure no-throw handling, the missing
runner, 4 untested parallel sites + the long long path, -pthread in runners,
a false sentinel claim, the demo default and a comment). Serial remainder
documented: split_ffwd_proj/split_qkv_attn's status-returning inner calls.
Next: P3 (SIMD where P2 leaves headroom) then the Vulkan GPU stage per
family with parity + per-kernel timing (VRAM bandwidth is the expected
larger win).

## Performance phase G1: Vulkan projection matmul (29/10/2026)

First GPU slice published: `vr/native-vulkan.hpp` + `vr/shaders/matmul_e4m3.comp`
+ tests (57 checks at -O2/-O0) + `vr/NATIVE-VULKAN.md`. CPU-vs-GPU
bit-identity is contractual for normal-range intermediates (proven at
dispatch padding boundaries and full size); the denormal-intermediate
flush-to-zero divergence is measured/driver-dependent and pinned as such
(the over-claimed "flush(cpu)" contract was a review finding and is now
corrected). RTX 4070 Ti SUPER, 100x1024x4096: **gpu kernel 2.54 ms vs
cpu-1t 1,578.8 ms (622x) and cpu-mt 201.1 ms (79x)**; staging transfers
measured separately. Independent review raised 10 findings, all fixed
(tolerance contract + counterexample case, FTZ sign/NaN scoping, uint32
dispatch overflow, 128-wide workgroups within guaranteed limits,
timestamp-family preference, VkResult propagation, runner leftovers).
Next: the optional GPU backend in the eval dispatch (CPU reference intact),
then the split/attention + swin GPU families and end-to-end GPU frame
timings.

## Performance phase G2: the eval-dispatch GPU backend (30/10/2026)

Published: the optional ProjectionExecutor seam in native-eval.hpp (the
vit rows' raw projections on the GPU; post chains and all other rows stay
the CPU reference, null keeps the pure CPU path), the VulkanProjectionSite
(grow-only coverage, per-call decode+staging), demo `--gpu` (build with
-DDLSS5_DEMO_VULKAN; the default build stays CPU-only and Vulkan-free),
71 checks at -O2/-O0 including the three per-row seams, and the runner's
G2 gate (complete network CPU vs --gpu, byte-identical artefacts, GPU-run
asserted). End-to-end: **16,303 -> 14,242 ms/frame at 8 threads (1.14x)**;
GPU row cost is staging-dominated (kernel 223 ms vs decode+staging 509 ms).
Independent review raised 9 findings, all fixed (the CPU-only build break,
the kVitQKV wiring that had silently not landed — the recurring silent
patch-miss class, now asserted by grep count and the gate — the non-grow
resize thrash, missing Vulkan closes, unchecked SPIR-V read, the SKIP
contract, the silent-fallback gate, envelope/shader guards, comment
scoping). Next: G3 weight residency (persistent weight buffers; removes
the per-call decode+upload), then the split/attention + swin GPU families,
then the end-to-end GPU frame timings.

## Performance phase G3: the split/attention GPU family (30/10/2026)

Published: SplitProjectionSite (hoisted executor call per row; CPU post
chains) wired through split_ffwd_proj/split_proj/split_proj_pool/
split_final_head and the three eval rows; 86 checks at -O2/-OO including
per-row CPU-vs-site bit-identity and the pooled-scatter parity; the
complete-network CPU-vs-GPU gate stays byte-identical. Independent review
raised 5 findings, all fixed — including the MAJOR one where
split_proj_pool's scatter still took the CPU path despite the hoisted
matmul (my patch had silently not matched `value` vs `slot` — the silent
patch-miss class again; the new parity test now catches it), the gain-null
status divergence, the site/dimension validation order and a null-input
delegation gap. Next: G4 weight residency (removes the per-call
decode+upload that dominates the GPU row cost), then the swin GPU family,
then the end-to-end per-frame GPU timings vs the CPU baselines.

## Performance phase G4: the swin projection seam + a gitignore defect fix
(30/10/2026)

Published: SwinProjectionSite + ProjectionExecutor::matmul_f32 (the swin
family's decoded FLOAT weights; the shader is float-native) threaded through
swin_hidden/swin_mlp/swin_attention/swin_block/swin_core/swin_downsample/
swin_layer_upsample as a defaulted parameter — every default call stays on
the CPU, 95 checks at -O2/-O0 covering the strided seam bit-identity, both
fallbacks and the status ordering. SCOPE (stated in the docs): the eval
rows pass no site yet — per-window GPU calls (16 tokens) would thrash
under per-call staging, so the swin rows' GPU wiring lands with G5 weight
residency; the complete network is unchanged by this slice. Also fixed a
real G1 defect: .gitignore's allowlist never included *.comp, so the
shader source (vr/shaders/matmul_e4m3.comp) was NEVER TRACKED — a fresh
checkout could not build the tests; the allowlist and the source are now
committed. Independent review raised 6 findings, all fixed (swin_layer_upsample
dropping the site at its two swin_block calls — the silent-miss class again,
now caught and fixed with the downsample row threaded consistently; the
site validation keyed on the wrong callback; a push-constant overrun and
dead weight_mode residue from an earlier misfire; the executor-envelope
narrowing now falls back to the CPU kernel; three comment overclaims; two
test gaps). Next: G5 weight residency (persistent weight buffers; the
prerequisite for wiring the swin rows AND for the split rows' many small
calls), then the end-to-end per-frame GPU timings vs the CPU baselines.

## Performance phase G5: weight residency (30/10/2026)

VulkanWeightCache: persistent weight buffers, first-sight upload, verified
hits skip staging and the e4m3 decode. Soundness: the pointer key is
verified by a content hash (recycled per-call record storage delivers
different content behind one pointer — the demo's store does exactly this),
re-uploading in place on mismatch; a generation counter defeats VkBuffer
handle reuse after the matmul's grow-only rebuild; exhaustion falls back to
staging. 116 checks at -O2/-O0 including the mutated-key discriminator and
the rebuild-rewrite test. Measured: 12,380 ms/frame end-to-end (1.32x vs
cpu-mt 16,303; 1.15x vs G2's 14,242) — the demo's round-robin record store
recycles pointers and forces hash-mismatch re-uploads, so the full
residency benefit needs stable record storage (a G6 item along with the
swin/split eval rows). Independent review raised 7 defects
(2 HIGH: the e4m3 hit bypassing the stale-set rewrite; the unsound pointer
key), all fixed and reverified. Next: G6 end-to-end per-frame GPU timings
of the complete network vs the CPU baselines with the swin/split eval rows
wired (the residency now makes their per-window calls viable), then the
reported speedup summary.

## Performance phase G6: full swin wiring + the stable record store
(28/10/2026)

The swin eval rows (pre/post, downsample, upsample, core) are WIREABLE and
parity-verified (fixed per-window site sizing: the swin windows are always
64 rows; the review caught the map-based sizing being undersized for small
maps — HIGH finding, fixed). The per-key stable record store (one lazily
allocated buffer per record key, no round-robin eviction) collapses the
decode column 885 -> 455 ms/frame. MEASURED SURPRISE (honest): with the
swin rows' GPU site enabled the end-to-end REGRESSES to 14,418 ms/frame
(vs 12,380 G5) — the ~5k per-window matmul calls thrash the per-call
input/output staging (3,337 ms upload + 5,549 ms download vs 1,229 ms
kernel). The swin rows were therefore OPT-IN (EvalContext::swin_gpu,
default false) at G6 (SUPERSEDED by the G8 entry below — the batched seam
measured faster than the CPU rows and the default is now true); the
recorded G7 fix was to be an offset-dispatch seam (upload each swin row's
map once, per-window matmuls with device-local offsets, download once —
bit-parity preserved). G6
default measured: 11,717 ms/frame (1.39x vs cpu-mt, 3.22x vs cpu-1t),
parity byte-identical, 615,365 checks green. Review: 5 findings (the
HIGH scratch-sizing defect, the DemoStore double-free risk from the
implicit copy, three comment staleness items), all fixed and reverified.
Next: G7 the offset-dispatch window seam, then CPU SIMD (item 3 of
Michael's brief), then sustained-operation hardening (item 4).

## Performance phase G7: CPU decode + SIMD + split-QKV hoist + threading
(28/10/2026)

All bit-exact on the CPU reference. Branchless e4m3 decode (2.05x,
exhaustive 256-code verification against independent oracles); a
bit-exact AVX2 dot8 matmul (lanes keep the scalar key order; size-gated at
c_in >= 256 where measured 1.9x; SLOWER below, hence the gate; decode8
swept exhaustively); the split-QKV projection hoisted to one batched
matmul (bit-exact CPU fallback via split_matmul's wider envelope; the
executor path carries the documented tolerance); the P2 serial remainder
(split_ffwd_proj tokens, split_qkv_attn windows) threaded with recorded
failure status and 1-vs-{2,3,8,16}-thread memcmp parity tests.

Measured: 7,855 ms/frame end-to-end (4.80x vs cpu-1t, 2.08x vs cpu-mt;
from G6's 11,717). SplitQKVAttn row: 1,828 -> 89 ms. The swin family is
now 79% of the CPU profile (CPU-side per G6's opt-in finding). Parity gate
byte-identical; eval 615,365 checks green. Review: 6 findings (the
medium ones: the hoist's bit-exact claim did not hold on the executor path
-- comment scoped + fallback widened; the threading claim was untested --
cross-thread memcmp parity added), all fixed and reverified.

Next: item (4) of Michael's brief -- sustained-operation and stability
hardening of the GPU path (long runs, memory stability, device-loss
behaviour documented honestly); then the swin rows' batched/offset-dispatch
window seam (G6's recorded fix) if hardening holds.

## Performance stage 4: GPU-path hardening (28/10/2026)

vr/tests/native-vulkan-stress-test.cpp + vr/test-native-vulkan-stress.sh +
vr/NATIVE-VULKAN-STRESS.md: 1,965 checks at -O2/-O0 covering 50
init/teardown cycles (CPU allocation accounting balances each cycle),
200 mixed-shape resident runs bit-identical to the independent scalar
reference, the weight cache's content-hash re-upload and capacity
behaviour (511 accepted of 520 distinct keys; first past capacity
rejected), and failure paths (sentinel-untouched outputs; oversized
tokens/rows rejections added). Sustained operation: the 50-frame demo run
(mean 7,662 ms/frame) keeps the arena high-water flat at 48,531,972
floats; RSS plateaus at ~573 MB with a residual ~1.5 KiB/s creep after
warm-up (below the C++ layer — CPU allocation accounting balances exactly;
documented as an unresolved slow term, not claimed zero). Device loss is
an HONEST GAP (no fault injection): VkResult paths return Status and the
stale-set rewrite is guarded by the written_in/written_out handle check
(the generation counter is inert — recorded), but real device-loss
recovery is untested and not claimed. Review: 9 findings (the HIGH one:
the "400 runs" figure vs the actual 200; plus the inert generation-counter
claim, mislabelled/missing rejection classes, sentinel gaps, the operator
delete set) — all fixed and reverified.

Next: the swin rows' batched/offset-dispatch window seam (G6's recorded
fix, would lift the 79% CPU swin share), or close the performance stage
and hand over.

## Device-loss honesty slice (28/10/2026)

The gap is closed in both directions Michael named. (1) Forced-failure
testing: the test-only VulkanFaultPlan seam injects VkResult-level
failures (vkQueueSubmit / vkMapMemory) WITHOUT touching the driver (armed
for one failure; inert at -1, production behaviour unchanged); each armed
failure is verified to surface as Status with the output sentinel
untouched and the path usable afterwards. (2) The generation counter is
LIVE: destroy preserves it (it is identity, not a resource), create
advances it (a rebuild moves 1 -> 2), and the ensure-insert records
written_gen (previously 0); the stale-set rewrite is verified through a
rebuild + cache hit. Recovery is documented PRECISELY: forced failures
need no recovery (Status + sentinel + next run correct); real device loss
is NOT implemented/claimed (the caller must discard the Vulkan context;
in-flight command buffers are undefined) and the injection is call-result
level, not driver device-loss. native-vulkan-stress-test: 1,997 checks at
-O2/-O0; the main vulkan suite + the complete-network parity gate stay
byte-identical. Review: 14 findings; the real gaps (the content-hash
contract tests, doc wording) fixed; findings 1/MAJOR-1/MAJOR-2 were
verified against the tree as already-fixed or unfounded (the tests assert
written_gen-after-insert and the symmetric aligned accounting; the run
returns before the download on a failed dispatch at native-vulkan.hpp
933-935) — recorded rather than silently dropped.

Remaining: the batched/offset-dispatch swin seam (gap 1 of the 28/10
brief) with its honest measurement (the swin rows stay opt-in until a
batched path measures faster than CPU).

## Batched swin seam slice (28/10/2026) — gap 1 closed

The batched/offset-dispatch gap is closed with the batched form (it
subsumes offset dispatch and measured better). `swin_block`'s per-window
loop is one shared `run_window_batch` body with two extents: the per-window
path (byte-identical to the published G4/G6 behaviour) and a chunked path —
one `swin_mlp` + one `swin_attention` call per `kSwinBatchWindows` (32)
windows, so each site-projected stage (expand/hidden and QKV) is ONE
executor call per chunk instead of one per 64-token window. Every stage is
row/window independent, so chunked and per-window results are
bit-identical by construction; the tests memcmp them over the window-count
matrix {1, 2, 16, 32, 33, 92}, both head families (incl. the grouped-mid
path), both residual paths, zero/non-trivial corpora and both acc modes,
plus the executor call-shape contract (chunk-shaped calls — the thrash
elimination is pinned, not assumed), every capacity/shape fallback
predicate (per-window preserved), the null-executor and null-gather G4
rejection classes, the padded-weight CPU fallback and `swin_layer_plain`'s
site threading (that row was CPU-only; it is now wired like the other
three). Chunk scratch rides in the SwinProjectionSite (`batch_work`,
sized by `swin_batch_work_needed`/`swin_site_scratch_needed`); the eval
rows carve chunk-sized buffers (`swin_site_count`/`swin_site_carve`).

MEASURED (RTX 4070 Ti SUPER, 64x72 kSmall, 8 threads, 4-frame means):
cpu-mt 8,301 ms/frame; G7 config (swin CPU) 7,770; G8 batched K=32
**6,850 (1.21x vs cpu-mt, 1.13x vs G7)** — byte-identical artefacts. The
executor call count collapses ~5,000 -> 427 per frame; GPU-side is now
download-bound (1,348 ms/frame of ~1.2 GB products, the honest remaining
ceiling). K=8 pays thrash again (7,199); K=64 outgrows a 512 MiB arena
(recorded). DECISION: swin GPU rows are faster than the CPU rows, so
`EvalContext::swin_gpu` defaults TRUE with an executor attached
(`--cpu-swin` reproduces the G6/G7 configuration); the demo arena is now
512 MiB (high-water 185 -> 345 MiB, the batched scratch cost). The
complete-network gate runs three ways (CPU, --gpu --cpu-swin,
--gpu --gpu-swin) requiring byte-identical artefacts and routing evidence
(61 -> 427 calls). Check counts: swin-block 5,772,726 at -O2/-O0; all
sibling regressions green (eval 615,365/308,162; vulkan 116 + gates;
stress 1,997; the rest unchanged). Independent read-only review: PASS
(7 findings — the two doc-staleness and three nit items fixed here, the
two disclosures recorded in NATIVE-SWIN-BLOCK.md and the handover).
Speed numbers are real; synthetic weights mean the output is arbitrary —
no image-quality or vendor-parity claim.

Next: the CPU synthetic input-to-image evaluator hardening or the
temporal history/reset consumer (the second colour copy is still the
single-frame stub), then live-input evaluation — recorded in
NATIVE-PERFORMANCE.md's G8 section as the remaining performance-stage
items.

## Temporal history consumer slice (28/10/2026) — overnight phase 1

The single-frame stub (the lift's second colour copy) is replaced as the
temporal feed by the real history/reset consumer in the native path. New
in native-io.hpp: `image_map_evaluation` (the extracted map-level core of
`image_evaluation` — published behaviour unchanged), `frame_to_history`
(the FP16 output-texture store: planar -> interleaved, binary16 round,
stride-honouring, full-pass value validation) and `temporal_image_evaluation`
(the frame step of the capture contract: optional DLSSNR.Reset first, the
history feed ALWAYS the carrier's view — the zero map at start-up and
after a reset, never the stub — the run, the store, the host read-back and
the bit-exact commit with the counter advance). The demo's inline
convert/round/commit block is deleted in favour of the consumer (its
report now shows the frame's contract inputs).

SEMANTIC CORRECTION (intended output change): at start-up and post-reset
frames the old demo passed the stub (the current colour) to the lift; the
capture contract says those frames behave as if history = 0, so the feed
is now the carrier's zero map. The demo output changes at those frames
accordingly; the state machine is exactly periodic across a reset (the
demo's frame 1 and frame 3 float-deltas are identical), which is the
contract's restart behaviour made visible.

MEASURED CHECKS: native-io 2,351 at -O2 and -O0; demo smoke green; the
complete-network G2/G8 gates byte-identical. The feed pin is
discriminating (a frame-tagged layer gives each step a distinct map: the
frame-2 feed must equal frame 1's commit, not output_{N-2} or the frame's
own output); the counter literal [0,1,2,3,0,1,2,3] is pinned through the
consumer; the reset ignores staged content (injectreset); every rejection
class leaves out_host/store/carrier untouched (the reset-first form is
documented and pinned).

Independent read-only review: FAIL (3 findings + 4 nits) — all fixed and
reverified: the commit's counter-exhaustion guard could fail AFTER the
outputs were written (now hoisted to the step's entry; the reset cures
exhaustion), frame_to_history read strided views without validation
(bounds added before the scan), the feed pin was not discriminating (the
tagged layer), the doubled extent could overflow (bounded), the result
map's layout was not pinned (a padded result is now rejected rather than
misread), a never-initialised carrier map failed only mid-run (rejected
at the seam), and an overstated comment (scoped).

Next: overnight phase 2 — live-input evaluation (sustained multi-frame
feeding of the evaluator with measured throughput and no resource leaks).

## Live-input evaluation (29/10/2026, overnight phase 2) — `vr/NATIVE-LIVE.md`

`vr/tests/native-live-test.cpp` + `vr/test-native-live.sh` feed full-network
frames through the real consumer (`temporal_image_evaluation`) in three
sequences (sustain + mid-stream reset, bit-exact replay on fresh carriers,
different-input responsiveness) and assert the sustained-run contracts:
balanced allocations across every frame (the parallel layer's spawn-and-join
thread state is balanced churn, not zero), constant arena high-water, bounded
RSS after warm-up, and the counter schedule including the reset's zero
(observed directly on the carrier's post-call state).

Measured (RTX 4070 Ti SUPER, -O2, 8 threads, 9 frames): **8,260-8,440
ms/frame** (decode 0.24 ms, output 0.03 ms; pipeline derived), 89,320
allocations/frame balanced, arena constant, RSS growth 76-276 KiB. **87
checks at -O2**; suite green in ~93 s. Speed numbers are real measurements;
the synthetic weights make the transformation arbitrary (no image-quality
claim).

Independent read-only review: FAIL (6 findings: 4 MINOR, 2 NIT) — all fixed
and reverified: a tautological reset-frame counter assertion (now the
post-call carrier counter is asserted directly), an unguarded fixed 320x320
probe assumption (now asserted), two stale comment claims (-O0 run and
serial coverage; both corrected to the real configuration), the wrong RSS
baseline claim, and a hard-coded 4 KiB page conversion (now
`sysconf(_SC_PAGESIZE)`).

**Open defect (recorded, performance-only):** an uninitialised heap read in
the network path, proven by `MALLOC_PERTURB_=165` flipping clean -O0 probes
into stalling on their second run (output hashes unchanged; auto-initialised
locals exonerated; ruled out: contention/throttling, buffering artefacts,
denormal assists alone, the record store, the arena allocator, the consumer
phases). Exposure is -O0 codegen only — -O2 stays clean at both thread
counts even under `MALLOC_PERTURB_`. The exact site is not localised; the
suite's -O0 run is compile-only until it is. NATIVE-LIVE.md carries the full
evidence trail.

Next: localise the uninitialised read (candidates: the swin scratch
partitions and the per-row decode staging in `native-eval.hpp`; add a
regression reproducing under `MALLOC_PERTURB_`), then overnight phase 3
(int4 mixed-network path hardening), phase 4 (sustained GPU hardening) and
phase 5 (desktop delivery boundary).

## int4 mixed-network path (29/10/2026, overnight phase 3) — `vr/NATIVE-INT4.md`

**A real seam gap found and closed.** The pinned's `int4.side` is a dict
whose keys may be ABSENT; the vit FFN expansion's ("stream", block) key has
no producer for the FIRST vit block (the split block before it writes no
stream), so the pinned falls back to the e4m3 form there. The seam gated the
int4 expansion on the caller's buffer POINTER: with buffers present it
diverged from the pinned at the first block and read the caller's
never-written buffer (an uninitialised-content hazard). Fixed with
`EvalContext.stash_stream_block` — the exact dict-key presence semantics:
the projections stamp the block they write for (only after a successful
write), the expansion requires the stamp to match its own block,
`eval_reset` clears it, and callers without buffers are behaviourally
unchanged (the demo/live/GPU paths).

**The plan-level integration suite** (`vr/tests/native-eval-int4-network-
test.cpp`, 374 checks at -O2) runs the full 76-row small plan with the
supported rows on their int4 sites: the record-consumption policy over
every row (a mixed row never fetches its e4m3 twin; the first vit block's
expansion fetches "weight" in every config — the pinned's fallback), the
side channel's end-to-end routing discriminator (the mixed expansion's
missing "weight" fetch needs the ("stream", block+1) chain across block
boundaries; the has_ex-off control fetches "weight" again), row-level
discriminators (the final map cannot see vit-row-only changes — the
synthetic weights collapse the bottleneck's late rows to zero), determinism
pairs, the network_run rejection contract and the temporal consumer over
the mixed network. The row suite gained the producer-side stamp pin (519
checks at both levels).

Independent read-only review: FAIL (3 MINOR + 4 NIT) — all addressed (the
stamp after the successful write; the replay compares both frames; the
vacuous sentinel fixed; the symmetric stamp; two nits removed) with one
clang-wording nit recorded (family template; clang unsupported). Sibling
regressions green (eval 615,365/308,162; eval-int4 519/519; network 3,564;
vit 662; split-swin 34,272).

Next: localise the uninitialised read (phase 2's recorded defect), then
phase 4 (sustained GPU hardening: soak, device-loss re-verify, arena
stability at the 345 MiB high-water) and phase 5 (stretch: desktop delivery
boundary + a Centre Stage branch on a worktree).

## Sustained GPU hardening (29/10/2026, overnight phase 4) — `vr/NATIVE-PERFORMANCE.md`

`vr/test-native-vulkan.sh` gained the G8 soak: six frames with a
mid-stream reset through the temporal chain on the GPU path — artefacts
byte-identical to the CPU reference over the whole sequence (the G2/G8
gates are single-frame), the arena high-water pinned at the measured level
(90,298,116 floats = 344.4 MiB, under a 350 MiB bound), and peak-RSS
stability across the run (689,496 -> 689,664 KiB over four extra frames =
168 KiB growth; a 4 MiB creep bound). Device-loss honesty re-verified
green (the stress suite's 1,997 checks including the forced-failure
injection and the live generation counter).

Next: phase 5 (stretch: desktop delivery boundary + a Centre Stage branch
on a worktree — the tested boundary with a mock host is a valid outcome),
and the recorded defect (the uninitialised read's localisation).

## Desktop delivery boundary (29/10/2026, overnight phase 5 stretch) — `vr/NATIVE-HOST.md`

The stable host boundary: `vr/native-host.h` + `vr/native-host.cpp`, a C
ABI over the temporal frame step for desktop renderers (create/evaluate/
destroy, the counter with resets, the arena report, stable status strings)
with the labelling IN BAND — `dlss5_nr_label()` returns "synthetic weights
— arbitrary transform" and the contract says a host must display it. The
mock host (`vr/tests/native-host-test.c`, 45 checks at -O2) is a pure C
consumer compiled as C11 against the C++ implementation: the ABI exactly
as a host sees it, including the untouched-output-on-rejection contract
and the pipeline's cross-handle determinism.

The Centre Stage interop was investigated (C# + Silk.NET OpenGL 3.3;
P/Invoke over this ABI is mechanically clean — the app already reads
framebuffers): the full app branch is NOT delivered — the evaluator costs
~8 s/frame at the demo geometry, the renderer has no test harness, and the
brief's stated valid outcome is the tested boundary with a mock host when
the interop is not clean end to end. The app branch (a worktree, never
main) is the recorded next step for a session with display access.

Next: the recorded defect (the uninitialised read's localisation with its
MALLOC_PERTURB_ regression), then the Centre Stage worktree branch.

## Hygiene: the uninitialised-read investigation closed + regression (29/10/2026 delivery phase 1) — `vr/NATIVE-LIVE.md`

The recorded defect did not reproduce and the confound that produced its
signatures was identified: the workspace's command runner kills foreground
commands at ~120 s (recorded in `.tools/job-cap-ticks.log`, a 10-tick
heartbeat cut at tick 13 = 120 s), and an -O0 network frame is 36-94 s —
so "clean 4-run probe vs perturbed probe stalls on run 2 with run 1's hash
unchanged" is exactly the cap's kill signature, as is "-O2 immune" (8
s/frame, never near the cap). The original probe's per-run cost was not
recorded, so the exact boundary cannot be reconstructed; every long-run
failure observed in this workspace ended at ~119-125 s of work. Everything
below is from detached runs.

The localisation evidence: `MALLOC_PERTURB_` sweeps at -O0 (seven runs —
six at 6 frames, one at 3; 6 fills; 1 and 8 threads) — 39 frame hashes
identical, no stall; arena poisoning with NaN/denormal/random (what glibc's
fresh-malloc fills cannot express) — bit-identical hashes; ASan+UBSan -O0
4 frames clean; valgrind memcheck `--track-origins=yes` two full frames
(fresh-run and leftover-state) — both DONE with the clean hashes and
`ERROR SUMMARY: 0 errors from 0 contexts`; the live suite at -O0 passes 87
checks unperturbed and under `MALLOC_PERTURB_=165`; and the recorded
candidates themselves (the swin scratch partitions + per-row decode
staging) poisoned with 5 poisons against a zeroed baseline produce
bit-identical `swin_core`/`swin_mlp`/`swin_attention` outputs for all four
head families. No uninitialised read with any observable effect exists in
the exercised paths; the record is retired in `NATIVE-LIVE.md` (historical
text kept).

The standing guard: `vr/test-native-uninit.sh`
(`vr/tests/native-uninit-test.cpp`, 246,186 checks) — the MALLOC_PERTURB_
regression: kernel-level scratch/decode-staging poisoning, a full-network
arena sweep (painted run + leftover run), and a cross-process
`MALLOC_PERTURB_` sweep asserting identical output (the full value set at
-O2, the recorded 165/255 at -O0). The live suite's -O0
run is REINSTATED (depth 2) and the mock host's -O0 run reinstated; the
int4-network suite keeps its -O0 build compile-only as a runtime bound
(the defect-based reason retired).

Next: the Centre Stage worktree branch (in flight: `feature/dlss5-native-seam`
on a worktree of `/home/michael/Projects/CentreStage`, the C ABI wired by
P/Invoke with on-demand frame-by-frame processing and the in-band label),
then the VR/Genesis seam integration plan (stretch, milestones only —
delivered as `vr/VR-SEAM-PLAN.md`; the headset session happens with
Michael at the Genesis machine).

## Centre Stage seam delivered (29/10/2026 delivery phase 2) — `vr/NATIVE-HOST.md`, branch `feature/dlss5-native-seam`

(Delivery commits: this repository `e48275d`; CentreStage
`feature/dlss5-native-seam` at `6ebfa57`, its main untouched at `80c19c1`.)
The app branch is delivered on a worktree of `/home/michael/Projects/CentreStage`
(never its main): `Dlss5Interop.cs` (P/Invoke over the tested C ABI),
`DlssStage.cs` (64x72 capture framebuffer + nearest presenter),
`DlssSeam.cs` (Idle/Processing/Presenting/Failed state machine; evaluate on
a worker thread while the live view keeps rendering; F9 = one frame,
camera input unpins; `--dlss-demo outdir` drives the unattended evidence
run), `linux/build-native-host.sh` (builds `libdlss5_nr_host.so` from this
repository; artefact gitignored). The in-band label ("synthetic weights —
arbitrary transform") is displayed with every presented frame and captured
in the evidence; the measured latency is worker wall-clock around the
native call (8.7 s/frame; an earlier 60 s reading was traced to the
idle-blanked compositor's frame throttling, not the evaluator — the
latency is now measured on the worker and the presentation delay reported
separately). The app's own workflow is unbroken (56 core tests;
`--capture` regression verified).

Unattended evidence (screenshots + report, kept out of Git):
`/home/michael/Projects/CentreStage-dlss5/.evidence/dlss-demo-final2/`
(live.png, processing.png — the live view renders full-size during the
in-flight frame, processed.png — the presented synthetic output with the
label in-band, report.txt, run.log). The synthetic output rails G->0 and
B->255 while R tracks input structure — identical to this repository's own
demo artefacts (`.tools/asan-out/after.ppm`), i.e. the arbitrary transform
presented faithfully, not an integration artefact.

Next: interactive F9/session with Michael when convenient; the VR/Genesis
plan milestones (Milestone 1 needs the Genesis checkout on platinum).

## VR seam Milestone 1–2 + AMD/RADV portability (29/10/2026, delivery phase 3) — `vr/VR-SEAM-M1-ALIGNMENT.md`

Milestone 1 executed against the real contract: the Genesis checkout on platinum (repo `be98965`) carries `NEURAL-SEAM.md` + `neural-seam.h` at its root (the plan's `vr/` path was approximate). The contract mapping and conformance table, the ownership split with dlss5-vr and the key open question are resolved in `vr/VR-SEAM-M1-ALIGNMENT.md`: the 320x320 refusal is `native-host.cpp`'s fixed 512 MiB arena guard (`kMaxPaddedSide`), not a kernel limit (the plan is extent-parametric to 16384); the measured arena high-water at the floor is 48,531,972 floats (185 MiB), while a headset per-eye extent (~2064x2208) pads to 2112x2240 under the plan's own rules = ~46x the tokens (≈8.5 GiB arena per eye, ≈6.7 min/eye/frame on CPU), so the recommendation is a **bounded per-eye window (<= 320x320 padded)**, composited by the caller, with arena extension left to the dlss5-vr GPU path.

Milestone 2 plumbing done without the headset: `vr/test-native-stereo.sh` + `vr/tests/native-stereo-test.c` — two evaluator handles as the two per-eye contexts the contract ABI requires, synthetic stereo pairs, per-handle counters, per-eye history independence (a shared history must fail), history consumption, the reset semantics the rule-2 composition relies on (the composition itself is pinned by the seam adapter suite), reset determinism, and a side-by-side PPM artefact with the label recorded beside it. Independent review of this slice raised the history-independence hole (a shared history passed the first version), the padding arithmetic (corrected to 2112x2240/~46x), the shared process-wide parallelism setting and artefact-write checking — all fixed in this change set and reverified by the family. On the editor side, the Genesis worktree branch `feature/dlss5-native-seam` (main untouched at `be98965`) carries the seam adapter implementing `neural_seam_evaluate` over two `dlss5_nr` handles (`src/neural-seam-native.cc`, contract header unmodified, boundary vendored in `third_party/dlss5-native` from this workspace): 29 checks pin contract rules 1-3 (rule 2 enforced idempotently as well as caller-side), history accumulation (an always-reset adapter must fail), context-aliasing and mismatched-output-extent rejection, the FP16/RGB8 conventions (alpha 1.0, clamped colour) and the output-untouched-on-error contract; `run-tests.sh`'s core suite stays green at 859 checks. Independent review of the adapter found the output-extent validation hole (a presented extent smaller than the render extent overran the caller's buffer — the evaluator is same-extent by design, so mismatches are now refused untouched) and the missing aliasing rejection — both fixed here; the FP16 helpers' clamp/RNE behaviour is pinned by an in-adapter conversion self-test because the synthetic transform would otherwise hide it.

Portability leg (platinum, Radeon 8060S, RADV; logs in `~/dlss5-port-logs/`, source mirror `~/Projects/dlss5-linux-port`, everything `nice -n 19`, no services touched): `vr/test-native-vulkan.sh` **suite passed** — G1, **G2 complete-network parity byte-identical**, **G8 batched swin parity byte-identical** (the GPU really ran: 427 batched swin calls) and the G8 six-frame soak; `vr/test-native-stereo.sh` 42 checks (the strengthened suite: history accumulation, aliasing and extent rejections); `vr/test-native-host.sh` 45 checks at -O2 and -O0 (the sanitised build is skipped there — libubsan absent, a recorded environment gap). The byte-identical CPU-vs-GPU artefact contract holds on RADV as well as on the NVIDIA reference; no divergence to attribute. The seam adapter is vendored in the worktree from this workspace (`third_party/dlss5-native`), whose `dlss5_nr_create` also notes the one shared knob: the evaluator's parallelism is process-wide (last create wins), so both eye handles are created with the same thread count. Stretch item (the download-bound GPU ceiling) not started this phase.

Stopped at the headset boundary as planned: the in-headset inspect session, the passthrough/horizon-lock interplay check and the in-headset latency budget remain Milestone 3 work with Michael (see `vr/VR-SEAM-PLAN.md`).

## Stretch: the download-bound GPU ceiling characterised and partially attacked (29/10/2026, delivery phase 3) — `vr/NATIVE-PERFORMANCE.md` G9

The stretch item began with a measurement, and the measurement reframed it: the staging transfers dominate the GPU path (baseline `gpu totals`: kernel 781 ms vs upload 506 ms + download 2,552 ms over 2 frames), and the landing (one queue round trip per matmul call instead of three - same recorded operations, same order, one `vkQueueWaitIdle`; plus persistent staging mapping) removed the round-trip and map/unmap overhead (upload 506 -> 9 ms) but moved end-to-end only 6,604 -> 6,570 ms/frame (noise). The download stayed at ~2.4 s: it is BANDWIDTH-bound (the raw f32 products, ~3 MB per swin chunk call), not latency-bound. Byte-identity is proven across the old and new forms (G1/G2/G8 + the six-frame soak green; demo artefacts `cmp`-identical across all three configurations).

The actual fix is therefore the brief's first prong - keep post-processing resident (the CPU post rounds to binary16 immediately, so a resident post downloads half the bytes and deletes the CPU pass) - and it pays ONLY for rows without a residual: a `has_skip` row needs its f32 skip uploaded, costing as much as it saves. The next slice must measure the skip/no-skip split of the download volume first. "Streamed reductions" does not apply: nothing in the post chain reduces. The parity contract is unchanged - the G2/G8 byte-identical gates are the arbiter, or a declared measured tolerance as G1's denormal class does.

## Scope pivot item 1: the data-free weights interface (29/10/2026) — `vr/NATIVE-WEIGHTS-FILE.md`

The weight-file loader + compatibility-verification harness (`vr/native-weights-file.hpp`, `vr/weights-verify.cpp`, `vr/validate-weights.sh`): the NWF1 container (documented byte layout, FNV-1a integrity hash), a recorder that captures the plan's exact request set through the EvalWeights contract, a file-backed provider, and a verifier that names every mismatch class (missing/unexpected tensors, shape, structure, content). The FULL-NETWORK BYTE-IDENTICITY ROUND-TRIP is the core property: dump the request set from a synthetic run, reload it through the file provider, and the artefacts are `cmp`-identical. The resume path is the single command `vr/validate-weights.sh <file>` (manifest regeneration + verification + CPU-vs-GPU byte-identity reference validation + label check). 72 checks at -O2/-O0. Data-free: no NVIDIA assets, no proprietary payloads, no runtime probes; the container is our documented format and the verification consumes tensor records, so a legitimately obtained source-format file reaches the same checks through one decoder. Two contract subtleties are pinned by tests: the wildcard (`count == -1`) fetch semantics and the exact-count length-discovery probing (a provider that answers any count breaks `record_length_*`), and the `length_for` hook that captures wildcard probes at the plan's true table lengths (the wrong hook length was caught by the full-network run, not the unit tests - the family runs both).

## Scope pivot item 2: the visible-quality proof (30/10/2026) — `vr/NATIVE-DEMO-UPSCALE.md`

A REAL transform through the identical seam: `vr/native-demo-upscale.hpp` (Lanczos3 a=3 + documented optional unsharp — a CLASSICAL fixed-kernel scaler with no learned weights; the earlier wording here calling its kernel parameters "real weights" is superseded by the label policy below) behind the same C ABI (`dlss5_nr_set_demo_transform`/`dlss5_nr_label_for`), the same 64x72 capture and the same present path. Quality measured honestly (`vr/tests/native-demo-upscale-test.cpp`, generated 320x288 source, x4 box-downscale capture): the default pure Lanczos3 beats the bilinear baseline on BOTH metrics — PSNR 14.79 vs 14.74 dB, SSIM 0.793 vs 0.776 — with the parameter sweep printed as the tuning record. Geometry: 64x72 capture -> 256x288 presented (x4); the transform costs milliseconds per frame (the seam stays F9-triggered by design). Centre Stage runs it by default; the unattended evidence (`.evidence/dlss-demo-real/`) shows the in-band label "not DLSS5 - demonstration transform with real weights" on the presented frame. At the time every artefact carried that label; the label policy was later corrected (the "real weights" phrase now belongs only to the neural path - see the section below). This is a platform proof end to end and NEVER a DLSS5 claim.

## The demo experience fix + the neural slice (30/10/2026) — `vr/NATIVE-DEMO-UPSCALE.md`

Two verdicts from Michael drove this section. First ("it just pixelated the model" at 64x72): the capture geometry is the seam's knob - the demo default is now a half-window capture upscaled x2 (640x400 -> 1280x800, one uniform scale; 64x72 x4 kept as `CENTRESTAGE_DLSS_EXTREME=1`), F10 cycles the presented result between the scalers at the SAME capture (the visible A/B), and the boundary's create contract is two-tier (demo-only handles up to 2048x2048; the plan/arena tier unchanged at the network floor). The first evidence run was caught by the independent review as silently broken (the capture viewport was still 64x72 inside the large FBO - the 1.2% A/B diff fitted exactly the corner; the fixed run shows 7.8% of RGB bytes differing). Re-measured at x2: PSNR 20.49 vs 19.62 dB (+0.87), SSIM 0.9138 vs 0.9067 (+0.0071) - the SSIM margin is small and is stated plainly.

Second ("she goes pixelated"): Lanczos3 is a classical fixed-kernel scaler and cannot restore detail, so the neural slice was required. Real learned upscalers with real open weights now run behind the identical seam (`vr/native-demo-rrdbnet.hpp` safetensors/RRDBNet and `vr/native-demo-edsr.hpp` torch-save/EDSR; `vr/fetch-demo-model.sh` pins both by checksum; no NVIDIA assets, no proprietary data, no runtime probes): EDSR-base x2 (Apache-2.0, DIV2K, 1,369,883 measured params, the default) at 10-11 s/frame CPU and Real-ESRGAN x2 (BSD-3) at 43.2 s/frame. The honesty labels are fixed: the classical scalers are "classical demonstration scaler" (Lanczos3 has NO weights and no longer claims any); "demonstration transform with real weights" is reserved for the neural path, each under its own model name; "not DLSS5" is in-band everywhere. The measured verdict is documented without overselling (NATIVE-DEMO-UPSCALE.md's table), and it flipped once the review caught a real integration defect: the neural branch read the planar buffer as interleaved, silently degrading every neural measurement (the first numbers - EDSR 32.61 dB and an 8.5x detail overshoot - were artefacts of that scramble). With the layout fixed and pinned bit-exactly against a direct forward in the smoke, the real-capture numbers are: EDSR 36.98/0.9812 full frame and 31.92/0.9539 centre crop vs bilinear 40.41/0.9920 and 35.10/0.9731 - pixel fidelity still favours the classical scalers - while the detail-energy row supports the demo's actual claim: the ground truth carries 198,570, the classical scalers smooth 81-91% of it away (17,195/38,100 - the softness Michael saw) and the neural view reconstructs 94% (186,684), which is what the F10 A/B shows. Two model classes were tried per the brief; both measured honestly; the numbers support visible DETAIL recovery while pixel fidelity favours smoothing, and the docs say exactly that. The evidence run (`.evidence/dlss-demo-edsr/`) carries live/processing/processed/baseline/lanczos3 screenshots plus the per-view report (full frame + pre-declared centre crop + detail energy). Families green: demo-upscale 21 checks + both neural smokes (the analytic smooth-gradient implementation check pins EDSR at 30.6 dB - a broken sub-pixel shuffle would score under 20 dB), host 86 checks, CentreStage 56/56.

## Scope pivot item 3: the VR headset attempt (30/10/2026) — Genesis `334e7ae` + review fixes

Verified unattended on platinum (AMD 8060S, RADV): the OpenXR runtime resolves for native apps via `XR_RUNTIME_JSON` pointing at the WiVRn flatpak manifest (the path embeds a flatpak commit hash that changes on WiVRn updates - re-derive with `ls /var/lib/flatpak/app/io.github.wivrn.wivrn/x86_64/stable/`); with it set, `--smoke-xr` gets PAST instance creation and blocks waiting for the headset (the documented boundary - the session itself is NOT verified). The new `--capture-eyes DIR` mode renders the two eye views of a nominal head pose (64 mm IPD, app render FOV - explicitly NOT headset-equivalent: no runtime distortion/FOV/reprojection) and writes eye-left/eye-right.ppm (18,446 vs 18,456 covered pixels = real parallax). VR-SESSION-READINESS.md in the Genesis worktree carries the exact Michael headset recipe and what he should see. Independent review findings fixed and re-verified: the assembleFrame hoist (the eyes differ only by camera), zero-coverage failure, the snprintf result check, directory creation, the %d cast, and the CLI arg guard; the captureDir-placement defect in the one-shot patch script is recorded (single declarations verified).
