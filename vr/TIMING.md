# VR timing instrumentation

Status: measurement tooling built and reviewed; GPU timestamp runtime behaviour and actual VR performance NOT yet verified. First off1440 attempt stalled on invalid eye poses and ended without PASS; analyser correctly rejects it. Headset must be worn/tracked for captures.

Run alternating off/on captures with `bash vr/run-xr-scene.sh off 1440` (or on), preserving xr-output.txt and Proton/launcher logs separately for each run. Each complete run has600stereo submissions; discard first120submittedframes. Aim3captures/mode, alternatingorder, same720/1080/1440choice and similar scene/headmotion. Require launcherexit0, completePASS and retainedProtondiagnostics before accepting a capture.

`python3 vr/analyse-timing.py capture1.txt capture2.txt` emits per-captureJSON. Do not pool incomplete runs or compare changing resolutions as a neural off/on control.

## Measurements

8D3D12 GPU timestamp queries/frame,4per eye: startscene, endscene, endneural, endcompose/copy. Timestampfrequency fromdirectqueue. Resolvequeryreadback after allGPUwork; maponlyafter existingGPUfence. Sameinstrumentation off/on. Stereo stage sums and GPUspan fromfirsteye start tosecondeye finish. Neuralstage includesoutputtransition (tinyoffbaseline); GPUspan includesintereyegaps but excludesqueryresolve. Query/readback overhead exists, unbenchmarked. Noindividualeye stats emitted.

`frame_wall_ms`: postxrWaitFrame tocompletedxrEndFrame. IncludesCPUwork, swapchainwaits, serialisedGPUfencewait, queryreadback andruntimecalls; NOT CPUexecutioncost. `wait_ms`: walltimeinside xrWaitFrame. `cadence_ms`: successive submitted-framecompletion spacing; zeroif firstframe or no-submission/sessiongap. `period_ms`: XRpredictedDisplayPeriod; not measuredphysicaldisplayorlatency. Progressprintf occursaftermeasuredframewall interval but mayaffectnextcadence/wait. ExistingverboseProtonlogging retainedsamebothmodes, possibleoverhead. Untracked/nonsubmittedframes excluded,so no interpretation as whole-sessiondropcount.

Analyser validates exactly600sequentialrows andPASSmode,header,resolution,finite/nonnegative fields,positiveGPU/wall/displayperiod andsumstages<=GPUspan. Reportsmean,p50,p95,p99,max andsamplecountsafter120warmup; excludeszero cadence. Budgetexceedance andcadence>1.5period flags are indicators NOT confirmedcompositordrops. No encodertime, networklatency, displaylatency, reprojectionstatus ortrue deadline measured. Captureperiod meansdo notestablishheadsetlatency.

## Verification

All720/1080/1440XRbinariesstrictbuildpass. Synthetic600rowcapture/statistics pass; fiveinvalidcontrols (noPASS,nonfinite,stageoversum,mixedmode/resolution,nonsequentialframes) rejected; incompleteactualrun rejected. ReviewfixedmisleadingCPUlabel, progresslogging boundaries andsessioncadenceresets (direct STOPPINGreset handleseventdrain stop/start). RuntimeGPUtimingsstillpendingheadsetcapture.

## Runtime capture completed

Six1440captures now complete and validate; see TIMING-RESULTS.md and logs/timing/results.json. GPUqueries produced positive monotonic timestamps; all600rows/run present, first120excluded. Independent results reviewPASS with no execution/fullpooledp99recomputation by reviewer; parent computedpooledstatistics fromrawrows. Earlier blockedattempt was replaced by successfuloff-1capture. No remainingtestprocess; GPUresponsive.
