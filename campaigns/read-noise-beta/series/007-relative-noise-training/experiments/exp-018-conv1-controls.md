---
id: "exp-018"
title: "Conv1 clean three-seed controls and baseline relative-noise training"
status: "complete"
hypotheses: ["H-011"]
---
# Exp018 — Conv1 controls on the3090s

Filip requested Conv1 allthree schemes,threeseeds no-noise plusoneseed noisy
baseline, scheduledonthe3090s. Interpret consistentwithlatestcontrols:
9cleanruns (baseline/ours/legacy × seeds0/1/2), plus5baseline seed0noise runs
ateta1e-6,1e-5,1e-4,3e-4,1e-3.10uninterruptedepochs frominitialization.14total.
ReferenceinitialoutputD/F=1, aspriorConv1relative-noise training. Freezeaccepted
seed0beta foreachschemeacrossseeds,actualD/Fmaydifferseeds1/2. Sameweightseedand
shufflewithinmatchedschemecomparisons, splitseed0; trueindependentweightdraws.

Useexp014 numericalcontract, scheme-specificsourceAdamLRvectors, T=K4,
inputgain40,float64,explicitperfectdiodes,weights[0,100],frozenzerobiases,
batch16/validation64,ordinaryMNIST55k/5k, officialtestdisabled. RelativeGaussian
noiseeta*abs(cleanendpoint)*xi atevery noninputlayer withsameeta,independentdraws,
noabsolutefloor,cleaninputs/relaxation/force/validation. Originalacquisitionseed
2026081601. No newLR/TK/calibrationsweeps, reuseacceptedmatchingproof.

Primaryfixedepoch10cleanmeans/samplestdandpairedseedgapsforallthree;baseline
noiseepoch10compareswitholdConv1ours/legacy10epochstudy. Allnewnoisybaseline runs matchtheold10epoch duration.
No significanceclaimfrom3seeds orsingle-noiseseed. Saveepoch10metrics,
model+optimizercheckpointsateveryepoch.

Resultrootregisteredbeforecreation:
`results/conv1-clean-and-baseline-noise-20260927-v1/`.
Configs:`configs/conv/conv1_clean_baseline_noise_20260927/`.

## Execution and monitoring handoff

OnlylocalRTX3090andnom-cool-1RTX3090,peruser. Useexistingqueues towaitforcurrent
assigned3090work;no interference orCPUfallback.14runs,expected1–2GPUh,
allowance4GPUhincludingoverhead/retries,72hdeadlinefromqueueadmission.
Wallcompletionincludesexistinghostqueues. SameBen-sharingpolicyandlivecapacity
checks. Frozenexp015v2numericalsource,configs/initidentityseparate. Userstanding
no-tests/no-canariesinstructionapplies;retainCUDA/finiteprogressandinputidentity.
Onepersistentwatchownsnew14cases,collectlocallyandreconcilecoveragebefore
resultnote/ledgerreview. Existingruns/budgets/watchersremainunchanged.

Filip explicitly corrected the duration to10epochs for every newConv1case.

Placementrefinement:all14queuedonnom-cool-1RTX3090,whichisexpectedfree21:30versuslocal04:00. MeasuredConv1previous10epochreceiptmean8.506min,14case~1.985GPUh. Keepingallcasesonthefirstfree3090avoidsdelayinghalfonlocal. No5090/3080placement.

### Scheduled ten-epoch execution

All14configs ready/staged. Portable3initializerseedassetsareindependentand
matchedacrossall3schemes;seed0matchespreviousConv1runs. Acceptedseed0basebetas:
baseline4.019442283658996,ours0.8076502505643521,legacy0.05762054077568885.
OriginalschemeLRvectors preserved;no calibrationperformed.

Nom-cool-1RTX3090controller3819899 ownsall14cases; exactcommand/pathin
transport/launches.json. Itwaitsfor exp017nom1queuefinish beforeliveGPUadmission.
4GPUhcap,existingworkuntouched. NumericalsourceSHA256
`f337f482deb8a14614f5e8b8197613a06e365891d8d4ee2069810be29ffb8477`;
config/initializerproofinpreparation.json.

Harddeadline 2026-09-30T18:24:17.484226+00:00.

Finalanalysis `python -m experiments.analyze_conv1_controls --study-root results/conv1-clean-and-baseline-noise-20260927-v1` usingpy312 aftercollection. Expected14epoch10outcomes,3seedcleanmeans/sampleSD,noisybaselinecomparisonwitholdourslegacyepoch10,JPG.

### Reviewed completion — 2026-09-27

All14 ten-epoch cases collected and validated; 5795.243729 charged seconds /14400. No failures, exclusions or retries. See [reviewed result](../results/exp-018-conv1-controls.md), study `analysis/summary.json`, `analysis/identity_coverage.json`, and `monitor/FINAL_SUMMARY.md`. Monitoring may close; no further runs.
