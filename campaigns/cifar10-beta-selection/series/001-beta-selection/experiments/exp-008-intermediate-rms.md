---
id: "exp-008"
title: "Measure epoch10/30 RMS curves and recommend targets across training"
status: "complete"
hypotheses: ["H-006"]
---
# Intermediate checkpoint RMS curves

Filip requests the verified epoch10 and30 curves for baseline/ours/legacy and
an RMS recommendation. Replay the six original final checkpoints inventoried in
`results/cifar-block2-nudged-k-e50-20260929-v1/analysis/intermediate_checkpoints.json`.
Use each saved config and exact checkpoint; verify epochs and hashes against that
inventory. Original trajectory identity and initial tensor agreement are established
in exp-006. Reuse its unchanged validated numerical runner and epoch0/50 evidence.

Same256 training-image indices/order, preprocessing, batch32 with minibatch BN
statistics and frozen buffers, no noise/augmentation, native endpoint convention,
freeT and nudged/native/local K=[6,6,4]. No training, optimizer updates or smokes.
Per block/checkpoint, calibrate physicalB to nominal relativeRMS targets
`[0.0005,0.001,0.0025,0.005,0.01,0.02,0.035,0.05,0.065,0.075,0.085,0.10,0.125,0.20,0.40]`
within3%, at most24 evaluations per target. Warm-start from the measured epoch50
B atR≈0.005; never assume a beta itself transfers between checkpoints.

Primary score is minimum pooled native-BPTT cosine across a block's Conv weights.
Report per-layer cosine/norm/sparsity, matched-local reference, and minibatch spread.
Test H-006's<=0.01 regret for each epoch50 target at epochs10/30 against the new
sampled peaks. Recommend per-scheme/block fixed targets by maximizing the worst
checkpoint score: first across trained epochs10/30/50, then across0/10/30/50 using
only nominal targets measured at all included epochs. Report per-epoch optima and
the tradeoff, rather than inventing an unmeasured continuous schedule. Use the lower
target only for exact score ties. Flag boundary maxima and poor minibatch minima;
no selection here establishes EqProp training accuracy or a universal depth rule.

## Execution and monitoring handoff

Root `results/cifar-block-rms-intermediate-20260929-v1/`, with `e10/{scheme}/` and
`e30/{scheme}/` bundles, frozen inputs/source and CPU collectors. Configs:
`configs/cifar/block_rms_intermediate_20260929/e{epoch}_{scheme}.json`.
All epoch10 jobs use Loulou5090 sequentially; epoch30 jobs use Trex5090 sequentially.
Queue IDs `cifar-rms-e{10,30}-{ours,legacy,baseline}-5090-20260929`.
Expected3–6min/job;600s hard allowance/job,3600s total including attempts;
deadline October1 08:00 Paris. Queue owns admission, monitoring and collection.
Respect two-GPU daytime cap, measured memory/headroom and authorized Ben sharing.

Validate120 pooled/960 batch rows and45 calibration targets per checkpoint,
matching calibration/replay RMS, exact epoch/checkpoint hashes and unchanged
model/BN/source checkpoints. Stop and preserve failures on nonfinite values,
unattainable targets or exhausted allowance; no scientific retuning. Inspect
with installed `gpu_queue.py status JOB_ID`; collector is each bundle's
`analysis/collect.py`. All six jobs/collections complete and locally validated:
720 pooled/5760 batch rows,270 calibrated block targets; no failures/exclusions,
1535.5s combined charged time. All checkpoints/model/BN unchanged, GPUs released.
[Reviewed recommendations](../results/exp-008-intermediate-rms.md) record H-006's
scoped contradiction, selected RMS values, initialization grid gaps and minibatch
limitations. No training or additional checkpoint replays launched.
