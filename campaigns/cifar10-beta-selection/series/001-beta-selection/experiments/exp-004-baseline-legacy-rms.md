---
id: "exp-004"
title: "Repeat the focused CIFAR RMS/cosine sweep for baseline and legacy"
status: "complete"
hypotheses: ["H-003"]
---
# exp-004 — Baseline and legacy RMS/cosine sweeps

Filip explicitly requests repeating the focused tests for baseline and legacy.
Tests [H-003](../../../hypotheses/H-003-displacement-transfer-across-schemes.md).
Reuse [exp-003](exp-003-common-rms.md) as the ours comparator; do not rerun it.

Baseline V1/C1; legacy V4/C0.25, the native convention with
block_output_normalization=none in the latest saved legacy config. Ours used
V4/C1 with the same boundary convention. The same seed0 checkpoint tensors,
256 training images, batch32 BN statistics, no noise/augmentation, and
T/K=[6,6,4] are reused. The original ours and legacy initializers have all19
module and9 conductance tensors bitwise identical. Hold them fixed and change
only the amplification fields in the matched model config, including the head.
Labeled optimizer fields are unused: there are no parameter updates.

For each scheme/block, calibrate physical injected B separately to relative
output RMS targets [0.02,0.03,0.04,0.05,0.06,0.075,0.09,0.11,0.135,0.165,0.20,0.25]
on the same full256-example cohort. Warm-start the bounded log search from
ours' B at0.075, then the preceding measured response; guesses are not assumed
to transfer. Require target error<=3%, at most24 evaluations/target within the
job budget; preserve every search measurement. Reuse captured states/forces.
Then measure the centered EqProp gradients and output RMS from the same endpoints
against each scheme's own native and matched-local BPTT references.

Primary: worst Conv cosine in each block after averaging gradients across the
cohort, with >=0.95 high and >=0.99 stringent; also report per-batch variation,
absolute RMS, norms and dead responses. Test the0.075 target and shared observed
passing bands within/across schemes. Same native finite T/K is a fixed control,
not a claim that all schemes have reached equilibrium.

Result root: `results/cifar-block-rms-schemes-20260929-v1/`, separate baseline/
legacy bundles and immutable sources. Baseline on Loulou5090; legacy on Trex5090,
in parallel when weekday quota and approved sharing permit. Budget1200s per job,
2400s combined; expected5–10min per scheme. No smokes or retired solver gates.
Six focused CPU numerical tests pass after adding warm-start calibration.

Queue owns launch, monitoring, bounded operational recovery and CPU collection.
Completion requires12 targets x8 weights x8 batches per scheme, full calibration
histories, unchanged model/BN, local bundle/coverage validation and calibration-
versus-gradient replay displacement agreement. Preserve failed attempts and
report partial coverage. Config/job paths are recorded below before submission.

Prepared jobs: `cifar-rms-baseline-5090-20260929` (Loulou), `cifar-rms-legacy-5090-20260929` (Trex). Exact configs in `configs/cifar/block_rms_schemes_20260929/`; each scheme result folder contains immutable source identity, `launch/job.json` and CPU `analysis/collect.py`. Queue `status` is the routine inspection path. Target calibration and gradient progress update per case/block/batch; service owns monitoring and recovery. Deadline October1 08:00 Paris,1200s per job.

Completed and reviewed: both jobs and local collections complete; each has96 pooled/768 batch rows,36 calibrated targets and unchanged model/BN. Calibration and gradient replay RMS agree. No failures. [Reviewed findings](../results/exp-004-baseline-legacy-rms.md) contain the H-003 verdict, betas and all-scheme overlap.
