---
id: "exp-025"
title: "Layerwise displacement and common block targets across schemes and checkpoints"
status: "complete"
hypotheses: ["H-019"]
---
# Can one output d/f target preserve every gradient in a block?

Filip authorizes repeating the per-layer measurement for baseline, ours and
legacy, and explicitly selects both checkpoint groups below. Record this as
[H-019](../../../hypotheses/H-019-common-layer-compatible-output-displacement.md).
The goal is to size beta through a block-output response that preserves all
internal-layer gradients. This is a read-only diagnostic; stopped training
queues remain stopped.

## Checkpoints and fixed conditions

- **EqProp group: epochs 0/1/5.** Use the completed fixed-beta trajectories:
  ours 0.3×, baseline 0.3× (original 5090 trajectory), legacy 0.1× (V100
  trajectory). Verify each epoch5 checkpoint's declared epoch1 parent hash.
  These are scheme-specific existing trajectories, not a matched training
  intervention; preserve their original runtime and beta provenance.
- **BPTT group: epochs 0/10/30/50.** Reuse the original checkpoints and configs
  from exp-005/006/008. Keep the two training groups separate in every
  aggregate, figure and recommendation. Share an initialization replay between
  groups only after checking identical initial model tensors and model contract.
- Same original 256 training images and order, eight batches of 32, no
  augmentation/noise/test-set access. L8 blocks [3,3,2], T/K [6,6,4], saved
  weights/BN, minibatch statistics with running buffers frozen, no optimizer.
- **Full float64 throughout**, as explicitly corrected by Filip following
  [exp-024](../results/exp-024-precision-response.md): model, analog states,
  dense head, BN, forcing, beta calibration, EqProp and both BPTT references.
  Promote saved checkpoint values and already preprocessed float32 images
  exactly; these are still the original training histories. Use exp-024's
  verified conversion and numerical source, with typed replay inputs and
  runtime dtype guards. RTX5090 or authorized RTX3090, with matching
  torch2.11.0+cu128/CUDA12.8 for every case; retain GPU identity.
  Legacy's V100 training runtime remains part of its provenance.
  Superseded float32 replays are excluded from the new analysis.

## Sweep and decision

For each block, calibrate beta on the 256-image cohort to output relative RMS
targets [0.000001,0.000003,0.00001,0.00003,0.0001,0.00025,0.0005,0.001,
0.0025,0.005,0.01,0.02,0.035,0.05,0.065,0.075,0.085,0.1,0.125,0.2,0.4],
using the existing bracket/log-bisection search, 3% tolerance and at most24
measurements per target. Warm-start from earlier measured beta/RMS when
available. One beta per block, shared by all its layers.

Record each convolution's own centered displacement
`d/f = RMS((state_plus-state_minus)/2) / RMS(state_free)`, absolute RMS,
unchanged endpoints, native/local BPTT cosine, norm ratio, gradient magnitude
and zero/near-zero fractions. Also record block output d/f, beta and loss force.
References and saved free states/forces are shared across beta candidates.
Pool gradients by example count and displacements by raw squared sums; retain
all eight minibatches. Undefined/dead gradients cannot pass.

Primary high-cosine criterion: **every layer >=0.95 pooled native cosine**.
Also report >=0.99 and the joint cosine>=0.95/norm0.9–1.1 criterion, plus
batch minima/medians and number of batches passing. These are descriptive
thresholds, not paper gates. Find sampled passing bands and intersections:
within a checkpoint, across epochs within a scheme, then across schemes within
each training group. Include trained-only and initialization-inclusive results.
When no shared target passes, report the limiting layers/checkpoints and the
best worst-case compromise instead of prescribing a universal target.
Report actual achieved output RMS; grid gaps are not certified continuous bands.

## Execution and monitoring handoff

October1 analysis follow-up: Filip requests inspection of the saved matched
block-only BPTT comparisons. Reuse all18 completed float64 artifacts; run
`experiments/analyze_cifar_layer_rms_transfer.py --reference local_bptt` on CPU.
Outputs are in the existing root's `analysis/local_bptt/`. Report the same
layer/cohort/threshold scopes separately from the original full-network
comparison; see the updated result note. No new simulations are needed.
The subsequent requested >0.95 trained-checkpoint target/beta proposal is
recorded in that note and `analysis/local_bptt/trained_target_proposal.json`,
with45 checkpoint/block endpoint pairs in `trained_target_beta_brackets.csv`.

Active root: `results/cifar-layer-rms-all-schemes-float64-20260930-v1/`.
Configs: `configs/cifar/layer_rms_all_schemes_float64_20260930/`.
Eighteen unique replays: three shared initializations, six EqProp trained
checkpoints and nine BPTT trained checkpoints, giving21 labelled views.
All initial model tensors match between groups; all EqProp epoch5 parent
hashes match the selected epoch1 checkpoints. Reuse the verified input
inventory and original256-image cohort. Expected coverage per case:
168 pooled layer rows and1344 minibatch rows.

Use Fifi/Loulou RTX5090s through the shared queue, at most two GPUs during
weekday daytime. Reserve22GiB plus2GiB headroom per worker (exp-024 measured
21.51GiB peak). Expected20–40minutes per checkpoint, roughly3–6hours wall
on two GPUs; cap3600s per case including validation,64800GPU-seconds total
including retries for the original5090 allocation. Runner cap3450s leaves validation margin.
The authorized3090 expansion below revises the overall cap to72000s (20GPU-hours). Refine timing
from actual production progress. Deadline October1 08:00 Paris. No smokes
or extra solver qualification. The higher precision requires this revised
budget; charge the superseded float32 attempts separately.

The shared queue owns routine monitoring, placement and CPU collection.
Root owns scientific review. Jobs use
`cifar-layer-rms-fp64-{case_id}-20260930`; exact cases, commands and receipts
are in `cases.json`, `launch/` and `analysis/queue-submission.json`.
Inspect with installed `gpu_queue.py status JOB_ID`. No automatic scientific
retry, retuning or training. Stop on nonfinite values, mixed precision,
identity mismatch, changed checkpoint/model/BN or budget exhaustion.

Validate source/cohort/checkpoint identity, float64 states/forcing/gradients,
full layer/batch coverage, target attainment and pooled statistics. The CPU
collector `analysis/collect.py CASE_ID` follows actual queue placement.
The last successful collector invokes
`experiments/analyze_cifar_layer_rms_transfer.py --root RESULTS_ROOT` and
writes an analysis-complete marker only after all tables/JPGs finish.
`--allow-partial` writes explicitly incomplete intermediate analysis separately.
Both collector and analysis reject float32 artifacts. Keep EqProp/BPTT
groups separate in every figure and common-target intersection. The result
note will be `../results/exp-025-all-scheme-layer-rms.md`.

Superseded root: `results/cifar-layer-rms-all-schemes-20260930-v1/`, original
configs `configs/cifar/layer_rms_all_schemes_20260930/`, queue IDs
`cifar-layer-rms-{case_id}-20260930`. Filip corrected precision during the
initialization replays. All remaining float32 admissions were cancelled and
active workers stopped; preserve their partial/completed artifacts and
terminal queue receipts as superseded evidence. All18 queue entries are
cancelled; three launched attempts consumed939.24GPU-seconds in total.
Their artifacts and terminal receipts are collected locally. None enters the float64
study. Its original source identity was
`6e504e15e9605f30d7114e6216cc176a2c604ed1057918a12a5aa91ccee37d9b`.

Current state: **complete and reviewed**. All18 unique cases are locally
collected and validated; the full analysis covers3024 pooled/24192 minibatch
rows and21 labelled group views. Every declared scientific case is present.
Final queue completion was September30 21:45 Paris; root completed review on
October1. Charged float64-study time including the early3090 cancellation is
48790.70s, below72000s. No assigned jobs remain running or queued. Full outputs
are under `analysis/`, with `analysis-complete.json`; earlier `analysis/partial/`
remains explicitly partial. See the [reviewed result](../results/exp-025-all-scheme-layer-rms.md)
for scoped shared targets, the EqProp block3 exception and next diagnostic.
No further run or training is launched by this review.


Parallelism update (September30, 17:40 Paris): Filip asks to add the local
and nom-cool-1 RTX3090s. The first two5090 cases completed in2000/2062s;
both measured20850MiB peak reserved memory and passed local collection and
partial numerical/coverage analysis. Their queue charge totals4066.31s.
Transfer the unstarted EqProp legacy/ours epoch5 cases to local/nom-cool-1,
respectively. Preserve all scientific settings and batch32/256 images. An
isolated copy of the working5090 runtime provides identical torch/CUDA/NumPy
versions without altering installed environments. Reserve21504MiB plus1024MiB
headroom; admission rechecks memory and ownership and preserves Ben's work.

The first local3090 attempt used an85-minute cap. Measured block1 calibration
steps took about47s versus15.8s on5090, making that limit too short. Root stopped
it early after572.92s, preserved its output in
`superseded/eqprop-legacy-e5-3090-short-budget/` and its receipt in
`launch/eqprop-legacy-e5-3090-short-budget-terminal.json`, then restarted with
7050s runner/7200s queue allowance. This is an operational replacement, not
another scientific case. Nom-cool-1 receives the corrected limit before launch.
No scientific outcome is taken from the cancelled attempt.

Current replacement IDs are
`cifar-layer-rms-fp64-3090-long-eqprop-{legacy,ours}-e5-20260930`;
commands: `launch/eqprop-{legacy,ours}-e5-3090-long.json`. Both original5090
queue entries were cancelled with zero attempts. `source3090-long/` preserves
identical numerical code and changes only these operational time limits;
source identity SHA256:
`eaedfe9230d37a59f2112ae13c35b8f8126e2cd608d785e550c897e18560c35c`.
Older `source3090/` and short-limit launch files remain as provenance.
The revised ceiling is72000GPU-seconds (20GPU-hours), including the cancelled
3090 attempt and all new retries/validation; the current maximal accounting is
4066.31 +572.92 +14*3600 +2*7200 =69439.23s. Deadline unchanged. Expect roughly
90–110minutes per3090 case; further redistribution requires actual throughput
and remaining-budget review. The two existing5090 workers continue normally.

The collector follows all four authorized targets and copies replacements from
`cells3090/` into canonical `cells/` only after completion. Analysis retains
GPU/runtime provenance and rejects float32. Both replacement jobs completed and passed collection/analysis; their
terminal source/GPU identities and timing remain in the queue and bundles. The existing queue remains the
sole monitor/collector; no separate polling agent is installed.
