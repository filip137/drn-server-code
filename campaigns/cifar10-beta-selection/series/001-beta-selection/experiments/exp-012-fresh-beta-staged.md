---
id: "exp-012"
title: "Fresh beta tests with conditional one-epoch, five-epoch and ten-epoch stages"
status: "complete"
hypotheses: ["H-010"]
---
# Fresh beta transfer and conditional extensions

Completed September30: all eight fixed-beta candidates failed the epoch1 gate;
the predeclared fallback is exhausted. No epoch5/10 branches were eligible.
See the [reviewed results](../results/exp-012-fresh-beta-staged.md).

Filip approved fresh tests, continuation to5 then10 epochs if successful, and
further beta search otherwise. Current scope remains ours, seed0, no read noise.
Three initial cases: native BPTT, fixed0.3x and1x the exp-008 epoch10 beta vector
[0.09452322002142666,3.729986372379273,1.1695196708152935]. Use original initializer
tensors with fresh Adam, trainable BN/gains/readout, original LRs and50-epoch cosine
schedule, native T/K[6,6,4], batch32/eval16,45k/5k split and original augmentation.
Hybrid EqProp semantics are unchanged from exp-010/011. No solver audits or smokes.

Filip subsequently requested a full epoch for the initial screen. Each case
trains45,000 examples/1,407 updates. Save model/Adam/BN/RNG/scheduler at epoch1.
Stage5 continues epochs2–5; stage10 resumes epochs6–10. No restart or repeated
examples. Scheduler advances only at complete epochs. Measure full validation
at completed epochs; initial/1/5/10 EqProp cosine/RMS on the same256 diagnostic examples. Frozen
betas are not recalibrated between stages.

## Conditional decision rule

At epoch1 and epoch5, pass if validation accuracy>=matched native BPTT minus0.02
and validation CE<=1.05 times BPTT, with finite execution. This practical rule is
frozen before launch; it is not a statistical claim. Passing candidates and the
BPTT control continue. If neither initial candidate passes epoch1, or none passes
epoch5, launch six fresh one-epoch search cases at multipliers
[0.0001,0.001,0.01,0.03,0.1,3]. Reuse the matched BPTT boundary; do not rerun it.
Promote at most two passing search candidates, lowest validation CE first with
accuracy then multiplier as tie breakers. Apply the same epoch5 gate before10.
The wider search is conditional and executes at most once. If it yields no
passing candidate, retain the negative evidence and report the need for a changed
beta policy rather than running unbounded grids or silently changing science.

## Execution and monitoring handoff

Root `results/cifar-beta-fresh-staged-20260929-v1/`; readable cases in
`configs/cifar/beta_fresh_staged_20260929/cases.json`. Separate canonical bundles
`cells/{1,5,10}/CASE` preserve each stage. Frozen source/inputs identities,
initializers, split/cohort and checkpoint cursor checks are required.

Use two matching RTX3090 placements, local and nom-cool-1, sharing only as
authorized with Ben. Live inventory shows all5090s occupied by Kellian and below
our memory requirement. Reserve16GiB plus2GiB headroom per worker. Initial screen
allowance1.5GPU-hours/case, expected40–55min; each conditional5/10 continuation gets
4GPU-hours, expected3–4h on3090. Initial budget4.5GPU-hours; worst-case authorized
tree45.5GPU-hours including the six search screens and staged execution allowances.
Deadline October2 08:00 Paris. Attempts share their case-stage allowance; no fresh
retry budgets. Private verified cuDNN9.1 runtime on both3090s.

The shared queue owns monitoring. Completion collection validates each stage,
records the gate decision, transfers the exact next-stage checkpoint, and submits
only the predeclared passing continuations or search cases. No competing watch.
Operational failures preserve checkpoints and go to the queue's recovery handler;
they are not evidence of a bad beta. Incomplete stages never pass a scientific gate.
Verified numerical nonfinite failures are recorded as failed bundles with a
scientific-rejection artifact; validators certify that rejection evidence,
not completed training, and the candidate fails the gate. Native-control failure
blocks the comparison. Transport/runtime failures are not scientific rejections.
Collected summaries and JPG comparisons remain in `analysis/`; reviewed conclusions
belong in the campaign result note. Job IDs and startup verification follow here.

Implementation checks:5 staged CPU tests pass, including bitwise-equivalent
uninterrupted/resumed toy training with Adam, BN, sample order and RNG;24 decision
tests cover incomplete rounds, thresholds, rejections, deterministic search
selection, reference reuse and conditional extensions. Existing hybrid/short-run
tests were also reused. No training smoke or retired solver diagnostic.

Queue IDs `cifar-beta-fresh-e{1,5,10}-CASE-20260929` (case underscores become
hyphens). `experiments.run_cifar_beta_stages` supplies run/validate/collect; its
completion callback invokes the tested pure decision helper and submits only
prepared eligible jobs. `analysis/decision.json` owns the latest factual gate
state and links collected measurements; it does not replace scientific review.

Initial three stage1 jobs accepted: native BPTT running on nom-cool-1, beta0.3x
running locally, beta1x queued for the next slot. Verified live workers and fresh
queue heartbeat. The conditional callback is installed for each stage; no5/10
or fallback case is admitted before its recorded decision rule permits it.
Both first cases reproduced identical initial validation (10.04%, CE2.302585523).
Native BPTT reached optimizer updates; the beta arm completed its initial
gradient/RMS observation. Source/initializer checks passed on both hosts.

September29 follow-up: these3090 runs retain their original1/5/10 contract.
[exp-013](exp-013-all-scheme-long-eqprop.md) adds all three schemes on available5090s
with conditional extensions toward50 epochs and a shared read-only30-minute
summary covering both experiments. The persistent queue remains monitoring owner.

## Free-state BPTT control

The first collected EqProp candidate reached26.44% versus native BPTT37.06%.
Before attributing the entire gap to beta, add one fresh full-epoch `free_t_bptt`
control: exact autograd through the same reset freeT forward used by the hybrid
method, without replacing Conv gradients. Reuse the same initializer, freshAdam,
data order, BN, rates and T/K; no other change. This extends exp-011's diagnostic
comparison to initialization. It does not replace the native promotion reference.

Root `results/cifar-beta-fresh-free-t-control-20260929-v1/`, case `free_t_bptt`,
local3090, matching torch2.5.1+cu121 and private cuDNN9.1. One epoch only;
1.5GPU-hour cap, expected30–45minutes, original October2 deadline. Job
`cifar-beta-fresh-free-t-control-e1-20260929`. The queue collects and validates
without submitting continuations. No smokes. This uses the otherwise idle local
slot while beta1x continues on nom-cool-1. The conditional beta search stays intact.
Startup verified September29 22:53 Paris: queue reports a live local worker;
the shared monitor includes the control and reports no alerts.78 CPU tests pass
for control admission, staged continuation and comparison-only collection.

## Early-start recovery

Both initial beta cases failed the accuracy/CE gate, so the queue submitted the
six predeclared fallback screens. Local cases0.001x,0.01x and0.03x failed CUDA
initialization before any training: zero metric bytes and no checkpoints.
These are operational failures, not beta rejections. Subsequent0.1x initialized
successfully on the same local GPU;0.0001x is healthy on nom-cool-1. The underlying
transient driver failure is not yet explained; no GPU settings were changed.

Failed bundles are retained under the original root's
`failed_attempts/CASE/ATTEMPT_TOKEN/`. Replacement attempts use
`results/cifar-beta-fresh-staged-retry-20260929-v1/`, byte-identical scientific
source and inputs plus a separate collection helper, and job IDs ending`-retry1`.
Each has5,397seconds remaining after deducting its failed attempt from the
original5,400seconds; no budget reset. The helper validates the retry bundle,
copies it into the empty canonical case path, preserves retry provenance, and
reuses the original promotion policy and next-stage checkpoint paths.
`analysis/recovery.json` in the retry root maps every old/new job and archive.
Replacement jobs were accepted; old terminal attempts are cancelled in the queue
without deleting evidence.0.0001x and0.1x are live on the two3090s, and the held3x
case was released after local training progress resumed. The collector's five
CPU tests cover repeat collection, conflicting evidence and interrupted copies.
