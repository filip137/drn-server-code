---
id: "exp-012"
title: "Convolution LR response across fresh evaluation cohorts"
status: "complete"
hypotheses: []
---
# Convolution LR response across evaluation cohorts

Filip accepted the follow-up to [exp-011](../results/exp-011-conv-lr-reference.md):
check whether its convolution LR comparison survives more evaluation examples.
Reuse the original seed-0 epoch-30 legacy/ours checkpoints, saved Adam state,
four gradient batches of 16 (training indices 0:64), actual convolution LR
multipliers 0, 1/32, 1/8, 1/4, 1/2, 1, and all exp-011 identity controls.
Only the evaluation cohort changes: training indices 768:1792, split into four
consecutive, disjoint cohorts of 256, each evaluated at batch size 16. These
examples are disjoint from both earlier evaluation cohorts and the gradients.
No augmentation, validation/test access, new gradients or sustained training.

Primary: minibatch-BN CE change versus zero convolution LR, with other parameter
updates fixed. Report the pooled 1,024-example response, every 256-example cohort,
and all four gradient directions. Saved-stat BN is secondary. Reuse exp-011's
tolerance max(1e-5 CE, ten times original/repeated-original discrepancy), including
blockwise discrepancies. Its oversize rule (same interior multiplier beats zero
and full by tolerance for at least 3/4 directions, with mean improvements above
tolerance) is applied to each cohort and the pool. Call the evidence stable across
cohorts only if the same multiplier qualifies in the pool and at least 3/4 cohorts.
Also retain exp-011's adverse-down-to-1/32 rule for each cohort and the pool.
Report full-step mean signs and per-direction consistency without treating the
16 direction/cohort cells as independent replicates. No best-cohort selection.
This tests evaluation-sampling sensitivity, not gradient-sampling, seed or
checkpoint generality; it cannot identify a training-optimal LR or explain the
final accuracy gap by itself. These added stability rules are exploratory and
frozen before this run.

Execution root: `results/cifar10-conv-lr-cohorts-20260927-v1/`, including inputs,
frozen source, smoke, probe and analysis. Reuse the native runner with
`--probe-kind conv-lr --batch-size 16 --holdout-start 768 --holdout-count 1024`.
Eight directions, 64 variant records, each with four evaluation-cohort metrics
per BN mode. Expected 30 minutes on Akib RTX 3080; 45-minute total GPU cap including
smoke/validation, five-minute collection allowance, cutoff September 28 08:00
Paris. Same 8,192 MiB peak bound plus 1,024 MiB headroom, allocator fraction 0.70.
Stop on numerical, identity, coverage or deadline failure; no automatic retries.

Execution and monitoring handoff: shared run-watch queue is the sole owner;
job `cifar-conv-lr-cohorts-20260927`. Exact command, placement and collection are
in execution-root `launch/job.json`; source/input identities in
`source/source_identity.json` and `inputs/checkpoints.json`. Inspect with
`gpu_queue.py status cifar-conv-lr-cohorts-20260927`; collection runs the frozen
`source/queue_worker.py collect`, validating native bundles then executing
`analysis/review.py`. The coordinating agent owns scientific review and any
failure diagnosis. Current handoff: all cases collected, validated and scientifically reviewed.
Queue attempt `b1888edf00a940439862407b6c696031` completed within budget
(1,953.84 seconds including smoke/target validation). See
[result](../results/exp-012-conv-lr-cohorts.md); no further run launched.
