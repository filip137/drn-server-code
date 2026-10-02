---
id: "exp-011"
title: "Compare convolution learning-rate response for legacy and ours"
status: "complete"
hypotheses: []
---
# Convolution LR response with ours as reference

Filip requested an ours reference for the proposed LR diagnostic after
[exp-010](../results/exp-010-update-family-isolation.md). Test whether a smaller
positive convolution step beats both zero and the full step, or remains adverse
down to the smallest tested step. Exploratory local comparison, not LR selection
for a full training run or proof of the historical accuracy gap.

Original seed-0 legacy/ours epoch-30 checkpoints and saved Adam state. Four
gradient batches of 16 per scheme: split-training indices 0:64, as in exp-010.
Fresh separate training cohort: indices 512:768, 256 examples, also evaluated in
batches of 16. Disjoint from gradients and both preceding diagnostics' 256:512
probe cohort. No augmentation, validation-set or official-test access.

Convolution LR multipliers: 0, 1/32, 1/8, 1/4, 1/2, 1, relative to each scheme's
saved vector. Restore the same checkpoint/Adam/gradient per variant, multiply only
convolution-group LRs, perform the actual Adam step, then apply normal weight
projection. Do not interpolate already projected updates. BN affine, gain and
classifier parameter updates must exactly match the full step. Running BN buffers
remain at saved values. Original/repeated-original controls give 64 measurements
across eight directions. Record actual relative convolution movement as context.

Primary: separate-cohort CE under minibatch BN, relative to the zero-convolution
step (other groups updated). Secondary: saved-stat BN and gradient-minibatch CE.
Use tolerance max(0.00001 CE, ten times repeated-original discrepancy). Report all
curves and per-batch signs. Local oversize evidence requires the SAME interior
multiplier to beat zero and full by tolerance in at least 3/4 directions, with
mean improvements above tolerance. Report all qualifying multipliers, without
claiming a confirmed optimum. Adverse response down to the smallest tested step
requires a mean loss increase above tolerance at every positive multiplier and
an increase above tolerance in at least 3/4 directions at 1/32. Otherwise mixed
or sub-tolerance responses remain unresolved. Compare these outcomes for both
schemes and BN modes; do not mistake one mean driven by one batch for consistency.
This finite grid cannot establish an infinitesimal derivative or long-run LR.

Execution root registered before creation:
`results/cifar10-conv-lr-reference-20260927-v1/` (inputs, source, smoke, probe,
analysis). Reuse `experiments/probe_cifar_l8_update_families.py` with explicit
`--probe-kind conv-lr --batch-size 16 --holdout-start 512`.
Akib RTX3080, 8,192 MiB bounded process memory plus 1,024 MiB headroom, allocator
fraction 0.70. Expected 10 minutes, cap 30 GPU minutes including smoke/validation;
5-minute CPU collection allowance; cutoff September 28 08:00 Paris. No automatic
retries or training continuation. Shared queue owns launch/monitor/collection;
the coordinating agent reviews the resulting curves and scope.
Prepared command and identities: execution-root `launch/job.json`,
`source/source_identity.json`, `inputs/checkpoints.json`. Queue job
`cifar-conv-lr-reference-20260927`; Akib is the sole placement. Collection runs
`analysis/review.py` for exact coverage/cohort validation, full response curves,
predeclared classifications, a JPG and the campaign result note. No new work is
launched by analysis. Source compiled; inputs and native evaluation reuse exp-010.
Current handoff: all cases collected and validated; predeclared LR-response rules
applied to both schemes and BN modes. See [result](../results/exp-011-conv-lr-reference.md).
