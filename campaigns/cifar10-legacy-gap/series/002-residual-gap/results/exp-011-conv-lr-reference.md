---
experiment: "exp-011"
evidence: "validated-local"
summary: "Neither scheme meets the local oversized-LR criterion; legacy's mean effect reverses on the fresh cohort, with strong minibatch and BN-mode dependence."
verdicts: {}
---
# Convolution LR reference

All eight directions and 64 measurements collected and validated. Batch size 16, original
epoch 30 legacy/ours; four common gradient minibatches and one fresh 256-example training
cohort. Original checkpoints and BN buffers unchanged. Non-convolution parameter
updates are byte-identical across every multiplier. No validation/test data read.

Root: `results/cifar10-conv-lr-reference-20260927-v1/`. Native evidence/config/environment:
`probe/manifest.json`, `probe/result.json`; arithmetic: `analysis/summary.json`;
figure: `analysis/conv_lr_response.jpg`.

Positive delta below means higher separate-cohort CE than the zero-convolution step,
with BN/gain/classifier updates held fixed. Counts use the frozen tolerance.

| BN | Scheme | LR multiplier | Mean CE delta | Beats zero | Worse than zero |
|---|---|---:|---:|---:|---:|
| minibatch | legacy | 0 | +0.00000000 | 0/4 | 0/4 |
| minibatch | legacy | 0.03125 | -0.00002158 | 2/4 | 0/4 |
| minibatch | legacy | 0.125 | -0.00008149 | 2/4 | 0/4 |
| minibatch | legacy | 0.25 | -0.00015696 | 2/4 | 0/4 |
| minibatch | legacy | 0.5 | -0.00030364 | 2/4 | 1/4 |
| minibatch | legacy | 1 | -0.00049233 | 2/4 | 2/4 |
| minibatch | ours | 0 | +0.00000000 | 0/4 | 0/4 |
| minibatch | ours | 0.03125 | -0.00002477 | 1/4 | 0/4 |
| minibatch | ours | 0.125 | -0.00009862 | 3/4 | 1/4 |
| minibatch | ours | 0.25 | -0.00019228 | 3/4 | 1/4 |
| minibatch | ours | 0.5 | -0.00035380 | 3/4 | 1/4 |
| minibatch | ours | 1 | -0.00047725 | 1/4 | 3/4 |
| saved | legacy | 0 | +0.00000000 | 0/4 | 0/4 |
| saved | legacy | 0.03125 | -0.00002600 | 3/4 | 0/4 |
| saved | legacy | 0.125 | -0.00010200 | 3/4 | 0/4 |
| saved | legacy | 0.25 | -0.00017908 | 3/4 | 1/4 |
| saved | legacy | 0.5 | -0.00029035 | 3/4 | 1/4 |
| saved | legacy | 1 | -0.00039117 | 1/4 | 3/4 |
| saved | ours | 0 | +0.00000000 | 0/4 | 0/4 |
| saved | ours | 0.03125 | +0.00002406 | 1/4 | 3/4 |
| saved | ours | 0.125 | +0.00010070 | 1/4 | 3/4 |
| saved | ours | 0.25 | +0.00022337 | 1/4 | 3/4 |
| saved | ours | 0.5 | +0.00049909 | 1/4 | 3/4 |
| saved | ours | 1 | +0.00115089 | 1/4 | 3/4 |

Predeclared classifications (exploratory):

- minibatch, legacy: mixed_or_unresolved; qualifying interior multipliers [].
- minibatch, ours: mixed_or_unresolved; qualifying interior multipliers [].
- saved, legacy: mixed_or_unresolved; qualifying interior multipliers [].
- saved, ours: adverse_down_to_1_32; qualifying interior multipliers [].

These classifications apply only to the sampled LR grid, checkpoint, cohort and
batch size. Four directions share a cohort and one seed. Finite-step harm does not
prove an infinitesimal derivative; a sampled interior benefit is not a confirmed
training LR or explanation of the final accuracy gap. Every response and direction
is retained; no single minibatch or best multiplier substitutes for the frozen rule.

Review: under primary minibatch BN, the full convolution LR gives the lowest mean
CE on the sampled grid for both schemes (legacy -0.00049233; ours -0.00047725,
relative to freezing convolution updates). However, it helps only 2/4 legacy
directions and 1/4 ours directions; one strongly beneficial direction dominates
each mean. No interior multiplier beats both zero and full LR consistently enough
to meet the predeclared oversized-LR criterion in either BN mode.

The legacy full-versus-zero convolution effect changes from +0.00168990 on the
exp-010 cohort to -0.00049233 on this fresh cohort. The corresponding original,
full-update and zero-convolution gradient-cohort CE controls match exp-010 exactly
in both BN modes (maximum absolute difference 0). The earlier observed harm must
therefore remain a cohort-specific finding, rather than evidence of a generally
harmful legacy convolution update. Saved BN further changes the comparison:
legacy's mean full-step effect is -0.00039117, while ours is +0.00115089.

This does not support lowering legacy's convolution LR specifically. A useful next
diagnostic would average the matched comparisons across larger or multiple fresh
evaluation cohorts before interpreting an LR intervention. These are local results
at one checkpoint and seed, not an explanation of the final accuracy gap.
No sustained training or additional follow-up launched.
