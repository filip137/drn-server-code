---
experiment: "exp-012"
evidence: "validated-local"
summary: "Across 1,024 fresh examples, legacy has a weak mixed local response while ours has adverse convolution steps; neither meets the stable oversized-LR criterion."
verdicts: {}
---
# Convolution LR response across evaluation cohorts

Eight directions, 64 variant records and all four 256-example cohorts validated.
Root: `results/cifar10-conv-lr-cohorts-20260927-v1/`; native bundle: `probe/`;
full arithmetic: `analysis/summary.json`; figure: `analysis/conv_lr_cohorts.jpg`.
Gradient-cohort CE controls versus exp-011: maximum difference 0.

CE changes below are relative to zero convolution LR, with all other parameter
updates fixed. Negative means improvement; four gradient directions share each cohort.

| BN | Scheme | LR multiplier | Pooled CE delta | Pooled directions helped | Cohort mean deltas |
|---|---|---:|---:|---:|---|
| minibatch | legacy | 0 | +0.00000000 | 0/4 | +0.0000000, +0.0000000, +0.0000000, +0.0000000 |
| minibatch | legacy | 0.03125 | -0.00001169 | 3/4 | -0.0000273, -0.0000076, -0.0000204, +0.0000084 |
| minibatch | legacy | 0.125 | -0.00004116 | 3/4 | -0.0001049, -0.0000167, -0.0000831, +0.0000402 |
| minibatch | legacy | 0.25 | -0.00007414 | 3/4 | -0.0001909, -0.0000115, -0.0001773, +0.0000831 |
| minibatch | legacy | 0.5 | -0.00009791 | 3/4 | -0.0002865, +0.0000261, -0.0003261, +0.0001949 |
| minibatch | legacy | 1 | -0.00002243 | 3/4 | -0.0003451, +0.0001843, -0.0005004, +0.0005715 |
| minibatch | ours | 0 | +0.00000000 | 0/4 | +0.0000000, +0.0000000, +0.0000000, +0.0000000 |
| minibatch | ours | 0.03125 | +0.00003454 | 0/4 | +0.0000251, +0.0000556, +0.0000096, +0.0000479 |
| minibatch | ours | 0.125 | +0.00014233 | 0/4 | +0.0001054, +0.0002256, +0.0000294, +0.0002089 |
| minibatch | ours | 0.25 | +0.00028781 | 1/4 | +0.0002040, +0.0004636, +0.0000359, +0.0004477 |
| minibatch | ours | 0.5 | +0.00060699 | 0/4 | +0.0004199, +0.0009788, +0.0000649, +0.0009643 |
| minibatch | ours | 1 | +0.00131396 | 0/4 | +0.0008910, +0.0021252, +0.0003066, +0.0019329 |
| saved | legacy | 0 | +0.00000000 | 0/4 | +0.0000000, +0.0000000, +0.0000000, +0.0000000 |
| saved | legacy | 0.03125 | -0.00002245 | 4/4 | -0.0000486, -0.0000218, -0.0000107, -0.0000087 |
| saved | legacy | 0.125 | -0.00008132 | 4/4 | -0.0001868, -0.0000803, -0.0000355, -0.0000227 |
| saved | legacy | 0.25 | -0.00014803 | 4/4 | -0.0003622, -0.0001369, -0.0000586, -0.0000344 |
| saved | legacy | 0.5 | -0.00022665 | 4/4 | -0.0006485, -0.0001631, -0.0000706, -0.0000244 |
| saved | legacy | 1 | -0.00023080 | 3/4 | -0.0010783, -0.0000147, +0.0000303, +0.0001394 |
| saved | ours | 0 | +0.00000000 | 0/4 | +0.0000000, +0.0000000, +0.0000000, +0.0000000 |
| saved | ours | 0.03125 | +0.00007208 | 0/4 | +0.0000714, +0.0000913, +0.0000189, +0.0001067 |
| saved | ours | 0.125 | +0.00029086 | 0/4 | +0.0002917, +0.0003667, +0.0000775, +0.0004276 |
| saved | ours | 0.25 | +0.00058541 | 0/4 | +0.0005893, +0.0007392, +0.0001530, +0.0008601 |
| saved | ours | 0.5 | +0.00120972 | 0/4 | +0.0012126, +0.0015113, +0.0002794, +0.0018356 |
| saved | ours | 1 | +0.00255180 | 0/4 | +0.0024921, +0.0032154, +0.0006232, +0.0038765 |

Predeclared rule outputs (exploratory):

- minibatch, legacy: pooled mixed_or_unresolved; pooled qualifying multipliers []; cohort qualifiers [[], [0.25], [], []]; stable multipliers [].
- minibatch, ours: pooled adverse_down_to_1_32; pooled qualifying multipliers []; cohort qualifiers [[], [], [], []]; stable multipliers [].
- saved, legacy: pooled mixed_or_unresolved; pooled qualifying multipliers []; cohort qualifiers [[], [0.125, 0.25], [], []]; stable multipliers [].
- saved, ours: pooled adverse_down_to_1_32; pooled qualifying multipliers []; cohort qualifiers [[], [], [], []]; stable multipliers [].

Review: no multiplier satisfies the stable oversized-LR criterion for either
scheme in either BN mode. Legacy's primary pooled response is weakly beneficial:
full LR changes CE by -0.00002243 and half LR by -0.00009791. Half LR beats both
zero and full above tolerance for only 2/4 directions; its lower pooled mean
therefore does not meet the declared consistency rule. Only one of four cohorts
individually supports an interior LR for legacy. Full-step cohort means split
2 beneficial / 2 harmful, so the nearly zero pooled mean hides cancellation.

Ours has a different response: the full step increases primary pooled CE by
+0.00131396, with increases above tolerance in all four cohort means and all four
pooled gradient directions. Every positive multiplier has positive pooled mean
CE change, including +0.00003454 at 1/32; the smallest step is adverse above
tolerance for 3/4 directions. Three of four cohorts meet the adverse-down-to-1/32
rule under minibatch BN; all four do under saved BN. Reducing LR mitigates this
sampled harm, but no positive multiplier gives a mean benefit over zero. This
finite grid does not establish an infinitesimal derivative or isolate gradient,
Adam-momentum and interactions with the fixed other-group updates.

The larger cohort removes the strongly beneficial mean observed for ours in
exp-011 and leaves legacy approximately neutral under minibatch BN. The unchanged
gradient-cohort controls (maximum CE difference 0) support interpreting this as
evaluation-sampling sensitivity, rather than a changed update setup. The local
comparison now favors legacy despite its worse historical training performance;
these particular epoch-30 steps do not explain the final accuracy ranking.

Decision: do not lower legacy's training LR on this evidence. Any further causal
claim needs independent gradient batches and checkpoint coverage, rather than
more LR multipliers for these same four directions. One seed/checkpoint and four
fixed gradient batches remain the scope; the 16 cohort/direction combinations are
not independent replicates. No follow-up launched. Execution and validation used
32.6 GPU minutes including smoke; native production took 32.3 minutes, with
5,758 MiB peak reserved memory. All declared cases collected; no failures.
