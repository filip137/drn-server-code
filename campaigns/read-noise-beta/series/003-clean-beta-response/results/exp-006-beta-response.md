---
experiment: "exp-006"
evidence: "validated-local"
summary: "All114 clean beta replay cells validate; centered response is nearly linear and legacy has larger amplitude, without a general steeper decay."
verdicts: {"H-005": "inconclusive"}
---
# Clean Conv3 beta response

All114/114 local RTX3090 cells completed;16,416 state and16,416 gradient
rows validate, with no failures/exclusions/retries. Six smoke cells and
production consumed2.6275GPUh of the4GPUh cap. Fixed T=K8, nativefloat64,
shared initialization/clean EqProp epoch30, nineteen betas1e-6–1e3,
576 matched ordinary-MNIST validation examples; no optimizer/test/TK sweep.

Centered D is approximately proportional to beta across all schemes.
Legacy's larger amplitude at the same beta does not establish a larger
log-log slope. Layer/stage/range differences are mixed; the broad H-005
verdict is inconclusive, while the near-parallel linear regime contradicts
a systematic larger legacy exponent there. All adjacent intervals and
both contrast columns are retained, rather than choosing a favorable range.

At beta .01 initial output D/F is .018794/.174764/1.198195 and epoch30
D/F is5.7097e-6/.00092333/.197233 (baseline/ours/legacy). Small-beta
first-layer cancellation and high-beta direction bias limit usable signals.
At beta1000 trained Conv3-layer cosine is .9984/.9304/.5788; legacy's
large output displacement does not guarantee faithful gradients.

Trained baseline Conv1/Conv3 residual flags persist at all19 betas;
initialization and trained ours/legacy pass. Literal positive-free plateaus
include continued free relaxation and differ from the centered response.
BPTT is a matched K8 finite-unroll reference, not exact equilibrium. This
single-seed diagnostic does not establish an accuracy mechanism or optimal beta.

[Full report, numerical tables and figures](../../../../../results/conv3-clean-eqprop-rms-beta-20260924-v1/analysis/report.md)
record the source/config/cohort identities, inclusion rules, all slopes,
resolution flags and validation. The experiment record retains exact source
checkpoint identities and launch command. All11 runner/aggregation checks
passed and all114 canonical bundles validate locally.

The next step is already assigned:
[exp-007](../../004-relative-output-training/experiments/exp-007-initial-relative-output-training.md)
tests matched initial relative output targets .8/1.2/1.6/2.0 in actual noisy
training. Its result cannot be inferred from the response amplitude alone.
