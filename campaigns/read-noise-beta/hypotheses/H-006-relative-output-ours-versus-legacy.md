---
id: "H-006"
title: "Ours outperforms legacy under matched initial relative output displacement"
---
# H-006 — Ours outperforms legacy at matched relative output displacement

Filip stated this prospective hypothesis on September 24, 2026 after choosing
the output layer and initial ratios 0.8, 1.2, 1.6 and 2.0. See
[X-006](../explorations/X-006-normalized-output-target.md). This replaces the
earlier exploratory 10–20 suggestion with the user's selected grid; it does
not reinterpret the old p90/p99 results as a test of the new claim.

For each of Conv1, Conv2 and Conv3, compare ours and legacy after exactly ten
epochs of ordinary-MNIST centered EqProp training at additive endpoint read
noise sigma5e-4. Set scheme-specific beta by matching pooled centered output
displacement / free output RMS at their shared initialization, then hold beta
fixed. Preserve accepted architecture-specific T/K, initializer, zero biases,
Adam rates, batch size and data order. Include baseline as a reference.

Primary contrasts are epoch-10 validation accuracy(ours) minus
accuracy(legacy), paired by architecture and target ratio. Report all twelve
contrasts and each architecture's equal-weight mean over its four targets;
do not compare independently selected best targets or best epochs.

Support is scoped to an architecture if the mean paired contrast is positive
with all four training pairs complete and finite. A negative mean contradicts
the average-advantage claim there; exact ties or missing/failed pairs leave it
inconclusive, with failures and their direction reported separately. Also
report sign consistency across all four targets: a positive mean does not
mean ours wins at every target. The broad all-depth expectation is supported
only if each architecture meets its scoped criterion. These single-seed
descriptive signs are not statistical significance or a seed-general claim.

Calibration must reach every requested ratio within 1% relative error on the
frozen cohort before its training case is admitted. This is a proposed
numerical matching tolerance frozen before calibration, not an accuracy
margin or a convergence test. A failed match cannot be presented as testing
the matched-displacement hypothesis.

Interpretation limit: matching initial D/F does not match absolute output
D, hidden-layer displacement or signal-to-noise ratio under fixed additive
sigma5e-4. The schemes have different free output RMS. This is a test of the
specified normalization policy with inherited Adam rates, not an isolated
causal test of amplification or a test of relative Gaussian read noise.
Fixed beta also permits D/F to change during training. These distinctions
were already identified in X-006 and do not change the chosen cases or
primary contrast rule.
