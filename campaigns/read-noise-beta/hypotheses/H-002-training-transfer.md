---
id: "H-002"
title: "Matched initial output RMS improves the noisy-training beta policy"
---

# H-002 — Matched initial output RMS improves the noisy-training beta policy


## Provenance
Prospective candidate; [X-001](../explorations/X-001-fair-perturbations.md).

## Claim and regime
For Conv3 sigma=5e-4, at otherwise frozen within-scheme training contracts, the
initial-unit-output-RMS policy yields a positive mean paired final-validation accuracy
change over the current p99 policy for each scheme, without invalid or nonfinite runs.
The source initializer's values are anchors, not universal beta values across seeds.

## Support / falsification / inconclusive
Use exp-005's common endpoint and paired seed outcomes. Support is provisional if
means are positive and all validity guards pass; report seed uncertainty and do not
claim statistical resolution from three seeds alone. A qualified nonpositive difference
for a scheme, or a reproducible candidate-specific failure, contradicts improvement for
that scheme. Mixed seeds/small uncertain differences do not establish superiority.
No equivalence claim is allowed without a predeclared margin. A good RMS/cosine replay
alone does not test this hypothesis. Do not claim success for all schemes from two.
