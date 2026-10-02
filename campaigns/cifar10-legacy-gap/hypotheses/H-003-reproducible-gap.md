---
id: "H-003"
title: "The epoch50 legacy deficit is reproducible across paired seeds"
---

# H-003 — The epoch50 legacy deficit is reproducible across paired seeds


## Provenance
Prospective; [X-002](../explorations/X-002-optimization-and-regime.md).

## Claim and regime
For the frozen current L8 configurations and common runtime environment, legacy's
validation accuracy at epoch50 is below both baseline and ours on paired seeds {0,1,2}.
This is a reproducibility screen of selected configurations, not a universal ranking.

## Support / falsification / inconclusive
Support this strict observed-seed claim only if both paired differences are negative
for every seed. A sign reversal contradicts the all-seeds claim, without establishing
the opposite population ranking. Report means and sample dispersion in either case.
Zero/small differences, unresolved source/environment mismatch, invalid arms or missing
seeds are inconclusive for broader claims. Do not treat epoch values or continuations
as independent samples and do not claim significance from three seeds alone.
