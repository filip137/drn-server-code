---
id: "H-003"
title: "The observed T sensitivity depends on the trained checkpoint"
---

# H-003 — The observed T sensitivity depends on the trained checkpoint


## Provenance
Prospective matched-replay test motivated by historical, checkpoint-specific observations;
[X-002](../explorations/X-002-checkpoint-and-signal.md).

## Claim and regime
With a shared replay beta and cohort, older p90-trained ours weights show a large
readout-cosine response to increasing free-phase T that is absent at initialization
and current p99 weights. Instantaneous beta alone does not explain this difference.

## Support / falsification / inconclusive
A common-environment fixed-reference replay supports the claim if the old p90 point
has an absolute T8-to-T16 cosine change above 0.1 and the matched newer/initial points
remain below 0.01. These are proposed diagnostic thresholds, not published source
criteria; freeze before new measurements. Similar sensitivities after exact matching
would contradict the proposed contrast. Missing checkpoints or an inadequate reference
make it inconclusive. This does not identify when or why the trajectory became sensitive.
