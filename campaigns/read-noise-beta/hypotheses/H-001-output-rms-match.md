---
id: "H-001"
title: "The recorded injected betas match unit output RMS at initialization"
---

# H-001 — The recorded injected betas match unit output RMS at initialization


## Provenance
Retrospective restatement of the September 22 direct calibration, not a new prediction.
Motivation: [X-001](../explorations/X-001-fair-perturbations.md).

## Claim and regime
On the saved seed-0 Conv3 initializer and the 576-example cohort, T=K=8 and the exact
clean float64 source contract, injected betas baseline=88.7, ours=1.385 and legacy=0.02173
produce pooled RMS(v_plus-v_free) within 0.031% of one in all three schemes.

## Support / falsification / inconclusive
Source identity and the original table jointly support this narrowly stated observation.
A matched reanalysis or replay exceeding that recorded tolerance contradicts it in this
regime. Missing hashes/cohort or a changed config is inconclusive, not a falsifier.
The 0.031% bound describes the historical result, not a prospective acceptance threshold.
No statement about hidden layers, noisy training or other initializers follows.
