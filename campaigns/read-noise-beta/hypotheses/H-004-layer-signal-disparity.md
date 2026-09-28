---
id: "H-004"
title: "High readout cosine can coexist with weak hidden-layer alignment"
---

# H-004 — High readout cosine can coexist with weak hidden-layer alignment


## Provenance
Retrospective, checkpoint-specific claim; [X-002](../explorations/X-002-checkpoint-and-signal.md).

## Claim and regime
At the recorded p99 ours epoch-30 checkpoint, beta=0.987333678708, sigma=5e-4 and T=K=8,
median noisy readout cosine exceeds 0.99 while the first two convolution-layer medians
are below 0.1 on the source cohort and noise protocol.

## Support / falsification / inconclusive
Reproducing both inequalities supports the coexistence claim; failure of either in an
exact matched measurement contradicts it. Missing sources/reference adequacy or a
changed noise protocol is inconclusive. The cutoffs are a retrospective coarse
restatement of the reported table. This does not show that these cosines cause the
accuracy gap, nor prove that hidden-layer cosine predicts final accuracy.
