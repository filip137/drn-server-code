---
id: "H-010"
title: "Ours has reproducible layerwise alignment advantages in relative-noise transition regions"
---
# H-010 — Relative-noise advantage regions

## Provenance and claim

Prospective extension of exp012's Conv3 result to a finer initialization map,
requested September 25; see [X-009](../explorations/X-009-relative-noise-advantage-ridge.md).
In each of Conv1, Conv2 and Conv3, at least one convolution has a sampled
neighborhood of positive ours-minus-legacy mean gradient cosine to finite-K
BPTT under matched output D/F and equal nodewise relative-noise eta.
Each measurement uses a single global eta for all noninput layers, including
readout, and all schemes. Layer-specific noise coefficients are excluded.
Both stages use the same eta ranges and sampled coordinates for Conv1/2/3;
all architectures evaluate the union of selected refinement patches.
Evaluate and report this claim separately for each architecture.

The expected mechanism is an intermediate precision range where legacy loses
alignment sooner than ours. At very low noise both may approach their clean
ceiling, and at high noise both may approach the floor. The location and size
of the advantage, and whether it forms a ridge in eta/(D/F), are measurements
to discover, not assumed results. Baseline is retained as a second comparator.

## Evidence and decision

Use exp013's fixed screen and deterministic neighborhood-selection rule.
Support for an architecture requires at least one selected convolution's
four-cell refinement patch to have a positive paired ours-minus-legacy mean
cosine difference in every one of three fresh noise draws at all four cells.
Report the absolute cosines and whether ours exceeds baseline as well.

Complete nonpositive screen and refinement results contradict the sampled-grid
claim for that architecture. Mixed signs, a failed confirmation, missing cells,
or unusable comparisons leave the proposed neighborhood unsupported or
inconclusive; do not infer superiority everywhere outside the grid. A positive
isolated cell does not establish the four-cell neighborhood. Across architectures,
support for the full claim requires support in all three; retain partial outcomes.

These rules are proposed before new measurements. Noise draws are not model
seeds, the criterion is descriptive rather than a significance test, and the
selected maximum remains conditional on the searched domain. This does not
claim all-layer dominance, a training benefit, or causal isolation of coupling.
