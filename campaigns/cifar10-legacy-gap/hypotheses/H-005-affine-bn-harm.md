---
id: "H-005"
title: "Learning affine BN selectively harms legacy in the current setting"
---

# H-005 — Learning affine BN selectively harms legacy in the current setting


## Provenance
Retrospective explanation, tested by the completed frozen-affine controls;
[X-001](../explorations/X-001-coordinate-bn-equivalence.md).

## Claim and regime
At the recorded scheme-selected rates and ten-epoch horizon, freezing affine BN
improves legacy validation accuracy while retaining normalization/statistic updates.
This is not the same intervention as removing BN or fitting new inference statistics.

## Support / falsification / inconclusive
Improvement in an otherwise matched legacy frozen-affine control supports the scoped
claim; degradation contradicts it. The historical 74.68% trainable versus 73.02% frozen
result is counterevidence. It does not prove that BN is optimal for every LR or horizon.
Unmatched configs or invalid solver results are inconclusive.
