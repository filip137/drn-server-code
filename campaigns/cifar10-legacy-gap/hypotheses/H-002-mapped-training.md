---
id: "H-002"
title: "The complete coordinate-and-optimizer mapping preserves training behavior"
---

# H-002 — The complete coordinate-and-optimizer mapping preserves training behavior


## Provenance
Prospective extension of a reported 200-minibatch identity test;
[X-001](../explorations/X-001-coordinate-bn-equivalence.md).

## Claim and regime
Under the original exact mapping, normalized legacy and baseline with the complete
baseline optimizer/config follow numerically matched training beyond the tested prefix,
with an epoch50 accuracy difference no larger than 0.2 percentage points.

## Support / falsification / inconclusive
Recover original equality tolerances and mapping from source; apply them to the matched
prefix. The 0.2pp endpoint margin is a proposed practical equivalence threshold, not a
source result, and must be frozen before extension. Passing the prefix and endpoint
supports a bounded extension only; it does not prove mathematical identity for every
trajectory. Matched, qualified excess divergence contradicts the scoped operational
claim. Uncontrolled RNG/environment/BN state changes or failed qualification are
inconclusive until corrected. Existing short-window evidence does not test the endpoint.
