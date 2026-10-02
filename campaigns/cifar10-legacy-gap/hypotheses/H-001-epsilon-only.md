---
id: "H-001"
title: "Correcting effective BN epsilon alone closes the legacy endpoint gap"
---

# H-001 — Correcting effective BN epsilon alone closes the legacy endpoint gap


## Provenance
Retrospective candidate reconstructed after the completed epsilon tests;
[X-001](../explorations/X-001-coordinate-bn-equivalence.md).

## Claim and regime
At seed0 and the selected legacy optimizer, normalizing legacy block outputs before
BN is sufficient to reach or exceed the baseline epoch50 validation score under the
recorded L8 contract. This tests sufficiency, not whether epsilon has any effect.

## Support / falsification / inconclusive
Normalized legacy at or above the matched baseline endpoint supports sufficiency in
that run. A qualified result below it contradicts the scoped claim. Missing identity or
failed qualification is inconclusive. The reported 88.82% versus 89.48% is historical
counterevidence, not proof that epsilon never matters or cannot interact with rates.
