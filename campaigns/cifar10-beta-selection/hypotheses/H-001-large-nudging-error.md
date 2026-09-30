---
id: "H-001"
title: "Smaller beta restores CIFAR gradient alignment at fixed T/K"
---
# H-001 — Smaller beta restores CIFAR gradient alignment at fixed T/K

Prospective for the new sweep, motivated by the prior poor-alignment pilot and
[X-001](../explorations/X-001-beta-sensitivity.md).

Claim: large nudging is the principal cause of the weak three-conv gradients,
so decreasing B with T/K unchanged restores cosine>=0.90 for every Conv weight
within both three-conv blocks. The threshold is an exploratory criterion frozen
before this sweep, not a scientific pre-launch gate or proof of good training.

Support requires at least one tested reduced multiplier satisfying the criterion
in each block. If only some layers improve, report partial beta sensitivity and
leave the stronger restoration claim inconclusive. If no layer improves, this
bounded sweep contradicts the proposed explanation. Dead or unresolved responses
must be visible; a poor small-beta endpoint alone does not identify settling as
the cause. Keep both native and matched-local BPTT references in the report.
