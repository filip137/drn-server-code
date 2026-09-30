---
id: "H-017"
title: "Small internal responses contribute to poor trained EqProp gradients"
---
# Small responses as a mechanism for poor learning

Filip's September 30 hypothesis follows the large observed change in RMS
displacement during training: fixed beta may leave trained layers with responses
too small to estimate their gradients accurately. This is a prospective test of
that explanation, motivated by the existing training observations in
[X-001](../explorations/X-001-beta-sensitivity.md).

At a fixed checkpoint and on identical images, increasing beta should increase
internal displacement and restore gradient direction and scale if insufficient
response is the main limitation. Excessive beta may instead introduce bias.
Output-layer RMS alone may hide weak responses in earlier layers.

[exp-022](../series/001-beta-selection/experiments/exp-022-epoch-layer-response.md)
compares epochs 0, 1 and 5 of ours. Recovery across all layers of a block with
one common beta supports this mechanism in the tested replay. Persistent poor
alignment despite larger responses weakens a beta-only explanation under the
fixed finite-iteration contract. Boundary optima and incompatible layer optima
leave the question unresolved. Neither result alone establishes the cause of
the validation-accuracy gap or shows that a changed beta improves training.
