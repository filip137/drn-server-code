---
id: "H-005"
title: "More nudged iterations improve trained block2 gradient alignment at fixed beta"
---
# Nudged-K follow-up

The epoch50 curves suggest different useful displacement scales across layers,
especially in ours block2. Limited propagation of the nudged response is one
possible cause, alongside nonlinear and finite-precision effects.

At each scheme's selected epoch50 block2 beta, increase only nudgedK from6 to
12/24/48 while holding initial free states, frozen forces and BPTT references
fixed. Primary prediction: ours gains at least0.01 in its worst-layer pooled
cosine; baseline/legacy are controls. Changes below0.001 are practically unchanged
for this screen. Report all layer/batch outcomes and actualRMS. Changed displacement
can mediate any K effect; this fixed-beta test does not independently holdR fixed.

Test: [exp-007](../series/001-beta-selection/experiments/exp-007-fixed-beta-nudged-k.md).
