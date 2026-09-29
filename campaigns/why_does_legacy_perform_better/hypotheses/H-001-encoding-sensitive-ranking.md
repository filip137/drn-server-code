---
id: "H-001"
title: "The clean amplification ranking depends on the target encoding"
---
# H-001 — Encoding-sensitive ranking

Retrospective hypothesis adopted September 29 after the completed Conv2 pilot.
Under the tested clean BPTT, seed-0, fixed-per-scheme Adam recipes and finite
[0,100] conductance bounds, changing one-hot target amplitude between 1/4 and
1/16 changes the ordering of best validation accuracy between ours and legacy.

A strict observed reversal supports this claim within that pilot; a tie does
not count as a reversal. The effect's population robustness requires independent
seeds. This scoped claim does not say that encoding explains most of the gap.
See [exp-001](../series/001-clean-voltage-and-encoding/results/exp-001-conv2-target-encoding.md).
