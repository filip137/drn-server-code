---
id: "H-016"
title: "The larger block3 RMS target may improve learning at the lower beta ceiling"
---
# Complete the existing ceiling and target comparison

The ceiling10,000/block3 RMS.001 cell reached32.64% validation accuracy
(exp015), while ceiling30,000/RMS.001 reached28.04% (exp017) and
ceiling30,000/RMS.005 reached32.88% (exp018). The missing lower-ceiling,
larger-target cell can separate the target effect from its interaction with
the ceiling. These observed effects need not combine additively.

[Exp021](../series/001-beta-selection/experiments/exp-021-low-cap-block3-rms.md)
tests that single missing cell. Compare directly with015 for the target effect
and018 for the ceiling effect. A matched native BPTT gate remains the criterion
for useful training quality: accuracy at least35.06% and CE at most1.812363172.
A relative improvement below that gate is evidence about the interaction,
not successful BPTT-like learning or permission to relax the threshold.

Exp019's exact prefix probes showed a transient block3 gradient discrepancy
but otherwise strong sampled agreement. They do not establish why the full
epochs failed, or predict this missing cell's result. Keep the original
optimizer, rates and architecture fixed rather than changing that question.
