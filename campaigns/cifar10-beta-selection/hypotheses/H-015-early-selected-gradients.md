---
id: "H-015"
title: "Trained-checkpoint RMS targets may not preserve early training-gradient quality"
---
# Inspect the gradients selected during early learning

Exp-015 and017 both failed the full-epoch accuracy gate. Removing all RMS
misses in017 worsened accuracy, so target tracking alone does not explain
learning quality. The targets came from trained-checkpoint diagnostics, whereas
the early optimization transient has a rapidly changing response.

The prospective diagnostic in exp-019 tests whether the actual selected Conv
gradients disagree with finite-T/K BPTT during the first101 updates, and whether
that disagreement changes after the two ceiling trajectories diverge. Compare
also reset-free-T gradients to separate the reference's differentiation path.
Good agreement would weaken this explanation and redirect investigation toward
optimization sensitivity; it would not itself establish learning equivalence.
