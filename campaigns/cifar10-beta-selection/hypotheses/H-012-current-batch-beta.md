---
id: "H-012"
title: "Choosing beta on the current minibatch can avoid lagged-response overshoots"
---
# Can current-batch response selection improve learning?

Exp-014's lagged controller tracks median RMS but sometimes overshoots block2
by orders of magnitude; epoch1 accuracy34.34% and CE1.981908 miss matched BPTT.
This motivates a new, prospective comparison: use the current minibatch's frozen
free states and loss force to choose beta before its optimizer update. Keep the
RMS targets unchanged so this tests timing rather than a new target search.

Maintaining response is not the success criterion. The full-epoch validation
gate in [exp-015](../series/001-beta-selection/experiments/exp-015-current-batch-beta.md)
must pass before longer training; current-batch calibration may still give
biased gradients or unsuitable targets.
