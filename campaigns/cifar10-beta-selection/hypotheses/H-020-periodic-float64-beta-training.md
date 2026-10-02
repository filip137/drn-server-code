---
id: "H-020"
title: "Periodic response calibration improves float64 learning over epoch-five fixed beta"
---
# Periodic calibration versus a trained-checkpoint beta

The [local-reference replay](../series/001-beta-selection/results/exp-025-all-scheme-layer-rms.md)
identifies common output d/f targets but strongly changing beta requirements.
Filip requests two fresh five-epoch ours runs in float64: recalibrate several
times per epoch, or fix beta using the saved EqProp epoch5 checkpoint.

Prospective claim: four calibrations per epoch better maintain targets
[0.01,0.005,0.00001] and improve epoch5 validation accuracy and cross-entropy
relative to the fixed vector. Compare response tracking, training fit, final
validation metrics and runtime separately. A tracking improvement without a
learning improvement supports response control only. Opposing accuracy/CE
changes are mixed evidence; one seed does not establish general superiority.
No accuracy margin or early-extension gate is imposed. This is local-gradient
motivated hybrid EqProp training, not proof of equivalence to full-network BPTT.

[exp-026](../series/001-beta-selection/experiments/exp-026-float64-periodic-vs-fixed-beta.md)
owns the two runs and their execution handoff.
