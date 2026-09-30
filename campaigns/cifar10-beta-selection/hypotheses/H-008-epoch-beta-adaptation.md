---
id: "H-008"
title: "Epoch-wise beta recalibration improves ten-epoch CIFAR learning"
---
# Does beta need recalibration each epoch?

Fixed initialization beta gave much smaller trained RMS and poor convolution
gradient alignment in exp-009. Compare two fresh ours runs: maintain relative
RMS0.09 by recalibrating before every epoch, versus freeze the shared initial beta.
Both arms use convolution-only centered EqProp with autograd boundary signals,
BN/gains/readout updates. This is a noiseless hybrid learning experiment.

Primary comparison is final epoch10 validation accuracy and cross-entropy.
Higher accuracy together with lower CE supports adaptation in this one-seed
pilot; both worse contradict the proposed benefit; ties/mixed outcomes are
inconclusive. Report actual differences, not statistical significance. This
comparison does not establish that every epoch is necessary rather than a less
frequent update, nor that RMS0.09 is optimal or robust to read noise.

[exp-010](../series/001-beta-selection/experiments/exp-010-epoch-beta-adaptation.md)
owns implementation, execution and evidence.
