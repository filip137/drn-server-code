---
id: "H-012"
title: "A larger batch speeds up ours without materially reducing validation accuracy"
---
# H-012 — Practical batch-size improvement

For the latest ours CIFAR setup with trainable BN, at least one GPU-resident batch
larger than32 reduces mean epoch2–10 time by >=10% while losing <=0.5pp epoch10
validation accuracy, on the same V10032GB and seed0. Keep learning rates fixed.
Test in [exp-018](../series/002-residual-gap/experiments/exp-018-batch-size.md).
A passing case supports the scoped claim. Complete comparisons in which every
larger fitting batch misses the criterion contradict it for these tested cases.
Capacity failures alone do not test accuracy; incomplete training is inconclusive.
