---
id: "H-011"
title: "Relative-noise alignment advantages transfer to ten-epoch validation accuracy"
---
# H-011 — Relative-noise training transfer

Prospective training follow-up to exp013, requested September 25. Initialization
alignment alone is not training evidence, especially when earlier layers have
nearly zero cosine despite a large last-layer advantage.

At matched initial output D/F and a uniform nodewise relative endpoint-noise
coefficient, ours achieves higher epoch-10 ordinary-MNIST validation accuracy
than legacy at at least one declared cell in each architecture, Conv1, Conv2 and Conv3.
Report every cell and per-architecture mean across the complete declared grid;
selected wins do not imply superiority across the grid. A positive mean is a
separate descriptive average-advantage claim. No significance claim from one seed.

Support is scoped by architecture: at least one valid paired positive difference;
contradiction requires complete valid coverage with no positive difference.
Missing paired coverage is inconclusive. Numerical failures remain visible and
are not removed to improve a mean. A future multiple-model-seed study is needed
for reproducibility. See [exp014](../series/007-relative-noise-training/experiments/exp-014-relative-noise-training.md).
