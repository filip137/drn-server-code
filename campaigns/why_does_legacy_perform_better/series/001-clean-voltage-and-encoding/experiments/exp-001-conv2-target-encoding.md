---
id: "exp-001"
title: "Import the completed clean Conv2 target-encoding intervention"
status: "imported"
hypotheses: ["H-001", "H-002"]
---
# exp-001 — Conv2 target encoding

Historical execution: September 28. Campaign import: September 29; no new
training, validation or retrospective preregistration is claimed.

Six clean BPTT runs: baseline/ours/legacy × target amplitude 1/16 or 1/4,
seed 0, 30 epochs, T=K=6, input gain 100, frozen zero biases, bounds [0,100],
paired-output native MSE without compensating loss normalization, inherited
scheme-specific Adam rates and a shared Kaiming initializer/data order.
Ordinary MNIST 55,000/5,000 split; official test disabled.

Original primary metric: best validation accuracy over 30 epochs, with final
accuracy and all six outcomes retained. Historical target-1 runs are references
only. Slurm array 278574 completed all six tasks; no failed or excluded cells.
The original review records canonical validation and 10.5722 GPU-hours used.

Preserved local records: `campaigns/pilots/conv-output-scale-evidence-20260928.md`,
`configs/conv/target_scale_conv2_20260928/`, and
`results/conv2-target-scale-20260928-v1/` (including smoke, transport and runs).
Published measurements: [result note](../results/exp-001-conv2-target-encoding.md).
The import has no new budget or execution handoff.
