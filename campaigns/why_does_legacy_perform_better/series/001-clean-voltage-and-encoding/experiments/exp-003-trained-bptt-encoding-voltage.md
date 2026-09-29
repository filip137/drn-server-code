---
id: "exp-003"
title: "Measure trained Conv2 BPTT voltages at target encodings 1/16 and 1/4"
status: "complete"
hypotheses: ["H-002"]
---
# exp-003 — Trained BPTT encoding voltages

September 29: Filip requested the voltage magnitudes of the trained networks
from exp-001. This is a new read-only replay of those six checkpoints; it does
not reuse the different EqProp checkpoints in exp-002.

Compare baseline (1,1), ours (4,1), legacy (4,0.25), target amplitudes 1/16 and
1/4, seed 0, fixed final epoch 30. Preserve native float32, asynchronous T=6,
input gain 100, perfect diodes, weights [0,100], frozen zero biases and original
preprocessing. Use the same 576 validation examples (36 batches of 16) and
reset each batch. No optimizer steps or official-test reads.

Measure count-weighted RMS, signed mean, standard deviation, mean absolute
voltage, extrema and zero occupancy in Conv1, Conv2 and readout. Compare
layerwise rankings descriptively; one seed cannot establish a causal mechanism.
No initial/best/intermediate checkpoints or epoch trajectory are included.

The execution contract and review were recorded in the existing source pilot,
`campaigns/pilots/conv-output-scale-evidence-20260928.md`; preserve that record.
Config: `configs/conv/conv2_target_scale_voltage_20260929.json`.
Runner: `experiments/replay_conv2_target_scale_voltages.py`.
Analyzer: `experiments/analyze_conv2_target_scale_voltages.py`.
All evidence, including smoke, is under
`results/conv2-target-scale-voltage-20260929-v1/`.

One local CPU thread, root monitoring owner; budgets 180 seconds smoke and
600 seconds production, stop on identity, restore, nonfinite or budget failure.
All six smoke and six production cases completed and validated; production
elapsed 11.92 seconds. No failures or excluded cases. Monitoring is complete.
[Reviewed result](../results/exp-003-trained-bptt-encoding-voltage.md).
