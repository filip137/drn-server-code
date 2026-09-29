---
id: "exp-002"
title: "Import clean EqProp initialization and trained layerwise voltages"
status: "imported"
hypotheses: ["H-002"]
---
# exp-002 — Layerwise voltage magnitude

Historical read-only replay: September 28. Campaign import: September 29;
no new replay or validation is claimed.

Nine seed-0 trained clean EqProp checkpoints and nine corresponding
initializations: Conv1 final epoch 10, Conv2/3 final epoch 30; baseline (1,1),
ours (4,1), legacy (4,0.25). Replay the same 576 validation examples, 36 batches
of 16, float64, perfect diodes, T=K=4/6/8, gains 40/100/360, frozen zero biases.
The source review records 18 validated contexts and 54 layer rows, exact restore,
unchanged checkpoints/parameters and no optimizer steps or official-test reads.

Primary measures: count-weighted free-state RMS and mean absolute voltage over
all examples and nodes in each layer. Saved-beta nudged endpoint RMS is a
separate diagnostic; trained output displacement was not rematched.

Preserved local records: `campaigns/pilots/clean-eqprop-trained-voltage-20260928.md`,
`configs/conv/clean_eqprop_trained_voltage_20260928/`, and
`results/conv123-clean-eqprop-trained-voltage-20260928-v1/` (including replay,
smoke and collection evidence). Original charge: 0.02593 GPU-hours.
[Imported result](../results/exp-002-layerwise-voltage.md). No new runs assigned.
