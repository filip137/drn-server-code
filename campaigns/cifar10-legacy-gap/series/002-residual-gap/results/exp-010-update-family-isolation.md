---
experiment: "exp-010"
evidence: "validated-local"
summary: "Batch16: convolution updates account for 97% of legacy's non-classifier held-out CE increase; all four directions are adverse in both BN modes."
verdicts: {}
---
# Update-family isolation

User-authorized batch-16 revision: four gradient batches per scheme use the first64 training indices.
The held-out training cohort remains256 examples, evaluated in batches16. This changes
BN statistics and update directions relative to exp009; it is not an exact replay.
All eight saved-state directions and80 measurements validated locally. Original
checkpoint files and BN running buffers are unchanged. Every enabled/frozen tensor
matches its full-step/checkpoint value exactly. No validation or official-test data.

Execution root: `results/cifar10-update-family-probe-20260927-v1/`.
Inputs, environment and source identities: `probe-b16/manifest.json`. Detailed values:
`analysis-b16/summary.json`; plot: `analysis-b16/update_family_attribution.jpg`.
Production seconds: 640.0; peak PyTorch reserved memory: 5758.0 MiB.
The earlier Akib smoke failed at its memory cap; it produced no scientific measurements
and is retained in `smoke/790a842e94af47c6bc1f29f77ced476a/`.
The intervening batch32 nom-cool-1 smoke failed with CUDNN_STATUS_NOT_INITIALIZED,
retained in `smoke/0bf3d27a310d455a8c0c1e280d7769d6/`. Neither produced scientific measurements.

The table is separate-training-cohort CE. Positive attribution increases loss;
positive omission benefit means freezing that group improves on the full update.
Shapley values average marginal effects across all six group orderings, conditional
on the full classifier update. They sum to full-step minus classifier-only loss.

| BN mode | Scheme | Group | Mean attribution | Mean omission benefit | Adverse directions |
|---|---|---|---:|---:|---:|
| minibatch | legacy | conv | +0.00168910 | +0.00168990 | 4/4 |
| minibatch | legacy | bn | +0.00005127 | +0.00005207 | 4/4 |
| minibatch | legacy | gain | -0.00000001 | +0.00000000 | 0/4 |
| minibatch | ours | conv | -0.00024887 | -0.00024532 | 3/4 |
| minibatch | ours | bn | -0.00006999 | -0.00006644 | 0/4 |
| minibatch | ours | gain | -0.00000001 | +0.00000002 | 0/4 |
| saved | legacy | conv | +0.00209507 | +0.00209754 | 4/4 |
| saved | legacy | bn | +0.00006189 | +0.00006437 | 4/4 |
| saved | legacy | gain | -0.00000002 | -0.00000004 | 0/4 |
| saved | ours | conv | +0.00082720 | +0.00083189 | 3/4 |
| saved | ours | bn | +0.00000630 | +0.00001100 | 1/4 |
| saved | ours | gain | -0.00000001 | -0.00000002 | 0/4 |

Under the frozen primary criterion, legacy-specific candidate groups are: **conv, bn**. Candidates also passing in saved-stat BN: **conv, bn**.
Primary tolerance: 1e-05 CE. Batch size16 changes the BN and gradient comparison regime.

Reviewer interpretation: convolution updates dominate legacy's local loss increase:
97.1% of the non-classifier increase, or 88.7% of the entire full-step increase.
BN affine updates contribute about 2.9% of the non-classifier increase; gain
effects are below tolerance. Mean conv/BN interaction is only 0.00000160 CE in
minibatch BN, below tolerance, so the dominant signal is the convolution update
itself. Saved-stat BN gives the same ordering and four adverse legacy directions.

Legacy's convolution updates decrease loss on all four gradient minibatches
(mean attribution −0.04423 CE), while increasing separate-cohort loss in all four.
This identifies a local update-transfer problem. It does not distinguish excessive
step size, stochastic direction mismatch, or sharpness. Ours is variable: its
minibatch-BN mean convolution contribution is beneficial, driven by one strongly
beneficial direction; three directions are adverse, and the saved-stat mean is
adverse. Do not describe ours as consistently improving across batches or BN modes.

These are conditional effects of one update, at one checkpoint/seed and batch16.
The four directions share one cohort. They do not establish the cause of the
historical final-accuracy gap. The next discriminating check would vary only the
convolution step size and use a fresh separate cohort to distinguish oversize
from a direction that remains adverse near zero. No follow-up launched.

The predeclared arithmetic and criteria were applied automatically after
collection; the interpretation above was subsequently reviewed against per-batch
measurements, both BN modes, and the measured interaction terms.
