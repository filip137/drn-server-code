---
experiment: "exp-010"
evidence: "validated-local"
summary: "Stopped early at Filip's request: epoch5 adaptive/fixed accuracy51.82/53.56%, versus66.68% historical BPTT; planned epoch10 comparison incomplete."
verdicts: {"H-008": "inconclusive"}
---
# Beta-adaptation test stopped early

Filip requested cancellation after the substantial historical BPTT deficit was
reported. All four queue jobs are cancelled, workers stopped and reservations
released. No further runs or retries are authorized. Completed coverage:

| Hardware pair | Last completed epoch | Adaptive accuracy | Fixed accuracy |
|---|---:|---:|---:|
| Local/nom-cool-1 RTX3090 |2|39.56%|39.34%|
| Loulou/Riri RTX5090 |5|51.82%|53.56%|

At epoch5, adaptive/fixed validation CE is1.345932/1.274742. Historical ours
BPTT reached66.68% and CE0.941809 at the same epoch: deficits14.86/13.12pp.
Adaptive training accuracy49.44% also trails BPTT61.92%. Architecture,
initialization, batch and LR settings match, but gradient estimation and solver
differentiation differ; this does not isolate beta as the cause. Hardware/runtime
versions also differ across pairs. The repeated seed0 runs are not independent
seed replicates. H-008's predeclared epoch10 endpoint was not reached, so its
verdict remains inconclusive despite the unfavorable epoch5 adaptive result.

Adaptive epoch5 post-training relative RMS is[0.12129,0.04662,0.008622], versus
the0.09 start-of-epoch target: epoch-wise calibration does not hold RMS constant
throughout learning. No read noise was tested.

Partial bundles and checkpoints are preserved under
`results/cifar-beta-adaptation-20260929-v2/` and
`results/cifar-beta-adaptation-5090-20260929-v1/`; each root's
`analysis/cancellation.json` records confirmed queue shutdown. Interrupted
epochs3/6 are excluded. nom-cool-1's termination raised a DataLoader signal
exception during pause handling, leaving bundle status failed; completed epoch2
metrics/checkpoint remain evidence, not a scientific nonfinite failure.
Local validation is limited to collected metrics, confirmed shutdown and loaded
epoch2/epoch5 full-state checkpoints; it does not certify ten-epoch coverage.
The original pre-update cuDNN failures remain in the V1 root linked by the
experiment note. Ten-epoch completion validators were not claimed or run.
