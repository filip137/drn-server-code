---
experiment: "exp-014"
evidence: "validated-local"
summary: "84 cases and 42 pairs validated; sampled wins absent in Conv1, present in Conv2/Conv3"
verdicts: {"H-011": "contradicts"}
---
# Exp014 reviewed completion

All 84 original cases and 42 matched pairs were collected and validated locally: 840 epochs, finite epoch metrics, canonical artifact hashes, frozen scientific config, CUDA execution, disabled official-test evaluation, and matching pair host/source identities. No missing, failed, excluded or duplicate cases. Final checkpoints and launcher receipts are preserved in production/local and collected/{nom1,trex,fifi,loulou,riri}/production under the study root.

| Architecture | Positive pairs | Full-grid mean ours minus legacy (pp) | H-011 sampled existence |
|---|---:|---:|---|
| Conv1 | 0/7 | -0.068571 | contradicts |
| Conv2 | 12/21 | +0.196190 | supports |
| Conv3 | 6/14 | +0.728571 | supports |

H-011 requires a positive cell in each architecture; its combined claim is contradicted by Conv1. Positive Conv2/Conv3 means are descriptive grid averages, not universal cellwise superiority. This is one initializer/seed, ten epochs, validation-only evidence with beta matched at initialization and frozen thereafter. It establishes neither significance nor a global optimum. No further simulations are authorized by this review.

Launcher wall receipts total **79.878562 GPU-hours**, including startup/teardown and retained controller history, below the unchanged 120-hour cap. Each lane total reconciles exactly to its case receipts (28/28/8/8/6/6 successful exits); no GPU preparation was charged according to the transport handoff. All lanes terminated before the original deadline. This is charged workload wall time under actual sharing, not exclusive-device utilization or an estimate of unshared throughput. Historical controller replacements and the preserved Trex first case remain in launch receipts; there was no training retry or budget reset.

The supplied analyzer validated the complete declared grid using the six roots in monitor/collection-receipt.json. Detailed per-cell differences, coverage and JPG are in analysis/epoch10_differences.csv, analysis/coverage.csv, and analysis/epoch10_accuracy.jpg. No tests, canaries, retuning or official-test evaluation were run. Monitoring can close for exp014; unrelated and later experiments are outside this decision. No external callback was sent.

Study root: `results/conv123-relative-noise-training-20260925-v1`.

[Coverage and report](../../../../../results/conv123-relative-noise-training-20260925-v1/analysis/report.md) · [Epoch-10 JPG](../../../../../results/conv123-relative-noise-training-20260925-v1/analysis/epoch10_accuracy.jpg).
