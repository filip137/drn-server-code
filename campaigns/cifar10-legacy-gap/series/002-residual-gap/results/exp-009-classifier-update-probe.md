---
experiment: "exp-009"
evidence: "validated-local"
summary: "Halving the classifier step helps legacy at epoch 30, but epoch-50 effects fall below tolerance; the two-checkpoint prediction is inconclusive."
verdicts: {"H-009": "inconclusive"}
---
# Result: classifier update imbalance

All 16 directions completed: original seed-0 legacy/ours at epochs 30/50,
four gradient minibatches each, evaluated on the same 256 separate training
examples. The 96 variant measurements include no-update and repeat controls.
Trex execution, smoke and validation took 375 seconds; collection succeeded.
No training continuation, validation-set evaluation or official-test read occurred.

Evidence: [canonical probe](../../../../../results/cifar10-classifier-update-probe-20260927-v1/probe/result.json),
[analysis](../../../../../results/cifar10-classifier-update-probe-20260927-v1/analysis/summary.json),
[JPG](../../../../../results/cifar10-classifier-update-probe-20260927-v1/analysis/classifier_half_step.jpg).
Frozen inputs, source hashes, commands and environment are in the probe manifest
and its execution root. Local `queue_worker.py validate` and `analysis/review.py`
passed canonical and exact-coverage checks. Original checkpoint hashes and BN
buffers were unchanged; all non-classifier parameter updates matched exactly.

Benefit below means **full-step CE minus half-classifier-step CE** on separate
training examples; positive favors halving. Primary evaluation uses minibatch BN.
Repeated no-update loss discrepancies were zero; the frozen tolerance is 0.00001 CE.

| Epoch | Legacy mean benefit | Directions improved | Ours mean benefit | Directions improved |
|---|---:|---:|---:|---:|
| 30 | +0.00007375 | 4/4 | −0.00000048 | 1/4 |
| 50 | −0.00000057 | 0/4 | −0.00000035 | 0/4 |

Epoch 30 passes the local prediction, including excess benefit over ours.
Saved-stat BN agrees: legacy benefits by 0.00004960, whereas ours worsens by
0.00002140. The epoch-50 effects are below tolerance in both modes, so the
predeclared requirement for support at **both** epochs fails. The contradiction
criterion also fails. The overall H-009 verdict remains **inconclusive**.

The classifier contributes a small adverse effect at epoch 30. Legacy's full
update increases separate-cohort CE by 0.002141 on average; halving the classifier
step removes 3.4% of that increase. Even suppressing the classifier update entirely
removes only 6.9%. The same-gradient-batch loss improves with the full step, so
this is a local transfer issue rather than failure to descend on that minibatch.
These secondary controls do not replace the failed two-checkpoint prediction.

This does not establish the cause of the historical accuracy gap or justify a
long classifier-LR sweep. Four directions share one cohort and one seed; epoch 50
uses the saved LR floor. A useful next intervention would isolate convolution,
affine-BN and gain updates at epoch 30 to identify the remaining loss increase.
The authorized follow-up is [exp-010](../experiments/exp-010-update-family-isolation.md).
The broader question remains open.
