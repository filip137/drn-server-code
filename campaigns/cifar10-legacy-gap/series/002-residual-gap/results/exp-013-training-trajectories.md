---
experiment: "exp-013"
evidence: "validated-local"
summary: "Legacy fits the augmented training stream less well; its persistent late validation deficit is confounded with separately selected optimizer settings, and existing floor continuations do not remove it."
verdicts: {}
---
# CIFAR training trajectories

Reconstructed all 150 original epoch records (three selected seed-0 runs,
epochs 1–50) and checked their nine canonical bundles, continuation checkpoint
provenance, complete epoch/step coverage and unchanged scientific settings
within each trajectory. Also validated 40 existing baseline/legacy floor epochs
as separate context. No new GPU work or dataset evaluation.

Root: `results/cifar10-training-trajectories-20260927-v1/`.
Reproduce: `python results/cifar10-training-trajectories-20260927-v1/analysis/review.py`.
Source paths, hashes, config differences and arithmetic: `analysis/summary.json`;
all original epoch metrics: `analysis/trajectories.csv`.
Figures: `analysis/training_trajectories.jpg`, `analysis/late_training.jpg` and
`analysis/floor_context.jpg`. Historical result roots remain unchanged.

| Epoch 50 | Online training CE | Online training accuracy | Validation CE | Validation accuracy |
|---|---:|---:|---:|---:|
| Baseline | 0.06097 | 98.287% | 0.36882 | 89.48% |
| Ours | 0.03602 | 99.093% | 0.37218 | 89.92% |
| Legacy | 0.07195 | 97.916% | 0.38564 | 88.72% |

Legacy training CE exceeds ours in every fixed five-epoch window, and baseline
in every window from 11–15 onward. This is a persistent fitting difference,
not a sudden epoch-30 failure. Legacy validation accuracy is below both in every
five-epoch window from 31–35 onward; before that, rankings fluctuate. In 46–50,
mean validation accuracy is 88.676% legacy, 89.364% baseline and 89.700% ours.
The isolated epoch-30 legacy advantage over baseline is not a sustained late lead.

The low-LR floor continuation provides an already executed test of extra time:
legacy mean training CE falls from 0.07602 (46–50) to 0.05809 (66–70), while mean
validation accuracy rises only 0.160 pp to 88.836%. Baseline improves 0.040 pp to
89.404%; the legacy deficit narrows by 0.120 pp but remains 0.568 pp. Validation
CE worsens for both. These two collected arms do not establish the full three-arm
horizon or renewal comparison; ours/renewal coverage was not collected for this
analysis. No other study is marked complete by this review.

Matching: identical initializer and split hashes, architecture, augmentation,
seed, batch32/eval16, iterations, Adam betas/epsilon/decay, BN/gain learning rates,
and cosine schedule. Differences include amplification and separately selected
convolution/head LR vectors. Ours' raw convolution LRs are approximately 9 times
legacy's, while its head LR is approximately one third of legacy's. These are
coordinate-dependent settings, not comparable physical update sizes. Baseline
and legacy share the same later GPU/software family; ours uses a different one.
All nine bundles pin the same model implementation hash; runner snapshots differ.
This is not a controlled estimate of amplification alone or a multi-seed result.

Interpretation: the evidence points to a fitting/optimization component and argues
against describing the gap as purely generalization. However, online training
metrics use augmentation, minibatch BN and changing parameters, whereas validation
uses endpoint weights and saved BN. Their subtraction cannot measure a clean
generalization gap. Baseline's better final CE but worse accuracy than ours also
shows that fitting CE alone does not determine the validation ranking.

Next proposed test, not launched: branch the original legacy epoch-30 checkpoint
into two matched ten-epoch continuations (31–40): original settings versus 3 times
the original convolution LR, keeping head/BN/gain LRs, Adam moments, scheduler
shape, batches, augmentation, solver and all other settings identical. Replay both
on the same GPU/software stack. This is an exploratory test of stronger convolution
learning during the interval where the validation deficit becomes persistent,
not a recommendation derived from raw LR ratios or an assumed optimum. The earlier
ten-epoch initializer LR search does not test this late intervention.

Use mean validation accuracy over 36–40 as the primary paired outcome, with a
proposed +0.20 pp practical threshold; require lower mean training CE as well to
support the proposed fitting explanation, and retain validation CE as a secondary
outcome. Improved fit without improved validation redirects the question toward
generalization/BN; worsening or no useful change rejects this particular late LR
intervention, not all optimizer explanations. Maximum 20 scheme-epochs, provisional
6 GPU-hour cap including smoke, subject to measured feasibility on one RTX 5090.
This plan does not authorize submission or change the completed runs. Its limited
single-seed result would be a screen before any full training or replication.

September28 follow-up: the proposed LR intervention was subsequently shortened,
executed and reviewed as [exp-014](exp-014-one-epoch-conv-lr.md). The abrupt 3×
increase worsened fitting and validation; extending it is not the next priority.

The requested frozen-affine-BN review revalidated four frozen bundles, including
legacy's recovered prefix, and reconciled all ten epochs for each scheme.
At epoch10, trainable/frozen validation accuracies are baseline75.26/70.72%,
ours75.36/73.76%, legacy74.68/73.02%. Freezing worsens training CE as well:
increases are0.13573,0.10507,0.09836 respectively. Every frozen trajectory has
lower validation accuracy than its own reference at every measured epoch.

Checkpoint verification confirms gamma=1, beta=0 and14,070 running-stat updates;
BN parameters are absent from the Adam groups. This freezes affine learning,
not normalization or running-stat updates. Initializer files were repackaged
and their hashes differ, but actual epoch0 model tensors match exactly. Split,
augmentation, batch sizes, iteration counts, scheduler and optimizer settings
match each scheme's reference; hardware histories differ and rates were not
retuned for frozen BN. Raw evidence/checks and the plot are in this root's
`analysis/bn_review.json` and `analysis/bn_affine_review.jpg`.

Legacy overtakes baseline when BN affine learning is removed (epoch10 difference
changes from-0.58pp to+2.30pp), because baseline loses more. This is not improved
legacy performance. The epoch10 ours-minus-legacy gap stays similar:0.68pp with
trainable versus0.74pp with frozen affine BN. Over the6–10 mean, however, that
gap falls from0.552pp to0.020pp, so endpoint and short-window observations must
not be generalized to the missing frozen-BN epoch50 regime.

The early frozen-BN data argue against selective legacy harm from learning BN,
while leaving late co-adaptation open. Current evidence synthesis and prioritized
proposals are in [X-002](../../../explorations/X-002-optimization-and-regime.md).
