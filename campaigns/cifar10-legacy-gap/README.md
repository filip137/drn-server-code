# Why does legacy not retain its expected advantage on CIFAR-10?

## Research question and deliverable

Why is legacy behind baseline and ours at the current CIFAR-10 L8 endpoint, despite
advantages observed in other regimes? First establish whether the endpoint gap is
reproducible and the comparison is matched; then distinguish coordinate/normalization
changes from optimizer/history effects. Deliver a supported explanation or a bounded
set of explanations ruled out, plus an honest residual uncertainty statement.
A result need not restore a legacy advantage to answer the question.

## Scope and comparison regime

Use the current analog L8: eight convolutions in blocks [3,3,2], logical widths
128/256/512, differential physical widths 256/512/1024, and analog 8192-to-10 head.
Digital pooling/BN separate voltage-clamped circuits; electrical feedback is within
blocks, while BPTT crosses boundaries. This is not the same experiment as fully coupled
MNIST Conv3 endpoint-noise EqProp; a ranking does not transfer automatically.

The historical primary comparison uses CIFAR training split 45,000/5,000, seed0,
Adam, crop/flip, CE without label smoothing, batch32/eval16, 50-epoch cosine decay to
2% of starting LR, trainable affine BN, four initial scalar gains of 100, and
T=K=[6,6,4] with higher-iteration audits. Baseline (1,1), ours (4,1) and legacy (4,0.25)
use separately selected conductance/head LR vectors. Full optimizer/scheduler/BN/RNG
state is preserved across continuations; continuation segments are not replicates.

Do not mix analog validation scores with exploratory digital references that repeatedly
read the official test set. Do not open CIFAR official test for this campaign's selection.
Do not add read noise, change topology, or retune all hyperparameters without a separate
bounded experiment: those interventions answer different questions.

## Starting evidence — imported, not freshly validated

[Consolidated September 24 review](../../docs/cifar_experiment_review_20260924.md):

| Analog condition | Epoch30 validation | Epoch50 validation |
|---|---:|---:|
| Baseline | 86.54% | 89.48% |
| Ours | 87.20% | 89.92% |
| Legacy | 86.88% | 88.72% |
| Normalized legacy, legacy-selected optimizer | 86.44% | 88.82% |

Legacy is above baseline at epoch30 and below it at epoch50. The endpoint deficit is
0.76pp versus baseline and 1.20pp versus ours in this seed, not an established
multi-seed ranking. Accuracy and CE also do not give identical baseline/ours orderings.

The completed controls already constrain explanations: epsilon correction alone adds
only 0.10pp at epoch50; affine BN helps all schemes at the tested rates; recalibration
does not repair legacy's final accuracy; sampled native solver checks pass. The
matched-coordinate identity test uses the **complete baseline optimizer**, unlike the
normalized-legacy 88.82% run that retains legacy-selected rates. Do not conflate them.
See [historical series](series/001-existing-evidence/README.md) and [ledger](ledger.md).

## Runbook and authoritative sources

September24 priority update: first qualify the training horizon through
[series003](series/003-training-horizon/README.md). Its exp-006 prepares four
unchanged epoch50→70 continuations; exp-007 specifies two matched schedule
renewals; explicit runner support is now implemented and CPU-tested. All six cases are assigned
provisionally to Fifi after the existing EqProp allocation releases its lane,
with a 36 GPUh combined cap. Filip subsequently authorized scheduling the prepared six-case study for
September24 at22:00 Europe/Paris; preflight preparation is now in progress.
Production remains subject to the declared gates and protected existing lane. Existing source audit,
optimizer-mapping and replication questions remain open.

The narrow preparation check found the four epoch50 parent checkpoints and
prepared assets locally, validated their canonical bundles and pinned hashes
in exp-006. This removes missing access for those inputs only; it does not
complete exp-003's wider mapping/comparability audit or reclassify imports.

Start with [ideas](ideas.md) and [exp-003: source audit](series/002-residual-gap/experiments/exp-003-source-audit.md).
The [mechanism plan](../../docs/cifar_l8_mechanism_plan_20260923.md),
[findings](../../docs/cifar_l8_mechanism_findings_20260923.md) and
[consolidated review](../../docs/cifar_experiment_review_20260924.md) identify exact
controls and original output paths. Recover complete resolved configs, not approximate
LR vectors from prose. Keep completed negative tests; do not repeat the same epsilon
or no-BN ablation as though it were untested.

Then [exp-004](series/002-residual-gap/experiments/exp-004-mapped-optimizer-control.md)
extends the mapped-coordinate/optimizer control if the audit leaves a worthwhile gap.
[exp-005](series/002-residual-gap/experiments/exp-005-paired-seed-ranking.md) tests endpoint
reproducibility. The priority between those tests may be revised from audit evidence,
not by quietly adding an unbounded queue. Execute with the existing
[workflow](../../docs/experiment_workflow.md) and [reporting](../../docs/experiment_reporting.md).

## Metric roles and acceptance

| Role | Measurement | Decision use |
|---|---|---|
| Primary outcome | Validation accuracy at epoch50, with paired seed differences | Establish a reproducible endpoint deficit before generalizing it. |
| Secondary outcome | Validation CE, training CE, declared early endpoints | Separate calibration/optimization/generalization and horizon effects. |
| Mechanism | Matched logits, gradient norms/directions, actual Adam updates, BN states/effective epsilon, gains | Test an intervention, not merely correlate raw gradients with performance. |
| Validity | Original source/config/continuation identity; solver/reference checks; finite outputs and complete coverage | Invalid or unmatched scores remain labelled rather than promoted. |
| Cost | Run/replay counts and measured runtime | Decide whether a new test is worth its bounded cost. |

Use final epoch50 rather than per-arm best epochs for the primary comparison. For
the new horizon series, use the explicitly declared mean 66–70 versus 46–50 and
paired schedule contrasts instead; do not rewrite the historical epoch50 endpoint.
For replication, report seed-level paired differences, mean and dispersion for both
legacy-baseline and legacy-ours. Three paired seeds provide an initial reproducibility
screen, not a definitive significance claim. A proposed practical equivalence margin
for the controlled mapping extension is 0.2pp at the endpoint; it must be frozen before
new measurements and is not a historical source tolerance.

## Budget and stop rules

Setup uses zero GPU time and starts no queue. Proposed caps: two-hour read-only audit;
a mapped-control extension of at most two 50-epoch trajectories / 24 GPU-hours; and
six new seed1/2 production runs / 48 GPU-hours only if audited seed0 is comparable.
If seed0 cannot be matched, any replacement design must be recorded first; the absolute
fallback cap is nine 50-epoch runs / 72 GPU-hours. Runtime feasibility requires a measured
smoke, not the assumption that a named GPU is idle.

The newly designed horizon series adds at most 120 scheme-epochs and 36 GPUh
(24 for four floor continuations,12 for two renewal branches; reused controls
charged once). Its initial estimate is 25.3 GPUh and roughly three overnight
windows on one RTX5090, with at most four eligible nights within seven calendar
days of first production start. This budget is a plan, not launch authority.
Detailed admission, recovery and stop rules live in the series README.

Stop on failed source/reference/solver qualification, exceeded caps or a scientific
contract mismatch. Keep a negative mechanism test as a result. Do not claim one cause
from multiple simultaneous changes, and do not declare the campaign solved just because
an optimizer transplant changes a score. Record unresolved alternatives explicitly.
