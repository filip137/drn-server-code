# X-003 — Is fifty epochs an adequate training horizon?

Subsequent September24 design: the three proposals below are now registered as
[H-006](../hypotheses/H-006-useful-late-learning.md),
[H-007](../hypotheses/H-007-legacy-late-catchup.md) and
[H-008](../hypotheses/H-008-lr-floor-limits-progress.md), with bounded experiments
in [series003](../series/003-training-horizon/README.md). The text below preserves
the exploratory reasoning before that registration. No new outcomes exist.

## Question and provenance

On September 24, 2026, Filip proposed that the current CIFAR results are not yet
final because loss is still decreasing and more epochs may be needed. This
exploration separates three questions: whether useful generalization gains
remain, whether the scheme ranking changes with training duration, and whether
the learning-rate schedule limits late progress.

The hypotheses below are new proposals motivated by already-observed results.
They are not preregistered explanations of those observations or registered
campaign hypotheses. No training or checkpoint evaluation was performed for
this exploration, and no new run is scheduled by it.

## Measured late-training behavior

The four completed analog L8 trajectories use seed 0, BPTT, Adam and the
45,000/5,000 CIFAR train/validation split. The training loss is cross-entropy
on the augmented training stream; validation CE uses the validation procedure.
They are different measurements, not losses on an identical fixed cohort.

The table compares means over epochs 41–45 and 46–50. Five-epoch windows reduce
sensitivity to individual evaluations; their epochs are correlated, not
independent replicates.

| Condition | Mean training CE, 41–45 → 46–50 | Mean validation CE, 41–45 → 46–50 | Mean validation accuracy change |
|---|---:|---:|---:|
| Baseline | 0.08485 → 0.06573 | 0.36979 → 0.36785 | +0.108pp |
| Ours | 0.05541 → 0.03965 | 0.36719 → 0.36825 | +0.472pp |
| Legacy | 0.09603 → 0.07602 | 0.38742 → 0.38322 | +0.180pp |
| Normalized legacy | 0.08110 → 0.06468 | 0.38902 → 0.38799 | +0.404pp |

For the broader endpoint comparison from epoch 40 to 50, training CE decreases
by 42.5% / 49.0% / 38.8% / 40.5%, respectively. Training CE also decreases from 49
to 50 in every condition, while validation CE increases on that last step in
all four. A single step does not establish an overfitting trend; the windowed
measurements show small validation CE improvements for three conditions and a
small increase for ours despite its larger accuracy improvement.

Legacy's mean accuracy deficit to baseline narrows from 0.760pp to 0.688pp
between these windows, but its deficit to ours widens from 0.732pp to 1.024pp.
Existing late behavior therefore does not show consistent catch-up. Fifty
epochs is a completed experimental endpoint, not a demonstrated convergence
point; falling training CE alone does not prove more epochs will improve
validation accuracy or change the ranking.

### Sources and read-only verification

The [existing stagnation analysis](../../../results/cifar10-l8-bn-gain-followup-seed0-20260923-v1/analysis/stagnation_e50.json)
records the methods, endpoint changes and late-window differences. The window
means above were recomputed from the original epoch metrics and checked against
the consolidated [training CSV](../../../results/cifar10-l8-bn-gain-followup-seed0-20260923-v1/analysis/training.csv).
Its SHA-256 matches the recorded analysis source:
`8b9629cd0c1ebb7b0f9d3373436109014f11f4708cc159d63d302802f7366e65`.

Source epoch metrics:

- [Baseline, epochs31–50](../../../results/cifar10-l8-analog-baseline-legacy-e30-e50-seed0-20260922-v1/cells/baseline_e30_e50/metrics.jsonl).
- [Ours, epochs31–50](../../../results/cifar10-l8-analog-ours-e30-e50-seed0-20260922-v1/cells/ours_e30_e50/metrics.jsonl).
- [Legacy, epochs31–50](../../../results/cifar10-l8-analog-baseline-legacy-e30-e50-seed0-20260922-v1/cells/legacy_e30_e50/metrics.jsonl).
- Normalized legacy: [retained continuation through48](../../../results/cifar10-l8-bn-gain-followup-seed0-20260923-v1/cells/legacy_voltage_normalized_e10_e50/metrics.jsonl)
  and [completed49–50 continuation](../../../results/cifar10-l8-bn-gain-followup-seed0-20260923-v1/cells/legacy_voltage_normalized_e48_e50/metrics.jsonl).

All 132 checked values (four conditions × eleven epochs 40–50 × training CE,
validation CE and validation accuracy) agree with the CSV within 1e-12. This
verifies those measurements, not full run-bundle validity, checkpoint identity
or the outstanding campaign source audit. Historical imports retain their
original evidence classification.

## Three proposed hypotheses and discriminating observations

1. **Useful generalization gains remain beyond epoch 50.** A bounded candidate
   test is 20 more epochs for all four conditions, preserving full training
   state and the existing schedule. Compare mean validation accuracy over 66–70
   with 46–50, report epoch 70 separately, and report validation CE alongside
   training CE. Accuracy gains would support remaining useful learning in
   those trajectories. Continued training-loss reduction without validation
   improvement would favor the competing explanation of continued fitting
   with little generalization benefit. Accuracy and CE may disagree; report
   that disagreement rather than calling both improved.

2. **Legacy's disadvantage partly reflects slower convergence.** Use the same
   matched-duration extension to compare legacy-minus-baseline and
   legacy-minus-ours accuracy differences between the two declared windows.
   Shrinking deficits against both comparators would support catch-up within
   the tested horizon. Persistent or widening deficits would count against
   this explanation there, even if legacy's own score improves. Mixed gap
   changes are inconclusive for a general catch-up claim. Track normalized
   legacy separately; it must not replace the raw legacy arm after seeing
   which result is more favorable.

3. **The learning-rate floor limits late progress.** The
   [runner's cosine schedule](../../../experiments/train_cifar_l8_analog.py)
   clamps its horizon fraction at 1. With the existing horizon 50 and final
   ratio 0.02, unchanged continuation beyond 50 holds rates at 2% of their
   starting values. Epoch 50 itself uses the preceding scheduler step, so its
   logged rates are slightly above that floor. A later paired continuation
   from the same full-state checkpoint could compare the unchanged floor
   with a separately specified renewed schedule. Improvement confined to the
   latter would support schedule limitation; comparable progress at the floor
   would weaken the need for renewal. This is a secondary control, not a
   selected new learning rate. Changing the original cosine horizon would
   alter the intervention and is not an unchanged continuation; the current
   restore contract also rejects scheduler-config changes.

## Interpretation limits and next decisions

Prioritize horizon qualification before treating epoch 50 rankings as settled.
Retain the source audit, complete-optimizer mapping and paired-seed questions
in [X-002](X-002-optimization-and-regime.md); additional epochs cannot substitute
for them. Preserve the negative BN/epsilon/gain controls, which concern the
conditions and horizons already measured.

The 20-epoch extension is a bounded test idea, not an executable experiment
contract. Before execution, register the chosen hypotheses and experiment,
freeze practical-effect thresholds and the accuracy/CE decision rules, audit
source checkpoints/configs and continuation comparability, and record targets,
measured runtime, budget and validity gates under the existing workflow.
Failure or negligible improvement by 70 would limit the claim over that
extension, not prove convergence at every possible horizon or learning rate.

These are single-seed trajectories with mixed historical GPU/software
environments and separately selected optimizer vectors. Within-run trends
are descriptive; no statistical significance or general scheme ranking is
inferred. Official-test evaluation, broad retuning and topology changes are
outside this exploration.
