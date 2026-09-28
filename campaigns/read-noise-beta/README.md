# Choosing beta for endpoint-read-noise EqProp

## Research question

Which output-displacement target and relative read-noise level maximize the
gradient-alignment advantage of ours over both legacy and baseline?

## Current conclusions — September 28

- **Choose beta to match output RMS displacement across amplification schemes.**
  This is the adopted comparison rule: use scheme-specific beta values to reach
  a common displacement target. The current experiments express that target as
  normalized output displacement D/F.
- **Initialization alignment gains depend on the noise model.** Relative noise
  exposes regions where ours leads legacy and baseline; see the completed
  [alignment map](series/006-relative-endpoint-noise/results/exp-013-relative-noise-advantage-map.md).
- **Training gains depend on architecture and noise.** The ten-epoch sweep finds
  wins for ours in Conv2/3, but none in Conv1. At epoch 30, the largest selected
  ours-minus-legacy gap is +6.80 pp for Conv3 at D/F=6 and eta=1e-4. Legacy leads
  the clean means and several noisy cells. See the
  [training results](series/007-relative-noise-training/README.md).

Training measurements are validation results. Noisy comparisons use one seed;
the Conv2/3 baseline and ours/legacy trajectories also differ in Adam history.
The result notes retain these limits and distinguish replay from training.

## Scope

Perfect-diode Conv1/2/3 on ordinary MNIST, using training/validation data only.
The campaign studies how beta, physical displacement and the noise model affect
EqProp gradients and learning. Exact numerical contracts belong to each experiment;
results from different regimes must remain distinguishable.

## Current direction — September 28

The relative-noise alignment map and exp014–018 training studies are complete.
The [training series](series/007-relative-noise-training/README.md) indexes the
ten-epoch sweep, selected continuations, three-seed clean controls and noisy
baseline controls. The read-noise table in Overleaf's
`bidir_paper_theory_revised.tex` was updated from these results on September 28
(Overleaf commit `c05d0b9`). No further run is assigned by this review.

- [Ideas](ideas.md): possible explanations and directions.
- [Current exploration](explorations/X-009-relative-noise-advantage-ridge.md) and
  [hypothesis](hypotheses/H-010-relative-noise-advantage-regions.md): reasoning and testable claim.
- [Ledger](ledger.md): experiment states, results and reviewed conclusions.

Keep grids, metrics, acceptance rules, budgets and execution handoffs in the
linked experiments. New runs follow the repository's [execution policy](../../docs/experiment_workflow.md).
