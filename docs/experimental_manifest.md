# Experimental Manifest

Updated: 2026-08-25

This is the manually curated ledger of analyzed studies. It is not generated
from raw metrics and does not track transient job state.

Each finished entry records:

- scientific question and evidence class;
- frozen setup, arms, seeds, and inclusion rule;
- outcome: `positive`, `negative`, `mixed`, or `inconclusive`;
- headline measurements;
- interpretation;
- limitations and protocol deviations;
- raw run and study-analysis locations; and
- links to detailed tables, plots, or reviewed historical result cards.

As of 2026-08-17, ordinary MNIST is the active paper dataset. Its held-out
5,000-example validation accuracy remains selection/mechanism evidence and
must never be presented as paper accuracy. Only a once-read official-test
metric from a sealed eligible checkpoint is paper-facing. Existing entries
retain their recorded evidence class until a separate reuse audit and test
evaluation explicitly promote them. An operational failure needs an entry
only when it materially affects the scientific conclusion.

## Published Learning-Rate Handoffs

These are the current parameter-wise optimizer inputs for downstream
unbounded/wide-range work. Publishing a handoff does not promote its
ordinary-MNIST validation measurements to paper evidence or erase its stated
review and confirmation limits.

| Handoff | Scope | Evidence status | Machine-readable authority |
|---|---|---|---|
| `perfectdiode-conv12-unbounded-fixed-lr-20260729-v1` | Conv1/Conv2, `[0,100]` | Conv1 ten-epoch confirmations complete, review pending; Conv2 three-epoch evidence only | [`perfectdiode_conv12_unbounded_fixed_lr_handoff_20260729_v1.json`](../configs/conv/perfectdiode_conv12_unbounded_fixed_lr_handoff_20260729_v1.json) |
| `perfectdiode-conv3-unbounded-selected-lr-20260729-v1` | Conv3, downstream `[0,100]` authorized from a `weight_max=null` source study | six-surface rho selection complete; source-contract mismatch recorded; long confirmation pending | [`perfectdiode_conv3_unbounded_fixed_lr_handoff_20260729_v1.json`](../configs/conv/perfectdiode_conv3_unbounded_fixed_lr_handoff_20260729_v1.json) |

### Conv1/Conv2 current fixed vectors

Notation is `C_i=ConvWeight_i`, `B_i=Bias_i`, and `D=DenseWeight_0`.
Runtime order is `[C0,D,B0]` for Conv1 and `[C0,C1,D,B0,B1]` for
Conv2. The machine-readable handoff derives each ordered vector from the named
mapping in that runtime order.

| Architecture | Scheme | Optimizer | Parameter-wise LR vector |
|---|---|---|---|
| Conv1 | baseline | SGD | `C0=B0=0.142696, D=0.0120238` |
| Conv1 | baseline | Adam | `C0=B0=8.66763e-4, D=1.09401e-4` |
| Conv1 | ours | SGD | `C0=B0=0.0375994, D=0.00311655` |
| Conv1 | ours | Adam | `C0=B0=8.66753e-4, D=1.09388e-4` |
| Conv1 | legacy | SGD | `C0=B0=2.05753e-4, D=4.76259e-5` |
| Conv1 | legacy | Adam | `C0=B0=2.88917e-4, D=3.64618e-5` |
| Conv2 | baseline | SGD | `C0=B0=7.90864, C1=B1=3.81476, D=0.820630` |
| Conv2 | baseline | Adam | `C0=B0=2.60078e-3, C1=B1=4.59053e-4, D=4.67660e-4` |
| Conv2 | ours | SGD | `C0=B0=0.523358, C1=B1=0.246043, D=0.0520201` |
| Conv2 | ours | Adam | `C0=B0=2.60036e-3, C1=B1=4.58651e-4, D=1.55136e-4` |
| Conv2 | legacy | SGD | `C0=B0=0.00496733, C1=0.00367839, B1=6.74726e-4, D=0.00236044` |
| Conv2 | legacy | Adam | `C0=B0=8.66750e-4, C1=B1=1.52841e-4, D=5.16982e-5` |

## Finished Studies

### Physical-KCL rerun scope closeout — 2026-09-02

The required corrected physical-KCL compatibility scope is complete.  It
consists of the Conv1 clean-wide checkpoint and same-LR BPTT diagnostic; the
Conv2/Conv3 clean-wide same-LR BPTT runs; the Conv2/Conv3 clean and
endpoint-noisy centered-EqProp repeats; and the three fixed-initialization
bounded Conv3 BPTT cells at `Gmax={1e-4,5e-4,1e-3}`.  Every scheduled run is
terminal, collected locally, and validated; the official MNIST test split
remained sealed.  The original 18-arm table was an exhaustive impact audit,
not the final execution contract.  Its unrun Conv1 EqProp and Conv1/Conv2
bounded cells, together with deferred wide SGD qualification, are optional
future work rather than missing required reruns.  See the
[closed inventory](legacy_physical_kcl_rerun_inventory.md) and the terminal
[operational record](current_simulations.md#physical-kcl-rerun-closeout).

## `perfectdiode-conv1-legacy-physical-kcl-same-weights-and-bptt-rerun-seed0-20260825-v1` — Conv1 legacy checkpoint compatibility and exact-LR retraining under the corrected physical KCL energy

- Analyzed: 2026-08-25
- Evidence class:
  `ordinary_mnist_physical_kcl_same_lr_diagnostic`; implementation-change
  compatibility evidence, non-paper-facing.
- Outcome: `mixed`.  Corrected-energy Adam remains equivalent at the
  predeclared `0.5 pp` resolution, whereas both unchanged-checkpoint and
  exact-LR SGD comparisons change materially.
- Scientific question: Does replacing the formerly implemented coherent but
  differently parameterized legacy energy with the physical KCL energy
  materially change Conv1 validation accuracy when old checkpoint tensors
  are replayed unchanged or training is repeated with the exact old rates?
- Frozen setup: seed-0 ordinary MNIST deterministic `55,000/5,000`
  train/validation split; legacy `(A,B)=(4,.25)`; perfect-diode BPTT;
  exact-zero hidden bias; `T=K=4`; batch size `16`; validation batch size
  `64`; ten epochs; source SGD rates
  `[2.05753e-4,4.76259e-5,0]` and Adam rates
  `[2.88917e-4,3.64618e-5,0]`.  Checkpoint replays use the source study's
  exact best/final bytes, reset state each batch, and apply no optimizer step.
  The official test split remains sealed.
- Coverage and integrity: both exact-config one-batch smokes, four full
  5,000-example checkpoint replays, and both ten-epoch retrains complete.
  All six scientific production bundles validate; every production manifest
  records clean fix commit `f016bf6a`; config, seed, split/order, epoch,
  optimizer, and LR contracts match the source arms.  Both new runs report
  zero official-test evaluations, and every best/final `Bias_0` tensor is
  exactly zero.  The initial checkpoint smoke is retained but excluded from
  inference because it compared one batch with a full-split source accuracy;
  corrected `smoke_v2` is the accepted gate.
- Headline unchanged-checkpoint measurements: source-to-corrected best/final
  validation is SGD `95.92/95.60 -> 93.84/93.88%`
  (`-2.08/-1.72 pp`) and Adam `96.44/96.38 -> 96.12/96.24%`
  (`-.32/-.14 pp`).
- Headline exact-LR retraining measurements: source-to-corrected best/final
  validation is SGD `95.92/95.60 -> 91.78/91.78%`
  (`-4.14/-3.82 pp`) and Adam `96.44/96.38 -> 96.50/96.44%`
  (`+.06/+.06 pp`).  Corrected SGD selects epoch 10 and corrected Adam epoch
  9.
- Interpretation: historical convergence is expected because the former code
  still minimized a coherent energy; it does not validate the manuscript's
  physical circuit equations.  Conv1 Adam's learned decision rule and frozen
  optimizer scale are empirically robust to the correction at one seed.  SGD
  is not: its same-checkpoint loss is material, and the larger retraining loss
  shows that optimizer scaling is more sensitive than checkpoint reuse.
  Corrected Conv1 legacy SGD would therefore require LR reselection before it
  could be used as corrected evidence; it is excluded from the closed required
  scope.  The old Adam rate is accepted for the completed compatibility scope,
  not as a generally requalified legacy result.
- Limits and scope: one architecture, one seed, two BPTT optimizers, and
  validation-only evidence.  This study alone neither estimates uncertainty
  nor clears Conv2/Conv3, EqProp phase gradients and beta, read noise, bounded
  weights, or an official-test evaluation.  The selected Conv2/Conv3 checks
  were completed separately and are recorded in the closeout above.  Broader
  Conv1 SGD, EqProp, and bounded follow-up is optional rather than required for
  the current paper integration.
- Artifacts: [study report](../results/perfectdiode-conv1-legacy-physical-kcl-same-weights-and-bptt-rerun-seed0-20260825-v1/analysis/report.md),
  [checkpoint replays](../results/perfectdiode-conv1-legacy-physical-kcl-same-weights-and-bptt-rerun-seed0-20260825-v1/checkpoint_replay/),
  [exact-LR retrains](../results/perfectdiode-conv1-legacy-physical-kcl-same-weights-and-bptt-rerun-seed0-20260825-v1/same_lr_bptt/), and
  [legacy rerun inventory](legacy_physical_kcl_rerun_inventory.md).

## `perfectdiode-conv13-zero-bias-adam-centered-float64-eqprop-one-decade-sigma-5em4-ordinary-mnist-10-30ep-seed0-20260817-v1` — Conv1/Conv3 one-decade EqProp training under endpoint read noise

- Analyzed: 2026-08-17
- Evidence class:
  `ordinary_mnist_eqprop_zero_bias_read_noise_training_diagnostic`;
  user-directed exploratory robustness evidence, non-paper-facing.
- Outcome: `mixed`. All six runs complete finite `10/30`-epoch budgets and
  pass the frozen within-run stability rule. Conv1 remains effectively
  unchanged from clean, but Conv3 loses validation accuracy, with a severe
  legacy penalty.
- Scientific question: At the already clean-stable one-decade beta, do the
  exact-zero-bias shared-`T/K` Conv1 and Conv3 Adam EqProp cases remain stable
  when endpoint voltage reads used by the local update have sigma `5e-4`
  Gaussian noise?
- Frozen setup: seed-0 ordinary MNIST's deterministic `55,000/5,000`
  train/validation split; baseline/ours/legacy; exact inherited BPTT weight
  learning rates; exact-zero bias rates and tensors; centered frozen-current
  true-float64 EqProp; `T=K=4/8` for Conv1/Conv3; `10/30` epochs; injected
  beta Conv1 `100/30/3` and Conv3 `100/3/.001`; independent negative/positive
  layer-endpoint noise with seed `2026081601`; input noise disabled; noiseless
  validation; no official-test read. The six exact clean controls come from
  the reviewed zero-noise one-decade qualification.
- Coverage and integrity: Jean Zay UMG V100-32GB array `1037835_[0-2]`
  completed all tasks with exit `0:0` in `05:00:24`--`05:07:34`, after
  canary `1037825_0` passed. The authoritative local copy has zero checksum
  differences from Jean Zay. All `6/6` canonical bundles, six run receipts,
  three pack receipts, three semantic summaries, `2,887,920` expected noise
  draws, initialization/split/order matching, zero-bias, completion, and
  no-test-read guards validate.
- Headline measurements: noisy best/final validation for Conv1
  baseline/ours/legacy is `96.18/96.10`, `96.46/96.28`, and
  `96.40/96.38%`; clean-relative final changes are `+.06/-.06/-.04 pp`, and
  no epoch differs by more than `.08 pp`. Conv3 noisy best/final is
  `96.94/96.84`, `96.96/96.82`, and `88.30/87.64%`; clean-relative
  best/final changes are `-.82/-.84`, `-1.76/-1.82`, and
  `-10.78/-11.38 pp`.
- Interpretation: the one-decade tier is a finite optimizer-level training
  point at sigma `5e-4`, but it is not a common practical robustness point.
  Conv3 legacy still learns and remains only `.66 pp` below its own best at
  the end, so it does not meet the declared collapse rule; nevertheless, its
  `11.38 pp` final clean-relative penalty is a large performance failure.
  Training stability and read-noise tolerance must therefore be recorded
  separately, and a within-run best-to-final rule alone is inadequate for the
  noise decision.
- Limits and retained deviations: one model seed and one fixed noise
  realization do not estimate uncertainty. The study does not isolate depth,
  beta, or amplification causally, freeze a paper beta, or authorize an
  official-test read. The one-decade `213/216` gradient result and trained
  Conv3-baseline `T=8` residual caveat remain explicitly unresolved.
- Artifacts: [report](../results/perfectdiode-conv13-zero-bias-adam-centered-float64-eqprop-one-decade-sigma-5em4-ordinary-mnist-10-30ep-seed0-20260817-v1/analysis/report.md),
  [summary table](../results/perfectdiode-conv13-zero-bias-adam-centered-float64-eqprop-one-decade-sigma-5em4-ordinary-mnist-10-30ep-seed0-20260817-v1/analysis/summary.csv),
  [trajectory plot](../results/perfectdiode-conv13-zero-bias-adam-centered-float64-eqprop-one-decade-sigma-5em4-ordinary-mnist-10-30ep-seed0-20260817-v1/analysis/validation_accuracy_trajectories.png),
  [machine summary](../results/perfectdiode-conv13-zero-bias-adam-centered-float64-eqprop-one-decade-sigma-5em4-ordinary-mnist-10-30ep-seed0-20260817-v1/analysis/summary.json), and
  [collection receipt](../results/perfectdiode-conv13-zero-bias-adam-centered-float64-eqprop-one-decade-sigma-5em4-ordinary-mnist-10-30ep-seed0-20260817-v1/analysis/collection_validation_receipt.json).

## `perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-above-max-to-10xmax-sigma-5em4-seed0-20260817-v1` — Conv3 broadened all-layer cosine versus beta under endpoint read noise

- Analyzed: 2026-08-17
- Evidence class:
  `ordinary_mnist_learning_algorithm_read_noise_gradient_diagnostic`;
  exploratory read-only mechanism study, non-paper-facing.
- Outcome: `negative` for finding a per-minibatch all-layer faithful beta and
  `mixed` for locating layer-specific optima. None of the six scheme x
  checkpoint contexts reaches noisy EqProp/BPTT cosine `>=.99` on every
  `C0/C1/C2/Dense` layer and all four batches. The enlarged range does bound
  the trained C2 average optimum for baseline and ours, but not the early
  trained convolutional-layer optima.
- Scientific question: How do all Conv3 layerwise EqProp/BPTT cosine curves
  continue when beta is swept one decade above the original maximum under
  endpoint-voltage read noise sigma `5e-4`?
- Frozen setup: the reviewed nine-point parent is joined with eight new
  `1/8`-decade points, producing beta-scale factors `10^(i/8)`, `i=0..16`.
  Injected ranges are baseline `100--10000`, ours `3--300`, and legacy
  `.001--.1`. The same seed-0 exact-zero-bias Adam BPTT source checkpoints,
  reconstructed initialization, best-validation checkpoint, deterministic
  four-batch ordinary-MNIST validation cohort, `T=K=8`, centered
  frozen-current true-float64 EqProp, exact BPTT reference, and all
  `C0/C1/C2/Dense` weights are used throughout. Independent sigma-`5e-4`
  negative/positive endpoint reads use seed `2026081601`; the standard-normal
  tensors are identical across schemes and all 17 beta points. A clean EqProp
  control remains embedded at every point.
- Coverage and integrity: the local RTX 3090 extreme-point smoke and all eight
  extension production bundles validate canonically; the reviewed nine parent
  bundles also revalidate in the join. Combined coverage is `408/408`
  checkpoint-batch replays, `1632/1632` layer-batch comparisons, and
  `3264/3264` endpoint-noise records. All 17 noise signatures match exactly.
  All 96 BPTT reference coordinates are beta-invariant with maximum norm delta
  `0`; source/checkpoint bytes, in-memory parameters, exact-zero biases,
  true-float64 arithmetic, shared `T/K`, frozen force, cohort, input
  exactness, no-optimizer-step, and no-official-test-read guards pass.
- Headline conservative measurements: best all-layer worst-batch cosine at
  initialization/best is baseline `.162914 @ B=1778.28` / `.786156 @
  B=10000`, ours `-.015895 @ B=53.3484` / `-.039670 @ B=300`, and legacy
  `.012911 @ B=.1` / `-.071393 @ B=.1`. The corresponding clean worst cosine
  at each trained best point is baseline `.978357`, ours `.926021`, and legacy
  `.969854`. Dense passes from the lower endpoint throughout; trained baseline
  C2 remains the only convolutional all-batch `.99` crossing, first at
  `B=1000` and peaking conservatively at `.994911 @ B=1778.28`.
- Headline four-batch averages at the trained checkpoint: baseline peaks at
  C0/C1/C2/Dense `.897801 @ 10000`, `.980900 @ 10000`, `.996141 @ 1778.28`,
  and `.999999 @ 1778.28`; ours at `.091214 @ 300`, `.189009 @ 300`,
  `.976147 @ 168.702`, and `.999996 @ 30`; legacy at `.056734 @ .1`,
  `.003716 @ .1`, `.577923 @ .1`, and `.999992 @ .023714`. Baseline/ours C2
  fall to `.983442/.970383` at the enlarged endpoint after their interior
  optima.
- Interpretation: the extra decade exposes the expected read-noise versus
  finite-beta tradeoff and identifies some deep-layer peaks, but it does not
  produce a common all-layer useful beta. Early C0/C1 estimates remain
  noise-limited even where their averages continue rising; initialization
  curves eventually deteriorate strongly from finite-beta distortion. These
  above-maximum values are diagnostic only and are not training-qualified.
  Prior training already shows optimizer-dependent collapse at the original
  boundary, so this study does not nominate larger training betas.
- Residual and limitations: trained baseline retains the known
  beta-independent post-`T=8` residual caveat. The study has one model seed,
  one noise seed, four minibatches, and one absolute noise scale. Some trained
  early-layer curves remain ceiling-bound, so the experiment bounds neither
  every layer-specific optimum nor a read-noise-only asymptote.
- Artifacts: [analysis report](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-above-max-to-10xmax-sigma-5em4-seed0-20260817-v1/analysis/report.md),
  [four-batch-average table](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-above-max-to-10xmax-sigma-5em4-seed0-20260817-v1/analysis/best_checkpoint_average_cosine_vs_beta.csv),
  [average-per-layer plot](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-above-max-to-10xmax-sigma-5em4-seed0-20260817-v1/analysis/plots/best_checkpoint_average_cosine_vs_injected_beta_all_layers.png),
  [all-layer table](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-above-max-to-10xmax-sigma-5em4-seed0-20260817-v1/analysis/layerwise_cosine_vs_beta.csv),
  [per-layer thresholds](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-above-max-to-10xmax-sigma-5em4-seed0-20260817-v1/analysis/per_layer_thresholds.csv),
  [median/spread plot](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-above-max-to-10xmax-sigma-5em4-seed0-20260817-v1/analysis/plots/median_cosine_vs_injected_beta_all_layers.png),
  [worst-batch plot](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-above-max-to-10xmax-sigma-5em4-seed0-20260817-v1/analysis/plots/worst_batch_cosine_vs_injected_beta_all_layers.png),
  [machine summary](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-above-max-to-10xmax-sigma-5em4-seed0-20260817-v1/analysis/summary.json),
  and [frozen extension config](../configs/conv/perfectdiode_conv3_zero_bias_adam_eqprop_bptt_gradient_beta_sweep_above_max_to_10xmax_sigma_5em4_seed0_20260817_v1.json).

## `perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-one-decade-to-max-sigma-5em4-seed0-20260817-v1` — Conv3 all-layer cosine versus beta under endpoint read noise

- Analyzed: 2026-08-17
- Evidence class:
  `ordinary_mnist_learning_algorithm_read_noise_gradient_diagnostic`;
  exploratory read-only mechanism study, non-paper-facing.
- Outcome: `negative` for finding a per-minibatch all-layer faithful beta in
  the declared one-decade-to-original-maximum interval. None of the six
  scheme x checkpoint contexts reaches cosine `>=.99` on every
  `C0/C1/C2/Dense` layer and all four batches at any tested point.
- Scientific question: For every Conv3 amplification scheme and weight
  layer, how does centered EqProp versus same-`T/K` BPTT cosine change when
  injected beta rises logarithmically from the accepted one-decade tier to
  the original maximum under endpoint-voltage read noise sigma `5e-4`?
- Frozen setup: the same seed-0 exact-zero-bias Adam BPTT source checkpoints
  and deterministic four-batch ordinary-MNIST validation cohort as the
  accepted direct gradient gate; Conv3 baseline/ours/legacy; reconstructed
  initialization and best-validation checkpoints; `T=K=8`; centered
  frozen-current true-float64 EqProp; exact BPTT reference; all
  `C0/C1/C2/Dense` weights; nine uniformly log-spaced beta-scale factors
  `10^(i/8)`, `i=0..8`; injected ranges baseline `100--1000`, ours `3--30`,
  and legacy `.001--.01`. Independent sigma-`5e-4` negative/positive
  endpoint reads use seed `2026081601`; standard-normal tensors are identical
  across schemes and all beta points. The clean EqProp replay is retained at
  each beta.
- Coverage and integrity: the local RTX 3090 semantic smoke and all nine
  production bundles validate canonically. The joined surface contains
  `216/216` checkpoint-batch replays, `864/864` layer-batch comparisons, and
  `1728/1728` endpoint-noise records. All nine noise signatures match exactly
  across beta; all 96 BPTT reference norms are beta-invariant with maximum
  delta `0`; source/checkpoint bytes, in-memory parameters, exact-zero biases,
  true-float64 arithmetic, shared-`T/K`, frozen force, cohort, input
  exactness, no-optimizer-step, and no-official-test-read guards pass.
- Headline measurements: the best observed all-layer worst-batch cosine at
  initialization/best is baseline `.1157/.0841` at `B=1000`, ours
  `-.0243/-.1892` at `B=30`, and legacy `-.00497/-.0720` at `B=.01`.
  Dense passes cosine `.99` from the one-decade anchor in all six contexts.
  Among convolutional layers, only trained baseline C2 reaches an all-batch
  `.99` crossing, at `B=1000` with minimum `.9923`. Initialization C2 peaks
  below threshold at baseline `.9863` (`B=316.2`), ours `.9804`
  (`B=12.65`), and legacy `.8369` (`B=.01`). No C0 or C1 context crosses
  `.99`; their best minima remain at most `.1157/.4875` for baseline,
  `-.0243/-.00355` for ours, and `.0233/-.00109` for legacy across
  initialization/best.
- Interpretation: raising beta improves read-noise signal margin in several
  deeper layers, most visibly C2, but the early convolutional readouts remain
  noise-dominated throughout the already qualified training interval. At the
  original maximum, the clean initialization worst-layer cosine has also
  fallen to baseline/ours/legacy `.8022/.8016/.9767`, exposing the opposing
  finite-beta distortion. Therefore choosing the largest trainable beta does
  not recover strict single-minibatch all-layer fidelity at sigma `5e-4`;
  reducing the endpoint noise or averaging multiple independent reads or
  gradients is the more direct remedy. This does not imply optimizer-level
  collapse, because the separate noisy training study measures multi-step
  averaging rather than one instantaneous gradient.
- Residual and limitations: trained baseline retains the known
  beta-independent post-`T=8` residual caveat. The sweep has one model seed,
  one noise seed, four batches, one absolute noise scale, and stops at the
  original maximum because larger betas are outside the previously qualified
  training interval and clean finite-beta distortion is already visible.
- Artifacts: [analysis report](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-one-decade-to-max-sigma-5em4-seed0-20260817-v1/analysis/report.md),
  [all-layer table](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-one-decade-to-max-sigma-5em4-seed0-20260817-v1/analysis/layerwise_cosine_vs_beta.csv),
  [per-layer thresholds](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-one-decade-to-max-sigma-5em4-seed0-20260817-v1/analysis/per_layer_thresholds.csv),
  [median/spread plot](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-one-decade-to-max-sigma-5em4-seed0-20260817-v1/analysis/plots/median_cosine_vs_injected_beta_all_layers.png),
  [worst-batch plot](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-one-decade-to-max-sigma-5em4-seed0-20260817-v1/analysis/plots/worst_batch_cosine_vs_injected_beta_all_layers.png),
  and [machine summary](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-one-decade-to-max-sigma-5em4-seed0-20260817-v1/analysis/summary.json).

## `perfectdiode-conv23-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-one-decade-sigma-5em4-seed0-20260817-v1` — Conv2/Conv3 gradient fidelity with endpoint read noise and Conv2 high-beta extension

- Analyzed: 2026-08-17
- Conv2 extension analyzed: 2026-08-18
- Evidence class:
  `ordinary_mnist_learning_algorithm_read_noise_gradient_diagnostic`; formal
  read-only hyperparameter/robustness diagnostic, non-paper-facing.
- Outcome: `negative` for direct per-minibatch gradient fidelity at endpoint
  read-noise sigma `5e-4`. None of the 12 architecture x scheme x checkpoint
  contexts passes the all-layer noisy-versus-clean acquisition gate or the
  noisy-versus-BPTT task gate.
- Scientific question: If the current one-decade EqProp beta is retained, do
  Conv2 and Conv3 gradients remain faithful when the established independent
  Gaussian endpoint-voltage read-noise model is applied at sigma `5e-4`?
- Frozen setup: seed-0 zero-bias Adam BPTT source checkpoints; Conv2/Conv3 x
  baseline/ours/legacy; reconstructed initialization and best validation;
  four fixed 16-example ordinary-MNIST validation batches; shared
  `T/K=6/6,8/8`; centered frozen-current EqProp in true float64; injected
  one-decade beta Conv2 `100/10/.03` and Conv3 `100/3/.001`; all Conv/Dense
  weights scored; cosine `>=.99` and symmetric norm delta `<=.10`; input and
  BPTT reference exact. Independent sigma-`5e-4` noise is added to every
  negative/positive free-layer endpoint voltage after equilibrium and before
  the local energy-gradient readout, with seed `2026081601` and matched
  standard-normal draws across schemes.
- Coverage and integrity: the local RTX 3090 smoke and production bundles
  validate canonically. Production completes `48/48` replay contexts,
  `168/168` layer-batch comparisons, and `336/336` noised endpoint tensors in
  54 seconds. All source/checkpoint and in-memory parameter bytes remain
  unchanged; exact-zero-bias, true-float64, shared-`T/K`, frozen-force,
  cohort, matched-noise, finite-gradient, input-exactness, and zero-test-read
  guards pass. The embedded clean replay is exactly identical to all 168
  accepted one-decade parent rows.
- Conv2 high-beta extension: five additional local RTX 3090 production
  bundles cover relative beta scales `10,17.7828,31.6228,56.2341,100`, from
  the original maximum through ten times that maximum. Their Conv2 slice has
  `120` checkpoint-batch replays, `360` layer-batch rows, and `720` endpoint
  noise records. All five bundles validate canonically; Conv2 residuals and
  all scientific/read-only guards pass; BPTT norms are exactly beta-invariant;
  and all noise signatures match byte-for-byte across beta. The bundle-level
  residual failures are solely the retained Conv3 baseline caveat. The
  scale-10 embedded clean replay agrees with the accepted clean maximum to
  `1.30e-14` cosine and exactly in BPTT norm and norm delta. One first smoke
  attempt is an excluded pre-case operational failure because sandboxed CUDA
  was unavailable; the replacement noisy endpoint smoke completes and
  validates.
- Headline measurements: layer-batch acquisition/task/combined passes are
  `52/50/50` of `168`. Every Dense row passes (`48/48`), and Conv2-baseline
  C1 contributes the only two additional task passes; every other
  convolutional layer/scheme group has `0/8` task passes across the two
  checkpoints. Conv2 minimum noisy EqProp/BPTT cosine at initialization/best
  is baseline `.5476/.2136`, ours `.2423/-.0888`, and legacy
  `.00715/-.1713`. Conv3 is baseline `-.0862/-.0554`, ours
  `-.0551/-.2066`, and legacy `-.00696/-.0720`. Maximum context norm deltas
  reach `1.99999`.
- Interpretation: a single centered two-phase gradient estimate is strongly
  noise-dominated in the convolutional layers at sigma `5e-4`; the Dense
  gradient remains well resolved. The effect is especially severe for the
  smaller scheme-specific base betas because endpoint-gradient readout noise
  is divided by the finite-difference denominator. This is not a training
  collapse result: separate noisy long-training runs can remain finite and
  accurate because stochastic gradient errors may average over many
  minibatches and optimizer steps. It does show that sigma `5e-4` cannot be
  called per-minibatch all-layer gradient-faithful at the one-decade tier.
- High-beta interpretation: larger beta materially improves noisy upstream
  direction before finite-nudge bias dominates. At the trained checkpoint,
  baseline C0 noisy mean peaks at `.983554` (`B=3162.28`), ours C0 reaches
  `.953794` (`B=1000`), and legacy C0 reaches only `.332723` (`B=3`). By
  relative scale `31.6228`, only Dense layers retain the conservative
  all-batch `.99` gate; by `56.2341`, every trained convolution-layer mean is
  below `.99`. At the ten-times-maximum endpoint, clean trained C0/C1 means
  are baseline `.969744/.946317`, ours `.965700/.943066`, and legacy
  `.932222/.938292`. This is unambiguous direct-gradient failure at high beta,
  not a new training-collapse measurement. It supports a layer-dependent
  signal-to-noise/locality tradeoff but does not qualify a layer-specific
  training estimator.
- Residual and limitations: trained Conv3 baseline retains the known
  beta/noise-independent post-`T=8` residual caveat. One model seed, one noise
  seed, four batches, one absolute uncalibrated noise scale, and a strict
  per-layer/per-batch gate; no multi-step optimizer-noise averaging or paper
  accuracy is estimated.
- Artifacts: [analysis report](../results/perfectdiode-conv23-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-one-decade-sigma-5em4-seed0-20260817-v1/analysis/report.md),
  [layer table](../results/perfectdiode-conv23-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-one-decade-sigma-5em4-seed0-20260817-v1/analysis/layer_summary.csv),
  [clean-versus-noisy plot](../results/perfectdiode-conv23-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-one-decade-sigma-5em4-seed0-20260817-v1/analysis/plots/layerwise_clean_vs_noisy_cosine.png),
  [machine summary](../results/perfectdiode-conv23-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-one-decade-sigma-5em4-seed0-20260817-v1/analysis/summary.json),
  [production bundle](../results/perfectdiode-conv23-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-one-decade-sigma-5em4-seed0-20260817-v1/production/),
  [paper Conv2 clean/noisy beta figure](../papers/amplification_overleaf/figures/beta_cosine/conv2_by_scheme_all_layers_cosine_vs_beta.pdf),
  [paper summary table](../papers/amplification_overleaf/figures/beta_cosine/beta_cosine_summary.csv),
  and [paper provenance](../papers/amplification_overleaf/figures/beta_cosine/provenance.json).

## `perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-one-decade-ordinary-mnist-10-30-30ep-seed0-20260816-v1` — zero-bias shared-T/K one-decade Adam EqProp training qualification

- Analyzed: 2026-08-17
- Evidence class: `ordinary_mnist_eqprop_zero_bias_training_qualification`;
  user-directed exploratory hyperparameter-selection evidence,
  non-paper-facing.
- Outcome: `positive` for the declared training-stability question. All nine
  Conv1/Conv2/Conv3 x baseline/ours/legacy cases complete their full epoch
  budget with finite metrics and finish less than `.14 pp` below their own
  best validation, far inside the predeclared `<5 pp` rule.
- Scientific question: At the one-decade scheme-specific injected beta, do all
  nine Adam EqProp cases learn stably when BPTT and EqProp use identical
  learning-rate vectors, every bias is fixed at exact zero, and
  `T/K=4/4,6/6,8/8` is shared by architecture?
- Frozen setup: seed-0 ordinary MNIST; Conv1/2/3 baseline/ours/legacy;
  inherited zero-bias BPTT Adam vectors unchanged; injected beta Conv1
  `100/30/3`, Conv2 `100/10/.03`, Conv3 `100/3/.001` (base beta
  `100/7.5/.1875`, `100/.625/.0001171875`, and
  `100/.046875/2.44140625e-7`); centered frozen-current EqProp; true float64;
  `T/K=4/4,6/6,8/8`; Kaiming `[0,100]` weights; 10/30/30 epochs; endpoint
  read-noise sigma `0`; no official-test read; best-validation checkpoint
  selection.
- Coverage and integrity: Jean Zay UMG array `1030857_[0-4]` completed all
  five V100-32GB tasks with exit `0:0` in `02:17:19`--`05:02:24`; four packs
  ran two cases concurrently on one GPU and one pack ran Conv2 legacy alone.
  The 106 MB authoritative local copy has zero checksum differences from the
  remote tree. All nine canonical bundles, five pack summaries/receipts, nine
  run receipts, frozen source/config hashes, exact-zero bias rates and
  best/final checkpoint tensors, architecture-matched initialization,
  common train/validation split, common ten-epoch minibatch prefix, complete
  Conv2/Conv3 30-epoch order, zero-noise, and zero-test-read guards pass.
- Headline measurements: best/final validation is Conv1 baseline
  `96.18/96.04%`, ours `96.42/96.34%`, legacy `96.44/96.42%`; Conv2
  `97.22/97.22%`, `98.10/98.02%`, `98.44/98.42%`; and Conv3
  `97.76/97.68%`, `98.72/98.64%`, `99.08/99.02%`. Final drops from best are
  `.14/.08/.02 pp`, `0/.08/.02 pp`, and `.08/.08/.06 pp`, respectively.
- Interpretation: the one-decade tier is demonstrably training-stable under
  the exact zero-bias/shared-`T/K` Adam contract, so ordinary-MNIST training
  stability is not a reason by itself to reduce beta another decade. This does
  not freeze a paper beta: the retained direct gradient result is `213/216`
  at one decade versus `216/216` at two decades, and trained Conv3 baseline
  retains its beta-independent post-`T=8` residual deviation. The remaining
  decision is a signal-margin/read-noise versus local-gradient-fidelity
  tradeoff; a matched two-decade stability qualification would complete the
  other branch.
- Limitations and deviations: one seed and ordinary-MNIST selection accuracy;
  no BPTT accuracy comparison in this study; no harder-dataset transfer test.
  The user-directed launch retained three failed Conv3-initialization C0 rows
  from the one-decade gradient gate and did not reclassify them or the Conv3
  residual deviation as passing or paper-ready.
- Artifacts: [report](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-one-decade-ordinary-mnist-10-30-30ep-seed0-20260816-v1/analysis/report.md),
  [summary table](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-one-decade-ordinary-mnist-10-30-30ep-seed0-20260816-v1/analysis/summary.csv),
  [machine summary](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-one-decade-ordinary-mnist-10-30-30ep-seed0-20260816-v1/analysis/summary.json),
  [trajectories](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-one-decade-ordinary-mnist-10-30-30ep-seed0-20260816-v1/analysis/validation_accuracy_trajectories.png),
  and [verification](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-one-decade-ordinary-mnist-10-30-30ep-seed0-20260816-v1/analysis/verification.json).

## `perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-5em4-sgd-10ep-seed0-20260816-v1` — Conv2 SGD endpoint-read-noise repeat and optimizer-contract comparison

- Analyzed: 2026-08-16
- Evidence class: `ordinary_mnist_eqprop_read_noise_training_diagnostic`;
  exploratory and non-paper-facing.
- Outcome: `mixed`. Sigma `5e-4` causes no observed instability with either
  historical optimizer contract and preserves the same scheme ordering, but
  the clean-relative penalty magnitude changes with optimizer/LR contract.
- Scientific question: At the same clean-stable one-decade scheme-specific
  betas used by the Adam read-noise study, do Conv2 SGD runs remain finite at
  endpoint-noise sigma `5e-4`, and does the earlier qualitative robustness
  conclusion survive the optimizer/LR-contract change?
- Frozen setup: baseline/ours/legacy injected beta `100/10/.03` (base beta
  `100/.625/.0001171875`); historical parameter-wise SGD vectors; centered
  frozen-current EqProp; endpoint read-noise sigma `5e-4` and seed
  `2026081201`; true float64; `T=K=8`; seed-0 ordinary MNIST; exact common
  Kaiming `[0,100]` initialization, split, and ten minibatch orders; ten
  epochs; no official-test read. Existing one-decade clean SGD/Adam controls
  and matched noisy Adam runs are reused read-only.
- Coverage and integrity: Jean Zay `fmu@v100` canary `1029073_0` and
  production array `1029094_[0-2]` all complete with exit `0:0`. The three
  production tasks take `00:57:51/00:57:00/01:10:18`; all three canonical
  bundles, summaries, transport receipts, source/config hashes, optimizer
  identity, 206,280 noise draws per run, initialization/split/order hashes,
  finite histories, and zero-test-read guards pass. The authoritative local
  copy has zero checksum-dry-run differences from Jean Zay, and the final
  focused suite passes `29/29`.
- Headline measurements: SGD clean -> noisy final validation is baseline
  `96.52 -> 96.34%` (`-.18 pp`), ours `97.80 -> 97.46%` (`-.34 pp`), and
  legacy `97.52 -> 96.72%` (`-.80 pp`). Against the same later paired clean
  controls, Adam changes are `96.88 -> 96.78%` (`-.10 pp`),
  `97.92 -> 97.44%` (`-.48 pp`), and `98.40 -> 96.90%` (`-1.50 pp`). The
  noisy SGD-minus-Adam endpoints are `-.44/+.02/-.18 pp`.
- Interpretation: ours has the highest noisy final validation accuracy and
  legacy the largest clean-relative loss under both contracts. The earlier
  qualitative conclusion is therefore not Adam-specific, but its numeric
  size is optimizer/LR-contract dependent, especially for legacy. The
  original Adam extension's `-.22/-.42/-1.44 pp` values remain valid against
  its own earlier clean parent; its noisy endpoints are unchanged.
- Limitations: one seed; ordinary-MNIST diagnostic accuracy; optimizer and
  historical LR vector change together, so this is not an optimizer-algorithm
  ablation. Clean and noisy anchors come from separate retained launches and
  source archives rather than a single co-scheduled factorial, so small
  endpoint differences should not be overinterpreted.
- Artifacts: [report](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-5em4-sgd-10ep-seed0-20260816-v1/analysis/report.md),
  [terminal table](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-5em4-sgd-10ep-seed0-20260816-v1/analysis/terminal_summary.csv),
  [trajectories](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-5em4-sgd-10ep-seed0-20260816-v1/analysis/validation_accuracy_trajectories.png),
  [verification](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-5em4-sgd-10ep-seed0-20260816-v1/analysis/verification.json),
  and [collection receipt](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-5em4-sgd-10ep-seed0-20260816-v1/analysis/collection_validation_receipt.json).

## `perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-tk4-6-8-seed0-20260816-v1` — zero-bias shared-T/K EqProp--BPTT gradient qualification

- Analyzed: 2026-08-16
- Evidence class: `ordinary_mnist_learning_algorithm_gradient_qualification`;
  formal hyperparameter-selection diagnostic, non-paper-facing.
- Outcome: `mixed`. The original maximum tier fails broadly, the one-decade
  beta tier fails three Conv3-initialization first-convolution comparisons,
  and the two-decade tier passes all direct gradient comparisons. Trained
  Conv3 baseline remains residual-limited at the shared `T=8`, so the
  two-decade result is gradient-qualified but not an unqualified
  equilibrium-qualified paper handoff.
- Scientific question: Before matched long training, do centered
  frozen-current EqProp and BPTT weight gradients agree when both use exact
  zero biases and shared Conv1/2/3 `T/K=4/4,6/6,8/8`, and does the answer
  distinguish the clean-stable one- and two-decade beta candidates?
- Frozen setup: all nine seed-0 Conv1/Conv2/Conv3 x baseline/ours/legacy Adam
  source cases from the completed zero-bias ordinary-MNIST study;
  reconstructed initialization and best-validation checkpoints; four fixed
  16-example validation batches; true-float64 dynamics; centered
  frozen-current EqProp; exact common post-`T` state and exact `K`-step BPTT
  reference; Conv/Dense weights scored; cosine `>=.99` and symmetric norm
  delta `<=.10`; projected-residual p90 `<.01` recorded separately; no
  optimizer or official-test read.
- Coverage and integrity: each of the three tiers completes 72
  checkpoint-batch replays and 216 layer-batch comparisons after a completed
  Conv3-legacy semantic smoke. All six canonical bundles validate. All nine
  source bundles and cohort
  hashes reproduce; all source/checkpoint bytes and in-memory parameter
  tensors remain unchanged; every bias LR and initial/best tensor is exact
  zero; and all float64, iteration, common-state, and frozen-force guards pass.
- Headline measurements: at the original maximum tier Conv1 `1000/300/30`,
  Conv2 `1000/100/.3`, Conv3 `1000/30/.01`, only `159/216` rows and `9/18`
  checkpoint contexts pass; worst cosine is `.801608` and maximum norm delta
  `.533422`. At the one-decade injected-beta tier Conv1
  `100/30/3`, Conv2 `100/10/.03`, Conv3 `100/3/.001`, `213/216` rows pass.
  Conv3-baseline initialization C0 fails one batch (worst cosine `.987983`),
  and Conv3-ours initialization C0 fails two (worst cosine `.939168`, norm
  ratio `1.32312`, symmetric norm delta `.278178`). Every Conv1/Conv2 row and
  every best-checkpoint context passes. At the two-decade tier Conv1
  `10/3/.3`, Conv2 `10/1/.003`, Conv3 `10/.3/1e-4`, all `216/216` pass;
  global worst cosine is `.998877` and maximum norm delta `.039421`.
- Residual result: only trained Conv3 baseline fails, and only at the beta-
  independent post-`T=8` free endpoint. Layer_1 p90 is `.106--.156` and
  Layer_3 p90 `.0179--.0247` across the four batches. Its direct gradients
  pass at both beta tiers.
- Interpretation: clean training stability at the one-decade tier was not
  sufficient to make it universally gradient-faithful under the new matching
  contract. The two-decade tier is the supported candidate for the next
  Adam-first zero-bias ordinary-MNIST training qualification. Sealing paper
  checkpoints and reading the official MNIST test split remain blocked until
  that stability run completes and the Conv3-baseline truncated-T residual
  caveat is explicitly accepted or a new shared BPTT/EqProp T/K is
  requalified.
- Limitations: one seed, Adam-trained endpoint checkpoints, 64 validation
  examples, and two checkpoint roles rather than an online full-training
  trace. This measures gradient fidelity, not training stability or accuracy;
  SGD endpoint checkpoints require their own gate before an SGD-first launch.
- Artifacts: [review](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-tk4-6-8-seed0-20260816-v1/review.md),
  [maximum-beta summary](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-tk4-6-8-seed0-20260816-v1/production-max-beta/case_summary.csv),
  [maximum-beta layer table](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-tk4-6-8-seed0-20260816-v1/production-max-beta/layer_metrics.csv),
  [one-decade summary](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-tk4-6-8-seed0-20260816-v1/production/case_summary.csv),
  [one-decade layer table](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-tk4-6-8-seed0-20260816-v1/production/layer_metrics.csv),
  [two-decade summary](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-tk4-6-8-seed0-20260816-v1/production-two-decades/case_summary.csv),
  [two-decade layer table](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-tk4-6-8-seed0-20260816-v1/production-two-decades/layer_metrics.csv),
  and [validation receipt](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-tk4-6-8-seed0-20260816-v1/validation_receipt.json).

## `perfectdiode-conv1-legacy-zero-bias-conv-lr-times3-bptt-sgd-adam-10ep-seed0-20260816-v1` — Conv1 legacy factor-three Conv-LR counterfactual

- Analyzed: 2026-08-16
- Evidence class: `ordinary_mnist_learning_rate_counterfactual`; exploratory
  and non-paper-facing.
- Outcome: `negative` for adopting a general factor-three Conv-LR increase.
  SGD becomes worse and Adam is effectively unchanged, so the current Conv1
  legacy learning-rate handoff is not changed.
- Scientific question: Does the open factor-three upper-Conv plateau from the
  bounded, zero-bias Conv1 legacy-SGD surface justify tripling the wide-range
  `ConvWeight_0` learning rate when bias learning is excluded, and does the
  same transfer help Adam?
- Bounded-evidence verification: legacy-SGD ties at `94.50%` between
  `(rho_conv,rho_dense)=(.003,.0033333)` and the factor-three upper Conv point
  `(.009,.0033333)`, leaving that edge open.  Legacy-Adam is already bracketed:
  its selected `(.009,.01)` point reaches `95.42%`, while the factor-three
  upper neighbor `(.027,.01)` reaches `95.14%` (`-.28 pp`).  The prior result
  therefore motivated an SGD sentinel only; it was not a general LR handoff.
- Frozen setup: two seed-0 ordinary-MNIST Conv1 legacy BPTT/BP arms, SGD and
  Adam, ten epochs, accepted `T=K=4`, Kaiming weights projected to `[0,100]`,
  no official-test read, exact-zero Bias_0 LR/tensors, and unchanged Dense LR.
  Only ConvWeight_0 LR changes relative to the reviewed zero-bias controls:
  SGD `2.05753e-4 -> 6.17259e-4` and Adam
  `2.88917e-4 -> 8.66751e-4`.
- Coverage and integrity: exact-config sequential CUDA smokes and a concurrent
  two-process packing smoke passed.  `main` handles `@169/@168` completed with
  launcher exit `0`; both ten-epoch production bundles validate, share current
  initial-state SHA `594521a3...91ccb7b`, and have finite best/final checkpoint
  arrays with exact-zero Bias_0.  All four new/historical BPTT runs share the
  exact train/validation split and all ten minibatch-order hashes.
- Headline measurements: SGD best validation changes
  `95.92 -> 95.52%` (`-.40 pp`) and final changes
  `95.60 -> 95.52%` (`-.08 pp`); the times-three run is lower at every epoch
  with a mean epochwise delta of `-.548 pp`.  Adam best changes
  `96.44 -> 96.46%` (`+.02 pp`) and final changes
  `96.38 -> 96.46%` (`+.08 pp`); epochwise differences are mixed
  (`4/1/5` higher/tied/lower) with mean `-.010 pp`.
- Interpretation: the bounded SGD tie does not transfer into a better
  wide-range SGD setting.  Adam supplies no practically meaningful gain and
  is consistent with its bounded surface already being bracketed.  Retain the
  original Conv1 legacy SGD and Adam Conv LRs for bias-free work unless a new
  matched, multi-seed selection study establishes otherwise.
- Limitations and deviations: one seed and ordinary MNIST only.  Historical
  controls ran on Jean Zay V100s at `ac941dff`; counterfactuals ran on the local
  RTX 3090 from a dirty worktree based at `68d2f855`, although data orders and
  scientific configs are matched.  An unrelated GPU process started after
  launch and affected runtime only.  The production commands omitted explicit
  `--study-id`, so valid bundles have top-level `study_id=production` while
  their resolved configs retain the intended full study ID; this reporting
  label deviation is recorded and does not affect metrics or checkpoints.
- Artifacts: [review](../results/perfectdiode-conv1-legacy-zero-bias-conv-lr-times3-bptt-sgd-adam-10ep-seed0-20260816-v1/analysis/report.md),
  [comparison table](../results/perfectdiode-conv1-legacy-zero-bias-conv-lr-times3-bptt-sgd-adam-10ep-seed0-20260816-v1/analysis/comparison.csv),
  [epoch table](../results/perfectdiode-conv1-legacy-zero-bias-conv-lr-times3-bptt-sgd-adam-10ep-seed0-20260816-v1/analysis/epoch_metrics.csv),
  [bounded points](../results/perfectdiode-conv1-legacy-zero-bias-conv-lr-times3-bptt-sgd-adam-10ep-seed0-20260816-v1/analysis/bounded_evidence.csv),
  [trajectory plot](../results/perfectdiode-conv1-legacy-zero-bias-conv-lr-times3-bptt-sgd-adam-10ep-seed0-20260816-v1/analysis/plots/validation_accuracy_by_epoch.png),
  and [validation receipt](../results/perfectdiode-conv1-legacy-zero-bias-conv-lr-times3-bptt-sgd-adam-10ep-seed0-20260816-v1/analysis/validation_receipt.json).

## `perfectdiode-conv123-centered-float64-eqprop-beta-collapse-boundary-vs-1decade-sgd-adam-10ep-seed0-20260815-v1` — direct EqProp beta-boundary training test

- Analyzed: 2026-08-15
- Evidence class: ordinary-MNIST exploratory beta selection; non-paper-facing.
- Outcome: `mixed`. Seven of eighteen candidate/control pairs meet the frozen
  beta-collapse criterion, one additional candidate is practically unusable
  at chance, and ten do not confirm collapse. All eighteen one-decade-lower
  controls are stable.
- Scientific question: Do the original worst-layer cosine-`.99` candidate
  betas actually cause training collapse in Conv1, Conv2, and Conv3, and does
  moving one decade lower remove that instability under both SGD and Adam?
- Frozen setup: Conv1/Conv2/Conv3 x baseline/ours/legacy x SGD/Adam x original
  candidate/one-decade-lower beta, 36 cases total; seed-0 ordinary-MNIST
  55,000/5,000 split; ten epochs; exact fixed optimizer vectors; centered
  frozen-current EqProp; true float64; `T=K=8`; `[0,100]` Kaiming weights;
  zero endpoint read noise; and no official-test read. Candidate/control
  injected betas are Conv1 `1000/100`, `300/30`, `30/3`; Conv2 `1000/100`,
  `100/10`, `.3/.03`; and Conv3 `1000/100`, `30/3`, `.01/.001` for
  baseline, ours, and legacy. Collapse is an explicit
  `NonFiniteTrainingError` or a final-validation drop of at least `5.00 pp`
  from that run's best epoch.
- Coverage and integrity: all 18 two-case V100-32GB allocations in Jean Zay
  UMG array `1017012_[0-17]` completed with exit `0:0` in
  `00:31:45--02:21:11`. The exact staged source passed 43 focused local tests,
  31 staged Jean Zay tests, and `sbatch --test-only` job `1017005`. All 36
  one-batch scientific smokes passed sequentially on Trex. The first
  concurrent Trex pack is retained and excluded after a Trex-only Conv3 CUDA
  OOM; its Conv1 peer passed, and no production V100 had that failure. The
  authoritative local production copy has zero itemized differences under an
  rsync checksum dry-run against Jean Zay. All 36 canonical bundles, 36 run
  receipts, 18 pack receipts, source/config hashes, architecture-shared
  initialization hashes, split/order hashes, true-float64 guards, and zero
  official-test-read guards validate. Production contains 30 finite
  completions and six explicit scientific non-finite endpoints.
- Headline measurements: confirmed collapse occurs for all six Conv1/Conv2
  Adam candidates and Conv3-ours SGD. The non-finite endpoints are Conv1
  baseline Adam (`e2/b164`), ours Adam (`e1/b2137`), legacy Adam
  (`e3/b3047`); Conv2 baseline Adam (`e4/b3422`) and ours Adam (`e4/b2493`);
  and Conv3 ours SGD (`e1/b75`). Conv2-legacy Adam completes but drops
  `13.00 pp`, from `97.74%` best to `84.74%` final. Conv1-legacy SGD remains
  finite but flat at `10.00%`, finishing `85.62 pp` below its matched control;
  this is classified as degradation without internal collapse because it has
  no within-run drop. The remaining ten candidate pairs are stable under the
  frozen rule. Every one-decade-lower control completes ten finite epochs and
  finishes within `.28 pp` of its own best validation.
- Interpretation: the original cosine candidate is not a uniformly safe
  training beta, and its usable bound is optimizer-dependent. A one-decade
  margin is a clean-stable safety point for every tested MNIST contract, but
  the ten stable candidate pairs show that this experiment does not locate a
  universal maximum. The result supports scheme-, depth-, and optimizer-aware
  beta qualification. It does not select between the one-decade signal margin,
  the two-decade perturbative margin, or a matched-output-displacement rule.
- Limitations: one seed, ordinary MNIST, fixed LR vectors, and only ten epochs;
  therefore diagnostic rather than paper-facing. A harder dataset could
  narrow the safety margin. Conv3 baseline retains the predeclared
  residual-unqualified `T=K=8` cosine-boundary caveat. The excluded Trex OOM
  means smoke transport was sequential while production used demonstrated
  two-worker V100 packs; scientific configs were unchanged.
- Artifacts: [beta decision record](beta_study.md),
  [analysis report](../results/perfectdiode-conv123-centered-float64-eqprop-beta-collapse-boundary-vs-1decade-sgd-adam-10ep-seed0-20260815-v1/analysis/report.md),
  [pair table](../results/perfectdiode-conv123-centered-float64-eqprop-beta-collapse-boundary-vs-1decade-sgd-adam-10ep-seed0-20260815-v1/analysis/pairs.csv),
  [case table](../results/perfectdiode-conv123-centered-float64-eqprop-beta-collapse-boundary-vs-1decade-sgd-adam-10ep-seed0-20260815-v1/analysis/cases.csv),
  [full summary](../results/perfectdiode-conv123-centered-float64-eqprop-beta-collapse-boundary-vs-1decade-sgd-adam-10ep-seed0-20260815-v1/analysis/summary.json),
  [collection receipt](../results/perfectdiode-conv123-centered-float64-eqprop-beta-collapse-boundary-vs-1decade-sgd-adam-10ep-seed0-20260815-v1/analysis/collection_validation_receipt.json),
  and [trajectory plot](../results/perfectdiode-conv123-centered-float64-eqprop-beta-collapse-boundary-vs-1decade-sgd-adam-10ep-seed0-20260815-v1/analysis/validation_accuracy_trajectories.png).
- Provenance: frozen base commit `68d2f855`; staged source archive SHA-256
  `e0d4e1762c87e4247a8eccf09c34adb2ac540b9c3ef77a037fe2568327433577`;
  study-config SHA-256
  `ddc0f81e1ed2b7b6e548c5ef5ca780bb55dfe39bd3135804de827546dafb7326`;
  materialized-config-set SHA-256
  `009b77ba430eb72558dbb4927586d659c9a0e9d68b39f0e5aa838cd5345c7135`;
  Jean Zay job `1017012` under `umg@v100`.

## `perfectdiode-conv1-positive-vs-zero-bias-ordinary-mnist-seed0-20260815-v1` — matched Conv1 historical-bias ablation

- Analyzed: 2026-08-15
- Evidence class: `ordinary_mnist_bias_ablation`; diagnostic and non-paper-facing.
- Outcome: `positive` for the predeclared practical-similarity criterion. No
  material validation-accuracy effect is observed when the active historical
  positive-only bias is replaced by an exact-zero bias in this matched Conv1
  case.
- Scientific question: Does the current nonnegative-projected hidden-bias
  implementation materially change baseline perfect-diode Conv1 validation
  performance relative to fixing the otherwise identical Bias_0 at zero?
- Frozen setup: baseline Conv1, BPTT, plain SGD, seed 0, ordinary-MNIST
  55,000/5,000 train/validation split, ten epochs, `T=K=4`, input gain `40`,
  `[0,100]` Kaiming-initialized conductance weights, and weight LRs
  `C0=.142696,D=.0120238`. The only scientific difference is Bias_0 LR
  `.142696` versus `0`; both biases initialize exactly to zero. The official
  test is disabled. The predeclared threshold is an absolute best-validation
  difference no larger than `.5 pp`, conditional on the learned arm acquiring
  positive bias values.
- Coverage and integrity: both one-batch CPU smokes and both sequential Akib
  RTX 3080 production arms pass. Production ran under `nohup` PID `3957295`
  from `2026-08-15T14:35:25Z` to `14:44:47Z` with exit `0`. Both ten-epoch
  canonical bundles validate locally and remotely; all parameter tensors are
  exactly matched at initialization; train, validation, first-epoch, and full
  ten-epoch minibatch-order hashes match; every metric is finite; and the
  official test is never read. The 112-path local/remote content manifests
  share SHA-256 `5ff107f12d6f...8e9fc68557d`.
- Headline measurements: both arms select epoch 7 at `96.28%` best validation
  and finish at `96.00%`. Accuracy is identical in 7/10 epochs and differs by
  at most `.02 pp` in any epoch. Final loss is `.0923912023` with learned
  positive-only bias and `.0923836840` with fixed-zero bias; the maximum
  epoch-level loss difference is `9.574e-5`. At best/final, respectively,
  `94.85%/96.76%` of the learned arm's 12,544 Bias_0 entries are positive;
  the final range is `[0,.0163113]`. Every learned-bias checkpoint is
  nonnegative, and every control-bias checkpoint is exactly zero.
- Interpretation: this active comparison meets the practical-similarity
  criterion and supports describing the affected simulation matrix explicitly
  as bias-free or biases fixed at zero. It does not validate the separate
  signed/amplification-scaled learned-bias contract. Historical learned-bias
  and fixed-zero rows must not be pooled as one contract.
- Limitations: one architecture/scheme/optimizer and one seed; ordinary MNIST
  only; no uncertainty estimate or official-test read. This is neither a
  statistical equivalence test nor paper-facing accuracy evidence. Baseline
  has `voltage_amp=current_amp=1`, so this case tests active positive-only
  learning versus zero but not the historical depth-scaling confound.
- Artifacts: [bias state review](../results/perfectdiode-conv1-positive-vs-zero-bias-ordinary-mnist-seed0-20260815-v1/bias_state_review.md),
  [analysis report](../results/perfectdiode-conv1-positive-vs-zero-bias-ordinary-mnist-seed0-20260815-v1/analysis/report.md),
  [run table](../results/perfectdiode-conv1-positive-vs-zero-bias-ordinary-mnist-seed0-20260815-v1/analysis/summary.csv),
  [epoch table](../results/perfectdiode-conv1-positive-vs-zero-bias-ordinary-mnist-seed0-20260815-v1/analysis/epoch_metrics.csv),
  [bias audit](../results/perfectdiode-conv1-positive-vs-zero-bias-ordinary-mnist-seed0-20260815-v1/analysis/bias_checkpoint_stats.csv),
  [trajectory plot](../results/perfectdiode-conv1-positive-vs-zero-bias-ordinary-mnist-seed0-20260815-v1/analysis/accuracy_loss_trajectories.png),
  and [collection receipt](../results/perfectdiode-conv1-positive-vs-zero-bias-ordinary-mnist-seed0-20260815-v1/analysis/collection_validation_receipt.json).
- Provenance: frozen base commit `68d2f855`; source archive SHA-256
  `f10761eedb120ded72a9c6e728ebe2ed79e158cffa7d48ebf6c5a6460a74a9b3`;
  Akib launcher PID `3957295`; config SHAs `366054ee...af8f` and
  `4f17359f...7dc`.

## `perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1` — Conv1/2/3 Adam upper-weight-limit sensitivity

- Analyzed: 2026-08-15
- Evidence class: ordinary-MNIST exploratory / noncanonical.
- Outcome: `positive`. The tight `1e-4` training ceiling materially suppresses
  accuracy, increasingly so with depth for baseline and ours, and explains a
  substantial part of legacy's preliminary advantage.
- Scientific question: With each architecture/scheme's best observed Adam
  learning-rate vector held fixed, how does validation accuracy change when
  only the post-update upper weight limit increases from `1e-4` to `5e-4` or
  `1e-3`?
- Frozen setup: Conv1/2/3 perfect-diode BPTT at accepted `T=K=4/6/8`;
  baseline, ours, and legacy; Adam only; seed 0; ordinary MNIST; respectively
  10/30/30 epochs; exact-zero biases; exact architecture-shared checkpoints
  sampled once from `Uniform[1e-5,1e-4)`; fixed transferred raw LR vectors;
  lower projection limit `1e-5`; and no official-test read. The upper limit is
  the only within-surface factor.
- Coverage and integrity: Jean Zay R3 arrays `979501_[0-8]`, `979502_[0-8]`,
  and `979503_[0-8]` completed all 27 tasks with exit `0:0`, consuming
  `51.479` GPU-hours. All 27 canonical local bundles, task summaries, and
  validation receipts pass. Every run has complete epoch/checkpoint coverage,
  preserves its architecture's exact initial-state hash, shares the same MNIST
  split and first-epoch order, keeps biases exactly zero, obeys its configured
  interval, and records no official-test read. The local and remote production
  trees both contain `3,331,146,916` apparent bytes under `runs/`; their 108
  canonical files share aggregate SHA-256
  `8e27fab62fa13863d7561365e487ec9fbf24a7382e47d041c0e10cd50bf2dbee`.
- Headline measurements: best validation from `wmax=1e-4` to `1e-3` changes
  by Conv1 baseline/ours/legacy `+5.12/+2.10/+.90 pp`, Conv2
  `+7.92/+6.88/+1.36 pp`, and Conv3 `+11.94/+11.26/+4.18 pp`. Best values at
  `1e-3` are Conv1 `94.80/95.02/96.68%`, Conv2 `93.52/97.18/98.20%`, and
  Conv3 `89.60/95.80/97.86%` for baseline/ours/legacy.
- Mechanism and interpretation: final exact upper-endpoint occupancy across
  the nine surfaces falls from `0.89%`--`7.79%` at `wmax=1e-4` to
  `0.03%`--`1.64%` at `1e-3`. Together with the paired accuracy gains, this
  supports upper-bound projection throttling as the mechanism. The
  legacy-minus-ours best-accuracy gap contracts from `2.86 -> 1.66 pp` in
  Conv1, `6.54 -> 1.02 pp` in Conv2, and `9.14 -> 2.06 pp` in Conv3. The tight
  ceiling therefore explains much, but not all, of legacy's advantage.
  Legacy gains only `0.00/.04/.18 pp` from `5e-4` to `1e-3` for Conv1/2/3, so
  `5e-4` is effectively sufficient for legacy; deeper baseline/ours still
  benefit from `1e-3`.
- Limitations: one seed and ordinary MNIST make this diagnostic rather than
  paper-facing evidence. Architecture epoch budgets differ, so depth-level
  comparisons are descriptive. Within-surface deltas isolate the ceiling
  conditional on a fixed initializer and transferred LR vector; cross-scheme
  levels remain confounded by different LR vectors. Conv2 baseline and Conv3
  ours inherit open-upper-Conv-rho source searches, while Conv3 legacy inherits
  a noncanonical below-gate LR source. This study is not a new LR handoff.
- Artifacts: [analysis report](../results/perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1/analysis/report.md),
  [full table](../results/perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1/analysis/summary.csv),
  [surface table](../results/perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1/analysis/surface_summary.csv),
  [accuracy plot](../results/perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1/analysis/accuracy_vs_weight_max.png),
  [endpoint-occupancy table](../results/perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1/analysis/endpoint_occupancy.csv),
  and [collection/validation receipt](../results/perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1/analysis/collection_validation_receipt.json).

## `perfectdiode-conv13-centered-float64-eqprop-beta-qualification-1to2decades-10ep-seed0-20260814-v1` — Conv1/Conv3 clean-beta qualification

- Analyzed: 2026-08-14
- Evidence class: ordinary-MNIST exploratory / noncanonical.
- Outcome: `positive` for clean stability and `inconclusive` for choosing
  between one- and two-decade beta margins.
- Scientific question: For Conv1 and Conv3 baseline/ours/legacy, do centered
  true-float64 EqProp training trajectories remain finite and competitive at
  one and two decades below each independently measured worst-layer
  cosine-`.99` injected-beta boundary?
- Frozen setup: seed 0; ordinary MNIST deterministic 55k/5k train/validation
  split; exact current Adam vectors; ten epochs; `T=K=8`; centered
  frozen-current EqProp; endpoint read-noise sigma `0`; Kaiming initialization
  and projection interval `[0,100]`; explicit diode dictionaries; no
  official-test read. Injected beta is Conv1 baseline/ours/legacy
  `100/30/3` versus `10/3/.3`, and Conv3 `100/3/.001` versus
  `10/.3/1e-4`, at one versus two decades below the measured boundary.
- Coverage and integrity: Jean Zay UMG production array `980913_[0-5]`
  completed six V100-32GB tasks with exit `0:0`; each GPU ran one Conv1 and one
  Conv3 case concurrently. All 12 canonical bundles, 12 logical receipts, and
  six pack receipts validate. Every trajectory has ten finite epochs, an
  applied optimizer step, zero read-noise draws, and zero official-test
  evaluations. Initialization hashes match within architecture, and all cases
  share exact split and ten-epoch minibatch-order hashes. All copied result
  subtrees match Jean Zay by relative path and content manifest. Frozen source
  archive SHA-256 is `68c17dc5...fb69`; exact-config set SHA-256 is
  `b18c91be...fefc7`.
- Headline measurements: Conv1 best validation at one/two decades is baseline
  `96.24/96.22%`, ours `96.44/96.42%`, and legacy `96.64/96.62%`.
  Conv3 is `97.02/97.06%`, `98.40/98.44%`, and `98.46/98.42%`.
  Two-minus-one differences span only `-.04` to `+.04 pp`; all twelve runs
  remain finite and complete.
- Interpretation: both tested margins are clean-qualified. There is no clean
  accuracy evidence for selecting one over the other, so no beta is selected
  here. The one-decade tier retains a tenfold larger phase signal; the
  two-decade tier retains the larger small-perturbation margin and matches the
  prior Conv2 rule. A subsequent read-noise comparison, not this clean result,
  should resolve that tradeoff.
- Limitations: one seed, ordinary MNIST, ten epochs, and exploratory accuracy;
  not paper-facing evidence. Conv3 baseline's source boundary is cosine-only
  and residual-unqualified at `T=K=8` because its shared free state was
  under-relaxed. These training results show stability at the tested betas but
  do not repair that boundary qualification.
- Operational provenance: first live canary `980540` completed both packed GPU
  smokes but failed only in the post-run receipt validator, which read launch
  beta keys rather than trainer-runtime beta keys. The attempt is retained and
  excluded. The regression fix passed `58/58` focused tests and validated the
  preserved GPU outputs end to end; isolated replacement canary `980865`
  passed the scheduler and semantic gates before production.
- Artifacts: [analysis report](../results/perfectdiode-conv13-centered-float64-eqprop-beta-qualification-1to2decades-10ep-seed0-20260814-v1/analysis/report.md),
  [full table](../results/perfectdiode-conv13-centered-float64-eqprop-beta-qualification-1to2decades-10ep-seed0-20260814-v1/analysis/summary.csv),
  [one-vs-two-decade table](../results/perfectdiode-conv13-centered-float64-eqprop-beta-qualification-1to2decades-10ep-seed0-20260814-v1/analysis/one_vs_two_decades.csv),
  [trajectory plot](../results/perfectdiode-conv13-centered-float64-eqprop-beta-qualification-1to2decades-10ep-seed0-20260814-v1/analysis/validation_accuracy_trajectories.png),
  [scientific verification](../results/perfectdiode-conv13-centered-float64-eqprop-beta-qualification-1to2decades-10ep-seed0-20260814-v1/analysis/verification.json),
  and [collection verification](../results/perfectdiode-conv13-centered-float64-eqprop-beta-qualification-1to2decades-10ep-seed0-20260814-v1/analysis/collection_verification.json).

## `perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-3em4-5em4-10ep-seed0-20260813-v1` — Conv2 high endpoint-read-noise extension

- Analyzed: 2026-08-17
- Evidence class: `ordinary_mnist_eqprop_read_noise_training_diagnostic`;
  exploratory and non-paper-facing.
- Outcome: `positive` for a selective robustness gap and `negative` for
  literal noisy-training failure. All six runs remain finite and complete,
  while legacy has the largest clean-relative penalty at both added noise
  levels.
- Scientific question: Does the Conv2 one-decade beta remain trainable when
  endpoint read-noise sigma increases from the earlier `1e-4` cap to
  `3e-4` and `5e-4`, and does amplification scheme change the resulting
  degradation?
- Frozen setup: Conv2 baseline/ours/legacy; injected beta `100/10/.03`
  (base beta `100/.625/.0001171875`); current scheme-specific Adam LR
  vectors; centered frozen-current EqProp; true float64; `T=K=8`; seed-0
  ordinary MNIST; exact matched initialization, split, minibatch order, and
  endpoint-noise draws; ten epochs; noiseless validation; no official-test
  read. Noise is added only to copied equilibrium endpoint voltages used by
  the local EqProp gradient.
- Coverage and integrity: Jean Zay array `910848_[0-5]` completed all six
  tasks with exit `0:0` in about 58--60 minutes each. The authoritative local
  copy is byte-identical to the remote files. All six canonical bundles,
  summaries, receipt hashes, initialization/split/order/noise matching, finite
  histories, and zero-test-read guards validate.
- Headline measurements: final validation at sigma `3e-4` is baseline/ours/
  legacy `96.76/97.60/97.14%`; at `5e-4` it is
  `96.78/97.44/96.90%`. Against this study's matched clean anchors, the
  sigma-`5e-4` penalties are `-.22/-.42/-1.44 pp`. Legacy penalties are
  `-1.20/-1.44 pp` at the two added sigmas, versus baseline
  `-.24/-.22 pp` and ours `-.26/-.42 pp`.
- Interpretation: ours has the highest final validation at both new noise
  levels, and legacy is the most noise-sensitive on this matched seed-0
  surface. Together with the later SGD repeat, this qualitative result holds
  under both tested optimizer/LR contracts, although its magnitude changes.
- Limitations and deviations: one seed, one architecture, ordinary-MNIST
  diagnostic accuracy, and an absolute simulator-voltage noise scale that is
  not a calibrated hardware or ENOB claim. Small non-monotonic differences
  between adjacent noise levels are not a trend. The separate live canary was
  explicitly bypassed for this exploratory launch; six local semantic smokes
  and every production live guard passed.
- Artifacts: [analysis report](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-3em4-5em4-10ep-seed0-20260813-v1/analysis/report.md),
  [terminal table](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-3em4-5em4-10ep-seed0-20260813-v1/analysis/terminal_summary.csv),
  [noise curve](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-3em4-5em4-10ep-seed0-20260813-v1/analysis/final_validation_vs_noise.png),
  and [verification](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-3em4-5em4-10ep-seed0-20260813-v1/analysis/verification.json).

## `perfectdiode-conv12-bounded-kaiming-adam-lr-transfer-3ep-seed0-20260813-v1` — bounded-Kaiming raw-LR transfer diagnostic

- Analyzed: 2026-08-17
- Evidence class: ordinary-MNIST bounded-initializer diagnostic; explicit
  noncanonical cross-initializer LR transfer.
- Outcome: `negative` for the hypothesis that replacing bounded-uniform with
  bounded-Kaiming initialization explains the preliminary legacy advantage at
  the transferred Adam rates.
- Scientific question: If the bounded-uniform source cells' exact raw Adam LR
  vectors are held fixed, does changing only to gain-1 bounded-Kaiming
  initialization reduce the legacy-minus-baseline or legacy-minus-ours gap?
- Frozen setup: Conv1/Conv2 x baseline/ours/legacy; seed 0; ordinary MNIST;
  three epochs; Adam; accepted `T=K=4/6`; exact-zero biases; projection to
  `[1e-5,1e-4]`; architecture-shared bounded-Kaiming initial states; exact
  bounded-uniform source LR vectors transferred without reselection; matched
  dataset/order hashes; no official-test read.
- Coverage and integrity: all six Jean Zay `umg@v100` array `911653` tasks
  complete with exit `0:0`. All six canonical bundles validate, remain within
  the configured interval, preserve exact-zero biases, share the intended
  architecture-level initial states and data-order hashes, and select epoch 3
  as best.
- Headline measurements: Kaiming-minus-uniform final-accuracy changes are
  Conv1 baseline/ours/legacy `-.42/-.28/-.16 pp` and Conv2
  `-.22/-.32/-.18 pp`. Legacy margins change by only `+.04` to
  `+.26 pp`, and none shrinks. Final bound occupancy is baseline/ours/legacy
  `28.20/15.23/4.73%` for Conv1 and `45.50/22.04/15.04%` for Conv2.
- Interpretation: the scheme ordering and gaps reproduce almost unchanged,
  so initializer family alone does not explain legacy's advantage conditional
  on these transferred rates. The occupancy ordering is consistent with
  stronger projection pressure in baseline and ours, but it is associative
  rather than causal because schemes use different LR vectors.
- Limitations and deviations: the user explicitly authorized raw LR transfer
  across initializers, overriding the canonical protocol requirement to
  reselect rho independently. This is a three-epoch, one-seed, Conv1/Conv2,
  Adam-only diagnostic, not a bounded LR handoff, convergence result, or
  paper-facing accuracy result. Conv2 baseline retains the source search's
  open upper-Conv-rho caveat.
- Artifacts: [analysis report](../results/perfectdiode-conv12-bounded-kaiming-adam-lr-transfer-3ep-seed0-20260813-v1/analysis/report.md),
  [summary table](../results/perfectdiode-conv12-bounded-kaiming-adam-lr-transfer-3ep-seed0-20260813-v1/analysis/summary.csv),
  [legacy-margin table](../results/perfectdiode-conv12-bounded-kaiming-adam-lr-transfer-3ep-seed0-20260813-v1/analysis/legacy_margins.csv),
  [epoch trajectories](../results/perfectdiode-conv12-bounded-kaiming-adam-lr-transfer-3ep-seed0-20260813-v1/analysis/epoch_trajectory.csv),
  and [comparison plot](../results/perfectdiode-conv12-bounded-kaiming-adam-lr-transfer-3ep-seed0-20260813-v1/analysis/accuracy_comparison.png).

## `perfectdiode-conv1-bounded-uniform-scale1e4-adam-3ep-seed0-20260813-v1` — Conv1 bounded absolute-scale repeat

- Analyzed: 2026-08-13
- Evidence class: ordinary-MNIST exploratory / noncanonical.
- Outcome: `negative` for the hypothesis that small absolute conductance values
  materially cause the weak bounded Conv1 results through numerical
  imprecision.
- Scientific question: Do the recent bounded-uniform Conv1 Adam results improve
  when the full weight interval and initialization are scaled from
  `[1e-5,1e-4]` to `[1e-1,1]` and every nonzero raw LR is scaled by the same
  `1e4` factor?
- Frozen setup: baseline, ours, and legacy; seed 0; ordinary MNIST; three
  epochs; accepted `T=K=4`; exact-zero biases; the three source cells' selected
  raw Adam vectors; no official-test read. The shared initializer preserves
  every source quantile by multiplying the exact source checkpoint tensors by
  `1e4`; the dynamic range remains 10:1.
- Coverage and integrity: all three local RTX 3090 production bundles from
  `main` handle `@142` completed with launcher exit `0`, validate canonically,
  contain three epoch rows and four checkpoints, share initial-state SHA-256
  `71bfa790...6d7e9c`, obey `[.1,1]`, and match the source split and first-epoch
  order hashes. The unchanged three-case CUDA smoke passed. A preceding
  restricted-sandbox attempt failed before model construction because CUDA was
  hidden and is retained and excluded.
- Headline measurements: final small/scaled accuracies are baseline
  `88.84/88.84%`, ours `91.60/91.60%`, and legacy `95.42/95.40%`; deltas are
  `0.00/0.00/-.02 pp`. The largest absolute difference across all nine epoch
  accuracies is `.04 pp`. Final exact-bound occupancy changes only
  `29.8716 -> 29.8549%`, `19.2797 -> 19.2733%`, and
  `4.1844 -> 4.1610%` for baseline/ours/legacy.
- Interpretation: increasing the absolute weight and LR scale by four decades
  recovers no performance and leaves the epoch trajectories, endpoint
  occupancy, and `legacy > ours > baseline` ordering effectively invariant.
  This is evidence against material small-value numerical imprecision in these
  cells. The bounded 10:1 dynamic range, projection behavior, or another
  scale-invariant part of the bounded contract remains more plausible.
- Limitations: Conv1, Adam, seed 0, the selected vectors, and three epochs only;
  no SGD, depth, long-convergence, or medium-affine claim. Scaling weights and
  LRs together tests the complete absolute-scale transformation rather than
  isolating one arithmetic operation. Adam epsilon and solver tolerances remain
  fixed. Normalized final weights differ by `.796%`--`3.007%` RMS rather than
  bitwise reproducing, and the reviewed small-scale sources and repeat retain
  different code-state provenance.
- Artifacts: [review](../results/perfectdiode-conv1-bounded-uniform-scale1e4-adam-3ep-seed0-20260813-v1/review.md),
  [analysis report](../results/perfectdiode-conv1-bounded-uniform-scale1e4-adam-3ep-seed0-20260813-v1/analysis/report.md),
  [summary table](../results/perfectdiode-conv1-bounded-uniform-scale1e4-adam-3ep-seed0-20260813-v1/analysis/summary.csv),
  [epoch comparison](../results/perfectdiode-conv1-bounded-uniform-scale1e4-adam-3ep-seed0-20260813-v1/analysis/epochs.csv),
  and [accuracy plot](../results/perfectdiode-conv1-bounded-uniform-scale1e4-adam-3ep-seed0-20260813-v1/analysis/accuracy_comparison.png).

## `perfectdiode-conv2-fixed-uniform-init-wmax-sweep-sgd-adam-10ep-seed0-20260813-v1` — Conv2 upper-weight-limit sensitivity

- Analyzed: 2026-08-13
- Evidence class: ordinary-MNIST exploratory / noncanonical.
- Outcome: `positive`. The tight `1e-4` training ceiling materially suppresses
  accuracy, especially for ours, and explains a substantial part of legacy's
  preliminary advantage under the bounded initializer.
- Scientific question: With the best observed fixed learning-rate vector for
  each Conv2 scheme/optimizer, how does ten-epoch validation accuracy change
  when only the post-update upper weight limit is increased from `1e-4` to
  `{2e-4,5e-4,1e-3}`?
- Frozen setup: Conv2 perfect-diode BPTT at accepted `T=K=6`; baseline, ours,
  and legacy; SGD and Adam; seed 0; ordinary MNIST; ten epochs; exact-zero
  biases; one byte-identical checkpoint whose non-bias weights were sampled
  once from `Uniform[1e-5,1e-4)`; fixed transferred raw LR vectors; lower
  projection limit `1e-5`; and no official-test read. The upper limit is the
  only within-surface factor. User-authorized exploratory execution waived
  repeated scientific security and post-training T/K gates.
- Coverage and integrity: all 24 clean production cases from Jean Zay array
  `921005_[0-23]` completed with exit `0:0`, contain exactly ten epoch records
  and eleven checkpoints, and validate canonically and scientifically. All
  cases share the initializer, split, and minibatch-order hashes; final
  weights obey their configured intervals. The authoritative local and remote
  production-run trees have identical aggregate SHA-256
  `fbea6096d7aa17116f6c54b906905e3b0a3cb1803c956f6f3463ca915652af25`.
- Headline measurements: from `wmax=1e-4` to `1e-3`, epoch-10 accuracy changes
  by baseline `+4.94/+6.44 pp`, ours `+8.30/+9.18 pp`, and legacy
  `+3.20/+1.34 pp` for SGD/Adam. The legacy-minus-ours gap contracts from
  `8.34` to `3.24 pp` with SGD and from `8.88` to `1.04 pp` with Adam.
  Highest epoch-10 values are baseline-SGD `86.92%` at `5e-4`, baseline-Adam
  `90.46%` at `1e-3`, ours-SGD `94.20%` at `1e-3`, ours-Adam `96.74%` at
  `1e-3`, legacy-SGD `97.46%` at `5e-4`, and legacy-Adam `97.78%` at `1e-3`.
- Mechanism and interpretation: final upper-endpoint occupancy falls from
  `0.99%`--`5.27%` across surfaces at `1e-4` to `0.00%`--`0.45%` at `1e-3`.
  The paired accuracy gains support projection throttling as a major reason
  ours and baseline looked worse at the tight ceiling. Baseline-SGD and
  legacy-SGD are effectively flat from `5e-4` to `1e-3`; ours-SGD/Adam and
  baseline-Adam still gain about half a point, so `1e-3` is the best single
  tested ceiling for follow-up, without proving that it is globally optimal.
- Limitations: one seed and ordinary MNIST make this diagnostic rather than
  paper-facing evidence. Within-surface deltas isolate the ceiling conditional
  on a fixed initializer and LR vector; cross-scheme levels remain confounded
  by different LR vectors. Baseline-Adam's source rho search remains open at
  its upper Conv edge. The initializer's `1e-4` sampling ceiling is fixed even
  when the subsequent training ceiling is larger.
- Operational provenance: array `917600` failed before training because of a
  submitted archive-path defect. Array `920119` exposed double array-index
  selection; its cancelled task-0 partial is retained under `failed_attempts/`
  and excluded. Full recovery canary `917728_0` and two-index selector canary
  `920841_[0-1]` passed before clean production `921005`.
- Artifacts: [report](../results/perfectdiode-conv2-fixed-uniform-init-wmax-sweep-sgd-adam-10ep-seed0-20260813-v1/analysis/report.md),
  [full table](../results/perfectdiode-conv2-fixed-uniform-init-wmax-sweep-sgd-adam-10ep-seed0-20260813-v1/analysis/summary.csv),
  [surface table](../results/perfectdiode-conv2-fixed-uniform-init-wmax-sweep-sgd-adam-10ep-seed0-20260813-v1/analysis/surface_summary.csv),
  [accuracy plot](../results/perfectdiode-conv2-fixed-uniform-init-wmax-sweep-sgd-adam-10ep-seed0-20260813-v1/analysis/accuracy_vs_weight_max.png),
  [occupancy plot](../results/perfectdiode-conv2-fixed-uniform-init-wmax-sweep-sgd-adam-10ep-seed0-20260813-v1/analysis/endpoint_occupancy_vs_weight_max.png),
  and [collection/validation receipt](../results/perfectdiode-conv2-fixed-uniform-init-wmax-sweep-sgd-adam-10ep-seed0-20260813-v1/analysis/collection_validation_receipt.json).

## `perfectdiode-conv12-directed-exploratory-lr-search-seed0-20260812-v1` — Conv1/Conv2 directed bounded-uniform LR map

- Analyzed: 2026-08-13
- Evidence class: ordinary-MNIST exploratory / noncanonical.
- Outcome: `positive` for locating the Conv2 SGD basins and `mixed` for exact
  endpoint occupancy as a learning-rate diagnostic. This is not an LR handoff
  or paper-facing accuracy.
- Frozen setup: Conv1/Conv2 baseline, ours, and legacy with SGD/Adam; exact
  architecture-shared `Uniform[1e-5,1e-4)` initializers; projection to
  `[1e-5,1e-4]`; exact-zero bias rates; seed 0; three epochs; Conv1 `T=K=4`
  and Conv2 `T=K=6`; no official-test read. The exploratory extension disables
  canaries, scientific safety rejection, and post-training T/K rejection.
- Coverage: all 12 shards, 12 terminal semantic receipts, and 257/257 directed
  numeric cells are collected locally with zero nonfinite outcome. The join
  with 114 prior same-contract numeric cells contains 371 distinct coordinates
  and no numeric duplicate. The archival checker passes 12/12 authorities and
  257/257 directed bundles.
- Headline result: reducing rates moves Conv2 baseline-SGD from the prior
  `49.34%` maximum to a bracketed `82.84%` at
  `(rho_conv,rho_dense)=(3.70370e-5,4.11523e-5)` with `20.28%` final clipping,
  and ours-SGD from `69.50%` to a bracketed `85.72%` at
  `(3.33333e-4,1.23457e-4)` with `8.77%` clipping. Thus the earlier Conv2 SGD
  rates were too high for this fixed weight interval.
- Adam result: ours-Adam improves only `+.32 pp` to a bracketed `87.70%` at
  `(0.027,0.01)`. Baseline-Adam improves only `+.18 pp` to `84.54%` at the
  maximum Conv rho `0.243`; it remains formally open but already has `46.83%`
  final clipping. Legacy retains the best observed accuracy: `95.42%` with
  Adam in both architectures.
- Range result: ten of twelve combined surfaces are bracketed. Conv1
  legacy-SGD has a tied upper-Conv plateau, and Conv2 baseline-Adam remains
  open toward higher Conv rho. These are the only scientifically motivated
  sparse extensions if complete boundary closure is required.
- Clipping interpretation: combined within-surface Pearson coefficients span
  `-.744` to `+.449` and Spearman coefficients span `-.645` to `+.623`.
  Clipping is therefore not a universal proxy for LR quality. Conv2 ours-SGD
  is the clearest case where entering the correct lower-rate basin both raises
  accuracy and reduces clipping (`33.07% -> 8.77%`); baseline-Adam instead
  gains slightly while clipping increases. Correlations are descriptive and
  noncausal, and combined Adam rows are host/runtime-confounded across studies.
- Operational note: the jobs finished successfully, but the conversational
  watchdog paused after its second checkpoint and delayed collection/reporting
  until the following day. This affected latency, not the scientific bundles.
  Future exploratory runs must report terminal receipts immediately and defer
  deep post-transfer auditing until after the preliminary result.
- Artifacts: [review](../results/perfectdiode-conv12-directed-exploratory-lr-search-seed0-20260812-v1/review.md),
  [combined report](../results/perfectdiode-conv12-directed-exploratory-lr-search-seed0-20260812-v1/analysis/combined/report.md),
  [surface table](../results/perfectdiode-conv12-directed-exploratory-lr-search-seed0-20260812-v1/analysis/combined/combined_surface_summary.csv),
  [cell table](../results/perfectdiode-conv12-directed-exploratory-lr-search-seed0-20260812-v1/analysis/combined/combined_cells.csv),
  [correlations](../results/perfectdiode-conv12-directed-exploratory-lr-search-seed0-20260812-v1/analysis/combined/combined_correlations.csv),
  and [accuracy/clipping plot](../results/perfectdiode-conv12-directed-exploratory-lr-search-seed0-20260812-v1/analysis/combined/accuracy_vs_final_clipping_combined.png).

## `perfectdiode-conv2-centered-float64-eqprop-read-noise-beta-down-1to2decades-10ep-seed0-20260812-v1` — Conv2 lower-beta read-noise training

- Analyzed: 2026-08-13
- Evidence class: `ordinary_mnist_eqprop_read_noise_training_diagnostic`
- Evidence tier: exploratory / noncanonical.
- Outcome: `positive` for a selective robustness gap, `negative` for literal
  legacy training failure. All lower-beta clean controls and noisy cases stay
  finite and complete ten epochs.
- Scientific question: Can beta be selected by one relative rule across
  amplification schemes while keeping clean centered-float64 EqProp stable
  and exposing a larger endpoint-read-noise penalty for legacy than for ours?
- Frozen setup: Conv2; seed-0 ordinary-MNIST 55k/5k split; current selected
  scheme-specific Adam LR vectors; shared initialization and ten minibatch
  orders; centered frozen-current EqProp; true float64; `T=K=8`; noiseless
  validation; matched endpoint-noise seed/draws; no official-test read. The
  one-decade injected betas are baseline/ours/legacy `100/10/.03`; the
  two-decade betas are `10/1/.003`. These are common nominal 10% and 1%
  safety-margin rules relative to the independently measured Conv2
  worst-layer cosine-`.99` upper boundaries (`>=1000/100/.3`), not
  accuracy-tuned betas; the open baseline boundary makes its margins at least
  one and two decades.
- User-truncated coverage: clean, sigma `1e-5`, and sigma `1e-4` for both beta
  tiers and all three schemes (`18` cases). Six materialized sigma-`1e-6`
  configs were never launched. A partial index-10 bundle interrupted solely
  to change launcher scope is retained and excluded; its clean exact
  replacement is canonical.

| Beta margin | sigma | Baseline final change | Ours final change | Legacy final change | Legacy excess penalty vs ours |
|---|---:|---:|---:|---:|---:|
| one decade | `1e-5` | `-.04 pp` | `+.02 pp` | `-.18 pp` | `.20 pp` |
| one decade | `1e-4` | `-.14 pp` | `-.06 pp` | `-.82 pp` | `.76 pp` |
| two decades | `1e-5` | `-.06 pp` | `-.14 pp` | `-.94 pp` | `.80 pp` |
| two decades | `1e-4` | `-.12 pp` | `-.76 pp` | `-1.92 pp` | `1.16 pp` |

- Interpretation: the two-decade rule gives stable clean controls and the
  clearest noise separation. Clean legacy leads ours by `.54 pp`, but ours
  leads legacy under noise by `.26 pp` at `1e-5` and `.62 pp` at `1e-4`.
  Smaller beta increases sensitivity to fixed absolute read noise, but the
  effect is disproportionate for legacy, consistent with prior layerwise
  displacement/read-noise diagnostics. This study does not itself localize
  the degradation to a layer.
- Recommendation: use one relative beta rule, not one common numeric beta nor
  independently accuracy-picked values. The exploratory paper candidate is
  the two-decade rule (`10/1/.003` in Conv2) with clean versus sigma `1e-4`;
  sigma `1e-5` is a milder supplementary point. Freeze and confirm under the
  final ordinary-MNIST zero-bias/shared-`T/K` contract before the sealed
  official-test evaluation. Treat `1e-4` as a controlled stress level unless
  hardware calibration justifies it. The supported claim is
  robustness/degradation, not failure.
- Integrity: `18/18` canonical bundles validate with zero failures; local and
  Trex hashes match for all 72 canonical manifest/status/metrics/result files;
  initialization, split, and minibatch-order guards pass; frozen-source tests
  passed `17/17`, representative CUDA smokes `12/12`, and final runner,
  correctness-guard, and analysis tests `21/21`.
- Limitations: one seed, ten epochs, ordinary MNIST, absolute uncalibrated
  noise, and post-exploration choice of the recommended margin/noise point.
- Artifacts: [review](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-beta-down-1to2decades-10ep-seed0-20260812-v1/review.md),
  [terminal table](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-beta-down-1to2decades-10ep-seed0-20260812-v1/analysis/terminal_summary.csv),
  [penalty plot](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-beta-down-1to2decades-10ep-seed0-20260812-v1/analysis/noise_penalty_by_beta_tier.png),
  [training curves](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-beta-down-1to2decades-10ep-seed0-20260812-v1/analysis/terminal_training_side_by_side.png),
  and [verification](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-beta-down-1to2decades-10ep-seed0-20260812-v1/analysis/verification.json).

## `perfectdiode-conv123-legacy-centered-float64-beta-1em3-to-1e3-tk8-seed0-20260812-v1` — legacy Conv1/2/3 beta fidelity and displacement

- Analyzed: 2026-08-12
- Evidence class: `ordinary_mnist_exploratory_checkpoint_gradient_diagnostic`
- Outcome: `negative`. Legacy's centered true-float64 gradient loses
  all-layer agreement with same-T/K BPTT at progressively smaller injected
  beta as depth grows.
- Scientific question: On the same T8 beta grid already used for baseline and
  ours, what is the largest sampled injected beta whose legacy EqProp gradient
  has minimum all-weight-layer cosine `.90`, `.95`, or `.99`, and what signed
  state displacement accompanies it?
- Frozen setup: trained legacy best-validation Conv1/2/3 checkpoints; seed-0
  ordinary-MNIST batch 0 with 16 fixed examples; centered frozen-current
  phases from a common post-free-state start; `T=K=8`; injected
  `B={.001,.003,.01,.03,.1,.3,1,3,10,30,100,300,1000}`. Base beta is
  `B/16`, `B/256`, and `B/4096` for Conv1/2/3. Biases remain active in
  dynamics but are excluded from scoring.

| Architecture | max B, cosine >= .90 | max B, cosine >= .95 | max B, cosine >= .99 |
|---|---:|---:|---:|
| Conv1 legacy | `300` | `100` | `30` |
| Conv2 legacy | `3` | `1` | `.3` |
| Conv3 legacy | `.1` | `.03` | `.01` |

- Headline boundary cosines: Conv1 `.913462/.977821/.994894` at
  `B=300/100/30`; Conv2 `.910511/.976413/.990301` at `B=3/1/.3`; Conv3
  `.933893/.983855/.991996` at `B=.1/.03/.01`. Every lower sampled beta
  through each boundary also passes, and all nine boundaries are
  equilibrium-residual-qualified.
- Interpretation: at the `.99` gate, legacy permits only `30/.3/.01` versus
  ours `300/100/30` across Conv1/2/3. The signed displacement curves show
  rapidly growing downstream perturbations beyond these boundaries. This is
  a checkpoint-gradient diagnostic, not evidence about high-beta training
  accuracy.
- Integrity: authoritative GPU smoke and `39/39` local RTX 3090 production
  bundles validate; handle `@118` exited `0`. The analysis reconciles 39
  curve, 117 per-layer cosine, 117 per-layer signed-displacement, and nine
  threshold rows. Source/checkpoint/cohort/parameter/common-start/frozen-force/
  iteration/dtype/native-arithmetic guards pass. No optimizer was constructed
  or stepped and no official-test example was read.
- Operational exclusions: the restricted-shell smoke could not see CUDA and
  produced no bundle; detached handle `@117` then used the wrong workdir and
  exited `127` before starting any case. The unchanged absolute-path retry
  `@118` is the sole production authority.
- Limitations/deviations: one seed/checkpoint and one minibatch; maxima are
  sampled-grid values rather than interpolated crossings. The reusable
  runner's internal study ID is source-compatible rather than equal to the
  enclosing study ID; the exact inventory and summary bind the extension.
- Artifacts: [review](../results/perfectdiode-conv123-legacy-centered-float64-beta-1em3-to-1e3-tk8-seed0-20260812-v1/review.md),
  [threshold table](../results/perfectdiode-conv123-legacy-centered-float64-beta-1em3-to-1e3-tk8-seed0-20260812-v1/analysis/max_beta_by_cosine_threshold.csv),
  [per-layer cosine](../results/perfectdiode-conv123-legacy-centered-float64-beta-1em3-to-1e3-tk8-seed0-20260812-v1/analysis/layer_cosine.csv),
  [per-layer displacement](../results/perfectdiode-conv123-legacy-centered-float64-beta-1em3-to-1e3-tk8-seed0-20260812-v1/analysis/layer_displacement.csv), and
  [figure](../results/perfectdiode-conv123-legacy-centered-float64-beta-1em3-to-1e3-tk8-seed0-20260812-v1/analysis/float64_cosine_and_displacement_vs_beta.png).

## `perfectdiode-conv123-baseline-ours-centered-float64-beta-1em3-to-1e3-tk8-seed0-20260812-v1` — matched Conv1/2/3 beta fidelity and displacement

- Analyzed: 2026-08-12
- Evidence class: `ordinary_mnist_exploratory_checkpoint_gradient_diagnostic`
- Outcome: `mixed`. Baseline remains above worst-layer cosine `.99` at the
  tested `B=1000` ceiling for every architecture, while ours has a clear
  depth-dependent finite-perturbation boundary; Conv3 baseline is not
  equilibrium-qualified at `T=K=8`.
- Scientific question: On one common low-to-high beta grid, what is the
  largest sampled injected beta whose centered true-float64 EqProp gradient
  has minimum all-weight-layer cosine `.90`, `.95`, or `.99` against
  same-T/K BPTT, and what state displacement accompanies it?
- Frozen setup: trained best-validation Conv1/2/3 baseline and ours
  checkpoints; seed-0 ordinary MNIST batch 0 with 16 fixed examples;
  `T=K=8`; centered frozen-current phases from a common post-free-state start;
  injected `B={.001,.003,.01,.03,.1,.3,1,3,10,30,100,300,1000}`. Baseline
  base beta is `B`; ours uses `B/4`, `B/16`, and `B/64` for Conv1, Conv2, and
  Conv3. Biases remain active in dynamics but are excluded from scoring.

| Architecture | Scheme | max B, cosine >= .90 | max B, cosine >= .95 | max B, cosine >= .99 |
|---|---|---:|---:|---:|
| Conv1 | baseline | `>=1000` | `>=1000` | `>=1000` |
| Conv1 | ours | `>=1000` | `>=1000` | `300` |
| Conv2 | baseline | `>=1000` | `>=1000` | `>=1000` |
| Conv2 | ours | `>=1000` | `300` | `100` |
| Conv3 | baseline | `>=1000`* | `>=1000`* | `>=1000`* |
| Conv3 | ours | `100` | `100` | `30` |

- `>=1000` is an open tested edge, not an estimated crossing. Conv3
  baseline's marked values are cosine-only: the shared free state has
  float64 residual p90 `1.75151` and a hard failure at every beta, leaving no
  residual-qualified point. Every other reported boundary passes all float64
  residual gates.
- Headline cosines at non-baseline limits: ours reaches `.979564` at Conv1
  `B=1000` and `.994410` at `B=300`; `.911775/.984057/.995390` at Conv2
  `B=1000/300/100`; and `.967556/.995566` at Conv3 `B=100/30`. The limiting
  layers are C0, C1, and C2, respectively.
- Interpretation: Ours' `.99` boundary moves downward with depth from
  `B=300` to `100` to `30`, matching the increase in layerwise displacement.
  This is a read-only BPTT-checkpoint diagnostic, not evidence that high-beta
  training is valid or beneficial.
- Integrity: local RTX 3090 launcher exit `0`; authoritative GPU smoke and
  `78/78` production bundles validate. The analysis reconciles 78 curve rows,
  234 per-layer cosine rows, 234 per-layer signed-displacement rows, and 18
  threshold rows. Checkpoint bytes, source files, cohort, parameter tensors,
  phase starts, dtype, and native arithmetic guards pass. No optimizer was
  constructed or stepped and no official-test example was read. An
  interrupted CPU smoke with no result is retained and excluded.
- Limitations/deviations: one seed/checkpoint and one minibatch; sampled
  thresholds are not interpolated crossings. The reusable runner's internal
  study ID is source-compatible rather than equal to the enclosing study ID;
  the exact inventory and analysis summary bind the evidence.
- Artifacts: [review](../results/perfectdiode-conv123-baseline-ours-centered-float64-beta-1em3-to-1e3-tk8-seed0-20260812-v1/review.md),
  [threshold table](../results/perfectdiode-conv123-baseline-ours-centered-float64-beta-1em3-to-1e3-tk8-seed0-20260812-v1/analysis/max_beta_by_cosine_threshold.csv),
  [per-layer cosine](../results/perfectdiode-conv123-baseline-ours-centered-float64-beta-1em3-to-1e3-tk8-seed0-20260812-v1/analysis/layer_cosine.csv),
  [per-layer displacement](../results/perfectdiode-conv123-baseline-ours-centered-float64-beta-1em3-to-1e3-tk8-seed0-20260812-v1/analysis/layer_displacement.csv), and
  [figure](../results/perfectdiode-conv123-baseline-ours-centered-float64-beta-1em3-to-1e3-tk8-seed0-20260812-v1/analysis/float64_cosine_and_displacement_vs_beta.png).

## `perfectdiode-conv3-exploratory-lr-ladder-seed0-20260811-v1` — Conv3 baseline and ours exploratory LR ladder

- Analyzed: 2026-08-12
- Evidence class: `ordinary_mnist_exploratory`
- Evidence tier: `exploratory_noncanonical`
- Evidence label: **EXPLORATORY / NONCANONICAL ordinary-MNIST diagnostic.**
- Outcome: `positive`. Ours has a higher observed accuracy frontier than
  baseline within both matched optimizer/runtime pairs and is better at most
  shared grid coordinates, although neither the study nor its open boundaries
  authorize an LR selection.
- Scientific question: How do baseline and ours respond to a broad low-to-high
  Conv3 LR grid when occupancy and the other scientific safety rules are
  report-only or disabled?
- Frozen setup: Seed-0 ordinary MNIST; baseline/ours x SGD/Adam; ordered `8x8`
  `(rho_conv,rho_dense)` grid; 64 independent three-epoch restarts per
  surface; exact shared `Uniform[1e-5,1e-4)` initializer SHA-256
  `8ebe9dd916e1e9299c31053e2bb37b4f5121f8322f26af7746a92875e555438f`;
  projection to `[1e-5,1e-4]`; zero biases and bias LRs; `T=K=8`; fixed
  split/shuffle seed 0. No official-test examples were read.
- Coverage and integrity: All `4/4` surfaces and `256/256` declared cells are
  canonical, numeric, finite, and reconciled. Four terminal receipts, nine
  exact collected roots, both launchers, three target-gate attempt trees,
  source `result.json` hashes, CSV/Markdown regeneration, and all four PNGs
  validate. The inventory retains 13 excluded operational/non-authoritative
  bundles and exactly two canceled zero-artifact scheduler attempts.

| Scheme / optimizer | Target | Highest observed rho pair | Actual weight LR vector `(C0,C1,C2,D)` | Accuracy | Loss | Exact final either-bound occupancy | Median projection efficiency | Range status |
|---|---|---:|---|---:|---:|---:|---:|---|
| baseline / SGD | Akib | `(3.7037037e-5,1.3717421e-5)` | `(9.476769e-7,2.8743254e-4,2.6388743e-4,2.6545789e-6)` | `72.64%` | `.286757` | `60.79%` | `.2928` | open toward lower dense rho |
| ours / SGD | Akib | `(1.2345679e-5,4.1152263e-5)` | `(6.4107133e-8,1.9908568e-5,1.8416615e-5,1.6719532e-6)` | **`76.84%`** | `.249718` | `18.97%` | `.9418` | locally bracketed |
| baseline / Adam | Trex | `(1e-3,3.3333333e-3)` | `(6.0547261e-8,6.1436962e-8,6.1456227e-8,2.0994218e-7)` | `76.16%` | `.279628` | `69.75%` | `.2803` | locally bracketed |
| ours / Adam | Trex | `(9e-3,3.3333333e-3)` | `(5.4491566e-7,5.4897250e-7,5.4913819e-7,2.0396740e-7)` | **`80.90%`** | `.229190` | `38.55%` | `.3887` | open toward higher Conv rho |

- Within-optimizer comparison: The observed ours-minus-baseline frontier gain
  is `+4.20 pp` for SGD and `+4.74 pp` for Adam. At each baseline maximum's
  same rho coordinate, ours is `+3.20 pp` for SGD and `+3.32 pp` for Adam;
  at each ours maximum, the matched gain is `+7.58 pp` and `+5.36 pp`.
  Ours is better/tied/worse over the full shared grid in `42/13/9` SGD cells
  and `53/0/11` Adam cells. Baseline and ours share initialization, cohort,
  order, target, and validated host/runtime within each optimizer.
- Endpoint occupancy at the maxima: Per-parameter `(C0,C1,C2,D)` percentages
  are `(94.44,88.89,94.53,12.71)` baseline-SGD,
  `(29.25,31.20,28.85,3.71)` ours-SGD,
  `(100.00,87.13,97.12,32.32)` baseline-Adam, and
  `(72.22,44.01,64.95,5.75)` ours-Adam. This is exact final-checkpoint
  either-bound occupancy, not the percentage clipped at any time during
  training.
- Clipping diagnostics: Accuracy versus pooled exact endpoint occupancy has
  Pearson/Spearman `+.381/+.386` for baseline-SGD, `-.098/-.171` for
  ours-SGD, `+.914/+.928` for baseline-Adam, and `+.774/+.610` for
  ours-Adam (`n=64` each). Descriptive and noncausal: `rho_conv` and
  `rho_dense` jointly vary, so the correlations do not isolate an effect of
  clipping on accuracy.
- Range/LR status: The unique baseline-Adam and ours-SGD maxima are locally
  bracketed by four worse axial neighbors. Baseline-SGD is open toward lower
  dense rho, while ours-Adam peaks at the highest tested Conv rho `0.009` and
  is open upward. These are highest observed accuracies, not selected learning
  rates.
- Target assignment: SGD ran on Akib RTX 3080/PyTorch 2.5.1 and Adam on Trex
  RTX 5090/PyTorch 2.11.0. Within-optimizer scheme comparisons are matched;
  SGD-versus-Adam differences are host/runtime-confounded and must not be
  interpreted causally as optimizer effects.
- Operational provenance: Jean Zay jobs `844032` and `844396` were canceled
  before allocation with exit `0:0`, elapsed `00:00:00`, no start, and zero
  scientific artifacts. The checksum-converged Akib/Trex shards are the sole
  scientific authority.
- Interpretation: The broad grid confirms that the previous narrow Conv3
  searches understated baseline/ours performance. Ours reaches higher
  accuracy with much lower endpoint occupancy at matched high-performing
  coordinates. Positive Adam occupancy correlations are consistent with the
  lowest rates undertraining while both learning and boundary contact rise;
  they are not evidence that clipping itself improves accuracy. The open
  ours-Adam edge motivates a later controlled higher-Conv-rho extension.
- Limitations and protocol deviations: Single seed, ordinary MNIST, three
  epochs, broad joint-rho grid, deliberately disabled canaries/scientific
  safety rejections/post-training T/K, and a cross-optimizer host/runtime
  confound. This study creates neither a canonical LR handoff nor paper-facing
  evidence.
- Results and analysis: [report](../results/perfectdiode-conv3-exploratory-lr-ladder-seed0-20260811-v1/analysis/exploratory_lr_ladder/report.md),
  [surface table](../results/perfectdiode-conv3-exploratory-lr-ladder-seed0-20260811-v1/analysis/exploratory_lr_ladder/surfaces.csv),
  [cell table](../results/perfectdiode-conv3-exploratory-lr-ladder-seed0-20260811-v1/analysis/exploratory_lr_ladder/cells.csv),
  [collection validation](../results/perfectdiode-conv3-exploratory-lr-ladder-seed0-20260811-v1/provenance/local_collection_validation.json),
  [accuracy heatmaps](../results/perfectdiode-conv3-exploratory-lr-ladder-seed0-20260811-v1/analysis/exploratory_lr_ladder/plots/final_validation_accuracy_heatmaps.png),
  [endpoint-occupancy heatmaps](../results/perfectdiode-conv3-exploratory-lr-ladder-seed0-20260811-v1/analysis/exploratory_lr_ladder/plots/endpoint_bound_occupancy_heatmaps.png),
  [diagnostic scatter](../results/perfectdiode-conv3-exploratory-lr-ladder-seed0-20260811-v1/analysis/exploratory_lr_ladder/plots/accuracy_vs_diagnostics.png),
  and [study config](../configs/conv/perfectdiode_conv3_exploratory_lr_ladder_seed0_20260811_v1.json).

## `perfectdiode-conv12-baseline-ours-eqprop-higher-beta-seed0-20260812-v1` — Conv1/Conv2 higher-beta true-dtype EqProp

- Analyzed: 2026-08-12
- Evidence class: `ordinary_mnist_learning_algorithm_gradient_diagnostic`
- Evidence tier: `exploratory_noncanonical`
- Outcome: `mixed`. Centered baseline remains all-layer faithful through
  injected beta `1000` for both Conv1 and Conv2. Centered ours passes through
  `300` for Conv1 but only `100` for Conv2; positive one-sided differencing
  has no passing point on the tested high-beta grid.
- Scientific question: Do the baseline/ours higher-beta findings previously
  measured for Conv3 repeat in Conv1 and Conv2, and how do the one-sided,
  centered, float32, and float64 results change with depth?
- Frozen setup: Accepted BPTT-trained best-validation Conv1/Conv2
  baseline/ours checkpoints, one fixed 16-example ordinary-MNIST validation
  batch, `T=K=64`, frozen-current nudging, active biases excluded from
  gradient scoring, injected `B={100,300,1000}`, and independent native
  float32/float64 replay. Baseline uses base beta `B`; ours uses `B/4` for
  Conv1 and `B/16` for Conv2.
- Centered result: At `B=100/300/1000`, worst-layer float64 cosine is
  `.998945/.997131/.990309` for Conv1 baseline,
  `.998584/.994410/.979564` for Conv1 ours,
  `.999678/.997736/.993496` for Conv2 baseline, and
  `.995390/.984057/.911775` for Conv2 ours. With the all-layer cosine
  `>=.99`, symmetric norm delta `<=.1`, and residual gate, baseline passes all
  six architecture/beta points, ours Conv1 passes `100/300`, and ours Conv2
  passes only `100` in both dtypes.
- One-sided result: There is no passing point. At beta `100`, worst-layer
  float64 cosine is `.914346/.943015` for baseline Conv1/Conv2 and
  `.758829/.263565` for ours; at beta `1000` it is
  `.264351/.397352` and `.141423/.141112`. Centering therefore removes the
  leading one-sided finite-beta error but not the eventual higher-order and
  active-set bias.
- Precision and displacement: Float32 and float64 are nearly coincident over
  this large-signal grid, so failure is finite-perturbation bias rather than
  small-contrast float32 cancellation. At beta `100`, positive float64 output
  displacement is `1.9368/4.8651` for Conv1 baseline/ours and
  `.5663/17.9988` for Conv2; at `1000` it reaches
  `19.371/48.744` and `5.664/180.14`.
- Coverage and guards: The CPU smoke and all 24 local RTX 3090 production
  bundles validate; `main` handle `@114` exited zero without retries. Source,
  checkpoint, parameter, cohort, phase-start, frozen-force, iteration,
  beta-scaling, residual, dtype, and arithmetic guards pass. No optimizer was
  constructed or stepped and the official test split was not read. The
  normalized analysis has 24 context rows and 120 layer/precision rows.
- Reporting deviation: The reusable runner retains its source-compatible
  manifest study ID. The enclosing directory, exact launch script, complete
  inventory, and result hashes bind this Conv1/Conv2 extension.
- Limitations: One seed, one minibatch, ordinary MNIST, BPTT-trained
  checkpoints, and deliberately out-of-source-grid diagnostic betas. This is
  not EqProp training accuracy and does not propose these betas for training.
- Results and analysis: [review](../results/perfectdiode-conv12-baseline-ours-eqprop-higher-beta-seed0-20260812-v1/review.md),
  [report](../results/perfectdiode-conv12-baseline-ours-eqprop-higher-beta-seed0-20260812-v1/analysis/report.md),
  [context table](../results/perfectdiode-conv12-baseline-ours-eqprop-higher-beta-seed0-20260812-v1/analysis/summary.csv),
  [layer table](../results/perfectdiode-conv12-baseline-ours-eqprop-higher-beta-seed0-20260812-v1/analysis/layer_metrics.csv), and
  [figure](../results/perfectdiode-conv12-baseline-ours-eqprop-higher-beta-seed0-20260812-v1/analysis/cosine_vs_injected_beta.png).

## `perfectdiode-conv3-centered-float64-voltage-read-noise-tk8-tk12-20260812-v1` — lower-T/K endpoint-read-noise replay

- Analyzed: 2026-08-12
- Evidence class: `ordinary_mnist_learning_algorithm_read_noise_tk_diagnostic`
- Evidence tier: `exploratory_noncanonical`
- Outcome: `mixed`. The complete gradient/noise response is effectively
  unchanged when equal iteration budgets are reduced from `T=K=64` to `12`
  or `8`, but the trained baseline checkpoint is not equilibrium-valid at
  either lower budget.
- Scientific question: Does the centered-float64 Conv3 endpoint-voltage
  read-noise hierarchy between baseline, ours, and legacy persist at the
  lower iteration budgets used by training?
- Frozen setup: The same six reconstructed-initialization/best-validation
  checkpoints, frozen 16-example ordinary-MNIST validation batch,
  `T=K=64`-selected injected betas, centered frozen-current nudge, true
  float64 dynamics and endpoint subtraction, 32 matched trials per nonzero
  `sigma_v`, and biases active but excluded from gradient scoring. EqProp,
  its endpoints, residuals, and BPTT are recomputed at each `T=K in {8,12}`;
  the accepted `64/64` tensors are used only for drift comparisons.
- Threshold result: Every acquisition and usable threshold is identical at
  `T=K=8/12/64`: baseline initialization `1e-6/1e-6/1e-6`, baseline best
  `3e-6/3e-6/3e-6`, ours initialization and best
  `3e-7/3e-7/3e-7`, legacy initialization `3e-9/3e-9/3e-9`, and legacy best
  `3e-10/3e-10/3e-10`. All 192 lower-T/K tested-sigma rows exactly match
  their `64/64` acquisition, task, usable, and all-trials-usable gate
  classifications. Ours therefore retains its `100x` initialization and
  `1000x` best-checkpoint usable-noise margin over legacy.
- Clean-gradient and state result: All six lower-T/K clean EqProp gradients
  pass the all-layer cosine `>=.99` and symmetric norm-delta `<=.1` gate
  against same-T/K BPTT. The worst clean cosine across all three budgets is
  legacy-initialization C0 `.990074641`. The largest `8/8` EqProp drift
  versus `64/64` is trained-baseline C0 (cosine `.9999428`, relative L2
  `.0112593`); at `12/12` it shrinks to cosine `.999999945` and relative L2
  `3.45e-4`. Layerwise positive-minus-negative delta RMS changes by at most
  `.053344%` at `8/8` and `.0014916%` at `12/12`.
- Residual qualification: Five of six cases pass the strict residual
  `p90<.01` gate at both lower budgets. Trained baseline is under-relaxed:
  its limiting free Layer 1 p90 is `1.75151` at `8/8` (hard failure) and
  `.0734647` at `12/12`, versus `2.64e-11` at `64/64`. Its numeric `3e-6`
  lower-budget threshold is retained as a gradient-only diagnostic but is
  explicitly null as equilibrium-valid evidence. All other lower-budget
  thresholds are equilibrium-valid.
- Coverage and guards: Corrected runs `tk8-v2` and `tk12-v2` each validate
  with six cases, 11,544 gradient rows and 23,088 state-noise rows; source,
  checkpoint, parameter, input, float64, zero-noise, reference-hash, and
  no-update guards pass. The completed `smoke-tk8-v2` validates. An earlier
  smoke had a manifest-only K64 task-reference label and the first `tk8`
  production attempt was intentionally interrupted after that review finding;
  both are excluded and retained as provenance.
- Interpretation: The baseline/ours/legacy read-noise separation is not an
  artifact of using `T=K=64`; the noisy-gradient curves overlap across
  `8/12/64`. Reducing T/K does expose a separate convergence problem for the
  trained baseline checkpoint, so gradient-only agreement must not be used as
  evidence that every lower-budget state is equilibrated.
- Limitations: One ordinary-MNIST minibatch, one seed/checkpoint family, a
  fixed discrete sigma grid and 32 trials, fixed `64/64`-selected betas, and
  BPTT-trained weights. This is not EqProp training accuracy, formal tail
  evidence, physical-voltage calibration, or hardware ENOB.
- Results and analysis: [comparison report](../results/perfectdiode-conv3-centered-float64-voltage-read-noise-tk8-tk12-20260812-v1/analysis/report.md),
  [overview plot](../results/perfectdiode-conv3-centered-float64-voltage-read-noise-tk8-tk12-20260812-v1/analysis/tk_overview.png),
  [overlaid noise curves](../results/perfectdiode-conv3-centered-float64-voltage-read-noise-tk8-tk12-20260812-v1/analysis/noise_task_cosine_vs_sigma_by_tk.png),
  [threshold table](../results/perfectdiode-conv3-centered-float64-voltage-read-noise-tk8-tk12-20260812-v1/analysis/threshold_comparison.csv),
  [residual table](../results/perfectdiode-conv3-centered-float64-voltage-read-noise-tk8-tk12-20260812-v1/analysis/residual_summary.csv),
  and [gradient-drift table](../results/perfectdiode-conv3-centered-float64-voltage-read-noise-tk8-tk12-20260812-v1/analysis/tk64_gradient_drift.csv).

## `perfectdiode-bounded-uniform-zero-bias-lr-search-conv3-occupancy-report-only-seed0-20260811-v1` — Conv3 LR search without occupancy rejection

- Analyzed: 2026-08-11
- Evidence class: `ordinary_mnist_selection`
- Outcome: `mixed`. Removing occupancy as a rejection rule exposes three
  accuracy-eligible legacy-SGD cells and a best observation of `93.62%`, but
  no Conv3 surface yields a formal selected LR because that best remains on
  the expanded lower Conv-rho boundary.
- Scientific question: Under the exact parent initializer, zero bias LRs, and
  unchanged Conv3 LR-search contract, what training performance and endpoint
  clipping appear when bound occupancy is recorded but no longer rejects a
  candidate?
- Frozen setup: Seed-0 ordinary MNIST with the official test unread; exact
  shared `Uniform[1e-5,1e-4)` initializer SHA
  `8ebe9dd9...438f`; projection to `[1e-5,1e-4]`; exact-zero bias tensors and
  bias LRs; baseline/ours/legacy x SGD/Adam; `T=K=8`; 640-step canaries;
  three-epoch candidates; 90% gate; one adaptive boundary-expansion wave.
  Projection-efficiency, non-finite, loss, gradient, and epochwise Conv3 T/K
  gates remain active.
- Legacy-SGD result: Eleven numeric cells all pass three epochwise K64/T/K
  checks. At fixed `rho_dense=.0033333`, reducing `rho_conv` from
  `3.333e-4` to `1.111e-4` improves accuracy `92.56 -> 93.62%` and reduces
  exact endpoint bound occupancy `11.45 -> 7.65%`. The best cell's actual
  LRs are `C0=3.469e-8`, `C1=1.224e-5`, `C2=1.153e-5`, and
  `Dense=9.448e-6`; bias LRs are zero. Its layerwise either-bound occupancy
  is `3.21/9.79/9.39/5.01%`. The middle dense rho is bracketed, but the Conv
  rho is not; runner status is `unresolved_after_boundary_expansion` and
  `selected=null`.
- Other surfaces: Baseline-SGD remains probe-unresolved. Baseline-Adam and
  ours-Adam have no numeric cell because every candidate trips the retained
  projection gate. Ours-SGD completes six cells at `9.94--10.00%`, all of
  which fail post-training T/K. Legacy-Adam completes four cells, peaks at
  `88.66%` with `26.25%` exact clipping, and has no accuracy-eligible row.
- Clipping interpretation: Within legacy-SGD, accuracy versus exact final
  bound occupancy is Pearson/Spearman `-0.695/-0.664` (`n=11`), and the best
  row has the lowest endpoint occupancy. This argues against more endpoint
  clipping being beneficial, but remains descriptive: LRs vary jointly and
  exact final equality is not the removed persistent occupancy-increase
  statistic. Ours-SGD's positive `0.757/0.778` correlation is scientifically
  meaningless because all six runs are at chance and fail T/K.
- Interpretation: On this current V100 surface, the Conv LR was not too small;
  a threefold decrease improves both accuracy and clipping. Cell `046` is a
  next-search center, not a final handoff. The next range test should add a
  lower row near `rho_conv=3.70e-5` while keeping the already bracketed dense
  neighborhood around `.0033333`.
- Counterfactual limitation: The parent legacy-SGD path ran on Trex RTX 5090
  with PyTorch 2.11, while this path ran on Jean Zay V100 with PyTorch 2.5 and
  no bitwise CUDA determinism. Init, split/order, seed, normalized signature,
  and training/model code match, but the retained projection gate branches
  differently at the nominal center and shifts the adaptive grid down. The
  `93.62%` result therefore cannot be attributed to occupancy-rule removal
  alone. A causal comparison requires both policies on one runtime with fixed
  LRs and no re-probe/adaptive-grid drift.
- Guards and validation: Jean Zay job `832379` tasks 0--1 completed `0:0`.
  Remote/staging checksum reconciliation is empty; all `47/47` canonical
  bundles validate. Final analysis has six terminal surfaces, six valid
  receipts, 21 numeric bundles, 84 per-parameter occupancy rows, and zero
  missing, duplicate, in-progress, or invalid authorities. Akib's pre-search
  T/K-audit OOM produced no candidate and is retained; unchanged ours
  surfaces completed on main.
- Limitations: One seed, three-epoch ordinary-MNIST selection only; no long
  confirmation or paper-facing accuracy. Endpoint occupancy is not a causal
  LR diagnostic and report-only artifacts do not reconstruct whether the old
  persistence rule would reject the best row.
- Results and analysis: [study review](../results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv3-occupancy-report-only-seed0-20260811-v1/review.md),
  [final report](../results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv3-occupancy-report-only-seed0-20260811-v1/analysis/final/report.md),
  [candidate table](../results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv3-occupancy-report-only-seed0-20260811-v1/analysis/final/candidate_rows.csv),
  [occupancy correlations](../results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv3-occupancy-report-only-seed0-20260811-v1/analysis/final/accuracy_occupancy_correlations.csv),
  and [collection validation](../results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv3-occupancy-report-only-seed0-20260811-v1/provenance/collection_validation.json).

## `perfectdiode-conv3-eqprop-beta-1em3-to-1e2-init-best-true-dtype-seed0-20260811-v1` — complete Conv3 current-nudge beta and precision sweep

- Analyzed: 2026-08-11
- Evidence class: `ordinary_mnist_learning_algorithm_gradient_diagnostic`
- Evidence tier: `exploratory_noncanonical`
- Outcome: `mixed`. No baseline, ours, or legacy context has an all-layer
  float32 estimator, but centered float64 has a common all-scheme,
  initialization-and-best window at injected current
  `B in {1e-3,3e-3,1e-2}`.
- Scientific question: Across the complete requested injected-beta range, how
  do bias-excluded Conv3 EqProp--BPTT cosine, matched free/nudged relative
  displacement, and delta RMS vary by layer, checkpoint, amplification
  scheme, native precision, and one- versus two-sided differencing?
- Frozen setup: Hash-verified reconstructed initialization and best-validation
  BPTT weights for Conv3 baseline/ours/legacy; one fixed 16-example
  ordinary-MNIST validation minibatch; `T=K=64`; injected
  `B={1e-3,3e-3,1e-2,3e-2,1e-1,.3,1,3,10,30,100}`; positive one-sided and
  centered frozen-current nudges; independent native float32/float64
  dynamics. Base beta is `B`, `B/64`, and `B/4096` for baseline, ours, and
  legacy. Biases remain active in the dynamics but are excluded from scoring.
- Float32 result: Neither estimator has an all-layer point. For centered
  differencing, the best worst-layer cosine over the full grid is
  baseline `.164965/.335705`, ours `.596808/.355025`, and legacy
  `.161268/.149566` at initialization/best. All 33 undefined cosines are
  exact-zero float32 `C0` EqProp vectors.
- Float64 result: Centered all-layer cosine is at least `.99` with symmetric
  norm error at most `.1` over `.001--100` for both baseline checkpoints,
  `.001--10/.001--30` for ours initialization/best, and
  `.001--.03/.001--.01` for legacy. The common window is therefore
  `.001--.01` on the tested log grid. One-sided legacy has no all-layer point
  at `B>=.001`; its best maximin is `.877393/.847661`, limited by the
  Dense/readout gradient.
- Displacement result: At centered float64 `B=.001`, legacy H0/H1/H2/output
  relative displacement is
  `9.45e-9/2.51e-5/.00469/1.34555` at initialization and
  `1.50e-10/2.37e-7/.000391/.520166` at best. The corresponding output delta
  RMS is `.738209/.187204`. Thus the centered directional window is not a
  small-output-perturbation window; positive and negative motion is nearly
  symmetric and centered differencing cancels the leading one-sided error.
- Residual result: All float64 residual gates pass. Only trained baseline
  float32 `Layer_2` is slightly residual-limited, with p90
  `.01025--.01147` and maximum `.01660`; the shared free/zero state also fails,
  so this is not signed-nudge blow-up. No row crosses the `.1` hard-failure
  threshold, and all ours/legacy float32 residual contexts pass.
- Interpretation: The float32 early-layer failure is a shared numerical
  contrast problem, not legacy-specific evidence. Legacy nevertheless has the
  sharpest tradeoff: its output moves by order one at the smallest common
  centered float64 beta while H0 remains barely displaced. This replay does
  not establish EqProp training accuracy.
- Coverage and guards: Both valid smokes and all 22 production bundles pass
  canonical validation. The aggregate reconciles 132 logical contexts, 264
  dtype contexts, 1,056 joined layer rows, 3,696 residual rows, 4,620
  displacement rows, 660 phase rows, and 132 scaling rows. Positive endpoints
  agree exactly between estimators; source/checkpoint/parameter/dtype guards
  pass; no optimizer step or official-test read occurs. The first sandboxed
  smoke failed before canonical bundle creation because CUDA was not visible;
  the unchanged retry and stress smoke pass and are retained.
- Reporting deviation: Bundle manifests retain the generic source-compatible
  study ID. The aggregate SHA-binds the enclosing plan and launcher and
  reconstructs exact `11 x 2` bundle coverage. At `B=.01`, inherited
  `beta_capped` metadata is true for ours/legacy, but requested, actual, and
  effective beta are unchanged and explicitly validated.
- Limitations: One seed, one fixed ordinary-MNIST minibatch, read-only replay,
  and no EqProp optimizer-step or accuracy experiment.
- Results and analysis: [review](../results/perfectdiode-conv3-eqprop-beta-1em3-to-1e2-init-best-true-dtype-seed0-20260811-v1/review.md),
  [full table](../results/perfectdiode-conv3-eqprop-beta-1em3-to-1e2-init-best-true-dtype-seed0-20260811-v1/analysis/summary.csv),
  [validation summary](../results/perfectdiode-conv3-eqprop-beta-1em3-to-1e2-init-best-true-dtype-seed0-20260811-v1/analysis/summary.json),
  [float32 cosine plot](../results/perfectdiode-conv3-eqprop-beta-1em3-to-1e2-init-best-true-dtype-seed0-20260811-v1/analysis/plots/cosine_vs_injected_beta_float32.png),
  and [float64 cosine plot](../results/perfectdiode-conv3-eqprop-beta-1em3-to-1e2-init-best-true-dtype-seed0-20260811-v1/analysis/plots/cosine_vs_injected_beta_float64.png).

## `perfectdiode-conv3-baseline-ours-float64-beta-observables-20260811-v1` — float64 phase-displacement and squared-drop sweep

- Analyzed: 2026-08-11
- Evidence class: `ordinary_mnist_learning_algorithm_state_diagnostic`
- Evidence tier: `exploratory_noncanonical`
- Outcome: `mixed`. Correctly scaled float64 positive one-sided EqProp has a
  broad all-layer window, but very small beta is cancellation-limited and the
  ours readout becomes finite-beta-limited at the largest tested point.
- Scientific question: Across a very wide float64 beta range, how large is
  the matched free-to-nudged state motion layer by layer, and how much
  separately read phase-local squared-voltage-drop contrast remains for the
  normalized EqProp update?
- Frozen setup: Best-validation BPTT Conv3 baseline epoch `30` and ours epoch
  `28`, one fixed 16-example ordinary-MNIST validation minibatch, positive
  one-sided frozen-current EqProp, common residual-clean `T=K=64`, float64,
  and weight-only scoring. Biases remain active in the dynamics.
- Scaling and grid: Effective/injected beta is
  `B=beta_base*(voltage_amp/current_amp)^3`; baseline therefore uses
  `beta_base=B` and ours uses `B/64`. Nineteen common `B` values span
  `2.8046421e-9` through `1` over `8.55` decades.
- Fidelity result: With the exploratory all-layer criterion cosine `>=.99`
  and symmetric norm error `<=.1`, both schemes first pass at
  `B=2.8046421e-5`. Baseline remains faithful through `B=1` (worst cosine
  `.999999`); ours passes through `.3` (worst `.995143`) but its Dense cosine
  falls to `.947298` at `1`, where output relative displacement is `.197275`.
- State result: At `B=.1`, relative H0/H1/H2/output displacement is
  `1.94e-10/2.01e-9/5.40e-7/1.04e-4` baseline and
  `4.21e-10/6.11e-8/3.13e-5/1.97e-2` ours. Output/H0 absolute displacement
  ratios are about `3.48e3` and `9.62e5`, respectively.
- Squared-drop result: The exact endpoint statistic is `q=dE/dw`, and the
  update is `(q_plus-q_zero)/B`. At `B=.1`, C0
  `RMS(delta q)/RMS(q_zero)` is `7.48e-12` baseline and `7.91e-12` ours,
  exposing a severe common-mode readout requirement even though float64 is
  directionally stable there.
- Coverage and guards: The CUDA smoke and all 19 production bundles validate;
  the analysis reconciles 152 scheme/layer/beta rows and verifies the requested
  beta scaling and archived update identity. No optimizer is stepped and the
  official test split is not read.
- Limitations: One seed/checkpoint per scheme, one 16-example ordinary-MNIST
  minibatch, positive one-sided replay only, and no training-accuracy sweep.
  This characterizes numerical endpoint subtraction and finite-beta bias; it
  is not yet a hardware ADC/ENOB measurement.
- Results and analysis: [review](../results/perfectdiode-conv3-baseline-ours-float64-beta-observables-20260811-v1/review.md),
  [complete layer-by-beta table](../results/perfectdiode-conv3-baseline-ours-float64-beta-observables-20260811-v1/analysis/observables_table.md),
  [state CSV](../results/perfectdiode-conv3-baseline-ours-float64-beta-observables-20260811-v1/analysis/state_displacement_wide.csv),
  [learning-rule CSV](../results/perfectdiode-conv3-baseline-ours-float64-beta-observables-20260811-v1/analysis/learning_rule_wide.csv), and
  [figure](../results/perfectdiode-conv3-baseline-ours-float64-beta-observables-20260811-v1/analysis/beta_observables_loglog.png).

## `perfectdiode-conv3-baseline-ours-eqprop-one-vs-two-sided-higher-beta-seed0-20260811-v1` — higher-beta one- versus two-sided true-dtype EqProp

- Analyzed: 2026-08-11
- Evidence class: `ordinary_mnist_learning_algorithm_gradient_diagnostic`
- Evidence tier: `exploratory_noncanonical`
- Outcome: `negative` for float32 training.  Centered two-sided differencing is
  materially better than positive one-sided differencing, but neither trained
  Conv3 baseline nor ours has an all-layer float32 point through the tested
  order-one-displacement boundary.
- Scientific question: Does increasing the frozen-current nudge beyond the
  previous grid recover early-layer float32 contrast, how large is the
  free-to-nudged motion at each layer, and does centered `-beta/+beta`
  differencing repair the one-sided estimate?
- Frozen setup: Best-validation BPTT Conv3 baseline epoch `30` and ours epoch
  `28`, one fixed 16-example ordinary-MNIST validation minibatch, `T=K=64`,
  true float32 and float64 dynamics, and weight-only scoring.  All primary
  points run on one Trex RTX 5090 surface after the occupied local 3090 lane
  was excluded as transport.
- Nudge and scaling: One `F=-dC/dy` tensor is frozen at the common post-T state
  and reused byte-for-byte for zero and every signed phase in the final-output
  `b_k` current interaction.  Baseline tests injected beta `{100,300,1000}`;
  ours tests `{3,5,10}` using base beta divided by the guarded Conv3 factor
  `64`.  The estimators are `(G+ - G0)/beta_injected` and
  `(G+ - G-)/(2*beta_injected)`, formed entirely in the endpoint dtype.
- Float32 result: Centered baseline C0 rises
  `.314963/.644491/.915589`, but never reaches `.99`.  Centered ours C0 is
  `.024906/.066527/.046875`, while C1 is only
  `.234048/.321744/.494860`.  Increasing beta therefore does not expose a
  usable all-layer float32 gradient.
- Float64 result: Centered gradients remain extremely well aligned at all
  tested points; the worst cosine is `.997707` for baseline beta `1000` and
  `.998990` for ours beta `10`.  This is cancellation of centered finite-beta
  error, not evidence that the perturbation is small.
- Displacement/nonlinearity: Float64 matched-zero output displacement grows
  `.103767 -> .311282 -> 1.037796` for baseline and
  `.591827 -> .986378 -> 1.972761` for ours.  At the largest beta, one-sided
  Dense cosine has fallen to `.709103/.194545`, while centered Dense remains
  essentially one.  Negative and positive displacement magnitudes are nearly
  symmetric; matched positive endpoint hashes are exactly identical across
  estimator runs.
- Residual interpretation: Ours passes every residual gate in both dtypes and
  baseline float64 passes.  Baseline float32 is residual-limited because the
  shared post-T/zero Layer 2 p90 is already `.0119629` against the strict
  `.01` threshold; signed endpoints remain finite in the same narrow range
  rather than blowing up with beta.
- Guards and coverage: Six production bundles and two smokes validate locally.
  All checkpoint/source/cohort, common-state, frozen-force, iteration,
  parameter, and native-dtype guards pass; biases are active but excluded from
  scoring, no optimizer is constructed or stepped, and the official test split
  is not read.  The normalized report reconciles 168 rows and explicitly
  supersedes duplicate local one-sided anchors with matched Trex points.
- Reporting deviation: Individual runner manifests retain the compatible
  source-study ID `perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1--float64-shadow`;
  their enclosing directory, run inventory, hashes, and this manifest provide
  the higher-beta study association.  Canonical bundle validation passes.
- Limitations: One seed, one fixed minibatch, ordinary MNIST, BPTT-trained
  checkpoints, and no EqProp optimizer steps.  Betas above the source `.01`
  injected-current cap are explicit diagnostic extrapolations.
- Results and analysis: [review](../results/perfectdiode-conv3-baseline-ours-eqprop-one-vs-two-sided-higher-beta-seed0-20260811-v1/review.md),
  [layer/displacement report](../results/perfectdiode-conv3-baseline-ours-eqprop-one-vs-two-sided-higher-beta-seed0-20260811-v1/analysis/conv3_eqprop_one_vs_two_sided_higher_beta.md),
  [figure](../results/perfectdiode-conv3-baseline-ours-eqprop-one-vs-two-sided-higher-beta-seed0-20260811-v1/analysis/conv3_eqprop_one_vs_two_sided_higher_beta.png), and
  [normalized CSV](../results/perfectdiode-conv3-baseline-ours-eqprop-one-vs-two-sided-higher-beta-seed0-20260811-v1/analysis/conv3_eqprop_one_vs_two_sided_higher_beta.csv).
  CSV SHA-256 is
  `86296c131142bcfdd9251fe0d22a51fec644095fb4d7d8732d9e02bbb55f1f4e`.

## `perfectdiode-conv3-ours-eqprop-beta-above10-bug-audit-seed0-20260811-v1` — ours beta-above-10 injection and displacement audit

- Analyzed: 2026-08-11
- Evidence class: `ordinary_mnist_learning_algorithm_gradient_diagnostic`
- Evidence tier: `exploratory_noncanonical`
- Outcome: `negative` for a beta-injection bug and for float32 recovery.
  Injected beta is applied exactly once, but the output-to-early-layer response
  is severely attenuated; increasing beta introduces finite-perturbation bias
  before producing an all-layer float32 estimator.
- Scientific question: Was the missing trained-Conv3 ours tail above injected
  beta `10` hiding a beta-scaling or displacement bug, and can still larger
  one- or two-sided nudges recover the early float32 layers?
- Frozen setup: BPTT-trained Conv3 ours best-validation epoch `28`, one fixed
  16-example ordinary-MNIST validation minibatch, `T=K=64`, final-output
  frozen-current nudging, true float32 and float64 dynamics, and weights-only
  scoring.  The new production surface used one free Trex RTX 5090 lane.
- Scaling: The injected magnitude is
  `B=beta_base*(voltage_amp/current_amp)^3=64*beta_base`.  New injected/base
  pairs are `30/.46875`, `100/1.5625`, `300/4.6875`, and `1000/15.625`.
  The output `b_k` interaction and finite-difference denominator use the same
  `B`; the runtime factor, frozen-force identity, common phase starts, and
  paired endpoint hashes all pass.
- Displacement result: Positive float64 H0 relative displacement at
  `B=30/100/300/1000` is
  `1.263e-7/4.147e-7/2.498e-6/1.367e-5`, while output displacement is
  `5.918/19.745/59.441/198.624`.  Absolute RMS shows the same attenuation.
  Output displacement remains almost proportional to `B`, while hidden-layer
  asymmetry and active-set transitions grow at the largest points.
- Gradient result: Centered float64 is still all-layer `>=.99` at `B=30`
  (`.999566/.997234/.995566/.999992`) but fails that threshold by `B=100`.
  Centered float32 never has an all-layer point; its best worst-layer cosine is
  `.481631` at `B=300`.  One-sided Dense cosine is already `-.006012` at
  `B=30` and remains negative, exposing its uncanceled finite-beta term.
- Residual interpretation: Every ours phase and layer passes the strict
  residual gates in both dtypes through `B=1000`.  The high-beta loss of
  direction is finite-perturbation and diode-active-set bias, not residual
  blow-up or insufficient equilibrium iterations.
- Coverage and guards: Eight new production bundles and one CPU smoke validate
  locally; all eight remote/local `result.json` hashes match.  The joined
  analysis adds the paired `B={3,5,10}` anchors, yielding 14 included bundles
  and 112 normalized rows.  Biases remain active but excluded from scoring;
  no optimizer is constructed or stepped and the official test split is not
  read.
- Reporting deviation: Runner manifests retain the compatible source-study
  identity rather than the enclosing tail directory name.  Manifest and
  result IDs agree and validate; the top-level directory, launch script,
  analysis inventory, and review record the study association.
- Limitations: One seed/checkpoint and one minibatch; ordinary-MNIST diagnostic
  replay of a BPTT-trained model, not EqProp training.  These deliberately huge
  betas exceed the source current cap and are failure-audit points, not
  candidate training settings.
- Results and analysis: [review](../results/perfectdiode-conv3-ours-eqprop-beta-above10-bug-audit-seed0-20260811-v1/review.md),
  [exact joined table](../results/perfectdiode-conv3-ours-eqprop-beta-above10-bug-audit-seed0-20260811-v1/analysis/conv3_ours_eqprop_beta_above10.md),
  [normalized CSV](../results/perfectdiode-conv3-ours-eqprop-beta-above10-bug-audit-seed0-20260811-v1/analysis/conv3_ours_eqprop_beta_above10.csv), and
  [figure](../results/perfectdiode-conv3-ours-eqprop-beta-above10-bug-audit-seed0-20260811-v1/analysis/conv3_ours_eqprop_beta_above10.png).
  CSV SHA-256 is
  `0ed5c7207955e32bcd90adf686fa50075f9c4668968e7e69b6c12390a28ce0ec`.

## `perfectdiode-conv3-baseline-ours-eqprop-high-beta-high-tk-seed0-20260811-v1` — high-beta/high-T/K true-dtype EqProp extension

- Analyzed: 2026-08-11
- Evidence class: `ordinary_mnist_learning_algorithm_gradient_diagnostic`
- Evidence tier: `exploratory_noncanonical`
- Outcome: `negative`; neither baseline nor ours has a tested true-float32
  all-layer EqProp--BPTT window, and raising `T/K` through `128/128` does not
  repair the early convolutions.
- Scientific question: Can a much larger positive one-sided frozen-current
  nudge make the trained Conv3 float32 early-layer contrast measurable, and is
  the previous failure instead caused by insufficient free/nudged relaxation?
- Frozen setup: Best-validation Conv3 baseline and ours checkpoints, one fixed
  16-example ordinary-MNIST validation minibatch, true float32 and float64
  dynamics, and weight-only scoring. Biases remain active in the dynamics;
  no optimizer is constructed or stepped and the official test split is not
  read.
- Scaling and grid: `beta_injected=beta_base*(voltage_amp/current_amp)^3`, so
  the factors are baseline `1` and ours `64`. Common injected beta
  `{.01,.1,1}` is replayed at native and `64/64`; `.1` is confirmed at
  `128/128`; the adaptive tail adds beta `3` for both and `{10,30,100}` for
  baseline at `64/64`. Values above the source `.01` cap are explicit
  diagnostic extrapolations, not proposed training nudges.
- T/K result: At injected beta `.1`, the complete baseline and ours layer
  vectors at `64/64` and `128/128` are identical at recorded precision.
  Baseline float64 worst projected-KKT p90 improves from about `.07346` at
  native `12/8` to `2.7e-11` at `64/64`, while float32 stalls near `.01099`
  and its direction remains bad. Ours passes its float32 residual gates yet
  also retains bad C0/C1. More iterations are not the remedy.
- High-beta result: At beta `100`, baseline float32 `C0/C1/C2/D` cosines are
  `.245752/.773328/.999596/.992606`; C0 has norm ratio `4.67` and `83.5%`
  exactly-zero contrast entries, while float64 Dense is already `.992606`.
  Ours reaches the finite-beta crossover earlier: at beta `3`, output relative
  displacement is `.591827`, float64 Dense cosine is `.674475`, and float32
  C0/C1 remain only `.086006/.157173`.
- Displacement mechanism: At beta `.1`, matched-free/nudged float64 relative
  displacements `H0/H1/H2/output` are
  `1.94e-10/2.01e-9/5.40e-7/1.04e-4` for baseline and
  `4.21e-10/6.11e-8/3.13e-5/.01973` for ours. The nudge is strongly
  depth-skewed: the readout leaves linear response before the earliest
  float32 contrast becomes reliable.
- Interpretation: This extension strengthens the numerical
  cancellation/transport explanation. Float64 is converged and aligned at
  the same T/K where float32 is dead or nearly orthogonal. Do not choose a
  larger one-sided beta or extra iterations as the float32 training remedy;
  change precision/contrast acquisition or the learning-rule transport.
- Guards and validation: All 11 included canonical bundles validate, source
  checkpoint and in-memory parameter hashes remain unchanged, all iteration
  contracts pass, and the normalized analysis contains 152 layerwise rows.
  The failed `tk64-baseline-injected-30` and
  `tk64-baseline-injected-300` directories are unrelated-lane CUDA-OOM
  attempts with no result and are excluded; beta 30 is represented by the
  successful `-retry1` authority.
- Limitations: One seed, one fixed 16-example minibatch, ordinary MNIST, and a
  checkpoint gradient diagnostic rather than EqProp training. No beta above
  `100` completed, so the measured claim is the absence of a usable window
  through `100`, with the finite-beta crossover already visible there.
- Results and analysis: [review](../results/perfectdiode-conv3-baseline-ours-eqprop-high-beta-high-tk-seed0-20260811-v1/review.md),
  [layerwise report](../results/perfectdiode-conv3-baseline-ours-eqprop-high-beta-high-tk-seed0-20260811-v1/analysis/conv3_eqprop_high_beta_high_tk.md),
  [figure](../results/perfectdiode-conv3-baseline-ours-eqprop-high-beta-high-tk-seed0-20260811-v1/analysis/conv3_eqprop_high_beta_high_tk.png), and
  [normalized CSV](../results/perfectdiode-conv3-baseline-ours-eqprop-high-beta-high-tk-seed0-20260811-v1/analysis/conv3_eqprop_high_beta_high_tk.csv).
  CSV SHA-256 is
  `450634102b25b8784e5441d7d30fc09102418c878737e7b00f71928be654abe2`;
  the machine-readable summary records every input result hash.

## `perfectdiode-conv3-legacy-eqprop-adc-readout-precision-20260811-v1` — EqProp endpoint-readout precision

- Analyzed: 2026-08-11
- Evidence class: `ordinary_mnist_learning_algorithm_gradient_diagnostic`
- Outcome: `mixed`; independent absolute endpoint conversion needs 44
  nominal ideal-ADC bits at the accepted primary beta, whereas an oracle
  analog-difference front end needs 8.  The latter is an architectural control,
  not a realizable hardware specification.
- Scientific question: How does ideal node-voltage ADC resolution affect each
  minibatch EqProp weight update when the matched zero and positive endpoints
  are read separately, and how much resolution could be saved by subtracting
  the endpoints before conversion?
- Frozen setup: Trained Conv3 legacy best-validation checkpoint, native
  `T=K=8`, one fixed 64-example ordinary-MNIST validation cohort in four
  minibatches, and true-float64 positive one-sided frozen-current EqProp at
  base beta hats `{3e-10,1e-9,3e-9}`.  Biases remain active in the dynamics,
  but the four weight gradients alone are scored.  The official test split is
  not read and no optimizer is constructed.
- Acquisition contracts: The primary arm uses one signed symmetric mid-tread
  quantizer per state layer.  Its range is calibrated on the same cohort's
  zero endpoint with `1.05x` headroom and frozen across phases, betas, and
  minibatches.  The control subtracts states before conversion, retains the
  common state exactly, and uses an oracle per-layer/per-beta difference
  range.  Both recompute the actual local squared-voltage-drop statistics from
  quantized nodes before forming the EqProp numerator.
- Headline measurements:

  | Base beta hat | Effective beta | Worst clean EqProp/BPTT cosine | Separate endpoint threshold | Oracle difference threshold |
  |---:|---:|---:|---:|---:|
  | `3e-10` | `3.4463442e-5` | `.999282` | `44` | `8` |
  | `1e-9` | `1.1487814e-4` | `.998395` | `42` | `8` |
  | `3e-9` | `3.4463442e-4` | `.985886` | `41` | `8` |

  A threshold requires every weight and minibatch to have cosine at least
  `.99` and symmetric norm delta at most `.10` relative to clean float64
  EqProp, and to remain passing at every higher tested integer bit.  No
  pass-to-fail reversal occurs.
- Mechanism: At the primary beta, Layer 1 has full scale `121.9267` but mean
  free-to-nudged RMS displacement only `3.3065e-11`, a `2^41.75`
  range-to-signal ratio.  At 43 absolute bits, its mean p90 displacement is
  `.896` LSB, only `20.63%` of endpoint codes change, and the limiting
  `ConvWeight_0` minibatch cosine is `.983829`.  At 44 bits these become
  `1.792` LSB, `29.08%`, and `.994313`.  Increasing beta improves acquisition
  SNR approximately as expected, but the largest beta's clean Dense EqProp
  cosine to BPTT falls to `.985886`, exposing the finite-beta/readout tradeoff.
- Interpretation: Independent absolute conversion spends nearly all codes on
  common mode and is unattractive for this small-nudge operating point.
  Correlated sample/hold plus differential sensing, or analog/mixed-signal
  formation of the factorized local observable
  `(z_plus-z_zero)(z_plus+z_zero)/(2 beta_eff)`, is the supported direction.
  The next gate should add differential offset/noise, CMRR, temporal
  decorrelation, and a causal range policy before any DAC/write study.
- Limitations: Nominal quantizer bits are not hardware ENOB.  Both range
  calibrations are in-sample; the control is oracle and beta-dependent; read
  noise, converter nonlinearity, gain/offset mismatch, sample/hold error, and
  conductance programming are absent.  The primary path subtracts separately
  computed endpoint statistics in float64, so its threshold combines ADC
  quantization and digital cancellation.  This is one checkpoint and one
  ordinary-MNIST cohort, not paper-facing evidence or a training result.  The
  exact executed analyzer source snapshot was not retained; promotion to
  formal evidence therefore requires a fresh run under the formal tier.
- Guards and validation: The passing `smoke-02` and production bundles
  validate.  Production reconciles all 1,968 expected layer/minibatch rows;
  every residual, coverage, beta-scale, source, and parameter guard passes.
  The current batch-0 primary-beta replay exactly equals all 12 accepted
  archived float64 EqProp, BPTT, and numerator tensors.  Result SHA-256 is
  `1ecbe8557e777316ab7dbef5ee7e2a97d70ffb4fa11781962d63c7bd3790e291`.
- Results and analysis: [study review](../results/perfectdiode-conv3-legacy-eqprop-adc-readout-precision-20260811-v1/review.md),
  [production report](../results/perfectdiode-conv3-legacy-eqprop-adc-readout-precision-20260811-v1/production/report.md),
  [threshold table](../results/perfectdiode-conv3-legacy-eqprop-adc-readout-precision-20260811-v1/production/minimum_passing_bits.csv),
  [state measurements](../results/perfectdiode-conv3-legacy-eqprop-adc-readout-precision-20260811-v1/production/state_adc_metrics.csv),
  and [fidelity plot](../results/perfectdiode-conv3-legacy-eqprop-adc-readout-precision-20260811-v1/production/gradient_fidelity_vs_adc_bits.png).

## `perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-20260810-v1` — bounded-uniform zero-bias LR selection

- Analyzed: 2026-08-10
- Evidence class: `ordinary_mnist_selection`
- Outcome: `mixed`; six of eighteen surfaces produced reportable three-epoch
  selections, while six completed numeric surfaces stayed below the 90% gate
  and all six Conv3 surfaces remained scientifically unresolved.
- Scientific question: With one matched initializer per architecture, which
  weight learning rates are viable for baseline, ours, and legacy
  amplification under the narrow `[1e-5,1e-4]` conductance interval, and does
  final exact-bound occupancy diagnose whether a learning rate is adequate?
- Frozen setup: Seed 0 ordinary MNIST with 55,000/5,000 train/validation
  examples and no official-test read; Conv1/2/3 use accepted `T/K=4/4,6/6,8/8`.
  Each architecture shares one scheme/optimizer-matched checkpoint whose
  weights were sampled from `Uniform[1e-5,1e-4)` and projected to
  `[1e-5,1e-4]`. Hidden biases start at exactly zero and every `Bias_*` learning
  rate is exactly zero. The core rho grid is conv `{0.001,0.003,0.009}` by
  dense `{0.0033333,0.01,0.03}`, with one declared edge-expansion wave;
  Conv3 baseline/ours use their predeclared fixed higher grid. Candidates train
  for three epochs after a restarted 640-step gate, and reportable selections
  must meet 90% validation accuracy.
- Reportable selections:

  | Architecture | Scheme / optimizer | Selected `(rho_conv,rho_dense)` | Actual weight LRs | Accuracy | Exact lower / upper / either bound |
  |---|---|---:|---|---:|---:|
  | Conv1 | ours / SGD | `(0.001,0.0033333)` | `C0 3.980e-8, D 2.017e-6` | `90.18%` | `5.99 / 0.55 / 6.54%` |
  | Conv1 | ours / Adam | `(0.001,0.01)` | `C0 6.055e-8, D 6.086e-7` | `91.60%` | `18.25 / 1.03 / 19.28%` |
  | Conv1 | legacy / SGD | `(0.003,0.0033333)` | `C0 2.582e-8, D 4.386e-7` | `94.50%` | `0.54 / 0.13 / 0.67%` |
  | Conv1 | legacy / Adam | `(0.009,0.01)` | `C0 5.449e-7, D 6.086e-7` | `95.42%` | `3.32 / 0.86 / 4.18%` |
  | Conv2 | legacy / SGD | `(0.003,0.0033333)` | `C0 1.253e-7, C1 5.458e-6, D 1.572e-6` | `93.34%` | `5.41 / 0.80 / 6.21%` |
  | Conv2 | legacy / Adam | `(0.009,0.01)` | `C0 5.449e-7, C1 5.476e-7, D 6.092e-7` | `95.42%` | `13.08 / 1.23 / 14.31%` |

- Unresolved numeric surfaces: Conv1 baseline SGD/Adam peak at
  `87.52/88.84%`; Conv2 baseline SGD/Adam and ours SGD/Adam peak at
  `49.34/84.36/69.50/87.38%`. These are highest observations, not winners.
  Conv2 ours-Adam peaks on the tested conv-rho edge at `(0.009,0.01)` with
  `87.38%` accuracy and `22.19%` pooled exact-bound occupancy, so a larger
  convolutional rate is a weak follow-up hypothesis, but the `0.28 pp` gain
  from conv rho `0.003` to `0.009` at dense rho `0.01` is not a settled
  under-learning-rate diagnosis.
- Conv3 outcome: baseline-SGD fails the stability probe through
  `ConvWeight_0`. Baseline-Adam and ours SGD/Adam have all nine fixed-grid
  cells rejected by the Conv3 safety contract. Legacy SGD/Adam downshift to
  smaller adaptive grids, but each has five canary and four full-candidate
  safety rejections. There is therefore no numeric Conv3 accuracy or clipping
  result and no bounded-uniform Conv3 LR selection.
- Clipping interpretation: Final pooled either-bound occupancy has mixed-sign
  within-surface correlations with accuracy. Pearson/Spearman range from
  `-0.953/-0.817` (Conv1 baseline-SGD) and `-0.932/-0.883` (Conv2 legacy-SGD)
  to `+0.982/+0.967` (Conv1 baseline-Adam); Conv2 ours-Adam is nearly null at
  `-0.081/-0.117`, while Conv2 legacy-Adam is `+0.527/+0.615`. The strongest
  Conv1 legacy rows clip only `0.67/4.18%`, whereas Conv2 baseline-Adam clips
  `44.04%` and remains below gate. Layer occupancy also differs sharply: at
  the selected Conv2 legacy-SGD row, exact either-bound occupancy is
  `8.51/10.93/3.41%` for `C0/C1/D`; for legacy-Adam it is
  `24.91/26.24/7.21%`. Endpoint clipping is consequently a useful saturation
  symptom, but not a causal or standalone LR selector: it loses clipping
  history, update direction and magnitude, near-bound mass, and layer-size
  weighting.
- Interpretation: The uniform initializer removes the earlier initializer
  confound and shows a coherent legacy advantage at this narrow bound, but it
  does not establish viable rates for baseline/ours Conv2 or any Conv3 scheme.
  The grid geometry, not clipping percentage, supports only a limited LR
  follow-up: extend Conv2 ours-Adam conv rho while holding dense rho near
  `0.01`. Do not publish a complete bounded-weight handoff from this study.
- Guards and validation: All 18 expected surfaces are terminal with 18 valid
  semantic receipts. The final analyzer accepts 114/114 canonical completed
  candidates, retains 37 canary and 11 full-candidate scientific safety
  rejections, records 282 per-parameter occupancy rows, and finds zero invalid
  bundles or duplicate authorities. Akib and Trex checksum comparisons are
  empty, and the Jean Zay content mirror is complete; every receipt preserves
  the frozen source, config, and architecture-shared initialization hashes. No
  10/30/30-epoch confirmation was launched.
- Operational provenance: The original main/Akib wrappers completed only their
  first surface before a receipt-writer `hashlib` import defect; the failed
  attempts remain retained and are excluded. Recovery source commit
  `950b845519b6ed91c424cf7accdc59a193b8377f` and archive SHA-256
  `f9f73f5c...6e407fe` passed regression tests and replacement smokes. Conv1
  ran on main, Conv2 on Akib, Conv3 baseline/ours on Jean Zay job `819812`
  tasks 0--1, and Conv3 legacy on Trex. Pending Jean Zay task `819812_2` was
  canceled before allocation and replaced without duplicate execution.
- Limitations: One seed, three-epoch ordinary-MNIST selection accuracy, strong
  safety censoring in Conv3, and no long confirmation or paper-facing
  medium-affine evidence. Cross-surface comparisons mix different physical LR
  mappings and should not be read as causal effects of occupancy.
- Results and analysis: [study root](../results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-20260810-v1/),
  [final report](../results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-20260810-v1/analysis/final/report.md),
  [surface table](../results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-20260810-v1/analysis/final/surface_summary.csv),
  [candidate table](../results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-20260810-v1/analysis/final/candidate_rows.csv),
  [per-parameter occupancy](../results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-20260810-v1/analysis/final/parameter_endpoint_occupancy.csv),
  [accuracy--occupancy correlations](../results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-20260810-v1/analysis/final/accuracy_occupancy_correlations.csv),
  and [correlation plot](../results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-20260810-v1/analysis/final/plots/accuracy_vs_either_bound_occupancy.png).
- Provenance: Study config SHA-256 `ee8688a1...fac6a05`; initializer checkpoint
  SHA-256 is `235d7ae5...cb1f16e` for Conv1,
  `6469bc8f...c4d88d2` for Conv2, and `8ebe9dd9...e555438f` for Conv3.

## `perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1` — positive one-sided EqProp precision and transport

- Analyzed: 2026-08-11
- Evidence class: `ordinary_mnist_learning_algorithm_gradient_diagnostic`
- Outcome: `mixed`; literal amplification-scaled frozen-current EqProp has a
  case-specific small-beta window with all-layer float64 cosine above `0.99`
  for every trained Conv2/Conv3 case, but no precision-certified deep float32
  update and no tested common base beta that works across schemes.
- Scientific question: After replacing centered EqProp with a positive
  one-sided estimator, reducing beta, and scaling its output current by
  `(voltage_amp/current_amp)^L` as the bias current is scaled, do EqProp
  gradients match BPTT layer-by-layer; does the answer depend on T/K,
  residual convergence, amplification scheme, state displacement,
  convolution, or arithmetic precision?
- Frozen setup: seed-0 baseline `v1/c1`, ours `v4/c1`, and legacy `v4/c0.25`
  Conv1/2/3 Adam leaders at reconstructed initialization and best validation;
  one shared 64-example ordinary-MNIST validation cohort; native T/K plus a
  fixed strong-beta cross through `64/64`; true-dtype shadows use the first
  fixed 16-example batch. Biases are active but excluded from scored
  gradients.
- Learning rule and scaling: `nudging_mode=current` freezes
  `force=-dC/dy` at the common post-T state, adds
  `-beta_injected<y,force>`, and reports
  `[dtheta E(s_plus)-dtheta E(s_zero)]/beta_injected`. The literal scaling is
  `beta_injected=beta_base*(voltage_amp/current_amp)^L`, using the guarded
  output bias-current-row exponent `L=1/2/3`. Multipliers for
  baseline/ours/legacy are `1/4/16`, `1/16/256`, and `1/64/4096` for depths
  one, two, and three. This is frozen-current one-sided EqProp, not full-cost
  `E+beta*C` EqProp.
- Supersession: earlier cost-mode bundles only reported the current scaling;
  the runtime ignored `current_scale` in their dynamics. They remain controls
  but are superseded for the literal-current question. The current-mode
  `conv-current-*`, `fc-current-*`, and `conv-shadow-current-*` bundles are
  authoritative.
- Float32 beta result: both the literal common-base and equal-injected Conv
  sweeps select `0/18` contexts by the residual + SNR + adjacent-consistency
  rule. Readouts align first, while trained early convolutions remain poor or
  exactly zero. The strong-endpoint trained vectors `C0/C1[/C2]/D` are:
  Conv2 baseline `-.027/.065/.996`, ours `.007/.184/1`, legacy
  `-.015/.992/.990`; Conv3 baseline `zero/.010/.025/.978`, ours
  `zero/.013/.268/1`, and legacy `zero/.036/.996/.217`.
- Precision result: independent full-dynamics float32/float64 shadows compute
  EqProp and BPTT from identical hashed starts in their native dtype. Every
  trained Conv2/Conv3 case has a tested case-specific beta with all-layer
  float64 cosine above `0.99`. For trained Conv3 legacy, base beta hat
  `3e-10` means base beta `8.4139263e-9`, injected beta `3.4463442e-5`, and
  float64 cosines `.999791/1/1/.999764`; the identical float32 cosines are
  `zero/.006617/.561176/.999764`.
- Complete trained-Conv3 curve: the canonical true-dtype grid has 14 unique
  baseline, 11 ours, and 7 legacy points. All four float64 layers have cosine
  at least `.99` for baseline at base beta hat `1e-6`--`3e-4`, ours at
  `1e-8`--`1e-5` (capped at the last point), and legacy at
  `1e-10`--`1e-9`. Float32 has no all-layer point; `ConvWeight_0` is exactly
  zero at every tested beta for all three schemes. Thus the float32
  first-layer failure is not legacy-specific.
  These are directional windows only: the trained baseline shadow remains
  residual-limited at native `T=12`.
- Common-beta result: no tested common base beta works across all schemes. At
  base beta hat `3e-10`, trained Conv3 baseline/ours first-layer float64
  cosines are `.112/.419`, versus `.999791` for legacy. At `3e-8`, baseline
  is still `.543` and ours is `.999702`, while legacy's Dense cosine falls to
  `.426627` because the `4096x` scale overnudges the output.
- Displacement and finite-beta result: trained Conv3-legacy state displacement
  at the good beta is `5.19e-12/8.17e-9/1.35e-5/.01793`. At injected beta
  `.003446`, output displacement is `1.793` and Dense cosine `.426627`; at
  `.01`, they are `5.202` and `.167644`. All float64 residuals still pass. A
  two-point Richardson extrapolation of the saved Dense gradients recovers
  BPTT with cosine `.999999843` and norm ratio `.999999972`, identifying the
  failure as one-sided finite-beta bias, not an incorrect zero-beta direction.
- T/K and residual result: the fixed strong-beta T/K cross changes finite
  cosines by at most `.007988` between native and `64/64`. Raising T repairs
  trained Conv3-baseline free residuals without repairing its direction;
  trained ours and legacy pass residual gates while their float32 C0 is zero.
  At the good Conv3-legacy beta, the worst float32 projected-KKT p90 is
  `3.05e-4` and float64 p90 is at most `4.55e-13`. Residual convergence does
  not imply a perturbatively small nudge.
- FC control: literal common-base selects only FC1 legacy in native and
  `64/64`; equal-injected selects `0/18`. Best descriptive native worst-layer
  cosine falls from FC1 legacy `.99990` to FC2 legacy `.98181` and FC3 legacy
  `.31322`; native and `64/64` are materially unchanged. Convolution is not
  necessary. FC evidence is initialization-only because matched trained FC
  checkpoints do not exist.
- Interpretation: use base beta hat `3e-10` as the frozen-current float64
  legacy Conv3 pilot point. Do not launch naive float32 deep training. Select
  beta using injected current and layerwise displacement; the `.01` cap is
  residual-safe but not a small-nudge cap for amplified legacy. Treat
  full-cost EqProp as a separate learning-rule experiment.
- Guards and validation: all authoritative Conv, FC, T/K, and shadow bundles
  pass canonical artifact-hash validation, source/checkpoint and in-memory
  parameters remain unchanged, all phases restore their declared common
  state, no optimizer step occurs, and the official test split is never read.
- Limitations: one seed; Conv/FC beta and T/K replays use 64 examples, while
  the true-dtype shadow uses one fixed 16-example batch; FC is initialization
  only; the T/K cross uses the strong endpoint rather than the recommended
  Conv3-legacy beta; no EqProp optimizer step or training-accuracy claim is
  made.
- Results and analysis: [study root](../results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/),
  [review](../results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/review.md),
  [Conv beta table](../results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/conv-current-commonbase/parameter_summary.csv),
  [T/K table](../results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/conv-current-commonbase-tk/fixed_beta_tk_summary.csv),
  [FC table](../results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/fc-current-commonbase/parameter_summary.csv),
  [true-dtype heatmap](../results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/analysis/eqprop_current_commonbase_true_dtype_cosine.png),
  [trained Conv3 true-dtype curve](../results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/analysis/conv3_current_true_dtype_beta_curve.png),
  [exact Conv3 table](../results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/analysis/conv3_current_true_dtype_beta_curve.md),
  and [recommended-beta float64 table](../results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/conv-shadow-current-commonbase-3em10/parameter_precision_comparison.csv).
- Authoritative result SHA-256 values: literal Conv beta
  `9f24538070cc9eeb85b6e11d155e36e228caaebbc133949f3230e9857ad5c80c`;
  equal-injected Conv `2011fbbe0e8f67fb08fb32714d4843cd36e916bd6a49d4ce7666d3928234b946`;
  T/K `685ebceb80aececb9357a778d096ec46e33d33cf2222139122af1b3e813236d8`;
  literal FC `c73e76f1ad4d9195e8e55360e23c9d4da10d820b06f3cee9047316c91b6585cd`;
  equal-injected FC `7124617c736bc1683b7b5d261c10769cc1e325270108fee45fc04ba10a50fc0c`;
  common-base true-dtype `3e-10` shadow
  `85cd345df45300c35717262d5a52765a750c140b134a015e011a481cb2ea0144`;
  completed Conv3 curve CSV
  `369fa9f3001fa8ee82b4a7d5006d0db98ead7d322ae7aa9c763597b669310975`.

## `perfectdiode-conv123-all-amplification-adam-eqprop-bptt-beta-tk-displacement-seed0-20260810-v1` — all-scheme EqProp gradient transport

- Analyzed: 2026-08-10
- Evidence class: `ordinary_mnist_learning_algorithm_gradient_diagnostic`
- Outcome: `mixed`; positive for all three Conv1 schemes, negative for a direct
  centered-EqProp replacement of BPTT in Conv2/Conv3.
- Scientific question: How does weight-only, layerwise EqProp--BPTT gradient
  direction vary with beta, independent free/nudged phase lengths T/K, and
  amplification scheme; and how far do the positive and negative states move
  from the free state?
- Frozen setup: baseline `voltage/current=1/1`, ours `4/1`, and legacy
  `4/0.25`; Conv1/Conv2/Conv3 coherent Adam leaders at hash-verified
  reconstructed initialization and best-validation weights; one identical
  64-example/four-batch ordinary-MNIST validation cohort; centered cost nudge;
  beta `{0.003,0.01,0.03,0.1,0.25,0.5,1}`. Native T/K is `4/4`, `6/6`,
  Conv3-baseline `12/8`, and Conv3-ours/legacy `8/8`; the full Cartesian grids
  independently vary Conv1 T,K over `{4,8,16,64}`, Conv2 over
  `{6,12,24,64}`, and Conv3 T over `{8,12,16,32,64}` with K over
  `{8,16,32,64}`. Biases remain active in the dynamics but are excluded from
  gradient scoring.
- Native maximin results:

  | Architecture | Scheme | Init beta / worst cosine | Best beta / worst cosine |
  |---|---|---:|---:|
  | Conv1 | baseline | `1 / 0.998771` | `1 / 0.993006` |
  | Conv1 | ours | `1 / 0.999917` | `1 / 0.995962` |
  | Conv1 | legacy | `0.03 / 0.999936` | `0.1 / 0.974661` |
  | Conv2 | baseline | `1 / 0.202739` | `0.5 / 0.143147` |
  | Conv2 | ours | `0.5 / 0.476067` | `1 / 0.286438` |
  | Conv2 | legacy | `0.003 / 0.321898` | `0.1 / 0.005915` |
  | Conv3 | baseline | `0.25 / 0.042111` | `0.5 / 0.039319` |
  | Conv3 | ours | `0.1 / 0.057662` | `1 / 0.034167` |
  | Conv3 | legacy | `0.003 / -0.998542` | `none / dead C0` |

- Layer/beta interpretation: Conv1 baseline/ours have a readout cosine near one
  throughout and a first convolution that improves toward beta one. Legacy is
  strong only in a narrow beta window and reverses its Conv1 first-convolution
  direction at larger beta. Conv2/Conv3 later layers again align before the
  early convolutions. Trained Conv2 ours is the least-bad deep result, but its
  worst cosine is only `0.286438`; trained legacy is essentially orthogonal.
  Conv3 legacy has a dead C0 at best and an initialization C2 cosine of
  `-0.998542`.
- T/K result: with beta fixed from the shared anchor, K dominates 41/50
  complete layer grids and has range `>0.01` in 17/50. T has only one material
  effect, trained Conv3-baseline C0 (`delta-T=0.059510`,
  `delta-K=0.050393`). Beta retuning can hide K sensitivity, most clearly for
  trained Conv2 legacy. The best retuned trained Conv3-ours coordinate is
  `T/K=8/16`, beta `1`, worst cosine `0.112146`, still not viable. Conv3 legacy
  best is dead at K `8/16`; K `32/64` revives all layers only at beta `0.003`,
  with worst cosine `-0.889231/-0.095818`.
- State displacement: both post-T-free and matched-zero-K references are saved;
  the latter isolates nudging. At selected native points, displacement grows
  monotonically with depth and the concatenated global norm understates the
  output-state motion by `19.6x`--`30599x`. Baseline is nearly phase-symmetric;
  ours and legacy show strong negative-phase asymmetry. Conv2 legacy
  initialization has global matched-zero displacement `0.00476/0.804`
  (positive/negative) and output displacement `0.0931/15.7`. Trained Conv3
  baseline is the important reference exception: residual zero-nudge K
  relaxation makes post-T displacement about `76x` the actual matched-nudge
  displacement. Displacement and cosine are not monotone, so both must be
  gated layerwise.
- Interpretation: Legacy's superior BPTT training does not yield superior deep
  EqProp gradient fidelity. The failure is localized to backward transport
  through early convolutions, not merely one bad global beta or too little T.
  More K can rotate gradients but also produces asymmetric and unstable
  negative phases. Deep EqProp training should wait for a frozen-checkpoint
  intervention such as staged/layerwise nudging or an explicit feedback path,
  with joint gates for finite phases, non-dead gradients, direction, scale,
  and per-layer state displacement.
- Guards and validation: 18 checkpoint cases, 312 T/K contexts, 6,720
  parameter rows, and 36,888 state-summary rows. The canonical production
  bundle and authoritative local copy validate; all source bytes and parameter
  tensors are unchanged; optimizer-step and official-test-read guards are
  false. Of 8,736 phase records, 8,584 are finite and all 152 unstable records
  are negative; no native/shared anchor is unstable. Jean Zay job `823643`
  completed in `01:10:46`. Result SHA-256 is
  `b4071c5fcf9e167e47fed2f7d0a1169edbd703b955e929c2bb6f8ede1238e8de`.
- Limitations: one seed and one 64-example ordinary-MNIST cohort;
  initialization is deterministic reconstruction; cosine measures direction,
  not scale; and this replay does not establish EqProp training accuracy or
  paper-facing medium-affine evidence. The earlier legacy-only replay remains
  provenance but is superseded by this matched all-scheme T/K analysis for
  cross-scheme and phase-length conclusions.
- Raw results and analysis: [study root](../results/perfectdiode-conv123-all-amplification-adam-eqprop-bptt-beta-tk-displacement-seed0-20260810-v1/),
  [reviewed report](../results/perfectdiode-conv123-all-amplification-adam-eqprop-bptt-beta-tk-displacement-seed0-20260810-v1/review.md),
  [runner report](../results/perfectdiode-conv123-all-amplification-adam-eqprop-bptt-beta-tk-displacement-seed0-20260810-v1/analysis/report.md),
  [layer/beta table](../results/perfectdiode-conv123-all-amplification-adam-eqprop-bptt-beta-tk-displacement-seed0-20260810-v1/analysis/parameter_summary.csv),
  [T/K table](../results/perfectdiode-conv123-all-amplification-adam-eqprop-bptt-beta-tk-displacement-seed0-20260810-v1/analysis/fixed_beta_tk_summary.csv),
  [state table](../results/perfectdiode-conv123-all-amplification-adam-eqprop-bptt-beta-tk-displacement-seed0-20260810-v1/analysis/state_displacement_summary.csv),
  and [read-only guards](../results/perfectdiode-conv123-all-amplification-adam-eqprop-bptt-beta-tk-displacement-seed0-20260810-v1/analysis/read_only_guards.json).
- Provenance: clean scientific runtime head
  `154d80e845cbec95ef9ba08981efd3079d765ddb`; analyzer checkout
  `8a5d39c3dc10ecb683cd626fa5bc17ba5946e23b`; runner SHA-256
  `aeb8c2d9...a042d08a`, config SHA-256 `17c372d3...229b1335`, and target
  `jean-zay` V100.

## `perfectdiode-conv123-legacy-adam-eqprop-bptt-beta-cosine-seed0-20260810-v1` — EqProp/BPTT checkpoint-gradient replay

- Analyzed: 2026-08-10
- Evidence class: `ordinary_mnist_learning_algorithm_gradient_diagnostic`
- Outcome: `mixed`; positive for Conv1 gradient direction, negative for directly
  reusing the accepted BPTT phase contract in Conv2/Conv3 EqProp.
- Scientific question: At reconstructed initialization and the
  maximum-validation checkpoint of the coherent signed/scaled-bias `wide`
  legacy-Adam Conv1/Conv2/Conv3 leaders, which EqProp nudging strength best
  matches BPTT layer-by-layer when bias gradients are excluded?
- Cases and inclusion: Six checkpoint cases, comprising three deterministic
  reconstructed initializations and three best checkpoints at epochs
  `9/29/22` with ordinary-MNIST validation accuracies
  `96.86/98.42/99.04%`. Every declared case, beta, operating phase, weight
  layer, and fixed cohort batch was retained. The three source bundles and
  their configs, results, manifests, PT checkpoints, and NPZ tensors were
  hash-verified; PT and NPZ tensors match exactly. Reconstructed initialization
  includes the active signed/scaled biases and matches the previously recorded
  complete-parameter tensor digests.
- Setup: One deterministic 256-example subset (16 batches of 16, cohort SHA-256
  `034c4791...fd37229`) from the stored 5,000-example ordinary-MNIST validation
  order; no official-test read. Accepted free/gradient iterations remained
  Conv1 `4/4`, Conv2 `6/6`, and Conv3 `8/8`. Every BPTT, zero, negative, and
  positive branch started from an identical cloned post-T state. Centered
  cost-nudged EqProp was primary over
  `beta={0.003,0.01,0.03,0.1,0.25,0.5,1.0}`; positive and negative one-sided
  estimates used the same phase endpoints. `Bias_*` tensors remained loaded
  and active in all dynamics, while only `ConvWeight_*` and `DenseWeight_*`
  entered gradient summaries. No optimizer was constructed or stepped.
- Headline direction measurements: Beta selection maximized the minimum
  centered cohort-gradient cosine over weight layers for each checkpoint.

  | Architecture / checkpoint | Selected beta | Layerwise centered cosine |
  |---|---:|---|
  | Conv1 reconstructed init | `0.01` | `C0 0.999948`, `D 1.000000` |
  | Conv1 best | `0.1` | `C0 0.988112`, `D 0.998487` |
  | Conv2 reconstructed init | `0.003` | `C0 0.981387`, `C1 0.961676`, `D 0.324420` |
  | Conv2 best | `0.003` | `C0 0.020944`, `C1 0.993538`, `D 0.973267` |
  | Conv3 reconstructed init | `0.003` | `C0 0.032305`, `C1 -0.101655`, `C2 -0.998761`, `D 0.973938` |
  | Conv3 best | none | `C0` EqProp gradient is exactly zero for every beta despite nonzero BPTT |

- Gradient scale and depth: Conv1 initialization also matches scale
  (`EqProp/BPTT` norms `1.035/1.036` for `C0/D`); at its best checkpoint the
  high-direction gradients are about `1.43/1.44x` BPTT. Conv2 does not admit a
  useful all-layer match. Its selected initialization norm ratios are
  `21.52/38.84/118.80` for `C0/C1/D`, and its selected best-checkpoint ratios
  are `89.95/1.98/2.04`. At Conv3 initialization the selected `C2` gradient is
  nearly exactly reversed (`-0.998761`); at the best checkpoint, centered
  EqProp has an exactly dead `C0`, near-orthogonal `C1`, negative `C2`, and
  well-aligned `D`. No beta corrects that layerwise split. Positive and
  negative one-sided variants also fail: their best worst-layer cosines at
  Conv2 best are only `0.0439/0.0158`, and neither has a viable all-layer
  Conv3-best candidate.
- Phase-length diagnostic: On four fixed minibatches, all 18 BPTT
  checkpoint/layer comparisons pass the accepted-K versus K64 gates
  (`cosine>=0.9`, relative norm delta `<=0.1`, zero-fraction delta `<=0.02`).
  Centered EqProp passes only `16/126` rows. At the selected beta it passes
  `2/2` Conv1-init layers, `1/2` Conv1-best layers, `3/3` Conv2-best layers,
  and `0/3` plus `0/4` Conv2/Conv3-init layers. Sixteen K64 negative-phase
  executions are non-finite across Conv1 init beta `0.1`, Conv1 best beta
  `0.25/0.5`, and Conv2 best beta `0.01`; these remain explicit structured
  outcomes. K64 therefore demonstrates severe EqProp phase-length sensitivity
  but is not a uniformly viable equilibrium reference.
- Interpretation: The accepted BPTT operating points transfer to centered
  EqProp for Conv1 direction, but not with depth. The observed pattern—aligned
  deep/readout gradients alongside orthogonal, reversed, badly scaled, or dead
  early-convolution gradients—is consistent with inadequate backward signal
  transport under the present reciprocal dynamics. This is a checkpoint-level
  necessary-condition failure, so deep EqProp training should not start by
  merely swapping estimators and retaining BPTT `T/K`. The next diagnostic
  should separate positive-phase iteration count from beta and require finite,
  non-dead, direction- and scale-viable gradients in every weight layer before
  training. Extra centered K alone is not the answer because the negative phase
  becomes unstable.
- Guards and validation: The production reporting bundle validates; all source
  bytes and all six in-memory parameter states are unchanged; all 1,800 phase
  records start from the declared common-state hash; all explicit beta-zero
  endpoints pass the `1e-5` relative-L2 equivalence gate; all selected gradient
  arrays exclude biases; and official-test/optimizer-step guards are false.
  Result SHA-256 is `cc17aebe...b37e7e5`. Smoke attempts `01`--`04` are retained
  as operational failures while the corrected all-case/all-beta
  `smoke-attempt-05` validates and is excluded from production aggregation.
- Limitations: One seed, one 256-example ordinary-MNIST validation cohort, and
  only the best coherent legacy-Adam family. Initialization is deterministic
  reconstruction rather than a saved epoch-0 file. Cosine does not measure
  scale, hence the separately reported norms. K64 uses the accepted-T common
  state and is a phase-length sensitivity test, not the independent protocol
  `T=K=64` operating-point sentinel. This study measures gradients; it does not
  establish EqProp training accuracy or compare amplification schemes under
  EqProp.
- Raw results and analysis: [study root](../results/perfectdiode-conv123-legacy-adam-eqprop-bptt-beta-cosine-seed0-20260810-v1/),
  [reviewed report](../results/perfectdiode-conv123-legacy-adam-eqprop-bptt-beta-cosine-seed0-20260810-v1/review.md),
  [canonical runner report](../results/perfectdiode-conv123-legacy-adam-eqprop-bptt-beta-cosine-seed0-20260810-v1/analysis/report.md),
  [layer/beta table](../results/perfectdiode-conv123-legacy-adam-eqprop-bptt-beta-cosine-seed0-20260810-v1/analysis/parameter_summary.csv),
  [phase-length table](../results/perfectdiode-conv123-legacy-adam-eqprop-bptt-beta-cosine-seed0-20260810-v1/analysis/phase_sentinel_summary.csv),
  [phase diagnostics](../results/perfectdiode-conv123-legacy-adam-eqprop-bptt-beta-cosine-seed0-20260810-v1/analysis/phase_diagnostics.csv),
  [read-only guards](../results/perfectdiode-conv123-legacy-adam-eqprop-bptt-beta-cosine-seed0-20260810-v1/analysis/read_only_guards.json),
  and [cosine plot](../results/perfectdiode-conv123-legacy-adam-eqprop-bptt-beta-cosine-seed0-20260810-v1/analysis/cosine_by_beta_layer.png).
- Provenance: Exact compatible runtime worktree
  `/home/filip/server_code_conv_learning_rate_protocol/.codex/worktrees/signed-scaled-bias`
  was clean at `154d80e845cbec95ef9ba08981efd3079d765ddb`; frozen production
  source commit `08dd52e44818adf43b011542748e140dc0e01aeb`, source archive
  `e80f0510...ecb45ce`; analyzer SHA-256
  `6f93afd2...9d08f4`, config SHA-256 `f28e684d...dabaa0`; target `main`
  RTX 3090.

## `perfectdiode-conv123-zero-bias-lr-counterfactual-ordinary-mnist-seed0-20260807-v1` + `perfectdiode-conv123-zero-bias-baseline-conv-lr-div3-long50-ordinary-mnist-seed0-20260808-v1` — baseline LR/time counterfactuals

- Analyzed: 2026-08-09
- Evidence class: `ordinary_mnist_lr_counterfactual` and
  `ordinary_mnist_lr_duration_counterfactual`
- Outcome: `mixed` overall and negative for the hypothesis that a simple
  learning-rate rescaling or extra time explains the Conv2/Conv3 zero-bias
  baseline gap.
- Scientific question: Can baseline recover the zero-bias ours/legacy
  trajectory by tripling its convolutional rates, matching the amplified
  schemes' initial relative-proposal vectors, or dividing its convolutional
  rates by three and training for 50 epochs?
- Runs and exclusions: All 15 requested seed-0 production arms completed: four
  30-epoch baseline-SGD LR vectors for each of Conv1/Conv2/Conv3 plus one
  50-epoch Conv-LR/3 baseline per architecture. The six fresh ours/legacy
  controls at array indices `4,5,10,11,16,17` were intentionally not launched;
  the reviewed prior controls are reused only with their duration limits made
  explicit.
- Setup and guards: Ordinary-MNIST 55,000/5,000 train/validation split,
  accepted `T/K`, exact-zero bias rates, `[0,100]` conductance projection, five
  traced transitions per epoch, every epoch checkpointed, and no official-test
  access. The authoritative local copies contain 15/15 successful bundles and
  510/510 epoch records. Canonical and semantic validation re-hashed every
  indexed artifact and checkpoint and found every saved bias tensor exactly
  zero. The repeated current-LR metric values exactly reproduce all matched
  epochs of the prior baseline controls.
- Thirty-epoch measurements: Best validation accuracy for current / Conv-LR x3
  / ours-proposal / legacy-proposal is
  `96.60/96.56/96.60/96.66%` for Conv1,
  `97.22/96.82/97.08/97.00%` for Conv2, and
  `97.04/96.56/96.92/96.86%` for Conv3. Current is best at Conv2/Conv3;
  tripling Conv rates changes those two values by `-0.40/-0.48` points.
- Duration measurements: Conv-LR/3 best accuracy through epoch 30 and epoch 50
  is `96.62/96.64%`, `97.16/97.30%`, and `96.44/96.78%` for Conv1/2/3.
  Epochs 31--50 therefore add only `+0.02/+0.14/+0.34` points. Relative to the
  current 30-epoch baseline, the 50-epoch values are `+0.04/+0.08/-0.26`
  points.
- Scheme context: Even with 50 epochs, the best Conv2 baseline remains
  `0.68/0.66` points below the prior 30-epoch ours/legacy SGD controls. Conv3's
  best tested baseline remains `1.16/1.72` points below those controls, and the
  Conv-LR/3 run itself is `1.42/1.98` points below. Conv1's longer baseline
  exceeds the prior 10-epoch amplified controls, but that comparison is not
  duration-matched.
- Interpretation: Conv1 is insensitive across the tested vectors (a complete
  0.10-point span), so its small earlier gap can disappear with extra epochs.
  At Conv2/Conv3, larger and proposal-matched rates do not transfer the
  amplified schemes' behavior into baseline coordinates. Lower rates plus
  extra time give a modest Conv2 gain and fail at Conv3. The depth-dependent
  separation is therefore unlikely to be ordinary baseline under-training at
  these rates. The next useful step is read-only analysis of the saved traces
  and checkpoints, not another longer Conv3 /3 allocation.
- Limitations: One seed and one fixed validation cohort; no uncertainty
  estimate; ordinary-MNIST diagnostic only. The Conv-LR/3 arm combines a rate
  change with a longer total budget, while its within-arm epoch-30/50 contrast
  isolates the added time. Reused Conv2/Conv3 controls are protocol- and
  duration-matched; reused Conv1 controls have only 10 epochs.
- Raw results and analysis: [30-epoch study](../results/perfectdiode-conv123-zero-bias-lr-counterfactual-ordinary-mnist-seed0-20260807-v1/),
  [50-epoch study](../results/perfectdiode-conv123-zero-bias-baseline-conv-lr-div3-long50-ordinary-mnist-seed0-20260808-v1/),
  [reviewed report](../results/perfectdiode-conv123-zero-bias-baseline-conv-lr-div3-long50-ordinary-mnist-seed0-20260808-v1/analysis/report.md),
  [run table](../results/perfectdiode-conv123-zero-bias-baseline-conv-lr-div3-long50-ordinary-mnist-seed0-20260808-v1/analysis/summary.csv),
  [curve plot](../results/perfectdiode-conv123-zero-bias-baseline-conv-lr-div3-long50-ordinary-mnist-seed0-20260808-v1/analysis/plots/validation_accuracy_by_epoch.png),
  and [validation receipt](../results/perfectdiode-conv123-zero-bias-baseline-conv-lr-div3-long50-ordinary-mnist-seed0-20260808-v1/analysis/validation_receipt.json).
- Provenance: Jean Zay `fmu@v100` jobs `747602` and `748806`; staged commits
  `b90099fab1901bd3ceaf25385dc6ef064c600caa` and
  `338b5a57b13c227d3d8de6d9259b4b0697c43952`.

## `perfectdiode-conv123-zero-bias-ordinary-mnist-seed0-20260805-v1` — all-depth zero-bias training diagnostic

- Analyzed: 2026-08-06
- Evidence class: `ordinary_mnist_bias_ablation`
- Outcome: `positive`
- Scientific question: Can Conv1/Conv2/Conv3 baseline, ours, and legacy train
  with every hidden bias initialized at zero and frozen there when all accepted
  weight learning rates are retained unchanged?
- Runs and seeds: All 18 seed-0 production arms completed: three
  architectures x three schemes x SGD/Adam. Conv1 ran 10 epochs and
  Conv2/Conv3 ran 30. Eighteen one-batch local smokes were excluded. All 36
  canonical bundles validate.
- Setup and guards: Deterministic ordinary-MNIST 55,000/5,000
  train/validation split, no affine corruption, `[0,100]` conductance
  projection, accepted `T/K` values, and no official-test access. Independent
  checks compared all 18 configs with their sources, confirmed every non-bias
  learning rate unchanged, and found every bias learning rate and all bias
  tensors in the 36 best/final production NPZ checkpoints exactly zero. All
  checked numeric checkpoint arrays are finite.
- Headline measurements: Every arm learns strongly; best validation accuracy
  spans `95.92%`--`99.00%`. Taking the better optimizer per scheme gives
  baseline/ours/legacy values of `96.28%/96.50%/96.44%` for Conv1,
  `97.36%/98.10%/98.46%` for Conv2, and `97.80%/98.66%/99.00%` for Conv3.
  Under matched SGD, legacy minus ours is `-0.58/-0.02/+0.56` points from
  Conv1 to Conv3; under Adam it is `0.00/+0.36/+0.34` points.
- Curve and mechanism follow-up: Ours exceeds baseline validation accuracy at
  all `140/140` matched epochs. Legacy exceeds ours throughout Conv3 and Conv2
  Adam, but not Conv1 or Conv2 SGD. An exact zero-bias coordinate transform
  gives effective edge weights `W'_l=g^l W_l`, with Conv3 multipliers
  `[1,1,1,1]`, `[1,4,16,64]`, and `[1,16,256,4096]`. Read-only replay on one
  deterministic 128-example cohort retained all checkpoint and parameter
  bytes. Reconstructed initialization has nearly identical scale-invariant
  class separation across schemes; at best Conv3 checkpoints the deepest
  between/within-class variance orders baseline below ours below legacy. The
  initial Conv3-SGD layerwise relative-proposal spread is
  `41.6x/5.3x/4.6x` for baseline/ours/legacy. A separate guarded endpoint
  comparison finds Conv3-SGD baseline/ours/legacy final total raw conductance
  `13277.6/15068.0/4125.9`, endpoint absolute displacement
  `15318.2/16262.2/4663.3`, and exact-zero occupancy
  `7.91%/2.70%/0.92%`. These are raw-coordinate endpoint facts, not training
  path length, write energy, or circuit power.
- Interpretation: Trainable biases are not necessary for strong ordinary-
  MNIST learning at the accepted weight rates. The amplified schemes separate
  more clearly from baseline with depth, and legacy retains a small Conv2/3
  Adam lead plus a Conv3 SGD lead in this diagnostic. This is evidence about
  optimization without biases, not evidence for the proposed signed/scaled
  contract or paper accuracy. The zero-bias schemes have the same
  classification function family after the coordinate change; the depth trend
  is consistent with effective-conductance, fixed-target output-scale, and
  learning-rate conditioning rather than extra feed-forward capacity.
- Limitations: One model/loader seed and one fixed 5,000-example validation
  cohort; no uncertainty estimate or official-test read; architecture budgets
  differ by protocol; ordinary-MNIST validation accuracy is never
  paper-facing. Any checkpoint reuse now requires the separate fail-closed
  paper audit and once-only official-test evaluation. The
  mechanism replay uses 128 examples and current-source seed reconstruction
  because no production initialization checkpoint/hash was saved; its
  associations are not a causal matched-coordinate ablation.
- Raw results and analysis: [authoritative local study](../results/perfectdiode-conv123-zero-bias-ordinary-mnist-seed0-20260805-v1/),
  [reviewed report](../results/perfectdiode-conv123-zero-bias-ordinary-mnist-seed0-20260805-v1/analysis/report.md),
  [mechanism report](../results/perfectdiode-conv123-zero-bias-ordinary-mnist-seed0-20260805-v1/analysis/mechanism/report.md),
  [run table](../results/perfectdiode-conv123-zero-bias-ordinary-mnist-seed0-20260805-v1/analysis/summary.csv),
  and [machine summary](../results/perfectdiode-conv123-zero-bias-ordinary-mnist-seed0-20260805-v1/analysis/summary.json).
- Provenance: Jean Zay `fmu@v100` array `604461_[0-17]`; frozen dirty-worktree
  snapshot based at commit `ac941dff`; source archive SHA-256
  `382df19e2baaea3b5d544a9e38fc5a348c7e1f8d6868254fcb2176d8c7ad213b`;
  ordered config-set SHA-256
  `0d29871f8393c6052244c8478a1317a6780800a434bf225904aa198a1b017483`.

## `perfectdiode-conv23-legacy-zero-bias-medium-affine-seed0-20260731-v1` — legacy training necessity of biases

- Analyzed: 2026-08-06
- Evidence class: `diagnostic_medium_affine_bias_ablation`
- Outcome: `positive`
- Scientific question: Does legacy's large post-training sensitivity to its
  learned deepest biases imply that trainable biases are necessary, or can its
  weights compensate when Conv2/Conv3 are trained from initialization with all
  biases fixed at zero?
- Runs and seeds: Two seed-0 legacy-SGD production arms, Conv2 and Conv3, each
  for 30 epochs. Four smoke/canary bundles were excluded. All six retained
  bundles validate. The original Conv3 array task `490130_1` failed before
  training because the config index was applied twice and produced no
  scientific output; corrected live canary `490690_1` passed and unchanged
  production replacement `490787_1` completed.
- Setup and guards: Deterministic medium-affine MNIST, original `[0,100]`
  weight contract and exact weight learning rates, every hidden-bias rate and
  checkpoint tensor exactly zero, five gradient-trace samples per epoch, and
  epoch checkpoints. The official test was deliberately skipped.
- Headline measurements: Conv2 zero-bias best validation accuracy is `91.58%`
  versus `91.42%` in the original learned-bias run, with final loss
  `0.164916` versus `0.165221`. Conv3 is `96.82%` versus `96.70%`, with final
  loss `0.081811` versus `0.082174`. The accuracy differences are only eight
  and six predictions out of 5,000, respectively, and all four runs select
  epoch 30.
- Interpretation: There is no observed zero-bias performance deficit. Legacy
  weights compensate when biases are absent from the start, so the severe
  post-hoc checkpoint intervention establishes checkpoint dependence but not
  training necessity. The historical unscaled-bias contract remains a
  scientific confound, but it is not by itself an adequate explanation for
  legacy's Conv2/Conv3 SGD advantage.
- Limitations: Legacy + SGD only, one seed, validation-only comparison, and no
  baseline/ours or Adam medium-affine zero-bias controls. This does not test
  signed or amplification-scaled biases and does not replace official-test
  paper rows.
- Raw results and analysis: [authoritative local study](../results/perfectdiode-conv23-legacy-zero-bias-medium-affine-seed0-20260731-v1/),
  [reviewed report](../results/perfectdiode-conv23-legacy-zero-bias-medium-affine-seed0-20260731-v1/analysis/report.md),
  [matched table](../results/perfectdiode-conv23-legacy-zero-bias-medium-affine-seed0-20260731-v1/analysis/summary.csv),
  and [machine summary](../results/perfectdiode-conv23-legacy-zero-bias-medium-affine-seed0-20260731-v1/analysis/summary.json).
- Provenance: Jean Zay `umg@v100`; source commit `ade6248a`; archive SHA-256
  `f7f9d80b439ba8f0f4d1a3ed1e1dde4b63135f4b5e9fc5e028e9c76ee4e1df5a`;
  initial array `490130_[0-1]` and Conv3 production replacement `490787_1`.

## `perfectdiode-conv12-bounded-uniform-signed-bias-rho-seed0-v1` — signed-only bounded bias screen

- Analyzed: 2026-08-06
- Evidence class: `ordinary_mnist_selection_mechanism_diagnostic`
- Outcome: `mixed`
- Scientific question: With bounded-uniform conductances, do signed,
  still-unscaled biases improve the three-epoch Conv1/Conv2 baseline/ours
  operating points, and are the searched bias-rate ranges sufficient?
- Runs and seeds: Seed 0; 53 production cells across eight Conv1/Conv2 x
  baseline/ours x SGD/Adam surfaces plus two excluded smokes. All 55 bundles
  validate, all eight surfaces publish one selected checkpoint, and the
  official test was not read.
- Setup: Ordinary MNIST, fixed accepted `T=K=4/6`, conductance bounds
  `[1e-5,1e-4]`, bounded-uniform initialization, signed unclamped biases, and
  three epochs per cell. Conductance rho pairs came from a finite previously
  useful shortlist while bias rho scales were screened independently. This
  implementation does not apply depth-dependent amplification scaling to
  biases.
- Headline measurements: Every selected checkpoint uses both bias signs;
  `36.99%`--`48.96%` of biases are negative. Relative to the recorded
  nonnegative selections, all SGD changes lie in `[-0.06,+0.08]` points,
  while three Adam surfaces gain `+1.06`--`+1.84` points and Conv1 ours/Adam
  is unchanged. Conv2 ours/Adam reaches `89.64%`, the largest gain
  (`+1.84` points). Five of eight surfaces select the tested upper bias-scale
  edge `9`, so those bias ranges remain open. The worst projection efficiency
  is `0.228` for Conv2 baseline/Adam; Conv2 ours/SGD remains high at `0.959`
  and changes only `+0.08` points.
- Interpretation: Signed biases are actively used and targeted Adam rows may
  benefit from a separately selected bias rate. They do not repair the Conv2
  ours/SGD shortfall, and the open upper edges prevent terminal handoff. The
  comparison changes bias sign and rate together; two selected rows also use
  alternate shortlisted conductance-rho pairs, so it is not a pure causal
  sign ablation.
- Limitations: One seed, three epochs, ordinary-MNIST selection evidence,
  bounded-uniform only, no legacy or Conv3, and no amplification-scaled bias
  energy. It cannot select the global bounded initializer or validate the
  proposed corrected bias contract.
- Raw results and analysis: [authoritative local study](../results/perfectdiode-conv12-bounded-uniform-signed-bias-rho-seed0-v1/),
  [reviewed report](../results/perfectdiode-conv12-bounded-uniform-signed-bias-rho-seed0-v1/analysis/report.md),
  [surface table](../results/perfectdiode-conv12-bounded-uniform-signed-bias-rho-seed0-v1/analysis/surface_summary.csv),
  and [machine summary](../results/perfectdiode-conv12-bounded-uniform-signed-bias-rho-seed0-v1/analysis/summary.json).
- Provenance: clean implementation commit `3564bd55`; working-tree SHA-256
  `a6f642dfbab3ab923b5b608a0c953eafad55e345889756e9c2b5312505515618`.

## `perfectdiode-paper-gradient-trace-limited-20260731-v1` — optimizer-state and depth-wise gradient/update diagnosis

- Analyzed: 2026-07-31
- Evidence class: `diagnostic_medium_affine_gradient_trace`
- Outcome: `mixed`
- Scientific question: Are the apparent seed-0 SGD advantages caused by an
  optimizer learning-rate mismatch, and does legacy's Conv2/Conv3 advantage
  coincide with abnormal gradients, projection failure, or better layerwise
  step conditioning?
- Runs and seeds: Twelve seed-0 production diagnostics: all six Conv1
  scheme/optimizer rows and baseline/ours/legacy SGD for Conv2 and Conv3. Every
  row executed 10 epochs, recorded 50 optimizer transitions, and saved model
  plus optimizer checkpoints at epochs 0--10. The passing one-batch Jean Zay
  canary was excluded. All 13 bundles validate; all diagnostic metric prefixes
  are bit-identical to the source paper trajectories.
- Setup and provenance: Deterministic medium-affine MNIST with the exact paper
  configurations, frozen commit `ade6248a5e9d80ec8157b8a5273f095e6b30715b`,
  source archive SHA-256
  `f7f9d80b439ba8f0f4d1a3ed1e1dde4b63135f4b5e9fc5e028e9c76ee4e1df5a`,
  and Jean Zay `umg@v100`. The trace recorded raw gradients, real optimizer
  proposals, post-projection applied updates, fresh-state rho shadows, bounds,
  and Adam moments. All 132 epoch checkpoints were replayed read-only on one
  frozen 256-example validation cohort; source bytes and parameter states were
  unchanged.
- Rho and weight contract: Every included row uses the wide/unbounded rho
  handoff family; none belongs to the separate bounded-weight initializer/rho-
  selection study whose conductance interval is `[1e-5,1e-4]`. “Unbounded”
  describes selector provenance, not complete rho bracketing: most Conv1/2
  surfaces remained at an outer search edge. Downstream conductances are still
  projected to `[0,100]`. Conv3 retains the caveat that its rates came from a
  `weight_max=null` source selection before transfer to `[0,100]`.
- Headline measurements: Across the original nine matched scheme/optimizer
  paper comparisons, Adam has lower official-test loss in all nine and higher
  accuracy in seven. SGD's only accuracy wins are Conv1 baseline and ours, by
  `0.74` and `0.84` points; their best-validation differences are only `0.12`
  and `0.10` points at one seed. In the Conv1 traces, real stateful-Adam
  proposals have median
  `0.211` times the fresh rho-shadow RMS and median direction cosine `0.312`,
  with `38.2%` momentum-sign disagreement. Conv3 median applied-update/current-
  weight spreads are `14.95x/8.55x/3.08x` for baseline/ours/legacy. On the
  fixed cohort, epoch-10/epoch-0 raw-gradient RMS for `C1/C2` is
  `38.7x/81.0x`, `4.33x/8.64x`, and `2.10x/3.33x`, respectively. Gradients
  remain finite and essentially dense. Median projection efficiency stays
  high, no upper-bound occupancy appears, and lower-bound pressure is least in
  legacy.
- Interpretation: The results do not support a general claim that SGD
  outperforms Adam. They expose a structural Adam-calibration mismatch: rho was
  evaluated with a fresh Adam shadow, whereas training uses accumulated
  moments, producing both a smaller and differently directed real step. A
  simple global LR multiplication is therefore not justified. Legacy's deeper
  advantage is associated with more conservative, better-balanced effective
  layerwise updates and lower boundary pressure. Given the strongly
  scheme-specific LR and bias vectors, this is more consistent with
  scheme-by-LR/bias conditioning than intrinsic superiority of legacy.
  Adam transition tracing is Conv1-only, so the mismatch magnitude is not
  directly measured for Conv2/Conv3.
- End-of-training bias follow-up: A guarded fixed-cohort replay of all six
  Conv2/Conv3 SGD best and final checkpoints shows that the legacy advantage is
  strongly dependent on its deepest learned bias, despite that tensor being
  smaller in raw RMS. Removing only the deepest bias changes the deepest state
  by `25.41%` (Conv2) and `67.01%` (Conv3) of learned-state RMS and changes
  `17.19%` and `16.41%` of predictions. Removing every bias reduces diagnostic
  fixed-cohort accuracy by `9.38` and `17.58` points, whereas baseline and ours
  change by at most one of 256 examples. The trained local-curvature estimate
  predicts the same scheme/depth ordering: legacy's deepest coordinate shift
  is `41.0%` (Conv2) and `158.7%` (Conv3) of learned-state RMS. The mechanism is
  structural: the linear bias drive is unscaled while legacy's deep resistive
  curvature is strongly attenuated. Retaining only spatial channel means keeps
  most legacy accuracy, with losses of `0.78` and `1.56` points. This establishes
  checkpoint dependence, not training necessity. The subsequent matched
  zero-bias legacy retrains reached essentially unchanged Conv2/Conv3
  validation accuracy and are reviewed separately above.
- Convergence: All source paper runs and diagnostics completed their fixed
  budgets with finite measurements, but completion is not optimizer
  convergence. Conv1 peaks around epochs 8--9 and fluctuates thereafter; nine
  of the 12 original Conv2/Conv3 surfaces select their best checkpoint at
  epochs 28--30 of 30. The 10-epoch deep diagnostics are prefixes, not
  convergence tests. Accepted `T/K` gates support the operating points, but
  per-step circuit residuals were not recorded, so direct equilibrium
  convergence throughout training is not established here.
- Limitations and deviations: One seed; five sampled optimizer steps per
  epoch; Conv2/Conv3 optimizer-step traces are SGD-only; fixed-cohort replay is
  associative rather than causal. Official-test values come from the original
  paper bundles—the diagnostics did not read the official test. Conv2 lacks
  long LR confirmation, and Conv3 retains the source `weight_max=null` to
  downstream `[0,100]` mismatch. A decisive comparison requires stateful Adam
  calibration or a matched effective-update/per-layer LR sweep including
  biases and multiple seeds.
- Raw results: [authoritative local study](../results/perfectdiode-paper-gradient-trace-limited-20260731-v1/);
  remote source
  `/lustre/fsn1/projects/rech/umg/ucy17uy/server_code/results/perfectdiode-paper-gradient-trace-limited-20260731-v1`.
- Analysis: [reviewed report](../results/perfectdiode-paper-gradient-trace-limited-20260731-v1/analysis/report.md),
  [study summary](../results/perfectdiode-paper-gradient-trace-limited-20260731-v1/analysis/summary.json),
  [source inventory](../results/perfectdiode-paper-gradient-trace-limited-20260731-v1/analysis/summary.csv),
  [online trace summary](../results/perfectdiode-paper-gradient-trace-limited-20260731-v1/analysis/online_trace/gradient_trace_summary.json),
  [anomaly inventory](../results/perfectdiode-paper-gradient-trace-limited-20260731-v1/analysis/online_trace/gradient_trace_anomalies.json),
  [fixed-cohort gradient replay](../results/perfectdiode-paper-gradient-trace-limited-20260731-v1/analysis/fixed_validation_replay/analysis_summary.json),
  and [end-of-training bias-effect report](../results/perfectdiode-paper-gradient-trace-limited-20260731-v1/analysis/bias_implementation_audit/paper_end_bias_effect/report.md).

## `perfectdiode-paper-medium-affine-wide-seed0-20260729-v1-analysis` — Conv1/Conv2 final weights and projection pressure

- Analyzed: 2026-07-31
- Evidence class: `paper_medium_affine_mechanism_diagnostic`
- Outcome: `mixed`
- Scientific question: Do the final `[0,100]` conductance distributions of the
  seed-0 Conv1/Conv2 baseline, ours, and legacy paper models show boundary
  pinning or poor gradient/update projection that explains their accuracy
  differences?
- Runs and seeds: All 12 production rows for Conv1/Conv2 x
  baseline/ours/legacy x SGD/Adam, seed 0. The 12 locally staged reporting
  bundles validated, and all 48 checkpoint artifacts matched their recorded
  size and SHA-256. Conv1/Conv2 smoke bundles were excluded.
- Setup: Deterministic medium-affine MNIST paper configs with the original
  fixed handoffs, `T=K=4` for Conv1 and `T=K=6` for Conv2, and conductance
  projection to `[0,100]`. The exact production source commit `1fd3e034` was
  used to reconstruct the seed-0 initializer and replay every final checkpoint
  without optimizer steps on the same first 16 training minibatches (256
  examples). Source checkpoint hashes and replayed tensors were unchanged.
- Headline measurements: The reconstructed initializer has `49.92%` (Conv1)
  and `49.86%` (Conv2) exact lower-bound occupancy. Every final model moved
  away from that initial atom: aggregate final occupancy is `0.17%`--`9.35%`
  for Conv1 and `0.79%`--`25.36%` for Conv2. No final weight equals the upper
  bound; the maximum is `65.30`. The worst layer median retains `96.7%` of raw
  tangent-gradient L2 norm. Exact momentum-free SGD proposal efficiency is at
  least `97.1%`. The strongest proxy clipping is Conv2 baseline Adam, where a
  fresh-moment Adam proposal retains `87.1%`--`88.5%` L2 and clips
  `24.7%`--`27.0%` of components.
- Interpretation: The general boundary-pinning/projection-collapse hypothesis
  is not supported at the final checkpoints. The clearer anomaly is the
  split, inflated Conv2 baseline distribution: its three layer RMS values are
  `37.4x/11.5x/22.9x` initialization for SGD and `53.1x/7.0x/19.2x` for Adam.
  Conv2 baseline-Adam's top 10% of first-layer output-channel filters carry
  `60.7%` of layer L2 energy. This scale and filter imbalance is more
  consistent with the poor baseline result than an active upper ceiling,
  while remaining associative rather than causal.
- Limitations: One seed; final-checkpoint replay rather than a recorded
  training trajectory; local float32 CPU replay rather than V100. Historical
  Adam optimizer moments were not saved, so fresh-Adam proposal efficiency is
  a labeled proxy. Official test accuracy belongs to the best-validation
  checkpoint; distributions and replay conclusions use final checkpoints.
  The separate `[1e-5,1e-4]` ordinary-MNIST bounded-rho studies are not mixed
  into this comparison.
- Raw results: Jean Zay
  `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/perfectdiode-paper-medium-affine-wide-seed0-20260729-v1`.
- Analysis:
  [reviewed report](../results/perfectdiode-paper-medium-affine-wide-seed0-20260729-v1-analysis/final_weight_projection/report.md),
  [surface summary](../results/perfectdiode-paper-medium-affine-wide-seed0-20260729-v1-analysis/final_weight_projection/summary.csv),
  [layer distributions](../results/perfectdiode-paper-medium-affine-wide-seed0-20260729-v1-analysis/final_weight_projection/weight_distribution_by_layer.csv),
  [projection table](../results/perfectdiode-paper-medium-affine-wide-seed0-20260729-v1-analysis/final_weight_projection/gradient_projection_summary.csv),
  [weight histograms](../results/perfectdiode-paper-medium-affine-wide-seed0-20260729-v1-analysis/final_weight_projection/final_weight_histograms.png),
  and
  [projection diagnostics](../results/perfectdiode-paper-medium-affine-wide-seed0-20260729-v1-analysis/final_weight_projection/final_gradient_projection.png).

## `perfectdiode-conv12-bounded-rho-checkpoint-analysis-seed0-v1` — bounded Conv1/Conv2 checkpoint mechanisms

- Analyzed: 2026-07-31
- Evidence class: `ordinary_mnist_selection_mechanism_diagnostic`
- Outcome: `mixed`
- Scientific question: Do the selected `[1e-5,1e-4]` bounded-rho Conv1/Conv2
  checkpoints fail because conductances accumulate at the physical limits or
  because projection removes most proposed updates, and do any failures have a
  different gradient mechanism?
- Runs and seeds: Seed 0; all 16 canonical baseline/ours selections across
  Conv1/Conv2, two bounded initializers, and SGD/Adam, plus the five completed
  historical legacy selections (four Conv1 and bounded-uniform/SGD Conv2).
  Initialization, best-validation, and final checkpoints gave 63 guarded
  checkpoint roles. Historical legacy remains a separate, incomplete evidence
  panel.
- Setup: Ordinary MNIST, `T=K=4` for Conv1 and `T=K=6` for Conv2, physical
  conductance projection to `[1e-5,1e-4]`, and no official-test access. Raw
  gradients were replayed without optimizer steps on the same first 16
  deterministic training minibatches (256 examples). All source result and
  selection hashes matched; all 63 source files and replayed tensor states were
  unchanged. SGD proposal replay is exact for the saved zero-momentum,
  zero-weight-decay contract. Adam moments were not saved, so only recorded
  training projection diagnostics are authoritative for Adam; fresh-state
  proposal replay is labeled as a shadow.
- Headline measurements: Whole-model exact-bound occupancy never exceeds
  `41.96%`, so no final model has a majority of all conductances exactly pinned.
  Pinning is nevertheless severe within specific Conv2 layers: the maximum is
  `94.44%`, and baseline/Adam has `92.26%`--`93.40%` exact occupancy in
  `ConvWeight_1`. Conv2 baseline/Adam records only `0.091`--`0.096` aggregate
  projection efficiency; its first two convolution tensors retain as little as
  `0.006`--`0.079` of proposed RMS during training and `0.382`--`0.416` of
  feasible raw-gradient L2 at the final checkpoint. Matched ours/Adam lowers
  aggregate occupancy by `4.48`--`16.27` points, raises recorded projection
  efficiency by `0.320`--`0.403`, and improves accuracy by `2.50`--`2.84`
  points.
- Failure diagnosis: Kaiming/ours/SGD Conv2 is the clear non-clipping failure.
  It remains at `10.08%`; all conductance tensors have zero median proposed
  update RMS, final `ConvWeight_0` has zero gradient on all 16 replay batches,
  and the deeper tensors have zero gradient on 15 of 16. Their means moved
  upward to normalized conductance positions `0.561` and `0.569`. The raw
  conductance rates are approximately `304x/456x/127x` the corresponding
  successful bounded-uniform/ours/SGD rates. This is consistent with an early
  oversized operating-point shift followed by dead-gradient collapse, not
  projection clipping. The high recorded projection ratio is conditional on
  nonzero proposals and is therefore not evidence of active learning here.
- Broader interpretation: Conv1 ours improves all four matched rows by
  `2.38`--`3.34` points while reducing bound occupancy and modestly improving
  projection. Historical legacy Conv1 combines `94.46%`--`95.60%` accuracy,
  `0.64%`--`6.12%` aggregate exact occupancy, and `0.931`--`0.983` projection
  efficiency, but its predecessor training policy prevents a causal matched
  claim. Historical bounded-uniform/legacy/SGD Conv2 and active
  bounded-uniform/ours/SGD have nearly equal training projection efficiency
  (`0.957` versus `0.958`) despite `93.80%` versus `85.80%` accuracy, so clipping
  cannot explain the full historical gap. Every non-collapsed active Conv2 row
  selects epoch three as best, consistent with the short selection budget also
  limiting accuracy.
- Rho-range status: Four active choices remain boundary-selected: both Conv1
  Kaiming/SGD rows at lower/lower, Conv2 Kaiming/baseline/SGD at dense lower,
  and Conv2 bounded-uniform/ours/SGD at dense lower. The other 12 active choices
  are bracketed (`unbounded` in the search terminology). Bound occupancy and
  projection efficiency remained report-only and excluded no checkpoint.
- Limitations and deviations: One seed and three training epochs; ordinary
  MNIST is diagnostic rather than paper-facing evidence; only
  initialization/best/final checkpoints exist; historical coverage and policy
  are unmatched; checkpoint replay establishes association rather than
  causality. The first local smoke attempt failed before replay because the
  optional OpenMP workaround hid CUDA on this host; the second completed replay
  but exposed an SGD-only report-builder bug. Both are retained, and the fixed
  third smoke passed all nine guards before the full run.
- Analysis:
  [reviewed report](../results/perfectdiode-conv12-bounded-rho-checkpoint-analysis-seed0-v1/analysis/report.md),
  [surface summary](../results/perfectdiode-conv12-bounded-rho-checkpoint-analysis-seed0-v1/analysis/surface_summary.csv),
  [layer distributions](../results/perfectdiode-conv12-bounded-rho-checkpoint-analysis-seed0-v1/analysis/weight_distribution_by_layer.csv),
  [recorded training projection](../results/perfectdiode-conv12-bounded-rho-checkpoint-analysis-seed0-v1/analysis/training_projection_summary.csv),
  [gradient replay](../results/perfectdiode-conv12-bounded-rho-checkpoint-analysis-seed0-v1/analysis/checkpoint_gradient_summary.csv),
  [checkpoint guards](../results/perfectdiode-conv12-bounded-rho-checkpoint-analysis-seed0-v1/analysis/checkpoint_guards.json),
  and
  [plots](../results/perfectdiode-conv12-bounded-rho-checkpoint-analysis-seed0-v1/analysis/).
- Provenance: implementation commits `7b03095b` and `6e7cc751`; the full
  replay manifest records a clean `6e7cc751` checkout and the frozen cohort
  hash `ffac016d871e7c49c2574c8dfe00b892bb38865e1f533458d2cb83c5362da6dc`.

## `perfectdiode-conv2-bounded-rho-continuation-wave3-seed0-v1` — bounded Conv2 rho continuation and weight-distribution diagnosis

- Analyzed: 2026-07-30
- Evidence class: `ordinary_mnist_selection`
- Outcome: `mixed`
- Scientific question: After the initial bounded Conv2 baseline/ours rho
  search failed to reach 90% validation accuracy, do factor-of-three rho
  extensions locate better three-epoch operating points, and do the selected
  conductance distributions explain the remaining shortfall?
- Runs and seeds: Seed 0; two bounded initializers, baseline and ours, SGD and
  Adam. Wave two added 35 terminal cells over eight surfaces. User-directed
  wave three added 30 terminal cells over the six wave-two boundary-selected
  surfaces; all 30 canonical wave-three bundles validated. Legacy, Conv1, and
  Conv3 were excluded. Bound occupancy and projection efficiency were
  report-only diagnostics.
- Setup: Ordinary MNIST 55,000/5,000 deterministic train/validation split;
  official test split not read; Conv2 perfect-diode BPTT with fixed
  `T=K=6`; input gain 100; channels `[64,128]`; asynchronous minimizer;
  batch size 16; validation batch size 64; three epochs per cell. Conductance
  weights were projected to `[1e-5,1e-4]`. Initializers were full-range
  bounded uniform and bounded Kaiming uniform. Schemes were baseline
  `(voltage_amp=1,current_amp=1)` and ours
  `(voltage_amp=4,current_amp=1)`.
- Headline measurements: No surface reached the 90% reporting threshold.
  “Unbounded” below means that the selected rho is bracketed by the tested
  search range, not that the conductance weights lacked physical bounds.

  | Initializer | Scheme | Optimizer | `rho_conv` | `rho_dense` | Validation accuracy | Rho range |
  |---|---|---|---:|---:|---:|---|
  | bounded uniform | baseline | SGD | 0.001 | 0.000370370 | 77.28% | unbounded/bracketed |
  | bounded uniform | baseline | Adam | 0.081 | 0.00333333 | 85.30% | unbounded/bracketed |
  | bounded uniform | ours | SGD | 0.000333333 | 0.000123457 | 85.80% | bounded at dense lower edge |
  | bounded uniform | ours | Adam | 0.027 | 0.01 | 87.80% | unbounded/bracketed |
  | bounded Kaiming | baseline | SGD | 0.000111111 | 0.00111111 | 63.66% | bounded at dense lower edge |
  | bounded Kaiming | baseline | Adam | 0.081 | 0.00333333 | 85.26% | unbounded/bracketed |
  | bounded Kaiming | ours | SGD | 0.003 | 0.01 | 10.08% | unbounded/bracketed |
  | bounded Kaiming | ours | Adam | 0.081 | 0.03 | 88.10% | unbounded/bracketed |

- Weight-distribution measurements: The Kaiming initializer starts
  `ConvWeight_1` and `DenseWeight_0` in very narrow normalized supports
  `[0.449,0.551]` and `[0.485,0.515]`, respectively, inside the physical
  interval. Every 85%--88% selected row moved the final deeper-layer means
  downward: `ConvWeight_1` ended at `1.83e-5`--`2.47e-5`. The two weak
  Kaiming-SGD rows did not: baseline/SGD ended at `5.09e-5`, while
  ours/SGD shifted upward to `6.05e-5`; its dense mean also shifted upward to
  `6.12e-5`. The latter row remained at chance from epoch one
  (`10.04%,10.08%,10.08%`) with loss approximately `0.5`.
- Interpretation: Exact bound occupancy is not the primary explanation.
  Strong Adam rows have as much as 80%--93% exact occupancy in an early
  convolutional layer, and useful low-bound accumulation accompanies the
  downward deeper-layer shift. The clearest pathological case is
  Kaiming/ours/SGD: initializer-specific proposal normalization produced raw
  rates of `4.85e-5`, `3.27e-3`, and `8.47e-5` for the two convolutional and
  dense weights. These are roughly 304x, 456x, and 127x the corresponding
  rates in the successful bounded-uniform/ours/SGD selection. The final
  conductances are broadly redistributed but carry no useful class signal,
  and neighboring cells either remained at chance or encountered sustained
  gradient-RMS rejection. For the other seven selections, the best epoch was
  epoch three and validation accuracy was still increasing. Their shortfall
  from 90% is therefore primarily consistent with the deliberately short
  training budget; it is not evidence that projection alone prevents
  learning. Kaiming/baseline/SGD also remains rho-bounded and improved from
  `20.12%` to `48.46%` to `63.66%`, so it is visibly under-trained as well as
  poorly positioned.
- Limitations and protocol deviations: This is a one-seed, three-epoch
  selection diagnostic and is not paper-facing accuracy evidence. Weight
  distributions show association, not causality; a read-only gradient replay
  or longer matched confirmation would be needed to distinguish persistent
  gradient imbalance from an early transient. Wave three was an explicit
  extra continuation beyond the protocol's original single expansion wave.
  Trex was occupied by an unrelated job, so two Trex-planned wave-two
  surfaces ran on main with recorded target failover; scientific configs and
  cohorts were unchanged.
- Raw results:
  [`wave two`](../results/perfectdiode-conv2-bounded-rho-continuation-wave2-seed0-v1)
  and
  [`wave three`](../results/perfectdiode-conv2-bounded-rho-continuation-wave3-seed0-v1).
- Analysis:
  [machine-readable summary](../results/perfectdiode-conv2-bounded-rho-continuation-wave3-seed0-v1/analysis/weight_distribution/summary.json),
  [layer table](../results/perfectdiode-conv2-bounded-rho-continuation-wave3-seed0-v1/analysis/weight_distribution/layer_statistics.csv),
  [histograms](../results/perfectdiode-conv2-bounded-rho-continuation-wave3-seed0-v1/analysis/weight_distribution/selected_weight_histograms.png),
  and
  [diagnostic comparison](../results/perfectdiode-conv2-bounded-rho-continuation-wave3-seed0-v1/analysis/weight_distribution/weight_diagnostics.png).
- Provenance: continuation implementation commits `ca2c44f5`, `c9bd9da6`,
  and `a8c5b00f`; wave-three canonical summary reports
  `official_test_read=false`.

## `conv3_pd_unbounded_rho_t8k8_20260729T125734Z` — Conv3 perfect-diode rho selection

- Analyzed: 2026-07-29
- Completed: 2026-07-29 19:48 CEST
- Evidence class: `ordinary_mnist_selection`
- Outcome: `positive`
- Scientific question: Which `(rho_conv, rho_dense)` pairs minimize the
  three-epoch validation loss for Conv3 perfect-diode networks with `T=K=8`
  and nonnegative weights without an upper bound, for SGD and Adam across the
  baseline, ours, and legacy amplification schemes? Are the initial rho bounds
  sufficient?
- Runs and seeds: Seed 0; six optimizer/scheme surfaces; all 91 semantically
  complete cells included (54 initial cells and 37 boundary-expansion cells).
  All six final search bounds are closed, the retained cells share one cohort
  contract, and the final audit found zero semantic issues.
  Stability probes and canaries were operational gates and were excluded from
  the scientific comparison.
- Setup: Ordinary MNIST 55,000/5,000 deterministic train/validation split;
  official test split not read; input gain 360; Conv3 channels
  `[64, 128, 256]`; kernels 3; strides `[2, 2, 1]`; padding 1; output width
  20; Kaiming-uniform initialization; asynchronous minimizer; batch size 16;
  validation batch size 64; three epochs per cell. The schemes were baseline
  `(voltage_amp=1, current_amp=1)`, ours `(4, 1)`, and legacy `(4, 0.25)`.
  Weights used `weight_min=0` and `weight_max=null`. Runs used one
  `v100-16g` GPU each on Jean Zay under `fmu@v100`.
- Headline measurements: Selection used minimum final validation loss. Every
  selected point is interior to the final explored bounds. Accuracy is shown
  only as an ordinary-MNIST selection diagnostic.

  | Scheme | Optimizer | `rho_conv` | `rho_dense` | Validation loss | Validation accuracy |
  |---|---|---:|---:|---:|---:|
  | baseline | SGD | 0.001 | 0.03 | 0.09482917 | 94.74% |
  | baseline | Adam | 0.027 | 0.27 | 0.06915623 | 96.20% |
  | ours | SGD | 0.009 | 0.03 | 0.05100432 | 97.18% |
  | ours | Adam | 0.081 | 0.09 | 0.04339848 | 97.70% |
  | legacy | SGD | 0.003 | 0.01 | 0.04871860 | 97.44% |
  | legacy | Adam | 0.027 | 0.01 | 0.03983585 | 97.96% |

- Interpretation: Adam lowered the selected validation loss relative to SGD
  by `0.02567295` for baseline, `0.00760584` for ours, and `0.00888274`
  for legacy. Both amplified schemes improved on the baseline for both
  optimizers in this selection study. Legacy Adam produced the lowest
  three-epoch validation loss. The boundary expansions turned back around all
  six selected points, so no further rho expansion is indicated for this
  protocol.
- Limitations: This is a one-seed, three-epoch operating-point selection
  study, not final-training or paper-test evidence. Ordinary-MNIST selection
  accuracy is diagnostic only. “Unbounded” means nonnegative weights without
  an upper cap, not signed unconstrained weights. The run predates
  `docs/experiment_reporting.md`, so its raw bundle uses the reviewed
  rho-search layout rather than per-cell
  `manifest.json`/`status.json`/`result.json`; the frozen source hashes,
  semantic cell records, scheduler logs, final receipt, and 1,817-file
  checksum manifest provide the retained provenance. An initial canary failed
  because `git` was absent from the compute-node module path; the corrected
  canary passed before any scientific batch was released and does not affect
  the conclusion.
- Raw results:
  [validated local copy](../results/conv3_pd_unbounded_rho_t8k8_20260729T125734Z/remote_full/conv3_pd_unbounded_rho_t8k8_20260729T125734Z)
  and
  `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/conv3_pd_unbounded_rho_t8k8_20260729T125734Z`
  on Jean Zay.
- Analysis:
  [`results/conv3_pd_unbounded_rho_t8k8_20260729T125734Z/analysis`](../results/conv3_pd_unbounded_rho_t8k8_20260729T125734Z/analysis)
- Selected LR handoff:
  [`perfectdiode_conv3_unbounded_fixed_lr_handoff_20260729_v1.json`](../configs/conv/perfectdiode_conv3_unbounded_fixed_lr_handoff_20260729_v1.json).
  It copies the six exact named vectors from the selected cells, including
  bias rates modified by the `q90_cap` policy; rho is not recomputed
  downstream.
- Provenance: frozen source commit
  `20b495ec8f3bf4c6179480e2686bde1984b643c9`; source archive SHA-256
  `23ea58b784d189dd2e373e1e019af3b53bd6638a7d51390e654e19f1d2e9b9ac`;
  run-input archive SHA-256
  `43b8114218c566f88ab2454d82915b245ed9b8c9089254d8e2d70e0184ae62cd`;
  1,817-file manifest SHA-256
  `07bc63932bbe3df5dbc9d0013432315392fed3feca3555b16ca7cf8403120190`,
  verified locally with exit code 0.
- Detailed evidence:
  [summary](../results/conv3_pd_unbounded_rho_t8k8_20260729T125734Z/analysis/study_summary.md),
  [selected-point table](../results/conv3_pd_unbounded_rho_t8k8_20260729T125734Z/analysis/best_validation_by_surface.csv),
  [rho surfaces](../results/conv3_pd_unbounded_rho_t8k8_20260729T125734Z/analysis/rho_validation_surfaces.png),
  [optimizer comparison](../results/conv3_pd_unbounded_rho_t8k8_20260729T125734Z/analysis/best_validation_loss_by_surface.png),
  and
  [final receipt](../results/conv3_pd_unbounded_rho_t8k8_20260729T125734Z/analysis/final_receipt.json).

Existing reviewed JSON cards under `result_registry/` remain historical
evidence and need not be rewritten.

<!--
## <study-id> — <title>

- Analyzed: YYYY-MM-DD
- Evidence class:
- Outcome:
- Runs and seeds:
- Setup:
- Headline measurements:
- Interpretation:
- Limitations:
- Raw results:
- Analysis:
- Detailed evidence:
-->
