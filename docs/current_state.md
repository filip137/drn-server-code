# Current Perfect-Diode Conv State

Updated: 2026-09-24

This is a dashboard, not a second protocol. Selection rules remain in the
[active protocol index](conv_paper_hyperparameter_protocol.md). Live running
state is generated separately in
[`current_simulations.md`](current_simulations.md).

## CIFAR-10 Evidence Review, September 24

The [consolidated CIFAR evidence summary](cifar_experiment_review_20260924.md)
covers the digital references, architecture, batch-size and LR searches,
fifty-epoch continuations, amplification/KCL checks, BN and epsilon controls,
gradient/Adam diagnostics, learned gains and late-training stagnation. It
includes measured results, failed and excluded cases, conclusions, limitations,
and links to the underlying reports and curves. The cause of legacy's final
disadvantage remains unresolved in this single-seed evidence.

## Findings From 2026-08-17

### The user-selected one-decade beta is training-stable in all nine zero-bias Adam cases

The matched ordinary-MNIST qualification is complete for Conv1/Conv2/Conv3 x
baseline/ours/legacy.  It uses the exact inherited BPTT Adam learning-rate
vectors with every `Bias_*` rate fixed at zero, shared
`T/K=4/4,6/6,8/8`, centered frozen-current true-float64 EqProp, injected beta
Conv1 `100/30/3`, Conv2 `100/10/.03`, and Conv3 `100/3/.001`, seed 0,
10/30/30 epochs, zero endpoint read noise, and no official-test read.

| Architecture | Scheme | Best validation | Final validation | Final drop from best |
|---|---|---:|---:|---:|
| Conv1 | baseline | `96.18%` | `96.04%` | `.14 pp` |
| Conv1 | ours | `96.42%` | `96.34%` | `.08 pp` |
| Conv1 | legacy | `96.44%` | `96.42%` | `.02 pp` |
| Conv2 | baseline | `97.22%` | `97.22%` | `0 pp` |
| Conv2 | ours | `98.10%` | `98.02%` | `.08 pp` |
| Conv2 | legacy | `98.44%` | `98.42%` | `.02 pp` |
| Conv3 | baseline | `97.76%` | `97.68%` | `.08 pp` |
| Conv3 | ours | `98.72%` | `98.64%` | `.08 pp` |
| Conv3 | legacy | `99.08%` | `99.02%` | `.06 pp` |

All `9/9` runs complete their full epoch budget with finite metrics and finish
well inside the frozen `<5 pp` stability rule.  This is therefore positive
evidence that the one-decade tier is training-stable under this exact Adam,
zero-bias, shared-`T/K` contract.  The within-architecture final ordering is
legacy > ours > baseline, but these are one-seed selection-dataset
measurements and are not paper accuracy or yet a controlled BPTT--EqProp
comparison.

Jean Zay UMG array `1030857_[0-4]` completed all five V100 tasks with exit
`0:0` in `02:17:19`--`05:02:24`.  The authoritative local copy is
checksum-identical to the 106 MB remote tree.  All nine canonical bundles,
five pack summaries and receipts, nine run receipts, exact-zero checkpoint
biases, architecture-matched initialization, common split, matched common
ten-epoch minibatch prefix, full Conv2/Conv3 30-epoch order, zero-noise, and
zero-official-test guards pass.

This result does **not** reclassify the one-decade gradient tier: the earlier
gate remains `213/216`, with three Conv3-initialization C0 failures, while the
two-decade tier remains `216/216`.  The trained Conv3-baseline `T=8`
free-state residual caveat also remains.  Thus the evidence now separates the
decision cleanly: one decade is demonstrated training-stable and retains the
larger phase signal; two decades is the fully direct-gradient-qualified tier
but still lacks this matched long stability qualification.  No paper beta is
frozen by the present study alone.  See the [beta decision
record](beta_study.md), [analysis report](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-one-decade-ordinary-mnist-10-30-30ep-seed0-20260816-v1/analysis/report.md),
[summary table](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-one-decade-ordinary-mnist-10-30-30ep-seed0-20260816-v1/analysis/summary.csv),
[trajectories](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-one-decade-ordinary-mnist-10-30-30ep-seed0-20260816-v1/analysis/validation_accuracy_trajectories.png),
and [verification](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-one-decade-ordinary-mnist-10-30-30ep-seed0-20260816-v1/analysis/verification.json).

## Findings From 2026-08-16

### Conv2 one-decade beta remains stable at sigma `5e-4` with SGD

The matched SGD repeat keeps the same Conv2 one-decade injected betas
`100/10/.03` for baseline/ours/legacy, centered float64 EqProp, `T=K=8`,
seed-0 ordinary MNIST, initialization, split, minibatch order, and endpoint
noise seed as the Adam study, while using the historical scheme-specific SGD
learning-rate vectors. All three Jean Zay `fmu@v100` runs completed ten finite
epochs and passed their canonical, source/config, optimizer, noise-draw,
cohort, and zero-official-test-read guards.

| Scheme | SGD clean -> sigma `5e-4` final | SGD change | Adam clean -> sigma `5e-4` final | Adam change |
|---|---:|---:|---:|---:|
| baseline | `96.52 -> 96.34%` | `-.18 pp` | `96.88 -> 96.78%` | `-.10 pp` |
| ours | `97.80 -> 97.46%` | `-.34 pp` | `97.92 -> 97.44%` | `-.48 pp` |
| legacy | `97.52 -> 96.72%` | `-.80 pp` | `98.40 -> 96.90%` | `-1.50 pp` |

There is no observed sigma-`5e-4` instability under either optimizer
contract. Ours has the highest noisy final validation accuracy with both, and
legacy has the largest clean-relative penalty with both. The magnitude does
depend on the optimizer/LR contract, most visibly for legacy, but this study
does not isolate optimizer algorithm effects because SGD and Adam use their
own historical LR vectors. The Adam deltas here use the later paired boundary
study controls; the original higher-noise report's `-.22/-.42/-1.44 pp`
deltas remain the correct values against its original clean parent.

Production array `1029094_[0-2]` and canary `1029073_0` completed `0:0`; the
authoritative local collection has zero checksum differences from Jean Zay,
and the final focused suite passes `29/29`. See the [beta decision
record](beta_study.md), [comparison report](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-5em4-sgd-10ep-seed0-20260816-v1/analysis/report.md),
[table](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-5em4-sgd-10ep-seed0-20260816-v1/analysis/terminal_summary.csv),
and [trajectories](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-5em4-sgd-10ep-seed0-20260816-v1/analysis/validation_accuracy_trajectories.png).

### Paper-ready seed-0 BPTT--EqProp pairs now use one exact zero-bias contract

The active
[one-seed matching protocol](conv_paper_one_seed_bptt_eqprop_protocol.md)
defines a paper comparison unit as one BPTT and one EqProp run with the same
architecture, amplification scheme, optimizer, weight contract, and seed.
Within each pair, the named and ordered learning-rate vector must be exactly
identical: accepted weight rates are retained unchanged and every `Bias_*`
rate is `0`. Bias tensors initialize and remain exact zero.

Both algorithms also use the same accepted operating point: Conv1 `T/K=4/4`,
Conv2 `6/6`, and Conv3 `8/8`, with seed `0` and `10/30/30` epochs. The prior
EqProp beta studies used learned biases, and their Conv1/Conv2 cases used
`8/8`; they remain exploratory candidate evidence and cannot be promoted
unchanged. EqProp beta must be requalified on ordinary MNIST under the new
zero-bias/shared-`T/K` contract before the long deterministic medium-affine
paper runs are launched.

### The two-decade beta passes the zero-bias/shared-T/K Adam gradient gate; the maximum and one-decade betas do not

The new read-only qualification compares centered frozen-current EqProp with
BPTT on the accepted zero-bias Adam checkpoints at reconstructed
initialization and best validation. It covers all nine Conv1/Conv2/Conv3 x
baseline/ours/legacy contexts, four fixed 16-example ordinary-MNIST validation
batches, true-float64 dynamics, and exact shared `T/K=4/4,6/6,8/8`. Every
phase starts from the same hashed post-`T` state, BPTT and EqProp both use
exactly `K` further iterations, every bias LR/tensor is exact zero, and no
optimizer step or official-test read occurs. The gate requires every weight
layer and batch to have cosine `>=.99` and symmetric norm delta `<=.10`.

At the original maximum tier—Conv1 `1000/300/30`, Conv2 `1000/100/.3`,
Conv3 `1000/30/.01`—only `159/216` comparisons and `9/18` checkpoint
contexts pass. There are `10/16/31` failing rows in Conv1/Conv2/Conv3; the
global worst cosine is `.801608` and maximum norm delta `.533422`. The
Conv3-legacy best-checkpoint smoke illustrates the finite-beta scale error:
its first-convolution EqProp norm is `.884682x` BPTT despite cosine `.997944`,
while the output-state RMS displacement is `4.5935x` its free-state norm. The
same row one decade lower has norm ratio `.998242`. Thus the original maximum
is outside the local gradient regime and is not a paper-beta candidate.

At the one-decade injected-beta tier—Conv1 `100/30/3`, Conv2
`100/10/.03`, Conv3 `100/3/.001`—`213/216` comparisons pass. All Conv1,
Conv2, and best-checkpoint rows pass. The only failures are Conv3
initialization `ConvWeight_0`: baseline beta `100` fails one batch with worst
cosine `.987983`, while ours beta `3` fails two batches with worst cosine
`.939168`, norm ratio `1.32312`, and symmetric norm delta `.278178`.

One further decade lower—Conv1 `10/3/.3`, Conv2 `10/1/.003`, Conv3
`10/.3/1e-4`—all `216/216` comparisons pass. The global worst cosine is
`.998877` and the maximum symmetric norm delta is `.039421`; every
best-checkpoint context passes. Thus the one-decade tier is not a universal
gradient-faithful point under the paper matching contract, while the
two-decade tier is the supported Adam-first candidate.

The full unqualified launch gate remains false only because trained Conv3
baseline is under-relaxed after free `T=8`: its post-`T` residual fails in
Layer_1 and Layer_3 on all four batches, independently of beta. This is the
existing Conv3-baseline residual caveat, not an EqProp--BPTT direction
mismatch. On gradient evidence alone, the one-decade tier is not a universal
matched beta. The later user-directed one-decade training qualification is
stable in all nine cases but retains this `213/216` classification. A matched
two-decade ordinary-MNIST Adam stability qualification remains the supported
way to complete the gradient-qualified branch before freezing a paper beta;
deterministic medium-affine paper runs also require the Conv3 residual caveat
to be explicitly accepted or shared BPTT/EqProp T/K to be requalified. See the
[review](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-tk4-6-8-seed0-20260816-v1/review.md),
[maximum-beta table](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-tk4-6-8-seed0-20260816-v1/production-max-beta/case_summary.csv),
[one-decade table](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-tk4-6-8-seed0-20260816-v1/production/case_summary.csv),
and [two-decade table](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-tk4-6-8-seed0-20260816-v1/production-two-decades/case_summary.csv).

### Tripling the Conv1 legacy Conv LR is worse for SGD and neutral for Adam

The bounded-weight recollection was only partly correct.  In the zero-bias
`[1e-5,1e-4]` study, Conv1 legacy-SGD tied at `94.50%` when `rho_conv` moved
from `.003` to `.009` at fixed `rho_dense=.0033333`, so the factor-three
upper-Conv edge remained open.  Legacy-Adam was already bracketed: its
selected `(.009,.01)` point reached `95.42%`, while the factor-three neighbor
`(.027,.01)` reached `95.14%` (`-.28 pp`).  Thus the prior evidence motivated
an SGD sentinel, not a general factor-three learning-rate change.

The new wide `[0,100]`, zero-bias Conv1 legacy BPTT runs keep Dense LR fixed,
multiply only `ConvWeight_0` LR by exactly three, and run seed 0 for ten
ordinary-MNIST epochs:

| Optimizer | Conv LR, original / times three | Best validation, original / times three | Final validation, original / times three |
|---|---:|---:|---:|
| SGD | `2.05753e-4 / 6.17259e-4` | `95.92 / 95.52%` (`-.40 pp`) | `95.60 / 95.52%` (`-.08 pp`) |
| Adam | `2.88917e-4 / 8.66751e-4` | `96.44 / 96.46%` (`+.02 pp`) | `96.38 / 96.46%` (`+.08 pp`) |

SGD times three is below the historical control at all ten epochs, with mean
epochwise difference `-.548 pp`.  Adam is mixed (`4/1/5` epochs
higher/tied/lower) and its best change is only one validation example out of
5,000, so there is no meaningful improvement.  Both new bundles complete and
validate; initial states match across the two new arms, all new/historical
runs share the split and ten minibatch orders, all checkpoint arrays are
finite, Bias_0 stays exact zero, and the official test is not read.

Decision: **retain the original Conv1 legacy SGD and Adam Conv learning
rates**.  The bounded rho edge should not be transferred as a new wide-range
handoff.  This is one-seed diagnostic evidence and the historical/new runs use
different host/code-state provenance, so a future formal re-selection would
need fresh same-runtime controls.  See the
[review](../results/perfectdiode-conv1-legacy-zero-bias-conv-lr-times3-bptt-sgd-adam-10ep-seed0-20260816-v1/analysis/report.md),
[comparison table](../results/perfectdiode-conv1-legacy-zero-bias-conv-lr-times3-bptt-sgd-adam-10ep-seed0-20260816-v1/analysis/comparison.csv),
and [trajectory plot](../results/perfectdiode-conv1-legacy-zero-bias-conv-lr-times3-bptt-sgd-adam-10ep-seed0-20260816-v1/analysis/plots/validation_accuracy_by_epoch.png).

## Findings From 2026-08-15

### Original EqProp beta boundaries are optimizer-dependent; every one-decade-lower control is stable

The direct boundary study covers all 36 Conv1/Conv2/Conv3 x
baseline/ours/legacy x SGD/Adam x candidate/one-decade-lower cases. It uses
seed-0 ordinary MNIST, fixed learning-rate vectors, centered frozen-current
EqProp, true float64, `T=K=8`, `[0,100]` Kaiming initialization, ten epochs,
zero read noise, and no official-test read. Collapse was frozen as an explicit
non-finite endpoint or a final-validation drop of at least `5.00 pp` from the
run's best epoch.

| Architecture | Scheme | Candidate / control injected beta | SGD at candidate | Adam at candidate |
|---|---|---:|---|---|
| Conv1 | baseline | `1000 / 100` | finite and stable | non-finite |
| Conv1 | ours | `300 / 30` | finite and stable | non-finite |
| Conv1 | legacy | `30 / 3` | finite but flat at chance | non-finite |
| Conv2 | baseline | `1000 / 100` | finite and stable | non-finite |
| Conv2 | ours | `100 / 10` | finite and stable | non-finite |
| Conv2 | legacy | `.3 / .03` | finite and stable | finite, `13.00 pp` late collapse |
| Conv3 | baseline | `1000 / 100` | finite and stable | finite and stable |
| Conv3 | ours | `30 / 3` | non-finite | finite and stable |
| Conv3 | legacy | `.01 / .001` | finite and stable | finite and stable |

Thus `7/18` matched pairs confirm candidate-beta collapse and Conv1-legacy
SGD supplies an eighth practically unusable candidate despite missing the
within-run drop criterion. The other `10/18` candidate pairs do not confirm a
training boundary. In contrast, all `18/18` one-decade-lower controls complete
ten finite epochs and finish within `.28 pp` of their own best validation.
This confirms a clean MNIST safety margin but does not prove transfer to a
harder dataset or select the final EqProp beta. In particular, the strong
optimizer interaction prevents treating a cosine-boundary beta as a universal
scheme/depth constant. Conv3 baseline retains its `T=K=8` residual caveat.

Jean Zay UMG array `1017012_[0-17]` ran two cases per V100-32GB; all 18
allocations exited `0:0`. The authoritative 36-bundle local copy has zero
itemized differences under a checksum dry-run against Jean Zay, and all
canonical bundles, 36 run receipts, 18 pack receipts, initialization hashes,
split/order hashes, and official-test guards validate. See the
[beta-study decision record](beta_study.md),
[analysis report](../results/perfectdiode-conv123-centered-float64-eqprop-beta-collapse-boundary-vs-1decade-sgd-adam-10ep-seed0-20260815-v1/analysis/report.md),
[pair table](../results/perfectdiode-conv123-centered-float64-eqprop-beta-collapse-boundary-vs-1decade-sgd-adam-10ep-seed0-20260815-v1/analysis/pairs.csv),
[collection receipt](../results/perfectdiode-conv123-centered-float64-eqprop-beta-collapse-boundary-vs-1decade-sgd-adam-10ep-seed0-20260815-v1/analysis/collection_validation_receipt.json), and
[trajectory plot](../results/perfectdiode-conv123-centered-float64-eqprop-beta-collapse-boundary-vs-1decade-sgd-adam-10ep-seed0-20260815-v1/analysis/validation_accuracy_trajectories.png).

### Learned positive-only bias and fixed-zero bias are indistinguishable in the matched Conv1 diagnostic

The current worktree still carries the historical bias implementation: hidden
biases initialize at zero, inherit the nonnegative parameter projection, and
are clamped after every optimizer step; their linear energy term is unscaled
with depth. The proposed signed, amplification-scaled replacement remains a
separate contract on `codex/signed-scaled-bias` and requires its own bias-LR
selection.

A new matched baseline Conv1 BPTT/SGD diagnostic changes only Bias_0 LR from
`.142696` to `0`, while preserving the exact initialization, ordinary-MNIST
split, all ten minibatch orders, weight rates, seed, `T=K=4`, and ten-epoch
budget. Both arms reach `96.28%` best validation at epoch 7 and `96.00%`
final validation. Accuracy is identical in 7/10 epochs and differs by at most
`.02 pp`, well inside the predeclared `.5 pp` practical-similarity threshold.
The comparison is active: `94.85%/96.76%` of Bias_0 entries are positive at
best/final in the learned arm, while every saved control bias is exact zero.

This supports the narrow conclusion that biases do not materially affect this
retrained seed-0 Conv1 diagnostic. For paper runs using Bias LR `0`, state the
model contract explicitly as **bias-free / biases fixed at zero**. That is a
coherent way to avoid the historical positive-only and unscaled-bias confound,
but it is not evidence for a trainable-bias implementation. Do not pool those
runs with historical learned-bias or corrected signed/scaled-bias rows. The
result is ordinary-MNIST diagnostic evidence, not paper-facing accuracy or a
statistical equivalence test. See the [review](../results/perfectdiode-conv1-positive-vs-zero-bias-ordinary-mnist-seed0-20260815-v1/bias_state_review.md),
[table](../results/perfectdiode-conv1-positive-vs-zero-bias-ordinary-mnist-seed0-20260815-v1/analysis/summary.csv),
and [trajectory plot](../results/perfectdiode-conv1-positive-vs-zero-bias-ordinary-mnist-seed0-20260815-v1/analysis/accuracy_loss_trajectories.png).

## Findings From 2026-08-14

### Both tested Conv1/Conv3 beta margins are clean-stable; clean accuracy does not select between them

The completed exploratory qualification tests baseline, ours, and legacy in
Conv1 and Conv3 at one and two decades below each scheme/depth's independently
measured worst-layer cosine-`.99` injected-beta boundary. All twelve cases use
seed-0 ordinary MNIST, the exact fixed Adam vectors, centered frozen-current
EqProp, true float64, `T=K=8`, ten epochs, Kaiming initialization in `[0,100]`,
zero endpoint read noise, and no official-test read.

| Architecture | Scheme | Injected beta, one/two decades | Base beta, one/two decades | Best validation, one/two decades | Final validation, one/two decades |
|---|---|---:|---:|---:|---:|
| Conv1 | baseline | `100 / 10` | `100 / 10` | `96.24 / 96.22%` | `96.12 / 96.10%` |
| Conv1 | ours | `30 / 3` | `7.5 / .75` | `96.44 / 96.42%` | `96.34 / 96.32%` |
| Conv1 | legacy | `3 / .3` | `.1875 / .01875` | `96.64 / 96.62%` | `96.58 / 96.52%` |
| Conv3 | baseline | `100 / 10` | `100 / 10` | `97.02 / 97.06%` | `97.02 / 97.06%` |
| Conv3 | ours | `3 / .3` | `.046875 / .0046875` | `98.40 / 98.44%` | `98.40 / 98.44%` |
| Conv3 | legacy | `.001 / 1e-4` | `2.4414e-7 / 2.4414e-8` | `98.46 / 98.42%` | `98.44 / 98.42%` |

All `12/12` trajectories complete with finite ten-epoch histories. Moving one
additional decade lower changes best validation accuracy by only
`-.02/-.02/-.02 pp` for Conv1 baseline/ours/legacy and
`+.04/+.04/-.04 pp` for Conv3. The maximum absolute best-accuracy difference
is therefore `.04 pp`, too small for this single-seed clean study to
distinguish the tiers. Initialization is exactly matched within architecture,
and all cases share the train split, validation split, and ten minibatch-order
hashes.

Both margins are thus clean-qualified for a subsequent read-noise comparison;
there is no observed instability at the one-decade tier or one tier below it.
No beta has been selected automatically. The choice remains a robustness
tradeoff for review: the one-decade tier retains a tenfold larger phase signal,
whereas the two-decade tier retains the larger small-perturbation margin and
matches the rule previously studied in Conv2. Clean accuracy alone supplies no
basis for preferring either. Conv3 baseline retains its declared caveat: its
source cosine boundary at `T=K=8` is residual-unqualified because the shared
free state was under-relaxed, even though both training trajectories here are
finite and accurate.

The open selection question, the rejected original Conv2 boundary tier,
complete per-layer one-/two-decade displacement tables, and the alternatives
of maximum clean-stable beta versus matched output-layer displacement are
recorded in the dedicated [EqProp beta study](beta_study.md).

Jean Zay UMG production array `980913_[0-5]` ran two logical cases concurrently
per V100-32GB and completed all tasks with exit `0:0` in `2:11:35--2:14:58`.
The authoritative local copy matches all remote relative-path content
manifests. Canary `980540` is retained and excluded as an operational
post-run-validator failure after both GPU smokes succeeded; the tested fix and
replacement canary `980865` passed before production. See the
[analysis report](../results/perfectdiode-conv13-centered-float64-eqprop-beta-qualification-1to2decades-10ep-seed0-20260814-v1/analysis/report.md),
[summary table](../results/perfectdiode-conv13-centered-float64-eqprop-beta-qualification-1to2decades-10ep-seed0-20260814-v1/analysis/summary.csv),
[trajectory plot](../results/perfectdiode-conv13-centered-float64-eqprop-beta-qualification-1to2decades-10ep-seed0-20260814-v1/analysis/validation_accuracy_trajectories.png),
and [verification](../results/perfectdiode-conv13-centered-float64-eqprop-beta-qualification-1to2decades-10ep-seed0-20260814-v1/analysis/verification.json).

## Findings From 2026-08-13

### A four-decade absolute scale-up leaves Conv1 bounded-Adam performance unchanged

The matched exploratory repeat scales the exact shared bounded-uniform Conv1
initializer and projection interval from `[1e-5,1e-4]` to `[.1,1]`, multiplies
every nonzero raw Adam LR by `1e4`, and changes nothing else in the three
baseline/ours/legacy source cells. It preserves the 10:1 dynamic range, exact
initializer quantiles, zero biases, seed-0 ordinary-MNIST split/order, accepted
`T=K=4`, and three-epoch budget.

Final small/scaled validation accuracies are baseline `88.84/88.84%`, ours
`91.60/91.60%`, and legacy `95.42/95.40%`. The largest absolute difference
across all nine epoch accuracies is only `.04 pp`; final exact-bound occupancy
also changes by at most `.0234 pp`. All three production bundles and CUDA
smokes validate and record no official-test read.

The absolute scale-up therefore recovers no Conv1 performance and provides no
evidence that weights of order `1e-5` to `1e-4` materially hurt these Adam cells
through numerical imprecision. The bounded dynamic range, projection behavior,
or another scale-invariant contract feature remains more plausible. This is a
single-seed, three-epoch Conv1/Adam diagnostic, not an SGD, deeper-network,
long-convergence, or paper-facing result. See the
[review](../results/perfectdiode-conv1-bounded-uniform-scale1e4-adam-3ep-seed0-20260813-v1/review.md),
[summary](../results/perfectdiode-conv1-bounded-uniform-scale1e4-adam-3ep-seed0-20260813-v1/analysis/summary.csv),
and [accuracy comparison](../results/perfectdiode-conv1-bounded-uniform-scale1e4-adam-3ep-seed0-20260813-v1/analysis/accuracy_comparison.png).

### The directed bounded-uniform search locates the missing Conv2 SGD basins

The completed extension adds 257 numeric Conv1/Conv2 cells to the 114 prior
same-contract cells: shared `Uniform[1e-5,1e-4)` initialization, projection to
that interval, exact-zero bias rates, seed 0, three epochs, Conv1 `T=K=4`, and
Conv2 `T=K=6`. The 371-coordinate combined map has no numeric duplicate. All
12 directed shards and receipts are local; 257/257 bundles are complete with
zero nonfinite result.

The important correction is Conv2 SGD. Lowering both rates raises baseline
from `49.34%` to a bracketed `82.84%` at
`(rho_conv,rho_dense)=(3.70370e-5,4.11523e-5)` and raises ours from `69.50%`
to a bracketed `85.72%` at `(3.33333e-4,1.23457e-4)`. Final exact clipping at
those maxima is `20.28%` and `8.77%`. The earlier rates were therefore too
high for Conv2 SGD under this fixed weight range. Ours-Adam improves only
`+.32 pp` to a bracketed `87.70%`; baseline-Adam improves only `+.18 pp` to
`84.54%` at the maximum tested Conv rho and remains formally open with
`46.83%` clipping. Legacy remains strongest at `95.42%` with Adam in both
architectures.

Ten of twelve combined surfaces are bracketed. Conv1 legacy-SGD retains a
tied upper-Conv plateau, and Conv2 baseline-Adam remains open upward. Clipping
still does not identify LR quality on its own: combined within-surface Pearson
coefficients span `-.744` to `+.449` and Spearman coefficients span `-.645`
to `+.623`. This is ordinary-MNIST exploratory evidence with safety and
post-training T/K rejection disabled, not an LR handoff. See the
[review](../results/perfectdiode-conv12-directed-exploratory-lr-search-seed0-20260812-v1/review.md),
[combined table](../results/perfectdiode-conv12-directed-exploratory-lr-search-seed0-20260812-v1/analysis/combined/combined_surface_summary.csv),
and [accuracy/clipping plot](../results/perfectdiode-conv12-directed-exploratory-lr-search-seed0-20260812-v1/analysis/combined/accuracy_vs_final_clipping_combined.png).

### Conv2 ceiling relaxation favors Adam for baseline and ours, but not legacy

The downstream matched 24-run Conv2 sweep uses the optimizer-specific rates
above, one exact shared `Uniform[1e-5,1e-4)` checkpoint, exact-zero biases,
seed 0, `T=K=6`, and ten ordinary-MNIST epochs. Within each
scheme/optimizer surface, only the training ceiling changes across
`wmax={1e-4,2e-4,5e-4,1e-3}`. Best validation accuracies are:

| `wmax` | Baseline SGD / Adam | Ours SGD / Adam | Legacy SGD / Adam |
|---:|---:|---:|---:|
| `1e-4` | `84.22 / 85.60%` | `86.78 / 88.96%` | `94.56 / 96.44%` |
| `2e-4` | `87.40 / 87.84%` | `91.40 / 93.08%` | `96.84 / 97.44%` |
| `5e-4` | `87.78 / 90.52%` | `93.68 / 96.18%` | `97.46 / 97.74%` |
| `1e-3` | `87.88 / 91.14%` | `94.20 / 96.74%` | `97.44 / 97.78%` |

Adam is better in all 12 matched optimizer comparisons, but the interaction
is scheme-dependent. At `wmax=1e-3`, Adam exceeds SGD by `3.26 pp` for
baseline and `2.54 pp` for ours, versus only `.34 pp` for legacy. The closest
zero-bias wide Conv2 controls have much smaller Adam-minus-SGD gaps: using
their first ten epochs gives `.20/.14/.68 pp` for baseline/ours/legacy, and
their full 30-epoch best checkpoints give `.14/.12/.50 pp`. Thus the large
optimizer separation under the relaxed bounded contract is specific to
baseline and ours; legacy is nearly optimizer-insensitive there.
Correspondingly, the legacy-minus-ours gap at `wmax=1e-3` is `3.24 pp` with
SGD but only `1.04 pp` with Adam.

All 24 bundles validate and share the initialization, split, minibatch order,
and epoch budget. This remains one-seed exploratory evidence. The rates were
selected under the tight `wmax=1e-4` contract and transferred unchanged to
larger ceilings, so the growing Adam advantage can also reflect poorer SGD-LR
transfer rather than an intrinsic optimizer effect. The wide comparison is
cross-contract because its initializer and LR vectors differ. See the
[analysis report](../results/perfectdiode-conv2-fixed-uniform-init-wmax-sweep-sgd-adam-10ep-seed0-20260813-v1/analysis/report.md),
[full table](../results/perfectdiode-conv2-fixed-uniform-init-wmax-sweep-sgd-adam-10ep-seed0-20260813-v1/analysis/summary.csv),
and [curated manifest entry](experimental_manifest.md#perfectdiode-conv2-fixed-uniform-init-wmax-sweep-sgd-adam-10ep-seed0-20260813-v1--conv2-upper-weight-limit-sensitivity).

### A common relative beta rule exposes Conv2 legacy read-noise sensitivity without a clean-control collapse

The completed exploratory Conv2 training study compares two beta rules derived
from the independent worst-layer cosine-`.99` boundaries rather than from
training accuracy. One decade below the boundary uses injected beta
`100/10/.03`; two decades below uses `10/1/.003` for
baseline/ours/legacy. Each tier covers clean, endpoint read-noise sigma
`1e-5`, and sigma `1e-4`, with ten epochs, current selected Adam LR vectors,
true float64 centered EqProp, `T=K=8`, and matched initialization, data order,
and noise draws.

This lower-beta study replaced the original boundary-tier training at injected
beta `1000/100/.3`. At that tier, all eight baseline/ours cases became
non-finite, including both clean controls; legacy remained finite but its
clean validation accuracy collapsed from `97.82%` best to `89.34%` final.
Those original beta candidates were therefore rejected for Conv2 training.

Here sigma is the standard deviation of zero-mean Gaussian noise added
independently to every non-input equilibrium endpoint voltage read used by the
local EqProp gradient: `V_read = V_equilibrium + Normal(0,sigma^2)`. The two
centered phases, layers, elements, minibatches, and updates receive independent
draws. Noise changes only the copied endpoint readings after equilibrium; it
does not perturb relaxation, input voltages, or validation. Sigma is in the
simulator's voltage units and is not yet a hardware-calibrated voltage or ADC
specification.

Final epoch-10 validation accuracies are:

| Beta margin and injected baseline/ours/legacy beta | Read-noise sigma | Baseline | Ours | Legacy |
|---|---:|---:|---:|---:|
| one decade: `100/10/.03` | `0` | `97.00%` | `97.86%` | `98.34%` |
| one decade: `100/10/.03` | `1e-5` | `96.96%` | `97.88%` | `98.16%` |
| one decade: `100/10/.03` | `1e-4` | `96.86%` | `97.80%` | `97.52%` |
| one decade: `100/10/.03` | `3e-4` | `96.76%` | `97.60%` | `97.14%` |
| one decade: `100/10/.03` | `5e-4` | `96.78%` | `97.44%` | `96.90%` |
| two decades: `10/1/.003` | `0` | `96.96%` | `97.84%` | `98.38%` |
| two decades: `10/1/.003` | `1e-5` | `96.90%` | `97.70%` | `97.44%` |
| two decades: `10/1/.003` | `1e-4` | `96.84%` | `97.08%` | `96.46%` |

All `18/18` retained runs complete. The two-decade rule gives the clearer
separation. At sigma `1e-5`, final validation changes by baseline `-.06 pp`,
ours `-.14 pp`, and legacy `-.94 pp` relative to matched clean controls. At
sigma `1e-4`, the changes are `-.12/-.76/-1.92 pp`. Clean legacy leads ours
by `.54 pp`; under noise, ours leads legacy by `.26 pp` at `1e-5` and
`.62 pp` at `1e-4`. The one-decade rule produces smaller legacy excess
penalties of `.20 pp` and `.76 pp` at those noise levels.

The Jean Zay higher-noise extension adds sigma `3e-4` and `5e-4` at the
one-decade rule. All `6/6` runs remain finite and complete ten epochs, so
there is still no observed instability. At sigma `5e-4`, clean-relative final
penalties are baseline/ours/legacy `-.22/-.42/-1.44 pp`; ours retains the
highest final validation accuracy (`97.44%`) and legacy is the most
noise-sensitive. This is a single-seed descriptive extension, and the small
baseline non-monotonicity between `3e-4` and `5e-4` is not evidence of a
trend. See the [higher-noise report](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-3em4-5em4-10ep-seed0-20260813-v1/analysis/report.md),
[table](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-3em4-5em4-10ep-seed0-20260813-v1/analysis/terminal_summary.csv),
and [noise curve](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-3em4-5em4-10ep-seed0-20260813-v1/analysis/final_validation_vs_noise.png).

This study initially motivated a two-decade safety-margin candidate rather
than one shared numerical beta. The later clean Conv1/Conv3 qualification,
direct boundary-collapse study, and displacement replays leave the final beta
selection open: one decade retains more read-noise signal and is clean-stable
in every tested control, while two decades provides a larger
small-perturbation margin. Any paper choice still requires a frozen
confirmatory repeat on deterministic medium-affine MNIST. The current decision
record is the [EqProp beta study](beta_study.md). See also the historical
[review](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-beta-down-1to2decades-10ep-seed0-20260812-v1/review.md),
[penalty plot](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-beta-down-1to2decades-10ep-seed0-20260812-v1/analysis/noise_penalty_by_beta_tier.png),
and [full table](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-beta-down-1to2decades-10ep-seed0-20260812-v1/analysis/terminal_summary.csv).

## Findings From 2026-08-12

### Exploratory Conv3 ladder improves the baseline/ours frontier without selecting learning rates

The completed seed-0 grid is an **EXPLORATORY / NONCANONICAL
ordinary-MNIST diagnostic**. It compares baseline and ours with SGD and Adam
over the same ordered `8x8` `(rho_conv,rho_dense)` grid. Every cell restarts
from the same hash-verified `Uniform[1e-5,1e-4)` initializer, projects to
`[1e-5,1e-4]`, keeps all bias LRs at zero, uses `T=K=8`, and trains for three
epochs. No official-test examples were read.

| Scheme / optimizer | Highest observed `(rho_conv,rho_dense)` | Accuracy | Exact final either-bound occupancy | Range status |
|---|---:|---:|---:|---|
| baseline / SGD | `(3.7037e-5,1.3717e-5)` | `72.64%` | `60.79%` | open toward lower dense rho |
| ours / SGD | `(1.2346e-5,4.1152e-5)` | **`76.84%`** | `18.97%` | locally bracketed |
| baseline / Adam | `(1e-3,3.3333e-3)` | `76.16%` | `69.75%` | locally bracketed |
| ours / Adam | `(9e-3,3.3333e-3)` | **`80.90%`** | `38.55%` | open toward higher Conv rho |

Ours raises the observed frontier by `4.20 pp` with SGD and `4.74 pp` with
Adam. This is not only a different-maxima effect. At each baseline optimum,
the same-coordinate ours cell is better by `3.20 pp` for SGD and `3.32 pp`
for Adam while placing `42.16 pp` and `31.93 pp` fewer weights exactly at a
bound. At each ours optimum, the matched gain is `7.58 pp` and `5.36 pp`, with
`37.96 pp` and `32.45 pp` lower endpoint occupancy. Across all 64 matched
coordinates, ours is better/tied/worse in `42/13/9` SGD cells and `53/0/11`
Adam cells. Baseline and ours share the host/runtime contract within each
optimizer; SGD-versus-Adam is confounded because SGD ran on Akib and Adam on
Trex.

Final-checkpoint occupancy does not by itself identify a good LR. Accuracy
versus exact either-bound occupancy has Pearson/Spearman correlation
`+.381/+.386` for baseline-SGD, `-.098/-.171` for ours-SGD,
`+.914/+.928` for baseline-Adam, and `+.774/+.610` for ours-Adam. The Adam
surfaces therefore improve across much of this deliberately broad grid while
occupancy rises, consistent with low-LR undertraining; yet ours reaches its
higher maxima with substantially less endpoint pinning than baseline.
Descriptive and noncausal: `rho_conv` and `rho_dense` jointly vary, so the
correlations do not isolate an effect of clipping on accuracy. Exact endpoint
occupancy is also not clipping history.

The ours-Adam maximum on the highest tested Conv-rho edge supports extending
that direction in a subsequent controlled search. Baseline-Adam and ours-SGD
are locally bracketed; baseline-SGD remains open toward a still lower dense
rho. These are highest observed accuracies, not selected learning rates. This
study creates neither a canonical LR handoff nor paper-facing evidence.

All `256/256` canonical cells, four receipts, nine exact Akib/Trex evidence
roots, source hashes, plots, and analysis tables validate with zero nonfinite
cells. Jean Zay jobs `844032` and `844396` were canceled before allocation at
zero elapsed time and produced no scientific artifacts; the Akib/Trex shards
are the sole authority. See the [report](../results/perfectdiode-conv3-exploratory-lr-ladder-seed0-20260811-v1/analysis/exploratory_lr_ladder/report.md),
[surface table](../results/perfectdiode-conv3-exploratory-lr-ladder-seed0-20260811-v1/analysis/exploratory_lr_ladder/surfaces.csv),
[accuracy heatmaps](../results/perfectdiode-conv3-exploratory-lr-ladder-seed0-20260811-v1/analysis/exploratory_lr_ladder/plots/final_validation_accuracy_heatmaps.png),
and [curated manifest entry](experimental_manifest.md#perfectdiode-conv3-exploratory-lr-ladder-seed0-20260811-v1--conv3-baseline-and-ours-exploratory-lr-ladder).

### A matched T=K=8 float64 surface now covers Conv1/Conv2/Conv3 through beta 1000

The completed checkpoint-gradient diagnostic uses the same trained
best-validation baseline/ours/legacy checkpoints, fixed 16-example
ordinary-MNIST batch, centered frozen-current estimator, and common 13-point
injected-beta grid `B=.001--1000` for all three Conv architectures. All 117
production bundles across the companion baseline/ours and legacy studies
validate. Their analyses contain 351 per-layer true-float64 cosine rows
against same-T/K BPTT and 351 corresponding signed-displacement rows.

| Architecture | Scheme | max B at cosine >= .90 | max B at cosine >= .95 | max B at cosine >= .99 |
|---|---|---:|---:|---:|
| Conv1 | baseline | `>=1000` | `>=1000` | `>=1000` |
| Conv1 | ours | `>=1000` | `>=1000` | `300` |
| Conv1 | legacy | `300` | `100` | `30` |
| Conv2 | baseline | `>=1000` | `>=1000` | `>=1000` |
| Conv2 | ours | `>=1000` | `300` | `100` |
| Conv2 | legacy | `3` | `1` | `.3` |
| Conv3 | baseline | `>=1000`* | `>=1000`* | `>=1000`* |
| Conv3 | ours | `100` | `100` | `30` |
| Conv3 | legacy | `.1` | `.03` | `.01` |

These are largest sampled injected betas, not interpolated crossings;
`>=1000` is an open tested edge. Ours uses actual base beta `B/4`, `B/16`,
and `B/64` in Conv1/2/3; legacy uses `B/16`, `B/256`, and `B/4096`. The
Conv3-baseline entries marked `*` are cosine-only because its shared T8 free
state is under-relaxed (limiting float64 residual p90 `1.75151`, hard
failure); no Conv3-baseline beta is residual-qualified at T8. All other
displayed boundaries pass the float64 residual contract. See the
[baseline/ours review](../results/perfectdiode-conv123-baseline-ours-centered-float64-beta-1em3-to-1e3-tk8-seed0-20260812-v1/review.md),
[legacy review](../results/perfectdiode-conv123-legacy-centered-float64-beta-1em3-to-1e3-tk8-seed0-20260812-v1/review.md), and
[legacy cosine/displacement figure](../results/perfectdiode-conv123-legacy-centered-float64-beta-1em3-to-1e3-tk8-seed0-20260812-v1/analysis/float64_cosine_and_displacement_vs_beta.png).

### Higher beta confirms a depth-dependent centered-EqProp boundary for ours

The matched Conv1/Conv2 extension repeats the trained-Conv3 higher-beta
diagnostic at common injected `B={100,300,1000}`, `T=K=64`, one fixed
16-example ordinary-MNIST batch, and true float32/float64. Baseline uses base
beta `B`; ours uses `B/4` for Conv1 and `B/16` for Conv2. Both positive
one-sided and centered frozen-current estimators are evaluated from identical
phase starts.

Centered baseline remains all-layer faithful through beta `1000` for both
architectures. Its worst-layer float64 cosine at that point is `.990309` for
Conv1 and `.993496` for Conv2. Ours crosses the finite-perturbation boundary
earlier with depth: Conv1 passes the cosine `>=.99`, symmetric norm delta
`<=.1`, and residual gate at beta `100/300` but not `1000`; Conv2 passes only
at beta `100`. Its worst-layer centered float64 cosine is
`.998584/.994410/.979564` for Conv1 and
`.995390/.984057/.911775` for Conv2 over the three betas.

No one-sided point passes. At beta `100`, worst-layer float64 cosine is
`.914346/.943015` for baseline Conv1/Conv2 and `.758829/.263565` for ours,
then falls further with beta. Float32 and float64 nearly coincide throughout
this large-signal grid, so the failure is finite-perturbation and active-set
bias rather than small-contrast float32 cancellation. Positive output
displacement for ours is already `4.865` in Conv1 and `17.999` in Conv2 at
beta `100`, reaching `48.744/180.14` at beta `1000`.

Together with Conv3, the centered ours boundary moves downward with depth:
Conv1 is faithful through `300`, Conv2 only through `100`, and Conv3 through
`30` on the tested grids. This does not justify high-beta EqProp training;
these are BPTT-checkpoint gradient diagnostics far outside the source `.01`
current cap. See the [review](../results/perfectdiode-conv12-baseline-ours-eqprop-higher-beta-seed0-20260812-v1/review.md),
[layer table](../results/perfectdiode-conv12-baseline-ours-eqprop-higher-beta-seed0-20260812-v1/analysis/layer_metrics.csv),
and [comparison figure](../results/perfectdiode-conv12-baseline-ours-eqprop-higher-beta-seed0-20260812-v1/analysis/cosine_vs_injected_beta.png).

## Findings From 2026-08-11

### Occupancy report-only exposes a strong Conv3 legacy-SGD region, still open at lower Conv LR

The matched Conv3 repeat keeps the parent study's exact
`Uniform[1e-5,1e-4)` initializer, zero bias LRs, `T=K=8`, rho-search runner,
three-epoch budget, 90% gate, and retained projection/T/K checks. Its only
intended scientific change is that exact-bound occupancy is measured but does
not reject a run.

Legacy-SGD completes eleven numeric, epochwise-T/K-valid candidates. Three
meet 90%. At fixed `rho_dense=.0033333`, reducing `rho_conv` by three gives:

| `rho_conv` | Accuracy | Exact final either-bound occupancy |
|---:|---:|---:|
| `3.333e-4` | `92.56%` | `11.45%` |
| `1.111e-4` | **`93.62%`** | **`7.65%`** |

The best cell uses actual weight LRs
`C0=3.469e-8, C1=1.224e-5, C2=1.153e-5, Dense=9.448e-6` and zero bias LRs.
Its middle dense coordinate is bracketed by worse neighbors, but it sits on
the newly expanded lower Conv-rho edge. The surface therefore ends
`unresolved_after_boundary_expansion`, with no formal selected LR. The next
range test should extend Conv rho downward near `3.70e-5` on the same runtime;
cell `046` is a next-search center, not a published handoff.

Within legacy-SGD, accuracy versus exact final occupancy is
Pearson/Spearman `-0.695/-0.664` over eleven cells. The best row has the lowest
endpoint occupancy. This argues against more endpoint clipping being
beneficial, but is not causal and does not reproduce the removed persistent
occupancy-increase statistic.

The other Conv3 surfaces remain unresolved: baseline-SGD fails its probe;
baseline-Adam and ours-Adam have every cell rejected by the retained
projection gate; ours-SGD's six numeric endpoints remain at chance and fail
post-training T/K; legacy-Adam peaks at `88.66%`. Occupancy report-only
therefore does not create a complete Conv3 handoff.

The repeat is not a clean occupancy-only counterfactual for adaptive
legacy-SGD. Parent legacy ran on Trex RTX 5090/PyTorch 2.11; current legacy ran
on Jean Zay V100/PyTorch 2.5. Matched initialization, data order, signatures,
and code still branch differently at the retained projection threshold,
shifting the adaptive grid. A causal gate test must run both policies on one
runtime with fixed LR vectors and no re-probe. Final local validation covers
all six surfaces and receipts, 47 canonical bundles, 21 numeric endpoints,
and 84 layer-occupancy rows with no authority or integrity error. See the
[study review](../results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv3-occupancy-report-only-seed0-20260811-v1/review.md)
and [curated manifest entry](experimental_manifest.md#perfectdiode-bounded-uniform-zero-bias-lr-search-conv3-occupancy-report-only-seed0-20260811-v1--conv3-lr-search-without-occupancy-rejection).

## Findings From 2026-08-10

### Bounded-uniform zero-bias LR selection is incomplete beyond legacy

The matched seed-0 ordinary-MNIST search now has complete transport coverage
for all 18 Conv1/Conv2/Conv3 x baseline/ours/legacy x SGD/Adam surfaces. Each
architecture uses one shared `Uniform[1e-5,1e-4)` checkpoint, projection to
`[1e-5,1e-4]`, and exact-zero hidden-bias learning rates. Six surfaces meet the
predeclared 90% three-epoch handoff gate:

| Architecture | Scheme / optimizer | Selected `(rho_conv,rho_dense)` | Accuracy | Exact either-bound occupancy |
|---|---|---:|---:|---:|
| Conv1 | ours / SGD | `(0.001,0.0033333)` | `90.18%` | `6.54%` |
| Conv1 | ours / Adam | `(0.001,0.01)` | `91.60%` | `19.28%` |
| Conv1 | legacy / SGD | `(0.003,0.0033333)` | `94.50%` | `0.67%` |
| Conv1 | legacy / Adam | `(0.009,0.01)` | `95.42%` | `4.18%` |
| Conv2 | legacy / SGD | `(0.003,0.0033333)` | `93.34%` | `6.21%` |
| Conv2 | legacy / Adam | `(0.009,0.01)` | `95.42%` | `14.31%` |

Conv1 baseline peaks at `87.52/88.84%` for SGD/Adam. Conv2 baseline and ours
peak at `49.34/84.36%` and `69.50/87.38%`, respectively, so these are
observations rather than selected LRs. Conv2 ours-Adam has a weak upper-conv
edge hint at `(0.009,0.01)`, but increasing conv rho from `0.003` to `0.009`
there adds only `0.28 pp`. Conv3 produces no numeric candidates: baseline-SGD
fails its stability probe and every tested cell in the other five surfaces is
rejected by the Conv3 safety contract, including the smaller adaptive legacy
grids.

Final exact-bound occupancy is not a reliable standalone LR diagnostic.
Within-surface accuracy correlations switch sign, from Pearson/Spearman
`-0.953/-0.817` to `+0.982/+0.967`; Conv2 ours-Adam is approximately null at
`-0.081/-0.117`. The best Conv1 legacy rows have only `0.67/4.18%` pooled
clipping, whereas below-gate Conv2 baseline-Adam has `44.04%`. Endpoint
occupancy loses clipping history, update direction and magnitude, near-bound
mass, and layer weighting. It can flag saturation but cannot establish that an
LR is too high or too low. This study therefore does not publish a complete
bounded handoff, and no long confirmations were launched. See the
[final report](../results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-20260810-v1/analysis/final/report.md)
and [curated manifest entry](experimental_manifest.md#perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-20260810-v1--bounded-uniform-zero-bias-lr-selection).

### Literal amplification-scaled current EqProp has a float64 window, but not a float32 deep update

The latest read-only replay now applies the proposed scaling to the nudging
dynamics:
`beta_injected=beta_base*(voltage_amp/current_amp)^L`, with the exact output
bias-current-row exponent `L=1/2/3` for Conv1/2/3. The factors are
baseline `1/1/1`, ours `4/16/64`, and legacy `16/256/4096`. Earlier cost-mode
bundles only reported this factor and are superseded for the literal-current
question.

The runtime rule freezes `force=-dC/dy` at the common post-`T` state and
applies the linear output term `-beta_injected<y,force>`. The original replay
uses `[dtheta E(s_plus)-dtheta E(s_zero)]/beta_injected`; the complete Conv3
extension also evaluates the centered
`[dtheta E(s_plus)-dtheta E(s_minus)]/(2 beta_injected)` estimator. Biases
remain active but are excluded from scored gradients. These are frozen-current
EqProp diagnostics, not full-cost `E+beta*C` EqProp.

In float32, neither the literal common-base nor equal-injected Conv sweep
selects any of the 18 architecture/scheme/checkpoint contexts. Readouts align
first, while the earliest deep convolution remains poor or exactly zero.
Increasing to `T/K=64/64` changes finite cosines by at most `0.007988`.
FC1/2/3 controls reproduce the depth trend; only FC1 legacy selects a beta,
so convolution is not the cause.

True float64 resolves the direction. Every trained Conv2/Conv3 case has a
tested case-specific beta with all-layer cosine above `0.99`. The clearest
result is trained Conv3 legacy at base beta hat `3e-10`: base beta
`8.4139263e-9`, injected beta `3.4463442e-5`, and float64 layer cosines
`.999791/1/1/.999764`. The identical float32 replay gives
`zero/.006617/.561176/.999764`. Its matched-zero state displacement is
`5.19e-12/8.17e-9/1.35e-5/.01793` from first state through output.

The completed trained-Conv3 true-dtype curve contains 14 unique baseline,
11 ours, and 7 legacy points. In float64, every layer exceeds cosine `0.99`
for baseline at base beta hat `1e-6`--`3e-4`, ours at
`1e-8`--`1e-5` (the last point is capped), and legacy at
`1e-10`--`1e-9`. In float32, no scheme has an all-layer point and `C0` is an
exact-zero EqProp gradient at every tested beta for all three schemes. The
float32 early-layer failure is therefore shared by baseline, ours, and legacy,
not unique evidence against legacy amplification. These are cosine windows;
the trained baseline shadow remains residual-limited at native `T=12`, so its
float64 window is not a separate equilibrium-convergence certification.

The user-directed trained-Conv3 extension now raises common injected beta to
`{.01,.1,1,3}` for baseline and ours, extends baseline alone through
`{10,30,100}`, and raises the replay to `64/64` plus a `128/128` confirmation.
There is still no float32 all-layer window. At injected beta `100`, baseline
float32 `C0/C1/C2/D` cosines are `.245752/.773328/.999596/.992606`; C0 has
norm ratio `4.67` and `83.5%` exactly-zero contrast entries, while the float64
Dense cosine has already fallen to `.992606`. Ours reaches finite-beta bias
much sooner: at beta `3`, its output displacement is `.591827` and its
float64 Dense cosine is `.674475`, while float32 C0/C1 are only
`.086006/.157173`.

At injected beta `.1`, `64/64` and `128/128` are numerically identical.
Longer relaxation repairs baseline's float64 projected-KKT residual from a
native worst p90 near `.07346` to about `2.7e-11`, but does not repair its
float32 direction; the float32 residual stalls near `.01099`. Ours passes the
float32 residual gates and still has bad C0/C1. The failure is therefore not
an iteration-budget effect. Its matched-free/nudged float64 displacement at
beta `.1` is `4.21e-10/6.11e-8/3.13e-5/.01973` from H0 through output,
versus `1.94e-10/2.01e-9/5.40e-7/1.04e-4` for baseline.

The matched one- versus two-sided extension tests the next boundary on one
Trex surface: injected beta baseline `{100,300,1000}` and ours `{3,5,10}` at
`T=K=64`.  Centered `(G_plus-G_minus)/(2 beta_injected)` is much better than
one-sided differencing, especially at the readout, but it still does not yield
an all-layer float32 estimator.  Centered baseline float32 C0 rises only
`.314963 -> .644491 -> .915589`; centered ours C0 is
`.024906/.066527/.046875` and C1 reaches only `.494860`.  In float64, centered
worst-layer cosine remains `.997707` for baseline beta `1000` and `.998990`
for ours beta `10`.

Those float64 cosines occur outside a small-nudge regime.  At the largest
betas, matched-zero output displacement is `1.038` baseline and `1.973` ours;
one-sided Dense cosine has fallen to `.709103/.194545`, while centered Dense
remains essentially one by canceling even-order finite-beta error.  Negative
and positive displacement magnitudes are nearly symmetric, and every matched
positive endpoint hash is identical across estimator runs.  Ours passes every
residual gate; baseline float32 is already slightly residual-limited at the
common free/zero Layer 2 state, with no signed-phase residual blow-up.  Thus
neither a still larger beta nor two-sided subtraction repairs the float32
early-layer transport problem.  See the
[one-vs-two review](../results/perfectdiode-conv3-baseline-ours-eqprop-one-vs-two-sided-higher-beta-seed0-20260811-v1/review.md) and
[combined figure](../results/perfectdiode-conv3-baseline-ours-eqprop-one-vs-two-sided-higher-beta-seed0-20260811-v1/analysis/conv3_eqprop_one_vs_two_sided_higher_beta.png).

The complete Conv3 matrix now closes the common injected-current grid for all
three schemes and both checkpoint roles. It covers
`B={.001,.003,.01,.03,.1,.3,1,3,10,30,100}`, `T=K=64`, native
float32/float64, and both positive one-sided and centered estimators. Base beta
is scaled as `B`, `B/64`, and `B/4096` for baseline, ours, and legacy.
Centered float64 has a common all-layer cosine-and-norm window at
`B={.001,.003,.01}` across initialization and best weights. At `B=.001`, the
worst layer cosine is `.999999/.999999` for baseline init/best,
`.999998/.9999998` for ours, and `.999638/.999906` for legacy.

That common centered window is directional, not perturbatively small for
legacy. At `B=.001`, legacy H0/H1/H2/output relative displacement is
`9.45e-9/2.51e-5/.00469/1.34555` at initialization and
`1.50e-10/2.37e-7/.000391/.520166` at best; output delta RMS is
`.738209/.187204`. One-sided float64 therefore has no legacy all-layer point
on this grid: Dense limits its best maximin cosine to `.877393/.847661`.
Centered subtraction cancels that leading finite-beta error.

Float32 has no all-layer point for either estimator or any scheme. For the
centered estimator, the best worst-layer cosine over the full grid is only
`.164965/.335705` baseline, `.596808/.355025` ours, and
`.161268/.149566` legacy at initialization/best. All 33 undefined cosines are
exact-zero float32 `C0` updates. All float64 residual gates pass; only trained
baseline float32 Layer 2 is slightly above the strict residual gate
(`p90=.01025--.01147`, maximum `.01660`), including at the shared free/zero
state, while ours and legacy pass. The completed grid therefore strengthens
the conclusion that float32 early-layer failure is shared numerical contrast
loss, not a legacy-only mechanism. See the
[complete review](../results/perfectdiode-conv3-eqprop-beta-1em3-to-1e2-init-best-true-dtype-seed0-20260811-v1/review.md),
[full layer table](../results/perfectdiode-conv3-eqprop-beta-1em3-to-1e2-init-best-true-dtype-seed0-20260811-v1/analysis/summary.csv),
[float32 curves](../results/perfectdiode-conv3-eqprop-beta-1em3-to-1e2-init-best-true-dtype-seed0-20260811-v1/analysis/plots/cosine_vs_injected_beta_float32.png),
and [float64 curves](../results/perfectdiode-conv3-eqprop-beta-1em3-to-1e2-init-best-true-dtype-seed0-20260811-v1/analysis/plots/cosine_vs_injected_beta_float64.png).

The dedicated ours-only tail now closes the unmeasured region above injected
beta `10`: `B={30,100,300,1000}`, base beta `B/64`, both one-sided and
centered estimators, true float32/float64, and `T=K=64`.  It finds no missing
or doubled beta factor.  The current scale and finite-difference denominator
both use the guarded factor `64`; zero and positive endpoint hashes match
exactly between estimators; displacement is proportional to `B` through the
linear-response portion; and every ours residual gate passes.

The apparently small motion is confined to the early layers.  Positive
float64 relative displacement at `B=30/100/300/1000` is only
`1.26e-7/4.15e-7/2.50e-6/1.37e-5` in H0, but already
`5.92/19.75/59.44/198.62` at the output.  Absolute RMS confirms real
output-to-H0 attenuation, not merely a cross-layer normalization artifact.
Centered float64 remains all-layer cosine `>=.99` at `B=30`, then loses that
property by `100` as signed diode active sets diverge.  Centered float32 never
has an all-layer point: its best worst-layer cosine is only `.481631` at
`B=300`, after later layers have become finite-beta biased.  Thus larger beta
trades float32 subtraction zeros for nonlinear bias before exposing a usable
deep update; it does not rescue ours or reveal an injection bug.  See the
[beta-tail review](../results/perfectdiode-conv3-ours-eqprop-beta-above10-bug-audit-seed0-20260811-v1/review.md) and
[cosine/displacement figure](../results/perfectdiode-conv3-ours-eqprop-beta-above10-bug-audit-seed0-20260811-v1/analysis/conv3_ours_eqprop_beta_above10.png).

Larger legacy nudges leave the readout's linear-response regime without
blowing up the residual. At injected beta `.003446`, output displacement is
`1.793` and Dense cosine `.426627`; at `.01`, they are `5.202` and `.167644`.
All float64 residual gates still pass. A small residual therefore proves
equilibration, not a perturbatively small nudge. No tested common base beta is
good for every scheme: baseline needs a larger signal while the `4096x`
legacy factor overnudges its readout.

Supported handoff: a positive one-sided float64 frozen-current legacy Conv3
pilot should still start at base beta hat `3e-10`. A centered float64 pilot can
use common injected `B=.001` (legacy base beta `B/4096`), but it must be labeled
as a symmetric finite-difference point with order-one legacy output motion,
not a small-nudge point. Do not launch naive float32 deep training. Select beta
using injected current and layerwise displacement, and treat full-cost EqProp
as a separate learning-rule experiment. See the
[review](../results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/review.md),
[Conv beta curves](../results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/conv-current-commonbase/artifacts/plots/),
[true-dtype heatmap](../results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/analysis/eqprop_current_commonbase_true_dtype_cosine.png),
[trained Conv3 true-dtype curve](../results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/analysis/conv3_current_true_dtype_beta_curve.png),
[exact Conv3 table](../results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/analysis/conv3_current_true_dtype_beta_curve.md),
[high-beta/high-T/K review](../results/perfectdiode-conv3-baseline-ours-eqprop-high-beta-high-tk-seed0-20260811-v1/review.md),
[high-beta layerwise figure](../results/perfectdiode-conv3-baseline-ours-eqprop-high-beta-high-tk-seed0-20260811-v1/analysis/conv3_eqprop_high_beta_high_tk.png),
[T/K table](../results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/conv-current-commonbase-tk/fixed_beta_tk_summary.csv),
and [FC curves](../results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/fc-current-commonbase/artifacts/plots/).

### Across schemes, centered EqProp matches BPTT only in Conv1

A read-only replay compared centered cost-nudged EqProp with BPTT at
hash-verified reconstructed initialization and maximum-validation weights for
baseline `voltage/current=1/1`, ours `4/1`, and legacy `4/0.25`. All 18 cases
used the same fixed 64-example ordinary-MNIST cohort and
`beta in {0.003,0.01,0.03,0.1,0.25,0.5,1.0}`. Biases remained active in the
dynamics but were excluded from scoring; no optimizer step or official-test
read occurred.

At each source-native T/K, the beta maximizing the worst weight-layer cosine
gives:

| Architecture | Scheme | Init: beta / worst cosine | Best: beta / worst cosine |
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

Conv1 is the only architecture with strong all-layer direction for every
scheme. In Conv2/Conv3, later and readout layers can align while the early
convolutions are orthogonal, reversed, or dead. Legacy's excellent Conv1 match
exists only in a narrow beta window and its BPTT training advantage does not
translate into deep EqProp gradient fidelity. Conv3 baseline uses native
`T/K=12/8`; at the scheme-shared `8/8` trained anchor its worst cosine is
`-0.012847`, rather than the native `0.039319`.

The independent T x K grid resolves the earlier phase-length ambiguity. With
beta frozen at the shared-anchor selection, K produces the larger cosine range
in 41/50 complete layer grids and a range above `0.01` in 17/50. T has only one
material effect: trained Conv3-baseline C0. Retuning beta improves trained
Conv3 ours from `0.034167` at `8/8` to `0.112146` at `8/16`, but no deep scheme
reaches a useful all-layer match. More K can also destabilize the negative
phase; all 152 unstable production phase records are negative.

Matched-zero-K state displacement shows why the global state norm is
insufficient. The largest displacement is always in the final/output state,
which can move by `9.35x` for Conv2-ours initialization and `15.7x` for
Conv2-legacy initialization even when the concatenated global ratios are only
`0.0357` and `0.804`. Baseline phases are nearly symmetric; ours and legacy
develop strong negative-phase asymmetry with depth. Displacement magnitude is
not monotone with cosine: trained Conv3 baseline moves its matched output state
by only `3.94e-4` yet has a worst cosine of `0.039319`.

The supported conclusion is unchanged but now applies across amplification
schemes: do not launch deep EqProp training under the present centered
whole-output-nudge contract. The next diagnostic should intervene on backward
transport using staged/layerwise nudging or an explicit feedback/residual path,
and gate every weight layer jointly on finite dynamics, non-dead gradient,
direction, scale, and state displacement. See the
[reviewed report](../results/perfectdiode-conv123-all-amplification-adam-eqprop-bptt-beta-tk-displacement-seed0-20260810-v1/review.md),
[layer/beta table](../results/perfectdiode-conv123-all-amplification-adam-eqprop-bptt-beta-tk-displacement-seed0-20260810-v1/analysis/parameter_summary.csv),
[T/K table](../results/perfectdiode-conv123-all-amplification-adam-eqprop-bptt-beta-tk-displacement-seed0-20260810-v1/analysis/fixed_beta_tk_summary.csv),
and [state-displacement table](../results/perfectdiode-conv123-all-amplification-adam-eqprop-bptt-beta-tk-displacement-seed0-20260810-v1/analysis/state_displacement_summary.csv).

## Findings From 2026-08-09

### Learning-rate rescaling and extra time do not close the deeper zero-bias baseline gap

All 15 requested seed-0 ordinary-MNIST baseline-SGD counterfactuals completed
and validate locally, including every epoch checkpoint, five traced transitions
per epoch, exact-zero saved bias tensors, and zero official-test reads. The new
current-LR controls reproduce the prior metric curves exactly over every
matched epoch.

| Architecture | Current, 30 epochs | Best other 30-epoch LR vector | Conv-LR/3, 50 epochs | Prior ours SGD | Prior legacy SGD |
|---|---:|---:|---:|---:|---:|
| Conv1 | `96.60%` | **`96.66%`** | `96.64%` | `96.50%` (10 ep) | `95.92%` (10 ep) |
| Conv2 | `97.22%` | `97.08%` | **`97.30%`** | `97.98%` | `97.96%` |
| Conv3 | **`97.04%`** | `96.92%` | `96.78%` | `98.20%` | `98.76%` |

Tripling Conv rates reduces Conv2/Conv3 best accuracy by `0.40/0.48` points,
and matching ours' or legacy's initial relative-proposal vectors does not help.
For Conv-LR/3, epochs 31--50 add only `0.02/0.14/0.34` points for
Conv1/Conv2/Conv3. The 50-epoch Conv2 baseline still trails the 30-epoch
ours/legacy controls by `0.68/0.66` points; Conv3 remains farther behind.
Conv1 is effectively insensitive across the tested vectors and its amplified
controls are not duration-matched. This is diagnostic evidence that ordinary
under-training is not the explanation for the deeper gap. See the
[combined review](../results/perfectdiode-conv123-zero-bias-baseline-conv-lr-div3-long50-ordinary-mnist-seed0-20260808-v1/analysis/report.md).

### Aggregate performance follows `legacy > ours > baseline`

The completed 72-run signed/scaled-bias mechanism study rejects a
flat-parameter-minimum explanation for legacy amplification. Legacy is much
sharper in parameter space but makes the physical equilibrium problem less
stiff, suppresses runaway weight and bias growth, and leaves substantially
fewer bounded conductances pinned while producing sparse, stable hidden diode
states. The evidence supports depth-dependent structural regularization and
preconditioning, with the strongest benefit in deeper bounded networks. See
the [legacy amplification mechanism study](legacy_amplification_mechanism_study_20260807.md).

Together with the zero-bias and learning-rate counterfactuals, the aggregate
performance ordering for the tested family is now `legacy > ours > baseline`,
with the separation strongest in deeper and bounded networks. This is not the
ordering of every individual shallow or optimizer-matched cell, but the overall
pattern is established. The leading interpretation is that the networks are
easier to train as their effective dynamics become more feed-forward-dominant.
That interpretation still needs to be quantified and tested causally; flat
parameter minima, larger effective learned weights, ordinary under-training,
and a requirement for trainable biases are not sufficient explanations.

### Relation to feedback-regulated equilibrium propagation

Liu and Chen's ICLR 2026 paper,
[*Toward Practical Equilibrium Propagation*](https://iclr.cc/virtual/2026/poster/10008357),
provides useful evidence that attenuating feedback can make equilibrium-style
recurrent networks converge faster and improve accuracy up to a
depth-dependent optimum. Feedback that is too weak loses deep learning signal;
their residual connections compensate for that failure. Its relevance here is
suggestive rather than direct: it uses distinct forward and backward
connections in general vector-field dynamics with a local, STDP-compatible
learning rule, rather than deriving both directions from one reciprocal energy.
Settling speed is secondary in the present study. The transferable hypothesis
is that weakening the effective recurrent return path while preserving forward
and deep update transmission improves accuracy. This should be tested through
the amplification-by-bound interaction, not assumed from equilibrium speed.

## Findings From 2026-08-06

### Legacy can retrain without biases

The completed Conv2/Conv3 legacy-SGD medium-affine controls distinguish
post-training checkpoint dependence from training necessity. Both models were
trained from initialization with every original weight learning rate unchanged
and every hidden bias fixed exactly at zero:

| Architecture | Original learned-bias validation | Zero-bias validation | Difference | Original / zero-bias validation loss |
|---|---:|---:|---:|---:|
| Conv2 legacy SGD | `91.42%` | `91.58%` | `+0.16 pp` | `0.165221 / 0.164916` |
| Conv3 legacy SGD | `96.70%` | `96.82%` | `+0.12 pp` | `0.082174 / 0.081811` |

The differences are only eight and six predictions out of 5,000 and should be
read as no observed zero-bias deficit, not as an accuracy improvement. Legacy's
weights can compensate when biases are absent from the start. The historical
unscaled-bias contract remains a confound, but the large accuracy losses from
post-hoc bias removal do not show that biases caused legacy's advantage. See
the [reviewed matched control](../results/perfectdiode-conv23-legacy-zero-bias-medium-affine-seed0-20260731-v1/analysis/report.md).

### All schemes train strongly without biases on ordinary MNIST

All 18 Conv1/Conv2/Conv3 x scheme x optimizer seed-0 ordinary-MNIST controls
completed with exact-zero bias rates and tensors and with the accepted weight
rates unchanged. Best validation accuracy spans `95.92%`--`99.00%`. Broken
out by optimizer, the complete matrix is:

| Architecture | Epochs | Optimizer | Baseline | Ours | Legacy |
|---|---:|---|---:|---:|---:|
| Conv1 | 10 | SGD | `96.28%` | **`96.50%`** | `95.92%` |
| Conv1 | 10 | Adam | `96.16%` | **`96.44%`** | **`96.44%`** |
| Conv2 | 30 | SGD | `97.22%` | **`97.98%`** | `97.96%` |
| Conv2 | 30 | Adam | `97.36%` | `98.10%` | **`98.46%`** |
| Conv3 | 30 | SGD | `97.04%` | `98.20%` | **`98.76%`** |
| Conv3 | 30 | Adam | `97.80%` | `98.66%` | **`99.00%`** |

This establishes that trainable biases are not required for strong learning at
the accepted weight rates. The official test was disabled, so these values are
optimization diagnostics rather than paper evidence. See the
[18-arm review](../results/perfectdiode-conv123-zero-bias-ordinary-mnist-seed0-20260805-v1/analysis/report.md).

### The zero-bias depth ordering is a conditioning effect, not extra capacity

The complete curves show that ours exceeds baseline at all `140/140` matched
epochs. Legacy exceeds ours throughout Conv3 and Conv2 Adam, but not Conv1 or
Conv2 SGD, so `legacy > ours > baseline` is a depth-dependent pattern rather
than a universal ordering.

For zero biases, the implemented energy has an exact coordinate transform. If
`g=voltage_amp/current_amp`, then normalized states turn edge weights into
`W'_l=g^l W_l`. Conv3's four effective edge multipliers are `[1,1,1,1]`,
`[1,4,16,64]`, and `[1,16,256,4096]` for baseline, ours, and legacy. The
schemes therefore have the same classification function family, but the shared
raw initialization represents progressively stiffer deep effective models.
Their normalized outputs are also multiplied by `1`, `4^d`, and `4^(2d+1)`
respectively in the raw coordinate system. Paired argmax is invariant to that
positive scale; the fixed-one-hot squared loss and its gradients are not.

A guarded 128-example best/final/init replay found nearly identical
scale-invariant class separation and clamp occupancy at reconstructed
initialization. By the best Conv3 checkpoints, deepest-layer between/within
class variance orders baseline below ours below legacy for both optimizers.
For Conv3 SGD, the initial largest-to-smallest layerwise relative-proposal
spread is `41.6x/5.3x/4.6x`; this imbalance is absent in the baseline Conv1
negative control. These observations support depth-wise preconditioning and
loss/LR scaling as the mechanism. See the
[mechanism report](../results/perfectdiode-conv123-zero-bias-ordinary-mnist-seed0-20260805-v1/analysis/mechanism/report.md).

### Signed-only bounded screen is suggestive but incomplete

The separate bounded-uniform Conv1/Conv2 screen used signed but still-unscaled
biases. All 55 bundles validate. Every selected checkpoint uses both bias
signs. SGD changes versus the recorded nonnegative selections stay within
`-0.06` to `+0.08` points, while three Adam rows gain `+1.06`--`+1.84`
points. Five of eight surfaces select the maximum tested bias scale `9`, so the
bias search is not closed. The study excludes legacy, Conv3, bounded Kaiming,
and amplification-scaled biases; it cannot publish a global bounded handoff or
validate the proposed corrected bias contract. See the
[reviewed signed-only report](../results/perfectdiode-conv12-bounded-uniform-signed-bias-rho-seed0-v1/analysis/report.md).

## Findings From 2026-07-31

### Historical seed-0 wide-range paper outcome (unscaled nonnegative biases)

The first deterministic medium-affine seed-0 paper batch completed its fixed
budgets for all 18 Conv1/Conv2/Conv3 x scheme x optimizer rows. At every depth,
**legacy has the highest best-checkpoint official-test accuracy when each
scheme is represented by its better of SGD and Adam**:

| Architecture | Baseline best | Ours best | Legacy best | Legacy minus ours |
|---|---:|---:|---:|---:|
| Conv1 | `69.60%` (SGD) | `72.60%` (SGD) | **`74.64%` (Adam)** | `+2.04 pp` |
| Conv2 | `85.83%` (Adam) | `92.26%` (Adam) | **`94.06%` (Adam)** | `+1.80 pp` |
| Conv3 | `91.54%` (Adam) | `96.38%` (Adam) | **`97.01%` (Adam)** | `+0.63 pp` |

This is a surprising one-seed result, not evidence that legacy wins every
optimizer-matched row: Conv1 legacy-SGD reaches only `68.15%`, while legacy
wins both matched optimizer comparisons for Conv2 and Conv3. These runs belong
to the wide/unbounded **LR-handoff family**, but their paper conductance weights
are projected to `[0,100]`; they are not literally unconstrained weights. See
the [reviewed comparison](../results/perfectdiode-paper-gradient-trace-limited-20260731-v1/analysis/report.md#sgd-versus-adam)
and [validated metric inventory](../results/perfectdiode-paper-gradient-trace-limited-20260731-v1/analysis/summary.csv).

### Bias-contract audit and corrective branch

The audit found a potential implementation/protocol mismatch in the historical
models. Their hidden-bias energy is `-<z,b>` at every depth and is not scaled
with amplification, whereas the resistive interactions carry depth-dependent
amplification factors. This means the old comparisons do not vary only the
resistive amplification convention: an unscaled linear bias can become
progressively stronger relative to the deep resistive curvature, especially
for legacy.

The proposed replacement was deliberately isolated on branch
`codex/signed-scaled-bias` in worktree
`/home/filip/server_code_conv_learning_rate_protocol/.codex/worktrees/signed-scaled-bias`
at commit `b0bd5a61260c0a13db25a88b9feedcff363973cf`. It changes two coupled
scientific factors:

- biases become signed and unclamped rather than nonnegative-projected; and
- hidden bias `B_n` is scaled by `(current_amp / voltage_amp)^n` at zero-based
  hidden depth `n`.

The resulting `[B0,B1,B2]` coefficients are `[1,1,1]` for baseline,
`[1,1/4,1/16]` for ours, and `[1,1/16,1/256]` for legacy. Because both the sign
constraint and scaling change, this branch tests a proposed new bias contract
rather than a clean scaling-only ablation, and all bias learning rates must be
reselected.

The associated study
`perfectdiode-signed-scaled-bias-conv123-seed0-20260731-v1` is prepared to
search independent bias rho values while holding accepted weight rates fixed,
over bounded and wide x Conv1/Conv2/Conv3 x baseline/ours/legacy x SGD/Adam.
As verified at `2026-07-31 22:18 CEST`, it had **not yet been submitted**: no
Slurm ID, launch receipt, or result directory existed. Separate user-authorized
current-rho/T12 long-run successors later completed on 2026-08-06; their
current results are reported below. The terminal v3 selector remains a
separate incomplete lineage. A distinct signed-only, still-unscaled
bounded-uniform Conv1/Conv2 rho study also completed on 2026-07-31; its eight
surface records are [here](../results/perfectdiode-conv12-bounded-uniform-signed-bias-rho-seed0-v1/summary.json),
but they must not be cited as evidence for amplification-scaled biases.

### Bias sensitivity in the historical checkpoints

Read-only re-equilibration on the same guarded 256-example validation cohort
shows that the learned biases are much more important to legacy than to
baseline or ours, even though the deepest legacy bias tensors are smaller in
raw RMS:

| Intervention at final SGD checkpoint | Baseline | Ours | Legacy |
|---|---:|---:|---:|
| Conv2 deepest-bias removal: deep-state change | `0.12%` | `0.96%` | **`25.41%`** |
| Conv2 all-bias removal: accuracy change | `0.00 pp` | `+0.39 pp` | **`-9.38 pp`** |
| Conv3 deepest-bias removal: deep-state change | `0.033%` | `1.00%` | **`67.01%`** |
| Conv3 all-bias removal: accuracy change | `0.00 pp` | `-0.39 pp` | **`-17.58 pp`** |

For Conv3, legacy's deepest bias is `12.3x` smaller in RMS than ours, yet its
removal causes `39.5x` more absolute deep-state displacement. This establishes
strong checkpoint dependence, not training necessity: weights may compensate
when trained from the start under a different bias contract. See the
[end-of-training bias-effect report](../results/perfectdiode-paper-gradient-trace-limited-20260731-v1/analysis/bias_implementation_audit/paper_end_bias_effect/report.md).

## Current Focus

The active question is perfect-diode BPTT training for:

- Conv1, Conv2, and Conv3;
- strides `[2]`, `[2,2]`, and `[2,2,1]`;
- kernel `3`, padding `1`, and no pooling;
- baseline `v1/c1`, proposed/ours `v4/c1`, and legacy `v4/c0.25`;
- plain SGD and Adam; and
- wide-range reference and bounded hardware weight contracts.

Rho and learning-rate selection uses ordinary MNIST. Paper training uses
deterministic medium-affine MNIST.

## Wide-Range Rho Status

The wide-range reference projects conductance weights to `[0,100]`.

### Current long-run signed/scaled-bias accuracies

The current learned-bias matrix uses signed, unclamped,
amplification-scaled biases and wide `[0,100]` conductance weights. All 18
seed-0 wide arms completed and validate locally at exact `10/30/30`-epoch
budgets for Conv1/Conv2/Conv3. The runs use the user-authorized current-rho/T12
deviation and are awaiting direct study review. Best ordinary-MNIST validation
accuracy over the full training budget is:

| Architecture | Epochs | Optimizer | Baseline | Ours | Legacy |
|---|---:|---|---:|---:|---:|
| Conv1 | 10 | SGD | `96.30%` (ep 7) | **`96.60%` (ep 9)** | `95.80%` (ep 9) |
| Conv1 | 10 | Adam | `96.42%` (ep 9) | `96.74%` (ep 9) | **`96.86%` (ep 9)** |
| Conv2 | 30 | SGD | `97.08%` (ep 29) | **`98.20%` (ep 29)** | `97.96%` (ep 29) |
| Conv2 | 30 | Adam | `97.28%` (ep 30) | `98.16%` (ep 20) | **`98.42%` (ep 29)** |
| Conv3 | 30 | SGD | `97.06%` (ep 30) | `98.28%` (ep 23) | **`98.64%` (ep 20)** |
| Conv3 | 30 | Adam | `97.76%` (ep 30) | `98.68%` (ep 28) | **`99.04%` (ep 22)** |

Every Conv2/Conv3 maximum occurs after epoch 3, at epochs `20`--`30`, which
confirms that the three-epoch selector snapshot understated the long-run
accuracy.

Ordinary MNIST uses the deterministic 5,000-example validation split and never
reads the official test split, so these are optimization diagnostics rather
than paper accuracy. The corresponding deterministic medium-affine paper rows
read the official test exactly once from the maximum-validation checkpoint:

| Architecture | Epochs | Optimizer | Baseline | Ours | Legacy |
|---|---:|---|---:|---:|---:|
| Conv1 | 10 | SGD | `70.28%` | **`74.46%`** | `68.89%` |
| Conv1 | 10 | Adam | `70.46%` | `76.55%` | **`79.85%`** |
| Conv2 | 30 | SGD | `80.73%` | `91.00%` | **`92.64%`** |
| Conv2 | 30 | Adam | `86.43%` | `92.49%` | **`94.37%`** |
| Conv3 | 30 | SGD | `86.48%` | `94.30%` | **`97.02%`** |
| Conv3 | 30 | Adam | `91.58%` | `96.29%` | **`97.95%`** |

These are the newer corrected-bias-contract measurements, not the historical
unscaled-bias paper rows above. They remain one-seed results and retain the
current-rho/T12 deviation and direct-review limitation.

The higher long-run accuracies are real. The earlier
`96.70%/97.78%/97.96%` snapshot mixed a 10-epoch Conv1 confirmation with
three-epoch Conv2/Conv3 selectors and was not the best current long-run
summary. The current long runs are not literal continuations of those old
selectors: the bias contract and bias rates changed, and Conv3 baseline uses
`T/K=12/8`, so the cross-study increase cannot be attributed to duration
alone. The separate exact-zero-bias `10/30/30`-epoch controls reach
`96.50%/98.46%/99.00%`; their full scheme-by-optimizer table is reported in
[the zero-bias finding](#all-schemes-train-strongly-without-biases-on-ordinary-mnist).
Do not pool the signed/scaled learned-bias and zero-bias cells: they use
different bias contracts.

Authority: the validated `perfectdiode-ordinary-mnist-signed-scaled-bias-current-rho-t12-conv123-seed0-20260806-v1`
and `perfectdiode-signed-scaled-bias-current-rho-t12-conv123-seed0-20260806-v1`
study summaries in the `signed-scaled-bias` worktree, plus the reviewed
aggregate [legacy amplification mechanism study](legacy_amplification_mechanism_study_20260807.md).

### Conv1/Conv2

The fixed ordinary-MNIST operating points are:

| Architecture | Input gain | Operational `T/K` |
|---|---:|---:|
| Conv1 | `40` | `4/4` |
| Conv2 | `100` | `6/6` |

All six fixed-`T/K` security rows passed comparison with `(T=64,K=64)`.
The 12 scheme x optimizer core rho surfaces completed and produced generally
strong three-epoch diagnostic candidates. Filip has now authorized all 12
explicit parameter-wise LR vectors for current unbounded result rows. Use
those vectors unchanged, at the supplied precision, from the machine-readable
handoff below.

All six Conv1 vectors completed ten-epoch confirmations with final
ordinary-MNIST validation accuracy between `95.60%` and `96.48%`; final review
is pending. Conv2 remains three-epoch best-observed evidence, with final
validation accuracy between `95.96%` and `97.78%`. Terminal rho selection
remains incomplete, and the current handoff does not imply Conv2
long-confirmation evidence.

Authority:
[`perfectdiode_learning_protocol.md`](perfectdiode_learning_protocol.md) and
[`perfectdiode_conv12_unbounded_fixed_lr_handoff_20260729_v1.json`](../configs/conv/perfectdiode_conv12_unbounded_fixed_lr_handoff_20260729_v1.json).

### Conv3

The ordinary-MNIST Conv3 contract uses:

- channels `[64,128,256]`, strides `[2,2,1]`, and padding `[1,1,1]`;
- shared diagnostic input gain `360`; and
- the user-fixed `T=K=8` operating point for the completed rho study.

The six optimizer-specific rho surfaces completed on Jean Zay: 54 initial
cells plus 37 boundary-expansion cells, with all final search bounds closed.
The selected `(rho_conv,rho_dense)` pairs are baseline SGD
`(0.001,0.03)`, baseline Adam `(0.027,0.27)`, ours SGD `(0.009,0.03)`,
ours Adam `(0.081,0.09)`, legacy SGD `(0.003,0.01)`, and legacy Adam
`(0.027,0.01)`. These are one-seed, three-epoch ordinary-MNIST selection
results. The execution used `weight_min=0` and `weight_max=null`; it is
unbounded selector evidence rather than an identical `[0,100]` selection
surface. Filip explicitly authorized applying its six published vectors to
the downstream `[0,100]` paper configs; the source mismatch and missing long
confirmation remain recorded limitations.

Authority:
[`perfectdiode_conv3_learning_protocol.md`](perfectdiode_conv3_learning_protocol.md),
the
[`perfectdiode_conv3_unbounded_fixed_lr_handoff_20260729_v1.json`](../configs/conv/perfectdiode_conv3_unbounded_fixed_lr_handoff_20260729_v1.json)
handoff, and the
[`conv3_pd_unbounded_rho_t8k8_20260729T125734Z` manifest entry](experimental_manifest.md#conv3_pd_unbounded_rho_t8k8_20260729t125734z--conv3-perfect-diode-rho-selection).

## Bounded Hardware Status

The results on training with bounded weights give a clear picture: the legacy
amplification scheme retains a large advantage over baseline and ours in the
presence of bounded weights. There are individual runs—for example Conv1
legacy at `wmax=5e-4` and `wmax=1e-3`—where bounded weights give better
validation accuracy than the closest available zero-bias wide-range control.
This probably means that the learning-rate vector for the corresponding
wide-range legacy run was suboptimal and that the run needs to be redone with
a matched initializer, data order, bias contract, and epoch budget. The
measurements, comparison limits, and exact rerun motivation are recorded in
the [bounded-weight study](bound_weights_study.md). The matched Conv2
SGD/Adam interaction is summarized in the
[ten-epoch ceiling sweep](#conv2-ceiling-relaxation-favors-adam-for-baseline-and-ours-but-not-legacy).

The bounded condition projects `ConvWeight_*` and `DenseWeight_*` to
`[1e-5,1e-4]`. It independently screens:

- `bounded_uniform` sampled from `[1e-5,1e-4)`; and
- `bounded_kaiming_uniform` with gain `1`.

Both initializers require independent ordinary-MNIST rho selection for all 18
architecture x scheme x optimizer surfaces. A single initializer is then
chosen globally by matched 2% loss-plateau wins.

The repository retains an immutable Conv1/Conv2 predecessor config, but no
completed all-depth selector or global winner is recorded. Bounded
medium-affine paper runs therefore remain pending.

The main open bounded-weight question is whether the training difficulty comes
from the absolute scale of the weight interval or from its limited dynamic
range `wmax/wmin`, and how that dependence changes with amplification. The
current `[1e-5,1e-4]` evidence does not separate those effects.

Authority:
[`perfectdiode_bounded_weight_protocol.md`](perfectdiode_bounded_weight_protocol.md).

## Paper-Run Status

The deterministic medium-affine paper grid has 36 rows per model seed:

```text
3 architectures x 3 schemes x 2 optimizers x 2 weight contracts
```

Each row must import its matching ordinary-MNIST LR handoff unchanged. The
12 user-directed Conv1/Conv2 vectors may be used for current unbounded result
rows. Ordinary-MNIST validation accuracy is never reported as paper evidence.

Current epoch decisions:

| Architecture | Paper budget |
|---|---:|
| Conv1 | 10 epochs |
| Conv2 | 30 epochs |
| Conv3 | 30 epochs |

The first wide-range batch used model seed `0`, loader seed `0`, and affine
seed `1729`. All 18 rows completed their fixed budgets. They trained on the
deterministic medium-affine 55,000/5,000 train/validation split, selected the
maximum-validation-accuracy checkpoint, and read the official 10,000-example
medium-affine test split exactly once from that checkpoint. The result is
reported above, subject to the newly identified historical bias-contract
confound and the recorded one-seed limitation. The later zero-bias legacy-SGD
controls found no Conv2/Conv3 validation deficit after retraining, but they did
not read the official test and therefore do not replace these paper rows.

Filip authorized the Conv3 vectors selected under `weight_max=null` for use
with the `[0,100]` paper contract. That source-contract mismatch remains
explicit provenance rather than being silently erased. The completed batch
used the 18 exact seed-0 wide-range configs under
`configs/conv/paper_medium_affine_perfectdiode_wide_seed0_20260729_v1/`.
Bounded medium-affine rows remain blocked on the global bounded initializer and
LR handoff.

## Compute Targets

This table records stable access and routing information, not live occupancy.
Always check the target immediately before launch.

| Resource | Launcher target | Access / workdir | Preferred work |
|---|---|---|---|
| Local foreground | `local` | current checkout | unit tests and foreground smokes |
| Local GPU tmux | `main` | session `main`; `/home/filip/server_code` | Conv1, smokes, diagnostics |
| Akib | `akib` | SSH alias `akib`; `/home/filiposana/server_code` | Conv1/Conv2 surfaces |
| Trex | `trex` | `filip@trex`; `/home/filip/server_code` | Conv2/Conv3 surfaces and long confirmations |
| Jean Zay | `jean-zay` | SSH alias `jean-zay`; Slurm `fmu@v100` | Conv3, arrays, and scheduled production |

Jean Zay source checkout:
`/lustre/fswork/projects/rech/umg/$USER/server_code`.

Jean Zay result root:
`/lustre/fsn1/projects/rech/fmu/$USER/server_code/results`.

Keep one scientific surface on one recorded target. Host availability must not
change the surface's config, batch size, initialization, cohorts, or training
order.

## Next Actions

1. Before deep EqProp training, distinguish the frozen-current rule from
   full-cost `E+beta*C` EqProp. For frozen-current legacy Conv3, validate a
   float64 pilot at base beta hat `3e-10` (injected `3.4463442e-5`) with
   per-layer direction, scale, residual, and matched-zero displacement gates.
   The same point loses the first-layer numerator in float32; extra T/K is not
   a remedy. If float32 is required, change the contrast arithmetic or signal
   transport before training.
2. Determine whether bounded-weight training difficulty is caused by the
   absolute weight scale or by `wmax/wmin`, and measure how this depends on
   amplification. Use the depth-by-bound study described in the
   [legacy amplification mechanism note](legacy_amplification_mechanism_study_20260807.md#next-study-scaling-across-depth-and-conductance-bounds).
3. Quantify feed-forward dominance, for example through physical round-trip
   gain or local state-Jacobian transfer, and test whether it tracks the
   amplification performance ordering across depths and weight bounds.
4. Analyze the saved LR-counterfactual traces and checkpoints for layerwise
   relative updates, projection activity, and late-epoch gradients. Do not
   allocate a longer Conv3 Conv-LR/3 run unless that read-only analysis exposes
   a new non-plateau mechanism.
5. Finish terminal wide-range rho review, then revisit the bounded initializer
   execution in light of the absolute-scale-versus-dynamic-range result.
6. Decide later multi-seed scope after the weight-bound question and matching
   rho handoffs are resolved.
7. Treat any remaining bias-contract comparisons as secondary follow-up. Add
   Adam or additional medium-affine zero-bias controls only if needed to test a
   specific residual scheme-by-bias interaction.

## Bias Conclusion

Bias choices can strongly affect a trained checkpoint under post-hoc removal,
but matched retraining shows no material performance deficit when trainable
biases are absent. Biases therefore do not significantly alter retrained
performance and do not explain the aggregate `legacy > ours > baseline`
ordering. Remaining bias-contract questions are lower priority than resolving
the weight-bound dependence.

The 2026-08-15 matched Conv1 baseline-SGD check strengthens this conclusion:
active positive-only learning and exact-zero bias have identical best/final
validation accuracy and differ by at most `.02 pp` at any epoch. Paper runs
with bias LR `0` should be labeled bias-free/fixed-zero; they do not validate
the separate corrected learned-bias contract.
