# EqProp Beta Study for Conv1, Conv2, and Conv3

Updated: 2026-08-18

## Decision Still Needed

We need to decide what beta to pick for the EqProp runs. It is clear that
different betas have to be used for the baseline, ours, and legacy
amplification schemes. However, it is not yet clear what these betas should
be. One idea would be to pick the largest beta that allows training for all
amplification schemes; however, this could fail for datasets more difficult
than MNIST. Another idea would be to pick betas such that the displacement of
the output layer is approximately the same for all amplification schemes.

No paper beta is frozen here. The new matched zero-bias/shared-`T/K` Adam
training qualification shows that the user-selected one-decade tier is stable
in all nine Conv1/Conv2/Conv3 x baseline/ours/legacy cases. The direct gradient
evidence still favors two decades: it passes `216/216` layer-batch rows,
whereas one decade passes `213/216` and retains three Conv3-initialization C0
failures. The decision is therefore now a measured signal-margin versus
small-perturbation-fidelity tradeoff, not an unresolved training-stability
question. At endpoint read-noise sigma `5e-4`, the same one-decade zero-bias
Adam contract remains stable in all tested Conv1 and Conv3 cases, but Conv3
performance is noise-sensitive: the final clean-relative penalty is
`-.84/-1.82/-11.38 pp` for baseline/ours/legacy. A matched two-decade
stability qualification, a matched two-decade noise comparison, and the
Conv3-baseline residual decision remain open before a paper handoff. The new
Conv2 clean/noisy extension confirms that increasing beta improves upstream
read-noise SNR but eventually destroys clean finite-nudge fidelity; no tested
common beta gives all-layer `.99` fidelity, and the optimum is layer-specific.

### Paper-readiness under the seed-0 matching rule

The new
[one-seed BPTT--EqProp protocol](conv_paper_one_seed_bptt_eqprop_protocol.md)
requires identical complete learning-rate vectors and `T/K` within every
algorithm pair, with every bias fixed at exact zero. The earlier beta-training
studies below do not meet that contract unchanged: all of their EqProp configs
use the historical nonzero bias rates, and Conv1/Conv2 use `T=K=8` instead of
the shared paper values `4/4` and `6/6`. The new one-decade qualification is
the exception: it uses the exact inherited BPTT Adam vectors with every bias
rate fixed at zero and shared `T/K=4/4,6/6,8/8`.

An exact audit found that all 18 exploratory EqProp weight-rate mappings match
their zero-bias BPTT counterparts, but none of the 18 complete LR vectors
match because every BPTT bias rate is zero. All 18 one-decade-lower EqProp
final checkpoints contain nonzero learned biases. Those training runs
therefore remain exploratory candidate evidence rather than a paper handoff.

The direct gradient component of the new protocol has now been requalified at
Conv1 `4/4`, Conv2 `6/6`, and Conv3 `8/8`, as summarized below. The matching
one-decade training-stability component is now also complete and is summarized
in the following section; it does not erase the three one-decade gradient
failures.

## Zero-Bias Shared-T/K EqProp--BPTT Gradient Gate

The formal read-only Adam-first replay uses the accepted zero-bias BPTT
checkpoints at reconstructed initialization and best validation. It covers all
nine Conv1/Conv2/Conv3 x baseline/ours/legacy cases, four fixed 16-example
ordinary-MNIST validation batches, centered frozen-current EqProp, true
float64, and exact shared `T/K=4/4,6/6,8/8`. Every negative/positive phase and
the BPTT reference starts from the byte-identical post-`T` state; every bias
LR and tensor is exact zero. The predeclared direct gate is all-layer cosine
`>=.99` and symmetric norm delta `<=.10`.

| Beta tier | Injected beta, Conv1 baseline/ours/legacy | Conv2 | Conv3 | Passing layer-batch rows | Worst cosine | Max norm delta |
|---|---:|---:|---:|---:|---:|---:|
| original maximum | `1000 / 300 / 30` | `1000 / 100 / .3` | `1000 / 30 / .01` | `159/216` | `.801608` | `.533422` |
| one decade | `100 / 30 / 3` | `100 / 10 / .03` | `100 / 3 / .001` | `213/216` | `.939168` | `.278178` |
| two decades | `10 / 3 / .3` | `10 / 1 / .003` | `10 / .3 / 1e-4` | `216/216` | `.998877` | `.039421` |

The original maximum tier fails broadly: only `9/18` checkpoint contexts pass,
with `10/16/31` failing layer-batch rows in Conv1/Conv2/Conv3. For the
Conv3-legacy best-checkpoint row that triggered the maximum-beta smoke,
`ConvWeight_0` remains directionally close to BPTT (cosine `.997944`) but its
EqProp norm is only `.884682x` BPTT, giving symmetric norm delta `.122374`.
Its output-state RMS displacement is `4.5935x` the free-state norm; one decade
lower the same gradient norm ratio is `.998242`. This is direct evidence that
the maximum finite nudge has left the local derivative regime, rather than a
float64, phase-start, or frozen-force failure.

The one-decade failures are confined to Conv3 initialization
`ConvWeight_0`: baseline beta `100` fails one of four batches (worst cosine
`.987983`) and ours beta `3` fails two (worst `.939168`, norm ratio `1.32312`).
Every Conv1 and Conv2 row and every best-checkpoint context passes at the
one-decade tier. Moving one further decade lower fixes all three failures and
keeps every best-checkpoint context inside the gate.

This changes the beta decision: clean training stability alone had not
distinguished the two tiers, but the zero-bias/shared-`T/K` gradient replay
does. On gradient fidelity alone, the one-decade tier is not a universal
matched beta and the two-decade tier is the current Adam-first nomination.
The subsequent user-directed one-decade training branch is reported below;
its positive stability result does not alter this gradient classification.

The projected-residual gate remains false only for trained Conv3 baseline:
its post-`T=8` free state fails in Layer_1 and Layer_3 on all four batches,
independently of beta. Consequently, the two-decade tier is gradient-qualified
but not an unqualified equilibrium-qualified handoff. Either retain that
known truncated-`T=8` caveat explicitly or requalify a new shared BPTT/EqProp
T/K; increasing EqProp iterations alone would violate the matching protocol.
See the [review](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-tk4-6-8-seed0-20260816-v1/review.md),
[one-decade layer table](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-tk4-6-8-seed0-20260816-v1/production/layer_metrics.csv),
and [two-decade layer table](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-tk4-6-8-seed0-20260816-v1/production-two-decades/layer_metrics.csv).

## Zero-Bias Shared-T/K One-Decade Adam Training Qualification

The user-directed follow-up tests the one-decade tier under the exact
one-seed matching contract rather than the historical learned-bias, `T=K=8`
contract. All nine cases use the inherited BPTT Adam weight rates, exact-zero
bias rates and checkpoint tensors, shared `T/K=4/4,6/6,8/8`, seed-0 ordinary
MNIST, centered frozen-current float64 EqProp, zero endpoint read noise,
10/30/30 epochs, and no official-test read. Stability was frozen as full
finite completion with final validation strictly less than `5 pp` below the
run's own best epoch.

| Architecture | Scheme | Injected / base beta | Best validation | Final validation | Final drop | Stable |
|---|---|---:|---:|---:|---:|---:|
| Conv1 | baseline | `100 / 100` | `96.18%` | `96.04%` | `.14 pp` | yes |
| Conv1 | ours | `30 / 7.5` | `96.42%` | `96.34%` | `.08 pp` | yes |
| Conv1 | legacy | `3 / .1875` | `96.44%` | `96.42%` | `.02 pp` | yes |
| Conv2 | baseline | `100 / 100` | `97.22%` | `97.22%` | `0 pp` | yes |
| Conv2 | ours | `10 / .625` | `98.10%` | `98.02%` | `.08 pp` | yes |
| Conv2 | legacy | `.03 / .0001171875` | `98.44%` | `98.42%` | `.02 pp` | yes |
| Conv3 | baseline | `100 / 100` | `97.76%` | `97.68%` | `.08 pp` | yes |
| Conv3 | ours | `3 / .046875` | `98.72%` | `98.64%` | `.08 pp` | yes |
| Conv3 | legacy | `.001 / 2.44140625e-7` | `99.08%` | `99.02%` | `.06 pp` | yes |

All `9/9` cases therefore pass the predeclared training-stability criterion.
This closes the earlier uncertainty about whether one-decade beta remains
trainable once biases are frozen and the shared paper `T/K` values are used.
It does not close the beta-selection question: the same tier retains the
`213/216` direct gradient result, while two decades remains `216/216` but has
not yet received this matched 10/30/30-epoch stability test.

Jean Zay UMG array `1030857_[0-4]` completed all five V100 tasks with exit
`0:0`. Four packs ran two cases concurrently on one GPU and the fifth ran the
Conv2-legacy singleton. The checksum-reconciled authoritative local copy
contains nine valid canonical bundles, five pack receipts, and nine run
receipts; initialization matches across schemes within architecture, all
cases share the same split and common ten-epoch minibatch prefix, Conv2/Conv3
share all 30 orders, and all bias, noise, and official-test guards pass. See
the [report](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-one-decade-ordinary-mnist-10-30-30ep-seed0-20260816-v1/analysis/report.md),
[summary table](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-one-decade-ordinary-mnist-10-30-30ep-seed0-20260816-v1/analysis/summary.csv),
[trajectories](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-one-decade-ordinary-mnist-10-30-30ep-seed0-20260816-v1/analysis/validation_accuracy_trajectories.png),
and [verification](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-one-decade-ordinary-mnist-10-30-30ep-seed0-20260816-v1/analysis/verification.json).

## Evidence Scope

The new zero-bias one-decade qualification is the only training study in this
record that already uses the complete seed-0 matching rule: identical inherited
BPTT/EqProp learning-rate vectors with every bias rate zero, shared architecture
`T/K`, 10/30/30 epochs, matched initialization/split/order, ordinary MNIST,
and no official-test read. It is training-stability and selection-dataset
validation evidence, not yet a paper-facing result or a direct BPTT
comparison. Under the ordinary-MNIST paper-dataset decision, its checkpoints
may be promoted only after beta is frozen, the BPTT pair audit passes, and the
official test split is read once under the sealed inclusion set.

The clean qualification covers Conv1 and Conv3 baseline, ours, and legacy at
one and two decades below each scheme/depth's independently measured
worst-layer cosine-`.99` injected-beta boundary. All twelve training cases use
seed-0 ordinary MNIST, the exact fixed Adam learning-rate vectors, centered
frozen-current EqProp, true float64, `T=K=8`, ten epochs, matched Kaiming
initialization in `[0,100]`, zero endpoint read noise, and no official-test
read.

The displacement measurements are read-only replays at the trained
best-validation checkpoints used to establish the beta boundaries. They use
one fixed 16-example ordinary-MNIST batch, the same centered estimator,
float64 arithmetic, and `T=K=8`. They are therefore checkpoint-and-cohort
measurements, not averages over all minibatches or all ten epochs of the clean
qualification runs.

Conv3 baseline retains the source study's caveat: its `T=K=8` cosine boundary
is residual-unqualified because the shared free state was under-relaxed. Its
two clean training trajectories nevertheless remained finite and accurate.

The Conv2 evidence directly tests the same one-/two-decade beta idea under
endpoint read noise. It uses seed-0 ordinary MNIST, matched initialization,
split, minibatch order, fixed Adam learning-rate vectors, centered
frozen-current EqProp, true float64, `T=K=8`, ten epochs, noiseless validation,
and no official-test read. This is also exploratory evidence rather than
paper-facing accuracy evidence; the same reuse and once-only test gate applies
if this noise condition enters the final paper figure.

The Conv1/Conv3 sigma-`5e-4` follow-up uses the newer exact-zero-bias,
shared-BPTT/EqProp-`T/K` contract and the clean one-decade qualification as
its byte-identified control. It therefore directly measures training
robustness for the contemplated one-seed protocol, but still uses only one
model seed and one fixed noise realization and remains non-paper-facing.

The boundary-collapse study directly trains every Conv1/Conv2/Conv3
baseline, ours, and legacy candidate at the original cosine-`.99` boundary
and one decade below it with both fixed SGD and Adam vectors. Its 36 matched
cases use the same seed-0 ordinary-MNIST, centered-float64, `T=K=8`,
ten-epoch, `[0,100]` Kaiming, zero-read-noise contract. Collapse was frozen as
either an explicit `NonFiniteTrainingError` or a final validation accuracy at
least `5.00 pp` below that run's best epoch. The study is exploratory and does
not read the official test set.

Before that lower-beta study, Conv2 was trained at the cosine-`.99` boundary
itself: injected beta `1000/100/.3` for baseline/ours/legacy. That experiment
is essential negative evidence. The clean baseline and ours controls became
non-finite, and legacy completed but suffered a large late accuracy collapse.
The boundary tier was therefore rejected and superseded by the one-/two-decade
study; it must not be mistaken for an untested or successful maximum.

## Were the Original Cosine-Boundary Betas Training-Bounded?

They are now directly tested, and the answer depends strongly on optimizer as
well as architecture. The table gives best-to-final validation for finite
ten-epoch runs and best-to-last-recorded validation before a non-finite
endpoint. `eN/bM` identifies the failing epoch and minibatch.

| Architecture | Scheme | Candidate injected beta | SGD at candidate | Adam at candidate | One-decade control, SGD/Adam final |
|---|---|---:|---|---|---|
| Conv1 | baseline | `1000` | finite `10/10`, `96.26 -> 96.04%` | **non-finite** `e2/b164`, `86.18%` last | `B=100`: `96.00 / 96.12%` |
| Conv1 | ours | `300` | finite `10/10`, `96.50 -> 96.20%` | **non-finite** `e1/b2137`, no complete epoch | `B=30`: `96.24 / 96.34%` |
| Conv1 | legacy | `30` | finite but unusable at `10.00%` throughout | **non-finite** `e3/b3047`, `95.26%` last | `B=3`: `95.62 / 96.58%` |
| Conv2 | baseline | `1000` | finite `10/10`, `96.56 -> 96.42%` | **non-finite** `e4/b3422`, `95.98%` last | `B=100`: `96.52 / 96.88%` |
| Conv2 | ours | `100` | finite `10/10`, `97.50 -> 97.28%` | **non-finite** `e4/b2493`, `96.36 -> 63.06%` | `B=10`: `97.80 / 97.92%` |
| Conv2 | legacy | `.3` | finite `10/10`, `97.56 -> 97.42%` | **late collapse**, `97.74 -> 84.74%` | `B=.03`: `97.52 / 98.40%` |
| Conv3 | baseline | `1000` | finite `10/10`, `95.76%` final | finite `10/10`, `97.02 -> 96.98%` | `B=100`: `96.14 / 97.02%` |
| Conv3 | ours | `30` | **non-finite** `e1/b75`, no complete epoch | finite `10/10`, `98.48%` final | `B=3`: `97.76 / 98.40%` |
| Conv3 | legacy | `.01` | finite `10/10`, `98.16%` final | finite `10/10`, `98.48%` final | `B=.001`: `98.20 / 98.44%` |

Under the frozen rule, `7/18` candidate-versus-control pairs confirm beta
collapse: Conv1 baseline/ours/legacy with Adam, Conv2 baseline/ours/legacy
with Adam, and Conv3 ours with SGD. Conv1 legacy with SGD is an eighth
practically unusable candidate: it remains finite and flat at chance, so it
does not satisfy the within-run `5 pp` drop rule, but its one-decade control
finishes `85.62 pp` higher. The other `10/18` pairs do not confirm collapse.

Crucially, every one-decade-lower control is finite, completes all ten epochs,
and stays within `.28 pp` of its own best validation; there is no instability
one beta decade lower for either optimizer. This provides coarse
optimizer-specific upper brackets for the seven confirmed pairs and a
practical bracket for Conv1-legacy SGD. It does not prove that every original
cosine candidate is the maximum trainable beta, nor that the one-decade tier
will transfer safely beyond MNIST. Conv3 baseline retains its predeclared
`T=K=8` residual caveat.

### How Optimizer-Dependent Is the Boundary?

Optimizer dependence is the dominant pattern, especially in Conv1 and Conv2,
but the evidence does not support calling the instability entirely optimizer
dependent. All six Conv1/Conv2 Adam candidate runs collapse under the frozen
rule, while their SGD counterparts are stable except for Conv1-legacy, which
remains finite but stays at chance. Conv3 does not follow that pattern:
Conv3-ours becomes non-finite with SGD while remaining stable with Adam, and
Conv3 baseline and legacy are stable with both optimizers.

The defensible statement is therefore that candidate-beta stability depends
on the **architecture x amplification scheme x optimizer/learning-rate
contract**. This study used the existing fixed parameter-wise LR vector for
each SGD and Adam surface, so it cannot isolate the optimizer algorithm from
its learning-rate vector. A claim about Adam versus SGD alone would require a
separate matched-retuning experiment. Until then, beta qualification should be
recorded against the complete training contract rather than assigned only by
architecture and amplification scheme.

## Conv1 and Conv3 Candidate Betas and Clean Training

| Architecture | Scheme | Injected beta, one/two decades | Base beta, one/two decades | Best validation, one/two decades | Final validation, one/two decades |
|---|---|---:|---:|---:|---:|
| Conv1 | baseline | `100 / 10` | `100 / 10` | `96.24 / 96.22%` | `96.12 / 96.10%` |
| Conv1 | ours | `30 / 3` | `7.5 / .75` | `96.44 / 96.42%` | `96.34 / 96.32%` |
| Conv1 | legacy | `3 / .3` | `.1875 / .01875` | `96.64 / 96.62%` | `96.58 / 96.52%` |
| Conv3 | baseline | `100 / 10` | `100 / 10` | `97.02 / 97.06%` | `97.02 / 97.06%` |
| Conv3 | ours | `3 / .3` | `.046875 / .0046875` | `98.40 / 98.44%` | `98.40 / 98.44%` |
| Conv3 | legacy | `.001 / 1e-4` | `2.44140625e-7 / 2.44140625e-8` | `98.46 / 98.42%` | `98.44 / 98.42%` |

All `12/12` trajectories complete with finite ten-epoch histories. Moving one
additional decade lower changes best validation accuracy by only
`-.02/-.02/-.02 pp` for Conv1 baseline/ours/legacy and
`+.04/+.04/-.04 pp` for Conv3. The largest absolute best-accuracy difference
is `.04 pp`, so this single-seed clean MNIST study does not select between the
two tiers. There is no observed instability at either tier.

## Conv2 Beta Under Endpoint Read Noise

### Rejected original boundary tier

The first Conv2 training study used the cosine-boundary injected betas
`1000/100/.3`, corresponding to base beta `1000/6.25/.001171875` for
baseline/ours/legacy. It covered sigma `{0,1e-6,1e-5,1e-4}`. The retained
Trex histories give the following outcomes; “recorded epochs” is the number of
complete epoch records before termination or completion.

| Scheme | Sigma | Injected beta | Recorded epochs | Best validation | Last recorded validation | Outcome |
|---|---:|---:|---:|---:|---:|---|
| baseline | `0` | `1000` | `5` | `96.32%` | `95.72%` | non-finite termination |
| baseline | `1e-6` | `1000` | `5` | `95.86%` | `66.22%` | non-finite termination |
| baseline | `1e-5` | `1000` | `4` | `95.96%` | `95.42%` | non-finite termination |
| baseline | `1e-4` | `1000` | `5` | `96.42%` | `96.04%` | non-finite termination |
| ours | `0` | `100` | `2` | `96.18%` | `81.24%` | non-finite termination |
| ours | `1e-6` | `100` | `1` | `96.44%` | `96.44%` | non-finite termination |
| ours | `1e-5` | `100` | `2` | `95.58%` | `80.78%` | non-finite termination |
| ours | `1e-4` | `100` | `1` | `95.86%` | `95.86%` | non-finite termination |
| legacy | `0` | `.3` | `10` | `97.82%` | `89.34%` | finite completion with late collapse |
| legacy | `1e-6` | `.3` | `6` | `97.80%` | `94.46%` | intentionally stopped during epoch 7 after the lower-beta replacement was requested |
| legacy | `1e-5` | `.3` | `10` | `97.78%` | `88.04%` | finite completion with late collapse |
| legacy | `1e-4` | `.3` | `10` | `97.60%` | `87.28%` | finite completion with late collapse |

All `8/8` baseline/ours cases failed, including both clean controls. This
shows that their failures cannot be attributed to read noise. The legacy
cases demonstrate a different failure mode: numerically finite trajectories
can still be unusable because validation accuracy destabilizes after the early
peak. In every baseline/ours failure, the explicit finite-value guard raised
`NonFiniteTrainingError` on `Layer_1`: during epoch 5 or 6 for baseline and
epoch 2 or 3 for ours, depending on sigma. These observations are why the
lower-beta runs below were performed.

### Beta and noise conditions

The Conv2 one-decade tier uses injected beta `100/10/.03` for
baseline/ours/legacy; the corresponding base betas are
`100/.625/.0001171875`. The two-decade tier uses injected beta `10/1/.003`
and base beta `10/.0625/1.171875e-5`.

| Scheme | Injected beta, one/two decades | Base beta, one/two decades |
|---|---:|---:|
| baseline | `100 / 10` | `100 / 10` |
| ours | `10 / 1` | `.625 / .0625` |
| legacy | `.03 / .003` | `.0001171875 / 1.171875e-5` |

For these runs, sigma is the standard deviation of independent zero-mean
Gaussian noise added to every copied non-input equilibrium endpoint voltage
used by the local EqProp gradient:

```text
V_read = V_equilibrium + Normal(0, sigma^2)
```

The two centered phases, layers, elements, minibatches, and updates receive
independent draws. Noise changes only the endpoint readings after equilibrium;
it does not perturb relaxation, input voltages, or validation. Sigma is in the
simulator's voltage units and is not yet a hardware-calibrated voltage or ADC
specification.

### One- versus two-decade comparison through sigma `1e-4`

| Beta tier | Scheme | Sigma | Injected beta | Best validation | Final validation | Final change versus matched clean |
|---|---|---:|---:|---:|---:|---:|
| one decade | baseline | `0` | `100` | `97.10%` | `97.00%` | `0 pp` |
| one decade | baseline | `1e-5` | `100` | `97.02%` | `96.96%` | `-.04 pp` |
| one decade | baseline | `1e-4` | `100` | `96.94%` | `96.86%` | `-.14 pp` |
| one decade | ours | `0` | `10` | `97.86%` | `97.86%` | `0 pp` |
| one decade | ours | `1e-5` | `10` | `97.92%` | `97.88%` | `+.02 pp` |
| one decade | ours | `1e-4` | `10` | `97.80%` | `97.80%` | `-.06 pp` |
| one decade | legacy | `0` | `.03` | `98.34%` | `98.34%` | `0 pp` |
| one decade | legacy | `1e-5` | `.03` | `98.16%` | `98.16%` | `-.18 pp` |
| one decade | legacy | `1e-4` | `.03` | `97.66%` | `97.52%` | `-.82 pp` |
| two decades | baseline | `0` | `10` | `96.98%` | `96.96%` | `0 pp` |
| two decades | baseline | `1e-5` | `10` | `97.00%` | `96.90%` | `-.06 pp` |
| two decades | baseline | `1e-4` | `10` | `96.92%` | `96.84%` | `-.12 pp` |
| two decades | ours | `0` | `1` | `97.86%` | `97.84%` | `0 pp` |
| two decades | ours | `1e-5` | `1` | `97.72%` | `97.70%` | `-.14 pp` |
| two decades | ours | `1e-4` | `1` | `97.10%` | `97.08%` | `-.76 pp` |
| two decades | legacy | `0` | `.003` | `98.38%` | `98.38%` | `0 pp` |
| two decades | legacy | `1e-5` | `.003` | `97.74%` | `97.44%` | `-.94 pp` |
| two decades | legacy | `1e-4` | `.003` | `96.70%` | `96.46%` | `-1.92 pp` |

All `18/18` retained runs completed all ten epochs with finite histories. The
six configured sigma-`1e-6` cases were intentionally omitted when the grid was
stopped at sigma `1e-5`; they are not failures or missing retained cases.

At sigma `1e-5`, the two-decade clean-relative final penalties are
`-.06/-.14/-.94 pp` for baseline/ours/legacy, compared with
`-.04/+.02/-.18 pp` at one decade. At sigma `1e-4`, they are
`-.12/-.76/-1.92 pp`, compared with `-.14/-.06/-.82 pp` at one decade.
Consequently, at sigma `1e-4`, one-decade final validation exceeds the
two-decade result by `+.02/+.72/+1.06 pp` for baseline/ours/legacy. The
smaller phase signal at two decades therefore makes ours and especially
legacy more sensitive to the same absolute read noise.

### One-decade extension through sigma `5e-4`

The original six-case extension below uses Adam and its original matched clean
parent.

| Scheme | Sigma | Injected beta | Best validation | Final validation | Final change versus matched clean | State |
|---|---:|---:|---:|---:|---:|---|
| baseline | `3e-4` | `100` | `96.92%` | `96.76%` | `-.24 pp` | complete `10/10` |
| baseline | `5e-4` | `100` | `97.00%` | `96.78%` | `-.22 pp` | complete `10/10` |
| ours | `3e-4` | `10` | `97.62%` | `97.60%` | `-.26 pp` | complete `10/10` |
| ours | `5e-4` | `10` | `97.46%` | `97.44%` | `-.42 pp` | complete `10/10` |
| legacy | `3e-4` | `.03` | `97.28%` | `97.14%` | `-1.20 pp` | complete `10/10` |
| legacy | `5e-4` | `.03` | `97.12%` | `96.90%` | `-1.44 pp` | complete `10/10` |

All `6/6` higher-noise runs remained finite and completed ten epochs, so no
training instability was observed at either added noise level. At sigma
`5e-4`, ours retains the highest final validation accuracy (`97.44%`), while
legacy has the largest clean-relative penalty (`-1.44 pp`). The slight
baseline non-monotonicity between sigma `3e-4` and `5e-4` is not evidence of a
trend in this single-seed study.

#### SGD repeat at sigma `5e-4`

The exact one-decade betas, `T=K=8`, initialization, split, minibatch order,
and noisy endpoint seed were retained while switching to the historical SGD
learning-rate vectors. For a common optimizer-paired reference, the table uses
the later boundary study's clean one-decade SGD and Adam controls.

| Scheme | SGD clean final | SGD noisy best / final | SGD final change | Adam clean final | Adam noisy best / final | Adam final change |
|---|---:|---:|---:|---:|---:|---:|
| baseline | `96.52%` | `96.34 / 96.34%` | `-.18 pp` | `96.88%` | `97.00 / 96.78%` | `-.10 pp` |
| ours | `97.80%` | `97.46 / 97.46%` | `-.34 pp` | `97.92%` | `97.46 / 97.44%` | `-.48 pp` |
| legacy | `97.52%` | `96.72 / 96.72%` | `-.80 pp` | `98.40%` | `97.12 / 96.90%` | `-1.50 pp` |

All `3/3` SGD runs completed ten finite epochs, so sigma `5e-4` causes no
observed instability with either optimizer contract. Ours has the highest
noisy final accuracy under both (`97.46%` SGD and `97.44%` Adam), and legacy
has the largest clean-relative penalty under both (`-.80 pp` and `-1.50 pp`).
Thus the qualitative noise result is unchanged, while its magnitude is
optimizer/LR-contract dependent. The noisy SGD-minus-Adam final differences
are `-.44/+.02/-.18 pp` for baseline/ours/legacy.

This does not isolate the optimizer algorithm: SGD and Adam necessarily use
different historical scheme-specific LR vectors. The earlier Adam table
above anchors to its original clean parent and therefore reports
`-.22/-.42/-1.44 pp`; the noisy Adam endpoints are the same in both tables.
The small clean-control differences are retained rather than selected away.

### Conv1/Conv3 zero-bias shared-`T/K` Adam at sigma `5e-4`

The matched follow-up uses the exact zero-bias one-seed contract and its
reviewed clean qualification: inherited BPTT Adam weight rates, every bias
rate and tensor fixed at zero, shared `T/K=4/4` for Conv1 and `8/8` for
Conv3, `10/30` epochs, and the one-decade injected betas. The fixed endpoint
noise seed is `2026081601`; input noise and official-test access are disabled.

| Network | Scheme | Injected beta | Noisy best / final | Clean best / final | Noisy minus clean, best / final | Stable |
|---|---|---:|---:|---:|---:|---:|
| Conv1 | baseline | `100` | `96.18 / 96.10%` | `96.18 / 96.04%` | `0.00 / +.06 pp` | yes |
| Conv1 | ours | `30` | `96.46 / 96.28%` | `96.42 / 96.34%` | `+.04 / -.06 pp` | yes |
| Conv1 | legacy | `3` | `96.40 / 96.38%` | `96.44 / 96.42%` | `-.04 / -.04 pp` | yes |
| Conv3 | baseline | `100` | `96.94 / 96.84%` | `97.76 / 97.68%` | `-.82 / -.84 pp` | yes |
| Conv3 | ours | `3` | `96.96 / 96.82%` | `98.72 / 98.64%` | `-1.76 / -1.82 pp` | yes |
| Conv3 | legacy | `.001` | `88.30 / 87.64%` | `99.08 / 99.02%` | `-10.78 / -11.38 pp` | yes |

All `6/6` runs complete their finite budgets and finish less than `5 pp`
below their own best, so none meets the predeclared within-run collapse rule.
That classification must not hide the performance result. Every Conv1 noisy
trajectory stays within `.08 pp` of its clean partner, whereas all three
Conv3 schemes lose accuracy and legacy is severely degraded. Conv3 legacy
continues learning to `88.30%` best validation, but the `11.38 pp` final
penalty shows that one-decade beta is not a practically robust common point at
this noise level.

The authoritative local copy is checksum-identical to Jean Zay and all six
bundles, six run receipts, three pack receipts, initialization/split/order,
`2,887,920` endpoint-noise draws, zero-bias, and no-test-read guards validate.
This remains one-seed, one-noise-realization exploratory evidence.

### Direct one-decade gradient fidelity at sigma `5e-4`

The read-noise training results above establish finite multi-step learning,
but they do not show whether each individual EqProp gradient remains close to
BPTT. A new read-only replay measures that stricter question for Conv2 and
Conv3 under the exact zero-bias/shared-`T/K` contract. It uses reconstructed
initialization and best-validation checkpoints, four fixed batches of 16,
centered true-float64 EqProp, matched independent Gaussian reads on every
negative/positive free-layer endpoint, and the existing `.99` cosine / `.10`
symmetric-norm gate. The input and BPTT reference remain exact.

| Network | Scheme | Injected beta | Worst clean cosine, init / best | Worst noisy cosine, init / best |
|---|---|---:|---:|---:|
| Conv2 | baseline | `100` | `.9949 / .9981` | `.5476 / .2136` |
| Conv2 | ours | `10` | `.9951 / .9993` | `.2423 / -.0888` |
| Conv2 | legacy | `.03` | `.9997 / .9993` | `.00715 / -.1713` |
| Conv3 | baseline | `100` | `.9880 / .9999` | `-.0862 / -.0554` |
| Conv3 | ours | `3` | `.9392 / .9996` | `-.0551 / -.2066` |
| Conv3 | legacy | `.001` | `.9989 / 1.0000` | `-.00696 / -.0720` |

None of the 12 architecture x scheme x checkpoint contexts passes the
all-layer acquisition or task gate. At layer-batch resolution, noisy versus
clean EqProp passes `52/168`, noisy EqProp versus BPTT passes `50/168`, and
the combined usable gate passes `50/168`. All `48/48` Dense rows pass. Only
two additional task rows pass, both from Conv2-baseline C1; every other
convolutional layer/scheme group has `0/8` task passes. The embedded clean
replay is exactly identical to the accepted one-decade parent over all 168
rows, so the degradation is attributable to the declared endpoint readout.

This resolves an apparent tension rather than contradicting the finite
training runs. Sigma `5e-4` does not cause observed optimizer-level collapse,
but one individual two-phase estimate is strongly noise-dominated in the
convolutional layers—especially when divided by the very small legacy base
beta. Many minibatches and optimizer steps can average stochastic errors that
fail a strict single-batch `.99/.10` direction-and-scale gate. Therefore the
one-decade tier is training-stable at sigma `5e-4`, but it is not
per-minibatch all-layer gradient-faithful at that noise level.

### Conv2 clean/noisy extension through ten times the original maximum

To expose the high-beta side of the tradeoff, the Conv2 replay was extended
from the original maximum to ten times that maximum. Relative to the
one-decade reference, the added scales are `10`, `17.7828`, `31.6228`,
`56.2341`, and `100`. They correspond to injected beta baseline
`1000/1778.28/3162.28/5623.41/10000`, ours
`100/177.828/316.228/562.341/1000`, and legacy
`.3/.533484/.948683/1.68702/3`. Each point uses the same Conv2
reconstructed-initialization and best-validation checkpoints, four fixed
16-example batches, `T=K=6`, exact-zero biases, centered true-float64 EqProp,
and the identical sigma-`5e-4` endpoint-noise tensors. The noisy runner also
retains the matched clean gradient before noise is added.

All five canonical bundles validate. Their Conv2 slice contains `120`
checkpoint-batch replays, `360` layer-batch rows, and `720` endpoint-noise
records. BPTT norms are exactly invariant across beta; all noise signatures
match byte-for-byte; all Conv2 residual, source/checkpoint, parameter,
frozen-force, cohort, zero-bias, float64, no-optimizer, and no-test-read guards
pass. The scale-10 embedded clean replay reproduces the accepted clean maximum
within `1.30e-14` in cosine and exactly in BPTT norm and norm delta.

At the trained checkpoint, the four-batch mean noisy-cosine optima and the
clean means at the enlarged endpoint are:

| Scheme | Best noisy C0 | Best noisy C1 | Clean C0/C1/Dense at 10x maximum |
|---|---:|---:|---:|
| baseline | `.983554 @ B=3162.28` | `.994513 @ B=1000` | `.969744/.946317/.995145` |
| ours | `.953794 @ B=1000` | `.985062 @ B=177.828` | `.965700/.943066/.993353` |
| legacy | `.332723 @ B=3` | `.922408 @ B=3` | `.932222/.938292/.991046` |

Larger beta does improve the noise-limited upstream signal: for example,
trained baseline C0 rises from noisy mean `.274004` at the one-decade point
to `.983554`, ours C0 from `.038457` to `.953794`, and legacy C0 from
`-.074354` to `.332723`. However, no upstream layer reaches a common
all-batch `.99` point before clean finite-beta distortion takes over. By
relative scale `31.6228`, only Dense layers retain the conservative all-batch
gate; by `56.2341`, every trained convolution-layer mean is below `.99`. At
the `100` endpoint, only baseline Dense retains an all-batch gate pass (ours
and legacy Dense remain above `.99` on average but have at least one failing
batch). Initialization deteriorates even more sharply.

This is direct-gradient-fidelity failure, not a new training-collapse test.
It confirms the proposed mechanism: a larger layer-specific beta can improve
an early layer's read-noise SNR, but the best value differs by layer and is
bounded by finite-nudge bias. Any layer-specific or scheduled-beta estimator
would therefore require a separate training and estimator-consistency
qualification.

Artifacts: [paper summary table](../papers/amplification_overleaf/figures/beta_cosine/beta_cosine_summary.csv),
[paper provenance](../papers/amplification_overleaf/figures/beta_cosine/provenance.json),
and [Conv2 clean/noisy figure](../papers/amplification_overleaf/figures/beta_cosine/conv2_by_scheme_all_layers_cosine_vs_beta.pdf).

### Conv3 broadened cosine sweep through ten times the original maximum at sigma `5e-4`

The direct sweep now contains 17 uniformly log-spaced beta factors
`10^(i/8)`, `i=0..16`. The actual injected ranges are baseline
`100--10000`, ours `3--300`, and legacy `.001--.1`: one decade below the
original maximum through one decade above it. Every point reuses the same
reconstructed-initialization and best-validation checkpoints, four 16-example
validation batches, `T=K=8`, exact-zero biases, centered true-float64
estimator, BPTT reference, and endpoint-noise tensors. The joined surface
contains `408` checkpoint-batch replays, `1632` layer-batch cosine
comparisons, and `3264` endpoint-noise records. All 17 canonical production
bundles and the new extreme-point smoke validate. All 17 noise signatures
match exactly, and all 96 BPTT reference norms are beta-invariant with maximum
delta zero.

The first table reports the best *worst-batch* noisy EqProp/BPTT cosine found
for each layer, followed by the injected beta where it occurs. This is the
conservative view used for the `.99` gate.

| Scheme | Checkpoint | C0 | C1 | C2 | Dense |
|---|---|---:|---:|---:|---:|
| baseline | initialization | `.162914 @ 1778.28` | `.727609 @ 4216.97` | `.986319 @ 316.228` | `.999845 @ 100` |
| baseline | best | `.786156 @ 10000` | `.976993 @ 10000` | `.994911 @ 1778.28` | `.999998 @ 1778.28` |
| ours | initialization | `-.015895 @ 53.3484` | `.424453 @ 300` | `.980376 @ 12.6509` | `.999961 @ 3` |
| ours | best | `-.039670 @ 300` | `.145193 @ 300` | `.962701 @ 126.509` | `.999995 @ 30` |
| legacy | initialization | `.035054 @ .1` | `.012911 @ .1` | `.980881 @ .056234` | `.999997 @ .004217` |
| legacy | best | `-.071393 @ .1` | `.000380 @ .1` | `.524586 @ .1` | `.999990 @ .023714` |

For the trained best checkpoints, the four-batch arithmetic-average peaks are:

| Scheme | C0 | C1 | C2 | Dense |
|---|---:|---:|---:|---:|
| baseline | `.897801 @ 10000` | `.980900 @ 10000` | `.996141 @ 1778.28` | `.999999 @ 1778.28` |
| ours | `.091214 @ 300` | `.189009 @ 300` | `.976147 @ 168.702` | `.999996 @ 30` |
| legacy | `.056734 @ .1` | `.003716 @ .1` | `.577923 @ .1` | `.999992 @ .023714` |

None of the six scheme x checkpoint contexts has an all-layer/all-batch
cosine-`.99` point. Dense passes from the one-decade anchor in every context;
the only convolutional crossing remains trained baseline C2, first at
`B=1000`. The broader range identifies interior trained-C2 average optima for
baseline (`B=1778.28`) and ours (`B=168.702`), followed by declines to
`.983442` and `.970383` at their enlarged maxima. By contrast, trained C0/C1
for baseline and ours and C0/C1/C2 for legacy still improve on average up to
the new ceiling, but remain far below `.99`. Their noise-limited optima are
therefore not bounded even by this extension, and increasing beta further
would not yield an all-layer faithful point.

The initialization rows show the opposing finite-beta effect directly:
baseline and ours reach interior all-layer optima and then deteriorate, while
at the largest tested point the clean worst-layer cosine is negative for some
initialization batches. At the trained checkpoints, the best observed
all-layer worst-batch values are only baseline `.786156` at `B=10000`, ours
`-.039670` at `B=300`, and legacy `-.071393` at `B=.1`; the corresponding
clean worst cosines are `.978357`, `.926021`, and `.969854`. Endpoint read
noise, not only finite-beta bias, therefore remains the limiting mechanism.

These enlarged betas are diagnostic points outside the previously qualified
training interval, not nominated training hyperparameters. The earlier
optimizer study already found optimizer-dependent collapse at or below the
original maximum. The extension establishes curve shape and layer-specific
peaks; it does not overturn the one-/two-decade training decision. Reducing
endpoint noise or averaging independent reads or minibatch gradients targets
the remaining early-layer failure more directly.

Artifacts: [broadened analysis report](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-above-max-to-10xmax-sigma-5em4-seed0-20260817-v1/analysis/report.md),
[best-checkpoint four-batch averages](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-above-max-to-10xmax-sigma-5em4-seed0-20260817-v1/analysis/best_checkpoint_average_cosine_vs_beta.csv),
[average-per-layer plot](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-above-max-to-10xmax-sigma-5em4-seed0-20260817-v1/analysis/plots/best_checkpoint_average_cosine_vs_injected_beta_all_layers.png),
[all-layer table](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-above-max-to-10xmax-sigma-5em4-seed0-20260817-v1/analysis/layerwise_cosine_vs_beta.csv),
[per-layer thresholds](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-above-max-to-10xmax-sigma-5em4-seed0-20260817-v1/analysis/per_layer_thresholds.csv),
[median/spread plot](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-above-max-to-10xmax-sigma-5em4-seed0-20260817-v1/analysis/plots/median_cosine_vs_injected_beta_all_layers.png),
and [worst-batch plot](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-above-max-to-10xmax-sigma-5em4-seed0-20260817-v1/analysis/plots/worst_batch_cosine_vs_injected_beta_all_layers.png).

## Layer-Displacement Definition

For each non-input state layer, the tables report

```text
mean absolute displacement = 0.5 * (RMS(V+ - V0) + RMS(V- - V0))
mean relative displacement = 0.5 * (RMS(V+ - V0) / RMS(V0)
                                  + RMS(V- - V0) / RMS(V0))
```

Here `V0` is the matched zero-nudge endpoint after the same `K` iterations.
`V+` and `V-` are the positive and negative centered-nudge endpoints. RMS is
over all examples and elements in the stated layer. Thus “average” below
means the arithmetic mean of the positive- and negative-phase displacement
magnitudes; it is not a signed mean. Each table entry is
`absolute per-node RMS displacement (relative displacement)`.

Layer names map to the downstream parameter interaction: Conv1 `Layer_1` is
the C0 state and `Layer_2` is the dense/output state. Conv3 `Layer_1`,
`Layer_2`, and `Layer_3` are the C0, C1, and C2 states, and `Layer_4` is the
dense/output state.

## Conv1 Layer Displacement

| Scheme | Tier and injected beta | Layer 1: C0 | Layer 2: output |
|---|---:|---:|---:|
| baseline | one decade, `100` | `5.79234e-3 (3.23851e-3)` | `.294638 (1.93678)` |
| baseline | two decades, `10` | `5.81398e-4 (3.25061e-4)` | `.0294635 (.193676)` |
| ours | one decade, `30` | `2.07022e-3 (2.94309e-3)` | `.237357 (1.45921)` |
| ours | two decades, `3` | `2.07130e-4 (2.94463e-4)` | `.0237351 (.145917)` |
| legacy | one decade, `3` | `2.41817e-3 (3.15992e-3)` | `1.16115 (7.23603)` |
| legacy | two decades, `.3` | `2.42092e-4 (3.16352e-4)` | `.116115 (.723603)` |

## Conv3 Layer Displacement

| Scheme | Tier and injected beta | Layer 1: C0 | Layer 2: C1 | Layer 3: C2 | Layer 4: output |
|---|---:|---:|---:|---:|---:|
| baseline | one decade, `100` | `8.28185e-6 (1.93673e-7)` | `3.06801e-5 (2.01382e-6)` | `5.26434e-4 (5.39792e-4)` | `.0287977 (.103791)` |
| baseline | two decades, `10` | `8.28097e-7 (1.93653e-8)` | `3.06763e-6 (2.01357e-7)` | `5.26518e-5 (5.39879e-5)` | `.00287977 (.0103791)` |
| ours | one decade, `3` | `1.55381e-7 (1.26351e-8)` | `2.51870e-6 (1.83185e-6)` | `1.95268e-4 (9.37193e-4)` | `.149367 (.591825)` |
| ours | two decades, `.3` | `1.55297e-8 (1.26283e-9)` | `2.51885e-7 (1.83196e-7)` | `1.95346e-5 (9.37571e-5)` | `.0149367 (.0591826)` |
| legacy | one decade, `.001` | `1.21380e-9 (1.50453e-10)` | `7.42667e-8 (2.36981e-7)` | `4.27121e-5 (3.90506e-4)` | `.187203 (.520165)` |
| legacy | two decades, `1e-4` | `1.21365e-10 (1.50435e-11)` | `7.42528e-9 (2.36937e-8)` | `4.27030e-6 (3.90422e-5)` | `.0187203 (.0520165)` |

The Conv3-legacy two-decade point at injected `B=1e-4` was measured with an
exact replay rather than extrapolated from `B=.001`. The replay used base
beta `2.44140625e-8`, matched the source cohort and checkpoint-selection
hashes, completed all eight float64 signed layer-endpoint rows, and passed the
dtype, iteration, parameter, arithmetic, cohort, and equilibrium-residual
guards. It applied no optimizer step and did not read the official test set.

## What the Measurements Show

- The original cosine-boundary candidate is not a uniformly safe training
  point. All six Conv1/Conv2 Adam candidate runs collapse, Conv3-ours SGD also
  becomes non-finite, and Conv1-legacy SGD stays at chance. In contrast, all
  18 matched one-decade-lower controls are stable. The strong optimizer
  interaction means “maximum beta” must be tied to the complete
  optimizer-and-LR training contract rather than treated as an
  architecture-and-scheme constant or attributed to the optimizer algorithm
  alone.
- Clean Conv1/Conv3 training does not distinguish the one- and two-decade
  tiers, but the new zero-bias/shared-`T/K` gradient gate does: the one-decade
  tier fails three Conv3-initialization C0 batch rows, whereas two decades
  passes all `216/216`. The Conv2 read-noise comparison exposes the opposing
  signal-to-noise tradeoff: at fixed absolute read noise, one decade is
  materially more robust for Conv2 ours and legacy.
- Under the exact zero-bias/shared-`T/K` Adam training contract, all nine
  one-decade cases complete 10/30/30 finite epochs and finish within `.14 pp`
  of their own best validation. Thus training stability no longer argues for
  moving to two decades on ordinary MNIST. The remaining reason to prefer two
  decades is direct small-perturbation gradient fidelity, while the reasons to
  prefer one decade are its demonstrated stability and tenfold larger phase
  signal.
- The Conv2 one-decade tier remains finite through sigma `5e-4` under both the
  historical SGD and Adam contracts. Ours has the highest noisy final
  validation and legacy the largest clean-relative penalty with both, although
  the penalty magnitude changes with optimizer/LR contract. This argues
  against moving another decade lower solely for clean-training safety on
  MNIST. It does not establish that the same choice is safe on harder datasets.
- Under the exact zero-bias/shared-`T/K` Adam contract, all six Conv1/Conv3
  sigma-`5e-4` runs also remain finite and stable. Conv1 changes by at most
  `.08 pp` anywhere in its trajectory, but Conv3 final clean-relative
  penalties are `-.84/-1.82/-11.38 pp` for baseline/ours/legacy. Thus
  within-run stability alone is too weak as the read-noise acceptance rule:
  Conv3 legacy is formally stable yet practically strongly degraded.
- The direct sigma-`5e-4` replay adds a different constraint: none of the 12
  Conv2/Conv3 scheme/checkpoint contexts passes the strict all-layer
  per-minibatch gradient gate, even though all Dense rows pass. Thus
  optimizer-level training stability and instantaneous gradient fidelity must
  be reported separately; finite noisy training does not imply that each
  noisy EqProp update is a close BPTT estimate.
- The broadened 17-point Conv3 sweep confirms that increasing beta above the
  original maximum does not create an all-layer faithful point at sigma
  `5e-4`. It locates interior trained-C2 average peaks for baseline at
  `B=1778.28` and ours at `B=168.702`, followed by finite-beta decline, while
  trained early-layer averages remain far below `.99` even when still rising
  at the enlarged ceiling. These above-maximum values characterize the
  mechanism; they are not training-qualified candidates.
- Reducing beta by one decade reduces displacement by almost exactly one
  decade in every measured layer. The one-/two-decade absolute-displacement
  ratios range from about `9.96` to `10.01`.
- The output layer moves much more than the upstream convolutional states,
  especially in Conv3. This makes an output-displacement target measurable,
  but it does not by itself guarantee adequate upstream signal.
- A common margin below each scheme's cosine boundary does not equalize output
  displacement. At the one-decade tier, Conv1 output displacement is
  `.294638/.237357/1.16115` for baseline/ours/legacy, a `4.89x` maximum/minimum
  range. Conv3 output displacement is `.0287977/.149367/.187203`, a `6.50x`
  range. The two-decade tier preserves essentially the same ratios because all
  layers remain nearly linear in beta over this interval.
- The largest clean-stable beta observed on MNIST is not necessarily the
  safest transferable choice for harder datasets. The current clean accuracy
  evidence cannot resolve this concern.
- Matching output displacement would require an explicitly scheme-specific
  beta rule, followed by a new check that layerwise gradient fidelity,
  upstream displacement, clean training, and read-noise robustness all remain
  acceptable. Output displacement should therefore be treated as a candidate
  selection coordinate, not yet as the selection rule.

## Artifacts

- Zero-bias/shared-`T/K` one-decade Adam training qualification: [report](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-one-decade-ordinary-mnist-10-30-30ep-seed0-20260816-v1/analysis/report.md), [summary](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-one-decade-ordinary-mnist-10-30-30ep-seed0-20260816-v1/analysis/summary.csv), [trajectories](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-one-decade-ordinary-mnist-10-30-30ep-seed0-20260816-v1/analysis/validation_accuracy_trajectories.png), and [verification](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-one-decade-ordinary-mnist-10-30-30ep-seed0-20260816-v1/analysis/verification.json).
- Zero-bias/shared-`T/K` Adam gradient qualification: [review](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-tk4-6-8-seed0-20260816-v1/review.md), [one-decade summary](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-tk4-6-8-seed0-20260816-v1/production/case_summary.csv), and [two-decade summary](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-tk4-6-8-seed0-20260816-v1/production-two-decades/case_summary.csv).
- Maximum-beta shared-`T/K` extension: [case summary](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-tk4-6-8-seed0-20260816-v1/production-max-beta/case_summary.csv) and [layer table](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-tk4-6-8-seed0-20260816-v1/production-max-beta/layer_metrics.csv).
- Direct Conv1/Conv2/Conv3 boundary-collapse study: [report](../results/perfectdiode-conv123-centered-float64-eqprop-beta-collapse-boundary-vs-1decade-sgd-adam-10ep-seed0-20260815-v1/analysis/report.md), [pair table](../results/perfectdiode-conv123-centered-float64-eqprop-beta-collapse-boundary-vs-1decade-sgd-adam-10ep-seed0-20260815-v1/analysis/pairs.csv), [case table](../results/perfectdiode-conv123-centered-float64-eqprop-beta-collapse-boundary-vs-1decade-sgd-adam-10ep-seed0-20260815-v1/analysis/cases.csv), [full summary](../results/perfectdiode-conv123-centered-float64-eqprop-beta-collapse-boundary-vs-1decade-sgd-adam-10ep-seed0-20260815-v1/analysis/summary.json), [collection receipt](../results/perfectdiode-conv123-centered-float64-eqprop-beta-collapse-boundary-vs-1decade-sgd-adam-10ep-seed0-20260815-v1/analysis/collection_validation_receipt.json), and [trajectories](../results/perfectdiode-conv123-centered-float64-eqprop-beta-collapse-boundary-vs-1decade-sgd-adam-10ep-seed0-20260815-v1/analysis/validation_accuracy_trajectories.png).
- Rejected Conv2 boundary-tier training: [study root](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-training-10ep-seed0-20260812-v1/) and [frozen study config](../configs/conv/perfectdiode_conv2_centered_float64_eqprop_read_noise_training_10ep_seed0_20260812_v1.json). The retained production bundles and partial metrics remain on Trex at the remote path recorded in `current_simulations.md`.
- Clean qualification: [report](../results/perfectdiode-conv13-centered-float64-eqprop-beta-qualification-1to2decades-10ep-seed0-20260814-v1/analysis/report.md), [summary](../results/perfectdiode-conv13-centered-float64-eqprop-beta-qualification-1to2decades-10ep-seed0-20260814-v1/analysis/summary.csv), and [trajectories](../results/perfectdiode-conv13-centered-float64-eqprop-beta-qualification-1to2decades-10ep-seed0-20260814-v1/analysis/validation_accuracy_trajectories.png).
- Conv2 one-/two-decade read-noise comparison: [review](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-beta-down-1to2decades-10ep-seed0-20260812-v1/review.md), [report](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-beta-down-1to2decades-10ep-seed0-20260812-v1/analysis/report.md), [terminal table](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-beta-down-1to2decades-10ep-seed0-20260812-v1/analysis/terminal_summary.csv), and [penalty plot](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-beta-down-1to2decades-10ep-seed0-20260812-v1/analysis/noise_penalty_by_beta_tier.png).
- Conv2 one-decade higher-noise extension: [report](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-3em4-5em4-10ep-seed0-20260813-v1/analysis/report.md), [terminal table](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-3em4-5em4-10ep-seed0-20260813-v1/analysis/terminal_summary.csv), and [noise curve](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-3em4-5em4-10ep-seed0-20260813-v1/analysis/final_validation_vs_noise.png).
- Conv2 SGD repeat and optimizer/LR-contract comparison at sigma `5e-4`: [report](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-5em4-sgd-10ep-seed0-20260816-v1/analysis/report.md), [terminal table](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-5em4-sgd-10ep-seed0-20260816-v1/analysis/terminal_summary.csv), [trajectories](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-5em4-sgd-10ep-seed0-20260816-v1/analysis/validation_accuracy_trajectories.png), [verification](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-5em4-sgd-10ep-seed0-20260816-v1/analysis/verification.json), and [collection receipt](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-5em4-sgd-10ep-seed0-20260816-v1/analysis/collection_validation_receipt.json).
- Conv1/Conv3 zero-bias/shared-`T/K` Adam training at sigma `5e-4`: [report](../results/perfectdiode-conv13-zero-bias-adam-centered-float64-eqprop-one-decade-sigma-5em4-ordinary-mnist-10-30ep-seed0-20260817-v1/analysis/report.md), [summary](../results/perfectdiode-conv13-zero-bias-adam-centered-float64-eqprop-one-decade-sigma-5em4-ordinary-mnist-10-30ep-seed0-20260817-v1/analysis/summary.csv), [trajectories](../results/perfectdiode-conv13-zero-bias-adam-centered-float64-eqprop-one-decade-sigma-5em4-ordinary-mnist-10-30ep-seed0-20260817-v1/analysis/validation_accuracy_trajectories.png), and [collection receipt](../results/perfectdiode-conv13-zero-bias-adam-centered-float64-eqprop-one-decade-sigma-5em4-ordinary-mnist-10-30ep-seed0-20260817-v1/analysis/collection_validation_receipt.json).
- Conv2/Conv3 one-decade direct gradient replay at sigma `5e-4`: [report](../results/perfectdiode-conv23-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-one-decade-sigma-5em4-seed0-20260817-v1/analysis/report.md), [layer table](../results/perfectdiode-conv23-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-one-decade-sigma-5em4-seed0-20260817-v1/analysis/layer_summary.csv), [clean-versus-noisy plot](../results/perfectdiode-conv23-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-one-decade-sigma-5em4-seed0-20260817-v1/analysis/plots/layerwise_clean_vs_noisy_cosine.png), and [machine summary](../results/perfectdiode-conv23-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-one-decade-sigma-5em4-seed0-20260817-v1/analysis/summary.json).
- Conv3 broadened 17-point gradient sweep at sigma `5e-4`: [report](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-above-max-to-10xmax-sigma-5em4-seed0-20260817-v1/analysis/report.md), [four-batch-average table](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-above-max-to-10xmax-sigma-5em4-seed0-20260817-v1/analysis/best_checkpoint_average_cosine_vs_beta.csv), [average-per-layer plot](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-above-max-to-10xmax-sigma-5em4-seed0-20260817-v1/analysis/plots/best_checkpoint_average_cosine_vs_injected_beta_all_layers.png), [worst-batch plot](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-above-max-to-10xmax-sigma-5em4-seed0-20260817-v1/analysis/plots/worst_batch_cosine_vs_injected_beta_all_layers.png), and [machine summary](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-above-max-to-10xmax-sigma-5em4-seed0-20260817-v1/analysis/summary.json).
- Baseline/ours displacement source: [layer table](../results/perfectdiode-conv123-baseline-ours-centered-float64-beta-1em3-to-1e3-tk8-seed0-20260812-v1/analysis/layer_displacement.csv).
- Legacy displacement source: [layer table](../results/perfectdiode-conv123-legacy-centered-float64-beta-1em3-to-1e3-tk8-seed0-20260812-v1/analysis/layer_displacement.csv).
- Exact Conv3-legacy `B=1e-4` replay: [state displacement](../results/perfectdiode-conv13-centered-float64-eqprop-beta-qualification-1to2decades-10ep-seed0-20260814-v1/analysis/displacement_replay/conv3-legacy-B1em4-tk8/state_displacement.csv), [guards](../results/perfectdiode-conv13-centered-float64-eqprop-beta-qualification-1to2decades-10ep-seed0-20260814-v1/analysis/displacement_replay/conv3-legacy-B1em4-tk8/read_only_guards.json), and [result](../results/perfectdiode-conv13-centered-float64-eqprop-beta-qualification-1to2decades-10ep-seed0-20260814-v1/analysis/displacement_replay/conv3-legacy-B1em4-tk8/result.json).
