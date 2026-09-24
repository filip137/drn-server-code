# Baseline EqProp read-noise results

Completed September 18, 2026, at **00:39:15 CEST**. **All 22/22 new full
trainings are collected and fully revalidated:** 20 noisy runs and two new
clean controls. The [run table](baseline_read_noise_run_status_20260916.md)
contains every best/final measurement. All measurements use the
5,000-example ordinary-MNIST validation
cohort, with training seed 0 and noise seed 2026081601. Official-test reads
remain zero.

Conv1 shows at most 0.04 percentage points of observed final accuracy loss;
Conv2 shows no observed loss against its audited historical control. Conv3
loses 0.16–1.60 points as noise increases. At the highest noise level, ours
retains higher absolute accuracy than baseline for all three depths, while
baseline has the smaller observed clean-relative loss at depths 2 and 3.
Different betas, training contracts and environments limit causal comparisons.

## Completed Conv1 curves

Final validation accuracy, percent:

| Baseline beta | Clean | sigma 1e-5 | 3e-5 | 1e-4 | 3e-4 | 5e-4 |
|---|---:|---:|---:|---:|---:|---:|
| 100 | 96.04 | 96.04 | 96.04 | 96.00 | 96.02 | 96.04 |
| 200 | 96.02 | 96.02 | 96.02 | 96.02 | 96.02 | 96.04 |

Beta 100's largest observed clean-relative loss is **0.04 percentage points**,
equivalent to two validation examples. Beta 200 matches its clean accuracy
at the first four noise levels and exceeds it by one example at 5e-4.
Best-checkpoint accuracies span 96.18–96.20% for beta 100 and 96.16–96.18%
for beta 200. Every checkpoint is finite and all trainings complete their
declared ten epochs.

Both baseline curves therefore show little observed sensitivity over this
noise range for this seed. This agrees descriptively with the earlier Conv1
ours/legacy curves, whose final clean-relative changes were at most 0.08
points. These small changes do not establish statistical equivalence or a
benefit from increasing beta. The beta comparison also crosses GPU/software
noise-stream groups: beta 100 ran on Akib's RTX3080, whereas beta 200 ran on
Nom's RTX3090. Identical noise seeds do not give identical streams across
these groups. The beta-100 clean reference is an audited historical run;
the beta-200 clean reference is newly trained on Nom.

## Completed Conv2 results

Beta 100, full 30-epoch trainings:

| Sigma | Best validation % | Final validation % | Final clean-relative loss, pp |
|---|---:|---:|---:|
| 0, historical reference | 97.22 | 97.22 | 0.00 |
| 1e-5 | 97.30 | 97.30 | −0.08 |
| 3e-5 | 97.38 | 97.38 | −0.16 |
| 1e-4 | 97.36 | 97.36 | −0.14 |
| 3e-4 | 97.34 | 97.34 | −0.12 |
| 5e-4 | 97.24 | 97.24 | −0.02 |

None of the five noise levels shows observed degradation for this seed.
The increases correspond to four, eight, seven, six and one additional correctly classified
validation examples. They cannot be attributed to noise because the clean
reference uses a different GPU/software environment. Across the full grid,
final validation accuracy spans 97.24–97.38%.

At the highest noise level, baseline achieves 97.24%, ours 97.60%, and legacy
95.72%. Their respective clean references are 97.22%, 98.02%, and 98.06%:
baseline shows no observed loss, while ours retains the highest absolute
accuracy. These are descriptive comparisons under the inherited beta values
(100/10/.03), learning rates and amplification contracts, with environment
differences; they do not isolate the effect of amplification.

All five local production packs exit zero, each after about 1 hour 48 minutes.
The final case completes at 00:50:57 CEST on September 17. The queue and its
wrapper exit zero, the tmux window closes, and the local GPU has no training
process. Full source and collected bundles pass revalidation.

## Completed Conv3 results

Beta 10, full 30-epoch trainings on Fifi:

| Sigma | Best validation % | Final validation % | Final clean-relative loss, pp |
|---|---:|---:|---:|
| 0, new matched control | 97.64 | 97.64 | 0.00 |
| 1e-5 | 97.48 | 97.48 | 0.16 |
| 3e-5 | 97.08 | 97.04 | 0.60 |
| 1e-4 | 96.70 | 96.64 | 1.00 |
| 3e-4 | 96.52 | 96.48 | 1.16 |
| 5e-4 | 96.10 | 96.04 | 1.60 |

The lowest noise level loses 0.16 percentage points, or eight validation
examples, against its matching-beta clean control. Both runs use the same
GPU/software environment, initializer and complete data order. This remains
a single training/noise seed. Final loss grows monotonically across the
tested grid, reaching 1.60 points (80 validation examples) at sigma 5e-4.

At sigma 1e-4, baseline/ours/legacy finish at 96.64/97.62/83.08%, compared
with clean references of 97.64/98.64/98.78%. Baseline and ours have similar
observed clean-relative losses (1.00/1.02 points), while ours retains the
higher absolute accuracy. Legacy loses 15.70 points. These descriptive
comparisons inherit different betas (10/3/.001), learning rates and
amplification contracts, along with the recorded environment limitations;
they do not isolate amplification's causal effect.

At the highest noise level, baseline/ours/legacy finish at
**96.04/96.60/75.44%**, with respective clean-relative losses of
**1.60/2.04/23.34 points**. Baseline therefore shows substantially less
observed degradation than legacy in this experiment. Ours remains 0.56
points higher in absolute accuracy than baseline; baseline's smaller
clean-relative loss does not imply higher noisy accuracy.

The new beta-10 clean result is close to the earlier beta-100 baseline's
97.68% final validation accuracy: 0.04 points lower. Its best accuracy is
97.64%, compared with 97.76% previously. This is a descriptive cross-beta,
cross-environment comparison, not evidence that the betas are equivalent.
The earlier beta-100 score is not used to compute this noise curve's losses.
The recorded seed-1 beta-confirmation failure and T8 residual caveat remain.

Fifi runs all six cases in three consecutive pairs with two concurrent
workers, as instructed. Each pair takes about 10 hours 57 minutes. The final
pair finishes at 00:39:15 CEST on September 18; all three packs, the queue
and its wrapper exit zero. The tmux session closes and the GPU has no
compute process. All remote production files match the authoritative local
copies by checksum.

## Protocol, coverage and verification

Centered frozen-current float64 EqProp, wide conductances [0,100], exact-zero
frozen biases, perfect diode, original shared initializer and unchanged Adam
learning rates. T=K is 4/6/8 and epoch budgets are 10/30/30 for Conv1/2/3.
Independent Gaussian noise affects only endpoint voltages used for gradient
readout; relaxation, inputs and validation remain noiseless.

No declared baseline read-noise case remains unfinished. The 22 trainings
complete 440 epochs and 1,512,720 optimizer steps. The separate clean
campaign remains paused at 202/216; this completion does not resume it.
Conv3 beta 10 retains its seed-1 confirmation failure and T8 free-state
residual caveat. These runs cannot establish qualification across seeds.
The older Conv3 beta-100 accuracy is not used as a beta-10 clean control.

Akib and Nom exited successfully at 16:29:34 and 16:33:26 CEST respectively.
Their completed production files match the authoritative local copies by
remote checksum comparison. All 22 source and collected bundles pass full revalidation:
canonical artifacts, exact scientific configurations, source identity,
initializer/cohort/order matching, complete epochs and expected noise draws,
finite float64 checkpoints, zero biases, conductance bounds and PT/NPZ equality.
All 39 semantic smokes and six timing runs also validate. No full training
failed, was excluded or replaced. The first Fifi wrapper failed before
training because of an embedded-Python quoting error; its script, log and
exit receipt are retained, and the corrected wrapper uses the unchanged
scientific source/configuration.

Settled usage is **43.294845 physical GPU-hours out of 60**: Fifi 32.902291,
local 9.032269, Nom 0.712355 and Akib 0.647931. Concurrent workers on one
GPU are counted once. Completion precedes the September 18, 15:47:43 CEST
wall deadline. All 22 canonical manifests match the frozen source identity
and their recorded Python versions. PyTorch/CUDA versions are audited at
host level; they are not independently pinned by each canonical manifest.

- [Best/final metrics and per-run status](baseline_read_noise_run_status_20260916.csv)
- [Collection and revalidation proofs](provenance/baseline_read_noise_collection_20260916.json)
- [Coverage, terminal receipts and budget closeout](provenance/baseline_read_noise_closeout_20260918.json)
- [Comparison figure: PNG](read_noise_all_schemes_validation_20260916.png) · [PDF](read_noise_all_schemes_validation_20260916.pdf)
- [Completed ours/legacy study](read_noise_sweep_results.md)
- [Launch plan and exact parameters](../docs/eqprop_baseline_read_noise_launch_plan_20260916.md)
- [Source, outputs, logs and receipts](../results/eqprop-read-noise-seed0-baseline-20260916-v1/)

The comparison figure includes all 50 collected, validated noisy trainings:
30 from the completed ours/legacy study and 20 new baseline cases. Lines
connect measurements only within one noise-stream group. These are
single-seed validation diagnostics, not official-test paper accuracies.
