# X-006 — Choose a common relative output-displacement target

Filip selected this direction on September 24, 2026 and clarified that the
reference should be the **output/readout layer**. The immediate task is to
review existing values against performance, not launch another training sweep.
This review covers ordinary-MNIST Conv3 centered frozen-current EqProp, not
the separate CIFAR BPTT evidence.

Subsequent decision on the same day: Filip selected **0.8, 1.2, 1.6, 2.0 at
initialization**, ten epochs on Conv1/2/3 and sigma5e-4, and stated the
hypothesis that ours should outperform legacy. This is now assigned in
[exp-007](../series/004-relative-output-training/experiments/exp-007-initial-relative-output-training.md)
and [H-006](../hypotheses/H-006-relative-output-ours-versus-legacy.md).
The 10–20 discussion below is the preceding evidence review, not the selected
grid or an active proposal to launch extra cases.

## Working definition

Use the physical 20-component output-state voltage tensor, with

\[
r_{\rm out}=\frac{D_{\rm out}}{F_{\rm out}},\qquad
D_{\rm out}=\operatorname{RMS}((v_+-v_-)/2),\qquad
F_{\rm out}=\operatorname{RMS}(v_{\rm free}).
\]

Pool each squared RMS with its element count over the common 576-example
cohort, then take the ratio. This is not the mean of batch ratios, a ratio
of median RMS values, or displacement of the ten paired prediction scores.
Batch-ratio quantiles remain available as a spread diagnostic. A ratio of
1 means a centered displacement equal to the free-state output RMS; 10
means ten times that RMS, not 10 percent.

This definition separates the nudge response from continued free relaxation.
It differs from the older signed-phase average relative to the zero-nudge
endpoint in the August tables. It also differs from D/P in
[X-004](X-004-relative-read-noise.md), where P is the pooled RMS of the two
nudged endpoints. Writing their midpoint as m gives P²=RMS(m)²+D², so D/P
is bounded by 1 and compresses very large excursions. D/P is useful for
relative read-noise SNR; D/F retains the size of the excursion relative to
the original operating point. Neither scalar guarantees gradient fidelity.

## Existing performance and output displacement

All rows below are final epoch-30 checkpoints measured at their own training
beta, with T=K=8. Training uses independent additive endpoint noise; validation
is clean. Each is one seed. Displacement is measured in a clean physical
replay of the noisy-trained weights, consistent with that noise placement.

| Training family | Scheme | Training sigma | Injected beta | Output D/F at epoch 30 | Final validation |
|---|---|---:|---:|---:|---:|
| p90 | Baseline | 3e-4 | 404.141 | 0.289 | 97.56% |
| p90 | Ours | 3e-4 | 5.26876 | 0.997 | 97.48% |
| p90 | Legacy | 3e-4 | 4.42250 | 2221 | 96.26% |
| p90 | Baseline | 5e-4 | 404.141 | 0.321 | 97.40% |
| p90 | Ours | 5e-4 | 5.26876 | 6.649 | 97.12% |
| p90 | Legacy | 5e-4 | 4.42250 | 3335 | 95.72% |
| p99 | Ours | 5e-4 | 0.987334 | 0.806 | 96.04% |
| p99 | Legacy | 5e-4 | 0.1 | 7.950 | 96.86% |

p90 and p99 refer to the original .90/.99 calibration cosine thresholds.
p90 training here used A100; p99 used V100. The p90 replays used Akib RTX3080
and p99 replays used Fifi RTX5090. These remain distinct histories/environments,
not a matched causal test of beta. Source learning-rate vectors and hashes
are retained in the joined tables.

The p90 ours/sigma5e-4 replay has known residual failures and a clean readout
cosine of -0.293 at T8. Its D/F=6.649 is not a qualified good-gradient target.
Baseline retains known free-phase residual flags in hidden layers. All of
these points remain visible rather than being removed for a cleaner trend.

The p99 output cosines are approximately one for both schemes, despite their
tenfold difference in D/F. Legacy's final accuracy is higher by 0.82 points.
Thus the data do not justify declaring output D/F>1 intrinsically too large.
Conversely, the p90 legacy excursions in the thousands accompany poor clean
readout alignment (~.08–.10), so that range is a poor starting target.

For clean p90 training at the same scheme-specific betas, final validation
was 97.58% / 98.50% / 98.72% for baseline / ours / legacy on A100. Their
output displacement must come from their own clean checkpoints. The noisy
checkpoint ratios above must not be assigned to those accuracies. The active
[exp-006](../series/003-clean-beta-response/experiments/exp-006-rms-and-cosine-beta.md)
will supply the complete clean displacement surface separately.

![Output displacement versus performance](../../../results/conv3-clean-eqprop-rms-beta-20260924-v1/normalization_review/normalized_displacement_vs_accuracy.png)

## Initialization is a different calibration point

The validated absolute-output-RMS-one calibration has these initial free
output RMS values and centered ratios:

| Scheme | Free output RMS | Injected beta at absolute D approximately 1 | Initial D/F |
|---|---:|---:|---:|
| Baseline | 0.0060004 | 88.7 | 166.705 |
| Ours | 0.0413063 | 1.385 | 24.205 |
| Legacy | 0.3840264 | 0.02173 | 2.604 |

So equal absolute displacement did not mean equal relative displacement.
The small baseline output norm also makes relative ratios large; this alone
does not establish a nonlinear or inaccurate gradient.

Linear projections from these measured initial references put the p90
training points near D/F=760 / 92.1 / 530, and the p99 points near 17.3 for
ours and 12.0 for legacy. These are explicitly **initial linear estimates**,
not exact measurements at every listed training beta. They show why the
epoch-30 ratios cannot simply be reused as initialization targets. Fixed
beta does not maintain a fixed normalized displacement while weights learn.

## Candidate decision

If the policy is to **match at initialization and then hold beta fixed**,
an initial ratio around **10–20** is a reasonable provisional range to examine:
it is close to the initial scales of the completed p99 runs. It is not an
established optimum, and their accuracy does not validate this rule for
baseline. Ratio 1 is a useful smaller diagnostic reference, not an automatically
safer training choice.

First-order beta estimates are:

| Target initial output D/F | Baseline beta | Ours beta | Legacy beta |
|---:|---:|---:|---:|
| 1 | 0.5321 | 0.05722 | 0.008346 |
| 10 | 5.321 | 0.5722 | 0.08346 |
| 20 | 10.64 | 1.144 | 0.1669 |

These use beta_new = beta_reference * target / measured_reference_ratio.
They are predictions to check against the existing beta-response surface,
not newly measured roots or training-qualified settings. No target is frozen.

The noise model matters to this choice. At target 10, absolute output D would
be about .060 / .413 / 3.84 for baseline / ours / legacy. Thus matching D/F
does not match SNR under the historical fixed absolute read noise. In
particular, this reduces baseline beta roughly 76-fold from its p90 setting.
Those historical accuracy results cannot certify performance under the
new relative-read model or a noisy-nudging model that has not been tested.

If the intention is instead to keep D/F matched **throughout training**, an
adaptive beta policy is needed. That is a distinct, currently unassigned
intervention; the observed endpoint drift makes it scientifically material.
There is no evidence yet for one universal optimal output ratio. The next
decision is calibration time and noise model, then the common numerical
target; hidden-layer signal and gradient fidelity remain diagnostic checks.
No additional T/K experiments are proposed here.

## Evidence and reproducibility

This is a CPU-only reaggregation of existing local artifacts: 11 training
bundles, eight T8/K8 final-checkpoint replay bundles and three initialization
calibration bundles all validate. The 32 joined layer rows share the same
576-example replay cohort. Source checkpoint/config hashes match the declared
configs, all parent metrics contain 30 consecutive epochs, and calibration
summary RMS values agree with correctly pooled raw state records. No model
was built, no training or GPU replay was launched, and no official test read
or optimizer step occurred. This scoped validation is not completion of the
campaign's broader exp-003 audit.

- [Joined output/layer values and performance](../../../results/conv3-clean-eqprop-rms-beta-20260924-v1/normalization_review/final_phase_vs_performance.csv).
- [Training outcomes, exact parents and LR vectors](../../../results/conv3-clean-eqprop-rms-beta-20260924-v1/normalization_review/training_outcomes.csv).
- [Initial projections](../../../results/conv3-clean-eqprop-rms-beta-20260924-v1/normalization_review/initial_output_linear_estimates.csv)
  and [candidate beta estimates](../../../results/conv3-clean-eqprop-rms-beta-20260924-v1/normalization_review/candidate_output_betas.csv).
- [Source hashes and validation](../../../results/conv3-clean-eqprop-rms-beta-20260924-v1/normalization_review/validation.json).
- [Reproducible analyzer](../../../experiments/review_conv3_normalized_displacement.py):
  `python -m experiments.review_conv3_normalized_displacement` in the py312 environment.

Historical source reports: [p90 replay](../../../results/eqprop-conv3-trained-beta-noise-20260922-v1/analysis/report.md),
[p99 training](../../../results/eqprop-conv3-p99-read-noise-5em4-20260922-v1/production-02-summary.md),
and [absolute RMS-one calibration](../../../docs/conv3_unit_output_displacement_beta_20260922.md).
