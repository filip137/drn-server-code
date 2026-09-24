# Free-to-nudged phase displacement across Conv1, Conv2 and Conv3

Completed 2026-09-16. All **9/9 architecture/scheme cases** are collected and
validated locally: baseline, ours and legacy, at initialization and the saved
BPTT best-validation checkpoint. These are clean mechanism measurements on
seed 0, not training or official-test accuracy results.

The main qualification is visible in the measurements: for trained Conv3
baseline, first-layer RMS displacement from the post-T free state is
**0.000102807**, whereas the response relative to a matched zero-nudge
endpoint is only **4.57975e-07**—a factor of **224.5**.
Thus continued relaxation dominates that raw free-to-nudged difference at T=8.
H2 shows the same issue, with a factor of **33.2**.

At the trained Conv3 checkpoints, ours has a larger matched-zero nudging
response than legacy in every layer: approximately
**545× / 321× / 191× / 9.88×**
for H1/H2/H3/output. This ordering does not extend to baseline in every layer:
baseline's H1 response is **3.50×** ours. The declared schemes
use different betas and training contracts; these comparisons do not isolate
the effect of amplification at a common beta or establish read-noise accuracy.

## Contract and beta qualification

| Architecture | T/K | Baseline injected beta | Ours injected beta | Legacy injected beta |
|---|---|---:|---:|---:|
| Conv1 | 4/4 | 100 | 30 | 3 |
| Conv2 | 6/6 | 100 | 10 | .03 |
| Conv3 | 8/8 | **10, unconfirmed candidate** | 3 | .001 |

Conv3 baseline beta10 passes seed0 selection and confirmation seeds0/2, but
fails two initialization comparisons on seed1. It is used here only as the
explicitly labeled seed0 candidate from the completed sweep. No common
three-seed training beta was promoted. T=K=8 and the separate equilibrium
caveat remain as Filip requested. Conv3 ours retains its known clean-gradient
qualification exception. Passing seed0 diagnostic gates below does not add
three-seed qualification to any other scheme.

The same 36 validation batches of 16 (576 examples) are replayed for every
case and both checkpoint roles. Source-index order and input payload hashes
match across all nine cases. These are the September14 selection batches,
not a new confirmation cohort. Models use their original wide [0,100]
weights, Kaiming initializer, input gain, source learning rates, exact-zero
biases, and float64 centered frozen-current EqProp. No optimizer is stepped,
no endpoint noise is injected, and the official test split is never read.

The base beta is injected beta divided by the output-row scale: 1 for
baseline, 4^L for ours and 16^L for legacy, where L is the architecture depth.
Both values and every checkpoint hash are retained in the
[source table](phase_displacement_20260916_sources.csv).

## What is measured

For each hidden and output layer, let F be the common free state after T
steps; P and N the positive/negative endpoints after K steps; and Z the
matched endpoint after K additional steps with zero nudge, starting from F.

- Raw phase displacement: RMS(P−F), RMS(N−F).
- Displacement relative to the zero-nudge control: RMS(P−Z), RMS(N−Z).
- Centered phase span: RMS(P−N), with half-span RMS also saved.
- Relative raw displacement: RMS(P−F)/RMS(F), likewise for N.

RMS is pooled from squared differences and element counts across all 36
batches; it is not the mean of batchwise RMS values. Signed means, RMS,
standard deviations, extrema, active-set transitions and batchwise 5th/50th/
95th percentiles are preserved. Cohort percentiles describe batch spread,
not uncertainty across training seeds. Signed mean voltage can cancel;
free-state RMS is reported separately. The control removes the shared
zero-nudge trajectory from the endpoint comparison, but it does not turn an
under-relaxed free state into an equilibrium-qualified checkpoint.

## Per-layer RMS displacement

All table entries use the **positive nudge**, in simulator voltage units.
H1/H2/H3 are hidden layers; Output is always the final layer. Both signs are
shown in the figures and fully tabulated in the CSV.

### Initialization: relative to the post-T free state

| Architecture | Scheme | H1 | H2 | H3 | Output |
|---|---|---:|---:|---:|---:|
| Conv1 | baseline | 0.010867 | — | — | 1.1357 |
| Conv1 | ours | 0.012058 | — | — | 1.4598 |
| Conv1 | legacy | 0.0014215 | — | — | 0.59419 |
| Conv2 | baseline | 0.00018847 | 0.0034037 | — | 1.5891 |
| Conv2 | ours | 9.9093e-05 | 0.0052758 | — | 2.5446 |
| Conv2 | legacy | 2.6998e-06 | 6.5851e-05 | — | 0.12223 |
| Conv3 | baseline | 5.1579e-07 | 2.2826e-06 | 0.00012006 | 0.11277 |
| Conv3 | ours | 9.8165e-07 | 1.4085e-05 | 0.0022583 | 2.1657 |
| Conv3 | legacy | 1.546e-07 | 3.5933e-07 | 1.2616e-05 | 0.046014 |

### Best checkpoint: relative to the post-T free state

| Architecture | Scheme | H1 | H2 | H3 | Output |
|---|---|---:|---:|---:|---:|
| Conv1 | baseline | 0.0049914 | — | — | 0.22383 |
| Conv1 | ours | 0.0029483 | — | — | 0.18535 |
| Conv1 | legacy | 0.00057258 | — | — | 0.1452 |
| Conv2 | baseline | 7.3131e-05 | 0.0015378 | — | 0.087892 |
| Conv2 | ours | 1.6501e-05 | 0.00074208 | — | 0.27502 |
| Conv2 | legacy | 2.0584e-07 | 1.5327e-05 | — | 0.025693 |
| Conv3 | baseline | 0.00010281 | 8.1082e-05 | 6.7791e-05 | 0.0020447 |
| Conv3 | ours | 1.3085e-07 | 2.0735e-06 | 0.00016527 | 0.077818 |
| Conv3 | legacy | 3.386e-10 | 6.4692e-09 | 8.6337e-07 | 0.007874 |

![Absolute free-to-nudged displacement](phase_displacement_absolute_20260916.png)

### Initialization: relative to the matched zero-nudge endpoint

| Architecture | Scheme | H1 | H2 | H3 | Output |
|---|---|---:|---:|---:|---:|
| Conv1 | baseline | 0.010867 | — | — | 1.1357 |
| Conv1 | ours | 0.012058 | — | — | 1.4598 |
| Conv1 | legacy | 0.0014215 | — | — | 0.59419 |
| Conv2 | baseline | 0.00018833 | 0.0034037 | — | 1.5891 |
| Conv2 | ours | 9.8886e-05 | 0.0052758 | — | 2.5446 |
| Conv2 | legacy | 9.0521e-07 | 6.5655e-05 | — | 0.12223 |
| Conv3 | baseline | 4.8692e-07 | 2.2792e-06 | 0.00012006 | 0.11277 |
| Conv3 | ours | 9.803e-07 | 1.4084e-05 | 0.0022583 | 2.1657 |
| Conv3 | legacy | 3.1101e-09 | 5.8174e-08 | 1.2251e-05 | 0.046014 |

### Best checkpoint: relative to the matched zero-nudge endpoint

| Architecture | Scheme | H1 | H2 | H3 | Output |
|---|---|---:|---:|---:|---:|
| Conv1 | baseline | 0.0049914 | — | — | 0.22383 |
| Conv1 | ours | 0.0029484 | — | — | 0.18535 |
| Conv1 | legacy | 0.00057258 | — | — | 0.1452 |
| Conv2 | baseline | 7.313e-05 | 0.0015378 | — | 0.087892 |
| Conv2 | ours | 1.6501e-05 | 0.00074208 | — | 0.27502 |
| Conv2 | legacy | 2.0573e-07 | 1.5327e-05 | — | 0.025693 |
| Conv3 | baseline | 4.5798e-07 | 2.4432e-06 | 4.4993e-05 | 0.0020442 |
| Conv3 | ours | 1.3089e-07 | 2.0735e-06 | 0.00016527 | 0.077818 |
| Conv3 | legacy | 2.4022e-10 | 6.4684e-09 | 8.6337e-07 | 0.007874 |

![Nudging response relative to the zero-nudge endpoint](phase_displacement_matched_zero_20260916.png)

## Relative displacement and voltage scale

![Displacement divided by free-state RMS](phase_displacement_relative_20260916.png)

![Free-state signed mean and RMS](phase_free_voltage_20260916.png)

## Sources, integrity and limits

The best epoch below is copied from each source's stored `best_epoch` field;
the source's indexing convention is retained. Initialization is reconstructed
from the hashed seed0 initializer. No intermediate epoch trajectory is inferred.

| Architecture | Scheme | Stored best epoch | Gradient gate, this seed0 cohort | Equilibrium gate, this seed0 cohort |
|---|---|---:|---|---|
| Conv1 | baseline | 9 | pass | pass |
| Conv1 | ours | 7 | pass | pass |
| Conv1 | legacy | 9 | pass | pass |
| Conv2 | baseline | 30 | pass | pass |
| Conv2 | ours | 20 | pass | pass |
| Conv2 | legacy | 24 | pass | pass |
| Conv3 | baseline | 24 | pass | fail |
| Conv3 | ours | 30 | fail | pass |
| Conv3 | legacy | 27 | pass | pass |

Coverage: **648 checkpoint/batch replays, 1,944 gradient comparisons, 9,720
individual-layer displacement rows and 270 pooled layer/context rows**.
All nine formal bundles and the smoke validate. There are no missing cases,
excluded formal runs, operational failures or replacements. Scientific gate
failures are retained as diagnostic outcomes, not discarded runs. Every
source/checkpoint byte guard passes. Added signed-voltage statistics reproduce
all prior shared smoke gradient, residual and displacement measurements exactly.

Execution used one local RTX3090 for **12.13 minutes
(0.202105/1 GPU-hours)**, including smoke and validation overhead. No
5090 or remote GPU was allocated. No training or beta promotion occurred.
Scheme-specific betas, input gains, learning contracts and trained checkpoint
histories limit causal attribution; one model seed does not establish
training-seed robustness. These endpoint-voltage diagnostics do not measure
gradient signal-to-noise ratio or predict noisy-training accuracy by themselves.

Files:

- [All pooled measurements, both nudge signs and all references](phase_displacement_20260916_layer_displacement_summary.csv).
- [Sources and checkpoint hashes](phase_displacement_20260916_sources.csv).
- [Structured summary and provenance](phase_displacement_20260916_summary.json).
- PDF figures: [absolute](phase_displacement_absolute_20260916.pdf),
  [relative](phase_displacement_relative_20260916.pdf),
  [matched zero-nudge](phase_displacement_matched_zero_20260916.pdf),
  [free voltage](phase_free_voltage_20260916.pdf).
- [Per-batch layer measurements](../results/eqprop-phase-displacement-conv123-20260916-v1/analysis/layer_batch_displacement.csv).
- [Full canonical bundles and execution record](../results/eqprop-phase-displacement-conv123-20260916-v1/).
- [Execution plan and scientific contract](../docs/eqprop_phase_displacement_plan_20260916.md).
- [Conv3 beta-sweep result and failed seed1 confirmation](conv3_baseline_beta_tk8_20260916.md).
