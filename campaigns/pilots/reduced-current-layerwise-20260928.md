# Reduced-current amplification: initialization layer profiles

## Question and frozen comparison

Filip authorized implementation on September 28 after choosing all five schemes
and the existing output targets. Before training, test whether voltage/current
settings **1/0.5 and 1/0.25** preserve hidden-layer phase signals and noisy gradient
alignment. The prospective prediction was weaker upstream EqProp signals with
reduced feedback; the review below is based on the completed measurements.

Six new seed-0 contexts: two settings × Conv1/2/3. Reuse nine exp013 stage-A
contexts for baseline 1/1, ours 4/1 and legacy 4/0.25. Match output D/F to 1/4/6,
respectively, within 1% using one initial measurement and at most two corrections.
Internal beta = injected beta / (voltage/current)^depth. All new contexts matched
on their first correction. No matching failures, numerical failures or retries.

The numerical contract is the archived exp013 float64, centered frozen-current
EqProp replay, with each scheme's own finite-K BPTT reference; T=K=4/6/8, input
gains 40/100/360, explicit perfect-diode dictionaries, weights [0,100], identical
initial parameter tensors within each depth and zero biases. No optimizer,
training, accuracy evaluation or official-test read. Source/config/initializer
and cohort checks passed; runtime and analyzer archives are unchanged.

D = RMS((v_plus − v_minus)/2); P = pooled RMS of both clean nudged endpoints.
The displacement curves pool squared measurements over all 576 validation
examples and nodes before computing D/P. Absolute D is provided alongside it.
Cosines use the same first 192 examples for clean and noisy conditions, with
eta = 0, 1e-6, 1e-5, 1e-4, 3e-4. Each nonzero eta uses acquisition seeds
111000000, 112000000 and 113000000, matched across all schemes. Noise is
v_read = v + eta*abs(v)*xi in copied noninput endpoint states only; force,
relaxation and input stay clean. Batch cosines are averaged within each draw,
then across draws. Bands are draw-mean ranges, not confidence intervals.

## Evidence and figures

**Complete and reviewed:** 6/6 new contexts plus 9/9 reused references validated;
45 pooled layer measurements and 450 gradient-summary cells (two references).
All 648 distinct selected Gaussian acquisition identities match across schemes.
No undefined cosines. Six separate GPU smoke cases passed; 22 unit tests passed.
Six production case bundles and their three parent lanes passed canonical
reporting validation; the analyzer also validates selected reference artifacts,
identity, coverage, output matching and unchanged-parameter/force guards.

Results root (including all smoke, source, transport and production artifacts):
`results/conv123-reduced-current-layerwise-20260928-v1/`.

- [Normalized displacement versus layer](../../results/conv123-reduced-current-layerwise-20260928-v1/analysis/D_over_P_combined.jpg)
- [Absolute displacement versus layer](../../results/conv123-reduced-current-layerwise-20260928-v1/analysis/D_combined.jpg)
- [Free-state RMS voltage versus layer](../../results/conv123-reduced-current-layerwise-20260928-v1/analysis/F_combined.jpg)
- [Nudged endpoint RMS voltage versus layer](../../results/conv123-reduced-current-layerwise-20260928-v1/analysis/P_combined.jpg)
- [Cosine to each scheme's BPTT reference](../../results/conv123-reduced-current-layerwise-20260928-v1/analysis/cosine_finite_K_BPTT_combined.jpg)
- [Cosine to clean EqProp](../../results/conv123-reduced-current-layerwise-20260928-v1/analysis/cosine_clean_EqProp_combined.jpg)

Individual architecture JPGs accompany these combined figures. Data and checks:
`analysis/state_profiles.csv`, `gradient_profiles.csv`, `reference_provenance.json`
and `validation.json`. Both main combined figures were visually inspected.

The voltage plots were added at Filip's request from the same validated state
measurements, with no new simulation. They show RMS magnitude, not mean absolute
voltage: F = sqrt(mean(v_free²)) and P = sqrt(mean((v_plus² + v_minus²)/2)),
pooled over all nodes and 576 examples per layer. Axes are logarithmic and
share voltage limits across architectures within each figure. Both combined
voltage figures were visually inspected; all 15 source contexts revalidated.

## Review and decision

Reducing current amplification reduces D/P in **every convolutional layer** at
these matched output targets. Ratios relative to baseline 1/1 are:

| Architecture / layer | D/P at 1/0.5 ÷ baseline | D/P at 1/0.25 ÷ baseline |
|---|---:|---:|
| Conv1 / convolution 1 | 0.5174 | 0.2634 |
| Conv2 / convolution 1 | 0.3750 | 0.1279 |
| Conv2 / convolution 2 | 0.4867 | 0.2375 |
| Conv3 / convolution 1 | 0.4629 | 0.1494 |
| Conv3 / convolution 2 | 0.3736 | 0.1214 |
| Conv3 / convolution 3 | 0.4928 | 0.2411 |

Clean EqProp/BPTT alignment is near one for all five schemes. With relative read
noise, the new settings lose useful gradient alignment sooner. At eta=1e-4,
Conv1's convolution cosine is 0.8369 / 0.6234 / 0.3761 for baseline / 1/0.5 /
1/0.25. Conv3's last convolution is 0.9066 / 0.7297 / 0.4587. Conv3's first two
convolutions are already essentially unaligned at that eta for all schemes;
small fluctuations around zero do not establish a useful advantage. Readout
alignment remains near one and does not represent the hidden layers.

The evidence supports the predicted loss of hidden-layer fractional phase
signal and argues against a read-noise robustness benefit for these two new
settings at initialization. It does not establish their trained accuracy or
rule out a clean BPTT benefit. This is one initializer, one validation cohort,
fixed finite-T/K dynamics and four nonzero noise levels. Reference contexts
retain their historical devices; cross-device numerical equivalence was not
qualified. No solver/residual qualification was added or implied.

The voltage profiles show that reducing current gain increases free-state RMS
voltage in every convolutional layer relative to baseline in this pilot. For
Conv3, F across the three convolutions is 6.8267 / 0.10357 / 0.02322 V at
1/1, 10.1268 / 0.24389 / 0.05456 V at 1/0.5, and
13.6501 / 0.46153 / 0.10150 V at 1/0.25. Ours has
2.3928 / 0.04445 / 0.03895 V; its later-layer profile is flatter. Clean
nudged endpoint RMS closely follows free-state RMS in hidden layers, while
nudging substantially raises the readout RMS. Voltage scale alone does not
establish robustness: it must be compared with displacement through D/P.

The later-layer flatness largely reflects compensation by voltage gain A=4.
Across all nine free-state layer measurements in Conv1/2/3, legacy F_l equals
4^(l-1) times baseline F_l exactly at the precision of the saved CSV (l=1 for
the first convolution). This agrees with their common AB=1, identical weights,
zero biases and homogeneous perfect-diode constraints. In Conv3, baseline's
Conv2→Conv3 and Conv3→readout RMS ratios are 0.2242 and 0.2584; multiplying
by four gives legacy's 0.8969 and 1.0336. Ours has corresponding ratios
0.8762 and 1.0605, but its AB=4 changes the normalized dynamics, so it is
not a rescaled baseline. The near-flat region starts at Conv2; Conv1→Conv2
still drops sharply. These profiles do not imply voltage preservation for
arbitrary architectures, initializers or trained weights.

The layerwise pilot is finished. The previously agreed six clean BPTT trainings
remain deferred for review: ten epochs, seed 0, Adam with the existing 1/1 rates,
and no additional baseline training. No training was launched here.

## Execution, monitoring and reproduction

All new GPU work ran on Akib's RTX 3080 through the shared run-watch queue;
existing jobs were untouched. The queue owned launch, polling and CPU collection.
Root Codifier/Reviewer owned bounded recovery and scientific review. No second
watcher and no recovery were needed. Job IDs:
`reduced-current-{smoke,production}-conv{1,2,3}-20260928`.
Current terminal receipts are retained in `launch/queue-completion.json`.

The 3 GPU-hour cap included smoke and overhead; production allowances were
20/40/90 minutes plus 30 minutes reserved for smoke. Queue-charged worker time
was 269.958 seconds (0.07499 GPU-hours), including all smoke/production
workers and target validation. Controller-side CPU collection/plotting is separate.
The common deadline was 2026-09-29T15:09:46.840208Z. Peak production CUDA
reservation is recorded per case; smoke peaks were 166/1000/4564 MiB. The
7.5 GiB allocator cap and CPU offloading preserved float64 and batch size 16.

The isolated 614-file snapshot and full hashes are in `source-v1/` and
`source_identity.json`. The admission wrapper verifies every staged file before
running. Remote root:
`/home/filiposana/server_code/results/conv123-reduced-current-layerwise-20260928-v1/`.
Exact argv, environment, target, budgets and collection hooks are in `launch/`;
`checks/unit-tests.json` records the two passing test commands.

Preparation: `python -m experiments.prepare_reduced_current_layerwise`.
Configs: `configs/conv/reduced_current_layerwise_20260928/conv{1,2,3}.json`.
Replay command: `python -m experiments.replay_relative_noise_advantage --config
CONFIG --data-root DATASET --output-root NEW_OUTPUT [--smoke]`.
The smoke explicitly uses one batch per case and separate evidence paths.
Historical/default replay and plotting interfaces remain available unchanged.

Reproduce figures from the collected evidence:

```bash
MPLCONFIGDIR=/tmp/mpl-reduced-current /home/filip/miniconda3/envs/py312/bin/python -m experiments.analyze_reduced_current_layerwise
```

The analyzer requires all 15 contexts and does not publish partial success.
