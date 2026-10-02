# Trained Conv3 voltages under finite conductance ratios

## Question and comparison

Filip requested voltage magnitudes in trained finite-wmax/wmin Conv3 versus
the unbounded reference for all three amplification schemes. Default to the
paper collection's clean BPTT/Adam checkpoints; the optional algorithm question
was left open while source discovery proceeded. Compare baseline (1,1), ours
(4,1), and legacy (4,.25), seeds 0/1/2, best-validation checkpoints after
30 training epochs. Finite ratios 10/50/100 have wmin=1e-5 and
wmax=1e-4/5e-4/1e-3. The nominally unbounded reference actually uses [0,100].

Use the original native T=K=8 collection for all 36 cases, not the separate
T24 baseline revision. This controls replay iterations across schemes/ranges.
Describe trained operating points, not a bounds-only causal ablation: bounded
and wide runs have different initialization and learning-rate contracts.
No accuracy claim, new training, or official-test access.

## Measurement and decision rule

Reuse the prior 576-example validation cohort (36 batches of 16), SHA-256
95af3eebd9416ec643b7d698d061b7dac3fe0d52311dd888293609bada629acd.
Native float32, fixed asynchronous T=8, input gain 360, identical resets and
preprocessing. Collect count-weighted RMS, mean absolute voltage, signed mean,
standard deviation, min/max, and exact-zero occupancy in H1/H2/H3/readout.
Summarize each setting across three seeds, showing individual-seed spread and
paired bounded/wide ratios. Directional conclusions require agreement across
all three seeds; otherwise report mixed evidence. No significance test.
Only best checkpoints are replayed; no epoch trajectory or initialization
delta is inferred. Hash source configs/checkpoints before and after and verify
exact restored and unchanged parameters.

## Execution and monitoring handoff

One local CPU worker, root owns monitoring through collection and review.
Budget 1800 seconds production plus 180 seconds smoke, expected minutes subject
to same-path smoke timing. All attempts and analysis live under
`results/conv3-bounded-wide-voltage-20260929-v1/`.
Config: `configs/conv/conv3_bounded_wide_voltage_20260929.json`.
Runner: `python -m experiments.replay_conv3_bounded_wide_voltages --config
configs/conv/conv3_bounded_wide_voltage_20260929.json --output-root
results/conv3-bounded-wide-voltage-20260929-v1/production`.
Smoke adds `--smoke` and uses output `smoke/`; one batch per seed-0 case.
Check case/batch progress at least once per minute during this bounded replay.
Stop on nonfinite states, input identity/restore failures, or budget exhaustion.

## Results and review

Complete and reviewed: 36/36 checkpoints, 144 layer profiles, and the full
576-example cohort per checkpoint. Production took 290.14 CPU seconds;
12 one-batch smoke cases took 5.03 seconds. All 37 production and 13 smoke
bundles validated. Exact parameter restoration, unchanged parameter/config/
checkpoint bytes, native float32, finite states, weight bounds, and zero biases
passed throughout. Six existing tests of the shared voltage replay helpers passed.
Both main RMS figures were visually inspected. No training, gradient audit,
new convergence qualification, or official-test evaluation occurred.

The initial default-Python invocation failed before computation because
matplotlib was absent. Using the existing
`/home/filip/miniconda3/envs/py312/bin/python` environment resolved that bootstrap
issue; no scientific setting changed and no partial numerical bundle was produced.
Exec session 73048 completed successfully; no running job or monitoring obligation
remains. All generated artifacts remain under the single declared result root.

Layerwise RMS, mean across seeds 0/1/2 (simulator voltage units):

| Scheme | wmax/wmin | H1 | H2 | H3 | Readout |
|---|---|---:|---:|---:|---:|
| baseline | wide [0,100] | 51.01199 | 20.72958 | 1.39406 | 0.34547 |
| baseline | 10 | 6.65253 | 1.86240 | 1.00302 | 0.54375 |
| baseline | 50 | 18.04865 | 4.09330 | 2.41955 | 0.70006 |
| baseline | 100 | 21.97878 | 4.94391 | 2.81563 | 0.70397 |
| ours | wide [0,100] | 18.82613 | 2.63364 | 0.44004 | 0.44414 |
| ours | 10 | 2.11103 | 0.53972 | 0.85126 | 1.26536 |
| ours | 50 | 5.63395 | 1.24788 | 1.41063 | 1.86039 |
| ours | 100 | 7.83834 | 1.95973 | 1.67650 | 2.22974 |
| legacy | wide [0,100] | 26.63712 | 4.96256 | 0.53488 | 0.44305 |
| legacy | 10 | 5.65822 | 1.96219 | 1.16188 | 1.41238 |
| legacy | 50 | 8.99819 | 3.39219 | 1.06515 | 0.87154 |
| legacy | 100 | 9.73809 | 3.86307 | 1.04952 | 0.83989 |

The finite ratios redistribute voltage magnitude rather than uniformly
reducing it. H1 RMS decreases in every paired seed, scheme and finite range
(0.106–0.559 times its wide reference). H2 also decreases except ours ratio100,
where one seed increases (paired range 0.515–1.169); that case is mixed across
seeds despite its lower average.

H3 RMS increases for every bounded ours/legacy case, by 1.64–5.50x and
1.60–3.10x respectively across paired seeds. Baseline H3 decreases at ratio10
(0.617–0.871x), but increases at ratios50/100 (1.53–2.30x).
Readout RMS increases in every paired bounded case: baseline 1.25–2.54x,
ours 2.33–9.19x, legacy 1.81–5.17x. Ours has particularly strong late-layer
voltage growth despite reduced first-layer voltage.

Magnitude definitions matter: baseline ratio10 has lower H3 RMS but higher
mean absolute H3 voltage (0.569 vs 0.496); the distribution changed, not just
one overall scale factor. Signed means, standard deviations, extrema and
exact-zero occupancies are retained alongside both magnitude measures.
This study supports a descriptive change in the trained voltage profile.
It does not isolate the effect of bounds from initialization and LR choices,
prove equilibrium convergence at T8, or explain accuracy by voltage alone.
The primary comparison uses original native-T8 BPTT checkpoints; the separate
T24 baseline revisions and EqProp checkpoints are outside this replay.

Figures and data:
- [RMS profiles](../../results/conv3-bounded-wide-voltage-20260929-v1/analysis/rms_by_layer.jpg)
- [Paired RMS ratios](../../results/conv3-bounded-wide-voltage-20260929-v1/analysis/rms_bounded_over_wide.jpg)
- [Mean absolute voltage](../../results/conv3-bounded-wide-voltage-20260929-v1/analysis/mean_absolute_by_layer.jpg)
- [Signed mean](../../results/conv3-bounded-wide-voltage-20260929-v1/analysis/signed_mean_by_layer.jpg)
- [Per-checkpoint statistics and source paths/epochs](../../results/conv3-bounded-wide-voltage-20260929-v1/analysis/voltage_profiles.csv)
- [Three-seed summaries](../../results/conv3-bounded-wide-voltage-20260929-v1/analysis/voltage_summary.csv)
- [Paired ratios and epochs](../../results/conv3-bounded-wide-voltage-20260929-v1/analysis/bounded_over_wide.csv)
- [Coverage and validation](../../results/conv3-bounded-wide-voltage-20260929-v1/analysis/validation.json)

Checkpoint epochs range from 10 to 30 according to each source's existing
maximum-validation rule. No intermediate-epoch trajectory was requested or
replayed; no missing epoch measurements are interpolated. Config contains all
36 source paths and hashes. No additional simulation is needed for this request.
