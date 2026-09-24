# Free/nudged phase displacement across Conv1–3 and amplification schemes

Date: 2026-09-16. Status: **completed, validated and analyzed**.

Ran 13:50–14:02 CEST on local RTX3090; driver1239772 exited0 and the GPU is idle.
All nine cases and the smoke validate: 648 replays, 1,944 gradient comparisons,
9,720 individual-layer rows and 270 pooled layer/context rows. Actual cost:
**0.202105/1 GPU-hours**, including smoke. No failures, exclusions or replacements.
[Report, per-layer tables and figures](../paper_ready_results/phase_displacement_20260916.md).

Conv3 baseline uses beta10,
which passed seed0 selection/confirmation but failed confirmation on seed1;
it is an explicitly unconfirmed candidate. The same-runner smoke reproduces
all prior shared gradient, residual and displacement measurements exactly.
Signed-voltage and unequal-count pooling checks pass. All nine full cases
are complete; only diagnostic statistics were added to the frozen analyzer.
Trained Conv3 baseline H1 displacement is 1.02807e-4 from the post-T free
state versus 4.57975e-7 from the matched zero-nudge endpoint: a factor of
224.5, showing that continued relaxation dominates the raw difference.
Ours exceeds legacy's matched-zero response in all trained Conv3 layers,
but it does not exceed baseline in every layer. Different betas and source
training histories limit causal attribution. No training was resumed.

Filip requests per-layer free/nudged displacement for baseline, ours and legacy
on Conv1, Conv2 and Conv3 after beta selection. This is a clean read-only
mechanism comparison on the same initialization and BPTT best-validation
checkpoints used for the beta checks. It does not launch training.

## Fixed cases

| Architecture | T/K | Baseline injected beta | Ours injected beta | Legacy injected beta |
|---|---|---:|---:|---:|
| Conv1 | 4/4 | 100 | 30 | 3 |
| Conv2 | 6/6 | 100 | 10 | .03 |
| Conv3 | 8/8 | 10, failed seed1 confirmation | 3 | .001 |

Use the inherited ours/legacy settings and the baseline Conv3 nominated beta,
retaining the sweep's confirmation outcome rather than silently promoting a
failed candidate. Record injected and base beta; base beta is injected beta
divided by the scheme's output-row scale (1, 4^L, 16^L). Each architecture's
original input gain, Adam rates, initialization and [0,100] weights remain
unchanged. Biases are exact zero, float64 centered frozen-current EqProp,
with no endpoint read noise. Preserve the Conv3-ours clean-gradient exception
and baseline T8 equilibrium caveat. Measurements remain usable even where a
gradient or equilibrium qualification gate fails; label those outcomes.

Nine cases use seed0, initialization and the BPTT best checkpoint, each on the
same36 batches of16 from the September14 selection cohort. Keep source-index
order and payload hashes exact. Checkpoint epochs are recorded from source
metadata; no intermediate checkpoints or training trajectories are invented.
Official-test data are not read and no optimizer is constructed or stepped.

## Measurements

For every hidden and output layer, both positive and negative nudged endpoints:

- Compare with the common post-T free state: RMS(V_nudged - V_free), maximum
  absolute displacement, signed mean displacement, and RMS displacement
  divided by free-state RMS. Preserve small/zero denominator diagnostics.
- Also compare with the matched zero-nudge endpoint after K extra steps.
  This separates the nudging response from further free relaxation, which
  is particularly relevant to the retained baseline Conv3 T8 caveat.
- Include the positive-minus-negative endpoint span and its half-span RMS.
  These are voltage differences, not a gradient/noise SNR measurement.
- Report signed mean, RMS, standard deviation, min/max and element count for
  reference and nudged voltages, plus perfect-diode active-set changes.
- Pool sums and squared sums by element count across batches; do not average
  batch RMS values or relative ratios as the pooled result. Also report
  batchwise median and 5th/95th percentiles as cohort spread.

The existing replay already retains per-layer displacement for both
references and the endpoint span. Add only signed sums, current squared sums
and voltage extrema to the shared measurement helper. No solver, estimator,
nudging or training dynamics changes. The focused test checks signed
cancellation versus nonzero RMS. A same-runner baseline Conv3 smoke must
reproduce the previous beta-sweep smoke's shared numerical measurements
before the nine-case wave starts.

## Execution and deliverables

- Local/Main RTX3090, allocated only after the beta sweep completes and a
  fresh authorized-host inventory/capacity check. One sequential GPU process.
- Separate **one physical GPU-hour** cap including smoke and failures;
  expected10–20 minutes. Each case has a600-second cap, driver3,600 seconds.
- Nine full cases:648 checkpoint/batch replays,1,944 gradient comparisons
  retained as diagnostics, and9,720 individual-layer displacement rows
  (five phase/reference contexts). Aggregate-across-layer rows are excluded
  from layerwise pooling. Expected270 pooled layer/context rows.
- Result root: `results/eqprop-phase-displacement-conv123-20260916-v1/`.
  Configs: `configs/conv/eqprop_phase_displacement_20260916_v1/`.
  `python -m experiments.run_phase_displacement prepare`, then `run` using
  py312; maintain execution.json, logs, canonical bundles and GPU accounting.
- Validate all nine local bundles and coverage, plot with matplotlib, and
  publish per-layer CSV/JSON, free-state-relative and absolute displacement
  figures, companion voltage statistics and a short report in
  `paper_ready_results/`. Curate the result in the experimental manifest.

Differences reflect the declared scheme/beta/learning contracts together;
they do not isolate amplification at a common beta. Best checkpoints have
scheme-specific training histories. These deterministic, single-seed cohort
statistics do not provide training-seed uncertainty or paper accuracy.

[Conv3 beta sweep](eqprop_conv3_baseline_beta_tk8_plan_20260916.md) ·
[Current read-noise settings and evidence](../paper_ready_results/read_noise_sweep_results.md).
