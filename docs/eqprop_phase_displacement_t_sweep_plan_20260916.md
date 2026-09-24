# T-only displacement check for the affected Conv cases

Date: 2026-09-16. Status: **completed, validated and analyzed**.

Completed14:58 CEST; recovery driver1264547 exited0 and the GPU is idle.
All15 included configurations and both smokes validate:900 included replays,
3,420 gradient comparisons,20,520 individual-layer rows and570 pooled groups.
Cost **.460455/1 GPU-hours**, including the failed metadata attempt and recovery.
[Results, per-layer tables and figures](../paper_ready_results/phase_t_sweep_20260916.md).

The smallest tested T with drift <=1% of the nudging response in every layer
and both signs, sustained at larger tested T values, is:

| Case | Pooled cohort | Every individual batch |
|---|---:|---:|
| Conv3 baseline initialization | 12 | 12 |
| Conv3 baseline best BPTT | 16 | 24 |
| Conv3 legacy initialization | 16 | 16 |
| Conv3 legacy best BPTT | 10 | 12 |
| Conv2 legacy initialization | 10 | 10 |

Controlled-response RMS varies by less than.001% throughout the grids relative
to the largest T. T12 already passes the baseline residual gate, yet trained
H1 drift is1.067 times the response; T16 gives.587% pooled but1.30% worst-batch
drift. The direct relative-drift criterion is therefore stricter than the
existing residual gate for these weak signals. This does not promote beta10
across seeds or change training settings. The historical launch/recovery
record and frozen pre-execution plan remain below and in the result root.

Started14:27 CEST on local RTX3090, driver PID1260692. Eight focused
iteration/measurement checks passed. The same-runner smoke reproduced all
previous shared measurements exactly before the full wave started.
Execution deadline15:27 CEST; one sequential worker, observed11.6GiB GPU use.

Recovery14:36 CEST: the initialization-only Conv2 legacy attempt completed36
replays then failed because inherited completion metadata expected72. The
failed `runs/conv2_legacy_seed0_T6` is preserved. Five Conv2 configs have corrected
counts36/108 in `recovery_v2/`, with no scientific changes; the replacement
run is `runs/conv2_legacy_seed0_T6_v2`. Driver1264547 retains both completed
Conv3 cases. A new coverage-count regression test and the recovery smoke pass.
The original deadline and cumulative one-GPU-hour cap include the failed attempt.

Filip requests varying T only for cases affected by residual relaxation in
the completed [nine-case displacement study](../paper_ready_results/phase_displacement_20260916.md).
This read-only diagnostic changes the replay free-phase length; it does not
change any training contract, beta qualification, checkpoint or paused run.

## Fixed cases

Select architecture/scheme/checkpoint-role pairs where the previous positive
nudge raw/control RMS ratio differed from one by at least 1% in a hidden
layer. This is a descriptive inclusion rule, not a new scientific gate.

| Model/scheme | Checkpoints | T grid | Fixed K | Fixed injected beta |
|---|---|---|---:|---:|
| Conv3 baseline | initialization and best BPTT | 8,10,12,16,24 | 8 | 10, unconfirmed candidate |
| Conv3 legacy | initialization and best BPTT | 8,10,12,16,24 | 8 | .001 |
| Conv2 legacy | initialization only | 6,8,10,12,16 | 6 | .03 |

All state layers, both nudge signs and the same36 batches of16 are measured.
Keep seed0, source checkpoints/initializers, beta/base-beta convention,
preprocessing, input gain, wide [0,100] weights, exact-zero biases and float64
centered frozen-current EqProp unchanged. Source Adam learning rates remain
checkpoint provenance; no optimizer is constructed or stepped. Reuse the
previous selection cohort and hashes, without treating it as fresh validation
of a tuned beta. The official MNIST test split stays unread.

The user explicitly authorizes this T sensitivity diagnostic after choosing
T=K=8 for beta selection. It does not silently adopt a longer training T.
Conv3 baseline beta10's seed1 confirmation failure remains in force.

## Measurements and interpretation

Retain the previous five phase/reference contexts: positive and negative
relative to the post-T free state and the matched zero-nudge K-step endpoint,
plus positive-minus-negative span. Add direct per-layer zero-nudge drift,
RMS(Z_K − F_T). This uses already computed states and does not alter dynamics.
Record signed mean, RMS, spread, extrema, active-set transitions, and counts.
Pool sums and squared sums by element count across batches.

For each T and layer report:

- raw/control RMS ratio, for direct comparison with the previous tables;
- direct zero-nudge drift divided by the positive and negative controlled
  nudging-response RMS, which detects drift even if the raw RMS ratio is near1;
- raw displacement, controlled displacement and direct drift in voltage units;
- controlled response relative to the largest tested T, since increasing T can
  change the free operating point and thus the nudging response itself;
- the unchanged equilibrium and gradient gates as separate diagnostic outcomes.

Report the smallest tested T with pooled drift <=1% of the controlled response
in every observed layer and both signs, sustained at larger tested T values.
This 1% reporting threshold is descriptive; it is not a replacement for the
existing gradient/residual gates or a three-seed training qualification.
Preserve batchwise spread and failing comparisons; do not infer a precise
minimum T between grid points or full equilibrium from one finite K-step drift.

## Execution and checks

- Target: local/Main RTX3090, matching the previous study and Filip's local
  preference. Check all authorized hosts and Jean Zay before allocation.
- Budget: one physical GPU-hour, including smoke/failures. Expected25–35min.
  Driver cap3,600sec; smoke180sec, per-case600sec. One sequential GPU process.
- Fifteen full configurations cover900 checkpoint/batch replays and3,420
  gradient comparisons, with20,520 individual-layer displacement rows across
  six contexts and570 pooled layer/context groups. No unaffected Conv1 or
  ours cases are added. Conv2 legacy's unaffected trained role is omitted.
- Repeat the three native-T configurations because direct zero-nudge drift
  was not saved previously. Every shared native-T gradient/residual/displacement
  measurement must reproduce the earlier run exactly. Run a same-runner smoke
  at baseline Conv3 T8 and compare shared rows before admitting the full wave.
- Snapshot the numerical analyzer chain and its measurement-only addition;
  retain the existing pinned scientific runtime. Canonical bundles retain
  config, command, environment/commit, hashes, logs, metrics and source guards.
- Result root: `results/eqprop-phase-displacement-t-sweep-20260916-v1/`.
  Configs: `configs/conv/eqprop_phase_displacement_t_sweep_20260916_v1/`.
  Direct driver: `python -m experiments.run_phase_displacement_t_sweep`.
- Monitor real metrics through completion, validate all local bundles,
  reconcile coverage/cost, make matplotlib figures and publish a per-layer
  report/CSV/JSON in `paper_ready_results/`. Curate the conclusion in the
  experimental manifest and keep the persistent dashboard row current.

Limits: seed0 only, two observed checkpoint roles where affected, no retraining,
no noise injection, and no official-test results. Each scheme keeps its own
beta, so cross-scheme differences do not isolate amplification alone.
