# Single-seed read-noise results: September 14–15 overnight window

This report covers the first overnight window. On September 15, Filip
authorized the remaining 18 cases on local, Akib, and Nom; see the
[continuation plan](../docs/eqprop_read_noise_continuation_plan_20260915.md)
and [current tracker](read_noise_run_status_20260914.md) for that later work.

Updated September 15 after final collection. **All 12 scheduled runs are
complete, collected, and validated.** The last pack finished at **05:52:58
CEST**, more than two hours before the user's 08:00 deadline. All four GPUs
are released. The declared sweep contains 30 cells: **12 complete, 0 running,
18 held for a later window**. No production run failed, was retried, or timed out.

These are ordinary-MNIST **validation** measurements, not official-test
accuracy. Every result below is a full 10- or 30-epoch training. Short timing
runs and one-batch smokes do not count toward training coverage.

## Measurements

| Architecture / scheme | Read-noise sigma | Best validation (%) | Final validation (%) | Final loss vs clean (pp) |
|---|---:|---:|---:|---:|
| Conv1 ours | 1e-5 | 96.44 | 96.34 | 0.00 |
| Conv1 ours | 3e-5 | 96.44 | 96.30 | 0.04 |
| Conv1 ours | 1e-4 | 96.40 | 96.34 | 0.00 |
| Conv1 ours | 5e-4 | 96.50 | 96.28 | 0.06 |
| Conv1 legacy | 1e-5 | 96.52 | 96.40 | 0.04 |
| Conv1 legacy | 1e-4 | 96.56 | 96.40 | 0.04 |
| Conv1 legacy | 5e-4 | 96.48 | 96.38 | 0.06 |
| Conv2 ours | 1e-4 | 98.02 | 98.00 | 0.02 |
| Conv2 legacy | 1e-4 | 96.64 | 96.52 | 1.54 |
| Conv3 ours | 1e-5 | 98.38 | 98.26 | 0.38 |
| Conv3 ours | 1e-4 | 97.62 | 97.62 | 1.02 |
| Conv3 ours | 5e-4 | 96.78 | 96.60 | 2.04 |

Positive losses mean lower accuracy than the corresponding collected clean seed-0 result.
The clean best/final references are Conv1 ours 96.42/96.34%, Conv1 legacy
96.48/96.44%, Conv2 ours 98.10/98.02%, Conv2 legacy 98.30/98.06%, and Conv3
ours 98.72/98.64%. Existing clean results are references; no new sigma-0
training was launched.

[Figure: PNG](read_noise_validation_20260915.png) ·
[Figure: PDF](read_noise_validation_20260915.pdf) ·
[Per-run results and remaining cases](read_noise_run_status_20260914.md) ·
[CSV](read_noise_run_status_20260914.csv)

## Interpretation and limits

- The seven completed Conv1 cases are close to their clean references: all
  final changes are between 0 and -.06 percentage points, or at most three
  fewer correct classifications out of 5,000. This single-seed evidence does not establish
  statistical equivalence or seed-to-seed uncertainty.
- The clean references were collected previously on Jean Zay under
  PyTorch 2.5.0. Initial parameters and complete split/order records match,
  but the noisy runs use different GPU/software environments. No fresh
  same-host sigma-0 controls were run, as requested; small clean-relative
  differences cannot be attributed uniquely to noise in this comparison.
- At Conv2 sigma 1e-4, ours loses .02 points and legacy loses 1.54 points.
  The two cases use matched initialization, complete minibatch order, noise
  seed, and the same Nom host. This supports better observed robustness for
  ours **under the inherited beta/LR/T/K contracts** at this one noise level;
  it does not isolate amplification from the different beta choices.
- Conv3 ours has final losses of .38, 1.02, and 2.04 points at 1e-5, 1e-4,
  and 5e-4, respectively. Conv3 legacy is not yet run in this sweep, so these
  measurements do not establish its relative robustness.
- Conv3 ours retains the explicitly recorded clean-gradient qualification
  exception of the inherited beta-3 training contract. These results are
  diagnostic robustness evidence, not a claim that this beta has passed the
  revised broader gradient-selection audit.
- A CUDA sample audit found identical standard-normal arrays on the three
  5090 hosts, but different arrays on Nom's 3090/software combination despite
  the same seed. Within-sigma ours/legacy pairs share a host. Conv1 ours at
  3e-5 on Nom is therefore an independent-noise-realization point relative to
  the 5090 curve. The figure marks this distinction and joins points only
  within an environment group.

## Frozen protocol and provenance

Wide [0,100] perfect-diode networks, exact-zero frozen biases, centered
frozen-current float64 EqProp, model/shuffle/split seed 0, Adam with the exact
parent learning-rate vectors, ordinary MNIST 55,000/5,000 train/validation,
batch size 16, and maximum-validation checkpoint selection. Epoch budgets
are 10/30/30 and T/K is 4/4, 6/6, 8/8. Injected beta anchors are:

| Architecture | Ours | Legacy |
|---|---:|---:|
| Conv1 | 30 | 3 |
| Conv2 | 10 | .03 |
| Conv3 | 3 | .001 |

The prepared grid is sigma `1e-5, 3e-5, 1e-4, 3e-4, 5e-4`, seed 0, ours and
legacy only. Baseline and bounded conditions are excluded. Independent
Gaussian voltage noise with dedicated seed 2026081601 affects EqProp endpoint
gradient readout only; inputs, relaxation, and validation remain noiseless.
The sigma values are simulator voltage units, not calibrated hardware noise.
Official-test access is disabled. The clean campaign remains paused.

The immutable production snapshot is `source-v2`, SHA-256
`4055090e16606583c276b06c3aac59396e87dcedf58b8cde9be94c5824948fb1`.
Every collected run passes canonical validation, full config/source identity,
initializer and full cohort/order matching, expected endpoint-noise draw
count, complete finite histories, successful exit receipts, float64 finite
checkpoints, exact-zero biases, projection bounds, and PT/NPZ equality.
All four completed packs are checksum-identical to their remote copies,
including terminal receipts. No completed run read the official test split.

- [Execution plan](../docs/eqprop_read_noise_overnight_plan_20260914.md)
- [Raw source, runs, checks, and monitoring receipts](../results/eqprop-read-noise-seed0-ours-legacy-20260914-v1/)
- [Collected full bundles](bundles/read_noise_20260914/)
- [Per-run collection proofs](provenance/read_noise_collection_20260914.json)
- [Noise-stream audit](../results/eqprop-read-noise-seed0-ours-legacy-20260914-v1/checks/noise_rng_cross_host.json)

## Runtime and final resource accounting

Three workers started concurrently on each host at approximately 23:27 CEST
on September 14. Shorter workers finished first; each pack then stopped after
its final admitted run, without launching more work.

| Host | Final pack completion, September 15 CEST | Production GPU-hours | Checks GPU-hours | Total GPU-hours |
|---|---|---:|---:|---:|
| Trex | 05:52:58 | 6.423990 | .050023 | 6.474013 |
| Fifi | 04:37:10 | 5.164150 | .040576 | 5.204726 |
| Loulou | 04:36:39 | 5.155258 | .040576 | 5.195834 |
| Nom-cool-1 | 04:41:04 | 5.223038 | .045024 | 5.268062 |
| **Total** | | **21.966437** | **.176199** | **22.142636** |

This is physical GPU time, counting each shared host interval once. It includes
both concurrent timing rounds and the admitted CUDA smokes; the brief CUDA
noise-stream probes overlap production. CPU smokes, data transfers, idle gaps,
and unrelated studies are excluded. Usage is below the separate **36 GPU-hour**
noise budget. The earlier clean campaign's budget is unchanged.

The deadline-process tests pass, including termination of a real child and
grandchild. All 12 exact local smokes, 12 exact GPU smokes, and 24 short CUDA
timing diagnostics succeeded. These checks are excluded from full-training
coverage. All production workers and launchers exited zero, and live process
and GPU checks confirmed release after completion.

[GPU accounting and source receipts](provenance/read_noise_gpu_budget_20260915.json) ·
[Remote/local checksum checks](../results/eqprop-read-noise-seed0-ours-legacy-20260914-v1/checks/completed_pack_remote_copy_checks.json)

## Coverage still missing

The remaining 18 cells are:

| Architecture | Remaining cases | Count |
|---|---|---:|
| Conv1 | Legacy at 3e-5; ours and legacy at 3e-4 | 3 |
| Conv2 | Ours and legacy at 1e-5, 3e-5, 3e-4, 5e-4 | 8 |
| Conv3 | Ours at 3e-5 and 3e-4; legacy at all five sigmas | 7 |

These 18 were held at the overnight handoff. Their subsequent September 15
continuation is recorded separately in the linked plan and current tracker.
