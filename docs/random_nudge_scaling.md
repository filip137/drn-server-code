# Random-sign Monte Carlo scaling

The follow-up [baseline and optimizer experiments](random_nudge_improvements.md)
test whether past measurements can reduce the fixed-eight-probe deficit at 256
states, with four/eight probe budgets and validation-only optimizer selection.

Exploratory, non-canonical follow-up, 12 September 2026. This study checks the
dimension/probe scaling of the estimator and whether the earlier 32-state digits
training result extends to larger recurrent networks. The original outputs are
preserved. Current source starts at `e7d20ec8` on
`codex/hopfield-random-nudge-adjoint`; run receipts record subsequent changes.

## Planned comparisons and monitoring

- Ideal statistical check: normalized random-sign probes, state counts 32 through
  4096, fixed eight probes versus one probe per four states. Record empirical MSE,
  its analytic expectation, and independent voltage-read noise at fixed total
  current amplitude. This is an estimator experiment, not network training.
- Frozen-network gradient check: sizes 32/64/128/256/512, seeds 0/1/2, independent
  probe designs for each example, minibatch prefixes 1/8/32/96. Use exact local
  linear responses to isolate sampling error; label these oracle-derived
  diagnostics separately from physical nudging. Compare fixed eight random-sign
  probes, proportional random-sign probes, and eight unbiased orthogonal probes.
- Physical training: smoke-test the 256-state forward solver and budgets first.
  Then compare exact adjoint, eight random-sign probes, and n/4 random-sign probes
  at 64/128/256 states, three seeds, the existing 1149/288/360 digits split,
  15 epochs, batch 96, learning rate 0.3, fixed skew norm 1, nudge norm 0.01 and
  voltage-read noise 1e-5. Reuse the completed 32-state adjoint/mc8 controls,
  explicitly recording their source. The increasing-width comparison changes
  model capacity; the exact adjoint at each width is its learning control.

Runs use foreground CPU Python exec handles, one BLAS thread per process.
`simulation_results/mc_scaling_smoke256/` is the small launch gate;
`mc_scaling_diagnostics/` contains the statistical diagnostics;
`mc_scaling_digits_n64/`, `mc_scaling_digits_n128/`, and
`mc_scaling_digits_n256_seed{0,1,2}/` contain the training shards. The 256-state
smoke test completed three trajectories in 15.9 seconds with valid checkpoints;
its large-probe path suggests roughly 15–25 minutes per complete seed. The three
256-state seeds run concurrently on separate CPU processes. Each launcher writes
its PID/config/source receipt, status, log and terminal metrics to its directory.
Training heartbeats occur every epoch and trajectory; diagnostic heartbeats occur
every dimension/design case. Expected duration is seconds for the smoke test and
minutes to tens of minutes for complete training, refined from the launch gate.
Verify process identity and first metrics immediately; monitor artifact progress
through completion. Preserve failures and only retry the same scientific config
after diagnosing operational defects. Completion requires declared coverage,
finite metrics, valid residuals, correct phase counts and checkpoint replay.

The common learning rate is inherited from the earlier validation-only screen;
it is not tuned separately for each width or estimator. Physical cost counts
exclude validation, diagnostic Jacobians and the oracle's dense solves. CPU
timings include implementation overhead and are not hardware scaling claims.

A targeted exploratory noise control was added after the fixed-eight-probe
256-state trajectories completed: `mc_scaling_noise_control_n256/` runs mc8,
seeds 0/1/2, with the same 15 epochs/settings and read noise **1e-30**. This is
effectively zero at the state precision while keeping the noise-sampling branches
active, so RNG consumption and random probe patterns match the 1e-5-noise runs.
The control uses one additional CPU process, epoch heartbeats, and the same
completion/checkpoint checks; expected runtime is roughly 8–12 minutes. It remains
separate from the primary width comparison and was not predeclared with it.

The exec session handles used for launch/monitoring are recorded below. PIDs in
these local sandbox receipts are namespace-local (several are PID 2); session
handles identify the distinct jobs. Old stdout is collected through these handles
and semantic progress is checked in each run's own status/metrics files.

| Run directory suffix | Exec handle | Declared coverage |
|---|---:|---|
| `mc_scaling_diagnostics` | 71594 | 57 statistical/network cases |
| `mc_scaling_digits_n64` | 61263 | 9 training trajectories |
| `mc_scaling_digits_n128` | 36698 | 9 training trajectories |
| `mc_scaling_digits_n256_seed0` | 83583 | 3 training trajectories |
| `mc_scaling_digits_n256_seed1` | 91992 | 3 training trajectories |
| `mc_scaling_digits_n256_seed2` | 35919 | 3 training trajectories |
| `mc_scaling_noise_control_n256` | 53640 | 3 separate noise-control trajectories |

## Analytic expectation

For unit probes u with entries ±1/sqrt(n), define P=(n/m) sum u u^T and
delta=lambda-q. The correction is q+P delta. In exact linear response,

    E[||lambda_hat-lambda||²] = ((n-1)/m) ||delta||².

Thus fixed relative accuracy of the full correction generally requires m
proportional to n. Eight probes give RMS error factors sqrt((n-1)/8) relative to
the uncorrected discrepancy, not relative to the true adjoint.

Independent isotropic read error eta_q in q with coordinate variance v_q and
independent scalar reading noise with variance v_y give

    E[||lambda_hat-lambda||²]
      = ((n-1)/m) (||delta||² + n*v_q) + (n²/m)*v_y.

For paired independent voltage reads at each sign, v_y=||c||²*sigma²/(2*h²),
and v_q=sigma²/(2*beta_eff²). The baseline error is shared by the m readings;
it must not be counted as m independent baseline noises. These expressions hold
for the specified linear/noise model and do not include finite-amplitude bias.

At fixed total current h and fixed ||c||, read-noise amplification can grow as
n²/m. Increasing per-node current with width would change that excitation budget.
Independent examples can reduce batch gradient variance, but the relevant norm
is after the local parameter-force mapping and batch averaging. A universal
1/B relative-gradient-error law does not follow when true gradients cancel.

For completeness, E[P]=I and E[P²]=(1+(n-1)/m)I for the IID unit-sign design.
Consequently E[(P-I)²]=(n-1)I/m. The shared-baseline term contributes
v_q*tr(E[(P-I)²]), while independent scalar reading errors contribute
(n/m)²*m*v_y. This gives the stated noise formula without assuming independent
copies of the baseline. A test enumerates every three-state, two-probe sign design
and independent sign-distributed measurement noises, checking both mean and full
covariance exactly.

More generally, the noiseless adjoint-error covariance for one example is

    C_i = (||delta_i||² I + delta_i delta_i^T - 2 diag(delta_i²)) / m.

For independent probe designs across B examples and local parameter-force maps
D_i=-(partial_theta F_i)^T, the batch gradient-error covariance is

    (1/B²) sum_i D_i C_i D_i^T.

The relative batch-gradient MSE divides its trace by the squared true batch
gradient norm. That denominator can shrink when example gradients cancel. This
explains why applying a universal 1/sqrt(B) factor to a relative error would be
incorrect, even though independent example noises do average down in absolute
variance. These statements are for the local gradient before optimizer projection.

## Results

The statistical and frozen-network diagnostics completed all 57 declared cases
in 44.9 seconds, producing 45 statistical rows and 168 frozen-gradient rows. All
metrics are finite. Every empirical statistical MSE lies within 1.46 estimated
standard errors of its predicted value; this is a consistency check, not a formal
multiple-comparison significance test.

| States n | Eight-probe MSE / clean discrepancy squared, measured | Predicted |
|---:|---:|---:|
| 32 | 3.823 | 3.875 |
| 64 | 7.803 | 7.875 |
| 128 | 16.865 | 15.875 |
| 256 | 31.693 | 31.875 |
| 512 | 66.302 | 63.875 |
| 1024 | 131.344 | 127.875 |
| 2048 | 263.749 | 255.875 |
| 4096 | 493.426 | 511.875 |

The table is noiseless synthetic projection data with unit discrepancy and 128
independent designs per setting. The normalized RMS error grows as sqrt(n/m).
For m=n/4, the predicted MSE factor stays near four; this is a constant-error
control, not accurate recovery of the entire individual adjoint. To attain RMS
error <= epsilon times the original discrepancy using this IID estimator requires
m >= (n-1)/epsilon² under the exact-response model. A full orthogonal basis instead
recovers the adjoint with n directions in the noiseless linear limit.

The second check uses the actual initialized nonlinear classifier, but obtains
local linear responses from its analytic Jacobian. These are **oracle-derived
diagnostics**, not additional physical measurements. It fixes a 96-example batch
per seed and repeats each random design 32 times. Mean batch parameter-gradient
cosines across the three seeds are:

| States | 8 IID-sign probes | n/4 IID-sign probes | 8 orthogonal MC probes |
|---:|---:|---:|---:|
| 32 | 0.581 | 0.581 | 0.634 |
| 64 | 0.478 | 0.616 | 0.505 |
| 128 | 0.396 | 0.654 | 0.406 |
| 256 | 0.322 | 0.694 | 0.324 |
| 512 | 0.262 | 0.734 | 0.263 |

Independent probes per example and minibatch averaging help, but fixed eight
probes still lose gradient quality with increasing width. Orthogonal MC only
changes the ideal MSE factor from (n-1)/m to (n-m)/m for m<=n; at m=8, its relative
advantage shrinks as n grows. The improved proportional-budget cosine is specific
to this parameterization and data. It is not a universal monotonic scaling law.

The baseline adjoint relative error stays around 0.75–0.83 across all five widths,
so the change is not explained by a vanishing baseline mismatch at large width.
RMS relative batch-gradient error with eight probes rises from 1.41 at n=32 to
3.69 at n=512. Minibatch prefixes 1/8/32/96 are included in the CSV and plot; their
true gradient norms change as examples cancel, so relative error does not obey a
universal inverse-square-root batch-size curve.

[Diagnostic plots](../simulation_results/mc_scaling_diagnostics/diagnostics.png),
[statistical measurements](../simulation_results/mc_scaling_diagnostics/statistical.csv),
and [frozen-network gradient measurements](../simulation_results/mc_scaling_diagnostics/frozen_gradients.csv)
preserve the conditions and measurement sources.

### Completed physical training

All 27 new primary trajectories completed, plus the three separate noise controls.
Together with the six reused 32-state controls, **all 36 selected checkpoints were
replayed**, reproducing their stored test losses and accuracies. The primary merge
contains 528 epoch rows and the noise control another 48; the maximum primary force
residual was 9.999815e-10, below the 1e-9 tolerance. Initialization is matched across
methods at each width, and the dataset/preprocessing is identical across widths.
All declared sessions terminated successfully. The mathematical and training
pipeline checks now comprise **74 passing tests**.

The table shows test accuracy, mean ± population SD across three seeds after 15
epochs. Standard deviations are descriptive spreads, not confidence intervals.

| States | Trainable parameters | Exact adjoint | 8 IID-sign probes | n/4 IID-sign probes |
|---:|---:|---:|---:|---:|
| 32 | 1,936 | 95.65% ± 0.13 pp | 95.09% ± 0.57 pp | 95.09% ± 0.57 pp (m=8) |
| 64 | 5,536 | 95.74% ± 0.35 pp | 94.91% ± 0.47 pp | 95.37% ± 0.26 pp (m=16) |
| 128 | 15,808 | 95.65% ± 0.35 pp | 93.06% ± 0.99 pp | 95.46% ± 0.57 pp (m=32) |
| 256 | 48,640 | 95.56% ± 0.23 pp | 89.72% ± 0.68 pp | 95.56% ± 0.68 pp (m=64) |

Eight probes are sufficient for useful training at the smaller widths but do not
retain the reference accuracy as width grows under this fixed optimizer/budget.
Increasing m in proportion to n restores accuracy close to the reference here.
This is consistent with the sampling law and frozen-gradient results, but is not
a theorem that successful training must use n probes per update. Different
learning rates, longer training, batch sizes, or structured baselines may change
the accuracy/cost tradeoff; they were not optimized in this comparison.

The paired noise control at 256 states reached **89.91% ± 0.69 pp**, compared with
89.72% ± 0.68 pp at read noise 1e-5. The mean paired change was only +0.19 pp; two
seeds improved by one test example and the third was unchanged. Thus removing read
noise did not close the roughly 5.8 pp gap to the exact-adjoint reference. The
limited-probe estimator interacting with this optimizer remains the main issue
in this experiment; read-noise amplification can still matter in other regimes.

The cost of retaining accuracy is visible in both phase counts and CPU work:

| States | Eight-probe phases | n/4-probe phases | Eight-probe CPU seconds | n/4-probe CPU seconds |
|---:|---:|---:|---:|---:|
| 32 | 19 | 19 | 23.4 | 23.4 |
| 64 | 19 | 35 | 42.0 | 73.3 |
| 128 | 19 | 67 | 77.6 | 260.5 |
| 256 | 19 | 131 | 172.6 | 1229.6 |

Phases are per example/update; times are mean complete 15-epoch trajectories,
including evaluation, gradient audits and parameter projection. These are local
CPU timings with concurrent runs, not a controlled timing benchmark. The 32-state
timing is reused from the earlier source, before redundant coupling-norm
recomputation was removed. At 256 states the exact-adjoint control took 45.5 seconds
on average. Measuring gradients through nudges took more CPU time than the
analytic reference in this small dense digital simulator.

First validation-threshold hits are retained in `summary.json`. All three
256-state 64-probe runs first reached 95% validation accuracy by epoch 3; only one
of the three eight-probe runs reached that threshold within 15 epochs (at epoch
14). A first crossing is not a sustained target: the final validation/test
metrics and unreached seeds remain in the report. This comparison does not tune
stopping times or report test-set-selected checkpoints.

[Training plots](../simulation_results/mc_scaling_combined/scaling.png),
[combined report](../simulation_results/mc_scaling_combined/report.md),
[primary checkpoint verification](../simulation_results/mc_scaling_combined/verification.json),
and [paired noise-control verification](../simulation_results/mc_scaling_combined/noise_control_verification.json)
contain the final evidence. Lock-in measurements, fluctuation inference, larger
datasets and hardware wall-time scaling remain outside this comparison.

## What the operation count means

One physical gradient measurement uses 3+2m equilibrations per example: the free
state, two cost nudges, and m positive/negative probe pairs. Eight probes cost 19
phases regardless of width, while n/4 probes cost 19, 35, 67 and 131 phases for
32, 64, 128 and 256 states. This count alone does not make a larger equilibrium
equally cheap: there are more nodes/connections and potentially different physical
settling times. The simulator counts a full state-vector read per phase; a literal
all-node scalar-read count adds another factor n. Output-only scalar projection
readout could change that readout accounting but is not implemented here.

For a dense digital simulation, a forward force evaluation costs O(n²) per
example. With T relaxation iterations per phase, probing costs O(m*T*n²), before
stability checks, gradient construction, and optimizer projection. Holding
individual-adjoint sampling error constant with m=O(n) therefore gives O(T*n³)
forward work. This is an operation-count argument; CPU wall times in these small
concurrent runs do not establish an asymptotic timing exponent. Hardware
parallelism, sparse coupling, a useful low-dimensional correction, or a known
reciprocity transformation could change the practical comparison.

## Reproduction

From `/home/filip/server_code_random_nudge`, using the existing `py312` environment:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DISABLE_SHM=1
DRN_PYTHON=/home/filip/miniconda3/envs/py312/bin/python
"$DRN_PYTHON" -m labs.tools.measure_mc_scaling \
  --output simulation_results/mc_scaling_diagnostics
"$DRN_PYTHON" -m labs.tools.train_recurrent_eqprop_digits \
  --size 64 --methods adjoint mc8 mc16 --learning-rate 0.3 \
  --output simulation_results/mc_scaling_digits_n64
"$DRN_PYTHON" -m labs.tools.train_recurrent_eqprop_digits \
  --size 128 --methods adjoint mc8 mc32 --learning-rate 0.3 \
  --output simulation_results/mc_scaling_digits_n128
for seed in 0 1 2; do
  "$DRN_PYTHON" -m labs.tools.train_recurrent_eqprop_digits \
    --size 256 --methods adjoint mc8 mc64 --learning-rate 0.3 --seeds "$seed" \
    --output "simulation_results/mc_scaling_digits_n256_seed${seed}"
done
"$DRN_PYTHON" -m labs.tools.train_recurrent_eqprop_digits \
  --size 256 --methods mc8 --learning-rate 0.3 --read-noise 1e-30 \
  --output simulation_results/mc_scaling_noise_control_n256
"$DRN_PYTHON" -m labs.tools.summarize_mc_scaling \
  --runs simulation_results/digits_comparison \
    simulation_results/mc_scaling_digits_n64 \
    simulation_results/mc_scaling_digits_n128 \
    simulation_results/mc_scaling_digits_n256_seed0 \
    simulation_results/mc_scaling_digits_n256_seed1 \
    simulation_results/mc_scaling_digits_n256_seed2 \
  --noise-control-run simulation_results/mc_scaling_noise_control_n256 \
  --output simulation_results/mc_scaling_combined
```

Existing output directories are deliberately rejected; choose fresh names for a
rerun. The seed loop above is sequential for readability; the original three
256-state seeds ran concurrently with one BLAS thread per process. The combined
summary checks source coverage and per-epoch physical counts, enforces identical
datasets across widths and identical initialization across methods at each width,
then replays all selected checkpoints. The 32-state adjoint/mc8 trajectories are
reused from the previous completed comparison; they are not new independent runs.
