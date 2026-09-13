# Testing cheaper random-nudge corrections

Exploratory, non-canonical follow-up, 13 September 2026. Worktree
`server_code_random_nudge`, branch `codex/hopfield-random-nudge-adjoint`, starting
at `d1fdddca`. Existing scaling artifacts remain unchanged.

The strongest result was the **learned baseline with four fresh probes**: 96.30%
test accuracy at nine training equilibrations per example. The tuned original
eight-probe method reached 94.54% at 19 equilibrations. These are three-seed
results at 256 states, not a demonstration of dimension-independent scaling.

## Completed results

All 30 validation-screen trajectories and 27 full training trajectories
completed without failed or excluded cases. Together with nine reused controls,
all **66 checkpoints** reproduced their saved validation metrics and, for final
training runs, their test metrics. The separate five-checkpoint smoke gate also
passed. Epoch coverage, matching data/initializations, phase counts, predictor
observation counts, finite metrics, symmetry/stability constraints and residuals
at or below 1e-9 passed. The implementation suite passed 84 tests.

Training is 15 epochs on the fixed 1149/288/360 digits split. Values below are
mean ± population SD across initialization/probe seeds 0/1/2. “Selected” settings
come from the predeclared eight-epoch, seed-0 validation screen.

| Baseline and probes | Original LR 0.3, no momentum | Selected optimizer | Selected test accuracy | Equilibrations/example |
|---|---:|---|---:|---:|
| Ordinary EqProp, 8 | 89.72% ± 0.68% | LR 0.1, EMA 0.9 | 94.54% ± 0.13% | 19 |
| Local slope, 4 | 92.59% ± 1.67% | LR 0.3, EMA 0.9 | 95.09% ± 0.69% | 9 |
| Local slope, 8 | 92.69% ± 0.52% | LR 0.1, EMA 0.9 | 95.19% ± 0.13% | 17 |
| Learned map, 4 | 95.19% ± 0.47% | LR 0.3, EMA 0.9 | **96.30% ± 0.13%** | **9** |
| Learned map, 8 | 95.46% ± 0.47% | LR 0.3, EMA 0.9 | 95.93% ± 0.47% | 17 |

Reused controls at LR 0.3 without momentum: exact adjoint 95.56% ± 0.23%; ordinary
EqProp baseline with 64 random probes 95.56% ± 0.68%, using 131 equilibrations per
example. Exact adjoint requires one free phase plus a dense adjoint solve; its
one-phase count is not the full cost of a physically measurable learning rule.

Three findings are supported here. Optimizer tuning recovers much of the original
eight-probe deficit. The local-slope baseline helps at the original optimizer,
but leaves appreciable noise and variability. Learning the baseline from past
readings adds a further improvement at both the original and selected settings.
Four learned probes gave the highest final test mean in this comparison. This
does not establish that four probes are universally better than eight, or that
stochastic feedback outperforms the exact gradient: the exact control and the
noisy methods use different selected optimizers, and only one small dataset and
three seeds were tested.

A subsequent [direct feedback alignment connection](direct_feedback_alignment_connection.md)
examines using the predictor directly and learning from vector output responses.
Its frozen-checkpoint audit is separate from these completed training results;
lower-probe training has not yet been tested there.

The selected learned-four method uses 155,115 training equilibrations per seed,
versus 327,465 for tuned ordinary-eight and 2,257,785 for ordinary-64 at 15 epochs.
Those are 2.11 and 14.56 times fewer phases, respectively, excluding screening,
evaluation and digital arithmetic. The **screen itself used 3,915,792 training
equilibrations** across all five methods; the 27 confirmations/ablations used
another 6,359,715. Tuning was not free. Predictor updates took approximately one
CPU second per learned-four trajectory; H stores 2560 floating-point coefficients.
Wall times are in the machine-readable summary, with differing concurrency as
described below; these are not hardware speedup measurements.

First 95% validation crossings, with costs including all training through that
epoch:

| Method/settings | Crossing epoch, seeds 0/1/2 | Training equilibrations, seeds 0/1/2 |
|---|---|---|
| Original ordinary-eight | never, 14, never | never, 305,634, never |
| Tuned ordinary-eight | 3, 6, 5 | 65,493; 130,986; 109,155 |
| Tuned local-four | 3, 7, 8 | 31,023; 72,387; 82,728 |
| Tuned learned-four | 3, 5, 6 | 31,023; 51,705; 62,046 |
| Tuned learned-eight | 3, 3, 3 | 58,599; 58,599; 58,599 |
| Original ordinary-64 | 3, 3, 1 | 451,557; 451,557; 150,519 |

Crossings can be transient; these numbers do not describe sustained target
accuracy or a separately evaluated early-stopping policy.

### Why the learned baseline helped

A post-run explanatory audit compares all three baselines at the **same final
learned-four models**, using the first 96 validation inputs. This audit uses
analytic Jacobians for reference only and never updates H. Mean relative
adjoint errors across seeds were 0.662 for ideal ordinary EqProp, 0.478 for local
slopes, and **0.152 for the learned map**. Thus H generalizes beyond the training
measurements in this limited check. At these fixed states, the predicted ideal
MC adjoint MSE with four learned probes is 0.106 times that with eight
ordinary-EqProp probes (mean of per-seed ratios). This excludes read noise and
finite-nudge bias and is an adjoint statistic, not a parameter-gradient or
training-speed guarantee.

During actual training, the final audited minibatch gradient cosine was 0.626
for selected learned-four versus 0.197 for tuned ordinary-eight. The stability
projection remained active on every learned-four update. Applied-update cosines
were lower (0.291 and 0.096 respectively) because they also include momentum
history and projection. Unbiased adjoint estimation does not imply unbiased
projected optimizer steps.

No full Jacobian was measured for learning. The promising next scaling test is
to hold four learned probes fixed while increasing to 512/1024 states, with a
matched optimizer control and accounting for the predictor's warm-up. This
experiment does not yet test that scaling, one-time skew calibration, or a DRN
hardware implementation.

As in the previous runs, independent read noise is added to the paired nudged
measurements; the free-state readout and the known local cubic coefficient are
clean. Noise in free-state/local-slope calibration remains untested.

[Plots](../simulation_results/nudge_improvements_20260913/combined/improvements.png),
[validation screen](../simulation_results/nudge_improvements_20260913/combined/screen.png),
[complete result table](../simulation_results/nudge_improvements_20260913/combined/report.md),
[metrics](../simulation_results/nudge_improvements_20260913/combined/metrics.csv),
[checkpoint verification](../simulation_results/nudge_improvements_20260913/combined/verification.json),
and [held-out baseline audit](../simulation_results/nudge_improvements_20260913/combined/held_out_baseline_audit.json)
preserve the exploratory evidence. The combined report replays each unique
checkpoint, including every screened checkpoint on validation data only.

## Plan recorded before launch

Use the existing 256-state nonlinear recurrent digits classifier, fixed skew
norm 1, train/validation/test split 1149/288/360, batch 96, nudge L2 norm 0.01,
voltage-read noise 1e-5 and force tolerance 1e-9. Train the symmetric recurrent
matrix, hidden input matrix and bias under the existing spectral constraint.

Compare five methods: ordinary EqProp-initialized MC with eight probes (`mc8`),
local-slope-initialized MC with four/eight probes (`local_mc4`, `local_mc8`),
and a baseline learned from earlier measurements with four/eight probes
(`learned_mc4`, `learned_mc8`). Every correction retains the unbiased n/m factor.
The latter four methods need no error-nudged phases: respectively 9/17
equilibrations per example, compared with 19 for `mc8`.

For each method screen learning rates 0.03/0.1/0.3 and momentum 0/0.9, seed 0,
eight epochs (30 trajectories). Momentum is a bias-corrected exponential average
of gradients, so the learning rate retains its steady-gradient meaning. Select
one setting per method by final validation accuracy, breaking ties with lower
validation cross entropy. Never evaluate test data during this screen.

Confirm each selected setting for 15 epochs, seeds 0/1/2 (15 trajectories).
Also run the four new methods at the original learning rate 0.3, momentum 0,
15 epochs, three seeds, reusing a selected confirmation when it has exactly
that configuration. Reuse the prior `mc8`, `mc64` and exact-adjoint controls at
256 states. This separates the baseline change from optimizer tuning. Seed 0
is reused for selection; seeds 1 and 2 provide fresh initialization checks,
not an independent dataset. Test metrics are reported only at final epoch.

The learned baseline is b=D^-1(-c+H*c_output), D_i=1+3*cubic*s_i^2.
H starts at zero and is trained only on measured equations from previous
minibatches. After computing the current corrected gradient, use relaxed
normalized row projections (relaxation 0.25) on that minibatch's equations to
update H for the next minibatch. Freeze b before drawing current probes. The
predictor learning rate is predeclared and is not tuned in this experiment.
Its digital arithmetic/storage cost is reported separately from phase counts.

Checks: ideal arbitrary-baseline unbiasedness; physical methods barred from
Jacobian/oracle access when audits are disabled; absence of error nudges in the
new methods; predictor update ordering; momentum semantics; validation/test
isolation; a small 256-state smoke run before the full screen. Audit the first
minibatch of each epoch against the true adjoint and record raw gradient quality,
baseline error, applied projected-update quality and clipping counts. Dense
Jacobians are diagnostics only and never train the predictor or physical learner.

## Monitoring contract

Use the experiment-run-watchdog exploratory tier. One BLAS thread per foreground
CPU Python process, independent method shards in parallel. Fresh directories
under `simulation_results/nudge_improvements_20260913/` contain readable configs,
source/environment receipts, per-epoch metrics, status JSON, log and final
checkpoints. The runner records PID; exec session handles are recorded here
after launch because sandbox PIDs are namespace-local. Heartbeat every epoch
and trajectory. Expected duration is minutes per shard, refined by the smoke
run. Check process handle and first semantic progress, then monitor through
completion. Preserve failed artifacts and only retry the same scientific config
after diagnosing an operational fault. Complete coverage, finite metrics,
phase counts, residual bounds and checkpoint replay are required for handoff.

Report final accuracy, training equilibrations, first validation-95% crossing
(including runs that never reach it), CPU wall time, and baseline/gradient
diagnostics. Include the screening overhead rather than calling tuning free.
These tests address a 256-state toy classifier; they do not establish improved
asymptotic scaling or a circuit implementation.

Launch gate: 84 tests passed. The 256-state, five-method smoke run completed
in 9.7 seconds (exec session 32091), using 192 training examples, one epoch,
momentum 0.9 and validation-only evaluation. All five final checkpoints replayed
exactly on validation data; phase counts and force residuals passed. Full screens
are expected to take roughly 5–15 minutes per six-setting method shard.

Full-screen source: `95bbc419`. Five independent launch handles were verified
through multiple completed epoch heartbeats, correct commands/configs and fresh
metrics. Directories below are relative to the improvement artifact root.

| Directory | Exec session | Coverage |
|---|---:|---|
| `screen_mc8` | 28829 | Six validation-only settings |
| `screen_local_mc4` | 82299 | Six validation-only settings |
| `screen_local_mc8` | 27352 | Six validation-only settings |
| `screen_learned_mc4` | 84697 | Six validation-only settings |
| `screen_learned_mc8` | 93215 | Six validation-only settings |
| `confirm_local_mc4` | 76056 | Selected LR 0.3, EMA 0.9; seeds 0/1/2 |
| `reference_local_mc4` | 15020 | Original LR 0.3, EMA 0; seeds 0/1/2 |
| `confirm_learned_mc4` | 36783 | Selected LR 0.3, EMA 0.9; seeds 0/1/2 |
| `reference_learned_mc4` | 57655 | Original LR 0.3, EMA 0; seeds 0/1/2 |
| `confirm_local_mc8_seed{0,1,2}` | 98865, 57015, 68889 | Selected LR 0.1, EMA 0.9 |
| `reference_local_mc8_seed{0,1,2}` | 83940, 98457, 59870 | Original LR 0.3, EMA 0 |
| `confirm_learned_mc8_seed{0,1,2}` | 68827, 87477, 18616 | Selected LR 0.3, EMA 0.9 |
| `reference_learned_mc8_seed{0,1,2}` | 26792, 67441, 66848 | Original LR 0.3, EMA 0 |
| `confirm_mc8_seed{0,1,2}` | 24474, 31450, 11365 | Selected LR 0.1, EMA 0.9 |

The longer eight-probe confirmations/reference runs may use independent
one-seed processes (`..._seed0`, `..._seed1`, `..._seed2`) to reduce waiting.
This changes only scheduling, preserving initialization, data order, probe RNG,
settings and budgets. Each process has a separate receipt/status/output path.
Concurrent-process counts differ, so elapsed CPU wall times are implementation
measurements; physical phase counts are the primary cost comparison.

## What learning rule is being tested

The `mc8` control starts with the ordinary error-nudged EqProp response q=R*c.
The four new variants instead start with a local or learned prediction b; they
remove those two error-nudged phases. All five measure fresh equilibrium
responses r_k to signed physical current probes and use

    lambda_hat = b + (n/m) sum_k u_k (c^T r_k - u_k^T b).
    gradient_hat = -(partial_theta F)^T lambda_hat.

Thus the new variants use equilibrium perturbations and the same local
parameter-force derivatives as the earlier corrected learner. They are not the
ordinary two-phase contrastive EqProp rule. No full response/Jacobian is fitted.
The learned predictor stores n*outputs=2560 coefficients; its diagonal state
dependence comes from the known local cubic slope. It cannot represent an
arbitrary input-dependent inverse response. Conditional unbiasedness of the
fresh MC correction is a linear-response statement; finite-nudge bias remains.

For any baseline b fixed before the fresh independent sign probes,

    E[lambda_hat | b, s, theta] = lambda,
    E[||lambda_hat-lambda||^2 | b, s, theta]
        = ((n-1)/m) ||lambda-b||^2.

These formulas omit read noise and finite-nudge error. A better baseline reduces
the coefficient of the dimensional variance cost. Learning a shared approximate
map from past measurements can amortize that cost, but this experiment does not
remove the n dependence or show that a fixed probe count works at arbitrary n.
The current predictor assumes that local slope factors plus one shared map
approximate the input-dependent response well enough.

The applied-update audit compares the actual momentum-plus-projection step with
an exact-gradient projected SGD step from the same current parameters and at the
same learning rate. This measures the combined effects of sampling, momentum
history and the stability projection. It is not an unbiased-gradient test or a
comparison against an exact optimizer with its own momentum trajectory.

## Reproduction

Use one BLAS thread per process, with the repository's Python environment:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DISABLE_SHM=1
task_python=/home/filip/miniconda3/envs/py312/bin/python
task_results=simulation_results/nudge_improvements_repeat

for method in mc8 local_mc4 local_mc8 learned_mc4 learned_mc8; do
  "$task_python" -m labs.tools.test_nudge_improvements \
    --stage screen --methods "$method" --output "$task_results/screen_$method"
  "$task_python" -m labs.tools.test_nudge_improvements \
    --stage run --methods "$method" \
    --selection "$task_results/screen_$method/selection.json" \
    --output "$task_results/confirm_$method"
done
for method in local_mc4 local_mc8 learned_mc4 learned_mc8; do
  "$task_python" -m labs.tools.test_nudge_improvements \
    --stage run --methods "$method" --learning-rate 0.3 --momentum 0 \
    --output "$task_results/reference_$method"
done
"$task_python" -m labs.tools.summarize_nudge_improvements \
  --root "$task_results" --output "$task_results/combined"
```

These commands are sequential for readability. The recorded run parallelized
independent processes as listed above. The collector expects the previous
`mc_scaling_digits_n256_seed{0,1,2}` controls beside the improvement root; their
reproduction commands are in [the scaling note](random_nudge_scaling.md).
All output directories must be fresh. A reference run matching the selected
settings can be reused by the collector; none of this screen's five selected
settings matches the original optimizer.
