# Testing cheaper random-nudge corrections

Exploratory, non-canonical follow-up, 13 September 2026. Worktree
`server_code_random_nudge`, branch `codex/hopfield-random-nudge-adjoint`, starting
at `d1fdddca`. Existing scaling artifacts remain unchanged.

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
