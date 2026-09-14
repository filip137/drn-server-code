# Fashion-MNIST: shortened Jacobian-homeostasis comparison

Status: training cancelled on user steering; read-only cosine analysis completed,
14 September 2026. The user selected
[Laborieux & Zenke, arXiv:2309.02214v2](https://arxiv.org/html/2309.02214v2)
and explicitly reduced the training budget from 50 to **10 epochs**.
Target: the adjacent-layer, independently trainable forward/backward network
in Figure 4a-d, comparing its VF rule, homeostasis, and measured feedback.
This study does not reproduce the paper's 50-epoch endpoints, loop architecture,
predictive-coding experiment, or complex-nudge Table 1 sweep.

The user subsequently requested cosine-similarity analysis instead of further
accuracy training. The training launcher was stopped with exit 130: seven cases
completed, four retain partial checkpoints, and nine never started. The original
20-case grid is **not complete**. See `main/cancellation.json` for exact coverage.
The active deliverable is [fmnist_feedback_cosines.md](fmnist_feedback_cosines.md).

## Source and fixed protocol

The public authors' code was inspected at commit
`30592f576bd4d4a20d3c13632f0792b0fa452781`. Relevant files:
`run.sh`, `arch/mlp_2-256.json`, `models/vfs.py`, `models/dyn.py`,
`models/utils.py`, `utils/funcs.py`, and `utils/data.py`.
Source copies are under `/tmp/jacobian_homeostasis_source_20260914`.
The direct public download succeeded after a temporary approval-review service
capacity failure. No repository source was uploaded to any remote host.

- All 60,000 official Fashion-MNIST training images, all 784 pixels divided by
  255 only. Official 10,000-image test set evaluated each epoch, as in the authors'
  loader. Their plots call this validation error. We label it test error and do
  not select hyperparameters or checkpoints using it.
- Two 256-unit hidden layers, 10 dynamical output states, trainable 10x10 readout.
  Shifted sigmoid `phi(u)=sigmoid(4u-2)`; 522 dynamical states in total.
- Independent directional weights, initial mixing angle 90 degrees; LeCun
  truncated-normal initialization with each direction's own fan-in, zero biases.
  No norm cap, weight tying, fixed skew matrix, or operating-point centering.
- Adam, learning rate 1e-4, default moments 0.9/0.999, epsilon 1e-8, no weight
  decay, batch size 50. Authors use Optax AdamW with zero decay, equivalent here.
- Reset free states to zero, 150 synchronous free steps, 20 response steps.
  Report actual convergence residuals; a phase is not assumed equilibrated
  merely because its allotted steps ended. No hidden extra settling or clipping.
- Five paired seeds 0..4, same initial weights and minibatch ordering across
  methods at each seed; **10 epochs**. Four methods, 20 declared trajectories.

## Compared learning rules

We retain the existing `F-b=0` convention, whereas the authors' code adds
`+beta*c` to the force. Consequently our response has the opposite sign and
our local parameter gradient is `-partial_theta(F)^T ell`. The final parameter
updates agree after this sign translation. A directed weight uses the
presynaptic free activity times the negative postsynaptic feedback. The readout
has its direct cross-entropy gradient.

1. `vf_ad`: exact forward tangent of the 20-step zero-beta response trajectory,
   including the final activation derivative. Equivalent to the paper's `N=0`
   forward differentiation. This remains a biased VF feedback rule in an
   asymmetric network; exact differentiation of the response does not make it
   a true adjoint.
2. `vf_ad_homeo`: the same VF signal plus coefficient-1 homeostasis, five
   Gaussian vectors per example and two JVPs per vector. Match the public code's
   activation-bypassed Jacobian `J_code=W-I`. The expected loss is
   `||W-W.T||_F^2/(2*n)`; take its parameter gradient with AD at fixed free states.
3. `vf_central`: paired real cost nudges at beta +/-0.01, 20 steps each,
   activity response. This control isolates finite-nudge bias relative to `N=0`.
4. `zo_learned4`: the earlier measured predictor `D*(-c+H*c_out)` with H initially
   zero, scalar normalized-LMS step 0.25, and four fresh, independent per-example
   unit-norm random-sign probes. For each probe, pair +/-0.01 fixed-current
   phases, measure `y=c.T*r_z`, and form
   `ell=baseline+(n/4)*sum(z*(y-z.T*baseline))`.
   Only after the current correction is formed are those observations used to
   update H. No Jacobian, transpose, homeostatic AD, or reference gradients enter
   this training path. The free local parameter-force derivatives remain known.
   The predictor also uses the known local activation slope `D`; zero-order
   refers to measuring the recurrent response through nudges, not to eliminating
   all knowledge of local element derivatives.

For 522 states the ideal residual-estimator variance factor is `(522-1)/4=130.25`.
Few probes do not guarantee a precise per-example direction. A finite 20-step
measurement estimates a finite-time response; the equilibrium identity also
requires sufficient free and perturbed convergence. This is checked, not assumed.

## Adaptations and checks

This is an equation-level PyTorch reimplementation, not an execution of the
authors' original JAX environment. The public utility module has stale imports
and missing configuration fields. Torch's random streams differ from JAX and
NumPy minibatch shuffling differs from TensorFlow. The initialization distribution,
model, update equations, and primary optimizer settings are matched. Main
training is float32; read-only oracle diagnostics use float64. The finite-probe
arm and 10-epoch limit are explicit additions/changes.

Tests independently compare the finite-time tangent to automatic differentiation,
the local learning gradient to parameter finite differences, full coordinate
probe responses to an adjoint, and homeostatic derivatives to the exact penalty.
Physical-path tests prohibit derivative/oracle/projection access and check
that predictor fitting follows, rather than contaminates, the fresh correction.
Dataset files are checked against raw SHA256 values derived from gzip archives
matching the dataset publisher's MD5 checksums.

Diagnostics use the same first eight training examples at initialization and
epochs 1, 5, and 10. Record layerwise feedback and parameter-gradient alignment,
weight angles, the paper-code Jacobian symmetry, and exact adjoint solve
residuals. Double the free and response horizons on this cohort to measure
finite-time error. Diagnostic work is outside the training budget.

## Work accounting and monitoring

Training accounting separates state iterations, digital tangent JVPs,
homeostasis JVPs/Gaussian vectors, measured nudge phases, scalar probe reads,
and total squared probe excitation. Per-example budgets are:

| Method | State iterations | Response tangent JVPs | Homeostasis JVPs | Measured nudged phases |
|---|---:|---:|---:|---:|
| VF AD | 170 | 20 | 0 | 0 |
| VF AD + homeostasis | 170 | 20 | 10 | 0 |
| Paired VF | 190 | 0 | 0 | 2 |
| Learned + four probes | 310 | 0 | 0 | 8 |

Every arm has one free phase. Phase counts are not wall time or physical energy.
Probe currents have norm 0.01 per sign, not amplitude 0.01 at every node.

Use the experiment-run-watchdog workflow. Before the main batch, run all methods
at full width on 2,000 training examples for one epoch, without test evaluation.
Outputs: `simulation_results/fmnist_homeostasis_20260914/smoke` and `main`.
Each run records source snapshots/hashes, config, command, environment and cases.
Workers write status heartbeats every 20 seconds, epoch metrics and checkpoints,
plus terminal result/failure JSON. Preserve failed results. Operational defects
may be repaired and retried with identical science; poor accuracy is not a reason
to change the protocol. Verify launch and initial artifact progress, then monitor
to complete declared coverage. Main process handles and completion evidence will
be recorded below. Plotting uses matplotlib; final checkpoint replay is independent.

Main invocation (local CPU, single-thread workers):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DISABLE_SHM=1 \
  /home/filip/miniconda3/envs/py312/bin/python -m labs.tools.train_fmnist_homeostasis \
  --output simulation_results/fmnist_homeostasis_20260914/main --workers 4 --epochs 10
```

## Live execution and additional validity checks

Training source is commit `a1361fdc`. The main batch runs in unified exec session
`83374`, local CPU, four single-thread workers. The full-width, four-method smoke
run (`55830`, exit 0) and 156 lab tests passed before launch. The main source
snapshot remains frozen while separate analysis tools are added.

The full-data run exposed non-equilibrated examples that were absent in the
2,000-example smoke gate. A spot check at seed-0 epochs 2-4 found force residual
norm above 1e-3 on roughly 3-5% of 500 test examples. A large one-step change
with a small two-step change suggests period-two dynamics. An even-horizon
comparison alone can miss such oscillations. This is retained as a scientific
limitation of the fixed-step runs; the protocol is not silently retuned.

The final reporter independently replays all 10,000 test examples and records
residual quantiles, the fraction above 1e-3, period-two counts, and prediction
changes after one additional step. A separate float64 audit filters the original
eight-example training cohort by force residual below 1e-7 before reporting
equilibrium feedback/parameter-gradient alignment. The original unfiltered
training audits are retained, but inverses at non-equilibria are not interpreted
as true equilibrium task gradients.

A direct parity check executes the authors' actual MLP vector-field and dynamics
routines under isolated JAX 0.4.35/Flax 0.10.2. On a small-network fixture, states,
VF responses, every parameter-gradient block, and the homeostatic loss/gradient
agree to numerical precision. To load the public modules, unused CNN pooling and
two missing, unused constraint-helper imports are omitted. Common Gaussian draws
are supplied for the homeostasis comparison. Public computation routines are
otherwise unchanged. A full-width trained-checkpoint parity check is also run.
All source hashes and detailed errors are retained in `parity_*.json` alongside
fixtures and the public source snapshot.

The trained 522-state parity fixture selects the three largest-residual examples
among the first 500 test images of seed-0 VF at epoch 10. It passed as well:
maximum state difference 4.0e-15, force difference 4.9e-15, and translated VF
response difference 1.1e-15. Every task and homeostatic parameter-gradient block
passed. This confirms equation parity at these weights; it does not establish
that the original published training trajectories had the same convergence issue.
