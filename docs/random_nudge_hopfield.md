# Random-nudge adjoint correction in toy Hopfield networks

This is an exploratory implementation of the proposed reciprocity diagnostic and
Monte Carlo adjoint correction, using NumPy in float64. It is a standalone toy
under `labs/`, so it does not depend on or modify the main checkout's ongoing
Hopfield work. No GPU, downloaded dataset, autograd, or remote launch is needed.

Branch: `codex/hopfield-random-nudge-adjoint`. Worktree:
`/home/filip/server_code_random_nudge`. Base: `9083aa2217ff244e816b562353e3bf937ee1c13e`.

## Model and measurement protocol

The state follows

```
F(s) = -s - gamma*s**3 + (S + alpha*A) @ s + drive
S.T = S, ||S||_2 = 0.4; A.T = -A, ||A||_2 = 1.
```

At `alpha=0` this is a continuous soft-spin Hopfield network with energy
`E = s.T@s/2 + gamma*sum(s**4)/4 - s.T@S@s/2 - drive.T@s`.
The skew part breaks reciprocity. The symmetric part of the Jacobian is bounded
above by `-0.6 I`, so the model has a unique stable equilibrium for every constant
nudge even at large asymmetry. `gamma=0` is a linear control; `gamma=0.25` gives
nonlinear states. This choice avoids confusing asymmetric state coordinates or
mobility with an actual non-conservative coupling.

`relax` uses forward Euler evolution and checks the **force residual**, failing
loudly on nonconvergence. No inverse, Jacobian, transpose dynamics, or autograd is
used in physical response measurements. Dense solves in `exact_feedback` and the
ideal sweep are explicitly labeled evaluation oracles.

The nudge is `F(s) - beta*p = 0`. The error direction `c` is frozen at the free
equilibrium (the gradient of the linearized cost). Centered measurements use
`[s(+beta)-s(-beta)]/(2 beta)`; forward measurements use
`[s(+beta)-s(0)]/beta`. These approach `J^-1 p`. State-dependent error nudging has
the same infinitesimal response but different finite-beta errors; it is not
implemented here. Both phases start from the same free equilibrium. Random sign
vectors are unnormalized, with independent entries ±1 and covariance `I`.

```
q = J^-1 c; lambda = J^-T c; delta = lambda - q
r_z = J^-1 z
d(z) = c.T @ r_z - z.T @ q = z.T @ delta
lambda_hat = q + mean(z*d(z))
E[d(z)^2] = ||delta||^2
E[||lambda_hat-lambda||^2] = (n-1)/m * ||delta||^2
```

Two clearly distinguished controls are included. A shrinkage multiplier
`a=m/(m+n-1)` is **biased**, with ideal MSE
`[(1-a)^2 + a^2*(n-1)/m] * ||delta||^2`; it minimizes this expression under the
ideal independent-sign model. A full Hadamard sign basis has `m=n` and
`Z.T@Z/n=I`, so it reconstructs the adjoint exactly in the linear, noiseless
limit. It costs `n` probes and is not an independent Monte Carlo sample.

Read noise is independent Gaussian noise of standard deviation `sigma` on each
settled-state read. Response variance is `sigma^2/(2 beta^2)` for centered reads
and `2 sigma^2/beta^2` for one-sided reads. The latter explicitly reads the free
state independently for each response. Within one correction estimate, the
single measured error response `q` is reused in every projection. The simulator
reuses noiseless settled states across noise levels, then draws fresh read noise
for each estimator trial. There is no noisy dynamic forcing or MCMC.

The diagnostic's additive noise floor in the linear limit is
`response_variance * (||c||^2+n)`; both raw and noise-floor-subtracted values are
saved. The subtraction assumes known noise variance and can yield negative
estimates in finite samples; values are not clipped. Full orthogonal probes
cancel the shared `q` noise algebraically, but random-response read noise remains.
Force-tolerance error and floating-point cancellation eventually matter at tiny
beta. Finite nonlinear nudges can produce a nonzero diagnostic even at alpha=0.

## Checks and run commands

From the worktree root, using an environment with NumPy, Matplotlib and pytest:

```bash
export KMP_DISABLE_SHM=1 KMP_SHM_DISABLE=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHON=/home/filip/miniconda3/envs/py312/bin/python
"$PYTHON" -m pytest -q labs/tests/test_random_nudge_hopfield.py
"$PYTHON" -m labs.tools.run_random_nudge_hopfield --quick --output simulation_results/random_nudge_smoke
"$PYTHON" -m labs.tools.run_random_nudge_hopfield --output simulation_results/random_nudge_full
```

Output directories must be new; runs never overwrite old results. `--mode ideal`,
`--mode finite`, and `--mode train` run individual parts; `--epochs` changes the
toy learning update count. Full defaults are declared below before launch.

| Part | Coverage |
|---|---|
| Ideal identities and variance | n=8,16,32; gamma=0,0.25; alpha=0,0.25,0.75,1.5; seeds=0,1,2; m=1,4,16,64,256; 128 independent trials |
| Physical finite nudges | n=8; gamma=0.25; alpha=0,1.5; seeds=0,1,2; forward and central; beta=0.1,0.03,0.01,0.003,0.001,0.0001,0.00001; sigma=0,1e-8,1e-6,1e-4; m=1,8,32,128; 32 trials |
| Learning | n=8, six hidden and two output nodes; alpha=0,1.5; seeds=0,1,2; 64 train / 128 held-out inputs; 80 full-batch updates; learning rate=1; beta=1e-3; no read noise |
| Learning feedback | Exact adjoint, ordinary frozen-error EqProp, independent MC with 1/8/32 probes, full orthogonal 8-probe control |

The teacher and student share fixed recurrent weights. Only the input-to-hidden
matrix is trained; outputs receive no direct input drive. Targets are the
teacher's free output equilibria on independent synthetic inputs. All methods
share data and initial weights for each seed. The update is based on
`grad_B C = -mean(lambda_hidden outer x)`. Parameter finite differences verify
this sign. Learning curves match update counts, not physical computation budgets:
centered EqProp costs three equilibrations per example and an m-probe correction
costs `3+2m`, counting the free phase. Dense-oracle costs are not equivalent to
equilibration counts. The cost audit excludes evaluation states and diagnostic
Jacobian solves; it does not claim hardware speedups. Fixed recurrent couplings
also mean this is not a demonstration of training every synapse of a DRN.

## Exploratory run handoff and monitoring

Local foreground CPU runs are monitored through their exit code, `run.log`,
`status.json`, and completed-case counts. A case heartbeat is written after each
diagnostic batch, and every ten learning updates. The quick smoke is expected to
take seconds; the full run several minutes. Check active runs at least every
30 minutes if extended. The runner records the PID, exact command, source branch,
commit, uncommitted file list, Python/NumPy versions, configuration, planned
coverage and start time in `run.json` before doing scientific work. `status.json`
must say `complete`, coverage must match, and the CSVs, `summary.json`, `report.md`,
plots, and (for learning) final checkpoints must exist before handoff. A failed
run is retained; diagnose it and retry the unchanged configuration in a new
directory after fixing the operational error.

The full suite has 72 ideal cases, 84 physical cases (noise levels and probe
budgets reuse their noiseless responses), and 36 learning trajectories. This is
exploratory/non-canonical evidence; it is not a workflow-managed or paper-facing
study.

## Collected results (2026-09-11)

Both the smoke and full runs completed successfully. The full run took 173.3 s
locally, completing all 72 ideal cases, 84 physical cases and 36 learning
trajectories. It produced 792 ideal rows, 3,024 finite-nudge rows and 2,916
learning rows. All numeric values are finite; every measured force residual is
at most `1e-12`. All 36 final checkpoints were loaded and their held-out losses
independently recomputed, with zero discrepancy from the recorded values. The
14 focused tests pass, covering the projection identity, unbiasedness, variance,
finite-nudge convergence, finite-difference parameter gradients, shared read
noise, stability, and failure on nonconvergence.

Artifacts in this worktree:

- [Full report](../simulation_results/random_nudge_full/report.md),
  [machine-readable summary](../simulation_results/random_nudge_full/summary.json),
  [run configuration](../simulation_results/random_nudge_full/run.json) and
  [terminal status](../simulation_results/random_nudge_full/status.json).
- [Overview figure](../simulation_results/random_nudge_full/overview.png) and
  [SVG](../simulation_results/random_nudge_full/overview.svg).
- [Nudge-error figure](../simulation_results/random_nudge_full/finite_nudges.png).
- Detailed `ideal.csv`, `finite.csv`, `training.csv`, and `checkpoints/*.npz`
  under `simulation_results/random_nudge_full/`. These generated files are
  ignored by git; the commands above reproduce them.

In the ideal sweep the maximum error in `d(z)=z.T@delta` was `2.22e-15`.
Across asymmetric cases, the mean empirical/theoretical correction-MSE ratio
was **1.0148**, and the mean diagnostic/truth ratio at 256 probes was **1.00024**.
The full orthogonal control's largest adjoint MSE was `1.63e-31`.

Learning results below use alpha=1.5, an eight-node nonlinear network, three
seeds, 80 updates, and a shared mean initial held-out loss of **0.0558923**.
Loss is half the mean squared output error, summed over the two outputs.
SD is the population standard deviation over the three seeds (descriptive,
not a confidence interval).

| Feedback | Final held-out loss, mean ± SD | Initial gradient cosine | Equilibrations per example/update |
|---|---:|---:|---:|
| Exact adjoint oracle | 0.00044048 ± 0.00018730 | 1.0000 | 1 plus a dense adjoint solve |
| Ordinary frozen-error EqProp | 1.039716 ± 0.284423 | -0.5654 | 3 |
| MC, 1 independent probe | 0.00045655 ± 0.00021694 | 0.7220 | 5 |
| MC, 8 independent probes | 0.00044989 ± 0.00019445 | 0.9560 | 19 |
| MC, 32 independent probes | 0.00043083 ± 0.00018359 | 0.9781 | 67 |
| Full orthogonal 8-probe control | 0.00044048 ± 0.00018731 | 1.0000 | 19 |

For alpha=0 all methods give essentially the same final mean held-out loss,
about `0.000624984`, from `0.0103365` initially. Thus the conservative control
does not require a correction. At alpha=1.5, uncorrected feedback initially
points against the parameter gradient and worsens loss in **all three seeds**;
every correction method reduces it in all three. The orthogonal method follows
the exact-adjoint trajectory to finite-nudge accuracy.

One random probe per example already learns on this toy, despite its poor
per-example adjoint MSE. Each update averages **64 independent example-specific
probe estimates**, probes are redrawn across updates, and only the hidden input
matrix is learned. This is consistent with the variance formula; it does not
establish that a one-probe estimate recovers the full feedback vector accurately.
The small differences among corrected methods are not evidence that Monte Carlo
outperforms exact gradients. The asymmetric coupling is deliberately strong;
these results do not imply ordinary EqProp fails at every nonzero asymmetry.

Physical measurements show the expected nonlinear-bias/noise tradeoff. At
alpha=1.5 and zero read noise, the mean RMS projection error is:

| beta | Forward nudge | Centered nudge |
|---:|---:|---:|
| 0.1 | 1.080e-2 | 2.046e-3 |
| 0.01 | 1.064e-3 | 2.072e-5 |
| 0.001 | 1.062e-4 | 2.072e-7 |
| 0.0001 | 1.062e-5 | 1.365e-8 |
| 0.00001 | 1.063e-6 | 9.944e-8 |

Centered measurements have quadratic bias over the useful range; at the
smallest amplitudes the fixed equilibrium tolerance is amplified by division by
beta. In the orthogonal centered control, the lowest mean adjoint MSE among the
tested amplitudes occurs at beta=`1e-4`, `0.003`, `0.01`, and `0.03` for read
sigma=`0`, `1e-8`, `1e-6`, and `1e-4`, respectively. These are grid results for
this model and noise convention, not universal optimal amplitudes. Physical
readout, internal-node actuation, training recurrent weights, larger networks,
gain-rescaled DRNs, and noisy learning remain untested.

## Context

The equations above implement the proposal in the task; no claim of novelty is
made. Related primary sources: [Laborieux and Zenke, Jacobian homeostasis](https://arxiv.org/abs/2309.02214),
[Scurria et al., non-conservative EqProp](https://arxiv.org/abs/2602.03670), and
[Baydin et al., forward gradients](https://arxiv.org/abs/2202.08587). Those works
motivate the mismatch and random-projection context; they are not evidence for
the performance of this particular measurement estimator.
