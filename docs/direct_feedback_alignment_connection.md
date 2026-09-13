# Direct feedback alignment and fewer equilibrium probes

Research connection and read-only diagnostic, 13 September 2026. Worktree
`server_code_random_nudge`, branch `codex/hopfield-random-nudge-adjoint`.
The new numerical results below use frozen checkpoints; no lower-probe training
trajectory has been run for this note.

The most promising next experiment is to use probes to maintain a reusable
feedback predictor, and use that predictor directly for network updates. The
current learner already has such a predictor, but adds a fresh unbiased Monte
Carlo residual correction to every update. Removing that requirement accepts
bias in exchange for fewer measurements and potentially much less noise.

**Connections to published work.** Nøkland's [Direct Feedback Alignment
(2016)](https://arxiv.org/abs/1609.01596) broadcasts output errors through fixed
random feedback matrices, with local activation derivatives. The forward network
adapts while those matrices remain fixed. An exact adjoint is not a requirement
of that algorithm. Its results do not establish convergence for our recurrent
equilibrium model with tied symmetric trainable couplings and fixed skew drift.

Bacho and Chu's [Forward Direct Feedback Alignment
(FDFA)](https://arxiv.org/html/2212.07282v4) is a closer match: it estimates
output-response derivatives using forward-mode automatic differentiation,
exponentially averages the resulting feedback matrices, and uses them for task
updates. Equations 7–8 retain vector output derivatives. The momentum in that
construction averages feedback estimates. Our selected momentum averages network
parameter gradients; our predictor instead uses relaxed normalized regression.
Physical force nudges measure equilibrium responses, a different derivative
from feedforward activation perturbations.

Lansdell, Prakash and Kording's [Learning to solve the credit assignment problem
(ICLR 2020)](https://arxiv.org/abs/1906.00889) also fits synthetic feedback from
node-perturbation estimates and uses that feedback to train network weights.
Its fixed-forward-weight consistency results and co-adaptation experiments
support this general route without proving our particular implementation.

**The map our probes can teach.** Let P select the o output states, let
e be the cost derivative with respect to those states, and write c=P^T e.
With F(s)-p=0, J=partial_s F and R=J^-1, define

\[
L=R^{\mathsf T}P^{\mathsf T}\in\mathbb R^{n\times o},
\qquad \lambda=Le.
\]

One unit-norm signed probe u, measured through paired equilibria, yields

\[
r_u=\frac{s(hu)-s(-hu)}{2h}\approx Ru,
\qquad v=Pr_u\approx L^{\mathsf T}u.
\]

The current implementation reduces this o-vector to y=e^T v. For predictor
fitting, preserving v gives one linear equation for each output, from the same
two equilibrations. All ten output responses are already present in our
simulator's readings. This does not promise a tenfold reduction in probe count:
the equations share a probe direction, and arbitrary dense map identification
still requires n independent directions at a fixed operating point. For our
softmax loss, only the nine-dimensional output-error contrast subspace affects
the task gradient, so a task-specific fit could discard the common output mode.

A direct normalized regression update for a predicted map L_hat is

\[
\widehat L\leftarrow\widehat L+
\eta\frac{u(v-\widehat L^{\mathsf T}u)^{\mathsf T}}{u^{\mathsf T}u}.
\]

For compatibility with our existing local-slope predictor, retain

\[
D=\operatorname{diag}(1+3\,\mathrm{cubic}\,s_i^2),\qquad
b=D^{-1}(H-P^{\mathsf T})e.
\]

Set a=D^-1 u. Its response prediction is v_hat=H^T a-Pa, giving the candidate
vector update

\[
\boxed{H\leftarrow H+
\eta\frac{a\,[v+Pa-H^{\mathsf T}a]^{\mathsf T}}{a^{\mathsf T}a}.}
\]

This derived adaptation can learn without labels: its targets are output
responses. A shared H remains approximate as inputs and weights change. The
existing scalar fit instead updates using the single equation a^T H e=y+e^T Pa.
The vector fit needs separately observed output channels; scalar-only hardware
would have an additional readout requirement. No full J needs to be estimated.

For the proposed task update, predict b before incorporating the current probe,
form g_hat=-(partial_theta F)^T b, update H from measurements for subsequent
batches, and apply the chosen optimizer to g_hat. This retains our local
parameter-force derivatives. It is an equilibrium analogue of learned direct
feedback, and it gives up conditional unbiasedness of each estimated gradient.
The literature does not demonstrate this specific adaptation.

**What the saved predictors already do.** The new
[audit script](../labs/tools/audit_feedback_prediction.py) replayed the first 96
validation examples at each final selected learned-four checkpoint, seeds 0/1/2.
The same frozen model, inputs, free states and predictor were used for every
method within each seed. Exact analytic adjoints are diagnostic references only.
The MC comparison uses ideal linear-response projections plus independent paired
voltage-read noise (sigma=1e-5, h=0.01), 64 draws per seed. It omits finite-nudge
and settling bias. No parameters or predictor coefficients were updated.

| Feedback used for the raw batch gradient | Gradient cosine | Relative gradient error |
|---|---:|---:|
| Local slope only | 0.855 | 0.530 |
| Learned prediction only | **0.985** | **0.175** |
| Learned prediction + 1 fresh MC probe | 0.384 | 2.603 |
| Learned prediction + 2 fresh MC probes | 0.475 | 1.911 |
| Learned prediction + 4 fresh MC probes | 0.604 | 1.338 |

Values are means across checkpoint seeds and, for MC, the 64 draws. The learned
prediction's cosine range is 0.982–0.987. These are batch gradients before
momentum and stability projection. The predictors had each already received
68,940 scalar probe equations during four-probe training. Thus the audit
supports testing direct predictor use, but establishes neither zero-probe
training from initialization nor successful continued training with a frozen H.

The noise increase is consistent with the known ideal identity

\[
\mathbb E\|\widehat\lambda_{\mathrm{MC}}-\lambda\|^2
=\frac{n-1}{m}\|b-\lambda\|^2.
\]

At n=256 and m=4, that factor is 63.75. Unbiasedness therefore has a substantial
variance cost. Removing the correction leaves systematic error that batching
cannot eliminate. The observed high gradient alignment makes that trade worth
testing; lower instantaneous MSE alone would not establish better training.

The script also checked the vector-fit equations on a stable 16-state linear
system with three outputs. Sixteen orthogonal probes recovered the exact
feedback map to norm error 5.15e-16; a relaxed single-row update with eta=0.25
reduced its measured residual to 0.75 of its initial value. This checks algebra and signs, not a
measurement-budget advantage or noise robustness. Full results are in
[frozen_predictor_audit.json](../simulation_results/dfa_connection_20260913/frozen_predictor_audit.json).

**Tests to run next, with separate controls.** Start with the same 256-state
digits setup and optimizer. First remove the current MC correction while
retaining all four scalar probes for H; this isolates direct predictor use.
Then compare scalar and vector predictor fits at one probe per example. Only
after establishing those controls, test periodic calibration and freezing H
after a declared warm-up. A fixed random H from initialization and H=0
(local-only feedback) are useful zero-probe controls. The former is DFA-inspired;
the latter tests how much this toy task can learn from output-local updates.

| Policy | Mean training equilibrations/example |
|---|---:|
| Existing learned-four MC correction | 9 |
| One paired probe per example to train H; task update uses b | 3 |
| One paired probe per example every fourth minibatch; task update uses b | 1.5 after warm-up |
| Freeze H after calibration; task update uses b | 1 after warm-up |

All proposed costs include one free equilibrium and two per paired probe,
exclude digital arithmetic, and require counting the initial calibration in the
total. They are phase counts, not demonstrated accuracy or hardware speedups.
Calibrating on a subset of examples per batch is another way to lower average
cost. Held-out vector-response residuals can test feedback drift and eventually
support adaptive refresh rates; those audits also consume probes.

Compare accuracy versus total training equilibrations including warm-up,
gradient bias/alignment, applied projected steps, and held-out response
prediction. Keep feedback fitting and parameter momentum distinct. Use the
same validation-only optimizer selection budget if retuning is needed; reserve
test data for final evaluation. Successful constant probes per example would
still depend on a reusable approximate map and its drift, not imply constant
cost for identifying unrestricted inverse responses. Wider networks and a
harder task remain necessary follow-ups.

Reproduce the diagnostic from the worktree with fresh output paths:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DISABLE_SHM=1
task_python=/home/filip/miniconda3/envs/py312/bin/python
task_source=simulation_results/nudge_improvements_20260913/confirm_learned_mc4
"$task_python" -m labs.tools.audit_feedback_prediction \
  --checkpoints \
  "$task_source/learned_mc4_seed0_lr0.3_mu0.9/learned_mc4_seed0_lr0.3.npz" \
  "$task_source/learned_mc4_seed1_lr0.3_mu0.9/learned_mc4_seed1_lr0.3.npz" \
  "$task_source/learned_mc4_seed2_lr0.3_mu0.9/learned_mc4_seed2_lr0.3.npz" \
  --output simulation_results/dfa_connection_repeat/frozen_predictor_audit.json
```
