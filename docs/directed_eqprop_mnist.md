# Directed vector-field EqProp and Jacobian homeostasis on MNIST

Status: completed and independently verified, 14 September 2026. All 48 declared
full-MNIST trajectories finished, and all 147 available lab tests passed.
Worktree `/home/filip/server_code_random_nudge`, branch
`codex/hopfield-random-nudge-adjoint`.

The user requested that the tests reflect vector-field EqProp and Jacobian
homeostasis. This supersedes the fixed-four-gain MNIST study described in
`mnist_feedback_experiments.md`. That study was cancelled before any final
training trajectory completed; its partial files and `cancellation.json` remain
under `simulation_results/mnist_eqprop_20260914/main`. They are not final results
for the new comparison.

## What changes

The new model has two hidden layers and a dynamical output layer, with **separate,
trainable matrices in both directions** between adjacent layers. No fixed skew
matrix, known four-dimensional asymmetry basis, or pre-calibrated controller
remains. Forward and backward matrices are untied even when initialized as
transposes. Their asymmetry can evolve through both task learning and
homeostasis.

The model uses the membrane equation and local directed-synapse update from
[Scellier et al., 2018](https://arxiv.org/html/1808.04873), and the initialization
mixture, shifted sigmoid, activity response and homeostasis estimator from
[Laborieux and Zenke, 2024](https://arxiv.org/html/2309.02214v2). The implementation
also follows a coordinate detail in the authors' public code at commit
`30592f576bd4d4a20d3c13632f0792b0fa452781`, in
[`models/dyn.py`](https://github.com/Laborieux-Axel/generalized-holo-ep/blob/30592f576bd4d4a20d3c13632f0792b0fa452781/models/dyn.py)
and
[`models/vfs.py`](https://github.com/Laborieux-Axel/generalized-holo-ep/blob/30592f576bd4d4a20d3c13632f0792b0fa452781/models/vfs.py).
This is a MNIST adaptation of these mechanisms, not a reproduction of either
paper's accuracy, full architecture, or optimization schedule.

## Coordinates and the actual learning rule

Let `r = phi(u)`, `phi(u) = sigmoid(4u-2)`, and `D = diag(phi'(u))`.
The physical solver relaxes

```
F(u) = W phi(u) + Bx + b - u
F(u) - external_current = 0.
```

There is a trainable softmax readout of the activated output layer. Let
`c_r = partial C / partial r`, so `c_u = D c_r`. The equivalent activity
equilibrium force and the two local Jacobians are

```
G(r) = Wr + Bx + b - phi^-1(r)
J_u = W D - I
J_r = W - D^-1
J_u = J_r D.
```

Both coordinate descriptions give the same force adjoint:
`lambda = J_u^-T c_u = J_r^-T c_r`. Parameter gradients are
`-partial_theta F^T lambda`, plus the direct cost gradient of the readout.
Every directed synapse `W_ij` uses `-lambda_i * r_j` (averaged over the batch).
The input weights use `-lambda_hidden1 * x`. This is a directed local plasticity
rule; the earlier symmetric-weight energy contrast is not used.

The eight matched methods are:

| Method | Feedback used in the local rule | Additional operation |
|---|---|---|
| `vf_membrane` | Paired membrane displacement under `+/- beta c_u(u)` | Original raw-state VF construction |
| `ep_activity` | Paired activity displacement under `+/- beta c_r(r)` | Matches the homeostasis code's EP response |
| `ep_homeo` | Same activity response | Add the homeostatic parameter gradient |
| `ep_probe4` | Activity EP plus four-probe residual correction | Eight extra equilibrations per example |
| `ep_probe4_homeo` | Same four-probe correction | Also add homeostasis |
| `learned_probe4` | Learned feedback baseline plus four fresh probes | Online measured feedback fit; nine equilibrations |
| `learned_probe4_homeo` | Same learned baseline and correction | Also add homeostasis |
| `adjoint` | Digital solution of `J_u^T lambda = c_u` | Reference, using transposed weights |

For infinitesimal nudges, the first two give respectively `J_u^-1 c_u` and
`q_r = J_r^-1 c_r`. At symmetric **weights**, activity EP agrees with the true
adjoint; raw membrane VF need not, since `W D` need not be symmetric. This
distinction is tested explicitly. At finite beta the error currents are
recomputed at every relaxation step; they are not silently frozen at the free
cost gradient.

## Homeostasis and random probes have different roles

The authors' regularizer bypasses the activation when taking state derivatives,
giving `J_code = W-I` in this MLP. Its antisymmetric part equals that of `J_r`,
although neither matrix equals `J_u`. Five independent Gaussian vectors per
training example estimate

```
L_H = E[(||J_code eps||^2 - eps^T J_code^2 eps) / n]
    = 2 ||(W-W^T)/2||_F^2 / n.
```

We implement the two exact forward JVPs as matrix-vector products and use
automatic differentiation in the parameters to obtain its gradient. This is
the public code's normalized estimator. It does not differentiate through the
free equilibrium. In this MLP the penalty is independent of the operating point;
its gradient is no longer zero because the directed weights are trainable.
The fixed-state, activation-bypassed interpretation is intentional. Raw
membrane-Jacobian regularization would be a different control.

The **physical** correction instead applies four independent unit-norm
random-sign directions `z_i = +/-1/sqrt(n)`. Measured activity displacements
give `r_z ~= J_r^-1 z`, and output changes suffice for `y = c_r^T r_z`.
With `q` from activity EqProp:

```
lambda_hat = q + (n / m) sum_k z_k (y_k - z_k^T q),  m = 4.
```

The `ep_probe4` arms use the plain residual Monte Carlo estimator with measured
EqProp as the baseline. The `learned_probe4` arms preserve the earlier idea of
an online measured baseline, adapted to the current activity coordinates:
`b = D(-c_r + H c_out)`, with an initially zero 138-by-10 matrix `H`. Each scalar
response equation updates H with a normalized least-mean-squares step of 0.25.
The current gradient is formed before these updates, so its fresh-probe residual
correction is conditionally unbiased in the linear/noiseless limit. Read-only
audits do not update H. These arms use no cost-nudged phase, but use the same
local directed parameter-force learning rule with corrected feedback.
Adam is the common **parameter optimizer**; it does not estimate a Jacobian.
No Jacobian,
transpose multiplication, adjoint solve or AD is available to the physical
feedback path. Tests prohibit those calls. Homeostasis and the adjoint reference
are explicitly digital controls and are charged separately.

In the ideal linear-response limit, the correction is unbiased and its
mean squared feedback error is `(n-1)/m * ||lambda-q||^2`. A low probe count can
therefore be noisy, particularly when homeostasis has not reduced the residual.
Lower feedback error is not assumed to imply better task learning.

## Matched experimental design

Declared main coverage after the operational gates: 48 full-MNIST trajectories,
eight methods, initial mixing angles 0/45/90 degrees, two seeds, five epochs.
The two hidden layers have 64 states each, followed by 10 output states:
138 dynamical states, all original 784 image pixels. The readout is 10 by 10.
Use the same 55,000/5,000 training/validation split and official 10,000-image test
set as the previous infrastructure. Pixel normalization is fitted only on the
training split. Use Adam at 0.001, batch size 128, beta 0.01, homeostasis
coefficient 1, five Gaussian vectors per example, zero read noise, float64, and
equilibrium residual tolerance 1e-10. There is no hyperparameter selection or
test-set selection. Evaluate the official test set only at the fixed final epoch.
Zero noise initially separates learning-rule/asymmetry effects from readout error.

Initialization uses `W_backward = cos(alpha) W_forward^T + sin(alpha) G` with
independent Gaussian `G`. Here each reverse draw has the same entry variance as
its matching forward matrix; the paper's software framework can initialize
reverse layers with a different fan-in convention. A common, alpha-independent
scale bounds the initial recurrent norm, preserving identical forward draws
across the angle sweep. Alpha is a mixing parameter, not an exactly enforced
finite-matrix angle; measured pair angles are recorded.
Biases initially make `u=.5` the zero-input fixed point, where `phi(u)=.5`.
That operating-point centering is an additional choice in this adaptation.
The author's beta sign is translated to our `F-b=0` convention; end-to-end
parameter finite differences verify the resulting learning-rule sign.

For reliable convergence in this exploratory CPU experiment, a common scalar
rescaling caps the recurrent operator norm at 0.95 after every parameter update.
Because `max phi' = 1`, the free map is then a contraction. This is an additional
constraint relative to the papers, applied equally to every method. It can also
reduce absolute asymmetry through global scaling, so we record how often it
acts, weight angles, absolute antisymmetry and normalized symmetry. It never
ties entries or projects onto symmetric weights. Nudged equilibria must also
pass the residual check; a cap alone is not used to certify their convergence.

Five epochs, two seeds, the norm cap, finite real central differences, the
MNIST dataset and smaller architecture limit conclusions. In particular, the
paper's complex/Cauchy nudging procedure and full published learning schedules
are not reproduced. This is an exploratory algorithm comparison, not evidence
of a physical-hardware speedup.

## Evidence and accounting

Training counts include free and nudged equilibrations, state/scalar reads,
actual squared error-nudge current and fixed probe current norms, and relaxation
iterations. Standard VF/EP uses three equilibrations per example. The four-probe
methods with measured EqProp as baseline use eleven; the learned-baseline
methods use nine. Homeostasis additionally uses five Gaussian vectors and ten
digital JVPs per example, with parameter AD. The adjoint uses one free equilibrium
plus a separately counted digital transpose iteration. Evaluation and oracle
audits are recorded outside the training budget. Counts do not equate GPU/CPU
operations with physical settling time.

The same 16 training examples are replayed at epochs 0, 1 and final. Audits record
per-layer feedback alignment, each parameter block's task-gradient angle/error,
absolute antisymmetry, and symmetry scores for both `J_code` and `J_u`.
The homeostatic gradient itself is excluded from *task*-gradient alignment.
This avoids letting a well-aligned output/readout block hide a wrong hidden
feedback direction. Activation saturation and projection frequency are also
recorded. Full Jacobians and reference adjoints occur only in these diagnostics
or the explicitly named reference method.

Implementation: `labs/directed_eqprop.py`. Runner:
`labs/tools/train_directed_eqprop_mnist.py`. Scientific tests:
`labs/tests/test_directed_eqprop.py`. Outputs:
`simulation_results/directed_eqprop_20260914/`.

Operational gates: all six methods at both endpoint angles on a 500-image
subset with small hidden layers; all six methods at the main hidden width on
a 2,000-image subset. The two additional learned-baseline controls receive a
separate full-width gate with identical subset size/settings. These gates do
not evaluate the official test set. All run configurations are retained.

Monitoring uses the experiment-run-watchdog skill's exploratory tier: local CPU,
single-thread workers, a parent process handle, per-worker heartbeat every
20 seconds, per-epoch metrics/checkpoints, and required terminal `result.json`.
Source hashes, environment, settings and declared cases are recorded in
`run.json`. Operational failures may be repaired with settings held fixed;
poor accuracy is a scientific result and is not grounds for an undeclared retry.

## Completed results

Training source: commit `65958664e3a308fe2f319e107c1904f159b3d1a0`.
Local launcher session 72801 exited zero. The final run status, 48 final
checkpoints and 48 terminal results agree. Independent NumPy replay of all
5,000 validation and 10,000 test examples matched each checkpoint's reported
accuracy exactly and mean cross-entropy within 1e-9. Verification also checked
the recurrent norm bounds, finite parameters, initial/final symmetry metrics,
training measurement counts and learned-feedback observation counts.

Test accuracy (%), fixed epoch five, mean ± population SD across two seeds:

| Method | Initial mixing 0° | Initial mixing 45° | Initial mixing 90° |
|---|---:|---:|---:|
| Membrane VF | 20.55 ± 0.59 | 20.56 ± 0.73 | 21.11 ± 0.07 |
| Activity EP | 49.45 ± 16.09 | 36.11 ± 0.78 | 34.28 ± 0.25 |
| Activity EP + homeostasis | 71.01 ± 3.58 | 36.25 ± 0.63 | 35.86 ± 0.04 |
| EP + four probes | 74.49 ± 4.57 | 69.36 ± 10.15 | 69.58 ± 10.40 |
| EP + four probes + homeostasis | 75.19 ± 3.98 | 70.76 ± 10.08 | 70.28 ± 10.19 |
| Learned baseline + four probes | 74.40 ± 5.84 | 74.59 ± 5.40 | 75.18 ± 4.78 |
| Learned baseline + four probes + homeostasis | 75.30 ± 5.55 | 75.79 ± 4.59 | 75.92 ± 4.61 |
| Digital adjoint | 73.96 ± 2.28 | 69.53 ± 3.33 | 68.42 ± 2.67 |

The main mechanistic finding is more definite than the accuracy ranking. At
90-degree mixing, adding homeostasis to activity EP reduces final absolute
antisymmetry from 0.524 to 0.231 and increases first-hidden-layer feedback
cosine from 0.712 to 0.998. It therefore improves the intended feedback
alignment while learning independently directed weights. It produces only
a 1.58-point accuracy gain after five epochs at that mixing level. Initial
transpose symmetry is not maintained automatically: at zero mixing, plain
EP develops asymmetry during training, while homeostasis limits that drift.

The raw membrane Jacobian score gives a different ordering from the
activation-bypassed score. This is expected from `J_u=W D-I` versus `J_code=W-I`;
the two should not be interchanged when interpreting these results.

These are modest accuracies for MNIST, including the exact-adjoint reference.
The run is deliberately short and uses one common untuned optimizer setting.
The recurrent cap acts on **96.2–99.1% of parameter updates**, and individual
rescalings can be as small as 0.915. It materially constrains training. The
results establish the implementation and the alignment effect in this setting;
they do not establish a competitive MNIST learner or the paper's performance.
The next comparison needed before general algorithm claims is a longer,
appropriately tuned run with a less restrictive stability treatment. The
present results are retained unchanged, rather than selected or rerun to improve
their accuracy.

The learned baseline's high test accuracy does not imply an accurate
per-example reconstructed adjoint. On the frozen 90-degree seed-zero models,
32 random-probe trials on eight training images reproduce the expected
`(138-1)/m` dimension penalty. At four probes, observed correction MSE is
29.5–35.3 times the corresponding baseline MSE, versus the ideal factor 34.25.
The finite-response error of activity EP at beta .01 is 0.025–0.274% on those
checkpoints, so that test's large sampling variance cannot be attributed to
large finite-nudge bias. The training algorithm averages gradients over 128
examples and uses Adam; this diagnostic does not isolate why its noisy updates
sometimes improve short-run accuracy relative to the reference.

The read-only diagnostic also sweeps beta .1/.01/.001 and probe-read noise
standard deviations 0 and 1e-5. Noise is applied to fresh probe responses while
the free state and baseline remain fixed and noiseless. Smaller beta amplifies
that read noise. It is a conditional measurement diagnostic, not noisy-hardware
training. Nothing in this diagnosis changes checkpoints, learned predictors,
training settings or test selection.

Artifacts under `simulation_results/directed_eqprop_20260914/main/`:

- `report.md`, `summary.csv`, `aggregate.json`: full comparisons and seed values.
- `comparison.png`, `comparison.pdf`, `learning_curves.png`: matplotlib figures.
- `verification.json` and each case's `independent_verification.json`: replay checks.
- `probe_budget_diagnostic.json`, `probe_budget_diagnostic.png`: frozen-checkpoint
  probe-budget/amplitude/noise measurements.
- `run.json`, `dataset.json`, `launcher.json`, `monitor.jsonl`: settings,
  coverage, source hashes, process handle and progress records.
- Each case retains initial/final weights, complete epoch history, checkpointed
  optimizer/RNG/feedback state and read-only gradient/Jacobian audits.

Validation command (147 passed):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DISABLE_SHM=1 \
  /home/filip/miniconda3/envs/py312/bin/python -m pytest -q labs/tests \
  --ignore=labs/tests/test_single_conv.py
```

The omitted pre-existing convolution test cannot be collected because it imports
the unavailable `custom_classes` module. No new EqProp test is skipped.

To reproduce the comparison, from the worktree, using a fresh output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DISABLE_SHM=1 \
  /home/filip/miniconda3/envs/py312/bin/python -m labs.tools.train_directed_eqprop_mnist \
  --output simulation_results/directed_eqprop_reproduction \
  --hidden 64 --epochs 5 --seeds 0 1 --angles 0 45 90 --batch-size 128 --workers 6
```

Run `labs.tools.report_directed_eqprop_mnist OUTPUT` for independent replay and
plots, then `labs.tools.diagnose_directed_probe_budget OUTPUT` for the read-only
probe diagnostic. Exact source and full settings for the completed run are
recorded in its manifest; future code changes should not silently replace them.
