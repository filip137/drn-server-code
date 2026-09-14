# On-chip asymmetry: evidence and research direction

Analysis prepared 14 September 2026. The recommendation is to use perturbations
to maintain useful feedback across many learning updates. The first priority is
a measured homeostatic calibration rule if local state clamping and current
readout are available. With equilibrium current injection and output readout,
the immediate alternative is a reusable feedback map learned from vector-valued
probe responses. Neither is yet demonstrated on-chip by our experiments.

The target remains independently trainable forward/backward couplings, as in
the VF/homeostasis Fashion-MNIST model. Earlier fixed-skew and four-known-gain
experiments are historical controls, not evidence that arbitrary untied
non-reciprocity is cheaply correctable.

## What the completed results support

The [frozen Fashion-MNIST comparison](fmnist_feedback_cosines.md) is the most
relevant evidence. It uses 522 dynamical states, two seeds, and the same 27
examples across checkpoints. Those 27 are the intersection of 40 candidates
that settled at every checkpoint. Results are conditional on that screening.

| Frozen weights | VF first-hidden cosine | Four-probe MC | Four-probe projection |
|---|---:|---:|---:|
| Initialization | -0.043 | 0.010 | -0.041 |
| VF after 10 epochs | 0.354 | 0.027 | 0.353 |
| Homeostasis after 10 epochs | 0.965 | 0.169 | 0.964 |

Cosines compare feedback with the true equilibrium adjoint. Parameter gradients
tell a similar story: the mean input-weight gradient cosine is 0.412 for VF on
VF-trained weights and 0.951 for VF on homeostasis-trained weights. These are
different trained networks; homeostasis improved the feedback of the network
it helped train. It did not estimate the adjoint of the unchanged VF checkpoint.

On the separate learned-feedback checkpoint, prediction alone reaches 0.454
first-hidden cosine; adding four fresh MC probes reduces it to 0.030.
Projection retains 0.453. This supports removing the compulsory fresh MC
correction, but the predictor itself is still inadequate. Its earlier 0.985
gradient cosine on a fixed-skew digits toy did not transfer to this model.

The [probe-error audit](../simulation_results/fmnist_homeostasis_20260914/probe_error_seed0.json)
separates measurement from reconstruction. At amplitude 0.01, paired nudges
measure scalar projections with relative error 3.32e-5 against the exact
20-step response. Error against the equilibrium target is about 3.18%, largely
from the finite response horizon. This was a noiseless float64 audit on 16
equilibrated training images, not a hardware noise result.

The [settling audit](../simulation_results/fmnist_homeostasis_20260914/settling_seed0.json)
also matters: 8.4% of the first 500 training images failed the force-residual
criterion on the homeostasis checkpoint after 150 synchronous steps. After
1,500 synchronous steps, 5.8% remained in period-two behavior. Damping reduced
the failure fraction to 0.2%. The corresponding VF fractions were 9.8%, 0.4%,
and 0%. Damping can change the selected basin; it was not used in training.
Homeostasis is therefore an alignment result on checked equilibria, not a
general convergence guarantee.

The full 20-run accuracy grid was cancelled on user steering: seven runs
completed, four were partial, and nine never started. No claim about a completed
five-seed accuracy comparison follows. The older MNIST comparison also had a
recurrent norm cap active on almost every update, limiting its relevance here.

## Why reducing a fresh correction to a few probes has stalled

Use activity coordinates, with equilibrium force G(r), J=partial_r G,
R=J^-1, cost derivative c, VF response q=Rc, and adjoint lambda=R^T c.
For unit-norm random signs z, E[zz^T]=I/n. An ideal probe measures

\[
y=c^\mathsf{T}Rz=z^\mathsf{T}\lambda.
\]

Given a baseline b fixed before the fresh probes, the unbiased estimator is

\[
\widehat\lambda=b+\frac{n}{m}\sum_{k=1}^m
z_k(y_k-z_k^\mathsf{T}b),\qquad
\mathbb E\|\widehat\lambda-\lambda\|^2
=\frac{n-1}{m}\|b-\lambda\|^2.
\]

At n=522 and m=4, the variance factor is 130.25. The oracle-projection audit
measured 128.59 over 128 trials. Those variance trials deliberately exclude
finite-nudge and read-noise errors: the dimensional problem already exists in
ideal measurements.

Shrinking the fresh correction is a sensible control, but has limited upside
without structure. Let kappa=(n-1)/m and multiply the correction by alpha. Its
ideal MSE relative to baseline squared error is

\[
(1-\alpha)^2+\alpha^2\kappa.
\]

The optimum is alpha=m/(n-1+m), giving only a 4/525, or 0.762%, expected MSE
reduction for four probes here. Four uniformly sampled orthogonal directions
give an expected 4/522 reduction by projection. These are full-vector MSE
statements, not layerwise cosine guarantees. Read noise further reduces the
case for a large fresh correction.

Estimating n unrelated adjoint components from fewer than n scalar linear
measurements is underdetermined without additional structure. A small output
dimension helps parameterize a reusable map; it does not reveal that map's
unknown state-space directions for free. The useful question is how much
calibration can be reused as inputs and weights change.

## Priority 1: perturbation-driven homeostasis with local access

[Laborieux and Zenke](https://arxiv.org/html/2309.02214v2) motivate reducing
functional Jacobian asymmetry and optimize a stochastic penalty using AD.
Our inspected, pinned [source implementation](https://github.com/Laborieux-Axel/generalized-holo-ep/blob/30592f576bd4d4a20d3c13632f0792b0fa452781/models/dyn.py)
bypasses the activation inside the regularizer's JVPs. For the reproduced MLP,
its expected normalized penalty is

\[
H(W)=\frac{1}{2n}\|W-W^\mathsf{T}\|_F^2,
\qquad \partial_W H=\frac{2}{n}(W-W^\mathsf{T}).
\]

This simplification is specific to these force coordinates and parameterization.
The general paper objective need not be ordinary weight matching. In our model,
G(r)=Wr+Bx+b-phi^-1(r), so J=W-D^-1 and J-J^T=W-W^T; D=diag(phi'(u)).
The raw-voltage drift Jacobian W D-I is a different matrix.

A candidate physical replacement uses paired **clamped activity** perturbations:

\[
v_z=\frac{G(r+h z)-G(r-h z)}{2h}\approx Jz,
\qquad
\widehat g_{ij}=2(v_{z,i}z_j-z_i v_{z,j}).
\]

For the same unit-norm random-sign convention,

\[
\mathbb E[\widehat g_{ij}]
=\frac{2}{n}(J_{ij}-J_{ji})+O(h^2)
=\partial_{W_{ij}}H+O(h^2).
\]

Update each existing directed coupling by -eta times this estimate, averaged
over a calibration window. Each reciprocal pair needs the two endpoint probe
signals and measured force changes, not the opposite weight's stored value.
Only the known physical edge support receives updates. The estimator matches
the **mean** homeostatic gradient; its finite-probe noise differs from the
authors' stochastic AD gradient. With equal programming gains, paired updates
preserve the symmetric part of W, but this does not preserve recurrent inference.

This is a derived candidate for the current untied-weight model. It is related
to [weight mirrors](https://proceedings.neurips.cc/paper_files/paper/2019/file/f387624df552cea2f369918c5e1e12bc-Paper.pdf),
which use correlations between perturbations and responses to learn reciprocal
feedback. That work supplies precedent for learning feedback from activity;
it does not validate this recurrent on-chip implementation or its noise budget.

The decisive hardware requirement is access to the force change: clamp states
and measure the required holding currents, or isolate a layer and measure its
current response. A settled voltage displacement from our existing current
nudge gives Rz, not Jz, and cannot be substituted in this update. A transient
version would need calibrated capacitances/time constants and sampling times.
For nonlinearly programmed physical couplings, the known local conversion from
parameter changes to force changes must also enter the update.

A new [eight-state identity check](../labs/tools/check_measured_homeostasis_identity.py)
uses an untied PaperNet, actual finite force evaluations, and eight orthogonal
signed directions. Across amplitudes 0.01, 0.001, and 0.0001, every recurrent
parameter block matches the mean homeostatic gradient within 6.6e-13 maximum
absolute error. It uses 16 clamped configurations and 128 scalar current reads
per amplitude. This checks signs, normalization and the objective connection;
it does not establish low-probe calibration, robustness or Fashion-MNIST gains.

With few random patterns, even a symmetric network can receive noisy updates.
Test accumulated orthogonal patterns, averaging, decreasing calibration steps,
read noise and programming mismatch. Locality makes accumulation plausible;
it does not remove the number of measurements needed for arbitrary couplings.

## Priority 2: reusable feedback from equilibrium probes

If only equilibrium injection and voltage readout are available, start here.
Let P select the ten dynamical outputs and c=P^T e. Define

\[
L=R^\mathsf{T}P^\mathsf{T},\qquad\lambda=Le.
\]

One paired probe produces the output vector v=P Rz=L^T z. Our current scalar
predictor fit instead keeps only e^T v. Retaining all ten outputs teaches all
columns from the same pair of physical experiments. A candidate map update is

\[
\widehat L\leftarrow\widehat L+
\eta\frac{z(v-\widehat L^\mathsf{T}z)^\mathsf{T}}{z^\mathsf{T}z}.
\]

Use the fitted map for task updates and refresh it periodically, with no
mandatory unshrunk MC correction. Keep the existing local activation-slope
features as an explicit modeling choice. A fixed shared map is approximate
because R changes with the operating point. First test its representation
limit using a privileged oracle fit on calibration examples and held-out
evaluation; that oracle is a diagnostic ceiling, never an on-chip learner.
Then compare measured scalar and vector fits at equal cumulative probe budgets.

This is closest to [Forward Direct Feedback Alignment](https://arxiv.org/html/2212.07282v4),
which keeps vector output derivatives and averages learned feedback, and to
[learning synthetic feedback from node perturbations](https://arxiv.org/abs/1906.00889).
FDFA's forward AD derivative is not already a physical equilibrium measurement;
replacing it by paired equilibrium probes is an adaptation to test.

The unstructured map has 522x10=5,220 coefficients at this width. It requires
separate output measurements and a path that distributes learned feedback to
internal learning sites. Calibration, storage, refresh and broadcast cost must
be counted. One probe gives ten equations with a shared state direction; it
does not imply that ten probes recover an arbitrary map.

The VF local rule remains explicit in both routes:
g_theta=-(partial_theta F)^T ell, plus any direct readout gradient. In route 1,
ell remains the measured VF response while a separate homeostatic update repairs
the weights. In route 2, ell is the predicted adjoint. For a directed synapse,
the task gradient is -ell_i r_j. Neither route eliminates the local
parameter-force mapping. Route 2 is learned feedback combined with that local
rule, rather than ordinary EqProp response alone.

## A fallback when only a scalar calibration score is available

The measurable reciprocal-response discrepancy

\[
d(z)=c^\mathsf{T}Rz-z^\mathsf{T}Rc,
\qquad n\,\mathbb E_z[d(z)^2]=\|R^\mathsf{T}c-Rc\|^2
\]

can be averaged over inputs and fixed output-error directions to define a
calibration objective. A parameter perturbation method such as
[SPSA](https://www.jhuapl.edu/SPSA/) could optimize physical controls using two
perturbed objective evaluations. Each objective evaluation itself needs error
and random-direction response measurements: this is not two total nudges.
Use the same examples and probe directions at both parameter settings.

This is a response-discrepancy objective, not the paper's Jacobian penalty.
It can fall when the circuit merely loses sensitivity, so track response gain,
gradient norms and free prediction changes. Dense parameter-space SPSA also
moves the dimensional burden into a much larger space. It is a secondary
baseline for a declared set of programmable controls, not the lead method for
all 136,192 recurrent weights here. A small control set must be physically
justified and tested for residual mismatch; do not assume a four-gain prior.

Changing the network to make its own VF rule accurate and compensating an
unchanged asymmetric network are different objectives. If preserving inference
is essential, the [AsymEP correction](https://arxiv.org/html/2602.03670v2)
provides a target for a separately calibrated learning-phase circuit, but
requires realizing the missing antisymmetric Jacobian action. Our old structured
controller evidence does not solve that problem for unrestricted untied weights.

## Next experiment and decision criteria

Run a calibration-only pilot on copies of the existing initialization and VF
epoch-10 checkpoints, with the homeostasis endpoints as reference outcomes.
Keep the untied architecture. Use separate calibration and held-out cohorts
drawn from training data; do not tune on the already examined official test
examples. No classifier-training epochs are needed for this pilot.

1. Establish a residual-controlled damped solver and separately check response
   settling. Record failures and basin changes; do not discard failures silently
   or restore the previous global norm cap.
2. For physical repair, compare the digital homeostatic update with the measured
   local update at matched calibration settings. Recompute the true task gradient
   of each resulting network as an offline diagnostic. Measure free-output drift
   and task loss as well as feedback alignment.
3. For compensation of fixed weights, compare prediction alone, scalar-fit and
   vector-fit feedback at identical total probe counts. Assess held-out inputs
   and freshness after controlled weight changes. Keep current MC/projection
   results as baselines and distinguish representational error from fit error.
4. Plot first-hidden feedback and every parameter-block cosine, relative error,
   norm ratio, and convergence coverage versus **cumulative physical work**.
   Count calibration, refresh, audit probes, all read channels, settling time,
   total excitation and parameter writes. A clamped-current experiment and an
   equilibrium phase are different resources.
5. Only candidates with useful held-out alignment should progress to noise,
   width and drift sweeps. A practical proposed gate is mean first-hidden and
   input-gradient cosine above 0.9 with controlled magnitude error and no
   collapse of sensitivity; this is a decision target, not an observed result.
   Then test whether the amortized measurement cost stays acceptable as width
   grows. Short training runs follow only after this mechanism test passes.

The central research question is whether local or reusable physical structure
lets a small ongoing calibration budget maintain useful feedback. The present
results establish a reason to pursue that question, not a demonstrated cheap
on-chip solution.

Reproduce the new algebra check:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DISABLE_SHM=1 \
  /home/filip/miniconda3/envs/py312/bin/python -m labs.tools.check_measured_homeostasis_identity \
  --output simulation_results/on_chip_direction_repeat/clamped_homeostasis_identity.json
```
