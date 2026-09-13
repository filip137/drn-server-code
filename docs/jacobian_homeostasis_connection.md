# Jacobian homeostasis, measured corrections, and learned feedback

Analysis of the user-specified [Laborieux and Zenke paper, version 2, April
2024](https://arxiv.org/html/2309.02214v2), prepared 13 September 2026.
No training experiment was launched for this note.

The paper separates finite-nudge bias from Jacobian-asymmetry bias. Holomorphic
EP addresses the former; Jacobian homeostasis regularizes the latter. Its
Equation 12 estimates

\[
\mathcal L_{\rm homeo}
=\mathbb E_\varepsilon[\|J\varepsilon\|^2-\varepsilon^{\mathsf T}J^2\varepsilon]
=2\|A\|_F^2,\qquad A=(J-J^{\mathsf T})/2.
\]

The experiments use five Gaussian draws per sample. These are Jacobian-product
probes, distinct from our equilibrium-force probes. The Cauchy construction
requires a holomorphic equilibrium branch and complex nudges; its exact integral
does not remove the asymmetry error. See Sections 3.1–3.4 and 4 of the paper.

**Derived connection to our measurements.** Use our sign convention F(s)-p=0,
R=J^-1, q=Rc, lambda=R^T c, and delta=lambda-q. The paper's neuronal-error symbol
has a different sign and meaning from our discrepancy delta. Matrix algebra gives

\[
R^{\mathsf T}-R=R^{\mathsf T}(J-J^{\mathsf T})R,
\qquad \boxed{\delta=2R^{\mathsf T}ARc.}
\]

Thus our original swapped-response diagnostic measures the effect of local
asymmetry after weighting by the equilibrium susceptibility and current error.
For isotropic random signs z with E[zz^T]=I,

\[
d(z)=c^{\mathsf T}Rz-z^{\mathsf T}Rc=z^{\mathsf T}\delta,
\qquad \mathbb E[d(z)^2]=\|\delta\|^2.
\]

This differs from measuring ||A||_F^2. Conditioning and error direction matter:
even small A can have a large effect if R is large. For example,
||delta|| <= 2 ||R||_2^2 ||A||_2 ||c||. Conversely, fixed nonzero A may have little
effect on a particular example. The corresponding response penalty would be
||R-R^T||_F^2, another distinct objective.

The original q-initialized MC estimator has ideal MSE
((n-1)/m)||delta||^2. If a network modification halves ||delta||, one quarter as
many probes attains the same absolute MSE, ignoring integer rounding, read noise
and finite-nudge bias. This reduces a variance coefficient; it does not remove
the dimensional scaling. For a learned baseline b, the relevant coefficient is
||lambda-b||^2, so the effect of a Jacobian regularizer must be measured rather
than inferred from its effect on q.

**A constraint of the current implementation.** The toy classifier's
[model and update](../labs/recurrent_eqprop.py) use a trainable symmetric S and
fixed skew K. Its [Jacobian](../labs/random_nudge_hopfield.py) is

\[
J=S+K-D(s),\qquad D(s)=\operatorname{diag}(1+3\,\mathrm{cubic}\,s_i^2).
\]

Consequently A=K and L_homeo=2||K||_F^2 are constant with respect to all current
trainable parameters, including their effect through equilibrium states.
Adding the exact regularizer has zero gradient. A finite-sample trace estimator
can still generate stochastic gradients, but their expectation is zero here.
An increasing normalized symmetry score could arise from a changing denominator
without reducing this constant penalty.

This also means the existing learned-predictor result cannot be attributed to
reducing absolute Jacobian asymmetry. The predictor H changes the feedback
estimate. Changes in S, D and c can change the inverse-weighted discrepancy
2R^T K Rc even while K stays fixed.

**How the approaches fit together.** Our
[DFA connection note](direct_feedback_alignment_connection.md) describes learning
a reusable error-to-node map and using it directly for task updates. Homeostasis
would instead alter the physical network so its intrinsic feedback is more
useful. They could be combined, but their objectives and costs remain separate.

For a meaningful comparison, use independently trainable directed couplings
while retaining a stability constraint on the symmetric part, or use a model
with adjustable state-dependent off-diagonal Jacobian entries. Compare the same
parameterization with and without regularization; retain fixed-K runs as a
control for persistent non-reciprocity. In the simple trainable-K variant, the
penalty is just proportional to ||K||_F^2 and can drive K toward zero. Success
there alone would not establish compensation for immutable hardware asymmetry.

For a DRN with a known positive diagonal reciprocity metric M satisfying
J^T M=MJ, the transformed Jacobian M^(1/2) J M^(-1/2) is symmetric already.
Any proposed regularizer should respect that coordinate structure and target
residual mismatch after gain calibration, rather than penalizing raw-coordinate
asymmetry that the transformed nudge/readout can already account for.

One proposed physical alternative is to minimize measured squared discrepancies
over task examples. Estimating such an objective is already possible with our
probes, but obtaining its parameter gradient requires additional work. Merely
measuring a scalar penalty does not supply a free local learning rule. Absolute
response penalties can also decrease by suppressing susceptibility, so task
accuracy, response magnitude and gradient alignment must be checked together.

For the immediate goal of fewer probes in the existing toy, direct use of the
learned predictor and vector-response fitting remain applicable. A Jacobian
homeostasis comparison requires the model change above. Neither combined
training nor a physical implementation of the regularizer has been tested here.
