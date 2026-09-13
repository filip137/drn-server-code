# Jacobian homeostasis, measured corrections, and learned feedback

Analysis of the user-specified [Laborieux and Zenke paper, version 2, April
2024](https://arxiv.org/html/2309.02214v2), prepared 13 September 2026.
The initial 13 September note was read-only. Follow-up controller experiments
are tracked separately in the linked experiment record.

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

## Public implementation audit, 14 September 2026

The author's repository linked from Section 4 was inspected independently.
GitHub's repository and branch APIs identify `main` as the default and only
listed branch, at commit
`30592f576bd4d4a20d3c13632f0792b0fa452781` (commit timestamp 5 September 2023).
The following observations concern that public snapshot; it is not established
that it is the exact source used for every revised-v2 experiment.

In [the regularizer implementation](https://github.com/Laborieux-Axel/generalized-holo-ep/blob/30592f576bd4d4a20d3c13632f0792b0fa452781/models/dyn.py#L392-L422),
two forward-mode Jacobian-vector products supply the two terms, and the sample
loss is divided by the total number of state variables. Five draws are averaged
in each training batch, consistent with Section 4 of the paper. Parameter
gradients are computed with the already obtained free state supplied as a
separate, fixed argument. Thus this is a partial derivative at the operating
point, not a total derivative through the free relaxation. The observed
normalization gives an expected penalty of \(2\|A\|_F^2/n\), whereas Equation
12 states the unnormalized objective. These distinctions matter when reproducing
regularization coefficients or comparing widths.

The Jacobian's coordinates also deserve care. The code first activates the free
state and then differentiates a vector-field variant that bypasses its activation.
The [MLP force definition](https://github.com/Laborieux-Axel/generalized-holo-ep/blob/30592f576bd4d4a20d3c13632f0792b0fa452781/models/vfs.py#L46-L71)
and [elementwise activation](https://github.com/Laborieux-Axel/generalized-holo-ep/blob/30592f576bd4d4a20d3c13632f0792b0fa452781/models/act.py#L7-L11)
make this explicit. Write the raw force as
\(F_u(u)=H(\phi(u))-u\), absorbing the input and bias into \(H\), and set
\(r=\phi(u)\), \(D=\operatorname{diag}\phi'(u)\). At a locally invertible
activation branch, the three relevant Jacobians are

\[
J_u=H'(r)D-I,\qquad
J_{\rm equilibrium}=\partial_r[H(r)-\phi^{-1}(r)]
=H'(r)-D^{-1},\qquad
J_{\rm code}=H'(r)-I.
\]

The last two differ only on the diagonal and therefore have exactly the same
antisymmetric part. Consequently the expected code penalty has a consistent
interpretation as homeostasis of the equivalent equilibrium-force Jacobian in
activity coordinates. It should not be identified without qualification with
the Euclidean Jacobian of the raw membrane-state dynamics, nor with the actual
transformed time-dynamics Jacobian \(D J_u D^{-1}\). For the MLP, \(H'=W\),
so this activity-coordinate asymmetry reduces to that of the recurrent weights.
Nonlinear pooling can make \(H'\) state-dependent in the convolutional model.
Finite-draw trace estimators for Jacobians differing by a diagonal can still
differ in variance, despite having the same expectation. This coordinate
interpretation is an analysis of the public source, not an explicit explanation
given by the authors.

Finally, Appendix E.1 says that forward-mode AD supplies the exact EqProp
response unless a finite-nudge estimator is specified. Table 2 accordingly
separates the 84.3% CIFAR-10 result using that response from 81.4% with two
finite-nudge phases. The [source implementation](https://github.com/Laborieux-Axel/generalized-holo-ep/blob/30592f576bd4d4a20d3c13632f0792b0fa452781/models/dyn.py#L266-L319)
differentiates a finite number of subsequent relaxation iterations with respect
to nudge amplitude; absence of finite-nudge bias does not itself remove finite
relaxation error. The paper demonstrates a learning principle in simulation,
not an entirely measurement-only implementation of its headline results.

Our auxiliary circulation controller has a related target but a different role:
its calibration reduces \(2\|K+C\|_F^2\) using time-ordered physical noise
measurements, while the task network retains its original \(K\) for inference.
Doubling the calibrated controller only during anchored task nudging implements
the transpose correction. The calibration descent identity does not claim to
compute the same parameter gradient as the paper's AD regularizer, and the
recording duration must be charged separately from its five digital Gaussian
Jacobian probes.

**14 September follow-up: direct feedback and an auxiliary circuit.** Figure
4e–h of the paper includes an output-to-first-hidden-layer feedback loop. This
is relevant to direct feedback architectures, although its recurrent dynamics
and learned weights differ from fixed random DFA broadcasts. Homeostasis helps
that example without providing missing reciprocal edges. A higher normalized
symmetry score therefore does not establish exact Jacobian symmetry. Appendix
E.1 also states that most reported hEP experiments use forward-mode AD for the
exact nudge derivative; Table 2 separately reports finite-nudge results. The
paper supplies evidence for the objective, not an entirely physical way to
obtain all training signals.

The new [circulation controller](circulation_feedback_theory.md) gives a concrete
physical connection. Write the original local Jacobian as J=H+K, with H symmetric
and K fixed skew, and add an adjustable skew C during calibration. The paper's
objective on the augmented dynamics is then 2||K+C||_F². Our stationary-circulation
update Cdot=-eta*Omega decreases V=||K+C||_F²/2 under the conditions in the theory
note, even though Omega is not generally its Euclidean gradient. It is therefore
a dynamics-based route toward the same symmetry target in this restricted model.

Deployment serves a different purpose: after calibration, disable C in free
inference and use 2C(s-s0) during cost nudging. With C=-K this produces J+2C=J^T
and retains the original free equilibrium. This doubled anchored correction is
the established AsymEP mechanism; the proposed component is how to calibrate its
unknown coefficients. Using C alone would symmetrize the learning dynamics to H,
whose inverse generally does not yield the original network's adjoint.

For fewer measurements, a physical schematic may restrict K and C to L known
skew loop patterns. The new projected calibration learns only L gains from 2L
projection histories. It still requires stochastic excitation of the physical
states, sufficient recording time, and known wiring. This is a structural prior;
it is not a dimension-independent solution for unknown dense non-reciprocity.
Experiments and their status are tracked in
[the circulation experiment record](circulation_feedback_experiments.md).
