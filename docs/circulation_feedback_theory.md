# Calibrating physical feedback from stationary circulation

Research note, 13 September 2026. This is a proposed combination of established
ingredients. The convergence argument below concerns an ideal slow adaptation
using stationary expectations; it is not a convergence claim for the finite-record
online implementation in [circulation_feedback.py](../labs/circulation_feedback.py).

## Physical construction and learning rule

The current soft-spin model has force

\[
F(s)=-\nabla U_0(s)+Ks,\qquad K^{\mathsf T}=-K,
\]

where the trainable symmetric couplings and local cubic terms belong to \(U_0\).
Save a free equilibrium \(s^0\), set \(x=s-s^0\), and temporarily add a separately
adjustable skew feedback matrix \(C\). During calibration run the physical SDE

\[
dx=f_C(x)\,dt+\sqrt{2T}\,dW,
\qquad f_C(x)=F(s^0+x)+Cx=-\nabla U(x)+(K+C)x.
\]

Here \(U(x)=U_0(s^0+x)-U_0(s^0)-\nabla U_0(s^0)^{\mathsf T}x\).
No Jacobian is needed to implement this force or observe the trajectory.

Measure the antisymmetric lag derivative

\[
\Omega=\lim_{\tau\downarrow0}
\frac{\mathbb E[x(t+\tau)x(t)^{\mathsf T}
-x(t)x(t+\tau)^{\mathsf T}]}{\tau}
=\mathbb E[f_Cx^{\mathsf T}-xf_C^{\mathsf T}].
\]

The proposed calibration rule is

\[
\dot C=-\eta\Omega,\qquad
\eta=\frac{\texttt{learning\_rate}}{2T}.
\]

This is the normalization used by production code. Its window update multiplies
the estimated area rate by the window duration before applying this coefficient.
Each edge needs only its two endpoint voltage histories; the plasticity rule sees
neither force samples nor a reconstructed Jacobian.

After calibration, disable noise and switch off \(C\) during free inference.
During the actual positive and negative cost-nudged phases use

\[
F(s)+2C(s-s^0)-\beta\nabla_s C_{\rm task}(s)=0.
\]

If calibration reaches \(C=-K\), the linearized learning-phase Jacobian is
\(J+2C=J-2K=J^{\mathsf T}\). Thus the measured displacement approaches
\(J^{-\mathsf T}c\). The existing centered contrastive EqProp parameter rule then
recovers the appropriate gradient as nudge amplitude vanishes. Finite nudges and
read noise still cause error. The anchor preserves the free equilibrium at zero
nudge, even though the learning dynamics change.

## Linear Lyapunov derivation

For the OU model write \(B=H+G\), with \(H=H^{\mathsf T}\prec0\) and
\(G=K+C\) skew. Its stationary covariance satisfies

\[
B\Sigma+\Sigma B^{\mathsf T}=-2TI.
\]

Set \(Q=B\Sigma+TI=\Omega/2\). Since \(Q\) is skew and
\(B=Q\Sigma^{-1}-T\Sigma^{-1}\), Frobenius inner products give

\[
\langle G,\Omega\rangle_F
=\langle B,\Omega\rangle_F
=2\operatorname{tr}(\Sigma^{-1}Q^{\mathsf T}Q)
=\tfrac12\|\Omega\Sigma^{-1/2}\|_F^2.
\]

Therefore

\[
\frac{d}{dt}\tfrac12\|K+C\|_F^2
=-\frac{\eta}{2}\|\Omega\Sigma^{-1/2}\|_F^2\le0.
\]

Equality requires \(Q=0\), making \(B=-T\Sigma^{-1}\) symmetric and hence
\(C=-K\). Every skew controller preserves the negative definite symmetric part
of \(B\), so calibration cannot destabilize this continuous-time OU model.
Bounded controller trajectories and the unique zero of the dissipation establish
convergence for the ideal stationary adaptation.

## Nonlinear extension

The result does not require linearizing the cubic model. Assume a smooth,
normalizable stationary density \(\rho_C\) with full support and sufficient decay
for integration by parts. Define its probability-current velocity

\[
v=f_C-T\nabla\log\rho_C,\qquad \nabla\cdot(\rho_Cv)=0.
\]

Stationarity gives \(\mathbb E[\nabla\phi\cdot v]=0\) for admissible scalar
\(\phi\). Also \(\mathbb E[Gx\cdot\nabla\log\rho_C]=-\operatorname{tr}G=0\).
Using \(Gx=v+\nabla U+T\nabla\log\rho_C\),

\[
\boxed{\langle G,\Omega\rangle_F
=2\mathbb E[Gx\cdot f_C]
=2\mathbb E\|v\|^2
=2T\dot S_{\rm prod}.}
\]

Here \(\dot S_{\rm prod}=T^{-1}\mathbb E\|v\|^2\) is stationary entropy
production for this isotropic overdamped diffusion. Consequently the production
normalization gives

\[
\dot V=-\texttt{learning\_rate}\,\dot S_{\rm prod},
\qquad V=\tfrac12\|K+C\|_F^2.
\]

Zero current makes \(f_C\) a gradient field. Its antisymmetric derivative is
\(G\), so full support implies \(G=0\). The current model's uniformly positive
potential Hessian supplies confinement: its symmetric coupling cap is below the
unit leak. Establishing convergence of a particular stochastic adaptation still
requires additional averaging and step-size arguments.

Because training changes only the conservative part, the exact target \(-K\)
is the same across inputs and training epochs. This enables amortized calibration,
but is a favorable property of this experiment, not a general property of DRNs.

## Access, restricted feedback, and limitations

- **Physical resources:** controlled isotropic white-noise injection, local state
  histories, stored free-state anchors, signed feedback couplings, and selectable
  calibration/learning gains. A dense controller has \(n(n-1)/2\) independent
  coefficients. No random adjoint-probe pairs are required after calibration,
  but recording time, state reads, and controller hardware must be charged.
- **Sparse support:** let \(\mathcal P\) project onto an allowed skew edge support
  containing every nonreciprocal edge. Then \(G=\mathcal P G\), and replacing
  \(\Omega\) by \(\mathcal P\Omega\) preserves the Lyapunov identity. Arbitrary
  low-rank or output-only feedback does not inherit this guarantee.
- **Finite lag:** use future-times-present minus its transpose. For
  \(B=-aI+kA\) in two dimensions, finite-lag circulation is proportional to
  \(\sin(k\tau)A/\tau\). Long lags can reverse the update or create false zeros.
- **Integrator:** production uses stochastic Heun with additive noise, separately
  from the equilibrium solver. This remains a finite-step approximation; nonlinear
  discretizations can generate apparent irreversibility even in a conservative
  system. Timestep refinement is necessary.
- **Finite data:** adaptation changes the distribution being sampled. Short
  windows, constant learning rates, and noisy reads can leave error or bias.
  Averaging controller iterates is a practical estimator, not the stationary
  theorem. Independent replicas consume additional physical observation time.
- **Noise metric:** anisotropic or colored noise, unequal mobilities, hidden
  states, and asynchronous reads can invalidate the raw circulation target.
  Detailed balance may correspond to a weighted symmetry. State-dependent skew
  forces also fall outside the constant-\(K\) argument.

## Structured controllers with a few unknown loop gains

Suppose the schematic supplies fixed skew operators \(B_1,\ldots,B_L\),
while their physical gains are unknown:

\[
K=\sum_{j=1}^{L}a_jB_j,\qquad C=\sum_{j=1}^{L}c_jB_j.
\]

If these operators are Frobenius-orthonormal, define \(r=a+c\),
\(h_j=\langle B_j,\Omega\rangle_F\), and update
\(\dot c_j=-\eta h_j\). The residual stays in the known span, so

\[
V=\tfrac12\|r\|^2=\tfrac12\|K+C\|_F^2,
\qquad
\dot V=-\eta\sum_jr_jh_j
=-\eta\langle K+C,\Omega\rangle_F
=-2\eta T\dot S_{\rm prod}.
\]

Thus the ideal stationary argument above survives with only \(L\) learned
scalars. It does not require the projected circulation to equal the full
circulation. For independent nonorthogonal \(B_j\), the same update still
decreases the coefficient error \(\tfrac12\|r\|^2\); only its equality with
the Frobenius error is lost. If the true skew has a component outside the
supplied span, this proof no longer applies to the full residual, and the
controller cannot cancel that component.

Each coefficient statistic is a scalar stochastic-area measurement:

\[
h_j=2\lim_{dt\to0}\frac{\mathbb E[(B_jx)\cdot dx]}{dt}.
\]

For known rank-two wiring
\(B_j=(u_jv_j^{\mathsf T}-v_ju_j^{\mathsf T})/\sqrt2\), with unit orthogonal
\(u_j,v_j\), write \(p_j=u_j^{\mathsf T}x\), \(q_j=v_j^{\mathsf T}x\).
Then

\[
h_j=\sqrt2\lim_{dt\to0}
\frac{\mathbb E[q_j\,dp_j-p_j\,dq_j]}{dt},
\qquad
Cx=\frac1{\sqrt2}\sum_jc_j(u_jq_j-v_jp_j).
\]

Only two projection histories and two corresponding actuator patterns per
loop are needed. The Itô and Stratonovich area expressions coincide because
the correction is proportional to \(\operatorname{tr}B_j=0\). The finite-step
implementation still needs the timestep and noise checks described above.
After calibration the same coefficients are doubled and anchored for EqProp.

The implementation in [loop_circulation_feedback.py](../labs/loop_circulation_feedback.py)
uses \(2L\) mutually orthonormal supplied wiring columns. It learns no dense
matrix or covariance during calibration; a dense controller is materialized
only on return for the existing EqProp simulator. Its accounting charges
\(2L\) projection channels, \(L\) learned coefficients, \(2nL\) fixed wiring
coefficients, and the complete record duration across replicas. Projection
read noise is separate from the ideal physical sensing/injection paths.

This reduction is a physical prior, not information recovered for free.
The patterns must come from independently specified sensor/actuator wiring,
not an SVD, eigendecomposition, or oracle inspection of the unknown \(K\).
A few unknown amplifier gains qualify when each multiplies a known, fixed
coupling pattern and the controller can implement its skew part. A gain
that multiplies an unknown or changing weight matrix does not automatically
give a known fixed basis. Paired signed injection paths must realize skew
feedback accurately; unequal gains, delay, saturation, or unmodeled gain-dependent
nonlinearities can invalidate the assumed operator family. Even a known
unidirectional rank-one coupling has a symmetric part as well as a skew part;
the stability assumptions concern the complete physical drift.

## Relation to prior work

[Scurria et al., *Equilibrium Propagation for Non-Conservative Systems*](https://arxiv.org/abs/2602.03670)
already introduce learning-phase correction using nonreciprocal interactions.
The doubled anchored deployment above uses that established mechanism; the
proposal is to calibrate its missing coefficients from physical circulation.

[Gierlich et al., *Spike-based alignment learning solves the weight transport problem*](https://www.nature.com/articles/s41467-026-74460-8)
use noisy spike timing and anti-Hebbian plasticity to align reciprocal synapses,
including separate calibration phases. Thus noise-based local symmetrization is
not new by itself. The candidate contribution to investigate is the separate
skew controller, its circulation Lyapunov argument, and switching its gain to
obtain gradient-correct learning while retaining asymmetric inference. These
connections do not establish novelty or experimental efficiency.
