# Calibrating skew feedback loops from swapped equilibrium responses

Theory note, 14 September 2026. This is a proposed calibration rule with an exact
local-response guarantee under the assumptions below. It is not a claim of
novelty, demonstrated hardware performance, or improved task learning.

The idea is to adjust a small number of known feedback-loop gains until swapping
the nudge and readout directions gives the same response. This can calibrate an
auxiliary circuit for asymmetric EqProp without reconstructing a Jacobian,
measuring an entire adjoint, or injecting noise throughout the physical network.

## Fixed operating point and known loop wiring

Hold the input and task parameters fixed. Let the free equilibrium satisfy
\(F(s^0)=0\), and write its Jacobian as

\[
J=H+K,\qquad H=H^{\mathsf T}\prec0,\qquad K=-K^{\mathsf T}.
\]

Assume the physical schematic supplies skew matrices \(B_1,\ldots,B_L\) with
\(\langle B_i,B_j\rangle_F=\delta_{ij}\), and that the unknown skew part is
\(K=\sum_j a_jB_j\). The adjustable auxiliary feedback is
\(C(c)=\sum_j c_jB_j\). Its calibration dynamics use

\[
\dot s=F(s)+C(c)(s-s^0)-b.
\]

The anchor preserves the same equilibrium at \(b=0\) for every controller
setting. Consequently its local Jacobian and equilibrium response are

\[
J_C=H+G,\qquad G=K+C=\sum_j(a_j+c_j)B_j,\qquad R_C=J_C^{-1}.
\]

The condition \(H\prec0\) is stronger than stability of the original \(J\).
It guarantees local stability and invertibility for every skew controller.
For nonlinear forces it does not guarantee global convergence of a finite
nudge, so measurements must stay on the local equilibrium branch.

The wiring must be known independently of the unknown gains: for example, from
specified sensor/actuator paths. Extracting the basis from an oracle Jacobian
or the singular vectors of the true \(K\) would change the information budget.
A few unknown amplifier gains fit this model only when their contributions to
the local skew Jacobian lie in known, fixed patterns. Unknown changing weights,
nonlinear slopes or delays can invalidate that assumption.

## Measurements and update

Define the measured response mismatch

\[
h_j(c)=\langle B_j,R_C^{\mathsf T}-R_C\rangle_F,
\qquad \dot c_j=-\eta h_j(c),\quad \eta>0.
\]

For a rank-two loop with known unit orthogonal directions \(u_j,v_j\),

\[
B_j=\frac{u_jv_j^{\mathsf T}-v_ju_j^{\mathsf T}}{\sqrt2},\qquad
h_j=\sqrt2\left(v_j^{\mathsf T}R_Cu_j-u_j^{\mathsf T}R_Cv_j\right).
\]

Thus nudge in direction \(u_j\) and read along \(v_j\), then exchange those
directions. Joint orthonormality of all \(2L\) wiring vectors is a convenient
sufficient condition for orthonormal \(B_j\); it is not needed for the scalar
measurement identity itself.

With central differences, measure

\[
t_{vu}=\frac{v_j^{\mathsf T}s_C(\epsilon u_j)
                   -v_j^{\mathsf T}s_C(-\epsilon u_j)}{2\epsilon},
\qquad
t_{uv}=\frac{u_j^{\mathsf T}s_C(\epsilon v_j)
                   -u_j^{\mathsf T}s_C(-\epsilon v_j)}{2\epsilon},
\qquad \widehat h_j=\sqrt2(t_{vu}-t_{uv}).
\]

One complete parallel coefficient update therefore needs **two paired probe
directions, four perturbed equilibrations and four scalar readings per loop**,
before averaging or repeated sweeps. Only two selectable projection channels
per loop are required. The free anchor can be measured once and reused while
the operating point is unchanged. A sequential loop update would need its
responses at the current controller; the batch proof below uses all responses
at the same controller setting.

These are unit-norm directions with a total nudge norm of \(\epsilon\), not
dense sign vectors of norm \(\epsilon\sqrt n\). Injection and readout scales
must be calibrated: unknown channel gains can create an apparent response
asymmetry even when the physical system is reciprocal.

## Continuous-time Lyapunov guarantee

Let \(r=a+c\), \(D=-H\succ0\), and
\(M=D^{-1/2}GD^{-1/2}\), so \(M^{\mathsf T}=-M\). Then

\[
R_C=D^{-1/2}(-I+M)^{-1}D^{-1/2}.
\]

For the coefficient error \(V=\tfrac12\|r\|^2\), orthonormality gives
\(V=\tfrac12\|G\|_F^2\), and

\[
\dot V=-\eta\sum_jr_jh_j
       =-\eta\langle G,R_C^{\mathsf T}-R_C\rangle_F.
\]

Skew symmetry and cyclicity of the trace imply

\[
\langle G,R_C^{\mathsf T}-R_C\rangle_F
 =2\operatorname{tr}(GR_C)
 =2\operatorname{tr}\!\left[M(-I+M)^{-1}\right]
 =4\sum_{k}\frac{\alpha_k^2}{1+\alpha_k^2}\ge0,
\]

where the nonzero eigenvalues of \(M\) occur in pairs \(\pm i\alpha_k\).
The last expression vanishes precisely when \(G=0\). Each eigenvalue pair
contributes \(2\alpha_k^2/(1+\alpha_k^2)\) to the trace before its factor of two.

Therefore the ideal continuous flow decreases the true controller error and
converges to \(c=-a\). In particular, its bounded coefficient trajectory lies
in a compact sublevel set of \(V\), and the only invariant set with
\(\dot V=0\) is the correct controller. This conclusion uses exact response
measurements, fixed \(H,K\), and complete representation of \(K\) in the known
loop span.

For linearly independent but nonorthogonal \(B_j\), the same update still
decreases the coefficient norm \(V=\tfrac12\|a+c\|^2\), since
\(\sum_j r_jh_j=\langle G,R_C^{\mathsf T}-R_C\rangle_F\). It no longer equals
the Frobenius error, and the discrete bound below must account for the basis
Gram matrix. If part of \(K\) lies outside the controlled span, this complete
cancellation guarantee does not apply.

## A log-determinant potential

The same flow is a gradient flow for a different objective:

\[
\Phi(c)=\log\det(-J_C)
       =\log\det D+\frac12\log\det(I-M^2).
\]

The determinant is positive because \(D\succ0\) and the skew eigenvalues occur
in conjugate pairs. Differentiating at fixed operating point gives

\[
\partial_{c_j}\Phi=\operatorname{tr}(R_CB_j)=\frac12h_j,
\qquad \dot c=-2\eta\nabla\Phi,
\qquad \dot\Phi=-\frac\eta2\|h\|^2.
\]

The unique minimum in the representable family has \(G=0\), although \(\Phi\)
is not globally convex: even \(\log(1+x^2)\) loses convexity at large \(|x|\).
Neither the determinant nor \(D\) needs to be measured or computed to run the
swapped-response update. They are proof devices. This is not the Euclidean
gradient of the squared Jacobian-asymmetry objective.

## A sufficient finite-step bound

For exact responses and orthonormal \(B_j\), suppose a valid lower bound
\(D\succeq dI\), \(d>0\), is known. Write

\[
Q=M(I-M^2)^{-1},\qquad
\Delta R=R_C^{\mathsf T}-R_C=2D^{-1/2}QD^{-1/2}.
\]

Bessel's inequality and the skew eigenvalue pairs give

\[
\|h\|^2\le\|\Delta R\|_F^2
\le\frac4{d^2}\|Q\|_F^2
\le\frac2{d^2}\langle G,\Delta R\rangle_F.
\]

The last inequality follows from
\(\alpha^2/(1+\alpha^2)^2\le\alpha^2/(1+\alpha^2)\). For a batch update
\(c^+=c-\eta h(c)\),

\[
V(c^+)-V(c)
=-\eta\langle G,\Delta R\rangle_F+\frac{\eta^2}{2}\|h\|^2
\le-\eta\left(1-\frac\eta{d^2}\right)
           \langle G,\Delta R\rangle_F.
\]

Consequently \(0<\eta<d^2\) is sufficient for strict error decrease away from
the solution. It is a sufficient bound, not an optimal step size. If \(d\) is
unknown, the theorem does not supply a usable numerical learning rate without
an additional physical bound or step-size procedure.

## Finite nudges, noise and scaling

For a sufficiently smooth force and a fixed local branch, the central-response
bias is \(O(\epsilon^2)\). If each scalar equilibrium read has independent
variance \(\sigma^2\), and \(N\) independent readings are averaged at each
sign, then

\[
\operatorname{Var}(t_{vu})=\operatorname{Var}(t_{uv})
 =\frac{\sigma^2}{2N\epsilon^2},\qquad
\operatorname{Var}(\widehat h_j)=\frac{2\sigma^2}{N\epsilon^2}.
\]

Correlated read noise, drift and incomplete relaxation change this calculation.
A constant-step noisy update has an error floor; it does not inherit the exact
per-step guarantee. Amplitude sweeps, repeated reads, decreasing steps and
held-out response checks are practical controls. Controller saturation,
imperfect anchors, time delay and injection/readout gain errors also change the
model. Piecewise-linear switching requires a fixed active set or an explicitly
directional interpretation of the measured derivative.

The number of learned scalars is \(L\), and one central-difference sweep costs
\(4L\) equilibrations. This reduces the dependence on the number of physical
states only because the known wiring supplies a structural prior. It does not
make the number of sweeps, read averages or settling time independent of the
network. Strong restoring forces attenuate the measured signal; strong skew
can also produce small updates. For example, with \(D=dI\) and one normalized
rank-two loop of residual gain \(r\),

\[
h=\frac{2r}{d^2+r^2/2}.
\]

Its magnitude decays like \(4/|r|\) at large skew. Convergence can therefore be
slow even for a single unknown coefficient. A dense wiring pattern also needs
physical weighted projections and distributed injection despite the small
number of adaptive gains. Only the plasticity/readout requirement is reduced;
the physical network and its settling cost remain.

## Deployment in EqProp

After calibration, disable the auxiliary controller during ordinary free
inference. At the new free equilibrium \(s^0\), use the doubled anchored
controller in the cost-nudged dynamics:

\[
F(s)+2C(s-s^0)-\beta\nabla_s\mathcal C(s)=0.
\]

With \(C=-K\), its nudge derivative at \(\beta=0\) satisfies

\[
\left.\frac{ds}{d\beta}\right|_0
=(J+2C)^{-1}\nabla_s\mathcal C
=J^{-\mathsf T}\nabla_s\mathcal C=\lambda.
\]

The parameter gradient still requires the local parameter-force map,
\(- (\partial_\theta F)^{\mathsf T}\lambda\), or its corresponding EqProp
readout. Calibrating a controller does not replace that learning rule. Using
\(C\) rather than \(2C\) during the task nudge would produce \(H^{-1}\nabla_s
\mathcal C\), which is generally not the original adjoint. The controller and
anchor are held fixed while measuring the nudge response.

Calibration can be amortized over examples only when the same \(K\) and wiring
remain valid. A nonlinear force with fixed additive skew and changing symmetric
part has this useful property. State-dependent skew generally requires renewed
calibration. The local proof at one anchor alone does not establish transfer
across inputs or task updates.

## Connection to prior work

[Laborieux and Zenke's Jacobian homeostasis](https://arxiv.org/html/2309.02214v2)
penalizes Jacobian asymmetry with Gaussian Jacobian-vector probes and automatic
differentiation. For the augmented family here, that objective is
\(2\|K+C\|_F^2\), with the same cancellation target. The proposed rule instead
adjusts separate controller gains using measured inverse-response asymmetry.
Its potential is \(\log\det(-J_C)\), and the Frobenius error decreases by the
identity above. See the [paper and public-code audit](jacobian_homeostasis_connection.md)
for coordinate and implementation distinctions.

The doubled anchored deployment is the established correction in
[Scurria et al., AsymEP](https://arxiv.org/html/2602.03670v2).
The proposed component is calibrating its unknown coefficients through swapped
physical responses, not the transpose correction itself.

[Spike-based alignment learning](https://www.nature.com/articles/s41467-026-74460-8)
already learns reciprocal connectivity through noisy asymmetric spike timing,
and discusses combining it with spiking EqProp. Our earlier
[circulation-controller derivation](circulation_feedback_theory.md) uses
time-ordered stochastic areas for a related cancellation target. This note
uses controlled equilibrium responses and requires neither spontaneous
fluctuation records nor a noise covariance model.

[Perturbative Contrastive Physical Learning](https://arxiv.org/pdf/2606.09756)
is a broader precedent for learning from physical response contrasts; its
Section II.B explicitly describes measured parameter-response Jacobians and
external pseudoinverse computation. It does not supply the specific skew-loop
response rule or its cancellation proof above. A bounded search found no exact
match for this combination, which is not evidence sufficient to establish
novelty. Its practical value still needs calibration measurements at matched
nudge, readout and settling budgets, followed by separate learning experiments.
