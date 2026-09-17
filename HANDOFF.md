# Handoff: population programming-error hardware-aware training

## Purpose

This document records the **theoretical HWA contract** used for the IBM-OM
standard-crossbar comparator so that it can be transferred to a different
network. It deliberately describes the stochastic model and training semantics,
not the repository-specific implementation.

The central idea is simple:

> Keep a clean, globally normalized FP32 master weight; draw a fresh
> target-conditioned programming error for every analog weight on every
> minibatch; use the resulting apparent post-write weight for that minibatch's
> forward pass; pass gradients to the clean master with a straight-through
> estimator; and project the updated master back into the legal global range.

This is population-level programming-error HWA. It is not training on one fixed
array, not a simulation of programming pulses during training, and not a model
of inference-time read noise.

## 1. State and notation

For every trainable scalar that will be represented by an analog crossbar state,
define:

- \(q\): the clean FP32 master state, normalized to \([-1,1]\);
- \(x=(q+1)/2\): the corresponding programming target in \([0,1]\);
- \(x_{\mathrm{app}}\): the apparent normalized state seen after programming;
- \(\epsilon_x=x_{\mathrm{app}}-x\): the apparent endpoint residual;
- \(\epsilon_q=2\epsilon_x\): the same residual in the signed \(q\) coordinate;
- \(\alpha_e\): the programming-error strength in epoch \(e\).

The clean master is the state optimized and checkpointed. A noisy forward state
is temporary and is never written back into the master.

For the standard-crossbar comparator, the forward-relevant OM state is the
apparent effective state \(q=a-r\). Hidden persistent device state is important
for later pulse evolution and robustness diagnostics, but it is not substituted
for the apparent state in the MVM forward pass.

## 2. The programming-error distribution

### 2.1 What was fitted

The new HWA policy did not assume a zero-mean Gaussian. It used a conditional
endpoint-error law

\[
  \epsilon_x \sim K(\epsilon_x\mid x),
\]

where the distribution changes with the requested target \(x\). This permits
target-dependent bias, variance, skew, and support.

The kernel was fitted from a population of characterized program-and-verify
trajectories. Only endpoints that were:

- accepted by the verify rule,
- from healthy/non-corrupt trajectories, and
- represented in the apparent forward state

entered this HWA residual model. Consequently, this is more precisely

\[
  p(x_{\mathrm{app}}-x\mid x,\ \text{accepted},\ \text{non-corrupt}).
\]

Programming failure probability, unreachable targets, stuck/corrupt devices,
and hidden persistent-state error were intentionally excluded. They belong to
separate deployment and robustness controls rather than being silently folded
into the ordinary programming-noise distribution.

### 2.2 The frozen model used in this study

The characterized model had:

- 41 target locations spanning \([0,1]\);
- an eight-bin residual histogram at every target;
- adaptive lower-to-target programming with step parameter \(0.5\);
- a maximum of 128 pulses;
- an apparent-state acceptance tolerance of \(0.023725\).

For a target between characterized locations \(x_i\) and \(x_{i+1}\), with

\[
  \lambda=\frac{x-x_i}{x_{i+1}-x_i},
\]

the two neighboring categorical probability vectors and corresponding residual
bin edges were linearly interpolated. The probabilities were renormalized, a
residual bin was drawn from that categorical distribution, and the residual was
drawn uniformly within the selected interpolated bin.

In symbols, for residual bin \(j\),

\[
  p_j(x) \propto (1-\lambda)p_{i,j}+\lambda p_{i+1,j},
\]

followed by

\[
  J\sim\operatorname{Categorical}(p(x)),\qquad
  \epsilon_x\sim\operatorname{Uniform}(b^-_J(x),b^+_J(x)).
\]

This interpolation makes the error law continuous in target, while retaining
the non-Gaussian shape learned from endpoint populations.

The frozen source artifact was
`data/ibm_reram_om_pv128_hwa_v1.json`, with SHA-256
`3030e04d6205dc90d0894ac453d2c1c522dffdc004f69b9c6ab7eaf7ef4b8ba3`.
It is based on the pinned AIHWKit 1.1.0 OM ReRAM model, not raw measured-device
data. Any device-facing conclusion must retain that qualification.

## 3. How noise was injected during training

### 3.1 Epoch ramp

The error amplitude was ramped rather than enabled at full strength
immediately:

\[
  \alpha_e = \alpha_0+
  \min\!\left(\frac{e}{E_{\mathrm{ramp}}},1\right)
  (\alpha_1-\alpha_0).
\]

The frozen values were \(\alpha_0=0\), \(\alpha_1=1\), and
\(E_{\mathrm{ramp}}=10\). With one-indexed epochs, the strengths were therefore
\(0.1,0.2,\ldots,1.0\), not \(0\) in the first epoch.

The ramp changes residual amplitude, not the target supplied to the conditional
sampler. At full strength, the noisy value is exactly the sampled endpoint; at
partial strength, it lies linearly between the clean target and that endpoint.

### 3.2 Per-minibatch realization

For each minibatch, a new residual was independently drawn for every analog
weight, conditioned on that weight's current clean target:

\[
  x=\frac{q+1}{2},\qquad
  \epsilon_x\sim K(\cdot\mid x),
\]

and the state used by the forward computation was

\[
  \widetilde q
    =q+\alpha_e\epsilon_q
    =q+2\alpha_e\epsilon_x.
\]

At \(\alpha_e=1\), this is equivalent to

\[
  \widetilde q=2(x+\epsilon_x)-1=2x_{\mathrm{app}}-1.
\]

One parameter realization is shared by all examples in the minibatch. Noise is
not redrawn per example. A new full realization is drawn for the next
minibatch.

The noisy state \(\widetilde q\) was **not clipped** back into \([-1,1]\).
Out-of-range apparent samples were counted as a diagnostic. Silently clipping
them would change the fitted endpoint law and is therefore a different HWA
policy.

### 3.3 Straight-through gradient

Sampling and the target-dependent histogram are treated as non-differentiable.
The value presented to the network is constructed conceptually as

\[
  q_{\mathrm{STE}}
  =q+\operatorname{stopgrad}(\widetilde q-q).
\]

Thus,

\[
  q_{\mathrm{STE}}=\widetilde q
  \quad\text{in the forward pass},\qquad
  \frac{\partial q_{\mathrm{STE}}}{\partial q}=1
  \quad\text{in the backward pass}.
\]

The same sampled realization is used for the complete forward/backward pair of
that minibatch. There is no second draw during backward, and no gradient is
taken through the sampled residual or its conditional distribution.

After the optimizer update, the clean master is projected back into its legal
range:

\[
  q\leftarrow\operatorname{clip}(q,-1,1).
\]

The master must also start in that range. Only the clean master is optimized,
projected, checkpointed, and ultimately deployed.

### 3.4 Conceptual training cycle

For a network-independent reconstruction, one minibatch has the following
semantics:

1. Compute the current epoch's ramp strength \(\alpha_e\).
2. For each analog-mapped master tensor, convert every \(q\) to target \(x\).
3. Draw one conditional endpoint residual per physical/effective weight.
4. Form \(\widetilde q=q+2\alpha_e\epsilon_x\).
5. Use the straight-through form of \(\widetilde q\) for the network forward.
6. Evaluate the task loss and backpropagate to the clean masters.
7. Apply the optimizer update to the clean masters.
8. Project the clean masters into \([-1,1]\) and discard the noisy realization.

The original study used Adam and a teacher-KL objective against a frozen ReLU
teacher. Those choices belong to that task. The programming-noise mechanism is
compatible with another differentiable objective and optimizer, provided the
clean-master/noisy-forward distinction is preserved.

## 4. What “fresh programming realization” means

The minibatch draw represents a hypothetical fresh programming outcome at the
current target. It does **not** mean that a particular simulated device is
pulsed, retained, and then updated from its previous hidden state.

There is therefore no evolving training array:

- no fixed device identity is attached to a weight during HWA;
- no array-specific minimum/maximum support is applied during HWA;
- no fixed reference value is attached during HWA;
- no corruption mask is active during HWA;
- no pulse history or programming latency accumulates during HWA; and
- no sampled noisy state becomes the next optimizer state.

This population interpretation is what makes the resulting clean master a
candidate for transfer to independently assigned arrays.

## 5. What was deliberately kept separate

### Mapping versus HWA

Mapping decides how a logical tensor is represented in normalized device
coordinates. HWA perturbs the mapped target using a population endpoint law.
An array's realized bounds, reference values, and codebook do not belong in the
population HWA loop unless fixed-array adaptation is the explicitly declared
intervention.

### Programming error versus program-and-verify execution

HWA sampled a fitted terminal endpoint residual. It did not execute pulse
trajectories in the training loop. Pulse count, controller failures, and hidden
state evolution must be evaluated by a separate deployment/programming model.

### Programming error versus corruption

The HWA kernel represented successful, healthy programming. Persistent corrupt
devices and non-corrupt programming failures were evaluated separately. This
separation is scientifically important: robustness to ordinary endpoint error
does not imply robustness to stuck devices or unreachable targets.

### Programming error versus read noise

No independent inference-time read-noise process was added by this policy.
Programming error is a sampled post-write parameter state that remains fixed
for the minibatch forward; read noise would perturb reads/outputs and may vary
between evaluations. Adding read noise is a distinct experimental arm.

### Apparent versus persistent state

The standard MVM consumes the apparent post-write effective state. Persistent
state determines subsequent pulse behavior and is logged during actual
programming, but substituting it into the forward would model a different
device interface.

### Off-chip adaptation versus on-chip recovery

The described HWA is off-chip optimization of an FP32 master under sampled
population error. On-chip recovery begins only after a named deployed state is
created on a fixed array and updated with an allowed device controller. The two
must not be described as the same training phase.

## 6. Contrast with the legacy fixed-array Gaussian HWA

The older policy formed an array-specific base state and then added
unconditional Gaussian noise:

\[
  q_{\mathrm{base}}
  =\operatorname{clip}(q_{\mathrm{master}},q_{\min,A},q_{\max,A}),
\]

\[
  \epsilon\sim\mathcal N(0,\sigma_q^2),\qquad
  q_{\mathrm{forward}}=q_{\mathrm{base}}+\epsilon,
\]

with

\[
  \sigma_q=
  \texttt{write\_noise\_std}\,
  \texttt{nominal\_dw\_min}\,
  \texttt{relative\_scale}
  =1.4113\times0.0949\times1
  =0.13393237.
\]

The Gaussian standard deviation itself was a global preset scalar; it was not
estimated from Array A. Array A entered through the base-state bounds and
reference-relative mapping, the fixed device/corruption identities in relevant
variants, and the assignment-dependent random stream. In other words, the old
policy was still array-conditioned even though its scalar Gaussian width did
not come from that array.

The population policy changed all of the relevant semantics:

| Question | Legacy policy | Population endpoint policy |
|---|---|---|
| Clean base state | Array-A-conditioned | Global FP32 master |
| Error shape | Zero-mean Gaussian | Target-conditioned histogram |
| Source | Generic preset scale | Accepted endpoint population |
| Target dependence | None | Explicit \(K(\epsilon_x\mid x)\) |
| Fixed array identity | Yes | No |
| Corruption in HWA | Possible in published variants | Excluded |
| Assignment in RNG identity | Yes | Excluded |
| Intended interpretation | Adaptation to one mapped array | Transferable population robustness |

## 7. Porting the method to a different network

### 7.1 Define the analog coordinate first

Do not inject this residual directly in arbitrary logical-weight units. For
each analog layer, define a mapping into the signed device coordinate. If

\[
  W_\ell=s_\ell q_\ell,
\]

then sample error in \(q_\ell\), run the analog-equivalent forward with
\(s_\ell\widetilde q_\ell\), and update the clean \(q_\ell\) master. Any scale
\(s_\ell\), clipping rule, or trainability of that scale must be declared.

Perturb every state that will actually be programmed. Leave digital-only
parameters untouched. If biases are implemented digitally, do not perturb
them; if they occupy analog cells, include them under the same declared device
model.

### 7.2 Match samples to physical identity

Draw one residual per independently programmed effective state. Reuse that draw
wherever the same physical state is reused during the minibatch. For example:

- a recurrent/shared parameter should keep one draw across all of its uses;
- a convolutional kernel mapped once and reused should keep one draw;
- physically replicated copies require independent draws if they are separately
  programmed.

The present kernel models one effective signed \(q\) state per logical weight.
If the new hardware represents a logical weight with multiple independently
programmed devices, either derive the effective-state distribution for that
topology or sample the constituent devices and combine them. Do not silently
pretend that topology has not changed.

### 7.3 Preserve the temporal semantics

Use one parameter realization per minibatch, shared across the batch and across
its backward pass, then redraw on the next minibatch. If the intended hardware
is programmed less frequently, changing the redraw interval is legitimate but
constitutes a different HWA policy and should be evaluated as such.

### 7.4 Keep the population model independent of the evaluation array

The HWA seed should be derived from the experiment seed and the immutable
population-model identity. It must not depend on a deployment assignment seed,
corruption seed, or a particular array's bounds. Changing the evaluation array
while holding the training seed and model fixed should leave the HWA trajectory
unchanged.

Use an explicit random generator and checkpoint its state. For distributed
training, define deterministic non-overlapping streams or a canonical global
sampling order so that resume and replay have unambiguous meaning.

### 7.5 Deploy the clean master, not a training draw

After HWA, freeze the exact final FP32 master. Each target array must independently
map and program that master through its own bounds, identities, and codebook.
Never transfer:

- the last minibatch's noisy realization;
- a source array's realized endpoint;
- a source array's codebook; or
- hidden persistent state,

unless a separately named recovered-state transfer arm is intentionally being
tested.

For a repaired-versus-corrupt comparison, the master must be byte-identical and
assignment/programming seeds must be matched. Only the declared corruption
intervention may differ.

## 8. Minimum validation contract for a port

Before trusting a new-network implementation, verify the following properties:

1. **Zero-strength identity:** with \(\alpha=0\), the noisy forward equals the
   clean forward exactly.
2. **Full-strength endpoint law:** with \(\alpha=1\), sampled forward states
   reproduce the fitted target-conditioned residual distributions.
3. **Correct coordinate conversion:** \(\epsilon_q=2\epsilon_x\).
4. **Straight-through derivative:** the forward sees \(\widetilde q\), while
   the local gradient with respect to \(q\) is one.
5. **One draw per minibatch:** the forward and backward use the same sample;
   the next minibatch gets a new sample.
6. **No master contamination:** sampling never overwrites the FP32 master.
7. **Projected master:** the master stays in \([-1,1]\) after every update.
8. **No hidden forward clipping:** apparent out-of-range samples remain visible
   and are counted.
9. **Assignment invariance:** changing deployment-array assignment or corruption
   metadata does not alter HWA samples or the trained master.
10. **Exact replay:** restoring model, optimizer, and HWA RNG states reproduces
    the subsequent sample sequence and updates.
11. **Topology-aware sampling:** tied and replicated physical states follow the
    declared identity rule.
12. **Clean deployment handoff:** every deployment arm begins from the exact
    frozen master and performs its own mapping/programming.

Useful diagnostics include per-epoch residual mean, RMS, quantiles, saturation
by target region, out-of-range apparent counts, the ramp value, sample-sequence
digests, and clean-master versus noisy-forward accuracy. These should be
reported separately from actual deployment success/failure and pulse-cost
statistics.

## 9. Interpretation and limitations

This policy trains robustness to **successful apparent endpoint error sampled
from a device population**. It does not, by itself, demonstrate robustness to:

- persistent corrupt or stuck devices;
- target unreachability or non-corrupt P&V failure;
- hidden persistent-state mismatch;
- inference read noise;
- ADC/DAC, input-range, quantization, or output-scaling effects;
- pulse-count or energy constraints;
- a particular physical array; or
- on-chip recovery dynamics.

There is also a controller-match limitation: the HWA kernel came from an
adaptive, capped P&V characterization condition. If deployment uses a different
controller, pulse budget, acceptance rule, or device population, refit the
endpoint law or declare the mismatch. Do not present the old kernel as a
universal hardware model.

The approach is “IBM-like” in its use of a common normalized master range,
post-update projection, fresh programming-error samples, minibatch-consistent
forward/backward realizations, and a ramped error level. It is not exact parity
with IBM PCM HWA, and the fitted OM ReRAM source is model-based rather than raw
IBM hardware data.

## 10. Evidence from the source study

The artifact-verified source study found a mean repaired deployment accuracy of
**96.025%** for population programming-error HWA, versus **95.250%** for the
legacy fixed-array Gaussian HWA, a difference of **+0.775 percentage points**.
Deploying the same population-HWA master under the published corrupt-device
condition gave **90.675%**, a **-5.350 percentage-point** change relative to its
matched repaired deployment.

The repaired result was a numerical improvement on all three fresh-array
assignment means, but it **did not pass** the predeclared benefit conjunction:
the pooled gain was below the required 1.0 percentage point. The matched
corruption-materiality conjunction did pass. The transferable lesson is
therefore not that the method is generally superior or that it solves all
device nonidealities; it is that the healthy population-error model and the
persistent-defect penalty produce separately attributable outcomes. This study
also does not establish that on-chip recovery is required or sufficient.

These results are context for the handoff, not a performance prediction for a
different architecture. Formal scientific interpretation/finalization of the
source study remains a separate workflow step.

## 11. Recommended extension ladder

If the different network needs a richer hardware model, add mechanisms as
separate, attributable interventions:

1. reproduce the clean population programming-error HWA baseline;
2. refit the endpoint kernel to the intended controller and population;
3. add inference read noise as its own forward-time process;
4. add explicit programming-failure and corruption augmentation as separate
   arms;
5. introduce persistent-state-aware update dynamics only if on-chip updates are
   being modeled; and
6. compare off-chip HWA, HWA-only deployment, and on-chip recovery with matched
   masters, assignments, seeds, and budgets.

This ladder makes it possible to identify which nonideality causes a gain or
failure instead of collapsing all hardware effects into a single undiagnosable
“noise” term.

## Related scientific contracts

- [`docs/ibm_om_crossbar_population_hwa.md`](docs/ibm_om_crossbar_population_hwa.md)
- [`docs/ibm_om_deployment_scheme_investigation.md`](docs/ibm_om_deployment_scheme_investigation.md)
- [`docs/initialization_protocols.md`](docs/initialization_protocols.md)
- [`docs/synapse_data.md`](docs/synapse_data.md)
- [`docs/experiment_workflow.md`](docs/experiment_workflow.md)
