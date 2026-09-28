# X-005 — Noise in the nudging current

Proposed by Filip on September 24, 2026, alongside
[relative endpoint read noise](X-004-relative-read-noise.md). This is an
untested mechanism idea for MNIST Conv3 EqProp. Endpoint read noise corrupts
the measurement after relaxation; nudging noise changes the teaching current
and therefore the states reached during relaxation.

The question is whether the schemes transmit the intended teaching signal
and errors in that signal differently. Larger clean displacement need not
mean better robustness if the same response also amplifies current noise.
There is no presumption that this intervention makes legacy worse.

## Two different noise models

Let `B` be the actual injected beta and `f` the frozen output teaching force
computed at the common post-T free state. Nominal currents are `+B*f` and
`-B*f`. Any implementation must use this physical injected-current scale,
not silently attach noise to the scheme-dependent internal base beta.

| Model | Currents in the two phases | Small-signal implication |
|---|---|---|
| Fixed current noise | `I_plus = B*f + eta_plus`; `I_minus = -B*f + eta_minus` | Signal shrinks with beta while independent current noise need not; smaller beta can worsen SNR. |
| Relative teaching-force noise | `I_plus = B*(f + epsilon_plus)`; `I_minus = -B*(f + epsilon_minus)` | Signal and noise both scale with beta; lowering beta alone need not improve their ratio. |

A concrete relative candidate is nodewise
`epsilon = r_nudge * abs(f) * xi`, with independent standard Gaussian `xi`.
It changes the force direction as well as its magnitude. A single noisy
scalar beta instead mainly changes the magnitude in the linear regime and
may have little effect on gradient cosine; these controls are not equivalent.

For a local linear approximation, let `R_layer` map a perturbation of the
output current to the layer's endpoint state under the fixed solver unroll.
The clean centered displacement tensor is approximately `B * R_layer * f`.
With fixed additive current errors, its error is approximately
`R_layer * (eta_plus - eta_minus) / 2`. With relative force errors, it is
`B * R_layer * (epsilon_plus + epsilon_minus) / 2`.

Consequently, robustness depends on the response in the intended-force
direction relative to its response to the noise directions. Raw displacement
alone cannot establish that comparison. This reasoning is local and may fail
at large beta, active-set changes or other nonlinear responses; `R_layer`
here is a finite-unroll response, not an asserted equilibrium inverse Hessian.

## Correlation and timing must be explicit

Using one noisy force `f + epsilon` and reversing its sign for the two phases
does **not** cancel the force error: it reverses along with the signal and
survives the centered contrast. In contrast, adding the same current offset
`eta` with the same sign to both phases cancels to first order about the same
linear operating point. Independent errors between phases form a third case.

For the first candidate, draw the error once per example/output component
and phase and hold it fixed for all K nudged steps. This measures a noisy
frozen teaching current. Resampling each solver iteration changes the temporal
noise model and must be treated separately. Keep free relaxation and inference
clean in the first checkpoint replay.

## Discriminating test, when assigned

Start with the same clean checkpoints, cohort, beta points and T=K=8 as the
clean response comparison. Use clean reads and noisy nudging first; test noisy
reads alone separately, then their combination only if the individual effects
justify it. Choose a small declared subset of beta/noise levels before running,
not another broad search or any additional T/K tests.

Measure layerwise clean/noisy state contrasts, noisy/clean gradient cosine and
relative error, and noisy-gradient cosine against the unchanged clean BPTT
reference. Report repeated-draw variability, current noise RMS and actual
injected beta. Pair draws across schemes and hold the chosen fractional or
absolute current-noise rule fixed. These are alternative comparison regimes,
not both the same physical current budget.

Evidence for the idea would be a scheme-dependent loss of gradient direction
under matched current noise that is explained by signal-versus-noise response,
rather than clean displacement magnitude alone. An unchanged ranking or mainly
scalar gradient variation would weaken that explanation. Training accuracy
remains a separate question after this diagnostic.

No noise levels, executable experiment or training launch are assigned here.
The active [clean beta sweep](X-003-clean-beta-response.md) stays unchanged.
