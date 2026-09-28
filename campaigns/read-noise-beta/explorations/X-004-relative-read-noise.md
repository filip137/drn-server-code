# X-004 — Relative read noise and relative phase displacement

Proposed by Filip on September 24, 2026, after discussing legacy's advantage
under fixed absolute endpoint read noise. This is a new, untested explanation,
not a reinterpretation of that comparison as invalid. It concerns the MNIST
Conv3 EqProp regime in this campaign, not the separate CIFAR BPTT experiments.

The question is whether equal fractional read precision changes the scheme
comparison, and whether relative phase displacement predicts the resulting
gradient error. A reversal of legacy's ranking is possible, not assumed.

September 25 follow-up: Filip selected fixed absolute read noise favoring larger
voltages as the working explanation of the normalized-displacement observations.
This is registered in [H-009](../hypotheses/H-009-absolute-noise-voltage-scale.md);
the relative-noise model below remains a proposed, untested mechanism control.

## Model and prediction

For each measured endpoint voltage, consider

\[
\widetilde v_\pm = v_\pm + r|v_\pm|\xi_\pm,
\qquad \xi_\pm\sim\mathcal N(0,1),
\]

with independent draws between nodes and phases and the same dimensionless
noise fraction `r` for every scheme. Preserve the existing endpoint-only
placement: clean relaxation, exact input, then noisy positive/negative state
reads used consistently in the local gradient calculation.

Define the centered displacement and endpoint voltage scale as

\[
D=\operatorname{RMS}((v_+-v_-)/2),\qquad
P=\sqrt{(\operatorname{RMS}(v_+)^2+\operatorname{RMS}(v_-)^2)/2}.
\]

The expected squared RMS noise in the centered voltage contrast is
`r² P² / 2`. Its signal-to-noise scale is therefore `sqrt(2) D / (r P)`.
For small nudges, `P` is approximately the free-state RMS `F`, so the useful
quantity is **relative displacement D/F**, rather than absolute displacement
alone. Use `D/P` when nudging appreciably changes endpoint voltage magnitudes.

This is an analytic voltage-contrast prediction. It is not an exact formula
for weight-gradient error or training accuracy: local gradients also depend
on products of adjacent states and their joint noise.

## Choices that can change the answer

- Phase correlation: shared fractional error, implemented as
  `v_tilde = v * (1 + r * xi)` with the same `xi` in both phases, multiplies
  the centered contrast itself. It does not inject independent noise from
  the large common voltage into the subtraction. This is a different control.
- A mixed model, `v_tilde = v + sqrt(sigma0² + (r*v)²) * xi`, includes an
  absolute floor. Pure relative noise vanishes at zero voltage; adopting it
  is a modeling choice, not a demonstrated property of our hardware.
- Nodewise voltage, layer RMS and ADC full-scale are different noise scales.
  Scaling noise by displacement itself would approximately fix contrast SNR
  by construction and is less useful for testing this explanation.

## Discriminating test, when assigned

First compare clean, existing absolute-noise and relative-noise endpoint
replays on the same checkpoints, cohorts, injected betas and fixed T=K=8.
Hold noise fraction, placement and phase correlation constant across schemes;
pair random draws across comparisons. Measure noisy/clean gradient cosine and
relative error per layer, alongside clean EqProp/BPTT cosine and `D/P`.

The explanation gains support if relative-read gradient degradation follows
`D/P` more closely than absolute `D`, with uncertainty over batches/draws.
It is weakened if the scheme gap persists without the predicted relation.
A changed diagnostic ranking alone does not establish a training advantage.
Freeze noise levels and the comparison rule before measuring this new model.

The clean sweep in [X-003](X-003-clean-beta-response.md) already saves endpoint
RMS values, so `D/P` can be derived without another GPU run. It does not measure
relative-noise gradient outcomes. [X-005](X-005-noisy-nudging.md) considers noise
in the force that generates the displacement, a separate intervention.

No new replay or training is launched by this exploration.
