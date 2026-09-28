---
id: "H-009"
title: "Fixed absolute read noise contributes to legacy's alignment advantage through voltage scale"
---
# H-009 — Absolute read noise and voltage scale

## Provenance and motivation

Working hypothesis proposed by Filip on September 25, 2026, after the completed
initialization cosine/displacement maps: "the read-noise is the absolute value
and that higher voltages help." This is a retrospective explanation of the
observations in [X-008](../explorations/X-008-legacy-alignment-despite-attenuation.md),
with a prospective mechanism test; it is not a confirmed causal result.
The earlier [relative-noise idea](../explorations/X-004-relative-read-noise.md)
provides a discriminating intervention.

## Claim and regime

In the ordinary-MNIST Conv1/2/3 initialization replay, fixed absolute additive
Gaussian endpoint noise gives larger-voltage schemes smaller fractional read
errors. This contributes to legacy's observed advantage in later Conv3 layers
under matched normalized output displacement. It need not explain every layer
or imply a training-accuracy advantage. Preserve the accepted centered EqProp,
clean finite-K BPTT reference, shared initializer/cohort, and fixed T/K4/6/8.

The matching rule is `r = D_out/F_out`, with
`D_out = RMS((v_plus-v_minus)/2)` and `F_out = RMS(post-T free output)`.
Consequently, at fixed r and absolute noise standard deviation sigma,

`D_out = r * F_out` and `D_out/sigma = r * F_out/sigma`.

Independent additive noise in the two phase reads has centered-contrast noise
standard deviation `sigma/sqrt(2)`. Thus its output voltage-contrast SNR scale
is `sqrt(2) * D_out/sigma`; matching D/F does not equalize this quantity.

The measured Conv3 values at r=2 are:

| Scheme | Free-output RMS F | Absolute output displacement D |
|---|---:|---:|
| Baseline | 0.00600041 | 0.0120008 |
| Ours | 0.0413063 | 0.0826126 |
| Legacy | 0.384026 | 0.768053 |

Legacy therefore has approximately 64 times baseline's and 9.30 times ours'
output voltage-contrast SNR scale at equal sigma. These factors concern the
output voltage contrast, not gradient SNR or cosine. The measured stronger
relative attenuation toward legacy's input can coexist with larger absolute
displacements near its output; see the hidden-layer values in X-008.

## Prediction and discriminating control

Replacing fixed absolute noise with equal fractional endpoint read noise should
reduce the part of legacy's alignment advantage caused by voltage scale. The
nodewise model proposed in X-004 is `v_tilde = v + eta * abs(v) * xi`, with the
same eta across schemes, independent positive/negative phase reads, and clean
relaxation. Its relevant voltage-contrast scale is D/P, where P is the pooled
RMS of the two endpoints. P need not approximate free-state F at large nudges.

Use paired clean/absolute/relative endpoint replays to distinguish noise-induced
gradient degradation from pre-existing clean EqProp/BPTT mismatch. Hold states,
betas, cohorts and references fixed. Relative-noise levels, their comparison
with absolute-noise levels, layer/cell coverage, and decision criteria must be
specified before testing; a layer-RMS noise model is a separate modeling choice.
Equal absolute-D matching is another possible control, but changes beta and
potentially clean finite-beta error, so it does not isolate the same intervention.

## Support, falsification and inconclusive cases

Support would be a reproducible reduction in legacy's noise-induced gradient
advantage after controlling fractional read precision, consistent with measured
layerwise endpoint and displacement scales. A ranking reversal is not required.
An unchanged or larger gap over a valid, prespecified comparison would argue
against this proposed contribution. Arbitrary unmatched noise severity, mixed
effects, or cosines all at their floor/ceiling leave the explanation unresolved.

Voltage RMS alone cannot establish gradient direction: local energy gradients
depend on adjacent-state products, phase differences and noise correlations.
Keep this hypothesis scoped to the measured layerwise pattern, including ours'
confirmed worst-layer advantages in Conv1/2. No relative-noise measurement or
new run is assigned by registering this hypothesis.
