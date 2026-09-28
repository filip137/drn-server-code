# X-009 — Locate the relative-noise advantage ridge

September 25, 2026. Filip requested an Explorer proposal for a tighter noise and
output-displacement sweep on Conv1/2/3, using Akib, nom-cool-1 and nom-cool-2.
The working noise model is the nodewise relative endpoint noise from exp012.
Filip clarified that eta must be fixed across all layers at each sweep point.
Every noninput layer, including readout, receives the same global eta in an
acquisition, shared across schemes. Different absolute errors arise from their
different voltage magnitudes.
He also requires the same eta range for Conv1/2/3. Exp013 therefore uses one
common 14-value grid from 3e-9 to 1e-2, and applies the union of refinement
coordinates to all architectures. The earlier architecture-specific grids
are superseded; no results were generated from them.
This is an initialization study; the request does not introduce training.

## Evidence that sets the ranges

[Exp012](../series/006-relative-endpoint-noise/results/exp-012-relative-endpoint-noise.md)
shows a third-convolution advantage for ours at both D/F=2 and 6. At D/F=2,
eta=1e-4, the mean relative gradient-error norms are 0.356 for ours and 1.407
for legacy. At eta=1e-3, ours also loses much of its alignment. A transition
between these levels is more informative than simply increasing noise.

For orientation only, approximating cosine by `1/sqrt(1 + (a*eta)^2)` for
roughly orthogonal noise whose norm is linear in eta predicts a maximum gap
near eta=2e-4 at D/F=2 and roughly 6e-4 at D/F=6. This is a heuristic from
mean norm ratios, not a fitted result or a constraint on the new measurements.
If displacement is locally proportional to beta, the peak may form a ridge
with approximately constant eta/(D/F), rather than a unique preferred D/F.

Resolving the first two Conv3 convolutions requires sampling lower global eta: at D/F=2,
eta=1e-4 their legacy relative gradient-error norms are approximately 10,753
and 8,168. Linear noise scaling places their order-one error near eta=1e-8.
Both were already at the cosine floor in exp012's noise range. Hidden-state
D/P alone would miss the cancellation and adjacent-state dependence of the
second convolution's gradient.

Conv1/2 have no completed relative-noise map. The common grid is informed by
older gradient norms and voltage scales as well as Conv3's relative-noise
measurements; it does not transfer optima to Conv1/2. At D/F=2 and absolute sigma=1e-4, legacy's
gradient-error norms are 0.0583 for Conv1 and 18.89/1.106 for Conv2's two
convolutions. See the series005 [screen](../series/005-initial-displacement-noise-map/results/exp-010-initial-cosine-surface.md)
and its linked physical/gradient controls.

## Proposed decision

Use [exp013](../series/006-relative-endpoint-noise/experiments/exp-013-relative-noise-advantage-map.md)
to map per-layer ours-minus-legacy cosine, retain baseline, and refine each
convolution's best sampled neighborhood with fresh noise draws. Show each
scheme's absolute cosine alongside the difference. Also report worst-layer
alignment: a favorable last layer must not stand in for trainable early layers.

The report should distinguish a broad near-maximum ridge, an interior peak,
an unresolved boundary maximum, and an apparent advantage at the noise floor.
The comparison always uses the same eta and matched output D/F across schemes.
The hidden-D/P matching control proposed after exp012 remains a separate
mechanism experiment; it would answer a different question.

One initializer and one validation cohort support only conditional diagnostic
conclusions. Fresh noise draws test acquisition variability, not model-seed
generality. Finite-K BPTT alignment is not a training-accuracy prediction.
