---
id: "H-008"
title: "Ours or baseline has a reproducible gradient-alignment advantage in an initial displacement/noise region"
---
# H-008 — A region of initial gradient-alignment advantage

## Provenance and regime

Prospective diagnostic hypothesis requested September 25, after observed noisy
training rankings; see [X007](../explorations/X-007-initial-displacement-noise-map.md).
Ordinary MNIST, seed-0 shared Conv1/2/3 initializers, clean finite-K BPTT reference,
centered EqProp, fixed accepted T/K4/6/8, relative output displacement D/F in
[0.2,10], absolute additive endpoint Gaussian sigma in[1e-4,5e-3]. No training.

## Claim and measurements

There exists at least one sampled neighborhood for at least one architecture where
ours or baseline has better worst-layer gradient alignment than legacy. For each
layer, take the arithmetic mean of individual batch/noise-draw cosines; the network
summary W is the minimum of those layer means, including the readout. Compare
W(ours)−W(legacy) and W(baseline)−W(legacy). All per-layer differences are primary
reported evidence; W is the predeclared region-selection statistic, not a replacement
for those maps. Also label all-layer dominance separately when every layer's mean
difference is positive. A weaker isolated layer advantage does not satisfy the W claim.

## Support, contradiction and inconclusive cases

A screen candidate is the same competitor with positive W difference at two adjacent
sampled cells in the same architecture (same sigma and adjacent r, or same r and
adjacent nonzero sigma). Select at most one such pair per architecture by the largest
minimum W margin across the two cells. Ties use increasing r then sigma, baseline
before ours. Include all three schemes in confirmation; never compare independently
optimized targets or noise levels across schemes.

Support requires positive W difference at both cells for each of five new independent
noise draws, with complete finite per-layer comparisons and valid matching, measured
on the same fixed cohort. This is descriptive reproducibility conditional on that
initializer/cohort, not model-seed significance or proof about a continuous region.
Report draw variation and batch spread. A clean-only advantage is separate and does
not satisfy a noisy-region claim. Invalid/zero-norm comparisons remain undefined.

No candidates in a complete, valid screen contradicts the sampled-grid version of
the claim. Failure to reproduce a candidate leaves that proposed region unsupported;
missing or invalid cases and mixed confirmation are inconclusive. No finite screen
can establish that legacy wins everywhere between or outside sampled points.
Better cosine does not establish better training accuracy, gradient magnitude or
Adam updates. Record per-layer norms, relative error and signal/noise alongside cosine.
