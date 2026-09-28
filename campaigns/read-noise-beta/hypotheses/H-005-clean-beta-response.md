---
id: "H-005"
title: "Legacy displacement decays faster with decreasing beta in clean Conv3"
---

# H-005 — Clean beta-response slope

## Provenance and regime

Prospective hypothesis requested by Filip on September 24, 2026;
[X-003](../explorations/X-003-clean-beta-response.md).
Compare baseline, ours and legacy at the shared initialization and the
clean EqProp epoch-30 checkpoints, with fixed T=K=8 and a common injected-beta
grid. This is an exploratory single-seed mechanism diagnostic.

## Measurement and interpretation

For each state layer and training stage, report neighboring-point slopes
`p = delta(log(centered RMS))/delta(log(injected beta))` and the contrasts
`p_legacy-p_baseline` and `p_legacy-p_ours` on identical intervals. A larger
positive slope means faster decay as beta decreases; a greater absolute
displacement does not establish a greater slope.

Report where both contrasts are positive, negative or mixed, without
selecting a favorable interval after seeing the data. Restrict slope
interpretation to positive, finite signals above the declared numerical
resolution indicator. A broadly consistent positive contrast supports the
claim within the recorded layer/stage/range; parallel curves or consistently
nonpositive contrasts contradict it there. Mixed ranges or unresolved signals
leave the broader claim inconclusive. These descriptive comparisons are not
multi-seed statistical evidence.

Literal positive-free displacement, zero-nudge drift and clean EqProp/BPTT
cosine contextualize the response. Fixed-budget residual failures are shown
and restrict any equilibrium interpretation. No T/K sweep is part of this test.
