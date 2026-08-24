# IBM OM array-specific common-window HWA pilot

These configs are shared only by the preserved
`mnist-ibm-om-common-window-hwa-program-verify-pilot-20260824-v1` attempt and
its corrected
`mnist-ibm-om-common-window-hwa-program-verify-pilot-20260824-v2` retry.
They are a corrected follow-up to the global-range pilot; they do not replace
or reinterpret its immutable results.

V2 contains two corrections discovered before v2 preparation or launch. Both
v1 arms passed preflight, wrote one initialization metric, and then failed
while serializing nested immutable modifier parameters into checkpoint
metadata. They executed zero training or optimizer updates. V2 fixes that
operational boundary and explicitly adds the eight-cell hybrid fallback
described below, which v1 did not define. Thus the hypothesis, arms, configs,
inputs, and seeds remain matched, but v2 is not scientifically identical to
v1. The failed v1 attempts remain preserved; v2 does not resume or overwrite
them.

- `clean.json` trains without HWA and selects under the common repaired
  pulse-resolved deployment.
- `exact_bounds_hwa.json` trains with compact endpoint sampling from the exact
  fixed array and selects under that same repaired pulse-resolved deployment.

Both physical paths use assignment seed `84001`, repaired corruption, the
`dual_rail_quad_common_window` mapping, layouts `halves` for
`base.dense_weight.0` and `paired` for `base.dense_weight.1`, and a `0.25`
fractional margin on each side. This maps into the central 50% of each
nonempty common window. It is a conservative range contraction, not a claim
of a two-standard-deviation write-noise guard. The common assignment is
intentional: this first pilot
isolates whether exact-array HWA improves deployment to the array against
which it trained. It is not a held-out-device generalization study and it
does not support a published-corruption claim.

The compact HWA path is an explicit hybrid only at a predeclared structural
edge case. For the fixed seed-`84001` population, four permitted empty common
windows produce four cell targets below and four above their individual
bounds. The fitted compact endpoint sampler stays fail-closed. It handles the
other `158792` cells; exactly those eight out-of-bound cells use the physical
pulse-resolved controller under policy
`pulse_resolved_noncorrupt_out_of_bound_empty_quad_only`. No in-range cell,
nonempty-quad cell, or generic unsupported compact class may fall back. The
launcher executes and records one real compact training context before either
native arm may start, including fallback identity, pulse cost, and non-finite
gates.

The endpoint application policy is
`aihwkit_apparent_forward_persistent_update_state`: the AIHWKit apparent
endpoint is the conductance used for the forward evaluation, while the
persistent endpoint is the hidden physical state from which a later update
would continue. Persistent residuals remain required device-level diagnostics;
they are not substituted for the apparent forward weights.

The phase launcher must run the authoritative mapper preflight before it may
start either arm. Its one pulse-resolved programming context is evaluated
directly on the first 100 deterministic validation batches (1,600 examples,
batch size 16), without reprogramming. Student accuracy and teacher agreement
must both be at least `0.25`. A hand-authored reachability report is not an
acceptable substitute. The launcher deliberately does not apply ideal-mapped
or persistent endpoint vectors to the network; any later persistent-only
network replay is exploratory and non-gating.
