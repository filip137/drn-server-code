# Series 002 — Qualify a beta-selection policy

Start from the existing Conv3 sigma=5e-4 contract, not a fresh architecture or LR search.
exp-003 is a source/coverage audit with zero training. exp-004 measures the candidate
without weight updates. exp-005 is a bounded proposed training comparison, downstream
of valid sources and instrumentation. None of these plans has been executed by setup.

The present setup budget is zero GPU time. Before an assigned execution, resolve exact
configs, reference adequacy and the explicit per-stage runtime cap. Keep noisy/clean
replays, training seeds and best/final checkpoint measurements separate.
