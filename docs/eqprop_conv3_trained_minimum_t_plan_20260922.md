# Minimum free-phase T at the trained ours/sigma5e-4 checkpoint

Exploratory follow-up to Filip's question about the minimum T that converges.
Use the same final epoch-30 checkpoint, float64 runtime, perfect diodes,
K=8, small injected beta B=0.000526875648112 and clean endpoint reads as the
preceding settling diagnostic. Replay all 36 matched validation batches of 16
(576 examples); no training, accuracy evaluation or official-test access.

Measure the free-phase residual after every integer T from 1 through 64.
Reuse the frozen residual implementation: per-example maximum projected-KKT
residual for each hidden layer and raw residual for the unconstrained output.
For continuity with the preceding replay, the primary minimum requires every
layer in every batch to have p90 < 0.01 and maximum < 0.1. Also report the
pooled 576-example p90 threshold and the stricter all-example maximum < 0.01
threshold separately. This is a trained-checkpoint validation diagnostic,
not the protocol's 1,024-training-example initializer T selection.

Advance the unmodified inference minimizer one complete iteration at a time.
Verify its T8 state hashes against all 36 prior direct replays and T32 against
the three earlier direct checks. Record per-example residuals so no monotonicity
assumption or interpolation is needed to identify the smallest tested integer.
T64 is the upper sentinel; report unresolved if it fails rather than extending
the range silently.

At the first candidate passing the free-phase gate, run the original full
clean centered-EP/BPTT replay on all 36 batches. Require the free, zero,
positive and negative phase gates to pass. If a nudged phase fails, try the
next free-phase-passing integer until the first complete passing candidate.
Also replay T32 and T64 as references. Gradient direction and norm are
reported separately; a residual pass does not establish K convergence or
exact EP/BPTT agreement.

Akib RTX3080 is idle and preserves the preceding diagnostic's target. Local,
nom-cool-1, Riri and Fifi are occupied; Trex and Loulou are also idle, and
Jean Zay has no current user allocation. Use Akib's existing isolated replay
workspace, require an empty GPU compute lane at admission, and retain all
attempts. Expected duration about five minutes; hard budget 1,200 seconds
including a one-batch smoke through the same runner. Monitor semantic progress
at least every minute while active.

Command: `python -m experiments.diagnose_conv3_minimum_t --config
configs/conv/eqprop_conv3_trained_minimum_t_20260922.json` (add `--smoke` for
one-batch validation). The authoritative local result root is
`results/eqprop-conv3-trained-minimum-t-20260922-v1/`. Collect and validate
remote outputs before interpreting the minimum, and record the conclusion in
`docs/experimental_manifest.md`.

## Operational correction

The first launcher exited before computation because the frozen loader reads
`source_contract.runtime_source_root` directly from the command-line config.
The new config initially supplied only its parent config reference. Add the
identical explicit frozen runtime path and rerun smoke plus production under
`launcher-retry1`; retain the original launcher log and initial payload. No
checkpoint, cohort, solver choice, threshold or noise setting changes.
