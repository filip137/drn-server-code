---
id: "exp-004"
title: "Qualify matched-RMS beta using fixed-weight, fixed-reference replay"
status: "blocked"
hypotheses: ["H-001", "H-003", "H-004"]
---

# exp-004 — Candidate-policy replay

## Question and dependencies
Do the existing output-RMS values remain numerically qualified, and does the observed
T sensitivity survive a common-environment replay? Depends on exp-003's exact sources.
No raw source config was available to make a launch-ready command in this setup.

## Proposed bounded cases
Use the shared seed-0 initialization and the exact 576-example cohort, clean endpoints
plus four matched noise draws at sigma=5e-4. Candidate policy betas are baseline=88.7,
ours=1.385, legacy=0.02173. Current p99 controls are ours=0.987333678708 and legacy=0.1;
recover the baseline control from its source, rather than guessing it.

For policy diagnostics: at most nine checkpoints (initial, available epoch10 and final
for each scheme), two policies, and T in {8,16} with K=8: at most 36 checkpoint-policy-T
cells. Report missing checkpoints instead of manufacturing substitutes. For H-003's
history contrast, add at most six cells: initialization, old p90 ours and current p99
ours, each at replay beta=5.26875648112 and T in {8,16}, K=8. Total cap: 42 cells.
Identical cases can share cached endpoints, not count twice as evidence.

## Reference, controls and metrics
Use a fixed K64 BPTT reference within each checkpoint/batch, and an adequacy sentinel
at a larger unroll under the existing gradient-audit protocol. Qualify the reference
before interpreting its cosine; finite BPTT is not asserted to be exact. Source-defined
solver tolerances must be recovered in exp-003. Report any mismatch to historical
references separately, not as a failed replication of a different quantity.

Record pooled output RMS using all examples/coordinates, each sign, centered displacement,
per-layer RMS, cosine, relative gradient error, phase residuals, clipping/saturation and
nonfinite counts. Noise reads follow the source node-sharing semantics; couple draws
across compared policies/T for precision without changing their marginal noise model.

H-001's exact historical replication uses its own source reference-free RMS definition.
H-003 uses its proposed 0.1/0.01 contrast criteria. For H-004 also compute the original
source-configured BPTT reference at its exact checkpoint/beta/T/K/noise conditions;
keep it separate from the fixed-K64 diagnostic, since changing the reference changes
the measurement. Include this one source-reference control inside the runtime cap.
A valid replay is only a gate/diagnostic for H-002, never an accuracy selection outcome.

## Execution blocker and output
Reuse/adapt the existing source replay runner after inspecting it; freeze its exact
config, command, code and input hashes in this file before marking ready. Do not add
an invented CLI. Write a new indexed results root under the reporting contract, then
results/exp-004-replay.md with coverage and scoped verdicts. No training updates.

## Budget and stop
Proposed cap: 42 cells, one cohort, five endpoint-read conditions (clean plus four draws),
one qualified fixed reference per checkpoint/batch, and four GPU-hours total including
reference checks. Measure smoke throughput before admitting the matrix; if it cannot
fit, reduce to a scientifically complete prespecified subset or leave the rest unstarted.
Stop a cell on nonfinite/failed reference qualification; retain it. No silent T/K or
precision changes and no open-ended beta search.
