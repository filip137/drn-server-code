# Runs needed for the provisional stability-based beta protocol

Reviewed September 21, 2026. Metadata-only audit: no new training, replay,
official-test evaluation, or remote scheduling. See the
[provisional scientific decision](../docs/eqprop_beta_stability_protocol_20260921.md).

## Coverage being preserved

The primary count preserves the original wide-weight evidence grid: clean
seeds 0/1/2 plus five noisy seed-0 conditions (sigma 1e-5, 3e-5, 1e-4, 3e-4, 5e-4)
for each architecture/scheme. This is 72 full-budget trainings: 27 clean and 45
noisy. Conv1 uses 10 epochs; Conv2/3 use 30. Sigma 1e-3 is an optional extension
counted separately, not silently added to the original paper grid.

The audit joins existing trackers to local resolved configs, checking exact
injected beta, T/K, bounds, zero bias rates, completion status, and epoch
budget. A beta-compatible completed run is a reuse candidate, not automatic
paper promotion or a new full hash audit. Host/noise-stream limitations and
the ordinary-MNIST official-test rules continue to apply.

## Literal latest-p99 interpretation for Conv1

Conv1 uses p99/10=30/30/6; Conv2 p99=100/10/3; Conv3 uses the exact refined
p95 values. All values are injected beta.

| Model | Scheme | Proposed beta | Existing matching clean runs | Existing matching noisy runs | Full-budget runs still needed |
|---|---|---:|---|---|---:|
| Conv1 | baseline | 30 | None | None | 3 clean + 5 noisy = 8 |
| Conv1 | ours | 30 | Seeds 0/1/2 | All five noise levels, seed 0 | 0 |
| Conv1 | legacy | 6 | None | None | 3 clean + 5 noisy = 8 |
| Conv2 | baseline | 100 | Seeds 0/1/2 | All five noise levels, seed 0 | 0 |
| Conv2 | ours | 10 | Seeds 0/1/2 | All five noise levels, seed 0 | 0 |
| Conv2 | legacy | 3 | None | None | 3 clean + 5 noisy = 8 |
| Conv3 | baseline | 147.682614594 | Ten-epoch seed-0 pilot only | None completed locally | 3 clean + 5 noisy = 8 |
| Conv3 | ours | 2.49274796756 | Ten-epoch seed-0 pilot only | None completed locally | 3 clean + 5 noisy = 8 |
| Conv3 | legacy | 2.81845428732 | Ten-epoch seed-0 pilot only | None completed locally | 3 clean + 5 noisy = 8 |

**24 completed reuse candidates; 48 full-budget completions still needed**
(18 clean and 30 noisy). Three of those 48 are the Conv3 seed-0 clean controls
already declared in the separate p95 study. Thus 45 runs are not covered by
that existing plan, conditional on the final science
decision. Do not launch duplicate clean controls. The ten-epoch H100 pilots
do not replace the planned thirty-epoch V100 controls.

If the immediate review keeps only clean seed 0 plus the five noisy levels,
the grid is 54 cases: 18 already complete, 36 still needed, of which 3 clean
controls are already declared and 33 are not.

This arithmetic does not justify lowering Conv1 baseline from its stable
100/200 values to 30. It shows the consequence of enforcing the literal new
p99/10 formula. That interpretation needs to be distinguished from the goal
of finding the largest stable beta.

## Alternative: retain the historical Conv1 choices 100/30/3

All three Conv1 groups then already have the three clean seeds and five noisy
seed-0 cases. Only Conv2 legacy and the three Conv3 groups change beta.

- Full original coverage: 40 completed reuse candidates; 32 completions needed
  (12 clean + 20 noisy). Three clean controls are already declared, leaving 29
  runs not covered by the current plan.
- Seed-0-only clean/noisy comparison: 30 completed; 24 needed. Three clean
  controls are already declared, leaving 21 runs not covered by the current plan.

This retains the existing Conv1 training choices; it is not an exact
latest-p99/10 rule. No choice between the two interpretations is silently
made by this audit.

## Existing Conv3 p95 study and optional sigma 1e-3

The separate [six-case plan](../docs/eqprop_conv3_p95_read_noise_1em3_plan_20260921.md)
already declares three seed-0 clean controls and three seed-0 sigma 1e-3 runs,
each 30 epochs. The local analysis snapshot reports 0/6 completed. The first
noisy chunk was submitted as Slurm 19589_[0-2]; the retained monitor failure
at 2026-09-21T11:46:18 UTC is an SSH timeout. Current remote progress is not
established by this metadata audit. It is not evidence of scientific failure.

If sigma 1e-3 is added for all nine architecture/scheme groups, nine additional
seed-0 target-beta completions are needed; none is already complete in this
audit. The three Conv3 cases are already declared, so only six would be newly
declared. This is conditional scope, not an instruction to launch them.

## What changes and what remains evidence

- BPTT has no EqProp beta; this decision creates no beta-driven BPTT reruns.
- Bounded-weight studies are governed by separate calibration and iteration
  contracts; this wide-weight decision does not prescribe bounded reruns.
- Existing runs at other betas remain useful sensitivity evidence. Preserve
  their results and scientific failures instead of deleting or relabeling them.
- The Conv3 baseline beta 10 sweep matches p99, but no longer matches the
  proposed Conv3 p95 setting. It remains supporting evidence.
- None of the Conv3 p90 full-training results substitutes for a p95 run.
- Once the Conv1 interpretation and clean-versus-noisy stability objective
  are fixed, qualify the changed seed-0 candidates before expanding costly
  repeats. Selection must not be described as a measured maximum without
  compatible tests of larger candidate values.

[Per-cell CSV for both interpretations](beta_stability_protocol_run_audit_20260921_coverage.csv)
is generated from the existing trackers by
[the audit script](audit_beta_stability_protocol_20260921.py). It records the
authoritative local reuse path or the already-declared plan for each cell.
It is a review table, not a launch queue.
