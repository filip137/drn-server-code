# Crossbar schedule / writer comparison

**Complete.** All eight arms exited successfully after 30 full epochs each
(240 epochs total). Exact P0, all data streams and schedules match within
each writer pair. Physical invariants and selected-state replays pass. Test
metrics, final-epoch outcomes and standalone PNG/SVG/PDF figures are in
[the results report](analysis/report.md). The plots were visually inspected.
No numerical job or analysis remains active; routine monitoring is finished.

Exploratory follow-up authorized by the user on September 6: reuse the newer
crossbar learning rates and decay when comparing open-loop and P&V updates.
The science is frozen in `plan.json`. Expected production coverage is eight
arms, each with 30 full epochs: healthy/faulted × constant/exponential ×
open-loop/P&V, from the exact original development-array P0 states.

Target: local RTX 3090, actual CUDA interpreter
`/home/filip/miniconda3/envs/py312/bin/python`. The hardware canary passed
allocation, matrix multiplication and synchronization; GPU was idle.

Gate: CUDA scheduler/controller tests, then two full-epoch exponential
open-loop canaries for healthy/faulted states and two short P&V canaries.
Compare the full canaries with the recent native runs at the same schedule,
including data order, pulse counts and state/metric replay.

`launch.py` uses a bounded subprocess queue. `launch.json` records the
supervisor; `logs/*.launch.json` records every worker PID, command, input
plan, output, status and log. `logs/*.exit.json` records exits and semantic
completion. Canaries use `smoke_` records and `smoke-*` logs. Production
outputs are `runs/<condition>-<schedule>-<writer>/`.

Heartbeats update every 100 minibatches and each epoch. Expected runtime is
several minutes per arm, to be refined after canaries. Codex inspects real
GPU/process and artifact progress at least every 30 minutes and performs
shorter artifact checks during this active turn. No passive launcher is
treated as a diagnostic watchdog.

Safe recovery changes only operational defects, preserving failed attempts.
Do not change rates, schedules, tolerances, pulse caps, cohorts or epochs in
a retry. Preserve both prior studies and all unrelated jobs.

Completion requires all eight successful results, full epoch/example
coverage, paired P0/data/schedule checks, physical invariants, selected-state
replays, final metrics, standalone matplotlib figures and a readable report.

## Validated launch

Seven CUDA tests passed. All four canaries completed with physical bounds,
immutable faults and selected-state replay intact. The two full open-loop
canaries matched the remote epoch schedules and minibatch order; their
stochastic pulse trajectories differed. The remote host used an RTX 3080,
while this paired comparison uses an RTX 3090. The cause of the trajectory
differences is not isolated. A separate local comparison against the native
gradient/update loop passed bitwise for 512 minibatches per condition,
including the decay transition, physical states and pulse-selection RNG.
Records: `smoke/gate.json`, `smoke/remote_comparison.json`, and
`smoke/native_loop_parity.json`.

The production supervisor is PID 3367796, with four CUDA workers. Its first
four workers were verified live with the intended commands and fresh epoch
artifacts. GPU utilization was 96%, memory 2374 MiB. Both writers are rerun
locally rather than importing remote open-loop trajectories into a local
closed-loop pair. `watchdog_observations.jsonl` records ongoing artifact
checks; no scientific setting has changed after launch.

The later process/GPU check encountered an automatic approval-review timeout.
The read-only GPU retry succeeded: the three remaining worker PIDs matched
the launched decay arms, GPU utilization was 73%, and artifacts continued
advancing. No experiment job failed or required a retry. All final worker
exit receipts record code 0 and semantic completion. The original two
studies remain unchanged.
