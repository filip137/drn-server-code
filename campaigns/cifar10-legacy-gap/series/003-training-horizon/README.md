# Series 003 — Training horizon and learning-rate floor

> **September 25, 2026 — pre-launch policy update:** Filip removed CIFAR solver
> and migration audits as launch prerequisites, including the epoch-50 full-validation
> replay and accuracy/CE drift thresholds. Future launches retain source/config/data
> identity, full checkpoint restore, execution smoke, resource admission and runtime
> nonfinite checks. The older gate requirements and observations below are historical;
> do not rerun them automatically or record an omitted audit as passed. Already launched
> jobs retain their frozen execution; post-training audits are outside this pre-launch change.

## September25 user revision — authoritative current scope

The user cancelled Fifi and all floor runs, added baseline to renewal, and
authorized immediate daytime execution of baseline/local, legacy/nom-cool-1,
ours/Riri. Three original epoch50→70 renewal branches,6h each including
gates;18h total. Follow [current handoff](HANDOFF.md).
The earlier allocation and six-case contract below are preserved as history.
H-008 renewal-minus-floor cannot be evaluated without matched floor controls.

Current execution state, commands and ownership: [run handoff](HANDOFF.md) (September25 reboot recovery; queue PID18566). The September24 snapshots below are historical.

Design prepared September 24, 2026. Filip subsequently authorized scheduling
the six-case study for **September 24, 22:00 Europe/Paris (20:00 UTC)**.
Preparation, runner qualification and small preflight checks are authorized
now; production must wait until that time and satisfy lane/resource gates.
The scheduler and actual handles will be recorded below once armed.

[exp-006](experiments/exp-006-floor-continuation.md) extends all four seed 0
trajectories from 50 to 70 at the unchanged LR floor and tests H-006/H-007.
[exp-007](experiments/exp-007-schedule-renewal.md) adds two renewed-schedule
branches from the same original epoch 50 checkpoints (legacy and ours), reusing
their exp-006 floor arms. It tests H-008 without a broad LR search. The
schedule intervention is a new comparison regime, so it has its own series.

All six cases use **Fifi, one worker at a time**, after the existing EqProp
allocation releases its lane. This is a proposed assignment, not a resource
reservation. No scheme is assigned a different host. The historical prefix
still has mixed hardware/software; native baseline/legacy used RTX 3090 with
Torch 2.5.1/cu121, whereas Fifi uses RTX 5090 with Torch 2.11.0/cu128.

## GPU allocation snapshot

Read-only inspection on September 24 at approximately 14:47–14:49 UTC.
Remote clocks differ (notably Riri); use coordinator UTC for scheduling.

| Target | GPU / free memory snapshot | Existing work / allocation decision |
|---|---|---|
| Fifi | RTX 5090 /32149 MiB | No compute process observed; active EqProp campaign has pending Fifi work. Primary CIFAR target only after that allocation releases the lane. |
| Trex | RTX 5090 /25239 MiB | Ben's GPU workload; sharing is permitted if memory fits, but existing EqProp pending allocation has priority. Whole-study fallback before the first CIFAR case only. |
| Loulou | RTX 5090 /30460 MiB | Active Filip EqProp queue 2166929; leave it intact. |
| Riri | RTX 5090 /30460 MiB | Active Filip EqProp queue 2915962; leave it intact. |
| Local / main | RTX 3090 /11454 MiB | Active replay PID 1924031, below the planned memory admission floor; no CIFAR assignment. |
| Nom-cool-1 | RTX 3090 /23030 MiB | Ben workload; the standing sharing exception applies to 5090s, not this 3090. No assignment. |
| Akib | RTX 3080 /9189 MiB | Active EqProp queue 524075 and insufficient memory for historical CIFAR peak; no CPU-offloaded alternative. |
| Jean Zay | fmu@v100, gpu_p13 | Access verified; array 152887 has three running tasks plus pending tasks; two idle nodes observed. No CIFAR reservation/submission; different runtime would need a revised whole-study contract. |

Fifi checkout `/home/filip/server_code`, Python
`/home/filip/miniconda3/envs/py312/bin/python`, Torch 2.11.0+cu128,
cuDNN 91900, training data and 3.2 TiB free storage were verified. No source or
checkpoint staging was performed. At actual admission recheck owners, queued
work, memory and environment. Do not stop, replace or occupy an existing lane.
Use a distinct `cifar-horizon-20260924` launcher name when later authorized.

## Compute and scheduling budget

Historical Fifi epochs take about 758 seconds for ours and normalized legacy;
20 epochs cost approximately 4.22 GPUh. Baseline/raw legacy timing on Fifi is
an estimate transferred from the same architecture, not a measurement.
Six branches imply about 25.3 GPUh before checks; budget is **36 GPUh total**:
24 for exp-006 (four branches),12 for exp-007 (two added branches), including
smokes, boundary checks, final audits and understood retries. Reused floor
controls are charged once. Hard case cap6 GPUh across all segments/retries;
no extra epochs or replacement scientific conditions.

Historical peak reserved memory is 10.91 GiB. Admit only if at least 14 GiB is
free (historical peak plus at least 3 GiB headroom), and increase that threshold
if the eventual same-path smoke measures more. One CIFAR worker per GPU,
batch32/eval16, float32/TF32-off, no CPU offloading. Ben sharing remains
eligible on5090s; measure actual throughput and retain shared timing labels.

Proposed order: legacy floor, baseline floor, ours floor, normalized-legacy
floor, legacy renewal, ours renewal. Complete the whole prespecified matrix;
the renewal branches are not selected based on which floor score is favorable.
Default authorized execution would use 22:00–08:00 Europe/Paris, beginning on
the first eligible night after a future execution instruction and lane release.
At historical speed, expect about three nights; allow at most four eligible
night windows within seven calendar days of first production start. The authorized dated windows are September24→25,25→26,26→27 and27→28,
22:00–08:00 Europe/Paris (+02:00); final cutoff September28 at08:00.
No production is admitted before the first window. The eventual launcher must record exact dates and
UTC offsets before launch and pause by 08:00 without daytime continuation.

Use the existing full-state epoch-boundary deadline pause, not SIGSTOP or
epoch truncation. Initial duration estimate 900s and shutdown reserve 180s;
then use measured throughput and the runner's 25% admission margin. Carry a
partial branch to the next window with its original config and scheduler.
Cap overruns or loss of target availability yield an explicit partial result;
do not migrate selected schemes silently. Before the first production case,
Trex may replace Fifi only for the entire six-case surface after matched
environment and memory checks, re-estimation and updating this assignment.

## Evidence and handoff

Result roots and execution contracts are in the experiment records. No new
result directories exist yet. Source/input checks, CPU tests and data-free
config validation are preparation; actual scientific smokes and boundary
replays must pass before production under the new scheduling authorization. Monitor every 30 minutes during future
execution; retain progress, PID/log/exit records and per-case budget. Collect
and validate all local bundles, audit expected coverage and exclusions, write
the experiment result notes, then regenerate the ledger. Use matplotlib for
loss/accuracy curves, showing both the original prefix and continued segments.

## Scheduling preparation record — September 24

The six-case source snapshot is frozen at
`results/cifar10-l8-horizon-e50-e70-seed0-20260924-v1/source-20260924-v1/`;
its identity SHA-256 is
`56833739b6f7e16e807b7dab73c6cb6e39b19f477f88945eb63157db18e0ce12`.
The same files, four parent bundles and four asset bundles have been staged
and hash-validated on Fifi. The schedule-branch implementation passed 28 CPU
regression tests, with two CUDA-only tests skipped; eight transport safety tests
passed, including the required `CUBLAS_WORKSPACE_CONFIG=:4096:8` child setting. These tests do not replace scientific smokes or native solver gates.

The local monitoring scheduler is armed in tmux session
`cifar-horizon-monitor-20260924` (PID 2028277), first observation September24
22:00 Paris. It invokes bounded independent agent checks every30 minutes
within the four night windows, plus morning collection. It runs on the local
workstation and requires that machine, API access and SSH to remain available;
its logs distinguish a live scheduler from completed observations.

All six local one-batch CPU smokes passed sequentially under `smoke-local-02/` (epoch51, step70351, 32 training and16 validation examples; bundles validated). The
initial sandboxed legacy attempt failed because the data loader could not
open its local IPC socket; it is retained under `smoke-local/` and excluded.
The retry preserves config, batch size, checkpoint and frozen source, and
runs outside the sandbox. CPU preflight elapsed time is conservatively debited
to the overall runtime cap, while recorded actual GPU consumption is zero.
All four read-only epoch50 migration gates and six CUDA smokes must pass on
Fifi before any production case. They may run only within the dated windows.

The Fifi queue is armed: PID2786584, verified detached and in `scheduled_waiting_night` state at17:35 Paris on September24. Its authoritative handles, state, commands, logs and elapsed accounting are in the floor study `launch/` directory. No production or CUDA checks have started. A coordinator-time monotonic anchor corrects the remote host's approximately34-second clock lag without changing host settings; the original waiting controller was retired before any child ran, with history preserved. CPU preflights conservatively debit1276.063287 seconds to the budget (actual GPU use zero). The first window opens September24 at22:00 Paris; GPU admission and scientific gates precede training.
