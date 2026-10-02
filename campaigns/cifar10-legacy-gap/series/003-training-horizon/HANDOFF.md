# CIFAR training-horizon execution handoff

> **Latest scope — September 25 shared queue:** Filip reauthorized the 20-epoch
> low-LR floor options for **baseline, legacy and ours**, and explicitly requested
> automatic GPU placement through the running shared `run-watch` queue. This
> supersedes floor cancellation and Fifi-only placement for these three jobs.
> Older renewal instructions below remain provenance; do not relaunch cancelled jobs.


## September 27 status check — low-LR shared queue

Shared queue alive with fresh observation and no host probe errors. Baseline
(Riri/5090) and legacy (Loulou/5090) completed epochs51–70, were collected locally,
and passed declared validation. Final validation accuracy: baseline89.36%,
legacy88.76%; mean66–70:89.404% and88.836%. Charged execution:5.66h and4.19h.

Ours ran on nom-cool-1/3090 sharing with Ben and checkpoint-paused after epoch59,
9/20 added epochs, last validation accuracy89.84%. Direct SSH recheck confirms
paused status and checkpoint existence; the recovery agent's earlier DNS failure
is no longer current. Epoch59 took2283.56s (38.1min). Charged5.72h of6h leaves
1004.7s, insufficient to finish11epochs. Queue state is attention; no automatic
resume is prepared or running. Completion needs an increased authorized budget
and a compatible continuation command; preserve checkpoint SHA256
`3086356c369054ecaec1e0e68d9a205524b72703121322ab523e5542ceda51da`.
These are operational measurements, not a completed three-arm comparison.

## Shared queue: three low-LR floor jobs

Submitted IDs: `cifar-floor20-baseline-20260925`, `cifar-floor20-legacy-20260925`,
`cifar-floor20-ours-20260925`. All three accepted as `queued` and reconciled by the running service.
Eligible placements: local, nom-cool-1, Trex, Fifi, Loulou, Riri, chosen by the
service when ownership and memory permit. Akib's 10 GiB GPU cannot fit these jobs.
No Fifi pin or night-only window. Other jobs remain untouched.

Each restores its original epoch-50 parent, trains epochs51–70 at the unchanged
2% LR floor, batch32/eval16, float32/TF32 off, seed0 and T=K[6,6,4]. No renewal
schedule and no official test access. Normalized legacy is excluded from this
new submission, so this does not complete the original four-arm scope.

Result root: `results/cifar10-l8-horizon-e50-e70-seed0-20260924-v1/`.
Frozen source: `source-shared-queue-20260925/`; identity SHA256
`56833739b6f7e16e807b7dab73c6cb6e39b19f477f88945eb63157db18e0ce12`.
Exact queue specs/foreground worker/collector:
`launch/shared-queue-20260925/{baseline,legacy,ours}.job.json`, `worker.py`, `collect.py`.
Production: `cells/{case}_floor_e50_e70`; execution smokes:
`gates/shared-queue-20260925/{case}_{attempt-token}`. Remote root is relative to
`/home/filip/server_code`. Each placement verifies source/config/parent/data and
full-state restoration, then runs the native one-batch CUDA execution smoke.
Retired pre-launch solver/migration audits are omitted; the frozen final audit remains.
Local/nom-cool-1 use the previously verified isolated cuDNN9.1 process environment.

Budget:6h/case including smoke and target validation,18h total; expected4.5h/case
on a5090. A3090 may pause short of20epochs within this cap. Calendar deadline
September28 08:00 Paris. Native checkpoint deadlines reserve teardown time inside
the queue allowance; no automatic fresh budget or unsafe restart from epoch50.
Retry is disabled unless a compatible checkpoint continuation is prepared and
remaining budget is reconciled. Preserve partial/failed outputs and escalate.

Ownership verified at **2026-09-25T18:20:44+02:00**: service PID3007181 alive, admission enabled,
no host probe errors; all three jobs queued with zero charged seconds, waiting
for eligible GPU memory/ownership. No training has started. Frozen inputs are
staged and source-verified on all five remote hosts, and all three local parent
bundles/checkpoint hashes validate. Queue specs pass native schema validation.

The existing `run-watch-queue.service` owns placement,60-second observations,
budget enforcement and bounded incident recovery. No second monitor is armed.
Inspect with `python /home/filip/.codex/skills/run-watch/scripts/gpu_queue.py status JOB_ID`.
Successful compute triggers CPU-only rsync collection and local native validation.
Completion requires epochs51–70 exactly once, epoch70/98490steps, native bundle
validation and the final audit; conclusions remain in the exp-006 result note.

---

> **Earlier placement instruction (superseded above for the new floor jobs):**
> all future continuations move to Fifi.
> V100 successor jobs189958/189969 are cancelled; never resubmit them.
> Current V100 jobs181102/181103 run to native pauses, then transfer their
> validated checkpoints to Fifi. The10h extra/case budget stays authorized.
> The later-submitted Fifi queue has no priority reservation over this work,
> per the user; do not stop running unrelated jobs or bypass memory limits.

> Current reporting authority: updated root AGENTS.md makes
> `docs/current_simulations.md` and `docs/experimental_manifest.md` historical
> snapshots. Do not edit or refresh them, including on completion or recovery.
> Record execution updates here and conclusions in the experiment result note.
> This overrides older instructions in this handoff and watcher authorization.

> **September 25, 2026 — pre-launch policy update:** Filip removed CIFAR solver
> and migration audits as launch prerequisites, including the epoch-50 full-validation
> replay and accuracy/CE drift thresholds. Future launches retain source/config/data
> identity, full checkpoint restore, execution smoke, resource admission and runtime
> nonfinite checks. The older gate requirements and observations below are historical;
> do not rerun them automatically or record an omitted audit as passed. Already launched
> jobs retain their frozen execution; post-training audits are outside this pre-launch change.

Updated September 25, 2026. **Earlier renewal-only scope, superseded above for new floor jobs:** cancel Fifi, run only renewal, include baseline,
use local GPU, nom-cool-1 and one RTX5090, and start **now** with daytime
continuation authorized. At that time no floor production was authorized; the new three-case queue
assignment above now authorizes it.

## Cases and scientific contract

| Case | Target | GPU | Cap including gates/retries |
|---|---|---|---|
| baseline renewal | local / nom-cool-2 | RTX3090 | 6 hours |
| legacy renewal | nom-cool-1 | RTX3090, Ben sharing authorized | 6 hours |
| ours renewal | riri | RTX5090, Ben sharing authorized | 6 hours |

Each restores its own original epoch50 checkpoint, Adam/RNG/BN state and
70350 steps. Epochs51–70 use `.02+.08*(1+cos(pi*(epoch-51)/19))/2`
for every original LR group. Fixed batch32/eval16, float32, TF32 off,
T=K[6,6,4], original45k/5k split, no official test. Baseline is the same
schedule intervention newly applied to its own parent, without LR retuning.
Total new cap18 GPUh including checks/retries, within the original36h study
allowance. Previously consumed preflight remains recorded in the old study.
Native checkpoint/epoch deadlines enforce per-case caps. Final calendar
cutoff remains September28 08:00 Paris; no automatic budget extension.

Future admissions require identity/full-state restore checks and an exact
one-batch CUDA renewal smoke (epoch51/70351 steps,32train/16validation).
The epoch50 solver/full-validation migration gate and drift thresholds are
retired by the September25 user decision. Frozen controllers already launched
under the former three-boundary-gate/three-smoke contract keep their recorded
execution. The final solver audit remains outside this pre-launch change.

This is a three-arm renewed-schedule comparison. Without new matched floor
controls it does **not** test H-008's renewal-minus-floor criterion, nor complete
exp-006/H-006/H-007. Report mean66–70 versus original46–50, final70 accuracy/CE,
LR/step history and all validity checks. Hardware/software differences are
confounded with arm: local/nom use Torch2.5.1+cu121; Riri2.11.0+cu128, Riri
cuDNN91900; the3090 processes use isolated cuDNN90100 after runtime repair. Passing boundary gates does not prove trajectory equivalence.

## Source, paths and commands

Local root `results/cifar10-l8-schedule-renewal-e50-e70-seed0-20260924-v1`; remote root `/home/filip/server_code/results/cifar10-l8-schedule-renewal-e50-e70-seed0-20260924-v1`.
Source `source-20260925-renewal-only/` retains the original frozen runner;
only baseline renewal config and source identity were added. Identity SHA256:
`1adbf324de2c32abe36a4f57c022be237cb2ffdacd893e167823bc92faa0e14d`.
The scientific runner code hashes remain unchanged. Exact configs are in
source `configs/cifar/horizon_e50_e70_20260924/`.

Transport: `launch/renewal-only-20260925/night_queue.py`, one controller per
host with env `CIFAR_TARGET=local|nom-cool-1|riri`,
`CIFAR_RESULTS=<host repository>/results`. Python on all hosts:
`/home/filip/miniconda3/envs/py312/bin/python`.
Host launch directory `launch/renewal-only-20260925/TARGET/` contains
`queue.pid`, `queue-status.json`, `queue.log`, per-child exact `.command.txt`
and logs, `budget.tsv`, `last-admission.json`, clock anchor and exit evidence.
Read these live handles rather than assuming an old PID. Native outputs:
`gates/TARGET/CASE_epoch50_boundary`, `gates/TARGET/CASE_renewal_cuda_smoke`,
`cells/CASE_renewal_e50_e70` plus `_segmentNN` for validated full-state pauses.
The queue translates coordinator deadlines to the remote wall clock for the
native runner and uses a boot-bound monotonic anchor for its own accounting.

No automatic scientific retry. Exit75 allows only validated full-state
checkpoint continuation with matching hash and original schedule; remaining
six-hour budget must admit an epoch. Unexpected exits, invalid results,
failed gates,45-minute stalls and epoch70 audit-only interruptions escalate
to a bounded Codifier incident. Never restart a controller blindly over prior
outputs or reset budget files. Never stop Ben's workloads or change MPS.

## Monitoring and collection

Monitoring owner is the persistent run-watch transport at
`launch/renewal-only-20260925/watch.json`, polling every300seconds with bounded
incident dispatch. It observes queue ownership, child/native progress, logs and
completion. Configuration handoff is this file. Confirm watcher PID and a real
observation before claiming armed; handles and observations are recorded below.

Use `/home/filip/.codex/skills/run-watch/scripts/run_watch.py status WATCH_JSON`.
For each remote host, inspect the target launch directory via
`ssh -F /home/filip/.ssh/config -o BatchMode=yes HOST ...`.
Collect stable output only, without `--delete`:
`rsync -a -e 'ssh -F /home/filip/.ssh/config -o BatchMode=yes' --exclude='/source-20260925-renewal-only/' --exclude='/launch/' HOST:/home/filip/server_code/results/cifar10-l8-schedule-renewal-e50-e70-seed0-20260924-v1/ results/cifar10-l8-schedule-renewal-e50-e70-seed0-20260924-v1/`.
Collect each target's launch evidence separately into the same target-named
local launch directory. Do not overwrite local monitor configuration.
Validate each included native bundle with `python -m experiments.reporting
validate-run RUN_DIR` using the frozen source, then reconcile epochs51–70
exactly once, final70/98490 steps, final_solver_audit_passed=true and all three
cases. Exit0 alone is insufficient. Update exp-007 result note and dashboard;
record missing floors and untested H-008 honestly. No scientific conclusion
has yet been measured.

## Cancelled / superseded work

Fifi PID18566 was verified waiting with no child and terminated on explicit
user instruction September25. Its remote `cancelled-by-user-20260925.json`
preserves prior state; queue status is cancelled. Old local monitor received
`monitor/STOP`. No Fifi production or CUDA gate ran. The tentative two-host
six-case split was never armed; partial staging is not a launch. Do not launch
old `arm-fifi.sh`, split scripts, or any exp-006 floor case.
Original handoff retained in floor study
`launch/split-20260925/HANDOFF-before-split.md`.

## Initial launch observation

Controllers started September25 around13:02 Paris: local2534280,
nom-cool-1 3074357, Riri3532950. Riri passed boundary+CUDA smoke and
started production worker3534411 at13:03 Paris; epoch51/batch100 observed.
Local baseline and nom-cool-1 legacy boundary attempts failed before any
training with `CUDNN_STATUS_NOT_INITIALIZED`; investigation reproduces on
a tiny BatchNorm independent of experiment. Preserve these attempts and
charge their5seconds each; no scientific threshold failed or was relaxed.
The root Codifier owns recovery until replacement handles and watchdog
observations are recorded. Old Fifi monitor has exited after its STOP marker.

## Understood runtime recovery

The3090 shared py312 installations contain CUDA13 cuDNN9.19 files overwriting
the installed CUDA12 cuDNN9.1 package. A minimal BatchNorm fails independently
of the model. Isolated `nvidia-cudnn-cu12==9.1.0.70` is staged under the renewal
root `runtime/cudnn91/lib`; file hashes are in `runtime/cudnn91/identity.json`.
Both `LD_PRELOAD=<lib>/libcudnn.so.9` and `LD_LIBRARY_PATH=<lib>` are required;
LD_LIBRARY_PATH alone mixes versions. GPU forward/backward passed with both.
No shared environment changes; frozen runner/configs unchanged.

The two failed boundary bundles remain at their original paths. Replacement
boundary checks use `_attempt02` via `CIFAR_GATE_SUFFIX=_attempt02`. Old
controller/log evidence lives in each launch directory's
`attempt01-cudnn13-mismatch/`; budget files remain cumulative at their original
paths. Current local controller2542602. Preserve runtime environment on all
subsequent smoke/train/recovery commands; see `runtime-recovery.json`.

## Active monitoring handoff — September25 13:10 Paris

Verified detached watcher PID2544660, PPID1, running `run_watch.py ... --dispatch`.
Its first real observation reports all three controllers alive, native progress,
no probe errors, and no pending incident. Poll every5minutes, next13:15 Paris;
bounded Astra/medium incident turns (600s,2 attempts/incident,9 invocations/day).
The adapter corrects remote native timestamps using each coordinator clock anchor.
The local host must remain awake/connected; no reboot persistence is configured.

| Case | Controller PID | Current child / phase at handoff |
|---|---|---|
| baseline/local |2542602|2544007; solver audit passed, full epoch50 validation running|
| legacy/nom-cool-1 |3076851|3076862; epoch50 solver boundary audit running|
| ours/Riri |3532950|3534411; epoch51 batch700 / step71050|

Root Codifier recovery is finished; run-watch owns future observations and
incident/collection dispatch. Baseline/legacy automatically proceed through
CUDA smoke to production only if their scientific gates pass. Their original
failed attempts are retained; no production checkpoint was advanced by those
failures. Riri gate bundles have been collected and locally validated: audit
passed, accuracy drift0pp, CE drift0, smoke epoch51/step70351, peak reserved
11,710,496,768 bytes. Remote five-training-batch hashes match frozen data on
nom-cool-1 and Riri, with no official-test access.

Remaining cap at last admission: ours about5h55m; baseline and legacy under6h
minus their recorded checks. Historical5090 production estimate4.2h;3090 and
shared timings require measured epochs. Partial coverage remains possible at
the6h cap. No floor branch or additional condition may be started by recovery.

## September25 13:54 Paris status update

All three gates passed and production is advancing. Baseline/local and
legacy/nom-cool-1: epoch52, batch500/1407; latest complete epoch51 validation
88.82% and88.24%, respectively. Ours/Riri: epoch54, batch1300/1407; latest
complete epoch53 validation89.12%. No new errors or stalls. Watcher last
scheduled poll13:50, no dispatched incident agents.

Measured runtime risk: first epoch baseline1750.77s (~29.2min),
legacy1862.37s (~31.0min), ours recent epochs~765.4s (~12.8min).
At these rates the3090 trajectories require roughly9.7/10.3h of production
for20epochs, exceeding their unchanged6h caps; expect partial checkpointed
coverage unless the user revises budget. Do not silently extend. Riri's
~4.25h production estimate fits its cap, with approximate17:20 Paris finish
plus final audit. Estimates are provisional and contention-dependent.

## September25 Jean Zay migration — latest user instruction

User explicitly requested stopping both3090 runs and resuming baseline/legacy
on Jean Zay **V10032GB**, with tests and canary skipped. This instruction
supersedes the skill's default pre-submit tests/canary requirements for these
two submissions. No test/canary or retired migration audit is claimed passed.
Riri/ours remains untouched. The two3090 controllers and workers are stopped;
validated native paused bundles retain epoch51 and full Adam/scheduler/RNG state.
Partial epoch52 is replayed, not counted twice. Checkpoint hashes:

- baseline: `d6e47a3e7f758207769ab27bd9f8578e6354810028bd391fffd953febb8594d7`
- legacy: `01263fac93ec64c0c592bb00ac1c58ab351f1faffe5d0b74e6ce876729d069ca`

Both paused bundles are collected and validated locally. Controller termination
prevented its normal final accounting append, so `migration-accounting.json`
and cumulative `budget.tsv` explicitly charge training through native pause
plus10seconds overhead. Baseline spent3694.407s, remaining17905s;
legacy spent3910.251s, remaining17689s. Until a user budget extension arrives,
these are hard compute limits for the resumed jobs. Hardware speed on V100 is
unmeasured; full epoch70 coverage is not promised under the remaining limits.

Submit two independent jobs: account `fmu@v100`, partition `gpu_p13`,
QOS `qos_gpu-t3`, constraint `v100-32g`,1GPU/1task/10CPUs each,
`--hint=nomultithread`, about5hours allocation per job. Queried associations,
project membership and feature names support these exact settings. Module
`pytorch-gpu/py3/2.5.0`: Torch2.5.0/CUDA12.2/cuDNN8907/NumPy2.1.2.
Record this additional runtime change; do not infer cross-hardware trajectory
identity. No shared3090 runtime library is injected into Jean Zay.

Frozen source staged under
`/lustre/fswork/projects/rech/fmu/ucy17uy/server_code/cifar-renewal-source-20260925`;
its `migration/` holds the exact wrapper, contracts and sbatch files copied from
local `launch/jean-zay-migration-20260925/`. Scientific source identity remains
`1adbf324de2c32abe36a4f57c022be237cb2ffdacd893e167823bc92faa0e14d`.
Remote inputs/results:
`/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/cifar10-l8-schedule-renewal-e50-e70-seed0-20260924-v1`.
Resume outputs: `cells/{baseline,legacy}_renewal_e50_e70_segment02`.
Each production worker first verifies source/config/training-data/checkpoint
hashes and paused-parent/asset bundles, then uses `--resume-checkpoint` and
the identical `--schedule-branch`. Native strict restore and finite checks
remain active. Completion still requires the final native solver audit.

Monitoring changes: old3090 entries were removed from active watch config;
retained backup `watch-before-v100-migration.json` is historical and must not
be started. Riri monitoring was restarted as PID2625779. New Slurm entries
will use `launch/jean-zay-migration-20260925/probe_slurm.py`, observing
`squeue`, `sacct`, native progress and bounded logs, never login-node GPUs.
Collection uses rsync from the remote root's `cells/*_segment02/` to the same
local study/cells paths, plus `slurm/`; retain original paused segments and
reconcile epoch51 followed by52–70 exactly once. A paused/time-limited Slurm
job is partial, never complete, and must not be blindly resubmitted with a
reset budget.

## Jean Zay submission handles — September25 14:18 Paris

- Baseline job **181102**, time limit05:02:00; native compute budget17905s.
- Legacy job **181103**, time limit04:58:00; native compute budget17689s.
- Both `PENDING (Priority)` at14:18; no allocation or production claim yet.
  `scontrol` confirms fmu@v100/gpu_p13/qos_gpu-t3, v100-32g,1GPU,10CPUs,
  40000MiB host memory each, correct wrapper and unique output/error files.
- Inspect: `ssh -F /home/filip/.ssh/config jean-zay 'squeue -j 181102,181103;
  sacct -X -j 181102,181103 -o JobIDRaw,State,ExitCode,Elapsed -P'`.
- Job IDs are persisted in local `launch/jean-zay-migration-20260925/job-ids.json`.
  Do not resubmit a pending job. Queued time does not consume compute budget.
- Current watch config retains ours and adds `cifar-renewal-v100-20260925-
  baseline` / `...-legacy`, using the Slurm adapter. Root requested an optional
  budget extension; without an explicit reply the inherited caps remain.

Migration monitoring owner: detached run-watch PID2638436 with dispatch armed.
First observation confirms Riri training epoch56/batch1200 and both V100 jobs
queued(Priority), with no probe errors. Five-minute polls, next~14:24 Paris;
Slurm startup must publish native artifacts within10minutes of running, and
45-minute semantic stalls trigger bounded recovery. The monitor will verify
first production restore/artifacts when the queued jobs receive allocations.
No V100 throughput has yet been measured.

## September25 14:40 Paris observation

Both V100 jobs running since about14:20: baseline181102 onr6i5n6 and
legacy181103 onr7i2n2, each epoch52/batch700 of1407. Native full-state
restore succeeded and step72457 confirms continuation; no errors. No complete
V100 epoch yet. First~19minutes for700batches suggests~38–40minutes/epoch
before a full measured epoch is available; slower than observed3090 rates,
with runtime/hardware confounds. Unchanged stop deadlines: baseline19:17:59
Paris, legacy19:14:23 Paris; checkpoint pause occurs earlier as needed.
Ours/Riri epoch58/batch700, latest full epoch57 validation89.36%,
~765seconds/epoch. Watcher healthy, no incident dispatches.

## September25 ~15:48 Paris observation

Fresh probes: baseline181102 and legacy181103 running epoch54/batch300,
latest complete epoch53 validation88.88% and88.62%. Full V100 epochs now
measured: baseline2337.5–2337.8s, legacy2341.1–2342.7s (~39min).
Ours/Riri epoch63/batch1200, latest complete epoch62 validation89.46%,
~765s/epoch. No current job errors/stalls. One monitor incident invocation
followed transient Jean Zay probe unreachability; latest probes and scheduled
status are healthy and no incident agent is active. Native V100 deadlines
remain19:14–19:18 Paris; expect partial coverage at those unchanged caps.

## September25 ~17:11 Paris observation

Riri/ours is on its final epoch70, batch300/1407; latest completed epoch69
validation89.64% (epoch68:89.72%). Final validation and native audit still
pending; do not claim completion. Both V100 jobs remain healthy at epoch56,
batch500/1407; latest epoch55 validation baseline89.28%, legacy88.76%.
Throughput ~39min/epoch on V100. No new training errors or active incident
agent. Existing compute caps remain unchanged; V100 coverage will be partial.
Monitor continues; historical dashboard/manifest updates are now forbidden
by the user's replacement root instructions.

## Authorized V100 continuation budget — September25

User: “also extend the budgets of the v100 runs when they are done”.
Baseline and legacy may each use **10 additional GPUh** after their current
V100 allocations finish, with **16h cumulative cap per case** including prior
segments. This supersedes earlier six-hour/remaining-five-hour statements.
Ours/Riri's six-hour cap is unchanged. Combined active three-case cap is now
38GPUh; historical excluded preparation remains separately recorded.

Two continuation jobs are submitted with `afterany:181102` and
`afterany:181103`. An expected native pause exits75, so `afterok` is unsuitable.
The dependency releases resources only after the predecessor exits; the worker
then requires a valid **paused** bundle, matching checkpoint SHA and epoch<70.
An unexpected failed/nonfinite predecessor cannot silently become a retry.
Outputs `cells/{baseline,legacy}_renewal_e50_e70_segment03`; epoch horizon70,
Adam/RNG/scheduler, batch/precision/LR schedule and frozen scientific runner
are unchanged. Only `maximum_runtime_hours` becomes10 in a new source copy.
Each Slurm allocation is10:03 with native deadline<10h and calendar cutoff
September28 08:00 Paris. Expected remaining training~8h after epoch58, plus
final audit; budget includes throughput headroom. Tests/canary remain skipped
under the user's standing instruction. Do not edit historical dashboards.

Source: `/lustre/fswork/projects/rech/fmu/ucy17uy/server_code/cifar-renewal-source-20260925-v100-extension`.
Source identity SHA256:
`324a44dd4d3274bc5f05664231c55c88a3f859c03a52a36ee90f94550858d1c4`.
Local transport/authorization: renewal root `launch/jean-zay-extension-20260925/`.
Monitor must follow the current jobs through their checkpoint pauses and then
their dependent successors, preserving all3 segments and consumed budgets.

V100 successor handles verified: baseline **189958** depends afterany181102;
legacy **189969** depends afterany181103. Both PENDING(Dependency), with
10:03 Slurm limits. Watcher PID2913024 is armed and its first observation
correctly follows live predecessor181102/181103 while recording successors.
It switches to segment03 when the dependencies release. Ours remains observed.

User additionally requested an ours follow-up on Fifi with “the other LR
adaptation strategy”, and gave a priority exception over later-submitted
Fifi work. Scientific starting checkpoint/schedule and whether “Fifi only”
also replaces the V100 successors are awaiting explicit clarification; no
Fifi follow-up has been armed and no successor has been cancelled on that
ambiguous phrase. Fifi has~1GiB free with Adrien72167 and Kellian90217 running;
neither process is modified. A later Filip queue's reservation alone need not
take priority under the user's exception, but actual GPU memory must still fit.
