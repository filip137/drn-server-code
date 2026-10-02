---
id: "exp-021"
title: "Batch128 read-noise training with matched noiseless EP and BPTT"
status: "ready-for-review"
hypotheses: ["H-011"]
---
# Exp021 — Batch128 read noise and clean algorithm comparison

The [partial batch-size result](../results/exp-021-batch128-read-noise.md)
records the requested batch16/batch128 read-noise comparison, including
historical optimizer resets, runtime tradeoffs and the outstanding case.

## Current execution handoff — September29, 07:50 Paris

Terminal collection September29,15:30Paris: **45/45 cases locally validated;
ready for scientific review**. CPU `collect.py --allow-partial` completed with
CUDA disabled; all declared epochs, frozen identities, matched initializers and
cohorts, finite metrics/model/Adam state, full RNG checkpoints and Adam step
counts passed. Final and best checkpoint biases are exactly zero for all45
cases. Official-test evaluations remain zero. This supersedes the partial
execution snapshots below; scientific interpretation is still pending.

The final recovered case, Conv3 legacy EP eta1e-5, completed as `302485_11`
on `r6i5n3` in5913 allocation seconds. Its accepted attempt3 reaches30epochs,
12900Adam steps and97.16% final validation accuracy. The restore receipt confirms
`exact_epoch_state_continuation` from epoch5, restored Adam, and empty missing/
reconstructed RNG lists. All90 production attempt directories are retained;
the cancelled original attempt's stale running bundle status is historical,
with terminal cancellation established by Slurm.

Fresh Slurm accounting reconciles94 allocations:45 accepted completed cases,
4 completed remote canaries,44 failures and1 cancelled superseded attempt.
Total charged allocation wall time is **40.091944GPUh of100GPUh**, including
startup/teardown, failed attempts and recovery gates. All accepted jobs are
terminal `COMPLETED` with exit0:0; the study's queue query is empty. No new
submission, relocation, numerical change or budget extension was needed.

Evidence under the existing result root:
`analysis/summary.json`, `analysis/coverage.csv`,
`analysis/final_epoch_accuracy.jpg`,
`analysis/completion-validation-20260929.json` and
`analysis/terminal-accounting-20260929.txt`.
This bounded incident returns operational `complete` to the sole run-watch
owner; no future compute checks are needed. Reviewer retains the scientific
result-note and campaign-ledger handoff, now backed by full local coverage.

Latest update September29,11:48Paris:44/45cases locally validated; only Conv3
legacy EP eta1e-5 remains incomplete. Conv3 ours EP eta5e-4 finished93.64%,
legacy clean/eta1e-4/eta5e-4 EP98.78/94.66/88.28%, clean legacy BPTT98.74%.
Recovery details below supersede the10:34diagnosis-in-progress entry: original
epoch5 model/Adam/RNG preserved, strict transport-path restore guard repaired
and gated, continuation302485_11 queued on V10032 with r8i7n7 excluded.
Direct Slurm forecast at11:47 moved to October2,00:10Paris; it is tentative,
and the prior14:18today forecast is withdrawn. Healthy remaining runtime is
about97minutes after allocation. Root assigned the existing Jean Zay Codifier
a bounded pending-resource/backfill check; no duplicate or hardware change
authorized. Persistent watcher1179513 retains sole monitoring ownership.
All accuracy figures are final epoch30 validation for seed0; study remains partial.

Latest update September29,10:34Paris:39/45cases locally validated, including
all15Conv2 and9Conv3. Conv3 ours clean BPTT finished at98.58%. Direct remote
metrics show ours EP eta5e-4 at23/30 (estimated11:00 finish), legacy clean EP
17/30 (11:24), legacy EP eta1e-4 and5e-4 at16/30 (11:27/11:26), and legacy
clean BPTT29/30 (10:35). These are throughput estimates, not deadlines.
Legacy EP eta1e-5 job295298_11 is anomalously slow:5/30 at10:31:53Paris,
mean880seconds across its four measured epoch intervals versus~232seconds
for peers. Its former11:25 ETA is withdrawn; current throughput threatens
the four-hour allocation. Root assigned bounded operational diagnosis to
the existing Jean Zay Codifier `/root/batch128_jz`; the persistent watcher
remains the sole monitor. No numerical changes, cancellation or duplicate
submission was made during this update. Earlier snapshots below are historical.

Conv3 update September29,09:35Paris:8/15Conv3 outcomes are locally validated
(37/45 across the study). All five baseline settings and ours EP eta0/1e-5/1e-4
finished. Final EP accuracies in that eta order are97.50/95.30/93.46% baseline
and98.66/97.30/95.52% ours; baseline EP eta5e-4 is89.38% and clean BPTT97.42%.
Direct Slurm observation at09:34Paris confirms all seven remaining Conv3 tasks
running, with none queued. The09:34artifact observation puts ours EP eta5e-4
at7/30epochs and ours BPTT at6/30; legacy tasks have recently started, with
no logged errors.
The existing watcher retains monitoring ownership; collection updated the same
`analysis/summary.json`. Earlier per-depth/fleet snapshots below are historical.

Conv2 update September29,09:05Paris: collection validated12/15Conv2 cases
(28/45 across the study). All baseline/ours cases and legacy EP eta0/1e-5 are
complete. Direct Slurm observation confirms the remaining legacy EP eta1e-4,
EP eta5e-4 and clean BPTT tasks running, with none queued in array295304.
The09:04artifact observation puts them at29/30,1/30,1/30 respectively, with
no logged errors. Final values and validation receipts are in
`results/conv123-read-noise-batch128-jeanzay-20260929-v1/analysis/summary.json`.
Persistent monitoring remains with the existing watcher; no new launch or retry.
The fleet-wide counts below are the earlier07:50snapshot.

All45 cases are assigned exclusively to V10032GiB. Accepted replacement arrays:
Conv1 `295303` (indices1–14), Conv2 `295304` (indices1–14), Conv3 `295298`
(indices0–14). Original Conv1 `295170_0` completed and original Conv2
`295176_0` remains accepted. Immutable45-case mapping:
`results/conv123-read-noise-batch128-jeanzay-20260929-v1/launch/jean-zay/submissions.json`.
The indexing/allocator startup issues were repaired without numerical-source
changes; both remote V100 gates passed and replacement EP/BPTT training is finite.

The user withdrew the proposed eight-GPU TOTAL limit. All19 temporary pending
holds were released and every added dependency was verified cleared. No healthy
run was interrupted. Receipt:
`launch/jean-zay/v2/eight-gpu-limit-withdrawn.json`. Earlier cap/dependency plans
below are withdrawn history and must not be reapplied.

At07:49:46Paris,17cases were complete and locally validated,17running and11
pending on Slurm priority. All15Conv1 cases and the Conv2 baseline clean EP/BPTT
pair are validated in `analysis/summary.json`. The existing Conv2/Conv3 array
throttles were increased to15 so their remaining tasks can start as capacity
allows; there is no overall eight-GPU limit. Both arrays retain `v100-32g`.
No new submissions, scientific changes or budget increases were made. Readback:
`launch/jean-zay/v2/parallelism-20260929-0748.json`.

Sole persistent watcherPID995295 has dispatch enabled, preserves incident/budget
history and polls every300seconds; the latest inspected artifact observation
was07:43:51Paris with no new errors. Slurm counts continue changing as cases
finish and pending tasks start. Normal monitoring authorization was restored;
no custom admission service was created. Inspect using the installed `run_watch.py status` with
`launch/jean-zay/watch.json` under the result root. The monitor owns subsequent
progress, bounded recovery and terminal collection/review; root has no competing
polling process.

Budget100GPUh remains unchanged; initial failed attempts and both remote canaries
charged0.282778GPUh. Production wall caps plus that overhead total90.282778GPUh.
Source/config/learning rates/batch128/epochs10,30,30 remain frozen. Timeout
recoveries use the latest complete model/Adam/RNG checkpoint and remaining
original epochs. Completion still requires all45 locally validated outcomes and
scientific review; do not equate scheduler submission with completed evidence.


## Question and user authority

On September29 Filip requested the first seed of the batch128 rerun, selecting
eta=0,1e-5,1e-4,5e-4 with Conv1/2/3 horizons10/30/30, original learning rates,
both EP and BPTT at eta0, and parallel execution. He explicitly selected Jean
Zay for all production. This narrows the preceding three-seed cost estimate:
no seeds1/2, bounded-weight sweep, A4/B2 scheme, LR tuning or extra noise levels
are included. The timeout instruction remains active: continue the latest saved
model/Adam state toward the original horizon, never restart training from epoch0.

Dependencies: [exp019](exp-019-relative-noise-eta5em4.md), the accepted seed0
initializers and D/F beta values from exp013–018, and the
[batch128 pilot](../../../../pilots/2026-09-29-conv2-batch256.md).
The matching rules in docs/conv_paper_one_seed_bptt_eqprop_protocol.md govern
clean EP/BPTT pairs. This is exploratory validation, with official test disabled.

## Frozen cases and comparisons

45 cases = 3 architectures × 3 schemes × 5 algorithm/noise settings × seed0.
Schemes: baseline(V/I=1/1), ours(4/1), legacy(4/0.25).
Settings: centered EP at eta0,1e-5,1e-4,5e-4; BPTT only at eta0.
Conv1 has10epochs; Conv2 and Conv3 have30epochs.
Training batch128, validation batch64, ordinary-MNIST deterministic55k/5k
split, model/loader/split seed0, float64, perfect diodes, wide weights[0,100],
exact-zero biases and bias learning rates. Input gains40/100/360 and T=K4/6/8
remain fixed. Explicit diode dictionaries and shared initialization hashes are
required. EP retains the accepted scheme-specific beta at initial D/F1/4/6.

Copy original named/ordered per-scheme Adam learning-rate vectors exactly;
clean BPTT and EP share them. No batch-size scaling or A4/B2 doubled rate.
All arms start their matched initializer with fresh Adam. Noise is
v_read=v+eta*abs(v)*xi, independent Gaussian phase/layer/node draws, with the
original noise seed2026081601; validation, inputs and relaxation remain clean.

Primary measurements: final-epoch validation accuracy for every cell, EP
ours-minus-legacy/baseline at each eta and depth, and clean EP-minus-BPTT per
scheme/depth. Also retain epoch10 accuracy, elapsed allocation/runner time and
GPU memory. Use all valid outcomes; no best-epoch substitution or seed selection.
H-011's epoch10 claim can be assessed separately at the chosen grid. Final-epoch
comparisons are descriptive and do not redefine that hypothesis. Negative and
failed outcomes remain visible. One seed does not support a significance claim.

Batch128 yields430 optimizer steps per full epoch (4300/12900 totals). The
same epochs have about8x fewer updates than batch16; cross-batch accuracy is
not an equal-update comparison. Missing source/host equivalence is disclosed.

## Execution contract and outputs

Config directory: `configs/conv/read_noise_batch128_20260929/`.
Local result root, including every canary, attempt, collection and summary:
`results/conv123-read-noise-batch128-jeanzay-20260929-v1/`.
Frozen source and portable initializers are below `source-training-v1/`;
`cases.json` lists the exact45 cases and config identities.
The numerical source derives from the accepted exp020 recovery snapshot,
excluding unrelated working-tree target-scale changes. Identities are frozen
before staging and verified before execution.

Use existing `experiments.exact_run` with explicit CUDA, dataset root,
`--checkpoint-every-epoch --epoch-chunk-size 30 --skip-terminal-official-test`.
Every completed epoch retains full model, Adam and RNG state in diagnostic
model/optimizer checkpoint pairs. Recovery uses `--resume-diagnostic-run SOURCE`. A new preserved attempt restores the latest completed diagnostic checkpoint
with its full global/sampler/worker/endpoint-noise RNG after verified timeout; no extra epochs or fresh
optimizer. Preserve and charge partial/failed attempts. If only diagnostic
state is available, the user's approximate-resume authority applies with an
explicit missing-RNG receipt, never an exact-trajectory claim.

Launch wrapper and exact Slurm commands: `launch/jean-zay/` under result root.
Expected remote root:
`/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/conv123-read-noise-batch128-jeanzay-20260929-v1/`.
WORK source location and returned Slurm array IDs will be added before/after
submission respectively. All production is Slurm-owned; no lab production jobs.

## Budget, gates and stopping

Provisional compute estimate40–70 GPU-hours; operational ceiling100 GPU-hours
including canaries, failed attempts, collection allocation overhead and recovery.
This is a new authorized batch, separate from the previous overnight09:00 target.
Scheduling/monitoring window ends October6 at07:00 Europe/Paris. The user
explicitly requires V10032GiB for all runs; this overrides the coordinator's
provisional72h queue allowance when V100 forecasts fall after October2. Queue delay is additional to compute;
array concurrency follows live project/QOS limits and resource availability.
Memory fit for batch128 Conv3/BPTT is unmeasured; choose a suitable authorized
GPU without changing batch, precision or algorithm. Any OOM/nonfinite preserves
evidence and triggers bounded operational diagnosis; no scientific retuning.

The current repository execution policy and docs/jean-zay.md require synchronous
local same-runner smoke, static wrapper/resource checks and recorded submission
identity. They supersede the external skill's older mandatory remote-canary
rule. Reinstate a target-specific remote canary only after a target-specific
production failure. No gradient/residual/rho qualification is assigned.

## Execution and monitoring handoff

Root owns scientific specification and final review. Config Codifier owns exact
cases/source; Jean Zay Codifier owns static gates, staging and submission;
monitor Codifier prepares one persistent run-watch owner for all arrays, CPU
collection and coverage validation. No simultaneous independent monitor.
Before handoff, verify fresh scheduler observations and monitor heartbeat.
Cadence300seconds; expected artifact progress once per epoch, with bounded
incident handling and timeout recovery within the same100GPUh ceiling.
Collect terminal cases and validate local reporting bundles, config/init hashes,
zero official tests, finite complete metrics, zero biases, epoch coverage and
Adam step counts. Reconcile all45 cases, attempts and charges. Write partial
analysis during execution; scientific conclusion goes in the associated
exp021 result note after coverage is reconciled. No further seeds are authorized.

### Planned Slurm resource contract before submission

Three15-task arrays, each concurrency8. Conv1 and Conv2 use
`fmu@v100/gpu_p13/qos_gpu-t3/v100-32g`,1GPU and4CPUs per task;
Conv3 also uses V10032GiB through an authorized fmu/umg V100 account,
`gpu_p13/qos_gpu-t3/v100-32g`,1GPU and4CPUs.
Task wall limits are30minutes/90minutes/4hours, totaling90GPU-hours for all45
initial attempts and leaving10GPU-hours within the100GPU-hour ceiling.
Module `pytorch-gpu/py3/2.5.0` for every task.
Accounts/features were checked live; mixed/idle inventory is not an allocation.
WORK root is
`/lustre/fswork/projects/rech/fmu/ucy17uy/server_code_pilots/conv123-read-noise-batch128-jeanzay-20260929-v1/`,
with `source-training-v1/` and `launch/` children. Production outputs remain
under the SCRATCH result root recorded above, split by arm and attempt.
The wrapper is `launch/jean-zay/run.slurm` locally and `launch/run.slurm` remotely.
Original LR equality, Adam hyperparameters, float64, batch128/64 and disabled
official-test policy independently checked across all45 generated configs.

### Frozen source and gate evidence

Source identity SHA256:
`eeea0b654abdcac01745625993b366b8da955865fba2e617c0d1e3a99f5d301c`.
Case-list SHA256:
`4bc32b17e502890b7d43a745951ad5980f8f672f217d10c65ae1e7282b297ba4`.
`preparation-validation.json` records45 config-contract checks and45 focused
frozen-source tests. The narrow operational patch adds full private RNG state
to each diagnostic optimizer checkpoint; `exact_epoch_state_continuation`
requires the same scientific/runtime contract and rejects missing private state
for this study. Historical incomplete checkpoints keep the explicit approximate
path. The numerical training update is unchanged.
Six local full-batch CPU smoke cases cover EP at the highest selected eta and
BPTT across all depths, each with3epochs ×2training batches and one validation
batch per epoch. These are execution probes, not production or scientific
qualification. They run sequentially with a30GiB process address-space ceiling
to protect local memory. Conv3 EP exceeded that virtual-memory ceiling; a
preserved retry uses38GiB on CPU, with no scientific/config change. The planned
source remains frozen throughout the gate.
Logs and receipt are below `local-smoke/`; source/config and Slurm resource
checks remain required before submission. No official-test evaluation is run.

### Mandatory target correction — September29

Filip explicitly reiterated that every production run must use **V10032GiB**.
A proposed H100 placement based on queue forecasts was withdrawn before any
production submission. No H100/A100 jobs are authorized for this assignment.
The V100 queue delay is accepted; it does not authorize a different GPU model.
All45 cases retain batch128, float64, originalLR and epoch horizons. CPU smoke
memory is not evidence of exact CUDA memory use; any actual V100OOM remains
visible and requires diagnosis without silently changing batch or hardware.
The total compute ceiling remains100GPU-hours; only the coordinator-assumed
queue window is extended to accommodate the required hardware.

### Submission gate passed

`launch/jean-zay/local-gate.json` validates all6 full-batch128 local CPU smoke
bundles against the frozen source identity. Every smoke completed3epochs with
2training batches per epoch and one validation batch. The Conv3EP30GiB
virtual-memory-cap failure is preserved; its identical-config38GiB retry
passed. This is an operational memory allowance correction, not a numerical
change or a CUDA memory-fit claim. All6 canonical reporting validations passed.
Remote static checks verified2096 frozen source files. V100-only wrapper and
submission guards reject H100/A100 profiles and non-V100 accounts. Exact
`sbatch --test-only` resource checks passed. The current V100 t3 forecast is
October3 afternoon; both accounts are authorized. Dev cannot accept these15-task
arrays under its submit limit; t4 forecasts are later. No duplicate submissions
or speculative GPU switches are permitted.

All3 production arrays use `gpu_p13/qos_gpu-t3/v100-32g`,1GPU4CPUs,15tasks
with concurrency8. Conv1/2 use `fmu@v100`; Conv3 uses `umg@v100`. Compute
caps remain30min/90min/4h respectively. The local gate and all source/config
identities are frozen before the live submit calls. Actual IDs and persistent
watch observation follow below.

### Submitted arrays — September29, about07:01 Paris

| Depth | Slurm parent | Cases | Account | Constraint | Concurrency | Cap/task |
| --- | --- | --- | --- | --- | --- | --- |
| Conv1 |295170|15|fmu@v100|v100-32g|8|30minutes|
| Conv2 |295176|15|fmu@v100|v100-32g|8|90minutes|
| Conv3 |295177|15|umg@v100|v100-32g|8|4hours|

`launch/jean-zay/submissions.json` contains45 unique case/job mappings; root
independently checked coverage, uniqueness, V100 accounts and32GiB constraint.
Per-array receipts and exact commands remain alongside it. No H100/A100
production or separate remote canary was submitted. The actual remote outputs
are below the previously recorded SCRATCH root's `jean-zay/full/ARM/attempt-0/`.
All runs are scheduled; running/completed evidence requires fresh Slurm/artifact
observations. Monitoring ownership will be recorded after its live first probe.

### Startup incident and authorized operational recovery

The first live observation found healthy Conv1/Conv2 task0 training, systematic
startup failures for task indices above0, and Conv3 task0 CUDA OOM. The wrapper
already selects one config but exact_run also inherited SLURM_ARRAY_TASK_ID;
pass explicit `--index 0` to prevent double indexing. Preserve first attempts
and their charges; do not rerun the healthy Conv1/2 task0 cases.
Conv3 requested13.78GiB with14.00GiB allocated and13.80GiB reserved but unused;
allocator fragmentation may be responsible. The first recovery tests execution
setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, keeping batch128,
float64, gradient code and all science unchanged. No different GPU is permitted.

These actual target-specific failures invalidate the local-only gate for the
affected launch contract. Before releasing repaired production, run one minimal
live V10032GiB array canary through the repaired wrapper/source path and require
semantic/artifact/resource validation. Planned recovery launch is the v2 revision
under the owning result root's `launch/jean-zay/` with a preserved v1 wrapper and
receipts. Expected remote canary outputs remain under the recorded SCRATCH root
`jean-zay/smoke/`, separated by arm/attempt; exact v2 WORK path, command and ID
will be recorded as resolved. This record authorizes only the operational gate
and replacement of failed/unstarted cases, within the same100GPU-hour ceiling.

Root/Jean Zay Codifier own this recovery. The persistent watcher initially
started asPID979122 and correctly detected failures; its dispatched incident
was stopped to avoid competing repairs. The monitor is switched to passive
observation while the Codifier repairs the launch. Healthy training remains in
place. Recovery dispatch returns only after the new handles and gate are
verified. No epoch0 rerun of completed training is authorized.

The resolved remote repair wrapper is in the existing WORK root's `launch-v2/`,
with original numerical source untouched. Canary launch name
`b128-v2-conv3-ep`, single-task array0-0, selects Conv3 ours EP eta5e-4 at full
batch128 through an explicit case override. Resource contract:
`umg@v100/gpu_p13/qos_gpu-t3/v100-32g`,1GPU4CPUs. Expected output:
`jean-zay/smoke/conv3_ours_ep_eta5em4_batch128_seed0/canary-v2/` below the
recorded SCRATCH root. Required wrapper fix is explicit exact-run `--index 0`;
allocator setting is `expandable_segments:True`. Static tests and a fresh
live semantic gate precede replacements. At07:04Paris all43 affected old tasks
were already terminal, so no cancellation was necessary; only295170_0 and
295176_0 remain healthy. Failed allocation charge is about0.27GPUh pending
exact reconciliation. Passive monitorPID981906 observes all45 cases with no
probe errors; automated recovery dispatch is off during this coordinated repair.

Recovery canary295268_0 was submitted on umg@v100/V10032 using the frozen
v2 wrapper and allocator setting, with the4h upper wall cap retained from its
reviewed resource contract. A second minimal planned gate covers the fmu
account/Conv2 BPTT path: launch name `b128-v2-conv2-bp`, array0-0, explicit
caseindex9 (Conv2 ours BP), `fmu@v100/gpu_p13/qos_gpu-t3/v100-32g`,1GPU4CPUs,
90-minute cap, output `jean-zay/smoke/conv2_ours_bp_eta0_batch128_seed0/canary-v2/`.
It runs full batch128 through the same repaired wrapper/source and artifact
path. Both gate allocations and all failed allocations count toward100GPUh.
No production replacement is released before its applicable gate passes.

Canary295268 (Conv3 EP full batch128) completed0:0 in53seconds on required
V10032GiB after the allocator-only correction. Numerical source remains
unchanged; no gradient-memory rewrite was applied. The separate fmu Conv2
BPTT canary is295276_0, with semantic/resource validation pending at this
handoff observation. The corrected `--index 0` behavior was additionally
tested against the real exact_run config selector with inherited Slurm index13.
The repaired contract can replace each account's failed cases only after its
respective live gate validates. The two original healthy task0 cases remain
canonical. All old receipts, stderr and failed output directories are retained.

The umg V100 gate passed full semantic/hash/resource validation. Conv3EP
peak CUDA allocated/reserved was27.78/27.88GiB, within32GiB; no numerical
implementation change is needed. Fmu Conv2BP canary295276 completed0:0 in
52seconds and undergoes the same gate validation. Corrected production uses
immutable `launch-v2/` and attempt-separated outputs. Only the43 failed cases
are replaced;295170_0 and295176_0 remain original canonical runs. Before any
replacement, verify its old attempt terminal and preserve source/config IDs,
initializer, failed allocation charge and all previous artifacts.

### Withdrawn concurrency steering — no eight-GPU total limit

Filip subsequently requested eight V100s in parallel. At the last verified
observation Slurm had already admitted25 training tasks under the earlier
per-array caps. Preserve these healthy allocations; the user did not request
preemption or discarded work. The new target is **eight GPUs total across the
study**, not eight per architecture.

For the remaining queued cases, use native Slurm dependencies rather than a new
admission service: briefly hold only pending owned tasks, snapshot the currently
active owned task IDs, and organize pending cases into at most eight chains.
The first task of every chain waits for all currently active tasks to finish;
each subsequent task waits on its own predecessor with `afterany`, because
these scientific cases are independent. Prioritize longer Conv3/Conv2 cases.
Verify dependencies and V10032 constraints before releasing pending holds.
Existing parent array caps8 remain secondary; the eight dependency chains
bound simultaneous future allocations after the current wave finishes.

The ordinary persistent watcher remains the sole read-only progress/recovery
owner. Record dependency mapping and live verification beside Slurm receipts;
future recovery submissions must preserve the eight-GPU admission limit and
existing100GPU-hour budget. No numerical changes, new cases or extra seeds.

Filip withdrew the eight-total limit immediately after learning that25 V10032GiB
jobs were already running: “even better — do not apply eight gpu limit”. This
latest instruction supersedes the proposed chain procedure above. Remove any
just-applied chain dependencies from the19 pending owned tasks, release their
holds, and retain the original array concurrency8 EACH (up to24 replacement
jobs plus the retained original Conv2 job). Healthy running jobs are unchanged.
No eight-total admission cap or extra admission controller remains authorized.
Preserve the temporary hold/dependency receipts as withdrawn operational history;
the persistent watcher continues the original45-case monitoring/recovery scope.

### September29 slowdown diagnosis and planned state-preserving migration

At10:37Paris, Conv3 legacy EP eta1e-5 (`295298_11`, raw295364) on
`r8i7n7` had completed5/30epochs. Its latest two epochs took1268/1411seconds
versus about233seconds for matched Conv3 cases. Read-only checks inside its
allocation found repeated Lustre shared-library major page faults while executing
`libtorch_cpu.so` and NumPy: about2.35million major faults, rising1726 in3seconds,
with56GB physical reads but only286MB application reads. The worker waits in
`cl_sync_io_wait`; GPU utilization is0% with normal clocks, no thermal/power
throttle, and normal memory headroom. Job cgroup use is0.87GB/16.78GB with zero
limit/OOM events and the node has over100GB available. This implicates the
node/shared-library I/O path rather than a scientific or checkpoint-write issue.
The earlier completion estimate for this one case is withdrawn.

The Codifier prepares a migration of only this case, retaining its full saved
model/Adam/global and private RNG state at the latest completed epoch, unchanged
source/config/module/wrapper and V10032 constraint. Exclude `r8i7n7`; never
restart epoch0. Before replacing it, verify its checkpoint and remaining100GPUh
budget, pass the applicable target gate, stop only this allocation, verify it
terminal, and submit one continuation into `jean-zay/full/conv3_legacy_ep_eta1em5_batch128_seed0/attempt-2/`.
Planned gate name `b128-io-recovery-gate`, same umg@v100/gpu_p13/qos_gpu-t3/
v100-32g/1GPU4CPU v2 wrapper, singleton0-0 smoke selecting original case11,
output `jean-zay/smoke/conv3_legacy_ep_eta1em5_batch128_seed0/io-recovery-gate/`.
Gate and resumed allocation retain4h upper caps; actual charged time is
reconciled before release. Evidence goes under existing result root
`launch/jean-zay/recovery-legacy-eta1em5-20260929/`. Watcher995295 remains the
sole progress owner; its recovery dispatch is coordinated during migration.

The migration gate301397 passed on r8i1n0 in124seconds. Original slow task
295298_11 was stopped only after checkpoint verification and is terminal
CANCELLED with4789seconds charged. Continuation301691_11 failed before training
on a strict restore-contract mismatch. Diagnosis found one transport-only
difference: the trainer hashes the declared Jean Zay dataset root, whereas the
diagnostic-resume helper hashed the frozen config's local dataset path. Model,
Adam, RNG, source and all scientific fields agree; no saved contract is edited.

A narrow `launch-v3/resume_entry.py` supplies the same declared dataset-root
override to the restore-contract constructor that the frozen trainer already
uses. It retains all original manifest/checkpoint checks and strict full-contract
equality. The frozen training source, config and checkpoints remain unchanged.
18 focused tests pass, including an actual epoch5 checkpoint: the original
failure is reproduced, corrected context restores exact saved contract and
private RNG, and an incorrect root is still rejected. Planned fresh gate
`b128-restore-v3-gate` uses the same V10032 resources and excludes r8i7n7, with
smoke output `jean-zay/smoke/conv3_legacy_ep_eta1em5_batch128_seed0/restore-v3-gate/`.
After it passes, only this case continues from attempt1's latest valid state
into `attempt-3`, preserving attempt2's failure. Watcher1171588 is temporarily
passive while the Codifier owns recovery, preventing competing retries.

Fresh v3 gate302030 was queued; its smoke-only wall cap was reduced to15minutes
from4hours to permit earlier backfill. Prior equivalent smoke took124seconds;
all resource/scientific settings remain unchanged, and production retains4hours.

Fresh v3 gate302030 passed with exit0:0 in73seconds on r6i6n5, with valid
three-epoch full-batch128 smoke artifacts and unchanged source/config identities.
Continuation `302485_11` now owns only the legacy eta1e-5 case in `attempt-3`,
using the verified epoch5 model/Adam/global and private RNG state from attempt1.
It remains queued for V10032, excluding r8i7n7, with a4hour cap. At11:06Paris
the scheduler forecast14:18Paris start; matched healthy throughput predicts
about97minutes after start (roughly15:55Paris if that forecast holds). Actual
restoration and epoch6 throughput remain pending; this is not a completion claim.

Sole watcher1179513 is verified live with `--dispatch` and a fresh error-free
11:06Paris observation:41 remote-complete,3 running,1 queued. All earlier
watch/incident histories and GPU charges are preserved. The aborted competing
incident did not submit or change jobs; no incident is currently inflight.
The existing watcher owns subsequent restore/progress checks and any bounded
recovery. See `launch/jean-zay/recovery-legacy-eta1em5-20260929/final-handoff.json`
and `monitoring-handoff.json`. Other44 cases were not changed by this recovery.
The100GPUh total and Oct6 queue deadline remain unchanged; last pre-recovery
accounting was35.698GPUh consumed, leaving ample margin for the single4hour
continuation and minute-scale canaries. Final accounting must include all attempts.

At11:48Paris, pending continuation302485_11 was shortened in place from4hours
to2hours to improve V100 backfill eligibility. Matched completed legacy Conv3
EP cases eta0/1e-4/5e-4 used7056/7037/7026seconds for all30epochs;25remaining
epochs project at most5880seconds, leaving1320seconds of startup/checkpoint
and runtime headroom. No job was duplicated, and no source, wrapper, numerical
config, account, QoS, hardware constraint or excluded node changed. The wrapper
contains no separate timeout. Watch metadata and contract identity were
reconciled with all dispatch/budget history retained, then the same sole service
was restarted with dispatch asPID1198757. Evidence is
`launch/jean-zay/recovery-legacy-eta1em5-20260929/two-hour-adjustment.json`.
The earlier14:18Paris forecast had already moved toOct2 00:10 before this change;
queue forecasts are provisional and no fixed completion time is promised.
