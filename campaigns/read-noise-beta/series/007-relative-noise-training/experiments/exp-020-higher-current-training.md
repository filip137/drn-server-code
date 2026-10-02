---
id: "exp-020"
title: "Seed0 A4 B2 clean and relative-noise training at twice ours learning rates"
status: "running"
hypotheses: ["H-012"]
---
# Exp020 — Higher current amplification

## Current timeout recovery authority — September 28

Filip explicitly instructed: “if it goes over timeout just do epochs from the
last saved point [no] need to restart all”. This supersedes the earlier
restart-from-initializer requirement when diagnostic checkpoints lack exact
noise/data generator state. On a future timeout, preserve the latest valid
completed-epoch model and Adam checkpoint and continue only the remaining
epochs to the original total horizon. Restore available state and label any
missing/reconstructed stochastic state honestly; do not claim bit-exact
uninterrupted replay. A timeout is not authorization to reset to epoch0.
Keep all failed/partial receipts and charges, recover to a separate linked
bundle, and retain the existing study budget and09:00 target. Healthy active
runs continue without another interruption. OOM/nonfinite incidents still
require diagnosis and are not automatically covered by this timeout policy.
A checked continuation bridge is being prepared; it must preserve optimizer
moments/step counters, original config identity, epoch numbers and history.


### Applied recovery — September 29, 00:00 Paris

The unstarted held `amp2-conv2-legacy-eta1em4-20260928-restart1` was
cancelled with zero attempts. Its replacement
`amp2-conv2-legacy-eta1em4-20260929-resume28` is running on the local RTX3090,
supervisor808767 / runner808782, queue token
`1d5fa88a3316468c9eaa7db1734a0238`. The persistent shared queue owns monitoring
and collection, with its first live observation at00:00:08 Paris and 60-second
cadence. The next check is its ordinary service poll. Healthy runs were untouched.

This restores the archived Riri model and Adam at epoch28 and runs epochs29–30
only, using `--resume-diagnostic-run`, `--resume-diagnostic-epoch 28` and
`--epoch-chunk-size 30`. Expected additional runtime is about10minutes with a
30-minute ceiling and unchanged09:00 deadline. Failed-attempt charges remain;
this replaces a six-hour unstarted retry allowance within the original80GPUh.
The receipt identifies restored global RNG, reconstructed loader RNG and missing
endpoint-noise generators/draw count. Results are explicitly approximate
stochastic continuation, never an uninterrupted trajectory.

Output: `results/conv123-current-amp2-20260928-v1/production/conv2_legacy_relative4_eta1em4_seed0-resume28/`.
Queue spec: `results/conv123-current-amp2-20260928-v1/launch/jobs/amp2-conv2-legacy-eta1em4-20260929-resume28.json`.
Recovery source: `results/conv123-current-amp2-20260928-v1/source-diagnostic-recovery-v1/`,
identity SHA256 `06b8b44f70a98dc036d36846ea5cfa2ec8c2a34b10fc1918a2f51fcaac122c47`.
This overlay changes continuation orchestration only and excludes unrelated
working-tree target-scale changes. Original source and Riri evidence are retained.
49 focused recovery, continuation, exact-run and analysis tests passed.
The existing completion hook validates reporting and restored Adam step counts,
collects and analyzes this replacement through the updated jobs index. Inspect
with `gpu_queue.py status amp2-conv2-legacy-eta1em4-20260929-resume28`.
Future timeout incidents return to the Codifier under the authority above;
this does not install automatic resume after an exhausted queue budget.

## Authorized scope and frozen decision

September 28: Filip said “yes, get started. I think we can set the lr to be 2x
of ours. initially just do one seed”. This authorizes execution of the staged
[X-010](../../../explorations/X-010-higher-current-amplification.md) comparison,
replacing its LR sweep and seed replication with fixed2x ours rates, seed0 only.
No LR search or extra seeds. First run clean Conv1/2/3, then eta=1e-5 and1e-4.
The wider noise grid remains outside this first tranche.

New arm amp2: A=4,B=2. Target D/F1/4/6; horizons10/30/30 for Conv1/2/3.
Calibrate the new arm's beta at initialization on the accepted576-example
validation cohort (1% tolerance, at most3 correction rounds); freeze thereafter.
Internal beta=beta_inj/2^depth. Exactly double every ours weight rate; biases0.
Keep the existing numerical/data/optimizer settings from exp014/019: centered
frozen-current EqProp, physical20-output squared error, float64, batch16/64,
input40/100/360, T=K4/6/8, explicit perfect-diode dictionaries, weights[0,100],
frozen zero biases, deterministic55k/5k MNIST splitseed0. No official test.
Noise v_read=v+eta*abs(v)*xi, same eta across all noninput layers including
readout, independent phases/nodes/layers, std0/no floor, acquisitionseed2026081601.
Validation, physical relaxation and teaching force remain clean.

Nine new amp2 trainings (clean plus two noise levels across three depths).
Reuse six clean ours/legacy seed0 controls from exp016/018, and four noisy
Conv1 ours/legacy cells from exp014 after source/config/history matching.
Add eight fresh Conv2/3 ours/legacy noisy controls because exp015's epoch10
Adam reset prevents a matched uninterrupted comparison. Total17 new trainings,
450epochs;27 case endpoints including10 reused controls and18 paired gaps.
All new trainings start from accepted identical seed0 weights within each depth,
with uninterrupted Adam and original per-scheme comparator rates/betas.
Retain optimizer/model/RNG state on any operational resume; no epoch10 reset.

The 9-context initialization cosine replay in X-010 is a subsequent mechanism
measurement, not a training gate or a prerequisite for this fixed-rate launch.
No new T/K/rho/gradient qualification is assigned. Same-runner execution smoke,
source/initializer identity, finite guards and artifact validation remain.

## Evidence and decision

[H-012](../../../hypotheses/H-012-higher-current-noisy-learning.md) owns the claim.
Primary fixed epoch10/30/30 validation accuracy, paired amp2-minus-ours/legacy,
clean penalty and degradation from each scheme's clean endpoint. Save complete
curves and all failures. A single-seed positive cell is exploratory only;
>=0.2pp motivates replication, not significance. No best-epoch comparisons.
Clean deficits do not automatically cancel the noise stage. If a clean run
becomes nonfinite, pause dependent noise cases at that depth and return to the
Codifier without changing scientific settings. Calibration failure likewise
blocks that depth; unaffected depths may proceed. Noisy failures remain outcomes.

## Budget and result paths

Study root registered before preparation:
`results/conv123-current-amp2-20260928-v1/`.
Configs: `configs/conv/current_amp2_20260928/`.
Expectedresult: `../results/exp-020-higher-current-training.md`.

Lean allowance80GPUh including calibration, smoke, all17 trainings, collection
validation and operational retries; 72h from first production admission.
Provisional allocation: Conv1 lane3GPUh, Conv2 lane27GPUh, Conv3 lane50GPUh.
Estimate35–55GPUh total, approximately12–30wallhours with available parallel
lab capacity; reforecast from measured shared throughput. Caps are ceilings.
No more admissions if a full case cannot fit remaining budget/deadline. GPU
selection follows current queue/fleet policy; no unrelated work is interrupted.

## Execution and monitoring handoff

Root Codifier prepares source/configs and resource placement. Shared run-watch
queue will own each submitted lane after its fresh service observation is
verified. No second monitor. Expected validation metrics each epoch;45min stale
threshold with60s scheduler observations. Handles, exactcommands, config/source
identity, deadline, budget and inspect/collect instructions follow before handoff.
Preparation in progress; no production submitted yet.

### Placement steering

Filip explicitly assigned Conv3 to Loulou, confirming its approximately1.2GB
existing occupancy. Live verification found only Ben:1168MiB allocated,30970MiB
free on RTX5090. Honor this requested shared placement using measured peak plus
headroom; the generic31GiB reservation exceeds current free memory despite this
workload fitting. This is an exp020/Loulou-specific placement exception, not a
fleet-policy change. Use one exclusive queue worker with default resource profile
and an explicit actual-memory bound on this lane; preserve Ben's jobs/MPS.
Conv1/2 use local3090. No Trex placement is assigned.

### Individual shared-queue jobs

Filip requested that remaining runs enter the shared queue and distribute across
eligible GPUs. Seventeen independent job specifications are in study
`launch/jobs/`, indexed by `launch/jobs.json`. First submit the3cleanamp2jobs;
Conv3 is pinned to Loulou with its explicit measured-memory exception. Conv1/2
can use local/nom-cool-1 or available5090s. The14noisy jobs are prepared for
automatic submission by `collect.py` once all3cleanamp2 endpoints validate.
NoisyConv3 uses the normal conv3 queue profile onTrex/Fifi/Loulou/Riri; Conv2
uses conv2. The scheduler protects unidentified owners and enforces fleet limits.
Different matched schemes may run on different eligible hosts, as requested;
record environments and interpret single-seed comparisons with that caveat.

Queue budgets total71.5GPUh (0.5h eachConv1,4h eachConv2,6h eachConv3), leaving
8.5GPUh of the80h allowance for completed calibration/smokes, collection and
overhead. Common deadline2026-10-01T13:38:56.995691Z. Expected compute35–55GPUh;
no deadlines/budgets reset on operational recovery. No automatic training retries
or weights-only resumes; bounded queue recovery may diagnose and return toCodifier.

Calibration passed accepted576examples onlocalGPU with no parameter updates:
Conv1 injected/internalbeta5.696269386852665/2.8481346934263323, achievedD/F
1.0000003215672977; Conv2 2.286212994352934/0.5715532485882335, D/F
4.000002347444158; Conv3 0.7574642494273278/0.09468303117841598, D/F
6.000000122892841. All below1% tolerance. Matching evidence in`calibration/convN/`.
Three exact-run CUDA optimizer-step smokes passed in`checks/smoke/`; Conv3
peak reserved4737466368bytes (~4518MiB). Queue bound6144MiB plus2048headroom.

Source snapshot:`source-training-v1/`, identities:`source_identity.json`.
Source/config/initializer checks run again at queue admission. Target Python
`/home/filip/miniconda3/envs/py312/bin/python`, dataset`/home/filip/datasets/mnist`;
only ordinary-MNIST training files are read. Exact per-target argv, output paths,
validators and completion hook are in each prepared job JSON.

Collector:`python results/conv123-current-amp2-20260928-v1/collect.py` receives
RUN_WATCH_JOB_ID fromqueue; remote bundles rsync into`collected/HOST/production/ARM/`.
It validates the case, serializes partial analysis, and submits the idempotent
noisy specs only after3cleancompletions. Individual collection receipts are
in`launch/collected/`; no other watcher owns these jobs.
Analyzer:`MPLCONFIGDIR=/tmp/amp2-matplotlib /home/filip/miniconda3/envs/py312/bin/python
-m experiments.analyze_current_amp2 --study-root results/conv123-current-amp2-20260928-v1
--allow-partial`. All10historical controls validated; source/environment and
historicalConv1missingoptimizer-checkpoint limitations are explicit inanalysis.
Full scientific review remains withReviewer after all17newoutcomes reconcile.

### Transport recovery before training

Initialcleanjob specs included hundreds of redundant prepared-file entries,
exceeding the OS argument limit at Popen beforeworkerstart. The service reconciled
Conv1local/Conv3Loulou attempts as`not_started` after admission expired; Conv2
was stillqueued. All three originalIDs were cancelled through the queue; no
trainingprocess existed and no scientificcoverage or GPUtime was consumed.
Original specs/status are preserved in`launch/`. Corrected clean IDs append
`-transport2`, with distinct output paths and the same configs/seed/beta/LR,
remaining budgets and original deadline. All17specs now check the frozen source
manifest plus core numericalfiles/config/initializer atadmission; full staged
2045-file identities were verified onall5remote hosts withzero mismatches.
Noisyspecs were corrected before submission.

### Verified monitoring handoff

Sharedqueue servicePID3007181 is active with a fresh observation. Corrected
clean jobIDs are`amp2-conv{1,2,3}-amp2-eta0-20260928-transport2`. Conv1 runs
onlocalRTX3090 (exactrunner570418; supervisor570403), Conv3 onLoulouRTX5090
(exactrunner140335; supervisor140332, Ben-sharing recorded). Conv2 isqueued
for eligible capacity. Both active workers passed the second semantic check:
Conv1 completed epoch1 at95.04% validation and reached epoch2batch1000 with
finite loss0.1030; Conv3 reached epoch1batch500 with finite loss0.2173. These
are progress observations, not outcome comparisons.

Current compact receipt:`launch/monitoring-handoff.json`; complete queue handles
and logs are available via`python /home/filip/.codex/skills/run-watch/scripts/gpu_queue.py
status JOB_ID`. Queue polls every60s; nextcheck within60s of its latest observation.
The persistent service owns monitoring and bounded incident dispatch, and each
job's completion hook owns collection/validation.14noisy cases are prepared for
automatic queue admission after all3cleanamp2 endpoints validate. No second
watcher or live polling agent. Scientific review remains pending finalcoverage.

### Nom-cool-1 admission after explicit Kellian-sharing instruction

Filip requested immediate use of nom-cool-1 after cancelling his previous job,
and explicitly confirmed that scheduling alongside Kellian is permitted. Live
inventory showed 19,335 MiB free, Ben's MPS server and two Kellian workers.
The first automatic approval review rejected a proposed global queue change;
no settings changed then. After Filip's explicit instruction, the approved
update added `kellian` to nom-cool-1's `sharing_users` alongside `ben`. The
existing queue service was reloaded without stopping training workers. No
other target's sharing policy, GPU settings or existing job was changed.

Conv1 clean had already completed and validated. Its two existing noisy cases
are now pinned to nom-cool-1, submitted at priority 10, with unchanged configs,
budgets and deadline. This user-directed admission supersedes waiting for all
three clean depths for these two cases only. The remaining twelve noisy cases
still enter automatically after the complete clean stage validates.

`amp2-conv1-amp2-eta1em5-20260928` was admitted on nom-cool-1, token
`e011ded0e12241e58003ecbb73dcbcda`, with sharing recorded as Ben and Kellian.
`amp2-conv1-amp2-eta1em4-20260928` follows in the same queue. Service PID 585501
has a fresh observation; the existing per-GPU queue concurrency rule is retained.
Evidence: `launch/nom1-immediate-admission.json`; prior queue configuration is
preserved in `launch/queue-before-authorized-kellian.json`. Callback submission
is idempotent, so the later clean-stage callback cannot duplicate these jobs.

### Automatic admission correction and Trex placement

Filip freed Trex and expected the queue to use available capacity automatically.
The original callback withheld all noisy jobs until every new clean depth
completed. That global barrier was unnecessarily restrictive. `admit_ready.py`
now submits each noise case when its own (architecture, scheme) clean control
has validated. Thus eight ours/legacy noisy controls entered immediately using
the accepted historical clean controls; the two Conv1 amp2 cases were already
submitted. Four Conv2/3 amp2 noisy cases remain dependent on their respective
clean endpoint, with automatic submission by the existing completion callback.
Scientific configs, counts, horizon, deadline and aggregate budget are unchanged.
Pure admission checks verify10 eligible now and12 after Conv2 clean completes,
without prematurely admitting Conv3 amp2.

The scheduler initially selected Riri for Conv3 ours eta1e-5. Because Filip
explicitly requested Trex, withdraw that just-started agent-owned attempt through
the queue and requeue the same case on Trex, retaining its attempt and measured
charge. No other workload is stopped. The daytime two-5090 cap remains intact.
The unrelated first Conv1 noisy case hit its1800s per-case allowance on the
heavily shared nom-cool-1; recovery is being diagnosed within80GPUh total.

Trex replacement admitted as `amp2-conv3-ours-eta1em5-20260928-trex`, token
`49d8749e25494b808cfaf0c8fd6eda88`. The Riri attempt was confirmed stopped with
no live descendants/reservation and 101.00922846794128 charged seconds. Its
21,600-second allowance is reduced to21,498.99077153206 for the replacement;
original deadline is unchanged. Riri evidence is preserved remotely and its
queue receipt is in `launch/riri-superseded-trex.json`. All seven temporary
study-only placement holds were released after the Trex-pinned replacement
was submitted at higher priority. The remaining ready cases are queued, not
withheld. Four amp2 noisy cases retain only their own depth's clean dependency.

### Conv1 contention timeout and unchanged restart

The first nom-cool-1 eta=1e-5 attempt timed out after 1800.032906 seconds,
just after epoch1, while sharing with two Kellian workers. Preserve that charged
0.500009 GPUh and its original remote bundle; a hash-verified local copy is in
`attempts/nom-cool-1/amp2-conv1-amp2-eta1em5-20260928/` under the study root,
outside production/collected discovery. Model and Adam epoch1 checkpoints are
intact (3438 updates), but diagnostic checkpoints omit sampler, worker and
endpoint-noise generator states. They cannot support exact full-state resume;
the existing planned-pause continuation API cannot resume this attempt.

The following eta=1e-4 run remained healthy: four epochs completed with ~47.17s
per epoch after the Kellian GPU workers departed, versus ~1799s for the failed
first epoch. Historical matched Conv1 noisy runs took ~507–511s total. This
supports contention as the slowdown, without requiring scientific/source changes.
Leave eta=1e-4 running. No unrelated process was stopped or modified.

Submitted `amp2-conv1-amp2-eta1em5-20260928-restart1` through the shared queue,
nom-cool-1 only, using the identical source/config/seed0 initializer and distinct
`production/conv1_amp2_relative1_eta1em5_seed0-restart1/` destination. Expected600s,
cap1800s; allocate this additional0.5GPUh from the original study reserve and
retain the failed0.500009GPUh charge and October1 deadline. The old specification
is preserved; `launch/jobs.json` now names the replacement so callbacks remain
idempotent. The service acknowledged the queued replacement with a fresh
observation and capacity-wait reason; shared run-watch retains sole monitoring
ownership. Evidence: `launch/conv1-nom1-recovery-diagnosis.json`,
`launch/conv1-restart1-queue-status.json`, `launch/conv1-restart1-service-ack.json`.

### Status update — September 28, approximately17:32Paris

Three of17new runs completed, collected and validated (allConv1 amp2 cases);
four active, six queued, four awaiting their own clean depth. Active completed
epochs: Conv2 amp2 clean23/30local, Conv3 amp2 clean13/30Loulou, Conv2 ours
eta1e-5 8/30nom-cool-1, Conv3 ours eta1e-5 4/30Trex. Latest compact observations
are in`launch/latest-update.json`; service heartbeat fresh, no new incidents.
The earlier Conv1 timeout restart completed successfully; archived raw status
is stale but its queue receipt proves stopped, and it is not an active run.

Conv1 fixed epoch10 validation (amp2 / ours / legacy): clean96.08/96.32/96.42%;
eta1e-5 96.04/96.34/96.40%; eta1e-4 96.12/96.28/96.38%. Thus no Conv1 accuracy
advantage at these settings in this seed. Deeper training remains incomplete;
no whole-study verdict. Analysis validates13outcomes including10saved controls,
with zero errors.

## Authorized clean seed-1/2 supplement

Filip requested two additional noiseless A=4,B=2 seeds and queue submission.
Interpretation: seeds1/2 for each Conv1/2/3, six new runs, retaining exactly2x
ours weight-rate vectors, frozen zero biases and seed0-calibrated betas. Horizons
remain10/30/30 epochs; eta=0 and absolute noise std=0. Model initialization and
training shuffle use the new seed; the55k/5k split stays seed0. Reuse accepted
seed1/2 initializers from exp018 (Conv1) and exp016 (Conv2/3), matched to their
legacy/ours controls. D/F1/4/6 labels refer to seed0 calibration; actual new-seed
displacement is unmeasured. No recalibration, LR search or extra diagnostics.

Supplement root registered before creation:
`results/conv123-current-amp2-clean-seeds12-20260928-v1/`.
Configs:`configs/conv/current_amp2_clean_seeds12_20260928/`.
Keep parent17-case scheduling and evidence unchanged. Total assigned new coverage
is now23 cases: original17 plus supplemental6. Original budgets/deadlines stay
unchanged; this newly requested supplement has its own24GPUh allowance including
smokes, six trainings and retries, and72h from submission. Expected12–16GPUh,
subject to placement/sharing. Queue per-case ceilings0.5/4/6h for Conv1/2/3
sum21GPUh, leaving3GPUh for operational overhead/recovery.

Submit all six individual jobs immediately once prepared. No clean-stage
dependencies remain for these fixed clean replications. The shared queue chooses
eligible lab GPUs; Conv2 uses its two-worker5090 profile, Conv3 the normal31GiB
reservation. Original Loulou-specific exception does not silently extend to
these additional cases. Current explicit Kellian sharing on nom-cool-1 remains
in force; retain other ownership protections and fleet limits.

Use the parent's frozen numerical source, new immutable configs and accepted
seed checkpoints. Each assigned GPU executes a one-step same-runner smoke then
full training; smoke/production outputs are separate. Current source smoke
coverage for all three depths already passed locally in exp020. Require new
seed identity/zero-bias checks, exact-run finite guards and terminal validation.

Report fixed-horizon clean accuracies for seeds0/1/2, means/sampleSD, and paired
amp2-minus-ours/legacy gaps. Reuse the corresponding historical controls after
matching seeds/configs; host/source caveats remain. Scientific conclusions are
limited to three seeds at the selected amplification-plus-LR policy. Seed0 is
linked to its original run, never rerun or counted twice. Supplement analysis
and callback own the six new cases; combined scientific review must reconcile
both roots. Queue handles/identity/collection instructions follow preparation.

### Supplemental execution and monitoring handoff

All six jobs submitted and acknowledged queued by `run-watch-queue.service`
(PID585501), with a fresh observation at 2026-09-28T16:14:02.332801+00:00.
The service owns placement, monitoring and bounded recovery at60-second cadence;
no second watch is active. All six currently wait for eligible capacity.
Job IDs: `amp2-clean-conv1-seed1-20260928`,
`amp2-clean-conv1-seed2-20260928`, `amp2-clean-conv2-seed1-20260928`,
`amp2-clean-conv2-seed2-20260928`, `amp2-clean-conv3-seed1-20260928`,
`amp2-clean-conv3-seed2-20260928`.

Supplement `launch/jobs.json` links exact commands and placement specs;
`launch/monitoring-handoff.json` preserves acknowledgment. Deadline
2026-10-01T16:10:29.762397+00:00;24GPUh supplemental ceiling, zero charged at
handoff. Allowed targets local/nom-cool-1/trex/fifi/loulou/riri for Conv1/2,
and trex/fifi/loulou/riri for Conv3, subject to live admission policy.
Seed identity, zero biases and config checks passed; all2060 staged files
matched their manifest on all five remote hosts. The unchanged numerical
source matches the parent snapshot. Preparation provenance is in the new
config directory's `provenance/preparation.json`.

Inspect with `python /home/filip/.codex/skills/run-watch/scripts/gpu_queue.py
status JOB_ID`. The supplemental `collect.py` completion hook automatically
collects remote production and smoke artifacts, validates each run and refreshes
partial three-seed analysis. Run analysis manually with
`python -m experiments.analyze_current_amp2_clean_seeds --study-root
results/conv123-current-amp2-clean-seeds12-20260928-v1 --allow-partial`.
Current analysis validates20/27 clean comparison outcomes, zero errors;
six new runs and parent Conv3 seed0 remain pending. Recovery may repair
operational issues within the unchanged science/budget; material scientific
choices return to the Explorer. No new scientific conclusion yet.

### Supplemental Conv1 seed1 collection recovery — September 28

Assigned job `amp2-clean-conv1-seed1-20260928` completed on nom-cool-1,
worker token `0514c6da10464b868740ddde0d47a598`: training and remote validation
both exited0, charged471.773210seconds. Queue receipt confirms terminal worker,
no live supervisor and no reservation. Controller collection token
`1458f5effdf94c0489ca643f6a03df79` terminated with exit1 after local bundle
validation passed; its failed receipt/log remain preserved in queue history.

The supplemental analyzer incorrectly compared different digest schemas:
preparation uses `drn-ordered-parameter-state/v1`, training uses
`mnist-train-ordered-parameter-state/v1`. The accepted Conv1 seed1 checkpoint
reproduces preparation digest `dc48d076f60f0718efcb465a65690348b8b37d03a12c7f9d049075161e26f537`
and actual training digest `397c2a14332ec8bd2b882c6743e6be8425ccec90efa93c4e3202d869c76c31f4`
from identical float64 tensor bytes. Controller analyzer now verifies the
checkpoint file identity and both schemas without modifying frozen preparation
or run artifacts. CPU checks passed for all six accepted initializers against
the trainer's hashing function; eighteen negative checks rejected altered
checkpoint/preparation/runtime hashes. The assigned collected bundle passed
full validation, including ten epochs and96.32% final validation accuracy.

Request scheduler `collect` only using the existing retry-safe completion hook.
The old collection receipt confirms terminal exit and released reservation.
No compute restart, new job, queue mutation or scientific/budget change occurred;
the scheduler remains sole execution owner. Analysis refresh and successful
collection receipt remain pending that scheduled collection retry.

### Status 2026-09-28T18:19:16.282866+00:00

Supplement: Conv1 seeds1/2 completed, collected and validated on nom-cool-1;
fixed epoch10 validation96.32/96.30%. With seed0=96.08%, mean96.233%,
versus ours96.333% and legacy96.493% across the three matched seeds.
No clean Conv1 advantage at this amplification-plus-LR policy.
Four supplemental Conv2/3 seed1/2 jobs remain queued.

Original17 cases:7 complete,7 running,3 queued. Conv3 amp2 clean seed0
finished98.46%, versus ours98.76% and legacy98.70%. Conv2 amp2 noisy
eta1e-5 is20/30 epochs (latest validation97.44%) on nom-cool-1;
eta1e-4 is4/30 (96.36%) local. Other active controls: Conv2 eta1e-4
ours/legacy16/30 on Riri; Conv3 ours eta1e-5 22/30 Trex, legacy eta1e-5
9/30 Fifi, ours eta1e-4 2/30 Loulou. Partial epochs are not endpoints.
Queue service585501 fresh, admission enabled, no host errors; all current
receipts have no runtime error. Raw compact-handle/receipt snapshots are in
each root launch/status-update-raw.json. Supplemental analysis refreshed to
23/27 clean outcomes, zero validation errors, retaining four pending seeds.
Service remains sole monitoring owner at60-second cadence.

### September 28 deadline recovery and single-seed priority

Filip requested completion by September29 09:00 Europe/Paris (07:00UTC),
prioritizing the original single seed over additional clean seeds. Queue
priority now250 for the endangered Conv3 legacy eta1e-5 restart,200 for
Conv2 control restarts,150 for both pending noisy amp2 Conv3 jobs, and-10
for the remaining clean Conv3 seed2. Healthy active jobs continue unchanged.
Prepared Conv3 replacements may use local/nom-cool-1 RTX3090s as well as
Trex/Loulou/Riri5090s. Numerical source/configs/initializers are unchanged;
standard conv3 profile still applies on5090s. No two-Conv3 concurrency
experiment was launched. Fifi is excluded from these new placements because
an independently started Adrien workload caused severe contention.

The two original Conv2 eta1e-4 controls on Riri exhausted their4h limits at
28/30 completed epochs. Both workers/reservations were confirmed terminal.
Their diagnostic checkpoints omit explicit sampler/noise generator state;
a continuation would not be an exact restart. Preserve both raw attempts at
`results/conv123-current-amp2-20260928-v1/attempts/riri/`, with checksum-verified
copies and original queue charges. Replacement IDs:
`amp2-conv2-ours-eta1em4-20260928-restart1` on Akib3080 (priority200; CUDA smoke
passed, running with finite first-batch loss0.4972), and
`amp2-conv2-legacy-eta1em4-20260928-restart1` queued for local3090.
Both restart the original initializer, use6h caps and the09:00 deadline;
expected runtimes3h Akib and2h local. Source manifest verification passed on
Akib. The existing collector now uses Akib's SSH alias for its distinct user.

Fifi's Conv3 legacy eta1e-5 completed13/30 epochs after nearly4h and could
not finish before its immutable six-hour worker timeout00:14Paris. Cancelled
only our worker through the queue at22:15:53; supervisor/trainer absent and
reservation released, Adrien untouched. Charged14482.105091seconds. Full raw
attempt and receipt preserved at
`results/conv123-current-amp2-20260928-v1/attempts/fifi/amp2-conv3-legacy-eta1em5-20260928/`.
Replacement `amp2-conv3-legacy-eta1em5-20260928-restart1` is queued at priority250,
expected/capped6h, with the same science and09:00 deadline. It will get the
next suitable free GPU, expected nom-cool-1 after its healthy Conv2 finishes.

Replaced only unstarted noisy amp2 Conv3 handles with
`amp2-conv3-amp2-eta1em5-20260928-deadline` and
`amp2-conv3-amp2-eta1em4-20260928-deadline`; cancelled originals retain history
and zero compute charge. New distinct output directories end in`-deadline`.
Expected/capped6h and09:00 deadline account conservatively for3090 placement.
Supplemental unstarted clean Conv3 seed2 likewise replaced by
`amp2-clean-conv3-seed2-20260928-deadline`, retaining6h cap and low priority,
with the same prepared3090/5090 fallback. The other supplemental jobs remain
unchanged. Each root's existing jobs index points to current handles.

Budget audit: terminal original charges including Fifi sum about31.47GPUh;
remaining original caps are24h for the four other Conv3 cases,6h for Fifi's
replacement, and12h for the two Conv2 restarts, totaling73.47GPUh including
charged attempts, below the existing80GPUh allowance with room for recorded
smoke/calibration overhead. Supplement retains its separate24GPUh ceiling.
Neither failure is an additional scientific replicate.

Service585501 acknowledged replacements with a fresh22:18Paris observation;
60-second shared-queue monitoring remains sole owner. Operational evidence:
parent`launch/conv2-deadline-recovery.json`,
`launch/fifi-contention-recovery-20260928.json`, and each root's
`launch/deadline-monitoring-handoff.json`. Completion callbacks collect and
validate all successful replacements and regenerate partial evidence. The
09:00 forecast assumes remaining isolated throughput stays near observed
rates; increased contention or nonfinite/transport failures are exceptions,
not permission to shorten horizons, manufacture results or alter science.

### September 28 23:45 Paris update and measured Conv3 runtime repair

Collected original evidence now10/17 complete, zero validation errors. Conv2
amp2 final epoch30: clean98.00%, eta1e-5 97.64%, eta1e-4 96.82%. Ateta1e-5,
ours97.94% and legacy97.84% are complete; amp2 remains lower. High-noise
controls are not yet complete. Supplemental clean Conv2 seed1 finished97.98%;
seed2 was25/30 at23:42. JeanZay's separate batch-size pilot is pending04:00.

The nom-cool-1 Conv3 legacy eta1e-5 restart revealed a six-hour budget error:
first four epochs775.04,771.16,770.97,771.26seconds, stable about12m51s each.
Expected full training6.43h ends05:06Paris, after its immutable04:41 timeout.
No contention improvement can be assumed (only our trainer plus small Ben MPS).
Recover now by unchanged initializer restart with7.5h allowance rather than
wasting the remaining five hours before a predictable timeout. Current index
points`amp2-conv3-legacy-eta1em5-20260928-restart2`, pinnednom-cool-1,priority250,
expected6.5h,deadline09:00. Prior worker cancellation, archival, charged runtime
and new service acknowledgment are owned by the assigned bounded Codifier;
detailed evidence:`launch/nom1-conv3-runtime-recovery-20260928.json`.

The two still-unstarted noisy amp2 Conv3 jobs and clean Conv3 seed2 now use
only Trex/Loulou/Riri5090s, whose measured4.2–4.6h runtime fits the unchanged6h
caps. Current handles end`-deadline-5090`, priorities150/150/-10, expected5h
and09:00deadline. Old unstarted handles were cancelled with zero charge;
scientific configs and initializers remain unchanged. This supersedes their
3090 fallback, which measured throughput showed was too slow for that cap.
All replacement outputs remain under the registered parent/supplement roots.

Runtime repair completed: restart1 terminated23:45:09Paris, charged3842.654647s;
known supervisor/runner/trainer absent and reservation released. Checksum-verified
archive:`results/conv123-current-amp2-20260928-v1/attempts/nom-cool-1/amp2-conv3-legacy-eta1em5-20260928-restart1/`.
Restart2 started23:46:52Paris onnom-cool-1,token
`e7f7b790e9b44caba31a5a90d0d2d72e`,supervisor30344/exactrunner30345; first training
batch finite(loss0.4994). Hardstop07:16:52Paris; measured completion forecast
06:12:30. Original actual charged time plus all remaining full allowances
76.039863GPUh, leaving3.960137h of80 for recorded preparation/overhead. Shared
queue retains sole monitoring; extra work remains below original single-seed
priority. The earlier six-hour3090 allowance was an operational estimate error,
not a scientific or numerical failure.
