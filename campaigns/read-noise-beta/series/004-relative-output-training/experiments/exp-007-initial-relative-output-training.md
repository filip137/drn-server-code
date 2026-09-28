---
id: "exp-007"
title: "Train Conv1/2/3 at matched initial relative output displacement"
status: "partial"
hypotheses: ["H-006"]
---
# exp-007 — Initial relative output displacement, ten-epoch noisy training

## Assignment and question

Filip assigned targets **0.8, 1.2, 1.6, 2.0 at initialization**, ten epochs
for Conv1/Conv2/Conv3, and read noise **5e-4**, on September 24, 2026. He
specified the hypothesis that ours should outperform legacy and assigned
three V100s for Conv3, Fifi/Loulou RTX5090 for Conv2, and local/Akib for Conv1.
This authorizes preparation, calibration, smoke, training, monitoring and
collection. It adds no relative-read or noisy-nudging arm.

## Cases and held-fixed contract

There are 36 seed-0 trainings: three architectures × three schemes × four
target ratios. Every case runs exactly ten epochs from the shared
architecture-specific initialization; no continuation of the p90/p99 weights.

- Ordinary MNIST, deterministic 55,000/5,000 split; model/split/loader seed 0.
  Official-test reads remain disabled. This is exploratory validation evidence.
- Baseline (voltage,current)=(1,1), ours=(4,1), legacy=(4,.25).
- Native float64, perfect diodes with source-explicit parameter dictionaries,
  wide [0,100] conductances, exact-zero bias tensors and bias learning rates.
- Centered frozen-current EqProp, physical paired 20-node output, source
  squared-error objective. Do not import the earlier CIFAR cross-entropy loss.
- Existing Adam parameterwise learning rates; no new LR/rho search. Preserve
  batch size 16, validation batch size 64, input gain and preprocessing.
- Accepted T=K is **4 for Conv1, 6 for Conv2, 8 for Conv3**. A coordinator's
  initial all-T8 assumption was corrected by source inspection before any
  calibration or training. No additional T/K experiments are authorized.
- Independent additive Gaussian noise sigma5e-4 on noninput positive/negative
  endpoint reads before local energy-gradient formation. Physical relaxation,
  clamped inputs and validation inference stay clean. Use the inherited noise
  seed and identical draw policy across matched cases.

Source starting points are the exact zero-bias shared-4/6/8 family in
`configs/conv/perfectdiode_conv123_zero_bias_adam_eqprop_one_decade_ordinary_mnist_10_30_30ep_seed0_20260816_v1.json`,
the Conv1/3 and Conv2 sigma5e-4 follow-ups of August 17, and the current Conv3
p90/p99 source configs. Resolve exact initializer, config and LR identities
before admitting production. Preserve any relevant source differences in the
prepared configs; do not silently substitute a general paper BPTT contract.

## Calibration

For each architecture/scheme/target, choose injected beta B such that

`r = pooled_RMS((v_plus - v_minus)/2) / pooled_RMS(v_post_T_free)`

equals its target at the output layer within 1% relative error. Pool squared
RMS with element counts, not batch-ratio averages. Use the existing frozen
576-example, 36-batch ordinary-MNIST validation cohort where applicable, with
identical source indices for all schemes. The post-T start and frozen force
are common to both signs; no optimizer update occurs during calibration.

Use a linear-response estimate as the first proposal, then measure the
actual ratio at that beta. Allow at most three correction rounds per target,
retaining every measurement. Stop an unresolved/nonfinite calibration case
without substituting a different target. Reject a zero/unusable free-output
normalizer rather than adding an arbitrary epsilon. Record actual injected
and internal base beta separately: base beta = B/(voltage/current)^depth.
Freeze each accepted beta for all ten training epochs. Do not adapt it during
training or silently rematch at the final checkpoint.

## Execution and placement

Results: `results/eqprop-conv123-relative-output-20260924-v1/`, indexed before
creation. Preparation owns a small calibration runner and 36 readable exact
training configs; record their actual paths/hashes here once finalized.
Use `python -m experiments.exact_run` for training and its synchronous local
`--smoke` path (one optimizer step and one validation batch) for every exact
config. No new catalog, launch schema or control plane is introduced.

| Architecture | Requested resources | Initial grouping |
|---|---|---|
| Conv1 | Local RTX3090 and Akib RTX3080 | Targets .8/1.6 local; 1.2/2.0 Akib; all three schemes per target on the same host. |
| Conv2 | Authorized RTX5090 pool; initially Riri and Loulou | Targets .8/1.6 Riri; 1.2/2.0 Loulou. Trex/Fifi may take whole unstarted target triplets when admissible; all three schemes per target stay on the same host/runtime. |
| Conv3 | Three concurrent Jean Zay V100 allocations | Twelve one-case tasks, array concurrency three, ordered by target then scheme; one V100 class/runtime and one recorded Jean Zay target. |

Host effects can confound comparisons between target levels for Conv1/2;
primary ours-minus-legacy contrasts are within target and host for Conv1/2,
and within the same V100 pool/runtime for Conv3. All shared
initialization, cohort and data-order identities must match. Do not move only
one member of a matched triplet to a different host for speed. Record any
whole-triplet placement change before launch.

Recheck memory and owners at admission. At16:28 Paris, Fifi has Adrien's
large job and Trex has Kellian's small Python job, so neither is admitted
under the Ben-specific sharing rule. Riri and Loulou have
Ben's workload and may be shared only if measured memory plus headroom fits,
without changing or stopping his process. Local currently runs exp-006 until
approximately17:00 Paris; its training queue must wait for that GPU release.
Akib must use CUDA; CPU fallback is prohibited. Calibration may use a short
available GPU lane during preparation. Record its target and duration.

The user's current resource assignment is implemented as launches once
calibration and smoke pass, including daytime operation; retain this explicit
daytime scheduling assumption rather than silently postponing to another
night. No unrelated queue or job may be changed. If runs cross 08:00, this
assigned daytime continuation remains permitted within the cap below.

For Jean Zay use fmu@v100, outputs below
`/lustre/fsn1/projects/rech/fmu/$USER/server_code/results/eqprop-conv123-relative-output-20260924-v1`.
Follow the current local synchronous smoke then one production submission
rule, with no separate remote canary absent a target-specific failure.
The persistent planned row must exist before any live submission. Record
Slurm IDs immediately, require actual CUDA work and semantic artifact progress.

## Measurements and decision

The primary outcome is epoch-10 validation accuracy, paired ours-minus-legacy
at each architecture/target. Report all twelve differences and each
architecture's equally weighted mean, following
[H-006](../../../hypotheses/H-006-relative-output-ours-versus-legacy.md).
Baseline supplies the third-scheme reference. Keep the four target choices
visible; independently choosing the best target for each scheme is not the
primary comparison.

Also report validation loss/trajectory, best validation as a secondary
diagnostic, achieved initial ratio and beta, finite completion/failures,
best-to-final accuracy drop, elapsed runtime and GPU memory. The historical
within-run stability screen (finite completion and final less than five
percentage points below its own best) may be reported separately; it does
not define a high-accuracy result or license excluding poor performers.
No early stop for low accuracy alone. Retain numerical failures rather than
tuning or repeating them until they succeed.

## Budget, deadline and completion

Provisional cap: **60 GPU-hours total**, including up to two hours for
calibration/smoke and understood operational retries. Initial admission caps
are twelve V100 tasks at2h each (24 GPU-hours), plus four
workstation queues at8 hours each (32 GPU-hours), plus2 GPU-hours for
calibration/smoke and2 GPU-hours unallocated recovery reserve. The two
Conv2 queues share a16-GPU-hour allowance; adding Trex/Fifi redistributes
that allowance instead of increasing it. Redistribution requires a recorded measured-throughput
update before admission and must preserve the total cap. No extra seeds, target
ratios, clean-training controls, LR searches or T/K grids. Historical timings
suggest roughly 8–12 wall hours once requested resources are available;
replace this estimate with measured throughput before training admission.
End admission/monitoring at **September 26, 2026 08:00 Europe/Paris**, or the
compute cap, whichever is first. Record partial coverage if queueing prevents
the assigned cases from finishing; never silently extend the budget.

Preserve one execution owner and unique run per case. Diagnose operational
failures before retrying; changes of scientific settings require a new
decision. Stop any training that falls back to CPU. Keep heartbeat and
artifact progress observable and monitor at least every 30 minutes.

Collect remote outputs locally, validate canonical bundles, and reconcile
all 36 original cases, calibrations, smokes, failures and retries. Save
curves/tables/report under the study root, record the interpretation in
`../results/exp-007-relative-output-training.md`, regenerate the campaign
ledger, and update current_simulations and experimental_manifest. Do not
edit the human-authored current_state dashboard.

## Preparation record

The calibration runner is
`experiments/calibrate_initial_relative_output.py`, configured by
`configs/conv/eqprop_conv123_relative_output_calibration_20260924.json`.
It emits accepted exact configs into
`configs/conv/eqprop_conv123_relative_output_20260924/`. Three focused
calibration tests passed. The full local CPU smoke completed all nine
architecture/scheme contexts in251.3 seconds, with fixed T4/6/8, maximum
target-relative error5.65e-6 and all nine endpoint-parity checks against the
established replay passing. These are fixed-setting execution/parity checks,
not extra T/K experiments.

Fifi was idle at its initial admission check but acquired Adrien's large
Python workload during the CUDA smoke. Our smoke failed with OOM and exited;
the other process was not changed. The failed28-second attempt and artifacts
are retained under the study. Fifi training must wait for an admissible lane.
Calibration moved to Loulou, where Ben's verified workload leaves about27GiB
free and the standing sharing policy applies. The isolated calibration
workspace is separate from the training workspace. Its detached command runs
CUDA smoke then full calibration, with an external6900-second limit that
keeps the earlier attempt inside the two-GPU-hour preparation allowance.

Training transport is prepared under the study's `transport/`: the workstation
queues enforce CUDA, require accepted local smoke evidence, retain numerical
failures and stop on operational failure. The Conv3 Slurm array contains
twelve one-case tasks, concurrency three, time limit2h, fmu@v100.
The selected QoS is qos_gpu-dev, partition gpu_p13, constraint v100-16g,
one GPU and four CPUs per task, module pytorch-gpu/py3/2.5.0. This QoS
limits submitted tasks to ten: submit array0-8%3 first, then9-11%3 only
after the first array is terminal. Historical100-minute cases plus a
ten-minute margin fit the two-hour task wall limit.
No training admission is established by these prepared wrappers alone.

The Loulou CUDA smoke passed all nine contexts, reproducing the local CPU
measurements with no OOM, and its launcher2159440 advanced to full calibration.
Filip subsequently added **Trex and Riri**, explicitly allowing sharing with
Ben. They are available for whole target triplets within the unchanged
36-case/60-GPU-hour contract; update the concrete split at training admission.
Fifi's current non-Ben occupancy remains a separate restriction.

A Jean Zay scheduler-only preflight accepted the wrapper after removing an
unsupported explicit memory request, but projected September30 admission,
beyond this study's deadline. No live Slurm job was created by that check.
Eligible alternative V100 queue options are being inspected before committing
to a queue or assigning whole matched groups to the newly authorized GPUs.

At16:28 Paris calibration was complete and collected locally: **36/36
accepted targets**,45 measured points, no correction rounds; maximum
relative error5.2348e-6 (0.0005235%). Loulou CUDA smoke57.66s and
calibration161.13s, plus preserved failed Fifi smoke27.91s, used about
0.069 GPU-hours. Evidence and measured beta table are in
`results/eqprop-conv123-relative-output-20260924-v1/calibration_summary.json`;
local path-only config changes and hashes are in
`calibration_config_path_mapping.json`. Accepted configs are frozen under
`configs/conv/eqprop_conv123_relative_output_20260924/`. Training preparation
now runs every exact config's local smoke before admission.

The successful scheduler-only alternative is fmu@v100/qos_gpu-dev,
array0-8%3 at2h, projected September24 20:21:59 Paris. This is an estimate,
not a reservation or evidence of a live job. Keep Conv3 on the requested
V100 class. Riri's host clock is about two hours behind actual UTC; use
coordinator-corrected deadline and elapsed duration for admission/monitoring,
and preserve raw host times without treating them as cross-host ordering.

The final read-only audit passed all36 canonical and36 staged configs:
internal/injected beta conversion, frozen beta, sigma5e-4, ten epochs,
batches16/64, seeds, initializer hashes, gains, Adam rates and T4/6/8.
All177 staged numerical Python files are byte-identical to the calibration
runtime. Production explicitly disables official-test reads. Frozen training
archive SHA256: `19f297708ca43a699398e17791c90ddd12981beb3bfffa1af5193cae0ec58d1c`.
Transport changes only initializer paths to local assets and records them
in `transport/config-path-substitutions.tsv`.

Accepted injected betas, in target order0.8/1.2/1.6/2.0 (rounded here;
exact JSON values govern training):

| Architecture | Scheme | Injected betas |
|---|---|---|
| conv1 | baseline | 3.2155538, 4.8233307, 6.4311077, 8.0388846 |
| conv1 | ours | 2.5844808, 3.8767212, 5.1689616, 6.461202 |
| conv1 | legacy | 0.73754292, 1.1063144, 1.4750858, 1.8438573 |
| conv2 | baseline | 0.61803412, 0.92705119, 1.2360682, 1.5450853 |
| conv2 | ours | 0.21713065, 0.32569598, 0.4342613, 0.54282663 |
| conv2 | legacy | 0.038569993, 0.057854989, 0.077139986, 0.096424982 |
| conv3 | baseline | 0.42566121, 0.63849182, 0.85132242, 1.064153 |
| conv3 | ours | 0.045775991, 0.068663986, 0.091551981, 0.11443998 |
| conv3 | legacy | 0.0066767077, 0.010015062, 0.013353415, 0.016691769 |

QoS decision: use dev for these bounded, independent exploratory training
cases, each expected to finish within100 minutes and capped at2h. Live
scheduler eligibility and syntax checks pass with at most three GPUs and
nine submitted tasks. The indexed official [IDRIS user guide](https://www.idris.fr/media/eng/ia/guide_nouvel_utilisateur_ia-eng.pdf)
includes a batch-script dev example (page33); full current web documentation
could not be retrieved. Treat suitability as the coordinator's inference
from this evidence, not an explicit blanket production-policy statement.
No long job is split or chained to bypass the wall limit.

Reviewed first live submission command (after all36 exact smoke completions):

```bash
study=/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/eqprop-conv123-relative-output-20260924-v1/training_workspace
sbatch --parsable --chdir="$study/source-training-v1" \
  --output="$study/launch/slurm/%A_%a.out" \
  --error="$study/launch/slurm/%A_%a.err" \
  --export="ALL,EXPERIMENT_STUDY_ROOT=$study" "$study/transport/conv3.slurm"
```

The reviewed wrapper specifies array0-8%3, fmu@v100, gpu_p13, qos_gpu-dev,
v100-16g, one GPU/four CPUs,2h, and pytorch-gpu/py3/2.5.0. The later9-11%3
submission uses the same contract after the first array is terminal. Record
both IDs below at actual submission; this command alone is not a launch.
Measured local exact-smoke peak reservations are approximately166MiB Conv1,
992MiB Conv2,4538MiB Conv3; initial free-memory admission floors are
1536/4096/7000MiB respectively. Shared-host throughput is measured in
production rather than inferred from exclusive-GPU timing.

At16:36 Paris, all36 local exact CUDA smokes validated, including shared
initializer identities, T4/6/8 noise-draw counts and zero official-test
reads. The gate marker was published and first Jean Zay array **152887**
submitted for IDs0–8%3. Scheduler-only estimate was about20:30 Paris;
actual allocation is monitored separately. IDs9–11 remain unsubmitted.
Workstation admission verified Akib9627MiB free/idle, Riri28130MiB
free/Ben-only and Loulou27084MiB free/Ben-only. Their queue launches
are underway; local waits for the exp-006 worker's natural exit.

First verified launches: Akib queuePID524075, CUDAworker524113 (epoch1);
Riri queuePID2915962, CUDAworker2916016; Loulou queuePID2166929,
CUDAworker2166990. Riri/Loulou have canonical running bundles and about
1.62GiB GPU allocations. Actual Slurm state beat the estimate:152887_0,
152887_1,152887_2 are running on r6i0n1/r10i2n5/r6i0n5, with3–8 pending
only JobArrayTaskLimit. Reviewed account/QoS/resources/array throttle match.
Exact launch commands are retained in `transport/launch-commands.txt`;
initial semantic training progress is checked before monitoring handoff.

Initial throughput at approximately16:41 Paris: all three V100 cases reached
training batch1500/3438 in epoch1 with finite losses, correct initializer,
Tesla V100-SXM2-16GB and torch2.5.0/CUDA12.2. Startup-inclusive ten-epoch
training projection89–90 minutes leaves roughly30 minutes for validation,
checkpointing and margin under the2h cap. Akib reached epoch5 in about4
minutes, suggesting8–10 minutes per Conv1 case. Shared Conv2 projected
82–84 minutes training, approximately90 minutes including validation per
case; six cases may require9h per host rather than the initially reserved8h.
Keep the existing running wrappers unchanged. After Conv1 completes, record
its actual spend and reallocate unused allowance to same-host unstarted Conv2
tail cases if needed; alternatively move an entire unstarted target triplet
to newly admissible Trex/Fifi within the overall60-GPU-hour cap. This does
not authorize duplicate runs, moving a partly started triplet, or dropping
a late scheme to claim complete coverage.

At16:44 Paris the monitor found Fifi idle (no compute processes,32149MiB
free). High preparation now claims a planned transfer of the entire unstarted
Conv2 target1.6 triplet from Riri to Fifi, after a fresh owner/launcher check.
Riri must relinquish those three pending cases before Fifi admission. Preserve
its active baseline0.8 training and the remaining ours/legacy0.8 cases on
Riri; safely retire/replace only our queue controller if needed, without
editing the active wrapper or interrupting the current training process.
Fifi's expected unshared triplet duration is about2–2.5h; this redistributes
the existing Conv2 budget. Record actual controller handoff and commands
before admitting the transfer; planned movement alone is not a launch.

First full collected training: Conv1 baseline target1.2 on Akib, exit0,
465seconds (0.1292GPUh), final validation96.02%, best96.18%. Full bundle and
launch receipts are under `collected/akib/production/akib/` and
`collected/akib/launch/akib/production/` in the study root. Canonical validation
and the offline36-case analysis helper both pass; official test remains
unread. This single arm establishes no ours-versus-legacy contrast.
The analysis helper is `experiments/plot_initial_relative_output_training.py`;
its six focused tests pass and it retains all36case/12pair coverage slots.

Fifi transfer admitted after a fresh empty-GPU check: queuePID2773995,
`transport/run-fifi-target1p6.sh fifi-target1p6 production`,3h lane cap,
45-minute initial case estimate plus10-minute admission margin. Riri's
controller2915962 is deliberately stopped and permanently retired from
future dispatch; its timeout2916012, exact runner2916013 and trainer2916016
continue the current baseline0.8 case. No1.6 bundle existed on Riri before
the transfer. Remaining ours/legacy0.8 will run on Riri after that case
finishes; the controller must never receive SIGCONT. The detailed handoff
preserves the stopped-parent/zombie-exit handling and verified PIDs.

Budget at this admission: reserve Fifi3h by reducing the now-completed
preparation allowance from2h to1h and using the2h reserve. Thus V10024h +
original four workstation ceilings32h + Fifi3h + preparation1h =60GPUh.
The Riri replacement shares its original8h total ceiling, not an additional
8h. Actual Conv1 completion will release substantial unused allowance for
any later justified tail; record that release before admission. No compute
cap increase is implied by the extra host.

Fifi initial semantic progress verified: CUDAtrainer2774028, approximately
1622MiB allocated, correct target1.6 initializer/config and finite loss
through epoch1 batch2000. Startup-inclusive98.05seconds for those2000
training batches projects28.1minutes for ten training epochs, before
validation/checkpoint overhead; the45-minute admission estimate and3h
triplet cap remain adequate. Riri's active baseline continues uninterrupted.

Filip subsequently requested another run **in parallel on the 5090s**.
This explicitly permits concurrent workers from this study on each eligible
5090, beyond the existing host-level parallelism. Admit one additional
pending Conv2 case on each Fifi/Riri/Loulou after live memory/owner checks:
ours at target1.6 on Fifi, target0.8 on Riri, target1.2 on Loulou. Retain
all36 original cases and the60-GPU-hour cap, with no repeats or science
changes. Claim each pending case and retire its old future dispatch before
starting it concurrently; preserve all active training children. Keep the
remaining legacy and Loulou target2.0 cases explicitly assigned, and measure
throughput with two workers before updating remaining duration estimates.

Parallel-admission update16:54 Paris: Riri/Loulou each have about30.46GiB
free with our existing1.62GiB worker and idle Ben MPS. Add ours0.8/1.2
respectively. Fifi acquired Adrien CUDAprocess2774918 (~3.86GiB); preserve
our already admitted single-worker queue but do not add a concurrent worker
under the Ben-only sharing rule. Trex's Kellian CUDA worker has departed;
its GPU is now Ben-only with25.24GiB free. Move the entire unstarted target2
Conv2 triplet from Loulou to Trex after Loulou dispatch relinquishes it,
and start baseline/ours concurrently there, with legacy retained on Trex.
This fulfills the explicitly authorized Trex use without cross-host scheme
pairs. Split the original Loulou8h physical-host allowance into Loulou5h
and Trex3h; parallel processes share each host's remaining allowance instead
of receiving a fresh full cap per PID. Total case and compute caps remain
36 and60GPUh.

Additional concurrency verified on real GPUs around17:00 Paris:

| Host / Conv2 target | Existing or first worker | Additional worker | Launch handles |
|---|---|---|---|
| Riri /0.8 | baseline2916016 | ours2923830 | ours wrapper2923796; original controller2915962 permanently stopped |
| Loulou /1.2 | baseline2166990 | ours2173674 | ours wrapper2173640; original controller2166929 permanently stopped |
| Trex /2.0 | baseline3292126 | ours3292330 | wrappers3292059/3292260 |

Each host has two actual CUDA workers at approximately1.6GiB each. The new
Riri/Loulou ours cases passed training batch500 with finite losses; both
Trex cases passed the first numerical batch with the correct initializer.
All remaining legacy cases are retained for same-host admission, with High
owning dispatch. Fifi retains its existing single-worker target1.6 queue.
No scientific config changed and the retired original dispatchers must
never resume or launch duplicate cases.

Filip's next instruction supersedes further5090 expansion: **free two5090s
when possible**. Gracefully finish the already assigned matched triplets,
choose the two hosts with the earliest projected group completion from actual
progress, and release our workers there. Do not assign new groups or reuse
those released hosts for additional work. Preserve running training and
same-host scheme comparisons; pending members of an already assigned triplet
may finish before release. Report the selected hosts and actual release,
without claiming to remove Ben's/Adrien's unrelated work. V100, local3090
and Akib3080 allocations are unaffected.

Selected release hosts: **Riri and Loulou**, after finishing targets0.8 and
1.2 respectively. Their baselines were already at epoch8; observed parallel
Loulou epoch cadence6.55minutes (prior single-worker approximately2.8minutes)
projects baseline completion around17:18 and total group completion around
**18:15–18:35 Paris**, provisional. Admit only their remaining legacy case,
then release our workers and assign no further groups. Other users' processes
remain untouched. Local Conv1 queue2019402 launched after verified exp-006
exit and an empty compute-process list with23060MiB free.

Filip then explicitly authorized **killing the run on Fifi**. High execution
is stopping our Fifi queue and all its training descendants, preserving logs,
partial checkpoints and any already completed evidence; Adrien's unrelated
job must remain untouched. No automatic restart or replacement is authorized
by this cancellation. Reconcile target1.6 members as completed, user-cancelled
or unstarted according to actual artifacts, never as a numerical failure.
Fifi becomes the first released allocation from our study; the second is the
earliest completed Riri/Loulou triplet. Their already planned legacy members
may finish, with no additional groups. Any missing ours/legacy target1.6 pair
leaves the Conv2 four-target mean and corresponding broad hypothesis verdict
inconclusive rather than dropping the missing target.

Fifi termination verified: controller2773995 and its verified timeout2774024,
exact runner2774025 and trainer2774028 are absent; CUDA process list is
empty. Baseline1.6 is partial through epoch6 batch3000; ours/legacy1.6 never
started. Evidence is preserved and the cancellation is recorded remotely in
`launch/fifi-user-cancelled.txt`. Adrien2774918 had already exited before
our action and was never signalled. Fifi is released; no restart/replacement.

Local launch correction: initial nohup2019402 exited before creating a lane
or scientific output. Its empty attempt is retained. The recovered isolated
tmux `relative-output-local-20260924` has queue2020926, CUDAtrainer2021046
and verified finite first-batch progress with the correct initializer.

Monitoring continued at Filip's explicit request. At approximately17:15
Paris the local preview has four fully validated completions: Conv1 baseline
0.8 at95.92%, and the complete Akib target1.2 triplet at96.02/96.26/96.36%
(baseline/ours/legacy). The one complete paired difference is−0.10pp for
ours-minus-legacy; no architecture-level verdict is inferred. Preview tables
retain all36 slots, including three explicitly user-cancelled Fifi arms.
Other active/remote cases remain distinct from uncollected local evidence.
The Medium worker continues30-minute checks and collection, High owns
pending same-host legacy/Slurm admissions, and terminal analysis remains
assigned after reconciliation. No new operational failure was observed.

## Terminal collection and review — September 24, 23:15 Paris

The monitor reconciled all 36 original cases: 33 completed and three explicit
Fifi user cancellations. All 35 attempted canonical bundles validate locally:
33 successful, the partial Fifi baseline, and one preserved operationally
failed Conv3 baseline1.6 attempt. The other two cancelled Fifi arms never
started. Conv3's failed `jz-6` attempt is explicitly excluded and replaced by
same-source/config `jz-6-recovery1`; its first two epoch metric dictionaries
match exactly. The final array160519 ended at23:11:05 Paris. No numerical
failure or poor-accuracy exclusion occurred.

Scientific validation passes all 33 completed results, including exact epochs
1–10, sigma/beta, inherited rates, initialization, dataset/order identity,
float64, frozen zero biases, finite bounded weights, matched placement and no
official-test reads. The V100 recovery's recorded target alias is validated
against its own V100 wrapper. Thirteen focused aggregation tests pass.
Authoritative roots are the study's `production/local` and
`collected/{akib,riri,loulou,trex,fifi,jean-zay}/production` directories.

There are 11 complete ours-minus-legacy pairs out of 12. The fixed four-target
means are **−0.065pp for Conv1** (zero positive pairs) and **−4.460pp for Conv3**
(one positive pair). Conv2 has no four-target mean because target1.6 was
cancelled. H-006 is contradicted in the complete Conv1/3 scopes and inconclusive
for Conv2; its broad complete-grid verdict remains inconclusive under the
cancellation rule above. Every baseline endpoint and every intended contrast
is retained in the [terminal report](../../../../../results/eqprop-conv123-relative-output-20260924-v1/analysis/report.md)
and [campaign result note](../results/exp-007-relative-output-training.md).

Approximate production/recovery allocation or physical occupancy is26.662
GPU-hours; known calibration and local smokes bring observed use to roughly26.8
GPU-hours, within the60-hour cap and deadline. Slurm accounting, launch receipts,
clock offsets, failed attempts and cancellation evidence remain in the local
study. Review and collection are finished; `partial` preserves the missing
requested trainings rather than claiming36 successful runs. No automatic
restart of Fifi is authorized. Exp008 remains a separate active extension.
