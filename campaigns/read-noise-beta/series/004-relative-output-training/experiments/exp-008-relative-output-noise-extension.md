---
id: "exp-008"
title: "Extend relative output targets and additive endpoint noise"
status: "ready"
hypotheses: ["H-007"]
---
# exp-008 — Relative output displacement and noise extension

## Assignment and provenance

On September24 Filip requested displacement2.5/3 and noise1e-3/2e-3 in addition
to exp007. This authorizes preparation, immediate execution, monitoring and review,
including daytime continuation until September26 08:00 Europe/Paris. The extension
was selected after partial exp007 outcomes; reused cells are retrospective, the117
new cells are prospective. Preserve exp007 and H006's original decision rule.

## Cases and controls

Filip narrowed the extension before any production launch: Conv2 and Conv3
use only displacements strictly greater than1.6. Filip subsequently added
"6 or so", resolved to exactly6.0 for every architecture. Conv2/3 therefore use
2.0,2.5,3.0,6.0; Conv1 retains0.8,1.2,1.6,2.0,2.5,3.0,6.0. Each is crossed with baseline/ours/legacy and additive
Gaussian endpoint sigma5e-4/1e-3/2e-3.
There are135 active logical cells:63 Conv1,36 Conv2,36 Conv3. Reuse18 selected
exp007 cells (12 Conv1,3 Conv2,3 Conv3); launch117 new ten-epoch seed0 trainings
(51 Conv1,33 Conv2,33 Conv3). All36 exp007 outcomes remain in that original study,
including its three user cancellations; they are neither rerun nor erased.
The36 unstarted lower-target Conv2/3 extension configurations are removed from
active launch lists and retained outside the active config glob as provenance.

Inherit the entire [exp007 contract](exp-007-initial-relative-output-training.md):
ordinary MNIST55k/5k split, official test disabled, paired physical20-node output,
centered frozen-current EqProp squared loss, float64 perfect diodes, frozen zero
bias, weights[0,100], source Adam parameterwise LRs, batch16/validation64,
input gains40/100/360, T=K4/6/8. Baseline(V,I)=(1,1), ours=(4,1), legacy=(4,.25).
Noise remains independent absolute additive Gaussian on signed endpoint reads,
not relative noise or noisy relaxation. Preserve seed2026081601 and draw policy.

## Calibration and execution contract

D=pooled RMS((v_plus-v_minus)/2); F=pooled RMS(post-T free physical output).
Calibrate27 architecture/scheme/new-target combinations (2.5,3.0,6.0) on the same frozen
576 validation examples, within1% relative tolerance, at most three corrections.
Retain completed2.5/3 calibration; add only nine clean target6 matches, in a
separate recorded calibration output if the earlier calibration is already running.
Reuse measured old-target beta at all noise levels. Calibration remains clean;
freeze accepted injected beta and internal B/(V/I)^depth for all ten epochs.
No T/K tests, LR search or official-test reads.

Configs: `configs/conv/eqprop_conv123_relative_output_noise_extension_20260924/`.
Calibration config: `configs/conv/eqprop_conv123_relative_output_noise_extension_calibration_20260924.json`.
Use existing `experiments.calibrate_initial_relative_output` and `experiments.exact_run`.
Require exact local one-training/one-validation-batch smoke for each new config.
Preserve the exp007 frozen numerical runtime and record actual hashes/commands.
No source/config identity is claimed until verified.

## Metrics and decision

[H007](../../../hypotheses/H-007-relative-output-noise-extension.md) governs this
extension: report all45 paired epoch10 validation accuracy differences ours−legacy,
plus baseline accuracy and loss. At each architecture/noise, the equal-weight
mean covers seven targets for Conv1 and four targets for Conv2/3, and is defined
only when every declared pair is valid and complete.
Positive/negative mean supports/contradicts the scoped average advantage; exact
tie or missing pair is inconclusive. Report sign consistency and all failures.
Do not select each scheme's best epoch or target. Single-seed descriptive evidence
is not significance. The original cancelled target1.6 is outside the narrowed
Conv2 extension scope; its original exp007 coverage remains incomplete.
Matching initial D/F does not equalize absolute displacement or SNR. Fixed beta
allows D/F to drift. Cross-noise host changes must be disclosed as confounds.

## Budget, placement, stop and outputs

Result root: `results/eqprop-conv123-relative-output-noise-extension-20260924-v1/`,
indexed before creation. Revised cap100 GPU-hours includes calibration0.5h, exact
smokes, production and operational recoveries; it is separate from exp007's60h.
Estimated70–86 GPU-hours before contingency, extrapolating prior measured rates:
Conv1~7–8h on local RTX3090/Akib RTX3080; Conv2~19–27h on eligible Trex/Riri
RTX5090s; Conv3~44–51h across three V100s (~15–17 wall hours after original jobs).
Conservative initial allocations:66h V100,10h Conv1,22h Conv2,2h preparation/recovery.
Transfer verified unused allocation between depths within100h if sharing requires it.
The original126-case/120h and interim90-case/80h proposals are superseded before
production launches; no added target6 outcome was observed before this amendment.
Aim to finish September25 evening; hard deadline September26 08:00 Paris.
Sharing/queue delays may reduce completed coverage. Fifi and Loulou stay released.
Check live admission and owners before allocating. Keep each matched scheme triplet
on one workstation/runtime or the declared JZ V100 pool. Keep noise comparisons
on the same host where feasible; record any deviations. Never run training on CPU.

Stop for nonfinite state, invalid config/source/cohort, exhausted budget/deadline;
preserve poor scientific outcomes. Operational retries need diagnosis, preserved
attempts and unchanged science. Do not restart cancelled arms. Do not disturb
unrelated jobs. At most three concurrent V100s including original study/recoveries.
JZ requires planned row before submission and local smoke; restored canary applies
to the diagnosed node-specific original failure before using its repaired contract.

Collect remote bundles locally and validate before interpreting. Extend analysis
for the117 new cells and explicit18-cell reuse without inferring completion from
files alone. Result note: `../results/exp-008-relative-output-noise-extension.md`.

## Execution and monitoring handoff

### Measured calibration and frozen config preparation

The local RTX3090 was admitted after the exp007 recovery smoke released it:
23,025 MiB free, no compute processes, accepted Python3.12/torch2.5.1+cu121.
The existing calibration runner was used unchanged. Targets2.5/3.0 produced
18 accepted matches from27 measured points in91.10 seconds; target6.0 was added
in a separate child bundle and produced9 accepted matches from18 measured points
in64.40 seconds. All27 targets passed without correction rounds. Maximum relative
target error was5.2594e-6 (0.000526%). Two same-path CUDA smokes took18.80 and
18.78 seconds and passed all nine endpoint-parity contexts each. Total measured
calibration/smoke time was193.08 seconds, within the0.5-GPU-hour cap. These are
execution/parity checks at accepted T4/6/8, not solver-setting experiments.

The576-example cohort hash is
`95af3eebd9416ec643b7d698d061b7dac3fe0d52311dd888293609bada629acd`.
Both calibration bundles validate locally; parameters were unchanged, all bias
tensors remained exactly zero, and no optimizer step or official-test read occurred.
Commands used `python -m experiments.calibrate_initial_relative_output --config
CONFIG --device cuda --target local:RTX3090`, with and without `--smoke`.
CONFIG is the main calibration JSON above for2.5/3.0, and
`configs/conv/eqprop_conv123_relative_output_noise_extension_target6_calibration_20260924.json`
for6.0. Bundles are `calibration/{smoke_all,all}` and
`target6/calibration/{smoke_all,all}` below the study root; measurements and beta
values are summarized in `preparation/calibration_summary.json`.

The active config directory contains exactly117 cases:51 Conv1,33 Conv2,33 Conv3.
The36 excluded Conv2/3 low-target drafts were never smoked or launched; their
paths are retained in `preparation/excluded-draft-configs.txt`, outside the active
config directory. Higher-noise configs preserve their measured clean beta and
record the source config/hash. All117 configs pass exact-config loading and
preserve source initialization, seeds, Adam rates, gain, batch16/validation64,
ten epochs, T4/6/8 and disabled official test.

`source-training-v1.tar.gz` SHA256 is
`5a1e95b8e9e5cf022209f4ea0f8e76c053bdc10ba39ecb8360a5bc736d0cda37`.
All177 numerical Python files under labs/model/training and exact_run/reporting
match the accepted exp007 snapshot byte-for-byte. The snapshot includes
`TRAINING_SHA256SUMS`, `SOURCE_COMMIT` and initializer assets; its117 configs
differ from canonical configs only by initializer path, recorded in
`preparation/config-path-substitutions.tsv`. The earlier90-config draft archive
is preserved as `preparation/source-training-before-target6-90configs.tar.gz` and
is not an execution source. All117 exact CUDA smokes completed successfully and
passed canonical and semantic validation against this frozen snapshot before
the production handoff. Each performed one real training batch and one validation
batch, used the configured sigma and internal beta, shared its architecture's
initializer, retained source rates and float64, and kept biases exactly zero and
weights finite within[0,100]. Expected endpoint-noise draw counts were4/6/8;
official-test reads were zero. Evidence: `local-smokes/validation.json`,
`preparation/validate_exact_smokes.py`, `preparation/exact-smokes-command.txt`
and `transport/local-smokes-passed.txt`.

Smoke elapsed span was784.81 seconds (summed bundle durations721.72 seconds).
Maximum reserved CUDA memory was166/992/4538 MiB for Conv1/2/3. Calibration plus
calibration smokes and the exact-smoke span total977.89 measured seconds,
approximately0.272 GPU-hours. All source-file/archive hashes were rechecked after
smokes. The local GPU was released with no compute processes and22,849 MiB free;
the Codifier received the validated source, markers and production handoff.
No full training was launched by the preparation worker.

The extension's offline helper is
`experiments/plot_relative_output_noise_extension.py`. It accepts both study roots
and repeated authoritative production roots, preserves the original helper's
exp007 behavior, validates calibration reuse and each source/archive identity,
and emits135 active rows,45 fixed contrasts and9 complete-only architecture/noise
means. The18 inherited out-of-scope cells, including Fifi cancellations, remain
in a separate coverage table. Twelve focused aggregation tests passed; source
calibration reuse was checked for all153 retained configs. A temporary empty-input
CLI/plot check emitted all declared coverage without inventing results.

Preparation: relative_output_prepare; placement/launch/recovery: relative_output_capacity;
routine monitoring: clean_beta_monitor after executable handoff and acknowledgment.
Scope-change check: both Codifiers confirm no extension production launched or
queued and no exact smoke started. Calibration is running for the unchanged18 new
2.5/3 targets; nine target6 matches are added without repeating those measurements.
The36 low-target Conv2/3 configs are removed from the active glob. Capacity confirms
no production lists frozen or queues submitted when target6 was assigned.
Local calibration follows the original recovery smoke and fresh admission, with
0.5GPUh cap. Root coordinates records and final review.
Before launch append exact commands, source hashes, handles, host/result paths,
inspect/collect/validate commands, next check and remaining budget here. Monitor
checks real artifact progress every30min and escalates operational failures to
Codifier. Training processes alone are not an active monitoring handoff.

### Prepared placement and source (before production)

Frozen archive: `results/eqprop-conv123-relative-output-noise-extension-20260924-v1/source-training-v1.tar.gz`,
SHA256 `5a1e95b8e9e5cf022209f4ea0f8e76c053bdc10ba39ecb8360a5bc736d0cda37`.
All117 staged configs preserve their scientific values; only initializer paths
are rebased to byte-identical `assets/convN_seed0.pt`. The final source includes
`TRAINING_SHA256SUMS` and `SOURCE_COMMIT`; source hashes passed on all remote
workspaces. Exact local smoke validation remains a separate required gate.

- Conv1 local: targets0.8/1.6/2.5/6,30 new cases,5h host ceiling.
- Conv1 Akib: targets1.2/2/3,21 new cases,5h host ceiling.
- Conv2 Riri: targets2.5/6,18 new cases,11h host ceiling.
- Conv2 Trex: targets2/3,15 new cases,11h host ceiling.
- Conv3:33 cases on the same V100 pool, arrays0–8,9–17,18–26,27–32,
  maximum3 concurrently,2h per case,66h ceiling. Exclude the node r10i2n8
  diagnosed with severe paging during exp007; its repaired contract passed
  restored live canary159207. Original study/recovery retains pool priority.

All schemes and noise levels for each target stay on the same workstation.
Where exp007 controls exist, the same local/Akib/Trex placement is retained.
Total ceilings are100GPUh including2h preparation/recovery. No source, batch,
epoch, precision, learning-rate or beta change is made for placement.

Direct wrappers and readable case lists are under the result root's `transport/`:
`run-queue.sh`, `conv3.slurm`, and `configs-{local,akib,riri,trex,conv3}.txt`.
They cover117 unique cases (30/21/18/15/33 respectively), use exact_run with
explicit `--index 0`, and guard every admission with measured memory/owner
checks and an expected-runtime-plus10min margin. Workstation expected times
start at10min Conv1 and40min Conv2 and increase from actual completed-case
elapsed time. Numerical failures are retained; unrelated operational failures
stop a lane for Codifier diagnosis. No extension production was launched at
this preparation checkpoint. Actual commands and handles follow after gates.

### Workstation launch and monitoring handoff — September24 20:32 Paris

All117 exact CUDA smokes passed semantic validation: paired initializers,
float64, configured endpoint-noise draws, frozen zero biases, finite bounded
weights/losses, and zero official-test reads. Evidence is
`local-smokes/validation.json`; the archive hash above is unchanged. Final
transport hashes are in `transport/TRANSPORT_SHA256SUMS`; exact executed shell
commands are in `transport/workstation-launch-commands.txt`.

| Host | Queue/session | Initial CUDA PID | Assigned cases | Host cap |
|---|---|---:|---:|---:|
| local | tmux `relative-output-extension-local-20260924`, queue2125707 | 2125807 | 30 | 5h |
| Akib | nohup queue530768 | 530812 | 21 | 5h |
| Riri | nohup queue3033465 | 3033529 | 18 | 11h |
| Trex | nohup queue3418872 | 3418946 | 15 | 11h |

All four initial cases have real CUDA allocations and finite batches. Conv1
workers reached3000 batches at the first check. RTX5090 admissions found only
Ben's workloads, with27.1/25.8GiB free; our Conv2 workers use about1.6GiB each.
Record these as shared-GPU timings. Riri's clock was7225s behind UTC; its launch
deadline was adjusted without changing the host clock. Use elapsed/corrected
wall time for estimates. No further work was admitted on Fifi or Loulou.

Local root is this study's result directory. Remote roots are:
`/home/filiposana/server_code/results/eqprop-conv123-relative-output-noise-extension-20260924-v1/training_workspace`
on Akib and
`/home/filip/server_code/results/eqprop-conv123-relative-output-noise-extension-20260924-v1/training_workspace`
on Riri/Trex. Each has `source-training-v1`, `transport`, `launch`, and
`production`. Logs are `launch/<lane>/production/<arm>.log`; queue receipts
include `pid`, `started_at`, per-case elapsed/exit files and final `exit_code`.
Canonical runs are `production/<lane>/<arm>/000_<arm>_<hash>/`.

Monitoring owner: clean_beta_monitor, explicitly acknowledged the handles, caps and deadline, with
next check around20:40 for initial shared throughput, then every30min. Monitor
reads status/metrics and bounded logs, collects terminal cases locally, validates,
and escalates failures or budget risks to relative_output_capacity. Codifier
owns all remaining Slurm admissions, source/queue mutations and recovery.
At handoff, all33 extension V100 cases remain unsubmitted; original exp007 and
its recovery retain the three-GPU pool until complete.

Executable inspection/collection pattern (substitute the recorded host/lane
and a terminal run path; do not validate or hash active checkpoints):

```bash
ssh -F /home/filip/.ssh/config -o BatchMode=yes HOST 'tail -n 6 REMOTE_ROOT/launch/LANE/production/ARM.log; cat REMOTE_ROOT/launch/LANE/production/ARM-exit_code'
rsync -a -e 'ssh -F /home/filip/.ssh/config -o BatchMode=yes' HOST:REMOTE_ROOT/production/LANE/ARM/ LOCAL_STUDY/collected/HOST/production/LANE/ARM/
/home/filip/miniconda3/envs/py312/bin/python -m experiments.reporting validate-run LOCAL_COMPLETED_RUN
```

Local status inspection uses the same bundle paths under `production/local`.
Per-case admission logs retain available memory, owner-visible GPU PIDs,
remaining time and expected-case margin. All initially assigned workers may
continue during daytime under the explicit plan; global deadline is
September26 08:00Paris. Added ceiling100GPUh remains independent of exp007.
Do not restart excluded lower-target Conv2/3 cases or cancelled exp007 cases.

### First V100 batch — September24 23:12 Paris

Original exp007 final array160519 is fully terminal, all three tasks completed
with exit0. The extension first array is **162366**, tasks0–8%3, nine cases:
target2 at sigma1e-3/2e-3 and target2.5 at sigma5e-4, each with all three schemes.
It uses the already recorded fmu@v100/dev contract,1GPU/4CPU,2h per case,
excluding r10i2n8. Remaining tasks9–32 are still unsubmitted.

All117 prior exact local smokes passed; a fresh immediate first-case smoke also
passed canonical and semantic checks (CUDA,float64,shared initializer,eight
endpoint-noise draws,finite losses,zero official-test reads). Exact commands
and evidence are `transport/first-v100-local-smoke-command.sh`,
`transport/first-v100-local-smoke-passed.txt`, and
`transport/first-v100-submission-command.txt`. Frozen numerical/config files
and transport hashes were verified remotely before submission; archive remains
5a1e95b8e9e5cf022209f4ea0f8e76c053bdc10ba39ecb8360a5bc736d0cda37.

Remote root is
`/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/eqprop-conv123-relative-output-noise-extension-20260924-v1/training_workspace`.
Slurm logs: `launch/slurm/162366_<task>.out` and `.err`; direct lane receipts:
`launch/jz-<task>/production/`; canonical cases: `production/jz-<task>/<arm>/`.
Admission ownership remains relative_output_capacity; routine monitoring and
terminal collection transfer to clean_beta_monitor after initial allocation
and finite progress are verified. Added100GPUh ceiling and September26
08:00Paris deadline remain unchanged.

At23:15, tasks0/1/2 are running on r10i4n3/r3i7n4/r3i7n5; all reached500
finite training batches. Admission logs identify TeslaV100-SXM2-16GB with
16,145MiB initially free. Canonical manifests preserve the frozen archive and
Python3.12.7/pytorch-gpu2.5.0 runtime. Tasks3–8 are pending on the array
throttle. Medium monitoring was requested with an early epoch-pace check
then30min cadence; later batches remain Codifier-owned and unsubmitted.

Medium explicitly accepted162366 monitoring/collection at23:16, first epoch
pace check23:25 and then30min; terminal notification triggers Codifier review
for the next batch. Akib completed21/21 extension cases with zero validation
errors at23:13:39, using9718s queue wall time; approximately2.30h of its5h
ceiling is unused and remains available only after a recorded redistribution.

### Second V100 batch — September25 03:58 Paris

First array162366 completed all9 tasks with exit0; the last, legacy target2.5
at sigma5e-4, took1h47m and remained within its2h limit. Second array
**166933**, tasks9–17%3, was submitted after verifying the pool was empty.
This batch covers target2.5 at sigma1e-3/2e-3 and target3 at sigma5e-4,
all three schemes. Tasks18–32 remain unsubmitted.

A fresh synchronous first-case local CUDA exact smoke passed canonical and
semantic validation; numerical/config and transport hashes passed remotely.
Evidence and exact commands are `transport/second-v100-local-smoke-command.sh`,
`transport/second-v100-local-smoke-passed.txt`, and
`transport/second-v100-submission-command.txt`. Source archive,1V100/4CPU/2h
per case,dev QoS,bad-node exclusion,total100GPUh ceiling and deadline are
unchanged. Logs follow `launch/slurm/166933_<task>.{out,err}`; case receipts
and canonical outputs use `jz-9` through `jz-17` under the previously recorded
remote root. Codifier verifies allocation and initial progress before Medium
assumes regular monitoring; later admissions remain Codifier-owned.

At04:01, tasks9/10/11 reached500 finite batches on V100-SXM2-16GB nodes
r10i4n3/r10i5n8/r3i4n4. Manifests retain the expected source archive and
module/runtime. Regular monitoring was handed to Medium with early epoch
pace check around04:10, then30min; next admission waits the whole array
terminal, estimated08:25–08:45 if pace holds. Explicit daytime continuation
permits work past08:00 today; the hard deadline remains September26 08:00.

### Third V100 batch — September25 08:26 Paris

Second array166933 completed all9 tasks with exit0, releasing the full pool.
Third array **169756**, tasks18–26%3, was then submitted: target3 at
sigma1e-3/2e-3 and target6 at sigma5e-4, all three schemes. The final6 cases
(tasks27–32) remain unsubmitted.

The fresh local CUDA exact smoke passed canonical and semantic checks;
remote numerical/config and transport hashes passed. Exact commands and
evidence: `transport/third-v100-local-smoke-command.sh`,
`transport/third-v100-local-smoke-passed.txt`, and
`transport/third-v100-submission-command.txt`. Source archive and all
scientific values are unchanged;1V100/4CPU/2h per task,dev QoS and bad-node
exclusion remain active. Logs use `launch/slurm/169756_<task>.{out,err}` and
case lanes `jz-18` through `jz-26`. Codifier retains admissions/recovery;
Medium receives routine monitoring after initial finite progress verification.
The100GPUh cap and September26 08:00Paris deadline are unchanged.

At08:29, tasks18/19/20 reached500–1000 finite batches on V100-SXM2-16GB
nodes r3i7n7/r6i1n3/r10i4n3, with the frozen archive and runtime verified.
Regular monitoring was handed to Medium with first epoch-fit check08:38,
then30min. The whole batch is expected terminal around12:50–13:00 if
measured pace holds; final27–32 admission remains Codifier-owned.

### User scope change: stop further training — September25 11:15 Paris

The user redirected the work to planning a cosine/noise question and stopped
further training admissions. No new experiment is authorized by that planning
request. Preserve the running cases and all existing evidence.

At the live check,169756_21/22/23 were running their final epoch;24–26 were
still pending on the array throttle. The exact command
`scancel --state=PENDING 169756_24 169756_25 169756_26` cancelled only the
three unstarted tasks. A subsequent scheduler/accounting check confirmed
24–26 cancelled with zero elapsed while21–23 remained running, untouched.
Tasks27–32 were never submitted and are now held with no automatic admission.

The nine unstarted exclusions are all Conv3 target6 cases: baseline/ours/legacy
at sigma5e-4 (cancelled tasks24–26),1e-3 (held27–29), and2e-3 (held30–32).
They are user-directed scope exclusions, not numerical failures. No canonical
training bundles are fabricated for these unstarted cases; frozen configs,
source archive and Slurm cancellation receipts remain preserved.

At this handoff,105 new cases were complete and locally validated, three
remained active (Conv3 target3/sigma2e-3), and nine were unstarted exclusions.
If the active cases finish successfully, terminal coverage will be108/117
originally planned new cases, plus18 reused controls. All84 workstation cases
are already complete and their workers released. Medium explicitly accepted
final21–23 monitoring, collection, validation and coverage reconciliation.
Codifier admissions are stopped; no further training or cosine run is queued.
The original100GPUh ceiling and September26 deadline remain historical
bounds, not authority to restart excluded work.
