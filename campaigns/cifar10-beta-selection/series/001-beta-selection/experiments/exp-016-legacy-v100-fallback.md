---
id: "exp-016"
title: "Parallel legacy beta fallback on V100 with matched BPTT"
status: "running"
hypotheses: ["H-010"]
---
# Legacy fallback on available V100s

**September30 09:00 Paris: Filip stopped new runs.** Keep collecting the three
running EqProp epoch5 jobs; do not extend any of them. The same Slurm owner
is now collection-only and writes `user_paused` instead of advancing. All
conditional permissions below are suspended until explicit user resumption.
Record: root `analysis/user-stop-new-runs.json`; no running job was cancelled.

Current stage (September30 09:11 Paris): all nine epoch1 cases validated.
The [review](../results/exp-016-legacy-v100-fallback.md) selects0.1x and0.0001x.
Exact epoch5 continuations submitted: native340008,0.1x340009,0.0001x340011,
14,400seconds each. Native5 is now collected at62.38%, CE1.055644;
0.1x epoch5 is collected at57.00%, CE1.218064 and fails both checks;
0.0001x remains running. The additional0.3x screen failed;
1x passed and its exact epoch5 continuation is running under the
allocation amendment below. The existing Slurm owner
retains collection and gating for the original path.

The retained3090 fallback0.0001x completed at04:51 Paris (28.32%, CE1.929404,
2,782.7seconds) and failed its matched gate. All six V100 candidates have now
produced real artifacts; their old queued3090 counterparts are cancelled.
The V100 cohort remains separately matched and active. See exp-013's result
for the retained3090 evidence; its inactive root is removed from live summaries.

Continue the authorized [exp-013](exp-013-all-scheme-long-eqprop.md) search.
The collected legacy3090 initial candidates failed the epoch1 gate: native
36.52%, CE1.754233; 0.3x29.96%, CE1.863588; 1x32.98%, CE1.863064.
Move the remaining six fallback multipliers (.0001,.001,.01,.03,.1,3) onto
available Jean Zay32GB V100s and run one new matched native BPTT control.
These seven cases start fresh in the same runtime; do not pool the3090 control
with V100 evidence or migrate partially trained tensors between runtimes.

Preserve exp-013's initializer, seed0,45k/5k split, ordering/augmentation,
batch32/eval16, original legacy amplification and learning rates, trainable
BN/gains/head, T/K[6,6,4],50-epoch schedule and no read noise. The only changes
are execution environment and starting directly with the already-required
fallback branch. The maintained decision policy accepts `start_branch: fallback`
and `final_stage: 50`; default behavior for existing studies remains unchanged.

At1/5/10/20/30, retain the existing accuracy-within2pp and CE<=1.05x matched
native gates. Promote at most two passing fallback cases, ranked by CE then
accuracy. Continue exact full-state checkpoints to5/10/20/30/50. Exhaustion
stops for review; epoch50 remains a measurement, not automatic success.

## Execution and monitoring handoff

Root: `results/cifar-eqprop-legacy-v100-20260930-v1/`.
Target: Jean Zay `fmu@v100`, `gpu_p13`, `qos_gpu-t3`, explicit `v100-32g`,
one GPU per case. Module `pytorch-gpu/py3/2.5.0` (torch2.5.0/CUDA12.2/cuDNN8907),
with deterministic cuBLAS workspace export. Freeze source, inputs and exact
Slurm scripts; use static launch checks. Filip's explicit no-smokes instruction
overrides the skill's live-canary requirement; no smoke or solver audit.

Initial allowance:5,400seconds per case, expected roughly40–90minutes each,
subject to measured V100 throughput. Preserve the started3090 fallback attempt to completion as additional runtime-specific evidence; its trained state had no saved recoverable checkpoint at the transfer decision. The fresh V100 .0001 case retains its full5,400seconds, funded explicitly below. Its five unstarted3090 siblings remain held while their V100 replacements are pending; never release a lab copy until its pending V100 job is cancelled and terminal. Preserve all3090 results and partial artifacts.

Fund the new native control and fresh V100 .0001 screen with3GPU-hours reclaimed by reducing legacy's unused epoch50 stage caps16h→15h across at most three promoted branches. The old3090 .0001 allocation remains charged in full. Other
stage caps remain4h to5,4h to10,8h to20,8h to30. The combined legacy141.5GPU-hour
ceiling includes previous attempts and this cohort; it is not reset. Deadline
October6 08:00 Paris. Only eligible stages execute.

The existing Slurm monitoring adapter must own these jobs, collection and
bounded continuation; the lab queue relinquishes only the migrated legacy
fallback cases. Preserve one operational owner per job. Report every30minutes,
verify actual artifact progress, and record accepted IDs, monitor handle and
executable inspect/collect commands here after launch. The lab queue continues
admitting the remaining authorized cases onto available5090s.


### Accepted Slurm transport (September30, 04:28 Paris)

Seven one-GPU jobs were accepted on `fmu@v100`, `gpu_p13`, `qos_gpu-t3`,
`v100-32g`, each with a90-minute allocation including launch overhead.
The native job entered running/prolog on `r8i4n0` at04:25:45 Paris. Initial
`sbatch --test-only` estimated October5, but actual backfill started within two
minutes; that estimate was not reliable availability evidence. A pending-only
QoS adjustment stopped before any scheduler mutation when native was already
running. All original job IDs and original t3 directives were retained; archived
initial scripts are under `launch/submitted_t3/`. Development QoS was not applied.

| Case | Slurm ID | Allocation cap |
| --- | --- | --- |
| native_bptt |337743|5400s|
| beta_0p0001 |337744|5400s|
| beta_0p001 |337762|5400s|
| beta_0p01 |337763|5400s|
| beta_0p03 |337764|5400s|
| beta_0p1 |337765|5400s|
| beta_3 |337785|5400s|

Remote results: `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/cifar-eqprop-legacy-v100-20260930-v1`.
Frozen source: `/lustre/fswork/projects/rech/fmu/ucy17uy/server_code/cifar-eqprop-legacy-v100-source-20260930-v1`.
The404 frozen source hashes, input identity, and all five CIFAR training-batch
hashes passed remote static verification. Scientific source files and initializer,
config, cases, split, and cohort are byte-identical to legacy3090. Only the tested
decision/collection helpers and declared study runtime/start branch differ.
Runtime CPU inspection confirmed torch2.5.0/CUDA12.2/cuDNN8907. All42 prepared
stage scripts passed `bash -n`; focused transport and summary tests passed10/10.
No smoke, live canary, training probe, or solver audit was run.

The sole operational owner is user systemd `cifar-legacy-v100-owner.timer`,
which invokes `cifar-legacy-v100-owner.service` every300seconds. Its frozen
`monitor/transport.py` combines the existing read-only Slurm account/state probe
with collection, the unchanged full-state validator, and the declared decision
helper. It stores accepted IDs in `monitor/jobs.json`, progress in
`monitor/status.json`, and success/errors in `monitor/owner-status.json`.
Before each submission it checks the absolute deadline and transport identity;
actual start checks runtime identity and remaining time. A persisted submission
intent prevents duplicate sbatch calls after uncertain acknowledgements. Terminal
exit without a result or declared numerical rejection blocks promotion. Validated
milestones alone request the next stage, reusing the exact parent checkpoint.
There is no automatic retry, scientific retuning, or second Slurm monitor.

The existing `cifar-eqprop-summary.timer` includes this root and reads the Slurm
owner's heartbeat/errors and job artifacts every30minutes. An owner error or
heartbeat older than15minutes raises attention; the summary performs no allocation.
Inspect with `systemctl --user status cifar-legacy-v100-owner.timer` and
`cat results/cifar-eqprop-legacy-v100-20260930-v1/monitor/status.json`.
A manual one-shot collection/advance uses
`/home/filip/miniconda3/envs/py312/bin/python results/cifar-eqprop-legacy-v100-20260930-v1/monitor/transport.py poll`;
the same owner lock prevents concurrent mutation.

Budget transfer details are in `analysis/budget-transfer.json`; eighteen existing
unsubmitted legacy final-stage launch specs now cap at54,000seconds (15hours).
The new cohort's stage5/10 caps remain14,400seconds and20/30 remain28,800seconds.
Elapsed GPU allocation time, including failed/old attempts, remains inside the
combined141.5GPU-hour legacy ceiling; this is an allocation transfer, not a reset.


### Verified startup and bounded infrastructure recovery (04:34 Paris)

The native allocation337743 never reached user code on `r8i4n0`: Slurm prolog
failed and drained the node. `sacct -D` preserves184seconds between04:25:45 and
04:28:49. No user-code log, allocation record, manifest or checkpoint existed.
Slurm automatically requeued it held; this is an infrastructure failure, not a
scientific rejection. A single explicit release preserved the same job ID and
reduced its hard allocation cap to5,160seconds (86minutes):184+5160=5344seconds,
leaving56seconds unused within its original5400second allowance. The unchanged
frozen trainer still has its original internal upper bound, so the reduced Slurm
hard cap can precede its internal checkpoint reserve. The runner handles SIGTERM
with a paused checkpoint, subject to scheduler termination grace. No claim of a
successful resumed training run is made until artifacts establish it.
`analysis/native-prolog-recovery.json` and `monitor/jobs.json` retain this charge.

The `.0001/.001/.01/.03` V100 cases have real85,690,312-byte initial checkpoints,
matching torch2.5.0/CUDA12.2 manifests, correct Slurm IDs and advancing initial
validation/beta observations on `r7i3n3`, `r8i3n8`, `r7i1n4`, and `r9i2n7`.
Lightweight evidence is collected in `analysis/startup-proof.json`. The old held
3090 `.001/.01/.03` jobs were cancelled only after these replacement artifacts
existed. Old3090 `.1` and `3` remain held while their corresponding V100 jobs
are pending; never release both placements. The old3090 `.0001` continues to
completion as explicitly funded additional runtime-specific evidence.

The monitor recognizes launch-failed/held pending jobs as operational errors;
ordinary `Priority`/`Resources` waits remain healthy. This fix uses the separate
frozen `monitor/transport.py`, preserving `launch/transport.py` and the shared
launch hash manifest used by already-running jobs. The existing30-minute summary
also surfaces held launch failures, owner exceptions and stale heartbeats.


At04:35 Paris, V100 `.0001/.001` had reached20 optimizer updates with growing
metrics; `.01/.03` were progressing through initial beta observations. V100
`.1` (337765) also had its initial checkpoint and validation on `r6i5n5`, so its
old unstarted held3090 counterpart was cancelled. Only old3090 `beta_3` remains
held: V100337785 is pending with an estimated05:10 start. Native337743 is pending
with an estimated05:08 start after its bounded recovery. These estimates are
provisional. The owner service is healthy and the300second timer remains active.


At04:40 Paris, V100 `beta_3`337785 had its real initial checkpoint and advancing
GPU observations on `r7i2n8`. Its last unstarted held3090 predecessor was cancelled.
All five queued legacy3090 fallback replacements are now cancelled; the earlier
running3090 `.0001` attempt remains preserved as separate evidence. V100 and lab
monitoring ownership remain unchanged.


### Read-only monitor repair (05:49 Paris)

The owner snapshot stopped advancing after05:34:40 Paris because a completed
epoch1 job had been purged from Slurm's live queue. `squeue -j337744` returned
`Invalid job id specified`, although `sacct` still reported COMPLETED/0:0. This
was a probe failure; the stage5 training jobs continued without interruption.
The owner exposed a blocked state instead of silently claiming healthy progress.

The maintained helper and separate `monitor/transport.py` now obtain one
current-user live-queue snapshot (`squeue --me`) and combine it with accounting
for each declared job. Missing live entries can therefore use retained accounting;
missing both records or a scheduler failure still raises an error. Captured
stderr now reaches the owner status and summary. No frozen launch/source/inputs,
job IDs, training processes, resource caps or scientific policy changed.
The immutable launch hash manifest was verified unchanged. Thirteen focused
transport/summary CPU tests passed, including completed-job purging, unknown
records and failed scheduler queries.

The existing owner service completed successfully at05:49:04 Paris with an empty
error list; its five-minute timer remains active. The snapshot gap was about
14minutes23seconds. At recovery native340008 was training at update1900,
`.1`340009 at1760, and `.0001`340011 at1660, each with growing metrics and valid
stage5 identities. The existing30-minute summary was refreshed with the healthy
owner and all three handles. Exact before/after status and gap evidence are in
`analysis/monitor-repair-20260930.json`. No training job was restarted or cancelled.

### Matched V100 initial-candidate extension, September30

Complete the two original fixed candidates0.3x and1x on the same V100 runtime,
frozen source and inputs, using the existing native epoch1 result33.22%,
CE1.797997. Each screen retains1407 updates and5,000 validation examples;
the unchanged gate is accuracy>=31.22% and CE<=1.8878969208. Each receives
5,400seconds including allocation overhead, with approximately57–60minutes
expected after allocation, account`fmu@v100`, partition`gpu_p13`,
QOS`qos_gpu-t3`, constraint`v100-32g`, deadline October6 08Paris. No new
native, scientific change, GPU test or canary is authorized.

Close the unused allocations of the six completed V100 fallback epoch1 jobs:
32,400seconds allocated minus20,639 authoritative single-GPU Slurm allocation
seconds leaves11,761seconds. Transfer10,800seconds to these two screens;
961seconds remain unassigned. Native/prolog accounting, active epoch5 budgets
and final50 caps are untouched; the combined legacy141.5GPU-hour ceiling stays
unchanged. The detailed accounting and collection evidence is
`analysis/matched-initial-extension-funding.json` in the existing exp016 root.

Use the existing Slurm owner, registry and collection paths in this same root.
Keep original immutable launch files and numerical source unchanged; freeze a
versioned transport admission artifact and two additional sbatch scripts only.
These two cases are admitted for epoch1 only. Their results require manual
review after the current epoch5 results; `start_branch: fallback` ignores them
for automatic promotion. At most two EqProp branches may progress beyond
five epochs cumulatively. The persistent timer continues polling all registered
jobs even if the fallback decision is terminal, so these screens remain owned
and collected independently of that decision.

Accepted at07:36Paris:0.3x Slurm`342304`,1x Slurm`342307`, each90minutes.
Submission used the existing idempotent `submit('1',case)` path under its
`monitor/owner.lock`, adding both IDs to the same`monitor/jobs.json`. The
versioned `launch/transport-e1-extension.py` admits these cases only at epoch1;
original45 frozen launch files and404 scientific source files remain exact.
Both new scripts verify`launch/extension-e1.sha256` before running. All static
hash/shell/source/input checks and Slurm test-only checks passed;12 focused CPU
tests cover admission, forbidden promotion, idempotence and collection despite
a terminal fallback decision. No GPU canary ran, following the explicit
no-smokes instruction. Exact submission and static receipts are
`analysis/matched-initial-extension-submission.json` and
`analysis/matched-initial-extension-static.json`. The live scheduler snapshot
is`analysis/matched-initial-extension-slurm.txt`.

Both allocations started immediately on`r6i5n1` under the declared account,
partition and QOS; the test-only October5 estimate was not an actual wait.
The existing owner reports healthy and tracks both IDs. Initial manifests and
validation status confirm the expected torch2.5.0/CUDA12.2 runtime, PIDs2967395
(0.3x) and2967364(1x), with all extension hashes accepted. Existing three epoch5
jobs remain running unchanged. The five-minute owner timer remains active and
the existing30-minute summary includes these jobs through the same root.

At07:40:33Paris both new screens reached20 optimizer updates, with finite
losses2.76088715 (0.3x) and2.85801482 (1x) and matching training-order hashes.
Each wrote an85,690,312-byte initial checkpoint; runtime allocation evidence
confirms cuDNN8907. `analysis/matched-initial-extension-startup.json` preserves
this real training proof. The durable owner resumes routine checks; no separate
agent monitor was created.

### Reviewed1x continuation, September30

Collected V1000.3x is30.54%, CE2.015569 and stops. V1001x is32.16%,
CE1.860963 and passes both original epoch1 limits. Advance only1x through5
from its exact validated epoch1 checkpoint, on one V10032GB in the same
account/runtime,4h cap, expected about3–4h; same source/inputs, no smokes.
Reuse the existing native5 result62.38%, CE1.055644 for its milestone gate.

This explicitly supersedes waiting for the two existing fallback epoch5 results
before allocating1x. Its original initial-candidate epoch5 allowance of14,400s
was never submitted in either lab cohort (both corresponding IDs are unknown).
Transfer that allowance once to the V100 continuation. Four stage5 allocations
including native now total57,600s within the original72,000s initial-plus-fallback
envelope;14,400s remain unassigned. No final-stage debit, budget increase or
extra native control. `analysis/beta1-e5-funding.json` records the gate and charge.

Use a new versioned `launch/transport-beta1-e5.py`, `beta_1-5.sbatch` and
`extension-beta1-e5.sha256`; pin the exact parent. Original launch artifacts and
all scientific source stay unchanged. Fourteen CPU transport tests passed,
including parent mismatch rejection, idempotence and denial of1x stages10+.
The same owner collects this job, even if the fallback policy terminates.
Main reviews1x after5; the original automatic0.1x/0.0001x path retains priority.
At most two distinct EqProp candidates may ever progress beyond5 cumulatively;
if both existing branches take those slots,1x stops at5. No automatic1x promotion.

Accepted Slurm ID343367, four-hour allocation, initially pending for priority;
now running on r8i1n2, PID1194983. The owner verified update1420 with finite
loss1.777071 and the expected resumed epoch2 training-order hash.
The parent hash, new transport/script hashes, shell syntax and scheduler static
check passed; receipts: `analysis/beta1-e5-{static,submission}.json`.
Observed V100 epoch durations are~3,176s for0.1x and~3,250s for0.0001x;
five additional epochs would exceed the prepared14,400s stage10 allowance.
Reallocate10,800s of the unused initial0.3x epoch5 allowance to raise stage10
to18,000s per branch across native plus at most two EqProp candidates.0.3x
failed epoch1 and has no stage5 submission; its remaining3,600s stay unassigned.
The combined141.5h ceiling and final-stage caps remain unchanged. Native,
0.1x and0.0001x use new `*-10-timed.sbatch` files and the versioned
`transport-stage10-timed.py`; the original launch files remain immutable.
The owner selects these files only after the unchanged gate passes, retaining
the same parent verification and allocation registry.1x still requires manual
epoch5 review and is denied at10 until then. Fifteen CPU transport tests pass,
including refusal to substitute the old undersized script when the timed one
is missing. Timing/funding evidence: `analysis/stage5-observed-timing.json`
and `analysis/stage10-timing-budget.json`. No next-stage job is submitted yet.
The revised monitor was installed locally/remotely under its existing owner
lock after static hash/shell/Slurm checks; its first normal poll is healthy.
Static receipt: `analysis/stage10-timing-static.json`. Original45 frozen
launch files remain unchanged, as does the immutable1x epoch5 transport.
The owner also projects stages10/20/30/50 from the last validated continuation's
seconds per epoch plus10% margin. If that exceeds the allocation or remaining
deadline, it raises an explicit budget-review error before submission; main
must reallocate or choose a justified stopping point. This operational check
does not change scientific gates or running jobs. Sixteen CPU transport tests
pass; installed monitor identity is in `analysis/timing-guard.json`.
