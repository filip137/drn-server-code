---
id: "exp-013"
title: "All-scheme EqProp beta screening and conditional long training"
status: "running"
hypotheses: ["H-010"]
---
# All-scheme EqProp training

**Current user instruction, September30 09:00 Paris: stop new runs and report
existing evidence.** Do not launch, release, retry or extend any simulation.
Already-running jobs may finish their assigned endpoints and be collected.
The six ours fallback jobs remain held; the original baseline1x epoch5 job
remains cancelled, so existing lab callbacks cannot dispatch another run.
Baseline3090 is collect-only and the adaptive pilot is one-epoch-only. The
legacy Slurm owner now collects without advancing. This supersedes all launch
and promotion permissions below until Filip explicitly resumes them. Audit:
`results/cifar-eqprop-monitor-20260929-v1/user-stop-new-runs.json`.

Filip requested monitoring every30 minutes, expansion onto available5090s, and
high-accuracy EqProp for as many epochs as possible in baseline, ours and legacy.
First long-run target:50 epochs. This extends the question and staged policy in
[exp-012](exp-012-fresh-beta-staged.md); its active ours3090 jobs remain intact.

Each scheme starts fresh with its original epoch0 tensors, scheme-specific Adam
rates, trainable BN/gains/head, seed0, batch32/eval16, original45k/5k split and
augmentation, T/K[6,6,4],50-epoch cosine schedule, no read noise. The existing
hybrid update replaces Conv gradients with centered EqProp and retains autograd
BN/gain/head gradients. Legacy uses the original amplification configuration
from the matched diagnostic trajectory, not the separate end-voltage-normalized
experiment. No smokes or retired solver audits. Official test data remain unread.

Initial cases per scheme: native BPTT and fixed0.3x/1x the scheme's epoch10 beta
vector. The original learning rates are preserved; they are not copied between
schemes. Explicit candidates and runtime contracts live in
`configs/cifar/eqprop_all_schemes_20260929/`.

| Scheme | Epoch10 anchor RMS by block | Base beta by block |
| --- | --- | --- |
| Baseline | .01, .02, .0025 | .0362755257, 1.2233372947, .9409883142 |
| Ours | .02, .035, .001 | .0945232200, 3.7299863724, 1.1695196708 |
| Legacy | .01, .005, .0005 | .0314483196, 5.4672969664, .6754808249 |

These are candidates from exp-008, not proven optima. Legacy block3's anchor
was a search boundary. Fixed beta does not imply fixed RMS during training.

At milestones1,5,10,20,30, continue candidates only if accuracy is within2pp of
the same-scheme native BPTT and CE is at most1.05x BPTT. Extend to5,10,20,30,50
by restoring the exact prior checkpoint including Adam, BN, RNG and scheduler.
Apply exp-012's one-time six-multiplier fallback if all initial candidates fail
at1 or5; promote at most two passing fallback cases. Failure at10/20/30 stops
that branch for scientific review, rather than consuming longer budgets.
Epoch50 is a terminal measurement, not an automatic claim of success.

## Execution and monitoring handoff

Current routing, September30. Collected conclusions live in the
[partial review](../results/exp-013-all-scheme-long-eqprop.md); queue and Slurm
bundles own live progress. Main reviews the manual branches below.

| Branch | Current execution and next decision |
| --- | --- |
| Baseline0.3x | Stopped at epoch5:59.26%, CE1.147454 versus native62.58%, CE1.049603; fails both continuation checks. Await baseline1x review before deciding whether the baseline fallback is needed. |
| Baseline1x | [exp020](exp-020-baseline-3090-continuation.md): epoch5 collected58.14%, CE1.185933 versus matched native60.02%, CE1.101680; fails CE. No extension. |
| Baseline early screens |0.1x collected28.00%, CE2.214752: fails CE.3x collected34.78%, CE1.822241: passes epoch1. Both stop at1 under the user's no-new-runs instruction. |
| Ours fixed0.3x | Original ours5090 root: native5 collected; EqProp5 running on loulou. Explicit one-image boundary exception below; six fallback jobs held. Main owns subsequent decisions. |
| Ours current-batch beta | [exp021](exp-021-low-cap-block3-rms.md): one fresh epoch running on nom-cool-1 after native5 finished; manual review. |
| Legacy | [exp016](exp-016-legacy-v100-fallback.md):0.1x epoch5 collected57.00%, CE1.218064 versus native62.38%, CE1.055644; fails both.0.0001x and1x (343367) still through5. Collection only, no extension. Older5090/3090 roots are inactive provenance. |

The original preparations and later attempt records below provide provenance;
the current routes above govern execution. Original5090 roots:

- `results/cifar-eqprop-baseline-5090-20260929-v1/`
- `results/cifar-eqprop-ours-5090-20260929-v1/`
- `results/cifar-eqprop-legacy-5090-20260929-v1/`

September30 08:20 Paris allocation amendment: baseline0.3x failed both epoch5
gates and its completed queue job charged4,321.034s of14,400s. Allocate5,400s
of the closed10,078.966s remainder to run the existing baseline0.1x epoch1 case
early on available riri; reserve4,678.966s unassigned. This is a scheduling
exception to waiting for both initial epoch5 outcomes, not a changed scientific
gate. It tests the next lower predeclared multiplier with identical source,
initialization, runtime and original5090 native control. No extra native run,
budget increase or automatic early promotion. If the full fallback is needed,
reuse this exact case/result, never rerun or count its allowance twice. If1x
passes, this remains a one-epoch screen unless a later explicit review allocates
continuation. Main owns that decision; the existing callback still waits for
the original initial-stage evidence. Job:
`cifar-eqprop-baseline-e1-beta-0p1-20260929`; evidence and funding stay in the
original baseline root, `analysis/early-screen-budget.json`.
Startup verified on riri: attempt `49d18f39246c429b8b45b83a52b9d769`,
launcher PID2502972;20 real training updates with finite loss observed.
After0.1x completed and failed CE, close its actual charged time and allocate
5,400s of the remaining donor slack to the existing3x screen. The original
10,078.966s donor cap covers actual0.1x time plus the new reservation; no new
budget is created. `analysis/early-screen-budget.json` records exact charges.
Case selection tests the next higher predeclared multiplier because1x had the
strongest baseline epoch1. Preserve the same one-epoch/manual-review limit,
native reference and reuse rule. Job `cifar-eqprop-baseline-e1-beta-3-20260929`.

Frozen source, input identities and separate `cells/STAGE/CASE` bundles retain
all attempts. These three original cohorts were prepared for RTX5090 placements
on trex, fifi, loulou and riri, with torch2.11.0+cu128/CUDA12.8 and a separate
native control per scheme. Later cohorts retain their own matched controls.
Admission reserves16GiB plus2GiB headroom. Preserve other people's jobs; share
only with Ben. Respect the weekday09–18 two-5090 cap; nights/weekends are uncapped.

Initial nine screens:1.5GPU-hours maximum each,13.5GPU-hours total. Expected
roughly30–60minutes each, subject to measured shared-GPU throughput. The
original stage allowances were4h to5,4h to10,8h to20,8h to30,16h to50, charged
across attempts. Recorded transfers now set the final-stage cap to52,800seconds
for baseline,44,600 for ours and54,000 for legacy. The combined ceiling remains
141.5GPU-hours per scheme,424.5total,
including a failed initial branch and fallback. Only eligible stages execute;
this is a ceiling, not a compute target. Deadline October6 08:00 Paris.

The shared run-watch queue owns placement, minute-level progress checks,
bounded operational recovery and completion collection. Its completion callback
validates evidence, records decisions and submits prepared continuations.
Job IDs: `cifar-eqprop-SCHEME-eSTAGE-CASE-20260929` (case underscores become hyphens).
Numerical rejection is failed-training evidence; operational failures cannot
be interpreted as a beta failure. Runtime mismatch blocks promotion.

A read-only user-systemd timer produces a summary every30minutes in
`results/cifar-eqprop-monitor-20260929-v1/`: `latest.md`, `latest.json`, and
append-only `history.jsonl`. It reads queue-owned progress and collected metrics;
it does not launch jobs, probe GPUs, or duplicate queue recovery. No external
message delivery is configured. Reviewers inspect this report and each cohort's
`analysis/decision.json`; scientific conclusions still require result review.

Launch verified September29 22:21 Paris: all nine initial jobs accepted and queued;
both existing ours3090 jobs are advancing. The four5090s currently have no admissible
capacity. Source/input/training-data identities verified on all four hosts; no
GPU smoke executed. The shared queue service is active with a fresh heartbeat.
The first timer report has no alerts; next check22:30 Paris, then every half hour.
`cifar-eqprop-summary.timer` is enabled and its service completed successfully.
66 CPU tests passed for staged restoration, decision gates, runtime matching and
read-only summaries. Later-stage jobs remain prepared and are submitted only when
collected evidence passes the declared gate.

First5090 expansion verified: baseline native BPTT was admitted on riri at
September29 23:01 Paris (controller time), sharing with Ben, and collected by23:24.
Epoch1 validation27.92%, CE2.095195;1,407updates and full5,000-example validation
passed collection checks. Training/evaluation took1,241.4seconds. Receipt:
`results/cifar-eqprop-baseline-5090-20260929-v1/analysis/native_bptt-1-collection.json`.
The baseline EqProp cases remain queued. Riri's wall clock is about2hours30seconds
behind the controller; use queue issuance/collection times for cross-host chronology
and measured elapsed duration for run timing. No host clock settings were changed.

The midnight check found baseline0.3x OOM during initial validation on loulou:
an external process occupied28.79GiB after admission. The preserved initial
checkpoint has zero updates and empty Adam state; no training metrics or result.
Archive: original baseline root's
`failed_attempts/beta_0p3/64fd5ad2ca9b42d293b41a168edb883f/`.
Retry root `results/cifar-eqprop-baseline-5090-retry-20260930-v1/`, job
`cifar-eqprop-baseline-e1-beta-0p3-20260929-retry1`, retains5,366seconds after
deducting33.89seconds from the original allowance. The existing retry collector
preserves provenance and restores validated evidence to the original study.
No beta rejection, batch-size change or intervention in the external job.

September30 transport update: baseline remains on5090s because its1x job was
already running on riri when the relocation began. Attempt
`d6e7c8bdf26f439498501a2ca86e323d` (supervisor2386435, child2386437) had440
updates at584.5seconds; its0.3x retry was released back to the existing queue.
The successful5090 native result and failed0.3x attempt remain preserved.

Legacy is relocated to `results/cifar-eqprop-legacy-3090-20260930-v1/`, with
nom-cool-1 as its only prepared placement. All404 identified source files and
scientific inputs are byte-identical to the frozen5090 cohort; only the runtime
contract and transport paths change. The new matchednative and both EqProp
screens use torch2.5.1+cu121/CUDA12.1/privatecuDNN9.1, verified without a GPU
smoke. Training-data hashes, initializer, split, cases, original learning rates,
staged decisions and full-state continuation remain unchanged. The54 prepared
case/milestone jobs cover1/5/10/20/30/50; only passing branches are submitted.

The three original legacy initial jobs were held before staging, with zero
attempts and zero charged time. All three were cancelled before replacement
submission; all replacements were accepted by the shared queue. Initial IDs are:

- `cifar-eqprop-legacy-3090-e1-native-bptt-20260930`
- `cifar-eqprop-legacy-3090-e1-beta-0p3-20260930`
- `cifar-eqprop-legacy-3090-e1-beta-1-20260930`

Continuations use `cifar-eqprop-legacy-3090-eSTAGE-CASE-20260930`, with underscores
in CASE replaced by hyphens. Each initial job retains5,400seconds; subsequent
stage caps and the existing141.5GPU-hour conditional ceiling remain unchanged,
with no attempt-budget reset. Deadline remains October6 08:00 Paris. Expected
initial duration is roughly30–60minutes each under measured shared throughput.
Admission reserves16GiB plus2GiB headroom. The local3090 is excluded so the
separate ours adaptive comparison can use it.

Nom-cool-1 currently hosts Ben's workload. A hashed transport-only owner guard
checks the queue-assigned GPU immediately before the unchanged runner and
permits only identified Ben/Filip processes. Unknown or other owners cause an
operational failure before training; they are not beta rejections. Global
sharing configuration, MPS and other people's jobs remain untouched. The guard
cannot prevent another process arriving after its check; queue memory admission
and reservations remain authoritative. Runtime/hash proof and transfer
provenance are in the new root's `analysis/transport-verification.json` and
`analysis/placement-transfer.json`. Shared run-watch remains the sole launcher
and monitor; the existing half-hour summary retains original roots and gains
this replacement root.

Legacy3090 launch verified September30 01:53 Paris: native BPTT running on
nom-cool-1, queue attempt`f949d269bd78451084f2def500557aa2`, worker411099,
GPU`GPU-2262d56b-2c1d-1546-3e93-2e441399cd4f`; both EqProp cases queued.
The owner guard observed only Ben. The native bundle has its manifest, running
status and initial checkpoint, with the matched2.5.1+cu121/CUDA12.1 environment
and full initial validation in progress. The service heartbeat was28seconds old
and its controller PID1124310 remained active. The summary service completed
successfully after adding only the new legacy root; the existing timer is active
with next check02:00 Paris, then every30minutes. Baseline1x remained active with
640updates at875.5seconds in the contemporaneous queue observation; another
process arriving later on riri was preserved and is not claimed as an authorized
new sharing placement.

The prepared ours5090 final-stage allowance is now15hours; baseline and legacy
retain16hours. Across at most three promoted branches this reserves3GPU-hours
for the bounded exp-014 pair within the existing combined ours141.5GPU-hour
ceiling; the experiment budget is not increased.

### Baseline0.3x second operational retry, September30

The first retry on loulou, attempt`08bf1e4fb7974975bdb430d03040c47d`, failed
during initial validation after37.6990075seconds. The log records an external
process725118 using28.79GiB after admission observed only Ben. Both queue process
and reservation checks are terminal. The collected initial checkpoint proves
zero updates, empty Adam state and epoch/scheduler0; metrics are empty and there
is no final checkpoint or result. This is operational failure, not beta evidence.

The failed retry root and remote bundle remain unchanged. An additional archive
with the initial checkpoint, queue receipt and`pretraining-proof.json` is under
the canonical baseline root's
`failed_attempts/beta_0p3/08bf1e4fb7974975bdb430d03040c47d/`.

New root: `results/cifar-eqprop-baseline-5090-retry2-20260930-v1/`.
Replacement ID: `cifar-eqprop-baseline-e1-beta-0p3-20260929-retry2`.
The allowance is5,328seconds:5,366 minus the latest37.6990075seconds rounded up
to38; the original33.892seconds were already charged as34seconds. Source and
inputs remain byte-identical to retry1. All405 identified source files, six
inputs and prepared-file hashes were verified on trex, fifi and riri. Loulou is
excluded from this retry because repeated short capacity windows were followed
by external memory arrivals before training. No GPU/MPS settings or other jobs
were changed, and no additional scientific test or GPU smoke was run.

The existing retry collector validates the new evidence, restores only an empty
or identical canonical bundle, preserves provenance and resumes the original
staged decision policy. Retry1 is held for this recovery; replacement acceptance
is recorded below. The queue remains the sole monitoring owner. Baseline native
and1x continuations, and the ours minibatch comparison, are outside this recovery.

Retry1 was cancelled only after its failure and released reservation were
reconfirmed. Retry2 is accepted and queued on trex/fifi/riri with5,328seconds;
this is queue ownership, not a claim of compute startup. The persistent queue
service has a fresh heartbeat. Current handle/progress evidence is
`results/cifar-eqprop-baseline-5090-retry2-20260930-v1/analysis/queue-handoff.json`.

### Ours5090 initial-cohort placement recovery, September30

Native attempt`444ba7db71d24830b43491c47e8c3d6d` failed on loulou after
32.8889949seconds as another process746203 used28.78GiB after admission. The
initial validation completed (one metric row,10.04% accuracy); the traceback
places OOM in an early training forward pass. The saved initial checkpoint has
zero updates, empty Adam and epoch/scheduler0. No training update was recorded,
no later checkpoint/result exists, and the exact unsaved in-memory update count
cannot be reconstructed from logs emitted every20updates. This is operational
failure, not an EqProp or beta result. The original remote evidence is preserved;
the local archive and full traceback are under the canonical ours root's
`failed_attempts/native_bptt/444ba7db71d24830b43491c47e8c3d6d/`, with
`saved-state-proof.json` separating the proven saved state from that limitation.

The failed native and both never-started EqProp jobs were held before changes.
Replacement root: `results/cifar-eqprop-ours-5090-retry-20260930-v1/`, with
byte-identical frozen source and scientific inputs. All404 identified source
files, six inputs and collector hashes were verified on trex, fifi and riri.
These are the only allowed placements. Loulou is excluded after repeated
external-memory arrivals around startup; no GPU/MPS settings or other people's
jobs are altered. The same native/EqProp computation and existing collector
remain in use.

The mappings are:

| Original initial job | Replacement job | Remaining allowance |
| --- | --- | --- |
| `cifar-eqprop-ours-e1-native-bptt-20260929` | `cifar-eqprop-ours-e1-native-bptt-20260929-retry1` |5,367s|
| `cifar-eqprop-ours-e1-beta-0p3-20260929` | `cifar-eqprop-ours-e1-beta-0p3-20260929-placement2` |5,400s|
| `cifar-eqprop-ours-e1-beta-1-20260929` | `cifar-eqprop-ours-e1-beta-1-20260929-placement2` |5,400s|

Native's32.8889949seconds are charged as33seconds. Both EqProp originals have
zero attempts and retain their full allowance. Cancelled IDs retain queue
ownership of their original output destinations, so all three new IDs execute
in the replacement root. The existing retry collector validates each bundle,
restores its original canonical cell atomically, records retry provenance and
calls the original cohort's decision policy. The original prepared initial
launch files reference replacement IDs, preserving submission idempotence.
No queue database ownership was changed.

Loulou was also removed from100 unsubmitted future ours/baseline prepared job
specs, without changing queued/running jobs or their budgets. The current ours
final-stage cap remains14hours (50,400seconds); baseline and legacy remain
16hours. Six GPU-hours are reserved for bounded exp-014/exp-015 within the
combined ours141.5GPU-hour ceiling. Detailed mappings, unchanged-budget pruning
and verified identities are in the replacement root's `analysis/` receipts.
Legacy3090 and the ours minibatch recovery remain untouched. The existing queue
continues to own launch, capacity waits and monitoring; acceptance follows below.

All three held originals were cancelled after their terminal/zero-attempt
conditions were reconfirmed. The native retry and both placement replacements
are accepted and queued on trex/fifi/riri; no compute startup is claimed. The
shared queue has a fresh heartbeat. `analysis/queue-handoff.json` in the new
retry root records accepted IDs, allowances and placement restrictions.

### Ours initial-only loulou readmission, September30 06:29Paris

At06:25Paris loulou again had27,198MiB free and only Ben's GPU processes,
exceeding the unchanged16GiB reservation plus2GiB headroom. The coordinating
Monitor authorized readmission for the three already-assigned initial cells
only. No future-stage placement restrictions or scientific settings changed.
A fresh transport root is
`results/cifar-eqprop-ours-5090-loulou-retry-20260930-v1/`; the previous
`results/cifar-eqprop-ours-5090-retry-20260930-v1/` remains preserved and linked.
Original failed native evidence and its32.88899493-second debit remain under
`results/cifar-eqprop-ours-5090-20260929-v1/failed_attempts/`.

All scientific source and inputs are byte-identical to the preceding retry
root. The existing canonical retry collector is unchanged. A frozen Ben-only
prelaunch owner guard was added as transport, while the shared queue retains
memory admission and operational ownership. CPU imports/source/input/prepared
hash checks passed with torch2.11.0+cu128, CUDA12.8 and cuDNN91900 on all
prepared targets. No GPU smoke or GPU-wide change occurred. Live loulou owners
and headroom were rechecked immediately before native dispatch; the actual
prelaunch guard again found only Ben. A later external arrival remains possible.

| Unstarted predecessor | Accepted replacement | Remaining allowance | Placement |
| --- | --- | --- | --- |
| `cifar-eqprop-ours-e1-native-bptt-20260929-retry1` | `cifar-eqprop-ours-e1-native-bptt-20260929-retry2` |5,367s|loulou|
| `cifar-eqprop-ours-e1-beta-0p3-20260929-placement2` | `cifar-eqprop-ours-e1-beta-0p3-20260929-placement3` |5,400s|trex/fifi/riri/loulou|
| `cifar-eqprop-ours-e1-beta-1-20260929-placement2` | `cifar-eqprop-ours-e1-beta-1-20260929-placement3` |5,400s|trex/fifi/riri/loulou|

Each predecessor was verified unstarted with zero attempts, held during
preparation, and cancelled only after its replacement was fully staged. The
canonical original root's three initial launch specs now reference these IDs,
so its unchanged stage decision cannot recreate a superseded job. The pair was
transferred only after native actually started. Both EqProp jobs are queued;
they will not duplicate the currently running control. No allowance was reset.

Native started06:27:14Paris, PID809210, attempt
`3b1a0b3d1863469486d224c9372f2c49`, GPU
`GPU-4673400d-1878-5972-d12e-eafaa740890b`. At06:28:35 it had completed60
optimizer updates with finite loss2.32105494, proving training beyond the
previous initial-validation OOM point. The canonical retry collector will
validate and atomically return its bundle to the original empty cell, then
reuse the original stage policy. `analysis/queue-handoff.json`,
`analysis/prelaunch-check.json`, `analysis/placement-transfer.json` and
`analysis/eqprop-placement-transfer.json` in the new root preserve the exact
handles, checks and budget mapping. Shared queue remains the sole owner;
the existing half-hour summary was refreshed after replacement acceptance.

### Prepared loulou option for ours epoch5, September30

Added loulou only to the three unsubmitted canonical prepared specs
`launch/native_bptt-5.json`, `launch/beta_0p3-5.json` and `launch/beta_1-5.json`
under `results/cifar-eqprop-ours-5090-20260929-v1/`. All three queue IDs were
verified absent before editing. Their14,400-second budgets,16GiB+2GiB memory
admission, existing trex/fifi/riri placements, scientific source/inputs and
completion callbacks remain unchanged. The added placement uses the existing
Ben-only prelaunch guard. Exact frozen source, inputs, training data and prepared
file hashes pass on loulou; CPU imports confirm torch2.11.0+cu128/CUDA12.8/
cuDNN91900. Parent checkpoint destination directories are ready; the unchanged
`submit` function will copy and hash the actual epoch1 parent at authorized
launch. No stage5 output directory was created, no GPU smoke ran, and no job
was submitted or cancelled. This is transport readiness only, with no new
scientific promotion or boundary exception. Future stages and fallback specs
were not modified. Before/after specs, queue-absence receipts and static checks
are in the canonical root's `analysis/loulou-stage5-preparation/`.

### Explicit boundary follow-up for ours, September30

Both initial candidates failed the original epoch1 allocation rule:0.3x reached
34.14%, CE1.787529 versus native36.16%, CE1.730916, missing accuracy by one
of5,000 validation images;1x reached35.20%, CE1.867973 and failed CE.
Keep both failures recorded. The coordinating reviewer adopts one posthoc
allocation exception: test0.3x through epoch5 before spending on the broad
fallback. Its CE passes and its one-image accuracy shortfall is too narrow to
settle whether the trajectory remains competitive over several epochs. This
justifies further measurement, not a reclassification or statistical claim.

Continue native and0.3x from their exact full epoch1 checkpoints, preserving
all science and the50-epoch schedule. Use their existing14,400-second stage5
allowances; expect approximately45–90minutes each on an available5090, with
loulou currently eligible. No aggregate budget increase, extra control rerun,
smoke or official test access. Six automatically queued fallback cases were
verified unstarted with zero time used; hold them while this comparison runs.

At epoch5 apply the unchanged rule: accuracy>=matched native minus2pp and
CE<=1.05 times matched native. A failure stops0.3x and releases the original
fallback; a pass may continue to10 and later milestones under the same gates
and recorded budget. Cumulatively promote at most two distinct EqProp cases
beyond epoch5, plus native, as originally budgeted; a later failed continuation
still consumes its slot. Report the changed selection
path in every later review. This exception changes allocation, not the failed
epoch1 receipt, the scientific measurements or the definition of a pass.

Main owns this explicit scientific decision and any later promotion. The shared
queue remains sole operational owner. Existing callbacks may still report the
original fallback phase and see held job IDs; they do not release held jobs.
The six held fallbacks and accepted stage5 handles are recorded below after
queue operations. No automatic gate or frozen numerical source is modified.

Accepted stage5 IDs: `cifar-eqprop-ours-e5-native-bptt-20260929` and
`cifar-eqprop-ours-e5-beta-0p3-20260929`, each14,400seconds. Native is training
on loulou (attempt`dc3899f9fe7040d1b4c831b2c2f8e110`, PID822842); actual
progress reached1,460 total updates beyond its1,407-update epoch1 parent,
with finite loss. The EqProp job is queued for an eligible5090. The six fallback
IDs `cifar-eqprop-ours-e1-beta-{0p0001,0p001,0p01,0p03,0p1,3}-20260929`
are held with zero attempts and zero time used. Queue-operation receipts are
`analysis/boundary-followup-holds.json` and `analysis/boundary-followup-launch.json`
in the original ours5090 root. These paths and the existing bundles own live state.

### Ours epoch5 endpoint, September30 09:25 Paris

Collected and validated:0.3x54.12%, CE1.255383 versus matched native66.68%,
CE0.941809. Both gates fail; no extension and no release of held fallbacks.
Result interpretation lives in the exp013 result note. Only legacy1x job343367
remains running; verified live at09:24 Paris, update2300, finite loss. The
Slurm owner remains collection-only and next30-minute summary is09:30.
