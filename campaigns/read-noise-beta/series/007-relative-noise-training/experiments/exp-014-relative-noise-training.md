---
id: "exp-014"
title: "Ten-epoch Conv1/Conv2/Conv3 training at matched initial displacement and relative noise"
status: "complete"
hypotheses: ["H-011"]
---
# Exp014 — Relative-noise training sweep

## Authorized execution scope

Filip authorized codification and immediate scheduling on September25, adding
Conv1 at output RMS=1, interpreted consistently as normalized output D/F=1.
Launch Conv1/2 on local RTX3090 and nom-cool-1; wait for eligible RTX5090 capacity
for Conv3 and admit it immediately. This authorizes daytime/overnight continuation.

Frozen cases: ours and legacy, seed0,10epochs; Conv1 D/F=1, Conv2 D/F=1,2,4,
Conv3 D/F=4,6. Every architecture uses the same global relative eta grid:
**1e-8,1e-7,1e-6,1e-5,1e-4,3e-4,1e-3**. No baseline, clean controls or additional
model seeds are included in this execution. Total **84 trainings /840epochs**:
14Conv1,42Conv2,28Conv3. The added3e-4 samples the observed Conv3 last-layer
advantage;1e-8 covers its early-layer transition. Every noninput layer uses the
same eta, including the readout. Scientific source settings remain unchanged.

Result root (registered here before creation):
`results/conv123-relative-noise-training-20260925-v1/`.
Agent-created attempts and collections remain under this single root.

## Scientific contract

- Ordinary MNIST deterministic55k/5k split, seed0, official-test evaluation disabled.
- Use the exp007/008 Conv1/2/3 initialization and numerical training contract:
  centered frozen-current EqProp, paired physical20-output squared-error loss,
  Adam with unchanged source parameterwise LR vectors, batch16/validation64,
  float64, perfect diodes with explicit source dictionaries, weights[0,100],
  exact-zero frozen biases, input gains40/100/360, T=K4/6/8. This is not CIFAR/BPTT
  training and does not inherit CIFAR cross-entropy/BN settings.
- Ours(V,I)=(4,1), legacy=(4,.25). Preserve within-depth
  initializer, minibatch ordering, noise seed/draw pairing and optimizer setup.
- Relative endpoint acquisition: v_read = v + eta*abs(v)*xi independently for
  positive/negative phases and nodes/layers. Use the SAME eta at all noninput
  layers including the readout, without an absolute floor. Physical relaxation,
  input, nudging force and validation inference remain clean.
- Match initial output D/F on the accepted576-example validation cohort, using
  D=pooled RMS((v_plus-v_minus)/2) and F=pooled free-output RMS. Reuse exp013's
  accepted beta only where initializer/runtime/cohort and target identities match;
  match missing targets cleanly to1% tolerance, at most3 correction rounds.
  Freeze each beta for all10epochs; no adaptive rematching. Record injected B and
  internal beta B/(V/I)^depth separately. D/F may drift during training.
- Training acquisition in training/sgd.py now has an explicit nodewise_relative
  mode and endpoint_read_noise_eta, with config plumbing through labs/mnist_train.py.
  Absolute noise remains the default; the new relative configs require std=0. Reuse exact_run; do not silently
  relabel its absolute-noise std as relative eta. Implementation and actual source identity are recorded in the execution handoff below.

## Evidence and decision

Primary: epoch10 validation accuracy ours-minus-legacy for every matched cell,
not independently selected best epochs. Report epochwise training/validation loss,
accuracy, numerical failures, actual training duration and final checkpoints.
Keep all cells, including high-noise failures; display each architecture separately.
H-011's sampled-existence and full-grid mean claims are distinct. Single-seed
exploratory evidence cannot establish significance or a global optimum. Clean
controls, if added, quantify degradation from noise rather than only scheme ranking.

## Historical estimate before the authorized placement change

Measured timing source: exp008's authoritative collected terminal status files,
results/eqprop-conv123-relative-output-noise-extension-20260924-v1/collected/
{riri,trex,jean-zay}/production/**/status.json. Durations are completed_at minus
started_at on each same host, not timestamps compared between machines.

- Conv2:33 complete10-epoch runs on RTX5090s, mean0.4961GPUh/run,
  observed0.4716–0.5353h. Ours/legacy mean0.4948h (~30minutes).
- Conv3:24 complete10-epoch V100 runs, mean1.4684GPUh/run,
  observed1.4314–1.7742h. Ours/legacy mean1.4761h (~89minutes).
- Prior runs used absolute noise. Relative noise adds pointwise arithmetic but
  its training throughput has not been measured. Contention, large-beta dynamics,
  validation/startup and recovery can affect runtime. These are planning estimates,
  not equal-performance claims across GPU models.

| Scope | Conv2 runs / GPUh | Conv3 runs / GPUh | Production total | Planning allowance |
|---|---:|---:|---:|---:|
| Ours + legacy |42 /20.8 |28 /41.3 |62.1h |70–80 GPUh |
| All three |63 /31.3 |42 /61.7 |92.9h |105–120 GPUh |

Optional clean controls add about8.9production GPUh for two schemes, or13.3h
for three. If omitting the proposed3e-4 level, subtract those same amounts.
Every additional eta across all displacement targets has that same cost.
Proposed hard allowance80GPUh for the two-scheme core, or120GPUh for three,
including preparation/retries; not an allocation or approval to launch.

The earlier, superseded placement proposal used two RTX5090s for Conv2 and three V100s for Conv3. With those
five GPUs concurrently available, production lower-bound makespans are about
max(20.8/2,41.3/3)=13.8hours for two schemes, or20.6hours for three. Allow roughly
16–20 or24–30wallhours respectively, plus queue delays. More/faster GPUs could
change wall time and totalGPUh; no Conv3-on-5090 estimate is asserted without
training measurements. Live resource availability has not been checked because
this is planning only. Keep matched scheme pairs/triplets on the same host/class;
Ben sharing is authorized with ownership/memory checks and measured throughput.

## Execution and monitoring handoff

Authorized budget: **120GPUh including preparation, startup/teardown and retries**;
absolute wall deadline48hours after first production admission. This replaces the
prior two-depth80GPUh proposal because Conv1 is added and placement changes to
3090s for Conv2 and5090s for Conv3. Earlier timing figures above are historical
reference, not a forecast for these new hosts. Re-estimate from first production
epochs; do not spend the cap merely because it is available.

Immediate lanes: local3090 and nom-cool-1 for complete matched ours/legacy pairs
of Conv1/2. Conv3 eligible hosts: verified5090s among trex,fifi,loulou,riri. User's
standing Ben-sharing permission applies when ownership and memory permit; preserve
unrelated jobs. Never CPU fallback. Check all configured resources read-only;
no JeanZay submission is assigned in this execution. Keep each matched pair on
one recorded host and avoid duplicate queue owners.

Preparation owns the relative-noise implementation and exact readable configs.
Capacity owns source staging, detached host queues and per-case outer-wall receipts.
The Monitor owns persistent run-watch with300-second deterministic polling, one
bounded incident agent at a time and6daily/2perincident calls. Preassigned host queues
admit successive cases without one model call per case; capacity-wait incidents
admit Conv3 queues when an eligible5090 appears. No healthy model polling loops.

Filip's explicit waiver of tests/canaries remains active. Do not reintroduce
T/K, LR, parity or source qualification campaigns. Keep CUDA/nonfinite guards,
config/source identity, matched-beta evidence and real artifact-progress checks.
Admit complete10epoch cases expected to fit the remaining lane/wall budget.
The existing epoch-continuation helper stores the private acquisition RNG along
with optimizer/model/data-order/global RNG state. This execution admits whole
cases; do not replace that full-state contract with weights-only resumption.
If the deadline interrupts a case, preserve artifacts and report partial/failed
coverage. Never call an interrupted run complete. Routine accounting differences
are reconciled automatically from outer-wall receipts, without increasing caps.

Before root relinquishes monitoring: record actual source/config paths, pair
assignment, queue and worker PIDs, first semantic progress, wall deadline, remaining
lane allowances, executable inspect/collect commands, and watcher PID/observation.
On completion collect remote bundles locally, reconcile all84cases, interpret
H-011 perarchitecture and gridcell, produce JPGs and the result note, and regenerate
the ledger. Missing chat callbacks must not block final collection/review; retain
a durable final summary and truthfully report callback availability.

### Codified source and beta reuse

The readable generator is `experiments/prepare_relative_noise_training.py`.
It produced84exact configs under `configs/conv/relative_noise_training_20260925/`;
`provenance/preparation.json` records eachconfigsha and12reused matching contexts.
Every target already exists in exp013StageA; no newGPUcalibration or tests/canaries
were run. Compact source/model/optimizer/initializer/cohort comparisons verified
that the calibration and inherited training contracts agree.

The maintained implementation changes only relative acquisition plumbing in
`training/sgd.py` and `labs/mnist_train.py`: explicitnodewise_relative mode and eta,
absolute std0, no floor, noise scales from clean endpoint abs into a cloned read.
The inherited absolute mode remains thedefault. Existing full-state epoch
continuation stores acquisition RNG as well as optimizer/data/global RNG state;
no new continuation implementation or weights-only fallback is introduced.

Final analysis: `python -m experiments.analyze_relative_noise_training --study-root
results/conv123-relative-noise-training-20260925-v1 --config-root
configs/conv/relative_noise_training_20260925 --production-root LOCAL_PRODUCTION_ROOT`.
Repeat the production-root argument for authoritative collected hosts. The helper
reports every84arm/42pair, validates completedbundles, retains failures/missing
coverage, and writes epoch10validationcomparisons and JPGplots underanalysis/.

Exact transport and watch handoffs live under the registered study's `transport/`
and `monitor/`, respectively. Frozen lane allocations are local20GPUh, nom-cool-1
36h, Trex18h, Fifi18h, Loulou14h and Riri14h, totaling120h. Local owns14Conv1cases
then14Conv2D/F4cases; nom-cool-1 owns28Conv2D/F1/2cases; Conv3assignments are8/8/6/6
cases across those5090lanes. Pairclaims prevent duplicate admission. Allocation
transfers, if needed, apply only to verified unstarted completepairs and preserve
the aggregatecap and originaldeadline.

### First production admission — September25,17:10Paris

Local controller2903194 launched in tmux `relative-noise-training-local-20260925`;
first exact_run PID2903208, CUDA worker2903227. Conv1 ours D/F1 eta1e-8 completed
epoch1 with finite trainingloss0.1236 and validationaccuracy95.12%, then advanced
toepoch2. Nom1controller3146320/exact_run3146337 admitted Conv2 ours D/F1 eta1e-8.
Trexcontroller3934209 is waiting: freshadmission found protected Kellian jobs and
only634MiBfree, superseding the earlier eligibleBen-only inventory. NoTrextraining
was admitted under that occupancy.

Frozen source archive SHA256:
`dd630ddc54c30e059598ba197d6fb2f4b921394e9e3b111ae10674d5bdb41517`.
Configs only rebase initializer paths inside this source snapshot; scientific
settings are unchanged. Absolute deadline **September27,17:10:30Europe/Paris**
(epoch1790521830); per-host launchers enforce equivalent monotonic remaining time,
so remote clockskew does not extend the window. Runtime receipt paths and exact
commands are under `transport/` and `launch/` in the studyroot.

### Monitoring handoff acknowledged

All six controllers are staged and active: local2903194 and nom13146320 are
training; Trex3934209, Fifi96447, Loulou2836636 and Riri3613052 are waiting for
protected workloads/memory. The queues enforce pair claims, capacity checks and
remaining budgets themselves. No Conv3 worker had been admitted at this handoff.

Persistent run-watch **PID2910801**, armed with `--dispatch`, was verified alive
with real observations from all six controllers and no probe/log errors. Checks
run every300seconds; bounded agents handle incidents and final collection/review,
not healthy polling or each successive case. First local run advanced to epoch5;
nom1 produced finite batch500 progress on CUDA3146340. Remote heartbeat checks
account for host clock differences.

[Transport handoff](../../../../../results/conv123-relative-noise-training-20260925-v1/transport/launch-handoff.md)
and [monitor handoff](../../../../../results/conv123-relative-noise-training-20260925-v1/monitor/handoff.md)
contain exact paths/commands. The executable collector is
`results/conv123-relative-noise-training-20260925-v1/monitor/collect_queues.py`.
Monitoring ownership is acknowledged; operational recovery escalates to the
Codifier while healthy queues continue. Completion writes a durable final summary;
automatic chat delivery is best-effort and is not guaranteed.

### Kellian sharing authorized — September 25 evening

Filip explicitly instructed: "launch it in parallel with kellian; allow this".
Exp014 may now share with verified Kellian workloads as well as Ben, provided
measured Conv3 peak memory plus headroom fits. This overrides the generic exclusive
Conv3 reservation for this campaign. Other owners, including Adrien, remain
protected. Scientific configs, completed runs, pair placement, accumulated runtime
and the original September27 deadline remain unchanged. Only waiting controllers
may be replaced to load the new admission rule; active training is not interrupted.
The watcher is paused for capacity's exclusive recovery and will be rearmed after
verified admission. Actual recovery handles and memory evidence follow in the
transport/monitor handoff.

Trex admission confirmed: replacement controller4027562, exact_run4027575 and
CUDA worker4027578 run the pending legacy D/F4 eta1e-8 case; epoch1batch1 produced
finite loss0.4994. The completed ours case was not repeated. Its11360.241924seconds
remain charged against the unchanged64800-second Trex allowance. Measured peak
reserved memory of that completed Conv3 case was4546MiB; the campaign admission
threshold is now8192MiB. Live shared occupancy: our worker5148MiB, Kellian19928MiB,
with6872MiBfree. Other users' jobs and GPU/MPS settings were not changed.

Monitoring resumed as PID3151033 with fresh, error-free observations from all six
controllers. The other 5090 controllers now also carry the Kellian-sharing rule:
Fifi182169, Loulou2927444, Riri3709633; they remain waiting on other protected
workloads or memory. Local and nom1 training were untouched. Watch identity,
incident history, original budget and deadline are preserved.

### Terminal collection and review — September 27

All84 cases /42 pairs validated locally with finite ten-epoch histories and intact indexed artifacts. Launcher receipts reconcile to79.878562GPUh within120GPUh. No exclusions or additional training. [Reviewed result](../results/exp-014-relative-noise-training.md) owns the scoped H-011 verdicts; monitor/collection-receipt.json and analysis/summary.json under the registered study root preserve validation evidence.
