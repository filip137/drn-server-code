# CIFAR L8 BN ablations — September 23, 2026

User-approved exploratory follow-up, seven fresh seed0 cases, ten epochs,
batch32/evaluation16. No repeats or LR search. Reuse existing three selected
references and the active legacy_voltage_normalized run in the mechanism study.

| Lane | Cases, in order | Compute budget |
|---|---|---|
| Fifi RTX5090 | baseline_epsilon_legacy, baseline_frozen_bn, baseline_no_bn | 8 GPUh |
| Loulou RTX5090 | legacy_frozen_bn, legacy_no_bn, after current normalized legacy completes | 5.5 GPUh |
| Local RTX3090 | ours_frozen_bn, ours_no_bn | 10.5 GPUh |

Total24GPUh including smokes/qualification; expected17–20GPUh. Per training
case2.6h on5090,5h on3090. One worker per GPU; no CPU execution/offloading.
Queue allowance24h, heartbeat30s or less; monitor every30min, stall deadline45min.
No automatic extension past10, no LR retuning, no official test reads.

Baseline epsilon is [3.90625e-8,3.90625e-8,6.25e-7]. Frozen BN keeps gamma1,
beta0, minibatch normalization and running-statistic updates; boundary gains
remain trainable. No BN replaces its modules with identity, preserving pooling
and gains without compensation. All conductances and surviving parameters
come from the exact saved common initializer. Use each scheme's exact selected
Adam vector, original50epoch cosine horizon, crop/flip augmentation, CE, perfect
diodes, zero frozen analog biases, original bounds and [6,6,4] T/K.

Readable configs: configs/cifar/bn_ablation_20260923/. Source defaults preserve
old BN behavior; preparation and resume reject BN-contract mismatches. Frozen
parameters are excluded from optimizer and gradient checks. No-BN transfer
explicitly omits only BN initializer keys, ordinary restore remains strict.

Before production use a same-runner real-data smoke and existing small solver
qualification, with only [6,6,4] admitted. Failed qualification/nonfinite results
are retained and excluded; no silent changes to epsilon, gains, LR or T/K.
Compare train/validation CE and accuracy, first-batch pooled voltage/boundary
output/logit scale and conductance gradients. Epsilon, learned BN affine terms,
and overall BN removal are distinct interventions. Hardware differences and
single seed limit cross-scheme conclusions. Collect and validate local outputs
before concluding. Record failures/exclusions instead of repeating controls.

Result root: results/cifar10-l8-bn-ablation-seed0-20260923-v1/.

Implementation checks:35 tests passed, two CUDA-only skips in the sandbox.
Local GPU uses PyTorch2.5.1+cu121, unlike the5090 PyTorch2.11/cu128 stack.
Initial local qualification exposed the known cuDNN9.19 incompatibility;
archive operational-attempt-01 retains those checks. Existing isolated cuDNN9.1
preload fixes the minimal BN forward/backward probe; all checks repeat under
that local runtime. No global environment changes, no scientific changes.
Both the first attempt and retries count toward the local10.5GPUh budget.

All seven qualifications and real-data one-batch smokes passed using the
accepted local runtime. Every qualification selected exactly[6,6,4]. All
surviving initializer tensors match the shared source exactly; full-width
frozen-BN smoke checkpoints preserve gamma1/beta0 and advance counters once.
Launch uses direct run_lane.py --target fifi|loulou|local in tmux
cifar-l8-bn-ablation-20260923. State/exit receipts: launcher/<target>.json
and launcher/<target>.exit_code. Logs: logs/<case>.log. Loulou waits for its
current control; Fifi/local start independently. Read-only summaries use
python -m experiments.summarize_cifar_l8_bn_ablation RESULT_ROOT.

User steering at15:46UTC: launch on Loulou now alongside the current normalized
legacy control. Admit only with at least14GiB free and no unrelated GPU
clients; the known existing control is explicitly allowed. Extend only the
legacy_frozen_bn runtime cap to4h for sharing; Loulou5.5h and total24h compute
budgets remain binding. All scientific settings unchanged. The immutable
source snapshot retains the earlier2.6h config copy; the launched config is
the explicit runtime-updated root configs/legacy_frozen_bn.json.

Launch confirmed15:47UTC: Fifi lane2611559/worker2611564 runs the epsilon
control; Loulou lane1576002/worker1576007 runs legacy_frozen_bn alongside
existing normalized-legacy worker1568364; local lane1534737/worker1534743
runs ours_frozen_bn. All show batch progress, correct GPU allocation and
persistent tmux sessions. Initial shared-Loulou pace is about1.15s/batch,
versus about0.54s alone. Check remaining lane budget against this measured
pace before the second legacy case starts; total24GPUh remains authoritative.
Next full watchdog check16:20UTC; collect/validate terminal artifacts before
scientific closeout. No new reference training was scheduled.

Operational placement correction16:05UTC: another user's20GiB process started
on Fifi after our admission, leaving about650MiB free and slowing the worker.
Riri is now idle and verified identical RTX5090/PyTorch2.11/cu128/cuDNN9.19.
Move only our baseline lane to Riri after its first saved epoch; preserve
Fifi's interrupted attempt under recovery-fifi and resume weights, Adam,
scheduler and RNG without repeating epoch1. Transfer the original8h baseline
lane budget, counting Fifi time; other lanes and total24h remain unchanged.
Source-recovery adds an explicit interrupted-checkpoint CLI path with strict
scientific-contract and Adam-step checks; physical code is unchanged.

Riri recovery smoke passed: one real training and validation batch from
epoch1, restoring Adam step1407 and scheduler epoch1. Production resumed
epoch2 at about16:12UTC, lane1965770, worker1965775. Checkpoint SHA256
b4227b4a087ce84077be2ecf98b4b17a88819e19a6c7ad5839daac0fdf3a8576.
Fifi elapsed1285.28s plus recovery smoke1.71s count against the original
8h baseline lane budget. The saved epoch1 metrics are stitched into analysis;
unsaved partial epoch2 work on Fifi is discarded and charged. No full epoch
is repeated. Riri's system UTC clock is approximately2h behind controller
time; monitor heartbeat age against Riri's own clock and use monotonic elapsed
time for budgets. Do not change the host clock. Physical source is unchanged.

Watchdog16:18UTC: all four workers advance. Riri epsilon control epoch2
batch600; Loulou frozen legacy epoch2 batch100; normalized legacy epoch4
batch600; local frozen proposed epoch2 batch100. First frozen epochs take
1690.87s on shared Loulou and1782.32s locally. Epoch1 val accuracy: epsilon
baseline31.86% (reference27.92%), frozen legacy30.48% (35.52%), frozen
proposed34.46% (36.16%). These are provisional matched-epoch observations.
Read-only analysis of the accepted no-BN smokes finds initial logit RMS
baseline8.52383e-7, ours5.93856e-6, legacy8.72840e-4; legacy/baseline is
exactly1024, consistent with accumulated16*16*4 voltage factors. No new GPU
diagnostic was run for this observation.

Watchdog16:47UTC: all workers advance. Riri completes epochs2/3 in767.50/
768.60s; frozen legacy epochs1/2 in1690.87/1691.43s; local frozen proposed
epochs1/2 in1782.32/1782.53s. Normalized legacy epoch4 takes1691.15s.
At this steady shared pace, both Loulou workers may exceed their4h limits by
a few minutes: current deadlines19:17UTC for normalized legacy and19:47UTC
for frozen legacy. Review before18:45UTC and continue from saved epoch
boundaries with the strict recovery path if necessary; preserve Adam, scheduler,
RNG and completed metrics. No complete-epoch repeats. Total15h old-study and
24h new-study budgets remain binding; reconcile unused baseline-lane time
before admitting any recovery or the remaining legacy_no_bn case.

Analysis clarification16:54UTC: baseline_epsilon_legacy is the same effective
training condition as the earlier cancelled legacy_baseline_lr control, by
BN(s*x,eps)=BN(x,eps/s^2) and the shared baseline optimizer. All four train/val
loss/accuracy metrics match exactly for collected epochs1–3. The earlier
record has only five completed epochs; this running control extends the
condition to ten. Do not count their overlapping prefixes as independent
evidence. Historical cancellation and charged compute remain recorded.

Placement update17:36UTC: Fifi's RTX5090 is idle again (32,149MiB free,
0% utilization, no compute clients). Continue legacy_frozen_bn there from
Loulou's next saved checkpoint, preserving Adam/scheduler/RNG and all completed
epochs; then run legacy_no_bn. The legacy lane retains its total5.5GPUh cap,
including time already spent on Loulou. Expected remaining work is roughly
3.4h on the dedicated5090; total study cap remains24GPUh. This also frees
Loulou's GPU for the older normalized-legacy control and should remove its
predicted runtime overrun. Stage the already tested source-recovery runner;
Fifi's Torch2.11+cu128/cuDNN91900 matches Loulou and Riri. No additional
scientific cases or repeat epochs. Output remains cells/legacy_frozen_bn
and cells/legacy_no_bn under the same study root; preserve interrupted prefix
under recovery-loulou and use launcher/fifi_legacy.json for the continuation.

Transfer verified17:43UTC: saved epoch4 val54.76%, Adam5628, SHA256
`bf508267b66c5e9b6b509898a048665ac67af765ccaf6f48d77c56b89155ffc3`.
Loulou consumed6769.84s, charged to the same5.5h legacy budget. Fifi
parent2628530/worker2628559 restored epoch4/scheduler4 and started epoch5
(first-batch loss1.10369, matching the discarded unsaved first batch on Loulou).
GPU11862MiB; tmux `cifar-l8-bn-legacy-20260923`. Loulou normalized control
1568364 remained alive. Earlier prefixes and true interrupted status retained.

Parallel placement17:51UTC: Trex RTX5090 idle,31,960MiB free,0% GPU,
only idle MPS server; Torch2.11+cu128/cuDNN91900. Move the pending ours_no_bn
case from the local queue to Trex now, using the same accepted initializer,
config and smoke, with a2.6h runtime cap (expected about2.1h), charged to
the existing10.5h proposed-lane budget. Local ours_frozen_bn continues unchanged;
stop its parent after its successful terminal bundle to cancel the now-outsourced
queue item. No repeated training. Trex launcher/run output remain within the
same study root: launcher/trex.json and cells/ours_no_bn. This introduces
the already documented3090/5090 environment difference within the proposed
ablations; keep the interpretation exploratory. Combined local5h cap, checks
and Trex2.6h cap remain below10.5h. Total study budget remains24GPUh.

Parallel placement18:13UTC: queue baseline_no_bn on Loulou after the older
normalized-legacy control finishes, using exclusive GPU admission. Existing
accepted config/initializer, ten epochs/batch32,2.6h runtime cap, expected2.1h,
charged within the original8h baseline allocation. Riri continues
baseline_frozen_bn; stop its parent after that terminal bundle to cancel the
outsourced no-BN queue item. Baseline time already spent is8215.66s; adding
both frozen-BN and no-BN2.6h caps stays below8h. Record launcher/loulou_baseline.json
and cells/baseline_no_bn under the same study root. No extra cases, repeats or
LR changes. Both hosts use Torch2.11+cu128/cuDNN91900 on RTX5090.

Handoff verified18:28UTC: normalized legacy completed at75.82% validation
(+1.14pp versus its original reference), with passing audit and validated local
checkpoint. Loulou baseline_no_bn parent1613350/worker1616897 now runs on CUDA.
Its genuine metadata was also copied to Riri as a duplicate-start guard; collect
baseline_no_bn only from Loulou, not that Riri cache. Riri parent1965770 still
needs cancellation after baseline_frozen_bn finishes. Local parent1534737 likewise
needs cancellation after ours_frozen_bn finishes because ours_no_bn is on Trex.

Scientific failure18:41UTC: baseline_no_bn completed epoch1 at10% validation
and stopped on the first epoch2 batch because its gain gradient was zero.
A single3.834s read-only replay found every block output, logit and parameter
gradient exactly zero (also float64 norm/nonzero-count checks), with Adam
first moments at subnormal scale. No optimizer updates were performed in replay.
The failed bundle/checkpoint are local; record this as a scientific collapse
at the unchanged selected settings, with no LR retry or claim of ten-epoch
completion. Charge failed training750.951s plus replay3.834s to baseline.

Parallel placement18:47UTC: move the remaining legacy_no_bn case to now-idle
Loulou, unchanged config/initializer and2.6h cap, expected about2.1h. Record
launcher/loulou_legacy.json and cells/legacy_no_bn. Cancel Fifi parent2628530
after legacy_frozen_bn succeeds to remove its outsourced queue item. Reallocate
the unchanged24h total budget: baseline6h, legacy7.5h, proposed10.5h. Conservative
remaining case caps fit: baseline<=5.1h including its failed no-BN case and
diagnostic; legacy<=7.1h including original Loulou prefix, Fifi recovery cap and
new no-BN cap; proposed<=7.7h including local checks, frozen-BN cap and Trex cap.
Scientific configs and epoch targets are unchanged; only resource allocation shifts.

Placement correction18:48UTC, before launch: Loulou acquired nine unrelated
GPU clients and100% utilization (27,096MiB free). The staged legacy_no_bn
wrapper was NOT launched. Keep legacy_no_bn in the original Fifi queue after
legacy_frozen_bn; do NOT cancel Fifi parent2628530. The proposed budget
reallocation above was not activated: original baseline8h/legacy5.5h/proposed10.5h
remain the24h total. The unused Loulou wrapper is staging only. Riri/local
parent cancellations after frozen-BN completion remain required.

Fifi handoff verified19:00UTC: frozen legacy completed all10epochs at73.02%
validation (-1.66pp versus trainable BN), passing final solver audit. Local
checkpoint confirms gamma1/beta0 and14070 running-stat updates; stitched
epochs1–10 complete. Legacy no-BN worker2636677 started18:58UTC, CUDA11786MiB,
and reached batch200. Parent2628530 retains original5.5h legacy budget, with
11335.52s already spent and8464.48s remaining for this case. All7 cases have
now started;2 completed successfully, baseline/no-BN is a documented scientific
collapse, and4 cases remain active.

Scientific failure19:10UTC: legacy_no_bn also stopped at the first epoch2
batch after epoch1 validation10%/CE2.302585. Exact read-only replay on idle
Fifi confirms zero outputs/logits/all13gradients with float64 norm checks;
all gains99.999657, Adam moments subnormal. Replay1.031s, no optimizer updates.
Failed bundle and epoch1 checkpoint/Adam1407 are local; no LR retry or
ten-epoch/final-solver claim. Fifi parent completed_with_exclusions, total
legacy-lane compute12086.46s plus replay1.031s, under5.5h. Three cases remain
active: Riri baseline frozen, local proposed frozen, Trex proposed no-BN.

Queue-cancellation guards started19:49UTC for Riri and local. They wait up
to one hour for their frozen-BN child to finish and its bundle to validate,
then signal only the verified owned parent queue. Raw exit codes/receipts are
preserved, and outsourced queue items are annotated explicitly. Current GPU
training is untouched. Guards: analysis/stop_outsourced_queue.py --target riri
and --target local; Riri parent1965770/child1999518, local parent1534737/child1534743.
The genuine cached case manifests remain an additional duplicate-start guard.

Trex completion20:05UTC: proposed/no-BN completed ten epochs at52.08%
validation, CE1.538326, Adam14070. Full bundle/checkpoint/log collected and
internally valid. Final solver qualification failed: reference-versus-sentinel
free-logit relative L2 difference8.61% exceeds1%; operational[6,6,4] could
not be qualified. Preserve this as a terminal scientific limitation, with no
iteration or LR retry. Learned gains100.024506/100.024422/100.017387 and
head100.017365. Two cases remain active: frozen baseline on Riri and frozen
proposed locally, both now in epoch10 and advancing on CUDA.

Riri completion20:15UTC controller time (remote clock18:15UTC): frozen
baseline completed ten epochs at70.72% validation/CE0.849729, passing final
solver audit. Full local bundle/checkpoint valid, Adam14070, gamma1/beta0
and running-stat counters14070 verified. Queue guard stopped only the parent
after worker exit, preserving original exit1 from the intentional interruption;
outsourced baseline_no_bn was not repeated. Final Riri spent15876.84s includes
the earlier Fifi prefix and recovery smoke. Its raw receipt is retained under
analysis/riri_lane_before_outsource_annotation.json. Only local proposed frozen
BN remains active, expected completion around20:42UTC.

Reporting cleanup20:18UTC: collected the existing completed auxiliary baseline
reference from Fifi; it remains excluded from primary comparisons and no repeat
was launched. Preserved the early cached Fifi status bytes as
recovery-fifi/early_cached_metadata/status-at-collection.json so an obsolete
snapshot no longer advertises a running job in the generated dashboard.

Terminal handoff20:43UTC: proposed frozen BN completed ten epochs at73.76%
validation, CE0.748592 (-1.60pp versus trainable BN), with passing solver audit.
Final checkpoint confirms gamma1/beta0, running-stat counters14070 and Adam14070;
block gains100.009857/100.017990/99.957016 and head99.702438. Local queue guard
stopped the parent after successful child exit; the outsourced no-BN case was
not repeated. Original interruption exit1 and raw receipt are preserved.

All7 cases are terminal and locally validated:4 qualified ten-epoch results,
2 early zero-activation failures,1 completed but solver-unqualified no-BN result.
The existing-reference and older normalized-legacy comparisons are retained.
No active workers remain. Budget accounting, without double-counting migrated
prefixes: baseline16631.62s, legacy12087.49s, proposed26043.66s; total15.2119GPUh
of24GPUh, including checks, discarded partial work and failed-batch replays.
Analysis/collection_validation.json reconciles epoch/checkpoint/Adam coverage,
failed cases, interrupted prefixes, queue cancellations and budget. Findings
are recorded in docs/experimental_manifest.md and the linked mechanism report.
No new gain-parameterization or LR experiment was launched.
