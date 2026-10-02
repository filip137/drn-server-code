# Showing the effect of legacy output scaling

User question September 28: how can we show or plot the proposed connection
between legacy's voltage rescaling and its clean Conv1/2/3 training advantage?
Related discussion: [CIFAR legacy gap](../cifar10-legacy-gap/README.md).

## Scope and figure

Existing-artifact analysis and an analytic illustration; no new model evaluation,
dataset access, optimizer steps or simulations. State: complete and reviewed.
Target: local CPU, expected seconds, cap 2 CPU-minutes.
Result root: `results/conv-output-scale-evidence-20260928-v1/`.
Command: `/home/filip/miniconda3/envs/py312/bin/python experiments/plot_conv_output_scale_evidence.py`.

Three panels, with distinct evidence roles:

1. Existing same-initializer Conv3 free-state RMS ratios, legacy/baseline,
   before and after division by the predicted layer factors 1/4/16/64.
   Use only output-D/F target 2 from exp012's collected physical-state summary,
   where all schemes ran on the local GPU. The free states do not depend on the
   nudging target; the target-6 rows are not additional replicates.
2. An analytic single-positive-class squared-error slice,
   `L_s(f)/s^2 = (f - 1/s)^2 / 2`, at s=1 and s=64. Dividing the plotted loss
   by s^2 exposes the changed minimizer. It does not assert that a scalar loss
   factor has no optimizer effect. These curves are not training measurements.
3. Existing corrected wide clean-BPTT best validation accuracy for Conv1/2/3,
   all three schemes, three seeds, mean and sample SD. Recompute each aggregate
   from its canonical result metrics and check the published summary. These
   historical arms use separately selected rates and do not isolate gain.

Sources: [voltage review](../read-noise-beta/series/006-relative-endpoint-noise/results/exp-012-relative-endpoint-noise.md),
its `results/conv3-relative-endpoint-noise-20260925-v1/analysis/physical_state_controls.csv`,
and [three-seed BPTT summaries](../../paper_ready_results/current_contract_validation_tables.md).
Reuse the original reviews; check summary arithmetic and preserve source hashes,
without claiming fresh validation of their underlying runs.

## Earlier discriminating follow-up — unlaunched, superseded in scope

First test the identity on identical weights and inputs, at initialization and
a saved trained checkpoint. For each architecture, compare legacy with baseline
whose paired output scores are multiplied by s=4/16/64 **inside the BPTT loss**.
Plot paired scores and conductance gradients against the identity line, with
relative error and gradient cosine. Positive scaling alone preserves the current
argmax, but its derivative must remain in the graph to match training.
Raw baseline-versus-legacy state RMS is insufficient to prove this identity.

For the training demonstration, use a common initializer, exact-zero biases,
clean BPTT, identical batches, fixed T/K, conductance bounds and runtime:

| Arm | Circuit | Output-score multiplier | Optimizer rates |
|---|---|---:|---|
| A | baseline | 1 | baseline |
| B | baseline | 1 | legacy |
| C | baseline | 4/16/64 | legacy |
| D | legacy | 1 (native amplification) | legacy |

Plot training loss and validation accuracy against optimizer steps, accompanied
by C-minus-D parameter/update discrepancies. C and D should agree under the
ideal matched computation. A versus B measures the rate change at baseline gain;
B versus C measures gain at legacy's rates. This four-arm path separates those
specific interventions but does not estimate a full gain-by-optimizer interaction;
that would require a fifth, gain-scaled baseline arm at baseline rates.

Begin with a short identical-minibatch prefix to test C/D identity before choosing
full training coverage. A full accuracy conclusion requires a declared horizon
and independent seeds. No new run or long-training budget is assigned by this
visualization request. Fixed-beta or noisy EqProp needs additional transformation
of nudging/noise and is outside this clean BPTT control. Ours has AB=4 and is not
covered by the baseline/legacy rescaling identity.

## Review of the existing-artifact figure

[Figure JPG](../../results/conv-output-scale-evidence-20260928-v1/analysis/output_scale_evidence.jpg)
and [source hashes and checked values](../../results/conv-output-scale-evidence-20260928-v1/analysis/summary.json).
Figure generation completed on local CPU in under one second. The rendered
figure was inspected; all 27 source best-validation metrics reproduce the
nine plotted means and sample SDs. No runs or seeds are excluded within the
declared wide-BPTT comparison. Other algorithms and bounded contracts are
outside this figure, not failed cases.

The saved same-initializer Conv3 free-voltage RMS ratios are exactly
1/4/16/64 at the precision of the CSV; all four normalized ratios are 1.
This supports the expected scaling of aggregate amplitudes, while an exact
functional/gradient identity still needs the matched replay described above.
The analytic minima lie at baseline-coordinate target 1 versus 1/64 for
Conv3. The performance panel retains the historical ranking: Conv3 mean best
validation is 97.767% baseline, 98.560% ours and 98.893% legacy.

The three panels make the explanation concrete, but they do not establish
that output gain caused the accuracy improvement. The next discriminating
evidence is C/D agreement in scores and gradients, followed by the four-arm
training comparison if assigned. No simulation or follow-up was launched.

## Authorized target-amplitude pilot — September 28

Filip approved implementing the Conv2 amplitude-only plan on Jean Zay: six
trainings, baseline / ours / legacy crossed with target scales 1/16 and 1/4,
seed 0 only. This replaces the earlier unlaunched four-arm proposal above.
Question: does target amplitude change the amplification ranking under the
existing fixed per-scheme recipes? A ranking reversal is supporting exploratory
evidence; it is not proof that one-hot amplitude fully explains legacy's advantage.

Use scaled one-hot targets and native paired-output MSE without compensating
loss normalization. Copy the corrected seed-0 wide BPTT Conv2 configs from
`paper_ready_results/bundles/table1_wide_bptt/conv2/`, retaining their Adam rates,
30 epochs, T=K=6, input gain 100, exact-zero frozen biases, [0,100] conductance
bounds, and ordinary MNIST 55,000/5,000 split. Share one hash-verified Kaiming
initializer and loader seed 0 across all six runs. The official test stays unread.

Primary comparison: maximum validation accuracy over all 30 epochs, with final
accuracy also reported. Existing seed-0 target-1 bundles are historical references
only; no new target-1 runs, extra seeds, centering, rate tuning or EqProp runs.
Report all six outcomes, including nonfinite failures, without selecting on accuracy.
Compare rankings at the two new amplitudes and separately against historical
target 1; a tie is not a reversal. Report raw and amplitude-normalized MSE, but
only native MSE enters optimization. No uncertainty bars for one seed.

### Execution and monitoring handoff

- State: complete and reviewed September 28; all six production bundles collected
  and validated locally (see completion review below). Routine monitoring owner: the persistent run-watch
  service, PID **636685**, started with bounded incident/completion dispatch.
  Slurm array **278574**, tasks 0–5, submitted
  September 28 after all six local smoke runs completed successfully.
  All six tasks started on September 28 and reached epoch 2 with fresh artifacts;
  the preliminary October 3 queue estimate was superseded by actual startup.
- Configs: `configs/conv/target_scale_conv2_20260928/`, ordered baseline 1/16,
  baseline 1/4, ours 1/16, ours 1/4, legacy 1/16, legacy 1/4.
- Target: Jean Zay, `fmu@v100`, `gpu_p13`, `qos_gpu-t3`, `v100-16g`;
  one array `0-5%6`, one GPU and ten CPUs per task, three-hour task ceiling.
- Expected compute: approximately 11 GPU-hours; total ceiling 18 GPU-hours
  including retries. No scientific retuning during operational recovery.
  Monitoring deadline: October 5, 2026, 18:00 Europe/Paris (UTC+02:00).
  This replaces the provisional September 29 operational deadline because
  Slurm's preflight estimate puts the queued start on October 3; it does not
  increase the 18 GPU-hour compute ceiling or change the scientific cases.
- Local result root, including source, smoke, production collection and analysis:
  `results/conv2-target-scale-20260928-v1/`.
- Remote result root:
  `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/conv2-target-scale-20260928-v1/`.
- Frozen source will be staged under
  `/lustre/fswork/projects/rech/umg/ucy17uy/server_code/staged/conv2-target-scale-20260928-v1/`.
- Gate: focused loss/training tests, wrapper syntax, scheduler dry run, then
  a synchronous local CPU smoke through the exact runner on all six configs.
  Require valid completion artifacts and target-scale/initializer identity.
  The repository's current local-only gate supersedes the external skill's
  unconditional remote-canary requirement.
- Inspect: `squeue`/`sacct` for the recorded array and each run's status/epoch
  artifacts; require fresh semantic progress after startup. Normal cadence
  five minutes, stall threshold 45 minutes; queued tasks are not failures.
- Collect: rsync only this remote result root into its local counterpart;
  validate six complete 30-epoch bundles, common initialization/cohorts,
  finite metrics, frozen zero biases, unchanged rates and no official-test access.
- Review: generate JPG accuracy-versus-amplitude and learning-curve figures;
  retain distinct historical target-1 markers. Record coverage and interpretation
  here after local validation. Submission receipt: result-root
  `transport/launch.json`. Watch configuration: result-root `monitor/watch.json`.
  The read-only Slurm adapter passed 11 synthetic transport/artifact tests and
  one real observation covering all six active jobs, with no probe/runtime errors.
- Monitoring handoff verified: PID alive and fresh service observation at
  September 28, 17:32:08 Europe/Paris (15:32:08 UTC); next check due 17:37:08.
  All six tasks had completed epoch 1 and were training epoch 2, with no errors.
  Summed allocation time at that observation was 0.687 GPU-hours; approximately
  17.313 GPU-hours remain in the total ceiling. The task limits enforce 3 h each.
  The service checks every five minutes and dispatches a bounded owner for
  failures or final collection/validation/review; no second polling owner.
  Watch status command: `python /home/filip/.codex/skills/run-watch/scripts/run_watch.py status results/conv2-target-scale-20260928-v1/monitor/watch.json`.
  CLI authentication was verified; chat notifications are not configured.

Pre-submit evidence: 69 focused loss/training/exact-run/launcher tests passed.
A staging check found the launcher config omitted from the archive; corrected
in source-v2 before submission, with the launcher checks passing. The original
source archive remains preserved and was never launched. Source-v2 digest:
`3b2e0f82469b95ec986e10bae9365f5cd17abd858823aaf537373d4f895c260e`.
A broader unrelated EqProp test uses an incomplete noise-estimator fixture and
fails for missing `_endpoint_read_noise_mode`; this BPTT pilot does not use it.
The exact launch command and scheduler dry run are under the result root's
`transport/`. Slurm `--test-only` accepted the request; its October 3 estimated
start is a scheduling estimate, not a production launch receipt.

The six local smoke bundles also passed the collection validator:
`python -m experiments.analyze_conv2_target_scale --smoke` (6/6, no errors).
This checked matching initialization, split and loader order, both requested
target scales in resolved runtime configuration, inherited rates, exact-zero
biases in best/final checkpoints, finite metrics and zero official-test reads.
The smoke is implementation evidence only and will not be plotted as accuracy
evidence. Historical target-1 seed-0 best/final validation references are
97.36/97.36% baseline, 98.10/98.08% ours and 98.30/98.08% legacy.

Completion instructions for the monitoring owner:

1. Wait for all six array tasks to terminate successfully with complete matching
   30-epoch artifacts. Any failed task is an operational exception, not an
   accuracy-based exclusion. Preserve it and the remaining 18 GPU-hour budget.
2. Copy the exact remote result root into the same local result root with
   `rsync -a --partial --protect-args` using SSH alias `jean-zay`; never use
   `--delete`. Keep local source, smoke, transport and watch records.
3. From the worktree, run
   `/home/filip/miniconda3/envs/py312/bin/python -m experiments.analyze_conv2_target_scale`
   with the documented OpenMP environment if needed. Require `complete=true`,
   `validated=6`, no errors, and matching initializer and cohorts.
4. Inspect `analysis/accuracy_vs_target.jpg` and
   `analysis/validation_learning_curves.jpg`. Record the CSV/summary, final
   scheduler states and summed GPU time; update this note with the factual
   coverage and reviewed interpretation. No separate campaign scaffold/ledger.
5. State whether rankings reverse between the new target amplitudes, and
   separately whether they differ from historical target 1. Preserve the
   single-seed, fixed-per-scheme-rate and historical-reference limitations.
   The causal equivalence question and optimal targets remain outside this pilot.

### Completion and scientific review — September 28

Coverage is complete: all six seed-0 cells finished 30/30 epochs, with no failed,
excluded, replacement or retried production cells. A fresh `sacct -X` check
reported tasks `278574_0` through `278574_5` as `COMPLETED`, exit `0:0`, and
`squeue` returned no active tasks. Allocation wall times in task order were
6487, 6312, 6397, 6159, 6285 and 6420 seconds: **38,060 GPU-seconds = 10.5722
GPU-hours**, within the original 18 GPU-hour budget and each three-hour ceiling.
Hosts were r3i4n7, r10i2n7, r10i3n0, r10i5n8, r10i5n8 and r10i6n4 respectively.
The scheduler receipt is retained in the result root's `transport/completion.json`.

Collected only the declared remote result root into its existing local counterpart
using `rsync -a --partial --protect-args`, without deletion; transfer exited 0.
The declared py312 analysis command exited 0 with `complete=true`, `validated=6`
and no errors. It verified all six config identities, common initializer and
cohort/order provenance, 30-epoch finite artifacts, inherited rates, zero biases
in finite best/final checkpoints, and zero official-test evaluations. Both JPGs
were generated and visually inspected; historical target-1 markers remain distinct.

Evidence: [validation summary](../../results/conv2-target-scale-20260928-v1/analysis/summary.json),
[measurements CSV](../../results/conv2-target-scale-20260928-v1/analysis/accuracy.csv),
[accuracy versus amplitude JPG](../../results/conv2-target-scale-20260928-v1/analysis/accuracy_vs_target.jpg),
and [learning curves JPG](../../results/conv2-target-scale-20260928-v1/analysis/validation_learning_curves.jpg).

| Scheme | Target | Best validation % | Final validation % | Final native MSE | Final MSE / target² |
|---|---:|---:|---:|---:|---:|
| baseline | 1/16 | 98.28 | 98.10 | 0.000166458 | 0.0426133 |
| baseline | 1/4 | 97.98 | 97.90 | 0.00300224 | 0.0480358 |
| ours | 1/16 | 98.42 | 98.28 | 0.000162582 | 0.0416210 |
| ours | 1/4 | 98.36 | 98.24 | 0.00261701 | 0.0418721 |
| legacy | 1/16 | 98.38 | 98.22 | 0.000162611 | 0.0416285 |
| legacy | 1/4 | 98.40 | 98.34 | 0.00261062 | 0.0417700 |

Under the predeclared best-validation comparison, the ranking reverses between
ours and legacy: **ours > legacy > baseline at 1/16**, versus **legacy > ours >
baseline at 1/4**. The ours-minus-legacy gap changes from +0.04 to -0.04 percentage
points. These are strict differences, but each is only two correct predictions
on a 5,000-example validation set, with each best epoch selected separately.
Final accuracies show the same ranking reversal (+0.06 versus -0.10 points).
Baseline improves by 0.30 points in best validation when moving from 1/4 to
1/16, narrowing its gap to legacy from 0.42 to 0.10 points without overtaking it.

Separately, historical target 1 ranks legacy (98.30%) > ours (98.10%) > baseline
(97.36%). Thus target 1/16 reverses the historical ours/legacy order; target 1/4
retains it. These historical comparisons are references, not new matched runs.

Verdict: the observed reversal meets the exploratory decision rule and supports
target-amplitude sensitivity of the ranking under these fixed recipes. Its small
size and single seed do not establish a robust population ranking. Unnormalized
target changes alter optimization as well as the target; per-scheme rates remain
different. This does not prove that amplitude fully explains legacy's advantage,
establish functional/gradient equivalence, or identify an optimal target. All
accuracies are validation evidence, not official-test results. The authorized
pilot is finished; additional seeds or causal controls would require a new
scientific assignment, and none were launched.


## Trained encoding voltage replay — September 29

Filip clarified that the requested voltages are for the trained BPTT Conv2
1/4 and 1/16 encoding networks. Replay all six seed-0 final-epoch-30
checkpoints on the existing paired 576-example validation cohort, SHA-256
95af3eebd9416ec643b7d698d061b7dac3fe0d52311dd888293609bada629acd.
Keep native float32, asynchronous T=6, input gain 100, reset each batch,
original amplification and preprocessing; no training or official-test access.
Measure count-weighted RMS, signed mean, standard deviation, mean absolute
voltage, extrema and zero occupancy in Conv1/Conv2/readout. Compare rankings
at each encoding descriptively; no causal or significance claim from one seed.
Only final checkpoints are requested here; no epoch trajectory is inferred.

Execution and monitoring: one local CPU worker, root owns checks through review.
Budget: 180 seconds for six one-batch smoke cases, 600 seconds production;
expected a few minutes. Stop on identity, restore, nonfinite or budget failure.
All attempts live in `results/conv2-target-scale-voltage-20260929-v1/`.
Config: `configs/conv/conv2_target_scale_voltage_20260929.json`.
Runner: `python -m experiments.replay_conv2_target_scale_voltages --config
configs/conv/conv2_target_scale_voltage_20260929.json --output-root
results/conv2-target-scale-voltage-20260929-v1/production`.
Smoke uses the same command with `--smoke` and output `smoke/`.
Use the training study's frozen source runtime. Hash checkpoint/config bytes
before and after, require exact restored/unchanged parameters and zero biases,
then validate all bundles and plot RMS with a signed-mean companion.

Results and review: completed all six one-batch smoke cases and all six
production cases (576 examples each, 18 layer profiles). Parent and case bundles
validated, including exact restoration, unchanged checkpoint/config bytes and
parameters, native float32, finite states and zero biases. Production took
11.92 seconds on one CPU thread; no failed attempts or official-test reads.
Analyzer: `experiments/analyze_conv2_target_scale_voltages.py`.

| Encoding | Scheme | Conv1 RMS V | Conv2 RMS V | Readout RMS V |
|---|---|---:|---:|---:|
| 1/16 | baseline | 4.298558 | 0.130596 | 0.017830 |
| 1/16 | ours | 0.849857 | 0.046231 | 0.017277 |
| 1/16 | legacy | 1.635189 | 0.074384 | 0.017855 |
| 1/4 | baseline | 7.806969 | 0.425027 | 0.055000 |
| 1/4 | ours | 1.586160 | 0.159423 | 0.073745 |
| 1/4 | legacy | 2.394683 | 0.202369 | 0.087426 |

Both encodings rank hidden-layer RMS baseline > legacy > ours. Legacy has the
largest readout RMS at both amplitudes, but at 1/16 all three readouts are
close (0.01728–0.01786 V). Reducing the target lowers every measured layer
for every scheme; this is a trained operating-point comparison, not a
fixed-weight scaling identity or proof that voltage causes the accuracy ranking.
No additional training or checkpoint selection was performed.

[RMS plot](../../results/conv2-target-scale-voltage-20260929-v1/analysis/rms_trained.jpg),
[signed-mean plot](../../results/conv2-target-scale-voltage-20260929-v1/analysis/signed_mean_trained.jpg),
[full statistics and checkpoint paths](../../results/conv2-target-scale-voltage-20260929-v1/analysis/voltage_epoch_comparison.csv),
and [validation](../../results/conv2-target-scale-voltage-20260929-v1/analysis/validation.json).

## Conv3 target-amplitude extension — September 29

Filip authorized this extension, briefly paused submission, then explicitly
resumed it with batch16. Six fresh seed0 clean-BPTT Conv3 trainings cross
baseline(V/I=1/1), ours(4/1), legacy(4/0.25) with one-hot target amplitudes
1/16 and1/64. Every case runs30epochs at batch16 (validation64).
Use native paired-output MSE, without compensating normalization or LR scaling.
Copy each scheme's corrected wide Conv3 seed0 reference config and exact Adam
rates; retain float32, asynchronous T=K8, input gain360, perfect diodes with
explicit dictionaries, conductance bounds[0,100], and frozen-zero biases.
All cases share the existing hash-verified Conv3 Kaiming seed0 initializer,
fresh Adam, and ordinary-MNIST55k/5k split/shuffle seed0. Official test is disabled.

Primary: best validation accuracy over30epochs; secondary: final accuracy,
native MSE and MSE/target², runtime and charged GPU-hours. Retain all outcomes,
including failures. Compare schemes at each target; historical target1 remains
a separate reference. Single-seed ranking changes are exploratory, without
uncertainty/significance or causal-equivalence claims.

### Conv3 execution and monitoring handoff

- State: **complete and reviewed September30**. All six production bundles collected
  and validated for30epochs; final handles306577_0,2,3,4,5 and352044_1.
  Original301424 failure, canary306273 and cancelled306577_1 are preserved.
  Completion receipt: `transport/completion-probe-20260930.json` under the local
  result root. The existing Python watcher receives this terminal decision.
- Configs: `configs/conv/target_scale_conv3_20260929/`, array order baseline1/16,
  baseline1/64, ours1/16, ours1/64, legacy1/16, legacy1/64.
- Wrapper: `experiments/run_conv3_target_scale_jeanzay.sh`; exact_run receives
  all six configs and selects the Slurm task index once. Save full model,
  Adam and RNG checkpoints every epoch, with production epoch chunk30.
- Target: `umg@v100/gpu_p13/qos_gpu-t3/v100-32g`, array0-5%6, oneGPU,
  fourCPUs, five-hour limit per task, no explicit host-memory directive.
- Expected duration3.3–3.7h per case plus checkpoint overhead; approximately
  20–24GPUh total. Initial reservations30GPUh; total ceiling36GPUh including
  failures/recovery. Queue time additional; deadlineOctober6,18:00Europe/Paris.
- Local result root: [conv3-target-scale-20260929-v1](../../results/conv3-target-scale-20260929-v1/),
  including source, initializers, smoke, transport, monitor, collection and plots.
- Remote source: `/lustre/fswork/projects/rech/fmu/ucy17uy/server_code_pilots/conv3-target-scale-20260929-v1/source`.
- Remote results: `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/conv3-target-scale-20260929-v1/`.
- Gates: focused target1/64 loss/gradient and checkpoint/restore tests, frozen
  source/config checks, wrapper syntax, scheduler test-only, and six synchronous
  full-batch16 CPU smokes (3epochs×2batches, one validation batch). Smoke chunk1
  explicitly checks planned-pause restoration through the same exact runner.
  Repository local-smoke policy applies; remote canary only after a failure
  specific to that execution contract.
- Monitoring: five-minute polls,45-minute semantic-stall threshold; pending
  Slurm jobs are not failures. Record actual IDs/source identity and watcher
  acknowledgment here after submission. Do not modify unrelated jobs.
- Timeout recovery: establish old process terminal; restore latest complete
  model/Adam/private RNG checkpoint into a preserved new attempt toward30 total
  epochs, within remaining36GPUh. Require `exact_epoch_state_continuation`;
  missing full state blocks recovery. No fresh-optimizer or epoch0 restart.
- Collection: rsync only the declared remote root without deletion, run
  `python -m experiments.analyze_conv3_target_scale --study-root results/conv3-target-scale-20260929-v1`,
  validate all six terminal outcomes/checkpoints/cohort identities and zero
  official-test reads, inspect JPGs and record coverage/accounting/review here.
  No historical dashboard updates or new campaign scaffold.

### Conv3 pre-submit gates passed

Frozen source archive SHA256 `2b7139d751d5681c42cca721a32db7cd838fac1dcb28bd7fe8c8068f961165c7`; all114 staged files verified remotely. Sixty focused tests passed on the frozen source,11 collector tests and17 monitor tests passed. All six true-batch16 CPU smokes completed3epochs×2batches with two model/Adam/RNG pause-restores each; all six reporting/scientific artifact validations passed. No official-test reads. Config checks preserve every scheme's model, optimizer and learning-rate vector. Wrapper syntax, all six array indices and out-of-range rejection passed. Exact `sbatch --test-only` accepted fmu/V10032/5h and currently forecasts October3 at22:04Paris; this is not a submitted job. Gate receipts and exact command are under the result root `transport/`.

Conv3 submission receipt: `results/conv3-target-scale-20260929-v1/transport/launch.json`. Array301424 was submitted once with the frozen command after all gates passed; actual scheduler state and monitoring acknowledgment follow.

### Conv3 verified monitoring handoff

Live `scontrol`/`squeue` confirmed one array301424, six unique tasks, concurrency6, fmu@v100/gpu_p13/qos_gpu-t3/v100-32g,1GPU4CPUs and5h per task. All six were pending for priority; compute charge0GPUh at handoff. No runtime artifacts are expected before allocation. Persistent watcherPID1164217 is verified alive outside the sandbox process namespace, with dispatch enabled and a fresh all-six observation at 2026-09-29T08:44:59.520433+00:00. Next check 2026-09-29T08:49:59.520433+00:00. No probe/log errors; no competing poller. The existing run-watch service owns startup artifact checks, bounded recovery and collection/review. Receipt: result-root `monitor/monitoring-handoff.json`; inspect with `python /home/filip/.codex/skills/run-watch/scripts/run_watch.py status results/conv3-target-scale-20260929-v1/monitor/watch.json`. Controller must stay awake/connected; external notifications are not configured. Training completion remains pending.

### Conv3 startup incident — September 29, bounded decision

Array301424 tasks0–5 are all `FAILED`, exit `1:0`; a fresh scheduler
inspection found no queued/running tasks. Allocation wall times were
24,19,19,19,19,24 seconds, totaling **124 GPU-seconds**. Remaining total
budget is **129,476 seconds (35.9656 GPU-hours)**. Keep the original handles
tracked and `prior_gpu_seconds=0`: these allocations remain counted by the
adapter, so adding them to prior time now would double-count. Before any
replacement handles are installed, preserve these124seconds as prior charge.

No training bundles, task summaries or checkpoints were published; the remote
root contains only logs. Task0 stdout is empty and stderr contains module-load
text only. All six JSON configs exist at the frozen source path. The leading
operational diagnosis is Slurm script relocation: the wrapper computes its
source root from `BASH_SOURCE`, which can refer to the spool copy, then silently
exits on the six-config-count assertion. This is an inference, not a confirmed
runtime trace. Scheduler WorkDir is the remote home directory.

Decision: **blocked for automatic retry**, with passive monitoring retained.
No jobs submitted, source/configs changed, handles replaced or results removed.
Recovery needs a Codifier to reproduce the spool-path failure, repair explicit
frozen-source resolution, add a regression check and validate the exact
launcher through a live target canary before releasing production. This
target-specific startup failure reinstates that canary gate. Preserve the
original scientific contract, failed logs and total budget; no scientific
conclusion or completed coverage is available. Incident accounting receipt:
`results/conv3-target-scale-20260929-v1/transport/incident-301424.json`.

### Conv3 monitoring adapter repair — September 29

The controller purged array301424 from `squeue`, which returned `Invalid job
id specified`; the adapter previously aborted before reading accounting.
Repaired only that exact empty-output error to fall through to `sacct`; other
transport errors still fail closed. All19 adapter tests pass, including the
expired-controller regression and rejection of unrelated transport failures.
One live verification reports all six tasks failed, no active worker, no
completion evidence and **124 GPU-seconds** charged, without probe errors.
Receipt: `results/conv3-target-scale-20260929-v1/transport/probe-repair-301424.json`.
Original handles and prior charge0 remain unchanged. Training recovery remains
blocked on the previously recorded launcher repair/regression and live target
canary; no submission or scientific/source-setting change was made.

### Conv3 bounded incident recheck — 2026-09-29T09:10:53.209047+00:00

Fresh remote accounting again confirms all six allocations failed with exit1:0,
124GPU-seconds charged and129,476seconds remaining. No study job appears in
the live user queue. The frozen remote wrapper still derives SOURCE_ROOT from
BASH_SOURCE; the previously required launcher repair/regression and live target
canary evidence remain absent. Decision remains **blocked**; no submission,
source change or handle replacement. Original handles retain the124seconds
charge, with prior_gpu_seconds0 to avoid double-counting. Unrelated jobs were
left untouched. Receipt: `results/conv3-target-scale-20260929-v1/transport/incident-301424-recheck.json`.

### Conv3 launcher recovery — September29

Root took ownership of the startup repair after Filip requested an update. All six original tasks are terminal with no training artifacts;124GPU-seconds remain charged. The launcher now requires an explicit absolute source-root argument because Slurm relocates the script. Numerical source/configs/initializers are unchanged. Regression tests execute a copied spool wrapper against a separate source tree and check task0/task5, smoke routing and missing-root rejection.

Before replacing production, submit one live singleton0-0 canary named `conv3-target-canary-v2`, fmu@v100/gpu_p13/qos_gpu-t3/v100-32g,1GPU4CPU,15minutes, same frozen training source and repaired wrapper. Canary uses baseline target1/16,batch16,3epochs×2batches,official test disabled. Expected remote output: `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/conv3-target-scale-20260929-v1/canary-v2/`. Wrapper is staged separately under the existing WORK study's `launch-v2/run.slurm`; source archive identity remains unchanged. Require terminal resource/accounting and semantic checkpoint validation before production.

Planned replacement name `conv3-target-scale-v2`, array0-5%6,same fmu/V10032/5h resources and six frozen configs. Expected output: `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/conv3-target-scale-20260929-v1/production-v2/`. Preserve original logs/receipts. These are previously unstarted trainings, not discarded epoch0 restarts. Total36GPUh and October6deadline unchanged, including original and canary charges. Root owns canary checks; existing blocked watcher remains passive on original handles until replacement handoff.

Recovery regression gate passed34tests including spool relocation; repaired wrapper SHA256 is recorded in `transport/v2/preparation.json`. Live canary **306273_0** submitted once with same fmu/V10032/module/source/config contract, output `canary-v2/`. Root owns its terminal accounting and semantic validation; six production replacements remain gated.

Canary306273 completed before its12:40forecast, in72seconds. Aggregate read-only observation at2026-09-29T09:54:56.828815+00:00 confirms3complete epochs, valid basic artifacts and no log/probe errors. Detailed scientific validation and production release are assigned to bounded completion agent1204118 under sole canary watcher1204100; PID verified alive, dispatch enabled. Original watcher1164217 stopped without signalling any training. Prepared actions are `transport/v2/collect_canary.py`, `release_production.py` (requires passed gate), and `handoff_production.py`; replacement output remains `production-v2/`. The latter preserves incident/model-call history and accounts124+72=196GPU-seconds once before restarting the original production watcher. Canary state/ownership receipt: `monitor/canary-monitoring-handoff.json`. No replacement array is claimed until its receipt exists.

### Conv3 replacement production submitted

Live canary306273 passed terminal resource/source/semantic validation. Six unchanged batch16/30epoch runs submitted as array**306577**, using repaired explicit-source wrapper and `production-v2/` outputs. Original failure and canary charge total196GPU-seconds carried as prior usage;36GPUh budget/deadline unchanged. Persistent production watcherPID1206608 restarted with fresh observation and bounded dispatch; old incident history retained. Local `runs/` links `production-v2/runs` for the existing analyzer. Receipt `monitor/recovery-handoff.json`.

Verified replacement handoff: first all-six observation at **2026-09-29T09:57:12.842016+00:00** reports tasks `306577_0`–`306577_5` pending, no probe/log errors, and 196 GPU-seconds charged (124 original + 72 canary), counted once. Live scheduler verification confirms one parent, array `0-5%6`, `fmu@v100/gpu_p13/qos_gpu-t3/v100-32g`, one GPU and four CPUs per task, five-hour limits, no requeue; hosts/GPUs remain unassigned while pending. Remaining total allowance is 129,404 GPU-seconds. Production watcher PID **1206608** is alive and is the sole owner of startup, progress, recovery and collection; its fresh state has no blocked attention or terminal latch. Existing dispatch history is preserved. Training is queued, not complete.

Canary **306273_0 only** is complete: strict gate validated three epochs, four checkpoint pairs, 114 frozen source files, resource identity and zero official-test evaluations. Receipts under `results/conv3-target-scale-20260929-v1/`: `transport/v2/canary-gate.json`, `transport/v2/production-launch.json`, `transport/v2/production-first-observation.json`, `transport/v2/production-scheduler-verification.json`, and `monitor/recovery-handoff.json`. The prepared handoff initially failed before mutations because its dynamic watcher import could not locate `probe`; adding the watcher script directory to the local import path fixed it. Syntax compilation, watch validation and the real six-task observation passed afterward. No numerical source, wrapper, scientific settings, budget or deadline changed.

### Timed alternate-account fallback — September29

User authorizes moving still-unstarted Conv3 cases to `umg@v100` at/after **September29 16:20 Paris** (approximately four hours from request). Active umg V100 association verified live. Preserve running/completed fmu cases; withdraw only pending cases and establish terminal nonexecution before any replacements. Same six configs, V10032 resources,36GPUh aggregate budget and October6 deadline. The existing sole watcher now uses `monitor/watch_with_fallback.py`, a study-local extension that dispatches an account-fallback incident on the first five-minute check after this deadline. Operational instructions are in existing watch authorization; account-fallback.json records timing. No jobs moved at setup. Controller must remain awake and connected.

### Immediate umg switch — September29

User superseded the four-hour wait and requested umg immediately. All six PENDING tasks in array306577 were changed **in place** with `scontrol update JobId=306577 Account=umg@v100`; live scheduler verified all six on umg. No new jobs or cancellations; original source, outputs, indices, resources and aggregate budget unchanged. Timed fallback disabled. Monitor now validates explicit `run.account`;39 focused adapter/fallback tests passed. Receipt: `results/conv3-target-scale-20260929-v1/monitor/umg-switch.json`.

### Conv3 node prolog incident — September30, bounded decision

Fresh `sacct`, `squeue` and `scontrol` identify task **306577_1**
(raw job336431, baseline target1/64) as `CANCELLED by 0`, reason `Prolog`.
Node **r7i6n4** is `MIXED+DRAIN` with an explicit prolog error.
No task stdout/stderr files or matching training bundle exist; the failure
occurred before observable training. The five other tasks remain RUNNING
on umg@v100 and were left untouched. No scientific failure is inferred.

The cancelled allocation consumed **182 GPU-seconds**. All original handles
remain tracked, so prior_gpu_seconds stays196 to avoid double-counting.
At the recorded observation total charged time was2661seconds, leaving
126939seconds of the original36GPUh ceiling; running tasks continue charging.
No jobs submitted, cancelled or requeued and no config/source changes made.
Decision: **blocked for this incident**, with passive monitoring retained.
A replacement requires gated target recovery and validated per-case scheduler
mapping/accounting, preserving the other five live tasks and all six identities.
The current adapter assumes one parent array; do not replace the parent handle
or blindly resubmit all six cases. Cluster prolog repair itself is outside this
agent's authority. No replacement gate or recovery is claimed complete.
Receipt: `results/conv3-target-scale-20260929-v1/transport/incident-306577-1-prolog.json`.

### Baseline target1/64 retry — September30

User explicitly requests scheduling the missing baseline. Submit only array index1 on umg@v100, V10032, batch16,30epochs, unchanged frozen wrapper/source/config, five-hour limit (~3.5h expected). Output remains `production-v2/runs/001_01_conv3_baseline_target_1over64_seed0_6e4d71a0`; no original training artifacts exist. Preserve cancelled-task logs/accounting. Five sibling productions completed on the same account/wrapper; this successful live execution establishes target execution health after the isolated node prolog failure, without repeating a canary for unchanged code.43 focused wrapper/monitor tests passed, including replacement mapping and exact-once charge accounting. Before retry verify frozen source/wrapper and scheduler test-only. Existing study ceiling36GPUh unchanged; prior charges196 plus cancelled182, completed tasks61508seconds. Retry receipt under `transport/baseline-retry/`; one existing watcher owns the aggregate, replacing only index1 mapping.

Baseline retry submitted once as **352044_1** on umg@v100. Original cancelled case charge182seconds moved into prior usage378seconds; five completed sibling charges remain live-accounted. Watch retains six-case coverage with explicit index1 replacement. Source/wrapper hashes and scheduler test-only passed; no science changes.

### Conv3 completed coverage and scientific review — September30

Collected the entire remote study with rsync without deletion; preserved the local
`runs -> production-v2/runs` symlink and all failed/canary/replacement evidence.
The declared analyzer exited0: **6/6 validated**, no errors,30epochs each,
31 model/Adam checkpoint pairs per case including initialization, matched
initializer and cohort/order identities, unchanged scientific contracts, finite
checkpoint tensors and frozen-zero biases. Official-test evaluations remain0.
Both generated JPGs were visually inspected. Fresh remote adapter validation
confirms all six tracked allocations COMPLETED with exit0:0 and complete matching
artifacts; no probe or log errors. The original timeout did not imply worker failure.

Charge is **73,692 GPU-seconds (20.47 GPU-hours)**:61,508 for five siblings,
11,806 for352044_1 and378 prior seconds (124 original failure +72 canary
+182 cancelled case). Prior usage remains378, counted exactly once; total is
within36GPUh and the October6 deadline. No new submission or scientific change.

| Scheme | Target | Best validation % | Best epoch | Final validation % | Final native MSE | MSE/target² |
|---|---|---:|---:|---:|---:|---:|
| Baseline | 1/16 | 98.78 | 30 | 98.78 | 9.84237e-5 | 0.0251965 |
| Baseline | 1/64 | 98.98 | 28 | 98.94 | 5.31583e-6 | 0.0217736 |
| Ours | 1/16 | 98.94 | 23 | 98.90 | 8.58986e-5 | 0.0219900 |
| Ours | 1/64 | 99.00 | 23 | 98.98 | 5.49105e-6 | 0.0224913 |
| Legacy | 1/16 | 99.16 | 28 | 99.06 | 7.26170e-5 | 0.0185900 |
| Legacy | 1/64 | 99.12 | 24 | 99.08 | 5.13274e-6 | 0.0210237 |

**Verdict:** no ranking reversal in best validation accuracy: legacy > ours >
baseline at both new amplitudes, also matching the historical target1 ranking.
The exploratory reversal criterion is not met for Conv3. Reducing target from
1/16 to1/64 narrows legacy's lead over baseline from0.38 to0.14 percentage
points and over ours from0.22 to0.12 points; baseline improves0.20 points,
ours0.06, while legacy falls0.04. Final accuracies retain the same ordering.
This supports target-amplitude sensitivity under the fixed per-scheme recipes,
but does not establish that amplitude explains legacy's advantage or prove
functional/gradient equivalence. Single seed, selected validation maxima and
inherited distinct rates preclude significance or causal claims. Historical
seed0 target1 values (97.80/98.66/98.84%) remain separate reference evidence.
Normalized MSE is reporting only; no loss normalization was applied in training.
No additional training is authorized or needed to close this six-case pilot.

Evidence: [validation summary](../../results/conv3-target-scale-20260929-v1/analysis/summary.json),
[metrics CSV](../../results/conv3-target-scale-20260929-v1/analysis/accuracy.csv),
[accuracy JPG](../../results/conv3-target-scale-20260929-v1/analysis/accuracy_vs_target.jpg),
[learning curves JPG](../../results/conv3-target-scale-20260929-v1/analysis/validation_learning_curves.jpg),
and [terminal scheduler/artifact receipt](../../results/conv3-target-scale-20260929-v1/transport/completion-probe-20260930.json).
