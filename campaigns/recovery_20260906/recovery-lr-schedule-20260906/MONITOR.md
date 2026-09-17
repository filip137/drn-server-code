# Longer low-rate and decaying recovery, September 6

**Numerical work complete.** Final coverage is 25 screening arms, 16 frozen-schedule confirmation runs, and 14 supplementary one-epoch references: 55 runs and 1,244 full epochs. All 60 paired test cases passed zero-write and unchanged-state checks. All current launchers exited 0, and the local supervisors were confirmed absent at 16:10:15 UTC. No experiment or readout remains active; the handles below are historical and no further watchdog polling is needed.

The final [results report](analysis/RECOVERY_RESULTS.md), [test table and per-array data](analysis/paired_test_readouts.csv), [test plot](analysis/paired_test_comparison.png), and [completion receipt](analysis/completion_receipt.json) contain the numerical closeout. All three native crossbar studies are `ready_for_review=true`. Their managed scientific finalization remains pending the user's interpretation and next tests; no review or managed manifest conclusion has been invented.

## Original launch protocol and operational history

Exploratory/noncanonical. The original question and coverage are in [plan.json](plan.json); the later single-arm DRN amendment is in [plan.with_lr_boundary.json](plan.with_lr_boundary.json).

- DRN: local RTX 3090; three-worker screen tmux `om_recovery_drn_parallel_20260906`; active worker PIDs in `drn_screen/status.json`; worker progress in `drn_screen/runs/<arm>/status.json`; logs in `drn_screen/logs/`; launcher log/exit in `launchers/drn_parallel.*`. The original serial tmux/launcher is retained as handoff history.
- Crossbar: Akib RTX 3080; existing SSH launcher, name `om_recovery_crossbar_20260906`; remote root `/home/filiposana/staged/recovery_lr_schedule_20260906`; launcher PID/log/exit in `crossbar_screen.launcher/`; supervisor progress in `crossbar_screen/status.json`; native metrics/status/results in `crossbar_screen/results/mnist-ibm-om-crossbar-long-kl-schedules-20260906-v1/runs/`.
- Each original screen contained 12 arms: four constant rates and two exponential schedules, healthy and faulted. A later declared DRN lower-rate probe brought the final combined count to 25. All screening arms received 30 full 55,000-example epochs, batch 16. Exponential schedules reached 1% of their initial rate at epoch 10 and held it through epoch 30.
- Checkpoint selection minimizes apparent validation KL, including P0; accuracy breaks ties. DRN additionally evaluates at quarter-epoch intervals in the first epoch. Source hashes are in `source_receipt.json`.
- Expected supervisor heartbeats: 15 seconds. DRN worker heartbeat: every 500 minibatches (normally several seconds); native crossbar metrics: once per full epoch. Initial estimate: several hours for screening; refine from actual epoch durations.
- During execution, the originating Codex task inspected process/GPU/metric/log progress at least every 30 minutes, with shorter artifact checks between full watchdog checks. Shell supervisors tracked progress and stopped on failures; they did not diagnose or repair failures.
- Operational recovery may restart an understood failed arm in a new directory. Preserve completed trajectories and original P0. Never change scientific settings while repairing an operational fault.
- Initial local nohup launch was interrupted before its worker produced any native artifact; retained in `drn_screen_interrupted_launch_01/`. Replaced by tmux.
- After screening, freeze best constant and decaying schedules per architecture/condition and check two additional saved arrays. Extend qualifying boundary winners to 60 epochs as declared in the plan. Screening alone is not the full intended completion.

## Validation

Both hosts passed actual CUDA allocation/kernel/synchronization checks. The DRN two-minibatch/two-epoch canary passed with P0 selection and selected-state replay. The crossbar two-full-epoch decaying canary passed; its epoch-1 KL exactly reproduced the old LR=3e-6 corrupted trajectory. Resuming from epoch 1 exactly reproduced epoch-2 metrics and apparent/persistent/selected state hashes.

Two focused CUDA tests passed: explicit schedule validation/KL ordering, and reduced pulse exposure with exact optimizer-state replay. The DRN control also exactly reproduced its previously saved epoch-1 validation KL.

Confirmation adds the previous crossbar one-epoch schedules as paired references on both additional arrays; this supplementary control is declared before those arrays are evaluated under any new schedule. All confirmation readouts remain validation-only for both architectures.

## Common test replay

`finish_readouts.py` waits for complete confirmation, freezes the best within-one-epoch reference from the original validation screen, runs any missing reference cases, and only then replays the common official 10,000-example test set. It compares P0, the historical one-epoch rate, the tuned within-one-epoch reference, and the frozen best constant/decaying schedules on all three arrays. Read-only validation canaries reproduced the DRN and crossbar selected-state KL exactly and preserved state hashes.

Local readout launcher: tmux `om_recovery_drn_readouts_20260906`, wrapper/log/exit in `launchers/drn_readouts*`, progress in `drn_readouts/status.json`. Remote readout launcher: `om_recovery_crossbar_readouts_20260906`, sidecars in `crossbar_readouts.launcher/`, progress in `crossbar_readouts/status.json`. These waiters allocate no GPU while screening and confirmation are active.

## DRN concurrency handoff

The sole DRN worker used approximately 37% GPU utilization and 468 MiB. Two simultaneous tiny CUDA repeats, alongside that worker, exactly reproduced the original serial canary's initial/final/candidate evaluations and pulse counts (`canaries/drn-concurrency-01/status.json`). The numerical runner is unchanged. Remaining screen and confirmation commands therefore use up to three independent subprocesses on the local GPU.

The original screen supervisor was paused while `healthy-constant-3e-6` finished its full 30 epochs. `handoff_drn.py` verifies its terminal result and the worker's exit before replacing only the serial supervisor. The intentional supervisor termination and all original artifacts are retained in `drn_screen/parallel_handoff_request.json`, `serial_supervisor_at_handoff.json`, and the original launcher logs. This is a scheduling change, not an interrupted or restarted numerical arm.

The empty original confirmation waiter is archived in `drn_confirmation_serial_waiter_01/`; its replacement tmux is `om_recovery_drn_confirmation_parallel_20260906`, with log/exit in `launchers/drn_confirmation_parallel.*`. It waits on the same complete screen coverage and applies the unchanged selection and extension rules.

Automatic approval review initially rejected staging final readout code on Akib as an unverified destination. Read-only host/ownership/source-hash checks established that the exact repository was already staged for this experiment; the reviewed retry was approved. The readout code is staged and its waiter is live; no access block remains.

## DRN portability check (excluded from the sweep)

To explore using the second GPU after the crossbars, the original DRN source and eight exact input artifacts were staged on Akib and their SHA256 hashes verified. A two-minibatch/two-epoch canary started from the same state but did not reproduce the local pulse stream: total pulses were 316 versus 322. Initial held-apparent KL differed by only 6.3e-9. A direct seed-108101 CUDA draw check confirmed that 1,000-element draws match while 813,056-element draws differ across the two host environments (RTX 3090/82 SMs and RTX 3080/68 SMs, both PyTorch 2.5.1/CUDA 12.1).

No full DRN run was moved to Akib. The declared DRN comparisons and confirmations retain the local host and matched stochastic stream. `canaries/drn_akib_result.json`, `canaries/drn_cross_host_parity.json`, the staged input receipt, and the source archive preserve this excluded portability diagnostic. The crossbar pipeline remains on Akib throughout.

## Crossbar terminal evidence (12:48 UTC)

All three Akib launchers exited 0. Completed coverage is 12 screening arms, 8 frozen-schedule confirmation runs, 4 previous-rate confirmation references, and 8 supplementary one-epoch references. All 32 runs and 216 recorded artifacts were collected locally and verified by `audit_completed.py crossbar`; first-epoch state hashes match across paired rate/horizon references, data-order hashes match, and checkpoint selection minimizes validation KL with P0 eligible. The largest observed per-cell update count was 21, below the cap of 640.

All 30 frozen common-test readout cases passed zero-write and unchanged-state checks. `analysis/paired_test_readouts.csv`, `analysis/TEST_COMPARISON.md`, and `analysis/paired_test_comparison.{png,pdf,svg}` currently contain the completed crossbar comparison; the combined files will add DRNs after their readouts complete.

The three native studies report `ready_for_review=true`; their exact plans/configs and verified summaries are collected. Managed scientific finalization awaits the user's interpretation and next tests, as required by the EBL closeout skill. No managed review or manifest entry has been invented. DRN screening, confirmation, and readout monitoring remain active.

## Lower-rate boundary probe (13:44 UTC)

The completed constant-rate screen selected faulted DRN LR=3e-6 at epoch 8, the lowest rate in the original DRN grid. Before any DRN confirmation or test readout, `plan.with_lr_boundary.json` declares one additional 30-epoch faulted DRN run at LR=1e-6. The original `plan.json`, its twelve DRN arms, and the numerical runner remain unchanged. The combined screen now expects 13 DRN and 12 crossbar arms. `lr_boundary_source_receipt.json` records the amended plan and driver hashes.

`run_lr_boundary.py` waits until the original screen has no queued cases and fewer than three active workers, then runs the same scientific command on array 1. Its tmux session is `om_recovery_drn_lr_boundary_20260906`, parent PID 3309028; log/exit files are `launchers/drn_lr_boundary.*`, numerical log is `drn_lr_boundary/run.log`, and status is `drn_lr_boundary/status.json`. The new result is saved alongside the original screen in `drn_screen/runs/faulted-constant-1e-6/`.

The two empty dependent waiters were verified to have no numerical child or completed case before their parents were intentionally terminated. Their statuses are preserved in `drn_confirmation_waiter_before_lr_boundary/` and `drn_readouts_waiter_before_lr_boundary/`. Replacements are tmux `om_recovery_drn_confirmation_boundary_20260906` (PID 3309031) and `om_recovery_drn_readouts_boundary_20260906` (PID 3309033), with corresponding `launchers/*_boundary.*` logs/exit files. Confirmation requires both the original screen and supplementary probe to complete and all thirteen results to exist before freezing choices. The original late-improvement extension criterion and two-additional-array confirmation remain unchanged.

At the handoff, all three original numerical workers were live and progressing; GPU utilization was 98%, memory 2191 MiB, temperature 70 C. No numerical worker was stopped. The next full process/GPU/artifact watchdog check is due by 14:14 UTC, with shorter semantic progress checks in between.

## Original DRN screen complete (14:03 UTC)

The original twelve DRN arms completed all 30 epochs; `launchers/drn_parallel.exit` is 0. `audit_completed.py drn` passed for all twelve, including paired first-epoch candidate equality, declared learning rates, full epoch coverage, minimum-KL selection, and selected-state replay. Maximum observed cell update count was 131, below the cap of 640. The added LR=1e-6 arm is active and confirmation reports `waiting_for_additional_screen`; no DRN schedule or common-test selection has yet been frozen. The complete combined screen still requires its thirteenth result.

## Full DRN screen and confirmation launch (14:14 UTC)

The supplementary LR=1e-6 arm completed 30 epochs with selected-state replay passing and launcher exit 0. Its best validation KL was 0.0226575 at epoch 19; final KL was 0.0232993. It did not beat the original LR=3e-6 winner (0.0221234 at epoch 8). All 13 DRN screening runs passed artifact/schedule/selection checks. The combined screen now contains all 25 declared arms.

`drn_confirmation/frozen_selection.json` freezes healthy LR=3e-5 for both constant and exponential schedules, faulted constant LR=3e-6, and faulted exponential LR=1e-5. None of these winners meets the predeclared late-improvement criterion for a 60-epoch extension. All eight additional-array cases therefore run 30 epochs. Healthy constant/decay have an identical first epoch on a given array; the development selection favors the same 0.75-epoch state for both.

The first three confirmation workers (PIDs 3317198, 3317199, 3317200 at launch) produced matching paired first-epoch pulse counts and fresh minibatch artifacts. The supervisor remains PID 3309031; the readout waiter remains PID 3309033. GPU utilization was 85% during startup, memory 1974 MiB, temperature 62 C. Next full process/GPU/artifact watchdog check is due by 14:44 UTC; shorter artifact checks continue throughout.

## Confirmation progress (14:55 UTC)

Three of eight DRN confirmation cases have completed 30 epochs and selected-state replay. Healthy constant LR=3e-5 selected epoch 0.7502545 on array 2 (validation KL 0.0135528; final 0.111003) and epoch 1 on array 3 (0.0131237; final 0.110004). Healthy decay on array 2 selected the same 0.7502545 state (0.0135528; final 0.0237759). The two faulted constant LR=3e-6 runs have started, alongside healthy decay on array 3; all have fresh minibatch artifacts.

The scheduled 14:43:53 UTC watchdog check passed: GPU utilization 97%, 2191 MiB, 70 C, 706 GiB free disk, and all worker artifacts less than six seconds old. Its structured receipt is in `watchdog_observations.jsonl`; the next full check is due at 15:13:53 UTC. No numerical failure or retry has occurred.

## Six confirmation runs complete (15:36 UTC)

All four healthy DRN confirmation runs and both faulted constant-rate runs completed all 30 epochs with selected-state replay passing. Healthy decay on array 3 selected epoch 1 (KL 0.0131237; final 0.0196948). Faulted constant LR=3e-6 selected epoch 8 on array 2 (KL 0.0228836; final 0.0362689) and epoch 6 on array 3 (KL 0.0208461; final 0.0304514).

Only faulted decay LR=1e-5 remains: active workers are 3333779 (array 2) and 3335439 (array 3), with no queued confirmation case. The first run currently favors epoch 3 (validation KL 0.0217754), better than that array's frozen constant-rate checkpoint. This is a validation observation; the common DRN test cohort remains unopened for the new choices.

The 15:13:06 UTC full watchdog check passed with 99% GPU utilization, 2228 MiB, 70 C, and fresh artifacts. Its receipt and next full check at 15:43:06 UTC are recorded in `watchdog_observations.jsonl`.

## Final numerical closeout (16:10 UTC)

The last faulted decay run completed 30 epochs and selected epoch 24 on array 3 (validation KL 0.0198276; final 0.0198929). Array 2 selected epoch 3 (0.0217754; final 0.0234758). Both supplementary DRN short references then completed from their exact saved deployments. All 23 DRN runs passed the final artifact/epoch/schedule/selection audit; all 32 crossbar runs and 216 native artifacts had already passed their audit. No probability clipping or pulse-cap exhaustion occurred; maximum per-cell counts were 131 for DRNs and 21 for crossbars, with cap 640.

All 30 DRN and 30 crossbar paired test cases completed after validation-based choices were frozen. Identical DRN short-reference states reused three readouts, giving 57 unique zero-write test executions. Historical DRN readouts reproduced the original saved test predictions and KL. `verify_completion.py` passed declared screen/confirmation/reference/readout coverage and native review-readiness checks; its final receipt includes the source hashes used for analysis.

Mean faulted-DRN test KL was 0.0252110 for the tuned short reference, 0.0252716 for longer constant recovery, and 0.0247030 for decay. The constant comparison was mixed by array; decay improved both, by 2.0% in the mean. Faulted-crossbar means were 0.0397202, 0.0372834, and 0.0384001 respectively; a second constant-rate epoch improved both arrays, by 6.1% in the mean. Healthy DRNs selected identical early states for all tuned comparisons; healthy-crossbar differences beyond the tuned short reference were small and mixed. The report records accuracy changes and all protocol limitations.

The final local launcher exit files (`drn_parallel`, `drn_lr_boundary`, `drn_confirmation_boundary`, and `drn_readouts_boundary`) all contain 0. Remote launchers had already exited 0 and their outputs were collected. The 15:43:09 UTC watchdog check passed; the terminal process check at 16:10:15 UTC supersedes its next-check reminder. No active numerical work remains. The only pending step is user scientific review before native EBL managed finalization, as documented in the results report and closeout skill.
