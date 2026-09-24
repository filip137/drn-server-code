# Baseline EqProp read-noise launch plan

Prepared 2026-09-16. Status: **complete, collected and validated** on September 18.
The original prelaunch plan is preserved in the result root as `plan.frozen.md`.

All22 trainings finish by00:39:15 CEST on September18. All source and
collected bundles pass full revalidation, and remote production files match
local checksums. Every production queue exits0; Fifi's tmux session closes
and its GPU is idle. Final usage is43.294845/60 physical GPU-hours, within
the48h deadline. No full training failed, was excluded or replaced.
[Completed results](../paper_ready_results/baseline_read_noise_results_20260916.md) ·
[Coverage and budget proof](../paper_ready_results/provenance/baseline_read_noise_closeout_20260918.json).

## Execution record and subsequent concurrency instruction

Filip prioritized Fifi, then explicitly selected **two concurrent Conv3
workers**, superseding the throughput-based single-worker choice below.
Fifi production began at **15:47:43 CEST, September 16**, in tmux session
`baseline-noise-production-v2-20260916`, launcher pane PID935407. Its three
pairs are clean + sigma1e-5, sigma3e-5 + sigma1e-4, and sigma3e-4 + sigma5e-4.
Two real CUDA workers and canonical running bundles are observed; GPU use is
about10.3GiB and99%. The preliminary concurrent timing predicts about11.0h
per pair (roughly33h for the whole Conv3 queue); actual production progress
will update that estimate. Single-worker timing was about4.2h per run, so
two workers are a user-directed concurrency choice, not a measured speedup.

All22 local exact-config semantic smokes pass. The six Fifi smokes and its
one/two-worker timings pass. Conv1/Conv2 beta100 historical clean controls
pass canonical, full-scientific-config, shared-initializer and dataset-setting
reuse checks; environment differences remain labeled. Local Conv2 production
is in `main`/tmux window `@12`, pane PID1294573; same-runner timing predicts
1.859h per run. Akib/Nom checks and their dependent serial production queues
are started using the SSH helper's nohup fallback, with launcher PIDs197482
and1753220 respectively. Their measured admissions are recorded on each host.

Budget allocations remain within60 physical GPU-hours: Fifi40h, local14h,
Akib2.5h, Nom2.5h, plus1h for checks/recovery. Each host has a finite queue
cutoff and each pack a finite case limit. Fifi's pair limit is48,000s, with
46,800s admitted expected duration including margin; local uses9,600/9,000s.
The overall48h wall deadline is September18,15:47:43 CEST.

The first Fifi production wrapper failed a quoting/syntax check before
training or output creation. Its script/log/exit receipt are preserved.
The replacement `fifi_queue_v2.sh` passes embedded-Python compilation and
`bash -n`; science and frozen source/config bytes are unchanged. The source
identity is `61498ceba4f3b06e83cbd410a285ea1e05060f1fb77177977b54b1d15c81a4df`,
derived from the prior accepted read-noise `source-v2` trainer.

At 16:34 CEST, both Conv1 queues have finished: all 11 trainings are
collected and revalidated, and Akib/Nom production files match local copies
by checksum. Their launcher exits are zero (16:29:34 / 16:33:26 CEST).
Local Conv2 and Fifi's two Conv3 workers continue; 11 full trainings remain
running or queued. All 39 semantic smokes and six timing runs validate.
See the [partial results](../paper_ready_results/baseline_read_noise_results_20260916.md)
and [live tracker](../paper_ready_results/baseline_read_noise_run_status_20260916.md).

At 00:53 CEST on September17, all Conv1/Conv2 cases are complete: 16/22
collected bundles pass full revalidation. Local's five-case Conv2 queue
exited0 at00:50:57 CEST, closed its tmux window and released its GPU.
Only Fifi's six Conv3 cases remain running or queued; its first two workers
are in epoch25/30. The declared source, scientific parameters and budgets
remain unchanged.

At 02:48 CEST, Fifi's first pair has completed and passed collection:
18/22 total. Its production pack exits0 at02:44:53 CEST after10h57m, and
remote/local files match by checksum. The second pair (sigma3e-5/1e-4)
starts immediately with two CUDA workers, PIDs1010926/1010927. The final
pair remains sigma3e-4/5e-4; the measured duration predicts completion
around00:40 CEST on September18, within the declared deadline and budget.

At 13:46 CEST on September17, 20/22 trainings are collected and validated.
Fifi pair2 exits0 at13:42:23 CEST after10h57m, with no remote/local checksum
differences. Its sigma3e-5/1e-4 final validation accuracies are97.04/96.64%.
The final pair starts immediately: sigma3e-4/5e-4, CUDA PIDs1080460/1080461,
same source identity and two workers. Expected completion remains around
00:40 CEST on September18; no scientific parameters or budgets changed.

## Cases

Use the same five nonzero endpoint-voltage noise standard deviations as the
completed ours/legacy study: **1e-5, 3e-5, 1e-4, 3e-4, 5e-4**.
Every listed beta is fixed across its curve and is the injected beta;
baseline amplification makes base beta equal to injected beta.

| Architecture | Beta | T/K | Epochs | Noisy trainings | New clean trainings | Intended host |
|---|---:|---:|---:|---:|---:|---|
| Conv1 baseline | 100 | 4/4 | 10 | 5 | 0; reuse existing beta-100 control after audit | akibscomputer |
| Conv1 baseline | 200 | 4/4 | 10 | 5 | 1 at sigma=0 | nom-cool-1 |
| Conv2 baseline | 100 | 6/6 | 30 | 5 | 0; reuse existing beta-100 control after audit | local (nom-cool-2) |
| Conv3 baseline | 10 | 8/8 | 30 | 5 | 1 at sigma=0 | fifi, conditional on prelaunch availability and timing |

Total: **22 new full trainings**, comprising 20 noisy runs and two matching
clean controls. Use model, split and minibatch-order seed **0** only, matching
the existing single-seed ours/legacy sweep. No additional seed is queued.
Conv1 beta 200 and Conv3 beta 10 need new clean controls because the collected
baseline controls used beta 100. The existing Conv1/Conv2 beta-100 controls
are reuse candidates, not automatically assumed numerically identical across
GPU environments. Verify their full scientific contract, initializer and
dataset/order identities before computing clean-relative losses. If that
audit fails, record the required replacement and revised count before launch.

## Fixed scientific parameters

- Centered, frozen-current EqProp; float64; Adam with betas (0.9,0.999),
  epsilon 1e-8, weight decay 0, no LR decay (factor 1), and inherited exact
  weight learning rates. All bias tensors and bias learning rates stay zero.
- Baseline voltage/current amplification 1/1; perfect diode; wide conductance
  projection [0,100]; original shared Kaiming initializer for each architecture,
  loaded in float32 and converted to float64; fresh optimizer state per run.
  Preserve explicit diode parameter dictionaries and solver settings.
- Conv1/2/3 hidden channels [64], [64,128], [64,128,256]; strides [2], [2,2],
  [2,2,1]; kernel 3, padding 1, no pooling, paired 20-output squared-error loss.
  Input gains are 40,100,360 respectively.
- Ordinary MNIST, deterministic stratified 55,000/5,000 train/validation split;
  no affine augmentation; signed two-channel input from
  0.3*(x-0.1307)/0.3081. Train batch 16, validation batch 64. Full epochs and
  full data; reset voltages at each batch; fixed iteration counts.
- Noise seed **2026081601**. Add independent Gaussian noise of standard
  deviation sigma to copied non-input voltages at the positive and negative
  endpoints used for gradient readout. Relaxation, inputs and validation
  remain noiseless. Match underlying standard-normal draws within compatible
  environments and verify the streams rather than assuming matching seeds
  suffice across GPU/software environments.
- Save best and final checkpoints and validation accuracy; select best by
  maximum validation accuracy. The official MNIST test split stays disabled.
  These are validation robustness diagnostics, not official-test evidence.

Exact weight learning rates (all Bias_* entries are 0):

| Architecture | ConvWeight_0 | ConvWeight_1 | ConvWeight_2 | DenseWeight_0 |
|---|---:|---:|---:|---:|
| Conv1, both betas | 0.000866763 | — | — | 0.000109401 |
| Conv2 | 0.00260078 | 0.000459053 | — | 0.00046766 |
| Conv3 | 0.0026037179886572314 | 0.00046993113678684014 | 0.0003322605058323205 | 0.001023541308739098 |

## Qualification and interpretation

Conv1 beta 100 and 200 pass the expanded seed-0 initialization/best-checkpoint
gradient and residual checks; neither has three-seed confirmation in that
audit. Conv2 beta 100 passed all three confirmation seeds. Conv3 beta 10
passes seed-0 selection and seeds 0/2 confirmation but fails two initialization
comparisons on seed 1. Its trained T8 free-state residual caveat also remains.
Filip's requested Conv3 run therefore uses the recent candidate as a labeled
single-seed robustness diagnostic; it is not a promoted universally qualified
beta. The original beta-100 Conv3 clean accuracy is not its clean control.

The [iteration decision](../paper_ready_results/eqprop_drift_gradient_report_20260916.md)
retains T=K=4/6/8 across schemes. No T/K, beta, LR or noise-grid search is part
of this launch. Preserve low-accuracy or divergent scientific outcomes as
outcomes, rather than silently changing the parameters or dropping cells.

## Placement and throughput

The September 16 15:31 CEST inventory found local RTX3090, Akib RTX3080,
Nom RTX3090, Trex RTX5090 and Fifi RTX5090 idle. Loulou's RTX5090 has unrelated
GPU work at 100% utilization; Riri fails host-key verification and Jean Zay
times out. The requested campaign uses local/SSH GPUs, not Jean Zay.

Keep one architecture/beta curve on one recorded host. Local, Akib and Nom
begin with one worker each: the previous study found worse aggregate speed
when packing Conv2/Conv3 on these GPUs. Fifi is the first choice for the six
Conv3 cases. Trex is an alternative if Fifi is unavailable before admission.
Retain the older **one campaign RTX5090 at a time on weekdays** limit; the
new instruction permits concurrent workers on that GPU. Never occupy an
unrelated busy lane or treat low memory usage as availability.

Before full runs, time one versus two concurrent representative Conv3 workers
on the selected 5090; test a third only if memory and utilization leave room.
Use the concurrency with the best aggregate throughput and adequate memory
headroom. Do not claim that three full Conv3 workers fit until measured.
If a 5090 becomes available later, move an unstarted whole curve where
possible; preserve recorded environments and identify any unavoidable
cross-environment comparisons. Do not migrate or restart valid active runs
just to change hosts. No automatic expansion of scientific coverage occurs.

For Conv1 beta 200 and Conv3 beta 10, begin with the clean case and lowest
noise value, then admit the remaining noise values after semantic progress
and finite updates are observed. Noise levels remain the full declared grid.

## Time, budget and monitoring

Provisional planning allowance: **60 physical GPU-hours**, including at most
1 GPU-hour for smokes/timing and recovery within the total. This is separate
from the paused clean-training campaign. Shared-GPU workers are charged by
the host allocation interval, not once per worker.

Earlier same-runner estimates were about 0.14–0.15 hours per Conv1 training,
1.8–1.9 hours per Conv2 on Ampere GPUs, 6.3–6.5 hours per Conv3 there, and
4.2–4.5 hours per Conv3 on 5090s. Applying those rates to this scope gives
about **36–40 GPU-hours** before overhead and roughly **25–30 hours elapsed**
if the six Conv3 cases remain serial on one 5090. Packing may shorten this;
the baseline-specific ETA and per-host allocation caps must be replaced by
measured same-runner timing before production. Use a finite **48-hour wall
deadline from the first production admission**, with per-case limits that
do not shorten epochs. These are estimates and operational limits, not an
already measured baseline completion forecast.

Use `python -m experiments.exact_run` with each unchanged full config for
one-train-batch/one-validation-batch semantic smoke, then production. Use the
already accepted scientific operating points and explicitly retained Conv3
qualification exception; do not rerun a broad beta search. Freeze the trainer
from the completed read-noise study's `source-v2`, preserving unrelated working
changes. Record exact commands, source/config hashes, environment and handles.

Check every new launch twice for real artifacts, then monitor all active
hosts at least every 30 minutes. Track batch/epoch progress, heartbeat, GPU
processes and logs; diagnose operational failure and preserve failed attempts.
Collect remote outputs locally and validate canonical bundles, full epoch
coverage, zero biases, finite float64 PT/NPZ checkpoints, noise-draw counts,
initializer/cohort/order and terminal receipts. Publish best/final validation
and clean-relative changes, retaining environment and beta qualifications.

## Files

- Exact prepared configs: [`configs/conv/eqprop_baseline_read_noise_20260916_v1/`](../configs/conv/eqprop_baseline_read_noise_20260916_v1/).
- Run list: [`run_plan.csv`](../configs/conv/eqprop_baseline_read_noise_20260916_v1/run_plan.csv).
- Expected result root (not created by this plan):
  `results/eqprop-read-noise-seed0-baseline-20260916-v1/`.
- Collected results and report destination: `paper_ready_results/`.
- Prior evidence: [ours/legacy sweep](../paper_ready_results/read_noise_sweep_results.md),
  [baseline beta 100/200/300 audit](../paper_ready_results/baseline_beta_audit_20260914.md),
  [Conv3 beta-10 confirmation](../paper_ready_results/conv3_baseline_beta_tk8_20260916.md).
