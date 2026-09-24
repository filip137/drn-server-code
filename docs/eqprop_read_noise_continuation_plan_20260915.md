# Finish the single-seed read-noise sweep before 08:00

Closed September 16, 2026, Europe/Paris. Status: **complete and analyzed**.
All 18 continuation runs are collected and validated, bringing total sweep
coverage to **30/30**. The last run finished at **06:20:46 CEST**, before the
08:00 deadline; all five continuation GPUs are released. No case remains
active, queued or held. Final results and limits are in the
[combined report](../paper_ready_results/read_noise_sweep_results.md).

The continuation used **55.089836/78 physical GPU-hours**, including checks
and recovery; the separately settled first window used 22.142636 hours.
The complete sweep used **77.232472 physical GPU-hours**. The final remote
copies match by checksum. All full training packs exited 0. Local/Akib's
queue exit 1 is the documented transfer guard, with no duplicate training;
the pretraining local nohup failure and its tmux replacement are preserved.

The sections below preserve the launch decisions and intermediate checks.
Filip first requested local, Akib and Nom until completion, then authorized
other free GPUs with a hard September 16 08:00 deadline. The two queued
legacy transfers to Trex/Fifi completed within that window. The clean
training campaign remains paused.

## Unchanged science

Reuse the exact 30 configs in `configs/conv/eqprop_read_noise_20260914_v1/`
and immutable scientific source `source-v2`, SHA-256
`4055090e16606583c276b06c3aac59396e87dcedf58b8cde9be94c5824948fb1`.
The remaining cases retain wide [0,100] weights, exact-zero biases, centered
current-nudged float64 EqProp, Adam, seed 0, dedicated noise seed 2026081601,
ordinary-MNIST 55k/5k split, batch 16, 10/30/30 epochs and T/K 4/4,6/6,8/8.
Injected betas remain ours 30/10/3 and legacy 3/.03/.001; every learning-rate
vector stays unchanged. Noise affects endpoint gradient readout only. No
official-test read, new zero-noise case, 1e-3 case, baseline, or clean-training
continuation is included. The inherited Conv3-ours qualification exception
remains explicit; these are validation robustness diagnostics.

## Initial serial placement, before the two deadline transfers

Run **one training worker per GPU**, with each host's cases in the listed
order. Each new architecture/sigma ours/legacy pair stays on one host.

| Host | Serial cases, all seed 0 |
|---|---|
| Local — 6 | Conv2 ours then legacy at 3e-4; Conv2 ours then legacy at 5e-4; Conv3 ours then legacy at 3e-5 |
| Akib — 8 | Conv1 ours then legacy at 3e-4; Conv2 ours then legacy at 1e-5; Conv2 ours then legacy at 3e-5; Conv3 ours then legacy at 3e-4 |
| Nom-cool-1 — 4 | Conv1 legacy at 3e-5; Conv3 legacy at 1e-5, 1e-4, and 5e-4 |

At 11:03 CEST, local and Nom have idle RTX 3090s (24 GiB), while Akib has an
idle RTX 3080 (10 GiB). All use PyTorch 2.5.1/CUDA 12.1. The 5090s are occupied
and are not selected. Riri's host-key check fails and Jean Zay times out.
The user corrected `integnano-1` to `nomcool-1`; the configured working alias
is `nom-cool-1`.

The concurrent timing candidates are retained as diagnostics. Three Conv2
workers on Akib require about .269 seconds per batch each, while a single
worker needs .061: serial work has approximately 47% higher aggregate
throughput. Two Conv3 workers on a 3090 need .513–.518 seconds per batch each,
while one needs .213–.221. The measured serial queues therefore finish faster.
Single Conv3 also fits Akib and runs at about .218 seconds per batch; two
Conv3 cases move there to balance the three queues. Full configs are unchanged.

## Measured duration and budget

Reserve **78 additional physical GPU-hours**: 77.5 for full-run operational
caps plus at most .5 for CUDA timing/smokes. Expected consumption is about
59.8 GPU-hours across the three hosts, with maximum expected elapsed time
**20.44 hours**. For a launch around 11:35 CEST September 15, the expected
last completion is around **08:00 CEST September 16**. This is an estimate;
the operational limits include a 25% timing margin and cleanup overhead.

| Host | Cases | Expected hours | Expected with 25% margin | Reserved GPU-hours | Absolute queue cutoff, September 16 CEST |
|---|---:|---:|---:|---:|---|
| Local | 6 | 20.437 | 25.546 | 26.500 | 14:28:59 |
| Akib | 8 | 20.403 | 25.506 | 26.667 | 14:38:59 |
| Nom-cool-1 | 4 | 18.953 | 23.692 | 24.333 | 12:18:59 |

The forecast uses median steady batch times, 3,438 train and 79 validation
batches per epoch, validation timings, five seconds overhead per epoch,
and 30 seconds startup per case. Each case gets its own deadline-protected
limit. Timeouts remain incomplete evidence; epochs are never shortened to fit.
All source rates, case commands, deadlines and caps are in
[`serial_commands.json`](../results/eqprop-read-noise-seed0-ours-legacy-20260914-v1/launch/continuation_20260915/serial_commands.json).
The completed overnight 22.142636 GPU-hours and the paused clean campaign
keep their existing accounting; neither is charged to this new allowance.

The three Conv3 legacy counterparts at 1e-5, 1e-4 and 5e-4 move from their
originally proposed 5090 hosts to Nom at the user's request. Their completed
ours references retain the old environment. The prior CUDA audit showed
different noise samples across 5090 and 3090 environments despite the same
seed. These three cross-environment pairs must be labeled accordingly;
do not claim identical noise realizations. Check the local and Akib noise
streams and retain separate environment groups in combined plots. Conv1
legacy at 3e-5 stays on Nom with its already completed ours counterpart.

## Execution, monitoring, and collection

Run all 18 unchanged configs through local one-batch exact-run smoke mode.
Measure each distinct concurrent architecture mix on its selected GPU using
separate short diagnostic configs. Smoke the admitted packs on their actual
targets, then run the full configs. Reuse the accepted scientific operating
points rather than retuning beta, T/K, or rates.

Use the existing deadline pack runner for one case at a time, with an explicit dataset-directory
transport override for Akib; the scientific source and configs remain intact.
Keep the continuation wrapper, its hash, exact commands, launcher handles,
logs and exit receipts under the existing study directory. A direct shell
script runs each host's queue in order and stops on failure. Every case
checks that the GPU has no other compute process before admission. Launchers
survive disconnects and refuse duplicate output roots.

Check each launch twice for actual artifact progress, then monitor every
30 minutes. Diagnose missing processes, stale progress, OOM/nonfinite errors,
or nonzero receipts. Retry understood operational failures within the recorded
budget without changing the science. Keep failed evidence and use a distinct
replacement output path. Never stop unrelated jobs.

Outputs remain under
`results/eqprop-read-noise-seed0-ours-legacy-20260914-v1/production/continuation_20260915/`.
Copies, tables and plots continue in `paper_ready_results/`. For every full
run, validate its canonical bundle, exact config/source identity, full epoch
and data coverage, initializer/cohort/order, noise-draw count, finite float64
checkpoints, zero biases, bounds and PT/NPZ equality. Reconcile all 30 cells,
remote/local checksums, exit receipts and physical GPU-hours before finishing.

## Launch verification and local recovery

All 18 local and 18 final-target GPU semantic smokes pass. Uncompressed MNIST
training-image and training-label hashes are identical on all three hosts;
no official-test file was read. Akib's resolved config records its permitted
dataset-directory override. The collector accepts only that recorded path
change; regression checks reject a changed beta or cohort. Four transport and
collector tests pass. Source and queue hashes and idle GPUs were verified
immediately before production.

| Host | Current launcher | Queue PID | Output below `production/` |
|---|---|---:|---|
| Local | main/tmux window `@10` | 1020868 | `continuation_20260915/local_tmux_v2/` |
| Akib | nohup, launcher PID 175487 | 175496 | `continuation_20260915/akibscomputer/` |
| Nom-cool-1 | nohup, launcher PID 1625346 | 1625355 | `continuation_20260915/nom-cool-1/` |
| Trex | nohup, launcher PID 1980435 | 1980444 | `deadline_20260915/trex/` |
| Fifi | nohup, launcher PID 838703 | 838712 | `deadline_20260915/fifi/` |

The first local nohup attempt lost both launcher and queue processes before
creating any training artifact. Its `local/` directory and `local.launcher/`
receipts are retained. Executor cleanup is the suspected cause. An unchanged
one-batch GPU smoke subsequently completed under the repository's main/tmux
launcher with a zero exit receipt, and the full local queue was relaunched
into `local_tmux_v2/`. No trained seed or completed result was discarded.
Conservatively charge .1 GPU-hours for this failed launch inside the .5-hour
checks allowance; full-run caps and the 78-hour total remain unchanged.

The scientific source, configs and deadlines are unchanged. The current
local script is `local_queue_tmux_v2.sh`; original commands are preserved in
`serial_commands.initial.json`, with the active paths in `serial_commands.json`.
Early artifact checks show all three queues making real training progress.
The first three collected continuation results are Conv1 ours at 3e-4 and
legacy at 3e-5/3e-4, finishing at 96.30% and 96.40%/96.36% validation
respectively. All ten Conv1 cells are now collected and validated. The
19:05 collection includes all Conv1 and Conv2 cells. Across sigma
1e-5/3e-5/1e-4/3e-4/5e-4, Conv2-ours final validation is
98.08/98.08/98.00/97.80/97.60%; legacy is
97.62/97.38/96.52/95.94/95.72%. These same-host pairs show greater observed
legacy degradation under the inherited beta and learning-rate contracts;
they do not isolate amplification from beta. Conv3-legacy 1e-5 finished at
17:57:49 CEST, best/final 96.12/95.68%, versus the historical clean
98.84/98.78% and the noisy ours counterpart 98.38/98.26%. The latter
comparison crosses GPU/software environments and noise realizations.
Local/Akib started Conv3-ours 3e-5/3e-4 at 18:58/19:02 CEST, with real
GPU batch progress; each queues its matching legacy case. Nom
Conv3-legacy 1e-4 has completed five epochs and queues 5e-4. All three
handles, GPU workers and fresh batch events are healthy. Final queue
completions remain expected around 06:32–08:09 CEST on September 16.

- [Existing results](../paper_ready_results/read_noise_overnight_results_20260915.md)
- [Continuing run tracker](../paper_ready_results/read_noise_run_status_20260914.md)
- [Original overnight plan](eqprop_read_noise_overnight_plan_20260914.md)

## September 15, 23:14 CEST: extra-GPU deadline acceleration

Filip authorized outsourcing to other GPUs and set a hard finish deadline of
**September 16, 08:00 CEST (06:00 UTC)**. Enforce a worker cutoff of 07:55
CEST and leave time for collection. The earlier serial forecast put local
and Akib near 07:54, so move their unstarted last cases to free GPUs rather
than relying on that narrow margin. The three current trainings continue.

| Queued case, unchanged seed 0 | Previous queue | New target | New output below the study root |
|---|---|---|---|
| Conv3 legacy, sigma 3e-5 | Local, last case | Trex RTX 5090 | `production/deadline_20260915/trex/` |
| Conv3 legacy, sigma 3e-4 | Akib, last case | Fifi RTX 5090 | `production/deadline_20260915/fifi/` |

Trex and Fifi have no GPU compute process and their complete source-v2 input
hashes match. Loulou has an existing MPS server; Riri's host-key check fails
and Jean Zay's SSH connection times out. These are not selected. The extra
GPU authorization applies to this read-noise deadline window, including two
free 5090s. Use one worker per new GPU. Nom's final legacy 5e-4 case remains
on Nom; measured progress predicts its queue finishing around 06:25 CEST.

Expected new-case duration is approximately 5–6.5 hours, to be replaced by
the measured legacy-case timing before admission. Require a fresh unchanged
one-batch GPU smoke and a separate short timing diagnostic on each target.
Admit full training only if the estimate plus 25% and a five-minute margin
fits the 07:55 cutoff and the original case cap (Trex 29,700 seconds, Fifi
29,400). Transfer those reservations from the original queues; the total
continuation allowance stays **78 physical GPU-hours**, including at most
.5 hours of checks. No additional seed, epoch, noise value or training repeat
is authorized by this transport change.

The frozen scientific source, configs, initialization and cohort/order remain
unchanged. The two moved legacy runs use the audited 5090 noise stream and
therefore differ from their current ours counterparts on local/Akib. All five
Conv3 pairs must now retain the cross-environment/noise-realization limit.
This supersedes the earlier same-host placement for the two moved pairs.

Before launching a transferred case, reserve its original unstarted output
directory with a readable transfer notice. The existing exclusive-directory
guard will then stop the original queue before it can launch a duplicate,
after its current training finishes. Preserve the resulting intentional
queue exit and all prior receipts; it is a transport handoff, not a failed
training. New outputs and launch receipts use distinct directories. Monitor
all five GPUs through actual artifacts and collect every complete run; stop
any remaining owned training by 07:55 if the deadline is reached.

Both extra-host smokes and timing diagnostics passed, including canonical,
source, noise-draw-count and disabled-test checks. The measured full-case
estimates are Trex **4.520 hours**, **5.650 with 25% margin**, and Fifi
**4.206 hours**, **5.257 with margin**. Both fit the unchanged individual
caps and hard cutoff. They launched at about **23:29 CEST**, each with one
real GPU worker and an initial training batch. Current expected completion
is approximately 04:00 on Trex and 03:45 on Fifi; Nom's existing queue then
sets the overall forecast at **06:25 CEST**, about 90 minutes before the
user deadline.

The original local/Akib queued-output reservations and transfer notices are
installed. Their queue exit code 1 is expected only after the current full
training completes and the exclusive-directory guard rejects the transferred
case. The new one-case queues have separate `trex.queue/` and `fifi.queue/`
metadata, with successful production still requiring ordinary zero receipts.
All commands, source/queue hashes, admission rates, transfer notices and
launcher receipts are under
[`launch/deadline_20260915/`](../results/eqprop-read-noise-seed0-ours-legacy-20260914-v1/launch/deadline_20260915/).
