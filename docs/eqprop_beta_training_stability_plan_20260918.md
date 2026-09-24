# Larger-beta EqProp early training stability — September 18, 2026

Filip authorized training pilots and then explicitly narrowed the budget to
10 epochs for all architectures, on local RTX5090 machines with concurrent runs.
This replaces the initial 10/30/30-epoch and Akib/local/Jean Zay placement plan.
No full training or Jean Zay job had started at that revision.

| Architecture | Baseline injected beta | Ours | Legacy | T=K | Epochs |
|---|---:|---:|---:|---:|---:|
| Conv1 | 5000 | 1500 | 900 | 4 | 10 |
| Conv2 | 1000 | 100 | 30 | 6 | 10 |
| Conv3 | 750 | 22.5 | 1 | 8 | 10 |

These are the largest tested candidates passing whole-gradient cosine >=.99
and symmetric norm mismatch <=.10 on all 72 clean calibration replays.
Legacy Conv2/Conv3 are open upper grid edges, not established maxima.
Base beta divides injected beta by 1, 4^depth, or 16^depth for baseline,
ours, or legacy. These are deliberately exploratory whole-gradient-rule
pilots, not certified under the previous per-matrix rule. The existing
Conv3 finite-T residual caveat remains unchanged.

Use the preserved September 11 numerical source and `experiments.exact_run`.
Inherit each matching seed-0 wide EqProp config and exact Adam learning rates
from `paper_ready_results/bundles/table2_wide_ep/`. Preserve zero biases,
[0,100] weights, input gains 40/100/360, centered frozen-current float64 EP,
batch size 16, seed 0, saved wide Kaiming initializer checked by SHA256,
and deterministic ordinary MNIST 55,000/5,000 split/preprocessing/order.
Only beta and the explicitly reduced Conv2/Conv3 epoch horizon change.
Read noise is zero; the official test remains unread. Checkpoint selection
uses maximum validation accuracy. No automatic three-seed promotion.

Record loss and accuracy every epoch, and eight evenly spaced raw gradient
and update samples per epoch. Existing guards check every step for finite
states, costs, gradients, weights, and optimizer state. Early stability means
all ten epochs finite and final validation accuracy strictly less than 5pp
below the best. Separately report absolute accuracy and the difference from
the matching control's first ten epochs: stable but poor learning is not
satisfactory performance. These pilots do not establish full 30-epoch
stability for Conv2/Conv3. Check actual initializer and split/order identities
before claiming matched comparisons. Preserve all failures and incomplete
runs; never repair a scientific failure by silently tuning beta or learning rate.

Placement: verified Riri RTX5090, two concurrent workers, one study GPU.
Riri had about 32 GiB free at launch; only an idle 52 MiB CUDA MPS daemon
was present. The launcher admits work only when there is no training process
and total GPU memory use is below 256 MiB. It never terminates another job.
Expected completion 4–7 hours after
admission, subject to measured throughput. Acquisition deadline 12 hours;
per-case timeout 4 hours, maximum 36 worker-hours (up to ~20 physical GPU-hours
including acquisition-free overhead). Raise timeouts only after a recorded
review; an incomplete case is not a stable result.

Run all nine unchanged configs through the existing one-train-batch and
one-validation-batch smoke, including tracing. Inspect a paired smoke on Riri
before production to verify memory headroom and local CUDA compatibility.
A source/config hash manifest fixes the runtime. Keep at least 2 GiB GPU memory
headroom. Additional concurrency requires measured spare memory and compute.
Monitor live PID/GPU/logs/artifacts at most 30 minutes apart, with a 45-minute
artifact-stall threshold. Copy remote outputs locally and validate canonical
bundles before drawing final conclusions.

Configs: `configs/conv/eqprop_beta_training_stability_20260918_v1/`.
Local study: `results/eqprop-beta-training-stability-20260918-v1/`.
Riri study: `/home/filip/server_code/results/eqprop-beta-training-stability-20260918-v1/`.
Small report and summaries will be published under `paper_ready_results/`.

Filip identified Riri as free and requested its use. The initial SSH access
problem was resolved before launch. No Jean Zay submission occurred.
Fresh public-package environment at
`/home/filip/server_code/.venv-beta-stability-py312/bin/python`
(Python3.12.14, torch2.11.0+cu128, torchvision0.26.0+cu128, numpy2.4.4).
The prior access blocker is resolved. Source and training-only MNIST staged.
The launcher ignores the idle MPS service only when total VRAM use is <256MiB.

Production began 15:52 UTC on Riri with two workers after all nine local and
remote smoke bundles validated. Launcher PID: 2585907. The exact launcher is
`experiments/run_beta_stability_5090.sh`; its staged copy and command logs are
inside the remote study directory. Nonfinite failures are retained scientific
outcomes; no retry with changed beta or LR is permitted by this pilot contract.
See the [result report](../paper_ready_results/beta_training_stability_20260918.md)
for current measurements, and `handoff.json` inside the local study for handles.

## Placement revision after six numerical failures

Trex was rechecked and is now idle (RTX5090, 92MiB, zero compute processes).
To use the user-authorized local 5090 GPUs and reduce completion time, move
only the unstarted Conv3 ours beta22.5 case to Trex. Preserve the exact frozen
config, initializer, dtype, cohorts and minibatch order. Riri baseline and
legacy continue uninterrupted. Hold the obsolete Riri queue worker so it
cannot admit a duplicate; retain its current baseline child until completion.
Run the identical one-step smoke on Trex and compare to Riri before training.
This is an exploratory multi-host diagnostic on the same GPU model, not a
new paired paper-accuracy comparison. Record host differences explicitly.
Trex runtime: /home/filip/miniconda3/envs/py312/bin/python (torch2.11+cu128).
Trex output: /home/filip/server_code/results/eqprop-beta-training-stability-20260918-v1/.
Ten epochs and the four-hour per-case limit remain unchanged; expect roughly
1.5–3 hours for the newly admitted single-worker case, revised from measurements.

## Runtime-guard review

A conservative early throughput estimate of 30–32 minutes per epoch prompted
a review of the four-hour guard. Subsequent completed-epoch timestamps show
approximately 22 minutes with two Conv3 workers on Riri and nine minutes for
the single Trex worker; GPU activity is healthy. Retain the reviewed extension
of only the two active Riri Conv3 guards to six hours from their original
start times. Trex's single-worker guard stays at four hours. The six earlier
failures plus these remaining caps stay below the original 36 worker-hour
and approximately 20 physical-GPU-hour ceilings. Scientific configs, optimizer
steps, learning rates, betas, and ten-epoch budgets are unchanged.

The operational replacement holds only queue/timeout wrappers; scientific
children keep running. A persistent supervisor enforces the new deadline,
records the native `exact_run` process exit status, and retires the obsolete
queue shells after their children exit. The old timeout may report 124 after
an already-successful extended child; its status is preserved separately in
`.original_timeout_code`, never mistaken for a scientific failure. Native
child receipts and `*.supervision.json` are authoritative for these two cases.
The success, scientific-failure, and enforced-new-deadline paths passed local
process tests (`deadline-supervisor-tests.json`). The reviewed controller is
`extend_riri_deadlines.py` inside the study result directory.

## Completion — September 19, 2026 (Europe/Paris)

All nine outcomes are collected and validated: six Conv1/Conv2 nonfinite
failures and three Conv3 passes of the ten-epoch criterion. No exclusions,
scientific retuning, production retries, or further seed launches occurred.
Both Riri cases completed before even the original four-hour timeout; the
supervisor preserved native exit 0 receipts and retired the obsolete queues.
No study-owned GPU, queue, or supervisor process remains. Production host
occupancy totals about 6.03 GPU-hours (Riri 4.54, Trex 1.49); worker time is
10.28 hours, below the original budget. Source identities reverify on all
hosts. Results, limitations, and matched-control comparisons are in the
[result report](../paper_ready_results/beta_training_stability_20260918.md).
