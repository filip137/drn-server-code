# Conv3 p90 read-noise follow-up

Status (September 20, 18:36 UTC): complete and reviewed. All24 thirty-epoch
outcomes are collected and validated:15 noisy cases and nine GPU-matched
clean controls. No declared work remains active or queued. All54 Slurm
allocations and four local launchers exited successfully, with matching
remote/local production checksums. Actual production used88.99817 physical
GPU-hours, including26.41622 on the four local RTX5090 hosts; preparation
checks are separate. All24 pass the predeclared endpoint screen, but the
legacy higher-noise curves show substantial temporary deterioration.
See paper_ready_results/conv3_p90_read_noise_20260919.md and the reviewed
experimental_manifest.md entry. The original21-case arrangement below is
historical; the final accelerated placement has24 outcomes.
Filip authorized this follow-up and many concurrent Jean Zay GPUs on September19.
The scope is the current Conv3 study, all three amplification schemes, seed0.
Filip explicitly corrected the horizon to30 epochs, matching the previous
paper-ready read-noise results; the earlier ten-epoch proposal is superseded.
This is exploratory validation robustness evidence, not official-test evidence.

| Scheme | Fixed injected beta B | Base beta |
|---|---:|---:|
| Baseline | 404.141105702 | 404.141105702 |
| Ours | 5.26875648112 | 0.0823243200175 |
| Legacy | 4.42250110273 | B / 4096 |

Use sigma={1e-5,3e-5,1e-4,3e-4,5e-4}, the previous read-noise grid.
Fifteen noisy trainings plus six hardware-matched clean controls yield21
settings. V100 covers sigma0/1e-5/3e-5 and A100 covers0/1e-4/3e-4/5e-4,
for all three schemes. Each scheme/GPU pair has its own30-epoch clean control.
Compare the earlier30-epoch curves separately, labeling their different betas
and environments. The completed ten-epoch pilots are admission evidence, not
the clean controls for this thirty-epoch sweep.

Freeze T=K=8, ordinary MNIST fixed55k/5k split, batch16/validation64,
reset-each-batch states, matched saved Conv3 initializer, centered frozen-current
float64 EqProp, fresh Adam, unchanged exact scheme-specific weight rates,
zero biases, perfect diode and wide [0,100] weights. Preserve all explicit
diode dictionaries. No new beta/LR/solver search. Official test is disabled.

Noise is independent zero-mean Gaussian perturbation of the copied non-input
positive/negative endpoint voltages used for gradient readout. Inputs,
relaxation and validation remain noiseless. Noise seed2026081601 is fixed,
beta is selected at zero noise and then held fixed across each curve.
Use the same GPU/PyTorch contract across schemes at each noise level;
compare every noisy result to the clean control on that GPU class. Verify noise
implementation, draw counts and matched initializer/cohort/order; do not infer
cross-environment noise-stream identity from the seed alone.

Primary endpoint: epoch30 validation accuracy and its drop from the exact
clean control. Also report best validation, losses, and nonfinite failure
location. Retain failures and low accuracies as scientific outcomes. No
scientific failure may silently suppress the next assigned case. Operational
failures stop that task for diagnosis, with retries in new directories.
The preceding ten-epoch clean pass qualifies only admission to this full
horizon diagnostic; numerical failures during the30-epoch sweep remain included.

## Execution split across A100 and V100 with exact continuation

Filip authorized V100 or A100 after long-queue forecasts. A100 timing check
2179514 completed the full highest-noise baseline epoch in6m26s including
startup, with a valid local bundle and27504 noise draws. Thirty epochs do
not fit the two-hour development queue as one allocation.

Use A10080GB,umg@a100,gpu_p5,qos_gpu_a100-dev,4CPUs and one GPU per
allocation. Preserve30 epochs through three10-epoch chunks. Each allocation
has90min (native timeout89min); estimated runtime about60min per chunk.
Keep one A100 and one V100 active concurrently, at most two GPUs total.
V100 usesfmu@v100,gpu_p13,qos_gpu-dev,v100-16g,4CPUs,120min
allocations with119min native timeout. Its three10-epoch chunks preserve
the same continuation contract; the longer cap accounts for measured V100 speed. Queue three cases per group, with three
arrays of the same three case indices and concurrency1; phase1 depends
on successful completion of the whole phase0 array, and phase2 on phase1.
This keeps at most9 A100 jobs submitted, below the development-QoS cap10.
Submit subsequent groups after completion/reconciliation of the preceding
one. Measured baseline epoch times are about5.87min on A100 and8.79min on
V100. Extrapolating these gives about75GPU-hours total (35A100+40V100),
about40h elapsed with both lanes admitted, plus queue time. This estimate
will be revised as other schemes and noisy cases run. Maximum108GPU-hours
(36 ninety-minute A100 allocations +27 two-hour V100 allocations),
plus checks within the original109GPU-hour cap.

The continuation stores parameters, Adam moments/step, full metric history,
best-validation checkpoint, loader sampler/worker RNGs, global RNGs, dedicated
endpoint-noise generator state/count and gradient trace counts. It restores
only intact planned epoch-boundary pauses under the identical source/config/
GPU/PyTorch contract. No optimizer or noise-stream reset. Final canonical
result is published only after epoch30. Every chunk has a separate Slurm
receipt; failed/extra artifact tails are rejected rather than discarded.

Six local real-Conv3 comparisons (three schemes, zero/highest noise) with two
process restarts match the original frozen runner bitwise in best/final
weights, all accuracy/loss histories and complete gradient traces. Dedicated
source/config/tamper guards and a local continuation smoke precede one live
A100 canary exercising both restores. Full production waits for that canary.
The original source and excluded timing/smoke bundles remain preserved.
New source receipt: results/eqprop-conv3-p90-read-noise-20260919-v1/continuation-source.json.
Active A100 configs: configs/conv/eqprop_conv3_p90_read_noise_a100_20260919_v1/.
Remote production root: /lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/eqprop-conv3-p90-read-noise-20260919-v1/a100-production/.

Group order per GPU lane:

- A100:0,3,6 (clean);11,14,17 (highest noise);9,12,15;10,13,16.
- V100:0,3,6 (clean);1,4,7;2,5,8.

Each group uses three dependent arrays (one per chunk), with one task
running at a time and afterok dependencies between whole arrays. No later
group on that lane is submitted until its predecessor is reconciled.
An operational/scientific failure is diagnosed and recorded; other declared
cases are not silently excluded when a dependency is blocked.

Monitor each new allocation through two semantic checks, then at most30min
apart. Collect remote results and validate the local copy before reporting
completion. Expected coverage21 full30-epoch outcomes, including scientific
failures. Official test remains unread. Keep local result bundles under
results/eqprop-conv3-p90-read-noise-20260919-v1/jz/{a100,v100}-production/ and final
summaries in paper_ready_results/conv3_p90_read_noise_20260919.md.

Earlier H100 and V100 preparations were never released as noisy production.
H100 check2178929 and A100 timing check2179514 are excluded diagnostics.
V1002179098 and2179134 were canceled while pending, with no allocated
GPU time. The validated short-queue continuation contract supersedes them. Old configs, smokes, source receipts and
queue forecasts remain preserved under the study root. Only descriptor fields
changed between V100/A100 configs; all scientific settings are identical.

Monitor deadline:2026-09-23T06:00:00Z; refresh status at most30min apart,
and revise elapsed estimates from measured full epochs on each GPU class.

## September20 user-directed local acceleration

Filip requested parallel local GPUs to shorten the remaining26–30h. All four
RTX5090s and both RTX3090s were idle at inspection. Stage the unchanged frozen
scientific source; use three concurrent workers per host after concurrent
same-runner smokes and excluded100-batch throughput checks. All configs retain
30epochs,seed0,T=K=8,selected injected betas,exact rates and data/noise contracts.
RTX5090 hosts use PyTorch2.11.0+cu128/CUDA12.8; both3090 hosts use
PyTorch2.5.1+cu121/CUDA12.1. New matched clean controls are required per class.

- Fifi: all3 clean5090 controls. Riri: all3 sigma3e-5.
- Loulou: all3 sigma1e-4. Trex: all3 sigma3e-4.
- Local3090: all3 clean controls. Nom-cool-1: all3 sigma1e-5.
- Jean Zay: finish A100 sigma5e-4. Retain started V100 sigma1e-5 baseline
  as an additional check; replace its unstarted legacy/ours cases locally.

Once local admission passes, retire the three unsubmitted HPC groups and
cancel only the obsolete pending V100 low-noise legacy/ours tasks/dependent
arrays. Rebuild the remaining baseline continuation dependencies without
resetting its checkpoint. Preserve all original plans and outputs.

This adds18 local runs,including6 matched controls. Expected total becomes
28 outcomes (27 planned main rows plus the extra V100 low-noise baseline).
Matched noise comparisons use one GPU/PyTorch/CUDA class for all schemes at
each sigma. Different physical hosts within a class are recorded; no
cross-class bitwise noise-stream identity is assumed. These remain single-seed
validation diagnostics,not official-test results.

Provisional cap16h per local host (three workers concurrent),96 local GPU-hours
and150 total GPU-hours including prior/current Jean Zay and preparation.
Expected elapsed time will be set from the measured concurrent batch rate.
Local production runs continuously for30epochs; existing exact-continuation
source is retained but no allocation boundary is needed. Save canonical
bundles plus native exit/log/config/source/environment receipts.

### Throughput-based final acceleration placement

The18-run local layout above is superseded before production. Three-worker
100-batch probes took61–65s on5090s and86–87s on3090s,including startup,
and reached full GPU utilization. Six local30-epoch cases are retained:
fifi baseline clean+sigma3e-5; riri legacy clean+sigma3e-5; loulou ours clean;
trex ours sigma3e-5. Same5090/PyTorch2.11.0+cu128/CUDA12.8 class throughout.
The3090 runs are excluded operational smokes/probes only.

Keep V100 sigma1e-5 and A100 sigma5e-4/1e-4/3e-4 on Jean Zay,with
three concurrently admitted tasks per array instead of one. Verified live
QoS permits32 GPUs per user/account and10 submitted tasks; each task still
uses one GPU,fourCPUs,unchanged source/wrapper and existing90/120min caps.
No scientific configuration or per-task execution contract changes. Reuse
the passed local/live task gates and verify each newly admitted task twice.
Future groups still await predecessor collection/reconciliation.

Expected coverage24 outcomes:15 noisy and9 matched/retained clean controls.
Only the unsubmitted V100 sigma3e-5 group is replaced locally. Preserve all
original plans. No running training is discarded or selected by accuracy.
Estimated remaining elapsed10–13h,subject to queues/full-epoch timing.
Up to4 local GPUs plus3A100+3V100 concurrently. Local cap16h/host (64GPUh),
study cap150GPUh including completed and current HPC work and all checks.


### First full local epoch timing (September 20, 12:00 Paris)

All six local runs advanced through their first full epoch. Paired workers
on Fifi and Riri took 21.96 and 22.38 minutes per epoch, respectively;
single workers on Loulou and Trex took 8.28 and 9.00 minutes.
The first-epoch projection puts the slowest local completion around
22:30 Paris today, subject to sustained throughput. GPU utilization was
99% on both paired hosts and 86–90% on the single-worker hosts.
Riri's wall clock is about two hours behind the other hosts; elapsed times
are measured within each host, and remote timestamps are retained unchanged.


### Scheduling correction after measuring contention (September 20)

Two workers made the GPU utilization counter reach 99%, but reduced aggregate
training throughput. Bounded four-minute pause/resume checks on Fifi and Riri
measured approximately 71–73 seconds per 500 batches for one computing worker,
compared with about 190–195 seconds per worker under paired execution. All
paused workers resumed successfully. The projected throughput gain for a pair
run sequentially is about 32%; utilization alone was misleading.

Keep all six local cases and their live states. On Fifi and Riri, pause the clean
trainer in memory while the noisy trainer finishes, then resume the clean
trainer automatically. A four-hour maximum pause and signal-handler cleanup
ensure resumption; dummy-process checks exercise normal exit and interrupted
waiting. No process is restarted, no case is dropped, and the scientific source,
configurations, RNG state and epoch budgets are unchanged. The other local hosts
and all Jean Zay allocations remain parallel. Original two-worker launch
receipts remain authoritative for how the processes started; the additional
scheduling receipts describe when a worker was deliberately suspended.

The resulting provisional overall completion estimate is 20:00–21:00 Paris on
September 20, subject to full-epoch timing and Jean Zay queues. The GPU-hour and
per-host wall-time caps remain unchanged. Monitoring must distinguish these
explicit scheduling pauses from a stalled worker and check the bounded
resumption process as well as the active trainer.


### Completed scheduling and collection (September 20)

Fifi and Riri clean workers resumed automatically after their noisy peers
finished, retained their live training state and completed30 epochs. All six
local outcomes are collected and checksum-matched; the four launcher exits
are zero. Last training completed around20:35 Paris, within the revised
20:00–21:00 estimate. The continuation and local pauses changed scheduling
only. All24 declared cases, including the fluctuating legacy noise curves,
remain included. The final coverage/accounting receipt is
`results/eqprop-conv3-p90-read-noise-20260919-v1/closeout-validation.json`.
