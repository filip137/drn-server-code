# Ours/legacy read-noise runs before 08:00 on September 15

Later September 15 authorization: the remaining 18 cases are continuing on
local, Akib, and Nom under the [continuation plan](eqprop_read_noise_continuation_plan_20260915.md).
The deadline and resource accounting below describe the completed first window.

Prepared: September 14, 2026, Europe/Paris. Updated September 15: **all 12 admitted
full runs completed by 05:52:58 CEST, collected and validated, before the 08:00
deadline.** All four GPUs are released. The 30-cell study remains partial,
with 18 cases held and no automatic follow-up queued.

## Authorized scientific scope

Wide [0,100] Conv1/Conv2/Conv3, ours and legacy, model/shuffle/split seed 0,
with injected endpoint read-noise standard deviations
`1e-5, 3e-5, 1e-4, 3e-4, 5e-4`: **30 noisy cells**. No new sigma-0 or
sigma-1e-3 run is scheduled. Baseline and bounded conditions are excluded.
The earlier clean campaign remains paused. Existing clean seed-0 results
provide reference curves only after the complete contract audit.

Preserve the inherited Adam vectors, corrected physical KCL, exact-zero
biases, float64 centered frozen-current EqProp, the ordinary MNIST 55k/5k
split, batch size 16 and 10/30/30 epochs. T/K stays 4/4, 6/6, 8/8. Retain
the already trained injected beta anchors: ours 30/10/3, legacy 3/.03/.001.
No beta or LR tuning is included in this overnight noise comparison.
The accepted historical Conv3-ours clean-gradient exception remains explicit;
these are validation robustness measurements under that training contract,
not a claim of full gradient qualification. The separately running baseline
beta audit does not alter these ours/legacy configs.

Use the existing independent Gaussian endpoint-voltage noise model, with
dedicated noise seed 2026081601. Noise affects gradient readout, not relaxation,
inputs or validation. Match initial weights and training order. Match standard-normal
draws within each architecture and host across schemes; cross-host matching is
subject to the measured limitation below. Start every run from
the recorded seed-0 initializer with fresh optimizer state. Official-test
access is disabled. Keep failures and report both best/final validation.

## Deadline and resource budget

The hard user deadline is **2026-09-15 08:00 CEST = 06:00 UTC**. Aim for
completion by 07:30 CEST and enforce worker termination before 07:55 CEST,
leaving cleanup margin. A timeout is an incomplete run, never a completion.
Freeze the UTC cutoff as an absolute timestamp so Trex's US timezone cannot
alter it. The five-minute cleanup margin also covers the observed remote
clock differences of approximately 20 seconds.

The instruction permits multiple free 5090s during this window, superseding
the older one-weekday-5090 limit for these jobs only. Interpret three runs as
three concurrent workers per 5090, admitted only after a mixed-size timing
check. Nom-cool-1 may host additional work when its measured time fits.
The physical-GPU budget is at most **36 GPU-hours** over four hosts for this
overnight window, including checks and failures. Account shared-GPU workers
by their host's allocation interval, not three times that interval. This is
a separate noise-study budget; it does not raise the clean campaign's cap.

## Initial resource observations

At 22:58 CEST, Trex, Fifi and Loulou each report one idle RTX 5090; Nom-cool-1
reports an idle RTX 3090. Riri fails SSH host-key verification and is excluded.
Jean Zay still times out and is not used. Local GPU time is reserved for the
independently running baseline audit; Akib is not required for this window.

Trex and Loulou have Python 3.12 / PyTorch 2.11.0+cu128 with MNIST staged.
Nom has PyTorch 2.5.1+cu121. Fifi's initially missing environment and MNIST
were copied from Loulou into previously absent paths; both concurrent CUDA
timing packs completed there. All three 5090 hosts now use the same version.

## Admitted overnight packs

| Host | Three concurrent cases, all seed 0 | Estimate including 25% margin |
|---|---|---:|
| Trex | Conv1 ours + legacy at `1e-5`; Conv3 ours at `1e-5` | 8.10 h |
| Fifi | Conv1 ours + legacy at `1e-4`; Conv3 ours at `1e-4` | 6.55 h |
| Loulou | Conv1 ours + legacy at `5e-4`; Conv3 ours at `5e-4` | 6.50 h |
| Nom-cool-1 | Conv2 ours + legacy at `1e-4`; Conv1 ours at `3e-5` | 6.48 h |

This admits **12/30 cells**; **18 remain held for a later window**. It gives
three paired Conv1 noise levels, one paired Conv2 noise level, and an initial
low/middle/high Conv3-ours curve. Conv3 legacy counterparts retain the same
architecture/sigma host assignments for later execution. This is partial
sweep coverage, not a completed ours-versus-legacy comparison for Conv3.
No automatic second wave is included tonight.

The measured timing model accounts for the shorter Conv1 workers finishing
before the long workers, then uses the measured remaining-worker throughput;
it includes validation, five seconds per epoch of overhead, and a 25% margin.
The first mixed Conv1/Conv2/Conv3 timing pack was rejected for deadline risk.
Two deadline-process tests pass, including termination of a real child and
grandchild. Both 12-case CUDA timing rounds completed and are retained as
diagnostics, never as completed training runs. Detailed rates and admission
estimates are in
[`launch/admission.json`](../results/eqprop-read-noise-seed0-ours-legacy-20260914-v1/launch/admission.json).
The immutable production snapshot is `source-v2`, archive SHA-256
`4055090e16606583c276b06c3aac59396e87dcedf58b8cde9be94c5824948fb1`;
the full configs and scientific trainer are unchanged from the first snapshot.
Only transport log timestamps and the diagnostic log interval changed.

## Admission and observability

Prepare all 30 plain exact-run configs and preserve a single source snapshot,
initializer identities, source-result references and the matching-noise proof.
First measure a representative concurrent mixed-size pack through the same
scientific runner; a short diagnostic copy may limit batches for timing but
must be kept separate from full-run configs and results. Smoke each admitted
full config unchanged with the exact runner's one-batch smoke mode.

Use measured throughput plus a 25% margin and model/data/validation overhead
to choose the three-case packs. Do not assume three full Conv3 runs fit:
historical clean Conv3 runs take about four hours each, and earlier packing
reduced throughput. Mixed Conv1/Conv2/Conv3 packs may be faster to finish.
Record the final host/case table and estimates after timing. Keep both
schemes' counterpart cases on the same recorded architecture/sigma target;
unfinished counterparts remain queued for a later authorized window.

All admitted workers have absolute deadline protection, independent logs and
exit receipts, successful-only canonical result bundles, and a duplicate
launch guard. Verify every launch twice through actual artifact progress.
Check GPU/process/artifact/log progress every 30 minutes; reserve no automatic
follow-up beyond the declared overnight packs. Do not start late work that
cannot finish with margin before the cutoff. Preserve any operational or
scientific failure and retry only within the remaining time and scope.

Collect remote results locally, validate their artifacts and full epoch/data
coverage, reconcile planned/completed/failed/unstarted cells, settle host
GPU-hours, and update the result table. All study output lives under
`results/eqprop-read-noise-seed0-ours-legacy-20260914-v1/`; final collected
summaries belong in `paper_ready_results/`. The deadline limits tonight's
admissions, not the declared 30-cell scientific grid.

## Launch receipts

All four packs use resilient `nohup` launchers, with commands in
[`launch/pack_commands.json`](../results/eqprop-read-noise-seed0-ours-legacy-20260914-v1/launch/pack_commands.json).
Remote outputs are `/home/filip/server_code/results/eqprop-read-noise-seed0-ours-legacy-20260914-v1/production/HOST`;
launcher receipts are beside them at `production/HOST.launcher`.

| Host | Launcher PID | Pack PID | Worker cutoff UTC |
|---|---:|---:|---|
| Trex | 1861390 | 1861399 | 2026-09-15 05:55 |
| Fifi | 712270 | 712279 | 2026-09-15 05:55 |
| Loulou | 1160305 | 1160314 | 2026-09-15 05:55 |
| Nom-cool-1 | 1573633 | 1573642 | 2026-09-15 05:55 |

The first live check confirms three intended workers and manifests on every
host; GPU compute utilization is 98–99%, with approximately 6.8 GiB used on
each 5090 and 3.2 GiB on Nom. Compute saturation, rather than memory capacity,
limits additional workers in these packs.

## Noise-stream control audit at 00:15 CEST

A short CUDA probe using the actual float64 layer shapes, both centered
phases, full batch size 16 and final batch size 8, confirms identical sampled
standard-normal arrays on Trex/Fifi/Loulou. Nom's RTX3090/PyTorch2.5.1/CUDA12.1
combination differs from their RTX5090/PyTorch2.11/CUDA12.8 combination despite
the common seed. For the first Conv1 hidden tensor at batch16, 62.7551% of
elements match exactly and the RMS difference is .863064 standard-normal
units; the small output and batch8 tensors match. This is a different sample
assignment, not merely floating-point rounding; the probe does not isolate
hardware from software as its cause.

All ours/legacy pairs are assigned to one host, so this does not break their
within-sigma noise pairing. The isolated Conv1-ours `3e-5` point on Nom is
retained as an independent-noise-realization diagnostic, not represented as
sharing the 5090 curve's exact noise arrays. The initially intended global
cross-sigma noise matching is therefore narrowed to each verified environment
group. No running config, seed, beta, or scientific result was changed.
The provenance is in `checks/noise_rng_cross_host.json`, with the quantified
Conv1 difference in `checks/noise_rng_difference.json`. This limitation must
remain visible in any combined noise-level plot or result interpretation.

## Collection checkpoint, September 15 at 01:08 CEST

All seven admitted Conv1 cases completed their full 10 epochs and are copied
into `paper_ready_results/bundles/read_noise_20260914/`. All canonical, exact
config, source, initializer/cohort/order, noise-draw-count, full-history,
float64/finite/zero-bias/bounds, PT/NPZ, and successful-exit checks pass. Their
final validation losses relative to clean are 0–.06 percentage points.
The two Conv2 and three Conv3 jobs remain active; 18 other grid cases are held.
[Live results and remaining cases](../paper_ready_results/read_noise_run_status_20260914.md)
and the provisional [figure](../paper_ready_results/read_noise_validation_20260915.png)
include the noise-stream limitation. These are partial validation results.

## Final overnight handoff, September 15

All **12/12 admitted trainings** completed and validate locally: seven Conv1,
two Conv2, and three Conv3. All four remote packs, including launcher receipts,
are checksum-identical to the local copy. There were no production failures,
retries, or deadline truncations. The last pack, Trex, finished at 05:52:58 CEST;
all GPU workers have exited. Actual use was **22.142636 physical GPU-hours**,
including .176199 hours for CUDA timing and smokes, against the separate
36-hour allowance. See the [budget receipt](../paper_ready_results/provenance/read_noise_gpu_budget_20260915.json).

Conv1 final losses from its historical clean references are 0–.06 percentage
points. At Conv2 sigma 1e-4, ours/legacy lose .02/1.54 points. Conv3 ours loses
.38/1.02/2.04 points at sigma 1e-5/1e-4/5e-4. These are single-seed validation
measurements under inherited beta/LR/T/K contracts, with the stated Conv3
qualification exception, cross-host noise-stream difference, and historical
clean-reference environment differences. They do not complete the full sweep.

The remaining 18 cells are three Conv1, eight Conv2, and seven Conv3 cases,
listed in the [continuing per-run tracker](../paper_ready_results/read_noise_run_status_20260914.md).
Full results, interpretation, checkpoints, figures, and provenance are linked
from the [overnight report](../paper_ready_results/read_noise_overnight_results_20260915.md).
The clean campaign remains paused; no additional training was launched.
