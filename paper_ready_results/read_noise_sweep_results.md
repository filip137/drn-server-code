# Single-seed EqProp read-noise sweep

Completed September 16, 2026, **06:20:46 CEST**. **All 30/30 full trainings
are collected and validated.** No sweep run remains active or queued.
All five continuation GPUs were released before the 08:00 deadline.

These are ordinary-MNIST validation measurements on the deterministic
5,000-example validation cohort. Official-test evaluations remain zero.
Every number below comes from a completed 10- or 30-epoch training; short
smokes and timing probes are excluded.

## Final validation accuracy

All entries are percentages. The clean column is the previously collected
seed-0 reference; no new zero-noise training was run. Best-checkpoint
accuracies and individual result links are in the
[full tracker](read_noise_run_status_20260914.md) and
[CSV](read_noise_run_status_20260914.csv).

| Architecture | Scheme | Clean reference | 1e-5 | 3e-5 | 1e-4 | 3e-4 | 5e-4 |
|---|---|---:|---:|---:|---:|---:|---:|
| Conv1 | ours | 96.34 | 96.34 | 96.30 | 96.34 | 96.30 | 96.28 |
| Conv1 | legacy | 96.44 | 96.40 | 96.40 | 96.40 | 96.36 | 96.38 |
| Conv2 | ours | 98.02 | 98.08 | 98.08 | 98.00 | 97.80 | 97.60 |
| Conv2 | legacy | 98.06 | 97.62 | 97.38 | 96.52 | 95.94 | 95.72 |
| Conv3 | ours | 98.64 | 98.26 | 97.82 | 97.62 | 96.82 | 96.60 |
| Conv3 | legacy | 98.78 | 95.68 | 93.42 | 83.08 | 82.34 | 75.44 |

[Accuracy-loss figure: PNG](read_noise_validation.png) ·
[PDF](read_noise_validation.pdf) ·
[Collected bundles](bundles/read_noise_20260914/)

## Interpretation of the completed coverage

Conv1 changes little across this grid: every final result is within .08
percentage points of its historical clean reference, at most four validation
examples. This single seed does not establish statistical equivalence.

Conv2 shows better observed robustness for ours at every tested noise level.
At 5e-4, ours loses .42 points from its clean reference and legacy loses
2.34 points. Every within-sigma Conv2 ours/legacy pair shares one host,
initializer, complete minibatch order and noise stream. These are comparisons
under the inherited beta, learning-rate and T/K contracts; they do not isolate
amplification from the different beta choices.

Conv3 also shows a larger observed legacy loss at every noise level. Across
the five increasing sigmas, ours loses .38/.82/1.02/1.82/2.04 points and
legacy loses 3.10/5.36/15.70/16.44/23.34 points. At 5e-4, ours finishes at
96.60% and legacy at 75.44%; their best checkpoints achieve 96.78% and
76.36%. Legacy's high-noise training fluctuates strongly: at 5e-4 its
validation accuracy spans 31.26–76.36% over 30 epochs. All runs complete
with finite checkpoints; this instability is a scientific outcome, not an
operational failure. All Conv3 comparisons cross environment/noise-stream
groups, so they support a descriptive robustness difference under these
contracts, not a controlled attribution to amplification alone.

## Fixed protocol and comparison limits

The sweep uses wide [0,100] perfect-diode networks, frozen exact-zero biases,
centered frozen-current float64 EqProp, Adam with unchanged parent learning
rates, training seed 0 and noise seed 2026081601. Epochs are 10/30/30 and
T/K is 4/4, 6/6 and 8/8 for Conv1/2/3. Independent Gaussian endpoint-voltage
noise affects gradient readout only; inputs, relaxation and validation stay
noiseless. Sigma is in simulator voltage units, without hardware calibration.

| Architecture | Ours beta | Legacy beta |
|---|---:|---:|
| Conv1 | 30 | 3 |
| Conv2 | 10 | .03 |
| Conv3 | 3 | .001 |

Conv3 ours retains the recorded clean-gradient qualification exception for
beta 3. This noise sweep does not resolve the broader beta-selection audit.
Only one training/noise seed is measured, so seed-to-seed uncertainty is
unmeasured.

Local and Nom reproduce the audited noise samples; Akib's 3080 and the
5090/software group produce two other streams. Lines in the figure join only
points within one stream group. All Conv1/2 pairs share a host. Conv3 at
1e-5/1e-4/5e-4 compares earlier ours/5090 with legacy/3090. For the 08:00
deadline, legacy at 3e-5/3e-4 moved to Trex/Fifi, whereas their ours runs
continue on local/Akib. All five Conv3 pairs therefore differ in environment
and noise realization. Clean references were trained earlier on Jean Zay under
PyTorch 2.5.0. Their initialization and full data/order records match, but
small clean-relative changes cannot be attributed uniquely to read noise.

## Evidence and completed execution

All included local bundles pass full-epoch coverage, canonical validation,
unchanged source/config checks, initializer/cohort/order matching, expected
noise-draw counts, finite float64 checkpoints, zero biases, [0,100] bounds,
PT/NPZ equality and successful worker receipts. The scientific source is
`source-v2`, SHA-256
`4055090e16606583c276b06c3aac59396e87dcedf58b8cde9be94c5824948fb1`.

The final run finished on Nom at 06:20:46 CEST, **1 hour 39 minutes before
the deadline**. Local/Akib finished at 01:23:12/01:27:57; Fifi at about
03:37 and Trex at 03:57:42. Final worker, queue and GPU checks show no
remaining sweep process. All four remote continuation output trees and
terminal receipts match their local copies by checksum; the first-window
reconciliation remains preserved separately.

The continuation used **55.089836 physical GPU-hours of its 78-hour
allowance**: 54.748831 for the 18 full trainings and .341005 for CUDA checks,
noise probes and a conservative .1-hour pretraining recovery charge.
Together with the first window's 22.142636 hours, the full sweep used
**77.232472 physical GPU-hours**. Intervals are counted once per physical
host; simultaneous workers do not multiply the charge.

One local nohup attempt ended before any training artifact; its unchanged
tmux replacement completed, and the original attempt is preserved. Local
and Akib's queues intentionally exited 1 at the recorded output guards after
their retained full trainings completed, preventing duplicate admission of
the cases transferred to Trex/Fifi. Every full training pack exited 0.
There are no excluded, discarded, unfinished or retried full trainings.

- [Continuation plan and current handles](../docs/eqprop_read_noise_continuation_plan_20260915.md)
- [Collection proofs](provenance/read_noise_collection_20260914.json)
- [Final remote checksum reconciliation](provenance/read_noise_final_reconciliation_20260916.json)
- [Final continuation and total GPU accounting](provenance/read_noise_gpu_budget_continuation_20260916.json)
- [Study source, logs, checkpoints and monitoring snapshots](../results/eqprop-read-noise-seed0-ours-legacy-20260914-v1/)
- [Preserved first overnight-window report](read_noise_overnight_results_20260915.md)

The clean training campaign remains paused. Baseline, extra seeds, sigma 0,
sigma 1e-3 and official-test evaluation are outside this sweep.

The requested sweep is complete. A future causal comparison would first
resolve the inherited beta qualification issue and match Conv3 environments
and noise draws; additional seeds would then measure training variability.
These are interpretation limits and possible follow-ups, not newly launched
or required runs in this completed 30-case contract.
