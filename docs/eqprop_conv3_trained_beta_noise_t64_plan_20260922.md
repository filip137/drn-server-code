# Trained-checkpoint beta/read-noise replay at T64

Filip requested higher-T graphs and explicitly authorized parallel machines
and a smaller beta grid. Repeat the same six final epoch-30 p90 checkpoints
(baseline/ours/legacy, training sigma3e-4/5e-4) with T=64 and unchanged K=8.
Filip clarified that cosine changes are the scientific objective and requested
a narrower beta interval. Residuals may remain recorded by the existing runner
but do not select points or gate this diagnostic. This is a diagnostic override,
not a new training qualification or a K-convergence claim.

Five beta factors relative to each source training beta are
`[.1,.3,1,3,10]`, a two-decade interval around the training value. Use the same36 validation batches of16,
four matched read-noise seeds, float64 centered frozen-current estimator,
explicit perfect diodes, original weights and exact-zero biases. No training,
accuracy evaluation or official-test read occurs. Expected coverage:30 cells,
1,080 phase replays and21,600 layer comparisons, plus excluded smokes.

Every host processes all six checkpoints at its assigned beta factors:

| Target | Shard index/count | Beta factors |
|---|---|---|
| Trex RTX5090 | 0/3 | .1,3 |
| Loulou RTX5090 | 1/3 | .3,10 |
| Akib RTX3080 | 2/3 | 1 |

This user-authorized parallel placement overrides the earlier single-host
placement. Host assignment is shared across schemes at each beta. Check live
occupancy and require no compute worker immediately before each launch.
Keep one identical config and runner, with CLI target/data-root/shard transport
arguments. Use identical input indices, source hashes and resolved scientific
settings on every target. Trex and Loulou use Torch2.11/CUDA12.8; Akib uses
Torch2.5.1/CUDA12.1. Before production, run the same18 one-batch smoke cells
on each host at factors .1/1/10, and compare the common training-beta cell for all six cases:
absolute cosine difference <=1e-7 and relative EP/BPTT norm differences <=1e-6.
If parity fails, diagnose and retain results; do not silently mix incompatible
implementations. Record actual errors. Noise draw identity is checked within
each shard; source beta, cohort and BPTT reference remain invariant.

The direct command is `python -m experiments.replay_conv3_trained_beta_noise
--config configs/conv/eqprop_conv3_trained_beta_noise_t64_20260922.json
--shard-index INDEX --shard-count 3 --target TARGET --data-root DATA_ROOT`.
Add `--smoke` for the common one-batch gate. Use the same scientific runner
and source snapshot for smoke and production. Each worker has a1,800s cap
including smoke; total maximum is5,400s (1.5GPUh). Expected wall duration15–20min.
Monitor every minute while active. Stop only our workers if an operational
failure or unrelated GPU occupancy invalidates their placement.

Authoritative local output is
`results/eqprop-conv3-trained-beta-noise-t64-20260922-v1/`.
Collect each host under `collected/HOST/`, validate complete disjoint coverage,
and expose the30 unique run directories together for analysis. Retain host
identities and reports; never overwrite a duplicate cell. Redraw cosine and
norm-ratio curves and add a matched T8/T64 comparison on this reduced beta
grid. Focus conclusions on layerwise clean/noisy cosine changes. Keep any
recorded residual metadata as ancillary diagnostics. Record conclusions in
`docs/experimental_manifest.md`.

## Admission and smoke outcome

The initial Trex/Loulou preflight rejected low-memory CUDA MPS servers, before
any scientific computation. Those daemons belong to another user and their
private client lists are inaccessible. They remain untouched. Revised admission
allows only MPS daemon entries using at most128MiB, requires no other compute
worker, GPU utilization <=5% and total memory <=512MiB. Both hosts showed0%
utilization and under160MiB total memory. Continue monitoring for new workers
or increasing MPS allocation; do not interfere with unrelated workloads.

All54 smoke bundles passed and validate locally. Across240 matched layer/readout
comparisons at training beta, the largest cosine discrepancy versus Akib is
2.22725e-10, EP norm relative discrepancy4.83388e-9, and BPTT norm relative
discrepancy7.01096e-15. All are below the predeclared tolerances. Smoke runtimes
are26.60s Trex,24.42s Loulou and50.19s Akib. Config and runner hashes match.

## Completed coverage and interpretation

All 30 declared production cells completed: Trex 12, Loulou 12, Akib 6. All
three launchers exited 0 and released their workers. The authoritative local
copy validates 30 production and 54 smoke bundles, 21,600 production layer
comparisons, all 36 matched batch identities and all 1,194 remote output
hashes. All 24 checkpoint source files remain unchanged. Total runtime
including smoke was 1,856.73 seconds (0.516 GPUh), below the 5,400-second cap.
There are no missing or excluded scientific cells. The two initial admission
failures and all smoke logs are retained under `launchers/`.

At the five matched beta factors, legacy's noisy layer median cosines change
by less than 3.5e-8 from T8 to T64. Baseline and ours/sigma 3e-4 also change
little. Ours/sigma 5e-4 changes substantially: at the training beta, its noisy
Conv1/Conv2/Conv3/readout medians change from [.0201,.0258,.2457,-.2932] to
[.0803,.1117,.2890,.8854]. The comparison keeps K fixed and does not identify
the minimum T for any cosine threshold. The narrowed beta grid is the final
declared scope; no further sweep is pending.

[Curated interpretation](experimental_manifest.md#conv3-noisy-cosine-changes-from-t8-to-t64-september-22)
· [Final report and plots](../results/eqprop-conv3-trained-beta-noise-t64-20260922-v1/analysis/report.md)
· [Collection validation](../results/eqprop-conv3-trained-beta-noise-t64-20260922-v1/collection_validation.json).
