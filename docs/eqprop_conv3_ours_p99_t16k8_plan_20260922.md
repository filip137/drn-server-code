# Conv3 ours p99 at T16/K8: ten-epoch diagnostic

Filip requested p.99 beta, ten epochs, T=16 and K=8, then corrected the
scheme from baseline to ours. This authorizes one fresh-initialization run.
The existing read-noise sigma5e-4, Adam vector and seed0 contract are retained.

## Frozen scientific case

- Conv3 ours: voltage amplification4/current amplification1, input gain360,
  explicit perfect diodes, paired20-output squared loss, weights[0,100],
  frozen exact-zero biases and true float64 centered frozen-current EqProp.
- Injected beta0.987333678708 (base beta0.0154270887298125), the existing p99
  choice. Here p99 means every matrix cosine>.99 over the original72 clean
  initialization/BPTT-checkpoint replay batches, not a percentile.
- T16/K8 and exactly10 epochs. Fresh saved seed0 initialization and optimizer;
  original deterministic55,000/5,000 MNIST split and minibatch ordering.
- Endpoint read noise sigma5e-4, seed2026081601; inherited Adam weight rates
  unchanged; batch16 and validation batch64; no official-test read.
- Config: `configs/conv/eqprop_conv3_ours_p99_t16k8_10ep_20260922_v1/conv3_ours_p99_sigma5em4_t16k8_10ep_seed0.json`.

The p99 calibration was at T8/K8. T16 is a user-directed exploratory change,
not a new p99 calibration or a paper-qualified T/K handoff. Existing matched
T/K cosine replay supports investigating this phase length; it does not certify
training stability at this beta. No rho search or beta reselection is added.

## Execution

One Jean Zay V100 via `fmu@v100`, `gpu_p13`, `qos_gpu-t3`, v100-16g,
one GPU/task, four CPUs, three-hour limit. Expected runtime1.5–2hours.
All5090 hosts and the local/nom-cool-1 GPUs were occupied at placement;
Akib was idle but V100 is preferred for this float64 Conv3 training run.
The allocation and source/dataset paths are verified before submission.

Use the unchanged proven p90 continuation runtime source, commit
`022098cfe6eb151dff3f7403d3f4b358709152b9`, archive SHA256
`74f41a03ce4718eebcd227bb38e4bbd2309e6b30bb9dd5515c099bb880068819`.
Run `experiments.exact_run` with the same config in a synchronous local CPU
one-train/one-validation-batch smoke, then production. Repository AGENTS
explicitly selects this local-smoke/one-submission policy over the skill's
default remote-canary sequence. Previous successful V100 runtime/module
contract is reused; the earlier p99 canary monitoring SSH timeout was not a
worker/environment failure. Check wrapper syntax and `sbatch --test-only`.

Local authoritative root:
`results/eqprop-conv3-ours-p99-t16k8-10ep-20260922-v1`.
Remote output root:
`/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/eqprop-conv3-ours-p99-t16k8-10ep-20260922-v1`.
Record the jobID, command, smoke result, environment, logs and canonical bundle.
Verify scheduler/resources and first semantic progress after submission.
Monitor artifact growth and errors every30minutes, with three-hour execution
deadline; record queue delays separately. Recover understood operational
failures only. Scientific nonfinite outcomes are retained without retuning.

## Completion and interpretation

Collect and validate the local run bundle. Report all ten epoch validation
accuracies/losses, best and final validation, numerical finite outcome and
final drop from best (<5pp is the existing separate stability screen).
Retain any scientific failure and its epoch/batch. No ten-epoch result is a
thirty-epoch stability claim. Compare T8 only if its existing matching run
is collected and validated; GPU/environment differences must remain explicit.
Curate the outcome in the experimental manifest and update the persistent
directory row. No baseline arm is included after Filip's correction.

## Placement update before submission

The three-hour V100 test-only estimate was September28, and A100 was later.
Use two sequential five-epoch segments on fmu@v100/gpu_p13/qos_gpu-dev, each
90minutes, total reserved GPU budget3hours. The unchanged exact runner's
`--epoch-chunk-size 5` retains weights, optimizer, RNG, data order and noise
draw state; configured epochs remain10. No science is reset between jobs.
Before production, the same-config local continuation smoke exercises three
one-epoch segments (two train batches and one validation batch per segment)
and its existing resume/source guards. There is no remote canary. Both
planned segments are covered by the existing persistent row; record each
Slurm jobID immediately upon submission. Production segment1 is released
only after segment0's successful five-epoch checkpoint.

Launched: jobs73267 and73276 (`afterok:73267`) on September22. First segment
running on r10i4n8 with first optimizer step observed21:11UTC. The two same-config
local smoke paths validate; exact-run tests17/17 and continuation tests5/5 pass.
The collected T8 p99 reference bundle validates; its epoch10 validation is95.44%
and best in epochs1–10 is95.64%. Scientific config differences are only T and
terminal epoch horizon; all optimizer/noise/model fields match.

## Completed result, September23

All ten epochs completed with finite metrics and checkpoints. Final validation
is95.38%; best95.58% at epoch9. The matched T8 first-ten-epoch reference gives
95.44% final and95.64% best, also at epoch9. The epochwise accuracy gap never
exceeds0.12pp, so this run shows no validation benefit from doubling T. Both
finals are0.20pp below their own best. This is ten-epoch, single-seed evidence;
it does not qualify a larger beta or a thirty-epoch stability boundary.

Jobs73267/73276 exited0:0 in3258/3281seconds, total1.8164GPUh of the3GPUh cap.
The source/config/initialization/split/order/noise-count checks pass. All four
local bundles (production, reference and two smokes) validate. Remote file
checksums match; only the study directory timestamp differs because local
analysis files were added. No scientific failures, retries or pending jobs.

[Report](../results/eqprop-conv3-ours-p99-t16k8-10ep-20260922-v1/analysis/report.md) ·
[Collection validation](../results/eqprop-conv3-ours-p99-t16k8-10ep-20260922-v1/collection_validation.json).
