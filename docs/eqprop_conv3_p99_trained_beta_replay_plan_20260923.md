# Conv3 p99 trained-checkpoint beta replay

Exploratory mechanism diagnostic requested September 23, 2026: explain the
legacy validation advantage in the completed p99 sigma5e-4 runs. Replay only;
no optimizer steps, new training, accuracy evaluation, or official-test reads.

- Checkpoints: legacy final/best epoch30 (96.86% validation), ours final
  epoch30 (96.04%) and ours best epoch27 (96.32%). All trained on V100, seed0.
- Injected beta: .001, .01, .1, .3, .987333678708, 3, 10. These include both
  training values. Do not add a nearly duplicate beta1. Report base beta too.
- T=8 and16, K=8 for both EP and BPTT. The accepted T8/K8 operating point is
  inherited; T16 is an explicit diagnostic override. Report residual failures
  without treating this replay as a training requalification.
- Reuse the existing36 validation batches of16 (576 samples), saved source
  indices/order and preprocessing. Float64 centered frozen-current EqProp,
  same post-T state for both estimators, exact-zero biases, explicit source
  diode dictionaries, input gain360, paired20-output squared loss.
- Clean endpoints plus four matched independent endpoint-noise draws at
  sigma5e-4, with existing seeds. Physical dynamics stay clean. Every weight
  tensor and checkpoint hash is checked unchanged; BPTT must be identical
  across beta within checkpoint/T/batch.
- Layerwise cosine, norm ratio, relative error, RMS, near-zero fraction,
  added-noise magnitude, phase displacement, diode activity, and projected-KKT
  hidden/raw-output residuals. Summarize medians and10–90% ranges across
  batches/draws; these are descriptive spreads, not confidence intervals.

All42 cells and30,240 weight-layer comparisons must be accounted for.
Smoke: first original batch at minimum, both training values, and maximum
beta for each checkpoint/T (24 cells), excluded from analysis. Same runner
and frozen config locally on RTX3090, then one remote GPU worker.

Target: idle Fifi RTX5090, PyTorch2.11.0+cu128, Python environment
`/home/filip/miniconda3/envs/py312`. Other GPU hosts and Jean Zay checked;
Akib was idle, nom-cool-1/trex/riri/loulou occupied; no active Jean Zay jobs.
Local GPU is accessible outside the sandbox and idle. Recheck Fifi immediately before admission.
All science stays on Fifi; isolate source and required inputs beneath
`/home/filip/server_code/results/eqprop-conv3-p99-trained-beta-replay-20260923-v1/workspace`.
Expected30–60min; hard7200s combined smoke/replay budget, with a process
timeout. Per-batch heartbeat; inspect launch and first semantic artifact,
then at least every30min through completion. Preserve failed attempts.

Config: `configs/conv/eqprop_conv3_p99_trained_beta_replay_20260923.json`.
Command: `python -m experiments.replay_conv3_trained_beta_noise --config
configs/conv/eqprop_conv3_p99_trained_beta_replay_20260923.json --device cuda`.
Local smoke appends `--smoke --target local:RTX3090`.

Local result root: `results/eqprop-conv3-p99-trained-beta-replay-20260923-v1`.
Collect remote outputs beneath `collected/`, validate canonical bundles and
coverage, then generate matplotlib figures and analysis/report.md and curate
the experimental manifest. No missing cases may be hidden by aggregation.

Interpretation: separate clean finite-beta error, noisy signal limitations,
and sensitivity to T. Compare each scheme at its training beta and on the
common grid. Check whether ours' best checkpoint changes the conclusion.
Report sampled boundary optima. These trained weights and scheme-specific
learning rates already differ, and replay cannot establish that changing
beta during training would close the accuracy gap. No initialization replay
is included; gradient evolution from initialization is outside this scope.

Operational note: base Python lacked dependencies before any computation.
The project CPU smoke completed one cell but took80s per batch; it was
interrupted for GPU smoke. Its complete/failed cells are retained under
`stopped-cpu-smoke/` and excluded. Charge100s for that attempt. Its target
label inherited Fifi incorrectly; its runtime host/device correctly record
local CPU. The replacement explicitly records local:RTX3090.

Local GPU smoke completed24/24 cells in88.32s; all canonical bundles passed.
The same runner/config launched on Fifi at08:05:59UTC, tmux
`p99-beta-replay-20260923`, launcher2557815, worker2557822. The first complete
36-batch cell took36.90s and GPU use was verified. Ten targeted tests pass.
Existing training histories and eight recorded trace minibatches per epoch
are summarized as context, with source hashes and matched source-index checks;
these are existing measurements, distinct from the36-batch validation replay.

## Completion

All42 production cells and30,240 gradient comparisons completed on Fifi;
all24 local GPU smoke bundles pass. The593 remote output hashes match the
authoritative local copy, all parent/source hashes remain unchanged, and no
scientific case is missing or excluded. All24,192 residual records pass.
Production elapsed1560.60s, total charged1748.92s within7200s; remote exit0
and the worker released the GPU. Import failure and stopped CPU smoke remain
preserved/excluded. Full analysis supersedes all partial summaries.

The supported interpretation and limitations are in
[the report](../results/eqprop-conv3-p99-trained-beta-replay-20260923-v1/analysis/report.md)
and the [experimental manifest](experimental_manifest.md#conv3-p99-trained-checkpoint-beta-replay-september-23).
No training or further beta points were launched.
