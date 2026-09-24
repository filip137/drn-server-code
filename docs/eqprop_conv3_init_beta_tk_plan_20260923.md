# Conv3 initialization beta/T/K replay

User-requested initialization extension of the checkpoint-dependent cosine
comparison, September 23, 2026. Exploratory, read-only ordinary-MNIST
validation replay; no optimizer steps, accuracy evaluation, or official test.

- Baseline, ours, legacy, all at the same saved seed-0 initialization, epoch0.
  Reuse the hash-verified initializer from the earlier initialization replay.
  Training source configs supply the unchanged architecture and explicit
  diode/zero-bias contract; initialization has no training beta or noise history.
- Injected beta .987333678708 and5.26875648112, T8/16/32, K8/32:36 cases.
  These betas directly match the p99 and p90 ours training values. They are
  common replay probes, not training-beta nominations for all schemes.
- Reuse the exact36 batches of16 validation examples and source indices,
  clean endpoints and four matched draws at sigma5e-4 with seeds
  2026092101/2026092111/2026092121/2026092131. Expected25,920 layer comparisons.
- Float64 centered frozen-current EqProp, input gain360, exact-zero biases,
  original [0,100] weights and paired20-output squared loss.
- Cosine medians and10–90% spread for every weight matrix; paired per-batch
  changes (average noise draws within batch), sign and prevalence of changes.
  Noise draws are not independent training seeds. Also preserve RMS, norm
  ratios and near-zero fractions from the existing runner.
- Matching-K BPTT plus fixed K32 BPTT and clean-EqProp references within each
  T; compute K32 first. Verify identical post-T state/force across K and beta,
  BPTT invariance across beta, unchanged parameters and input bytes.
  Changing T changes the starting state; K32 is a finite reference.
- Local same-config smoke: one original batch at beta5.26875648112 for every
  scheme/T/K context (18 cells), with all guards and canonical outputs.
  The grid is an explicit diagnostic override, not a new training handoff.

Target: Fifi RTX5090, PyTorch2.11.0+cu128, Python
`/home/filip/miniconda3/envs/py312/bin/python`. Inventory checked before
placement: Fifi/Akib idle, local available for smoke; nom-cool-1, Trex, Riri,
Loulou occupied; own Jean Zay queue empty. Recheck Fifi immediately before
launch. Use an isolated workspace under the result directory.

Expected25–40min; hard3600s combined smoke/replay budget. Per-batch status
heartbeat; check first semantic output and monitor to the deadline, at least
every30min. Collect and validate local outputs before concluding.

Config: `configs/conv/eqprop_conv3_init_beta_tk_20260923.json`.
Command: `python -m experiments.replay_conv3_trained_beta_noise --config
configs/conv/eqprop_conv3_init_beta_tk_20260923.json --device cuda`.
Smoke appends `--smoke --target local:RTX3090`.
Results: `results/eqprop-conv3-init-beta-tk-20260923-v1`.

Completion requires all36 cases and18 smoke bundles, matched cohort and
immutable source checks, locally validated collection, matplotlib figure,
paired-batch CSV, report, and an update to the beta notes and manifest.
Compare with the earlier trained-checkpoint results while preserving their
different replay environments and available beta/T/K coverage.

Launch: all18 local GPU smoke cases pass in76.25s, including initialization
tensor identity and K32 references;12 focused tests pass. All606 frozen
input/source hashes verify; dashboard registration is checked separately. Fifi was rechecked idle and production launched
in tmux `conv3-init-beta-tk-20260923`. One staging copy initially named a
nonexistent namespace-package marker; it was removed from the transfer list
and staging completed before launch. No scientific attempt failed.
A prelaunch hash check caught a concurrent update to the live dashboard;
that mutable metadata was excluded from the frozen scientific-file hash set
and its study registration was checked directly. Production then launched.

Verified first live GPU worker and canonical artifact progress: launch
2026-09-23T11:49:18Z, supervisor2583806, timeout2583832, Python2583833.
The first full cells complete with36 batches each; no failure is present.

## Completion

All 36 production cells and 25,920 gradient comparisons completed. All 18
smoke and 36 production bundles validate in the authoritative local copy;
509 collected file hashes and 606 frozen source/input hashes match. Fifi
exited 0 and released its GPU. Production took 1851.02 seconds; including
smoke, 1927.27 seconds is within the 3600-second budget. No scientific case
failed or was excluded. Partial analyses are superseded by the complete report.

There is no meaningful T/K dependence at initialization: maximum noisy
layer-median changes are 5.87e-8 across T and 6.41e-6 across K. The maximum
paired-batch change is 2.35e-5. The same-batch comparison with the older ours
checkpoint supports emergence of its strong sensitivity after initialization;
the onset epoch remains unknown. See the
[report](../results/eqprop-conv3-init-beta-tk-20260923-v1/analysis/report.md),
[beta notes](beta_study.md#initialization-check-at-the-same-replay-betas), and
[manifest](experimental_manifest.md#conv3-initialization-betatk-replay-september-23).
