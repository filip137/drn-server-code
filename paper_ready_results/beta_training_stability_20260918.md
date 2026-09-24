# Larger-beta EqProp: ten-epoch stability pilots

Updated 2026-09-18T22:27:17.261679+00:00. **9/9 terminal; 3 passed ten epochs; 6 nonfinite training failures.**

Seed 0; RTX5090 GPUs on Riri (two workers) and Trex (Conv3 ours only); clean endpoints; original Adam rates; T/K=4/6/8.
Conv3 ours was moved before training when Trex became free. Its Trex/Riri smoke losses and gradient/update norms match exactly; the frozen config and initializer are unchanged. Host placement is recorded separately from the scientific cases.
Betas were selected using whole-gradient cosine >=.99 plus norm mismatch <=.10.
These are early stability diagnostics, not paper test accuracies or full 30-epoch qualifications.

| Model | Scheme | Injected beta | Outcome | Best/final validation | Control epoch-10 |
|---|---|---:|---|---|---|
| conv1 | baseline | 5000 | Nonfinite: epoch 1, batch 376 | — | — |
| conv1 | ours | 1500 | Nonfinite: epoch 1, batch 546 | — | — |
| conv1 | legacy | 900 | Nonfinite: epoch 1, batch 1696 | — | — |
| conv2 | baseline | 1000 | Nonfinite: epoch 7, batch 813 | Before failure: 96.48/93.92% | — |
| conv2 | ours | 100 | Nonfinite: epoch 3, batch 481 | Before failure: 95.96/84.74% | — |
| conv2 | legacy | 30 | Nonfinite: epoch 5, batch 1910 | Before failure: 97.18/97.08% | — |
| conv3 | baseline | 750 | 10-epoch stable | 97.08/97.08% | 97.02% (beta 100) |
| conv3 | ours | 22.5 | 10-epoch stable | 98.38/98.38% | 98.42% (beta 3) |
| conv3 | legacy | 1 | 10-epoch stable | 98.64/98.64% | 98.64% (beta 0.001) |

All nine outcomes are locally collected and their canonical bundles validate; no scientific failures are excluded.
Final validation differences from the matching smaller-beta epoch-10 controls: conv3 baseline: +0.06pp; conv3 ours: -0.04pp; conv3 legacy: +0.00pp.
The successful candidates pass the early stability criterion, but this single-seed clean comparison does not establish an accuracy improvement or longer/noisy training stability.

Stability requires ten finite epochs and a final validation drop strictly below 5pp from the run's best.
Accuracy relative to the smaller-beta control is a separate question; finite poor learning is not evidence of satisfactory performance.
Failed runs are retained and are not retried with changed scientific parameters.

The observed failures show that clean gradient agreement at two fixed checkpoints does not guarantee stability along a new training trajectory.
The whole-gradient criterion can mask layerwise errors; this pilot does not isolate that mechanism from finite-nudge effects during training.
Conv1 ours at beta 1500 also passes the 0.95 per-matrix cosine rule with the norm gate: worst matrix cosine 0.951337 and norm mismatch 0.084590. Its epoch-1 failure also rejects this particular looser per-matrix candidate; this is not a training test of every 0.90/0.95/0.99 selection.

[Diagnostic figure](beta_training_stability_20260918.png) · [PDF](beta_training_stability_20260918.pdf)
Sampled gradient norms are absolute; their scales differ across schemes. Sparse samples may miss the final increase immediately before a failure.

[Launch plan](../docs/eqprop_beta_training_stability_plan_20260918.md) · [Local evidence](../results/eqprop-beta-training-stability-20260918-v1/)
