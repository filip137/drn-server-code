---
experiment: "exp-004"
evidence: "validated-local"
summary: "The common R≈0.075 target gives cosine >=0.95 in baseline and legacy as well as ours; all-scheme overlap is roughly0.04–0.11 pooled and0.05–0.09 for every measured minibatch."
verdicts: {"H-003": "supports"}
---
# Result: baseline and legacy displacement sweeps

Both new schemes completed all12 targets x8 Conv weights x8 batches:96 pooled and768 batch rows each, with36 block/target calibration histories per scheme. Local bundles/coverage validate, calibrated and gradient-replay RMS values agree, and all model/BN tensors remain unchanged. No failed or excluded cases. Baseline took235.8s on Loulou5090; legacy246.2s on Trex5090. Each stayed within its1200s budget. Ours is reused from exp-003, not rerun.

Matched scope: identical original seed0 tensors and256 training images, batch32, no noise/augmentation, T/K=[6,6,4]. Baseline V1/C1; ours V4/C1; legacy V4/C0.25. All use native block_output_normalization=none. Each scheme has its own calibrated physical B and its own native/matched-local BPTT reference. This is an initialization comparison under fixed finite iteration counts.

At nominal relative displacement R=0.075, minimum pooled native-BPTT cosine across the Conv weights in each block:

| Scheme | Block 1 | Block 2 | Block 3 | Worst measured minibatch/layer |
|---|---:|---:|---:|---:|
| baseline | 0.995267 | 0.992865 | 0.996944 | 0.970602 |
| ours | 0.997933 | 0.964795 | 0.996350 | 0.977892 |
| legacy | 0.995565 | 0.971041 | 0.995899 | 0.960516 |

Injected B values and actually measured R for that target:

| Scheme | B, block 1 | B, block 2 | B, block 3 | Measured R, blocks 1/2/3 |
|---|---:|---:|---:|---|
| baseline | 0.02653276545 | 0.002334463689 | 0.01485938425 | 0.074248, 0.074375, 0.074588 |
| ours | 0.0045817721 | 0.0008414149986 | 0.01704487339 | 0.075523, 0.075539, 0.075614 |
| legacy | 0.02165096517 | 0.002180988063 | 0.03395259243 | 0.074230, 0.074368, 0.074505 |

Sampled high-cosine bands shared by all three blocks within a scheme:

| Scheme | Pooled cosine >=0.95 | Every measured minibatch >=0.95 | Pooled cosine >=0.99 |
|---|---|---|---|
| baseline | 0.0200–0.1616 | 0.0399–0.1084 | 0.0299–0.1084 |
| ours | 0.0401–0.1110 | 0.0401–0.0908 | None in tested range |
| legacy | 0.0399–0.1326 | 0.0498–0.0892 | None in tested range |

Across all nine scheme/block combinations, the common >=0.95 band is approximately R=0.040–0.111 for pooled gradients and R=0.050–0.089 when every measured minibatch must pass. Matched-local BPTT gives the same broad conclusion. Spans connect neighboring tested passing samples; neither exact cutoffs nor every intervening point are certified.

Review: supports H-003. The previously selected R≈0.075 target transfers to both new schemes without changing T/K or parameters. Baseline has the broadest high-alignment interval here and meets pooled >=0.99 near the selected target. Ours and legacy are limited by block2, and no common >=0.99 interval exists across all schemes. This is a gradient-fidelity observation; it does not establish a training accuracy ranking. Different curve shapes remain, so R is a useful operating target rather than a complete predictor of alignment. Absolute RMS also has overlapping bands; this test does not uniquely establish one scaling convention.

Decision: retain R≈0.075 as a shared candidate for the planned10-epoch training comparison. Each scheme/block needs its own B. Recheck how R evolves during learning before treating these initialization betas as a permanent normalization rule; no training was launched by this experiment.

Evidence: `results/cifar-block-rms-schemes-20260929-v1/baseline/run/`, `results/cifar-block-rms-schemes-20260929-v1/legacy/run/`, each scheme’s `analysis/summary.json` and `analysis/batch_summary.json`, and `results/cifar-block-rms-schemes-20260929-v1/analysis/comparison.json`. Configs, source/input identities, full calibration histories and queue job IDs are linked by the experiment handoff.

[Cross-scheme RMS/cosine figure](../../../../../results/cifar-block-rms-schemes-20260929-v1/analysis/scheme_rms_cosine.jpg)
