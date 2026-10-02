---
id: "exp-016"
title: "Matched early dense-readout learning-rate screen"
status: "complete"
hypotheses: ["H-010"]
---
# Dense-readout LR screen

Filip authorized stopping legacy on nom-cool-2 and ours on Trex to explore dense
readout rates. Baseline on nom-cool-1 continues. Preserve stopped exp-015 bundles
and completed-epoch checkpoints. No smoke, following his standing instruction.

Test [H-010](../../../hypotheses/H-010-frozen-bn-readout-rate.md) using both original
seed0 frozen-affine-BN epoch10 checkpoints from
`results/cifar10-l8-bn-ablation-seed0-20260923-v1/`. This common epoch avoids
confounding the schemes with their different progress at cancellation. Legacy
has unnormalized block-end voltages (`block_output_normalization: none`).

For each scheme restore the same model, Adam moments/steps, scheduler phase and
RNG before three branches: original readout LR, one-third, and multiplier
`ours_original_head_lr / (9 * legacy_original_head_lr)` (about1/27).
The last legacy base rate is exactly5.926787224434896e-6. Apply that same
multiplicative reduction to ours for the specificity comparison. Only
`head.weight` LR and its scheduler base/current rates change. Convolutions,
scalar gains, frozen BN affine, BN statistics policy, seed, data, loss, batch32,
eval16, T/K=[6,6,4], amplification and original50-epoch schedule remain unchanged.

Run400 identical updates from the epoch11 permutation (12,800 examples) in each
branch. Fixed unaugmented evaluation cohorts:2,048 training and1,024 validation
examples, sampled without replacement with seed20260928; evaluate saved-stat BN
at0,100,400 updates. Reuse exp-014's frozen cohort identities. Preserve BN buffers
and training RNG across evaluations. Starting measurements must match exactly
within each scheme. No official test read. Partial-epoch checkpoints are retained
but are explicitly not safe for native epoch-boundary resume.

Primary rule is H-010's exact low-rate legacy-versus-control prediction at400
updates, including specificity versus ours. Report both reductions and all
measurements, including failures; no selecting the best dose to rescue the primary
prediction. The one-third dose and100-update point are secondary evidence.

Root registered before creation: `results/cifar10-dense-readout-screen-20260928-v1/`.
Native runner `experiments.screen_cifar_l8_conv_lr` now accepts explicit head-only
interventions and bounded update counts. Frozen source, inputs, command and hashes
live under `source/`, `inputs/`, and `launch/{legacy,ours}.json`. No new solver/rho audits.

## Execution and monitoring handoff

Run all six branches sequentially on Trex RTX5090, using the same runtime for
both schemes. Expected30–45 minutes; hard GPU cap90 minutes including validation;
CPU collection allowance10 minutes; deadline September29 08:00 Paris. Reserve
12,288MiB plus2,048MiB headroom. Stop on nonfinite, source/restore mismatch, unequal
starting measurements or exhausted budget. Preserve all failures; no blind retry
or automatic longer training.

Shared queue jobs `cifar-dense-readout-legacy-20260928` and
`cifar-dense-readout-ours-20260928` share one Trex sequentially, capped at45 minutes
each. Inspect with `gpu_queue.py status JOB_ID`. The queue is sole monitoring owner.
`python -m experiments.run_cifar_dense_readout_screen collect SCHEME` from frozen
source collects each bundle and validates rates, saved states and all18 evaluation
records. Once both are collected, the completion hook evaluates H-010's frozen
criteria, writes the result note, summary and JPG, updates ideas.md and refreshes
the ledger. No additional model call or longer training is required.
Current handoff: both schemes collected, validated and reviewed against H-010. See the result note.
