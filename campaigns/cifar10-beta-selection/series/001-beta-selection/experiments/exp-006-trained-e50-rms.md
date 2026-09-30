---
id: "exp-006"
title: "Repeat the blockwise RMS/cosine sweep on original epoch50 BPTT checkpoints"
status: "complete"
hypotheses: ["H-004"]
---
# Trained checkpoint RMS/cosine replay

Filip explicitly requests repeating the measurements on epoch50 BPTT networks,
with a tighter beta search. Test [H-004](../../../hypotheses/H-004-trained-block2-alignment.md)
and whether the initialization targetR≈0.075 transfers to trained states.

Use original seed0 baseline/ours/legacy **final epoch50**, not best checkpoints,
with their saved configs, trainable BN affine, native endpoints and T=K=[6,6,4].
Parents are `results/cifar10-l8-analog-baseline-legacy-e30-e50-seed0-20260922-v1/cells/{baseline,legacy}_e30_e50/`
and `results/cifar10-l8-analog-ours-e30-e50-seed0-20260922-v1/cells/ours_e30_e50/`.
Each source checkpoint's epoch and SHA match exp-006 in the legacy-gap campaign.
Their original epoch0 module/conductance tensors are bitwise equal to the initializer
used in exp-003–005. Saved model contracts agree; older omitted BN/endpoint fields
resolve to the same defaults. Optimizer histories differ by scheme and are not replayed.

Same frozen256 training-image indices, order, preprocessing, no augmentation/noise,
batch32, minibatch BN statistics with buffers frozen. R remains centered physical
output displacement divided by free-state output RMS. Calibrate physical injected B
per block/scheme to targets `[0.005,0.01,0.02,0.035,0.05,0.065,0.075,0.085,0.10,0.125,0.20,0.40]`
within3%, at most24 evaluations each. Recalibration matters because trained states
and CE output forces have changed. Use exp-005 beta guesses, not fixed trained betas.

Measure native and matched-local BPTT cosine, norms, sparsity and per-batch spread
for all8 Conv weights. Also retain local/native reference agreement. Primary score:
minimum pooled native cosine per block, maximized over the12 sampled targets;
report boundary maxima and passing bands at0.95/0.99, plus the shared0.075 target.
Compare with exp-005 without pooling initialization and trained curves. Preserve
undefined/dead responses as failures of alignment; do not exclude them from coverage.
No optimizer steps, training smokes, solver qualification or extra training.

## Execution and monitoring handoff

Root: `results/cifar-block-rms-e50-20260929-v1/`, scheme-specific inputs/source/run/
analysis/launch folders; configs `configs/cifar/block_rms_e50_20260929/`.
Ours/baseline on Loulou5090 sequentially; legacy on Trex5090, sharing with Ben only
when memory fits. Expected4–7min/scheme,1200s hard budget each,3600s combined including
attempts; deadline October1 08:00 Paris. Runtime guards stop nonfinite measurements,
unattainable targets or budget exhaustion; preserve partial evidence without retuning.

Shared queue jobs `cifar-rms-e50-{ours,legacy,baseline}-5090-20260929` own scheduling,
monitoring and CPU collection. Inspect with the installed `gpu_queue.py status JOB_ID`.
Collectors validate canonical bundles, checkpoint epoch50 and unchanged file hashes,
unchanged model/BN,36 calibration targets,96 pooled/768 batch rows and calibration/replay
RMS agreement. Frozen source identities include the small checkpoint-loading extension;
numerical gradient computation is unchanged. Fresh service observations confirm queue
monitoring ownership. Six existing numerical tests passed; only checkpoint loading/
identity and reference-comparison reporting changed.

Adaptive extension after observing the first12-target results: ours block3 and legacy
block2 peak at the lowestR≈0.005. Add `[0.0005,0.001,0.0025]` for all schemes, preserving
the initial grid/evidence. Root `results/cifar-block-rms-e50-low-20260929-v1/`, configs
`configs/cifar/block_rms_e50_low_20260929/`, jobs `cifar-rms-e50-low-{ours,legacy,baseline}-5090-20260929`.
Ours/legacy use Trex sequentially; baseline uses Loulou after its main sweep.
600s cap/extension, counted against the original3600s combined allowance; submit only
when main-job charged time plus600s fits that scheme's1200s allocation. Extension
coverage:9 calibration targets,24 pooled/192 batch rows per scheme. Treat any remaining
boundary maximum as unresolved; do not silently call it an optimum.

All six main/extension queue jobs and local collections are complete. All15 targets
per scheme validate (120 pooled/960 batch rows,45 calibrated block targets), no
failures/exclusions,997.8s combined charged time. GPU reservations released.
[Reviewed result](../results/exp-006-trained-e50-rms.md) records the H-004 contradiction,
changed displacement scales, per-layer effects and per-minibatch limitations.
