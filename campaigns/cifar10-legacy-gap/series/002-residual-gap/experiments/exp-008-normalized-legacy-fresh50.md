---
id: "exp-008"
title: "Fresh 50-epoch legacy with normalized block-end voltages"
status: "cancelled"
hypotheses: ["H-003"]
---
# Fresh normalized legacy, 50 epochs

Filip requested a fresh 50-epoch legacy run with normalized end voltages on
September 27. This repeats the established voltage-normalized condition with
legacy-selected optimizer settings; it is not the baseline-optimizer transplant.

One seed0 trajectory, epochs1–50, original L8/45k–5k split, batch32/eval16,
Adam and legacy LR vector, cosine horizon50 to2%, T=K[6,6,4], voltage/current
amplification4/0.25. `block_output_normalization: voltage` divides block-end
voltages by16,16,4 before the digital bridges. No official test access.
Config: `configs/cifar/normalized_legacy_fresh50_20260927.json`.
The recovered epoch0 initializer matches the original normalized-legacy prepared
bundle's checksum; model, optimizer and schedule start fresh, with no trained
checkpoint continuation. Reuse the validated preparation; no new solver audit.

Primary metric: final epoch50 validation accuracy; also report validation CE,
training curve and final native audit. Compare descriptively with the historical
normalized legacy88.82%, raw legacy88.72%, baseline89.48% and ours89.92%.
A repeated seed0 is a reproducibility check, not an independent seed or a matched
new multi-arm comparison. New scientific conclusions require completed coverage.

## Execution and monitoring handoff

Result root: `results/cifar10-l8-normalized-legacy-fresh50-seed0-20260927-v1/`.
Frozen current source/config hashes: `source/source_identity.json`; canonical
preparation: `prepare/`; exact execution smoke: `smoke/<attempt>/`; production:
`cells/legacy_voltage_normalized_fresh50/`. Source/transport changes relative to
historical runs are recorded by the frozen hashes and runtime environment.

One prepared queue job `cifar-normalized-legacy-fresh50-20260927`; eligible RTX5090
hosts Trex/Fifi/Loulou/Riri, standard sharing policy. Peak12,000MiB plus3,072MiB
headroom, one worker/GPU. Expected11h, cap16h including smoke/target validation;
calendar cutoff September30 08:00 Paris. These are lean operational assumptions
for this new assignment, not extensions to earlier CIFAR continuations.
Native epoch admission saves a checkpoint on insufficient remaining time;
no automatic restart from epoch0 or budget reset. A pause requires Codifier review.

The shared run-watch service owns 60-second monitoring and bounded recovery.
Prepared queue specification: `queue-job.json` in the result root. The foreground
helper `experiments/run_cifar_normalized_legacy_fresh.py` performs exact one-batch
smoke, training, coverage validation and controller collection. Completion requires
all50 epochs exactly once,70,350steps, native bundle validation and final audit.
Controller collection is CPU-only and capped at15minutes. No duplicate watch.
Current state: cancelled on Filip's explicit request after confirming that this
repeats the already completed normalized-legacy seed0 setup (historical epoch50
validation88.82%). The first8 epochs match the earlier recorded training loss,
validation accuracy and validation loss exactly; this is not an independent seed.

Queue job `cifar-normalized-legacy-fresh50-20260927` and attempt
`171ada24964b4709b1461dc940a8211b` are confirmed cancelled. The worker is no longer
alive and its GPU reservation is released. Native final status records a
checkpointed pause after epoch17/50; coverage is partial, not a completed50-epoch
result. Existing outputs remain preserved on Trex under
`/home/filip/server_code/results/cifar10-l8-normalized-legacy-fresh50-seed0-20260927-v1/`.
Do not retry or resume this duplicate. No further monitoring or launch is assigned.
