# Current State

Last updated: 2026-08-17

## Purpose

This document states why the repository exists, what research belongs in it,
which direction is currently active, and how strongly the available evidence
supports each interpretation. It is also the living notebook for working
decisions, questions, and short handoffs. Operational run status belongs in the
[current-simulation ledger](current_simulations.md), while concluded studies
and artifact locations belong in the
[finished-simulation ledger](experimental_manifest.md).

The wording under **Personal notes** is user-owned. Preserve it verbatim
unless the user explicitly asks for an edit. Assistant-maintained context
belongs in the other sections.

## Repository goals

### Primary goal: ReLU-to-DRN transfer across realistic ReRAM devices

The primary goal is to study a controlled three-part deployment and recovery
problem:

1. **Initialization and knowledge distillation:** determine how faithfully a
   DRN with realistic ReRAM synapses can be initialized from a pretrained ReLU
   network. Keep direct weight mapping and further training by
   `KL(teacher || DRN)` as separate baselines. Here, realistic means that the
   synapses are constrained by measured device traces or an explicitly
   validated statistical device distribution; an ideal bounded-conductance
   DRN is a control rather than the target result.
2. **Cross-cohort or cross-distribution deployment:** after obtaining DRN
   weights using one source cohort or device distribution, initialize a new
   set of devices drawn from a different cohort or statistical distribution
   to those trained targets. Measure how much predictive behavior changes when
   the teacher, dataset, architecture, and trained targets are held fixed and
   only the device population changes.
3. **Performance recovery:** determine how much of the cross-cohort loss can be
   recovered by post-deployment fine-tuning. Candidate interventions include
   full-conductance fine-tuning, selective-layer updates, low-rank adaptation,
   and device-aware or probabilistic write strategies.

The core measurements are teacher-to-DRN KL, student accuracy, teacher
agreement, conductance/programming statistics, and hidden/output operating
points before transfer, after transfer, and after recovery.

### Secondary goal: DRN architectures that distill more faithfully

The secondary goal is to determine whether the initial knowledge-distillation
result can be improved by changing the physical DRN architecture. The first
architectural hypothesis is to split every nonlinear node into matched signed
ports: one with `voltage_amp = +4`, `current_amp = +0.25`, and one with
`voltage_amp = -4`, `current_amp = -0.25`. The reciprocal gain magnitudes
retain `A * B = 1`, while the polarity split provides the positive and negative
signals required by a differential `G+ - G-` edge. This must be evaluated
through the full equilibrium equations: an architecture only counts as an
improvement if it increases teacher fidelity without merely hiding the floor
in a normalized readout, and its physical routing and loading assumptions must
be stated explicitly.

The engineering purpose of the repository is to express these primary and
secondary questions as strict versioned experiments, run them through
`python -m ebl`, and compare them using explicit checkpoints, device data,
metrics, and provenance.

## Scope and evidence boundary

In scope:

- coordinate-descent equilibrium simulation of DRNs with dissipative
  nonlinearities;
- equilibrium propagation and backpropagation through the configured solver;
- mappings between conventional neural-network weights and physical DRN
  conductances, including dual-rail and differential parameterizations;
- ideal bounded devices, measured conductance traces, endpoint programming
  models, and pulse-update models when the source data supports them;
- deployment, full or selective fine-tuning, low-rank adaptation, and
  teacher-student distillation;
- reproducible experiment schemas, campaigns, checkpoints, evaluation probes,
  plots, and research documentation.

Out of scope unless a dedicated model is added and validated:

- transistor-level or layout-accurate circuit signoff;
- treating an abstract signed branch, reference subtraction, or normalized
  device fit as a demonstrated physical circuit;
- inferring incremental pulse dynamics from endpoint program-and-verify data;
- broad hardware or algorithmic claims from a single seed, device split, or
  virtual-device assignment;
- versioning large generated outputs or raw external datasets in Git.

Every result should therefore say whether it is an ideal numerical control, a
device-data-driven simulation, or a circuit-level claim. Unless explicitly
demonstrated otherwise, results in this repository establish simulation
behavior rather than fabricated-hardware behavior.

## Personal notes

<!-- Add personal notes below this line. -->

## Current research direction

### Primary experimental sequence

1. Train and freeze one conventional ReLU teacher checkpoint.
2. Map it into an ideal bounded DRN to establish the numerical control.
3. Initialize realistic ReRAM synapses from the teacher mapping using cohort A,
   optionally continue same-cohort knowledge distillation, and save both the
   selected physical weights and the corresponding digital targets.
4. Program the trained targets onto independently assigned cohort-B devices or
   devices sampled from a shifted statistical distribution. Evaluate this
   deployment before applying any recovery update.
5. Fine-tune the transferred DRN and compare recovery against the unchanged
   post-transfer initialization. Full, selective, and low-rank recovery must
   use the same teacher, data split, transferred device realization, and
   checkpoint-selection rule.

Each stage should report an initialization-to-transfer-to-recovery trajectory,
not only final accuracy. Mapping error, device reassignment, programming error,
and fine-tuning updates must remain separately measurable.

### Secondary architectural sequence

1. Keep the teacher checkpoint, data, device cohort, and distillation objective
   fixed while changing only the DRN architecture.
2. Establish the present `[1568, 100, 20]` dual-rail DRN with
   `voltage_amp = 4` and `current_amp = 0.25` as the reference architecture.
3. Add matched `(+4,+0.25)` and `(-4,-0.25)` amplifier ports, or other
   explicitly defined physical structures intended to compensate the
   conductance floor.
4. Compare calibration KL, post-distillation KL, accuracy, teacher agreement,
   voltage statistics, and `G+ + G-` or equivalent loading. A lower KL is not
   sufficient if it depends on an unmodeled subtraction or amplifier.

### Methodological commitments

- Preserve a measured nonzero conductance floor unless the modeled circuit
  explicitly cancels it. Do not silently subtract `G_min` during conversion
  back into DRN units.
- Keep source-cohort training, target-cohort deployment, and post-transfer
  recovery as distinct run stages with explicit artifacts between them.
- Use pure `KL(teacher || DRN)` for teacher-student training; labels remain
  evaluation metrics unless a separately named experiment introduces a
  supervised term.
- Use BPTT through the configured equilibrium solver as the default
  fine-tuning method, with EP retained as a controlled alternative.
- Keep target-programmed nearest-state projection distinct from a genuine
  incremental-pulse or Tiki-Taka implementation.
- MNIST with a `[1568, 100, 20]` dual-rail DRN is the primary benchmark;
  sklearn digits remains a compact development and numerical-parity benchmark.

## Results and current interpretation

### Primary teacher-initialized measured-trace result

- **Differential-device simulation:** we performed a single-seed MNIST
  comparison of ideal and measured one-device versus differential `G+ - G-`
  physical edges. A bias-free `784 -> 50 -> 10` ReLU teacher initialized a
  dual-rail `[1568, 100, 20]` DRN, after which all DRN conductances were
  fine-tuned for 20 epochs using only `KL(teacher || DRN)`. Ideal one-device
  and differential arms were numerically tied at `97.42%` test accuracy and
  `0.000712` test KL. With measured cohort-A traces, the per-device-affine
  one-device arm collapsed to `7.99%`, while paired-common-window
  differential initialization and trace-projected fine-tuning reached
  `97.30%` test accuracy, `98.98%` teacher agreement, and `0.010119` test KL.
  The result supports local common-floor cancellation as the key benefit, but
  remains exploratory and single-seed. The common floor cancels only from
  signed transfer; it remains in `G+ + G-` loading, and the modeled
  opposite-polarity branch still requires a complementary-routing or
  subtractive-readout circuit interpretation.

### RESET-trained single-device teacher-matching result

- **Training without teacher-weight programming:** we trained the original
  dual-rail, one-device-per-physical-edge DRN directly from cohort-A pulse-zero
  states with `voltage_amp = 4`, `current_amp = 0.25`, and fixed readout gain
  one. A pure `KL(teacher || DRN)` arm reached `85.52%` held-out test accuracy,
  versus `86.88%` for a matched hard-label cross-entropy arm. The KL arm was
  therefore below the `94%` target and `1.36` percentage points below the
  control. Both objectives learned substantially from the common `9.96%`
  RESET initialization, but their raw-trace-projected trajectories oscillated
  strongly. The comparison suggests that KL is not the dominant limitation.
  It does not reproduce the historical approximately `95%` cohort-A run,
  which trained a hidden bias with paired squared error.
  Full metrics are documented in [`mnist_relu_drn_reset.md`](mnist_relu_drn_reset.md).
- **Historical-control replay and amplifier-index confound:** the exact
  bias-inclusive replay revealed that the archived process built its LR-search
  stack at logical/process indices 0/1/2, rebuilt production at process-global
  indices 3/4/5, and evaluated the selected weights in a fresh process at
  0/1/2. Thus its configured `voltage_amp = 4`, `current_amp = 0.25` did not
  describe one consistent circuit. An explicitly named legacy adapter
  reproduced the archived paired-loss selected validation (`95.10%`) and test
  (`95.97%`) exactly. Under the same mismatch, cross-entropy reached `92.37%`
  test accuracy and teacher KL only `83.72%`. The supervision objective is
  therefore material for this historical artifact, and the archived `95%`
  result must not be treated as evidence that the corrected logical-index
  circuit reaches `95%`. The implementation, provenance checks, and campaign
  are documented in
  [`mnist_relu_drn_reset_bias.md`](mnist_relu_drn_reset_bias.md).
- **Controlled bias/loss/indexing attribution:** a completed single-seed
  `2 x 2 x 2` factorial held the teacher, measured devices, device assignments,
  input-reset semantics, seed, amplification, and rebuild lifecycle fixed.
  Corrected logical indexing reached `93.90%` without bias and `93.94%` with
  bias under paired MSE, versus `86.28%` and `85.49%` under teacher KL. Legacy
  indexing raised the paired-MSE cells to `95.72%` and `95.70%`, but reduced
  the KL cells to `77.59%` and `80.45%`. Thus the historical indexing defect
  contributed about `1.8` percentage points of MSE uplift, while changing from
  paired MSE to KL cost `7.62-8.45` points in the corrected circuit. Bias was
  negligible in the corrected MSE comparison (`+0.04` points) and is not the
  missing explanation. The historical result is therefore modestly too high
  for the corrected circuit, while the newer approximately `85.52%` result is
  representative of the KL objective rather than an indexing-induced
  underestimate. These are seed-42 estimates without run-to-run uncertainty;
  details and artifacts are documented in
  [`mnist_relu_drn_reset_factorial.md`](mnist_relu_drn_reset_factorial.md).
- **Literal-RESET differential screen:** we trained the bias-free corrected
  logical circuit for ten production epochs with one independently assigned
  measured cohort-A `G+`/`G-` pair per physical dual-rail edge. Paired MSE
  training raised validation accuracy from `9.32%` at RESET to a selected
  `92.98%`; fresh-process test accuracy was `93.94%`, with `94.70%` teacher
  agreement and `0.0933816` paired squared error. The run produced an exact
  epoch-10 resume state, a four-branch named-weights checkpoint, and matching
  model-local amplification metadata across construction order and fresh
  validation. This is evidence that the differential implementation trains
  stably from literal RESET, not evidence of an advantage over one-device
  synapses: the nearest factorial arms used 20 epochs, the selected learning
  rates lay on the low/low search-grid edge, and neither a matched ten-epoch
  one-device control nor a shared-reachable-zero differential arm was run.
  Details are documented in
  [`mnist_relu_drn_reset_differential_10ep.md`](mnist_relu_drn_reset_differential_10ep.md).

### Historical endpoint-model, conductance-floor, and adaptation results

- **Study scope:** completed device experiments are target-programmed
  deployment and adaptation simulations, not in-situ pulsed-update Tiki-Taka
  demonstrations. The Wan and IBM PCM/CMO arms use endpoint programming
  models; the measured cohort-A/B arms project digital targets to globally
  nearest trace states. Neither path reproduces a sequence of local
  potentiation/depression updates. Wan et al.'s NeuRRAM work is a non-IBM
  hardware paper; IBM later implemented an approximate model of its published
  device data in AIHWKit.
  The Wan hardware used `G_min=1 µS` for all models, `G_max=40 µS` for CNNs,
  and `G_max=30 µS` for LSTMs and RBMs. Our main Wan-2022 experiments instead
  approximated the CNN device window by mapping nonnegative DRN conductances
  proportionally into `0` to `40 µS`, with `1.0` in DRN units corresponding
  to `40 µS`; this omits the paper's physical `1 µS` floor. The configured
  interval is a target-mapping range rather than a hard upper bound on every
  noisy sample. The exploratory CMO/PCM screen used those models' own
  conductance ranges.
- **Historical Wan noise model:** HWA training used temporary additive
  Gaussian parameter noise (`std_dev=0.0025` in the completed Wan studies).
  Deployment then used IBM AIHWKit's phenomenological Wan-2022 model:
  `G_device = max(0, G_target + sigma_t(G_target) * epsilon)`, where
  `epsilon ~ N(0, 1)` and `sigma_t` is a conductance-dependent fourth-order
  polynomial fitted to measured variation at the selected age. The main
  experiments used the one-day fit, full noise scale, and independent seeds
  for the two base arrays. Each device realization was sampled once and saved
  in the checkpoint; it was not resampled per minibatch or epoch. The model
  has no separate read-noise term, and the LoRA factor arrays remained ideal.
- **IBM PCM/CMO HWA model:** start from the existing FP32 MNIST checkpoint and
  fine-tune for two epochs. For every training minibatch, independently add
  Gaussian noise to each W1/W2 output channel with
  `sigma = 0.03 * max(abs(channel))`, compute BPTT gradients at that noisy
  realization, restore the clean master weights, and then apply the optimizer
  update. This reproduces the IBM Analog Foundation Models weight-modifier
  choice (`ADD_NORMAL_PER_CHANNEL`, `std_dev=0.03`) but does not reproduce its
  LLM-specific clipping, DAC/ADC quantization, optimizer, or architecture.
  The DRN perturbation is also clamped to nonnegative conductance bounds,
  whereas IBM applies the modifier to signed neural-network weights.
- **IBM PCM/CMO deployment models:** the PCM arm uses IBM's released
  Analog-Foundation-Models mapping: each output channel, or each input tile of
  at most 512 values, is normalized to a nominal maximum of `180` in the
  helper's unitless fit coordinate before a
  fit-coordinate-dependent Gaussian programming error is sampled. Values mapped
  below `0.6` are set exactly to zero. The AFM paper's Figures 7a and 9
  use normalized axes; IBM's released code supplies `GMAX=180` without a
  physical unit. The underlying Hermes paper gives a nominal per-PCM-device
  range of `0.1–10 µS`, but uses unit-cell conductance in ADC counts for
  weight mapping, so this is not a conversion for the helper's 180
  coordinate. The CMO/HfOx arm uses the AIHWKit
  programming, relaxation, and read-noise equations over the fitted
  `9–88.199997 µS` window with the `0.2%` program-and-verify acceptance fit and
  a one-second inference time. Falcone et al.'s Figure 3 and Supplementary
  Figure S5 directly show the corresponding experimental window at
  approximately `10–90 µS`. Their reported MVM simulation omitted read noise
  and included converter quantization plus wire IR drop; our CMO arm enables
  the AIHWKit read term but omits those circuit-level effects.
  Three strict mappings now separate calibration from floor cancellation.
  `literal_conductance` maps `w` to
  `clip(88.199997 w, 9, 88.199997) µS`, so every `w < 0.1020408`
  collapses to the same value. `affine_floor` maps `[0,1]` across the full
  physical window and exposes the passive DRN to
  `0.1020408 + 0.8979592 w`; it retains the floor without merging the learned
  targets. `normalized_offset` uses the same affine physical targets but
  subtracts and rescales the floor on conversion back, so it remains an
  explicitly labelled cancellation abstraction.
- **IBM PCM/CMO experiment protocol:** (1) retain the existing clean FP32
  baseline; (2) program FP32 into each device model; (3) create one shared
  two-epoch HWA checkpoint and program it into the same device-seed
  realization; (4) from that HWA checkpoint, compare full W1/W2 BPTT recovery
  with rank-4 physical LoRA recovery. The hidden bias is frozen in both
  recovery arms. The base is frozen only for LoRA.
- **Noisy BPTT write semantics:** BPTT gradients are evaluated at the current
  realized device state. The optimizer updates a clean digital target, after
  which every trainable dense array is reprogrammed through the selected
  endpoint model. Full recovery rewrites W1 and W2 after every minibatch;
  LoRA rewrites A1, B1, A2, and B2 after every minibatch while W1/W2 remain
  fixed. These are repeated closed-loop program-and-verify writes, not
  Tiki-Taka pulses or an in-situ outer-product update.
- **IBM PCM/CMO result:** the endpoint models produced essentially no
  deployment gap in the one tested realization. HWA deployment was `96.35%`
  on PCM and `96.38%` on CMO. After 75,000 noisy writes, cost-selected full
  BPTT reached `96.43%` on both devices (`+0.08` and `+0.05 pp`), while
  rank-4 LoRA reached `96.30%` on PCM and `96.40%` on CMO (`-0.05` and
  `+0.02 pp`). These archived CMO numbers use the normalized-offset
  cancellation mapping. The changes are too small and under-replicated to
  establish a full-versus-LoRA winner.
- **CMO literal hard-clipping follow-up:** the literal mapping lowers FP32
  from `96.15%` to `79.51%` and HWA from `96.37%` to `80.61%`. The initial
  write raises `98.13%` of W1 and `88.10%` of W2 targets to the device floor.
  Cost-selected full noisy BPTT reaches `89.73%` after 75,000 writes, a
  `+9.12 pp` gain and `57.87%` recovery of the clean-HWA gap, but remains
  `6.64 pp` below clean HWA. Rank-4 LoRA reaches only `80.56%`; at the last
  write all four factor-target arrays remain entirely below the physical
  threshold and are programmed to `9 µS`. This is a literal range/clipping
  and correction-direction mismatch, not a failure to inject write noise.
- **CMO floor-mechanism separation:** an affine mapping retains the same
  `9 µS` passive floor while preserving every clean target's ordering before
  endpoint noise.
  It obtains `96.12%` on the first noisy write and `96.11%` in the stored
  realization, versus `96.37%` clean HWA. The deterministic affine result is
  `95.94%`, while deterministic literal clipping is `81.83%`. The earlier
  collapse was therefore dominated by erasing sub-floor weight information,
  not by the finite floor alone. The floor still dominates the node sums:
  median affine floor shares are `86.79%` hidden and `68.22%` output, and
  hidden/output voltage RMS is attenuated from `0.8761/0.2188` to
  `0.1922/0.02469`. Noise and the physical clamp still leave `18.32%` of
  stored W1 and `11.80%` of stored W2 exactly at the floor.
- **Cancellation implication:** the coordinate update divides a weighted
  voltage numerator by the incident-conductance sum. A uniform floor changes
  both terms. Standard reference-current subtraction corrects only the
  numerator; exact DRN cancellation must also correct the row/column-sum
  denominator and preserve bias/nonlinearity scaling. The deterministic
  full-correction oracle recovers `96.37%`, but it is an equation-level active
  bound rather than a demonstrated circuit.
- **Literal-versus-affine divergence:** raw-softmax predictive KL is
  misleadingly small (`0.0000879` mean symmetric nats) because both mapped
  DRNs have strongly attenuated output voltages. After deterministic
  per-example score-RMS normalization, mean symmetric KL is `0.2971` nats,
  JS is `0.06449`, and top-1 agreement is `81.23%`. A 128-bin smoothed
  conductance histogram gives `0.8157` symmetric KL and `0.15614` JS for W1
  and W2 combined.
- **Conclusion on bounded-weight expressivity:** when the mapping from the
  loose-range FP32 network to the bounded physical conductance window is done
  correctly, the observed predictive divergence does not materially
  increase. The FP32 reference was configured over `[1e-7,100]` rather than
  being mathematically unbounded, but its learned W1/W2 maxima were only
  `0.720/0.734`; consequently, the deployment ceiling of `1` clipped no
  weights. Direct deterministic affine mapping gave `0.00919` mean symmetric
  score-RMS-normalized KL and `98.71%` top-1 agreement with FP32. The final
  HWA-plus-noisy-affine network gave `0.00875` symmetric KL, `0.00212` JS,
  and `98.64%` agreement, while accuracy changed only from `96.15%` to
  `96.11%`. In contrast, the noisy literal-floor network gave `0.30455`
  symmetric KL and only `81.17%` agreement because its many-to-one mapping
  erased most sub-floor distinctions. Thus, for this checkpoint and test
  distribution, a bounded window did not itself produce the expressivity
  loss suggested by the earlier experiments; the apparent loss came mainly
  from an incorrect non-injective mapping. This was contrary to our initial
  interpretation of those experiments. These results measure functional
  fidelity on the 10,000-example MNIST test set, not equality of the global
  hypothesis classes. Full numerical results are in the local ignored
  artifact
  `results/mnist-cmo-floor-mitigation/analysis/fp32-to-bounded-noisy-kl-v2/summary.json`.
- **Why the affine floor is unexpectedly benign:** each mapped crossbar is
  `W' = f J + (1-f)W`. This is one-to-one for `f<1`; the all-ones offset
  cancels exactly against the balanced `50[x,-x]` input and is largely
  rejected by paired output subtraction. Output-pair degree mismatch falls
  from `11.02%` to `3.52%`, so residual floor leakage is only `2.60%` of
  differential-score RMS. Positive degree loading then mostly rescales
  voltages: deterministic hidden-gate agreement is `100%` after the first
  update and `99.997%` after four, and all 108 changed predictions are in the
  lowest clean-margin quintile. Denominator correction raises affine accuracy
  from `95.94%` to `96.34%`; removing numerator floor current alone gives
  only `95.92%`. This confirms that the remaining drift comes mainly from KCL
  degree normalization, not erased conductance distinctions.
- **Electrical-margin qualification:** the same affine mapping reduces
  centered paired-score RMS from `0.2985` to `0.01969` and median absolute
  class margin from `0.9419` to `0.05585`. Exact float32 `argmax` and
  per-example RMS-normalized KL hide this attenuation. In an explicitly
  synthetic post-inference control, output-node Gaussian noise with
  `sigma=0.01` leaves clean HWA at `96.35%` but lowers deterministic affine
  to `91.32%`; at `sigma=0.03`, the respective accuracies are `96.29%` and
  `48.52%`. A quantization step of `0.03` gives `96.35%` versus `92.31%`.
  These perturbations are in model-voltage units and are not calibrated
  circuit models, but they show that preserved noiseless ordering is not
  preserved analog robustness. The current CMO path freezes one sampled
  conductance realization and omits dynamic output noise, ADC resolution,
  amplifier offsets, IR drop, and finite diode thresholds. Detailed evidence
  is in the local ignored artifact
  `results/mnist-cmo-floor-mitigation/analysis/affine-floor-mechanism-v3/summary.json`.
- The useful finding from the normalized-offset study is optimization
  stability, not gap recovery: every trajectory remained finite, LoRA W1/W2
  stayed bit-exactly frozen, full-recovery bias stayed frozen, and all factor
  arrays remained within bounds despite a noisy endpoint write after every
  minibatch.
- The literal-floor follow-up supplies the missing high-headroom control.
  Full BPTT can strengthen a small subset of useful base conductances and
  recover substantially, whereas the current positive-only LoRA branch cannot
  subtract the dense floor and its factors never cross the device threshold.
- Physical LoRA shows positive recovery on the full-MNIST Wan-2022 deployment
  sample, with most of the observed rank benefit available by rank 2.
- On the original five MNIST devices, ideal full-model fine-tuning improves
  HWA-to-ReRAM accuracy from `93.162%` to `96.788%`. This is `+1.908` points
  above rank-4 LoRA (`94.880%`) and `+0.548` points above the clean HWA
  checkpoint (`96.240%`).
- The accuracy advantage requires a fundamentally larger intervention:
  approximately `136,630` of `158,900` existing W1/W2/bias values change
  per selected checkpoint. Rank-4 LoRA adds `7,152` conductances while
  leaving all existing base and bias values bit-identical.
- The completed Wan full-model result is an ideal upper control. The new
  PCM/CMO full-model arms include stochastic endpoint write error after every
  update, but still do not model individual pulses, asymmetric incremental
  updates, endurance, write energy, or program-and-verify latency.
- A major device-model limitation is that much of the available experimental
  data and the AIHWKit Tiki-Taka device fits are expressed in normalized state
  and update units rather than absolute conductance. Even when a source-paper
  figure has a `µS` axis, the corresponding preset does not preserve a
  traceable physical calibration, so `G_min`, `G_max`, and `delta_G` per pulse
  cannot be recovered without an additional device-specific mapping.
- The same normalization limitation affects the new endpoint models. IBM's
  PCM helper rescales each channel/tile independently before applying the
  unitless 0-to-180 fit, so it does not preserve one absolute
  DRN-to-conductance scale.
  The first archived CMO arm mapped logical `[0, 1]` to physical
  `[9, 88.199997] µS` and subtracted `9 µS` after noise. Logical zero in that
  arm therefore assumed cancellation of the physical common-mode floor.
  Future physical-device studies retain the floor: use `affine_floor` when
  preserving the learned ordering, or `literal_conductance` when explicitly
  testing absolute scaling and hard clipping. Nominally zero LoRA factors
  also retain the floor unless a cancellation or open-circuit mechanism is
  part of the experiment.
- Program-and-verify measurements cannot be substituted directly for a
  Tiki-Taka pulsed-device model. A distribution of final programming errors
  around a requested target conductance does not determine the trajectory or
  noise of individual potentiation and depression pulses.
- The smaller digits studies are mixed: hard-sigmoid recovery was positive,
  while the perfect-diode comparison left little recovery headroom.
- PCM and HERMES screens did not materially enlarge the deployment gap under
  the selected mapping. The larger literal CMO/HfOx gap was dominated by
  hard clipping and conductance-range mismatch, which a positive-only LoRA
  branch cannot subtract. An affine floor-retaining CMO mapping does not
  reproduce that accuracy gap, although it strongly attenuates node voltages.
- Each completed comparison uses a single base-training seed; none includes
  within-study base-seed replication. Multiple independently trained bases and
  a separate untouched final test split are the next major reliability step.

Exact measurements, limitations, and raw artifact locations are in the
[finished-simulation ledger](experimental_manifest.md).

## Open questions

- What physical calibration should map normalized DRN conductances to each
  device's `G_min` and `G_max`?
- How much affine-floor voltage attenuation can realistic diode, readout,
  converter, and residual-current noise tolerate?
- Can a rank-one active reference correct both floor current and the
  conductance-sum denominator without destabilizing the reciprocal network?
- How do learned selector sparsity and bounded fan-in trade accuracy against
  floor loading, area, and power?
- Which pulsed-device datasets provide both incremental update trajectories and
  absolute `G_min`/`G_max` values, rather than only normalized Tiki-Taka fits?
- Should the base DRN be trained from the beginning with the device minimum
  conductance enforced?
- Which realistic conductance-loss mechanism gives the fairest recovery test
  for the current positive-only passive LoRA circuit?
- Is a differential or otherwise signed physical LoRA branch required for
  bidirectional corrections?
- Does reprogramming `A` and `B` after every minibatch help or hurt relative
  to programming the trained factors only once at the end?
- Do the observed diminishing rank returns persist across multiple base
  training seeds and a separate final test split?
- How do retention time and factor learning rate interact with rank?
- How much of the full-model advantage survives pulse-aware ReRAM updates?
- At what accuracy target or write budget does rewriting the deployed base
  become preferable to adding a frozen-base LoRA branch?
- Is fine-tuning only W2 or W2 plus bias enough to close most of the
  `1.908`-point gap between rank-4 LoRA and full-model fine-tuning?

## Next experiment candidates

1. Start a separate Tiki-Taka pulsed-device study using measured incremental
   potentiation/depression data. Do not reuse the current program-and-verify
   HWA models as pulse-update models unless their source papers provide the
   required per-pulse trajectories. Measure write count, update noise, energy,
   and endurance alongside accuracy.
2. Before spending more recovery compute on the IBM endpoint models, run a
   multi-seed deployment-only severity screen. Choose a physically supported
   acceptance/retention condition that produces a reproducible nonzero gap,
   without tuning that choice on the final test set.
3. Compare three targeted post-HWA interventions on the same deployment:
   rank-4 LoRA, W2-plus-bias fine-tuning, and full-model fine-tuning. This
   tests whether the full rewrite is actually needed.
4. Train a perfect-diode DRN from initialization through the affine CMO
   mapping and endpoint noise. Track voltage/noise margin and conductance
   loading as well as accuracy.
5. Test the current passive LoRA branch on conductance-loss errors such as
   drift or stuck-low devices, where an added conductance path can compensate
   the failure direction.
6. Design a differential physical LoRA branch and compare it with the
   positive-only branch on identical signed perturbations.
7. Compare active denominator calibration, full rank-one KCL cancellation,
   and a validation-trained selector mask under matched mismatch and read
   noise. Ordinary numerator-only crossbar subtraction is not an exact DRN
   control.

The floor mechanism, controlled ablations, and recommended circuit studies are
documented in
[Finite conductance floors in DRNs](conductance_floor_mitigation.md).

## Related documents

- [Differential pairs using signed amplifier ports](differential_pair_amplifier_implementation.md)
- [Differential memristor scheme](differential_scheme.md)
- [Teacher-initialized MNIST DRN distillation](mnist_relu_drn_kd.md)
- [Controlled RESET bias/loss/indexing factorial](mnist_relu_drn_reset_factorial.md)
- [Literal-RESET differential MNIST screen](mnist_relu_drn_reset_differential_10ep.md)
- [MNIST HWA and LoRA experiment setup](mnist_hwa_lora_experiment_setup.md)
- [MNIST IBM PCM/CMO noisy-recovery study](mnist_ibm_pcm_cmo_noisy_recovery.md)
- [Current LoRA/HWA simulations](current_simulations.md)
- [Finished LoRA/HWA simulations and progress](experimental_manifest.md)
- [ReRAM device catalog](reram_device_catalog.md)
- [Physical layerwise low-rank recovery](passive_layerwise_low_rank_recovery.md)
- [Measured cohort-B memristor LoRA recovery](measured_cohort_b_lora_recovery.md)
- [Digital low-rank recovery](digital_low_rank_recovery.md)
- [Experiment runtime](experiment_runtime.md)
