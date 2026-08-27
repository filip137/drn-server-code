# Current State

Last updated: 2026-08-25

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

## Big picture

The repository asks one central question: **how can a dissipative resistive
network preserve useful computation when ideal weights are mapped onto
realistic, nonzero-floor, noisy ReRAM devices, and what is the least costly
way to recover any lost performance?**

The strongest current interpretation is:

- Physical representation matters before optimization. Differential/common-
  window mappings preserve signed information far better than independent
  one-device affine mappings, while literal floor clipping can erase most
  learned small conductances.
- A finite conductance floor is not automatically destructive when the map is
  injective, but it can strongly attenuate voltage and class margins. Accuracy
  without a calibrated noise-margin test is therefore incomplete evidence.
- Full-model rewriting is the strongest recovery control observed so far.
  Passive low-rank adaptation can recover useful accuracy with far fewer new
  conductances and no base rewrite, but the present positive-only branch
  cannot correct every failure direction.
- The adapted four-device common-window representation is highly sensitive to
  physical-device reassignment: zero-update transfer from cohort A to five
  cohort-B assignments lost `54.62` accuracy points on average despite scalar
  recalibration and broadly similar aggregate reachable-window statistics.
- Cell-specific IBM OM QAT shows the same issue in a stricter pulse-resolved
  setting. The selected fixed-array QAT checkpoint reached `93.31%` programmed
  validation accuracy on development assignment 84001 but only `74.154%` mean
  programmed test accuracy after frozen deployment to assignment 85001. It
  still beat continuous HWA by `6.552` points on the replacement array, so QAT
  helps, but much of its compensation is array-specific. The transfer loss is
  therefore a concrete starting gap for deployed-array, on-chip-compatible
  quantized adaptation.
- Gradient direction alone is sufficient for substantial same-cohort
  four-device recovery in the endpoint-projection simulator: balanced
  signSGD reached `95.77%` test accuracy from a `61.48%` initialization, but
  remained `1.69` points behind ordinary SGD and had `3.50x` its test KL.
- The model-based IBM ReRAM pulse-count reconstruction does not support using
  adaptive batching as an unconditional replacement for one-pulse verify. It
  saves `9.85–85.63%` of verify reads with essentially unchanged accepted-
  endpoint RMSE, but uses `53.70–210.51%` more pulses and loses `0.96–5.31`
  percentage points of programming success. Its Gaussian endpoint surrogate
  failed every predeclared adequacy gate. A post-completion v2 check found that
  one global verify-window uniform passed only 7 of 16 conditions, whereas a
  target-conditioned eight-bin piecewise-uniform density passed all 16
  held-out gates. This compact result is exploratory until the v3 rerun; the
  empirical kernel and separate failure/corrupt/cost models remain the
  authoritative v2 outputs. A paired persistent-bound extension further
  separates non-corrupt targets below RESET, inside the reachable interval,
  and above SET. Its fit-to-validation class-probability MAE is `0.88-1.53`
  points. Roughly half the devices cannot reach the exact nominal endpoint;
  HfO2 nevertheless often passes apparent verification because its apparent
  write variation is large.
- A declared 128-pulse successor now tests whether the 512-pulse tail is useful
  programming or mostly unreachable-bound chasing and noisy late admission.
  The completed v2 configurations remain immutable; the new cap is isolated in
  the `production_short_cap128` profile and the 2026-08-22 v3 study.
- Most conclusions remain exploratory: the key studies use one base-training
  seed, and endpoint program-and-verify models are not substitutes for
  measured incremental pulse dynamics.

The active direction is to turn those observations into matched, predeclared
studies with independent seeds, explicit device mappings, voltage/noise-margin
measurements, and intervention-cost accounting. Exact completed-study
measurements and limitations belong in
[`experimental_manifest.md`](experimental_manifest.md); transient execution
state belongs in [`current_simulations.md`](current_simulations.md).

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

## Current evidence

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
  subtractive-readout circuit interpretation. A clean ten-epoch replay matched
  the first ten epochs exactly, selected epoch 9, and reached `97.48%` test
  accuracy, `98.76%` agreement, and `0.0140905` KL. The additional ten-epoch
  budget improved KL and agreement but not accuracy on this single seed; this
  is consistent with checkpoint selection by validation KL and does not
  establish that the shorter budget generalizes better. A clean, budget-
  matched one-conductance replay also reproduced its earlier trajectory
  exactly: it remained at `7.98%` test accuracy, `7.81%` agreement, and
  `2.246097` KL. The ten-epoch differential pair therefore gains `89.50`
  accuracy points and reduces KL by `99.37%` under the matched teacher, data,
  seed, solver, amplification, objective, and minibatch budget. The nominal
  one-conductance map was accurate before measured projection, but independent
  device-affine floors destroyed its score direction, forced the fitted gain
  to the `0.001` lower boundary, and left trace-projected fine-tuning almost
  stationary.

- **Four-device clamped-input follow-up:** At the fixed input boundary
  `[x, -x]`, the eight-device differential interface can be regrouped exactly
  into four post-facing conductances, but replacing each device pair sum with
  one measured memristor loses range and mismatch averaging. A shared-four-
  device-window arm therefore initialized below the eight-device arm
  (`61.48%` versus `89.12%`), yet after ten measured fine-tuning epochs their
  test accuracies were effectively tied (`97.46%` versus `97.48%`). The
  four-device test KL remained higher (`0.0183103` versus `0.0140905`), so the
  current evidence favors four devices when adaptation and device economy
  matter, and eight when initialization-only feed-forward fidelity matters.
  See the [four-versus-eight report](dual_rail_input_four_vs_eight_devices.md).

- **Four-device held-out cohort-B follow-up:** Re-encoding the selected
  four-device checkpoint with one common cohort-B window per four-cell block
  reduced fresh-test accuracy from `97.46%` to `35.79%`. Ten matched off-chip
  epochs recovered it to `97.31%` (`97.61%` teacher agreement, `0.0557048`
  KL), only `0.15` accuracy points below its cohort-A source. The window
  widths themselves were comparable across cohorts; the immediate failure
  coincided with `6.16x/7.32x` signed-drive contraction and roughly `9%`
  higher four-cell load. Relative to the eight-device common-window arm, four
  devices were `7.17` points lower immediately and `0.50` points lower after
  adaptation, while using half the conductances. See the
  [cohort-B report](mnist_four_device_cohort_b_adaptation.md).

- **Four-device cohort transfer:** We reprojected that fixed cohort-A
  four-device checkpoint onto five deterministic assignments from held-out
  cohort-B traces without applying a learning update. Fresh test accuracy fell
  from `97.46%` at the source checkpoint to `42.844%` on average (`5.021`
  percentage-point population standard deviation; `35.79%` to `50.85%`
  range), for a mean loss of `54.616` points. Mean teacher agreement was
  `43.02%` and mean KL was `1.665943`, versus `98.68%` and `0.0183103` at the
  source. The two layers' cohort-B quad windows averaged `11.288 uS` and
  `11.064 uS` in span with `4.952%` and `6.080%` empty windows, broadly similar
  to the earlier cohort-A aggregate. That makes device-level reassignment and
  nearest-state reprojection the leading explanation, but not yet a causally
  isolated one. The next matched control is the eight-device transfer on the
  same assignments, followed by a fixed-budget four-device recovery study.
  This remains single-model, endpoint-projection simulation evidence.

- **Four-device sign-only fine-tuning:** Starting from the same deterministic
  cohort-A quad-common-window initialization (`61.48%` validation accuracy),
  ten epochs of signSGD at the original numeric learning rates reached
  `95.46%` fresh test accuracy, `96.67%` teacher agreement, and KL `0.0784538`.
  Giving both layers a balanced `0.21 nS` shadow step improved those values to
  `95.77%`, `97.03%`, and `0.0640143`. On the like-for-like validation split,
  the balanced arm improved from `61.48%` to `94.94%`, a `33.46`-point
  recovery. This supports the claim that gradient direction alone can provide
  useful adaptation. It still trailed ordinary SGD by `1.69` test-accuracy
  points, and its KL was `3.50x` higher. These updates changed an ideal digital
  shadow and globally selected the nearest measured endpoint after every
  minibatch; they were not one-pulse-local sign updates on physical devices.

### Cell-specific IBM OM QAT and held-out-array transfer

- **Exact-bounds nine-level QAT:** Starting from the finalized full-span
  logical checkpoint, four matched arms were trained or selected on repaired
  IBM OM assignment 84001 and frozen before assignment 85001 was released.
  Quantized QAT selected at `93.3133%` programmed validation accuracy on the
  development array, compared with `93.1467%` for continuous HWA. On the
  untouched replacement array, five pulse-resolved programming repeats gave
  `74.154%` mean apparent accuracy for QAT versus `67.602%` for continuous
  HWA, a paired gain of `6.552` percentage points; the zero-update and clean-
  BPTT pipelines reached `65.124%` and `64.328%`. QAT also passed its
  predeclared retention gate: apparent accuracy was `100.616%` of its
  `73.70%` ideal mapped accuracy. The close agreement between ideal and
  apparent held-out accuracy shows that programming noise alone does not
  explain the transfer gap. The result supports the narrow QAT-versus-
  continuous hypothesis while supporting the user's broader interpretation
  that fixed-array training learns substantial array-specific compensation.
  The development-to-held-out comparison also changes from validation to test
  data, and only one development and one replacement assignment were used, so
  its `19.159`-point drop is not yet a population estimate. Exact target
  generation consumed every cell's hidden bounds and is an oracle
  characterization control. The next scientific question is how to continue
  quantization-aware adaptation from the already programmed replacement-array
  state using an update rule that is genuinely compatible with on-chip
  training.

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
- **IBM ReRAM fixed-amplitude program-and-verify reconstruction:** a shortened
  four-arm CUDA study sampled the AIHWKit 1.1.0 optimized-material and
  baseline-HfO2 fitted populations, with and without their published corrupt-
  device mixture. It retained all 41 targets, lower and upper boundary starts,
  one-pulse and adaptive controllers, 1024 identities, four repeats, and 204
  held-out validation identities at the primary half-step tolerance. The
  campaign completed `2,686,976` trajectories in `1,546.18 s`. Adaptive
  batching reduced verify reads by `9.85–85.63%`, but increased pulses by
  `53.70–210.51%` and reduced success by `0.96–5.31` points; accepted-endpoint
  RMSE changed by only `-0.43%` to `+0.05%`. All 16 Gaussian conditions failed
  both the predeclared coverage and normalized-Wasserstein gates. The
  empirical endpoint kernel, target-conditioned failure/corruption/saturation
  model, and pulse/verify-cost model must therefore travel together. This is
  reconstruction from fitted preset dynamics, not replay of raw IBM pulse
  traces, and the normalized Wan-2022 comparison is operational rather than a
  physical calibration.
  A matched v3 successor is planned with a 128-pulse cap. Retrospective v2
  prefix counts estimate a 21.97–59.33% pulse-work reduction across all eight
  population/controller conditions, but the new cap must be rerun because
  adaptive batch truncation changes the final verify opportunity.
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
- **Historical independently trained CMO versus Wan-2022 control:** a seed-17
  comparison used the same `96.15%` independently trained FP32 DRN checkpoint,
  the same two-epoch HWA checkpoint, one-day retention, retained affine floors,
  and ten epochs of full BPTT with a fresh endpoint write after every
  minibatch. The first FP32 device write gives
  `96.02%` on CMO and `87.80%` on Wan. The clean HWA checkpoint is `95.67%`;
  its first device write gives `96.19%` on CMO and `75.78%` on Wan.
  Cost-selected noisy BPTT finishes at `95.56%` on CMO and `94.23%` on Wan,
  verified in fresh processes. Thus BPTT changes the deployed HWA result by
  `-0.63 pp` for CMO but `+18.45 pp` for Wan. Wan's direct-write programming
  RMSE relative to the ideal affine target is `0.03145` DRN units, versus
  `0.00963` for CMO. This is a matched single-seed result, not a population
  estimate; the shared IBM-style HWA noise is not fitted to the Wan
  polynomial. This remains a useful endpoint-noise control, but it is
  superseded as the primary ReLU-to-DRN protocol because its DRN weights were
  not initialized from the frozen feed-forward ReLU teacher. Full protocol,
  provenance, and limitations are in
  [`mnist_wan_cmo_head_to_head.md`](mnist_wan_cmo_head_to_head.md).
- **Teacher-initialized CMO versus Wan-2022 result:** the corrected matched
  protocol starts every arm from one frozen bias-free `784 -> 50 -> 10` ReLU
  teacher and maps its signed matrices directly into the bias-free
  `[1568,100,20]` dual-rail DRN before any HWA or endpoint write. On the
  untouched MNIST test split, the teacher is `97.36%`, the immediate ideal DRN
  mapping is `96.91%` with teacher KL `0.009345`, and two generic 3%-modifier
  HWA epochs select a clean `97.34%` checkpoint with KL `0.002681`. The first
  ideal-map write gives `91.15%` CMO and `78.62%` Wan; the first HWA write
  gives `92.18%` and `80.31%`. Ten epochs of full BPTT, with gradients at the
  realized state and a fresh endpoint rewrite after every minibatch, select
  `96.46%` CMO and `91.84%` Wan, with KL `0.032341` and `0.186471`. HWA adds
  only `+1.03/+1.69 pp` at first write, while noisy BPTT adds
  `+4.28/+11.53 pp`, so post-deployment adaptation is the dominant
  intervention in this realization. HWA necessity is still unresolved because
  a matched no-HWA BPTT arm was not run. Wan programming error relative to the
  ideal affine target is `4.45x` the CMO value (`5.130` versus `1.152 uS` in
  effective DRN units), even though CMO has the larger finite floor. The full
  setup, KL results, provenance, and limitations are in
  [`mnist_wan_cmo_teacher_initialized.md`](mnist_wan_cmo_teacher_initialized.md).
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
- Program-and-verify endpoint measurements cannot be substituted directly for
  a Tiki-Taka pulsed-device model. A distribution of final programming errors
  around a requested target conductance does not determine the trajectory or
  noise of individual potentiation and depression pulses. The new IBM ReRAM
  study reconstructs such trajectories from fitted AIHWKit incremental-device
  equations, but this remains model-based evidence rather than measured pulse
  replay.
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
- Which quantized state and update rule can adapt the already deployed array
  in situ: persistent nine-level code changes, mixed-precision or Tiki-Taka
  accumulation, or a quantized adapter? What characterization, gradient, and
  program-and-verify information does each option require?

## Next steps

1. Define and predeclare a deployed-array QAT recovery study. Start every arm
   from the same preserved assignment-85001 apparent/persistent deployment,
   keep a frozen no-update control, and define an oracle STE-QAT recovery arm
   only as an upper bound. Before choosing the physical recovery arm, specify
   whether discrete updates act directly on persistent nine-level base codes,
   accumulate through a mixed-precision or Tiki-Taka state, or train a
   quantized adapter; in every case state how gradients become available
   SET/RESET pulses and which bounds or verify measurements the controller may
   use. Do not regenerate a fresh deployment between recovery arms.
2. Run the declared 128-pulse IBM ReRAM successor and compare it with the
   immutable 512-pulse endpoint model. Select the deployment cap explicitly,
   then integrate that empirical endpoint kernel with its separate
   target-conditioned failure, corruption, saturation, and cost models.
   Preserve each sampled programmed endpoint as the common starting state for
   a predeclared HWA-only versus on-chip-recovery comparison; do not redraw
   endpoints between arms.
3. Start a separate Tiki-Taka pulsed-device study using measured incremental
   potentiation/depression data. Do not reuse the current program-and-verify
   HWA models as pulse-update models unless their source papers provide the
   required per-pulse trajectories. Measure write count, update noise, energy,
   and endurance alongside accuracy.
4. Repeat the teacher-initialized CMO/Wan comparison over multiple endpoint
   seeds and add a direct ideal-map-to-device-to-BPTT arm. This separates
   whether HWA is necessary for recovery from whether it merely improves the
   first write. Then compare the generic 3% modifier with device-matched HWA,
   without tuning either choice on the final test set.
5. Compare three targeted post-HWA interventions on the same deployment:
   rank-4 LoRA, W2-plus-bias fine-tuning, and full-model fine-tuning. This
   tests whether the full rewrite is actually needed.
6. Train a perfect-diode DRN from initialization through the affine CMO
   mapping and endpoint noise. Track voltage/noise margin and conductance
   loading as well as accuracy.
7. Test the current passive LoRA branch on conductance-loss errors such as
   drift or stuck-low devices, where an added conductance path can compensate
   the failure direction.
8. Design a differential physical LoRA branch and compare it with the
   positive-only branch on identical signed perturbations.
9. Compare active denominator calibration, full rank-one KCL cancellation,
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
- [MNIST CMO versus Wan-2022 head-to-head](mnist_wan_cmo_head_to_head.md)
- [Teacher-initialized CMO versus Wan-2022](mnist_wan_cmo_teacher_initialized.md)
- [Current LoRA/HWA simulations](current_simulations.md)
- [Finished LoRA/HWA simulations and progress](experimental_manifest.md)
- [ReRAM device catalog](reram_device_catalog.md)
- [Physical layerwise low-rank recovery](passive_layerwise_low_rank_recovery.md)
- [Measured cohort-B memristor LoRA recovery](measured_cohort_b_lora_recovery.md)
- [Digital low-rank recovery](digital_low_rank_recovery.md)
- [Experiment runtime](experiment_runtime.md)
