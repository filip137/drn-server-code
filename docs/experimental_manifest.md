# Finished LoRA/HWA Simulations and Research Progress

Last updated: 2026-08-19

## Program goal

Determine when physical low-rank adaptation reliably recovers
hardware-induced DRN degradation, and quantify its accuracy, robustness, and
conductance overhead.

## How this ledger is used

This is the repository-maintained, human-edited index of concluded LoRA and
hardware-aware studies. It records one entry per scientific study rather than
one entry per seed, subprocess, or native run directory.

Outcomes and milestone states are descriptive research summaries. They do not
approve future runs, enforce a workflow, or change the native `python -m ebl`
run contract.

Lifecycle status lives in the current-simulation ledger, each finished entry
has a scientific outcome, and the milestone table summarizes broader program
evidence. These are separate descriptive views, not one state machine.

New raw outputs belong under `results/<study-id>/`. The studies indexed below
predate that convention and remain under their original ignored
`labs/cases/` roots. Those local raw directories are not moved or duplicated.
Inline paths labeled as local ignored detail exist only in a workspace that
still has those ignored raw roots.

## Progress toward the program goal

The states `supported`, `partial`, and `open` are a compact view of the
evidence available today, not pass/fail gates.

| ID | Milestone | State | Evidence | Remaining gap |
| --- | --- | --- | --- | --- |
| `M1` | Verify the physical LoRA mechanics, frozen-base behavior, and independent conductance bounds. | supported | [`hard-sigmoid-lora-smoke`](#hard-sigmoid-lora-smoke) and [`mnist-hwa-lora-rank-study`](#mnist-hwa-lora-rank-study) | Repeat the controls once factor-device nonidealities are introduced. |
| `M2` | Demonstrate positive recovery on the digits benchmark. | supported | Positive evidence in [`hard-sigmoid-lora-smoke`](#hard-sigmoid-lora-smoke), with the limited-headroom boundary in [`perfect-diode-hwa-lora-digits`](#perfect-diode-hwa-lora-digits) | Replicate across independently trained base models and a separate final test split. |
| `M3` | Demonstrate recovery at full-MNIST scale. | partial | [`mnist-hwa-lora-rank-study`](#mnist-hwa-lora-rank-study) | Current evidence uses one base-training seed despite replication over device seeds. |
| `M4` | Characterize device-seed robustness and adaptation-overhead behavior. | partial | Ten rank-4 device pairs and a matched five-pair rank sweep in [`mnist-hwa-lora-rank-study`](#mnist-hwa-lora-rank-study), plus the rewrite-versus-added-array control in [`mnist-hwa-reram-full-finetune`](#mnist-hwa-reram-full-finetune) | Repeat the adaptation comparison across independent base models and add physical write costs. |
| `M5` | Replicate with multiple base-training seeds and separate validation and final-test data. | open | Tracked as paused in the [current-simulation ledger](current_simulations.md) | Resume and analyze the matched multi-base comparison when base-seed generalization returns to active priority. |
| `M6` | Characterize base-device degradation mechanisms, then extend them to realistic factor-device effects aligned with a positive-only correction path. | partial | Base-device evidence comes from [`measured-device-screen`](#measured-device-screen) and [`cmo-range-mismatch-lora`](#cmo-range-mismatch-lora); repeated endpoint noise is exercised in [`mnist-ibm-pcm-cmo-noisy-recovery`](#mnist-ibm-pcm-cmo-noisy-recovery); [`mnist-cmo-literal-floor-noisy-recovery`](#mnist-cmo-literal-floor-noisy-recovery) establishes the hard-clipping failure; and [`mnist-cmo-floor-mitigation`](#mnist-cmo-floor-mitigation) separates that failure from passive floor loading. | Replicate across device/base seeds, train through the affine floor from initialization, validate selector and active-cancellation circuits, and add a physical incremental-pulse model. |

## Finished simulations

### hard-sigmoid-lora-smoke

**Hard-sigmoid physical LoRA smoke test**

- **Finished:** 2026-07-28
- **Outcome:** positive
- **Question:** Can a rank-4 positive physical LoRA branch recover accuracy
  lost when a hard-sigmoid digits DRN is deployed to Wan-2022 ReRAM?
- **Setup:** One `[128, 64, 20]` base model, five W1/W2 device-seed pairs,
  one-day retention, and 20 LoRA recovery epochs.
- **Headline result:** Mean deployed accuracy increased from `89.5556%` to
  `90.1111%`, a gain of `+0.5556` percentage points. Four of five device
  realizations improved; clean selected accuracy was `90.2778%`, giving
  `76.9%` recovery of the mean clean-to-deployed gap.
- **Interpretation:** The experiment provides positive smoke-test evidence
  that the physical branch can recover part of a small deployment gap.
- **Main limitations:** One base-training seed, five device pairs, and the
  held-out split was also used for checkpoint selection and exploratory
  choices.
- **Raw artifacts:** `labs/cases/passive_layerwise_lora_physical_test/`
- **Local ignored detail:**
  `labs/cases/passive_layerwise_lora_physical_test/README.md`

### perfect-diode-hwa-lora-digits

**Perfect-diode FP32, HWA, ReRAM, and LoRA comparison**

- **Finished:** 2026-07-28
- **Outcome:** mixed
- **Question:** Does additive-noise hardware-aware training create a better
  Wan-2022 deployment base, and can physical LoRA recover the remaining gap?
- **Setup:** One `[128, 64, 20]` perfect-diode base, additive-normal HWA with
  selected standard deviation `0.0025`, five device-seed pairs, rank 4, and
  20 recovery epochs.
- **Headline result:** FP32 clean accuracy was `94.7222%`. FP32-to-ReRAM and
  HWA-to-ReRAM both averaged `92.6667%`; HWA-to-ReRAM-to-LoRA averaged
  `92.7778%`, a `+0.1111` point change and `16.7%` recovery of the selected
  HWA clean-to-deployed gap.
- **Interpretation:** LoRA produced a small selected-checkpoint improvement,
  but the chosen HWA perturbation did not improve deployment over the ordinary
  base and left little resolved recovery headroom.
- **Main limitations:** One base-training seed, five device pairs, exploratory
  HWA selection, and no separate final test set.
- **Raw artifacts:** `labs/cases/perfect_diode_hwa_lora_comparison/`
- **Local ignored detail:**
  `labs/cases/perfect_diode_hwa_lora_comparison/README.md`

### measured-device-screen

**Measured-device deployment severity screen**

- **Finished:** 2026-07-28
- **Outcome:** mixed
- **Question:** Do the available measured-device models create a larger,
  well-matched recovery gap than Wan-2022 ReRAM under the selected
  conductance mapping?
- **Setup:** The clean perfect-diode checkpoint was deployed over five
  layerwise seed pairs using Wan-2022 ReRAM, PCM-like, HERMES PCM, and
  CMO/HfOx inference-noise models at selected retention times.
- **Headline result:** Wan-2022, PCM-like, and HERMES means ranged from
  `92.5000%` to `93.3333%` against `94.7222%` clean accuracy. One-day CMO/HfOx
  reached `89.1111%`, but its larger gap was dominated by its nonzero minimum
  conductance rather than stochastic noise alone.
- **Interpretation:** PCM and HERMES did not materially enlarge the gap under
  this calibration. CMO produced a more severe but differently composed
  perturbation.
- **Main limitations:** This was a post-deployment exploratory screen, and its
  later Wan-2022 sampling run is not interchangeable with the four-arm result
  above.
- **Raw artifacts:**
  `labs/cases/perfect_diode_hwa_lora_comparison/device_screen/`
- **Local ignored aggregates:**
  `labs/cases/perfect_diode_hwa_lora_comparison/device_screen/results.json`
  and
  `labs/cases/perfect_diode_hwa_lora_comparison/device_screen/hwa_results.json`

### cmo-range-mismatch-lora

**CMO/HfOx range-mismatch and LoRA probe**

- **Finished:** 2026-07-28
- **Outcome:** negative
- **Question:** Can the positive-only physical LoRA branch recover the larger
  CMO/HfOx deployment gap?
- **Setup:** The mild-HWA perfect-diode checkpoint was mapped to one-day CMO
  with its `9 uS` minimum conductance, then recovered with rank-4 physical
  LoRA over five device-seed pairs.
- **Headline result:** Mean accuracy changed from `89.0556%` after deployment
  to `88.7778%` after LoRA, or `-0.2778` percentage points. About `92.7%` of
  W1 and `94.6%` of W2 targets were raised to the device minimum by the
  mapping.
- **Interpretation:** The current branch can add conductance but cannot
  subtract the excess caused by the CMO floor. The result diagnoses a
  correction-direction mismatch rather than a general failure of low-rank
  recovery.
- **Main limitations:** The CMO runs used a different Python/PyTorch
  environment from most Wan-2022 runs, and the perturbation combines
  post-training range mismatch with device effects.
- **Raw artifacts:**
  `labs/cases/perfect_diode_hwa_lora_comparison/device_screen/cmo_hwa_deployed/`
  and
  `labs/cases/perfect_diode_hwa_lora_comparison/device_screen/cmo_lora_run/`
- **Local ignored detail:**
  `labs/cases/perfect_diode_hwa_lora_comparison/README.md`

### mnist-hwa-lora-rank-study

**Full-MNIST BPTT, HWA, ReRAM, and physical-LoRA rank study**

- **Finished:** 2026-07-29
- **Outcome:** positive
- **Question:** Does physical LoRA recover Wan-2022 deployment loss at MNIST
  scale, and how does recovery change with rank?
- **Setup:** One `[1568, 100, 20]` perfect-diode base-training seed, BPTT,
  additive-normal HWA at standard deviation `0.0025`, Wan-2022 ReRAM at
  one-day retention, ten rank-4 device pairs, and a matched five-pair sweep
  over ranks 1, 2, 4, and 8.
- **Headline result:** Ordinary clean accuracy was `96.150%` and HWA clean
  accuracy was `96.240%`. On the original five device pairs, ordinary
  deployment averaged `92.442%`, HWA deployment `93.162%`, and rank-4 LoRA
  `94.880%`. HWA added `+0.720` points and LoRA added another `+1.718` points.
- **Replication result:** Across ten unfiltered rank-4 device pairs, deployed
  HWA averaged `92.408%` and recovery averaged `94.639%`; all ten improved,
  for a mean paired gain of `+2.231` points and `58.2%` mean gap recovery.
- **Rank result:** Matched five-device means were `94.518%`, `94.732%`,
  `94.880%`, and `94.984%` for ranks 1, 2, 4, and 8. Rank 4 adds `7,152`
  factor conductances, about `4.50%` of the `158,800` dense-edge base
  conductances. Most observed benefit was available by rank 2; the incremental
  paired differences were unresolved at this sample size.
- **Physical controls:** The deployed base weights and hidden bias remained
  bit-identical through all 25 LoRA trajectories. All factor matrices stayed
  within their separate conductance bounds, and zero-output-factor
  initialization reproduced the corresponding deployed accuracy.
- **Interpretation:** This is the strongest positive evidence in the current
  program, but device-seed replication does not replace independent
  base-training replication.
- **Main limitations:** One base-training seed, five devices in the matched
  rank sweep, checkpoint selection and reporting on the canonical MNIST test
  split, and ideal/noise-free LoRA factor devices.
- **Raw artifacts:**
  `labs/cases/mnist_perfect_diode_hwa_lora_comparison/`
- **Local ignored detail:**
  `labs/cases/mnist_perfect_diode_hwa_lora_comparison/README.md`
- **Tracked detail:** [experiment setup](mnist_hwa_lora_experiment_setup.md)

### mnist-hwa-reram-full-finetune

**Full-model post-HWA/ReRAM fine-tuning control**

- **Finished:** 2026-07-29
- **Outcome:** positive control
- **Question:** After HWA and Wan-2022 ReRAM deployment, how much accuracy
  can be recovered by directly fine-tuning the deployed W1, W2, and hidden
  bias without LoRA?
- **Setup:** The original five full-MNIST HWA-to-ReRAM checkpoints, no model
  adapter, 20 epochs of BPTT with the original base learning rates, and
  independent dense-conductance bounds `[1e-7, 1.1]`. The upper bound admits
  the largest realized input value without changing the starting baseline.
- **Headline result:** Mean accuracy increased from `93.162%` after
  deployment to `96.788%`, a gain of `+3.626` points. This is `+1.908`
  points above matched rank-4 LoRA and `+0.548` points above the original
  clean HWA checkpoint.
- **Selection detail:** All cost-selected checkpoints came from epoch 17 or
  18. The mean best transient accuracy was `96.860%`; the canonical result
  continues to use validation-cost selection.
- **Rewrite footprint:** On average, `85.81%` of W1, `99.42%` of W2, and
  `85.40%` of hidden-bias entries changed. Approximately `136,630` of
  `158,900` existing values were rewritten per selected checkpoint. Rank-4
  LoRA instead added `7,152` factor conductances while keeping every base
  value bit-identical.
- **Interpretation:** The remaining post-LoRA accuracy gap is not an
  optimization ceiling of the deployed DRN. Direct access to all base
  parameters can recover and exceed clean-HWA accuracy, but it solves a
  different engineering problem: large-scale reprogramming instead of a
  small frozen-base adapter.
- **Main limitations:** Updates were ideal continuous tensor steps rather
  than ReRAM pulses; update asymmetry, write noise, endurance, energy, and
  reprogramming latency were not modeled. The study also uses one base seed
  and the canonical MNIST test split for selection and reporting.
- **Raw artifacts:** `results/mnist-hwa-reram-full-finetune/`
- **Local ignored detail:**
  `results/mnist-hwa-reram-full-finetune/README.md` and
  `results/mnist-hwa-reram-full-finetune/analysis/summary.json`

### mnist-ibm-pcm-cmo-noisy-recovery

**IBM PCM/CMO endpoint deployment and noisy BPTT recovery**

- **Finished:** 2026-07-29
- **Outcome:** mixed, stability-positive
- **Question:** Starting from the existing FP32 MNIST checkpoint and a shared
  two-epoch IBM-style HWA checkpoint, how do full BPTT and rank-4 physical
  LoRA behave when every post-HWA update is followed by a noisy device write?
- **Setup:** One `[1568, 100, 20]` base, one device seed, IBM AFM PCM and
  CMO/HfOx endpoint models, 20 recovery epochs, 75,000 minibatch updates,
  frozen bias in both recovery arms, and a frozen programmed base for LoRA.
- **Headline result:** HWA deployment was `96.35%` on PCM and `96.38%` on
  CMO. Cost-selected full BPTT reached `96.43%` on both (`+0.08` and
  `+0.05 pp`). Rank-4 LoRA reached `96.30%` on PCM and `96.40%` on CMO
  (`-0.05` and `+0.02 pp`).
- **Physical controls:** All four trajectories remained finite. LoRA W1/W2
  were bit-exact to their initial programmed CUDA realization, full-recovery
  bias was bit-exact to HWA, every factor stayed within bounds, and the noisy
  write stream is exactly resumable.
- **Interpretation:** The study validates stable repeated endpoint writes on
  both full and factor arrays, but the initial device gap was effectively
  zero. The small changes cannot establish recovery efficacy or a
  full-versus-LoRA winner.
- **Main limitations:** One base/device seed, canonical test-set selection,
  endpoint rather than pulse updates, unitless per-tile PCM calibration,
  CMO common-mode-floor cancellation, and no endurance, energy, latency,
  converter-quantization, or IR-drop model.
- **Raw artifacts:** `results/mnist-ibm-pcm-cmo-noisy-recovery/`
- **Tracked detail:**
  [study document](mnist_ibm_pcm_cmo_noisy_recovery.md)
- **Local ignored detail:**
  `results/mnist-ibm-pcm-cmo-noisy-recovery/README.md` and
  `results/mnist-ibm-pcm-cmo-noisy-recovery/analysis/summary.json`

### mnist-cmo-literal-floor-noisy-recovery

**CMO/HfOx literal-floor deployment and noisy recovery**

- **Finished:** 2026-07-29
- **Outcome:** mixed; full-recovery positive, LoRA negative
- **Question:** What changes when the measured `9 µS` CMO floor remains an
  effective DRN connection instead of being removed by normalized-offset
  conversion?
- **Setup:** The same FP32 and two-epoch HWA MNIST checkpoints, CMO device
  seed `17`, `9–88.199997 µS` literal mapping, one-second inference age,
  20 recovery epochs, and a noisy endpoint write after every full-BPTT or
  rank-4 factor update.
- **Headline result:** FP32 fell from `96.15%` to `79.51%`; HWA fell from
  `96.37%` to `80.61%`. Full BPTT recovered to `89.73%`, a `+9.12 pp` gain
  and `57.87%` recovery of the clean-HWA gap. Rank-4 LoRA reached `80.56%`,
  effectively unchanged.
- **Literal hard-clipping diagnosis:** `98.13%` of W1 and `88.10%` of W2
  HWA targets were raised to `9 µS`. Full recovery strengthened a small
  usable subset, but every LoRA factor target remained below the device
  threshold at the last write and therefore programmed to the same floor.
- **Physical controls:** Both runs performed 75,000 noisy writes. The
  full-recovery bias remained bit-exact to HWA; LoRA base and bias remained
  bit-exactly frozen; all factors were finite; and all eight native artifact
  hashes verified.
- **Interpretation:** The earlier normalized-offset mapping both removed the
  floor from the effective equations and avoided literal many-to-one clipping.
  Under literal absolute scaling, full base access provides substantial but
  incomplete recovery. A positive-only LoRA branch cannot subtract the dense
  excess conductance and has no controllable factor state while its targets
  remain below `9 µS`.
- **Main limitations:** One base/device seed, post-hoc mapping rather than
  floor-aware base training, canonical test-set selection, endpoint rather
  than pulse updates, and no endurance, energy, latency, converter, or
  wire-drop model.
- **Raw artifacts:** `results/mnist-cmo-literal-floor-noisy-recovery/`
- **Tracked detail:**
  [study document](mnist_ibm_pcm_cmo_noisy_recovery.md)
- **Local ignored detail:**
  `results/mnist-cmo-literal-floor-noisy-recovery/README.md` and
  `results/mnist-cmo-literal-floor-noisy-recovery/analysis/summary.json`

### mnist-cmo-floor-mitigation

**CMO/HfOx floor mechanism and mitigation ablations**

- **Finished:** 2026-07-30
- **Outcome:** mechanism resolved; affine mitigation positive
- **Question:** Is the literal CMO collapse caused by the finite passive floor
  itself, or by hard clipping of learned conductances, and which passive or
  active interventions can reduce it?
- **Setup:** The same `[1568, 100, 20]` clean HWA checkpoint, CMO device seed
  `17`, `9–88.199997 µS`, one-second inference age, a matched noisy affine
  deployment probe, and exact validation-only replays for mapping,
  floor-ratio, cancellation, gain/iteration, selector, gate, common-mode,
  imbalance, and synthetic readout ablations.
- **Headline result:** Clean HWA was `96.37%`. Deterministic/noisy stored
  literal mapping was `81.83%/80.34%`; deterministic/noisy stored affine
  floor mapping was `95.94%/96.11%`, with `96.12%` on the affine first
  write. The affine physical targets had zero low/high clipping.
- **Electrical result:** Affine mapping retains the floor, but hidden/output
  RMS falls from `0.8761/0.2188` clean to `0.1922/0.02469` noisy affine.
  The floor supplies median `86.79%/68.22%` of the affine hidden/output
  conductance sums, so accuracy alone does not establish analog margin.
- **Distribution result:** Literal and affine stored predictions have
  `0.2971` mean symmetric KL and `0.06449` JS after per-example score-RMS
  normalization, with `81.23%` top-1 agreement. Their combined stored
  conductance histograms have `0.8157` symmetric KL and `0.15614` JS at 128
  bins; the histogram result remains qualitatively stable from 64 to 256
  bins.
- **Mitigation controls:** Deterministic numerator-only correction reached
  `95.92%`, denominator correction `96.34%`, and full numerator-plus-
  denominator correction `96.37%`. The full oracle also scales the bias by
  the affine span and was not applied to a noisy checkpoint. A noiseless
  post-hoc top-`384/60` selector oracle reached `94.93%`.
- **Common-mode mechanism:** Balanced `[x,-x]` input cancels the uniform W1
  floor exactly, paired output subtraction leaves only `2.60%` floor leakage
  relative to differential-score RMS, and deterministic final gate agreement
  is `99.997%`. All 108 deterministic prediction changes lie in the lowest
  clean-margin quintile.
- **Readout-margin control:** Centered score RMS falls `15.2x` and median
  absolute class margin falls `16.9x`. Synthetic output-node noise at
  `sigma=0.01` leaves clean accuracy at `96.35%` but lowers deterministic
  affine accuracy to `91.32%`; at `sigma=0.03`, the comparison is
  `96.29%/48.52%`. These model-voltage controls are not calibrated hardware
  predictions.
- **Interpretation:** Hard clipping of `98.13%` of W1 and `88.10%` of W2,
  rather than passive floor loading alone, caused most of the literal
  accuracy collapse. Ordinary MVM reference subtraction is not exact for a
  DRN because the floor also changes the coordinate-update denominator.
- **Main limitations:** One base/device seed, canonical test-set analysis,
  selector counts explored on the same data, equation-level active terms
  rather than a circuit simulation, synthetic rather than calibrated dynamic
  readout controls, and no converter, IR-drop, power, endurance, or stability
  model.
- **Raw artifacts:** `results/mnist-cmo-floor-mitigation/`
- **Tracked detail:** [mechanism study](conductance_floor_mitigation.md)
- **Local ignored detail:**
  `results/mnist-cmo-floor-mitigation/analysis/replay-v5/summary.json`,
  `results/mnist-cmo-floor-mitigation/analysis/affine-floor-mechanism-v3/summary.json`,
  and
  `results/mnist-cmo-floor-mitigation/affine_floor_probe_v2/20260730T095125.044508Z-ee47b30f-7664fae3/result.json`

### mnist-relu-drn-reset-factorial

**Controlled RESET bias, loss, and amplifier-indexing factorial**

- **Finished:** 2026-08-17
- **Outcome:** historical result optimistic; attribution resolved, mechanisms open
- **Question:** Was the gap between the historical `95.97%` RESET-trained
  result and the newer approximately `85.52%` result caused by hidden bias,
  paired MSE versus teacher KL, or the process-global amplifier-indexing
  defect?
- **Setup:** A seed-42 `2 x 2 x 2` factorial crossed hidden bias off/on,
  hard-label paired MSE/teacher KL, and corrected logical/legacy process-global
  amplifier indexing. Every arm used the same frozen teacher, cohort-A source,
  dense-device assignments, measured pulse-zero initialization, explicit
  minibatch input reset, controlled rebuild, 20 production epochs,
  `voltage_amp = 4`, and `current_amp = 0.25`. Complementary cells were run on
  local, Trex, and Akib GPUs so every main-factor level was balanced within
  each host.
- **Headline result:** Corrected logical indexing reached `93.90%/93.94%`
  paired-MSE test accuracy with bias off/on, but only `86.28%/85.49%` under
  teacher KL. Legacy indexing reached `95.72%/95.70%` under paired MSE and
  `77.59%/80.45%` under KL. The average KL-minus-MSE effect was `-12.36`
  percentage points and the average bias effect was only `+0.52` points.
- **Interaction result:** The average indexing main effect is misleading.
  Legacy indexing raised paired-MSE accuracy by `+1.82/+1.76` points with bias
  off/on, but lowered KL accuracy by `-8.69/-5.04` points. Under the intended
  logical circuit, changing paired MSE to KL cost `7.62` points without bias
  and `8.45` points with bias. Logical paired MSE changed by only `+0.04`
  points when bias was enabled.
- **Interpretation:** The original approximately `95.97%` result was too
  optimistic for the intended DRN: its matching legacy-index/MSE cells reach
  `95.70-95.72%`, while the corrected logical circuit reaches
  `93.90-93.94%`. The working explanation is that the amplifier-index mismatch
  probably made the production network too feed-forward, weakening its
  intended recurrent/equilibrium character in a way that happened to favor
  paired MSE. The factorial establishes the optimism and the loss-by-indexing
  interaction, but it does **not** yet directly establish that feed-forward
  mechanism; voltage, feedback-current, and convergence diagnostics are
  required. The corrected logical KL result (`85.49-86.28%`) agrees with the
  newer run, so that result is representative of this KL objective rather than
  an indexing-induced underestimate.
- **Why-MSE question:** Paired MSE in this study regresses paired output scores
  to hard one-hot labels; it is not an MSE distillation loss. The earlier
  hard-label cross-entropy control reached only `86.88%`, close to teacher KL,
  so label supervision alone does not explain MSE's advantage. "Better" is
  also metric-dependent: without bias, logical KL reduces raw test KL from
  `1.598` for the MSE checkpoint to `0.674`, even though accuracy falls from
  `93.90%` to `86.28%`; the bias-on comparison is `1.601` versus `0.690` KL
  and `93.94%` versus `85.49%` accuracy. The next controlled comparison should
  explain why improved distribution matching produces worse class rankings
  and separate absolute-score anchoring and gradient geometry from target
  choice. Test one-hot paired MSE, MSE to teacher probabilities,
  centered/scaled teacher-logit MSE, hard-label cross-entropy, and
  temperature-scaled KL under both common gradient/update budgets and the
  existing independently selected learning rates. Record validation-only
  accuracy/objective Pareto curves, layerwise gradient scale and direction,
  class margins, score RMS and common offset, equilibrium convergence, and
  the fraction of digital updates erased or reversed by measured-state
  projection.
- **Floor-cancelling architecture question:** Repeat the corrected logical
  MSE/KL comparison with the
  [two-device signed differential interaction](differential_scheme.md),
  alongside the present one-device control. Include both literal pulse-zero
  device pairs and pairs written to a shared reachable zero baseline so floor
  mismatch is not silently hidden. This architecture cancels a matched common
  conductance floor from signed inter-node transfer, but not from
  `G+ + G-` loading or the coordinate-update denominator. A separate
  constant-sum or active full-KCL cancellation arm is needed before claiming
  exact floor cancellation.
- **Epoch budget for these follow-ups:** Ten production epochs are sufficient
  for the exploratory loss and floor-cancelling-architecture screens. In the
  completed factorial, restricting checkpoint selection to epochs 1-10
  changed selected validation accuracy by at most `0.48` percentage points and
  preserved the bias, loss, and indexing conclusions. Every ten-epoch run
  should retain an epoch-boundary resume checkpoint. Promote the matched
  one-device controls and promising new-architecture arms to 20 epochs only
  for confirmatory reporting, and do not compare a ten-epoch arm directly
  against a 20-epoch arm in the final table.
- **Main limitations:** One independent run per factorial cell, one seed, and
  no uncertainty estimate. Main effects are protected against additive host
  offsets, but host-by-interaction effects cannot be separated from factorial
  interactions. The proposed feed-forward explanation remains a hypothesis.
- **Raw artifacts:** `results/mnist-relu-drn-reset-factorial-*-20260817-v3/`
  and `results/factorial_parallel_collected_20260817_v3/`
- **Tracked detail:** [factorial report](mnist_relu_drn_reset_factorial.md)
- **Local ignored detail:**
  `results/factorial_parallel_analysis_20260817_v3/factorial_summary.json`,
  `results/factorial_parallel_analysis_20260817_v3/factorial_arms.csv`, and
  `results/factorial_parallel_analysis_20260817_v3/factorial_accuracy.png`

### mnist-differential-reram-reset-10ep

**Literal-RESET differential ReRAM MNIST screen**

- **Finished:** 2026-08-17
- **Outcome:** completed; stable learning demonstrated, comparative benefit
  still open
- **Question:** Can the implemented two-device signed differential interaction
  train the corrected bias-free MNIST DRN directly from measured pulse-zero
  RESET states for ten production epochs?
- **Setup:** Every physical dual-rail edge used independently assigned cohort-A
  `G+` and `G-` traces, both initialized at pulse zero. The run used seed 42,
  logical model-local amplification, `voltage_amp = 4`,
  `current_amp = 0.25`, explicit input reset, paired hard-label squared error,
  bounded two-group learning-rate selection, and exactly ten production
  epochs. There was no teacher mapping or common-window prewrite.
- **Headline result:** Validation accuracy rose from `9.32%` to a selected
  `92.98%`. Fresh-process test accuracy was `93.94%`, teacher agreement was
  `94.70%`, paired squared error was `0.0933816`, and raw teacher-to-student KL
  was `1.60862`. The diagnostic-only validation-fitted gain reduced test KL to
  `0.138329`; it was not used for training, selection, or accuracy.
- **Integrity:** Both campaign stages exited zero from clean commit
  `187441fd11d27cdcc021aea35609703d5d4d6f8a`. The metrics stream contains
  exactly epochs 1-10, `resume.pt` records epoch 10, and fresh-process
  validation matched the checkpoint's model-local indices and stage scales.
  The selected checkpoint contains four stable conductance keys for the two
  `G+`/`G-` layer pairs.
- **Interpretation:** Literal dual-RESET differential training is numerically
  viable under this protocol. This run does not establish an architecture
  uplift because it lacks a matched ten-epoch one-device arm, and it does not
  isolate common-floor cancellation because independently assigned devices do
  not share an exactly matched RESET baseline. The selector also chose the
  low/low boundary cell of its declared non-expanding grid.
- **Next comparison:** Run a matched ten-epoch one-device control and a
  differential arm written to a shared reachable zero baseline with identical
  seed, device assignment policy, objective, update budget, and selection
  rules. Promote only promising matched arms to the 20-epoch confirmatory
  budget.
- **Raw artifacts:**
  `results/mnist-differential-reram-reset-10ep-20260817-v1/`
- **Tracked detail:**
  [differential RESET report](mnist_relu_drn_reset_differential_10ep.md)

### mnist-differential-reram-initialized-finetune-10ep

**Teacher-initialized differential ReRAM MNIST fine-tuning**

- **Finished:** 2026-08-17
- **Outcome:** completed; ten epochs retained strong accuracy while the longer
  budget improved KL and teacher agreement
- **Question:** How well does the implemented measured `G+ - G-` student
  perform after common-window teacher initialization and exactly ten epochs of
  fine-tuning?
- **Setup:** The frozen bias-free ReLU teacher was mapped into independently
  assigned cohort-A device pairs using their shared affine reachable windows.
  The bias-free `[1568, 100, 20]` student then used pure teacher KL, seed 42,
  batch size 16, learning rates `[4.2e-10, 1.14e-12]`, model-local
  `voltage_amp = 4`, `current_amp = 0.25`, and exactly ten epochs.
- **Headline result:** Mapping began at `89.12%` validation accuracy and
  `0.231839` KL. Epoch 9 was selected at `97.42%` and `0.0138965` validation
  KL. Fresh-process held-out evaluation reached `97.48%` accuracy, `98.76%`
  teacher agreement, and `0.0140905` KL.
- **Integrity:** Training and validation exited zero from clean commit
  `00bc68760f7cd0018e8bf62c1c7c98ad57bd7553`, with exactly ten epoch records,
  an exact epoch-10 resume state, no non-finite metrics, and matching recorded
  topology, teacher, measured-device source, and four stable assignment keys.
- **Epoch-budget comparison:** Its first ten epochs exactly match the earlier
  20-epoch initialized-differential run. Ten epochs had `0.18` percentage
  points higher test accuracy, while 20 epochs improved test KL from
  `0.0140905` to `0.0101188` and agreement from `98.76%` to `98.98%`.
- **Limitation:** This is a deterministic single-seed epoch-budget screen. It
  is not a clean initialization ablation against the literal-RESET run because
  their objectives and readout-gain protocols differ.
- **Raw artifacts:**
  `results/mnist-differential-reram-initialized-finetune-10ep-20260817-v1/`
- **Tracked detail:**
  [initialized differential report](mnist_relu_drn_initialized_differential_10ep.md)

### mnist-single-reram-initialized-finetune-10ep

**Teacher-initialized single-conductance ReRAM MNIST control**

- **Finished:** 2026-08-17
- **Outcome:** completed; ReLU programming and ten KL epochs did not rescue the
  measured one-conductance scheme
- **Question:** How does one measured conductance per physical edge perform
  after per-device-affine ReLU-teacher programming and exactly ten epochs of
  matched KL fine-tuning?
- **Setup:** The frozen bias-free ReLU teacher was lifted into non-negative
  dual rails and mapped into independently assigned cohort-A trace ranges. The
  bias-free `[1568, 100, 20]` student used seed 42, batch size 16, learning
  rates `[2.1e-10, 5.7e-13]`, model-local `voltage_amp = 4`,
  `current_amp = 0.25`, and exactly ten epochs.
- **Headline result:** Validation moved only from `8.10%` and KL `2.242688` to
  `8.16%` and KL `2.242688`. Fresh-process test accuracy was `7.98%`, teacher
  agreement `7.81%`, and KL `2.246097`; the recovery gate failed.
- **Matched comparison:** The initialized `G+ - G-` arm reached `97.48%` test
  accuracy and KL `0.0140905`, an `89.50`-point accuracy gain and `99.37%` KL
  reduction. Ideal one-conductance and differential arms remain tied, so the
  measured benefit is attributed to preserving signal direction through local
  shared-window cancellation rather than to two devices intrinsically.
- **Integrity:** Both stages exited zero from clean commit
  `c507dd9d13825721aa4e25c05f2303302c2a050c`, with exactly ten epoch records,
  an exact epoch-10 resume state, no non-finite metrics, two stable conductance
  keys, and matching model-local topology, teacher, device source, and
  assignment provenance. The first-ten trajectory exactly matches the earlier
  20-epoch arm.
- **Limitation:** This isolates the implemented mapping-plus-architecture
  combination, not device count alone. The one-conductance and pair mappings
  necessarily treat measured floors differently.
- **Raw artifacts:**
  `results/mnist-single-reram-initialized-finetune-10ep-20260817-v1/`
- **Tracked detail:**
  [initialized single-conductance report](mnist_relu_drn_initialized_single_10ep.md)

### mnist-dual-rail-four-vs-eight-common-window-10ep

**Four-device clamped-input collapse versus eight-device differential realization**

- **Finished:** 2026-08-19
- **Outcome:** completed; the four-device arm passed the preregistered
  post-adaptation accuracy-retention screen, while the eight-device arm
  retained better initialization and teacher-logit fidelity
- **Question:** At a clamped dual-rail input `[x, -x]`, can the exact
  eight-to-four port collapse be realized with one measured conductance at
  each of the four post-facing connections while retaining the benefit of the
  eight-device differential implementation?
- **Setup:** Both arms used the same frozen bias-free `784 -> 50 -> 10` ReLU
  teacher, MNIST split, seed 42, minibatch order, solver, model-local
  amplification, raw cohort-A traces and assignment seed, pure teacher-KL
  objective, and ten-epoch budget. The four-device arm required all four
  symmetry-related conductances of one teacher weight to share a single
  reachable window (`dual_rail_quad_common_window`). The eight-device control
  retained one measured `G+`/`G-` pair per physical dual-rail edge. Mapping
  and checkpoint selection used calibration or validation teacher KL; test
  examples were not used for mapping, gain, epoch, or checkpoint selection.
- **Exact boundary result:** On the clamped input subspace, regrouping the
  eight conductances into four pair sums preserves the input-edge energy,
  hidden-node KCL, hidden coordinate coefficients, and derivative in the
  permitted logical-input direction. It does not preserve independent source-
  port currents and does not apply when both rails are free dynamic states.
  Twelve focused equivalence and mapping tests passed.
- **Headline result:** The four-device arm initialized at `61.48%` validation
  accuracy and KL `0.958320`, selected epoch 10 at `97.26%` accuracy,
  `98.74%` teacher agreement, and KL `0.0197817`, and reached `97.46%`
  accuracy, `98.68%` agreement, and KL `0.0183103` on the 10,000-example
  fresh test. The eight-device arm initialized at `89.12%` and KL `0.231839`,
  selected epoch 9 at `97.42%`, `98.94%` agreement, and KL `0.0138965`, and
  reached `97.48%`, `98.76%` agreement, and KL `0.0140905` on test. The
  four-device test-accuracy deficit was only `0.02` percentage points, inside
  the preregistered `1.0`-point retention threshold.
- **Reachable-window diagnostic:** Averaged over the two layers, the four-cell
  intersection had an `11.195 uS` mean span and `5.918%` empty-window rate,
  versus `18.225 uS` and `1.451%` for the eight-device pairwise windows. No
  nominal targets were clipped. The stricter intersection and independent
  nearest-state projection explain why the four-device initialization was
  `27.64` accuracy points below the eight-device initialization even though
  adaptation nearly closed the final classification gap.
- **Earlier four-device comparison:** Relative to the prior four-device
  pairwise-window mapping, the shared quad window improved initialization
  from `34.20%` to `61.48%`, test accuracy from `96.96%` to `97.46%`, and test
  KL from `0.041213` to `0.0183103`.
- **Interpretation:** Four measured conductances per teacher weight retain
  essentially all classification benefit after adaptation while halving the
  conductance count relative to the eight-device realization. They are not a
  physical initialization-equivalent replacement: using one device for each
  collapsed pair sum loses reachable range and mismatch averaging, and the
  four-device test KL remains `29.95%` higher. The eight-device realization is
  therefore preferred for initialization-only feed-forward fidelity; the
  four-device realization is attractive when device count matters and
  measured fine-tuning is available. This is an exploratory single-seed
  result, not a universal equivalence claim.
- **Integrity:** All four train/test stages exited zero from clean commit
  `bb88b007fa424f88bec86f8602690dac988b022a`. An immediate `--resume` audit
  validated fingerprints and artifact hashes and reused every stage without
  creating another attempt. The selected four- and eight-device weight hashes
  are `e95eb276ed30d20c024495480c1f7d985ced853b761b8a3d5f2151aa9614fcb8`
  and `898131127b636893b3cc50e895635243e1ee0e0efd4ce4d9d1d35129a50195f9`.
- **Raw artifacts:**
  `results/mnist-dual-rail-four-vs-eight-common-window-10ep-20260819-v1/`
- **Tracked detail:**
  [four-versus-eight report](dual_rail_input_four_vs_eight_devices.md)

## Shared validity notes

- These studies are exploratory rather than final paper-facing evidence.
- Each comparison currently uses a single clean base-training seed; none
  includes within-study base-seed replication.
- The recorded native manifests identify the source commit and dirty-state
  fingerprint used for each run. This ledger does not independently validate
  them.
- Most base, hard-sigmoid, and Wan-2022 runs used Python 3.12.12,
  PyTorch 2.5.1, and AIHWKit 1.1.0 where required. The CMO screen and CMO LoRA
  runs used Python 3.13.5, PyTorch 2.12.1, and AIHWKit 1.1.0.
- HWA used additive Gaussian parameter perturbations rather than the deployed
  AIHWKit device model.
- The earlier Wan studies applied device effects only to the base DRN and
  kept LoRA factors ideal. The IBM PCM/CMO study is the exception: every
  factor update receives a fresh endpoint write, including programming,
  relaxation, and read terms for CMO.
- The IBM PCM/CMO runs and literal-floor follow-up used Python 3.12.13 and
  PyTorch 2.11.0+cu128 on Trex. Their exact equation-level models did not
  import AIHWKit at runtime.
- Ten superseded MNIST attempts retain stale native `running` statuses despite
  having no active process. They are treated as interrupted attempts, not
  active or completed evidence.
- The raw directories and checkpoints are local and ignored by Git.

## Related documents

- [Current LoRA/HWA simulations](current_simulations.md)
- [Current research state and personal notes](current_state.md)
- [Raw result home](../results/README.md)
- [Physical layerwise low-rank recovery](passive_layerwise_low_rank_recovery.md)
- [Digital low-rank recovery](digital_low_rank_recovery.md)
- [Experiment runtime](experiment_runtime.md)
