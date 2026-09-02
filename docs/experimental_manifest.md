# Finished LoRA/HWA Simulations and Research Progress

Last updated: 2026-08-28

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

New workflow-managed studies begin with a tracked plan under `studies/`. Once
the exact declared coverage is complete, a human writes the outcome, final
interpretation, limitations, and next steps in the study review. Running
`python -m ebl study finalize` adds an idempotent marked entry to this ledger.
Every such entry retains the **initial hypothesis** as well as the **final
interpretation**; negative and inconclusive outcomes are first-class results.
The operational procedure is in
[`experiment_workflow.md`](experiment_workflow.md).

## Study-family terminology

The **symmetry-reference-anchored memristor studies**, shortened to
**reference-anchored studies**, cover IBM OM schemes whose stored zero-state
baseline is derived from an identity's intrinsic fitted symmetry reference
`r_i`.  Exact fixed-reference arms use `B_i=r_i`; local compensation uses a
nearby stored `B_i≈r_i` while keeping `r_i` immutable.  The family includes
both four-device rail-quad and eight-device active/reference-pair topologies;
the name does not imply that every member contains a separate reference branch
per edge.  The family contract and on-chip-learning boundary are documented in
[`ibm_om_reference_anchored_studies.md`](ibm_om_reference_anchored_studies.md).

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
| `M6` | Characterize base-device degradation mechanisms, then extend them to realistic factor-device effects aligned with a positive-only correction path. | partial | Base-device evidence comes from [`measured-device-screen`](#measured-device-screen) and [`cmo-range-mismatch-lora`](#cmo-range-mismatch-lora); repeated endpoint noise is exercised in [`mnist-ibm-pcm-cmo-noisy-recovery`](#mnist-ibm-pcm-cmo-noisy-recovery); [`mnist-cmo-literal-floor-noisy-recovery`](#mnist-cmo-literal-floor-noisy-recovery) establishes the hard-clipping failure; [`mnist-cmo-floor-mitigation`](#mnist-cmo-floor-mitigation) separates that failure from passive floor loading; [`mnist-wan-cmo-teacher-initialized-seed17-10ep`](#mnist-wan-cmo-teacher-initialized-seed17-10ep) compares retained-floor endpoints from a common teacher initialization; and [`mnist-relu-to-bounded-fp32-map-20260824-v1`](#mnist-relu-to-bounded-fp32-map-20260824-v1) establishes the bounded-FP32 logical initialization control. | Replicate across device/base seeds, add the no-HWA BPTT control, validate selector and active-cancellation circuits, and add a physical incremental-pulse model. |
| `M7` | Determine how quantization-aware training should continue after deployment on a replacement array. | partial | [`mnist-ibm-om-cell-aware-exact-bounds-transfer-20260824-v1`](#mnist-ibm-om-cell-aware-exact-bounds-transfer-20260824-v1) shows that fixed-array QAT reaches `93.31%` on its development assignment but `74.154%` after frozen transfer, while remaining `6.552` points above continuous HWA on the replacement array. | Define a QAT state and update rule compatible with persistent on-chip conductances, then compare it with a frozen control and an oracle STE-QAT recovery ceiling from one shared deployed state. |

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

### mnist-wan-cmo-head-to-head-seed17-10ep

**Historical independently trained retained-floor CMO versus Wan-2022 control**

- **Finished:** 2026-08-20
- **Outcome:** CMO deployment-positive; Wan recovery-positive
- **Question:** Under identical DRN, HWA, retention, mapping, seed, and noisy
  BPTT budgets, how do the CMO/HfOx and Wan-2022 endpoint models compare?
- **Setup:** One `[1568, 100, 20]` FP32 checkpoint, a shared two-epoch
  additive-normal HWA checkpoint, `V_amp=4`, `I_amp=0.25`, one-day device
  age, device seed 17, affine maps retaining `9–88.199997 µS` CMO and
  `1–40 µS` Wan floors, and ten full-BPTT epochs with a fresh endpoint write
  after every minibatch.
- **Headline result:** Clean FP32 was `96.15%`. Its first device write was
  `96.02%` CMO and `87.80%` Wan. Clean HWA was `95.67%`; its first device
  write was `96.19%` CMO and `75.78%` Wan. Cost-selected noisy BPTT reached
  `95.56%` CMO and `94.23%` Wan, both reproduced by fresh validation.
- **Interpretation:** BPTT changed deployed HWA accuracy by `-0.63 pp` on
  CMO and `+18.45 pp` on Wan. The direct-write programming RMSE relative to
  the ideal affine target was `0.00963` CMO versus `0.03145` Wan DRN units;
  stochastic endpoint distortion, not finite-floor ratio, dominates the
  one-seed gap. This is retained as an endpoint-noise control but superseded
  as the primary initialization protocol because its DRN was trained
  independently rather than mapped from the frozen ReLU teacher.
- **Main limitations:** One device seed, a generic IBM-style HWA modifier
  rather than a Wan-fitted modifier, test-set checkpoint selection, endpoint
  rather than pulse updates, and a one-device nonnegative-edge encoding
  rather than NeuRRAM's signed differential pair.
- **Raw artifacts:**
  `results/mnist_wan_cmo_head_to_head_seed17_10ep/`
- **Tracked detail:**
  [study document](mnist_wan_cmo_head_to_head.md)

### mnist-wan-cmo-teacher-initialized-seed17-10ep

**Teacher-initialized retained-floor CMO versus Wan-2022 comparison**

- **Finished:** 2026-08-20
- **Outcome:** HWA modestly positive; noisy BPTT strongly positive; CMO ahead
- **Question:** When every DRN arm is initialized from the same frozen ReLU
  teacher, how useful are HWA and post-deployment noisy BPTT on CMO/HfOx and
  Wan-2022 endpoint models?
- **Setup:** One frozen bias-free `784 -> 50 -> 10` ReLU teacher, direct lift
  into a bias-free `[1568,100,20]` dual-rail DRN, `V_amp=4`, `I_amp=0.25`,
  logical `0..110 uS`, true 5,000-example validation selection and untouched
  10,000-example test evaluation, two generic 3%-modifier HWA epochs, retained
  affine CMO `9..88.199997 uS` and Wan `1..40 uS` floors, one-day age, device
  seed 17, and ten full-BPTT epochs with a fresh endpoint rewrite after every
  minibatch.
- **Headline result:** The teacher reached `97.36%`; its immediate ideal DRN
  mapping reached `96.91%` with teacher KL `0.009345`, and clean HWA reached
  `97.34%` with KL `0.002681`. The ideal-map first write was `91.15%` CMO and
  `78.62%` Wan. The HWA first write was `92.18%` and `80.31%`. Selected noisy
  BPTT reached `96.46%` CMO and `91.84%` Wan, with KL `0.032341` and
  `0.186471`.
- **Intervention result:** HWA improved the first write by `+1.03 pp` CMO and
  `+1.69 pp` Wan. Noisy BPTT then added `+4.28 pp` and `+11.53 pp`, recovering
  `82.9%` and `67.7%` of the respective clean-HWA accuracy gaps. HWA necessity
  remains unresolved because the study did not include a no-HWA BPTT arm.
- **Device interpretation:** Wan programming-error RMSE relative to the ideal
  affine target was `5.130 uS` in effective DRN units, `4.45x` CMO's
  `1.152 uS`. CMO nevertheless had the larger deterministic floor-mapping
  error (`10.681 uS` versus `2.617 uS`). The common affine floor is structured
  and largely rejected by the dual-rail computation, whereas stochastic
  endpoint distortion changes relative edge strengths.
- **Physical controls:** Both BPTT arms performed exactly `34,380` rewrite
  steps, all 17 campaign stages completed from clean source commit
  `f60d4406c5a5cde7b75726626e4adeae5f4e4c03`, and fresh-process test
  validation loaded the explicitly selected named checkpoints.
- **Main limitations:** One teacher and device seed, generic rather than
  device-matched HWA, endpoint rather than pulse updates, single-conductance
  nonnegative physical edges, and no endurance, energy, ADC/DAC, IR-drop, or
  dynamic inference-noise model.
- **Raw artifacts:**
  `results/mnist_wan_cmo_teacher_initialized_seed17_f60d4406/`
- **Tracked detail:**
  [study document](mnist_wan_cmo_teacher_initialized.md)

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

### mnist-four-device-reram-cohort-b-quad-common-window-10ep

**Four-device common-window deployment and adaptation on held-out cohort B**

- **Finished:** 2026-08-19
- **Outcome:** completed; immediate cross-cohort transfer failed, but ten
  matched off-chip epochs recovered `99.76%` of the lost classification
  accuracy while retaining four conductances per teacher weight
- **Question:** Can the selected cohort-A four-device checkpoint be
  re-encoded on held-out cohort-B devices using one common reachable window
  per four-cell block, and can device-constrained fine-tuning recover any
  transfer loss?
- **Setup:** The source was the selected four-device checkpoint from the
  preceding comparison. Cohort B used the same teacher, split and assignment
  seeds, the `halves`/`paired` model-local layouts, raw measured curves,
  global-nearest projection, inherited learning rates `[2.1e-10, 5.7e-13]`,
  pure teacher KL, and ten epochs. A zero-learning-rate control isolated the
  immediate deployment state. There was no cohort-B learning-rate search.
- **Headline result:** Fresh-test accuracy/agreement/KL changed from
  `97.46%`/`98.68%`/`0.0183103` on cohort A to
  `35.79%`/`35.72%`/`1.761729` immediately on cohort B. The selected tenth
  cohort-B epoch reached `97.31%`/`97.61%`/`0.0557048`. Fine-tuning therefore
  gained `61.52` accuracy points and reduced KL by `96.84%`, finishing only
  `0.15` accuracy points below the source.
- **Four-versus-eight comparison:** The earlier eight-device paired-common-
  window arm reached `42.96%` immediately and `97.81%` after adaptation.
  Four devices were `7.17` points lower before adaptation and `0.50` points
  lower afterward, while using half the conductances. Its final KL was
  `76.50%` higher (`0.0557048` versus `0.03156`).
- **Mechanism:** Cohort-A and cohort-B four-cell windows have similar spans
  and empty-window rates. Re-encoding nevertheless contracts layerwise
  effective signed-drive RMS from `3.401/4.023 uS` to `0.552/0.550 uS`
  (`6.16x/7.32x`) and raises mean four-cell load by `9.48%/8.84%`. The failure
  is therefore device-specific signed rescaling plus changed passive loading,
  not simply an unusually narrow cohort-B window. Adaptation finds a new
  cohort-B-specific solution rather than restoring the source matrix.
- **Interpretation:** Differential initialization cancels a nominal common
  baseline but does not preserve symmetry, signed scale, or circuit
  denominators after independent measured-state projection. The four-device
  scheme is expressive after adaptation; the eight-device scheme remains
  better for immediate transfer and final logit fidelity in this single-seed
  comparison.
- **Integrity:** All four stages exited zero from clean implementation commit
  `8325b2a2d503a0b1afe49ecb052bf2bfe3fd863a`, with exactly ten epoch records,
  finite metrics, a fresh 10,000-example test for both endpoints, and only
  `attempt-001` directories. A clean `--resume` audit validated fingerprints
  and result hashes and reused every stage.
- **Raw artifacts:**
  `results/mnist-four-device-reram-cohort-b-quad-common-window-10ep-20260819-v1/`
- **Tracked detail:**
  [four-device cohort-B report](mnist_four_device_cohort_b_adaptation.md)

<!-- BEGIN EBL STUDY mnist-four-device-cohort-b-transfer-5seed-20260819-v1 -->
### mnist-four-device-cohort-b-transfer-5seed-20260819-v1

**Four-device cohort-A to cohort-B transfer across five assignments**

- **Finished:** 2026-08-19
- **Evidence class:** `exploratory`
- **Outcome:** supported
- **Initial hypothesis:** Reprojecting the selected cohort-A four-device quad-common-window checkpoint onto held-out cohort-B devices will reduce mean held-out test accuracy by more than 1 percentage point after scalar recalibration alone.
- **Completion criteria:**
  - The source checkpoint has one fresh completed 10,000-example test evaluation.
  - All five cohort-B assignment seeds complete a zero-learning-rate deployment with exactly one batch and no conductance update.
  - All five transferred checkpoints have one fresh completed 10,000-example test evaluation.
  - Every run retains the source checkpoint, teacher checkpoint, device-data, config, and study provenance in its native manifest.
  - The final interpretation reports accuracy loss distribution as exploratory single-model evidence.
- **Coverage:** 11 declared run(s) completed; 0 failed attempt(s) retained.
- **Final interpretation:** The predeclared five-assignment series supports the hypothesis. The fresh cohort-A source test accuracy was 97.46%. Zero-update reprojection onto cohort-B four-device assignments produced test accuracies of 35.79%, 41.65%, 50.85%, 45.28%, and 40.65%: a mean of 42.844%, population standard deviation of 5.021 percentage points, and range of 35.79% to 50.85%. The corresponding accuracy loss averaged 54.616 percentage points and ranged from 46.61 to 61.67 points. Mean teacher agreement fell from 98.68% to 43.02%, while mean teacher-student KL rose from 0.0183103 to 1.665943. Scalar logit recalibration did not recover classification accuracy. All five zero-learning-rate deployment runs retained identical initial and final conductance summaries and reported zero adaptation change. Cohort-B quad-window diagnostics averaged 11.288 uS and 11.064 uS mean span with 4.952% and 6.080% empty-window fractions in the two layers, respectively. Because these aggregate window statistics are similar to the earlier cohort-A four-device result, the observed collapse is more consistent with sensitivity to device-level reassignment and nearest-state reprojection than with a simple cohort-wide loss of reachable range; that mechanism remains an inference rather than a causal isolation.
- **Main limitations:** This is exploratory evidence from one trained four-device checkpoint, one teacher, one MNIST split, and five deterministic device-assignment seeds drawn from the same held-out cohort-B trace pool; the assignments are not independent model-training seeds. The simulator uses global-nearest endpoint projection of measured program-and-verify traces, not measured incremental-pulse dynamics or a fabricated circuit. The study measures immediate zero-update transfer after scalar gain recalibration and does not test cohort-B adaptation. Aggregate common-window statistics do not identify which layer, weights, or device mismatches cause the loss.
- **Next steps:**
  - Run a matched five-assignment eight-device cohort-B transfer control to determine whether the four-device collapse is specifically worsened by the quad-common-window realization.
  - Fine-tune each transferred four-device cohort-B checkpoint with a fixed matched budget and report the initialization-to-transfer-to-recovery trajectory.
  - Measure layerwise realized-weight error, score-direction change, and voltage operating points for the five assignments before changing the mapping or recovery method.
- **Raw artifacts:** `results/mnist-four-device-cohort-b-transfer-5seed-20260819-v1/`
- **Workflow summary:** `results/mnist-four-device-cohort-b-transfer-5seed-20260819-v1/analysis/summary.json`
<!-- END EBL STUDY mnist-four-device-cohort-b-transfer-5seed-20260819-v1 -->

<!-- BEGIN EBL STUDY mnist-four-device-cohort-a-sign-sgd-20260819-v1 -->
### mnist-four-device-cohort-a-sign-sgd-20260819-v1

**Four-device cohort-A fine-tuning with sign-only SGD**

- **Finished:** 2026-08-19
- **Evidence class:** `exploratory`
- **Outcome:** supported
- **Initial hypothesis:** At least one fixed-magnitude signSGD schedule will recover the cohort-A four-device initialization to at least 95% fresh test accuracy within ten epochs, showing that gradient direction alone can provide useful same-cohort adaptation under global-nearest measured-state projection.
- **Completion criteria:**
  - Both train arms reproduce the same deterministic cohort-A four-device initialization and complete exactly ten full epochs.
  - Both selected checkpoints have a fresh completed 10,000-example test evaluation.
  - Every checkpoint records the measured_cohort_a_sign_sgd backend, quad-common-window layout, device-data identity, assignment identity, and model-local amplification metadata.
  - The interpretation reports whether either sign-only schedule reaches at least 95% test accuracy and distinguishes update-rule effects from step-scale effects.
  - The result is labeled exploratory single-model evidence and retains the ordinary-SGD result as a historical matched control rather than a newly randomized replicate.
- **Coverage:** 4 declared run(s) completed; 0 failed attempt(s) retained.
- **Final interpretation:** The predeclared hypothesis is supported. Both sign-only schedules reproduced the deterministic cohort-A four-device initialization at 61.48% validation accuracy and KL 0.958320, completed ten full epochs, and exceeded 95% on a fresh 10,000-example test. Using the ordinary-SGD arm's numeric rates [2.1e-10, 5.7e-13], signSGD reached 95.46% test accuracy, 96.67% teacher agreement, and KL 0.0784538. Using a balanced 0.21 nS shadow step in both layers reached 95.77%, 97.03%, and KL 0.0640143. Balancing improved accuracy by 0.31 points and reduced KL by 18.41% relative to matched-rate signSGD. Direction alone therefore provides useful same-cohort adaptation, recovering 34.29 accuracy points from initialization in the better arm. Gradient magnitude still matters for fidelity: the balanced sign arm remained 1.69 accuracy points below the matched ordinary-SGD historical control at 97.46%, and its KL was 3.50 times higher than the control's 0.0183103. The matched-rate arm never changed the measured output-layer state because its 0.00057 nS sign step was too small; balancing produced a 3.804% output-layer state-change fraction and explains part of its improvement. Neither sign arm clipped its shadow or reached conductance bounds.
- **Main limitations:** This is exploratory evidence from one teacher, model seed, MNIST split, cohort-A device assignment, and two hand-selected sign-step schedules. The ordinary-SGD comparison is an earlier deterministic matched run rather than a concurrently rerun randomized control, although both new arms exactly reproduced its initialization. SignSGD operates on an ideal digital conductance shadow and each minibatch globally reprojects the target to the nearest of 5,000 measured endpoint states. It is not a sequential one-pulse potentiation/depression rule; raw nonmonotonic curves permit large pulse-index jumps. The result therefore shows that gradient signs are sufficient within this endpoint-projection simulator, not that sign-only local hardware pulses would achieve the same accuracy.
- **Next steps:**
  - If the intended hardware rule is one pulse step in the gradient-sign direction, implement it as a separate sequential-pulse backend and compare it without reusing this result as evidence for pulse-local learning.
  - Run a small validation-only layerwise sign-step sweep, especially increasing the first-layer step, because balanced signSGD changed first-layer measured states on 3.11% of cell-updates versus 15.28% for ordinary SGD.
  - Replicate the chosen sign-step schedule across independent teacher/model seeds and device assignments before making a general optimizer claim.
- **Raw artifacts:** `results/mnist-four-device-cohort-a-sign-sgd-20260819-v1/`
- **Workflow summary:** `results/mnist-four-device-cohort-a-sign-sgd-20260819-v1/analysis/summary.json`
<!-- END EBL STUDY mnist-four-device-cohort-a-sign-sgd-20260819-v1 -->

<!-- BEGIN EBL STUDY mnist-four-device-cohort-a-one-pulse-down-20260819-v1 -->
### mnist-four-device-cohort-a-one-pulse-down-20260819-v1

**Four-device cohort-A fine-tuning with one-pulse conductance decreases**

- **Finished:** 2026-08-19
- **Evidence class:** `exploratory`
- **Outcome:** refuted
- **Initial hypothesis:** Strict one-pulse-down-or-hold fine-tuning will improve selected validation accuracy by at least 5 percentage points over its own isotonic cohort-A four-device initialization within ten epochs, while producing no conductance increase and no pulse-index jump larger than one.
- **Completion criteria:**
  - The train arm completes exactly ten full epochs from its recorded isotonic cohort-A four-device initialization.
  - The selected checkpoint has one fresh completed 10,000-example test evaluation.
  - Every checkpoint records the measured_cohort_a_one_pulse_down backend, quad-common-window layout, device-data identity, assignment identity, and model-local amplification metadata.
  - Both layer reports show max_abs_pulse_jump at most one, zero conductance increases, no fine-tuning global-nearest projection, and no digital-shadow accumulation.
  - The interpretation tests the predeclared five-percentage-point validation-improvement threshold and labels comparisons with raw-trace SGD/signSGD as unmatched historical context.
- **Coverage:** 2 declared run(s) completed; 0 failed attempt(s) retained.
- **Final interpretation:** The predeclared hypothesis is refuted. The strict one-pulse-down-or-hold backend completed all ten epochs and satisfied its physical-update invariants exactly, but validation-KL selection retained the isotonic initialization at epoch -1: selected validation accuracy remained 38.32%, so the required improvement was 0 rather than at least 5 percentage points. The selected initialization reached 38.40% accuracy, 38.56% teacher agreement, and KL 1.535809 on a fresh 10,000-example test. There was a transient classification improvement after one epoch: live validation accuracy rose by 21.52 points to 59.84%, but KL simultaneously worsened from 1.555411 to 6.477425, so that state was not selected. Continued one-sided updates then collapsed accuracy to 10.30% and raised KL to 105.524120 by epoch 10. The mechanism is directly visible in the programming diagnostics. Final-pulse occupancy increased from 4.794% to 99.881% in the first layer and from 5.350% to 99.750% in the second. Across all cell-updates, 49.168% and 47.760% requested a downward pulse, but only 6.952% and 7.307% could still advance; 42.216% and 40.453% were blocked at the final pulse. Isotonic plateaus were also dominant: only 1.992% and 1.972% of applied pulse advances produced a measurable conductance decrease. The implementation nevertheless did exactly what was requested: both layers report maximum pulse jump 1, zero conductance increases, zero fine-tuning projection error, no global-nearest fine-tuning, no digital-shadow accumulation, and no use of learning-rate magnitude. The result therefore indicates that an unconditional per-minibatch decrease-only rule is not a viable ten-epoch fine-tuning mechanism in this setup; it rapidly exhausts the available pulse trajectory. The earlier raw-trace ordinary-SGD and signSGD accuracies are not matched controls because isotonic fitting also changed the initialization substrate: the present initialization was 38.32% validation accuracy versus 61.48% in those raw-trace studies.
- **Main limitations:** This is exploratory evidence from one teacher, model seed, MNIST split, cohort-A device assignment, and one ten-epoch update schedule. Checkpoint selection uses validation teacher-student KL, so the epoch-1 accuracy increase was intentionally not selected; no fresh test was run on that transient live state. The source consists of measured program-and-verify endpoint sequences, while this experiment traverses isotonic non-increasing fits to those sequences. PAVA introduces long exact plateaus and changes both initialization and step statistics; the simulation is not independent evidence that one fabricated-device pulse produces the fitted next state. The update fires on every eligible positive minibatch gradient without a pulse budget, confidence threshold, persistence rule, stochastic write probability, recalibration, or upward/homeostatic correction. Historical raw-trace SGD and signSGD results differ in preprocessing and update semantics and therefore provide context only, not causal optimizer comparisons.
- **Next steps:**
  - Run a predeclared short-horizon sweep over exact minibatch budgets before saturation, retaining the same backend and reporting both KL and accuracy, to determine whether the epoch-1 accuracy gain occurs in a reproducible usable window.
  - Test a pulse-budget or sparse eligibility rule that still permits only one downward pulse per accepted update but prevents nearly every cell from exhausting its trajectory within three epochs.
  - Evaluate a complementary-rail logical update that realizes both logical gradient directions by decreasing one of the appropriate physical conductances, while continuing to prohibit conductance increases on every individual cell.
- **Raw artifacts:** `results/mnist-four-device-cohort-a-one-pulse-down-20260819-v1/`
- **Workflow summary:** `results/mnist-four-device-cohort-a-one-pulse-down-20260819-v1/analysis/summary.json`
<!-- END EBL STUDY mnist-four-device-cohort-a-one-pulse-down-20260819-v1 -->

<!-- BEGIN EBL STUDY mnist-four-device-cohort-a-gradient-threshold-sweep-20260819-v1 -->
### mnist-four-device-cohort-a-gradient-threshold-sweep-20260819-v1

**Four-device cohort-A one-pulse-down gradient-threshold sweep**

- **Finished:** 2026-08-20
- **Evidence class:** `exploratory`
- **Outcome:** supported
- **Initial hypothesis:** A fixed layer-specific positive-gradient threshold can prevent the destructive saturation seen in the zero-threshold one-pulse-down control while retaining useful decrease-only adaptation: at least one p90, p99, or p99.9 arm will improve selected validation accuracy by at least 5 percentage points over the shared isotonic initialization and keep epoch-10 last-pulse saturation below 50 percent in both layers.
- **Completion criteria:**
  - All three threshold-train runs complete exactly ten full epochs from numerically identical isotonic cohort-A four-device initializations.
  - Each selected checkpoint receives one fresh completed 10,000-example test evaluation using its predeclared matching config.
  - Every threshold value and its raw-gradient strict-greater-than semantics are present in config, checkpoint provenance, and programming diagnostics.
  - Both layer reports in every arm show max_abs_pulse_jump at most one, zero conductance increases, no fine-tuning global-nearest projection, and no digital-shadow accumulation.
  - The analysis tests whether any threshold arm improves selected validation accuracy by at least five percentage points over the shared initialization and whether any arm keeps epoch-10 last-pulse saturation below 50 percent in both layers.
  - The final interpretation reports the validation-derived threshold-calibration limitation and treats the fresh test split as the held-out accuracy estimate.
- **Coverage:** 6 declared run(s) completed; 1 failed attempt(s) retained.
- **Final interpretation:** The predeclared hypothesis is supported. The three arms used strict instantaneous raw-gradient gates with fixed layer-specific thresholds and otherwise identical isotonic cohort-A four-device initialization. All began at exactly 38.32% validation accuracy and KL 1.555411. P99 and p99.9 both exceeded the required five-point selected-validation gain while keeping epoch-10 final-pulse occupancy below 50% in both layers. P99 selected epoch 7 at 82.22% validation accuracy and KL 0.534162, reached 83.65% accuracy and KL 0.501183 on the fresh 10,000-example test, ended at 77.00% validation accuracy, and had only 6.995%/8.800% final saturation. P99.9 selected the epoch-10 state at 72.98% validation accuracy and KL 0.924399, reached 74.38% test accuracy, and had 4.945%/5.900% saturation. P90 produced the strongest selected checkpoint at epoch 1: 84.18% validation accuracy, KL 0.437640, and 85.91% fresh test accuracy. It was not stable for a fixed ten-epoch run, however; continued one-way updates drove saturation to 93.705%/81.650% and final validation accuracy to 13.76%. Thus p90 is an early-stopping regime, while p99 is the best observed balance for a fixed ten-epoch protocol. Relative to the finalized matched zero-threshold control, selected test accuracy improved from 38.40% to 85.91%, 83.65%, and 74.38% for p90, p99, and p99.9. Across every arm and both layers, maximum pulse jump was one, conductance-increase count was zero, and the backend used neither fine-tuning global-nearest projection, digital-shadow accumulation, nor learning-rate magnitude. The mechanism is clear in the write statistics: p99 admitted only 0.0852%/0.2220% of cumulative cell-updates and ended with low saturation, whereas cumulative p90 eligibility rose to 21.351%/17.062% and still exhausted most traces. The threshold therefore repairs the unconditional update rule, but its useful operating point depends strongly on whether early stopping is allowed.
- **Main limitations:** This is exploratory evidence from one teacher, model and data seed, MNIST split, cohort-A device split, virtual-device assignment, batch size, fixed gain, and solver. Thresholds were calibrated from a read-only replay of the same validation split used for checkpoint selection, so validation improvements are not a fully untouched estimate; fresh test metrics provide the held-out accuracy check, but there are no repetitions or confidence intervals. The absolute raw-gradient values are specific to this loss scaling, batch size, parameterization, and operating point and should not be transferred unchanged to another setup. The measured source contains program-and-verify endpoint sequences, while the update traverses PAVA-fitted non-increasing curves; an adjacent simulated pulse index is not independent evidence that one open-loop physical pulse reproduces that fitted state. Long exact PAVA plateaus remain, so only about 2.0% to 5.7% of applied pulse advances changed conductance. The p99.9 launcher was interrupted after epoch 1 and recovered through the guarded exact epoch-boundary checkpoint; the failed attempt and recovery receipt are retained, but no duplicate uninterrupted p99.9 production run was performed for bitwise trajectory comparison.
- **Next steps:**
  - Run a predeclared p95/p97/p98 sweep with the same initialization and ten-epoch budget to locate the transition between p90's high early accuracy and p99's low saturation.
  - Repeat the leading p90 early-stop and p99 fixed-budget protocols across several data seeds and device assignments, reporting confidence intervals on selected test accuracy, write count, and saturation.
  - Test a pulse-budget or adaptive threshold schedule that retains p90-like early improvement but raises the gate or stops writes before saturation, without permitting any conductance increase.
- **Raw artifacts:** `results/mnist-four-device-cohort-a-gradient-threshold-sweep-20260819-v1/`
- **Workflow summary:** `results/mnist-four-device-cohort-a-gradient-threshold-sweep-20260819-v1/analysis/summary.json`
<!-- END EBL STUDY mnist-four-device-cohort-a-gradient-threshold-sweep-20260819-v1 -->

<!-- BEGIN EBL STUDY ibm-reram-program-verify-noise-20260821-v2 -->
### ibm-reram-program-verify-noise-20260821-v2

**Short IBM ReRAM pulse-count program-and-verify endpoint model**

- **Finished:** 2026-08-21
- **Evidence class:** `model_based_aihwkit_preset`
- **Outcome:** mixed
- **Initial hypothesis:** At the primary half-step acceptance tolerance, an adaptive fixed-amplitude pulse-count controller reduces verify reads relative to one-pulse verify without materially increasing held-out endpoint error or failure probability, and a target-dependent Gaussian is adequate only when its predeclared coverage and Wasserstein gates pass.
- **Completion criteria:**
  - All four declared arms complete the 41-target, half-step-tolerance, two-start, two-controller design on 1024 device identities with four repeats: exactly 671744 trajectories per arm and 2686976 trajectories in total.
  - The complete four-arm local campaign reaches terminal artifacts within 24 elapsed hours of the first production-arm launch, with exact launcher handles, logs, and start/finish timestamps retained.
  - Each arm records CUDA pulse-plant execution, a pinned AIHWKit-1.1.0 population-sampling receipt, the sampled population hash, and exact per-trajectory conditioning and programming seeds.
  - Trajectory and verify-event artifacts preserve accepted, failed, saturated, non-finite, and corrupt outcomes; every persisted ledger passes the exact integrity contract.
  - Gaussian adequacy is decided only by the predeclared coverage and normalized Wasserstein gates on the 204 held-out validation identities and their four repeats at every target.
  - Wan-2022 raw and explicitly clipped one-second comparisons are present with the normalization and timing limitations stated.
  - No tolerance-sensitivity, HWA, Tiki-Taka, LoRA, or network-accuracy conclusion is drawn from this device-characterization study.
- **Coverage:** 4 declared run(s) completed; 0 failed attempt(s) retained.
- **Final interpretation:** The predeclared hypothesis has a mixed outcome. Across the two IBM AIHWKit presets, corruption settings, and SET/RESET start directions, adaptive pulse-count batching reduced held-out verify reads by 9.85% to 85.63% relative to one-pulse verify. Success-conditioned endpoint RMSE changed by only -0.43% to +0.05%, so batching did not materially degrade the accepted endpoint distribution. The savings were not free, however: adaptive batching increased mean target pulses by 53.70% to 210.51% and reduced held-out programming success by 0.96 to 5.31 percentage points. The claim that verify savings occur without a material failure penalty is therefore not supported by these settings. The target-dependent Gaussian surrogate was rejected in all 16 preset/corruption/controller/start conditions: absolute 90% or 95% coverage errors were 0.0452 to 0.0551 against the 0.03 gate, and median target-binned Wasserstein distance normalized by residual standard deviation was 0.1543 to 0.1731 against the 0.10 gate. Downstream deployment simulation should use the empirical endpoint kernel together with separate target-conditioned failure, saturation, corrupt-device, and pulse/verify-cost models. In the operational normalized Wan-2022 comparison, optimized-material IBM accepted-endpoint standard deviation was about 0.0136 versus Wan's 0.0424, whereas baseline-HfO2 was about 0.0668 to 0.0670; this is not a physical-equivalence claim. No HWA, Tiki-Taka, LoRA, or network-accuracy conclusion follows from this characterization.
- **Main limitations:** This is model-based evidence generated from AIHWKit 1.1.0 fitted IBM ReRAM presets, not replay of raw measured IBM programming trajectories. The shortened design retains all 41 targets, both programming directions, both controllers, identity-held-out validation, 1024 sampled identities, and four repeats, but evaluates only the half-step tolerance and therefore provides no tolerance-sensitivity result. It uses fixed normalized pulse amplitude, one preselected adaptive-controller setting, target-independent blocked boundary conditioning, and a 512-pulse programming budget; the study did not predeclare a numerical threshold for what constitutes a material success-rate penalty. Published corrupt identities are discrete modeled outcomes rather than new measured devices. The IBM coordinate has no unique mapping to Wan's 0-40 microSiemens range, IBM endpoint timing is unspecified while Wan is evaluated at one second, and Wan supplies no matched pulse trajectory, failure, or cost model. The four arms share construction and analysis seeds and are not independent fabricated arrays. This milestone does not exercise HWA training or post-deployment updates.
- **Next steps:**
  - Integrate the authoritative empirical endpoint kernel and its separate target-conditioned failure, corruption, saturation, and cost models into a deployment backend, preserving each sampled programmed endpoint for the later matched HWA-only versus on-chip-recovery comparison.
  - Run a small predeclared controller-tuning study over adaptive batch size and eta with an explicit maximum acceptable success-rate loss, while retaining the half-step primary tolerance and identity-held-out evaluation.
  - Repeat the comparison with raw IBM pulse trajectories and a defensible physical conductance/time mapping if those data become available; until then, keep the Wan-2022 comparison operational and normalized only.
- **Raw artifacts:** `results/ibm-reram-program-verify-noise-20260821-v2/`
- **Workflow summary:** `results/ibm-reram-program-verify-noise-20260821-v2/analysis/summary.json`
<!-- END EBL STUDY ibm-reram-program-verify-noise-20260821-v2 -->

<!-- BEGIN EBL STUDY mnist-relu-to-bounded-fp32-map-20260824-v1 -->
### mnist-relu-to-bounded-fp32-map-20260824-v1

**Direct ReLU compression into the bounded FP32 DRN**

- **Finished:** 2026-08-24
- **Evidence class:** `exploratory`
- **Outcome:** supported
- **Initial hypothesis:** A signed 784-50-10 ReLU checkpoint can be compressed directly into the continuous bounded [0.1020408197973068, 1.0] four-device dual-rail DRN with no more than one percentage point of clean test-accuracy loss before any device programming.
- **Completion criteria:**
  - The mapping arm performs exactly zero optimizer updates and saves one valid selected named checkpoint derived from the frozen teacher.
  - All 158,800 mapped values are finite FP32 conductances inside the declared bounds, with exact single-encoding halves/paired layouts and model-local amplification metadata.
  - A fresh completed evaluation covers all 10,000 MNIST test examples from the mapped checkpoint.
  - The mapped bounded DRN loses no more than 1.0 absolute percentage point of clean test accuracy relative to the frozen ReLU teacher; otherwise the direct-compression hypothesis is refuted and any later bounded FP32 distillation is declared as a separate intervention.
- **Coverage:** 2 declared run(s) completed; 0 failed attempt(s) retained.
- **Final interpretation:** The evidence supports the predeclared hypothesis. A zero-update signed lift of the frozen bias-free 784-50-10 ReLU checkpoint into the continuous float32 four-device dual-rail DRN, with every conductance constrained to [0.1020408197973068, 1.0], reached 97.38% accuracy on the fresh 10,000-example MNIST test split versus 97.36% for the ReLU teacher. Teacher agreement was 99.94% and teacher-to-student KL was 0.00013019. The selected nominal layer fractions were 0.25 and 0.125, with fixed positive output gain 56.234132519. The checkpoint contains 158,800 finite float32 conductances, all within bounds; the realized layer ranges were [0.1020408198, 0.3265306056] and [0.1020408198, 0.2142857164]. The observed +0.02 percentage-point difference is two examples and should be interpreted as effectively lossless compression, not a meaningful improvement over the teacher. This establishes the correct bounded-FP32 logical initialization control and supersedes the separately trained 93.58% bounded-DRN checkpoint as the starting state for future matched IBM OM HWA and deployment comparisons. It does not establish measured-device deployability.
- **Main limitations:** This is exploratory evidence from one frozen teacher checkpoint, one data split and seed, one bias-free perfect-diode DRN topology, and one predeclared 4x4 layer-fraction grid. The fraction pair and output gain were selected on a fixed 1,024-example validation calibration cohort, although the reported 10,000-example test evaluation was fresh. The large output gain compensates for small raw DRN scores and preserves clean argmax accuracy, but it does not prove adequate physical signal-to-noise margin. Conductances are ideal continuous float32 values: there is no measured-device assignment, per-cell common-window remapping, quantization, HWA perturbation, program-and-verify error, corrupt-device model, converter noise, IR drop, endurance model, or on-chip update. The result therefore isolates mapping fidelity only.
- **Next steps:**
  - Use the hash-pinned bounded-FP32 checkpoint as the identical logical starting point for matched clean, array-specific HWA, and IBM OM program-and-verify deployment arms.
  - Before HWA, quantify how per-synapse conservative common-window remapping changes clean accuracy, raw differential signal, passive loading, and required output gain.
  - Evaluate programmed apparent-endpoint accuracy and differential signal-to-write-noise ratio before deciding whether a matched on-chip recovery arm is required.
- **Raw artifacts:** `results/mnist-relu-to-bounded-fp32-map-20260824-v1/`
- **Workflow summary:** `results/mnist-relu-to-bounded-fp32-map-20260824-v1/analysis/summary.json`
<!-- END EBL STUDY mnist-relu-to-bounded-fp32-map-20260824-v1 -->

<!-- BEGIN EBL STUDY mnist-ibm-om-common-window-hwa-program-verify-pilot-20260824-v2 -->
### mnist-ibm-om-common-window-hwa-program-verify-pilot-20260824-v2

**MNIST DRN array-specific IBM OM common-window HWA pilot — corrected retry**

- **Finished:** 2026-08-24
- **Evidence class:** `model_based_aihwkit_preset`
- **Outcome:** supported
- **Initial hypothesis:** For one fixed counterfactually repaired IBM OM assignment whose quad common windows and apparent-forward network pass the predeclared gates, off-chip BPTT with the exact fixed bounds and compact endpoint model will reduce pulse-resolved apparent-forward deployment KL relative to matched clean BPTT deployed through the same assignment, mapper, inner-window margin, endpoint stream, and endpoint-application policy.
- **Completion criteria:**
  - Before either native run starts, the authoritative runtime mapper preflight must sample assignment seed 84001 under pinned AIHWKit 1.1.0 and the counterfactual_repaired policy, record the population artifact and receipt, and pass with zero corrupt cells, zero malformed quad groups, at most 20 empty quad windows and an empty-window fraction at most 0.0005, zero mapped targets outside a constituent cell bound for every nonempty quad, positive inner spans for every nonempty quad, and a finite mapped target tensor inside [0, 1]. Empty quads are explicit structural mapping exceptions: their out-of-bound cells may use only the separately declared exact fallback and must not be silently clipped, repaired, or reassigned. A hand-authored report or duplicated launcher-side grouping implementation is not eligible.
  - Before either native run starts, the exact endpoint-seed-84003 pulse-resolved selection modifier must program the full preflight population end to end, save its persistent deployment bundle, and pass without a controller exception or non-finite endpoint, with programming success at least 0.99 and budget-exhausted fraction at most 0.01.
  - The pulse canary and every native IBM OM report must declare endpoint_application_policy=aihwkit_apparent_forward_persistent_update_state. The AIHWKit apparent endpoint is the forward weight; the persistent endpoint is hidden physical update state retained for a possible later programming operation.
  - Inside the pulse canary's single already-programmed evaluation context, the production DRN evaluator must process exactly the first 100 ordered validation batches from the seed-17 5,000-example holdout: 1,600 examples at batch size 16, with no second programming pass. Apparent-forward student accuracy and teacher agreement must each be at least 0.25, and the metrics artifact must be retained before either native run starts.
  - The preflight report must bind the exact device-model SHA-256 3030e04d6205dc90d0894ac453d2c1c522dffdc004f69b9c6ab7eaf7ef4b8ba3, bounded-DRN checkpoint SHA-256 f0036b36cebe970c5105c22ebe703d53fec99b7716215cca0e55ee09cfaf70cf, both config hashes, assignment seed 84001, corruption policy counterfactual_repaired, target mapping dual_rail_quad_common_window, parameter layouts base.dense_weight.0=halves and base.dense_weight.1=paired, and common-window margin fraction 0.25.
  - Before either native run starts, the exact compact training modifier must execute one real training context on the same fixed population and mapped targets. Its report and deployment must preserve target_out_of_support=error, population-fingerprint parity, endpoint seed 84002, 158,792 fitted compact cells, and exactly eight pulse-resolved_noncorrupt_out_of_bound_empty_quad_only fallback cells matching the authoritative four-below plus four-above empty-quad counts. The fallback must report zero non-finite endpoints and finite pulse, verify, and reversal cost within the 128-pulse cap; no other cell may use the fallback.
  - The exact-bounds training modifier and both selection modifiers must resolve to the same population fingerprint. The exact-bounds arm uses compact endpoint seed 84002 for minibatches; both matched pulse-resolved selections use endpoint seed 84003, adaptive lower-from-RESET control, tolerance ratio 0.5, and a 128-pulse cap.
  - Both arms use literal named-weight copy from the exact bounded-DRN checkpoint, identical MNIST split, topology, solver, optimizer schedule, training seed 17, and ten epochs. The only training intervention is none versus the exact fixed-array compact endpoint modifier.
  - Every pulse-resolved selection reports structural common-window statistics, mapped-target reachability, programming success, SET/RESET/total pulses, verify reads, reversals, clipping, saturation, failure, apparent and persistent residuals, and saves the exact two-state deployment bundle linked to its selected checkpoint and population fingerprint.
  - This first corrected pilot makes no claim about published corruption, donor realism, reassignment, held-out-array generalization, or the need for on-chip training. No Tiki-Taka, LoRA, or soft SET/RESET recovery is run.
- **Coverage:** 2 declared run(s) completed; 0 failed attempt(s) retained.
- **Final interpretation:** The evidence narrowly supports the predeclared hypothesis on this one fixed counterfactually repaired IBM OM array: array-specific compact-endpoint HWA reduced selected pulse-resolved apparent-forward validation KL from 0.0547058969 for the matched clean control to 0.0542751399, an absolute reduction of 0.0004307570 (0.7874 percent relative). The HWA-selected checkpoint improved physical accuracy from 51.80 to 62.98 percent and teacher agreement from 53.20 to 64.60 percent. This improvement did not make the deployment adequate. HWA reduced clean logical accuracy from 93.58 to 86.64 percent, and a read-only decomposition of the selected saved deployment found 81.76 percent accuracy at the exact ideal common-window target but only 62.98 percent at the apparent programmed endpoint. Thus mapping cost 4.88 percentage points and programming error cost a further 18.78 points. W1 dominated the endpoint sensitivity: W1-only endpoint errors produced 70.54 percent accuracy versus 79.02 percent for W2-only errors; W1 mapped contrast error RMS was 5.29 times its target contrast RMS, and remained 3.05 times the signal even for quads whose four devices all met their individual acceptance criteria. The selected deployment nevertheless accepted 157657 of 158800 cells, showing that per-cell programming success is not a sufficient differential-precision or network-quality criterion. The result supports the narrow HWA-versus-clean KL hypothesis, but it neither establishes satisfactory deployability nor demonstrates that on-chip recovery is required.
- **Main limitations:** This is a mechanistic model-based AIHWKit-preset pilot using one bounded-DRN checkpoint, one MNIST split and training seed, one fixed assignment seed, one endpoint stream, and one counterfactually repaired OM population. Published-origin corrupt identities were replaced by sampled healthy donors, so the result does not estimate deployment on the published-corrupt array or validate donor realism. There are no independent array assignments, base-training seeds, confidence intervals, measured circuit effects, read-noise process, retention, endurance, ADC/DAC, IR-drop, temperature, or fabricated-hardware validation. Compact endpoint sampling was used during HWA while exact cap-128 pulse-resolved programming was used for selection. Four empty quad intersections required the predeclared exact eight-cell fallback. Checkpoint selection used teacher KL, whose absolute improvement was very small and whose near-collapsed-logit regime was poorly aligned with argmax accuracy. The starting 93.58-percent bounded checkpoint has since been superseded as a clean initialization control by the separately finalized 97.38-percent direct bounded-FP32 mapping study. The ideal-mapped and layerwise endpoint decompositions are read-only post-hoc analyses of the saved selected bundle rather than predeclared completion gates.
- **Next steps:**
  - Repeat the matched clean-versus-array-specific-HWA deployment from the finalized 97.38-percent bounded-FP32 initialization, with predeclared physical accuracy, teacher agreement, output-margin, and layerwise differential-SNR gates rather than KL alone.
  - On one frozen two-state deployment bundle, run matched HWA-only, soft SET/RESET fine-tuning, Tiki-Taka, and LoRA recovery controls so any claim that on-chip training is needed is based on a shared persistent/apparent starting state.
  - Repeat the leading protocol across fixed held-out OM assignments and then reintroduce published corrupt identities or an explicitly declared reassignment policy, reporting population-level uncertainty and programming cost.
- **Raw artifacts:** `results/mnist-ibm-om-common-window-hwa-program-verify-pilot-20260824-v2/`
- **Workflow summary:** `results/mnist-ibm-om-common-window-hwa-program-verify-pilot-20260824-v2/analysis/summary.json`
<!-- END EBL STUDY mnist-ibm-om-common-window-hwa-program-verify-pilot-20260824-v2 -->

<!-- BEGIN EBL STUDY mnist-ibm-om-cell-aware-exact-bounds-transfer-20260824-v1 -->
### mnist-ibm-om-cell-aware-exact-bounds-transfer-20260824-v1

**MNIST four-device IBM OM cell-aware exact-bounds quantized HWA transfer**

- **Finished:** 2026-08-25
- **Evidence class:** `model_based_aihwkit_preset`
- **Outcome:** supported
- **Initial hypothesis:** On an untouched repaired IBM OM array, nine-level quantization-aware HWA using exact per-cell affine targets will improve mean programmed test accuracy by at least two percentage points over the matched continuous cell-aware HWA control while retaining at least 90 percent of its own ideal quantized mapped-target accuracy.
- **Completion criteria:**
  - Before preparation or native execution, verify full-span checkpoint SHA-256 a99995a3e5b321a56bb7e00e29840c3b76f3fee95b8d8c80fdf0fa16e93b3563, derivation receipt SHA-256 d58b0d709312061dc2425d143744f45fdf9de9d5e583a619e7a91325c53059b8, frozen teacher SHA-256 9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52, device-model SHA-256 3030e04d6205dc90d0894ac453d2c1c522dffdc004f69b9c6ab7eaf7ef4b8ba3, and every prepared config hash.
  - The exact-bounds mapper must consume the fixed assigned cell identities and compute L_i=max(0,(logical_min_i+1)/2), H_i=min(1,(logical_max_i+1)/2). It must reconstruct u=(q++-q+--q-++q--)/2 and, in the quantized mode, use n=clamp(round_half_away_from_zero(4u),-4,4). Cell signs are (+,-,-,+), and each target must equal L_i+max(s_i*n,0)*(H_i-L_i)/4. The continuous control replaces n with 4u.
  - Targets must be quantized before compact endpoint sampling or exact programming. No endpoint is rounded afterward. No target may be clipped, silently narrowed, reassigned, or replaced. All target-support and programming failures remain part of the result.
  - Every development and held-out deployment must save an immutable exact-bounds codebook artifact containing population and binding provenance, exact and usable per-cell bounds, floors, spans, logical shadows, continuous coordinates, signed codes and indices when quantized, requested targets, target hashes, and baseline/sign-flip diagnostics. The artifact must state that hidden bounds were consumed by target generation.
  - Development uses the repaired assignment seed 84001, compact endpoint seed 84002, and pulse-resolved selection seed 84003. Training and selection must share the exact population fingerprint, binding order, halves/paired layouts, cell-aware mode, nine-level declaration, gain, and endpoint policy.
  - The zero-update arm executes no optimizer update. Each other arm completes exactly ten epochs. Checkpoints are selected by the highest mean apparent-forward student accuracy across exactly three programming repeats on assignment 84001; strict improvement preserves the earliest exact tie.
  - Learning rates are not selected in this study. They must equal the corresponding frozen RESET-relative schedules: clean BPTT [0.1388429752066116,0.00037685950413223146], continuous HWA [0.0017531914893617025,0.000004757787234042554], and quantized QAT [0.004411914893617021,0.000011974808510638298]. Device-forward gains are frozen from assignment 84001 at 3.5481338923357533 continuous and 2.818382931264453 quantized by the immutable cell-aware design receipt, and must be recorded separately from each checkpoint's FP32 shadow gain.
  - Continuous HWA and quantized QAT use a fresh compact programmed apparent endpoint in every minibatch forward, then restore the clean continuous FP32 shadow before the ideal off-chip optimizer step. Persistent physical state must not accumulate between minibatches.
  - Assignment seed 85001 must not be sampled, mapped, preflighted, or used for selection until all four development checkpoints and hashes are frozen. Each frozen checkpoint then receives exactly five full-test pulse-resolved deployments with endpoint seeds 85101 through 85105 and fresh exact bounds from assignment 85001.
  - Every held-out deployment covers all 10,000 test examples and retains apparent and persistent endpoints, acceptance/failure masks, pulse costs, population, exact codebook, source-checkpoint hash, and replayable RNG state. Unsupported or failed targets are reported rather than repaired after inspection.
  - The study completes only with one valid run for all four development arms and all twenty held-out deployments under artifact-hash verification. No on-chip fine-tuning, Tiki-Taka, LoRA, soft pulse update, extra endpoint seed, or held-out adaptation is eligible evidence.
- **Coverage:** 24 declared run(s) completed; 1 failed attempt(s) retained.
- **Final interpretation:** The predeclared hypothesis is supported. On untouched assignment 85001, nine-level cell-specific QAT reached 74.154% mean apparent programmed test accuracy, 6.552 percentage points above the matched continuous cell-aware HWA control at 67.602%; its 100.616% apparent-to-ideal retention also exceeded the 90% gate. At the same time, the selected QAT checkpoint fell from 93.3133% programmed validation accuracy on development assignment 84001 to 74.154% mean programmed test accuracy on the replacement array. The user's scientific conclusion is therefore that array-specific training helps substantially on the characterized training array, while the large transfer loss shows that much of the learned compensation is tied to that array and does not carry over unchanged. The held-out result is not a total collapse: QAT still outperformed continuous HWA on assignment 85001. Its mean ideal mapped accuracy there was 73.70%, close to the 74.154% apparent result, so programming noise alone does not explain the transfer loss; the cell-specific encoding and fixed-array adaptation are central. This exact-bounds mapper is an oracle characterization control, not a no-characterization deployment method.
- **Main limitations:** This is model-based AIHWKit-preset evidence from one frozen ReLU teacher and full-span DRN initialization, one 5,000-example development-validation split, one 10,000-example MNIST final-test split, one characterized development assignment (84001), and one independently assigned held-out replacement array (85001). The five held-out repeats vary programming endpoint seeds 85101-85105 on the same physical assignment; they are not five independent replacement arrays. Checkpoint selection used three programming repeats on the development array, and all trained arms used ten epochs with previously frozen schedules and fixed forward gains. Target generation deliberately consumed every cell's exact hidden minimum and maximum, so the method requires oracle per-cell characterization and storage. The counterfactually repaired IBM OM population, compact endpoint sampling during training, and pulse-resolved deployment remain simulations rather than fabricated-hardware evidence. The study did not compare multiple QAT parameterizations, train across a distribution of array assignments, complete the matched RESET-relative comparison, or include on-chip fine-tuning, Tiki-Taka, LoRA, retention, converter, IR-drop, endurance, or independent base-training-seed controls.
- **Next steps:**
  - Run a predeclared matched QAT-method study that holds the teacher, logical initializer, nine-level quantizer, optimizer budget, endpoint model, and checkpoint-selection rule fixed while comparing fixed-array exact-bounds QAT, QAT that resamples or cycles through multiple development-array assignments, and deployable RESET-relative shared-codebook QAT.
  - Evaluate every frozen QAT method on several entirely untouched array assignments, with multiple programming endpoint seeds nested within each array, so array-to-array generalization is measured separately from programming-repeat noise.
  - Include a logical-only nine-level QAT control without cell-specific bounds to isolate the benefit of quantization itself from compensation learned for a particular array, and report development accuracy, ideal mapped transfer, apparent programmed transfer, sign flips, and programming cost for every arm.
- **Raw artifacts:** `results/mnist-ibm-om-cell-aware-exact-bounds-transfer-20260824-v1/`
- **Workflow summary:** `results/mnist-ibm-om-cell-aware-exact-bounds-transfer-20260824-v1/analysis/summary.json`
<!-- END EBL STUDY mnist-ibm-om-cell-aware-exact-bounds-transfer-20260824-v1 -->

<!-- BEGIN EBL STUDY mnist-ibm-om-four-device-baseline-selection-20260828-v1 -->
### mnist-ibm-om-four-device-baseline-selection-20260828-v1

**Four-device IBM OM ideal baseline selection**

- **Finished:** 2026-08-28
- **Evidence class:** `model_based_aihwkit_preset`
- **Outcome:** mixed
- **Initial hypothesis:** A baseline shared within each destination-column pair, or enforced exactly by a partner intrinsic-reference cell, will make logical zero physically faithful and recover ideal bounded continuous initialization accuracy relative to independent per-cell RESET means, while a single quad-wide baseline will reveal the tradeoff between exact zero contrast and equal zero-state destination loading.
- **Completion criteria:**
  - All twelve declared native validate runs complete from the frozen ReLU source SHA-256 9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52, repeat development assignment 86001, and evaluate exactly one of held-out assignments 87001-87003 on all 10000 test examples.
  - Every production config resolves runtime.device=cuda, and the formal launcher records a successful CUDA availability and GPU-identity probe before creating any native task.
  - At each assignment seed, every policy uses the same jointly repaired four-cell hardware identities and the same commissioned RESET observations; semantic population, donor-selection, commissioning, source, topology, solver, and data fingerprints match across arms.
  - Every physical target is finite and nonnegative, every programmed active target is within its sampled bounds, every saved mapping satisfies G=B+d exactly, and full G enters both signed transfer and conductance-sum loading.
  - Shared-quad, shared-destination-column, and reference-enforced policies satisfy their declared exact zero-contrast gate within 1e-9 of the four-delta spacing; donor exhaustion, incomplete coverage, or a failed physical invariant invalidates the assignment rather than silently clipping it.
  - Continuous bounded mapping alone selects each policy's development scale pair and positive logit gain. The four-delta mapping reuses those frozen values without refitting and cannot select the preferred baseline.
  - Optimizer updates, BPTT, QAT, program-and-verify, target write noise, inference read noise, retention, drift, and post-deployment recovery are exactly absent.
  - The initialization gate passes only if the preferred policy reaches at least 90 percent mean held-out continuous accuracy. A unique winner is declared only if it has strictly more continuous correct predictions than every other arm on all three paired held-out assignments and leads the runner-up mean by at least 1.00 percentage point; otherwise the comparison is inconclusive.
- **Coverage:** 12 declared run(s) completed; 0 failed attempt(s) retained.
- **Final interpretation:** The evidence supports the shared-zero mechanism in the initial hypothesis: exact baseline cancellation is essential for useful ideal bounded initialization, with mean continuous accuracies of 51.58% for independent per-cell RESET means, 93.82% for one shared quad baseline, 94.97% for shared destination-column baselines, and 95.00% for reference-enforced destination columns. For follow-on deployment work, shared_destination_columns_reset_max is the practical baseline choice because it preserves exact logical zero, improves on a single quad-wide baseline, and retains 88.56% accuracy on the frozen standard four-delta diagnostic, whereas reference enforcement collapses to 12.38% on that diagnostic. The predeclared unique-winner comparison remains inconclusive: reference enforcement leads shared destination columns by only 0.0233 percentage point in mean continuous accuracy, does not dominate on all three held-out assignments, and does not meet the required 1.00-point lead. The study therefore resolves the baseline-choice stage operationally without claiming a statistically or predeclared uniquely superior continuous policy.
- **Main limitations:** This is model_based_aihwkit_preset evidence from one development assignment and three matched held-out hardware assignments, using a jointly repaired donor assignment. It evaluates ideal bounded initialization only: target-write noise, inference read noise, program-and-verify, BPTT, QAT, HWA, and post-deployment training are absent. The result therefore isolates baseline selection but does not establish programmed or replacement-hardware deployment accuracy.
- **Next steps:**
  - Freeze shared_destination_columns_reset_max as the baseline and run a separately predeclared ideal study of level size and level count while retaining the full physical conductance G=B+d in both signed transfer and conductance-sum loading.
- **Raw artifacts:** `results/mnist-ibm-om-four-device-baseline-selection-20260828-v1/`
- **Workflow summary:** `results/mnist-ibm-om-four-device-baseline-selection-20260828-v1/analysis/summary.json`
<!-- END EBL STUDY mnist-ibm-om-four-device-baseline-selection-20260828-v1 -->

#### Post-finalization conductance-range correction

This amendment does not alter the finalized run, review, or generated study
block above. It narrows their interpretation. The runtime intersected native
`x=(a+1)/2` support with a public `[0,1]` interval; 49.90% of held-out raw
upper coordinates exceed `1` and were censored, so the apparent
individual-cell `p90=1.000` headroom is not a device ceiling. The exact-zero
algebra and the choice of a four-cell- or destination-pair-shared baseline are
retained. Numerical headroom, baseline placement, spacing, and accuracy are
evidence only for the clipped mapping. The corrected successor below now uses
one globally frozen affine map from raw OM state to nonnegative conductance
without an upper clip at `x=1`; its independent result supersedes the clipped
mapping for the unrestricted-conductance question.

### mnist-ibm-om-shared-destination-baseline-spacing-pv-20260828-v1

**Shared-destination baseline position and spacing under raw-active P&V —
clipped-range direct exploratory CUDA screen**

- **Finished:** 2026-08-28
- **Evidence class:** `model_based_aihwkit_preset`
- **Lifecycle:** `exploratory_noncanonical`; this is deliberately not an EBL
  managed/finalized study block
- **Question:** Within the historical public `x=[0,1]` embedding and with exact
  destination-column zero sharing fixed, how do baseline position and uniform
  level spacing affect ideal quantization and stochastic persistent
  program-and-verify accuracy?
- **Setup:** Four-device, no-fixed-reference quads; baseline positions
  `alpha={0,0.25,0.5}`; spacings `h={1,2,4} delta_x`; held-out assignments
  87001-87003; five matched P&V seeds per assignment; 27 complete CUDA
  configurations and 135 complete persistent endpoints. Five failed or
  interrupted attempts are retained separately.
- **Baseline definition:** Within each destination pair `(G++,G-+)` or
  `(G+-,G--)`, `L` is the larger of the two cells' bounded means from eight
  RESET/read observations, `U` is the smaller sampled upper bound after
  intersection with `[0,1]`, and `B=L+alpha(U-L)`. Thus `alpha=0` assigns both
  cells the higher of their two commissioned RESET means; it is not a
  four-cell maximum.
- **Headline clipped-range result:** The best ideal quantized result was
  94.8767% for `alpha=0.25,h=delta_x`, only 0.1067 point above
  `alpha=0,h=delta_x`.
  Persistent P&V instead strongly selected `alpha=0,h=delta_x` at 78.1553%,
  versus 67.2020% for `alpha=0.25` and 45.0507% for the midpoint
  `alpha=0.5`. At `alpha=0`, persistent accuracy fell from 78.1553% to
  76.8060% and 73.3507% as spacing increased from one to two and four
  `delta_x`.
- **Interpretation:** Inside the clipped screen, one-shot initialization favored
  the lowest feasible shared destination baseline `B=L` and the finest tested
  spacing `h=delta_x`. Raising `B` created unused downward headroom while
  increasing full-conductance denominator loading, reducing upward headroom,
  increasing SET distance and pulse cost, and amplifying persistent write
  errors. The selected arm reached 93.994% apparent-endpoint accuracy but only
  78.1553% persistent accuracy. This is not a deployment selection for
  unrestricted positive conductance: removing the imposed upper ceiling
  changes `U`, baseline placement, capacity, loading, calibration, and endpoint
  handoff. The corrected successor below supersedes and rejects this entry's
  numerical unrestricted-deployment conclusion while preserving these exact
  historical clipped measurements.
- **Main limitations:** This direct run was not prepared, summarized, reviewed,
  or finalized through the managed study lifecycle. It uses the AIHWKit 1.1.0
  OM preset with counterfactually repaired identities rather than raw measured
  trajectories or fabricated arrays. The mapping first clipped 49.90% of raw
  sampled cell upper coordinates that exceeded `x=1`; 23.25% of individual
  headrooms consequently pile up at exactly `1`. Native P&V endpoints were
  also clipped at the declared public `x=[0,1]` circuit handoff; about 7.4% of
  persistent cells in the selected arm were projected. There is no QAT, BPTT,
  HWA, inference read noise, retention, drift, repeated logical rewrite, or
  on-chip recovery.
- **Successor:**
  `mnist-ibm-om-baseline-spacing-pv-no-clip-exploratory-20260828-v1`
  completed the corrected affine no-clipping matrix. It independently ranks
  the same nominal `B=L,h=delta_x` design first, but obtains only 54.7013%
  persistent accuracy rather than 78.1553%.
- **Next step:** First hold `alpha=0,h=delta_x` fixed and evaluate
  `[1.0,1.0]` layer scales on all three held-out assignments; only then
  investigate the controller/state estimator or spacing before any QAT, HWA,
  BPTT, or on-chip recovery arm.
- **Tracked report:**
  [`ibm_om_baseline_spacing_pv.md`](ibm_om_baseline_spacing_pv.md)
- **Local ignored artifacts:**
  `results/mnist-ibm-om-baseline-spacing-pv-exploratory-20260828-v1/analysis/`

### mnist-ibm-om-baseline-spacing-pv-no-clip-exploratory-20260828-v1

**Shared-destination baseline position and spacing under raw-native affine
P&V — no-clipping direct exploratory CUDA screen**

- **Finished:** 2026-08-28
- **Evidence class:** `model_based_aihwkit_preset`
- **Lifecycle:** `exploratory_noncanonical`; this direct CUDA result was not
  prepared, reviewed, or finalized through the workflow-managed lifecycle
- **Experiment ID:** `mnist_ibm_om_baseline_spacing_pv_no_clip.v1`
- **Configs:**
  `examples/mnist_relu_drn/ibm_om_baseline_spacing_pv_no_clip/`
- **Source:** revision
  `329c6912c944e00b2f1441af57bab5420cd6d1cd` with per-run dirty-state
  fingerprints retained in the native manifests; frozen ReLU source/teacher
  SHA-256
  `9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52`
- **Question:** With exact destination-column zero sharing retained, which
  baseline position and uniform spacing survive both ideal quantization and
  persistent P&V when raw native OM support is embedded into strictly positive
  conductance without any `[0,1]` clipping?
- **Affine embedding:** `x_raw=(a+1)/2`; exact frozen support floor/ceiling
  `-1.3759238719940186/1.8474750518798828`; study-wide origin
  `x_origin=-1.3759248719940185`; slope `s=0.00011`; and
  `G=s*(x_raw-x_origin)`. The strict mapped minimum is `1.1e-10` and configured
  ceiling is `0.0003545739916261292`. These are normalized circuit values, not
  an absolute-Siemens calibration.
- **Setup:** Four-device, no-fixed-reference quads; one baseline per
  destination column; `alpha={0,0.25,0.5}`;
  `h={1,2,4} delta_x`; development assignment 86001; held-out assignments
  87001-87003; five matched P&V seeds per held-out assignment; per-`alpha`
  development refit frozen across spacing, held-out hardware, and endpoint
  repeats.
- **Calibration caveat:** For `alpha=0`, development selected `[1.0,0.5]`
  over `[1.0,1.0]` by only 990 versus 989 correct examples out of 1,024. The
  selected scale halves W2 target contrast while the P&V contrast-error scale
  remains about `6.2 uS`.
- **Coverage:** All 27 CUDA configurations and 135 persistent endpoints
  completed. Two preserved earlier attempts failed before numerical execution
  because the CUDA device was inaccessible inside the sandbox; they are not
  configurations, results, or CPU runs.
- **Accuracy matrix:**

  | `alpha` | Spacing | Continuous | Ideal quantized | Persistent P&V |
  | ---: | ---: | ---: | ---: | ---: |
  | 0.00 | `1 delta_x` | 94.1133% | **93.6967%** | **54.7013%** |
  | 0.00 | `2 delta_x` | 94.1133% | 85.3967% | 52.3760% |
  | 0.00 | `4 delta_x` | 94.1133% | 68.6033% | 42.4633% |
  | 0.25 | `1 delta_x` | 93.9167% | 85.9433% | 27.6213% |
  | 0.25 | `2 delta_x` | 93.9167% | 55.5400% | 23.7733% |
  | 0.25 | `4 delta_x` | 93.9167% | 15.1933% | 13.0167% |
  | 0.50 | `1 delta_x` | 93.2100% | 87.4133% | 30.5787% |
  | 0.50 | `2 delta_x` | 93.2100% | 73.7767% | 26.5440% |
  | 0.50 | `4 delta_x` | 93.2100% | 25.3533% | 15.4440% |

- **Headline result:** Only `alpha=0,h=delta_x` passes the 90% ideal gate.
  Its ideal assignment range is 93.38--93.92%, while its three assignment-level
  persistent means are 55.978%, 57.436%, and 50.690%. It is the sole
  P&V-eligible design in this matrix, but its 54.7013% persistent mean is not a
  usable deployment result.
- **Validity:** Every saved ideal and persistent `G` is strictly positive; all
  targets and persistent endpoints remain in exact native support; projection
  count is zero; full conductance enters numerator and denominator; and
  apparent endpoints are controller diagnostics that are never deployed to the
  DRN.
- **Weight/programming diagnosis:** For the selected arm, ideal quantized
  DRN-versus-ReLU relative-L2 error is 0.3388/0.3325 in W1/W2 and persistent
  error is 0.6668/0.6162. Requested-code correctness is 49.9727%, persistent-
  window success is 35.3374%, apparent acceptance is 99.9627%, and mean cost is
  7.106 pulses per cell. The affine shift adds `151.35 uS` per cell and
  `605.41 uS` per quad in the configured embedding. Corrected W1/W2 baseline
  loads are `676.94/679.09 uS`, versus `83.68/84.95 uS` in the clipped screen;
  these are embedding micro-units, not an absolute device calibration. Raw P&V
  residual/code diagnostics barely change from the clipped run.
- **Interpretation:** The corrected screen independently selects the same
  nominal one-sided, finest-spacing design as the clipped predecessor, but
  rejects its 78.1553% unrestricted-deployment accuracy and immediate
  apparent-controller interpretation. The baseline cancels from signed zero
  while remaining fully in denominator loading; stochastic persistent error is
  large relative to the useful contrast, especially after the weakly selected
  `[1.0,0.5]` scale halves W2 signal. Only about half the persistent states
  resolve to the requested code, far below the 90% progression criterion, but
  the calibration confound must be removed before assigning the full gap to
  the controller.
- **Main limitations:** This is one frozen ReLU source, one development
  assignment, three held-out counterfactually repaired model assignments, and
  five endpoint seeds per assignment. All per-`alpha` gains hit the declared
  `1000` calibration-grid ceiling, and the selected alpha-zero layer scales
  led `[1.0,1.0]` by only one development example. There is no absolute conductance calibration,
  raw measured-device or fabricated-array evidence, fixed intrinsic reference,
  eight-device topology, inference read noise, retention, drift, logical
  rewrite, QAT, HWA, BPTT, replacement-hardware transfer after training, or
  on-chip recovery.
- **Next step:** At fixed `alpha=0,h=delta_x`, rerun `[1.0,1.0]` across all
  three held-out assignments with the same five endpoint seeds and corrected
  loading. If the persistent gap remains after this scale control, test a
  controller/state estimator or persistently distinguishable spacing. Training
  and recovery remain blocked until the persistent-code gate passes.
- **Tracked report:**
  [`ibm_om_baseline_spacing_pv.md`](ibm_om_baseline_spacing_pv.md)
- **Local ignored artifacts:**
  `results/mnist-ibm-om-baseline-spacing-pv-no-clip-exploratory-20260828-v1/`

### mnist-ibm-om-baseline-spacing-pv-winsorized-nominal-exploratory-20260828-v2

**Shared-destination baseline and spacing after nominal-bound Winsorization —
direct exploratory CUDA screen**

- **Finished:** 2026-08-28
- **Evidence class:**
  `model_based_aihwkit_preset_nominal_bound_winsorization_control`
- **Lifecycle:** `exploratory_noncanonical`; this direct CUDA result was not
  prepared, reviewed, or finalized through the workflow-managed lifecycle
- **Experiment ID:**
  `mnist_ibm_om_baseline_spacing_pv_truncated_nominal.v1`
- **Question:** Does the normalized workaround `a in [-1,1]`, `G=a+1`
  recover ideal and persistent deployment accuracy when the bound operation is
  applied before RESET commissioning and P&V rather than as an unphysical
  circuit-handoff clip?
- **IBM-model audit:** Default AIHWKit 1.1.0 does not perform this operation.
  `w_min=-1,w_max=1` are nominal abstract signed-weight bounds; Gaussian
  device-to-device variation samples cell bounds beyond them, and the C++
  sampler enforces sign/order without capping to the nominal means. The
  `SoftBoundsReferenceDevice` is a differential signed-weight abstraction and
  supplies no unique absolute single-device conductance origin.
- **Intervention:** Reuse each frozen jointly repaired identity, Winsorize its
  sampled hard pulse bounds to `a in [-1,1]`, repeat eight-read RESET
  commissioning on the modified population, build shared destination-column
  baselines, run the same one-pulse P&V, and deploy persistent `G=a+1=2x`
  directly. No identity is rejected/resampled and no persistent endpoint is
  clipped. About 75% of held-out cells have at least one changed bound and 25%
  have both changed.
- **Coverage:** 27/27 CUDA configurations completed: three baseline positions,
  three spacings, three held-out assignments, and five persistent P&V seeds per
  assignment, for 135 persistent deployments. Zero targets or persistent
  endpoints were projected at circuit handoff, and zero persistent endpoints
  left their transformed native support. Four of 238,200 unique destination
  pairs required a separately recorded pre-programming adjustment of their
  noisy RESET-max request into the exact pairwise common support.
- **Accuracy matrix:**

  | `alpha` | Spacing | Continuous | Ideal quantized | Persistent P&V |
  | ---: | ---: | ---: | ---: | ---: |
  | 0.00 | `1 delta_x` | 94.9800% | 94.5267% | 74.7913% |
  | 0.00 | `2 delta_x` | 94.9800% | 93.0933% | **75.7860%** |
  | 0.00 | `4 delta_x` | 94.9800% | 87.8867% | 71.7620% |
  | 0.25 | `1 delta_x` | 95.0667% | **94.9933%** | 67.9993% |
  | 0.25 | `2 delta_x` | 95.0667% | 88.7467% | 58.6040% |
  | 0.25 | `4 delta_x` | 95.0667% | 77.6333% | 54.9020% |
  | 0.50 | `1 delta_x` | 94.6900% | 22.7767% | 11.9080% |
  | 0.50 | `2 delta_x` | 94.6900% | 12.7867% | 10.3027% |
  | 0.50 | `4 delta_x` | 94.6900% | 13.1533% | 8.9993% |

- **Interpretation:** Winsorization removes the extreme common-mode translation
  of the global-affine stress test and restores high ideal mapped accuracy.
  It does not solve persistent deployment. The best persistent mean is 75.7860%
  at `alpha=0,h=2 delta_x`, with 71.26% requested-code correctness, 35.98%
  persistent-window success, 99.955% apparent acceptance, and persistent
  target RMSE 0.05523 raw `x`. Its +0.995-point mean lead over one-delta
  spacing is not robust: it wins 9/15 paired endpoints and the paired deltas
  range from -13.34 to +14.19 points. One-delta spacing preserves more ideal
  resolution but has only 49.63% code correctness; four-delta spacing improves
  the code label by erasing too much logical resolution. The midpoint rows are
  additionally confounded by a `[1.0,0.125]` continuous scale selection and a
  gain at the 1000 ceiling.
- **Claim boundary:** This is a large analyst-imposed normalized-preset
  sensitivity control, not default IBM behavior, an absolute conductance
  calibration, raw measured-device evidence, or fabricated-array evidence. It
  contains no inference read noise, retention, drift, QAT, HWA, BPTT,
  replacement-hardware transfer after training, or on-chip recovery.
- **Next step:** Supply a versioned physical HRS/LRS conductance calibration or
  measured distribution. Within the normalized control, improve the
  controller/state estimator at fixed targets until persistent requested-code
  correctness approaches the declared 90% progression gate; do not begin
  training/recovery solely from the current 75.7860% endpoint.
- **Tracked report:**
  [`ibm_om_baseline_spacing_pv.md`](ibm_om_baseline_spacing_pv.md)
- **Local ignored artifacts:**
  `results/mnist-ibm-om-baseline-spacing-pv-winsorized-nominal-exploratory-20260828-v2/`

<!-- BEGIN EBL STUDY mnist-ibm-om-four-reference-balance-ideal-init-20260828-v1 -->
### mnist-ibm-om-four-reference-balance-ideal-init-20260828-v1

**Four-device intrinsic-reference identity balance at ideal initialization**

- **Finished:** 2026-08-28
- **Evidence class:** `model_based_aihwkit_preset`
- **Outcome:** inconclusive
- **Initial hypothesis:** Weight-blind identity binding that minimizes the signed intrinsic-reference sum of each four-cell quad will strongly reduce zero-weight contrast and recover at least 90 percent held-out ideal bounded continuous initialization accuracy relative to the sampled identity order.
- **Completion criteria:**
  - Both declared native validate arms complete from the frozen pre-BPTT checkpoint and teacher, cover development assignment 86001 and held-out assignments 87001-87003, and evaluate all 10000 test examples per held-out binding/calibration arm.
  - The balanced binding is a deterministic, bijective, within-layer permutation of exactly the same sampled identities, uses no weights, labels, bounds, calibration outcomes, or test results, and its main/replay plan hashes match.
  - Every selected physical branch is finite and nonnegative, every active branch is within its sampled bounds, every mapping records B, d, and G with zero decomposition residual, and the full G enters both four-rail contrast and loading.
  - Optimizer updates, quantization, pulse programming, HWA, deployment write noise, inference read noise, retention, and drift are exactly absent.
  - The primary accuracy gate passes only if balanced@balanced held-out mean accuracy is at least 90 percent. Failure is retained and interpreted rather than repaired after inspection.
- **Coverage:** 2 declared run(s) completed; 0 failed attempt(s) retained.
- **Final interpretation:** The measurements support heterogeneous signed intrinsic-reference contrast as an important failure mechanism: weight-blind reference-balanced binding reduced zero-reference contrast RMS by 58 to 150 times and increased held-out scheme-optimized ideal bounded continuous accuracy from 84.49 percent to 91.83 percent, passing the declared 90 percent gate. With the random-binding calibration held fixed, the mean improvement was 7.68 percentage points. The result nevertheless remains inconclusive as a deployable initialization scheme because the intervention globally regrouped complete device identities within each layer. That nonlocal reassignment is not available when programming a fixed physical array. The study therefore validates the zero-bias mechanism and a nonlocal upper control, but it does not establish a practical baseline-selection method.
- **Main limitations:** This is model-based AIHWKit 1.1.0 OM-preset evidence from one frozen checkpoint and teacher, one MNIST split, one development assignment (86001), and three held-out assignments (87001-87003). The populations are counterfactually repaired rather than published-corrupt or fabricated arrays. Development selected each binding policy from 16 scale pairs and fitted a logit gain; the held-out cross-calibration matrix diagnoses this dependence but does not remove it. The primary metric is ideal bounded continuous initialization only. It excludes discrete levels, pulse programming and verify, write and read noise, retention, drift, BPTT, QAT, HWA, and on-chip recovery. Most importantly, the balanced policy moves whole device identities through a global within-layer permutation, changing each quad's identities, bounds, headroom, and loading together; it is not a local programming intervention on fixed hardware.
- **Next steps:**
  - Run a predeclared ideal bounded continuous study on the same fixed identity binding that compares the nearest intrinsic-symmetry baseline with the box-constrained minimum-L2 baseline adjustment satisfying exact four-cell zero contrast, records every conductance as G=B+d in both numerator and denominator, and uses identical matched d tensors plus a frozen cross-calibration matrix.
  - If the local continuous compensation succeeds, apply the same fixed binding to the standard four-delta level codebook without refitting the baseline objective, then test whether deterministic pulse programming can resolve the required baseline corrections before adding stochastic write noise, HWA, or on-chip recovery.
- **Raw artifacts:** `results/mnist-ibm-om-four-reference-balance-ideal-init-20260828-v1/`
- **Workflow summary:** `results/mnist-ibm-om-four-reference-balance-ideal-init-20260828-v1/analysis/summary.json`
<!-- END EBL STUDY mnist-ibm-om-four-reference-balance-ideal-init-20260828-v1 -->

<!-- BEGIN EBL STUDY mnist-ibm-om-local-reference-compensation-ideal-init-20260828-v1 -->
### mnist-ibm-om-local-reference-compensation-ideal-init-20260828-v1

**Fixed-identity local exact-zero reference compensation at ideal initialization**

- **Finished:** 2026-08-28
- **Evidence class:** `model_based_aihwkit_preset`
- **Outcome:** supported
- **Initial hypothesis:** With sampled identities fixed at their existing four-cell addresses, programming the nearest in-bounds local baseline whose signed four-rail contrast is exactly zero will improve held-out ideal continuous initialization by at least 5 percentage points under the frozen control calibration and recover at least 90 percent under per-scheme refitting.
- **Completion criteria:**
  - Both native commands receive data/ibm_om_cell_aware_full_span_v1.pt (SHA-256 a99995a3e5b321a56bb7e00e29840c3b76f3fee95b8d8c80fdf0fa16e93b3563) through --weights and data/mnist_relu_teacher_fixed_init_20260816.pt (SHA-256 9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52) through --teacher-weights; the initialization derivation receipt is data/ibm_om_cell_aware_full_span_v1.receipt.json (SHA-256 d58b0d709312061dc2425d143744f45fdf9de9d5e583a619e7a91325c53059b8).
  - Both declared native validate arms complete from the frozen pre-BPTT checkpoint and teacher, cover development assignment 86001 and held-out assignments 87001-87003, and evaluate all 10000 test examples per held-out policy/calibration cell.
  - The sampled device identities and their flat physical addresses are identical between baseline policies; no identity, intrinsic r, bound, or other device field is reassigned or mutated.
  - All local exact-zero box projections are feasible, remain inside sampled active bounds, and have absolute signed baseline-contrast residual at most 1e-12 in normalized double precision before canonical physical conversion.
  - For every assignment, layer, scale pair, and logical weight, the control and treatment d tensors are bitwise identical after canonical physical conversion and are limited by the common headroom available to both policies.
  - Every selected physical branch is finite and nonnegative, every active branch is within its sampled bounds, every mapping records rho, bounds, B, d, and G with zero decomposition residual, and the full G enters both four-rail contrast and denominator loading.
  - Optimizer updates, quantization, pulse programming, HWA, deployment write noise, inference read noise, retention, and drift are exactly absent.
  - The mechanistic accuracy gate passes only if the primary fixed-control-calibration held-out mean improvement is at least 5 percentage points. The absolute recovery gate passes only if the treatment scheme-refitted held-out mean accuracy is at least 90 percent. Any failed gate is retained and interpreted rather than repaired after inspection.
- **Coverage:** 2 declared run(s) completed; 0 failed attempt(s) retained.
- **Final interpretation:** The predeclared ideal-continuous hypothesis is supported. With every sampled identity fixed at its existing physical address and with bitwise-identical matched offset tensors, the minimum-L2 local baseline compensation improved held-out accuracy by 8.2267 percentage points under the frozen nearest-symmetry calibration, with paired gains of 6.50, 6.28, and 11.90 points. The compensated scheme-refitted mean was 91.8033 percent across assignments 87001-87003, with a range of 90.73 to 93.58 percent. It therefore passed both the 5-point mechanistic-effect gate and the 90-percent absolute-accuracy gate. This supports local signed zero-state baseline mismatch as an important cause of the nearest-symmetry initialization loss and establishes fixed-binding local compensation as a successful ideal continuous reference-anchored initialization candidate. It does not establish that the commonly sub-pulse corrections can be programmed, that they survive stochastic program-and-verify, or that reference anchoring improves on-chip learning.
- **Main limitations:** This is model_based_aihwkit_preset evidence from AIHWKit 1.1.0 rather than raw measured conductance traces or fabricated hardware. It uses one frozen checkpoint and teacher, one MNIST split, one development assignment (86001), and three held-out assignments (87001-87003) drawn from counterfactually repaired OM populations. Each policy searched the same 16 development scale pairs and fitted a positive logit gain. The experiment is bounded but continuous and excludes discrete levels, pulse reachability, P&V, write and read noise, retention, drift, HWA, QAT/BPTT, persistent updates, Tiki-Taka, LoRA, endurance, and absolute-Siemens calibration. The compensated baseline enforces circuit zero but need not coincide with either the active-range midpoint or the pulse-symmetry point. Many required baseline corrections are smaller than one nominal OM pulse increment, so continuous feasibility is not programmability evidence.
- **Next steps:**
  - Keep the same identities and local baseline objective and run a predeclared discrete-codebook and deterministic-then-stochastic program-and-verify study that measures whether the compensated baseline can be resolved persistently, preserves full G=B+d circuit accounting, and passes the persistent distinguishability and coverage gates.
  - Only after a realizable persistent deployment and an HWA-only recovery gap exist, run the matched two-by-two on-chip-learning study crossing shared-zero lower versus centered reference-anchored baseline placement with frozen versus direct persistent-pulse recovery. Start paired frozen/update arms from byte-identical deployed states, predeclare the difference-in-differences threshold, and report update bias, return-to-zero error, saturation, pulse count, loading drift, voltage, endurance, and accuracy.
  - Introduce Tiki-Taka or LoRA only as a separately predeclared matched recovery arm after the direct-pulse comparison; do not infer that either is necessary from the initialization result alone.
- **Raw artifacts:** `results/mnist-ibm-om-local-reference-compensation-ideal-init-20260828-v1/`
- **Workflow summary:** `results/mnist-ibm-om-local-reference-compensation-ideal-init-20260828-v1/analysis/summary.json`
<!-- END EBL STUDY mnist-ibm-om-local-reference-compensation-ideal-init-20260828-v1 -->

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
