# MNIST HWA and LoRA Experiment Setup

Last updated: 2026-07-29

## Purpose

This experiment asks two separate questions:

1. Does hardware-aware training (HWA) make a DRN more robust when its trained
   conductances are programmed into noisy ReRAM devices?
2. After deployment, can physical low-rank conductance branches recover some
   of the remaining accuracy without changing the deployed base
   conductances?

The baseline depends on which question is being answered:

- **HWA baseline:** an ordinarily trained FP32 DRN programmed into ReRAM.
- **LoRA baseline:** the HWA-trained DRN after it has been programmed into
  ReRAM.
- **Clean FP32 accuracy:** a reference for the accuracy available before
  device programming. It is not the direct LoRA baseline.

This distinction is important because LoRA never starts from the clean FP32
network. It starts from the already degraded HWA-to-ReRAM checkpoint.

## Four-arm comparison

The controlled comparison has four arms:

| Arm | Training and deployment path | Role |
| --- | --- | --- |
| a | Ordinary FP32 BPTT, evaluated before device programming | Clean reference |
| b | Ordinary FP32 BPTT, then programmed into ReRAM | HWA baseline |
| c | HWA BPTT, then programmed into the same ReRAM realizations | LoRA baseline |
| d | HWA BPTT, programmed into ReRAM, then adapted with physical LoRA | Recovered model |

Arms b and c use the same device model, conductance calibration, retention
time, and device seed pairs. The ordinary and HWA base trainings share their
initialization, data order, optimizer, solver, and epoch budget. Their
training-time hardware perturbation is the intended difference.

## Dataset and DRN

- Dataset: full torchvision MNIST
- Training examples: 60,000
- Test examples: 10,000
- Input normalization: mean `0.1307`, standard deviation `0.3`
- Batch size: 16
- Logical network dimensions: `[784, 100, 20]`
- Physical DRN dimensions: `[1568, 100, 20]`
- Nonlinearity: perfect diode
- Input gain: 50
- Numerical precision: FP32

Each of the 784 flattened input pixels is represented by positive and
negative input nodes, giving 1,568 physical input nodes. The 20 output nodes
form differential pairs for the 10 MNIST classes.

The DRN equilibrium is solved with:

- asynchronous coordinate updates;
- 4 iterations in training and inference;
- overrelaxation factor 1.1;
- BPTT through the unrolled DRN iterations.

In the strict experiment schema, BPTT is selected with:

```json
"algorithm": "backprop"
```

## Stage 1: ordinary FP32 training

The ordinary base DRN is trained for 20 epochs with:

- W1 learning rate: `0.08`
- W2 learning rate: `0.05`
- hidden-bias learning rate: `0.15`
- no training-time weight modifier

Its selected clean accuracy is:

```text
Ordinary FP32 DRN: 96.150%
```

This is arm a and serves as the clean reference.

## Stage 2: hardware-aware training

The HWA base uses the same model initialization, data seed, batches, solver,
learning rates, and 20-epoch budget as the ordinary base. During training,
an additive normal weight modifier is applied:

```text
standard deviation = 0.0025
modifier seed       = 101
```

The selected HWA checkpoint reaches 96.240% before ReRAM programming. The
small clean difference from ordinary training is not the main HWA result.
The meaningful HWA comparison is made after both checkpoints are programmed
into matched ReRAM realizations.

## Stage 3: ReRAM deployment

The selected ordinary and HWA checkpoints are programmed after training
using the AIHWKit `ReRamWan2022NoiseModel`.

- Device model: Wan-2022 ReRAM
- AIHWKit version: 1.1.0
- Retention time: 86,400 seconds, or one day
- Device maximum conductance: `40 uS`
- DRN conductance represented by `G_max`: `1.0` for W1 and W2
- Noise scale: `1.0`
- Original W1/W2 seed pairs:
  `17/29`, `31/37`, `41/43`, `47/53`, and `59/61`

The conductance reference is above the largest conductance in both selected
base checkpoints, so no target base conductance saturates at the ReRAM high
bound.

Device programming creates a new named-weights checkpoint containing the
realized ReRAM base conductances. Training does not resume during this
stage. Each programmed checkpoint is evaluated directly on the DRN.

## Stage 4: physical LoRA recovery

LoRA is added to both dense edges of the already programmed HWA DRN:

```text
input  -> A1 -> rank nodes -> B1 -> hidden
hidden -> A2 -> rank nodes -> B2 -> output
```

The effective network therefore contains three distinct conductance paths:

```text
W1_device in parallel with A1/B1
W2_device in parallel with A2/B2
```

`W1_device`, `W2_device`, `A1`, `B1`, `A2`, and `B2` are separate physical
conductance arrays. They are not combined and clipped as one effective
matrix.

For the primary rank-4 experiment:

- rank: 4 on both dense edges
- factor gain: `0.01`
- A1/A2 bounds: `[1e-7, 1.0]`
- B1/B2 bounds: `[0.0, 1.0]`
- A1/B1/A2/B2 learning rates: `0.01`
- recovery algorithm: BPTT
- recovery epochs: 20

B1 and B2 are initialized to zero. Consequently, inserting the LoRA
branches initially makes no contribution and exactly preserves the
corresponding deployed accuracy.

During recovery:

- the programmed W1 and W2 conductances remain frozen;
- the hidden bias remains frozen;
- only A1, B1, A2, and B2 train;
- each base or factor array is subject to its own conductance bounds.

The LoRA arrays are modeled as physical conductances in the DRN equations,
but their own programming noise, read noise, and retention drift are kept
ideal in this experiment. The Wan-2022 device effects apply to the deployed
base arrays.

## Primary four-arm results

| W1/W2 seeds | FP32 -> ReRAM | HWA -> ReRAM | HWA -> ReRAM -> rank-4 LoRA |
| --- | ---: | ---: | ---: |
| 17 / 29 | 90.46% | 93.51% | 94.71% |
| 31 / 37 | 93.43% | 93.30% | 94.93% |
| 41 / 43 | 91.95% | 92.01% | 95.09% |
| 47 / 53 | 92.57% | 92.53% | 94.44% |
| 59 / 61 | 93.80% | 94.46% | 95.23% |
| **Mean** | **92.442%** | **93.162%** | **94.880%** |
| **Sample SD** | **1.324 pp** | **0.942 pp** | **0.313 pp** |

The resulting comparison is:

| Measurement | Accuracy | Change |
| --- | ---: | ---: |
| Clean ordinary FP32 reference | 96.150% | — |
| Ordinary FP32 -> ReRAM | 92.442% | -3.708 pp from clean FP32 |
| HWA -> ReRAM | 93.162% | +0.720 pp over FP32 -> ReRAM |
| HWA -> ReRAM -> rank-4 LoRA | 94.880% | +1.718 pp over HWA -> ReRAM |

Therefore:

- **HWA improvement:** `+0.720` percentage points relative to the matched
  FP32-to-ReRAM deployment baseline.
- **LoRA improvement:** `+1.718` percentage points relative to the matched
  HWA-to-ReRAM deployment baseline.
- **Combined HWA and LoRA improvement:** `+2.438` percentage points relative
  to the ordinary FP32-to-ReRAM mean.
- **Remaining gap from clean ordinary FP32:** `1.270` percentage points.
- **LoRA recovery of the clean-HWA deployment gap:** `55.8%`.

The clean-HWA recovery calculation is:

```text
(94.880 - 93.162) / (96.240 - 93.162) = 55.8%
```

HWA and LoRA improve different stages of the workflow. The `+0.720` and
`+1.718` point improvements should not both be measured from clean FP32:
HWA is evaluated against arm b, while LoRA is evaluated against arm c.

## Expanded rank-4 device replication

Rank-4 LoRA was repeated on five additional Wan-2022 device seed pairs:

```text
67/71, 73/79, 83/89, 97/101, 103/107
```

The additional realizations were retained without filtering, including a
hard 87.79% deployment.

Across all 10 HWA-to-ReRAM realizations:

| Measurement | Mean accuracy | Sample SD |
| --- | ---: | ---: |
| HWA -> ReRAM baseline | 92.408% | 2.045 pp |
| HWA -> ReRAM -> rank-4 LoRA | 94.639% | 0.463 pp |

- Mean paired LoRA improvement: `+2.231 pp`
- 95% t-interval for the mean paired gain: `[+1.011, +3.451] pp`
- Improved realizations: 10 out of 10
- Mean clean-HWA deployment gap recovered: 58.2%
- Correlation between deployed accuracy and gain: `r = -0.986`

The negative correlation means that the more damaged deployments generally
received the largest recovery. It is exploratory and is also influenced by
the clean-accuracy ceiling.

The 10-device result is not a new measurement of the HWA improvement because
the ordinary FP32 checkpoint was not programmed into the five additional
device pairs. It expands only the HWA-to-ReRAM versus LoRA comparison.

## Rank expansion

Ranks 1, 2, 4, and 8 were compared on the original five HWA-to-ReRAM device
pairs with all other settings fixed.

| Rank | Added factor conductances | Mean accuracy | Gain over HWA -> ReRAM |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 93.162% | 0.000 pp |
| 1 | 1,788 | 94.518% | +1.356 pp |
| 2 | 3,576 | 94.732% | +1.570 pp |
| 4 | 7,152 | 94.880% | +1.718 pp |
| 8 | 14,304 | 94.984% | +1.822 pp |

The additional factor count across both branches is:

```text
rank * (1568 + 100 + 100 + 20) = rank * 1788
```

The base dense edges contain 158,800 conductances. Ranks 1, 2, 4, and 8 add
approximately 1.13%, 2.25%, 4.50%, and 9.01% additional conductances.

Most of the recovery is already available at rank 2. Rank 8 doubles the
rank-4 factor conductance count but improves mean accuracy by only
`0.104 pp`. With five matched device pairs, the paired 95% intervals for
each incremental rank difference include zero.

## Full-model fine-tuning control

A fifth arm tests whether the deployed DRN contains additional recoverable
accuracy when the frozen-base constraint is removed:

```text
HWA -> ReRAM -> direct W1/W2/bias BPTT fine-tuning
```

It starts from the same five original HWA-to-ReRAM checkpoints, uses no
adapter or weight modifier, and trains for 20 epochs with the original base
learning rates:

```text
W1 = 0.08
W2 = 0.05
bias = 0.15
```

The dense arrays are independently bounded by `[1e-7, 1.1]`. The upper
bound is slightly above the nominal 1.0 device reference because one
AIHWKit-realized starting value is `1.05865`; using 1.0 would silently alter
that deployment before the comparison.

| W1/W2 seeds | HWA -> ReRAM | Rank-4 LoRA | Full-model fine-tuning |
| --- | ---: | ---: | ---: |
| 17 / 29 | 93.51% | 94.71% | 96.93% |
| 31 / 37 | 93.30% | 94.93% | 96.59% |
| 41 / 43 | 92.01% | 95.09% | 96.80% |
| 47 / 53 | 92.53% | 94.44% | 96.87% |
| 59 / 61 | 94.46% | 95.23% | 96.75% |
| **Mean** | **93.162%** | **94.880%** | **96.788%** |
| **Sample SD** | **0.942 pp** | **0.313 pp** | **0.130 pp** |

Full-model fine-tuning gains `+3.626 pp` over deployment and `+1.908 pp`
over rank-4 LoRA. It exceeds the clean HWA checkpoint by `+0.548 pp`, so it
is continuing classifier optimization rather than only reversing device
error.

That additional accuracy has a much larger rewrite footprint. On average,
85.81% of W1, 99.42% of W2, and 85.40% of the hidden bias change in the
cost-selected checkpoints: approximately 136,630 of 158,900 existing
values. Rank-4 LoRA adds 7,152 conductances but leaves all 158,900 existing
values bit-identical.

The full-model arm is an ideal upper control. Its continuous tensor updates
do not model physical ReRAM pulses, update asymmetry, write noise, endurance,
energy, or latency. It establishes available adaptation capacity, not yet
the feasibility of in-device fine-tuning.

## Physical and numerical validation

Across the original rank-4 runs and the 20-run expansion, 25 completed LoRA
trajectories were audited:

- every zero-output-factor initialization reproduced its deployed baseline;
- every run completed 20 epochs;
- the cost-selected checkpoint was epoch 20 for every run;
- W1, W2, and the hidden bias were bit-identical before and after recovery;
- A1 and A2 stayed within `[1e-7, 1.0]`;
- B1 and B2 stayed within `[0.0, 1.0]`;
- all checkpoints and validation metrics were finite.

Repository validation after the expansion:

```text
Focused tests: 62 passed, 1 skipped
Legacy labs:   42 passed, 1 skipped
```

## Artifact flow

The experiment artifacts are under:

```text
labs/cases/mnist_perfect_diode_hwa_lora_comparison/
```

The main flow is:

```text
fp32_base_run/.../checkpoints/weights.pt
    -> deployed/fp32/seeds_*/weights.pt
    -> deployed_validation/fp32/seeds_*/*

hwa_base_run/.../checkpoints/weights.pt
    -> deployed/hwa/seeds_*/weights.pt
    -> deployed_validation/hwa/seeds_*/*
    -> lora_initial/seeds_*/weights.pt
    -> lora_initial_validation/seeds_*/*
    -> lora_run/seeds_*/*/checkpoints/weights.pt
```

Expansion artifacts are grouped under:

```text
expansion/deployed/
expansion/deployed_validation/
expansion/rank1/
expansion/rank2/
expansion/rank4_new/
expansion/rank8/
```

The full-model control follows the new raw-result convention:

```text
results/mnist-hwa-reram-full-finetune/
  runs/baseline-validation/
  runs/full-finetune/
  analysis/summary.json
```

The selected named-weights artifact for each completed training run is:

```text
checkpoints/weights.pt
```

## Configuration files

- Ordinary clean base:
  `examples/small_drn/mnist_perfect_diode.json`
- HWA base:
  `labs/cases/mnist_perfect_diode_hwa_lora_comparison/hwa_base.json`
- Physical LoRA recovery:
  `labs/cases/mnist_perfect_diode_hwa_lora_comparison/lora_hwa_deployed.json`
- Full-model post-deployment fine-tuning:
  `examples/small_drn/mnist_hwa_reram_full_finetune.json`
- Device-programming records:
  `labs/cases/mnist_perfect_diode_hwa_lora_comparison/deployed/deployment_manifest.json`

Training and validation use the stable public CLI:

```bash
python -m ebl train \
  --config <config.json> \
  --output-dir <new-run-directory>

python -m ebl validate \
  --config <config.json> \
  --output-dir <new-validation-directory> \
  --weights <explicit-checkpoints/weights.pt>
```

Each command owns a new run directory. Checkpoints are always supplied by an
explicit path; the workflow does not scan for the newest model.

## Interpretation

The experiment supports the following claims:

1. HWA improves robustness to the tested Wan-2022 ReRAM deployment compared
   with ordinary FP32 training.
2. Physical low-rank conductance branches can recover additional accuracy
   while the deployed base conductances remain frozen.
3. Recovery is consistent across the 10 tested rank-4 device realizations.
4. Increasing rank gives diminishing returns; rank 2 captures most of the
   observed recovery, and rank 4 is a reasonable area/accuracy compromise.
5. LoRA also reduces the observed variance across device realizations.
6. Direct full-model fine-tuning provides a substantially higher ideal
   accuracy ceiling, but requires rewriting most existing conductances.

It does not yet establish that:

- the result generalizes across multiple independently trained base models;
- the reported accuracy generalizes to an untouched final test split;
- LoRA remains effective when its own A/B arrays receive device programming
  noise and retention drift;
- the additive Gaussian HWA modifier is the best match for Wan-2022 ReRAM;
- incremental rank differences are statistically distinct.
- the ideal full-model advantage survives a physical pulse-aware ReRAM
  update process.

## Limitations

- Only one ordinary/HWA base-training seed was used.
- The canonical MNIST test split was used for checkpoint selection and final
  reporting.
- The original four-arm and rank comparisons use five device pairs; only
  rank-4 LoRA recovery was expanded to 10.
- HWA uses additive Gaussian perturbations rather than the exact Wan-2022
  forward model during training.
- ReRAM effects are applied to the base DRN, while LoRA A/B device effects
  remain ideal.
- The conductance reference was calibrated from the selected checkpoints
  rather than preregistered.
- The current nonnegative passive LoRA branch can add conductance but cannot
  directly subtract excess base conductance.
- Full-model fine-tuning uses ideal continuous updates and omits physical
  write noise, asymmetry, endurance, energy, and latency.

## Related records

- [Current research state](current_state.md)
- [Experimental manifest](experimental_manifest.md)
- [Physical layerwise low-rank recovery](passive_layerwise_low_rank_recovery.md)
