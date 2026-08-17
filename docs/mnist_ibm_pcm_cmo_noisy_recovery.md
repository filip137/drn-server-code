# MNIST IBM PCM/CMO noisy-recovery study

Last updated: 2026-07-30

## Question

Starting from the existing FP32 MNIST DRN, compare two measured IBM endpoint
device models:

1. IBM Analog Foundation Models PCM;
2. IBM CMO/HfOx ReRAM.

For each device, measure FP32 deployment, HWA deployment, full BPTT recovery,
and rank-4 physical LoRA recovery. Unlike the earlier Wan study, every
trainable device array is reprogrammed noisily after every recovery update.

The original nine runs documented below used `normalized_offset` for CMO,
which subtracts its `9 µS` floor after applying device noise. Two
floor-retaining mappings are now separate: `literal_conductance` directly
scales and hard-clips sub-floor targets, while `affine_floor` spans the full
physical window without merging them. The follow-ups are reported separately
so the original result remains reproducible.

## Controlled sequence

| Stage | Starting point | Trainable arrays | Device write |
| --- | --- | --- | --- |
| FP32 reference | existing selected FP32 checkpoint | none | none |
| FP32 deployment | FP32 reference | none | one endpoint write |
| HWA | FP32 reference | W1, W2, bias | temporary 3% channel-wise noise during BPTT; clean digital update |
| HWA deployment | selected HWA checkpoint | none | one endpoint write |
| Full recovery | selected HWA checkpoint | W1 and W2; bias frozen | endpoint write after every BPTT update |
| LoRA recovery | selected HWA checkpoint | A1, B1, A2, B2; base and bias frozen | base written once; every factor rewritten after every BPTT update |

The HWA checkpoint is shared between the two devices. Device seed `17` and
the same per-array RNG order are used for matched full and LoRA deployments.
This is one device realization, not a replicated estimate.

## HWA procedure

HWA starts from the selected 96.15% FP32 checkpoint and runs for two epochs:

- W1/W2 learning rates: `0.008`, `0.005`;
- hidden-bias learning rate: `0.015`;
- Gaussian modifier standard deviation: `0.03`;
- scale: independent absolute maximum of every output channel;
- modifier seed: `101`;
- noisy validation: disabled.

For each minibatch, the modifier samples a temporary noisy W1/W2 realization,
runs the forward equilibrium and BPTT gradient computation, restores the clean
master tensors, and applies the optimizer update. This matches the
`ADD_NORMAL_PER_CHANNEL` and `std_dev=0.03` choices in IBM's released Analog
Foundation Models configuration. The DRN study does not copy that LLM's
Gaussian clipping, input/output quantization, optimizer, distillation, or
architecture. It also applies the temporary perturbation to nonnegative
conductances and clamps it to the DRN bounds; IBM applies the modifier to
signed neural-network weights.

## Device models

### IBM Analog Foundation Models PCM

The implementation is an equation-level transcription of IBM's released
`add_PCM_noise` helper:

- nominal mapped maximum: `180` in the helper's unitless fit coordinate;
- exact-zero threshold: `0.6` in the same coordinate;
- low/high polynomial switch: `52.5` in the same coordinate;
- maximum input tile: 512;
- each output channel/tile has its own absolute-maximum normalization;
- polynomial standard deviations retain the helper's FP16 conversion.

This is an endpoint programming-error model, not a pulse model. It has no
retention law or separate read-noise term. Because every channel/tile is
renormalized, it does not define an absolute DRN-to-conductance calibration.
The released signed-weight helper can also produce Gaussian tails below zero or
above 180 in fit units; this positive-conductance DRN clamps the mapped-back
tensor to its configured logical bounds.

The provenance matters: Figures 7a and 9 in the AFM paper show the PCM
signal-to-noise/programming-error relationship on normalized axes. They do
not provide a physical min/max-conductance figure. IBM's released helper names
the constants `GMAX=180` and `OD_TD_DIVERGENCE=52.5`, but assigns no physical
unit to them. The underlying Hermes-chip paper reports a nominal per-device
resistance range of `100 kΩ–10 MΩ`, equivalent to `0.1–10 µS`, while its
weight mapping uses unit-cell maxima of 80 or 160 **ADC counts**. Those
physical/device and ADC-count ranges cannot be identified one-to-one with the
AFM helper's 180 coordinate.

### CMO/HfOx ReRAM

The implementation follows AIHWKit 1.1's `ReRamCMONoiseModel` coefficients:

- measured window: `9` to `88.199997 µS`;
- program-and-verify acceptance fit: `0.2%`;
- programming-noise scale: `1.0`;
- inference age: 1 second;
- relaxation and read-noise scales: `1.0`;
- archived logical calibration: `[0, 1]` maps to
  `[9, 88.199997] µS`, followed by normalized-offset conversion.

Unlike the AFM PCM case, Falcone et al.'s Figure 3 and Supplementary
Figure S5 directly show 35 programmed levels over approximately
`10–90 µS`; the AIHWKit fit uses the slightly more specific
`9–88.199997 µS` endpoints.
Falcone et al. report about 89 set/reset pulses on average to converge at the
selected `0.2%` acceptance range. This experiment samples the fitted residual
endpoint error; it does not replay those 89 pulses or their trajectory.

The last mapping is explicit and nontrivial. It allows a logical zero and a
zero-initialized LoRA output factor only by subtracting the physical `9 µS`
floor on conversion back to the DRN. A literal passive circuit would require
a reference/differential cancellation path; without one, the CMO floor would
be an actual nonzero connection.

This arm uses all three stochastic terms exposed by AIHWKit's CMO class.
Falcone et al.'s own reported MVM simulation omitted read noise while adding
6-bit/8-bit input/output quantization and wire IR drop. This DRN experiment
does the converse for those terms: it enables the class's read-noise law but
does not model converter quantization or IR drop.

## Noisy recovery update

BPTT always computes its gradient at the current realized device tensors. The
optimizer then:

1. restores a clean digital target tensor;
2. applies the gradient update to that target;
3. clamps the target to its configured logical conductance bounds;
4. samples a fresh program-and-verify endpoint realization;
5. installs that realization for the next minibatch.

Full recovery uses 20 epochs, W1/W2 learning rates `0.008/0.005`, and a frozen
hidden bias. Rank-4 LoRA uses 20 epochs and learning rate `0.01` for each of
A1, B1, A2, and B2. The LoRA base W1/W2 realization is frozen.

This serial closed-loop interpretation deliberately satisfies the requirement
that every post-HWA BPTT update include write noise. It does not claim
constant-time in-situ training: it rewrites a complete trainable array after
every minibatch and omits pulse count, latency, energy, and endurance.
With 60,000 examples, batch size 16, and 20 epochs, each recovery arm performs
75,000 optimizer/write steps. That corresponds to 11.91 billion individual
W1/W2 endpoint programs for full recovery (`158,800` values per step), versus
536.4 million factor endpoint programs for rank-4 LoRA (`7,152` values per
step), before counting verification pulses. These are modeled write events,
not a hardware time or energy estimate.

## Results

| Measurement | Accuracy |
| --- | ---: |
| Existing FP32 reference | 96.15% |
| Shared two-epoch HWA checkpoint | 96.37% |
| FP32 → PCM | 96.15% |
| HWA → PCM | 96.35% |
| FP32 → CMO | 96.21% |
| HWA → CMO | 96.38% |
| HWA → PCM → full noisy BPTT | 96.43% |
| HWA → PCM → noisy rank-4 LoRA | 96.30% |
| HWA → CMO → full noisy BPTT | 96.43% |
| HWA → CMO → noisy rank-4 LoRA | 96.40% |

These are selected-checkpoint accuracies for FP32/HWA and pre-update
`initial_validation` accuracies for device deployment. A one-seed
`+0.20 pp` PCM or `+0.17 pp` CMO HWA difference is descriptive only.
Recovery checkpoints are selected by cost on one persistent device state at
each epoch; they are not selected from a multi-realization noise average.
Relative to matched HWA deployment, the selected recovery changes are:

| Device | Full noisy BPTT | Noisy rank-4 LoRA |
| --- | ---: | ---: |
| PCM | +0.08 pp (epoch 7) | -0.05 pp (epoch 20) |
| CMO | +0.05 pp (epoch 17) | +0.02 pp (epoch 20) |

The endpoint writes create essentially no deployment loss at this seed:
PCM leaves the FP32 accuracy unchanged and lowers the clean HWA accuracy by
only `0.02 pp`, while CMO is slightly above both clean checkpoints. The HWA
deployment advantage therefore mostly follows the extra HWA-stage training,
not demonstrated robustness to a large device perturbation. The two recovery
arms should be interpreted primarily as tests of optimization with repeated
noisy writes, not as a high-headroom recovery benchmark.
For the HWA checkpoint, the first write has an RMSE of `0.00517` in logical
DRN units for PCM and `0.00477` for CMO, against a logical reference range
of `[0, 1]`.

The four 75,000-step trajectories remained finite and stable. In both LoRA
arms, reconstructed CUDA programming confirms that W1 and W2 in the selected
checkpoint are bit-exact copies of the initial programmed base
(`max_abs=0`); all four factor arrays are finite and within their configured
bounds. Both full-recovery hidden biases remain bit-exact to the HWA
checkpoint.

The single-seed differences are too small to establish a full-versus-LoRA
winner. The defensible result is narrower: repeated endpoint noise did not
destabilize either optimization path, while neither device supplied enough
deployment damage to measure substantial recovery.

## Literal-floor CMO follow-up

The follow-up changes only the CMO mapping and reuses the exact FP32 and HWA
checkpoints, device seed, noise scales, training hyperparameters, and
cost-selection protocol. It retains the physical floor:

```text
G_target_uS = clamp(88.199997 * G_drn, 9, 88.199997)
G_drn_realized = G_realized_uS / 88.199997
```

The DRN therefore sees a minimum connection of
`9/88.199997 = 0.1020408`; the `9 µS` offset is not subtracted.

| Measurement | Accuracy | Matched change |
| --- | ---: | ---: |
| Existing FP32 reference | 96.15% | — |
| Shared clean HWA checkpoint | 96.37% | — |
| FP32 → literal-floor CMO | 79.51% | -16.64 pp vs FP32 |
| HWA → literal-floor CMO | 80.61% | -15.76 pp vs HWA |
| HWA → CMO → full noisy BPTT | 89.73% | +9.12 pp vs deployed HWA |
| HWA → CMO → noisy rank-4 LoRA | 80.56% | -0.05 pp vs base-only deployment |

The HWA write clips `153,862 / 156,800` W1 targets (`98.13%`) and
`1,762 / 2,000` W2 targets (`88.10%`) to the floor. Its programming RMSE is
`0.08966` in DRN units, compared with `0.00477` under the earlier
normalized-offset mapping. The gap is therefore dominated by range mismatch,
not the stochastic residual alone.

Full BPTT reaches a transient peak of `89.74%` at epoch 19 and the
cost-selected `89.73%` at epoch 20. It recovers `57.87%` of the clean-HWA
gap but remains `6.64 pp` below clean HWA. Its last write still clips
`94.89%` of W1 and `91.15%` of W2 targets; recovery comes from strengthening
a small subset of useful connections, including values at the `88.2 µS`
ceiling.

The LoRA run initializes at `80.55%` because the nominally zero factor arrays
also retain the floor. Its cost-selected epoch-17 result is `80.56%`, only
`+0.01 pp` relative to its own initialization. At the last write, every clean
target in A1, B1, A2, and B2 remains below the physical threshold, so all
factor targets program to `9 µS` and differ only through sampled endpoint
noise. The positive-only branch has no subtractive path for cancelling the
excess base conductance.

All native statuses and eight checkpoint hashes verify. Full and LoRA biases
remain bit-exact to HWA, and a CUDA reconstruction confirms that both frozen
LoRA base arrays are bit-exact to their initial programmed realization. The
raw follow-up, exact summary, and staged source bundle are local ignored
artifacts under `results/mnist-cmo-literal-floor-noisy-recovery/`.
The source bundle SHA-256 is
`a6d96d837d21f4347dd6843de4d3dba3ba57751ef9a5dc106a38e741dedc4d43`.

This is still a post-hoc deployment calibration. The DRN was not trained from
initialization with the `9 µS` floor, and the endpoint model does not include
individual verification pulses, endurance, energy, latency, converter
quantization, or wire IR drop. The result establishes that literal absolute
scaling is a severe clipping/range-mismatch test. It does not by itself show
that every floor-retaining calibration must produce the same gap.

## Affine floor-mechanism follow-up

The next controlled probe uses the same clean HWA checkpoint, device seed,
physical range, acceptance fit, retention time, and stochastic terms, but
programs

```text
G_target_uS = 9 + (88.199997 - 9) * G_drn
G_drn_realized = G_realized_uS / 88.199997
```

The DRN still sees the physical floor:

```text
G_drn_nominal = 0.1020408 + 0.8979592 * G_drn
```

Unlike literal mapping, this affine transformation preserves the ordering and
uniformly scales the differences between all clean physical targets. Neither
W1 nor W2 has a low/high target clip. Endpoint noise and the final physical
clamp still place `18.32%` of stored W1 and `11.80%` of stored W2 exactly at
the floor.

| Measurement | Accuracy |
| --- | ---: |
| Shared clean HWA checkpoint | 96.37% |
| Deterministic literal hard floor | 81.83% |
| Noisy literal first write / stored realization | 80.61% / 80.34% |
| Deterministic affine retained floor | 95.94% |
| Noisy affine first write / stored realization | 96.12% / 96.11% |

The finite floor still changes the DRN normalization and strongly attenuates
voltages. Stored affine hidden/output RMS is `0.1922/0.02469`, compared with
`0.8761/0.2188` clean. Nevertheless, preserving the small learned values
removes almost the entire accuracy loss in this perfect-diode checkpoint.
The literal collapse was therefore dominated by many-to-one clipping rather
than floor loading alone.

A uniform floor changes both the voltage-weighted numerator and the incident
conductance sum in each coordinate update. Ordinary MVM reference subtraction
addresses only the first. The deterministic oracle reaches `96.37%` only when
both terms are corrected and the bias is scaled consistently; this is an
equation-level active bound, not a circuit result. Full derivation, the
on/off-ratio sweep, selector ablation, and artifacts are in
[Finite conductance floors in DRNs](conductance_floor_mitigation.md).

## Reproducibility

Study configs are under
[`examples/small_drn/mnist_ibm_devices`](../examples/small_drn/mnist_ibm_devices/).
Focused model and resume tests are in
[`tests/test_device_noise_backends.py`](../tests/test_device_noise_backends.py).

The source bundle staged for the GPU run has SHA-256
`2aedd9fde0cb7799678e57255557d69d56123ba1b8d215e46c0f17579eef36c4`.
The FP32 input checkpoint has SHA-256
`ee7dd6c9936e051e54c38e5167133f72b64624946183b86e349b3e878ebbc27e`.
The archived run configuration calls the PCM fit fields `g_max_us` and
`zero_threshold_us`, reflecting an initial unit interpretation. The current
strict schema corrects those names to `fit_max` and `zero_threshold`; the
numerical values and equations are unchanged. For the same reason, archived
PCM programming-report keys beginning `physical_` and ending `_us` contain
fit-coordinate values, not measured microSiemens. Current runs emit
`fit_target_*` and `fit_realized_*` instead.
The complete raw study, corrected summary, and exact source bundle are local
ignored artifacts under `results/mnist-ibm-pcm-cmo-noisy-recovery/`. An
original remote copy is retained outside the repository; its
machine-specific location is intentionally omitted.

## Primary sources

- [Analog Foundation Models](https://arxiv.org/abs/2505.09663)
- [IBM Analog Foundation Models code, commit 4f4c505](https://github.com/IBM/analog-foundation-models/tree/4f4c505b9d9af43af289788b3a29eeecaeb4e1ab)
- [IBM Hermes 64-core PCM chip](https://arxiv.org/abs/2212.02872)
- [Falcone et al., CMO/HfOx ReRAM](https://arxiv.org/abs/2502.04524)
- [AIHWKit 1.1 ReRAM model source](https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/inference/noise/reram.py)
