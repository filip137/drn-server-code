# Differential ReRAM cohort A-to-B adaptation

Status: complete (2026-08-17)

## Question

Does adapting differential ReRAM conductances on a held-out hardware cohort
recover performance that is lost when a cohort-A model is deployed onto
cohort-B devices?

The A/B cohorts here are **measured device-trace cohorts**, not disjoint MNIST
example cohorts. The deterministic split contains 317 physical cells / 633
traces in A and 317 physical cells / 632 traces in B. The active trace sets are
disjoint.

## Frozen protocol

- Differential `G+ / G-` encoding, no biases.
- Model-local amplifier semantics: `voltage_amp=4.0`, `current_amp=0.25`.
- Cohort A: teacher mapping into paired common device windows, followed by 10
  epochs of measured-curve training.
- Cohort B receives the selected cohort-A checkpoint through two transfer
  rules using identical B virtual-device assignments:
  - `literal`: project each loaded A branch conductance independently onto its
    B curve;
  - `paired_affine_common_window`: map both branches into their shared
    per-pair B window before projection.
- A positive scalar readout gain is fitted on the 1,024-example calibration
  set after B deployment and then frozen. The source gain and recalibrated
  deployment metrics are both recorded.
- Each transfer rule has a matched zero-learning-rate deployment control and a
  10-epoch B adaptation run. Control and adaptation initial validation metrics
  and conductance tensors matched exactly.
- Both B adaptation arms reuse the cohort-A learning rates
  `(4.2e-10, 1.14e-12)`, seed, data order, solver, and epoch budget; there is no
  B-specific learning-rate search.
- Selection uses validation `KL(teacher || student)`. Every reported test
  endpoint is evaluated by a separate `ebl validate` process on 10,000 MNIST
  test examples.

## Results

| Hardware state | Test accuracy | Teacher agreement | Test KL |
|---|---:|---:|---:|
| Cohort A, selected after 10 epochs | 97.48% | 98.76% | 0.01409 |
| Cohort B literal transfer, no adaptation | 9.45% | 9.45% | 2.23586 |
| Cohort B literal transfer, after 10 epochs | 92.90% | 93.95% | 0.18439 |
| Cohort B common-window transfer, no adaptation | 42.96% | 43.32% | 2.03767 |
| Cohort B common-window transfer, after 10 epochs | **97.81%** | 98.12% | 0.03156 |

Measured adaptation effects on the test set:

- Literal transfer: **+83.45 percentage points** accuracy and **91.75%** KL
  reduction. It recovers 94.80% of the classification-accuracy loss relative
  to the cohort-A source, but remains 4.58 points below it.
- Common-window transfer: **+54.85 percentage points** accuracy and **98.45%**
  KL reduction. Its final accuracy is 0.33 points above the cohort-A source,
  although its KL is still 2.24 times larger and its teacher agreement is 0.64
  points lower.

The selected checkpoint is the tenth completed B epoch in both adaptation
arms. On validation, literal transfer moves from 9.94% / KL 2.22985 to 91.44%
/ KL 0.22053; common-window transfer moves from 42.52% / KL 2.03568 to 97.50%
/ KL 0.03037.

## Interpretation

There is strong value in **adapting the physical conductances after hardware
transfer** under this measured-device model. A fixed scalar readout calibration
cannot repair either deployment, while device-constrained updates recover most
or all classification performance.

The transfer parameterization still matters substantially:

- Literal projection discards the pair relationship. The source differential
  RMS values are 3.68 uS and 1.06 uS by layer, while immediate B values become
  7.71 uS and 7.61 uS. This branch mismatch dominates the intended signed
  signal; the fitted gain collapses from 7,943 to 56.2.
- Common-window transfer preserves a shared baseline, but each edge receives a
  different reachable span. The mean overlap spans are only about 18.9 uS out
  of the nominal 110 uS range, 0.85--1.18% of pairs have no overlap, and the
  immediate differential RMS contracts to 0.836 uS and 0.528 uS. One scalar
  gain cannot undo this edge- and layer-dependent distortion.
- B adaptation reshapes the conductance differences. In the common-window arm,
  final differential RMS recovers to 3.32 uS and 0.799 uS, close enough to the
  source operating scale to restore accuracy.

The 0.33-point common-window accuracy advantage over cohort A should not be
read as evidence that B is intrinsically better: this is one deterministic
seed, no paired significance test was run, and the worse KL/agreement show that
the adapted B logits reproduce the teacher less faithfully despite slightly
more correct class decisions.

## Scope limits

This experiment demonstrates the value of **analog/device-constrained weight
adaptation**, not a complete autonomous on-chip learning system. Gradients and
teacher targets are computed digitally, adaptation uses the full MNIST
training set for 10 epochs, and the measured backend uses a digital shadow with
global-nearest access to any point on a trace. It is not a sequential-pulse,
write-energy, endurance, latency, or few-shot adaptation model and may
overestimate what a physical programming circuit can recover economically.

## Reproducibility

- Branch: `codex/mnist-differential-cohort-ab-adaptation`
- Implementation commit: `4ef5cb8b`
- Campaign manifest:
  `campaigns/manifests/mnist_relu_drn_differential_cohort_ab_10ep.json`
- Raw ignored artifacts:
  `results/mnist-differential-reram-cohort-ab-adaptation-10ep-20260817-v1/`
- Terminal state: 10/10 stages complete; a resume/reuse pass revalidated all
  recorded result and artifact hashes.
- Verification: 492 repository tests passed (2 skipped), 42 legacy
  `labs/tests` passed (1 skipped), public describe checks passed, and GPU smoke
  plus exact-resume checks completed.
