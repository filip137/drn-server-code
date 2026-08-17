# Differential RESET-trained MNIST DRN (10 epochs)

This exploratory experiment applies the implemented weighted differential-pair
equilibrium to the strongest corrected arm of the recent single-device RESET
factorial. It is registered as `mnist_relu_drn_reset_differential.v1`.

## Controlled setup

- Student: bias-free dual-rail `[1568, 100, 20]` DRN.
- Synapse: one independently assigned measured `G+`/`G-` pair for every
  physical dual-rail edge.
- Amplification: model-local logical indices with `voltage_amp = 4` and
  `current_amp = 0.25`. The resolved differential stages are `(0, 1)` and
  `(1, 2)`, with forward gains `1` and `4` and post-layer energy metrics `1`
  and `0.0625`.
- Initialization: every physical branch starts at its assigned cohort-A raw
  trace's pulse-zero RESET state. There is no teacher-weight mapping or
  common-window pre-programming.
- Supervision: paired-output squared error against MNIST labels. The frozen
  ReLU teacher is used for agreement and KL diagnostics only.
- Production budget: 10 epochs after bounded learning-rate selection.

The two branches of each logical edge share one selected learning rate. The
relative-update probe computes an element-weighted RMS across both branch
gradients and normalizes it by the combined physical conductance RMS. The same
bounded center reduction, safety canaries, and 3-by-3 candidate grid used by
the RESET factorial then select one rate for each of the two logical layers.

Learning-rate selection and production are separate constructions. Production
is exactly reseeded and rebuilt, and the run fails closed unless device
assignments and the model-local differential stage scales match.

## Scope

This run tests literal dual RESET initialization. It does not guarantee
`G+ = G-` at initialization because independently assigned physical devices
can have different pulse-zero conductances. It therefore differs from the
teacher-initialized differential experiment, which first writes both devices
into a shared reachable window. The common conductance still cancels from
signed transfer only to the extent that the two realized branch baselines
match; it remains present in loading and power in every case.

The result is exploratory, single-seed numerical evidence rather than a
fabricated-circuit claim or a replicate-based uncertainty estimate.

## Versioned entry points

- Config:
  `examples/mnist_relu_drn_reset_differential/cohort_a_paired_squared_error_10ep.json`
- Campaign:
  `campaigns/manifests/mnist_relu_drn_reset_differential_10ep.json`
- Raw output:
  `results/mnist-differential-reram-reset-10ep-20260817-v1/`

The campaign runs training followed by explicit held-out test validation using
the selected named-weights checkpoint.

## Result

The campaign completed on 2026-08-17 from clean source commit
`187441fd11d27cdcc021aea35609703d5d4d6f8a`. The launcher exited zero, and both
the training and fresh-process test stages completed without non-finite values
or error markers. Learning-rate selection chose grid cell `cell_00`, with
rates `5.529865262288025e-7` and `1.7375886495148892e-7` for the first and
second differential pairs. This was the low/low edge of the bounded grid; the
declared protocol did not expand the grid, so this run does not establish that
either rate is globally optimal.

| Metric | Value |
| --- | ---: |
| Initial validation accuracy | 9.32% |
| Selected validation accuracy | 92.98% |
| Selected validation teacher agreement | 93.58% |
| Test accuracy | 93.94% |
| Test teacher agreement | 94.70% |
| Test paired squared error | 0.0933816 |
| Test raw teacher-to-student KL | 1.60862 |
| Diagnostic post-hoc calibrated test KL | 0.138329 |
| Teacher test accuracy | 97.36% |

The selected checkpoint is the tenth completed production epoch (zero-based
selection index `9`). The epoch-boundary resume checkpoint records epoch `10`.
The metrics stream contains exactly ten production records, with completed
epochs `1` through `10`; the training stage, including bounded learning-rate
selection, took 914.1 seconds.

Fresh-process validation reproduced the recorded model-local differential
semantics: resolved edges `(0, 1)` and `(1, 2)`, forward gains `1` and `4`, and
post-layer metrics `1` and `0.0625`. Generated diagnostic layer names differed
between the training and validation processes, as expected, but did not alter
the resolved equations. The selected named-weights catalog contains exactly
`base.conductance_plus.0`, `base.conductance_minus.0`,
`base.conductance_plus.1`, and `base.conductance_minus.1`.

Artifact integrity checks reproduced these hashes:

- selected weights:
  `9f73bfb5783ed1fce0d13a31df4eb1115da76ef1bff4b005de1c440a1047bb6b`;
- epoch-10 resume state:
  `21f32f510f3cbf199a5b4c5fb96875ec39199d459b64be78996841bce7e068e2`;
- learning-rate-selection record:
  `471be7bc391e203d074f793856582ffc0a7dae2033c896ca785e56154effdd43`.

The result demonstrates that the literal dual-RESET differential model can be
trained stably to 93.94% test accuracy under the declared ten-epoch protocol.
It does not yet estimate a differential-architecture benefit: the closest
single-device factorial arms used a 20-epoch budget, and a matched ten-epoch
single-device replay was not part of this campaign. It also does not test the
shared-reachable-zero initialization needed to distinguish independent RESET
baseline mismatch from common-floor cancellation. Those matched controls are
required before making a comparative architecture claim.

Machine-readable artifacts are under
`results/mnist-differential-reram-reset-10ep-20260817-v1/`. In particular,
`aggregate/results.json` records both completed stages, while each stage's
`result.json` records the numerical metrics and checkpoint provenance.
