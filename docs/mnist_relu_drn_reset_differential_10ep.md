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
the selected named-weights checkpoint. Results are added here only after both
stages reach semantic completion.
