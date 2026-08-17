# Teacher-initialized single-conductance ReRAM MNIST fine-tuning (10 epochs)

This exploratory control measures the original one-conductance-per-physical-
edge student after ReLU-teacher programming and exactly ten epochs of
measured-trace KL fine-tuning. It uses the registered
`mnist_relu_drn_kd.v1` experiment.

## Controlled setup

- Teacher: the same frozen bias-free `784 -> 50 -> 10` ReLU checkpoint used by
  the initialized differential run.
- Student: bias-free dual-rail `[1568, 100, 20]` DRN with one independently
  assigned measured cohort-A conductance per physical edge.
- Mapping: lift each signed teacher matrix into non-negative dual rails, search
  the same non-clipping layer-scale grid, map every target to the assigned
  device's affine trace range, and fit one positive readout gain on the same
  1,024-example calibration subset.
- Programming: begin each device at raw pulse-zero RESET, perform exactly one
  global-nearest teacher-target write, then project fine-tuning targets onto
  that assigned raw trace.
- Fine-tuning: pure `KL(teacher || student)`, seed 42, batch size 16,
  calibrated one-conductance learning rates `[2.1e-10, 5.7e-13]`, and exactly
  10 epochs.
- Circuit scaling: model-local logical indexing, `voltage_amp = 4`, and
  `current_amp = 0.25`.

The differential arm uses twice these learning rates because its effective
weight update is the difference of two branch updates. Thus the declared rates
are gradient-equivalent in the ideal-device limit.

## Interpretation boundary

This is the explicit ten-epoch replay of the measured one-conductance arm in
the earlier 20-epoch KD study. It is a matched control for the initialized
`G+ - G-` run in teacher, data, seed, solver, objective, calibration, device
cohort, amplification, and minibatch budget. The architecture-appropriate
initial mappings differ: one conductance uses its device-local affine trace
range, while a differential pair uses the two traces' shared reachable window.

## Entry points

- Config: `examples/mnist_relu_drn/measured_raw_single_finetune_10ep.json`
- Campaign: `campaigns/manifests/mnist_relu_drn_initialized_single_10ep.json`
- Raw output:
  `results/mnist-single-reram-initialized-finetune-10ep-20260817-v1/`

Results are added only after all ten epoch records, the exact epoch-boundary
resume state, selected named weights, and fresh-process held-out validation
complete.
