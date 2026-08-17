# Teacher-initialized differential ReRAM MNIST fine-tuning (10 epochs)

This exploratory follow-up measures the implemented differential `G+ - G-`
student after teacher mapping and exactly ten epochs of measured-trace KL
fine-tuning. It uses the registered `mnist_relu_drn_kd.v1` experiment.

## Controlled setup

- Teacher: the frozen bias-free `784 -> 50 -> 10` ReLU checkpoint used by the
  RESET studies. Its tensors are exactly equal to the checkpoint used by the
  earlier 20-epoch initialized-differential study.
- Student: bias-free dual-rail `[1568, 100, 20]` DRN with one independently
  assigned measured cohort-A `G+`/`G-` pair per physical edge.
- Mapping: search the fixed non-clipping layer-scale grid, map the teacher into
  each pair's shared affine reachable window, and fit one positive readout gain
  on the 1,024-example calibration subset.
- Programming: begin each device at raw pulse-zero RESET, perform exactly one
  global-nearest teacher-target write, then project fine-tuning targets onto
  the assigned raw trace.
- Fine-tuning: pure `KL(teacher || student)`, seed 42, batch size 16, calibrated
  differential learning rates `[4.2e-10, 1.14e-12]`, and exactly 10 epochs.
- Circuit scaling: model-local logical indexing, `voltage_amp = 4`, and
  `current_amp = 0.25`.

The campaign invokes training and then a separate held-out test validation of
the selected named-weights checkpoint. Checkpoints record the resolved
model-local differential stages and measured-device assignments, and
validation fails closed if the numerical topology or teacher provenance does
not match.

## Interpretation boundary

This run is directly comparable to the earlier 20-epoch initialized
differential arm as an epoch-budget screen. It is not an initialization-only
ablation against the literal-RESET run: the initialized path uses teacher KL
and a fitted readout gain, whereas the literal-RESET run used hard-label paired
squared error and fixed gain one.

## Entry points

- Config:
  `examples/mnist_relu_drn/measured_raw_differential_finetune_10ep.json`
- Campaign:
  `campaigns/manifests/mnist_relu_drn_initialized_differential_10ep.json`
- Raw output:
  `results/mnist-differential-reram-initialized-finetune-10ep-20260817-v1/`

Results are added only after the ten epoch records, epoch-boundary resume
state, selected checkpoint, and fresh-process test stage all complete.
