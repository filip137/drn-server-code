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

## Result

The campaign completed from clean source commit
`00bc68760f7cd0018e8bf62c1c7c98ad57bd7553`. Both stages exited zero, the
training stream contains one initialization record followed by exactly ten
completed epochs, and neither stage contains a non-finite metric. The
common-window map selected teacher scale fractions `[1.0, 0.125]` and fitted
readout gain `7943.282347`.

| Evaluation point | KL | Accuracy | Teacher agreement |
| --- | ---: | ---: | ---: |
| mapped initialization, validation | 0.231839 | 89.12% | 90.06% |
| selected epoch 9, validation | 0.0138965 | 97.42% | 98.94% |
| selected checkpoint, held-out test | 0.0140905 | 97.48% | 98.76% |

The selected validation KL recovered `94.01%` of the initialization gap. The
tenth epoch regressed slightly to validation KL `0.0156730`, accuracy `97.30%`,
and agreement `98.72%`, so selection correctly retained epoch 9. Training took
`222.72 s`; fresh-process validation of all 10,000 held-out examples took
`1.49 s`.

The selected checkpoint records the expected model-local stages
`[(0, 1, 1, 1), (1, 2, 4, 0.0625)]`, all four stable `G+`/`G-` conductance
keys, and the measured-device assignment hashes. Its SHA-256 is
`82924b43399920ad3df342c1cf12dd72fe3ea0e9b34908b8396dcac7f210f031`.
The exact epoch-10 resume state has SHA-256
`b1c843b6d8000b6272052db6625f83dc1c4a26abc32fe3eb62c96868731ca4b6`.
Fresh validation reproduced the checkpoint's topology and device provenance.

## Epoch-budget comparison

All first-ten epoch metrics match the earlier 20-epoch initialized-
differential run exactly. This makes the comparison a deterministic epoch-
budget screen rather than a separate stochastic replicate. The 20-epoch run
ultimately selected validation KL `0.0101526` and produced test KL `0.0101188`,
test accuracy `97.30%`, and agreement `98.98%`. Thus ten epochs produced
slightly higher test accuracy (`+0.18` percentage points, 18 examples), while
the longer run produced lower KL and higher teacher agreement. Because
selection minimizes validation KL and this is one seed, the accuracy
difference is not evidence that ten epochs generalize better.

The initialized ten-epoch result is also `3.54` percentage points above the
literal-RESET differential run's `93.94%` test accuracy. That gap is not a
clean initialization ablation: the literal-RESET run used paired hard-label
squared error with fixed gain one, while this run used teacher KL and a fitted
gain.
