# Teacher-initialized MNIST DRN distillation

This is an exploratory protocol for testing whether measured ReRAM updates can
recover a DRN after deployment, and whether a differential pair of devices is
more useful than one device per physical edge.

## Protocol

1. Train a bias-free `784 -> 50 -> 10` ReLU teacher on the deterministic MNIST
   train/validation split. The selected checkpoint is keyed by
   `teacher.dense_weight.0` and `teacher.dense_weight.1`.
2. Construct a bias-free `1568 -> 100 -> 20` DRN. Inputs and outputs use signed
   dual rails. Normalize each teacher matrix by its absolute maximum, search a
   fixed grid of non-clipping conductance-scale fractions per layer, and select
   the pair with the smallest calibration KL.
3. Fit one positive global readout gain on the calibration subset and freeze it
   for training and evaluation. The training loss is exclusively
   `KL(teacher || DRN)`; labels are used only for accuracy reporting.
4. Use `voltage_amp = 4` and `current_amp = 0.25`. The ideal arms compare one
   bounded conductance with a differential `G+ - G-` pair. In the differential
   interaction, transfer depends on `G+ - G-` and physical loading depends on
   `G+ + G-`.
5. For measured cohort A, assign virtual devices deterministically from the
   first half of formed physical cells. Every assigned device starts at RESET
   pulse zero, then receives exactly one global-nearest write. The single-device
   arm maps each normalized target affinely over that assigned virtual cell's
   calibrated range. The differential arm uses the shared reachable window of
   each independently assigned `G+`/`G-` pair, giving zero weights a common
   baseline before programming. Subsequent SGD targets accumulate in a digital
   shadow and are projected onto each assigned raw pulse-conductance trace.

The canonical configs are:

- `examples/mnist_relu/teacher.json`
- `examples/mnist_relu_drn/ideal_single.json`
- `examples/mnist_relu_drn/ideal_differential.json`
- `examples/mnist_relu_drn/measured_raw_single.json`
- `examples/mnist_relu_drn/measured_raw_differential.json`

The static campaign manifests are
`campaigns/manifests/mnist_relu_drn_kd_ideal.json` and
`campaigns/manifests/mnist_relu_drn_kd_measured.json`.

## Learning-rate calibration

An initial gradient-scale probe targeted a 0.1% conductance-RMS update per
minibatch. With 3,438 minibatches per epoch, that setting produced excessive
cumulative drift: the ideal single-device arm moved from validation KL
`0.010683` to `0.082855` after one epoch, and its selected checkpoint remained
the initialization after 20 epochs.

A matched three-epoch sweep then compared rates 100 and 1,000 times smaller.
The 100-times-smaller vector reduced validation KL to `0.001119`, versus
`0.002002` for the 1,000-times-smaller vector. The canonical rates are therefore
`[2.1e-10, 5.7e-13]` for a single device and the gradient-equivalent doubled
rates `[4.2e-10, 1.14e-12]` for a differential pair.

## Exploratory result

The bias-free ReLU teacher selected at epoch 11 reached 97.26% validation and
97.36% held-out test accuracy. The four matched DRN arms then produced:

| Backend and encoding | Initial validation KL | Selected validation KL | Validation accuracy | Test KL | Test accuracy | Test agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ideal, one device | 0.010683 | 0.000685 | 97.16% | 0.000712 | 97.42% | 99.83% |
| ideal, differential pair | 0.010683 | 0.000685 | 97.16% | 0.000712 | 97.42% | 99.83% |
| cohort A, one device | 2.242688 | 2.242688 | 8.20% | 2.246097 | 7.99% | 7.82% |
| cohort A, differential pair | 0.231839 | 0.010153 | 97.36% | 0.010119 | 97.30% | 98.98% |

The ideal arms are equal to numerical precision, as expected when doubling the
differential learning rate makes its effective `G+ - G-` update match the
single bounded parameter update. Two devices therefore provide no intrinsic
optimization benefit without device variation.

With measured cohort-A traces, the differential arm lowers selected validation
KL by 99.55% relative to the one-device arm and passes the predefined comparison
gate. Its own KL falls by 95.62% from its programmed initialization. The
one-device arm changes negligibly despite small target-projection errors. This
supports the interpretation that independently varying nonzero device offsets
destroy the dual-rail signal, while the common-window differential mapping
cancels each pair's baseline before fine-tuning. The result is specific to this
single seed and deterministic virtual-device assignment.

Machine-readable metrics and the matplotlib learning curves are in
`results/mnist-relu-drn-kd-exploratory-20260816/analysis/summary.json` and
`validation_comparison.png`.

## Comparison rule

The differential arm is called better only when its selected validation KL is
at least 5% below the matched single-device arm and neither student accuracy
nor teacher agreement is more than 0.2 percentage points worse. Use
`labs/tools/plot_mnist_relu_drn_kd_comparison.py` to create `summary.json` and
the matplotlib validation plot from completed run directories.

## Evidence boundary

These runs are exploratory, single-seed evidence. They establish numerical
feasibility and define a reproducible comparison, but they do not estimate
uncertainty across teacher seeds, device splits, or virtual-device assignments.
