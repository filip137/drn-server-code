# Conv1 legacy zero-bias Conv-LR times-three counterfactual

This exploratory ordinary-MNIST study repeats the reviewed Conv1 legacy
zero-bias BPTT controls with the convolutional learning rate multiplied by
exactly three. The Dense learning rate is unchanged and `Bias_0` remains
frozen at exact zero. Both SGD and Adam run for ten epochs from seed 0; the
official test split is disabled.

The motivating bounded-uniform study only left the Conv1 legacy-SGD upper
Conv-rho edge open: moving from `rho_conv=.003` to `.009` tied the best
three-epoch validation accuracy at `94.50%`. Legacy-Adam was bracketed and its
corresponding times-three Conv-rho neighbor was lower (`95.14%` versus
`95.42%`). The Adam arm here is therefore a requested cross-optimizer
counterfactual, not a bounded-study-selected transfer.

Exact learning-rate vectors, ordered as `[ConvWeight_0, DenseWeight_0,
Bias_0]`, are:

- SGD: `[6.17259e-4, 4.76259e-5, 0]`
- Adam: `[8.66751e-4, 3.64618e-5, 0]`

These configs differ scientifically from their parent controls only in
`ConvWeight_0` learning rate and experiment-reporting metadata.
