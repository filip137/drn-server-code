# Conv1/2/3 zero-bias baseline Conv-LR /3 long follow-up

This directory contains three exact ordinary-MNIST diagnostic configs:
one baseline-SGD arm for each of Conv1, Conv2, and Conv3.

The intervention mirrors the earlier Conv-LR x3 stress arm in the opposite
direction:

- divide every `ConvWeight_*` learning rate by exactly `3`;
- keep `DenseWeight_0` at its current baseline learning rate;
- keep every `Bias_*` learning rate exactly `0`; and
- increase the training budget from 30 to 50 epochs.

All seeds, data split and order, architecture, amplification, initialization,
preprocessing, weight projection, optimizer, and accepted operating points
remain unchanged. The official test split is disabled. These runs are
ordinary-MNIST optimization diagnostics, not paper-facing evidence.

Materialize or verify the deterministic files with:

```bash
python -m experiments.prepare_conv123_zero_bias_baseline_conv_lr_div3_long50
python -m experiments.prepare_conv123_zero_bias_baseline_conv_lr_div3_long50 --check
```

Launch transport is
`experiments/run_conv123_zero_bias_baseline_conv_lr_div3_long50_jeanzay.slurm`.
