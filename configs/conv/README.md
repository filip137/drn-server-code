# Canonical MNIST Conv configs

The active paper protocol starts at
[`docs/conv_paper_hyperparameter_protocol.md`](../../docs/conv_paper_hyperparameter_protocol.md).
Scientific meaning and current results belong in the protocol documents; this
file is only a config inventory.

## Hard-sigmoid learning-rate configs

| Config | Schema / role | Disposition |
|---|---|---|
| [`hardsigmoid_lr_study_sgd_bs16_v1.json`](hardsigmoid_lr_study_sgd_bs16_v1.json) | `mnist-conv-lr-study/v1`; six-row scalar-LR study | executed; two baseline LRs frozen |
| [`hardsigmoid_lr_rescue_warmin_sgd_bs16_v2.json`](hardsigmoid_lr_rescue_warmin_sgd_bs16_v2.json) | `mnist-conv-lr-study/v2`; Conv1 amplified warm-in rescue | executed, exhausted, no LR selected |
| [`hardsigmoid_lr_relative_rho_sgd_bs16_v3.json`](hardsigmoid_lr_relative_rho_sgd_bs16_v3.json) | `mnist-conv-lr-study/v3`; four amplified rows | executed; four amplified LRs frozen |
| [`hardsigmoid_lr_layerwise_relative_rho_constant_sgd_bs16_v4.json`](hardsigmoid_lr_layerwise_relative_rho_constant_sgd_bs16_v4.json) | `mnist-conv-lr-study/v4`; layer-wise constant SGD | `retired_incomplete`; preserve content identity |
| [`hardsigmoid_lr_architecture_relative_rho_constant_sgd_bs16_v5.json`](hardsigmoid_lr_architecture_relative_rho_constant_sgd_bs16_v5.json) | `mnist-conv-lr-study/v5`; ordinary-MNIST architecture diagnostic | complete diagnostic |
| [`hardsigmoid_lr_conv2_two_rho_constant_sgd_bs16_v6.json`](hardsigmoid_lr_conv2_two_rho_constant_sgd_bs16_v6.json) | `mnist-conv-lr-study/v6`; ordinary-MNIST Conv2 two-rho diagnostic | complete, unbracketed |
| [`hardsigmoid_lr_conv3_scheme_two_rho_constant_sgd_bs16_v7.json`](hardsigmoid_lr_conv3_scheme_two_rho_constant_sgd_bs16_v7.json) | `mnist-conv-lr-study/v7`; ordinary-MNIST Conv3 study | canonical execution incomplete |
| [`hardsigmoid_lr_conv2_sgd_adam_boundary_constant_bs16_v1.json`](hardsigmoid_lr_conv2_sgd_adam_boundary_constant_bs16_v1.json) | optimizer-boundary diagnostic | attached-bias surface complete |

The active full-precision Conv1/Conv2 handoff is
[`docs/conv_hardsigmoid_lr_active_handoff_relative_rho_v3.csv`](../../docs/conv_hardsigmoid_lr_active_handoff_relative_rho_v3.csv).
The v4-v7 and optimizer-boundary results are ordinary-MNIST or incomplete
diagnostics curated in
[`docs/conv_learning_rate_diagnostics.md`](../../docs/conv_learning_rate_diagnostics.md).
They do not replace the active medium-affine handoff.

## Perfect-diode ordinary-MNIST diagnostic

| Config | Schema / role | Disposition |
|---|---|---|
| [`perfectdiode_conv12_sgd_adam_hparam_ordinary_mnist_v1.json`](perfectdiode_conv12_sgd_adam_hparam_ordinary_mnist_v1.json) | `mnist-conv-perfectdiode-hparam-study/v1`; Conv1/Conv2, three schemes, SGD/Adam, adaptive two-rho screen | protocol frozen; measurements pending |

Its immutable study identity is
`lrstudy_5afe8bc7c9180cd1c677d1d87a497e1d6a89c1599cd015d7a6c127ebb18f2e46`.
The user-fixed gains and `T/K`, security check, rho search, and conditional
long confirmations are defined in
[`docs/perfectdiode_learning_protocol.md`](../../docs/perfectdiode_learning_protocol.md).
This config is an ordinary-MNIST diagnostic and does not resolve any
medium-affine perfect-diode paper handoff.

## Conv3 legacy controls

These configs preserve separate content-addressed negative controls. None
selects an LR or modifies v7:

- [`hardsigmoid_lr_conv3_legacy_low_rho_rescue_bs16_v1.json`](hardsigmoid_lr_conv3_legacy_low_rho_rescue_bs16_v1.json);
- [`hardsigmoid_lr_conv3_legacy_gain200_rho5e5_3e4_bs16_v1.json`](hardsigmoid_lr_conv3_legacy_gain200_rho5e5_3e4_bs16_v1.json);
- [`hardsigmoid_lr_conv3_legacy_gain200_bias2_0p1x_bs16_v1.json`](hardsigmoid_lr_conv3_legacy_gain200_bias2_0p1x_bs16_v1.json).

## Diagnostic v1 examples

[`run_v1.diagnostic.example.json`](run_v1.diagnostic.example.json) and
[`sweep_v1.diagnostic.example.json`](sweep_v1.diagnostic.example.json)
demonstrate the generic v1 run/sweep grammar only. Their small `T/K`, LR,
epoch, and batch-limit values are placeholders, not protocol recommendations.

Plan the example without training:

```bash
python -m experiments.mnist_conv sweep \
  --config configs/conv/sweep_v1.diagnostic.example.json \
  --results-root /path/to/results \
  --plan-only
```

LR candidate run schemas v2-v7 are internal immutable stage artifacts. Execute
them through `python -m experiments.mnist_conv lr-study`; do not feed them to
the generic `run`, `sweep`, or `collect` surfaces.
