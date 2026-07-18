# Conv Paper Hyperparameter Protocol Index

Updated: 2026-07-18

Status: experiment definition frozen; hard-sigmoid input gains and operational `T/K` values frozen; perfect-diode calibration and `T/K` pending.

This file is the stable entry point for the Conv amplification paper protocol. The [former monolithic protocol](archive/conv_paper_hyperparameter_protocol_20260705.md) is the verbatim pre-split working document: it contains the July 5 fixed-`T/K` policy plus later July 18 working annotations, and is historical only. Its preserved SHA-256 is `dfda33f0c6436a95dab365353aa98d36fdcd68fc284619b6bfa8e01b6f69064d`.

## Protocol Order

The stages must be completed in this order:

1. Freeze the dataset, architecture, amplification schemes, and nonlinearities.
2. Calibrate and freeze `input_gain` for every architecture, nonlinearity, and amplification scheme.
3. Define and execute the operational `T/K` selection protocol using the frozen gains.
4. Define training batch size, optimizer, learning-rate selection, epoch budget, final seeds, and final inclusion rules in later protocols.

Changing an earlier stage invalidates every later stage that depends on it.

## Active Protocols

| Stage | Protocol | Status |
|---|---|---|
| Experiment definition | [`conv_paper_experiment_definition.md`](conv_paper_experiment_definition.md) | frozen |
| Hard-sigmoid input gain | [`conv_paper_hard_sigmoid_input_gain_protocol.md`](conv_paper_hard_sigmoid_input_gain_protocol.md) | protocol and all nine gains frozen |
| Perfect-diode input gain | [`conv_paper_perfect_diode_input_gain_protocol.md`](conv_paper_perfect_diode_input_gain_protocol.md) | frozen; measurements pending |
| Operational `T/K` | [`conv_paper_tk_protocol.md`](conv_paper_tk_protocol.md) | all nine hard-sigmoid values frozen; perfect-diode rule pending |

## Current Gate

Hard-sigmoid input-gain calibration and operational `T/K` selection are complete. Paper-facing hard-sigmoid training must still wait for the later protocol that freezes training batch size, optimizer, learning rate, epoch budget, final seeds, and checkpoint selection. The `T=64` used during input-gain calibration and as the residual reference is not an operational training value unless selected for an individual row.

Training batch size, SGD versus Adam, learning rates, epoch budget, final seed list, and final paper inclusion rules are intentionally unresolved. Historical defaults for those values are not active protocol decisions.

## Terminology

The dataset is **deterministic medium affine MNIST**. It must not be called permuted MNIST or pixel-permuted MNIST.

The amplification labels are:

- baseline: `v1/c1`;
- legacy amplification: `v4/c0.25`;
- proposed amplification / ours: `v4/c1`.

If an older document conflicts with this index or one of the active protocols above, the active protocol set wins.
