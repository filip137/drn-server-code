# Conv Paper Hyperparameter Protocol Index

Updated: 2026-07-26

Status: the experiment definition, all nine hard-sigmoid input gains, all nine
hard-sigmoid operational `T/K` pairs, and the six Conv1/Conv2 hard-sigmoid
seed-0 learning rates are frozen. Perfect-diode calibration, perfect-diode
`T/K`, deterministic-medium-affine Conv3 and perfect-diode learning-rate
rules, and the final-training protocol remain unresolved.

This file is the stable entry point for the Conv amplification paper protocol.
The [former monolithic protocol](archive/conv_paper_hyperparameter_protocol_20260705.md)
is historical only. Its preserved SHA-256 is
`dfda33f0c6436a95dab365353aa98d36fdcd68fc284619b6bfa8e01b6f69064d`.

## Protocol order

The stages must be completed in this order:

1. Freeze the dataset, architecture, amplification schemes, nonlinearities,
   output encoding, and loss.
2. Calibrate and freeze `input_gain` for every architecture, nonlinearity, and
   amplification scheme.
3. Select and freeze operational `T/K` using those gains.
4. Select and freeze a seed-0 learning-rate handoff for each eligible row.
5. Freeze the final epoch budget, model seeds, checkpoint rule, and inclusion
   rules.

Changing an earlier stage invalidates every later stage that depends on it.

## Active protocols

| Stage | Protocol | Status |
|---|---|---|
| Experiment definition | [`conv_paper_experiment_definition.md`](conv_paper_experiment_definition.md) | frozen |
| Hard-sigmoid input gain | [`conv_paper_hard_sigmoid_input_gain_protocol.md`](conv_paper_hard_sigmoid_input_gain_protocol.md) | protocol and all nine gains frozen |
| Operational `T/K` | [`conv_paper_tk_protocol.md`](conv_paper_tk_protocol.md) | all nine hard-sigmoid pairs frozen; perfect-diode rule pending |
| Hard-sigmoid LR | [`conv_paper_learning_rate_protocol.md`](conv_paper_learning_rate_protocol.md) | Conv1/Conv2 six-row seed-0 handoff complete; medium-affine Conv3 unresolved |

The learning-rate handoff combines the two v1 baseline selections with the
four v3 parameter-relative-rho amplified selections. The authoritative
full-precision table is
[`conv_hardsigmoid_lr_active_handoff_relative_rho_v3.csv`](conv_hardsigmoid_lr_active_handoff_relative_rho_v3.csv).
Batch size 16 and plain SGD are frozen only for this completed Conv1/Conv2
screen.

## Current gate

The completed seed-0 handoff is not final paper training. The following remain
blocked:

- all perfect-diode calibration, `T/K`, and LR work;
- the deterministic-medium-affine Conv3 hard-sigmoid LR;
- final epoch budget and model seeds;
- final checkpoint and inclusion rules;
- paper-facing multi-seed training.

The `T=64` calibration/reference value is not an operational training value
unless a row's `T/K` protocol selected it.

Ordinary-MNIST optimizer studies do not fill any of these gaps. Their useful
results and negative controls are curated separately in
[`conv_learning_rate_diagnostics.md`](conv_learning_rate_diagnostics.md).
The focused completed Conv2 SGD/Adam study remains in
[`conv2_sgd_adam_boundary_diagnostic_protocol.md`](conv2_sgd_adam_boundary_diagnostic_protocol.md).
The former pending medium-affine perfect-diode input-gain specification is
preserved for provenance in
[`archive/conv_paper_perfect_diode_input_gain_protocol_20260718.md`](archive/conv_paper_perfect_diode_input_gain_protocol_20260718.md);
it is historical rather than an active calibration handoff.

## Terminology

The paper dataset is **deterministic medium affine MNIST**. It must not be
called permuted MNIST or pixel-permuted MNIST. A document explicitly labeled
as an ordinary-MNIST diagnostic is not paper-facing evidence.

The amplification labels are:

- baseline: `v1/c1`;
- proposed amplification / ours: `v4/c1`;
- legacy amplification: `v4/c0.25`.

If an older document conflicts with this index or an active protocol above,
the active protocol set wins.
