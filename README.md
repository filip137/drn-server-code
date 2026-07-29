# Perfect-Diode Conv Amplification Experiments

This repository contains simulations and evidence for bidirectional
amplification in dissipative resistive networks. The active Conv work is
perfect-diode BPTT training of Conv1, Conv2, and Conv3 networks.

Start here:

- [`docs/conv_paper_hyperparameter_protocol.md`](docs/conv_paper_hyperparameter_protocol.md)
  indexes the active scientific contract;
- [`docs/conv_paper_experiment_definition.md`](docs/conv_paper_experiment_definition.md)
  defines the deterministic medium-affine paper grid;
- [`docs/perfectdiode_learning_protocol.md`](docs/perfectdiode_learning_protocol.md)
  defines the ordinary-MNIST Conv1/Conv2 rho workflow;
- [`docs/perfectdiode_conv3_learning_protocol.md`](docs/perfectdiode_conv3_learning_protocol.md)
  defines the ordinary-MNIST Conv3 `T/K` and rho workflow;
- [`docs/perfectdiode_bounded_weight_protocol.md`](docs/perfectdiode_bounded_weight_protocol.md)
  defines bounded-initializer selection; and
- [`docs/experiment_workflow.md`](docs/experiment_workflow.md) documents local,
  Akib, Trex, and Jean Zay execution.

Scientific runners own their readable configs and outputs.
`python -m experiments.launch` transports an existing command to a configured
target. Complete configs with frozen learning-rate vectors run through
`python -m experiments.exact_run`; `python -m experiments.rho_search` is a
probe-and-grid building block for the larger rho protocols.
