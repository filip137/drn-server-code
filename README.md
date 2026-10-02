# Perfect-Diode Conv Amplification Experiments

This repository contains simulations and evidence for bidirectional
amplification in dissipative resistive networks, including Conv training,
endpoint-read-noise EqProp and the blockwise analog CIFAR-10 comparison.

## Current research questions

Start at [campaigns/README.md](campaigns/README.md):

- [Choosing beta for read-noise runs](campaigns/read-noise-beta/README.md).
- [Understanding the CIFAR-10 legacy gap](campaigns/cifar10-legacy-gap/README.md).

Each campaign separates its question/runbook, idea map, explorations, hypotheses,
experiments and reviewed results. Only its ledger is generated. The generic
[experiment-loop skill](.agents/skills/experiment-loop/SKILL.md) owns the full
Explorer → Codifier → Monitor → Reviewer lifecycle. The Codifier implements and
launches; a separate lightweight Monitor checks progress and escalates problems.
[run-experiment](.agents/skills/run-experiment/SKILL.md) remains a compatibility entry point.
See the [migration map](docs/campaign_workflow_migration.md) for existing plans.

## Scientific and execution contracts

Use the selected campaign's task-specific protocols. The Conv BPTT protocols
below are not a substitute for the declared EqProp or CIFAR experiment contract.

Start here:

- [`docs/conv_paper_hyperparameter_protocol.md`](docs/conv_paper_hyperparameter_protocol.md)
  indexes the active scientific contract;
- [`docs/conv_paper_experiment_definition.md`](docs/conv_paper_experiment_definition.md)
  defines the paper evidence and checkpoint-reuse contract;
- [`docs/perfectdiode_learning_protocol.md`](docs/perfectdiode_learning_protocol.md)
  defines the ordinary-MNIST Conv1/Conv2 rho workflow;
- [`docs/perfectdiode_conv3_learning_protocol.md`](docs/perfectdiode_conv3_learning_protocol.md)
  defines the ordinary-MNIST Conv3 `T/K` and rho workflow;
- [`docs/perfectdiode_bounded_weight_protocol.md`](docs/perfectdiode_bounded_weight_protocol.md)
  defines bounded-initializer selection; and
- [`docs/experiment_workflow.md`](docs/experiment_workflow.md) documents local,
  Akib, Trex, and Jean Zay execution; and
- [`docs/experiment_reporting.md`](docs/experiment_reporting.md) defines run
  bundles, execution records, and study analysis.

Current run handoffs and result paths belong to their [campaign experiment](campaigns/README.md)
or [pilot note](campaigns/pilots/README.md); reviewed conclusions belong to its result/pilot
note. [`docs/current_simulations.md`](docs/current_simulations.md) and
[`docs/experimental_manifest.md`](docs/experimental_manifest.md) are historical snapshots.

Scientific runners own their readable configs and outputs.
`python -m experiments.launch` transports an existing command to a configured
target and can run a synchronous local canary before making a single remote
production submission. Complete configs with frozen learning-rate vectors run
through `python -m experiments.exact_run`; `python -m experiments.rho_search`
is a probe-and-grid building block for the larger rho protocols.
