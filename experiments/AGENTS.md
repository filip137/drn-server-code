# AGENTS

This guidance applies to runners, launchers, collectors, and monitors in
`experiments/`.

Role ownership and handoffs live in
[experiment-loop](../.agents/skills/experiment-loop/SKILL.md); resource, gate,
launch and recovery policy lives in [experiment_workflow](../docs/experiment_workflow.md).
The following rules concern implementation, not a separate execution workflow.

- Read the active scientific protocol before changing a Conv amplification
  experiment.
- Treat ordinary MNIST as the `T/K`, rho, learning-rate, and bounded-initializer
  selection dataset. Active paper evidence uses the official MNIST test split
  only after the frozen-contract gate in root AGENTS.md; medium-affine MNIST
  is historical or optional robustness work. Never report selection accuracy
  as paper evidence. Explicit EqProp/CIFAR contracts retain their own scope.
- Make architecture, dataset, preprocessing, nonlinearity, amplification,
  initialization, `T/K`, optimizer, learning rate, seeds, epoch budget, and
  checkpoint rule explicit.
- Label output as paper-facing, diagnostic, or historical replay. Do not merge
  unlike protocols silently.
- Keep numerical work in the scientific runner and transport in
  `experiments.launch`. Transport treats the command as opaque.
- Treat a complete config with an explicit optimizer and learning-rate vector
  as executable science. Run it with `experiments.exact_run`; do not derive,
  tune, or rewrite its scientific fields.
- Prefer one readable script over layered adapters. Add a shared abstraction
  only after a second real use.
- Smoke mode constructs the real data and model, performs the principal
  computation (including an optimizer step for training), and writes an artifact.
- Give every run its own output directory. Record the config, command, commit,
  environment, host/job identity, logs, metrics, and checkpoints there.
- New runners and maintained active runners use the lightweight bundle in
  `docs/experiment_reporting.md`: immutable `manifest.json`, atomic
  `status.json`, append-only `metrics.jsonl`, and successful-only
  `result.json`.
- Run bundles record operational observations; experiment/pilot notes own the
  current handoff and result paths. The Reviewer writes interpretation in the
  result/pilot note. Do not require entries in retired central documents.
- Keep local and Slurm entry points on the same command/config contract.
- Routine launches use declared T/K and LR vectors without the retired gradient,
  residual, rho-probe or CIFAR migration gates in older plans. CIFAR preparation
  uses configured iterations by default; `--qualify-solver` requests the historical
  solver diagnostic explicitly. Preserve unaudited status rather than claiming a pass.
- Propagate failures; do not emit successful completion for incomplete artifacts.
- Analyze results against the declared completion criteria and protocol.
  Clearly label partial evidence, confounds, and deviations; otherwise state
  the supported scientific conclusion directly.
- Preserve scientific results and lightweight summaries; do not commit large
  generated artifacts.
