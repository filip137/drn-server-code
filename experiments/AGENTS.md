# AGENTS

This guidance applies to runners, launchers, collectors, and monitors in
`experiments/`.

- Read the active scientific protocol before changing a Conv amplification
  experiment.
- Treat ordinary MNIST as the `T/K`, rho, learning-rate, and bounded-initializer
  selection dataset. Treat deterministic medium-affine MNIST as the paper-run
  dataset; never report selection accuracy as paper evidence.
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
- Within an assigned experiment task, agents may smoke, launch, monitor,
  diagnose, retry, cancel their own invalid or obsolete jobs, collect outputs,
  and interpret the results without separate approval checkpoints.
- Prefer one readable script over layered adapters. Add a shared abstraction
  only after a second real use.
- Before a long run, execute a short same-path smoke that constructs the real
  data and model, performs the principal computation (including an optimizer
  step for training), and writes an artifact.
- Give every run its own output directory. Record the config, command, commit,
  environment, host/job identity, logs, metrics, and checkpoints there.
- Keep local and Slurm entry points on the same command/config contract.
- Keep one scientific surface on one recorded target. Host selection must not
  become a scheme-dependent experimental factor.
- Propagate failures. Retry only after identifying the cause.
- Analyze results against the declared completion criteria and protocol.
  Clearly label partial evidence, confounds, and deviations; otherwise state
  the supported scientific conclusion directly.
- Preserve scientific results and lightweight summaries; do not commit large
  generated artifacts.
