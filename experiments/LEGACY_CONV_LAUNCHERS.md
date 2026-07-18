# Historical Conv launchers

All MNIST Conv shell and Slurm launchers that encode architecture, gain,
learning rate, diode, or solver settings in their command line are deprecated.
Their definitions are retained only to inventory historical diagnostics; they
must not be used to create new paper-facing rows. The former Python runner is
preserved in commit `a4a92aad` and the active path now rejects legacy scientific
flags with status 2.

The only supported launch surface for new resistive MNIST Conv BP work is:

```bash
python -m experiments.mnist_conv run --config RUN.json --results-root RESULTS
python -m experiments.mnist_conv sweep --config SWEEP.json --results-root RESULTS
python -m experiments.mnist_conv collect --sweep RESULTS/sweeps/NAME--SWEEP_ID
```

Operational values in historical launchers are not migrated or guessed.
Existing result roots remain immutable and should be classified through the
curated inventory/protocol documents before they are cited.

The former calibrated CSV pipeline and its fixed-size local/Slurm launchers
are fail-closed shims. New local execution uses
`experiments/run_mnist_conv_local.sh`; profile-driven Slurm execution uses
`experiments/submit_mnist_conv_slurm.py` and the generic array/collector
wrappers. See `docs/mnist_conv_workflow.md`.

Every retained launcher named `run_mnist_bp_conv*.sh`,
`run_mnist_bp_conv*.slurm`, `submit_mnist_bp_conv*.sh`, or
`submit_mnist_bp_conv*.slurm` has an executable fail-closed guard before
environment setup, remote commands, submission, or other work. Retained Slurm
scripts also carry an intentionally invalid deprecation directive, so `sbatch`
rejects them during submission rather than allocating a job. Their remaining
bodies are historical records only. CI inventories these filename patterns,
independently of whether a launcher directly or indirectly called the former
Python runner.
