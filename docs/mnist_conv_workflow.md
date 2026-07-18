# Canonical MNIST Conv workflow

The supported paper-facing interface is one module with three commands:

```bash
python -m experiments.mnist_conv run ...
python -m experiments.mnist_conv sweep ...
python -m experiments.mnist_conv collect ...
```

The numerical implementation remains in `labs/mnist_train.py`, but that module
does not define run identities, result paths, completion, or collection.

## Protocol gate

Read `conv_paper_hyperparameter_protocol.md` and every active protocol it links
before preparing a run. The hard-sigmoid T/K rule is defined, but its values and
the later training protocol are pending; the perfect-diode T/K rule also remains
pending. Therefore no paper-facing LR screen, long check, or final training may
be launched yet. Calibration/reference `T=64` is not automatically a training
value. This is enforced in code: final-category JSON can be validated and
planned for review, but execution fails before creating a result or attempt
until its exact protocol ID is added to the reviewed approval registry.

The examples below document the interface using minimal diagnostic placeholder
values. Those values do not imply selected T/K, learning rates, epoch budgets,
seed coverage, or other unresolved scientific settings.

## One run

A run JSON is the complete scientific input. Only infrastructure belongs on
the command line:

```bash
python -m experiments.mnist_conv run \
  --config RUN.json \
  --results-root RESULTS \
  --data-root MNIST_DATA \
  --device cuda
```

There are no scientific command-line overrides. Change a scientific setting
by creating and validating a different run JSON; its content-addressed run ID
will then change. Optimizer name, momentum, weight decay, LR, T/K, minimizer,
reset policy, initialization provenance, and every other executed scientific
choice are explicit in that JSON. Repeating a completed run skips it. A display
label does not force a rerun; an intentional stochastic replication must set a
new `replicate_id`. Retry and stale-claim recovery are explicit infrastructure
operations.

## A sweep

A sweep embeds or references one complete run, declares linked cases and
independent axes, and names every field allowed to vary. Planning expands and
validates every job without training:

```bash
python -m experiments.mnist_conv sweep \
  --config SWEEP.json \
  --results-root RESULTS \
  --plan-only
```

The command publishes the immutable canonical manifest below the sweep bundle
and prints its absolute path. Local workers consume that same manifest:

```bash
experiments/run_mnist_conv_local.sh \
  --config SWEEP.json \
  --results-root RESULTS \
  --data-root MNIST_DATA \
  --device cuda \
  --workers 2
```

Cases retain file order, axis paths are sorted, and values retain listed order.
Duplicate resolved run IDs are rejected. `varying_fields` must cover exactly
the fields that change. `collection.group_by` defines the comparison groups in
`summary_by_case.csv`; seeds are aggregated within each group and therefore
cannot themselves be grouping fields.

For calibration-driven amplification grids, first prepare a complete base run
and complete canonical calibration records. The converter copies no training
defaults and does not derive LR from calibration data:

```bash
python experiments/build_mnist_bp_conv_amp_calibrated_training_manifest.py \
  --base-run RUN.json \
  --calibration-records CALIBRATIONS.json \
  --name SWEEP_NAME \
  --seeds SEED ... \
  --output-sweep SWEEP.json
```

Calibration records must cover exactly `v1/c1`, `v4/c1`, and `v4/c0.25` for
the base architecture and nonlinearity. They include the calibration seed,
affine seed, provisional settling count, fixed-step flag, sample and batch
counts, target and measured first-hidden occupancy, every hidden-layer
measurement, and the nonlinearity-specific diode provenance. Incomplete,
failed, duplicate, off-protocol, or LR-bearing legacy CSV rows are rejected.

## Slurm

The Slurm executor first publishes the same manifest consumed locally. The
profile is operational only; it contains Slurm resources and concurrency, not
scientific settings:

```bash
python -m experiments.mnist_conv sweep \
  --config SWEEP.json \
  --results-root RESULTS \
  --executor slurm \
  --profile configs/executors/jeanzay-v100.json
```

The executor sizes the array from the manifest. Each array index calls the
same canonical sweep worker. It then submits collection with an `afterany`
dependency, so failed jobs are represented as missing/failed coverage rather
than suppressing collection. Manifest paths and job indices are private worker
plumbing rather than public scientific inputs.

## Collection and results

Workers publish only their own validated run bundles. One collector owns the
shared summaries:

```bash
python -m experiments.mnist_conv collect \
  --sweep RESULTS/sweeps/NAME--SWEEP_ID
```

Complete coverage writes the canonical summary. Incomplete, failed, pruned,
missing, corrupt, or mismatched coverage writes a partial summary and exits
nonzero; partial collection is an explicit monitoring-only option.

Before publication, the runner checks finite histories and metrics, LR history,
best-epoch consistency, strict checkpoint schemas, exact checkpoint/NPZ tensor
values, relative paths, and every artifact checksum. The completion manifest is
written last. Failed and pruned staging bundles remain under `attempts/`, and a
crashed collector cannot leave a permanent summary lock.

The result root has three namespaces:

```text
RESULTS/
  runs/LABEL--RUN_ID/        # validated, immutable completed bundles
  sweeps/NAME--SWEEP_ID/     # resolved sweep, manifest, collection, summaries
  attempts/RUN_ID/ATTEMPT_ID # running, failed, pruned, or superseded attempts
```

Paths serialized inside these bundles are relative. A run invoked directly
and the same run expanded from a sweep share one run ID and completed bundle.
Paper eligibility is derived only by collection; workers and launchers cannot
assert it.

## Legacy data

Historical launchers are deprecated and generated result roots remain
immutable. See `legacy_conv_result_inventory.md` for the read-only
`legacy_unverified` and curated classifications, and
`amplification_experiment_curation.md` before using any historical row.
Carried-state Conv results are diagnostic and are not promoted into the new
store.
