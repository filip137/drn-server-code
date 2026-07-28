# Canonical MNIST Conv workflow

The supported interface is one module with four command families:

```bash
python -m experiments.mnist_conv run ...
python -m experiments.mnist_conv sweep ...
python -m experiments.mnist_conv collect ...
python -m experiments.mnist_conv lr-study ...
```

The numerical implementation remains in `labs/mnist_train.py`, but that module
does not define run identities, result paths, completion, or collection.

## Protocol gate

Read `conv_paper_hyperparameter_protocol.md` and every active protocol it links
before preparing a run. All nine hard-sigmoid gains and operational `T/K`
pairs are frozen. The six Conv1/Conv2 seed-0 LRs are frozen through the
combined v1-baseline/v3-amplified handoff.

Perfect-diode calibration and `T/K`, medium-affine Conv3 and perfect-diode LR
rules, and the final epoch, seed, checkpoint, and inclusion protocol remain
pending. No paper-facing long run or final training is authorized.
Calibration/reference `T=64` is not automatically a training value.

Final-category JSON can be validated and planned for review, but execution
fails before creating a result or attempt until its exact protocol ID is added
to the reviewed approval registry. Ordinary-MNIST optimizer studies are
curated separately in
[`conv_learning_rate_diagnostics.md`](conv_learning_rate_diagnostics.md) and
cannot bypass this gate.

The examples below document the interface using minimal diagnostic placeholder
values. Those placeholders do not reproduce the frozen hard-sigmoid rows and do
not imply the frozen LR handoff, epoch budgets, seed coverage, or other
unresolved scientific settings.

## A staged LR study

An LR study has a content-addressed source config and publishes immutable stage
manifests below `RESULTS/lr_studies/`. The available stages depend on the
study schema. For the canonical v1-v3 chain:

```text
probe -> range -> candidates -> select
```

For example:

```bash
python -m experiments.mnist_conv lr-study \
  --stage probe \
  --config configs/conv/hardsigmoid_lr_study_sgd_bs16_v1.json \
  --results-root RESULTS \
  --data-root MNIST_DATA \
  --device cuda
```

Later stages take `--study STUDY_DIR`. Scientific choices come only from the
study JSON. Local and Slurm workers consume the same manifest, completed
entries resume by validated hash, and the stage-completion marker is written
last.

Run schemas `mnist-conv-run/v2` through `/v7` are internal LR-stage artifacts.
The generic `run`, `sweep`, and `collect` commands accept only
`mnist-conv-run/v1`; this prevents an LR artifact from being executed through
the legacy backend or summarized with the wrong contract.

## One run

A generic run JSON uses `mnist-conv-run/v1` and is the complete scientific
input. Only infrastructure belongs on the command line:

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

A generic sweep contains only `mnist-conv-run/v1` entries, embeds or references
one complete run, declares linked cases and
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

## Long simulations on main and Trex

After the experiment plan, tracker entry, smoke test, and applicable scientific
gates have passed, aggregate and verify their schema-specific receipts with
`python -m experiments.local_dispatch build-preflight|verify-preflight`, then
dispatch with `plan|start|status`. The dispatcher uses one immutable request
for either a fresh window in the existing `tmux main` session or one bounded
SSH transaction into a fresh attempt-named tmux session on Trex. It never uses
`send-keys`, retries automatically, or falls back to a different target.

See [`local_dispatch.md`](local_dispatch.md) for the request contract, checked-in
profiles, commands, receipts, and failure semantics. `start` requires the
approved plan and a fresh production tracker-gate receipt, then revalidates the
plan-declared v2 preflight envelope, each inner semantic receipt, the pinned
effective environment, and the exact clean Git commit on the target. This
transport layer does not authorize a run or bypass any protocol, plan, tracker,
or preflight gate. Its terminal fail-stop behavior begins only after the
long-run `start`, not during preflight-envelope construction, planning,
code/test repair, environment setup, or bounded developer probes.

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

The result root has four namespaces:

```text
RESULTS/
  runs/LABEL--RUN_ID/        # validated, immutable completed bundles
  sweeps/NAME--SWEEP_ID/     # resolved sweep, manifest, collection, summaries
  lr_studies/NAME--STUDY_ID/ # immutable staged LR-study bundles
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
