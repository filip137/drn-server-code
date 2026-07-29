# Experiment Result Reporting

Updated: 2026-07-29

This is the reporting contract for new active experiments. It records evidence
produced by scientific runners; it is not a launcher, scheduler, experiment
catalog, or substitute for the governing scientific protocol.

## Run Bundle

Every independently executable arm and seed receives one bundle beneath its
study directory. A runner may retain one descriptive grouping directory such
as `cells/` for compatibility:

```text
results/<study-id>/[<runner-group>/]<run-id>/
├── manifest.json
├── status.json
├── metrics.jsonl
├── result.json
├── checkpoints/
└── artifacts/
```

The files have distinct roles:

- `manifest.json` is immutable after the run starts. It records the study,
  arm, seed, evidence class, resolved scientific configuration, exact command,
  inputs and hashes, Git state, environment, host, and job identity.
- `status.json` is atomically replaced. Its state is exactly `running`,
  `complete`, or `failed`, with timestamps, heartbeat, progress, runtime
  identity, and a structured error for failures.
- `metrics.jsonl` is append-only. Each line names the stage, epoch or step,
  dataset split, and finite measurements recorded during execution.
- `result.json` is written only after a successful run. It records terminal
  metrics, completion criteria, protocol deviations, and a relative-path,
  SHA-256-indexed list of checkpoints and artifacts.
- `checkpoints/` and `artifacts/` expose the resulting files. Compatibility
  links may point to legacy flat trainer outputs without copying large files.

`result.json` is written before `status.json` transitions to `complete`. A
failed run retains its manifest, measurements, partial artifacts, and error in
`status.json`, but it must not contain `result.json`.

Validate a bundle with:

```bash
python -m experiments.reporting validate-run RESULTS/STUDY/RUN
```

## Evidence Classes

Ordinary-MNIST selection runs use
`evidence_class=ordinary_mnist_selection`, call the held-out 5,000 examples
`validation`, and record `official_test_read=false`. Their accuracy is never a
paper result.

Deterministic medium-affine runs use an explicit medium-affine evidence class
and report the actual evaluated split. Interim unbounded rows additionally
record that their LR source is an interim user-directed handoff.

## Active Simulations

[`current_simulations.md`](current_simulations.md) contains a generated Active
block and manual Queued, Paused, and Analyzing sections. The generated block
includes only reporting-contract runs whose `status.json` says `running`.
Concurrent runs are grouped by `study_id`.

Refresh it with:

```bash
python -m experiments.reporting refresh-active \
  --results-root results \
  --document docs/current_simulations.md
```

Local runners refresh the block at state transitions. Remote status files must
be copied or synchronized into the local results tree before refreshing the
central checkout.

## Study Analysis And Finished Reporting

After the relevant arms and seeds are collected, place aggregation under:

```text
results/<study-id>/analysis/
├── summary.json
├── summary.csv
├── report.md
└── plots/
```

The analysis records included and excluded runs, source `result.json` hashes,
per-seed measurements, aggregates, comparisons, limitations, and protocol
deviations. It must not silently combine unlike datasets, architectures,
optimizers, weight contracts, initializers, or handoff states.

After scientific review, manually add the study conclusion to
[`experimental_manifest.md`](experimental_manifest.md). Automation may check
links and hashes, but it must not write the scientific outcome or
interpretation.

Only small final CSVs, tables, and plots selected for the paper are promoted
from `analysis/` into a tracked paper-results directory. Raw datasets,
checkpoints, logs, and result bundles remain outside Git.
