# Experiment Result Reporting

Updated: 2026-08-17

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
`validation`, and record `official_test_read=false`. Their validation accuracy
is never a paper result.

An active paper result uses `evidence_class=ordinary_mnist_paper`, identifies
the sealed source checkpoint and reuse-or-fresh-run audit receipt, reports the
evaluated split as `test`, and records exactly one authorized official-test
evaluation. It must not overwrite or relabel the source selection bundle.
Repeated test access, a post-test config change, or a missing sealed inclusion
set makes the result ineligible for the paper table.

Historical deterministic medium-affine runs retain their explicit
medium-affine evidence classes and report the actual evaluated split. Current
Conv1/Conv2 handoffs additionally record the exact fixed handoff ID
`perfectdiode-conv12-unbounded-fixed-lr-20260729-v1` and preserve its
architecture-specific evidence status.

## Simulation Dashboard And Result-Directory Index

[`current_simulations.md`](current_simulations.md) contains a generated Active
block, a persistent Recent Experiment Directories table, and manual Queued,
Paused, and Analyzing sections. The generated block includes only
reporting-contract runs whose `status.json` says `running`. Concurrent runs are
grouped by `study_id`.

Refresh it with:

```bash
python -m experiments.reporting refresh-active \
  --results-root results \
  --document docs/current_simulations.md
```

Local runners refresh the block at state transitions. Remote status files must
be copied or synchronized into the local results tree before refreshing the
central checkout.

The generated Active block is operational run state. It is not a durable study
handoff: a run leaves that block when its `status.json` becomes `complete` or
`failed`.

The agent responsible for an experiment maintains one row in Recent Experiment
Directories for every top-level result directory it creates. Add the row before
creating the directory, and do not silently replace or omit smoke, failed,
partial, recovery, or superseded directories. Prefer one top-level directory
per study, with smoke and recovery outputs beneath it. When a separate
top-level directory is necessary, give it its own row and state its role.

Keep each row human-readable and limited to three fields:

- a descriptive link to the exact repository-relative result directory;
- the study-level state; and
- a short plain-language summary of the run.

Mention the target, a remote-only source location, an important blocker, or a
replacement directory in the summary when it materially helps discovery.
Detailed configuration, coverage counts, full scheduler provenance, validation
evidence, and artifact provenance remain in the run bundles and study analysis
rather than in this index.

Every live Jean Zay submission must be visible in this table before `sbatch`,
including production jobs, arrays, restored remote canaries, and retries. Use
the intended result-directory link whenever it is already known. If a direct
link cannot exist until Slurm assigns a job ID, put a concise pending label in
the Results cell and include the launch name, config or wrapper, and expected
remote output root or filename pattern in the summary. Immediately after
submission, add the returned Slurm job ID, change the state to `running`, and
replace the pending label with the result link as soon as it resolves. A retry
that writes a new top-level directory receives a new row; a retry that reuses
the same directory updates the existing row and retains prior job identity in
the run bundle.

For example:

```markdown
| `Jean Zay submission pending: conv3-fixed` | `planned` | `run_conv3.slurm`; expected under `/lustre/.../results/conv3-fixed/`. |
```

Use these exact study-level states:

| State | Meaning |
|---|---|
| `planned` | The directory and cases are declared, but no run has started. |
| `running` | At least one intended run is executing. |
| `collecting` | Remote execution is terminal, but the authoritative local copy has not been completely validated. |
| `ready-for-review` | Expected coverage is reconciled, the local evidence is present, included bundles validate, and exclusions or failures are identified. |
| `partial` | Some usable evidence exists, but the declared coverage or completion criteria were not met. |
| `complete` | Terminal supporting work such as a smoke or gate is retained but does not claim a scientific review handoff. |
| `failed` | The directory contains no valid result that meets its declared purpose. |
| `under-review` | A reviewing agent is actively analyzing the study. |
| `reviewed` | The scientific conclusion and evidence locations are recorded in `experimental_manifest.md`. |
| `superseded` | Another named directory replaces this directory; the row identifies the replacement. |

The controlling agent writes `planned`, `running`, `collecting`,
`ready-for-review`, `partial`, `complete`, `failed`, or `superseded`. A
reviewing agent writes `under-review` when it begins and `reviewed` only after
the finished manifest entry exists. `ready-for-review` is an explicit
collection and validation gate; it must not be inferred merely because every
currently discovered `status.json` is terminal.

The reporting refresher replaces only the generated Active block between its
markers, so it preserves the persistent table. Entries that are not yet
reviewed remain in the table. A reviewed or superseded entry may be pruned only
after `experimental_manifest.md` or its named replacement preserves the result
location and provenance.

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
