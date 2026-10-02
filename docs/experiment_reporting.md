# Experiment Result Reporting

Updated: 2026-09-25

This is the reporting contract for new active experiments. It records evidence
produced by scientific runners; it is not a launcher, scheduler, experiment
catalog, or substitute for the governing scientific protocol.

## Campaign result notes and ledgers

Small pilots use one evolving note under the skill's
[pilot convention](../.agents/skills/experiment-loop/references/artifacts.md#small-pilots)
for specification, handoff, evidence and review. Full campaign scaffolding and a separate
analysis report are not required. Canonical bundles still apply. The following
campaign metadata/ledger rules apply to full campaigns.

For adopted research questions, [campaigns](../campaigns/README.md) contain the
scientific question, claims and experiment specifications. A small campaign
`series/NNN-*/results/*.md` note points to canonical local run bundles and records
coverage, measurements, limitations and reviewed hypothesis judgments. It does
not replace any manifest, status, metric, checkpoint or validation requirement.

[experiment-loop](../.agents/skills/experiment-loop/SKILL.md) owns the handoff:
Codifier/Monitor writes factual coverage, measurements and validation evidence;
Reviewer adds interpretation and verdicts to that same note. Before review,
`verdicts: {}` is valid; it does not mean inconclusive or not-tested. A successful
run may contradict its hypothesis. Keep its conclusion in that result/pilot note.

Only the campaign `ledger.md` is generated, from experiment metadata and those
reviewed notes. It does not infer scientific conclusions or live job state.
Imported historical prose is labelled `imported-summary` until a separate audit
actually checks its source evidence. The old current_simulations and
experimental_manifest documents are historical snapshots; do not refresh or append
to them. This supersedes older plans and external closeout skills. See the
[artifact conventions](../.agents/skills/experiment-loop/references/artifacts.md).

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
  `paused`, `complete`, or `failed`, with timestamps, heartbeat, progress,
  runtime identity, a checkpoint reference/hash for pauses, and a structured
  error for failures.
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

A planned pause also has no `result.json`. It records a non-empty full-state
checkpoint in `status.pause`, with a relative path, SHA-256 and reason. The
runner remains responsible for scientific resume compatibility. Exact MNIST
continuations resume the existing bundle after identity and checkpoint checks;
CIFAR recovery retains the paused bundle and starts a linked new bundle.
Historical `running`/`checkpointed_pause` and `failed`/`PlannedPause` (exit 75)
records are recognized as pauses without rewriting their evidence.

Validate a bundle with:

```bash
python -m experiments.reporting validate-run RESULTS/STUDY/RUN
```

Validation checks result/manifest/metric/artifact hashes, matching identities,
finite JSON metric records, well-formed completion criteria, and pause checkpoint
integrity. A valid running, paused or failed bundle is not a completed experiment.
Scientific eligibility and full study coverage still require the declared checks.
Some existing sweeps record `completion.criteria_met=false` for a scientifically
rejected candidate whose computation completed. Preserve that negative evidence;
it is not an operational failure or an eligible selected candidate.

Before publishing success, `exact_run` additionally requires ordered epoch records
covering the resolved budget, finite training/evaluation measurements, a valid best
epoch, and non-empty best/final checkpoints. Smoke and diagnostic overrides use
their resolved epoch budget; skipping official-test evaluation does not skip these
checks. Exit zero without the required training evidence is recorded as failure.

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

## Execution records and status

The owning experiment/pilot note is the durable execution record. Before launch,
record cases, configs/commands, target, budget, expected duration and top-level result
paths there. Add handles after launch and keep one current monitoring handoff. Retain
links to smoke, failed, partial, recovery and superseded outputs; prefer one study
root with such attempts beneath it. Detailed provenance stays in the run bundles.

This also applies before every live Jean Zay submission, including arrays, canaries
and retries. If the output path depends on a job ID, record the wrapper/config and
expected remote path pattern, then add the returned ID and resolved path. Preserve
prior attempt identities and identify replacements in the same record.

Runners write observations to their own bundles. For routine checks, synchronize
assigned remote observations and inspect only the explicit run directories:

```bash
python -m experiments.reporting inspect-runs RESULTS/STUDY/RUN1 RESULTS/STUDY/RUN2
```

The compact JSON reports progress, runtime handles, heartbeat age, metric file size
and alerts without hashing checkpoints or scanning the result tree. A recent heartbeat
does not prove process liveness; `stale` means older than 45 minutes by default and
`unknown` means missing/invalid heartbeat. Pauses remain `paused` regardless of age.
Exit 1 means a record needs attention, not that a process is dead or recovery is
authorized. Compare successive observations with process/launcher state and bounded
logs; failed synchronization alone is a visibility gap. Use `--stale-after-minutes`
for an explicitly different monitoring cadence. Exit 75 is a planned pause only
with the runner's checkpoint evidence. Validate fully at collection, not every poll.

Update the handoff at meaningful state changes. Distinguish execution from collection,
review readiness and completed review: `ready-for-review` requires reconciled expected
coverage, locally validated included bundles and identified failures/exclusions. Report
partial or failed coverage honestly. Record review completion only once interpretation
and the next decision are in the result/pilot note. These handoff phases do not change
the existing campaign metadata vocabulary or run-bundle schema.

[`current_simulations.md`](current_simulations.md) and
[`experimental_manifest.md`](experimental_manifest.md) are frozen historical references.
There is no required central table, dashboard refresh or duplicate conclusion entry.
The optional legacy `refresh-active` command requires an explicit `--document` with
existing Active block markers and rejects these retired documents. Routine monitoring
uses `inspect-runs`; it does not scan all historical results.

## Study Analysis And Finished Reporting

After collection, put any needed machine-readable aggregation or plots under
`results/<study-id>/analysis/`. A possible layout is:

```text
results/<study-id>/analysis/
├── summary.json
├── summary.csv
├── report.md
└── plots/
```

Create only the outputs the question needs; this is not a required file checklist.
The pilot/result note is the human review. Do not duplicate its prose in `report.md`
or add a CSV/plot merely to fill this layout. Keep raw manifests and hash lists in
bundles; role handoffs contain coverage, key measurements, validation outcomes,
exceptions and artifact paths. Read selected fields rather than dumping every record.

The analysis records included and excluded runs, source `result.json` hashes,
per-seed measurements, aggregates, comparisons, limitations, and protocol
deviations. It must not silently combine unlike datasets, architectures,
optimizers, weight contracts, initializers, or handoff states.

After scientific review, complete the interpretation and next decision in the
existing result/pilot note, update the experiment state and refresh an established
campaign ledger when its inputs change. Automation may check links and hashes;
scientific verdicts require review rather than inference from process status.

Only small final CSVs, tables, and plots selected for the paper are promoted
from `analysis/` into a tracked paper-results directory. Raw datasets,
checkpoints, logs, and result bundles remain outside Git.
