# Raw LoRA/HWA Results

The repository-root `results/` directory is the canonical home for new raw
LoRA and hardware-aware experiment output in this worktree. Raw contents are
local and ignored by Git; this guide is the only file in the tree intended
for Git.

Use one stable, kebab-case study ID for a scientific comparison:

```text
results/<study-id>/
  runs/<arm>/<ebl-run-id>/
  deployment/
  analysis/
  README.md
```

- `runs/` contains native `python -m ebl` run directories.
- `deployment/` contains device-programming or deployment outputs when used.
- `analysis/` contains aggregate JSON, tables, and plots.
- The study-local `README.md` is an optional working summary and is also
  ignored.

The subdirectories are a convention, not a required schema. Create only the
ones a study needs.

## Launch example

Pass an arm-specific parent to the public CLI. The runtime creates a new
exclusive run directory below it:

```bash
python -m ebl train \
  --config examples/small_drn/base.json \
  --output-dir results/<study-id>/runs/<arm>
```

Never reuse an existing native run directory. Keep the native files
(`config.resolved.json`, `manifest.json`, `metrics.jsonl`, `result.json`,
`status.json`, artifacts, and checkpoints) together.

## Experiment indexes

- The `Active` block in
  [`docs/current_simulations.md`](../docs/current_simulations.md) is refreshed
  automatically from running native statuses below `results/`. Its paused,
  analyzing, and queued sections remain human-maintained.
- [`docs/experimental_manifest.md`](../docs/experimental_manifest.md) lists
  concluded studies and progress toward the LoRA/HWA research goal.

The automatic refresh is informational and best-effort. Update the other
sections and the finished ledger manually. None of these documents approves,
validates, or blocks a run.

## Existing raw results

Existing exploratory studies remain at their original locations:

- `labs/cases/passive_layerwise_lora_physical_test/`
- `labs/cases/perfect_diode_hwa_lora_comparison/`
- `labs/cases/mnist_perfect_diode_hwa_lora_comparison/`

They are indexed in the finished-simulation ledger but are not moved into
`results/`.
