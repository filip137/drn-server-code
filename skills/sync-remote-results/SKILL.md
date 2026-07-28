---
name: sync-remote-results
description: Repatriate experiment results from Akib, Trex, or Jean Zay into one local result store, verify resumable transfers, and evaluate only the local copy. Use when Codex needs to find, copy, recover, validate, archive, summarize, or register remote run, study, shard, checkpoint, metric, or Slurm artifacts.
---

# Sync Remote Results

Use a two-pass transfer so small evaluation artifacts arrive first and large
checkpoints or step logs do not block the initial review.

## 1. Fix the bundle identity

Record these values before copying:

- stable run, study, sweep, or shard ID;
- exact SSH host and run-specific absolute remote path;
- one permanent local result root and an ID-derived `.incoming` path;
- terminal job/process evidence and required completion marker;
- experiment-specific local validator and evaluator.

Use these SSH targets:

- Akib: `akib`
- Trex: `filip@trex`
- Jean Zay: `jean-zay`

Take the exact result path from the launch record or remote resolved config.
Never infer it from a label, copy a shared result root, or create a different
timestamped destination on retry.

Keep one canonical local result root across worktrees. Stage each host below:

```text
RESULT_ROOT/
  .incoming/<bundle-id>/shards/<host>/
  _transfer_receipts/<bundle-id>/<host>-<scope>.json
```

## 2. Close or classify the remote snapshot

Confirm that the producing process has stopped. On Jean Zay, also require a
terminal `sacct` state. For a completed bundle, require the experiment's final
completion marker, which must have been written last.

For a crash, copy only after the writer has stopped and use
`--allow-incomplete`. Also pass the last valid `--completion-marker` when one
exists. Preserve that completion chain and call the overall copy incomplete;
do not synthesize a missing aggregate, selector, or handoff.

## 3. Pull evaluation metadata first

Run the bundled script from the repository root:

```bash
python skills/sync-remote-results/scripts/sync_remote_results.py \
  --host HOST \
  --remote-path /exact/remote/run-or-shard \
  --local-stage /absolute/RESULT_ROOT/.incoming/BUNDLE_ID/shards/HOST \
  --receipt /absolute/RESULT_ROOT/_transfer_receipts/BUNDLE_ID/HOST-metadata.json \
  --mode metadata \
  --completion-marker relative/path/to/complete.json
```

Use `--plan-only` first when paths are new. Repeat `--completion-marker` when
several host-shard markers are required. Use `--include-relative
'slurm/*.err'` for a specific diagnostic log.

Metadata mode copies JSON, JSONL, compact tables, configs, manifests, and
completion records. It excludes checkpoints and `step_log.csv`, then performs
a checksum dry-run over the small copied set and content-hashes the local
metadata receipt. Re-running the command resumes the same staging directory.
An interrupted attempt can reuse its receipt path because no receipt was
written; after a successful attempt, use a new receipt path for a later pass.

Treat this as an evaluation snapshot, not a complete physical archive.

## 4. Evaluate only from the local path

Read the resolved config and valid completion chain first. Run the
experiment-specific evaluator against `.incoming`; never build a conclusion
from a remote path or terminal scrollback.

For a canonical MNIST Conv sweep, collect locally:

```bash
python -m experiments.mnist_conv collect \
  --sweep /absolute/local/path/to/sweep
```

For a full perfect-diode screen shard, validate locally:

```bash
python experiments/run_mnist_conv_perfectdiode_hparam.py \
  status --shard /absolute/local/path/to/shard
```

That status command requires all manifest-bound artifacts, so run it after the
full pass. On a metadata-only or crashed perfect-diode copy, reconstruct only
the supported compact screen summary from resolved configs, manifests, and
candidate `result.json` files. Do not run `merge` until both shards have valid
`select_final` completion markers.

Write generated summaries and plots under the ignored result bundle, ideally
under `_derived/<evaluation-id>/`. Put qualified measurements in
`result_registry/cards/`, reviewed interpretation in
`result_registry/reviews/`, and render `docs/results/`:

```bash
python labs/tools/render_result_registry.py --write
python labs/tools/render_result_registry.py --check --verify-artifacts
```

Do not manually edit generated result pages.

## 5. Resume the complete archive

Start this pass after the metadata review; it need not block initial
evaluation:

```bash
python skills/sync-remote-results/scripts/sync_remote_results.py \
  --host HOST \
  --remote-path /exact/remote/run-or-shard \
  --local-stage /absolute/RESULT_ROOT/.incoming/BUNDLE_ID/shards/HOST \
  --receipt /absolute/RESULT_ROOT/_transfer_receipts/BUNDLE_ID/HOST-full.json \
  --mode full \
  --completion-marker relative/path/to/complete.json
```

Full mode uses rsync's size-and-mtime quick check. Then run the local
experiment validator, which must verify the bundle's own manifest hashes. Add
`--full-checksum` only for a legacy bundle without trustworthy manifests; it
is deliberately slower.

For a crashed run, include `--allow-incomplete` and its last valid completion
marker here too. “Full” then means every file in the stopped remote snapshot
was copied; it does not mean the experiment reached scientific completion.

## 6. Publish without recopying

Publish a completed bundle only after the required full local validators pass.
An intentionally salvaged crash may be archived under an explicitly
incomplete diagnostic path after its available completion chains validate,
but it must never be relabeled as a completed study. Rename the same-filesystem
`.incoming` bundle into its canonical immutable path; do not copy or hash it
into a second local tree. Record the canonical path and receipt in
`docs/current_experiments.md` while the experiment is active.

Keep transfer state, scientific completion, and reviewed acceptance separate:

```text
remote closed -> transferring -> locally verified -> evaluated/reviewed
```

Never delete remote results in this workflow. Remote cleanup is a separate
explicitly authorized action after local publication.
