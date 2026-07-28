# Remote Result Workflow

Use the repository skill
[`sync-remote-results`](../skills/sync-remote-results/SKILL.md) for results
produced on Akib, Trex, or Jean Zay. Avoid one-off `scp` commands.

The workflow has four distinct states:

```text
remote closed -> transferring -> locally verified -> evaluated/reviewed
```

A finished process is not yet a locally verified result, and a locally
verified result is not automatically scientifically accepted.

## Local layout

Choose one absolute local result root and keep using it for the workstream.
For the current Conv learning-rate worktree, that root is:

```text
/home/filip/server_code_conv_learning_rate_protocol/results
```

Do not derive a new result root from whatever worktree happens to be the
current directory. If this archive is deliberately migrated later, move it
once and update registry paths together.

Use this layout:

```text
results/
  .incoming/<bundle-id>/shards/<host>/  # resumable transfer
  _transfer_receipts/<bundle-id>/       # operational copy evidence
  <experiment namespace>/<bundle-id>/   # immutable published artifacts
    _derived/<evaluation-id>/           # local summaries and plots
```

The host is transfer provenance, not part of the scientific run identity.

## Record at launch

Put the following in the experiment tracker or launch receipt:

- stable run, study, sweep, or shard ID;
- host and exact absolute remote bundle path;
- job ID or tmux process identity;
- final completion marker expected from that job;
- intended absolute local staging and canonical paths;
- local validator and evaluator commands.

Typical SSH targets are `akib`, `filip@trex`, and `jean-zay`. Current result
roots commonly begin with:

```text
Akib:     /home/filiposana/results/...
Trex:     /home/filip/results/...
Jean Zay: /lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/...
```

Always use the exact path recorded by the launcher; these prefixes are not a
license to copy the shared root.

## Fast path after a run

1. Confirm that the writer stopped. On Jean Zay, also check terminal `sacct`
   state.
2. Pull a checksum-verified metadata snapshot into `.incoming` with
   `$sync-remote-results`.
3. Inspect or summarize only the local snapshot.
4. Start the resumable full transfer into the same staged bundle.
5. Run the experiment validator locally. Publish by same-filesystem rename
   only after it passes.
6. Register qualified measurements and the reviewed conclusion.

Metadata mode copies resolved configs, manifests, completion records, result
JSON, and compact tables. It excludes checkpoints and `step_log.csv`. In the
recent perfect-diode study, those two excluded classes accounted for about
98.9% of the payload; the broad metadata view is about 3.3 MiB instead of about
306 MiB. This is why evaluation can begin before the archive pass finishes.

For a crashed run, use `--allow-incomplete` only after the remote writer has
stopped, and also require the last valid `--completion-marker` when one
exists. Preserve that completion chain and explicitly record what is missing.
Do not fabricate a final selector or aggregate. A full copy of a crashed
snapshot is still an incomplete experiment.

## Local evaluation

Use the result contract to select the evaluator:

- canonical MNIST Conv sweep:
  `python -m experiments.mnist_conv collect --sweep LOCAL_SWEEP`;
- complete perfect-diode shard:
  `python experiments/run_mnist_conv_perfectdiode_hparam.py status --shard LOCAL_SHARD`;
- registry consistency:
  `python labs/tools/render_result_registry.py --check --verify-artifacts`.

The perfect-diode `status` command verifies all manifest-bound artifacts and
therefore belongs after the full transfer. A metadata-only snapshot supports
the compact screen summary but not the claim that the entire shard is locally
verified.

Store the layers separately:

- `results/`: physical configs, metrics, checkpoints, logs, receipts, and
  derived analysis;
- `result_registry/cards/`: tracked qualified measurements;
- `result_registry/reviews/`: tracked conclusion, decision, and limitations;
- `docs/results/`: generated browsing view;
- `docs/current_experiments.md`: temporary transfer and validation state;
- protocol documents: intended scientific rules, never daily run notes.

Never delete the remote copy as part of transfer or evaluation. Cleanup is a
separate, explicitly authorized operation after local publication.
