# Experiment runtime

## Design

The repository has three layers:

```text
experiments/small_network/  compose one versioned scientific workflow
             |
             +--> training/ generic execution, modifiers, probes, updates
             |
             +--> model/    energy functions, networks, parameter adapters
```

This keeps the useful shape of the original energy-based-learning repository:
the model and training packages describe reusable mechanics, while experiments
assemble them. Dataset choice, CLI paths, artifact layout, and paper-specific
choices do not belong in model classes or generic training loops.

`small_drn.v1` has four independent choices:

| Axis | Implementations | Responsibility |
| --- | --- | --- |
| training algorithm | `ep`, `backprop` | estimate a complete minibatch gradient |
| model adapter | `none`, `passive_low_rank` | define the trainable parameterization |
| parameter modifier | `none`, `add_normal` | temporarily alter parameters during solver phases |
| update backend | `direct`, `tiki_taka` | apply or accumulate the completed gradient |

LoRA is therefore not another name for Tiki-Taka. Passive low-rank recovery
changes which parameters produce the effective model weights; Tiki-Taka
changes how an already-computed gradient is accumulated and transferred. They
can share the same experiment, engine, checkpoint, and evaluation structure
without being coupled to each other.

Each worktree's experiment definition lists only combinations implemented in
that worktree. The shared foundation advertises base direct and Tiki-Taka
training; the HWA branch adds `add_normal` combinations, and the LoRA branch
adds `passive_low_rank` combinations. Unlisted combinations, including LoRA
plus hardware-aware perturbation, are rejected before numerical execution.

## Commands

Inspect the machine-readable protocol:

```bash
python -m ebl describe --experiment small_drn.v1 --json
```

Run one mode:

```bash
python -m ebl train \
  --config examples/small_drn/base.json \
  --output-dir runs

python -m ebl linspace \
  --config examples/small_drn/base.json \
  --weights runs/<training-run>/checkpoints/weights.pt \
  --output-dir runs

python -m ebl validate \
  --config examples/small_drn/base.json \
  --weights runs/<training-run>/checkpoints/weights.pt \
  --output-dir runs
```

The JSON config is strict, nested, and immutable after parsing. It records
scientific intent. Operational paths are explicit CLI arguments:

- `--weights` initializes/evaluates the complete named parameter set;
- `--base-weights` initializes only the base group, leaving an adapter
  independently initialized;
- `--resume` restores a full epoch-boundary state;
- `--output-dir` is the parent in which a new exclusive run directory is
  created.

These initialization options are mutually exclusive.

## Run contract

Each command creates one new run directory:

```text
<output-dir>/<run-id>/
  config.resolved.json
  manifest.json
  status.json
  metrics.jsonl
  result.json
  artifacts/
  checkpoints/
  logs/
```

`result.json` is the only campaign-facing completion contract. It records the
terminal status and content-hashed artifacts. A failed run retains its
manifest and failure status; it is not silently reused.

Named weights and full resume state are intentionally different:

- `weights.pt` contains stable parameter keys and is suitable for evaluation
  or transfer.
- `resume.pt` contains model, optimizer, scheduler, modifier, progress, random
  number generator, and named dataloader-generator state at an epoch boundary.

Restore is transactional. All payloads are validated before mutation, and any
failure rolls back already-restored components. AIHWKit-backed runs may
declare `stateful_nondeterministic`; an unsupported runtime must not emit a
resume checkpoint.

Old positional tensor lists are not auto-detected:

```bash
python -m ebl checkpoint import-legacy \
  --config examples/small_drn/base.json \
  --source legacy-model.pt \
  --kind full \
  --output named-weights.pt
```

Use `--kind base` only when the positional file contains base parameters and
an adapter must remain independently initialized.

## Cross-worktree campaigns

The campaign controller treats each Git worktree as a separate executable. It
first calls that worktree's `ebl describe ... --json`, records the exact
commit/dirty fingerprint, and then launches stages as subprocesses. This
prevents Python import-cache and module-path contamination between branches.

```bash
python -m ebl campaign run \
  --manifest path/to/campaign.json \
  --output-dir campaign-runs \
  --dry-run
```

A campaign manifest declares targets, stages, dependencies, configs, and
explicit artifact references. A later validation stage can consume the
`weights` artifact produced by a training stage without guessing a filename.
Use `--resume` to reuse a completed stage only when its config, inputs, source
identity, and result hash still match.

By default targets must be clean. `--allow-dirty` is intended for deliberate
local experiments and records a dirty-source fingerprint.

## Worktree development workflow

Develop cross-cutting runtime changes once on the shared foundation branch.
Then rebase the feature branches and implement only their extension:

```text
shared foundation
  +-- HWA branch: add_normal ParameterModifier
  +-- LoRA branch: passive_low_rank model adapter
```

Do not merge HWA and LoRA branches merely to obtain the common pipeline.
Compare them with a campaign. Promote an experimentally validated combination
to the explicit capability matrix only after its focused and numerical tests
pass.

## Adding an experiment or extension

For a new experiment:

1. Add `experiments/<name>/config.py` with a versioned immutable schema.
2. Register a stable ID explicitly in `experiments/definitions.py`.
3. Build numerical objects in the experiment package.
4. Reuse `training.engine`, probes, named checkpoints, and `RunStore`.
5. Add a small nested example and CLI smoke test.

For a new extension, prefer one protocol over a conditional in the engine.
Test its lifecycle/order separately, then add a numerical acceptance case for
each combination exposed by `describe --json`.
