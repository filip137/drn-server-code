# September 6 recovery source archive

This directory preserves the exploratory recovery comparison drivers, plans,
monitoring notes, readout tools, and later HWA analysis scripts that originally
lived in ignored `.codex/worktrees` directories. Original script and protocol
bytes are retained; `source_manifest.json` records their origins and SHA-256
hashes. Raw results, model checkpoints, datasets, and collected external-document
snapshots are stored separately and are not included here.

The source dependencies are pinned to:

- DRN: `2b283365063931e9fdd41fbd30d2672e0e600f04`.
- Original crossbar: `207a8f6b25105b6ceaee85f00bf5f5826b9821b2`.
- Scheduled crossbar: `9cc1496dfb10b68702f66dbbbf7f6199c44d4711`.
- Scheduled readout: the scheduled crossbar plus the two exact files under
  `source_overlays/readout/`. These add the historical `recovery-readout` CLI
  without altering the scheduled numerical runtime.

The scheduled commit also preserves the original 37 schedule/confirmation
configs, study definitions, and schedule tests. Its numerical files match the
frozen September 6 exports byte for byte. Two inherited regression-test files
were updated to recognize the extended learning-rate grid, epoch range, and
schedule metadata.

From the repository root, verify the archive and reconstruct its ignored source
dependencies with Python 3.12 or newer:

```bash
python campaigns/recovery_20260906/prepare_sources.py --check
python campaigns/recovery_20260906/prepare_sources.py
```

`--only <relative-export-path>` can be repeated to restore just the dependencies
needed for a particular driver. Each export is approximately 210 MiB. Existing
destinations are never overwritten; an interrupted export remains available for
inspection. These are source exports, not Git worktrees. Their authority is the
commit and overlay information in `RECOVERY_SOURCE.json`, not a parent Git
checkout discovered by `git rev-parse`.

To check the controllers independently, use their original working directories
and real CUDA. The separate processes avoid collisions between the historical
modules that share names:

```bash
cd campaigns/recovery_20260906/closed-loop-recovery-20260906
python -m pytest --confcutdir=. -p no:cacheprovider -q test_closed_loop.py
cd ../closed-loop-crossbar-schedules-20260906
python -m pytest --confcutdir=. -p no:cacheprovider -q test_closed_loop.py test_schedule.py
```

Archive-wide test discovery excludes these isolated scripts and reconstructed
dependencies. The commands above run the intended controller tests explicitly.
The root `tests/test_long_recovery_schedule.py` checks the integrated schedule
schema, physical pulse scaling, and checkpoint replay.

Replaying training or readout additionally requires the named external inputs:

- The original teacher weights, healthy/faulted P0 states, saved DRN deployments,
  prepared states, and matched OM populations named in the plans and receipts.
- The original MNIST dataset and declared split/order, with `EBL_MNIST_ROOT`
  configured for the execution environment.
- For later analysis/HWA tools, the explicitly named analysis receipts and
  intermediate input populations referenced by those tools.

The archived plans retain their original absolute paths, including some `/tmp`
inputs. On another machine, create a separate adapted plan with explicit new
input paths, verify the original content hashes, and supply it with the driver's
`--plan` option. Preserve teacher path bindings inside prepared-state inputs as
well; the archived remote DRN wrapper provides an explicit teacher override.
Do not rewrite the archived plans or rerun their historical `prepare.py` merely
to discover newer checkpoints. Launchers and remote shell scripts retain their
original machine paths and must be configured explicitly before use elsewhere.

This archive preserves exploratory source and provenance. It does not promote
these results to finalized, workflow-managed evidence or claim a new training
replay. No training campaign is launched by the source restoration utility.
