---
id: "exp-018"
title: "Fresh learning at the larger block3 RMS target"
status: "complete"
hypotheses: ["H-014"]
---
# Use the checkpoint-supported block3 target

Completed and reviewed September30 05:53 Paris:32.88%, CE1.945375 improves
the same-ceiling comparator but misses both continuation gates. See the
[result](../results/exp-018-block3-rms-training.md); no longer stage is launched.

One fresh ours epoch runs alongside [exp-017](exp-017-current-beta-ceiling.md).
Change only block3 RMS target.001→.005 in `rms_targets` and
`beta_search.targets`. Keep targets1/2 at.02/.035 and beta bounds[1e-8,3e4].
Reuse its exact frozen numerical source, initialization, fresh Adam, seed0,
45k/5k split/order/augmentation, original rates, trainable BN/gains/head,
batch32/eval16, T/K[6,6,4],50-epoch schedule, starting beta vector and bounded
five-pair current-batch search. No noise, smoke, solver audit or control rerun.

The target comes from exp-008's cross-checkpoint recommendation; its training
benefit is untested. Compare directly with exp-017 for the target effect.
Relative to exp-015, both ceiling and target change, so do not attribute that
comparison to the target alone. The existing matched native control remains
37.06%, CE1.726060. The unchanged long-run gate is accuracy>=35.06% and
CE<=1.812363. Report RMS misses and beta/solver counts alongside full5,000-image
validation. Passing candidates may continue under existing staged rules and
remaining budget; failure returns to scientific review.

## Execution and monitoring handoff

Root: `results/cifar-beta-current-batch-block3-rms-20260930-v1/`.
Use now-free nom-cool-1, torch2.5.1+cu121/privatecuDNN9.1, preserving Ben's
workload. One5,400-second job; expect50–75minutes under measured sharing.
Fund1.5GPU-hours by reducing unused ours epoch50 allowances13.5h→13h across
at most three branches. The combined ours141.5GPU-hour ceiling is unchanged;
this is a transfer, not a new allowance. Deadline October2 08:00 Paris.
Use shared-queue ownership and the existing half-hour summary. Record accepted
ID, actual startup and measured sharing here. Official test data remain unread.


### Frozen preparation and accepted queue job

Accepted ID: `cifar-beta-fresh-current-batch-block3-rms-e1-20260930`.
Prepared job: `results/cifar-beta-current-batch-block3-rms-20260930-v1/launch/current_batch-1.json`.
Small configs: `configs/cifar/beta_current_batch_block3_rms_20260930/`.
The sole case remains `current_batch`. All404 numerical-source files and their
source identity are byte-identical to exp017. Initializer, config, split, cohort,
and study-runtime inputs are unchanged. Only block3 entries in `rms_targets`
and `beta_search.targets` change.001→.005; the ceiling remains30,000.
Source/input and all five training-batch hashes, unchanged CPU case validation,
and private-library hashes passed on nom-cool-1. The checked runtime is
torch2.5.1+cu121/CUDA12.1/cuDNN90100. Existing67-test validation of the unchanged
implementation is reused; no GPU smoke or new numerical code was introduced.
Preparation and preflight evidence are in `analysis/preparation.json` and
`analysis/preflight.json`.

Only nom-cool-1 is eligible, reserving16GiB plus2GiB headroom. The unchanged
legacy transport owner guard is frozen as `launch/ben_only_exec.py` and hashed
in prepared files. Immediately before the exact runner it resolves owners of
processes on the queue-selected GPU and allows only Ben/Filip. The shared queue
retains memory admission and operational ownership; its global sharing settings
were not changed. An owner-guard rejection is operational, not a beta failure.

The5400second cap is funded by nine unsubmitted ours stage50 specs changing
48,600→46,800seconds (13.5→13hours), across at most three promoted branches.
All earlier allocations remain charged inside the141.5GPU-hour ceiling.
The existing comparison-only collector verifies exact one-epoch state and beta
search history before recording measurement completion. Scientific gate review
is separate from that completion state. The shared30-minute summary includes
this root and removes only the now-inactive legacy3090 root; its artifacts remain
preserved. Existing V100 monitoring and all other study roots are unchanged.


### Verified remote startup (04:56 Paris)

Queue attempt `c8e44c6bedab486f85b3296840e5d835` is running on nom-cool-1,
PID443214, GPU `GPU-2262d56b-2c1d-1546-3e93-2e441399cd4f` (RTX3090).
The transport log confirms `TRANSPORT_OWNER_CHECK ... ['ben']`; the queue
receipt records seven observed Ben processes and no other sharing owner. The run
has an85,690,824-byte initial checkpoint and advancing initial validation.
Its manifest records the declared targets `[.02,.035,.005]` in both fields,
ceiling30,000, and matching torch2.5.1+cu121/CUDA12.1. Lightweight checkpoint,
manifest, progress and sharing proof is saved in `analysis/startup-proof.json`.
This establishes actual execution, not completion or learning-quality success.
The shared queue owns subsequent monitoring, recovery and validated collection.
