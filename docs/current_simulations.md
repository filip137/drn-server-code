# Current LoRA/HWA Simulations

Manual sections last updated: 2026-07-29

## Purpose

This is the short ledger for LoRA/HWA studies that are running, being
analyzed, paused, or likely to start next. The `Active` section is generated
from native `status.json` files under `results/`; the other sections remain
human-maintained.

Statuses (`queued`, `running`, `analyzing`, and `paused`) are descriptive
only. This file does not approve, validate, launch, or block simulations.

<!-- BEGIN AUTOMATIC ACTIVE SIMULATIONS -->
## Active

No simulations currently running.

<!-- END AUTOMATIC ACTIVE SIMULATIONS -->

## Analyzing or paused

### `mnist-multibase-lora-replication` — Multi-base MNIST replication

- **Question:** Does the positive MNIST HWA-to-LoRA recovery persist across
  independently trained base models and an untouched final test split?
- **Goal milestone:** `M5`
- **Status:** `paused`
- **Last updated:** 2026-07-29
- **Intended raw home:** `results/mnist-multibase-lora-replication/`
- **Reason paused:** The current priority has shifted from seed/rank sweeps to
  determining when post-HWA base-model fine-tuning is needed.
- **Next action:** Resume only when base-seed generalization becomes the
  active research question.

## Raw-attempt note

Ten superseded attempts inside the finished MNIST study still contain native
`status.json` files that say `running`, but no matching process is active.
They are interrupted raw attempts, not active studies; this ledger tracks the
study-level state.

## Queued next

### `mnist-post-hwa-adaptation-locality` — Is full rewriting necessary?

- **Question:** Can targeted W2-plus-bias fine-tuning close most of the gap
  between frozen-base rank-4 LoRA and ideal full-model fine-tuning?
- **Goal milestone:** `M4` — adaptation accuracy versus physical overhead
- **Status:** `queued`
- **Intended setup:** Reuse the original five MNIST HWA-to-ReRAM checkpoints
  and compare the existing rank-4 and full-model arms with a 20-epoch BPTT
  arm that updates only W2 and the hidden bias.
- **Intended raw home:** `results/mnist-post-hwa-adaptation-locality/`
- **Next action:** Add an explicit selective-base parameterization or update
  boundary that freezes W1, then cover it with lifecycle and numerical
  acceptance tests.

## Entry template

<!--
### `<study-id>` — <title>

- **Question:** <one sentence>
- **Goal milestone:** <milestone from the finished-simulation ledger>
- **Status:** `queued`, `running`, `analyzing`, or `paused`
- **Started:** YYYY-MM-DD
- **Last updated:** YYYY-MM-DD
- **Setup:** <essential arms, seeds, or variation>
- **Raw home:** `results/<study-id>/`
- **Latest observation:** <provisional observation; not a final conclusion>
- **Next action:** <one concrete action>
-->

## Related documents

- [Current research state and notes](current_state.md)
- [Finished LoRA/HWA simulations and progress](experimental_manifest.md)
- [Raw result home](../results/README.md)
