# Current LoRA/HWA Simulations

Manual sections last updated: 2026-07-29

## Purpose

This is the short ledger for LoRA/HWA studies that are running, being
analyzed, paused, or likely to start next. The `Active` section is generated
from native `status.json` files under `results/`; the other sections remain
human-maintained.

Statuses (`queued`, `running`, `analyzing`, and `paused`) are descriptive
only. This file does not approve, validate, launch, or block simulations.
Prepared workflow studies retain their hypothesis in
`results/<study-id>/study.json`; exact coverage and review readiness come from
`python -m ebl study summarize`, not from this live view.

<!-- BEGIN AUTOMATIC ACTIVE SIMULATIONS -->
## Active

### `measured-cohort-b-lora-rank4-reset-20260815-v1`

- **Status:** `running`
- **Active native runs:** `1`
- **Raw home:** `results/measured-cohort-b-lora-rank4-reset-20260815-v1/`
- **Runs:**
  - `measured-cohort-b-lora-rank4-reset-20260815-v1/20260815T171027.734599Z-89c7d36c-7c5ad177` — `train`, `small_drn.v1`, started `2026-08-15T17:10:27.734755+00:00`

### `mnist-ibm-om-differential-pair-hwa-program-verify-pilot-20260824-v1`

- **Status:** `running`
- **Active native runs:** `1`
- **Raw home:** `results/mnist-ibm-om-differential-pair-hwa-program-verify-pilot-20260824-v1/`
- **Runs:**
  - `mnist-ibm-om-differential-pair-hwa-program-verify-pilot-20260824-v1/runs/train-exact-bounds-pair-hwa/20260824T120938.532361Z-d9f199ff-39777998` — `train`, `mnist_relu_drn_kd.v1`, arm `train-exact-bounds-pair-hwa`, started `2026-08-24T12:09:38.533493+00:00`

### `mnist-relu-drn-kd-exploratory-20260816`

- **Status:** `running`
- **Active native runs:** `2`
- **Raw home:** `results/mnist-relu-drn-kd-exploratory-20260816/`
- **Runs:**
  - `mnist-relu-drn-kd-exploratory-20260816/teacher_lr3e4/20260816T132511.786114Z-224bcaf4-ba2076eb` — `train`, `mnist_relu.v1`, started `2026-08-16T13:25:11.786279+00:00`
  - `mnist-relu-drn-kd-exploratory-20260816/teacher_lr5e4/20260816T132511.786248Z-e8c48023-c4551d53` — `train`, `mnist_relu.v1`, started `2026-08-16T13:25:11.786374+00:00`

### `mnist-relu-drn-reset-factorial-20260817-v1`

- **Status:** `running`
- **Active native runs:** `1`
- **Raw home:** `results/mnist-relu-drn-reset-factorial-20260817-v1/`
- **Runs:**
  - `mnist-relu-drn-reset-factorial-20260817-v1/stages/bias_free_logical_mse/local/bias_free_logical_mse_train/attempt-002/runs/20260817T052654.046087Z-9aae840c-4f019e33` — `train`, `mnist_relu_drn_reset_factorial.v1`, started `2026-08-17T05:26:54.046273+00:00`

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
- [Experiment workflow](experiment_workflow.md)
