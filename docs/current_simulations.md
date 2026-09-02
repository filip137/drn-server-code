# Current LoRA/HWA Simulations

Manual sections last updated: 2026-09-02

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

### `mnist-ibm-om-baseline-spacing-pv-winsorized-nominal-exploratory-20260828-v1`

- **Status:** `running`
- **Active native runs:** `2`
- **Raw home:** `results/mnist-ibm-om-baseline-spacing-pv-winsorized-nominal-exploratory-20260828-v1/`
- **Runs:**
  - `mnist-ibm-om-baseline-spacing-pv-winsorized-nominal-exploratory-20260828-v1/runs/alpha_025_spacing_2delta-heldout-87003/20260828T173708.357122Z-12b21972-b548a052` — `validate`, `mnist_ibm_om_baseline_spacing_pv_truncated_nominal.v1`, arm `alpha_025_spacing_2delta-heldout-87003`, started `2026-08-28T17:37:08.357387+00:00`
  - `mnist-ibm-om-baseline-spacing-pv-winsorized-nominal-exploratory-20260828-v1/runs/alpha_025_spacing_4delta-heldout-87001/20260828T173708.549593Z-9c2ad4f2-ef125e3f` — `validate`, `mnist_ibm_om_baseline_spacing_pv_truncated_nominal.v1`, arm `alpha_025_spacing_4delta-heldout-87001`, started `2026-08-28T17:37:08.549872+00:00`

<!-- END AUTOMATIC ACTIVE SIMULATIONS -->

## Analyzing or paused

### Winsorized multi-assignment QAT and persistent pulse-mediated Adam

- **Question:** Can deterministic QAT improve transfer across Winsorized IBM
  OM assignments, and can pulse-mediated Adam recover one poor persistent
  deployment without remapping it?
- **Goal milestone:** `M7`
- **Status:** `analyzing`
- **Last updated:** 2026-08-29
- **Setup:** Ten-epoch deterministic QAT alternated assignments 86001/87001,
  selected spacing on 87002 at fixed epoch 10, and evaluated five persistent
  P&V streams on 87003. A follow-up cloned one newly programmed 87003 state
  across frozen, direct-rail, and coordinated-contrast recovery arms.
- **Raw homes:**
  `results/mnist-ibm-om-winsorized-multi-assignment-qat-exploratory-20260829-v1/`
  and
  `results/mnist-ibm-om-winsorized-onchip-adam-exploratory-20260829-v1/`
- **Latest observation:** Assignment 87002 selected `h=delta_x`. On 87003,
  deterministic QAT changed the five-seed P&V mean only from 68.998% to
  70.380% and reduced ideal accuracy from 94.88% to 94.47%. Starting from one
  separately programmed 59.20% persistent state, direct-rail pulse-mediated
  Adam at `3e-5` reached 94.14% after 54,834 pulses. This is one-state
  `exploratory_noncanonical` recovery, not a robust on-chip claim.
- **Next action:** Freeze one-delta, direct rail, `3e-5`, and one epoch, then
  replicate paired frozen/recovery arms from multiple predeclared persistent
  endpoints and assignments without retuning. Add inference-read and retention
  controls only after the paired recovery effect replicates.
- **Detailed report:**
  [`ibm_om_baseline_spacing_pv.md`](ibm_om_baseline_spacing_pv.md)

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

### `ibm-om-positive-g-population-hwa-cross-array` — IBM-style DRN HWA transfer

- **Question:** Does array-agnostic healthy population programming-error HWA
  improve transfer from a raw Array-A baseline to Arrays B--D, and do
  published corrupt devices remain a material independent penalty?
- **Status:** `running` — raw-active characterization prerequisite
- **Last updated:** 2026-09-02
- **Setup:** Raw ReLU-derived signed masters map to four physical
  conductances in `G=[0,2]`; Array A is written before HWA but is excluded
  from the HWA sampler; continuous and one-delta population-HWA arms are
  frozen before paired repaired/published-corrupt writes on B--D.
- **Intended raw home:**
  `results/exploratory_noncanonical-ibm-om-population-hwa-cross-array/`
- **Active prerequisite:** Local CUDA characterization under tmux session
  `ibm_om_raw_active_hwa_char`, writing
  `results/exploratory_noncanonical-ibm-om-raw-active-characterization/` and
  `results/exploratory_noncanonical-ibm-om-raw-active-characterization.launch.log`.
  It covers 1,024 healthy OM identities, four repeats, 41 targets, both
  one-pulse and adaptive controllers, and the frozen 128-pulse cap. The source
  config SHA-256 is
  `d0ea8df03a60de609414bcaa3558a9fdad5fc26bebf2aed13647a626c1571b35`;
  the numerical implementation is commit `627ea8ff`.
- **Monitoring:** Require the native `status.json`, growing trajectory/metric
  artifacts, plausible GPU activity, and a terminal `result.json`; inspect at
  least every 30 minutes. A failed attempt may be retried only into a fresh
  output directory without changing its seeds or scientific settings.
- **Next action:** When characterization completes, validate and hash its
  bounded endpoint and step-estimator artifacts, run the smoke A--D protocol,
  then launch the full exploratory comparison only if the smoke artifacts and
  ordering invariants pass.
- **Detailed protocol:**
  [`ibm_om_population_hwa_cross_array.md`](ibm_om_population_hwa_cross_array.md)

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
