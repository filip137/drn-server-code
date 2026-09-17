# Current LoRA/HWA Simulations

Manual sections last updated: 2026-08-26

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
- **Research relevance:** Independent-base replication and untouched-test
  generalization.
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

## Ready for scientific review

### `ibm-om-raw-active-common-cell-9-program-verify-20260826-v1` — Are nine common physical cell states distinguishable?

- **Question:** Can one array-wide baseline and increment provide nine
  programmable single-`a_i` IBM OM cell states on development and held-out
  assignments?
- **Research relevance:** Establishes physical cell-level P&V viability before
  declaring a new logical mapper or repeating QAT.
- **Status:** `ready_for_review`
- **Last updated:** 2026-08-26
- **Setup:** Exact support audit over 158,800 cells per assignment; 1,024
  selected identities, four independent repeats, nine global targets,
  lower-bound conditioning, one-pulse controller, `tau=0.00791586015294392`,
  and 128 target pulses. Assignment 84001 derives the codebook and assignment
  85001 is held out.
- **Launcher contract:** First run the separately prepared eight-identity
  canary through `python -m ebl characterize`. After it completes with valid
  population, trajectory, integrity, analysis, and plot artifacts, run the
  two production arms locally and serially. Each target appends a semantic
  progress metric; treat a missing process, nonzero exit, incomplete
  trajectory coverage, or 45 minutes without target/artifact growth as an
  operational failure.
- **Raw home:**
  `results/ibm-om-raw-active-common-cell-9-program-verify-20260826-v1/`
- **Latest observation:** Both production arms are complete and pass the
  full-artifact audit. All-nine physical support is 90.000% on development
  and 90.161% on held-out, but validation persistent nearest-code correctness
  after apparent acceptance is only 28.623% and 28.500%, respectively. No
  validation identity/repeat programmed all nine levels nearest-code
  correctly. Thus common-range support succeeds while nine-level P&V
  distinguishability under the frozen tolerance and controller does not.
- **Safe retry:** Retain failed attempts and retry only an operationally
  failed declared config with identical source, seeds, codebook, tolerance,
  and pulse budgets.
- **Next action:** Record the human scientific interpretation and smallest
  useful next test, then finalize through the study workflow.

## Queued next

### `mnist-om-cell-bounds-uniform-fp32-sgd-expanded-lr-20260826-v1` — Was the original SGD grid simply too narrow?

- **Question:** How much of the observed Adam advantage remains after a
  geometric layerwise SGD search that extends beyond both boundaries of the
  original grid?
- **Research relevance:** Separates optimizer-specific benefit from an
  under-tuned fixed SGD learning rate before choosing the clean ceiling for
  physical pulse-training comparisons.
- **Status:** `running`
- **Started:** 2026-08-26
- **Last updated:** 2026-08-26
- **Setup:** Twelve zero-momentum SGD arms forming
  `W1 in {0.512,1.024,2.048,4.096}` by `W2 in {0.02,0.04,0.08}`, followed by
  one validation-selected full-test run. All arms retain the original
  seed-84001 population, seed-17 initialization and data order, ten epochs,
  exact per-cell FP32 bounds, and initial-weight hash
  `d8b66130...95f26e`.
- **Launcher contract:** Run the highest-rate formal arm first as the numerical
  canary, then use local serial CUDA execution in tmux session
  `om_bounds_sgd_ext_20260826`. Every native arm writes `status.json`,
  `metrics.jsonl`, `result.json`, a resume checkpoint, and a bounds report;
  epoch metrics should advance about once every 20 seconds. Poll artifact
  coverage at least every five minutes and treat 45 minutes without growth as
  stalled.
- **Raw home:**
  `results/mnist-om-cell-bounds-uniform-fp32-sgd-expanded-lr-20260826-v1/`
- **Safe retry:** Retry only an operationally failed declared arm with its
  identical frozen config, population input, and source identity. Retain the
  failed attempt; do not change rates or the ten-epoch budget.
- **Latest observation:** All thirteen declared runs are complete and
  artifact-verified. The cost-selected `[4.096,0.08]` checkpoint reached
  96.04% validation and 96.48% full-test accuracy, improving the original SGD
  result by +1.22/+1.15 percentage points and leaving a 0.72/0.46-point gap to
  Adam on validation/test. The winner lies on both upper grid boundaries, so
  this does not yet establish a fully tuned SGD ceiling.
- **Next action:** Obtain the human scientific interpretation and decide
  whether to run one more narrow upper-bound SGD extension before closing the
  optimizer comparison.

### `mnist-om-cell-bounds-uniform-fp32-optimizer-reference-20260826-v1` — What can the sampled OM cell ranges learn ideally from scratch?

- **Question:** What ten-epoch accuracy is attainable when SGD or Adam uses
  exact FP32 arithmetic but every conductance is initialized and projected
  inside its own seed-84001 IBM OM interval?
- **Research relevance:** Establishes an array-specific bounds-only ceiling
  before pulse-update limitations are attributed to an on-chip optimizer.
- **Status:** `analyzing`
- **Started:** 2026-08-26
- **Last updated:** 2026-08-26
- **Setup:** Six SGD and four Adam development arms, followed by one frozen
  full-test checkpoint per optimizer; fixed 158,800-cell population SHA-256
  `7e0bcfb7...ad49d2f`, identical initialization hash
  `d8b66130...95f26e`, ten epochs, seed 17.
- **Launcher contract:** Local serial CUDA execution in the predeclared tmux
  session `om_cell_bounds_ref_20260826`; every native run writes its own
  `status.json`, `metrics.jsonl`, `result.json`, resume checkpoint, and bounds
  report below the arm directory. Metrics should advance once per epoch; poll
  artifacts at least every five minutes and treat 45 minutes without growth
  as stalled.
- **Raw home:**
  `results/mnist-om-cell-bounds-uniform-fp32-optimizer-reference-20260826-v1/`
- **Latest observation:** All twelve declared runs are complete and
  artifact-verified. Validation-selected SGD reached 94.82% validation and
  95.33% test accuracy; validation-selected Adam reached 96.76% validation
  and 96.94% test accuracy. These are bounds-only FP32 measurements, not
  physical pulse-training results.
- **Safe retry:** Retry only an operationally failed declared arm with the
  identical config, population input, and source revision; retain the failed
  attempt and do not alter rates or the ten-epoch budget.
- **Next action:** Obtain the human scientific interpretation and next-test
  decision, then finalize the workflow-managed study.

### `mnist-ibm-om-reset-relative-on-chip-recovery-pilot-20260825-v1` — Which pulse-compatible recovery should follow RESET-relative QAT?

- **Question:** Can direct single-pulse or Tiki-Taka updates improve one exact
  deployed array under matched slow-write caps?
- **Research relevance:** Tests the current on-chip-training motivation without
  conflating recovery with a fresh mapping or deployment.
- **Status:** `queued`
- **Intended setup:** One source deployment; rail refresh, direct pulses,
  ideal-fast Tiki-Taka, and physical-fast Tiki-Taka at 0.1, 1, and 4 slow
  pulses per cell; one development seed only.
- **Intended raw home:**
  `results/mnist-ibm-om-reset-relative-on-chip-recovery-pilot-20260825-v1/`
- **Next action:** Pass the pre-submit tests, prepare the study, create the
  source deployment, and launch the twelve fixed-terminal arms.

### `mnist-post-hwa-adaptation-locality` — Is full rewriting necessary?

- **Question:** Can targeted W2-plus-bias fine-tuning close most of the gap
  between frozen-base rank-4 LoRA and ideal full-model fine-tuning?
- **Research relevance:** Whether targeted recovery can avoid a full deployed
  base rewrite.
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
- **Research relevance:** <one sentence linking the work to current_state.md>
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
- [Experimental manifest](experimental_manifest.md)
- [Raw result home](../results/README.md)
- [Experiment workflow](experiment_workflow.md)
