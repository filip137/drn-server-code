# Digital ReLU references for Tables 1 and 2

Frozen 2026-09-17 for the selection-stage tables in
`papers/amplification_overleaf/bidir_paper_theory_revised.tex`.

Requested: ordinary digital ReLU networks with half the DRN channels, MSE
and cross-entropy, and bias learning rate zero. The digital reference is
unconstrained and reused across Table 2's conductance ceilings; applying a
conductance-to-signed-weight mapping would be a separate scientific choice.

The explicit [config](../configs/digital_relu_table12_20260917.json) defines
18 trainings: Conv1/2/3 × MSE/cross-entropy × seeds 0/1/2. Channels are
32, 32/64, 32/64/128, strides 2, 2/2, 2/2/1, kernel 3, padding 1, no pooling.
One signed input channel and ten raw output logits replace DRN sign-paired
inputs and outputs. All biases, including the classifier bias, start at zero
and have explicit learning rate zero. Weights use PyTorch's default signed
Kaiming uniform initialization (`a=sqrt(5)`). MSE is mean squared error against
one-hot 0/1 targets, averaged over examples and classes. Cross-entropy takes
raw logits. The two losses start from identical tensors and training orders
for each depth/seed.

Match the displayed DRN budgets: 10/30/30 epochs, batch size 16, Adam
(betas .9/.999, epsilon 1e-8, zero weight decay). The weight learning rate
is fixed at .001, the existing digital trainer default, for both losses;
no LR search or claim of optimal tuning. Bias LR is exactly zero.
Use float32, deterministic operations, no TF32, no schedule or augmentation.
Input is `.3 * (pixel/255 - .1307)/.3081`, without DRN sign duplication or
architecture-specific analog input gain. These model/loss/input differences
must be disclosed, not described as an isolated amplification ablation.

Use the repository's stratified split seed 0 (55,000/5,000) and exact
per-seed minibatch-order function. The official test resources are never
loaded or checked. Select the maximum-validation-accuracy checkpoint,
earliest epoch on ties; report mean ± sample SD over three seeds. These
are validation results for the present draft, not final official-test evidence.

Run `python -m experiments.train_digital_relu_reference` with the config,
dataset root, output root and target. Same-path local smoke covers all six
depth/loss cases using one train and validation batch each, seed 0. Production
retains canonical reporting bundles, initial/best/final checkpoints, final Adam
state, raw training-file hashes, split and per-epoch batch hashes, source files,
Git identity, environment, and exact command. Require all 18 successes, full
budget, finite parameters/metrics, exact-zero biases and matched loss-pair
initialization/order before aggregation. Preserve failed attempts and do not
select results by outcome.

Initial placement: distribute complete depth surfaces across idle Akib/local/
Trex GPUs, finalized after smoke timing. Keep both losses and all seeds of a
depth on one host. The inventory check found local, Akib, Nom-cool-1, Trex,
Loulou idle; Fifi occupied; Riri host-key verification failed; Jean Zay access
timed out. No unrelated jobs are changed.

Results: `results/digital-relu-table12-20260917-v1/`, including smokes and
remote collections. Budget: four total GPU-hours and four hours wall time;
initial expected duration 30–120 minutes, refined using timing. Every worker
writes progress at least every 500 batches and each epoch. Check immediately
after launch and at most every 30 minutes. Stop at the time budget, record
partial coverage, and retain all artifacts if incomplete. Recover only understood
operational failures without changing science. Validate authoritative local
copies before updating tables and the experiment manifest.

## Launch placement and smoke

Conv1: local RTX3090, `py309` (torch2.5.1+cu121). Conv2: Akib RTX3080,
`py312` (torch2.5.1). Conv3: Trex RTX5090, `py312` (torch2.11.0+cu128).
All six runs per depth stay together. Cross-depth environment differences
are recorded; within-depth MSE/CE comparisons share environment and GPU.
Each lane is limited to 4,800 seconds, keeping production under 4 GPU-hours
combined; expected 15–45 minutes with three concurrent depth lanes; observed first-epoch timings later refined this to about 12 minutes.

The initial local py312 smoke failed at its first Conv1 forward call with
`CUDNN_STATUS_NOT_INITIALIZED`, before an optimizer step. Preserve its
failed bundle in `smoke/conv1_mse_seed0`. Local py309 successfully ran all six
one-train-batch/one-validation-batch cases, with artifact validation and
exact-zero bias checks, under `local/smoke/`. This is an environment-only
recovery; scientific config and source are unchanged. The CPU regression
suite passed 7 checks for geometry, updates, frozen biases and loss definitions.

Local smoke command:
```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/filip/miniconda3/envs/py309/bin/python -m experiments.train_digital_relu_reference \
  --config configs/digital_relu_table12_20260917.json \
  --output-root results/digital-relu-table12-20260917-v1/local \
  --dataset-root /home/filip/datasets/mnist --target local --device cuda --smoke
```

A frozen minimal source copy resides at `source/` in the study root, staged
unchanged into the matching remote study root. Remote output roots are
`/home/filiposana/server_code/results/digital-relu-table12-20260917-v1/akib`
and `/home/filip/server_code/results/digital-relu-table12-20260917-v1/trex`.
Remote same-path smokes precede their lane production. Exact commands,
launcher logs, PID and exit-code records live under each lane's `launcher/`.

Live handles: local tmux session `digital-relu-conv1`; Akib nohup launcher
PID 243696; Trex tmux launcher PID 261501. Launch date 2026-09-17. Each lane
uses the stated command plus `--depths 1`, `2` or `3`, respectively, and
`--max-seconds 4800`. Launcher directories retain the exact command and times.

At 14:17 UTC a read-only resource check found an unrelated Trex workload
(`binary_main.py`, PID261823), started after our idle-lane allocation. It uses
about 9 GiB beside our under 1 GiB worker. No unrelated process was changed;
our Conv3 lane remains on its frozen host and configuration. Epoch times rose
from about 3.2 s to 6–8 s under contention; retain this operational fact when
interpreting runtime. Accuracy comparisons are within the same target.

## Completion

All 18/18 full trainings completed and validate; every best-checkpoint validation
accuracy reproduces exactly. Both remote trees match local copies by checksum.
All launchers exited zero. Production cost: 0.497500 aggregate GPU-hours.
Ten smokes passed; the initial failed local cuDNN smoke remains preserved.
No full training failed, was excluded or retried. Official-test reads remain zero.
The [final report](../paper_ready_results/digital_relu_table12_20260917.md)
contains all six three-seed aggregates. Manuscript Tables 1 and 2 include only
the three MSE aggregates; CE results remain supporting evidence.

## Current manuscript inclusion

Filip selected the digital MSE reference for **Table 1 only**. The latest
Overleaf revision `c5ec8e3` was fetched, the Table 1 column and supporting
paragraph applied in an isolated checkout, and commit `8a889d7` pushed
to `origin/main`. Table 2 and Table 3 are unchanged from that remote base.
All 18 scientific run bundles remain preserved; CE is supporting evidence.
The corresponding local manuscript files now match the pushed revision;
prior local versions are preserved under `manuscript_before_overleaf_refresh/`
in the study directory. Unrelated local edits were preserved.
