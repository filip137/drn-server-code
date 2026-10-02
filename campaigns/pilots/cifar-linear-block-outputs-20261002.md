# Digital L8 with linear block outputs

## Question and frozen contract

Filip proposed removing the output nonlinearity from each L8 block because
nudging/noise is applied there (October 2, 2026). This belongs to the
[CIFAR beta-selection campaign](../cifar10-beta-selection/README.md), motivated
by [inactive-channel forces](../cifar10-beta-selection/explorations/X-002-inactive-channel-force.md).
First measure feed-forward performance; analog changes and noise experiments
are deferred. Filip selected the new variant alone, zero weight decay, and the
local RTX 3090.

Eight convolutions, depths [3,3,2], widths 128/256/512; replace only each block's
last ReLU with Identity. Keep five internal ReLUs, max-pooling, affine BN and
the digital classifier. Parameter count 5,395,594. Float32, fresh seed 0,
PyTorch default initialization, deterministic kernels, TF32 disabled. Adam
LR 0.001, betas 0.9/0.999, epsilon 1e-8, **weight decay 0 for all parameters**;
50-epoch cosine to 0.00002. Batch 512 train/validation; 50 epochs, 4400 updates.
Retain the original digital normalization and crop/flip augmentation. Explicit
diode dictionaries remain empty because this is a digital ReLU model.

Use only the five CIFAR10 training files: deterministic stratified 45k/5k split,
seed 0, 500 validation images per class. Save indices/hashes; never open the
official test files. Primary measurements: final validation accuracy and CE;
also best validation accuracy/CE, epoch, runtime and peak memory. Best checkpoint
uses maximum validation accuracy, then minimum loss, then earliest exact tie.
Save initial/best/final checkpoints and per-epoch curves. No accuracy threshold
or causal improvement claim is predeclared for this descriptive single-arm pilot.
The older 92.62% digital test result differs in activation, split and weight decay.

Config: [digital_l8_linear_outputs_20261002.json](../../configs/cifar/digital_l8_linear_outputs_20261002.json).
Runner: `python -m experiments.train_cifar_digital_l12 CONFIG --output-root ROOT
--dataset-root /home/filip/datasets/cifar10 --source-identity SOURCE --target local`.

## Budget and evidence

Only the local RTX 3090, through the shared run-watch queue after local smoke.
One-hour total GPU execution allowance including smoke/retries; reserve measured
peak memory plus headroom. Initial estimate 12–25 min, to be updated from local
throughput. Preserve all attempts. Nonfinite values, source mismatch or exhausted
budget stop the run. No retuning or extra seeds. A retry must preserve the science
and fit the remaining allowance.

All outputs, including source snapshot, smoke, launcher records and analysis:
[results/cifar10-digital-l8-linear-outputs-20261002-v1/](../../results/cifar10-digital-l8-linear-outputs-20261002-v1/).
Local collection is inherent; run `python -m experiments.reporting validate-run RUN`
for smoke and production, then verify contiguous 50 epochs/4400 updates, complete
45k/5k coverage per epoch and finite checkpoints. Plots use matplotlib JPG.

## Execution and monitoring handoff

State: complete and reviewed (October 2). One smoke and one production run,
no retries. Queue monitoring acknowledged the local RTX 3090 worker, PID 2973688,
attempt `5c62d1c3d1274282a5860e2f323a4b85`. The worker and bundle validator exited
successfully; the CPU collector validated coverage/checkpoints and produced the
summary and JPG. Eleven focused tests pass (eight model/data/compatibility tests,
three collector tests); campaign ledger check passes. The launched source snapshot
contains 16 files, and the trainer/config still match it.
Smoke: three updates and one validation batch, finite nonzero gradients in every
parameter tensor, valid bundle, 4.12 s wall time and 10,750 MiB peak reserved memory.
Production received a 3,595 s queue allowance and 3,565 s trainer cap.
Actual queue-charged execution was 847.30 s; smoke plus production used 851.42 s
of the one-hour total allowance. Training itself took 844.30 s.
Reservation: 10750MiB peak bound plus 1024 MiB headroom.

Queue ID: `cifar-digital-l8-linear-outputs-20261002`; local RTX 3090 only. Spec/command:
[launch/job.json](../../results/cifar10-digital-l8-linear-outputs-20261002-v1/launch/job.json).
Frozen source/config identity:
[source/source_identity.json](../../results/cifar10-digital-l8-linear-outputs-20261002-v1/source/source_identity.json).
Runtime: existing PyTorch 2.11.0+cu128 / torchvision 0.26.0+cu128 environment.
The queue owned production monitoring at 60 s cadence, with a 5 min stall limit.
Root completed scientific review; no recovery or further monitoring is required. Completion hook validates coverage/checkpoints and writes
`analysis/summary.json` and `analysis/learning_curves.jpg` on CPU. Inspect with
`python /home/filip/.codex/skills/run-watch/scripts/gpu_queue.py status cifar-digital-l8-linear-outputs-20261002`.
Terminal launcher receipts and full training log are copied into `ROOT/launch/`.
The source, smoke and production bundles, analysis and all launcher records stay
under the single result root above.

Exact production command (working directory `ROOT/source`):

```bash
/home/filip/server_code_conv_learning_rate_protocol/.codex/worktrees/lean-experiment-workflow/results/cifar-layer-rms-all-schemes-float64-20260930-v1/runtime/py312/bin/python -u -m experiments.train_cifar_digital_l12 configs/cifar/digital_l8_linear_outputs_20261002.json --output-root /home/filip/server_code_conv_learning_rate_protocol/.codex/worktrees/lean-experiment-workflow/results/cifar10-digital-l8-linear-outputs-20261002-v1 --dataset-root /home/filip/datasets/cifar10 --source-identity /home/filip/server_code_conv_learning_rate_protocol/.codex/worktrees/lean-experiment-workflow/results/cifar10-digital-l8-linear-outputs-20261002-v1/source/source_identity.json --target local --maximum-runtime-seconds 3565
```

## Results and review

Both smoke and production bundles validate locally. Production covers all 50
ordered epochs and 4,400 optimizer steps, 45,000 training and 5,000 validation
examples each epoch. Initial/best/final checkpoint epochs are 0/48/50; their
model and optimizer states are finite. Every optimizer group has weight decay 0.
The official test set was never opened. No exclusions, failed runs or retries.

| Measurement | Result |
|---|---:|
| Best validation accuracy | **91.36%**, epoch 48 |
| Validation cross-entropy at selected checkpoint | 0.398385 |
| Final validation accuracy | **91.34%**, epoch 50 |
| Final validation cross-entropy | 0.400233 |
| Final augmented-training accuracy | 99.6822% |
| Training runtime | 844.30 s (14 min 4 s) |
| Mean epoch duration | 16.806 s |
| Peak reserved GPU memory | 10.498 GiB |

[Learning curves (JPG)](../../results/cifar10-digital-l8-linear-outputs-20261002-v1/analysis/learning_curves.jpg)
· [Validated summary](../../results/cifar10-digital-l8-linear-outputs-20261002-v1/analysis/summary.json)
· [Implementation checks](../../results/cifar10-digital-l8-linear-outputs-20261002-v1/analysis/implementation_checks.json).
The source collector is `experiments/analyze_cifar_digital.py`; a copy is retained
in the study's `analysis/` directory.

**Interpretation:** this digital L8 with linear block outputs trains successfully
and reaches 91.36% validation accuracy with no weight decay. Accuracy stabilizes
near 91% late in training while augmented-training accuracy approaches 100%.
This establishes feasibility for this seed and recipe, with an evident train–
validation gap. It does not establish accuracy parity or improvement over original
L8: there is no matched control, and the historical 92.62% result used the official
test split, 50k training examples and nonzero weight decay. The proposed advantage
for analog noise/nudging remains **not tested** by this purely digital run.

**Decision:** the requested digital pilot is complete. Retain the explicit
linear-output option and its config/checkpoints as a candidate for subsequent
analog experiments. Analog implementation and noise testing remain deferred;
no additional runs have been launched.
