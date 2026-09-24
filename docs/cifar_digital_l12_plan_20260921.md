# CIFAR-10 fully digital L12 logical-width baseline

User request, 2026-09-21: train the feed-forward digital BP network on
akibscomputer for 50 epochs. User clarified **match logical width**: halve
the analog physical channel counts, keeping logical widths 128/256/512.

- One exploratory seed-0 run, 50 epochs, CIFAR-10 50,000 train / 10,000 test.
- Twelve bias-free 3x3 stride-1 padded convolutions with ReLU after each:
  `[128,128] / [128,128] / [256,256,256] / [256,512,512] / [512,512]`.
- Digital bridges match the all-analog graph: non-affine BatchNorm after
  blocks 1--4, preceded by 2x2 MaxPool after blocks 2--4. Flatten 512x4x4,
  then a learned digital Linear(8192,10), with bias. No amplification,
  voltage duplication, conductance projection, learned boundary gains or T/K.
- Adam LR .001, betas .9/.999, eps 1e-8, weight decay .0003; cosine LR over
  50 epochs ending at .00002. Frozen batch size **512** for training and test.
  User explicitly permits larger batches. Target smokes passed at 256 and
  512: peak reserved memory 2.58 and 6.68 GiB respectively on the 10 GiB GPU;
  three train batches plus one test batch took .82 and .94 seconds. Batch 512
  has higher measured example throughput and over 3 GiB nominal headroom.
  Float32, TF32 disabled; no LR scaling is applied for the larger batch.
- Same flip/crop/normalization as the historical CIFAR hybrid/all-analog runs.
  Official test evaluated each epoch; checkpoint selected by maximum test
  accuracy, then minimum test loss, then earliest epoch. This is explicitly
  exploratory monitored-test evidence, not a paper test estimate.
- This comparison changes the learning algorithm, parameterization,
  nonlinearity, learning rates and batch size; it is not a single-factor test.
- Target: Akib RTX 3080 10 GiB, SSH `akibscomputer`; Python
  `/home/filiposana/miniconda3/envs/py312/bin/python`.
- Exact config: `configs/cifar_digital_l12_logical_width_20260921.json`.
  Direct runner: `python -m experiments.train_cifar_digital_l12`.
- Local result root: `results/cifar10-digital-l12-logical-width-adam-seed0-20260921-v1/`.
  Remote root: `/home/filiposana/server_code/results/cifar10-digital-l12-logical-width-adam-seed0-20260921-v1/`.
- Budget: one production run, maximum 8 GPU-hours plus a short smoke;
  pre-smoke expectation 1--3 hours; target smoke suggests 30--60 minutes,
  to be refined from the first complete epoch.
- Gate: three real training batches and one real test batch through the
  same runner/config, finite nonzero gradients, completed canonical bundle.
- Monitor after launch through first semantic progress, then at most every
  30 minutes; status heartbeat every 100 batches and before evaluation.
  A missing worker or 45 minutes without artifact progress triggers diagnosis.
  Collect and validate the local terminal bundle before interpreting results.

## Launch and monitoring

Started 2026-09-21 12:08 CEST using the Akib skill's nohup fallback, because
tmux and screen are unavailable. Wrapper PID 419651, training PID 419660.
The launcher state and combined log are in `<remote-root>/launcher/`;
scientific state, metrics and checkpoints are in `<remote-root>/production-seed0/`.
The exact command is preserved in `launch_receipt.txt` and `launcher/command.sh`.
Source and config hashes are in `source/source_identity.json` and the manifest.

All three target smokes (batch 256, batch 512, frozen batch-512 source) pass
and validate. The auxiliary local GPU smoke first lacked sandbox CUDA access;
the permitted retry failed at local cuDNN initialization before an optimizer
step. Preserve `local-smoke.log`, `local-smoke-gpu.log`, and the failed
`local-check/smoke` bundle. This target-independent local environment failure
does not invalidate the passing same-runner/config Akib smoke.

The first complete epoch takes 19.30 seconds and reaches 46.42% official-test
accuracy. Refined total runtime estimate: about 20 minutes, with a monitoring
deadline of 20:08 CEST (eight-hour production cap). One training parent and
one CUDA process are present, with normal DataLoader child workers. Check
status, launcher exit, metrics growth and GPU process through completion.

## Completion

Finished 2026-09-21 12:24:48 CEST, wrapper exit 0. All 50 epochs / 4900
optimizer steps complete. Best/final official-test accuracy 92.30% at
epoch 50; final augmented-train accuracy 99.754%. Runner time 975.60s,
.2710 GPU-hours; peak reserved memory 6.68 GiB. No production retry.

The authoritative local copy validates, and all 26 remote production/launcher
file hashes match. Best/final checkpoint epoch and finite state checks pass;
final BatchNorm counters are 4900. All three target smoke bundles and the
preserved failed local auxiliary bundle validate. No training GPU process
remains. The analyzed outcome and limitations are recorded in
`docs/experimental_manifest.md`; the report, learning curves and epoch table
are in the study's `analysis/` directory. No follow-up run is scheduled.
