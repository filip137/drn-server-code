# CIFAR-10 wider digital L8 with trainable boundary BN

User request September 21: train the wider L8 digitally. Carry forward the
previous run's Akib target, 50 epochs, seed 0 and batch512, subject to GPU smoke.
One exploratory run, not a paper accuracy estimate or a single-factor ablation.

- Blocks: [128,128,128] / [256,256,256] / [512,512], logical channels.
  Eight bias-free 3x3 stride1 padding1 convolutions, ReLU after each.
  Each block ends with 2x2 max-pooling then affine BatchNorm (gamma1/beta0
  initialization, learned scale/shift and running statistics). No internal BN.
  Flatten512x4x4 then Linear8192-to-10 with bias; 5,395,594 parameters.
- Fully digital float32 BP, no amplification, solver, bounds or input gain.
  Same PyTorch default convolution/head initialization as L12.
- Adam .001, betas .9/.999, epsilon1e-8, weight decay .0003 for convolution
  and head; zero decay for BN gamma/beta. Cosine50 epochs to .00002.
- Batch512 train/test, four workers, deterministic kernels, TF32 disabled.
- CIFAR10 full50k training,10k official test monitored each epoch. Existing
  horizontal flip and padded random crop, inherited mean/std unchanged.
  Best checkpoint: maximum test accuracy, minimum loss, earliest exact tie.
  This continues the exploratory monitored-test contract; not held-out
  architecture-selection evidence. Comparison with L12 changes architecture
  and BN jointly, so cannot isolate the contribution of learned BN.
- Config: configs/cifar_digital_l8_wide_affinebn_20260921.json.
  Reuse config-driven experiments.train_cifar_digital_l12, extended to accept
  the configured depth and optional BN decay exemption; L12 defaults preserved.
- Target: akibscomputer / integnano-akib, RTX3080 10GiB; Python
  /home/filiposana/miniconda3/envs/py312/bin/python. Stage isolated source under
  the result root, preserving all other remote work.
- Root: results/cifar10-digital-l8-wide-affinebn-adam-seed0-20260921-v1;
  remote root is the same basename under /home/filiposana/server_code/results.
- Budget: one50-epoch production, maximum1GPUh plus short smoke; expected
  15–25min based on completed L12. Same-runner three-training-batch plus
  one-test-batch target smoke must finish with finite nonzero gradients,
  including all BN parameters. Existing local cuDNN environment failure and
  unavailable local CIFAR dataset make the target smoke the execution gate.
- Launcher: Akib skill nohup fallback with tracked PID/log/exit receipt.
  Check one intended process and first epoch, then progress within30min;
  stay active through completion, collect and validate locally. Heartbeat at
  least once per epoch; hard runtime cap prevents indefinite execution.

All inventory targets checked before placement. Akib idle; other targets
have unrelated small processes or work and are left untouched. Jean Zay
user queue empty. No analog T/K qualification applies to digital BP.

## Launch

Target smoke passes at batch512 with peak reserved memory6.65GiB and all
16 parameter-tensor gradients finite/nonzero. Local CPU shape/count and BN
gradient check passes; unchanged L12 still has10,849,674 parameters.
Production launched September21 at about10:42UTC using nohup. The first
launcher attempt was sandbox-network-blocked before contacting the host;
permitted retry uses the unchanged command and source. No duplicate run.
Launcher receipts: <root>/launcher; scientific bundle: <root>/production-seed0.
Hard monitoring deadline about11:43UTC.

Verified wrapper424041, training PID424050, one CUDA worker. Started
10:42:31UTC; first epoch14.62s, refining expected duration to about12min.
Target smoke is collected locally and its canonical bundle validates.

## Completion

All50 epochs/4900steps completed, wrapper exit0 and no GPU worker remains.
Best92.62% at48; final92.57%; final augmented-train99.888%. Runtime734.55s
(12m15s,.2040GPUh); mean epoch14.603s; peak6.65GiB. Local production and
smoke bundles validate, complete contiguous full-dataset metrics and finite
checkpoints. BN gamma/beta changed in every layer; all running-statistic
counters4900. Best/final checkpoint epochs48/50; final LR.00002; BN decay0.
Report and L12 comparison curves in analysis/. No training retry or follow-up
run; digital accuracy is promising but does not establish analog accuracy.

## Recommendation for subsequent analog tests

At Filip's request on September 21, recommend the wider L8 as the reference
architecture for subsequent all-analog convolution tests: three coupled
blocks of depths **[3, 3, 2]**, logical widths **128/256/512**, with digital
max-pooling and trainable affine BN at the block boundaries and the same
digital classifier. All eight convolutions should be analog; do not insert
digital BN between the convolutions within a coupled block.

Retain crop/flip augmentation and cross-entropy without label smoothing.
The digital reference is **92.62% best / 92.57% final test accuracy**, final
test cross-entropy **0.3224**, versus L12's 92.30% / 92.30% and 0.4268.
This supports trying the shallower architecture, not an analog performance
claim. New analog solver/T/K and learning-rate qualification are still needed;
the digital batch size and learning rate are not automatically transferable.
No analog run has been launched by this recommendation.

See the durable conclusion in [the experimental manifest](experimental_manifest.md#cifar10-digital-l8-wide-affinebn-adam-seed0-20260921-v1--wider-digital-l8).
