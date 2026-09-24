# CIFAR L8 analog convolution and classifier batch comparison

Approved September 21, 2026. Three exploratory seed-0 BPTT runs on Fifi RTX5090:
five epochs each at batch16,32,64, using all eight analog convolutions and an
analog dense classifier. Independent blocks [3,3,2] at logical widths128/256/512;
physical differential widths256/512/1024. Digital max-pool2 and affine BN after
every block, flatten8192 then resistive dense from16384 physical inputs to10
linear voltage logits. No digital learned output matrix. Dense equilibrium is
one exact coordinate update, -b/(2a), from the DenseResistive energy.

Trainable BN gamma/beta, crop/flip augmentation, cross-entropy with no smoothing.
All analog biases frozen zero; conductance bounds[1e-7,10], inherited Kaiming
initialization/gain1. Voltage/current amplification4/.25, four positive trainable
boundary gains initialized100. Digital preprocessing, BN, pooling and loss remain.

Adam .9/.999 eps1e-8, all weight decay0. Shared rates, no calibration or search:
conv [4.5e-3,6.9e-4,6.9e-4]/[6.9e-4,4.9e-4,4.9e-4]/[4.9e-4,3.45e-4]; dense1.6e-4;
BN1e-3; gains5e-5. Relative cosine horizon50, final ratio.02; stop after5 epochs.
Configs: configs/cifar/cifar10_l8_analog_bs{16,32,64}_seed0.json.

Deterministic stratified45k/5k split,500 validation samples/class, split seed0;
only five training batch files are read. No official test file is staged or read.
Shared saved initialization, per-epoch sample permutation, and stateless sample/
epoch keyed augmentations. Common validation batch16, incomplete batches retained.
Rank epoch5 validation accuracy, then lower cross-entropy, then shorter runtime.
Digital L8's92.62% monitored test at50epochs is context, not a matched comparison.

Initial T=K[6,6,4], alternative[12,12,8]; reference[24,24,16], stability sentinel
[48,48,32]. Eight deterministic training examples, microbatch2. All nine weight
gradients: mean cosine>=.9, relative norm difference<=.1, zero fraction difference
<=.02; reference median RMS>1e-12 and zero Q90<.99. Free and tracked relative
logit difference<=.01. Freeze first passing schedule for all arms; fail closed
if reference unstable or neither candidate passes. Final checkpoints get the
same read-only audit. This is an exploratory CIFAR audit, not MNIST paper T/K
qualification. Final audit failure is reported alongside observed accuracy.

Model: labs/cifar_l8_analog.py; runner: experiments/train_cifar_l8_analog.py.
Copy the prior CIFAR block implementation into the new focused model and use the
current worktree's physical resistive primitives. Preserve prior runners.
Prepare shared assets and qualification; smoke two real optimizer batches and
one validation batch at every requested batch size, then execute sequentially.
OOM is an explicit infeasible arm, with no gradient accumulation substitution.

Root: results/cifar10-l8-analog-conv-and-head-adam-bs16-32-64-seed0-v1.
Remote: /home/filip/server_code/results/ with the same basename, isolated source.
Fifi Python: /home/filip/miniconda3/envs/py312/bin/python (torch2.11/cu128).
Expected3–8 GPUh pending smoke timing; cap4h per production arm plus1h checks.
Record actual estimate before full training. Heartbeat every100batches and at
evaluation/audit transitions. Monitor processes, GPU, metrics and logs at least
every30min through completion; collect and validate locally. Preserve failures,
checkpoints, source/config identities and launcher exit codes. Do not edit the
human-authored current_state.md. No automatic50-epoch continuation.

## Gates and launch, September 21

Three focused CPU tests pass. The same-source Fifi initialization audit passes
T=K[6,6,4]: minimum mean weight-gradient cosine .99999995, maximum relative
norm delta .0005403; free/tracked logit relative differences .0007507/.00000414.
Reference stability passes. All three same-runner two-step smokes pass, using
5.38/10.90/21.46GiB for batches16/32/64. All19 trainable tensors have finite
nonzero gradients. Smoke elapsed epoch0.93/1.41/2.41s includes validation and
startup; expected full sweep3–5h, to refine from complete epochs.
Initial prepare bundle is collected locally and validates. Other live GPU and
Slurm jobs were inspected and left untouched; Fifi remains the only target.
Wrapper run_fifi.sh records child timeouts, separate logs/exit codes,4h caps,
and proceeds through all three requested arms. Monitoring remains active.

Batch16 first epoch completed: validation34.36%, CE1.805179,790.69s.
Expected batch16 runtime about66min; sweep estimate remains roughly3h.
All five locally staged training files match official CIFAR MD5 checksums;
training_data_verification.json records MD5/SHA256 without test access.

## User-authorized parallel placement

Filip requested outsourcing to idle Nom-cool-1. Move only the unstarted batch32
arm to its RTX3090 (24GiB), leaving batch16 active and batch64 queued on Fifi.
Preserve the exact saved initialization/split, rates, augmentation and T/K.
Nom uses torch2.5.1/cu121 versus Fifi2.11/cu128, so hardware/software now covary
with batch32. Report this deviation; do not infer isolated batch-size throughput
effects or numerical equivalence. Source-nom differs only by a --target metadata
argument; model and config bytes are unchanged. Run a destination smoke before
production. Existing Fifi wrapper2237523 is STOPPED to prevent duplicate32;
training PID2237528 and timeout2237527 continue. Replace only the waiting queue
controller after recording its child exit, then run64 from unchanged Fifi source.
Nom result root has the same basename under /home/filip/server_code/results.

Nom destination smoke nom-smoke-bs32 failed before its first optimizer step
with CUDNN_STATUS_NOT_INITIALIZED in BatchNorm. Minimal BN replay reproduces
the failure. PyTorch2.5.1 requires cuDNN9.1.0.70, which package metadata claims
is installed, but the actual loaded library reports9.19.0.56. Repair is an
isolated study-local cuDNN9.1 install/preload; global environment untouched.
Fifi handoff first attempt failed its protective cwd assertion (actual cwd is
source/); corrected guard passes. Handoff PID2242208 awaits timeout2237527
exit, records its real exit status from the unreaped child, retires only the
paused obsolete queue controller, and launches64. Native exit adoption is tested.

The isolated cuDNN9.1 runtime passes the conv/BN regression probe, with every
loaded cuDNN library verified under the private runtime directory. Destination
smoke nom-smoke-bs32-cudnn91 passes. Batch32 production started12:46:32UTC
on Nom-cool-1: wrapper1649855, trainer1649864. Same source/model/initial state
except the target-metadata CLI addition; global packages were not modified.
Fifi handoff2242208 remains waiting, with active training16 unaffected.

Batch16 completed five epochs with native exit0: final validation61.02%,
CE1.094020993, elapsed3968.316s. Final T/K solver audit passed. Bundle
collected and locally validated. The handoff observed the native child exit,
retired only the stopped obsolete wrapper, and launched batch64 on Fifi
(trainer2246546). No batch32 training directory was created on Fifi.

Batch64 completed with native exit0: final validation60.02%, CE1.127519691,
elapsed3771.297s and final solver audit passed. Fifi queue exit0; no GPU
worker remains. Both Fifi production bundles and launcher receipts collected
and validated locally. Batch32 on Nom remains active.

## Completed comparison

Batch32 completed with native/launcher exit0: final validation65.50%,
CE0.971837180, elapsed9449.655s, final solver audit passed. All three production
bundles and all operational records are now collected locally. The analyzer
validates all nine canonical bundles, five-epoch example coverage, shared
initialization/split, all19 gradient norms per epoch, finite/bounded checkpoints,
BN updates, and all launch exits. Ranking32/16/64; all final audits pass.
Total production elapsed time is 4.7748 GPU-hours. No study worker remains.

Batch32 is the provisional choice, with the recorded host/software confound;
no long-run or statistically robust optimum is claimed. A matched 5090 repeat
would clarify that confound before longer training; no extra case is launched.
The first queue-handoff guard failure and failed Nom smoke are retained and
excluded. The current lightweight repository workflow and AGENTS authorize
agent interpretation and manual manifest curation, superseding the closeout
skill's historical managed-study schema and human-review gate.

[Report](../results/cifar10-l8-analog-conv-and-head-adam-bs16-32-64-seed0-v1/analysis/report.md) ·
[Validation](../results/cifar10-l8-analog-conv-and-head-adam-bs16-32-64-seed0-v1/closeout_validation.json) ·
[Manifest](experimental_manifest.md).
