---
id: "exp-023"
title: "Post-BatchNorm and injected boundary voltages at epochs 0/1/5"
status: "complete"
hypotheses: []
---
# What voltage actually reaches the next block?

Filip requests post-BatchNorm values and the values injected into the next
convolution. Extend the descriptive voltage analysis of
[exp-022](exp-022-epoch-layer-response.md) with a read-only forward replay.

## Frozen comparison and decision

Reuse exp-022's ours fixed-0.3× EqProp trajectory at epochs 0, 1 and 5,
the identical ordered 256 training images in eight batches of 32, saved model
and BN affine parameters, minibatch BN statistics with running buffers frozen,
no augmentation/noise, float32, V4/C1, and T=[6,6,4]. Use the saved configuration
and checkpoints in `results/cifar-ours-epoch-beta-gradients-20260930-v1/inputs/`.
No optimizer, training, test split, beta sweep or equilibrium qualification.

Measure count-weighted signed mean, RMS, standard deviation, min/max and zero
fraction for all eight internal states, differential block outputs, pooled BN
inputs, normalized values before affine, actual BN outputs, and physical
inputs to blocks 1/2/3 and the dense head. Retain per-batch statistics and BN
per-channel statistics, epsilon, gamma/beta and learned input gains. Verify
the actual signed duplicated boundary equals the preceding logical activation
times its gain. Reconcile raw state RMS with exp-022, and verify checkpoint
bytes, parameters and BN buffers remain unchanged.

Describe how much upstream scale change persists after BN and the next gain.
Distinguish the variance+epsilon normalization from the learned affine output;
do not assume unit RMS. These measurements cannot identify the cause of poor
gradient alignment or establish a universal displacement bound.

## Execution and monitoring handoff

Root: `results/cifar-boundary-voltage-20260930-v1/`, including local `smoke/`,
remote/collected `replay/`, frozen `source/`, and `analysis/` artifacts.
Runner: `experiments/replay_cifar_boundary_voltages.py`.
One local CPU smoke on the first two images at epoch 0; one queued Riri RTX5090
job replays all three checkpoints sequentially in torch2.11.0+cu128/CUDA12.8.
Reserve the previously demonstrated 12GiB ceiling plus 2GiB headroom, share
only with identified Ben workloads, and respect the weekday fleet cap.
Expected GPU duration 2–5 minutes; total GPU run/validation/retry budget
900 seconds. Local smoke budget 300 seconds. Deadline October1 08:00 Paris.

The run-watch queue owns GPU launch and monitoring; root owns collection and
scientific review. Record the job receipt after submission. Preserve failed
attempts; no scientific retuning or training follow-up. Stop on nonfinite
values, input/source/restore mismatch, changed parameters/BN, inconsistent
boundary identities or exhausted budget. Validate the canonical run bundle,
all 3×256 examples, complete stage coverage and pooled statistics before review.

The CPU smoke completed and its canonical bundle validated: 28 stages on two
images, exact boundary identities, unchanged checkpoint bytes/parameters/BN.
Prepared queue job `cifar-boundary-voltage-20260930` completed; its command
and prepared-file hashes are in `launch/replay.json`. Inspect with
`gpu_queue.py status cifar-boundary-voltage-20260930`. Collect `replay/` from
the matching Riri root with rsync, then run
`python -m experiments.reporting validate-run RESULTS_ROOT/replay`.

Collected and reviewed all three checkpoints, 84 pooled stage rows and 672
batch-stage rows. Riri RTX5090 shared with identified Ben processes; queue
receipt verified the running worker and terminal command/validator success.
Replay elapsed 39.61 seconds and peak reserved CUDA memory was 1.19GiB.
All input/source/runtime/restore identities, checkpoint-byte and model/BN
immutability, exact boundary input identities, BN reconstruction, pooled
statistics and reference RMS checks passed. No failed scientific cases or
missing epochs within the declared 0/1/5 scope. No simulation remains active.
Source identity SHA256:
`4e9cd31010212c3e835c7813216e796f5ae8bb43f79f80451fb6271e07404443`.

Analyze with `python -m experiments.plot_cifar_boundary_voltages RESULTS_ROOT`.
The [review](../results/exp-023-boundary-voltage-replay.md) records the measured
normalization and gain effects. This assignment authorizes no new training.
