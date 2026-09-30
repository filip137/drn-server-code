---
id: "exp-022"
title: "Epoch 0/1/5 layer response and gradient agreement across beta"
status: "complete"
hypotheses: ["H-017"]
---
# Does increasing trained-layer response recover gradient quality?

Filip explicitly requests epochs 0, 1 and 5 on the same 256 images. The idea
is that poor learning may follow RMS displacement becoming too small as the
network changes. Test [H-017](../../../hypotheses/H-017-small-trained-layer-response.md)
with read-only replay of ours, the fixed-0.3× trajectory from exp-013. This
assignment does not reopen the stopped training queue.

## Frozen comparison

- Checkpoints: `inputs/initializer.pt`, `cells/1/beta_0p3/checkpoints/final_model.pt`
  and `cells/5/beta_0p3/checkpoints/final_model.pt` under
  `results/cifar-eqprop-ours-5090-20260929-v1/`.
- Same original ordered 256 training indices; eight batches of 32, no augmentation.
  Saved model/BN, minibatch BN statistics with running buffers frozen, seed 0,
  ours V4/C1, L8 blocks [3,3,2], T/K [6,6,4], float32,
  torch 2.11.0+cu128 / CUDA 12.8. No noise, optimizer steps or test-set access.
- Base beta vector: [0.028356966006427994, 1.1189959117137818,
  0.35085590124458804]. Independently interpret each block's sweep of
  multipliers [0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100,
  300,1000,3000,10000]. Each candidate uses one common beta for its block.
- Cached native BPTT and matched frozen-current local BPTT references; stable
  factored centered EqProp. Every nudged pair starts from the same saved free
  state and force. Native and local references remain separately labelled;
  neither is claimed to be a qualified equilibrium gradient.

## Measurements and decision

Measure cosine, norm ratio, exact-zero gradient fractions, absolute and relative
RMS of half the plus/minus state difference for each convolution's postsynaptic
state, unchanged-coordinate fraction, block output force and injected force.
Pool gradients over 256 images and displacement through raw squared sums;
also retain eight-batch spread. Compare gradient magnitudes with epoch 0.
Checkpoint bytes, parameters and BN buffers must remain unchanged.

Plot alignment against beta and against each layer's measured RMS. At each epoch,
report the original beta and the common per-block beta maximizing the worst
layer's native cosine, with norm ratios and minibatch variation alongside it.
Cosine >=0.95 and norm ratio 0.9–1.1 for every layer is a proposed descriptive
criterion, not an established paper gate. Report boundary optima explicitly.

Recovery as displacement increases supports insufficient response as a gradient
mechanism. No recovery within this sweep rules out beta adjustment alone only
within the tested range and T/K. Good pooled agreement may conceal poor batch
agreement. This experiment cannot establish improved training accuracy.

## Execution and monitoring handoff

Root: `results/cifar-ours-epoch-beta-gradients-20260930-v1/`.
Configs: `configs/cifar/epoch_layer_response_20260930/e{0,1,5}.json`.
Frozen source and input hashes live under the root. Three queue jobs, sequential
on Riri RTX 5090, sharing only with identified Ben processes when memory fits.
Reserve 12 GiB plus 2 GiB headroom; weekday fleet cap remains in force.
Expected 5–10 minutes per checkpoint; maximum 1,800 seconds each, 5,400 total,
including validation/retries. Deadline October 1, 08:00 Paris. No smokes.
The optional measurement extension passed 37 relevant CPU tests before freezing.

Queue IDs: `cifar-ours-epoch-beta-gradients-e{0,1,5}-20260930`.
The shared queue service owns execution monitoring and CPU-only collection.
Root owns scientific review; no second polling owner. Inspect with
`gpu_queue.py status JOB_ID`. Prepared commands are in `launch/e*.json`.
Collector: `analysis/collect.py EPOCH`; validates each canonical bundle and
copies it to `cells/eEPOCH`. Analyze with `analysis/analyze.py` after all three
are collected. Failures and partial artifacts stay under the same result root;
no automatic scientific changes or training follow-ups. Escalate operational
failure, nonfinite output, stalled progress or exhausted budget to the Codifier.

Status: all three checkpoints completed on Riri and were collected and validated.
The queue owns terminal receipts; no simulation remains in this assignment.
All 408 pooled and 3,264 minibatch rows passed coverage/pooling checks. Compute
took 748 seconds, about 754 seconds including queue-side validation. The source,
checkpoints, parameters and BN identity checks passed. Host timestamps are about
two hours behind the controller; elapsed times and observed progress govern.
Source identity SHA256: `9e5ee03efc2684d3ba5dc16f93c845af2e36f2ebc6e2cdd56750568b62ca8491`.
The [review](../results/exp-022-epoch-layer-response.md) supports H-017 for trained
gradient recovery, with an epoch1 block3 minibatch limitation. All plots and
tables are under `analysis/`. No training follow-up is authorized by completion.
