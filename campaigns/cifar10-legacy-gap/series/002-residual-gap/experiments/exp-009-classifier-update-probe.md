---
id: "exp-009"
title: "Isolate classifier-weight Adam steps on saved CIFAR checkpoints"
status: "complete"
hypotheses: ["H-009"]
---
# Classifier-update diagnostic

Filip requested an Explorer hypothesis and its check, confirming CIFAR BPTT scope.
Test [H-009](../../../hypotheses/H-009-classifier-update-imbalance.md) without
restarting training or modifying saved checkpoints. Existing whole-update and
separate-cohort probes were already completed; this intervention isolates the
classifier conductance group rather than repeating those tests.

Legacy/ours, original epochs30/50, four gradient minibatches32 drawn from the same
first128 split-training indices as the existing replay. Separate held-out training
cohort: split-training indices256:512 (256examples), no augmentation or validation/
official-test access. At every direction restore saved model, Adam and BN state.
Compute the real BPTT gradient and actual projected Adam step with head.weight LR
multiplied by0,0.5,1,2; all remaining groups keep original LR/state/gradients.
Compare against the no-update state. Primary: held-out CE under minibatch BN,
without buffer changes. Secondary: saved-stat BN, same-gradient-batch CE, head
movement and relative logit changes. Check exact non-head-update equality, finite
outputs, original checkpoint hashes and full restoration. No new solver audit.

Decision rule is frozen in H-009 before measurements. A mixed or sub-tolerance
result remains inconclusive; no subsequent best-multiplier selection substitutes
for the half-step prediction. This diagnostic does not measure long-run accuracy.

Execution root: `results/cifar10-classifier-update-probe-20260927-v1/`, including
frozen source/input hashes, independent smoke, and canonical `probe/` bundle.
Exact script: `experiments/probe_cifar_l8_classifier_update.py`.
Use the shared run-watch queue, target Trex RTX5090 if eligible, ordinary Ben
sharing permission, peak12,000MiB plus3,072MiB headroom. Expected10minutes,
cap30minutes for smoke/probe/validation; collection has a separate5minute CPU cap.
Calendar cutoff September28 08:00 Paris. No automatic retries or new training.
The queue owns routine observation; the coordinating Explorer reviews local output.
Prepared source/input hashes and launch command: execution-root
`source/source_identity.json`, `inputs/checkpoints.json`, `launch/job.json`.
Queue ID: `cifar-classifier-update-probe-20260927`; Trex is the sole staged target.
Final handoff: Trex attempt `2bf5e72bfcf24e65895fca6cec3c199b` completed all
16 directions and 96 measurements, including smoke/validation in 375 seconds.
Queue collection and local coverage validation succeeded; no retries or failed
attempts. Original checkpoints are unchanged. The GPU reservation is released.
Reviewed conclusion: [result note](../results/exp-009-classifier-update-probe.md).
