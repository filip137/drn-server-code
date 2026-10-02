---
id: "exp-010"
title: "Isolate convolution, affine-BN and gain updates at epoch 30"
status: "complete"
hypotheses: []
---
# Update-family isolation

Authorized follow-up to [exp-009](../results/exp-009-classifier-update-probe.md):
which parameter groups contribute to the remaining separate-batch loss increase?
This is exploratory attribution, not a new confirmatory hypothesis or training run.

Original batch-32 design (superseded before production by the user-requested
batch-16 revision below): reuse seed-0 legacy/ours epoch-30 checkpoints, saved Adam state, four
gradient batches of 32 (split-training indices 0:128), and separate training
indices 256:512. Same native finite-iteration BPTT, preprocessing and no augmentation.
Compute one full projected Adam update per direction. Partition every trainable
parameter into convolution conductances, BN affine parameters, input gains, or
classifier conductances; require exhaustive, disjoint membership. Apply all eight
on/off combinations of convolution/BN/gain updates with classifier update fixed
to full. Add original and repeated-original controls: 80 production measurements.
Selected tensors must exactly equal the full-step tensors; frozen tensors must
exactly equal the checkpoint. Hold all BN running buffers at saved values.

Primary: separate-cohort CE under minibatch BN. Secondary: saved-stat BN and
gradient-batch CE. Report each group's effect alone, benefit of omitting it from
the full update, and exact three-group Shapley attribution of full-minus-head-only
loss (average marginal effect over all six group orderings). Also report pair and
three-way interactions; attribution is conditional on the classifier update.
Tolerance: max(0.00001 CE, ten times repeated-original discrepancy). A consistent
adverse contributor requires both positive Shapley contribution and positive
omission benefit above tolerance in at least 3/4 directions, with both means above
tolerance. Label it legacy-specific only if mean omission benefit exceeds ours
by that tolerance. Report all groups; no significance claim or inferred final
accuracy gain from these local interventions. Single seed and shared probe cohort.

Result root registered before creation:
`results/cifar10-update-family-probe-20260927-v1/` (inputs, frozen source, smoke,
canonical probe, analysis). Script: `experiments/probe_cifar_l8_update_families.py`.
Expected 10–15 minutes; cap 30 GPU minutes including smoke and target validation,
plus 5 minutes CPU collection, cutoff September 28 08:00 Paris. Prefer currently
free Akib RTX3080 with PyTorch allocator capped at 70% of device capacity,
8,192 MiB bounded process memory plus 1,024 MiB headroom. Same-path smoke must fit;
an OOM preserves its failed bundle and returns the remaining allowance for a
larger-GPU placement, with unchanged scientific settings. No automatic retries.
Shared queue owns launch/monitor/collection; coordinating agent reviews evidence.
Prepared source/input identities and command: execution-root
`source/source_identity.json`, `inputs/checkpoints.json`, `launch/job.json`.
Initial queue ID: `cifar-update-family-probe-20260927` on Akib.
The wrapper performs one same-path smoke, all eight directions, exact coverage
validation and controller-side collection. Source compiled; original checkpoint
files and data are reused from exp-009.

The Akib smoke hit its allocator cap, before any production measurements.
Its failed bundle is retained under `smoke/790a842e94af47c6bc1f29f77ced476a/`;
the worker stopped and its reservation is released. Charged time: 5.267 seconds.
Numerical settings and the original frozen source are preserved.

Larger-GPU handoff: `launch/job-large.json`, frozen `source-large-v2/`, job
`cifar-update-family-probe-large-20260927`. Prepared placements: local/nom-cool-1
3090 and Trex/Fifi/Loulou/Riri 5090, subject to normal ownership/admission rules.
Remaining cap 1,794 seconds, unchanged deadline; bounded peak 18,000 MiB plus
3,072 MiB headroom. Allocator limits are 65% on 3090 and 50% on 5090, without
changing the scientific workload. No jobs are interrupted to obtain capacity.
Collection runs `analysis/review.py`, which checks coverage, computes all
attributions/interactions, applies the frozen criteria, writes the result note
and JPG, and refreshes the ledger. It cannot launch another scientific test.
The larger-GPU attempt ran its smoke on nom-cool-1 and failed with
`CUDNN_STATUS_NOT_INITIALIZED`; no production measurements were made. Retained
bundle: `smoke/0bf3d27a310d455a8c0c1e280d7769d6/`. Its worker/reservation stopped,
and its queue job is cancelled. Additional charged time: 3.634 seconds.

User revision: reduce batch size and execute on the 3080. Use batch 16 for both
BPTT and evaluation, four gradient batches per scheme (first 64 split-training
indices), with the same 256 separate training examples. Keep saved optimizer,
LRs and all intervention/decision rules unchanged. This changes BN statistics
and gradient directions; it cannot directly attribute exp-009's batch-32 loss
increase. Still eight directions and 80 measurements. No new training.
Queue `cifar-update-family-probe-b16-20260927`, `launch/job-b16.json`, frozen
`source-b16-v3/`, canonical `probe-b16/`, analysis `analysis-b16/`. Akib only,
8,192 MiB process bound plus 1,024 MiB headroom; remaining total cap 1,791 seconds.
Both prior failures remain preserved. Collection applies the unchanged criteria
with an explicit batch-size qualifier and writes the result/plot automatically.
Current handoff: all eight directions collected and validated; the predeclared
attribution and decision rules have been applied to every family and both BN modes.
See [result](../results/exp-010-update-family-isolation.md).
