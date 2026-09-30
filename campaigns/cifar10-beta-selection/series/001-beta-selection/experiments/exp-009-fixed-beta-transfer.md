---
id: "exp-009"
title: "Freeze initialization beta and measure RMS/cosine at epochs 10/30/50"
status: "complete"
hypotheses: ["H-007"]
---
# Fixed-beta checkpoint replay

Filip requests a fixed-beta check following the proposed six fresh 10-epoch
training arms (three schemes, initialization RMS 0.03/0.09). Run this diagnostic
first. The six training arms remain the follow-up in exp-002 and are not launched
by this diagnostic.

For each baseline/ours/legacy block, reuse the exact physical beta previously
calibrated at initialization for relative RMS 0.03 and 0.09. Freeze these two
vectors across epochs 10/30/50; do not recalibrate beta at trained checkpoints.
"Same beta" means unchanged through training within a scheme/block, not one
numerical coefficient shared across all schemes/blocks. Reuse epoch-zero
measurements from exp-003/004, with provenance in the new bundle's analysis.

Reuse exp-008's verified checkpoint ancestry and exp-006's unchanged frozen
runner. Same 256 training examples, batch32, native endpoint convention,
T/K=[6,6,4], minibatch BN with frozen buffers, no noise/augmentation/updates.
No smokes. Measure actual pooled relative and absolute displacement, native and
matched-local BPTT cosine/norm/sparsity, and minibatch spread for all eight Conv
weights. Report ratios to initialization and the 0.95 descriptive alignment
threshold; do not interpolate missing measurements or claim noise robustness.

## Execution and monitoring handoff

Root `results/cifar-block-fixed-beta-20260929-v1/`, nine `e{10,30,50}/{scheme}`
bundles; configs `configs/cifar/block_fixed_beta_20260929/`. Jobs
`cifar-fixed-beta-e{10,30,50}-{ours,legacy,baseline}-5090-20260929`, assigned
alternately to Loulou and Trex, one worker per GPU. Expected 1–2 minutes/job;
300 seconds/job including attempts, 2700 seconds total; deadline October1 08:00
Paris. Reserve 12 GiB plus 2 GiB headroom, respecting Ben sharing and two-GPU cap.
Shared run-watch queue owns placement, monitoring and CPU collection; no other
watch. Inspect using installed `gpu_queue.py status JOB_ID`.

Each collector validates canonical completion, 16 pooled/128 batch rows, exact
two beta vectors, unchanged checkpoint/model/BN, 256 examples and zero undefined
cosines. Stop and preserve numerical/identity failures; no scientific retuning.
Local summaries and combined analysis must be reviewed before declaring complete.

All nine jobs were accepted by the shared queue; fresh service observations and
worker receipts confirm monitoring ownership. After epoch10 legacy completed,
Kellian occupied Trex with about 24 GiB. The three unstarted Trex jobs (epoch30
ours/baseline and epoch50 legacy) were cancelled with zero attempts/charged time
and replaced on Riri with the same IDs suffixed `-riri`. Original placement/job
files are preserved as `launch/*-trex-cancelled.json`; no computations discarded,
no other user's job changed, and original budgets retained. At most two of our
GPUs are active at once. Loulou jobs continue unchanged.

Loulou then finished its original cases, so epoch30 baseline was withdrawn from
Riri before any attempt and assigned to Loulou with suffix `-loulou`, keeping two
workers available for the remaining cases. That unused Riri spec is also retained
as `launch/*-riri-cancelled.json`. These placement changes consumed no compute.

All nine computations/collections are complete and locally validated: 144 pooled
and 1152 batch rows, fixed beta identity and unchanged checkpoints/model/BN.
Charged 436.84 seconds total, no failed computations or exclusions. Queue jobs
are terminal and GPUs released. [Reviewed results](../results/exp-009-fixed-beta-transfer.md)
support substantial effective-displacement drift and poor block2 alignment with
frozen initialization beta. Both beta settings and every planned checkpoint are
covered. No training or noise tests were launched by this diagnostic.
