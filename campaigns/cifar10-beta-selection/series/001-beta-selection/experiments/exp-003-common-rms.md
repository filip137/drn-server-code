---
id: "exp-003"
title: "Focused beta sweep for a common CIFAR output displacement"
status: "complete"
hypotheses: ["H-002"]
---
# exp-003 — Does a common output RMS predict high cosine?

Filip explicitly requests a focused beta sweep. Tests
[H-002](../../../hypotheses/H-002-common-output-displacement.md), following exp-001.

Hold the original ours initializer, 256 examples, batch32, T/K=[6,6,4], no noise
and the stable centered EqProp calculation fixed. Measure both native and matched
local BPTT cosine for all eight Conv weights, output relative/absolute RMS,
gradient norms and batch variation. No training or optimizer updates.

Place beta samples by interpolating exp-001's measured B-to-R curves in log
coordinates, separately for each block, at nominal relative RMS targets
[0.02,0.03,0.04,0.05,0.06,0.075,0.09,0.11,0.135,0.165,0.20,0.25].
These targets only place the samples: always analyze the newly measured RMS.
Report target errors; within5% is an approximate common-R match, not an excuse
to substitute nominal targets for measurements. Include the prior0.01x vector
as an instrument reference. No adaptive retuning in this declared sweep.

Primary: minimum cosine over all Conv tensors in a block, after averaging
256-example gradients; high means >=0.95. Also report >=0.99 and batch-level
spread. Compare the three curves on relative and absolute RMS axes. Identify
sampled high-cosine bands and their overlap; label interpolated boundaries as
estimates. A shared useful band does not establish that R alone determines the
cosine or that all architectures/checkpoints obey the same cutoff.

Result root: `results/cifar-block-rms-focused-20260929-v1/`. One Loulou RTX5090
shared-queue worker; expected3–6min, cap1200s across attempts. Retain the existing
six CPU numerical checks; no new smoke or solver qualification. The source and
inputs are frozen from the previous sweep plus the per-block beta-grid extension.
Completion:13 vectors x8 layers x8 batches, unchanged model/BN, finite reported
references, local bundle/coverage validation and reference-point agreement.
Exact config/job and the single queue monitoring owner are recorded below.

Prepared job `cifar-block-rms-focused-5090-20260929`; exact beta vectors in `configs/cifar/block_rms_focused_20260929.json`. Source identity, launch command, and CPU collector are under `results/cifar-block-rms-focused-20260929-v1/{source,launch,analysis}`. Six focused numerical tests pass. Queue `status` is the inspection path; persistent service owns placement, monitoring, recovery and collection. Canonical progress is per block/batch; completion checks the full13-case coverage and previous0.01x reference. Deadline October1 08:00 Paris; charged cap1200s.

Completed and reviewed: all13 vectors, 104 pooled and832 batch rows collected and validated, unchanged model/BN and matching reference. Compute200.8s. Queue completion releases the GPU; [the reviewed result](../results/exp-003-common-rms.md) owns the measurements, H-002 verdict and next decision.
