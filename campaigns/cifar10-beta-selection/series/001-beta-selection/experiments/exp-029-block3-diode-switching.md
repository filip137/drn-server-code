---
id: "exp-029"
title: "Block3 local-gradient failure: switching, linearity and settling"
status: "complete"
hypotheses: ["H-024"]
---
# Quantify block3's small upper RMS limit

Filip authorized implementation and execution of the proposed mechanism diagnostic.
Test [H-024](../../../hypotheses/H-024-block3-diode-switching.md) using ours
EqProp epoch1 from the existing beta0.3x trajectory. Preserve the exp025 cohort:
256 training images, eight batches of32, no augmentation or official test access.
Snapshot, architecture, explicit diode dictionaries and preprocessing come from
`results/cifar-block3-float64-k443-20261001-v1/inputs/eqprop-ours-e1/`.
BN uses minibatch statistics with buffers frozen. No optimizer update or noise.

All model states, parameters, free capture, forcing, nudged solves and gradient
arithmetic are float64. Free T=[6,6,4]; compare only block3 with EP K4 and
block-local BPTT K4. Compute each batch's BPTT reference once; hold block input
and output-loss forcing fixed. This is a finite-iteration local comparison.

Calibrate one beta per target to pooled output centered relative RMS
[1e-6,1e-5,2e-5,3e-5,1e-4], tolerance3%, at most12 endpoint-only evaluations
per target. Warm-start from beta1.6455980388298777 at1e-5, scaled by target.
Use d=(v_plus-v_minus)/2; normalize by the saved free-state RMS. Retain actual
per-batch/layer responses as well as pooled values; do not average cosines.

For every target and layer, measure EP/local-BPTT cosine and norm ratio,
response nonlinearity relative to the1e-6 reference
`norm(d/beta-d_ref/beta_ref)/norm(d_ref/beta_ref)`, and per-update diode
decision changes versus the zero-nudge K4 trajectory. Measure diode margins
from unconstrained candidate updates, before clipping. Also record zero-nudge
drift relative to the saved free state and phase asymmetry
`norm((v_plus+v_minus)/2-v_zero)/norm(d)`.

The first target with any minibatch/layer cosine<=0.95 or norm ratio outside
[0.9,1.1] triggers one intervention on all eight batches. Repeat +/- solves
using each corresponding zero-nudge update's diode decisions: clamped entries
stay zero, free entries keep the unconstrained candidate. Keep beta, K and
BPTT fixed. Retain a fixed-mask low-beta response reference. Apply H024's
prospective recovery rule; scientific failure does not mean execution failure.
The intervention is not a physical solver or a proposed production change.

## Execution and monitoring handoff

Root: `results/cifar-block3-switching-20261001-v1/`, with frozen source, inputs,
launch command and analysis. Remote root uses the same basename under
`/home/filip/server_code/results/` on Loulou. Queue owns admission, polling and
collection; root owns scientific review. One conv3-profile worker on Loulou5090;
22.5GiB bounded peak plus2.25GiB headroom, based on prior full64 reserved peaks
of20.36GiB (exp025) and21.26GiB (exp028). Queue rechecks availability and owners.
Leave other workloads and GPU settings unchanged.

Expected5–10 minutes active compute; total cap1200s including startup/validation,
runner cap1000s. Admission deadline October1 20:00 Paris. Preserve partial
outputs on budget exhaustion, nonfinite, OOM, invalid identity or failed target
calibration; no automatic retry or scientific fallback. Focused CPU tests cover
trace equivalence and the fixed-mask intervention, followed by this assigned
scientific replay; no extra qualification sweep.

Submitted as `cifar-b3-switching-e1-20261001` on October1. Exact commands and
prepared-file hashes: `launch/eqprop-ours-e1.json`; frozen source identity:
`analysis/source_identity.json`. All four focused CPU tests passed from the
frozen source (3.82s); staged source SHA256 matched. No scientific changes to
the production solver. The Fifi job was cancelled while still queued (zero attempts/compute): another
training job occupied Fifi first. The unchanged scientific replay ran
on Loulou, sharing with Ben within measured memory headroom, as
`cifar-b3-switching-e1-loulou-20261001`; its exact command is
`launch/eqprop-ours-e1-loulou.json`. The study budget remains1200s total.
The queue completed its single Loulou attempt and local collection/validation.
All five targets and both fixed-mask controls were collected and reviewed;
parameters/BN, source and input identity checks passed. Total charged runtime
379.71s of1200s; peak reserved20.36GiB. Receipts: `analysis/collection.json`
and `analysis/queue_closeout.json`. The cancelled Fifi placement used no compute.

Inspect with `gpu_queue.py status cifar-b3-switching-e1-loulou-20261001`.
The declared `analysis/collect.py` performed canonical/diagnostic validation and
source/config matching. `analysis/analyze.py` produced the reviewed summary/JPG.
There is no remaining run or monitoring action for this experiment.

Results and review belong in `../results/exp-029-block3-diode-switching.md`.
