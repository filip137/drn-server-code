---
id: "exp-028"
title: "5090 replay: K[4,4,3], float64 only in block3 at selected RMS"
status: "complete"
hypotheses: ["H-023"]
---
# Can blocks1/2 use float32 at the chosen response targets?

Filip explicitly requests a5090 test at initialization and epochs1/5/10.
Follow [H-023](../../../hypotheses/H-023-block3-only-float64.md). Reuse the prior
256 training images, batch32, ours V4/C1 L8, frozen checkpoint/BN buffers,
no augmentation, optimizer update, read noise or official test access.
Free T remains[6,6,4]; only nudged K becomes[4,4,3]. Sequential +/- phases.

Candidate: float32 free capture/forcing in every block; nudged solves and gradient
formula in32 for blocks1/2,64 for block3. Control: same32 capture, same beta and
shorter K, all blocks nudged in64. Reference: full64 block-local BPTT at original
K=[6,6,4], also retain full64 original-K EqProp for interpretation. Do not relabel
these as full-network BPTT or equilibrium references.

Calibrate a separate beta per block/checkpoint using the candidate precision/K
to pooled output relative centered RMS[0.01,0.005,0.00001] within3%, at most12
endpoint-only evaluations/block, warm-starting prior native-K target betas. Report
actual beta and achieved output/layer RMS. On eight minibatches, retain every
layer's cosine, norm and sparsity plus pooled results. Scientific pass: every
layer/minibatch cosine>0.95 and norm ratio0.9–1.1 vs fixed local reference.
Failures are completed negative results, not grounds for changing RMS or dropping
cases. Report candidate-vs-control agreement separately to isolate precision.

Checkpoint cases: initialization, EqProp epochs1/5 from the prior beta0.3x
trajectory, and the established BPTT epoch10 checkpoint. An optional clarification
was offered; absent steering, these existing histories remain explicitly separate.
Checkpoint hashes/provenance are copied into each input directory.

After calibration, two fresh35-update timing probes (5 warmups/30 measured) on the
same5090 compare all64 nudging with block3-only64, both at K[4,4,3] and the same
initialization-calibrated beta. Free/autograd/Adam stay32 and unchanged. Report
mean/median update time, peak allocated/reserved memory and1407-update epoch compute
projection, excluding data loading/validation/recalibration. These updates are
discarded; no training continuation or accuracy claim.

## Execution and monitoring handoff

Root: `results/cifar-block3-float64-k443-20261001-v1/`; inputs, frozen source,
queue launch specs and analysis live here. Remote root is the same basename under
`/home/filip/server_code/results/` on Loulou. All attempts remain in this root.
Use existing CUDA12.8/PyTorch2.11 py312 environment and cached CIFAR training batches.
Shared run-watch queue owns admission, monitoring and collection; root owns review.
Resource profile conv3, one worker on Loulou5090, replay peak reservation24GiB plus
2.4GiB headroom (prior fullprecision replay bound). Timing reservation16GiB plus
1.6GiB headroom. Ben's idle MPS server observed; leave other processes/settings intact.
Respect daytime two-distinct5090 cap. No other GPU model or scientific fallback.

Budget four replay jobs at900s including validation (runner cap720s), two timing
jobs at300s each: total70GPU-minutes across attempts. Expected20–40min active time
including calibration/reference work; admission deadline October1 19:00 Paris.
Stop/escalate on OOM, nonfinite, invalid identities, calibration failure or budget
exhaustion. No automatic retry or beta/RMS changes. Initial-gradient failures do
not prevent later checkpoint tests. Code tests cover actual precision routing,
separate K/referenceK and immutable parameters/BN; no redundant solver audit.

Four replay jobs submitted October1 at17:13 Paris:
`cifar-b3fp64-k443-{init-ours-e0,eqprop-ours-e1,eqprop-ours-e5,bptt-ours-e10}-20261001`.
Source identity is frozen in`analysis/source_identity.json`; exact commands and input
hashes are in`launch/<case>.json`. Frozen-source CPU tests:4 passed, including a
real tiny256-image replay and bounded calibration-failure preservation. Remote source
SHA256 and queue-spec validation passed. Initialization calibration completed using the original beta guesses within3%.
Both timing jobs are queued as`cifar-b3fp64-k443-timing_{all64,block3}-20261001`,
using identical calibrated betas and300s caps each. All six jobs completed and were collected/validated by the shared service on
October1 around17:30 Paris. Each used one attempt; none remain queued/running.
Total charged runtime776.27s of4200s; see`analysis/queue_closeout.json`.
Root reviewed per-layer quality, beta targets, timing and memory; coverage is complete.
Scientific failures are retained as completed negative evidence. No next monitoring
check or training continuation is scheduled. Canonical validation and the local
`analysis/analyze.py` passed, including matched timing identities and35-update coverage.

Results and review belong in `../results/exp-028-block3-float64-k443.md`.
