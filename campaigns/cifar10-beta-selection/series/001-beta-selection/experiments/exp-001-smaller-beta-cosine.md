---
id: "exp-001"
title: "CIFAR cosine and displacement across smaller betas"
status: "complete"
hypotheses: ["H-001"]
---
# exp-001 — CIFAR cosine and displacement across smaller betas

Tests [H-001](../../../hypotheses/H-001-large-nudging-error.md). Reuses the
[original pilot](../../../../pilots/cifar-block-beta-20260929.md) initializer,
force definition, calibrated B values and validated stable gradient computation.


September 29: Filip authorizes the proposed decrease in B at fixed T/K. Test
multipliers [1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001] of each block's calibrated B.
Reuse the exact initializer, 256 images, batch32 BN statistics, frozen summed-CE
force, float32 solver, no noise/augmentation, and T/K=[6,6,4]. Use the validated
stable centered-gradient formula. Cache each batch's native and local BPTT
references across multipliers; each EqProp sign starts from the same post-T state.
Measure centered output displacement from those same endpoints, all eight weight
cosines/norms, and batch variation. Keep dead/zero-response cases visible.

Question: does lowering B improve agreement, consistent with the original
displacement targets causing nonlinear nudging error? Report the curve and the
largest tested multiplier meeting pooled cosine>=0.90 for every Conv within each
block, if any; this descriptive threshold is not a training qualification.
Persistently poor agreement at small B leaves finite settling/float32 response
resolution as possible causes, not a demonstrated cause. No T/K sweep or training
is part of this assignment.

Result root: `results/cifar-block-beta-sweep-20260929-v1/`. Shared queue owns one
eligible Loulou5090 worker, monitoring and CPU collection. Expected5–10min,
total cap1200s including attempts. Register exact queue/config paths below before
launch. Completion requires7 multipliers x8 layers x8 batches, both references,
unchanged parameters/BN and validated local results; no training smokes.

Execution handoff: shared run-watch queue is the monitoring owner; queue/config and collection details will be recorded here before submission.

Prepared queue job `cifar-block-beta-sweep-5090-20260929`; config `configs/cifar/block_beta_sweep_20260929.json`. Source identity, exact argv, and CPU collection hook are under `results/cifar-block-beta-sweep-20260929-v1/{source,launch,analysis}`. Six focused CPU numerical tests pass after adding cached references and endpoint displacement capture. The 1x point is compared with the prior validated replay during collection. Queue `status` is the routine observation path; its completion hook collects and validates locally. Deadline October1 08:00 Paris,1200s charged budget, expected5–10min. No official test data or training smoke.

Completed and reviewed: shared queue reports complete; local collection validates56 pooled layer rows and448 batch rows, with matching1x anchor and unchanged model/BN. Compute129.3s; no failures. Findings and H-001 review are in [the result](../results/exp-001-smaller-beta-cosine.md).
