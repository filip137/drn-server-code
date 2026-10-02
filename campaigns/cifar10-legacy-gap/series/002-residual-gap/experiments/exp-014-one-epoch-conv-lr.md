---
id: "exp-014"
title: "One-epoch matched legacy convolution LR screen"
status: "complete"
hypotheses: []
---
# One-epoch convolution LR screen

Filip approved the shorter screen proposed after exp-013: original versus 3 times
convolution LR, each starting from the same original legacy epoch-30 checkpoint.
Run one complete epoch (31; 1,407 updates, batch32), not repeated training on a
small subset. Both branches run sequentially on the same device/software stack.
Restore model, Adam moments/steps, scheduler, BN buffers and RNG before each branch.
Keep the original epoch-31 permutation and deterministic crop/flip augmentation.
Scale only convolution LR/base LR by 3 for the intervention; leave head, BN, gain
LRs and every other scientific setting unchanged. Preserve full epoch-31 states
so an explicitly assigned extension can continue without restarting.

Evaluate at 0, 100, 400 and 1,407 updates (the proposed approximately 1,400-step
endpoint). Fixed training cohort: 2,048 indices sampled without replacement from
the training split with seed 20260928; fixed validation cohort: 1,024 indices from
the validation split using the same generator after the training draw. Freeze
their exact identities before launch. Evaluate without augmentation, batch16,
saved-stat BN for both cohorts. Evaluation must preserve parameters, BN buffers,
training RNG and optimizer state. Training-cohort examples remain part of the full
epoch; only the validation cohort is excluded from updates. No official test read.

Primary: intervention-minus-control training-cohort CE at 400 and 1,407 updates.
Report all four measurements, validation CE/accuracy and online training metrics.
Exploratory screening rule, frozen before execution: at both 400 and 1,407 updates,
require at least 1% lower training CE, with validation CE no more than 1% higher
and validation accuracy no more than 0.5 pp lower. The starting measurements must
match exactly in CE/accuracy with identical starting model and Adam state. Failure of a threshold does not
prove an optimal LR or distinguish all causes; validation cohort uncertainty and
one seed/checkpoint limit the conclusion. Passing makes a 3–5-epoch extension
worth proposing, not automatic. Stop on nonfinite, identity or runtime failure.

Root registered before creation: `results/cifar10-conv-lr-one-epoch-20260928-v1/`.
Native scientific runner: `experiments/screen_cifar_l8_conv_lr.py`; frozen source,
inputs/cohorts, exact queue command and identities live under `source/`, `inputs/`
and `launch/job.json`. Execution smoke uses both branches, two real batch32 steps
and evaluation before/after on 16 examples per cohort. No solver/rho audits.

Target Trex RTX5090; same-device sequential branches consume one GPU. Expected
30–45 minutes, 90 GPU-minute cap including smoke/validation, 10-minute CPU
collection allowance. Peak bound 12,288 MiB plus 2,048 MiB headroom; allocator
fraction 0.50. Respect the weekday two-5090 cap and standing Ben sharing policy.
Cutoff September29 08:00 Paris. No automatic retry or further epochs.

Execution and monitoring handoff: shared run-watch queue is sole monitoring owner,
job `cifar-conv-lr-one-epoch-20260928`. Inspect with `gpu_queue.py status` and that ID;
`source/queue_worker.py collect` collects native output, validates coverage and
produces the response plot/summary. The coordinating agent owns scientific review
and failure diagnosis. Current handoff: both branches collected, validated and scientifically reviewed.
Queue attempt `54b4ffed2ae342d7a0161c79fea77014` completed on Trex within budget
(1,876.44 seconds including smoke/target validation). Full epoch-31 checkpoints
retained; no extension launched. See [result](../results/exp-014-one-epoch-conv-lr.md).
