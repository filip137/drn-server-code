---
experiment: "exp-030"
evidence: "validated-local"
summary: "Masking inactive-channel force is exact and restores block3 where those channels carry most of the force (ours e1/e5, legacy e1, BPTT legacy e10), but not in baseline e1 or legacy e5; no common float32 target exists in any block"
verdicts: {"H-025": "contradicts"}
---
# Result: the mask helps only where exactly-dead channels dominate

## Evidence and coverage

All 36 jobs (18 checkpoints × masked/control) completed and passed the queue
collection hook (`analysis/<case>-<arm>-collection.json`): canonical bundles,
config/environment identity (PyTorch2.11.0+cu128), float32 precision, unchanged
checkpoint/model/BN, full layer/minibatch coverage, every target within 3%.
Root `results/cifar-force-mask-float32-replay-20261001-v1/`; analysis
`analysis/summary.json` and [worst-layer figure](../../../../../results/cifar-force-mask-float32-replay-20261001-v1/analysis/worst_layer_cosine.jpg),
reproduced by `python -m experiments.run_cifar_force_mask_replay analyze`.
Placements were the local RTX3090 and Trex RTX5090 as they became free.

## Measurements

**Criterion 1 (exactness): passes.** Over every masked block/layer/minibatch, the
reference recomputed with the masked force has minimum cosine 1−4.0e-13 and
norm ratio error 0.

Block3, upper end of the contiguous run with every layer's pooled local
cosine ≥0.99 (float32 masked / float32 unmasked / full64 unmasked exp-025),
with inactive channels per batch and their share of Σf²:

| Checkpoint | Inactive (share) | Masked | Control | Full64 |
|---|---|---:|---:|---:|
| eqprop-ours-e1 | 4 (1.000) | **2e-2** | 3e-5 | 3e-5 |
| eqprop-ours-e5 | 28–30 (0.986) | **2.5e-3** | 2.5e-4 | 2.5e-4 |
| eqprop-legacy-e1 | 1 (0.999) | **1e-2** | none | 3e-5 |
| bptt-legacy-e10 | 24–25 (0.924) | **2.5e-3** | 5e-4 | 5e-4 |
| eqprop-baseline-e1 | 21–23 (0.927) | none (best 0.9895 at 1e-4) | none | 3e-5 |
| eqprop-legacy-e5 | 55–57 (0.990) | none (best 0.987 at 2.5e-4) | none | 1e-4 |
| eqprop-baseline-e5 | 85–88 (0.866) | 1e-3 | 1e-3 | 1e-3 |
| BPTT others | 0–38 (≤0.24) | = control | 5e-3–1e-2 or none | 5e-3–1e-2 |

**Criterion 2: fails as frozen.** Recovery to ≥1e-3 with a control below it
occurs in 4 of the 15 checkpoints that have inactive channels. In 9 the
inactive channels carry ≤24% of the force and masked equals control. In
eqprop-baseline-e1 and eqprop-legacy-e5 they carry ≥93% but block3 still
collapses above 1e-4–2.5e-4.

**Criterion 3: fails.** No sampled target passes the strict criterion at every
checkpoint of either training group, in any block. Blocks1/2 also lack float32
passing targets at late BPTT checkpoints (best pooled worst-layer cosine
0.95–0.989 near 1e-2–3.5e-2) and block2 at all three initializations (best
0.975–0.977), whereas full64 passes there from 1e-6.

## Interpretation

The mask is an exact estimator change and removes the dominant failure in
the four checkpoints where exactly-dead channels hold nearly all the force,
including the two ours checkpoints that motivated it. It is not a general
fix. Baseline e1 and legacy e5 still lose block3 at small displacement despite
masking most of the force, so another switching population must remain;
near-dead low-variance channels (large but finite BN gain, not masked by an
exact-zero test) are the leading untested candidate. Separately, float32 itself
is inadequate for blocks1/2 at late BPTT checkpoints and block2 at
initialization, independent of the mask. H-025's claim of recovery across
checkpoints and a common float32 target is therefore contradicted in this
scope, while its exactness criterion holds.

## Decision

Keep the mask as an option; it is exact and helps where its assumption holds.
Do not treat float32 as sufficient across training histories. Training
([exp-031](../experiments/exp-031-float32-masked-training.md)) continues as
declared on the ours trajectory. A discriminating follow-up would mask channels
by BN gain (or all units off in the free state) at baseline e1/legacy e5 and
check whether block3 recovers.
