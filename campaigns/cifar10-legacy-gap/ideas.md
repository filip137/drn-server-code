# Idea map — CIFAR-10 legacy gap

This map records alternatives and priority, not a duplicated outcome table.

September 24 exploration priority: qualify the training horizon before treating
epoch 50 rankings as settled. Training CE is still falling, but validation CE and
accuracy need separate interpretation. The three new ideas below are motivated
by existing observations and are now registered for the bounded
[horizon series](series/003-training-horizon/README.md). Design and GPU allocation
are prepared; subsequent user authorization schedules the six-case study for
September24 at22:00 Europe/Paris, subject to preflight and resource gates.
Source/config auditing and paired-seed replication remain necessary; extending
one trajectory cannot establish a general scheme ranking.

| Idea | Priority / maturity | Reasoning | Claim |
|---|---|---|---|
| Larger batches speed up ours with acceptable validation accuracy | Completed batch64 run; reference cancelled; [exp-018](series/002-residual-gap/experiments/exp-018-batch-size.md) | Measure capacity, wall time and epoch10 accuracy with fixed LRs and trainable BN | [H-012](hypotheses/H-012-batch-size.md) |
| Readout-rate candidates transfer to fresh training with trainable BN | Cancelled at user request; no complete10-epoch comparison; [exp-017](series/002-residual-gap/experiments/exp-017-fresh-readout-rates.md) | Doses selected in the short frozen-BN screen; transfer of their benefit is untested | [H-011](hypotheses/H-011-fresh-readout-rates.md) |
| Legacy's dense readout adapts too quickly relative to its convolution features | Completed 400-update screen, frozen BN; H-010 inconclusive; [exp-016](series/002-residual-gap/experiments/exp-016-dense-readout-lr.md) | [X-002](explorations/X-002-optimization-and-regime.md); H-009's one-step evidence does not establish sustained learning | [H-010](hypotheses/H-010-frozen-bn-readout-rate.md) |
| Useful validation gains remain beyond epoch 50 | exp-006 planned: four unchanged 50→70 continuations | [X-003](explorations/X-003-training-horizon.md) | [H-006](hypotheses/H-006-useful-late-learning.md) |
| Legacy's deficit partly reflects slower convergence and shrinks with more epochs | exp-006 reuses the same four cases; compare gaps to both comparators | [X-003](explorations/X-003-training-horizon.md) | [H-007](hypotheses/H-007-legacy-late-catchup.md) |
| The learning-rate floor limits late progress | exp-007 designed: two paired renewal branches; runner support implemented, preflight pending | [X-003](explorations/X-003-training-horizon.md) | [H-008](hypotheses/H-008-lr-floor-limits-progress.md) |
| Effective BN epsilon alone accounts for the final deficit | Historical test; audit rather than repeat | [X-001](explorations/X-001-coordinate-bn-equivalence.md) | [H-001](hypotheses/H-001-epsilon-only.md) |
| Normalize coordinates and map the complete baseline optimizer | Strong existing short-window control; extend only for a new decision | [X-001](explorations/X-001-coordinate-bn-equivalence.md) | [H-002](hypotheses/H-002-mapped-training.md) |
| The endpoint deficit survives new paired seeds | High-priority unresolved premise | [X-002](explorations/X-002-optimization-and-regime.md) | [H-003](hypotheses/H-003-reproducible-gap.md) |
| Within matched coordinates, optimizer choice accounts for a remaining endpoint deficit | Bounded control, not global LR optimization | [X-002](explorations/X-002-optimization-and-regime.md) | [H-004](hypotheses/H-004-optimizer-transplant.md) |
| Learning affine BN selectively harms legacy | Historical negative evidence; preserve | [X-001](explorations/X-001-coordinate-bn-equivalence.md) | [H-005](hypotheses/H-005-affine-bn-harm.md) |
| Short-horizon selection misses long-horizon LR/epsilon optima | Parked until matched comparison/reproducibility clarified | [X-002](explorations/X-002-optimization-and-regime.md) | Not registered |
| Blockwise digital boundaries remove an advantage of fully coupled networks | Regime explanation; topology test would be a new series | [X-002](explorations/X-002-optimization-and-regime.md) | Not registered |
| Solver error, raw-gradient overshoot or weak scalar gain adaptation | Existing checks constrain, but do not universally eliminate | [X-002](explorations/X-002-optimization-and-regime.md) | Not registered |
| Environment, stochastic trajectories and train/validation behavior | Audit/control first | [X-002](explorations/X-002-optimization-and-regime.md) | Not registered |

September27: check classifier-versus-convolution update imbalance through [H-009](hypotheses/H-009-classifier-update-imbalance.md) and [exp-009](series/002-residual-gap/experiments/exp-009-classifier-update-probe.md), a bounded read-only BPTT diagnostic.
