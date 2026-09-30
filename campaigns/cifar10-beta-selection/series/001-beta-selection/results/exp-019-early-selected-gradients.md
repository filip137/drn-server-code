---
experiment: "exp-019"
evidence: "validated-local"
summary: "Exact101-update replays show a block3 cosine dip at20 but otherwise strong sampled agreement; raising the beta ceiling improves RMS tracking without improving the100-update block2 gradient comparison."
verdicts: {"H-015": "inconclusive"}
---
# A transient gradient discrepancy, not evidence of persistent collapse

Both frozen trajectories reproduced all101 original per-update losses exactly.
All10 next-minibatch probes passed collection validation, with Adam-input
gradients captured before the unchanged update and parameter deltas measured
after projection. Source, initialization, data order, augmentation, optimizer,
BN and RNG were retained. The local3090 job used473.24seconds of scientific
runtime and478.324seconds of queue allowance; peak allocation10.37GiB.
No new full validation, noise evaluation or official test read was performed.

The two trajectories have identical probe results at0/1/20/50 completed updates.
The table reports selected EqProp cosine against native finite-T/K BPTT;
each probe measures the next actual augmented training minibatch.

| Completed updates | Block1 | Block2 | Block3 |
| --- | ---: | ---: | ---: |
|0, both |0.9997|0.9828|0.9878|
|1, both |0.9997|0.9994|0.9805|
|20, both |0.9929|0.9982|0.6780|
|50, both |0.9805|0.9965|0.9927|
|100, ceiling10,000 |0.9985|0.9972|0.9965|
|100, ceiling30,000 |0.9966|0.9929|0.9964|

At20, block3 cosine against reset-free-T is also low,0.6814, and its gradient
norm is1.408 times native. The discrepancy therefore persists against the
hybrid's differentiation path; it is not removed by switching references.
At50 it recovers to0.9942 against free-T. Direct free-T/native cosine was not
measured and cannot be reconstructed from the paired summary comparisons.

At100, raising the ceiling changes selected block2 beta10,000→18,082 and
relative RMS0.01687→0.02949 toward the0.035 target. Nevertheless, block2
cosine(native/free-T) changes0.99724/0.99728→0.99288/0.99247 and the gradient
norm ratio against native changes0.979→0.915. Across the101-update prefixes,
block2 tolerance misses fall43→0. This fixes target tracking without improving
that sampled gradient comparison. The trajectories have different parameters
by this point, so this is not a same-state causal beta comparison.

Digital gradients agree with free-T to numerical precision at every probe;
their pooled native cosine is at least0.99990. The first projected Conv update
has norm0.6123 times the initial Conv parameter norm, compared with0.01450
for the head,0.001414 for BN and5.34e-7 for raw gain parameters. These are
within-family relative changes, not directly interchangeable physical units.
No counterfactual native Adam delta was measured. Large initial Conv movement
and high gradient cosine do not establish either an optimizer defect or the
cause of the validation gap.

H-015 remains inconclusive as an explanation of failed learning: an early
block3 discrepancy is observed, but the five sampled batches do not show
persistent gradient collapse. Optimization sensitivity remains a hypothesis
requiring a matched intervention rather than another inference from cosine.

Evidence root: `results/cifar-beta-early-selected-gradients-20260930-v1/`.
`cells/replay/metrics.jsonl` contains actual beta trials, layer and pooled
comparisons, update norms and exact losses; `cells/replay/checkpoints/` retains
both full101-update states. `analysis/collection.json` records successful queue
and strict CPU validation; `analysis/budget_transfer.json` records the unchanged
combined141.5GPU-hour ceiling. No retry or coverage exception occurred.


## Individual convolution layers

[Ceiling10,000: per-layer cosine and norm](../../../../../results/cifar-beta-early-selected-gradients-20260930-v1/analysis/per-layer-early-gradients-exp015.jpg)
and [ceiling30,000](../../../../../results/cifar-beta-early-selected-gradients-20260930-v1/analysis/per-layer-early-gradients-exp017.jpg)
show each Conv separately at the five recorded probes. Adjacent CSV/JSON retain
unrounded values and source paths. These figures use existing validated records;
no additional simulation or checkpoint replay was performed.

At20 completed updates, block3 Conv1/Conv2 cosines are0.5315/0.6788 and norm
ratios1.402/1.408; the pooled0.6780 conceals the lower first-Conv agreement.
Both convolutions are affected, although the block's output relative RMS0.000983
closely matches0.001. At50, block1 Conv3 cosine0.9520 accompanies norm ratio1.302,
illustrating why direction and magnitude must both be retained. These remain
isolated minibatch observations, not a measured full-epoch error frequency.
The lower figure rows are output-state RMS only; internal-layer displacement
was not recorded and is not inferred from the weight-gradient measurements.
