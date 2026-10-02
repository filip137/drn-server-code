---
experiment: "exp-002"
evidence: "imported-summary"
summary: "Completed controls limit epsilon-only, affine-BN harm and simple solver/overshoot stories; mapped baseline/legacy identity is demonstrated only for its tested prefix."
verdicts: {"H-001": "contradicts", "H-002": "inconclusive", "H-004": "not-tested", "H-005": "contradicts"}
---

# Imported result: exp-002

Primary source: [September 24 review](../../../../../docs/cifar_experiment_review_20260924.md),
commit 6be7d41038018b225cef7d1771fdb905084b0db7. All statements below describe reported
historical coverage; local files and validation outputs were not inspected here.

The complete baseline optimizer plus matched normalized-legacy coordinates yields
identical logits, gradients, BN state and updates over 200 real minibatches. This is
strong short-window evidence, but H-002's prospective long-horizon extension remains
inconclusive and H-004's endpoint comparison is not tested by the prefix.

Frozen affine BN lowers baseline/ours/legacy ten-epoch accuracy by 4.54/1.60/1.66pp,
respectively. Legacy moves from 74.68% to 73.02%, contradicting H-005 in this scope.
Removing BN causes baseline/legacy collapse; ours reaches 52.08% but fails solver
qualification. These are negative/incomplete results, not a successful BN-free ranking.
Recalibrated BN inference statistics change legacy epoch50 accuracy from 88.72% to
88.34%, not a repair of the final deficit.

The review reports 136 synthetic checks including 21 CIFAR-architecture checks, and
native solver qualification at 12 original checkpoints. It finds no systematic legacy
gradient collapse on that cohort. At nine saved-Adam checkpoints, all 36 sampled
directions improve same-batch loss at half/nominal/double steps. This constrains sampled
overshoot, not every future step or generalization behavior.

Original gains change at most 0.353%. Matched fixed/log-gain frozen-BN legacy controls
reach 68.00/69.72%, below the historical softplus reference of 73.02%. The interventions
and references do not establish a unique gain mechanism. Epsilon correction alone
also fails to close the final gap; preserve optimizer/epsilon interaction as distinct.

Next: audit exact mappings and source comparability before any extension or replication.
