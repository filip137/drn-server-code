# X-001 — Coordinate scaling is not the same as a matched training procedure

[Existing review](../../../docs/cifar_experiment_review_20260924.md) reports that
baseline and legacy both have A*B=1, with matched-weight block-output unit factors
16/16/4. At BN epsilon=1e-5 the effective legacy epsilons in baseline units are
[1e-5/256,1e-5/256,1e-5/16]. Boundary normalization therefore changes the interpretation
of amplification relative to a physically coupled network without those boundaries.

Dividing legacy block outputs before BN restores baseline effective epsilon while
keeping its physical KCL equations. But matching this forward transformation alone
need not match the optimizer, BN initialization, gains, complete state, or stochastic
training history. Two existing observations must be kept distinct:

The 200-minibatch identity experiment maps normalized legacy to the **complete baseline
optimizer** and reports identical logits, gradients, BN state and updates in its tested
window. The 50-epoch normalized legacy comparison retains the **legacy-selected
optimizer** and reaches 88.82%, versus raw legacy 88.72%. Those experiments do not
contradict each other: they isolate different collections of factors.

Early epsilon effects depend strongly on optimizer choice: under baseline rates,
75.26% becomes 70.18% with smaller effective epsilon; under legacy rates, 75.82% becomes
74.68%. These are not a universal epsilon penalty or proof of a global optimum.
At epoch50 the epsilon-only correction does not repair the observed deficit.

Affine BN freezing and removing BN are different interventions. Freezing affine BN
hurts every scheme at tested rates. Removing BN changes scale and dynamics so sharply
that two runs collapse and the third fails solver qualification. That invalid no-BN
comparison cannot establish a causal legacy disadvantage. Likewise, BN running-stat
recalibration changes inference estimates, not the training trajectory, and its measured
legacy endpoint accuracy falls rather than recovering.

[H-001](../hypotheses/H-001-epsilon-only.md) and [H-005](../hypotheses/H-005-affine-bn-harm.md)
retain already tested explanations. [H-002](../hypotheses/H-002-mapped-training.md) asks
whether the mapped training control extends beyond its demonstrated short window.
