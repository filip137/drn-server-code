---
experiment: "exp-001"
evidence: "validated-local"
summary: "Reducing beta restores CIFAR initialization gradient alignment at unchanged T/K; extremely small beta degrades block 2 early-layer agreement."
verdicts: {"H-001": "supports"}
---
# Result: smaller-beta cosine sweep

All7 multipliers x8 Conv weights x8 batches completed on Loulou RTX5090. Compute129.3s, charged131.3s, peak reserved10.21GiB. The original256 training images, initializer, batch32, no noise/augmentation and T/K=[6,6,4] were retained. Model/BN tensors are unchanged; every reference/estimate is finite and non-dead. The1x anchor matches the previous validated replay within1e-5 relative /1e-6 absolute tolerance. CPU collection validated the canonical bundle and complete coverage.

Minimum native-BPTT cosine across the Conv weights in each block, after averaging gradients over256 examples:

| B / calibrated B | Block 1, 3 conv | Block 2, 3 conv | Block 3, 2 conv |
|---:|---:|---:|---:|
| 1 | -0.1357 | -0.0461 | 0.7041 |
| 0.3 | -0.0928 | 0.0719 | 0.9616 |
| 0.1 | 0.4326 | 0.7017 | 0.9913 |
| 0.03 | 0.9039 | 0.9421 | 0.9980 |
| 0.01 | 0.9987 | 0.9522 | 0.9995 |
| 0.003 | 0.9992 | 0.8752 | 0.9992 |
| 0.001 | 0.9967 | 0.6604 | 0.9957 |

At0.01x, per-block B values are [0.0034178884,0.00063142913,0.0032483228], and relative output RMS displacements are [0.05708,0.05727,0.01469]. Native cosines range0.99875–0.99884,0.95220–0.99863 and0.99948–0.99958 respectively. Matched-local BPTT confirms the same conclusion. These are explicit injected B values under the frozen summed-CE convention.

The largest tested multiplier meeting pooled cosine>=0.90 for every Conv in a block is0.03 for blocks1/2 and0.3 for block3 using the native reference. If requiring both native and matched-local references, block1 needs0.01: its first-Conv local cosine at0.03 is0.8971, close to the threshold. This descriptive cutoff is not a training qualification.

Review: supports H-001 in the tested initialization regime. Lowering B alone, with T/K unchanged, restores close gradient directions; excessively large nudging accounts for much of the initial disagreement. Performance on MNIST does not establish CIFAR training tolerance, and cosine alone does not decide which beta learns best. At0.001x block2 first-Conv cosine falls to0.6604 (batch median0.7844), versus0.9522 (median0.9895) at0.01x. Thus smaller is not uniformly better. Weak-response numerical resolution is a possible explanation, not established by this sweep.

Next: [exp-002](../experiments/exp-002-ten-epoch-beta.md) plans10-epoch learning comparisons at1x,0.03x and0.01x. These span the original large-beta setting and two better-aligned settings; they are candidates, not a proven training optimum. Define the beta-dependent EqProp training update before running them.

Evidence: `results/cifar-block-beta-sweep-20260929-v1/run/`, `analysis/summary.json`, `analysis/batch_summary.json` and the exact source/config identity. No failures or excluded cases in this sweep; predecessor failed/superseded attempts remain in the original pilot.

[Beta/cosine/displacement figure](../../../../../results/cifar-block-beta-sweep-20260929-v1/analysis/blockwise_beta_sweep.jpg)
