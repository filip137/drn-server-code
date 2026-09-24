# Completed bounded Conv1 pilot checks

All nine bounded Conv1 EqProp seed-0 pilots are collected and pass the declared full-training stability rule. Filip authorized seeds 1/2 on September 12 after each complete three-ceiling group passes; these three groups are released within the existing 300 GPU-hour limit. Conv2/3 pilots remain in progress or scientifically held. These are validation measurements; the official test split remains unread.

![Conv1 bounded seed-0 validation curves](figures/bounded_conv1_seed0_pilot_validation.png)

Curves use the matched shared initializer, split, epoch order, and frozen learning rates. Every panel shows ten epochs at T=K=4. Baseline uses injected beta 0.1; ours and legacy use 0.03, each shared across all three ceilings. Dashed curves show matched BPTT. Vertical scales vary by panel to make each trajectory visible.

| Scheme | Gmax | Best validation | Final validation | Drop | Result |
|---|---:|---:|---:|---:|---|
| baseline | 1e-4 | 89.68% | 87.24% | 2.44 pp | [bundle](bundles/table3_bounded_ep/conv1/baseline/gmax_1em4/seed0/result.json) |
| baseline | 5e-4 | 94.42% | 93.82% | 0.60 pp | [bundle](bundles/table3_bounded_ep/conv1/baseline/gmax_5em4/seed0/result.json) |
| baseline | 1e-3 | 94.80% | 94.24% | 0.56 pp | [bundle](bundles/table3_bounded_ep/conv1/baseline/gmax_1em3/seed0/result.json) |
| ours | 1e-4 | 92.92% | 91.50% | 1.42 pp | [bundle](bundles/table3_bounded_ep/conv1/ours/gmax_1em4/seed0/result.json) |
| ours | 5e-4 | 95.00% | 94.60% | 0.40 pp | [bundle](bundles/table3_bounded_ep/conv1/ours/gmax_5em4/seed0/result.json) |
| ours | 1e-3 | 95.00% | 94.70% | 0.30 pp | [bundle](bundles/table3_bounded_ep/conv1/ours/gmax_1em3/seed0/result.json) |
| legacy | 1e-4 | 94.94% | 94.20% | 0.74 pp | [bundle](bundles/table3_bounded_ep/conv1/legacy/gmax_1em4/seed0/result.json) |
| legacy | 5e-4 | 96.28% | 96.18% | 0.10 pp | [bundle](bundles/table3_bounded_ep/conv1/legacy/gmax_5em4/seed0/result.json) |
| legacy | 1e-3 | 96.32% | 96.26% | 0.06 pp | [bundle](bundles/table3_bounded_ep/conv1/legacy/gmax_1em3/seed0/result.json) |

The largest drop is 2.44 percentage points in baseline at Gmax=1e-4, below the predeclared strict 5-point limit. This satisfies the stability gate; it does not establish accuracy equivalence to BPTT or a three-seed performance estimate.

Across all nine matched trajectories (90 epoch pairs), the largest absolute EqProp–BPTT validation difference is 0.08 percentage points. Every pair has matching train/validation cohorts and epoch order. The final decline at the tight ceiling appears in both algorithms; these pilot data do not indicate a beta-specific divergence. This is a single-seed observation.
