---
experiment: "exp-001"
evidence: "imported-summary"
summary: "At seed0 epoch50, baseline / ours / legacy score 89.48 / 89.92 / 88.72%; epsilon-normalized legacy reaches 88.82%."
verdicts: {"H-001": "contradicts", "H-003": "not-tested", "H-004": "not-tested"}
---

# Imported result: exp-001

Source: [September 24 consolidated review](../../../../../docs/cifar_experiment_review_20260924.md),
commit 6be7d41038018b225cef7d1771fdb905084b0db7. No raw local bundles were rechecked.

| Condition | Epoch10 val | Epoch30 val | Epoch50 val | Validation CE50 | Training CE50 |
|---|---:|---:|---:|---:|---:|
| Baseline | 75.26% | 86.54% | 89.48% | 0.368819 | 0.060967 |
| Ours | 75.36% | 87.20% | 89.92% | 0.372177 | 0.036016 |
| Legacy | 74.68% | 86.88% | 88.72% | 0.385642 | 0.071951 |
| Normalized legacy, legacy optimizer | 75.82% | 86.44% | 88.82% | 0.390240 | 0.060160 |

These results contradict H-001's epsilon-only sufficiency in the tested setting.
They do not test H-003 across seeds, nor H-004's complete optimizer mapping. Legacy's
relative position changes between epochs30 and50. Baseline and ours have different
accuracy/CE orderings. Per-arm best checkpoints and correlated final-ten-epoch averages
must not replace the primary common endpoint or supply fake independent samples.

Keep the original scope/environment/selection limitations. The broader explanation
remains open; there is no newly established global LR/epsilon optimum.
