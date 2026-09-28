---
experiment: "exp-001"
evidence: "imported-summary"
summary: "At one initializer, injected betas 88.7 / 1.385 / 0.02173 match unit output RMS within 0.031%; training untested."
verdicts: {"H-001": "supports"}
---

# Imported result: exp-001

Source: [September 22 RMS note](../../../../../docs/conv3_unit_output_displacement_beta_20260922.md),
read at commit 6be7d41038018b225cef7d1771fdb905084b0db7.

| Scheme | Injected beta | Pooled positive-free RMS |
|---|---:|---:|
| Baseline | 88.7 | 1.0003009717 |
| Ours | 1.385 | 0.9998121959 |
| Legacy | 0.02173 | 0.9998813740 |

The source says all 1,728 projected-KKT residual checks passed; it preserves an initial
OOM and the successful scientific-config-preserving retry. This is reported historical
validation, not validation rerun by the campaign setup. The result supports H-001 only
for the stated initializer/cohort. It does not test noisy-training stability, equality
of hidden-layer perturbations or transfer to other seeds. Reuse rather than duplicate
the calibration; source audit comes next.
