---
experiment: "exp-001"
evidence: "imported-summary"
summary: "Conv2 target amplitude reverses the small ours/legacy ranking; voltage/encoding is a plausible explanation, with causal magnitude unresolved."
verdicts: {"H-001": "supports", "H-002": "inconclusive"}
---
# Clean Conv2: encoding changes the observed ranking

Imported September 29 from the completed September 28 pilot. Its original
review reports six complete, locally validated 30-epoch runs with no failures,
exclusions or retries. This import reuses that validation; it is not a new audit.

Validation accuracy (%), seed 0, clean BPTT:

| Target | Baseline best / final | Ours best / final | Legacy best / final |
|---|---:|---:|---:|
| 1/16 | 98.28 / 98.10 | **98.42 / 98.28** | 98.38 / 98.22 |
| 1/4 | 97.98 / 97.90 | 98.36 / 98.24 | **98.40 / 98.34** |
| 1 (historical) | 97.36 / 97.36 | 98.10 / 98.08 | 98.30 / 98.08 |

![Best validation accuracy versus target amplitude](figures/accuracy_vs_target.jpg)

[Exact measurements, losses and original run paths](figures/target_encoding_accuracy.csv).
The hollow target-1 markers are historical references, not newly matched controls.

Filip cites this clean intervention as evidence that legacy's advantage depends
largely on voltage magnitude/encoding. The measured result establishes **encoding
sensitivity** under the fixed recipes: ours leads legacy by 0.04 points in best
accuracy at 1/16, while legacy leads by 0.04 points at 1/4. Final gaps are +0.06
and -0.10 points. Baseline gains 0.30 best-accuracy points from 1/4 to 1/16.

The best-accuracy differences each correspond to two predictions out of 5,000
validation examples, with each best epoch selected separately. They support
H-001 for this pilot, not a robust population ordering. The broader claim that
voltage/encoding explains most of the advantage remains H-002: one seed,
unnormalized loss scaling, scheme-specific rates and fixed [0,100] bounds do
not isolate that causal contribution. No official-test evidence is claimed.

Original paths and copied-file hashes are in [provenance](figures/provenance.json).
Retain the completed evidence and pursue matched controls only after assignment.
