---
experiment: "exp-002"
evidence: "imported-summary"
summary: "p99 legacy leads ours in the reported seed-0 run; T gains are checkpoint-specific and readout cosine hides weak early-layer alignment."
verdicts: {"H-002": "not-tested", "H-003": "inconclusive", "H-004": "supports"}
---

# Imported result: exp-002

Source: [beta synthesis](../../../../../docs/beta_study.md), read at
6be7d41038018b225cef7d1771fdb905084b0db7. Underlying local bundles were not accessed.

At 30 epochs, sigma=5e-4 and T=K=8, p99 legacy beta=0.1 reaches 96.86% final validation;
ours beta=0.987333678708 reaches 96.04% (best 96.32%). This is one seed and is not a
matched output-RMS-policy comparison, hence it does not test H-002.

The p99 ours ten-epoch repeat yields 95.44% at T8/K8 versus 95.38% at T16/K8, with no
meaningful matching p99-trained cosine gain. In contrast, older p90-trained ours at
beta=5.26875648112 has readout cosine -0.29315 at T8 and 0.68455 at T16. Initialization
and current p99 replays are nearly T/K-insensitive. This motivates H-003, but the exact common-environment/shared-beta claim remains
inconclusive: environments and comparison grids differ, and the proposed thresholds
were not preregistered historically. exp-004 supplies that discriminating test.

At current p99 ours epoch30 the noisy layer medians are -0.005645, 0.008845, 0.631672,
0.999948. This supports H-004's coexistence observation, not a causal accuracy theory.
All reported values are imported from the synthesis; do not infer new validation or
statistical significance. The original source retains the constituent reports and
qualification limitations, including the changing BPTT reference in older K sweeps.
