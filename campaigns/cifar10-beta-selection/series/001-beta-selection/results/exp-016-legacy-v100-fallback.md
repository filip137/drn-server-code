---
experiment: "exp-016"
evidence: "validated-local"
summary: "Legacy0.1x finishes epoch5 at57.00% versus native62.38%, failing accuracy and CE.0.0001x finishes48.08%;1x remains running; all new launches and extensions are stopped by user."
verdicts: {"H-010": "inconclusive"}
---
# Legacy candidate qualifications

All nine fresh V100 cases are collected and locally validated:1,407 updates,
45,000 training examples and5,000 validation examples each, matching runtime
torch2.5.0/CUDA12.2/cuDNN8907. Compare within this cohort; the3090 control is
not substituted. Numerical settings, initializer and original legacy rates
remain those declared in exp-013/016. No read noise or official test use.

| Beta multiplier | Accuracy | CE | Epoch1 gate |
| --- | ---: | ---: | --- |
| Native BPTT |33.22%|1.797997|Reference |
|0.0001|32.54%|1.846556|Pass |
|0.001|27.96%|1.905440|Fail |
|0.01|30.98%|1.822212|Fail accuracy |
|0.03|29.14%|2.073501|Fail |
|0.1|33.70%|1.807540|Pass |
|0.3 (additional screen)|30.54%|2.015569|Fail both |
|1 (additional screen)|32.16%|1.860963|Pass |
|3|25.80%|2.234216|Fail |

The unchanged limits are accuracy>=31.22% and CE<=1.887897. The original
fallback selected0.1x, then0.0001x by CE; the later1x screen also passes.
Their near-BPTT first epochs
justify a longer test, not a claim of high final accuracy, statistical advantage
or read-noise robustness. H-010 remains inconclusive for long-run transfer and
the full all-scheme question.

The persistent owner submitted exact full-state continuations through epoch5:
native `340008`,0.1x `340009`,0.0001x `340011`. Each retains14,400seconds and
the existing account/runtime/contract. Subsequent promotion still requires the
matched milestone gate. Parent checkpoint hashes are checked before submission;
collection will verify the restored state and full coverage.

Native epoch5 has now passed full-state collection:7,035 total updates,
**62.38% accuracy, CE1.055644**,9,433.0s for the four-epoch continuation.
The unchanged EqProp promotion limits are accuracy>=60.38% and
CE<=1.108426657.0.1x epoch5 is now collected at **57.00%, CE1.218064**,
7,035 total updates and12,763.6s for the continuation. Accuracy is5.38pp
below native and CE15.39% higher: both checks fail.0.0001x is also now
collected at **48.08%, CE1.467040**,7,035 updates and13,043.5seconds. It
misses native by14.30pp and has38.97% higher CE; both checks fail.
Its Slurm job340011 completed with exit0:0 and collection validated.
Only1x (343367) remains running through its assigned epoch5 endpoint. Filip stopped new launches
and extensions at09:00 Paris; the owner only collects existing jobs.
Evidence: root `analysis/beta_0p1-5-collection.json` and
`analysis/beta_0p0001-5-collection.json`.
The separately funded0.3x/1x screens have now passed collection, each1,407
updates and5,000-image validation.1x is1.06pp below native with3.50% higher
CE, passing both epoch1 limits.0.3x fails both. Main allocates the unused
original1x epoch5 allowance to test its longer trajectory, explicitly amending
the prior scheduling wait. The handoff records that change and the unchanged
two-candidate cumulative cap beyond5. This remains single-seed selection evidence.

The native's earlier node-prolog failure remains a separate184-second
operational charge; its recovered initial run completed within the reduced
allowance. No scientific case failed numerically. Evidence and decision:
`results/cifar-eqprop-legacy-v100-20260930-v1/analysis/*-collection.json` and
`analysis/decision.json`; Slurm handles remain in `monitor/jobs.json`.
