---
experiment: "exp-021"
evidence: "validated-local"
summary: "Lower ceiling with larger block3 target reaches28.92%; the ceiling effect reverses across targets and block3 misses its target on1116/1407 updates."
verdicts: {"H-016": "contradicts"}
---
# The lower ceiling and larger target do not combine beneficially

The fresh ours epoch completed and passed collection validation:1,407 updates,
45,000 training examples,5,000 validation examples,3,233.2seconds. No retry,
injected noise or official test read. Runtime and scientific controls match
the three preceding cells. Values below are epoch1 validation accuracy / CE.

| Block3 relative RMS target | Beta ceiling10,000 | Beta ceiling30,000 |
| --- | ---: | ---: |
|0.001|32.64% /1.788320 (exp015)|28.04% /1.995131 (exp017)|
|0.005|28.92% /2.149137 (exp021)|32.88% /1.945375 (exp018)|

Matched native BPTT is37.06% /1.726060. Exp021 fails both unchanged gates,
accuracy>=35.06% and CE<=1.812363. The larger target hurts at the lower
ceiling (-3.72pp), whereas it helped at the higher ceiling (+4.84pp).
Likewise, raising the ceiling hurts for target0.001 but helps for0.005.
H-016's proposed benefit at the lower ceiling is contradicted in this seed;
none of the four settings qualifies as BPTT-like training.

This compares requested controller settings, not equal achieved displacement.
Median actual/target RMS is[0.9975,0.9875,0.1413]. Tolerance misses are
[0,38,1116] out of1,407 updates, with ceiling hits[0,106,1148]. Block3
therefore substantially undershoots its requested target on this trajectory.
Average centered-pair counts are[1.490,1.316,1.070]; beta ranges are
[0.00125665,5763.6383], [0.000312568,10000], [0.000922972,10000].
The measured interaction does not establish finite-beta bias, cancellation,
or solver error as the cause. Neither uniformly increasing nor uniformly
decreasing the ceiling is supported as a remedy.

The stored final fixed-cohort diagnostic has a block2 layer cosine of-0.420
and norm ratio4.909 against native BPTT. That diagnostic uses warm-start beta
without the training minibatch search; it is not a measurement of the actual
gradient supplied to Adam and does not prove sustained training misalignment.

Evidence: `results/cifar-beta-current-batch-low-cap-block3-rms-20260930-v1/`,
especially `analysis/current_batch-1-collection.json`,
`analysis/controller-summary.json` and `cells/1/current_batch/metrics.jsonl`.
No continuation or new test: Filip stopped new launches at09:00 Paris.
