---
experiment: "exp-013"
evidence: "validated-local"
summary: "Both baseline epoch5 candidates miss the combined gate:0.3x fails accuracy/CE;1x finishes58.14% versus native60.02% but fails CE. Ours0.3x epoch5 reaches54.12% versus66.68% native and fails both gates. New launches stopped; legacy1x is still running."
verdicts: {"H-010": "inconclusive"}
---
# All-scheme training: partial evidence

Filip stopped new runs and continuations at September30 09:00 Paris.
Existing jobs may finish and be collected. Baseline1x has just completed its
matched3090 epoch5 at58.14%, CE1.185933 versus native60.02%, CE1.101680:
accuracy gap1.88pp passes, but CE7.65% higher fails. Final full-state collection
passed. Evidence:
`results/cifar-eqprop-baseline-3090-continuation-20260930-v1/analysis/beta_1-5-collection.json`.
No completed epoch5 EqProp candidate currently passes both metrics.

[Learning-curve snapshot, September30 09:14 Paris](../../../../../results/cifar-eqprop-monitor-20260929-v1/validation-comparison-20260930-0914.jpg)
keeps each GPU/runtime cohort with its own control; open markers identify
epochs from stages whose final collection is pending. The adjacent JSON stores
the plotted values and source paths. Ours0.3x's latest plotted epoch4 is50.68%
versus60.70% native. Its subsequently collected epoch5 is54.12% versus66.68%.

Validated full-epoch receipts cover1,407 updates,45,000 training examples and
5,000 validation examples with fresh initialization and unchanged scientific
settings. Compare within each runtime/scheme cohort.

| Cohort | Case | Epoch1 accuracy | CE | Gate |
| --- | --- | ---: | ---: | --- |
| Baseline5090 | Native BPTT |27.92%|2.095195|Reference |
| Baseline5090 | Fixed0.3x beta |32.48%|1.838954|Pass |
| Baseline5090 | Fixed1x beta |34.86%|1.754371|Pass |
| Baseline5090 | Fixed0.1x beta (early fallback) |28.00%|2.214752|Fail: CE |
| Baseline5090 | Fixed3x beta (early fallback) |34.78%|1.822241|Pass; no extension authorized |
| Ours5090 | Native BPTT |36.16%|1.730916|Reference |
| Ours5090 | Fixed0.3x beta |34.14%|1.787529|Fail: accuracy |
| Ours5090 | Fixed1x beta |35.20%|1.867973|Fail: CE |
| Legacy3090 | Native BPTT |36.52%|1.754233|Reference |
| Legacy3090 | Fixed0.3x beta |29.96%|1.863588|Fail |
| Legacy3090 | Fixed1x beta |32.98%|1.863064|Fail |
| Legacy3090 | Fixed0.0001x beta |28.32%|1.929404|Fail |

Baseline1x is6.94pp higher with16.27% lower CE than its matched control. This
supports a longer test of this candidate; one seed and one epoch do not establish
high final accuracy or an advantage over the other amplification schemes.
Baseline0.3x completed after operational recovery:32.48%, CE1.838954 in
1,126.2seconds. Both metrics passed its matched epoch1 gate. At epoch5,
validated collection gives **59.26%, CE1.147454**, versus original5090 native
**62.58%, CE1.049603**: accuracy is3.32pp lower and CE9.32% higher. It fails
both unchanged gates and stops at5. The four-epoch continuation took4,317.2s
and reached7,035 total updates. Await baseline1x's separate matched3090 review
in exp020 before opening a baseline fallback; those controls are not pooled.
Evidence: `results/cifar-eqprop-baseline-5090-20260929-v1/analysis/beta_0p3-5-collection.json`.
The early0.1x screen passes the epoch1 accuracy floor25.92%, but its CE2.214752
exceeds2.199954436, so it stops at1. The strongest observed baseline epoch1
remains1x. The already-started3x screen is now collected at34.78%, CE1.822241,
passing both epoch1 metrics. It does not improve on1x's34.86%, CE1.754371.
This does not establish an optimal beta;3x is not extended after Filip's stop.
Evidence: `results/cifar-eqprop-baseline-5090-20260929-v1/analysis/beta_3-1-collection.json`.
Legacy0.3x fails by6.56pp accuracy and6.23% excess CE. Legacy1x also fails:
3.54pp lower accuracy and6.20% higher CE. Its predeclared six-case fallback is
active in the separate matched V100 cohort in
[exp-016](../experiments/exp-016-legacy-v100-fallback.md). The already-started
3090 fallback0.0001x was allowed to finish and collected at04:51 Paris:28.32%,
CE1.929404 in2,782.7seconds. It fails the same gate and is not extended.
The other unstarted3090 fallback jobs were cancelled only after their V100
replacements had real artifacts. These runtime cohorts are not pooled.
The separate ours5090 native reference completed on loulou on September30:
36.16%, CE1.730916, with1,407 updates and validated full-state
collection. Its successful numerical runtime was805.9seconds; the earlier
failed startup remains charged separately. Fixed0.3x completed in1,138.0seconds
at34.14%, CE1.787529. Its unchanged matched gate is accuracy>=34.16% and
CE<=1.8174618901: CE passes, but accuracy misses by0.02pp, exactly one of
5,000 validation images. Keep this classified as a failed gate. Fixed1x completed
in1,103.5seconds at35.20%, CE1.867973: accuracy passes, CE fails.

An explicit posthoc allocation exception assigns0.3x and its matched native
control to epoch5, within their existing allowances. This tests whether the
one-image boundary miss predicts a sustained learning gap; it does not turn
the epoch1 failure into a pass. The six unstarted fallback cases are held.
Epoch5 and all later continuation decisions retain the original accuracy/CE
gates; the historical fallback-release rule is suspended by Filip's09:00 stop.
The six fallback jobs remain held regardless of the epoch5 outcome.
The changed selection path is recorded in the experiment handoff and must
accompany any longer result. No gate implementation or numerical source changed.
This reference is separate from the torch2.5.1 ours reference used by
exp015–019 and021. Evidence:
`results/cifar-eqprop-ours-5090-20260929-v1/analysis/native_bptt-1-collection.json`.

Scheduling decision September30 02:30 Paris: submit baseline1x and its matched
native control through epoch5 now, since the collected evidence already meets
the frozen gate. This advances an independently eligible branch while0.3x is
recovered; it does not change thresholds, add cases, discard the other candidate
or increase the declared budget. The original round-completion callback remains
unchanged. Following the later exp020 transfer, the cancelled original1x
epoch5 ID is not relaunched; main reviews that branch in its new cohort.

Evidence: `results/cifar-eqprop-baseline-5090-20260929-v1/analysis/*-1-collection.json`
and `results/cifar-eqprop-legacy-3090-20260930-v1/analysis/native_bptt-1-collection.json`.

September30 06:00 Paris: the original baseline5090 native control completed
epoch5 at **62.58% validation accuracy, CE1.049603**, with7,035 total updates.
The four-epoch continuation took5,020.6seconds; its full-state collection passed.
This is a BPTT reference, not an EqProp result. Baseline1x EqProp has not yet
completed epoch5. The separate matched3090 continuation in
[exp-020](../experiments/exp-020-baseline-3090-continuation.md) will use its own
continued native control for the gate; preserve this5090 result independently.
Evidence: `results/cifar-eqprop-baseline-5090-20260929-v1/analysis/native_bptt-5-collection.json`.

Ours5090 native epoch5 also passed collection: **66.68% accuracy,
CE0.941809**,7,035 total updates. Its0.3x EqProp continuation started on the
freed loulou5090; promotion requires accuracy>=64.68% and CE<=0.988898951.
The explicit epoch1 allocation exception above remains part of this trajectory.
Evidence: `results/cifar-eqprop-ours-5090-20260929-v1/analysis/native_bptt-5-collection.json`.

September30 09:25 Paris: ours0.3x completed and passed local full-state
collection at **54.12% validation accuracy, CE1.255383**,7,035 total updates.
The four-epoch continuation took4,205.3seconds. Against its matched native
66.68%, CE0.941809, the gap is12.56pp and CE33.29% higher, failing both
unchanged gates. The initial one-image boundary exception did not recover
BPTT-like learning by epoch5. This tests one beta vector and seed, not every
possible ours EqProp operating point. No epoch10 extension or fallback release.
Evidence: `results/cifar-eqprop-ours-5090-20260929-v1/analysis/beta_0p3-5-collection.json`.
Only the already-running legacy1x V100 continuation remains active; its
collection-only owner and30-minute summary timer remain enabled.


## Retrospective review: larger fixed beta can still give a tiny trained response

Existing epoch5 fixed-cohort diagnostics give the following block3 output
relative RMS and first-Conv gradient comparisons against native BPTT:

| Trajectory | Block3 beta | Relative RMS | Cosine | Gradient norm/native |
| --- | ---: | ---: | ---: | ---: |
| Legacy0.0001x V100 |0.0000675481|1.486e-8|0.0154|81.285|
| Legacy0.1x V100 |0.0675481|7.611e-6|0.4765|2.048|
| Ours0.3x5090 |0.350856|7.530e-6|0.7868|1.253|
| Baseline1x3090 continuation |0.940988|4.859e-5|0.8677|1.148|

These are different trained trajectories, not a same-checkpoint beta sweep.
The larger fixed settings do not exclude insufficient response as a contributor.
On ours0.3x's fixed cohort, unchanged block3 beta gives relative RMS1.148 at
initialization,2.132e-6 at epoch1, and7.530e-6 at epoch5. Absolute displacement
also falls0.013764→4.088e-6 by epoch1, while free-state RMS rises0.011987→1.917366.
This reflects both changing forcing/state response and the normalization;
it cannot be assigned to susceptibility alone. Legacy3x reaches block3 cosine
0.9947 at epoch1, so distortion is not universal across all larger settings.

Numerical precision remains a retrospective hypothesis, not an isolated cause:
output RMS does not establish precision at every internal layer, and these
cohort probes do not measure every training minibatch. The adaptive studies
also fail despite close target tracking. Distinguishing finite-beta bias,
precision and solver error requires a matched-state intervention; none was
launched for this review. Sources are the existing named roots' stage1/stage5
`cells/*/*/metrics.jsonl` beta_observation records, with legacy in exp016.
