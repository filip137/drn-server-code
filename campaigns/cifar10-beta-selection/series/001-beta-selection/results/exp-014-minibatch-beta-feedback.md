---
experiment: "exp-014"
evidence: "validated-local"
summary: "Lagged beta feedback tracks median RMS but fails the epoch1 gate:34.34%, CE1.981908 versus fixed34.22%, CE1.845956."
verdicts: {"H-011": "contradicts"}
---
# Response tracking did not recover matched-BPTT learning

Both full epochs validated1,407 updates,45,000 training examples and5,000
validation examples with matched initialization, rates, BN and data order.
Collector verified every-update beta history and checkpoint counters. The
instrumented fixed run exactly reproduced the earlier fixed0.1x endpoint.

| Arm | Accuracy | CE | Long-run gate |
| --- | ---: | ---: | --- |
| Native BPTT reference |37.06%|1.726060|Reference |
| Fixed beta |34.22%|1.845956|Fail |
| Lagged minibatch feedback |34.34%|1.981908|Fail |

Feedback adds0.12pp to fixed accuracy but increases CE7.36%; it remains2.72pp
below native with14.82% higher CE. These seed0 results contradict the tested
controller's proposed learning benefit under the declared gate. No extension
is eligible; this does not rule out other targets or response-selection rules.

After the first50 updates, median actual/target RMS ratios are[1.000,1.003,.993].
However, maxima are[4.21,1093.05,9.66], and only[69.2%,81.9%,91.2%] of batches
fall within25% of target. The upper beta bound is never hit. Block2's rare large
overshoots coexist with good median tracking. The first optimizer update also
changes response sharply: block2 RMS29.89 on batch1 versus1.35e-5 on batch2.
The lagged controller cannot correct the current batch before its update.

Final diagnostic minimum pooled cosine by block improves to[.9982,.9451,.9997],
versus fixed[.9729,.8657,.3218]. Good terminal alignment therefore does not
establish good learning throughout the earlier trajectory. The next bounded
[current-batch test](../experiments/exp-015-current-batch-beta.md) keeps targets
unchanged and selects beta before each update. It tests a possible cause rather
than assuming these overshoots explain the accuracy deficit.

Both arms took about45minutes. The feedback startup failure preceded all updates;
its preserved retry is not an independent replicate. Evidence:
`results/cifar-beta-minibatch-feedback-20260930-v1/analysis/*-1-collection.json`,
`cells/1/{fixed,feedback}/metrics.jsonl` and final beta-boundary artifacts.


## First-epoch fixed-beta response history

Filip requested response tracking within epoch1. Existing fixed0.1x telemetry
already covers every one of1,407 actual training pairs; no replay or new run
was needed. All three betas remain constant. This is ours on the matched
2.5.1 runtime, not the separate5090 cohort.

| Block | Batch1 relative RMS | Batch2 relative RMS | Median batches11–50 |
| --- | ---: | ---: | ---: |
|1|0.150437|6.391e-5|8.797e-7|
|2|29.8917|2.694e-5|2.286e-6|
|3|0.506164|1.454e-5|4.304e-7|

[Full epoch and first50 updates](../../../../../results/cifar-beta-minibatch-feedback-20260930-v1/analysis/first-epoch-fixed-response.jpg)
plot relative displacement, absolute displacement and free-state RMS separately.
The adjacent JSON retains all measured values, window medians and provenance.
For block2, batch1→2 absolute response falls0.144934→4.729e-5 while free-state
RMS rises0.004849→1.75539. The relative response drop is therefore not merely
a denominator effect. The response changes immediately and remains small
for much of the epoch, rather than only degrading near its end.

Different minibatches and changed model state are confounded in this comparison;
it does not isolate the first optimizer update or prove numerical cancellation.
Only block outputs were recorded. Internal-layer displacement, unchanged-state
fractions, forcing RMS and selected actual-minibatch gradient comparisons would
be needed to separate weak forcing, poor response propagation and gradient error.
These are proposed follow-up measurements, not launched work. Baseline/legacy
fixed runs lack corresponding every-update response logs; their epoch-boundary
measurements cannot reconstruct the missing history. Existing stop on new runs
and extensions remains in effect.
