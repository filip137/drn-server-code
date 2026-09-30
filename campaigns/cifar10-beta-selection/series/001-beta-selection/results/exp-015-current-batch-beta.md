---
experiment: "exp-015"
evidence: "validated-local"
summary: "Current-batch beta selection improves CE but misses the accuracy gate:32.64%, CE1.788320 versus native37.06%, CE1.726060."
verdicts: {"H-012": "contradicts"}
---
# Better response tracking did not pass the learning gate

One complete fresh epoch is locally collected and validated:45,000 training
examples,1,407 updates,5,000 validation examples, matched initialization,
rates, digital updates and data order. Collector checked every-update selected
beta, trial history, controller counters and final checkpoint. No official test
data were read. Runtime3,058.1seconds, within10,800seconds.

| Arm | Validation accuracy | CE |
| --- | ---: | ---: |
| Matched native BPTT |37.06%|1.726060|
| Fixed beta |34.22%|1.845956|
| Lagged feedback |34.34%|1.981908|
| Current-batch selection |32.64%|1.788320|

Current-batch selection passes the CE limit1.812363 but falls4.42pp below
native, failing the accuracy threshold35.06%. It lowers CE relative to both
EqProp controls while reducing accuracy. This contradicts the tested H-012
learning-benefit criterion; the improved response tracking is a separate,
supported mechanism. No longer continuation is authorized by this gate.

Median training RMS/target by block was[.9965,.9811,.9904]. Blocks1/3 had no
tolerance misses; block2 had77/1,407. Of these,75 hit beta10,000 and stopped
at the repeated upper bound (updates57–219, mostly before200). All undershot;
worstRMS.015128 versus target.035. Two later misses at1055/1056 exhausted the
five-pair limit during an abrupt response change. Average centered pairs per
block/update were[1.377,1.190,1.172]. Correcting the early ceiling may help,
but these observations do not establish it as the cause of the accuracy gap.

End-of-epoch fixed-cohort diagnostics use stored warm-start betas, without
current-batch search: these are not the cosines of the gradients selected
during training. Good endpoint alignment cannot establish a good earlier
optimization trajectory.

Next: [exp-017](../experiments/exp-017-current-beta-ceiling.md) changes only the
search ceiling to30,000, keeping the target, five-pair limit and learning gate.
Evidence: `results/cifar-beta-current-batch-20260930-v1/`, especially
`analysis/current_batch-1-collection.json` and the cell's metrics/controller
artifacts. The shared queue completed collection; no operational retries.
