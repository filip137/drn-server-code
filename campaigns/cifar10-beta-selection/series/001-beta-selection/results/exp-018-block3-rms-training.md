---
experiment: "exp-018"
evidence: "validated-local"
summary: "Block3 RMS.005 improves accuracy28.04→32.88% at ceiling30,000, but both matched-BPTT continuation gates still fail."
verdicts: {"H-014": "inconclusive"}
---
# The larger target helps this comparison but does not qualify for continuation

The fresh epoch on nom-cool-1 is collected and locally validated:1,407 updates,
45,000 training examples and5,000 validation examples. The matched software
runtime, initialization, rates and data order were retained; only block3 target
changed relative to exp-017. Runtime3,225.6seconds, within5,400seconds. No
operational retry, numerical failure, noise or official test use.

| Arm | Accuracy | CE |
| --- | ---: | ---: |
| Matched native BPTT |37.06%|1.726060|
| Block3 RMS.001, ceiling30,000 |28.04%|1.995131|
| Block3 RMS.005, ceiling30,000 |32.88%|1.945375|

The larger target improves accuracy4.84pp and lowers CE2.49% in this single-seed
comparison. It still misses accuracy>=35.06% and CE<=1.812363, so no long
extension. H-014 has a positive relative comparison, but remains inconclusive
as a route to the declared BPTT-like learning quality. This does not measure
read-noise robustness or prove a numerical mechanism.

RMS tracking was close: median actual/target[.9981,.9879,.9842], with only one
block2 tolerance miss at the five-pair cap. There were no ceiling hits causing
misses. Average centered-pair counts were[1.377,1.224,1.109]. Maximum block2
beta15,582 was below the30,000 ceiling. Thus failure to reach the requested
RMS is not a sufficient explanation for this validation deficit.

The lower-ceiling/smaller-target exp-015 yielded32.64%, CE1.788320; comparing
it directly to this run changes two settings. The lower-ceiling/larger-target
combination subsequently reached28.92% in [exp021](exp-021-low-cap-block3-rms.md),
reversing the ceiling effect across requested targets. Exp-019 inspects
the actual early gradients of the two already-failed ceiling trajectories.
Evidence: `results/cifar-beta-current-batch-block3-rms-20260930-v1/`, including
`analysis/current_batch-1-collection.json` and every-update controller metrics.
