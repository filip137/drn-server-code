---
experiment: "exp-011"
evidence: "validated-local"
summary: "All seven trained half-epoch branches complete:0.3x gives best validation76.44% versus BPTT75.16%;0.1x is closest in CE and1x retains strong gradient alignment."
verdicts: {"H-009": "supports"}
---
# Trained beta supports short continuations

All seven branches completed22,500 identical examples/704 updates from the
same full epoch10 parent, on matching3090 runtimes. Local collectors verified
initial tensors, data-order hashes, constant inherited LRs/scheduler10, Adam14774,
BN counters, partial-checkpoint metadata and diagnostic coverage. No failures,
missing cases, noise or official-test reads. Shared-queue jobs are complete.

| Case | Validation accuracy | CE | Conv-update cosine to BPTT |
|---|---:|---:|---:|
| Native BPTT |75.16%|0.728017|1.000|
| Free-T autograd |75.44%|0.709455|0.618|
| Beta0.1x |74.74%|0.737147|0.544|
| Beta0.3x |76.44%|0.694985|0.605|
| Beta1x |75.72%|0.715240|0.582|
| Beta3x |76.00%|0.704314|0.306|
| Beta10x |75.44%|0.715191|0.224|

H-009 is supported only as a short trained-continuation result: candidate betas
can sustain BPTT-like validation performance.0.1x is closest by the declared
absolute CE-gap ranking, but0.3x performs best and is closest among EqProp arms
in aggregate Conv-update direction. These are distinct measures, not one winner.
At1x the worst final pooled Conv cosine is0.9654; at3x block2 falls to0.1343.
Large betas can have poor gradient alignment without an immediate accuracy
collapse. Every final CE exceeds the starting0.691127, including BPTT; this small
continuation is not evidence of uniform loss improvement or statistical superiority.

The free-T control itself differs from native BPTT in accumulated update
direction, limiting attribution of that discrepancy solely to beta. This is one
seed, half an epoch, with inherited Adam history and already learned features.
It does not establish fresh-training performance. Filip requested fresh0.3x/1x
tests next, with conditional extensions/search in exp-012.

Evidence: `results/cifar-beta-half-epoch-e10-20260929-v1/analysis/comparison.json`
and six-plus-control collection receipts. [Comparison plot](../../../../../results/cifar-beta-half-epoch-e10-20260929-v1/analysis/comparison.jpg).
