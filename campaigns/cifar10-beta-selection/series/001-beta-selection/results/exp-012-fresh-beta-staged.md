---
experiment: "exp-012"
evidence: "validated-local"
summary: "All eight fresh fixed-beta candidates fail the epoch1 gate; best0.1x34.22% versus native37.06%."
verdicts: {"H-010": "contradicts"}
---
# Fresh transfer: fixed search exhausted

Collectors verified the full45,000-example/1,407-update boundary, fresh optimizer
start, restored initializer, BN/Adam/scheduler counters,5,000-example validation
and no official-test read. All collected cases used matching3090 runtimes.

| Epoch1 case | Validation accuracy | CE | Promotion |
| --- | ---: | ---: | --- |
| Native BPTT |37.06%|1.726060|Reference |
| Free-T autograd |37.84%|1.778235|Diagnostic only |
| Fixed0.0001x trained beta |26.18%|1.927826|Fail |
| Fixed0.001x trained beta |29.10%|2.112950|Fail |
| Fixed0.01x trained beta |29.70%|2.124774|Fail |
| Fixed0.03x trained beta |29.14%|1.919018|Fail |
| Fixed0.1x trained beta |34.22%|1.845956|Fail |
| Fixed0.3x trained beta |26.44%|2.026749|Fail |
| Fixed1x trained beta |31.84%|1.834271|Fail |
| Fixed3x trained beta |26.84%|1.975348|Fail |

The0.3x gap is10.62pp; CE is17.42% higher. This candidate does not transfer well
to fresh training under the original rates.1x is better than0.3x but still5.22pp
below native, with6.27% higher CE. Neither initial candidate passes.
All six fallback cases also fail:0.1x has the best EqProp accuracy,
but its2.84pp deficit and6.95% excess CE miss the fixed thresholds. The0.001x
and0.01x retries also fail after validated full epochs. The final0.03x/3x receipts
validate full epochs at29.14%/26.84%. The predeclared search is exhausted and no
long continuation is eligible. This contradicts transfer under these tested
fixed coefficients, rates and seed; it does not rule out another beta policy,
learning rates or other schemes. Exp-013 retains its separate all-scheme scope.

The0.3x initialization diagnostic has block2 minimum native cosine−0.1025.
By epoch1 it is0.8363, while block3 has minimum0.6685. Relative output RMS changes
from[.39185,72.8370,1.14833] to[2.344e-5,1.442e-5,1.977e-6]. Fixed beta therefore
does not fix effective nudging. These observations alone do not establish whether
initial gradient bias, later small response, or the hybrid free-state computation
caused the learning gap. The fresh free-T BPTT control isolates that last factor.

The free-T control now completes with native-like accuracy and CE only3.02%
above native. Thus the large0.3x accuracy deficit is not reproduced merely by
using the hybrid free-state forward graph with exact autograd. This points to
the EqProp gradient replacement/beta choice, including its interaction with the
fixed learning rates. It does not yet distinguish initial bias from later small
response or establish a better beta policy. This remains a single-seed comparison.

At epoch1, block3's minimum pooled native-gradient cosine for multipliers
[0.0001,0.001,0.01,0.1,0.3,1] is[.003,.003,.133,.322,.669,.927]; maximum norm
ratios are[125.16,39.39,10.47,3.27,1.54,1.09]. The smallest coefficients therefore
do not retain useful late-epoch gradient agreement.1x has much better final
alignment but worse initial alignment, so a time-dependent beta is a plausible
next hypothesis, now specified in
[exp-014](../experiments/exp-014-minibatch-beta-feedback.md). Retry provenance is preserved in the collection receipts
and `results/cifar-beta-fresh-staged-retry-20260929-v1/analysis/recovery.json`.

The final3x case reinforces the distinction: epoch1 minimum pooled cosine is
[.99985,.97887,.95176] by block, with norm ratios at most1.053, despite its26.84%
validation accuracy. Good terminal alignment cannot undo its earlier training
trajectory or establish that the same beta was useful throughout the epoch.

Evidence: `results/cifar-beta-fresh-staged-20260929-v1/analysis/*-1-collection.json`
and `cells/1/beta_0p3/metrics.jsonl`. The added control is declared in the
[experiment handoff](../experiments/exp-012-fresh-beta-staged.md).
Its validated receipt is
`results/cifar-beta-fresh-free-t-control-20260929-v1/analysis/free_t_bptt-1-collection.json`.
