---
experiment: "exp-016"
evidence: "validated-local"
summary: "Matched 400-update dense-LR screen: H-010 inconclusive in its predeclared short-window regime."
verdicts: {"H-010": "inconclusive"}
---
# Dense-readout LR screen

All six branches completed400 updates from their original epoch10 full-state checkpoints.
Within each scheme, starting predictions, model, Adam moments and RNG matched; only
readout LR differed. Convolution rates were unchanged. All18 evaluation records validated.
Frozen BN affine, original unnormalized legacy, fixed train2048/validation1024 cohorts;
one seed, same Trex runtime. No smokes or official test read.

| Scheme | Head LR multiplier | Training CE at400 | Training CE improvement | Validation CE change | Validation accuracy change |
|---|---:|---:|---:|---:|---:|
| legacy | 1 | 0.654961 | +0.00% | +0.00% | +0.000 pp |
| legacy | 0.33333333 | 0.646762 | +1.25% | -3.97% | +1.465 pp |
| legacy | 0.03704242 | 0.674605 | -3.00% | -0.01% | -0.684 pp |
| ours | 1 | 0.648638 | +0.00% | +0.00% | +0.000 pp |
| ours | 0.33333333 | 0.671765 | -3.57% | +5.00% | -0.586 pp |
| ours | 0.03704242 | 0.625701 | +3.54% | -4.17% | +1.367 pp |

H-010 verdict: **inconclusive**, applying the predeclared primary low-rate criterion.
Legacy-minus-ours relative training-loss improvement: -6.535 percentage points.
The one-third dose is secondary and cannot replace the primary endpoint.
This is a bounded fitting test, not an optimal-rate estimate or a causal explanation
of the historical scheme ranking. No longer continuation is automatically launched.

Scientific review: the requested aggressive legacy reduction fails the primary
prediction: training CE worsens3.00% and validation accuracy falls0.684pp, while
ours benefits from the same relative reduction. The overall registered verdict
remains inconclusive because legacy's one-third-rate secondary branch improves
both training CE and validation. There is no general monotonic benefit from
lowering the readout LR. Legacy's aggressive reduction helps at100 updates but
loses that advantage by400; ours' one-third branch similarly reverses its early
benefit. These reversals argue for a longer confirmation before choosing rates.

Next proposed test, not launched: compare legacy head LR divided by3 and ours
head LR multiplied by0.03704242 against their respective original-rate controls
over1–3 full epochs, with full validation. These are screening candidates, not
established optima. Both queue jobs completed execution, collection and validation;
the result plot and exact predeclared criterion arithmetic were inspected.

Evidence root: `results/cifar10-dense-readout-screen-20260928-v1/`; complete arithmetic
`analysis/summary.json`, plot `analysis/dense_readout_response.jpg`, and canonical
`probes/{legacy,ours}` bundles. Partial-epoch checkpoints are not native-resume-safe.
