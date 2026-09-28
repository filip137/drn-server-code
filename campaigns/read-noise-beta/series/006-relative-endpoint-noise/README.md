# Relative endpoint read-noise comparisons

## Active Conv1/2/3 refinement

[Exp013](experiments/exp-013-relative-noise-advantage-map.md) extends the same
initialization acquisition model to a denser D/F and relative-eta map on all
three architectures, using the same 14 global eta values for every depth and
one coefficient across all layers at each point. The union of selected
refinement patches is evaluated on all architectures, with baseline retained
and fresh noise draws on the full cohort. The revised proposed three-host
budget is 5 GPUh. Conv1/Akib and Conv3/local started September 25 at about
14:59 Paris; Conv2 started at 15:07 on nom-cool-1 sharing with Ben under explicit
all-GPU sharing permission. Tests and canaries
were explicitly waived. Persistent run-watch owns admission, collection and
the stage transition, with a common 18:58:56 Paris deadline. Exp012's execution
budget and completed evidence below remain separate. Screen subset means and
full-cohort results must not be pooled silently.

## Completed compact Conv3 comparison

Filip authorized implementation and execution on the local GPU and Akib on
September 25, 2026. This short initialization replay tests the prediction behind
[H-009](../../hypotheses/H-009-absolute-noise-voltage-scale.md): legacy's larger
voltages help under absolute noise, and its advantage may shrink under equal
fractional read precision. See [exp-012](experiments/exp-012-relative-endpoint-noise.md).

Use Conv3 only, output D/F targets 2 and 6, and two positive noise levels per
model. Reuse the accepted initializer, cohort, beta calibration and T=K=8.
This changes endpoint acquisition only. All layers, including the readout,
are measured; there is no training or new T/K sweep.

Absolute noise standard deviation is in simulator voltage units. Relative
noise is a dimensionless fraction of each clean endpoint voltage. Equal
numerical levels in the two models do not mean equal noise power. Report
within-model scheme comparisons and actual layerwise noise scales separately.
This is an exploratory mechanism comparison, not a hardware-noise validation
or a claim about training accuracy.

Budget: at most 1 GPU-hour including smokes, split into two 1,800-second lanes.
Expected production duration is 10–25 minutes per lane, to be refined from
the smoke. Daytime execution is explicitly assigned; wall deadline is
September 25, 16:00 Europe/Paris. Admit only GPU executions with demonstrated
memory headroom, preserving unrelated jobs. Results go to
`results/conv3-relative-endpoint-noise-20260925-v1/`.

## Completed exploratory result

All six contexts and 11,232 rows are collected. Relative noise reduces legacy
third-convolution alignment advantage in all paired comparisons, and ours
leads all four relative-noise cells. Cross-device BPTT equivalence and further
qualification were waived explicitly; causal attribution remains unresolved.
See the [reviewed result](results/exp-012-relative-endpoint-noise.md).
