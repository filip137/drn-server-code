# IBM OM baseline-position by spacing P&V configs

These strict CUDA-only configs belong to
`mnist-ibm-om-shared-destination-baseline-spacing-pv-20260828-v1` and the
scientific contract in
[`docs/ibm_om_baseline_spacing_pv.md`](../../../docs/ibm_om_baseline_spacing_pv.md).

The 27 production files cover exactly:

```text
alpha   = 0.0, 0.25, 0.5
h       = 1, 2, 4 times delta_x
heldout = 87001, 87002, 87003
```

Each native validation computes one ideal quantized endpoint and five
pulse-resolved persistent P&V repeats. Endpoint seeds are fixed by held-out
assignment and identical across all nine designs. Calibration is selected
continuously once per `alpha` on assignment 86001 and must be identical across
all three spacing files and held-out assignments for that `alpha`.

`smoke-alpha_000_spacing_4delta-heldout-87001.json` is an excluded CUDA
canary. It exercises the full physical array and all five P&V seeds while
evaluating only 32 test examples. It is not listed in the study plan and
cannot satisfy production coverage.
