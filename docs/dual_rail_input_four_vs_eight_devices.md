# Clamped dual-rail input: four versus eight devices

## Question

Does the double-differential input interface introduce independent nodes, or
does it reduce to the original dual-rail interface because the input is
clamped to `[x, -x]`? If it reduces, can a four-device logical synapse retain
the measured-device benefit previously obtained with eight devices?

The first part has an exact circuit answer. The second part depends on device
range and mismatch and is therefore covered by a frozen MNIST experiment.

## Exact input-boundary identity

For one logical input,

```text
x+ = x                         x- = -x
x++ = x+                      x+- = -x+ = x-
x-+ = x-                      x-- = -x- = x+
```

Thus `x+-` is exactly `x-`; it is not another independently evolving node.
Likewise, `x--` is `x+`. The apparent four input ports contain only the two
original rail voltages.

Let `G+` and `G-` be the two non-negative branch matrices of the signed
interaction. Split their source rows into the `x+` and `x-` halves. On the
clamped subspace, the eight physical devices of each logical input/output
rail block can be regrouped into the four post-facing conductances

```text
C[x+, :] = G+[x+, :] + G-[x-, :]
C[x-, :] = G-[x+, :] + G+[x-, :].
```

For any hidden voltage, this regrouping preserves both the edge energy and
the current entering every hidden node. It also preserves the derivative in
the permitted logical-input direction, where a change in `x+` is accompanied
by the opposite change in `x-`.

This is deliberately a boundary result. The individual source-port currents
are not preserved, and the identity fails if the two source rails are free
dynamic variables. Therefore the same reduction cannot generally be applied
between hidden layers, even when the desired equilibrium is approximately
antipodal.

For the ideal teacher map, with common branch baseline `b`, scale `s`, and
signed differential lift `D`,

```text
G+ = b + s max(D, 0)
G- = b + s max(-D, 0)
C  = 2 b + s dual_rail_lift(W).
```

So, at the clamped input, the eight-device implementation is exactly a
four-conductance dual-rail block whose conductances are sums of device pairs.

## Why this is not merely four unchanged devices

Replacing each sum by one existing-range memristor is exact only if that
memristor can realize the summed conductance, or if all relevant circuit terms
are rescaled consistently. Two devices in parallel can provide twice one
device's reachable conductance and change mismatch statistics. Consequently,
the node identity removes redundant *ports*, but it does not automatically
remove the range and averaging supplied by the second device.

This distinction is what the experiment tests:

- The numerical equivalence tests use arbitrary `G+` and `G-` and require
  float64 agreement of input-edge energy, hidden-node quadratic and linear
  coefficients, hidden-node KCL, and the logical input derivative.
- A negative control releases the two input rails and requires the proposed
  collapse to cease being equivalent.
- The teacher-map test requires the collapsed differential pair to equal
  `2 b + s dual_rail_lift(W)` for both repository rail layouts.
- The measured four-device arm assigns all four cells of one logical
  synapse a single reachable baseline and span. This
  `dual_rail_quad_common_window` mapping is stricter than the earlier mapping,
  which shared a window only between each output-rail pair.
- The measured eight-device control retains the existing `G+`/`G-` common-
  window implementation.

## Frozen end-to-end comparison

Both arms use the same frozen teacher, MNIST split, seed, minibatch order,
solver, model-local amplification, raw cohort-A traces, assignment seed,
teacher-KL objective, calibration subset, and ten-epoch budget. Each arm uses
its already calibrated gradient-equivalent learning rates.

- Four-device config:
  `examples/mnist_relu_drn/measured_raw_single_quad_common_window_finetune_10ep.json`
- Eight-device config:
  `examples/mnist_relu_drn/measured_raw_differential_finetune_10ep.json`
- Campaign:
  `campaigns/manifests/mnist_relu_drn_four_vs_eight_common_window_10ep.json`

The primary empirical retention criterion is that the four-device arm's
selected held-out test accuracy is no more than 1.0 percentage point below
the eight-device arm, with finite training and validation metrics. Validation
teacher KL, teacher agreement, selected epoch, mapping overlap, programming
error, and initial accuracy are recorded as diagnostics. This is an
exploratory, single-seed screen; passing it would show that the benefit is
retained in this protocol, not that the implementations are universally
equivalent.

After placing the frozen inputs described in
`campaigns/manifests/README.md`, run from the repository root:

```bash
python -m ebl campaign run \
  --manifest campaigns/manifests/mnist_relu_drn_four_vs_eight_common_window_10ep.json \
  --output-dir results/mnist-dual-rail-four-vs-eight-common-window-10ep-20260819-v1 \
  --fail-fast
```

## Status

The exact equivalence checks, strict four-cell mapping, frozen configs, and
campaign are implemented. The full CUDA campaign is intentionally left
unreported until it is run from a clean committed worktree with the frozen
teacher and measured-trace inputs.
