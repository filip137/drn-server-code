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
  --output-dir results \
  --fail-fast
```

## Result

All four stages completed from clean commit
`bb88b007fa424f88bec86f8602690dac988b022a`. An immediate `--resume` replay
reused all four stages after validating their campaign fingerprints and
artifact hashes; it created no additional attempts. The 12 focused mapping
and circuit-equivalence tests also passed before launch.

| Arm | Devices per teacher weight | Selected layer scales | Initial validation | Selected validation | Selected checkpoint | Fresh test |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| Four-device quad window | 4 | `(1.0, 1.0)` | 61.48% | 97.26% | epoch 10 | **97.46%** |
| Eight-device differential | 8 | `(1.0, 0.125)` | 89.12% | 97.42% | epoch 9 | **97.48%** |

Selection minimized validation teacher KL, not accuracy. The corresponding
function-matching diagnostics are:

| Arm | Selected validation KL | Validation agreement | Fresh-test KL | Test agreement |
| --- | ---: | ---: | ---: | ---: |
| Four-device quad window | 0.0197817 | 98.74% | 0.0183103 | 98.68% |
| Eight-device differential | 0.0138965 | 98.94% | 0.0140905 | 98.76% |

The four-device arm passes the preregistered retention screen: its held-out
accuracy is only `0.02` percentage points (two of 10,000 examples) below the
eight-device arm, well inside the `1.0`-point threshold. Its test agreement is
`0.08` points lower. Its test KL is nevertheless `0.004220`, or `29.95%`,
higher, so equal classification accuracy should not be read as equal logit-
level approximation.

Requiring one intersection across all four conductance curves makes the
initialization more restrictive than the eight-device pairwise construction:

| Arm and layer | Window group | Empty windows | Mean baseline | Mean span | Initial projection RMS error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Four-device, layer 1 | 4 | 5.837% | 76.332 uS | 11.145 uS | 0.350 uS |
| Four-device, layer 2 | 4 | 6.000% | 76.303 uS | 11.244 uS | 0.272 uS |
| Eight-device, layer 1 | 2 | 1.351% | 70.730 uS | 18.383 uS | 0.238 uS |
| Eight-device, layer 2 | 2 | 1.550% | 71.066 uS | 18.068 uS | 0.287 uS |

Thus the four-cell intersection has only `61%`--`62%` of the pairwise mean
span and about four times the empty-window rate. No nominal targets were
clipped, but independently projecting the four symmetry-related targets onto
their measured pulse states still perturbs the cancellation. This explains
why the four-device initialization is materially below the eight-device
initialization (`61.48%` versus `89.12%`).

The strict quad window is still a large improvement over the earlier
four-device *pairwise*-window control: initial accuracy rises from `34.20%`
to `61.48%`; selected test accuracy rises from `96.96%` to `97.46%`; and test
KL falls from `0.041213` to `0.018310`, a `55.57%` reduction. Ten epochs then
recover `97.94%` of the quad arm's initial validation-KL gap. The eight-device
arm recovers `94.01%` and retains the better initialization and KL.

The exploratory conclusion is therefore narrow but positive: under this
frozen single-seed protocol, measured adaptation retains essentially all of
the eight-device classification benefit with four conductances per teacher
weight. The eight-device realization remains better if initialization-only
fidelity or close teacher-logit matching is the goal.

Raw campaign output is under
`results/mnist-dual-rail-four-vs-eight-common-window-10ep-20260819-v1/` in
the operational `tiki-taka-lora-integration` worktree. The selected four- and
eight-device weights have SHA-256 hashes
`e95eb276ed30d20c024495480c1f7d985ced853b761b8a3d5f2151aa9614fcb8` and
`898131127b636893b3cc50e895635243e1ee0e0efd4ce4d9d1d35129a50195f9`,
respectively.
