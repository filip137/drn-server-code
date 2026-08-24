# ReLU to bounded FP32 DRN mapping

This zero-update mapping control starts from the frozen bias-free
`784 -> 50 -> 10` ReLU checkpoint and constructs the single-conductance
dual-rail DRN used by the IBM OM work. Every DRN master conductance remains a
continuous `float32` value constrained to
`[0.1020408197973068, 1.0]`.

The mapper normalizes each signed ReLU layer, lifts it onto the four dual-rail
edges, adds the shared lower conductance floor, searches the predeclared layer
fractions `{0.125, 0.25, 0.5, 1.0}`, and fits one positive output gain on the
fixed 1,024-example validation calibration cohort. It performs no optimizer
update, HWA perturbation, device assignment, quantization, or
program-and-verify operation.

The frozen teacher is:

```text
results/mnist-relu-drn-kd-exploratory-20260816/
teacher_fixed_init/
20260816T132557.806720Z-fbff3c26-6f6867f1/checkpoints/weights.pt
SHA-256: 9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52
```

Its tensors were independently verified to reproduce exactly the teacher
mapping used by the earlier teacher-initialized CMO/Wan study.

## Result

The zero-update mapping selected layer scale fractions `(0.25, 0.125)` and a
fixed positive output gain of `56.234132519`. On the fresh 10,000-example MNIST
test split:

- frozen ReLU teacher accuracy: `97.36%`;
- bounded FP32 DRN accuracy: `97.38%`;
- teacher agreement: `99.94%`;
- teacher-to-DRN KL: `0.00013019`.

The mapped checkpoint is:

```text
results/mnist-relu-to-bounded-fp32-map-20260824-v1/
runs/bounded-map/
20260824T134046.646663Z-d07d32ea-498b4429/checkpoints/weights.pt
SHA-256: 0865b16dbf504a902f42914c719ec1685e80189957b97a77d101efd85efbc4fb
```

It contains 158,800 finite `float32` conductances. All are inside the declared
range. Layer 1 occupies `[0.1020408198, 0.3265306056]` and layer 2 occupies
`[0.1020408198, 0.2142857164]`; exactly half of each layer's conductances sit
at the shared floor as required by the four-device signed mapping.

This establishes only the clean compression control. It does not include a
measured-device assignment, per-cell common-window remapping, HWA noise,
quantization, program-and-verify error, or on-chip updates. Subsequent IBM OM
comparisons should use this checkpoint as their common logical starting point
rather than the separately trained bounded-DRN checkpoint used by the earlier
pilot.
