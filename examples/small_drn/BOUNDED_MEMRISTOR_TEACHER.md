# Bounded MNIST DRN teacher

`mnist_bounded_memristor_teacher_10ep.json` trains a bias-free DRN from
scratch for 10 epochs. Every dense conductance is constrained during
initialization and after every update to

\[
0.1020408197973068 \leq w \leq 1.
\]

The lower ratio is the stricter CMO ReRAM floor, `9 / 88.199997`. Deployment
therefore needs only a positive global scale, with no additive remapping and
no floor subtraction:

- CMO ReRAM: `G_CMO = 88.199997 uS * w`, giving `9--88.199997 uS`.
- Wan-2022 ReRAM: `G_Wan = 40 uS * w`, giving
  `4.0816327919--40 uS`, a floor-safe subset of its `1--40 uS` range.

The initialization is the historical non-negative Kaiming initialization
shifted by the finite floor. This preserves its effective random signed
dual-rail differences while ensuring that the initial network is already
physically deployable.
