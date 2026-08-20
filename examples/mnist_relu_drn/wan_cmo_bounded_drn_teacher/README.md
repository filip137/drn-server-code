# CMO versus Wan from a bounded DRN teacher

These configs consume the selected 10-epoch teacher from
`examples/small_drn/mnist_bounded_memristor_teacher_10ep.json`.

The initialization is an exact stable-key tensor copy. It uses fixed logit
gain 1, performs no mapping search, subtracts no floor, and adds no affine
offset. The loader requires the checkpoint to describe the same bias-free
`[1568, 100, 20]` DRN, paired output semantics, four-iteration solver,
model-local amplifier topology, MNIST split, and conductance bounds.

The shared normalized interval is `[9 / 88.199997, 1]`. Device deployment is
purely multiplicative:

- CMO: `G = 88.199997 uS * w`, so the full permitted interval is
  `9--88.199997 uS`.
- Wan-2022: `G = 40 uS * w`, so the permitted teacher interval is the
  `4.08163279--40 uS` subset of its reported `1--40 uS` range.

`hwa_add_normal_10ep.json` trains clean master weights through additive-normal
perturbations. Checkpoint selection averages four fixed validation-noise
realizations, reused at every epoch, because clean KL starts at exactly zero
for a literal teacher copy and therefore cannot select a robustness update.

The normalized KD learning rates are the prior physical-conductance rates
divided by `(110e-6)^2`: `[0.017355371900826448,
0.00004710743801652893]`. Program-and-verify configs perform one deterministic
seed-17 noisy write at initialization. The full-BPTT configs then repeat that
device write after every minibatch update for 10 epochs.
