# MNIST Wan-2022 versus CMO endpoint comparison

These configs define one controlled DRN comparison in which only the
endpoint-device branch changes.

- Common circuit: `[1568, 100, 20]`, perfect diode, four asynchronous
  iterations, `input_gain=50`, `voltage_amp=4`, and `current_amp=0.25`.
- Common HWA: two additional BPTT epochs from the selected ordinary FP32
  checkpoint with output-channel-scaled additive Gaussian noise
  (`std_dev=0.03`, seed 101).
- Common retention time: one day (`86400 s`).
- Common logical transfer: affine mapping across each measured physical
  range while retaining the physical floor in the DRN equations.
- Common full-BPTT recovery: 10 epochs, W1/W2 learning rates
  `0.008/0.005`, frozen hidden bias, and a fresh noisy endpoint rewrite of
  both dense arrays after every minibatch update.

The two physical targets for a nonnegative DRN edge `w in [0,1]` are:

```text
Wan-2022: G_target = 1 + 39 w          uS
CMO:      G_target = 9 + 79.199997 w   uS
```

This is a matched one-device-per-positive-DRN-edge comparison. It is not the
NeuRRAM signed-weight encoding, where one logical signed weight is represented
by the difference of two Wan devices. The affine mapping is chosen to isolate
the two endpoint models under identical DRN semantics without cancelling
either physical floor.

For a deployment probe, report `metrics.initial_validation`: it is evaluated
after the first noisy write and before the deliberately zero-rate probe step.
For full BPTT, the same field is the post-HWA deployment accuracy and the
selected named checkpoint is the post-recovery result.
