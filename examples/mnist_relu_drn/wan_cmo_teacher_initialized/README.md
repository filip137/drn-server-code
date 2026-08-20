# Teacher-initialized CMO versus Wan-2022 MNIST protocol

All arms begin from the same frozen bias-free `784 -> 50 -> 10` ReLU teacher.
Its signed matrices are lifted into the bias-free single-conductance dual-rail
`[1568, 100, 20]` DRN before any hardware-aware training or device write.

The shared logical DRN range is `0..110 uS`, with `input_gain=100`,
`voltage_amp=4`, `current_amp=0.25`, and four asynchronous iterations. The
physical endpoint maps retain their floors:

```text
CMO: G_target = 9 + (88.199997 - 9) * G_DRN / 110 uS
Wan: G_target = 1 + (40 - 1) * G_DRN / 110 uS
```

The protocol records the following trajectory:

1. frozen ReLU teacher test accuracy;
2. ideal teacher-mapped FP32 DRN, without DRN pretraining;
3. first CMO or Wan endpoint write from that exact mapped checkpoint;
4. two clean-master HWA epochs using 3% output-channel-scaled additive-normal
   conductance noise, followed by the first endpoint write;
5. ten full-BPTT epochs from the HWA checkpoint, with gradients evaluated at
   the realized device state and a fresh noisy endpoint rewrite after every
   minibatch update.

Mapping and endpoint gains are selected only on the 5,000-example validation
split. Final results are evaluated once on the untouched 10,000-example test
split. Device seed 17 is an exploratory single-realization comparison.
