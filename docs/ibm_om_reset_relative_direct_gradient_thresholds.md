# RESET-relative direct-gradient thresholding

This development protocol tests whether the direct-pulse recovery rule spends
writes on gradients that are too small to justify a full physical SET or RESET
pulse. It starts every arm from the exact saved deployment used by the
unthresholded recovery pilot. It does not remap, reprogram, resample, or inspect
hidden device bounds.

## Fixed gate

Before the first update, the runtime replays 64 deterministic training
minibatches at the source apparent state. The replay restores the training
dataloader and global RNG state afterward. For each stable conductance
parameter separately, it gathers all strictly nonzero absolute raw gradients
and computes one linear-interpolated p90, p95, or p99 threshold.

During training, cell `i` is eligible only when

```text
abs(gradient_i) > threshold_for_its_parameter
```

Equality holds the cell. The sign of `-learning_rate * gradient_i` still
chooses SET versus RESET. The gate is symmetric because this deployed OM model
permits both directions; the earlier one-pulse-down ReRAM study used only a
positive-gradient gate because its update rule could only decrease
conductance.

The thresholds are fixed after the source-state replay. A label such as p99
therefore describes the calibration distribution, not a promise that exactly
one percent of later minibatch gradients remain eligible.

## No pulse redistribution

Each budget reuses the exact probability scale fitted by its unthresholded
control:

| Slow-pulse cap per cell | Frozen scale | Source report SHA-256 |
| ---: | ---: | --- |
| 0.1 | `0.005644706616388747` | `f3ded58c814858f3bb1a6c8da282521e04ee8f3fdaebf9cba8ffacb1cc02a1c2` |
| 1 | `0.05644706616388745` | `e9944e8ce1494e95369bc632be02a8ca60588be20d22ba59471d836290b2dbc1` |
| 4 | `0.2257882646555498` | `b36e5d09f648ad47b12c12c629587cd826587fc5460d3dac4f8aea55c3cc23b1` |

The runtime first proves that the frozen scale applied to the ungated replay
still reproduces the earlier expected full-run cap. It then zeros probabilities
below the threshold without increasing the remaining probabilities. Thus a
gate may save writes; it does not force the full budget onto fewer cells.

## Grid and evidence boundary

The one-seed grid crosses p90, p95, and p99 with caps 0.1, 1, and 4. Every arm
runs ten epochs from source deployment SHA-256
`0b99f4dd73841a891993708bdace0ab508864e2e60e56620c165e3df99f006cd`.
Epoch 10 is primary; the best intermediate epoch is descriptive only.

Each run retains the per-parameter thresholds, raw-gradient sample digests,
strict comparison rule, dynamic eligibility counts, frozen-scale provenance,
pulse ledgers, apparent and persistent endpoint states, and exact resume RNG
state. The experiment remains on the development array and validation split.
No additional seed or held-out deployment is released by this study.
