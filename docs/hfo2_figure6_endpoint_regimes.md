# Figure-6-informed HfO2 endpoint regimes

This exploratory control keeps the AIHWKit 1.1.0
`ReRamArrayHfO2PresetDevice` unchanged. It defines separate synthetic
full-endpoint operations in a unitless positive-state coordinate:

```text
full_tile_RESET = max(0.05, 0.1 + sigma_RESET z_RESET)
full_tile_SET   =           2.0 + 0.4295 z_SET
```

The SET standard deviation is the unchanged absolute upper-bound deviation of
the normalized HfO2 preset. The RESET standard deviation is scaled by the
ratio digitized from the Baseline-HfO2 marginal CDFs in IEDM 2022 Figure 6:

```text
sigma_RESET / sigma_SET = 6.4411 / 32.5241 = 0.198040837...
sigma_RESET             = 0.4295 * 0.198040837... = 0.085058540...
```

The ratio is applied to the raw Gaussian parameters before RESET clipping, as
requested. Consequently, the realized RESET standard deviation is smaller
than `0.085058540`; it is always reported separately. Clipping is lower
Winsorization, not device selection: all cells remain in the population and
every raw RESET draw below `0.05` becomes exactly `0.05`. SET receives no
lower or upper clip, redraw, or repair. A non-positive or RESET-crossing SET
draw invalidates the sampled population instead of being silently changed.

Figure 6 exposes endpoint marginals but not recoverable device-by-device
pairs. Two joint-distribution regimes therefore share the exact same endpoint
laws:

- `figure6_ratio_independent`: independent RESET and SET Gaussian quantiles,
  matching AIHWKit's independent bound-sampling assumption;
- `figure6_ratio_rank_matched`: the exact same finite RESET and SET marginal
  samples are paired in matching quantile order, providing a positively
  correlated pairing sensitivity control without changing either marginal.

Neither regime enforces the reported all-device `5x` on/off yield. The sampler
reports the achieved count and ratio distribution instead of modifying SET or
RESET samples to force that outcome.

The executable definition is
[`hfo2_figure6_endpoint_regimes.py`](../experiments/mnist_relu_drn/hfo2_figure6_endpoint_regimes.py).
This is an `analyst_defined_figure6_informed_synthetic_control`, not raw
measured-device evidence, a physical microSiemens calibration, or a modified
IBM preset.

Its direct DRN mapping, endpoint-noise HWA, and persistent pulse-recovery
composition are documented in
[`hfo2_figure6_drn_experiment.md`](hfo2_figure6_drn_experiment.md).

The matched follow-up that retains both endpoint distributions while replacing
the programming law with IBM-OM pulse dynamics is documented in
[`figure6_om_pulse_corruption_experiment.md`](figure6_om_pulse_corruption_experiment.md).
