# Figure-6 endpoints with IBM-OM pulse dynamics

This exploratory experiment keeps the complete synthetic Figure-6 endpoint
model and changes only the programming dynamics. For every healthy physical
cell,

```text
G = G_RESET + p (G_SET - G_RESET),       p in [0,1]
raw_a = 2p - 1.
```

The endpoint operations are unchanged between all arms:

```text
full_tile_RESET = max(0.05, 0.1 + sigma_RESET z_RESET)
full_tile_SET   =           2.0 + 0.4295 z_SET
```

RESET has only the lower `0.05` clamp. It has no upper clamp. SET is not
clipped on either side. Independent and rank-matched RESET/SET pairing remain
separate sensitivity regimes.

## OM pulse model

AIHWKit 1.1.0 samples one literal `ReRamArrayOMPresetDevice` identity for
every physical DRN cell. The experiment retains its per-cell `dwmin_up` and
`dwmin_down`, additive soft-bounds response, cycle-to-cycle noise
(`dw_min_std=0.4158`), and apparent write noise
(`write_noise_std=1.4113`). The nominal raw-`a` pulse is `0.0949`, so the
nominal central progress step is `0.04745`.

The Figure-6 endpoint model replaces the OM bound locations. Consequently,
native OM bound device-to-device variation is not applied a second time, and
the OM reference offset is not subtracted. This is a hybrid endpoint/dynamics
model, not an unmodified AIHWKit OM tile.

P&V uses one stochastic OM pulse followed by an apparent verify. Its progress
tolerance is `0.023725`, half the nominal central OM step, and its maximum is
128 target-programming pulses per cell. Healthy exact-RESET targets are
already realized by the declared full-RESET initialization and receive no
target-programming pulses.

## Matched corruption control

The two corruption views are sampled from one assignment seed:

- `published` enables the repository's published OM corruption probability
  (`0.1348`). A corrupt identity retains AIHWKit's native singleton raw-`a`
  coordinate and has zero SET and RESET pulse steps.
- `counterfactual_repaired` replaces only those exact corrupt coordinates
  with deterministic healthy donor identities. Every originally healthy
  bound, reference, and pulse parameter is bitwise identical to the
  `published` view.

A published corrupt cell is not falsely treated as full RESET. The controller
does not see the corruption mask and may spend its pulse budget on the stuck
cell. Because verify observations contain write noise, apparent acceptance is
reported separately from persistent target tolerance.

## Compared paths

For both endpoint-pairing regimes and both corruption views, the executable
compares:

1. direct teacher-direction target followed by P&V;
2. endpoint-distribution HWA target followed by P&V;
3. open-loop stochastic OM-pulse recovery from the direct P&V endpoint; and
4. open-loop stochastic OM-pulse recovery from the HWA P&V endpoint.

HWA is trained once per endpoint-pairing regime and reused unchanged across
the paired corruption views. It injects the RESET and SET endpoint
distributions, but does not inject P&V residuals or corruption. This keeps the
corruption comparison as a deployment intervention rather than changing both
training and hardware at once.

Recovery starts from a bitwise clone of the named persistent P&V endpoint.
Digital BPTT and Adam generate Bernoulli one-pulse commands with learning rate
`3e-5`; the persistent cells can change only through OM SET/RESET pulses.
Recovery performs no verify reads and is therefore open-loop hardware-in-loop
recovery, not autonomous on-chip Adam.

The executable implementation is
[`figure6_om_pulse_corruption_experiment.py`](../experiments/mnist_relu_drn/figure6_om_pulse_corruption_experiment.py),
and the pulse plant is
[`figure6_om_pulse.py`](../experiments/mnist_relu_drn/figure6_om_pulse.py).

## Pilot result (2026-09-04)

The CUDA pilot used all 10,000 MNIST test examples, 8,192 HWA/recovery
training examples, five HWA epochs, and three recovery epochs. The exact run
bundle is under
`simulation_results/figure6_om_pulse_corruption/20260904T100623.123164Z-94d89292-b8cfae43`.
It records the clean staged source commit
`f0468b5faebf9627d5ee168c896538b62ac47625`. A second execution reproduced
the complete scientific payload exactly; only run-specific checkpoint paths
and the hashes of checkpoints containing those paths changed.

The sampled OM assignment contained 21,485 published corrupt cells out of
158,800 (13.5296%). The repaired view retained every healthy identity
bit-for-bit and replaced only those 21,485 positions. The table reports test
accuracy; each P&V/recovery pair is `apparent / persistent`.

| OM view | Endpoint pairing | Logical target | Ideal target | Corruption-constrained target | P&V | Selected recovery |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Published corrupt | Independent | Direct | 95.94% | 60.16% | 48.28% / 52.25% | 73.75% / 85.48% |
| Published corrupt | Independent | HWA | 91.89% | 60.35% | 49.55% / 54.81% | 76.13% / 85.87% |
| Published corrupt | Rank-matched | Direct | 94.99% | 55.17% | 43.09% / 49.02% | 76.22% / 85.23% |
| Published corrupt | Rank-matched | HWA | 90.94% | 60.16% | 52.06% / 56.05% | 79.03% / 83.88% |
| Counterfactual repaired | Independent | Direct | 95.94% | 95.94% | 95.64% / 93.52% | 94.73% / 95.63% |
| Counterfactual repaired | Independent | HWA | 91.89% | 91.89% | 89.97% / 88.58% | 91.97% / 94.07% |
| Counterfactual repaired | Rank-matched | Direct | 94.99% | 94.99% | 93.27% / 92.61% | 93.96% / 95.51% |
| Counterfactual repaired | Rank-matched | HWA | 90.94% | 90.94% | 88.60% / 89.21% | 90.45% / 92.00% |

The published P&V completion fraction was 86.73--87.13%, while repaired P&V
completed 99.992--99.997%. Because the controller accepts noisy apparent
reads, only about 66.0% of healthy persistent endpoints were inside the same
tolerance after P&V. A small fraction of corrupt cells also produced false
apparent acceptances even though their persistent singleton could not move.

With the one-sided `0.05` RESET clamp, 27.85% of RESET samples land exactly at
the floor. The realized RESET standard deviation is `0.06535`, while the
unclipped SET standard deviation is `0.43176`. The endpoint model does not
force the earlier all-device 5x condition: 157,766 of 158,800 independent
pairs and 158,795 of 158,800 rank-matched pairs meet 5x.
