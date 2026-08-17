# Measured cohort-B deployment and fine-tuning

This exploratory phase starts from the explicitly selected named-weights
checkpoint produced by raw cohort-A training. It tests transfer to the other
half of the physical devices without allowing cohort-A traces into the
cohort-B numerical path.

## Deployment semantics

- The HDF5 screening, physical-cell split, seeds, and conductance conversion
  are identical to cohort A. Cohort B contains 632 formed traces from 317
  physical cells for the current dataset.
- Each dense weight receives a deterministic convex interpolation of two
  distinct cohort-B traces. The cohort-specific assignment seed namespace
  makes this mapping independent of cohort A.
- The loaded cohort-A conductance is the desired deployment target. It is
  clamped to the model bounds and globally projected to the nearest point on
  the assigned raw cohort-B curve; ties select the lower pulse index.
- The desired loaded value remains the initial digital shadow while the live
  model receives the projected cohort-B conductance. This matches the
  accumulated desired-update semantics used during cohort-A training.
- Hidden biases remain digital and are loaded unchanged.
- Validation is measured once before remapping and once immediately after
  remapping. Both use only the held-out 5,000-example validation split.

This remains a global-nearest study, not a local or directional sequential
pulse model. The deployment report therefore records target-to-realized
conductance error and pulse-index statistics so that optimistic long jumps are
visible rather than hidden.

## Fine-tuning protocol

Learning rates are selected again after the cohort-B remap. The full
bounded-relative-update probe, 640-minibatch canaries, 3x3 restarted
three-epoch candidate grid, safety gates, and fresh 20-epoch production run
are the same as for cohort A. Every restart reloads the same named cohort-A
checkpoint and repeats the deterministic cohort-B deployment before any
gradient update.

## Command

```bash
python -m ebl train \
  --config examples/small_drn/measured_cohort_b_raw_mnist.json \
  --output-dir results/measured-cohort-b-raw-mnist-exploratory-20260814-v1 \
  --weights results/measured-cohort-a-raw-mnist-exploratory-20260814-v1/<run-id>/checkpoints/weights.pt \
  --device-data data/march_slope_x3_5k.hdf5
```

New cohort-B runs require `--weights`; exact continuation requires
`--resume`. The two initialization sources remain mutually exclusive.

## Exploratory feasibility gate

Treat transfer as viable when the run completes, post-deployment validation
is at least 85%, selected fine-tuned validation is at least 90%, held-out test
accuracy is within five percentage points of raw cohort A, and the late
fine-tuning curve drops by less than five points from its peak. Projection
diagnostics still require physical interpretation even when these accuracy
gates pass.

## Exploratory outcome (2026-08-14)

The raw cohort-B run completed all nine LR candidates and 20 production
epochs. The named cohort-A checkpoint measured 95.10% validation accuracy;
the initial cohort-B write measured 93.64%. Fine-tuning selected epoch 19
(zero-based epoch 18) at 95.22% validation accuracy and reached 95.75% on all
10,000 official test examples, versus 95.97% for the source cohort-A
checkpoint. All automatic transfer gates passed.

The selected learning rates were `2.5922685e-7` for W1 and its bias and
`1.9080487e-8` for W2. The initial A-to-B projection had zero target clipping
and mean absolute conductance errors of 2.55 uS for W1 and 3.78 uS for W2.
Ongoing global-nearest fine-tuning still admitted rare near-full-range pulse
jumps, so this result establishes held-out-device feasibility but does not
validate a sequential local-pulse programming policy.

## Accumulated programming deadband

`measured_cohort_b` optionally separates digital gradient accumulation from
device programming. With
`programming_deadband_mode: accumulated_shadow_relative_rms`, every gradient
still updates the bounded digital shadow, but an individual device is
reprogrammed only after the absolute shadow displacement from its last
programming target reaches its layer threshold. The threshold is fixed at
initialization as `programming_deadband_relative` times that layer's initial
shadow RMS. After a programming request, the current shadow becomes the new
accumulation reference even when global-nearest projection selects the same
pulse index.

This is an accumulated deadband rather than gradient clipping: sub-threshold
updates are retained and can trigger a later write. Checkpoints persist both
the shadow at the last programming request and the absolute per-layer
threshold, so pending updates survive exact resume. Reports distinguish the
fraction of element-steps that request programming from the fraction that
actually change the selected pulse index.

The example
`examples/small_drn/measured_cohort_b_threshold_mnist.json` fixes the learning
rates selected by the 2026-08-14 cohort-B search and uses a relative deadband
of 0.003. This isolates deadband behavior from learning-rate selection; sweep
variants should change only `programming_deadband_relative` and write to a
new result root.

### Deadband sweep outcome (2026-08-15)

A fixed-learning-rate sweep at relative thresholds 0.001, 0.003, and 0.01
completed 20 epochs per arm from the identical cohort-A checkpoint and
deterministic cohort-B deployment. The correct matched no-deadband control is
the fixed-rate run at 93.82% selected validation and 94.56% held-out test
accuracy. The earlier summary incorrectly used the 2026-08-14 automatic-LR
run at 95.22%/95.75% as its control. That older run remains a valid transfer
feasibility result, but its earlier software and LR-selection path make it an
invalid control for the fixed-rate write-policy intervention.

Against the matched control, thresholds 0.001, 0.003, and 0.01 produced
93.82%, 93.80%, and 93.72% selected validation accuracy and 94.52%, 94.52%,
and 94.59% held-out test accuracy. No threshold improved selected validation;
the largest test change was only +0.03 percentage points, or three examples.
Weighted across dense weights, the thresholds reduced programming requests by
39.1%, 72.0%, and 93.3% and actual pulse-index changes by 37.1%, 73.9%, and
94.3%. The corrected machine-readable summary and plot are in
`results/measured-cohort-b-threshold-sweep-20260815-v1/`.

## Probabilistic writing

`measured_cohort_b` also supports two stochastic programming policies. Both
continue to apply every gradient to the bounded digital shadow and only gate
the subsequent physical programming request:

- `uniform_bernoulli` writes each element independently with a fixed
  `probabilistic_write_probability`.
- `displacement_proportional` writes with probability
  `min(1, displacement / scale)`, where displacement is from the last
  programming target and `scale` is `probabilistic_write_scale_relative`
  times the layer's initial shadow RMS.

Suppressed displacement therefore accumulates rather than being discarded.
Each parameter owns a deterministically seeded random generator, and exact
checkpoints persist its state together with the last programmed shadow.
Probabilistic writing is mutually exclusive with the deterministic deadband.
Reports record configured and realized write probability, programming-request
fraction, pending displacement, and actual pulse-index changes. The nested
example `examples/small_drn/measured_cohort_b_probabilistic_mnist.json`
defaults to the selected uniform policy at p=0.85.

### Matched policy outcome (2026-08-15)

Four 20-epoch arms used identical fixed learning rates, initialization,
cohort-B assignments, data order, source checkpoint, and software revision.
Only the write policy changed. Each selected checkpoint was then evaluated on
all 10,000 held-out test examples.

| Policy | Selected validation acc./cost | Test acc./cost | Write requests | Pulse changes |
| --- | ---: | ---: | ---: | ---: |
| Every update | 93.82% / 0.092204 | 94.56% / 0.086236 | 100.00% | 20.34% |
| Deadband, 3e-4 RMS | 93.70% / 0.092434 | 94.60% / 0.086232 | 85.34% | 18.33% |
| Uniform Bernoulli, p=0.85 | 93.80% / 0.091962 | 94.55% / 0.085801 | 85.00% | 17.38% |
| Displacement proportional, scale=5e-4 RMS | 93.72% / 0.092222 | 94.55% / 0.086125 | 87.56% | 18.71% |

Uniform Bernoulli is the most promising tested probabilistic strategy. It
suppressed 15.0% of programming requests and 14.6% of pulse changes while test
accuracy changed by only -0.01 percentage points, one example. It also gave
the lowest selected-validation cost and lowered held-out test cost by 0.50%.
Displacement-proportional writing saved fewer writes and did not improve on
uniform accuracy or cost. The deadband's +0.04-point test change is only four
examples and coincides with lower selected-validation accuracy, so it is not
evidence of an accuracy benefit.

This is a single-training-seed exploratory comparison under global-nearest
programming. It supports uniform p=0.85 as the next policy to replicate across
seeds; it does not yet establish a statistically reliable accuracy gain. The
controlled summary and plot are in
`results/measured-cohort-b-write-policy-comparison-20260815-v1/`.
