# IBM-style population HWA for positive-conductance DRNs

## Status and question

The `improved-hwa-drn` branch implements the exploratory sequence

> raw ReLU weights -> independent write on Array A -> array-agnostic HWA ->
> independent writes on Arrays B--D.

The scientific question is whether a clean master trained against ordinary
healthy population programming error transfers better than the former
Array-A-conditioned HWA, and whether published corrupt devices still cause a
material loss after that source-array dependence is removed. The full
exploratory programming run completed on 2026-09-02, but its headline
evaluation used the hidden persistent endpoints. Those values are retained
below as a secondary state diagnostic. The frozen apparent-state replay
completed on 2026-09-03 and is now the primary inference result. On B--D,
continuous and one-delta population HWA reduced repaired apparent-state
accuracy by 15.10 and 18.72 percentage points relative to raw ReLU. Published
corruption still cost 27.95 and 26.19 points in those HWA arms.

All reduced runs are `exploratory_noncanonical`.

## Compatibility with the population-HWA handoff

The method is compatible with
`revised_HWA_TRAINING/HANDOFF.md` at the population-HWA level, but it is not a
bit-for-bit reuse of the standard-crossbar coordinate. The topology adaptation
is required because each signed logical ReLU weight is represented by four
independently programmed, nonnegative DRN conductances.

| Contract item | This DRN implementation |
|---|---|
| Clean global master | Signed normalized `u` in `[-1,1]`, initialized from each raw ReLU layer divided by its absolute maximum |
| Positive physical mapping | `G=2*(relu(u), relu(-u), relu(-u), relu(u))`, so every circuit weight is in `[0,2]` |
| Programming coordinate | One target `x=G/2` for every physical cell |
| Programming-error law | Target-conditioned accepted healthy apparent-endpoint residual from the raw-active OM population |
| Sampling scope | Fresh independent residual per physical cell and minibatch; one realization is retained through forward and backward |
| Optimizer state | Only the clean FP32 logical masters; the temporary noisy conductances are discarded |
| Ramp | One-indexed epoch strengths `0.1, 0.2, ..., 1.0` over ten epochs |
| Deployment | Each array independently owns its assignment, bounds, shared baseline, one-delta codebook, and pulse trajectory |
| Defects | Excluded from HWA and evaluated as a paired repaired-versus-published-corrupt deployment intervention |

The source handoff samples one apparent effective signed state per logical
weight. Applying that residual once per logical weight would be incorrect for
the DRN topology: the four constituent devices are separately programmed. The
implementation consequently samples all four cells and differentiates through
their actual positive-conductance realization with a sign-selected
straight-through lift back to `u`.

### Declared range adaptation

The source handoff leaves apparent samples outside its nominal signed interval
visible. A passive DRN cannot consume `G<0` or `G>2`, and the requested mapping
requires the physical circuit weights to remain in `[0,2]`. The port therefore
uses rejection-resampling for an apparent raw-active draw outside `x=[0,1]`.
It never silently clips the sample. Rejection rounds and rejected scalar draws
are recorded per epoch and in the checkpointed sampler state. This is a
hardware-specific conditional-endpoint policy and a known difference from the
standard-crossbar handoff.

## Original implemented ladder

1. Load the exact frozen ReLU teacher checkpoint and create clean normalized
   logical masters. No newest-checkpoint discovery is allowed.
2. Sample Array A, build its repaired shared-destination one-delta mapping, and
   independently program/evaluate the raw ReLU state on paired repaired and
   published-corrupt populations.
3. Train two fresh copies of the same clean master using only the immutable
   healthy raw-active endpoint population:
   `population_hwa_continuous` and a separate
   `population_hwa_one_delta_qat` quantization control.
4. Freeze and hash both fixed epoch-10 checkpoints. There is no best-epoch or
   deployment-array selection.
5. Only after both hashes exist, sample Arrays B, C, and D. Independently map
   and program the raw, continuous-HWA, and one-delta-HWA masters on each
   array.
6. Evaluate the persistent full conductance `G=2x`. Apparent verify values are
   controller observations and HWA samples; they are not substituted for the
   deployed persistent state in DRN inference.
7. Report HWA-minus-raw transfer, quantization-minus-continuous HWA, and the
   repaired-minus-published-corrupt penalty separately for every array and
   over B--D.

Steps 6--7 describe the already completed persistent-state evaluation. They
must not be interpreted as the primary deployed inference result now that the
saved post-write apparent endpoints are available.

Array A is deliberately a pre-HWA baseline, not an HWA input. Its sampled
bounds, references, identities, corruption mask, programming endpoint, and
random streams never enter the population sampler. B--D assignments are not
opened before HWA checkpoint freeze.

## Characterization input

The previous artifact `data/ibm_reram_om_pv128_hwa_v1.json` is
reference-relative (`x=(w+1)/2`) and cannot be relabeled for the four raw-active
positive cells. A new healthy OM characterization is required. The added
configuration runs the same adaptive lower-to-target, half-step tolerance,
128-pulse, 41-target contract in the native raw-active coordinate while
retaining the sampled reference only as unused provenance:

```bash
python -m ebl characterize \
  --config examples/reram_program_verify/hwa_production_cap128_om_raw_active_continuous.json \
  --output-dir results/exploratory_noncanonical-ibm-om-raw-active-characterization
```

The HWA experiment accepts the resulting
`artifacts/bounded_uniform_model.json` as `--device-model` and its matched
`artifacts/step_estimators.json` as `--device-data`. It verifies their hashes,
trajectory provenance, coordinate, OM preset and AIHWKit version, healthy
population policy, controller, pulse cap, and estimator fallback step before
creating the run.

## Execution

A small configuration validates the end-to-end path with one training batch
per epoch, 64 evaluation examples, and one programming endpoint per array:

```bash
python -m ebl train \
  --config examples/mnist_relu_drn/ibm_om_population_hwa_cross_array/smoke.json \
  --output-dir results/exploratory_noncanonical-ibm-om-population-hwa-smoke \
  --teacher-weights data/mnist_relu_teacher_fixed_init_20260816.pt \
  --device-model <raw-active-characterization>/artifacts/bounded_uniform_model.json \
  --device-data <raw-active-characterization>/artifacts/step_estimators.json
```

Replace `smoke.json` with `full.json` for all MNIST training batches, the full
test split, and four predeclared endpoint seeds per array. Both protocols use
the fixed assignments A=`87004`, B=`87005`, C=`87006`, D=`87007` and preserve
the layer learning rates from the preceding DRN HWA ladder. The full run is
still exploratory rather than canonical evidence.

## Apparent-state replay protocol

The diagnostic is a read-only replay of all 96 saved endpoints from
`20260902T174313.144856Z-4602e186-5b00f3fc`. It freezes the teacher checkpoint
(`9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52`),
MNIST split and order (runtime/data seeds 42), arrays A--D (assignment seeds
87004--87007), all four endpoint seeds per array, logical HWA state, repaired
versus published-corrupt population, fixed logit gain, DRN topology, and
solver. It invokes no programming, resampling, optimizer update, or fresh
inference noise.

For each endpoint, one saved post-write apparent sample is held fixed for the
entire test-set forward. Native raw-active apparent values are mapped by
`x=(a+1)/2`. Because the passive DRN and the requested raw-ReLU mapping require
nonnegative conductances in `[0,2]`, inference uses the explicit passivity
projection `G=2*clamp(x,0,1)`. The literal unbounded apparent values, their
out-of-range rate, accepted out-of-range count, and projection displacement
are retained as diagnostics. This projection is a DRN-specific inference
policy, not another programming operation and not a claim of literal
unbounded AIHWKit forward parity.

The primary readout is full-test apparent-state accuracy and the paired
repaired-minus-corrupt penalty on untouched Arrays B--D. HWA-minus-raw and
one-delta-minus-continuous comparisons remain separate. Persistent accuracy is
reported on the identical examples and settings, and one full endpoint must
replay its saved persistent prediction hash exactly before the apparent result
is accepted. This direct CUDA replay is `exploratory_noncanonical` and remains
a model-based AIHWKit 1.1.0 OM diagnostic rather than measured-hardware
evidence.

## Completed apparent-state exploratory result

The full replay
`20260903T090511.037538Z-e7f05444-b3c9bd37` completed from clean commit
`41c0cd49f182a7d76c2fbeb42cc3e8ecc4505cd8` in 202.51 seconds on the local
RTX 3090. It evaluated all 96 unique saved endpoint identities, exactly 24 per
array and 10,000 test examples per endpoint. The first persistent endpoint
reproduced its saved 10,000-example prediction hash exactly before apparent
evaluation. The scientific-summary SHA-256 is
`4d4393bf480f8257f6bb493b37985908f6dcd6f1a95499952ad585fd5a7817ae`.

| Logical state | Repaired B | Repaired C | Repaired D | Repaired B--D mean | Change from raw |
|---|---:|---:|---:|---:|---:|
| Raw ReLU master | 94.2550% | 93.1500% | 92.7775% | 93.3942% | -- |
| Continuous population HWA | 78.1300% | 79.4300% | 77.3275% | 78.2958% | -15.0983 pp |
| One-delta population HWA/QAT | 77.2750% | 74.7850% | 71.9625% | 74.6742% | -18.7200 pp |

| Logical state | Apparent repaired B--D | Apparent corrupt B--D | Apparent corruption penalty | Persistent repaired B--D | Persistent corrupt B--D | Persistent corruption penalty |
|---|---:|---:|---:|---:|---:|---:|
| Raw ReLU master | 93.3942% | 47.3300% | 46.0642 pp | 81.1725% | 40.4292% | 40.7433 pp |
| Continuous population HWA | 78.2958% | 50.3500% | 27.9458 pp | 65.3775% | 46.7283% | 18.6492 pp |
| One-delta population HWA/QAT | 74.6742% | 48.4800% | 26.1942 pp | 65.3825% | 44.9467% | 20.4358 pp |

Using the apparent rather than persistent state raises repaired B--D accuracy
by 12.2217, 12.9183, and 9.2917 points for raw, continuous HWA, and one-delta
HWA, respectively. It raises corrupt accuracy by only 6.9008, 3.6217, and
3.5333 points. Apparent inference therefore does not make the defect problem
disappear; it increases the paired corruption gap for every logical state.

HWA still fails the repaired-transfer comparison. Its smaller corruption gap
than raw ReLU must not be called defect robustness: most of that reduction
comes from lowering the repaired reference by 15--19 points, while corrupt
accuracy changes by only +3.02 points for continuous HWA and +1.15 points for
one-delta HWA relative to raw ReLU.

Across B--D, 732,525 of 11,433,600 saved apparent cell-endpoint values (6.4068%)
fell outside nominal `x=[0,1]`; 98.73% of those were below zero and 96.35% had
been accepted by noisy verify. The passivity projection had RMS displacement
0.00458 in raw `x`, or 0.00916 in `G`, while literal extrema were -0.24294 and
1.21552. Thus the projected apparent result is the correct positive-`G` DRN
diagnostic requested here, but it is not a literal unbounded AIHWKit forward.

The result strengthens the preliminary indication that this passive DRN is
more defect-sensitive than the earlier crossbar-plus-digital-ReLU smokes.
It is not yet a matched architecture proof: those crossbar smokes used
different tensor sizes, device populations, and adaptation histories. A
neck-to-neck causal architecture claim still requires the already specified
matched topology/device/codebook/programming control.

The scientific conclusion is therefore unchanged but better grounded:
array-agnostic population HWA does not improve clean B--D transfer under this
training protocol, and published corrupt devices remain a large independent
problem under the primary apparent-state inference rule. The next training
diagnostic remains a matched ten-epoch clean/no-modifier optimizer control;
on-chip recovery is not implied by this replay alone.

## Completed persistent-state exploratory result

The prerequisite characterization, smoke, and full run completed on the local
RTX 3090 through the host-level CUDA path:

- characterization run
  `20260902T173750.508771Z-a8b27fe4-2fce664a`: 201,392 trajectories,
  1,024 healthy identities, four repeats, both 41-target controller blocks,
  and a passing SQLite/invariant integrity report;
- bounded endpoint model SHA-256
  `82df08f1251ca1dfa992dd1eeb6a3d3ef2d9533d1a10d00e171c83cffd1d1bbf`;
- step-estimator SHA-256
  `ca17553cd7fc6b1f1f4524078474ce2cc69502a3cb25220be0f3c7af2f0f5271`;
- smoke run `20260902T174023.488270Z-0a83b36b-ca3d0789`, which completed all
  24 declared canary deployments and passed every ordering invariant; and
- full run `20260902T174313.144856Z-4602e186-5b00f3fc`, launched from clean
  commit `b149ac3ce465f44fc6a4f245c000644ba00c9716`, which completed in
  1,761.52 seconds. All 127 declared artifacts (1,294,448,640 bytes) passed
  size and SHA-256 verification. Coverage is exactly 96 deployments: four
  endpoint seeds for every array, logical state, and repaired/corrupt role.

The following accuracies are arithmetic means over the four endpoint seeds in
each array and then over untouched Arrays B--D:

| Logical state | Clean global test | Repaired B | Repaired C | Repaired D | Repaired B--D mean | Change from raw |
|---|---:|---:|---:|---:|---:|---:|
| Raw ReLU master | 96.88% | 79.6050% | 80.8625% | 83.0500% | 81.1725% | -- |
| Continuous population HWA | 92.04% | 66.4575% | 64.6575% | 65.0175% | 65.3775% | -15.7950 pp |
| One-delta population HWA/QAT | 91.93% | 68.2650% | 65.8400% | 62.0425% | 65.3825% | -15.7900 pp |

| Logical state | Published-corrupt B--D mean | Paired repaired-minus-corrupt penalty |
|---|---:|---:|
| Raw ReLU master | 40.4292% | 40.7433 pp |
| Continuous population HWA | 46.7283% | 18.6492 pp |
| One-delta population HWA/QAT | 44.9467% | 20.4358 pp |

Every saved invariant passed: all physical conductances stayed in `[0,2]`,
Array A bounds/state/reference/corruption never entered HWA, Arrays B--D were
not sampled until both fixed epoch-10 checkpoint hashes existed, each physical
cell received a fresh healthy endpoint residual per minibatch, forward and
backward shared that realization, and deployment inference consumed persistent
full `G=2x` rather than the apparent verify endpoint. The continuous and QAT
checkpoint SHA-256 values are respectively
`9ee8d45533f65539be13efd116e92750e3d06c870b926118115c66b83344b50a`
and
`d8e68b5b20eb56da84bddcb0f6a7ff5fbf01a0af9c12b3eeed8c4360286b7f48`.

The primary transfer hypothesis is therefore not supported by this frozen
implementation. Removing Array-A conditioning was not sufficient; the HWA
optimization itself reduced clean global accuracy by about five points before
target-array programming and produced nearly identical repaired B--D means for
continuous and one-delta training. Rejection-resampling discarded 31.20% and
32.01% of scalar apparent draws in the two arms, respectively. That is a
declared no-clipping policy, but it is now an important mechanism diagnostic.

Corrupt devices nevertheless remain significant: their matched penalties are
still roughly 19--20 points after HWA. The smaller penalty than the raw arm
does not mean corruption is solved, because the repaired HWA reference state
also became much worse. These data neither establish that corruption is the
only remaining limitation nor justify on-chip recovery. The next matched
diagnostic should add a ten-epoch clean/no-modifier optimizer control with the
same minibatches and learning rates, then inspect residual bias,
rejection-conditioning, and ramp/learning-rate sensitivity without selecting
on B--D.

## Evidence boundary

This uses the AIHWKit 1.1.0 optimized-material preset, not independent raw
device traces. It models successful healthy programming error during HWA and
explicit pulse-resolved deployment with the preset's published corruption
mechanism. It does not include inference read noise, retention, peripheral
nonideality, fabricated-array measurements, or on-chip recovery. The observed
persistent and apparent corruption penalties show that defects remain
significant under this HWA; they do not by themselves show that on-chip
training is necessary or sufficient. The primary apparent result also
includes the declared positive-G passivity projection and cannot be relabelled
as literal unbounded AIHWKit inference.
