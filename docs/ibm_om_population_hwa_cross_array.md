# IBM-style population HWA for positive-conductance DRNs

## Status and question

The `improved-hwa-drn` branch implements the exploratory sequence

> raw ReLU weights -> independent write on Array A -> array-agnostic HWA ->
> independent writes on Arrays B--D.

The scientific question is whether a clean master trained against ordinary
healthy population programming error transfers better than the former
Array-A-conditioned HWA, and whether published corrupt devices still cause a
material loss after that source-array dependence is removed. No result is
reported here; the implementation and smoke/full protocols are ready, but the
new raw-active characterization and A--D experiment have not been run in this
worktree.

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

## Implemented ladder

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

## Evidence boundary

This uses the AIHWKit 1.1.0 optimized-material preset, not independent raw
device traces. It models successful healthy programming error during HWA and
explicit pulse-resolved deployment with the preset's published corruption
mechanism. It does not include inference read noise, retention, peripheral
nonideality, fabricated-array measurements, or on-chip recovery. A persistent
corruption penalty would show that defects remain significant under this HWA;
it would not by itself show that on-chip training is necessary or sufficient.
