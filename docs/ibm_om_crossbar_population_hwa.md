# Population programming-error HWA for the IBM-OM standard crossbar

- Status: CPU-pinned v2 study complete and artifact-verified; human scientific review pending
- Architecture: analog MVM (`q=a-r`) -- digital ReLU -- analog MVM
- Study: [`mnist-ibm-om-crossbar-population-programming-error-hwa-20260902-v2.json`](../studies/mnist-ibm-om-crossbar-population-programming-error-hwa-20260902-v2.json)
- Results: [`analysis/report.md`](../results/mnist-ibm-om-crossbar-population-programming-error-hwa-20260902-v2/analysis/report.md)
- Device evidence: AIHWKit 1.1.0 `ReRamArrayOMPresetDevice`, not raw measured traces

## Why the earlier HWA control is not population HWA

The existing `stochastic_apparent_hwa` control binds all 39,700 logical
weights to assignment A=87004 for the whole optimization. At every minibatch
it first applies A's exact per-cell logical bounds and then draws

```text
q_forward = clamp(q_master, q_min,A, q_max,A)
            + 1.4113 * 0.0949 * Normal(0, 1).
```

The Gaussian scale is a global OM-preset scalar, not a value fitted separately
from Array A. The important Array-A dependence is its base state: the exact
per-cell support, reference-relative coordinate, and (for published arms)
fixed corrupt map stay attached to each logical weight throughout training.
The noise is also unconditional and therefore does not reproduce a verified
programming endpoint.

IBM's inference-HWA procedure provides the relevant design principles rather
than a drop-in OM law: keep a common normalized weight interval; perturb with
a generic programming-error model instead of one named chip; redraw the
programming realization for each minibatch; use the same realization while
forming forward and backward gradients; and ramp the perturbation strength
during training. IBM's published system additionally includes PCM-specific
error fits, read noise, trainable output scales, input-range handling, and
other inference effects. Those extra mechanisms are outside this first
architecture control.

## Implemented policy

`population_programming_error_hwa` has the following frozen contract:

| Stage | State or operation | Fixed-array information available? |
| --- | --- | --- |
| FP32 master | Clip only to global `q in [-1,1]` | No |
| HWA target | `x_target=(q_master+1)/2` | No |
| HWA realization | Fresh target-conditioned accepted, non-corrupt apparent-endpoint residual for every one of 39,700 weights and every minibatch | No |
| HWA schedule | Multiply the sampled `q` residual by `epoch/10`, giving strengths `0.1,...,1.0` | No |
| Gradient | Teacher-KL at the sampled apparent state, with identity STE to the clean FP32 master | No |
| Deployment | Request the exact fixed-final master through P&V on A and independently remapped B--D | Yes, only after HWA is frozen |
| Defect test | Deploy the byte-identical master on paired repaired and published-corrupt B--D populations | Yes, as the declared intervention |

The sampled HWA error is

```text
residual_x ~ K_healthy,accepted(residual_x | x_target)
q_forward  = q_master + strength(epoch) * 2 * residual_x.
```

`K` comes from the adequate
`adaptive__lower_to_target__tau_step_0.5` condition in the canonical OM
cap-128 bundle `data/ibm_reram_om_pv128_hwa_v1.json`, pinned at SHA-256
`3030e04d6205dc90d0894ac453d2c1c522dffdc004f69b9c6ab7eaf7ef4b8ba3`.
The source study, construction command, adequacy gates, and embedded artifact
hashes are documented in
[`ibm_om_hwa_program_verify_pilot.md`](ibm_om_hwa_program_verify_pilot.md).

The runtime compiles the 41 target-bin, eight-residual-bin density once. It
linearly interpolates adjacent target histograms, performs explicit-generator
inverse-histogram sampling, and records the compiled table fingerprint, RNG
states, per-epoch residual sequence hashes, moments, draw counts, and applied
strength. The HWA seed deliberately excludes source assignment and corruption
policy. Tests require bit-identical training when the supplied fixed-array
bounds and assignment seed are changed.

## What is intentionally not mixed into this arm

Failure terminals, corrupt/stuck identities, exact persistent reachability,
and hidden persistent endpoints are not relabelled as programming error. They
remain explicit deployment outcomes. A separate future defect-augmentation
arm may resample independent defect masks, but it must not be folded into the
healthy-population result after seeing deployment accuracy.

This first policy also does not:

- claim native AIHWKit RNG-stream parity or physically pulse an array during HWA;
- reproduce IBM's PCM-specific programming-error polynomial or measured chip;
- add read noise, drift, retention, converter limits, line resistance, or power;
- add IBM's trainable per-output scaling or input-range adaptation;
- equate the adequate adaptive-controller HWA density with the exact
  deployment's repeated one-pulse verify controller.

The last difference is recorded rather than hidden. Accepted endpoint error
was the only adequate population fit available for this control; failures and
controller-cost differences are evaluated at deployment.

## Decision procedure

The declared four-arm study crosses the existing fixed repaired-A stochastic
HWA and the new population HWA with repaired versus published-corrupt B--D
deployment. Population HWA must improve pooled apparent accuracy by at least
one point, lower teacher KL, improve at least two of three assignment means,
and avoid a greater-than-two-point regression on any assignment to be called
beneficial. Corruption remains material only if the byte-identical population
master loses at least five pooled accuracy points on published B--D, all three
assignment differences are negative, and pooled KL increases.

Apparent post-P&V state is the network forward metric. Hidden persistent state,
acceptance, exhaustion, saturation, pulse cost, and defect counts are reported
separately. Four endpoint seeds on one assignment are repeated writes; B, C,
and D are the three fresh-array units. Even if the corruption gate passes, the
study does not establish that on-chip recovery is required: defect-aware
mapping and spare-resource controls precede a matched recovery fork.

## Artifact-verified v2 results

All four declared runs completed exactly once, and
`python -m ebl study summarize --verify-artifacts` reports
`ready_for_review=true`. Every run required and recorded
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=1`
and verified one Torch intra-op thread. This repaired a provenance omission in
the v1 diagnostic attempt. The fixed-HWA control now reproduces its historical
master exactly at SHA-256
`8a2f31abf08e9eafcb70cb96ae1ae20af5ebb05eb2cd8de6383d9a46c0726cbc`.
The two population arms share master SHA-256
`be562feb59ca51df13d453c181899e051458a948a458bba13ce3001a6fc3ea2a`.
Within each repaired/published pair, the complete off-chip state is
semantically identical: FP32 masters, realized tensors, Adam state, noise or
programming-error RNG states, and epoch reports all match before deployment.

The primary network-forward measurements are:

| HWA policy | Clean fixed-final master test | Repaired B--D apparent test | Repaired KL | Published B--D apparent test | Published KL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed repaired-A support plus additive noise | `96.700%` | `95.250%` | `0.100316` | `90.750%` | `0.257968` |
| Global-bound population programming error | `98.100%` | `96.025%` | `0.068773` | `90.675%` | `0.303733` |

The fresh-array unit is the assignment mean; each value below averages four
paired programming realizations:

| Assignment | Fixed, repaired | Population, repaired | Fixed, published | Population, published |
| --- | ---: | ---: | ---: | ---: |
| B (`87005`) | `95.575%` | `96.875%` | `91.650%` | `91.050%` |
| C (`87006`) | `94.925%` | `95.250%` | `90.775%` | `88.025%` |
| D (`87007`) | `95.250%` | `95.950%` | `89.825%` | `92.950%` |

### Predeclared gates

- **Population HWA versus fixed-A HWA on repaired arrays:** the population
  policy gains `0.775` pooled accuracy points, lowers KL by `0.031543`, and
  improves all three assignment means by `1.300`, `0.325`, and `0.700`
  points. The conjunction nevertheless **does not pass**, because the pooled
  gain is below the predeclared `1.0`-point threshold.
- **Population HWA versus fixed-A HWA on published arrays:** the population
  policy changes pooled accuracy by `-0.075` points, raises KL by `0.045765`,
  and changes B/C/D by `-0.600`, `-2.750`, and `+3.125` points. The benefit
  conjunction **does not pass**.
- **Published corruption under the byte-identical population master:** pooled
  apparent accuracy falls from `96.025%` to `90.675%`, or `-5.350` points;
  B/C/D fall by `-5.825`, `-7.225`, and `-3.000` points; and KL increases by
  `0.234959`. The predeclared corruption-materiality conjunction **passes**.

Published B--D populations contain a mean `13.610%` final corrupt-cell
fraction. Mean apparent acceptance remains high (`39,496.9 / 39,700` cells),
but that is not persistent reachability: mean exhausted cells increase from
`93.7` on repaired arrays to `203.1` on published arrays, while pooled hidden-
persistent test accuracy falls from `71.317%` to `57.992%` (`-13.325`
points). The network result above correctly uses the post-write apparent
state; the hidden state is retained as the robustness diagnostic.

## Evidence boundary

The measurements show that replacing the fixed Array-A support/noise model
with this global-bound, target-conditioned healthy-population HWA does not
remove the published-corruption penalty. They do not establish a universal
IBM-HWA result, because the error density is fitted from the AIHWKit OM preset
rather than raw measured hardware, uses an adaptive-controller accepted-
endpoint branch while deployment uses repeated one-pulse P&V, and omits
inference read noise, drift, retention, converters, line resistance,
trainable output scales, and input-range adaptation. They also do not show
that on-chip recovery is required or sufficient. The managed manifest entry
remains pending the required human choice of scientific outcome and next
experiment.
