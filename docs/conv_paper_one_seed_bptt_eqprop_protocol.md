# One-Seed Paper-Ready BPTT--EqProp Matching Protocol

Updated: 2026-08-17

Status: active matching contract. The shared learning-rate and operating-point
rules are frozen below. EqProp beta selection is still open, so this document
does not yet authorize an official-test evaluation or final paper EqProp
classification.

## Purpose And Precedence

This protocol defines the first seed-0 paper comparison between BPTT and
EqProp for the perfect-diode Conv1, Conv2, and Conv3 models. It is an additive
extension of the
[paper experiment definition](conv_paper_experiment_definition.md). The
architecture, amplification, dataset, optimizer, weight-contract, epoch,
checkpoint, and inclusion rules in that definition remain in force.

For a matched BPTT--EqProp row, this document is authoritative for algorithm
matching and bias treatment. In particular, its exact-zero-bias rule
supersedes the learned-bias policies used while producing the historical
learning-rate handoffs. It does not rewrite those historical artifacts.

The unit of comparison is:

```text
weight contract x architecture x amplification scheme x optimizer x seed
```

Each unit contains one BPTT run and one EqProp run. A one-seed result is
descriptive evidence for seed `0`; it must not be presented as a multi-seed
estimate.

## Frozen One-Seed Contract

- Model seed: `0`.
- Train-loader seed: `0`.
- Paper dataset: ordinary MNIST without affine augmentation or pixel
  permutation, using the frozen `55,000/5,000` train/validation partition.
- Epoch budget: Conv1 `10`, Conv2 `30`, Conv3 `30`.
- Checkpoint selection: maximum validation accuracy.
- Official test: evaluate exactly once from the selected checkpoint after a
  successful complete run; never use it for learning-rate, beta, or checkpoint
  selection.
- Primary comparison: zero endpoint read noise. Read-noise studies are
  separate robustness experiments.

Within every comparison unit, BPTT and EqProp must use hash-identical initial
parameters, split indices, per-example inputs, minibatch order, architecture,
amplification values, preprocessing, output encoding, loss, weight bounds and
initializer, diode parameters, optimizer type and hyperparameters, batch
sizes, and epoch budget. Algorithm-specific gradient construction and the
EqProp nudging parameters are the intended differences.

## Exact-Zero Bias Rule

For both algorithms, every `Bias_*` parameter must:

1. initialize to exact zero;
2. have named and ordered learning rate exactly `0.0`; and
3. remain exact zero in the initial, best, and final checkpoints.

The model may retain bias tensors for implementation compatibility, but this
is a **bias-free / biases-fixed-at-zero** scientific contract. A run with a
nonzero bias learning rate or a nonzero saved bias is ineligible for the
matched paper table. Historical positive-only, signed, or
amplification-scaled learned-bias runs remain separate evidence.

## Learning-Rate Matching Rule

Start from the accepted parameter-specific handoff for the matching
architecture, amplification scheme, optimizer, and weight contract:

- copy every `ConvWeight_*` and `DenseWeight_*` rate unchanged;
- replace every `Bias_*` rate by `0.0`; and
- use that same complete named mapping and ordered vector in BPTT and EqProp.

Equality is exact, not approximate. The two resolved configs must have the
same `parameter_order`, `learning_rates_by_parameter`, and ordered `lr` /
optimizer learning-rate vector. Do not scale an EqProp learning rate by beta,
amplification, gradient norm, or an algorithm-specific factor.

If a weight rate is changed for either algorithm, both members of the pair
must be regenerated and rerun with that same rate. Such a change is a new
learning-rate contract and cannot be mixed with the old member. The rejected
Conv1 legacy factor-three diagnostic does not change the current handoff;
retain the original Conv1 legacy weight rates.

## Shared T/K Rule

Use the existing accepted BPTT operating point for both algorithms, shared
across all amplification schemes and both optimizers within an architecture:

| Architecture | `num_iterations_training` (`T`) | `num_iterations_inference` (`K`) |
|---|---:|---:|
| Conv1 | `4` | `4` |
| Conv2 | `6` | `6` |
| Conv3 | `8` | `8` |

The corresponding resolved iteration fields must be numerically identical in
the BPTT and EqProp configs. A larger EqProp-only `T` or `K` is not a
paper-ready matched comparison. Changing an architecture's operating point
requires requalifying the dependent learning-rate and EqProp-beta handoffs
before either paper member is launched.

## EqProp-Specific Gate

EqProp beta is the one intended algorithm-specific training hyperparameter.
It must be selected on ordinary MNIST, without official-test access, while
using the exact zero-bias learning-rate vector and shared `T/K` above. The
selection must be frozen before any paper test evaluation and must not be
revised from official-test performance. Training/validation metrics may be
used only under the predeclared selection rule.

If one beta is chosen to be safe for both optimizers, that beta must pass the
predeclared stability and gradient-fidelity gates with both SGD and Adam at
the shared contract. A beta qualified for only one optimizer is recorded as
optimizer-specific.

The completed exploratory beta studies do not satisfy this gate unchanged:

- they trained biases with nonzero learning rates; and
- their Conv1 and Conv2 cases used `T=K=8`, rather than `4/4` and `6/6`.

Those results may nominate beta candidates, but Conv1 and Conv2 beta must be
requalified under this protocol. Conv3 also requires a zero-bias recheck at
its already matching `8/8` operating point.

### Current qualification evidence

The 2026-08-16 read-only Adam-first gradient gate now covers all nine
architecture x amplification cases at reconstructed initialization and best
validation, over four fixed 16-example ordinary-MNIST batches. It uses exact
zero biases, the shared `T/K`, centered frozen-current EqProp, true float64,
and the direct all-layer gate cosine `>=.99` plus symmetric norm delta
`<=.10`.

- The one-decade injected-beta tier (`100/30/3`, `100/10/.03`,
  `100/3/.001` for Conv1/2/3 baseline/ours/legacy) passes `213/216`
  layer-batch rows. The three failures are Conv3-initialization
  `ConvWeight_0` rows: one baseline batch and two ours batches.
- The original maximum tier (`1000/300/30`, `1000/100/.3`,
  `1000/30/.01`) passes only `159/216` rows and `9/18` checkpoint contexts;
  its worst cosine is `.801608` and maximum norm delta `.533422`. It is
  gradient-ineligible in addition to its optimizer-dependent training
  failures.
- The two-decade tier (`10/3/.3`, `10/1/.003`, `10/.3/1e-4`) passes all
  `216/216` rows, with worst cosine `.998877` and maximum symmetric norm delta
  `.039421`. It is therefore the current Adam-first beta nomination.
- The subsequent user-directed one-decade Adam training qualification uses
  this protocol's exact zero-bias LR vectors, shared `T/K`, and 10/30/30-epoch
  budgets. All nine cases complete with finite metrics and final validation
  drops of only `0--.14 pp`, so one decade is training-stable under the
  matched contract. This does not reclassify its three direct-gradient
  failures or replace the two-decade gradient nomination.
- Trained Conv3 baseline remains residual-limited after free `T=8`, although
  its direct EqProp--BPTT gradients pass. This is the existing shared-T/K
  caveat and is independent of beta.

This closes the direct Adam gradient-fidelity component, not the full beta
handoff. The one-decade stability branch is complete; before sealing paper
checkpoints or reading the official MNIST test split, run the corresponding
matched zero-bias ordinary-MNIST Adam stability qualification at the
two-decade tier and then freeze the beta decision. An unqualified equilibrium
gate additionally requires either explicit acceptance of the Conv3-baseline
truncated-`T=8` caveat or requalification of a new T/K shared by BPTT and
EqProp. See the
[gradient qualification review](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-tk4-6-8-seed0-20260816-v1/review.md)
and [one-decade training report](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-one-decade-ordinary-mnist-10-30-30ep-seed0-20260816-v1/analysis/report.md).

### Beta and response reporting rule

Beta is the coefficient of the nudging force in the energy; it is not a
prescribed output displacement. In the local regime the state response is set
by both beta and network susceptibility (schematically,
`delta_s = -beta H_E^{-1} grad_s C`, up to the phase sign convention).
Rescaling weights or conductances changes `H_E` and can therefore change the
displacement, the usable beta range, and the read-noise signal-to-noise ratio
even when every numerical beta value is left unchanged.

The beta qualification record and every paper EqProp result must report:

- base beta and the actual injected beta;
- the scheme-specific distance or ratio from the measured gradient-fidelity
  boundary used to nominate that beta;
- absolute and free-state-relative RMS `V_plus - V_zero` and
  `V_minus - V_zero` at the output and each hidden layer, measured on the
  predeclared replay cohort; and
- gradient-fidelity, equilibrium-residual, clean-stability, and read-noise
  evidence as separate quantities rather than reducing them to one beta
  number.

A common relative margin below each scheme's gradient boundary is a
defensible safety rule, but it does not match physical displacement. At the
current one-decade tier, measured output displacement spans `4.89x` across
schemes in Conv1 and `6.50x` in Conv3. Those differences are reported as
susceptibility-dependent responses. They are not normalized away after seeing
paper accuracy. If an equal-output-displacement rule is adopted instead, it
is a new scheme-specific beta contract and must repeat the full qualification
gate before paper training.

## Preflight And Inclusion Gate

Before launching a paper pair, generate a machine-readable comparison receipt
from the two resolved configs. It must fail closed unless all of the following
hold:

- the pair key and seed are identical;
- the initial-parameter hash and all data/order hashes are identical;
- `parameter_order`, the named LR mapping, and the ordered LR vector are
  identical;
- every bias LR and initial bias tensor is exact zero;
- `T` and `K` equal the architecture row above in both configs;
- the epoch budget is `10/30/30` for Conv1/2/3; and
- the only declared scientific differences are the training algorithm and
  frozen EqProp-specific nudging fields.

After completion, require canonical result bundles, finite expected-length
metrics, exact-zero biases in best/final checkpoints, and the predeclared
official-test policy. Every predeclared run remains part of the outcome;
numerical failure or low accuracy is not a post-hoc exclusion rule.

No existing learned-bias medium-affine BPTT run or trained-bias EqProp beta
run is grandfathered into this matched table. Existing ordinary-MNIST
zero-bias runs remain selection evidence until they pass the
[paper experiment reuse gate](conv_paper_experiment_definition.md#existing-ordinary-mnist-reuse-gate).
An eligible checkpoint may then be reused for one sealed official-test
evaluation; its validation metric does not become paper-facing.
