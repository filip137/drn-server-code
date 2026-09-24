# Tunable Weight-Write Noise Model

Updated: 2026-08-27

Status: reviewed and implemented proposal for validation-only deployment
diagnostics. This is not yet a frozen paper protocol. The source-checkpoint
set, noise grid, write seeds, inclusion rule, and official-test handoff remain
to be frozen before any paper-facing evaluation.

## Decision

Use a one-time additive Gaussian programming error on physical conductance
weights:

```text
sigma_G = eta * (G_max - G_min)
Z_i ~ Normal(0, 1)
G_written_i = clip(G_target_i + sigma_G * Z_i, G_min, G_max)
```

`eta >= 0` is the tunable full-scale write-noise standard deviation. For
example, `eta=0.01` requests an RMS programming error equal to 1% of the
configured conductance span before clipping. The model is phenomenological:
it is a controlled robustness axis, not a calibrated claim about one specific
PCM or ReRAM technology.

This is a useful first model because measured analog-memory programming errors
are commonly represented as Gaussian conductance errors, and prior
hardware-aware work normalizes their scale to the available conductance range.
More detailed device models make the error standard deviation depend on the
target conductance; the constant-full-scale model is the smallest uncalibrated
special case.

Primary sources:

- [Joshi et al., *Accurate deep neural network inference using computational phase-change memory*](https://www.nature.com/articles/s41467-020-16108-9)
  define a Gaussian conductance error and normalize its standard deviation by
  full-scale conductance.
- [Rasch et al., *Hardware-aware training for large-scale and diverse deep learning inference workloads using in-memory computing-based accelerators*](https://www.nature.com/articles/s41467-023-40770-4)
  use a measured, conductance-dependent Gaussian programming-error model; the
  proposal above intentionally collapses that curve to one tunable scale.
- [Wan et al., *A compute-in-memory chip based on resistive random-access memory*](https://www.nature.com/articles/s41586-022-04992-8)
  report approximately Gaussian conductance relaxation over most programmed
  states and show why rail behavior must be measured separately.

## Exact Semantics

- Perturb only `ConvWeight_*` and `DenseWeight_*`. Biases and amplification
  parameters are not conductance weights and remain unchanged.
- Draw one programmed array per source checkpoint and programming seed. Hold
  it fixed for every example, batch, equilibrium iteration, and free or
  nudged phase. Resampling on every forward pass would model read noise, not
  write noise.
- Apply the perturbation to the selected clean checkpoint, not cumulatively
  between noise levels or seeds.
- Hard-clip once to each parameter's explicit finite `G_min/G_max`. Do not
  reflect, redraw, or silently expand the hardware interval.
- Generate the standard-normal tensors from a stable key containing the model
  version, programming seed, canonical parameter name, and tensor shape. The
  key deliberately omits `eta`, so every noise level follows the same random
  direction. The same named draw is reused across matched amplification
  schemes.
- Generate draws on CPU in float64 and cast them to the runtime tensor only
  after generation. This makes the common-random-number contract independent
  of execution device and runtime precision up to the final cast.

The same programmed conductances must be used in both phases of an EqProp
evaluation. A future on-chip-training study is a different contract: it must
define whether every optimizer update causes a new absolute or incremental
write, when noise is redrawn, and how optimizer state interacts with the
written value.

## Why The Historical Lognormal Model Is Secondary

[`evaluate_mnist_bp_write_noise_sweep.py`](../experiments/evaluate_mnist_bp_write_noise_sweep.py)
implements persistent

```text
G_written = clip(G_target * exp(sigma * Z), G_min, G_max).
```

That evaluator is historical and must not be used unchanged for active Conv
evidence: it is wired to the old result summary and loader interface, labels
the evaluation split as test, optionally perturbs biases, and does not record
matched draw hashes.

Its model can remain a relative-noise sensitivity control, but it is not the
recommended primary coordinate:

- `sigma` is a log-space standard deviation rather than a direct full-scale
  error fraction;
- the pre-clipping mean is multiplied by `exp(sigma^2 / 2)`, so the
  perturbation also shifts the conductance scale upward;
- exact-zero conductances never move; and
- schemes with different conductance distributions receive different
  absolute error scales.

If a new relative-noise control is added, use a separately versioned,
mean-preserving lognormal model and parameterize it by coefficient of
variation. Do not reinterpret historical `sigma` values.

## Weight Contracts And Comparability

The bounded `[1e-5,1e-4]` hardware contract is the primary physical target.
It has explicit nonzero bounds and a meaningful finite programming range. The
wide `[0,100]` condition may be evaluated as a separate reference, but it must
not be pooled with the bounded condition.

Equal `eta` means equal fractional programming precision within a configured
range. It does not mean equal noise in siemens across the wide and bounded
contracts. A same-device comparison across contracts would require an
explicit conductance-unit mapping and a common absolute `sigma_G`.

Clipping is part of the measured response, not a nuisance to normalize away.
The amplification schemes can have substantially different rail occupancy,
so every realization records both the requested and realized perturbation:

- target and written lower/upper rail occupancy;
- pre-clipping lower/upper overflow fractions;
- newly clipped fractions among targets that were initially interior;
- signed mean error, MAE, and RMS error in conductance units and as a fraction
  of span; and
- the same statistics per named weight tensor.

Do not rescale a realization after clipping to force equal empirical RMS
errors across schemes. That would change the programming model and erase a
real interaction with conductance placement.

## Matched Amplification Experiment

Within one architecture x optimizer x weight-contract x source-seed block,
hold fixed:

- dataset realization and preprocessing;
- source initialization, training order, budget, and selected checkpoint;
- input gain, `T/K`, perfect-diode dictionaries, parameter-specific learning
  rates, and training algorithm; and
- target/device class and evaluation precision.

Change only amplification scheme, `eta`, and programming seed. Use the same
named standard-normal tensors across baseline `v1/c1`, ours `v4/c1`, and
legacy `v4/c0.25`. Do not increase `T` for a noisy scheme: replay the frozen
operating point and retain non-finite states or residual-gate failures as
scientific outcomes.

Noise-level selection uses only the deterministic 5,000-example ordinary-
MNIST validation split. It must not instantiate the official test split. A
reasonable predeclared starting grid is:

```text
eta = {0, 0.0025, 0.005, 0.01, 0.02, 0.05}
programming seeds = 30 per nonzero eta
```

Evaluate `eta=0` once per source checkpoint. Permit one extension to `eta=0.1`
only when the starting grid does not bracket a one-percentage-point accuracy
drop. A smaller five-seed execution is a smoke or variance pilot, not the
terminal robustness comparison.

Analyze architectures, optimizers, weight contracts, initializer families,
and checkpoint handoff states separately. Report at minimum:

- validation accuracy, loss, and clean-relative accuracy drop;
- paired scheme differences under identical programming seeds;
- mean, standard deviation, and confidence interval across seeds;
- the complete per-layer programming receipt described above;
- output margin, hidden saturation, and fixed-`T` residual/failure metrics;
- normalized area under the accuracy-versus-`eta` curve; and
- the interpolated `eta` for a one-percentage-point accuracy drop only when it
  is bracketed, with no extrapolation.

## Execution And Evidence Gate

The reusable programming primitive is
[`experiments/weight_write_noise.py`](../experiments/weight_write_noise.py).
The active-Conv replay entry point is
[`experiments/evaluate_conv_weight_write_noise.py`](../experiments/evaluate_conv_weight_write_noise.py).
It accepts only complete, non-smoke, perfect-diode Conv1/2/3 sources built on
the ordinary-MNIST train/validation loader. It fingerprints the source files,
noise model, evaluation budget, and device in every case identity; a complete
smoke result therefore cannot be silently reused as a full-validation result.
Non-finite equilibrium, score, cost, or residual behavior is retained as a
terminal scientific outcome, while malformed inputs and reporting failures
remain operational failures.

A one-batch real-checkpoint smoke has the form:

```text
python -m experiments.evaluate_conv_weight_write_noise \
  --source-run PATH_TO_CANONICAL_SOURCE_RUN \
  --output-root PATH_TO_SMOKE_OUTPUT \
  --dataset-root PATH_TO_MNIST \
  --std-fractions 0 0.01 \
  --programming-seeds 0 \
  --max-validation-batches 1
```

Omitting the explicit noise grid and seeds uses the proposed validation grid
and 30 programming seeds above. A production invocation still requires its
study row and frozen source/inclusion contract.

The active paper checkpoint and bounded-initializer handoffs are incomplete,
so a write-noise run now is an ordinary-MNIST validation diagnostic, not
paper-facing accuracy evidence. Before a production run:

1. Freeze a readable config containing the formula/version, bounds,
   persistence, RNG key, source checkpoint/result hashes, `eta` grid, seeds,
   inclusion rule, and `official_test_read=false`.
2. Add the top-level result directory to `current_simulations.md` as
   `planned` before creating it.
3. Run a same-runner smoke that loads a real checkpoint and validation batch,
   evaluates both zero and nonzero noise, and writes canonical artifacts.
4. Give every source-checkpoint x `eta` x programming-seed arm the canonical
   `manifest.json`, `status.json`, `metrics.jsonl`, and successful-only
   `result.json` bundle, including a `noise_receipt.json`.
5. Keep the official MNIST test split sealed until the complete source set,
   noise grid, seeds, inclusion rule, and checkpoint identities are frozen.

The implementation must be validated independently of any scientific run:
zero noise is bitwise identity, seeds are reproducible, parameter order and
device do not change named draws, matched schemes receive identical standard
normals, biases remain unchanged, and rail/error receipts agree with the
written tensors.
