# Conv1 hard-sigmoid finite-conductance diagnostic on ordinary MNIST

Updated: 2026-07-25

Status: complete, diagnostic only.

## Correction

An initial replay used the active deterministic-medium-affine v1/v3
learning-rate-screen checkpoints and merely changed their evaluation loader to
ordinary MNIST. Those checkpoints produced clean accuracies of 87.21%,
71.30%, and 87.73% for baseline, legacy, and ours, respectively. They were the
wrong checkpoint family for the requested ordinary-MNIST comparison and are
superseded by this result.

The corrected replay uses the three selected Conv1 entries from the completed
v5 ordinary-MNIST diagnostic: historical-profile arm, architecture alpha
`1e-3`, and each entry's immutable best-validation checkpoint.

## Method

No model is retrained and no optimizer step is taken. For every requested
`(Gmin,Gmax)` cell, the evaluator:

1. reloads the exact saved best-validation checkpoint state;
2. clamps only `ConvWeight` and `DenseWeight` tensors;
3. leaves `Bias` tensors unchanged;
4. evaluates all 10,000 ordinary-MNIST test examples with fixed `T=K=4`,
   `adaptive_equilibrium=false`, and reset state per batch;
5. reports accuracy loss relative to an unclipped replay of that same
   checkpoint;
6. restores the pristine tensors before the next cell and verifies the source
   checkpoint SHA-256 before and after replay.

## Checkpoints and clean accuracy

| Scheme | Best-validation epoch | Checkpoint SHA-256 | Clean test accuracy |
|---|---:|---|---:|
| baseline `v1/c1` | 4 | `6ba07835051e508635c960ba7229acc527a3164f680ad2b10b257970112777c5` | 95.37% |
| legacy `v4/c0.25` | 4 | `9eaeff46d72dbb78ff40f56cfac40677cbfda59fbcfc4c6b4cab905fe841855e` | 95.18% |
| ours `v4/c1` | 3 | `8ee49482414803aa963dd9d252020a1ff74540c367f13e5460d1f311d3289b36` | 96.07% |

## Minimum-conductance floor

These are discrete-grid brackets. "Largest floor" is the largest tested
`Gmin` whose accuracy loss is at most one percentage point. "Next floor" is
the next larger tested value.

| Scheme | Largest tested floor within 1 pp | Accuracy loss | Next floor | Accuracy loss |
|---|---:|---:|---:|---:|
| baseline | `3e-3` | 0.10 pp | `1e-2` | 3.41 pp |
| legacy | `1e-3` | 0.60 pp | `3e-3` | 9.50 pp |
| ours | `3e-3` | 0.89 pp | `1e-2` | 21.42 pp |

Legacy is the floor-limiting scheme.

## Maximum-conductance ceiling

"Smallest ceiling" is the smallest tested positive `Gmax` whose accuracy loss
is at most one percentage point. "Next ceiling" is the next smaller tested
value.

| Scheme | Smallest tested ceiling within 1 pp | Accuracy loss | Next ceiling | Accuracy loss |
|---|---:|---:|---:|---:|
| baseline | `0.15` | 0.34 pp | `0.1` | 1.99 pp |
| legacy | `0.3` | 0.00 pp | `0.2` | 1.80 pp |
| ours | `0.15` | 0.60 pp | `0.1` | 8.61 pp |

Legacy is the ceiling-limiting scheme.

## Shared sampled window

The conservative common sampled window inferred from the one-sided sweeps is
`[Gmin,Gmax]=[1e-3,0.3]`. Its direct full-test replay gives:

| Scheme | Clean accuracy | Windowed accuracy | Accuracy change |
|---|---:|---:|---:|
| baseline | 95.37% | 95.58% | +0.21 pp |
| legacy | 95.18% | 94.58% | -0.60 pp |
| ours | 96.07% | 95.94% | -0.13 pp |

All three remain within one percentage point of their own unclipped replay.
The small baseline improvement is a clipping perturbation on a finite test
set, not evidence that imposing a floor is generally beneficial. This is a
checkpoint- and sampled-grid-dependent diagnostic bracket, not a continuous
hardware limit.

## Artifacts

- Full floor sweep:
  `simulation_results/conv1_hardsigmoid_finite_conductance_v5_ordinary_mnist_20260725/analysis/min_floor_full/`
- Full ceiling sweep:
  `simulation_results/conv1_hardsigmoid_finite_conductance_v5_ordinary_mnist_20260725/analysis/max_ceiling_full/`
- Full common-window replay:
  `simulation_results/conv1_hardsigmoid_finite_conductance_v5_ordinary_mnist_20260725/analysis/common_window_full/`
- Evaluator:
  `labs/tools/evaluate_conv1_hardsigmoid_finite_conductance.py`

The initial medium-affine-checkpoint ordinary-MNIST replay remains only as
superseded provenance under
`simulation_results/conv1_hardsigmoid_finite_conductance_active_handoff_ordinary_mnist_20260725/`.
