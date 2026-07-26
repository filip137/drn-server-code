# Conv1 hard-sigmoid finite-conductance diagnostic

Updated: 2026-07-25

Status: complete, diagnostic only.

## Question

Measure the accuracy loss caused by imposing finite raw conductance bounds on
the selected Conv1 hard-sigmoid checkpoints. No model is retrained and no
optimizer step is taken.

For every requested `(Gmin,Gmax)` cell, the evaluator:

1. reloads the exact saved best-validation checkpoint state;
2. clamps only `ConvWeight` and `DenseWeight` tensors;
3. leaves `Bias` tensors unchanged;
4. evaluates all 10,000 deterministic medium-affine MNIST test examples with
   fixed `T=4`, `adaptive_equilibrium=false`, and reset state per batch;
5. reports accuracy loss relative to an unclipped replay of that same
   checkpoint;
6. restores the pristine tensors before the next cell and verifies the source
   checkpoint SHA-256 before and after replay.

This separate diagnostic reads the official test split. It is not an LR or
checkpoint-selection stage, does not alter the active seed-0 LR handoff, and
does not authorize final paper training.

## Checkpoints

The checkpoint rule is `best_validation`. For all three selected candidates,
the best-validation epoch is epoch 5, which is also the final epoch.

| Scheme | Source selection | Frozen peak LR | Checkpoint SHA-256 | Clean test accuracy |
|---|---|---:|---|---:|
| baseline `v1/c1` | v1 `fast` | `0.30174007288404703` | `be30747ae70d42ed40fa1a35014ccdbad236794f73e76444d600c94565ff7745` | 66.23% |
| legacy `v4/c0.25` | v3 `high` | `4.639114646319357e-5` | `28ac8d828ae290016e10a379e8f6ee9ba37c67199ba1d3e437d82803adb5a6dc` | 56.13% |
| ours `v4/c1` | v3 `middle` | `0.001838414260143094` | `1b18cfb3089974a9394addb301e7522eca886cbc7e6751c415b2ef5bc584cee3` | 68.96% |

## Minimum-conductance floor

The values below are discrete-grid brackets. "Largest floor" is the largest
tested `Gmin` whose accuracy loss is at most one percentage point. "Next
floor" is the next larger tested value.

| Scheme | Largest tested floor within 1 pp | Accuracy loss | Next floor | Accuracy loss |
|---|---:|---:|---:|---:|
| baseline | `3e-3` | 0.34 pp | `1e-2` | 2.18 pp |
| legacy | `1e-4` | 0.64 pp | `3e-4` | 2.97 pp |
| ours | `3e-4` | 0.26 pp | `1e-3` | 1.43 pp |

Legacy is the floor-limiting scheme. At `Gmin=1e-3`, baseline changes from
66.23% to 66.32% (+0.09 pp), legacy falls from 56.13% to 37.89% (-18.24 pp),
and ours falls from 68.96% to 67.53% (-1.43 pp).

## Maximum-conductance ceiling

These are also discrete-grid brackets. "Smallest ceiling" is the smallest
tested positive `Gmax` whose accuracy loss is at most one percentage point.
"Next ceiling" is the next smaller tested value.

| Scheme | Smallest tested ceiling within 1 pp | Accuracy loss | Next ceiling | Accuracy loss |
|---|---:|---:|---:|---:|
| baseline | `1` | 0.00 pp | `0.5` | 1.97 pp |
| legacy | `0.3` | 0.00 pp | `0.2` | 2.70 pp |
| ours | `0.3` | 0.00 pp | `0.2` | 2.52 pp |

Baseline is the ceiling-limiting scheme. Its maximum saved conductance is
`0.737158`, versus `0.236202` for legacy and `0.248548` for ours. Consequently,
`Gmax=0.3` is an exact no-op for legacy and ours but clips the baseline.

## Shared sampled window

The conservative common sampled window inferred from the one-sided sweeps is
`[Gmin,Gmax]=[1e-4,1]`. Its direct full-test replay gives:

| Scheme | Clean accuracy | Windowed accuracy | Accuracy change |
|---|---:|---:|---:|
| baseline | 66.23% | 66.25% | +0.02 pp |
| legacy | 56.13% | 55.49% | -0.64 pp |
| ours | 68.96% | 68.91% | -0.05 pp |

All three remain within one percentage point of their own unclipped replay.
This is a checkpoint- and grid-dependent diagnostic bracket, not a continuous
hardware limit.

## Artifacts

- Full floor sweep:
  `simulation_results/conv1_hardsigmoid_finite_conductance_active_handoff_20260725/analysis/min_floor_full/`
- Full ceiling sweep:
  `simulation_results/conv1_hardsigmoid_finite_conductance_active_handoff_20260725/analysis/max_ceiling_full/`
- Full common-window replay:
  `simulation_results/conv1_hardsigmoid_finite_conductance_active_handoff_20260725/analysis/common_window_full/`
- Evaluator:
  `labs/tools/evaluate_conv1_hardsigmoid_finite_conductance.py`

The floor, ceiling, and common-window CSV SHA-256 values are respectively:

- `1ad42c644b595d8b797cd1d32def16b8bfb5f847f924fa40db1589ab12d306f3`;
- `ed7bf14f7c56fd2282ecef6387da94f77f26b49cae39e806afdefd5a73654a6b`;
- `8775df04b6df9c86eecf4bf5c9776bec011a6a5b135611300c940d7c0b4e183c`.
