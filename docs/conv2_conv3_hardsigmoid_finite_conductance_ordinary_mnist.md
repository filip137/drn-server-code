# Conv2/Conv3 hard-sigmoid finite-conductance status on ordinary MNIST

Updated: 2026-07-25

Status: Conv2 complete; Conv3 blocked by checkpoint availability.

## Conv2 method

The Conv2 replay uses the three selected entries from the completed v5
ordinary-MNIST diagnostic: strict-equal arm, architecture alpha `3e-3`, and
each entry's immutable best-validation checkpoint.

No model is retrained and no optimizer step is taken. Every `(Gmin,Gmax)` cell
starts from the pristine checkpoint, clamps only `ConvWeight` and
`DenseWeight` tensors, leaves every `Bias` unchanged, and evaluates all 10,000
ordinary-MNIST test examples. The row-specific operational values are:

| Scheme | Operational `T/K` | Checkpoint SHA-256 | Clean test accuracy |
|---|---:|---|---:|
| baseline `v1/c1` | `16/6` | `f102d7f661331fe963db31b6719b61111c53fce0a1be3651bf33b2c541e294f6` | 93.99% |
| legacy `v4/c0.25` | `8/4` | `c7569bb0d6bb266be1928b87299d73b2ec4ae9ca27e993fdef6dfa0c1c132b29` | 95.02% |
| ours `v4/c1` | `24/6` | `49058cf6975776e7790dab1049344a36c5caa5f84037c38f5b0d14192ba49487` | 95.72% |

## Conv2 minimum-conductance floor

These are discrete-grid brackets. "Largest floor" is the largest tested
`Gmin` whose accuracy loss is at most one percentage point.

| Scheme | Largest tested floor within 1 pp | Accuracy loss | Next floor | Accuracy loss |
|---|---:|---:|---:|---:|
| baseline | `1e-3` | -0.11 pp | `3e-3` | 2.21 pp |
| legacy | `1e-4` | 0.44 pp | `3e-4` | 1.14 pp |
| ours | `3e-3` | 0.74 pp | `1e-2` | 16.34 pp |

Legacy is the shared-floor limiter.

## Conv2 maximum-conductance ceiling

"Smallest ceiling" is the smallest tested positive `Gmax` whose accuracy loss
is at most one percentage point.

| Scheme | Smallest tested ceiling within 1 pp | Accuracy loss | Next ceiling | Accuracy loss |
|---|---:|---:|---:|---:|
| baseline | `0.3` | 0.26 pp | `0.2` | 2.05 pp |
| legacy | `0.2` | 0.48 pp | `0.15` | 6.26 pp |
| ours | `0.3` | 0.00 pp | `0.2` | 3.52 pp |

Baseline and ours jointly set the shared ceiling.

## Conv2 shared sampled window

The conservative common sampled window is
`[Gmin,Gmax]=[1e-4,0.3]`. Direct full-test replay gives:

| Scheme | Clean accuracy | Windowed accuracy | Accuracy change |
|---|---:|---:|---:|
| baseline | 93.99% | 93.80% | -0.19 pp |
| legacy | 95.02% | 94.58% | -0.44 pp |
| ours | 95.72% | 95.84% | +0.12 pp |

All three remain within one percentage point. The small improvement for ours
is a finite-test clipping perturbation, not evidence that clipping is
generally beneficial.

The floor and joint replays ran on an RTX 3090; the full ceiling replay ran on
an RTX 3080. All 24 overlapping clean/knee accuracy and clipping rows matched
exactly across hosts. Timing is therefore a hardware-confounded diagnostic,
but the reported accuracy results are cross-host replicated.

## Conv3 blocker

A matched Conv3 three-scheme replay was not run because there is no selected
ordinary-MNIST checkpoint trio.

The canonical remote v7 tree currently contains:

- 7 of 9 baseline core candidates;
- 9 of 9 ours core candidates;
- 9 of 9 legacy core candidates;
- no completed `select_core` stage;
- no completed `finalize` stage.

All nine legacy core candidates are inadmissible. A representative low-corner
entry failed the loss-EMA safety gate at step 40 and has no best-validation
epoch. The separate safety-clean legacy rescue also failed the 90% accuracy
gate, reaching at most 76.38% after three epochs, and froze no learning rate.

Using an unfinished baseline/ours core entry together with a failed or rescue
legacy checkpoint would repeat the checkpoint-family mismatch that invalidated
the first Conv1 ordinary-MNIST replay. Conv3 finite-conductance evaluation must
wait for a valid selected baseline/legacy/ours checkpoint set or be explicitly
authorized as a mixed low-accuracy diagnostic.

## Artifacts

- Conv2 floor:
  `simulation_results/conv2_hardsigmoid_finite_conductance_v5_ordinary_mnist_20260725/analysis/min_floor_full/`
- Conv2 ceiling:
  `simulation_results/conv2_hardsigmoid_finite_conductance_v5_ordinary_mnist_20260725/analysis/max_ceiling_full/`
- Conv2 joint window:
  `simulation_results/conv2_hardsigmoid_finite_conductance_v5_ordinary_mnist_20260725/analysis/common_window_full/`
- Evaluator:
  `labs/tools/evaluate_conv1_hardsigmoid_finite_conductance.py`

The floor, ceiling, and common-window CSV SHA-256 values are respectively:

- `f7f763d0c4a8992c2889c69c775220753f854705dfe839dc1bed5260932e651c`;
- `016a82179aee4daea22dc909987b65d5dcf6f3b411e18050450503e5937f0a63`;
- `a0dca769fcce227567c497d4e4f888c2faf7a04cb6a533404bef19400659980f`.
