# Bounded-Weight Study

Updated: 2026-08-17

## Main Finding

The exploratory bounded-weight results give a clear picture: the legacy
amplification scheme retains a large advantage over baseline and ours when
the conductance weights are tightly bounded. Increasing the upper weight
limit improves every architecture and scheme, but it helps the deeper
baseline and ours models much more than legacy.

There are also individual comparisons in which a bounded run is better than
the closest available wide-range control. In particular, Conv1 legacy reaches
`96.68%` with either `wmax=5e-4` or `wmax=1e-3`, compared with `96.44%` in the
zero-bias ordinary-MNIST wide `[0,100]` Adam run. This probably means that the
learning-rate vector used for the corresponding wide-range legacy run was
suboptimal and that the run needs to be redone.

This is a rerun motivation rather than a controlled claim that bounding the
weights improves accuracy. The bounded and existing wide runs use different
initializers and learning-rate vectors. A decisive comparison must repeat the
wide run from the exact bounded-uniform checkpoint, data order, zero-bias
contract, optimizer, epoch budget, and learning-rate vector, changing only
the upper weight limit. Wide learning-rate selection should then be repeated
if that matched wide arm remains below the bounded result.

## Bounded Fixed-Initialization Sweep

These are best validation accuracies from the completed Adam-only, seed-0,
ordinary-MNIST study. Conv1 ran for 10 epochs and Conv2/Conv3 for 30 epochs.
Within each architecture, every scheme and ceiling starts from the same exact
checkpoint sampled once from `Uniform[1e-5,1e-4)`. Biases remain exactly zero,
the lower projection limit is `1e-5`, and the official test split was not
read.

| Architecture | `wmax` | Baseline | Ours | Legacy |
|---|---:|---:|---:|---:|
| Conv1 | `1e-4` | `89.68%` | `92.92%` | **`95.78%`** |
| Conv1 | `5e-4` | `94.44%` | `95.00%` | **`96.68%`** |
| Conv1 | `1e-3` | `94.80%` | `95.02%` | **`96.68%`** |
| Conv2 | `1e-4` | `85.60%` | `90.30%` | **`96.84%`** |
| Conv2 | `5e-4` | `92.44%` | `96.56%` | **`98.16%`** |
| Conv2 | `1e-3` | `93.52%` | `97.18%` | **`98.20%`** |
| Conv3 | `1e-4` | `77.66%` | `84.54%` | **`93.68%`** |
| Conv3 | `5e-4` | `87.44%` | `94.74%` | **`97.68%`** |
| Conv3 | `1e-3` | `89.60%` | `95.80%` | **`97.86%`** |

Legacy is best in all nine architecture-by-ceiling comparisons. At the tight
`1e-4` ceiling, legacy leads ours by `2.86/6.54/9.14 pp` for Conv1/2/3. At
`1e-3`, those leads contract to `1.66/1.02/2.06 pp`, showing that the tight
ceiling explains a substantial part, but not all, of the preliminary legacy
advantage.

Raising `wmax` from `1e-4` to `1e-3` changes best validation accuracy by:

| Architecture | Baseline | Ours | Legacy |
|---|---:|---:|---:|
| Conv1 | `+5.12 pp` | `+2.10 pp` | `+0.90 pp` |
| Conv2 | `+7.92 pp` | `+6.88 pp` | `+1.36 pp` |
| Conv3 | `+11.94 pp` | `+11.26 pp` | `+4.18 pp` |

Legacy is effectively saturated by `wmax=5e-4`: moving from `5e-4` to `1e-3`
changes its best accuracy by only `0.00/0.04/0.18 pp` for Conv1/2/3. Conv2 and
Conv3 baseline and ours still improve at the largest tested ceiling.

## Closest Available Wide-Range Control

The closest existing all-depth comparison also uses seed-0 ordinary MNIST,
zero biases, Adam, and 10/30/30 epochs, but it uses Kaiming initialization in
the wide `[0,100]` conductance interval and the earlier wide-range
learning-rate handoff. In this repository, “unbounded” commonly refers to
this wide family; the trained weights are still projected to `[0,100]`.

| Architecture | Baseline | Ours | Legacy |
|---|---:|---:|---:|
| Conv1 | `96.16%` | `96.44%` | `96.44%` |
| Conv2 | `97.36%` | `98.10%` | **`98.46%`** |
| Conv3 | `97.80%` | `98.66%` | **`99.00%`** |

Wide minus bounded `wmax=1e-3` is:

| Architecture | Baseline | Ours | Legacy |
|---|---:|---:|---:|
| Conv1 | `+1.36 pp` | `+1.42 pp` | `-0.24 pp` |
| Conv2 | `+3.84 pp` | `+0.92 pp` | `+0.26 pp` |
| Conv3 | `+8.20 pp` | `+2.86 pp` | `+1.14 pp` |

The remaining wide-range advantage is much larger for baseline than for
legacy and grows strongly with depth. The negative Conv1 legacy difference is
the clearest reason to revisit the wide learning rate, but the cross-study
deltas must not be interpreted as isolated effects of the weight ceiling.

## Historical Medium-Affine Wide Result

The earlier deterministic medium-affine paper batch also found a legacy
advantage under the wide `[0,100]` contract. Representing each scheme by its
better optimizer gives the following official-test accuracies from the
best-validation checkpoints:

| Architecture | Baseline | Ours | Legacy |
|---|---:|---:|---:|
| Conv1 | `69.60%` (SGD) | `72.60%` (SGD) | **`74.64%` (Adam)** |
| Conv2 | `85.83%` (Adam) | `92.26%` (Adam) | **`94.06%` (Adam)** |
| Conv3 | `91.54%` (Adam) | `96.38%` (Adam) | **`97.01%` (Adam)** |

This historical result confirms that legacy can also lead in the wide
contract, but it is not directly comparable with the bounded table: it uses
medium-affine MNIST, learned nonnegative unscaled biases, and official-test
accuracy rather than ordinary-MNIST zero-bias validation accuracy.

## Evidence and Limitations

All 27 bounded production bundles validate and have complete epoch and
checkpoint coverage. They preserve the intended architecture-level initial
state, MNIST cohort and order, exact-zero biases, configured bounds, and
`official_test_read=false`. The study remains single-seed exploratory evidence
and does not define a new learning-rate handoff. Its validation accuracies are
not paper-facing. Under the 2026-08-17 ordinary-MNIST paper-dataset decision,
the checkpoints are candidates for reuse only after the final dynamic-range
contract and inclusion set are frozen and the explicit paper reuse gate
passes; any eligible checkpoint may then receive one sealed official-test
evaluation.

- [Bounded analysis report](../results/perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1/analysis/report.md)
- [Bounded full table](../results/perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1/analysis/summary.csv)
- [Bounded accuracy plot](../results/perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1/analysis/accuracy_vs_weight_max.png)
- [Ordinary-MNIST zero-bias wide table](../results/perfectdiode-conv123-zero-bias-ordinary-mnist-seed0-20260805-v1/analysis/summary.csv)
- [Historical medium-affine wide table](../results/perfectdiode-paper-gradient-trace-limited-20260731-v1/analysis/summary.csv)
