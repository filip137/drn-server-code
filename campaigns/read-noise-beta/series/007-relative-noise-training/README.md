# Relative endpoint noise during training

Completed September 28. This series tests whether the initialization alignment
advantages in exp013 transfer to MNIST validation accuracy. Noise is nodewise
Gaussian with standard deviation `eta * abs(clean endpoint voltage)`, with the
same eta at every noninput layer. Scheme-specific beta values are matched to
output D/F at seed-0 initialization and then frozen. Validation is noiseless;
the official test split is not evaluated.

| Study | Completed coverage | Results |
|---|---|---|
| exp014: initial training sweep | 84 ours/legacy runs, 10 epochs; Conv1 D/F=1, Conv2 D/F=1/2/4, Conv3 D/F=4/6; seven noise levels | [Ten-epoch results](results/exp-014-relative-noise-training.md) |
| exp015: selected continuations | 20 ours/legacy runs to cumulative epoch 30; Conv2 D/F=4, Conv3 D/F=6; five noise levels | [Epoch-30 results](results/exp-015-epoch30-continuation.md) |
| exp016: clean controls | 18 runs: all three schemes, Conv2/3, seeds 0/1/2; 30 epochs | [Clean means and seed comparisons](results/exp-016-noiseless-three-seed.md) |
| exp017: noisy baseline | 10 baseline runs, Conv2/3 at the selected D/F values; 30 epochs | [Baseline noise results](results/exp-017-baseline-relative-noise.md) |
| exp018: Conv1 controls | Nine clean runs across three schemes/seeds, plus five noisy baseline runs; 10 epochs | [Conv1 clean and noisy results](results/exp-018-conv1-controls.md) |

The five noise levels shared by the latest comparisons are eta=1e-6, 1e-5,
1e-4, 3e-4 and 1e-3. Noisy cases use seed 0. Clean seed-1/2 runs reuse the
seed-0 beta rather than rematching displacement.

Interpret the regimes separately: exp015 restarts Adam and data/noise streams
after epoch 10, whereas exp016–018 train uninterrupted. Fixed-epoch validation
accuracy, clean three-seed variability and initialization gradient alignment
answer different questions. Result notes retain the numerical outcomes,
validation evidence, source paths, exceptions and scoped interpretations.
