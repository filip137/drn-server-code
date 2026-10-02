# Relative endpoint noise during training

Studies exp014–019 completed September 28; exp022 seed replication completed October 1. See the
[exp019 eta=5e-4 extension](results/exp-019-relative-noise-eta5em4.md).
This series tests whether the initialization alignment
advantages in exp013 transfer to MNIST validation accuracy. Noise is nodewise
Gaussian with standard deviation `eta * abs(clean endpoint voltage)`, with the
same eta at every noninput layer. Scheme-specific beta values are matched to
output D/F at seed-0 initialization and then frozen. Validation is noiseless;
the training studies did not evaluate the official test split. Subsequent official
test evaluations are recorded below.

| Study | Completed coverage | Results |
|---|---|---|
| exp014: initial training sweep | 84 ours/legacy runs, 10 epochs; Conv1 D/F=1, Conv2 D/F=1/2/4, Conv3 D/F=4/6; seven noise levels | [Ten-epoch results](results/exp-014-relative-noise-training.md) |
| exp015: selected continuations | 20 ours/legacy runs to cumulative epoch 30; Conv2 D/F=4, Conv3 D/F=6; five noise levels | [Epoch-30 results](results/exp-015-epoch30-continuation.md) |
| exp016: clean controls | 18 runs: all three schemes, Conv2/3, seeds 0/1/2; 30 epochs | [Clean means and seed comparisons](results/exp-016-noiseless-three-seed.md) |
| exp017: noisy baseline | 10 baseline runs, Conv2/3 at the selected D/F values; 30 epochs | [Baseline noise results](results/exp-017-baseline-relative-noise.md) |
| exp018: Conv1 controls | Nine clean runs across three schemes/seeds, plus five noisy baseline runs; 10 epochs | [Conv1 clean and noisy results](results/exp-018-conv1-controls.md) |
| exp022: two additional seeds | 72 uninterrupted seeds 1/2 runs, all schemes, four etas, Conv1/2/3 at 10/30/30 epochs | [Validated seed comparisons](results/exp-022-relative-noise-seeds.md) |
| exp019: eta=5e-4 extension | Nine uninterrupted seed-0 runs, all three schemes; Conv1 10 epochs, Conv2/3 30 epochs | [Additional noise level](results/exp-019-relative-noise-eta5em4.md) |

[Best and final accuracy: full tables and figures](results/figures/best-final-noise-summary/README.md)
displays all 155 recorded outcomes (128 noisy, 27 clean), retaining earlier
displacement settings and clean seeds. The [main JPG](results/figures/best-final-noise-summary/best_and_final_vs_noise.jpg)
uses seed 0, D/F=1/4/6 and budgets 10/30/30 for Conv1/2/3. Best accuracy spans
the full declared history, including both parts of exp015 continuations; symbols
identify their epoch-10 Adam/RNG reset. This September 29 display reuses saved
metrics and adds no training or replay. Continuations and their parents are not
independent replicates.

The five noise levels shared by the latest comparisons are eta=1e-6, 1e-5,
1e-4, 3e-4 and 1e-3. Historical exp014–019 noisy cases use seed 0. Clean seed-1/2 runs reuse the
seed-0 beta rather than rematching displacement.

Interpret the regimes separately: exp015 restarts Adam and data/noise streams
after epoch 10, whereas exp016–018 train uninterrupted. Fixed-epoch validation
accuracy, clean three-seed variability and initialization gradient alignment
answer different questions. Result notes retain the numerical outcomes,
validation evidence, source paths, exceptions and scoped interpretations.

The completed [exp022 seed replication](experiments/exp-022-relative-noise-seeds.md)
validated72 batch16 cases: seeds1/2, all three schemes and Conv1/2/3, at eta1e-5,
1e-4,3e-4 and5e-4. New runs use uninterrupted Adam; historical reset seed0
comparators remain separate. Its [result note](results/exp-022-relative-noise-seeds.md) records the reviewed comparisons; ours beats baseline throughout, but its advantage over legacy is inconsistent.

[Completed exp023 official-test evaluation](results/exp-023-official-test.md) covers153
best-validation-selected checkpoints across the main D/F=1/4/6 read-noise
comparison. It evaluates the official10000-image test split once per model,
with clean inference and validation-only batch sizing. Training/reset histories
remain separate; all153 outcomes passed collection, validation and review.

[Completed exp024 final-checkpoint correction](results/exp-024-final-test.md)
evaluates final epoch10/30/30 for Conv1/2/3, matching the fixed-final validation
comparison:127 new evaluations and26 audited identical-model reuses, all validated.
This is the requested primary endpoint; exp023 remains a separate best-model
comparison. Prior test exposure is explicitly recorded. Ours exceeds baseline in
all12 noisy means, while legacy exceeds ours in10/12; the two reverse gaps are tiny.

The [requested three-seed view](../../../../results/read-noise-final-test-20261001-v1/analysis/three_seed_report.md)
includes seeds0/1/2, with reset-history labels. Its observed variability does not
consistently favor either ours or legacy; keep it distinct from the uninterrupted
two-seed comparison above.
