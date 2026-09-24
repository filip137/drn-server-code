# Conv3 refined-beta training: ten-epoch comparison

Updated 2026-09-19T17:54:04.407605+00:00. **6/6 completed, 6/6 terminal.**

Exploratory, seed0, ordinary-MNIST fixed55k/5k validation, zero read noise, T=K=8, float64 centered EqProp, fixed scheme-specific Adam rates, matched initializer and minibatch order. Official-test evaluations remain zero.
Select the largest measured beta whose cosine exceeds the listed threshold for every matrix on each of36 batches at both initialization and the saved BPTT checkpoint. The refined passing/failing beta brackets are within5%. No gradient-norm gate is imposed.

| Scheme | Threshold | Injected beta | Worst cosine | State / epochs | Final / best validation | Delta vs old beta (pp) |
|---|---:|---:|---:|---|---|---:|
| baseline | 0.95 | 147.682614594 | 0.951316 | complete / 10 | 97.14% / 97.14% | +0.12 |
| baseline | 0.90 | 404.141105702 | 0.901977 | complete / 10 | 97.16% / 97.16% | +0.14 |
| legacy | 0.95 | 2.81845428732 | 0.951908 | complete / 10 | 98.58% / 98.58% | -0.06 |
| legacy | 0.90 | 4.42250110273 | 0.902633 | complete / 10 | 98.54% / 98.58% | -0.10 |
| ours | 0.95 | 2.49274796756 | 0.950604 | complete / 10 | 98.48% / 98.48% | +0.06 |
| ours | 0.90 | 5.26875648112 | 0.900348 | complete / 10 | 98.52% / 98.52% | +0.10 |

| Scheme | Larger (.90) minus smaller (.95), final validation pp |
|---|---:|
| baseline | +0.02 |
| ours | +0.04 |
| legacy | -0.04 |

**All six settings complete ten finite epochs and pass the final-drop screen.** Larger-minus-smaller final validation is baseline +0.02pp, ours +0.04pp and legacy -0.04pp. These one-seed differences show no consistent performance penalty from the larger p90 beta; they do not establish statistical equivalence or a universal stability guarantee.
The p90 settings are admitted to the separately authorized thirty-epoch V100/A100 read-noise sweep. That follow-up does not retroactively make these ten-epoch clean pilots full-horizon evidence.
[Closeout validation and accounting](../results/eqprop-conv3-refined-beta-training-20260919-v1/closeout-validation.json)

Previous-beta controls use only the first ten epochs of the named historical30-epoch runs: baseline beta100: 97.02%; legacy beta0.001: 98.64%; ours beta3: 98.42%.

Stability requires ten finite epochs and final validation less than5pp below the run’s own best; this only screens gross collapse. Single-seed differences do not establish a reproducible performance advantage. No full30-epoch, noisy-training or multiseed qualification is implied.

Operational provenance: original2178212_0 is retained; original indices1–5 exited before training because the wrapper and runner both selected by Slurm array index. The wrapper fix explicitly selects exact-run index0 after choosing the config. Recovery canary2178310 passed, then array2178358 retried only those five cases. Failed startup attempts and smokes remain preserved and excluded. All accepted runs use Jean Zay H100 and unchanged scientific source/configs.

[Validation curves](conv3_refined_beta_training_20260919.png) · [Summary CSV](conv3_refined_beta_training_20260919.csv) · [Epoch CSV](conv3_refined_beta_training_20260919_epochs.csv) · [Plan](../docs/eqprop_conv3_refined_beta_training_plan_20260919.md) · [Raw evidence](../results/eqprop-conv3-refined-beta-training-20260919-v1/)
