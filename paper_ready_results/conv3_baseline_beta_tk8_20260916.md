# Conv3 baseline beta sweep at T=8, K=8

Completed September16, 2026. **Beta10 passes expanded seed-0 selection and
confirmation on seeds0/2, but fails seed1. No beta in this tested grid is
confirmed across all three seeds.** T and K stayed at8 as requested.

## Selection and confirmation

Each case uses36 batches of16 at the verified initializer and BPTT best
checkpoint, wide [0,100] weights, exact-zero biases and float64 centered
frozen-current EqProp against same-state BPTT. Every layer/batch must have
cosine >=.99 and symmetric norm mismatch <=.10.

| Injected/base beta | Worst initial cosine | Largest initial norm mismatch | Failed layer/batches | Gradient selection |
|---|---:|---:|---:|---|
| 10 | .994886 | .057006 | 0/288 | pass |
| 30 | .980206 | .086235 | 1/288 | fail |
| 50 | .978235 | .097693 | 2/288 | fail |
| 75 | .977008 | .160800 | 5/288 | fail |
| 100 (reused) | .967393 | .184260 | 7/288 | fail |

All selection failures are at initialization; all trained-checkpoint gradient
comparisons pass. Beta100 is the unchanged September14 reference, not a new run.
Beta10 was frozen before confirmation as the largest passing tested value.

Confirmation uses the original four regression batches plus32 fresh batches,
whose512 examples are disjoint from both earlier audit cohorts. This cohort
was prepared but never measured in the preceding T-only study.

| Seed | Worst initial cosine | Largest initial norm mismatch | Worst trained cosine | Failed layer/batches | Gradient confirmation |
|---|---:|---:|---:|---:|---|
| 0 | .991247 | .080920 | .999887 | 0/288 | pass |
| 1 | .968773 | .101208 | .998126 | 2/288 | fail |
| 2 | .992655 | .072714 | .999711 | 0/288 | pass |

Seed1 fails two initialization `ConvWeight_0` comparisons on new confirmation
batches: batch16 has norm mismatch .101208 (cosine .998992), and batch29 has
cosine .968773 (norm mismatch .014157). Batch indices are zero-based. These
are retained scientific failures, not missing or failed execution bundles.

## Interpretation and limits

Reducing beta substantially improves seed-0 gradient agreement at fixed T/K.
Beta10 is a passing **seed-0 candidate**, but the three-seed result prevents
calling it a confirmed common beta. No fallback candidate was evaluated on
the now-observed confirmation cohort, and no training beta was changed.
Values below10 and between the tested grid points were not measured in this
study; there is no inferred exact threshold. A further selection round would
need its own declared cases and an unused confirmation cohort.

The separate T8 equilibrium caveat persists at every beta. The seed-0 free
residuals are numerically identical across the grid, with worst trained
free-state p90 .188054. Confirmation trained free-state maxima are .195286,
.259774 and1.858277 for seeds0/1/2. These are reported as residual failures;
the .01 threshold was not relaxed. Filip explicitly retained T=K=8, so this
study selects gradient fidelity separately and does not claim equilibrium
qualification.

The follow-up displacement study uses beta10 for the Conv3 baseline as a
seed-0 passing candidate, with the failed three-seed confirmation explicitly
retained. It is a diagnostic comparison, not promotion to training. Any new
beta still needs a matching clean training control before a noise curve.

## Coverage and artifacts

All four new selection and three confirmation bundles validate: **504 new
checkpoint/batch replays and2,016 layer comparisons**, with10 retained failing
comparisons. The separate smoke validates; beta100 reuse is excluded from
new-work counts. Selection input hashes match exactly across beta; clean
source-byte, zero-bias and float64 guards pass. No optimizer steps or
official-test reads occurred. Driver1230581 exited0 after992.979 seconds,
charging **.275828/1 physical GPU-hours**. The local GPU was released before
the separately authorized displacement study.

![Selection and confirmation](conv3_baseline_beta_tk8_20260916.png)

[Case table](conv3_baseline_beta_tk8_20260916.csv) ·
[Layer/cohort distributions](conv3_baseline_beta_tk8_layers_20260916.csv) ·
[JSON summary](conv3_baseline_beta_tk8_20260916.json) ·
[PDF figure](conv3_baseline_beta_tk8_20260916.pdf) ·
[Full evidence](../results/eqprop-conv3-baseline-beta-tk8-20260916-v1/) ·
[Execution plan](../docs/eqprop_conv3_baseline_beta_tk8_plan_20260916.md).
