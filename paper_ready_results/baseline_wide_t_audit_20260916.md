# Wide Conv3 baseline: increasing T at beta 100

Completed and analyzed September 16, 2026. **Increasing T resolves the
measured equilibrium failure but does not resolve the initialization gradient
mismatch. Beta 100 remains unqualified at every tested T.**

This is clean, read-only ordinary-MNIST diagnostic evidence. No training,
optimizer steps, endpoint read noise or official-test evaluation occurred.

## Measured result

Only free-phase T changes. K=8 and injected/base beta=100 remain fixed, with
wide [0,100] weights, exact-zero biases, the same float64 centered EqProp
and same-T/K BPTT estimators, and identical input payloads. Each setting covers
36 batches of 16 at initialization and the seed-0 BPTT best checkpoint.

| T | Worst initialization cosine | Largest initialization norm mismatch | Worst trained cosine | Worst trained free-state residual p90 | Gradient gate | Equilibrium gate |
|---|---:|---:|---:|---:|---|---|
| 8 (previous audit) | .967393 | .184260 | .999421 | .188054 | fail | fail |
| 12 | .967393 | .184260 | .999421 | .00111043 | fail | pass |
| 16 | .967393 | .184260 | .999421 | 6.71346e-6 | fail | pass |
| 24 | .967393 | .184260 | .999421 | 3.33387e-10 | fail | pass |
| 32 | .967393 | .184260 | .999421 | 2.36469e-11 | fail | pass |

Thresholds are every-layer/batch cosine >=.99, symmetric norm mismatch <=.10,
and every endpoint residual p90 <.01. The table shows the worst trained free
endpoint; the equilibrium outcome also checks initialization and all other
endpoints. All trained-checkpoint gradient comparisons pass at every T.

At each setting, exactly seven initialization comparisons fail, all in
`ConvWeight_0`, on batches 1, 6, 17, 23, 26, 31 and 33 (zero-based indices).
The worst initialization cosine changes by only about 3.2e-9 between T8 and
T32; the largest norm mismatch remains about .18426. Initialization already
passes the residual criterion at T8, and its free-state residual reaches
about 1.85e-13 at T24/32 without resolving gradient fidelity.

## Interpretation and next beta check

Insufficient free-phase relaxation explains the trained free-state residual
failure on this cohort: T12 is the smallest **tested** longer T that passes.
The sweep does not locate the exact minimum between 8 and 12 or establish
three-seed equilibrium qualification.

Increasing T alone does not explain or cure the beta-100 initialization
gradient mismatch. The next useful diagnostic is to reduce beta while holding
T12/K8 fixed, starting with candidates 10 and 30 supported or motivated by
the earlier beta studies. That is a proposed next study, not a measured
qualification or a launched beta sweep. K was not varied here. Noisy training
accuracy must not be used to choose beta.

No setting passes both gates, so no T/beta candidate was nominated and the
conditional three-seed confirmation was correctly omitted. Its fresh cohort
is prepared but no confirmation gradients were measured. The current
checkpoints were trained at T8/K8; replaying them at larger T does not provide
a matching longer-T clean training control. Future longer-T training needs
the dependent operating-point/LR checks and a clean full-training control.
Existing ours/legacy noise curves retain T8 and their recorded qualifications
and environment limitations.

## Coverage, validation and cost

All four new formal cases validate: **288 checkpoint/batch replays and 1,152
layer comparisons**, including 28 failing comparisons retained as evidence.
The separate smoke validates (one trained batch, four layer comparisons).
The T8 reference is a validated reused result and is excluded from new-work
counts. No operational attempt failed or required replacement.

Fifteen existing focused cohort/T-K tests pass. All 16 prepared configs retain
the original source checkpoint contracts; diagnostic T is explicit and K/beta
are unchanged. Selection payload hashes match the T8 reference exactly.
Canonical bundles, source-byte immutability, finite float64 replay and zero-bias
guards pass. The driver exited zero after 597.733 seconds, charged as
**.166037 physical GPU-hours** against the separate two-hour allowance.
The local RTX3090 is idle afterward; the paused clean campaign is unchanged.

![Gradient fidelity and equilibrium versus T](baseline_wide_t_audit_20260916.png)

[Case measurements](baseline_wide_t_audit_20260916.csv) ·
[Per-layer/cohort distributions](baseline_wide_t_layer_summary_20260916.csv) ·
[Machine-readable summary](baseline_wide_t_audit_20260916.json) ·
[PDF figure](baseline_wide_t_audit_20260916.pdf) ·
[Complete local evidence](../results/eqprop-baseline-wide-t-relaxation-20260916-v1/) ·
[Execution plan](../docs/eqprop_baseline_read_noise_beta_plan_20260916.md).
