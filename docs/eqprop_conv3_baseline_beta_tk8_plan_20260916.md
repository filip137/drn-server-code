# Conv3 baseline beta sweep at fixed T=8, K=8

Date: 2026-09-16. Status: **completed and analyzed; confirmation failed**.

Started13:32 CEST on local RTX3090, driver PID1230581, now exited0. The same-runner
smoke and all seven executed full cases validate. All16 prepared configs preserve
the original source case except beta; T=K=8 and gradient thresholds are exact.
Preallocation checks found local/Akib/Nom idle; Trex/Fifi/Loulou occupied,
Riri failing host-key verification and Jean Zay timing out. The complete
inventory and original frozen plan are retained in the result root.

Final outcome: beta10 passes selection and confirmation seeds0/2, but seed1
fails two initialization comparisons (cosine .968773; norm mismatch .101208).
No beta is confirmed across all three seeds. All7 new formal bundles and
the smoke validate, with504 replays/2,016 layer comparisons. Driver exited0;
**.275828/1 GPU-hours** settled. [Results](../paper_ready_results/conv3_baseline_beta_tk8_20260916.md).
The requested displacement follow-up uses beta10 only as a labeled seed-0
candidate. No training beta has been promoted or training resumed.

Filip explicitly chose to retain T=K=8 after the T-only diagnostic and asked
to sweep Conv3 beta now. This supersedes the proposed T12/K8 follow-up.
Scope is the wide [0,100], zero-bias Adam baseline, in preparation for the
baseline read-noise comparison. This stage performs clean read-only replays.

## Cases and decision

Evaluate injected/base beta **10, 30, 50, 75, 100** with **T=8, K=8 fixed**.
The existing beta100 selection bundle supplies that unchanged reference;
four new beta cases run on the same seed-0 initializer and BPTT best checkpoint,
the same 36 batches of16 and the same frozen numerical source as that reference.
The grid brackets the earlier four-batch beta10 pass and expanded beta100
failure, with intermediate candidates to avoid an unnecessarily small beta.
It is a finite tested grid, not an estimate of the exact maximum beta.

After all five selection points are available, freeze the largest candidate
passing every weight-layer/batch gradient comparison at both checkpoint roles:
cosine >=.99, symmetric norm mismatch <=.10, and finite phases/gradients.
Confirm only that candidate on existing model seeds0/1/2, at initialization
and best validation, using a reserved cohort. If it fails, record the failure
without tuning another candidate on that confirmation cohort. No automatic
grid expansion is included.

The separate equilibrium residual criterion remains p90 <.01 and is reported
without changing its threshold. The user's fixed-T8 decision retains the
known trained free-state residual caveat. A gradient-confirmed candidate must
therefore be labeled **gradient-confirmed at fixed T8/K8 with an equilibrium
caveat**, not fully equilibrium-qualified. Residual failure alone does not
prevent the requested beta selection or its gradient confirmation.

## Source and cohort controls

Reuse the September14 frozen analyzer and scientific runtime. Preserve the
source checkpoints, original initializers, preprocessing, input gain, loss,
exact-zero biases and named Adam rates. Use float64 centered frozen-current
EqProp versus same-state, same-T/K BPTT. Only beta changes in each source case.
No optimizer step, endpoint read noise or official-test access is permitted.

Selection uses the original four regression batches plus the September14
32-batch selection cohort: 576 examples, 36 batches, two checkpoint roles.
The frozen plan reused the **then-unmeasured** cohort prepared for the T-only audit:
the original64 examples plus512 fresh validation examples sampled with NumPy
PCG64 seed2026091601 after excluding all1,088 earlier audit examples. No
confirmation gradients were measured in the T-only study, which nominated
no passing setting. Its execution record and absence of confirmation outputs
were checked before reuse. This cohort has now been measured in the beta10
confirmation and must not be reused as fresh confirmation for a later search.
The fresh part remains disjoint from both earlier
audit cohorts. Validation data have previously informed checkpoint selection;
this is a reserved gradient cohort, not an untouched accuracy test set.

The four new selection cases require288 checkpoint/batch replays and1,152
layer comparisons. Three conditional confirmation cases add216 replays and
864 comparisons. The reused beta100 case is excluded from new-work accounting.
Freeze all configs, source hashes and the decision rule before measurements;
freeze the selected beta before confirmation. Preserve failed comparisons.

## Execution and outputs

- Target: local/Main RTX3090 preferred. Recheck all authorized GPU hosts and
  Jean Zay before allocation; use only the selected idle GPU and leave
  unrelated work untouched. Record the observed inventory in the result root.
- Budget: **one physical GPU-hour**, separate baseline-noise preparation
  allowance. Expected15–20 minutes from the just-completed T sweep. Includes
  the same-runner GPU smoke, failed attempts and validation overhead.
  Driver deadline3,600 seconds; per-case maximum600 seconds, smoke180 seconds.
- Run `python -m experiments.run_conv3_baseline_beta_sweep prepare`, then
  `python -m experiments.run_conv3_baseline_beta_sweep run` using py312.
  Result root: `results/eqprop-conv3-baseline-beta-tk8-20260916-v1/`.
  Configs: `configs/conv/eqprop_conv3_baseline_beta_tk8_20260916_v1/`.
- Keep monitoring through terminal coverage and validate every included local
  canonical bundle. Retain per-case logs, execution.json, frozen inputs,
  raw layer/batch metrics and residuals. Charge actual elapsed GPU allocation.
  Publish CSV/JSON, a matplotlib figure and an interpreted report in
  `paper_ready_results/`; curate the conclusion in the experimental manifest.

This sweep does not itself qualify full-training stability at a changed beta.
A new beta needs a matching clean EqProp training control before interpreting
its noise curve. Keep the same beta across all noise levels. The paused clean
216-cell campaign is not resumed by this diagnostic.

[T-only diagnostic](../paper_ready_results/baseline_wide_t_audit_20260916.md) ·
[Earlier beta audit](../paper_ready_results/baseline_beta_audit_20260914.md) ·
[Existing read-noise results](../paper_ready_results/read_noise_sweep_results.md).
