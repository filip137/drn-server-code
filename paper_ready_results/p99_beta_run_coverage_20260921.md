# Existing training coverage against the later per-matrix p99 beta selections

Reviewed September 21, 2026 using existing local records and resolved training
configs. No new training, checkpoint replay, or official-test evaluation.

Scope: wide [0,100] weights, exact-zero biases, centered float64 EqProp,
unchanged Adam rates, T=K=4/6/8, and full 10/30/30-epoch budgets for Conv1/2/3.
Bounded-weight experiments and older incompatible operating points are separate.

Here p99 means every weight matrix has cosine strictly above .99 on all
36 calibration batches at initialization and 36 at the saved BPTT checkpoint,
at zero read noise. It does not mean a percentile. The selected beta is the
largest measured passing point in the existing combined grid, without a norm
gate or a further safety-factor reduction. These are seed-0 selections, not
guarantees for other seeds or the subsequent EP trajectory. The Conv3 ours
selection .987333678708 updates the original-grid choice .9; there was no
dedicated p99 refinement campaign.

## Exact selection versus threshold compliance

The original clean campaign has seeds 0/1/2 at the listed beta. Noise curves
have seed 0 only. The calibration minimum below is from the two reference
checkpoints, not from the noisy-trained final checkpoints or all three seeds.

| Architecture | Scheme | Original clean beta | Original noise-sweep beta | Later p99 beta | Original clean beta: minimum cosine | Assessment |
|---|---|---:|---:|---:|---:|---|
| Conv1 | baseline | 100 | 100 and 200 | 300 | .997616 | Both noise-sweep betas pass p99, below selected maximum; beta200 minimum .994516 |
| Conv1 | ours | 30 | 30 | 300 | .999077 | Passes p99, 10x below selected maximum |
| Conv1 | legacy | 3 | 3 | 60 | .998727 | Passes p99, 20x below selected maximum |
| Conv2 | baseline | 100 | 100 | 100 | .992029 | Exact match |
| Conv2 | ours | 10 | 10 | 10 | .991964 | Exact match |
| Conv2 | legacy | .03 | .03 | 3 | .999883 | Passes p99, 100x below selected maximum |
| Conv3 | baseline | 100 | 10 | 10 | .967393 | Original clean beta100 fails p99; newer clean/noisy beta10 runs match |
| Conv3 | ours | 3 | 3 | .987333678708 | .939168 | Original beta3 fails p99; about 3.04x above selected value |
| Conv3 | legacy | .001 | .001 | .1 | .999961 | Passes p99, 100x below selected maximum |

Thus a different beta does not automatically violate the .99 threshold. It
violates an exact-largest-measured-beta rule if that is the intended rule.
The only failures among the nine original clean seed-0 calibration settings
are Conv3 baseline100 and ours3. This is a gradient-calibration finding,
not a claim of numerical training divergence.

## Existing full-budget runs at the exact selected beta

| Architecture / scheme | Beta | Clean training seeds | Noisy training coverage | Total completed runs |
|---|---:|---|---|---:|
| Conv2 baseline | 100 | 0, 1, 2 | Seed0 at 1e-5, 3e-5, 1e-4, 3e-4, 5e-4 | 8 |
| Conv2 ours | 10 | 0, 1, 2 | Seed0 at 1e-5, 3e-5, 1e-4, 3e-4, 5e-4 | 8 |
| Conv3 baseline | 10 | 0, from the newer baseline noise study | Seed0 at 1e-5, 3e-5, 1e-4, 3e-4, 5e-4 | 6 |
| **Total** | | | | **22** |

All22 retained runs completed30 finite epochs; there is no recorded training
divergence among them. Their beta fields were checked against their local
`config.used.json` files. Six are the original Conv2 clean seed0/1/2 runs;
16 come from the noise studies, including Conv3's new clean control.
The per-run inventory links each collected bundle and records its state.

Within the reviewed current-contract studies, no matching full-budget
clean/noisy training is available at Conv1 baseline300, ours300, legacy60;
Conv2 legacy3; or Conv3 legacy.1 and ours.987333678708. Conv3 ours beta.9 has
ten clean epochs and passes p99, but is neither the latest exact beta nor a
full30-epoch run.

## Qualification and failure distinctions

- The smaller original betas were supported by measured training failures,
  not merely an arbitrary safety factor. In the August15 Conv1 Adam
  boundary-versus-one-decade study, baseline1000 became nonfinite in epoch2
  (batch164), ours300 in epoch1 (batch2137), and legacy30 in epoch3
  (batch3047). Their matched beta100/30/3 controls each completed ten finite
  epochs. These were candidates from the earlier cosine-.99 diagnostic.
  The same numerical ours beta300 is selected by the later p99 calibration,
  so its historical failure must remain visible. The historical run used
  T=K=8 and a nonzero bias learning rate; the current Conv1 contract uses
  T=K=4 and frozen zero biases. It is therefore relevant failure evidence,
  but not an exact-current-contract failure test. Legacy's historical
  equations also predate the physical-KCL correction. The mismatch table
  above is a coverage audit, not a recommendation to raise stable betas to
  calibration maxima without training qualification.
- Conv1 baseline300 passes seed-0 selection but failed the separate seed-2
  gradient confirmation (worst cosine .976462). This was not a training crash.
- Conv3 baseline10 passes seed-0 selection but failed seed-1 gradient
  confirmation (worst cosine .968773). Its completed noise sweep is seed0;
  the finite-T residual caveat also remains.
- Conv2 baseline100 passed the separate three-seed gradient confirmation.
- The later p90 Conv3 beta choices404.141105702/4.42250110273/5.26875648112
  for baseline/legacy/ours do not satisfy the later p99 selection rule.
  Legacy and ours at sigma1e-3 became nonfinite in epochs8 and18. Those are
  p90 training failures and must not be attributed to the untrained p99 choices.
- Likewise, the larger-beta Conv2 legacy30 pilot failed in epoch5. Its beta
  is ten times the p99 choice3, so it is not a training test of that choice.
- A seed-0 beta match alone does not establish complete cross-host protocol
  equivalence, noisy-gradient fidelity, or paper eligibility. All accuracies
  here remain ordinary-MNIST validation evidence.

## Sources

- [Per-run exact-beta inventory](p99_beta_matching_runs_20260921.csv)
- [Original clean-run tracker](current_contract_run_status.csv)
- [Ours/legacy noise-run tracker](read_noise_run_status_20260914.csv)
- [Baseline noise-run tracker](baseline_read_noise_run_status_20260916.csv)
- [Original-grid selections](beta_rule_comparison_20260918_selections.csv)
- [Original-grid checkpoint minima](beta_rule_comparison_20260918_case_summary.csv)
- [Combined refined measurements](beta_refinement_20260919_metrics.csv)
- [Baseline multi-seed confirmation](baseline_beta_audit_20260914.md)
- [Conv3 baseline beta10 confirmation](conv3_baseline_beta_tk8_20260916.md)
- [Ten-epoch beta training](layerwise_beta_training_20260919.md)
- [Historical boundary-versus-one-decade training failures](../results/perfectdiode-conv123-centered-float64-eqprop-beta-collapse-boundary-vs-1decade-sgd-adam-10ep-seed0-20260815-v1/analysis/cases.csv)
- [p90 sigma1e-3 failures](conv3_p90_read_noise_1em3_20260920.md)
