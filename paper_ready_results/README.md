# Paper-ready results collection

Updated: 2026-09-16. **202 / 216 results satisfy the revised training contract;
14 full training completions remain. Official-test results: 0 / 216.**

All **198 original collected bundles** remain intact. The matched BPTT/EqProp
revision requires eighteen native-T/K baseline BPTT replacements; fifteen are
collected and three remain. The [revision manifest](baseline_tk_revision/README.md)
tracks its 36 new trainings separately. The [remaining-training list](current_contract_remaining_runs.md)
shows the missing seeds and weight limits under the current contract;
the continuing manifest records live jobs and scientific gates.

The [61 numerical T/K replays](baseline_tk_beta_qualification.md) support common
candidate beta .1 at Conv2 T=12/K=6 and Conv3 T=24/K=8. Qualification on new
BPTT best checkpoints passes for both architectures. Conv2's three full
EqProp pilots pass; Conv3 still needs complete full-pilot coverage.

This folder collects the available evidence for the clean Adam
[three-table launch plan](provenance/references/papers/amplification_overleaf/experiment_launch_plan_20260911.md).
The current accuracy values are **validation** measurements. Final
paper accuracy requires the remaining training, matching/reuse audit, contract
and checkpoint seal, and one official-test evaluation per eligible run.

- [Continuing experimental manifest](../docs/paper_ready_results_manifest.md):
  completed evidence, exact remaining coverage, dependencies, and completion gates.
- [Baseline read-noise training tracker](baseline_read_noise_run_status_20260916.md):
  completed22-run seed0 extension: Conv1 beta100/200, Conv2 beta100, Conv3 beta10;
  20 noisy cases and two new clean controls at T=K=4/6/8. All22 bundles pass
  full revalidation; Fifi used two concurrent Conv3 workers as instructed.
  [Launch plan](../docs/eqprop_baseline_read_noise_launch_plan_20260916.md) and
  [per-run CSV](baseline_read_noise_run_status_20260916.csv).
  [Completed results](baseline_read_noise_results_20260916.md): both Conv1 curves
  are complete; beta 100/200 show at most 0.04 percentage points of final
  clean-relative loss over the tested noise grid, seed 0 only.
  Conv2 beta 100 finishes at 97.24–97.38%, versus its 97.22% historical clean
  reference. Conv3 beta10 finishes at97.48/97.04/96.64/96.48/96.04% across
  increasing noise, versus97.64% clean. Final completion00:39:15 CEST
  September18;43.294845/60 GPU-hours used. Single-seed validation only.
- [Residual drift versus EP–BPTT gradient agreement](eqprop_drift_gradient_report_20260916.md):
  definitions, direct gradient evidence and the decision to retain T=K=4/6/8
  for Conv1/2/3 across all amplification schemes. The 1% drift ratio is
  descriptive; known gradient, residual and beta-confirmation caveats remain.
  [Complete checkpoint/T comparison table](eqprop_drift_gradient_comparison_20260916.csv).
- [Evening paper experimental review](../docs/paper_experimental_review_20260914.md):
  202/216 after collecting Loulou's stable first Conv3 baseline pilot; 14
  trainings remain, with beta qualification, noise evidence, manuscript
  corrections, and final official-test requirements assessed separately.
- [Original seed-0 validation results](validation_results.md) and
  [new training results](training_completion_results.md).
- [Results and agreement with earlier runs](results_agreement_20260914.md):
  paired earlier/new results, algorithm comparisons and a standalone figure.
- [Compute forecast for the revised baseline](baseline_tk_revision/compute_projection.md):
  approximately 335.6 GPU-hours under current placement. The optional faster
  plan and 375-hour cap remain unapproved; new admissions are paused for review.
- [Beta checks on newly trained references](baseline_tk_revision/beta_qualification.md).
- [Conv3 baseline beta sweep at fixed T8/K8](../docs/eqprop_conv3_baseline_beta_tk8_plan_20260916.md):
  [completed results](conv3_baseline_beta_tk8_20260916.md). Beta10 passes
  seed0 selection and confirmation seeds0/2 but fails seed1; no tested beta
  is confirmed across all three seeds. T=K=8 and its equilibrium caveat remain.
- [T-only follow-up for affected displacement cases](phase_t_sweep_20260916.md):
  all15 configurations complete. Direct drift falls below1% of nudging response
  on every batch at T12/24 for Conv3 baseline initialization/best, T16/12 for
  Conv3 legacy initialization/best, and T10 for Conv2 legacy initialization.
  Controlled-response RMS changes by less than.001%; beta10 remains unconfirmed
  and no training setting is changed. Cost.460455 GPU-hours including recovery.
- [Phase displacement across all nine architecture/scheme cases](phase_displacement_20260916.md):
  complete and validated at initialization and BPTT best checkpoints, with
  both nudge signs, post-T and matched zero-nudge references, and voltage scale.
  Conv3 baseline H1 raw displacement is 224.5 times the matched-zero response;
  continued relaxation dominates that raw difference. Beta10 remains unconfirmed.
- [Wide Conv3 baseline T-only audit](baseline_wide_t_audit_20260916.md):
  increasing T to12/16/24/32 at beta100/K8 fixes equilibrium but leaves the
  initialization gradient mismatch unchanged. No setting passes both gates;
  the [completed smaller-beta sweep](conv3_baseline_beta_tk8_20260916.md)
  still finds no three-seed-confirmed candidate.
- [Baseline beta audit](../docs/eqprop_baseline_beta_audit_20260914.md):
  completed wide Conv1/2/3 baseline search at 100/200/300 with larger disjoint
  selection/confirmation cohorts. [Results and figures](baseline_beta_audit_20260914.md):
  Conv2 beta100 confirmed; Conv1 beta300 fails seed2 confirmation; Conv3 has
  no passing candidate in this grid. Training admissions remain paused.
- [Overnight read-noise execution plan](../docs/eqprop_read_noise_overnight_plan_20260914.md):
  all 12 admitted runs on Trex, Fifi, Loulou and Nom completed by September 15
  05:52:58 Paris time, before the 08:00 deadline, and are collected and validated.
  That window ended at 12/30 complete and 18 held; 22.142636 GPU-hours were used.
- [Read-noise continuation on local, Akib, and Nom](../docs/eqprop_read_noise_continuation_plan_20260915.md):
  all 18 continuation runs finished by **September 16, 06:20:46 CEST**,
  before the 08:00 deadline. Trex/Fifi completed the two transferred cases.
  All **30/30** sweep results are collected and validated; all GPUs are
  released. The continuation used 55.089836/78 GPU-hours, and the entire
  sweep used 77.232472 GPU-hours, including checks and recovery.
- [Completed read-noise tracker](read_noise_run_status_20260914.md)
  ([per-run CSV](read_noise_run_status_20260914.csv)): separate from the clean 216-cell grid.
- [Current read-noise measurements and interpretation](read_noise_sweep_results.md):
  all Conv1/2/3 results, the [final figure](read_noise_validation.png),
  and the qualification, single-seed and environment limits. These are
  validation measurements; no official-test evaluation was performed.
- [Overnight read-noise measurements and interpretation](read_noise_overnight_results_20260915.md),
  with the remaining coverage and qualification/noise-stream limits.
- [Proposed single-seed read-noise sweep](../docs/eqprop_read_noise_sweep_proposal_20260914.md):
  ours/legacy first, suggested sigma grid, clean controls and reuse conditions;
  planning only, with baseline deferred.
- [Current 216-cell contract](current_contract_run_status.csv) and
  [remaining training in Markdown](current_contract_remaining_runs.md)
  ([per-run CSV](current_contract_remaining_runs.csv)).
- [Original inventory](run_status.csv), [original unfinished cells](remaining_runs.csv),
  and [original collected results](collected_results.csv), retained for provenance.
- [Current three-seed validation tables](current_contract_validation_tables.md) and
  [best/final aggregates](current_contract_validation_table_summary.csv): mean and sample
  standard deviation for complete groups, with unfinished seed counts shown.
- [Original validation tables](validation_tables.md), including the retained
  native-T/K baseline BPTT evidence.
- [Active launch plan](../docs/paper_training_completion_launch_plan_20260911.md):
  300 GPU-hour campaign, EP pilot gates, and weekday/weekend GPU limits.
- [Collection summary](collection_summary.json),
  [original 38-run integrity record](provenance/collection_validation.json),
  and [new-run integrity record](provenance/training_completion_collection.json).
- Completed Conv1 comparisons: [wide BPTT/EqProp](wide_conv1_validation.md) and
  [bounded BPTT/EqProp](bounded_conv1_three_seed_validation.md), each with all
  three seeds and matched initialization/cohort/order checks.
- Completed wide [Conv2](wide_conv2_validation.md) and [Conv3](wide_conv3_validation.md)
  comparisons, including all three seeds and matched cohort/order audits.
- [Complete bounded Conv2 ours three-seed comparison](bounded_conv2_ours_three_seed_validation.md).
- Completed bounded [Conv2 legacy](bounded_conv2_legacy_three_seed_validation.md):
  all three seeds and ceilings, with paired initializer/cohort/order checks.
- [Bounded EqProp beta qualification](bounded_beta_qualification.md): selected
  betas, held conditions, and links to all collected numerical checks.

```text
paper_ready_results/
  bundles/
    table1_wide_bptt/<architecture>/<scheme>/seed<seed>/
    table2_wide_ep/<architecture>/<scheme>/seed<seed>/
    table3_bounded_bptt/<architecture>/<scheme>/gmax_<ceiling>/seed<seed>/
    table3_bounded_ep/<architecture>/<scheme>/gmax_<ceiling>/seed<seed>/
  assets/<initializer>/<architecture>/seed<seed>/final_model.pt
  provenance/
    inventory/          # checked local and remote result locations
    source_configs/    # exact parent configs
    source_studies/    # source analyses, logs, and supporting records
    references/        # original plan, matrix, and governing documents
```

Each copied run retains its original `manifest.json`, `status.json`,
`metrics.jsonl`, `result.json`, resolved configs, best/final checkpoints, and
all other files present in that source run. The original 1,578 regular files and 550
internal symlinks match their source bytes/targets, and all 38 canonical
validators pass. That original payload was about 2.24 GiB. New bundles are
individually checked and recorded in
[completion collection provenance](provenance/training_completion_collection.json).
Full bundles and provenance are ignored by Git;
these small Markdown/CSV/JSON summaries remain visible to Git.

Validate a copied run from the repository root with:

```bash
python -m experiments.reporting validate-run \
  paper_ready_results/bundles/table1_wide_bptt/conv1/legacy/seed0
```

CSV `collected_result` paths are relative to this folder. Original source and
parent-config paths are repository-relative. The original manifests keep their
historical source/remote paths; the collected initializer mapping is in
[`provenance/initializer_assets.json`](provenance/initializer_assets.json).

The remote discovery limits, including Riri's failed SSH host-key check, are
recorded in the experimental manifest. There were no additional matching
results in the audited roots on local/main, Akib, Trex, or Jean Zay.

The [bounded EqProp pilot report](bounded_ep_pilot_validation.md) includes the completed Conv1 seed-0 validation curves and their matched BPTT references.

Completed launcher logs and the frozen source archives are retained under `provenance/completion_launches/` and `provenance/completion_sources/`, with hashes in `provenance/completion_execution_files.json`.
