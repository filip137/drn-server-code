**Paper experimental review — September 14, 2026, evening**

The clean Adam comparison has **202/216 collected, validated training runs**,
with **14 full trainings remaining and 0/216 official-test evaluations**.
The results support an amplification benefit, especially with tight conductance
bounds, and close empirical BPTT–EqProp agreement. They do not yet support a
fully qualified equilibrium-gradient claim for every wide EqProp condition.

This review follows the September three-table plan and its recorded revisions:
Adam, Conv1/2/3, three amplification schemes, seeds 0/1/2, a wide reference,
and three bounded ceilings. The older SGD and global bounded-initializer
programs are not additional requirements of this campaign. See the
[launch plan](../papers/amplification_overleaf/experiment_launch_plan_20260911.md)
and [current run ledger](../paper_ready_results/current_contract_run_status.csv).

| Clean comparison | Collected / required | Remaining |
|---|---:|---:|
| Wide BPTT | 27 / 27 | 0 |
| Wide EqProp | 27 / 27 | 0 |
| Bounded BPTT | 78 / 81 | 3 |
| Bounded EqProp | 70 / 81 | 11 |
| Total | 202 / 216 | 14 |

During this review, Loulou's previously uncollected Conv3 baseline EqProp
pilot at Gmax=1e-4 was recovered. It finished at 18:29 CEST with 77.68% best
and final validation accuracy, matching its BPTT control. All 30 epochs,
checkpoint finiteness/float64 precision, zero biases, bounds, initializer,
PT/NPZ equality, complete split/order matching, remote/local checksums, and
worker/supervisor exit-zero receipts pass. Its 6.6025 GPU-hours are settled.
Conv3 baseline pilot coverage is now **1/3**, not the afternoon snapshot's
0/3. No clean follow-up was launched. See the
[pilot report](../paper_ready_results/baseline_tk_revision/pilot_stability.md)
and [pair check](../paper_ready_results/baseline_tk_revision/conv3_seed0_tight_pair_check_20260914.json).

**What the completed measurements support.** Both amplified schemes have
higher mean best-validation accuracy than baseline in every complete
three-seed comparison. Legacy has the highest mean throughout those complete
rows. Selected BPTT examples are below; these are means ± sample standard
deviations on the same 5,000-example selection split.

| Condition | Baseline | Ours | Legacy |
|---|---:|---:|---:|
| Conv1, wide | 96.267 ± .110% | 96.493 ± .076% | 96.540 ± .053% |
| Conv2, wide | 97.340 ± .020% | 98.187 ± .076% | 98.300 ± .040% |
| Conv3, wide | 97.767 ± .031% | 98.560 ± .100% | 98.893 ± .061% |
| Conv2, Gmax=1e-4 | 85.393 ± .142% | 90.193 ± .168% | 95.633 ± .205% |

At the tight Conv2 ceiling, ours gains 4.80 percentage points and legacy
10.24 points over baseline. These gains are conditional on the frozen
scheme-specific learning rates. The bounded experiment holds its initializer
fixed across ceilings, but the wide-versus-bounded comparison changes the
initialization family as well as the range.

Across **97 completed same-seed BPTT–EqProp pairs**, the largest absolute
best-validation difference is .66 percentage points. Across the **31 complete
three-seed condition pairs**, the largest absolute mean difference is .193
points. This supports close agreement under these controls, not statistical
equivalence. The comparison also retains BPTT float32 versus EqProp float64.
Bounded baseline uses T/K=12/6 for Conv2 and 24/8 for Conv3; the other schemes
retain 6/6 and 8/8. Report this additional relaxation and its computational
cost. See the [complete validation tables](../paper_ready_results/current_contract_validation_tables.md).

**The beta audit adds an important qualification.** The completed larger-cohort
baseline audit includes 9 selection and 6 conditionally required confirmation
cases, with 1,080 checkpoint/batch replays. Its conclusions are:

| Wide baseline | Result | Consequence |
|---|---|---|
| Conv1 | Beta 300 passes seed-0 selection but fails seed-2 confirmation; worst cosine .976462 | No confirmed increase to 300; do not replace the current trained beta 100 with it |
| Conv2 | Beta 100 passes selection and confirmation on all three seeds; 200/300 fail selection | The existing beta 100 gains stronger sampled qualification |
| Conv3 | None of 100/200/300 passes; beta 100 has worst cosine .967393 and maximum norm difference .184260, plus residual failures | Wide baseline remains unqualified for a full equilibrium-gradient claim |

The earlier Conv3-ours beta-3 initialization failures also remain: the new
audit covered baseline only. More samples or good accuracy do not erase a
failed all-layer gradient gate. A lower beta can address finite-nudge bias,
but cannot by itself resolve a beta-independent free-state residual failure.
The successful bounded T/K revision does not qualify the wide model.

For a stronger equilibrium claim, specify and test a revised wide operating
point/beta contract and its independent confirmation, then repeat affected
training qualification. Changing beta affects EqProp eligibility; changing
T/K affects both members of a pair. Otherwise retain the existing runs as
explicitly finite-step empirical results with the recorded exceptions.
Do not tune another candidate on the failed confirmation cohort. See the
[completed beta audit](../paper_ready_results/baseline_beta_audit_20260914.md)
and [matching protocol](conv_paper_one_seed_bptt_eqprop_protocol.md).

**The remaining clean trainings are exactly these 14 cases.** Every case uses
30 epochs and the currently frozen rates.

| Architecture / algorithm / scheme | Missing cases | Runs |
|---|---|---:|
| Conv2 EqProp baseline, T/K=12/6 | Gmax=5e-4: seed 2; Gmax=1e-3: seeds 1,2 | 3 |
| Conv3 BPTT baseline, T/K=24/8 | Seed 2 at Gmax=1e-4, 5e-4, 1e-3 | 3 |
| Conv3 EqProp baseline, T/K=24/8 | Seed 0 at Gmax=5e-4, 1e-3; then seeds 1,2 at all three ceilings | 8 |

Both remaining Conv3 seed-0 pilots must pass the full stability rule at the
common beta .1 before the six repetitions are released. The first pilot is
stable, but does not establish the other ceilings. The
[remaining-run list](../paper_ready_results/current_contract_remaining_runs.md)
now reflects this collection. Clean admissions remain paused. The recorded
full-campaign forecast is about 335.6 GPU-hours, above the unchanged 300-hour
cap; after this collection and the beta audit, 270.300605 hours are committed,
including one outstanding check-hour reservation. The forecast needs updating
before a completion wave; the proposed budget extension remains unapproved.

**Noise robustness still needs its current-contract curve.** There is already
corrected single-seed evidence of substantial legacy sensitivity: at
sigma=5e-4, corrected Conv3 legacy final validation changes from 98.78% clean
to 74.94% noisy, a 23.84-point penalty. It nevertheless completes with finite
metrics and a 3.70-point drop from its own best. This is a severe accuracy
loss, not a divergent run. Those source results are in the
[physical-KCL correction inventory](legacy_physical_kcl_rerun_inventory.md).

The authorized overnight study declares **30 noisy cells**: Conv1/2/3 ×
ours/legacy × sigma {1e-5,3e-5,1e-4,3e-4,5e-4}, with one model/noise seed.
Complete and validate that coverage, or label the deadline-limited subset
explicitly. Audit the reused clean controls, initializers, minibatch order,
and matched random draws. Baseline and bounded noise curves are outside that
study. The inherited Conv3-ours gradient exception stays explicit, and beta
is held constant across sigma. These curves can establish descriptive
robustness under the chosen scheme/beta contracts; multiple noise/model seeds
would be needed for uncertainty claims. Sigma is in simulator units, not a
calibrated hardware specification. See the
[overnight plan](eqprop_read_noise_overnight_plan_20260914.md).

**The manuscript must be reconciled with the corrected evidence.** The current
[experimental section](../papers/amplification_overleaf/bidir_paper_theory_revised.tex)
still says all results use seed 0. Its accuracy tables and legacy columns of
the beta/displacement/weight-distribution figures refer to pre-correction
sources. In particular, its Conv3 legacy noise penalty is still 11.38 points,
whereas the corrected repeat gives 23.84. Replace the tables with the current
three-seed results, recompute derived quantities from eligible corrected
checkpoints, and rerun corrected legacy gradient/displacement measurements if
those figures are retained as physical-KCL evidence. Alternatively identify
the old-model figures as historical or omit them. Existing corrected training
need not be repeated merely to update plots. The introduction's claim that
noise prevents successful training is stronger than the finite-but-degraded
training outcomes shown here.

**Before paper accuracy is reported**, finish the inclusion/reuse and pair
audit, freeze the scientific exceptions or revised contracts, and seal each
validation-selected checkpoint and its hash. Then perform one official
10,000-example MNIST test evaluation per eligible run: **up to 216 evaluation
passes, not 216 new trainings**. Preserve all predeclared outcomes and produce
test mean ± sample SD over seeds 0/1/2. The current validation tables are
selection evidence throughout this process.

Write-noise deployment tests, renewed broad LR or bounded-initializer searches,
SGD expansion, extra datasets, and extra seeds are optional extensions to the
current three-table scope. If the paper claims write-noise tolerance or a
causal mechanism independent of the selected optimizer/beta contracts, those
claims need their own matched tests; the present evidence does not establish
them.

Review validation: all 201 previously collected canonical bundles and indexed
artifact hashes were rechecked; the additional Loulou bundle passed canonical,
semantic, and remote/local collection checks. Accuracy aggregation, exact
paired LR/parameter-order/T/K equality, and the recorded splits and complete
epoch orders were independently checked. This review
does not substitute for the final paper reuse seal. See the
[verification record](../paper_ready_results/paper_review_verification_20260914.json).
