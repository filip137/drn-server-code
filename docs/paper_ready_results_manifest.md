# Paper-ready results experimental manifest

Updated: 2026-09-21. Clean-paper study state: **partial**.

**202 of 216 results satisfy the revised training contract; 14 full training
completions remain. All 198 original collected bundles are preserved.
No official-test results are available yet.** The collection is
in [`paper_ready_results/`](../paper_ready_results/). Its name identifies the
destination for completing the paper; the current measurements retain their
original validation evidence classes.

This is the continuing completion ledger for the supplied
[three-table launch plan](../papers/amplification_overleaf/experiment_launch_plan_20260911.md).
It records collected evidence, unfinished training, and the gates between
training completion and paper-ready accuracy. The broader
[`experimental_manifest.md`](experimental_manifest.md) remains the historical
ledger of analyzed studies. The original collection did not launch training;
the clean completion campaign below remains paused. The official MNIST
test split remains unread.

## Read-noise robustness: beta selection first (2026-09-23)

First select and freeze a defensible beta rule and its scheme-specific values;
then finalize the paper's read-noise robustness comparison. Existing noisy
runs inform this choice but do not settle the final comparison. Reassess
their eligibility and matched clean controls once beta is selected. The
[current objective](current_state.md#current-focus-finding-the-right-beta)
includes considering a common RMS output displacement at initialization.

## Provisional beta-selection revision (2026-09-21)

Filip's proposed objective is the largest tested beta that avoids training
divergence, using Conv1 approximately p99/10, Conv2 approximately p99, and
Conv3 approximately p95 as provisional reference points. See the
[recorded protocol](eqprop_beta_stability_protocol_20260921.md) and
[conditional rerun audit](../paper_ready_results/beta_stability_protocol_run_audit_20260921.md).
The audit finds 32 full-budget completions needed if historical Conv1 values
are retained, or 48 if latest-p99/10 is applied literally. Each includes
three Conv3 clean controls already covered by the separate p95 plan.

The maximum-stability claim is not yet established. The Conv1 numerical
reference and clean-only versus noise-inclusive qualification remain under
review. These conditional replacement counts do not change the 202/216
paused-contract tally above; no new runs were launched by this review.

## Conv3 p90 sigma1e-3 extension (2026-09-21)

Three new RTX5090 cases are terminal and collected: baseline completed30 epochs
at97.16% validation (clean97.72%,drop0.56pp); legacy became non-finite in
epoch8 and ours in epoch18. Their last complete validation values94.76%
(epoch7) and95.14% (epoch17) are partial, not final30-epoch results.
Previous clean controls were reused; no noisy case was repeated. This does
not change the paused clean-paper completion counts above. Zero-noise cosine
qualification did not guarantee stability at this noise level.

[Results and failure trajectories](../paper_ready_results/conv3_p90_read_noise_1em3_20260920.md) ·
[Verified coverage](../results/eqprop-conv3-p90-read-noise-1em3-20260920-v1/closeout-validation.json).

## Completed Conv3 p90 beta read-noise follow-up (2026-09-20)

All 24 thirty-epoch outcomes are collected and validated: 15 noisy runs and
nine GPU-matched clean controls, seed0, fixed T=K=8, injected betas
404.141105702/4.42250110273/5.26875648112 for baseline/legacy/ours.
This is a separate validation sensitivity study; it does not replace the
paused clean-paper completion contract counted above. No official test was read.
At sigma5e-4, validation drops are 0.18/3.00/1.38pp. Legacy's higher-noise
curves fluctuate substantially despite passing the final-epoch screen.

[Results and curves](../paper_ready_results/conv3_p90_read_noise_20260919.md) ·
[Defensible beta protocol](../paper_ready_results/beta_selection_protocol_assessment_20260919.md) ·
[Verified coverage](../results/eqprop-conv3-p90-read-noise-20260919-v1/closeout-validation.json).

## Retained iteration counts after the drift review (2026-09-16)

Filip's current choice is **T=K=4 for Conv1, T=K=6 for Conv2, and T=K=8
for Conv3, across baseline, legacy and ours** in the present wide-weight
EP/BPTT and read-noise discussion. The tentative Conv3 baseline T12/K8
increase is not adopted. Reducing voltage drift changed direct EP–BPTT
agreement little in the checked cases, so the descriptive 1% drift ratio
is not a training gate. Existing gradient/residual failures and the
unconfirmed Conv3 baseline beta remain explicit. Historical bounded
longer-T runs retain their own contracts; this documentation decision does
not relabel those results or resume the paused campaign.

[Report: residual drift versus gradient agreement](../paper_ready_results/eqprop_drift_gradient_report_20260916.md)
and [all 25 checkpoint/T comparisons](../paper_ready_results/eqprop_drift_gradient_comparison_20260916.csv).

## Baseline qualification caveat and possible BPTT-only scope (2026-09-16)

Filip asked to retain this caveat for the paper review and suggested that
the bounded-weight results might use BPTT only. **This is an open paper-scope
option, not a finalized exclusion of EqProp or a change to the completion
contract.** Its significance depends on which claims and comparisons the
paper retains.

The baseline training settings and numerical evidence differ by weight range:

| Weight contract / baseline | Training T/K | EqProp beta | Qualification evidence |
|---|---:|---:|---|
| Wide [0,100], Conv1 | 4/4 | 100 | Expanded seed-0 selection passes; beta 100 has no three-seed confirmation in this audit |
| Wide [0,100], Conv2 | 6/6 | 100 | Expanded selection and all three confirmation seeds pass |
| Wide [0,100], Conv3 | 8/8 | 100 | Initialization gradient gate fails: worst cosine .967393 < .99 and maximum norm mismatch .184260 > .10; trained gradient comparisons pass, but the separate trained free-state residual gate fails |
| Bounded, Conv2 | 12/6 | 0.1 | Initialization and newly trained seed-0 best-checkpoint checks pass at all three ceilings; worst cosine .997674 |
| Bounded, Conv3 | 24/8 | 0.1 | Initialization and newly trained seed-0 best-checkpoint checks pass at all three ceilings; worst cosine .997096 |

Qualification concerns the joint T/K, beta, checkpoint and cohort setting.
The bounded revision increased **T only**, keeping K at 6/8; both BPTT and
EqProp were assigned the revised counts. Smaller beta and longer free-phase
relaxation addressed distinct problems. For example, bounded Conv2 at T6/K12
still failed, while T12/K6 passed; Conv3 at T20/K8 still failed the norm gate
at the largest ceiling, while T24/K8 passed. These fixes were not applied to
the wide runs. The bounded checks used four fixed batches at seed 0, so their
passing result is not an expanded-cohort, three-seed confirmation.

Interpretation for the possible BPTT-only bounded comparison: a failure of
EqProp-to-BPTT gradient agreement does not by itself invalidate the BPTT
training result. BPTT-only evidence cannot establish EqProp gradient fidelity
or equilibrium convergence. Any retained BPTT results still need their own
protocol/reuse checks, declared T/K and training-cost disclosure. The original
native-T/K BPTT baselines and revised longer-T baselines remain distinct;
choosing between them is unresolved. The wide Conv3 caveat remains relevant
if its EqProp comparison is retained, even if bounded EqProp is omitted.

This note records the discussion without launching follow-ups or changing
the paused clean campaign. Existing accuracies remain validation evidence;
the final inclusion set and checkpoint rule must be frozen before the
once-read official-test evaluation.

[Wide beta audit](../paper_ready_results/baseline_beta_audit_20260914.md) ·
[Bounded T/K diagnostics](../paper_ready_results/baseline_tk_beta_qualification.md) ·
[Bounded recheck on new BPTT references](../paper_ready_results/baseline_tk_revision/beta_qualification.md) ·
[Original and revised bounded training](../paper_ready_results/baseline_tk_revision/README.md).

### Baseline read-noise priority and T-only follow-up

Later on September16, Filip identified baseline EqProp read-noise training as
the next major result and directed increasing T first to investigate the
wide Conv3 qualification failure. The completed beta100/K8 sweep at
T12/16/24/32 **fixes equilibrium but leaves the initialization gradient
mismatch unchanged** (worst cosine .967393, norm mismatch .184260). No setting
passes both gates, so three-seed confirmation is omitted and no new beta or
training setting is adopted. Filip then explicitly chose to **keep T=K=8**
and sweep Conv3 baseline beta. The [fixed-T8/K8 beta study](eqprop_conv3_baseline_beta_tk8_plan_20260916.md)
tests 10/30/50/75 with the prior100 reference, retaining the equilibrium
caveat separately while selecting and confirming gradient fidelity. This
supersedes the proposed T12/K8 beta follow-up. The clean training backlog
and tentative bounded BPTT-only scope remain unchanged.
[T-only results and evidence](../paper_ready_results/baseline_wide_t_audit_20260916.md).

The fixed-T8/K8 beta study is now complete. Beta10 passes seed0 selection
and confirmation seeds0/2, but fails two initialization comparisons on seed1
(cosine .968773 and norm mismatch .101208). No beta in the tested grid is
confirmed across all three seeds; the confirmation cohort has been consumed.
No common training beta was promoted. [Beta results](../paper_ready_results/conv3_baseline_beta_tk8_20260916.md).

The requested phase-displacement follow-up is also complete: all nine
Conv1/2/3 × baseline/ours/legacy cases, seed0 initialization and BPTT best
checkpoints, on 576 matched validation examples. Conv3 baseline uses beta10
as an explicitly unconfirmed candidate. Its trained H1 RMS displacement is
1.02807e-4 relative to the post-T free state but only 4.57975e-7 relative to
the matched zero-nudge endpoint, a factor of224.5. This directly shows the
importance of separating continued relaxation from the nudging response.
All per-layer values, both signs, voltage scales and qualification caveats
are in the [displacement report](../paper_ready_results/phase_displacement_20260916.md).
These read-only studies do not change the paused clean-training count or
provide official-test accuracy.

The subsequent [T-only displacement check](../paper_ready_results/phase_t_sweep_20260916.md)
is complete for the five affected checkpoint cases, with K and beta fixed.
Direct zero-nudge drift falls below1% of the controlled nudging-response RMS
on every tested batch/layer/sign at T12 for Conv3 baseline initialization,
T24 for its trained checkpoint, T16/T12 for Conv3 legacy initialization/trained,
and T10 for Conv2 legacy initialization. These are the smallest tested points
that remain below the threshold at larger T. Pooled criteria pass earlier at
T16 for trained Conv3 baseline and T10 for trained Conv3 legacy.
The controlled-response RMS changes by less than.001% across all these grids.
Longer T therefore removes residual-relaxation contamination of displacement;
this does not resolve the previous seed1 beta confirmation failure or adopt
a new training contract. All15 included cases validate; one Conv2 completion-
count metadata failure and its unchanged scientific replacement are preserved.

## Separate read-noise sweep: complete

### Baseline extension launched September16

Filip authorized the [22-run baseline plan](eqprop_baseline_read_noise_launch_plan_20260916.md)
after inspecting the exact parameters. It adds20 noisy seed0 trainings at
sigma1e-5/3e-5/1e-4/3e-4/5e-4, plus clean controls for Conv1 beta200 and
Conv3 beta10. Conv1 beta100 and Conv2 beta100 reuse audited existing clean
controls. T=K remains4/6/8; all exact baseline Adam rates remain unchanged.

Fifi was prioritized and began production at15:47:43 CEST with two concurrent
Conv3 workers at Filip's explicit request after the timing comparison. Local
runs Conv2 beta100, Akib Conv1 beta100 and Nom Conv1 beta200. All22 local
smokes pass; the remote queues also run semantic smokes and timing before
production. The first Fifi wrapper failed before training on a quoting error;
its corrected, syntax-checked replacement retains the same frozen science.
This baseline extension is complete and does not change the paused clean
campaign's202/216 count. Its60 physical GPU-hour allowance is separate.
Conv3 beta10's seed1 gradient failure and T8 residual caveat remain explicit.

The [live collected-results tracker](../paper_ready_results/baseline_read_noise_run_status_20260916.md)
and [CSV](../paper_ready_results/baseline_read_noise_run_status_20260916.csv)
include only full validated results as completed; running epochs are progress.
At 16:34 CEST, all 11 Conv1 trainings are collected and revalidated, including
the new beta 200 clean control. Beta 100's largest final clean-relative loss
is .04 percentage points; beta 200's is 0.00. Both host queues exited 0, and
their production trees match the local copies by checksum. Conv2's five
cases and Conv3's six cases remain active/queued. This is single-seed,
cross-environment validation evidence; see the
[partial report](../paper_ready_results/baseline_read_noise_results_20260916.md).
At 00:53 CEST on September17, all five Conv2 cases bring the total to
16/22 collected and revalidated trainings. Best/final validation is
97.30/97.38/97.36/97.34/97.24% in increasing sigma order, versus the historical
clean reference's97.22%. These increases are descriptive, not evidence of a
noise benefit across differing environments. At sigma5e-4, ours retains higher
absolute accuracy (97.60%) than baseline (97.24%), while baseline shows no
observed clean-relative loss. Beta values and environments differ. Local's
queue exited0 at00:50:57 CEST and released its GPU. Only the six Fifi Conv3
cases remain; the first pair is in epoch25/30.
At 02:48 CEST, the first Conv3 pair brings the total to18/22 validated
trainings. The new beta10 clean control is97.64% best/final; sigma1e-5 is
97.48%, a.16-point clean-relative loss. The old beta100 clean final is97.68%,
close descriptively but not this curve's control. Fifi's first pair exits0
at02:44:53 CEST and matches local files by checksum; sigma3e-5/1e-4 are now
running concurrently. Sigma3e-4/5e-4 form the final pair. The first-pair
duration is10h57m; estimated whole-study completion is around00:40 CEST
September18. The beta-confirmation/residual caveats remain unresolved.
At 13:46 CEST, the second Conv3 pair brings coverage to20/22 validated.
Sigma3e-5/1e-4 finish at97.04/96.64%, losses of.60/1.00 points against the
new97.64% beta10 clean control. Pair2 exits0 at13:42:23 CEST, and its remote
files match local checksums. The final pair, sigma3e-4/5e-4, starts
immediately with two workers; the around00:40 CEST completion estimate holds.
At sigma1e-4, ours retains higher absolute accuracy (97.62%), with a similar
observed clean-relative loss (1.02 points); legacy is83.08%, down15.70 points.
Different betas, inherited training contracts and environment limits remain.
Final closeout, September18: **22/22 trainings collected and fully
revalidated**. The final pair exits0 at00:39:15 CEST; sigma3e-4/5e-4 finish
at96.48/96.04%, losses of1.16/1.60 points against97.64% clean. At sigma5e-4,
ours is96.60% (2.04-point loss) and legacy75.44% (23.34-point loss): ours
retains higher absolute accuracy, while baseline loses fewer points from
its own clean control. Fifi's queue/wrapper exit0 and its GPU is released;
all remote production checksums match local copies. All45 smoke/timing
runs validate. Settled usage is43.294845/60 physical GPU-hours. There are
no full-training failures, exclusions or replacements; the pretraining
wrapper failure is preserved. No official-test reads or clean-campaign
resumption occurred. [Completed report](../paper_ready_results/baseline_read_noise_results_20260916.md) ·
[Closeout proof](../paper_ready_results/provenance/baseline_read_noise_closeout_20260918.json).
The completed30-run ours/legacy sweep below remains a separate study.

All **30/30** single-seed ours/legacy read-noise trainings are collected and
validated as of September 16, 06:22 CEST. The final training finished at
**06:20:46**, before the 08:00 deadline; all training GPUs are released.
The sweep used **77.232472 physical GPU-hours** across its two separately
funded windows. It does not change the clean 202/216 count above or resume
the fourteen paused clean cases. No official-test read occurred.

[Results and scientific limits](../paper_ready_results/read_noise_sweep_results.md) ·
[All 30 run bundles](../paper_ready_results/read_noise_run_status_20260914.md) ·
[Figure](../paper_ready_results/read_noise_validation.png) ·
[Completed continuation record](eqprop_read_noise_continuation_plan_20260915.md).

## Current admission update

**Evening review, September 14, 23:19 CEST:** Loulou's last admitted clean
pilot is collected and validated at **77.68/77.68%** best/final validation.
It matches its revised BPTT control and all 30 epoch-order/split fingerprints;
both exit receipts are zero. Conv3 baseline pilot coverage is now **1/3**.
There are **202/216** current-contract results and **14 unstarted runs**:
three Conv2 baseline EqProp, three Conv3 baseline BPTT, and eight Conv3
baseline EqProp. No clean follow-up was launched. The 6.6025-GPU-hour pilot
is settled; total commitment, including the completed beta audit, is
**270.300605/300 GPU-hours** (269.300605 settled, one check hour reserved).
The final paper reuse/checkpoint seal and official-test evaluations remain
pending. [Full review](paper_experimental_review_20260914.md) ·
[Pilot proof](../paper_ready_results/baseline_tk_revision/conv3_seed0_tight_pair_check_20260914.json).

### Earlier afternoon snapshot

The following host states, counts, and reservations describe the earlier
14:26 CEST handoff; the evening collection above supersedes them.

**New admissions are paused for Filip's review.** Main completed at **13:15:08 CEST** and is idle; its next seed-2 case was not launched. The stated default is to let the one remaining single-case worker on Loulou finish and stop at its current run boundary. Do not admit, launch or resume further work automatically. The budget/placement extension remains unapproved and is deferred to the review.

[Accuracy agreement analysis](../paper_ready_results/results_agreement_20260914.md) · [Review of completed and remaining work](../paper_ready_results/review_20260914.md) · [Remaining runs](../paper_ready_results/current_contract_remaining_runs.md) · [Validation tables](../paper_ready_results/current_contract_validation_tables.md)

The revised contract has **201/216 collected and validated training results; 15 remain**. All 198 original bundles are preserved; 21 revised bundles bring the preserved total to **219**, including 18 native-T/K BPTT baselines retained as earlier evidence. There are **64/72 complete three-seed conditions**. Official MNIST test evaluations remain zero.

| Target | Current state | Last verified progress | Reservation / expected finish |
|---|---|---|---|
| Local/Main RTX 3090 | Idle after Conv3 baseline BPTT seed 1/Gmax 1e-3, T24/K8, source v9 | Collected 30/30 epochs; **89.58/89.58%** best/final validation. Worker and supervisor exited zero; GPU worker absent. | Three-hour reservation settled at **2.166111 GPU-hours**; no next case admitted |
| Akib RTX 3080 | Idle after Conv2 baseline EqProp seed 1/Gmax 5e-4 | Collected 30/30 epochs at 14:15:59 CEST; **91.50/91.50%** best/final validation. Both exits zero, GPU worker absent; matched BPTT best 91.52%. | Eight-hour reservation settled at **2.243056 GPU-hours**; no follow-up launched |
| Nom-cool-1 RTX 3090 | Idle after Conv2 baseline EqProp seed 2/Gmax 1e-4 | Collected 30/30 epochs at 14:20:26 CEST; **85.22/85.22%** best/final validation. Both exits zero, GPU worker absent; matched BPTT best 85.24%. | Eight-hour reservation settled at **2.120000 GPU-hours**; no follow-up launched |
| Loulou RTX 5090 | Finishing first Conv3 baseline EqProp pilot, seed 0/Gmax 1e-4, beta .1, T24/K8, source v11 | 11/30 epochs complete at 14:23 CEST; supervisor 1107229, worker 1107238, GPU 1107255 | 8 GPU-hours; expected around 18:30 CEST, then stop |

Remaining coverage is **Conv2 EqProp: 3 runs; Conv3 BPTT: 3; Conv3 EqProp: 9**. One is running and fourteen are unstarted. All three Main seed-1 BPTT ceilings are collected; all three seed-2 ceilings remain unstarted.

**Conv2 EqProp's numerical and full-pilot gates pass** at beta .1, T12/K6. All three full seed-0 pilots and three repetitions are collected. The tight ceiling now has all three seeds: BPTT 85.393 ± .142% versus EqProp 85.333 ± .103% best validation. The newly collected middle seed-1 and tight seed-2 repetitions are each .02 pp below matched BPTT; their full 30-epoch cohort/order checks pass. All nine revised Conv2 BPTT runs are collected. [Matched-cohort audit](../paper_ready_results/baseline_tk_revision/conv2_matched_cohort_check.json).

**Conv3 numerical qualification passes** at beta .1, T24/K8, but full EqProp pilot coverage is still 0/3. Loulou's current case is the first pilot; two additional seed-0 ceilings and all six repetitions remain unstarted. Repetitions require three stable full pilots and a new frozen guard. Source v11 (7b67ca41, 712 inputs) remains unchanged. [Numerical gate](../paper_ready_results/baseline_tk_revision/beta_qualification.md) · [Full-pilot record](../paper_ready_results/baseline_tk_revision/pilot_stability.md).

Both original bounded Conv3 EqProp waves are fully collected and validated. Loulou's three ours cases settled 12.389722 GPU-hours and Nom's two legacy cases settled 12.382778, with successful worker/supervisor receipts and identical local/remote checksums. The complete three-seed BPTT–EqProp comparisons have maximum paired best-validation differences of .66 pp for ours and .18 pp for legacy; no seed is excluded. [Ours](../paper_ready_results/bounded_conv3_ours_three_seed_validation.md) · [Legacy](../paper_ready_results/bounded_conv3_legacy_three_seed_validation.md).

Settled compute is **262.418304 GPU-hours**, with **9 reserved: 271.418304/300 committed**. The reservations cover the one active eight-hour case and the existing one-hour checks allowance. Retain them until terminal collection and accounting. The full-training forecast is about 335.6 hours at current placement, or 348.3 hours under the unapproved faster plan. The Nom runtime diagnostic projects 9.788 hours per Conv3 EqProp case, exceeding its current eight-hour cap. The proposed 375-hour campaign cap, twelve-hour Nom case limit and split pilot placement remain deferred to review. [Proposal](paper_training_parallel_acceleration_proposal_20260914.md).

Jean Zay is authorized for at most **four concurrent campaign GPUs**, but SSH checks through 12:11 CEST timed out; the last scheduler record lists maintenance through **September 16, 18:00 CEST**. No new job is submitted or reserved. The prepared seed-0 array is superseded by the already-started Loulou group. Do not submit it or any other prepared follow-up during the pause. Loulou remains the sole weekday campaign RTX 5090. The tested two-EqProp packing achieved only .775x aggregate throughput.

Last full host check was **14:23:38 CEST**. Main, Akib and Nom are idle with successful terminal receipts; Loulou shows GPU activity and advancing artifacts without detected errors. Monitoring is handed back for the requested review; no automatic continuation is scheduled.

Historical bookkeeping exception: the three Main seed-0 BPTT bundles and worker exit-zero receipt validate, but the old supervisor finish/exit receipt is missing. Its exit code remains unknown; the reservation was conservatively settled at 6.438082 GPU-hours through verified process absence. Preserve this exception and the recovered transfer/approval-service records.

## Earlier campaign history

The entries below preserve the earlier admissions and scientific checks.
Current placement and reservations are given above.

The following historical entries precede the current update; the
per-cell CSVs carry the latest queue state.

The initial 25 trainings and local checks reserve **141 GPU-hours** in
[`gpu_hours.csv`](../results/paper-training-completion-20260911-v1/gpu_hours.csv).
Each training is limited to two hours for Conv1 or eight hours for Conv2/3.
Initial handles: Main tmux `main/@1` (nine Conv1 BPTT configs), Akib nohup
PID `7068` (nine Conv2 BPTT configs), Trex tmux `paper-conv3-bptt` (six Conv3
BPTT configs), and Jean Zay `2023182_0` (corrected wide Conv1 legacy EP pilot).
All four workers reached real training. Jean Zay started immediately through
backfill despite its pessimistic scheduling preview. This paragraph records
the initial admission; current reservations are in the linked ledger.

At 14:55 CEST, the nine initial Main cases and corrected EP pilot are complete
and collected. All nine wide EP pilots passed, releasing wide seeds 1/2.
Jean Zay array `2024429_[0-11]%4` completed all twelve tasks, is collected
with zero checksum differences, and consumed 3.638333 GPU-hours. Its six
bounded Conv1 baseline/ours pilots pass the full-training stability rule;
the largest best-to-final drop is 2.44 percentage points. Bounded coverage is
then only 6/27 pilots. Under the September 12 extension, each complete qualified group may release its repetitions after budget admission.
Their [paired validation curves](../paper_ready_results/bounded_ep_pilot_validation.md)
show the matched BPTT and EP trajectories, including the shared decline at
the tight ceiling. This remains single-seed validation evidence.

Main `main/@2` completed its nine bounded Conv1 seed-1 BPTT cases in
.975833 GPU-hours; Main `main/@3` completed all nine seed-2 cases in
1.021389 GPU-hours. All 27 bounded Conv1 BPTT cells are now collected.
Main `main/@4` reached training at 16:27 CEST for the
three bounded Conv3 seed-1 BPTT schemes at Gmax=1e-4;
remaining ceilings of this BPTT architecture/seed stay assigned to Main.
nom-cool-1's initial three bounded Conv3 seed-2 cases completed and are
collected in 3.198889 GPU-hours. Nohup `1231687` now runs its six remaining
cases at Gmax=5e-4/1e-3, with 48 GPU-hours reserved and a 6–8-hour estimate.
Main and nom-cool-1 retain their architecture/seed comparison groups.

Main's three Conv3 seed-1 cases at Gmax=1e-4 are now complete and collected,
taking 3.956667 GPU-hours. At 20:27 CEST, Main `main/@5` reached training on
the next three cases at Gmax=5e-4, using the unchanged source-v3 configs and
shared initializer. It reserves 24 GPU-hours, with about four elapsed hours
expected. After settling the completed batch and admitting this continuation,
total committed time is 288.8936/300 GPU-hours. The final three seed-1 cases
at Gmax=1e-3 remain assigned to Main for a separate later admission.

At 18:59 CEST, Akib's initial nine-case batch is complete and collected with
zero checksum differences, taking 5.588333 GPU-hours. All 27 corrected bounded
BPTT seed-0 references are now available. The Conv2 legacy beta ladder
subsequently completed on CPU against its three collected references and
qualified beta .003. Akib nohup `22979`
has reached training on the next nine bounded Conv2 seed-1 cases, reserving
72 GPU-hours with about six elapsed hours expected. Total settled time plus
full outstanding reservations is 284.9369/300 GPU-hours. nom-cool-1's first
Gmax=5e-4 case is also collected, bringing overall coverage to 97/216.

Trex's wide Conv3 baseline seed-1 case completed and is collected at
97.74% best validation. Its paused old queue was retired after that one case;
the conservative worker elapsed time is 4.273056 GPU-hours. The raw EXIT trap
wrote zero, but the six-case list did not complete: five cases were unstarted.
Its two remaining seed-1 schemes stay on Trex and require fresh admission.
They remain held while Trex is occupied or another campaign 5090 is active.

Loulou became idle and now runs the complete wide Conv3 seed-2 group under
nohup `721734`, with a 24 GPU-hour reservation and a 3–4-hour initial estimate.
It is the sole campaign 5090. Its Python 3.12.13 / PyTorch 2.11.0+cu128
environment is an exact archived copy from Trex (`21a06eb5`); source v3
`82b4e032` and train-only MNIST are hash-verified. Three same-config GPU smokes
passed in 5.24 seconds and are locally collected and validated, including
shared initial states and finite float32/zero-bias checkpoints. See the
[local smoke audit](../results/paper-training-completion-20260911-v1/checks/loulou_smokes_local_validation.json).
The earlier nom-cool-1 wide wrapper remains an unlaunched fallback.

At 18:24 CEST, an unrelated `kellian` process (`722698`, about 24 GiB)
was also present on Loulou; it started after our launch. The current baseline
case is progressing under its original eight-hour limit. Source v3 checks
GPU occupancy before each subsequent case and will hold the remaining list
if that process is still present. The initial group ETA is therefore uncertain;
the comparison group stays on Loulou and the other user's process is untouched.
The 18:53 CEST inventory found that competing process had ended. Loulou's
case is advancing faster again; Trex is also idle, but its two continuations
remain held while Loulou uses the sole weekday campaign 5090.

Twelve wide Conv2/3 EP replications are admitted in Jean Zay array
`2028114_[0-11]%4`, after twelve passed local CPU smokes and the previous
array's successful completion. It reserves 96 GPU-hours and has reached real
training; all six Conv2 repetitions and the first two Conv3 repetitions are
collected, and the final four Conv3 repetitions are running. After settling the
completed Trex and nom-cool-1 batches and admitting the new work, total settled
time plus full reservations is 279.3486/300 GPU-hours at 17:54 CEST.
The Conv2 EP repeats take about two hours each on V100. The retained Conv3
EP pilots took 4.3–5.0 hours each on the same target class; the newly completed
Conv3 repetitions took about 4.3 hours. The current estimate is 01:30–02:00
CEST on September 12, subject to the remaining task rates. Its 8-hour
per-task limit and 96-hour reservation are unchanged. No next EP array can
start until this array is terminal.
Bounded Conv1 baseline/ours qualify at injected beta .1/.03, and Conv2 ours
qualifies at .001 across all three ceilings.
The corrected bounded Conv1 legacy references also qualify at injected beta
.03 across all three ceilings; its three seed-0 trainings await admission.
Conv3 legacy qualifies at injected beta .001 across all three ceilings;
its full seed-0 pilots also await the next EP batch. See its
[numerical qualification](../results/paper-training-completion-20260911-v1/checks/bounded_conv3_legacy_selection.json).
Conv3 ours qualifies at injected beta .003, with worst cosine .9969936 and
maximum symmetric norm difference .0850763 across its three ceilings.
The [reconciled beta ledger](../results/paper-training-completion-20260911-v1/checks/bounded_beta_selection.json)
now includes all nine resolved groups. Seven groups (21 ceiling conditions)
are numerically qualified; two baseline groups remain held. Conv2 legacy
qualifies at .003 after .03 fails: worst selected cosine .9979283 and maximum
norm difference .0283384, with all residual gates passing. Its CPU ladder
took 578.38 seconds. Replication configs now require their complete stable seed-0 group and an available full budget reservation.
The [collected beta summary](../paper_ready_results/bounded_beta_qualification.md)
and its CSV preserve all 99 completed beta/ceiling replays in
`paper_ready_results/provenance/bounded_beta_checks/`; their source/copy
hashes and canonical validators pass. The declared numerical checks are
complete; full pilot coverage remains incomplete.
The numerical evidence is in the separate
[Conv1 legacy check](../results/paper-training-completion-20260911-v1/checks/bounded_conv1_legacy_selection.json).
The first 19 new bundles also pass an independent
[initialization and cohort audit](../results/paper-training-completion-20260911-v1/checks/collected_pairing_audit.json).

Preparation produced 178 exact configs/candidates and 18 shared initializer
files, preserving the three original bounded seed-0 files. All 34 focused
regression tests, 24 BPTT GPU smokes, and the EP pilot GPU smoke passed.
The corrected Conv1 legacy beta-3 numerical check passed 16/16 comparisons
at initialization and best BPTT: minimum cosine .9998157, maximum symmetric
norm difference .0022935, finite phases, and passing residual gates.
The eight retained wide EP pilots pass the full-budget stability check and
their reconstructed float64 initialization hashes match their training records.
The bounded checks ran on CPU without occupying a training GPU.
Conv3 baseline/ours/legacy finished their independent CPU checks, with separate
reports under `checks/bounded_conv3_*_selection.json` and a reconciled common
ledger. The old sequential dispatcher was retired after its active replay
completed and validated; all scientific configurations and completed replay
bundles were preserved. Conv2 legacy subsequently completed its independent
check and the final common ledger reconciliation.
All 42 future bounded BPTT configurations also passed exact-config CPU smokes
and canonical validation. Their plain lists are prepared, with admission
still subject to available lanes and full wall-time reservations.

The next 15 qualified bounded EP seed-0 pilots are frozen and staged on
Jean Zay in source v4 `2092876a`: Conv1 legacy plus Conv2/Conv3 ours and legacy,
each at all three ceilings. Its 637 input hashes and plain config list match
locally and remotely; all 355 Python files and both transport scripts are
unchanged from v3. The three Conv1 pilots were submitted as `2060809` on September 12 after fresh local semantic smokes. Use successive arrays:
three Conv1 cases with two-hour Slurm limits (six GPU-hours reserved), then
twelve Conv2/3 cases with eight-hour limits (96 GPU-hours reserved). Their
combined expected duration is about 10–13 elapsed hours on at most four V100s.
Each admission requires the preceding array to be terminal, available budget,
and the immediate local semantic smoke. See the
[staging record](../results/paper-training-completion-20260911-v1/launch/jz_bounded_pilots_next.staging.json).

Newly completed bundles are copied and validated as they finish. See the
[new validation results](../paper_ready_results/training_completion_results.md),
[completed three-seed wide Conv1 comparison](../paper_ready_results/wide_conv1_validation.md),
[completed three-seed wide Conv2 comparison](../paper_ready_results/wide_conv2_validation.md),
[completed bounded Conv1 BPTT table](../paper_ready_results/bounded_conv1_bptt_validation.md),
[source identity](../results/paper-training-completion-20260911-v1/source_identity.json),
and [collection provenance](../paper_ready_results/provenance/training_completion_collection.json).
The initial backlog tables later in this file retain the pre-launch snapshot;
the coverage table and CSVs below are the continuing counts.

## Earlier native-T/K failures

These original failures motivated the subsequently authorized baseline T/K
revision. Numerical qualification now passes for both architectures; Conv2
has three stable full pilots and Conv3 has one. The records below preserve
the failed native-T/K conditions, not the current training backlog.

The bounded Conv2 baseline group is held after exhausting injected betas
100, 10, 1, .1, and .01 at unchanged T=K=6. All phase and residual checks
pass. At Gmax=1e-4, however, eight trained-checkpoint layer/batch comparisons
still fail at beta .01: minimum cosine .9626977 and maximum symmetric norm
difference .2358226, against .99/.10. The failure barely changes from beta
.1, so smaller beta alone is not supported as a resolution by these data.
The other two ceilings pass at .1 and .01, but the approved rule requires
one qualified beta across all three. All three baseline pilots therefore
stay unlaunched pending the scientific choice; other eligible work continues.
See [beta selection evidence](../results/paper-training-completion-20260911-v1/checks/bounded_beta_selection.json).

Bounded Conv3 baseline also exhausts the same five-point ladder at T=K=8.
The tight ceiling passes both gates at beta .1 and .01, but no beta passes
all three ceilings. At beta .01, Gmax=5e-4 has minimum cosine .7332778 and
maximum symmetric norm difference .4390250; Gmax=1e-3 has .1412846 and
.8490334. At the largest ceiling, eight of 128 residual rows fail; the worst
projected residual p90 is .022109 after the free phase, against .01.
The higher-ceiling gradient mismatch changes little over beta 1/.1/.01.
All three Conv3 baseline pilots remain unlaunched. Changing T/K or extending
the ladder would be a new scientific choice, so the current contract stays
fixed. See the [complete Conv3 baseline ladder](../results/paper-training-completion-20260911-v1/checks/bounded_conv3_baseline_selection.json).

## Coverage and result locations

This table preserves the original collection before revised baseline results
are overlaid. Current coverage is in the [202/216 current-contract
summary](../paper_ready_results/current_contract_summary.json) and
[remaining-run list](../paper_ready_results/current_contract_remaining_runs.md).

| Block | Required runs | Collected and validated | Full training remaining | Official-test results |
|---|---:|---:|---:|---:|
| Table 1: wide BPTT | 27 | 27 | 0 | 0 / 27 |
| Table 2: wide centered EP | 27 | 27 | 0 | 0 / 27 |
| Table 3: bounded BPTT | 81 | 81 | 0 | 0 / 81 |
| Table 3: bounded centered EP | 81 | 63 | 18 | 0 / 81 |
| **Total** | **216** | **198** | **18** | **0 / 216** |

The original collection contained 38 seed-0 runs. Its initial backlog was Conv1 **61**,
Conv2 **60**, and Conv3 **57**, totaling **4,120 epochs** at the unchanged
10/30/30-epoch budgets. These are run and epoch counts, not a GPU-hour estimate.
They exclude numerical checks, operational smokes, and evaluation-only work.
They assume all 38 retained candidates pass the eventual frozen-contract reuse
audit; a failed reuse audit must explicitly revise the count.

| File or directory | Purpose |
|---|---|
| [Validation tables](../paper_ready_results/validation_results.md) | All 38 collected seed-0 best/final validation scores, linked to their copied results |
| [All 216 run statuses](../paper_ready_results/run_status.csv) | One row per table/algorithm/architecture/scheme/ceiling/seed, with action, dependency, source, and current status |
| [Remaining runs](../paper_ready_results/remaining_runs.csv) | The current training backlog, including parent configs and scientific dependencies |
| [Collected results](../paper_ready_results/collected_results.csv) | All included sources, copied result paths, scores, source versions, and result hashes |
| [Bundle collection](../paper_ready_results/bundles/) | Full run directories with original manifests, configs, metrics, best/final and other saved checkpoints, and artifacts |
| [Initialization assets](../paper_ready_results/assets/bounded_uniform/) | Preserved Conv1/Conv2/Conv3 seed-0 bounded initializers |
| [Original collection validation](../paper_ready_results/provenance/collection_validation.json) | Source-to-copy file/link equality and canonical validation for the original 38 bundles |
| [Original checkpoint validation](../paper_ready_results/provenance/checkpoint_validation.json) | Tensor, precision, bias, initialization, and recorded cohort/order checks for the original collection |
| [New-run collection validation](../paper_ready_results/provenance/training_completion_collection.json) | Per-file/link hashes, canonical/config validation, finite bounded checkpoints, zero biases, and shared initializer checks for every newly collected training |

The CSVs are snapshots of this ledger, not a launcher. Paths in
`collected_result` are relative to `paper_ready_results/`; original source and
parent-config paths are relative to the repository root. The target/job fields
identify admitted work, including cells waiting serially in a recorded worker.

## Initial collected evidence and backlog (before launch)

### Table 1: wide BPTT — 9 / 27 training runs

All nine seed-0 architecture/scheme cells are copied under
[`bundles/table1_wide_bptt/`](../paper_ready_results/bundles/table1_wide_bptt/).
Baseline and ours use the original zero-bias study. Legacy uses corrected
physical-KCL training for all three architectures.

The launch plan's outstanding retrieval item is resolved: corrected Conv1
legacy Adam is the complete ten-epoch run
[`001_05_legacy_adam_bias_zero_seed0_4469a635`](../results/perfectdiode-conv1-legacy-physical-kcl-same-weights-and-bptt-rerun-seed0-20260825-v1/same_lr_bptt/001_05_legacy_adam_bias_zero_seed0_4469a635/).
Its [collected result](../paper_ready_results/bundles/table1_wide_bptt/conv1/legacy/seed0/result.json)
reports **96.50% best / 96.44% final validation accuracy**. It had already been
relocated into this worktree before collection, so no replacement training is
needed merely to recover its files.

Remaining: seeds **1 and 2** for every architecture/scheme — **18 runs**.

### Table 2: wide centered EP — 8 / 27 training runs

Baseline and ours are collected for every architecture, together with corrected
legacy Conv2 and Conv3, under
[`bundles/table2_wide_ep/`](../paper_ready_results/bundles/table2_wide_ep/).
Corrected Conv1 legacy seed 0 is missing. Its historical uncorrected result is
not a substitute.

Remaining: seeds **1 and 2** for all nine architecture/scheme cells, plus
corrected Conv1 legacy **seed 0** — **19 runs**. Check the corrected Conv1
legacy EP condition before training its three seeds with one frozen config.

### Table 3: bounded BPTT — 21 / 81 training runs

The 18 baseline/ours seed-0 cells and three corrected Conv3 legacy seed-0 cells
are copied under
[`bundles/table3_bounded_bptt/`](../paper_ready_results/bundles/table3_bounded_bptt/).
Every ceiling retains the same architecture-specific seed-0 initializer.

Remaining: seeds **1 and 2** for all 27 architecture/scheme/ceiling conditions
— **54 runs** — and corrected legacy **seed 0** for Conv1/Conv2 at all three
ceilings — **6 runs**. Total: **60 runs**.

### Table 3: bounded centered EP — 0 / 81 training runs

No matching bounded EP training manifests were found in the audited result
roots. All **27 architecture/scheme/ceiling conditions × seeds 0/1/2** remain.
The wide EP beta values are references for the short checks, not a qualified
bounded EP contract. No empty result bundles have been created for these runs.

## Initial remaining training coverage

A bounded entry applies separately at **each** `G_max` in
`{1e-4, 5e-4, 1e-3}`. Numbers in parentheses count full training runs.

| Architecture | Scheme | Wide BPTT: missing seeds | Wide EP: missing seeds | Bounded BPTT: missing seeds at each ceiling | Bounded EP: missing seeds at each ceiling | Total |
|---|---|---|---|---|---|---:|
| Conv1 | baseline | 1,2 (2) | 1,2 (2) | 1,2 (6) | 0,1,2 (9) | 19 |
| Conv1 | ours | 1,2 (2) | 1,2 (2) | 1,2 (6) | 0,1,2 (9) | 19 |
| Conv1 | legacy | 1,2 (2) | 0,1,2 (3) | 0,1,2 (9) | 0,1,2 (9) | 23 |
| Conv2 | baseline | 1,2 (2) | 1,2 (2) | 1,2 (6) | 0,1,2 (9) | 19 |
| Conv2 | ours | 1,2 (2) | 1,2 (2) | 1,2 (6) | 0,1,2 (9) | 19 |
| Conv2 | legacy | 1,2 (2) | 1,2 (2) | 0,1,2 (9) | 0,1,2 (9) | 22 |
| Conv3 | baseline | 1,2 (2) | 1,2 (2) | 1,2 (6) | 0,1,2 (9) | 19 |
| Conv3 | ours | 1,2 (2) | 1,2 (2) | 1,2 (6) | 0,1,2 (9) | 19 |
| Conv3 | legacy | 1,2 (2) | 1,2 (2) | 1,2 (6) | 0,1,2 (9) | 19 |
| **Total** | | **18** | **19** | **60** | **81** | **178** |

The seven corrected seed-0 additions above were outside the previously closed
physical-KCL correction scope. They are newly needed for this three-table plan,
not failed or unfinished jobs from that closed study.

## Work to finish, in dependency order

- [x] Recover the corrected Conv1 legacy wide BPTT bundle.
- [x] Collect and validate the plan's 38 existing training candidates.
- [x] Preserve all three bounded seed-0 initializer files and their hashes.
- [ ] Complete the remaining remote discovery limitation: Riri's SSH host-key
  verification failed. Recheck any newly available or relocated result roots
  before allocating duplicate training. See the inventory below.
- [x] Freeze the corrected physical-KCL source revision and materialize the
  178 missing configs from the exact parent learning-rate vectors. Preserve
  Adam settings, architecture, preprocessing, zero biases, precision, split,
  epoch budget, and the accepted `T/K` operating points.
- [x] Create six independent bounded initializer assets: Conv1/2/3 × seeds
  1/2. For seed `s`, set both model and shuffle seed to `s`, keep split seed
  0, and update both initializer-path fields and their hashes. Share each
  architecture/seed asset across schemes, ceilings, and algorithms. Verify
  equality within a paired seed and inequality between seeds; record dtype
  conversion and reset optimizer state. Changing `seed` alone is insufficient.
- [x] Verify wide numerical initialization equality across schemes and
  BPTT/EP for each architecture/seed, preserving the Kaiming recipe.
- [x] Run the short corrected Conv1 legacy EP numerical check using its
  inherited learning rates, `T=K=4`, and one-decade injected beta `3`; freeze
  one config for its three full seeds.
- [x] Finish and collect the **37 new wide training runs**: 18 BPTT and
  19 EP. Both wide tables have all 54 training bundles including retained seeds.
- [x] Finish and collect the original **60 new bounded BPTT training runs**.
  All 81 original bounded BPTT bundles are preserved.
- [ ] Complete the **18 matched baseline BPTT replacements** after the T/K
  revision. All nine Conv2 runs and all six Conv3 seed-0/1 runs are collected;
  only Conv3 seed 2 at the three ceilings remains. Earlier
  native-T/K baselines do not fill these revised cells.
- [x] Complete the original bounded EP numerical ladders across all 27
  ceiling conditions and record both failures and passing groups. Seven groups
  qualified; the two baseline groups required the authorized T/K revision.
- [x] Qualify the revised baseline beta on all **six new seed-0 BPTT best
  checkpoints**. Both architectures have 3/3 passing ceilings at beta .1.
- [ ] Complete all three revised seed-0 EqProp pilots per architecture before
  its seeds 1/2. Conv2 has 3/3 stable pilots; Conv3 has 1/3.
- [ ] Finish the bounded EP contract's **81 full training runs**. Currently
  70 are collected. Three Conv2 and eight Conv3 revised baseline runs remain;
  all original ours/legacy repetitions are complete. Preserve every seed and ceiling.
- [ ] Reconcile every planned seed, collect remote outputs, and validate all
  216 included bundles. Name operational failures, replacements, exclusions,
  divergent outcomes, and any changed reuse count. Preserve poor seed outcomes.
- [ ] Complete the fail-closed paper reuse and paired-comparison audit in the
  [experiment definition](conv_paper_experiment_definition.md#existing-ordinary-mnist-reuse-gate)
  and [matching protocol](conv_paper_one_seed_bptt_eqprop_protocol.md). Record
  exact initialization, cohort/order, rates, `T/K`, zero-bias, beta/noise, and
  precision checks. Resolve or explicitly retain the known Conv3 residual and
  one-decade gradient caveats in the final claim. A changed beta, precision,
  or relaxation contract requires reconsidering affected seed-0 reuse.
- [ ] Seal the complete comparison contract, all three seed outcomes per
  condition, inclusion rule, and maximum-validation-selected checkpoint
  identities and hashes before official-test access.
- [ ] Perform **one evaluation-only official MNIST test pass per eligible
  sealed run**, up to 216 passes for the complete grid. Retain separate paper
  evaluation bundles with the selected checkpoint hash, 10,000 examples,
  explicit `test` split, and one-read record. Do not overwrite selection bundles.
- [ ] Produce Tables 1/2 and both Table 3 panels with all three seeds, mean,
  and sample standard deviation (`ddof=1`). Disclose BPTT float32 versus EP
  float64 and the finite-relaxation qualification limits. Verify 216 training
  cells and the corresponding eligible test results before marking this
  ledger complete and promoting final small tables/figures.

For each future launch, follow the repository workflow: make the cases,
target, compute budget, expected duration, and result path visible; add its
persistent planned row; run the applicable same-runner smoke and scientific
gates; then launch and record the handle/job. Complete exact configs use
`python -m experiments.exact_run`. No broad rho/LR search is in this plan.
Choose and check targets at launch time rather than treating today's
availability as a reservation.

## Contract to preserve

| Setting | Conv1 | Conv2 | Conv3 |
|---|---|---|---|
| Epochs | 10 | 30 | 30 |
| `T=K` | 4 | 6 | 8 |
| Input gain | 40 | 100 | 360 |
| Wide EP injected beta: baseline / ours / legacy | 100 / 30 / 3 | 100 / 10 / .03 | 100 / 3 / .001 |

The supplied plan uses Adam; baseline `(A_V,A_I)=(1,1)`, ours `(4,1)`,
legacy `(4,.25)`; wide bounds `[0,100]`; bounded `G_min=1e-5` with three
ceilings; and clean perfect-diode training. Bounded initialization remains
`Uniform[1e-5,1e-4)` at every ceiling. Ordinary MNIST uses the fixed
55,000/5,000 split, split seed 0, train batch size 16, validation batch size
64, and model/shuffle seeds 0/1/2. All biases are frozen at zero.

Copy exact rates from the archived configs; do not use rounded manuscript
values. Wide EP inherits the matching wide BPTT rates. Bounded EP initially
inherits the corresponding bounded BPTT rates, fixed across ceilings, subject
to its numerical qualification. Preserve float64 centered frozen-current EP,
its current normalization, and the distinction between base and injected beta.

This ledger follows the expanded three-table scope in the supplied plan.
Historical SGD comparisons, medium-affine MNIST, read/write-noise studies,
other ceilings, and the earlier bounded-initializer search are not extra
training obligations in its 178-run count.

## Collection validation and limitations

The original 38 copied bundles contain **1,578 regular files and 550 internal symlinks**,
totaling **2,408,530,205 bytes** (about **2.24 GiB**). Source and destination
file hashes, directory entries, and symlink targets match. All **38 source
and copied canonical validators pass**, including every result-indexed artifact.
The immutable source manifests/results retain their original bytes and IDs.

The configuration audit checks the matrix coordinates, exact parent learning
rates, full epoch metrics, dataset/preprocessing, disabled official-test
access, amplification, bounds, gain, `T/K`, explicit diode dictionaries,
zero bias rates, and EP settings. All **76 best/final checkpoints** contain
finite tensors, obey their weight bounds, have exact-zero biases and the
expected dtype, and match their NumPy exports.

All three copied bounded initializers match their recorded source hashes.
The **18 available saved epoch-zero bounded checkpoints** match their
architecture's initializer tensor-for-tensor; the three corrected Conv3
bounded runs do not save an epoch-zero checkpoint, so their initializer
evidence here is the recorded source path/hash. The **eight existing wide
BPTT/EP pairs** match the recorded training/validation cohort hashes and full
minibatch-order hashes. This collection audit does not claim that the full
paper reuse/pair audit or final contract seal has been completed.

Some old trainer fields and filenames contain `test`; the canonical dataset
metadata classifies these measurements as **validation**. Every included
bundle records `official_test_read=false`. No three-seed average or standard
deviation should be inferred from the seed-0 values.

### Result-location inventory

The read-only inventory was performed on 2026-09-11; exact roots and timestamps
are saved under [`provenance/inventory/`](../paper_ready_results/provenance/inventory/).

| Location | Discovery result |
|---|---|
| Local / `main`, current and sibling/base result roots | 2,787 manifests inspected; all 38 selected sources found; no additional seed-1/2 or bounded EP training manifests found |
| Akib through `akibscomputer` | 162 manifests inspected under `/home/filiposana/server_code/results` and `/home/filiposana/experiments`; no additional matching training |
| Trex | 146 manifests inspected under `/home/filip/server_code/results` and `/home/filip/experiments`; no additional matching training |
| Jean Zay | 348 manifests inspected under both `fmu` and `umg` scratch result roots; no new target cells; 31 included local result hashes also match discovered remote results |
| `nom-cool-1`, `fifi`, `loulou` | SSH succeeded, but the probed `/home/filip/server_code*/results` and `/home/filip/experiments` roots were absent; other possible locations were not searched |
| `riri` | Inventory unavailable: SSH host-key verification failed; no result discovery claim for this host |

The old `akib` alias failed with `No route to host`; the successful
`akibscomputer` inventory supersedes that failed attempt. Jean Zay's configured
`fswork/.../server_code/results` path was absent, so its existing `fsn1` result
roots were used. Inventories exclude archived source trees and do not prove
absence from arbitrary unrecorded directories or remote scheduler queues.

### Inclusion and exclusions

The original inclusion set contained 38 rows selected from the supplied matrix
by scientific identity. The continuing
[`collected_results.csv`](../paper_ready_results/collected_results.csv) also
includes newly completed and validated campaign runs. Old
pre-correction legacy scores, source-study SGD siblings, read/write-noise
runs, smokes, shortened runs, and other diagnostic conditions are excluded.
Corrected Conv1 legacy wide BPTT replaces the plan's retrieval placeholder;
corrected Conv2/Conv3 wide BPTT, Conv2/Conv3 clean EP, and Conv3 bounded BPTT
replace their historical legacy counterparts.

Supporting source-study analyses and logs are copied under
[`provenance/source_studies/`](../paper_ready_results/provenance/source_studies/).
They can discuss excluded sibling conditions and remain provenance, not
additional included runs. The collection also preserves exact parent configs,
the original launch plan/matrix, and governing references. Original scientific
configs retain their historical absolute paths; local copies and initializer
paths are mapped in the collection records without rewriting those manifests.

## Maintenance rule

After a launch or collection, update the affected row in `run_status.csv`
with the target/job, actual source and copied result, checksum, and scientific
state; reconcile `remaining_runs.csv`, the counts here, and the persistent
[`current_simulations.md`](current_simulations.md) row. Validate each new bundle
before adding its measurements to `collected_results.csv` and the validation
tables. Keep incomplete/failed runs and replacements visible. Record official
test results separately after sealing, and mark this ledger complete only
when the declared coverage and all paper-readiness gates are satisfied.

### Change log

- **2026-09-11:** Created this completion ledger and the `paper_ready_results`
  collection. Recovered the Conv1 legacy Adam retrieval item, copied and
  validated all 38 reuse candidates, preserved three initializers and source
  records, inventoried the reachable result roots, and enumerated the 178
  remaining full training runs. Final paper eligibility and official-test
  evaluation remain pending.
