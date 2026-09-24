# Baseline beta audit with larger validation cohorts

Date: 2026-09-14. Status: **completed and analyzed; mixed outcome**.
Filip authorized implementing the larger-cohort beta experiment and narrowed
the scope to baseline beta **100, 200, 300**, leaving the other chosen rates
unchanged. This replaces the broader candidate grid and initial ours/baseline
scope in the [earlier proposal](eqprop_beta_selection_revision_proposal_20260914.md).
Training admissions remain paused; this authorization covers read-only beta
selection and confirmation.

## Completed outcome

All 9 selection cases and all 6 conditionally required confirmation cases
complete and validate: 1,080 checkpoint/batch replays and 3,024 layer
comparisons. Conv2 beta 100 passes selection and confirmation on all three
seeds. Conv1 beta 300 passes seed-0 selection but fails three trained-checkpoint
C0 comparisons on confirmation seed 2 (worst cosine .976462); it is not
promoted. Conv1 beta 100/200 passed seed-0 selection but did not receive a
subsequent three-seed confirmation. Conv3 beta 100/200/300 all fail the
initialization gradient gate and retain the separate T=8 free-state residual
limitation. No training beta or LR was changed.

All 72 original baseline beta-100 comparisons reproduce exactly in cosine,
norm mismatch and input payload. The new failures therefore reflect broader
cohort/seed coverage. Cost settled at **.279801 GPU-hours** (about 16.8 minutes
of charged run time), within the two-hour reservation. Main is idle and the
driver exited zero. [Full measurements and figures](../paper_ready_results/baseline_beta_audit_20260914.md).

## Frozen scientific cases

| Architecture | Scheme | Injected/base beta candidates | T/K | Selection seed |
|---|---|---|---|---|
| Conv1 | baseline | 100, 200, 300 | 4/4 | 0 |
| Conv2 | baseline | 100, 200, 300 | 6/6 | 0 |
| Conv3 | baseline | 100, 200, 300 | 8/8 | 0 |

These are the clean wide-range [0,100] Adam contracts, with their original
Kaiming initializers, exact-zero biases, unchanged learning rates, input gains,
loss and preprocessing. Ours, legacy and bounded beta choices are unchanged.
No optimizer is constructed and no official MNIST test data are read.

Replay centered frozen-current EqProp and the exact same-T/K BPTT reference
in float64 at the hash-verified initializer and matched BPTT best checkpoint.
Each pair starts from identical parameters and the same post-T free state.
The existing scientific runtime is the preserved completion-campaign
`source/` snapshot; the four modified/required analyzer files are frozen under
the new study's `source/experiments/` with recorded hashes.

## Cohorts and qualification

Retain the original four batches of 16 exactly. Exclude their 64 source indices
from the validation pool, permute the remaining indices with NumPy PCG64 seed
2026091401, and assign 512 examples to selection and the next 512 to
confirmation. The new pools are disjoint. Each phase replays the four original
batches plus its 32 new batches: 576 examples, 36 batches, two checkpoint roles,
72 replays per case. All source indices, batch boundaries, payload hashes and
class counts are frozen before gradient measurements. The two phases share
only the original regression cohort.

The confirmation cohort is reserved for this beta-selection round, not an
untouched accuracy test set; the validation partition was already used to
select checkpoints. Use the same source examples across architectures and
seeds, with each architecture's original preprocessing.

Every weight-layer/batch comparison must have cosine >= .99 and symmetric
relative norm difference <= .10, with finite phases and gradients. Preserve
undefined references as unresolved. Report projected-KKT residual p90 < .01
separately and require it for full equilibrium qualification. More batches do
not remove the known Conv3 beta-100 initialization failure or the T=8 trained
free-state residual caveat.

Complete all nine selection cases before reading confirmation gradients. For
each architecture, freeze the largest candidate passing every selection
gradient check, including the original batches. Confirm that single beta on
seeds 0/1/2 using the other 32 batches plus the original four. If no candidate
passes, the architecture remains unresolved with no confirmation or expanded
search. If confirmation fails, hold the choice; do not tune on its results.
An upper-edge pass at 300 is a tested choice, not a measured true maximum.

Selection coverage is 648 checkpoint/batch replays and 1,944 layer comparisons.
Confirmation adds at most 648 replays and 1,944 comparisons, depending on which
architectures qualify. All completed/failed comparisons are retained.
Report per-layer minimum, fifth percentile, median, norm mismatch and failure
counts, with separate historical/new-cohort and initial/trained summaries.

## Execution and checks

- Target: local/Main RTX 3090, verified idle before the first smoke. The remote
  inventory was also checked: Akib, Nom, Trex, Fifi and Loulou had idle GPUs;
  Riri failed host-key verification and Jean Zay timed out. Neither is used.
- Budget: two GPU-hours reserved as `baseline_beta_cohort_audit_20260914` in
  the existing 300-hour campaign ledger. Commitment after admission is
  273.418304 GPU-hours. Training reservations are retained pending accounting.
- Estimated duration before production: 30-60 minutes from the first smoke;
  refine using full-case throughput. A 10-minute first-case limit plus a
  90-minute remaining-wave limit and bounded smokes stay within two hours.
- Results: [study directory](../results/eqprop-beta-selection-audit-20260914-v1/).
  Each replay case writes its normal manifest/status/metrics/result bundle;
  `execution.json` records progress and `decisions/` freezes selection before
  confirmation. The agent monitors PID/GPU/artifact progress to completion.
- Preparation: `python -m experiments.prepare_beta_cohort_audit`.
- Remaining wave: `python -m experiments.run_beta_cohort_audit`, using the
  py312 environment and standard OpenMP environment limits. This direct loop
  runs the existing scientific replay command unchanged for each frozen config.
- Validation: 15 focused cohort/order/isolation and iteration-contract tests
  passed. The real MNIST historical four-batch hashes reproduce exactly.
  The corrected Conv3 GPU smoke validates canonically and passes its gradient
  checks while retaining the known residual failure.

The first smoke stopped before numerical execution because an inherited runtime
path pointed at the working checkout and no longer matched its frozen hash map.
Its log and original configs remain preserved. Version-2 configs point to the
original frozen runtime; all its expected hashes match. No model dynamics or
selection threshold changed during that correction.

Additional checks at an unchanged beta do not themselves require retraining.
A changed beta requires fresh EqProp training qualification before use in a
new training comparison. Matching BPTT results can be reused only if their
complete scientific contract remains unchanged. Changing T/K is outside this
audit and would affect both algorithms.
