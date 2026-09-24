# Proposed revision of EqProp beta selection

**Superseded in scope:** Filip subsequently authorized execution and narrowed
the search to wide Conv1/2/3 baseline beta 100, 200, 300. See the
[active baseline audit](eqprop_baseline_beta_audit_20260914.md). The broader
grid and ours-scheme proposal below are retained as proposal history only.

Date: 2026-09-14. Status: **draft for the paused experiment review**.
This proposal does not change the frozen training configs or resume admissions.
Its first scope is the clean, wide-range Adam comparison discussed with Filip;
the same procedure can subsequently audit the bounded conditions separately.

## Why revise the sampling

The original wide-range sweep used a best-validation checkpoint and one batch
of 16. The later zero-bias gate used initialization and best validation, with
four batches of 16. At the chosen one-decade tier, that later gate passed
213/216 layer-batch comparisons: Conv3 baseline beta 100 failed one initial
C0 comparison (cosine .987983), and Conv3 ours beta 3 failed two (worst
cosine .939168). Another tenfold reduction passed all 216 gradient comparisons.
The trained Conv3 baseline also retained a separate free-state residual failure
at T=8. See the [gradient review](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-tk4-6-8-seed0-20260816-v1/review.md).

More batches test sensitivity to the sampled data. They do not erase those
known failures, establish stability throughout training, or repair an
under-relaxed free state. Preserve the historical four batches as mandatory
regression cases, and keep gradient fidelity, equilibrium, and training
stability as separate decisions.

## Proposed cohorts and controls

| Cohort | Size | Purpose |
|---|---:|---|
| Historical regression | 4 x 16 = 64 examples | Retain every previously tested batch, including failures |
| Selection | 32 x 16 = 512 new examples | Compare the declared beta candidates |
| Confirmation | 32 x 16 = 512 other examples | Check the frozen candidate without further tuning |

All three cohorts come from the existing deterministic 5,000-example MNIST
validation partition. Keep the training split and batch size unchanged. Exclude
the historical 64 source indices, permute the remaining indices using a
recorded NumPy PCG64 seed `2026091401`, then assign the first 512 to selection
and the next 512 to confirmation. Freeze the explicit source-index lists,
batch boundaries, labels/class counts, and payload hashes before any new
gradient measurement. Use the same examples and grouping across schemes,
seeds, checkpoints, and candidate betas. Preserve each architecture's own
preprocessing and input gain.

The confirmation set is held aside for this gradient-selection round. It is
not a previously untouched accuracy test set: the validation partition already
contributed to checkpoint and hyperparameter selection. Official MNIST test
access remains disabled. If confirmation fails, record the failure and hold
the group. Using its measurements to change beta makes that cohort selection
data; any revised choice needs a newly declared, unused confirmation cohort.

## Checkpoints and seeds

Select candidates using model seed 0 at verified initialization and the matched
BPTT best-validation checkpoint. Confirm the chosen beta on those two roles for
all three existing model seeds, 0/1/2; also retain the historical four-batch
checks for those seeds. Use one beta per architecture/scheme, not one per seed.
The existing seeds provide a robustness audit, not fresh training repetitions.

The local inventory contains all 27 wide BPTT bundles, each with best and final
PT/NPZ checkpoints. No separate intermediate-epoch checkpoints were found in
those bundles. Initialization must load the original asset or reproduce its
recorded tensor hash; it must never be inferred from a trained checkpoint.
Final checkpoints can provide a separately labeled diagnostic where their
tensors differ from best. Do not invent early-epoch gradients from metric logs.
Any future required training pilot should additionally save epoch 1 and the
midpoint so that early and intermediate states become directly auditable.

Replay uses centered frozen-current EqProp and the same-T/K BPTT reference,
identical starting parameters and post-T states, exact-zero biases, float64,
zero endpoint read noise, and the existing loss and parameter order. Hash
source files before and after; verify that parameters remain unchanged; take
no optimizer step. The [matching protocol](conv_paper_one_seed_bptt_eqprop_protocol.md)
continues to govern the controls.

## Candidate grid and decision

For the first wide audit, let B0 be the current injected beta. Use the five
declared points `B0 x {0.1, 0.3, 1, 3, 10}`. This retains the existing setting
and tests both sides with finer spacing than the original decade-only margin.
The baseline grid is therefore `10, 30, 100, 300, 1000` for each architecture;
Conv3 ours uses `.3, .9, 3, 9, 30`. Record injected and base beta separately.
The historical sweep remains provenance, not new confirmation evidence.

Nominate the largest tested candidate that passes **every** selection and
historical layer-batch check at both checkpoint roles:

- EqProp/BPTT cosine >= .99;
- symmetric relative gradient-norm difference <= .10, using the existing
  implementation unchanged;
- finite gradients and phases; zero/undefined-reference comparisons remain
  unresolved rather than being silently omitted.

Then require the same gradient criteria on the new confirmation cohort and
the other seeds. Report per-layer minimum, fifth percentile, median, maximum
norm mismatch, and failure counts with their actual denominators. Include
gradient L2/RMS and zero fraction at `abs(g) <= 1e-12`, and hidden/output
nudged-state displacement in absolute and free-state-relative units. A mean
cosine above .99 cannot substitute for the declared all-row gate. The larger
sample can expose additional failures; thresholds stay fixed.

Keep the projected-KKT residual p90 < .01 criterion, including the free-state
check, as a separate equilibrium qualification using the existing evaluator.
A beta-independent residual failure holds equilibrium qualification even if
the gradients agree. Changing T/K requires a separate matched BPTT/EqProp
operating-point revision and its dependent checks.

If the upper grid edge passes, report an open upper edge and a tested operating
point; do not call it the true maximum. If none pass, report an unresolved
group. No automatic grid expansion or extra tenfold reduction is part of this
draft. A changed beta still requires a complete seed-0 training pilot with
finite completion and final validation drop strictly below 5 percentage points
before seeds 1/2 are repeated. Passing replay alone does not qualify training.

## Order, compute, and impact on completed results

Start with **wide Conv3 baseline and ours**, where failures are already known.
After their audit, apply the same frozen procedure to the other seven wide
architecture/scheme groups. A later bounded audit must use its own initializers,
current matched T/K and trained references, with one beta common to all three
conductance ceilings. Never transfer a wide-range beta to the bounded grid.

When work resumes, prefer the local RTX 3090 for the first diagnostic surface,
subject to a fresh capacity check. Time a same-runner semantic smoke before
admitting a larger wave. Proposed first diagnostic reservation: **at most two
GPU-hours**, including preparation smokes and failed checks, charged to the
existing 300-GPU-hour campaign ceiling. No GPU time is reserved by this draft.
Selection for the two first groups has at most 720 checkpoint-batch-beta
replays (`2 groups x 5 betas x 2 roles x 36 batches`); confirmation adds at
most 432 at the selected betas (`2 x 3 seeds x 2 roles x 36 batches`). These
are read-only measurements, not full training runs. Report measured throughput
and the projected duration before admission; if the work cannot fit the cap,
declare a smaller partial wave instead of silently extending the reservation.
Cost and coverage for the broader audit must be recorded after this timing.

The existing replay code already accepts a larger example count but builds
only a prefix of validation order. Before execution, extend that shared helper
to accept explicit source-index lists, validate disjoint cohorts and preserved
batch order, and replace its hard-coded four-batch report text with the actual
counts. Keep the historical path reproducible and use ordinary readable configs;
no separate launcher or experiment control framework is needed.

Intended result root, only after its persistent planned row is added to
`current_simulations.md`: `results/eqprop-beta-selection-audit-20260914-v1/`.
Publish the collected layer/batch CSVs, cohort identities, chosen/tested beta
table, and a concise report into `paper_ready_results/` after validation.

Preserve all existing training bundles and their original qualifications.
Additional checks at an unchanged beta do not by themselves require retraining.
If beta changes, affected EqProp results belong to the old beta contract;
matching BPTT results can be reused only if their complete contract is unchanged.
Changing T/K affects both algorithms. This draft authorizes no retrospective
relabeling, new training, or official-test evaluation; admissions remain paused
under Filip's review instruction.
