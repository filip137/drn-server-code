# Baseline read-noise preparation: increase free-phase T first

Date: 2026-09-16. Status: **completed and analyzed; mixed outcome**.

All four new cases pass equilibrium but retain the same seven initialization
gradient failures: worst cosine .967393 and norm mismatch .184260. Increasing
T alone does not qualify beta100. No joint passing T was selected; conditional
three-seed confirmation was correctly omitted and its fresh cohort remains
unmeasured. The four formal bundles and smoke validate;15 focused tests pass.
Driver1218622 exited zero after597.733 seconds, charging **.166037/2 GPU-hours**.
Local GPU is idle. [Full results](../paper_ready_results/baseline_wide_t_audit_20260916.md).

The design below records the completed T-only study. The proposed lower-beta
sweep at T12/K8 was superseded by Filip's subsequent instruction to keep
**T=K=8** and [sweep beta at that fixed setting](eqprop_conv3_baseline_beta_tk8_plan_20260916.md).
The exact
pre-execution plan bytes are retained in the result root as `plan.frozen.md`,
matching the hash recorded before execution.

Filip identified baseline EqProp read-noise training as the next major result,
then directed: increase T first and see whether insufficient relaxation caused
the wide Conv3 baseline qualification failures. This first study holds beta
and K fixed; it does not launch noisy or clean training.

## Scientific question and controls

At wide [0,100] weights, does increasing only free-phase T fix Conv3 baseline
gradient fidelity and equilibrium at injected/base beta **100**, **K=8**?
Replay T **12,16,24,32** and retain the exact T8 selection result from the
September14 audit as the comparison. Complete all four new selection cases.
These are diagnostic replays of checkpoints originally trained at T8/K8;
they do not become checkpoints trained at the diagnostic T.

Use the unchanged September14 frozen analyzer and scientific runtime,
float64 centered frozen-current EqProp versus same-T/K BPTT from identical
post-T states, exact-zero biases, original initializers and best-validation
BPTT checkpoints, loss, preprocessing, input gain and learning-rate contracts.
No optimizer step, endpoint read noise or official-test access is permitted.
Report initialization and trained checkpoint measurements separately.

Every weight-layer/batch must pass cosine >=.99 and symmetric norm delta
<=.10. Separately require every endpoint projected-KKT residual p90 <.01,
including the common free endpoint. Undefined or nonfinite comparisons do
not pass. Equilibrium agreement and direct gradient fidelity must both pass.

## Cohorts, decision and coverage

Selection reuses exactly the prior 36 batches of16 (576 examples): four
historical batches and32 selection batches, on both checkpoint roles at seed0.
Match their source indices, grouping and input hashes to the T8 reference.
After all four cases, freeze the **smallest tested T** passing both gates.
Confirm only that T, beta100/K8, on existing seeds0/1/2. No automatic fallback
to another T or beta after confirmation failure.

Before measurements, prepare a new confirmation cohort: retain the original
64 examples, exclude all1,088 examples in the September14 cohort partition
from the remaining validation pool, permute with NumPy PCG64 seed2026091601,
and take512 fresh examples. Thus confirmation has36 batches of16, including
four historical regression batches and32 new batches. Its fresh part is
disjoint from both prior audit cohorts. This is a reserved gradient-check
cohort; the validation partition previously informed checkpoint selection.

Four selection cases require288 checkpoint/batch replays and1,152 layer
comparisons. A qualifying T adds216 replays and864 comparisons for three-seed
confirmation. If no T passes, confirmation is correctly omitted. Preserve
all failures and report no qualified beta100/T setting within the tested grid.

## Execution and accounting

- Target: local/Main RTX3090, queried idle (132MiB,0%, no compute process).
  Preallocation inventory: Akib RTX3080, Nom RTX3090 and Trex RTX5090 idle;
  Fifi and Loulou occupied by unrelated work; Riri host-key verification
  fails and Jean Zay times out during SSH banner exchange. Do not disturb
  occupied hosts or bypass host verification.
- Budget: **two physical GPU-hours**, a separate baseline-noise preparation
  allowance, not a reservation against the paused clean training campaign.
  Expect20–40 minutes from the prior Conv3 replay timings. One sequential
  process on the local GPU, including a same-runner semantic smoke.
- Driver deadline:7,200 seconds including smoke and checks; each formal case
  has at most900 seconds. Charge actual elapsed GPU allocation, including
  failed attempts; do not extend the cap silently.
- Result root: `results/eqprop-baseline-wide-t-relaxation-20260916-v1/`.
  Configs: `configs/conv/eqprop_baseline_wide_t_20260916_v1/`.
  Runner: `python -m experiments.run_baseline_wide_t_audit prepare`, then
  `python -m experiments.run_baseline_wide_t_audit run` in the py312 environment.
  `execution.json`, per-case logs and canonical bundles track real progress.
  Keep the task active through local validation and analysis; check artifacts
  at least every30 minutes and report failures promptly.
- Publish the measured CSV, figure and interpreted report under
  `paper_ready_results/`, and curate the conclusion in the experimental
  manifest. Preserve original training and beta-audit artifacts unchanged.

## Consequences for the noise experiment

A passing replay setting is a candidate, not full-training qualification.
Changing T requires its dependent operating-point/LR checks and a matching
clean full-training control before interpreting noise-relative accuracy.
Do not reuse a T8 clean accuracy as the clean control for a longer-T curve.
Any BPTT/EqProp algorithm pair must share T/K; comparing longer-T baseline
with the existing ours/legacy T8 curves also introduces a compute difference.

After clean qualification, freeze one beta per architecture across all five
existing nonzero sigma values (1e-5,3e-5,1e-4,3e-4,5e-4). Select beta without
examining noisy training accuracy. Conv2 beta100 already passed the prior
three-seed audit; Conv1 beta100 still needs a separately declared confirmation.
This T-only study does not resolve those future training admissions or the
existing Conv3-ours beta qualification exception.

[Previous baseline audit](../paper_ready_results/baseline_beta_audit_20260914.md) ·
[Completed ours/legacy noise results](../paper_ready_results/read_noise_sweep_results.md) ·
[Paper-scope discussion](paper_ready_results_manifest.md#baseline-qualification-caveat-and-possible-bptt-only-scope-2026-09-16).
