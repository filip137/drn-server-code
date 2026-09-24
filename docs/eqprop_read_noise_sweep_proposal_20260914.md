# Single-seed EqProp read-noise sweep proposal

**September 14 overnight update:** Filip selected only `1e-5, 3e-5, 1e-4,
3e-4, 5e-4`, with no new zero-noise or `1e-3` runs. He authorized scheduling
three runs on each free RTX 5090, plus Nom as useful, ending before September
15 at 08:00 CEST. The earlier grid and paused-launch proposal below are
historical; execution follows the [overnight plan](eqprop_read_noise_overnight_plan_20260914.md).

Date: 2026-09-14. Status: **planning only; new admissions remain paused**.

Filip requests one seed per noise level, starting with ours and legacy while
baseline beta remains under review. The working first scope is wide-range
Conv1/Conv2/Conv3, Adam, model/shuffle/split seed 0, and the current corrected
physical-KCL, zero-bias EqProp contract. Wide versus bounded coverage was asked
as an optional clarification; wide first is the proposed assumption, not a
recorded user decision.

## Recommended noise values

Use the same absolute endpoint-voltage standard deviations for both schemes:

```text
sigma = [0, 1e-5, 3e-5, 1e-4, 3e-4, 5e-4, 1e-3]
```

| Values | Purpose |
|---|---|
| 0 | Matched clean control |
| 1e-5, 3e-5 | Resolve the onset of degradation |
| 1e-4, 3e-4 | Sample the transition region |
| 5e-4 | Connect to the existing read-noise experiments |
| 1e-3 | Test beyond the previously trained range |

Run the coarse subset `0, 1e-5, 1e-4, 5e-4, 1e-3` first, then fill in
`3e-5, 3e-4` to complete the declared curve. This gives 24 nonzero-noise
trainings initially, then 12 more: **36 noisy trainings and six clean controls**
for three architectures and two schemes. Existing clean runs are reuse
candidates if beta and the complete scientific contract stay unchanged;
36 is the noisy-cell count before any eligible historical noisy-run reuse.

Declare at most one optional range extension before launching it. If a curve
already loses at least 1 percentage point of final validation accuracy at
`1e-5`, add `1e-6` as a lower anchor. If it loses less than 1 point at `1e-3`,
extend upward to `3e-3` and `1e-2`. Apply an extension to both schemes of the
affected architecture to preserve matching. Those additional cells require
their own recorded compute reservation; they are outside the 36-cell core.
Do not expand repeatedly until a preferred outcome appears.

## Why this range

Historical Conv2 experiments covered `1e-5, 1e-4, 3e-4, 5e-4`, showing
increasing loss with noise, especially for legacy and smaller beta. They used
older biases/T/K and support the choice of range rather than current reuse.

At `5e-4`, the zero-bias Conv1/Conv3 experiment measured final clean-relative
changes of -.06 pp for Conv1 ours and -1.82 pp for Conv3 ours. Historical
Conv3 legacy lost 11.38 pp. That legacy result predates the physical-KCL
correction and must not be treated as a current-contract noise measurement or
reused without a valid scientific audit. The weak Conv1 response motivates
allowing a higher-noise extension; the Conv3 response motivates sampling
below `5e-4`. See the [measured table](../results/perfectdiode-conv13-zero-bias-adam-centered-float64-eqprop-one-decade-sigma-5em4-ordinary-mnist-10-30ep-seed0-20260817-v1/analysis/summary.csv)
and [noise study](beta_study.md#beta-and-noise-conditions).

## Fixed controls

The existing model is `V_read = V_equilibrium + sigma * epsilon`, with
independent standard-normal draws for the positive/negative phases, layers,
elements and training updates. Noise is added to copied non-input endpoint
voltages used for the local EqProp gradient. Relaxation, inputs and validation
remain noiseless. Sigma is in simulator voltage units, not a percentage or
a hardware-calibrated voltage specification. Keep this identical model for
every noise level; see [the implementation](../training/sgd.py).

Use one fixed dedicated noise seed, proposed `2026081601` for continuity.
For a given architecture, match initialization, minibatch order and the
underlying standard-normal draws across schemes and nonzero sigmas, scaling
only by sigma. The two phases must remain independent within each run. Verify
matching draws rather than assuming seed equality guarantees them across
different environments. Preserve the 10/30/30 epoch budgets, exact named
learning rates, T/K=4/4,6/6,8/8, float64 EqProp and zero biases. Every new
training starts from its original initializer with fresh optimizer state.

Freeze one clean-qualified beta per architecture/scheme before training the
curve; keep it constant across sigma. The current injected anchors are
Conv1 ours/legacy `30/3`, Conv2 `10/.03`, Conv3 `3/.001`. The proposed
[larger-cohort beta audit](eqprop_beta_selection_revision_proposal_20260914.md)
still matters for Conv3 ours: its current beta has known clean-gradient
failures. Resolve the intended clean-beta contract for ours/legacy before
their production sweep. Baseline work need not be completed for that decision.
If beta changes, requalify training and use the matching clean control; do not
retune beta separately at noisy points or describe a scheme/beta comparison
as a beta-independent amplification effect.

Noisy gradients are allowed to lose cosine fidelity: that degradation is a
measurement, not a reason to omit the noisy condition. Record per-layer
gradient direction/norm diagnostics separately from training accuracy and
numerical stability. Preserve non-finite endpoints as failed conditions.
Report best and final noiseless validation accuracy and the paired loss from
the same seed's clean run; keep the maximum-validation checkpoint rule.
One model/noise seed gives descriptive curves, without seed-variance error
bars. Official test access remains disabled during sweep design and selection.

## Scope and execution handoff

Start with Conv3 ours/legacy because earlier evidence shows a useful noise
response there, followed by Conv2 and Conv1. The existing local/Akib/Nom
preferences, one weekday campaign RTX 5090 limit, and maximum four concurrent
Jean Zay GPUs apply when execution is resumed. Host capacity and runtime must
be checked before any allocation; no live availability or runtime estimate is
claimed here.

Adding all three bounded conductance ceilings at the same grid would add
108 noisy cells, giving 144 with wide range and 24 clean controls. A bounded
sweep should first check the useful sigma range with saved-checkpoint replays;
its different betas and state scales can move the transition substantially.
Keep wide and bounded curves separate.

Read noise was explicitly excluded from the clean three-table launch budget.
This proposal reserves no compute. Before launch, record measured timing,
target, a finite noise-study budget and expected duration, frozen configs,
the exact reuse decisions, and the persistent planned result-directory row.
Keep the clean campaign's accounting separate. Intended future result root:
`results/eqprop-read-noise-seed0-ours-legacy-20260914-v1/`, not created by this
proposal. Collected, validated curves and their source references should be
published under `paper_ready_results/`.
