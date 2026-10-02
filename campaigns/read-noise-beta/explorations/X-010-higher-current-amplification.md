# X-010 — Does increasing current amplification beyond ours improve noisy learning?

September 28, 2026. Explorer proposal requested by Filip. Scope is investigation
and planning; no calibration, replay or training was launched. A=voltage_amp and
B=current_amp here; use beta_inj for the nudging amplitude to avoid confusing it
with the existing configs' `injected_beta_B` field.

## Evidence and competing explanations

Compare legacy (A,B)=(4,.25), ours=(4,1), and the proposed (4,2).
The [relative-noise map](../series/006-relative-endpoint-noise/results/exp-013-relative-noise-advantage-map.md)
supports local ours-minus-legacy alignment advantages on all three architectures.
It does not establish monotonic improvement with B or predict training accuracy.
For example, Conv3 at D/F=1, eta=1e-4 has ours convolution cosines
−0.0196, 0.0075, 0.8136: a good last layer can hide unusable early gradients.

The clean controls favor legacy under their inherited scheme-specific rates:

| Architecture / horizon | Ours mean validation (%) | Legacy mean validation (%) |
|---|---:|---:|
| Conv1 / 10 epochs | 96.3333 | 96.4933 |
| Conv2 / 30 epochs | 98.0467 | 98.2400 |
| Conv3 / 30 epochs | 98.6067 | 98.8133 |

Each mean uses seeds 0/1/2. Sources: [exp018](../series/007-relative-noise-training/results/exp-018-conv1-controls.md)
and [exp016](../series/007-relative-noise-training/results/exp-016-noiseless-three-seed.md).
These support the observed clean ranking, not a causal claim that less feedback
always trains better. LR, finite-iteration dynamics and beta all matter.
The [30-epoch noisy comparison](../series/007-relative-noise-training/results/exp-015-epoch30-continuation.md)
is nonmonotonic: Conv3 ours-minus-legacy is +6.80 pp at eta=1e-4, +2.42 pp
at 3e-4, and −1.12 pp at 1e-3. Those runs reset Adam at epoch 10;
their contrast with uninterrupted clean training is descriptive.
These are existing reviewed notes, not new raw-bundle validation by this Explorer.

The interaction energy in [interaction.py](../../../model/resistive/interaction.py)
is proportional to (B/A)^p g (A v_pre − v_post)^2 for internal edges.
In normalized coordinates u_l=v_l/A^(l−1), adjacent edge coefficients differ
by AB. Inside an unchanged diode active region, the hidden-node balance has
the schematic form

    (G_in + AB G_out) u_l = W_in u_(l−1) + AB W_out u_(l+1).

Convolutions replace these products/sums with their corresponding spatial
operators. Thus AB=1,4,8 for legacy, ours and the proposal. Increasing B from
1 to 2 doubles normalized downstream loading AND feedback; it moves away from
the feed-forward limit. It does not simply double the backpropagated signal.
Changed free voltages, loading, active sets and finite-T/K convergence can all
affect training. Better hidden fractional displacement D/P could protect noisy
gradients, but the size or even sign of that change is unmeasured for B=2.

Prospective hypothesis: B=2 improves alignment preservation under relative noise,
and at some fixed eta its training gain exceeds any clean-accuracy penalty.
Alternative: improved cosine fails to compensate for degraded clean dynamics;
another is that apparent gains come mostly from LR selection. Do not label any
of these tested until the new evidence exists.

## Proposed matched contract

Use the existing MNIST centered frozen-current EqProp contract, not BPTT training:
55k/5k split, float64, batch 16/64, paired 20-output squared-error objective,
input gains 40/100/360, explicit unchanged diode dictionaries, weights [0,100],
zero frozen biases, Adam and T=K=4/6/8 for Conv1/2/3. Preserve initialized
weights and sample order within each depth/seed. No official-test access.
No new T/K, residual or optimizer qualification campaign is proposed.

Calibrate B=2 separately at seed0 to the existing output D/F targets 1/4/6,
using the accepted 576-example cohort, pooled centered half-difference RMS
divided by free-output RMS, 1% tolerance and at most three correction rounds.
Freeze beta thereafter across rates, epochs, noise levels and replication seeds.
For depth d, beta_inj = beta_internal (A/B)^d: the new multiplier is 2^d,
not ours' 4^d. Copying either beta value without matching physical displacement
would confound the comparison. Record both values and actual achieved D/F.
Seeds1/2 inherit seed0 beta; do not claim their D/F is exactly rematched.

The noise model remains v_read=v+eta*abs(v)*xi independently across endpoint
phases/nodes/layers, one eta across all noninput layers including readout,
absolute std=0 and no floor. Input, relaxation, force and validation remain clean.
Use paired standard-normal acquisitions, with the existing seed0 acquisition
seed 2026081601. Independent replication seeds must be frozen before execution.

## Learning-rate selection before noisy training

The user's final phrase, “probably a bit larger than ours amplification scheme
and lower”, is incomplete; its intended upper/lower comparator remains open.
The saved rates do not put legacy above ours: ours uses about 3x legacy's
convolution rates, and about 9x legacy's Conv3 readout rate.

Use a common scalar m on the complete ours weight-rate vector within each depth,
keeping all bias rates zero. Source vectors from
`configs/conv/relative_noise_training_20260925/`:

| Depth | Convolution rates, in layer order | Readout rate |
|---|---|---:|
| Conv1 | 8.66753e-4 | 1.09388e-4 |
| Conv2 | 2.60036e-3, 4.58651e-4 | 1.55136e-4 |
| Conv3 | 7.802510895760823e-3, 1.3830409123187907e-3, 9.79562021138172e-4 | 3.2928176432380754e-4 |

Proposed grid m={0.5,1,2,4}. This brackets the suggested increase and retains
a lower-rate control. There is no established B-to-optimal-Adam-rate formula.
Do not infer that doubling current amplification requires doubling LR.

1. Train B=2, seed0, eta=0 for 10 epochs at all four rates, on every depth:
   12 runs / 120 epochs. Rank by fixed epoch10 validation accuracy; break exact
   ties by lower validation loss, then smaller m. Retain failures as outcomes.
2. For Conv2/3, continue the top two finite candidates to epoch30, preserving
   full optimizer and RNG state: four continuations / 80 additional epochs.
   Select by fixed epoch30 accuracy, with the same tie rule. Conv1 selects at
   epoch10. This is a bounded screen, not proof of the global optimum.
3. Freeze one rate per depth, then run clean seeds1/2 at 10/30/30 epochs:
   six new runs / 140 epochs. Report seed0 as selection evidence and seeds1/2
   as checks without tuning on them. Reuse the matched clean legacy/ours records
   only after verifying numerical/config/initializer/optimizer-history identity.

Do not silently expand the grid if a boundary wins. Report an unresolved rate
optimum; any expansion is a new Explorer decision. Stop an individual run for
nonfinite values. If no candidate completes, stop that depth and diagnose;
do not lower LR or change beta/T/K invisibly. A clean deficit alone does not
reject a noise-robustness hypothesis.

This compares a newly tuned B=2 scheme against the currently selected legacy/ours
settings. It is not an equal-search-budget demonstration of fundamental scheme
superiority. A stronger claim would require matched retuning of the comparators.

## Noisy stage and interpretation

After clean selection, start all noisy cases from initialization using frozen
rates/betas and uninterrupted Adam. Use 10/30/30 epochs respectively.
Proposed lean first tranche: eta={1e-5,1e-4}, all three depths and all three
schemes, seed0: at most 18 runs / 420 epochs. These are existing noise levels
chosen retrospectively to sample a modest-noise regime and the observed Conv3
advantage. Reuse a prior case only if its full training history matches;
exp015's epoch10-reset cases are not matched uninterrupted controls.

The full follow-up grid, if requested, is the original exp014 grid
{1e-8,1e-7,1e-6,1e-5,1e-4,3e-4,1e-3} plus the current exp019 extension 5e-4.
This preserves “the same noise as the others” without introducing a different
noise model. The lean tranche is not full-grid validation. Fresh seed replication
of any promising noise cell precedes a generality claim; the initial seed0
comparison is exploratory. Baseline may remain a contextual saved reference;
the direct intervention keeps A=4 and varies B=.25,1,2.

Report fixed-horizon accuracy, training curves, failure coverage and paired
B=2-minus-ours/legacy gaps at each eta. Also report degradation relative to each
scheme's own clean run. Reduced degradation with lower absolute noisy accuracy
supports robustness only, not an accuracy advantage. Do not pool architectures
or select each scheme's best epoch independently.

For the requested mechanism question, add a bounded read-only initialization
replay at eta={0,1e-5,1e-4}, the same D/F targets and all three schemes:
9 scheme/depth contexts, 192 examples, three paired noisy draws per nonzero eta.
Measure every convolution and readout, minimum-layer cosine, clean EqProp versus
finite-K BPTT cosine, noisy versus clean EqProp cosine, relative gradient error
and hidden D/P. This is a scientific measurement, not a mandatory launch gate.
Clean controls separate read-noise preservation from finite-beta estimator bias.
Do not treat the BPTT reference as a qualified equilibrium gradient.

Decision: evidence for a useful B=2 tradeoff requires higher absolute noisy
accuracy than ours at a shared eta, with clean/noisy penalties and failures
visible. Improvement over both ours and legacy is stronger evidence. A cosine
gain without an accuracy gain does not meet the training claim. Per-depth
results may disagree. Proposed follow-up prioritization: a >=0.2 pp accuracy
gain at one frozen noise cell merits independent-seed confirmation; this is a
practical threshold proposed here, not a significance test or measured result.

## Budget and handoff

All numbers are proposed allowances, not launch authorization. Clean selection
and replication total at most 340 training epochs: 40 GPUh for selection,
30 GPUh for clean replication, including overhead/recovery. Mechanism replay:
3 GPUh. First noisy tranche: 100 GPUh. Total cap 173 GPUh if all phases are later
assigned. Initial screening is the first independently useful handoff; do not
queue all later phases merely because this exploration exists.

Existing clean baseline Conv2/3 six-run timing was 18.84 GPUh for 180 epochs;
on that rough scale clean work is tens of GPUh and the noisy tranche roughly
40–60 GPUh before contention. This is not measured B=2 throughput. Reforecast
from first completed epochs, retain caps and stop admission when a full case
cannot fit. Suggested per-phase wall limit: 72h from first admission. Prefer
eligible lab GPUs, Conv1/2 on 3090s and Conv3 on 5090s, under the current sharing
and fleet rules. Verify live availability only when execution is assigned.

Candidate result root: `results/conv123-current-amp2-20260928-v1/`; not created.
Before implementation, the Codifier should create a uniquely numbered experiment
record in this campaign, freeze the assigned phase and exact configs, check
generic A/B handling in the existing calibration/replay/preparation helpers,
and preserve all source identities and explicit diode parameters. Reuse
`experiments.exact_run`, existing calibration/replay methods and reporting;
do not silently map the new arm to the hardcoded `ours` numerical settings.
No jobs, GPU reservations, configs or measured B=2 conclusions exist from this
Explorer pass. The unfinished LR phrase remains a clarification, not a selected
numerical constraint.

## Execution decision — September 28

Filip authorized starting with one seed and fixed LR2x ours. This supersedes
the proposed LR grid and extra clean seeds. [Exp020](../series/007-relative-noise-training/experiments/exp-020-higher-current-training.md)
owns the executed scope, beta calibration, clean-then-noisy staging, budget and
monitoring. The incomplete earlier LR phrase no longer blocks a numeric choice.
