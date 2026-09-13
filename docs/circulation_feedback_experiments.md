# Exploratory circulation calibration for asymmetric EqProp

13--14 September 2026. Exploratory research toward physical modifications for learning
with few adjoint probes. This study tests a candidate combination, not a claim
of a new established algorithm. It compares calibration of a dense controller
with calibration restricted to known physical feedback loops.

The proposed controller C is skew. During stochastic calibration it supplies
C(s-s0) and learns from the negative antisymmetric lagged state correlation.
During task nudging it supplies 2C(s-s0). The intended target is C=-K, which
keeps the original asymmetric free network while producing J+2C=J^T in the
nudged tangent dynamics. The actual task update remains centered contrastive
EqProp. No random adjoint probes are needed after calibration.

Calibration uses isotropic white noise, state observations, and local pairwise
time order. The simulator implements a physical additive-noise SDE using
stochastic Heun, separate from the deterministic equilibrium solver. It does
not provide force values, a Jacobian, known skew, or an adjoint to the plasticity
rule. Oracle quantities are permitted only in post-update diagnostic callbacks
and gradient audits; they never choose a stopping time or controller update.

Close prior art includes AsymEP's exact known-skew correction; July 2026
Spike-based Alignment Learning's noise-driven local symmetrization (including
its suggested combination with spiking EqProp); and stochastic-area inference.
The candidate distinction is a separate controller with cancellation during
calibration and doubled, anchored deployment during task nudging. Novelty of
the full combination is unestablished.

Exploratory monitoring contract: use a foreground one-BLAS-thread CPU process,
fresh directories under simulation_results/circulation_feedback_20260913.
The runner writes run.json with source/config/environment, status.json, run.log,
per-case calibration.csv and controller.npz, calibration_summary.json, and
summary.json. If training is selected, require metrics.csv and every declared
checkpoint, with final test evaluation only. Calibration progress is reported
roughly forty times per case; training progress every epoch. Expected runtime
is seconds for a 16-state smoke, minutes for wider/longer exploratory cases.
Preserve failed outputs and diagnose before retrying the same config.

Initial gate: run the scientific circulation tests and existing recurrent
EqProp integration tests. Then run a 16-state, one-seed calibration smoke with
duration 100, dt .02, temperature .05, 16 independent records, rate .1,
10-step controller updates, burn-in 10, averaging after 30, and two quick
training epochs for ordinary EP, known-skew correction and calibrated feedback.
This smoke checks operation and coverage; it is not a competitive result.

After that gate, compare longer calibration, timestep refinement, and width.
Useful evidence must include controller error, unseen-input gradient error,
input-weight gradient alignment, actual training, and total calibration time
and state reads. A continuous noise record is not free or dimension independent:
a dense skew controller stores n(n-1)/2 coefficients. The fixed K in this toy
makes once-only calibration unusually favorable; later experiments should test
drifting or state-dependent non-reciprocity and constrained controller support.

The 51-test gate passed. Smoke source 96b70775, exec session 47794, completed
one calibration and all three training trajectories in 0.6 seconds. All final
checkpoints replayed their validation/test metrics and charged three training
equilibrations per example. The calibrated controller had relative error 0.185;
two-epoch quick-data performance is an operational check only.

Next declared exploratory coverage: sizes 32, 64 and 256, seeds 0/1/2, each with
calibration duration 400, dt .02, 32 records, temperature .05, rate .1,
10-step updates, burn-in 10 and averaging after 50. Train ordinary EP,
known-skew AsymEP, circulation-calibrated AsymEP, exact adjoint and learned-MC4
for 15 epochs at common LR .1 and EMA .9 on the complete existing digits split.
This is 9 calibrations and 45 trajectories, with no optimizer search. Common
settings are for a controlled exploratory comparison, not best-achievable
performance claims. The learned-MC4 setting differs from its earlier selected
LR .3 and must be labeled accordingly.

Separate numerical refinement: size 64, seeds 0/1/2, identical duration and
physical update interval, dt .01 and 20-step updates, gradient audits only.
This tests whether the practical conclusion changes under timestep refinement;
independent Brownian paths mean it is not a paired deterministic error bound.
All calibration observation time and state-read counts remain explicit.

Launcher record: main_n32 session 1130, main_n64 session 77885, main_n256 session
19817 and refinement session 8856 all completed. They use source 2538bd84 and the
commands/configs recorded in their run.json. The combined
[main verification](../simulation_results/circulation_feedback_20260913/main_verified/verification.json)
confirms all nine main calibrations, 45 training checkpoint replays and the
three-seed n64 timestep refinement. Its separate
[report](../simulation_results/circulation_feedback_20260913/main_verified/report.md)
retains the unrestricted-skew results and calibration costs.

14 September extension: a known physical loop basis restricts the unknown skew
to four independently drawn gains. The wiring is generated independently of the
gains; no Jacobian-derived basis enters calibration. The new controller learns
four scalar coefficients from eight projected measurement channels. Noise still
excites all physical states. Compare against a dense controller on the identical
structured physical K, with the same duration, replicas and process-noise seed;
charge this reference calibration separately.

The combined 71-test gate passed, including ten projected-controller tests and
ten sign-routing architecture tests. A small all-method integration smoke passed
independent replay before this extension launched. Declared main
extension coverage after that gate: sizes 64 and 256, seeds 0/1/2, four loops,
duration 100, dt .02, eight independent records, temperature .05, rate .1,
updates every ten steps, burn-in 10, averaging after 30. Run the matched dense
calibration control for each of the six cases. Train ordinary EP, known-skew
AsymEP, projected-circulation AsymEP and learned-MC4 for 15 epochs, full digits
split, common LR .1 and EMA .9. Thus six structured calibrations, six additional
dense calibrations, and 24 training trajectories are required. This compares
methods within the same structured model; its physical prior differs from the
unrestricted-skew main grid. Expected duration is minutes per size. Use the same
foreground one-thread launcher and status/log/checkpoint monitoring contract.

The n16 one-epoch all-five-method integration passed independent replay at
/tmp/structured_circulation_review_20260914, including the same-K dense control,
initialization, frozen controller and phase accounting. Explicit n-channel noise
metadata was checked separately after that smoke. Extension source f734a7ff;
structured_n64 session 49145 and structured_n256 session 88136 launched with
the declared settings and --dense-control. Their run.json records exact commands;
status.json, run.log, per-case calibration_summary.json, metrics.csv and training
checkpoints are the monitored artifacts. Both passed the first progress check.

Additional bounded calibration audit: exchanged DC responses on the same four
known loops can update their gains without process-noise injection. Two paired
directions per loop measure the required scalar reciprocity defects; 60 updates
cost 960 perturbed equilibrations plus one free anchor, and 960 scalar readouts
plus eight anchor projections. The learner sees defects only. Two algebra/access
tests passed and a 16-state ten-update smoke completed (relative controller error
0.0135). Next audit is six cases (64/256 states, seeds 0/1/2), fixed 60 updates,
rate .2, amplitude .01, read noise 1e-5, no training. Save config/source status,
per-case calibration trajectory/controller, exact gradient audits, and terminal
case coverage under response_main. This tests calibration and gradient quality;
it does not inherit the stochastic controller's training results automatically.

The response_10steps audit is now complete for all six 64/256-state, three-seed
cases. The independently checked response_verified/verified.json covers twelve
calibrations across the ten- and sixty-update budgets. Ten updates give roughly
1.1--1.5% controller error while charging 160 perturbed equilibrations, one free
anchor equilibrium, 160 scalar response reads and eight anchor projections per
case. These remain calibration/gradient results until actual training is tested.

Declared DC-controller training extension: six trajectories, sizes 64 and 256
and seeds 0/1/2, using the previously measured ten-update controllers frozen
throughout training. Use the identical structured physical K, dataset split,
initialization and common LR .1 / EMA .9 as the existing structured comparisons.
Train for 15 epochs, with ordinary positive/negative cost nudges plus doubled
anchored feedback: three equilibrations per training example and no fresh random
adjoint probes. Calibration cost is charged once per trajectory. The
train_response_loop_feedback.py n64 quick one-epoch gate and checkpoint replay
passed before launch. Run response_train_n64 and response_train_n256 as separate
foreground one-thread shards; monitor their status.json, run.log, metrics and
checkpoints through semantic completion. Require all six final validation/test
replays and matched-control checks before reporting the training comparison as
complete. This extension supplies the missing training test for DC calibration;
its outcomes will not be inferred from the stochastic-calibration trajectories.

DC-training launcher record: response_train_n64 session 17498 and
response_train_n256 session 27224, source c023f21a. Their run.json files record
the exact commands and controller sources. Both were launched after the coverage
declaration above was recorded.

## Terminal structured and DC results

Both structured shards completed without retries or scientific changes:
structured_n64 in 300.75 seconds and structured_n256 in 1444.14 seconds. The
[combined verifier](../simulation_results/circulation_feedback_20260913/structured_verified/verification.json)
checked all six structured calibrations, six additional same-K dense calibrations,
24 final validation/test checkpoints, 384 epoch rows, and 48 recomputed gradient
audit cases. It regenerated wiring, gains and initial models independently;
checked fixed K, frozen controllers, shared data and initialization; and reconciled
calibration and training resource counts. The
[report and figures](../simulation_results/circulation_feedback_20260913/structured_verified/report.md)
were inspected after verification.

Test accuracy after 15 epochs, mean ± population standard deviation over seeds
0/1/2, on the same structured models and complete digits split:

| Method | 64 states | 256 states | Training equilibrations per example |
|---|---:|---:|---:|
| Ordinary contrastive EP | 12.50 ± 4.59% | 5.19 ± 1.02% | 3 |
| Known-skew AsymEP | 95.19 ± 0.35% | 95.19 ± 0.13% | 3 |
| Stochastic loop calibration, doubled deployment | 95.19 ± 0.35% | 95.28 ± 0.23% | 3 |
| Learned baseline plus MC4 | 95.09 ± 0.13% | 95.09 ± 0.47% | 9 |
| Ten-update DC calibration, doubled deployment | 95.28 ± 0.45% | 95.19 ± 0.13% | 3 |

The six DC-controller trajectories also completed. Their separate
[verification](../simulation_results/circulation_feedback_20260913/response_training_verified/verification.json)
replayed every DC checkpoint and checked its pairing with all 24 structured
control checkpoints; its
[report](../simulation_results/circulation_feedback_20260913/response_training_verified/report.md)
charges the 161 calibration equilibrations in addition to 51,705 task-training
equilibrations per trajectory. Thus the DC total is 51,866, compared with 155,115
task equilibrations for learned-MC4. DC calibration uses 168 scalar reads in
total: 160 response readings and eight anchor projections for four fixed unknown
loop gains. The final 73-test scientific gate passed. Across the unrestricted-skew main study,
structured study and DC extension, all 75 final training checkpoints have now
been verified; the scopes remain separate scientific comparisons.

At the matched stochastic calibration budget, relative controller error was
0.0404 ± 0.0065 at n64 and 0.0597 ± 0.0114 at n256. The unrestricted dense
controller on the *same structured K* had error 1.1592 ± 0.1200 and
5.5093 ± 0.5709 respectively. These dense calibrations are retained as negative
results and were not used for the structured training rows above. They test
the cost of fitting unnecessary controller degrees of freedom under this budget,
not an impossibility result for longer dense calibration.

On the held-out validation cohort, mean initial full-gradient cosine with the
exact gradient was 0.9996/0.9985 for doubled calibrated feedback at n64/n256;
input-weight gradient cosine was 0.9997/0.9989. Single-gain deployment gave
0.6209/0.7918 and input cosine 0.4005/0.4504. Ordinary EP gave full-gradient
cosine -0.1075/0.3645 and input cosine -0.5692/-0.5000. This is a gradient audit
of the gain switch, not a training comparison against single-gain deployment.

Stochastic resources per calibrated model, including burn-in and all eight
independent records:

| Resource | 64 states | 256 states |
|---|---:|---:|
| Learned loop gains | 4 | 4 |
| Supplied wiring coefficients, 2nL | 512 | 2,048 |
| Projection readout channels | 8 | 8 |
| Scalar projection reads | 352,064 | 352,064 |
| Process-noise injection channels | 64 | 256 |
| Scalar noise increments simulated | 2,816,000 | 11,264,000 |
| Total physical recording time, normalized units | 880 | 880 |

The six structured calibrations therefore consume 5,280 recording-time units
and 2,112,384 scalar projection reads. Their six matched dense controls add a
further 5,280 time units and 42,247,680 scalar reads, with 2,016/32,640 learned
coefficients at n64/n256. No conversion from recording time to equilibrium count
is assumed. The 24 structured training trajectories use 1,861,380 task
equilibrations altogether, separately from these calibration costs.

The positive result relies on an exact supplied four-loop basis and constant
unknown K, so the calibrated controller transfers across inputs and throughout
symmetric-weight learning. It does not demonstrate few-parameter correction for
arbitrary non-reciprocity or robustness to drifting/state-dependent skew. The
comparison uses one small dataset, a common exploratory optimizer setting, ideal
feedback wiring, zero stochastic-calibration read noise, and simulated process
noise; task phases have read noise 1e-5. Parameter updates and the spectral
stability projection are digital. Similar final accuracies are not evidence
that approximate gradients outperform exact ones. Hardware efficiency and
novelty of the proposed combination remain unestablished.
