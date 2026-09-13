# Exploratory circulation calibration for asymmetric EqProp

13 September 2026. Active research toward physical modifications for learning
with few adjoint probes. This study tests a candidate combination, not a claim
of a new established algorithm. The preceding turn made progress by documenting
the constant Jacobian-homeostasis penalty in the fixed-skew toy; the present
work changes authoritative code and gathers new physical-dynamics evidence.

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

Launcher record: main_n32 session 1130 and main_n64 session 77885 have completed;
main_n256 session 19817 is active, with two calibrations and nine trajectories
finished at its 677-second heartbeat. Refinement session 8856 completed. All use
source 2538bd84 and the commands/configs recorded in their run.json. Final
coverage and checkpoint replay remain pending for the combined main grid.

14 September extension: a known physical loop basis restricts the unknown skew
to four independently drawn gains. The wiring is generated independently of the
gains; no Jacobian-derived basis enters calibration. The new controller learns
four scalar coefficients from eight projected measurement channels. Noise still
excites all physical states. Compare against a dense controller on the identical
structured physical K, with the same duration, replicas and process-noise seed;
charge this reference calibration separately.

The combined 71-test gate passed, including ten projected-controller tests and
ten sign-routing architecture tests. A small all-method integration smoke is
being independently replayed before launching this extension. Declared main
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
