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
