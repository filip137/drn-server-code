# CIFAR L8 BN, Adam-step and gain follow-up, September 23 night

Filip authorized the proposed tests on all available RTX 5090s, ending by
September 24, 2026 at 08:00 Europe/Paris (06:00 UTC). These are exploratory
seed0 validation experiments, not official CIFAR test evaluations.

The scientific specifications are in the
[BN/gain proposal](cifar_l8_bn_conclusions_and_next_experiments_20260923.md)
and [Adam-step proposal](cifar_l8_gradient_scale_interpretation_20260923.md).
The result directory was registered as planned before creation:
`results/cifar10-l8-bn-gain-followup-seed0-20260923-v1/`.

| Case | Target | Completion criterion | Planning allowance |
|---|---|---|---|
| Voltage-normalized legacy continuation | Fifi | Restore epoch10 full state; train toward50, inspect30; original50-epoch cosine schedule | 10 GPUh; about8.4h exclusive for40epochs |
| Legacy frozen-affine BN, fixed conv gains100 | Trex | Ten epochs from shared original initializer; native final solver audit | Gain pair5 GPUh; about2.1h each exclusive |
| Legacy frozen-affine BN, log conv gains | Loulou | Ten epochs, gains100*exp(theta), theta0, gain Adam LR1e-3; final solver audit | Shared with fixed-gain allowance |
| Read-only Adam-step probes | Riri | Three original epoch30 checkpoints, four identical gradient batches each, step multipliers0/.5/1/2, same-batch and disjoint training probes | Diagnostics together1 GPUh |
| Read-only BN-stat recalibration | Riri | Original baseline/ours/legacy at10/50 and normalized legacy10; 4096 training examples, sequential BN1→3 moments; full5000 validation before/after | Shared diagnostic allowance |

The total cap is16 GPUh, including smokes and final audits. Sharing can increase
elapsed compute use; record measured throughput and any unfinished coverage.
At21:41UTC, before production admission, the gain pair was allowed up to6.4h
combined (3.2h per worker) by reallocating continuation time that cannot fit
before08:00. The continuation has at most about8.2h remaining, diagnostics
retain1h, and approximately0.4h remains for checks. The16h total is unchanged;
the table's original sub-allowances are planning estimates, not extra compute.
The deadline overrides the desired epoch50 endpoint. Keep the scientific
epoch target and schedule intact and pause at a completed epoch checkpoint
when insufficient time remains. Retain model, optimizer, scheduler, BN, gains,
RNG and deterministic sample-order state, with no success result for an
unfinished training case. Detached runtime limits remain effective after SSH
disconnect. Reserve time for checkpointing and shutdown; never extend GPU work
past08:00 to complete a scientific endpoint.

Live admission at21:36–21:39UTC found Fifi idle and Trex/Loulou/Riri sharing
Ben-owned mumax jobs. All four are RTX5090,32607MiB; at least25GiB was free on
each, versus the prior measured roughly12GiB training requirement. Sharing is
authorized by AGENTS.md; preserve Ben's jobs, launchers and MPS settings.
All four environments report Torch2.11.0+cu128. Local RTX3090 is reserved only
for sequential short CUDA smokes; Akib, nom-cool-1 and the configured Jean Zay
queue were also checked. No training is assigned to CPU.

Riri's clock is about two hours behind the controller; measure the deadline
using controller-derived remaining duration and monotonic runtime. Do not
change host clocks. Other hosts also have small clock offsets.

All training keeps batch32/eval16, the deterministic45000/5000 split,
crop/flip augmentation, cross-entropy, explicit perfect-diode settings,
conductance bounds[1e-7,10], selected legacy conductance/head rates and T/K6/6/4.
Normalized legacy retains its output divisors16/16/4 and trainable affine BN.
The gain pair fixes BN gamma1/beta0 while updating running statistics; only
the three convolution-block gain policies differ. The classifier retains its
original trainable softplus gain and5e-5 Adam LR in both arms. No reference
training or additional LR sweep is included.

Diagnostics operate on disposable copies and preserve original checkpoint
bytes. Recalibration fits training-only moments without augmentation or
validation-label use; each later BN sees preceding recalibrated BN layers.
Adam probes restore original BN buffers for evaluation and derive the next
full-parameter Adam step from each checkpoint's saved moments and current LR.
Scale the nominal bound-projected direction by each multiplier and project
again; multiplier1 is the actual next projected update. The fixed gradient
cohort is the first128 saved training indices, and the disjoint probe cohort
is the next128. Epoch10/50 Adam probes are optional only after primary
diagnostic coverage and within its allowance.

At22:04UTC the primary diagnostics completed and validated locally: all three
epoch30 Adam cases and all seven BN recalibrations,0.2591GPUh including smokes.
Recalibration does not close legacy's epoch50 accuracy gap, and the gap arises
after the already-probed epoch30. Therefore the predeclared optional Adam
extension is admitted on Riri for all three schemes at10 and50, using the
identical fixed cohorts, saved moments, modes and multipliers. Two additional
bundles `diagnostics/adam_e10` and `diagnostics/adam_e50` have a combined600s
operational ceiling including the new-selector smoke, within the original
one-hour diagnostic and16-hour total allowances. No training rates change.

Verify each launch immediately and within a few minutes, then inspect GPU,
process, log and artifact progress at least every30minutes. Collect completed
remote bundles locally, validate them with `experiments.reporting`, reconcile
partial/failed/unstarted cases, and record the morning scientific conclusions
in `docs/experimental_manifest.md`. Operational source, commands, handles and
receipts live under the study directory. One execution owner controls each case.

If the continuation pauses before50, retain its actual completed epoch as
partial evidence. Its normal final solver audit lies after the full training
loop and will not have run. When the remaining overnight window permits,
apply the existing read-only checkpoint replay/audit to that paused checkpoint
without training or changing its state, within the unused diagnostic allowance
and08:00 stop. This qualifies the available checkpoint's numerical operating
point; it does not turn incomplete50-epoch coverage into completion.

## Overnight outcome, September 24

All owned GPU workers were confirmed stopped by 07:49 Paris. The two gain
controls, nine Adam checkpoint cases, seven BN recalibrations and conditional
terminal replay completed and validate locally. Normalized legacy paused
cleanly at epoch48/50, preserving full state; the available checkpoint passes
its read-only solver audit. Epochs49–50 remain missing, so the study row is
`partial` despite successful completion of the other tests. Recorded compute
is 12.765 GPUh of the 16 GPUh cap. See the
[analyzed findings](../results/cifar10-l8-bn-gain-followup-seed0-20260923-v1/analysis/report.md)
and [coverage record](../results/cifar10-l8-bn-gain-followup-seed0-20260923-v1/analysis/collection_validation.json).

## Prepared completion from epoch 48, September 24

After receiving the overnight partial result, Filip said "cont" at about
07:56 Paris. The two remaining epochs were prepared. The coordinator proposed
finishing immediately beyond the previous 08:00 cutoff, but automatic approval
review rejected that launch at about 08:01 Paris: continuation was authorized
in substance, while an extension of the explicit cutoff was not. No production
job was launched from that rejected attempt. At about 08:09 Paris, Filip
answered "ok" to the explicit question asking whether to run the two epochs
on Fifi past 08:00, with a one-hour cap. The deadline extension is now approved.
Recheck live admission and use fresh timeout/deadline arguments; retain the
passed smoke because its exact config and source are unchanged.

The approved single case resumes the saved epoch-48 full state toward the unchanged
epoch-50 endpoint on Fifi, subject to fresh GPU admission. Expected duration
is about 26 minutes of training plus a short native final audit. The additional
allowance is one GPU-hour including the local smoke and audit, within the
original 16 GPUh total (12.765 GPUh recorded so far). A detached monotonic
runtime limit bounds this completion; the old overnight launcher is not reused.

Config: `configs/cifar/normalized_legacy_e48_e50_20260924.json`.
Output: the existing study's `cells/legacy_voltage_normalized_e48_e50/`.
Parent: `cells/legacy_voltage_normalized_e10_e50/checkpoints/final_model.pt`,
SHA256 `3dd39e1a2ce536860ad7dee3b7f3a09488ad01a58dcf20399eb956c1cfcffa89`.
Keep the paused parent bundle intact. Restore Adam at 67,536 steps, scheduler
epoch 48, model/BN/gains and every RNG state; deterministic sample order and
augmentation continue at epochs 49 and 50. No LR, batch size, precision,
training horizon or scientific solver setting changes. The accepted epoch-48
solver audit applies; run one local train/validation smoke through the same
frozen runner, then the two production epochs and native final audit. Stop
if execution unexpectedly falls back to CPU.

Collect and validate the new bundle, reconcile the original paused segment
with the completed continuation, and update the report and manifest around
the actual epoch-50 result. No other tests or reference repeats are included.

## Approved completion outcome, September 24

The continuation launched on Fifi at 08:10:50 Paris and completed epochs 49–50
in 1,523.75 seconds, plus the 8.53-second local smoke. Final validation accuracy
is 88.82% (raw legacy reference 88.72%); best observed accuracy is 88.94% at
epoch 49, while epoch 50 remains the declared primary endpoint. The native
final T/K audit passed. The saved state verifies scheduler epoch 50, 70,350
Adam/BN updates, the original cosine floor and all RNG states.

The completed child and all checkpoints were collected and validated, with
remote/local hashes and ancestry verified. The paused epoch-48 parent remains
unchanged. All nine production bundles validate locally: eight complete plus
that retained ancestor. All declared training and diagnostic coverage is now
complete, all owned workers have exited, and GPU resources are released.
Total recorded compute is 13.191 GPUh of the original 16 GPUh; the approved
daytime phase also stayed within its one-hour cap. See the linked findings
and coverage record above for scientific interpretation and retained exclusions.
