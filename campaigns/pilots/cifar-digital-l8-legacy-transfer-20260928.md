# Digital CIFAR-10 L8 → legacy, evaluation-only transfer

## Question and source audit

User request September 28: determine whether legacy retains the trained digital
L8 accuracy when initialized from that run, and check architecture compatibility.
Related context: [CIFAR campaign](../cifar10-legacy-gap/README.md).
This is a bounded transfer pilot, not continuation of its analog LR selection.

Source: `results/cifar10-digital-l8-wide-affinebn-adam-seed0-20260921-v1/production-seed0`.
Best checkpoint epoch48: 92.62% original test; final epoch50: 92.57%.
Primary checkpoint is the existing best, selected by the historical run, without
new checkpoint selection. The original run trained on all50,000 training images
and repeatedly monitored official test. Its5,000-image analog-validation overlap
must never be called held-out digital validation.

The models share logical blocks[128,128,128]/[256,256,256]/[512,512],
3×3 stride1 padding1 bias-free convolutions, boundary2×2 max-pool then affine BN,
and8192-to-10 readout dimensions. Native legacy has doubled physical channels,
positive bounded conductances, within-block electrical feedback, four boundary
gains initialized to100, and an analog readout without bias. Digital has signed
weights, feedforward ReLUs and an affine digital readout. Checkpoint inspection:
5,395,594 digital parameters versus21,413,124 native legacy trainable parameters.
There is no direct state-dictionary load or guaranteed functional equivalence.

## Prospective comparison and decision

Use symmetric sign splitting for each convolution: G=[[W+,W-],[W-,W+]],
adding the same1e-7 conductance floor to every entry (preserves differences).
Check reconstruction numerically and reject upper-bound overflow rather than clip.
Copy all BN affine parameters and running buffers exactly; evaluation mode only.
Retain native legacy(A,B)=(4,.25), gains100, T=[6,6,4], no training, gain tuning,
BN recalibration, or solver sweep. This specifically tests a direct sign-split
transplant; a negative result does not reject more elaborate conversion methods.

Two scientific choices were presented before data evaluation: original10,000-image
test replay versus the5,000 training-overlap diagnostic; original digital classifier
versus full native analog classifier. No response received during preparation; proceed with stated defaults: original
test replay and both heads, each explicitly labelled. Both share the same analog
features, so including the native-head diagnostic adds negligible compute.
A native-head transplant has no slot for the digital classifier bias and must be
explicitly labelled if dropped; it cannot silently be called exact conversion.

Primary metrics: paired accuracy, CE, prediction agreement, logit relative L2.
Proposed practical retention threshold: no more than0.2 percentage-point accuracy
loss from the digital replay on identical examples, frozen before measurements.
Synthetic tests are mathematical implementation checks, not accuracy evidence.

## Execution and monitoring handoff

State: complete and reviewed. Shared queue reports execution, collection and declared validation succeeded; no replay worker remains assigned.
Owner: shared queue completed its execution/collection assignment; coordinating agent completed scientific review.
Queue job: `cifar-digital-l8-legacy-transfer-20260928`.
Verified worker PID2200355, supervisor2200344, attempt4666cf283414487e85368e533e2be7d4;
first service progress observation3760/10000 examples at59.2s. Queue owns routine checks.
Exact command/transport: `results/cifar10-digital-l8-legacy-transfer-20260928-v1/job.json`.
Remote root: `/home/filiposana/server_code/results/cifar10-digital-l8-legacy-transfer-20260928-v1`.
Inspect: `python ~/.codex/skills/run-watch/scripts/gpu_queue.py status cifar-digital-l8-legacy-transfer-20260928`.
Queue collects and validates full bundle; no automatic scientific retry.
Result root reserved: `results/cifar10-digital-l8-legacy-transfer-20260928-v1`.
Budget: one fixed-checkpoint full replay plus at most32-image smoke; at most1 GPUh,
no training. Target: Akib RTX3080, idle at admission; batch8. Smoke elapsed0.68s, peak444MiB; expected full replay approximately3min plus startup/validation. All lab targets and Jean Zay checked;
no existing jobs changed. Source identity and staging archive are in the result root.
Stop on nonfinite output, conversion mismatch, BN mutation, source identity mismatch,
or budget exhaustion; retain failed/partial attempts. Do not tune on test outcomes.

## Architecture comparison

| Property | Digital source | Native legacy target |
|---|---|---|
| Logical convolution widths |128,128,128 /256,256,256 /512,512|Same|
| Physical channels |3 input;128/256/512 blocks|6 input;256/512/1024 blocks|
| Convolution geometry |3×3, stride1, padding1, no bias|Same spatial geometry; differential pairs|
| Hidden operation |Feedforward signed convolution then ReLU|Coupled perfect-diode equilibrium, conductance normalization and downstream loading|
| Boundaries |MaxPool2 then affine BN|Same pooling order and logical BN dimensions|
| Readout |Linear8192→10 with bias|Positive16384×10 conductances, normalized linear voltage output, no bias|
| Gains |No extra gains|Four positive boundary gains, initialized100|
| Parameters |5,395,594|21,413,124|

## Results and review

Full10,000-image original-test replay completed. Source epoch48 digital accuracy
reproduces exactly:9262/10000=92.62%. Both direct legacy transfers score1000/10000
=10.00%, a loss of82.62 percentage points; both predict class2 (bird) for every image.
The predeclared0.2pp retention criterion fails for this direct mapping.

| Evaluated model | Accuracy | CE | Agreement with digital | Logit relative L2 |
|---|---:|---:|---:|---:|
| Original digital checkpoint |92.62%|0.321730|100%|0|
| Legacy convolution blocks + original digital head |10.00%|4.788563|9.73%|1.080153|
| Native legacy head, digital bias explicitly omitted |10.00%|4.778738|9.73%|1.081914|

Native computation158.63s; queue execution/validation161.22s, plus short smoke;
peak reserved444MiB. One seed, one fixed best checkpoint, no trainable updates or
BN recalibration. Full coverage/finite logits and correct counts were independently
recomputed from the saved predictions; indices are exactly0…9999 and labels balanced.
Both source checkpoint and all transferred parameters/BN buffers are unchanged.

Evidence: [full result](../../results/cifar10-digital-l8-legacy-transfer-20260928-v1/full/result.json),
[predictions](../../results/cifar10-digital-l8-legacy-transfer-20260928-v1/full/artifacts/predictions.npz),
[scientific validation](../../results/cifar10-digital-l8-legacy-transfer-20260928-v1/analysis/scientific_validation.json).

**Interpretation:** the logical convolution/pooling/BN layout matches, but direct
sign splitting into the existing coupled legacy equations does not preserve this
trained digital function. Keeping the exact original digital classifier still
collapses to chance, so classifier bias omission alone cannot explain the failure.
The architecture audit supplies concrete differences: channel/spatial conductance
normalization, downstream electrical loading, differential decoding and analog
boundary voltage gains. This replay does not isolate their separate causal effects.

**Scope:** negative for this untuned native-gain direct mapping with copied frozen
BN, not proof that every digital-to-analog conversion is impossible. It is historical
exploratory test replay, not a new held-out paper accuracy or an analog-training
ranking. The digital source originally used the test set for checkpoint selection.

**Next decision:** a faithful conversion would need a separate explicit mapping
of per-layer activation scales/loads and BN coordinates, and a decision on the
classifier. Validate intermediate feature agreement before another full accuracy
replay. Such fitting/recalibration, retraining or a solver sweep was not performed.

Implementation validation: Two synthetic tests pass: signed-current/ReLU-pair reconstruction,
and exact BN copy/native-forward/head-equation checks. Target smoke canonical validation
passes with finite outputs and unchanged source checkpoint, BN buffers and parameters.
Maximum signed-weight reconstruction error1.86265e-9. The archived digital runner is
byte-identical to the current one. The32-image smoke is not used as an accuracy estimate.

The original analog JSON is a model/configuration template. Its training-only
optimizer/epoch fields are unused in this evaluation runner. The manifest's dataset
and transfer fields explicitly record the effective official-test diagnostic and zero
optimizer steps; this does not reopen analog hyperparameter selection.
