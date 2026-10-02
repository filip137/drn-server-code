# Rescaled-target checkpoint read-noise replay

Question: does the legacy/ours gap remain small when the output targets were
rescaled during training? Authorized October 1: use existing runs only, Conv2
target 1/16 and Conv3 targets 1/16 and 1/64. Six seed-0 best-validation checkpoints
from [the target-scale pilot](conv-output-scale-evidence-20260928.md); no training.

Compare centered current-nudged EqProp with clean finite-K BPTT from the same
post-T state. Native T=K=6/8 for Conv2/3; replay in float64, preserving float32
checkpoint values and each run's target encoding. Match pooled output D/F to
4/6 respectively, following the earlier initialization gradient plots. This is
an exploratory trained-checkpoint extension, not a causal target-scale comparison.
Use the first 12 fixed validation batches (192 images) from the established
guarded 576-example cohort, three paired draws, and relative endpoint noise
eta*abs(v)*N(0,1) at 14 levels from 3e-9 to 1e-2. Input remains exact. Include
clean controls and every convolution plus readout. No official-test access.

Report cosine curves and paired ours-minus-legacy gaps descriptively. No
significance threshold or expectation of a preferred winner; clean agreement
must be visible because nonlinear beta error can affect noisy agreement.

Config: `configs/conv/target_scale_read_noise_20261001.json`.
All smoke, production, failed attempts, analysis and plots live under
`results/conv23-target-scale-read-noise-20261001-v1/`.
Budget: one local RTX3090 GPU-hour including smoke; expected duration determined
by smoke. Current coordinator owns synchronous smoke and subsequent monitoring,
with checks at most five minutes apart and a one-hour elapsed cap. Stop for
nonfinite values, checkpoint/target mismatch, or unmatched displacement after
the bounded calibration. No optimization or checkpoint writes.

Execution state: complete and reviewed; exec session 38784 exited 0 after all six cases. No active monitoring remains.
Smoke completed on both architectures, two bounded matching passes each; all
smoke bundles validate. Clean layer cosines 0.99963–1.00000. Production command:
`python -m experiments.replay_target_scale_read_noise --config
configs/conv/target_scale_read_noise_20261001.json --output-root
results/conv23-target-scale-read-noise-20261001-v1/production --prior-seconds 15`.
Smoke uses the same command with `--smoke` and output `smoke/`. Local GPU had
no compute processes before admission, 842 MiB display use. GPU allocator cap
7.5 GiB with CPU offload of saved autograd buffers; all live model tensors CUDA.
Analysis: `python -m experiments.plot_target_scale_read_noise --study-root
results/conv23-target-scale-read-noise-20261001-v1`.
The diagnostic loader explicitly restores target_scale because the historical
analyzer helper assumes unit targets; both BPTT and EP share this same loss.


## Measurements and review

All 6 checkpoints completed with no failed/excluded/retried simulation cells.
Production elapsed 437.748 seconds plus 15 seconds charged preparation/smoke,
about 0.126 GPUh, below the one-hour cap. All 7 production bundles and 3 smoke
bundles validate with no errors. Coverage: 11,352 layer/batch/draw comparisons,
330 summary cells, all finite. Matching converged in two passes for every
checkpoint, with final D/F relative error below 6e-6. Every checkpoint and source
config remained byte-identical; model parameters unchanged. Target-encoding
checks passed in each replay; the 15 existing target-scale loss/gradient tests
also passed. No residual/equilibrium qualification was performed.

[Combined JPG](../../results/conv23-target-scale-read-noise-20261001-v1/analysis/conv23_rescaled_targets_noise_vs_bptt.jpg)
shows all layers, clean controls and paired-draw ranges. Individual figures:
[Conv2 target1/16](../../results/conv23-target-scale-read-noise-20261001-v1/analysis/conv2_target1over16_noise_vs_bptt.jpg),
[Conv3 target1/16](../../results/conv23-target-scale-read-noise-20261001-v1/analysis/conv3_target1over16_noise_vs_bptt.jpg),
[Conv3 target1/64](../../results/conv23-target-scale-read-noise-20261001-v1/analysis/conv3_target1over64_noise_vs_bptt.jpg).
All four were generated; the combined figure was visually inspected.
[Layer summaries](../../results/conv23-target-scale-read-noise-20261001-v1/analysis/layer_summary.csv),
[paired differences](../../results/conv23-target-scale-read-noise-20261001-v1/analysis/paired_gaps.csv),
and [coverage validation](../../results/conv23-target-scale-read-noise-20261001-v1/analysis/validation.json)
retain all measurements and checks. A plotting-only empty-selection bug was
corrected before review and guarded by an exact 14-points-per-curve assertion;
no simulation rerun or data change was needed.

Representative prespecified-grid measurements (means over 12 batches and 3 draws):

| Model | Target | Layer | Relative noise eta | Ours cosine | Legacy cosine |
|---|---|---|---:|---:|---:|
| Conv2 |1/16| convolution1 |1e-6|0.9278|0.1854|
| Conv2 |1/16| convolution2 |1e-6|0.9965|0.7975|
| Conv3 |1/16| convolution1 |1e-8|0.7300|0.0550|
| Conv3 |1/16| convolution2 |1e-8|0.8116|0.0625|
| Conv3 |1/16| convolution3 |1e-4|0.3190|0.0210|
| Conv3 |1/64| convolution1 |1e-8|0.4355|0.0039|
| Conv3 |1/64| convolution2 |1e-8|0.5123|0.0109|
| Conv3 |1/64| convolution3 |1e-4|0.3573|0.0097|

The legacy/ours difference is substantial in convolutional gradient retention
under these replay conditions. Ours' transitions occur at higher relative noise
than legacy's in all three panels. At sufficiently high noise both lose
convolutional agreement, while readout curves mostly overlap near one. Clean
mean layer cosines span 0.99663–1.00000. Conv3's first two convolutions are nearly
unaligned at eta = 1e-4 even where its third-convolution advantage persists; a
last-layer advantage does not imply useful whole-network gradients. Some legacy
Conv3 early-layer transitions lie below the sampled noise grid.

These are single-seed, clean-BPTT-trained, independently selected checkpoints;
matched output D/F leaves hidden displacement unmatched. Three noise draws are
acquisition replicates, not training seeds. Different inherited scheme learning
rates and training trajectories remain confounds. The plot establishes the
scheme gap in the rescaled-target cases, not that rescaling caused a larger gap:
there is no matched unit-target replay control here. No conclusion about noisy
training accuracy or official-test performance follows. Decision: the requested
existing-run replay and plotting task is complete; no further simulation launched.

## Authorized extension: A=4, B=2 at initialization

Filip then requested adding (voltage_amp,current_amp)=(4,2) to the initialization
noise plots to examine whether it merits testing. Reuse exp020's existing beta
calibrations; training already exists separately and is not relaunched here.
Extend all three panels of the existing Conv1/2/3 initialization figure, retaining
baseline, ours and legacy reference curves. Unit targets, common original
initializer, original paired seeds 111000000/112000000/113000000, the 14-level eta
grid, output D/F=1/4/6, and T=K=4/6/8 remain unchanged. Match on 576 examples and
measure noise on 192, exactly as the earlier screen. This extension is distinct
from the trained rescaled-target curves above. Cross-device parity with historical
curves is unchecked; parameter/cohort/source identities and paired RNG remain guarded.

Configs: `configs/conv/amp2_initial_noise_20261001/{conv1,conv2,conv3}.json`.
Runner: `python -m experiments.replay_relative_noise_advantage --config CONFIG`.
All artifacts remain in the same registered root under `initialization/`, with
`smoke/<arch>`, `production/<arch>` and `analysis/`. Budget: at most 600 seconds
per architecture including 15 seconds prior preparation, total 0.5 GPUh. Local 3090
placement, sequential execution, expected about 3 minutes based on the preceding
replay. Coordinator owns smoke, active monitoring, collection and review.
Decision rule: inspect clean agreement and each layer's B=2-minus-ours/legacy
gap across eta; an initialization gradient advantage is motivation, not proof
of improved training. No training launch is implied by this extension.


### Initialization extension: completed review

All three production cases and all three smokes completed successfully; each
parent and case bundle validates. Coordinator exec session 81854 exited 0 and
monitoring is finished. Production elapsed 162.153 seconds plus 45 seconds charged
smoke/preparation, about 0.058 GPUh. No failed/retried cases. Added 4,860 finite
layer comparisons and 135 new summary cells, giving 540 plotted cells with the
historical curves. All B=2 output matches have relative error below 6e-7.
Clean B=2 layer cosines span 0.99801–1.00000. No training or test evaluation ran.

[Initialization JPG with B=2](../../results/conv23-target-scale-read-noise-20261001-v1/initialization/analysis/initialization_amp2_noise_vs_bptt.jpg)
and per-architecture
[Conv1](../../results/conv23-target-scale-read-noise-20261001-v1/initialization/analysis/conv1_initialization_amp2_noise_vs_bptt.jpg),
[Conv2](../../results/conv23-target-scale-read-noise-20261001-v1/initialization/analysis/conv2_initialization_amp2_noise_vs_bptt.jpg),
[Conv3](../../results/conv23-target-scale-read-noise-20261001-v1/initialization/analysis/conv3_initialization_amp2_noise_vs_bptt.jpg)
were generated using `python -m experiments.plot_amp2_initial_noise --study-root
results/conv23-target-scale-read-noise-20261001-v1`; combined figure visually checked.
Plotted measurements and validation are in the same `initialization/analysis/`
directory. Ours is orange, legacy blue and B=2 purple consistently in the figures.

| Initialization context | eta | Layer | B=2 | Ours B=1 | Legacy B=0.25 |
|---|---:|---|---:|---:|---:|
| Conv1 D/F=1 |1e-3| convolution1 |0.67683|0.48559|0.16518|
| Conv2 D/F=4 |1e-5| convolution1 |0.94986|0.83477|0.28540|
| Conv2 D/F=4 |1e-5| convolution2 |0.99635|0.98927|0.89789|
| Conv3 D/F=6 |3e-8| convolution1 |0.92769|0.91272|0.75677|
| Conv3 D/F=6 |3e-8| convolution2 |0.86693|0.86547|0.77798|
| Conv3 D/F=6 |1e-4| convolution3 |0.99747|0.99253|0.90291|

Interpretation: B=2 improves convolutional noise retention visibly for Conv1/2
and Conv3's last convolution. The extra gain over ours in Conv3's first two
convolutions is much smaller; these remain the low-noise bottleneck. Thus B=2
is a meaningful gradient comparison, especially in Conv2, but this evidence
does not justify predicting better deep-network training.

Separately, the pre-existing exp020
[training summary](../../results/conv123-current-amp2-20260928-v1/analysis/summary.json)
reports all 27 outcomes complete. This session did not revalidate those training
bundles. Its single-seed fixed-LR2x recipe gives Conv2 B=2 versus ours 97.64/97.94%
at eta = 1e-5 and 96.82/97.10% at eta = 1e-4; Conv3 gives 96.42/96.26% and 91.06/95.06%.
Those mixed outcomes reinforce keeping gradient preservation and trained accuracy
separate. Review existing B=2 training before proposing another training test;
no training is launched by this plotting extension. Task complete.
