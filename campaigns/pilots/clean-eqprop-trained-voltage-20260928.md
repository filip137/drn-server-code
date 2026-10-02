# Clean EqProp: trained layerwise voltage profiles

## Question and authorized scope

Filip requested repeating the layerwise profiles for the three original
amplification schemes at trained points and explicitly selected clean EqProp.
The immediate comparison is voltage magnitude, following the
[initialization pilot](reduced-current-layerwise-20260928.md). Use baseline
(1,1), ours (4,1), and legacy (4,0.25), seed 0, fixed final epoch 10 for
Conv1 and epoch 30 for Conv2/3. These are the completed clean controls from
exp018 and exp016, with uninterrupted Adam and no endpoint training noise.
No training or checkpoint selection is assigned.

Question: does the nearly flat later-layer RMS voltage in the amplified
initializations persist after clean training? Compare the complete layer
profiles and adjacent-layer RMS ratios descriptively; no accuracy or
statistical-significance claim. The exact baseline/legacy rescaling seen with
identical initial weights need not survive independently trained weights.

## Frozen replay and evidence

Nine trained checkpoints and their nine corresponding seed-0 initializations,
using the same 576 ordinary-MNIST validation examples in 36 batches of 16 as
the initialization pilot. Expected cohort SHA-256:
`95af3eebd9416ec643b7d698d061b7dac3fe0d52311dd888293609bada629acd`.
Float64, perfect diodes, input gains 40/100/360, native T=K=4/6/8,
zero biases and the saved model configuration. No optimizer steps or official
test reads. Finite-iteration states, without additional convergence qualification.

Primary curves: free-state RMS and mean absolute voltage, pooled over every
node and example within each layer; include readout. Also retain signed mean,
standard deviation, min/max and exact-zero occupancy. A companion endpoint
RMS uses the saved training beta with clean centered frozen-current nudges.
Do not recalibrate beta at trained checkpoints: output D/F can drift during
training and must be reported, not labelled as still matched. Do not add
gradient/noise sweeps to this voltage-only replay.

Require source checkpoint/config hashes, exact restore and unchanged-parameter
checks, unchanged checkpoint bytes after replay, cohort identity and complete
18-context coverage. Recomputed initialization free RMS must agree with the
prior validated values within floating-point tolerance before claiming
initialization-to-trained differences. JPGs and count-weighted CSVs are the
deliverables, with reproducible analysis code and canonical result bundles.

Source studies:
- `results/conv1-clean-and-baseline-noise-20260927-v1/`
- `results/conv23-noiseless-three-seed-20260927-v1/`
- `results/conv23-noiseless-baseline-three-seed-20260927-v1/`

All replay attempts, source staging and analysis belong under
`results/conv123-clean-eqprop-trained-voltage-20260928-v1/`.

## Execution and monitoring handoff

Source inventory and runner prepared; six focused tests and a Conv1
initialization CPU smoke passed. Its canonical parent and case bundles validate:
`results/conv123-clean-eqprop-trained-voltage-20260928-v1/local-runner-smoke/`.
Frozen numerical source and selected inputs (649 files, approximately 96 MiB)
are staged in `source-v1/`, with complete hashes in `source_identity.json`.
Readable configs: `configs/conv/clean_eqprop_trained_voltage_20260928/conv{1,2,3}.json`.
Selected original artifacts and epochs were checked against the completed
source bundles; receipts are in `inputs/source_inventory.json`.

Use Akib RTX 3080, where the prior
batch-16 float64 Conv3 replay fit under the 7.5 GiB allocator cap. Reserve one
GPU-hour including smoke and recovery; expected computation is minutes,
subject to a same-path one-batch smoke and live GPU admission. Deadline is
24 hours from first queue submission. Keep model settings unchanged.
Use the shared run-watch queue for launch and monitoring; record accepted IDs
and fresh service ownership here before relinquishing polling. Root owns
scientific validation, bounded recovery, collection and review.

GPU smoke jobs submitted:
`clean-eqprop-voltage-smoke-conv{1,2,3}-20260928`.
Queue specs and collection hook are in `launch/`. Allocations: 150 seconds per
smoke, 600/900/1200 seconds for Conv1/2/3 production, and 450 seconds reserved
for bounded recovery, totaling 3600 seconds. All 18 GPU smoke cases and their
three parent bundles validated locally; peak CUDA reservations were 44/52/70
MiB for Conv1/2/3. Receipt: `launch/gpu-smoke-validation.json`.
Production jobs submitted after these checks:
`clean-eqprop-voltage-production-conv{1,2,3}-20260928`.
Inspect with `python /home/filip/.codex/skills/run-watch/scripts/gpu_queue.py status JOB_ID`.
The collection hook rsyncs the lane and validates parent and all case bundles.
Root analyzes all three collected production lanes with
`python -m experiments.analyze_clean_eqprop_voltages`.
The shared service (PID 585501) has acknowledged ownership and produced fresh
observations and completed-case artifacts. Its identity and accepted job IDs
are retained in `launch/monitoring-handoff.json`; no second watcher runs.
Absolute deadline: 2026-09-29T15:58:56.641222Z.

## Results and review

**Complete and reviewed:** all nine trained checkpoints and nine matched
initializations replayed on all 576 examples, producing 54 layer profiles.
All nine initialization RMS profiles reproduce the earlier validated curves
within the predeclared tolerance. Every checkpoint restored exactly; parameters,
source configs and checkpoint bytes remained unchanged. All values were finite,
all frozen-zero-bias and centered-start/current-force guards passed, and all
three parent plus 18 case bundles validated locally. Six unit tests and 18 GPU
smoke contexts passed before production. No failed compute attempts or recovery.

All six queue jobs are complete, including collection and target/local validation.
Total queue-charged time was 93.351915 seconds (0.02593 GPU-hours) against the
one-GPU-hour cap. CPU plotting and the 0.31-second scientific CPU smoke are
separate. Final queue receipts: `launch/queue-completion.json`.
The previously failed rsync directory-creation attempt happened before any
compute and copied no results; creating the new remote parent resolved staging.
No active replay or monitoring obligation remains.

Figures (log voltage axes, matched limits within each metric):
- [Trained free-state RMS](../../results/conv123-clean-eqprop-trained-voltage-20260928-v1/analysis/free_rms_trained.jpg)
- [Initialization versus trained RMS](../../results/conv123-clean-eqprop-trained-voltage-20260928-v1/analysis/free_rms_initial_vs_trained.jpg)
- [Trained mean absolute voltage](../../results/conv123-clean-eqprop-trained-voltage-20260928-v1/analysis/free_mean_absolute_trained.jpg)
- [Trained nudged endpoint RMS](../../results/conv123-clean-eqprop-trained-voltage-20260928-v1/analysis/endpoint_rms_P_trained.jpg)
- [Trained signed mean](../../results/conv123-clean-eqprop-trained-voltage-20260928-v1/analysis/free_signed_mean_trained.jpg)

Each metric also has an initialization-versus-trained figure. The four primary
linked magnitude figures were visually inspected. Full statistics, including
signed mean, standard deviation, extrema and exact-zero fractions, are in
`analysis/voltage_profiles.csv`. Paired deltas/ratios are in
`analysis/voltage_epoch_comparison.csv`; coverage, source-result hashes and
analysis-source identity are in `analysis/validation.json`. Every case retains
per-batch sufficient statistics and explicit read-only guards.

Trained free-state RMS in volts:

| Architecture | Scheme | Conv1 | Conv2 | Conv3 | Readout |
|---|---|---:|---:|---:|---:|
| Conv1, epoch10 | baseline | 2.3780 | — | — | 0.15210 |
| Conv1, epoch10 | ours | 1.1378 | — | — | 0.17814 |
| Conv1, epoch10 | legacy | 1.3673 | — | — | 0.20025 |
| Conv2, epoch30 | baseline | 10.7464 | 1.32666 | — | 0.20718 |
| Conv2, epoch30 | ours | 2.9252 | 0.48545 | — | 0.19520 |
| Conv2, epoch30 | legacy | 3.9951 | 0.61702 | — | 0.22357 |
| Conv3, epoch30 | baseline | 49.9788 | 20.8434 | 1.38781 | 0.34034 |
| Conv3, epoch30 | ours | 18.9404 | 2.18640 | 0.31639 | 0.28393 |
| Conv3, epoch30 | legacy | 25.2334 | 4.66346 | 0.49192 | 0.39376 |

The nearly flat later-layer initialization profile does **not** persist through
the convolutional layers after training. In Conv3, ours' Conv3/Conv2 RMS ratio
changes from 0.8762 to 0.1447 and legacy's from 0.8969 to 0.1055. Their
readout/Conv3 ratios remain nearer one: 0.8974 and 0.8005, respectively.
Baseline has the largest trained hidden RMS in every architecture; ours has
the smallest. Hidden RMS grows through training in Conv2/3, while Conv1's
single hidden layer decreases for all schemes.

Independent training breaks the initial baseline/legacy relation
F_l(legacy)=4^(l-1) F_l(baseline). This is consistent with learned conductances
changing attenuation and loading; the constant A=4 does not enforce flat
voltage across arbitrary learned layers. No causal decomposition of learning,
scheme-specific learning rates or beta was performed.

The endpoint companion uses the **saved training beta**, as specified. Trained
output D/F is no longer the initial matching target:

| Architecture | Initial target | Trained baseline | Trained ours | Trained legacy |
|---|---:|---:|---:|---:|
| Conv1 | 1 | 0.05729 | 0.09281 | 0.21555 |
| Conv2 | 4 | 0.01322 | 0.12687 | 0.67473 |
| Conv3 | 6 | 0.001740 | 0.03109 | 1.00592 |

Thus this replay measures actual trained operating voltages at the original
training beta. It does not compare trained noisy gradients at rematched
displacement. No gradient/noise sweep, training or official-test evaluation
was run. Findings apply to seed0 fixed-final clean EqProp checkpoints and
native finite T/K; no new equilibrium or cross-device qualification is claimed.
The requested voltage comparison is complete; no further simulation assigned.

## September 29 manuscript figures and conductance extension

Filip requested adding a possible conductance-redistribution mechanism to
Section 4.1, plotting Conv3 conductance distributions for all three schemes,
and plotting free-state voltage magnitude before and after training, then
pushing the changes to Overleaf. Reuse this pilot's fixed-final, seed-0 clean
EqProp checkpoints and its validated 576-example voltage replay. The voltage
figure covers Conv1/2/3; the conductance comparison covers each of Conv3's
three convolutional tensors and its readout, at initialization and epoch 30.
No training, new replay, checkpoint selection or test-set read is assigned.

This is a descriptive extension: compare layerwise conductance distributions,
including their exact-zero mass, without treating marginal weight statistics
as a measurement of each node's downstream/upstream KCL load. The proposed
mechanism remains a hypothesis. Each stored convolutional kernel coefficient
is counted once, rather than once per spatial use; biases are excluded.

Execution: local CPU analysis, expected under five minutes, no GPU allocation.
Reproducible script will live in the paper checkout's `scripts/`; generated
analysis goes in the existing result root's `analysis/manuscript_20260929/`.
Require checkpoint/config hashes matching the replay inputs, finite in-bound
weights, identical initial tensors across schemes, complete histogram mass
including zeros, and unchanged source bytes. Reuse the completed voltage
validation after checking its recorded source hashes and exact table contents.
Root owns execution, review and Overleaf synchronization in this session.

**Analysis complete and reviewed.** All 54 reused voltage rows match their
recorded source tables and hashes; all 24 Conv3 conductance tensors have the
expected shapes, finite values and bounds. Checkpoint/config hashes match the
frozen replay inputs and remain unchanged. Initial weights are exactly equal
across schemes, all biases are zero, and every histogram accounts for every
stored weight, including exact zeros. The figures were visually checked.
The base Python lacked Matplotlib; the existing `py312` environment ran the
analysis successfully. No new simulation was needed.

- [Voltage magnitude before/after](../../results/conv123-clean-eqprop-trained-voltage-20260928-v1/analysis/manuscript_20260929/free_state_rms_before_after.jpg)
- [Conv3 conductance distributions](../../results/conv123-clean-eqprop-trained-voltage-20260928-v1/analysis/manuscript_20260929/conv3_conductance_distributions.jpg)
- [Conductance statistics](../../results/conv123-clean-eqprop-trained-voltage-20260928-v1/analysis/manuscript_20260929/conv3_conductance_summary.csv)
- [Provenance and checks](../../results/conv123-clean-eqprop-trained-voltage-20260928-v1/analysis/manuscript_20260929/provenance.json)
- [Reproduction script](../../papers/amplification_final_accuracy_20260929/scripts/plot_clean_eqprop_conductances_voltages.py)

The final baseline zero fractions across the four weight tensors are
30.64%, 32.30%, 13.31%, 15.12%; balanced has 4.77%, 11.56%, 8.82%, 10.26%,
and legacy 7.12%, 6.15%, 3.27%, 6.07%. All start near 50% zeros. Conductance
reorganization is strongly layer-dependent. For baseline, ratios of adjacent
tensor means change from 0.1776, 0.7096, 0.3032 initially to 0.0272, 0.1975,
2.6232 finally. The first two ratios decrease but the readout ratio increases.
These are ratios of unweighted stored coefficient means, not per-node KCL
loads; spatial connectivity and scheme-specific amplification also matter.
Thus the distributions describe redistribution but do not establish the
proposed causal mechanism or a uniform downstream reduction. The manuscript
retains Filip's cautious mechanism paragraph and makes no stronger claim.

Manuscript integration complete: Overleaf `main` updated as `3a05a62`, adding
the requested paragraph, replacing Figure X with the RMS figure reference,
and including both figures, source CSVs, provenance and the reproduction script.
The full manuscript compiled successfully with Tectonic, and its figure page
was visually inspected. Only pre-existing duplicate supplement equation labels
were reported. No new unresolved references or layout overflow was reported.

Later September 29 presentation revision: at Filip's request, Figure 4 now
shows only the percentage of exactly zero weights by layer, retaining the
shared initialization and all three final schemes. Positive histograms remain
in the supporting CSV; numerical source data are unchanged. The figure,
caption, referring sentence and reproduction script were updated, the paper
compiled successfully, and Overleaf `main` was updated as `9ca814e`.
