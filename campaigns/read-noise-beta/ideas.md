# Idea map — read-noise beta

Priority is a planning choice, not a hypothesis verdict. Outcomes live in result notes
and the generated ledger. Raw/parked ideas need not become experiments.

| Idea | Priority / maturity | Reasoning | Claim |
|---|---|---|---|
| Relative-noise alignment advantages transfer to ten-epoch Conv1/2/3 training | Authorized84runs: Conv1 D/F1; Conv2 D/F1,2,4; Conv3 D/F4,6; shared relative-noise grid; [exp014](series/007-relative-noise-training/experiments/exp-014-relative-noise-training.md) | Follow-up to exp013 | [H-011](hypotheses/H-011-relative-noise-training-transfer.md) |
| Locate each convolution's largest relative-noise alignment advantage over legacy | Explorer proposal: common D/F and eta grids across Conv1/2/3, global eta across layers, and shared fresh-draw refinement; [exp013](series/006-relative-endpoint-noise/experiments/exp-013-relative-noise-advantage-map.md) | [X-009](explorations/X-009-relative-noise-advantage-ridge.md) | [H-010](hypotheses/H-010-relative-noise-advantage-regions.md) |
| Fixed absolute read noise favors legacy's larger voltage scale under D/F matching | Exploratory prediction supported in exp012; causal voltage-scale attribution remains untested | [X-008](explorations/X-008-legacy-alignment-despite-attenuation.md) | [H-009](hypotheses/H-009-absolute-noise-voltage-scale.md) |
| Initial layerwise alignment regions in normalized displacement / read noise | Completed screen/confirmation; ours wins selected Conv1/2 worst-layer comparisons, Conv3 candidate not confirmed | [X-007](explorations/X-007-initial-displacement-noise-map.md) | [H-008](hypotheses/H-008-initial-alignment-region.md) |
| Extended displacement and additive endpoint noise | Admissions stopped September25; retain completed results, finish active cases; no further ranking-only training | [X-006](explorations/X-006-normalized-output-target.md) | [H-007](hypotheses/H-007-relative-output-noise-extension.md) |
| Ours outperforms legacy at matched initial relative output displacement | Reviewed: original noisy-training comparison; no further repeats | [X-006](explorations/X-006-normalized-output-target.md) | [H-006](hypotheses/H-006-relative-output-ours-versus-legacy.md) |
| Relative Gaussian read noise may make relative phase displacement predict robustness | Completed exploratory [exp012](series/006-relative-endpoint-noise/results/exp-012-relative-endpoint-noise.md): legacy advantage shrinks in all paired third-convolution comparisons; cross-device BPTT unchecked | [X-004](explorations/X-004-relative-read-noise.md) | [H-009](hypotheses/H-009-absolute-noise-voltage-scale.md) |
| Ours' relative-noise advantage follows a larger fractional hidden-layer response | Retrospective mechanism interpretation; next proposed control matches third-convolution D/P | [Interpretation below](#why-ours-outperforms-legacy-under-relative-read-noise) | Mechanism extension of [H-009](hypotheses/H-009-absolute-noise-voltage-scale.md); not separately tested |
| Noise in the nudging current may expose sensitivity to the direction of the teaching signal | Proposed mechanism control; not run | [X-005](explorations/X-005-noisy-nudging.md) | Not registered |
| Legacy displacement decays more sharply as beta decreases after clean EqProp training | Completed clean replay; fixed T=K=8 | [X-003](explorations/X-003-clean-beta-response.md) | [H-005](hypotheses/H-005-clean-beta-response.md) |
| Match initial output RMS instead of using a shared injected beta | Earlier absolute target; retain measured calibration as a reference | [X-001](explorations/X-001-fair-perturbations.md) | [H-001](hypotheses/H-001-output-rms-match.md) |
| Test matched-RMS betas against current policy in noisy training | After evidence/validity audit | [X-001](explorations/X-001-fair-perturbations.md) | [H-002](hypotheses/H-002-training-transfer.md) |
| Treat T/K sensitivity as a property of learned state/history | Diagnostic; historical evidence exists | [X-002](explorations/X-002-checkpoint-and-signal.md) | [H-003](hypotheses/H-003-state-dependent-settling.md) |
| Use weakest-layer noisy credit assignment rather than readout cosine alone | Qualify before selecting a metric | [X-002](explorations/X-002-checkpoint-and-signal.md) | [H-004](hypotheses/H-004-layer-signal-disparity.md) |
| Largest stable beta with a prespecified safety margin | Parked; depends on stability horizon and task | [X-001](explorations/X-001-fair-perturbations.md) | Not registered |
| Recalibrate beta at trained states or use a schedule | Parked; changes training protocol and calibration cost | [X-002](explorations/X-002-checkpoint-and-signal.md) | Not registered |
| Beta–learning-rate interaction | Confound to control; no broad joint sweep yet | [X-002](explorations/X-002-checkpoint-and-signal.md) | Not registered |
| Environment, precision and endpoint-noise coupling | Audit before comparing historical trajectories | [X-002](explorations/X-002-checkpoint-and-signal.md) | Not registered |

## Why ours outperforms legacy under relative read noise

September 25 interpretation of [exp012](series/006-relative-endpoint-noise/results/exp-012-relative-endpoint-noise.md), recorded after observing the results. Matching
normalized displacement at the output does not match the fractional nudging
response in hidden layers. Ours preserves a larger fractional response in the
third convolution, consistent with its better noisy-gradient alignment.

Let D = RMS((v_plus-v_minus)/2) and P be the pooled RMS of the positive and
negative endpoints. Under independent nodewise relative noise
`v_tilde = v + eta * abs(v) * xi`, the expected RMS error in the centered
voltage contrast is `eta * P / sqrt(2)`. Its signal-to-noise scale is therefore
`sqrt(2) * D / (eta * P)`. This is a voltage-contrast measure, not an exact
formula for gradient cosine: the local gradient also depends on both adjacent
layers and their spatial structure.

At output D/F=2 and eta=1e-4, all schemes measured on the local GPU:

| Scheme | Third-convolution D/P | Mean gradient error / clean gradient norm | Mean cosine to BPTT |
|---|---:|---:|---:|
| Baseline | 0.0550% | 1.387 | 0.5892 |
| Ours | 0.2222% | 0.356 | 0.9412 |
| Legacy | 0.0550% | 1.407 | 0.5838 |

Ours has 4.04 times legacy's fractional displacement and about 3.95 times less
relative gradient error. Clean third-convolution EqProp/BPTT cosine exceeds
0.9999 for all three schemes, so the measured advantage concerns preservation
under read noise. Sources: [physical-state controls](../../results/conv3-relative-endpoint-noise-20260925-v1/analysis/physical_state_controls.csv)
and [layer summaries](../../results/conv3-relative-endpoint-noise-20260925-v1/analysis/layer_summary.csv).

Legacy's free-state voltages at this initialization are approximately
1, 4, 16 and 64 times baseline's across the three convolutions and readout;
displacements scale nearly the same way. This leaves their hidden D/P values
almost unchanged and explains why their relative-noise curves nearly overlap.
Absolute noise rewards increased voltage scale; relative noise scales with it.

The coupling equations give a plausible mechanism for ours being different.
Removing successive voltage gains from the state coordinates leaves an
interaction weighting involving voltage_amp * current_amp: 1 for baseline
and legacy, 4 for ours. Ours thus changes normalized interlayer coupling and
loading, rather than only rescaling voltages. The larger fractional response
is measured; attributing its exact size to this coupling is an interpretation.

Proposed discriminating test: match third-convolution D/P across schemes at
initialization, hold eta fixed, and compare noisy/clean EqProp and BPTT cosine.
If the fractional-signal explanation dominates, ours' advantage should shrink.
Retain clean-gradient controls because changing beta can change finite-beta
error; also measure adjacent-layer D/P, since matching one layer does not
equalize the complete gradient-noise geometry. This is a proposed test, not a
new launch.

Scope: one Conv3 initializer and validation cohort, endpoint-noise EqProp on
MNIST. The first two convolutions remain strongly noise-dominated; this is
not evidence of better training or a CIFAR result. Absolute sigma and relative
eta are not matched noise powers. Cross-device BPTT equivalence across the
two displacement targets remains unchecked under the recorded user waiver.
