# Idea map — read-noise beta

Priority is a planning choice, not a hypothesis verdict. Outcomes live in result notes
and the generated ledger. Raw/parked ideas need not become experiments.

| Idea | Priority / maturity | Reasoning | Claim |
|---|---|---|---|
| Reduce current amplification at voltage amplification 1 | Initialization pilot reviewed; six clean BPTT trainings remain deferred | [Five-scheme layerwise interpretation](#reduced-current-pilot-and-the-five-scheme-mechanism) and [pilot evidence](../pilots/reduced-current-layerwise-20260928.md) | Smaller hidden D/P and earlier loss of noisy alignment at 1/0.5 and 1/0.25; clean training remains untested |
| Increase current amplification to A=4, B=2 | Authorized seed0 at2x ours LR: [exp020](series/007-relative-noise-training/experiments/exp-020-higher-current-training.md) | [X-010](explorations/X-010-higher-current-amplification.md) | [H-012](hypotheses/H-012-higher-current-noisy-learning.md) |
| Ours may have its largest training advantage over legacy near eta=1e-4 | Retrospective exploratory hypothesis; authorized eta=5e-4 extension in [exp019](series/007-relative-noise-training/experiments/exp-019-relative-noise-eta5em4.md) | [Interpretation below](#where-the-training-advantage-may-peak) | Extension of [H-011](hypotheses/H-011-relative-noise-training-transfer.md); optimum not established |
| Relative-noise alignment advantages transfer to ten-epoch Conv1/2/3 training | Authorized84runs: Conv1 D/F1; Conv2 D/F1,2,4; Conv3 D/F4,6; shared relative-noise grid; [exp014](series/007-relative-noise-training/experiments/exp-014-relative-noise-training.md) | Follow-up to exp013 | [H-011](hypotheses/H-011-relative-noise-training-transfer.md) |
| Locate each convolution's largest relative-noise alignment advantage over legacy | Explorer proposal: common D/F and eta grids across Conv1/2/3, global eta across layers, and shared fresh-draw refinement; [exp013](series/006-relative-endpoint-noise/experiments/exp-013-relative-noise-advantage-map.md) | [X-009](explorations/X-009-relative-noise-advantage-ridge.md) | [H-010](hypotheses/H-010-relative-noise-advantage-regions.md) |
| Fixed absolute read noise favors legacy's larger voltage scale under D/F matching | Exploratory prediction supported in exp012; causal voltage-scale attribution remains untested | [X-008](explorations/X-008-legacy-alignment-despite-attenuation.md) | [H-009](hypotheses/H-009-absolute-noise-voltage-scale.md) |
| Initial layerwise alignment regions in normalized displacement / read noise | Completed screen/confirmation; ours wins selected Conv1/2 worst-layer comparisons, Conv3 candidate not confirmed | [X-007](explorations/X-007-initial-displacement-noise-map.md) | [H-008](hypotheses/H-008-initial-alignment-region.md) |
| Extended displacement and additive endpoint noise | Admissions stopped September25; retain completed results, finish active cases; no further ranking-only training | [X-006](explorations/X-006-normalized-output-target.md) | [H-007](hypotheses/H-007-relative-output-noise-extension.md) |
| Ours outperforms legacy at matched initial relative output displacement | Reviewed: original noisy-training comparison; no further repeats | [X-006](explorations/X-006-normalized-output-target.md) | [H-006](hypotheses/H-006-relative-output-ours-versus-legacy.md) |
| Relative Gaussian read noise may make relative phase displacement predict robustness | Completed exploratory [exp012](series/006-relative-endpoint-noise/results/exp-012-relative-endpoint-noise.md): legacy advantage shrinks in all paired third-convolution comparisons; cross-device BPTT unchecked | [X-004](explorations/X-004-relative-read-noise.md) | [H-009](hypotheses/H-009-absolute-noise-voltage-scale.md) |
| Ours' relative-noise advantage follows a larger fractional hidden-layer response | Retrospective mechanism interpretation, reinforced by the reduced-current pilot; proposed hidden-D/P matching control remains unrun | [Interpretation below](#why-ours-outperforms-legacy-under-relative-read-noise) and [five-scheme evidence](#reduced-current-pilot-and-the-five-scheme-mechanism) | Mechanism extension of [H-009](hypotheses/H-009-absolute-noise-voltage-scale.md); causal attribution remains untested |
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

## Reduced-current pilot and the five-scheme mechanism

September 28 update requested by Filip after the completed
[layerwise initialization pilot](../pilots/reduced-current-layerwise-20260928.md).
The pilot owns the reviewed conclusions and complete evidence. It adds six
seed-0 contexts for 1/0.5 and 1/0.25 to nine existing baseline/ours/legacy
contexts, matching output D/F to 1/4/6 in Conv1/2/3. State statistics use 576
examples; clean and noisy cosine curves use the same 192 examples and three
paired acquisition draws for each nonzero eta. All 15 contexts validated.

The strongest explanation for ours' larger noisy cosine is its larger hidden
fractional phase response D/P. Matching output D/F leaves hidden D/P free to
differ. In Conv3, ours has 2.06, 5.18 and 4.03 times legacy's D/P in the three
convolutions. Clean EqProp/BPTT cosines are near one for all five schemes, and
the noisy-to-clean-EqProp curves closely follow the noisy-to-BPTT curves. The
observed separation therefore concerns resistance to endpoint acquisition noise.

The difference involves both displacement and voltage scale. In Conv3's last
convolution, ours has D=2.594e-4 and P=0.03895, versus legacy D=6.133e-4 and
P=0.37155. Its smaller absolute displacement still gives about four times the
fractional signal: D/P=0.006660 versus 0.001651. With the same relative eta,
the centered voltage contrast's signal-to-noise scale is proportional to D/P.
This is a mechanism diagnostic; gradient noise also depends on adjacent-layer
voltages, spatial correlations and the squared-edge local estimator.

For voltage gain A and current gain B, remove successive voltage gains in the
noninput layers with u_l=v_l/A^(l-1), leaving the input unchanged. Within a
fixed diode active region, the schematic node balance is

```text
(G_in + AB G_out) u_l = W_in u_(l-1) + AB W_out u_(l+1).
```

The normalized downstream coupling/loading coefficient AB is 4 for ours, 1
for both baseline and legacy, and 0.5/0.25 for the new settings. This explains
why voltage scaling can leave baseline and legacy's fractional signals almost
unchanged, while ours changes the hidden response and operating voltages.
Stronger normalized feedback is a plausible source of the measured advantage;
the exact D/P ratios also depend on loading, diode states and finite iterations.
They are not determined by AB alone, and a larger B need not improve training.

Both reduced-current settings decrease D/P in every convolution relative to
baseline. At eta=1e-4, Conv3's last-convolution BPTT cosine is 0.9925 for ours,
0.9066 for baseline, 0.9029 for legacy, 0.7297 for 1/0.5 and 0.4587 for
1/0.25. The first two Conv3 convolutions are already essentially unaligned at
this eta in every scheme; their near-zero fluctuations are not useful gains.
See the [displacement JPG](../../results/conv123-reduced-current-layerwise-20260928-v1/analysis/D_over_P_combined.jpg),
[cosine JPG](../../results/conv123-reduced-current-layerwise-20260928-v1/analysis/cosine_finite_K_BPTT_combined.jpg)
and [plotted state measurements](../../results/conv123-reduced-current-layerwise-20260928-v1/analysis/state_profiles.csv).

The added [free-state voltage profiles](../../results/conv123-reduced-current-layerwise-20260928-v1/analysis/F_combined.jpg)
and [nudged endpoint voltage profiles](../../results/conv123-reduced-current-layerwise-20260928-v1/analysis/P_combined.jpg)
show RMS magnitude pooled over the same 576 examples, not mean absolute voltage.
Reducing current gain increases free-state RMS in every convolution in this
pilot. Conv3's first convolution is 6.83 V at 1/1, 10.13 V at 1/0.5,
13.65 V at 1/0.25, and 2.39 V for ours. The larger operating voltages of
the reduced-current settings coexist with smaller D/P; their larger voltage
magnitude therefore does not imply a stronger fractional learning signal.
These figures reuse the completed replay and add no new simulation.

The near-flat later-layer voltage profiles have a direct gain-compensation
explanation. Legacy's free-state RMS is exactly 4^(l-1) times baseline's in
all nine saved layer measurements across Conv1/2/3 (l=1 at the first
convolution). Their AB=1 is identical. In Conv3, baseline's successive RMS
ratios from Conv2 onward are 0.2242 and 0.2584; voltage gain four converts
these to legacy's 0.8969 and 1.0336. Ours also approximately balances
attenuation, with ratios 0.8762 and 1.0605, while AB=4 changes its normalized
loading and feedback. Thus a flat absolute-voltage profile alone does not
establish the noisy-gradient advantage. The remaining large Conv1→Conv2
drop and the architecture/initializer dependence should remain explicit.

This strengthens the fractional-signal interpretation at initialization, while
leaving the proposed hidden-D/P matching control above as the discriminating
test. No additional simulation is assigned here. One initializer, fixed finite
T/K and unchecked cross-device equivalence limit the inference; clean BPTT
accuracy and trained noisy learning for the new settings remain unmeasured.

## Voltage profiles after clean EqProp training

Filip selected clean EqProp for the trained-checkpoint follow-up. The
[completed voltage pilot](../pilots/clean-eqprop-trained-voltage-20260928.md)
replays baseline/ours/legacy, seed0, fixed epoch10 for Conv1 and epoch30 for
Conv2/3 alongside their initializations, on the same 576 validation examples.
All 18 contexts and 54 layer profiles validated; initial RMS values reproduce
the prior curves. Source checkpoints and parameters remain unchanged.

The flat later-layer initialization profile does not fully survive training.
Conv3 trained RMS across Conv1/Conv2/Conv3/readout is
49.98/20.84/1.388/0.340 V for baseline,
18.94/2.186/0.316/0.284 V for ours, and
25.23/4.663/0.492/0.394 V for legacy. Ours and legacy now attenuate strongly
between the second and third convolutions, while their readouts remain close
to the last convolution's RMS. Baseline has the largest trained hidden RMS
at every depth; ours has the smallest. Independent learning removes the exact
initial baseline/legacy voltage-rescaling relation. Gain four can compensate
the initial attenuation without enforcing flatness for learned conductances.

See [initialization versus trained RMS](../../results/conv123-clean-eqprop-trained-voltage-20260928-v1/analysis/free_rms_initial_vs_trained.jpg)
and [trained mean absolute voltage](../../results/conv123-clean-eqprop-trained-voltage-20260928-v1/analysis/free_mean_absolute_trained.jpg).
The endpoint companion keeps each saved training beta. Conv3 output D/F falls
from the initial target6 to 0.001740/0.03109/1.00592 for baseline/ours/legacy;
these trained endpoints are not rematched. Voltage profiles alone do not
establish trained read-noise robustness. A future trained-gradient comparison
must distinguish original-training-beta behavior from freshly matched output
displacement. No such comparison is assigned by this voltage replay.

## Where the training advantage may peak

September 28 retrospective hypothesis, formed after exp014–018: ours may have
its largest advantage over legacy in the relative-noise region around eta=1e-4.
This is a question about the difference between schemes, not necessarily the
noise level giving the highest accuracy for ours. Architecture, finite training
horizon and optimizer history can shift the apparent peak. The old Conv2/3
ours/legacy epoch-30 trajectories reset Adam/RNG after epoch 10, unlike their
baseline controls.

[Exp019](series/007-relative-noise-training/experiments/exp-019-relative-noise-eta5em4.md)
adds eta=5e-4 for baseline, ours and legacy in Conv1/2/3 at fixed epoch10/30/30,
seed 0, and uninterrupted Adam from initialization. This extends exploration
between eta=3e-4 and 1e-3. It does not establish a global or architecture-independent
optimum, and the cross-study optimizer-history difference must remain visible.
