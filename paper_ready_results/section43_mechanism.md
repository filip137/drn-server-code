# Section 4.3: gradient alignment and phase contrast

Completed 2026-09-18. Evidence: ordinary-MNIST validation mechanism replay, one training seed; no official-test access or optimizer steps. Outcome: **supports greater legacy noise sensitivity at the selected operating points**, with qualifications described below.

Figures: [gradient cosine, JPG](figures/section43_gradient_cosine.jpg) · [PDF](figures/section43_gradient_cosine.pdf); [phase contrast, JPG](figures/section43_phase_displacement.jpg) · [PDF](figures/section43_phase_displacement.pdf). PNG and SVG versions are in the same directory. [LaTeX captions](section43_mechanism_captions.tex). Both figures use matplotlib, omit panel letters and checkpoint-selection footers, and retain the Section 4.2 scheme colors. The cosine axis uses a shared multiplier of 10^-5.

## Measurements and interpretation

Every scheme has strong clean gradient alignment: all 27 mean layerwise cosines are above 0.9986. The legacy result is therefore not explained by an already misaligned clean gradient at these checkpoints.

At sigma=10^-5, Conv2 first-weight-layer mean cosine is 0.9980 / 0.9622 / 0.0851 for baseline / balanced / legacy. For Conv3 third-weight-layer gradients the corresponding means are 0.9778 / 0.9915 / 0.0426. These measurements show much earlier loss of legacy gradient direction under endpoint noise.

The centered voltage contrast decreases toward the input in every architecture/scheme. Conv3 H1 RMS is 4.073e-07 / 1.577e-07 / 2.919e-10 for baseline / balanced / legacy; legacy is 540.2 times smaller than balanced. This supports the proposed small-signal mechanism under the selected training configurations. It does not mean that smaller displacement creates noise: the externally imposed noise obscures an increasingly small phase contrast.

Baseline and balanced Conv3 are also sensitive in their early layers: their first-layer mean cosines at the smallest noise level are 0.1482 and 0.0672. Baseline retains better Conv2 convolution-gradient alignment than balanced across this grid. Thus the figures do not establish a universal baseline/balanced ordering or a complete causal explanation of the final training accuracies.

| Architecture | Scheme | H1 contrast RMS | H1 / output RMS | First-weight cosine, clean | At sigma=10^-5 | At sigma=5×10^-4 |
|---|---|---:|---:|---:|---:|---:|
| Conv1 | Baseline | 5.015e-03 | 2.238e-02 | 0.99867 | 0.99867 | 0.99750 |
| Conv1 | Balanced | 2.930e-03 | 1.584e-02 | 0.99960 | 0.99960 | 0.99467 |
| Conv1 | Legacy | 5.856e-04 | 3.636e-03 | 0.99989 | 0.99984 | 0.89237 |
| Conv2 | Baseline | 7.471e-05 | 8.393e-04 | 0.99962 | 0.99801 | 0.36209 |
| Conv2 | Balanced | 1.519e-05 | 5.757e-05 | 0.99974 | 0.96218 | 0.08472 |
| Conv2 | Legacy | 2.397e-07 | 8.191e-06 | 1.00000 | 0.08510 | 0.00304 |
| Conv3 | Baseline | 4.073e-07 | 2.207e-04 | 1.00000 | 0.14823 | 0.00400 |
| Conv3 | Balanced | 1.577e-07 | 1.819e-06 | 0.99996 | 0.06716 | 0.00293 |
| Conv3 | Legacy | 2.919e-10 | 3.390e-08 | 1.00000 | 0.00343 | 0.00331 |

## What is plotted

The heatmaps contain all nine clean EqProp checkpoints, all ConvWeight tensors and the dense readout, and all six declared noise levels. Each cell is the mean of individual minibatch/noise-draw cosines against a noiseless, matched finite-K BPTT reference. It is not the cosine of averaged gradients. There are 36 clean comparisons per layer and 288 comparisons per nonzero-noise cell. Undefined zero-norm cosines would remain missing; none occurred.

The displacement curves use sqrt(sum((v_plus-v_minus)^2/4)/number_of_state_coordinates), pooled over the 576 examples within each layer. They exclude the clamped input and use absolute simulator voltage units. The shared log scale permits comparisons across architectures. The gray band is the standard deviation sigma/sqrt(2) of independent centered voltage-read noise, spanning 7.071e-06 to 3.536e-04. It is not an exact gradient SNR or a hard failure boundary.

Raw free-to-nudged movement contains continued relaxation. For baseline Conv3 H1, positive-minus-free RMS is 1.179e-04, matched-zero drift is 1.179e-04, and centered contrast is 4.073e-07: the raw value is 289.3 times the centered contrast. The supporting data preserve both signed raw displacements, both matched-zero displacements, and zero-nudge drift; the main figure isolates the centered signal used by the trained EqProp estimator.

## Frozen protocol and limitations

- Nine seed-0 clean EqProp maximum-validation checkpoints, frozen throughout the noise sweep. Conv3 baseline uses the new beta_inj=10 clean control; the extra Conv1 beta_inj=200 training is outside this declared replay.
- The same ordered 36 validation batches of 16 examples from the September 16 displacement study; source indices, preprocessing, split, and every batch payload hash match. Official MNIST test data are never read.
- Native float64 checkpoint loading and dynamics; exact-zero frozen biases; perfect diodes; wide [0,100] conductance bounds. The loader verifies saved PT tensors against NPZ without passing through float32.
- T=K=4/6/8 for Conv1/2/3. Centered, frozen-current EqProp and BPTT start from the same post-T states. No optimizer is constructed. Clean phases are computed once per checkpoint/batch, then reused for noisy endpoint readout.
- Noise levels 0, 10^-5, 3×10^-5, 10^-4, 3×10^-4, 5×10^-4; eight draws per nonzero level. Standard-normal draws are paired across schemes within architecture and across sigmas. Positive/negative endpoint streams and layer streams are independent. Input reads remain noiseless.
- Scheme-specific beta, trained weights, and operating points differ. These figures characterize the selected schemes and nudging settings; they do not isolate amplification from beta or checkpoint differences. Output-normalized displacement is included as a supporting measure of attenuation.
- Finite-T/K agreement does not certify equilibrium convergence. Baseline Conv3 fails 72/144 batch-layer post-T projected-residual p90 checks at threshold 0.01 (largest batch-layer p90 0.26808; maximum residual 0.33324). Subsequent zero/positive/negative endpoints pass those p90 checks. All cases remain included. Historical beta qualification and seed-confirmation caveats remain attached to the source records.
- These are fixed clean checkpoints, not checkpoints retrained separately under each noise level. Batch/draw percentiles describe within-checkpoint variation, not confidence intervals over training seeds or proof of a final-accuracy effect.

| Architecture | Baseline injected beta | Balanced injected beta | Legacy injected beta |
|---|---:|---:|---:|
| Conv1 | 100 | 30 | 3 |
| Conv2 | 100 | 10 | 0.03 |
| Conv3 | 10 | 3 | 0.001 |

## Coverage and reproducibility

All 9 checkpoints × 36 batches completed: 324 clean phase/BPTT evaluations, 39,852 layerwise gradient comparisons, 162 cosine cells, and 162 displacement groups. All 11 canonical bundles (nine sources, smoke, full replay) validate. All checkpoint hashes and float64 tensor guards pass, and all 492 smoke comparisons are bitwise identical to their production counterparts. Eight targeted tests pass. There are no failed attempts or excluded cases. Total local RTX 3090 use, including smokes: 11.302 GPU-minutes (0.188374 GPU-hours), below the one-hour cap.

- [Cosine summary CSV](section43_gradient_cosine_summary.csv): means, medians, p10/p90, standard deviations, norms, relative errors, and noisy-versus-clean EP comparisons.
- [Displacement summary CSV](section43_phase_displacement_summary.csv): pooled RMS, signed/absolute means, batch variation, and output-normalized values for all six displacement definitions.
- [Checkpoint sources](section43_checkpoint_sources.csv), [configuration](section43_mechanism_config.json), [verification](section43_mechanism_verification.json), [figure provenance](section43_mechanism_provenance.json).
- [Full canonical replay](../results/section43-eqprop-mechanism-20260918-v1/full-1/): raw comparison CSVs, residuals, cohort, batch guards, resolved config, source snapshots, and manifest/status/result records.
- [Replay runner](run_section43_mechanism.py), [matplotlib generator](plot_section43_mechanism.py), [targeted tests](test_section43_mechanism.py).

Regenerate the plots without GPU work from this directory:

```bash
MPLCONFIGDIR=/tmp/section43-matplotlib /home/filip/miniconda3/envs/py312/bin/python plot_section43_mechanism.py
```
