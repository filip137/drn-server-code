# Conv3: existing per-matrix cosine >0.90 versus >0.99 evidence

Compiled 2026-09-21 without new replay or training. Seed 0, T=K=8, float64 centered frozen-current EqProp, weights [0,100], frozen zero biases. Noise comparison: sigma=0 versus 1e-3. Accuracies are ordinary-MNIST 5,000-example validation measurements, not official-test results.

Select the largest beta already measured in the combined original/refined grid that passes strictly for every weight matrix in every one of 36 batches at initialization and 36 at the saved BPTT checkpoint. No norm gate. The p90 choices are the trained refined choices. The p99 choices are a retrospective lookup in those existing measurements, not a new refined p99 search or training qualification. Ours beta0.987333678708 passes and exceeds the original-grid beta0.9; baseline10 and legacy0.1 remain coarse-grid choices. These are measured maxima, not proven global beta limits.

## A. Initial per-layer gradient cosine

Entries are **minimum / median** across the same 36 batches at initialization only. The four columns correspond to ConvWeight_0, ConvWeight_1, ConvWeight_2, DenseWeight_0.

| Threshold | Scheme | Beta | Conv1 | Conv2 | Conv3 | Dense |
|---|---|---:|---:|---:|---:|---:|
| 0.90 | baseline | 404.141106 | 0.901977 / 0.992760 | 0.971476 / 0.988974 | 0.988203 / 0.991160 | 0.999277 / 0.999463 |
| 0.90 | legacy | 4.4225011 | 0.902633 / 0.993805 | 0.977080 / 0.992200 | 0.991368 / 0.993662 | 0.999781 / 0.999826 |
| 0.90 | ours | 5.26875648 | 0.900348 / 0.994578 | 0.987788 / 0.994114 | 0.993933 / 0.995729 | 0.999926 / 0.999943 |
| 0.99 | baseline | 10 | 0.994886 / 0.999711 | 0.998891 / 0.999676 | 0.999686 / 0.999782 | 1.000000 / 1.000000 |
| 0.99 | legacy | 0.1 | 0.991161 / 0.999649 | 0.999371 / 0.999780 | 0.999794 / 0.999860 | 1.000000 / 1.000000 |
| 0.99 | ours | 0.987333679 | 0.991123 / 0.998905 | 0.997422 / 0.998990 | 0.998931 / 0.999184 | 1.000000 / 1.000000 |

**With sigma=1e-3: no matching initial noisy-gradient cosine measurements were located in the existing evidence.** The training accuracy or scalar gradient traces cannot supply those missing BPTT/EP angles. Clean cosine passing does not imply noisy cosine passing.

## B. Initial state RMS displacement

Below: pooled RMS((v_plus-v_minus)/2), in simulator voltage units, across all elements in all 36 batches. This centered half-difference separates the odd nudge signal from common continued relaxation. The CSV also supplies both RMS(v_plus-v_free) and RMS(v_minus-v_free).

| Threshold | Scheme | Hidden 1 | Hidden 2 | Hidden 3 | Output |
|---|---|---:|---:|---:|---:|
| 0.90 | baseline | 1.8979e-05 | 8.8856e-05 | 0.0046851 | 4.5576 |
| 0.90 | legacy | 1.3413e-05 | 0.00025094 | 0.052886 | 203.5 |
| 0.90 | ours | 1.697e-06 | 2.4406e-05 | 0.0039176 | 3.8034 |
| 0.99 | baseline | 4.8671e-07 | 2.2782e-06 | 0.00012001 | 0.11277 |
| 0.99 | legacy | 3.1083e-07 | 5.814e-06 | 0.0012244 | 4.6014 |
| 0.99 | ours | 3.2331e-07 | 4.6434e-06 | 0.00074445 | 0.71274 |

For the literal free-to-positive-nudged displacement, pooled RMS(v_plus-v_free):

| Threshold | Scheme | Hidden 1 | Hidden 2 | Hidden 3 | Output |
|---|---|---:|---:|---:|---:|
| 0.90 | baseline | 1.9251e-05 | 9.0303e-05 | 0.0047698 | 4.5576 |
| 0.90 | legacy | 1.3572e-05 | 0.00025392 | 0.053536 | 203.5 |
| 0.90 | ours | 1.7141e-06 | 2.4627e-05 | 0.0039515 | 3.8034 |
| 0.99 | baseline | 5.1579e-07 | 2.2826e-06 | 0.00012006 | 0.11277 |
| 0.99 | legacy | 3.4971e-07 | 5.8299e-06 | 0.0012248 | 4.6014 |
| 0.99 | ours | 3.2646e-07 | 4.6534e-06 | 0.00074563 | 0.71274 |

At identical initial parameters and inputs these **physical displacements are the same with and without read noise by construction**: noise is added only to copied endpoint voltages for gradient readout, after clean relaxation. This does not claim equal states later along noisy training.

The apparent displacement of the noisy readings differs. For independent endpoint noise with sigma=1e-3, sqrt(E[RMS_squared]) for the centered half-difference is sqrt(d_odd^2 + sigma^2/2), giving a 7.071e-4 noise floor. For one noisy positive endpoint against the exact free state it is sqrt(d_plus^2 + sigma^2). These are analytic expectations, not measured noisy displacements; they are included separately in the CSV. No sampled noisy displacement or cosine is fabricated.

## C. Existing accuracy

| Threshold | Scheme | Beta | Clean validation | sigma=1e-3 validation/outcome |
|---|---|---:|---|---|
| 0.90 | baseline | 404.141106 | 97.72% (epoch 30) | 97.16% (epoch 30) |
| 0.90 | legacy | 4.4225011 | 98.70% (epoch 30) | Nonfinite in epoch 8; last 94.76% at epoch 7 |
| 0.90 | ours | 5.26875648 | 98.52% (epoch 30) | Nonfinite in epoch 18; last 95.14% at epoch 17 |
| 0.99 | baseline | 10 | 97.64% (epoch 30) | Not measured |
| 0.99 | legacy | 0.1 | Not measured | Not measured |
| 0.99 | ours | 0.987333679 | Not measured | Not measured |

Additional exact-beta evidence: ours beta0.9 (the original-grid p99 choice) completed ten clean epochs at 98.46%; it is not a training result for beta0.987334 and is not a 30-epoch comparison. The p90 clean/noisy pairs above use RTX5090 controls. Baseline p99 clean is also RTX5090. All are single-seed observations; different scheme-specific learning rates and beta-grid resolution limit causal comparisons.

The p90 hidden-layer phase signal is larger than for the p99 choices, but the first two hidden-layer signals remain below the sigma/sqrt(2) readout floor. This voltage-level comparison does not determine gradient cosine or accuracy, because gradients aggregate products across sites and samples.

[Layer CSV](conv3_p90_p99_existing_20260921_layers.csv) · [Accuracy CSV](conv3_p90_p99_existing_20260921_accuracy.csv) · [Source validation and hashes](conv3_p90_p99_existing_20260921_provenance.json)

Source studies: [calibration](beta_rule_comparison_20260918.md), [refinement](beta_refinement_20260919.md), [p90 noise extension](conv3_p90_read_noise_1em3_20260920.md), [historical baseline](baseline_read_noise_results_20260916.md), [beta0.9 ten-epoch training](layerwise_beta_training_20260919.md).
