# Conv3 baseline: initialization versus trained checkpoint

Completed 2026-09-18. Outcome: **the early-layer near-zero noisy-gradient cosine is already present at initialization**. The proposed explanation that this appears mainly because gradients shrink late in training is not supported for the first two convolution layers.

[Cosine comparison JPG](figures/section43_conv3_baseline_initialization_cosine.jpg) · [PDF](figures/section43_conv3_baseline_initialization_cosine.pdf) · [Phase contrast and gradient strength JPG](figures/section43_conv3_baseline_initialization_signal.jpg) · [PDF](figures/section43_conv3_baseline_initialization_signal.pdf). Additional [signed mean/RMS voltage diagnostics](figures/section43_conv3_baseline_initialization_voltages.pdf) and CSVs are retained. All figures use matplotlib; PDF/JPG/PNG/SVG formats are available.

At sigma=5e-4, mean per-minibatch/per-noise-draw cosine against the matched noiseless BPTT reference is:

| Weight layer | Initialization, epoch 0 | Trained, epoch 30 |
|---|---:|---:|
| First convolution | 0.005745 | 0.003998 |
| Second convolution | 0.004053 | 0.004179 |
| Third convolution | 0.306445 | 0.099958 |
| Readout | 0.999882 | 0.961369 |

At the lowest nonzero noise level, sigma=1e-5, the first two convolution means are 0.1626/0.2676 at initialization and 0.1482/0.3210 after training. Without noise, all initialization layerwise mean cosines exceed 0.9989. Thus the poor noisy alignment is not an intrinsic clean-EP direction error or a two-decimal plotting artifact.

## Signal comparison

| State layer | Centered contrast RMS, initialization | Centered contrast RMS, trained | Trained / initial |
|---|---:|---:|---:|
| H1 | 4.867051e-07 | 4.073277e-07 | 0.8369 |
| H2 | 2.278202e-06 | 2.119449e-06 | 0.9303 |
| H3 | 1.200127e-04 | 3.970799e-05 | 0.3309 |
| Output | 1.127735e-01 | 1.845394e-03 | 0.0164 |

The first two hidden-layer phase signals are already small at initialization and change only modestly. H1 contrast declines by about 16%, whereas H3 and output contrast decline by approximately 3.02× and 61.11×. The noise band in the figure shows sigma/sqrt(2), a voltage-contrast noise scale, not an exact gradient SNR.

| Weight layer | Mean clean BPTT L2, initialization | Mean clean BPTT L2, trained | Ratio of means, trained / initial |
|---|---:|---:|---:|
| First convolution | 1.485343e-03 | 1.510699e-03 | 1.017 |
| Second convolution | 7.622751e-04 | 8.602800e-03 | 11.286 |
| Third convolution | 1.426067e-03 | 9.830819e-02 | 68.937 |
| Readout | 4.740498e-02 | 3.292028e-02 | 0.694 |

The first-layer clean gradient norm is approximately unchanged; the second and third increase. These observations rule out late-stage gradient shrinkage as the main explanation of the first two near-zero cosines. They do not establish that those layers lack useful accumulated learning: cosine is measured for individual raw minibatch gradient estimates, whereas training uses repeated Adam updates and conductance projection. The third convolution and readout retain substantially greater alignment at initialization. Whether later-layer learning, accumulation of a weak mean signal, or early-feature robustness explains the high noisy-training accuracy requires an additional controlled test.

## Matched protocol and limits

- Conv3 baseline only, seed 0, same beta_inj=10 and T=K=8 as the trained comparison. Native float64 centered frozen-current EP; perfect diode; fixed zero biases; [0,100] conductances; unchanged source preprocessing.
- Saved initialization at epoch 0 is loaded from the exact checkpoint named in the training config. Its float32 training fingerprint is verified; exact float64 promotion also reproduces the initial-parameter fingerprint recorded by training. No new random initialization substitutes for the saved one.
- The same 36 ordered validation minibatches of 16 examples (576 total) and identical noise draws as the prior trained replay. Six sigma levels 0,1e-5,3e-5,1e-4,3e-4,5e-4; eight draws per nonzero level, independent positive/negative endpoint noise. Input reads remain noiseless.
- Trained noisy measurements are reused from the validated earlier replay; all 36 trained clean phase/BPTT computations were repeated and matched every stored phase-state hash and gradient metric exactly.
- We compare actual epoch 0 and maximum-validation epoch 30 checkpoints. Source metadata records checkpoint_every_epoch=false and zero intermediate checkpoint files. No epoch 1–29 gradient trajectory is inferred. This follow-up does not compare all architectures or amplification schemes.
- No optimizer steps or official-test reads. The 576 examples are validation mechanism evidence, not an additional accuracy evaluation.
- All initialization projected-residual p90 checks pass the 0.01 gate at the free and subsequent endpoints. The trained baseline Conv3 free-state caveat persists: 72/144 batch-layer post-T checks fail, maximum p90 0.26808. Subsequent endpoints pass. The BPTT reference remains finite-K, not a convergence certificate.
- The CSVs retain cosine medians/p10/p90, norms, RMS, exact-zero and abs(g)<=1e-12 fractions, relative gradient scale and direction across checkpoints, and count-weighted signed voltage mean/RMS/std/min/max/clamp occupancy. Within-batch/draw spread is not uncertainty over training seeds.

## Provenance and completion

Initialization source:

```text
/home/filip/server_code_conv_learning_rate_protocol/.codex/worktrees/lean-experiment-workflow/results/paper-training-completion-20260911-v1/assets/wide_kaiming/conv3/seed0.pt
```

Trained checkpoint source:

```text
/home/filip/server_code_conv_learning_rate_protocol/.codex/worktrees/lean-experiment-workflow/paper_ready_results/bundles/baseline_read_noise_20260916/conv3/beta_10/sigma_0/seed0/best_model.pt
```

All 36 initialization batches completed: 5,904 new gradient comparisons, 24 initialization cosine cells, and 36 repeated trained clean controls. The 164 smoke comparisons exactly match the full run. Four relevant canonical bundles validate. The saved checkpoints remain unchanged. One regression test passes; the initial pre-GPU smoke failure compared two differently prefixed hash schemas, was corrected and regression-tested, and its log is retained. No scientific cases were excluded. Total local RTX 3090 use including the successful smoke: 208.707 seconds (3.4784 GPU-minutes), below the 600-second cap.

- [Cosine summary CSV](section43_conv3_baseline_initialization_cosine.csv), [displacement CSV](section43_conv3_baseline_initialization_displacement.csv), [gradient statistics CSV](section43_conv3_baseline_initialization_gradients.csv), [voltage statistics CSV](section43_conv3_baseline_initialization_voltages.csv).
- [Verification](section43_conv3_baseline_initialization_verification.json), [configuration](section43_initialization_config.json), [figure provenance](section43_conv3_baseline_initialization_provenance.json).
- [Canonical full run](../results/section43-conv3-baseline-initialization-20260918-v1/full/), [runner](run_section43_initialization.py), [plot generator](plot_section43_initialization.py), [regression test](test_section43_initialization.py).

For the paper, the cosine heatmap should be described as a layerwise instantaneous-gradient noise diagnostic. The initialization result strengthens the warning against treating near-zero cosine as a direct prediction of final classification accuracy.
