# Conv3 p90: final-checkpoint gradient cosine versus read noise

Measured by read-only replay of 25 existing epoch-30 checkpoints; no training or official-test evaluation. Seed 0; beta baseline404.141105702, legacy4.42250110273, ours5.26875648112; T=K=8, float64 centered frozen-current EP. The BPTT reference differentiates exactly K zero-nudge iterations from the same post-T state. Gradients use the training amplification normalization, with no Adam or LR transformation.

Every checkpoint uses the same 36 validation batches of16. At nonzero sigma, four independent endpoint-noise draws per batch give144 comparisons per weight matrix; zero noise gives36. Standard-normal draws are matched across cases, while positive/negative phases and non-input layers use independent draws. Noise changes endpoint readout only. Checkpoint bytes, native float64 loading against the saved NPZ, zero biases and unchanged parameters are verified. All replays use the local RTX3090.

## Per-layer median cosine: noisy EP versus clean BPTT

| Training σ | Scheme | Training GPU | Conv1 | Conv2 | Conv3 | Dense |
|---:|---|---|---:|---:|---:|---:|
| 0 | baseline | RTX5090 | 0.999930 | 0.999824 | 0.999465 | 1.000000 |
| 0 | legacy | RTX5090 | 0.999819 | 0.999164 | 0.997514 | 0.999999 |
| 0 | ours | RTX5090 | 0.999992 | 0.999969 | 0.999904 | 1.000000 |
| 1e-05 | baseline | V100 | 0.985441 | 0.996814 | 0.999482 | 1.000000 |
| 1e-05 | legacy | V100 | 0.602313 | 0.869716 | 0.997049 | 0.999999 |
| 1e-05 | ours | V100 | 0.258183 | 0.414092 | 0.997959 | 1.000000 |
| 3e-05 | baseline | RTX5090 | 0.892144 | 0.981224 | 0.999394 | 1.000000 |
| 3e-05 | legacy | RTX5090 | 0.253928 | 0.594767 | 0.993103 | 0.999994 |
| 3e-05 | ours | RTX5090 | 0.092419 | 0.223305 | 0.994888 | 1.000000 |
| 0.0001 | baseline | A100 | 0.582660 | 0.846154 | 0.998410 | 0.999999 |
| 0.0001 | legacy | A100 | 0.136892 | 0.351693 | 0.983437 | 0.999908 |
| 0.0001 | ours | A100 | 0.058012 | 0.118870 | 0.978738 | 0.999999 |
| 0.0003 | baseline | A100 | 0.273776 | 0.562762 | 0.992725 | 0.999994 |
| 0.0003 | legacy | A100 | 0.020100 | 0.167413 | 0.053566 | 0.097676 |
| 0.0003 | ours | A100 | 0.021539 | 0.048112 | 0.917814 | 0.999995 |
| 0.0005 | baseline | A100 | 0.236215 | 0.457898 | 0.985971 | 0.999985 |
| 0.0005 | legacy | A100 | 0.001441 | 0.083346 | -0.003131 | 0.082845 |
| 0.0005 | ours | A100 | 0.020070 | 0.025777 | 0.245687 | -0.293154 |
| 0.001 | baseline | RTX5090 | 0.179658 | 0.357652 | 0.971671 | 0.999960 |
| 0.001 | legacy | RTX5090 | — | — | — | — |
| 0.001 | ours | RTX5090 | — | — | — | — |

Legacy and ours at sigma1e-3 became nonfinite during epochs8 and18 respectively. They have no epoch-30 checkpoint, so no end-of-training gradient is substituted from a best or earlier checkpoint.

## Clean EP at the same trained checkpoints

| Training σ | Scheme | Conv1 | Conv2 | Conv3 | Dense |
|---:|---|---:|---:|---:|---:|
| 0 | baseline | 0.999930 | 0.999824 | 0.999465 | 1.000000 |
| 0 | legacy | 0.999819 | 0.999164 | 0.997514 | 0.999999 |
| 0 | ours | 0.999992 | 0.999969 | 0.999904 | 1.000000 |
| 1e-05 | baseline | 0.999902 | 0.999858 | 0.999498 | 1.000000 |
| 1e-05 | legacy | 0.999784 | 0.998942 | 0.997081 | 0.999999 |
| 1e-05 | ours | 0.999942 | 0.999743 | 0.999154 | 1.000000 |
| 3e-05 | baseline | 0.999892 | 0.999817 | 0.999460 | 1.000000 |
| 3e-05 | legacy | 0.999447 | 0.997961 | 0.993185 | 0.999994 |
| 3e-05 | ours | 0.999806 | 0.999484 | 0.998072 | 1.000000 |
| 0.0001 | baseline | 0.999925 | 0.999818 | 0.999362 | 0.999999 |
| 0.0001 | legacy | 0.998582 | 0.996051 | 0.983894 | 0.999908 |
| 0.0001 | ours | 0.999760 | 0.999348 | 0.997841 | 0.999999 |
| 0.0003 | baseline | 0.999829 | 0.999776 | 0.999326 | 1.000000 |
| 0.0003 | legacy | 0.619609 | 0.666308 | 0.053574 | 0.097677 |
| 0.0003 | ours | 0.999744 | 0.998854 | 0.997196 | 0.999999 |
| 0.0005 | baseline | 0.999790 | 0.999684 | 0.998580 | 1.000000 |
| 0.0005 | legacy | -0.165296 | 0.171045 | -0.003132 | 0.082845 |
| 0.0005 | ours | 0.723292 | 0.688886 | 0.245614 | -0.293156 |
| 0.001 | baseline | 0.999753 | 0.999544 | 0.998677 | 1.000000 |

These are medians, not all-batch minima. The CSV includes minima, p10/p90, norm ratios, relative errors, gradient RMS and defined-cosine counts. Zero directions remain undefined. The original beta qualification tested initialization and a saved BPTT checkpoint, not these final EP-trained checkpoints. Its >.90 threshold is not guaranteed along training.

Training noise changes the learned weights; thus the solid curves combine trajectory changes and readout noise. The dotted controls isolate readout noise at each fixed checkpoint. Training used V100 at1e-5, RTX5090 at3e-5/1e-3, and A100 at1e-4/3e-4/5e-4. The displayed zero point is RTX5090; all nine GPU-specific clean controls remain in the CSV. Lines guide the eye and do not establish a causal hardware-independent noise curve. Four draws quantify only limited Monte Carlo variability. Finite T/K residuals are reported separately; this is not an exact-equilibrium gradient guarantee.

![Cosine versus noise](conv3_p90_final_noise_cosine_20260921.png)

[Summary CSV](conv3_p90_final_noise_cosine_20260921_summary.csv) · [PDF](conv3_p90_final_noise_cosine_20260921.pdf) · [Source bundles and raw replay](../results/eqprop-conv3-p90-final-noise-cosine-20260921-v1/)
