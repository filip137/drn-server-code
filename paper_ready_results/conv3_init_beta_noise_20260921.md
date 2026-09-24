# Conv3 initialization: beta and endpoint read noise

Read-only diagnostic requested September21,2026. All six cases completed at the identical saved seed-0 initialization (epoch0). No trained checkpoint, optimizer step, accuracy evaluation or official-test access.

Injected beta B is0.01/0.1 for baseline(v1/c1),legacy(v4/c0.25),ours(v4/c1). Base beta=B/(v/c)^3. T=K=8; centered frozen-current EP; float64; explicit perfect diodes; input gain360; zero biases; wide[0,100] weights. The initializer was stored in float32 and promoted exactly to float64, matching the prior initialization replay. The BPTT reference differentiates K zero-nudge steps from the same post-T state.

The exact prior36 batches of16 validation examples are reused (576 examples from the55k/5k MNIST split). One endpoint-noise draw per batch/sigma uses seed2026092101, matched across schemes and beta; the same underlying standard normals are reused across sigma. Noise affects copied endpoint voltages only; relaxation and inputs are clean. Positive/negative phases and state layers have independent samples.

![Gradient cosine](conv3_init_beta_noise_20260921.png)

## Per-layer median noisy EP–BPTT cosine

| Injected beta | Scheme | Readout sigma | Conv1 | Conv2 | Conv3 | Dense |
|---:|---|---:|---:|---:|---:|---:|
| 0.01 | baseline | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 0.01 | legacy | 0 | 0.999982 | 0.999980 | 0.999986 | 1.000000 |
| 0.01 | ours | 0 | 0.999993 | 0.999991 | 0.999992 | 1.000000 |
| 0.01 | baseline | 1e-05 | -0.049824 | 0.008126 | 0.018007 | 0.998347 |
| 0.01 | legacy | 1e-05 | -0.027079 | 0.025639 | 0.984603 | 1.000000 |
| 0.01 | ours | 1e-05 | -0.053399 | 0.000564 | 0.335984 | 0.999996 |
| 0.01 | baseline | 3e-05 | -0.049897 | 0.007971 | 0.007169 | 0.985088 |
| 0.01 | legacy | 3e-05 | -0.032243 | 0.010825 | 0.883062 | 1.000000 |
| 0.01 | ours | 3e-05 | -0.053871 | -0.000687 | 0.117786 | 0.999967 |
| 0.01 | baseline | 0.0001 | -0.049922 | 0.007917 | 0.003376 | 0.863775 |
| 0.01 | legacy | 0.0001 | -0.034372 | 0.004608 | 0.492779 | 1.000000 |
| 0.01 | ours | 0.0001 | -0.054036 | -0.001124 | 0.035901 | 0.999632 |
| 0.01 | baseline | 0.0003 | -0.049929 | 0.007901 | 0.002295 | 0.505646 |
| 0.01 | legacy | 0.0003 | -0.034980 | 0.002832 | 0.185324 | 0.999999 |
| 0.01 | ours | 0.0003 | -0.054084 | -0.001248 | 0.013154 | 0.996685 |
| 0.01 | baseline | 0.0005 | -0.049931 | 0.007898 | 0.002081 | 0.328456 |
| 0.01 | legacy | 0.0005 | -0.035102 | 0.002476 | 0.112507 | 0.999997 |
| 0.01 | ours | 0.0005 | -0.054093 | -0.001271 | 0.008058 | 0.990855 |
| 0.01 | baseline | 0.001 | -0.049932 | 0.007896 | 0.001927 | 0.163153 |
| 0.01 | legacy | 0.001 | -0.035193 | 0.002209 | 0.056536 | 0.999989 |
| 0.01 | ours | 0.001 | -0.054100 | -0.001288 | 0.004555 | 0.965262 |
| 0.1 | baseline | 0 | 0.999998 | 0.999998 | 0.999998 | 1.000000 |
| 0.1 | legacy | 0 | 0.999649 | 0.999780 | 0.999860 | 1.000000 |
| 0.1 | ours | 0 | 0.999892 | 0.999898 | 0.999916 | 1.000000 |
| 0.1 | baseline | 1e-05 | -0.048942 | 0.010219 | 0.163659 | 0.999983 |
| 0.1 | legacy | 1e-05 | 0.058268 | 0.226320 | 0.999695 | 1.000000 |
| 0.1 | ours | 1e-05 | -0.046607 | 0.017786 | 0.962394 | 1.000000 |
| 0.1 | baseline | 3e-05 | -0.049572 | 0.008669 | 0.054050 | 0.999851 |
| 0.1 | legacy | 3e-05 | -0.007366 | 0.075200 | 0.998409 | 1.000000 |
| 0.1 | ours | 3e-05 | -0.051757 | 0.005083 | 0.764339 | 1.000000 |
| 0.1 | baseline | 0.0001 | -0.049825 | 0.008126 | 0.018008 | 0.998342 |
| 0.1 | legacy | 0.0001 | -0.027011 | 0.025577 | 0.984408 | 1.000000 |
| 0.1 | ours | 0.0001 | -0.053402 | 0.000564 | 0.335767 | 0.999995 |
| 0.1 | baseline | 0.0003 | -0.049897 | 0.007971 | 0.007173 | 0.985045 |
| 0.1 | legacy | 0.0003 | -0.032197 | 0.010808 | 0.882805 | 1.000000 |
| 0.1 | ours | 0.0003 | -0.053872 | -0.000686 | 0.117780 | 0.999959 |
| 0.1 | baseline | 0.0005 | -0.049911 | 0.007940 | 0.005008 | 0.959379 |
| 0.1 | legacy | 0.0005 | -0.033415 | 0.007262 | 0.748773 | 1.000000 |
| 0.1 | ours | 0.0005 | -0.053966 | -0.000936 | 0.070946 | 0.999886 |
| 0.1 | baseline | 0.001 | -0.049922 | 0.007917 | 0.003390 | 0.863322 |
| 0.1 | legacy | 0.001 | -0.034350 | 0.004602 | 0.492609 | 0.999999 |
| 0.1 | ours | 0.001 | -0.054037 | -0.001123 | 0.035919 | 0.999543 |

## Physical centered phase response at initialization

Pooled RMS((v_plus-v_minus)/2) in simulator voltage units. These physical states are unchanged across the readout-noise sweep.

| Injected beta | Scheme | Hidden1 | Hidden2 | Hidden3 | Output |
|---:|---|---:|---:|---:|---:|
| 0.01 | baseline | 4.87244e-10 | 2.2805e-09 | 1.20114e-07 | 0.000112774 |
| 0.01 | legacy | 3.11016e-08 | 5.81723e-07 | 0.000122504 | 0.460139 |
| 0.01 | ours | 3.28744e-09 | 4.71941e-08 | 7.56349e-06 | 0.00721886 |
| 0.1 | baseline | 4.87212e-09 | 2.28045e-08 | 1.20113e-06 | 0.00112774 |
| 0.1 | legacy | 3.10829e-07 | 5.81399e-06 | 0.00122445 | 4.60139 |
| 0.1 | ours | 3.28561e-08 | 4.71742e-07 | 7.56126e-05 | 0.0721886 |

## Coverage and limits

All6 bundles validate, with6048 comparisons and10368 endpoint-noise tensor draws. Checkpoint bytes, in-memory parameters, zero biases, shared cohorts, BPTT invariance across beta and matched noise samples pass. Replay including smoke took10.56 minutes on one local RTX3090.

Undefined cosines: 0. Residual failures: 0/3456 batch/layer/phase rows. Residuals are separate from gradient fidelity; finite-T/K BPTT is not an exact-equilibrium guarantee.

Medians and10–90% ranges summarize batches with one draw each; they are not confidence intervals or within-batch Monte Carlo variance estimates. This is one initializer and two beta values, not a training-performance or universal scheme-ranking result. Equal injected beta does not equalize physical output displacement. No accuracy result is inferred.

Initializer SHA256: `5e5782bd9bf166b8a432cbf283d745ffe4d25fe4a69ed656f14f9e7e9407392f`. Cohort SHA256: `95af3eebd9416ec643b7d698d061b7dac3fe0d52311dd888293609bada629acd`.

[Layer summary CSV](conv3_init_beta_noise_20260921_summary.csv) · [PDF](conv3_init_beta_noise_20260921.pdf) · [Readout-only figure](conv3_init_beta_noise_20260921_readout_only.png) · [Voltage response CSV](conv3_init_beta_noise_20260921_state_signals.csv) · [Raw bundles](../results/eqprop-conv3-init-beta-noise-20260921-v1/)
