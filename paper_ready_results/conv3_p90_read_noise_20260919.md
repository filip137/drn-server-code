# Conv3 p90 beta: thirty-epoch read-noise sweep

Updated 2026-09-20T18:42:05.554237+00:00. **24/24 complete; 24/24 terminal.**

Seed 0, ordinary MNIST with a deterministic 55,000/5,000 train/validation split, T=K=8, float64 centered EqProp. All cases train from fresh matched initialization for 30 epochs. Jean Zay uses three exact ten-epoch continuations; local workers retain their in-memory training state, including scheduling pauses. Official test remains unread; these are validation diagnostics.

Injected beta is fixed from the zero-noise per-matrix cosine >.90 calibration. Gaussian noise perturbs copied positive/negative non-input endpoint voltages only during gradient readout; relaxation and validation are noise-free. Each scheme/GPU pair uses its own matched clean control. Positive drop means lower validation accuracy with noise. Failures remain listed. After the user-requested local acceleration, V100 covers sigma0/1e-5, RTX5090 covers 0/3e-5, and A100 covers0/1e-4/3e-4/5e-4. Matched controls share GPU/PyTorch/CUDA class; physical hosts are recorded in the CSV. There are 15 noisy runs and nine matched clean controls. Fifi/Riri clean workers were temporarily suspended in memory to reduce GPU contention and resumed automatically after their noisy peers finished, preserving their live training states. Matched-control drops remain blank until both outcomes finish.

| GPU | Scheme | Injected beta | Noise sigma | State / epochs | Final / best validation (%) | Final drop (pp) |
|---|---|---:|---:|---|---:|---:|
| A100 | baseline | 404.141 | 0 | complete / 30 | 97.58 / 97.74 | +0.00 |
| V100 | baseline | 404.141 | 0 | complete / 30 | 97.56 / 97.66 | +0.00 |
| RTX5090 | baseline | 404.141 | 0 | complete / 30 | 97.72 / 97.72 | +0.00 |
| V100 | baseline | 404.141 | 1e-05 | complete / 30 | 97.50 / 97.72 | +0.06 |
| RTX5090 | baseline | 404.141 | 3e-05 | complete / 30 | 97.56 / 97.62 | +0.16 |
| A100 | baseline | 404.141 | 0.0001 | complete / 30 | 97.56 / 97.60 | +0.02 |
| A100 | baseline | 404.141 | 0.0003 | complete / 30 | 97.56 / 97.56 | +0.02 |
| A100 | baseline | 404.141 | 0.0005 | complete / 30 | 97.40 / 97.44 | +0.18 |
| A100 | legacy | 4.4225 | 0 | complete / 30 | 98.72 / 98.86 | +0.00 |
| V100 | legacy | 4.4225 | 0 | complete / 30 | 98.76 / 98.88 | +0.00 |
| RTX5090 | legacy | 4.4225 | 0 | complete / 30 | 98.70 / 98.86 | +0.00 |
| V100 | legacy | 4.4225 | 1e-05 | complete / 30 | 98.72 / 98.78 | +0.04 |
| RTX5090 | legacy | 4.4225 | 3e-05 | complete / 30 | 98.70 / 98.72 | +0.00 |
| A100 | legacy | 4.4225 | 0.0001 | complete / 30 | 98.64 / 98.68 | +0.08 |
| A100 | legacy | 4.4225 | 0.0003 | complete / 30 | 96.26 / 98.04 | +2.46 |
| A100 | legacy | 4.4225 | 0.0005 | complete / 30 | 95.72 / 97.58 | +3.00 |
| A100 | ours | 5.26876 | 0 | complete / 30 | 98.50 / 98.58 | +0.00 |
| V100 | ours | 5.26876 | 0 | complete / 30 | 98.58 / 98.58 | +0.00 |
| RTX5090 | ours | 5.26876 | 0 | complete / 30 | 98.52 / 98.62 | +0.00 |
| V100 | ours | 5.26876 | 1e-05 | complete / 30 | 98.38 / 98.38 | +0.20 |
| RTX5090 | ours | 5.26876 | 3e-05 | complete / 30 | 98.12 / 98.24 | +0.40 |
| A100 | ours | 5.26876 | 0.0001 | complete / 30 | 97.86 / 97.86 | +0.64 |
| A100 | ours | 5.26876 | 0.0003 | complete / 30 | 97.48 / 97.48 | +1.02 |
| A100 | ours | 5.26876 | 0.0005 | complete / 30 | 97.12 / 97.16 | +1.38 |

Completion means all thirty finite epochs were collected and validated. The separate predeclared stability screen requires epoch30 accuracy to finish strictly less than 5 percentage points below that run's own best; passing this screen does not establish robustness to noise.

Final-drop screen: 24/24 completed runs pass. The CSV records each run's best-to-final drop.

A final-epoch screen can miss temporary deterioration. The CSV also records the largest decline from any earlier validation best within each completed run, and the first epoch attaining it. The largest observed decline is 9.16pp for legacy on A100 at σ=0.0005, epoch 20; its final best-to-final drop is 1.86pp. This is a descriptive trajectory diagnostic, not a new selection or exclusion rule.

Previous thirty-epoch validation curves use baseline beta 10, ours beta 3 and legacy beta 0.001. They ran on the recorded RTX 3080/3090/5090 environments. Differences against the new curves combine the beta change with environment changes; they are not a controlled beta-only effect.

| Scheme | Old injected beta | Sigma | Old final validation (%) | Old drop (pp) | New minus old final (pp) |
|---|---:|---:|---:|---:|---:|
| baseline | 10 | 1e-05 | 97.48 | 0.16 | +0.02 |
| baseline | 10 | 3e-05 | 97.04 | 0.60 | +0.52 |
| baseline | 10 | 0.0001 | 96.64 | 1.00 | +0.92 |
| baseline | 10 | 0.0003 | 96.48 | 1.16 | +1.08 |
| baseline | 10 | 0.0005 | 96.04 | 1.60 | +1.36 |
| legacy | 0.001 | 1e-05 | 95.68 | 3.10 | +3.04 |
| legacy | 0.001 | 3e-05 | 93.42 | 5.36 | +5.28 |
| legacy | 0.001 | 0.0001 | 83.08 | 15.70 | +15.56 |
| legacy | 0.001 | 0.0003 | 82.34 | 16.44 | +13.92 |
| legacy | 0.001 | 0.0005 | 75.44 | 23.34 | +20.28 |
| ours | 3 | 1e-05 | 98.26 | 0.38 | +0.12 |
| ours | 3 | 3e-05 | 97.82 | 0.82 | +0.30 |
| ours | 3 | 0.0001 | 97.62 | 1.02 | +0.24 |
| ours | 3 | 0.0003 | 96.82 | 1.82 | +0.66 |
| ours | 3 | 0.0005 | 96.60 | 2.04 | +0.52 |

Single-seed differences are descriptive. The prior ten-epoch clean pilots qualify admission only; they are not controls for this full-horizon sweep.

[Epoch-by-epoch validation curves](conv3_p90_read_noise_20260919_epochs.png) cover all thirty epochs for every declared case. The tables above use completed thirty-epoch outcomes only.

[Plan](../docs/eqprop_conv3_p90_read_noise_plan_20260919.md) · [CSV](conv3_p90_read_noise_20260919.csv) · [Local evidence](../results/eqprop-conv3-p90-read-noise-20260919-v1/)
