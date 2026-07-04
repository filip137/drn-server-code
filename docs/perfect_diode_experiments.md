# Perfect-Diode Experiments

Updated: 2026-07-03 CEST.

## Current Takeaways

- The completed perfect-diode comparison used the newer stride/padding variant:
  Conv1 strides/paddings `[2]`/`[1]`, Conv2 `[2, 2]`/`[1, 1]`, Conv3
  `[2, 2, 1]`/`[1, 1, 1]`.
- No matching Jean Zay jobs remain in the queue as of this update.
- Conv2 still favors `v4/c1`: the repeat gives `v4/c1` best/final
  `0.9835 / 0.9834`, while the tuned `v1/c1` 50-epoch rescue reaches
  `0.9787 / 0.9777`.
- New high-K Conv2 reruns change how this should be reported: the best
  `v4/c1` row is a source-`K=4`/transient-gradient result. Training the same
  row at settled higher K (`K=16` or `K=64`) leaves `ConvWeight_0` and
  `ConvWeight_1` gradients exactly zero from initialization and drops accuracy
  to about `90%`, while the dense/readout gradient remains nonzero.
- Conv3 now favors tuned `v1/c1`: the rescue selected gain `480`, LR `0.0864`,
  and the 50-epoch run reaches `0.9741 / 0.9738`, above the `v4/c1` repeat at
  `0.9671 / 0.9665`.
- The pruning concern is real in the completed grid: `39/60` calibration rows
  pruned, including `26/30` Conv3 rows. Selection should be read as selection
  among surviving fast-starting rows, not a proof that all lower-LR rows are bad.
- Follow-up Conv3 investigation job `1170774` is running on Jean Zay R3 to test
  seed stability and no-prune lower-LR alternatives.
- Conv2/Conv3 `v1/c1` massive input-gain LR30 grid job `1175141` is pending on
  Jean Zay R3, with collector `1175142` dependent on successful completion.

## Conv1/2/3 Repeat: v1/c1 and v4/c1

Root:
`/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv123_s221_pad1_perfect_diode_v1c1_v4c1_bestparams_k4k8_seed0_10_30_50epoch`

Jean Zay job: `1114636`, R3 account `fmu@v100`.

| Depth | Amp | Epochs | K | Input gain | LR | Status | Best test acc | Final test acc | Best epoch |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Conv1 | `v1/c1` | 10 | 4 | 40 | 0.006 | complete | 0.9655 | 0.9641 | 9 |
| Conv1 | `v4/c1` | 10 | 4 | 40 | 0.009 | complete | 0.9676 | 0.9667 | 7 |
| Conv2 | `v1/c1` | 30 | 4 | 100 | 0.024 | complete | 0.9709 | 0.9708 | 26 |
| Conv2 | `v4/c1` | 30 | 4 | 100 | 0.012 | complete | 0.9835 | 0.9834 | 24 |
| Conv3 | `v1/c1` | 50 | 8 | 360 | 0.0432 | complete | 0.9731 | 0.9727 | 49 |
| Conv3 | `v4/c1` | 50 | 8 | 360 | 0.0864 | complete | 0.9671 | 0.9665 | 49 |

Current count from the root: `6/6` metrics, `0` pruned markers.

## Loss Plateau Check

Last-five test losses from the completed/current logs showed late noisy
plateaus rather than clean divergence:

| Run | Last-five loss trend |
| --- | --- |
| Conv1 `v1/c1` | `0.08866, 0.09037, 0.08628, 0.08586, 0.08740` |
| Conv1 `v4/c1` | `0.08379, 0.08555, 0.08144, 0.08063, 0.08652` |
| Conv2 `v1/c1` | `0.06843, 0.07600, 0.07103, 0.07130, 0.06822` |
| Conv2 `v4/c1` | `0.04805, 0.05320, 0.04806, 0.04753, 0.04623` |
| Conv3 `v1/c1` live sample | `0.0561, 0.0562, 0.0570, 0.0552, 0.0551` |
| Conv3 `v4/c1` | `0.06376, 0.06408, 0.06285, 0.06279, 0.06249` |

## Conv2/Conv3 v1/c1 Gain/LR Rescue Sweep

Goal: test whether Conv2/Conv3 `v1/c1` is underperforming because the previous
gain/LR was poorly matched, with particular attention to lower learning rates.

Grid root:
`/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv23_s221_pad1_perfect_diode_v1c1_gain_lr_downbias_k4k8_seed0_10epoch`

Final selected-run root:
`/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv23_s221_pad1_perfect_diode_v1c1_gain_lr_downbias_bestlr_k4k8_seed0_30_50epoch`

Jobs:

- Grid: `1131906`, completed.
- Collector: `1131907`, completed.
- 30/50-epoch follow-up: `1131908`, completed.

Grid settings:

| Depth | Amp | Epochs | K | Gains | LR grid | Prune threshold |
| --- | --- | ---: | ---: | --- | --- | ---: |
| Conv2 | `v1/c1` | 10 | 4 | `100, 140, 200, 280, 400` | `0.0015, 0.003, 0.006, 0.012, 0.024, 0.048` | 0.9537 |
| Conv3 | `v1/c1` | 10 | 8 | `360, 480, 640, 900, 1200` | `0.0027, 0.0054, 0.0108, 0.0216, 0.0432, 0.0864` | 0.8921 |

Completed grid counts:

| Depth | Rows | Pruned | Survivors |
| --- | ---: | ---: | ---: |
| Conv2 | 30 | 13 | 17 |
| Conv3 | 30 | 26 | 4 |
| Total | 60 | 39 | 21 |

Selected 10-epoch rows:

| Depth | Selected gain | Selected LR | 10-epoch best/final | Best epoch | Candidates |
| --- | ---: | ---: | ---: | ---: | ---: |
| Conv2 | 400 | 0.012 | 0.9753 / 0.9731 | 8 | 17 |
| Conv3 | 480 | 0.0864 | 0.9703 / 0.9703 | 10 | 4 |

Top surviving grid rows:

| Depth | Gain | LR | Best/final | Best epoch |
| --- | ---: | ---: | ---: | ---: |
| Conv2 | 400 | 0.012 | 0.9753 / 0.9731 | 8 |
| Conv2 | 400 | 0.024 | 0.9748 / 0.9733 | 8 |
| Conv2 | 280 | 0.048 | 0.9739 / 0.9733 | 7 |
| Conv3 | 480 | 0.0864 | 0.9703 / 0.9703 | 10 |
| Conv3 | 360 | 0.0864 | 0.9629 / 0.9629 | 10 |
| Conv3 | 360 | 0.0432 | 0.9556 / 0.9556 | 10 |

30/50-epoch follow-up:

| Depth | Epochs | Gain | LR | Best test acc | Final test acc | Best epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Conv2 | 30 | 400 | 0.012 | 0.9782 | 0.9780 | 21 |
| Conv2 | 50 | 400 | 0.012 | 0.9787 | 0.9777 | 47 |
| Conv3 | 30 | 480 | 0.0864 | 0.9729 | 0.9717 | 26 |
| Conv3 | 50 | 480 | 0.0864 | 0.9741 | 0.9738 | 45 |

Current count from the grid root: `60/60` metrics, `39` pruned markers.
Current count from the final root: `4/4` metrics, `0` pruned markers.

## Previous Conv3 Perfect-Diode Per-Amp Follow-Up

Root:
`/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv3_pad1_perfect_diode_nonlegacy_peramp_bestlr_k8_seed0_30epoch`

Job `1053392_0` timed out at the 20-hour walltime limit before writing final
metrics. Current count from this root: `0/3` metrics, `0` pruned markers.

## Pruning Assessment

Pruning is useful here as a compute-saving gate, but it is scientifically
dangerous if we interpret pruned rows as evidence that a setting cannot work.
The main risk is exactly the one we are testing: lower learning rates can be
below the old baseline after 5 epochs and still catch up later. A prune rule
based on epoch-5 accuracy therefore biases selection toward faster-starting
settings and against lower-LR runs.

The current implementation is somewhat conservative because it uses best test
accuracy seen by epoch 5, not just the exact epoch-5 value, and because pruned
rows are marked explicitly and excluded from automatic selection. Still, for
interpretation, pruned rows should be treated as "not worth spending this grid's
budget on" rather than "worse model." After the grid finishes, inspect pruned
rows with falling loss or close-to-threshold accuracy, especially the lowest-LR
high-gain cases, and rerun the interesting ones without pruning if the selected
result looks sensitive.

The finished grid confirms the practical impact: Conv3 selection had only four
unpruned candidates, and the selected LR is the highest tested LR. That is a
useful result, but it does not fully test the hypothesis that a lower LR might
catch up given more than five epochs.

## Conv3 Follow-Up Investigation

Launched: 2026-07-01 11:34 CEST.

Job: `1170774`, `mnist_c3_pd_inv`, Jean Zay R3 `fmu@v100`, array `0-6%7`.
Early health check: all seven tasks were running after launch; logs confirm
Conv3 shapes `[2,28,28] -> [64,14,14] -> [128,7,7] -> [256,7,7] -> [20]` with
strides `[2,2,1]`, paddings `[1,1,1]`.

Output root:
`/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv3_s221_pad1_perfect_diode_investigate_v1v4_k8_seed0_1_2_50epoch`

Rationale from the completed grid:

- Selected row `gain=480`, LR `0.0864` learned fastest and reached `97.03%` at
  10 epochs, then `97.41% / 97.38%` at 50 epochs.
- The selected 50-epoch test loss kept improving late
  (`0.0550 -> 0.0528` over the last 10 epochs), while accuracy was already near
  a plateau.
- `gain=480`, LR `0.0432` was pruned at epoch 5 at only `80.38%`, but its
  accuracy and loss were still improving monotonically. This is the cleanest
  test of whether pruning hid a slower lower-LR row.
- High gains `900` and `1200` peaked very early around `81-83%` and then lost
  accuracy despite falling loss; these look more like overdriven/poor
  operating-point rows than slow learners.
- Hard-sigmoid stride `[2,2,1]` `v_off` control rows completed lower:
  `v1/c1` fixed/trainable `94.63% / 94.44%`, `v4/c1` fixed/trainable
  `95.90% / 95.93%`. Trainable `v_off` did not rescue Conv3 in that control.

Investigation rows, all Conv3, strides `[2,2,1]`, paddings `[1,1,1]`, `K=8`,
50 epochs, no pruning:

| Task | Purpose | Run | Seed | Gain | LR |
| ---: | --- | --- | ---: | ---: | ---: |
| 0 | seed check for selected row | `v1/c1` | 1 | 480 | 0.0864 |
| 1 | seed check for selected row | `v1/c1` | 2 | 480 | 0.0864 |
| 2 | seed check for comparator | `v4/c1` | 1 | 360 | 0.0864 |
| 3 | seed check for comparator | `v4/c1` | 2 | 360 | 0.0864 |
| 4 | no-prune lower-LR follow-up | `v1/c1` | 0 | 480 | 0.0432 |
| 5 | no-prune old-gain faster-LR follow-up | `v1/c1` | 0 | 360 | 0.0864 |
| 6 | no-prune high-gain survivor follow-up | `v1/c1` | 0 | 640 | 0.0864 |

## Conv2/Conv3 Massive-Gain LR30 Grid

Launched: 2026-07-01 14:16 CEST.

Jobs:

- Grid: `1175141`, `mnist_c23_pd_mgain`, Jean Zay R3 `fmu@v100`, array
  `0-39%12`.
- Collector: `1175142`, `mnist_c23_pd_mgcol`, dependency `afterok:1175141`.

Output root:
`/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv23_s221_pad1_perfect_diode_v1c1_massive_gain_lr30_k4k8_seed0_30epoch`

Purpose: directly optimize large raw input gains on 30-epoch training curves for
Conv2 and Conv3 `v1/c1`, without pruning.

LR scaling rule:

`lr = reference_gain * reference_lr / input_gain * lr_multiplier`

Reference points:

- Conv2: reference gain `400`, reference LR `0.012`.
- Conv3: reference gain `480`, reference LR `0.0864`.

Grid:

| Depth | Gains | LR multipliers | Epochs | K | Strides | Paddings |
| --- | --- | --- | ---: | ---: | --- | --- |
| Conv2 | `800, 1200, 1600, 2400, 3200` | `0.25, 0.5, 1, 2` | 30 | 4 | `2 2` | `1 1` |
| Conv3 | `960, 1440, 1920, 2880, 3840` | `0.25, 0.5, 1, 2` | 30 | 8 | `2 2 1` | `1 1 1` |

Expected metrics: `40/40`. The collector will write `grid_results.csv` and
`selected_gain_lr_by_conv_depth.csv`.

## Conv2 v4/c1 vs v1/c1 GPU Saturation Analysis

GPU analysis job: `1175610`, Jean Zay R3 dev QoS, completed successfully on
2026-07-01. A redundant pending t3 copy, `1175598`, was canceled after the dev
job wrote the outputs.

Output root:
`/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/conv2_perfect_diode_v4_vs_v1_saturation_gpu_20260701`

CSV outputs:

- `conv2_pd_v4_vs_v1_test_saturation_256_gpu.csv`
- `conv2_pd_v4_vs_v1_train_saturation_256_gpu.csv`
- `conv2_pd_v4_vs_v1_conductance_scale.csv`

The two runs have almost identical initial operating points when viewed through
perfect-diode saturation:

| Split | Run | Init h1 | Init h2 | Init all |
| --- | --- | ---: | ---: | ---: |
| test | `v1/c1`, gain `400`, LR `0.012`, 50ep | 47.05% | 50.58% | 48.23% |
| test | `v4/c1`, gain `100`, LR `0.012`, 30ep | 46.93% | 50.77% | 48.21% |
| train | `v1/c1`, gain `400`, LR `0.012`, 50ep | 47.17% | 50.54% | 48.29% |
| train | `v4/c1`, gain `100`, LR `0.012`, 30ep | 47.05% | 50.72% | 48.27% |

After training, the difference is mainly in layer 2:

| Split | Run | Final h1 | Final h2 | Final all | Best/final acc |
| --- | --- | ---: | ---: | ---: | ---: |
| test | `v1/c1`, gain `400`, LR `0.012`, 50ep | 50.73% | 57.56% | 53.01% | 97.87% / 97.77% |
| test | `v4/c1`, gain `100`, LR `0.012`, 30ep | 47.56% | 69.81% | 54.97% | 98.35% / 98.34% |
| train | `v1/c1`, gain `400`, LR `0.012`, 50ep | 50.74% | 58.08% | 53.19% | 97.87% / 97.77% |
| train | `v4/c1`, gain `100`, LR `0.012`, 30ep | 47.65% | 70.35% | 55.22% | 98.35% / 98.34% |

Interpretation: `v1/c1` with raw gain `400` works because it matches the
first-layer drive of `v4/c1` with raw gain `100` (`400` effective input scale).
But `v4/c1` is not equivalent to raw input scaling. It changes the internal
interaction/updater coefficients, and the trained checkpoint lands in a much
more nonlinear second-layer operating point while keeping h1 slightly less
saturated. This is the likely source of the remaining Conv2 accuracy gap.

Conductance-scale follow-up: GPU job `1176326` completed on 2026-07-01 and
wrote `conv2_pd_v4_vs_v1_conductance_scale.csv` under the same output root.
This compares `weights_best.npz` and `weights_final.npz` for the good
`v4/c1` row, the high-gain `v1/c1` rescue, and the older gain-100 `v1/c1`
repeat.

Final-checkpoint raw learned scales:

| Run | ConvW0 mean/max | ConvW1 mean/max | DenseW mean/max | Bias0 mean | Bias1 mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v4/c1`, gain `100`, LR `0.012`, 30ep | `0.1339 / 0.7094` | `0.00657 / 0.3116` | `0.00606 / 0.4686` | `1.79e-4` | `2.96e-3` |
| high-gain `v1/c1`, gain `400`, LR `0.012`, 50ep | `0.0917 / 0.5282` | `0.00634 / 0.2309` | `0.00751 / 0.8665` | `2.45e-5` | `1.29e-4` |
| old `v1/c1`, gain `100`, LR `0.024`, 30ep | `0.1506 / 2.0565` | `0.00433 / 0.7276` | `0.00806 / 0.7193` | `1.96e-4` | `1.25e-3` |

The better `v4/c1` row does not win by having larger raw hidden-to-hidden or
dense conductances. Relative to high-gain `v1/c1`, its final raw mean is
`1.46x` on `ConvWeight_0`, only `1.04x` on `ConvWeight_1`, and `0.81x` on
`DenseWeight_0`. With the energy interaction factor `(current_amp/voltage_amp)^k`,
`v4/c1` is actually lower on deeper inter-layer terms: `ConvWeight_1` has a
`0.25x` factor and `DenseWeight_0` has a `0.0625x` factor. Those factors are not
the whole update rule because the implementation also rescales pre/post layer
states, but they rule out the simple explanation that `v4/c1` just learned
bigger deeper conductances.

The older gain-100 `v1/c1` row looks spikier: it has much larger maxima
(`ConvWeight_0` max `2.06`, `ConvWeight_1` max `0.73`) but worse accuracy. The
high-gain `v1/c1` rescue reduces those spiky first-layer and hidden-layer
maxima while improving accuracy, so the gain rescue appears to fix the
operating point rather than simply increasing learned conductance scale. The
remaining `v4/c1` advantage is more consistent with amplification-induced
hidden-layer dynamics and much larger learned hidden biases, especially
`Bias_1` (`22.8x` the high-gain `v1/c1` mean).

Gradient follow-up: GPU job `1177541` completed on 2026-07-01 and wrote:

- `conv2_pd_v4_vs_v1_train_gradient_init_final_256_gpu.csv`
- `conv2_pd_v4_vs_v1_test_gradient_init_final_256_gpu.csv`

These use 256 samples, `phase=init/final`, and report both ordinary BP
gradient norms and log-conductance gradients `G * dL/dG`.

Train-split gradient summary:

| Phase | Run | Global rel grad | Global rel step | W0 logG grad | W1 logG grad | W2 logG grad |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| init | `v1/c1`, gain `400` | `0.1215` | `1.46e-3` | `3.15e-3` | `1.74e-3` | `3.70e-3` |
| init | `v4/c1`, gain `100` | `0.1609` | `1.93e-3` | `4.34e-3` | `2.28e-3` | `4.86e-3` |
| final | `v1/c1`, gain `400` | `0.0315` | `2.31e-4` | `0` | `0` | `1.21e-2` |
| final | `v4/c1`, gain `100` | `0.0395` | `3.54e-4` | `1.10e-2` | `7.00e-3` | `8.27e-3` |

Final-checkpoint train gradients show the clearest difference: high-gain
`v1/c1` has exactly zero `ConvWeight_0` and `ConvWeight_1` gradients
(`grad_zero_fraction=1.0`) on the 256-sample train diagnostic, while `v4/c1`
keeps nonzero gradients in both early conv layers. The same pattern holds on
the test diagnostic. This gives a more concrete mechanism for the accuracy gap:
`v1/c1` reaches a decent operating point, but its early conv layers become
gradient-dead and the remaining signal is concentrated in the dense/readout
connection. `v4/c1` retains trainable early-conv signal while also driving the
second hidden layer into a more nonlinear regime.

Broader gradient survey: GPU job `1180441` completed on 2026-07-01 and wrote
`/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/perfect_diode_gradient_survey_20260701/perfect_diode_conv123_gradient_survey_final_train256_gpu.csv`.
This measured final-checkpoint train gradients on 256 samples for the Conv1/2/3
repeat rows plus the tuned Conv2/Conv3 `v1/c1` rescues.

| Run | Best/final acc | W0 grad | W1 grad | W2/dense status |
| --- | ---: | ---: | ---: | --- |
| Conv1 `v1/c1`, gain `40` | `96.55% / 96.41%` | nonzero | dense nonzero | n/a |
| Conv1 `v4/c1`, gain `40` | `96.76% / 96.67%` | nonzero | dense nonzero | n/a |
| Conv2 old `v1/c1`, gain `100` | `97.09% / 97.08%` | nonzero | nonzero | dense nonzero |
| Conv2 tuned `v1/c1`, gain `400` | `97.87% / 97.77%` | zero | zero | dense nonzero |
| Conv2 `v4/c1`, gain `100` | `98.35% / 98.34%` | nonzero | nonzero | dense nonzero |
| Conv3 repeat `v1/c1`, gain `360` | `97.31% / 97.27%` | zero | zero | W2 and dense nonzero |
| Conv3 tuned `v1/c1`, gain `480` | `97.41% / 97.38%` | zero | zero | W2 and dense nonzero |
| Conv3 `v4/c1`, gain `360` | `96.71% / 96.65%` | zero | zero | W2 and dense nonzero |

This makes the earlier conclusion more precise. Exact zero early-conv gradients
are real in the BP diagnostic, but they are not unique to high-gain Conv2
`v1/c1`: they also appear in all checked Conv3 final checkpoints. They are not
a universal artifact either, because Conv1, old Conv2 `v1/c1`, and Conv2
`v4/c1` retain nonzero early-conv gradients. The best current interpretation is
that hard-clipped perfect-diode networks can enter an operating point where the
output-cost signal no longer propagates through the earliest conv layers under
the unrolled BP minimizer. Conv2 `v4/c1` avoids that failure mode, while tuned
Conv2 `v1/c1` and all checked Conv3 rows do not.

Async/K sanity check: GPU job `1182068` completed on 2026-07-01 and wrote
`/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/perfect_diode_gradient_k_sweep_20260701/perfect_diode_gradient_k{1,2,4,8,16}_init_final_train256_gpu.csv`.
The checked source configs all use `energy_minimizer.mode=asynchronous`.
Conv2 source runs use `K=4`; Conv3 source runs use `K=8`. In the K sweep,
both inference and BP/training minimizer iterations were overridden together.

Source-K behavior on the representative rows:

| Run | Source K | Init W0/W1 | Final W0/W1 |
| --- | ---: | --- | --- |
| Conv2 old `v1/c1`, gain `100` | 4 | nonzero | nonzero |
| Conv2 tuned `v1/c1`, gain `400` | 4 | nonzero | zero |
| Conv2 `v4/c1`, gain `100` | 4 | nonzero | nonzero |
| Conv3 tuned `v1/c1`, gain `480` | 8 | zero | zero |

Changing K changes the diagnostic qualitatively. For the Conv2 rows, W0/W1
are generally nonzero at `K=2` and `K=4`, but zero at `K=1`, `K=8`, and
`K=16`. For the tuned Conv2 `v1/c1`, final W0/W1 are already zero at the
source `K=4`; for Conv2 `v4/c1`, final W0/W1 are nonzero at source `K=4` but
become zero at `K=8/16`. For the tuned Conv3 `v1/c1`, W0/W1 can be nonzero at
shorter final diagnostic lengths (`K=2/4`) but are zero at the source `K=8`.

This means the earlier "dead gradient" statement should be read as
source-protocol specific: these are exact zeros under the same asynchronous
unroll length used by the run, not an invariant property under every K. The
current hypothesis is that intermediate unroll lengths carry transient
output-cost signal into early layers, while longer unrolls converge into a
hard-clipped fixed-point branch where those early-layer Jacobians are zero.

Follow-up fixed-inference/fixed-tail probe: implemented
`experiments/run_mnist_bp_perfect_diode_gradient_tk_sweep_gpu_jeanzay.slurm`,
`experiments/run_mnist_bp_perfect_diode_gradient_tk_sweep_seq_gpu_jeanzay.slurm`,
and `experiments/summarize_mnist_bp_perfect_diode_gradient_tk_sweep.py` to
separate inference settling `T` from BP tail length `K`.

Dev jobs:

- `1186381`: `T=8`, `K=1,2,4,6,8`.
- `1186505`: `T=4`, `K=1,2,4,6,8`.
- `1186667`: `T=16`, `K=1,2,4,6,8`.
- `1186263`: full sequential `T=4,8,16` job, canceled while pending after the
  dev probes produced all outputs.

Output root:
`/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/perfect_diode_gradient_tk_sweep_20260701`.

Summary files:

- `perfect_diode_gradient_tk_sweep_final_summary.csv`
- `perfect_diode_gradient_tk_sweep_final_summary.md`

The dev probes used 128 train samples; the inherited per-combo filenames still
contain `train256`, but the CSV `num_samples` field records the actual count.

Final-checkpoint result:

| Run | `T=4,K=1` | `T=4,K=2` | `T=4,K=4` | `T=8,K=1..8` | `T=16,K=1..8` |
| --- | --- | --- | --- | --- | --- |
| Conv2 old `v1/c1`, gain `100` | dead | alive | alive | dead | dead |
| Conv2 tuned `v1/c1`, gain `400` | dead | dead | dead | dead | dead |
| Conv2 `v4/c1`, gain `100` | dead | alive | alive | dead | dead |
| Conv3 repeat `v1/c1`, gain `360` | dead | alive | alive | dead | dead |
| Conv3 tuned `v1/c1`, gain `480` | dead | alive | alive | dead | dead |
| Conv3 `v4/c1`, gain `360` | dead | alive | alive | dead | dead |

Representative Conv3 log-conductance gradients at `T=4,K=4`:

| Run | W0 logG grad | W1 logG grad | W2 logG grad | Global rel grad |
| --- | ---: | ---: | ---: | ---: |
| Conv3 repeat `v1/c1`, gain `360` | `6.42e-3` | `2.29e-3` | `3.99e-3` | `1.21e-2` |
| Conv3 tuned `v1/c1`, gain `480` | `7.04e-3` | `1.86e-3` | `3.33e-3` | `6.50e-3` |
| Conv3 `v4/c1`, gain `360` | `1.70e-2` | `2.73e-3` | `3.51e-3` | `9.26e-3` |

At fixed `T=8`, every checked row has W0/W1 exactly zero for all
`K=1,2,4,6,8`. At fixed `T=16`, the same rows remain zero for all checked K.
The coupled K-sweep revival at `K=2/4` therefore came primarily from using a
less-settled inference state, not merely from backpropagating through a shorter
tail. The useful training hypothesis is now: use paper-like smaller inference
settling (`T≈4`) with at least `K=2` BP-tail steps, then retune LR.

High-K Conv2 training rerun follow-up, 2026-07-03:

Case note:
`/home/filip/server_code/labs/cases/mnist_conv2_perfect_diode_highK_gradient_collapse/README.md`.

The new check reran the good no-padding Conv2 `v4/c1`, input gain `100`, LR
`0.012` row at higher K. The goal was to use a K large enough that residuals
stagnate before trusting the gradient diagnostic.

| Run | K | Padding | Best / final test acc | Output |
| --- | ---: | ---: | ---: | --- |
| Original source row | 4 | 0 | `0.9866 / 0.9864` | `/home/filip/server_code/results/mnist_bp_conv2_epbpK_large_50epoch_64_128ch_s2_valid_seed0_local_distributed/perfect_diode/mnist_bp_amp_v4_c1/seed_0` |
| Local high-K rerun | 64 | 0 | `0.9015 / 0.8978` | `/home/filip/server_code/results/mnist_bp_conv2_selected_v4c1_highK64_seed0_50epoch/perfect_diode/mnist_bp_amp_v4_c1/seed_0` |
| Jean Zay high-K rerun | 16 | 0 | `0.9015 / 0.8978` | `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv2_v4c1_plateauK_seed0_50epoch/k_16/perfect_diode/mnist_bp_amp_v4_c1/seed_0` |
| Jean Zay padding-1 rerun | 16 | 1 | `0.8977 / 0.8896` | `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv2_pad1_v4c1_k16_seed0_30epoch/k_16/perfect_diode/mnist_bp_amp_v4_c1/seed_0` |

Gradient recheck outputs:

- `/home/filip/server_code/results/diagnostic_mirrors/perfect_diode_recheck_20260703/sourceK_init`
- `/home/filip/server_code/results/diagnostic_mirrors/perfect_diode_recheck_20260703/sourceK_final`
- `/home/filip/server_code/results/diagnostic_mirrors/perfect_diode_recheck_20260703/forcedK16_final`

| Check | K | ConvWeight_0 grad L2 | ConvWeight_1 grad L2 | DenseWeight_0 grad L2 | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| Original init | 4 | `0.1795193665` | `0.3727669716` | `1.8526757061` | upstream alive |
| Original final | 4 | `0.1092378637` | `0.6823540777` | `0.6188042089` | upstream alive |
| Original final forced | 16 | `0.0` | `0.0` | `0.6050187796` | upstream dead when settled |
| High-K init | 64 | `0.0` | `0.0` | `1.8489904404` | upstream dead from initialization |
| High-K final | 64 | `0.0` | `0.0` | `1.0185021609` | dense/readout-only learning |
| High-K final forced | 16 | `0.0` | `0.0` | `1.0185021609` | same dead-upstream pattern |

Conclusion: the residual-converged perfect-diode Conv2 protocol kills the
upstream conv gradient. The source `K=4` result appears to rely on a transient
unsettled gradient path. Future protocol checks should require both residual
stagnation and a nonzero/useful upstream gradient check; for perfect diode,
these two requirements currently conflict in Conv2.

10-input follow-up: job `1189725` reran the same `T={4,8,16}`,
`K={1,2,4,6,8}` grid with `MAX_SAMPLES=10` and wrote outputs under
`/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/perfect_diode_gradient_tk_sweep_n10_20260701`.
The summary files are `perfect_diode_gradient_tk_sweep_final_summary.csv` and
`.md`.

The zero/nonzero pattern is unchanged from the 128-sample probe:

| Run | `T=4,K=1` | `T=4,K=2` | `T=4,K=4` | `T=8,K=1..8` | `T=16,K=1..8` |
| --- | --- | --- | --- | --- | --- |
| Conv2 old `v1/c1`, gain `100` | dead | alive | alive | dead | dead |
| Conv2 tuned `v1/c1`, gain `400` | dead | dead | dead | dead | dead |
| Conv2 `v4/c1`, gain `100` | dead | alive | alive | dead | dead |
| Conv3 repeat `v1/c1`, gain `360` | dead | alive | alive | dead | dead |
| Conv3 tuned `v1/c1`, gain `480` | dead | alive | alive | dead | dead |
| Conv3 `v4/c1`, gain `360` | dead | alive | alive | dead | dead |

Representative 10-input Conv3 log-conductance gradients at `T=4,K=4`:

| Run | W0 logG grad | W1 logG grad | W2 logG grad | Global rel grad |
| --- | ---: | ---: | ---: | ---: |
| Conv3 repeat `v1/c1`, gain `360` | `3.92e-3` | `1.88e-3` | `3.59e-3` | `1.23e-2` |
| Conv3 tuned `v1/c1`, gain `480` | `3.53e-3` | `1.45e-3` | `2.96e-3` | `6.62e-3` |
| Conv3 `v4/c1`, gain `360` | `6.94e-3` | `2.42e-3` | `3.94e-3` | `8.26e-3` |

The magnitudes shift with the input subset, but the exact-zero structure does
not. This makes the `T=8`/`T=16` early-gradient death unlikely to be a
mini-batch averaging artifact.

Output-node follow-up: the prior perfect-diode Conv checkpoints used
`output_dim=20`, which selects the paired-output MNIST loss. The corrected
follow-up tests `output_dim=10`, which uses the standard 10-output squared-error
loss. Because the output layer shape changes, the output-20 checkpoints cannot
be reloaded directly as output-10 models; new checkpoints are required.

Implemented and staged:

- `experiments/run_mnist_bp_conv23_r3_perfect_diode_output10_representative_jeanzay.slurm`
- `experiments/run_mnist_bp_perfect_diode_output10_gradient_tk_sweep_seq_gpu_jeanzay.slurm`
- `experiments/run_mnist_bp_conv23_r3_perfect_diode_output10_representative_10epoch_jeanzay.slurm`
- `experiments/run_mnist_bp_perfect_diode_output10_10epoch_gradient_tk_sweep_seq_gpu_jeanzay.slurm`
- Updated `experiments/summarize_mnist_bp_perfect_diode_gradient_tk_sweep.py`
  so it summarizes output-10 diagnostic filenames as well as the old `train256`
  filenames.

The first 30/50-epoch launch was canceled at 2026-07-01 17:45 CEST after the
user requested a 10-epoch smoke only:

- Training array `1190233`: tasks `0-2` were canceled after about 34 seconds;
  tasks `3-5` were still pending.
- Dependent T/K diagnostic `1190234`: canceled while pending.

Relaunched at 2026-07-01 17:45 CEST:

- Training array `1190525`, Jean Zay R3 `fmu@v100`, array `0-5%6`, pending on
  priority.
- Dependent T/K diagnostic `1190531`, pending on `afterok:1190525`.

Training rows, all seed `0`, perfect diode, strides/paddings `[2,2]`/`[1,1]`
for Conv2 and `[2,2,1]`/`[1,1,1]` for Conv3:

| Task | Row | Epochs | K | Input gain | LR | Output dim |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | Conv2 old `v1/c1` | 10 | 4 | 100 | 0.024 | 10 |
| 1 | Conv2 tuned `v1/c1` | 10 | 4 | 400 | 0.012 | 10 |
| 2 | Conv2 `v4/c1` | 10 | 4 | 100 | 0.012 | 10 |
| 3 | Conv3 repeat `v1/c1` | 10 | 8 | 360 | 0.0432 | 10 |
| 4 | Conv3 tuned `v1/c1` | 10 | 8 | 480 | 0.0864 | 10 |
| 5 | Conv3 `v4/c1` | 10 | 8 | 360 | 0.0864 | 10 |

Training root:
`/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv23_s221_pad1_perfect_diode_output10_representative_k4k8_seed0_10epoch`.

Diagnostic root:
`/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/perfect_diode_output10_10epoch_gradient_tk_sweep_20260701`.

Completed status:

- Training array `1190525`: `6/6` completed.
- T/K diagnostic `1190531`: completed successfully and wrote
  `perfect_diode_output10_10epoch_gradient_tk_sweep_final_summary.{csv,md}`.

10-epoch output-10 training results:

| Row | Best / final test acc | Best epoch |
| --- | ---: | ---: |
| Conv2 old `v1/c1`, gain `100`, LR `0.024` | `96.49% / 96.49%` | 10 |
| Conv2 tuned `v1/c1`, gain `400`, LR `0.012` | `95.99% / 95.94%` | 9 |
| Conv2 `v4/c1`, gain `100`, LR `0.012` | `97.61% / 97.61%` | 9 |
| Conv3 repeat `v1/c1`, gain `360`, LR `0.0432` | `63.30% / 63.30%` | 10 |
| Conv3 tuned `v1/c1`, gain `480`, LR `0.0864` | `78.26% / 78.26%` | 10 |
| Conv3 `v4/c1`, gain `360`, LR `0.0864` | `74.48% / 73.41%` | 3 |

Initial-gradient check:

| Setup | Source protocol | Initial W0/W1 status |
| --- | --- | --- |
| Conv2 output-20, representative rows | `T=4,K=4` | alive |
| Conv3 output-20 `v1/c1` rows | `T=8,K=8` | zero |
| Conv3 output-20 `v4/c1` | `T=8,K=8` | zero |
| Conv2 output-10, 10-epoch rows | `T=4,K=4` | alive |
| Conv3 output-10 `v1/c1` rows | `T=8,K=8` | zero |
| Conv3 output-10 `v4/c1` | `T=8,K=8` | zero |

Representative initial log-conductance gradients:

| Row | Diagnostic | W0 logG grad | W1 logG grad |
| --- | --- | ---: | ---: |
| Conv2 output-20 tuned `v1/c1` | `T=4,K=4` | `2.77e-3` | `1.73e-3` |
| Conv2 output-20 `v4/c1` | `T=4,K=4` | `3.98e-3` | `2.28e-3` |
| Conv3 output-20 tuned `v1/c1` | `T=8,K=8` | `0` | `0` |
| Conv3 output-20 `v4/c1` | `T=8,K=8` | `0` | `0` |
| Conv2 output-10 tuned `v1/c1` | `T=4,K=4` | `1.53e-1` | `6.52e-3` |
| Conv2 output-10 `v4/c1` | `T=4,K=4` | `2.26e-1` | `8.61e-3` |
| Conv3 output-10 tuned `v1/c1` | `T=8,K=8` | `0` | `0` |
| Conv3 output-10 `v4/c1` | `T=8,K=8` | `0` | `0` |

The key distinction is therefore depth/protocol, not simply "initial" versus
"trained": Conv2 starts with early gradients alive under its source `T=4,K=4`,
whereas Conv3 starts with early gradients already dead under its source
`T=8,K=8`. Shorter `T=4` can revive some Conv3 gradients diagnostically, but
that is not the source Conv3 training protocol.

## 2026-07-02 Checkpoint Weight-Movement Audit

Question: did the weights really stay unchanged throughout training?

Method: on Jean Zay, reconstructed the seed-0 initialization from each archived
`config.json`, then compared it against `best_model.pt` and `final_model.pt`.
The run directories only contain `best_model.pt` and `final_model.pt`, so this
audit checks init-to-best/init-to-final and best-to-final behavior, not a full
per-epoch trajectory.

Conclusion:

- Most output-20 representative runs changed weights substantially. Final
  checkpoint relative L2 deltas for Conv3 were:
  - `v1/c1`, gain `360`, LR `0.0432`, 50 epochs: W0 `2.33`, W1 `1.44`,
    W2 `2.04`, dense `7.97`; changed fraction roughly `97-100%`.
  - tuned `v1/c1`, gain `480`, LR `0.0864`, 50 epochs: W0 `2.01`,
    W1 `1.24`, W2 `2.72`, dense `11.06`; changed fraction roughly
    `94-99%`.
  - `v4/c1`, gain `360`, LR `0.0864`, 50 epochs: W0 `0.53`, W1 `0.35`,
    W2 `2.84`, dense `10.51`; W0/W1 moved from initialization but were
    unchanged between best and final.
- The output-10 10-epoch Conv3 repeat `v1/c1` row is the exception:
  W0 and W1 were exactly unchanged versus initialization at both best and
  final (`delta_l2=0`, `changed_fraction=0`). W2 and dense still moved
  strongly (`relative_delta_l2=0.90` and `5.53`).
- Output-10 tuned Conv3 `v1/c1` did move W0/W1 modestly by final
  (`relative_delta_l2=0.22/0.22`), while W2/dense moved much more
  (`1.39/8.63`).

Interpretation: the earlier "dead gradient" result does not mean every Conv3
checkpoint kept all weights fixed. For output-20, downstream/output weights can
move first and later give the early conv layers nonzero updates. For the
output-10 repeat row, the first two conv layers genuinely remained fixed under
the saved checkpoints.
