# Current Research State

Updated: 2026-07-04 17:24 CEST

## Problem We Are Solving

The working hypothesis is that deeper Conv DRN networks need amplification; otherwise, adding depth does not translate into better accuracy. So far, even when amplification is included, the deeper hard-sigmoid networks are not clearly giving the accuracy jump we expected.

The likely issue is initialization/operating point. Initial and final saturation checks show that later hidden layers often have no saturated neurons, especially in the good `v_off=4` hard-sigmoid runs. That may mean the deeper layers are not being driven into a useful nonlinear regime. Before making that conclusion, we still have to make sure the learning rates are not the limiting factor.

Additional tests currently meant to clarify this:

1. Trainable layer-wise `v_off`: make `v_off` trainable and different by layer to see whether learned operating points improve deeper-layer usefulness.
2. `perfect_diode` networks: test the same architectures with a nonlinearity where neurons are easier to saturate.

Hard-sigmoid inclusion rule for this note: only list runs with `v_off=4.0` and `input_gain` chosen from a target-saturation calibration. Older `v_off=1.5` or fixed-raw-gain hard-sigmoid rows are diagnostics only and are excluded from the current-best hard-sigmoid rows below.

“Initial saturation” is measured on 256 train samples at initialization and reported per hidden layer when available.

## Conv1

| Nonlinearity | Current best eligible run | Best / final acc | Initial saturation for best run | Source |
|---|---|---:|---|---|
| `hard_sigmoid` | `mnist_bp_amp_v2_c1`, target sat30, input gain `65.658`, `v_off=4.0`, 10 epochs | `97.12% / 97.12%` | hidden1 `27.93%` | `/home/filip/server_code/results/mnist_bp_conv1_hardsigmoid_amp_saturation_targets_voff4_seed0_10epoch_lr0p5_akib` |
| `perfect_diode` | `mnist_bp_amp_v4_c1`, input gain `40`, 50 epochs, LR `0.009` | `97.45% / 97.41%` | hidden1 `47.40%` | `/home/filip/server_code/results/mnist_bp_conv1_perfect_diode_inputgain40_lr_sweep_seed0_50epoch` |

Notes:

- Conv1 hard-sigmoid best eligible row is `A=2,B=1`, not `A=4,B=1`, once the stricter `v_off=4` + saturation-calibrated rule is enforced.
- Conv1 perfect-diode best currently remains `A=4,B=1`.

## Conv2

| Nonlinearity | Current best eligible run | Best / final acc | Initial saturation for best run | Source |
|---|---|---:|---|---|
| `hard_sigmoid` | `mnist_bp_amp_v2_c1`, target sat30, input gain `181.990`, initial `v_off=4.0`, trainable `v_off` only, fixed amp `A=2,B=1`, `K=6`, 50 epochs | `97.94% / 97.94%` | first hidden `30.00%`; later-layer saturation not yet remeasured for this trainable row | `/home/filip/server_code/results/mnist_bp_conv2_trainable_voff4_only_voff_sat30params_seed0_50epoch` |
| `perfect_diode` | `mnist_bp_amp_v4_c1`, input gain `100`, source `K=4`, 50 epochs | `98.66% / 98.64%` | hidden1 `46.94%`, hidden2 `50.57%`, all hidden `48.02%` | `/home/filip/server_code/results/mnist_bp_conv2_epbpK_large_50epoch_64_128ch_s2_valid_seed0_local_distributed` |

Perfect-diode high-K warning, 2026-07-03:

- The listed Conv2 perfect-diode row is a source-`K=4`/transient-gradient result, not settled-K behavior.
- Rerunning the same no-padding `v4/c1` setting with high K collapses to about `90%`: local `K=64` best/final `90.15% / 89.78%`, Jean Zay no-padding `K=16` `90.15% / 89.78%`, and Jean Zay padding-1 `K=16` `89.77% / 88.96%`.
- Gradient recheck shows `ConvWeight_0` and `ConvWeight_1` gradients are nonzero for the original source `K=4` row, but exactly zero when the same final checkpoint is forced to `K=16`. The high-K run has exactly zero W0/W1 gradients at initialization and after training; `DenseWeight_0` remains nonzero.
- Case note: `/home/filip/server_code/labs/cases/mnist_conv2_perfect_diode_highK_gradient_collapse/README.md`.
- Protocol consequence: before launching new comparison runs, check that K makes residuals stagnate and that upstream gradients are still alive. For Conv2 perfect diode, those two checks currently conflict.

### Conv2 T/K Gradient Diagnostic Update (2026-07-03)

Separate `T/K` sweeps must disable adaptive equilibrium stopping. The corrected Conv2 sat30 `v_off=4` hard-sigmoid runs and the historical saved training configs omitted `minimizer.adaptive_equilibrium`, and the MNIST builder defaulted that value to `True`. Therefore the first separate-`T/K` hard-sigmoid tables were still adaptive-stop diagnostics, not exact fixed-step measurements.

With adaptive stopping disabled, the corrected Conv2 sat30 `v_off=4` hard-sigmoid sweep used 256 train samples, batch size 32, phases `init` and `final`, `T in {6,24,48,96,256}`, and `K in {1,2,4,6,16,32,64,128,256}`. Artifacts are under `/home/filip/server_code_conv_amplification_paper/labs/cases/hard_sigmoid_tk_gradient_sweep_20260703`.

| Run | Phase | T behavior | K effect | Early ConvWeight grads |
|---|---|---|---|---:|
| `v1/c1` | init | nonzero for all tested `T=6..256` | `K=1` zero; `K>=2` nonzero for every tested `T` | up to `~0.180` summed L2 |
| `v1/c1` | final | nonzero for all tested `T=6..256` | `K=1` zero; `K>=2` nonzero for every tested `T` | up to `~0.371` summed L2 |
| `v4/c1` | init | nonzero for all tested `T=6..256` | `K=1` zero; `K>=2` nonzero for every tested `T` | up to `~4.11` summed L2 |
| `v4/c1` | final | nonzero for all tested `T=6..256` | `K=1` zero; `K>=2` nonzero for every tested `T` | up to `~1.02` summed L2 |

Protocol consequence: exact `T/K` diagnostics must pass `adaptive_equilibrium=False`; otherwise an already-settled free state can stop after one odd/even sweep and make upstream conv gradients look zero. Under fixed-step hard-sigmoid Conv2 diagnostics, there is no high-`T` early-conv zero-gradient threshold. The only robust zero row is `K=1`, which does not unroll enough dynamics to propagate into the early conv weights. These ConvWeight norms are for the shared kernel parameters after summing over all spatial uses; they do not by themselves prove that every pre-reduction spatial copy has zero local contribution.

Fully connected hard-sigmoid controls also have no zero-gradient T threshold under fixed-step diagnostics. Separate `T/K` sweeps over one-hidden-layer FC MNIST runs found `DenseWeight_0` and `DenseWeight_1` alive for every tested `T in {6,24,48,96,256}` and `K in {1,2,4,6,16,32,64,128,256}` at both initialization and final checkpoint. This was checked for legacy `v_off=1.5` `v1/c1` and `v4/c1` runs and for the newer dense-amp-fix `v1/c2` and `v1/c4` runs. Artifacts are under `/home/filip/server_code_conv_amplification_paper/labs/cases/dense_hard_sigmoid_tk_gradient_sweep_20260703`.

Current Conv2 hard-sigmoid status:

<!-- conv2-stride1-recalibrated-sat30-autoupdate:start -->
- Conv2 stride-1 hard-sigmoid repeats with stride-1 recalibrated sat30 input gains, launched locally in tmux `conv2_stride1_recal_sat30`, updated 2026-06-29 23:33 CEST. No final `metrics.json` files yet. The latest logs show both rows training in epoch `15/30`; completed test checkpoints include `A=1,B=1`: epoch 13 `95.82%`, epoch 14 `95.92%`; `A=4,B=1`: epoch 13 `96.88%`, epoch 14 `96.76%`.

| Amp | Run | State | Best / final acc | Best epoch | Input gain | LR mult | LR |
|---|---|---|---:|---:|---:|---:|---:|
| A=1,B=1 | `mnist_bp_amp_v1_c1` | running | n/a | n/a | `621.020` | `4` | `0.00927507` |
| A=4,B=1 | `mnist_bp_amp_v4_c1` | running | n/a | n/a | `2233.578` | `0.5` | `0.000322353` |

- Output root: `/home/filip/server_code/results/mnist_bp_conv2_hardsigmoid_stride1_recalibrated_sat30_voff4_k6_seed0_30epoch`
<!-- conv2-stride1-recalibrated-sat30-autoupdate:end -->

<!-- conv2-stride1-best30-autoupdate:start -->
- Conv2 stride-1 hard-sigmoid repeats using the old stride-2 calibrated gains were stopped on 2026-06-29 after calibration showed these gains are not comparable under stride 1. Treat this as an underdriven diagnostic, not the fair stride-1 comparison.

| Amp | Run | State | Best / final acc | Best epoch | Input gain | LR mult | LR |
|---|---|---|---:|---:|---:|---:|---:|
| A=1,B=1 | `mnist_bp_amp_v1_c1` | stopped | n/a | n/a | `221.068` | `4` | `0.0260553` |
| A=4,B=1 | `mnist_bp_amp_v4_c1` | stopped | n/a | n/a | `158.035` | `0.5` | `0.00455595` |

- Output root: `/home/filip/server_code/results/mnist_bp_conv2_hardsigmoid_stride1_best_sat30_voff4_k6_seed0_30epoch`
<!-- conv2-stride1-best30-autoupdate:end -->

<!-- conv2-trainable-ab-autoupdate:start -->
### Conv2 Trainable A/B Completion (2026-06-29 04:46 CEST)

The 30-epoch trainable-amplification Conv2 hard-sigmoid job `1005020` has completed on Jean Zay. It was submitted before the default-account switch, so these outputs are under the `umg` scratch result root.

| Run | Initial A/B | Final learned A/B | Input gain | LR | Best / final test acc | Best epoch | Final train/test loss | Metrics |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `mnist_bp_amp_v1_c1` | `A=1,B=1` | `A=1.647239,B=0.125846` | `221.068374634` | `0.00651382180914` | `97.65% / 97.63%` | `26` | `0.07194 / 0.07458` | `/lustre/fsn1/projects/rech/umg/ucy17uy/server_code/results/mnist_bp_conv2_hardsigmoid_trainable_amp_sat30_voff4_seed0_30epoch/target_sat30/input_gain_221p068374634/lr_0p00651382180914/hard_sigmoid/mnist_bp_amp_v1_c1/seed_0/metrics.json` |
| `mnist_bp_amp_v4_c1` | `A=4,B=1` | `A=4.257293,B=1.071110` | `158.035232544` | `0.00911189218265` | `96.34% / 96.34%` | `30` | `0.08306 / 0.08409` | `/lustre/fsn1/projects/rech/umg/ucy17uy/server_code/results/mnist_bp_conv2_hardsigmoid_trainable_amp_sat30_voff4_seed0_30epoch/target_sat30/input_gain_158p035232544/lr_0p00911189218265/hard_sigmoid/mnist_bp_amp_v4_c1/seed_0/metrics.json` |

- Best of the two trainable-A/B rows: `mnist_bp_amp_v1_c1` with best test accuracy `97.65%` and final test accuracy `97.63%`.
- Final learned A/B values are from `learned_amplification` in the final `metrics.json`; saturation diagnostics are still pending.
<!-- conv2-trainable-ab-autoupdate:end -->

<!-- conv2-voff-only-50epoch-autoupdate:start -->
- Trainable `v_off`-only sat30 50-epoch run completed locally:
  - Run: `/home/filip/server_code/results/mnist_bp_conv2_trainable_voff4_only_voff_sat30params_seed0_50epoch/hard_sigmoid/mnist_bp_amp_v2_c1/seed_0`.
  - Config: Conv2 `mnist_bp_amp_v2_c1`, seed `0`, BP, 50 epochs, batch size `4`, `K=6`, sat30-calibrated `input_gain=181.9896698`, initial `v_off=4.0`, fixed amp `A=2,B=1`.
  - LR vector: weights/biases/`v_off` at `0.0316501481`; LR decay `0.99`; amplification was not trainable.
  - Result: best/final test `97.94% / 97.94%`, best epoch `50`, final train/test loss `0.0595 / 0.0608`.
  - Learned thresholds: layer0 `8.4189`, layer1 `3.2176`.
  - Amplification remained fixed: voltage amp `2.0000`, current amp `1.0000`.
  - Comparison: fixed-boundary sat30 reference best `96.94%` (`+1.00 pp`); 10-epoch trainable `v_off` + trainable amp best `97.18%` (`+0.76 pp`).
<!-- conv2-voff-only-50epoch-autoupdate:end -->

- Trainable `v_off` + trainable amplification sat30 run completed locally:
  - Run: `/home/filip/server_code/results/mnist_bp_conv2_trainable_voff4_amp_lr4x_sat30params_seed0_10epoch/hard_sigmoid/mnist_bp_amp_v2_c1/seed_0`.
  - Config: Conv2 `mnist_bp_amp_v2_c1`, seed `0`, BP, 10 epochs, batch size `4`, `K=6`, sat30-calibrated `input_gain=181.9896698`, initial `v_off=4.0`, initial amp `A=2,B=1`.
  - LR vector: weights/biases/`v_off` at `0.0316501481`; `VoltageAmp`/`CurrentAmp` at `0.1266005924` (`4x` the training LR); LR decay `0.99`.
  - Result: best/final test `97.18% / 97.10%`, best epoch `9`, final train/test loss `0.0841 / 0.0833`.
  - Learned thresholds: layer0 `3.5505`, layer1 `4.0000`.
  - Learned amplification: voltage amp `2.1706`, current amp `0.4882`.
  - Comparison: the recent fixed-boundary sat30 Conv2 reference was `mnist_bp_amp_v2_c1`, `input_gain=181.990`, `K=6`, LR mult `4`, best `96.94%`; this trainable run is `+0.24 pp` on best test accuracy.

- Earlier trainable layer-wise `v_off` diagnostic completed locally:
  - Run: `/home/filip/server_code/results/mnist_bp_conv2_trainable_voff_bp_10epoch_seed0/hard_sigmoid/mnist_bp_amp_v1_c1/seed_0`.
  - Config: Conv2 `mnist_bp_amp_v1_c1`, seed `0`, BP, 10 epochs, batch size `4`, `K=4`, `input_gain=100`, LR `0.006`, LR decay `0.99`, initial `v_off=[1.5,1.5]`, trainable `v_off_min=0`.
  - Result: best/final test `93.56% / 93.56%`, best epoch `10`, final train/test loss `0.1285 / 0.1214`.
  - Learned thresholds: layer0 `4.3695`, layer1 `1.5340`.
  - Closest fixed-`v_off=1.5` 10-epoch Conv2 row found locally is not an exact hyperparameter match: `/home/filip/server_code/results/mnist_bp_conv2_hardsigmoid_inputgain60_k6_lr_protocol_seed0_10epoch/lr_0p030/hard_sigmoid/mnist_bp_amp_v1_c1/seed_0`, with `input_gain=60`, `K=6`, LR `0.030`, best/final `93.93% / 93.93%`. The trainable run is `-0.37 pp` vs this closest fixed row.
  - No exact fixed-control row was found locally with `input_gain=100`, `K=4`, LR `0.006`, 10 epochs.

- The corrected per-amplification calibrated sat30 10-epoch LR screen found:
  - `v1/c1`: LR mult `4`, best `96.01%`.
  - `v2/c1`: LR mult `4`, best `96.94%`.
  - `v4/c1`: LR mult `0.5`, best `96.59%`.
  - legacy `v4/c0.25`: LR mult `0.25`, best `86.50%`.
- Local copied low-LR half is under `/home/filip/server_code/results/mnist_bp_conv2_hardsigmoid_amp_calibrated_sat30_lr_sweep_voff4_seed0_10epoch_akib`.
- Local target CSV for corrected sat30 calibration: `/home/filip/server_code/results/mnist_bp_conv2_hardsigmoid_amp_calibrated_sat30_lr_sweep_voff4_seed0_10epoch/targets/conv2_amp_calibrated_targets.csv`.
- Conv2 calibrated sat30 50-epoch continuation completed on 2026-06-29:
  - Trex root `/home/filip/server_code/results/mnist_bp_conv2_hardsigmoid_amp_calibrated_sat30_bestlr_voff4_seed0_50epoch_trex`: `4/4` metrics complete.
  - Akib root `/home/filiposana/server_code/results/mnist_bp_conv2_hardsigmoid_amp_calibrated_sat30_bestlr_voff4_seed0_50epoch_akib`: `3/3` metrics complete.
  - Trex and Akib GPUs are idle.

| Amp | Run | LR mult | Best / final acc | Best epoch |
|---|---|---:|---:|---:|
| A=4,B=1 | `mnist_bp_amp_v4_c1` | `0.5` | `97.63% / 97.59%` | 49 |
| A=2,B=1 | `mnist_bp_amp_v2_c1` | `4` | `97.58% / 97.58%` | 50 |
| A=1,B=1 | `mnist_bp_amp_v1_c1` | `4` | `96.72% / 96.72%` | 50 |
| A=2,B=2 | `mnist_bp_amp_v2_c2` | `4` | `96.58% / 96.58%` | 50 |
| A=1,B=2 | `mnist_bp_amp_v1_c2` | `4` | `94.07% / 94.00%` | 43 |
| legacy A=4,B=0.25 | `mnist_bp_amp_v4_c0p25` | `0.25` | `91.90% / 91.83%` | 48 |
| A=1,B=4 | `mnist_bp_amp_v1_c4` | `4` | `84.88% / 84.51%` | 48 |

Final saturation for these 50-epoch checkpoints was measured on 256 train samples with the final checkpoint and `K=6`. Saturation means hidden voltage outside `[-4, 4]`.

| Amp | Run | Final L1 sat | Final L2 sat | Final all-hidden sat |
|---|---|---:|---:|---:|
| A=4,B=1 | `mnist_bp_amp_v4_c1` | 19.36% | 0.00% | 13.58% |
| A=2,B=1 | `mnist_bp_amp_v2_c1` | 34.20% | 0.00% | 23.99% |
| A=1,B=1 | `mnist_bp_amp_v1_c1` | 43.17% | 0.00% | 30.27% |
| A=2,B=2 | `mnist_bp_amp_v2_c2` | 34.35% | 0.15% | 24.13% |
| A=1,B=2 | `mnist_bp_amp_v1_c2` | 58.57% | 0.85% | 41.32% |
| legacy A=4,B=0.25 | `mnist_bp_amp_v4_c0p25` | 98.95% | 27.49% | 77.60% |
| A=1,B=4 | `mnist_bp_amp_v1_c4` | 95.20% | 1.03% | 67.06% |

Interpretation: in the successful voltage-amplified rows, the second hidden layer is basically not saturated; the useful operating-point difference is mostly how hard the first hidden layer is driven. The poor current-heavy and legacy rows end with extreme first-layer saturation, and legacy also has substantial second-layer saturation.

Notes:

- Conv2 hard-sigmoid best eligible row is now the Jean Zay trainable-amplification `A=1,B=1` 30-epoch row at `97.65%` best test accuracy. Among fixed-amplification 50-epoch sat30 continuations, `A=4,B=1` is best so far at `97.63%`.
- The perfect-diode best remains much higher than the hard-sigmoid best, consistent with the idea that easier saturation/nonlinearity may help deeper layers.

## Conv3

| Nonlinearity | Current best eligible run | Best / final acc | Initial saturation for best run | Source |
|---|---|---:|---|---|
| `hard_sigmoid` | `mnist_bp_amp_v4_c1`, target sat10, input gain `296.905`, `v_off=4.0`, `K=8`, LR mult `2`, 30 epochs | `96.71% / 96.64%` | hidden1 `9.87%`, hidden2 `0.00%`, hidden3 `0.00%`, all hidden `5.40%` | `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv3_pad1_hardsigmoid_amp_saturation_targets_voff4_sat10_20_30_50_k8_seed0_30epoch` |
| `perfect_diode` | tuned `mnist_bp_amp_v1_c1`, input gain `480`, LR `0.0864`, `K=8`, 50 epochs | `97.41% / 97.38%` | not measured | `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv23_s221_pad1_perfect_diode_v1c1_gain_lr_downbias_bestlr_k4k8_seed0_30_50epoch` |

Current Conv3 hard-sigmoid diagnostics on Jean Zay R3:

| Job | State at 22:52 CEST | What it runs | Metrics so far | Output root |
|---|---|---|---:|---|
| `1004671` | completed | legacy `mnist_bp_amp_v4_c0p25`, fixed input gains `50,100,150,200,250`, LR `=(1.44/input_gain)*0.5`, 10 epochs | `5/5` | `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv3_pad1_hardsigmoid_legacy_lowgain_voff4_k8_seed0_10epoch` |
| `1004672` | timed out at 3h on all 15 array tasks | `mnist_bp_amp_v1_c1`, `mnist_bp_amp_v2_c1`, `mnist_bp_amp_v4_c1`, sat30, LR multipliers `0.25,0.5,1,2,4`, 10 epochs | `0/15` | `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv3_pad1_hardsigmoid_nonlegacy_sat30_lr_screen_voff4_k8_seed0_10epoch` |

Current Conv3 perfect-diode shared-gain diagnostic on Jean Zay R3:

| Job | State at 23:08 CEST | What it runs | Metrics / output | Output root |
|---|---|---|---:|---|
| `1004864` | completed | input-gain screen for `mnist_bp_amp_v1_c1`, `mnist_bp_amp_v2_c1`, `mnist_bp_amp_v4_c1`; gains `60,100,160,240,360`; `K=8`; 10 epochs; `PACK_SIZE=3` | `15/15` | `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv3_pad1_perfect_diode_inputgain_screen_k8_seed0_10epoch` |
| `1004880` | completed | collected the diagnostic shared-gain screen with only the three requested amps | shared diagnostic gain `360`, base LR `0.0216`, `num_complete=3`; do not use for main LR/final protocol | same as input-gain root |
| `1004881` | failed before writing metrics | original LR screen; failed from stale staged source where `FlexibleDeepResistiveEnergy` did not accept `trainable_amplification` | `0/15` | `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv3_pad1_perfect_diode_nonlegacy_lr_screen_k8_seed0_10epoch` |
| `1004882`, `1004883` | cancelled/stale dependency branch | old LR collector and final follow-up after failed job `1004881` | n/a | same LR/final roots |
| `1017778` | failed before writing metrics | second LR attempt; still hit the stale top-level `custom_classes.py` shadow, because training imports `custom_classes` from `${SOURCE_ROOT}` before `labs/custom_classes.py` | `0/15` | same LR-screen root |
| `1017779`, `1017780` | cancelled/stale dependency branch | old LR collector and final follow-up after failed job `1017778` | n/a | same LR/final roots |
| `1017895` | canceled by request while running | third shared-gain LR attempt after fixing remote imports; canceled because the shared-gain LR/final protocol is not intended | `0/15` | same LR-screen root |
| `1017896`, `1017897` | canceled by request | shared-gain LR collector and 30-epoch final follow-up | n/a | same LR/final roots |

Conv3 completion tracking:

- Local tmux monitor `conv3_state_monitor` was stopped after canceling the shared-gain LR/final branch.
- Jean Zay check at 2026-06-29 10:16 CEST: no related Conv3 perfect-diode Slurm jobs remain in the queue; shared-gain LR metrics are still `0`, `selected_lr_by_nonlinearity_amp.csv` is missing, and the shared-gain 30-epoch final root is missing.
- Per-amp Conv3 perfect-diode chain launched at 2026-06-29 10:34 CEST:
  - LR screen job `1053390` completed on Jean Zay R3 (`fmu@v100`) with `15/15` metrics. It used `PACK_SIZE=3`, one V100 per LR multiplier, and each task ran the three non-legacy amplification rows with the per-amp selected input gain.
  - LR collector job `1053391` completed and wrote `selected_lr_by_nonlinearity_amp.csv`.
  - Selected LR rows: `mnist_bp_amp_v1_c1` uses LR `0.0432` at input gain `360` with 10-epoch best/final `82.51% / 80.22%`; `mnist_bp_amp_v2_c1` uses LR `0.0864` at input gain `360` with `92.18% / 91.97%`; `mnist_bp_amp_v4_c1` uses LR `0.0864` at input gain `360` with `95.43% / 95.40%`.
  - 30-epoch final job `1053392_0` timed out at the 20-hour walltime limit before writing final metrics. The final root still has `0/3` metrics: `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv3_pad1_perfect_diode_nonlegacy_peramp_bestlr_k8_seed0_30epoch`.
- Conv3 hard-sigmoid stride/v_off trainability control launched at 2026-06-29 23:40 CEST:
  - Initial job `1114179` was canceled because it used scalar `stride=2`, `padding=0` for all conv layers, which did not match the requested Conv3 per-layer stride schedule.
  - Corrected job `1114285` is a four-task Jean Zay R3 (`fmu@v100`) array for `mnist_bp_amp_v1_c1` and `mnist_bp_amp_v4_c1`, each with fixed `v_off=4.0` and trainable `v_off` initialized at `4.0`. As of 2026-06-30 11:17 CEST, fixed-`v_off` tasks `1114285_0` and `1114285_2` are still running, and no metrics are written yet.
  - Trainable-`v_off` tasks `1114285_1` and `1114285_3` failed immediately because the staged remote `labs/custom_minimizer.py` still added fallback `v_min/v_max` to `hard_sigmoid_param` even when `v_off` was supplied. Synced the guarded local `labs/custom_minimizer.py` and `model/resistive/minimizer.py` to Jean Zay and relaunched only the failed trainable rows as job `1129272_[1,3%2]`; that rerun is pending on `fmu@v100`.
  - Architecture/protocol: Conv3 channels `64 128 256`, per-layer strides `[2, 2, 1]` where the final `1` means no additional downsampling on the last conv layer, paddings `[1, 1, 1]`, `K=8`, seed `0`, 30 epochs, batch size `4`, hard-sigmoid `g_on=100`, `g_off=0`.
  - It reuses the previous sat10 calibrated gains and selected LRs: `v1/c1` gain `103.8655776977539`, LR `0.0554562938721`; `v4/c1` gain `296.9048767089844`, LR `0.00970007644173`.
  - Output root: `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv3_s2s2s1_pad1_hardsigmoid_sat10_v1c1_v4c1_voff_trainability_k8_seed0_30epoch`.
- Conv1/Conv2/Conv3 perfect-diode stride `[2,2,1]` padding-1 repeat launched at 2026-06-29 23:59 CEST and completed by 2026-07-01:
  - Staged launcher: `/lustre/fswork/projects/rech/umg/ucy17uy/server_code/experiments/run_mnist_bp_conv123_r3_perfect_diode_s221_pad1_bestparams_jeanzay.slurm`.
  - Initial job `1114618` was canceled immediately because a 24h walltime hit `QOSMaxWallDurationPerJobLimit`.
  - Corrected job `1114636` completed on Jean Zay R3 (`fmu@v100`) as array `0-5%6`, with `6/6` metrics and no pruned rows.
  - Output root: `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv123_s221_pad1_perfect_diode_v1c1_v4c1_bestparams_k4k8_seed0_10_30_50epoch`.
  - Perfect-diode has no hard-sigmoid target-saturation parameter, so this reuses the prior best input-gain operating point and LR per depth/amp: Conv1 `v1/c1` gain `40`, LR `0.006`, 10 epochs, `K=4`, strides `[2]`, padding `[1]`; Conv1 `v4/c1` gain `40`, LR `0.009`, 10 epochs, `K=4`; Conv2 `v1/c1` gain `100`, LR `0.024`, 30 epochs, `K=4`, strides `[2,2]`, paddings `[1,1]`; Conv2 `v4/c1` gain `100`, LR `0.012`, 30 epochs, `K=4`; Conv3 `v1/c1` gain `360`, LR `0.0432`, 50 epochs, `K=8`, strides `[2,2,1]`, paddings `[1,1,1]`; Conv3 `v4/c1` gain `360`, LR `0.0864`, 50 epochs, `K=8`.
  - Completed rows: Conv1 `v1/c1` best/final `96.55% / 96.41%`; Conv1 `v4/c1` `96.76% / 96.67%`; Conv2 `v1/c1` `97.09% / 97.08%`; Conv2 `v4/c1` `98.35% / 98.34%`; Conv3 `v1/c1` `97.31% / 97.27%`; Conv3 `v4/c1` `96.71% / 96.65%`.
- Conv2/Conv3 `v1/c1` perfect-diode gain/LR rescue launched at 2026-06-30 12:25 CEST and completed by 2026-07-01:
  - Goal: test higher input gain and a downward-biased LR grid because the old `v1/c1` LR may be too high.
  - Code changes staged to Jean Zay: `train_mnist_bp_conv_amplification_sweep.py` now supports `--prune-after-epochs` and `--prune-min-best-test-accuracy`; pruned runs still write metrics and `pruned_after_epoch.json`, and the collector excludes pruned rows from selection.
  - Grid job `1131906`, collector `1131907`, and final job `1131908` all completed on Jean Zay R3 (`fmu@v100`). No matching Jean Zay jobs remain queued as of 2026-07-01 11:24 CEST.
  - Grid job `1131906` produced `60/60` metrics with `39` pruned rows. Conv2 had `13/30` pruned rows and `17` surviving candidates; Conv3 had `26/30` pruned rows and only `4` surviving candidates.
  - Conv2 grid: gains `100,140,200,280,400`; LRs `0.0015,0.003,0.006,0.012,0.024,0.048`; `K=4`; prune after 5 epochs if best test accuracy is below the previous Conv2 `v1/c1` epoch-5 baseline `0.9537`.
  - Conv3 grid: gains `360,480,640,900,1200`; LRs `0.0027,0.0054,0.0108,0.0216,0.0432,0.0864`; `K=8`; prune after 5 epochs if best test accuracy is below the previous Conv3 `v1/c1` epoch-5 baseline `0.8921`.
  - Collector job `1131907` selected Conv2 `input_gain=400`, LR `0.012` from a 10-epoch best/final `97.53% / 97.31%`; it selected Conv3 `input_gain=480`, LR `0.0864` from `97.03% / 97.03%`.
  - Final job `1131908` produced `4/4` metrics: Conv2 30 epochs `97.82% / 97.80%`; Conv2 50 epochs `97.87% / 97.77%`; Conv3 30 epochs `97.29% / 97.17%`; Conv3 50 epochs `97.41% / 97.38%`.
  - Interpretation: tuning improved `v1/c1`, especially Conv3, but the pruning gate means the Conv3 LR conclusion is based on only four non-pruned rows; lower-LR rows may still be under-tested if they learn slowly.
  - Grid output root: `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv23_s221_pad1_perfect_diode_v1c1_gain_lr_downbias_k4k8_seed0_10epoch`.
  - Final output root: `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv23_s221_pad1_perfect_diode_v1c1_gain_lr_downbias_bestlr_k4k8_seed0_30_50epoch`.
- Conv3 perfect-diode follow-up investigation launched at 2026-07-01 11:34 CEST:
  - Job `1170774` is running on Jean Zay R3 (`fmu@v100`) as array `0-6%7`, with all seven tasks active at launch.
  - Early logs show the intended Conv3 geometry `[2,28,28] -> [64,14,14] -> [128,7,7] -> [256,7,7] -> [20]`, `perfect_diode`, `K=8`, strides `[2,2,1]`, paddings `[1,1,1]`, and no immediate import/config failures.
  - Goal: test seed stability of the selected `v1/c1` row, seed stability of the `v4/c1` comparator, and no-prune 50-epoch alternatives for rows that the epoch-5 gate may have hidden.
  - Rows: `v1/c1` gain `480`, LR `0.0864`, seeds `1,2`; `v4/c1` gain `360`, LR `0.0864`, seeds `1,2`; `v1/c1` seed `0` no-prune follow-ups at `(gain,LR)=(480,0.0432),(360,0.0864),(640,0.0864)`.
  - Rationale: selected `v1/c1` still has slowly falling test loss at 50 epochs; `gain=480, LR=0.0432` was pruned at epoch 5 despite monotonic improvement; high gains `900/1200` look less promising because accuracy peaked early and degraded.
  - Output root: `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv3_s221_pad1_perfect_diode_investigate_v1v4_k8_seed0_1_2_50epoch`.
- Conv2/Conv3 `v1/c1` massive input-gain LR30 grid launched at 2026-07-01 14:16 CEST:
  - Grid job `1175141` is pending on Jean Zay R3 (`fmu@v100`) as array `0-39%12`; collector job `1175142` is dependency-held on `afterok:1175141`.
  - Goal: directly optimize much larger raw input gains over 30 epochs with no pruning, using an LR grid scaled inversely with input gain.
  - LR scaling rule: `lr = reference_gain * reference_lr / input_gain * lr_multiplier`.
  - Conv2 reference is gain `400`, LR `0.012`; grid gains `800,1200,1600,2400,3200`, LR multipliers `0.25,0.5,1,2`, `K=4`, strides `[2,2]`, paddings `[1,1]`.
  - Conv3 reference is gain `480`, LR `0.0864`; grid gains `960,1440,1920,2880,3840`, LR multipliers `0.25,0.5,1,2`, `K=8`, strides `[2,2,1]`, paddings `[1,1,1]`.
  - Output root: `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv23_s221_pad1_perfect_diode_v1c1_massive_gain_lr30_k4k8_seed0_30epoch`.
- Conv2 perfect-diode `v4/c1` vs high-gain `v1/c1` GPU saturation analysis completed at 2026-07-01 14:30 CEST:
  - GPU job `1175610` completed under dev QoS; redundant pending job `1175598` was canceled after outputs were written.
  - Output root: `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/conv2_perfect_diode_v4_vs_v1_saturation_gpu_20260701`.
  - On 256 test samples, the two good rows have nearly identical initial perfect-diode saturation: `v1/c1` gain `400` h1/h2/all `47.05%/50.58%/48.23%`; `v4/c1` gain `100` `46.93%/50.77%/48.21%`.
  - Final test saturation differs mainly in layer 2: `v1/c1` gain `400` h1/h2/all `50.73%/57.56%/53.01%`; `v4/c1` gain `100` `47.56%/69.81%/54.97%`.
  - Interpretation: high raw `input_gain` makes `v1/c1` good by matching the `v4/c1` first-layer drive, but `v4/c1` remains better because voltage amplification changes internal layer dynamics and yields a much more nonlinear second hidden layer.
- Conv2 perfect-diode learned-conductance scale follow-up completed at 2026-07-01 14:36 CEST:
  - GPU job `1176326` wrote `conv2_pd_v4_vs_v1_conductance_scale.csv` under the same output root.
  - Final raw means for `v4/c1` vs high-gain `v1/c1`: `ConvWeight_0` `0.1339` vs `0.0917`, `ConvWeight_1` `0.00657` vs `0.00634`, `DenseWeight_0` `0.00606` vs `0.00751`.
  - The older gain-100 `v1/c1` row is spikier but worse: `ConvWeight_0` max `2.06` and `ConvWeight_1` max `0.73`, versus high-gain `v1/c1` max `0.53` and `0.23`.
  - Interpretation: high-gain `v1/c1` improved by fixing the operating point and reducing spiky compensation, not by making conductances larger. `v4/c1` still wins without larger deeper raw conductances; its advantage is more consistent with amplification dynamics and much larger learned hidden biases.
- Conv2 perfect-diode gradient follow-up completed at 2026-07-01 14:55 CEST:
  - GPU job `1177541` wrote `conv2_pd_v4_vs_v1_train_gradient_init_final_256_gpu.csv` and `conv2_pd_v4_vs_v1_test_gradient_init_final_256_gpu.csv` under the same output root.
  - On 256 train samples at initialization, `v4/c1` has about `1.32x` larger global relative gradient/step than high-gain `v1/c1`; log-conductance gradients are also `1.31-1.38x` larger across W0/W1/W2.
  - At the final checkpoint, high-gain `v1/c1` has exactly zero `ConvWeight_0` and `ConvWeight_1` gradients (`grad_zero_fraction=1.0`) on train and test diagnostics; only the dense/readout weight has nonzero gradient. `v4/c1` keeps nonzero W0/W1/W2 gradients.
  - Interpretation: this is the clearest mechanism so far for why `v4/c1` trains better. It preserves early-conv learning signal while `v1/c1` becomes early-conv-gradient-dead after reaching a decent but lower-accuracy operating point.
- Broader perfect-diode gradient survey completed at 2026-07-01:
  - GPU job `1180441` wrote `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/perfect_diode_gradient_survey_20260701/perfect_diode_conv123_gradient_survey_final_train256_gpu.csv`.
  - Final-checkpoint train-gradient survey over 8 trained rows: Conv1 `v1/c1` and `v4/c1` keep nonzero early-conv gradients; old Conv2 `v1/c1` gain `100` and Conv2 `v4/c1` keep nonzero W0/W1 gradients; tuned Conv2 `v1/c1` gain `400` has zero W0/W1 gradients; all checked Conv3 rows (`v1/c1` repeat, tuned `v1/c1`, and `v4/c1`) have zero W0/W1 gradients while W2/dense remain nonzero.
  - Interpretation refinement: exact early-conv gradient death is real but not unique to the high-gain Conv2 `v1/c1` row. It appears to be a depth/operating-point failure mode of hard-clipped perfect-diode BP. Conv2 `v4/c1` is notable because it avoids this mode while also giving the best Conv2 accuracy.
- Perfect-diode async/K gradient sanity check completed at 2026-07-01:
  - GPU job `1182068` wrote K-sweep CSVs under `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/perfect_diode_gradient_k_sweep_20260701`.
  - Checked configs all use `energy_minimizer.mode=asynchronous`. Conv2 source K is `4`; Conv3 source K is `8`.
  - At source K: Conv2 old `v1/c1` gain `100` has nonzero init/final W0/W1 gradients; Conv2 tuned `v1/c1` gain `400` has nonzero init W0/W1 but zero final W0/W1; Conv2 `v4/c1` gain `100` has nonzero init/final W0/W1; Conv3 tuned `v1/c1` gain `480` has zero init/final W0/W1 at source K=8.
  - Changing K matters qualitatively. Conv2 W0/W1 gradients are generally nonzero at `K=2/4` but zero at `K=1/8/16`; Conv3 tuned `v1/c1` has nonzero final W0/W1 at shorter `K=2/4` but zero at source `K=8`. So "dead" means exact zero under the source asynchronous unroll length, not under every possible K.
- Perfect-diode fixed-inference/fixed-tail gradient probe completed at 2026-07-01 16:47 CEST:
  - Implemented `experiments/run_mnist_bp_perfect_diode_gradient_tk_sweep_{gpu,seq_gpu}_jeanzay.slurm` and `experiments/summarize_mnist_bp_perfect_diode_gradient_tk_sweep.py`.
  - Dev jobs `1186381`, `1186505`, and `1186667` ran `T={8,4,16}`, `K={1,2,4,6,8}` on the same representative Conv2/Conv3 perfect-diode checkpoints, using 128 train samples. The initially queued full sequential job `1186263` was canceled after the dev probes wrote all outputs.
  - Output root: `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/perfect_diode_gradient_tk_sweep_20260701`; summary CSV/MD: `perfect_diode_gradient_tk_sweep_final_summary.{csv,md}`.
  - Result: changing only the BP tail length at fixed `T=8` does not revive early-conv gradients. For Conv3 `v1/c1` gain `480`, W0/W1 log-conductance gradients are exactly zero for every `K=1,2,4,6,8` at `T=8`, and also at `T=16`.
  - Reducing inference settling to `T=4` does revive early gradients for Conv3 and most non-tuned Conv2 rows when `K>=2`: tuned Conv3 `v1/c1` gain `480` reaches W0/W1 logG gradients `7.04e-3 / 1.86e-3` at `T=4,K=4`; Conv3 `v4/c1` reaches `1.70e-2 / 2.73e-3`. `K=1` remains dead.
  - Interpretation update: the earlier coupled K-sweep revival was mostly an inference-settling effect, not just "shorter BP through the last K iterations." Once the free state is settled with `T=8`, the hard-clipped branch has zero W0/W1 Jacobian and changing the BP tail alone cannot recover it.
- 10-input version of the same fixed-`T`/fixed-`K` probe completed at 2026-07-01 17:32 CEST:
  - Dev job `1189725` reran the full `T={4,8,16}`, `K={1,2,4,6,8}` grid with `MAX_SAMPLES=10`.
  - Output root: `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/perfect_diode_gradient_tk_sweep_n10_20260701`; summary CSV/MD: `perfect_diode_gradient_tk_sweep_final_summary.{csv,md}`.
  - The zero/nonzero pattern is unchanged from the 128-sample probe. `T=4,K>=2` revives W0/W1 for Conv3 and most non-tuned Conv2 rows; `T=8` and `T=16` stay dead for every checked K; tuned Conv2 `v1/c1` gain `400` stays dead for all checked `T,K`.
  - Magnitudes change with the sampled inputs, but the conclusion does not. For tuned Conv3 `v1/c1` gain `480`, `T=4,K=4` gives W0/W1 logG gradients `3.53e-3 / 1.45e-3`; Conv3 `v4/c1` gives `6.94e-3 / 2.42e-3`. Both are exactly zero at `T=8/16` for all K.
- Output-node follow-up relaunched as a 10-epoch smoke at 2026-07-01 17:45 CEST:
  - Corrected hypothesis: test output dimension `10` instead of the existing paired-output dimension `20`. Existing output-20 checkpoints cannot be reloaded as output-10 models, so this requires new checkpoints.
  - The initial 30/50-epoch output-10 chain `1190233 -> 1190234` was canceled after tasks `0-2` had run for about 34 seconds; tasks `3-5` and the diagnostic were still pending.
  - Added and staged 10-epoch replacements `experiments/run_mnist_bp_conv23_r3_perfect_diode_output10_representative_10epoch_jeanzay.slurm` and `experiments/run_mnist_bp_perfect_diode_output10_10epoch_gradient_tk_sweep_seq_gpu_jeanzay.slurm`.
  - Training array `1190525`, Jean Zay R3 `fmu@v100`, pending on priority as array `0-5%6`. It trains the same six representative rows for 10 epochs with `output_dim=10`: Conv2 old `v1/c1`, Conv2 tuned `v1/c1`, Conv2 `v4/c1`, Conv3 repeat `v1/c1`, Conv3 tuned `v1/c1`, Conv3 `v4/c1`.
  - Dependent diagnostic job `1190531` is pending on `afterok:1190525` and will run `T={4,8,16}`, `K={1,2,4,6,8}`, `MAX_SAMPLES=128` on the output-10 10-epoch checkpoints.
  - Training root: `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv23_s221_pad1_perfect_diode_output10_representative_k4k8_seed0_10epoch`.
  - Diagnostic root: `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/perfect_diode_output10_10epoch_gradient_tk_sweep_20260701`.
  - Completed by 2026-07-01 19:54 CEST: training array `1190525` completed `6/6`; diagnostic `1190531` completed successfully.
  - 10-epoch output-10 best/final test accuracy: Conv2 old `v1/c1` `96.49% / 96.49%`; Conv2 tuned `v1/c1` `95.99% / 95.94%`; Conv2 `v4/c1` `97.61% / 97.61%`; Conv3 repeat `v1/c1` `63.30% / 63.30%`; Conv3 tuned `v1/c1` `78.26% / 78.26%`; Conv3 `v4/c1` `74.48% / 73.41%`.
  - Initial-gradient answer: at source protocol, Conv2 initial W0/W1 gradients are alive (`T=4,K=4`), but Conv3 initial W0/W1 gradients are zero at source `T=8,K=8`. This holds for output-20 and output-10. For output-10 at `T=4,K=4`, initial Conv2 W0/W1 are nonzero, while initial Conv3 `v1/c1` remains zero and Conv3 `v4/c1` is nonzero only under the shorter `T=4` diagnostic. At `T=8/16`, initial W0/W1 are zero for all checked Conv3 rows and also for Conv2 if over-settled.

Conv3 perfect-diode launch details:

- Implemented collector support for `--run-names` so the three-amp input-gain screen is not treated as missing `v1/c2` and `v1/c4`.
- Staged the three new Slurm launchers and updated collector to `/lustre/fswork/projects/rech/umg/ucy17uy/server_code/experiments`.
- Spot-checked early logs for job `1004864`: runs start with Conv3 padding `1`, `perfect_diode`, `K=8`, seed `0`, batch size `4`, and one V100 allocation launches the three requested amp runs.
- After jobs `1004881` and `1017778` failed from the stale staged energy-class signature, found that the batch import path was using a remote-only top-level `custom_classes.py` shadow. Replaced that shadow with the current `labs/custom_classes.py`, recreated corrupt remote `model/resistive/layer.py`, verified the actual top-level import under `pytorch-gpu/py3/2.5.0`, then relaunched the LR/final dependency chain as `1017895 -> 1017896 -> 1017897`.
- The relaunched branch still used the shared-gain protocol, so it was canceled by request before any LR metrics were written. The main perfect-diode follow-up should instead select gain separately per amp, then run per-amp LR and 30-epoch follow-up.
- Per-amp selected gains from `selected_input_gain_by_nonlinearity_amp.csv`: `mnist_bp_amp_v1_c1=360`, `mnist_bp_amp_v2_c1=360`, `mnist_bp_amp_v4_c1=360`. These happen to match, but they are selected independently per amp, not by the shared mean rule.
- Added and staged per-amp Jean Zay launchers `run_mnist_bp_conv3_r3_perfect_diode_peramp_lr_screen_jeanzay.slurm` and `run_mnist_bp_conv3_r3_perfect_diode_peramp_bestlr_30epoch_jeanzay.slurm`.

Completed Conv3 R3 hard-sigmoid counts:

- Sat10 LR screen: `38/38` metrics at `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv3_pad1_hardsigmoid_sat10_lr_screen_voff4_k8_seed0_10epoch_full`.
- 30-epoch saturation screen: `28/28` metrics at `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv3_pad1_hardsigmoid_amp_saturation_targets_voff4_sat10_20_30_50_k8_seed0_30epoch`.

Conv3 interpretation so far:

- The best completed Conv3 hard-sigmoid row is voltage amplification `A=4,B=1` at low target saturation.
- Later layers are essentially unsaturated at initialization in the best row: hidden2 and hidden3 are both `0.00%`.
- Legacy `A=4,B=0.25` starts with unintended deep-layer saturation even when first-layer target saturation is low: old sat10 init `[9.9, 0.0, 8.0]%`; old sat30 init `[29.8, 0.0, 44.7]%`.
- Legacy final checkpoints show early-conv gradient death: final `ConvWeight_0` and `ConvWeight_1` gradients were exactly zero for old target saturations `10/20/30/50`.
- The new low-gain legacy diagnostic is testing whether reducing raw input gain avoids the deep-layer saturation/gradient-death failure mode.

## Hopfield EqProp Conv MNIST Saturation

<!-- hopfield-eqprop-autoupdate:start -->
- Updated: 2026-07-04 17:24 CEST.
- All main Hopfield EqProp jobs completed: LR screen `1005921`, collector `1005954`, final training `1017213`, saturation `1017214`, report `1017219`.
- Selected LR multiplier from the 5-epoch screen was `2.0` for Conv1/Conv2/Conv3.
- Final 30-epoch test accuracy over seeds `0,1,2`: Conv1 best/final `97.29%/97.27%`, Conv2 `97.61%/97.50%`, Conv3 `97.01%/97.01%`.
- Conv2 stride-1 diagnostic completed: final job `1017856`, saturation job `1017857`. Test best/final over seeds `0,1,2`: `98.88%/98.32%`.
- Conv2 saturation, stride-2, test split, final checkpoints: all-unit total `52.44%`, low/high `50.22%/2.22%`, sample p50/p90 `52.49%/53.26%`, layer totals h1 `40.09%`, h2 `81.45%`.
- Conv2 saturation, stride-1, test split, final checkpoints: all-unit total `77.28%`, low/high `76.93%/0.35%`, sample p50/p90 `77.30%/77.66%`, layer totals h1 `49.62%`, h2 `93.51%`.
- Interpretation: stride-1 Conv2 is both more accurate and much more saturated, especially in hidden layer 2. The extra saturation is almost entirely low-bound saturation, not high-bound saturation.
- MNIST Conv1 stride-1 completed: final job `1048788`, saturation job `1048789`. Test best/final over seeds `0,1,2`: `98.21%/98.16%`. Final-checkpoint test saturation: all-unit total `85.33%`, low/high `85.32%/0.001%`, sample p50/p90 `85.24%/86.51%`, layer total h1 `85.33%`. Initial test total saturation was `50.26%`.
- MNIST Conv3 stride-1 completed: final job `1054868`, saturation job `1054869`. Test best/final over seeds `0,1,2`: `99.13%/99.04%`. Final-checkpoint test saturation: all-unit total `78.83%`, low/high `78.02%/0.82%`, layer totals h1 `57.01%`, h2 `49.31%`, h3 `99.05%`.
- MNIST layerwise-stride padding1 diagnostic completed: jobs `1114567 -> 1114568 -> 1114569`. Test best/final after 30 epochs: Conv1 `[2]` `97.06%/97.04%`, Conv2 `[2,2]` `97.43%/97.39%`, Conv3 `[2,2,1]` `97.67%/97.64%`. Final test saturation at the 30-epoch checkpoints: Conv1 `60.28%`, Conv2 `55.19%`, Conv3 `53.69%`.
- MNIST layerwise-stride padding1 continuation completed from the 30-epoch `final_model.pt` checkpoints: job `1129861`, array `0-8%9`, 20 more epochs for Conv1 `[2]`, Conv2 `[2,2]`, Conv3 `[2,2,1]`, seeds `0,1,2`. Best/final after 50 total epochs: Conv1 `97.44%/97.40%`, Conv2 `97.74%/97.63%`, Conv3 `97.95%/97.94%`. Loss did not stagnate: mean test loss from epoch 30 to 50 fell Conv1 `0.07143 -> 0.06344`, Conv2 `0.02765 -> 0.02291`, Conv3 `0.02059 -> 0.01817`. Final-checkpoint test saturation after 50 total epochs: Conv1 `66.27%`; Conv2 `55.28%` with layer totals h1 `41.34%`, h2 `83.15%`; Conv3 `54.77%` with layer totals h1 `30.43%`, h2 `40.88%`, h3 `86.04%`. No per-epoch saturation curve can be reconstructed because only best/final checkpoints were saved. Output root: `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_hopfield_eqprop_conv_strides221_pad1_continue20_20260630`.
- Affine-MNIST Conv2/Conv3 depth-gap runs launched on Jean Zay R3. Shared geometry: Conv2 `[2,2]` pad1 versus Conv3 `[2,2,1]` pad1, 50 epochs, seeds `0,1,2`, `PACK_SIZE=2`. Preset `medium` (`+-25 deg`, translation `20%`, scale `[0.8,1.2]`) smoke `1317786` completed `4/4`; final job `1317803` is running; saturation job `1317804` is pending dependency. Preset `mnist_affine` (`+-60 deg`, translation `15%`, scale `[0.8,1.2]`, shear `+-15 deg`) smoke `1318395` completed `4/4`; final job `1318422` launched but array task `_1` failed on a Slurm stdout I/O error before Python; task `_1` was rerun as `1318449` after redirecting packed-run logs; guarded saturation job `1318467` waits for all six metrics before measuring saturation. Output roots: `mnist_hopfield_eqprop_conv_affine_s221_pad1_20260704` and `mnist_hopfield_eqprop_conv_mnist_affine_s221_pad1_20260704`.
- Fashion-MNIST stride grid completed: final job `1054871`, saturation job `1054872`. Test best/final: Conv1 stride1 `90.12%/89.90%`, Conv1 stride2 `87.74%/87.73%`, Conv2 stride1 `91.75%/91.44%`, Conv2 stride2 `87.67%/87.64%`, Conv3 stride1 `92.07%/91.61%`, Conv3 stride2 `86.69%/86.42%`.
- Fashion-MNIST layerwise `2,2,0` diagnostic completed: jobs `1114316 -> 1114317 -> 1114318`. Effective Conv3 strides `[2,2,1]`; test best/final Conv3 `87.96%/87.81%`, better than Conv3 `[2,2,2]` `86.69%/86.42%` but far below Conv3 `[1,1,1]` `92.07%/91.61%`. Final test saturation for Conv3 `[2,2,1]` was `55.14%` total.
- Tanh centered-nonlinearity comparison implemented locally: Hopfield hidden
  `tanh`, same primary MNIST inputs as Axel and the hard-sigmoid runs
  (`[0,1]`), centered `[-1,1]` fallback staged if the first tests are clearly
  worse. New diagnostics report tanh near-bound fractions and signal/gradient
  sizes, not subset accuracy. Jean Zay identity-input first-test chain
  submitted: smoke `1136935`, LR screen `1136936` after smoke, collector
  `1136937` after LR. Smoke `1136935` completed successfully with `5/5`
  metrics, including Conv3 stride-1 and Conv3 `[2,2,1]` padding1 memory checks.
  Identity LR screen `1136936` and collector `1136937` completed successfully
  with `45/45` LR metrics. Best identity-input tanh 5-epoch mean over seeds was
  Conv1 `88.07%`, Conv2 `93.20%`, Conv3 `95.29%`, all at LR multiplier `4.0`.
  Conv1/Conv2 are clearly below hard-sigmoid references, so centered-input
  fallback was launched and completed: LR screen `1146084`, collector
  `1146085`, `45/45` metrics. Centered-input tanh 5-epoch best mean over seeds:
  Conv1 `87.23%`, Conv2 `92.15%`, Conv3 `95.90%`, all at LR multiplier `4.0`.
  Centering did not help Conv1/Conv2 but did help Conv3. Final tanh training now
  uses Conv1/Conv2 identity inputs, Conv3 centered inputs, LR multiplier `4.0`;
  final job `1172067` has tasks `0,1` running and tasks `2-4` pending on
  resources; diagnostics job `1172068` is `PD (Dependency)`.
- Result root: `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_hopfield_eqprop_conv_saturation_20260628`.
- Local CSV copies: `/home/filip/server_code/results/mnist_hopfield_eqprop_conv_saturation_20260628`.
<!-- hopfield-eqprop-autoupdate:end -->

<!-- jeanzay-conv3-finish-report:start -->
## Jean Zay Conv3 Completion Report

Updated: 2026-06-30 14:13 CEST

Tracked jobs: `1004671`, `1004672`, `1004864`, `1004880`, `1004881`, `1004882`, `1004883`, `1017778`, `1017779`, `1017780`, `1017895`, `1017896`, `1017897`, `1053390`, `1053391`, `1053392`.

### Metrics Counts

| Result set | Metrics | Expected | Root |
|---|---:|---:|---|
| hard_sigmoid legacy low gain | 5 | 5 | `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv3_pad1_hardsigmoid_legacy_lowgain_voff4_k8_seed0_10epoch` |
| hard_sigmoid sat30 LR | 0 | 15 | `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv3_pad1_hardsigmoid_nonlegacy_sat30_lr_screen_voff4_k8_seed0_10epoch` |
| perfect_diode input gain | 15 | 15 | `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv3_pad1_perfect_diode_inputgain_screen_k8_seed0_10epoch` |
| perfect_diode per-amp LR | 15 | 15 | `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv3_pad1_perfect_diode_nonlegacy_peramp_lr_screen_k8_seed0_10epoch` |
| perfect_diode per-amp final 30 epoch | 0 | 3 | `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv3_pad1_perfect_diode_nonlegacy_peramp_bestlr_k8_seed0_30epoch` |

### Best Rows

#### hard_sigmoid legacy low gain

| Run | Best / final acc | Best epoch | Gain | LR mult | LR | K | Epochs | Target |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| mnist_bp_amp_v4_c0p25 | 83.77% / 83.77% | 10 | `200` | `0.5` | `-ult_0.5` | `8` | `10` | `` |
| mnist_bp_amp_v4_c0p25 | 82.81% / 82.81% | 10 | `250` | `0.5` | `-ult_0.5` | `8` | `10` | `` |
| mnist_bp_amp_v4_c0p25 | 75.47% / 73.83% | 9 | `150` | `0.5` | `-ult_0.5` | `8` | `10` | `` |
| mnist_bp_amp_v4_c0p25 | 70.42% / 70.00% | 9 | `100` | `0.5` | `-ult_0.5` | `8` | `10` | `` |
| mnist_bp_amp_v4_c0p25 | 48.58% / 48.58% | 10 | `50` | `0.5` | `-ult_0.5` | `8` | `10` | `` |

#### hard_sigmoid sat30 LR

- No completed metric rows found.

#### perfect_diode input gain

| Run | Best / final acc | Best epoch | Gain | LR mult | LR | K | Epochs | Target |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| mnist_bp_amp_v4_c1 | 94.37% / 94.37% | 10 | `360` | `` | `0.0216` | `8` | `10` | `` |
| mnist_bp_amp_v4_c1 | 94.31% / 94.31% | 10 | `240` | `` | `0.0144` | `8` | `10` | `` |
| mnist_bp_amp_v4_c1 | 93.24% / 93.24% | 10 | `160` | `` | `0.0096` | `8` | `10` | `` |
| mnist_bp_amp_v4_c1 | 91.21% / 91.21% | 10 | `100` | `` | `0.006` | `8` | `10` | `` |
| mnist_bp_amp_v2_c1 | 88.35% / 88.35% | 10 | `360` | `` | `0.0216` | `8` | `10` | `` |

#### perfect_diode per-amp LR

| Run | Best / final acc | Best epoch | Gain | LR mult | LR | K | Epochs | Target |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| mnist_bp_amp_v4_c1 | 95.43% / 95.40% | 8 | `360` | `4` | `-ult_4` | `8` | `10` | `` |
| mnist_bp_amp_v4_c1 | 94.37% / 94.37% | 10 | `360` | `1` | `-ult_1` | `8` | `10` | `` |
| mnist_bp_amp_v4_c1 | 92.37% / 79.46% | 8 | `360` | `2` | `-ult_2` | `8` | `10` | `` |
| mnist_bp_amp_v2_c1 | 92.18% / 91.97% | 8 | `360` | `4` | `-ult_4` | `8` | `10` | `` |
| mnist_bp_amp_v4_c1 | 91.64% / 91.00% | 9 | `360` | `0.5` | `-ult_0.5` | `8` | `10` | `` |

#### perfect_diode per-amp final 30 epoch

- No completed metric rows found.

### Slurm Accounting

```text
JobID|JobName|State|ExitCode|Elapsed|AllocTRES
1004671_0|mnist_c3_r3_legacy_lowg|COMPLETED|0:0|01:41:16|billing=1,cpu=20,energy=1315314,gres/gpu=1,mem=40000M,node=1
1004671_0.batch|batch|COMPLETED|0:0|01:41:16|cpu=20,gres/gpu=1,mem=40000M,node=1
1004671_0.extern|extern|COMPLETED|0:0|01:41:16|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1004671_0.0|python|COMPLETED|0:0|01:41:10|cpu=20,gres/gpu=1,mem=40000M,node=1
1004671_1|mnist_c3_r3_legacy_lowg|COMPLETED|0:0|01:30:42|billing=1,cpu=20,energy=1308345,gres/gpu=1,mem=40000M,node=1
1004671_1.batch|batch|COMPLETED|0:0|01:30:42|cpu=20,gres/gpu=1,mem=40000M,node=1
1004671_1.extern|extern|COMPLETED|0:0|01:30:42|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1004671_1.0|python|COMPLETED|0:0|01:30:36|cpu=20,gres/gpu=1,mem=40000M,node=1
1004671_2|mnist_c3_r3_legacy_lowg|COMPLETED|0:0|01:28:26|billing=1,cpu=20,energy=1040666,gres/gpu=1,mem=40000M,node=1
1004671_2.batch|batch|COMPLETED|0:0|01:28:26|cpu=20,gres/gpu=1,mem=40000M,node=1
1004671_2.extern|extern|COMPLETED|0:0|01:28:26|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1004671_2.0|python|COMPLETED|0:0|01:28:20|cpu=20,gres/gpu=1,mem=40000M,node=1
1004671_3|mnist_c3_r3_legacy_lowg|COMPLETED|0:0|01:36:50|billing=1,cpu=20,energy=1348294,gres/gpu=1,mem=40000M,node=1
1004671_3.batch|batch|COMPLETED|0:0|01:36:50|cpu=20,gres/gpu=1,mem=40000M,node=1
1004671_3.extern|extern|COMPLETED|0:0|01:36:50|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1004671_3.0|python|COMPLETED|0:0|01:36:44|cpu=20,gres/gpu=1,mem=40000M,node=1
1004671_4|mnist_c3_r3_legacy_lowg|COMPLETED|0:0|01:44:26|billing=1,cpu=20,energy=1452359,gres/gpu=1,mem=40000M,node=1
1004671_4.batch|batch|COMPLETED|0:0|01:44:26|cpu=20,gres/gpu=1,mem=40000M,node=1
1004671_4.extern|extern|COMPLETED|0:0|01:44:26|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1004671_4.0|python|COMPLETED|0:0|01:44:20|cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_0|mnist_c3_r3_lr30_v|TIMEOUT|0:0|03:00:26|billing=1,cpu=20,energy=2479441,gres/gpu=1,mem=40000M,node=1
1004672_0.batch|batch|CANCELLED|0:15|03:00:27|cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_0.extern|extern|COMPLETED|0:0|03:00:26|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_0.0|python|CANCELLED|0:15|03:00:22|cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_1|mnist_c3_r3_lr30_v|TIMEOUT|0:0|03:00:26|billing=1,cpu=20,energy=2479439,gres/gpu=1,mem=40000M,node=1
1004672_1.batch|batch|CANCELLED|0:15|03:00:27|cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_1.extern|extern|COMPLETED|0:0|03:00:26|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_1.0|python|CANCELLED|0:15|03:00:20|cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_2|mnist_c3_r3_lr30_v|TIMEOUT|0:0|03:00:26|billing=1,cpu=20,energy=2562839,gres/gpu=1,mem=40000M,node=1
1004672_2.batch|batch|CANCELLED|0:15|03:00:27|cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_2.extern|extern|COMPLETED|0:0|03:00:26|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_2.0|python|CANCELLED|0:15|03:00:22|cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_3|mnist_c3_r3_lr30_v|TIMEOUT|0:0|03:00:26|billing=1,cpu=20,energy=2562842,gres/gpu=1,mem=40000M,node=1
1004672_3.batch|batch|CANCELLED|0:15|03:00:27|cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_3.extern|extern|COMPLETED|0:0|03:00:26|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_3.0|python|CANCELLED|0:15|03:00:22|cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_4|mnist_c3_r3_lr30_v|TIMEOUT|0:0|03:00:26|billing=1,cpu=20,energy=2621801,gres/gpu=1,mem=40000M,node=1
1004672_4.batch|batch|CANCELLED|0:15|03:00:27|cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_4.extern|extern|COMPLETED|0:0|03:00:26|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_4.0|python|CANCELLED|0:15|03:00:22|cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_5|mnist_c3_r3_lr30_v|TIMEOUT|0:0|03:00:26|billing=1,cpu=20,energy=2621798,gres/gpu=1,mem=40000M,node=1
1004672_5.batch|batch|CANCELLED|0:15|03:00:27|cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_5.extern|extern|COMPLETED|0:0|03:00:26|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_5.0|python|CANCELLED|0:15|03:00:22|cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_6|mnist_c3_r3_lr30_v|TIMEOUT|0:0|03:00:26|billing=1,cpu=20,energy=2621797,gres/gpu=1,mem=40000M,node=1
1004672_6.batch|batch|CANCELLED|0:15|03:00:27|cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_6.extern|extern|COMPLETED|0:0|03:00:26|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_6.0|python|CANCELLED|0:15|03:00:22|cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_7|mnist_c3_r3_lr30_v|TIMEOUT|0:0|03:00:26|billing=1,cpu=20,energy=2621803,gres/gpu=1,mem=40000M,node=1
1004672_7.batch|batch|CANCELLED|0:15|03:00:27|cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_7.extern|extern|COMPLETED|0:0|03:00:26|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_7.0|python|CANCELLED|0:15|03:00:22|cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_8|mnist_c3_r3_lr30_v|TIMEOUT|0:0|03:00:26|billing=1,cpu=20,energy=3432689,gres/gpu=1,mem=40000M,node=1
1004672_8.batch|batch|CANCELLED|0:15|03:00:27|cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_8.extern|extern|COMPLETED|0:0|03:00:26|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_8.0|python|CANCELLED|0:15|03:00:22|cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_9|mnist_c3_r3_lr30_v|TIMEOUT|0:0|03:00:26|billing=1,cpu=20,energy=3411839,gres/gpu=1,mem=40000M,node=1
1004672_9.batch|batch|CANCELLED|0:15|03:00:27|cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_9.extern|extern|COMPLETED|0:0|03:00:26|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_9.0|python|CANCELLED|0:15|03:00:22|cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_10|mnist_c3_r3_lr30_v|TIMEOUT|0:0|03:00:26|billing=1,cpu=20,energy=3411853,gres/gpu=1,mem=40000M,node=1
1004672_10.batch|batch|CANCELLED|0:15|03:00:27|cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_10.extern|extern|COMPLETED|0:0|03:00:26|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_10.0|python|CANCELLED|0:15|03:00:22|cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_11|mnist_c3_r3_lr30_v|TIMEOUT|0:0|03:00:26|billing=1,cpu=20,energy=2921983,gres/gpu=1,mem=40000M,node=1
1004672_11.batch|batch|CANCELLED|0:15|03:00:27|cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_11.extern|extern|COMPLETED|0:0|03:00:26|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_11.0|python|CANCELLED|0:15|03:00:22|cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_12|mnist_c3_r3_lr30_v|TIMEOUT|0:0|03:00:26|billing=1,cpu=20,energy=2921986,gres/gpu=1,mem=40000M,node=1
1004672_12.batch|batch|CANCELLED|0:15|03:00:27|cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_12.extern|extern|COMPLETED|0:0|03:00:26|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_12.0|python|CANCELLED|0:15|03:00:22|cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_13|mnist_c3_r3_lr30_v|TIMEOUT|0:0|03:00:26|billing=1,cpu=20,energy=2921991,gres/gpu=1,mem=40000M,node=1
1004672_13.batch|batch|CANCELLED|0:15|03:00:27|cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_13.extern|extern|COMPLETED|0:0|03:00:26|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_13.0|python|CANCELLED|0:15|03:00:22|cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_14|mnist_c3_r3_lr30_v|TIMEOUT|0:0|03:00:26|billing=1,cpu=20,energy=2726772,gres/gpu=1,mem=40000M,node=1
1004672_14.batch|batch|CANCELLED|0:15|03:00:27|cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_14.extern|extern|COMPLETED|0:0|03:00:26|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1004672_14.0|python|CANCELLED|0:15|03:00:20|cpu=20,gres/gpu=1,mem=40000M,node=1
1004864_0|mnist_c3_pd_ig|COMPLETED|0:0|02:36:23|billing=1,cpu=20,energy=2866648,gres/gpu=1,mem=40000M,node=1
1004864_0.batch|batch|COMPLETED|0:0|02:36:23|cpu=20,gres/gpu=1,mem=40000M,node=1
1004864_0.extern|extern|COMPLETED|0:0|02:36:23|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1004864_1|mnist_c3_pd_ig|COMPLETED|0:0|02:23:32|billing=1,cpu=20,energy=2184143,gres/gpu=1,mem=40000M,node=1
1004864_1.batch|batch|COMPLETED|0:0|02:23:32|cpu=20,gres/gpu=1,mem=40000M,node=1
1004864_1.extern|extern|COMPLETED|0:0|02:23:32|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1004864_2|mnist_c3_pd_ig|COMPLETED|0:0|02:26:16|billing=1,cpu=20,energy=2381436,gres/gpu=1,mem=40000M,node=1
1004864_2.batch|batch|COMPLETED|0:0|02:26:16|cpu=20,gres/gpu=1,mem=40000M,node=1
1004864_2.extern|extern|COMPLETED|0:0|02:26:16|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1004864_3|mnist_c3_pd_ig|COMPLETED|0:0|02:22:03|billing=1,cpu=20,energy=2318213,gres/gpu=1,mem=40000M,node=1
1004864_3.batch|batch|COMPLETED|0:0|02:22:03|cpu=20,gres/gpu=1,mem=40000M,node=1
1004864_3.extern|extern|COMPLETED|0:0|02:22:03|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1004864_4|mnist_c3_pd_ig|COMPLETED|0:0|02:22:03|billing=1,cpu=20,energy=2318208,gres/gpu=1,mem=40000M,node=1
1004864_4.batch|batch|COMPLETED|0:0|02:22:03|cpu=20,gres/gpu=1,mem=40000M,node=1
1004864_4.extern|extern|COMPLETED|0:0|02:22:03|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1004880_0|mnist_c3_pd_ig|COMPLETED|0:0|00:00:08|billing=1,cpu=20,energy=2492,gres/gpu=1,mem=40000M,node=1
1004880_0.batch|batch|COMPLETED|0:0|00:00:08|cpu=20,gres/gpu=1,mem=40000M,node=1
1004880_0.extern|extern|COMPLETED|0:0|00:00:08|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1004881_0|mnist_c3_pd_lr|FAILED|1:0|00:00:17|billing=1,cpu=20,energy=5142,gres/gpu=1,mem=40000M,node=1
1004881_0.batch|batch|FAILED|1:0|00:00:17|cpu=20,gres/gpu=1,mem=40000M,node=1
1004881_0.extern|extern|COMPLETED|0:0|00:00:17|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1004881_1|mnist_c3_pd_lr|FAILED|1:0|00:00:22|billing=1,cpu=20,energy=7503,gres/gpu=1,mem=40000M,node=1
1004881_1.batch|batch|FAILED|1:0|00:00:22|cpu=20,gres/gpu=1,mem=40000M,node=1
1004881_1.extern|extern|COMPLETED|0:0|00:00:22|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1004881_2|mnist_c3_pd_lr|FAILED|1:0|00:00:22|billing=1,cpu=20,energy=7240,gres/gpu=1,mem=40000M,node=1
1004881_2.batch|batch|FAILED|1:0|00:00:22|cpu=20,gres/gpu=1,mem=40000M,node=1
1004881_2.extern|extern|COMPLETED|0:0|00:00:22|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1004881_3|mnist_c3_pd_lr|FAILED|1:0|00:00:22|billing=1,cpu=20,energy=4965,gres/gpu=1,mem=40000M,node=1
1004881_3.batch|batch|FAILED|1:0|00:00:22|cpu=20,gres/gpu=1,mem=40000M,node=1
1004881_3.extern|extern|COMPLETED|0:0|00:00:22|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1004881_4|mnist_c3_pd_lr|FAILED|1:0|00:00:22|billing=1,cpu=20,energy=4966,gres/gpu=1,mem=40000M,node=1
1004881_4.batch|batch|FAILED|1:0|00:00:22|cpu=20,gres/gpu=1,mem=40000M,node=1
1004881_4.extern|extern|COMPLETED|0:0|00:00:22|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1004882_[0]|mnist_c3_pd_lr|CANCELLED by 306943|0:0|00:00:00|
1004883_[0%1]|mnist_c3_pd_30ep|CANCELLED by 306943|0:0|00:00:00|
1017778_0|mnist_c3_pd_lr|FAILED|1:0|00:00:09|billing=1,cpu=20,energy=2137,gres/gpu=1,mem=40000M,node=1
1017778_0.batch|batch|FAILED|1:0|00:00:09|cpu=20,gres/gpu=1,mem=40000M,node=1
1017778_0.extern|extern|COMPLETED|0:0|00:00:09|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1017778_1|mnist_c3_pd_lr|FAILED|1:0|00:00:08|billing=1,cpu=20,energy=2192,gres/gpu=1,mem=40000M,node=1
1017778_1.batch|batch|FAILED|1:0|00:00:08|cpu=20,gres/gpu=1,mem=40000M,node=1
1017778_1.extern|extern|COMPLETED|0:0|00:00:08|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1017778_2|mnist_c3_pd_lr|FAILED|1:0|00:00:08|billing=1,cpu=20,energy=2193,gres/gpu=1,mem=40000M,node=1
1017778_2.batch|batch|FAILED|1:0|00:00:08|cpu=20,gres/gpu=1,mem=40000M,node=1
1017778_2.extern|extern|COMPLETED|0:0|00:00:08|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1017778_3|mnist_c3_pd_lr|FAILED|1:0|00:00:09|billing=1,cpu=20,energy=2071,gres/gpu=1,mem=40000M,node=1
1017778_3.batch|batch|FAILED|1:0|00:00:09|cpu=20,gres/gpu=1,mem=40000M,node=1
1017778_3.extern|extern|COMPLETED|0:0|00:00:09|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1017778_4|mnist_c3_pd_lr|FAILED|1:0|00:00:20|billing=1,cpu=20,energy=3981,gres/gpu=1,mem=40000M,node=1
1017778_4.batch|batch|FAILED|1:0|00:00:20|cpu=20,gres/gpu=1,mem=40000M,node=1
1017778_4.extern|extern|COMPLETED|0:0|00:00:20|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1017779_[0]|mnist_c3_pd_lr|CANCELLED by 306943|0:0|00:00:00|
1017780_[0%1]|mnist_c3_pd_30ep|CANCELLED by 306943|0:0|00:00:00|
1017895_0|mnist_c3_pd_lr|CANCELLED by 306943|0:0|00:09:06|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1017895_0.batch|batch|CANCELLED|0:15|00:09:08|cpu=20,gres/gpu=1,mem=40000M,node=1
1017895_0.extern|extern|COMPLETED|0:0|00:09:06|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1017895_1|mnist_c3_pd_lr|CANCELLED by 306943|0:0|00:09:06|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1017895_1.batch|batch|CANCELLED|0:15|00:09:08|cpu=20,gres/gpu=1,mem=40000M,node=1
1017895_1.extern|extern|COMPLETED|0:0|00:09:06|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1017895_2|mnist_c3_pd_lr|CANCELLED by 306943|0:0|00:09:06|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1017895_2.batch|batch|CANCELLED|0:15|00:09:08|cpu=20,gres/gpu=1,mem=40000M,node=1
1017895_2.extern|extern|COMPLETED|0:0|00:09:06|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1017895_3|mnist_c3_pd_lr|CANCELLED by 306943|0:0|00:09:06|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1017895_3.batch|batch|CANCELLED|0:15|00:15:00|cpu=20,gres/gpu=1,mem=40000M,node=1
1017895_3.extern|extern|COMPLETED|0:0|00:14:59|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1017895_4|mnist_c3_pd_lr|CANCELLED by 306943|0:0|00:09:06|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1017895_4.batch|batch|CANCELLED|0:15|00:09:08|cpu=20,gres/gpu=1,mem=40000M,node=1
1017895_4.extern|extern|COMPLETED|0:0|00:09:06|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1017896_[0]|mnist_c3_pd_lr|CANCELLED by 306943|0:0|00:00:00|
1017897_[0%1]|mnist_c3_pd_30ep|CANCELLED by 306943|0:0|00:00:00|
1053390_0|mnist_c3_pd_pamp_lr|COMPLETED|0:0|06:43:38|billing=1,cpu=20,energy=5778995,gres/gpu=1,mem=40000M,node=1
1053390_0.batch|batch|COMPLETED|0:0|06:43:38|cpu=20,gres/gpu=1,mem=40000M,node=1
1053390_0.extern|extern|COMPLETED|0:0|06:43:38|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1053390_1|mnist_c3_pd_pamp_lr|COMPLETED|0:0|07:01:34|billing=1,cpu=20,energy=6052627,gres/gpu=1,mem=40000M,node=1
1053390_1.batch|batch|COMPLETED|0:0|07:01:34|cpu=20,gres/gpu=1,mem=40000M,node=1
1053390_1.extern|extern|COMPLETED|0:0|07:01:34|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1053390_2|mnist_c3_pd_pamp_lr|COMPLETED|0:0|07:05:24|billing=1,cpu=20,energy=7014282,gres/gpu=1,mem=40000M,node=1
1053390_2.batch|batch|COMPLETED|0:0|07:05:24|cpu=20,gres/gpu=1,mem=40000M,node=1
1053390_2.extern|extern|COMPLETED|0:0|07:05:24|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1053390_3|mnist_c3_pd_pamp_lr|COMPLETED|0:0|07:12:32|billing=1,cpu=20,energy=6445392,gres/gpu=1,mem=40000M,node=1
1053390_3.batch|batch|COMPLETED|0:0|07:12:32|cpu=20,gres/gpu=1,mem=40000M,node=1
1053390_3.extern|extern|COMPLETED|0:0|07:12:33|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1053390_4|mnist_c3_pd_pamp_lr|COMPLETED|0:0|07:08:42|billing=1,cpu=20,energy=6387832,gres/gpu=1,mem=40000M,node=1
1053390_4.batch|batch|COMPLETED|0:0|07:08:42|cpu=20,gres/gpu=1,mem=40000M,node=1
1053390_4.extern|extern|COMPLETED|0:0|07:08:42|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1053391_0|mnist_c3_pd_pamp_lr|COMPLETED|0:0|00:00:09|billing=1,cpu=20,energy=1795,gres/gpu=1,mem=40000M,node=1
1053391_0.batch|batch|COMPLETED|0:0|00:00:09|cpu=20,gres/gpu=1,mem=40000M,node=1
1053391_0.extern|extern|COMPLETED|0:0|00:00:09|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1053392_0|mnist_c3_pd_pamp30|TIMEOUT|0:0|20:00:24|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
1053392_0.batch|batch|CANCELLED|0:15|20:00:25|cpu=20,gres/gpu=1,mem=40000M,node=1
1053392_0.extern|extern|COMPLETED|0:0|20:00:24|billing=1,cpu=20,gres/gpu=1,mem=40000M,node=1
```

<!-- jeanzay-conv3-finish-report:end -->

## 2026-07-02 Perfect-Diode Weight-Movement Audit

Compared reconstructed seed-0 initializations against `best_model.pt` and
`final_model.pt` on Jean Zay for the representative perfect-diode Conv2/Conv3
runs. Only best/final checkpoints are saved, so this is not a per-epoch
trajectory audit.

Result: weights generally did change. For output-20 Conv3 final checkpoints,
the early conv weights moved substantially from initialization:

- `v1/c1`, gain `360`, LR `0.0432`, 50 epochs: W0/W1 relative L2 deltas
  `2.33/1.44`.
- tuned `v1/c1`, gain `480`, LR `0.0864`, 50 epochs: W0/W1 relative L2 deltas
  `2.01/1.24`.
- `v4/c1`, gain `360`, LR `0.0864`, 50 epochs: W0/W1 relative L2 deltas
  `0.53/0.35`; these first two conv layers were unchanged from best to final.

Exception: output-10 10-epoch Conv3 repeat `v1/c1`, gain `360`, LR `0.0432`,
kept W0/W1 exactly at initialization in both best and final checkpoints
(`delta_l2=0`, `changed_fraction=0`), while W2/dense still moved strongly.
This supports the interpretation that some dead-gradient configurations truly
freeze early conv layers, but the output-20 representative runs did not leave
all weights fixed.

## Shared Result Roots

- Jean Zay source checkout: `/lustre/fswork/projects/rech/umg/ucy17uy/server_code`
- Jean Zay R3 results base: `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results`
- Local Conv1/Conv2 saturation summary source: `/home/filip/server_code/results/good_conv_saturation_analysis/good_conv_saturation_best_per_amp_by_layer.csv`
- Local Conv3 saturation analysis source: `/home/filip/server_code/results/mnist_bp_conv3_pad1_hardsigmoid_amp_saturation_targets_voff4_sat10_20_30_50_k8_seed0_30epoch/saturation_init_final_train256.csv`

## Next Actions

1. Aggregate the completed `1004671` legacy low-gain metrics; decide whether the timed-out sat30 LR screen `1004672` should be rerun with longer walltime or smaller packing.
2. Compare the completed tuned Conv3 perfect-diode `v1/c1` 50-epoch row (`97.41% / 97.38%`) against the completed Conv3 hard-sigmoid best row (`96.71% / 96.64%`), including whether the improvement is architecture/operating-point or nonlinearity-driven.
3. Track Conv3 perfect-diode investigation job `1170774`; compare selected `v1/c1` seed stability, `v4/c1` comparator seeds, and the no-prune lower-LR follow-up.
4. Track Conv2/Conv3 massive-gain LR30 grid `1175141 -> 1175142`; compare selected rows against Conv2 `v4/c1` `98.35%` and Conv3 tuned `v1/c1` `97.41%`.
5. Measure initial/final per-layer saturation for the new Conv3 diagnostic rows.
6. Pick the best legacy low-gain row and run gradient diagnostics; compare `ConvWeight_0/1/2` and dense gradients against the old legacy sat10/sat30 rows.
7. Combine existing sat10 LR results with the new sat30 LR results for Conv3 `v1/c1`, `v2/c1`, and `v4/c1`.
8. Recheck and sync Conv2 50-epoch continuation status on Trex and Akib.
9. When the Hopfield EqProp Conv MNIST saturation jobs in `docs/current_experiments.md` finish, report their completed accuracies and per-layer saturation results here.
