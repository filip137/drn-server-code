# Current Research State

Updated: 2026-06-29 11:38 CEST

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
| `hard_sigmoid` | `mnist_bp_amp_v1_c1`, target sat30, input gain `221.068`, initial `v_off=4.0`, trainable amp, `K=6`, 30 epochs | `97.65% / 97.63%` | first hidden `30.00%`; later-layer saturation and learned amplification diagnostics pending | `/lustre/fsn1/projects/rech/umg/ucy17uy/server_code/results/mnist_bp_conv2_hardsigmoid_trainable_amp_sat30_voff4_seed0_30epoch` |
| `perfect_diode` | `mnist_bp_amp_v4_c1`, input gain `100`, 50 epochs | `98.66% / 98.64%` | hidden1 `46.94%`, hidden2 `50.57%`, all hidden `48.02%` | `/home/filip/server_code/results/mnist_bp_conv2_epbpK_large_50epoch_64_128ch_s2_valid_seed0_local_distributed` |

Current Conv2 hard-sigmoid status:

<!-- conv2-stride1-recalibrated-sat30-autoupdate:start -->
- Conv2 stride-1 hard-sigmoid repeats with stride-1 recalibrated sat30 input gains, launched locally in tmux `conv2_stride1_recal_sat30`, updated 2026-06-29 11:48 CEST.

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
- Conv2 trainable `v_off`-only 50-epoch local run is active in tmux session `conv2_voff_only_50ep`. It has no final `metrics.json` yet, but the latest log shows epoch `23/50` in progress; recent completed test checkpoints were epoch 20 `97.17%`, epoch 21 `97.13%`, and epoch 22 `97.33%`. Output root: `/home/filip/server_code/results/mnist_bp_conv2_trainable_voff4_only_voff_sat30params_seed0_50epoch`.
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
| `perfect_diode` | per-amp LR/final chain launched; results pending | n/a | pending | `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv3_pad1_perfect_diode_nonlegacy_peramp_bestlr_k8_seed0_30epoch` |

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
  - LR screen job `1053390` is running on Jean Zay R3 (`fmu@v100`) as array `0-4%5`, with `PACK_SIZE=3` so each V100 runs the three non-legacy amplification rows for one LR multiplier. At 2026-06-29 11:38 CEST it had `0/15` final metrics written, but all five array tasks were training normally around epoch `2/10`.
  - LR collector job `1053391`, pending after `1053390`, writes `selected_lr_by_nonlinearity_amp.csv`.
  - 30-epoch final job `1053392`, pending after `1053391`, expected `3/3` metrics, root `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv3_pad1_perfect_diode_nonlegacy_peramp_bestlr_k8_seed0_30epoch`.
  - Local tmux monitor `conv3_state_monitor` tracks these jobs and will write the completion report into this file.

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
- Updated: 2026-06-29 11:26 CEST.
- All main Hopfield EqProp jobs completed: LR screen `1005921`, collector `1005954`, final training `1017213`, saturation `1017214`, report `1017219`.
- Selected LR multiplier from the 5-epoch screen was `2.0` for Conv1/Conv2/Conv3.
- Final 30-epoch test accuracy over seeds `0,1,2`: Conv1 best/final `97.29%/97.27%`, Conv2 `97.61%/97.50%`, Conv3 `97.01%/97.01%`.
- Conv2 stride-1 diagnostic completed: final job `1017856`, saturation job `1017857`. Test best/final over seeds `0,1,2`: `98.88%/98.32%`.
- Conv2 saturation, stride-2, test split, final checkpoints: all-unit total `52.44%`, low/high `50.22%/2.22%`, sample p50/p90 `52.49%/53.26%`, layer totals h1 `40.09%`, h2 `81.45%`.
- Conv2 saturation, stride-1, test split, final checkpoints: all-unit total `77.28%`, low/high `76.93%/0.35%`, sample p50/p90 `77.30%/77.66%`, layer totals h1 `49.62%`, h2 `93.51%`.
- Interpretation: stride-1 Conv2 is both more accurate and much more saturated, especially in hidden layer 2. The extra saturation is almost entirely low-bound saturation, not high-bound saturation.
- Active MNIST stride-1 follow-ups: Conv1 final job `1048788` running with saturation `1048789` pending; Conv3 final job `1054868` running with saturation `1054869` pending.
- Fashion-MNIST stride grid: final job `1054871` running, saturation `1054872` pending. Completed training block so far is Conv1 stride-2 over seeds `0,1,2`, with mean best/final test accuracy `87.74%/87.73%`; saturation is not measured yet.
- Result root: `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_hopfield_eqprop_conv_saturation_20260628`.
- Local CSV copies: `/home/filip/server_code/results/mnist_hopfield_eqprop_conv_saturation_20260628`.
<!-- hopfield-eqprop-autoupdate:end -->

<!-- jeanzay-conv3-finish-report:start -->
## Jean Zay Conv3 Completion Report

Updated: 2026-06-28 22:55 CEST

Tracked jobs: `1004671`, `1004672`, `1004864`, `1004880`, `1004881`, `1004882`, `1004883`.

### Metrics Counts

| Result set | Metrics | Expected | Root |
|---|---:|---:|---|
| hard_sigmoid legacy low gain | 5 | 5 | `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv3_pad1_hardsigmoid_legacy_lowgain_voff4_k8_seed0_10epoch` |
| hard_sigmoid sat30 LR | 0 | 15 | `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv3_pad1_hardsigmoid_nonlegacy_sat30_lr_screen_voff4_k8_seed0_10epoch` |
| perfect_diode input gain | 15 | 15 | `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv3_pad1_perfect_diode_inputgain_screen_k8_seed0_10epoch` |
| perfect_diode LR | 0 | 15 | `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv3_pad1_perfect_diode_nonlegacy_lr_screen_k8_seed0_10epoch` |
| perfect_diode final 30 epoch | 0 | 3 | `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv3_pad1_perfect_diode_nonlegacy_bestlr_k8_seed0_30epoch` |

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

#### perfect_diode LR

- No completed metric rows found.

#### perfect_diode final 30 epoch

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
```

<!-- jeanzay-conv3-finish-report:end -->


## Shared Result Roots

- Jean Zay source checkout: `/lustre/fswork/projects/rech/umg/ucy17uy/server_code`
- Jean Zay R3 results base: `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results`
- Local Conv1/Conv2 saturation summary source: `/home/filip/server_code/results/good_conv_saturation_analysis/good_conv_saturation_best_per_amp_by_layer.csv`
- Local Conv3 saturation analysis source: `/home/filip/server_code/results/mnist_bp_conv3_pad1_hardsigmoid_amp_saturation_targets_voff4_sat10_20_30_50_k8_seed0_30epoch/saturation_init_final_train256.csv`

## Next Actions

1. Aggregate the completed `1004671` legacy low-gain metrics; decide whether the timed-out sat30 LR screen `1004672` should be rerun with longer walltime or smaller packing.
2. Track the active Conv3 perfect-diode per-amp chain `1053390 -> 1053391 -> 1053392`; after completion, compare the 30-epoch best-LR rows against the completed Conv3 hard-sigmoid best rows.
3. Measure initial/final per-layer saturation for the new Conv3 diagnostic rows.
4. Pick the best legacy low-gain row and run gradient diagnostics; compare `ConvWeight_0/1/2` and dense gradients against the old legacy sat10/sat30 rows.
5. Combine existing sat10 LR results with the new sat30 LR results for Conv3 `v1/c1`, `v2/c1`, and `v4/c1`.
6. Recheck and sync Conv2 50-epoch continuation status on Trex and Akib.
7. When the Hopfield EqProp Conv MNIST saturation jobs in `docs/current_experiments.md` finish, report their completed accuracies and per-layer saturation results here.
