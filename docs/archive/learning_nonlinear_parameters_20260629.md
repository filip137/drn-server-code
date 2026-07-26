# Learning Nonlinear Parameters

> Historical mixed-protocol diagnostic snapshot. This note predates the
> deterministic-medium-affine Conv paper protocol and its frozen v1/v3
> learning-rate handoff. It is preserved for provenance only and is not an
> active protocol or paper-facing comparison.

Updated: 2026-06-29 14:08 CEST

## Scope

This note gathers the Conv2 hard-sigmoid MNIST results where nonlinear operating-point parameters are trainable.

- `A` means voltage amplification (`voltage_amp`).
- `B` means current amplification (`current_amp`).
- `v_off` means the symmetric hard-sigmoid half-width, so the layer uses `v_min=-v_off`, `v_max=v_off`.
- Unless otherwise stated, runs are seed `0`, BP, `K=6`, hard sigmoid with `g_on=100`, `g_off=0`, target sat30 input-gain calibration, batch size `4`.

## Motivation

The problem motivating these runs is that deeper Conv DRN hard-sigmoid networks are not improving as much as expected when later convolutional layers are added. Adding depth should help if later layers are contributing useful nonlinear feature transformations, but the hard-sigmoid Conv2/Conv3 results have not shown a clear accuracy jump from those later layers.

Two comparisons make this suspicious:

- `perfect_diode` DRNs perform much better on the same deeper Conv settings.
- Hopfield-energy networks trained with EqProp also look much better.

That suggests the architecture can benefit from depth, but the hard-sigmoid DRN may be operating at a poor nonlinear point. The saturation diagnostics are the main warning sign: in later hard-sigmoid layers, saturation is often either very small or exactly zero, or in some configurations very large. Both extremes are problematic. If later layers are mostly unsaturated, they may behave nearly linearly and add little useful depth. If they are heavily saturated, gradients and state updates can become poorly conditioned or uninformative.

`A`, `B`, and `v_off` directly change this operating point:

- Increasing/decreasing `A` changes the voltage scale seen by downstream interactions.
- Changing `B` changes the effective current/conductance scale.
- Changing `v_off` changes the voltage interval over which hard sigmoid is in its linear transition region.

So the trainable-parameter runs ask whether hard-sigmoid Conv DRNs are underperforming because we picked fixed nonlinear parameters that place later layers in the wrong saturation regime.

## Hypotheses

Primary hypothesis: deeper hard-sigmoid Conv DRNs are limited by a bad saturation operating point in later layers, not only by architecture or optimizer capacity. If this is true, making `A`, `B`, and/or layer-wise `v_off` trainable should improve accuracy and should learn values that move the network toward a more useful saturation regime.

Secondary hypotheses:

- Learning `A/B` should reveal the useful global amplification scale. The learned values are a proxy for the "optimal" amplification under the current architecture, target-saturation calibration, LR schedule, and seed.
- Learning layer-wise `v_off` should reveal whether each nonlinear layer wants a different hard-sigmoid boundary. If later layers are under-saturated, they may need smaller `v_off` or different amplification; if they are over-saturated, they may need larger `v_off` or different amplification.
- If learned nonlinear parameters only give small gains, then fixed-parameter hard-sigmoid performance is probably not failing solely because of a badly chosen `A/B/v_off`. The remaining gap to `perfect_diode` DRNs and Hopfield EqProp would then point to deeper issues in the hard-sigmoid dynamics, energy landscape, or training signal.
- If learned nonlinear parameters close much of the gap, then future fixed-parameter sweeps should be centered around the learned `A/B/v_off` values rather than around the hand-chosen amplification grid.

In this note, "optimal" does not mean a universal device-level optimum. It means the learned values that maximize validation/test accuracy for this specific Conv2 hard-sigmoid training setup while keeping the nonlinear layers in a useful saturation regime.

## Experimental Evidence Trail

The fixed-parameter baseline that motivated the trainable-parameter runs used a three-stage protocol:

1. Calibrate raw `input_gain` from target saturation. The corrected Conv2 calibration file `/home/filip/server_code/results/mnist_bp_conv2_hardsigmoid_amp_calibrated_sat30_lr_sweep_voff4_seed0_10epoch/targets/conv2_amp_calibrated_targets.csv` records per-amplification sat30 gains, e.g. `A=1,B=1 -> 221.068374634`, `A=2,B=1 -> 181.989669800`, `A=4,B=1 -> 158.035232544`, `A=1,B=2 -> 294.541473389`, `A=1,B=4 -> 485.916717529`, `A=4,B=0.25 -> 140.196121216`, and `A=2,B=2 -> 221.050064087`.
2. Sweep LR multipliers for those calibrated gains. The complete LR-screen summary in `/home/filip/server_code/docs/current_state.md` says the selected sat30 10-epoch rows were: `A=1,B=1`, LR mult `4`, best `96.01%`; `A=2,B=1`, LR mult `4`, best `96.94%`; `A=4,B=1`, LR mult `0.5`, best `96.59%`; legacy `A=4,B=0.25`, LR mult `0.25`, best `86.50%`. The local copied low-LR half is under `/home/filip/server_code/results/mnist_bp_conv2_hardsigmoid_amp_calibrated_sat30_lr_sweep_voff4_seed0_10epoch_akib`.
3. Continue the selected LR rows for 50 epochs. The fixed-amplification sat30 continuation summary in `/home/filip/server_code/docs/current_state.md` reports best/final: `A=4,B=1` `97.63% / 97.59%`, `A=2,B=1` `97.58% / 97.58%`, `A=1,B=1` `96.72% / 96.72%`, `A=2,B=2` `96.58% / 96.58%`, `A=1,B=2` `94.07% / 94.00%`, legacy `A=4,B=0.25` `91.90% / 91.83%`, and `A=1,B=4` `84.88% / 84.51%`. The local manifest for that follow-up is `/home/filip/server_code/results/mnist_bp_conv2_hardsigmoid_amp_calibrated_sat30_bestlr_voff4_seed0_50epoch/targets/`.

These baselines underline the operating-point concern:

- The best fixed sat30 continuation is already strong but not decisive: `A=4,B=1` reaches `97.63%` best test, while the trainable-`A/B` row reaches `97.65%`. That makes trainable `A/B` a small gain over the best fixed longer run, not a gap-closing result by itself.
- The successful fixed sat30 rows still leave hidden layer 2 almost unsaturated. Final saturation in `/home/filip/server_code/docs/current_state.md`: `A=4,B=1` has L1/L2 `19.36% / 0.00%`, `A=2,B=1` has `34.20% / 0.00%`, and `A=1,B=1` has `43.17% / 0.00%`.
- The bad fixed rows show the other failure mode. Legacy `A=4,B=0.25` ends at L1/L2 `98.95% / 27.49%`, and `A=1,B=4` ends at `95.20% / 1.03%`.
- The gradient diagnostics in `/home/filip/server_code/results/good_conv_saturation_analysis/conv2_voff4_existing_targets_gradient_summary.md` match this: current-heavy or legacy overdriven cases can collapse upstream gradients, e.g. `A=1,B=2`, sat70 has final `W0=W1=0`, and legacy `A=4,B=0.25`, sat70 also has final `W0=W1=0`.

The deeper-architecture comparisons support the idea that hard-sigmoid operating point, not just model capacity, is implicated:

- Conv2 `perfect_diode` with the same broad Conv2 family is higher than hard sigmoid: `/home/filip/server_code/results/mnist_bp_conv2_epbpK_large_50epoch_64_128ch_s2_valid_seed0_local_distributed/perfect_diode/mnist_bp_amp_v4_c1/seed_0/metrics.json` gives `98.66% / 98.64%`, and `A=2,B=1` gives `98.35% / 98.35%`.
- Hopfield EqProp Conv MNIST also reaches the high-accuracy regime. `/home/filip/server_code/docs/hopfield_experiments.md` reports 30-epoch mean best/final over seeds `0,1,2`: Conv1 stride2 `97.29% / 97.27%`, Conv2 stride2 `97.61% / 97.50%`, Conv3 stride2 `97.01% / 97.01%`, and Conv2 stride1 `98.88% / 98.32%`.
- Conv3 hard-sigmoid remains weak relative to those references. `/home/filip/server_code/docs/current_state.md` lists the best current Conv3 hard-sigmoid row as `96.71% / 96.64%`, with initial saturation hidden1/hidden2/hidden3 `9.87% / 0.00% / 0.00%`.

## Current Read

Learning `A/B` is the strongest completed result so far. The Jean Zay 30-epoch trainable-amplification run from initial `A=1,B=1` reached `97.65% / 97.63%` and learned a much lower current amplification, `B=0.125846`.

Learning `v_off` together with `A/B` is promising but has only been run for 10 epochs so far: it reached `97.18% / 97.10%`, with layer0 `v_off` moving down from `4.0` to `3.5505`, layer1 staying at `4.0`, `A` moving to `2.1706`, and `B` moving down to `0.4882`.

Learning only `v_off` has no final `metrics.json` yet. The latest completed checkpoint in the local log is epoch `25/50`, with test `97.27%`; the latest line is epoch `26/50` batch `1000/15000`. Best interim test so far is still epoch `22` at `97.33%`. This already clears the fixed-boundary 10-epoch sat30 reference (`96.94%`), but it is still below the completed trainable-`A/B` 30-epoch result.

## Completed Results

| Learned params | Run | Initial values | Trainable LR | Epochs | Best / final test | Best epoch | Final learned values | Source |
|---|---|---|---:|---:|---:|---:|---|---|
| `A/B` only | `mnist_bp_amp_v1_c1` | `A=1,B=1`, fixed `v_off=4` | `0.00651382180914` | 30 | `97.65% / 97.63%` | 26 | `A=1.647239,B=0.125846` | `/lustre/fsn1/projects/rech/umg/ucy17uy/server_code/results/mnist_bp_conv2_hardsigmoid_trainable_amp_sat30_voff4_seed0_30epoch/target_sat30/input_gain_221p068374634/lr_0p00651382180914/hard_sigmoid/mnist_bp_amp_v1_c1/seed_0/metrics.json` |
| `A/B` only | `mnist_bp_amp_v4_c1` | `A=4,B=1`, fixed `v_off=4` | `0.00911189218265` | 30 | `96.34% / 96.34%` | 30 | `A=4.257293,B=1.071110` | `/lustre/fsn1/projects/rech/umg/ucy17uy/server_code/results/mnist_bp_conv2_hardsigmoid_trainable_amp_sat30_voff4_seed0_30epoch/target_sat30/input_gain_158p035232544/lr_0p00911189218265/hard_sigmoid/mnist_bp_amp_v4_c1/seed_0/metrics.json` |
| `A/B + v_off` | `mnist_bp_amp_v2_c1` | `A=2,B=1`, `v_off=4` | weights/bias/`v_off`: `0.0316501481`; `A/B`: `0.1266005924` | 10 | `97.18% / 97.10%` | 9 | `A=2.170565,B=0.488191`; `v_off=[3.550537,4.000000]` | `/home/filip/server_code/results/mnist_bp_conv2_trainable_voff4_amp_lr4x_sat30params_seed0_10epoch/hard_sigmoid/mnist_bp_amp_v2_c1/seed_0/metrics.json` |
| `v_off` only | `mnist_bp_amp_v1_c1` diagnostic | `A=1,B=1`, `v_off=[1.5,1.5]`, non-sat30 `input_gain=100`, `K=4` | `0.006` initial LR | 10 | `93.56% / 93.56%` | 10 | `v_off=[4.369478,1.533968]` | `/home/filip/server_code/results/mnist_bp_conv2_trainable_voff_bp_10epoch_seed0/hard_sigmoid/mnist_bp_amp_v1_c1/seed_0/metrics.json` |

## Active Run

| Learned params | Run | Initial values | LR | Progress | Latest completed test | Source |
|---|---|---|---:|---|---:|---|
| `v_off` only | `mnist_bp_amp_v2_c1` | fixed `A=2,B=1`, `v_off=4`, sat30 `input_gain=181.9896698` | `0.0316501481` for weights/bias/`v_off` | no final `metrics.json` yet; latest log line epoch `26/50` batch `1000/15000` | epoch 25: `97.27%`, loss `0.0764`; best interim epoch 22: `97.33%` | `/home/filip/server_code/results/mnist_bp_conv2_trainable_voff4_only_voff_sat30params_seed0_50epoch` |

Epoch-end test trace for the active `v_off`-only 50-epoch run:

| Epoch | Test acc | Test loss |
|---:|---:|---:|
| 18 | `97.30%` | `0.0791` |
| 19 | `97.24%` | `0.0785` |
| 20 | `97.17%` | `0.0781` |
| 21 | `97.13%` | `0.0787` |
| 22 | `97.33%` | `0.0779` |
| 23 | `97.26%` | `0.0764` |
| 24 | `97.21%` | `0.0773` |
| 25 | `97.27%` | `0.0764` |

## Comparisons

| Reference | Best / final test | Comment |
|---|---:|---|
| Recent fixed-boundary sat30 Conv2 `mnist_bp_amp_v2_c1`, 10 epochs | `96.94% / pending local recopy` | Same sat30 family used as the first reference for `v_off` learning. |
| Trainable `A/B+v_off`, 10 epochs | `97.18% / 97.10%` | `+0.24 pp` vs the fixed-boundary 10-epoch sat30 reference. |
| Active `v_off`-only, epoch 22 interim best | `97.33%` | `+0.39 pp` vs the fixed-boundary 10-epoch sat30 reference; final still pending. |
| Trainable `A/B` only, 30 epochs, initial `A=1,B=1` | `97.65% / 97.63%` | Best completed nonlinear-parameter result so far. |
| Fixed-amplification sat30 50-epoch continuations | best fixed row noted as `97.63%` | Current `A/B` learning is roughly tied with the best longer fixed-amplification result, not yet a decisive win. |

## Interpretation

- The strongest signal is that reducing effective current amplification is useful. Both successful trainable cases push `B` down: `B=0.125846` for `A/B`-only from `A=1,B=1`, and `B=0.488191` for the 10-epoch `A/B+v_off` run from `A=2,B=1`.
- The `A=4,B=1` trainable-A/B run did not discover the same low-`B` regime and underperformed. Initialization still matters.
- Learned `v_off` is layer-dependent. In the sat30 `A/B+v_off` run, layer0 moved down (`4.0 -> 3.5505`) while layer1 stayed at `4.0`. In the older non-sat30 `v_off=1.5` diagnostic, layer0 moved up strongly (`1.5 -> 4.3695`) while layer1 stayed near its initial value.
- The active `v_off`-only run is the cleanest test of whether threshold learning alone can explain the gain. Its interim result is useful, but final learned thresholds are not available until `metrics.json` is written.

## Next Checks

1. Let `conv2_voff_only_50ep` finish and record final learned `v_off`, best/final accuracy, and final loss.
2. Measure per-layer saturation for the completed trainable `A/B` and active `v_off`-only rows.
3. If `v_off`-only finishes near or above `97.65%`, rerun with multiple seeds before treating it as better than trainable `A/B`.
4. If `v_off`-only stalls below trainable `A/B`, run a fair 30- or 50-epoch `A/B+v_off` job with the same sat30 setup and a controlled LR split.
