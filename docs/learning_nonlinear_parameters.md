# Learning Nonlinear Parameters

Updated: 2026-06-29 11:39 CEST

## Scope

This note gathers the Conv2 hard-sigmoid MNIST results where nonlinear operating-point parameters are trainable.

- `A` means voltage amplification (`voltage_amp`).
- `B` means current amplification (`current_amp`).
- `v_off` means the symmetric hard-sigmoid half-width, so the layer uses `v_min=-v_off`, `v_max=v_off`.
- Unless otherwise stated, runs are seed `0`, BP, `K=6`, hard sigmoid with `g_on=100`, `g_off=0`, target sat30 input-gain calibration, batch size `4`.

## Current Read

Learning `A/B` is the strongest completed result so far. The Jean Zay 30-epoch trainable-amplification run from initial `A=1,B=1` reached `97.65% / 97.63%` and learned a much lower current amplification, `B=0.125846`.

Learning `v_off` together with `A/B` is promising but has only been run for 10 epochs so far: it reached `97.18% / 97.10%`, with layer0 `v_off` moving down from `4.0` to `3.5505`, layer1 staying at `4.0`, `A` moving to `2.1706`, and `B` moving down to `0.4882`.

Learning only `v_off` is still running for 50 epochs. Its latest completed checkpoint in the log is epoch `22/50`, with test `97.33%`; this already clears the fixed-boundary 10-epoch sat30 reference (`96.94%`) and the 10-epoch trainable `A/B+v_off` result, but it is still below the completed trainable-`A/B` 30-epoch result.

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
| `v_off` only | `mnist_bp_amp_v2_c1` | fixed `A=2,B=1`, `v_off=4`, sat30 `input_gain=181.9896698` | `0.0316501481` for weights/bias/`v_off` | active in tmux `conv2_voff_only_50ep`; no final `metrics.json` yet | epoch 22: `97.33%`, loss `0.0779` | `/home/filip/server_code/results/mnist_bp_conv2_trainable_voff4_only_voff_sat30params_seed0_50epoch` |

Epoch-end test trace for the active `v_off`-only 50-epoch run:

| Epoch | Test acc | Test loss |
|---:|---:|---:|
| 18 | `97.30%` | `0.0791` |
| 19 | `97.24%` | `0.0785` |
| 20 | `97.17%` | `0.0781` |
| 21 | `97.13%` | `0.0787` |
| 22 | `97.33%` | `0.0779` |

## Comparisons

| Reference | Best / final test | Comment |
|---|---:|---|
| Recent fixed-boundary sat30 Conv2 `mnist_bp_amp_v2_c1`, 10 epochs | `96.94% / pending local recopy` | Same sat30 family used as the first reference for `v_off` learning. |
| Trainable `A/B+v_off`, 10 epochs | `97.18% / 97.10%` | `+0.24 pp` vs the fixed-boundary 10-epoch sat30 reference. |
| Active `v_off`-only, epoch 22 interim | `97.33%` | `+0.39 pp` vs the fixed-boundary 10-epoch sat30 reference; final still pending. |
| Trainable `A/B` only, 30 epochs, initial `A=1,B=1` | `97.65% / 97.63%` | Best completed nonlinear-parameter result so far. |
| Fixed-amplification sat30 50-epoch continuations | best fixed row noted as `97.63%` | Current `A/B` learning is roughly tied with the best longer fixed-amplification result, not yet a decisive win. |

## Interpretation

- The strongest signal is that reducing effective current amplification is useful. Both successful trainable cases push `B` down: `B=0.125846` for `A/B`-only from `A=1,B=1`, and `B=0.488191` for the 10-epoch `A/B+v_off` run from `A=2,B=1`.
- The `A=4,B=1` trainable-A/B run did not discover the same low-`B` regime and underperformed. Initialization still matters.
- Learned `v_off` is layer-dependent. In the sat30 `A/B+v_off` run, layer0 moved down (`4.0 -> 3.5505`) while layer1 stayed at `4.0`. In the older non-sat30 `v_off=1.5` diagnostic, layer0 moved up strongly (`1.5 -> 4.3695`) while layer1 stayed near its initial value.
- The active `v_off`-only run is the cleanest test of whether threshold learning alone can explain the gain. Its interim result is already useful, but final learned thresholds are not available until `metrics.json` is written.

## Next Checks

1. Let `conv2_voff_only_50ep` finish and record final learned `v_off`, best/final accuracy, and final loss.
2. Measure per-layer saturation for the completed trainable `A/B` and active `v_off`-only rows.
3. If `v_off`-only finishes near or above `97.65%`, rerun with multiple seeds before treating it as better than trainable `A/B`.
4. If `v_off`-only stalls below trainable `A/B`, run a fair 30- or 50-epoch `A/B+v_off` job with the same sat30 setup and a controlled LR split.
