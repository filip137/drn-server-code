# Amplification Experiment Curation

Last updated: 2026-06-16

This note is the paper-facing cleanup layer for the MNIST amplification experiments. Raw result directories are preserved, but analysis should use the curated status below.

## Main Conclusions

1. Early underperforming DRN-XS runs were mostly preprocessing/setup artifacts. The legacy DRN-XS preprocessing run recovers the expected baseline accuracy.
2. The old current-amplification results for hard-sigmoid/current-amp runs were invalidated by the amplification implementation bug. Use only roots with `dense_amp_fix` for current-amplification conclusions.
3. Perfect-diode residuals must be interpreted with projected KKT residuals, not raw `|dE/dz|` on clamped variables.
4. The useful dense result is consistent: voltage amplification increases physical/cost sharpness and write-noise fragility; current amplification lowers curvature but can hurt clean trainability/accuracy, especially `v1/c4`.
5. Input quantization and center occlusion are negative controls. They corrupt the input channel, not the write/conductance channel, and did not expose the same amplification mechanism.
6. Conv1 results are usable as preliminary architecture evidence. Conv2 results are not final for paper figures until rerun with the corrected iteration counts: `hard_sigmoid K=32`, `perfect_diode K=8`.

## Paper-Facing Dense Runs

Use these roots for dense DRN-XS claims:

| Purpose | Use root | Status |
|---|---|---|
| Perfect-diode DRN-XS clean BP, paper preprocessing | `results/mnist_bp_amplification_sweep_drn_xs_10epoch_legacy_preproc` | valid |
| Hard-sigmoid baseline/voltage clean BP | `results/mnist_bp_amplification_sweep_hardsigmoid_voff15_legacy` | valid for `v1/c1`, `v2/c1`, `v4/c1`; supersede current-amp rows |
| Hard-sigmoid current-amp clean BP after dense amp fix | `results/mnist_bp_amplification_sweep_hardsigmoid_current_amp_scratch_optuna_best_dense_amp_fix` | valid current-amp replacement |
| Perfect-diode current-amp clean BP after dense amp fix | `results/mnist_bp_amplification_sweep_perfect_diode_current_amp_dense_amp_fix_10epoch` | valid current-amp replacement |
| Hard-sigmoid write noise after dense amp fix | `results/mnist_bp_write_noise_sweep_hardsigmoid_voff15_sigma_response_30x_dense_amp_fix` | valid |
| Hard-sigmoid physical Hessian after dense amp fix | `results/mnist_bp_hessian_hardsigmoid_voff15_legacy_drnxs_dense_amp_fix` | valid |
| Hard-sigmoid physical cost sharpness after dense amp fix | `results/mnist_bp_physical_cost_sharpness_hardsigmoid_voff15_iter16_dense_amp_fix` | valid |
| Hard-sigmoid finite-difference cost Hessian after dense amp fix | `results/mnist_bp_cost_hessian_noise_curvature_hardsigmoid_voff15_iter16_dense_amp_fix` | valid |

Dense clean accuracy summary:

| nonlinearity/setup | v1/c1 | v2/c1 | v4/c1 | v1/c2 | v1/c4 |
|---|---:|---:|---:|---:|---:|
| perfect diode, legacy preprocessing | 0.9684 | 0.9686 | 0.9417 | 0.9677 | 0.9679 |
| hard sigmoid, legacy rows | 0.9576 | 0.9598 | 0.9482 | invalid | invalid |
| hard sigmoid, fixed current rows | - | - | - | 0.9520 | 0.9085 |
| perfect diode, fixed current rows | - | - | - | 0.9561 | 0.9136 |

## Dense Write-Noise Interpretation

For corrected hard-sigmoid write-noise runs, the 1 percent accuracy-drop sigma is:

| amp | clean acc | sigma for 1 percent drop | RAUC |
|---|---:|---:|---:|
| v1/c1 | 0.9576 | 0.247 | 0.797 |
| v2/c1 | 0.9598 | 0.155 | 0.651 |
| v4/c1 | 0.9482 | 0.090 | 0.494 |
| v1/c2 | 0.9458 | 0.219 | 0.746 |
| v1/c4 | 0.9086 | 0.122 | 0.646 |

Interpretation:

- Voltage amplification is consistently more fragile to write noise.
- Current amplification reduces physical/cost curvature, but its apparent robustness is limited by clean accuracy and margins. `v1/c2` is close to baseline robustness; `v1/c4` is not a clean robustness win because its starting accuracy is low.
- Perfect-diode current-amp fixed runs did not cross a 1 percent accuracy drop in the tested small-sigma grid, but those results cover only `v1/c2` and `v1/c4` and have lower clean accuracy for `v1/c4`.

## Hessian And Curvature Interpretation

Corrected hard-sigmoid physical write sharpness:

| amp | `S_write` mean |
|---|---:|
| v1/c4 | 0.092 |
| v1/c2 | 0.624 |
| v1/c1 | 2.612 |
| v2/c1 | 6.499 |
| v4/c1 | 16.287 |

Corrected finite-difference cost curvature in software/log-weight directions:

| amp | all-weight directional curvature |
|---|---:|
| v1/c4 | 0.090 |
| v1/c2 | 0.664 |
| v1/c1 | 2.666 |
| v2/c1 | 6.357 |
| v4/c1 | 14.934 |

Physical Hessian spectral-radius trends agree with the same qualitative story:

| amp | hard-sigmoid `rho` | LPW/perfect-diode `rho` |
|---|---:|---:|
| v1/c4 | 0.0060 | 0.0041 |
| v1/c2 | 0.0081 | 0.0081 |
| v1/c1 | 0.0082 | 0.0161 |
| v2/c1 | 0.0167 | 0.0427 |
| v4/c1 | 0.0491 | 0.1504 |

Paper interpretation:

- Voltage amplification makes the equilibrium/cost landscape sharper.
- Current amplification makes the physical Hessian smoother, but this is not sufficient by itself; training quality and output margins matter.
- Curvature predicts noise sensitivity best when the perturbation coordinate matches the noise model. Software-weight noise and physical-conductance/log-G noise should not be mixed in one claim.

## Conv Runs

Conv1 is a useful preliminary result:

| nonlinearity, conv1 continued | v1/c1 | v2/c1 | v4/c1 | v1/c2 | v1/c4 |
|---|---:|---:|---:|---:|---:|
| hard sigmoid | 0.9549 | 0.9671 | 0.9718 | 0.9337 | 0.9025 |
| perfect diode | 0.9733 | 0.9736 | 0.9737 | 0.9672 | 0.9378 |

Conv2 current status:

| nonlinearity, conv2 K=6 50 epoch | v1/c1 | v2/c1 | v4/c1 | v1/c2 | v1/c4 |
|---|---:|---:|---:|---:|---:|
| hard sigmoid | 0.9512 | 0.9706 | 0.9797 | 0.8611 | 0.8254 |
| perfect diode | 0.8830 | 0.9472 | 0.9807 | 0.7379 | 0.7079 |

Do not use conv2 K=6 runs as final paper figures. The residual diagnostic shows:

| nonlinearity | recommended uniform K |
|---|---:|
| hard sigmoid | 32 |
| perfect diode | 8 |

Useful conv2 diagnostic: the cost-Hessian ordering already mirrors the dense story. Voltage amplification increases curvature; current amplification lowers curvature but can undertrain or collapse, so the conv2 paper run must be rerun with correct K and tuned LR before drawing accuracy conclusions.

## Negative Controls

Keep as negative-control evidence, not as the main amplification mechanism:

| Control | Root | Useful conclusion |
|---|---|---|
| Input quantization | `results/mnist_bp_input_quantization_sweep_standard_best_bits2_3_4_8` | Input quantization is mild down to very low bit counts and does not reproduce write-noise amplification ordering. |
| Center occlusion | `results/mnist_bp_input_occlusion_sweep_standard_best_center_zero` | Accuracy drop is dominated by occlusion size and input information loss, not by a clean amplification effect. |
| Software-weight perfect-diode high-sigma noise | `results/mnist_bp_write_noise_sweep_standard_best_30x_highsigma_no_v4c1` | Useful for showing coordinate mismatch: software-weight noise was nearly amplification-independent for legacy perfect-diode models. |

## Exclude Or Supersede

Do not use the following as paper-facing quantitative evidence:

| Root/pattern | Reason |
|---|---|
| `results/mnist_bp_amplification_sweep` | partial/early run; not the final DRN-XS paper setup |
| `results/mnist_bp_amplification_sweep_bias_lower_lr` | exploratory rerun, not final setup |
| `results/mnist_bp_amplification_sweep_drn_xs_10epoch` | superseded by legacy preprocessing run |
| early physical-conductance LR sweeps under `results/mnist_bp_amp_physical_conductance_lr_sweep` | wrong conductance/gain assumptions for the final clean BP comparison |
| `results/mnist_bp_amplification_sweep_hardsigmoid_voff15_legacy` current-amp rows | invalidated by dense current-amplification bug |
| `results/mnist_bp_hessian_hardsigmoid_voff15_legacy_drnxs` | superseded by `..._dense_amp_fix` |
| `results/mnist_bp_physical_cost_sharpness_hardsigmoid_voff15_iter16` | superseded by `..._dense_amp_fix` |
| `results/mnist_bp_cost_hessian_noise_curvature_hardsigmoid_voff15_iter16` | superseded by `..._dense_amp_fix` |
| `results/mnist_bp_residual_current_diagnostic_hardsigmoid_iter16` | invalidated by hard-sigmoid updater conductance scaling bug |
| `results/mnist_bp_conv2_hardsigmoid_residual_vs_iterations_seed0` | superseded by corrected KKT/raw residual diagnostic |
| conv2 K=6 training roots | preliminary only; hard sigmoid underconverged and perfect diode should use K=8 |

## Standard Paper Setup Going Forward

1. Dense DRN-XS perfect diode: use `legacy_preproc` for baseline/voltage/current comparison, and fixed-current reruns for current-amplification bug checks.
2. Dense hard sigmoid: combine legacy baseline/voltage rows with fixed-current rows only when explicitly labeled; otherwise rerun all five amps in one clean fixed setup.
3. Conv1: can be discussed as preliminary, but final figures should use a single seed protocol only if labeled as such.
4. Conv2: rerun all five amps with `hard_sigmoid K=32`, `perfect_diode K=8`, tuned LR, and consistent epoch budget before paper figures.
5. Whenever reporting residuals:
   - hard sigmoid: raw `max |dE/dz|`.
   - perfect diode: projected KKT residual for clamped diode layers plus raw residual for unconstrained output layers.
6. Whenever reporting noise robustness, state the perturbation coordinate: software `Weight`, physical conductance/log-G, input quantization, or input occlusion.
