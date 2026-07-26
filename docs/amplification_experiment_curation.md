# Amplification Experiment Curation

Last updated: 2026-07-26

This is the paper-facing cleanup layer for historical MNIST amplification
experiments. Raw result directories remain immutable; use the classifications
below instead of selecting runs directly from the filesystem.

The active deterministic-medium-affine Conv experiment is governed by
[`conv_paper_hyperparameter_protocol.md`](conv_paper_hyperparameter_protocol.md).
Ordinary-MNIST LR studies are curated separately in
[`conv_learning_rate_diagnostics.md`](conv_learning_rate_diagnostics.md).

## Main conclusions

1. Early underperforming DRN-XS rows were mainly preprocessing/setup artifacts;
   the legacy-preprocessing baseline recovers the expected accuracy.
2. Old hard-sigmoid current-amplification rows were invalidated by the dense
   amplification bug. Use only `dense_amp_fix` roots for current-amplification
   claims.
3. Perfect-diode clamped hidden layers require projected KKT residuals. Raw
   `|dE/dz|` remains appropriate only for unconstrained layers and diagnostics.
4. Corrected dense evidence is consistent: voltage amplification increases
   curvature and write-noise fragility; current amplification lowers curvature
   but can hurt clean trainability and accuracy.
5. Input quantization and center occlusion are negative controls for input
   corruption, not evidence about the write/conductance mechanism.
6. Historical Conv accuracy tables are preliminary ordinary-MNIST or
   mixed-protocol evidence. They are not quantitative evidence for the active
   medium-affine paper grid.
7. The active hard-sigmoid Conv1/Conv2 LR handoff is complete; medium-affine
   Conv3, all perfect-diode LR rules, and final training remain unresolved.

## Paper-facing dense runs

| Purpose | Use root | Status |
|---|---|---|
| Perfect-diode DRN-XS clean BP, paper preprocessing | `results/mnist_bp_amplification_sweep_drn_xs_10epoch_legacy_preproc` | valid |
| Hard-sigmoid baseline/voltage clean BP | `results/mnist_bp_amplification_sweep_hardsigmoid_voff15_legacy` | valid for `v1/c1`, `v2/c1`, `v4/c1`; current-amp rows invalid |
| Hard-sigmoid current amp after fix | `results/mnist_bp_amplification_sweep_hardsigmoid_current_amp_scratch_optuna_best_dense_amp_fix` | valid replacement |
| Perfect-diode current amp after fix | `results/mnist_bp_amplification_sweep_perfect_diode_current_amp_dense_amp_fix_10epoch` | valid replacement |
| Hard-sigmoid write noise after fix | `results/mnist_bp_write_noise_sweep_hardsigmoid_voff15_sigma_response_30x_dense_amp_fix` | valid |
| Hard-sigmoid physical Hessian after fix | `results/mnist_bp_hessian_hardsigmoid_voff15_legacy_drnxs_dense_amp_fix` | valid |
| Hard-sigmoid physical cost sharpness after fix | `results/mnist_bp_physical_cost_sharpness_hardsigmoid_voff15_iter16_dense_amp_fix` | valid |
| Hard-sigmoid finite-difference cost Hessian after fix | `results/mnist_bp_cost_hessian_noise_curvature_hardsigmoid_voff15_iter16_dense_amp_fix` | valid |

Dense clean-accuracy summary:

| Nonlinearity/setup | `v1/c1` | `v2/c1` | `v4/c1` | `v1/c2` | `v1/c4` |
|---|---:|---:|---:|---:|---:|
| perfect diode, legacy preprocessing | 0.9684 | 0.9686 | 0.9417 | 0.9677 | 0.9679 |
| hard sigmoid, legacy rows | 0.9576 | 0.9598 | 0.9482 | invalid | invalid |
| hard sigmoid, fixed current rows | - | - | - | 0.9520 | 0.9085 |
| perfect diode, fixed current rows | - | - | - | 0.9561 | 0.9136 |

## Dense write-noise and curvature evidence

For corrected hard-sigmoid write-noise runs:

| Amp | Clean accuracy | Sigma for 1% drop | RAUC |
|---|---:|---:|---:|
| `v1/c1` | 0.9576 | 0.247 | 0.797 |
| `v2/c1` | 0.9598 | 0.155 | 0.651 |
| `v4/c1` | 0.9482 | 0.090 | 0.494 |
| `v1/c2` | 0.9458 | 0.219 | 0.746 |
| `v1/c4` | 0.9086 | 0.122 | 0.646 |

Corrected hard-sigmoid curvature:

| Amp | Physical write sharpness | Software/log-weight directional curvature |
|---|---:|---:|
| `v1/c4` | 0.092 | 0.090 |
| `v1/c2` | 0.624 | 0.664 |
| `v1/c1` | 2.612 | 2.666 |
| `v2/c1` | 6.499 | 6.357 |
| `v4/c1` | 16.287 | 14.934 |

Voltage amplification is consistently sharper and more write-noise fragile.
Current amplification lowers physical curvature, but the `v1/c4` clean
accuracy and margins are too poor to call it a robustness win. Curvature
claims must name the perturbation coordinate; software-weight and physical
conductance/log-G noise are not interchangeable.

## Conv evidence

These older ordinary-MNIST tables remain useful only as preliminary
architecture evidence:

| Nonlinearity, Conv1 continued | `v1/c1` | `v2/c1` | `v4/c1` | `v1/c2` | `v1/c4` |
|---|---:|---:|---:|---:|---:|
| hard sigmoid | 0.9549 | 0.9671 | 0.9718 | 0.9337 | 0.9025 |
| perfect diode | 0.9733 | 0.9736 | 0.9737 | 0.9672 | 0.9378 |

| Nonlinearity, Conv2 K=6 50 epochs | `v1/c1` | `v2/c1` | `v4/c1` | `v1/c2` | `v1/c4` |
|---|---:|---:|---:|---:|---:|
| hard sigmoid | 0.9512 | 0.9706 | 0.9797 | 0.8611 | 0.8254 |
| perfect diode | 0.8830 | 0.9472 | 0.9807 | 0.7379 | 0.7079 |

Do not use these as final figures. They mix ordinary MNIST with older gains,
geometry, `T/K`, LR, epoch, and sometimes output conventions.

For the active medium-affine setup:

- all nine hard-sigmoid gains and row-specific `T/K` pairs are frozen;
- Conv1/Conv2 baseline LRs come from v1;
- Conv1/Conv2 amplified LRs come from v3;
- the authoritative six-row table is
  [`conv_hardsigmoid_lr_active_handoff_relative_rho_v3.csv`](conv_hardsigmoid_lr_active_handoff_relative_rho_v3.csv);
- LR-screen validation values are not paper-facing accuracy;
- medium-affine Conv3 and perfect-diode LR rules remain unresolved.

V4-v7, the scheme-specific two-rho sweeps, and the Conv2 SGD/Adam boundary
study are optimizer diagnostics only. Their disposition and important results
are kept in the diagnostic ledger rather than repeated here.

## Negative controls

| Control | Root | Useful conclusion |
|---|---|---|
| Input quantization | `results/mnist_bp_input_quantization_sweep_standard_best_bits2_3_4_8` | Mild down to low bit counts; does not reproduce write-noise amplification ordering. |
| Center occlusion | `results/mnist_bp_input_occlusion_sweep_standard_best_center_zero` | Dominated by input information loss, not a clean amplification effect. |
| Perfect-diode high-sigma software-weight noise | `results/mnist_bp_write_noise_sweep_standard_best_30x_highsigma_no_v4c1` | Demonstrates perturbation-coordinate mismatch; nearly amplification-independent for legacy models. |

## Exclude or supersede

Do not use these as paper-facing quantitative evidence:

| Root/pattern | Reason |
|---|---|
| `results/mnist_bp_amplification_sweep` | partial early run |
| `results/mnist_bp_amplification_sweep_bias_lower_lr` | exploratory rerun |
| `results/mnist_bp_amplification_sweep_drn_xs_10epoch` | superseded by legacy preprocessing |
| early `results/mnist_bp_amp_physical_conductance_lr_sweep` | wrong conductance/gain assumptions |
| current-amp rows in `results/mnist_bp_amplification_sweep_hardsigmoid_voff15_legacy` | dense amplification bug |
| `results/mnist_bp_hessian_hardsigmoid_voff15_legacy_drnxs` | superseded by `dense_amp_fix` |
| `results/mnist_bp_physical_cost_sharpness_hardsigmoid_voff15_iter16` | superseded by `dense_amp_fix` |
| `results/mnist_bp_cost_hessian_noise_curvature_hardsigmoid_voff15_iter16` | superseded by `dense_amp_fix` |
| `results/mnist_bp_residual_current_diagnostic_hardsigmoid_iter16` | updater conductance-scaling bug |
| `results/mnist_bp_conv2_hardsigmoid_residual_vs_iterations_seed0` | superseded by corrected residual diagnostic |
| pre-2026-07-18 Conv gain and `T/K` handoffs | superseded for deterministic medium affine MNIST |

## Standard setup going forward

- Conv paper rows use only baseline `v1/c1`, proposed/ours `v4/c1`, and legacy
  `v4/c0.25`.
- Gain is calibrated separately per architecture, nonlinearity, and scheme
  before operational `T/K`.
- Hard-sigmoid residuals use raw `max |dE/dz|`.
- Perfect-diode clamped hidden layers use projected KKT residuals.
- Robustness reports always name the perturbation coordinate.
- Final paper training waits for the unresolved active protocols even when an
  ordinary-MNIST diagnostic has completed.
