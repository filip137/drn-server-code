---
id: "exp-006"
title: "Plot clean Conv3 RMS displacement and BPTT cosine versus beta"
status: "complete"
hypotheses: ["H-005"]
---

# exp-006 — RMS displacement and cosine versus beta

## Assigned question and source audit

Filip approved implementation on September 24, 2026 after specifying clean
EqProp training, a beta sweep at fixed weights, and cosine against BPTT at
the same beta values. Additional T/K tests are explicitly excluded.

Use the clean sigma0, seed0, A100 production bundles in
`results/eqprop-conv3-p90-read-noise-20260919-v1/jz/a100-production/`:

| Scheme | Task / run | Injected training beta |
|---|---|---:|
| Baseline | `task_0/runs/000_conv3_baseline_p90_sigma0_seed0_da5e36ad` | 404.141105702 |
| Ours | `task_6/runs/000_conv3_ours_p90_sigma0_seed0_03483376` | 5.26875648112 |
| Legacy | `task_3/runs/000_conv3_legacy_p90_sigma0_seed0_f7b6c852` | 4.42250110273 |

The three local bundles passed `experiments.reporting.validate_run` during
planning. Use `final_model.pt` at epoch 30, never per-arm best epochs. All
three configs specify the shared initializer SHA256
`5e5782bd9bf166b8a432cbf283d745ffe4d25fe4a69ed656f14f9e7e9407392f`,
available at `results/eqprop-conv3-init-beta-tk-20260923-v1/inputs/initialization.pt`.
Record current source/config/checkpoint hashes in the replay config. This is
a bounded source check for this experiment, not completion of exp-003's broader audit.

## Frozen measurement contract

- Baseline `(A,B)=(1,1)`, ours `(4,1)`, legacy `(4,0.25)`; epochs 0 and 30.
- Nineteen injected betas `10**(-6 + 0.5*i)`, `i=0..18`, common to all six
  contexts. Base beta is injected beta divided by `(voltage_amp/current_amp)**3`.
- Existing 576-example ordinary-MNIST validation cohort, 36 batches of 16,
  from `configs/conv/eqprop_conv3_p90_final_noise_cosine_20260921.json`.
- Float64, input gain 360, source perfect-diode parameters and weights in
  `[0,100]`, exact-zero biases, clean endpoints, T=K=8, reset each batch.
- Positive, negative and zero-nudge phases share their post-T start and frozen
  current force. No optimizer steps, training, official-test reads or T/K sweeps.
- BPTT differentiates exactly K=8 iterations from that same post-T free state.

## Figures, tables and decision

Three 2-by-4 figures: rows initialization/epoch30; columns Conv1, Conv2,
Conv3, readout; three scheme curves. The first two use log-log axes for
RMS(v_positive-v_free) and RMS((v_positive-v_negative)/2). The third uses
log beta versus clean EqProp/BPTT cosine, with median and 10th–90th batch
percentiles and a fixed cosine range [-1,1]. Undefined cosine remains missing.

Pool displacement from squared RMS and element counts. Save negative-free
displacement, zero-nudge drift, state means/RMS, gradient norms and neighboring
log-log slopes. Mark centered signals at or below
`100 * float64_epsilon * max(positive_state_RMS, negative_state_RMS)` as
numerically unresolved; this is a resolution indicator, not a convergence test.
Preserve residual flags and failed cells; do not present them as zero signal
or silently fit across gaps. Interpret [H-005](../../../hypotheses/H-005-clean-beta-response.md)
per layer, stage and beta interval. The diagnostic does not establish an
accuracy mechanism or a qualified new training beta.

## Execution and bounded budget

Config: `configs/conv/conv3_clean_eqprop_rms_beta_20260924.json`.
Runner: `python -m experiments.replay_conv3_trained_beta_noise --config CONFIG --device cuda`;
same command with `--smoke` before production. Clean-only replay extends the
existing runner without changing its historical noisy defaults.

Local RTX3090 selected after live inventory; recheck memory at admission.
Use one target/runtime for the entire surface. Expected 114 production cells,
4,104 batch replays, 16,416 state rows and 16,416 clean gradient comparisons.
Cap four GPU-hours including smoke; refine the duration estimate from smoke.
Monitor actual progress at least every 30 minutes and preserve partial outputs
at the deadline. An individual numerical failure remains a named failed cell;
source/config mismatch stops execution.

Result root: `results/conv3-clean-eqprop-rms-beta-20260924-v1/` (indexed before
creation). Save canonical per-cell bundles, CSVs, PNG/PDF figures and report
there. Summarize coverage and conclusions in `../results/exp-006-beta-response.md`,
refresh the campaign ledger and curate the repository experimental manifest.

## Launch record

Started September 24 at 14:24:04 Europe/Paris on the local RTX3090, tmux
`clean-beta-20260924`, CUDA worker PID1924031 (wrapper PID1924028).
The six one-batch beta=1 smoke cases passed in 20.80 seconds; eight focused
runner tests passed. Production expects approximately 2.5–3 hours from the
measured throughput, with the four-GPU-hour budget ending at about18:23:47.
Monitor `production.log`, `progress.json` and per-cell status; exit and finish
records are `production.exitcode` and `production.finished` in the result root.

Config SHA256: `b6407d78885ee60db220439c29a8d6d41cf8aa57d4b42920b43234461537c251`.
Runner SHA256: `0abca42dc1407076c63415b1dd6ed6173772c6883eaebfac2f8979c824157e64`.
Cohort SHA256: `95af3eebd9416ec643b7d698d061b7dac3fe0d52311dd888293609bada629acd`.

## Completion

Completed September24 at17:01:27 Paris, exit0;114/114 cells and all16,416
state/gradient rows validate locally, with no failures, retries or exclusions.
Production9438.337seconds plus smoke20.802seconds =2.6275GPUh. Workers
exited and released the local GPU. Three planned figures plus an output D/F
figure are saved as PNG/PDF. See
[review](../results/exp-006-beta-response.md) for the scoped H-005 conclusion.
