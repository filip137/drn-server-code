# Crossbar recovery with the September 6 schedules

The latest [two-panel Matplotlib figure](analysis/digital_hwa_open_loop_20260908/digital_hwa_pv_open_loop.png)
shows digital HWA, healthy/corrupt P&V and open-loop on-chip training, with
crossbar above DRN. [Values and baseline definitions](analysis/digital_hwa_open_loop_20260908/report.md).

The updated [stochastic HWA-to-recovery figure](analysis/stochastic_hwa_predeployment_20260908/open_closed_recovery.png)
adds the KL of the frozen stochastic HWA training model before deployment.
[Values and evaluation details](analysis/stochastic_hwa_predeployment_20260908/report.md)
include the original HWA validation metrics and the common-test averages.

The comparison is complete: [results and figures](analysis/report.md),
[full test metrics](analysis/comparison.csv), and
[pairing/coverage verification](analysis/verification.json).

This exploratory follow-up compares open-loop single-pulse Adam and full
pulse-resolved P&V from the exact healthy and corrupt HWA + P&V states used
in the original three-epoch comparison. The user requested the newer
learning-rate and decay choices from `recovery-lr-schedule-20260906`.

The frozen choices are 3e-6 for healthy devices and 1e-5 for corrupt devices.
Both constant and exponential schedules receive 30 complete epochs. Decay
multiplies the starting rate by `0.01 ** min((epoch-1)/9, 1)` at each epoch's
start. The multiplier scales Adam's command without resetting its moments or
the P&V target accumulator. Each schedule is shared by both writers.

The verify tolerance remains 0.04745 in q=a-r, with 128 pulses per cell per
minibatch and 640 per cell during recovery. Hidden bounds and immutable native
OM effective-weight faults remain in the plant. Training and validation use
held apparent states; ordinary reads never redraw write noise.

All eight arms use the same saved development array/write, teacher, source
config, 55,000/5,000 split, batch size 16 and minibatch sequence. Validation
teacher KL selects checkpoints at full epochs, including P0. The official
10,000-example test cohort is read only after training and selection. Both
selected and final metrics, pulse exposure, and persistent diagnostics are
reported. These schedules were tuned for open-loop recovery; this experiment
does not claim a P&V-specific optimum or an architecture ranking.

The original study and the recent schedule study are preserved. The local
`crossbar_code` archive copies the recent study's schedule extension. The
matched driver and closed-loop controller are derived from the original
comparison, with explicit epoch scheduling and additional diagnostics.

`plan.json` declares the eight arms and scientific settings. `MONITOR.md`,
`launch.json`, `logs/*.launch.json`, `logs/*.exit.json`, and per-run status,
metrics and result files provide the operational record. All numerical runs
require real CUDA. Failed attempts remain separate from successful outputs.
