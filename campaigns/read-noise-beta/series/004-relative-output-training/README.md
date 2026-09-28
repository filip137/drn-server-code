# Relative output displacement and noisy training

This series follows Filip's selected direction: match centered output
displacement divided by free output RMS at initialization, using a separate
fixed beta for each architecture, scheme and target. It tests actual training
at additive endpoint read noise sigma5e-4, not the untested relative-read or
noisy-nudging models.

[exp-007](experiments/exp-007-initial-relative-output-training.md) covers
Conv1/2/3, three amplification schemes, four initial ratios and ten epochs.
[H-006](../../hypotheses/H-006-relative-output-ours-versus-legacy.md) is the
prospective ours-versus-legacy hypothesis. The old absolute-output-RMS-one
calibration and p90/p99 histories remain separate comparisons.

[exp008](experiments/exp-008-relative-output-noise-extension.md) uses seven targets for Conv1 and only2/2.5/3/6 for Conv2/3, at three absolute noise levels;117 new trainings reuse18 selected exp007 cells. All original results and cancellations remain separately preserved. Never pool different noise levels into one mean.
