# T8 versus T16 at initial-output-displacement-one betas

User-requested September 23 follow-up. Run the four cases from
`eqprop-conv3-rms1-gradient-replay-20260923-v1` with only inference T changed
from 16 to 8. Keep K8, injected beta ours1.385 and legacy .02173, saved
initialization and p99 epoch30 final checkpoints, exact36 validation batches
of16, float64 centered frozen-current EqProp, all model/diode/input/loss
settings and four sigma5e-4 endpoint-noise seeds unchanged. Include clean
reads. This is exploratory read-only validation replay; no optimizer steps,
accuracy evaluation, or official-test reads.

Reuse the validated T16 measurements rather than repeat them. Pair every
case/batch/layer/noise draw by identity. Check source/checkpoint/cohort and
runtime identities, per-layer cosine, norm and error, noise/clean ratio,
phase displacement and projected-KKT residuals. Report changes in medians
and paired batches (averaging noise draws within batch); do not describe
differences of medians as paired effects. Inspect changes in the literal
free-to-nudged displacement separately from the centered phase contrast.

Use the same Akib RTX3080/PyTorch2.5.1/CUDA12.1 production environment as the
T16 comparison. Recheck the authorized host inventory and Akib occupancy.
Use an isolated workspace, byte-identical scientific runner, local same-config
smoke, then Akib GPU smoke before production. Record exact commands and logs.
Expected production5–10min, hard combined smoke/replay budget1,800s; monitor
first artifacts and per-batch progress through the deadline. Collect locally
and validate all four production/eight smoke bundles before interpretation.
Expected2,880 new layer comparisons, paired with2,880 existing T16 rows.

Config: `configs/conv/eqprop_conv3_rms1_t8_replay_20260923.json`.
Runner: `python -m experiments.replay_conv3_trained_beta_noise --config CONFIG
--device cuda`; smoke appends `--smoke`. Remote data root override is
`/home/filiposana/datasets/mnist` and changes no cohort bytes.
Result root: `results/eqprop-conv3-rms1-t8-replay-20260923-v1`.
Remote root: `/home/filiposana/server_code/results/eqprop-conv3-rms1-t8-replay-20260923-v1`.
Complete with paired CSV, matplotlib figure, report and manual manifest entry.

Launch: four local CPU smokes and residual gates pass in278.11s. The
scientific runner is byte-identical to T16; all196 staged hashes verify.
Akib was rechecked idle and launched with nohup supervisor511334, GPU
worker511366. All four GPU smokes pass in10.02s. The wrapper allows1,521s
for GPU smoke plus production after charging the CPU smoke. Other checked
GPUs were occupied; the Jean Zay read-only queue probe timed out and no work
was submitted there. All production remains on the same Akib environment.

Completed: four production cases, 2,880 comparisons and eight smoke bundles
validate locally, with119 matching remote file hashes. All2,304 new residual
checks pass (maximum7.92642e-6). All2,880 T8 rows pair with validated T16
rows on checkpoint/cohort/draw identity; runner and runtime are identical.
No failed or excluded cases. Akib exited zero and released the GPU after
337.22s production; combined charge625.35s within1,800s.

The largest noisy layer-median cosine change is1.50245e-7, clean change
8.72897e-8, and maximum individual cosine change1.41724e-6. Gradient quality
and the scheme comparison are unchanged. T8 adds continued-relaxation motion
to legacy's initial first-layer literal displacement, but not to its centered
phase signal. See the
[report](../results/eqprop-conv3-rms1-t8-replay-20260923-v1/analysis/report.md)
and [manifest](experimental_manifest.md#conv3-t8-check-at-output-displacement-one-betas-september-23).
