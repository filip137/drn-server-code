> Superseded before production: Filip replaced this grid with beta0.1/0.3/0.7
> for both legacy and ours. The old scheduler was stopped. See the
> [active six-case plan](eqprop_conv3_beta_overnight_plan_20260922.md).

# Conv3 legacy beta sweep, September 22–23

User-requested exploratory training diagnostic: injected beta B in
`[0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0]`, seed 0,
10 full epochs per case, endpoint read-noise sigma `0.0005`.
The ten-point spacing is an agent assumption chosen to fit the overnight budget.
Config beta is `B / 4096`, not B itself.

Inherit the legacy p99 config's exact Adam weight rates, zero frozen biases,
saved initialization, float64 centered frozen-current EqProp, T=K=8,
weights [0,100], input gain 360, batch sizes 16/64, and deterministic ordinary
MNIST 55,000/5,000 split. Independent Gaussian noise is applied only to
non-input endpoint voltages for gradient readout; relaxation and validation
are noise-free. Official test access is disabled. No p99 qualification is
claimed for the swept betas. Retain the accepted finite-T caveat.

Window: **2026-09-22 22:00 through 2026-09-23 08:00 Europe/Paris**
(20:00–06:00 UTC). Production cannot start earlier. Target pool: local and
nom-cool-1 RTX3090s; fifi, trex, loulou and riri RTX5090s. Riri receives a new compatible
Python environment copied read-only from Fifi at the same absolute path.
Akib RTX3080 and Jean Zay were checked but are outside the requested GPU class.
Initially Fifi was idle, but another authorized CIFAR queue occupied it
before the GPU smoke. The guard refused that smoke and preserved the receipt.
All ten CPU scientific smokes passed. Recheck at admission and between runs;
never terminate or displace another job. An idle MPS server alone is allowed
only with low memory/use and no clients. Detect existing serial launchers to
respect their lane during gaps between workers.

One worker per GPU, serial cases. Prior measured Conv3 throughput suggests
roughly 2.1–2.7h on3090 and1.4–1.5h on5090 per ten-epoch case.
Reserve3h per3090 case and2h per5090 case, including margin; at most 30 production GPU-hours for ten cases, plus
0.5 GPU-hour for smoke/timing. The absolute hard stop applies to only this
study's process groups, with cleanup before 08:00 (production cutoff07:55, leaving five minutes for final collection).
A run must have enough
time for the full ten epochs before admission. Limited capacity is recorded
as unstarted coverage, never compensated by shortening an epoch or dataset.
Prioritize beta range coverage (0.1, 2, 0.5, 1, 0.2, 1.6, 0.3, 1.3, 0.4, 0.7).

Use the existing frozen scientific source from
`results/eqprop-conv3-p90-read-noise-20260919-v1/source-continuation-v1`
(archive SHA256 `74f41a03ce4718eebcd227bb38e4bbd2309e6b30bb9dd5515c099bb880068819`).
Use `experiments.exact_run` and its local one-train/one-validation-batch smoke
for all configs. Do not recalibrate rates or repeat the accepted T/K gate.
Transport source/config copies are isolated under this study's directory.

Results: `results/eqprop-conv3-legacy-beta-sweep-5em4-10ep-20260922-v1/`;
the same basename under `/home/filip/server_code/results/` on remote hosts.
Record configs, exact commands, source identities, scheduler PID/handle,
environment, logs, canonical bundles, and collection/validation status there.
Scheduled wait heartbeats and a first semantic artifact verify scheduling;
once training starts, inspect logs/metrics/GPU/processes at least every30min.
Passive monitoring may detect and record problems but does not imply autonomous
scientific diagnosis. Preserve failures; do not retry scientific nonfinite
outcomes. Operational retries require diagnosis and remaining budget.

Collect remote outputs and validate local bundles before interpretation.
Report final and best validation accuracy, epoch traces, finite status and
final drop from best; retain all failures and missing cases. These are
single-seed validation measurements. Host/software versions can change the
Gaussian draws despite identical seeds; record environment groups and do not
claim identical realized noise across GPU classes. No paper-facing accuracy.

Preparation validation: all10 local canonical smoke bundles pass; source
SHA256 and CPU-only imports pass on the staged hosts. New scheduler admission
tests reject an occupied GPU, an active MPS client, a wrong GPU class, an
early start, and a late ten-epoch admission. Current lack of free GPUs prevents
a new timing run; use the prior matched-source Conv3 measurements and a
conservative3h/2h per-case caps on3090/5090 respectively.
The5090 estimate is also supported by the55 completed/partial epochs in the
7.64GPUh sigma1e-3 study (roughly1.39h per10epochs). No production is run during preparation.

Armed September22 at17:16:35 Paris: tmux session `beta-legacy-20260922`,
scheduler PID803081. Its scheduled-state artifact and advancing30s heartbeat
were verified. All six source/environment checks pass; Riri environment
setup completed. `scheduler-command.json` records the exact command,
`schedule_status.json` the pending cases/placements/progress, and
`scheduler.log` operational errors. Remote case wrappers carry their own
deadlines and survive controller/SSH disconnects. Automatic observation and
collection are active; intelligent diagnosis is not claimed for the passive
monitor. Full scientific coverage remains conditional on available lanes.

Loulou became idle at17:17Paris. Its exact GPU smoke passed and was collected
and validated locally. A separately labeled100-training/5-validation-batch
timing run then completed in16.82s, including startup/checkpoint overhead.
Scaling that entire time by34380/100 projects1.61h per ten-epoch case,
supporting the2h5090 allowance. Both diagnostic bundles validate locally.
They do not alter or count toward the ten production cases.
