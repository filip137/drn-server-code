# Conv3 legacy/ours beta sweep, September 22–23

Filip revised the pending overnight sweep to exactly six cases:
injected beta **0.1, 0.3, 0.7** for **legacy and ours**, each with seed0,
ten full epochs and Gaussian endpoint read-noise sigma **0.0005**.
The former ten-beta legacy-only scheduler was stopped before any production
case started. Its preparation evidence remains in the superseded v1 directory.

| Injected beta | Legacy config beta (B/4096) | Ours config beta (B/64) |
|---:|---:|---:|
| 0.1 | 0.0000244140625 | 0.0015625 |
| 0.3 | 0.0000732421875 | 0.0046875 |
| 0.7 | 0.0001708984375 | 0.0109375 |

Each scheme inherits its own exact p99 Adam learning-rate vector, zero frozen
biases, shared saved initialization, float64 centered frozen-current EqProp,
accepted T=K=8, conductances [0,100], input gain360, train/validation batches
16/64, and ordinary MNIST deterministic55k/5k split. Independent Gaussian noise
affects non-input endpoint gradient readout only. Relaxation and validation
are noise-free. Official test remains disabled. Swept betas are exploratory;
do not claim p99 qualification. Retain the accepted finite-T caveat.

Keep each same-beta legacy/ours pair on one host and execute it serially,
preserving the common noise seed2026081601 and matched initialization/order.
Prioritize beta0.1, then0.3, then0.7; legacy then ours within each pair.
Do not claim identical realized noise between different hosts. Scheme-specific
inherited learning rates remain a comparison limitation.

Window: **September22 22:00–September23 08:00 Europe/Paris**.
No production starts before22:00; hard production cutoff07:55 leaves five
minutes for collection. The pool is local/Nom RTX3090 and Fifi/Trex/Loulou/Riri
RTX5090, selected only when free. Respect other users, existing serial queues
and active MPS clients. An idle MPS server alone is allowed only with low
memory/use. Riri's compatible environment was prepared during v1.

One training worker per GPU. Allow3h per case on3090,2h on5090: a whole
pair requires6h/4h remaining at first admission. Expected elapsed per pair is
roughly4–5.4h on3090 or3.2h on5090. Maximum production budget18GPUh, plus
0.5GPUh for smoke/timing. Never truncate an epoch or dataset to fit. Record
unstarted coverage if no free host can fit a whole pair, and retain all
scientific failures. Do not move an admitted pair between hosts.

Results: `results/eqprop-conv3-beta-sweep-5em4-10ep-20260922-v2/` locally,
and the same basename under `/home/filip/server_code/results/` remotely.
Configs: `configs/conv/eqprop_conv3_beta_sweep_5em4_10ep_20260922_v2/`.
Direct transport: `experiments/schedule_conv3_beta_overnight.py`.
Reuse scientific source archive
`74f41a03ce4718eebcd227bb38e4bbd2309e6b30bb9dd5515c099bb880068819`.
Run all six unchanged configs through local `experiments.exact_run --smoke`
before rearming. Cite the accepted T/K instead of recalibrating.

Scheduler heartbeat every30s; runtime polling includes GPU, process, logs,
metrics and semantic artifacts. Collect remote artifacts at least every30min
and after each run; validate completed local copies. Detached remote cases
carry their own hard timeout. Passive monitoring records errors and stale
progress; it is not a claim of intelligent automatic diagnosis/recovery.
Final reporting requires reconciling all six cases, including failures and
unstarted partners, then interpreting best/final validation accuracy and
trajectories as single-seed diagnostic evidence.

All six local optimizer-step/validation smokes and their canonical bundles
pass. The ours B0.7 GPU smoke on Loulou also passes and is locally collected
and validated. All six hosts have matching source/config/scheduler hashes;
four focused tests cover beta conversions, exact six-case coverage, pair
affinity, whole-pair admission, occupied GPUs and early/late starts.

Rearmed September22 at2026-09-22T15:30:43.110511+00:00 UTC: tmux
`beta-sweep-20260922`, PID813483. Its initial status confirms exactly six
pending cases and zero production starts. The old PID803081 is gone.
Exact commands and live progress are in `scheduler-command.json`,
`schedule_status.json` and `scheduler.log` under the result root.
