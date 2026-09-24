# Conv3 p90 read-noise extension: sigma 1e-3

Three new cases: baseline, legacy, ours; one seed (0), 30 epochs each. Reuse the completed RTX5090 clean controls and all prior noise results. No clean or existing-noise repeats.

Injected betas: baseline 404.141105702; legacy 4.42250110273; ours 5.26875648112. Unchanged accepted T=K=8, true float64 centered frozen-current EqProp, exact Adam LRs, common saved initialization, ordinary-MNIST 55k/5k train/validation, batch16/64, zero frozen biases and weights [0,100]. Gaussian endpoint read noise sigma=0.001 with seed2026081601, independent non-input layer/phase draws after relaxation; validation is clean. Official test disabled. Residual caveat at T8 retained.

Same frozen scientific source as the completed 24-case study (source archive 74f41a03ce4718eebcd227bb38e4bbd2309e6b30bb9dd5515c099bb880068819). Parent source: results/eqprop-conv3-p90-read-noise-20260919-v1/source-continuation-v1; remote equivalent: local-acceleration/frozen-source. Configs and reused clean bundle paths are in cases.json.

Fifi RTX5090 is currently the only idle 5090; riri/loulou/trex have unrelated jobs. One worker, sequential baseline/legacy/ours: previous measurements showed higher aggregate throughput than paired workers. Expected 12.5–14h, cap 6h/case and18 GPUh total; overall monitoring deadline 2026-09-21 15:00 UTC. Unstarted cases may move to a verified idle compatible5090 without changing science, with placement recorded before launch. Never interrupt unrelated jobs.

Run one-train/one-validation-batch smoke for each unchanged config through experiments.exact_run, then production via existing run_exact_parallel.py. Verify parent/worker and semantic artifacts immediately, check progress at least every30min (heartbeat/log/metrics/GPU/exit), collect and validate local bundles. Preserve scientific failures and poor accuracy. Retry only understood operational failures. Result/command/log paths: fifi/production/{case}.*, native bundles under fifi/production/runs/.

Primary comparison: final epoch30 validation accuracy against the reused same-GPU-class clean control; report best accuracy, full epoch trace, maximum running-best drawdown, finite status and endpoint screen (final strictly less than5pp below own best). No accuracy-based exclusion. Single-seed diagnostic, not paper-facing test evidence.
