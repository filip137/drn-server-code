# Conv3 trained-checkpoint beta and read-noise replay

September 22, 2026. Exploratory ordinary-MNIST mechanism diagnostic requested
by Filip. The question is whether any single beta provides useful noisy
EP–BPTT gradient alignment across all four weight layers at already trained
checkpoints. Clean replay is a control, not the selection objective.

## Frozen cases and measurements

- Six final epoch-30 p90 checkpoints: baseline, ours and legacy, each trained
  with endpoint read-noise sigma 3e-4 or 5e-4. These six source runs all used
  A100 GPUs. Selection is by declared training condition and final epoch,
  not accuracy. Earlier checkpoints and unfinished sigma 1e-3 runs are excluded.
- Source training injected betas are baseline 404.141105702,
  ours 5.26875648112, legacy 4.42250110273.
- Replay beta factors relative to each source beta:
  `[1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,.1,.3,1,3,10,30,100]`.
  Report actual injected beta and base beta B/(voltage_amp/current_amp)^3.
- Reuse all 36 prior validation batches of16 (576 examples) in the same order,
  with exact input/index guards. No official-test access or accuracy evaluation.
- Fixed native T=K=8, float64 centered frozen-current EqProp, paired20-output
  squared loss, input gain360, explicit perfect diodes, zero biases and source
  weights unchanged. Use the already accepted operating point and report
  projected-KKT residuals separately; this sweep does not requalify training.
- For every beta, score clean endpoints once and four independent readouts at
  the source checkpoint's training sigma. Reuse the prior four deterministic
  seeds; underlying standard normals match across beta, schemes and noise
  levels, while phases and layers remain independent. Physical relaxation
  stays clean. The BPTT reference differentiates exactly K zero-nudge steps
  from the identical post-T state and must be invariant across beta.
- Report layerwise cosine, norm ratio, relative error, gradient RMS and
  near-zero fraction; clean bias and added-noise magnitude; physical phase
  displacement, midpoint shift and hidden-diode activity changes. Use batch/
  draw medians and 10–90% ranges, not confidence intervals.

Expected production coverage is 78 checkpoint/beta cells,2808 phase replays,
and 56160 layer comparisons (78×36×5×4). Each beta is a separate validated
run bundle. A one-batch smoke covers every checkpoint at minimum, training
and maximum beta before production. Smoke measurements are excluded.

For each checkpoint, report each layer's best sampled noisy cosine and the
single beta maximizing the minimum of the four layer median noisy cosines.
Also report sampled per-layer median-cosine windows above .90 and .99, their
intersection, norm errors and residual qualification. These descriptive
same-cohort maxima do not select a training beta or establish training recovery.
Record any tested-range boundary optimum rather than silently extending it.

Analysis clarification: the first small-beta baseline cells repeat the known
T=8 free-state residual limitation (72 post-T layer/batch failures per cell;
the zero and nudged endpoints pass). Therefore report the best observed
finite-T/K alignment for every checkpoint and its residual flag, alongside
the best fully residual-qualified point when one exists. Do not suppress
baseline's measured curve or call a residual-failing optimum equilibrium-
qualified. The scientific runner and T/K remain unchanged.

## Execution and budget

All measurements use one idle Loulou RTX5090. Local/nom-cool-1/fifi GPUs and
trex/riri GPUs were occupied at placement; Akib was idle but has less memory.
Jean Zay job52725 is unrelated and remains untouched. Recheck Loulou before
smoke and production. Use an isolated workspace beneath
`/home/filip/server_code/results/eqprop-conv3-trained-beta-noise-20260922-v1/`.

Expected duration1–2h; hard scientific budget10800s including smoke. Measure
throughput in smoke. The runner writes progress after each batch. Verify the
worker and first semantic artifact after launch, monitor at least every30min,
and retain failed attempts. Operational retries may preserve the exact science
within the budget; source changes invalidate the matching smoke.

Local authoritative root:
`results/eqprop-conv3-trained-beta-noise-20260922-v1/`.
Copy remote outputs back and validate every bundle and declared coverage before
interpreting the curves. Analysis goes under `analysis/`; record conclusions
in `docs/experimental_manifest.md` without editing `docs/current_state.md`.

Command:
`python -m experiments.replay_conv3_trained_beta_noise --config configs/conv/eqprop_conv3_trained_beta_noise_20260922.json --device cuda`.
Use the same command with `--smoke` for the initial check.

## Operational recovery

The initial Loulou smoke exposed a noise-identity bookkeeping bug: the frozen
analyzer hashes `(z*sigma)/sigma`, whose bytes can vary with sigma because of
roundoff. The fix verifies identical derived RNG seeds/shapes across sigma
and identical noise hashes within sigma, with a targeted regression test.
All 18 replacement smoke cells passed in19.23s.

Before the production admission completed, unrelated Mumax jobs occupied
Loulou. The first post-launch inspection found that contention; only our
worker966023 was terminated (exit143). No completed production cell is used.
The smoke and partial attempt are retained in `loulou-stopped-output.tar.gz`.

Relocate the entire scientific surface to Akib's idle RTX3080, repeat all 18
smoke cells, and use `CUDNN_CONV_WSCAP_DBG=256` and expandable allocator
segments to fit the same float64 batch16 computation. The configurations,
weights, cohort, beta grid, estimator and noise draws stay fixed; dataset path
and target are transport changes. Charge180s for prior attempts against the
unchanged10800s study compute budget. Admission now requires no existing GPU
compute worker, checked immediately before the Python process starts.

## Bounded clean-settling diagnosis

The completed ours/sigma 5e-4 cells fail the clean small-beta control at native
T=K=8, with hidden-layer-3 residual failures. Before attributing this case to
read noise, replay its smallest injected beta (0.000526875648112) without noise
at T=8,32,128,512 and unchanged K=8. Use only the first three original batches
(48 examples), the same checkpoint, float64 implementation and Akib GPU after
the main sweep releases it. Budget600s within the remaining three-hour study
budget; expected duration under two minutes. This is a separate diagnostic
under `settling_diagnostic/`, excluded from the78 main cells. It tests whether
longer free-phase settling helps; it does not establish convergence in K or
replace the fixed-T/K curves. Record outcomes even if settling does not repair
the control. Command: `python -m experiments.diagnose_conv3_clean_settling
--config configs/conv/eqprop_conv3_trained_beta_noise_20260922.json`.

## Completion

All 78 production cells,18 Akib smoke cells and four clean-settling bundles
completed and validate locally. All 1,392 collected output-file hashes match
Akib; all 24 checkpoint source-file hashes,177 runtime-source hashes and four
analyzer hashes are unchanged. The4,320 training-beta comparisons reproduce
the prior RTX3090 replay exactly in cosine, BPTT norm and EqProp norm.
The diagnostic's12 native-T8 comparisons reproduce the main clean controls
exactly. Both launchers exited0 and Akib's compute lane was released.

The total charged runtime is 6,811.13s (1.892GPUh), including180s reserved for
prior attempts,43.42s Akib smoke,6,550.65s main replay and 37.06s settling
diagnosis; this remains below 10,800s. Loulou's27 successful smoke bundles,
failed noise-guard smoke and terminated production bundle are retained and
excluded from the78 scientific cells; their local canonical bundles validate.
No main scientific case is missing or excluded. Measurements, interpretation,
limitations and hashes are in the local `analysis/` directory and
`collection_validation.json`; the finished conclusion is recorded in
`docs/experimental_manifest.md`.
