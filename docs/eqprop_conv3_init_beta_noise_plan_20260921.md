# Conv3 initialization beta/read-noise replay

User-requested exploratory gradient diagnostic, 2026-09-21. Isolate readout
noise at fixed initial parameters, without the changes in weights caused by
noisy training in the previous final-checkpoint figure.

- Conv3, seed 0, baseline v1/c1, legacy v4/c0.25, ours v4/c1.
- Injected beta B = 0.01 and 0.1 for every scheme. As in the preceding study,
  base beta = B / (voltage_amp/current_amp)^3. Equal B does not imply equal
  physical output displacement; retain measured phase displacement.
- Identical saved initialization from the previous beta calibration; no
  trained checkpoint is loaded. Exact-zero biases, wide [0,100] weights,
  explicit diode dictionaries, input gain360, paired20-output squared loss.
- Reuse the exact36 validation batches of16 from the final-checkpoint study:
  576 examples from ordinary MNIST's deterministic55k/5k split, no test read.
- T=K=8, centered frozen-current EP, float64 arithmetic, same-K BPTT from
  the identical post-T state. Cite the unchanged prior diagnostic operating
  point; report projected-KKT residuals separately from gradient fidelity.
- Readout sigma = 0,1e-5,3e-5,1e-4,3e-4,5e-4,1e-3. The noise range is retained
  from the figure by assumption because the user changed beta/checkpoint/draw
  count but did not request a different noise range.
- One draw per batch and nonzero sigma, seed2026092101. Match standard-normal
  samples across schemes, beta values and noise levels; phases/layers remain
  independent. Perturb copied endpoint voltages only, after relaxation.
- No optimizer, training, accuracy evaluation or official-test access.

Expected coverage:6 cases ×36 batches ×7 noise levels ×4 weight matrices =
6048 comparisons. Save noisy/BPTT, clean/BPTT and noisy/clean cosine, norms,
relative errors, gradient RMS/near-zero fractions, physical state displacement,
residuals, cohort and immutable input/parameter guards. Summaries show median
and10–90% batch range; one draw does not estimate within-batch noise variance.

Target:local nom-cool-2 RTX3090, Python py312/PyTorch2.5.1+cu121. The configured
hosts and Jean Zay were checked: all RTX5090 hosts and Nom-cool-1 are occupied;
local has no compute process. Keep all six cases on this target. Expected
runtime10–15min; cap3600s including smoke. Run one batch of all six cases
through the exact runner before production, and require semantic artifacts.
Observe progress each minute while active; operational recovery may reuse
completed cases only with intact provenance. Do not disturb unrelated jobs.

Result root: `results/eqprop-conv3-init-beta-noise-20260921-v1/`.
Command: `env KMP_DISABLE_SHM=1 KMP_SHM_DISABLE=1 OMP_NUM_THREADS=1
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
/home/filip/miniconda3/envs/py312/bin/python -m
experiments.replay_conv3_init_beta_noise --config
configs/conv/eqprop_conv3_init_beta_noise_20260921.json`.
Add `--smoke` for the same-path operational smoke.

Complete only after all6048 expected rows, input/parameter/cohort matching,
canonical bundle checks and local figures/tables validate. Scientific failures
remain evidence. Record interpretation manually in experimental_manifest.md.
