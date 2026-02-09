# mnist_tests refactor plan

## Goals
- keep PCA sweep outputs stable while simplifying evaluator flow
- reduce duplicate sweep helpers
- make beta sweep use the same sweep logic

## Tasks
- [ ] Remove build_evaluator usage in PCA sweep path
  - done when: `stats` output can be produced without build_evaluator and remains non-empty for a fixed config
- [ ] Use the same sweep logic for beta
  - done when: beta summary fields (`per_layer_mean`, `per_layer_median`, etc.) are produced for the same inputs
- [x] Remove unnecessary pca_sweep helper functions
  - done when: single sweep and sweep-loop CLI still produce the same files
- [x] Reduce the number of inputs to `prepare_mnist`
  - decide what must be explicit vs. inferred; avoid sprawling defaults
- [x] Stop being overly defensive in data handling (avoid default fallbacks + `isinstance` checks)
  - keep only necessary validation; prefer clear failures
- [x] Apply the same simplification to `prepare_mnist_parts_sweep`
- [x] Re-evaluate if `_build_pca_inference_minimizer` is needed
  - if it stays, justify with a clear reuse benefit
- [ ] Make `_compute_beta_summary` mirror `sweep_pca` style (cleaner control flow)
- [ ] Reconsider `track_training_statistics` placement; may belong elsewhere or in a separate module
- [ ] Simplify CLI wiring: `_build_arg_parser` and `main` are too complex
  - split into smaller helpers or move CLI to dedicated module

## Smoke checks
- run `server_code/labs/tests/smoke_pca_sweep.py` with a small PCA grid
- verify:
  - stats snapshot is non-empty
  - residual currents per layer have non-zero length
  - layer states have first dimension equal to grid size
