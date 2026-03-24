# small_network refactor plan

## Goals
- _set_seed, load_json_config, require_config_dict can go to another file and can be shared with diff function (for example mnist_test)
- create_run_dir should be in the io.py file as well
- separate the training part and the evaluation part
- find_latest_model is not needed
- require_keys as well
- no need to use non_linearity_requirements
- no need to use _parse_layer_shapes


## Tasks
- Define target file layout
  - `labs/small_network_core.py`: shared builders (`_set_seed`, dataset setup, `_build_energy_stack`, `train`, `moons_linspace`).
  - `labs/train_small_network.py`: CLI for training only (writes `model.pt`, `model.npz`, `run_metadata.json`).
  - `labs/evaluate_small_network.py`: CLI for linspace evaluation only (requires explicit weights path).
  - `labs/utils/config.py` (or extend `labs/common.py`): config helpers (`load_json_config`, `_require_*`, `_resolve_*`).
  - `labs/utils/run_io.py` (or `labs/io.py`): `create_run_dir`, metadata writing.

- 1) Extract shared utilities
  - Move `_set_seed`, `load_json_config`, `_require_config_dict`, `_require_keys`,
    `_resolve_config_value`, `_resolve_optional_config_value`, `_resolve_config_list`
    into `labs/utils/config.py` (or `labs/common.py`) and update imports.
  - Keep function names stable for now to minimize churn.

- 2) Centralize run IO
  - Move `_create_run_dir` and the metadata writing block into `labs/utils/run_io.py`.
  - Provide helpers: `create_run_dir(...)`, `write_run_metadata(path, data)`.

- 3) Simplify config schema
  - Drop `_parse_layer_shapes` usage: require `input_dim`, `hidden_dims`, `output_dim` in config.
  - Derive `layer_shapes` inside `small_network_core.py` to keep internal code unchanged.
  - Update config error messages to point to the new required keys.

- 4) Simplify non-linearity handling
  - Remove `_NON_LINEARITY_REQUIREMENTS` and `_parse_non_linearity_params` validation.
  - Accept `quadratic_diode_param`, `exponential_diode_param`, `hard_sigmoid_param` as-is.
  - Fail only if `non_linearity` is missing.

- 5) Split training/evaluation
  - `train_small_network.py`: calls `train(...)` only; no linspace step.
  - `evaluate_small_network.py`: calls `moons_linspace(...)` only; requires `--weights`.
  - Remove `find_latest_model`; do not auto-select weights.

- 6) Keep a compatibility wrapper (optional)
  - Keep `small_network.py` as a thin wrapper that calls train + eval,
    or replace it with a clear error pointing users to the new scripts.

- 7) Update documentation and examples
  - Add example configs for moons (input_dim/hidden_dims/output_dim format).
  - Document new CLIs and expected outputs (`run_metadata.json`, `linspace_*.npz`).

- 8) Clean up imports and dead code
  - Remove unused helpers from `small_network.py` once split is complete.
  - Remove unused `require_keys` and `_parse_layer_shapes` callers.

- 9) Linspace loader consistency
  - Update `moons_linspace` to use a standard `DataLoader` (e.g., `LinSpaceDataset.build`)
    instead of `build_mesh`, mirroring the MNIST flow.
  - Add an explicit option (config or CLI) to choose between 1D linspace vs 2D mesh,
    keeping current mesh behavior available if needed.
  - Ensure metadata records the linspace mode and grid shape.

- 10) Enable batching in training
  - Add `batch_size` to the small_network config and CLI.
  - Wire `batch_size` through `train(...)` so `MoonsDataset` uses it.
  - Keep default at 1 for backward compatibility.

- 11) Clean CLI surface
- Require `dims` inside config/metadata (remove `--dims` from CLI).
  - Replace `--output-root` with `--output-dir` (support `output_root` in config for compatibility).
  - Require `--mode {train, linspace}` and remove `--linspace-only` alias.
  - Remove `--weights-root`; require explicit `--weights` for linspace.
  - Keep `--config` as config/metadata source for `voltage_amp`, `current_amp`, `non_linearity`.
  - Keep `--num-iterations`, `--num-epochs`, `--linspace-*` overrides in CLI.
  - Hardcode `num_points` (no CLI/config), but still record it in metadata.

- 12) Training hyperparams in config
  - Add `learning_rate` and `nudging` to the small_network config JSON.
  - Wire them into `train(...)` and `EquilibriumProp` setup (defaults preserved if missing).
  - Switch `learning_rate` to a list (one per weight layer) and validate length.

- 13) Configurable weight gains
  - Add `weight_gains` to the small_network config JSON.
  - Use config `weight_gains` in `_build_energy_stack` (fallback to all-ones).

- 14) Configurable clipping bounds
  - Add `weight_min` and `weight_max` to the small_network config JSON.
  - Use config bounds in `_build_energy_stack` (fallback to current defaults).
  - Record bounds in linspace metadata.

- 15) Record evaluator layer states
  - Swap `Evaluator` for `CustomEvaluator` in `small_network.py`.
  - Enable `record_statistics=("store_states",)` so evaluation captures per-layer states.

- 16) Use CustomTrainer in small_network
  - Swap `Trainer` for `CustomTrainer` in `small_network.py`.


## Smoke checks
- run /home/filip/server_code/labs/train_small_network.py - it returns weights and the metadata
- run /home/filip/server_code/labs/evaluate_small_network.py - it returns states and the metadata
