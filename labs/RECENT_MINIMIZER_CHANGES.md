# Recent Minimizer And Linspace Changes

This note documents the recent behavior changes in `small_network.py` and `custom_minimizer.py`.

## 1) Double-diode runtime presets (config-only switch)

You can now choose the double-diode execution mode with the config key:

- Config key: `"double_diode_runtime": "float32_adaptive"` (or `"float64_fixed"`)

This value is required in the config (no CLI override, no default).

### `float32_adaptive` (default)

- Updater: `Float32ExponentialDoubleDiodeUpdater`
- Equilibrium loop: adaptive convergence in `CustomMinimizer.compute_equilibrium()`
  - Uses global infinity-norm convergence test:
    - stop when `inf_delta <= rtol * inf_ref + vntol`
    - current values: `rtol=1e-4`, `vntol=1e-6`
  - Dynamic polish policy based on previous `inf_delta`:
    - `inf_delta > 1e-2`: no polish, `z_thresh=1e4`
    - `1e-3 < inf_delta <= 1e-2`: polish 4 Newton steps, `z_thresh=1e6`
    - `inf_delta <= 1e-3`: polish 8 Newton steps, `z_thresh=1e8`

### `float64_fixed`

- Updater: `TimedExponentialDOubleDiodeUpdater`
- Equilibrium loop: fixed iterations (no adaptive early stop, no dynamic polish policy)
- Useful when you want to compare against a more "original/fixed" solver behavior and inspect timing output.

## 2) Float32 updater solver detail

`Float32ExponentialDoubleDiodeUpdater` was adjusted so that:

- general updater math stays in `float32`
- Lambert-W evaluation is computed in `float64` internally
- result is cast back to working dtype (`float32`) before continuing

This keeps most of the path lightweight while using higher precision where Lambert-W is evaluated.

## 3) Linspace outputs and metadata

Linspace runs save:

- `linspace_inputs.npz`
- `linspace_states.npz`
- `linspace_residual_currents.npz`
- `linspace_iteration_counts.npz` (when available)
- `run_metadata.json`

`linspace_iteration_counts.npz` includes:

- `iteration_counts` (flat)
- `iteration_grid` (2D grid shaped by `linspace_samples x linspace_samples`)

`run_metadata.json` now also records:

- `double_diode_runtime`
- `double_diode_updater`
- `adaptive_equilibrium`

## 4) Plotting commands

Example for one run directory:

```bash
RUN_DIR="/home/filip/server_code/simulation_results/moons_small_network/hidden_2/<timestamp>_double_diode_exponential_linspace"
PY="/home/filip/miniconda3/envs/py312/bin/python"
```

Residual current plots (all layers):

```bash
$PY /home/filip/server_code/labs/tools/plot_linspace_residual_grid.py \
  --run-dir "$RUN_DIR" \
  --all-layers \
  --output-root "$RUN_DIR/plots"
```

Iteration count grid plot:

```bash
$PY /home/filip/server_code/labs/tools/plot_linspace_iteration_grid.py \
  --iteration-npz "$RUN_DIR/linspace_iteration_counts.npz" \
  --output "$RUN_DIR/plots/iteration_grid.png"
```

## 5) Example run commands

Default adaptive mode:

```bash
python /home/filip/server_code/labs/small_network.py \
  --mode linspace \
  --config /home/filip/server_code/labs/configs/small_network_double_diode_exponential2h.json \
  --weights /path/to/model.pt
```

Fixed float64 timed mode:

```bash
python /home/filip/server_code/labs/small_network.py \
  --mode linspace \
  --config /home/filip/server_code/labs/configs/small_network_double_diode_exponential2h.json \
  --weights /path/to/model.pt

(Ensure `double_diode_runtime` is set to `float64_fixed` in the config.)
```
